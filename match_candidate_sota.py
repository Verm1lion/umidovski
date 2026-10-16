# -*- coding: utf-8 -*-
"""
match_candidate_sota.py — CV ↔ İş İlanı eşleştirme: iki aşamalı (two-stage) hibrit sistem.

Bu dosya HEM eğitim notebook'unun (CV_Ilan_SOTA_Egitim_ve_Test_A100.ipynb) HEM de çıkarım
(API/CLI) tarafının kullandığı TEK ortak çekirdektir. Böylece eğitimde hesaplanan özellikler
ile canlıda hesaplananlar birebir aynıdır (train/serve skew yoktur).

Mimari
------
Aşama 1  Aday üretimi (candidate generation)
         * Yoğun (dense) bi-encoder: ince ayarlı BGE-M3 / E5 (MNRL + zor negatifler)
         * Seyrek (sparse) Okapi BM25 (Türkçe F5 kök kırpma)
         * Reciprocal Rank Fusion (RRF, k=60) ile birleştirme → Top-50
Aşama 2  Yeniden sıralama (reranking)
         * Cross-encoder (bge-reranker-v2-m3, LambdaLoss ile ince ayar)
         * LambdaMART (LightGBM lambdarank) — elle çarpan YOK; ağırlıkları veriden öğrenir

Sızıntı kuralları (leakage guards)
----------------------------------
* Özellik üretimi YALNIZCA metinden türetilen bilgileri ve ilanın kendi meta verisini
  (sektör, pozisyon seviyesi, çalışma şekli, şehir) kullanır.
* Adayın altın (gold) etiketleri (Category, collar, ISCO/uzmanlık, kıdem) özellik olarak
  KULLANILMAZ. Sektör / yaka uyumu, yalnızca eğitim kümesinde eğitilmiş metin
  sınıflandırıcılarının olasılık çıktılarıdır (predicted industry / collar probability).
* `assert_no_gold_columns` öznitelik tablosunun SÜTUN ADLARINI denetler (gold_*, legacy_*, grade, label, Category,
  collar, occ_key, isco4, specialty adlı sütun varsa hata verir). Bu bir şema denetimidir: tarafsız bir ada konmuş
  etiket değerini yakalayamaz; asıl güvence özniteliklerin yalnızca metinden ve ilan meta verisinden üretilmesidir.

Kullanım (CLI)
--------------
    python match_candidate_sota.py --bundle export_model_bundle --cv-id 20951
    python match_candidate_sota.py --bundle export_model_bundle --cv-file aday.txt --top-k 10
    python match_candidate_sota.py --bundle export_model_bundle            # etkileşimli mod
    python match_candidate_sota.py --bundle export_model_bundle --serve   # FastAPI (opsiyonel)

Kullanım (Python)
-----------------
    from match_candidate_sota import SOTAMatcher
    m = SOTAMatcher.load("export_model_bundle")
    df = m.match(cv_text, headline="Baklava Ustası", top_k=10)
"""
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

__version__ = "1.0.0"

# =============================================================================
# 1. Sabitler
# =============================================================================
INDUSTRY_TR = {
    "AGRICULTURE": "Tarım ve Hayvancılık",
    "APPAREL_TEXTILE": "Tekstil ve Hazır Giyim",
    "AVIATION": "Havacılık",
    "BEAUTY_CARE": "Güzellik ve Kişisel Bakım",
    "CHEF_HORECA": "Mutfak, Restoran ve Otelcilik",
    "CONSTRUCTION": "İnşaat ve Yapı",
    "CREATIVE_DESIGN": "Yaratıcı Tasarım",
    "CUSTOMER_SERVICE": "Müşteri Hizmetleri",
    "EDUCATION": "Eğitim",
    "ENV_ESG": "Çevre ve Sürdürülebilirlik",
    "FINANCE_ACCOUNTING": "Finans ve Muhasebe",
    "HEALTHCARE": "Sağlık",
    "HR": "İnsan Kaynakları",
    "INDUSTRIAL_MANUFACTURING": "Endüstriyel Üretim ve İmalat",
    "IT_SOFTWARE": "Bilişim ve Yazılım",
    "JANITORIAL_FACILITY": "Temizlik ve Tesis Hizmetleri",
    "LEGAL": "Hukuk",
    "LOGISTICS_DRIVER": "Sürücülük ve Kurye",
    "MARITIME": "Denizcilik",
    "MARKETING_MEDIA": "Pazarlama ve Medya",
    "OFFICE_ADMIN_MGMT": "Ofis, İdari İşler ve Yönetim",
    "SALES": "Satış",
    "SCIENCE_RND": "Bilim ve Ar-Ge",
    "SECURITY_EMERGENCY": "Güvenlik ve Acil Durum",
    "SPORTS": "Spor",
    "SUPPLY_CHAIN_LOGISTICS": "Tedarik Zinciri ve Lojistik",
}
INDUSTRIES = sorted(INDUSTRY_TR)

JOB_LEVEL_MAP = {
    "INTERN": 0, "STUDENT": 0, "TRAINEE": 0,
    "ENTRY": 1, "ENTRY_LEVEL": 1, "NEW_GRADUATE": 1, "ASSISTANT": 1, "JUNIOR": 1,
    "MID": 2, "MID_LEVEL": 2, "SPECIALIST": 2, "ASSISTANT_SPECIALIST": 2,
    "SENIOR": 3, "SENIOR_SPECIALIST": 3,
    "LEAD": 4, "TEAM_LEAD": 4, "MANAGER": 4,
    "DIRECTOR": 5, "EXECUTIVE": 5,
}
SENIORITY_TR = {0: "Stajyer / Öğrenci", 1: "Giriş Seviyesi", 2: "Uzman / Orta Seviye",
                3: "Kıdemli", 4: "Lider / Şef / Müdür", 5: "Direktör / Üst Yönetim"}
MODE_TR = {"OnSite": "İş yerinde", "Hybrid": "Hibrit", "Remote": "Uzaktan"}

# =============================================================================
# 2. Metin işleme
# =============================================================================
_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_ASCII = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")
TOKEN_RE = re.compile(r"[a-z0-9#+]+")
STOPWORDS = set("""
ve ile icin bir bu da de ki mi mu ya veya ama fakat olarak olan olup en daha cok gibi kadar
her tum ayrica ise ne nasil hem sonra once uzerinde arasinda yapmak yapan olmak etmek eden icinde
the and of to in for with on as at by an a is are
konum ozet hakkinda beceriler deneyimler egitim projeler diller ilgi alanlari sertifikalar kurslar
is ilani aciklama firma tarih sirket universite okul pozisyon kidem sektor yetkinlikler deneyim
""".split())


def tr_lower(s) -> str:
    return str(s).translate(_TR_LOWER).lower()


def tr_ascii(s) -> str:
    return tr_lower(s).translate(_ASCII)


def clean_ws(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def tokenize(text, stem: int = 5) -> list:
    """Türkçe için BM25 tokenizasyonu: ASCII katlama + stopword + F5 kök kırpma.

    F5 (ilk 5 karakter) kırpması Türkçe bilgi getiriminde sözlük tabanlı köklemeye yakın
    başarı verir (Can vd., 2008, JASIST 59(3)).
    """
    out = []
    for t in TOKEN_RE.findall(tr_ascii(text)):
        if len(t) < 2 or t in STOPWORDS:
            continue
        out.append(t[:stem] if stem and len(t) > stem else t)
    return out


def norm_city(c) -> str:
    c = re.sub(r"\(.*?\)", "", str(c or ""))
    c = tr_ascii(c).upper().strip()
    return {"NAN": "", "NONE": "", "BELIRTILMEMIS": "", "UNSPECIFIED": ""}.get(c, c)


# =============================================================================
# 3. CV ayrıştırma ve yapılandırılmış metin şablonu
# =============================================================================
# Kanonik başlık -> kabul edilen yazımlar (küçük harf, Türkçe). Veri setindeki biçim + yaygın CV varyantları.
CV_HEADER_ALIASES = {
    "Konum": ["konum", "lokasyon", "adres", "location", "şehir", "sehir"],
    "Özet / Hakkında": ["özet / hakkında", "özet/hakkında", "özet", "ozet", "hakkında", "hakkinda", "hakkımda",
                        "hakkimda", "profil", "profile", "summary", "about", "about me"],
    "Beceriler": ["beceriler", "beceri", "yetenekler", "yetkinlikler", "skills"],
    "Deneyimler": ["deneyimler", "iş deneyimleri", "iş deneyimi", "is deneyimi", "deneyim", "work experience", "experience"],
    "Eğitim": ["eğitim bilgileri", "eğitim", "egitim", "education"],
    "Projeler": ["projeler", "projects"],
    "Sertifikalar": ["sertifikalar", "sertifika", "certifications", "certificates"],
    "Diller": ["yabancı diller", "yabancı dil", "diller", "languages"],
    "Kurslar": ["kurslar", "courses"],
    "İlgi Alanları": ["ilgi alanları", "ilgi alanlari", "hobiler", "interests"],
}
CV_HEADERS = list(CV_HEADER_ALIASES)
_ALIAS2HDR = {a: h for h, al in CV_HEADER_ALIASES.items() for a in al}
_HDR_RE = re.compile(r"(?m)^[ \t]*(" + "|".join(re.escape(a) for a in sorted(_ALIAS2HDR, key=len, reverse=True))
                     + r")[ \t]*:[ \t]*")


def _lower_same_len(t: str) -> str:
    """Türkçe küçük harf; karakter sayısını korur (başlık konumlarını özgün metne eşlemek için)."""
    t2 = t.translate(_TR_LOWER)
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in t2)


def normalize_cv_text(cv_text) -> str:
    """BOM ve satır sonlarını normalleştirir."""
    return str(cv_text or "").replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")


def parse_cv_sections(cv_text: str) -> dict:
    """Serbest CV metnini başlıklarına göre bölümler.

    * Başlıklar büyük/küçük harften bağımsız ve eş anlamlı yazımlarla tanınır (CV_HEADER_ALIASES).
    * İlk başlıktan önceki metin (ör. ad-soyad, unvan satırı, düz özet) atılmaz: özete eklenir.
    * Hiç başlık yoksa tüm metin özet kabul edilir.
    """
    t = normalize_cv_text(cv_text)
    low = _lower_same_len(t)
    parts = {}
    ms = list(_HDR_RE.finditer(low))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(t)
        h = _ALIAS2HDR[m.group(1)]
        body = t[m.end():end].strip()
        parts[h] = (parts[h] + "\n" + body).strip() if h in parts else body
    preamble = t[: ms[0].start()].strip() if ms else t.strip()
    if preamble:
        parts["Özet / Hakkında"] = (preamble + "\n" + parts.get("Özet / Hakkında", "")).strip()
    exps = [clean_ws(x.lstrip("-•* ")) for x in parts.get("Deneyimler", "").split("\n") if x.strip()]
    skills = [clean_ws(s) for s in re.split(r"[,;•\n]", parts.get("Beceriler", "")) if clean_ws(s)]
    return {
        "konum": clean_ws(parts.get("Konum", "")),
        "ozet": clean_ws(parts.get("Özet / Hakkında", "")),
        "beceriler": skills,
        "deneyimler": exps,
        "egitim": clean_ws(parts.get("Eğitim", "")),
        "projeler": clean_ws(parts.get("Projeler", "")),
        "sertifikalar": clean_ws(parts.get("Sertifikalar", "")),
        "diller": clean_ws(parts.get("Diller", "")),
        "kurslar": clean_ws(parts.get("Kurslar", "")),
        "_preamble": preamble,
    }


# Tarih aralıkları: "2019 - 2023", "2018-09 - Günümüz", "Mart 2019 – Ocak 2022", "03/2015 - 05/2024", "2021 - Devam Ediyor"
_MON = r"(?:[A-Za-zÇĞİÖŞÜçğıöşü]{3,}\.?\s+|\d{1,2}[./-]\s*)?"
YEAR_RANGE_RE = re.compile(_MON + r"((?:19|20)\d{2})(?:[./-]\d{1,2}(?!\d))?\s*[-–—]\s*" + _MON +
                           r"((?:19|20)\d{2}|devam\w*|günümüz|gunumuz|halen|present|current|now)", re.I)
YEARS_EXP_RE = re.compile(r"(?<!\d)(\d{1,2})\+?\s*(?:yıl|yil|years?)\s*(?:lık|lik|luk|lük|ı|i)?\s*"
                          r"(?:deneyim|tecrübe|tecrube|experience|sektör|sektor|of\s+(?:\w+\s+){0,2}experience)")
# Veri setinin referans yılı ("devam ediyor" = REF_YEAR). Eğitimde ve çıkarımda AYNI değer kullanılır;
# dışa aktarılan config.json'a yazılır ve SOTAMatcher tarafından geri yüklenir.
REF_YEAR = 2026
SEN_PATTERNS = [
    (0, r"\b(stajyer\w*|staj\b|intern\b|internship|öğrenci\w*|ogrenci\w*|son sınıf|son sinif|çırak\w*|cirak\w*|trainee)"),
    (5, r"\b(direktör\w*|direktor\w*|director|genel müdür\w*|genel mudur\w*|cto\b|ceo\b|cfo\b|coo\b|cio\b|vp\b|vice president|c-level)"),
    (4, r"\b(müdür\w*|mudur\w*|manager|yönetici\w*|yonetici\w*|takım lider\w*|takim lider\w*|team lead\w*|lead\b|lider\w*|şef\b|şefi\b|sef\b|sefi\b|ustabaşı\w*|ustabasi\w*|başusta\w*|basusta\w*|sorumlusu\b|supervisor|koordinatör\w*|koordinator\w*|head of)"),
    (3, r"\b(kıdemli|kidemli|senior|sr\.|başuzman\w*|basuzman\w*|principal|usta\b|ustası\b|ustasi\b)"),
    (1, r"\b(junior|jr\.|yeni mezun|asistan\w*|assistant|yardımcı\w*|yardimci\w*|komi\b|eleman\w*)"),
]


def parse_years_experience(cv_text: str):
    """Deneyim yılı: yalnızca 'Deneyimler' bölümündeki tarih aralıkları + 'X yıllık deneyim' ifadeleri."""
    t = normalize_cv_text(cv_text)
    exp = "\n".join(parse_cv_sections(t)["deneyimler"])
    m = YEARS_EXP_RE.search(tr_lower(t))
    explicit = int(m.group(1)) if m else None
    spans = []
    for a, b in YEAR_RANGE_RE.findall(exp):
        a = int(a)
        b = REF_YEAR if not b[:2].isdigit() else int(b)
        if 1960 <= a <= b <= REF_YEAR + 1:
            spans.append((a, min(b, REF_YEAR)))
    span_years = (max(b for _, b in spans) - min(a for a, _ in spans)) if spans else None
    cands = [x for x in (explicit, span_years) if x is not None]
    return max(cands) if cands else None


def parse_seniority_text(title: str, cv_text: str = "") -> int:
    """0–5 kıdem tahmini (yalnızca metinden): önce unvan sözcükleri, sonra deneyim yılı.

    Unvan hem Türkçe küçük harfle (İ→i, I→ı) hem standart küçük harfle aranır; böylece İngilizce
    'INTERN', 'DIRECTOR' gibi yazımlar da yakalanır.
    """
    t = tr_lower(title) + " || " + str(title or "").lower()
    for lvl, pat in SEN_PATTERNS:
        if re.search(pat, t):
            return lvl
    yrs = parse_years_experience(cv_text)
    if yrs is None:
        if re.search(r"(son sınıf|öğrencisiyim|ogrencisiyim|stajyer pozisyonu|staj imkan\w*|staj fırsat\w*)", tr_lower(cv_text)):
            return 0
        return 2
    return 1 if yrs < 2 else 2 if yrs < 5 else 3 if yrs < 12 else 4


def guess_headline(cv_text: str) -> str:
    """Unvan verilmemişse tahmin: staj dışındaki ilk deneyim unvanı > ilk deneyim > başlık öncesi kısa satır > özet."""
    sec = parse_cv_sections(cv_text)
    titles = [re.split(r"\s*\(", e)[0].strip(" -:") for e in sec["deneyimler"]]
    titles = [x for x in titles if x]
    for x in titles:
        if not re.search(r"(?i)staj|intern", x):
            return x[:80]
    if titles:
        return titles[0][:80]
    for line in sec["_preamble"].split("\n"):
        line = line.strip()
        if 3 <= len(line) <= 80 and ":" not in line:
            return line
    return sec["ozet"][:60]


def build_cv_struct(headline: str, cv_text: str, max_exp: int = 4) -> str:
    """CV için standartlaştırılmış semantik şablon (bi-encoder / BM25 girdisi).

    Pozisyon, Kıdem (metinden), Konum, Yetkinlikler, Özet, Deneyim, Eğitim, Sertifikalar, Diller.
    Altın etiket (sektör/yaka/ISCO) İÇERMEZ.
    """
    s = parse_cv_sections(cv_text)
    headline = clean_ws(headline) or guess_headline(cv_text)
    sen = parse_seniority_text(headline, cv_text)
    yrs = parse_years_experience(cv_text)
    kidem = SENIORITY_TR[sen] + (f" ({yrs} yıl deneyim)" if yrs is not None else "")
    lines = [f"Pozisyon: {headline}", f"Kıdem: {kidem}"]
    if s["konum"]:
        lines.append(f"Konum: {s['konum']}")
    if s["beceriler"]:
        lines.append("Yetkinlikler: " + ", ".join(s["beceriler"]))
    if s["ozet"]:
        lines.append(f"Özet: {s['ozet']}")
    if s["deneyimler"]:
        exps = [re.sub(r"\s+:", ":", re.sub(r"\(\s*\[TARİH\]\s*/\s*\[ŞİRKET\]\s*\)", "", e))
                for e in s["deneyimler"][:max_exp]]
        lines.append("Deneyim: " + " | ".join(clean_ws(e) for e in exps))
    if s["egitim"]:
        lines.append(f"Eğitim: {s['egitim']}")
    if s["sertifikalar"]:
        lines.append(f"Sertifikalar: {s['sertifikalar']}")
    if s["projeler"]:
        lines.append(f"Projeler: {s['projeler']}")
    if s["diller"]:
        lines.append(f"Diller: {s['diller']}")
    return "\n".join(lines)


def build_job_struct(title, company, industry, position_level, mode, city, description, job_type="") -> str:
    """İlan için standartlaştırılmış semantik şablon. Sektör/seviye ilanın KENDİ meta verisidir."""
    lvl = JOB_LEVEL_MAP.get(str(position_level).upper())
    kidem = SENIORITY_TR.get(lvl, "Belirtilmemiş")
    lines = [
        f"Pozisyon: {clean_ws(title)}",
        f"Sektör: {INDUSTRY_TR.get(str(industry), str(industry))}",
        f"Kıdem: {kidem}",
        f"Çalışma Şekli: {MODE_TR.get(str(mode), str(mode))}" + (f" / {job_type}" if job_type else ""),
    ]
    if city and str(city) != "nan":
        lines.append(f"Konum: {city}")
    if company and str(company) != "nan":
        lines.append(f"Firma: {clean_ws(company)}")
    lines.append(f"Açıklama: {clean_ws(description)}")
    return "\n".join(lines)


def compact_cv_for_ce(cv_struct: str, max_chars: int = 1400) -> str:
    """Cross-encoder için CV'nin en bilgilendirici ilk kısmı (Pozisyon/Kıdem/Yetkinlik/Özet/Deneyim)."""
    return cv_struct[:max_chars]


def compact_job_for_ce(job_struct: str, max_chars: int = 1400) -> str:
    return job_struct[:max_chars]


# =============================================================================
# 4. Seyrek getirim: Okapi BM25 (vektörize)
# =============================================================================
class SparseBM25:
    """Okapi BM25 (Robertson & Zaragoza, 2009), scipy.sparse ile vektörize.

    score(q,d) = Σ_t idf(t) · tf(t,d)(k1+1) / (tf(t,d) + k1(1 − b + b·|d|/avgdl))
    idf, rank_bm25.BM25Okapi ile aynıdır: log((N − df + 0.5)/(df + 0.5)), negatifler ε·ort(idf).
    """

    def __init__(self, docs_tokens, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1, self.b = k1, b
        vocab, rows, cols, vals = {}, [], [], []
        for i, toks in enumerate(docs_tokens):
            cnt = {}
            for t in toks:
                cnt[t] = cnt.get(t, 0) + 1
            for t, c in cnt.items():
                rows.append(i)
                cols.append(vocab.setdefault(t, len(vocab)))
                vals.append(c)
        self.vocab = vocab
        n = len(docs_tokens)
        tf = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float64)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if n else 1.0
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log((n - df + 0.5) / (df + 0.5))
        idf = np.where(idf < 0, epsilon * idf.mean(), idf)
        self.idf = idf
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / avgdl)) * idf[tf.col]
        self.W = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def _qmat(self, queries_tokens):
        rows, cols = [], []
        for i, toks in enumerate(queries_tokens):
            for t in toks:
                j = self.vocab.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(queries_tokens), len(self.vocab)))

    def scores(self, queries_tokens) -> np.ndarray:
        return np.asarray((self._qmat(queries_tokens) @ self.W.T).todense(), dtype=np.float32)


# =============================================================================
# 5. Füzyon ve sıralama yardımcıları
# =============================================================================
def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Her satır için 1-tabanlı sıra (1 = en iyi)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1)[None, :]
    return ranks


def rrf_scores(rank_mats, k: int = 60, weights=None) -> np.ndarray:
    """Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009): Σ_i w_i / (k + rank_i)."""
    weights = weights or [1.0] * len(rank_mats)
    return sum(w / (k + r) for w, r in zip(weights, rank_mats))


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    sub = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-sub, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


# =============================================================================
# 6. Değerlendirme metrikleri (graded relevance)
# =============================================================================
def dcg(gains, k):
    g = np.asarray(gains[:k], dtype=float)
    return float(((2.0 ** g - 1.0) / np.log2(np.arange(2, len(g) + 2))).sum())


def ndcg_at_k(ranked_grades, all_grades, k=10):
    """Standart NDCG@k (Järvelin & Kekäläinen, 2002; üstel kazanç 2^rel−1, Burges vd., 2005).
    IDCG, sorgunun TÜM havuzdaki alaka derecelerinden hesaplanır."""
    ideal = np.sort(np.asarray(all_grades, dtype=float))[::-1]
    idcg = dcg(ideal, k)
    return dcg(ranked_grades, k) / idcg if idcg > 0 else 0.0


def evaluate_rankings(ranked: dict, qrels: dict, k: int = 10, rel_threshold: int = 2, recall_ks=(10, 50, 100)) -> dict:
    """ranked: {qid: [doc_id, ...] (sıralı)}, qrels: {qid: {doc_id: grade}} (grade>=1 olanlar).

    Alakalı (binary metrikler için) = grade >= rel_threshold. Hiç alakalı belgesi olmayan
    sorgular TREC geleneğine uygun olarak ortalamaya alınmaz (sayısı raporlanır).
    """
    per = []
    skipped = 0
    for qid, docs in ranked.items():
        q = qrels.get(qid, {})
        n_rel = sum(1 for g in q.values() if g >= rel_threshold)
        if n_rel == 0:
            skipped += 1
            continue
        grades = [q.get(d, 0) for d in docs]
        relb = [g >= rel_threshold for g in grades]
        first = next((i for i, r in enumerate(relb[:k]) if r), None)
        row = {
            "qid": qid,
            f"NDCG@{k}": ndcg_at_k(grades, list(q.values()), k),
            f"MRR@{k}": 0.0 if first is None else 1.0 / (first + 1),
            f"P@{k}": sum(relb[:k]) / k,
            f"P@{k}_capped": sum(relb[:k]) / min(k, n_rel),
            "Top1": float(relb[0]) if relb else 0.0,
            "Top1_exact(grade3)": float(grades[0] == 3) if grades else 0.0,
            "n_rel": n_rel,
        }
        for rk in recall_ks:
            row[f"R@{rk}"] = sum(relb[:rk]) / n_rel
        per.append(row)
    df = pd.DataFrame(per)
    summary = {c: float(df[c].mean()) for c in df.columns if c not in ("qid", "n_rel")} if len(df) else {}
    summary["n_queries"] = int(len(df))
    summary["n_skipped_no_relevant"] = int(skipped)
    return {"summary": summary, "per_query": df}


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_randomization_test(a, b, n_perm=10000, seed=0):
    """İki sistemin sorgu bazlı skorları için iki yönlü Fisher randomizasyon testi
    (Smucker, Allan & Carterette, CIKM 2007)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm >= obs) + 1) / (n_perm + 1))


# =============================================================================
# 7. Öznitelik (attribute) sınıflandırıcıları — YALNIZCA train bölümünde eğitilir
# =============================================================================
class AttributeModels:
    """Metinden sektör ve yaka olasılığı tahmin eden hafif sınıflandırıcılar.

    * Sektör: TF-IDF (kelime 1-2gram) + çok sınıflı lojistik regresyon, CV yapılandırılmış metni.
    * Yaka: karakter 2-5gram TF-IDF + lojistik regresyon, UNVAN metni (CV başlığı ↔ ilan başlığı
      aynı dağılımdadır; böylece aynı model ilan başlığına da uygulanabilir).
    LTR'deki 'sektör uyumu' özelliği = P_cv(sektör = ilanın sektörü); altın Category kullanılmaz.
    """

    def __init__(self):
        self.ind_vec = self.ind_clf = self.col_vec = self.col_clf = None
        self.ind_classes = []

    def fit(self, cv_struct_texts, industries, cv_titles, collars_blue):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.ind_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True,
                                       preprocessor=tr_ascii)
        X = self.ind_vec.fit_transform(cv_struct_texts)
        self.ind_clf = LogisticRegression(max_iter=2000, C=8.0)
        self.ind_clf.fit(X, list(industries))
        self.ind_classes = list(self.ind_clf.classes_)
        self.col_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True,
                                       preprocessor=tr_ascii)
        Xc = self.col_vec.fit_transform(cv_titles)
        self.col_clf = LogisticRegression(max_iter=2000, C=4.0)
        self.col_clf.fit(Xc, np.asarray(collars_blue, dtype=int))
        return self

    def industry_proba(self, cv_struct_texts) -> np.ndarray:
        """(n, len(INDUSTRIES)) olasılık matrisi; sütun sırası INDUSTRIES."""
        P = self.ind_clf.predict_proba(self.ind_vec.transform(cv_struct_texts))
        out = np.zeros((P.shape[0], len(INDUSTRIES)), dtype=np.float32)
        for j, c in enumerate(self.ind_classes):
            if c in INDUSTRIES:
                out[:, INDUSTRIES.index(c)] = P[:, j]
        return out

    def blue_proba(self, titles) -> np.ndarray:
        return self.col_clf.predict_proba(self.col_vec.transform(list(titles)))[:, 1].astype(np.float32)


class TitleSimilarity:
    """Unvanlar arası karakter n-gram TF-IDF kosinüsü (alt-kelime tuzaklarını yakalamak için)."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True,
                                   preprocessor=tr_ascii)

    def fit(self, titles):
        self.vec.fit(list(titles))
        return self

    def transform(self, titles):
        return self.vec.transform(list(titles))


# =============================================================================
# 8. Çift (aday, ilan) öznitelikleri — LambdaMART girdisi
# =============================================================================
FEATURE_COLUMNS = [
    "dense_cos", "dense_rank_log", "dense_z", "dense_gap_top",
    "bm25", "bm25_rank_log", "bm25_z",
    "rrf", "rrf_rank_log",
    "ce_score", "ce_rank_log", "ce_z", "ce_gap_top",
    "ind_prob_job_sector", "ind_pred_top1_match", "ind_prob_cv_max",
    "cv_p_blue", "job_p_blue", "collar_agree",
    "cv_seniority", "job_seniority", "job_seniority_missing", "sen_diff", "sen_absdiff",
    "same_city", "job_remote", "job_hybrid", "loc_ok",
    "skill_overlap", "skill_jaccard", "title_sim", "title_tok_overlap",
    "job_desc_len_log", "cv_years_exp", "cv_years_missing",
]
GOLD_PREFIX = "gold_"


def assert_no_gold_columns(df: pd.DataFrame, where: str = ""):
    bad = [c for c in df.columns if str(c).startswith((GOLD_PREFIX, "legacy_")) or
           c in ("grade", "label", "Category", "collar", "occ_key", "isco4", "specialty")]
    if bad:
        raise AssertionError(f"[SIZINTI] Özellik tablosunda altın etiket sütunları var ({where}): {bad}")


@dataclass
class QueryMeta:
    """Bir aday (sorgu) için YALNIZCA metinden türetilen bilgiler."""
    title: str
    city: str
    seniority: int
    years: float
    skill_tokens: list
    title_tokens: set
    ind_proba: np.ndarray
    p_blue: float

    @staticmethod
    def from_text(headline, cv_text, attr: AttributeModels, cv_struct: str | None = None):
        headline = clean_ws(headline) or guess_headline(cv_text)
        sec = parse_cv_sections(cv_text)
        cv_struct = cv_struct or build_cv_struct(headline, cv_text)
        yrs = parse_years_experience(cv_text)
        return QueryMeta(
            title=headline,
            city=norm_city(sec["konum"]),
            seniority=parse_seniority_text(headline, cv_text),
            years=float(yrs) if yrs is not None else np.nan,
            skill_tokens=[set(tokenize(s)) for s in sec["beceriler"] if tokenize(s)],
            title_tokens=set(tokenize(headline)),
            ind_proba=attr.industry_proba([cv_struct])[0],
            p_blue=float(attr.blue_proba([headline])[0]),
        )


@dataclass
class JobTable:
    """İlan tarafı önbelleği (bir kez hesaplanır)."""
    titles: list
    industry_idx: np.ndarray
    seniority: np.ndarray
    remote: np.ndarray
    hybrid: np.ndarray
    city: np.ndarray
    p_blue: np.ndarray
    token_sets: list
    title_tokens: list
    desc_len_log: np.ndarray
    title_mat: object = None
    extra: dict = field(default_factory=dict)

    @staticmethod
    def build(jobs: pd.DataFrame, attr: AttributeModels, title_sim: TitleSimilarity):
        titles = jobs["title"].astype(str).tolist()
        ind = [INDUSTRIES.index(i) if i in INDUSTRIES else -1 for i in jobs["industry"].astype(str)]
        sen = jobs["position_level"].astype(str).str.upper().map(JOB_LEVEL_MAP).astype(float).to_numpy()
        mode = jobs["mode_of_operation"].astype(str).to_numpy(dtype=object)
        return JobTable(
            titles=titles,
            industry_idx=np.asarray(ind),
            seniority=sen,
            remote=(mode == "Remote"),
            hybrid=(mode == "Hybrid"),
            city=np.asarray([norm_city(c) for c in jobs["city"].astype(str)], dtype=object),
            p_blue=attr.blue_proba(titles),
            token_sets=[set(tokenize(f"{t} {d}")) for t, d in zip(titles, jobs["description"].astype(str))],
            title_tokens=[set(tokenize(t)) for t in titles],
            desc_len_log=np.log1p(jobs["description"].astype(str).str.len().to_numpy(dtype=float)),
            title_mat=title_sim.transform(titles),
        )


def _z(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def build_pair_features(q: QueryMeta, cand_idx, jt: JobTable, title_sim: TitleSimilarity,
                        dense, bm25, rrf, dense_rank, bm25_rank, rrf_rank, ce=None) -> pd.DataFrame:
    """Tek sorgunun aday listesi için öznitelik matrisi.

    dense/bm25/rrf/ce: aday listesindeki skorlar (len = len(cand_idx));
    *_rank: tüm korpustaki 1-tabanlı sıra.
    """
    cand_idx = np.asarray(cand_idx)
    n = len(cand_idx)
    ce = np.zeros(n) if ce is None else np.asarray(ce, dtype=float)
    ce_rank = np.empty(n)
    ce_rank[np.argsort(-ce, kind="stable")] = np.arange(1, n + 1)
    ind_idx = jt.industry_idx[cand_idx]
    ind_prob = np.where(ind_idx >= 0, q.ind_proba[np.clip(ind_idx, 0, None)], 0.0)
    jsen = jt.seniority[cand_idx]
    jsen_missing = np.isnan(jsen)
    jsen_f = np.where(jsen_missing, 2.0, jsen)
    same_city = np.asarray([bool(q.city) and q.city == c for c in jt.city[cand_idx]])
    remote = jt.remote[cand_idx]
    qt = title_sim.transform([q.title])
    tsim = np.asarray((jt.title_mat[cand_idx] @ qt.T).todense()).ravel()
    skill_ov, skill_jac, title_ov = [], [], []
    q_skill_all = set().union(*q.skill_tokens) if q.skill_tokens else set()
    for j in cand_idx:
        toks = jt.token_sets[j]
        if q.skill_tokens:
            skill_ov.append(sum(1 for s in q.skill_tokens if s <= toks) / len(q.skill_tokens))
            skill_jac.append(len(q_skill_all & toks) / max(1, len(q_skill_all | toks)))
        else:
            skill_ov.append(0.0)
            skill_jac.append(0.0)
        tt = jt.title_tokens[j]
        title_ov.append(len(q.title_tokens & tt) / max(1, len(q.title_tokens | tt)))
    dense = np.asarray(dense, dtype=float)
    f = pd.DataFrame({
        "dense_cos": dense,
        "dense_rank_log": np.log1p(np.asarray(dense_rank, dtype=float)),
        "dense_z": _z(dense),
        "dense_gap_top": dense - dense.max(),
        "bm25": np.asarray(bm25, dtype=float),
        "bm25_rank_log": np.log1p(np.asarray(bm25_rank, dtype=float)),
        "bm25_z": _z(bm25),
        "rrf": np.asarray(rrf, dtype=float),
        "rrf_rank_log": np.log1p(np.asarray(rrf_rank, dtype=float)),
        "ce_score": ce,
        "ce_rank_log": np.log1p(ce_rank),
        "ce_z": _z(ce),
        "ce_gap_top": ce - ce.max(),
        "ind_prob_job_sector": ind_prob,
        "ind_pred_top1_match": (ind_idx == int(np.argmax(q.ind_proba))).astype(float),
        "ind_prob_cv_max": np.full(n, float(q.ind_proba.max())),
        "cv_p_blue": np.full(n, q.p_blue),
        "job_p_blue": jt.p_blue[cand_idx],
        "collar_agree": 1.0 - np.abs(q.p_blue - jt.p_blue[cand_idx]),
        "cv_seniority": np.full(n, float(q.seniority)),
        "job_seniority": jsen_f,
        "job_seniority_missing": jsen_missing.astype(float),
        "sen_diff": jsen_f - q.seniority,
        "sen_absdiff": np.abs(jsen_f - q.seniority),
        "same_city": same_city.astype(float),
        "job_remote": remote.astype(float),
        "job_hybrid": jt.hybrid[cand_idx].astype(float),
        "loc_ok": (remote | same_city).astype(float),
        "skill_overlap": skill_ov,
        "skill_jaccard": skill_jac,
        "title_sim": tsim,
        "title_tok_overlap": title_ov,
        "job_desc_len_log": jt.desc_len_log[cand_idx],
        "cv_years_exp": np.full(n, 0.0 if np.isnan(q.years) else q.years),
        "cv_years_missing": np.full(n, float(np.isnan(q.years))),
    })
    return f[FEATURE_COLUMNS]


# =============================================================================
# 9. Çıkarım: SOTAMatcher
# =============================================================================
def ce_predict(ce_model, pairs, batch_size=64):
    """Cross-encoder ham logit skorları (Sigmoid YOK; LambdaLoss logit üzerinde eğitildi).

    Eğitim notebook'u ve çıkarım AYNI fonksiyonu kullanır (fp32 / TF32, autocast yok) -> öznitelik paritesi.
    """
    import torch
    s = ce_model.predict(list(pairs), batch_size=batch_size, activation_fn=torch.nn.Identity(),
                         show_progress_bar=len(pairs) > 5000, convert_to_numpy=True)
    return np.asarray(s, dtype=np.float32).ravel()


class SOTAMatcher:
    """Eğitilmiş paket (export_model_bundle) ile uçtan uca eşleştirme.

    Paket içeriği:
      config.json, jobs.parquet, bi_encoder/, cross_encoder/, lambdamart.txt,
      attribute_models.joblib, title_sim.joblib, job_embeddings.npy, faiss.index (ops.; çıkarımda sorgulanmaz)
    """

    def __init__(self, bundle_dir: str, device: str | None = None):
        import joblib
        self.dir = bundle_dir
        with open(os.path.join(bundle_dir, "config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        global REF_YEAR
        REF_YEAR = int(self.cfg.get("ref_year", REF_YEAR))          # eğitimdeki referans yılıyla aynı
        if self.cfg.get("feature_columns") and list(self.cfg["feature_columns"]) != FEATURE_COLUMNS:
            raise ValueError("Paketteki öznitelik şeması bu çekirdek sürümüyle uyuşmuyor (feature_columns). "
                             "Paketle birlikte gelen match_candidate_sota.py dosyasını kullanın.")
        self.jobs = pd.read_parquet(os.path.join(bundle_dir, "jobs.parquet"))
        self.attr = joblib.load(os.path.join(bundle_dir, "attribute_models.joblib"))
        self.title_sim = joblib.load(os.path.join(bundle_dir, "title_sim.joblib"))
        self.jt = JobTable.build(self.jobs, self.attr, self.title_sim)
        self.bm25 = SparseBM25([tokenize(t) for t in self.jobs["job_struct"]])
        import torch
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.cross_encoder import CrossEncoder
        # Ağırlıklar paket boyutu için fp16 kaydedilmiş olabilir; çıkarım, eğitimdeki özniteliklerle aynı
        # hassasiyette (fp32) yapılır.
        mk = {"dtype": torch.float32}
        self.bi = SentenceTransformer(os.path.join(bundle_dir, "bi_encoder"), device=device, model_kwargs=mk)
        self.bi.max_seq_length = int(self.cfg.get("max_seq_len", self.bi.max_seq_length or 512))
        self.ce = None
        if os.path.isdir(os.path.join(bundle_dir, "cross_encoder")):
            self.ce = CrossEncoder(os.path.join(bundle_dir, "cross_encoder"), device=device, model_kwargs=mk,
                                   max_length=int(self.cfg.get("ce_max_len", 512)))
        else:
            print("UYARI: cross_encoder bulunamadı; CE öznitelikleri 0 olacak, sıralama kalitesi düşer.")
        emb_p = os.path.join(bundle_dir, "job_embeddings.npy")
        self.job_emb = np.load(emb_p) if os.path.exists(emb_p) else self.bi.encode(
            [self.cfg.get("doc_prefix", "") + t for t in self.jobs["job_struct"]], normalize_embeddings=True,
            batch_size=64, show_progress_bar=False)
        # Yoğun puanlama tüm ilanlar üzerinde kesin iç çarpımdır (IndexFlatIP ile eşdeğer): RRF ve LambdaMART'ın sıra
        # öznitelikleri tam-korpus sıralarını gerektirir. Paketteki faiss.index yalnız büyük havuzlar için bırakılmıştır.
        self.job_emb = np.asarray(self.job_emb, dtype=np.float32)
        self.ltr = None
        lp = os.path.join(bundle_dir, "lambdamart.txt")
        if os.path.exists(lp):
            import lightgbm as lgb
            self.ltr = lgb.Booster(model_file=lp)

    @classmethod
    def load(cls, bundle_dir: str, device: str | None = None):
        return cls(bundle_dir, device)

    def match(self, cv_text: str, headline: str | None = None, top_k: int = 10, explain: bool = True) -> pd.DataFrame:
        cfg = self.cfg
        n_jobs = len(self.jobs)
        top_k = int(max(1, min(int(top_k), min(cfg.get("n_candidates", 50), n_jobs))))
        cv_text = normalize_cv_text(cv_text)
        if not cv_text.strip():
            raise ValueError("Boş CV metni")
        cv_struct = build_cv_struct(headline or "", cv_text)
        q = QueryMeta.from_text(headline or "", cv_text, self.attr, cv_struct)
        qe = np.asarray(self.bi.encode([cfg.get("query_prefix", "") + cv_struct], normalize_embeddings=True,
                                       show_progress_bar=False), dtype=np.float32)
        dense = (self.job_emb @ qe[0]).astype(np.float32)[None, :]
        bm = self.bm25.scores([tokenize(cv_struct)])
        d_rank, b_rank = ranks_from_scores(dense), ranks_from_scores(bm)
        fused = rrf_scores([d_rank, b_rank], k=cfg.get("rrf_k", 60))
        r_rank = ranks_from_scores(fused)
        n_cand = min(cfg.get("n_candidates", 50), n_jobs)
        cand = topk_indices(fused, n_cand)[0]
        ce = np.zeros(len(cand))
        if self.ce is not None:
            pairs = [(compact_cv_for_ce(cv_struct), compact_job_for_ce(self.jobs["job_struct"].iloc[j])) for j in cand]
            ce = ce_predict(self.ce, pairs, batch_size=32)
        feats = build_pair_features(q, cand, self.jt, self.title_sim, dense[0, cand], bm[0, cand], fused[0, cand],
                                    d_rank[0, cand], b_rank[0, cand], r_rank[0, cand], ce)
        assert_no_gold_columns(feats, "inference")
        if self.ltr is not None:
            score = self.ltr.predict(feats[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        else:
            score = ce if self.ce is not None else fused[0, cand]
        order = np.argsort(-score, kind="stable")[:top_k]
        rows = []
        contrib = None
        if explain and self.ltr is not None:
            contrib = self.ltr.predict(feats[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[order], pred_contrib=True)
        for r, o in enumerate(order):
            j = int(cand[o])
            job = self.jobs.iloc[j]
            row = {
                "rank": r + 1, "job_key": job.get("job_key", j), "title": job["title"],
                "company": job.get("company_name", ""), "industry": job["industry"],
                "position_level": job["position_level"], "mode": job["mode_of_operation"], "city": job.get("city", ""),
                "score": float(score[o]), "dense_cos": float(dense[0, j]), "bm25": float(bm[0, j]),
                "ce_score": float(ce[o]), "ind_prob": float(feats["ind_prob_job_sector"].iloc[o]),
                "collar_agree": float(feats["collar_agree"].iloc[o]), "sen_diff": float(feats["sen_diff"].iloc[o]),
                "loc_ok": float(feats["loc_ok"].iloc[o]),
            }
            if contrib is not None:
                c = contrib[r][:-1]
                top = np.argsort(-np.abs(c))[:3]
                row["top_factors"] = "; ".join(f"{FEATURE_COLUMNS[t]}={c[t]:+.2f}" for t in top)
            rows.append(row)
        return pd.DataFrame(rows)


# =============================================================================
# 10. CLI / API
# =============================================================================
def _print_result(df: pd.DataFrame):
    print("=" * 100)
    for r in df.itertuples():
        print(f"{r.rank:2d}. [{r.score:+.4f}] {r.title}  | {r.company} | {r.industry} | {r.position_level} | "
              f"{r.mode} | {r.city}")
        extra = f"    dense={r.dense_cos:.3f} bm25={r.bm25:.2f} ce={r.ce_score:+.3f} P(sektör)={r.ind_prob:.2f} " \
                f"yaka_uyum={r.collar_agree:.2f} kıdem_farkı={r.sen_diff:+.0f} lokasyon_ok={int(r.loc_ok)}"
        print(extra)
        if hasattr(r, "top_factors"):
            print(f"    en etkili faktörler: {r.top_factors}")
    print("=" * 100)


def build_app(matcher: "SOTAMatcher"):
    """FastAPI uygulaması: POST /match {cv_text, headline?, top_k?}; GET /health."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    class MatchRequest(BaseModel):
        cv_text: str = Field(..., min_length=20, description="Adayın CV metni (serbest metin)")
        headline: str | None = Field(None, description="Adayın unvanı (opsiyonel)")
        top_k: int = Field(10, ge=1, le=50)

    app = FastAPI(title="CV-İlan SOTA Eşleştirme", version=__version__)

    def match(req: MatchRequest):
        try:
            return matcher.match(req.cv_text, req.headline, req.top_k).to_dict(orient="records")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    match.__annotations__ = {"req": MatchRequest}       # yerel sınıfı FastAPI'ye açıkça bağla
    app.post("/match")(match)
    app.get("/health")(lambda: {"status": "ok", "n_jobs": len(matcher.jobs), "version": __version__})
    return app


def _serve(matcher: "SOTAMatcher", host: str, port: int):
    try:
        import uvicorn
        app = build_app(matcher)
    except ImportError:
        print("FastAPI modu için: pip install fastapi uvicorn")
        return 1
    uvicorn.run(app, host=host, port=port)
    return 0


def _read_multiline(prompt_txt: str) -> str:
    """Etkileşimli modda çok satırlı CV okur: boş satır veya EOF ile biter."""
    print(prompt_txt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="CV ↔ İlan SOTA eşleştirme (Hybrid RRF + Cross-Encoder + LambdaMART)")
    ap.add_argument("--bundle", default="export_model_bundle", help="Eğitilmiş model paketi klasörü")
    ap.add_argument("--cvs", default=None, help="clean_cvs.parquet (ID ile sorgu için); varsayılan paket içi")
    ap.add_argument("--cv-id", type=str, default=None)
    ap.add_argument("--cv-file", type=str, default=None, help="Düz metin CV dosyası")
    ap.add_argument("--headline", type=str, default=None, help="Adayın unvanı (opsiyonel)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="Sonucu JSON olarak yaz")
    ap.add_argument("--serve", action="store_true", help="FastAPI sunucusu başlat")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    m = SOTAMatcher.load(a.bundle)
    if a.serve:
        return _serve(m, a.host, a.port)
    cvs_path = a.cvs or os.path.join(a.bundle, "clean_cvs.parquet")
    cvs = pd.read_parquet(cvs_path) if os.path.exists(cvs_path) else None

    def run_one(text, headline):
        df = m.match(text, headline, a.top_k)
        if a.json:
            print(df.to_json(orient="records", force_ascii=False, indent=1))
        else:
            _print_result(df)

    if a.cv_file:
        with open(a.cv_file, encoding="utf-8-sig") as f:
            return run_one(f.read(), a.headline)
    if a.cv_id is not None:
        if cvs is None:
            print("clean_cvs.parquet bulunamadı; --cvs ile yol verin.")
            return 1
        row = cvs[cvs["cv_id"].astype(str) == str(a.cv_id)]
        if row.empty:
            print(f"Aday bulunamadı: {a.cv_id}")
            return 1
        row = row.iloc[0]
        print(f"Aday {row['cv_id']}: {row['headline']}")
        return run_one(row["cv_text"], row["headline"])
    while True:
        try:
            s = input("\nAday ID (çıkış: q) veya çok satırlı CV girmek için 'metin': ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if s.lower() in ("q", "0", "exit", "quit", "çıkış", "cikis"):
            break
        if s.lower().startswith("metin"):
            run_one(_read_multiline("CV metnini yapıştırın; bitirmek için boş satır bırakın:"), None)
            continue
        if cvs is not None:
            row = cvs[cvs["cv_id"].astype(str) == s]
            if not row.empty:
                row = row.iloc[0]
                print(f"Aday {row['cv_id']}: {row['headline']}")
                run_one(row["cv_text"], row["headline"])
                continue
        print("Bulunamadı.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
