# -*- coding: utf-8 -*-
"""Veri hazirlama scriptlerinin ortak yardimcilari.

* Turkce'ye duyarli kucuk harf donusumu ve tokenizasyon (F5 kok kirpma: Can vd., 2008)
* Seyrek (scipy.sparse) Okapi BM25 (Robertson & Zaragoza, 2009)
* Veri yukleyiciler (CV, ilan)
* CV metninden kidem / sehir / beceri ayristirici (YALNIZCA metinden; etiket kullanmaz)
"""
import json
import math
import os
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work")

_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_ASCII = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")

STOPWORDS = set("""
ve ile icin için bir bu da de ki mi mu mı mü ya veya ama fakat olarak olan olup en daha cok çok gibi kadar
her tum tüm ayrica ayrıca ise ne nasil nasıl hem sonra once önce uzerinde üzerinde arasinda arasında
yapmak yapan olan olmak etmek eden icinde içinde ile, the and of to in for with on as at by an a is are
konum ozet özet hakkinda hakkında beceriler deneyimler egitim eğitim projeler diller ilgi alanlari alanları
sertifikalar is iş ilani ilanı aciklama açıklama firma tarih şirket sirket universite üniversite okul
""".split())


def tr_lower(s):
    return str(s).translate(_TR_LOWER).lower()


def tr_ascii(s):
    return tr_lower(s).translate(_ASCII)


TOKEN_RE = re.compile(r"[a-z0-9#+]+")


def tokenize(text, stem=5):
    """Turkce metin -> token listesi. ASCII'ye katlama + stopword + F5 kok kirpma."""
    toks = TOKEN_RE.findall(tr_ascii(text))
    out = []
    for t in toks:
        if t in STOPWORDS or len(t) < 2:
            continue
        out.append(t[:stem] if stem and len(t) > stem else t)
    return out


class SparseBM25:
    """Okapi BM25, scipy.sparse ile vektorize. score(q, d) = sum_t idf(t) * tf*(k1+1)/(tf + k1*(1-b+b*dl/avgdl)).

    idf, rank_bm25.BM25Okapi ile ayni formul: log((N - df + 0.5) / (df + 0.5)), negatif idf'ler
    epsilon * ortalama_idf ile tabanlanir.
    """

    def __init__(self, docs_tokens, k1=1.5, b=0.75, epsilon=0.25):
        self.k1, self.b = k1, b
        vocab = {}
        rows, cols, vals = [], [], []
        for i, toks in enumerate(docs_tokens):
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                j = vocab.setdefault(t, len(vocab))
                rows.append(i)
                cols.append(j)
                vals.append(c)
        self.vocab = vocab
        n_docs = len(docs_tokens)
        tf = sp.csr_matrix((vals, (rows, cols)), shape=(n_docs, len(vocab)), dtype=np.float64)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if n_docs else 1.0
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log((n_docs - df + 0.5) / (df + 0.5))
        eps = epsilon * idf.mean()
        idf = np.where(idf < 0, eps, idf)
        self.idf = idf
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avgdl)
        w = tf.data * (k1 + 1) / denom * idf[tf.col]
        self.W = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)  # doc x vocab

    def query_matrix(self, queries_tokens):
        rows, cols = [], []
        for i, toks in enumerate(queries_tokens):
            for t in toks:  # rank_bm25 gibi tekrar eden sorgu terimleri tekrar sayilir
                j = self.vocab.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        data = np.ones(len(rows))
        return sp.csr_matrix((data, (rows, cols)), shape=(len(queries_tokens), len(self.vocab)))

    def scores(self, queries_tokens):
        return np.asarray((self.query_matrix(queries_tokens) @ self.W.T).todense())


def load_cvs():
    return pd.read_csv(os.path.join(ROOT, "veriseti", "tum_cvler.csv"), encoding="utf-8-sig")


def load_jobs():
    p = os.path.join(WORK, "tum_ilanlar.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError("Once scripts/01_parse_jobs_sql.py calistirin")
    df = pd.read_parquet(p)
    df.insert(0, "job_key", [f"J{i:04d}" for i in range(len(df))])
    return df


# ---------------------------------------------------------------------------
# Metinden kidem / sehir / beceri ayristirma (etiket KULLANMAZ; ozellik tarafi)
# ---------------------------------------------------------------------------
YEAR_RANGE_RE = re.compile(r"((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2}|devam|günümüz|gunumuz|halen|present|current|now)", re.I)
YEARS_EXP_RE = re.compile(r"(\d{1,2})\+?\s*(?:yıl|yil|years?)\s*(?:lık|lik|luk|lük|ı|i)?\s*(?:deneyim|tecrübe|tecrube|experience|sektör|sektor|of)", re.I)
REF_YEAR = 2025

# Turkce ekler icin kelime sonu serbest (\w*): "stajyeri", "mudurlugu", "ustasi" ...
SEN_PATTERNS = [
    (0, r"\b(stajyer\w*|staj\b|intern\b|internship|öğrenci\w*|ogrenci\w*|son sınıf|son sinif|çırak\w*|cirak\w*|trainee)"),
    (5, r"\b(direktör\w*|direktor\w*|director|genel müdür\w*|genel mudur\w*|cto\b|ceo\b|cfo\b|coo\b|cio\b|vp\b|vice president|c-level)"),
    (4, r"\b(müdür\w*|mudur\w*|manager|yönetici\w*|yonetici\w*|takım lider\w*|takim lider\w*|team lead\w*|lead\b|lider\w*|şef\b|şefi\b|sef\b|sefi\b|ustabaşı\w*|ustabasi\w*|başusta\w*|basusta\w*|sorumlusu\b|supervisor|koordinatör\w*|koordinator\w*|head of)"),
    (3, r"\b(kıdemli|kidemli|senior|sr\.|başuzman\w*|basuzman\w*|principal|usta\b|ustası\b|ustasi\b)"),
    (1, r"\b(junior|jr\.|yeni mezun|asistan\w*|assistant|yardımcı\w*|yardimci\w*|komi\b|eleman\w*)"),
]

SECTION_RE = re.compile(r"Deneyimler:(.*?)(?:\n\s*\n(?:Eğitim|Egitim|Projeler|Sertifikalar|Diller|İlgi|Ilgi)\b|\Z)", re.S)


def parse_years_experience(text):
    """CV metninden toplam deneyim yili tahmini.

    Yalnizca 'Deneyimler:' bolumundeki tarih araliklari (egitim yillari karismasin diye) ve
    'X yillik deneyim' ifadeleri kullanilir.
    """
    t = str(text)
    sec = SECTION_RE.search(t)
    exp_text = sec.group(1) if sec else ""
    m = YEARS_EXP_RE.search(t)
    explicit = int(m.group(1)) if m else None
    spans = []
    for a, b in YEAR_RANGE_RE.findall(exp_text):
        a = int(a)
        b = REF_YEAR if not b[:2].isdigit() else int(b)
        if 1960 <= a <= b <= REF_YEAR + 1:
            spans.append((a, b))
    span_years = None
    if spans:
        span_years = max(b for _, b in spans) - min(a for a, _ in spans)
    cands = [x for x in (explicit, span_years) if x is not None]
    return max(cands) if cands else None


def parse_seniority_text(title, text=""):
    """0-5 olcekli kidem tahmini: once unvan anahtar kelimeleri, sonra deneyim yili."""
    t = tr_lower(title)
    for lvl, pat in SEN_PATTERNS:
        if re.search(pat, t):
            return lvl
    yrs = parse_years_experience(text)
    if yrs is None:
        if re.search(r"(son sınıf|öğrencisiyim|ogrencisiyim|stajyer pozisyonu|staj imkan\w*|staj fırsat\w*)", tr_lower(text)):
            return 0
        return 2
    if yrs < 2:
        return 1
    if yrs < 5:
        return 2
    if yrs < 12:
        return 3
    return 4


def parse_cv_city(text):
    m = re.search(r"Konum:\s*([^\n\r]+)", str(text))
    return m.group(1).strip() if m else ""


JOB_LEVEL_MAP = {
    "INTERN": 0, "STUDENT": 0, "TRAINEE": 0,
    "ENTRY": 1, "ENTRY_LEVEL": 1, "NEW_GRADUATE": 1, "ASSISTANT": 1, "JUNIOR": 1,
    "MID": 2, "MID_LEVEL": 2, "SPECIALIST": 2, "ASSISTANT_SPECIALIST": 2,
    "SENIOR": 3, "SENIOR_SPECIALIST": 3,
    "LEAD": 4, "TEAM_LEAD": 4, "MANAGER": 4,
    "DIRECTOR": 5, "EXECUTIVE": 5,
}


def job_level_num(level):
    return JOB_LEVEL_MAP.get(str(level).upper(), np.nan)


def ndcg_at_k(gains_ranked, all_gains, k=10):
    """Standart NDCG@k (Jarvelin & Kekalainen, 2002; ustel kazanc 2^rel-1, Burges vd. 2005).

    IDCG, sorgunun TUM havuzdaki alaka derecelerinden hesaplanir (yalnizca donen listeden degil).
    """
    g = np.asarray(gains_ranked[:k], dtype=float)
    disc = 1.0 / np.log2(np.arange(2, len(g) + 2))
    dcg = float(((2 ** g - 1) * disc).sum())
    ideal = np.sort(np.asarray(all_gains, dtype=float))[::-1][:k]
    idcg = float(((2 ** ideal - 1) * (1.0 / np.log2(np.arange(2, len(ideal) + 2)))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def safe_div(a, b):
    return a / b if b else 0.0


def log2(x):
    return math.log(x, 2)
