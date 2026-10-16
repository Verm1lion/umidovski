# -*- coding: utf-8 -*-
"""FAZ 1 - Mevcut sistemin adli denetimi (forensic audit).

Bu script eski projedeki veri ve degerlendirme hatalarini SAYISAL olarak kanitlar:

  A. Veri envanteri ve profil (CV + ilan): kolonlar, eksikler, dagilimlar, maskeler
  B. Sablon tekrari (template repetition) ve neredeyse-kopya CV'ler
  C. Eski train/test ayriminda entity / template sizintisi
  D. train.csv etiketlerinin kaynagi (etiket = model benzerligi?) ve kisayol (Sektor: token'i)
  E. test.py degerlendirmesindeki metodolojik hatalar:
       E1 yerel IDCG (NDCG sisirmesi)
       E2 sektor carpani ile dongusel (circular) degerlendirme -> hedef sizintisi
       E3 CMQ "kalite" matrisinin modelin kendi skorunu icermesi
       E4 regex yaka filtresinin hem siralamada hem alakada kullanilmasi
  F. Kod / dokuman tutarsizliklari

Eski test.py fonksiyonlari dogrudan ice aktarilarak (import test) birebir yeniden uretilir.
Model agirliklari bu ortamda bulunmadigindan siralayici olarak (i) rastgele skorlar ve
(ii) BM25 kullanilir; kanitlar modelden BAGIMSIZDIR (metriklerin kendisindeki hatalar).

Cikti: reports/faz1_audit.json, reports/figures/*.png
       (reports/FAZ1_ADLI_DENETIM_RAPORU.md bu ciktilardan elle yazilmistir; script raporu uretmez)
Kullanim: python scripts/03_forensic_audit.py
"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, SparseBM25, load_cvs, load_jobs, ndcg_at_k, parse_seniority_text,  # noqa: E402
                    tokenize, tr_lower, write_json)

sys.path.insert(0, ROOT)
import test as old  # noqa: E402  (eski degerlendirme kodu, birebir)

REPORTS = os.path.join(ROOT, "reports")
FIG = os.path.join(REPORTS, "figures")
RNG = np.random.default_rng(20261015)
R = {}  # tum sayisal bulgular


def pct(x):
    return round(100.0 * float(x), 2)


# ---------------------------------------------------------------------------
# A. Profil
# ---------------------------------------------------------------------------
def profile(cv, jobs):
    a = {}
    parsed = cv.cv_json.apply(json.loads)
    a["cv_rows"] = len(cv)
    a["cv_columns"] = list(cv.columns)
    a["cv_id_range"] = [int(cv.cv_id.min()), int(cv.cv_id.max())]
    a["cv_unique_ids"] = int(cv.cv_id.nunique())
    a["cv_unique_titles"] = int(cv.NaturalTitle.nunique())
    a["cv_collar"] = cv.collar.value_counts().to_dict()
    a["cv_category"] = cv.Category.value_counts().to_dict()
    a["cv_location_top15"] = cv.location.value_counts().head(15).to_dict()
    a["cv_location_n"] = int(cv.location.nunique())
    a["cv_empty"] = {
        "about_bos": int(parsed.apply(lambda j: not str(j.get("about") or "").strip()).sum()),
        "skills_bos": int(parsed.apply(lambda j: not j.get("skills")).sum()),
        "experience_bos": int(parsed.apply(lambda j: not j.get("experience")).sum()),
        "education_bos": int(parsed.apply(lambda j: not j.get("education")).sum()),
        "certifications_bos": int(parsed.apply(lambda j: not j.get("certifications")).sum()),
    }
    for m in ["[TARİH]", "[ŞİRKET]", "[ÜNİVERSİTE]", "[OKUL]"]:
        a["cv_empty"][f"maske_{m}"] = int(cv.cv_text.str.contains(re.escape(m)).sum())
    _ws = lambda s: re.sub(r"\s+", " ", str(s or "")).strip()
    n_diff = int(sum(_ws(j.get("about")) not in _ws(t) for j, t in zip(parsed, cv.cv_text)))
    a["cv_about_cv_text_farkli"] = n_diff
    a["cv_about_cv_text_farkli_pct"] = pct(n_diff / len(cv))
    en_words = {"the", "and", "with", "of", "in", "for", "experience", "skilled"}
    a["cv_ingilizce_agirlikli"] = int(cv.cv_text.apply(
        lambda t: sum(w in en_words for w in re.findall(r"[a-z]+", t.lower())) >= 8).sum())
    a["cv_text_len"] = cv.cv_text.str.len().describe().round(1).to_dict()
    sen = [parse_seniority_text(t, x) for t, x in zip(cv.NaturalTitle, cv.cv_text)]
    a["cv_kidem_metinden_0_5"] = dict(sorted(Counter(sen).items()))
    a["cv_kategori_x_yaka"] = pd.crosstab(cv.Category, cv.collar).to_dict(orient="index")

    a["job_rows"] = len(jobs)
    a["job_unique_titles"] = int(jobs.title.nunique())
    a["job_industry"] = jobs.industry.value_counts().to_dict()
    a["job_position_level"] = jobs.position_level.value_counts().to_dict()
    a["job_mode"] = jobs.mode_of_operation.value_counts().to_dict()
    a["job_type"] = jobs.job_type.value_counts().to_dict()
    a["job_city_top15"] = jobs.city.value_counts().head(15).to_dict()
    a["job_language"] = jobs.raw_language.fillna("YOK").value_counts().to_dict()
    a["job_desc_len"] = jobs.description.str.len().describe().round(1).to_dict()
    a["job_desc_kisa_lt200"] = int((jobs.description.str.len() < 200).sum())
    a["job_desc_bos"] = int((jobs.description.fillna("").str.strip() == "").sum())
    a["job_sehir_maskeli"] = int(jobs.description.str.contains(r"\[şehir\]").sum())
    a["job_title_tekrar"] = int(jobs.title.duplicated().sum())
    a["job_industry_raw_category_esit"] = pct((jobs.industry == jobs.raw_category).mean())
    a["job_has_salary"] = jobs.raw_has_salary.astype(str).value_counts().to_dict()
    # CV / ilan dengesizligi
    ci = cv.Category.value_counts()
    ji = jobs.industry.value_counts()
    a["sektor_cv_ilan_orani"] = {k: {"cv": int(ci.get(k, 0)), "ilan": int(ji.get(k, 0)),
                                     "oran": round(ci.get(k, 0) / max(1, ji.get(k, 0)), 2)}
                                 for k in sorted(set(ci.index) | set(ji.index))}
    return a


# ---------------------------------------------------------------------------
# B. Sablon tekrari
# ---------------------------------------------------------------------------
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def mask_title(text, title):
    t = str(text)
    if title and str(title).strip():
        t = re.sub(re.escape(str(title).strip()), " UNVAN ", t, flags=re.I)
    return t


def sentences(text):
    out = []
    for s in SENT_SPLIT.split(str(text)):
        s = re.sub(r"[^\w\s]", " ", tr_lower(s))
        s = re.sub(r"\s+", " ", s).strip()
        if len(s.split()) >= 8:
            out.append(s)
    return out


def template_repetition(cv):
    b = {}
    masked = [mask_title(t, n) for t, n in zip(cv.cv_text, cv.NaturalTitle)]
    sents = [set(sentences(t)) for t in masked]
    cnt = Counter(s for ss in sents for s in ss)
    templ = {s for s, c in cnt.items() if c >= 5}
    b["sablon_cumle_sayisi_ge5cv"] = len(templ)
    b["sablon_cumle_iceren_cv_orani"] = pct(np.mean([len(ss & templ) > 0 for ss in sents]))
    b["en_sik_sablon_cumleler"] = [{"cumle": s[:160], "cv_sayisi": c} for s, c in cnt.most_common(12)]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform([tr_lower(t) for t in masked])
    S = (X @ X.T).tocsr()
    S.setdiag(0)
    S.eliminate_zeros()
    for th in (0.8, 0.9, 0.95):
        A = S.multiply(S >= th)
        n_pairs = int(A.nnz // 2)
        ncomp, lab = connected_components(A, directed=False)
        sizes = Counter(lab)
        in_cluster = sum(v for v in sizes.values() if v > 1)
        b[f"neredeyse_kopya_cift_cos_ge_{th}"] = n_pairs
        b[f"neredeyse_kopya_kume_icindeki_cv_cos_ge_{th}"] = int(in_cluster)
    return b, X, masked, sents


# ---------------------------------------------------------------------------
# C. Eski split sizintisi
# ---------------------------------------------------------------------------
def split_leakage(cv, train, test, X, sents, jobs):
    c = {}
    tr_ids = set(train.cv_id.unique())
    te_ids = set(test.cv_id.unique())
    c["train_cv"] = len(tr_ids)
    c["test_cv"] = len(te_ids)
    c["id_kesisimi"] = len(tr_ids & te_ids)
    idx = {cid: i for i, cid in enumerate(cv.cv_id)}
    tr_idx = np.array([idx[i] for i in sorted(tr_ids)])
    te_idx = np.array([idx[i] for i in sorted(te_ids)])
    tr_titles = set(cv.NaturalTitle.iloc[tr_idx].str.strip().str.lower())
    te_titles = cv.NaturalTitle.iloc[te_idx].str.strip().str.lower()
    c["test_unvani_trainde_birebir_var"] = pct(te_titles.isin(tr_titles).mean())
    S = (X[te_idx] @ X[tr_idx].T).toarray()
    mx = S.max(axis=1)
    c["test_train_max_cos_quantiles"] = {q: round(float(np.quantile(mx, q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    for th in (0.7, 0.8, 0.9):
        c[f"test_cv_trainde_cos_ge_{th}_ikizi_var"] = pct((mx >= th).mean())
    tr_sent = set().union(*[sents[i] for i in tr_idx])
    c["test_cv_trainle_birebir_cumle_paylasan"] = pct(np.mean([len(sents[i] & tr_sent) > 0 for i in te_idx]))
    # kategori dagilimi farki (Jensen-Shannon)
    cats = sorted(cv.Category.unique())
    p = cv.Category.iloc[tr_idx].value_counts(normalize=True).reindex(cats, fill_value=0).values
    q = cv.Category.iloc[te_idx].value_counts(normalize=True).reindex(cats, fill_value=0).values
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())

    c["kategori_js_divergence_bits"] = round(0.5 * kl(p, m) + 0.5 * kl(q, m), 4)
    c["test_yaka"] = cv.collar.iloc[te_idx].value_counts().to_dict()
    c["test_kategori_top10"] = cv.Category.iloc[te_idx].value_counts().head(10).to_dict()
    # ilan ezberi: testte "dogru" sayilan (ayni sektor) ilanlarin ne kadari trainde pozitif goruldu
    pos = train[train.match_type.isin(["EXACT_POS", "SOFT_POS"])]
    pos_jobs = set(pos.job_id)
    c["trainde_pozitif_gorulen_ilan"] = len(pos_jobs)
    c["trainde_herhangi_gorulen_ilan"] = int(train.job_id.nunique())
    ind_jobs = jobs.groupby("industry").job_id.apply(set).to_dict()
    frac = []
    for i in te_idx:
        s = ind_jobs.get(cv.Category.iloc[i], set())
        if s:
            frac.append(len(s & pos_jobs) / len(s))
    c["test_icin_ayni_sektor_ilanlarin_trainde_pozitif_orani"] = pct(np.mean(frac))
    return c, mx


# ---------------------------------------------------------------------------
# D. train.csv etiket kaynagi
# ---------------------------------------------------------------------------
def label_provenance(train):
    d = {}
    t = train.copy()
    t["label"] = t.label.astype(float)
    t["sim_score"] = t.sim_score.astype(float)
    d["satir"] = len(t)
    d["match_type"] = t.match_type.value_counts().to_dict()
    per = {}
    for mt, g in t.groupby("match_type"):
        per[mt] = {
            "n": len(g),
            "label_ort": round(g.label.mean(), 3),
            "label_tekil_deger": int(g.label.nunique()),
            "sim_ort": round(g.sim_score.mean(), 3),
            "pearson_label_sim": round(float(np.corrcoef(g.label, g.sim_score)[0, 1]), 3) if g.label.nunique() > 1 else None,
            "ayni_sektor_orani": pct((g.cand_industry == g.job_industry).mean()),
        }
    d["match_type_ozet"] = per
    d["pearson_label_sim_tum"] = round(float(np.corrcoef(t.label, t.sim_score)[0, 1]), 3)
    # Etiketlerin sim_score'un dogrusal donusumu olup olmadigi (SOFT_POS / SOFT_NEG)
    fits = {}
    for mt in ("SOFT_POS", "SOFT_NEG", "EXACT_POS"):
        g = t[t.match_type == mt]
        if len(g) > 10 and g.label.nunique() > 1:
            A = np.vstack([g.sim_score, np.ones(len(g))]).T
            coef, res, *_ = np.linalg.lstsq(A, g.label, rcond=None)
            pred = A @ coef
            ss_res = float(((g.label - pred) ** 2).sum())
            ss_tot = float(((g.label - g.label.mean()) ** 2).sum())
            fits[mt] = {"egim": round(float(coef[0]), 4), "kesim": round(float(coef[1]), 4),
                        "R2": round(1 - ss_res / ss_tot, 4) if ss_tot else None}
    d["label_sim_dogrusal_uyum"] = fits
    d["job_text_sektor_tokeni_iceriyor"] = pct(t.apply(lambda r: f"Sektör: {r.job_industry}" in str(r.job_text), axis=1).mean())
    d["job_text_pozisyon_seviyesi_iceriyor"] = pct(t.job_text.str.contains("Pozisyon Seviyesi:").mean())
    d["cand_seniority"] = t.drop_duplicates("cv_id").cand_seniority.value_counts().to_dict()
    neg = t[t.binary_label.astype(int) == 0]
    d["negatif_cift_farkli_sektor"] = pct((neg.cand_industry != neg.job_industry).mean())
    d["hard_neg_farkli_sektor"] = pct((t[t.match_type == "HARD_NEG"].cand_industry != t[t.match_type == "HARD_NEG"].job_industry).mean())
    d["pozitif_cift_ayni_sektor"] = pct((t[t.binary_label.astype(int) == 1].cand_industry == t[t.binary_label.astype(int) == 1].job_industry).mean())
    return d


# ---------------------------------------------------------------------------
# E. test.py metodoloji hatalari
# ---------------------------------------------------------------------------
def old_style_frames(cv, jobs, test_ids):
    """test.py'nin yaptigi on islemenin birebir kopyasi (DB yerine parquet)."""
    cands = cv[cv.cv_id.isin(test_ids)].drop_duplicates("cv_id").reset_index(drop=True).copy()
    cands["collar"] = cands.collar.apply(lambda c: "WHITE" if "BEYAZ" in str(c).upper() else ("BLUE" if "MAVI" in str(c).upper() else "WHITE"))
    titles = cands.NaturalTitle.fillna("").astype(str).str.strip()
    cands["seniority"] = titles.apply(old.extract_cand_seniority)
    cands["city"] = cands.apply(old.get_cand_city, axis=1)
    jb = jobs.copy()
    jb["collar"] = jb.title.apply(old.classify_collar).apply(lambda c: "WHITE" if c == "WHITE" else "BLUE")
    jb["raw_data"] = jb.raw_data_json.apply(json.loads)
    jb["city"] = jb.raw_data.apply(old.extract_job_city)
    jb["is_remote"] = jb.mode_of_operation.astype(str).str.lower() == "remote"
    return cands, jb


def topk(scores, k):
    return np.argsort(-scores, axis=1)[:, :k]


def eval_industry(scores, cand_ind, job_ind, k=10):
    """test.py'deki ana dongu: rel=2 ayni sektor ise; P@10, Top-1, NDCG (yerel IDCG) + dogru NDCG."""
    top = topk(scores, k)
    rel = (job_ind[top] == cand_ind[:, None]).astype(int) * 2
    p10 = (rel > 0).mean(axis=1).mean()
    top1 = (rel[:, 0] > 0).mean()
    nd_local = np.mean([old.calculate_ndcg(list(r), k=k)[2] for r in rel])
    nd_true = []
    for i in range(len(cand_ind)):
        all_g = (job_ind == cand_ind[i]).astype(int) * 2
        nd_true.append(ndcg_at_k(rel[i], all_g, k))
    return {"P@10": pct(p10), "Top1": pct(top1), "NDCG@10_testpy_yerel_IDCG": pct(nd_local),
            "NDCG@10_dogru_IDCG": pct(np.mean(nd_true))}


def evaluation_flaws(cv, jobs, test):
    e = {}
    cands, jb = old_style_frames(cv, jobs, set(test.cv_id))
    cand_ind = cands.Category.str.upper().to_numpy(dtype=object)
    job_ind = jb.industry.str.upper().to_numpy(dtype=object)
    nq, nd = len(cands), len(jb)

    # E1: yerel IDCG
    e["E1_ornek_[2,0,0,0,0,0,0,0,0,0]_testpy_ndcg"] = old.calculate_ndcg([2] + [0] * 9)[2]
    e["E1_ornek_[0,0,0,0,0,0,0,0,0,2]_testpy_ndcg"] = round(old.calculate_ndcg([0] * 9 + [2])[2], 4)
    e["E1_ornek_[2,0,...]_dogru_ndcg_(havuzda_50_ilgili)"] = round(ndcg_at_k([2] + [0] * 9, [2] * 50, 10), 4)

    # Siralayicilar: (i) rastgele e5-benzeri kosinus (0.70-0.90), (ii) BM25
    rand = RNG.uniform(0.70, 0.90, size=(nq, nd))
    bm = SparseBM25([tokenize(f"{t} {d}") for t, d in zip(jb.title, jb.description)])
    bm_scores = bm.scores([tokenize(t) for t in cands.cv_text])
    bm_norm = bm_scores / (bm_scores.max(axis=1, keepdims=True) + 1e-9)  # [0,1]

    same_ind = (cand_ind[:, None] == job_ind[None, :])
    res = {}
    res["rastgele"] = eval_industry(rand, cand_ind, job_ind)
    res["rastgele_x_sektor_carpani_0.85(match_candidate)"] = eval_industry(rand * np.where(same_ind, 1.0, 0.85), cand_ind, job_ind)
    res["rastgele_x_sektor_carpani_0.50(rehber)"] = eval_industry(rand * np.where(same_ind, 1.0, 0.50), cand_ind, job_ind)
    res["BM25"] = eval_industry(bm_norm, cand_ind, job_ind)
    res["BM25_x_sektor_carpani_0.50"] = eval_industry(bm_norm * np.where(same_ind, 1.0, 0.50), cand_ind, job_ind)
    e["E2_sektor_carpani_dongusel"] = res
    # analitik kosul: s in [a,b], carpan c ise c*b < a => capraz-sektor ilan asla ust siralara cikamaz
    e["E2_analitik"] = {"a": 0.70, "b": 0.90, "c=0.50": 0.50 * 0.90 < 0.70, "c=0.85": 0.85 * 0.90 < 0.70}
    n_same = same_ind.sum(axis=1)
    e["E2_ayni_sektor_ilan_sayisi_min_medyan"] = [int(n_same.min()), int(np.median(n_same))]

    # E3: CMQ, modelin kendi skorunu %40 agirlikla iceriyor
    Q_self = old.compute_all_qualities_matrix(cands, jb, rand, cands.Category.tolist())
    rand2 = RNG.uniform(0.70, 0.90, size=(nq, nd))
    Q_indep = old.compute_all_qualities_matrix(cands, jb, rand2, cands.Category.tolist())
    top = topk(rand, 10)
    g_self = np.mean([old.compute_graded_ndcg(top[i], Q_self[i], 10) for i in range(nq)])
    g_indep = np.mean([old.compute_graded_ndcg(top[i], Q_indep[i], 10) for i in range(nq)])
    e["E3_CMQ"] = {
        "rastgele_siralayici_kendi_skoruyla_kurulan_CMQ_graded_NDCG": round(g_self, 2),
        "ayni_siralayici_bagimsiz_skorla_kurulan_CMQ_graded_NDCG": round(g_indep, 2),
        "fark_puan": round(g_self - g_indep, 2),
    }
    # E4: yaka regex'i hem filtre hem alaka
    cand_col = cands.collar.to_numpy(dtype=object)
    job_col = jb.collar.to_numpy(dtype=object)
    same_col = cand_col[:, None] == job_col[None, :]
    out = {}
    for name, sc, relmode in [
        ("raw_alaka=sektor", rand, "raw"),
        ("collar_filtre+alaka=sektor&yaka", rand * np.where(same_col, 1.0, 0.75), "collar"),
    ]:
        t5 = topk(sc, 5)
        if relmode == "raw":
            rel = (job_ind[t5] == cand_ind[:, None])
        else:
            rel = (job_ind[t5] == cand_ind[:, None]) & (job_col[t5] == cand_col[:, None])
        out[name] = {"P@5": pct(rel.mean()), "Top1": pct(rel[:, 0].mean())}
    e["E4_yaka_regex_dongusel_(rastgele_skor)"] = out
    e["E4_regex_yaka_ile_veri_yakasi_uyusmazligi_CV"] = pct(
        (cv.NaturalTitle.apply(old.classify_collar).to_numpy(dtype=object) != cv.collar.map({"BEYAZ": "WHITE", "MAVI": "BLUE"}).to_numpy(dtype=object)).mean())
    return e


# ---------------------------------------------------------------------------
# F. Statik tutarsizliklar
# ---------------------------------------------------------------------------
def static_findings(cv):
    f = {}
    for cid in (10042, 10587, 20951):
        r = cv[cv.cv_id == cid].iloc[0]
        f[f"cv_{cid}"] = {"NaturalTitle": r.NaturalTitle, "Category": r.Category, "collar": r.collar}
    f["carpanlar"] = {
        "rehber_sektor": 0.50, "match_candidate_sektor": 0.85, "test.py_soft_rerank_sektor": "yok (CMQ icinde %25)",
        "match_candidate_yaka": 0.0, "test.py_yaka": 0.75,
        "kidem": "0.95^(fark-1) (kod) / 0.70-1.00 (rehber)",
        "lokasyon": "0.85 mavi / 0.95 beyaz (kod); CMQ icinde 0.40 / 0.70",
    }
    return f


def figures(c_mx, a):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    os.makedirs(FIG, exist_ok=True)
    out = []
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(c_mx, bins=40, color="#4C72B0")
    ax.axvline(0.8, color="r", ls="--", label="cos=0.8")
    ax.set_xlabel("Test CV'nin en yakin train CV'sine kosinusu (unvan maskeli TF-IDF)")
    ax.set_ylabel("CV sayisi")
    ax.set_title("Eski split: test -> train neredeyse-kopya dagilimi")
    ax.legend()
    p = os.path.join(FIG, "faz1_test_train_max_cos.png")
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out.append(p)
    d = pd.DataFrame(a["sektor_cv_ilan_orani"]).T.sort_values("ilan", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(d))
    ax.bar(x - 0.2, d.cv, 0.4, label="CV")
    ax.bar(x + 0.2, d.ilan, 0.4, label="Ilan")
    ax.set_xticks(x)
    ax.set_xticklabels(d.index, rotation=75, fontsize=7)
    ax.legend()
    ax.set_title("Sektor bazinda CV / ilan sayilari")
    fig.tight_layout()
    p = os.path.join(FIG, "faz1_sektor_cv_ilan.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out.append(p)
    return out


def main():
    cv = load_cvs()
    jobs = load_jobs()
    train = pd.read_csv(os.path.join(ROOT, "veriseti", "train.csv"), encoding="utf-8-sig")
    test = pd.read_csv(os.path.join(ROOT, "veriseti", "test.csv"), encoding="utf-8-sig")
    print("A. profil ...")
    R["A_profil"] = profile(cv, jobs)
    print("B. sablon tekrari ...")
    b, X, masked, sents = template_repetition(cv)
    R["B_sablon"] = b
    print("C. split sizintisi ...")
    c, mx = split_leakage(cv, train, test, X, sents, jobs)
    R["C_eski_split"] = c
    print("D. etiket kaynagi ...")
    R["D_train_etiketleri"] = label_provenance(train)
    print("E. degerlendirme hatalari ...")
    R["E_testpy"] = evaluation_flaws(cv, jobs, test)
    R["F_statik"] = static_findings(cv)
    R["figurler"] = [os.path.relpath(p, ROOT) for p in figures(mx, R["A_profil"])]
    write_json(R, os.path.join(REPORTS, "faz1_audit.json"))
    print(json.dumps({k: R[k] for k in ("C_eski_split", "E_testpy")}, ensure_ascii=False, indent=1, default=str)[:6000])
    print("-> reports/faz1_audit.json")


if __name__ == "__main__":
    main()
