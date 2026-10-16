# -*- coding: utf-8 -*-
"""FAZ 2 - Sifir sizintili veri muhendisligi, kademeli qrels, zor negatifler ve paketleme.

Girdiler
  veriseti/tum_cvler.csv, work/tum_ilanlar.parquet (01), annotation/occupation_labels.parquet (04),
  annotation/synthetic/synthetic_cvs.json (06, opsiyonel; YALNIZCA train'e eklenir),
  veriseti/train.csv (yalnizca Baseline-3 'eski yaklasim' ciftlerini sizintisiz yeniden kurmak icin)

Ciktilar (data_bundle/ ve data_bundle.zip)
  clean_cvs.parquet          tum CV'ler (gercek + sentetik), yapilandirilmis metin, split, gold_* etiketler
  clean_jobs.parquet         3.000 ilan, yapilandirilmis metin, job_pool (warm/cold), gold_* etiketler
  train_triplets.parquet     (anchor, positive, neg_lexical, neg_seniority_collar, neg_bm25) - yalnizca train
  val_triplets.parquet       dogrulama kaybi (eval loss) icin ayni bicim - yalnizca val
  train_queries / val_queries / test_queries.parquet   sorgu kimlikleri + sorgu metni (gold YOK)
  qrels_ground_truth.parquet (cv_id, job_key, grade 1..3) - grade 0 ortuk
  legacy_train_pairs.parquet eski train.csv'nin YALNIZCA yeni train bolumundeki CV'lere ait satirlari
  metadata.json, README.md, match_candidate_sota.py

Kurallar
  * Split, CV duzeyinde ayriktir; ayni normalize unvana sahip CV'ler ve neredeyse-kopyalar
    (unvan maskeli TF-IDF cos >= 0.8) ayni gruba konur (StratifiedGroupKFold).
  * Vitrin (showcase) adaylari test'e sabitlenir: hata analizinde gosterilen her ornek, egitimde
    hic gorulmemis olur.
  * Ilanlarin %10'u 'cold' havuzuna ayrilir: hicbir egitim sinyalinde (triplet, sentetik CV,
    LTR egitimi) kullanilmaz; soguk-baslangic (cold-start) genellemesi ayrica olculur.

Kullanim: python scripts/05_build_dataset.py [--no-zip]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_cvs, load_jobs, write_json  # noqa: E402

sys.path.insert(0, ROOT)
import match_candidate_sota as core  # noqa: E402
import test as old  # noqa: E402  (eski sistemin fonksiyonlari: yalnizca Baseline-3 sizinti gosterimi icin)

SEED = 20261015
OUT = os.environ.get("BUNDLE_OUT", os.path.join(ROOT, "data_bundle"))
ZIP_PATH = os.environ.get("BUNDLE_ZIP", os.path.join(ROOT, "data_bundle.zip"))
ANN = os.path.join(ROOT, "annotation")
LABELS_PATH = os.environ.get("OCC_LABELS", os.path.join(ANN, "occupation_labels.parquet"))
REPORTS = os.environ.get("REPORTS_DIR", os.path.join(ROOT, "reports"))
SHOWCASE = [20951, 10587, 10511, 21029]   # Baklava ustasi, ucus dispeceri, Java gelistirici, Java stajyeri
COLD_FRAC = 0.10
MAX_POS = 4
REL_THRESHOLD = 2
LEX_MIN_SIM = 0.05      # leksikal tuzak negatifinin anchor unvanina en dusuk karakter n-gram benzerligi
NEG_USE_CAP = 25        # ayni ilanin bir negatif turunde en fazla kac satirda kullanilacagi (hub negatifleri sinirla)

GRADE_RULES = {
    "3": "Tam eslesme: ayni uzmanlik (occ_key) + yaka uyumu + |kidem farki| <= 1 + lokasyon uyumu (ayni sehir veya Remote)",
    "2": "Yuksek alaka: ayni ISCO-08 birim grubu (isco4) + yaka uyumu + |kidem farki| <= 2 (kidem/lokasyonda tolere edilebilir fark)",
    "1": "Kismi alaka: yaka uyumu + (ayni ana sektor VEYA ayni isco4 fakat kidem farki >= 3)",
    "0": "Alakasiz / uygunsuz: diger tum durumlar (farkli yaka, farkli sektor ve farkli meslek)",
}


def spec_key(s):
    """Unvan anahtari: ASCII katlama + noktalama silme + KELIME SIRASINDAN BAGIMSIZ (sirali kume).

    "Satis Muduru Bolge" ile "Bolge Satis Muduru" ayni anahtari alir; boylece yalnizca sozcuk sirasi farkli
    unvanlar bolumler arasina dagilmaz.
    """
    toks = re.sub(r"[^a-z0-9]+", " ", core.tr_ascii(s)).split()
    return " ".join(sorted(set(toks)))


def clean_legacy(text):
    t = str(text).replace("\n", " ")
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
def load_labels():
    lab = pd.read_parquet(LABELS_PATH)
    # yaka, meslegin ozelligidir: ayni uzmanlik (occ_key) icin cogunluk oyu (CV + ilan birlikte)
    maj = lab.groupby("occ_key").collar.agg(lambda s: Counter(s).most_common(1)[0][0])
    lab["collar_harmonized"] = lab.occ_key.map(maj)
    return lab.set_index("id")


def load_synthetic():
    p = os.path.join(ANN, "synthetic", "synthetic_cvs.json")
    if not os.path.exists(p):
        return None
    rows = json.load(open(p, encoding="utf-8"))
    return pd.DataFrame(rows)


def build_cvs(lab, syn):
    cv = load_cvs()
    out = pd.DataFrame({
        "cv_id": cv.cv_id.astype(int),
        "source": "real",
        "headline": cv.NaturalTitle.astype(str),
        "cv_text": cv.cv_text.astype(str),
    })
    ids = "C" + out.cv_id.astype(str)
    out["gold_industry"] = cv.Category.astype(str).values
    out["gold_collar_dataset"] = cv.collar.map({"BEYAZ": "WHITE", "MAVI": "BLUE"}).values
    out["gold_city"] = [core.norm_city(c) for c in cv.location]
    for c_src, c_dst in [("isco4", "gold_isco4"), ("isco_title", "gold_isco_title"), ("specialty", "gold_specialty"),
                         ("occ_key", "gold_occ_key"), ("collar_harmonized", "gold_collar"),
                         ("seniority", "gold_seniority"), ("conf", "gold_conf")]:
        out[c_dst] = ids.map(lab[c_src]).values
    if syn is not None and len(syn):
        s = pd.DataFrame({
            "cv_id": syn.syn_id.astype(int),
            "source": "synthetic",
            "headline": syn.headline.astype(str),
            "cv_text": syn.cv_text.astype(str),
            "gold_industry": syn.gold_industry.astype(str),
            "gold_collar_dataset": syn.gold_collar.astype(str),
            "gold_city": [core.norm_city(c) for c in syn.city],
            "gold_isco4": syn.gold_isco4.astype(str),
            "gold_isco_title": syn.gold_isco_title.astype(str),
            "gold_specialty": syn.gold_specialty.astype(str),
            "gold_occ_key": syn.gold_occ_key.astype(str),
            "gold_collar": syn.gold_collar.astype(str),
            "gold_seniority": syn.gold_seniority.astype(int),
            "gold_conf": 3,
        })
        out = pd.concat([out, s], ignore_index=True)
    if out.gold_occ_key.isna().any():
        raise ValueError(f"Etiketi eksik CV: {out[out.gold_occ_key.isna()].cv_id.tolist()[:10]}")
    # eski sistemin (match_candidate.py / test.py) aday alanlari - YALNIZCA sizinti gosterimi (Baseline-3 + eski carpanlar)
    out["legacy_collar"] = out.gold_collar_dataset.fillna("WHITE")
    out["legacy_seniority"] = [old.extract_cand_seniority(h) for h in out.headline]
    out["legacy_city"] = [old.get_cand_city({"location": c, "cv_text": t}) for c, t in
                          zip(list(cv.location) + list(syn.city if syn is not None and len(syn) else []), out.cv_text)]
    # yalnizca metinden turetilen alanlar (ozellik tarafi)
    out["cv_struct"] = [core.build_cv_struct(h, t) for h, t in zip(out.headline, out.cv_text)]
    out["text_city"] = [core.norm_city(core.parse_cv_sections(t)["konum"]) for t in out.cv_text]
    out["text_seniority"] = [core.parse_seniority_text(h, t) for h, t in zip(out.headline, out.cv_text)]
    yrs = [core.parse_years_experience(t) for t in out.cv_text]
    out["text_years_exp"] = [np.nan if y is None else float(y) for y in yrs]
    return out


def build_jobs(lab):
    jobs = load_jobs()
    out = jobs[["job_key", "job_id", "title", "company_name", "description", "industry", "position_level",
                "mode_of_operation", "job_type", "city", "district"]].copy()
    out["job_struct"] = [core.build_job_struct(r.title, r.company_name, r.industry, r.position_level,
                                               r.mode_of_operation, r.city, r.description, r.job_type)
                         for r in out.itertuples()]
    for c_src, c_dst in [("isco4", "gold_isco4"), ("isco_title", "gold_isco_title"), ("specialty", "gold_specialty"),
                         ("occ_key", "gold_occ_key"), ("collar_harmonized", "gold_collar"),
                         ("seniority", "gold_seniority"), ("conf", "gold_conf")]:
        out[c_dst] = out.job_key.map(lab[c_src]).values
    if out.gold_occ_key.isna().any():
        raise ValueError("Etiketi eksik ilan var")
    out["gold_city"] = [core.norm_city(c) for c in out.city]
    out["is_remote"] = out.mode_of_operation.astype(str) == "Remote"
    # Baseline-3 (eski yaklasim) metinleri: egitimdeki ve test.py'deki bicimleri birebir
    legacy = {}
    tr = pd.read_csv(os.path.join(ROOT, "veriseti", "train.csv"), encoding="utf-8-sig", usecols=["job_id", "job_text"])
    for jid, txt in zip(tr.job_id, tr.job_text):
        legacy.setdefault(jid, txt)
    out["legacy_train_text"] = [legacy.get(r.job_id) or clean_legacy(
        f"İş İlanı: {r.title} Firma: {r.company_name} Sektör: {r.industry} Pozisyon Seviyesi: {r.position_level} "
        f"Açıklama: {r.description}") for r in out.itertuples()]
    out["legacy_collar"] = [("WHITE" if old.classify_collar(t) == "WHITE" else "BLUE") for t in out.title]
    out["legacy_city"] = [old.extract_job_city(json.loads(r)) for r in jobs.raw_data_json]
    out["legacy_infer_text"] = [clean_legacy(f"İş İlanı: {r.title} Firma: {'' if pd.isna(r.company_name) else r.company_name}"
                                             f" Açıklama: {str(r.description)[:2000]}") for r in out.itertuples()]
    return out


# ---------------------------------------------------------------------------
def make_split(cvs):
    real = cvs[cvs.source == "real"].copy()
    n = len(real)
    # 1) ayni normalize unvan -> ayni grup; 2) neredeyse-kopyalar (cos >= 0.8) -> ayni grup
    tkey = real.headline.map(spec_key)
    masked = [re.sub(re.escape(h), " UNVAN ", t, flags=re.I) for h, t in zip(real.headline, real.cv_text)]
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform([core.tr_lower(t) for t in masked])
    S = (X @ X.T).tocsr()
    S = S.multiply(S >= 0.8).tocsr()
    codes = pd.factorize(tkey)[0]
    rows = np.arange(n)
    T = sp.csr_matrix((np.ones(n), (rows, codes)), shape=(n, codes.max() + 1))
    A = ((T @ T.T) + S) > 0
    _, group = connected_components(A, directed=False)
    real["group_id"] = group
    y = real.gold_industry.values
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    fold = np.empty(n, dtype=int)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(n), y, groups=group)):
        fold[te] = k
    split = np.where(fold <= 2, "test", np.where(fold <= 5, "val", "train"))
    pinned_groups = set(real.group_id[real.cv_id.isin(SHOWCASE)])
    split = np.where(real.group_id.isin(pinned_groups), "test", split)
    real["split"] = split
    cvs = cvs.merge(real[["cv_id", "group_id", "split"]], on="cv_id", how="left")
    syn = cvs.source == "synthetic"
    cvs.loc[syn, "split"] = "train"
    cvs.loc[syn, "group_id"] = -1
    cvs["group_id"] = cvs.group_id.astype(int)
    return cvs


def make_job_pool(jobs):
    rng = np.random.default_rng(SEED)
    pool = np.array(["warm"] * len(jobs), dtype=object)
    for ind, idx in jobs.groupby("industry").groups.items():
        idx = np.asarray(list(idx))
        k = int(round(COLD_FRAC * len(idx)))
        if k:
            pool[rng.choice(idx, size=k, replace=False)] = "cold"
    jobs["job_pool"] = pool
    return jobs


# ---------------------------------------------------------------------------
def grade_matrix(cvs, jobs):
    """Tum CV x ilan icin kademeli alaka (0..3). Yalnizca gold_* alanlar kullanilir."""
    occ_c, occ_j = cvs.gold_occ_key.to_numpy(dtype=object), jobs.gold_occ_key.to_numpy(dtype=object)
    isc_c, isc_j = cvs.gold_isco4.to_numpy(dtype=object), jobs.gold_isco4.to_numpy(dtype=object)
    col_c, col_j = cvs.gold_collar.to_numpy(dtype=object), jobs.gold_collar.to_numpy(dtype=object)
    ind_c, ind_j = cvs.gold_industry.to_numpy(dtype=object), jobs.industry.to_numpy(dtype=object)
    city_c, city_j = cvs.gold_city.to_numpy(dtype=object), jobs.gold_city.to_numpy(dtype=object)
    sen_c, sen_j = cvs.gold_seniority.to_numpy(dtype=float), jobs.gold_seniority.to_numpy(dtype=float)
    remote = jobs.is_remote.to_numpy(dtype=bool)
    same_occ = occ_c[:, None] == occ_j[None, :]
    same_isco = isc_c[:, None] == isc_j[None, :]
    collar_ok = col_c[:, None] == col_j[None, :]
    same_ind = ind_c[:, None] == ind_j[None, :]
    sd = np.abs(sen_c[:, None] - sen_j[None, :])
    loc_ok = remote[None, :] | ((city_c[:, None] == city_j[None, :]) & (city_c[:, None] != ""))
    g3 = same_occ & collar_ok & (sd <= 1) & loc_ok
    g2 = same_isco & collar_ok & (sd <= 2) & ~g3
    g1 = collar_ok & (same_ind | same_isco) & ~g3 & ~g2
    G = (3 * g3 + 2 * g2 + 1 * g1).astype(np.int8)
    return G


def mine_triplets(cvs, jobs, G, split, rng, max_pos=MAX_POS):
    """Her anchor CV icin pozitif(ler) + 3 tur zor negatif."""
    warm = (jobs.job_pool == "warm").to_numpy()
    anchors = np.where((cvs.split == split).to_numpy())[0]
    # unvan karakter n-gram benzerligi (alt-kelime tuzaklari: Dispecer ~ Distributor, Usta ~ Usta)
    # vektorlestirici YALNIZCA ilan basliklari + train CV basliklariyla fit edilir (val/test IDF'e karismaz)
    ts = core.TitleSimilarity().fit(list(jobs.title) + list(cvs.headline[cvs.split == "train"]))
    TJ = ts.transform(jobs.title)
    TC = ts.transform(cvs.headline.iloc[anchors])
    tsim = np.asarray((TC @ TJ.T).todense())
    bm = core.SparseBM25([core.tokenize(t) for t in jobs.job_struct])
    bms = bm.scores([core.tokenize(t) for t in cvs.cv_struct.iloc[anchors]])
    ind_c = cvs.gold_industry.to_numpy(dtype=object)
    ind_j = jobs.industry.to_numpy(dtype=object)
    isc2_c = cvs.gold_isco4.str[:2].to_numpy(dtype=object)
    isc2_j = jobs.gold_isco4.str[:2].to_numpy(dtype=object)
    col_c = cvs.gold_collar.to_numpy(dtype=object)
    col_j = jobs.gold_collar.to_numpy(dtype=object)
    sen_c = cvs.gold_seniority.to_numpy(dtype=float)
    sen_j = jobs.gold_seniority.to_numpy(dtype=float)
    keys = jobs.job_key.to_numpy(dtype=object)
    rows, no_pos = [], 0
    stats = Counter()
    neg_use = {"lex": Counter(), "trap": Counter(), "bm25": Counter()}
    for ai, a in enumerate(anchors):
        g = G[a]
        p3 = np.where((g == 3) & warm)[0]
        p2 = np.where((g == 2) & warm)[0]
        if len(p3):
            pos = rng.permutation(p3)[:max_pos]
        elif len(p2):
            pos = rng.permutation(p2)[: max(1, max_pos // 2)]
        else:
            no_pos += 1
            continue
        lex = np.where((g == 0) & warm & (ind_j != ind_c[a]))[0]
        lex = lex[np.argsort(-tsim[ai, lex], kind="stable")]
        lex = lex[tsim[ai, lex] >= LEX_MIN_SIM]
        lex = np.asarray([j for j in lex if neg_use["lex"][j] < NEG_USE_CAP][:8] or list(lex[:8]), dtype=int)
        trap = np.where((g <= 1) & warm & ((ind_j == ind_c[a]) | (isc2_j == isc2_c[a]))
                        & ((col_j != col_c[a]) | (np.abs(sen_j - sen_c[a]) >= 3)))[0]
        trap = trap[np.argsort(-tsim[ai, trap], kind="stable")]
        trap = np.asarray([j for j in trap if neg_use["trap"][j] < NEG_USE_CAP][:8] or list(trap[:8]), dtype=int)
        bmn = np.where((g == 0) & warm)[0]
        bmn = bmn[np.argsort(-bms[ai, bmn], kind="stable")]
        bmn = np.asarray([j for j in bmn[:40] if neg_use["bm25"][j] < NEG_USE_CAP][:10] or list(bmn[:10]), dtype=int)
        trap_fb = not len(trap)
        if trap_fb:
            stats["trap_fallback_bm25"] += 1
            trap = bmn[3:]
        if not len(lex):
            stats["lex_fallback_bm25"] += 1
            lex = bmn[5:]
        for k, p in enumerate(pos):
            used = set()

            def pick(cands, off):
                for c in list(cands[off % max(1, len(cands)):]) + list(cands[: off % max(1, len(cands))]):
                    if c not in used:
                        used.add(c)
                        return c
                return None
            nl, nt, nb = pick(lex, k), pick(trap, k), pick(bmn, k)
            if None in (nl, nt, nb):
                stats["eksik_negatif"] += 1
                continue
            neg_use["lex"][nl] += 1
            neg_use["trap"][nt] += 1
            neg_use["bm25"][nb] += 1
            if trap_fb:
                stats["trap_fallback_bm25_satir"] += 1
            rows.append({
                "anchor_cv_id": int(cvs.cv_id.iloc[a]),
                "positive_job_key": keys[p], "positive_grade": int(g[p]),
                "neg_lexical_job_key": keys[nl], "neg_seniority_collar_job_key": keys[nt], "neg_bm25_job_key": keys[nb],
                "neg_lexical_grade": int(g[nl]), "neg_seniority_collar_grade": int(g[nt]), "neg_bm25_grade": int(g[nb]),
                "neg_lexical_title_sim": round(float(tsim[ai, nl]), 4),
                "neg_seniority_collar_is_fallback": bool(trap_fb),
                "anchor_source": cvs.source.iloc[a],
            })
    stats["anchor_pozitifsiz_atlandi"] = no_pos
    for t, c in neg_use.items():
        stats[f"en_cok_kullanilan_{t}_negatif_satir"] = int(max(c.values())) if c else 0
    stats["anchor"] = len(anchors)
    return pd.DataFrame(rows), dict(stats)


# ---------------------------------------------------------------------------
def leakage_checks(cvs, jobs, trip, vtrip, legacy):
    chk = {}
    sets = {s: set(cvs.cv_id[cvs.split == s]) for s in ("train", "val", "test")}
    chk["id_kesisim_train_val"] = len(sets["train"] & sets["val"])
    chk["id_kesisim_train_test"] = len(sets["train"] & sets["test"])
    chk["id_kesisim_val_test"] = len(sets["val"] & sets["test"])
    tk = cvs.headline.map(spec_key)
    chk["unvan_kesisim_train_test"] = len(set(tk[cvs.split == "train"]) & set(tk[cvs.split == "test"]))
    chk["unvan_kesisim_gercek_train_test"] = len(set(tk[(cvs.split == "train") & (cvs.source == "real")]) & set(tk[cvs.split == "test"]))
    chk["unvan_kesisim_train_val"] = len(set(tk[cvs.split == "train"]) & set(tk[cvs.split == "val"]))
    chk["sentetik_val_test_te"] = int(((cvs.source == "synthetic") & (cvs.split != "train")).sum())
    chk["triplet_anchor_train_disi"] = int((~trip.anchor_cv_id.isin(sets["train"])).sum())
    chk["val_triplet_anchor_val_disi"] = int((~vtrip.anchor_cv_id.isin(sets["val"])).sum())
    cold = set(jobs.job_key[jobs.job_pool == "cold"])
    tcols = ["positive_job_key", "neg_lexical_job_key", "neg_seniority_collar_job_key", "neg_bm25_job_key"]
    chk["triplette_cold_ilan"] = int(sum(trip[c].isin(cold).sum() for c in tcols))
    chk["legacy_pairs_train_disi_cv"] = int((~legacy.cv_id.isin(sets["train"])).sum())
    chk["legacy_pairs_cold_ilan"] = int(legacy.job_key.isin(cold).sum())
    # test -> train metin benzerligi (unvan maskeli TF-IDF)
    real = cvs[cvs.source == "real"]
    masked = [re.sub(re.escape(h), " UNVAN ", t, flags=re.I) for h, t in zip(real.headline, real.cv_text)]
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform([core.tr_lower(t) for t in masked])
    tr_i = np.where(real.split.values == "train")[0]
    te_i = np.where(real.split.values == "test")[0]
    mx = np.asarray((X[te_i] @ X[tr_i].T).max(axis=1).todense()).ravel()
    chk["test_train_max_cos_medyan"] = round(float(np.median(mx)), 3)
    chk["test_train_max_cos_p90"] = round(float(np.quantile(mx, 0.9)), 3)
    chk["test_cv_cos_ge_0.8_trainde_ikizi"] = int((mx >= 0.8).sum())
    # sentetik CV'lerin test CV'lerine en yakin benzerligi
    syn = cvs[cvs.source == "synthetic"]
    if len(syn):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(cvs.cv_text.map(core.tr_lower))
        A = vec.transform(syn.cv_text.map(core.tr_lower))
        B = vec.transform(cvs.cv_text[cvs.split == "test"].map(core.tr_lower))
        m2 = np.asarray((A @ B.T).max(axis=1).todense()).ravel()
        chk["sentetik_test_max_cos_p99"] = round(float(np.quantile(m2, 0.99)), 3)
        chk["sentetik_test_cos_ge_0.8"] = int((m2 >= 0.8).sum())
    ok = all(v == 0 for k, v in chk.items() if k in (
        "id_kesisim_train_val", "id_kesisim_train_test", "id_kesisim_val_test", "unvan_kesisim_train_test",
        "unvan_kesisim_train_val", "sentetik_val_test_te", "triplet_anchor_train_disi", "val_triplet_anchor_val_disi",
        "triplette_cold_ilan", "legacy_pairs_train_disi_cv", "legacy_pairs_cold_ilan", "test_cv_cos_ge_0.8_trainde_ikizi",
        "sentetik_test_cos_ge_0.8", "unvan_kesisim_gercek_train_test"))
    chk["TUM_KONTROLLER_GECTI"] = bool(ok)
    return chk


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    lab = load_labels()
    syn = load_synthetic()
    print("CV ve ilanlar hazirlaniyor ...")
    cvs = build_cvs(lab, syn)
    jobs = make_job_pool(build_jobs(lab))
    cvs = make_split(cvs)
    print(cvs.groupby(["split", "source"]).size())

    print("Kademeli alaka matrisi ...")
    G = grade_matrix(cvs, jobs)
    qi, ji = np.nonzero(G)
    qrels = pd.DataFrame({"cv_id": cvs.cv_id.to_numpy()[qi].astype(np.int32),
                          "job_key": jobs.job_key.to_numpy(dtype=object)[ji],
                          "grade": G[qi, ji].astype(np.int8)})
    qrels = qrels.merge(cvs[["cv_id", "split"]], on="cv_id", how="left")
    per_q = pd.DataFrame({
        "cv_id": cvs.cv_id, "split": cvs.split, "source": cvs.source,
        "n_g3": (G == 3).sum(1), "n_g2": (G == 2).sum(1), "n_g1": (G == 1).sum(1),
        "n_rel_ge2": (G >= 2).sum(1),
        "n_rel_ge2_warm": ((G >= 2) & (jobs.job_pool == "warm").to_numpy()[None, :]).sum(1),
        "n_rel_ge2_cold": ((G >= 2) & (jobs.job_pool == "cold").to_numpy()[None, :]).sum(1),
    })

    print("Zor negatif madenciligi ...")
    trip, tstats = mine_triplets(cvs, jobs, G, "train", rng)
    vtrip, vstats = mine_triplets(cvs, jobs, G, "val", rng, max_pos=1)
    print(f"train triplet: {len(trip)} ({tstats}); val triplet: {len(vtrip)} ({vstats})")

    print("Eski yaklasim (Baseline-3) ciftleri ...")
    tr = pd.read_csv(os.path.join(ROOT, "veriseti", "train.csv"), encoding="utf-8-sig",
                     usecols=["cv_id", "job_id", "label", "match_type"])
    key = dict(zip(jobs.job_id, jobs.job_key))
    tr["job_key"] = tr.job_id.map(key)
    train_ids = set(cvs.cv_id[(cvs.split == "train") & (cvs.source == "real")])
    warm_keys = set(jobs.job_key[jobs.job_pool == "warm"])
    keep = tr.cv_id.isin(train_ids) & tr.job_key.isin(warm_keys)   # cold ilanlar Baseline-3 egitimine de girmez
    legacy = tr[keep][["cv_id", "job_key", "label", "match_type"]].reset_index(drop=True)
    legacy_dropped = int((~tr.cv_id.isin(train_ids)).sum())
    legacy_dropped_cold = int((tr.cv_id.isin(train_ids) & ~tr.job_key.isin(warm_keys)).sum())

    print("Sizinti kontrolleri ...")
    chk = leakage_checks(cvs, jobs, trip, vtrip, legacy)
    print(json.dumps(chk, ensure_ascii=False, indent=1))
    if not chk["TUM_KONTROLLER_GECTI"]:
        raise SystemExit("SIZINTI KONTROLU BASARISIZ")

    # ---- yazim
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    cvs.to_parquet(os.path.join(OUT, "clean_cvs.parquet"), index=False)
    jobs.to_parquet(os.path.join(OUT, "clean_jobs.parquet"), index=False)
    trip.to_parquet(os.path.join(OUT, "train_triplets.parquet"), index=False)
    vtrip.to_parquet(os.path.join(OUT, "val_triplets.parquet"), index=False)
    qcols = ["cv_id", "split", "source", "headline", "cv_struct"]
    for s in ("train", "val", "test"):
        cvs[cvs.split == s][qcols].rename(columns={"cv_struct": "query_text"}).to_parquet(
            os.path.join(OUT, f"{s}_queries.parquet"), index=False)
    qrels.to_parquet(os.path.join(OUT, "qrels_ground_truth.parquet"), index=False)
    legacy.to_parquet(os.path.join(OUT, "legacy_train_pairs.parquet"), index=False)
    shutil.copy(os.path.join(ROOT, "match_candidate_sota.py"), os.path.join(OUT, "match_candidate_sota.py"))

    # vitrin adaylari icin kontrol tablosu
    show = {}
    jt = jobs.reset_index(drop=True)
    for cid in SHOWCASE:
        i = int(np.where(cvs.cv_id.to_numpy() == cid)[0][0])
        g = G[i]
        top = np.argsort(-g, kind="stable")[:8]
        traps = jt.index[jt.title.str.contains("Sıva|Distribüt|Tuğla|CNC|Direktör|CTO", regex=True)].tolist()
        show[str(cid)] = {
            "headline": cvs.headline.iloc[i], "split": cvs.split.iloc[i], "gold_specialty": cvs.gold_specialty.iloc[i],
            "gold_isco4": cvs.gold_isco4.iloc[i], "gold_seniority": int(cvs.gold_seniority.iloc[i]),
            "n_grade3": int((g == 3).sum()), "n_grade2": int((g == 2).sum()),
            "ornek_alakali": [f"{jt.title[j]} -> {int(g[j])}" for j in top if g[j] > 0],
            "tuzak_ilan_dereceleri": {jt.title[j]: int(g[j]) for j in traps[:12]},
        }

    split_stats = {}
    for s in ("train", "val", "test"):
        d = per_q[per_q.split == s]
        split_stats[s] = {
            "cv": int(len(d)), "gercek": int((d.source == "real").sum()), "sentetik": int((d.source == "synthetic").sum()),
            "ort_grade3": round(float(d.n_g3.mean()), 2), "ort_grade2": round(float(d.n_g2.mean()), 2),
            "ort_grade1": round(float(d.n_g1.mean()), 2),
            "alakali(>=2)_ilani_olmayan_cv": int((d.n_rel_ge2 == 0).sum()),
            "grade3_ilani_olan_cv_orani": round(float((d.n_g3 > 0).mean()), 4),
            "alakali_sayisi_medyan": float(d.n_rel_ge2.median()),
            "alakali_sayisi_lt10_orani": round(float((d.n_rel_ge2 < 10).mean()), 4),
            "kategori_dagilimi": cvs[cvs.split == s].gold_industry.value_counts().to_dict(),
        }
    files = sorted(os.listdir(OUT))
    meta = {
        "olusturma": "scripts/05_build_dataset.py",
        "seed": SEED,
        "alaka_dereceleri": GRADE_RULES,
        "alakali_esigi_binary_metrikler": REL_THRESHOLD,
        "split": split_stats,
        "ilan_havuzu": jobs.job_pool.value_counts().to_dict(),
        "vitrin_adaylari_test": SHOWCASE,
        "triplet": {"train": int(len(trip)), "val": int(len(vtrip)), "train_istatistik": tstats, "val_istatistik": vstats,
                    "negatif_turleri": {
                        "neg_lexical": f"Ayni/benzer unvan sozcugu (karakter n-gram TF-IDF kosinus >= {LEX_MIN_SIM}), FARKLI sektor, grade 0 (Baklava Ustasi ~ Siva Ustasi); benzerlik neg_lexical_title_sim sutununda",
                        "neg_seniority_collar": "Ayni sektor veya ayni ISCO alt-ana grubu; yaka zit veya |kidem farki|>=3, grade<=1. Boyle ilan yoksa BM25 sirasi 4-10'dan grade 0 ilan konur (neg_seniority_collar_is_fallback=True)",
                        "neg_bm25": "BM25 ile en yuksek skorlu fakat grade 0 ilanlar (kelime kesisimi yuksek, anlamca uyumsuz)"},
                    "negatif_grade_dagilimi": {c: trip[c].value_counts().to_dict() for c in
                                               ("neg_lexical_grade", "neg_seniority_collar_grade", "neg_bm25_grade")}},
        "qrels": {"satir(grade>=1)": int(len(qrels)), "grade_dagilimi": qrels.grade.value_counts().to_dict()},
        "legacy_train_pairs": {"satir": int(len(legacy)), "cikarilan(val/test CV'leri)": legacy_dropped,
                               "cikarilan(cold ilanlar)": legacy_dropped_cold},
        "sizinti_kontrolleri": chk,
        "vitrin_kontrol": show,
        "sutun_rolleri": {
            "ozellik_kullanilabilir": ["headline", "cv_text", "cv_struct", "text_city", "text_seniority", "text_years_exp",
                                       "ilan: title, company_name, description, industry, position_level, mode_of_operation, job_type, city, job_struct"],
            "yalnizca_degerlendirme(gold_*)": sorted([c for c in cvs.columns if c.startswith("gold_")]),
            "yalnizca_sizinti_gosterimi(legacy_*)": sorted({c for c in list(cvs.columns) + list(jobs.columns) if c.startswith("legacy_")}),
            "not": f"Ayni ilanin bir negatif turunde kullanimi en fazla {NEG_USE_CAP} satirla sinirlidir (hub negatifleri).",
        },
        "dosyalar": {},
    }
    for f in files:
        meta["dosyalar"][f] = {"bayt": os.path.getsize(os.path.join(OUT, f)), "sha256": sha256(os.path.join(OUT, f))}
    write_json(meta, os.path.join(OUT, "metadata.json"))
    write_json(meta, os.path.join(REPORTS, "faz2_veri_ozeti.json"))
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(BUNDLE_README)
    if not args.no_zip:
        zp = ZIP_PATH
        if os.path.exists(zp):
            os.remove(zp)
        with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for f in sorted(os.listdir(OUT)):
                z.write(os.path.join(OUT, f), arcname=f"data_bundle/{f}")
        print(f"-> data_bundle.zip ({os.path.getsize(zp) / 1e6:.1f} MB)")
    print(json.dumps({k: meta[k] for k in ("split", "qrels", "triplet")}, ensure_ascii=False, indent=1, default=str)[:5000])


BUNDLE_README = """# data_bundle — sızıntısız CV ↔ İlan veri paketi

Bu klasör `scripts/05_build_dataset.py` tarafından üretilir ve Colab notebook'una
(`CV_Ilan_SOTA_Egitim_ve_Test_A100.ipynb`) tek dosya (`data_bundle.zip`) olarak yüklenir.

| Dosya | İçerik |
|---|---|
| clean_cvs.parquet | Tüm CV'ler (gerçek + yalnızca train'de sentetik). `cv_struct` = standart semantik şablon. `split` ∈ {train,val,test}. `gold_*` sütunları YALNIZCA değerlendirme içindir. |
| clean_jobs.parquet | 3.000 ilan. `job_struct`, `job_pool` ∈ {warm,cold}. `gold_*` sütunları YALNIZCA değerlendirme içindir. |
| train_triplets.parquet | anchor CV + pozitif ilan + 3 tür zor negatif (lexical / seniority-collar / BM25). |
| val_triplets.parquet | Doğrulama kaybı için aynı biçim (val CV'leri). |
| {train,val,test}_queries.parquet | Sorgu kimlikleri ve sorgu metni (gold yok). |
| qrels_ground_truth.parquet | (cv_id, job_key, grade∈{1,2,3}); listede olmayan çiftler grade 0. |
| legacy_train_pairs.parquet | Eski `train.csv`'nin yalnızca yeni train bölümündeki CV'lere ait satırları (Baseline-3). |
| metadata.json | Kurallar, istatistikler, sızıntı kontrolleri, SHA-256 özetleri. |
| match_candidate_sota.py | Eğitim ve çıkarımın ortak çekirdeği (özellikler, BM25, metrikler, çıkarım sınıfı). |

`legacy_*` sütunları eski sistemin alanlarıdır; CV tarafında `legacy_collar`/`legacy_city` altın etikete denktir —
YALNIZCA Baseline-3 sızıntı gösterimi içindir, özellik olarak kullanılamaz (`assert_no_gold_columns` bunları da reddeder).

Alaka dereceleri:
* 3 — aynı uzmanlık + yaka uyumu + |kıdem farkı| ≤ 1 + lokasyon uyumu (aynı şehir / Remote)
* 2 — aynı ISCO-08 birim grubu + yaka uyumu + |kıdem farkı| ≤ 2
* 1 — yaka uyumu + (aynı ana sektör veya aynı ISCO grubu ama kıdem farkı ≥ 3)
* 0 — diğer
"""

if __name__ == "__main__":
    main()
