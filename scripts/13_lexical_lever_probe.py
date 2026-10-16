# -*- coding: utf-8 -*-
"""Iyilestirme kaldiraclarinin GPU'suz, YALNIZ VAL uzerinde capraz dogrulamali olcumu (test kullanilmaz).

Protokol (scripts/11 ile ayni aday/oznitelik hatti; dense ve cross-encoder oznitelikleri 0):
  * Aday: BM25 ilk-K (K = 50 / 100). Oznitelik siniflandiricilari, ISCO-4 siniflandiricisi ve kNN etiket aktarimi
    yalnizca TRAIN CV'lerinden ogrenilir; cold ilanlar kNN'de 0'lanir.
  * LambdaMART, val sorgularinda 5 katli GroupKFold (grup = CV'nin bolme grubu, group_id) ile egitilir; her katta
    erken durdurma egitim katlarinin icindeki ayri %15'lik bolmede yapilir; tutulan katin adaylari yeniden siralanir.
  * Tum val sorgularinin kat-disi siralamalari WARM havuzda olculur (cold ilanlar ne siralamada ne qrels'te; notebook'un
    secim degerlendiricileriyle ayni olcek); varyantlar "base, K=50"ye karsi eslestirilmis Fisher randomizasyon testiyle
    karsilastirilir. Aday kumesi BM25 ilk-K'dir (cold adaylar elenir, yani WARM aday sayisi K'dan az olabilir).

Varyantlar
  base       : core.FEATURE_COLUMNS (35)
  +isco      : ISCO-4 olasilik vektorlerinin ic carpimi (CV x ilan) ve ilk-1 eslesmesi (LinearSVC, sicaklik 0.1)
  +knn       : en yakin 30 train CV'sinin bu ilana verdigi derecelerin benzerlik agirlikli ortalamasi
               (cold ilanlara hic sinyal vermez: yeni ilanlara karsi yanlilik, urun karari gerektirir)
  +knn+isco  : ikisi birden

Bu sayilar sinir aglari OLMADAN olculur; dense ve CE oznitelikleri eklenince kazanclar kuculebilir.
Cikti: reports/lexical_lever_probe_val.json
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, write_json  # noqa: E402

sys.path.insert(0, ROOT)
import match_candidate_sota as core  # noqa: E402

D = os.path.join(ROOT, "data_bundle")
SEED = 42
KMAX = 100
KNN = 30


class SoftSVC:
    """LinearSVC karar degerlerinden sicaklikli softmax olasiliklari."""

    def __init__(self, m, T=0.1):
        self.m, self.T, self.classes_ = m, T, m.classes_

    def predict_proba(self, X):
        z = self.m.decision_function(X) / self.T
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)


def main():
    t0 = time.time()
    cvs = pd.read_parquet(os.path.join(D, "clean_cvs.parquet"))
    jobs = pd.read_parquet(os.path.join(D, "clean_jobs.parquet")).reset_index(drop=True)
    q = pd.read_parquet(os.path.join(D, "qrels_ground_truth.parquet"))
    jidx = {k: i for i, k in enumerate(jobs.job_key)}
    qrels = {}
    for c, j, g in zip(q.cv_id, q.job_key.map(jidx), q.grade):
        qrels.setdefault(int(c), {})[int(j)] = int(g)
    N = len(jobs)
    cold = (jobs.job_pool == "cold").to_numpy()
    tr = cvs[cvs.split == "train"].reset_index(drop=True)
    attr = core.AttributeModels().fit(tr.cv_struct.tolist(), tr.gold_industry.tolist(), tr.headline.tolist(),
                                      (tr.gold_collar == "BLUE").astype(int).tolist())
    tsim = core.TitleSimilarity().fit(list(jobs.title) + tr.headline.tolist())
    jt = core.JobTable.build(jobs, attr, tsim)
    bm = core.SparseBM25([core.tokenize(t) for t in jobs.job_struct])

    # kNN etiket aktarimi (train qrels, cold sutunlar 0)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, preprocessor=core.tr_ascii).fit(tr.cv_struct)
    Xtr = vec.transform(tr.cv_struct)
    G = np.zeros((len(tr), N), dtype=np.float32)
    for i, c in enumerate(tr.cv_id):
        for j, g in qrels.get(int(c), {}).items():
            G[i, j] = g
    G[:, cold] = 0.0
    GAIN, REL, G3 = (2 ** G - 1) / 7.0, (G >= 2).astype(np.float32), (G == 3).astype(np.float32)

    def knn_feats(texts):
        S = (vec.transform(texts) @ Xtr.T).toarray()
        top = np.argsort(-S, axis=1)[:, :KNN]
        W = np.take_along_axis(S, top, axis=1) ** 2
        W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
        return tuple(np.stack([W[i] @ M[top[i]] for i in range(len(texts))]) for M in (GAIN, REL, G3))

    # ISCO-4 siniflandiricisi (train CV metni -> gold_isco4); ilan tarafina ilan metniyle uygulanir (ilan altin etiketi yok)
    ivec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True, preprocessor=core.tr_ascii)
    iclf = SoftSVC(LinearSVC(C=1.0).fit(ivec.fit_transform(tr.cv_struct), tr.gold_isco4.astype(str)))
    P_job = iclf.predict_proba(ivec.transform(jobs.job_struct))

    d = cvs[cvs.split == "val"].reset_index(drop=True)
    S = bm.scores([core.tokenize(t) for t in d.cv_struct])
    br = core.ranks_from_scores(S)
    rrf = core.rrf_scores([br], k=60)
    cand = core.topk_indices(S, KMAX)
    kg, kr, k3 = knn_feats(d.cv_struct.tolist())
    Pq = iclf.predict_proba(ivec.transform(d.cv_struct))
    isco_acc = float((iclf.classes_[Pq.argmax(1)] == d.gold_isco4.astype(str).to_numpy()).mean())
    parts = []
    for i, r in d.iterrows():
        qm = core.QueryMeta.from_text(r.headline, r.cv_text, attr, r.cv_struct)
        c = cand[i]
        f = core.build_pair_features(qm, c, jt, tsim, np.zeros(KMAX), S[i, c], rrf[i, c], np.ones(KMAX), br[i, c], br[i, c], None)
        core.assert_no_gold_columns(f, "lever-probe")
        f["knn_gain"], f["knn_rel"], f["knn_g3"] = kg[i, c], kr[i, c], k3[i, c]
        f["isco_dot"] = P_job[c] @ Pq[i]
        f["isco_top1_match"] = (P_job[c].argmax(1) == Pq[i].argmax()).astype(float)
        f.insert(0, "pos", np.arange(KMAX))
        f.insert(0, "job_idx", c)
        f.insert(0, "qid", int(r.cv_id))
        f["grade"] = [qrels.get(int(r.cv_id), {}).get(int(j), 0) for j in c]
        parts.append(f)
    val = pd.concat(parts, ignore_index=True)
    grp = dict(zip(d.cv_id.astype(int), d.group_id.astype(int)))
    print("hazirlik", round(time.time() - t0, 1), "sn", flush=True)

    base = core.FEATURE_COLUMNS
    variants = {"base": base, "+isco": base + ["isco_dot", "isco_top1_match"], "+knn": base + ["knn_gain", "knn_rel", "knn_g3"],
                "+knn+isco": base + ["knn_gain", "knn_rel", "knn_g3", "isco_dot", "isco_top1_match"]}
    qids = d.cv_id.astype(int).to_numpy()
    folds = list(GroupKFold(n_splits=5).split(qids, groups=[grp[x] for x in qids]))
    # Karar sinyali: cold ilanlar ne siralamada ne qrels'te (notebook'un secim degerlendiricileriyle ayni WARM havuz)
    qr = {int(c): {j: g for j, g in qrels.get(int(c), {}).items() if not cold[j]} for c in qids}
    gsize = lambda df: df.groupby("qid", sort=False).size().to_numpy()  # noqa: E731
    out = {"protokol": "val-yalniz, 5 katli GroupKFold (group_id), kat-disi siralama, WARM havuz (cold yok); test KULLANILMADI",
           "isco_siniflandirici_val_top1_dogrulugu": round(isco_acc, 4), "varyantlar": {}}
    ref = None
    for K in (50, 100):
        for name, feats in variants.items():
            ranked = {}
            for k_, (tr_i, te_i) in enumerate(folds):
                trq, teq = set(qids[tr_i]), set(qids[te_i])
                v = val[(val.pos < K) & val.qid.isin(trq) & ~cold[val.job_idx.to_numpy()]]
                rng = np.random.default_rng(SEED + k_)
                uq = v.qid.unique()
                es_q = set(rng.choice(uq, size=max(1, int(0.15 * len(uq))), replace=False).tolist())
                a = v[~v.qid.isin(es_q)].sort_values("qid", kind="stable")
                b = v[v.qid.isin(es_q)].sort_values("qid", kind="stable")
                m = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", eval_at=[10], n_estimators=2000, learning_rate=0.05,
                                   num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   random_state=SEED, verbose=-1)
                m.fit(a[feats], a.grade, group=gsize(a), eval_set=[(b[feats], b.grade)], eval_group=[gsize(b)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
                tt = val[(val.pos < K) & val.qid.isin(teq) & ~cold[val.job_idx.to_numpy()]].copy()
                tt["s"] = m.predict(tt[feats].to_numpy(np.float32))
                for qid_, gq in tt.groupby("qid", sort=False):
                    i = int(np.where(qids == qid_)[0][0])
                    head = gq.sort_values("s", ascending=False).job_idx.astype(int).tolist()
                    hs = set(head)
                    ranked[int(qid_)] = (head + [int(j) for j in cand[i] if j not in hs and not cold[j]])[:100]
            for i, q_ in enumerate(qids):          # tum adaylari cold olan sorgu: BM25 (WARM) sirasi
                ranked.setdefault(int(q_), [int(j) for j in cand[i] if not cold[j]])
            r = core.evaluate_rankings(ranked, qr)
            s = r["summary"]
            pq = r["per_query"].set_index("qid")["NDCG@10"]
            lo, hi = core.bootstrap_ci(pq.to_numpy())
            o = {k: round(100 * s[k], 2) for k in ("NDCG@10", "MRR@10", "P@10", "P@10_capped", "Top1", "Top1_exact(grade3)", "R@50")}
            o["NDCG@10_GA95"] = [round(100 * lo, 1), round(100 * hi, 1)]
            o["n_queries"] = s["n_queries"]
            if ref is None:
                ref = (r["per_query"].set_index("qid"))
            else:
                o["p_NDCG@10_vs_base_K50"] = core.paired_randomization_test(pq.loc[ref.index].to_numpy(), ref["NDCG@10"].to_numpy(), n_perm=5000)
                o["p_Top1_vs_base_K50"] = core.paired_randomization_test(r["per_query"].set_index("qid").loc[ref.index, "Top1"].to_numpy(),
                                                                        ref["Top1"].to_numpy(), n_perm=5000)
            out["varyantlar"][f"K{K}|{name}"] = o
            print(f"K={K} {name}: {o}", flush=True)
    out["sure_sn"] = round(time.time() - t0, 1)
    write_json(out, os.path.join(ROOT, "reports", "lexical_lever_probe_val.json"))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
