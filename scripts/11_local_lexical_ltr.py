# -*- coding: utf-8 -*-
"""GPU gerektirmeyen GERCEK yerel olcum: BM25 aday uretimi + LambdaMART (yalnizca leksikal / metin-turevli
oznitelikler; dense ve cross-encoder oznitelikleri 0). Notebook'taki LTR protokolunun aynisi:
  * oznitelik siniflandiricilari yalnizca train CV'leriyle egitilir,
  * LambdaMART val sorgularinda (cold ilanlar haric) egitilir; erken durdurma val icindeki ayri bir bolmede,
  * test bir kez degerlendirilir; %95 bootstrap GA + Fisher randomizasyon testi (BM25'e karsi).
Amac: elle carpan yerine ogrenilen siralamanin katkisini, noral modellerden bagimsiz ve sizintisiz olcmek.

Cikti: reports/local_lexical_ltr.json
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, write_json  # noqa: E402

sys.path.insert(0, ROOT)
import match_candidate_sota as core  # noqa: E402

D = os.path.join(ROOT, "data_bundle")
K = 50
SEED = 42


def main():
    cvs = pd.read_parquet(os.path.join(D, "clean_cvs.parquet"))
    jobs = pd.read_parquet(os.path.join(D, "clean_jobs.parquet")).reset_index(drop=True)
    q = pd.read_parquet(os.path.join(D, "qrels_ground_truth.parquet"))
    jidx = {k: i for i, k in enumerate(jobs.job_key)}
    qrels = {}
    for c, j, g in zip(q.cv_id, q.job_key.map(jidx), q.grade):
        qrels.setdefault(int(c), {})[int(j)] = int(g)
    cold = (jobs.job_pool == "cold").to_numpy()
    tr = cvs[cvs.split == "train"]
    attr = core.AttributeModels().fit(tr.cv_struct.tolist(), tr.gold_industry.tolist(), tr.headline.tolist(),
                                      (tr.gold_collar == "BLUE").astype(int).tolist())
    tsim = core.TitleSimilarity().fit(list(jobs.title) + tr.headline.tolist())
    jt = core.JobTable.build(jobs, attr, tsim)
    bm = core.SparseBM25([core.tokenize(t) for t in jobs.job_struct])

    def frame(split):
        d = cvs[cvs.split == split].reset_index(drop=True)
        S = bm.scores([core.tokenize(t) for t in d.cv_struct])
        br = core.ranks_from_scores(S)
        rrf = core.rrf_scores([br], k=60)
        cand = core.topk_indices(S, K)
        parts = []
        for i, r in d.iterrows():
            qm = core.QueryMeta.from_text(r.headline, r.cv_text, attr, r.cv_struct)
            c = cand[i]
            f = core.build_pair_features(qm, c, jt, tsim, np.zeros(K), S[i, c], rrf[i, c], np.ones(K), br[i, c], br[i, c], None)
            core.assert_no_gold_columns(f, "local-ltr")
            f.insert(0, "job_idx", c)
            f.insert(0, "qid", int(r.cv_id))
            f["grade"] = [qrels.get(int(r.cv_id), {}).get(int(j), 0) for j in c]
            parts.append(f)
        return pd.concat(parts, ignore_index=True), d, S

    val, _, _ = frame("val")
    test, dtest, Stest = frame("test")
    feats = core.FEATURE_COLUMNS
    v = val[~cold[val.job_idx.to_numpy()]]
    rng = np.random.default_rng(SEED)
    qs = v.qid.unique()
    es_q = set(rng.choice(qs, size=int(0.15 * len(qs)), replace=False).tolist())
    a = v[~v.qid.isin(es_q)].sort_values("qid", kind="stable")
    b = v[v.qid.isin(es_q)].sort_values("qid", kind="stable")
    g = lambda df: df.groupby("qid", sort=False).size().to_numpy()
    m = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", eval_at=[10], n_estimators=2000, learning_rate=0.05,
                       num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       random_state=SEED, verbose=-1)
    m.fit(a[feats], a.grade, group=g(a), eval_set=[(b[feats], b.grade)], eval_group=[g(b)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    test["s"] = m.predict(test[feats].to_numpy(np.float32))
    ranked_ltr, ranked_bm = {}, {}
    top = core.topk_indices(Stest, 100)
    for i, cid in enumerate(dtest.cv_id):
        gq = test[test.qid == int(cid)].sort_values("s", ascending=False)
        head = gq.job_idx.astype(int).tolist()
        hs = set(head)
        ranked_ltr[int(cid)] = (head + [int(j) for j in top[i] if j not in hs])[:100]
        ranked_bm[int(cid)] = top[i].tolist()
    qr = {int(c): qrels.get(int(c), {}) for c in dtest.cv_id}
    r_bm = core.evaluate_rankings(ranked_bm, qr)
    r_ltr = core.evaluate_rankings(ranked_ltr, qr)
    a_ = r_ltr["per_query"].set_index("qid")["NDCG@10"]
    b_ = r_bm["per_query"].set_index("qid")["NDCG@10"]
    p = core.paired_randomization_test(a_.loc[b_.index].to_numpy(), b_.to_numpy(), n_perm=10000)
    out = {}
    for name, r in (("BM25", r_bm), ("BM25 + LambdaMART (leksikal, noral yok)", r_ltr)):
        s = r["summary"]
        lo, hi = core.bootstrap_ci(r["per_query"]["NDCG@10"].to_numpy())
        out[name] = {k: round(100 * v, 2) for k, v in s.items() if isinstance(v, float)}
        out[name]["NDCG@10_CI95"] = [round(100 * lo, 1), round(100 * hi, 1)]
        out[name]["n_queries"] = s["n_queries"]
    out["p_NDCG@10_LTR_vs_BM25"] = p
    out["best_iteration"] = int(m.best_iteration_ or 0)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    out["en_onemli_oznitelikler(gain)"] = {k: round(float(v), 1) for k, v in imp.head(10).items()}
    write_json(out, os.path.join(ROOT, "reports", "local_lexical_ltr.json"))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
