# -*- coding: utf-8 -*-
"""Metrik tavanlari (oracle): mevcut qrels tanimiyla HERHANGI bir siralayicinin ulasabilecegi ust sinirlar.

Hicbir model egitilmez veya secilmez; yalnizca etiketlerin yapisi olculur. Test tavanlari raporlama icindir:
hedeflerin (Top-1 > 88, NDCG@10 > 92, P@10 > 85) mevcut tanimla ulasilabilir olup olmadigini gosterir.

Olculenler
  * kahin (ilanlar gercek dereceye gore) — val/test x tam/warm/cold havuz x alakali esigi 1/2/3
  * "ikili kusursuz": tum grade>=2 ilanlar basta ama kendi aralarinda rastgele (grade 3'u 2'den ayirmayan model)
  * BM25 ilk-K aday kumesinin kahin yeniden siralamasi (Asama 2'nin aday kalitesine bagli tavani), K = 50/100/200;
    tam havuz ve WARM havuz (cold ilanlar ne adayda ne qrels'te — notebook §9.1 val tavaniyla ayni olcek)
  * sorgu basina alakali ilan sayisi dagilimi, grade-3'un lokasyona bagimliligi

Cikti: reports/metric_ceilings.json
Kullanim: python scripts/12_metric_ceilings.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, write_json  # noqa: E402

sys.path.insert(0, ROOT)
import match_candidate_sota as core  # noqa: E402

D = os.path.join(ROOT, "data_bundle")
KEYS = ("NDCG@10", "MRR@10", "P@10", "P@10_capped", "Top1", "Top1_exact(grade3)", "R@10", "R@50", "R@100")


def summ(res):
    s = res["summary"]
    out = {k: round(100 * s[k], 2) for k in KEYS if k in s}
    out["n_queries"] = s["n_queries"]
    out["n_skipped_no_relevant"] = s["n_skipped_no_relevant"]
    return out


def main():
    cvs = pd.read_parquet(os.path.join(D, "clean_cvs.parquet"))
    jobs = pd.read_parquet(os.path.join(D, "clean_jobs.parquet")).reset_index(drop=True)
    q = pd.read_parquet(os.path.join(D, "qrels_ground_truth.parquet"))
    jidx = {k: i for i, k in enumerate(jobs.job_key)}
    Q = {}
    for c, j, g in zip(q.cv_id, q.job_key.map(jidx), q.grade):
        Q.setdefault(int(c), {})[int(j)] = int(g)
    pools = {"tam": None, "warm": set(np.where(jobs.job_pool == "warm")[0]), "cold": set(np.where(jobs.job_pool == "cold")[0])}
    ids = {s: cvs.cv_id[cvs.split == s].astype(int).tolist() for s in ("val", "test")}
    out = {"aciklama": "Kahin = ilanlari gercek dereceye gore siralar. Hicbir model secimi yapilmaz; test degerleri yalnizca "
                       "hedeflerin mevcut tanimla ulasilabilirligini raporlamak icindir.",
           "alakali_esigi_notebook": 2}

    # 1) kahin tavanlari
    orc = {}
    for s, qids in ids.items():
        for pn, pool in pools.items():
            qr = {c: {j: g for j, g in Q.get(c, {}).items() if pool is None or j in pool} for c in qids}
            ranked = {c: [j for j, _ in sorted(qr[c].items(), key=lambda x: -x[1])][:100] for c in qids}
            for thr in (1, 2, 3):
                orc[f"{s}|{pn}|esik{thr}"] = summ(core.evaluate_rankings(ranked, qr, rel_threshold=thr))
    out["kahin"] = orc

    # 2) ikili kusursuz: grade>=2 hepsi basta, kendi aralarinda rastgele (5 tohum ortalamasi)
    ik = {}
    for s, qids in ids.items():
        vals = []
        for seed in range(5):
            r = np.random.default_rng(seed)
            ranked = {}
            for c in qids:
                rel = [j for j, g in Q.get(c, {}).items() if g >= 2]
                rest = [j for j, g in Q.get(c, {}).items() if g < 2]
                ranked[c] = (list(r.permutation(rel)) + sorted(rest, key=lambda j: -Q[c][j]))[:100]
            vals.append(core.evaluate_rankings(ranked, {c: Q.get(c, {}) for c in qids})["summary"])
        ik[s] = {k: round(100 * float(np.mean([v[k] for v in vals])), 2) for k in ("NDCG@10", "P@10", "Top1")}
    out["ikili_kusursuz(grade>=2_basta_kendi_icinde_rastgele__kalan_1>0_dogru)"] = ik

    # 3) BM25 ilk-K aday kumesinin kahin yeniden siralamasi
    bm = core.SparseBM25([core.tokenize(t) for t in jobs.job_struct])
    cand = {}
    for s, qids in ids.items():
        d = cvs.set_index("cv_id").loc[qids]
        S = bm.scores([core.tokenize(t) for t in d.cv_struct])
        top = core.topk_indices(S, 300)
        qr = {c: Q.get(c, {}) for c in qids}
        cand[f"{s}|bm25_dogrudan"] = summ(core.evaluate_rankings({c: top[i][:100].tolist() for i, c in enumerate(qids)}, qr))
        for K in (50, 100, 200):
            ranked = {}
            for i, c in enumerate(qids):
                head = sorted(top[i][:K].tolist(), key=lambda j: -Q.get(c, {}).get(j, 0))
                ranked[c] = (head + top[i][K:].tolist())[:100]
            cand[f"{s}|bm25_ilk{K}_kahin_yeniden_siralama"] = summ(core.evaluate_rankings(ranked, qr))
        # WARM havuz (cold ilanlar ne adayda ne qrels'te): notebook §9.1 val tavanı ve scripts/13 ile ayni olcek
        Sw = S.copy(); Sw[:, (jobs.job_pool == "cold").to_numpy()] = -np.inf
        topw = core.topk_indices(Sw, 300)
        qrw = {c: {j: g for j, g in Q.get(c, {}).items() if j in pools["warm"]} for c in qids}
        for K in (50, 100, 200):
            ranked = {}
            for i, c in enumerate(qids):
                head = sorted(topw[i][:K].tolist(), key=lambda j: -qrw[c].get(j, 0))
                ranked[c] = (head + topw[i][K:].tolist())[:100]
            cand[f"{s}|WARM|bm25_ilk{K}_kahin_yeniden_siralama"] = summ(core.evaluate_rankings(ranked, qrw))
    out["aday_kumesi_tavani(BM25)"] = cand

    # 4) sorgu basina alakali ilan dagilimi
    dist = {}
    cvi = cvs.set_index("cv_id")
    for s, qids in ids.items():
        nrel = np.array([sum(1 for g in Q.get(c, {}).values() if g >= 2) for c in qids])
        n3 = np.array([sum(1 for g in Q.get(c, {}).values() if g == 3) for c in qids])
        col = cvi.loc[qids, "gold_collar"].to_numpy()
        dist[s] = {"n_cv": len(qids), "alakali(>=2)_medyan": float(np.median(nrel)),
                   "alakali<10_orani_%": round(100 * float(np.mean(nrel < 10)), 2),
                   "alakali_yok": int((nrel == 0).sum()), "grade3_yok_orani_%": round(100 * float(np.mean(n3 == 0)), 2),
                   "yaka": {k: {"n": int((col == k).sum()), "alakali_medyan": float(np.median(nrel[col == k])),
                                "alakali<10_orani_%": round(100 * float(np.mean(nrel[col == k] < 10)), 2)}
                            for k in ("BLUE", "WHITE")}}
    out["alakali_dagilimi"] = dist

    # 5) grade-3'un lokasyona bagimliligi: ayni occ_key + yaka + |kidem|<=1 olup yalniz lokasyon yuzunden grade 2 kalanlar
    jo = jobs[["gold_occ_key", "gold_collar", "gold_seniority"]].to_numpy()
    loc = {}
    for s, qids in ids.items():
        same, g3 = 0, 0
        for c in qids:
            r = cvi.loc[c]
            m = (jo[:, 0] == r.gold_occ_key) & (jo[:, 1] == r.gold_collar) & (np.abs(jo[:, 2].astype(float) - float(r.gold_seniority)) <= 1)
            js = np.where(m)[0]
            same += len(js)
            g3 += sum(1 for j in js if Q.get(c, {}).get(int(j), 0) == 3)
        loc[s] = {"ayni_uzmanlik_yaka_kidem_cifti": same, "bunlardan_grade3": g3,
                  "yalniz_lokasyon_yuzunden_grade2_%": round(100 * (same - g3) / max(1, same), 2)}
    out["grade3_lokasyon_bagimliligi"] = loc
    write_json(out, os.path.join(ROOT, "reports", "metric_ceilings.json"))
    t = orc["test|tam|esik2"]
    print(f"test kahin: NDCG@10 {t['NDCG@10']} | P@10 {t['P@10']} | P@10_capped {t['P@10_capped']} | Top1 {t['Top1']}")
    print("ikili kusursuz:", ik)
    print("BM25 ilk-50 kahin (test):", cand["test|bm25_ilk50_kahin_yeniden_siralama"]["NDCG@10"])


if __name__ == "__main__":
    main()
