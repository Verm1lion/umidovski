# -*- coding: utf-8 -*-
"""Kural-tabanli kademeli qrels'in BAGIMSIZ cift-duzeyi yargilarla dogrulanmasi.

Kurallarla turetilen dereceleri (grade 0-3) bilmeyen anotatorlar, tabakali bir orneklemdeki
(CV, ilan) ciftlerini yalnizca metinleri okuyarak 0-3 olceginde yargilar. Ardindan kural derecesi ile
bagimsiz yargi arasindaki uyum (ikinci dereceden agirlikli Cohen kappa, birebir uyum, karisiklik
matrisi) raporlanir.

Alt komutlar
  sample  data_bundle -> annotation/qrels_validation/pairs_###.json (+ gizli anahtar _key.json)
  score   annotation/qrels_validation/judged_###.json -> reports/qrels_validation.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, write_json  # noqa: E402

D = os.path.join(ROOT, "data_bundle")
OUT = os.path.join(ROOT, "annotation", "qrels_validation")
PER_GRADE = 90
BATCH = 40
SEED = 777

RUBRIC = """Derece olcegi (CV'deki aday bu ilan icin ne kadar uygun?):
3 = Tam eslesme: ayni meslek/uzmanlik (ayni is), yaka uyumlu, kidem en fazla 1 kademe farkli, lokasyon uyumlu (ayni sehir ya da ilan uzaktan/Remote).
2 = Yuksek alaka: ayni meslek grubu / cok yakin rol, yaka uyumlu, kidem en fazla 2 kademe farkli; lokasyon farkli olabilir.
1 = Kismi alaka: ayni ana sektor ve yaka ama farkli rol; YA DA ayni rol ama kidem cok uyumsuz (>=3 kademe, or. stajyer vs direktor).
0 = Alakasiz: farkli yaka veya farkli sektor ve farkli meslek.
Kidem kademeleri: 0 stajyer/ogrenci, 1 giris/junior/yardimci, 2 uzman/orta, 3 kidemli/usta, 4 lider/sef/mudur, 5 direktor/ust yonetim."""


def sample():
    cvs = pd.read_parquet(os.path.join(D, "clean_cvs.parquet"))
    jobs = pd.read_parquet(os.path.join(D, "clean_jobs.parquet"))
    q = pd.read_parquet(os.path.join(D, "qrels_ground_truth.parquet"))
    rng = np.random.default_rng(SEED)
    real_test_val = cvs[(cvs.source == "real") & cvs.split.isin(["val", "test"])]
    ids = set(real_test_val.cv_id)
    q = q[q.cv_id.isin(ids)]
    rows = []
    for g in (3, 2, 1):
        d = q[q.grade == g]
        take = d.iloc[rng.choice(len(d), size=min(PER_GRADE, len(d)), replace=False)]
        rows += [(int(r.cv_id), r.job_key, g) for r in take.itertuples()]
    # grade 0: yarisi rastgele, yarisi BM25-benzeri zor (ayni unvan sozcugu) ciftler
    keys = jobs.job_key.to_numpy(dtype=object)
    have = set((c, j) for c, j, _ in rows) | set(zip(q.cv_id, q.job_key))
    cv_list = sorted(ids)
    while sum(1 for r in rows if r[2] == 0) < PER_GRADE:
        c = int(rng.choice(cv_list))
        j = keys[rng.integers(len(keys))]
        if (c, j) not in have:
            rows.append((c, j, 0))
            have.add((c, j))
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "pairs_*.json")):   # judged_*.json (yargilar) korunur
        os.remove(f)
    C = cvs.set_index("cv_id")
    J = jobs.set_index("job_key")
    key = {}
    for b in range(0, len(rows), BATCH):
        payload = []
        for k, (c, j, g) in enumerate(rows[b:b + BATCH]):
            pid = f"P{b + k:04d}"
            key[pid] = {"cv_id": c, "job_key": j, "rule_grade": g}
            payload.append({"id": pid, "cv": C.loc[c, "cv_struct"], "ilan": J.loc[j, "job_struct"][:1600]})
        with open(os.path.join(OUT, f"pairs_{b // BATCH:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=0)
    with open(os.path.join(ROOT, "annotation", "qrels_validation_key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False)
    with open(os.path.join(OUT, "RUBRIC.txt"), "w", encoding="utf-8") as f:
        f.write(RUBRIC)
    print(f"{len(rows)} cift, {len(range(0, len(rows), BATCH))} parti")


def score():
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    key = json.load(open(os.path.join(ROOT, "annotation", "qrels_validation_key.json"), encoding="utf-8"))
    judged = {}
    for p in sorted(glob.glob(os.path.join(OUT, "judged_*.json"))):
        for r in json.load(open(p, encoding="utf-8")):
            judged[r["id"]] = int(r["grade"])
    ids = [i for i in key if i in judged]
    y_rule = np.array([key[i]["rule_grade"] for i in ids])
    y_judge = np.array([judged[i] for i in ids])
    rep = {
        "n": len(ids),
        "quadratic_weighted_kappa": round(float(cohen_kappa_score(y_rule, y_judge, weights="quadratic")), 4),
        "linear_weighted_kappa": round(float(cohen_kappa_score(y_rule, y_judge, weights="linear")), 4),
        "birebir_uyum": round(float((y_rule == y_judge).mean()), 4),
        "fark_le1": round(float((np.abs(y_rule - y_judge) <= 1).mean()), 4),
        "ikili_uyum_(>=2_alakali)": round(float(((y_rule >= 2) == (y_judge >= 2)).mean()), 4),
        "ikili_kappa_(>=2)": round(float(cohen_kappa_score(y_rule >= 2, y_judge >= 2)), 4),
        "karisiklik_matrisi_satir=kural_sutun=yargi": confusion_matrix(y_rule, y_judge, labels=[0, 1, 2, 3]).tolist(),
        "spearman": round(float(pd.Series(y_rule).corr(pd.Series(y_judge), method="spearman")), 4),
    }
    write_json(rep, os.path.join(ROOT, "reports", "qrels_validation.json"))
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1]]()
