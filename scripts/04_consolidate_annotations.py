# -*- coding: utf-8 -*-
"""Meslek anotasyonlarini birlestirir, anotatorlar arasi uyumu (IAA) olcer ve
konsolidasyon girdilerini / nihai etiket tablosunu uretir.

Alt komutlar:
  prepare   annotation/raw/*.json -> annotation/merged_raw.parquet
            IAA (annotation/iaa_raw) -> reports/annotation_iaa.json
            ISCO 2-haneli alt ana gruplara gore konsolidasyon girdileri ->
            annotation/consolidation_inputs/group_XX.json
  finalize  annotation/consolidation_raw/group_XX.json (+ istege bagli
            annotation/adjudication.json) -> annotation/occupation_labels.parquet

Kullanim:
  python scripts/04_consolidate_annotations.py prepare
  python scripts/04_consolidate_annotations.py finalize
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, tr_ascii, write_json  # noqa: E402

ANN = os.path.join(ROOT, "annotation")
REPORTS = os.path.join(ROOT, "reports")


def norm_spec(s):
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s


def spec_key(s):
    return re.sub(r"[^a-z0-9]+", " ", tr_ascii(s)).strip()


def load_dir(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(p, encoding="utf-8") as f:
            for r in json.load(f):
                r["_src"] = os.path.basename(p)
                rows.append(r)
    return pd.DataFrame(rows)


def cohen_kappa(a, b, weights=None):
    from sklearn.metrics import cohen_kappa_score
    return round(float(cohen_kappa_score(a, b, weights=weights)), 4)


def iaa_report(main, iaa, items):
    m = main.set_index("id")
    rep = {"n": int(len(iaa))}
    j = iaa.join(m, on="id", rsuffix="_main", how="inner")
    rep["n_eslesen"] = int(len(j))
    rep["isco4_kappa"] = cohen_kappa(j.isco4, j.isco4_main)
    rep["isco4_birebir_uyum"] = round(float((j.isco4 == j.isco4_main).mean()), 4)
    rep["isco3_kappa"] = cohen_kappa(j.isco4.str[:3], j.isco4_main.str[:3])
    rep["isco2_kappa"] = cohen_kappa(j.isco4.str[:2], j.isco4_main.str[:2])
    rep["isco1_kappa"] = cohen_kappa(j.isco4.str[:1], j.isco4_main.str[:1])
    rep["collar_kappa"] = cohen_kappa(j.collar, j.collar_main)
    rep["collar_uyum"] = round(float((j.collar == j.collar_main).mean()), 4)
    rep["seniority_quadratic_kappa"] = cohen_kappa(j.seniority, j.seniority_main, weights="quadratic")
    rep["seniority_birebir"] = round(float((j.seniority == j.seniority_main).mean()), 4)
    rep["seniority_fark_le1"] = round(float(((j.seniority - j.seniority_main).abs() <= 1).mean()), 4)
    rep["specialty_birebir_(konsolidasyon_oncesi)"] = round(float((j.specialty.map(spec_key) == j.specialty_main.map(spec_key)).mean()), 4)
    # dataset collar (CV) ile anotator collar uyumu
    cv = items[items.kind == "CV"][["id", "dataset_collar"]]
    k = cv.join(m, on="id", how="inner")
    k = k[k.dataset_collar.notna()]
    rep["cv_dataset_collar_vs_anotator_kappa"] = cohen_kappa(k.dataset_collar, k.collar)
    rep["cv_dataset_collar_vs_anotator_uyum"] = round(float((k.dataset_collar == k.collar).mean()), 4)
    rep["cv_dataset_collar_uyusmayan_ornekler"] = k[k.dataset_collar != k.collar].id.head(15).tolist()
    return rep


def load_items():
    rows = [json.loads(l) for l in open(os.path.join(ANN, "items.jsonl"), encoding="utf-8")]
    df = pd.DataFrame(rows)[["id", "kind", "title", "industry"]]
    cv = pd.read_csv(os.path.join(ROOT, "veriseti", "tum_cvler.csv"), encoding="utf-8-sig")
    cmap = dict(zip("C" + cv.cv_id.astype(str), cv.collar.map({"BEYAZ": "WHITE", "MAVI": "BLUE"})))
    df["dataset_collar"] = df.id.map(cmap)
    return df


def prepare():
    items = load_items()
    main = load_dir(os.path.join(ANN, "raw"))
    missing = sorted(set(items.id) - set(main.id))
    dup = main.id[main.id.duplicated()].tolist()
    print(f"ana anotasyon: {len(main)} kayit, eksik {len(missing)}, tekrar {len(dup)}")
    if missing:
        print("EKSIK:", missing[:30])
    main = main.drop_duplicates("id", keep="last")
    main["specialty"] = main.specialty.map(norm_spec)
    main = main.merge(items, on="id", how="left")
    main.to_parquet(os.path.join(ANN, "merged_raw.parquet"), index=False)

    iaa_dir = os.path.join(ANN, "iaa_raw")
    if os.path.isdir(iaa_dir) and glob.glob(os.path.join(iaa_dir, "*.json")):
        iaa = load_dir(iaa_dir).drop_duplicates("id", keep="last")
        rep = iaa_report(main.drop(columns=["kind", "title", "industry", "dataset_collar"]), iaa, items)
        write_json(rep, os.path.join(REPORTS, "annotation_iaa.json"))
        print("IAA:", json.dumps(rep, ensure_ascii=False))

    # konsolidasyon girdileri: ISCO 2-haneli gruplar
    out_dir = os.path.join(ANN, "consolidation_inputs")
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*.json")):
        os.remove(f)
    main["g2"] = main.isco4.str[:2]
    # cok buyuk 2-haneli gruplar (>300 cift) 3-haneli alt gruplara bolunur
    pair_n = main.groupby("g2").apply(lambda d: d[["specialty", "isco4"]].drop_duplicates().shape[0])
    big = set(pair_n[pair_n > 300].index)
    main.loc[main.g2.isin(big), "g2"] = main.isco4.str[:3]
    # ayni specialty farkli 2-haneli gruplarda geciyorsa her grupta gorunsun diye capraz bilgi
    spec_groups = main.groupby(main.specialty.map(spec_key)).g2.agg(lambda s: sorted(set(s))).to_dict()
    summary = {}
    for g2, gdf in main.groupby("g2"):
        recs = []
        for (spec, code), sdf in gdf.groupby(["specialty", "isco4"]):
            recs.append({
                "specialty": spec,
                "isco4": code,
                "isco_title": Counter(sdf.isco_title).most_common(1)[0][0],
                "n_cv": int((sdf.kind == "CV").sum()),
                "n_job": int((sdf.kind == "JOB").sum()),
                "ornek_unvanlar": sorted(set(sdf.title))[:5],
                "sektorler": sorted(set(sdf.industry))[:4],
                "diger_gruplarda_da_var": [x for x in spec_groups.get(spec_key(spec), []) if x != g2],
            })
        recs.sort(key=lambda r: (r["isco4"], spec_key(r["specialty"])))
        with open(os.path.join(out_dir, f"group_{g2}.json"), "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=0)
        summary[g2] = {"pairs": len(recs), "items": int(len(gdf))}
    write_json(summary, os.path.join(ANN, "consolidation_inputs", "_summary.json"))
    print(f"{len(summary)} grup; toplam (specialty,isco4) cifti: {sum(v['pairs'] for v in summary.values())}")


def finalize():
    main = pd.read_parquet(os.path.join(ANN, "merged_raw.parquet"))
    cons = load_dir(os.path.join(ANN, "consolidation_raw"))
    cmap = {(norm_spec(r.specialty), r.isco4): (norm_spec(r.canonical_specialty), r.canonical_isco4)
            for r in cons.itertuples()}
    miss = 0
    can_spec, can_isco = [], []
    for r in main.itertuples():
        k = (r.specialty, r.isco4)
        if k in cmap:
            s, c = cmap[k]
        else:
            miss += 1
            s, c = r.specialty, r.isco4
        can_spec.append(s)
        can_isco.append(c)
    main["specialty_raw"] = main.specialty
    main["isco4_raw"] = main.isco4
    main["specialty"] = can_spec
    main["isco4"] = can_isco
    # capraz-grup hakem kararlari (ayni kanonik uzmanlik adi -> tek isco4)
    adj_p = os.path.join(ANN, "adjudication.json")
    if os.path.exists(adj_p):
        adj = json.load(open(adj_p, encoding="utf-8"))
        amap = {spec_key(a["specialty"]): (norm_spec(a["final_specialty"]), a["final_isco4"]) for a in adj}
        for i, r in main.iterrows():
            k = spec_key(r.specialty)
            if k in amap:
                main.at[i, "specialty"], main.at[i, "isco4"] = amap[k]
    # her kanonik uzmanlik tek bir isco4'e baglanir (cogunluk)
    maj = main.groupby(main.specialty.map(spec_key)).isco4.agg(lambda s: Counter(s).most_common(1)[0][0])
    main["isco4"] = main.specialty.map(spec_key).map(maj)
    disp = main.groupby(main.specialty.map(spec_key)).specialty.agg(lambda s: Counter(s).most_common(1)[0][0])
    main["specialty"] = main.specialty.map(spec_key).map(disp)
    main["occ_key"] = main.isco4 + ":" + main.specialty.map(spec_key).str.replace(" ", "_")
    title_map = main.groupby("isco4").isco_title.agg(lambda s: Counter(s).most_common(1)[0][0])
    main["isco_title"] = main.isco4.map(title_map)
    cols = ["id", "kind", "title", "industry", "isco4", "isco_title", "specialty", "occ_key", "collar",
            "seniority", "conf", "specialty_raw", "isco4_raw", "dataset_collar"]
    out = main[cols].copy()
    out.to_parquet(os.path.join(ANN, "occupation_labels.parquet"), index=False)
    stats = {
        "oge": int(len(out)),
        "konsolidasyon_eslesmeyen": miss,
        "isco4_sayisi": int(out.isco4.nunique()),
        "occ_key_sayisi": int(out.occ_key.nunique()),
        "isco4_degisen_oge": int((out.isco4 != out.isco4_raw).sum()),
        "specialty_degisen_oge": int((out.specialty != out.specialty_raw).sum()),
        "occ_key_hem_cv_hem_ilan": int(len(set(out[out.kind == "CV"].occ_key) & set(out[out.kind == "JOB"].occ_key))),
        "isco4_hem_cv_hem_ilan": int(len(set(out[out.kind == "CV"].isco4) & set(out[out.kind == "JOB"].isco4))),
        "conf_dagilimi": out.conf.value_counts().to_dict(),
        "collar": out.groupby("kind").collar.value_counts().unstack().to_dict(),
    }
    write_json(stats, os.path.join(REPORTS, "annotation_stats.json"))
    print(json.dumps(stats, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    {"prepare": prepare, "finalize": finalize}[cmd]()
