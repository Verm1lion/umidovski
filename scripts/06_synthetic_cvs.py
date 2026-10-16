# -*- coding: utf-8 -*-
"""Yalnizca TRAIN icin sentetik CV zenginlestirmesi (InPars / Promptagator yaklasimi).

Gerekce: Ilan havuzundaki bazi uzmanliklar (occ_key) icin train bolumunde hic / tek CV yoktur; bi-encoder
bu meslekler icin pozitif ornek goremez. Ilan tarafindaki meslek bilgisinden (yalnizca WARM ilanlar)
gercekci Turkce CV'ler uretilir. Hicbir val/test CV'si uretimde kullanilmaz.

Alt komutlar
  targets  annotation/occupation_labels.parquet + split -> annotation/synthetic/specs/specs_###.json
  merge    gen/*.json (uretici) + verify/*.json (kor dogrulayici anotasyonu) -> synthetic_cvs.json
           Gidis-donus tutarlilik filtresi (round-trip consistency, Dai vd. 2022 Promptagator):
           kor anotatorun isco4'u hedefle ayni, yaka ayni ve |kidem farki| <= 1 olmali.
           Ayrica normalize basligi val/test CV basliklarindan biriyle cakisan sentetik CV atilir.

Kullanim:
  python scripts/06_synthetic_cvs.py targets [--per-batch 25] [--max 900]   # bu depo: 868 hedef, 35 parti
  python scripts/06_synthetic_cvs.py merge
"""
import argparse
import glob
import importlib
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, write_json  # noqa: E402

B = importlib.import_module("05_build_dataset")
core = B.core
SYN = os.path.join(ROOT, "annotation", "synthetic")
SYN_ID0 = 30000
BIG_CITIES = ["İstanbul", "Ankara", "İzmir", "Bursa", "Kocaeli", "Antalya", "Konya", "Adana", "Gaziantep", "Eskişehir"]


def current_split(lab):
    cvs = B.build_cvs(lab, None)
    jobs = B.make_job_pool(B.build_jobs(lab))
    cvs = B.make_split(cvs)
    return cvs, jobs


def targets(per_batch, max_n):
    lab = B.load_labels()
    cvs, jobs = current_split(lab)
    rng = np.random.default_rng(B.SEED + 1)
    train_occ = Counter(cvs.gold_occ_key[(cvs.split == "train")])
    warm = jobs[jobs.job_pool == "warm"]
    rows = []
    for occ, g in warm.groupby("gold_occ_key"):
        n_tr = train_occ.get(occ, 0)
        need = 1 if n_tr <= 1 else 0          # kapsama (cesitlilik) derinlikten onemli: meslek basina 1 CV
        if not need:
            continue
        rows.append((occ, need, len(g)))
    # oncelik: hic train CV'si olmayan + daha cok ilani olan meslekler
    rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
    specs = []
    for occ, need, _ in rows:
        g = warm[warm.gold_occ_key == occ].reset_index(drop=True)
        for k in range(need):
            j = g.iloc[k % len(g)]
            sen = int(np.clip(int(j.gold_seniority) + rng.choice([-1, 0, 0, 1]), 0, 5))
            city = j.city if (isinstance(j.city, str) and j.city and j.mode_of_operation != "Remote") else str(rng.choice(BIG_CITIES))
            specs.append({
                "id": f"S{SYN_ID0 + len(specs) + 1}",
                "target_occ_key": occ,
                "target_specialty": j.gold_specialty,
                "target_isco4": j.gold_isco4,
                "target_isco_title": j.gold_isco_title,
                "target_collar": j.gold_collar,
                "target_seniority": sen,
                "target_seniority_tr": core.SENIORITY_TR[sen],
                "industry": j.industry,
                "industry_tr": core.INDUSTRY_TR.get(j.industry, j.industry),
                "city": city,
                "example_job_titles": sorted(set(g.title))[:3],
                "style": str(rng.choice(["maskeli ([TARİH]/[ŞİRKET] kullan)", "gercek firma ve yil", "gercek firma ve yil",
                                         "kisa ve oz", "detayli"])),
            })
            if len(specs) >= max_n:
                break
        if len(specs) >= max_n:
            break
    os.makedirs(os.path.join(SYN, "specs"), exist_ok=True)
    for f in glob.glob(os.path.join(SYN, "specs", "*.json")):
        os.remove(f)
    batches = [specs[i:i + per_batch] for i in range(0, len(specs), per_batch)]
    for i, b in enumerate(batches):
        with open(os.path.join(SYN, "specs", f"specs_{i:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(b, f, ensure_ascii=False, indent=1)
    write_json({"hedef_meslek": len(rows), "sentetik_cv": len(specs), "parti": len(batches),
                "train_occ_key_sayisi": len(train_occ), "warm_occ_key_sayisi": int(warm.gold_occ_key.nunique())},
               os.path.join(SYN, "specs", "_summary.json"))
    print(f"{len(rows)} hedef meslek, {len(specs)} sentetik CV, {len(batches)} parti")


def merge():
    specs = {}
    for p in sorted(glob.glob(os.path.join(SYN, "specs", "specs_*.json"))):
        for s in json.load(open(p, encoding="utf-8")):
            specs[s["id"]] = s
    gen = {}
    for p in sorted(glob.glob(os.path.join(SYN, "gen", "*.json"))):
        for g in json.load(open(p, encoding="utf-8")):
            gen[g["id"]] = g
    ver = {}
    for p in sorted(glob.glob(os.path.join(SYN, "verify", "*.json"))):
        for v in json.load(open(p, encoding="utf-8")):
            ver[v["id"]] = v
    lab = B.load_labels()
    cvs, _ = current_split(lab)
    heldout_titles = set(cvs.headline[cvs.split.isin(["val", "test"])].map(B.spec_key))
    out, stats = [], Counter()
    for sid, s in specs.items():
        g = gen.get(sid)
        if g is None:
            stats["uretilmedi"] += 1
            continue
        v = ver.get(sid)
        if v is None:
            stats["dogrulanmadi"] += 1
            continue
        ok_isco = v["isco4"] == s["target_isco4"]
        ok_col = v["collar"] == s["target_collar"]
        ok_sen = abs(int(v["seniority"]) - int(s["target_seniority"])) <= 1
        stats["isco4_tutarli"] += ok_isco
        stats["yaka_tutarli"] += ok_col
        stats["kidem_tutarli"] += ok_sen
        if not (ok_isco and ok_col and ok_sen):
            stats["tutarlilik_filtresi_ret"] += 1
            continue
        if B.spec_key(g["headline"]) in heldout_titles:
            stats["val_test_baslik_cakismasi_ret"] += 1
            continue
        if not g["cv_text"].startswith("Konum:"):
            stats["bicim_ret"] += 1
            continue
        out.append({
            "syn_id": int(sid[1:]), "headline": g["headline"].strip(), "cv_text": g["cv_text"].strip(), "city": s["city"],
            "gold_industry": s["industry"], "gold_isco4": s["target_isco4"], "gold_isco_title": s["target_isco_title"],
            "gold_specialty": s["target_specialty"], "gold_occ_key": s["target_occ_key"], "gold_collar": s["target_collar"],
            "gold_seniority": int(s["target_seniority"]),
        })
    stats["kabul"] = len(out)
    stats["hedef"] = len(specs)
    with open(os.path.join(SYN, "synthetic_cvs.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    write_json(dict(stats), os.path.join(ROOT, "reports", "synthetic_cv_stats.json"))
    print(dict(stats))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["targets", "merge"])
    ap.add_argument("--per-batch", type=int, default=25)
    ap.add_argument("--max", type=int, default=900)
    a = ap.parse_args()
    if a.cmd == "targets":
        targets(a.per_batch, a.max)
    else:
        merge()
