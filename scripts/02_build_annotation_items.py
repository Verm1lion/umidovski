# -*- coding: utf-8 -*-
"""Meslek anotasyonu icin girdi ogelerini hazirlar (CV + ilan).

Her oge icin anotatorun ihtiyac duydugu kisa baglam uretilir:
  * CV  : unvan (headline), sektor, sehir, ozet (ilk 350 kr), beceriler, deneyimler
          (unvan + tarih + kisa aciklama), egitim. CV'nin veri setindeki `collar`
          alani anotatora GOSTERILMEZ (yaka anlasmasini bagimsiz olcmek icin).
  * Ilan: unvan, sektor, position_level, calisma sekli, aciklamanin ilk 550 karakteri.

Ogeler (sektor, normalize unvan) sirasina gore dizilip ~90'lik parcalara bolunur;
boylece ayni meslegin CV ve ilan tarafi ayni anotator tarafindan gorulur (tutarlilik).

Cikti: annotation/items.jsonl, annotation/chunks/chunk_###.json
"""
import json
import os
import random
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(ROOT, "annotation")
CHUNK = 90


def norm(s):
    s = str(s or "").lower()
    s = s.translate(str.maketrans("çğıöşüâî", "cgiosuai"))
    return re.sub(r"\s+", " ", s).strip()


def short(s, n):
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def cv_items():
    cv = pd.read_csv(os.path.join(ROOT, "veriseti", "tum_cvler.csv"), encoding="utf-8-sig")
    out = []
    for r in cv.itertuples(index=False):
        j = json.loads(r.cv_json)
        exps = []
        for e in (j.get("experience") or [])[:4]:
            exps.append(short(f"{e.get('title','')} ({e.get('dates','')}): {e.get('description','')}", 170))
        edu = []
        for e in (j.get("education") or [])[:2]:
            edu.append(short(f"{e.get('school','')} - {e.get('degree','')} ({e.get('field','')}) {e.get('dates','')}", 110))
        out.append({
            "id": f"C{r.cv_id}",
            "kind": "CV",
            "title": str(r.NaturalTitle),
            "industry": r.Category,
            "city": r.location,
            "context": {
                "ozet": short(j.get("about") or "", 350),
                "beceriler": short(", ".join(map(str, (j.get("skills") or [])[:10])), 220),
                "deneyim": exps,
                "egitim": edu,
            },
        })
    return out


def job_items():
    jobs = pd.read_parquet(os.path.join(ROOT, "work", "tum_ilanlar.parquet"))
    out = []
    for i, r in enumerate(jobs.itertuples(index=False)):
        out.append({
            "id": f"J{i:04d}",
            "job_uuid": r.job_id,
            "kind": "JOB",
            "title": r.title,
            "industry": r.industry,
            "city": r.city,
            "position_level": r.position_level,
            "mode": r.mode_of_operation,
            "context": {"aciklama": short(r.description, 550)},
        })
    return out


def main():
    items = cv_items() + job_items()
    os.makedirs(os.path.join(ANN, "chunks"), exist_ok=True)
    with open(os.path.join(ANN, "items.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    items_sorted = sorted(items, key=lambda x: (x["industry"], norm(x["title"]), x["id"]))
    chunks = [items_sorted[i:i + CHUNK] for i in range(0, len(items_sorted), CHUNK)]
    for k, ch in enumerate(chunks):
        payload = [{kk: vv for kk, vv in it.items() if kk != "job_uuid"} for it in ch]
        with open(os.path.join(ANN, "chunks", f"chunk_{k:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=0)
    # Anotatorlar arasi uyum (IAA) icin %10 rastgele alt kume, farkli siralama ile
    rng = random.Random(20261015)
    iaa = rng.sample(items, k=int(0.10 * len(items)))
    rng.shuffle(iaa)
    os.makedirs(os.path.join(ANN, "iaa_chunks"), exist_ok=True)
    for k in range(0, len(iaa), 101):
        payload = [{kk: vv for kk, vv in it.items() if kk != "job_uuid"} for it in iaa[k:k + 101]]
        with open(os.path.join(ANN, "iaa_chunks", f"iaa_{k // 101:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=0)
    print(f"{len(items)} oge, {len(chunks)} parca; IAA alt kumesi {len(iaa)} oge")


if __name__ == "__main__":
    main()
