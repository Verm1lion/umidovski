# -*- coding: utf-8 -*-
"""FAZ 1.1 - veriseti/tum_ilanlar.sql dosyasini PostgreSQL olmadan ayristirir.

SQL dosyasi 3.000 adet `INSERT INTO job_advertisements (...) VALUES (...);` ifadesi
icerir. Degerler tek tirnakli string ('' kacisli), NULL ya da `'...'::jsonb` olabilir.
Bu script karakter duzeyinde bir ayristirici ile her satiri okur, raw_data JSON'unu
cozer ve duz kolonlara (sehir, ilce, kategori ...) acar.

Cikti: work/tum_ilanlar.parquet ve work/tum_ilanlar.csv

Kullanim: python scripts/01_parse_jobs_sql.py
"""
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQL_PATH = os.path.join(ROOT, "veriseti", "tum_ilanlar.sql")
OUT_DIR = os.path.join(ROOT, "work")

INSERT_RE = re.compile(r"INSERT INTO\s+job_advertisements\s*\(([^)]*)\)\s*VALUES\s*\(", re.IGNORECASE)


def parse_values(sql, pos):
    """`VALUES (` sonrasindan baslayip kapanan `)` karakterine kadar degerleri okur."""
    values = []
    n = len(sql)
    while pos < n:
        ch = sql[pos]
        if ch in " \t\r\n,":
            pos += 1
            continue
        if ch == ")":
            return values, pos + 1
        if ch == "'":
            pos += 1
            buf = []
            while True:
                q = sql.find("'", pos)
                if q < 0:
                    raise ValueError("Kapanmayan string")
                buf.append(sql[pos:q])
                if q + 1 < n and sql[q + 1] == "'":  # '' kacisi
                    buf.append("'")
                    pos = q + 2
                    continue
                pos = q + 1
                break
            val = "".join(buf)
            if sql.startswith("::jsonb", pos):
                pos += len("::jsonb")
                val = json.loads(val)
            values.append(val)
            continue
        if sql.startswith("NULL", pos):
            values.append(None)
            pos += 4
            continue
        m = re.match(r"-?\d+(\.\d+)?", sql[pos:pos + 40])
        if m:
            values.append(float(m.group(0)) if m.group(1) else int(m.group(0)))
            pos += len(m.group(0))
            continue
        raise ValueError(f"Beklenmeyen karakter {ch!r} @ {pos}: {sql[pos:pos + 80]!r}")
    raise ValueError("VALUES kapanmadi")


def parse_sql(path):
    with open(path, encoding="utf-8") as f:
        sql = f.read()
    rows = []
    for m in INSERT_RE.finditer(sql):
        cols = [c.strip() for c in m.group(1).split(",")]
        vals, _ = parse_values(sql, m.end())
        if len(vals) != len(cols):
            raise ValueError(f"Kolon/deger sayisi uyusmuyor: {len(cols)} vs {len(vals)}")
        rows.append(dict(zip(cols, vals)))
    return rows


def flatten(rows):
    out = []
    for r in rows:
        raw = r.get("raw_data") or {}
        loc = raw.get("location") or {}
        if not isinstance(loc, dict):
            loc = {"city": str(loc)}
        out.append({
            "job_id": r["id"],
            "slug": r.get("slug"),
            "title": r.get("title"),
            "company_name": r.get("company_name"),
            "description": r.get("description"),
            "job_type": r.get("job_type"),
            "mode_of_operation": r.get("mode_of_operation"),
            "position_level": r.get("position_level"),
            "industry": r.get("industry"),
            "raw_category": raw.get("category"),
            "raw_title": raw.get("title"),
            "raw_language": raw.get("language"),
            "raw_has_salary": raw.get("hasSalary"),
            "raw_position_level": raw.get("positionLevel"),
            "raw_mode": raw.get("modeOfOperation"),
            "city": loc.get("city"),
            "district": loc.get("district"),
            "zone": loc.get("zone"),
            "raw_data_json": json.dumps(raw, ensure_ascii=False),
        })
    return pd.DataFrame(out)


def main():
    rows = parse_sql(SQL_PATH)
    df = flatten(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_parquet(os.path.join(OUT_DIR, "tum_ilanlar.parquet"), index=False)
    df.to_csv(os.path.join(OUT_DIR, "tum_ilanlar.csv"), index=False, encoding="utf-8")
    print(f"{len(df)} ilan ayristirildi -> work/tum_ilanlar.parquet")
    assert df["job_id"].is_unique, "job_id tekil degil"
    print(df[["title", "industry", "position_level", "mode_of_operation", "city"]].head())
    return 0


if __name__ == "__main__":
    sys.exit(main())
