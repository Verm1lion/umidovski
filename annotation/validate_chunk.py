# -*- coding: utf-8 -*-
"""Anotasyon cikti dosyasini dogrular.

Kullanim: python3 annotation/validate_chunk.py <girdi_parca.json> <cikti.json>
Basarili ise 'OK <n>' yazar ve 0 ile cikar; aksi halde hatalari listeler ve 1 ile cikar.
"""
import json
import re
import sys

REQ = {"id": str, "isco4": str, "isco_title": str, "specialty": str, "collar": str, "seniority": int, "conf": int}


def main(inp, out):
    with open(inp, encoding="utf-8") as f:
        ids = [x["id"] for x in json.load(f)]
    with open(out, encoding="utf-8") as f:
        rows = json.load(f)
    errs = []
    if not isinstance(rows, list):
        print("Cikti bir JSON listesi olmali")
        return 1
    seen = {}
    for r in rows:
        for k, t in REQ.items():
            if k not in r:
                errs.append(f"{r.get('id')}: '{k}' eksik")
            elif not isinstance(r[k], t) or (t is int and isinstance(r[k], bool)):
                errs.append(f"{r.get('id')}: '{k}' tipi {t.__name__} olmali")
        if not re.fullmatch(r"\d{4}", str(r.get("isco4", ""))):
            errs.append(f"{r.get('id')}: isco4 dort haneli olmali")
        if r.get("collar") not in ("BLUE", "WHITE"):
            errs.append(f"{r.get('id')}: collar BLUE/WHITE olmali")
        if not (isinstance(r.get("seniority"), int) and 0 <= r["seniority"] <= 5):
            errs.append(f"{r.get('id')}: seniority 0-5 olmali")
        if not (isinstance(r.get("conf"), int) and 1 <= r["conf"] <= 3):
            errs.append(f"{r.get('id')}: conf 1-3 olmali")
        if not str(r.get("specialty", "")).strip():
            errs.append(f"{r.get('id')}: specialty bos")
        seen[r.get("id")] = seen.get(r.get("id"), 0) + 1
    missing = [i for i in ids if i not in seen]
    extra = [i for i in seen if i not in set(ids)]
    dup = [i for i, c in seen.items() if c > 1]
    if missing:
        errs.append(f"Eksik id'ler ({len(missing)}): {missing[:20]}")
    if extra:
        errs.append(f"Fazla id'ler: {extra[:20]}")
    if dup:
        errs.append(f"Tekrarlanan id'ler: {dup[:20]}")
    if errs:
        print("\n".join(errs[:60]))
        return 1
    print(f"OK {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
