# -*- coding: utf-8 -*-
"""Konsolidasyon ciktisini dogrular.

Kullanim: python3 annotation/validate_consolidation.py <group_girdi.json> <cikti.json>
Cikti: [{"specialty","isco4","canonical_specialty","canonical_isco4"}...] -- girdideki her
(specialty, isco4) cifti icin tam bir kayit; her canonical_specialty tek bir canonical_isco4'e bagli.
"""
import json
import re
import sys


def main(inp, out):
    pairs = {(r["specialty"], r["isco4"]) for r in json.load(open(inp, encoding="utf-8"))}
    rows = json.load(open(out, encoding="utf-8"))
    errs = []
    got = set()
    can = {}
    for r in rows:
        for k in ("specialty", "isco4", "canonical_specialty", "canonical_isco4"):
            if not isinstance(r.get(k), str) or not r[k].strip():
                errs.append(f"{r}: '{k}' eksik/bos")
        if not re.fullmatch(r"\d{4}", str(r.get("canonical_isco4", ""))):
            errs.append(f"{r.get('specialty')}: canonical_isco4 dort haneli olmali")
        got.add((r.get("specialty"), r.get("isco4")))
        cs = str(r.get("canonical_specialty", "")).strip()
        can.setdefault(cs, set()).add(r.get("canonical_isco4"))
    miss = pairs - got
    extra = got - pairs
    if miss:
        errs.append(f"Eksik ciftler ({len(miss)}): {sorted(miss)[:15]}")
    if extra:
        errs.append(f"Girdide olmayan ciftler ({len(extra)}): {sorted(extra)[:15]}")
    multi = {k: v for k, v in can.items() if len(v) > 1}
    if multi:
        errs.append(f"Birden fazla isco4'e bagli kanonik uzmanliklar: {list(multi.items())[:10]}")
    if errs:
        print("\n".join(errs[:40]))
        return 1
    print(f"OK {len(rows)} cift -> {len(can)} kanonik uzmanlik")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
