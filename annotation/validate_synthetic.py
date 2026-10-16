# -*- coding: utf-8 -*-
"""Sentetik CV uretim ciktisini dogrular.

Kullanim: python3 annotation/validate_synthetic.py <specs_###.json> <gen_###.json>
Cikti kaydi: {"id": "S30001", "headline": str, "cv_text": str}
"""
import json
import re
import sys

REQ_HEADERS = ["Konum:", "Özet / Hakkında:", "Beceriler:", "Deneyimler:", "Eğitim:"]


def main(spec_p, gen_p):
    specs = {s["id"]: s for s in json.load(open(spec_p, encoding="utf-8"))}
    gen = json.load(open(gen_p, encoding="utf-8"))
    errs = []
    seen = set()
    for g in gen:
        sid = g.get("id")
        if sid not in specs:
            errs.append(f"{sid}: specs'te yok")
            continue
        seen.add(sid)
        if set(g) - {"id", "headline", "cv_text"}:
            errs.append(f"{sid}: yalnizca id/headline/cv_text alanlari olmali (hedef etiketler YAZILMAMALI)")
        t = str(g.get("cv_text", ""))
        h = str(g.get("headline", "")).strip()
        if not h or len(h) > 90:
            errs.append(f"{sid}: headline bos/uzun")
        if not t.startswith(f"Konum: {specs[sid]['city']}"):
            errs.append(f"{sid}: cv_text 'Konum: {specs[sid]['city']}' ile baslamali")
        for hd in REQ_HEADERS:
            if hd not in t:
                errs.append(f"{sid}: '{hd}' bolumu eksik")
        if not (500 <= len(t) <= 3200):
            errs.append(f"{sid}: uzunluk {len(t)} (500-3200 olmali)")
        if re.search(r"(?i)\b(isco|occ_key|target_|grade)\b", t):
            errs.append(f"{sid}: metinde etiket/meta sizintisi")
    miss = set(specs) - seen
    if miss:
        errs.append(f"Eksik id'ler: {sorted(miss)}")
    if errs:
        print("\n".join(errs[:40]))
        return 1
    print(f"OK {len(gen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
