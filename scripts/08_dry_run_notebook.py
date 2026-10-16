# -*- coding: utf-8 -*-
"""Notebook dogrulamasi: (1) JSON/nbformat gecerliligi, (2) her kod hucresinin AST ile
ayristirilmasi, (3) tanimsiz isim taramasi (pyflakes), (4) CPU uzerinde DRY_RUN modunda
uctan uca calistirma (nbclient). Ag erisimi gerekmez (kucuk rastgele modeller notebook icinde uretilir).

Kullanim:
  python scripts/08_dry_run_notebook.py [--bundle data_bundle.zip] [--workdir work/dry_run] [--static-only]
Cikti: <workdir>/executed.ipynb ve reports/notebook_dogrulama.json
"""
import argparse
import ast
import json
import os
import shutil
import sys
import time

import nbformat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "CV_Ilan_SOTA_Egitim_ve_Test_A100.ipynb")


def static_checks(nb):
    rep = {"hucre": len(nb.cells), "kod_hucresi": 0, "ast_hatalari": [], "pyflakes": []}
    nbformat.validate(nb)
    rep["nbformat_gecerli"] = True
    src_all = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        rep["kod_hucresi"] += 1
        src = "\n".join(l for l in c.source.splitlines() if not l.strip().startswith(("%", "!")))
        try:
            ast.parse(src)
        except SyntaxError as e:
            rep["ast_hatalari"].append(f"hucre {i}: {e}")
        src_all.append(src)
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
        import io
        out, err = io.StringIO(), io.StringIO()
        check("\n\n".join(src_all), "notebook", Reporter(out, err))
        msgs = [m for m in out.getvalue().splitlines() if "undefined name" in m or "syntax" in m.lower()]
        # IPython yerlesik 'display' ve notebook icinde kosullu tanimlananlar haric
        rep["pyflakes"] = [m for m in msgs if "'display'" not in m]
    except ImportError:
        rep["pyflakes"] = ["pyflakes kurulu degil"]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(ROOT, "data_bundle.zip"))
    ap.add_argument("--workdir", default=os.path.join(ROOT, "work", "dry_run"))
    ap.add_argument("--kernel", default="python3")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--report", default=os.path.join(ROOT, "reports", "notebook_dogrulama.json"))
    a = ap.parse_args()
    nb = nbformat.read(NB, as_version=4)
    rep = static_checks(nb)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if rep["ast_hatalari"] or rep["pyflakes"]:
        print("STATIK KONTROL BASARISIZ")
        if not a.static_only:
            sys.exit(1)
    if a.static_only:
        return
    if os.path.isdir(a.workdir):
        shutil.rmtree(a.workdir)
    os.makedirs(a.workdir)
    os.environ["CV_DRY_RUN"] = "1"
    os.environ["CV_DATA_BUNDLE"] = os.path.abspath(a.bundle)
    from nbclient import NotebookClient
    t0 = time.time()
    client = NotebookClient(nb, timeout=a.timeout, kernel_name=a.kernel, resources={"metadata": {"path": a.workdir}})
    status = "OK"
    err = None
    try:
        client.execute()
    except Exception as e:  # hucre hatasi
        status = "HATA"
        import re
        err = re.sub(r"\x1b\[[0-9;]*m", "", str(e))[-3000:]
    nbformat.write(nb, os.path.join(a.workdir, "executed.ipynb"))
    rep["dry_run"] = {"durum": status, "sure_sn": round(time.time() - t0, 1), "hata": err,
                      "RUN_OK": os.path.exists(os.path.join(a.workdir, "outputs", "RUN_OK.json")),
                      "export_zip": os.path.exists(os.path.join(a.workdir, "export_model_bundle.zip"))}
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps(rep["dry_run"], ensure_ascii=False, indent=1))
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
