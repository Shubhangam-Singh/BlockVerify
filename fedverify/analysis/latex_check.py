"""Compile every generated .tex table inside a minimal IEEE two-column document (Phase 11.3).

    python3 -m fedverify.analysis.latex_check --engine /path/to/tectonic
    FEDVERIFY_TEX=/path/to/tectonic python3 -m fedverify.analysis.latex_check

Each <results>/tables/<name>.tex is compiled on its own (IEEEtran, conference, two columns; T1 fonts, graphicx), so
an error or an overfull box is attributed to one table. The tables put their tabular in a box that LaTeX shrinks to
the line width only when it is wider (never cropped, no column dropped); the check reads the box's natural width
through the \\fvcheck hook and reports the scale each table needed. A table shrunk below `--min-scale` is flagged:
it should be split into two tables instead. Then all tables are compiled into one document for a visual check.

Writes <results>/tables/latex_check.json and <results>/tables/latex_check/all_tables.pdf. Exits 1 if any table fails
to compile or overflows. Engines: tectonic (preferred; downloads the TeX files it needs on first use) or pdflatex.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

PREAMBLE = r"""\documentclass[conference]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\newcommand{\fvcheck}[1]{\typeout{FVCHECK #1 natural=\the\wd\fvbox\space line=\the\linewidth}}
\begin{document}
\title{FedVerify generated tables: LaTeX check}
\maketitle
\section{Check}
This document only checks that the generated tables compile and fit an IEEE two-column page.
"""
OVERFULL = re.compile(r"Overfull \\hbox \(([\d.]+)pt too wide\)")
CHECK = re.compile(r"FVCHECK (\S+) natural=([\d.]+)pt line=([\d.]+)pt")


def find_engine(given: Optional[str]) -> Optional[str]:
    for cand in (given, os.environ.get("FEDVERIFY_TEX"), shutil.which("tectonic"), shutil.which("pdflatex")):
        if cand and Path(cand).exists():
            return str(cand)
    return None


def compile_doc(engine: str, tex: Path, workdir: Path) -> tuple[int, str]:
    """(exit code, log text) of one compile."""
    engine = str(Path(engine).resolve())                    # the compile runs inside workdir
    if Path(engine).name.startswith("tectonic"):
        cmd = [engine, "--keep-logs", "--outdir", ".", tex.name]
    else:
        cmd = [engine, "-interaction=nonstopmode", "-halt-on-error", "-output-directory=.", tex.name]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=600)
    log = workdir / (tex.stem + ".log")
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    return proc.returncode, text + "\n" + proc.stdout + "\n" + proc.stderr


def check_table(engine: str, table: Path, workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(table, workdir / table.name)
    doc = workdir / f"check_{table.stem}.tex"
    doc.write_text(PREAMBLE + f"\\input{{{table.stem}}}\n\\end{{document}}\n", encoding="utf-8")
    code, log = compile_doc(engine, doc, workdir)
    over = [float(x) for x in OVERFULL.findall(log)]
    parts = [(float(n), float(w)) for _, n, w in CHECK.findall(log)]   # one per float (split tables have several)
    natural, line = (max(parts, key=lambda nw: nw[0] / nw[1]) if parts else (None, None))
    errors = [ln for ln in log.splitlines() if ln.startswith("!") or ln.startswith("error:")][:5]
    return {"compiled": code == 0, "errors": errors, "overfull_pt": over, "floats": len(parts),
            "natural_width_pt": natural, "line_width_pt": line,
            "scale": None if not natural else min(1.0, line / natural)}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fedverify.analysis.latex_check", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="fedverify/results")
    ap.add_argument("--engine", default=None, help="tectonic or pdflatex (default: $FEDVERIFY_TEX, then PATH)")
    ap.add_argument("--min-scale", type=float, default=0.75,
                    help="flag a table shrunk below this factor: split it rather than shrink it")
    a = ap.parse_args(argv)
    engine = find_engine(a.engine)
    tables = Path(a.results_dir) / "tables"
    if engine is None:
        print("No TeX engine found (tectonic or pdflatex on PATH, or FEDVERIFY_TEX). The tables were NOT compiled.",
              file=sys.stderr)
        return 2
    work = tables / "latex_check"
    if work.exists():
        shutil.rmtree(work)
    texs = sorted(p for p in tables.glob("*.tex"))
    report = {"engine": engine, "version": subprocess.run([engine, "--version"], capture_output=True,
                                                          text=True).stdout.strip().splitlines()[:1],
              "document": "IEEEtran conference, two columns, T1 fonts, graphicx", "min_scale": a.min_scale,
              "tables": {}}
    for t in texs:
        r = check_table(engine, t, work / t.stem)
        r["flag_split"] = r["scale"] is not None and r["scale"] < a.min_scale
        report["tables"][t.stem] = r
        print(f"  {t.stem:<28} {'ok ' if r['compiled'] else 'FAIL'}  overfull {len(r['overfull_pt'])}"
              + (f"  scale {r['scale']:.2f}" if r["scale"] is not None else "")
              + ("  (split?)" if r["flag_split"] else ""), flush=True)
    allw = work / "all"
    allw.mkdir(parents=True, exist_ok=True)
    for t in texs:
        shutil.copy(t, allw / t.name)
    doc = allw / "all_tables.tex"
    doc.write_text(PREAMBLE + "".join(f"\\input{{{t.stem}}}\n\\clearpage\n" for t in texs) + "\\end{document}\n",
                   encoding="utf-8")
    code, _ = compile_doc(engine, doc, allw)
    if code == 0 and (allw / "all_tables.pdf").exists():
        shutil.copy(allw / "all_tables.pdf", work / "all_tables.pdf")
    report["all_tables_pdf"] = code == 0
    (tables / "latex_check.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    bad = [n for n, r in report["tables"].items() if not r["compiled"] or r["overfull_pt"]]
    print(f"{len(texs)} tables: {len(texs) - len(bad)} compile cleanly; problems: {bad or 'none'}; "
          f"all-tables PDF: {'ok' if report['all_tables_pdf'] else 'FAILED'}")
    return 1 if bad or not report["all_tables_pdf"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
