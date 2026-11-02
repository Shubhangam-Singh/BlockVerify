"""Assemble <results>/report_pack/: final figures and every table, named by report section (Phase 11.4).

    python3 -m fedverify.analysis.make_report_pack

Reads the mapping in table.yaml (report_pack): which table, figure and diagram goes into which section of the VIT
report (chapter 4: design, experiments and results; an appendix for the earlier MNIST study) and onto slides 8 and 11
of the 13-slide deck. Copies, never regenerates:
  figures/  <section>_<name>.png   (300 dpi, checked; the serif font the figure generator used)
  tables/   <section>_<name>.csv and .md
  latex/    <section>_<name>.tex   (compile-checked: tables/latex_check.json), and all_tables.pdf
  diagrams/ <section>_<name>.mmd, .svg and .png   (Mermaid sources and their renders, PNG at 300 dpi; Phase 12:
            python3 -m fedverify.analysis.render_diagrams makes them, docs/diagrams/rendered.json records them)
and writes README.md saying where each file goes. Fails if a generated table is not assigned to a section, a figure
or a diagram PNG is not 300 dpi, or a diagram's render no longer matches its source.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

from .make_fl_healthcare_tables import REPO, load_spec

DIAGRAMS = REPO / "docs" / "diagrams"


def md_to_csv(md: Path, out: Path) -> None:
    """The earlier MNIST tables have no CSV: rebuild it from their generated Markdown (header and rows only)."""
    rows = [ln for ln in md.read_text(encoding="utf-8").splitlines() if ln.startswith("|")]
    cells = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in rows]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cells[0])
        w.writerows(cells[2:])


def png_dpi(path: Path) -> Optional[float]:
    from PIL import Image
    with Image.open(path) as im:
        dpi = im.info.get("dpi")
    return None if not dpi else float(round(dpi[0]))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fedverify.analysis.make_report_pack", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="fedverify/results")
    a = ap.parse_args(argv)
    spec = load_spec()
    cfg = spec["report_pack"]
    root = Path(a.results_dir)
    tables, figures = root / "tables", root / "figures"
    pack = root / "report_pack"
    if pack.exists():
        shutil.rmtree(pack)
    for sub in ("figures", "tables", "latex", "diagrams"):
        (pack / sub).mkdir(parents=True)
    problems, placed, where = [], set(), {}
    from .render_diagrams import INDEX, stale
    stale_diagrams = set(stale(DIAGRAMS))
    rendered = json.loads((DIAGRAMS / INDEX).read_text()) if (DIAGRAMS / INDEX).exists() else {}
    for sec in cfg["sections"]:
        sid = sec["id"]
        for name in sec.get("tables", []):
            md = tables / f"{name}.md"
            if not md.exists():
                problems.append(f"table {name} (section {sid}) is missing")
                continue
            base = f"{sid}_{name}"                      # section ids contain dots: never use with_suffix here
            shutil.copy(md, pack / "tables" / f"{base}.md")
            src_csv = tables / f"{name}.csv"
            if src_csv.exists():
                shutil.copy(src_csv, pack / "tables" / f"{base}.csv")
            else:
                md_to_csv(md, pack / "tables" / f"{base}.csv")
            if (tables / f"{name}.tex").exists():
                shutil.copy(tables / f"{name}.tex", pack / "latex" / f"{sid}_{name}.tex")
            placed.add(name)
            where[("table", name)] = sid
        for name in sec.get("figures", []):
            png = figures / f"{name}.png"
            if not png.exists():
                problems.append(f"figure {name} (section {sid}) is missing")
                continue
            dpi = png_dpi(png)
            if dpi != 300:
                problems.append(f"figure {name} is {dpi} dpi, not 300")
            shutil.copy(png, pack / "figures" / f"{sid}_{name}.png")
            where[("figure", name)] = sid
        for name in sec.get("diagrams", []):
            src = DIAGRAMS / f"{name}.mmd"
            if not src.exists():
                problems.append(f"diagram {name} (section {sid}) is missing")
                continue
            shutil.copy(src, pack / "diagrams" / f"{sid}_{name}.mmd")
            where[("diagram", name)] = sid
            if name in stale_diagrams:                  # Phase 12: only renders that match their source
                problems.append(f"diagram {name}: its SVG/PNG is missing or older than its source "
                                "(python3 -m fedverify.analysis.render_diagrams)")
                continue
            if png_dpi(src.with_suffix(".png")) != 300:
                problems.append(f"diagram {name}.png is not 300 dpi")
            for ext in (".svg", ".png"):
                shutil.copy(src.with_suffix(ext), pack / "diagrams" / f"{sid}_{name}{ext}")
    unassigned = sorted(p.stem for p in tables.glob("*.md") if p.stem not in placed)
    if unassigned:
        problems.append(f"tables not assigned to a report section in table.yaml: {unassigned}")
    pdf = tables / "latex_check" / "all_tables.pdf"
    if pdf.exists():
        shutil.copy(pdf, pack / "latex" / "all_tables.pdf")
    check = json.loads((tables / "latex_check.json").read_text()) if (tables / "latex_check.json").exists() else None
    from .make_fl_healthcare_figures import setup_style
    font = setup_style()

    lines = ["# Report pack", "",
             "*Generated by `python3 -m fedverify.analysis.make_report_pack` from the results files; rebuild, do not "
             "edit.* Every file is a copy of a generated result, named `<report section>_<name>`.", "",
             f"- `figures/`: PNG at 300 dpi (checked), serif font {font} (Times-like), with the captions in "
             "`results/figures/captions.md`.",
             "- `tables/`: every table as CSV (for Word or a spreadsheet) and Markdown (with its notes).",
             "- `latex/`: the same tables as LaTeX for the IEEE paper; they need `\\usepackage{graphicx}`. "
             + ("Compile check: " + (f"{sum(r['compiled'] and not r['overfull_pt'] for r in check['tables'].values())} "
                                     f"of {len(check['tables'])} tables compile in an IEEE two-column document with "
                                     "no overfull box (`results/tables/latex_check.json`); `all_tables.pdf` shows them."
                                     if check else "not run.")),
             "- `diagrams/`: each diagram as its Mermaid source (`.mmd`), SVG, and PNG at 300 dpi (checked), "
             + (f"rendered with mermaid-cli {rendered.get('mermaid_cli')} in {rendered.get('browser')} by "
                "`python3 -m fedverify.analysis.render_diagrams` (`docs/diagrams/rendered.json`)."
                if rendered else "not rendered (`python3 -m fedverify.analysis.render_diagrams`)."), "",
             "## Where each file goes in the VIT report", ""]
    for sec in cfg["sections"]:
        lines += [f"### {sec['id']} {sec['title']}", ""]
        for kind, sub, ext in (("tables", "tables", ".csv / .md"), ("figures", "figures", ".png"),
                               ("diagrams", "diagrams", ".png")):
            for name in sec.get(kind, []):
                lines.append(f"- `{sub}/{sec['id']}_{name}{ext if kind != 'tables' else '.csv'}`"
                             + (f" and `.md`" if kind == "tables" else "")
                             + (" (also `.svg`, and the `.mmd` source)" if kind == "diagrams" else ""))
        lines.append("")
    lines += ["## The 13-slide deck", ""]
    for num, slide in sorted(cfg.get("slides", {}).items(), key=lambda kv: int(kv[0])):
        items = ([f"diagram `diagrams/{where.get(('diagram', n), '?')}_{n}.png`" for n in slide.get("diagrams", [])]
                 + [f"table `tables/{where.get(('table', n), '?')}_{n}.csv`" for n in slide.get("tables", [])]
                 + [f"figure `figures/{where.get(('figure', n), '?')}_{n}.png`" for n in slide.get("figures", [])])
        lines.append(f"- Slide {num}, {slide['title']}: " + "; ".join(items) + ".")
    lines += ["", "The `_superseded/` results are kept outside this pack until the guide approves the numbers."]
    (pack / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for p in problems:
        print(f"[problem] {p}", file=sys.stderr)
    n = sum(1 for _ in pack.rglob("*") if _.is_file())
    print(f"wrote {pack} ({n} files); {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
