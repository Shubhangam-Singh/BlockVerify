"""Rebuild every FL-healthcare table, figure, the results draft, the showcase page and the demo script (DEMO.md)
from the results files.

    python3 -m fedverify.analysis.make_fl_healthcare_all
    python3 -m fedverify.analysis.make_fl_healthcare_all --results-dir fedverify/results

The draft inserts the tables and figure captions, so it runs after them; then the showcase page, then DEMO.md
(it quotes the live demo's last run and the study's results), then the report pack. Exits non-zero if any step
fails. The LaTeX check (needs a TeX engine) and the freeze (make_manifest) are separate, deliberate steps.
"""
from __future__ import annotations

import argparse
from typing import Optional

from . import (make_demo_md, make_fl_healthcare_figures, make_fl_healthcare_tables, make_report_pack, make_results_draft,
               make_showcase)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fedverify.analysis.make_fl_healthcare_all", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="fedverify/results")
    a = ap.parse_args(argv)
    for step in (make_fl_healthcare_tables, make_fl_healthcare_figures, make_results_draft, make_showcase, make_demo_md,
                 make_report_pack):
        print(f"== {step.__name__.rsplit('.', 1)[-1]}", flush=True)
        code = step.main(["--results-dir", a.results_dir])
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
