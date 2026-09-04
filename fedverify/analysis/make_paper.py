"""Emit LaTeX macros for every number the paper cites, straight from results files.

CLAUDE.md, "Numbers": no number is ever typed into a document by hand. The paper body
writes \\FVmnistK5EpsInf and this module defines it from rounds.jsonl. A macro with no
backing data expands to a loud \\textbf{[??]} that is impossible to miss in a PDF draft,
so an unfinished experiment cannot silently become a claim.

    python3 -m fedverify.analysis.make_paper
    python3 -m fedverify.analysis.make_paper --check     # exit 1 if any macro is missing
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

from .make_tables import eps_key, load_rounds, load_runs, mean_std

MISSING_TEX = r"\textbf{[??]}"


def _fmt(vals, pct=True, dp=2):
    m, s = mean_std(vals) if vals else (None, None)
    if m is None:
        return None
    if pct:
        return f"{m*100:.{dp}f}\\,$\\pm$\\,{(s or 0)*100:.{dp}f}"
    return f"{m:.{dp}f}\\,$\\pm$\\,{(s or 0):.{dp}f}"


# The macros the PAPER CITES, declared up front. Without this the tool only creates
# macros for data that happens to exist, so four unrun experiments would silently produce
# no macros and the coverage line would read "51/52 backed" — hiding the hole rather than
# showing it.
EXPECTED = (
    ["FVexpOneCells", "FVexpTwoCells", "FVepsBreaches", "FVepsMaxRatio"]
    + [f"FVacc{d}K{k}Eps{e}" for d in ("Mnist", "Fmnist") for k in (5, 10)
       for e in ("0p5", "1", "2", "4", "8", "Inf")]
    + [f"FVacc{a}{g}" for a in ("Scaling", "Signflip", "Labelflip", "Backdoor")
       for g in ("Fedavg", "Forensics", "Krum", "Median")]
    + [f"FVdet{a}Forensics" for a in ("Scaling", "Signflip", "Labelflip", "Backdoor")]
    + [f"FVov{n}K{k}{b}" for n in ("Digestms", "Merklems", "Anchorms")
       for k in (5, 10, 20) for b in ("Mock", "Local")]
    + [f"FVmitbihK{k}Eps{e}F1" for k in (5, 10) for e in ("Inf", "4", "1")]
    + [f"FVdpDetScalingEps{e}" for e in ("Inf", "8", "4", "2", "1", "0p5")]
    + ["FVmitbihBeats", "FVmitbihPatients", "FVmitbihMajority", "FVmitbihNaiveF1"]
    + [f"FVaucEpsInfK{k}" for k in (5, 10)] + [f"FVtauEpsInfK{k}" for k in (5, 10)]
)


def collect(results_dir: str) -> dict:
    """Every macro the paper can cite -> value, or None when the run is missing."""
    M = {k: None for k in EXPECTED}

    def put(key, val):
        if val is not None or key not in M:
            M[key] = val

    # ── exp1: privacy/utility ────────────────────────────────────────────────
    e1 = load_runs(results_dir, "exp1")
    acc, f1, realized = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in e1:
        c = r["config"]
        k = (c["dataset"], int(c["num_clients"]), eps_key(c))
        acc[k].append(float(r["final"]["test_acc"]))
        f1[k].append(float(r["final"]["macro_f1"]))
        if r["max_realized_eps"] is not None:
            realized[k].append(float(r["max_realized_eps"]))
    for (ds, K, e), v in acc.items():
        tag = f"{ds.capitalize()}K{K}Eps{e.replace('.', 'p').replace('inf', 'Inf')}"
        put(f"FVacc{tag}", _fmt(v))
        put(f"FVf1{tag}", _fmt(f1[(ds, K, e)]))
    put("FVexpOneCells", str(len(e1)) if e1 else None)
    # the headline privacy claim: worst realized epsilon never exceeds its target
    over = [(eps_key(r["config"]), r["max_realized_eps"]) for r in e1 if r["max_realized_eps"]]
    breaches = [k for k, v in over if k != "inf" and v > float(k) * 1.001]
    put("FVepsBreaches", str(len(breaches)) if over else None)
    if over and any(k != "inf" for k, _ in over):
        put("FVepsMaxRatio", f"{max(v/float(k) for k, v in over if k != 'inf'):.3f}")

    # ── exp2: robustness ─────────────────────────────────────────────────────
    e2f, e2r = load_runs(results_dir, "exp2"), load_rounds(results_dir, "exp2")
    a2 = defaultdict(list)
    for r in e2f:
        c = r["config"]
        a2[((c.get("attack") or "none"), c["aggregator"])].append(float(r["final"]["test_acc"]))
    for (atk, agg), v in a2.items():
        put(f"FVacc{atk.title().replace('_','')}{agg.title().replace('_','')}", _fmt(v))
    d2 = defaultdict(list)
    for r in e2r:
        c = r["config"]
        per = [float(x["attack"]["f1"]) for x in r["rounds"] if x.get("attack", {}).get("active")]
        if per:
            d2[((c.get("attack") or "none"), c["aggregator"])].append(sum(per)/len(per))
    for (atk, agg), v in d2.items():
        put(f"FVdet{atk.title().replace('_','')}{agg.title().replace('_','')}", _fmt(v))
    put("FVexpTwoCells", str(len(e2f)) if e2f else None)

    # ── exp3: commitment overhead ────────────────────────────────────────────
    e3 = load_rounds(results_dir, "exp3")
    ov = defaultdict(lambda: defaultdict(list))
    for r in e3:
        c = r["config"]
        for rec in r["rounds"]:
            cm = rec.get("commit")
            if cm:
                for fld in ("digest_ms", "merkle_ms", "anchor_ms", "aggregate_ms", "bytes_committed"):
                    if cm.get(fld) is not None:
                        ov[(int(c["num_clients"]), c.get("chain_backend"))][fld].append(float(cm[fld]))
    for (K, be), d in ov.items():
        for fld, v in d.items():
            nm = fld.replace("_ms", "Ms").replace("_", "")
            put(f"FVov{nm.capitalize()}K{K}{str(be).capitalize()}", _fmt(v, pct=False))

    # ── exp4 / exp5 ──────────────────────────────────────────────────────────
    e4 = load_runs(results_dir, "exp4")
    for r in e4:
        c = r["config"]
        if c["dataset"] == "mitbih":
            M[f"FVmitbihK{c['num_clients']}Eps{eps_key(c).replace('.','p').replace('inf','Inf')}F1"] = \
                _fmt([float(r["final"]["macro_f1"])])
    e5 = load_rounds(results_dir, "exp5")
    d5 = defaultdict(list)
    for r in e5:
        c = r["config"]
        per = [float(x["attack"]["f1"]) for x in r["rounds"] if x.get("attack", {}).get("active")]
        if per:
            d5[(c.get("attack"), eps_key(c))].append(sum(per)/len(per))
    for (atk, e), v in d5.items():
        M[f"FVdpDet{str(atk).title().replace('_','')}Eps{e.replace('.','p').replace('inf','Inf')}"] = _fmt(v)

    # ── dataset + calibration facts ──────────────────────────────────────────
    cache = os.path.join(results_dir, "..", "data", "mitdb", "mitbih_w256.npz")
    if os.path.exists(cache):
        import numpy as np
        d = np.load(cache)
        y = d["y"]
        counts = np.bincount(y, minlength=5)
        M["FVmitbihBeats"] = f"{len(y):,}"
        M["FVmitbihPatients"] = str(len(set(d["rec"].tolist())))
        M["FVmitbihMajority"] = f"{100*counts.max()/counts.sum():.2f}"
        maj = counts.max()/counts.sum()
        M["FVmitbihNaiveF1"] = f"{100*(2*maj/(1+maj))/5:.2f}"

    tp = os.path.join(results_dir, "calibration", "taus.json")
    if os.path.exists(tp):
        by = json.load(open(tp))["by_key"]
        for key, e in by.items():
            c = e.get("combined")
            if not c:
                continue
            tag = key.replace("eps=", "Eps").replace("|K=", "K").replace(".", "p").replace("inf", "Inf")
            M[f"FVauc{tag}"] = f"{c['auc']:.3f}"
            M[f"FVtau{tag}"] = f"{c['tau_fpr5']:.2f}"

    return M


def emit(M: dict, out_path: str) -> list:
    missing = sorted(k for k, v in M.items() if v is None)
    lines = [
        "% Generated by fedverify/analysis/make_paper.py — DO NOT EDIT.",
        "% Every value here comes from a results file. A macro with no backing run",
        "% expands to [??] so an unfinished experiment cannot become a silent claim.",
        "",
    ]
    for k in sorted(M):
        v = M[k]
        lines.append(f"\\newcommand{{\\{k}}}{{{v if v is not None else MISSING_TEX}}}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write("\n".join(lines) + "\n")
    return missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="fedverify/results")
    ap.add_argument("--out", default="fedverify/paper/generated/numbers.tex")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any cited number has no backing run")
    a = ap.parse_args(argv)

    M = collect(a.results_dir)
    missing = emit(M, a.out)
    have = len(M) - len(missing)
    print(f"wrote {a.out}: {have}/{len(M)} macros backed by results")
    if missing:
        print(f"\n{len(missing)} macro(s) with NO data — these render as [??] in the PDF:",
              file=sys.stderr)
        for m in missing[:20]:
            print(f"  {m}", file=sys.stderr)
        if len(missing) > 20:
            print(f"  … and {len(missing)-20} more", file=sys.stderr)
    return 1 if (a.check and missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
