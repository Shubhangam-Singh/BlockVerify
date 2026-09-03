"""Build every results table from results files. NEVER type a number by hand.

Rule (CLAUDE.md, "Numbers"): every number in every table is emitted here from a
results file. Missing cells render as "—" and are listed on stderr.

    python3 -m fedverify.analysis.make_tables
    python3 -m fedverify.analysis.make_tables --results-dir fedverify/results --only table1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Optional

MISSING = "—"


# ── loading ─────────────────────────────────────────────────────────────────
def load_runs(results_dir: str, exp: str) -> list[dict]:
    """Every completed cell under results_dir/exp as {config, final, max_realized_eps}."""
    runs, root = [], os.path.join(results_dir, exp)
    if not os.path.isdir(root):
        return runs
    for dirpath, _dirnames, filenames in os.walk(root):
        if "rounds.jsonl" not in filenames or "config.json" not in filenames:
            continue
        try:
            cfg = json.load(open(os.path.join(dirpath, "config.json")))["config"]
            last = None
            with open(os.path.join(dirpath, "rounds.jsonl")) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last = json.loads(line)
            if last is None:
                continue
        except (json.JSONDecodeError, KeyError, OSError):
            print(f"[warn] unreadable run at {dirpath}", file=sys.stderr)
            continue

        # An interrupted or orphaned cell must NEVER be averaged in as if it had
        # finished: its last round is mid-training and would silently drag the mean
        # down. Skip it and say so, exactly like a missing cell.
        want = int(cfg.get("rounds", 0) or 0)
        if want and int(last.get("round", 0)) < want:
            print(f"[skip] incomplete run ({last.get('round')}/{want} rounds): {dirpath}",
                  file=sys.stderr)
            continue

        eps_list = [p.get("realized_eps") for p in (last.get("privacy") or {}).values()
                    if p and p.get("realized_eps") is not None]
        runs.append({"dir": dirpath, "config": cfg, "final": last,
                     "max_realized_eps": max(eps_list) if eps_list else None})
    return runs


def load_rounds(results_dir: str, exp: str) -> list[dict]:
    """Every cell under results_dir/exp as {config, rounds:[...]}.

    Table 3 averages over ROUNDS within a cell (a per-round cost), not over seeds like
    Table 1, so it needs every record rather than just the final one.
    """
    out, root = [], os.path.join(results_dir, exp)
    if not os.path.isdir(root):
        return out
    for dirpath, _dirnames, filenames in os.walk(root):
        if "rounds.jsonl" not in filenames or "config.json" not in filenames:
            continue
        try:
            cfg = json.load(open(os.path.join(dirpath, "config.json")))["config"]
            rounds = [json.loads(l) for l in open(os.path.join(dirpath, "rounds.jsonl"))
                      if l.strip()]
        except (json.JSONDecodeError, KeyError, OSError):
            print(f"[warn] unreadable run at {dirpath}", file=sys.stderr)
            continue
        if rounds:
            out.append({"dir": dirpath, "config": cfg, "rounds": rounds})
    return out


# ── formatting ──────────────────────────────────────────────────────────────
def eps_key(cfg) -> str:
    e = cfg.get("epsilon")
    return "inf" if e in (None, "inf") or (isinstance(e, float) and math.isinf(e)) else f"{float(e):g}"


def mean_std(xs: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)     # sample std over seeds
    return m, math.sqrt(var)


def cell(xs: list[float], pct: bool = True) -> str:
    m, s = mean_std(xs)
    if m is None:
        return MISSING
    return f"{m*100:.2f} ± {s*100:.2f}" if pct else f"{m:.4f} ± {s:.4f}"


def _emit(name: str, header: list[str], rows: list[list[str]], out_dir: str,
          caption: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    md = [f"# {name}", "", caption, "",
          "| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    open(os.path.join(out_dir, f"{name}.md"), "w").write("\n".join(md) + "\n")

    tex = ["\\begin{table}[t]", "\\centering",
           f"\\caption{{{caption}}}",
           "\\begin{tabular}{l" + "r" * (len(header) - 1) + "}", "\\hline",
           " & ".join(h.replace("±", "$\\pm$") for h in header) + " \\\\", "\\hline"]
    tex += [" & ".join(c.replace("±", "$\\pm$").replace(MISSING, "--") for c in r) + " \\\\"
            for r in rows]
    tex += ["\\hline", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(out_dir, f"{name}.tex"), "w").write("\n".join(tex) + "\n")
    print(f"  wrote {out_dir}/{name}.md and .tex")


# ── table 1 ─────────────────────────────────────────────────────────────────
def table1(results_dir: str, out_dir: str) -> None:
    runs = load_runs(results_dir, "exp1")
    if not runs:
        print("[warn] table1: no exp1 runs found — run "
              "`python3 -m fedverify.experiments.exp1_privacy_utility` first", file=sys.stderr)

    acc = defaultdict(list); f1 = defaultdict(list); eps_real = defaultdict(list)
    datasets, ks, epsilons = set(), set(), set()
    for r in runs:
        c = r["config"]
        key = (eps_key(c), c["dataset"], int(c["num_clients"]))
        acc[key].append(float(r["final"]["test_acc"]))
        f1[key].append(float(r["final"]["macro_f1"]))
        if r["max_realized_eps"] is not None:
            eps_real[key].append(float(r["max_realized_eps"]))
        datasets.add(c["dataset"]); ks.add(int(c["num_clients"])); epsilons.add(eps_key(c))

    datasets = sorted(datasets) or ["mnist", "fmnist"]
    ks = sorted(ks) or [5, 10]
    order = ["0.5", "1", "2", "4", "8", "inf"]
    eps_rows = [e for e in order if e in epsilons] + sorted(epsilons - set(order))
    if not eps_rows:
        eps_rows = order

    cols = [(d, k) for d in datasets for k in ks]
    header = ["ε"] + [f"{d} K={k}" for d, k in cols] + ["max realized ε"]
    missing = []

    for src, name, cap in ((acc, "table1", "Final test accuracy (%) — mean ± std over seeds"),
                           (f1, "table1b", "Final macro-F1 (%) — mean ± std over seeds")):
        rows = []
        for e in eps_rows:
            row = [e]
            for d, k in cols:
                vals = src[(e, d, k)]
                row.append(cell(vals))
                if not vals and src is acc:
                    missing.append(f"table1[eps={e}, {d}, K={k}]")
            realized = [v for d, k in cols for v in eps_real[(e, d, k)]]
            row.append(f"{max(realized):.3f}" if realized else MISSING)
            rows.append(row)
        _emit(name, header, rows, out_dir, cap)

    for m in missing:
        print(f"[missing] {m}", file=sys.stderr)


# ── table 3 ─────────────────────────────────────────────────────────────────
def _ms(xs: list[float], fmt: str = "{:.2f}") -> str:
    m, s = mean_std(xs)
    if m is None:
        return MISSING
    return f"{fmt.format(m)} ± {fmt.format(s)}"


def table3(results_dir: str, out_dir: str) -> None:
    """Per-round cost of the commitment layer, by client count and chain backend."""
    runs = load_rounds(results_dir, "exp3")
    if not runs:
        print("[warn] table3: no exp3 runs found — run "
              "`python3 -m fedverify.experiments.exp3_chain_overhead` first", file=sys.stderr)

    FIELDS = [("digest_ms", "digest (ms)", "{:.2f}"),
              ("merkle_ms", "merkle (ms)", "{:.2f}"),
              ("anchor_ms", "anchor (ms)", "{:.2f}"),
              ("aggregate_ms", "aggregate (ms)", "{:.2f}"),
              ("bytes_committed", "bytes committed", "{:,.0f}")]

    by_cell, ks, backends = defaultdict(lambda: defaultdict(list)), set(), set()
    for r in runs:
        c = r["config"]
        k, backend = int(c["num_clients"]), c.get("chain_backend", "?")
        ks.add(k); backends.add(backend)
        for rec in r["rounds"]:
            cm = rec.get("commit")
            if not cm:            # a cell run without --commit has nothing to report
                continue
            for field, _label, _f in FIELDS:
                if cm.get(field) is not None:
                    by_cell[(k, backend)][field].append(float(cm[field]))

    ks = sorted(ks) or [5, 10, 20]
    order = ["mock", "local", "algorand"]
    backends = [b for b in order if b in backends] + sorted(backends - set(order))
    if not backends:
        backends = ["mock", "local"]

    header = ["K · backend"] + [label for _f, label, _fmt in FIELDS] + ["rounds"]
    rows, missing = [], []
    for k in ks:
        for b in backends:
            vals = by_cell.get((k, b))
            if not vals:
                missing.append(f"table3[K={k}, backend={b}]")
                rows.append([f"K={k} · {b}"] + [MISSING] * (len(FIELDS) + 1))
                continue
            row = [f"K={k} · {b}"]
            for field, _label, fmt in FIELDS:
                row.append(_ms(vals.get(field, []), fmt))
            row.append(str(len(vals.get("anchor_ms", []))))
            rows.append(row)

    _emit("table3", header, rows, out_dir,
          "Per-round cost of the chain commitment layer — mean ± std over rounds")
    for m in missing:
        print(f"[missing] {m}", file=sys.stderr)


# ── table 2 / 2b ────────────────────────────────────────────────────────────
ATTACK_ORDER = ["none", "label_flip", "sign_flip", "gaussian", "scaling", "backdoor"]
AGG_ORDER = ["fedavg", "krum", "multikrum", "trimmed_mean", "median", "forensics"]
MAIN_FRAC = 0.2


def _atk(cfg) -> str:
    return cfg.get("attack") or "none"


def _exp2_index(results_dir):
    """(final-accuracy runs, per-round runs) for exp2, both keyed the same way."""
    return load_runs(results_dir, "exp2"), load_rounds(results_dir, "exp2")


def _key(cfg):
    return (_atk(cfg), cfg.get("aggregator", "?"), float(cfg.get("attacker_frac", 0.0)),
            eps_key(cfg))


def _grid_axes(seen_attacks, seen_aggs):
    atk = [a for a in ATTACK_ORDER if a in seen_attacks] or ATTACK_ORDER
    agg = [a for a in AGG_ORDER if a in seen_aggs] or AGG_ORDER
    return atk, agg


def table2(results_dir: str, out_dir: str) -> None:
    """Final accuracy (and ASR for backdoor) per attack x aggregator at frac 0.2."""
    runs, _ = _exp2_index(results_dir)
    if not runs:
        print("[warn] table2: no exp2 runs found — run "
              "`python3 -m fedverify.experiments.exp2_byzantine` first", file=sys.stderr)

    acc, asr = defaultdict(list), defaultdict(list)
    seen_a, seen_g, seen_f = set(), set(), set()
    for r in runs:
        c, k = r["config"], _key(r["config"])
        acc[k].append(float(r["final"]["test_acc"]))
        a = (r["final"].get("attack") or {}).get("asr")
        if a is not None and not math.isnan(float(a)):
            asr[k].append(float(a))
        seen_a.add(_atk(c)); seen_g.add(c.get("aggregator")); seen_f.add(float(c.get("attacker_frac", 0.0)))

    attacks, aggs = _grid_axes(seen_a, seen_g)
    missing = []

    # main table: one frac, one epsilon — the readable summary
    for e in ["inf"]:
        header = ["attack"] + aggs
        rows = []
        for a in attacks:
            frac = 0.0 if a == "none" else MAIN_FRAC
            row = [a if a != "none" else "none (clean)"]
            for g in aggs:
                k = (a, g, frac, e)
                if not acc[k]:
                    missing.append(f"table2[{a}, {g}, frac={frac}, eps={e}]")
                    row.append(MISSING)
                    continue
                txt = cell(acc[k])
                if asr[k]:
                    txt += f" / ASR {cell(asr[k])}"
                row.append(txt)
            rows.append(row)
        _emit("table2", header, rows, out_dir,
              f"Final test accuracy (%) — and ASR (%) where a backdoor is present — "
              f"per attack x aggregator at attacker fraction {MAIN_FRAC}, epsilon = inf, "
              f"mean +- std over seeds")

    # appendix: every fraction and epsilon
    header = ["attack", "frac", "ε"] + aggs
    rows = []
    for a in attacks:
        for f in ([0.0] if a == "none" else sorted(x for x in seen_f if x > 0)):
            for e in ["inf", "4"]:
                row = [a, f"{f:g}", e]
                any_cell = False
                for g in aggs:
                    k = (a, g, f, e)
                    if acc[k]:
                        any_cell = True
                        txt = cell(acc[k])
                        if asr[k]:
                            txt += f" / ASR {cell(asr[k])}"
                        row.append(txt)
                    else:
                        row.append(MISSING)
                if any_cell or not runs:
                    rows.append(row)
    _emit("table2_full", header, rows, out_dir,
          "Appendix: final test accuracy (%) / ASR (%) across every attacker fraction "
          "and privacy level, mean +- std over seeds")

    for m in missing:
        print(f"[missing] {m}", file=sys.stderr)


def table2b(results_dir: str, out_dir: str) -> None:
    """Byzantine DETECTION quality: mean per-round F1 over rounds where the attack is live.

    Only rules that make a per-client accept/reject decision are scored. Krum's top-1
    selection and the coordinate-wise rules discard clients without claiming they are
    malicious, so they are marked n/a rather than being given a meaningless F1 — this is
    reported from diag["detector"], not assumed from the aggregator's name.
    """
    _, runs = _exp2_index(results_dir)
    if not runs:
        print("[warn] table2b: no exp2 runs found", file=sys.stderr)

    # Whether a rule is a DETECTOR is a property of the aggregator, not of whether a run
    # happened to exist — otherwise an empty grid renders every cell as "n/a" instead of "—".
    from ..aggregators import AGGREGATORS
    detector = {n: bool(c.detects) for n, c in AGGREGATORS.items()}
    f1, fer = defaultdict(list), defaultdict(list)
    seen_a, seen_g = set(), set()
    for r in runs:
        c = r["config"]
        k = _key(c)
        seen_a.add(_atk(c)); seen_g.add(c.get("aggregator"))
        per_f1, per_fer = [], []
        for rec in r["rounds"]:
            at = rec.get("attack")
            if not at:
                continue
            if at.get("active"):
                per_f1.append(float(at["f1"]))
            per_fer.append(float(at["false_exclusion_rate"]))
        if per_f1:
            f1[k].append(sum(per_f1) / len(per_f1))
        if per_fer:
            fer[k].append(sum(per_fer) / len(per_fer))

    attacks, aggs = _grid_axes(seen_a, seen_g)
    attacks = [a for a in attacks if a != "none"]
    header = ["attack"] + aggs
    rows, missing = [], []
    for a in attacks:
        row = [a]
        for g in aggs:
            k = (a, g, MAIN_FRAC, "inf")
            if not detector.get(g, False):
                row.append("n/a")            # selection rule, not a detector
            elif f1[k]:
                row.append(cell(f1[k]))
            else:
                missing.append(f"table2b[{a}, {g}, frac={MAIN_FRAC}]")
                row.append(MISSING)
        rows.append(row)

    # false-exclusion on the CLEAN configuration: the cost of the defence when nobody attacks
    row = ["false-exclusion (clean)"]
    for g in aggs:
        k = ("none", g, 0.0, "inf")
        row.append(cell(fer[k]) if fer[k] else ("n/a" if not detector.get(g, False) else MISSING))
    rows.append(row)

    _emit("table2b", header, rows, out_dir,
          f"Byzantine detection F1 per round while the attack is active, at attacker "
          f"fraction {MAIN_FRAC}, epsilon = inf, mean +- std over seeds. Final row: "
          f"honest-client false-exclusion rate on the clean configuration.")
    for m in missing:
        print(f"[missing] {m}", file=sys.stderr)


# ── table 4 ─────────────────────────────────────────────────────────────────
def _alpha_key(cfg) -> str:
    a = cfg.get("alpha")
    if a in ("inf", None) or (isinstance(a, float) and math.isinf(a)):
        return "iid"
    return f"{float(a):g}"


def table4(results_dir: str, out_dir: str) -> None:
    """Heterogeneity x privacy. Macro-F1 leads; accuracy is shown only so the gap shows.

    On MIT-BIH 89.5% of beats are class N, so a model that always answers N scores 89.5%
    accuracy and 18.9% macro-F1. Reading the accuracy column alone would say that model is
    excellent. That is the whole reason macro-F1 is the primary metric here.
    """
    runs = load_runs(results_dir, "exp4")
    if not runs:
        print("[warn] table4: no exp4 runs found — run "
              "`python3 -m fedverify.experiments.exp4_heterogeneity` first", file=sys.stderr)

    f1, acc = defaultdict(list), defaultdict(list)
    seen_eps = set()
    for r in runs:
        c = r["config"]
        row = (("mnist", _alpha_key(c)) if c["dataset"] != "mitbih"
               else ("mitbih", f"K={c['num_clients']}"))
        k = (row, eps_key(c))
        f1[k].append(float(r["final"]["macro_f1"]))
        acc[k].append(float(r["final"]["test_acc"]))
        seen_eps.add(eps_key(c))

    eps_cols = [e for e in ["inf", "4", "1"] if e in seen_eps] or ["inf", "4", "1"]
    rows_spec = ([("mnist", a) for a in ["0.1", "0.5", "1", "iid"]] +
                 [("mitbih", f"K={k}") for k in [5, 10]])

    header = ["dataset · split"] + [f"ε={e}" for e in eps_cols]
    rows, missing = [], []
    for spec in rows_spec:
        ds, lab = spec
        pretty = (f"mnist · α={lab}" if ds == "mnist" and lab != "iid"
                  else "mnist · IID" if ds == "mnist" else f"mit-bih · {lab} (patient)")
        row = [pretty]
        for e in eps_cols:
            k = (spec, e)
            if not f1[k]:
                missing.append(f"table4[{pretty}, eps={e}]")
                row.append(MISSING)
            else:
                row.append(f"{cell(f1[k])}  ({cell(acc[k])})")
        rows.append(row)

    _emit("table4", header, rows, out_dir,
          "Macro-F1 % (accuracy % in parentheses) under heterogeneity and privacy, "
          "mean ± std over seeds. MNIST heterogeneity is synthesised by the Dirichlet "
          "alpha; MIT-BIH heterogeneity is inherited from a patient-disjoint split. "
          "Macro-F1 is primary: an always-N classifier scores 89.5 accuracy / 18.9 macro-F1 "
          "on MIT-BIH.")
    for m in missing:
        print(f"[missing] {m}", file=sys.stderr)


# ── table 5 (qualitative, from analysis/comparison.yaml) ────────────────────
_MARK = {"yes": "✓", "no": "✗", "?": "?"}


def table5(results_dir: str, out_dir: str) -> None:
    """Qualitative positioning. Unknown cells stay "?" — guessing would be the defect."""
    import yaml
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison.yaml")
    if not os.path.exists(path):
        print(f"[warn] table5: {path} missing", file=sys.stderr)
        return
    spec = yaml.safe_load(open(path))
    cols = spec["columns"]
    header = ["system"] + [c["label"] for c in cols]
    rows = []
    for sysrow in spec["systems"]:
        name = sysrow["name"]
        cite = sysrow.get("cite")
        label = f"{name} [{cite}]" if cite and cite != "—" else name
        rows.append([label] + [_MARK.get(str(sysrow.get(c["key"], "?")), "?") for c in cols])
    n_unknown = sum(r.count("?") for r in rows)
    _emit("table5", header, rows, out_dir,
          f"Qualitative comparison of blockchain-based and private federated-learning "
          f"systems. ✓ = supported, ✗ = not supported, ? = not stated clearly enough in "
          f"the cited paper to score ({n_unknown} cells). Sourced from "
          f"fedverify/analysis/comparison.yaml.")


# ── table 6 ─────────────────────────────────────────────────────────────────
def table6(results_dir: str, out_dir: str) -> None:
    """Does DP noise blind Byzantine screening? Detection F1 and accuracy vs epsilon."""
    finals = load_runs(results_dir, "exp5")
    perround = load_rounds(results_dir, "exp5")
    if not finals:
        print("[warn] table6: no exp5 runs found — run "
              "`python3 -m fedverify.experiments.exp5_dp_byzantine_interaction` first",
              file=sys.stderr)

    acc, asr = defaultdict(list), defaultdict(list)
    for r in finals:
        c = r["config"]
        k = (c.get("attack"), eps_key(c))
        acc[k].append(float(r["final"]["test_acc"]))
        a = (r["final"].get("attack") or {}).get("asr")
        if a is not None and not math.isnan(float(a)):
            asr[k].append(float(a))

    f1 = defaultdict(list)
    for r in perround:
        c = r["config"]
        k = (c.get("attack"), eps_key(c))
        per = [float(rec["attack"]["f1"]) for rec in r["rounds"]
               if rec.get("attack", {}).get("active")]
        if per:
            f1[k].append(sum(per) / len(per))

    eps_rows = ["inf", "8", "4", "2", "1", "0.5"]
    header = ["ε", "scaling: acc", "scaling: det F1",
              "backdoor: acc", "backdoor: ASR", "backdoor: det F1"]
    rows, missing = [], []
    for e in eps_rows:
        row = [e]
        for atk in ("scaling", "backdoor"):
            k = (atk, e)
            row.append(cell(acc[k]) if acc[k] else MISSING)
            if atk == "backdoor":
                row.append(cell(asr[k]) if asr[k] else MISSING)
            row.append(cell(f1[k]) if f1[k] else MISSING)
            if not acc[k]:
                missing.append(f"table6[{atk}, eps={e}]")
        rows.append(row)

    _emit("table6", header, rows, out_dir,
          "Interaction of differential privacy with Byzantine screening: FedVerify-"
          "Forensics under a 20% attacker fraction as epsilon tightens. Detection F1 is "
          "the per-round mean while the attack is active. tau is looked up per epsilon "
          "from calibration, so a falling F1 reflects DP noise rather than a stale "
          "threshold. Mean ± std over seeds.")
    for m in missing:
        print(f"[missing] {m}", file=sys.stderr)


TABLES = {"table1": table1, "table2": table2, "table2b": table2b, "table3": table3,
          "table4": table4, "table5": table5, "table6": table6}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="fedverify/results")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", nargs="*", default=None, choices=sorted(TABLES))
    a = ap.parse_args(argv)
    out_dir = a.out_dir or os.path.join(a.results_dir, "tables")
    for name, fn in TABLES.items():
        if a.only and name not in a.only:
            continue
        print(f"building {name} ...")
        fn(a.results_dir, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
