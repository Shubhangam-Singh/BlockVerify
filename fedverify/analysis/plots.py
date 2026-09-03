"""Figures for the paper — one per results table, PNG + PDF into results/figures/.

Same rule as the tables (CLAUDE.md, "Numbers"): every point is read from a results file.
Nothing here accepts a hand-typed number, and a figure with no data is SKIPPED with a
message on stderr rather than drawn empty — a blank axis in a paper is worse than a
missing figure, because it looks like a result.

    python3 -m fedverify.analysis.plots
    python3 -m fedverify.analysis.plots --only fig1 fig6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

from .make_tables import eps_key, load_rounds, load_runs, mean_std

_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "evaluation")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

EPS_ORDER = ["0.5", "1", "2", "4", "8", "inf"]
AGG_ORDER = ["fedavg", "krum", "multikrum", "trimmed_mean", "median", "forensics"]
COLORS = {"fedavg": "#888888", "krum": "#e07b39", "multikrum": "#d4a017",
          "trimmed_mean": "#3aa8c1", "median": "#6b8fd4", "forensics": "#8b3fd4"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "figure.autolayout": True})
    return plt


def _save(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=160)
    print(f"  wrote {out_dir}/{name}.png and .pdf")


def _skip(name, why):
    print(f"[skip] {name}: {why}", file=sys.stderr)


def _x_eps(keys):
    """Plot epsilon on a log axis with inf pinned one step past the largest finite value."""
    fin = sorted({float(k) for k in keys if k != "inf"})
    xs, labels = list(fin), [f"{v:g}" for v in fin]
    if "inf" in keys:
        xs.append(max(fin) * 2 if fin else 1.0)
        labels.append("∞")
    return xs, labels


# ── fig1: accuracy vs epsilon (table 1) ─────────────────────────────────────
def fig1(results_dir, out_dir):
    runs = load_runs(results_dir, "exp1")
    if not runs:
        return _skip("fig1", "no exp1 runs")
    plt = _plt()
    acc = defaultdict(list)
    for r in runs:
        c = r["config"]
        acc[(c["dataset"], int(c["num_clients"]), eps_key(c))].append(
            float(r["final"]["test_acc"]) * 100)

    series = sorted({(d, k) for d, k, _e in acc})
    keys = {e for _d, _k, e in acc}
    xs, labels = _x_eps(keys)
    order = [e for e in EPS_ORDER if e in keys]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for d, k in series:
        ys, es, px = [], [], []
        for e, x in zip(order, xs):
            v = acc[(d, k, e)]
            if v:
                m, s = mean_std(v)
                ys.append(m); es.append(s or 0.0); px.append(x)
        if ys:
            ax.errorbar(px, ys, yerr=es, marker="o", ms=4, capsize=3, label=f"{d} K={k}")
    ax.set_xscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("privacy budget ε  (lower = more private)")
    ax.set_ylabel("final test accuracy (%)")
    ax.set_title("Privacy–utility trade-off")
    ax.legend(fontsize=7)
    _save(fig, "fig1_accuracy_vs_epsilon", out_dir)


# ── fig2: accuracy vs attacker fraction, per aggregator (table 2) ───────────
def fig2(results_dir, out_dir):
    runs = load_runs(results_dir, "exp2")
    if not runs:
        return _skip("fig2", "no exp2 runs")
    plt = _plt()
    acc = defaultdict(list)
    attacks = set()
    for r in runs:
        c = r["config"]
        atk = c.get("attack") or "none"
        if eps_key(c) != "inf":
            continue
        attacks.add(atk)
        acc[(atk, c["aggregator"], float(c.get("attacker_frac", 0.0)))].append(
            float(r["final"]["test_acc"]) * 100)
    real = sorted(a for a in attacks if a != "none")
    if not real:
        return _skip("fig2", "no attacked exp2 cells at eps=inf")

    fig, axes = plt.subplots(1, len(real), figsize=(3.0 * len(real), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    fracs = sorted({f for _a, _g, f in acc if f > 0})
    for ax, atk in zip(axes, real):
        for g in AGG_ORDER:
            ys, xs2 = [], []
            for f in fracs:
                v = acc[(atk, g, f)]
                if v:
                    xs2.append(f); ys.append(mean_std(v)[0])
            if ys:
                ax.plot(xs2, ys, marker="o", ms=4, label=g, color=COLORS.get(g))
        base = acc[("none", "fedavg", 0.0)]
        if base:
            ax.axhline(mean_std(base)[0], ls="--", lw=0.9, color="#555",
                       label="clean FedAvg")
        ax.set_title(atk, fontsize=9)
        ax.set_xlabel("attacker fraction")
    axes[0].set_ylabel("final test accuracy (%)")
    axes[-1].legend(fontsize=6.5)
    fig.suptitle("Robustness to Byzantine clients (ε = ∞)", fontsize=10)
    _save(fig, "fig2_accuracy_vs_attacker_fraction", out_dir)


# ── fig3: ROC from calibration ──────────────────────────────────────────────
def fig3(results_dir, out_dir):
    path = os.path.join(results_dir, "calibration", "taus.json")
    if not os.path.exists(path):
        return _skip("fig3", "no calibration/taus.json")
    plt = _plt()
    data = json.load(open(path)).get("by_key", {})
    rows = [(k, v) for k, v in sorted(data.items()) if "combined" in v]
    if not rows:
        return _skip("fig3", "taus.json has no combined ROC")

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for key, entry in rows:
        for score in ("s_norm", "s_dir", "s_coord", "combined"):
            e = entry.get(score)
            if not e:
                continue
            lw = 2.0 if score == "combined" else 1.0
            ax.plot([0, e["fpr_at_youden"], 1], [0, e["tpr_at_youden"], 1],
                    lw=lw, marker="o", ms=3,
                    label=f"{score} (AUC={e['auc']:.3f})")
        break                      # one panel per figure keeps the legend readable
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false-positive rate (honest clients excluded)")
    ax.set_ylabel("true-positive rate (attackers caught)")
    ax.set_title(f"Forensics score separability — {rows[0][0]}", fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, "fig3_forensics_roc", out_dir)


# ── fig4: commitment overhead vs K (table 3) ─────────────────────────────────
def fig4(results_dir, out_dir):
    runs = load_rounds(results_dir, "exp3")
    if not runs:
        return _skip("fig4", "no exp3 runs")
    plt = _plt()
    vals = defaultdict(lambda: defaultdict(list))
    for r in runs:
        c = r["config"]
        key = (int(c["num_clients"]), c.get("chain_backend", "?"))
        for rec in r["rounds"]:
            cm = rec.get("commit")
            if cm:
                for f in ("digest_ms", "merkle_ms", "anchor_ms", "aggregate_ms"):
                    if cm.get(f) is not None:
                        vals[key][f].append(float(cm[f]))
    if not vals:
        return _skip("fig4", "exp3 runs contain no commit blocks")

    ks = sorted({k for k, _b in vals})
    backends = sorted({b for _k, b in vals})
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for b in backends:
        for f, ls in (("digest_ms", "-"), ("merkle_ms", "--"), ("anchor_ms", ":")):
            ys = [mean_std(vals[(k, b)][f])[0] for k in ks if vals[(k, b)][f]]
            xs = [k for k in ks if vals[(k, b)][f]]
            if ys:
                ax.plot(xs, ys, ls, marker="o", ms=4, label=f"{b} · {f[:-3]}")
    ax.set_yscale("log")
    ax.set_xlabel("clients per round (K)")
    ax.set_ylabel("per-round cost (ms, log)")
    ax.set_title("Chain commitment overhead")
    ax.legend(fontsize=6.5, ncol=2)
    _save(fig, "fig4_commitment_overhead", out_dir)


# ── fig5: heterogeneity (table 4) ───────────────────────────────────────────
def fig5(results_dir, out_dir):
    runs = load_runs(results_dir, "exp4")
    if not runs:
        return _skip("fig5", "no exp4 runs")
    plt = _plt()
    f1 = defaultdict(list)
    for r in runs:
        c = r["config"]
        if c["dataset"] == "mitbih":
            lab = f"mit-bih K={c['num_clients']}"
        else:
            a = c.get("alpha")
            lab = "mnist IID" if a in ("inf", None) or (isinstance(a, float) and math.isinf(a)) \
                else f"mnist α={float(a):g}"
        f1[(lab, eps_key(c))].append(float(r["final"]["macro_f1"]) * 100)

    keys = {e for _l, e in f1}
    xs, labels = _x_eps(keys)
    order = [e for e in EPS_ORDER if e in keys]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for lab in sorted({l for l, _e in f1}):
        ys, es, px = [], [], []
        for e, x in zip(order, xs):
            v = f1[(lab, e)]
            if v:
                m, s = mean_std(v)
                ys.append(m); es.append(s or 0.0); px.append(x)
        if ys:
            ax.errorbar(px, ys, yerr=es, marker="o", ms=4, capsize=3, label=lab)
    ax.set_xscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("final macro-F1 (%)")
    ax.set_title("Heterogeneity × privacy (macro-F1, not accuracy)")
    ax.legend(fontsize=7)
    _save(fig, "fig5_heterogeneity", out_dir)


# ── fig6: detection F1 vs epsilon (table 6) ─────────────────────────────────
def fig6(results_dir, out_dir):
    per = load_rounds(results_dir, "exp5")
    fin = load_runs(results_dir, "exp5")
    if not per:
        return _skip("fig6", "no exp5 runs")
    plt = _plt()
    f1, acc = defaultdict(list), defaultdict(list)
    for r in per:
        c = r["config"]
        vals = [float(rec["attack"]["f1"]) for rec in r["rounds"]
                if rec.get("attack", {}).get("active")]
        if vals:
            f1[(c.get("attack"), eps_key(c))].append(sum(vals) / len(vals) * 100)
    for r in fin:
        c = r["config"]
        acc[(c.get("attack"), eps_key(c))].append(float(r["final"]["test_acc"]) * 100)

    keys = {e for _a, e in f1} | {e for _a, e in acc}
    xs, labels = _x_eps(keys)
    order = [e for e in EPS_ORDER if e in keys]
    attacks = sorted({a for a, _e in f1} | {a for a, _e in acc})

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax2 = ax.twinx()
    for atk in attacks:
        ys, px = [], []
        for e, x in zip(order, xs):
            if f1[(atk, e)]:
                ys.append(mean_std(f1[(atk, e)])[0]); px.append(x)
        if ys:
            ax.plot(px, ys, marker="o", ms=4, label=f"{atk} · detection F1")
        ys, px = [], []
        for e, x in zip(order, xs):
            if acc[(atk, e)]:
                ys.append(mean_std(acc[(atk, e)])[0]); px.append(x)
        if ys:
            ax2.plot(px, ys, ls="--", marker="s", ms=3, alpha=0.6,
                     label=f"{atk} · accuracy")
    ax.set_xscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("Byzantine detection F1 (%)")
    ax2.set_ylabel("final accuracy (%, dashed)")
    ax2.grid(False)
    ax.set_title("Does DP noise blind Byzantine screening?")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5)
    _save(fig, "fig6_detection_vs_epsilon", out_dir)


FIGURES = {"fig1": fig1, "fig2": fig2, "fig3": fig3,
           "fig4": fig4, "fig5": fig5, "fig6": fig6}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="fedverify/results")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", nargs="*", default=None, choices=sorted(FIGURES))
    a = ap.parse_args(argv)
    out_dir = a.out_dir or os.path.join(a.results_dir, "figures")
    for name, fn in FIGURES.items():
        if a.only and name not in a.only:
            continue
        print(f"building {name} ...")
        try:
            fn(a.results_dir, out_dir)
        except Exception as e:            # one bad figure must not kill the rest
            print(f"[fail] {name}: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
