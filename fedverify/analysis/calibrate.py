"""Calibrate the FedVerify-Forensics threshold tau from data. Never guess it.

docs/EVALUATION.md §5.2 is the precedent: BlockVerify's hand-set layer threshold z>8 turned
out to have FPR 0.49 on real weights because real distributions are heavy-tailed. The same
mistake is available one level up, so the client-level threshold is derived the same way —
by building a ROC over labelled clean and attacked rounds.

Procedure
  1. Run short FL cells with aggregator=forensics and tau=inf, so every client is SCORED
     but none is rejected on the threshold. (Hard health flags still fire; they are not
     calibrated and never were.)
  2. Label each per-client score: 1 if that client was an attacker in a round where the
     attack was active, else 0. Clean cells contribute negatives only.
  3. ROC per sub-score and for the combined score, using the same pure-numpy ROC the
     layer-level evaluation uses (evaluation/eval_lib.py).
  4. Write results/calibration/taus.json keyed by (epsilon, K).

exp2 READS tau from that file and records it, with its provenance, in config.json.

    python3 -m fedverify.analysis.calibrate --smoke
    python3 -m fedverify.analysis.calibrate
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

from ..config import FLConfig
from ..core.runner import run

_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "evaluation")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
from eval_lib import auc, rates_at, roc_curve            # noqa: E402

EXP = "calib"
SCORES = ["s_norm", "s_dir", "s_coord", "combined"]
CAL_ATTACKS = ["none", "label_flip", "sign_flip", "gaussian", "scaling", "backdoor"]
EPSILONS = [math.inf, 4.0, 1.0]


def eps_tag(e):
    return "inf" if e is None or (isinstance(e, float) and math.isinf(e)) else f"{float(e):g}"


def cells(out_dir, k, rounds, epsilons, attacks, seed=0, frac=0.3):
    for e in epsilons:
        for atk in attacks:
            yield FLConfig(
                dataset="mnist", num_clients=k, rounds=rounds, alpha=0.5, seed=seed,
                epsilon=(None if math.isinf(e) else e),
                aggregator="forensics",
                tau=float("inf"),                 # score everything, reject nothing
                attack=(None if atk == "none" else atk),
                attacker_frac=(0.0 if atk == "none" else frac),
                exp=EXP, out_dir=out_dir,
                cell=f"mnist_K{k}_eps{eps_tag(e)}_{atk}")


def harvest(run_dir):
    """(scores, labels) from one finished cell's rounds.jsonl + config.json."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    attackers = set(int(i) for i in cfg.get("attacker_ids", []))
    out = defaultdict(list)
    labels = []
    for line in open(os.path.join(run_dir, "rounds.jsonl")):
        if not line.strip():
            continue
        rec = json.loads(line)
        active = bool((rec.get("attack") or {}).get("active", False))
        for cid, sc in (rec.get("diag", {}).get("scores") or {}).items():
            if "combined" not in sc:              # health-flagged or too few clean peers
                continue
            for s in SCORES:
                out[s].append(float(sc.get(s, 0.0)))
            labels.append(1 if (active and int(cid) in attackers) else 0)
    return out, labels


def curve_stats(scores, labels):
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    finite = np.isfinite(s)
    s, y = s[finite], y[finite]
    if y.sum() == 0 or (y == 0).sum() == 0:
        return None                                # need both classes for a ROC
    fpr, tpr, thr = roc_curve(s, y)
    a = auc(fpr, tpr)

    j = int(np.argmax(tpr - fpr))                  # Youden
    tau_j = float(thr[j])
    ok = np.where(fpr <= 0.05)[0]
    tau_fpr5 = float(thr[ok[np.argmax(tpr[ok])]]) if ok.size else float("inf")

    t_j, f_j = rates_at(s, y, tau_j)
    t_5, f_5 = rates_at(s, y, tau_fpr5)
    return {"auc": float(a), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
            "tau_youden": tau_j, "tpr_at_youden": t_j, "fpr_at_youden": f_j,
            "tau_fpr5": tau_fpr5, "tpr_at_fpr5": t_5, "fpr_at_fpr5": f_5}


def plot(by_key, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib missing — skipping ROC plots", file=sys.stderr)
        return
    for key, data in by_key.items():
        raw = data.pop("_raw", None)
        if not raw:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        for s in SCORES:
            if s not in raw:
                continue
            fpr, tpr, _ = roc_curve(np.asarray(raw[s][0]), np.asarray(raw[s][1]))
            ax.plot(fpr, tpr, label=f"{s} (AUC={auc(fpr,tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("FPR (honest clients excluded)")
        ax.set_ylabel("TPR (attackers caught)")
        ax.set_title(f"FedVerify-Forensics ROC — {key}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"roc_{key.replace('|','_').replace('=','')}.png"), dpi=140)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="fedverify/results")
    ap.add_argument("--smoke", action="store_true", help="tiny, fast calibration")
    ap.add_argument("--num-clients", nargs="*", type=int, default=[10])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attacker-frac", type=float, default=0.3)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--harvest-only", action="store_true",
                    help="skip training; rebuild taus.json from cells already on disk")
    a = ap.parse_args(argv)

    ks = a.num_clients
    rounds, eps, attacks = a.rounds, EPSILONS, CAL_ATTACKS
    limit_train = limit_test = None
    if a.smoke:
        ks, rounds, eps = [6], 3, [math.inf]
        attacks = ["none", "sign_flip", "scaling"]
        limit_train, limit_test = 1200, 500

    todo = [] if a.harvest_only else [
        c for k in ks for c in cells(a.out_dir, k, rounds, eps, attacks,
                                     a.seed, a.attacker_frac)]
    print(f"calibration: {len(todo)} cells"
          + (" (harvest-only)" if a.harvest_only else ""), flush=True)
    for i, cfg in enumerate(todo, 1):
        if a.threads:
            cfg = cfg.replace(torch_threads=a.threads, cell=cfg.cell)
        done = os.path.exists(os.path.join(cfg.run_dir, "rounds.jsonl")) and \
            sum(1 for _ in open(os.path.join(cfg.run_dir, "rounds.jsonl"))) >= cfg.rounds
        if done and not a.force:
            print(f"  [{i}/{len(todo)}] {cfg.cell} (cached)", flush=True)
            continue
        print(f"  [{i}/{len(todo)}] {cfg.cell}", flush=True)
        run(cfg, limit_train=limit_train, limit_test=limit_test, progress=False)

    # ── pool per (epsilon, K) across attacks; clean cells supply the negatives ──
    #
    # Every completed cell UNDER results/<EXP> is harvested, not just the ones this
    # invocation ran. Two calibrations at different K (or a resumed one) would otherwise
    # each write a taus.json containing only their own keys, and the last to finish would
    # silently delete the other's entries.
    pooled = defaultdict(lambda: defaultdict(lambda: ([], [])))
    root = os.path.join(a.out_dir, EXP)
    seen = 0
    for dirpath, _dirs, files in os.walk(root):
        if "rounds.jsonl" not in files or "config.json" not in files:
            continue
        try:
            cfg_json = json.load(open(os.path.join(dirpath, "config.json")))
            c = cfg_json["config"]
            n_rounds = sum(1 for l in open(os.path.join(dirpath, "rounds.jsonl")) if l.strip())
            if n_rounds < int(c.get("rounds", 0) or 0):
                print(f"[skip] incomplete calibration cell "
                      f"({n_rounds}/{c.get('rounds')}): {dirpath}", file=sys.stderr)
                continue
            sc, lab = harvest(dirpath)
        except (OSError, KeyError, json.JSONDecodeError):
            print(f"[warn] unreadable calibration cell: {dirpath}", file=sys.stderr)
            continue
        seen += 1
        key = f"eps={eps_tag(c.get('epsilon'))}|K={c.get('num_clients')}"
        for sname in SCORES:
            if sname in sc:
                pooled[key][sname][0].extend(sc[sname])
                pooled[key][sname][1].extend(lab)
    print(f"harvested {seen} completed calibration cell(s) from {root}")

    out_dir = os.path.join(a.out_dir, "calibration")
    os.makedirs(out_dir, exist_ok=True)
    taus = {}
    for key, per_score in sorted(pooled.items()):
        entry = {"_raw": {s: per_score[s] for s in per_score}}
        for s in SCORES:
            if s not in per_score:
                continue
            st = curve_stats(*per_score[s])
            if st:
                entry[s] = st
        taus[key] = entry

    plot(taus, out_dir)                                    # pops _raw
    for k in taus:
        taus[k].pop("_raw", None)

    payload = {"scores": SCORES, "attacks": attacks, "rounds": rounds,
               "attacker_frac": a.attacker_frac, "seed": a.seed,
               "smoke": bool(a.smoke), "by_key": taus}
    path = os.path.join(out_dir, "taus.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {path}")
    for key, e in sorted(taus.items()):
        c = e.get("combined")
        if c:
            print(f"  {key}: combined AUC={c['auc']:.3f} "
                  f"tau_youden={c['tau_youden']:.2f} tau_fpr5={c['tau_fpr5']:.2f}")
        else:
            print(f"  {key}: no combined ROC (need both classes)")
    return 0


def load_tau(results_dir: str, epsilon, num_clients: int, policy: str = "tau_fpr5"):
    """Read tau for a cell from taus.json. Returns (tau, provenance_string).

    Falls back across keys deliberately loudly: an exp2 run must be able to say in its
    config.json exactly which calibration entry produced its threshold.
    """
    path = os.path.join(results_dir, "calibration", "taus.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python3 -m fedverify.analysis.calibrate` first — "
            "tau must come from calibration, never from a constant.")
    data = json.load(open(path))["by_key"]
    want = f"eps={eps_tag(epsilon)}|K={num_clients}"
    for key in (want, f"eps=inf|K={num_clients}", *sorted(data)):
        e = data.get(key, {}).get("combined")
        if e and math.isfinite(e.get(policy, float("inf"))):
            return float(e[policy]), f"{path}:{key}:{policy}"
    raise ValueError(f"no usable combined-score tau in {path} (wanted {want})")


if __name__ == "__main__":
    raise SystemExit(main())
