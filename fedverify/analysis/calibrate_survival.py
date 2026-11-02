"""Calibrate the ours_* screening thresholds of the survival experiments from data (Revision 2 §4B). Never guess tau.

The same procedure as analysis/calibrate.py, for K = 4 (heart) and K = 5 (TCGA):
  1. Run short cells of each ours_* method with tau = inf: every hospital is SCORED every round and none is rejected
     on the threshold (hard health flags still fire; they are not calibrated).
  2. Label every per-hospital score 1 if that hospital was the round's attacker, else 0. Clean cells give negatives;
     attacked cells give the attacker's positives and the honest hospitals' negatives.
  3. ROC per sub-score and for the score the screen compares with tau ("combined"), with the pure-numpy ROC of
     evaluation/eval_lib.py, through analysis/calibrate.py's curve_stats (AUC, Youden tau, tau at FPR <= 5%).
  4. Write results/calibration/taus_survival.json keyed by (dataset, family, epsilon, K); the AFT task is family
     xgb_aft. The runner reads tau from there (policy tau_fpr5 by default) and records tau and tau_source.

Cells per (dataset, family): one clean cell, and for every attack of the family one cell per hospital as the pinned
attacker, so every hospital is seen in both roles. One repeat at a separate seed (default 1000), so calibration never
shares a CV split with the evaluation seeds 0-4; only server-side screening scores are used, never test metrics.

    python3 -m fedverify.analysis.calibrate_survival --smoke     # 2 folds x 3 rounds -> taus_survival_smoke.json
    python3 -m fedverify.analysis.calibrate_survival             # every family, K=4 heart and K=5 TCGA -> taus_survival.json

A smoke file is never read by default: a run uses it only when pointed at it (--taus), and its tau_source says SMOKE.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

from ..survival import runner
from ..survival.config import DATASET_HOSPITALS, FAMILY_ATTACKS, SurvivalConfig
from ..survival.screening import tau_key
from .calibrate import curve_stats

SCORE_ONLY = "calibration: tau = inf, every hospital scored and none rejected on tau (hard health flags still fire)"
FAMILIES: dict[str, tuple[str, ...]] = {"heart": ("lr", "rf", "xgb"), "tcga": ("lr", "rf", "xgb", "xgb_aft")}
SCORES: dict[str, tuple[str, ...]] = {"lr": ("s_norm", "s_dir", "s_coord", "combined"),   # Forensics' sub-scores
                                      "rf": ("combined",), "xgb": ("combined",), "xgb_aft": ("combined",)}
CAL_SEED = 1000
DEFAULT_EPSILONS = (4.0,)       # ours_lr's default; add inf with --epsilons 4 inf


def family_task(family_key: str) -> tuple[str, str]:
    return ("xgb", "survival") if family_key == "xgb_aft" else (family_key, "binary")


def cells(dataset: str, family_key: str, epsilon: Optional[float], rounds: int, folds: int, seed: int, exp: str,
          out_dir: str, torch_threads: Optional[int] = None) -> Iterator[SurvivalConfig]:
    family, task = family_task(family_key)
    base = dict(dataset=dataset, family=family, method=f"ours_{family}", task=task, seed=seed, repeats=1,
                folds=folds, rounds=rounds, tau=math.inf, tau_source=SCORE_ONLY, exp=exp, out_dir=out_dir,
                epsilon=epsilon if family == "lr" else None, torch_threads=torch_threads)
    yield SurvivalConfig(**base)
    for attack in FAMILY_ATTACKS[family]:
        for h in DATASET_HOSPITALS[dataset]:
            yield SurvivalConfig(**base, attack=attack, attacker=h)


def complete(run_dir: Path, rounds: int, folds: int) -> bool:
    """A finished cell: summary.json written and every fold has all its rounds."""
    if not (run_dir / "summary.json").exists():
        return False
    for k in range(folds):
        p = run_dir / "repeat0" / f"fold{k}" / "rounds.jsonl"
        if not p.exists() or sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip()) < rounds:
            return False
    return True


def harvest(root: Path) -> tuple[dict, int]:
    """{key: {score: (values, labels)}} from every complete cell under root, and the number of cells used."""
    pooled: dict = defaultdict(lambda: defaultdict(lambda: ([], [])))
    used = 0
    for cfg_path in sorted(root.rglob("config.json")):
        run_dir = cfg_path.parent
        if not (run_dir / "summary.json").exists():
            continue                                    # fold-level config.json files, or an unfinished cell
        c = json.loads(cfg_path.read_text(encoding="utf-8"))["config"]
        if not complete(run_dir, int(c["rounds"]), int(c["folds"])):
            print(f"[skip] incomplete calibration cell: {run_dir}", file=sys.stderr)
            continue
        fam = "xgb_aft" if (c["family"] == "xgb" and c["task"] == "survival") else c["family"]
        eps = None if c.get("epsilon") in (None, "inf") else float(c["epsilon"])
        key = tau_key(c["dataset"], fam, eps, len(c["hospitals"]))
        for rounds_file in sorted(run_dir.glob("repeat*/fold*/rounds.jsonl")):
            for line in rounds_file.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                attackers = set((rec.get("attack") or {}).get("attackers", []))
                for h, sc in ((rec.get("diag") or {}).get("scores") or {}).items():
                    if "combined" not in sc:            # health-flagged, or too few healthy peers
                        continue
                    for s in SCORES[fam]:
                        if sc.get(s) is not None:
                            pooled[key][s][0].append(float(sc[s]))
                            pooled[key][s][1].append(1 if h in attackers else 0)
        used += 1
    return pooled, used


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fedverify.analysis.calibrate_survival", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="fedverify/results")
    ap.add_argument("--smoke", action="store_true", help="2 folds x 3 rounds; writes taus_survival_smoke.json")
    ap.add_argument("--datasets", nargs="*", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--families", nargs="*", default=None, help="subset of lr rf xgb xgb_aft")
    ap.add_argument("--epsilons", nargs="*", type=float, default=list(DEFAULT_EPSILONS), help="logistic only")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=CAL_SEED)
    ap.add_argument("--torch-threads", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-run cells that are already complete")
    ap.add_argument("--harvest-only", action="store_true", help="rebuild the file from cells already on disk")
    a = ap.parse_args(argv)
    rounds, folds = (3, 2) if a.smoke else (a.rounds, a.folds)
    root_name = "calib_survival_smoke" if a.smoke else "calib_survival"
    out_name = "taus_survival_smoke.json" if a.smoke else "taus_survival.json"

    todo = []
    for ds in a.datasets:
        for fam in FAMILIES[ds]:
            if a.families and fam not in a.families:
                continue
            for eps in (a.epsilons if fam == "lr" else [None]):
                # one folder per dataset: heart and TCGA cells share method names (results/<root>/<dataset>/<cell>)
                todo += list(cells(ds, fam, None if eps is None or math.isinf(eps) else eps, rounds, folds, a.seed,
                                   f"{root_name}/{ds}", a.out_dir, a.torch_threads))
    if a.harvest_only:
        todo = []
    print(f"calibration: {len(todo)} cells ({rounds} rounds x {folds} folds, seed {a.seed})", flush=True)
    plan_cache: dict = {}                               # DP sigma per hospital size, shared by every logistic cell
    for i, cfg in enumerate(todo, 1):
        if complete(Path(cfg.run_dir), rounds, folds) and not a.force:
            print(f"  [{i}/{len(todo)}] {cfg.exp}/{cfg.cell} (cached)", flush=True)
            continue
        print(f"  [{i}/{len(todo)}] {cfg.exp}/{cfg.cell}", flush=True)
        runner.run(cfg, verbose=False, plan_cache=plan_cache)

    root = Path(a.out_dir) / root_name
    pooled, used = harvest(root)
    by_key = {}
    for key, per_score in sorted(pooled.items()):
        entry = {}
        for s, (vals, labels) in per_score.items():
            stats = curve_stats(vals, labels)
            if stats:
                entry[s] = stats
        by_key[key] = entry
    payload = {"about": ("ours_* screening thresholds (analysis/calibrate_survival.py). 'combined' is the score the "
                         "screen compares with tau: Forensics' combined robust z (logistic) or the probe-distance "
                         "robust z (trees). Labels: 1 = the round's attacker."),
               "seed": a.seed, "rounds": rounds, "folds": folds, "smoke": bool(a.smoke), "cells_harvested": used,
               "attacks": {fam: list(FAMILY_ATTACKS[family_task(fam)[0]]) for fam in SCORES},
               "policy_default": "tau_fpr5", "by_key": by_key}
    out = Path(a.out_dir) / "calibration" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"harvested {used} cell(s) from {root}\nwrote {out}")
    for key, e in by_key.items():
        c = e.get("combined")
        print(f"  {key}: " + (f"combined AUC={c['auc']:.3f} tau_fpr5={c['tau_fpr5']:.3f} "
                              f"(TPR {c['tpr_at_fpr5']:.2f} at FPR {c['fpr_at_fpr5']:.2f})  tau_youden="
                              f"{c['tau_youden']:.3f}" if c else "no combined ROC (needs both classes)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
