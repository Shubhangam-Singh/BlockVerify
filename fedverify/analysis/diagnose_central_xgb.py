"""Phase 8 diagnosis: why TCGA centralized XGBoost scores below local and federated XGBoost.

    python3 -m fedverify.analysis.diagnose_central_xgb
    python3 -m fedverify.analysis.diagnose_central_xgb --results-dir fedverify/results --repeats 5

Replays local_xgb, central_xgb and fedxgb_bagging on TCGA (survival:aft and the binary death indicator) with the
runner's exact folds and seeds, through the same fl_xgb.run_fold that produced the results, and measures per round:

  train_c       C-index on the fold's training rows (mean over fold-runs)
  held_c        C-index of the pooled out-of-fold predictions (per repeat, as the runner computes it; mean and std)
  train_auc / held_auc   the same for ROC-AUC (binary task)
  trees_total   trees in the ensemble
  sequential_steps       boosting steps on the path from the initial margin to a prediction: central_xgb adds
                K x N per round, fedxgb_bagging one averaged N-tree booster, local_xgb N
  margin_move   mean |change of the training margin| in that round

and, at the last round, the C-index over same-site and cross-site comparable pairs, plus two site-only predictors (the
training fold's death rate of the patient's site, nothing else) and per-site censoring, follow-up and death rates.
The final-round held-out values are checked against <results>/survival/<cell>/seed0/summary.json, so the replay is
known to be the run on disk. Nothing here changes a hyperparameter: it measures the configuration as it is.

Writes <results>/diagnostics/central_xgb_tcga.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ..survival import datasets, fl_xgb, runner
from ..survival.common import LOG_TIME, RISK
from ..survival.data import fold_data, make_folds, repeat_seeds
from ..survival.metrics import c_index, c_index_time, roc_auc

METHODS = ("local_xgb", "central_xgb", "fedxgb_bagging")
TASKS = ("survival", "binary")
CELL = {("survival", m): f"{m}_aft" for m in METHODS} | {("binary", m): m for m in METHODS}
REPRODUCE_TOL = 1e-9


def cidx(kind: str, time: np.ndarray, event: np.ndarray, score: np.ndarray) -> Optional[float]:
    return c_index_time(time, event, score) if kind == LOG_TIME else c_index(time, event, score)


def pair_split(time: np.ndarray, event: np.ndarray, score: np.ndarray, site: np.ndarray, kind: str) -> dict:
    """Harrell's C over comparable pairs (i died before j's time), all / same-site / cross-site. A LOG_TIME score is a
    predicted time (higher = longer survival); a RISK score is the reverse. Ties in the score count one half."""
    s = np.asarray(score, float) if kind == LOG_TIME else -np.asarray(score, float)
    t, e, site = np.asarray(time, float), np.asarray(event, int), np.asarray(site)
    i, j = np.where((t[:, None] < t[None, :]) & (e[:, None] == 1))
    conc = np.where(s[i] < s[j], 1.0, np.where(s[i] == s[j], 0.5, 0.0))
    same = site[i] == site[j]
    out = {}
    for name, m in (("all", np.ones_like(same)), ("within_site", same), ("cross_site", ~same)):
        out[name] = {"c": float(conc[m].mean()) if m.any() else None, "pairs": int(m.sum())}
    return out


def steps_per_round(method: str, K: int, N: int) -> int:
    """Sequential boosting steps one round adds to every prediction (fl_xgb.run_fold)."""
    return {"central_xgb": K * N, "fedxgb_bagging": N, "local_xgb": N}[method]


def config(method: str, task: str, repeats: int):
    argv = ["--dataset", "tcga", "--family", "xgb", "--method", method, "--seed", "0", "--repeats", str(repeats)]
    if task == "survival":
        argv += ["--task", "survival"]
    return runner.build_config(argv)[0]


def replay(method: str, task: str, repeats: int) -> dict:
    cfg = config(method, task, repeats)
    loaded = datasets.load(cfg)
    cohort, X = loaded.cohort, loaded.X
    kind = LOG_TIME if task == "survival" else RISK
    R, n, K, N = cfg.rounds, cohort.n, len(cfg.hospitals), cfg.xgb_trees_per_round
    held, held_auc = np.full((repeats, R), np.nan), np.full((repeats, R), np.nan)
    train = [[] for _ in range(R)]
    train_auc = [[] for _ in range(R)]
    move = [[] for _ in range(R)]
    trees = [[] for _ in range(R)]
    final = []
    for r, seed in enumerate(repeat_seeds(cfg.seed, repeats)):
        oof = np.full((R, n), np.nan)
        for k, (tr, te) in enumerate(make_folds(cohort, seed, cfg.folds)):
            fd = fold_data(cohort, X, tr, te)
            prev = None
            for out in fl_xgb.run_fold(cfg, fd, seed, k):
                t = out.round - 1
                oof[t, te] = out.test_risk
                ok = ~np.isnan(out.train_risk)
                train[t].append(cidx(kind, fd.time_train[ok], fd.event_train[ok], out.train_risk[ok]))
                if kind == RISK:
                    train_auc[t].append(roc_auc(fd.event_train[ok], out.train_risk[ok]))
                m = out.train_risk if kind == LOG_TIME else np.log(out.train_risk / (1.0 - out.train_risk))
                if prev is not None:
                    move[t].append(float(np.nanmean(np.abs(m - prev))))
                prev = m
                trees[t].append(int(out.extra["trees_total"]))
        for t in range(R):
            ok = ~np.isnan(oof[t])
            held[r, t] = cidx(kind, cohort.time_days[ok], cohort.event[ok], oof[t, ok])
            if kind == RISK:
                held_auc[r, t] = roc_auc(cohort.event[ok], oof[t, ok])
        ok = ~np.isnan(oof[-1])
        final.append(pair_split(cohort.time_days[ok], cohort.event[ok], oof[-1, ok], cohort.hospital[ok], kind))
    sp = steps_per_round(method, K, N)
    per_round = []
    for t in range(R):
        per_round.append({"round": t + 1, "sequential_steps": sp * (t + 1), "trees_total": int(np.mean(trees[t])),
                          "train_c": float(np.mean([v for v in train[t] if v is not None])),
                          "held_c": float(np.mean(held[:, t])), "held_c_std": float(np.std(held[:, t])),
                          "train_auc": float(np.mean(train_auc[t])) if kind == RISK else None,
                          "held_auc": float(np.mean(held_auc[:, t])) if kind == RISK else None,
                          "held_auc_std": float(np.std(held_auc[:, t])) if kind == RISK else None,
                          "margin_move": float(np.mean(move[t])) if move[t] else None})
    pairs = {k: {"c": float(np.mean([f[k]["c"] for f in final])), "pairs": int(np.mean([f[k]["pairs"] for f in final]))}
             for k in final[0]}
    return {"method": method, "task": task, "cell": CELL[(task, method)], "K": K, "N": N, "rounds": R,
            "learning_rate": cfg.xgb_learning_rate, "max_depth": cfg.xgb_max_depth, "subsample": cfg.xgb_subsample,
            "steps_per_round": sp, "per_round": per_round, "final_pairs": pairs,
            "final_held_c_per_repeat": [float(v) for v in held[:, -1]],
            "final_held_auc_per_repeat": [float(v) for v in held_auc[:, -1]] if kind == RISK else None}


def site_facts(repeats: int) -> tuple[dict, dict]:
    """Per-site death, censoring and follow-up; and the site-only predictor (training-fold death rate of the site)."""
    cfg = config("local_xgb", "binary", repeats)
    c = datasets.load(cfg).cohort
    sites = {}
    for h in cfg.hospitals:
        m = c.hospital == h
        t, e = c.time_days[m], c.event[m]
        sites[h] = {"patients": int(m.sum()), "deaths": int(e.sum()), "death_rate": float(e.mean()),
                    "censored_rate": float(1.0 - e.mean()),
                    "median_followup_censored_days": float(np.median(t[e == 0])),
                    "followup_under_1y": float((t < 365.25).mean())}
    cs, aucs = [], []
    for seed in repeat_seeds(cfg.seed, repeats):
        oof = np.full(c.n, np.nan)
        for tr, te in make_folds(c, seed, cfg.folds):
            rate = {h: float(c.event[tr][c.hospital[tr] == h].mean()) for h in cfg.hospitals}
            oof[te] = [rate[h] for h in c.hospital[te]]
        cs.append(c_index(c.time_days, c.event, oof))
        aucs.append(roc_auc(c.event, oof))
    return sites, {"c_index": [float(np.mean(cs)), float(np.std(cs))], "roc_auc": [float(np.mean(aucs)),
                                                                                 float(np.std(aucs))]}


def check_reproduces(results: Path, run: dict) -> dict:
    """The replay's final-round pooled values against the summary on disk."""
    s = json.loads((results / "survival" / run["cell"] / "seed0" / "summary.json").read_text(encoding="utf-8"))
    got = {"c_index": run["final_held_c_per_repeat"]}
    if run["final_held_auc_per_repeat"] is not None:
        got["roc_auc"] = run["final_held_auc_per_repeat"]
    diffs = {k: max(abs(a - rp["pooled"][k]) for a, rp in zip(v, s["repeats"])) for k, v in got.items()}
    return {"max_abs_diff": diffs, "ok": all(d <= REPRODUCE_TOL for d in diffs.values())}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fedverify.analysis.diagnose_central_xgb", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="fedverify/results")
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args(argv)
    results = Path(a.results_dir)
    sites, site_only = site_facts(a.repeats)
    out = {"generated_by": "python3 -m fedverify.analysis.diagnose_central_xgb", "dataset": "tcga",
           "repeats": a.repeats, "sites": sites, "site_only": site_only, "runs": []}
    ok = True
    for task in TASKS:
        for method in METHODS:
            print(f"replaying {task} {method}", flush=True)
            run = replay(method, task, a.repeats)
            run["reproduces_results"] = check_reproduces(results, run)
            ok &= run["reproduces_results"]["ok"]
            out["runs"].append(run)
    path = results / "diagnostics" / "central_xgb_tcga.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}; replay reproduces the results on disk: {ok}")
    if not ok:
        print("the replay does not match the results on disk: re-run the methods first", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
