"""Every experiment of the FL-healthcare study, one command each (Revision 2 §6, README).

    python3 -m fedverify.survival.experiments --list
    python3 -m fedverify.survival.experiments methods --jobs 4
    python3 -m fedverify.survival.experiments poisoning --dry-run
    python3 -m fedverify.survival.experiments all --jobs 4

Experiments, in the order `all` runs them (later ones need earlier ones: poisoning reads the calibrated tau):

  methods        local / centralized / FedSGD / FedAvg logistic, RF and XGBoost on heart and TCGA, XGBoost AFT on
                 TCGA, the step-matched centralized XGBoost diagnostic row (Phase 9), and the heart full-mode
                 sensitivity runs
  dp             DP-FedAvg (logistic) at epsilon 1, 2, 4, 8 and inf, both datasets
  chain          the commit-only chain_* methods (mock anchor), both datasets, and the TCGA AFT one
  chain-pow      one chain_fedavg_lr run per dataset mined on the proof-of-work chain (Figs 7-8); heart is profiled
  calibration    the screening thresholds (analysis/calibrate_survival.py)
  chain-attacks  the reference paper's three chain attacks, 100 rounds, defences on and off, both datasets
  poisoning      every ours_* cell: clean, and each attack with and without screening, both datasets
  diagnostics    Phase 8: why TCGA central_xgb scores below local and federated (analysis/diagnose_central_xgb.py;
                 replays the XGBoost methods and checks the replay against their results, so it runs after them)

Every runner command pins seed 0, 5 repeats and 2 torch threads (the settings of the results on disk); results are
byte-reproducible at a fixed thread count, so --jobs (parallel processes) does not change them.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

PY = "python3"
COMMON = ["--seed", "0", "--repeats", "5", "--torch-threads", "2"]
DATASETS = ("heart", "tcga")
LR_ATTACKS = ("sign_flip", "scaling", "label_flip")
TREE_ATTACKS = ("label_flip_trees", "random_trees")


def runner(*args: str, common: Optional[list[str]] = None) -> list[str]:
    return [PY, "-m", "fedverify.survival.runner", *args, *(COMMON if common is None else common)]


def _methods() -> list[list[str]]:
    cmds = []
    for ds in DATASETS:
        for fam, methods in (("lr", ("local_lr", "central_lr", "fedsgd_lr", "fedavg_lr")),
                             ("rf", ("local_rf", "central_rf", "fed_rf")),
                             ("xgb", ("local_xgb", "central_xgb", "fedxgb_bagging", "central_xgb_steps"))):
            cmds += [runner("--dataset", ds, "--family", fam, "--method", m) for m in methods]
    cmds += [runner("--dataset", "tcga", "--family", "xgb", "--method", m, "--task", "survival")
             for m in ("local_xgb", "central_xgb", "fedxgb_bagging", "central_xgb_steps")]
    cmds += [runner("--dataset", "heart", "--family", f, "--method", m, "--heart-mode", "full")
             for f, m in (("lr", "fedavg_lr"), ("xgb", "fedxgb_bagging"))]
    return cmds


def _dp() -> list[list[str]]:
    return [runner("--dataset", ds, "--family", "lr", "--method", "dp_fedavg_lr", "--epsilon", e)
            for ds in DATASETS for e in ("1", "2", "4", "8", "inf")]


def _chain() -> list[list[str]]:
    cmds = [runner("--dataset", ds, "--family", f, "--method", m) for ds in DATASETS
            for f, m in (("lr", "chain_fedavg_lr"), ("rf", "chain_fed_rf"), ("xgb", "chain_fedxgb_bagging"))]
    return cmds + [runner("--dataset", "tcga", "--family", "xgb", "--method", "chain_fedxgb_bagging",
                          "--task", "survival")]


def _chain_pow() -> list[list[str]]:
    one = ["--seed", "0", "--repeats", "1", "--torch-threads", "2"]
    return [runner("--dataset", "heart", "--family", "lr", "--method", "chain_fedavg_lr", "--chain-backend", "local",
                   "--profile", common=one),
            runner("--dataset", "tcga", "--family", "lr", "--method", "chain_fedavg_lr", "--chain-backend", "local",
                   common=one)]


def _calibration() -> list[list[str]]:
    return [[PY, "-m", "fedverify.analysis.calibrate_survival", "--torch-threads", "2"]]


def _chain_attacks() -> list[list[str]]:
    return [[PY, "-m", "fedverify.survival.chain_attacks", "--attack", "all", "--rounds", "100", "--seed", "0",
             "--dataset", ds, *extra] for ds in DATASETS for extra in ([], ["--defence-off"])]


def _poisoning() -> list[list[str]]:
    cmds = []
    for ds in DATASETS:
        cmds.append(runner("--dataset", ds, "--family", "lr", "--method", "ours_lr"))
        cmds += [runner("--dataset", ds, "--family", "lr", "--method", "ours_lr", "--attack", a, *s)
                 for a in LR_ATTACKS for s in ([], ["--no-screening"])]
        for f in ("rf", "xgb"):
            cmds.append(runner("--dataset", ds, "--family", f, "--method", f"ours_{f}"))
            cmds += [runner("--dataset", ds, "--family", f, "--method", f"ours_{f}", "--attack", a, *s)
                     for a in TREE_ATTACKS for s in ([], ["--no-screening"])]
    cmds.append(runner("--dataset", "tcga", "--family", "xgb", "--method", "ours_xgb", "--task", "survival"))
    cmds += [runner("--dataset", "tcga", "--family", "xgb", "--method", "ours_xgb", "--task", "survival",
                    "--attack", a, *s) for a in TREE_ATTACKS for s in ([], ["--no-screening"])]
    return cmds


def _diagnostics() -> list[list[str]]:
    return [[PY, "-m", "fedverify.analysis.diagnose_central_xgb"]]


EXPERIMENTS = {"methods": _methods, "dp": _dp, "chain": _chain, "chain-pow": _chain_pow,
               "calibration": _calibration, "chain-attacks": _chain_attacks, "poisoning": _poisoning,
               "diagnostics": _diagnostics}


def commands(name: str) -> list[list[str]]:
    if name == "all":
        return [c for n in EXPERIMENTS for c in EXPERIMENTS[n]()]
    return EXPERIMENTS[name]()


def _run(cmd: list[str]) -> tuple[list[str], int, str]:
    exe = [sys.executable if cmd[0] == PY else cmd[0], *cmd[1:]]
    p = subprocess.run(exe, capture_output=True, text=True)
    lines = [l for l in (p.stdout + p.stderr).splitlines() if "repeat(s):" in l or l.startswith(("[", "wrote"))]
    return cmd, p.returncode, (lines[-1] if lines else (p.stderr.strip().splitlines() or [""])[-1])


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fedverify.survival.experiments", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?", choices=[*EXPERIMENTS, "all"])
    ap.add_argument("--list", action="store_true", help="list the experiments and their command counts")
    ap.add_argument("--dry-run", action="store_true", help="print the commands instead of running them")
    ap.add_argument("--jobs", type=int, default=1, help="parallel runner processes (each pins its torch threads)")
    a = ap.parse_args(argv)
    if a.list or not a.experiment:
        for n, fn in EXPERIMENTS.items():
            print(f"{n:14s} {len(fn()):3d} command(s)")
        return 0
    groups = list(EXPERIMENTS) if a.experiment == "all" else [a.experiment]
    if a.dry_run:
        for g in groups:
            for c in EXPERIMENTS[g]():
                print(shlex.join(c))
        return 0
    failed = 0
    for g in groups:                      # groups in dependency order; commands within a group in parallel
        t0 = time.time()
        cmds = EXPERIMENTS[g]()
        jobs = max(1, a.jobs) if g not in ("calibration", "chain-pow") else 1
        print(f"== {g}: {len(cmds)} command(s), {jobs} at a time", flush=True)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for cmd, code, last in pool.map(_run, cmds):
                failed += code != 0
                print(("  ok  " if code == 0 else "  FAIL") + f" {last}" + ("" if code == 0 else f"  <- {shlex.join(cmd)}"),
                      flush=True)
        print(f"== {g} done in {time.time() - t0:.0f} s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
