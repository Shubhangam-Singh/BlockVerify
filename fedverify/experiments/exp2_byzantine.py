"""Experiment 2 — Byzantine robustness (Table 2 / 2b).

Grid: attack x attacker_frac x aggregator x epsilon x seed, K=10, alpha=0.5.
`attack=none` ignores attacker_frac (there is only one clean configuration), so the grid
is 5*3*6*2*3 + 1*6*2*3 = 576 cells rather than 648.

tau for the forensics aggregator is READ FROM results/calibration/taus.json and recorded
in config.json together with its provenance (`tau_source`). There is no default: if
calibration has not been run, an exp2 cell using forensics fails loudly rather than
silently inventing a threshold.

    python3 -m fedverify.analysis.calibrate                      # FIRST: produce taus.json
    python3 -m fedverify.experiments.exp2_byzantine --dry-run
    python3 -m fedverify.experiments.exp2_byzantine --smoke
    python3 -m fedverify.experiments.exp2_byzantine --jobs 8 --threads 2
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

from ..analysis.calibrate import load_tau
from ..config import FLConfig

EXP = "exp2"
ATTACKS = ["none", "label_flip", "sign_flip", "gaussian", "scaling", "backdoor"]
FRACS = [0.1, 0.2, 0.3]
AGGREGATORS = ["fedavg", "krum", "multikrum", "trimmed_mean", "median", "forensics"]
EPSILONS = [math.inf, 4.0]
SEEDS = [0, 1, 2]
BASE = dict(dataset="mnist", num_clients=10, alpha=0.5, rounds=30, local_epochs=1,
            batch_size=64, lr=0.01, momentum=0.9)


def grid(out_dir, results_dir):
    for attack in ATTACKS:
        fracs = [0.0] if attack == "none" else FRACS
        for frac in fracs:
            for agg in AGGREGATORS:
                for eps in EPSILONS:
                    for seed in SEEDS:
                        tau, src = (None, None)
                        if agg == "forensics":
                            tau, src = load_tau(results_dir, eps, BASE["num_clients"])
                        yield FLConfig(
                            seed=seed, epsilon=(None if math.isinf(eps) else eps),
                            attack=(None if attack == "none" else attack),
                            attacker_frac=frac, aggregator=agg,
                            tau=tau, tau_source=src,
                            exp=EXP, out_dir=out_dir, **BASE)


def is_done(cfg) -> bool:
    p = os.path.join(cfg.run_dir, "rounds.jsonl")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        return sum(1 for _ in f) >= cfg.rounds


def _cell_cmd(cfg) -> list:
    cmd = [sys.executable, "-m", "fedverify.core.runner",
           "--dataset", cfg.dataset, "--num-clients", str(cfg.num_clients),
           "--rounds", str(cfg.rounds), "--local-epochs", str(cfg.local_epochs),
           "--batch-size", str(cfg.batch_size), "--lr", str(cfg.lr),
           "--momentum", str(cfg.momentum), "--alpha", str(cfg.alpha),
           "--seed", str(cfg.seed), "--exp", cfg.exp, "--out-dir", cfg.out_dir,
           "--aggregator", cfg.aggregator, "--attacker-frac", str(cfg.attacker_frac),
           "--delta", str(cfg.delta), "--max-grad-norm", str(cfg.max_grad_norm)]
    if cfg.epsilon is not None:
        cmd += ["--epsilon", str(cfg.epsilon)]
    if cfg.attack:
        cmd += ["--attack", cfg.attack]
    if cfg.tau is not None:
        cmd += ["--tau", str(cfg.tau)]
    if cfg.tau_source:
        cmd += ["--tau-source", cfg.tau_source]
    if cfg.torch_threads:
        cmd += ["--torch-threads", str(cfg.torch_threads)]
    return cmd


def _run_parallel(todo, jobs, threads):
    total = len(todo)
    print(f"running {total} cells with {jobs} parallel jobs x {threads} threads", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))

    def one(i_cfg):
        i, cfg = i_cfg
        tag = f"[{i}/{total}]"
        r = subprocess.run(_cell_cmd(cfg), capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print(f"{tag} FAILED {cfg.run_id}\n{r.stderr[-400:]}", file=sys.stderr, flush=True)
        else:
            tail = [l for l in r.stdout.strip().splitlines() if l.strip()]
            print(f"{tag} {tail[-1].strip() if tail else cfg.run_id}", flush=True)

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(one, enumerate(todo, 1)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="fedverify/results")
    ap.add_argument("--results-dir", default="fedverify/results",
                    help="where calibration/taus.json lives")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", "-j", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--attack", nargs="*", default=None)
    ap.add_argument("--aggregator", nargs="*", default=None)
    ap.add_argument("--attacker-frac", nargs="*", type=float, default=None)
    ap.add_argument("--seed", nargs="*", type=int, default=None)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")
    warnings.filterwarnings("ignore", message=".*Full backward hook.*")

    if a.smoke:
        from ..core.runner import run
        tau, src = load_tau(a.results_dir, math.inf, 6)
        cfg = FLConfig(dataset="mnist", num_clients=6, rounds=3, alpha=math.inf, seed=0,
                       aggregator="forensics", attack="scaling", attacker_frac=0.3,
                       tau=tau, tau_source=src, exp=f"{EXP}_smoke", out_dir=a.out_dir)
        print(f"[smoke] {cfg.run_id}  tau={tau:.3f} from {src}")
        out = run(cfg, limit_train=1200, limit_test=500, progress=False)
        fin = out["final"]; at = fin["attack"]
        print(f"[smoke] acc={fin['test_acc']:.4f} attackers={at['attackers']} "
              f"rejected={at['rejected']} P={at['precision']:.2f} R={at['recall']:.2f} "
              f"F1={at['f1']:.2f} FER={at['false_exclusion_rate']:.2f}")
        return 0

    cells = list(grid(a.out_dir, a.results_dir))
    if a.attack:
        want = {(None if x == "none" else x) for x in a.attack}
        cells = [c for c in cells if c.attack in want]
    if a.aggregator:
        cells = [c for c in cells if c.aggregator in a.aggregator]
    if a.attacker_frac is not None:
        cells = [c for c in cells if c.attacker_frac in a.attacker_frac or c.attack is None]
    if a.seed:
        cells = [c for c in cells if c.seed in a.seed]
    if a.threads:
        cells = [c.replace(torch_threads=a.threads, cell=c.cell) for c in cells]

    todo = cells if a.force else [c for c in cells if not is_done(c)]
    print(f"grid: {len(cells)} cells, {len(cells)-len(todo)} already done, "
          f"{len(todo)} to run", flush=True)
    if a.dry_run:
        for c in cells:
            print(f"  [{'done' if is_done(c) else 'todo'}] {c.run_id}")
        return 0

    t0 = time.time()
    if a.jobs > 1:
        _run_parallel(todo, a.jobs, a.threads)
    else:
        from ..core.runner import run
        for i, cfg in enumerate(todo, 1):
            print(f"\n=== [{i}/{len(todo)}] {cfg.run_id} ===", flush=True)
            try:
                fin = run(cfg)["final"]
                print(f"  acc={fin['test_acc']:.4f}")
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min. Build tables with:\n"
          f"  python3 -m fedverify.analysis.make_tables --only table2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
