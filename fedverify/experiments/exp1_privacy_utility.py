"""Experiment 1 — privacy/utility trade-off (Table 1).

Grid: epsilon x num_clients x dataset x seed, alpha=0.5, 30 rounds, 1 local epoch.
Resumable: a cell whose rounds.jsonl already has `rounds` lines is skipped.

    python3 -m fedverify.experiments.exp1_privacy_utility --dry-run
    python3 -m fedverify.experiments.exp1_privacy_utility --smoke
    python3 -m fedverify.experiments.exp1_privacy_utility           # the full grid
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

from ..config import FLConfig
from ..core.runner import run

EXP = "exp1"
EPSILONS = [0.5, 1.0, 2.0, 4.0, 8.0, math.inf]
NUM_CLIENTS = [5, 10]
DATASETS = ["mnist", "fmnist"]
SEEDS = [0, 1, 2]
BASE = dict(alpha=0.5, rounds=30, local_epochs=1, batch_size=64, lr=0.01, momentum=0.9)


def grid(out_dir: str):
    for dataset in DATASETS:
        for k in NUM_CLIENTS:
            for eps in EPSILONS:
                for seed in SEEDS:
                    yield FLConfig(dataset=dataset, num_clients=k, seed=seed,
                                   epsilon=(None if math.isinf(eps) else eps),
                                   exp=EXP, out_dir=out_dir, **BASE)


def is_done(cfg: FLConfig) -> bool:
    p = os.path.join(cfg.run_dir, "rounds.jsonl")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        return sum(1 for _ in f) >= cfg.rounds


def _cell_cmd(cfg: FLConfig) -> list[str]:
    """The equivalent `python3 -m fedverify.core.runner ...` invocation for one cell."""
    cmd = [sys.executable, "-m", "fedverify.core.runner",
           "--dataset", cfg.dataset, "--num-clients", str(cfg.num_clients),
           "--rounds", str(cfg.rounds), "--local-epochs", str(cfg.local_epochs),
           "--batch-size", str(cfg.batch_size), "--lr", str(cfg.lr),
           "--momentum", str(cfg.momentum), "--alpha", str(cfg.alpha),
           "--seed", str(cfg.seed), "--exp", cfg.exp, "--out-dir", cfg.out_dir,
           "--delta", str(cfg.delta), "--max-grad-norm", str(cfg.max_grad_norm)]
    if cfg.torch_threads:
        cmd += ["--torch-threads", str(cfg.torch_threads)]
    if cfg.epsilon is not None:
        cmd += ["--epsilon", str(cfg.epsilon)]
    return cmd


def _run_parallel(todo, jobs: int, threads: int) -> None:
    """Run cells as independent subprocesses. Each cell owns its own output directory."""
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               PYTHONWARNINGS="ignore")
    done = {"n": 0}
    total = len(todo)

    todo = [c.replace(torch_threads=threads) for c in todo]

    def one(cfg):
        r = subprocess.run(_cell_cmd(cfg), env=env, cwd=os.getcwd(),
                           capture_output=True, text=True)
        done["n"] += 1
        tag = f"[{done['n']}/{total}]"
        if r.returncode != 0:
            print(f"{tag} FAILED {cfg.run_id}\n{r.stderr[-400:]}", file=sys.stderr, flush=True)
        else:
            tail = [l for l in r.stdout.strip().splitlines() if "acc=" in l]
            print(f"{tag} {tail[-1].strip() if tail else cfg.run_id}", flush=True)

    print(f"running {total} cells with {jobs} parallel jobs x {threads} threads", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as ex:   # threads only supervise subprocesses
        list(ex.map(one, todo))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="fedverify/results")
    ap.add_argument("--dry-run", action="store_true", help="list cells and exit")
    ap.add_argument("--smoke", action="store_true", help="one tiny cell only")
    ap.add_argument("--force", action="store_true", help="re-run completed cells")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help="run this many cells in parallel as subprocesses (cells are "
                         "independent and write to separate directories)")
    ap.add_argument("--threads", type=int, default=2,
                    help="torch/OMP threads per job (jobs x threads <= cores)")
    ap.add_argument("--dataset", nargs="*", default=None, help="filter datasets")
    ap.add_argument("--epsilon", nargs="*", type=float, default=None, help="filter epsilons")
    ap.add_argument("--seed", nargs="*", type=int, default=None, help="filter seeds")
    a = ap.parse_args(argv)
    # Opacus' secure-RNG notice and the full-backward-hook notice are expected here.
    warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")
    warnings.filterwarnings("ignore", message=".*Full backward hook.*")
    warnings.filterwarnings("ignore", message=".*Optimal order is the largest alpha.*")

    if a.smoke:
        cfg = FLConfig(dataset="mnist", num_clients=2, rounds=2, local_epochs=1,
                       batch_size=32, alpha=float("inf"), seed=0, epsilon=2.0,
                       exp=f"{EXP}_smoke", out_dir=a.out_dir)
        print(f"[smoke] {cfg.run_id}")
        out = run(cfg, limit_train=400, limit_test=200)
        fin = out["final"]
        pr = (fin.get("privacy") or {}).get("0", {})
        print(f"[smoke] acc={fin['test_acc']:.4f} macro_f1={fin['macro_f1']:.4f} "
              f"sigma={pr.get('sigma')} realized_eps={pr.get('realized_eps')}")
        return 0

    cells = list(grid(a.out_dir))
    if a.dataset:
        cells = [c for c in cells if c.dataset in a.dataset]
    if a.epsilon is not None:
        want = {("inf" if math.isinf(e) else float(e)) for e in a.epsilon}
        cells = [c for c in cells if (c.epsilon if c.epsilon is not None else "inf") in want]
    if a.seed:
        cells = [c for c in cells if c.seed in a.seed]

    todo = cells if a.force else [c for c in cells if not is_done(c)]
    print(f"grid: {len(cells)} cells, {len(cells)-len(todo)} already done, {len(todo)} to run", flush=True)
    if a.dry_run:
        for c in cells:
            print(f"  [{'done' if is_done(c) else 'todo'}] {c.run_id}")
        return 0

    t0 = time.time()
    if a.jobs > 1:
        _run_parallel(todo, a.jobs, a.threads)
    else:
        for i, cfg in enumerate(todo, 1):
            print(f"\n=== [{i}/{len(todo)}] {cfg.run_id} ===", flush=True)
            try:
                out = run(cfg)
                fin = out["final"]
                print(f"  acc={fin['test_acc']:.4f} macro_f1={fin['macro_f1']:.4f}")
            except Exception as e:                  # one bad cell must not kill the grid
                print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min. Build tables with:\n"
          f"  python3 -m fedverify.analysis.make_tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
