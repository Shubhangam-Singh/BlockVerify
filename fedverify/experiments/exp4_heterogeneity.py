"""Experiment 4 — heterogeneity x privacy (Table 4).

Two halves that answer different questions:

  MNIST    alpha in {0.1, 0.5, 1.0, inf} x eps in {inf, 4, 1}, K=10.
           Heterogeneity is SYNTHESISED by the Dirichlet draw, so alpha is a dial and the
           degradation curve is clean.
  MIT-BIH  K in {5, 10} x eps in {inf, 4, 1}, patient split.
           Heterogeneity is INHERITED — each client holds different patients — so there is
           no alpha to sweep. This is the honest version of the same question.

Primary metric for MIT-BIH is macro-F1: 89.5% of beats are class N, so a model that
always answers N scores 89.5% accuracy and 18.9% macro-F1. Table 4 reports both, with
accuracy present only so that gap is visible.

    python3 -m fedverify.experiments.exp4_heterogeneity --dry-run
    python3 -m fedverify.experiments.exp4_heterogeneity --smoke
    python3 -m fedverify.experiments.exp4_heterogeneity --jobs 6 --threads 2
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

EXP = "exp4"
ALPHAS = [0.1, 0.5, 1.0, math.inf]
EPSILONS = [math.inf, 4.0, 1.0]
MITBIH_K = [5, 10]
SEEDS = [0, 1, 2]
BASE = dict(rounds=30, local_epochs=1, batch_size=64, lr=0.01, momentum=0.9)


def grid(out_dir):
    for eps in EPSILONS:
        e = None if math.isinf(eps) else eps
        for alpha in ALPHAS:                       # MNIST: sweep synthetic heterogeneity
            for seed in SEEDS:
                yield FLConfig(dataset="mnist", num_clients=10, alpha=alpha, seed=seed,
                               epsilon=e, exp=EXP, out_dir=out_dir, **BASE)
        for k in MITBIH_K:                          # MIT-BIH: heterogeneity is inherited
            for seed in SEEDS:
                yield FLConfig(dataset="mitbih", num_clients=k, seed=seed,
                               epsilon=e, exp=EXP, out_dir=out_dir, **BASE)


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
           "--delta", str(cfg.delta), "--max-grad-norm", str(cfg.max_grad_norm)]
    if cfg.epsilon is not None:
        cmd += ["--epsilon", str(cfg.epsilon)]
    if cfg.torch_threads:
        cmd += ["--torch-threads", str(cfg.torch_threads)]
    return cmd


def _run_parallel(todo, jobs, threads):
    total = len(todo)
    print(f"running {total} cells with {jobs} parallel jobs x {threads} threads", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))

    def one(i_cfg):
        i, cfg = i_cfg
        r = subprocess.run(_cell_cmd(cfg), capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print(f"[{i}/{total}] FAILED {cfg.run_id}\n{r.stderr[-400:]}",
                  file=sys.stderr, flush=True)
        else:
            tail = [l for l in r.stdout.strip().splitlines() if l.strip()]
            print(f"[{i}/{total}] {tail[-1].strip() if tail else cfg.run_id}", flush=True)

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(one, enumerate(todo, 1)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="fedverify/results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", "-j", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dataset", nargs="*", default=None)
    ap.add_argument("--seed", nargs="*", type=int, default=None)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")
    warnings.filterwarnings("ignore", message=".*Full backward hook.*")

    if a.smoke:
        from ..core.runner import run
        cfg = FLConfig(dataset="mitbih", num_clients=5, rounds=2, seed=0,
                       exp=f"{EXP}_smoke", out_dir=a.out_dir, **{**BASE, "rounds": 2})
        print(f"[smoke] {cfg.run_id}")
        out = run(cfg, progress=False)
        fin = out["final"]
        pr = out["partition_report"]
        print(f"[smoke] acc={fin['test_acc']:.4f} macro_f1={fin['macro_f1']:.4f}")
        print(f"[smoke] hospitals: {pr['patients_per_client']}")
        print(f"[smoke] sizes={pr['sizes']} mean_kl={pr['mean_kl_to_uniform']:.3f}")
        return 0

    cells = list(grid(a.out_dir))
    if a.dataset:
        cells = [c for c in cells if c.dataset in a.dataset]
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
                print(f"  acc={fin['test_acc']:.4f} macro_f1={fin['macro_f1']:.4f}")
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min. Build Table 4 with:\n"
          f"  python3 -m fedverify.analysis.make_tables --only table4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
