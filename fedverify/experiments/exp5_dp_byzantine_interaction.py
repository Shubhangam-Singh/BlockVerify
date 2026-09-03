"""Experiment 5 — does DP noise blind Byzantine screening? (Table 6)

FedVerify-Forensics separates clients by how their deltas differ from the consensus.
DP-SGD deliberately adds noise to every delta. Those two facts are in tension: as epsilon
tightens, honest deltas scatter, the robust scale widens, and an attacker has more room to
hide inside the honest band. Nobody can say a priori how sharp that trade-off is.

So this measures it: forensics only, attack in {scaling, backdoor} at fraction 0.2, sweeping
epsilon in {inf, 8, 4, 2, 1, 0.5} at K=10. Detection F1 and final accuracy are reported
against epsilon.

This is a genuine finding either way. If detection collapses at small epsilon, that is a
real limitation of combining DP with anomaly-based robustness and belongs in the paper as
such — it is NOT to be buried or explained away.

tau comes from calibration and is looked up PER EPSILON, because a threshold fitted on
noiseless deltas would be unfair to the DP cells: it would report DP as breaking detection
when in truth only the threshold was stale.

    python3 -m fedverify.analysis.calibrate            # required first
    python3 -m fedverify.experiments.exp5_dp_byzantine_interaction --smoke
    python3 -m fedverify.experiments.exp5_dp_byzantine_interaction --jobs 6 --threads 2
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

EXP = "exp5"
EPSILONS = [math.inf, 8.0, 4.0, 2.0, 1.0, 0.5]
ATTACKS = ["scaling", "backdoor"]
SEEDS = [0, 1, 2]
FRAC = 0.2
BASE = dict(dataset="mnist", num_clients=10, alpha=0.5, rounds=30, local_epochs=1,
            batch_size=64, lr=0.01, momentum=0.9, aggregator="forensics")


def grid(out_dir, results_dir):
    for attack in ATTACKS:
        for eps in EPSILONS:
            # per-epsilon tau: a threshold fitted on noiseless deltas would blame DP for
            # what is really a stale threshold.
            tau, src = load_tau(results_dir, eps, BASE["num_clients"])
            for seed in SEEDS:
                yield FLConfig(seed=seed, epsilon=(None if math.isinf(eps) else eps),
                               attack=attack, attacker_frac=FRAC, tau=tau, tau_source=src,
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
           "--aggregator", cfg.aggregator, "--attack", cfg.attack,
           "--attacker-frac", str(cfg.attacker_frac), "--tau", str(cfg.tau),
           "--tau-source", cfg.tau_source,
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
    ap.add_argument("--results-dir", default="fedverify/results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", "-j", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--attack", nargs="*", default=None)
    ap.add_argument("--seed", nargs="*", type=int, default=None)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")
    warnings.filterwarnings("ignore", message=".*Full backward hook.*")

    if a.smoke:
        from ..core.runner import run
        tau, src = load_tau(a.results_dir, math.inf, 6)
        cfg = FLConfig(dataset="mnist", num_clients=6, rounds=3, alpha=math.inf, seed=0,
                       aggregator="forensics", attack="scaling", attacker_frac=FRAC,
                       epsilon=4.0, tau=tau, tau_source=src,
                       exp=f"{EXP}_smoke", out_dir=a.out_dir)
        print(f"[smoke] {cfg.run_id}  tau={tau:.3f}")
        fin = run(cfg, limit_train=1200, limit_test=500, progress=False)["final"]
        at = fin["attack"]
        print(f"[smoke] acc={fin['test_acc']:.4f} attackers={at['attackers']} "
              f"rejected={at['rejected']} F1={at['f1']:.2f}")
        return 0

    cells = list(grid(a.out_dir, a.results_dir))
    if a.attack:
        cells = [c for c in cells if c.attack in a.attack]
    if a.seed:
        cells = [c for c in cells if c.seed in a.seed]
    if a.threads:
        cells = [c.replace(torch_threads=a.threads, cell=c.cell) for c in cells]

    todo = cells if a.force else [c for c in cells if not is_done(c)]
    print(f"grid: {len(cells)} cells, {len(cells)-len(todo)} already done, "
          f"{len(todo)} to run", flush=True)
    if a.dry_run:
        for c in cells:
            print(f"  [{'done' if is_done(c) else 'todo'}] {c.run_id}  tau={c.tau:.3f}")
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
                print(f"  acc={fin['test_acc']:.4f} F1={fin['attack']['f1']:.2f}")
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min. Build Table 6 with:\n"
          f"  python3 -m fedverify.analysis.make_tables --only table6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
