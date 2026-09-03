"""Experiment 3 — cost of the chain commitment layer (Table 3).

Grid: K in {5, 10, 20} x chain backend, 10 rounds, no DP (the point is commitment cost,
not privacy cost). Per round we record digest_ms, merkle_ms, anchor_ms, bytes_committed
and aggregate_ms; Table 3 reports mean +/- std across rounds.

Why aggregate_ms sits in the same table: the honest question is not "how many milliseconds
does anchoring take" in isolation, but "how much does it add to a round". Reporting the
aggregation cost beside it makes the overhead ratio readable straight off the table.

    python3 -m fedverify.experiments.exp3_chain_overhead --dry-run
    python3 -m fedverify.experiments.exp3_chain_overhead --smoke
    python3 -m fedverify.experiments.exp3_chain_overhead                 # mock + local
    python3 -m fedverify.experiments.exp3_chain_overhead --algorand      # + one real round
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

from ..config import FLConfig
from ..core.runner import run

EXP = "exp3"
NUM_CLIENTS = [5, 10, 20]
BACKENDS = ["mock", "local"]          # "algorand" only with --algorand (costs real ALGO)
SEEDS = [0]
BASE = dict(dataset="mnist", alpha=0.5, rounds=10, local_epochs=1,
            batch_size=64, lr=0.01, momentum=0.9, commit=True, checkpoint_every=5)


def grid(out_dir: str, backends=None, seeds=None, clients=None):
    for backend in (backends or BACKENDS):
        for k in (clients or NUM_CLIENTS):
            for seed in (seeds or SEEDS):
                yield FLConfig(num_clients=k, seed=seed, chain_backend=backend,
                               cell=f"mnist_K{k}_{backend}",
                               exp=EXP, out_dir=out_dir, **BASE)


def is_done(cfg: FLConfig) -> bool:
    p = os.path.join(cfg.run_dir, "rounds.jsonl")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        return sum(1 for _ in f) >= cfg.rounds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="fedverify/results")
    ap.add_argument("--dry-run", action="store_true", help="list cells and exit")
    ap.add_argument("--smoke", action="store_true", help="one tiny cell only")
    ap.add_argument("--force", action="store_true", help="re-run completed cells")
    ap.add_argument("--threads", type=int, default=None, help="pin torch intra-op threads")
    ap.add_argument("--algorand", action="store_true",
                    help="ALSO run the algorand backend (real testnet txns, real fees)")
    ap.add_argument("--btc-checkpoint", action="store_true",
                    help="use the real Bitcoin OP_RETURN path for checkpoints")
    ap.add_argument("--num-clients", nargs="*", type=int, default=None)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")
    warnings.filterwarnings("ignore", message=".*Full backward hook.*")

    if a.smoke:
        cfg = FLConfig(dataset="mnist", num_clients=3, rounds=2, alpha=float("inf"),
                       seed=0, commit=True, chain_backend="local", checkpoint_every=2,
                       exp=f"{EXP}_smoke", out_dir=a.out_dir)
        print(f"[smoke] {cfg.run_id}")
        out = run(cfg, limit_train=400, limit_test=200)
        c = out["final"]["commit"]
        print(f"[smoke] root={c['root'][:16]} leaves={c['leaf_count']} txid={c['txid'][:16]} "
              f"digest_ms={c['digest_ms']:.2f} merkle_ms={c['merkle_ms']:.2f} "
              f"anchor_ms={c['anchor_ms']:.1f}")
        print(f"[smoke] lineage: {len([r for r in out['lineage'] if r])} round roots, "
              f"{len(out['checkpoints'])} checkpoint(s)")
        return 0

    backends = list(BACKENDS) + (["algorand"] if a.algorand else [])
    cells = list(grid(a.out_dir, backends=backends, clients=a.num_clients))
    todo = cells if a.force else [c for c in cells if not is_done(c)]
    print(f"grid: {len(cells)} cells, {len(cells)-len(todo)} already done, "
          f"{len(todo)} to run", flush=True)
    if a.dry_run:
        for c in cells:
            print(f"  [{'done' if is_done(c) else 'todo'}] {c.run_id}")
        return 0

    # Deliberately serial: anchor_ms is a LATENCY measurement, and running cells in
    # parallel would contaminate it with scheduler queueing.
    t0 = time.time()
    for i, cfg in enumerate(todo, 1):
        # cell is passed through explicitly: FLConfig.replace() recomputes it by
        # default, which would collapse the mock and local cells into one directory.
        if a.threads:
            cfg = cfg.replace(torch_threads=a.threads, cell=cfg.cell)
        if a.btc_checkpoint:
            cfg = cfg.replace(btc_checkpoint=True, cell=cfg.cell)
        print(f"\n=== [{i}/{len(todo)}] {cfg.run_id} ===", flush=True)
        try:
            out = run(cfg)
            c = out["final"]["commit"]
            print(f"  root={c['root'][:16]} anchor_ms={c['anchor_ms']:.1f} "
                  f"bytes={c['bytes_committed']}")
        except Exception as e:                      # one bad cell must not kill the grid
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min. Build Table 3 with:\n"
          f"  python3 -m fedverify.analysis.make_tables --only table3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
