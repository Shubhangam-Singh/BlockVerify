"""Federated training loop.

Determinism (CLAUDE.md + amendment A1): every stochastic source is seeded, and a run
writes rounds.jsonl + config.json. Re-running with the same seed reproduces
rounds.jsonl byte-for-byte AFTER dropping wall-clock timing keys, which cannot be
deterministic; use ``strip_timings`` when comparing.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..config import FLConfig
from .client import Client, ClientUpdate
from .data import dataset_spec, dirichlet_partition, get_labels, load_dataset, partition_report
from .models import assert_no_batchnorm, build_model, get_flat_params, set_flat_params
from .server import build_aggregator, evaluate

# Keys excluded from determinism comparison (amendment A1)
TIMING_SUFFIXES = ("_ms", "_wall_s")
TIMING_KEYS = {"train_wall_s", "round_wall_s", "wall_time_s"}


def is_timing_key(k: str) -> bool:
    return k in TIMING_KEYS or k.endswith(TIMING_SUFFIXES)


def strip_timings(obj):
    """Recursively drop wall-clock fields so two seeded runs can be compared."""
    if isinstance(obj, dict):
        return {k: strip_timings(v) for k, v in obj.items() if not is_timing_key(k)}
    if isinstance(obj, list):
        return [strip_timings(v) for v in obj]
    return obj


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:                                   # not supported for every op/backend
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False


def _build_clients(train_set, partitions, cfg) -> List[Client]:
    clients = []
    for cid, idx in enumerate(partitions):
        g = torch.Generator()
        g.manual_seed(cfg.seed * 100_003 + cid)      # per-client, reproducible shuffling
        loader = DataLoader(Subset(train_set, idx), batch_size=cfg.batch_size,
                            shuffle=True, generator=g, num_workers=0, drop_last=False)
        clients.append(Client(cid, idx, loader))
    return clients


def run(cfg: FLConfig, limit_train: Optional[int] = None, limit_test: Optional[int] = None,
        progress: bool = True) -> dict:
    seed_everything(cfg.seed)
    dev = torch.device(cfg.device)
    os.makedirs(cfg.run_dir, exist_ok=True)

    # ── data ────────────────────────────────────────────────────────────────
    train_set, test_set = load_dataset(cfg.dataset)
    if limit_train:
        train_set = Subset(train_set, list(range(min(limit_train, len(train_set)))))
    if limit_test:
        test_set = Subset(test_set, list(range(min(limit_test, len(test_set)))))

    labels = get_labels(train_set.dataset)[np.asarray(train_set.indices)] \
        if isinstance(train_set, Subset) else get_labels(train_set)
    partitions = dirichlet_partition(labels, cfg.num_clients, cfg.alpha, cfg.seed)
    report = partition_report(partitions, labels)

    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    clients = _build_clients(train_set, partitions, cfg)

    # ── model ───────────────────────────────────────────────────────────────
    spec = dataset_spec(cfg.dataset)
    model = build_model(cfg.dataset, spec["num_classes"], spec["in_shape"]).to(dev)
    assert_no_batchnorm(model)
    global_params = get_flat_params(model).cpu()

    aggregator = build_aggregator(cfg.aggregator)
    sample_rng = np.random.default_rng(cfg.seed + 777)
    n_select = max(1, int(round(cfg.client_fraction * cfg.num_clients)))

    jsonl_path = os.path.join(cfg.run_dir, "rounds.jsonl")
    with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "partition_report": report,
                   "num_params": int(global_params.numel()),
                   "limit_train": limit_train, "limit_test": limit_test}, f, indent=2)

    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=cfg.rounds, desc=cfg.run_id, unit="round")
        except ImportError:
            bar = None

    history = []
    with open(jsonl_path, "w") as fout:
        for rnd in range(1, cfg.rounds + 1):
            t_round = time.perf_counter()

            selected = sorted(sample_rng.choice(cfg.num_clients, size=n_select,
                                                replace=False).tolist()) \
                if n_select < cfg.num_clients else list(range(cfg.num_clients))

            t_train = time.perf_counter()
            updates: List[ClientUpdate] = [clients[c].local_train(global_params, cfg, model)
                                           for c in selected]
            train_wall_s = time.perf_counter() - t_train

            updates = pre_aggregate(updates, cfg, rnd)          # Phase 3/4 hook
            delta, diag = aggregator.aggregate(updates, cfg, rnd)
            global_params = global_params + delta.cpu()

            set_flat_params(model, global_params.to(dev))
            metrics = evaluate(model, test_loader, cfg.device)

            rec = {
                "round": rnd,
                "test_acc": metrics["accuracy"],
                "test_loss": metrics["loss"],
                "macro_f1": metrics["macro_f1"],
                "mean_train_loss": float(np.mean([u.train_loss for u in updates])),
                "per_client_num_samples": {str(u.client_id): u.num_samples for u in updates},
                "diag": diag,
                "train_wall_s": train_wall_s,
                "round_wall_s": time.perf_counter() - t_round,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            history.append(rec)
            if bar:
                bar.update(1)
                bar.set_postfix(acc=f"{metrics['accuracy']:.4f}", f1=f"{metrics['macro_f1']:.4f}")
    if bar:
        bar.close()

    torch.save(model.state_dict(), os.path.join(cfg.run_dir, "final_model.pt"))
    return {"run_dir": cfg.run_dir, "rounds": history,
            "final": history[-1] if history else None, "partition_report": report}


def pre_aggregate(updates, cfg, round_num):
    """Hook: Phase 3 commits digests here, Phase 4 screens Byzantine clients. No-op now."""
    return updates


def main(argv=None):
    cfg, ns = FLConfig.from_args(argv)
    out = run(cfg, limit_train=ns.limit_train, limit_test=ns.limit_test)
    fin = out["final"]
    if fin:
        print(f"\n{cfg.run_id}: acc={fin['test_acc']:.4f} macro_f1={fin['macro_f1']:.4f} "
              f"-> {out['run_dir']}")
    return out


if __name__ == "__main__":
    main()
