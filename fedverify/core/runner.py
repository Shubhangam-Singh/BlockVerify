"""Federated training loop.

Determinism (CLAUDE.md + amendment A1): every stochastic source is seeded, and a run
writes rounds.jsonl + config.json. Re-running with the same seed reproduces
rounds.jsonl byte-for-byte AFTER dropping wall-clock timing keys, which cannot be
deterministic; use ``strip_timings`` when comparing.

Phase 3 chain commitment is OPT-IN (``cfg.commit``). When it is off — the default, and
what every Phase 1/2 experiment uses — the round record is byte-identical to Phase 2 and
no chain code is imported, so Table 1 cells produced before and after Phase 3 remain
mutually comparable. When it is on, one extra ``"commit"`` block is appended to each
record and leaves+proofs are streamed to ``commitments.jsonl``.
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
from .privacy import dp_enabled, plan_privacy, privacy_report
from ..attacks.byzantine import (DELTA_ATTACKS, apply_delta_attack,
                                 attack_active, attacker_ids)
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
    # Intra-op thread count changes float reduction order, so it changes results.
    # Pin it when given, and always RECORD the effective value (amendment A5).
    if cfg.torch_threads:
        torch.set_num_threads(int(cfg.torch_threads))
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
    # MIT-BIH partitions by PATIENT (real heterogeneity); everything else uses the
    # Dirichlet draw. The branch never fires for mnist/fmnist, so Phase-1/2 cells are
    # bit-for-bit what they were.
    if cfg.dataset == "mitbih":
        from .data import patient_partition_for
        partitions, part_extra = patient_partition_for(cfg, train_set)
    else:
        partitions, part_extra = dirichlet_partition(
            labels, cfg.num_clients, cfg.alpha, cfg.seed), {}
    report = partition_report(partitions, labels)
    report.update(part_extra)

    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    clients = _build_clients(train_set, partitions, cfg)

    # ── model ───────────────────────────────────────────────────────────────
    spec = dataset_spec(cfg.dataset)
    model = build_model(cfg.dataset, spec["num_classes"], spec["in_shape"]).to(dev)
    assert_no_batchnorm(model)
    global_params = get_flat_params(model).cpu()

    # ── privacy: solve sigma ONCE per client over its TOTAL steps (see privacy.py) ──
    if dp_enabled(cfg):
        for c in clients:
            c.privacy_plan = plan_privacy(cfg, c.id, len(c))

    aggregator = build_aggregator(cfg.aggregator)
    sample_rng = np.random.default_rng(cfg.seed + 777)
    n_select = max(1, int(round(cfg.client_fraction * cfg.num_clients)))

    # ── Phase 4: attacks (inert unless cfg.attack is set) ───────────────────
    attackers = attacker_ids(cfg)
    for c in clients:
        c.is_attacker = c.id in attackers
    triggered = None
    if cfg.attack == "backdoor" and attackers:
        from ..attacks.backdoor import attack_success_rate, build_triggered_testset
        triggered = build_triggered_testset(test_set, spec, cfg.backdoor_target)

    jsonl_path = os.path.join(cfg.run_dir, "rounds.jsonl")
    with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "partition_report": report,
                   "attacker_ids": sorted(attackers),
                   "num_params": int(global_params.numel()),
                   "torch_threads_effective": int(torch.get_num_threads()),
                   "limit_train": limit_train, "limit_test": limit_test}, f, indent=2)

    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=cfg.rounds, desc=cfg.run_id, unit="round")
        except ImportError:
            bar = None

    # ── Phase 3: chain commitment (opt-in; inert and unimported when cfg.commit is False)
    anchor, commit_fh = None, None
    if cfg.commit:
        from ..chain.anchor import ChainAnchor
        from ..chain.commitment import commit_round, warm_up
        warm_up()                       # pay the backend import BEFORE any timed region
        anchor = ChainAnchor(cfg.chain_backend, checkpoint_every=cfg.checkpoint_every,
                             bitcoin=cfg.btc_checkpoint)
        commit_fh = open(os.path.join(cfg.run_dir, "commitments.jsonl"), "w")

    history = []
    with open(jsonl_path, "w") as fout:
        for rnd in range(1, cfg.rounds + 1):
            t_round = time.perf_counter()

            selected = sorted(sample_rng.choice(cfg.num_clients, size=n_select,
                                                replace=False).tolist()) \
                if n_select < cfg.num_clients else list(range(cfg.num_clients))

            t_train = time.perf_counter()
            updates: List[ClientUpdate] = [
                clients[c].local_train(global_params, cfg, model, round_num=rnd)
                for c in selected]
            train_wall_s = time.perf_counter() - t_train

            # DELTA-level attacks replace the honestly-trained delta. This happens BEFORE
            # the commitment on purpose: the chain must bind what the client actually
            # sent, so the accept/reject lineage is auditable against the real payload.
            if cfg.attack in DELTA_ATTACKS and attack_active(cfg, rnd):
                for u in updates:
                    if u.client_id in attackers:
                        u.delta = apply_delta_attack(u.delta, cfg, rnd, u.client_id,
                                                     len(attackers))
                        u.meta["attacked"] = cfg.attack

            # Digests -> leaves -> root -> anchor, on the RAW updates: the commitment
            # must bind what each client actually sent, before any screening or
            # aggregation could alter it.
            commit_rec = None
            if anchor is not None:
                cm = commit_round(updates, rnd)
                an = anchor.anchor_round(cfg.run_id, rnd, cm["root"], cm["leaf_count"],
                                         cm["leaves"])
                cp = anchor.checkpoint(cfg.run_id, rnd) if anchor.due_for_checkpoint(rnd) else None
                commit_fh.write(json.dumps({
                    "run_id": cfg.run_id, "round": rnd, "root": cm["root"],
                    "leaf_count": cm["leaf_count"], "txid": an["txid"],
                    "backend": an["backend"], "leaves": cm["leaves"], "checkpoint": cp,
                }) + "\n")
                commit_fh.flush()
                commit_rec = {
                    "root": cm["root"], "leaf_count": cm["leaf_count"],
                    "txid": an["txid"], "backend": an["backend"],
                    "bytes_committed": cm["bytes_committed"], "fee": an["fee"],
                    "bytes_written": an["bytes_written"],
                    "digest_ms": cm["digest_ms"], "merkle_ms": cm["merkle_ms"],
                    "anchor_ms": an["latency_ms"],
                    "checkpoint_ms": (cp or {}).get("latency_ms"),
                }

            updates = pre_aggregate(updates, cfg, rnd)          # Phase 3/4 hook
            t_agg = time.perf_counter()
            delta, diag = aggregator.aggregate(updates, cfg, rnd)
            aggregate_ms = (time.perf_counter() - t_agg) * 1000.0
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
                "privacy": ({str(u.client_id): u.privacy for u in updates}
                            if any(u.privacy for u in updates) else None),
                "train_wall_s": train_wall_s,
                "round_wall_s": time.perf_counter() - t_round,
            }
            # Emitted when an attack is configured OR the aggregator is a real detector,
            # so the honest-client false-exclusion rate is measurable on CLEAN runs too —
            # that is the cost-of-defence number Table 2b reports. Aggregators that are not
            # detectors (fedavg in every Phase-1/2 cell) still emit nothing, so exp1
            # records keep their exact Phase-2 shape.
            if (cfg.attack and cfg.attack != "none") or getattr(aggregator, "detects", False):
                rec["attack"] = _attack_report(cfg, rnd, attackers, diag, selected)
                if triggered is not None:
                    from ..attacks.backdoor import attack_success_rate
                    rec["attack"]["asr"] = attack_success_rate(
                        model, triggered, cfg.backdoor_target, cfg.device)

            if commit_rec is not None:
                commit_rec["aggregate_ms"] = aggregate_ms
                rec["commit"] = commit_rec
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            history.append(rec)
            if bar:
                bar.update(1)
                bar.set_postfix(acc=f"{metrics['accuracy']:.4f}", f1=f"{metrics['macro_f1']:.4f}")
    if bar:
        bar.close()
    if commit_fh:
        commit_fh.close()

    torch.save(model.state_dict(), os.path.join(cfg.run_dir, "final_model.pt"))
    return {"run_dir": cfg.run_dir, "rounds": history,
            "final": history[-1] if history else None, "partition_report": report,
            "lineage": anchor.lineage() if anchor else None,
            "checkpoints": anchor.checkpoints if anchor else None}


def _attack_report(cfg, rnd, attackers, diag, selected) -> dict:
    """Screening quality for one round: did the aggregator reject the right clients?

    Precision/recall are over the clients that actually PARTICIPATED this round, and the
    attacker set is empty before attack_start_round — so a rejection during the clean
    warm-up counts as a false positive, which is the honest accounting.
    """
    present = set(int(c) for c in selected)
    active = set(attackers) if attack_active(cfg, rnd) else set()
    mal = active & present
    honest = present - mal
    rejected = set(int(c) for c in diag.get("rejected", []))

    tp = len(rejected & mal)
    fp = len(rejected & honest)
    fn = len(mal - rejected)
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if not mal else 0.0)
    rec_ = tp / (tp + fn) if (tp + fn) else (1.0 if not mal else 0.0)
    f1 = (2 * prec * rec_ / (prec + rec_)) if (prec + rec_) else 0.0

    return {
        "attack": cfg.attack,
        "active": bool(active),
        "attackers": sorted(mal),
        "rejected": sorted(rejected),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec_, "f1": f1,
        "false_exclusion_rate": (fp / len(honest)) if honest else 0.0,
    }


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
