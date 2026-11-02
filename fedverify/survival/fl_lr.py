"""Logistic-regression family (prompt §5): local_lr, central_lr, fedsgd_lr, fedavg_lr, dp_fedavg_lr.

One FOLD-RUN trains on one CV fold's training rows for cfg.rounds rounds and is evaluated after every round. Each
hospital sees only its own rows. Clients send DELTAS (local - global) as existing ClientUpdate objects, and the
EXISTING FedAvg aggregator (sample-count-weighted mean) combines them through the D10 integer-id adapter.

  local_lr    each hospital trains its own model; each patient is scored by their OWN hospital's model
  central_lr  one model on the pooled training rows (upper bound, not private); same epochs/batch/lr per round
  fedsgd_lr   each round every hospital takes ONE full-batch gradient step; the server averages the deltas
  fedavg_lr   each round every hospital runs local_epochs of minibatch SGD; the server averages the deltas
  chain_fedavg_lr  fedavg_lr where every delta is serialized, content-addressed, signed, verified, Merkle-committed
              and anchored, and the server averages the deltas rebuilt FROM THE COMMITTED BYTES (Phase 3)
  dp_fedavg_lr  fedavg_lr with DP-SGD at every hospital (Phase 2): the EXISTING core/privacy.py solves sigma once per
              hospital over its TOTAL steps (rounds x local_epochs x ceil(n/batch)), one accountant per hospital
              persists across rounds, every per-example gradient is clipped to C and Gaussian noise is added.
              epsilon = inf takes the fedavg_lr code path exactly and never touches Opacus.
  ours_lr     dp_fedavg_lr (epsilon 4 by default) + the chain_fedavg_lr commit + the EXISTING Forensics screen on the
              committed deltas, survivors FedAvg'd (Phase 4). With screening off it is the same pipeline with FedAvg,
              the undefended row. A malicious hospital (survival/poison.py) attacks before the commitment.

Loss: binary cross-entropy weighted for class imbalance, pos_weight = n_neg / n_pos of the rows that training step
sees (each hospital its own, computed locally, never shared).
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..aggregators import build_aggregator
from ..core.client import ClientUpdate
from ..core.privacy import PrivacyPlan, dp_enabled, make_private, privacy_report
from .common import RoundOutput, codes_diag, local_hospitals, plain_diag
from .config import SurvivalConfig, derive_seed
from . import poison
from .chain_round import FoldCommitter, skipped_record
from .data import FoldData
from .serialize import serialize_lr
from .models import LogisticModel, build_logistic, get_flat_params, predict_risk, set_flat_params


def pos_weight(y: torch.Tensor, weighted: bool = True) -> torch.Tensor:
    """n_neg / n_pos (1.0 without deaths or when weighting is off)."""
    n_pos = float(y.sum())
    n_neg = float(y.numel()) - n_pos
    return torch.tensor((n_neg / n_pos) if (weighted and n_pos > 0) else 1.0, dtype=torch.float32)


def loss_fn(model: LogisticModel, X: torch.Tensor, y: torch.Tensor, pw: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(model(X), y, pos_weight=pw)


def sgd_epochs(model: LogisticModel, X: torch.Tensor, y: torch.Tensor, pw: torch.Tensor, cfg: SurvivalConfig,
               gen: torch.Generator, epochs: int) -> float:
    """Minibatch SGD for `epochs` passes; returns the mean batch loss. The generator persists across rounds."""
    loader = DataLoader(TensorDataset(X, y), batch_size=cfg.batch_size, shuffle=True, generator=gen,
                        num_workers=0, drop_last=False)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    model.train()
    total, batches = 0.0, 0
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, xb, yb, pw)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            batches += 1
    return total / batches if batches else float("nan")


def dp_sgd_epochs(model: LogisticModel, X: torch.Tensor, y: torch.Tensor, pw: torch.Tensor, cfg: SurvivalConfig,
                  gen: torch.Generator, plan: PrivacyPlan, noise_gen: torch.Generator) -> float:
    """DP-SGD for local_epochs through the existing make_private (fixed sigma, persistent accountant, Poisson
    sampling). Mirrors core/client.py: a fresh module per round, empty Poisson batches skipped, hooks removed."""
    loader = DataLoader(TensorDataset(X, y), batch_size=cfg.batch_size, shuffle=True, generator=gen,
                        num_workers=0, drop_last=False)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    train_module, opt, dp_loader = make_private(model, opt, loader, plan)
    # Opacus draws the noise from the global RNG unless told otherwise; a dedicated generator per
    # (repeat, fold, round, hospital) keeps a DP fold-run reproducible on its own. Poisson sampling already uses
    # the loader's seeded generator.
    opt.generator = noise_gen
    total, batches = 0.0, 0
    try:
        train_module.train()
        for _ in range(cfg.local_epochs):
            for xb, yb in dp_loader:
                if yb.numel() == 0:                      # Poisson sampling can yield empty batches
                    continue
                opt.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(train_module(xb), yb, pos_weight=pw)
                loss.backward()
                opt.step()                               # clips, adds noise, steps the hospital's accountant
                total += float(loss.detach())
                batches += 1
    finally:
        if hasattr(train_module, "remove_hooks"):
            try:
                train_module.remove_hooks()
            except Exception:
                pass
    return total / batches if batches else float("nan")


def full_batch_step(model: LogisticModel, X: torch.Tensor, y: torch.Tensor, pw: torch.Tensor,
                    cfg: SurvivalConfig) -> float:
    """FedSGD: exactly one gradient step on ALL of a hospital's rows."""
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    model.train()
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model, X, y, pw)
    loss.backward()
    opt.step()
    return float(loss.detach())


def run_fold(cfg: SurvivalConfig, fold: FoldData, repeat_seed: int, fold_index: int,
             plan_cache: Optional[dict] = None,
             committer: Optional[FoldCommitter] = None,
             attackers: Optional[list] = None, **_: object) -> Iterator[RoundOutput]:
    """Train one method on one fold; yield a RoundOutput after every round. plan_cache (DP methods only) lets
    folds and repeats share the sigma solved for a given hospital size (see survival/dp.py). attackers (ours_lr only)
    are the malicious hospitals of this repeat."""
    method = cfg.method
    index = cfg.hospital_index                     # D10: integer ids only inside existing FedVerify code
    X_tr = torch.as_tensor(fold.X_train, dtype=torch.float32)
    X_te = torch.as_tensor(fold.X_test, dtype=torch.float32)
    y_tr = torch.as_tensor(fold.event_train, dtype=torch.float32)
    d = X_tr.shape[1]
    hosp = [h for h in cfg.hospitals if (fold.hosp_train == h).any()]
    rows = {h: np.flatnonzero(fold.hosp_train == h) for h in hosp}
    test_rows = {h: np.flatnonzero(fold.hosp_test == h) for h in hosp}
    n_samples = {h: int(len(rows[h])) for h in hosp}
    pw = {h: pos_weight(y_tr[rows[h]], cfg.class_weighted_loss) for h in hosp}
    gen = {h: torch.Generator().manual_seed(derive_seed(repeat_seed, fold_index, "loader", h)) for h in hosp}

    if method == "central_lr":
        model = build_logistic(d)
        g = torch.Generator().manual_seed(derive_seed(repeat_seed, fold_index, "loader", "central"))
        pw_all = pos_weight(y_tr, cfg.class_weighted_loss)
        for rnd in range(1, cfg.rounds + 1):
            t0 = time.perf_counter()
            loss = sgd_epochs(model, X_tr, y_tr, pw_all, cfg, g, cfg.local_epochs)
            yield RoundOutput(rnd, predict_risk(model, X_tr), predict_risk(model, X_te), plain_diag(hosp),
                              n_samples, loss, time.perf_counter() - t0)
        return

    if method == "local_lr":
        own = local_hospitals(cfg, hosp)               # heart: Switzerland gets no local model (Revision 2 §2.1)
        models = {h: build_logistic(d) for h in own}
        for rnd in range(1, cfg.rounds + 1):
            t0 = time.perf_counter()
            losses = [sgd_epochs(models[h], X_tr[rows[h]], y_tr[rows[h]], pw[h], cfg, gen[h], cfg.local_epochs)
                      for h in own]
            train_risk = np.full(len(fold.train_idx), np.nan)    # NaN = not modelled (excluded hospital)
            test_risk = np.full(len(fold.test_idx), np.nan)
            for h in own:                                   # each patient is scored by their own hospital's model
                train_risk[rows[h]] = predict_risk(models[h], X_tr[rows[h]])
                test_risk[test_rows[h]] = predict_risk(models[h], X_te[test_rows[h]])
            yield RoundOutput(rnd, train_risk, test_risk, plain_diag(own), {h: n_samples[h] for h in own},
                              float(np.mean(losses)), time.perf_counter() - t0)
        return

    if method not in ("fedsgd_lr", "fedavg_lr", "dp_fedavg_lr", "chain_fedavg_lr", "ours_lr"):
        raise ValueError(f"{method!r} is not a logistic-family method")
    # Phase 2: sigma is solved ONCE per hospital for the whole fold-run. No plans when epsilon is None/inf.
    plans: dict = {}
    if method in ("dp_fedavg_lr", "ours_lr") and dp_enabled(cfg):
        from .dp import hospital_plan                    # imports Opacus; only ever reached under DP
        plans = {h: hospital_plan(cfg, index[h], n_samples[h], plan_cache) for h in hosp}
    # ours_lr screens with the EXISTING Forensics aggregator, which reads cfg.tau (and raises without it)
    aggregator = build_aggregator("forensics" if cfg.screened else "fedavg")
    bad = set(attackers or ()) if method == "ours_lr" else set()
    global_model = build_logistic(d)
    global_params = get_flat_params(global_model)
    for rnd in range(1, cfg.rounds + 1):
        t0 = time.perf_counter()
        skip = committer.check_previous() if committer is not None else {}
        if skip:              # hospitals saw a bad previous block: they skip, the aggregator records it
            yield RoundOutput(rnd, predict_risk(global_model, X_tr), predict_risk(global_model, X_te),
                              {"accepted": [], "rejected": [], "scores": {}}, {}, None, time.perf_counter() - t0,
                              {"commit": skipped_record(skip)})
            continue
        updates = []
        for h in hosp:
            local = build_logistic(d)
            set_flat_params(local, global_params)
            X_h, y_h, pw_h = X_tr[rows[h]], y_tr[rows[h]], pw[h]
            if h in bad and cfg.attack == "label_flip":         # Phase 4: the attacker trains on flipped labels
                X_h, y_h = poison.poisoned_labels(X_h, y_h, cfg, repeat_seed, rnd, h)
                pw_h = pos_weight(y_h, cfg.class_weighted_loss)
            if plans:
                noise = torch.Generator().manual_seed(derive_seed(repeat_seed, fold_index, rnd, h, "dp-noise"))
                loss = dp_sgd_epochs(local, X_h, y_h, pw_h, cfg, gen[h], plans[h], noise)
            elif method in ("fedavg_lr", "dp_fedavg_lr", "chain_fedavg_lr", "ours_lr"):
                loss = sgd_epochs(local, X_h, y_h, pw_h, cfg, gen[h], cfg.local_epochs)
            else:
                loss = full_batch_step(local, X_h, y_h, pw_h, cfg)
            delta = get_flat_params(local) - global_params
            if h in bad and poison.is_delta_attack(cfg):        # Phase 4: sign_flip / scaling of the trained delta
                delta = poison.poisoned_delta(delta, cfg, repeat_seed, rnd, h, len(bad))
            updates.append(ClientUpdate(client_id=index[h],                   # D10: int only inside existing code
                                        delta=delta, num_samples=n_samples[h], train_loss=loss, wall_time_s=0.0))
        extra, commitment = {}, None
        if committer is not None:                    # Phase 3: the server only sees the committed bytes
            res = committer.commit_round(rnd, {u_h: (lambda h=u_h, u=u: serialize_lr(
                committer.run_id, rnd, h, u.num_samples, u.delta)) for u_h, u in zip(hosp, updates)})
            by_h = dict(zip(hosp, updates))
            updates_used = [ClientUpdate(client_id=index[h], delta=res.models[h], num_samples=by_h[h].num_samples,
                                         train_loss=by_h[h].train_loss, wall_time_s=0.0) for h in res.accepted]
            extra, commitment = {"commit": res.record}, res.commitment
        else:
            updates_used = updates
        if updates_used:
            delta, diag = aggregator.aggregate(updates_used, cfg, rnd)
            global_params = global_params + delta
        else:
            diag = {"accepted": [], "rejected": [], "scores": {}}
        set_flat_params(global_model, global_params)
        yield RoundOutput(rnd, predict_risk(global_model, X_tr), predict_risk(global_model, X_te),
                          codes_diag(diag, cfg.dataset_hospitals), n_samples,
                          float(np.mean([u.train_loss for u in updates])),
                          time.perf_counter() - t0, extra,
                          privacy={h: privacy_report(plans[h]) for h in hosp} if plans else None,
                          commitment=commitment)
