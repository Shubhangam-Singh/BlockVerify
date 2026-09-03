"""Federated client: local SGD, returns a DELTA (never raw weights).

When DP is enabled the client trains under Opacus with a FIXED noise multiplier and a
PERSISTENT accountant (see core/privacy.py). When epsilon is None/inf, Opacus is never
imported or involved and this is byte-for-byte the Phase-1 code path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .models import build_model, get_flat_params, set_flat_params
from .privacy import PrivacyPlan, make_private, privacy_report


@dataclass
class ClientUpdate:
    client_id: int
    delta: torch.Tensor              # local_params - global_params, 1-D
    num_samples: int
    train_loss: float
    wall_time_s: float
    privacy: Optional[dict] = None   # privacy_report(plan) when DP is on
    meta: dict = field(default_factory=dict)


class Client:
    def __init__(self, client_id: int, indices, loader):
        self.id = int(client_id)
        self.indices = list(indices)
        self.loader = loader
        self.privacy_plan: Optional[PrivacyPlan] = None   # set by the runner when DP is on
        self.is_attacker: bool = False                    # set by the runner (Phase 4)

    def __len__(self) -> int:
        return len(self.indices)

    def _fresh_model(self, cfg):
        from .data import dataset_spec
        spec = dataset_spec(cfg.dataset)
        return build_model(cfg.dataset, spec["num_classes"], spec["in_shape"])

    def _poison_fn(self, cfg, round_num):
        """Batch corrupter for DATA-level attacks, or None (the overwhelmingly common case)."""
        from ..attacks.byzantine import DATA_ATTACKS, attack_active
        if not (self.is_attacker and cfg.attack in DATA_ATTACKS
                and attack_active(cfg, round_num)):
            return None
        from ..attacks.byzantine import poison_batch
        from .data import dataset_spec
        spec = dataset_spec(cfg.dataset)
        nc = int(spec["num_classes"])
        return lambda xb, yb: poison_batch(xb, yb, cfg, round_num, self.id, nc, spec)

    def local_train(self, global_params: torch.Tensor, cfg, model: nn.Module = None,
                    round_num: int = 0) -> ClientUpdate:
        t0 = time.perf_counter()
        dev = torch.device(cfg.device)
        plan = self.privacy_plan
        use_dp = plan is not None

        # Under DP we always use a fresh module so Opacus hooks/grad-sample state can
        # never leak between clients or rounds. Without DP we reuse the runner's model
        # (identical to Phase 1).
        if use_dp or model is None:
            model = self._fresh_model(cfg)
        model = model.to(dev)
        set_flat_params(model, global_params.to(dev))
        model.train()

        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
        lossf = nn.CrossEntropyLoss()
        loader, train_module = self.loader, model

        if use_dp:
            train_module, opt, loader = make_private(model, opt, self.loader, plan)

        # DATA-level attacks corrupt the batches this client trains on, so the resulting
        # delta is genuinely what the poisoned data produced. `poison` stays None for every
        # honest client and for every non-attacked run, leaving the Phase-1/2 path exact.
        poison = self._poison_fn(cfg, round_num)

        total_loss, n_batches, seen, skipped = 0.0, 0, 0, 0
        try:
            for _ in range(cfg.local_epochs):
                for xb, yb in loader:
                    if yb.numel() == 0:          # Poisson sampling can yield empty batches
                        skipped += 1
                        continue
                    if poison is not None:
                        xb, yb = poison(xb, yb)
                    xb, yb = xb.to(dev), yb.to(dev)
                    opt.zero_grad(set_to_none=True)
                    loss = lossf(train_module(xb), yb)
                    loss.backward()
                    opt.step()                   # steps the persistent accountant under DP
                    total_loss += float(loss.detach())
                    n_batches += 1
                    seen += int(yb.numel())
        finally:
            if use_dp and hasattr(train_module, "remove_hooks"):
                try:
                    train_module.remove_hooks()
                except Exception:
                    pass

        delta = get_flat_params(model).cpu() - global_params.cpu()
        return ClientUpdate(
            client_id=self.id,
            delta=delta,
            num_samples=len(self.indices),
            train_loss=(total_loss / n_batches) if n_batches else float("nan"),
            wall_time_s=time.perf_counter() - t0,
            privacy=privacy_report(plan),
            meta={"batches": n_batches, "samples_seen": seen, "empty_batches": skipped},
        )
