"""Federated client: local SGD, returns a DELTA (never raw weights)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .models import build_model, get_flat_params, set_flat_params


@dataclass
class ClientUpdate:
    client_id: int
    delta: torch.Tensor              # local_params - global_params, 1-D
    num_samples: int
    train_loss: float
    wall_time_s: float
    privacy: Optional[dict] = None   # populated in Phase 2
    meta: dict = field(default_factory=dict)


class Client:
    def __init__(self, client_id: int, indices, loader):
        self.id = int(client_id)
        self.indices = list(indices)
        self.loader = loader

    def __len__(self) -> int:
        return len(self.indices)

    def local_train(self, global_params: torch.Tensor, cfg, model: nn.Module = None) -> ClientUpdate:
        t0 = time.perf_counter()
        dev = torch.device(cfg.device)
        if model is None:
            from .data import dataset_spec
            spec = dataset_spec(cfg.dataset)
            model = build_model(cfg.dataset, spec["num_classes"], spec["in_shape"])
        model = model.to(dev)
        set_flat_params(model, global_params.to(dev))
        model.train()

        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
        lossf = nn.CrossEntropyLoss()

        total_loss, n_batches, seen = 0.0, 0, 0
        for _ in range(cfg.local_epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(xb), yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.detach())
                n_batches += 1
                seen += int(yb.numel())

        delta = get_flat_params(model).cpu() - global_params.cpu()
        return ClientUpdate(
            client_id=self.id,
            delta=delta,
            num_samples=len(self.indices),
            train_loss=(total_loss / n_batches) if n_batches else float("nan"),
            wall_time_s=time.perf_counter() - t0,
            privacy=None,
            meta={"batches": n_batches, "samples_seen": seen},
        )
