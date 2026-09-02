"""Aggregation interface, FedAvg, and evaluation metrics.

Every aggregator returns (delta, diag) where diag ALWAYS carries accepted / rejected /
scores, so Phase 3 can commit an accept-reject lineage on-chain and Phase 4 can score
Byzantine screening without changing this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .client import ClientUpdate


class Aggregator(ABC):
    name = "base"

    @abstractmethod
    def aggregate(self, updates: Sequence[ClientUpdate], cfg, round_num: int
                  ) -> Tuple[torch.Tensor, dict]:
        """Return (aggregated_delta, diag). diag has accepted/rejected/scores."""

    @staticmethod
    def _empty_diag() -> dict:
        return {"accepted": [], "rejected": [], "scores": {}}


class FedAvg(Aggregator):
    """Sample-count-weighted mean of client deltas (McMahan et al., 2017)."""
    name = "fedavg"

    def aggregate(self, updates, cfg, round_num):
        if not updates:
            raise ValueError("no client updates to aggregate")
        w = torch.tensor([float(u.num_samples) for u in updates], dtype=torch.float64)
        if float(w.sum()) <= 0:
            raise ValueError("total num_samples across updates is zero")
        w = w / w.sum()
        stacked = torch.stack([u.delta.to(torch.float64) for u in updates])
        delta = (stacked * w.unsqueeze(1)).sum(dim=0).to(updates[0].delta.dtype)

        diag = self._empty_diag()
        diag["accepted"] = [int(u.client_id) for u in updates]
        diag["scores"] = {str(int(u.client_id)): {"weight": float(wi)}
                          for u, wi in zip(updates, w.tolist())}
        return delta, diag


AGGREGATORS = {"fedavg": FedAvg}


def build_aggregator(name: str) -> Aggregator:
    if name not in AGGREGATORS:
        raise ValueError(f"unknown aggregator {name!r}; have {sorted(AGGREGATORS)}")
    return AGGREGATORS[name]()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device="cpu") -> dict:
    """Loss, accuracy, macro-F1 and per-class accuracy (no sklearn dependency)."""
    dev = torch.device(device)
    model = model.to(dev).eval()
    lossf = nn.CrossEntropyLoss(reduction="sum")

    total_loss, total_n = 0.0, 0
    preds_all, targets_all = [], []
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        out = model(xb)
        total_loss += float(lossf(out, yb))
        total_n += int(yb.numel())
        preds_all.append(out.argmax(1).cpu())
        targets_all.append(yb.cpu())

    if total_n == 0:
        return {"loss": float("nan"), "accuracy": float("nan"),
                "macro_f1": float("nan"), "per_class_acc": {}}

    preds = torch.cat(preds_all).numpy()
    targets = torch.cat(targets_all).numpy()
    classes = np.unique(targets)

    f1s, per_class = [], {}
    for c in classes:
        tp = int(np.sum((preds == c) & (targets == c)))
        fp = int(np.sum((preds == c) & (targets != c)))
        fn = int(np.sum((preds != c) & (targets == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        support = int(np.sum(targets == c))
        per_class[str(int(c))] = (tp / support) if support else float("nan")

    return {
        "loss": total_loss / total_n,
        "accuracy": float(np.mean(preds == targets)),
        "macro_f1": float(np.mean(f1s)),
        "per_class_acc": per_class,
    }
