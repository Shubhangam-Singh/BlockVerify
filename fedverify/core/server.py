"""Evaluation metrics, and the aggregator surface Phase 1/2 imported from here.

The aggregators themselves moved to fedverify/aggregators/ in Phase 4 (FedAvg is now
aggregators/fedavg.py). They are re-exported here so every existing import keeps working
and exp1 results are unaffected — there is exactly ONE implementation, not a copy.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..aggregators import (AGGREGATORS, Aggregator, CoordinateMedian, FedAvg, Forensics,
                           Krum, MultiKrum, TrimmedMean, build_aggregator)
from .client import ClientUpdate

__all__ = ["Aggregator", "FedAvg", "Krum", "MultiKrum", "TrimmedMean", "CoordinateMedian",
           "Forensics", "AGGREGATORS", "build_aggregator", "evaluate"]


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
