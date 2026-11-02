"""Logistic model for the Survival LR family: one linear layer plus a sigmoid (prompt §5).

Flat-parameter helpers are the EXISTING ones (fedverify.core.models), so deltas, the FedAvg aggregator, DP and the
chain commitment all see the same 1-D parameter vector as every other FedVerify model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..core.models import assert_no_batchnorm, get_flat_params, set_flat_params

__all__ = ["LogisticModel", "build_logistic", "predict_risk", "get_flat_params", "set_flat_params"]


class LogisticModel(nn.Module):
    """x -> logit of death. The sigmoid is applied by the loss (BCE-with-logits) and by predict_risk.

    Zero-initialised: the logistic loss is convex, so a fixed start puts every method at the same point and keeps
    runs deterministic without drawing from any RNG.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(int(n_features), 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def build_logistic(n_features: int) -> LogisticModel:
    model = LogisticModel(n_features)
    assert_no_batchnorm(model)
    return model


@torch.no_grad()
def predict_risk(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """P(death) per row as float64 numpy; higher = higher risk."""
    model.eval()
    return torch.sigmoid(model(X)).double().numpy()
