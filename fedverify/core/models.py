"""Models and flat-parameter utilities.

GroupNorm everywhere, never BatchNorm: Opacus (Phase 2) cannot handle BatchNorm, and
BatchNorm running statistics would leak information across FL clients (CLAUDE.md,
"Modelling").
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """28x28x1 -> num_classes. conv-GN-ReLU-pool x2 then two FC layers."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5),   # 28 -> 24
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 24 -> 12
            nn.Conv2d(16, 32, 5),            # 12 -> 8
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name: str, num_classes: int = 10, in_shape=(1, 28, 28)) -> nn.Module:
    if name in ("smallcnn", "mnist", "fmnist"):
        return SmallCNN(num_classes=num_classes, in_channels=in_shape[0])
    raise ValueError(f"unknown model {name!r} (mitbih 1-D CNN arrives in Phase 5)")


def assert_no_batchnorm(model: nn.Module) -> None:
    bad = [n for n, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if bad:
        raise ValueError(f"BatchNorm is forbidden (Opacus + FL leakage); found: {bad}")


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """All trainable parameters as one 1-D tensor (detached clone)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, vec: torch.Tensor) -> None:
    """Exact inverse of get_flat_params."""
    expected = sum(p.numel() for p in model.parameters())
    if vec.numel() != expected:
        raise ValueError(f"expected {expected} params, got {vec.numel()}")
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[i:i + n].view_as(p).to(p.dtype))
            i += n
