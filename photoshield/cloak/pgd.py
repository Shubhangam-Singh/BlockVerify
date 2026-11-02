"""Projected gradient descent against an image encoder (Madry et al., ICLR 2018), design §8.

    untargeted:  maximise  MSE(enc(T(x + d)), enc(x))
    gray:        minimise  MSE(enc(T(x + d)), enc(gray))   (the PhotoGuard encoder attack)

with T drawn by EOT each step, a random start in ±eps/alpha_div, signed-gradient steps of
size alpha, projection onto the L-inf ball of radius eps, and clamping to [0, 1].
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


def pgd_attack(x: torch.Tensor, enc, eps: float, steps: int, alpha: float,
               target: str = "untargeted", eot: Optional[Callable] = None, seed: int = 0,
               log: Optional[list] = None) -> torch.Tensor:
    """Return x + delta with ||delta||_inf <= eps and values in [0, 1]. x: [B, 3, H, W]."""
    x = x.detach()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if target == "gray":
            ref = enc.encode(torch.full_like(x, 0.5))
            sign = -1.0                      # descend towards the decoy
        else:
            ref = enc.encode(x)
            sign = +1.0                      # ascend away from the clean latent
    start = alpha if steps > 0 else 0.0
    delta = ((torch.rand(x.shape, generator=g, dtype=x.dtype) * 2 - 1) * start).to(x.device)
    delta = (x + delta).clamp(0, 1) - x
    for step in range(steps):
        delta.requires_grad_(True)
        xin = x + delta
        if eot is not None:
            xin = eot(xin)
        loss = F.mse_loss(enc.encode(xin), ref)
        grad, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta = delta + sign * alpha * grad.sign()
            delta = delta.clamp(-eps, eps)
            delta = (x + delta).clamp(0, 1) - x
        if log is not None:
            log.append({"step": step, "loss": float(loss.detach())})
    return (x + delta).detach()
