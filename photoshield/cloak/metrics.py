"""Imperceptibility and disruption metrics (design §10, E1 and E2).

All functions take [B, 3, H, W] tensors in [0, 1] (or HxWx3 uint8 arrays where noted) and
return plain floats. A manifest stores them as decimal strings (manifest.metric).
"""
from __future__ import annotations

import math

import numpy as np
import torch


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(torch.mean((a.double() - b.double()) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    from skimage.metrics import structural_similarity
    x = a[0].permute(1, 2, 0).double().cpu().numpy()
    y = b[0].permute(1, 2, 0).double().cpu().numpy()
    return float(structural_similarity(x, y, channel_axis=2, data_range=1.0))


def linf_255(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max() * 255.0)


@torch.no_grad()
def latent_shift(enc, x: torch.Tensor, x_adv: torch.Tensor) -> float:
    """||enc(x_adv) - enc(x)|| / ||enc(x)||: how far the cloak moved the encoder's view."""
    z, za = enc.encode(x), enc.encode(x_adv)
    return float(torch.linalg.vector_norm(za - z) / (torch.linalg.vector_norm(z) + 1e-12))


@torch.no_grad()
def cosine(enc, x: torch.Tensor, x_adv: torch.Tensor) -> float:
    z, za = enc.encode(x).flatten(1), enc.encode(x_adv).flatten(1)
    return float(torch.nn.functional.cosine_similarity(z, za).mean())


@torch.no_grad()
def recon_psnr_drop(enc, x: torch.Tensor, x_adv: torch.Tensor):
    """PSNR(x, D(E(x))) - PSNR(x, D(E(x_adv))), in dB. None for encoders without a decoder.

    Positive means the cloaked photo reconstructs to something further from the real photo
    than the clean one does: the VAE round-trip every SD edit starts with is damaged.
    """
    if not getattr(enc, "can_reconstruct", False):
        return None
    return psnr(x, enc.reconstruct(x)) - psnr(x, enc.reconstruct(x_adv))


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float().div(255.0)[None]


def to_uint8(x: torch.Tensor) -> np.ndarray:
    return (x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
