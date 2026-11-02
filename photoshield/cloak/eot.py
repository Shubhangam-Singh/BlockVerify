"""Expectation over Transformation (Athalye et al., ICML 2018), design §8.

Each PGD step draws ONE transform from the enabled set (plus identity) with a seeded RNG, so
the perturbation is optimised to survive the things sharing does to a photo:

    resize   downscale by a factor in [0.7, 1.0] and back (bilinear, antialiased)
    blur     Gaussian blur, sigma in [0, 1]
    jpeg     real JPEG at quality in [q_min, q_max] via Pillow, made differentiable with the
             straight-through estimator: y = x + (jpeg(x) - x).detach(). The forward pass sees
             the true JPEG; the backward pass treats it as identity (BPDA, Athalye et al. 2018).

The JPEG choice is recorded in the run config as jpeg_mode = "straight-through".
"""
from __future__ import annotations

import io
import math
import random
from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F

JPEG_MODE = "straight-through"


def _resize(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    s = rng.uniform(0.7, 1.0)
    h, w = x.shape[-2:]
    small = F.interpolate(x, size=(max(8, round(h * s)), max(8, round(w * s))),
                          mode="bilinear", align_corners=False, antialias=True)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 1e-3:
        return x
    r = max(1, math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kx, groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), ky, groups=c)


def jpeg_real(x: torch.Tensor, quality: int) -> torch.Tensor:
    """Round-trip each image through Pillow JPEG. Not differentiable on its own."""
    from PIL import Image
    out = []
    for img in x.detach().clamp(0, 1):
        arr = (img.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "JPEG", quality=int(quality))
        dec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.float32) / 255.0
        out.append(torch.from_numpy(dec).permute(2, 0, 1))
    return torch.stack(out).to(device=x.device, dtype=x.dtype)


def _jpeg_st(x: torch.Tensor, rng: random.Random, qmin: int, qmax: int) -> torch.Tensor:
    y = jpeg_real(x, rng.randint(qmin, qmax))
    return x + (y - x).detach()


def make_eot(names: List[str], seed: int, qmin: int = 50, qmax: int = 95) -> Callable:
    """Return f(x) applying one seeded random transform per call (identity included)."""
    rng = random.Random(seed)
    choices = ["identity"] + list(names)

    def apply(x: torch.Tensor) -> torch.Tensor:
        t = rng.choice(choices)
        if t == "resize":
            return _resize(x, rng)
        if t == "blur":
            return gaussian_blur(x, rng.uniform(0.0, 1.0))
        if t == "jpeg":
            return _jpeg_st(x, rng, qmin, qmax)
        return x

    apply.choices = choices
    return apply
