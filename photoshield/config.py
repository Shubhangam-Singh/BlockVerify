"""ShieldConfig: every parameter that influences a cloak run (CLAUDE.md "Determinism")."""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

MODELS = ("sdvae", "resnet50", "facenet", "tiny")
EOT_CHOICES = ("resize", "blur", "jpeg")


@dataclass(frozen=True)
class ShieldConfig:
    model: str = "sdvae"          # encoder attacked (design §8)
    eps: int = 8                  # L-inf budget in x/255 on [0, 1] images
    steps: int = 40               # PGD iterations
    alpha_div: int = 10           # step size = eps / alpha_div (and random start in ±eps/alpha_div)
    size: int = 256               # working resolution: longest side, in pixels
    seed: int = 0
    threads: int = 2              # torch intra-op threads, pinned and recorded (A5)
    eot: List[str] = field(default_factory=lambda: ["resize", "blur"])
    target: str = "untargeted"    # "untargeted" (push away from enc(x)) or "gray" (PhotoGuard decoy)
    full_res: bool = False        # upsample delta to the original resolution instead of
                                  # outputting the working-size image
    jpeg_quality_min: int = 50    # EOT JPEG quality range (straight-through estimator)
    jpeg_quality_max: int = 95
    pretrained: bool = True       # False only in tests (random weights, no download)

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if not (1 <= self.eps <= 64):
            raise ValueError("eps is in x/255 and must be between 1 and 64")
        if self.steps < 0 or self.size < 32 or self.alpha_div < 1:
            raise ValueError("steps >= 0, size >= 32, alpha_div >= 1")
        bad = [t for t in self.eot if t not in EOT_CHOICES]
        if bad:
            raise ValueError(f"unknown EOT transform(s) {bad}; choose from {EOT_CHOICES}")
        if self.target not in ("untargeted", "gray"):
            raise ValueError("target must be 'untargeted' or 'gray'")

    @property
    def eps_float(self) -> float:
        return self.eps / 255.0

    @property
    def alpha_float(self) -> float:
        return self.eps_float / self.alpha_div

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def add_args(p: argparse.ArgumentParser) -> None:
    d = ShieldConfig()
    p.add_argument("--model", choices=MODELS, default=d.model)
    p.add_argument("--eps", type=int, default=d.eps, help="L-inf budget in x/255 (default 8)")
    p.add_argument("--steps", type=int, default=d.steps)
    p.add_argument("--alpha-div", type=int, default=d.alpha_div)
    p.add_argument("--size", type=int, default=d.size, help="working resolution, longest side")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--threads", type=int, default=d.threads)
    p.add_argument("--eot", default=",".join(d.eot),
                   help="comma list from resize,blur,jpeg; empty string for no EOT")
    p.add_argument("--target", choices=("untargeted", "gray"), default=d.target)
    p.add_argument("--full-res", action="store_true")


def from_args(ns: argparse.Namespace, **over) -> ShieldConfig:
    eot = [t for t in (ns.eot or "").split(",") if t]
    kw = dict(model=ns.model, eps=ns.eps, steps=ns.steps, alpha_div=ns.alpha_div, size=ns.size,
              seed=ns.seed, threads=ns.threads, eot=eot, target=ns.target, full_res=ns.full_res)
    kw.update(over)
    return ShieldConfig(**kw)


def seed_everything(seed: int, threads: Optional[int] = None) -> None:
    import random

    import numpy as np
    import torch
    if threads:
        torch.set_num_threads(int(threads))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
