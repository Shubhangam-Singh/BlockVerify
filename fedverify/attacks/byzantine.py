"""Byzantine client attacks.

Two families, applied at different points in the round:

  DATA attacks   corrupt the client's training batches, so the malicious delta is what
                 that corrupted data actually produced (label_flip, backdoor).
  DELTA attacks  replace or transform the delta after honest local training
                 (sign_flip, gaussian, zero, scaling).

Both are applied BEFORE the Phase-3 commitment, because the commitment must bind what the
client really sent — an attacker's poisoned delta is committed, then rejected by the
aggregator, which is precisely the accept/reject lineage FedVerify anchors on chain.

Attacker identity depends only on the seed and the grid coordinates, never on wall-clock
time or iteration order, so a cell is reproducible and the SAME clients are malicious for
every aggregator being compared.
"""
from __future__ import annotations

from typing import Set

import numpy as np
import torch

DELTA_ATTACKS = {"sign_flip", "gaussian", "zero", "scaling"}
DATA_ATTACKS = {"label_flip", "backdoor"}
ATTACKS = {"none"} | DELTA_ATTACKS | DATA_ATTACKS


def attacker_ids(cfg) -> Set[int]:
    """Deterministic attacker set: depends on seed, K and attacker_frac only."""
    if not cfg.attack or cfg.attack == "none" or cfg.attacker_frac <= 0:
        return set()
    n = int(round(float(cfg.attacker_frac) * int(cfg.num_clients)))
    n = max(0, min(n, int(cfg.num_clients)))
    if n == 0:
        return set()
    rng = np.random.default_rng(int(cfg.seed) * 7919 + 13)
    return set(int(i) for i in rng.choice(int(cfg.num_clients), size=n, replace=False))


def attack_active(cfg, round_num: int) -> bool:
    """Attacks stay dormant until attack_start_round, so the run has a clean warm-up."""
    if not cfg.attack or cfg.attack == "none" or cfg.attacker_frac <= 0:
        return False
    return int(round_num) >= int(getattr(cfg, "attack_start_round", 0) or 0)


def _rng(cfg, round_num: int, client_id: int) -> torch.Generator:
    """Per (seed, round, client) generator — reproducible and independent across clients."""
    g = torch.Generator()
    g.manual_seed((int(cfg.seed) * 1_000_003 + int(round_num) * 10_007
                   + int(client_id) * 31 + 7) % (2 ** 63 - 1))
    return g


def apply_delta_attack(delta: torch.Tensor, cfg, round_num: int, client_id: int,
                       n_attackers: int) -> torch.Tensor:
    """Transform one attacker's delta. Returns a NEW tensor; the input is not mutated."""
    a = cfg.attack

    if a == "sign_flip":
        # delta -> -s * delta: point the update the wrong way, s times as hard.
        return -float(getattr(cfg, "sign_flip_scale", 1.0)) * delta

    if a == "gaussian":
        # delta -> N(0, sigma^2): pure noise, carrying no information about the data.
        sigma = float(getattr(cfg, "gaussian_sigma", 1.0))
        return torch.normal(0.0, sigma, size=delta.shape, generator=_rng(cfg, round_num, client_id),
                            dtype=delta.dtype)

    if a == "zero":
        # delta -> 0: the free-rider / drop attack.
        return torch.zeros_like(delta)

    if a == "scaling":
        # Model replacement (Bagdasaryan et al., AISTATS 2020): scale by K/f so that after
        # FedAvg divides by K, the attackers' contribution survives at roughly full weight.
        f = max(1, int(n_attackers))
        return (float(cfg.num_clients) / f) * delta

    raise ValueError(f"{a!r} is not a delta-level attack; expected one of {sorted(DELTA_ATTACKS)}")


def poison_batch(xb: torch.Tensor, yb: torch.Tensor, cfg, round_num: int, client_id: int,
                 num_classes: int, spec=None):
    """Corrupt one training batch for a DATA-level attack. Returns (xb, yb)."""
    a = cfg.attack

    if a == "label_flip":
        # y -> (C-1) - y. For the 10-class datasets this is exactly the specified y -> 9-y.
        return xb, (num_classes - 1 - yb)

    if a == "backdoor":
        from .backdoor import apply_trigger
        frac = float(getattr(cfg, "poison_frac", 0.5))
        target = int(getattr(cfg, "backdoor_target", 0))
        g = _rng(cfg, round_num, client_id)
        mask = torch.rand(yb.shape[0], generator=g) < frac
        if mask.any():
            xb = xb.clone()
            yb = yb.clone()
            xb[mask] = apply_trigger(xb[mask], spec)
            yb[mask] = target
        return xb, yb

    raise ValueError(f"{a!r} is not a data-level attack; expected one of {sorted(DATA_ATTACKS)}")
