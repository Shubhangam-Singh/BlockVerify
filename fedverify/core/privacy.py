"""Sample-level differential privacy (DP-SGD) for FedVerify clients.

=============================================================================
 THE ACCOUNTING BUG THIS MODULE EXISTS TO AVOID
=============================================================================
A very common error in DP + federated-learning code is to account privacy
*per round*: solve for a noise multiplier that achieves epsilon over one
round's local steps, then run R rounds with it. That silently spends privacy
R times over — the client's真 epsilon is roughly the R-fold composition of the
per-round guarantee, not the target.

Privacy is a property of a client's ENTIRE participation. So we account over
each client's TOTAL local SGD steps across ALL rounds:

    steps_per_round_k = local_epochs * ceil(n_k / batch_size)
    total_steps_k     = rounds * steps_per_round_k
    sample_rate_k     = batch_size / n_k
    sigma_k           = get_noise_multiplier(target_epsilon, target_delta,
                                             sample_rate_k, steps=total_steps_k,
                                             accountant="rdp")     # solved ONCE

sigma_k is then held FIXED for the whole run. Each client owns ONE
RDPAccountant that persists across rounds and is stepped once per optimizer
step, so the epsilon we report at the end is the realized composition over
every step the client actually took, not a per-round figure.

Note on subsampling: sigma is solved for ceil(n_k/batch_size) steps per epoch,
which upper-bounds the number of steps Poisson sampling actually takes
(int(1/sample_rate)). Solving for at-least-as-many steps means the realized
guarantee is never weaker than the target — the error is in the safe
direction, and realized epsilon comes in at or slightly below target.
=============================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


def dp_enabled(cfg) -> bool:
    """DP is off when epsilon is None or infinite — then Opacus is never involved."""
    eps = getattr(cfg, "epsilon", None)
    return not (eps is None or (isinstance(eps, float) and math.isinf(eps)))


def steps_per_round(n_k: int, batch_size: int, local_epochs: int) -> int:
    if n_k <= 0:
        raise ValueError("client has no samples")
    return int(local_epochs * math.ceil(n_k / batch_size))


def total_steps(n_k: int, batch_size: int, local_epochs: int, rounds: int) -> int:
    return int(rounds * steps_per_round(n_k, batch_size, local_epochs))


@dataclass
class PrivacyPlan:
    """Per-client DP parameters, solved once and held fixed for the whole run."""
    client_id: int
    sigma: float                 # noise multiplier
    max_grad_norm: float         # per-sample clipping bound C
    sample_rate: float           # batch_size / n_k
    steps_per_round: int
    total_steps: int
    target_epsilon: float
    delta: float
    n_samples: int
    accountant: Any = field(default=None, repr=False)   # ONE per client, spans rounds

    def realized_epsilon(self) -> Optional[float]:
        if self.accountant is None or not getattr(self.accountant, "history", None):
            return None
        return float(self.accountant.get_epsilon(delta=self.delta))

    def steps_taken(self) -> int:
        if self.accountant is None:
            return 0
        return int(sum(n for _, _, n in getattr(self.accountant, "history", [])))


def validate_dp_compatible(model) -> None:
    """Raise a clear error if the model cannot be made private (e.g. BatchNorm)."""
    from opacus.validators import ModuleValidator
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        raise ValueError(
            "model is not DP-compatible (Opacus ModuleValidator): "
            + "; ".join(str(e) for e in errors)
            + ". Use GroupNorm instead of BatchNorm (see CLAUDE.md 'Modelling')."
        )


def plan_privacy(cfg, client_id: int, n_k: int) -> Optional[PrivacyPlan]:
    """Solve sigma ONCE for this client's whole participation. None when DP is off."""
    if not dp_enabled(cfg):
        return None
    from opacus.accountants import RDPAccountant
    from opacus.accountants.utils import get_noise_multiplier

    spr = steps_per_round(n_k, cfg.batch_size, cfg.local_epochs)
    tot = int(cfg.rounds * spr)
    sample_rate = min(1.0, cfg.batch_size / n_k)

    sigma = get_noise_multiplier(
        target_epsilon=float(cfg.epsilon),
        target_delta=float(cfg.delta),
        sample_rate=sample_rate,
        steps=tot,                       # TOTAL across all rounds — see module docstring
        accountant="rdp",
    )
    return PrivacyPlan(
        client_id=int(client_id), sigma=float(sigma),
        max_grad_norm=float(cfg.max_grad_norm), sample_rate=float(sample_rate),
        steps_per_round=spr, total_steps=tot,
        target_epsilon=float(cfg.epsilon), delta=float(cfg.delta),
        n_samples=int(n_k), accountant=RDPAccountant(),
    )


def make_private(model, optimizer, data_loader, plan: PrivacyPlan, criterion=None):
    """Wrap with Opacus using the client's FIXED sigma and PERSISTENT accountant.

    Returns (model, optimizer, data_loader). The engine's accountant is replaced by the
    client's own so privacy composes across rounds instead of resetting each round.
    """
    from opacus import PrivacyEngine

    validate_dp_compatible(model)
    engine = PrivacyEngine(accountant="rdp")
    engine.accountant = plan.accountant          # persist across rounds (the whole point)

    out = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=plan.sigma,             # fixed; never re-solved per round
        max_grad_norm=plan.max_grad_norm,
        poisson_sampling=True,
        **({"criterion": criterion} if criterion is not None else {}),
    )
    model, optimizer, data_loader = out[0], out[1], out[2]
    return model, optimizer, data_loader


def privacy_report(plan: Optional[PrivacyPlan]) -> Optional[dict]:
    """What gets stored in ClientUpdate.privacy."""
    if plan is None:
        return None
    return {
        "sigma": plan.sigma,
        "C": plan.max_grad_norm,
        "sample_rate": plan.sample_rate,
        "steps": plan.total_steps,
        "steps_per_round": plan.steps_per_round,
        "steps_taken": plan.steps_taken(),
        "realized_eps": plan.realized_epsilon(),
        "target_eps": plan.target_epsilon,
        "delta": plan.delta,
        "n_samples": plan.n_samples,
    }
