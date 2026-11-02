"""DP plumbing for the Survival logistic family: the EXISTING core/privacy.py, made affordable.

Measured on this laptop, ~90% of a DP fold-run was Opacus' pure-Python RDP arithmetic, not training: solving sigma
takes ~2 s per hospital and every 'realised epsilon' read ~0.2 s. Both are pure functions of a few numbers that repeat
across folds and repeats (a hospital's training size takes one or two values), so two caches remove the repetition
without changing any result:

  hospital_plan()       the EXISTING plan_privacy() (sigma solved once over the TOTAL steps of the run) is called once
                        per distinct (epsilon, delta, C, batch, local_epochs, rounds, n_k); every hospital and fold
                        gets a copy of that plan with its OWN fresh accountant.
  CachedRDPAccountant   Opacus' RDPAccountant with get_epsilon() memoised on (step history, delta). step() and the RDP
                        maths are Opacus' own; the cache only skips recomputing an identical answer.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

from opacus.accountants import RDPAccountant

from ..core.privacy import PrivacyPlan, plan_privacy

_EPSILON_CACHE: dict[tuple, float] = {}


class CachedRDPAccountant(RDPAccountant):
    """RDPAccountant whose get_epsilon() is memoised; history and stepping are unchanged."""

    def get_epsilon(self, delta: float, alphas: Optional[list] = None, **kwargs: Any) -> float:
        key = (tuple(self.history), float(delta), None if alphas is None else tuple(alphas),
               tuple(sorted(kwargs.items())))
        if key not in _EPSILON_CACHE:
            _EPSILON_CACHE[key] = float(super().get_epsilon(delta=delta, alphas=alphas, **kwargs))
        return _EPSILON_CACHE[key]


def plan_key(cfg: Any, n_k: int) -> tuple:
    """Everything plan_privacy() reads: two hospitals with the same key get the same sigma."""
    return (float(cfg.epsilon), float(cfg.delta), float(cfg.max_grad_norm), int(cfg.batch_size),
            int(cfg.local_epochs), int(cfg.rounds), int(n_k))


def hospital_plan(cfg: Any, client_index: int, n_k: int, cache: Optional[dict] = None) -> PrivacyPlan:
    """plan_privacy()'s plan for this hospital, with a fresh accountant that is this fold-run's alone."""
    key = plan_key(cfg, n_k)
    template = None if cache is None else cache.get(key)
    if template is None:
        template = plan_privacy(cfg, client_index, n_k)       # EXISTING solver: sigma over TOTAL steps
        if template is None:
            raise ValueError("hospital_plan() called with DP disabled (epsilon None or inf)")
        if cache is not None:
            cache[key] = template
    return dataclasses.replace(template, client_id=int(client_index), accountant=CachedRDPAccountant())
