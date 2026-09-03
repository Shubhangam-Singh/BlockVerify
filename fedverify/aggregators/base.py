"""Aggregator interface. Every aggregator returns (delta, diag).

`diag` ALWAYS carries accepted / rejected / scores so Phase 3 can commit an
accept-reject lineage on-chain and Phase 4 can score Byzantine screening without
changing this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import torch


class Aggregator(ABC):
    name = "base"
    #: True only for rules that make a per-client accept/reject DECISION. Krum's top-1
    #: selection and the coordinate-wise rules discard clients without claiming they are
    #: malicious, so scoring them as detectors would be a category error in Table 2b.
    detects = False

    @abstractmethod
    def aggregate(self, updates: Sequence, cfg, round_num: int) -> Tuple[torch.Tensor, dict]:
        """Return (aggregated_delta, diag). diag has accepted/rejected/scores."""

    @staticmethod
    def _empty_diag() -> dict:
        # Exactly the Phase-1 shape. `detects` is a CLASS property read from the registry
        # (see analysis/make_tables.table2b); putting it in diag would change the bytes of
        # every rounds.jsonl record, including Phase-1/2 cells.
        return {"accepted": [], "rejected": [], "scores": {}}


def stack_deltas(updates) -> torch.Tensor:
    """(K, D) float64 matrix of client deltas — the common input to every robust rule."""
    return torch.stack([u.delta.to(torch.float64) for u in updates])


def weighted_mean(updates, ids=None) -> torch.Tensor:
    """Sample-count-weighted mean over `ids` (default: all). The FedAvg kernel."""
    sel = [u for u in updates if ids is None or int(u.client_id) in ids]
    if not sel:
        raise ValueError("no client updates to aggregate")
    w = torch.tensor([float(u.num_samples) for u in sel], dtype=torch.float64)
    if float(w.sum()) <= 0:
        raise ValueError("total num_samples across updates is zero")
    w = w / w.sum()
    stacked = torch.stack([u.delta.to(torch.float64) for u in sel])
    return (stacked * w.unsqueeze(1)).sum(dim=0).to(sel[0].delta.dtype), w
