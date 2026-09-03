"""Coordinate-wise median (Yin et al., ICML 2018).

Each output coordinate is the median of that coordinate across clients, so a minority of
arbitrarily corrupted clients cannot move it far. Unlike Krum it keeps no notion of which
client was bad, so `rejected` is empty by construction — that is the honest report, not an
omission, and it is exactly the gap FedVerify-Forensics fills.
"""
from __future__ import annotations

import torch

from .base import Aggregator, stack_deltas


class CoordinateMedian(Aggregator):
    name = "median"

    def aggregate(self, updates, cfg, round_num):
        if not updates:
            raise ValueError("no client updates to aggregate")
        stacked = stack_deltas(updates)
        delta = stacked.median(dim=0).values.to(updates[0].delta.dtype)
        diag = self._empty_diag()
        diag["accepted"] = [int(u.client_id) for u in updates]
        diag["scores"] = {str(int(u.client_id)): {} for u in updates}
        return delta, diag
