"""FedAvg — sample-count-weighted mean of client deltas (McMahan et al., 2017).

Moved here from core/server.py in Phase 4; server.py re-exports it so every Phase-1/2
import keeps working and exp1 results are unaffected.
"""
from __future__ import annotations

from .base import Aggregator, weighted_mean


class FedAvg(Aggregator):
    name = "fedavg"

    def aggregate(self, updates, cfg, round_num):
        if not updates:
            raise ValueError("no client updates to aggregate")
        delta, w = weighted_mean(updates)
        diag = self._empty_diag()
        diag["accepted"] = [int(u.client_id) for u in updates]
        diag["scores"] = {str(int(u.client_id)): {"weight": float(wi)}
                          for u, wi in zip(updates, w.tolist())}
        return delta, diag
