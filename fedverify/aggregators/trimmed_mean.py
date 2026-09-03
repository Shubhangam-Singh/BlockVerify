"""Coordinate-wise beta-trimmed mean (Yin et al., ICML 2018).

Per coordinate, drop the beta largest and beta smallest values across clients and average
the rest. beta is a COUNT of clients trimmed from each end, derived from cfg.trim_beta
(a fraction) and clamped so at least one client survives.
"""
from __future__ import annotations

import torch

from .base import Aggregator, stack_deltas


class TrimmedMean(Aggregator):
    name = "trimmed_mean"

    def aggregate(self, updates, cfg, round_num):
        if not updates:
            raise ValueError("no client updates to aggregate")
        k = len(updates)
        beta = int(getattr(cfg, "trim_beta", 0.2) * k)
        beta = max(0, min(beta, (k - 1) // 2))          # always leave >= 1 client

        stacked = stack_deltas(updates)                  # (K, D) float64
        if beta == 0:
            delta = stacked.mean(dim=0)
        else:
            srt, _ = torch.sort(stacked, dim=0)
            delta = srt[beta:k - beta].mean(dim=0)
        # Cast back: stack_deltas promotes to float64 for numerical headroom, but the
        # runner does global_params + delta, so returning a double here would silently
        # promote the whole model to float64 from this round on.
        delta = delta.to(updates[0].delta.dtype)

        diag = self._empty_diag()
        diag["accepted"] = [int(u.client_id) for u in updates]
        diag["scores"] = {str(int(u.client_id)): {} for u in updates}
        diag["beta"] = beta
        return delta, diag
