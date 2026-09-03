"""Krum and Multi-Krum (Blanchard et al., NeurIPS 2017).

Krum picks the single client whose delta is closest to its n-f-2 nearest neighbours,
where f is the assumed number of Byzantine clients. Multi-Krum averages the m
best-scoring clients instead of taking only one, which recovers most of FedAvg's
statistical efficiency while keeping the same robustness argument.

Krum needs n - f - 2 >= 1 to have any neighbours to score against; with too few clients
for the assumed f it degrades to scoring against every other client rather than failing,
and says so in diag.
"""
from __future__ import annotations

import torch

from .base import Aggregator, stack_deltas, weighted_mean


def _krum_scores(stacked: torch.Tensor, n_neighbours: int) -> torch.Tensor:
    """Sum of squared distances to each client's n_neighbours closest peers."""
    d = torch.cdist(stacked, stacked) ** 2               # (K, K)
    d.fill_diagonal_(float("inf"))                       # never score against self
    closest, _ = torch.sort(d, dim=1)
    return closest[:, :n_neighbours].sum(dim=1)


class Krum(Aggregator):
    name = "krum"
    multi = False
    detects = False

    def aggregate(self, updates, cfg, round_num):
        if not updates:
            raise ValueError("no client updates to aggregate")
        k = len(updates)
        ids = [int(u.client_id) for u in updates]
        if k == 1:
            return updates[0].delta.clone(), {**self._empty_diag(), "accepted": ids,
                                              "scores": {str(ids[0]): {"krum": 0.0}}}

        f = int(round(getattr(cfg, "attacker_frac", 0.0) * cfg.num_clients))
        n_neighbours = k - f - 2
        degraded = n_neighbours < 1
        if degraded:                                     # too few clients for assumed f
            n_neighbours = k - 1

        stacked = stack_deltas(updates)
        scores = _krum_scores(stacked, n_neighbours)

        m = 1 if not self.multi else max(1, min(k - f, k))
        order = torch.argsort(scores)
        chosen = {ids[i] for i in order[:m].tolist()}

        delta, _ = weighted_mean(updates, chosen)
        diag = self._empty_diag()
        diag["accepted"] = sorted(chosen)
        diag["rejected"] = sorted(set(ids) - chosen)
        diag["scores"] = {str(cid): {"krum": float(s)} for cid, s in zip(ids, scores.tolist())}
        diag["n_neighbours"] = n_neighbours
        diag["assumed_f"] = f
        if degraded:
            diag["degraded"] = "k - f - 2 < 1; scored against all peers"
        return delta, diag


class MultiKrum(Krum):
    name = "multikrum"
    multi = True
    detects = True
