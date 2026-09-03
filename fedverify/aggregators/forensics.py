"""FedVerify-Forensics — BlockVerify's tamper probes applied ACROSS CLIENTS.

BlockVerify localises a tampered layer by running a robust median/MAD outlier probe over
the WEIGHTS of a model. The observation this aggregator rests on is that the same probe
family works unchanged one level up: within a federated round, the population is the K
client deltas rather than the D weights of one tensor, and a Byzantine client is an
outlier in that population the same way a poisoned layer is an outlier in a model.

So the probes are IMPORTED from evaluation/eval_lib.py — the byte-identical port of the
deployed in-browser detector (see docs/EVALUATION.md §3) — not re-implemented:

  bv_find_outliers  supplies the median and the 1.4826*MAD scale, so every score below is
                    a robust z on exactly the scale the paper already calibrated for layers
  bv_layer_health   supplies the hard flags: NaN, +-Inf, |w|>100, entropy, constant runs

Four per-client scores, each reduced to a robust z ACROSS clients so they are mutually
comparable and a single tau decides all of them:

  s_norm    |z| of ||delta_k||_2            two-sided: scaling attacks inflate it,
                                            zero/drop attacks collapse it
  s_dir     one-sided z of cos(delta_k, coordinate-wise median delta)
                                            only LOW similarity is suspicious
  s_coord   one-sided z of the fraction of coordinates where
            |delta_kj - med_j| / scale_j > tau_coord
  s_health  hard flags -> immediate rejection, independent of tau

combined = max(s_norm, s_dir, s_coord), mirroring the deployed detector's max-robust-z
operating point. A client is rejected if a hard health flag fires OR combined > tau.
Survivors are FedAvg'd.

tau is NEVER hardcoded: it comes from cfg.tau, which experiments populate from
results/calibration/taus.json (analysis/calibrate.py). Constructing this aggregator
without a tau is an error, not a silent default.
"""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import torch

from .base import Aggregator, stack_deltas, weighted_mean

_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "evaluation")
_probes = None


def _eval_lib():
    """Import the deployed probe port lazily and once."""
    global _probes
    if _probes is None:
        if _EVAL_DIR not in sys.path:
            sys.path.insert(0, _EVAL_DIR)
        import eval_lib
        _probes = eval_lib
    return _probes


# ── robust z across clients, using the deployed probe's median/scale ─────────
def robust_z(values: np.ndarray, one_sided: str = "both") -> np.ndarray:
    """Per-element robust z of a small population, via bv_find_outliers' median+scale.

    bv_find_outliers reports only max_z, so the median and the 1.4826*MAD scale are taken
    from it and the per-client z is formed here — the statistic is the deployed one, only
    the reduction differs (per-client instead of max).
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return v
    finite = v[np.isfinite(v)]
    if finite.size < 4:
        # bv_find_outliers refuses n < 4; fall back to the same median/MAD by hand so
        # small-K rounds still get a defined (if low-power) score.
        med = float(np.median(finite)) if finite.size else 0.0
        mad = float(np.median(np.abs(finite - med))) * 1.4826 if finite.size else 0.0
        scale = mad if mad > 1e-9 else (float(finite.std()) if finite.size and finite.std() > 1e-9 else 1.0)
    else:
        o = _eval_lib().bv_find_outliers(finite)
        med, scale = o["median"], o["scale"]
        scale = scale if scale > 1e-9 else 1.0

    dev = v - med
    if one_sided == "high":
        dev = np.maximum(dev, 0.0)          # only unusually LARGE values are suspicious
    elif one_sided == "low":
        dev = np.maximum(-dev, 0.0)         # only unusually SMALL values are suspicious
    else:
        dev = np.abs(dev)
    z = dev / scale
    return np.where(np.isfinite(z), z, np.inf)


def _hard_flags(health: dict) -> List[str]:
    """Health conditions that reject a client outright, regardless of tau."""
    flags = []
    if health["nan"]:
        flags.append("nan")
    if health["inf"]:
        flags.append("inf")
    if health["extremes"]:
        flags.append("extreme_magnitude")
    # A long constant run is the low-entropy-poisoning signature from docs/EVALUATION.md
    # §5.4. The statistic is meaningless on short vectors — a 16-element delta is
    # "all one run" trivially — so it is only applied above eval_lib's own
    # min_elems=1024 threshold for a tensor worth analysing.
    if health["n"] >= 1024 and health["max_run"] >= health["n"] // 2:
        flags.append("constant_run")
    return flags


class Forensics(Aggregator):
    name = "forensics"
    detects = True

    def aggregate(self, updates, cfg, round_num):
        if not updates:
            raise ValueError("no client updates to aggregate")

        tau = getattr(cfg, "tau", None)
        if tau is None:
            raise ValueError(
                "forensics aggregator requires cfg.tau. It must come from calibration "
                "(results/calibration/taus.json via analysis/calibrate.py); there is "
                "deliberately no default, because a hardcoded threshold is the bug this "
                "aggregator exists to avoid.")
        tau = float(tau)
        tau_coord = float(getattr(cfg, "tau_coord", 3.0))

        ids = [int(u.client_id) for u in updates]
        k = len(updates)
        probes = _eval_lib()

        # ── 1. health first: hard-flagged clients are excluded from the statistics too,
        #       so a NaN delta cannot poison the median every other score is measured against
        health = {cid: probes.bv_layer_health(u.delta.detach().cpu().numpy())
                  for cid, u in zip(ids, updates)}
        flags = {cid: _hard_flags(health[cid]) for cid in ids}
        clean_pos = [i for i, cid in enumerate(ids) if not flags[cid]]

        scores = {str(cid): {"health_flags": flags[cid],
                             "entropy": health[cid]["entropy"],
                             "max_abs": health[cid]["max_abs"]} for cid in ids}

        if len(clean_pos) >= 2:
            stacked = stack_deltas([updates[i] for i in clean_pos])      # (Kc, D)

            # s_norm — two-sided: both inflation (scaling) and collapse (zero) are anomalous
            norms = stacked.norm(dim=1).numpy()
            z_norm = robust_z(norms, "both")

            # s_dir — one-sided low: only poor alignment with the consensus is suspicious
            med_delta = stacked.median(dim=0).values
            mn = float(med_delta.norm())
            if mn > 1e-12:
                cos = (stacked @ med_delta / (stacked.norm(dim=1) * mn)).numpy()
            else:
                cos = np.ones(len(clean_pos))
            z_dir = robust_z(cos, "low")

            # s_coord — one-sided high: an unusual share of per-coordinate outliers
            med_j = stacked.median(dim=0).values
            dev_j = (stacked - med_j).abs()
            mad_j = dev_j.median(dim=0).values * 1.4826
            scale_j = torch.where(mad_j > 1e-12, mad_j, torch.ones_like(mad_j))
            frac = ((dev_j / scale_j) > tau_coord).to(torch.float64).mean(dim=1).numpy()
            z_coord = robust_z(frac, "high")

            for j, i in enumerate(clean_pos):
                cid = ids[i]
                combined = float(max(z_norm[j], z_dir[j], z_coord[j]))
                which = ["s_norm", "s_dir", "s_coord"][int(np.argmax(
                    [z_norm[j], z_dir[j], z_coord[j]]))]
                scores[str(cid)].update(
                    s_norm=float(z_norm[j]), s_dir=float(z_dir[j]),
                    s_coord=float(z_coord[j]), cosine=float(cos[j]),
                    norm=float(norms[j]), combined=combined, top_probe=which)
        else:
            # too few healthy clients to form a population statistic
            for i in clean_pos:
                scores[str(ids[i])].update(s_norm=0.0, s_dir=0.0, s_coord=0.0,
                                           combined=0.0, top_probe=None)

        accepted, rejected = [], []
        for cid in ids:
            if flags[cid]:
                scores[str(cid)]["rejected_by"] = "health:" + ",".join(flags[cid])
                rejected.append(cid)
            elif scores[str(cid)].get("combined", 0.0) > tau:
                scores[str(cid)]["rejected_by"] = scores[str(cid)].get("top_probe") or "combined"
                rejected.append(cid)
            else:
                accepted.append(cid)

        diag = self._empty_diag()
        diag.update(accepted=sorted(accepted), rejected=sorted(rejected),
                    scores=scores, tau=tau, tau_coord=tau_coord)

        if not accepted:
            # Every client looked untrustworthy. Contributing nothing is the honest
            # outcome: do not fall back to averaging what was just rejected.
            diag["fallback"] = "all clients rejected; zero update applied"
            return torch.zeros_like(updates[0].delta), diag

        acc_set = set(accepted)
        delta, w = weighted_mean(updates, acc_set)
        # weights come back in `updates` order, not sorted-id order
        for cid, wi in zip([c for c in ids if c in acc_set], w.tolist()):
            scores[str(cid)]["weight"] = float(wi)
        return delta, diag
