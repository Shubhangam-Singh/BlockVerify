"""Screening for the ours_* methods (Revision 2 §4B) and the survival calibration file.

Logistic (ours_lr): the EXISTING FedVerify-Forensics aggregator, unchanged (aggregators/forensics.py).

Trees (ours_rf, ours_xgb): TreeProbeScreen. Trees have no deltas to compare, so each hospital's NEW trees of the round,
rebuilt from the committed bytes, predict the public probe set (probe.py):
    RF       the mean over its new trees of P(event)
    XGBoost  the margin its new trees add (their tree sum)
  distance_k = root-mean-square difference between hospital k's probe predictions and the per-row median across
               the hospitals
  score_k    = robust z of distance_k across the hospitals, one-sided (only a LARGE distance is suspicious), with the
               median and 1.4826 x MAD of the deployed probe (forensics.robust_z -> eval_lib.bv_find_outliers)
  health     eval_lib.bv_layer_health on the probe predictions and on the leaf values, with Forensics' hard flags
             (NaN, Inf, |x| > 100, long constant run): a flagged hospital is rejected outright and left out of the
             median, like a NaN delta in Forensics
A hospital is rejected if a hard flag fires or score > tau; the survivors' trees are appended. With K = 4 or 5 the
median and MAD come from four or five numbers, which is why THREAT_MODEL.md claims at most one Byzantine hospital.

tau is never hardcoded. It is read from results/calibration/taus_survival.json (analysis/calibrate_survival.py),
keyed by (dataset, family, epsilon, K), and constructing a screen without it raises.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..aggregators.forensics import _eval_lib, _hard_flags, robust_z
from .config import eps_tag

SCORE = "combined"              # the score compared with tau (Forensics' name; the tree screen has one probe score)


class TreeProbeScreen:
    name = "probe_screen"
    detects = True

    def __init__(self, tau: Optional[float]):
        if tau is None:
            raise ValueError("the tree screen needs tau, and tau must come from calibration "
                             "(results/calibration/taus_survival.json via analysis/calibrate_survival.py); there is "
                             "deliberately no default")
        self.tau = float(tau)

    def screen(self, predictions: dict[str, np.ndarray],
               leaves: Optional[dict[str, np.ndarray]] = None) -> tuple[list[str], dict]:
        """predictions: hospital -> its new trees' probe predictions (same probe rows for all, hospital order).
        leaves: hospital -> its new trees' leaf values (health check only). Returns (accepted, diag)."""
        hospitals = list(predictions)
        if not hospitals:
            raise ValueError("no hospital updates to screen")
        probes = _eval_lib()
        flags, scores = {}, {}
        for h in hospitals:
            hp = probes.bv_layer_health(np.asarray(predictions[h], dtype=np.float64))
            flags[h] = _hard_flags(hp)
            if leaves is not None and h in leaves:
                flags[h] += ["leaves:" + f for f in _hard_flags(probes.bv_layer_health(
                    np.asarray(leaves[h], dtype=np.float64)))]
            scores[h] = {"health_flags": flags[h], "max_abs": hp["max_abs"]}
        clean = [h for h in hospitals if not flags[h]]
        if len(clean) >= 2:
            P = np.stack([np.asarray(predictions[h], dtype=np.float64) for h in clean])      # (Kc, probe rows)
            med = np.median(P, axis=0)
            dist = np.sqrt(np.mean((P - med) ** 2, axis=1))
            z = robust_z(dist, "high")
            for j, h in enumerate(clean):
                scores[h].update(distance=float(dist[j]), s_probe=float(z[j]), combined=float(z[j]))
        else:                                   # too few healthy hospitals to form a population statistic
            for h in clean:
                scores[h].update(distance=0.0, s_probe=0.0, combined=0.0)

        accepted, rejected = [], []
        for h in hospitals:
            if flags[h]:
                scores[h]["rejected_by"] = "health:" + ",".join(flags[h])
                rejected.append(h)
            elif scores[h][SCORE] > self.tau:
                scores[h]["rejected_by"] = "probe_distance"
                rejected.append(h)
            else:
                accepted.append(h)
        diag = {"accepted": accepted, "rejected": rejected, "scores": scores, "tau": self.tau}
        if not accepted:
            # Every hospital looked untrustworthy: append nothing rather than the trees just rejected.
            diag["fallback"] = "all hospitals rejected; no trees appended"
        return accepted, diag


def leaf_values(model: Any) -> np.ndarray:
    """The leaf outputs of a model rebuilt from committed bytes (NumpyForest or NumpyBooster)."""
    trees = getattr(model, "trees", [])
    if not trees:
        return np.zeros(0)
    if hasattr(trees[0], "children_left"):                                       # random forest
        return np.concatenate([t.value[t.children_left == -1].ravel() for t in trees])
    return np.concatenate([t.leaf[t.is_leaf] for t in trees]).astype(np.float64)  # XGBoost


# ── the calibration file ─────────────────────────────────────────────────────
def tau_key(dataset: str, family: str, epsilon: Optional[float], k: int) -> str:
    """taus_survival.json key: dataset, family (lr / rf / xgb / xgb_aft), epsilon ('inf' = no DP), K hospitals."""
    return f"dataset={dataset}|family={family}|eps={eps_tag(epsilon)}|K={int(k)}"


def cfg_tau_key(cfg: Any) -> str:
    return tau_key(cfg.dataset, cfg.calibration_family, cfg.epsilon, len(cfg.hospitals))


def load_tau(path: str | Path, key: str, policy: str = "tau_fpr5") -> tuple[float, str]:
    """(tau, provenance) for EXACTLY this key; raises if the file or the key is missing. No fallback to another key:
    a tau calibrated for a different dataset, family, epsilon or K is not this run's tau."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `python3 -m fedverify.analysis.calibrate_survival` first: tau "
                                "must come from calibration, never from a constant.")
    payload = json.loads(p.read_text(encoding="utf-8"))
    entry = (payload.get("by_key") or {}).get(key, {}).get(SCORE)
    if not entry or policy not in entry:
        raise KeyError(f"{p} has no {SCORE!r} {policy} for {key}; calibrate this (dataset, family, epsilon, K)")
    tau = float(entry[policy])
    if math.isnan(tau):
        raise ValueError(f"{p}: tau for {key} is NaN")
    source = f"{p}:{key}:{policy}" + (" (SMOKE calibration)" if payload.get("smoke") else "")
    return tau, source
