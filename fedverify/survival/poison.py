"""One malicious hospital for the ours_* methods (Revision 2 §4B). The attacks are NON-ADAPTIVE: fixed rules that do
not react to the screen.

Who: 1 of K hospitals, chosen from the REPEAT seed by the EXISTING attacks.byzantine.attacker_ids (attacker_frac =
1/K), malicious in every round from the first. A calibration cell may pin the attacker instead (cfg.attacker), so the
calibration sees every hospital in that role.

Logistic family: the existing rules of attacks/byzantine.py, unchanged.
  sign_flip   delta -> -delta, after honest local training (apply_delta_attack)
  scaling     delta -> (K / number of attackers) * delta, model replacement (apply_delta_attack)
  label_flip  the attacker trains on y -> 1 - y (poison_batch's y -> (C-1) - y with C = 2); its class weight is
              recomputed from the flipped labels, as every hospital computes its own locally
Tree families, new:
  label_flip_trees  the attacker grows its new trees on its own rows with flipped labels (binary: y -> 1 - y; XGBoost
                    AFT: the event indicator flipped, death <-> censored, times unchanged)
  random_trees      the attacker grows its new trees on random rows from the public probe distribution (probe.py,
                    with its own seed) paired with a random permutation of its own labels: well-formed trees of the
                    honest size whose splits and leaves carry no information about patients

Every attack happens BEFORE the commitment, so what the attacker really sent is what gets committed and screened.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np

from ..attacks.byzantine import DELTA_ATTACKS, apply_delta_attack, attacker_ids, poison_batch
from . import probe
from .config import derive_seed

N_ATTACKERS = 1                 # "1 of K hospitals"; THREAT_MODEL.md: the screen supports at most one


def byzantine_view(cfg: Any, repeat_seed: int) -> SimpleNamespace:
    """The fields the EXISTING attacks.byzantine functions read, for one repeat of a survival run."""
    k = len(cfg.hospitals)
    return SimpleNamespace(attack=cfg.attack, seed=int(repeat_seed), num_clients=k, attacker_frac=N_ATTACKERS / k,
                           attack_start_round=0, sign_flip_scale=1.0)


def attackers_for(cfg: Any, repeat_seed: int) -> list[str]:
    """The malicious hospital(s) of one repeat, as codes: pinned (calibration) or chosen from the repeat seed."""
    if not cfg.attack:
        return []
    if cfg.attacker:
        return [cfg.attacker]
    return [cfg.hospitals[i] for i in sorted(attacker_ids(byzantine_view(cfg, repeat_seed)))]


def is_delta_attack(cfg: Any) -> bool:
    return cfg.attack in DELTA_ATTACKS


def poisoned_delta(delta: Any, cfg: Any, repeat_seed: int, rnd: int, hospital: str, n_attackers: int) -> Any:
    """sign_flip / scaling through the existing apply_delta_attack (a NEW tensor; the input is not changed)."""
    return apply_delta_attack(delta, byzantine_view(cfg, repeat_seed), rnd, cfg.hospitals.index(hospital),
                              n_attackers)


def poisoned_labels(X: Any, y: Any, cfg: Any, repeat_seed: int, rnd: int, hospital: str) -> tuple[Any, Any]:
    """label_flip through the existing poison_batch, with C = 2 classes: y -> 1 - y."""
    return poison_batch(X, y, byzantine_view(cfg, repeat_seed), rnd, cfg.hospitals.index(hospital), num_classes=2)


def flipped(labels: np.ndarray) -> np.ndarray:
    """label_flip_trees: binary labels (or AFT event indicators) y -> 1 - y."""
    return (1 - np.asarray(labels, dtype=np.int64)).astype(np.asarray(labels).dtype)


def random_tree_rows(feature_names: Sequence[str], n: int, *seed_parts: object) -> tuple[np.ndarray, np.ndarray]:
    """random_trees: (n random rows of the public probe distribution, a permutation of range(n) for the labels)."""
    X = probe.draw_rows(feature_names, n, derive_seed("random_trees", "rows", *seed_parts))
    perm = np.random.default_rng(derive_seed("random_trees", "labels", *seed_parts)).permutation(int(n))
    return X, perm


def attack_report(cfg: Any, attackers: Iterable[str], present: Iterable[str], rejected: Iterable[str]) -> dict:
    """Screening quality for one round, the keys of core/runner.py's _attack_report with hospital codes (plus
    `present`, the hospitals screened this round, so rates can be pooled over rounds)."""
    order = list(cfg.hospitals)
    present_s = set(present)
    mal = set(attackers) & present_s
    honest = present_s - mal
    rej = set(rejected) & present_s
    tp, fp, fn = len(rej & mal), len(rej & honest), len(mal - rej)
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if not mal else 0.0)
    rec = tp / (tp + fn) if (tp + fn) else (1.0 if not mal else 0.0)
    return {"attack": cfg.attack, "active": bool(mal), "attackers": [h for h in order if h in mal],
            "rejected": [h for h in order if h in rej], "present": [h for h in order if h in present_s],
            "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
            "false_exclusion_rate": (fp / len(honest)) if honest else 0.0}


def _rates(tp: int, fp: int, fn: int, honest: int, rounds: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn, "honest_hospital_rounds": honest, "attacker_hospital_rounds": tp + fn,
            "rounds": rounds, "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
            "false_exclusion_rate": fp / honest if honest else None}


def pooled_detection(reports: list[dict]) -> dict:
    """Detection over many rounds, pooled over hospital-rounds: precision = TP / (TP + FP), recall = TP / (TP + FN),
    false-exclusion rate = FP / honest hospital-rounds. None where the denominator is zero."""
    return _rates(sum(r["tp"] for r in reports), sum(r["fp"] for r in reports), sum(r["fn"] for r in reports),
                  sum(len(r["present"]) - len(r["attackers"]) for r in reports), len(reports))


def combine_detection(dets: list[dict]) -> dict:
    """pooled_detection over several repeats at once (sums their counts)."""
    return _rates(*(sum(d[k] for d in dets) for k in ("tp", "fp", "fn", "honest_hospital_rounds", "rounds")))
