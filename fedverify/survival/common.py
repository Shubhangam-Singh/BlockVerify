"""Pieces shared by the LR, RF and XGBoost families: the per-round output and the D10 id translation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .config import HOSPITALS

RISK = "risk"            # train_risk / test_risk are P(event): higher = more likely to die / to have disease
LOG_TIME = "log_time"    # XGBoost AFT: predicted log survival time, higher = LONGER survival (Revision 2 §2.2)


@dataclass
class RoundOutput:
    """What one training round hands the runner: a score for every training and test row of the fold, plus the
    record fields that keep the existing rounds.jsonl structure (diag, per_client_num_samples, mean_train_loss).
    Rows of a hospital that a method does not model (heart: Switzerland under local_*) carry NaN."""
    round: int
    train_risk: np.ndarray
    test_risk: np.ndarray
    diag: dict
    per_client_num_samples: dict
    mean_train_loss: Optional[float]
    train_wall_s: float
    extra: dict = field(default_factory=dict)
    privacy: Optional[dict] = None     # {hospital code: core.privacy.privacy_report(...)} under DP, else None
    score_kind: str = RISK             # RISK or LOG_TIME: what train_risk / test_risk hold
    commitment: Optional[dict] = None  # chain_* only: one commitments.jsonl line (leaves and proofs)


def codes_diag(diag: dict, hospitals: Sequence[str] = HOSPITALS) -> dict:
    """Existing aggregators report integer client ids; Survival writes hospital codes everywhere (D10).
    `hospitals` is the dataset's canonical order, i.e. the order of the integer ids."""
    def code(i: object) -> str:
        return hospitals[int(i)]

    out = dict(diag)
    out["accepted"] = [code(i) for i in diag.get("accepted", [])]
    out["rejected"] = [code(i) for i in diag.get("rejected", [])]
    out["scores"] = {code(k): v for k, v in (diag.get("scores") or {}).items()}
    return out


def plain_diag(hospitals: list[str], scores: Optional[dict] = None) -> dict:
    """diag for rounds with no screening: every hospital's contribution was used."""
    return {"accepted": list(hospitals), "rejected": [], "scores": dict(scores or {})}


def local_hospitals(cfg: object, present: Sequence[str]) -> list[str]:
    """Hospitals that get their own model under local_* (heart excludes Switzerland, Revision 2 §2.1)."""
    excluded = set(getattr(cfg, "local_exclude", ()) or ())
    return [h for h in present if h not in excluded]
