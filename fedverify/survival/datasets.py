"""One entry point for the runner: load and encode the configured dataset (TCGA-BRCA or UCI heart)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import data as tcga
from . import heart
from .config import SurvivalConfig
from .data import REPO_ROOT, Cohort


@dataclass(frozen=True)
class Loaded:
    cohort: Cohort
    X: np.ndarray
    feature_names: list[str]
    vocab_path: Path
    info: dict          # data provenance for config.json


def resolve(path: str) -> Path:
    """Absolute path; a relative path is tried against the cwd first, then the repo root."""
    p = Path(path)
    return (p if p.is_absolute() or p.exists() else REPO_ROOT / p).resolve()


def load(cfg: SurvivalConfig, verbose: bool = False) -> Loaded:
    path = resolve(cfg.data_path)
    rel = os.path.relpath(path, REPO_ROOT) if path.is_relative_to(REPO_ROOT) else str(path)
    if cfg.dataset == "tcga":
        cohort = tcga.load_cohort(path, cfg.hospitals, include_race=cfg.include_race, verbose=verbose)
        X, names = tcga.encode_features(cohort.features, tcga.load_vocab(), cfg.include_race)
        info = {"dataset": "tcga_brca", "path": rel, "sha256": cohort.source_sha256, "patients": cohort.n,
                "deaths": cohort.n_deaths, "dropped_no_usable_time": cohort.dropped, "per_hospital": cohort.counts()}
        if cfg.task == "survival":
            info["aft_times_clipped"] = int((cohort.time_days < cfg.aft_min_time_days).sum())
            info["aft_min_time_days"] = cfg.aft_min_time_days
        return Loaded(cohort, X, names, tcga.VOCAB_PATH, info)
    cohort = heart.load_cohort(path, cfg.heart_mode, cfg.hospitals, verbose=verbose)
    X, names = heart.encode_features(cohort.features, heart.load_vocab(), cfg.heart_mode)
    per = {h: {"patients": c["patients"], "positives": c["deaths"]} for h, c in cohort.counts().items()}
    info = {"dataset": "heart_uci", "path": rel, "sha256": cohort.source_sha256, "mode": cfg.heart_mode,
            "patients": cohort.n, "positives": cohort.n_deaths, "dropped_incomplete": cohort.dropped,
            "per_hospital": per, "local_excluded": list(cfg.local_exclude)}
    return Loaded(cohort, X, names, heart.VOCAB_PATH, info)
