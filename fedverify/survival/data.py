"""TCGA-BRCA cohort: loader, leakage guard, data-independent preprocessing and repeated CV folds (prompt §3-§4).

Cohort    hospital = 2nd token of submitter_id; event = survival_event; time (days) = survival * 365.25 for deaths
          (survival is in YEARS and only set for deaths), else days_to_last_follow_up. Rows with no usable time are
          dropped and counted ('[]' is how this workbook spells a missing follow-up).
Features  ONLY diag_age, lymph_node, size_over_20mm, menop_status, er/pr/her2_status_by_ihc, histological_type
          (+ race behind --include-race). The leakage guard raises if any outcome column could reach the model.
Encoding  data-independent, so no hospital leaks statistics to another: numerics divided by fixed constants, missing
          numeric -> 0 plus a <col>_missing indicator, categoricals one-hot over the FIXED vocabulary in vocab.json
          (category names only, computed once from the full cohort). Nothing is fitted on data.
Folds     repeated stratified K-fold on (hospital, event); repeat r uses seed base + r (D7).

    python3 -m fedverify.survival.data                 # cohort summary
    python3 -m fedverify.survival.data --write-vocab   # regenerate vocab.json from the workbook
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import HOSPITALS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "fedverify" / "data" / "tcga_brca" / "clinical.xlsx"
VOCAB_PATH = Path(__file__).resolve().parent / "vocab.json"

FEATURES_NUMERIC: tuple[str, ...] = ("diag_age", "lymph_node", "size_over_20mm")
NUMERIC_SCALE: dict[str, float] = {"diag_age": 100.0, "lymph_node": 10.0, "size_over_20mm": 1.0}
FEATURES_CATEGORICAL: tuple[str, ...] = ("menop_status", "er_status_by_ihc", "pr_status_by_ihc",
                                         "her2_status_by_ihc", "histological_type")
RACE = "race"
NA = "NA"
NA_TOKENS = frozenset({"[Not Available]", "[Not Evaluated]", ""})   # plus NaN / None, see normalize_category

# LEAKAGE GUARD (hard requirement, §3): these must never reach the feature matrix.
LEAK_PREFIXES: tuple[str, ...] = ("OS_", "DFS_", "PFS_")
LEAK_COLUMNS = frozenset({"survival", "survival_event", "days_to_last_follow_up", "diag_date",
                          "year_of_birth", "submitter_id", "case_id"})

DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = 30.4375


class LeakageError(ValueError):
    """An outcome or identifier column would have reached the model."""


def assert_no_leakage(columns: Iterable[str]) -> None:
    """Raise LeakageError if any column is an outcome / identifier column (§3 deny-list)."""
    bad = sorted({c for c in columns if str(c).startswith(LEAK_PREFIXES) or str(c) in LEAK_COLUMNS})
    if bad:
        raise LeakageError(f"leakage guard: {bad} must never reach the model (outcome or identifier columns)")


def feature_columns(include_race: bool = False) -> tuple[str, ...]:
    """The whitelist. Only these source columns can reach the model."""
    cols = FEATURES_NUMERIC + FEATURES_CATEGORICAL + ((RACE,) if include_race else ())
    assert_no_leakage(cols)
    return cols


def source_column(feature_name: str) -> str:
    """'er_status_by_ihc=Positive' -> 'er_status_by_ihc'; 'lymph_node_missing' -> 'lymph_node'."""
    base = feature_name.split("=", 1)[0]
    return base[: -len("_missing")] if base.endswith("_missing") else base


# ── reading ──────────────────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def _read_cached(path: str, mtime_ns: int, size: int) -> pd.DataFrame:  # noqa: ARG001 (mtime/size key the cache)
    # dtype=object and no NA parsing: the raw tokens ('[]', '[Not Available]', ...) stay visible.
    return pd.read_excel(path, sheet_name=0, engine="openpyxl", dtype=object, keep_default_na=False, na_values=[])


def read_workbook(path: str | os.PathLike = DEFAULT_DATA) -> pd.DataFrame:
    """The raw workbook (all rows, all columns). Cached per (path, mtime, size); a copy is returned."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found: the TCGA-BRCA workbook must be at {DEFAULT_DATA} (gitignored)")
    st = p.stat()
    return _read_cached(str(p.resolve()), st.st_mtime_ns, st.st_size).copy()


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_float(values: Iterable[Any]) -> np.ndarray:
    """Cells -> float64; blank, '[]' and any other non-numeric token become NaN."""
    out = []
    for v in values:
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out.append(float(v))
            continue
        try:
            out.append(float(str(v).strip()))
        except ValueError:
            out.append(np.nan)
    a = np.asarray(out, dtype=np.float64)
    a[~np.isfinite(a)] = np.nan
    return a


def normalize_category(v: Any) -> str:
    """[Not Available] / [Not Evaluated] / blank / NaN / None -> 'NA'; anything else is kept verbatim (stripped)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return NA
    s = str(v).strip()
    return NA if s in NA_TOKENS or s.lower() == "nan" else s


def hospital_of(submitter_id: str) -> str:
    parts = str(submitter_id).strip().split("-")
    if len(parts) != 3 or parts[0] != "TCGA" or not parts[1]:
        raise ValueError(f"malformed submitter_id {submitter_id!r}; expected TCGA-<site>-<id>")
    return parts[1]


# ── cohort ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Cohort:
    """One dataset's analysed rows, in file order. TCGA: the five-hospital cohort after dropping rows with no usable
    time. Heart (survival/heart.py): label = num > 0 and no times (time_days is NaN)."""
    patient_id: np.ndarray          # submitter_id (TCGA) / row id (heart)
    hospital: np.ndarray            # site code / hospital name
    event: np.ndarray               # 1 = death (TCGA) / disease (heart), else 0
    time_days: np.ndarray
    features: pd.DataFrame          # ONLY the whitelisted source columns, raw tokens
    dropped: int                    # rows of the selected hospitals dropped (no usable time / incomplete)
    source_sha256: str
    site_order: tuple[str, ...] = HOSPITALS      # hospital order for reports

    @property
    def n(self) -> int:
        return int(len(self.patient_id))

    @property
    def n_deaths(self) -> int:
        return int(self.event.sum())

    def counts(self) -> dict[str, dict[str, int]]:
        present = [h for h in self.site_order if (self.hospital == h).any()]
        return {h: {"patients": int((self.hospital == h).sum()),
                    "deaths": int(self.event[self.hospital == h].sum())} for h in present}


def load_cohort(path: str | os.PathLike = DEFAULT_DATA, hospitals: Sequence[str] = HOSPITALS,
                include_race: bool = False, verbose: bool = False) -> Cohort:
    raw = read_workbook(path)
    need = {"submitter_id", "survival_event", "survival", "days_to_last_follow_up", *feature_columns(include_race)}
    missing = sorted(need - set(raw.columns))
    if missing:
        raise KeyError(f"workbook lacks required columns {missing}")

    site = np.array([hospital_of(s) for s in raw["submitter_id"]])
    keep = np.isin(site, list(hospitals))
    sub = raw.loc[keep].reset_index(drop=True)
    site = site[keep]

    event = to_float(sub["survival_event"])
    if not np.isin(event, (0.0, 1.0)).all():
        raise ValueError("survival_event must be 0/1 for every selected patient")
    surv_years = to_float(sub["survival"])
    follow_up = to_float(sub["days_to_last_follow_up"])
    time_days = np.where(event == 1, surv_years * DAYS_PER_YEAR, follow_up)
    usable = np.isfinite(time_days)
    dropped = int((~usable).sum())
    if verbose:
        print(f"[survival.data] dropped {dropped} row(s) with no usable time "
              f"({int(usable.sum())} of {len(usable)} kept)", flush=True)

    feats = sub.loc[usable, list(feature_columns(include_race))].reset_index(drop=True)
    assert_no_leakage(feats.columns)
    return Cohort(
        patient_id=sub.loc[usable, "submitter_id"].astype(str).str.strip().to_numpy(),
        hospital=site[usable], event=event[usable].astype(np.int64), time_days=time_days[usable],
        features=feats, dropped=dropped, source_sha256=sha256_file(path))


def unit_check(raw: pd.DataFrame, tolerance_months: float = 0.5) -> dict:
    """Sanity check of the time units, never a feature: |time_days / 30.4375 - OS_MONTHS| < tolerance.

    Runs on every workbook row that has both a usable time and OS_MONTHS. Returns the counts and the failing ids.
    """
    event = to_float(raw["survival_event"])
    time_days = np.where(event == 1, to_float(raw["survival"]) * DAYS_PER_YEAR, to_float(raw["days_to_last_follow_up"]))
    os_months = to_float(raw["OS_MONTHS"])
    chk = np.isfinite(time_days) & np.isfinite(os_months)
    diff = np.abs(time_days / DAYS_PER_MONTH - os_months)
    fail = chk & (diff >= tolerance_months)
    ids = raw["submitter_id"].astype(str).str.strip().to_numpy()
    return {"checked": int(chk.sum()), "failed": int(fail.sum()), "failing_ids": sorted(ids[fail].tolist())}


# ── vocabulary and encoding ──────────────────────────────────────────────────
def build_vocab(raw: pd.DataFrame) -> dict:
    """Category NAMES per categorical column (race included, for --include-race), from the full cohort."""
    cats = {}
    for col in FEATURES_CATEGORICAL + (RACE,):
        seen = {normalize_category(v) for v in raw[col]}
        cats[col] = sorted(seen - {NA}) + [NA]          # 'NA' always present and always last
    return cats


def write_vocab(path: str | os.PathLike = VOCAB_PATH, data: str | os.PathLike = DEFAULT_DATA) -> dict:
    raw = read_workbook(data)
    payload = {
        "_about": ("Fixed one-hot vocabulary: category NAMES only (public metadata, no counts), computed once from "
                   "the full TCGA-BRCA cohort. Regenerate with: python3 -m fedverify.survival.data --write-vocab"),
        "source_sha256": sha256_file(data),
        "source_rows": int(len(raw)),
        "na_category": NA,
        "categorical": build_vocab(raw),
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def load_vocab(path: str | os.PathLike = VOCAB_PATH) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["categorical"]


def encode_features(features: pd.DataFrame, vocab: dict, include_race: bool = False) -> tuple[np.ndarray, list[str]]:
    """Data-independent encoding -> (X float64 [n, d], feature names). Row i depends on row i only."""
    assert_no_leakage(features.columns)
    cols = feature_columns(include_race)
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise KeyError(f"features lack {missing}")

    n = len(features)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for c in FEATURES_NUMERIC:
        v = to_float(features[c])
        miss = ~np.isfinite(v)
        blocks.append(np.where(miss, 0.0, v / NUMERIC_SCALE[c]))
        names.append(c)
        blocks.append(miss.astype(np.float64))
        names.append(f"{c}_missing")
    for c in cols:
        if c in FEATURES_NUMERIC:
            continue
        cats = list(vocab[c])
        index = {k: i for i, k in enumerate(cats)}
        vals = [normalize_category(v) for v in features[c]]
        unseen = sorted(set(vals) - set(index))
        if unseen:
            raise ValueError(f"{c}: categories {unseen} are not in vocab.json; regenerate it with --write-vocab")
        onehot = np.zeros((n, len(cats)), dtype=np.float64)
        onehot[np.arange(n), [index[v] for v in vals]] = 1.0
        blocks.append(onehot)
        names.extend(f"{c}={k}" for k in cats)

    X = np.column_stack(blocks) if n else np.zeros((0, len(names)))
    assert_no_leakage(source_column(nm) for nm in names)       # second line of defence, on the matrix itself
    return X, names


# ── cross-validation ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FoldData:
    """One CV fold: every hospital's fold-k patients are the test set, the rest is training (§4)."""
    train_idx: np.ndarray
    test_idx: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    event_train: np.ndarray
    event_test: np.ndarray
    time_train: np.ndarray
    time_test: np.ndarray
    hosp_train: np.ndarray
    hosp_test: np.ndarray


def repeat_seeds(base_seed: int, repeats: int) -> list[int]:
    """D7: repeat r uses seed base + r, so base 0 with 5 repeats is seeds 0-4."""
    return [int(base_seed) + r for r in range(int(repeats))]


def make_folds(cohort: Cohort, seed: int, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold on (hospital, event). Returns sorted (train_idx, test_idx) pairs."""
    from sklearn.model_selection import StratifiedKFold

    strata = np.array([f"{h}_{int(e)}" for h, e in zip(cohort.hospital, cohort.event)])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    with warnings.catch_warnings():
        # A8 has only 4 deaths, so its death stratum is smaller than n_splits and sklearn says so. Expected and
        # documented (INVENTORY.md §4): one fold simply has no A8 death in its test part.
        warnings.filterwarnings("ignore", message="The least populated class in y has only", category=UserWarning)
        splits = list(skf.split(np.zeros(len(strata)), strata))
    return [(np.sort(tr), np.sort(te)) for tr, te in splits]


def fold_data(cohort: Cohort, X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> FoldData:
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("a patient is in both the training and the test part of a fold")
    return FoldData(train_idx=train_idx, test_idx=test_idx, X_train=X[train_idx], X_test=X[test_idx],
                    event_train=cohort.event[train_idx], event_test=cohort.event[test_idx],
                    time_train=cohort.time_days[train_idx], time_test=cohort.time_days[test_idx],
                    hosp_train=cohort.hospital[train_idx], hosp_test=cohort.hospital[test_idx])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--write-vocab", action="store_true", help="regenerate vocab.json from the full cohort")
    ap.add_argument("--include-race", action="store_true")
    a = ap.parse_args(argv)
    if a.write_vocab:
        payload = write_vocab(VOCAB_PATH, a.data)
        print(f"wrote {VOCAB_PATH} ({sum(len(v) for v in payload['categorical'].values())} category names)")
    c = load_cohort(a.data, include_race=a.include_race, verbose=True)
    X, names = encode_features(c.features, load_vocab(), a.include_race)
    print(f"cohort: {c.n} patients, {c.n_deaths} deaths; per hospital {c.counts()}; X {X.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
