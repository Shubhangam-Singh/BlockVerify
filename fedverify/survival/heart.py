"""UCI heart disease, four hospitals (Revision 2 §2.1): loader, leakage guard, data-independent encoding.

Hospital = the `dataset` column; label = num > 0. `id`, `num` and `dataset` never reach the model.

  paper  (default for every table) drop ca, thal, slope, THEN drop incomplete rows: 740 rows (Cleveland 304,
         Hungary 260, VA Long Beach 130, Switzerland 46), which is how the reference heart paper (arXiv 2408.06183)
         got 740. Never dropna() over all columns: that keeps 299 rows, 297 of them from Cleveland.
  full   all 920 rows and all features; missing numeric -> 0 plus <col>_missing, missing categorical -> NA.
         Used only for one sensitivity table.

Encoding is data-independent, as for TCGA: numerics divided by fixed constants, categoricals one-hot over the FIXED
vocabulary in heart_vocab.json (category names only, from all 920 rows). Nothing is fitted on data.

    python3 -m fedverify.survival.heart                  # cohort summary (paper mode)
    python3 -m fedverify.survival.heart --write-vocab    # regenerate heart_vocab.json
"""
from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import HEART_HOSPITALS
from .data import NA, REPO_ROOT, Cohort, LeakageError, sha256_file, to_float

DEFAULT_DATA = REPO_ROOT / "fedverify" / "data" / "heart" / "heart_disease_uci.csv"
VOCAB_PATH = Path(__file__).resolve().parent / "heart_vocab.json"

LABEL, HOSPITAL_COL, ID_COL = "num", "dataset", "id"
LEAK_COLUMNS = frozenset({LABEL, HOSPITAL_COL, ID_COL})     # label, hospital identity, row id
PAPER_DROPPED: tuple[str, ...] = ("ca", "thal", "slope")
MODES: tuple[str, ...] = ("paper", "full")

NUMERIC: tuple[str, ...] = ("age", "trestbps", "chol", "thalch", "oldpeak", "ca")
# fixed physiological / instrument ranges, never fitted: age y, resting BP mmHg, cholesterol mg/dl, max HR bpm,
# ST depression mm, major vessels (0-3)
NUMERIC_SCALE: dict[str, float] = {"age": 100.0, "trestbps": 200.0, "chol": 600.0, "thalch": 220.0,
                                   "oldpeak": 10.0, "ca": 3.0}
CATEGORICAL: tuple[str, ...] = ("sex", "cp", "fbs", "restecg", "exang", "slope", "thal")


def assert_no_leakage(columns: Iterable[str]) -> None:
    """Raise LeakageError if the label (num), the hospital (dataset) or the row id could reach the model."""
    bad = sorted({str(c) for c in columns} & LEAK_COLUMNS)
    if bad:
        raise LeakageError(f"leakage guard: {bad} must never reach the model (label, hospital or row id)")


def feature_columns(mode: str = "paper") -> tuple[str, ...]:
    if mode not in MODES:
        raise ValueError(f"heart mode must be one of {MODES}")
    drop = PAPER_DROPPED if mode == "paper" else ()
    cols = tuple(c for c in NUMERIC + CATEGORICAL if c not in drop)
    assert_no_leakage(cols)
    return cols


def normalize_category(v: Any) -> str:
    """NaN / None / blank -> 'NA'; booleans -> 'True' / 'False'; anything else verbatim (stripped)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return NA
    s = str(v).strip()
    return NA if s == "" or s.lower() == "nan" else s


@lru_cache(maxsize=4)
def _read_cached(path: str, mtime_ns: int, size: int) -> pd.DataFrame:  # noqa: ARG001 (mtime/size key the cache)
    return pd.read_csv(path)


def read_csv(path: str | os.PathLike = DEFAULT_DATA) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found: the heart CSV must be at {DEFAULT_DATA} (gitignored)")
    st = p.stat()
    return _read_cached(str(p.resolve()), st.st_mtime_ns, st.st_size).copy()


def select_rows(raw: pd.DataFrame, mode: str = "paper") -> pd.DataFrame:
    """paper: drop ca/thal/slope, then incomplete rows. full: every row."""
    if mode not in MODES:
        raise ValueError(f"heart mode must be one of {MODES}")
    if mode == "full":
        return raw.reset_index(drop=True)
    return raw.drop(columns=list(PAPER_DROPPED)).dropna().reset_index(drop=True)


def load_cohort(path: str | os.PathLike = DEFAULT_DATA, mode: str = "paper",
                hospitals: Sequence[str] = HEART_HOSPITALS, verbose: bool = False) -> Cohort:
    raw = read_csv(path)
    missing = sorted({ID_COL, LABEL, HOSPITAL_COL, *feature_columns(mode)} - set(raw.columns))
    if missing:
        raise KeyError(f"heart CSV lacks required columns {missing}")
    rows = select_rows(raw, mode)
    rows = rows[rows[HOSPITAL_COL].isin(list(hospitals))].reset_index(drop=True)
    total = int(raw[HOSPITAL_COL].isin(list(hospitals)).sum())
    if verbose:
        print(f"[survival.heart] mode={mode}: {len(rows)} of {total} rows kept "
              f"({total - len(rows)} dropped as incomplete)", flush=True)
    feats = rows[list(feature_columns(mode))].reset_index(drop=True)
    assert_no_leakage(feats.columns)
    return Cohort(patient_id=rows[ID_COL].astype(int).astype(str).to_numpy(),
                  hospital=rows[HOSPITAL_COL].astype(str).to_numpy(),
                  event=(to_float(rows[LABEL]) > 0).astype(np.int64),
                  time_days=np.full(len(rows), np.nan),
                  features=feats, dropped=total - len(rows), source_sha256=sha256_file(path),
                  site_order=tuple(HEART_HOSPITALS))


def build_vocab(raw: pd.DataFrame) -> dict:
    """Category NAMES per categorical column, from all rows; 'NA' always present and last."""
    return {c: sorted({normalize_category(v) for v in raw[c]} - {NA}) + [NA] for c in CATEGORICAL}


def write_vocab(path: str | os.PathLike = VOCAB_PATH, data: str | os.PathLike = DEFAULT_DATA) -> dict:
    raw = read_csv(data)
    payload = {
        "_about": ("Fixed one-hot vocabulary for the UCI heart data: category NAMES only (public metadata, no "
                   "counts), from all 920 rows. Regenerate with: python3 -m fedverify.survival.heart --write-vocab"),
        "source_sha256": sha256_file(data), "source_rows": int(len(raw)), "na_category": NA,
        "categorical": build_vocab(raw),
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def load_vocab(path: str | os.PathLike = VOCAB_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["categorical"]


def encode_features(features: pd.DataFrame, vocab: dict, mode: str = "paper") -> tuple[np.ndarray, list[str]]:
    """Data-independent encoding -> (X float64 [n, d], names). Row i depends on row i only."""
    assert_no_leakage(features.columns)
    cols = feature_columns(mode)
    absent = [c for c in cols if c not in features.columns]
    if absent:
        raise KeyError(f"features lack {absent}")
    n = len(features)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for c in (c for c in cols if c in NUMERIC):
        v = to_float(features[c])
        miss = ~np.isfinite(v)
        if mode == "paper" and miss.any():
            raise ValueError(f"paper mode must have no missing values, but {c} has {int(miss.sum())}")
        blocks.append(np.where(miss, 0.0, v / NUMERIC_SCALE[c]))
        names.append(c)
        if mode == "full":
            blocks.append(miss.astype(np.float64))
            names.append(f"{c}_missing")
    for c in (c for c in cols if c in CATEGORICAL):
        cats = list(vocab[c])
        index = {k: i for i, k in enumerate(cats)}
        vals = [normalize_category(v) for v in features[c]]
        unseen = sorted(set(vals) - set(index))
        if unseen:
            raise ValueError(f"{c}: categories {unseen} are not in heart_vocab.json; regenerate it")
        onehot = np.zeros((n, len(cats)), dtype=np.float64)
        onehot[np.arange(n), [index[v] for v in vals]] = 1.0
        blocks.append(onehot)
        names.extend(f"{c}={k}" for k in cats)
    X = np.column_stack(blocks) if n else np.zeros((0, len(names)))
    assert_no_leakage(n.split("=", 1)[0].removesuffix("_missing") for n in names)
    return X, names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--mode", default="paper", choices=list(MODES))
    ap.add_argument("--write-vocab", action="store_true")
    a = ap.parse_args(argv)
    if a.write_vocab:
        payload = write_vocab(VOCAB_PATH, a.data)
        print(f"wrote {VOCAB_PATH} ({sum(len(v) for v in payload['categorical'].values())} category names)")
    c = load_cohort(a.data, a.mode, verbose=True)
    X, names = encode_features(c.features, load_vocab(), a.mode)
    per = {h: {"rows": v["patients"], "positives": v["deaths"]} for h, v in c.counts().items()}
    print(f"heart ({a.mode}): {c.n} rows, {c.n_deaths} positives; per hospital {per}; X {X.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
