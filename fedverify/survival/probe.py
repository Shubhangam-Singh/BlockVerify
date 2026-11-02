"""The public probe set for screening tree updates (Revision 2 §4B). No patient row is ever used.

Rows are random draws over the FIXED feature schema, built in the encoded feature space the models see, column by
column from the feature names alone (the naming is the encoders' own: `c` numeric, `c_missing` its indicator,
`col=category` one-hot):

  numeric      uniform on [0, 1], i.e. uniform on [0, the fixed scaling constant] in raw units (data.NUMERIC_SCALE,
               heart.NUMERIC_SCALE): the preprocessing's own constant range, not a range read from patients
  missing      where a numeric column has an indicator, it is on with probability MISSING_RATE and the value is 0,
               exactly as the encoder writes a missing value
  categorical  one category per one-hot group, uniform over the vocabulary (the NA category included)

The generator's seed depends only on public things (a label, the dataset and its encoding), so every party can
rebuild the same probe set, and the screen's inputs are reproducible.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .config import derive_seed

MISSING_RATE = 0.125            # a design constant of the probe distribution, not fitted to anything
PROBE_LABEL = "fedverify-probe-set-v1"


def feature_groups(feature_names: Sequence[str]) -> list[tuple[str, list[int], Any]]:
    """[(kind, columns, extra)] in column order: ("numeric", [j], j_missing or None) or ("onehot", [j...], source)."""
    names = list(feature_names)
    index = {n: j for j, n in enumerate(names)}
    groups, seen = [], set()
    for j, n in enumerate(names):
        if j in seen:
            continue
        if "=" in n:
            src = n.split("=", 1)[0]
            cols = [k for k, m in enumerate(names) if m.split("=", 1)[0] == src and "=" in m]
            groups.append(("onehot", cols, src))
            seen.update(cols)
        elif n.endswith("_missing") and n[: -len("_missing")] in index:
            continue                                   # drawn together with its numeric column
        else:
            miss = index.get(f"{n}_missing")
            groups.append(("numeric", [j], miss))
            seen.add(j)
            if miss is not None:
                seen.add(miss)
    covered = set()
    for _, cols, extra in groups:
        covered.update(cols)
        if isinstance(extra, int):
            covered.add(extra)
    if covered != set(range(len(names))):
        raise ValueError(f"feature names do not follow the encoders' naming: {sorted(set(range(len(names))) - covered)}")
    return groups


def draw_rows(feature_names: Sequence[str], n: int, seed: int) -> np.ndarray:
    """n random rows over the schema (float32, encoded space). Depends on the names, n and seed only."""
    rng = np.random.default_rng(int(seed))
    X = np.zeros((int(n), len(feature_names)), dtype=np.float64)
    rows = np.arange(int(n))
    for kind, cols, extra in feature_groups(feature_names):
        if kind == "numeric":
            X[:, cols[0]] = rng.uniform(0.0, 1.0, size=int(n))
            if extra is not None:
                miss = rng.random(int(n)) < MISSING_RATE
                X[miss, cols[0]] = 0.0
                X[:, extra] = miss.astype(np.float64)
        else:
            pick = rng.integers(0, len(cols), size=int(n))
            X[rows, np.asarray(cols)[pick]] = 1.0
    return X.astype(np.float32)


def probe_seed(cfg: Any, feature_names: Sequence[str]) -> int:
    return derive_seed(PROBE_LABEL, cfg.dataset, cfg.heart_mode if cfg.dataset == "heart" else "-",
                       "|".join(feature_names))


def probe_set(cfg: Any, feature_names: Sequence[str]) -> np.ndarray:
    """The run's public probe set: cfg.probe_rows rows, the same for every fold, repeat and party."""
    return draw_rows(feature_names, cfg.probe_rows, probe_seed(cfg, feature_names))


def describe(cfg: Any, feature_names: Sequence[str]) -> dict:
    """Provenance for config.json."""
    X = probe_set(cfg, feature_names)
    return {"rows": int(cfg.probe_rows), "seed": probe_seed(cfg, feature_names), "label": PROBE_LABEL,
            "missing_rate": MISSING_RATE, "sha256": hashlib.sha256(X.tobytes()).hexdigest(),
            "rule": ("public, data-independent: numeric uniform on [0, 1] in the encoded space (= [0, the fixed scale] "
                     "raw), missing indicators on with probability missing_rate (value 0), one category per one-hot "
                     "group uniformly over the vocabulary; no patient row is used")}
