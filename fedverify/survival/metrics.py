"""Survival / classification metrics (prompt §4, Revision 2 §2.5), configured by metrics.yaml.

Two score kinds (survival/common.py):
  risk      P(event): higher = more likely to die / to have disease. lifelines' concordance_index wants HIGHER = LONGER
            survival, so c_index() passes -risk (pinned by test_c_index_sign).
  log_time  XGBoost AFT's predicted log survival time: higher = longer survival, so c_index_time() passes it
            UNCHANGED (pinned by the AFT sign test). Threshold metrics do not apply to it.

Rows whose score is NaN (a hospital a method does not model, e.g. heart's Switzerland under local_*) are left out.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .common import RISK

METRICS_YAML = Path(__file__).with_name("metrics.yaml")
ACCURACY_THRESHOLD = 0.5           # accuracy() default only; the runs use the F1 threshold (Phase 7, see accuracy_at)


def c_index(time: Sequence[float], event: Sequence[int], risk: Sequence[float]) -> Optional[float]:
    """Harrell's C via lifelines for a RISK score (negated); None when no pair is comparable (e.g. no deaths)."""
    from lifelines.utils import concordance_index

    event = np.asarray(event, dtype=np.int64)
    if event.sum() == 0:
        return None
    return float(concordance_index(np.asarray(time, dtype=np.float64), -np.asarray(risk, dtype=np.float64), event))


def c_index_time(time: Sequence[float], event: Sequence[int], predicted_time: Sequence[float]) -> Optional[float]:
    """Harrell's C for a predicted survival TIME (AFT): higher already means longer, so it is passed UNCHANGED."""
    from lifelines.utils import concordance_index

    event = np.asarray(event, dtype=np.int64)
    if event.sum() == 0:
        return None
    return float(concordance_index(np.asarray(time, dtype=np.float64), np.asarray(predicted_time, dtype=np.float64),
                                   event))


def roc_auc(event: Sequence[int], risk: Sequence[float]) -> Optional[float]:
    """ROC-AUC of risk vs the event indicator; None when only one class is present."""
    from sklearn.metrics import roc_auc_score

    event = np.asarray(event, dtype=np.int64)
    if event.size == 0 or event.min() == event.max():
        return None
    return float(roc_auc_score(event, np.asarray(risk, dtype=np.float64)))


def f1_score(event: Sequence[int], pred: Sequence[int]) -> float:
    """Binary F1 of the event class; 0.0 when there is nothing to find and nothing found (no warnings)."""
    e = np.asarray(event, dtype=np.int64)
    p = np.asarray(pred, dtype=np.int64)
    tp = int(((p == 1) & (e == 1)).sum())
    fp = int(((p == 1) & (e == 0)).sum())
    fn = int(((p == 0) & (e == 1)).sum())
    return (2.0 * tp / (2 * tp + fp + fn)) if tp else 0.0


def accuracy(event: Sequence[int], risk: Sequence[float], threshold: float = ACCURACY_THRESHOLD) -> Optional[float]:
    """Share of rows where (P(event) >= threshold) equals the label; None for no rows."""
    e = np.asarray(event, dtype=np.int64)
    r = np.asarray(risk, dtype=np.float64)
    if e.size == 0:
        return None
    return float(np.mean((r >= threshold).astype(np.int64) == e))


def accuracy_at(event: Sequence[int], pred: Sequence[int]) -> Optional[float]:
    """Share of rows whose prediction equals the label, where pred = (score >= the fold's training-F1 threshold).
    Phase 7: accuracy and F1 use ONE threshold rule (the threshold that maximises F1 on the training fold). Rows with
    no prediction (pred < 0) are skipped; None for no rows."""
    e = np.asarray(event, dtype=np.int64)
    p = np.asarray(pred, dtype=np.int64)
    ok = p >= 0
    if not ok.any():
        return None
    return float(np.mean(p[ok] == e[ok]))


def log_loss(event: Sequence[int], prob: Sequence[float]) -> Optional[float]:
    """Binary log-loss with labels=[0, 1], so a one-class hospital or fold does not crash; None for no rows."""
    from sklearn.metrics import log_loss as sk_log_loss

    e = np.asarray(event, dtype=np.int64)
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-15, 1 - 1e-15)
    if e.size == 0:
        return None
    return float(sk_log_loss(e, p, labels=[0, 1]))


def best_f1_threshold(event: Sequence[int], risk: Sequence[float]) -> float:
    """Threshold t maximising F1 of (risk >= t) on these (TRAINING) predictions.

    Candidates are the distinct risk values; ties in F1 go to the HIGHER threshold (fewer positives). With no events
    the answer is +inf (predict nobody), which F1 scores as 0 either way.
    """
    e = np.asarray(event, dtype=np.int64)
    r = np.asarray(risk, dtype=np.float64)
    if e.sum() == 0 or r.size == 0:
        return float("inf")
    order = np.argsort(-r, kind="mergesort")
    rs, es = r[order], e[order]
    tp = np.cumsum(es)                                  # predicting the top j+1 rows as events
    fp = np.cumsum(1 - es)
    last = np.r_[rs[1:] != rs[:-1], True]               # a cut is only possible at the end of a tie block
    tp, fp, cut = tp[last], fp[last], rs[last]
    f1 = 2.0 * tp / (2.0 * tp + fp + (e.sum() - tp))
    best = np.flatnonzero(f1 == f1.max())[0]            # descending thresholds: the first maximum is the highest t
    return float(cut[best])


def thresholds_for_test(train_event: np.ndarray, train_risk: np.ndarray, train_hosp: np.ndarray,
                        test_hosp: np.ndarray, per_hospital: bool) -> np.ndarray:
    """F1 threshold for every TEST row: one pooled threshold, or each hospital's own (local_* methods).
    Training rows with a NaN score (an unmodelled hospital) are ignored; such hospitals get a NaN threshold."""
    ok = np.isfinite(train_risk)
    if not per_hospital:
        return np.full(len(test_hosp), best_f1_threshold(train_event[ok], train_risk[ok]))
    out = np.full(len(test_hosp), np.nan)
    for h in np.unique(test_hosp):
        rows = (train_hosp == h) & ok
        if rows.any():
            out[test_hosp == h] = best_f1_threshold(train_event[rows], train_risk[rows])
    return out


Metric = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str], Optional[float]]
METRICS: dict[str, Metric] = {
    "c_index": lambda time, event, s, pred, kind: c_index(time, event, s) if kind == RISK
    else c_index_time(time, event, s),
    "roc_auc": lambda time, event, s, pred, kind: roc_auc(event, s) if kind == RISK else None,
    "f1": lambda time, event, s, pred, kind: f1_score(event, pred) if kind == RISK else None,
    "accuracy": lambda time, event, s, pred, kind: accuracy_at(event, pred) if kind == RISK else None,
}


def load_metric_config(key: str = "tcga", path: Path = METRICS_YAML) -> dict:
    """Metric names (table column order) and the per-hospital rule for one block of metrics.yaml:
    'tcga' (the top-level block), 'tcga_survival' or 'heart' (under `datasets:`)."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    block = raw if key == "tcga" else (raw.get("datasets") or {}).get(key)
    if block is None:
        raise KeyError(f"metrics.yaml has no block {key!r}")
    names = list(block["metrics"])
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise ValueError(f"metrics.yaml names unknown metrics {unknown}; known: {sorted(METRICS)}")
    ph = block.get("per_hospital", {}) or {}
    min_count = int(ph.get("min_class_count", ph.get("min_deaths", 5)))
    return {"metrics": names, "min_deaths": min_count, "min_class_count": min_count,
            "rank_metrics": list(ph.get("rank_metrics", ["c_index", "roc_auc"])),
            "event_name": str(ph.get("event_name", "deaths"))}


def compute(names: Sequence[str], time: np.ndarray, event: np.ndarray, score: np.ndarray, pred: np.ndarray,
            kind: str = RISK) -> dict[str, Optional[float]]:
    """Metrics over the rows that have a finite score."""
    s = np.asarray(score, dtype=np.float64)
    ok = np.isfinite(s)
    t, e, p = np.asarray(time)[ok], np.asarray(event)[ok], np.asarray(pred)[ok]
    return {n: METRICS[n](t, e, s[ok], p, kind) for n in names}


def per_hospital(names: Sequence[str], hospital: np.ndarray, time: np.ndarray, event: np.ndarray,
                 risk: np.ndarray, pred: np.ndarray, min_class_count: int, rank_metrics: Sequence[str],
                 kind: str = RISK, event_name: str = "deaths") -> dict[str, dict]:
    """The same pooled predictions grouped by hospital. Rank metrics are None when the rarer class has fewer than
    min_class_count rows (TCGA: deaths < 5, so A8 shows an em dash; prompt §4). A hospital with no score at all
    (not modelled by the method) is marked excluded."""
    out = {}
    score = np.asarray(risk, dtype=np.float64)
    for h in dict.fromkeys(np.asarray(hospital).tolist()):
        rows = np.asarray(hospital) == h
        ev = np.asarray(event)[rows]
        events = int(ev.sum())
        if not np.isfinite(score[rows]).any():
            out[h] = {"patients": int(rows.sum()), event_name: events, "excluded": True, **{n: None for n in names}}
            continue
        vals = compute(names, np.asarray(time)[rows], ev, score[rows], np.asarray(pred)[rows], kind)
        if min(events, int(rows.sum()) - events) < min_class_count:
            vals = {k: (None if k in rank_metrics else v) for k, v in vals.items()}
        out[h] = {"patients": int(rows.sum()), event_name: events, **vals}
    return out


def hospital_round_metrics(hospitals: Sequence[str], train_hosp: np.ndarray, train_event: np.ndarray,
                           train_score: np.ndarray, test_hosp: np.ndarray, test_event: np.ndarray,
                           test_score: np.ndarray, kind: str = RISK, train_thr: Optional[np.ndarray] = None,
                           test_thr: Optional[np.ndarray] = None) -> dict[str, dict]:
    """Per hospital, this round: train/test accuracy and log-loss — the data behind Figs 5 and 6 (Revision 2 §2.4).
    Accuracy uses the given per-row thresholds (the runs pass the fold's training-F1 threshold, the same rule as F1);
    without them it falls back to P >= 0.5. None where it does not apply (an AFT time score, or an unmodelled hospital)."""
    out = {}
    for h in hospitals:
        vals = {}
        for part, hosp, ev, sc, thr in (("train", train_hosp, train_event, train_score, train_thr),
                                        ("test", test_hosp, test_event, test_score, test_thr)):
            rows = (np.asarray(hosp) == h) & np.isfinite(np.asarray(sc, dtype=np.float64))
            ok = kind == RISK and rows.any()
            if ok and thr is not None:
                t = np.asarray(thr, dtype=np.float64)[rows]
                with np.errstate(invalid="ignore"):
                    pr = (np.asarray(sc, dtype=np.float64)[rows] >= t).astype(np.int64)
                vals[f"{part}_acc"] = accuracy_at(np.asarray(ev)[rows], np.where(np.isfinite(t), pr, -1))
            else:
                vals[f"{part}_acc"] = accuracy(np.asarray(ev)[rows], np.asarray(sc)[rows]) if ok else None
            vals[f"{part}_logloss"] = log_loss(np.asarray(ev)[rows], np.asarray(sc)[rows]) if ok else None
        out[h] = {k: vals[k] for k in ("train_acc", "train_logloss", "test_acc", "test_logloss")}
    return out
