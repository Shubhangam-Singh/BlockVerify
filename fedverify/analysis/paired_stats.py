"""Within-hospital metrics and paired comparisons with a patient-level bootstrap (Phase 11).

Everything is computed from the runs' out-of-fold predictions (<results>/<exp>/<cell>/seed0/repeat<r>/oof.jsonl),
as pair sums, so that a bootstrap resample is just a weighted sum:

  ROC-AUC   pairs (i, j) with event_i = 1 and event_j = 0; concordant when risk_i > risk_j; a tie counts 1/2
  C-index   pairs (i, j) with event_i = 1 and (t_i < t_j, or t_i = t_j with j censored), lifelines' admissible pairs;
            concordant when i's predicted survival is the shorter one; a tie counts 1/2
  accuracy, F1  per patient, from the run's own thresholded prediction

With w_i = how often patient i is drawn, metric(w) = w'(C*P)w / w'Pw (pair metrics) or a weighted mean (accuracy,
F1). With every w_i = 1 it is exactly the pooled metric the runs report (tested against lifelines and sklearn).

Within hospital: only pairs whose two patients are from the same hospital, over the hospitals with at least
`min_count` patients of their rarer class (table.yaml within_hospital; the same rule as the per-hospital tables).
That equals the per-hospital metrics combined, weighted by their numbers of comparable pairs.

Paired comparison of runs A and B (same folds, same repeats): per repeat d_r = metric(A_r) - metric(B_r), reported as
mean and sample std over the repeats. Bootstrap: `resamples` draws of the patients, stratified by hospital (each
hospital's patients drawn with replacement, its size kept), the same draws for every comparison of a dataset, from a
fixed seed; statistic = mean over repeats of metric(A_r; w) - metric(B_r; w); the CI is its 2.5th and 97.5th
percentiles. No p-value is computed: with 5 repeats an exact test cannot go below 0.0625.

Two virtual runs exist besides the cells: '@site_rate' (the "Site base rate only" reference row) and '@all_alive'
("Predict all alive"), both on the out-of-fold patients of the dataset's reference cell.

Results are cached in <results>/statistics/paired_bootstrap.json, keyed by the comparison and checked against a
SHA-256 of the out-of-fold files they came from, so a re-run invalidates them.
"""
from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

PAIR_METRICS = ("roc_auc", "c_index", "roc_auc_within", "c_index_within")
POINT_METRICS = ("accuracy", "f1")
WITHIN = {"roc_auc_within": "roc_auc", "c_index_within": "c_index"}
VIRTUAL = ("@site_rate", "@all_alive")


@dataclass
class Preds:
    patients: np.ndarray
    hospital: np.ndarray
    event: np.ndarray
    time: np.ndarray
    risk: Optional[np.ndarray]          # higher = more likely the event (risk runs); None for survival-time runs
    surv: np.ndarray                    # higher = longer predicted survival (what the C-index ranks)
    pred: np.ndarray                    # the run's thresholded prediction; -1 where it has none
    valid: np.ndarray                   # the patient has a score (local runs leave unmodelled hospitals out)
    fold: np.ndarray


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def oof_path(res: Any, ds: str, cell: str, repeat: int) -> Path:
    base = res.site_rate_cell(ds) or res.baseline_cell(ds) or "fedavg_lr" if cell in VIRTUAL else cell
    return res.run_dir(ds, base) / f"repeat{repeat}" / "oof.jsonl"


def load_preds(res: Any, ds: str, cell: str, repeat: int) -> Optional[Preds]:
    """One repeat's pooled out-of-fold predictions of a run (or of a virtual reference run)."""
    path = oof_path(res, ds, cell, repeat)
    if not path.exists():
        res.note_missing(path, "paired_stats")
        return None
    rows = _rows(path)
    hospital = np.array([r["hospital"] for r in rows])
    event = np.array([r["event"] for r in rows], dtype=np.int64)
    time = np.array([np.nan if r.get("time_days") is None else r["time_days"] for r in rows], dtype=np.float64)
    fold = np.array([r.get("fold", -1) for r in rows], dtype=np.int64)
    patients = np.array([str(r["patient"]) for r in rows])
    if cell == "@site_rate":
        from .make_fl_healthcare_tables import site_rate_oof
        risk, pred = site_rate_oof(hospital, event, fold)
    elif cell == "@all_alive":
        risk, pred = np.zeros(len(rows)), np.zeros(len(rows), dtype=np.int64)
    elif "risk" in rows[0]:
        risk = np.array([np.nan if r["risk"] is None else r["risk"] for r in rows], dtype=np.float64)
        pred = np.array([-1 if r.get("pred") is None else r["pred"] for r in rows], dtype=np.int64)
    else:                                                  # a survival-time (AFT) run: no risk, no prediction
        t = np.array([np.nan if r["pred_log_time"] is None else r["pred_log_time"] for r in rows], dtype=np.float64)
        return Preds(patients, hospital, event, time, None, t, np.full(len(rows), -1, dtype=np.int64),
                     np.isfinite(t), fold)
    return Preds(patients, hospital, event, time, risk, -risk, pred, np.isfinite(risk), fold)


def included_hospitals(res: Any, ds: str, spec: dict) -> tuple[list[str], dict]:
    """Hospitals that count within hospital, and why each excluded one is out ({hospital: reason})."""
    rule = spec.get("within_hospital") or {}
    k = int(rule.get("min_count", 5))
    cfg = res.config(ds, res.site_rate_cell(ds) or "fedavg_lr", "within_hospital") or {}
    data = cfg.get("data") or {}
    word = "deaths" if ds == "tcga" else "positives"
    keep, out = [], {}
    for h, v in (data.get("per_hospital") or {}).items():
        ev = int(v.get("positives", v.get("deaths", 0)))
        n = int(v["patients"])
        rarer = min(ev, n - ev)
        if rarer < k:
            out[h] = (f"{ev} {word}" if ev <= n - ev else f"{n - ev} negative{'' if n - ev == 1 else 's'}")
        else:
            keep.append(h)
    return keep, out


# ── pair matrices ─────────────────────────────────────────────────────────────
def restriction(p: Preds, scope: str, keep: Optional[list] = None, hospital: Optional[str] = None) -> np.ndarray:
    """Which (i, j) pairs a metric may use: all, same hospital among `keep`, or both in `hospital`."""
    if scope == "pooled":
        return np.ones((len(p.event), len(p.event)), dtype=bool)
    if scope == "within":
        inside = np.isin(p.hospital, keep or [])
        return (p.hospital[:, None] == p.hospital[None, :]) & inside[:, None] & inside[None, :]
    if scope == "hospital":
        m = p.hospital == hospital
        return m[:, None] & m[None, :]
    raise ValueError(scope)


def pair_terms(p: Preds, metric: str, allowed: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """(C*P, P) for a pair metric, as float64 n x n matrices; None if the run has no score for it."""
    base = WITHIN.get(metric, metric)
    v = p.valid
    if base == "roc_auc":
        if p.risk is None:
            return None
        r = np.where(v, p.risk, 0.0)
        P = (p.event[:, None] == 1) & (p.event[None, :] == 0)
        C = (r[:, None] > r[None, :]) + 0.5 * (r[:, None] == r[None, :])
    else:
        t, e = p.time, p.event.astype(bool)
        if not np.isfinite(t).all():
            return None
        P = e[:, None] & ((t[:, None] < t[None, :]) | ((t[:, None] == t[None, :]) & ~e[None, :]))
        u = np.where(v, p.surv, 0.0)
        C = (u[:, None] < u[None, :]) + 0.5 * (u[:, None] == u[None, :])
    P = P & v[:, None] & v[None, :] & allowed
    Pf = P.astype(np.float64)
    return C * Pf, Pf


def point_terms(p: Preds, metric: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Per-patient terms of accuracy (num, den) or F1 (tp, fp+fn parts), over `rows` with a prediction."""
    ok = rows & (p.pred >= 0) & p.valid                # unmodelled patients (no score) are not evaluated
    if metric == "accuracy":
        return ((p.pred == p.event) & ok).astype(np.float64), ok.astype(np.float64), None
    tp = ((p.pred == 1) & (p.event == 1) & ok).astype(np.float64)
    fp = ((p.pred == 1) & (p.event == 0) & ok).astype(np.float64)
    fn = ((p.pred == 0) & (p.event == 1) & ok).astype(np.float64)
    return tp, fp, fn


def values(p: Preds, metric: str, W: np.ndarray, scope: str = "pooled", keep: Optional[list] = None,
           hospital: Optional[str] = None) -> Optional[np.ndarray]:
    """metric(w) for every row of W (resamples x patients); NaN where no pair (or patient) is left."""
    if metric in PAIR_METRICS:
        if metric in WITHIN:
            scope = "within"
        terms = pair_terms(p, metric, restriction(p, scope, keep, hospital))
        if terms is None:
            return None
        CP, P = terms
        num = np.einsum("bi,bi->b", W @ CP, W)
        den = np.einsum("bi,bi->b", W @ P, W)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    if metric in POINT_METRICS:
        rows = np.ones(len(p.event), dtype=bool) if scope != "hospital" else (p.hospital == hospital)
        a, b, c = point_terms(p, metric, rows)
        with np.errstate(invalid="ignore", divide="ignore"):
            if metric == "accuracy":
                den = W @ b
                return np.where(den > 0, (W @ a) / np.where(den > 0, den, 1.0), np.nan)
            tp, fp, fn = W @ a, W @ b, W @ c
            den = 2 * tp + fp + fn
            return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1.0), 0.0)
    raise ValueError(f"no paired statistic for {metric!r}")


def value(p: Preds, metric: str, **kw: Any) -> Optional[float]:
    """The metric itself (every patient once)."""
    out = values(p, metric, np.ones((1, len(p.event))), **kw)
    return None if out is None or not np.isfinite(out[0]) else float(out[0])


# ── the bootstrap ─────────────────────────────────────────────────────────────
def stratified_weights(hospital: np.ndarray, resamples: int, seed: int, dataset: str) -> np.ndarray:
    """resamples x patients draw counts: each hospital's patients drawn with replacement, its size kept."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(dataset.encode("utf-8"))]))
    W = np.zeros((int(resamples), len(hospital)), dtype=np.float64)
    for h in sorted(set(hospital.tolist())):
        idx = np.flatnonzero(hospital == h)
        draws = rng.integers(0, len(idx), size=(int(resamples), len(idx)))
        counts = np.zeros((int(resamples), len(idx)), dtype=np.float64)
        np.add.at(counts, (np.repeat(np.arange(int(resamples)), len(idx)), draws.ravel()), 1.0)
        W[:, idx] = counts
    return W


def fingerprint(res: Any, ds: str, cells: tuple, repeats: int) -> str:
    h = hashlib.sha256()
    for c in cells:
        for r in range(repeats):
            p = oof_path(res, ds, c, r)
            h.update(p.read_bytes() if p.exists() else b"missing")
    return h.hexdigest()


class PairedStats:
    """Paired differences with bootstrap CIs, cached per comparison in <results>/statistics/paired_bootstrap.json."""

    def __init__(self, res: Any, spec: dict):
        self.res, self.spec = res, spec
        cfg = spec.get("paired_bootstrap") or {}
        self.resamples = int(cfg.get("resamples", 2000))
        self.seed = int(cfg.get("seed", 0))
        self.level = float(cfg.get("level", 0.95))
        self.path = Path(res.root) / "statistics" / "paired_bootstrap.json"
        self.cache = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        if self.cache.get("_settings") != self.settings():
            self.cache = {}
        self.cache["_settings"] = self.settings()
        self.dirty = False
        self._W: dict[str, np.ndarray] = {}
        self._keep: dict[str, list] = {}

    def settings(self) -> dict:
        return {"resamples": self.resamples, "seed": self.seed, "level": self.level, "stratified_by": "hospital",
                "statistic": "mean over repeats of metric(A_r; w) - metric(B_r; w)"}

    def repeats(self, ds: str, cell: str) -> int:
        base = self.res.site_rate_cell(ds) or "fedavg_lr" if cell in VIRTUAL else cell
        s = self.res.summary(ds, base, "paired_stats")
        return len(s["repeats"]) if s else 0

    def weights(self, ds: str, p: Preds) -> np.ndarray:
        if ds not in self._W:
            self._W[ds] = stratified_weights(p.hospital, self.resamples, self.seed, ds)
        return self._W[ds]

    def keep(self, ds: str) -> list:
        if ds not in self._keep:
            self._keep[ds] = included_hospitals(self.res, ds, self.spec)[0]
        return self._keep[ds]

    def compare(self, ds: str, a: str, b: str, metric: str, hospital: Optional[str] = None) -> Optional[dict]:
        """Paired difference A - B on `metric` (pooled, within hospital, or at one hospital)."""
        n_rep = min(self.repeats(ds, a), self.repeats(ds, b))
        if not n_rep:
            return None
        key = f"{ds}|{a}|{b}|{metric}|{hospital or ''}"
        fp = fingerprint(self.res, ds, (a, b), n_rep)
        hit = self.cache.get(key)
        if hit and hit.get("fingerprint") == fp:
            return hit
        scope = "hospital" if hospital else "pooled"
        diffs, boot, va, vb = [], None, [], []
        for r in range(n_rep):
            pa, pb = load_preds(self.res, ds, a, r), load_preds(self.res, ds, b, r)
            if pa is None or pb is None or not np.array_equal(pa.patients, pb.patients):
                return None
            W = self.weights(ds, pa)
            kw = {"scope": scope, "keep": self.keep(ds), "hospital": hospital}
            xa, xb = value(pa, metric, **kw), value(pb, metric, **kw)
            if xa is None or xb is None:
                return None
            va.append(xa)
            vb.append(xb)
            diffs.append(xa - xb)
            ba, bb = values(pa, metric, W, **kw), values(pb, metric, W, **kw)
            boot = (ba - bb) if boot is None else boot + (ba - bb)
        boot = boot / n_rep
        ok = np.isfinite(boot)
        lo_q, hi_q = 100 * (1 - self.level) / 2, 100 * (1 + self.level) / 2
        lo, hi = (np.percentile(boot[ok], [lo_q, hi_q]) if ok.any() else (np.nan, np.nan))

        def msd(x: list) -> tuple[float, float]:
            m = float(np.mean(x))
            return m, (float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)
        out = {"dataset": ds, "a": a, "b": b, "metric": metric, "hospital": hospital, "repeats": n_rep,
               "a_mean": msd(va)[0], "a_std": msd(va)[1], "b_mean": msd(vb)[0], "b_std": msd(vb)[1],
               "per_repeat_diff": diffs, "diff_mean": msd(diffs)[0], "diff_std": msd(diffs)[1],
               "ci": [float(lo), float(hi)], "level": self.level, "resamples": self.resamples,
               "resamples_valid": int(ok.sum()), "fingerprint": fp}
        self.cache[key] = out
        self.dirty = True
        return out

    def save(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.path)
        self.dirty = False


def within_value(res: Any, spec: dict, ds: str, cell: str, repeat: int, metric: str) -> Optional[float]:
    """A run's within-hospital metric for one repeat (for the tables)."""
    p = load_preds(res, ds, cell, repeat)
    if p is None:
        return None
    return value(p, metric, scope="within", keep=included_hospitals(res, ds, spec)[0])
