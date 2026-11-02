"""XGBoost family (Revision 2 §2.2-2.3): local_xgb, central_xgb, fedxgb_bagging. DP for XGBoost is n/a.

chain_fedxgb_bagging commits every hospital's new trees and appends them rebuilt from their bytes (Phase 3). ours_xgb
adds the probe-set screen (survival/screening.py) on each hospital's new trees, so only the survivors' trees are
appended (Phase 4); with screening off it is chain_fedxgb_bagging under the same attack.

Two tasks:
  binary    objective binary:logistic (heart disease; the TCGA death indicator). Score = P(event).
  survival  objective survival:aft, aft_loss_distribution normal, scale 1.0 (TCGA only). Lower bound = time,
            upper bound = time for a death, +inf if censored; times below aft_min_time_days are clipped up to it.
            Score = predicted log survival TIME: higher = LONGER survival, so the C-index takes it UNCHANGED.

FedXGBBagging, the combination rule used everywhere (also in DESIGN_NOTES.md):
  Each round the server broadcasts the global ensemble. Every hospital starts from the SAME global margin on its own
  training rows (passed as base_margin) and boosts N new trees on those rows only; it sends ONLY those trees. The
  server appends every hospital's new trees to the global ensemble, each round's boosters AVERAGED over the K_r
  hospitals that contributed in that round (Phase 7 correction, see DESIGN_NOTES.md). The global margin of a row x is

      margin(x) = m0 + sum over rounds r of (1 / K_r) * sum over contributing hospitals h of booster_{r,h}(x)

  where booster_{r,h}(x) is the sum of hospital h's N new trees of round r. m0 = 0 for binary (P = 0.5) and
  m0 = ln(aft_init_days) for AFT (a fixed, data-independent start). The prediction is sigmoid(margin) for binary and
  the raw margin (log survival time) for AFT. After round r the ensemble holds exactly hospitals x N x r trees.
  Until Phase 7 the boosters were summed, not averaged: every round then moved the margin by about K times the
  learning rate, and heart's held-out log-loss rose every round.

local_xgb grows each hospital's own ensemble the same way on its own rows and scores each patient with their own
hospital's ensemble (heart: Switzerland gets none). central_xgb boosts hospitals x N trees per round on the pooled
rows (same tree budget as fedxgb_bagging). central_xgb_steps (Phase 9, a diagnostic row whose rule is fixed in
table.yaml) boosts N trees per round on the pooled rows: the same number of sequential boosting steps as one averaged
FedXGBBagging booster per round, and otherwise exactly central_xgb (same settings, same seeds). Determinism: nthread 1, tree_method hist, seed from
(repeat seed, fold, round, hospital). Trees live in memory only; nothing is pickled.
"""
from __future__ import annotations

import math
import time
from typing import Iterator, Optional

import numpy as np
import xgboost as xgb

from . import poison
from .common import LOG_TIME, RISK, RoundOutput, local_hospitals, plain_diag
from .config import SurvivalConfig, derive_seed
from .chain_round import FoldCommitter, skipped_record
from .data import FoldData
from .probe import probe_set
from .screening import TreeProbeScreen, leaf_values
from .serialize import serialize_xgb


def params(cfg: SurvivalConfig, seed: int) -> dict:
    """Native-API twin of XGBClassifier(objective="binary:logistic", tree_method="hist", n_jobs=1, max_depth=3,
    learning_rate=0.1, subsample=0.8, random_state=seed), or the AFT objective for task=survival."""
    p = {"tree_method": "hist", "max_depth": cfg.xgb_max_depth, "eta": cfg.xgb_learning_rate,
         "subsample": cfg.xgb_subsample, "nthread": cfg.xgb_n_jobs, "seed": int(seed), "verbosity": 0}
    if cfg.task == "binary":
        p.update(objective="binary:logistic", base_score=0.5)
    else:
        p.update(objective="survival:aft", aft_loss_distribution=cfg.aft_distribution,
                 aft_loss_distribution_scale=cfg.aft_scale, base_score=1.0)
    return p


def initial_margin(cfg: SurvivalConfig) -> float:
    return 0.0 if cfg.task == "binary" else math.log(cfg.aft_init_days)


def aft_bounds(cfg: SurvivalConfig, time_days: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AFT interval labels: [t, t] for a death, [t, +inf) if censored; t clipped up to aft_min_time_days."""
    t = np.maximum(np.asarray(time_days, dtype=np.float64), cfg.aft_min_time_days)
    return t, np.where(np.asarray(event) == 1, t, np.inf)


def make_dmatrix(cfg: SurvivalConfig, X: np.ndarray, event: np.ndarray, time_days: np.ndarray,
                 base_margin: np.ndarray) -> xgb.DMatrix:
    dm = xgb.DMatrix(X, nthread=cfg.xgb_n_jobs)
    if cfg.task == "binary":
        dm.set_label(np.asarray(event, dtype=np.float32))
    else:
        lo, hi = aft_bounds(cfg, time_days, event)
        dm.set_float_info("label_lower_bound", lo)
        dm.set_float_info("label_upper_bound", hi)
    dm.set_base_margin(np.asarray(base_margin, dtype=np.float32))
    return dm


def train_new_trees(cfg: SurvivalConfig, X: np.ndarray, event: np.ndarray, time_days: np.ndarray,
                    base_margin: np.ndarray, n_trees: int, seed: int) -> xgb.Booster:
    """Boost n_trees NEW trees on top of base_margin; the returned booster holds exactly those trees."""
    return xgb.train(params(cfg, seed), make_dmatrix(cfg, X, event, time_days, base_margin),
                     num_boost_round=int(n_trees))


def zero_margin_matrix(cfg: SurvivalConfig, X: np.ndarray) -> xgb.DMatrix:
    """DMatrix whose base_margin is 0, so predict(output_margin=True) returns the SUM OF THE TREES only."""
    dm = xgb.DMatrix(X, nthread=cfg.xgb_n_jobs)
    dm.set_base_margin(np.zeros(len(X), dtype=np.float32))
    return dm


def tree_sum(booster: xgb.Booster, dm_zero: xgb.DMatrix) -> np.ndarray:
    return booster.predict(dm_zero, output_margin=True).astype(np.float64)


def to_score(cfg: SurvivalConfig, margin: np.ndarray) -> np.ndarray:
    """binary: sigmoid(margin) = P(event); survival: the raw margin = predicted log survival time."""
    if cfg.task == "binary":
        return 1.0 / (1.0 + np.exp(-margin))
    return margin.copy()


def attacker_trees(cfg: SurvivalConfig, X: np.ndarray, event: np.ndarray, time_days: np.ndarray,
                   margin: np.ndarray, n_trees: int, seed: int, feature_names: list, *seed_parts: object) -> xgb.Booster:
    """A malicious hospital's N new trees (survival/poison.py). label_flip_trees: its own rows, boosted from the
    broadcast margin like an honest hospital, with flipped labels (AFT: flipped event indicators, times unchanged).
    random_trees: random schema rows with a permutation of its (event, time) labels, boosted from the fixed initial
    margin (random rows have no broadcast margin)."""
    if cfg.attack == "label_flip_trees":
        return train_new_trees(cfg, X, poison.flipped(event), time_days, margin, n_trees, seed)
    if cfg.attack == "random_trees":
        X_rand, perm = poison.random_tree_rows(feature_names, len(event), *seed_parts)
        return train_new_trees(cfg, X_rand, event[perm], time_days[perm], np.full(len(event), initial_margin(cfg)),
                               n_trees, seed)
    raise ValueError(f"{cfg.attack!r} is not a tree attack")


def run_fold(cfg: SurvivalConfig, fold: FoldData, repeat_seed: int, fold_index: int,
             committer: Optional[FoldCommitter] = None, attackers: Optional[list] = None,
             feature_names: Optional[list] = None, **_: object) -> Iterator[RoundOutput]:
    """Train one method on one fold; yield a RoundOutput after every round. attackers and feature_names (ours_xgb
    only): this repeat's malicious hospitals, and the schema for the probe set and the random_trees attack."""
    method = cfg.method
    kind = RISK if cfg.task == "binary" else LOG_TIME
    N = cfg.xgb_trees_per_round
    X_tr = np.ascontiguousarray(fold.X_train, dtype=np.float32)
    X_te = np.ascontiguousarray(fold.X_test, dtype=np.float32)
    ev, tm = fold.event_train, fold.time_train
    hosp = [h for h in cfg.hospitals if (fold.hosp_train == h).any()]
    rows = {h: np.flatnonzero(fold.hosp_train == h) for h in hosp}
    test_rows = {h: np.flatnonzero(fold.hosp_test == h) for h in hosp}
    n_samples = {h: int(len(rows[h])) for h in hosp}
    m0 = initial_margin(cfg)
    dm_tr, dm_te = zero_margin_matrix(cfg, X_tr), zero_margin_matrix(cfg, X_te)

    if method == "fedxgb_bagging":
        margin_tr, margin_te, n_trees = np.full(len(X_tr), m0), np.full(len(X_te), m0), 0
        for rnd in range(1, cfg.rounds + 1):
            t0 = time.perf_counter()
            new = {}
            for h in hosp:          # every hospital starts from the SAME broadcast global ensemble
                r = rows[h]
                new[h] = train_new_trees(cfg, X_tr[r], ev[r], tm[r], margin_tr[r], N,
                                         derive_seed(repeat_seed, fold_index, rnd, h))
            for h in hosp:          # server: append every hospital's new trees, averaged over the round's boosters
                margin_tr += tree_sum(new[h], dm_tr) / len(hosp)
                margin_te += tree_sum(new[h], dm_te) / len(hosp)
                n_trees += new[h].num_boosted_rounds()
            yield RoundOutput(rnd, to_score(cfg, margin_tr), to_score(cfg, margin_te),
                              plain_diag(hosp, {h: {"trees_added": new[h].num_boosted_rounds()} for h in hosp}),
                              n_samples, None, time.perf_counter() - t0, {"trees_total": n_trees},
                              score_kind=kind)
        return

    if method == "chain_fedxgb_bagging":   # fedxgb_bagging, with every hospital's trees committed and rebuilt
        objective = params(cfg, 0)["objective"]
        margin_tr, margin_te, n_trees = np.full(len(X_tr), m0), np.full(len(X_te), m0), 0
        for rnd in range(1, cfg.rounds + 1):
            t0 = time.perf_counter()
            skip = committer.check_previous()
            if skip:
                yield RoundOutput(rnd, to_score(cfg, margin_tr), to_score(cfg, margin_te),
                                  {"accepted": [], "rejected": [], "scores": {}}, {}, None, time.perf_counter() - t0,
                                  {"commit": skipped_record(skip), "trees_total": n_trees}, score_kind=kind)
                continue
            new = {}
            for h in hosp:
                r = rows[h]
                new[h] = train_new_trees(cfg, X_tr[r], ev[r], tm[r], margin_tr[r], N,
                                         derive_seed(repeat_seed, fold_index, rnd, h))
            res = committer.commit_round(rnd, {h: (lambda h=h: serialize_xgb(
                committer.run_id, rnd, h, n_samples[h], new[h], objective)) for h in hosp})
            for h in res.accepted:               # server: append the trees rebuilt from committed bytes, averaged
                margin_tr += res.models[h].tree_sum(X_tr) / len(res.accepted)
                margin_te += res.models[h].tree_sum(X_te) / len(res.accepted)
                n_trees += len(res.models[h].trees)
            yield RoundOutput(rnd, to_score(cfg, margin_tr), to_score(cfg, margin_te),
                              plain_diag(res.accepted, {h: {"trees_added": len(res.models[h].trees)}
                                                        for h in res.accepted}),
                              n_samples, None, time.perf_counter() - t0,
                              {"commit": res.record, "trees_total": n_trees}, score_kind=kind,
                              commitment=res.commitment)
        return

    if method == "ours_xgb":   # chain_fedxgb_bagging + attack before the commitment + probe-set screen after it
        objective = params(cfg, 0)["objective"]
        bad = set(attackers or ())
        screen = TreeProbeScreen(cfg.tau) if cfg.screened else None
        X_probe = probe_set(cfg, feature_names) if screen is not None else None
        margin_tr, margin_te, n_trees = np.full(len(X_tr), m0), np.full(len(X_te), m0), 0
        for rnd in range(1, cfg.rounds + 1):
            t0 = time.perf_counter()
            skip = committer.check_previous()
            if skip:
                yield RoundOutput(rnd, to_score(cfg, margin_tr), to_score(cfg, margin_te),
                                  {"accepted": [], "rejected": [], "scores": {}}, {}, None, time.perf_counter() - t0,
                                  {"commit": skipped_record(skip), "trees_total": n_trees}, score_kind=kind)
                continue
            new = {}
            for h in hosp:
                r = rows[h]
                seed = derive_seed(repeat_seed, fold_index, rnd, h)
                if h in bad:
                    new[h] = attacker_trees(cfg, X_tr[r], ev[r], tm[r], margin_tr[r], N, seed, feature_names,
                                            repeat_seed, fold_index, rnd, h)
                else:
                    new[h] = train_new_trees(cfg, X_tr[r], ev[r], tm[r], margin_tr[r], N, seed)
            res = committer.commit_round(rnd, {h: (lambda h=h: serialize_xgb(
                committer.run_id, rnd, h, n_samples[h], new[h], objective)) for h in hosp})
            if screen is not None and res.accepted:
                kept, diag = screen.screen({h: res.models[h].tree_sum(X_probe) for h in res.accepted},
                                           {h: leaf_values(res.models[h]) for h in res.accepted})
            else:
                kept, diag = list(res.accepted), plain_diag(res.accepted)
            for h in kept:                       # server: append the surviving trees (rebuilt from their bytes), averaged
                margin_tr += res.models[h].tree_sum(X_tr) / len(kept)
                margin_te += res.models[h].tree_sum(X_te) / len(kept)
                n_trees += len(res.models[h].trees)
                diag["scores"].setdefault(h, {})["trees_added"] = len(res.models[h].trees)
            yield RoundOutput(rnd, to_score(cfg, margin_tr), to_score(cfg, margin_te), diag, n_samples, None,
                              time.perf_counter() - t0, {"commit": res.record, "trees_total": n_trees},
                              score_kind=kind, commitment=res.commitment)
        return

    if method == "local_xgb":
        modelled = local_hospitals(cfg, hosp)          # heart: Switzerland gets no local model (Revision 2 §2.1)
        margin_tr = np.full(len(X_tr), np.nan)
        margin_te = np.full(len(X_te), np.nan)
        for h in modelled:
            margin_tr[rows[h]] = m0
            margin_te[test_rows[h]] = m0
        own = {h: 0 for h in modelled}
        for rnd in range(1, cfg.rounds + 1):
            t0 = time.perf_counter()
            for h in modelled:      # each patient is scored by their own hospital's ensemble
                r, te = rows[h], test_rows[h]
                b = train_new_trees(cfg, X_tr[r], ev[r], tm[r], margin_tr[r], N,
                                    derive_seed(repeat_seed, fold_index, rnd, h))
                margin_tr[r] += tree_sum(b, zero_margin_matrix(cfg, X_tr[r]))
                if len(te):
                    margin_te[te] += tree_sum(b, zero_margin_matrix(cfg, X_te[te]))
                own[h] += b.num_boosted_rounds()
            yield RoundOutput(rnd, to_score(cfg, margin_tr), to_score(cfg, margin_te),
                              plain_diag(modelled, {h: {"trees_added": N} for h in modelled}),
                              {h: n_samples[h] for h in modelled}, None, time.perf_counter() - t0,
                              {"trees_total": int(sum(own.values()))}, score_kind=kind)
        return

    if method not in ("central_xgb", "central_xgb_steps"):
        raise ValueError(f"{method!r} is not an XGBoost-family method")
    # central_xgb: hospitals x N trees per round, the same tree count as fedxgb_bagging. central_xgb_steps: N trees per
    # round, the same sequential boosting steps as fedxgb_bagging's one averaged booster (Phase 9 diagnostic row).
    per_round = N * len(hosp) if method == "central_xgb" else N
    margin_tr, margin_te, n_trees = np.full(len(X_tr), m0), np.full(len(X_te), m0), 0
    for rnd in range(1, cfg.rounds + 1):
        t0 = time.perf_counter()
        b = train_new_trees(cfg, X_tr, ev, tm, margin_tr, per_round, derive_seed(repeat_seed, fold_index, rnd, "central"))
        margin_tr += tree_sum(b, dm_tr)
        margin_te += tree_sum(b, dm_te)
        n_trees += b.num_boosted_rounds()
        yield RoundOutput(rnd, to_score(cfg, margin_tr), to_score(cfg, margin_te), plain_diag(hosp), n_samples,
                          None, time.perf_counter() - t0, {"trees_total": n_trees}, score_kind=kind)
