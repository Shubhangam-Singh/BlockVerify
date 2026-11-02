"""Random-forest family (prompt §5): local_rf, central_rf, fed_rf. No DP (Phase 2 marks RF-DP cells n/a).

  local_rf    each hospital grows its own forest on its own rows; each patient is scored by their OWN hospital's
              forest
  central_rf  one forest on the pooled training rows, growing hospitals x trees_per_hospital trees per round, so it
              ends with the same number of trees as fed_rf
  fed_rf      each round every hospital trains trees_per_hospital trees on its own rows; the server concatenates
              all trees and the global prediction is the MEAN of the tree probabilities
  chain_fed_rf  fed_rf with every forest committed and rebuilt from its bytes (Phase 3)
  ours_rf     chain_fed_rf + the probe-set screen (survival/screening.py) on each hospital's new trees; only the
              survivors' trees are appended (Phase 4). With screening off it is chain_fed_rf under the same attack.

A hospital's update is the set of trees it adds that round (D6: trees have no weights to subtract, so "the trees
added this round" is the delta). Trees are kept in memory only; nothing is pickled. Determinism (rule 6): n_jobs=1
and every forest's random_state is derived from (repeat seed, fold, round, hospital).
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import poison
from .common import RoundOutput, local_hospitals, plain_diag
from .config import SurvivalConfig, derive_seed
from .chain_round import FoldCommitter, skipped_record
from .data import FoldData
from .probe import probe_set
from .screening import TreeProbeScreen, leaf_values
from .serialize import serialize_rf


def make_forest(cfg: SurvivalConfig, n_trees: int, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=int(n_trees), max_depth=cfg.max_depth,
                                  min_samples_leaf=cfg.min_samples_leaf, class_weight=cfg.rf_class_weight,
                                  n_jobs=cfg.rf_n_jobs, random_state=int(random_state))


def death_column(forest: RandomForestClassifier) -> Optional[int]:
    """Column of class 1 in predict_proba, or None if the training rows held no death."""
    hit = np.flatnonzero(np.asarray(forest.classes_) == 1)
    return int(hit[0]) if hit.size else None


def tree_probability_sum(forest: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """Sum over the forest's trees of P(death) — the server's running sum for 'mean of tree probabilities'."""
    col = death_column(forest)
    total = np.zeros(len(X), dtype=np.float64)
    if col is None or len(X) == 0:
        return total
    for tree in forest.estimators_:
        total += tree.predict_proba(X)[:, col]
    return total


def attacker_training_set(cfg: SurvivalConfig, X: np.ndarray, y: np.ndarray, feature_names: list,
                          *seed_parts: object) -> tuple[np.ndarray, np.ndarray]:
    """What a malicious hospital grows its trees on (survival/poison.py): flipped labels, or random schema rows with a
    permutation of its labels."""
    if cfg.attack == "label_flip_trees":
        return X, poison.flipped(y)
    if cfg.attack == "random_trees":
        X_rand, perm = poison.random_tree_rows(feature_names, len(y), *seed_parts)
        return X_rand, y[perm]
    raise ValueError(f"{cfg.attack!r} is not a tree attack")


def run_fold(cfg: SurvivalConfig, fold: FoldData, repeat_seed: int, fold_index: int,
             committer: Optional[FoldCommitter] = None, attackers: Optional[list] = None,
             feature_names: Optional[list] = None, **_: object) -> Iterator[RoundOutput]:
    """Train one method on one fold; yield a RoundOutput after every round. attackers and feature_names (ours_rf
    only): this repeat's malicious hospitals, and the schema for the probe set and the random_trees attack."""
    method = cfg.method
    X_tr = np.ascontiguousarray(fold.X_train, dtype=np.float32)
    X_te = np.ascontiguousarray(fold.X_test, dtype=np.float32)
    y = fold.event_train.astype(np.int64)
    hosp = [h for h in cfg.hospitals if (fold.hosp_train == h).any()]
    rows = {h: np.flatnonzero(fold.hosp_train == h) for h in hosp}
    test_rows = {h: np.flatnonzero(fold.hosp_test == h) for h in hosp}
    n_samples = {h: int(len(rows[h])) for h in hosp}

    if method == "fed_rf":
        sum_tr, sum_te, n_trees = np.zeros(len(X_tr)), np.zeros(len(X_te)), 0
        for rnd in range(1, cfg.rounds + 1):
            t0 = time.perf_counter()
            scores = {}
            for h in hosp:
                forest = make_forest(cfg, cfg.trees_per_hospital, derive_seed(repeat_seed, fold_index, rnd, h))
                forest.fit(X_tr[rows[h]], y[rows[h]])
                # server side: concatenate this hospital's new trees into the global forest
                sum_tr += tree_probability_sum(forest, X_tr)
                sum_te += tree_probability_sum(forest, X_te)
                n_trees += len(forest.estimators_)
                scores[h] = {"trees_added": len(forest.estimators_)}
            yield RoundOutput(rnd, sum_tr / n_trees, sum_te / n_trees, plain_diag(hosp, scores), n_samples, None,
                              time.perf_counter() - t0, {"trees_total": n_trees})
        return

    if method == "chain_fed_rf":         # fed_rf, with every forest committed and rebuilt from its bytes
        sum_tr, sum_te, n_trees = np.zeros(len(X_tr)), np.zeros(len(X_te)), 0
        for rnd in range(1, cfg.rounds + 1):
            t0 = time.perf_counter()
            skip = committer.check_previous()
            if skip:
                with np.errstate(invalid="ignore"):
                    yield RoundOutput(rnd, sum_tr / n_trees, sum_te / n_trees,
                                      {"accepted": [], "rejected": [], "scores": {}}, {}, None,
                                      time.perf_counter() - t0, {"commit": skipped_record(skip), "trees_total": n_trees})
                continue
            forests = {}
            for h in hosp:
                forests[h] = make_forest(cfg, cfg.trees_per_hospital, derive_seed(repeat_seed, fold_index, rnd, h))
                forests[h].fit(X_tr[rows[h]], y[rows[h]])
            res = committer.commit_round(rnd, {h: (lambda h=h: serialize_rf(
                committer.run_id, rnd, h, n_samples[h], forests[h], death_column(forests[h]))) for h in hosp})
            scores = {}
            for h in res.accepted:               # server: concatenate the trees rebuilt from committed bytes
                model = res.models[h]
                sum_tr += model.probability_sum(X_tr)
                sum_te += model.probability_sum(X_te)
                n_trees += len(model.trees)
                scores[h] = {"trees_added": len(model.trees)}
            with np.errstate(invalid="ignore"):
                yield RoundOutput(rnd, sum_tr / n_trees, sum_te / n_trees, plain_diag(res.accepted, scores),
                                  n_samples, None, time.perf_counter() - t0,
                                  {"commit": res.record, "trees_total": n_trees}, commitment=res.commitment)
        return

    if method == "ours_rf":              # chain_fed_rf + attack before the commitment + probe-set screen after it
        bad = set(attackers or ())
        screen = TreeProbeScreen(cfg.tau) if cfg.screened else None
        X_probe = probe_set(cfg, feature_names) if screen is not None else None
        sum_tr, sum_te, n_trees = np.zeros(len(X_tr)), np.zeros(len(X_te)), 0
        for rnd in range(1, cfg.rounds + 1):
            t0 = time.perf_counter()
            skip = committer.check_previous()
            if skip:
                with np.errstate(invalid="ignore"):
                    yield RoundOutput(rnd, sum_tr / n_trees, sum_te / n_trees,
                                      {"accepted": [], "rejected": [], "scores": {}}, {}, None,
                                      time.perf_counter() - t0, {"commit": skipped_record(skip), "trees_total": n_trees})
                continue
            forests = {}
            for h in hosp:
                Xh, yh = X_tr[rows[h]], y[rows[h]]
                if h in bad:
                    Xh, yh = attacker_training_set(cfg, Xh, yh, feature_names, repeat_seed, fold_index, rnd, h)
                forests[h] = make_forest(cfg, cfg.trees_per_hospital, derive_seed(repeat_seed, fold_index, rnd, h))
                forests[h].fit(Xh, yh)
            res = committer.commit_round(rnd, {h: (lambda h=h: serialize_rf(
                committer.run_id, rnd, h, n_samples[h], forests[h], death_column(forests[h]))) for h in hosp})
            if screen is not None and res.accepted:
                kept, diag = screen.screen(
                    {h: res.models[h].probability_sum(X_probe) / max(1, len(res.models[h].trees)) for h in res.accepted},
                    {h: leaf_values(res.models[h]) for h in res.accepted})
            else:
                kept, diag = list(res.accepted), plain_diag(res.accepted)
            for h in kept:                       # server: concatenate the surviving trees, rebuilt from their bytes
                model = res.models[h]
                sum_tr += model.probability_sum(X_tr)
                sum_te += model.probability_sum(X_te)
                n_trees += len(model.trees)
                diag["scores"].setdefault(h, {})["trees_added"] = len(model.trees)
            with np.errstate(invalid="ignore"):
                yield RoundOutput(rnd, sum_tr / n_trees, sum_te / n_trees, diag, n_samples, None,
                                  time.perf_counter() - t0, {"commit": res.record, "trees_total": n_trees},
                                  commitment=res.commitment)
        return

    if method == "local_rf":
        modelled = local_hospitals(cfg, hosp)          # heart: Switzerland gets no local model (Revision 2 §2.1)
        sum_tr, sum_te = np.zeros(len(X_tr)), np.zeros(len(X_te))
        own = {h: 0 for h in modelled}
        for rnd in range(1, cfg.rounds + 1):
            t0 = time.perf_counter()
            for h in modelled:                               # each patient is scored by their own hospital's forest
                forest = make_forest(cfg, cfg.trees_per_hospital, derive_seed(repeat_seed, fold_index, rnd, h))
                forest.fit(X_tr[rows[h]], y[rows[h]])
                sum_tr[rows[h]] += tree_probability_sum(forest, X_tr[rows[h]])
                sum_te[test_rows[h]] += tree_probability_sum(forest, X_te[test_rows[h]])
                own[h] += len(forest.estimators_)
            n_tr = np.array([own.get(h, 0) for h in fold.hosp_train], dtype=np.float64)
            n_te = np.array([own.get(h, 0) for h in fold.hosp_test], dtype=np.float64)
            with np.errstate(invalid="ignore"):              # 0/0 = NaN marks rows of an unmodelled hospital
                risk_tr, risk_te = sum_tr / n_tr, sum_te / n_te
            yield RoundOutput(rnd, risk_tr, risk_te,
                              plain_diag(modelled, {h: {"trees_added": cfg.trees_per_hospital} for h in modelled}),
                              {h: n_samples[h] for h in modelled}, None, time.perf_counter() - t0,
                              {"trees_total": int(sum(own.values()))})
        return

    if method != "central_rf":
        raise ValueError(f"{method!r} is not a random-forest-family method")
    sum_tr, sum_te, n_trees = np.zeros(len(X_tr)), np.zeros(len(X_te)), 0
    per_round = cfg.trees_per_hospital * len(hosp)
    for rnd in range(1, cfg.rounds + 1):
        t0 = time.perf_counter()
        forest = make_forest(cfg, per_round, derive_seed(repeat_seed, fold_index, rnd, "central"))
        forest.fit(X_tr, y)
        sum_tr += tree_probability_sum(forest, X_tr)
        sum_te += tree_probability_sum(forest, X_te)
        n_trees += len(forest.estimators_)
        yield RoundOutput(rnd, sum_tr / n_trees, sum_te / n_trees, plain_diag(hosp), n_samples, None,
                          time.perf_counter() - t0, {"trees_total": n_trees})
