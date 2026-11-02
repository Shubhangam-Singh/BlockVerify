"""FedVerify-Survival metrics (prompt §4): C-index sign, ROC-AUC, F1 threshold rule, per-hospital dashes."""
import numpy as np
import pytest

pytest.importorskip("lifelines")
pytest.importorskip("sklearn")

from fedverify.survival import metrics as M  # noqa: E402


def test_c_index_sign():
    """A PERFECT model: the earliest death has the highest risk. Fails if the sign passed to lifelines is wrong."""
    time, event = [1, 2, 3, 4], [1, 1, 1, 1]
    risk = [0.9, 0.7, 0.4, 0.1]
    assert M.c_index(time, event, risk) == 1.0
    assert M.c_index(time, event, [-r for r in risk]) == 0.0
    from lifelines.utils import concordance_index
    assert concordance_index(time, risk, event) == 0.0          # the trap: lifelines wants higher = LONGER survival


def test_c_index_respects_censoring_and_needs_a_death():
    # patient 2 is censored at t=2, so only the pairs anchored on the deaths at t=1 and t=3 count
    assert M.c_index([1, 2, 3], [1, 0, 1], [3.0, 2.0, 1.0]) == 1.0
    assert M.c_index([1, 2, 3], [0, 0, 0], [3.0, 2.0, 1.0]) is None


def test_roc_auc_and_single_class():
    assert M.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.roc_auc([1, 1, 1], [0.1, 0.2, 0.3]) is None
    assert M.roc_auc([0, 0, 0], [0.1, 0.2, 0.3]) is None


def test_f1_score_edge_cases():
    assert M.f1_score([1, 0, 1], [1, 0, 1]) == 1.0
    assert M.f1_score([1, 0, 1], [0, 0, 0]) == 0.0
    assert M.f1_score([0, 0], [0, 0]) == 0.0


def _brute_force(event, risk):
    best_f1, best_t = -1.0, None
    for t in sorted(set(risk), reverse=True):                   # descending: first max is the highest threshold
        f1 = M.f1_score(event, [1 if r >= t else 0 for r in risk])
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


@pytest.mark.parametrize("seed", range(25))
def test_best_f1_threshold_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 60))
    event = (rng.random(n) < 0.3).astype(int)
    event[0] = 1
    risk = np.round(rng.random(n), 1)                          # coarse values -> many ties
    assert M.best_f1_threshold(event, risk) == pytest.approx(_brute_force(event.tolist(), risk.tolist()))


def test_best_f1_threshold_without_deaths_predicts_nobody():
    assert M.best_f1_threshold([0, 0, 0], [0.2, 0.5, 0.9]) == float("inf")


def test_thresholds_pooled_vs_per_hospital():
    tr_e = np.array([1, 0, 0, 1, 0, 0])
    tr_r = np.array([0.9, 0.1, 0.2, 0.4, 0.3, 0.35])
    tr_h = np.array(["BH", "BH", "BH", "A2", "A2", "A2"])
    te_h = np.array(["BH", "A2"])
    pooled = M.thresholds_for_test(tr_e, tr_r, tr_h, te_h, per_hospital=False)
    own = M.thresholds_for_test(tr_e, tr_r, tr_h, te_h, per_hospital=True)
    assert pooled[0] == pooled[1] == M.best_f1_threshold(tr_e, tr_r)
    assert own[0] == M.best_f1_threshold(tr_e[:3], tr_r[:3])
    assert own[1] == M.best_f1_threshold(tr_e[3:], tr_r[3:])


def test_per_hospital_rank_metrics_dashed_below_min_deaths():
    hosp = np.array(["BH"] * 10 + ["A8"] * 10)
    event = np.array([1] * 5 + [0] * 5 + [1] * 4 + [0] * 6)     # BH 5 deaths, A8 4 deaths
    time = np.arange(20, dtype=float) + 1
    risk = np.linspace(1, 0, 20)
    pred = (risk > 0.5).astype(int)
    out = M.per_hospital(["c_index", "roc_auc", "f1"], hosp, time, event, risk, pred, 5, ["c_index", "roc_auc"])
    assert out["BH"]["c_index"] is not None and out["BH"]["roc_auc"] is not None
    assert out["A8"]["c_index"] is None and out["A8"]["roc_auc"] is None
    assert out["A8"]["f1"] is not None and out["A8"]["deaths"] == 4


def test_metrics_yaml_defaults():
    cfg = M.load_metric_config()
    assert cfg["metrics"] == ["c_index", "roc_auc", "f1"]
    assert cfg["min_deaths"] == 5 and cfg["rank_metrics"] == ["c_index", "roc_auc"]
