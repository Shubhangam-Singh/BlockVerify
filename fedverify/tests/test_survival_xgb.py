"""Revision 2 Phase 2b: XGBoost (binary and AFT survival) and the FedXGBBagging combination rule."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("lifelines")
pytest.importorskip("sklearn")

from fedverify.core.runner import strip_timings  # noqa: E402
from fedverify.survival import data as D  # noqa: E402
from fedverify.survival import fl_xgb, runner  # noqa: E402
from fedverify.survival import heart as H  # noqa: E402
from fedverify.survival import metrics as M  # noqa: E402
from fedverify.survival.common import LOG_TIME, RISK  # noqa: E402
from fedverify.survival.config import HEART_HOSPITALS, HOSPITALS, SurvivalConfig, derive_seed  # noqa: E402

needs_tcga = pytest.mark.skipif(not D.DEFAULT_DATA.exists(), reason="TCGA workbook (gitignored) not present")
needs_heart = pytest.mark.skipif(not H.DEFAULT_DATA.exists(), reason="heart CSV (gitignored) not present")


def _xgb(task="binary", dataset="tcga", method="fedxgb_bagging", **kw):
    return SurvivalConfig(family="xgb", method=method, task=task, dataset=dataset, rounds=kw.pop("rounds", 2), **kw)


@pytest.fixture(scope="module")
def tcga_fold():
    c = D.load_cohort()
    X, _ = D.encode_features(c.features, D.load_vocab())
    tr, te = D.make_folds(c, 0)[0]
    return D.fold_data(c, X, tr, te)


@pytest.fixture(scope="module")
def heart_fold():
    c = H.load_cohort()
    X, _ = H.encode_features(c.features, H.load_vocab(), "paper")
    tr, te = D.make_folds(c, 0)[0]
    return D.fold_data(c, X, tr, te)


# ── AFT sign and labels ──────────────────────────────────────────────────────
def test_aft_sign_predicted_time_goes_to_the_c_index_unchanged():
    """A PERFECT AFT model: longer predicted time for longer actual time. Fails if the time were negated."""
    time, event = [1, 2, 3, 4], [1, 1, 1, 1]
    predicted_time = [10.0, 20.0, 30.0, 40.0]
    assert M.c_index_time(time, event, predicted_time) == 1.0
    assert M.c_index(time, event, predicted_time) == 0.0     # treating a time as a risk is the trap


def test_aft_interval_labels_and_clipping():
    cfg = _xgb(task="survival")
    lo, hi = fl_xgb.aft_bounds(cfg, np.array([0.0, 5.0, 10.0]), np.array([0, 1, 0]))
    np.testing.assert_array_equal(lo, [1.0, 5.0, 10.0])        # t = 0 clipped up to aft_min_time_days
    assert hi[1] == 5.0 and np.isinf(hi[0]) and np.isinf(hi[2])


def test_xgb_refusals():
    with pytest.raises(ValueError, match="n/a"):
        SurvivalConfig(family="xgb", method="fedxgb_bagging", epsilon=4.0)
    with pytest.raises(ValueError):
        SurvivalConfig(family="lr", method="fedavg_lr", task="survival")
    with pytest.raises(ValueError):
        SurvivalConfig(family="xgb", method="fedxgb_bagging", xgb_n_jobs=2)
    assert _xgb(task="survival").cell == "fedxgb_bagging_aft"


# ── FedXGBBagging ────────────────────────────────────────────────────────────
@needs_tcga
@pytest.mark.parametrize("task", ["binary", "survival"])
def test_round_r_holds_exactly_hospitals_x_N_x_r_trees_tcga(tcga_fold, task):
    cfg = _xgb(task=task, rounds=3)
    for r, out in enumerate(fl_xgb.run_fold(cfg, tcga_fold, 0, 0), start=1):
        assert out.extra["trees_total"] == len(HOSPITALS) * cfg.xgb_trees_per_round * r
        assert all(s["trees_added"] == cfg.xgb_trees_per_round for s in out.diag["scores"].values())
        assert out.score_kind == (RISK if task == "binary" else LOG_TIME)


@needs_heart
def test_round_r_holds_exactly_hospitals_x_N_x_r_trees_heart(heart_fold):
    cfg = _xgb(dataset="heart", rounds=3)
    for r, out in enumerate(fl_xgb.run_fold(cfg, heart_fold, 0, 0), start=1):
        assert out.extra["trees_total"] == len(HEART_HOSPITALS) * cfg.xgb_trees_per_round * r


@needs_tcga
@pytest.mark.parametrize("task", ["binary", "survival"])
def test_combination_rule_rebuilt_by_hand(tcga_fold, task):
    """Two rounds rebuilt from the rule: every hospital boosts N trees from the SAME global margin; the server adds
    the AVERAGE of their tree outputs (Phase 7); prediction = sigmoid(margin) (binary) or the raw margin (AFT)."""
    cfg = _xgb(task=task, rounds=2)
    outs = list(fl_xgb.run_fold(cfg, tcga_fold, 0, 0))
    X_tr = np.ascontiguousarray(tcga_fold.X_train, dtype=np.float32)
    X_te = np.ascontiguousarray(tcga_fold.X_test, dtype=np.float32)
    m_tr = np.full(len(X_tr), fl_xgb.initial_margin(cfg))
    m_te = np.full(len(X_te), fl_xgb.initial_margin(cfg))
    for rnd in (1, 2):
        new = []
        for h in HOSPITALS:
            r = tcga_fold.hosp_train == h
            new.append(fl_xgb.train_new_trees(cfg, X_tr[r], tcga_fold.event_train[r], tcga_fold.time_train[r],
                                              m_tr[r], cfg.xgb_trees_per_round, derive_seed(0, 0, rnd, h)))
        for b in new:
            m_tr = m_tr + b.predict(fl_xgb.zero_margin_matrix(cfg, X_tr), output_margin=True) / len(new)
            m_te = m_te + b.predict(fl_xgb.zero_margin_matrix(cfg, X_te), output_margin=True) / len(new)
        expect = 1 / (1 + np.exp(-m_te)) if task == "binary" else m_te
        np.testing.assert_allclose(outs[rnd - 1].test_risk, expect, rtol=0, atol=1e-6)


@needs_heart
def test_local_xgb_gives_switzerland_no_model(heart_fold):
    out = next(fl_xgb.run_fold(_xgb(dataset="heart", method="local_xgb", rounds=1), heart_fold, 0, 0))
    swiss = heart_fold.hosp_test == "Switzerland"
    assert np.isnan(out.test_risk[swiss]).all() and np.isfinite(out.test_risk[~swiss]).all()
    assert "Switzerland" not in out.per_client_num_samples


@needs_tcga
def test_aft_scores_are_log_times_and_rank_survival(tcga_fold):
    outs = list(fl_xgb.run_fold(_xgb(task="survival", rounds=2), tcga_fold, 0, 0))
    score = outs[-1].test_risk
    assert np.isfinite(score).all() and np.all(score > 0)          # around ln(365.25), i.e. days
    assert M.c_index_time(tcga_fold.time_test, tcga_fold.event_test, score) is not None


@needs_tcga
@pytest.mark.parametrize("task", ["binary", "survival"])
def test_xgb_runs_are_deterministic(tmp_path, task):
    outs = []
    for sub in ("a", "b"):
        cfg = SurvivalConfig(family="xgb", method="fedxgb_bagging", task=task, rounds=2, repeats=1,
                             out_dir=str(tmp_path / sub))
        runner.run(cfg, verbose=False)
        d = Path(cfg.run_dir)
        outs.append(([strip_timings(json.loads(l)) for p in sorted(d.rglob("rounds.jsonl"))
                      for l in p.read_text().splitlines()],
                     (d / "repeat0" / "oof.jsonl").read_text(), (d / "summary.json").read_text()))
    assert outs[0] == outs[1]


@needs_tcga
def test_aft_run_reports_c_index_only_and_clipped_times(tmp_path):
    cfg = SurvivalConfig(family="xgb", method="fedxgb_bagging", task="survival", rounds=1, repeats=1,
                         out_dir=str(tmp_path))
    s = runner.run(cfg, verbose=False)
    assert s["metrics"] == ["c_index"] and s["score_kind"] == LOG_TIME
    c = json.loads((Path(cfg.run_dir) / "config.json").read_text())
    # 10 censored rows with follow-up 0 (INVENTORY §4) + TCGA-BH-A1F8, a death at one day stored as
    # 0.9999999999999998 days by floating point (survival in years x 365.25), clipped up to exactly 1.0
    assert c["data"]["aft_times_clipped"] == 11
    rec = json.loads((Path(cfg.run_dir) / "repeat0" / "fold0" / "rounds.jsonl").read_text().splitlines()[0])
    assert all(v["test_acc"] is None for v in rec["hospital_metrics"].values())   # no probability, no accuracy
    assert "pred_log_time" in json.loads((Path(cfg.run_dir) / "repeat0" / "oof.jsonl").read_text().splitlines()[0])
