"""Revision 2 Phase 2b: UCI heart data (4 hospitals), paper / full modes, leakage guard, Switzerland rule, DP on heart.

Tests that read the heart CSV skip when it is absent (it is gitignored); everything else runs anywhere.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("lifelines")

from fedverify.core.runner import strip_timings  # noqa: E402
from fedverify.survival import data as D  # noqa: E402
from fedverify.survival import fl_lr, runner  # noqa: E402
from fedverify.survival import heart as H  # noqa: E402
from fedverify.survival.config import HEART_HOSPITALS, SurvivalConfig  # noqa: E402

needs_heart = pytest.mark.skipif(not H.DEFAULT_DATA.exists(),
                                 reason="heart CSV (gitignored) is not at fedverify/data/heart/heart_disease_uci.csv")

# Revision 2 §2.1, verified facts: (rows, positives num > 0) per hospital in the raw CSV, and paper-mode rows.
RAW = {"Cleveland": (304, 139), "Hungary": (293, 106), "VA Long Beach": (200, 149), "Switzerland": (123, 115)}
PAPER = {"Cleveland": 304, "Hungary": 260, "VA Long Beach": 130, "Switzerland": 46}


def _cfg(tmp_path, method="fedavg_lr", family="lr", **kw):
    return SurvivalConfig(dataset="heart", family=family, method=method, rounds=kw.pop("rounds", 1),
                          repeats=1, out_dir=str(tmp_path), **kw)


def _row(**over):
    row = {"age": 50, "sex": "Male", "cp": "asymptomatic", "trestbps": 120.0, "chol": 240.0, "fbs": False,
           "restecg": "normal", "thalch": 150.0, "exang": True, "oldpeak": 1.0}
    row.update(over)
    return row


# ── counts and modes ─────────────────────────────────────────────────────────
@needs_heart
def test_raw_counts_match_the_spec():
    raw = H.read_csv()
    assert raw.shape == (920, 16)
    got = {h: (int((raw.dataset == h).sum()), int(((raw.dataset == h) & (raw.num > 0)).sum())) for h in RAW}
    assert got == RAW


@needs_heart
def test_paper_mode_is_740_rows_with_the_reference_counts():
    c = H.load_cohort(mode="paper")
    assert c.n == 740 and c.dropped == 180
    assert {h: v["patients"] for h, v in c.counts().items()} == PAPER


@needs_heart
def test_full_mode_keeps_all_920_rows():
    c = H.load_cohort(mode="full")
    assert c.n == 920 and c.dropped == 0
    assert {h: (v["patients"], v["deaths"]) for h, v in c.counts().items()} == RAW


@needs_heart
def test_dropna_over_all_columns_is_the_trap_the_spec_warns_about():
    kept = H.read_csv().dropna()
    assert len(kept) == 299 and int((kept.dataset == "Cleveland").sum()) == 297


def test_paper_mode_drops_ca_thal_slope():
    for col in ("ca", "thal", "slope"):
        assert col not in H.feature_columns("paper") and col in H.feature_columns("full")


@needs_heart
def test_paper_and_full_matrices():
    c = H.load_cohort(mode="paper")
    X, names = H.encode_features(c.features, H.load_vocab(), "paper")
    assert X.shape == (740, 23) and np.isfinite(X).all()
    assert not any(n.split("=")[0] in ("ca", "thal", "slope") for n in names)
    assert not any(n.endswith("_missing") for n in names)
    f = H.load_cohort(mode="full")
    Xf, names_f = H.encode_features(f.features, H.load_vocab(), "full")
    assert Xf.shape == (920, 38) and "chol_missing" in names_f and "thal=NA" in names_f


# ── leakage guard and preprocessing ──────────────────────────────────────────
@pytest.mark.parametrize("col", ["num", "id", "dataset"])
def test_heart_leakage_guard_fires(col):
    with pytest.raises(D.LeakageError):
        H.assert_no_leakage(["age", col])


def test_label_column_cannot_reach_the_model():
    with pytest.raises(D.LeakageError):
        H.encode_features(pd.DataFrame([_row(num=2)]), H.load_vocab(), "paper")


def test_heart_encoding_is_data_independent():
    rows = [_row(), _row(age=71, sex="Female", cp="typical angina"), _row(chol=0.0, exang=False)]
    whole, _ = H.encode_features(pd.DataFrame(rows), H.load_vocab(), "paper")
    one_by_one = np.vstack([H.encode_features(pd.DataFrame([r]), H.load_vocab(), "paper")[0] for r in rows])
    np.testing.assert_array_equal(whole, one_by_one)


def test_full_mode_missing_values_and_scaling():
    row = _row(chol=np.nan, slope=np.nan, ca=np.nan, thal="normal")
    X, names = H.encode_features(pd.DataFrame([row]), H.load_vocab(), "full")
    col = {n: i for i, n in enumerate(names)}
    assert X[0, col["age"]] == pytest.approx(0.5)                   # /100, a fixed constant
    assert X[0, col["chol"]] == 0.0 and X[0, col["chol_missing"]] == 1.0
    assert X[0, col["ca_missing"]] == 1.0 and X[0, col["slope=NA"]] == 1.0
    assert X[0, col["thal=normal"]] == 1.0


def test_paper_mode_refuses_missing_values():
    with pytest.raises(ValueError, match="no missing"):
        H.encode_features(pd.DataFrame([_row(chol=np.nan)]), H.load_vocab(), "paper")


@needs_heart
def test_heart_vocab_json_is_the_full_file_vocabulary():
    assert H.build_vocab(H.read_csv()) == H.load_vocab()


def test_heart_config_defaults_and_refusals():
    c = SurvivalConfig(dataset="heart", family="lr", method="fedavg_lr")
    assert c.hospitals == HEART_HOSPITALS and c.local_exclude == ("Switzerland",)
    assert c.exp == "heart" and c.cell == "fedavg_lr" and c.data_path.endswith("heart_disease_uci.csv")
    assert SurvivalConfig(dataset="heart", family="lr", method="fedavg_lr", heart_mode="full").cell == "fedavg_lr_full"
    with pytest.raises(ValueError):
        SurvivalConfig(dataset="heart", family="lr", method="fedavg_lr", include_race=True)
    with pytest.raises(ValueError):
        SurvivalConfig(dataset="heart", family="xgb", method="fedxgb_bagging", task="survival")


# ── runs ─────────────────────────────────────────────────────────────────────
@needs_heart
def test_switzerland_is_excluded_only_from_local_rows(tmp_path):
    local = runner.run(_cfg(tmp_path / "l", "local_lr"), verbose=False)
    fed = runner.run(_cfg(tmp_path / "f", "fedavg_lr"), verbose=False)
    assert local["excluded_hospitals"] == ["Switzerland"] and local["patients"] == 740 - 46
    assert local["repeats"][0]["per_hospital"]["Switzerland"]["excluded"] is True
    assert fed["excluded_hospitals"] == [] and fed["patients"] == 740
    assert "excluded" not in fed["repeats"][0]["per_hospital"]["Switzerland"]
    first = lambda c: json.loads((Path(c.run_dir) / "repeat0" / "fold0" / "rounds.jsonl").read_text().splitlines()[0])
    assert "Switzerland" not in first(_cfg(tmp_path / "l", "local_lr"))["per_client_num_samples"]
    assert "Switzerland" in first(_cfg(tmp_path / "f", "fedavg_lr"))["per_client_num_samples"]


@needs_heart
def test_undefined_per_hospital_rank_metrics_are_dashed(tmp_path):
    s = runner.run(_cfg(tmp_path, "fedavg_lr"), verbose=False)
    swiss = s["repeats"][0]["per_hospital"]["Switzerland"]
    assert swiss["roc_auc"] is None and swiss["accuracy"] is not None      # 1 negative of 46 in paper mode
    assert s["repeats"][0]["per_hospital"]["Cleveland"]["roc_auc"] is not None


@needs_heart
def test_rounds_log_per_hospital_accuracy_and_logloss(tmp_path):
    cfg = _cfg(tmp_path, "fedavg_lr", rounds=2)
    runner.run(cfg, verbose=False)
    for line in (Path(cfg.run_dir) / "repeat0" / "fold0" / "rounds.jsonl").read_text().splitlines():
        hm = json.loads(line)["hospital_metrics"]
        assert list(hm) == list(HEART_HOSPITALS)
        for vals in hm.values():
            assert set(vals) == {"train_acc", "train_logloss", "test_acc", "test_logloss"}
            assert 0.0 <= vals["train_acc"] <= 1.0 and vals["test_logloss"] >= 0.0


@needs_heart
def test_heart_runs_are_deterministic(tmp_path):
    outs = []
    for sub in ("a", "b"):
        cfg = _cfg(tmp_path / sub, "fedavg_lr", rounds=2)
        runner.run(cfg, verbose=False)
        d = Path(cfg.run_dir)
        outs.append(([strip_timings(json.loads(l)) for p in sorted(d.rglob("rounds.jsonl"))
                      for l in p.read_text().splitlines()],
                     (d / "repeat0" / "oof.jsonl").read_text(), (d / "summary.json").read_text()))
    assert outs[0] == outs[1]


@needs_heart
def test_dp_total_steps_guard_on_heart(monkeypatch):
    """Same spy as the TCGA test: sigma solved once per hospital over rounds x epochs x ceil(n / batch)."""
    import opacus.accountants.utils as ou
    calls = []
    real = ou.get_noise_multiplier
    monkeypatch.setattr(ou, "get_noise_multiplier", lambda **kw: (calls.append(kw), real(**kw))[1])
    cohort = H.load_cohort()
    X, _ = H.encode_features(cohort.features, H.load_vocab(), "paper")
    tr, te = D.make_folds(cohort, 0)[0]
    fd = D.fold_data(cohort, X, tr, te)
    cfg = SurvivalConfig(dataset="heart", family="lr", method="dp_fedavg_lr", epsilon=4.0, rounds=3)
    outs = list(fl_lr.run_fold(cfg, fd, repeat_seed=0, fold_index=0))
    assert len(calls) == len(HEART_HOSPITALS)
    for kw, h in zip(calls, HEART_HOSPITALS):
        n = int((fd.hosp_train == h).sum())
        assert kw["steps"] == cfg.rounds * cfg.local_epochs * math.ceil(n / cfg.batch_size)
    for rep in outs[-1].privacy.values():
        assert rep["realized_eps"] <= 1.05 * 4.0 and rep["steps_taken"] == rep["steps"]
