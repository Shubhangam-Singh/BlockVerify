"""FedVerify-Survival Phase 2: DP-SGD for the logistic family (prompt Phase 2).

Load-bearing: sigma is solved ONCE per hospital over the TOTAL steps of the whole run (the spy test), epsilon = inf is
exactly the Phase-1 fedavg_lr run and never touches Opacus, and random-forest DP is refused as n/a, never faked.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lifelines")
pytest.importorskip("sklearn")
pytest.importorskip("opacus")

from fedverify.core.runner import strip_timings  # noqa: E402
from fedverify.survival import data as D  # noqa: E402
from fedverify.survival import fl_lr, runner  # noqa: E402
from fedverify.survival.config import HOSPITALS, SurvivalConfig  # noqa: E402

needs_data = pytest.mark.skipif(not D.DEFAULT_DATA.exists(),
                                reason="TCGA-BRCA workbook (gitignored) is not at fedverify/data/tcga_brca/")

# The Phase-1 round record, key for key and in order. Later phases must not change or reorder it (cf. A9). The ONE
# addition is Revision 2 §2.4's per-hospital train/test accuracy and log-loss, placed right after "privacy".
PHASE1_RECORD_KEYS = ["round", "model_family", "method", "repeat", "fold", "test_c_index", "test_roc_auc",
                      "test_f1", "test_n", "test_deaths", "mean_train_loss", "per_client_num_samples", "diag",
                      "privacy", "train_wall_s", "round_wall_s"]
REVISION2_KEYS = ["hospital_metrics"]


def _phase1_view(rec: dict) -> list:
    """The record's keys with the Revision-2 additions removed; asserts they sit right after 'privacy'."""
    keys = list(rec)
    assert keys[keys.index("privacy") + 1: keys.index("privacy") + 1 + len(REVISION2_KEYS)] == REVISION2_KEYS
    return [k for k in keys if k not in REVISION2_KEYS]


@pytest.fixture(scope="module")
def fold0():
    cohort = D.load_cohort()
    X, _ = D.encode_features(cohort.features, D.load_vocab())
    tr, te = D.make_folds(cohort, 0)[0]
    return D.fold_data(cohort, X, tr, te)


def _dp(eps, rounds=2, **kw):
    return SurvivalConfig(family="lr", method="dp_fedavg_lr", epsilon=eps, rounds=rounds, **kw)


# ── configuration ────────────────────────────────────────────────────────────
def test_dp_method_defaults_and_cell_names():
    c = SurvivalConfig(family="lr", method="dp_fedavg_lr")
    assert c.epsilon == 4.0 and c.cell == "dp_fedavg_lr_eps4"          # main epsilon default 4
    assert (c.delta, c.max_grad_norm) == (1e-5, 1.0)                    # prompt defaults
    assert _dp(math.inf).cell == "dp_fedavg_lr_epsinf"
    assert _dp(1.0).cell == "dp_fedavg_lr_eps1"
    assert SurvivalConfig(family="lr", method="fedavg_lr").epsilon is None


def test_epsilon_is_refused_where_it_would_mean_nothing():
    with pytest.raises(ValueError):
        SurvivalConfig(family="lr", method="fedavg_lr", epsilon=4.0)    # the non-private baseline stays non-private
    with pytest.raises(ValueError):
        _dp(0.0)
    with pytest.raises(ValueError):
        _dp(-1.0)


def test_random_forest_dp_is_na_not_faked():
    for m in ("local_rf", "central_rf", "fed_rf"):
        with pytest.raises(ValueError, match="n/a"):
            SurvivalConfig(family="rf", method=m, epsilon=4.0)
    with pytest.raises(SystemExit):
        runner.build_config(["--family", "rf", "--method", "fed_rf", "--epsilon", "4"])


# ── accounting ───────────────────────────────────────────────────────────────
@needs_data
def test_noise_multiplier_is_solved_over_TOTAL_steps(fold0, monkeypatch):
    """sigma must be solved once per hospital over rounds * local_epochs * ceil(n/batch), not per round."""
    import opacus.accountants.utils as ou
    calls = []
    real = ou.get_noise_multiplier

    def spy(**kw):
        calls.append(kw)
        return real(**kw)
    monkeypatch.setattr(ou, "get_noise_multiplier", spy)

    cfg = _dp(4.0, rounds=3)
    outs = list(fl_lr.run_fold(cfg, fold0, repeat_seed=0, fold_index=0))
    assert len(outs) == 3
    assert len(calls) == len(HOSPITALS), "sigma must be solved once per hospital for the whole run"
    for kw, h in zip(calls, HOSPITALS):
        n = int((fold0.hosp_train == h).sum())
        per_round = cfg.local_epochs * math.ceil(n / cfg.batch_size)
        assert kw["steps"] == cfg.rounds * per_round, f"{h}: sigma must cover ALL rounds"
        assert kw["steps"] != per_round, "per-round accounting is the bug"
        assert kw["sample_rate"] == pytest.approx(min(1.0, cfg.batch_size / n))
        assert kw["target_epsilon"] == 4.0 and kw["target_delta"] == 1e-5 and kw["accountant"] == "rdp"


@needs_data
def test_per_hospital_sigma_and_realised_epsilon_are_logged(fold0):
    cfg = _dp(4.0, rounds=3)
    outs = list(fl_lr.run_fold(cfg, fold0, repeat_seed=0, fold_index=0))
    previous = {}
    for o in outs:
        assert set(o.privacy) == set(HOSPITALS)
        for h, rep in o.privacy.items():
            assert rep["sigma"] > 0 and rep["C"] == 1.0 and rep["delta"] == 1e-5 and rep["target_eps"] == 4.0
            assert rep["realized_eps"] <= 1.05 * rep["target_eps"], f"{h} overspent"
            assert rep["steps_taken"] <= rep["steps"]
            assert rep["realized_eps"] > previous.get(h, 0.0), "one accountant per hospital, growing every round"
            previous[h] = rep["realized_eps"]
    last = outs[-1].privacy
    for h in HOSPITALS:
        assert last[h]["steps_taken"] == last[h]["steps"], "the accountant must persist across rounds"
        assert last[h]["realized_eps"] >= 0.5 * last[h]["target_eps"], "realised epsilon should be near target"
    assert len({last[h]["sigma"] for h in HOSPITALS}) > 1, "hospitals of different size get different sigma"


def test_smaller_epsilon_means_more_noise():
    from fedverify.core.privacy import plan_privacy
    sigma = {e: plan_privacy(_dp(e, rounds=30), 0, 116).sigma for e in (1.0, 2.0, 4.0, 8.0)}
    assert sigma[1.0] > sigma[2.0] > sigma[4.0] > sigma[8.0]


def test_cached_accountant_gives_opacus_numbers():
    from opacus.accountants import RDPAccountant
    from fedverify.survival.dp import CachedRDPAccountant
    plain, cached = RDPAccountant(), CachedRDPAccountant()
    for _ in range(90):
        plain.step(noise_multiplier=7.9, sample_rate=0.25)
        cached.step(noise_multiplier=7.9, sample_rate=0.25)
    assert cached.history == plain.history
    assert cached.get_epsilon(delta=1e-5) == plain.get_epsilon(delta=1e-5)
    assert cached.get_epsilon(delta=1e-5) == cached.get_epsilon(delta=1e-5)


def test_plan_cache_shares_sigma_but_never_the_accountant(monkeypatch):
    import opacus.accountants.utils as ou
    from fedverify.survival.dp import hospital_plan
    calls = []
    real = ou.get_noise_multiplier
    monkeypatch.setattr(ou, "get_noise_multiplier", lambda **kw: (calls.append(kw), real(**kw))[1])
    cache, cfg = {}, _dp(4.0, rounds=30)
    a = hospital_plan(cfg, 0, 116, cache)
    b = hospital_plan(cfg, 3, 116, cache)                          # same size, other hospital / fold
    c = hospital_plan(cfg, 0, 61, cache)
    assert len(calls) == 2                                          # one solve per distinct hospital size
    assert a.sigma == b.sigma and a.total_steps == b.total_steps and c.sigma != a.sigma
    assert a.accountant is not b.accountant and (a.client_id, b.client_id) == (0, 3)


@needs_data
def test_dp_runs_are_deterministic(fold0):
    cache = {}
    a = list(fl_lr.run_fold(_dp(2.0), fold0, 0, 0))                    # no cache: fresh solves
    b = list(fl_lr.run_fold(_dp(2.0), fold0, 0, 0, plan_cache=cache))  # cached solves give the same run
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.test_risk, y.test_risk)
        assert x.privacy == y.privacy
    c = list(fl_lr.run_fold(_dp(2.0), fold0, 1, 0, plan_cache=cache))  # another repeat seed -> other noise
    assert not np.array_equal(a[-1].test_risk, c[-1].test_risk)


@needs_data
def test_dp_noise_actually_changes_the_model(fold0):
    noisy = list(fl_lr.run_fold(_dp(1.0), fold0, 0, 0))[-1]
    clean = list(fl_lr.run_fold(SurvivalConfig(family="lr", method="fedavg_lr", rounds=2), fold0, 0, 0))[-1]
    assert not np.allclose(noisy.test_risk, clean.test_risk)


# ── epsilon = inf is Phase 1, exactly ────────────────────────────────────────
@needs_data
def test_eps_inf_matches_phase1_fedavg_exactly_and_never_touches_opacus(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("Opacus must not be involved when epsilon is inf")
    monkeypatch.setattr(fl_lr, "make_private", boom)

    base = SurvivalConfig(family="lr", method="fedavg_lr", rounds=2, repeats=1, out_dir=str(tmp_path / "p1"))
    inf = _dp(math.inf, repeats=1, out_dir=str(tmp_path / "inf"))
    s_base, s_inf = runner.run(base, verbose=False), runner.run(inf, verbose=False)
    assert s_base["repeats"][0]["pooled"] == s_inf["repeats"][0]["pooled"]
    assert s_base["repeats"][0]["per_hospital"] == s_inf["repeats"][0]["per_hospital"]
    assert (Path(base.run_dir) / "repeat0" / "oof.jsonl").read_text() == \
        (Path(inf.run_dir) / "repeat0" / "oof.jsonl").read_text()
    for k in range(5):
        rel = Path("repeat0") / f"fold{k}" / "rounds.jsonl"
        ra = [strip_timings(json.loads(l)) for l in (Path(base.run_dir) / rel).read_text().splitlines()]
        rb = [strip_timings(json.loads(l)) for l in (Path(inf.run_dir) / rel).read_text().splitlines()]
        assert len(ra) == len(rb) == 2
        for x, y in zip(ra, rb):
            assert x.pop("method") == "fedavg_lr" and y.pop("method") == "dp_fedavg_lr"
            assert x == y and y["privacy"] is None
    assert s_inf["max_realized_eps"] is None


@needs_data
def test_phase1_record_shape_is_unchanged_by_phase2(tmp_path):
    cfg = SurvivalConfig(family="lr", method="fedavg_lr", rounds=1, repeats=1, out_dir=str(tmp_path))
    runner.run(cfg, verbose=False)
    rec = json.loads((Path(cfg.run_dir) / "repeat0" / "fold0" / "rounds.jsonl").read_text().splitlines()[0])
    assert _phase1_view(rec) == PHASE1_RECORD_KEYS


@needs_data
def test_runner_writes_dp_cells_with_privacy_and_max_realised_epsilon(tmp_path):
    cfg = _dp(4.0, repeats=1, out_dir=str(tmp_path))
    s = runner.run(cfg, verbose=False)
    assert Path(cfg.run_dir).parent.name == "dp_fedavg_lr_eps4"
    assert s["epsilon"] == 4.0
    assert 0 < s["max_realized_eps"] <= 1.05 * 4.0
    assert s["repeats"][0]["max_realized_eps"] == s["max_realized_eps"]
    rec = json.loads((Path(cfg.run_dir) / "repeat0" / "fold0" / "rounds.jsonl").read_text().splitlines()[0])
    assert _phase1_view(rec) == PHASE1_RECORD_KEYS                  # same keys; privacy is now filled in
    assert set(rec["privacy"]) == set(HOSPITALS)
    c = json.loads((Path(cfg.run_dir) / "config.json").read_text())
    assert c["config"]["epsilon"] == 4.0 and c["config"]["delta"] == 1e-5 and c["config"]["max_grad_norm"] == 1.0
