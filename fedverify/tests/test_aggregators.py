"""Phase-4 aggregator tests.

The load-bearing ones are the forensics pair: it must reject a client whose delta is
all-NaN, and it must reject NOBODY when every client agrees — a defence that fires on
honest rounds costs more accuracy than the attack it prevents.

`tau` is never allowed to have a default. A hardcoded threshold is the exact bug
docs/EVALUATION.md §5.2 documents one level down (the layer threshold z>8 had FPR 0.49
on real weights), so constructing forensics without a calibrated tau must fail loudly.
"""
import json
import math
import os

import pytest
import torch

from fedverify.aggregators import AGGREGATORS, build_aggregator
from fedverify.config import FLConfig

D = 2048


class U:
    def __init__(self, i, delta, n=100):
        self.client_id, self.delta, self.num_samples = i, delta, n


def honest(i, scale=0.01, shift=0.05):
    return torch.randn(D, generator=torch.Generator().manual_seed(i)) * scale + shift


def cfg(**kw):
    return FLConfig(**{**dict(num_clients=10, aggregator="fedavg", tau=8.0), **kw})


# ── registry ─────────────────────────────────────────────────────────────────
def test_registry_has_every_phase4_rule():
    assert set(AGGREGATORS) == {"fedavg", "krum", "multikrum", "trimmed_mean",
                                "median", "forensics"}


def test_unknown_aggregator_rejected():
    with pytest.raises(ValueError, match="unknown aggregator"):
        build_aggregator("nope")


def test_only_decision_rules_are_marked_detectors():
    """Krum's top-1 selection is not a malice judgement; scoring it as one is a category error."""
    assert AGGREGATORS["forensics"].detects and AGGREGATORS["multikrum"].detects
    assert not AGGREGATORS["krum"].detects
    assert not AGGREGATORS["median"].detects and not AGGREGATORS["trimmed_mean"].detects


# ── agreement case ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["median", "trimmed_mean", "krum", "multikrum", "forensics"])
def test_equals_fedavg_on_identical_deltas(name):
    same = honest(0)
    ups = [U(i, same.clone()) for i in range(10)]
    fed, _ = build_aggregator("fedavg").aggregate(ups, cfg(), 1)
    out, _ = build_aggregator(name).aggregate(ups, cfg(aggregator=name), 1)
    assert torch.allclose(out.double(), fed.double(), atol=1e-9)


def test_forensics_rejects_nobody_when_all_clients_agree():
    ups = [U(i, honest(0).clone()) for i in range(10)]
    _out, diag = build_aggregator("forensics").aggregate(ups, cfg(aggregator="forensics"), 1)
    assert diag["rejected"] == []
    assert diag["accepted"] == list(range(10))


def test_forensics_rejects_nobody_on_ten_merely_similar_clients():
    ups = [U(i, honest(i)) for i in range(10)]
    _out, diag = build_aggregator("forensics").aggregate(ups, cfg(aggregator="forensics"), 1)
    assert diag["rejected"] == []


# ── outlier case ─────────────────────────────────────────────────────────────
def _one_outlier():
    return [U(i, honest(i)) for i in range(9)] + [U(9, honest(9) * 500.0)]


def test_krum_excludes_the_obvious_outlier():
    _out, diag = build_aggregator("krum").aggregate(
        _one_outlier(), cfg(aggregator="krum", attacker_frac=0.1), 1)
    assert 9 in diag["rejected"]
    assert diag["accepted"] == [int(diag["accepted"][0])] and 9 not in diag["accepted"]


def test_multikrum_excludes_exactly_the_outlier():
    _out, diag = build_aggregator("multikrum").aggregate(
        _one_outlier(), cfg(aggregator="multikrum", attacker_frac=0.1), 1)
    assert diag["rejected"] == [9]


def test_forensics_excludes_the_obvious_outlier():
    _out, diag = build_aggregator("forensics").aggregate(
        _one_outlier(), cfg(aggregator="forensics"), 1)
    assert diag["rejected"] == [9]


def test_forensics_recovers_the_honest_mean_under_a_scaling_attack():
    ups = [U(i, honest(i)) for i in range(7)] + [U(i, honest(i) * 26.6) for i in (7, 8, 9)]
    out, diag = build_aggregator("forensics").aggregate(ups, cfg(aggregator="forensics"), 1)
    clean = torch.stack([honest(i) for i in range(7)]).mean(0)
    assert diag["rejected"] == [7, 8, 9]
    assert torch.allclose(out.double(), clean.double(), atol=1e-6)


def test_forensics_catches_sign_flip_by_direction_not_norm():
    """A sign-flipped delta has the SAME norm, so s_norm is blind; s_dir must fire."""
    ups = [U(i, honest(i)) for i in range(7)] + [U(i, -honest(i)) for i in (7, 8, 9)]
    _out, diag = build_aggregator("forensics").aggregate(ups, cfg(aggregator="forensics"), 1)
    assert diag["rejected"] == [7, 8, 9]
    s = diag["scores"]["7"]
    assert s["top_probe"] == "s_dir" and s["s_dir"] > s["s_norm"]


# ── health flags ─────────────────────────────────────────────────────────────
def test_forensics_rejects_an_all_nan_delta():
    ups = [U(i, honest(i)) for i in range(9)] + [U(9, torch.full((D,), float("nan")))]
    out, diag = build_aggregator("forensics").aggregate(ups, cfg(aggregator="forensics"), 1)
    assert diag["rejected"] == [9]
    assert diag["scores"]["9"]["rejected_by"].startswith("health:nan")
    assert bool(torch.isfinite(out).all())          # NaN must not leak into the model


def test_forensics_rejects_an_inf_delta():
    bad = honest(9).clone(); bad[5] = float("inf")
    ups = [U(i, honest(i)) for i in range(9)] + [U(9, bad)]
    _out, diag = build_aggregator("forensics").aggregate(ups, cfg(aggregator="forensics"), 1)
    assert diag["rejected"] == [9]


def test_forensics_rejects_extreme_magnitude():
    bad = honest(9).clone(); bad[7] = 1e6
    ups = [U(i, honest(i)) for i in range(9)] + [U(9, bad)]
    _out, diag = build_aggregator("forensics").aggregate(ups, cfg(aggregator="forensics"), 1)
    assert "extreme_magnitude" in diag["scores"]["9"]["health_flags"]


def test_constant_run_flag_ignores_short_vectors():
    """A 16-element delta is trivially 'one run'; the flag only applies above 1024."""
    ups = [U(i, torch.full((16,), 1.0)) for i in range(10)]
    _out, diag = build_aggregator("forensics").aggregate(ups, cfg(aggregator="forensics"), 1)
    assert diag["rejected"] == []


def test_all_clients_rejected_yields_a_zero_update_not_a_crash():
    ups = [U(i, torch.full((D,), float("nan"))) for i in range(4)]
    out, diag = build_aggregator("forensics").aggregate(ups, cfg(aggregator="forensics"), 1)
    assert diag["accepted"] == [] and "fallback" in diag
    assert torch.equal(out, torch.zeros(D))


# ── tau provenance ───────────────────────────────────────────────────────────
def test_forensics_without_tau_raises():
    ups = [U(i, honest(i)) for i in range(10)]
    with pytest.raises(ValueError, match="requires cfg.tau"):
        build_aggregator("forensics").aggregate(ups, FLConfig(num_clients=10), 1)


def test_tau_is_read_from_taus_json_and_recorded_in_config(tmp_path):
    """tau must be traceable to calibration output, not invented at the call site."""
    from fedverify.analysis.calibrate import load_tau
    from fedverify.core.runner import run

    cal = tmp_path / "calibration"
    cal.mkdir()
    (cal / "taus.json").write_text(json.dumps({"by_key": {
        "eps=inf|K=3": {"combined": {"auc": 0.9, "tau_youden": 5.5, "tau_fpr5": 7.25}}}}))

    tau, src = load_tau(str(tmp_path), float("inf"), 3)
    assert tau == 7.25 and src.endswith("eps=inf|K=3:tau_fpr5")

    c = FLConfig(dataset="mnist", num_clients=3, rounds=1, alpha=math.inf, seed=0,
                 aggregator="forensics", tau=tau, tau_source=src,
                 exp="test", out_dir=str(tmp_path / "out"))
    run(c, limit_train=120, limit_test=100, progress=False)
    saved = json.load(open(os.path.join(c.run_dir, "config.json")))["config"]
    assert saved["tau"] == 7.25
    assert saved["tau_source"] == src
    assert saved["tau_coord"] == c.tau_coord


def test_load_tau_without_calibration_raises(tmp_path):
    from fedverify.analysis.calibrate import load_tau
    with pytest.raises(FileNotFoundError, match="taus.json"):
        load_tau(str(tmp_path), float("inf"), 10)


# ── trimmed mean / median specifics ──────────────────────────────────────────
def test_trimmed_mean_trims_from_both_ends():
    ups = [U(0, torch.tensor([-100.0])), U(4, torch.tensor([100.0]))] + \
          [U(i, torch.tensor([1.0])) for i in range(1, 4)]
    out, diag = build_aggregator("trimmed_mean").aggregate(
        ups, cfg(aggregator="trimmed_mean", num_clients=5, trim_beta=0.2), 1)
    assert diag["beta"] == 1
    assert torch.allclose(out, torch.tensor([1.0]))


def test_trimmed_mean_always_leaves_at_least_one_client():
    ups = [U(i, torch.tensor([float(i)])) for i in range(3)]
    _out, diag = build_aggregator("trimmed_mean").aggregate(
        ups, cfg(aggregator="trimmed_mean", num_clients=3, trim_beta=0.9), 1)
    assert diag["beta"] == 1


def test_median_is_unmoved_by_a_minority_of_extremes():
    ups = [U(i, torch.tensor([1.0])) for i in range(7)] + \
          [U(i, torch.tensor([1e6])) for i in (7, 8, 9)]
    out, _ = build_aggregator("median").aggregate(ups, cfg(aggregator="median"), 1)
    assert torch.allclose(out, torch.tensor([1.0]))


@pytest.mark.parametrize("name", sorted(AGGREGATORS))
def test_empty_updates_rejected(name):
    with pytest.raises(ValueError, match="no client updates"):
        build_aggregator(name).aggregate([], cfg(aggregator=name), 1)


# ── regression guard ─────────────────────────────────────────────────────────
def test_phase1_record_shape_is_unchanged_by_phase4(tmp_path):
    """A no-attack fedavg cell must keep its exact Phase-1/2 record shape.

    Regression guard: adding a "detector" key to the shared _empty_diag() once changed the
    bytes of EVERY exp1 record mid-grid, which would have made Table-1 cells produced
    before and after Phase 4 non-comparable.
    """
    from fedverify.core.runner import run
    c = FLConfig(dataset="mnist", num_clients=3, rounds=1, alpha=math.inf, seed=0,
                 exp="test", out_dir=str(tmp_path))
    run(c, limit_train=150, limit_test=100, progress=False)
    rec = json.loads(open(os.path.join(c.run_dir, "rounds.jsonl")).readline())
    assert set(rec) == {"round", "test_acc", "test_loss", "macro_f1", "mean_train_loss",
                        "per_client_num_samples", "diag", "privacy",
                        "train_wall_s", "round_wall_s"}
    assert set(rec["diag"]) == {"accepted", "rejected", "scores"}
    assert "attack" not in rec and "commit" not in rec
