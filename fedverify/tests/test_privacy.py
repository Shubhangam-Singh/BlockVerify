"""Phase-2 DP tests.

The load-bearing one is `test_noise_multiplier_is_solved_over_TOTAL_steps`: it spies on
get_noise_multiplier to prove sigma is solved over rounds*steps_per_round, not one
round's steps (the classic DP-FL accounting bug — see core/privacy.py).
"""
import json
import math
import os

import pytest
import torch

from fedverify.config import FLConfig
from fedverify.core import client as client_mod
from fedverify.core.privacy import (dp_enabled, plan_privacy, privacy_report,
                                    steps_per_round, total_steps, validate_dp_compatible)
from fedverify.core.runner import run, strip_timings

TINY = dict(dataset="mnist", num_clients=2, rounds=2, local_epochs=1,
            batch_size=32, alpha=math.inf, seed=0, exp="test")


def _run(tmp_path, **kw):
    cfg = FLConfig(out_dir=str(tmp_path), **{**TINY, **kw})
    out = run(cfg, limit_train=200, limit_test=200, progress=False)
    return cfg, out


def _rounds(cfg):
    p = os.path.join(cfg.run_dir, "rounds.jsonl")
    return [strip_timings(json.loads(l)) for l in open(p)]


# ── DP off ──────────────────────────────────────────────────────────────────
def test_dp_disabled_for_none_and_inf():
    cfg = FLConfig(dataset="mnist", epsilon=None)
    assert not dp_enabled(cfg)
    assert not dp_enabled(cfg.replace(epsilon=math.inf))
    assert dp_enabled(cfg.replace(epsilon=1.0))
    assert plan_privacy(cfg, 0, 100) is None
    assert privacy_report(None) is None


def test_eps_inf_is_identical_to_phase1_and_never_touches_opacus(tmp_path, monkeypatch):
    """epsilon=inf must reproduce the Phase-1 (epsilon=None) run exactly, with no Opacus."""
    def boom(*a, **k):
        raise AssertionError("Opacus must not be involved when epsilon is None/inf")
    monkeypatch.setattr(client_mod, "make_private", boom)

    base, _ = _run(tmp_path / "none", epsilon=None)     # the Phase-1 code path
    inf, _ = _run(tmp_path / "inf", epsilon=math.inf)

    assert _rounds(base) == _rounds(inf)
    for rec in _rounds(inf):
        assert rec["privacy"] is None


# ── accounting ──────────────────────────────────────────────────────────────
def test_step_formulas():
    assert steps_per_round(6000, 64, 1) == math.ceil(6000 / 64) == 94
    assert steps_per_round(6000, 64, 2) == 188
    assert total_steps(6000, 64, 1, 30) == 30 * 94


def test_noise_multiplier_is_solved_over_TOTAL_steps(monkeypatch):
    """sigma must be solved once over rounds*steps_per_round, NOT one round."""
    import opacus.accountants.utils as ou
    seen = {}
    real = ou.get_noise_multiplier

    def spy(**kw):
        seen.update(kw)
        return real(**kw)
    monkeypatch.setattr(ou, "get_noise_multiplier", spy)

    cfg = FLConfig(dataset="mnist", num_clients=10, rounds=30, local_epochs=1,
                   batch_size=64, epsilon=2.0, delta=1e-5)
    n_k = 6000
    plan = plan_privacy(cfg, 0, n_k)

    expected = total_steps(n_k, cfg.batch_size, cfg.local_epochs, cfg.rounds)
    assert seen["steps"] == expected == 2820, "sigma must cover ALL rounds"
    assert seen["steps"] != plan.steps_per_round, "per-round accounting is the bug"
    assert seen["sample_rate"] == pytest.approx(cfg.batch_size / n_k)
    assert seen["target_epsilon"] == 2.0 and seen["target_delta"] == 1e-5
    assert seen["accountant"] == "rdp"
    assert plan.total_steps == expected


def test_sigma_is_monotone_in_epsilon():
    cfg = FLConfig(dataset="mnist", rounds=5, batch_size=64, epsilon=1.0)
    sigmas = [plan_privacy(cfg.replace(epsilon=e), 0, 2000).sigma for e in (0.5, 1, 2, 4, 8)]
    assert sigmas == sorted(sigmas, reverse=True), "stronger privacy needs more noise"


def test_realized_epsilon_within_tolerance_of_target(tmp_path):
    cfg, out = _run(tmp_path, epsilon=2.0)
    reports = out["rounds"][-1]["privacy"]
    assert reports, "DP run must record a privacy report per client"
    for cid, r in reports.items():
        assert r["realized_eps"] <= 1.05 * r["target_eps"], f"client {cid} overspent"
        assert r["steps_taken"] <= r["steps"], "cannot take more steps than accounted"
        assert r["sigma"] > 0 and r["C"] == cfg.max_grad_norm
        for k in ("sigma", "C", "sample_rate", "steps", "realized_eps", "delta"):
            assert k in r


def test_accountant_persists_across_rounds(tmp_path):
    """ONE accountant must accumulate every round's steps.

    If it were re-created per round, steps_taken would equal a single round's steps.
    """
    _, out = _run(tmp_path, epsilon=4.0, rounds=3)
    r = out["rounds"][-1]["privacy"]["0"]
    assert r["steps_taken"] > r["steps_per_round"], "accountant reset between rounds"
    assert r["steps_taken"] == 3 * r["steps_per_round"] == r["steps"]


def test_more_rounds_means_more_noise_but_same_final_epsilon(tmp_path):
    """The signature of TOTAL-step accounting.

    Because sigma is solved over rounds*steps_per_round, a longer run gets MORE noise
    and still lands at ~the target epsilon. (Under the per-round accounting bug, a
    longer run would instead overspend privacy.)
    """
    _, short = _run(tmp_path / "s", epsilon=4.0, rounds=2)
    _, long_ = _run(tmp_path / "l", epsilon=4.0, rounds=4)
    a = short["rounds"][-1]["privacy"]["0"]
    b = long_["rounds"][-1]["privacy"]["0"]
    assert b["sigma"] > a["sigma"], "more steps must be covered by more noise"
    assert b["steps"] == 2 * a["steps"]
    for r in (a, b):
        assert r["realized_eps"] <= 1.05 * r["target_eps"]
    assert abs(a["realized_eps"] - b["realized_eps"]) < 0.5 * a["target_eps"]


def test_epsilon_grows_monotonically_within_a_run(tmp_path):
    _, out = _run(tmp_path, epsilon=4.0, rounds=3)
    eps = [r["privacy"]["0"]["realized_eps"] for r in out["rounds"]]
    assert eps == sorted(eps) and eps[0] < eps[-1]


# ── validation ──────────────────────────────────────────────────────────────
def test_batchnorm_model_raises():
    bad = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3), torch.nn.BatchNorm2d(4),
                              torch.nn.Flatten(), torch.nn.Linear(4 * 26 * 26, 2))
    with pytest.raises(ValueError) as e:
        validate_dp_compatible(bad)
    assert "BatchNorm" in str(e.value) or "batch" in str(e.value).lower()


def test_groupnorm_model_passes():
    from fedverify.core.models import SmallCNN
    validate_dp_compatible(SmallCNN())        # must not raise


def test_dp_run_is_deterministic(tmp_path):
    a, _ = _run(tmp_path / "a", epsilon=2.0)
    b, _ = _run(tmp_path / "b", epsilon=2.0)
    assert _rounds(a) == _rounds(b), "DP runs must be reproducible for a fixed seed"
