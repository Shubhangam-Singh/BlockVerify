"""Smoke + determinism tests for the FL loop.

Determinism is asserted on rounds.jsonl with wall-clock timing keys removed
(CLAUDE.md amendment A1) — timings can never be byte-identical across runs.
"""
import json
import math
import os

from fedverify.config import FLConfig
from fedverify.core.runner import run, strip_timings, is_timing_key

SMOKE = dict(dataset="mnist", num_clients=2, rounds=2, local_epochs=1,
             batch_size=32, alpha=math.inf, seed=0, exp="test")


def _run(tmp_path, **kw):
    cfg = FLConfig(out_dir=str(tmp_path), **{**SMOKE, **kw})
    out = run(cfg, limit_train=200, limit_test=200, progress=False)
    return cfg, out


def test_two_round_smoke_produces_expected_artifacts(tmp_path):
    cfg, out = _run(tmp_path)
    assert len(out["rounds"]) == 2
    for f in ("rounds.jsonl", "config.json", "final_model.pt"):
        assert os.path.exists(os.path.join(cfg.run_dir, f)), f"missing {f}"

    rec = out["rounds"][0]
    for k in ("round", "test_acc", "test_loss", "macro_f1", "mean_train_loss",
              "per_client_num_samples", "diag", "train_wall_s", "round_wall_s"):
        assert k in rec, f"rounds.jsonl missing {k}"
    assert 0.0 <= rec["test_acc"] <= 1.0
    assert rec["diag"]["accepted"] == [0, 1] and rec["diag"]["rejected"] == []
    assert sum(rec["per_client_num_samples"].values()) == 200


def test_config_json_records_every_parameter(tmp_path):
    cfg, _ = _run(tmp_path)
    saved = json.load(open(os.path.join(cfg.run_dir, "config.json")))["config"]
    for field in FLConfig.__dataclass_fields__:
        assert field in saved, f"config.json must record {field}"
    assert saved["alpha"] == "inf"          # inf serialised JSON-safely
    assert saved["seed"] == 0


def test_same_seed_twice_is_byte_identical_modulo_timings(tmp_path):
    a, _ = _run(tmp_path / "a")
    b, _ = _run(tmp_path / "b")
    la = [strip_timings(json.loads(l)) for l in open(os.path.join(a.run_dir, "rounds.jsonl"))]
    lb = [strip_timings(json.loads(l)) for l in open(os.path.join(b.run_dir, "rounds.jsonl"))]
    assert la == lb, "same seed must reproduce identical rounds (excluding timings)"
    # and the canonical serialisation must match byte-for-byte too
    assert [json.dumps(r, sort_keys=True) for r in la] == \
           [json.dumps(r, sort_keys=True) for r in lb]


def test_different_seeds_differ(tmp_path):
    a, _ = _run(tmp_path / "a", seed=0)
    b, _ = _run(tmp_path / "b", seed=1)
    la = [strip_timings(json.loads(l)) for l in open(os.path.join(a.run_dir, "rounds.jsonl"))]
    lb = [strip_timings(json.loads(l)) for l in open(os.path.join(b.run_dir, "rounds.jsonl"))]
    assert la != lb, "different seeds should not produce identical trajectories"


def test_timing_keys_are_the_ones_excluded():
    assert is_timing_key("train_wall_s") and is_timing_key("round_wall_s")
    assert is_timing_key("anchor_ms") and is_timing_key("merkle_ms")
    assert not is_timing_key("test_acc") and not is_timing_key("round")
    assert strip_timings({"a": 1, "round_wall_s": 2, "d": {"anchor_ms": 3, "k": 4}}) == \
           {"a": 1, "d": {"k": 4}}


def test_partition_report_saved_and_non_iid_visible(tmp_path):
    cfg = FLConfig(out_dir=str(tmp_path), **{**SMOKE, "alpha": 0.5, "num_clients": 2})
    run(cfg, limit_train=400, limit_test=100, progress=False)
    rep = json.load(open(os.path.join(cfg.run_dir, "config.json")))["partition_report"]
    assert len(rep["counts"]) == 2 and rep["min_client_size"] >= 10
    assert "mean_kl_to_uniform" in rep
