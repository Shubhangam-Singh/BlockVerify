"""FedVerify-Survival runner: results layout, record keys, pooling, and same seed -> identical output (rule 6)."""
import json
from pathlib import Path

import pytest

pytest.importorskip("lifelines")
pytest.importorskip("sklearn")

from fedverify.core.runner import strip_timings  # noqa: E402
from fedverify.survival import data as D  # noqa: E402
from fedverify.survival import runner  # noqa: E402
from fedverify.survival.config import HOSPITAL_INDEX, HOSPITALS, SurvivalConfig  # noqa: E402

needs_data = pytest.mark.skipif(not D.DEFAULT_DATA.exists(),
                                reason="TCGA-BRCA workbook (gitignored) is not at fedverify/data/tcga_brca/")

# keys every existing FedVerify round record carries that Survival keeps (INVENTORY.md M02)
EXISTING_KEYS = {"round", "mean_train_loss", "per_client_num_samples", "diag", "privacy", "train_wall_s",
                 "round_wall_s"}


def _cfg(out, family="lr", method="fedavg_lr", **kw):
    return SurvivalConfig(family=family, method=method, rounds=2, repeats=1, out_dir=str(out), **kw)


def _outputs(run_dir: Path) -> dict:
    """Everything a run wrote that must be reproducible: rounds (timings stripped), out-of-fold rows, summary."""
    got = {}
    for p in sorted(run_dir.rglob("rounds.jsonl")):
        got[str(p.relative_to(run_dir))] = [strip_timings(json.loads(l)) for l in p.read_text().splitlines()]
    for p in sorted(run_dir.rglob("oof.jsonl")):
        got[str(p.relative_to(run_dir))] = p.read_text()
    got["summary.json"] = (run_dir / "summary.json").read_text()
    return got


@needs_data
@pytest.mark.parametrize("family,method", [("lr", "fedavg_lr"), ("rf", "fed_rf")])
def test_same_seed_twice_gives_identical_output(tmp_path, family, method):
    a = _cfg(tmp_path / "a", family, method)
    b = _cfg(tmp_path / "b", family, method)
    runner.run(a, verbose=False)
    runner.run(b, verbose=False)
    out_a, out_b = _outputs(Path(a.run_dir)), _outputs(Path(b.run_dir))
    assert len(out_a) == 5 + 1 + 1                                  # 5 fold rounds files, oof, summary
    assert out_a == out_b


@needs_data
def test_a_different_seed_changes_the_folds(tmp_path):
    s0 = runner.run(_cfg(tmp_path / "s0"), verbose=False)
    s1 = runner.run(_cfg(tmp_path / "s1", seed=1), verbose=False)
    assert s0["repeat_seeds"] == [0] and s1["repeat_seeds"] == [1]
    assert s0["repeats"][0]["pooled"] != s1["repeats"][0]["pooled"]


@needs_data
def test_records_keep_existing_keys_and_use_hospital_codes(tmp_path):
    cfg = _cfg(tmp_path)
    runner.run(cfg, verbose=False)
    lines = (Path(cfg.run_dir) / "repeat0" / "fold0" / "rounds.jsonl").read_text().splitlines()
    assert len(lines) == cfg.rounds                                 # one line per round
    rec = json.loads(lines[0])
    assert EXISTING_KEYS <= set(rec)
    assert rec["model_family"] == "lr" and rec["privacy"] is None
    assert set(rec["diag"]) >= {"accepted", "rejected", "scores"}
    assert rec["diag"]["accepted"] == list(HOSPITALS)
    assert {"test_c_index", "test_roc_auc", "test_f1"} <= set(rec)


@needs_data
def test_config_json_records_everything(tmp_path):
    cfg = _cfg(tmp_path)
    runner.run(cfg, verbose=False)
    c = json.loads((Path(cfg.run_dir) / "config.json").read_text())
    assert c["config"]["method"] == "fedavg_lr" and c["config"]["momentum"] == 0.0
    assert c["hospital_ids"] == HOSPITAL_INDEX and c["model_family"] == "lr"
    assert c["data"]["patients"] == 474 and c["data"]["deaths"] == 76
    assert c["data"]["dropped_no_usable_time"] == 0
    assert c["num_features"] == 30 and c["num_params"] == 31
    assert isinstance(c["torch_threads_effective"], int) and c["attacker_ids"] == []
    fold = json.loads((Path(cfg.run_dir) / "repeat0" / "fold0" / "config.json").read_text())
    assert fold["config"]["num_clients"] == 5 and fold["config"]["aggregator"] == "fedavg"


@needs_data
def test_out_of_fold_predictions_cover_every_patient_once(tmp_path):
    cfg = _cfg(tmp_path, family="rf", method="local_rf")
    summary = runner.run(cfg, verbose=False)
    rows = [json.loads(l) for l in (Path(cfg.run_dir) / "repeat0" / "oof.jsonl").read_text().splitlines()]
    assert len(rows) == 474 and len({r["patient"] for r in rows}) == 474
    assert sum(r["event"] for r in rows) == 76
    assert summary["patients"] == 474 and summary["deaths"] == 76
    a8 = summary["repeats"][0]["per_hospital"]["A8"]
    assert a8["deaths"] == 4 and a8["c_index"] is None and a8["roc_auc"] is None
    assert len(summary["repeats"][0]["pooled_by_round"]["c_index"]) == cfg.rounds


def test_cli_validates_family_and_method():
    with pytest.raises(SystemExit):
        runner.build_config(["--family", "rf", "--method", "fedavg_lr"])
    cfg, _ = runner.build_config(["--family", "lr", "--method", "fedavg", "--seed", "3", "--repeats", "2"])
    assert cfg.method == "fedavg_lr" and cfg.seed == 3 and cfg.repeats == 2 and cfg.rounds == 30
