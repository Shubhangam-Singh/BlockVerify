"""P6: experiments write complete, deterministic runs and the tables come only from them."""
import json
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("skimage")

from photoshield.analysis import make_tables  # noqa: E402
from photoshield.experiments import study  # noqa: E402


def _strip(path):
    out = []
    for line in open(path):
        r = json.loads(line)
        out.append({k: v for k, v in r.items() if not (k.endswith("_ms") or k.endswith("_wall_s"))})
    return out


@pytest.fixture(autouse=True)
def _restore_results():
    saved = study.RESULTS
    yield
    study.RESULTS = saved


def test_roc_thresholds_are_cumulative():
    roc = study._roc_thresholds([0, 1, 2, 10], [5, 30, 31, 40])
    assert roc[0] == {"t": 0, "tpr": 0.25, "fpr": 0.0}
    assert roc[5]["fpr"] == 0.25 and roc[10]["tpr"] == 1.0
    assert roc[64] == {"t": 64, "tpr": 1.0, "fpr": 1.0}


def test_e6_is_deterministic_and_installs_nothing_without_flag(tmp_path):
    before = open(study.CALIBRATION).read() if os.path.exists(study.CALIBRATION) else None
    for run in ("a", "b"):
        study.main(["e6", "--limit", "3", "--results", str(tmp_path / run)])
    pa = tmp_path / "a" / "e6" / "calibration_skimage" / "seed0"
    pb = tmp_path / "b" / "e6" / "calibration_skimage" / "seed0"
    assert (pa / "records.jsonl").read_bytes() == (pb / "records.jsonl").read_bytes()
    s = json.loads((pa / "summary.json").read_text())["summary"]
    assert 0 <= s["tau_near"] <= s["tau_far"] <= 64 and s["fpr_near"] <= 0.01
    cfg = json.loads((pa / "config.json").read_text())
    assert {"seed", "threads", "torch_threads_effective", "transforms", "fpr_near"} <= set(cfg)
    after = open(study.CALIBRATION).read() if os.path.exists(study.CALIBRATION) else None
    assert before == after


def test_e1_tiny_is_byte_identical_after_stripping_timings(tmp_path):
    args = ["e1", "--model", "tiny", "--no-pretrained", "--eps", "4", "--steps", "2", "--size", "64",
            "--limit", "2", "--threads", "1"]
    for run in ("a", "b"):
        study.main(args + ["--results", str(tmp_path / run)])
    cell = "tiny_eps4_steps2_size64_eotresize-blur-jpeg"
    ra = _strip(tmp_path / "a" / "e1" / cell / "seed0" / "records.jsonl")
    rb = _strip(tmp_path / "b" / "e1" / cell / "seed0" / "records.jsonl")
    assert ra == rb and len(ra) == 2
    assert all(r["linf"] <= 4.0 + 1e-9 for r in ra)


def test_e7_album_note_is_constant_size(tmp_path):
    study.main(["e7", "--trials", "2", "--results", str(tmp_path)])
    recs = [json.loads(l) for l in open(tmp_path / "e7" / "chain_cost" / "seed0" / "records.jsonl")]
    album = {r["n"]: r for r in recs if r["kind"] == "album"}
    assert set(album) == {1, 10, 100}
    assert max(r["note_bytes"] for r in album.values()) <= 1024
    assert album[100]["proof_hashes_max"] == 7
    assert all(r["accepted"] == r["n"] for r in recs if r["kind"] == "rebuild")


def test_tables_render_missing_cells_as_dash(tmp_path, capsys):
    md = make_tables.render(str(tmp_path))
    assert "_No E1 runs._" in md and "_No E7 runs._" in md
    assert any("E1" in m for m in make_tables.MISSING)


def test_tables_skip_runs_without_records(tmp_path, capsys):
    d = tmp_path / "e1" / "x" / "seed0"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    make_tables.render(str(tmp_path))
    assert "[skip] incomplete run" in capsys.readouterr().err


def test_tables_numbers_come_from_files(tmp_path):
    study.main(["e7", "--trials", "2", "--results", str(tmp_path)])
    md = make_tables.render(str(tmp_path))
    recs = [json.loads(l) for l in open(tmp_path / "e7" / "chain_cost" / "seed0" / "records.jsonl")]
    nb = max(r["note_bytes"] for r in recs if r["kind"] == "single")
    assert f"| mock | 2 | {nb} |" in md
