"""Phase 10: the live demo, its run in the dashboard (with the existing runs unchanged), and DEMO.md."""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("opacus")

from fedverify.survival import heart as H  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "fedverify" / "results"
TAUS = RESULTS / "calibration" / "taus_survival.json"
sys.path.insert(0, str(REPO / "backend"))
needs_demo_inputs = pytest.mark.skipif(not H.DEFAULT_DATA.exists() or not TAUS.exists(),
                                       reason="heart CSV or calibration file (gitignored) not present")
MNIST_META_KEYS = {"dataset", "numClients", "rounds", "alpha", "epsilon", "aggregator", "attack", "attackerFrac", "tau",
                   "tauSource", "seed", "attackerIds", "numParams", "detector"}


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    if not H.DEFAULT_DATA.exists() or not TAUS.exists():
        pytest.skip("heart CSV or calibration file (gitignored) not present")
    out = tmp_path_factory.mktemp("demo")
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "fedverify.survival.demo", "--out-dir", str(out),
                           "--results-dir", str(RESULTS)], cwd=REPO, capture_output=True, text=True, timeout=60,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    run_dir = out / "demo" / "heart_ours_lr_demo" / "seed0"
    return {"proc": proc, "elapsed": time.perf_counter() - t0, "run_dir": run_dir,
            "summary": json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists()
            else None}


@needs_demo_inputs
def test_demo_finishes_under_60_seconds_and_exits_0(demo_run):
    assert demo_run["proc"].returncode == 0, demo_run["proc"].stderr[-2000:]
    assert demo_run["elapsed"] < 60


@needs_demo_inputs
def test_demo_output_shows_the_rejections_and_the_failed_tamper_check(demo_run):
    out, s = demo_run["proc"].stdout, demo_run["summary"]
    attacker = s["attacker"]
    assert re.search(rf"{re.escape(attacker)}\s+norm .* REJECTED by the screen: .* fired: s_\w+ = [\d.]+ > tau", out)
    assert "REJECTED at the ledger: the update in" in out and "invalid signature" in out           # wrong key
    assert "replay: the submission is signed for round 1, this is round 3" in out                  # double spend
    assert "VERIFIED, it equals the root on the ledger" in out
    assert "Tamper check FAILED" in out
    assert s["proof_verified"] and s["tamper_detected"] and s["ledger_valid"]
    assert all(not x["accepted"] for x in s["events"]["ledger"]) and len(s["events"]["ledger"]) == 2
    assert (demo_run["run_dir"] / "demo_output.txt").read_text().strip() == out.strip()            # the saved copy


@needs_demo_inputs
def test_demo_tau_comes_from_the_calibration_file(demo_run):
    from fedverify.survival.screening import cfg_tau_key, load_tau
    from fedverify.survival.config import SurvivalConfig
    c = json.loads((demo_run["run_dir"] / "config.json").read_text())["config"]
    key = cfg_tau_key(SurvivalConfig(dataset="heart", family="lr", method="ours_lr", rounds=3))
    tau, source = load_tau(TAUS, key, c["tau_policy"])
    assert c["tau"] == tau and c["tau_source"].endswith(source.split("taus_survival.json")[-1])
    assert "taus_survival.json" in c["tau_source"]


@needs_demo_inputs
def test_demo_run_loads_in_the_dashboard_and_existing_runs_still_load(demo_run, tmp_path, monkeypatch):
    import app as app_module
    import fl_routes
    from fedverify.chain import commitment as C
    from fedverify.chain import push as P
    store = tmp_path / "fl_runs.json"
    real = REPO / "backend" / "data" / "fl_runs.json"
    if real.exists():
        shutil.copy(real, store)                      # the developer's own runs, on a copy: never written to
    monkeypatch.setattr(fl_routes, "FL_FILE", str(store))
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()
    existing = json.loads(store.read_text()) if store.exists() else {}
    before = {rid: client.get(f"/api/fl/lineage/{rid}").get_json() for rid in existing}
    for rid, lin in before.items():                   # every existing run loads, with every round it holds
        assert lin["success"] and lin["numRounds"] == len(existing[rid]["rounds"]), rid

    def via_test_client(url, payload, timeout=20.0):
        resp = client.post("/api" + url.split("/api", 1)[1], json=payload)
        return resp.status_code, resp.get_json()
    monkeypatch.setattr(P, "_post", via_test_client)
    ok, fail, rid = P.push(str(demo_run["run_dir"]), "http://dashboard.test/api", verbose=False)
    assert (ok, fail) == (3, 0)                       # the server re-derived and accepted every round root

    runs = client.get("/api/fl/runs").get_json()["runs"]
    assert rid in [r["runId"] for r in runs] and set(existing) <= {r["runId"] for r in runs}
    lin = client.get(f"/api/fl/lineage/{rid}").get_json()
    s = demo_run["summary"]
    assert lin["success"] and lin["numRounds"] == 3
    assert lin["lineage"][1]["rejected"] == [s["attacker"]] and lin["lineage"][1]["attackers"] == [s["attacker"]]
    assert lin["lineage"][0]["attackers"] == [] and lin["lineage"][2]["attackers"] == []
    meta = lin["meta"]
    assert meta["perRoundTruth"] is True and meta["attackRounds"] == [2] and meta["detector"] is True
    assert meta["dataset"] == "UCI heart demo" and meta["numClients"] == 4 and meta["attackerFrac"] == 0.25
    commitments = [json.loads(line) for line in (demo_run["run_dir"] / "commitments.jsonl").read_text().splitlines()]
    who = commitments[-1]["leaves"][0]["client_id"]
    p = client.get(f"/api/fl/proof/{rid}/3/{who}").get_json()      # what the browser's Verify proof fetches
    assert p["success"] and p["root"] == commitments[-1]["root"]
    assert C.verify_proof(p["leaf"], p["proof"], p["root"])
    after = {r: client.get(f"/api/fl/lineage/{r}").get_json() for r in existing}
    assert after == before                            # the existing runs are untouched by the new one


def test_mnist_runs_push_exactly_as_before():
    from fedverify.chain import push as P
    dirs = sorted((RESULTS / "demo").glob("mnist_*/seed0")) if (RESULTS / "demo").exists() else []
    if not dirs:
        pytest.skip("no MNIST demo runs (gitignored) present")
    for d in dirs:
        commits, metrics, rid, meta = P.load_run(str(d))
        assert set(meta) == MNIST_META_KEYS, d          # no new key, in particular no perRoundTruth
        assert not P.is_survival_config(json.loads((d / "config.json").read_text()))


@needs_demo_inputs
def test_algorand_option_degrades_cleanly_without_network(tmp_path, monkeypatch, capsys):
    import torch
    from fedverify.chain import anchor
    from fedverify.survival import demo
    real = anchor._import_backend
    stub = types.SimpleNamespace(broadcast_fl_round=lambda *a, **k: {"success": False, "error": "no network here"})
    monkeypatch.setattr(anchor, "_import_backend", lambda name: stub if name == "algorand_client" else real(name))
    threads = torch.get_num_threads()
    try:
        code = demo.main(["--chain-backend", "algorand", "--out-dir", str(tmp_path), "--results-dir", str(RESULTS)])
    finally:
        torch.set_num_threads(threads)
    out = capsys.readouterr().out
    assert code == 0 and out.count("Algorand broadcast failed (no network here)") == 3
    s = json.loads((tmp_path / "demo" / "heart_ours_lr_demo" / "seed0" / "summary.json").read_text())
    assert len(s["events"]["algorand"]) == 3 and s["proof_verified"] and s["tamper_detected"]


def _js_function(src: str, name: str) -> str:
    m = re.search(r"\n((?:async )?function %s\(.*?\n\}\n)" % re.escape(name), src, re.S)
    assert m, f"{name} not found in frontend/index.html"
    return m.group(1)


def test_dashboard_detection_is_unchanged_for_runs_without_per_round_truth():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    src = (REPO / "frontend" / "index.html").read_text(encoding="utf-8")
    truth, det = _js_function(src, "_flTruth"), _js_function(src, "flDetection")
    assert "const truth = _flTruth(r, meta, atkSet);" in det
    old = det.replace("function flDetection(", "function flDetectionOld(").replace("_flTruth(r, meta, atkSet)", "atkSet")
    lineage_mnist = [{"round": r, "accepted": ["1", "2", "3"] if r != 2 else ["1", "2"], "rejected": [] if r != 2 else ["3"],
                      "attackers": ["3"], "metrics": {"test_acc": 0.5 + r / 10}} for r in (1, 2, 3)]
    lineage_demo = [{"round": r, "accepted": ["A", "B", "C", "D"] if r != 2 else ["A", "B", "C"],
                     "rejected": [] if r != 2 else ["D"], "attackers": ["D"] if r == 2 else []} for r in (1, 2, 3)]
    script = truth + det + old + f"""
const out = {{
  mnist_new: flDetection({json.dumps(lineage_mnist)}, {{attackerIds: ["3"]}}),
  mnist_old: flDetectionOld({json.dumps(lineage_mnist)}, {{attackerIds: ["3"]}}),
  demo_flag: flDetection({json.dumps(lineage_demo)}, {{attackerIds: ["D"], perRoundTruth: true, attackRounds: [2]}}),
  demo_noflag: flDetection({json.dumps(lineage_demo)}, {{attackerIds: ["D"]}}),
}};
console.log(JSON.stringify(out));
"""
    res = json.loads(subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=True).stdout)
    assert res["mnist_new"] == res["mnist_old"]                  # older runs: the same numbers as before the change
    assert (res["demo_flag"]["tp"], res["demo_flag"]["fn"], res["demo_flag"]["fp"]) == (1, 0, 0)
    assert res["demo_noflag"]["fn"] == 2                         # why the flag exists: rounds 1 and 3 are honest


@pytest.mark.skipif(not (RESULTS / "demo" / "heart_ours_lr_demo" / "seed0" / "summary.json").exists()
                    or not (RESULTS / "survival" / "fedavg_lr" / "seed0" / "summary.json").exists(),
                    reason="demo run or study results (gitignored) not present")
def test_demo_md_takes_its_numbers_from_the_files():
    from fedverify.analysis import make_demo_md as DM
    from fedverify.analysis.make_fl_healthcare_tables import Results, load_spec
    from fedverify.analysis.make_results_draft import recall_sentence
    spec = load_spec()
    res = Results(RESULTS, spec)
    text = DM.build(res, spec)
    assert "—" not in text and not res.missing
    c = json.loads((RESULTS / "demo/heart_ours_lr_demo/seed0/config.json").read_text())["config"]
    s = json.loads((RESULTS / "demo/heart_ours_lr_demo/seed0/summary.json").read_text())
    assert f"tau = {c['tau']:.2f}" in text and f"\"{s['attacker']} flips the sign" in text
    dp = json.loads((RESULTS / "survival/dp_fedavg_lr_eps1/seed0/config.json").read_text())["config"]
    assert f"(C = {dp['max_grad_norm']:g})" in text
    assert recall_sentence(res, spec, "test") in text
    md = (REPO / "fedverify" / "survival" / "DEMO.md").read_text(encoding="utf-8")
    assert md.startswith("# FedVerify live demo") and "make_demo_md" in md      # generated, not hand-written
