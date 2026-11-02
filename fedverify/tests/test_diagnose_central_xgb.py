"""Phase 8: the central-XGBoost diagnosis measures what the runs did, and the draft reports it from its file."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
pytest.importorskip("lifelines")

from fedverify.analysis import diagnose_central_xgb as DG  # noqa: E402
from fedverify.survival.common import LOG_TIME, RISK  # noqa: E402
from fedverify.survival.metrics import c_index, c_index_time  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "fedverify" / "results" / "diagnostics" / "central_xgb_tcga.json"
have_diag = pytest.mark.skipif(not DIAG.exists(), reason="diagnostics (gitignored) not built")


def test_pair_split_by_hand():
    t, e, site = np.array([1.0, 2.0, 3.0]), np.array([1, 1, 0]), np.array(["a", "a", "b"])
    # comparable pairs: (0,1) same site; (0,2), (1,2) cross site
    good = DG.pair_split(t, e, np.array([1.0, 2.0, 3.0]), site, LOG_TIME)   # longer predicted time for the later one
    assert good["all"] == {"c": 1.0, "pairs": 3}
    assert good["within_site"] == {"c": 1.0, "pairs": 1} and good["cross_site"] == {"c": 1.0, "pairs": 2}
    assert DG.pair_split(t, e, np.array([3.0, 2.0, 1.0]), site, LOG_TIME)["all"]["c"] == 0.0
    assert DG.pair_split(t, e, np.array([3.0, 2.0, 1.0]), site, RISK)["all"]["c"] == 1.0   # a risk is the reverse
    assert DG.pair_split(t, e, np.zeros(3), site, LOG_TIME)["all"]["c"] == 0.5             # ties count one half


def test_pair_split_matches_lifelines_on_untied_times():
    rng = np.random.default_rng(0)
    t = rng.permutation(np.arange(1.0, 61.0))
    e = rng.integers(0, 2, 60)
    s = rng.normal(size=60)
    site = rng.choice(["A", "B", "C"], 60)
    assert DG.pair_split(t, e, s, site, LOG_TIME)["all"]["c"] == pytest.approx(c_index_time(t, e, s), abs=1e-12)
    assert DG.pair_split(t, e, s, site, RISK)["all"]["c"] == pytest.approx(c_index(t, e, s), abs=1e-12)


def test_steps_per_round_follow_the_combination_rules():
    K, N = 5, 5
    assert DG.steps_per_round("central_xgb", K, N) == K * N       # K x N trees boosted one after another
    assert DG.steps_per_round("fedxgb_bagging", K, N) == N        # one averaged N-tree booster per round
    assert DG.steps_per_round("local_xgb", K, N) == N             # each hospital's own N trees


@have_diag
def test_diagnosis_reproduces_the_results_and_counts_trees():
    d = json.loads(DIAG.read_text(encoding="utf-8"))
    assert len(d["runs"]) == len(DG.METHODS) * len(DG.TASKS)
    for run in d["runs"]:
        assert run["reproduces_results"]["ok"], run["cell"]          # the replay is the run on disk
        last = run["per_round"][-1]
        assert last["trees_total"] == run["K"] * run["N"] * run["rounds"]    # equal tree budget for all three
        assert last["sequential_steps"] == run["steps_per_round"] * run["rounds"]
    sites = d["sites"]
    assert sum(s["patients"] for s in sites.values()) == json.loads(
        (REPO / "fedverify/results/survival/fedavg_lr/seed0/config.json").read_text())["data"]["patients"]


@have_diag
def test_results_draft_reports_the_diagnosis(tmp_path):
    from fedverify.analysis import make_results_draft as DR
    out = tmp_path / "RESULTS_DRAFT.md"
    assert DR.main(["--out", str(out)]) == 0
    lim = out.read_text(encoding="utf-8").split("## Limitations", 1)[1]
    bullet = next(line for line in lim.splitlines() if line.startswith("- TCGA centralized XGBoost"))
    d = json.loads(DIAG.read_text(encoding="utf-8"))
    central = next(r for r in d["runs"] if r["task"] == "survival" and r["method"] == "central_xgb")
    assert f"{central['per_round'][-1]['held_c']:.3f}" in bullet and "—" not in bullet
    assert "not a configuration bug" in bullet
