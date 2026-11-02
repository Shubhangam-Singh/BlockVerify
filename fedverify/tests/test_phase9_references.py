"""Phase 9: the site-rate reference row, the step-matched diagnostic row, and how the draft and page report them."""
import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
pytest.importorskip("lifelines")
pytest.importorskip("matplotlib")

from fedverify.analysis import make_results_draft as DR  # noqa: E402
from fedverify.analysis import make_fl_healthcare_tables as TAB  # noqa: E402
from fedverify.survival import data as D  # noqa: E402
from fedverify.survival import fl_xgb  # noqa: E402
from fedverify.survival.config import HOSPITALS, SurvivalConfig  # noqa: E402
from fedverify.survival.metrics import thresholds_for_test  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "fedverify" / "results"
needs_tcga = pytest.mark.skipif(not D.DEFAULT_DATA.exists(), reason="TCGA workbook (gitignored) not present")
have_results = pytest.mark.skipif(
    not (RESULTS / "survival" / "central_xgb_steps_aft" / "seed0" / "summary.json").exists()
    or not (RESULTS / "diagnostics" / "central_xgb_tcga.json").exists()
    or not (RESULTS / "tables" / "methods_tcga_binary.md").exists(), reason="Phase 9 results (gitignored) not built")


# ── the site-rate reference ──────────────────────────────────────────────────
def test_site_rate_uses_the_training_folds_only():
    rng = np.random.default_rng(1)
    hosp = rng.choice(["A", "B", "C"], 90)
    event = rng.integers(0, 2, 90)
    fold = np.arange(90) % 5
    score, pred = TAB.site_rate_oof(hosp, event, fold)
    for i in range(90):                               # each score: the site's event rate in the OTHER folds
        tr = (fold != fold[i]) & (hosp == hosp[i])
        assert score[i] == pytest.approx(event[tr].mean(), abs=1e-12)
    flipped = event.copy()
    flipped[0] = 1 - flipped[0]                        # a patient's own label never reaches their own score
    assert TAB.site_rate_oof(hosp, flipped, fold)[0][0] == score[0]
    for k in range(5):                                # the runs' threshold rule, fitted on the training rows' scores
        te, tr = fold == k, fold != k
        rate = {h: event[tr & (hosp == h)].mean() for h in "ABC"}
        thr = thresholds_for_test(event[tr], np.array([rate[h] for h in hosp[tr]]), hosp[tr], hosp[te], False)
        np.testing.assert_array_equal(pred[te], (score[te] >= thr).astype(int))


def test_site_rate_sentence_names_only_the_rule_verdicts():
    pairs = [("LR: Local", "outperforms"), ("RF: Local", "is comparable with"),
             ("XGB: Local", "is not distinguishable at this sample size from"), ("LR: Ours", "is outperformed by")]
    text = DR.site_rate_sentence("ROC-AUC", pairs)
    assert "1 of 4 exceed it (95% CI of the paired difference above 0) (LR: Local)" in text
    assert "1 of 4 are below it (95% CI below 0) (LR: Ours)" in text
    assert "the other 2 are not distinguishable from it (RF: Local; XGB: Local)" in text
    none = DR.site_rate_sentence("F1", [("a", "is comparable with")])
    assert "none exceeds it" in none and "none is below it" in none


# ── the step-matched diagnostic row ──────────────────────────────────────────
def test_step_matched_cells_and_rule_are_fixed():
    assert SurvivalConfig(family="xgb", method="central_xgb_steps").cell == "central_xgb_steps"
    assert SurvivalConfig(family="xgb", method="central_xgb_steps", task="survival").cell == "central_xgb_steps_aft"
    spec = yaml.safe_load(TAB.SPEC.read_text(encoding="utf-8"))
    rule = spec["diagnostic_rows"]["central_step_matched"]
    assert rule["method"] == "central_xgb_steps" and rule["compares_with"] == "central_xgb"
    for name in ("methods_heart", "methods_tcga_binary", "methods_tcga_survival"):
        rows = spec["method_tables"][name]["rows"]
        labels = [r["label"] for r in rows]
        assert "Centralized" in labels                                  # the original row is kept
        diag = next(r for r in rows if r.get("diagnostic") == "central_step_matched")
        assert diag["label"] == "Centralized, step-matched (diagnostic)" and diag["lr"] == diag["rf"] == "n/a"
        assert diag["xgb"].startswith("central_xgb_steps")


@needs_tcga
@pytest.mark.parametrize("task", ["binary", "survival"])
def test_step_matched_grows_one_federated_boosters_worth_per_round(task):
    c = D.load_cohort()
    X, _ = D.encode_features(c.features, D.load_vocab())
    tr, te = D.make_folds(c, 0)[0]
    fold = D.fold_data(c, X, tr, te)
    for method, per_round in (("central_xgb_steps", 5), ("central_xgb", 5 * len(HOSPITALS))):
        cfg = SurvivalConfig(family="xgb", method=method, task=task, rounds=3)
        assert cfg.xgb_trees_per_round == 5
        for r, out in enumerate(fl_xgb.run_fold(cfg, fold, 0, 0), start=1):
            assert out.extra["trees_total"] == per_round * r


# ── the results on disk ──────────────────────────────────────────────────────
@have_results
def test_tables_carry_both_rows_and_the_footnotes_read_the_files():
    diag = json.loads((RESULTS / "diagnostics" / "central_xgb_tcga.json").read_text(encoding="utf-8"))
    for name in ("methods_heart", "methods_tcga_binary", "methods_tcga_survival"):
        text = (RESULTS / "tables" / f"{name}.md").read_text(encoding="utf-8")
        assert "| Site base rate only (reference) |" in text and "| Centralized, step-matched (diagnostic) |" in text
        assert "| Centralized |" in text
        assert "The rule was fixed in table.yaml before this row was run." in text
    survival = (RESULTS / "tables" / "methods_tcga_survival.md").read_text(encoding="utf-8")
    run = next(r for r in diag["runs"] if r["task"] == "survival" and r["method"] == "central_xgb")
    at150 = next(p for p in run["per_round"] if p["sequential_steps"] == 150)
    assert f"held-out C-index after 150 boosting steps ({at150['held_c']:.3f})" in survival     # not blind, from file
    assert "Heart was not diagnosed" in (RESULTS / "tables" / "methods_heart.md").read_text(encoding="utf-8")
    res = TAB.Results(RESULTS, TAB.load_spec())
    st, vals = res.site_rate("tcga", "roc_auc", "test")
    assert st == "ok"                      # two independent computations of the site-only predictor agree exactly
    assert float(np.mean(vals)) == pytest.approx(diag["site_only"]["roc_auc"][0], abs=1e-12)


@have_results
def test_draft_states_the_two_sentences_and_only_rule_verdicts_against_the_site_rate(tmp_path):
    out = tmp_path / "RESULTS_DRAFT.md"
    assert DR.main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    para = next(line for line in text.splitlines() if line.startswith("**Reference rows.**"))
    diag = json.loads((RESULTS / "diagnostics" / "central_xgb_tcga.json").read_text(encoding="utf-8"))
    run = next(r for r in diag["runs"] if r["task"] == "survival" and r["method"] == "central_xgb")
    assert f"training C-index of {run['per_round'][-1]['train_c']:.3f}" in para
    s = json.loads((RESULTS / "survival/central_xgb_aft/seed0/summary.json").read_text())
    held = np.mean([rp["pooled"]["c_index"] for rp in s["repeats"]])
    assert f"held-out C-index is {held:.3f}" in para
    assert "pooled ROC-AUC partly measures how base rates differ between hospitals" in para
    comps = json.loads(out.with_suffix(".comparisons.json").read_text(encoding="utf-8"))
    spec = yaml.safe_load(TAB.SPEC.read_text(encoding="utf-8"))
    for name, metrics in spec["site_rate"]["compare_on"].items():
        for m in metrics:
            mine = [c for c in comps if c["section"] == f"{name}/site_rate" and c["metric"] == m]
            assert mine
            n, k = len(mine), sum(c["relation"] == "outperforms" for c in mine)
            for c in mine:                                      # "outperforms" only with an interval above 0
                if c["relation"] == "outperforms":
                    assert c["ci"][0] > 0
            want = "none exceeds it" if k == 0 else (f"all {n} exceed it" if k == n else f"{k} of {n} exceed it")
            label = spec["metric_labels"][m]
            sentence = re.search(rf"\({name}\), against the site-rate reference \([^)]*\): on {re.escape(label)}, [^.]*\.", para)
            assert sentence and want in sentence.group(0), (name, m)


@have_results
def test_showcase_shows_the_site_rate_row_and_the_sentences(tmp_path):
    from fedverify.analysis import make_showcase as SH
    assert SH.main(["--out-dir", str(tmp_path)]) == 0
    body = (tmp_path / "showcase_body.html").read_text(encoding="utf-8")
    blob = re.search(r'<script type="application/json" id="showcase-data">(.*?)</script>', body, re.S).group(1)
    d = json.loads(blob.replace("<\\/", "</"))
    tcga = d["methods"]["tcga"]
    auc = next(m for m in tcga["metrics"] if m["key"] == "roc_auc")
    site = next(r for r in auc["rows"] if r["label"] == "Site base rate only (reference)")
    st, vals = TAB.Results(RESULTS, TAB.load_spec()).site_rate("tcga", "roc_auc", "test")
    assert st == "ok" and site["baseline"] and site["cells"]["lr"][0] == pytest.approx(float(np.mean(vals)), abs=1e-6)
    assert auc["site_verdict"] and "exceed" in auc["site_verdict"]["text"]
    assert any("over-fits" in n for n in tcga["notes"]) and any("base rates" in n for n in tcga["notes"])
    assert any(r["label"] == "Centralized, step-matched (diagnostic)" for r in auc["rows"])
