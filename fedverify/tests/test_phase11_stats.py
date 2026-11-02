"""Phase 11: within-hospital metrics, the paired bootstrap, and the verdict rule built on it."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lifelines")
pytest.importorskip("sklearn")

from lifelines.utils import concordance_index  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from fedverify.analysis import make_results_draft as DR  # noqa: E402
from fedverify.analysis import paired_stats as PS  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "fedverify" / "results"
have_results = pytest.mark.skipif(not (RESULTS / "survival" / "fedavg_lr" / "seed0" / "repeat0" / "oof.jsonl").exists()
                                  or not (RESULTS / "tables" / "paired_comparisons.md").exists(),
                                  reason="results (gitignored) not built")


def synthetic(n=120, seed=0, ties=True):
    rng = np.random.default_rng(seed)
    hosp = rng.choice(["A", "B", "C"], n)
    event = rng.integers(0, 2, n)
    time = rng.integers(1, 40 if ties else 10 ** 6, n).astype(float)        # integer days: many tied times
    risk = np.round(rng.normal(size=n), 1 if ties else 9)                  # tied scores too
    return PS.Preds(np.array([str(i) for i in range(n)]), hosp, event, time, risk, -risk,
                    (risk > 0).astype(int), np.ones(n, bool), np.arange(n) % 5)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pair_metrics_equal_lifelines_and_sklearn_on_tied_data(seed):
    p = synthetic(seed=seed)
    assert PS.value(p, "c_index") == pytest.approx(concordance_index(p.time, p.surv, p.event), abs=1e-12)
    assert PS.value(p, "roc_auc") == pytest.approx(roc_auc_score(p.event, p.risk), abs=1e-12)
    assert PS.value(p, "accuracy") == pytest.approx(np.mean(p.pred == p.event), abs=1e-12)


def test_within_hospital_is_the_pair_weighted_mean_of_the_per_hospital_metrics():
    p = synthetic(n=300, seed=3)
    num = den = 0.0
    for h in ("A", "B", "C"):
        m = p.hospital == h
        pairs = (p.event[m] == 1).sum() * (p.event[m] == 0).sum()
        num += roc_auc_score(p.event[m], p.risk[m]) * pairs
        den += pairs
    assert PS.value(p, "roc_auc_within", keep=["A", "B", "C"]) == pytest.approx(num / den, abs=1e-12)
    only_ab = PS.value(p, "roc_auc_within", keep=["A", "B"])                # a left-out hospital contributes nothing
    m = np.isin(p.hospital, ["A", "B"])
    pairs = {h: (p.event[p.hospital == h] == 1).sum() * (p.event[p.hospital == h] == 0).sum() for h in "AB"}
    want = sum(roc_auc_score(p.event[p.hospital == h], p.risk[p.hospital == h]) * pairs[h] for h in "AB") / sum(pairs.values())
    assert only_ab == pytest.approx(want, abs=1e-12) and m.any()


def test_a_score_that_only_knows_the_hospital_scores_exactly_half_within_hospital():
    p = synthetic(n=200, seed=4)
    site = {"A": 0.9, "B": 0.1, "C": 0.5}
    r = np.array([site[h] for h in p.hospital])
    q = PS.Preds(p.patients, p.hospital, p.event, p.time, r, -r, p.pred, p.valid, p.fold)
    assert PS.value(q, "roc_auc_within", keep=["A", "B", "C"]) == 0.5
    assert PS.value(q, "c_index_within", keep=["A", "B", "C"]) == 0.5
    assert PS.value(q, "roc_auc") != 0.5                                    # pooled, the hospital alone ranks


def test_stratified_bootstrap_keeps_each_hospitals_size_and_is_deterministic():
    hosp = np.array(list("AAAAABBBCCCCCCC"))
    w1 = PS.stratified_weights(hosp, 500, 0, "toy")
    w2 = PS.stratified_weights(hosp, 500, 0, "toy")
    w3 = PS.stratified_weights(hosp, 500, 1, "toy")
    assert np.array_equal(w1, w2) and not np.array_equal(w1, w3)
    for h in "ABC":
        assert (w1[:, hosp == h].sum(axis=1) == (hosp == h).sum()).all()


def test_ci_rule():
    assert DR.ci_relation([0.01, 0.05]) == "outperforms"
    assert DR.ci_relation([-0.05, -0.01]) == "is outperformed by"
    assert DR.ci_relation([-0.01, 0.05]) == DR.NOT_DISTINGUISHABLE
    assert DR.ci_relation(None) == DR.NOT_DISTINGUISHABLE
    assert DR.ci_guarded_relation([0.01, 0.05], 0.845, 0.70, True, "accuracy", 0.84, 0.01) == DR.NOT_RANKED


@have_results
def test_the_bootstrap_reproduces_the_published_numbers_and_itself(tmp_path):
    from fedverify.analysis.make_fl_healthcare_tables import Results, load_spec
    spec = load_spec()
    res = Results(RESULTS, spec)
    st1, st2 = PS.PairedStats(res, spec), PS.PairedStats(res, spec)
    for st in (st1, st2):
        st.cache, st.path = {"_settings": st.settings()}, tmp_path / "cache.json"      # no shared cache
    a = st1.compare("tcga", "fedxgb_bagging_aft", "local_xgb_aft", "c_index")
    b = st2.compare("tcga", "fedxgb_bagging_aft", "local_xgb_aft", "c_index")
    assert a["ci"] == b["ci"] and a["per_repeat_diff"] == b["per_repeat_diff"]       # fixed seed: identical
    s_a = json.loads((RESULTS / "survival/fedxgb_bagging_aft/seed0/summary.json").read_text())
    s_b = json.loads((RESULTS / "survival/local_xgb_aft/seed0/summary.json").read_text())
    for r in range(5):                                                   # unit weights give the runs' own numbers
        assert a["per_repeat_diff"][r] == pytest.approx(s_a["repeats"][r]["pooled"]["c_index"]
                                                        - s_b["repeats"][r]["pooled"]["c_index"], abs=1e-12)
    assert a["resamples_valid"] == spec["paired_bootstrap"]["resamples"]


@have_results
def test_within_hospital_references_and_the_fold_artefact():
    """The requested sanity check: Predict all alive scores exactly 0.5 within hospital. The site-rate row does not:
    its rate comes from the other folds, so it differs slightly between folds of a hospital. Within one fold (where its
    score is constant per hospital) it is exactly 0.5, so the gap is that fold artefact and nothing else."""
    from fedverify.analysis.make_fl_healthcare_tables import Results, load_spec
    spec = load_spec()
    res = Results(RESULTS, spec)
    for ds, metric in (("heart", "roc_auc_within"), ("tcga", "roc_auc_within"), ("tcga", "c_index_within")):
        keep, _ = PS.included_hospitals(res, ds, spec)
        for r in range(5):
            assert PS.value(PS.load_preds(res, ds, "@all_alive", r), metric, keep=keep) == 0.5
            p = PS.load_preds(res, ds, "@site_rate", r)
            same_fold = PS.pair_terms(p, metric, PS.restriction(p, "within", keep) & (p.fold[:, None] == p.fold[None, :]))
            assert (same_fold[0].sum() / same_fold[1].sum()) == 0.5
            full = PS.value(p, metric, keep=keep)
            assert full != 0.5 and abs(full - 0.5) < 0.05                   # small, and only the artefact


@have_results
def test_verdict_changes_lists_every_changed_verdict():
    comps = json.loads((REPO / "fedverify/survival/paper/RESULTS_DRAFT.comparisons.json").read_text())
    text = (REPO / "fedverify/survival/paper/VERDICT_CHANGES.md").read_text()
    changed = [c for c in comps if c["old_relation"] != c["relation"]]
    assert f"{len(changed)} of {len(comps)} generated verdicts changed wording" in text
    assert sum(1 for line in text.splitlines() if line.startswith("| ") and not line.startswith("| Section")) == len(changed)
