"""FedVerify-Survival data layer: counts, leakage guard, data-independent encoding, CV folds (prompt §3-§4).

Tests that read the TCGA workbook skip when it is absent (it is gitignored); everything else runs anywhere.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from fedverify.survival import data as D  # noqa: E402
from fedverify.survival.config import HOSPITALS  # noqa: E402

needs_data = pytest.mark.skipif(not D.DEFAULT_DATA.exists(),
                                reason="TCGA-BRCA workbook (gitignored) is not at fedverify/data/tcga_brca/")

# Master prompt §3: verified counts per hospital BEFORE dropping any rows: (patients, deaths).
EXPECTED = {"BH": (145, 45), "A2": (98, 11), "E2": (87, 6), "A8": (76, 4), "AR": (68, 10)}

# D2 (INVENTORY.md §4): the workbook rows whose derived time disagrees with OS_MONTHS by >= 0.5 months. The two
# clinical sources disagree on these patients; the units are right (median error 0.02 months). Pinned, so a real
# unit mistake (hundreds of failures) or any NEW disagreement fails the test.
KNOWN_OS_MONTHS_EXCEPTIONS = sorted([
    "TCGA-A2-A25D", "TCGA-A8-A08L", "TCGA-AC-A23H", "TCGA-AR-A1AJ", "TCGA-B6-A0IH", "TCGA-BH-A18N",
    "TCGA-BH-A1F8", "TCGA-BH-A1FG", "TCGA-C8-A26Y", "TCGA-D8-A1JK", "TCGA-E2-A15O",
])


@pytest.fixture(scope="module")
def cohort():
    return D.load_cohort()


def _row(**over):
    row = {"diag_age": 50, "lymph_node": 2, "size_over_20mm": 1, "menop_status": "Post",
           "er_status_by_ihc": "Positive", "pr_status_by_ihc": "[Not Evaluated]", "her2_status_by_ihc": "",
           "histological_type": "Infiltrating Ductal Carcinoma"}
    row.update(over)
    return row


# ── counts (§3) ──────────────────────────────────────────────────────────────
@needs_data
def test_workbook_shape():
    assert D.read_workbook().shape == (907, 34)


@needs_data
def test_loader_counts_match_prompt(cohort):
    assert cohort.counts() == {h: {"patients": p, "deaths": d} for h, (p, d) in EXPECTED.items()}
    assert (cohort.n, cohort.n_deaths) == (474, 76)
    assert cohort.dropped == 0                           # no row of the five hospitals lacks a usable time


@needs_data
def test_other_sites_have_too_few_deaths():
    raw = D.read_workbook()
    site = np.array([D.hospital_of(s) for s in raw["submitter_id"]])
    event = D.to_float(raw["survival_event"])
    big = [s for s in np.unique(site) if (site == s).sum() >= 30]
    assert len(big) == 12
    assert sum(event[site == s].sum() < 10 for s in big) == 8


@needs_data
def test_time_units_against_os_months():
    u = D.unit_check(D.read_workbook())
    assert u["failing_ids"] == KNOWN_OS_MONTHS_EXCEPTIONS
    assert 1 - u["failed"] / u["checked"] >= 0.98


# ── leakage guard (§3, hard requirement) ─────────────────────────────────────
@pytest.mark.parametrize("col", sorted(D.LEAK_COLUMNS) + ["OS_MONTHS", "OS_STATUS", "DFS_MONTHS", "DFS_STATUS",
                                                          "PFS_MONTHS", "PFS_STATUS", "OS_anything_new"])
def test_leakage_guard_fires_for_every_denied_column(col):
    with pytest.raises(D.LeakageError):
        D.assert_no_leakage(["diag_age", col])


def test_leakage_guard_blocks_the_encoder():
    frame = pd.DataFrame([_row(survival=3.2)])
    with pytest.raises(D.LeakageError):
        D.encode_features(frame, D.load_vocab())


@needs_data
def test_no_outcome_column_reaches_the_model(cohort):
    X, names = D.encode_features(cohort.features, D.load_vocab())
    sources = {D.source_column(n) for n in names}
    assert sources == set(D.FEATURES_NUMERIC + D.FEATURES_CATEGORICAL)
    assert list(cohort.features.columns) == list(D.feature_columns())
    assert X.shape == (474, 30) and np.isfinite(X).all()


@needs_data
def test_race_only_behind_the_flag():
    assert "race" not in D.feature_columns()
    c = D.load_cohort(include_race=True)
    _, names = D.encode_features(c.features, D.load_vocab(), include_race=True)
    assert any(n.startswith("race=") for n in names)


# ── data-independent preprocessing (§3) ──────────────────────────────────────
def test_numeric_scaling_and_missing_indicators():
    X, names = D.encode_features(pd.DataFrame([_row(), _row(diag_age="", lymph_node="")]), D.load_vocab())
    col = {n: i for i, n in enumerate(names)}
    assert X[0, col["diag_age"]] == pytest.approx(0.5)              # /100
    assert X[0, col["lymph_node"]] == pytest.approx(0.2)            # /10
    assert X[0, col["size_over_20mm"]] == 1.0                       # as is
    assert X[0, col["diag_age_missing"]] == 0.0
    assert X[1, col["diag_age"]] == 0.0 and X[1, col["diag_age_missing"]] == 1.0
    assert X[1, col["lymph_node"]] == 0.0 and X[1, col["lymph_node_missing"]] == 1.0
    assert X[0, col["pr_status_by_ihc=NA"]] == 1.0                  # [Not Evaluated] -> NA
    assert X[0, col["her2_status_by_ihc=NA"]] == 1.0                # blank -> NA


def test_encoding_is_data_independent():
    rows = [_row(), _row(diag_age=81.2, menop_status="Pre"), _row(lymph_node="", her2_status_by_ihc="Positive")]
    whole, _ = D.encode_features(pd.DataFrame(rows), D.load_vocab())
    one_by_one = np.vstack([D.encode_features(pd.DataFrame([r]), D.load_vocab())[0] for r in rows])
    np.testing.assert_array_equal(whole, one_by_one)                # nothing is fitted on the data


def test_unknown_category_is_refused():
    with pytest.raises(ValueError, match="not in vocab.json"):
        D.encode_features(pd.DataFrame([_row(menop_status="Peri")]), D.load_vocab())


@needs_data
def test_vocab_json_is_the_full_cohort_vocabulary():
    assert D.build_vocab(D.read_workbook()) == D.load_vocab()
    assert all(v[-1] == D.NA for v in D.load_vocab().values())


# ── cross-validation (§4) ────────────────────────────────────────────────────
def test_repeat_seeds_are_base_plus_r():
    assert D.repeat_seeds(0, 5) == [0, 1, 2, 3, 4]
    assert D.repeat_seeds(3, 2) == [3, 4]


@needs_data
def test_no_patient_in_two_hospitals(cohort):
    assert len(set(cohort.patient_id)) == cohort.n
    for pid, h in zip(cohort.patient_id, cohort.hospital):
        assert D.hospital_of(pid) == h
    assert set(cohort.hospital) == set(HOSPITALS)


@needs_data
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_folds_partition_patients_and_respect_strata(cohort, seed):
    folds = D.make_folds(cohort, seed)
    assert len(folds) == 5
    seen = np.concatenate([te for _, te in folds])
    assert sorted(seen.tolist()) == list(range(cohort.n))         # every patient tested exactly once
    for tr, te in folds:
        assert np.intersect1d(tr, te).size == 0                     # never in train and test of one fold
        assert len(tr) + len(te) == cohort.n
        assert 15 <= cohort.event[te].sum() <= 16
        for h in HOSPITALS:                                         # stratified on (hospital, event)
            for e in (0, 1):
                share = ((cohort.hospital[te] == h) & (cohort.event[te] == e)).sum()
                total = ((cohort.hospital == h) & (cohort.event == e)).sum()
                assert abs(share - total / 5) <= 1


@needs_data
def test_fold_data_refuses_overlap(cohort):
    X, _ = D.encode_features(cohort.features, D.load_vocab())
    with pytest.raises(ValueError):
        D.fold_data(cohort, X, np.array([0, 1, 2]), np.array([2, 3]))
