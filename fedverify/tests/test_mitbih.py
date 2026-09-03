"""Phase-5 MIT-BIH tests.

The load-bearing ones are the split tests. If a patient's beats appear in both a client's
training data and the global test set, test macro-F1 is inflated by memorised morphology
rather than learned arrhythmia structure — the quiet failure mode of intra-patient splits.
"""
import os

import numpy as np
import pytest
import torch

from fedverify.core.data import dataset_spec
from fedverify.core.models import assert_no_batchnorm, build_model, get_flat_params
from fedverify.datasets import mitbih as M

HAVE_DATA = os.path.exists(M.cache_path())
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="MIT-BIH cache not built")


# ── record selection ─────────────────────────────────────────────────────────
def test_paced_records_are_excluded():
    """AAMI EC57 excludes paced beats; keeping them fills class Q with a device artefact."""
    assert not set(M.PACED) & set(M.RECORDS)
    for r in M.PACED:
        assert r not in M.DS1 and r not in M.DS2


def test_ds1_ds2_are_disjoint_and_cover_every_record():
    assert not set(M.DS1) & set(M.DS2)
    assert sorted(M.DS1 + M.DS2) == sorted(M.RECORDS)
    assert len(M.DS1) == len(M.DS2) == 22


def test_aami_mapping_covers_the_five_classes():
    assert sorted(set(M.AAMI.values())) == [0, 1, 2, 3, 4]
    assert M.CLASSES == ["N", "S", "V", "F", "Q"]
    assert M.AAMI["N"] == 0 and M.AAMI["V"] == 2 and M.AAMI["F"] == 3
    assert M.AAMI["/"] == 4                       # paced beat symbol -> Q


def test_non_beat_annotations_are_not_mapped():
    """Rhythm and signal-quality marks are not beats and must not become training rows."""
    for sym in ("+", "~", "|", '"', "x"):
        assert sym not in M.AAMI


# ── partitioning ─────────────────────────────────────────────────────────────
def _fake_recs():
    return np.repeat(np.array([101, 106, 108, 109, 112, 114]), 10)


def test_patient_partition_keeps_patients_whole():
    parts = M.patient_partition(_fake_recs(), 3, seed=0)
    per = M.patients_per_client(_fake_recs(), parts)
    flat = [p for g in per for p in g]
    assert len(flat) == len(set(flat))                       # no patient in two hospitals
    assert sorted(flat) == [101, 106, 108, 109, 112, 114]    # every patient placed


def test_patient_partition_is_deterministic():
    a = M.patient_partition(_fake_recs(), 3, seed=0)
    assert a == M.patient_partition(_fake_recs(), 3, seed=0)
    assert a != M.patient_partition(_fake_recs(), 3, seed=1)


def test_partition_indices_are_disjoint_and_complete():
    parts = M.patient_partition(_fake_recs(), 3, seed=0)
    flat = [i for p in parts for i in p]
    assert sorted(flat) == list(range(60))


def test_more_clients_than_patients_is_an_error():
    with pytest.raises(ValueError, match="only 6 training patients"):
        M.patient_partition(_fake_recs(), 7, seed=0)


# ── model ────────────────────────────────────────────────────────────────────
def test_ecg_model_shape_and_no_batchnorm():
    m = build_model("mitbih", num_classes=5, in_shape=(1, 256))
    assert_no_batchnorm(m)
    assert tuple(m(torch.randn(4, 1, 256)).shape) == (4, 5)


def test_ecg_model_is_comparable_in_size_to_smallcnn():
    """Table 4 should compare datasets, not model capacity."""
    ecg = get_flat_params(build_model("mitbih", 5, (1, 256))).numel()
    cnn = get_flat_params(build_model("mnist", 10, (1, 28, 28))).numel()
    assert 0.5 < ecg / cnn < 2.0


def test_dataset_spec_for_mitbih():
    s = dataset_spec("mitbih")
    assert s["num_classes"] == 5 and s["in_shape"] == (1, 256)


# ── real data ────────────────────────────────────────────────────────────────
@needs_data
def test_beats_have_the_declared_window_and_are_finite():
    X, y, rec = M.load_arrays()
    assert X.shape[1] == M.WINDOW == 256
    assert X.dtype == np.float32 and np.isfinite(X).all()
    assert set(np.unique(y).tolist()) <= {0, 1, 2, 3, 4}
    assert set(np.unique(rec).tolist()) == set(M.RECORDS)


@needs_data
def test_signal_is_z_normalised_per_record():
    """Gain and baseline differ between recordings, so scaling must be per record."""
    X, _y, rec = M.load_arrays()
    for r in (101, 208, 234):
        m = X[rec == r]
        assert abs(float(m.mean())) < 0.6            # centred, not raw millivolts
        assert 0.1 < float(m.std()) < 6.0


@needs_data
def test_train_and_test_patients_never_overlap():
    _tr, _te, rec_tr = M.build_torch_datasets()
    assert set(np.unique(rec_tr).tolist()) == set(M.DS1)
    assert not set(np.unique(rec_tr).tolist()) & set(M.DS2)


@needs_data
def test_class_imbalance_makes_accuracy_misleading():
    """The justification for reporting macro-F1: always-N already scores ~89% accuracy."""
    _X, y, _rec = M.load_arrays()
    counts = np.bincount(y, minlength=5)
    majority = counts.max() / counts.sum()
    assert majority > 0.85
    assert counts[3] * 50 < counts[0]                # F is two orders below N


@needs_data
def test_hospitals_really_are_heterogeneous():
    """The claim the dataset exists to support: prevalence differs between hospitals."""
    _X, y_all, rec = M.load_arrays()
    _tr, _te, rec_tr = M.build_torch_datasets()
    y_tr = y_all[np.isin(rec, M.DS1)]
    parts = M.patient_partition(rec_tr, 10, seed=0)
    v_share = []
    for idx in parts:
        h = np.bincount(y_tr[np.asarray(idx)], minlength=5)
        v_share.append(h[2] / h.sum())
    assert max(v_share) > 10 * max(min(v_share), 1e-4)   # order-of-magnitude spread


@needs_data
def test_federated_mitbih_run_reports_macro_f1(tmp_path):
    from fedverify.config import FLConfig
    from fedverify.core.runner import run
    cfg = FLConfig(dataset="mitbih", num_clients=5, rounds=1, seed=0,
                   exp="test", out_dir=str(tmp_path))
    out = run(cfg, progress=False)
    assert cfg.cell == "mitbih_K5_patient_epsinf"        # no misleading alpha in the name
    assert 0.0 <= out["final"]["macro_f1"] <= 1.0
    pr = out["partition_report"]
    assert len(pr["patients_per_client"]) == 5
    flat = [p for g in pr["patients_per_client"] for p in g]
    assert sorted(flat) == sorted(M.DS1)
