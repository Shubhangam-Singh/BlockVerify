"""FedVerify-Survival methods (prompt §5, Phase 1): logistic and random-forest families on one real CV fold."""
import numpy as np
import pytest

pytest.importorskip("lifelines")
pytest.importorskip("sklearn")

from fedverify.core.models import assert_no_batchnorm  # noqa: E402
from fedverify.survival import data as D  # noqa: E402
from fedverify.survival import fl_lr, fl_rf  # noqa: E402
from fedverify.survival.config import HOSPITALS, SurvivalConfig, canonical_method, derive_seed  # noqa: E402
from fedverify.survival.models import build_logistic, get_flat_params  # noqa: E402

needs_data = pytest.mark.skipif(not D.DEFAULT_DATA.exists(),
                                reason="TCGA-BRCA workbook (gitignored) is not at fedverify/data/tcga_brca/")


@pytest.fixture(scope="module")
def fold0():
    cohort = D.load_cohort()
    X, _ = D.encode_features(cohort.features, D.load_vocab())
    tr, te = D.make_folds(cohort, 0)[0]
    return D.fold_data(cohort, X, tr, te)


def _run(fold, method, rounds=2):
    family = "lr" if method.endswith("_lr") else "rf"
    cfg = SurvivalConfig(family=family, method=method, rounds=rounds)
    module = fl_lr if family == "lr" else fl_rf
    return cfg, list(module.run_fold(cfg, fold, repeat_seed=0, fold_index=0))


def test_config_names_and_validation():
    assert canonical_method("lr", "fedavg") == "fedavg_lr"
    assert canonical_method("rf", "fed") == "fed_rf"
    with pytest.raises(ValueError):
        canonical_method("rf", "fedavg_lr")
    with pytest.raises(ValueError):
        SurvivalConfig(family="rf", method="fed_rf", rf_n_jobs=2)
    assert SurvivalConfig().momentum == 0.0                       # D12
    assert derive_seed(0, 0, 1, "BH") == derive_seed(0, 0, 1, "BH")
    assert len({derive_seed(0, 0, r, h) for r in (1, 2) for h in HOSPITALS}) == 10


def test_logistic_model_is_zero_initialised_and_batchnorm_free():
    model = build_logistic(30)
    assert_no_batchnorm(model)
    p = get_flat_params(model)
    assert p.numel() == 31 and float(p.abs().sum()) == 0.0


@needs_data
@pytest.mark.parametrize("method", ["local_lr", "central_lr", "fedsgd_lr", "fedavg_lr",
                                    "local_rf", "central_rf", "fed_rf"])
def test_every_method_runs_and_yields_valid_risks(fold0, method):
    cfg, outs = _run(fold0, method)
    assert [o.round for o in outs] == [1, 2]
    for o in outs:
        assert o.train_risk.shape == (len(fold0.train_idx),) and o.test_risk.shape == (len(fold0.test_idx),)
        assert np.all((o.test_risk >= 0) & (o.test_risk <= 1)) and np.isfinite(o.train_risk).all()
        assert o.diag["accepted"] == list(HOSPITALS) and o.diag["rejected"] == []
        assert sum(o.per_client_num_samples.values()) == len(fold0.train_idx)


@needs_data
def test_fedsgd_and_fedavg_differ(fold0):
    _, sgd = _run(fold0, "fedsgd_lr")
    _, avg = _run(fold0, "fedavg_lr")
    assert not np.allclose(sgd[-1].test_risk, avg[-1].test_risk)


@needs_data
def test_fedavg_aggregates_deltas_with_the_existing_fedavg(fold0):
    _, outs = _run(fold0, "fedavg_lr", rounds=1)
    w = {h: s["weight"] for h, s in outs[0].diag["scores"].items()}
    n = outs[0].per_client_num_samples
    assert set(w) == set(HOSPITALS)                                 # hospital codes on disk (D10), not 0-4
    for h in HOSPITALS:                                             # sample-count weights, as in FedAvg
        assert w[h] == pytest.approx(n[h] / sum(n.values()))


@needs_data
def test_lr_rounds_are_deterministic(fold0):
    _, a = _run(fold0, "fedavg_lr")
    _, b = _run(fold0, "fedavg_lr")
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.test_risk, y.test_risk)


@needs_data
def test_fed_rf_prediction_is_the_mean_of_all_tree_probabilities(fold0):
    cfg, outs = _run(fold0, "fed_rf", rounds=1)
    X_tr = np.ascontiguousarray(fold0.X_train, dtype=np.float32)
    X_te = np.ascontiguousarray(fold0.X_test, dtype=np.float32)
    trees, cols = [], []
    for h in HOSPITALS:                                             # rebuild round 1 from the derived seeds
        rows = fold0.hosp_train == h
        forest = fl_rf.make_forest(cfg, cfg.trees_per_hospital, derive_seed(0, 0, 1, h))
        forest.fit(X_tr[rows], fold0.event_train[rows])
        np.testing.assert_allclose(fl_rf.tree_probability_sum(forest, X_te) / len(forest.estimators_),
                                   forest.predict_proba(X_te)[:, fl_rf.death_column(forest)], rtol=0, atol=1e-12)
        trees += list(forest.estimators_)
        cols += [fl_rf.death_column(forest)] * len(forest.estimators_)
    mean = np.mean([t.predict_proba(X_te)[:, c] for t, c in zip(trees, cols)], axis=0)
    np.testing.assert_allclose(outs[0].test_risk, mean, rtol=0, atol=1e-12)
    assert outs[0].extra["trees_total"] == len(trees) == cfg.trees_per_hospital * len(HOSPITALS)


def test_rf_forests_are_single_threaded_and_seeded():
    cfg = SurvivalConfig(family="rf", method="fed_rf")
    forest = fl_rf.make_forest(cfg, 10, derive_seed(0, 0, 1, "BH"))
    assert forest.n_jobs == 1 and forest.random_state == derive_seed(0, 0, 1, "BH")
    assert forest.max_depth == 5 and forest.min_samples_leaf == 3
    assert forest.class_weight == "balanced_subsample"
