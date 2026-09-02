import math
import numpy as np
import pytest
from fedverify.core.data import (dirichlet_partition, iid_partition, partition_report)

LABELS = np.repeat(np.arange(10), 500)      # 5000 samples, 10 balanced classes


def _check_valid(parts, labels, min_size=10):
    flat = [i for p in parts for i in p]
    assert len(set(flat)) == len(flat), "partitions must be disjoint"
    assert sorted(flat) == list(range(len(labels))), "partitions must be complete"
    assert min(len(p) for p in parts) >= min_size


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0, 100.0, math.inf])
def test_partitions_disjoint_complete_and_min_size(alpha):
    _check_valid(dirichlet_partition(LABELS, 10, alpha, seed=0), LABELS)


def test_high_alpha_is_closer_to_uniform_than_low_alpha():
    lo = partition_report(dirichlet_partition(LABELS, 10, 0.1, seed=0), LABELS)
    hi = partition_report(dirichlet_partition(LABELS, 10, 100.0, seed=0), LABELS)
    assert hi["mean_kl_to_uniform"] < lo["mean_kl_to_uniform"], (
        "alpha=100 must be more IID than alpha=0.1")


def test_alpha_inf_routes_to_iid():
    a = dirichlet_partition(LABELS, 10, math.inf, seed=3)
    b = iid_partition(LABELS, 10, seed=3)
    assert a == b


def test_partition_is_deterministic_for_a_seed():
    assert dirichlet_partition(LABELS, 10, 0.3, seed=7) == \
           dirichlet_partition(LABELS, 10, 0.3, seed=7)


def test_impossible_partition_raises():
    with pytest.raises(RuntimeError):
        dirichlet_partition(np.repeat(np.arange(10), 2), 10, 0.01, seed=0, min_size=50)


def test_partition_report_counts_sum_to_sizes():
    parts = dirichlet_partition(LABELS, 5, 0.5, seed=1)
    rep = partition_report(parts, LABELS)
    assert [sum(row) for row in rep["counts"]] == rep["sizes"]
    assert sum(rep["sizes"]) == len(LABELS)
