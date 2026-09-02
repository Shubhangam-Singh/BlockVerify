"""Datasets and federated partitioning.

The Dirichlet partitioner is the standard non-IID construction: for each class c we
draw p ~ Dir(alpha * 1_K) and split that class's indices across clients by p. Small
alpha => each class concentrates on a few clients (severe non-IID); alpha -> inf
approaches IID. ``partition_report`` exists so the non-IID claim is provable rather
than asserted.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence

import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

_SPECS = {                       # name -> (torchvision class, mean, std, n_classes, in_shape)
    "mnist":  ("MNIST", (0.1307,), (0.3081,), 10, (1, 28, 28)),
    "fmnist": ("FashionMNIST", (0.2860,), (0.3530,), 10, (1, 28, 28)),
}


def dataset_spec(name: str) -> dict:
    if name not in _SPECS:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(_SPECS)} "
                         "(mitbih arrives in Phase 5)")
    cls, mean, std, n, shape = _SPECS[name]
    return {"torchvision_class": cls, "mean": mean, "std": std,
            "num_classes": n, "in_shape": shape}


def load_dataset(name: str, root: str = DATA_ROOT):
    """Return (train_dataset, test_dataset), downloading into fedverify/data/."""
    import torchvision
    import torchvision.transforms as T

    spec = dataset_spec(name)
    tf = T.Compose([T.ToTensor(), T.Normalize(spec["mean"], spec["std"])])
    ctor = getattr(torchvision.datasets, spec["torchvision_class"])
    os.makedirs(root, exist_ok=True)
    train = ctor(root=root, train=True, download=True, transform=tf)
    test = ctor(root=root, train=False, download=True, transform=tf)
    return train, test


def get_labels(dataset) -> np.ndarray:
    """Labels as a numpy array without materialising the images."""
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            t = getattr(dataset, attr)
            return np.asarray(t.numpy() if hasattr(t, "numpy") else t)
    return np.asarray([int(y) for _, y in dataset])


def iid_partition(labels: Sequence[int], num_clients: int, seed: int) -> List[List[int]]:
    """Uniform random split (alpha == inf)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    rng.shuffle(idx)
    return [sorted(chunk.tolist()) for chunk in np.array_split(idx, num_clients)]


def dirichlet_partition(labels: Sequence[int], num_clients: int, alpha: float,
                        seed: int, min_size: int = 10) -> List[List[int]]:
    """Per-class Dirichlet split. Resamples the WHOLE draw if any client is too small."""
    if alpha == math.inf:
        return iid_partition(labels, num_clients, seed)
    if alpha <= 0:
        raise ValueError("alpha must be > 0 (or inf for IID)")

    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    for _ in range(100):
        parts: List[List[int]] = [[] for _ in range(num_clients)]
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [sorted(p) for p in parts]

    raise RuntimeError(
        f"could not partition {len(labels)} samples across {num_clients} clients with "
        f"alpha={alpha} and min_size={min_size} after 100 attempts; raise alpha, "
        f"lower min_size, or use fewer clients")


def partition_report(partitions: Sequence[Sequence[int]], labels: Sequence[int]) -> Dict:
    """Per-client class histogram + skew summary, so non-IID is provable."""
    labels = np.asarray(labels)
    classes = sorted(int(c) for c in np.unique(labels))
    hist = np.zeros((len(partitions), len(classes)), dtype=int)
    for k, idx in enumerate(partitions):
        if len(idx):
            vals, cnt = np.unique(labels[np.asarray(idx)], return_counts=True)
            for v, n in zip(vals, cnt):
                hist[k, classes.index(int(v))] = n

    # mean KL( client class dist || uniform ), the standard skew scalar
    kls = []
    for row in hist:
        tot = row.sum()
        if tot == 0:
            continue
        p = row / tot
        nz = p > 0
        kls.append(float(np.sum(p[nz] * np.log(p[nz] * len(classes)))))

    return {
        "classes": classes,
        "counts": hist.tolist(),
        "sizes": [int(r.sum()) for r in hist],
        "mean_kl_to_uniform": float(np.mean(kls)) if kls else 0.0,
        "min_client_size": int(hist.sum(axis=1).min()),
        "max_client_size": int(hist.sum(axis=1).max()),
    }
