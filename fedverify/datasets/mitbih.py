"""MIT-BIH Arrhythmia (PhysioNet) as a federated, patient-partitioned dataset.

Why this dataset earns its place: MNIST non-IID is *synthesised* by a Dirichlet draw, so
the heterogeneity is a knob we chose. Here the heterogeneity is real — each client is a
distinct set of PATIENTS, and arrhythmia prevalence genuinely differs between them. No
alpha is tuned; the skew is whatever the patients are.

Construction (all standard, so the numbers are comparable to the literature):

  records      the 48 MIT-BIH records minus the four PACED ones (102, 104, 107, 217).
               AAMI EC57 excludes paced beats from arrhythmia scoring; keeping them would
               inflate the Q class with a device artefact rather than a physiological one.
  split        de Chazal's inter-patient DS1/DS2 (22 records each). DS1 becomes the
               federated clients, DS2 the global test set. Patients never cross the
               boundary, so test accuracy is not inflated by memorising a patient's
               morphology — the intra-patient split most papers use quietly does that.
  beats        256-sample windows centred on each annotated R-peak (~0.71 s at 360 Hz).
  scaling      z-normalised PER RECORD, using that record's own mean/std, because gain and
               baseline differ between recordings.
  labels       AAMI 5-class N / S / V / F / Q.

Class balance is extreme (N dominates), so **macro-F1 is the primary metric**; accuracy
looks excellent for a model that only ever predicts N.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

WINDOW = 256
HALF = WINDOW // 2
CLASSES = ["N", "S", "V", "F", "Q"]
NUM_CLASSES = len(CLASSES)

# Paced records, excluded per AAMI EC57.
PACED = (102, 104, 107, 217)

# de Chazal et al. (2004) inter-patient split; both halves already exclude the paced set.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
RECORDS = DS1 + DS2

# AAMI EC57 beat-class mapping.
AAMI: Dict[str, int] = {
    **{s: 0 for s in ("N", "L", "R", "e", "j")},          # N  normal / bundle-branch
    **{s: 1 for s in ("A", "a", "J", "S")},               # S  supraventricular ectopic
    **{s: 2 for s in ("V", "E")},                          # V  ventricular ectopic
    **{s: 3 for s in ("F",)},                              # F  fusion
    **{s: 4 for s in ("/", "f", "Q")},                     # Q  unclassifiable / paced beat
}


def data_root(root: str = None) -> str:
    if root:
        return root
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "mitdb")


def download(root: str = None, records: Sequence[int] = None) -> str:
    """Fetch the records from PhysioNet into fedverify/data/mitdb/ (skips what exists)."""
    import wfdb
    root = data_root(root)
    os.makedirs(root, exist_ok=True)
    recs = [str(r) for r in (records or RECORDS)]
    missing = [r for r in recs if not os.path.exists(os.path.join(root, f"{r}.dat"))]
    if missing:
        wfdb.dl_database("mitdb", dl_dir=root, records=missing, annotators=["atr"])
    return root


def load_record(rec: int, root: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """(beats [n, 256] float32, labels [n] int64) for one record."""
    import wfdb
    root = data_root(root)
    path = os.path.join(root, str(rec))
    sig = wfdb.rdrecord(path).p_signal[:, 0].astype(np.float64)   # channel 0 (usually MLII)
    ann = wfdb.rdann(path, "atr")

    mu, sd = sig.mean(), sig.std()
    sig = (sig - mu) / (sd if sd > 1e-9 else 1.0)                 # z-norm per record

    beats, labels = [], []
    n = sig.shape[0]
    for pos, sym in zip(ann.sample, ann.symbol):
        cls = AAMI.get(sym)
        if cls is None:                       # non-beat annotation (rhythm marks etc.)
            continue
        lo, hi = int(pos) - HALF, int(pos) + HALF
        if lo < 0 or hi > n:                  # window would run off the recording
            continue
        beats.append(sig[lo:hi])
        labels.append(cls)

    if not beats:
        return np.zeros((0, WINDOW), np.float32), np.zeros((0,), np.int64)
    return np.asarray(beats, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def cache_path(root: str = None) -> str:
    return os.path.join(data_root(root), f"mitbih_w{WINDOW}.npz")


def build_cache(root: str = None, force: bool = False, verbose: bool = True) -> str:
    """Parse every record once into a single .npz. Parsing 44 records takes minutes."""
    root = data_root(root)
    path = cache_path(root)
    if os.path.exists(path) and not force:
        return path
    download(root)

    X, y, rid = [], [], []
    for i, rec in enumerate(RECORDS, 1):
        bx, by = load_record(rec, root)
        X.append(bx)
        y.append(by)
        rid.append(np.full(by.shape[0], rec, dtype=np.int64))
        if verbose:
            print(f"  [{i}/{len(RECORDS)}] record {rec}: {by.shape[0]} beats", flush=True)

    np.savez_compressed(path, X=np.concatenate(X), y=np.concatenate(y),
                        rec=np.concatenate(rid))
    return path


def load_arrays(root: str = None):
    """(X [n,256], y [n], rec [n]) for every non-paced record."""
    path = cache_path(root)
    if not os.path.exists(path):
        build_cache(root)
    d = np.load(path)
    return d["X"], d["y"], d["rec"]


def patient_partition(rec_ids: np.ndarray, num_clients: int, seed: int) -> List[List[int]]:
    """Assign whole PATIENTS to hospitals; a patient's beats never span two clients.

    This is the point of the dataset: heterogeneity is inherited from who the patients
    are, not synthesised by a Dirichlet draw.
    """
    patients = np.unique(rec_ids)
    if num_clients > len(patients):
        raise ValueError(f"{num_clients} clients requested but only {len(patients)} "
                         "training patients are available; each client needs >= 1")
    order = np.random.default_rng(seed).permutation(len(patients))
    groups = np.array_split(patients[order], num_clients)

    by_patient = {int(p): [] for p in patients}
    for i, r in enumerate(rec_ids):
        by_patient[int(r)].append(i)

    out = []
    for g in groups:
        idx = []
        for p in g:
            idx.extend(by_patient[int(p)])
        out.append(sorted(idx))
    return out


def patients_per_client(rec_ids: np.ndarray, partitions) -> List[List[int]]:
    """Which patient records ended up at each hospital — for the partition report."""
    return [sorted(set(int(rec_ids[i]) for i in idx)) for idx in partitions]


def build_torch_datasets(root: str = None):
    """(train_set, test_set, train_rec_ids) as TensorDatasets shaped (1, 256).

    train = DS1 patients (split across hospitals), test = DS2 patients.
    """
    import torch
    from torch.utils.data import TensorDataset

    X, y, rec = load_arrays(root)
    ds1 = np.isin(rec, DS1)
    ds2 = np.isin(rec, DS2)

    def make(mask):
        xt = torch.from_numpy(X[mask]).unsqueeze(1)      # (n, 1, 256)
        yt = torch.from_numpy(y[mask])
        ds = TensorDataset(xt, yt)
        ds.targets = yt                                   # so get_labels() works unchanged
        return ds

    return make(ds1), make(ds2), rec[ds1]
