"""
BlockVerify — evaluation library.

Faithful Python port of the deployed in-browser detection probes
(frontend/index.html: bvFindOutliers, bvLayerHealth), a safetensors reader
(numpy only, no torch), threat-model attack simulators, and pure-numpy ROC.

The port is byte-for-byte cross-validated against the JavaScript engine
(see cross_validate.* / evaluate.py) so the evaluation measures the *deployed*
detector, not a re-derivation of it.
"""
import json
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  safetensors reader (numpy only)
# ─────────────────────────────────────────────────────────────────────────────
_ST_DTYPE = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


def load_safetensors(path):
    """Return {tensor_name: np.ndarray}. Handles BF16 → float32 widening."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n).decode("utf-8"))
        buf = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt, (s, e), shape = meta["dtype"], meta["data_offsets"], meta["shape"]
        if dt == "BF16":
            u16 = np.frombuffer(buf[s:e], dtype=np.uint16).astype(np.uint32) << 16
            arr = u16.view(np.float32)
        else:
            arr = np.frombuffer(buf[s:e], dtype=_ST_DTYPE[dt])
        out[name] = arr.reshape(shape) if shape else arr
    return out


def weight_layers(tensors, min_elems=1024):
    """Real *weight-matrix* tensors worth analysing (≥ min_elems, ≥2-D, floaty)."""
    layers = {}
    for name, arr in tensors.items():
        if arr.ndim >= 2 and arr.size >= min_elems and np.issubdtype(arr.dtype, np.floating):
            layers[name] = arr.astype(np.float64)   # widen to match JS float64 doubles
    return layers


# ─────────────────────────────────────────────────────────────────────────────
#  Detection probes — faithful ports of the DEPLOYED JavaScript
# ─────────────────────────────────────────────────────────────────────────────
def _flat(arr):
    """Row-major flatten of finite numbers (matches bvFlatten / .ravel(order='C'))."""
    a = np.asarray(arr, dtype=np.float64).ravel(order="C")
    return a[np.isfinite(a)]


def bv_numeric_stats(arr):
    nums = _flat(arr)
    n = nums.size
    if n == 0:
        return None
    mean = nums.sum() / n
    std = np.sqrt(((nums - mean) ** 2).sum() / n)   # population std (ddof=0), as in JS
    return {"n": int(n), "min": float(nums.min()), "max": float(nums.max()),
            "mean": float(mean), "std": float(std)}


def bv_find_outliers(arr, z_thresh=8.0):
    """Robust median/MAD outlier probe. Returns max_z, count>thresh, scale, n."""
    nums = _flat(arr)
    n = nums.size
    base = bv_numeric_stats(arr)
    if n < 4:
        return {"n": int(n), "count": 0, "max_z": 0.0, "scale": 0.0}
    srt = np.sort(nums)
    median = srt[n // 2]                              # JS: sorted[floor(n/2)] (upper-middle)
    devs = np.sort(np.abs(nums - median))
    mad = devs[n // 2] * 1.4826
    scale = mad if mad > 1e-9 else (base["std"] if base and base["std"] > 1e-9 else 1.0)
    z = np.abs(nums - median) / scale
    return {"n": int(n), "count": int((z > z_thresh).sum()),
            "max_z": float(z.max()), "scale": float(scale), "median": float(median)}


def bv_layer_health(arr):
    """NaN/Inf, extreme magnitude, Shannon entropy (32-bin/log2 32), constant runs."""
    a = np.asarray(arr, dtype=np.float64).ravel(order="C")
    nan = int(np.isnan(a).sum())
    inf = int(np.isinf(a).sum())
    nums = a[np.isfinite(a)]
    n = nums.size
    extremes = int((np.abs(nums) > 100).sum())
    max_abs = float(np.abs(nums).max()) if n else 0.0
    entropy = None
    if n > 16:
        mn, mx = nums.min(), nums.max()
        entropy = 0.0
        if mx > mn:
            b = np.minimum(31, np.floor((nums - mn) / (mx - mn) * 32)).astype(np.int64)
            counts = np.bincount(b, minlength=32)
            p = counts[counts > 0] / n
            entropy = float(-(p * np.log2(p)).sum() / 5.0)   # log2(32) = 5
    # longest identical consecutive run
    max_run = 1
    if n:
        eq = nums[1:] == nums[:-1]
        run = 1
        for e in eq:
            run = run + 1 if e else 1
            if run > max_run:
                max_run = run
    return {"n": int(n), "nan": nan, "inf": inf, "extremes": extremes,
            "max_abs": max_abs, "entropy": entropy, "max_run": int(max_run)}


# ─────────────────────────────────────────────────────────────────────────────
#  Threat-model attack simulators  (operate on a flat float64 copy)
# ─────────────────────────────────────────────────────────────────────────────
def attack_outlier(flat, k, delta, rng):
    """T1 weight-poisoning: set k random weights to ±delta (BadNets-style triggers)."""
    w = flat.copy()
    k = min(k, w.size)
    idx = rng.choice(w.size, size=k, replace=False)
    signs = rng.choice([-1.0, 1.0], size=k)
    w[idx] = delta * signs
    return w


def benign_drift(flat, sigma_rel, rng):
    """Legitimate fine-tuning: additive Gaussian noise on ALL weights (hard negative)."""
    std = flat.std()
    return flat + rng.normal(0.0, sigma_rel * (std if std > 0 else 1.0), size=flat.shape)


def attack_constant_block(flat, frac, rng):
    """Low-entropy / low-rank poisoning: overwrite a contiguous fraction with a constant."""
    w = flat.copy()
    m = max(1, int(frac * w.size))
    start = rng.integers(0, max(1, w.size - m + 1))
    w[start:start + m] = float(np.median(flat))
    return w


# ─────────────────────────────────────────────────────────────────────────────
#  Pure-numpy ROC / AUC
# ─────────────────────────────────────────────────────────────────────────────
def roc_curve(scores, labels):
    """Return (fpr, tpr, thresholds) sorted by descending threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    P = max(1, labels.sum())
    N = max(1, (labels == 0).sum())
    order = np.argsort(-scores)
    s, l = scores[order], labels[order]
    tps = np.cumsum(l)
    fps = np.cumsum(1 - l)
    # keep one point per distinct threshold
    keep = np.r_[np.diff(s) != 0, True]
    tpr = np.r_[0.0, tps[keep] / P]
    fpr = np.r_[0.0, fps[keep] / N]
    thr = np.r_[np.inf, s[keep]]
    return fpr, tpr, thr


def auc(fpr, tpr):
    return float(np.trapz(tpr, fpr))


def rates_at(scores, labels, thresh, direction=">"):
    """TPR/FPR when flagging score `direction` thresh (operating point)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    flag = scores > thresh if direction == ">" else scores >= thresh
    P = max(1, labels.sum()); N = max(1, (labels == 0).sum())
    tpr = float((flag & (labels == 1)).sum() / P)
    fpr = float((flag & (labels == 0)).sum() / N)
    return tpr, fpr
