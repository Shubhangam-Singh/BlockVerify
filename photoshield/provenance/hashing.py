"""SHA-256 and the canonical 64-bit perceptual hash (design §7).

The pHash is computed the same way, step by step, in `frontend/index.html` (`psPhashFromRGBA`).
The two must give identical bits for identical decoded pixels. The tests prove it on the PNG
vectors in tests/vectors/ by running the shipped JS under Node (tests/phash_js_test.mjs).

What may differ, and is measured rather than assumed (experiment E6): JPEG decoders. A browser
and PIL can decode the same .jpg to pixels that differ by ±1.

This module needs numpy and Pillow. The Flask backend never imports it: the server only ever
receives hashes.
"""
from __future__ import annotations

import hashlib
import io
from typing import Union

import numpy as np

from .dct_table import COS, S

SIDE = 32           # resize to SIDE x SIDE
LOW = 8             # keep the LOW x LOW low-frequency block
MIN_SIDE = 32       # smaller images are UNHASHABLE (empty bins otherwise)


class Unhashable(ValueError):
    """The image is smaller than MIN_SIDE x MIN_SIDE."""


# ── SHA-256 ──────────────────────────────────────────────────────────────────
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── decoding (§7 steps 1-2) ──────────────────────────────────────────────────
def decode_rgb(src: Union[str, bytes, "np.ndarray"]) -> np.ndarray:
    """Decode to an H x W x 3 uint8 RGB array, the way the browser does.

    - EXIF orientation is applied (the browser default `imageOrientation: "from-image"`).
    - The ICC profile is NOT applied (the browser is asked for `colorSpaceConversion: "none"`).
    - Alpha is composited onto white with the exact integer rule of §7 step 2.
    """
    if isinstance(src, np.ndarray):
        arr = src
    else:
        from PIL import Image, ImageOps
        img = Image.open(io.BytesIO(src) if isinstance(src, bytes) else src)
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
        arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("expected an H x W x 3 or H x W x 4 image")
    arr = arr.astype(np.int64)
    if arr.shape[2] == 4:
        rgb, a = arr[..., :3], arr[..., 3:4]
        arr = (rgb * a + 255 * (255 - a) + 127) // 255
    return arr.astype(np.uint8)


def composite_rgba(rgba: np.ndarray) -> np.ndarray:
    """§7 step 2 on an RGBA array (exposed for the cross-language tests)."""
    return decode_rgb(np.asarray(rgba, dtype=np.uint8))


# ── the canonical pHash (§7 steps 3-6) ───────────────────────────────────────
def _bin_means(rgb: np.ndarray) -> list:
    """Integer luma, area-average into SIDE x SIDE bins, then ONE division per bin.

    Sums are exact int64 (a 50 MP bin sum is about 1.3e10, far below 2**53), so numpy is safe
    here: there is no floating-point arithmetic until the single division, which JS performs
    identically (correctly rounded IEEE division of two exact integers).
    """
    h, w = rgb.shape[:2]
    if h < MIN_SIDE or w < MIN_SIDE:
        raise Unhashable(f"image is {w}x{h}; pHash needs at least {MIN_SIDE}x{MIN_SIDE}")
    rgb = rgb.astype(np.int64)
    y = 299 * rgb[..., 0] + 587 * rgb[..., 1] + 114 * rgb[..., 2]
    re = [(i * h) // SIDE for i in range(SIDE + 1)]
    ce = [(j * w) // SIDE for j in range(SIDE + 1)]
    rows = np.add.reduceat(y, re[:-1], axis=0)          # SIDE x w
    sums = np.add.reduceat(rows, ce[:-1], axis=1)       # SIDE x SIDE
    f = []
    for i in range(SIDE):
        row = []
        for j in range(SIDE):
            count = (re[i + 1] - re[i]) * (ce[j + 1] - ce[j])
            row.append(int(sums[i, j]) / count)
        f.append(row)
    return f


def _dct_low(f: list) -> list:
    """The 8 x 8 low block of the orthonormal 2-D DCT-II, in the fixed order of §7 step 5.

        g[x][v] = sum over y = 0..31 of COS[v][y] * f[x][y]          (y ascending, from 0.0)
        F[u][v] = (S[u] * S[v]) * (sum over x = 0..31 of COS[u][x] * g[x][v])   (x ascending)

    Plain Python floats, no numpy and no FMA, so the browser reproduces every rounding step.
    """
    g = [[0.0] * LOW for _ in range(SIDE)]
    for x in range(SIDE):
        fx = f[x]
        for v in range(LOW):
            cv = COS[v]
            acc = 0.0
            for y in range(SIDE):
                acc += cv[y] * fx[y]
            g[x][v] = acc
    out = [[0.0] * LOW for _ in range(LOW)]
    for u in range(LOW):
        cu = COS[u]
        for v in range(LOW):
            acc = 0.0
            for x in range(SIDE):
                acc += cu[x] * g[x][v]
            out[u][v] = (S[u] * S[v]) * acc
    return out


def phash_from_rgb(rgb: np.ndarray) -> str:
    """64-bit pHash as 16 lowercase hex characters (MSB first, row-major over u then v)."""
    return phash_from_bins(_bin_means(rgb))


def bin_means(rgb: np.ndarray) -> list:
    """The 32 x 32 luma bin means (x1000 luma units) that the pHash is computed from."""
    return _bin_means(rgb)


def phash_from_bins(f: list) -> str:
    """§7 steps 5-6 on precomputed bin means (so a caller can reuse the bins, e.g. regions)."""
    F = _dct_low(f)
    coeffs = [F[u][v] for u in range(LOW) for v in range(LOW) if (u, v) != (0, 0)]
    median = sorted(coeffs)[31]
    bits = 0
    for u in range(LOW):
        for v in range(LOW):
            bits <<= 1
            if (u, v) != (0, 0) and F[u][v] > median:
                bits |= 1
    return f"{bits:016x}"


def phash(src: Union[str, bytes, "np.ndarray"]) -> str:
    """Decode (§7 steps 1-2) and hash. Accepts a path, raw bytes or an RGB(A) array."""
    return phash_from_rgb(decode_rgb(src))


def hamming(a: str, b: str) -> int:
    """Bit distance between two equal-length hex hashes."""
    if len(a) != len(b):
        raise ValueError("hashes have different lengths")
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def image_size(src: Union[str, bytes]) -> tuple:
    """(width, height) after EXIF orientation, as the browser reports it."""
    rgb = decode_rgb(src)
    return int(rgb.shape[1]), int(rgb.shape[0])
