"""Bring a suspect image into the original's frame (design §15.3).

1. ORB keypoints + Hamming matching + Lowe's ratio test, then a RANSAC similarity transform
   (rotation, uniform scale, translation). Unedited regions anchor the fit even when part of the
   picture (a face, a background) was replaced.
2. Fallback: a plain resize when the aspect ratios agree within 1 % (same framing).
3. Otherwise: AlignError.

Deterministic: OpenCV's RANSAC RNG is seeded before every call.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

MIN_INLIERS = 12
ASPECT_TOL = 0.01          # same framing; a cropped copy goes through feature alignment
FEATURE_SIDE = 1200          # detect features on a copy whose longer side is at most this


class AlignError(ValueError):
    """The suspect could not be placed in the original's frame."""


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _gray_small(rgb: np.ndarray, cv2):
    h, w = rgb.shape[:2]
    s = min(1.0, FEATURE_SIDE / max(h, w))
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if s < 1.0:
        g = cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return g, s


def _features(original: np.ndarray, suspect: np.ndarray, cv2) -> Optional[dict]:
    go, so = _gray_small(original, cv2)
    gs, ss = _gray_small(suspect, cv2)
    orb = cv2.ORB_create(nfeatures=5000)
    ko, do = orb.detectAndCompute(go, None)
    ks, ds = orb.detectAndCompute(gs, None)
    if do is None or ds is None or len(ko) < MIN_INLIERS or len(ks) < MIN_INLIERS:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ds, do, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([ks[m.queryIdx].pt for m in good]) / ss      # suspect, full-res pixels
    dst = np.float32([ko[m.trainIdx].pt for m in good]) / so      # original, full-res pixels
    cv2.setRNGSeed(0)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0 / so,
                                         maxIters=5000, confidence=0.995)
    if M is None or inl is None or int(inl.sum()) < MIN_INLIERS:
        return None
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    if not 0.1 <= scale <= 10.0:
        return None
    return {"method": "orb-ransac", "M": M.astype(np.float64), "inliers": int(inl.sum())}


def _resize_matrix(original: np.ndarray, suspect: np.ndarray) -> Optional[dict]:
    ho, wo = original.shape[:2]
    hs, ws = suspect.shape[:2]
    if abs((wo / ho) - (ws / hs)) / (wo / ho) > ASPECT_TOL:
        return None
    M = np.array([[wo / ws, 0.0, 0.0], [0.0, ho / hs, 0.0]])
    return {"method": "resize", "M": M, "inliers": 0}


def warp(suspect: np.ndarray, M: np.ndarray, width: int, height: int):
    """(warped HxWx3 uint8, valid-pixel mask HxW bool) in the original's frame."""
    cv2 = _cv2()
    if cv2 is None:                       # resize-only path without OpenCV
        from PIL import Image
        if not (np.isclose(M[0, 1], 0) and np.isclose(M[1, 0], 0) and np.isclose(M[0, 2], 0)
                and np.isclose(M[1, 2], 0)):
            raise AlignError("a rotated or shifted alignment needs OpenCV (pip install opencv-python-headless)")
        img = Image.fromarray(suspect).resize((width, height), Image.BILINEAR)
        return np.asarray(img), np.ones((height, width), dtype=bool)
    out = cv2.warpAffine(suspect, M, (width, height), flags=cv2.INTER_AREA if np.linalg.det(M[:, :2]) < 1 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    ones = np.full(suspect.shape[:2], 255, dtype=np.uint8)
    mask = cv2.warpAffine(ones, M, (width, height), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return out, mask


def align(original: np.ndarray, suspect: np.ndarray, mode: str = "auto") -> dict:
    """{"method", "M" (2x3, suspect -> original pixels), "inliers", "warped", "mask"}.

    mode: "auto" (features, then resize), "features", or "resize".
    """
    cv2 = _cv2()
    res = None
    if mode in ("auto", "features") and cv2 is not None:
        res = _features(original, suspect, cv2)
    if res is None and mode in ("auto", "resize"):
        res = _resize_matrix(original, suspect)
    if res is None:
        why = "OpenCV is not installed" if cv2 is None and mode != "resize" else \
              "too few matching features and the aspect ratios differ"
        raise AlignError(f"cannot align the suspect to the original ({why})")
    h, w = original.shape[:2]
    res["warped"], res["mask"] = warp(suspect, res["M"], w, h)
    return res


def matrix_strings(M: np.ndarray) -> list:
    """The 2x3 matrix as six decimal strings (no floats in signed documents, design §15.5)."""
    return [f"{float(v):.6f}" for v in (M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2])]


def matrix_from_strings(vals) -> np.ndarray:
    v = [float(x) for x in vals]
    return np.array([[v[0], v[1], v[2]], [v[3], v[4], v[5]]], dtype=np.float64)
