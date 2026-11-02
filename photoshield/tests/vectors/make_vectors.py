"""Regenerate the pHash test vectors (design §7). Deterministic: same files every run.

    python3 -m photoshield.tests.vectors.make_vectors

Writes the PNGs next to this file and phash_vectors.json with the expected hex. The expected
values come from photoshield.provenance.hashing; tests/phash_js_test.mjs (run by
test_hashing.py) checks that the SHIPPED browser function gives the same bits.

Real photographs come from scikit-image's bundled sample data (public domain or CC0, see
skimage.data), downsampled so the repository stays small.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def _save(name, arr, mode=None):
    img = Image.fromarray(arr, mode) if mode else Image.fromarray(arr)
    img.save(os.path.join(HERE, name), optimize=False)
    return name


def synthetic():
    rng = np.random.default_rng(0)
    out = []
    x = np.linspace(0, 255, 64, dtype=np.float64)
    gh = np.repeat(x[None, :], 64, 0).astype(np.uint8)
    out.append(_save("grad_h_64x64.png", np.stack([gh, gh, gh], -1)))
    gv = np.repeat(np.linspace(0, 255, 48)[:, None], 64, 1).astype(np.uint8)
    out.append(_save("grad_v_64x48.png", np.stack([gv, 255 - gv, gv // 2], -1)))
    yy, xx = np.mgrid[0:75, 0:100]
    gd = ((xx + yy) * 255 // 173).astype(np.uint8)
    out.append(_save("grad_diag_100x75.png", np.stack([gd, gd[::-1], 255 - gd], -1)))
    cb = (((np.arange(64)[:, None] // 8) + (np.arange(64)[None, :] // 8)) % 2 * 255).astype(np.uint8)
    out.append(_save("checker_64x64.png", np.stack([cb] * 3, -1)))
    cf = (((np.arange(96)[:, None] // 3) + (np.arange(96)[None, :] // 3)) % 2 * 255).astype(np.uint8)
    out.append(_save("checker_fine_96x96.png", np.stack([cf] * 3, -1)))
    out.append(_save("noise_rgb_80x60.png", rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)))
    out.append(_save("noise_gray_64x64.png", rng.integers(0, 256, (64, 64), dtype=np.uint8), "L"))
    out.append(_save("min_32x32.png", rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)))
    out.append(_save("tall_33x200.png", rng.integers(0, 256, (200, 33, 3), dtype=np.uint8)))
    out.append(_save("wide_300x40.png", np.stack([np.tile(np.arange(300) % 256, (40, 1)).astype(np.uint8)] * 3, -1)))
    yy, xx = np.mgrid[0:64, 0:64]
    a = np.where((xx - 32) ** 2 + (yy - 32) ** 2 < 24 ** 2, 255, 0).astype(np.uint8)
    rgba = np.stack([np.full((64, 64), 200, np.uint8), (xx * 4).astype(np.uint8), (yy * 4).astype(np.uint8), a], -1)
    out.append(_save("alpha_circle_64x64.png", rgba, "RGBA"))
    a2 = (xx[:48, :48] * 5).clip(0, 255).astype(np.uint8)
    rgba2 = np.stack([rng.integers(0, 256, (48, 48), dtype=np.uint8)] * 3 + [a2], -1)
    out.append(_save("alpha_ramp_48x48.png", rgba2, "RGBA"))
    pal = Image.fromarray(rng.integers(0, 256, (50, 70, 3), dtype=np.uint8)).quantize(16)
    pal.save(os.path.join(HERE, "palette_70x50.png"))
    out.append("palette_70x50.png")
    return out


def photos():
    import skimage.data as d
    out = []
    for name in ("astronaut", "coffee", "chelsea", "rocket", "camera", "hubble_deep_field",
                 "coins", "brick"):
        arr = getattr(d, name)()
        img = Image.fromarray(arr)
        w, h = img.size
        img = img.resize((128, max(32, round(h * 128 / w))), Image.LANCZOS)
        fname = f"photo_{name}.png"
        img.save(os.path.join(HERE, fname), optimize=False)
        out.append(fname)
    return out


def main():
    from photoshield.provenance.hashing import phash
    names = synthetic() + photos()
    expected = {n: phash(os.path.join(HERE, n)) for n in sorted(names)}
    rng = np.random.default_rng(1)
    _save("unhashable_31x40.png", rng.integers(0, 256, (40, 31, 3), dtype=np.uint8))
    with open(os.path.join(HERE, "phash_vectors.json"), "w") as f:
        json.dump({"spec": "PHOTOSHIELD_DESIGN.md §7", "vectors": expected,
                   "unhashable": ["unhashable_31x40.png"]}, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"{len(expected)} vectors written")


if __name__ == "__main__":
    main()
