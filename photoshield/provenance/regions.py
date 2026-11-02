"""Region maps: per-tile canonical pHashes, committed by a Merkle root (design §15.2).

    G       = min(8, floor(min(W, H) / 32))          no map when G < 2
    tile    = rows [floor(r*H/G), floor((r+1)*H/G)), columns likewise (integers only)
    value   = "<16-hex pHash of the tile>:<flag>"     flag 1 = textured, 0 = flat
    leaf_i  = SHA256(JSON([i, "r{r}c{c}", value]))    the repository's canonical leaf
    root    = duplicate-last Merkle over the leaves  (backend/app.py, as for albums)

The manifest carries {"algo", "grid", "n", "root"}; the tile values stay with the owner, who can
recompute them from the original at any time. The browser reproduces every value bit for bit
(the tile pHash is the §7 pHash of the crop; the flag uses the same bin means).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import hashing, merkle

ALGO = "tile-phash@1"
MAX_GRID = 8
MIN_TILE = 32
FLAT_RANGE = 10000          # range of the 32x32 bin means, in x1000 luma units (10 grey levels)


def grid_for(width: int, height: int) -> int:
    g = min(MAX_GRID, min(int(width), int(height)) // MIN_TILE)
    return g if g >= 2 else 0


def tile_boxes(width: int, height: int, g: int) -> List[Tuple[int, int, int, int, int, int]]:
    """[(r, c, y0, y1, x0, x1)] in row-major order; the tiles partition the image exactly."""
    rows = [(i * height) // g for i in range(g + 1)]
    cols = [(j * width) // g for j in range(g + 1)]
    return [(r, c, rows[r], rows[r + 1], cols[c], cols[c + 1]) for r in range(g) for c in range(g)]


def names(g: int) -> List[str]:
    return [f"r{r}c{c}" for r in range(g) for c in range(g)]


def tile_value(crop: np.ndarray) -> str:
    f = hashing.bin_means(crop)
    flat = [v for row in f for v in row]
    textured = (max(flat) - min(flat)) >= FLAT_RANGE
    return f"{hashing.phash_from_bins(f)}:{1 if textured else 0}"


def split_value(value: str) -> Tuple[str, bool]:
    h, flag = value.split(":")
    return h, flag == "1"


def region_values(rgb: np.ndarray) -> Tuple[int, List[str]]:
    """(G, [value per tile, row-major]); G == 0 means the image is too small for a map."""
    h, w = rgb.shape[:2]
    g = grid_for(w, h)
    if not g:
        return 0, []
    return g, [tile_value(rgb[y0:y1, x0:x1]) for _, _, y0, y1, x0, x1 in tile_boxes(w, h, g)]


def commit(values: Sequence[str], g: int) -> Tuple[str, List[str]]:
    """(root, leaves) of the region map, by the canonical Merkle in backend/app.py."""
    if len(values) != g * g:
        raise ValueError("a region map needs grid*grid values")
    ns = names(g)
    root, leaves, _ = merkle._backend().compute_layer_merkle(dict(zip(ns, values)), ns)
    return root, leaves


def proofs(leaves: Sequence[str]) -> List[list]:
    app = merkle._backend()
    return [app.merkle_inclusion_proof(list(leaves), i) for i in range(len(leaves))]


def leaf(index: int, name: str, value: str) -> str:
    return merkle._backend()._manifest_leaf(index, name, value)


def region_map(rgb: np.ndarray) -> Optional[dict]:
    """The manifest's `regions` field for this image, or None if it is too small."""
    g, values = region_values(rgb)
    if not g:
        return None
    root, _ = commit(values, g)
    return {"algo": ALGO, "grid": g, "n": g * g, "root": root}
