"""Album Merkle commitments, built by the repository's canonical Merkle (design §6.2).

Nothing is re-implemented. The root, leaves and proofs come from backend/app.py
(`compute_layer_merkle`, `merkle_inclusion_proof`), the code the browser already re-verifies
with `bvMerkleLeaf` / `bvFoldProof`:

    leaf_i = SHA256(JSON([i, photoId_i, manifestDigest_i]))      compact, ensure_ascii=False
    parent = SHA256(hex(left) || hex(right)), duplicate-last on odd levels

The leaf count is anchored beside the root (`plc`), closing the duplicate-last ambiguity.
"""
from __future__ import annotations

import hashlib
import os
import sys
from typing import Iterable, List, Sequence, Tuple

_BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "backend")
_app = None


def _backend():
    """Import backend/app.py lazily and once (the fedverify/chain/commitment.py pattern)."""
    global _app
    if _app is None:
        if "app" in sys.modules and hasattr(sys.modules["app"], "compute_layer_merkle"):
            _app = sys.modules["app"]
        else:
            if _BACKEND not in sys.path:
                sys.path.insert(0, _BACKEND)
            import app as _mod
            _app = _mod
    return _app


def _entries(items: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out = [(str(pid), str(md)) for pid, md in items]
    ids = [p for p, _ in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate photoId in album")
    if not out:
        raise ValueError("an album needs at least one photo")
    return out


def album_root(items: Sequence[Tuple[str, str]]) -> Tuple[str, List[str], int]:
    """(root, leaves, leaf_count) for [(photoId, manifestDigest), ...] in album order."""
    ent = _entries(items)
    root, leaves, _ = _backend().compute_layer_merkle(dict(ent), [p for p, _ in ent])
    return root, leaves, len(leaves)


def album_proofs(items: Sequence[Tuple[str, str]]) -> List[list]:
    ent = _entries(items)
    _, leaves, _ = album_root(ent)
    app = _backend()
    return [app.merkle_inclusion_proof(leaves, i) for i in range(len(leaves))]


def leaf(index: int, photo_id: str, digest: str) -> str:
    return _backend()._manifest_leaf(index, photo_id, digest)


def verify_proof(leaf_hex: str, path: Iterable[dict], root: str) -> bool:
    """The verifier side, mirroring the browser's bvFoldProof."""
    h = str(leaf_hex)
    for step in path:
        sib = str(step["hash"])
        pair = (h + sib) if step.get("right") else (sib + h)
        h = hashlib.sha256(pair.encode()).hexdigest()
    return h == root
