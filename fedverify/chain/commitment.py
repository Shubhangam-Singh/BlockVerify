"""Round commitment: canonical delta bytes -> digest -> Merkle root -> inclusion proof.

The Merkle construction is deliberately NOT reimplemented here. Phase 0 identified five
distinct Merkle implementations in this repository with three different leaf encodings;
the canonical one is ``backend/app.py`` (``_manifest_leaf`` / ``compute_layer_merkle`` /
``merkle_inclusion_proof``), which is the one the browser already re-verifies against the
on-chain root for layer manifests. This module imports and reuses it, so an FL round root
is produced by exactly the same code path as a layer-manifest root:

    leaf_i = SHA-256(JSON([i, client_id, digest]))     compact separators, ensure_ascii=False
    parent = SHA-256(hex(left) || hex(right))          duplicate-last on odd levels

The leaf COUNT is returned alongside the root and anchored with it, because duplicate-last
trees are ambiguous without it (CVE-2012-2459-style): two different leaf multisets can fold
to the same root if the verifier is free to choose the level widths.

Position is bound into the leaf, so swapping two clients within a round changes the root.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# ── reuse the canonical implementation from backend/app.py ───────────────────
_BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "backend")
_app = None


def _backend():
    """Import backend/app.py lazily and once. Offline-safe: no network at import."""
    global _app
    if _app is None:
        if _BACKEND not in sys.path:
            sys.path.insert(0, _BACKEND)
        import app as _mod
        _app = _mod
    return _app


def warm_up() -> None:
    """Force the backend import before any timed region.

    ``_backend()`` costs ~1.5-2.5 s on first call (it imports the Flask app module). If
    that lands inside ``commit_round`` it is charged to round 1's merkle_ms and corrupts
    Table 3, so the runner warms it once at setup.
    """
    _backend()


# ── canonical bytes ──────────────────────────────────────────────────────────
def canon_update(delta) -> bytes:
    """Byte-stable serialization of a client delta: float32, little-endian, C-contiguous.

    Fixing dtype AND byte order makes the digest identical across machines and across
    processes; ``np.float32`` alone would still be endian-dependent on a big-endian host.
    """
    if hasattr(delta, "detach"):                 # torch.Tensor
        delta = delta.detach().cpu().numpy()
    arr = np.ascontiguousarray(np.asarray(delta), dtype="<f4")
    return arr.tobytes(order="C")


def update_digest(client_id: int, round_num: int, delta, num_samples: int) -> str:
    """SHA-256 over a compact-JSON header concatenated with the raw canonical bytes.

    The header binds the delta to WHO produced it, in WHICH round, over HOW MUCH data —
    so a delta cannot be replayed by another client or into another round.
    """
    raw = canon_update(delta)
    header = json.dumps(
        {"client_id": int(client_id), "num_samples": int(num_samples),
         "numel": len(raw) // 4, "round": int(round_num)},
        separators=(",", ":"), sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(header + raw).hexdigest()


# ── Merkle over the round's client updates ───────────────────────────────────
def _entries(leaves: Sequence) -> List[Tuple[str, str]]:
    """Normalise [(client_id, digest), ...] and reject duplicate client ids."""
    out = [(str(cid), str(dig)) for cid, dig in leaves]
    names = [c for c, _ in out]
    if len(set(names)) != len(names):
        raise ValueError("duplicate client_id in round leaves")
    return out


def leaf_hashes(leaves: Sequence) -> List[str]:
    """Ordered leaf hashes; position is bound in, so order is part of the commitment."""
    app = _backend()
    return [app._manifest_leaf(i, cid, dig) for i, (cid, dig) in enumerate(_entries(leaves))]


def round_merkle_root(leaves: Sequence) -> Tuple[str, int]:
    """(root_hex, leaf_count) for an ordered [(client_id, digest), ...] round manifest."""
    ent = _entries(leaves)
    if not ent:
        return None, 0
    app = _backend()
    order = [cid for cid, _ in ent]
    root, ls, _ = app.compute_layer_merkle(dict(ent), order)
    return root, len(ls)


def inclusion_proof(leaves: Sequence, index: int) -> List[dict]:
    """Sibling path for leaf `index`: [{"hash": hex, "right": bool}, ...]."""
    ent = _entries(leaves)
    if not 0 <= index < len(ent):
        raise IndexError(f"leaf index {index} out of range for {len(ent)} leaves")
    return _backend().merkle_inclusion_proof(leaf_hashes(ent), index)


def verify_proof(leaf: str, path: Iterable[dict], root: str) -> bool:
    """Fold a sibling path from a leaf hash and compare against the anchored root.

    This is the VERIFIER side, which backend/app.py does not implement in Python (the
    browser folds proofs in JS). It mirrors that fold exactly: ``right`` means the sibling
    sits to the right, so the parent is SHA-256(current || sibling).
    """
    h = str(leaf)
    for step in path:
        sib = str(step["hash"])
        pair = (h + sib) if step.get("right") else (sib + h)
        h = hashlib.sha256(pair.encode()).hexdigest()
    return h == root


def commit_round(updates, round_num: int):
    """Digest every ClientUpdate, build the round manifest, return everything needed to
    anchor and to serve proofs later.

    Returns {"leaves": [{client_id, digest, leaf, proof}], "root", "leaf_count",
             "bytes_committed", "digest_ms", "merkle_ms"}.
    """
    import time
    t0 = time.perf_counter()
    ent, nbytes = [], 0
    for u in updates:
        raw = canon_update(u.delta)
        nbytes += len(raw)
        ent.append((str(u.client_id),
                    update_digest(u.client_id, round_num, u.delta, u.num_samples)))
    digest_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    root, leaf_count = round_merkle_root(ent)
    lh = leaf_hashes(ent)
    proofs = [inclusion_proof(ent, i) for i in range(len(ent))]
    merkle_ms = (time.perf_counter() - t1) * 1000.0

    return {
        "root": root,
        "leaf_count": leaf_count,
        "leaves": [{"index": i, "client_id": cid, "digest": dig,
                    "leaf": lh[i], "proof": proofs[i]}
                   for i, (cid, dig) in enumerate(ent)],
        "bytes_committed": int(nbytes),
        "digest_ms": digest_ms,
        "merkle_ms": merkle_ms,
    }
