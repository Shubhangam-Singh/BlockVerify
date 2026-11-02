"""Album commitments reuse the canonical Merkle of backend/app.py (design §6.2)."""
import hashlib
import json

from photoshield.provenance import merkle

from .helpers import hx

ITEMS = [(hx(f"id{i}", 16), hx(f"md{i}")) for i in range(5)]


def test_leaf_is_the_canonical_manifest_leaf_encoding():
    pid, md = ITEMS[2]
    want = hashlib.sha256(json.dumps([2, pid, md], separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()
    assert merkle.leaf(2, pid, md) == want


def test_every_proof_verifies_and_a_flipped_leaf_fails():
    root, leaves, n = merkle.album_root(ITEMS)
    assert n == 5
    proofs = merkle.album_proofs(ITEMS)
    for i in range(n):
        assert merkle.verify_proof(leaves[i], proofs[i], root)
    flipped = ("0" if leaves[1][0] != "0" else "1") + leaves[1][1:]
    assert not merkle.verify_proof(flipped, proofs[1], root)


def test_order_is_part_of_the_commitment():
    r1, _, _ = merkle.album_root(ITEMS)
    r2, _, _ = merkle.album_root(list(reversed(ITEMS)))
    assert r1 != r2


def test_duplicate_or_empty_album_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        merkle.album_root([ITEMS[0], ITEMS[0]])
    with pytest.raises(ValueError):
        merkle.album_root([])
