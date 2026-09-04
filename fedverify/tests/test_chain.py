"""Phase-3 chain commitment tests.

The load-bearing ones are the two forgery cases from docs/TAMPER_DETECTION.md §3.0:
(a) a tampered leaf presented with the original proof fails the fold, and (b) a fully
self-consistent forged tree — whose own proofs verify against its own root — still fails
because that root is not the one anchored. (b) is the case a naive implementation misses.
"""
import json
import math
import os
import subprocess
import sys

import pytest
import torch

from fedverify.chain import commitment as C
from fedverify.chain.anchor import ChainAnchor
from fedverify.config import FLConfig
from fedverify.core.runner import run, strip_timings

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _delta(i, n=64):
    return torch.randn(n, generator=torch.Generator().manual_seed(i))


def _entries(k=5, rnd=1):
    return [(i, C.update_digest(i, rnd, _delta(i), 100 + i)) for i in range(k)]


class _U:
    """Minimal stand-in for ClientUpdate."""
    def __init__(self, i, rnd=1):
        self.client_id, self.delta, self.num_samples = i, _delta(i), 100 + i


# ── canonical bytes ──────────────────────────────────────────────────────────
def test_canon_is_float32_little_endian_c_contiguous():
    t = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = C.canon_update(t)
    assert len(b) == 6 * 4                                  # float32, not float64
    import numpy as np
    assert np.array_equal(np.frombuffer(b, dtype="<f4"), np.arange(6, dtype="<f4"))


def test_canon_is_order_stable_for_non_contiguous_input():
    """A transposed (non-contiguous) tensor must serialise in C order, not memory order."""
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert C.canon_update(t.T) == C.canon_update(t.T.contiguous())


def test_canon_bytes_identical_across_two_subprocesses():
    """Byte-stability across processes is what makes a digest portable between clients."""
    code = ("import torch,hashlib,sys;sys.path.insert(0,%r);"
            "from fedverify.chain.commitment import canon_update;"
            "t=torch.arange(1000,dtype=torch.float32)/7.0;"
            "print(hashlib.sha256(canon_update(t)).hexdigest())" % REPO)
    a = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    b = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert a.returncode == 0, a.stderr
    assert a.stdout.strip() == b.stdout.strip() != ""


def test_digest_binds_client_round_and_sample_count():
    d = _delta(0)
    base = C.update_digest(0, 1, d, 100)
    assert C.update_digest(1, 1, d, 100) != base          # replay as another client
    assert C.update_digest(0, 2, d, 100) != base          # replay into another round
    assert C.update_digest(0, 1, d, 101) != base          # restated sample count


# ── Merkle root, proofs, forgeries ───────────────────────────────────────────
def test_valid_proof_verifies_for_every_leaf():
    ent = _entries(5)
    root, count = C.round_merkle_root(ent)
    assert count == 5 and len(root) == 64
    lh = C.leaf_hashes(ent)
    assert all(C.verify_proof(lh[i], C.inclusion_proof(ent, i), root) for i in range(5))


def test_odd_leaf_count_still_verifies():
    """Duplicate-last is the odd-level rule; proofs must survive it."""
    ent = _entries(3)
    root, count = C.round_merkle_root(ent)
    lh = C.leaf_hashes(ent)
    assert count == 3
    assert all(C.verify_proof(lh[i], C.inclusion_proof(ent, i), root) for i in range(3))


def test_flipped_byte_in_leaf_breaks_the_proof():
    ent = _entries(5)
    root, _ = C.round_merkle_root(ent)
    lh = C.leaf_hashes(ent)
    flipped = lh[2][:-1] + ("0" if lh[2][-1] != "0" else "1")
    assert not C.verify_proof(flipped, C.inclusion_proof(ent, 2), root)


def test_forgery_a_tampered_digest_with_original_proof_fails_fold():
    """TAMPER_DETECTION §3.0 case (a)."""
    ent = _entries(5)
    root, _ = C.round_merkle_root(ent)
    tampered = list(ent); tampered[2] = (2, "f" * 64)
    bad_leaf = C.leaf_hashes(tampered)[2]
    assert not C.verify_proof(bad_leaf, C.inclusion_proof(ent, 2), root)


def test_forgery_b_self_consistent_forged_tree_fails_root_comparison():
    """TAMPER_DETECTION §3.0 case (b): internally valid, but not the anchored root."""
    ent = _entries(5)
    anchored, _ = C.round_merkle_root(ent)
    forged_entries = list(ent); forged_entries[2] = (2, "f" * 64)
    forged_root, _ = C.round_merkle_root(forged_entries)
    forged_leaf = C.leaf_hashes(forged_entries)[2]
    # the forged tree is self-consistent ...
    assert C.verify_proof(forged_leaf, C.inclusion_proof(forged_entries, 2), forged_root)
    # ... and is caught only by comparing against what was anchored
    assert forged_root != anchored


def test_swapping_two_clients_changes_the_root():
    """Position is bound into the leaf, so order is part of the commitment."""
    ent = _entries(5)
    root, _ = C.round_merkle_root(ent)
    swapped = [ent[1], ent[0]] + ent[2:]
    assert C.round_merkle_root(swapped)[0] != root


def test_duplicate_client_ids_rejected():
    with pytest.raises(ValueError, match="duplicate client_id"):
        C.round_merkle_root([(0, "a" * 64), (0, "b" * 64)])


def test_proof_index_out_of_range():
    with pytest.raises(IndexError):
        C.inclusion_proof(_entries(3), 7)


def test_round_root_matches_the_canonical_backend_merkle():
    """The root must come from backend/app.py, not a re-implementation."""
    sys.path.insert(0, os.path.join(REPO, "backend"))
    import app
    ent = _entries(4)
    expected, _leaves, _order = app.compute_layer_merkle(
        {c: d for c, d in ((str(c), d) for c, d in ent)}, [str(c) for c, _ in ent])
    assert C.round_merkle_root(ent)[0] == expected


# ── anchor backends ──────────────────────────────────────────────────────────
def test_mock_anchor_is_deterministic():
    ent = _entries(4)
    root, n = C.round_merkle_root(ent)
    a, b = ChainAnchor("mock"), ChainAnchor("mock")
    assert a.anchor_round("r", 1, root, n)["txid"] == b.anchor_round("r", 1, root, n)["txid"]


def test_anchor_record_shape():
    ent = _entries(4)
    root, n = C.round_merkle_root(ent)
    rec = ChainAnchor("mock").anchor_round("r", 1, root, n)
    for k in ("latency_ms", "txid", "fee", "bytes_written"):
        assert k in rec, k


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="unknown chain backend"):
        ChainAnchor("ethereum")


def test_local_backend_writes_both_new_tx_types_and_chain_stays_valid():
    cm = C.commit_round([_U(i) for i in range(3)], 1)
    a = ChainAnchor("local", difficulty=2)
    a.anchor_round("r", 1, cm["root"], cm["leaf_count"], cm["leaves"])
    txs = a.chain.chain[-1].transactions
    assert {t["type"] for t in txs} == {"fl_client_update", "fl_round_commit"}
    valid = a.chain.is_chain_valid()
    assert (valid.get("valid") if isinstance(valid, dict) else valid) is True


def test_checkpoint_covers_all_round_roots_so_far():
    a = ChainAnchor("mock", checkpoint_every=2)
    for r in (1, 2, 3, 4):
        cm = C.commit_round([_U(i) for i in range(3)], r)
        a.anchor_round("r", r, cm["root"], cm["leaf_count"], cm["leaves"])
        if a.due_for_checkpoint(r):
            a.checkpoint("r", r)
    assert [c["rounds_covered"] for c in a.checkpoints] == [2, 4]


# ── runner integration ───────────────────────────────────────────────────────
TINY = dict(dataset="mnist", num_clients=3, rounds=2, local_epochs=1,
            batch_size=32, alpha=math.inf, seed=0, exp="test")


def _run(tmp_path, **kw):
    cfg = FLConfig(out_dir=str(tmp_path), **{**TINY, **kw})
    return cfg, run(cfg, limit_train=200, limit_test=200, progress=False)


def _recs(cfg):
    with open(os.path.join(cfg.run_dir, "rounds.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def test_commit_off_by_default_leaves_the_record_untouched(tmp_path):
    """Phase-1/2 cells must not change shape now that Phase 3 exists."""
    cfg, _ = _run(tmp_path)
    rec = _recs(cfg)[0]
    assert "commit" not in rec
    assert not os.path.exists(os.path.join(cfg.run_dir, "commitments.jsonl"))


def test_commit_on_adds_block_and_writes_proofs(tmp_path):
    cfg, out = _run(tmp_path, commit=True, chain_backend="mock", checkpoint_every=2)
    rec = _recs(cfg)[0]
    cm = rec["commit"]
    for k in ("root", "leaf_count", "txid", "digest_ms", "merkle_ms",
              "anchor_ms", "aggregate_ms", "bytes_committed"):
        assert k in cm, k
    assert cm["leaf_count"] == cfg.num_clients
    lines = [json.loads(l) for l in
             open(os.path.join(cfg.run_dir, "commitments.jsonl")) if l.strip()]
    assert len(lines) == cfg.rounds
    assert len(lines[0]["leaves"]) == cfg.num_clients
    assert out["lineage"] == [r["commit"]["root"] for r in _recs(cfg)]


def test_committed_proofs_verify_against_the_recorded_root(tmp_path):
    cfg, _ = _run(tmp_path, commit=True, chain_backend="mock")
    line = json.loads(open(os.path.join(cfg.run_dir, "commitments.jsonl")).readline())
    assert all(C.verify_proof(lf["leaf"], lf["proof"], line["root"])
               for lf in line["leaves"])


def test_anchor_ms_is_separate_from_train_wall_s(tmp_path):
    cfg, _ = _run(tmp_path, commit=True, chain_backend="mock")
    rec = _recs(cfg)[0]
    assert "anchor_ms" in rec["commit"] and "anchor_ms" not in rec
    assert rec["train_wall_s"] > 0


def test_commitment_does_not_change_the_science(tmp_path):
    """Turning commitment on must not perturb training at all."""
    off_cfg, _ = _run(tmp_path / "off")
    on_cfg, _ = _run(tmp_path / "on", commit=True, chain_backend="mock")
    off, on = _recs(off_cfg), _recs(on_cfg)
    for a, b in zip(off, on):
        for k in ("test_acc", "test_loss", "macro_f1", "mean_train_loss"):
            assert a[k] == b[k], k


def test_same_seed_reproduces_commitment_after_stripping_timings(tmp_path):
    a_cfg, _ = _run(tmp_path / "a", commit=True, chain_backend="mock")
    b_cfg, _ = _run(tmp_path / "b", commit=True, chain_backend="mock")
    a, b = [strip_timings(r) for r in _recs(a_cfg)], [strip_timings(r) for r in _recs(b_cfg)]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ── browser/backend cross-validation ─────────────────────────────────────────
def _extract_js(*names):
    """Pull the named functions VERBATIM out of frontend/index.html.

    Same technique as evaluation/cross_validate.py: the test must exercise the SHIPPED
    browser code, not a re-typed copy of it, or it proves nothing about what users run.
    """
    import re
    html = open(os.path.join(REPO, "frontend", "index.html"), encoding="utf-8").read()
    out = []
    for n in names:
        m = re.search(r"(async\s+)?function\s+" + re.escape(n) + r"\s*\([^)]*\)\s*\{",
                      html)
        assert m, f"{n} not found in frontend/index.html"
        i, depth = m.end() - 1, 0
        while i < len(html):
            if html[i] == "{":
                depth += 1
            elif html[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out.append(html[m.start():i + 1])
    return "\n".join(out)


@pytest.mark.skipif(not __import__("shutil").which("node"), reason="node not installed")
def test_browser_leaf_encoding_matches_the_backend():
    """The trustless claim depends on the browser deriving the SAME leaf we anchored.

    The browser folds a proof it does not trust and compares against a root read from the
    public indexer. If its leaf encoding differed from ours by even one byte, every proof
    would fail — or worse, a wrong one could pass.
    """
    import json as _json
    import subprocess as _sp
    js = _extract_js("sha256Fallback", "bvSha256Hex", "bvMerkleLeaf", "bvFoldProof")
    ent = _entries(5)
    lh = C.leaf_hashes(ent)
    root, _ = C.round_merkle_root(ent)
    proof = C.inclusion_proof(ent, 2)

    # client_id is stringified on both sides: the backend leaf is JSON([i, str(cid),
    # digest]) and the frontend calls bvMerkleLeaf(p.index, String(p.clientId), ...).
    harness = """
const crypto = require('crypto');
// bvSha256Hex checks window.crypto.subtle and falls back to sha256Fallback when the page
// is not in a secure context. Provide `window` so the native path is exercised here.
global.window = { crypto: { subtle: { digest: async (_a, buf) =>
  crypto.createHash('sha256').update(Buffer.from(buf)).digest().buffer } } };
global.crypto = global.window.crypto;
global.TextEncoder = require('util').TextEncoder;
%s
(async () => {
  const inp = %s;
  const leaf = await bvMerkleLeaf(inp.index, inp.client_id, inp.digest);
  const folded = await bvFoldProof(leaf, inp.proof);
  console.log(JSON.stringify({leaf, folded}));
})();
""" % (js, _json.dumps({"index": 2, "client_id": str(ent[2][0]), "digest": ent[2][1],
                        "proof": proof}))

    r = _sp.run(["node", "-e", harness], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    got = _json.loads(r.stdout.strip())
    assert got["leaf"] == lh[2], "browser leaf encoding diverges from the backend"
    assert got["folded"] == root, "browser proof fold does not reach the anchored root"


@pytest.mark.skipif(not __import__("shutil").which("node"), reason="node not installed")
def test_browser_rejects_a_tampered_digest():
    """The browser fold must FAIL on a forged digest, or the tab is decorative."""
    import json as _json
    import subprocess as _sp
    js = _extract_js("sha256Fallback", "bvSha256Hex", "bvMerkleLeaf", "bvFoldProof")
    ent = _entries(5)
    root, _ = C.round_merkle_root(ent)
    proof = C.inclusion_proof(ent, 2)

    harness = """
const crypto = require('crypto');
// bvSha256Hex checks window.crypto.subtle and falls back to sha256Fallback when the page
// is not in a secure context. Provide `window` so the native path is exercised here.
global.window = { crypto: { subtle: { digest: async (_a, buf) =>
  crypto.createHash('sha256').update(Buffer.from(buf)).digest().buffer } } };
global.crypto = global.window.crypto;
global.TextEncoder = require('util').TextEncoder;
%s
(async () => {
  const i = %s;
  const leaf = await bvMerkleLeaf(i.index, i.client_id, i.digest);
  console.log(JSON.stringify({folded: await bvFoldProof(leaf, i.proof)}));
})();
""" % (js, _json.dumps({"index": 2, "client_id": str(ent[2][0]), "digest": "f" * 64,
                        "proof": proof}))

    r = _sp.run(["node", "-e", harness], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert _json.loads(r.stdout.strip())["folded"] != root


@pytest.mark.skipif(not __import__("shutil").which("node"), reason="node not installed")
def test_browser_leaf_encoding_matches_without_web_crypto():
    """The browser must still verify when crypto.subtle is unavailable.

    crypto.subtle exists only in a SECURE CONTEXT — https, http://localhost or
    http://127.0.0.1. Serving the UI from http://0.0.0.0:8080 leaves it undefined, which
    threw "Cannot read properties of undefined (reading 'digest')" and made every
    inclusion proof fail. This pins the pure-JS fallback against the Python backend.
    """
    import json as _json
    import subprocess as _sp
    js = _extract_js("sha256Fallback", "bvSha256Hex", "bvMerkleLeaf", "bvFoldProof")
    ent = _entries(5)
    lh = C.leaf_hashes(ent)
    root, _ = C.round_merkle_root(ent)
    proof = C.inclusion_proof(ent, 2)

    harness = """
// deliberately NO window.crypto — this is exactly the http://0.0.0.0 case
global.window = {};
global.TextEncoder = require('util').TextEncoder;
%s
(async () => {
  const i = %s;
  const leaf = await bvMerkleLeaf(i.index, i.client_id, i.digest);
  console.log(JSON.stringify({leaf, folded: await bvFoldProof(leaf, i.proof)}));
})();
""" % (js, _json.dumps({"index": 2, "client_id": str(ent[2][0]), "digest": ent[2][1],
                        "proof": proof}))

    r = _sp.run(["node", "-e", harness], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    got = _json.loads(r.stdout.strip())
    assert got["leaf"] == lh[2], "fallback SHA-256 diverges from the backend"
    assert got["folded"] == root, "fallback fold does not reach the anchored root"
