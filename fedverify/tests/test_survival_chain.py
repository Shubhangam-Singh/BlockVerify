"""Revision 2 Phase 3: content IDs, signed submissions, canonical bytes, per-round commitment, chain metrics."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
pytest.importorskip("lifelines")

from fedverify.chain import contentid as CID  # noqa: E402
from fedverify.chain import signing as SG  # noqa: E402
from fedverify.chain.commitment import round_merkle_root, verify_proof  # noqa: E402
from fedverify.core.runner import strip_timings  # noqa: E402
from fedverify.survival import data as D  # noqa: E402
from fedverify.survival import heart as H  # noqa: E402
from fedverify.survival import runner  # noqa: E402
from fedverify.survival import serialize as S  # noqa: E402
from fedverify.survival.chain_round import ChainSession, LOGICAL_TS_BASE  # noqa: E402
from fedverify.survival.config import HEART_HOSPITALS, SurvivalConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
needs_heart = pytest.mark.skipif(not H.DEFAULT_DATA.exists(), reason="heart CSV (gitignored) not present")
needs_tcga = pytest.mark.skipif(not D.DEFAULT_DATA.exists(), reason="TCGA workbook (gitignored) not present")


# ── synthetic models for byte-level tests (no patient data) ──────────────────
def _toy(seed=0, n=240, d=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.7 * X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
    return X, y


def _forest(seed=0):
    from sklearn.ensemble import RandomForestClassifier
    X, y = _toy(seed)
    return RandomForestClassifier(n_estimators=10, max_depth=5, min_samples_leaf=3, class_weight="balanced_subsample",
                                  n_jobs=1, random_state=seed).fit(X, y), X


def _booster(seed=0, objective="binary:logistic"):
    import xgboost as xgb
    X, y = _toy(seed)
    dm = xgb.DMatrix(X, nthread=1)
    if objective == "binary:logistic":
        dm.set_label(y.astype(np.float32))
        params = {"objective": objective, "base_score": 0.5}
    else:
        t = np.exp(5 + X[:, 0]).astype(np.float64)
        dm.set_float_info("label_lower_bound", t)
        dm.set_float_info("label_upper_bound", np.where(y == 1, t, np.inf))
        params = {"objective": "survival:aft", "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.0,
                  "base_score": 1.0}
    dm.set_base_margin(np.zeros(len(X), dtype=np.float32))
    params.update(tree_method="hist", max_depth=3, eta=0.1, subsample=0.8, nthread=1, seed=seed, verbosity=0)
    return xgb.train(params, dm, num_boost_round=5), X


def _zero_margin(X):
    import xgboost as xgb
    dm = xgb.DMatrix(X, nthread=1)
    dm.set_base_margin(np.zeros(len(X), dtype=np.float32))
    return dm


# ── content IDs ──────────────────────────────────────────────────────────────
def test_content_id_matches_the_ipfs_raw_cidv1_of_hello_world():
    # `ipfs add --cid-version 1 --raw-leaves` of the 11 bytes "hello world"
    assert CID.content_id(b"hello world") == "bafkreifzjut3te2nhyekklss27nh3k72ysco7y32koao5eei66wof36n5e"


def test_content_id_round_trip_and_tamper():
    data = b"\x00\x01payload"
    cid = CID.content_id(data)
    assert cid.startswith("bafkrei") and CID.decode_cid(cid) == hashlib.sha256(data).digest()
    assert CID.verify_cid(data, cid) and not CID.verify_cid(data + b"x", cid)
    assert not CID.verify_cid(data, "Qm-not-a-cidv1")


def test_ipfs_real_mode_degrades_gracefully_without_a_daemon():
    out = CID.ipfs_add(b"x", api="http://127.0.0.1:9", timeout=0.5)     # nothing listens on port 9
    assert out["ok"] is False and "error" in out


# ── signed submissions ───────────────────────────────────────────────────────
def _signed(h="BH", counter=0, ts=LOGICAL_TS_BASE + 1, seed=0, **over):
    fields = {"run_id": "r", "round": 1, "hospital_id": h, "cid": CID.content_id(b"u"),
              "digest": hashlib.sha256(b"u").hexdigest(), "timestamp": ts,
              "txn_id": SG.make_txn_id(h, counter, ts)}
    fields.update(over)
    return SG.sign_submission(SG.derive_test_key(seed, h), fields)


def _registry(*hospitals, seed=0):
    reg = SG.KeyRegistry()
    for h in hospitals:
        reg.register(h, SG.public_key_bytes(SG.derive_test_key(seed, h)))
    return reg


def test_valid_submission_verifies_and_cannot_be_replayed():
    reg, sub = _registry("BH"), _signed()
    assert reg.verify(dict(sub)) == (True, None)
    ok, reason = reg.verify(dict(sub))                          # the same txn_id again
    assert not ok and "counter" in reason


@pytest.mark.parametrize("field,value", [("timestamp", LOGICAL_TS_BASE + 2), ("txn_id", "00" * 32), ("round", 2),
                                         ("digest", "ff" * 32), ("cid", "bafkreiaaaa")])
def test_signature_covers_every_field(field, value):
    sub = dict(_signed(), **{field: value})
    ok, reason = _registry("BH").verify(sub)
    assert not ok and "invalid signature" in reason


def test_tampered_signature_is_rejected():
    sub = _signed()
    sig = bytearray(bytes.fromhex(sub["signature"]))
    sig[0] ^= 0x01
    ok, reason = _registry("BH").verify(dict(sub, signature=sig.hex()))
    assert not ok and "invalid signature" in reason


def test_missing_signature_unregistered_key_and_wrong_hospital_key_are_rejected():
    reg = _registry("BH", "A2")
    assert reg.verify({k: v for k, v in _signed().items() if k != "signature"})[1] == "missing signature"
    assert "unregistered" in reg.verify(_signed(h="E2"))[1]
    # a VALID signature, but made with A2's key on a submission that claims to come from BH
    forged = SG.sign_submission(SG.derive_test_key(0, "A2"), {k: v for k, v in _signed().items() if k != "signature"})
    assert "invalid signature" in reg.verify(forged)[1]


def test_counter_only_moves_forward_but_tolerates_a_lost_submission():
    reg = _registry("BH")
    assert reg.verify(_signed(counter=3))[0]                     # counters 0-2 were lost in transit
    assert not reg.verify(_signed(counter=1, ts=LOGICAL_TS_BASE + 9))[0]   # an older counter never verifies again
    assert reg.verify(_signed(counter=4, ts=LOGICAL_TS_BASE + 9))[0]


# ── canonical bytes and numpy predictors ─────────────────────────────────────
def test_logistic_payload_round_trips_exactly():
    import torch
    delta = torch.randn(31, generator=torch.Generator().manual_seed(0))
    hdr, back = S.deserialize(S.serialize_lr("run", 3, "BH", 116, delta))
    assert hdr == {"family": "lr", "run_id": "run", "round": 3, "hospital": "BH", "num_samples": 116, "format": 1,
                   "numel": 31}
    assert torch.equal(back, delta.float())


def test_forest_rebuilt_from_bytes_predicts_exactly_like_sklearn():
    forest, X = _forest()
    _, model = S.deserialize(S.serialize_rf("run", 1, "BH", 240, forest, 1))
    ours = np.stack([t.predict_proba(X)[:, 1] for t in model.trees])
    theirs = np.stack([e.predict_proba(X)[:, 1] for e in forest.estimators_])
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("objective", ["binary:logistic", "survival:aft"])
def test_xgboost_rebuilt_from_bytes_gives_the_same_margins(objective):
    booster, X = _booster(objective=objective)
    _, model = S.deserialize(S.serialize_xgb("run", 1, "BH", 240, booster, objective))
    np.testing.assert_array_equal(model.tree_sum(X), booster.predict(_zero_margin(X), output_margin=True).astype(np.float64))


def test_payload_bytes_are_identical_across_two_processes():
    code = ("import hashlib, sys; sys.path.insert(0, %r); "
            "from fedverify.tests.test_survival_chain import _forest, _booster; "
            "from fedverify.survival import serialize as S; "
            "f, _ = _forest(); b, _ = _booster(); "
            "print(hashlib.sha256(S.serialize_rf('run', 1, 'BH', 240, f, 1)).hexdigest(), "
            "hashlib.sha256(S.serialize_xgb('run', 1, 'BH', 240, b, 'binary:logistic')).hexdigest())") % str(REPO)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    outs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO),
                           check=True).stdout.strip() for _ in range(2)]
    assert outs[0] == outs[1] and len(outs[0].split()) == 2


def test_flipping_one_byte_of_any_tree_changes_the_root():
    forest, _ = _forest()
    booster, _ = _booster()
    payloads = {"BH": S.serialize_rf("run", 1, "BH", 240, forest, 1),
                "A2": S.serialize_xgb("run", 1, "A2", 240, booster, "binary:logistic")}
    entries = [(h, hashlib.sha256(p).hexdigest()) for h, p in payloads.items()]
    root, _ = round_merkle_root(entries)
    for h, p in payloads.items():
        hdr, body = S.split_payload(p)
        for pos in (len(p) - len(body), len(p) - 1):           # first and last byte of the tree data
            flipped = bytearray(p)
            flipped[pos] ^= 0x01
            changed = [(k, hashlib.sha256(bytes(flipped) if k == h else payloads[k]).hexdigest()) for k in payloads]
            assert round_merkle_root(changed)[0] != root


def test_the_header_binds_the_round_so_a_digest_cannot_be_replayed():
    forest, _ = _forest()
    a = hashlib.sha256(S.serialize_rf("run", 1, "BH", 240, forest, 1)).hexdigest()
    b = hashlib.sha256(S.serialize_rf("run", 2, "BH", 240, forest, 1)).hexdigest()
    assert a != b


# ── the commit pipeline ──────────────────────────────────────────────────────
def _session(backend="mock", seed=0):
    return ChainSession(SurvivalConfig(dataset="heart", family="lr", method="chain_fedavg_lr", chain_backend=backend),
                        run_seed=seed)


def _builders(rnd):
    import torch
    return {h: (lambda h=h: S.serialize_lr("run", rnd, h, 100 + i, torch.full((5,), float(i))))
            for i, h in enumerate(HEART_HOSPITALS)}


def test_commit_round_roots_proofs_and_forged_root():
    fc = _session().for_fold("heart_chain_fedavg_lr_s0_r0_f0")
    res = fc.commit_round(1, _builders(1))
    assert res.accepted == list(HEART_HOSPITALS) and res.record["rejected"] == []
    entries = [(l["client_id"], l["digest"]) for l in res.commitment["leaves"]]
    assert round_merkle_root(entries) == (res.record["root"], 4)
    for leaf in res.commitment["leaves"]:
        assert verify_proof(leaf["leaf"], leaf["proof"], res.record["root"])
        assert not verify_proof(leaf["leaf"], leaf["proof"], "0" * 64)
    swapped = [entries[1], entries[0]] + entries[2:]
    assert round_merkle_root(swapped)[0] != res.record["root"]
    for k in ("serialize_ms", "digest_ms", "cid_ms", "sign_ms", "verify_ms", "merkle_ms", "anchor_ms"):
        assert res.record[k] >= 0.0


def test_a_tampered_submission_is_rejected_with_a_reason_and_left_out():
    def tamper(h, sub, payload):
        if h == "Hungary":
            sig = bytearray(bytes.fromhex(sub["signature"]))
            sig[-1] ^= 0x01
            return dict(sub, signature=sig.hex()), payload
        if h == "Cleveland":
            return sub, payload[:-1] + bytes([payload[-1] ^ 0x01])
        return sub, payload
    res = _session().for_fold("f").commit_round(1, _builders(1), tamper=tamper)
    assert res.accepted == ["VA Long Beach", "Switzerland"]
    reasons = {r["hospital"]: r["reason"] for r in res.record["rejected"]}
    assert "invalid signature" in reasons["Hungary"] and "digest" in reasons["Cleveland"]
    assert [l["client_id"] for l in res.commitment["leaves"]] == ["VA Long Beach", "Switzerland"]


def test_previous_block_check_skips_the_round_when_the_ledger_was_tampered():
    s = _session("local")
    fc = s.for_fold("f")
    fc.commit_round(1, _builders(1))
    assert fc.check_previous() == {}
    s.chain.chain[-1].transactions[0]["digest"] = "ff" * 32      # rewrite history on the ledger
    reasons = fc.check_previous()
    assert set(reasons) == set(HEART_HOSPITALS)
    assert all("previous block verification failed" in r for r in reasons.values())


def test_same_seed_same_roots_and_keys_registered_on_the_ledger():
    roots = []
    for _ in range(2):
        s = _session("local")
        fc = s.for_fold("f")
        roots.append([fc.commit_round(r, _builders(r)).record["root"] for r in (1, 2, 3)])
        first_block = s.chain.chain[1]
        assert sum(t["type"] == "fl_key_register" for t in first_block.transactions) == 4
    assert roots[0] == roots[1]
    blocks = s.block_log
    assert [b["block_index"] for b in blocks] == [1, 2, 3] and all(b["block_bytes"] > 0 for b in blocks)


# ── end to end through the runner ────────────────────────────────────────────
def _outputs(run_dir: Path) -> list:
    recs = []
    for p in sorted(run_dir.rglob("rounds.jsonl")):
        for l in p.read_text().splitlines():
            r = strip_timings(json.loads(l))
            r.pop("method")
            r.pop("commit", None)
            recs.append(r)
    return recs


@needs_heart
@pytest.mark.parametrize("family,base,chain", [("lr", "fedavg_lr", "chain_fedavg_lr"), ("rf", "fed_rf", "chain_fed_rf"),
                                               ("xgb", "fedxgb_bagging", "chain_fedxgb_bagging")])
def test_commit_only_methods_learn_exactly_what_their_base_method_learns(tmp_path, family, base, chain):
    """chain_* aggregates from the committed bytes; with every submission valid it must equal the base method."""
    kw = dict(dataset="heart", family=family, rounds=2, repeats=1)
    a = SurvivalConfig(method=base, out_dir=str(tmp_path / "a"), **kw)
    b = SurvivalConfig(method=chain, out_dir=str(tmp_path / "b"), **kw)
    sa, sb = runner.run(a, verbose=False), runner.run(b, verbose=False)
    assert sa["repeats"][0]["pooled"] == sb["repeats"][0]["pooled"]
    assert (Path(a.run_dir) / "repeat0" / "oof.jsonl").read_text() == (Path(b.run_dir) / "repeat0" / "oof.jsonl").read_text()
    assert _outputs(Path(a.run_dir)) == _outputs(Path(b.run_dir))


@needs_heart
def test_chain_run_writes_commitments_and_same_seed_same_roots(tmp_path):
    roots = []
    for sub in ("a", "b"):
        cfg = SurvivalConfig(dataset="heart", family="xgb", method="chain_fedxgb_bagging", rounds=2, repeats=1,
                             out_dir=str(tmp_path / sub))
        runner.run(cfg, verbose=False)
        lines = [json.loads(l) for l in (Path(cfg.run_dir) / "repeat0" / "fold0" / "commitments.jsonl").read_text().splitlines()]
        assert [c["round"] for c in lines] == [1, 2] and all(c["leaf_count"] == 4 for c in lines)
        rec = json.loads((Path(cfg.run_dir) / "repeat0" / "fold0" / "rounds.jsonl").read_text().splitlines()[0])
        assert rec["commit"]["root"] == lines[0]["root"] and rec["commit"]["rejected"] == []
        roots.append([c["root"] for c in lines])
    assert roots[0] == roots[1]


@needs_heart
def test_local_backend_logs_blocks_and_chain_metrics(tmp_path):
    cfg = SurvivalConfig(dataset="heart", family="lr", method="chain_fedavg_lr", rounds=2, repeats=1,
                         chain_backend="local", out_dir=str(tmp_path))
    runner.run(cfg, verbose=False)
    blocks = [json.loads(l) for l in (Path(cfg.run_dir) / "chain_blocks.jsonl").read_text().splitlines()]
    assert len(blocks) == cfg.folds * cfg.rounds                 # one shared chain: one block per committed round
    assert [b["block_index"] for b in blocks] == list(range(1, len(blocks) + 1))
    m = json.loads((Path(cfg.run_dir) / "chain_metrics.json").read_text())
    assert m["blocks"] == len(blocks) and m["difficulty"] == cfg.chain_difficulty
    assert m["latency_ms"] > 0 and m["throughput_bytes_per_s"] > 0 and m["send_rate_tx_per_min"] > 0


@needs_heart
def test_survival_commitments_pass_the_existing_fl_routes(tmp_path, monkeypatch):
    """The existing POST /api/fl/round/commit re-derives the root from Survival leaves and accepts it."""
    cfg = SurvivalConfig(dataset="heart", family="lr", method="chain_fedavg_lr", rounds=1, repeats=1,
                         out_dir=str(tmp_path))
    runner.run(cfg, verbose=False)
    c = json.loads((Path(cfg.run_dir) / "repeat0" / "fold0" / "commitments.jsonl").read_text().splitlines()[0])
    sys.path.insert(0, str(REPO / "backend"))
    import app as app_module
    import fl_routes
    monkeypatch.setattr(fl_routes, "FL_FILE", str(tmp_path / "fl_runs.json"))
    client = app_module.app.test_client()
    body = {"run_id": "heart_chain_test", "round": 1, "root": c["root"], "leaf_count": c["leaf_count"],
            "txid": c["txid"], "backend": c["backend"],
            "leaves": [{"client_id": l["client_id"], "digest": l["digest"]} for l in c["leaves"]]}
    assert client.post("/api/fl/round/commit", json=body).status_code == 201
    proof = client.get("/api/fl/proof/heart_chain_test/1/Hungary").get_json()
    assert proof["success"] and verify_proof(proof["leaf"], proof["proof"], c["root"])


@needs_heart
def test_profile_writes_both_views(tmp_path):
    cfg = SurvivalConfig(dataset="heart", family="lr", method="chain_fedavg_lr", rounds=1, repeats=1,
                         out_dir=str(tmp_path))
    path = runner.profile_run(cfg, verbose=False)
    text = path.read_text()
    assert "all code" in text and "FedVerify / BlockVerify code only" in text
    top = json.loads((path.parent / "profile_top10.json").read_text())
    assert 0 < len(top["fedverify_code"]) <= 10 and all({"tsub_s", "ttot_s"} <= set(r) for r in top["all_code"])
