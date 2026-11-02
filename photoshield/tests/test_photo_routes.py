"""backend/photo_routes.py (design §6, §11). Algorand and the indexer are mocked."""
import json

import pytest

from photoshield.chain import client as cli_client
from photoshield.chain import notes as N
from photoshield.provenance import manifest as M
from photoshield.provenance import signing as S

from .helpers import FakeIndexer, hx, indexer_tx, make_manifest, signed_single


@pytest.fixture
def env(tmp_path, monkeypatch):
    import algorand_client
    import app as app_module
    import photo_routes
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"tau_near": 10, "tau_far": 20, "fpr_target": 0.01}))
    monkeypatch.setenv("PHOTOSHIELD_LEDGER_SYNC", "0")
    monkeypatch.setenv("PHOTOSHIELD_ANCHOR", "algorand")
    sent = []

    def fake_broadcast(note, wait=True):
        raw = N.encode(note)
        sent.append(note)
        return {"success": True, "txid": ("T%051d" % len(sent)), "round": 1000 + len(sent),
                "fee": 1000, "bytes_written": len(raw)}
    monkeypatch.setattr(algorand_client, "broadcast_photo_anchor", fake_broadcast)
    monkeypatch.setattr(algorand_client, "broadcast_hash_to_algorand",
                        lambda *a, **k: pytest.fail("photos must never use the model/contract path"))
    photo_routes.reset_for_tests(str(tmp_path / "reg.json"), str(tmp_path / "cache.json"), str(cal))
    app_module.app.config["TESTING"] = True
    import auth
    c = app_module.app.test_client()
    yield {"c": c, "sent": sent, "tok": {"alice": auth._make_token("alice"), "bob": auth._make_token("bob")},
           "routes": photo_routes, "tmp": tmp_path}
    photo_routes.reset_for_tests()


def _h(env, user="alice"):
    return {"Authorization": f"Bearer {env['tok'][user]}"}


def _payload(man, seed=b"\x11" * 32, privacy="full"):
    sk, pk = S.new_key(seed)
    return cli_client.registration_payload(man, sk, pk, privacy), pk, sk


def _register(env, man=None, user="alice", seed=b"\x11" * 32, privacy="full"):
    man = man or make_manifest()
    body, pk, sk = _payload(man, seed, privacy)
    return env["c"].post("/api/photo/register", json=body, headers=_h(env, user)), man, pk, sk


def test_register_happy_path_anchors_a_signed_note(env):
    r, man, pk, _ = _register(env)
    d = r.get_json()
    assert r.status_code == 200 and d["success"], d
    (note,) = env["sent"]
    assert note["md"] == M.manifest_digest(man) and note["pk"] == pk
    assert note["po"] == man["original"]["phash"]
    assert "own" not in note and "alice" not in json.dumps(note)        # no username on chain
    assert d["photoId"] == M.photo_id(note["md"])


def test_auth_pow_and_signature_are_enforced(env):
    man = make_manifest()
    body, _, _ = _payload(man)
    assert env["c"].post("/api/photo/register", json=body).status_code == 401
    bad = dict(body, powNonce=None)
    assert env["c"].post("/api/photo/register", json=bad, headers=_h(env)).status_code == 400
    bad = dict(body, powNonce=body["powNonce"] + 1)
    r = env["c"].post("/api/photo/register", json=bad, headers=_h(env))
    assert r.status_code == 400 and "Proof-of-Work" in r.get_json()["error"]
    bad = dict(body, sig=("0" if body["sig"][0] != "0" else "1") + body["sig"][1:])
    r = env["c"].post("/api/photo/register", json=bad, headers=_h(env))
    assert r.status_code == 400 and "signature" in r.get_json()["error"]
    assert env["sent"] == []


def test_invalid_manifest_and_xss_title_are_rejected(env):
    man = make_manifest()
    man["title"] = "<img src=x onerror=alert(1)>"
    body = {"manifest": man, "powNonce": 0, "pk": "0" * 64, "sig": "0" * 128}
    r = env["c"].post("/api/photo/register", json=body, headers=_h(env))
    assert r.status_code == 400 and "title" in r.get_json()["error"]


def test_duplicate_by_another_key_is_409_and_same_key_is_idempotent(env):
    r1, man, _, _ = _register(env)
    first = r1.get_json()
    r2, _, _, _ = _register(env, man, user="alice")
    assert r2.status_code == 200 and r2.get_json()["duplicate"]
    r3, _, _, _ = _register(env, man, user="bob", seed=b"\x22" * 32)
    d = r3.get_json()
    assert r3.status_code == 409 and d["algoTxId"] == first["algoTxId"]
    assert len(env["sent"]) == 1


def test_rate_limit(env, monkeypatch):
    monkeypatch.setattr(env["routes"], "WRITE_LIMIT", 2)
    for i in range(2):
        assert _register(env, make_manifest(seed=f"r{i}"))[0].status_code == 200
    assert _register(env, make_manifest(seed="r9"))[0].status_code == 429


def _verify(env, sha, phash=None):
    return env["c"].post("/api/photo/verify", json={"sha256": sha, "phash": phash}).get_json()


def _flip(ph, bits):
    v = int(ph, 16)
    for b in range(bits):
        v ^= 1 << b
    return f"{v:016x}"


def test_verify_returns_each_of_the_five_verdicts(env):
    _, man, _, _ = _register(env)
    o, c = man["original"], man["cloaked"]
    a = _verify(env, o["sha256"], o["phash"])
    assert a["verdict"] == "EXACT_ORIGINAL" and a["ownerUser"] == "alice"
    assert _verify(env, c["sha256"], c["phash"])["verdict"] == "EXACT_CLOAKED"
    d = _verify(env, hx("other"), _flip(o["phash"], 4))
    assert d["verdict"] == "DERIVED" and d["distance"] == 4 and d["warning"]
    assert "ownerUser" not in d                                     # no username for near matches
    s = _verify(env, hx("other2"), _flip(o["phash"], 15))
    assert s["verdict"] in ("SIMILAR", "DERIVED") and s["distance"] <= 15
    assert _verify(env, hx("x"), _flip(o["phash"], 40))["verdict"] in ("UNKNOWN", "SIMILAR")
    assert _verify(env, hx("y"))["verdict"] == "UNKNOWN"


def test_similar_band_is_between_the_thresholds(env):
    _, man, _, _ = _register(env, make_manifest(cloaked=False))
    ph = man["original"]["phash"]
    assert _verify(env, hx("s"), _flip(ph, 15))["verdict"] == "SIMILAR"
    assert _verify(env, hx("t"), _flip(ph, 25))["verdict"] == "UNKNOWN"


def test_without_calibration_near_matching_is_disabled(env, tmp_path):
    env["routes"].reset_for_tests(str(tmp_path / "r2.json"), str(tmp_path / "c2.json"),
                                  str(tmp_path / "missing.json"))
    _, man, _, _ = _register(env)
    d = _verify(env, hx("q"), _flip(man["original"]["phash"], 2))
    assert d["verdict"] == "UNKNOWN" and d["nearMatch"] == "uncalibrated"
    assert _verify(env, man["original"]["sha256"])["verdict"] == "EXACT_ORIGINAL"


def test_verify_validates_input_and_is_rate_limited(env, monkeypatch):
    assert env["c"].post("/api/photo/verify", json={"sha256": "nope"}).status_code == 400
    monkeypatch.setattr(env["routes"], "VERIFY_LIMIT", 2)   # the bad request above counted
    assert env["c"].post("/api/photo/verify", json={"sha256": hx("1")}).status_code == 200
    assert env["c"].post("/api/photo/verify", json={"sha256": hx("1")}).status_code == 429


def test_hash_only_note_has_no_phash_on_chain(env):
    _, man, _, _ = _register(env, privacy="hash-only")
    (note,) = env["sent"]
    assert "po" not in note and "pc" not in note and note["o"] == man["original"]["sha256"]


def _ledger_restart(env, txs):
    """Simulate a Render restart: fresh registry file, rebuild from a fake indexer."""
    import photo_routes
    from photoshield.chain.ledger_sync import LedgerSync
    photo_routes.reset_for_tests(str(env["tmp"] / "after.json"), str(env["tmp"] / "after_cache.json"))
    photo_routes._db()
    ls = LedgerSync(["ADDR"], http_get=FakeIndexer(txs, page=2))
    photo_routes._merge_ledger(ls.sync())


def test_after_restart_full_photos_still_near_match_and_hash_only_are_exact_only(env):
    full_note, full_man, _ = signed_single(seed_byte=1, seed="full")
    ho_note, ho_man, _ = signed_single(seed_byte=2, seed="ho", hash_only=True)
    _ledger_restart(env, [indexer_tx(full_note, "ADDR", "tx1", 10), indexer_tx(ho_note, "ADDR", "tx2", 11)])
    assert _verify(env, hx("z"), _flip(full_man["original"]["phash"], 3))["verdict"] == "DERIVED"
    assert _verify(env, ho_man["original"]["sha256"])["verdict"] == "EXACT_ORIGINAL"
    assert _verify(env, hx("w"), _flip(ho_man["original"]["phash"], 1))["verdict"] == "UNKNOWN"


def test_rehydrate_restores_phash_from_the_manifest_and_rejects_a_doctored_one(env):
    ho_note, ho_man, _ = signed_single(seed_byte=2, seed="ho", hash_only=True)
    _ledger_restart(env, [indexer_tx(ho_note, "ADDR", "tx2", 11)])
    doctored = json.loads(json.dumps(ho_man))
    doctored["original"]["phash"] = "0" * 16
    r = env["c"].post("/api/photo/rehydrate", json={"manifest": doctored}, headers=_h(env))
    assert r.status_code == 404
    r = env["c"].post("/api/photo/rehydrate", json={"manifest": ho_man}, headers=_h(env))
    assert r.status_code == 200
    assert _verify(env, hx("w"), _flip(ho_man["original"]["phash"], 1))["verdict"] == "DERIVED"


def test_album_registration_uses_the_canonical_merkle_and_plc(env):
    from photoshield.provenance import merkle
    mans = [make_manifest(seed=f"al{i}") for i in range(3)]
    items = [(M.photo_id(M.manifest_digest(m)), M.manifest_digest(m)) for m in mans]
    root, _, n = merkle.album_root(items)
    sk, pk = S.new_key(b"\x33" * 32)
    body = {"albumTitle": "Trip", "manifests": mans, "powNonce": cli_client.mine_pow(root),
            "pk": pk, "sig": S.sign(sk, S.album_message(root, n))}
    r = env["c"].post("/api/photo/register-batch", json=body, headers=_h(env))
    d = r.get_json()
    assert r.status_code == 200, d
    (note,) = env["sent"]
    assert note["pmr"] == root and note["plc"] == 3 and "pc" not in note
    for p in d["photos"]:
        assert merkle.verify_proof(p["leaf"], p["proof"], root)
    proof = env["c"].get(f"/api/photo/proof/{d['photos'][1]['photoId']}").get_json()
    assert proof["mode"] == "album" and proof["root"] == root and proof["leafCount"] == 3
    assert _verify(env, mans[2]["original"]["sha256"])["verdict"] == "EXACT_ORIGINAL"


def test_ownership_proof_links_the_key_after_a_restart(env):
    note, man, pk = signed_single(seed_byte=7, seed="own")
    _ledger_restart(env, [indexer_tx(note, "ADDR", "tx7", 70)])
    sk, _ = S.new_key(bytes([7]) * 32)
    ch = env["c"].get("/api/photo/challenge").get_json()["challenge"]
    r = env["c"].post("/api/photo/prove", headers=_h(env, "bob"),
                      json={"photoId": note["id"], "challenge": ch,
                            "sig": S.sign(sk, S.prove_message(note["id"], ch))}).get_json()
    assert r["proven"] and r["linkedTo"] == "bob"
    assert _verify(env, man["original"]["sha256"])["ownerUser"] == "bob"
    wrong_sk, _ = S.new_key(b"\x09" * 32)
    ch = env["c"].get("/api/photo/challenge").get_json()["challenge"]
    r = env["c"].post("/api/photo/prove", json={"photoId": note["id"], "challenge": ch,
                                                "sig": S.sign(wrong_sk, S.prove_message(note["id"], ch))})
    assert r.get_json()["proven"] is False
    r = env["c"].post("/api/photo/prove", json={"photoId": note["id"], "challenge": ch, "sig": "0" * 128})
    assert r.status_code == 400                                  # a challenge is single-use


def test_reads_stats_owner_listing_and_certificate(env):
    r, man, pk, _ = _register(env)
    pid = r.get_json()["photoId"]
    g = env["c"].get(f"/api/photo/{pid}").get_json()["photo"]
    assert g["ownerPkFp"] == pk[:16] and "ownerUser" not in g and "phash" not in g["original"]
    mine = env["c"].get("/api/photo/owner/alice", headers=_h(env)).get_json()
    assert mine["count"] == 1 and mine["photos"][0]["ownerUser"] == "alice"
    assert env["c"].get("/api/photo/owner/alice").status_code == 403
    assert env["c"].get(f"/api/photo/owner/{pk}").get_json()["count"] == 1
    st = env["c"].get("/api/photo/stats").get_json()
    assert st["photos"] == 1 and st["nearMatch"] == "calibrated" and st["ledgerSync"] == "disabled"
    pdf = env["c"].get(f"/api/photo/certificate/{pid}")
    assert pdf.status_code == 200 and pdf.data[:4] == b"%PDF"
    assert env["c"].get("/api/photo/0000000000000000").status_code == 404


def test_server_cloak_is_off_by_default(env):
    r = env["c"].post("/api/photo/cloak")
    assert r.status_code == 501 and "locally" in r.get_json()["error"]


def test_existing_model_registration_response_is_unchanged(env):
    """Guard: adding the Blueprint must not change /api/register."""
    import app as app_module
    assert {"photo", "fl", "auth"} <= set(app_module.app.blueprints)
    rules = {r.rule for r in app_module.app.url_map.iter_rules()}
    assert "/api/register" in rules and "/api/verify" in rules


def test_ledger_tab_labels_photo_notes_and_keeps_model_rows(monkeypatch):
    import base64
    import algorand_client
    note, _, _ = signed_single()
    txs = [{"id": "p", "confirmed-round": 1, "round-time": 1, "tx-type": "pay", "fee": 1000,
            "note": base64.b64encode(N.encode(note)).decode()},
           {"id": "m", "confirmed-round": 2, "round-time": 2, "tx-type": "pay", "fee": 1000,
            "note": base64.b64encode(json.dumps({"id": "mid", "name": "Model", "hash": "h" * 64,
                                                 "owner": "alice"}).encode()).decode()}]

    class R:
        status_code = 200

        def json(self):
            return {"transactions": txs}
    monkeypatch.setattr(algorand_client.requests, "get", lambda *a, **k: R())
    led = algorand_client.get_transaction_ledger(10)["ledger"]
    assert led[0]["action"] == "photo-anchor" and led[0]["photo_id"] == note["id"]
    assert led[1]["action"] == "registration" and led[1]["model_name"] == "Model"
    assert led[1]["model_hash"] == "h" * 16 + "..." and led[1]["owner"] == "alice"
