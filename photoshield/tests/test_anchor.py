"""Notes and anchoring backends (design §5.3, §6.7)."""
import json

import pytest

from photoshield.chain import notes as N
from photoshield.chain.anchor import PhotoAnchor

from .helpers import signed_single


def test_note_starts_with_the_prefix_and_fits():
    note, _, _ = signed_single()
    raw = N.encode(note)
    assert raw.startswith(b'{"ps":1,') and len(raw) <= 1024
    assert N.parse(raw) == note


def test_hash_only_note_omits_phash_but_keeps_the_digest():
    note, man, _ = signed_single(hash_only=True)
    assert "po" not in note and "pc" not in note and N.hash_only(note)
    assert N.parse(N.encode(note))["md"] == note["md"]


def test_out_of_order_or_oversize_notes_are_refused():
    note, _, _ = signed_single()
    shuffled = dict(reversed(list(note.items())))
    with pytest.raises(N.NoteError):
        N.encode(shuffled)
    big = dict(note)
    big["sig"] = "f" * 2000
    with pytest.raises(N.NoteError):
        N.encode(big)


def test_parse_rejects_model_and_fl_notes_and_junk():
    assert N.parse(json.dumps({"id": "x", "name": "m", "hash": "h", "owner": "o"}).encode()) is None
    assert N.parse(b'{"frr":"ab","flc":3,"rnd":1,"run":"r"}') is None
    assert N.parse(b'{"ps":1,"id":"zz"}') is None
    assert N.parse(b"not json") is None


def test_album_note_uses_plc_not_pc():
    n = N.album_note("a" * 64, 3, "b" * 64, "c" * 64, "d" * 128)
    assert "plc" in n and "pc" not in n
    assert N.parse(N.encode(n))["plc"] == 3


def test_mock_txids_are_deterministic():
    note, _, _ = signed_single()
    a, b = PhotoAnchor("mock").anchor(note), PhotoAnchor("mock").anchor(note)
    assert a["txid"] == b["txid"] and a["success"]


def test_local_backend_mines_a_block_holding_the_note():
    note, _, _ = signed_single()
    anc = PhotoAnchor("local")
    rec = anc.anchor(note)
    block = anc.chain.chain[-1]
    assert rec["txid"] == block.hash and block.hash.startswith("00")
    assert block.transactions[0]["type"] == "photo_anchor"
    assert json.loads(block.transactions[0]["note"]) == note
    assert anc.chain.is_chain_valid()["valid"]


def test_algorand_backend_calls_broadcast_photo_anchor_only(monkeypatch):
    import algorand_client
    sent = []
    monkeypatch.setattr(algorand_client, "broadcast_photo_anchor",
                        lambda n: sent.append(n) or {"success": True, "txid": "T" * 52, "round": 7, "fee": 1000})
    monkeypatch.setattr(algorand_client, "broadcast_hash_to_algorand",
                        lambda *a, **k: pytest.fail("photos must not call the contract path"))
    note, _, _ = signed_single()
    rec = PhotoAnchor("algorand").anchor(note)
    assert rec["success"] and rec["round"] == 7 and sent == [note]


def test_broadcast_photo_anchor_refuses_bad_notes_before_any_network(monkeypatch):
    import algorand_client
    monkeypatch.setattr(algorand_client.client, "account_info",
                        lambda *a: pytest.fail("must not reach the network"))
    assert not algorand_client.broadcast_photo_anchor({"id": "x"})["success"]
    note, _, _ = signed_single()
    big = dict(note)
    big["sig"] = "f" * 2000
    res = algorand_client.broadcast_photo_anchor(big)
    assert not res["success"] and "1024" in res["error"]
