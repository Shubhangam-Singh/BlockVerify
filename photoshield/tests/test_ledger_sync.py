"""Ledger rebuild (design §6.8): paging, filtering, signatures, cache, outages."""
import pytest

from photoshield.chain import notes as N
from photoshield.chain.ledger_sync import LedgerSync, backoff_delay

from .helpers import FakeIndexer, indexer_tx, signed_single

ADDR = "A" * 58


def _txs(n=7):
    out = []
    for i in range(n):
        note, _, _ = signed_single(seed_byte=i + 1, seed=f"s{i}")
        out.append(indexer_tx(note, ADDR, f"tx{i}", 100 + i))
    return out


def test_follows_next_token_across_every_page(tmp_path):
    fake = FakeIndexer(_txs(7), page=2)
    ls = LedgerSync([ADDR], http_get=fake, cache_path=str(tmp_path / "c.json"))
    entries = ls.sync()
    assert len(entries) == 7 and len(fake.calls) == 4          # 2+2+2+1
    assert all(c["note-prefix"] == "eyJwcyI6MSw=" for c in fake.calls)   # base64('{"ps":1,')
    assert ls.status["state"] == "ok" and ls.status["pages"] == 4


def test_rejects_foreign_senders_bad_signatures_and_junk(tmp_path):
    good, _, _ = signed_single(seed_byte=9, seed="g")
    forged = dict(good)
    forged["sig"] = ("0" if good["sig"][0] != "0" else "1") + good["sig"][1:]
    wrong_id = dict(good)
    wrong_id["id"] = "0" * 16
    txs = [indexer_tx(good, ADDR, "ok", 1),
           indexer_tx(good, ADDR, "foreign", 2, sender="B" * 58),
           indexer_tx(good, ADDR, "elsewhere", 3, receiver="C" * 58),
           indexer_tx(good, ADDR, "paid", 4, amount=5),
           indexer_tx(forged, ADDR, "forged", 5),
           indexer_tx(wrong_id, ADDR, "wrongid", 6),
           indexer_tx(b'{"ps":1,"junk":true}', ADDR, "junk", 7)]
    ls = LedgerSync([ADDR], http_get=FakeIndexer(txs, page=10))
    assert [e["txid"] for e in ls.sync()] == ["ok"]
    assert ls.status["rejected"] == 6


def test_incremental_scan_starts_after_the_cached_round(tmp_path):
    txs = _txs(3)
    cache = str(tmp_path / "c.json")
    LedgerSync([ADDR], http_get=FakeIndexer(txs), cache_path=cache).sync()
    more = txs + [indexer_tx(signed_single(seed_byte=50, seed="new")[0], ADDR, "tx-new", 200)]
    fake = FakeIndexer(more)
    ls = LedgerSync([ADDR], http_get=fake, cache_path=cache)
    entries = ls.sync()
    assert fake.calls[0]["min-round"] == 103 and len(entries) == 4


def test_hash_only_anchors_rebuild_without_phash():
    note, _, _ = signed_single(seed_byte=3, hash_only=True, seed="h")
    ls = LedgerSync([ADDR], http_get=FakeIndexer([indexer_tx(note, ADDR, "h", 5)]))
    (e,) = ls.sync()
    assert N.hash_only(e["note"]) and "po" not in e["note"]


def test_unreachable_indexer_keeps_the_cache_and_reports(tmp_path):
    cache = str(tmp_path / "c.json")
    LedgerSync([ADDR], http_get=FakeIndexer(_txs(2)), cache_path=cache).sync()
    ls = LedgerSync([ADDR], http_get=FakeIndexer([], fail=True), cache_path=cache)
    with pytest.raises(Exception):
        ls.sync()
    assert ls.status["state"] == "unreachable" and len(ls.entries()) == 2


def test_several_addresses_are_scanned():
    other = "Z" * 58
    note, _, _ = signed_single(seed_byte=4, seed="o")
    ls = LedgerSync([ADDR, other, ADDR],
                    http_get=FakeIndexer(_txs(1) + [indexer_tx(note, other, "o", 9, sender=other, receiver=other)]))
    assert ls.addresses == [ADDR, other]


def test_backoff_doubles_and_caps():
    assert [backoff_delay(i) for i in (0, 1, 2, 3)] == [1, 2, 4, 8]
    assert backoff_delay(20) == 300
