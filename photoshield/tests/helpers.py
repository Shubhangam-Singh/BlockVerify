"""Small builders shared by the tests."""
import hashlib

from photoshield.provenance import manifest as M


def hx(seed: str, n: int = 64) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:n]


def make_manifest(title="Test photo", seed="a", cloaked=True, **cloak_over):
    original = {"sha256": hx("orig" + seed), "phash": hx("po" + seed, 16), "width": 640, "height": 480}
    cl = {"sha256": hx("clk" + seed), "phash": hx("pc" + seed, 16), "format": "png"} if cloaked else None
    cloak = None
    if cloaked:
        cloak = {"model": "tiny", "eps": 8, "steps": 10, "size": 64, "seed": 0, "eot": ["resize"],
                 "psnr": "40.1234", "tool_version": "photoshield-0.1.0"}
        cloak.update(cloak_over)
    return M.build(title, original, cl, cloak)


def signed_single(seed_byte=1, hash_only=False, title="t", seed="a"):
    """(note, manifest, pk) for a valid single-photo anchor signed by a TEST key."""
    from photoshield.chain import notes as N
    from photoshield.provenance import signing as S
    man = make_manifest(title=title, seed=seed)
    md = M.manifest_digest(man)
    sk, pk = S.new_key(bytes([seed_byte]) * 32)
    sig = S.sign(sk, S.register_message(md))
    note = N.single_note(M.photo_id(md), man["original"]["sha256"], man["cloaked"]["sha256"],
                         man["original"]["phash"], man["cloaked"]["phash"], md, pk, sig,
                         hash_only=hash_only)
    return note, man, pk


def indexer_tx(note, addr, txid, rnd, sender=None, receiver=None, amount=0):
    import base64
    from photoshield.chain import notes as N
    raw = N.encode(note) if isinstance(note, dict) else note
    return {"id": txid, "confirmed-round": rnd, "round-time": 1_700_000_000 + rnd,
            "sender": sender or addr, "tx-type": "pay",
            "payment-transaction": {"receiver": receiver or addr, "amount": amount},
            "note": base64.b64encode(raw).decode()}


class FakeIndexer:
    """Serves transactions in pages of `page` with next-tokens; records every request."""

    def __init__(self, txs, page=2, fail=False):
        self.txs, self.page, self.fail, self.calls = list(txs), page, fail, []

    def __call__(self, url, params, timeout):
        self.calls.append(dict(params))

        class R:
            pass
        r = R()
        if self.fail:
            r.status_code = 503
            r.json = lambda: {}
            return r
        mr = params.get("min-round", 0)
        pool = [t for t in self.txs if t["confirmed-round"] >= mr]
        start = int(params.get("next", "0"))
        chunk = pool[start:start + self.page]
        body = {"transactions": chunk}
        if start + self.page < len(pool):
            body["next-token"] = str(start + self.page)
        r.status_code = 200
        r.json = lambda: body
        return r
