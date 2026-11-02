"""PhotoAnchor: commit a note to a ledger (design §6.7). Mirrors fedverify's ChainAnchor.

    mock      no I/O; the txid is the SHA-256 of the note, so runs are reproducible
    local     appends a "photo_anchor" transaction to backend/blockchain.py and mines a block
    algorand  algorand_client.broadcast_photo_anchor: a 0-ALGO self-payment carrying the note,
              never a contract call (the TEAL app's 64 slots belong to model registrations)
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Optional

from . import notes as N

BACKENDS = ("mock", "local", "algorand")
_BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "backend")
_TS_BASE = 1_700_000_000.0


def _import_backend(name):
    if _BACKEND_DIR not in sys.path:
        sys.path.insert(0, _BACKEND_DIR)
    return __import__(name)


class PhotoAnchor:
    def __init__(self, backend: str = "mock", difficulty: int = 2):
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
        self.backend = backend
        self.difficulty = difficulty
        self._chain = None
        self.log = []

    @property
    def chain(self):
        if self._chain is None:
            self._chain = _import_backend("blockchain").Blockchain(
                difficulty=self.difficulty, genesis_timestamp=_TS_BASE)
        return self._chain

    def anchor(self, note: dict) -> dict:
        """Returns {success, backend, txid, round, bytes_written, latency_ms[, error]}."""
        raw = N.encode(note)
        t0 = time.perf_counter()
        rec = {"backend": self.backend, "bytes_written": len(raw)}
        if self.backend == "mock":
            rec.update(success=True, txid=hashlib.sha256(raw).hexdigest(), round=None)
        elif self.backend == "local":
            bc = self.chain
            bc.add_transaction({"type": "photo_anchor", "note": raw.decode("ascii")})
            block = bc.mine_pending_transactions(block_timestamp=_TS_BASE + len(bc.chain))
            rec.update(success=True, txid=block.hash, round=block.index)
        else:
            res = _import_backend("algorand_client").broadcast_photo_anchor(note)
            rec.update(success=bool(res.get("success")), txid=res.get("txid"),
                       round=res.get("round"), fee=res.get("fee"))
            if not res.get("success"):
                rec["error"] = res.get("error")
        rec["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        self.log.append(rec)
        return rec
