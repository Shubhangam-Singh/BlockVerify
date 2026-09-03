"""ChainAnchor — commits a federated round root to a ledger.

Three backends, matching how much trust and I/O each experiment needs:

  "mock"      no I/O at all. TxIDs are a hash of the committed content, so they are
              DETERMINISTIC: a mock run is byte-reproducible for a fixed seed, which is
              what CLAUDE.md requires of rounds.jsonl. Default for experiments.
  "local"     appends fl_client_update / fl_round_commit transactions to the repository's
              own proof-of-work chain (backend/blockchain.py) and mines a block; the block
              hash is the txid. Default for tests.
  "algorand"  additionally submits a real note transaction to Algorand Testnet via
              algorand_client.broadcast_fl_round (note keys frr/flc/rnd/run).

Bitcoin OP_RETURN checkpoints every ``checkpoint_every`` rounds commit the Merkle root OF
THE ROUND ROOTS SO FAR — one cheap on-chain write that pins the entire lineage, so a
server cannot silently rewrite an earlier round without breaking the checkpoint. It uses
the existing bitcoin_client path; no new signing code is written here.

Every anchor returns {latency_ms, txid, fee, bytes_written} and is appended to ``.log``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import List, Optional

from .commitment import round_merkle_root

BACKENDS = ("mock", "local", "algorand")

# Deterministic block timestamps keep "local" txids reproducible for a fixed seed; the
# REAL elapsed time is reported separately as latency_ms, so nothing is lost.
_TS_BASE = 1_700_000_000.0

_BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "backend")


def _import_backend(name):
    if _BACKEND_DIR not in sys.path:
        sys.path.insert(0, _BACKEND_DIR)
    return __import__(name)


def _mock_txid(*parts) -> str:
    """Content-addressed fake txid — deterministic, so mock runs stay reproducible."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


class ChainAnchor:
    def __init__(self, backend: str = "mock", checkpoint_every: int = 10,
                 bitcoin: bool = False, difficulty: int = 2,
                 deterministic_ts: bool = True):
        if backend not in BACKENDS:
            raise ValueError(f"unknown chain backend {backend!r}; expected one of {BACKENDS}")
        self.backend = backend
        self.checkpoint_every = int(checkpoint_every)
        self.bitcoin = bool(bitcoin)
        self.deterministic_ts = bool(deterministic_ts)
        self.round_roots: List[str] = []          # ordered lineage, index 0 == round 1
        self.log: List[dict] = []
        self.checkpoints: List[dict] = []
        self._chain = None
        self._difficulty = int(difficulty)

    # ── the repository's own PoW chain ───────────────────────────────────────
    @property
    def chain(self):
        if self._chain is None:
            Blockchain = _import_backend("blockchain").Blockchain
            self._chain = Blockchain(difficulty=self._difficulty,
                                     genesis_timestamp=_TS_BASE)
        return self._chain

    # ── one round ────────────────────────────────────────────────────────────
    def anchor_round(self, run_id: str, round_num: int, root: str, leaf_count: int,
                     leaves: Optional[list] = None) -> dict:
        """Commit one round's Merkle root. Returns {latency_ms, txid, fee, bytes_written}."""
        t0 = time.perf_counter()
        payload = json.dumps({"run": str(run_id), "rnd": int(round_num),
                              "frr": root, "flc": int(leaf_count)},
                             separators=(",", ":"), sort_keys=True)
        txid, fee, nbytes, extra = None, 0, len(payload.encode()), {}

        if self.backend == "mock":
            txid = _mock_txid(run_id, round_num, root, leaf_count)

        elif self.backend in ("local", "algorand"):
            txid, fee, nbytes = self._anchor_local(run_id, round_num, root,
                                                   leaf_count, leaves, payload)
            if self.backend == "algorand":
                algo = _import_backend("algorand_client").broadcast_fl_round(
                    run_id, round_num, root, leaf_count)
                extra["algorand"] = algo
                if algo.get("success"):
                    txid = algo["txid"]
                    fee = algo.get("fee", fee)
                    nbytes = algo.get("bytes_written", nbytes)
                else:
                    extra["algorand_error"] = algo.get("error")

        rec = {"round": int(round_num), "root": root, "leaf_count": int(leaf_count),
               "backend": self.backend, "txid": txid, "fee": fee,
               "bytes_written": int(nbytes),
               "latency_ms": (time.perf_counter() - t0) * 1000.0, **extra}

        # keep the lineage dense and correctly ordered even if a round is re-anchored
        while len(self.round_roots) < round_num:
            self.round_roots.append(None)
        self.round_roots[round_num - 1] = root

        self.log.append(rec)
        return rec

    def _anchor_local(self, run_id, round_num, root, leaf_count, leaves, payload):
        """Append the two additive tx types to the PoW chain and mine one block."""
        bc = self.chain
        for lf in (leaves or []):
            bc.add_transaction({
                "type": "fl_client_update",              # additive tx type
                "run_id": str(run_id), "round": int(round_num),
                "client_id": str(lf["client_id"]), "digest": lf["digest"],
                "leaf": lf["leaf"], "timestamp": _TS_BASE + round_num,
            })
        bc.add_transaction({
            "type": "fl_round_commit",                   # additive tx type
            "run_id": str(run_id), "round": int(round_num),
            "root": root, "leaf_count": int(leaf_count),
            "timestamp": _TS_BASE + round_num,
        })
        ts = (_TS_BASE + round_num) if self.deterministic_ts else None
        block = bc.mine_pending_transactions(block_timestamp=ts)
        return (block.hash if block else None), 0, len(payload.encode())

    # ── periodic Bitcoin checkpoint over the lineage ─────────────────────────
    def due_for_checkpoint(self, round_num: int) -> bool:
        return self.checkpoint_every > 0 and round_num % self.checkpoint_every == 0

    def checkpoint(self, run_id: str, round_num: int) -> dict:
        """Anchor the Merkle root of all round roots so far via the existing BTC path."""
        t0 = time.perf_counter()
        roots = [(str(i + 1), r) for i, r in enumerate(self.round_roots) if r]
        cp_root, cp_count = round_merkle_root(roots)

        rec = {"round": int(round_num), "checkpoint_root": cp_root,
               "rounds_covered": cp_count, "backend": "bitcoin" if self.bitcoin else "mock"}

        if self.bitcoin:
            btc = _import_backend("bitcoin_client")
            if not getattr(btc, "BIT_AVAILABLE", False):
                rec.update(txid=None, fee=0, bytes_written=0,
                           error="bit library unavailable")
            else:
                res = btc.anchor_merkle_root(cp_root)
                rec.update(txid=res.get("txid"), fee=res.get("fee_satoshi", 0),
                           bytes_written=len(res.get("op_return_hex", "")) // 2,
                           success=res.get("success"), error=res.get("error"))
        else:
            rec.update(txid=_mock_txid("btc", run_id, round_num, cp_root),
                       fee=0, bytes_written=32)

        rec["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        self.checkpoints.append(rec)
        return rec

    def lineage(self) -> List[str]:
        return list(self.round_roots)
