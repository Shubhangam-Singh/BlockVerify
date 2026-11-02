"""Rebuild the photo registry from the public ledger (design §6.8).

The ledger is the source of truth. The JSON registry is a cache of it plus things the chain
does not hold (username, title). On hosts whose disk is wiped on restart (Render's free tier)
the registry is rebuilt from here.

Scan: GET {indexer}/v2/accounts/{addr}/transactions with note-prefix = {"ps":1, tx-type = pay,
limit = 1000 and min-round = the cached last round + 1, following `next-token` until it runs
out. Every page is read; the scan never stops at the first one.

A transaction is accepted ONLY if:
- sender == receiver == the scanned address, amount 0 (otherwise anyone could send a note TO
  our address and inject records);
- its note parses as a PhotoShield note (notes.parse);
- the owner signature verifies (register@1 over md, or album@1 over pmr|plc);
- for a single photo, id == photo_id(md).

Standard library + requests only: the backend imports this module.
"""
from __future__ import annotations

import base64
import json
import os
import time
from typing import Callable, Dict, List, Optional

from ..provenance import signing
from ..provenance.manifest import photo_id
from . import notes as N

DEFAULT_INDEXER = "https://testnet-idx.algonode.cloud"
PAGE_LIMIT = 1000
MAX_PAGES = 10_000                 # a hard stop against a misbehaving indexer


class IndexerError(RuntimeError):
    pass


def backoff_delay(attempt: int) -> float:
    """Retry delay after `attempt` consecutive failures: 1, 2, 4, ... capped at 300 s."""
    return float(min(300, 2 ** max(0, attempt)))


def _default_get(url, params, timeout):
    import requests
    return requests.get(url, params=params, timeout=timeout)


def verify_note(note: dict) -> bool:
    if "id" in note:
        return (note["id"] == photo_id(note["md"])
                and signing.verify(note["pk"], note["sig"], signing.register_message(note["md"])))
    return signing.verify(note["pk"], note["sig"], signing.album_message(note["pmr"], note["plc"]))


def accept(tx: dict, addr: str) -> Optional[dict]:
    """Return {txid, round, round_time, note} for a valid PhotoShield anchor, else None."""
    pay = tx.get("payment-transaction") or {}
    if tx.get("sender") != addr or pay.get("receiver") != addr or pay.get("amount", 0) != 0:
        return None
    try:
        raw = base64.b64decode(tx.get("note") or "", validate=True)
    except (ValueError, TypeError):
        return None
    note = N.parse(raw)
    if note is None or not verify_note(note):
        return None
    return {"txid": tx.get("id"), "round": tx.get("confirmed-round"),
            "round_time": tx.get("round-time"), "address": addr, "note": note}


class LedgerSync:
    def __init__(self, addresses: List[str], indexer: str = DEFAULT_INDEXER,
                 cache_path: Optional[str] = None, http_get: Optional[Callable] = None,
                 timeout: float = 12.0, page_limit: int = PAGE_LIMIT):
        self.addresses = [a for a in dict.fromkeys(a.strip() for a in addresses) if a]
        self.indexer = indexer.rstrip("/")
        self.cache_path = cache_path
        self.http_get = http_get or _default_get
        self.timeout = timeout
        self.page_limit = page_limit
        self.status = {"state": "idle", "pages": 0, "accepted": 0, "rejected": 0,
                       "last_error": None, "last_sync": None, "failures": 0}
        self.cache = self._load_cache()

    # ── cache ────────────────────────────────────────────────────────────────
    def _load_cache(self) -> dict:
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path) as f:
                    c = json.load(f)
                if c.get("version") == 1:
                    return c
            except (OSError, json.JSONDecodeError):
                pass
        return {"version": 1, "addresses": {}}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.cache, f)
        os.replace(tmp, self.cache_path)

    # ── scan ─────────────────────────────────────────────────────────────────
    def _scan(self, addr: str, min_round: int) -> List[dict]:
        url = f"{self.indexer}/v2/accounts/{addr}/transactions"
        params = {"note-prefix": base64.b64encode(N.PREFIX).decode(), "tx-type": "pay",
                  "limit": self.page_limit}
        if min_round > 0:
            params["min-round"] = min_round
        out, token = [], None
        for _ in range(MAX_PAGES):
            p = dict(params)
            if token:
                p["next"] = token
            r = self.http_get(url, p, self.timeout)
            if getattr(r, "status_code", 0) != 200:
                raise IndexerError(f"indexer returned {getattr(r, 'status_code', '?')}")
            body = r.json()
            self.status["pages"] += 1
            for tx in body.get("transactions", []):
                rec = accept(tx, addr)
                if rec is None:
                    self.status["rejected"] += 1
                else:
                    out.append(rec)
            token = body.get("next-token")
            if not token or not body.get("transactions"):
                break
        return out

    def sync(self) -> List[dict]:
        """Incremental scan of every address. On failure the cache is kept and re-raised."""
        self.status.update(state="rebuilding", pages=0, accepted=0, rejected=0)
        try:
            for addr in self.addresses:
                slot = self.cache["addresses"].setdefault(addr, {"last_round": 0, "entries": []})
                new = self._scan(addr, slot["last_round"] + 1 if slot["last_round"] else 0)
                seen = {e["txid"] for e in slot["entries"]}
                for rec in new:
                    if rec["txid"] not in seen:
                        slot["entries"].append(rec)
                        seen.add(rec["txid"])
                rounds = [e["round"] for e in slot["entries"] if isinstance(e.get("round"), int)]
                slot["last_round"] = max(rounds) if rounds else slot["last_round"]
                self.status["accepted"] += len(new)
            self._save_cache()
            self.status.update(state="ok", last_error=None, last_sync=int(time.time()), failures=0)
        except Exception as e:
            self.status.update(state="unreachable", last_error=str(e)[:200],
                               failures=self.status["failures"] + 1)
            raise
        return self.entries()

    def entries(self) -> List[dict]:
        """Every accepted anchor known (cache included), oldest round first."""
        allr = [e for a in self.addresses for e in self.cache["addresses"].get(a, {}).get("entries", [])]
        return sorted(allr, key=lambda e: (e.get("round") or 0, e.get("txid") or ""))
