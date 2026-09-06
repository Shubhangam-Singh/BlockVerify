"""Gemini API key pool with health tracking and automatic rotation.

Why a pool: a single free-tier key hits its per-minute and per-day quota quickly, and the
whole assistant goes dark. Several keys rotate round-robin, and a key that reports quota
exhaustion is benched until its window plausibly resets rather than retried into the same
error on every request.

Keys are read from the environment ONLY. Nothing here ever writes a key to disk, a log
line, or an HTTP response — `mask()` is the only way a key is allowed to be displayed.

    GEMINI_API_KEYS=key1,key2,key3      preferred (a pool)
    GEMINI_API_KEY=key1                 still honoured (single key)
"""
from __future__ import annotations

import os
import threading
import time
from typing import List, Optional

# A key that reports quota exhaustion is benched for this long. Gemini free-tier
# per-minute windows reset in 60s; per-day quotas obviously do not, but re-probing
# every few minutes is cheap and self-heals without a restart.
COOLDOWN_S = 90.0
# Consecutive hard failures (401/403 — revoked or invalid) before a key is retired.
MAX_HARD_FAILS = 3


class _Key:
    __slots__ = ("value", "cooldown_until", "hard_fails", "calls", "ok", "last_error")

    def __init__(self, value: str):
        self.value = value
        self.cooldown_until = 0.0
        self.hard_fails = 0
        self.calls = 0
        self.ok = 0
        self.last_error = None

    @property
    def retired(self) -> bool:
        return self.hard_fails >= MAX_HARD_FAILS

    def available(self, now: float) -> bool:
        return not self.retired and now >= self.cooldown_until


def mask(key: str) -> str:
    """The ONLY safe way to render a key. Never log or return the raw value."""
    if not key:
        return "(none)"
    if len(key) > 12:
        return f"{key[:6]}…{key[-4:]}"
    return f"(short: {len(key)} chars)"      # length only — still never the value


def _parse(raw: str) -> List[str]:
    seen, out = set(), []
    for part in raw.replace("\n", ",").split(","):
        k = part.strip().strip('"').strip("'")
        # Ignore placeholders left in a copied .env.example. Without this they are sent
        # to Google verbatim and come back as "API key not valid", which reads like a
        # revoked key rather than "you have not filled this in".
        low = k.lower()
        if (not k
                or low.startswith(("your_", "changeme", "<", "key_one", "key_two", "key_three"))
                or low.endswith(("_here", "_here'"))
                or low in ("none", "null", "todo")
                or len(k) < 20):            # no real provider key is this short
            continue
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


class KeyPool:
    def __init__(self, keys: Optional[List[str]] = None):
        if keys is None:
            keys = _parse(os.environ.get("GEMINI_API_KEYS", "")) or \
                   _parse(os.environ.get("GEMINI_API_KEY", ""))
        self._keys = [_Key(k) for k in keys]
        self._i = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def configured(self) -> bool:
        return bool(self._keys)

    def acquire(self) -> Optional[_Key]:
        """Next usable key, round-robin. None when every key is cooling down or retired."""
        now = time.time()
        with self._lock:
            n = len(self._keys)
            for offset in range(n):
                k = self._keys[(self._i + offset) % n]
                if k.available(now):
                    self._i = (self._i + offset + 1) % n
                    k.calls += 1
                    return k
        return None

    def report(self, key: "_Key", status: int, message: str = "") -> None:
        """Feed the outcome back so the next call routes around a dead key."""
        with self._lock:
            if status == 200:
                key.ok += 1
                key.hard_fails = 0
                key.last_error = None
                return
            key.last_error = (message or f"HTTP {status}")[:160]
            if status == 429 or "quota" in message.lower() or "exhaust" in message.lower():
                key.cooldown_until = time.time() + COOLDOWN_S
            elif status in (401, 403):
                key.hard_fails += 1            # revoked / invalid / not enabled
            elif status >= 500:
                key.cooldown_until = time.time() + 10.0   # upstream blip, brief bench

    def revive(self) -> None:
        """Clear soft cooldowns so a cheaper model gets a fresh chance at every key.

        Retired keys (repeated 401/403) stay retired — those are broken, not throttled.
        """
        with self._lock:
            for k in self._keys:
                if not k.retired:
                    k.cooldown_until = 0.0

    def cooldown_remaining(self) -> float:
        """Seconds until the soonest key frees up (0 if one is available now)."""
        now = time.time()
        live = [k for k in self._keys if not k.retired]
        if not live:
            return 0.0
        if any(k.available(now) for k in live):
            return 0.0
        return max(0.0, min(k.cooldown_until for k in live) - now)

    def status(self) -> dict:
        """Operator view. Masked values only — never the raw keys."""
        now = time.time()
        return {
            "configured": len(self._keys),
            "available": sum(1 for k in self._keys if k.available(now)),
            "retired": sum(1 for k in self._keys if k.retired),
            "cooldownSeconds": round(self.cooldown_remaining(), 1),
            "keys": [{
                "id": mask(k.value),
                "state": "retired" if k.retired else ("cooling" if now < k.cooldown_until else "ready"),
                "calls": k.calls, "ok": k.ok,
                "cooldownSeconds": max(0, round(k.cooldown_until - now, 1)),
                "lastError": k.last_error,
            } for k in self._keys],
        }
