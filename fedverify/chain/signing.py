"""Ed25519-signed submissions for federated rounds (Revision 2 §3.2). Keys here are TEST KEYS.

Keys are derived deterministically from (run seed, hospital), so a run is reproducible. That also means anyone who
knows the seed can rebuild them: they are TEST KEYS and must never protect anything real.

A submission is

    {run_id, round, hospital_id, cid, digest, timestamp, txn_id, signature}

and the signature covers EVERY other field (canonical JSON: sorted keys, compact separators, UTF-8), so changing the
timestamp, the txn_id or anything else invalidates it.

    txn_id = SHA-256(hospital_id || per-hospital counter || timestamp)

is the reference paper's unique-identifier idea. Unlike its toy dictionary, the per-hospital counter and the
registered public keys live here, in the verifier: KeyRegistry rejects missing or invalid signatures, unregistered
keys and a txn_id that does not match the hospital's next counter, and says why.

The optional arguments of sign_submission() and KeyRegistry.verify() exist ONLY so the Phase-4 tests can switch one
defence off and show that the attack it stops then succeeds (a defence test that cannot fail proves nothing). Their
defaults are the defences above, and every run uses the defaults.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

SIGNED_FIELDS: tuple[str, ...] = ("run_id", "round", "hospital_id", "cid", "digest", "timestamp", "txn_id")
# What a weaker scheme would sign: the model bytes' identity but not the transaction metadata. Defence-off tests only.
PAYLOAD_FIELDS: tuple[str, ...] = ("run_id", "hospital_id", "cid", "digest")
COUNTER_WINDOW = 64       # how far ahead of the next expected counter a txn_id may be (lost submissions)
TEST_KEY_LABEL = "TEST KEY (derived from the run seed; never use for anything real)"


def derive_test_key(run_seed: int, hospital_id: str) -> Ed25519PrivateKey:
    """Deterministic Ed25519 TEST key for (run seed, hospital)."""
    seed32 = hashlib.sha256(f"fedverify|TEST-KEY|{int(run_seed)}|{hospital_id}".encode("utf-8")).digest()
    return Ed25519PrivateKey.from_private_bytes(seed32)


def public_key_bytes(key: Ed25519PrivateKey | Ed25519PublicKey) -> bytes:
    pub = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    return pub.public_bytes(Encoding.Raw, PublicFormat.Raw)


def make_txn_id(hospital_id: str, counter: int, timestamp: float | int) -> str:
    return hashlib.sha256(f"{hospital_id}|{int(counter)}|{timestamp}".encode("utf-8")).hexdigest()


def signing_bytes(submission: dict, signed: tuple[str, ...] = SIGNED_FIELDS) -> bytes:
    """The exact bytes that are signed: every field except the signature, canonical JSON."""
    missing = [f for f in signed if f not in submission]
    if missing:
        raise KeyError(f"submission lacks {missing}")
    return json.dumps({f: submission[f] for f in signed}, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sign_submission(key: Ed25519PrivateKey, fields: dict, signed: tuple[str, ...] = SIGNED_FIELDS) -> dict:
    """Return fields + signature (hex). `fields` must hold every SIGNED_FIELDS entry; `signed` is what the signature
    covers (every field by default; a narrower tuple only in the defence-off tests)."""
    sub = {f: fields[f] for f in SIGNED_FIELDS}
    sub["signature"] = key.sign(signing_bytes(sub, signed)).hex()
    return sub


class KeyRegistry:
    """The verifier's state: one registered public key per hospital and the next expected counter per hospital."""

    def __init__(self) -> None:
        self._keys: dict[str, bytes] = {}
        self._counter: dict[str, int] = {}

    def register(self, hospital_id: str, public_key: bytes) -> None:
        if hospital_id in self._keys and self._keys[hospital_id] != bytes(public_key):
            raise ValueError(f"{hospital_id} already has a different registered key")
        self._keys[hospital_id] = bytes(public_key)
        self._counter.setdefault(hospital_id, 0)

    def is_registered(self, hospital_id: str) -> bool:
        return hospital_id in self._keys

    def public_key(self, hospital_id: str) -> Optional[bytes]:
        return self._keys.get(hospital_id)

    def next_counter(self, hospital_id: str) -> int:
        return self._counter.get(hospital_id, 0)

    def verify(self, submission: dict, commit: bool = True, signed: tuple[str, ...] = SIGNED_FIELDS,
               check_counter: bool = True, registered_keys: bool = True) -> tuple[bool, Optional[str]]:
        """(True, None) if the submission is acceptable, else (False, reason). With commit=True an accepted
        submission advances the hospital's counter, so the same txn_id can never be accepted twice.

        Defence-off tests only: `signed` narrower than SIGNED_FIELDS checks a signature over fewer fields,
        check_counter=False skips the txn_id counter, and registered_keys=False trusts a "public_key" the submission
        carries instead of the key registered for the claimed hospital."""
        if not isinstance(submission, dict):
            return False, "not a submission"
        missing = [f for f in SIGNED_FIELDS if f not in submission]
        if missing:
            return False, f"missing field(s) {missing}"
        if not submission.get("signature"):
            return False, "missing signature"
        h = submission["hospital_id"]
        if registered_keys:
            if h not in self._keys:
                return False, f"unregistered key: no public key is registered for {h!r}"
            pub, whose = self._keys[h], f"{h}'s registered key"
        else:
            try:
                pub, whose = bytes.fromhex(str(submission.get("public_key", ""))), "the key the submission carries"
            except ValueError:
                return False, "no usable public key in the submission"
        try:
            Ed25519PublicKey.from_public_bytes(pub).verify(bytes.fromhex(str(submission["signature"])),
                                                           signing_bytes(submission, signed))
        except (InvalidSignature, ValueError):
            return False, f"invalid signature: does not verify under {whose}"
        if not check_counter:
            return True, None
        # The counter only moves forward: the txn_id must hash a counter at or above the next expected one (within a
        # window, so one lost or rejected submission does not lock the hospital out). A used counter never verifies.
        nxt = self._counter.get(h, 0)
        used = next((c for c in range(nxt, nxt + COUNTER_WINDOW)
                     if make_txn_id(h, c, submission["timestamp"]) == submission["txn_id"]), None)
        if used is None:
            return False, f"txn_id does not match any unused counter of {h} (next is {nxt})"
        if commit:
            self._counter[h] = used + 1
        return True, None


def describe_registry(reg: KeyRegistry) -> dict[str, Any]:
    """Public keys (hex) for config.json, labelled as test keys."""
    return {"label": TEST_KEY_LABEL, "public_keys": {h: reg.public_key(h).hex() for h in sorted(reg._keys)}}
