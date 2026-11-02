"""The Algorand note format (design §5.3). Standard library only: the backend imports it.

Keys are emitted in a FIXED order with compact separators, so every PhotoShield note starts
with the bytes {"ps":1, which the ledger rebuild uses as an indexer note-prefix filter. A model
note ({"id": ...}) or an FL note ({"frr": ...}) never starts with those bytes.
"""
from __future__ import annotations

import json
import re
from typing import Optional

PREFIX = b'{"ps":1,'
MAX_NOTE = 1024

SINGLE_KEYS = ("ps", "id", "o", "c", "po", "pc", "md", "pk", "sig")
ALBUM_KEYS = ("ps", "pmr", "plc", "amd", "pk", "sig")

_H16 = re.compile(r"^[0-9a-f]{16}$")
_H64 = re.compile(r"^[0-9a-f]{64}$")
_H128 = re.compile(r"^[0-9a-f]{128}$")


class NoteError(ValueError):
    pass


def single_note(photo_id, orig_sha, cloaked_sha, orig_phash, cloaked_phash, digest, pk, sig,
                hash_only=False) -> dict:
    n = {"ps": 1, "id": photo_id, "o": orig_sha, "c": cloaked_sha}
    if not hash_only:
        n["po"] = orig_phash
        n["pc"] = cloaked_phash
    n.update({"md": digest, "pk": pk, "sig": sig})
    return n


def album_note(root, leaf_count, album_digest, pk, sig) -> dict:
    return {"ps": 1, "pmr": root, "plc": int(leaf_count), "amd": album_digest, "pk": pk, "sig": sig}


def encode(note: dict) -> bytes:
    """Serialise in the dict's insertion order (the builders above fix it) and check limits."""
    keys = tuple(note)
    order = SINGLE_KEYS if "id" in note else ALBUM_KEYS
    if keys != tuple(k for k in order if k in note):
        raise NoteError(f"note keys out of order: {keys}")
    b = json.dumps(note, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    if not b.startswith(PREFIX):
        raise NoteError("a PhotoShield note must start with {\"ps\":1,")
    if len(b) > MAX_NOTE:
        raise NoteError(f"note is {len(b)} bytes; Algorand allows {MAX_NOTE}")
    return b


def parse(raw: bytes) -> Optional[dict]:
    """Parse and validate a note's shape. None for anything that is not a PhotoShield note."""
    if not raw.startswith(PREFIX) or len(raw) > MAX_NOTE:
        return None
    try:
        n = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(n, dict) or n.get("ps") != 1:
        return None
    if not (isinstance(n.get("pk"), str) and _H64.match(n["pk"])
            and isinstance(n.get("sig"), str) and _H128.match(n["sig"])):
        return None
    if "id" in n:
        if set(n) - set(SINGLE_KEYS):
            return None
        ok = (_H16.match(str(n.get("id"))) and _H64.match(str(n.get("o"))) and _H64.match(str(n.get("md")))
              and (n.get("c") is None or _H64.match(str(n["c"]))))
        for k in ("po", "pc"):
            if k in n and n[k] is not None and not _H16.match(str(n[k])):
                return None
        if "po" in n and n["po"] is None:
            return None
        return n if ok else None
    if "pmr" in n:
        if set(n) - set(ALBUM_KEYS):
            return None
        ok = (_H64.match(str(n.get("pmr"))) and isinstance(n.get("plc"), int) and n["plc"] >= 1
              and _H64.match(str(n.get("amd"))))
        return n if ok else None
    return None


def hash_only(note: dict) -> bool:
    return "id" in note and "po" not in note
