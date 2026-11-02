"""Locally computed content IDs: CIDv1, raw codec, sha2-256 multihash, multibase base32 lower (Revision 2 §3.2).

    cid = "b" + base32_lower_nopad(0x01 0x55 0x12 0x20 || sha256(data))

This is a LOCALLY COMPUTED content ID. It equals the CID a real IPFS node reports only for data stored as ONE raw
block (`ipfs add --cid-version 1 --raw-leaves` on data no larger than one chunk); larger data is chunked by IPFS and
gets a different (dag-pb) CID. Nothing here touches a network unless ipfs_add() is called explicitly (the optional
`--ipfs real` mode), and no result ever depends on a daemon being reachable.
"""
from __future__ import annotations

import base64
import hashlib
import json
import urllib.error
import urllib.request
import uuid

CIDV1, RAW_CODEC, SHA2_256, DIGEST_LEN = 0x01, 0x55, 0x12, 0x20
PREFIX = bytes([CIDV1, RAW_CODEC, SHA2_256, DIGEST_LEN])
LABEL = "locally computed content ID (CIDv1 raw sha2-256; equals an IPFS CID only for single-block data)"
DEFAULT_IPFS_API = "http://127.0.0.1:5001"


def cid_bytes(data: bytes) -> bytes:
    """Binary CIDv1: version, raw codec, sha2-256 multihash code, digest length, digest."""
    return PREFIX + hashlib.sha256(bytes(data)).digest()


def content_id(data: bytes) -> str:
    """Multibase base32-lower string form, e.g. 'bafkrei...'."""
    return "b" + base64.b32encode(cid_bytes(data)).decode("ascii").lower().rstrip("=")


def decode_cid(cid: str) -> bytes:
    """The sha2-256 digest inside a CID made by content_id(); raises ValueError for anything else."""
    if not isinstance(cid, str) or not cid.startswith("b"):
        raise ValueError("not a multibase base32 CID")
    body = cid[1:].upper()
    raw = base64.b32decode(body + "=" * (-len(body) % 8))
    if raw[:4] != PREFIX or len(raw) != 4 + DIGEST_LEN:
        raise ValueError("not a CIDv1 raw sha2-256 content ID")
    return raw[4:]


def verify_cid(data: bytes, cid: str) -> bool:
    try:
        return decode_cid(cid) == hashlib.sha256(bytes(data)).digest()
    except ValueError:
        return False


def ipfs_add(data: bytes, api: str = DEFAULT_IPFS_API, timeout: float = 5.0) -> dict:
    """OPTIONAL: upload to a local IPFS daemon as one raw CIDv1 block. Never raises; never required.

    Returns {"ok": True, "cid": <daemon's CID>, "matches_local": bool} or {"ok": False, "error": "..."}.
    """
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"update.bin\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + bytes(data) + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{api.rstrip('/')}/api/v0/add?cid-version=1&raw-leaves=true&pin=false",
                                 data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            got = json.loads(r.read().decode().strip().splitlines()[-1])
        cid = str(got.get("Hash", ""))
        return {"ok": bool(cid), "cid": cid, "matches_local": cid == content_id(data)}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
