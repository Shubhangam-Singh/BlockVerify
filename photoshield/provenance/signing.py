"""Owner signatures (design §5.4, §9.2): Ed25519, verified with PyNaCl.

PyNaCl is already installed on the backend as a dependency of py-algorand-sdk
(photoshield/INVENTORY.md §3.6), so verification needs no new requirement. Browsers sign with
WebCrypto Ed25519 (or tweetnacl). Ed25519 is deterministic (RFC 8032), so every implementation
produces the same 64-byte signature for the same key and message.

Every signed message carries a purpose prefix, so a signature made for one purpose cannot be
replayed as another (a register@1 signature does not verify as album@1).
"""
from __future__ import annotations

import re

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX128 = re.compile(r"^[0-9a-f]{128}$")

REGISTER = "photoshield/register@1|"
ALBUM = "photoshield/album@1|"
PROVE = "photoshield/prove@1|"


def register_message(manifest_digest: str) -> bytes:
    return (REGISTER + manifest_digest).encode("utf-8")


def album_message(album_root: str, leaf_count: int) -> bytes:
    return (ALBUM + album_root + "|" + str(int(leaf_count))).encode("utf-8")


def prove_message(photo_id: str, challenge: str) -> bytes:
    return (PROVE + photo_id + "|" + challenge).encode("utf-8")


def well_formed(pk_hex: str, sig_hex: str) -> bool:
    return (isinstance(pk_hex, str) and HEX64.match(pk_hex) is not None
            and isinstance(sig_hex, str) and HEX128.match(sig_hex) is not None)


def verify(pk_hex: str, sig_hex: str, message: bytes) -> bool:
    """True only for a well-formed key and a valid signature over exactly `message`."""
    if not well_formed(pk_hex, sig_hex):
        return False
    from nacl.exceptions import BadSignatureError
    from nacl.signing import VerifyKey
    try:
        VerifyKey(bytes.fromhex(pk_hex)).verify(message, bytes.fromhex(sig_hex))
        return True
    except (BadSignatureError, ValueError, TypeError):
        return False


def pk_fingerprint(pk_hex: str) -> str:
    """The short public identity shown for near matches (design §6.3)."""
    return pk_hex[:16]


# ── for tests, the CLI and the demo (never used by the server to sign for anyone) ──
def new_key(seed: bytes | None = None):
    """(signing_key, pk_hex). Pass a 32-byte seed for a reproducible TEST key."""
    from nacl.signing import SigningKey
    sk = SigningKey(seed) if seed is not None else SigningKey.generate()
    return sk, sk.verify_key.encode().hex()


def sign(signing_key, message: bytes) -> str:
    return signing_key.sign(message).signature.hex()
