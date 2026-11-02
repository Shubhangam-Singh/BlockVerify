"""CLI client: a local signing key, and registering a manifest through the API.

The browser's key (design §9.2) is the main path. This is for scripted use. The key file holds
a raw Ed25519 seed in hex, unencrypted, so keep it private.
"""
from __future__ import annotations

import hashlib
import json
import os

from ..provenance import manifest as M
from ..provenance import signing

POW_PREFIX = "000"          # backend/app.py POW_DIFFICULTY = 3


def keygen(path: str) -> str:
    seed = os.urandom(32)
    _, pk = signing.new_key(seed)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"schema": "photoshield/cli-key@1", "alg": "Ed25519",
                   "seed_hex": seed.hex(), "pk": pk}, f, indent=2)
    return pk


def load_key(path: str):
    d = json.load(open(path))
    if d.get("schema") != "photoshield/cli-key@1":
        raise ValueError("not a photoshield CLI key file")
    sk, pk = signing.new_key(bytes.fromhex(d["seed_hex"]))
    if pk != d.get("pk"):
        raise ValueError("key file is corrupt: seed does not give the stored public key")
    return sk, pk


def mine_pow(value: str) -> int:
    """The Hashcash nonce backend/app.py verify_pow expects: sha256(value + str(n)) starts 000."""
    n = 0
    while not hashlib.sha256(f"{value}{n}".encode()).hexdigest().startswith(POW_PREFIX):
        n += 1
    return n


def registration_payload(man: dict, signing_key, pk: str, privacy: str = "full") -> dict:
    M.validate(man)
    digest = M.manifest_digest(man)
    return {"manifest": man, "powNonce": mine_pow(man["original"]["sha256"]), "pk": pk,
            "sig": signing.sign(signing_key, signing.register_message(digest)), "privacy": privacy}


def register_manifest(manifest_path: str, api: str, token: str, key_path: str,
                      privacy: str = "full") -> dict:
    import requests
    man = json.load(open(manifest_path, encoding="utf-8"))
    sk, pk = load_key(key_path)
    body = registration_payload(man, sk, pk, privacy)
    r = requests.post(api.rstrip("/") + "/photo/register", json=body,
                      headers={"Authorization": f"Bearer {token}"}, timeout=60)
    try:
        return r.json()
    except ValueError:
        return {"success": False, "error": f"HTTP {r.status_code}"}
