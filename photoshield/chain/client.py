"""CLI client: a local signing key, and registering a manifest through the API.

The browser's key (design §9.2) is the main path. This is for scripted use. A CLI key file holds
a raw Ed25519 seed in hex, unencrypted, so keep it private. `load_key` also reads the browser's
exported key backup, so reports made on the command line can be signed with the registering key.
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


PKCS8_ED25519_PREFIX = bytes.fromhex("302e020100300506032b657004220420")


def _seed_from_browser_backup(d: dict, passphrase=None) -> bytes:
    """The Ed25519 seed inside the browser's key backup (design §9.2): PBKDF2-SHA256 + AES-GCM over
    PKCS#8, or an unencrypted PKCS#8 export. The passphrase comes from the argument, then
    $PHOTOSHIELD_KEY_PASSPHRASE, then a prompt."""
    import base64
    if d.get("privateKeyPkcs8"):
        pkcs8 = base64.b64decode(d["privateKeyPkcs8"])
    else:
        if passphrase is None:
            passphrase = os.environ.get("PHOTOSHIELD_KEY_PASSPHRASE")
        if passphrase is None:
            import getpass
            passphrase = getpass.getpass("Passphrase of the browser key backup: ")
        k = d["kdf"]
        if k.get("name") != "PBKDF2" or k.get("hash") != "SHA-256" or d.get("cipher") != "AES-GCM":
            raise ValueError("unsupported key backup format")
        aes_key = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), base64.b64decode(k["salt"]),
                                      int(k["iterations"]), 32)
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as e:              # pragma: no cover - environment dependent
            raise ValueError("reading an encrypted browser backup needs `pip install cryptography`") from e
        try:
            pkcs8 = AESGCM(aes_key).decrypt(base64.b64decode(d["iv"]), base64.b64decode(d["encryptedPrivateKey"]), None)
        except Exception:                     # noqa: BLE001 - InvalidTag and friends
            raise ValueError("wrong passphrase or damaged backup")
    if len(pkcs8) != 48 or pkcs8[:16] != PKCS8_ED25519_PREFIX:
        raise ValueError("the backup does not hold an Ed25519 PKCS#8 key")
    return pkcs8[16:]


def load_key(path: str, passphrase=None):
    """(signing_key, pk_hex) from a CLI key file, or from the browser's exported key backup."""
    d = json.load(open(path))
    if d.get("schema") == "photoshield/key-backup@1":
        sk, pk = signing.new_key(_seed_from_browser_backup(d, passphrase))
        if d.get("publicKeyHex") and pk != d["publicKeyHex"]:
            raise ValueError("key backup is corrupt: the key does not give the stored public key")
        return sk, pk
    if d.get("schema") != "photoshield/cli-key@1":
        raise ValueError("not a photoshield CLI key file or browser key backup")
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
