"""PhotoShield: photo cloaking + provenance on the BlockVerify integrity layer.

See PHOTOSHIELD_DESIGN.md at the repository root. Subpackages:

    provenance/  hashing (SHA-256 + canonical pHash), manifest, Ed25519 verify, Merkle
    cloak/       PGD + EOT against an image encoder (local, CPU)
    chain/       anchoring backends and the ledger rebuild
    experiments/ the E1-E7 study; analysis/ turns its results into tables

Modules the Flask backend imports (provenance.manifest, provenance.signing, provenance.merkle,
chain.ledger_sync) import nothing heavier than the standard library, PyNaCl and requests, so
the free-tier backend never loads torch, numpy or PIL on their account.
"""
__version__ = "0.1.0"
