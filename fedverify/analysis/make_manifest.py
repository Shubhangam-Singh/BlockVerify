"""Freeze the study's results: SHA-256 of every results, table and figure file (Phase 11.4).

    python3 -m fedverify.analysis.make_manifest              # write MANIFEST.json, requirements-lock.txt, the block
    python3 -m fedverify.analysis.make_manifest --verify     # recompute everything; exit 1 on any difference

<results>/MANIFEST.json holds the Python version and platform, the `pip freeze` of this environment (saved as
<results>/requirements-lock.txt and hashed like every other file), and for each file under <results> its path,
size and SHA-256. Over the (path, sha256) leaves it computes a Merkle root with the existing round-commitment code
(fedverify.chain.commitment.round_merkle_root: leaf = SHA-256 of JSON([position, path, sha256])), then mines that
root into a block of a fresh local BlockVerify proof-of-work chain, saved as <results>/MANIFEST.chain.json, so the
frozen results are tamper-evident: change one byte of one file and the root no longer matches the mined block.

Left out: the manifest's own two files, and the live demo's folder (demo/heart_ours_lr_demo/), which every demo run
rewrites. _superseded/ is included (kept until the guide approves the numbers).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MANIFEST, CHAIN, LOCK = "MANIFEST.json", "MANIFEST.chain.json", "requirements-lock.txt"
EXCLUDE_PREFIXES = ("demo/heart_ours_lr_demo/",)
RUN_ID = "fedverify-results-manifest"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def listed(root: Path) -> list[str]:
    """Every file under root the manifest covers, as sorted posix paths relative to root."""
    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel in (MANIFEST, CHAIN) or rel.startswith(EXCLUDE_PREFIXES):
            continue
        out.append(rel)
    return sorted(out)


def file_entries(root: Path) -> list[dict]:
    return [{"path": rel, "bytes": (root / rel).stat().st_size, "sha256": sha256_file(root / rel)}
            for rel in listed(root)]


def merkle(entries: list[dict]) -> tuple[str, int]:
    from ..chain.commitment import round_merkle_root
    return round_merkle_root([(e["path"], e["sha256"]) for e in entries])


def mine(root_hash: str, count: int, difficulty: int = 2) -> tuple[dict, list[dict]]:
    """Mine the root into a fresh local proof-of-work chain; (anchor record, exported blocks)."""
    from ..chain.anchor import ChainAnchor
    anchor = ChainAnchor("local", checkpoint_every=0, difficulty=difficulty)
    rec = anchor.anchor_round(RUN_ID, 1, root_hash, count, leaves=None)
    blocks = [{"index": b.index, "timestamp": b.timestamp, "transactions": b.transactions,
               "previous_hash": b.previous_hash, "nonce": b.nonce, "hash": b.hash} for b in anchor.chain.chain]
    return {"backend": "local", "difficulty": difficulty, "block_index": blocks[-1]["index"],
            "block_hash": rec["txid"], "chain_file": CHAIN}, blocks


def check_chain(chain: dict, root_hash: str, count: int) -> list[str]:
    """Re-verify the exported blocks from their fields alone: hash, link, proof of work, and the root they hold."""
    from ..chain.anchor import _import_backend
    Block = _import_backend("blockchain").Block
    errors, prev = [], None
    d = int(chain["difficulty"])
    for b in chain["blocks"]:
        rebuilt = Block(b["index"], b["timestamp"], b["transactions"], b["previous_hash"], b["nonce"])
        if rebuilt.hash != b["hash"]:
            errors.append(f"block {b['index']}: stored hash does not match its contents")
        if not b["hash"].startswith("0" * d):
            errors.append(f"block {b['index']}: no proof of work at difficulty {d}")
        if prev is not None and b["previous_hash"] != prev:
            errors.append(f"block {b['index']}: does not link to the previous block")
        prev = b["hash"]
    held = [t for t in chain["blocks"][-1]["transactions"] if t.get("type") == "fl_round_commit"]
    if not held or held[0].get("root") != root_hash or int(held[0].get("leaf_count", -1)) != count:
        errors.append("the last block does not hold this manifest's Merkle root and file count")
    return errors


def build(root: Path, freeze: bool = True) -> dict:
    if freeze:                                         # the environment, frozen next to the results it produced
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (root / LOCK).write_text(out.stdout, encoding="utf-8")
    entries = file_entries(root)
    root_hash, count = merkle(entries)
    anchor, blocks = mine(root_hash, count)
    (root / CHAIN).write_text(json.dumps({"run_id": RUN_ID, "difficulty": anchor["difficulty"], "blocks": blocks},
                                         indent=1) + "\n", encoding="utf-8")
    lock = next((e for e in entries if e["path"] == LOCK), None)
    manifest = {"generated_by": "python3 -m fedverify.analysis.make_manifest",
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "python": sys.version, "platform": platform.platform(),
                "requirements_lock": lock, "excluded": [MANIFEST, CHAIN, *EXCLUDE_PREFIXES],
                "file_count": len(entries), "total_bytes": sum(e["bytes"] for e in entries),
                "merkle_root": root_hash, "leaf_count": count,
                "leaf_encoding": "SHA-256(JSON([position, path, sha256])), fedverify.chain.commitment",
                "anchor": anchor, "files": entries}
    (root / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def verify(root: Path) -> list[str]:
    """Every difference between the files and the manifest, the root and the mined block. Empty = intact."""
    m = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    problems = []
    want = {e["path"]: e for e in m["files"]}
    have = set(listed(root))
    for rel in sorted(set(want) - have):
        problems.append(f"missing: {rel}")
    for rel in sorted(have - set(want)):
        problems.append(f"not in the manifest: {rel}")
    for rel in sorted(set(want) & have):
        if sha256_file(root / rel) != want[rel]["sha256"]:
            problems.append(f"changed: {rel}")
    root_hash, count = merkle(m["files"])
    if root_hash != m["merkle_root"] or count != m["leaf_count"]:
        problems.append("the Merkle root does not match the listed files")
    chain = json.loads((root / CHAIN).read_text(encoding="utf-8"))
    problems += check_chain(chain, m["merkle_root"], m["leaf_count"])
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fedverify.analysis.make_manifest", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="fedverify/results")
    ap.add_argument("--verify", action="store_true", help="check the files against an existing manifest")
    a = ap.parse_args(argv)
    root = Path(a.results_dir)
    if a.verify:
        problems = verify(root)
        for p in problems[:50]:
            print(p)
        print(f"{'INTACT' if not problems else 'NOT INTACT'}: {len(problems)} problem(s)")
        return 1 if problems else 0
    m = build(root)
    print(f"wrote {root / MANIFEST}: {m['file_count']} files, {m['total_bytes'] / 1e6:.1f} MB")
    print(f"Merkle root {m['merkle_root']} over {m['leaf_count']} files, mined into block "
          f"#{m['anchor']['block_index']} ({m['anchor']['block_hash']}) of a local proof-of-work chain "
          f"(difficulty {m['anchor']['difficulty']}), saved as {root / CHAIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
