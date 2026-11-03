"""The PhotoShield manifest: build, validate, canonicalise, digest (design §5.1).

    manifestDigest = sha256(canonical_json(manifest))
    canonical_json = json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)

**No floats anywhere in a manifest.** Python writes 1.0 as "1.0" and JavaScript as "1", so a
float would make the browser's digest differ from the server's and every owner signature would
fail. Numeric parameters are integers; measured metrics (PSNR, SSIM, ...) are decimal strings.

The backend imports this module: standard library only.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

SCHEMA = "photoshield/manifest@1"
MAX_TITLE = 120
MAX_CANONICAL_BYTES = 4096
MIN_SIDE = 32

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX16 = re.compile(r"^[0-9a-f]{16}$")
_BAD_TITLE = re.compile(r"[<>\x00-\x1f\x7f]")
_TOKEN = re.compile(r"^[A-Za-z0-9._:+\-]{1,40}$")
_DECIMAL = re.compile(r"^-?(\d{1,6})(\.\d{1,6})?$|^inf$")

CLOAK_INT = ("eps", "steps", "size", "seed", "threads", "alpha_div", "full_res")
CLOAK_STR = ("model", "tool_version", "jpeg_mode", "target")
CLOAK_METRICS = ("psnr", "ssim", "latent_shift", "recon_psnr_drop", "linf")
REGIONS_ALGO = "tile-phash@1"          # design §15.2


class ManifestError(ValueError):
    """The manifest does not match schema photoshield/manifest@1."""


# ── canonical form ───────────────────────────────────────────────────────────
def _reject_floats(obj: Any, path: str = "$") -> None:
    if isinstance(obj, float):
        raise ManifestError(f"{path}: floats are not allowed in a manifest (use a decimal string)")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ManifestError(f"{path}: keys must be strings")
            _reject_floats(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _reject_floats(v, f"{path}[{i}]")


def canonical_json(obj: Any) -> str:
    _reject_floats(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()


def photo_id(digest: str) -> str:
    """Deterministic id, so a ledger rebuild recreates the same ids (design §5.2)."""
    return hashlib.sha256(("ps-id|" + digest).encode("utf-8")).hexdigest()[:16]


def metric(x: Optional[float], places: int = 4) -> Optional[str]:
    """Format a measured value as the decimal string a manifest stores."""
    if x is None:
        return None
    if x == float("inf"):
        return "inf"
    return f"{x:.{places}f}"


# ── validation ───────────────────────────────────────────────────────────────
def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ManifestError(msg)


def _image_part(d: Any, name: str, cloaked: bool) -> None:
    _need(isinstance(d, dict), f"{name} must be an object")
    allowed = {"sha256", "phash", "format"} if cloaked else {"sha256", "phash", "width", "height"}
    extra = set(d) - allowed
    _need(not extra, f"{name}: unknown field(s) {sorted(extra)}")
    _need(isinstance(d.get("sha256"), str) and HEX64.match(d["sha256"]) is not None,
          f"{name}.sha256 must be 64 lowercase hex characters")
    _need(isinstance(d.get("phash"), str) and HEX16.match(d["phash"]) is not None,
          f"{name}.phash must be 16 lowercase hex characters")
    if cloaked:
        _need(d.get("format") == "png", f"{name}.format must be \"png\"")
    else:
        for k in ("width", "height"):
            v = d.get(k)
            _need(isinstance(v, int) and not isinstance(v, bool) and MIN_SIDE <= v <= 100_000,
                  f"{name}.{k} must be an integer between {MIN_SIDE} and 100000")


def _cloak_part(c: Any) -> None:
    _need(isinstance(c, dict), "cloak must be an object or null")
    allowed = set(CLOAK_INT) | set(CLOAK_STR) | set(CLOAK_METRICS) | {"eot"}
    extra = set(c) - allowed
    _need(not extra, f"cloak: unknown field(s) {sorted(extra)}")
    for k in CLOAK_INT:
        if k in c:
            _need(isinstance(c[k], int) and not isinstance(c[k], bool) and 0 <= c[k] <= 1_000_000,
                  f"cloak.{k} must be a non-negative integer")
    for k in CLOAK_STR:
        if k in c:
            _need(isinstance(c[k], str) and _TOKEN.match(c[k]) is not None,
                  f"cloak.{k} must be a short token")
    for k in CLOAK_METRICS:
        if k in c and c[k] is not None:
            _need(isinstance(c[k], str) and _DECIMAL.match(c[k]) is not None,
                  f"cloak.{k} must be a decimal string such as \"34.1200\"")
    if "eot" in c:
        _need(isinstance(c["eot"], list) and len(c["eot"]) <= 8
              and all(isinstance(t, str) and _TOKEN.match(t) for t in c["eot"]),
              "cloak.eot must be a short list of tokens")
    _need("model" in c and "eps" in c, "cloak needs at least model and eps")


def _regions_part(r: Any) -> None:
    _need(isinstance(r, dict), "regions must be an object")
    _need(set(r) == {"algo", "grid", "n", "root"}, "regions needs exactly algo, grid, n and root")
    _need(r["algo"] == REGIONS_ALGO, f"regions.algo must be {REGIONS_ALGO!r}")
    g = r["grid"]
    _need(isinstance(g, int) and not isinstance(g, bool) and 2 <= g <= 16, "regions.grid must be 2..16")
    _need(r["n"] == g * g and not isinstance(r["n"], bool), "regions.n must equal grid * grid")
    _need(isinstance(r["root"], str) and HEX64.match(r["root"]) is not None,
          "regions.root must be 64 lowercase hex characters")


def validate(manifest: Any) -> dict:
    """Return the manifest unchanged if valid; raise ManifestError otherwise."""
    _need(isinstance(manifest, dict), "manifest must be a JSON object")
    _reject_floats(manifest)
    allowed = {"schema", "title", "original", "cloaked", "cloak", "regions"}
    extra = set(manifest) - allowed
    _need(not extra, f"unknown field(s) {sorted(extra)}")
    _need(manifest.get("schema") == SCHEMA, f"schema must be {SCHEMA!r}")
    title = manifest.get("title")
    _need(isinstance(title, str) and 1 <= len(title) <= MAX_TITLE,
          f"title must be 1 to {MAX_TITLE} characters")
    _need(_BAD_TITLE.search(title) is None, "title may not contain <, > or control characters")
    _image_part(manifest.get("original"), "original", cloaked=False)
    cloaked, cloak = manifest.get("cloaked"), manifest.get("cloak")
    if cloaked is not None:
        _image_part(cloaked, "cloaked", cloaked=True)
        _need(cloaked["sha256"] != manifest["original"]["sha256"],
              "cloaked.sha256 equals original.sha256")
    if cloak is not None:
        _need(cloaked is not None, "cloak parameters given without a cloaked image")
        _cloak_part(cloak)
    if "regions" in manifest:
        _regions_part(manifest["regions"])
    _need(len(canonical_json(manifest).encode("utf-8")) <= MAX_CANONICAL_BYTES,
          f"manifest is larger than {MAX_CANONICAL_BYTES} bytes")
    return manifest


def build(title: str, original: dict, cloaked: Optional[dict] = None,
          cloak: Optional[dict] = None, regions: Optional[dict] = None) -> dict:
    m = {"schema": SCHEMA, "title": title, "original": original,
         "cloaked": cloaked, "cloak": cloak}
    if regions is not None:              # absent, not null: v1 manifests keep their digests
        m["regions"] = regions
    return validate(m)
