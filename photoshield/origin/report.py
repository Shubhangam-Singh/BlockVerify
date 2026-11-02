"""Origin reports: which regions of a suspect image match a registered photo (design §15).

    build_report(original_bytes, suspect_bytes, manifest, thresholds, key=None)
    verify_report(report, suspect_bytes=None, recheck=False, anchored=None)

A report is signed by the owner key and carries a Merkle proof per tile, so anyone can check it
against the manifest digest anchored on chain without seeing the original photo. No floats
appear in a report (decimal strings instead), so the browser and Python sign the same bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Optional

import numpy as np

from ..provenance import hashing, manifest as M, regions as R, signing as S
from . import align as A

SCHEMA = "photoshield/origin-report@1"
DOMAIN = "photoshield/origin@1|"
STATUSES = ("match", "altered", "uncertain", "uncovered", "flat")
MIN_COVERAGE = 0.95
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALIBRATION = os.path.join(REPO, "photoshield", "calibration.json")


class OriginError(ValueError):
    """The inputs cannot produce, or do not support, an origin report."""


# ── thresholds (never hard-coded: calibration.json "tile", written by study e8 --install) ──
def load_thresholds(path: Optional[str] = None) -> dict:
    p = path or os.environ.get("PHOTOSHIELD_CALIBRATION") or CALIBRATION
    try:
        with open(p) as f:
            t = json.load(f)["tile"]
        out = {"tau_match": int(t["tau_match"]), "tau_alter": int(t["tau_alter"]),
               "k_min": int(t["k_min"]), "source": os.path.basename(p)}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise OriginError(f"no tile thresholds in {p}: run `python -m photoshield study e8 --install` first")
    _check_thresholds(out)
    return out


def _check_thresholds(t: dict) -> None:
    if not (0 <= t["tau_match"] < t["tau_alter"] <= 64 and t["k_min"] >= 1):
        raise OriginError("tile thresholds must satisfy 0 <= tau_match < tau_alter <= 64 and k_min >= 1")


def tile_status(value: str, suspect_hash: Optional[str], coverage: float, t: dict):
    """(status, distance or None) for one tile, by the table in design §15.4."""
    committed, textured = R.split_value(value)
    if not textured:
        return "flat", (hashing.hamming(committed, suspect_hash) if suspect_hash else None)
    if coverage < MIN_COVERAGE or suspect_hash is None:
        return "uncovered", None
    d = hashing.hamming(committed, suspect_hash)
    if d <= t["tau_match"]:
        return "match", d
    if d >= t["tau_alter"]:
        return "altered", d
    return "uncertain", d


def summarize(tiles: list, t: dict) -> dict:
    s = {k: 0 for k in STATUSES}
    for x in tiles:
        s[x["status"]] += 1
    s["verdict"] = "DERIVED_FROM_REGISTERED" if s["match"] >= t["k_min"] else "NOT_SHOWN"
    return s


def compare(original: np.ndarray, suspect: np.ndarray, t: dict, mode: str = "auto",
            M_fixed: Optional[np.ndarray] = None) -> dict:
    """Align, then score every tile. Returns tiles (without proofs), alignment and images."""
    h, w = original.shape[:2]
    g, values = R.region_values(original)
    if not g:
        raise OriginError("the original is too small for a region map (needs 64 px per side)")
    if M_fixed is not None:
        warped, mask = A.warp(suspect, M_fixed, w, h)
        al = {"method": "fixed", "M": M_fixed, "inliers": 0, "warped": warped, "mask": mask}
    else:
        al = A.align(original, suspect, mode)
    tiles = []
    for i, ((r, c, y0, y1, x0, x1), value) in enumerate(zip(R.tile_boxes(w, h, g), values)):
        cov = float(al["mask"][y0:y1, x0:x1].mean())
        sh = None
        if cov >= MIN_COVERAGE:
            sh = R.tile_value(al["warped"][y0:y1, x0:x1]).split(":")[0]
        status, d = tile_status(value, sh, cov, t)
        tiles.append({"i": i, "name": f"r{r}c{c}", "value": value, "suspect": sh, "d": d,
                      "coverage": f"{cov:.3f}", "status": status})
    return {"grid": g, "values": values, "tiles": tiles, "alignment": al}


# ── canonical form, signing ──────────────────────────────────────────────────
def core(report: dict) -> dict:
    return {k: v for k, v in report.items() if k != "sig"}


def report_digest(report: dict) -> str:
    return hashlib.sha256(M.canonical_json(core(report)).encode("utf-8")).hexdigest()


def report_message(report: dict) -> bytes:
    return (DOMAIN + report_digest(report)).encode("utf-8")


def sign(report: dict, signing_key) -> dict:
    report = dict(report)
    report["pk"] = signing_key.verify_key.encode().hex()
    report.pop("sig", None)
    report["sig"] = S.sign(signing_key, report_message(report))
    return report


# ── build ────────────────────────────────────────────────────────────────────
def build_report(original_bytes: bytes, suspect_bytes: bytes, manifest: dict, thresholds: dict,
                 signing_key=None, created: Optional[int] = None, mode: str = "auto") -> dict:
    """The full report (plus the alignment, for the heatmap) for one original/suspect pair."""
    _check_thresholds(thresholds)
    M.validate(manifest)
    reg = manifest.get("regions")
    if not reg:
        raise OriginError("this registration has no region map (it was made before PhotoShield v2); "
                          "re-register the photo to get origin reports")
    if hashing.sha256_bytes(original_bytes) != manifest["original"]["sha256"]:
        raise OriginError("the original photo does not match the manifest's SHA-256")
    original = hashing.decode_rgb(original_bytes)
    suspect = hashing.decode_rgb(suspect_bytes)
    res = compare(original, suspect, thresholds, mode)
    if res["grid"] != reg["grid"]:
        raise OriginError("the original's grid differs from the manifest's region map")
    root, leaves = R.commit(res["values"], res["grid"])
    if root != reg["root"]:
        raise OriginError("the original's region map does not fold to the anchored root")
    pr = R.proofs(leaves)
    for t, p in zip(res["tiles"], pr):
        t["proof"] = p
    digest = M.manifest_digest(manifest)
    al = res["alignment"]
    report = {
        "schema": SCHEMA, "created": int(created if created is not None else time.time()),
        "photo": {"photoId": M.photo_id(digest), "manifestDigest": digest, "manifest": manifest},
        "suspect": {"sha256": hashing.sha256_bytes(suspect_bytes),
                    "width": int(suspect.shape[1]), "height": int(suspect.shape[0])},
        "alignment": {"method": al["method"], "matrix": A.matrix_strings(al["M"]),
                      "inliers": int(al["inliers"])},
        "grid": res["grid"],
        "thresholds": {k: thresholds[k] for k in ("tau_match", "tau_alter", "k_min", "source")},
        "tiles": res["tiles"],
        "summary": summarize(res["tiles"], thresholds),
    }
    if signing_key is not None:
        report = sign(report, signing_key)
    return {"report": report, "alignment": al, "original": original}


# ── verify ───────────────────────────────────────────────────────────────────
def verify_report(report: dict, suspect_bytes: Optional[bytes] = None, recheck: bool = False,
                  anchored: Optional[dict] = None) -> dict:
    """Check a report. `anchored` = {"manifestDigest", "ownerPk"} read from the ledger/registry.

    Returns {"ok": bool, "checks": [{"name", "ok", "detail"}]}; ok is False if any check fails.
    """
    checks = []

    def add(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    try:
        add("schema", report.get("schema") == SCHEMA, report.get("schema", ""))
        man = report["photo"]["manifest"]
        M.validate(man)
        digest = M.manifest_digest(man)
        add("manifest digest", digest == report["photo"]["manifestDigest"], digest)
        add("photo id", M.photo_id(digest) == report["photo"]["photoId"], report["photo"]["photoId"])
        reg = man.get("regions") or {}
        g = report["grid"]
        tiles = report["tiles"]
        add("region map", reg.get("grid") == g and reg.get("n") == len(tiles) == g * g,
            f"{g}x{g} grid, {len(tiles)} tiles")
        bad = []
        for i, t in enumerate(tiles):
            name_ok = t["i"] == i and t["name"] == R.names(g)[i]
            from ..provenance.merkle import verify_proof
            if not (name_ok and verify_proof(R.leaf(i, t["name"], t["value"]), t["proof"], reg.get("root", ""))):
                bad.append(i)
        add("tile proofs fold to the anchored root", not bad, f"{len(tiles) - len(bad)}/{len(tiles)} tiles")
        thr = report["thresholds"]
        _check_thresholds(thr)
        wrong = []
        for t in tiles:
            cov = float(t["coverage"])
            status, d = tile_status(t["value"], t.get("suspect"), cov, thr)
            if status != t["status"] or (d is not None and d != t.get("d")):
                wrong.append(t["i"])
        add("tile statuses follow from the hashes", not wrong, f"{len(wrong)} inconsistent")
        add("summary", summarize(tiles, thr) == report["summary"], report["summary"].get("verdict", ""))
        if report.get("sig"):
            add("owner signature", S.verify(report.get("pk", ""), report["sig"], report_message(report)),
                str(report.get("pk", ""))[:16])
        else:
            add("owner signature", False, "the report is unsigned")
        if anchored is not None:
            add("manifest digest is the anchored one", anchored.get("manifestDigest") == digest)
            add("signed by the registering key", anchored.get("ownerPk") == report.get("pk"))
        if suspect_bytes is not None:
            add("suspect file is the one in the report",
                hashing.sha256_bytes(suspect_bytes) == report["suspect"]["sha256"])
            if recheck:
                suspect = hashing.decode_rgb(suspect_bytes)
                Mx = A.matrix_from_strings(report["alignment"]["matrix"])
                h = int(man["original"]["height"])
                w = int(man["original"]["width"])
                warped, mask = A.warp(suspect, Mx, w, h)
                diffs = []
                for (r, c, y0, y1, x0, x1), t in zip(R.tile_boxes(w, h, g), tiles):
                    if t["status"] in ("match", "altered", "uncertain"):
                        sh = R.tile_value(warped[y0:y1, x0:x1]).split(":")[0]
                        if sh != t["suspect"]:
                            diffs.append(t["i"])
                add("suspect tile hashes recomputed", not diffs,
                    f"{len(diffs)} tiles differ" + (" (a different resampler can shift a few bits)" if diffs else ""))
    except (KeyError, TypeError, ValueError, M.ManifestError) as e:
        add("well-formed", False, str(e)[:200])
    return {"ok": all(c["ok"] for c in checks), "checks": checks}
