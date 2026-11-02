"""PhotoShield routes (PHOTOSHIELD_DESIGN.md §6). Additive Blueprint, the fl_routes.py shape.

The server never receives pixels on the main path. It receives a manifest (hashes, sizes,
cloak parameters), a proof-of-work nonce and the owner's Ed25519 signature, and it anchors a
note on Algorand. Verification is public and uses only hashes the browser computed.

Trust:
- The owner signature binds the claim to a key the server does not hold.
- The ledger is the source of truth; this registry is a cache rebuilt from it on startup.
- Clients re-check every anchor against the public indexer themselves.

Error envelope: {"success": false, "error": "..."} with 400 / 401 / 403 / 404 / 409 / 429 /
500 / 501 / 503.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sys
import threading
from time import time

from flask import Blueprint, g, jsonify, request

from auth import decode_token, require_auth

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from photoshield.chain import notes as N                      # noqa: E402  (stdlib only)
from photoshield.chain.ledger_sync import LedgerSync, backoff_delay   # noqa: E402
from photoshield.provenance import manifest as M              # noqa: E402
from photoshield.provenance import signing as S               # noqa: E402

photo_bp = Blueprint("photo", __name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PHOTO_FILE = os.path.join(DATA_DIR, "photo_registry.json")
LEDGER_CACHE = os.path.join(DATA_DIR, "photo_ledger_cache.json")
CALIBRATION_FILE = os.environ.get(
    "PHOTOSHIELD_CALIBRATION", os.path.join(_REPO_ROOT, "photoshield", "calibration.json"))
os.makedirs(DATA_DIR, exist_ok=True)

INDEXER = "https://testnet-idx.algonode.cloud"
MAX_ALBUM = 100
WRITE_LIMIT, WRITE_WINDOW = 10, 60.0           # per user, own dict (not the model quota)
VERIFY_LIMIT, VERIFY_WINDOW = 30, 60.0         # per IP, slows registry probing
CHALLENGE_TTL = 300.0
NEAR_WARNING = "Perceptual match is evidence, not proof."
_HEX16 = re.compile(r"^[0-9a-f]{16}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

_lock = threading.RLock()
_state = {"loaded": False, "db": None, "idx": None, "sync": None, "sync_thread": None,
          "sync_started": False, "calibration": None}
_write_rate: dict = {}
_verify_rate: dict = {}
_challenges: dict = {}


def _err(msg, code):
    return jsonify({"success": False, "error": msg}), code


def _cfg(name, default=None):
    return os.environ.get(name, default)


# ── storage ──────────────────────────────────────────────────────────────────
def _empty_db():
    return {"version": 1, "photos": {}, "albums": {}, "pk_users": {}}


def _load_db():
    if not os.path.exists(PHOTO_FILE):
        return _empty_db()
    try:
        with open(PHOTO_FILE) as f:
            db = json.load(f)
        return db if db.get("version") == 1 else _empty_db()
    except (OSError, json.JSONDecodeError):
        return _empty_db()


def _save_db():
    tmp = PHOTO_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_state["db"], f, indent=1)
    os.replace(tmp, PHOTO_FILE)            # atomic; a concurrent GET never sees a half file


def _reindex():
    by_sha, phashes = {}, []
    for pid, r in _state["db"]["photos"].items():
        for part, kind in (("original", "original"), ("cloaked", "cloaked")):
            p = r.get(part)
            if not p:
                continue
            by_sha.setdefault(p["sha256"], (pid, kind))
            if p.get("phash"):
                phashes.append((int(p["phash"], 16), pid, kind))
    _state["idx"] = {"by_sha": by_sha, "phash": phashes}


def _db():
    with _lock:
        if not _state["loaded"]:
            _state["db"] = _load_db()
            _reindex()
            _state["calibration"] = _load_calibration()
            _state["loaded"] = True
        return _state["db"]


def reset_for_tests(path=None, cache=None, calibration=None):
    """Tests only: point the module at temporary files and forget all state."""
    global PHOTO_FILE, LEDGER_CACHE, CALIBRATION_FILE
    with _lock:
        if path:
            PHOTO_FILE = path
        if cache:
            LEDGER_CACHE = cache
        if calibration is not None:
            CALIBRATION_FILE = calibration
        _state.update(loaded=False, db=None, idx=None, sync=None, sync_thread=None,
                      sync_started=False, calibration=None)
        _write_rate.clear()
        _verify_rate.clear()
        _challenges.clear()


# ── calibration (thresholds are never hard-coded, design §6.3) ───────────────
def _load_calibration():
    try:
        with open(CALIBRATION_FILE) as f:
            c = json.load(f)
        near, far = int(c["tau_near"]), int(c["tau_far"])
        if 0 <= near <= far <= 64:
            out = {"tau_near": near, "tau_far": far, "source": os.path.basename(CALIBRATION_FILE),
                   "fpr_target": c.get("fpr_target"), "dataset": c.get("dataset")}
            out["tile"] = _tile_calibration(c.get("tile"))     # origin reports (design §15.4)
            return out
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        pass
    return None


def _tile_calibration(t):
    try:
        out = {k: int(t[k]) for k in ("tau_match", "tau_alter", "delta_level", "k_min")}
    except (KeyError, TypeError, ValueError):
        return None
    if 0 <= out["tau_match"] < out["tau_alter"] <= 64 and out["delta_level"] >= 1 and out["k_min"] >= 1:
        out["source"] = os.path.basename(CALIBRATION_FILE)
        return out
    return None


# ── rate limits ──────────────────────────────────────────────────────────────
def _rate_ok(table, key, limit, window):
    now = time()
    hits = [t for t in table.get(key, []) if now - t < window]
    if len(hits) >= limit:
        table[key] = hits
        return False
    hits.append(now)
    table[key] = hits
    if len(table) > 5000:
        for k in list(table)[:1000]:
            table.pop(k, None)
    return True


def _client_ip():
    return request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()


def _requester():
    h = request.headers.get("Authorization", "")
    if h.startswith("Bearer "):
        p = decode_token(h[7:])
        if p:
            return p.get("sub")                 # tokens carry the user in `sub`
    return None


# ── anchoring ────────────────────────────────────────────────────────────────
def _anchor(note: dict) -> dict:
    """PHOTOSHIELD_ANCHOR: algorand (default), local (PoW chain in memory) or mock."""
    backend = _cfg("PHOTOSHIELD_ANCHOR", "algorand")
    if backend == "algorand":
        import algorand_client
        res = algorand_client.broadcast_photo_anchor(note)
        res["backend"] = "algorand"
        return res
    from photoshield.chain.anchor import PhotoAnchor
    if "anchor" not in _state or _state.get("anchor_backend") != backend:
        _state["anchor"] = PhotoAnchor(backend)
        _state["anchor_backend"] = backend
    return _state["anchor"].anchor(note)


def _indexer_url(txid):
    if not txid or len(str(txid)) != 52:
        return None
    return f"{INDEXER}/v2/transactions/{txid}"


def _verify_pow(value, nonce):
    import app as _app                         # lazy: app imports this blueprint
    try:
        return _app.verify_pow(value, int(nonce))
    except (TypeError, ValueError):
        return False


# ── ledger rebuild (design §6.8) ─────────────────────────────────────────────
def _addresses():
    addrs = []
    try:
        import algorand_client
        addrs.append(algorand_client.ADDRESS)
    except Exception:
        pass
    addrs += [a.strip() for a in (_cfg("PHOTOSHIELD_ANCHOR_ADDRESSES", "") or "").split(",") if a.strip()]
    return addrs


def _merge_ledger(entries):
    """Fold accepted anchors into the registry. The chain wins on hashes, key and round."""
    db = _state["db"]
    changed = False
    for e in entries:
        n = e["note"]
        if "id" in n:
            pid = n["id"]
            r = db["photos"].get(pid)
            if r is None:
                r = {"photoId": pid, "ownerPk": n["pk"], "ownerUser": None, "title": None,
                     "registeredAt": e.get("round_time"), "manifestDigest": n["md"],
                     "signature": n["sig"], "privacy": "hash-only" if N.hash_only(n) else "full",
                     "original": {"sha256": n["o"], "phash": n.get("po")},
                     "cloaked": {"sha256": n["c"], "phash": n.get("pc")} if n.get("c") else None,
                     "cloak": None, "albumId": None, "albumIndex": None, "albumProof": None,
                     "algoTxId": e["txid"], "algoRound": e.get("round"), "isActive": True,
                     "source": "ledger"}
                db["photos"][pid] = r
                changed = True
            elif r.get("manifestDigest") == n["md"] and r.get("algoTxId") != e["txid"]:
                r.update(algoTxId=e["txid"], algoRound=e.get("round"), ownerPk=n["pk"])
                changed = True
        else:
            aid = _album_id(n["pmr"])
            if aid not in db["albums"]:
                db["albums"][aid] = {"albumId": aid, "root": n["pmr"], "count": n["plc"],
                                     "albumDigest": n["amd"], "ownerPk": n["pk"],
                                     "signature": n["sig"], "title": None, "photoIds": [],
                                     "algoTxId": e["txid"], "algoRound": e.get("round"),
                                     "registeredAt": e.get("round_time"), "source": "ledger"}
                changed = True
    if changed:
        _reindex()
        _save_db()


def _sync_once(attempt=0):
    ls = _state["sync"]
    try:
        entries = ls.sync()
        with _lock:
            _merge_ledger(entries)
    except Exception:
        delay = backoff_delay(attempt)
        t = threading.Timer(delay, _sync_once, kwargs={"attempt": attempt + 1})
        t.daemon = True
        t.start()


def _ensure_sync():
    """Start the ledger rebuild lazily, on the first photo request, never at import."""
    if _cfg("PHOTOSHIELD_LEDGER_SYNC", "1") == "0":
        return
    with _lock:
        if _state["sync_started"]:
            return
        _state["sync_started"] = True
        _state["sync"] = LedgerSync(_addresses(), indexer=_cfg("PHOTOSHIELD_INDEXER", INDEXER),
                                    cache_path=LEDGER_CACHE)
        # Serve whatever the cache already holds immediately.
        _merge_ledger(_state["sync"].entries())
        t = threading.Thread(target=_sync_once, daemon=True, name="photoshield-ledger-sync")
        _state["sync_thread"] = t
        t.start()


def _sync_status():
    s = _state.get("sync")
    if s is None:
        return "disabled" if _cfg("PHOTOSHIELD_LEDGER_SYNC", "1") == "0" else "idle"
    return s.status["state"]


@photo_bp.before_request
def _boot():
    _db()
    _ensure_sync()


# ── helpers ──────────────────────────────────────────────────────────────────
def _album_id(root):
    return hashlib.sha256(("ps-album|" + root).encode()).hexdigest()[:16]


def _link(pk, user):
    if not user:
        return
    users = _state["db"]["pk_users"].setdefault(pk, [])
    if user not in users:
        users.append(user)


def _owner_user(r):
    if r.get("ownerUser"):
        return r["ownerUser"]
    users = _state["db"]["pk_users"].get(r.get("ownerPk"), [])
    return users[0] if users else None


def _public(r, requester=None):
    out = {k: r.get(k) for k in ("photoId", "title", "registeredAt", "privacy", "manifestDigest",
                                 "albumId", "albumIndex", "algoTxId", "algoRound", "isActive",
                                 "source")}
    out["original"] = {"sha256": r["original"]["sha256"]}
    out["cloaked"] = {"sha256": r["cloaked"]["sha256"]} if r.get("cloaked") else None
    out["hasPhash"] = bool(r["original"].get("phash"))
    out["cloak"] = r.get("cloak")
    out["regions"] = r.get("regions")          # region-map commitment (root only, no tile values)
    out["ownerPkFp"] = S.pk_fingerprint(r["ownerPk"])
    out["ownerPk"] = r["ownerPk"]
    out["indexerUrl"] = _indexer_url(r.get("algoTxId"))
    owner = _owner_user(r)
    if requester and owner == requester:
        out["ownerUser"] = owner
        out["original"]["phash"] = r["original"].get("phash")
        if r.get("cloaked"):
            out["cloaked"]["phash"] = r["cloaked"].get("phash")
    return out


def _read_registration(body):
    man = body.get("manifest")
    try:
        M.validate(man)
    except M.ManifestError as e:
        raise ValueError(f"Invalid manifest: {e}")
    privacy = body.get("privacy", "full")
    if privacy not in ("full", "hash-only"):
        raise ValueError("privacy must be 'full' or 'hash-only'")
    return man, M.manifest_digest(man), privacy


def _record(man, digest, pk, sig, privacy, user, anchor, album=None):
    hide = privacy == "hash-only"
    o, c = man["original"], man.get("cloaked")
    # The registry keeps the pHash even for hash-only photos until the next restart; only the
    # chain omits it (design §6.9).
    return {"photoId": M.photo_id(digest), "ownerPk": pk, "ownerUser": user,
            "title": man["title"], "registeredAt": int(time()), "manifestDigest": digest,
            "signature": sig, "privacy": privacy,
            "original": {"sha256": o["sha256"], "phash": o["phash"],
                         "width": o["width"], "height": o["height"]},
            "cloaked": {"sha256": c["sha256"], "phash": c["phash"]} if c else None,
            "cloak": man.get("cloak"), "regions": man.get("regions"),
            "albumId": album and album["albumId"], "albumIndex": album and album["index"],
            "albumProof": album and album["proof"],
            "algoTxId": anchor.get("txid"), "algoRound": anchor.get("round"),
            "anchorBackend": anchor.get("backend"), "isActive": True, "source": "register",
            "hashOnlyOnChain": hide}


def _conflict(man, pk):
    """(existing_record, kind) if the original's hash is already registered."""
    hit = _state["idx"]["by_sha"].get(man["original"]["sha256"])
    if not hit:
        return None
    return _state["db"]["photos"][hit[0]]


# ── POST /api/photo/register ─────────────────────────────────────────────────
@photo_bp.route("/api/photo/register", methods=["POST"])
@require_auth
def register_photo():
    body = request.get_json(silent=True) or {}
    try:
        man, digest, privacy = _read_registration(body)
    except ValueError as e:
        return _err(str(e), 400)
    pk, sig = body.get("pk"), body.get("sig")
    if not S.well_formed(pk, sig):
        return _err("pk must be 64 and sig 128 lowercase hex characters.", 400)
    if body.get("powNonce") is None:
        return _err("Proof-of-Work required: mine a nonce so SHA-256(original.sha256 + nonce) "
                    "starts with '000'.", 400)
    if not _verify_pow(man["original"]["sha256"], body["powNonce"]):
        return _err("Proof-of-Work verification failed. Invalid nonce.", 400)
    if not S.verify(pk, sig, S.register_message(digest)):
        return _err("Owner signature does not verify for this manifest.", 400)
    if not _rate_ok(_write_rate, g.user, WRITE_LIMIT, WRITE_WINDOW):
        return _err(f"Rate limit exceeded: max {WRITE_LIMIT} photo registrations per "
                    f"{int(WRITE_WINDOW)}s.", 429)

    with _lock:
        prev = _conflict(man, pk)
        if prev is not None:
            if prev["ownerPk"] == pk:
                _link(pk, g.user)
                return jsonify({"success": True, "duplicate": True, "photoId": prev["photoId"],
                                "algoTxId": prev.get("algoTxId"), "algoRound": prev.get("algoRound"),
                                "indexerUrl": _indexer_url(prev.get("algoTxId")),
                                "message": "Already registered under this key."}), 200
            return jsonify({"success": False, "error": "This photo was already registered by another key.",
                            "photoId": prev["photoId"], "registeredAt": prev.get("registeredAt"),
                            "algoTxId": prev.get("algoTxId"), "algoRound": prev.get("algoRound"),
                            "ownerPkFp": S.pk_fingerprint(prev["ownerPk"]),
                            "indexerUrl": _indexer_url(prev.get("algoTxId"))}), 409

    pid = M.photo_id(digest)
    c = man.get("cloaked")
    note = N.single_note(pid, man["original"]["sha256"], c["sha256"] if c else None,
                         man["original"]["phash"], c["phash"] if c else None, digest, pk, sig,
                         hash_only=(privacy == "hash-only"))
    res = _anchor(note)
    if not res.get("success"):
        return _err(res.get("error") or "Anchoring failed.", 502)

    with _lock:
        rec = _record(man, digest, pk, sig, privacy, g.user, res)
        _state["db"]["photos"][pid] = rec
        _link(pk, g.user)
        _reindex()
        _save_db()
    return jsonify({"success": True, "photoId": pid, "manifestDigest": digest,
                    "algoTxId": rec["algoTxId"], "algoRound": rec["algoRound"],
                    "anchorBackend": res.get("backend"), "noteBytes": res.get("bytes_written"),
                    "indexerUrl": _indexer_url(rec["algoTxId"]),
                    "message": "Photo anchored."}), 200


# ── POST /api/photo/register-batch (album) ───────────────────────────────────
def album_manifest(title, items):
    """The album manifest whose digest is anchored as `amd`."""
    return {"schema": "photoshield/album@1", "title": title, "photos": [[p, d] for p, d in items]}


@photo_bp.route("/api/photo/register-batch", methods=["POST"])
@require_auth
def register_album():
    from photoshield.provenance import merkle
    body = request.get_json(silent=True) or {}
    title = body.get("albumTitle")
    if not isinstance(title, str) or not (1 <= len(title) <= M.MAX_TITLE) or re.search(r"[<>\x00-\x1f\x7f]", title):
        return _err("albumTitle must be 1 to 120 characters without <, > or control characters.", 400)
    mans = body.get("manifests")
    if not isinstance(mans, list) or not (1 <= len(mans) <= MAX_ALBUM):
        return _err(f"manifests must be a list of 1 to {MAX_ALBUM} manifests.", 400)
    privacy = body.get("privacy", "full")
    if privacy not in ("full", "hash-only"):
        return _err("privacy must be 'full' or 'hash-only'.", 400)
    items, seen = [], set()
    for i, m in enumerate(mans):
        try:
            M.validate(m)
        except M.ManifestError as e:
            return _err(f"Invalid manifest #{i}: {e}", 400)
        if m["original"]["sha256"] in seen:
            return _err(f"Manifest #{i} repeats a photo already in this album.", 400)
        seen.add(m["original"]["sha256"])
        d = M.manifest_digest(m)
        items.append((M.photo_id(d), d))
    root, leaves, count = merkle.album_root(items)
    pk, sig = body.get("pk"), body.get("sig")
    if not S.well_formed(pk, sig):
        return _err("pk must be 64 and sig 128 lowercase hex characters.", 400)
    if not _verify_pow(root, body.get("powNonce")):
        return _err("Proof-of-Work over the album root is missing or invalid.", 400)
    if not S.verify(pk, sig, S.album_message(root, count)):
        return _err("Owner signature does not verify for this album root.", 400)
    if not _rate_ok(_write_rate, g.user, WRITE_LIMIT, WRITE_WINDOW):
        return _err("Rate limit exceeded.", 429)
    with _lock:
        for m in mans:
            prev = _conflict(m, pk)
            if prev is not None and prev["ownerPk"] != pk:
                return jsonify({"success": False, "error": "A photo in this album was already registered by another key.",
                                "photoId": prev["photoId"], "algoTxId": prev.get("algoTxId")}), 409
    amd = M.manifest_digest(album_manifest(title, items))
    res = _anchor(N.album_note(root, count, amd, pk, sig))
    if not res.get("success"):
        return _err(res.get("error") or "Anchoring failed.", 502)
    aid = _album_id(root)
    proofs = merkle.album_proofs(items)
    with _lock:
        db = _state["db"]
        db["albums"][aid] = {"albumId": aid, "root": root, "count": count, "albumDigest": amd,
                             "ownerPk": pk, "signature": sig, "title": title,
                             "photoIds": [p for p, _ in items], "algoTxId": res.get("txid"),
                             "algoRound": res.get("round"), "registeredAt": int(time()),
                             "source": "register"}
        out = []
        for i, (m, (pid, d)) in enumerate(zip(mans, items)):
            if pid not in db["photos"]:
                db["photos"][pid] = _record(m, d, pk, sig, privacy, g.user, res,
                                            album={"albumId": aid, "index": i, "proof": proofs[i]})
            out.append({"photoId": pid, "index": i, "leaf": leaves[i], "proof": proofs[i]})
        _link(pk, g.user)
        _reindex()
        _save_db()
    return jsonify({"success": True, "albumId": aid, "root": root, "count": count,
                    "albumDigest": amd, "algoTxId": res.get("txid"), "algoRound": res.get("round"),
                    "indexerUrl": _indexer_url(res.get("txid")), "photos": out}), 200


# ── POST /api/photo/verify (public) ──────────────────────────────────────────
@photo_bp.route("/api/photo/verify", methods=["POST"])
def verify_photo():
    if not _rate_ok(_verify_rate, _client_ip(), VERIFY_LIMIT, VERIFY_WINDOW):
        return _err("Too many verifications. Please wait a minute.", 429)
    body = request.get_json(silent=True) or {}
    sha, ph = body.get("sha256"), body.get("phash")
    if not (isinstance(sha, str) and _HEX64.match(sha)):
        return _err("sha256 must be 64 lowercase hex characters.", 400)
    if ph is not None and not (isinstance(ph, str) and _HEX16.match(ph)):
        return _err("phash must be 16 lowercase hex characters.", 400)
    cal = _state["calibration"]
    base = {"success": True, "ledgerSync": _sync_status(),
            "nearMatch": "calibrated" if cal else "uncalibrated"}
    if cal:
        base["thresholds"] = {"tauNear": cal["tau_near"], "tauFar": cal["tau_far"],
                              "source": cal["source"]}
    with _lock:
        hit = _state["idx"]["by_sha"].get(sha)
        if hit:
            r = _state["db"]["photos"][hit[0]]
            base.update(verdict="EXACT_ORIGINAL" if hit[1] == "original" else "EXACT_CLOAKED",
                        photoId=r["photoId"], ownerUser=_owner_user(r),
                        ownerPkFp=S.pk_fingerprint(r["ownerPk"]), registeredAt=r.get("registeredAt"),
                        algoTxId=r.get("algoTxId"), algoRound=r.get("algoRound"),
                        indexerUrl=_indexer_url(r.get("algoTxId")), distance=0)
            return jsonify(base), 200
        if ph is None or not cal:
            base.update(verdict="UNKNOWN")
            return jsonify(base), 200
        q = int(ph, 16)
        best = None
        for val, pid, kind in _state["idx"]["phash"]:
            d = bin(q ^ val).count("1")
            if best is None or d < best[0]:
                best = (d, pid, kind)
        if best is None or best[0] > cal["tau_far"]:
            base.update(verdict="UNKNOWN", nearestDistance=best[0] if best else None)
            return jsonify(base), 200
        r = _state["db"]["photos"][best[1]]
        base.update(verdict="DERIVED" if best[0] <= cal["tau_near"] else "SIMILAR",
                    distance=best[0], matchedPart=best[2], photoId=r["photoId"],
                    ownerPkFp=S.pk_fingerprint(r["ownerPk"]), registeredAt=r.get("registeredAt"),
                    algoTxId=r.get("algoTxId"), algoRound=r.get("algoRound"),
                    indexerUrl=_indexer_url(r.get("algoTxId")), warning=NEAR_WARNING)
        return jsonify(base), 200


# ── POST /api/photo/rehydrate (design §6.9) ──────────────────────────────────
@photo_bp.route("/api/photo/rehydrate", methods=["POST"])
@require_auth
def rehydrate():
    from photoshield.provenance import merkle
    body = request.get_json(silent=True) or {}
    with _lock:
        db = _state["db"]
        if "album" in body:
            album, mans = body.get("album"), body.get("manifests")
            if not isinstance(album, dict) or not isinstance(mans, list) or not mans:
                return _err("album rehydration needs {album, manifests[]}.", 400)
            try:
                items = []
                for m in mans:
                    M.validate(m)
                    d = M.manifest_digest(m)
                    items.append((M.photo_id(d), d))
                root, _, count = merkle.album_root(items)
                amd = M.manifest_digest(album)
            except (M.ManifestError, ValueError) as e:
                return _err(f"Invalid album: {e}", 400)
            a = db["albums"].get(_album_id(root))
            if a is None or a["albumDigest"] != amd or a["count"] != count:
                return _err("These manifests do not match any anchored album.", 404)
            if album.get("photos") != [[p, d] for p, d in items]:
                return _err("The album manifest does not list these photos in this order.", 400)
            proofs = merkle.album_proofs(items)
            a.update(title=album.get("title"), photoIds=[p for p, _ in items])
            for i, (m, (pid, d)) in enumerate(zip(mans, items)):
                rec = _record(m, d, a["ownerPk"], a["signature"], "full", None,
                              {"txid": a["algoTxId"], "round": a["algoRound"]},
                              album={"albumId": a["albumId"], "index": i, "proof": proofs[i]})
                rec.update(source="rehydrate", ownerUser=None)
                db["photos"][pid] = rec
            _reindex()
            _save_db()
            return jsonify({"success": True, "albumId": a["albumId"], "restored": len(items)}), 200

        man = body.get("manifest")
        try:
            M.validate(man)
        except M.ManifestError as e:
            return _err(f"Invalid manifest: {e}", 400)
        d = M.manifest_digest(man)
        r = db["photos"].get(M.photo_id(d))
        if r is None or r.get("manifestDigest") != d:
            return _err("No anchored photo has this manifest digest.", 404)
        r["original"].update(phash=man["original"]["phash"], width=man["original"]["width"],
                             height=man["original"]["height"])
        if man.get("cloaked") and r.get("cloaked"):
            r["cloaked"]["phash"] = man["cloaked"]["phash"]
        r.update(title=man["title"], cloak=man.get("cloak"), regions=man.get("regions"), source="rehydrate")
        _reindex()
        _save_db()
        return jsonify({"success": True, "photoId": r["photoId"],
                        "message": "pHash restored from your manifest; near matching works again."}), 200


# ── ownership proof (design §9.2) ────────────────────────────────────────────
@photo_bp.route("/api/photo/challenge", methods=["GET"])
def challenge():
    c = secrets.token_hex(16)
    now = time()
    for k in [k for k, t in _challenges.items() if now - t > CHALLENGE_TTL]:
        _challenges.pop(k, None)
    _challenges[c] = now
    return jsonify({"success": True, "challenge": c, "expiresIn": int(CHALLENGE_TTL)}), 200


@photo_bp.route("/api/photo/prove", methods=["POST"])
def prove():
    """Prove control of the key that registered a photo. With a login, also links the key."""
    body = request.get_json(silent=True) or {}
    pid, c, sig = body.get("photoId"), body.get("challenge"), body.get("sig")
    t = _challenges.pop(c, None) if isinstance(c, str) else None
    if t is None or time() - t > CHALLENGE_TTL:
        return _err("Unknown or expired challenge. Request a new one.", 400)
    with _lock:
        r = _state["db"]["photos"].get(pid)
        if r is None:
            return _err("Photo not found.", 404)
        if not S.verify(r["ownerPk"], sig, S.prove_message(pid, c)):
            return jsonify({"success": True, "proven": False,
                            "message": "The signature does not match the registering key."}), 200
        user = _requester()
        if user:
            _link(r["ownerPk"], user)
            _save_db()
    return jsonify({"success": True, "proven": True, "photoId": pid,
                    "ownerPkFp": S.pk_fingerprint(r["ownerPk"]), "linkedTo": user}), 200


# ── reads ────────────────────────────────────────────────────────────────────
@photo_bp.route("/api/photo/stats", methods=["GET"])
def photo_stats():
    db = _state["db"]
    s = _state.get("sync")
    cal = _state["calibration"]
    return jsonify({"success": True, "photos": len(db["photos"]), "albums": len(db["albums"]),
                    "owners": len({r["ownerPk"] for r in db["photos"].values()}),
                    "ledgerSync": _sync_status(),
                    "ledgerSyncDetail": dict(s.status) if s else None,
                    "anchorBackend": _cfg("PHOTOSHIELD_ANCHOR", "algorand"),
                    "nearMatch": "calibrated" if cal else "uncalibrated",
                    "calibration": cal,
                    "serverCloak": _cfg("PHOTOSHIELD_SERVER_CLOAK", "0") == "1"}), 200


@photo_bp.route("/api/photo/owner/<key>", methods=["GET"])
def photos_by_owner(key):
    user = _requester()
    db = _state["db"]
    if _HEX64.match(key):
        rows = [r for r in db["photos"].values() if r["ownerPk"] == key]
    elif user and key == user:
        pks = {pk for pk, users in db["pk_users"].items() if user in users}
        rows = [r for r in db["photos"].values() if r.get("ownerUser") == user or r["ownerPk"] in pks]
    else:
        return _err("Look up by public key, or log in to list your own photos.", 403)
    rows.sort(key=lambda r: r.get("registeredAt") or 0, reverse=True)
    return jsonify({"success": True, "count": len(rows), "photos": [_public(r, user) for r in rows]}), 200


@photo_bp.route("/api/photo/proof/<photo_id>", methods=["GET"])
def photo_proof(photo_id):
    r = _state["db"]["photos"].get(photo_id)
    if r is None:
        return _err("Photo not found.", 404)
    out = {"success": True, "photoId": photo_id, "manifestDigest": r["manifestDigest"],
           "ownerPk": r["ownerPk"], "signature": r["signature"],
           "algoTxId": r.get("algoTxId"), "algoRound": r.get("algoRound"),
           "indexerUrl": _indexer_url(r.get("algoTxId")), "mode": "single"}
    if r.get("albumId"):
        a = _state["db"]["albums"].get(r["albumId"], {})
        out.update(mode="album", albumId=r["albumId"], index=r.get("albumIndex"),
                   proof=r.get("albumProof"), root=a.get("root"), leafCount=a.get("count"))
    return jsonify(out), 200


@photo_bp.route("/api/photo/<photo_id>", methods=["GET"])
def get_photo(photo_id):
    if not _HEX16.match(photo_id):
        return _err("Photo not found.", 404)
    r = _state["db"]["photos"].get(photo_id)
    if r is None:
        return _err("Photo not found.", 404)
    return jsonify({"success": True, "photo": _public(r, _requester())}), 200


# ── certificate (design §6.5) ────────────────────────────────────────────────
@photo_bp.route("/api/photo/certificate/<photo_id>", methods=["GET"])
def photo_certificate(photo_id):
    r = _state["db"]["photos"].get(photo_id)
    if r is None:
        return _err("Photo not found.", 404)
    try:
        import io
        from datetime import datetime, timezone

        from flask import Response
        from reportlab.graphics import renderPDF
        from reportlab.graphics.barcode import qr
        from reportlab.graphics.shapes import Drawing
        from reportlab.lib.colors import HexColor, white
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        return _err("Certificates need reportlab on the server.", 503)

    BG, PANEL, GREEN = HexColor("#06080c"), HexColor("#10151f"), HexColor("#00e5a0")
    TEXT, MUTED, BLUE = HexColor("#eef2f8"), HexColor("#8a94a6"), HexColor("#5b8cff")

    def safe(s):
        return str(s if s is not None else "-").encode("latin-1", "replace").decode("latin-1")

    W, H = letter
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter, pageCompression=1)
    c.setTitle("PhotoShield certificate")
    c.setFillColor(BG); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(TEXT); c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(W / 2, H - 90, "PHOTO PROVENANCE CERTIFICATE")
    c.setFillColor(GREEN); c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(W / 2, H - 112, "PhotoShield · BlockVerify · Algorand Testnet")
    c.setFillColor(MUTED); c.setFont("Helvetica", 9)
    c.drawCentredString(W / 2, H - 130, "Proves that these fingerprints existed at the time of the anchoring transaction,")
    c.drawCentredString(W / 2, H - 142, "signed by the key below. It does not prove authorship, and a perceptual match is evidence, not proof.")
    c.setFillColor(PANEL); c.roundRect(50, 170, W - 100, H - 340, 12, fill=1, stroke=0)
    y = H - 190

    def field(label, value, mono=False, color=TEXT):
        nonlocal y
        c.setFillColor(MUTED); c.setFont("Helvetica-Bold", 7.5); c.drawString(70, y, label)
        c.setFillColor(color); c.setFont("Courier" if mono else "Helvetica", 9)
        v = safe(value)
        for k in range(0, len(v), 64):
            y -= 12
            c.drawString(70, y, v[k:k + 64])
        y -= 16

    ts = r.get("registeredAt")
    when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts else "-"
    field("TITLE", r.get("title") or "(not stored on chain; re-attach the manifest)")
    field("PHOTO ID", r["photoId"], mono=True)
    field("ORIGINAL SHA-256", r["original"]["sha256"], mono=True, color=GREEN)
    if r.get("cloaked"):
        field("CLOAKED COPY SHA-256", r["cloaked"]["sha256"], mono=True, color=GREEN)
    field("MANIFEST DIGEST", r["manifestDigest"], mono=True)
    field("OWNER PUBLIC KEY (Ed25519)", r["ownerPk"], mono=True)
    field("ANCHORED", f"{when} · round {r.get('algoRound') or '-'} · privacy {r.get('privacy')}")
    field("ALGORAND TRANSACTION", r.get("algoTxId") or "-", mono=True, color=BLUE)
    if r.get("cloak"):
        ck = r["cloak"]
        field("CLOAK", f"{ck.get('model')} · eps {ck.get('eps')}/255 · {ck.get('steps')} steps · "
                       f"PSNR {ck.get('psnr', '-')} dB")
    if r.get("regions"):
        rg = r["regions"]
        field("REGION MAP (origin reports)", f"{rg.get('grid')}x{rg.get('grid')} tiles · root {rg.get('root')}",
              mono=True)
    link = (f"https://lora.algokit.io/testnet/transaction/{r['algoTxId']}"
            if _indexer_url(r.get("algoTxId")) else "https://lora.algokit.io/testnet")
    w = qr.QrCodeWidget(link)
    b = w.getBounds()
    d = Drawing(110, 110, transform=[110 / (b[2] - b[0]), 0, 0, 110 / (b[3] - b[1]), 0, 0])
    d.add(w)
    c.setFillColor(white); c.roundRect(W - 190, 190, 126, 126, 8, fill=1, stroke=0)
    renderPDF.draw(d, c, W - 182, 198)
    c.setFillColor(MUTED); c.setFont("Helvetica", 7)
    c.drawCentredString(W - 127, 180, "Scan to open the transaction")
    c.drawCentredString(W / 2, 120, "Re-verify any time: hash the file in the Photo Shield tab, then check the note on the public indexer.")
    c.showPage(); c.save()
    return Response(buf.getvalue(), mimetype="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=photoshield_{photo_id}.pdf"})


# ── optional server cloak (design §6.6) ──────────────────────────────────────
@photo_bp.route("/api/photo/cloak", methods=["POST"])
def server_cloak():
    if _cfg("PHOTOSHIELD_SERVER_CLOAK", "0") != "1":
        return _err("Server-side cloaking is off. Run `python -m photoshield protect` locally: "
                    "your photo never has to leave your machine.", 501)
    f = request.files.get("photo")
    if f is None:
        return _err("Upload the photo as multipart field 'photo'.", 400)
    data = f.read(12 * 1024 * 1024 + 1)
    if len(data) > 12 * 1024 * 1024:
        return _err("Photo too large (12 MB max).", 400)
    import base64
    import shutil
    import tempfile
    try:
        from photoshield.cloak.protect import protect
        from photoshield.config import ShieldConfig
        from photoshield.provenance.hashing import decode_rgb
    except ImportError:
        return _err("This server has no cloak engine installed (torch).", 503)
    try:
        rgb = decode_rgb(data)
    except Exception:
        return _err("Could not decode the image.", 400)
    if rgb.shape[0] * rgb.shape[1] > 2_000_000:
        return _err("Photo larger than 2 megapixels; cloak it locally instead.", 400)
    tmp = tempfile.mkdtemp(prefix="ps-")
    try:
        src = os.path.join(tmp, "in.img")
        with open(src, "wb") as out:
            out.write(data)
        cfg = ShieldConfig(model=_cfg("PHOTOSHIELD_SERVER_MODEL", "sdvae"), size=256, steps=40,
                           threads=int(_cfg("PHOTOSHIELD_SERVER_THREADS", "1")))
        res = protect(src, cfg, os.path.join(tmp, "out"), title=(request.form.get("title") or "photo")[:120])
        png = open(res["cloaked_path"], "rb").read()
        return jsonify({"success": True, "manifest": res["manifest"],
                        "cloakedPng": base64.b64encode(png).decode()}), 200
    except Exception as e:
        return _err(f"Cloaking failed: {type(e).__name__}", 500)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)        # never kept, never logged
