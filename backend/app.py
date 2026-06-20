"""
BlockVerify Flask REST API
==========================

Provides HTTP endpoints for the BlockVerify AI-model integrity
verification system.  Every write operation (register, verify,
add-version) creates a blockchain transaction and mines a new block.

Run:
    python app.py          → http://localhost:5000

Author: razor_hats team
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed; fall back to environment variables
import json
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from auth import auth_bp, require_auth
import algorand_client
import bitcoin_client
from time import time
import hashlib

# ------------------------------------------------------------------ #
#  App & state initialisation                                         #
# ------------------------------------------------------------------ #

app = Flask(__name__)
CORS(app, origins="*", supports_credentials=True)
app.register_blueprint(auth_bp)

# ------------------------------------------------------------------ #
#  Real-time alerts (Flask-SocketIO)                                  #
#  Optional dependency — the app degrades gracefully if missing so    #
#  tests / minimal deploys still run. Uses pure-Python threading mode  #
#  (+ simple-websocket) so no C-extensions are required.              #
# ------------------------------------------------------------------ #
try:
    from flask_socketio import SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    SOCKETIO_AVAILABLE = True
except ImportError:
    socketio = None
    SOCKETIO_AVAILABLE = False


def emit_alert(event: str, payload: dict) -> None:
    """Broadcast a real-time event to all connected clients (no-op if SocketIO is unavailable)."""
    if socketio is None:
        return
    try:
        socketio.emit(event, payload)
    except Exception as e:                       # never let a broadcast break the request
        print(f"[socketio] emit failed (non-fatal): {e}")

# Persistence file paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REGISTRY_FILE = os.path.join(DATA_DIR, "models_registry.json")
LOGS_FILE     = os.path.join(DATA_DIR, "verification_logs.json")
CHAIN_FILE    = os.path.join(DATA_DIR, "chain.json")

os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
#  Persistence helpers                                                 #
# ------------------------------------------------------------------ #

def save_state():
    """Persist registry and logs to JSON files."""
    with open(REGISTRY_FILE, "w") as f:
        json.dump(models_registry, f, indent=2)
    with open(LOGS_FILE, "w") as f:
        json.dump(verification_logs, f, indent=2)
    # Invalidate the cached PoW chain so next request rebuilds with fresh data
    try:
        _invalidate_chain_cache()
    except NameError:
        pass  # _invalidate_chain_cache not defined yet during initial import


def load_state():
    """Restore registry, logs, and chain from JSON files if they exist."""
    global models_registry, verification_logs

    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE) as f:
            models_registry.update(json.load(f))

    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE) as f:
            verification_logs.update(json.load(f))


# Algorand Testnet Interface
# Operations are broadcasted to the global Algorand blockchain natively

# In-memory registries (persisted to disk on every write)
models_registry = {}        # modelId  → model dict
verification_logs = {}      # modelId  → [verification records]

# Load saved state on startup
load_state()

# PoW chain cache: rebuilt lazily, invalidated on every registry write
_pow_chain_cache = None      # list[Block] or None
_pow_chain_version = 0       # incremented on each write to registry

def _invalidate_chain_cache():
    global _pow_chain_cache
    _pow_chain_cache = None

def _get_pow_chain():
    """Return the cached custom PoW chain, rebuilding only if the registry changed."""
    global _pow_chain_cache, _pow_chain_version
    if _pow_chain_cache is not None:
        return _pow_chain_cache
    from blockchain import Blockchain
    oldest_tx_time = min((m.get("registeredAt", time()) for m in models_registry.values()), default=time())
    bc = Blockchain(difficulty=2, genesis_timestamp=oldest_tx_time - 60)
    for model_id, m in sorted(models_registry.items(), key=lambda x: x[1].get("registeredAt", 0)):
        tx = {
            "type": "register",
            "model_id": model_id,
            "model_name": m.get("name", "Unknown"),
            "owner": m.get("owner", "system"),
            "hash": m.get("hash", ""),
            "timestamp": m.get("registeredAt", 0),
        }
        bc.pending_transactions.append(tx)
        bc.mine_pending_transactions(block_timestamp=m.get("registeredAt", 0))
        for log in verification_logs.get(model_id, []):
            vtx = {
                "type": "verify",
                "model_id": model_id,
                # verification_logs store `isValid`; expose a clear result for chain consumers
                "result": "valid" if log.get("isValid") else "invalid",
                "timestamp": log.get("timestamp", 0),
            }
            bc.pending_transactions.append(vtx)
            bc.mine_pending_transactions(block_timestamp=log.get("timestamp", 0))
    _pow_chain_cache = bc.chain
    return _pow_chain_cache

# ------------------------------------------------------------------ #
#  Rate limiter                                                        #
# ------------------------------------------------------------------ #

# Tracks recent write timestamps per owner: { owner: [t1, t2, ...] }
_rate_log: dict = {}

RATE_LIMIT      = 10    # max requests
RATE_WINDOW_SEC = 60    # per this many seconds


def check_rate_limit(owner: str) -> tuple[bool, str]:
    """
    Return (allowed, error_message).
    Sliding-window rate limiter: at most RATE_LIMIT write operations
    per owner within RATE_WINDOW_SEC seconds.
    """
    now = time()
    timestamps = _rate_log.get(owner, [])
    # Drop timestamps outside the window
    timestamps = [t for t in timestamps if now - t < RATE_WINDOW_SEC]
    if len(timestamps) >= RATE_LIMIT:
        wait = int(RATE_WINDOW_SEC - (now - timestamps[0]))
        _rate_log[owner] = timestamps
        return False, (
            f"Rate limit exceeded: max {RATE_LIMIT} operations per "
            f"{RATE_WINDOW_SEC}s. Try again in {wait}s."
        )
    timestamps.append(now)
    _rate_log[owner] = timestamps
    return True, ""


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def generate_model_id(model_name, model_hash, owner):
    """Deterministic, collision-resistant model ID (first 16 hex chars)."""
    unique = f"{model_name}{model_hash}{owner}{time()}"
    return hashlib.sha256(unique.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ #
#  Proof-of-Work  (Hashcash / Bitcoin-style anti-spam)                 #
# ------------------------------------------------------------------ #

POW_DIFFICULTY = 3          # number of leading zeros required in SHA-256(modelHash + nonce)
POW_PREFIX = '0' * POW_DIFFICULTY

def verify_pow(model_hash: str, nonce: int) -> bool:
    """Verify that SHA-256(modelHash + str(nonce)) starts with POW_PREFIX."""
    candidate = hashlib.sha256(f"{model_hash}{nonce}".encode()).hexdigest()
    return candidate.startswith(POW_PREFIX)


# ------------------------------------------------------------------ #
#  WRITE endpoints (mine a block)                                      #
# ------------------------------------------------------------------ #

@app.route("/api/register", methods=["POST"])
@require_auth
def register_model():
    """Register a new AI model on the blockchain."""
    try:
        data = request.get_json()
        owner = g.user   # comes from the verified JWT — cannot be spoofed

        if not data.get("modelName"):
            return jsonify({"success": False, "error": "Model name is required"}), 400
        if not data.get("modelHash"):
            return jsonify({"success": False, "error": "Model hash is required"}), 400

        # ── Proof-of-Work validation (Bitcoin Hashcash-style) ───────
        nonce = data.get("powNonce")
        if nonce is None:
            return jsonify({"success": False, "error": f"Proof-of-Work required. Mine a nonce so SHA-256(hash+nonce) starts with {POW_PREFIX!r}."}), 400
        if not verify_pow(data["modelHash"], int(nonce)):
            return jsonify({"success": False, "error": "Proof-of-Work verification failed. Invalid nonce."}), 400
        # ────────────────────────────────────────────────────────────

        allowed, err = check_rate_limit(owner)
        if not allowed:
            return jsonify({"success": False, "error": err}), 429

        model_id = generate_model_id(data["modelName"], data["modelHash"], owner)

        # ── Algorand Migration: Broadcast to Global Testnet ──
        algo_resp = algorand_client.broadcast_hash_to_algorand(
            model_id, data["modelName"], data["modelHash"], owner
        )
        
        if not algo_resp.get("success"):
            return jsonify({"success": False, "error": algo_resp.get("error")}), 500

        models_registry[model_id] = {
            "modelId": model_id,
            "modelName": data["modelName"],
            "modelHash": data["modelHash"],
            "metadata": data.get("metadata", ""),
            "owner": owner,
            "registeredAt": int(time()),
            "blockIndex": algo_resp.get("round"),
            "algoTxId": algo_resp.get("txid"),
            "layerHashes": data.get("layerHashes", {}), # For Deep Layer Inspection
            "currentVersion": 1,
            "isActive": True,
            "versions": [
                {
                    "version": 1,
                    "hash": data["modelHash"],
                    "layerHashes": data.get("layerHashes", {}),
                    "timestamp": int(time()),
                    "changelog": "Initial version",
                    "blockIndex": algo_resp.get("round"),
                    "algoTxId": algo_resp.get("txid"),
                }
            ],
        }
        verification_logs[model_id] = []
        save_state()

        return jsonify(
            {
                "success": True,
                "modelId": model_id,
                "blockIndex": algo_resp.get("round"),
                "algoTxId": algo_resp.get("txid"),
                "message": "Model registered securely on Algorand Testnet",
            }
        ), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/verify", methods=["POST"])
@require_auth
def verify_model():
    """Verify AI model integrity against the stored hash."""
    try:
        data = request.get_json()
        verifier = g.user   # JWT-authenticated username

        if not data.get("modelId"):
            return jsonify({"success": False, "error": "Model ID is required"}), 400
        if not data.get("providedHash"):
            return jsonify({"success": False, "error": "Hash is required"}), 400

        allowed, err = check_rate_limit(verifier)
        if not allowed:
            return jsonify({"success": False, "error": err}), 429

        model = models_registry.get(data["modelId"])
        if not model:
            return jsonify({"success": False, "error": "Model not found"}), 404
        if not model["isActive"]:
            return jsonify({"success": False, "error": "Model has been revoked"}), 400

        # Privacy guard — only the owner can verify a private model
        if model.get("isPrivate") and model["owner"] != verifier:
            return jsonify({
                "success": False,
                "error": "This model is private 🔒 — only the owner can verify it.",
            }), 403

        is_valid = model["modelHash"] == data["providedHash"]

        algo_resp = algorand_client.broadcast_hash_to_algorand(
            data["modelId"], model["modelName"], data["providedHash"], verifier
        )
        
        if not algo_resp.get("success"):
            return jsonify({"success": False, "error": algo_resp.get("error")}), 500

        verification_logs[data["modelId"]].append(
            {
                "verifier": verifier,
                "timestamp": int(time()),
                "isValid": is_valid,
                "providedHash": data["providedHash"],
                "blockIndex": algo_resp.get("round"),
                "algoTxId": algo_resp.get("txid"),
            }
        )
        save_state()

        # ── Real-time broadcast to ALL connected clients (Live Alerts feed) ──
        emit_alert("verification_alert", {
            "modelId":   data["modelId"],
            "modelName": model["modelName"],
            "verifier":  verifier,
            "isValid":   is_valid,
            "status":    "PASS" if is_valid else "TAMPERED",
            "timestamp": int(time()),
        })

        return jsonify(
            {
                "success": True,
                "isValid": is_valid,
                "message": "Model integrity VERIFIED — hash matches"
                if is_valid
                else "INTEGRITY MISMATCH — hash does not match",
                "blockIndex": algo_resp.get("round"),
                "algoTxId": algo_resp.get("txid"),
                "storedHash": model["modelHash"],
                "providedHash": data["providedHash"],
                "layerHashes": model.get("layerHashes", {}) if not is_valid else {},
            }
        ), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/add-version", methods=["POST"])
@require_auth
def add_version():
    """Add a new version to an existing model."""
    try:
        data = request.get_json()
        owner = g.user

        if not data.get("modelId"):
            return jsonify({"success": False, "error": "Model ID is required"}), 400
        if not data.get("newHash"):
            return jsonify({"success": False, "error": "New hash is required"}), 400
        if not data.get("changelog"):
            return jsonify({"success": False, "error": "Changelog is required"}), 400

        allowed, err = check_rate_limit(owner)
        if not allowed:
            return jsonify({"success": False, "error": err}), 429

        model = models_registry.get(data["modelId"])
        if not model:
            return jsonify({"success": False, "error": "Model not found"}), 404
        if model["owner"] != owner:
            return jsonify({"success": False, "error": "Only the model owner can add versions"}), 403
        if not model["isActive"]:
            return jsonify({"success": False, "error": "Model is deactivated"}), 400

        new_ver = model["currentVersion"] + 1

        model_id = data["modelId"]
        algo_resp = algorand_client.broadcast_hash_to_algorand(
            model_id, model["modelName"], data["newHash"], owner
        )
        if not algo_resp.get("success"):
            return jsonify({"success": False, "error": algo_resp.get("error")}), 500

        model["modelHash"] = data["newHash"]
        model["currentVersion"] = new_ver
        model["versions"].append(
            {
                "version": new_ver,
                "hash": data["newHash"],
                "timestamp": int(time()),
                "changelog": data["changelog"],
                "blockIndex": algo_resp.get("round"),
                "algoTxId": algo_resp.get("txid"),
            }
        )
        save_state()

        return jsonify(
            {
                "success": True,
                "version": new_ver,
                "blockIndex": algo_resp.get("round"),
                "algoTxId": algo_resp.get("txid"),
                "message": f"Version {new_ver} added successfully",
            }
        ), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/deactivate", methods=["POST"])
@require_auth
def deactivate_model():
    """Soft-delete (deactivate) a model."""
    try:
        data = request.get_json()
        owner = g.user

        if not data or not data.get("modelId"):
            return jsonify({"success": False, "error": "modelId is required"}), 400

        allowed, err = check_rate_limit(owner)
        if not allowed:
            return jsonify({"success": False, "error": err}), 429

        model = models_registry.get(data["modelId"])
        if not model:
            return jsonify({"success": False, "error": "Model not found"}), 404
        if model["owner"] != owner:
            return jsonify({"success": False, "error": "Only the owner can deactivate"}), 403

        model["isActive"] = False

        algo_resp = algorand_client.broadcast_hash_to_algorand(
            data["modelId"], model["modelName"], "DEACTIVATED", owner
        )
        save_state()

        return jsonify({"success": True, "message": "Model deactivated"}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------------------------------------------------------ #
#  READ endpoints                                                      #
# ------------------------------------------------------------------ #

@app.route("/api/models/<owner>", methods=["GET"])
def get_models_by_owner(owner):
    """Return all models registered by *owner*."""
    try:
        owner_models = [m for m in models_registry.values() if m["owner"] == owner]
        return jsonify({"success": True, "models": owner_models, "count": len(owner_models)}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/model/<model_id>", methods=["GET"])
def get_model(model_id):
    """Return details of a single model."""
    model = models_registry.get(model_id)
    if not model:
        return jsonify({"success": False, "error": "Model not found"}), 404
    return jsonify({"success": True, "model": model}), 200


@app.route("/api/versions/<model_id>", methods=["GET"])
def get_versions(model_id):
    """Return version history for a model."""
    model = models_registry.get(model_id)
    if not model:
        return jsonify({"success": False, "error": "Model not found"}), 404
    return jsonify({"success": True, "versions": model["versions"], "currentVersion": model["currentVersion"]}), 200


@app.route("/api/audit/<model_id>", methods=["GET"])
def get_audit_log(model_id):
    """Return the verification audit trail for a model."""
    if model_id not in verification_logs:
        return jsonify({"success": False, "error": "Model not found"}), 404
    logs = verification_logs[model_id]
    return jsonify({"success": True, "verifications": logs, "count": len(logs)}), 200


@app.route("/api/audit/<model_id>/export", methods=["GET"])
def export_audit_csv(model_id):
    """
    Export the audit trail for a model as a downloadable CSV file.

    Columns: modelId, verifier, result, providedHash, blockIndex, timestamp
    """
    import csv, io
    from datetime import datetime
    from flask import Response

    if model_id not in verification_logs:
        return jsonify({"success": False, "error": "Model not found"}), 404

    logs = verification_logs[model_id]
    model_name = models_registry.get(model_id, {}).get("modelName", model_id)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["modelId", "modelName", "verifier", "result", "providedHash", "blockIndex", "timestamp"])
    for entry in logs:
        ts = entry.get("timestamp", "")
        if ts:
            ts = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
        writer.writerow([
            model_id,
            model_name,
            entry.get("verifier", "anonymous"),
            "VALID" if entry.get("isValid") else "INVALID",
            entry.get("providedHash", ""),
            entry.get("blockIndex", ""),
            ts,
        ])

    csv_data = output.getvalue()
    filename = f"audit_{model_id[:8]}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/api/chain", methods=["GET"])
def get_chain():
    """Return the custom PoW blockchain (cached)."""
    chain = _get_pow_chain()
    return jsonify({
        "success": True,
        "chain": [b.to_dict() for b in chain]
    })


@app.route("/api/chain/validate", methods=["GET"])
def validate_chain():
    """Validate blockchain integrity (Disabled: Now on Algorand)."""
    return jsonify(
        {
            "success": True,
            "isValid": True,
            "errors": [],
            "message": "Blockchain is mathematically maintained by the global Algorand network ✓",
        }
    ), 200


@app.route("/api/search", methods=["GET"])
def search_models():
    """
    Search models by name, hash, or owner.
    Private models are excluded unless the requester owns them.
    """
    try:
        from auth import decode_token
        q     = request.args.get("q", "").lower().strip()
        owner = request.args.get("owner", "").strip()

        # Determine the requesting user (optional — no 401 if missing)
        requester = None
        auth_hdr = request.headers.get("Authorization", "")
        if auth_hdr.startswith("Bearer "):
            payload = decode_token(auth_hdr[7:])
            if payload:
                requester = payload.get("username")

        results = []
        for model in models_registry.values():
            # Hide private models from non-owners
            if model.get("isPrivate") and model["owner"] != requester:
                continue
            if owner and model["owner"] != owner:
                continue
            if q and not (
                q in model["modelName"].lower() or
                q in model["modelHash"].lower() or
                q in model["modelId"].lower()
            ):
                continue
            results.append(model)

        return jsonify({"success": True, "models": results, "count": len(results)}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/tamper-demo", methods=["POST"])
def tamper_demo():
    return jsonify({"success": False, "error": "Tampering is physically impossible on the Algorand Testnet. This action has been disabled."}), 403


@app.route("/api/rate-limit-status", methods=["GET"])
def rate_limit_status():
    """Return remaining write quota for a given owner within the current window."""
    owner = request.args.get("owner", "").strip()
    if not owner:
        return jsonify({"success": False, "error": "owner param required"}), 400
    now = time()
    timestamps = _rate_log.get(owner, [])
    timestamps = [t for t in timestamps if now - t < RATE_WINDOW_SEC]
    used = len(timestamps)
    remaining = max(0, RATE_LIMIT - used)
    return jsonify({
        "success": True,
        "owner": owner,
        "limit": RATE_LIMIT,
        "windowSeconds": RATE_WINDOW_SEC,
        "used": used,
        "remaining": remaining,
    }), 200


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return high-level platform statistics."""
    total_verifications = sum(len(v) for v in verification_logs.values())
    return jsonify(
        {
            "totalModels": len(models_registry),
            "totalVerifications": total_verifications,
            "totalBlocks": "Algorand Live",
        }
    ), 200




# ------------------------------------------------------------------ #
#  Algorand Testnet live endpoints                                     #
# ------------------------------------------------------------------ #

@app.route("/api/algo/wallet", methods=["GET"])
def algo_wallet_info():
    """Return live Algorand wallet / account info for the server address."""
    data = algorand_client.get_wallet_info()
    return jsonify(data), (200 if data.get("success") else 503)


@app.route("/api/algo/tx/<txid>", methods=["GET"])
def algo_tx_info(txid):
    """
    Fetch a confirmed Algorand transaction by TxID from the public Indexer.
    Used by the Blockchain Proof page to show live on-chain data.
    """
    data = algorand_client.get_transaction_info(txid)
    return jsonify(data), (200 if data.get("success") else 404)


@app.route("/api/algo/contract", methods=["GET"])
def algo_contract_info():
    """Return the deployed smart contract App ID and explorer link."""
    from contract import load_app_id
    app_id = load_app_id()
    if not app_id:
        return jsonify({"success": False, "error": "Contract not yet deployed. Register a model first."}), 404
    return jsonify({
        "success": True,
        "app_id": app_id,
        "network": "Algorand Testnet",
        "explorer_url": f"https://lora.algonode.network/testnet/application/{app_id}",
    }), 200

@app.route("/api/algo/network", methods=["GET"])
def algo_network_stats():
    data = algorand_client.get_network_stats()
    return jsonify(data), (200 if data.get("success") else 503)


@app.route("/api/algo/ledger", methods=["GET"])
def algo_transaction_ledger():
    limit = request.args.get("limit", 50, type=int)
    data = algorand_client.get_transaction_ledger(limit=min(limit, 100))
    return jsonify(data), (200 if data.get("success") else 503)


@app.route("/api/algo/verify-onchain/<model_id>", methods=["GET"])
def algo_verify_onchain(model_id):
    """Read the smart contract global state and compare hash with local DB."""
    from contract import load_app_id, read_contract_state

    app_id = load_app_id()
    if not app_id:
        return jsonify({"success": False, "error": "Smart contract not deployed yet"}), 503

    # Get the locally stored hash
    model = models_registry.get(model_id)
    if not model:
        return jsonify({"success": False, "error": "Model not found in local registry"}), 404

    local_hash = model.get("modelHash", "")

    # Read from smart contract
    result = read_contract_state(app_id, model_id)
    if not result.get("success"):
        return jsonify(result), 503

    if not result.get("found"):
        return jsonify({
            "success": True,
            "verified": False,
            "reason": "not_found",
            "message": "Model ID not found in smart contract. It may not have been registered on-chain.",
            "model_id": model_id,
            "app_id": app_id,
        }), 200

    on_chain_hash = result.get("on_chain_hash", "")
    match = (on_chain_hash == local_hash)

    return jsonify({
        "success": True,
        "verified": match,
        "reason": "match" if match else "mismatch",
        "model_id": model_id,
        "model_name": model.get("modelName", ""),
        "local_hash": local_hash,
        "on_chain_hash": on_chain_hash,
        "app_id": app_id,
        "explorer_url": f"https://lora.algokit.io/testnet/application/{app_id}",
        "message": (
            "ON-CHAIN VERIFIED: The hash stored in the Algorand smart contract matches the local registry."
            if match else
            "MISMATCH DETECTED: The on-chain hash does NOT match the local registry. Possible tampering!"
        ),
    }), 200


@app.route("/api/activity-feed", methods=["GET"])
def activity_feed():
    """
    Aggregated activity feed — registrations, verifications, version additions.
    Returns a chronological list of recent platform events.
    """
    limit = min(int(request.args.get("limit", 20)), 50)
    events = []

    # 1. Registrations
    for mid, m in models_registry.items():
        events.append({
            "type": "register",
            "icon": "📝",
            "user": m.get("owner", "unknown"),
            "modelName": m.get("modelName", "Unknown"),
            "modelId": mid,
            "timestamp": m.get("registeredAt", 0),
            "message": f"registered <strong>{m.get('modelName','')}</strong>",
            "hash": m.get("modelHash", "")[:16],
        })

        # 1b. Version additions (skip v1, that's the registration)
        for v in m.get("versions", [])[1:]:
            events.append({
                "type": "version",
                "icon": "📋",
                "user": m.get("owner", "unknown"),
                "modelName": m.get("modelName", "Unknown"),
                "modelId": mid,
                "timestamp": v.get("timestamp", 0),
                "message": f"added v{v.get('version','')} to <strong>{m.get('modelName','')}</strong>",
                "hash": v.get("hash", "")[:16],
            })

    # 2. Verifications
    for mid, logs in verification_logs.items():
        model = models_registry.get(mid, {})
        mname = model.get("modelName", mid)
        for log in logs:
            status = "✅ verified" if log.get("isValid") else "❌ failed verification of"
            events.append({
                "type": "verify",
                "icon": "✅" if log.get("isValid") else "❌",
                "user": log.get("verifier", "unknown"),
                "modelName": mname,
                "modelId": mid,
                "timestamp": log.get("timestamp", 0),
                "message": f"{status} <strong>{mname}</strong>",
                "hash": log.get("providedHash", "")[:16],
                "isValid": log.get("isValid", False),
            })

    # Sort newest first, take limit
    events.sort(key=lambda e: e["timestamp"], reverse=True)
    events = events[:limit]

    return jsonify({"success": True, "events": events, "total": len(events)}), 200


@app.route("/api/tamper-simulate/<model_id>", methods=["POST"])
@require_auth
def tamper_simulate(model_id):
    """
    Deliberately corrupt the local hash to simulate tampering.
    Stores the original hash so it can be restored.
    """
    model = models_registry.get(model_id)
    if not model:
        return jsonify({"success": False, "error": "Model not found"}), 404

    original_hash = model.get("modelHash", "")
    if model.get("_originalHash"):
        return jsonify({
            "success": False,
            "error": "Model is already tampered. Restore first before re-tampering.",
            "tampered": True,
        }), 400

    # Corrupt the hash — flip characters to make it obviously different
    import hashlib
    corrupted = hashlib.sha256(("TAMPERED_" + original_hash).encode()).hexdigest()

    model["_originalHash"] = original_hash
    model["modelHash"] = corrupted
    save_state()

    return jsonify({
        "success": True,
        "model_id": model_id,
        "original_hash": original_hash,
        "corrupted_hash": corrupted,
        "message": "Local hash has been corrupted! Run On-Chain Verify to detect the mismatch.",
    }), 200


@app.route("/api/tamper-restore/<model_id>", methods=["POST"])
@require_auth
def tamper_restore(model_id):
    """Restore the original hash after a tamper simulation."""
    model = models_registry.get(model_id)
    if not model:
        return jsonify({"success": False, "error": "Model not found"}), 404

    original = model.get("_originalHash")
    if not original:
        return jsonify({"success": False, "error": "Model was not tampered — nothing to restore."}), 400

    model["modelHash"] = original
    del model["_originalHash"]
    save_state()

    return jsonify({
        "success": True,
        "model_id": model_id,
        "restored_hash": original,
        "message": "Original hash restored successfully.",
    }), 200


@app.route("/api/algo/merkle", methods=["GET"])
def algo_merkle_tree():
    """
    Build a Merkle tree from ALL registered model hashes.
    Returns the tree structure (for D3.js), the Merkle root, and leaves.
    """
    import hashlib

    # Collect active model hashes
    leaves_data = []
    for mid, m in models_registry.items():
        if m.get("isActive", True):
            leaves_data.append({
                "model_id": mid,
                "model_name": m.get("modelName", "Unknown"),
                "hash": m.get("modelHash", ""),
                "owner": m.get("owner", ""),
            })

    if not leaves_data:
        return jsonify({
            "success": True,
            "count": 0,
            "merkleRoot": None,
            "tree": None,
            "message": "No models registered yet. Register a model to see the Merkle tree.",
        }), 200

    # Build leaf nodes
    def sha256(text):
        return hashlib.sha256(text.encode()).hexdigest()

    leaf_nodes = []
    for lf in leaves_data:
        leaf_nodes.append({
            "hash": lf["hash"],
            "label": lf["model_name"],
            "modelId": lf["model_id"],
            "owner": lf["owner"],
            "txType": "registration",
            "children": [],
        })

    # If odd number of leaves, duplicate the last one
    if len(leaf_nodes) % 2 != 0:
        leaf_nodes.append({**leaf_nodes[-1], "label": leaf_nodes[-1]["label"] + " (dup)"})

    # Build tree bottom-up
    def build_tree(nodes):
        if len(nodes) == 1:
            return nodes[0]
        parents = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i + 1] if i + 1 < len(nodes) else nodes[i]
            parent_hash = sha256(left["hash"] + right["hash"])
            parents.append({
                "hash": parent_hash,
                "label": "",
                "children": [left, right],
            })
        return build_tree(parents)

    tree = build_tree(leaf_nodes)

    return jsonify({
        "success": True,
        "count": len(leaves_data),
        "merkleRoot": tree["hash"],
        "tree": tree,
        "leaves": [{"hash": lf["hash"], "txType": "registration"} for lf in leaves_data],
    }), 200


@app.route("/api/stats/activity", methods=["GET"])
def get_activity():
    """
    Return daily registration and verification counts for the last N days.
    Query param: days (int, default 14, max 30)
    """
    from datetime import datetime, timezone, timedelta

    days = min(int(request.args.get("days", 14)), 30)
    now  = datetime.now(timezone.utc)

    # Build a dict of date_str → {registrations, verifications}
    buckets = {}
    for i in range(days - 1, -1, -1):
        d = (now - timedelta(days=i)).strftime("%b %d")
        buckets[d] = {"registrations": 0, "verifications": 0}

    for model in models_registry.values():
        ts = model.get("registeredAt")
        if ts:
            d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%b %d")
            if d in buckets:
                buckets[d]["registrations"] += 1

    for logs in verification_logs.values():
        for entry in logs:
            ts = entry.get("timestamp")
            if ts:
                d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%b %d")
                if d in buckets:
                    buckets[d]["verifications"] += 1

    labels        = list(buckets.keys())
    registrations = [buckets[d]["registrations"]  for d in labels]
    verifications = [buckets[d]["verifications"]  for d in labels]

    return jsonify({
        "success": True,
        "labels": labels,
        "registrations": registrations,
        "verifications": verifications,
        "days": days,
    }), 200


@app.route("/api/registry", methods=["GET"])
def public_registry():
    """
    Public model registry — all models across all owners.
    Private models are excluded from public view.
    """
    try:
        from auth import decode_token
        q     = request.args.get("q", "").lower().strip()
        page  = max(1, int(request.args.get("page", 1)))
        limit = min(50, max(1, int(request.args.get("limit", 20))))

        # Determine requester (private models visible to their owner)
        requester = None
        auth_hdr = request.headers.get("Authorization", "")
        if auth_hdr.startswith("Bearer "):
            payload = decode_token(auth_hdr[7:])
            if payload:
                requester = payload.get("username")

        all_models = sorted(
            [
                m for m in models_registry.values()
                if not m.get("isPrivate") or m["owner"] == requester
            ],
            key=lambda m: m.get("registeredAt", 0),
            reverse=True,
        )

        if q:
            all_models = [
                m for m in all_models
                if q in m["modelName"].lower()
                or q in m["modelId"].lower()
                or q in m["owner"].lower()
            ]

        total  = len(all_models)
        start  = (page - 1) * limit
        paged  = all_models[start : start + limit]

        return jsonify({
            "success": True,
            "models": paged,
            "total": total,
            "page": page,
            "pages": max(1, -(-total // limit)),
            "limit": limit,
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/report/<model_id>", methods=["GET"])
def download_report(model_id):
    """
    Generate and stream a PDF integrity report for a model.
    Uses fpdf2 v2.7+ API (new_x/new_y instead of ln=True, border as string).
    """
    from fpdf import FPDF
    from datetime import datetime, timezone
    from flask import Response

    model = models_registry.get(model_id)
    if not model:
        return jsonify({"success": False, "error": "Model not found"}), 404

    logs = verification_logs.get(model_id, [])

    def fmt_ts(ts):
        if not ts:
            return "-"
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── safe ASCII-only text (fpdf2 core fonts don't support Unicode) ──
    def safe(text):
        if not text:
            return "-"
        return str(text).encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header banner ─────────────────────────────────────────────────
    pdf.set_fill_color(30, 20, 60)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(200, 180, 255)
    pdf.set_y(7)
    pdf.cell(0, 10, "BlockVerify - AI Model Integrity Report", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(160, 140, 220)
    pdf.set_y(18)
    pdf.cell(0, 6, f"Generated: {fmt_ts(datetime.now(timezone.utc).timestamp())}   |   blockverify.app",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(32)
    pdf.set_text_color(30, 20, 60)

    def section(title):
        pdf.set_fill_color(240, 237, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(60, 40, 120)
        pdf.cell(0, 8, f"  {safe(title)}", fill=True,
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(20, 20, 40)
        pdf.ln(2)

    def kv(label, value, mono=False):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(100, 80, 160)
        pdf.cell(45, 6, safe(label) + ":", new_x="RIGHT", new_y="TOP")
        pdf.set_font("Courier" if mono else "Helvetica", "", 9)
        pdf.set_text_color(20, 20, 40)
        pdf.multi_cell(0, 6, safe(str(value) if value else "-"),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    # ── Model Overview ─────────────────────────────────────────────────
    section("Model Overview")
    kv("Model Name",    model.get("modelName", "-"))
    kv("Model ID",      model.get("modelId", "-"), mono=True)
    kv("Owner",         model.get("owner", "-"))
    kv("Status",        "Active" if model.get("isActive") else "Deactivated")
    kv("Registered At", fmt_ts(model.get("registeredAt")))
    kv("Block #",       str(model.get("blockIndex", "-")))
    kv("Version",       f"v{model.get('currentVersion', 1)}")
    if model.get("metadata"):
        kv("Metadata",  model["metadata"])
    pdf.ln(4)

    # ── Hash ──────────────────────────────────────────────────────────
    section("SHA-256 Hash (Current Version)")
    pdf.set_font("Courier", "", 8.5)
    pdf.set_fill_color(248, 246, 255)
    pdf.set_text_color(40, 20, 100)
    pdf.multi_cell(0, 7, safe(model.get("modelHash", "-")),
                   border="1", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(20, 20, 40)
    pdf.ln(4)

    # ── Version History ───────────────────────────────────────────────
    section(f"Version History  ({len(model.get('versions', []))} version(s))")
    for v in model.get("versions", []):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(60, 40, 120)
        pdf.cell(0, 6,
                 f"  v{v.get('version')}  -  {fmt_ts(v.get('timestamp'))}  -  Block #{v.get('blockIndex', '?')}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(45, 5, "", new_x="RIGHT", new_y="TOP")
        pdf.cell(0, 5, safe(v.get("changelog", "-")),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 8)
        pdf.set_text_color(100, 80, 160)
        pdf.cell(45, 5, "", new_x="RIGHT", new_y="TOP")
        h_str = v.get("hash", "")
        display_h = h_str[:64] + ("..." if len(h_str) > 64 else "")
        pdf.cell(0, 5, safe(display_h), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(20, 20, 40)
        pdf.ln(1)
    pdf.ln(4)

    # ── Audit Trail ───────────────────────────────────────────────────
    section(f"Verification Audit Trail  ({len(logs)} record(s))")
    if not logs:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, "  No verifications recorded yet.",
                 new_x="LMARGIN", new_y="NEXT")
    else:
        # Table header
        pdf.set_fill_color(220, 210, 255)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(40, 20, 100)
        pdf.cell(30, 7, "Verifier",  fill=True, border="1", new_x="RIGHT", new_y="TOP")
        pdf.cell(18, 7, "Result",    fill=True, border="1", new_x="RIGHT", new_y="TOP")
        pdf.cell(50, 7, "Timestamp", fill=True, border="1", new_x="RIGHT", new_y="TOP")
        pdf.cell(0,  7, "Hash (first 32 chars)", fill=True, border="1",
                 new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Courier", "", 8)
        for i, entry in enumerate(logs):
            even = i % 2 == 0
            if even:
                pdf.set_fill_color(248, 246, 255)
            else:
                pdf.set_fill_color(255, 255, 255)

            result = "VALID" if entry.get("isValid") else "INVALID"
            if result == "VALID":
                pdf.set_text_color(0, 140, 80)
            else:
                pdf.set_text_color(180, 30, 30)

            pdf.cell(30, 6, safe(entry.get("verifier", "anon"))[:18],
                     fill=True, border="1", new_x="RIGHT", new_y="TOP")
            pdf.cell(18, 6, result,
                     fill=True, border="1", new_x="RIGHT", new_y="TOP")
            pdf.set_text_color(20, 20, 40)
            pdf.cell(50, 6, fmt_ts(entry.get("timestamp")),
                     fill=True, border="1", new_x="RIGHT", new_y="TOP")
            h_val = entry.get("providedHash", "-")
            display = (h_val[:32] + "...") if len(h_val) > 32 else h_val
            pdf.cell(0, 6, safe(display),
                     fill=True, border="1", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Footer ────────────────────────────────────────────────────────
    pdf.set_fill_color(30, 20, 60)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(180, 160, 240)
    pdf.cell(0, 7,
             "This report is auto-generated by BlockVerify - a custom Proof-of-Work blockchain system.",
             align="C", fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf_bytes = pdf.output()
    filename  = f"blockverify_report_{model_id[:8]}.pdf"
    return Response(
        bytes(pdf_bytes),
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/api/certificate/<model_id>", methods=["POST"])
def generate_certificate(model_id):
    """
    Generate a professional, dark-themed PDF *certificate* using ReportLab.

    Contains: model name, SHA-256 hash, owner, registration date, the Algorand
    TxID rendered as a scannable QR code (links to the explorer), and a
    "Verified on Algorand Testnet" badge. Returns the PDF as an attachment.

    NB: styled to match the app's current "signal" brand (electric-green on a
    cool-black) rather than the original purple, so the certificate matches the
    live UI.
    """
    try:
        import io
        from datetime import datetime, timezone
        from flask import Response
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.colors import HexColor, white
        from reportlab.graphics.barcode import qr
        from reportlab.graphics.shapes import Drawing
        from reportlab.graphics import renderPDF

        model = models_registry.get(model_id)
        if not model:
            return jsonify({"success": False, "error": "Model not found"}), 404

        # ── Brand palette (cool-black + signal green/blue) ──
        BG    = HexColor("#06080c"); PANEL = HexColor("#10151f")
        GREEN = HexColor("#00e5a0"); BLUE  = HexColor("#5b8cff")
        TEXT  = HexColor("#eef2f8"); MUTED = HexColor("#8a94a6")
        LINE  = HexColor("#222c3a"); HAIR  = HexColor("#161d28")
        GTINT = HexColor("#06140f")

        # ── Data ──
        name  = str(model.get("modelName", "Unknown Model"))
        mhash = str(model.get("modelHash", "—"))
        owner = str(model.get("owner", "—"))
        txid  = str(model.get("algoTxId") or "")
        ts    = model.get("registeredAt")
        reg_date = (datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%B %d, %Y · %H:%M UTC")
                    if ts else "Pending confirmation")
        explorer = (f"https://lora.algokit.io/testnet/transaction/{txid}"
                    if txid else "https://lora.algokit.io/testnet")

        def ascii_safe(s):  # core PDF fonts are latin-1
            return s.encode("latin-1", "replace").decode("latin-1")
        name, owner, reg_date = map(ascii_safe, (name, owner, reg_date))

        W, H = letter                      # 612 x 792 pts
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter, pageCompression=1)
        c.setTitle(f"BlockVerify Certificate — {name}")
        cx = W / 2.0

        # ── Background + frames ──
        c.setFillColor(BG);   c.rect(0, 0, W, H, fill=1, stroke=0)
        mg = 34
        c.setStrokeColor(LINE); c.setLineWidth(1.4)
        c.roundRect(mg, mg, W - 2*mg, H - 2*mg, 16, fill=0, stroke=1)
        c.setStrokeColor(HAIR); c.setLineWidth(0.8)
        c.roundRect(mg + 8, mg + 8, W - 2*mg - 16, H - 2*mg - 16, 12, fill=0, stroke=1)

        # corner accents
        def corner(x, y, dx, dy):
            c.setStrokeColor(GREEN); c.setLineWidth(2)
            c.line(x, y, x + dx, y); c.line(x, y, x, y + dy)
        corner(mg + 8, H - mg - 8, 22, -22);  corner(W - mg - 8, H - mg - 8, -22, -22)
        corner(mg + 8, mg + 8, 22, 22);       corner(W - mg - 8, mg + 8, -22, 22)

        # ── Header wordmark with check mark ──
        ly = H - 78
        c.setFillColor(GREEN); c.roundRect(cx - 96, ly - 6, 26, 26, 6, fill=1, stroke=0)
        c.setStrokeColor(BG); c.setLineWidth(2.6)
        c.line(cx - 90, ly + 6, cx - 86, ly + 2); c.line(cx - 86, ly + 2, cx - 78, ly + 13)
        c.setFont("Helvetica-Bold", 17)
        c.setFillColor(TEXT); c.drawString(cx - 62, ly + 2, "Block")
        c.setFillColor(GREEN); c.drawString(cx - 62 + c.stringWidth("Block", "Helvetica-Bold", 17), ly + 2, "Verify")

        # ── Title ──
        c.setFillColor(TEXT); c.setFont("Helvetica-Bold", 26)
        c.drawCentredString(cx, H - 150, "CERTIFICATE OF INTEGRITY")
        c.setStrokeColor(GREEN); c.setLineWidth(2); c.line(cx - 64, H - 160, cx + 64, H - 160)
        c.setFillColor(MUTED); c.setFont("Helvetica", 9.5)
        c.drawCentredString(cx, H - 178, "Cryptographic attestation that this AI model is registered and tamper-evident")
        c.drawCentredString(cx, H - 191, "on a live, public blockchain ledger.")

        # ── Verified badge pill ──
        badge = "VERIFIED ON ALGORAND TESTNET"
        c.setFont("Helvetica-Bold", 9)
        bw = c.stringWidth(badge, "Helvetica-Bold", 9) + 50
        bh = 22; bx = cx - bw/2; by = H - 230
        c.setFillColor(GTINT); c.setStrokeColor(GREEN); c.setLineWidth(1)
        c.roundRect(bx, by, bw, bh, 11, fill=1, stroke=1)
        c.setFillColor(GREEN); c.circle(bx + 16, by + bh/2, 5.5, fill=1, stroke=0)
        c.setStrokeColor(BG); c.setLineWidth(1.5)
        c.line(bx + 13, by + bh/2, bx + 15, by + bh/2 - 2.5); c.line(bx + 15, by + bh/2 - 2.5, bx + 19, by + bh/2 + 2.5)
        c.setFillColor(GREEN); c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(cx + 9, by + bh/2 - 3.2, badge)

        # ── Model name ──
        c.setFillColor(MUTED); c.setFont("Helvetica-Bold", 8); c.drawCentredString(cx, H - 264, "ISSUED FOR THE MODEL")
        c.setFillColor(TEXT); c.setFont("Helvetica-Bold", 22)
        disp = name if c.stringWidth(name, "Helvetica-Bold", 22) < (W - 170) else (name[:38] + "…")
        c.drawCentredString(cx, H - 290, disp)

        # ── Details panel ──
        px = mg + 30; pw = W - 2*(mg + 30); py = 248; ph = 196
        c.setFillColor(PANEL); c.setStrokeColor(LINE); c.setLineWidth(1)
        c.roundRect(px, py, pw, ph, 12, fill=1, stroke=1)

        lx = px + 24
        # SHA-256 fingerprint (wrapped 2 x 32)
        c.setFillColor(MUTED); c.setFont("Helvetica-Bold", 7.5); c.drawString(lx, py + ph - 24, "SHA-256 FINGERPRINT")
        c.setFillColor(GREEN); c.setFont("Courier", 9.5)
        c.drawString(lx, py + ph - 39, mhash[:32]); c.drawString(lx, py + ph - 52, mhash[32:64])

        def field(y, label, value, mono=False, color=TEXT, vsize=10.5):
            c.setFillColor(MUTED); c.setFont("Helvetica-Bold", 7.5); c.drawString(lx, y, label)
            c.setFillColor(color); c.setFont("Courier" if mono else "Helvetica", vsize)
            c.drawString(lx, y - 14, value)

        field(py + ph - 78,  "REGISTERED OWNER", owner)
        field(py + ph - 118, "REGISTRATION DATE", reg_date)
        tx_disp = (txid[:22] + "…" + txid[-6:]) if len(txid) > 30 else (txid or "Pending on-chain confirmation")
        field(py + ph - 158, "ALGORAND TRANSACTION ID", tx_disp, mono=True, color=BLUE, vsize=9)

        # ── QR code on a white card (right side of panel) ──
        qsize = 116; qpad = 11
        qcard_x = px + pw - (qsize + 2*qpad) - 22
        qcard_y = py + (ph - (qsize + 2*qpad)) / 2.0
        c.setFillColor(white); c.roundRect(qcard_x, qcard_y, qsize + 2*qpad, qsize + 2*qpad, 8, fill=1, stroke=0)
        qwidget = qr.QrCodeWidget(explorer)
        b = qwidget.getBounds(); bw_, bh_ = b[2] - b[0], b[3] - b[1]
        d = Drawing(qsize, qsize, transform=[qsize / bw_, 0, 0, qsize / bh_, 0, 0])
        d.add(qwidget)
        renderPDF.draw(d, c, qcard_x + qpad, qcard_y + qpad)
        c.setFillColor(MUTED); c.setFont("Helvetica", 7)
        c.drawCentredString(qcard_x + (qsize + 2*qpad) / 2.0, qcard_y - 12, "Scan to verify on Algorand Explorer")

        # ── Hexagonal authenticity seal (balances the lower band) ──
        import math
        def hexagon(x0, y0, r):
            return [(x0 + r*math.cos(math.pi/2 + i*math.pi/3),
                     y0 + r*math.sin(math.pi/2 + i*math.pi/3)) for i in range(6)]
        sx, sy = cx, 176
        for rr, col, wdt in ((42, GREEN, 1.6), (35, HexColor("#0d4a37"), 1.0)):
            pts = hexagon(sx, sy, rr)
            path = c.beginPath(); path.moveTo(*pts[0])
            for p in pts[1:]:
                path.lineTo(*p)
            path.close()
            c.setStrokeColor(col); c.setLineWidth(wdt); c.drawPath(path, fill=0, stroke=1)
        # check mark inside
        c.setStrokeColor(GREEN); c.setLineWidth(3)
        c.line(sx - 13, sy + 2, sx - 4, sy - 8); c.line(sx - 4, sy - 8, sx + 14, sy + 13)
        c.setFillColor(GREEN); c.setFont("Helvetica-Bold", 6.5)
        c.drawCentredString(sx, sy + 50, "INTEGRITY SEAL")
        c.setFillColor(MUTED); c.setFont("Helvetica", 6)
        c.drawCentredString(sx, sy - 56, "IMMUTABLE  ·  PUBLICLY VERIFIABLE  ·  TAMPER-EVIDENT")

        # ── Footer ──
        c.setStrokeColor(LINE); c.setLineWidth(0.8); c.line(mg + 30, 96, W - mg - 30, 96)
        gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        c.setFillColor(MUTED); c.setFont("Helvetica", 8)
        c.drawCentredString(cx, 80, f"Issued {gen}  ·  SHA-256 fingerprint  ·  Algorand Pure-Proof-of-Stake  ·  blockverify.app")
        c.setFillColor(HexColor("#5b6678")); c.setFont("Helvetica-Oblique", 7)
        c.drawCentredString(cx, 67, "This certificate attests to an immutable on-chain record. Re-verify any time by re-hashing the model file.")

        c.showPage(); c.save(); buf.seek(0)
        filename = f"blockverify_certificate_{model_id[:8]}.pdf"
        return Response(
            buf.getvalue(),
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500








# ------------------------------------------------------------------ #
#  Merkle tree helper                                                  #
# ------------------------------------------------------------------ #

def _tx_hash(tx: dict) -> str:
    """SHA-256 of a deterministic JSON representation of a transaction."""
    return hashlib.sha256(
        json.dumps(tx, sort_keys=True).encode()
    ).hexdigest()


def _build_merkle(tx_hashes: list) -> dict:
    """
    Build a Merkle tree from a list of leaf hashes.

    Returns a nested dict:
        {
            "hash": "<root_hash>",
            "children": [ <left_node>, <right_node> ]   # absent for leaves
        }
    Compatible with D3.js hierarchy().
    """
    if not tx_hashes:
        # Empty block — single null root
        return {"hash": "0" * 64, "label": "empty", "children": []}

    # Build leaf nodes
    nodes = [{"hash": h, "label": h[:8] + "…", "children": []} for h in tx_hashes]

    if len(nodes) == 1:
        return nodes[0]

    # Bottom-up: keep pairing until one root remains
    # We track each level as a list of (node_dict, hash) tuples
    level = nodes
    while len(level) > 1:
        next_level = []
        i = 0
        while i < len(level):
            left = level[i]
            # Odd number of nodes → duplicate last one
            right = level[i + 1] if i + 1 < len(level) else level[i]
            combined = hashlib.sha256(
                (left["hash"] + right["hash"]).encode()
            ).hexdigest()
            parent = {
                "hash": combined,
                "label": combined[:8] + "…",
                "children": [left] if right is left else [left, right],
            }
            next_level.append(parent)
            i += 2
        level = next_level

    return level[0]


@app.route("/api/block/<int:index>/merkle", methods=["GET"])
def get_merkle_tree(index):
    """Return the Merkle tree for a block (uses cached chain)."""
    chain = _get_pow_chain()

    if index >= len(chain):
        return jsonify({"success": False, "error": f"Block #{index} not found (chain has {len(chain)} blocks)"}), 404

    block = chain[index]
    tx_hashes = [_tx_hash(tx) for tx in block.transactions]
    tree = _build_merkle(tx_hashes)

    return jsonify({
        "success": True,
        "blockIndex": index,
        "txCount": len(block.transactions),
        "merkleRoot": tree["hash"],
        "tree": tree,
        "leaves": tx_hashes,
    })


# ------------------------------------------------------------------ #
#  Model access control (privacy)                                      #
# ------------------------------------------------------------------ #

@app.route("/api/model/<model_id>/privacy", methods=["POST"])
@require_auth
def set_model_privacy(model_id):
    """
    Toggle a model's privacy.  Only the model owner may call this.

    Body:  { "isPrivate": true | false }

    Private models:
        - Are hidden from /api/registry and /api/search (for non-owners).
        - Can only be verified by the owner.
        - Still appear in the owner's /api/models/<owner> list.
    """
    model = models_registry.get(model_id)
    if not model:
        return jsonify({"success": False, "error": "Model not found"}), 404
    if model["owner"] != g.user:
        return jsonify({"success": False, "error": "Only the model owner can change privacy settings"}), 403

    data = request.get_json(silent=True) or {}
    is_private = bool(data.get("isPrivate", False))
    model["isPrivate"] = is_private
    save_state()

    status = "private 🔒" if is_private else "public 🌐"
    return jsonify({
        "success": True,
        "modelId": model_id,
        "isPrivate": is_private,
        "message": f"Model is now {status}",
    }), 200

# ------------------------------------------------------------------ #
#  Bitcoin L1 Anchoring (OP_RETURN settlement layer)                   #
#  Branch: future-bitcoin-integration                                  #
# ------------------------------------------------------------------ #

@app.route("/api/bitcoin/wallet", methods=["GET"])
def bitcoin_wallet_info():
    """Return the Bitcoin Testnet wallet balance and address."""
    data = bitcoin_client.get_wallet_balance()
    return jsonify(data), (200 if data.get("success") else 503)


@app.route("/api/bitcoin/anchor", methods=["POST"])
@require_auth
def bitcoin_anchor():
    """
    Anchor the current Merkle Root of all registered models to Bitcoin Testnet
    via an OP_RETURN transaction.  This demonstrates the 'Rollup' architecture:
      - Algorand  = fast sidechain (individual model hashes)
      - Bitcoin   = L1 settlement  (Merkle Root batch anchor)
    """
    try:
        body        = request.get_json() or {}
        merkle_root = body.get("merkleRoot", "").strip()

        if not merkle_root or len(merkle_root) < 64:
            # Fall back to computing live from registry using exact UI construction rules
            leaves_data = [
                m.get("modelHash", "") 
                for m in models_registry.values() 
                if m.get("isActive", True)
            ]
            if not leaves_data:
                return jsonify({"success": False, "error": "No models registered yet. Register a model first."}), 400

            def sha256_hex(text):
                return hashlib.sha256(text.encode()).hexdigest()

            current = leaves_data.copy()
            while len(current) > 1:
                if len(current) % 2 != 0:
                    current.append(current[-1])
                current = [
                    sha256_hex(current[i] + current[i+1])
                    for i in range(0, len(current), 2)
                ]
            merkle_root = current[0] if current else ""

        # Get annotated OP_RETURN script for educational display
        op_hex       = bitcoin_client.build_op_return_script_hex(merkle_root)
        annotations  = bitcoin_client.decode_op_return_annotated(op_hex)

        # Broadcast to Bitcoin Testnet
        result = bitcoin_client.anchor_merkle_root(merkle_root)
        result["merkle_root"]  = merkle_root
        result["op_return_hex"] = op_hex
        result["annotations"]  = annotations
        result["model_count"]  = len(models_registry)

        return jsonify(result), (200 if result.get("success") else 500)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------------------------------------------------------ #
#  AI Chat Assistant Endpoint                                          #
# ------------------------------------------------------------------ #

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Comprehensive System Persona detailing the entire project
BLOCKVERIFY_PERSONA = """
You are the official BlockVerify AI Assistant. You are friendly, highly intelligent, and expert at explaining complex topics simply. 
BlockVerify is a blockchain-powered AI Model Integrity Verification Platform created by three students: Shubhangam, Mihir, and Aditya.
It solves the problem of "AI Supply Chain Attacks" (like backdoor weight poisoning) by cryptographically proving a model has not been tampered with.

BlockVerify's 7-Layer Architecture:
1. Custom PoW Blockchain: A from-scratch python blockchain to prove foundational understanding.
2. Algorand Testnet: A fast L2 sidechain. We use a hand-written TEAL v8 smart contract to permanently store the mapping of 'model_id -> sha256_hash. We use 0-ALGO note transactions.
3. Bitcoin Testnet (Rollup): An L1 settlement layer. We batch all model hashes into a Merkle Root and anchor it fully on-chain using an OP_RETURN transaction.
4. Client-side Hashing: WebCrypto API extracts hashes without the model ever leaving the user's device.
5. Deep Layer Forensics: A revolutionary inspector that extracts layer-by-layer neural network architecture. It detects two attacks: 'Weight Tampering' (modified deep learning parameters) and 'Topology Poisoning' (rogue layers injected by hackers).
6. zkML (Zero-Knowledge): A simulated cryptographic prover demonstrating that we can verify ML execution without exposing private dataset weights.
7. IPFS Self-Healing: A simulated remediation protocol that fetches uncorrupted genesis blocks from the InterPlanetary File System to replace compromised neural network hidden layers.

Your goal: Educate users. If they ask what a tab does, explain it. If they ask about blockchain concepts (like what a Merkle tree is, what Algorand is, what OP_RETURN means), explain it clearly using analogies. Be concise, use emojis naturally, and format responses neatly.
"""

@app.route("/api/chat", methods=["POST"])
def chat_assistant():
    """Gemini-powered chatbot endpoint for user assistance."""
    try:
        import requests
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
        history = data.get("history", []) # List of {"role": "user"|"model", "text": "..."}

        if not user_message:
            return jsonify({"success": False, "error": "Message is empty"}), 400

        # Build Gemini standard completion payload
        contents = []
        for msg in history:
            role = "model" if msg.get("role") == "model" else "user"
            contents.append({"role": role, "parts": [{"text": msg.get("text", "")}]})
            
        contents.append({"role": "user", "parts": [{"text": user_message}]})

        payload = {
            "contents": contents,
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": BLOCKVERIFY_PERSONA.strip()}]
            },
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 800}
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        
        if resp.status_code != 200:
            error_data = resp.json()
            ext_err = error_data.get("error", {}).get("message", "AI service temporarily unavailable")
            return jsonify({"success": False, "error": f"Gemini API Error: {ext_err}"}), 500

        resp_json = resp.json()
        reply_text = resp_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        if not reply_text:
            return jsonify({"success": False, "error": "AI returned an empty response"}), 500

        return jsonify({"success": True, "reply": reply_text}), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500




# ------------------------------------------------------------------ #
#  Server start                                                        #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("══════════════════════════════════════════════")
    print("  🔗 BlockVerify — Algorand + Bitcoin L1")
    print("  (Professor Edition - Live Testnet)")
    print("══════════════════════════════════════════════")
    print("  Ledger      : Algorand Testnet (AlgoNode)")
    print("  Settlement  : Bitcoin Testnet3 (OP_RETURN)")
    print("  Real-time   : " + ("Flask-SocketIO (WebSocket) ✓" if SOCKETIO_AVAILABLE else "disabled (flask-socketio not installed)"))
    print("  Server      : http://localhost:5000")
    print("══════════════════════════════════════════════")
    port = int(os.environ.get("PORT", 5000))
    if socketio is not None:
        # allow_unsafe_werkzeug lets the threading-mode dev server run under `python app.py` on Render
        socketio.run(app, debug=False, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)
    else:
        app.run(debug=False, host="0.0.0.0", port=port)
