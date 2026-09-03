"""FedVerify federated-round routes (Phase 3).

Additive Blueprint: registered alongside the existing ``auth_bp`` in app.py. No existing
route, model or behaviour is touched.

The server is NOT trusted here. ``POST /api/fl/round/commit`` recomputes the Merkle root
from the submitted leaves and rejects the commit if it disagrees with the claimed root or
leaf count — so a caller cannot register a lineage that does not fold to its own root.
Clients then re-verify independently: ``GET /api/fl/proof/...`` returns the sibling path,
and the anchored root read straight off the ledger is what the fold must equal.

Error envelope matches docs/API_REFERENCE.md: {"success": false, "error": "..."}
with 200 / 400 / 404 / 500.
"""
from __future__ import annotations

import json
import os
import re
import sys
from time import time

from flask import Blueprint, jsonify, request

fl_bp = Blueprint("fl", __name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FL_FILE = os.path.join(DATA_DIR, "fl_runs.json")
os.makedirs(DATA_DIR, exist_ok=True)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _commitment():
    """Lazy import of the shared commitment module.

    Deferred on purpose: fedverify.chain.commitment imports backend/app.py for the
    canonical Merkle, and app.py imports this blueprint — importing eagerly would be
    circular. By request time app.py is fully loaded and sys.modules serves it.
    """
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from fedverify.chain import commitment
    return commitment


# ── storage ──────────────────────────────────────────────────────────────────
def _load() -> dict:
    if not os.path.exists(FL_FILE):
        return {}
    try:
        with open(FL_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(runs: dict) -> None:
    tmp = FL_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(runs, f, indent=2)
    os.replace(tmp, FL_FILE)          # atomic; a concurrent GET never sees a half file


def _err(msg, code):
    return jsonify({"success": False, "error": msg}), code


MAX_ROUND = 1_000_000
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._=-]{1,200}$")


def _client_list(v):
    """Coerce a client-id list, or None. Rejects anything that is not a flat list.

    The UI calls .map(String) on these; a string or object here would throw in the
    browser rather than degrade, so it is rejected at the door.
    """
    if v is None:
        return None
    if not isinstance(v, list):
        raise ValueError("must be a list of client ids")
    if len(v) > 10_000:
        raise ValueError("too many client ids")
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (str, int)):
            raise ValueError("client ids must be strings or integers")
        out.append(str(x))
    return out


def _metrics(v):
    """Coerce a {name: number} metrics block, or None."""
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError("metrics must be an object")
    out = {}
    for k, x in list(v.items())[:32]:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            continue                       # drop non-numeric silently; never store junk
        if x != x or x in (float("inf"), float("-inf")):
            continue                       # NaN/Inf are not valid JSON for a client
        out[str(k)[:40]] = float(x)
    return out or None


# Public Algorand indexer — the browser reads the anchored root from HERE, not from us.
# That is what makes the verification trustless: our response is untrusted input.
INDEXER = "https://testnet-idx.algonode.cloud/v2/transactions"


def _indexer_url(txid):
    """Only real Algorand txids are 52-char base32; mock/local txids are hex block hashes."""
    if not txid or len(str(txid)) != 52:
        return None
    return f"{INDEXER}/{txid}"


# ── POST /api/fl/round/commit ────────────────────────────────────────────────
@fl_bp.route("/api/fl/round/commit", methods=["POST"])
def commit_round_route():
    """Register one federated round's commitment after re-verifying it server-side."""
    data = request.get_json(silent=True) or {}
    run_id = str(data.get("run_id") or "").strip()
    if not run_id:
        return _err("run_id is required.", 400)
    # A run id becomes part of a URL path. Anything with a slash would create a run that
    # can never be read back, and anything with quotes or markup would reach the browser.
    if not RUN_ID_RE.match(run_id):
        return _err("run_id may only contain letters, digits, and . _ - = "
                    "(max 200 characters).", 400)

    try:
        rnd = int(data.get("round"))
    except (TypeError, ValueError):
        return _err("round must be an integer.", 400)
    if rnd < 1:
        return _err("round must be >= 1.", 400)
    if rnd > MAX_ROUND:
        return _err(f"round must be <= {MAX_ROUND}.", 400)

    leaves = data.get("leaves")
    if not isinstance(leaves, list) or not leaves:
        return _err("leaves must be a non-empty list.", 400)

    root = data.get("root")
    if not isinstance(root, str) or len(root) != 64:
        return _err("root must be a 64-character hex string.", 400)

    try:
        entries = [(str(lf["client_id"]), str(lf["digest"])) for lf in leaves]
    except (TypeError, KeyError):
        return _err("each leaf needs client_id and digest.", 400)

    C = _commitment()
    try:
        recomputed, count = C.round_merkle_root(entries)
    except ValueError as e:
        return _err(str(e), 400)

    if recomputed != root:
        return _err("Root mismatch: the submitted leaves do not fold to the claimed root.", 400)

    claimed = data.get("leaf_count")
    if claimed is not None and int(claimed) != count:
        return _err(f"leaf_count mismatch: claimed {claimed}, leaves give {count}.", 400)

    try:
        attackers = _client_list(data.get("attackers"))
        accepted = _client_list(data.get("accepted"))
        rejected = _client_list(data.get("rejected"))
        metrics = _metrics(data.get("metrics"))
    except ValueError as e:
        return _err(str(e), 400)

    leaf_hashes = C.leaf_hashes(entries)
    stored = [{"index": i, "client_id": cid, "digest": dig, "leaf": leaf_hashes[i],
               "proof": C.inclusion_proof(entries, i)}
              for i, (cid, dig) in enumerate(entries)]

    runs = _load()
    run = runs.setdefault(run_id, {"run_id": run_id, "created_at": int(time()), "rounds": {}})
    if isinstance(data.get("meta"), dict):
        run["meta"] = {**(run.get("meta") or {}), **data["meta"]}   # dataset, K, eps, attack…
    run["rounds"][str(rnd)] = {
        "round": rnd, "root": root, "leaf_count": count,
        "txid": data.get("txid"), "backend": data.get("backend"),
        "accepted": accepted if accepted is not None else [c for c, _ in entries],
        "rejected": rejected or [],
        # Ground truth, when the run was a controlled experiment. It lets the UI separate a
        # correct catch from a false positive instead of painting every rejection the same
        # colour. Absent for real deployments, where nobody knows who the attackers are.
        "attackers": attackers,
        "metrics": metrics,                  # optional {test_acc, macro_f1, asr, ...}
        "committed_at": int(time()), "leaves": stored,
    }
    run["updated_at"] = int(time())
    _save(runs)

    return jsonify({"success": True, "run_id": run_id, "round": rnd,
                    "root": root, "leaf_count": count,
                    "message": f"Round {rnd} committed with {count} client updates."}), 201


# ── GET /api/fl/runs ─────────────────────────────────────────────────────────
@fl_bp.route("/api/fl/runs", methods=["GET"])
def list_runs():
    """Every committed run, newest first — the Federated tab's run selector."""
    runs = _load()
    out = []
    for rid, run in runs.items():
        rounds = run.get("rounds", {})
        nums = sorted(int(r) for r in rounds)
        out.append({
            "runId": rid,
            "numRounds": len(nums),
            "firstRound": nums[0] if nums else None,
            "lastRound": nums[-1] if nums else None,
            "createdAt": run.get("created_at"),
            "updatedAt": run.get("updated_at"),
            "meta": run.get("meta") or {},
            "backend": (rounds.get(str(nums[-1])) or {}).get("backend") if nums else None,
        })
    out.sort(key=lambda r: (r.get("updatedAt") or 0), reverse=True)
    return jsonify({"success": True, "count": len(out), "runs": out}), 200


# ── GET /api/fl/run/<run_id> ─────────────────────────────────────────────────
@fl_bp.route("/api/fl/run/<run_id>", methods=["GET"])
def get_run(run_id):
    """Run metadata plus every round root, in round order."""
    run = _load().get(run_id)
    if not run:
        return _err(f"Run '{run_id}' not found.", 404)
    rounds = sorted(run["rounds"].values(), key=lambda r: r["round"])
    return jsonify({
        "success": True, "run_id": run_id,
        "createdAt": run.get("created_at"), "updatedAt": run.get("updated_at"),
        "meta": run.get("meta") or {},
        "numRounds": len(rounds),
        "roundRoots": [{"round": r["round"], "root": r["root"],
                        "leafCount": r["leaf_count"], "txid": r.get("txid"),
                        "indexerUrl": _indexer_url(r.get("txid")),
                        "metrics": r.get("metrics")}
                       for r in rounds],
    }), 200


# ── GET /api/fl/round/<run_id>/<r> ───────────────────────────────────────────
@fl_bp.route("/api/fl/round/<run_id>/<int:rnd>", methods=["GET"])
def get_round(run_id, rnd):
    run = _load().get(run_id)
    if not run:
        return _err(f"Run '{run_id}' not found.", 404)
    r = run["rounds"].get(str(rnd))
    if not r:
        return _err(f"Round {rnd} not found for run '{run_id}'.", 404)
    return jsonify({
        "success": True, "run_id": run_id, "round": rnd,
        "root": r["root"], "leafCount": r["leaf_count"],
        "txid": r.get("txid"), "backend": r.get("backend"),
        "indexerUrl": _indexer_url(r.get("txid")),
        "clientIds": [lf["client_id"] for lf in r["leaves"]],
        "accepted": r.get("accepted", []), "rejected": r.get("rejected", []),
    }), 200


# ── GET /api/fl/proof/<run_id>/<r>/<client_id> ───────────────────────────────
@fl_bp.route("/api/fl/proof/<run_id>/<int:rnd>/<client_id>", methods=["GET"])
def get_proof(run_id, rnd, client_id):
    """Everything a client needs to verify its update was committed, without trusting us."""
    run = _load().get(run_id)
    if not run:
        return _err(f"Run '{run_id}' not found.", 404)
    r = run["rounds"].get(str(rnd))
    if not r:
        return _err(f"Round {rnd} not found for run '{run_id}'.", 404)

    lf = next((x for x in r["leaves"] if x["client_id"] == str(client_id)), None)
    if lf is None:
        return _err(f"Client '{client_id}' did not contribute to round {rnd}.", 404)

    return jsonify({
        "success": True, "run_id": run_id, "round": rnd,
        "clientId": lf["client_id"], "index": lf["index"],
        "digest": lf["digest"], "leaf": lf["leaf"], "proof": lf["proof"],
        "root": r["root"], "leafCount": r["leaf_count"], "txid": r.get("txid"),
        "indexerUrl": _indexer_url(r.get("txid")),
    }), 200


# ── GET /api/fl/lineage/<run_id> ─────────────────────────────────────────────
@fl_bp.route("/api/fl/lineage/<run_id>", methods=["GET"])
def get_lineage(run_id):
    """Ordered round roots with the accepted/rejected client split for each round."""
    run = _load().get(run_id)
    if not run:
        return _err(f"Run '{run_id}' not found.", 404)
    rounds = sorted(run["rounds"].values(), key=lambda r: r["round"])
    return jsonify({
        "success": True, "run_id": run_id, "numRounds": len(rounds),
        "meta": run.get("meta") or {},
        "lineage": [{
            "round": r["round"], "root": r["root"], "leafCount": r["leaf_count"],
            "txid": r.get("txid"), "backend": r.get("backend"),
            "indexerUrl": _indexer_url(r.get("txid")),
            "accepted": r.get("accepted", []), "rejected": r.get("rejected", []),
            "attackers": r.get("attackers"), "metrics": r.get("metrics"),
        } for r in rounds],
    }), 200
