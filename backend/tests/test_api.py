"""
Tests for the Flask REST API endpoints.

Run:
    pytest backend/tests/test_api.py -v
"""

import pytest
import sys
import os
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app, models_registry, verification_logs, _rate_log
import app as app_module
import algorand_client

# Reset all state before each test so tests don't interfere with each other
@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Reset the registries, rate-limit log, and mock Algorand."""
    models_registry.clear()
    verification_logs.clear()
    _rate_log.clear()
    # Drop any cached PoW chain so chain/Merkle tests start from a clean registry
    app_module._invalidate_chain_cache()

    # Prevent tests from writing to the real JSON database
    monkeypatch.setattr(app_module, "save_state", lambda: None)

    # Mock Algorand Testnet API
    global mock_algo_round
    mock_algo_round = 1000

    def mock_broadcast(model_id, name, hash_val, owner):
        global mock_algo_round
        mock_algo_round += 1
        return {
            "success": True,
            "txid": f"MOCK_TXID_{mock_algo_round}",
            "round": mock_algo_round
        }
    
    monkeypatch.setattr(algorand_client, "broadcast_hash_to_algorand", mock_broadcast)
    yield


# Flask test client fixture
from auth import _make_token

class AuthTestClient:
    def __init__(self, test_client):
        self.test_client = test_client
        self.default_user = "alice"

    def _auth_kwargs(self, kwargs):
        headers = kwargs.get("headers", {})
        if "Authorization" not in headers:
            if headers.get("No-Auth") is True:
                del headers["No-Auth"]
            else:
                user = self.default_user
                if kwargs.get("json") and isinstance(kwargs["json"], dict):
                    if "owner" in kwargs["json"]:
                        user = kwargs["json"]["owner"]
                    elif "verifier" in kwargs["json"]:
                        user = kwargs["json"]["verifier"]
                headers["Authorization"] = f"Bearer {_make_token(user)}"
        else:
            if headers.get("No-Auth") is True:
                del headers["No-Auth"]
        kwargs["headers"] = headers
        return kwargs

    def post(self, *args, **kwargs):
        return self.test_client.post(*args, **self._auth_kwargs(kwargs))

    def get(self, *args, **kwargs):
        return self.test_client.get(*args, **self._auth_kwargs(kwargs))

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield AuthTestClient(c)


# Helper – mine a valid PoW nonce for a given hash at difficulty k
def mine_pow(hash_val: str, difficulty: int = 3) -> int:
    prefix = '0' * difficulty
    nonce = 0
    while True:
        candidate = hashlib.sha256(f"{hash_val}{nonce}".encode()).hexdigest()
        if candidate.startswith(prefix):
            return nonce
        nonce += 1


# Helper – return the first nonce that does NOT satisfy PoW (guaranteed-invalid)
def invalid_pow(hash_val: str, difficulty: int = 3) -> int:
    prefix = '0' * difficulty
    nonce = 0
    while hashlib.sha256(f"{hash_val}{nonce}".encode()).hexdigest().startswith(prefix):
        nonce += 1
    return nonce


# Helper – register a model and return the parsed JSON
def _register(client, name="TestModel", hash_val="abc123", owner="alice", metadata=""):
    nonce = mine_pow(hash_val)
    return client.post(
        "/api/register",
        json={"modelName": name, "modelHash": hash_val, "metadata": metadata, "owner": owner, "powNonce": nonce},
    )


# ═══════════════════════════════════════════════════════════════════
#  /api/register
# ═══════════════════════════════════════════════════════════════════


class TestRegister:

    def test_register_success(self, client):
        resp = _register(client)
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["success"] is True
        assert "modelId" in data
        assert "blockIndex" in data

    def test_register_missing_name(self, client):
        resp = client.post("/api/register", json={"modelHash": "abc", "owner": "alice"})
        assert resp.status_code == 400
        assert resp.get_json()["success"] is False

    def test_register_missing_hash(self, client):
        resp = client.post("/api/register", json={"modelName": "M", "owner": "alice"})
        assert resp.status_code == 400

    def test_register_missing_owner(self, client):
        # Without auth header, it should be intercepted by JWT middleware as 401
        resp = client.post("/api/register", json={"modelName": "M", "modelHash": "h"}, headers={"No-Auth": True})
        assert resp.status_code == 401

    def test_register_missing_pow(self, client):
        # No nonce submitted → should be rejected with 400
        resp = client.post("/api/register", json={"modelName": "M", "modelHash": "abc123"})
        assert resp.status_code == 400
        assert "Proof-of-Work" in resp.get_json()["error"]

    def test_register_creates_block(self, client):
        resp = _register(client)
        assert resp.get_json()["algoTxId"] is not None

    def test_register_stores_in_registry(self, client):
        data = _register(client).get_json()
        assert data["modelId"] in models_registry

    def test_register_initial_version_is_one(self, client):
        data = _register(client).get_json()
        model = models_registry[data["modelId"]]
        assert model["currentVersion"] == 1
        assert len(model["versions"]) == 1


# ═══════════════════════════════════════════════════════════════════
#  /api/verify
# ═══════════════════════════════════════════════════════════════════


class TestVerify:

    def test_verify_valid(self, client):
        # Register first, then verify with the exact same hash — should pass
        model_id = _register(client, hash_val="correct").get_json()["modelId"]
        resp = client.post(
            "/api/verify",
            json={"modelId": model_id, "providedHash": "correct", "verifier": "bob"},
        )
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["isValid"] is True

    def test_verify_invalid(self, client):
        # Provide a wrong hash — integrity check should fail
        model_id = _register(client, hash_val="correct").get_json()["modelId"]
        resp = client.post(
            "/api/verify",
            json={"modelId": model_id, "providedHash": "WRONG", "verifier": "bob"},
        )
        data = resp.get_json()
        assert data["isValid"] is False

    def test_verify_nonexistent_model(self, client):
        resp = client.post(
            "/api/verify",
            json={"modelId": "ghost", "providedHash": "h", "verifier": "bob"},
        )
        assert resp.status_code == 404

    def test_verify_missing_model_id(self, client):
        resp = client.post("/api/verify", json={"providedHash": "h"})
        assert resp.status_code == 400

    def test_verify_missing_hash(self, client):
        resp = client.post("/api/verify", json={"modelId": "x"})
        assert resp.status_code == 400

    def test_verify_logged_in_audit(self, client):
        # Verification result should be saved to the audit log
        model_id = _register(client).get_json()["modelId"]
        client.post(
            "/api/verify",
            json={"modelId": model_id, "providedHash": "abc123", "verifier": "carol"},
        )
        assert len(verification_logs[model_id]) == 1
        assert verification_logs[model_id][0]["verifier"] == "carol"


# ═══════════════════════════════════════════════════════════════════
#  /api/add-version
# ═══════════════════════════════════════════════════════════════════


class TestAddVersion:

    def test_add_version_success(self, client):
        model_id = _register(client, owner="alice").get_json()["modelId"]
        resp = client.post(
            "/api/add-version",
            json={"modelId": model_id, "newHash": "v2hash", "changelog": "fix bug", "owner": "alice"},
        )
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["version"] == 2

    def test_add_version_wrong_owner(self, client):
        # Only the original owner should be allowed to add versions
        model_id = _register(client, owner="alice").get_json()["modelId"]
        resp = client.post(
            "/api/add-version",
            json={"modelId": model_id, "newHash": "v2", "changelog": "c", "owner": "mallory"},
        )
        assert resp.status_code == 403

    def test_add_version_updates_hash(self, client):
        # After adding a version, the model's stored hash should be the new one
        model_id = _register(client, hash_val="old", owner="alice").get_json()["modelId"]
        client.post(
            "/api/add-version",
            json={"modelId": model_id, "newHash": "new", "changelog": "c", "owner": "alice"},
        )
        assert models_registry[model_id]["modelHash"] == "new"

    def test_add_version_not_found(self, client):
        resp = client.post(
            "/api/add-version",
            json={"modelId": "nope", "newHash": "h", "changelog": "c", "owner": "a"},
        )
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════
#  READ endpoints
# ═══════════════════════════════════════════════════════════════════


class TestReadEndpoints:

    def test_get_models_by_owner(self, client):
        _register(client, name="M1", owner="alice")
        _register(client, name="M2", owner="alice")
        _register(client, name="M3", owner="bob")

        resp = client.get("/api/models/alice")
        data = resp.get_json()
        assert data["count"] == 2

    def test_get_models_empty(self, client):
        resp = client.get("/api/models/nobody")
        assert resp.get_json()["count"] == 0

    def test_get_model_detail(self, client):
        model_id = _register(client, name="Detail").get_json()["modelId"]
        resp = client.get(f"/api/model/{model_id}")
        data = resp.get_json()
        assert data["success"] is True
        assert data["model"]["modelName"] == "Detail"

    def test_get_model_not_found(self, client):
        assert client.get("/api/model/ghost").status_code == 404

    def test_get_versions(self, client):
        model_id = _register(client, owner="alice").get_json()["modelId"]
        client.post(
            "/api/add-version",
            json={"modelId": model_id, "newHash": "v2", "changelog": "v2 log", "owner": "alice"},
        )
        resp = client.get(f"/api/versions/{model_id}")
        data = resp.get_json()
        assert data["currentVersion"] == 2
        assert len(data["versions"]) == 2

    def test_get_audit_log(self, client):
        model_id = _register(client).get_json()["modelId"]
        client.post("/api/verify", json={"modelId": model_id, "providedHash": "abc123", "verifier": "v"})
        client.post("/api/verify", json={"modelId": model_id, "providedHash": "wrong", "verifier": "v"})

        resp = client.get(f"/api/audit/{model_id}")
        data = resp.get_json()
        assert data["count"] == 2

    def test_get_audit_not_found(self, client):
        assert client.get("/api/audit/ghost").status_code == 404


# ═══════════════════════════════════════════════════════════════════
#  Chain endpoints
# ═══════════════════════════════════════════════════════════════════


class TestDeactivate:

    def test_deactivate_success(self, client):
        model_id = _register(client, owner="alice").get_json()["modelId"]
        resp = client.post("/api/deactivate", json={"modelId": model_id, "owner": "alice"})
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    def test_deactivate_wrong_owner(self, client):
        # A different user should not be allowed to deactivate someone else's model
        model_id = _register(client, owner="alice").get_json()["modelId"]
        resp = client.post("/api/deactivate", json={"modelId": model_id, "owner": "mallory"})
        assert resp.status_code == 403

    def test_deactivate_not_found(self, client):
        resp = client.post("/api/deactivate", json={"modelId": "ghost", "owner": "alice"})
        assert resp.status_code == 404

    def test_verify_deactivated_model_fails(self, client):
        # Once deactivated, the model should no longer be verifiable
        model_id = _register(client, owner="alice").get_json()["modelId"]
        client.post("/api/deactivate", json={"modelId": model_id, "owner": "alice"})
        resp = client.post("/api/verify", json={"modelId": model_id, "providedHash": "abc123", "verifier": "bob"})
        assert resp.status_code == 400

    def test_add_version_deactivated_fails(self, client):
        # A deactivated model should also block new version uploads
        model_id = _register(client, owner="alice").get_json()["modelId"]
        client.post("/api/deactivate", json={"modelId": model_id, "owner": "alice"})
        resp = client.post("/api/add-version", json={"modelId": model_id, "newHash": "h", "changelog": "c", "owner": "alice"})
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════
#  Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_add_version_missing_changelog(self, client):
        model_id = _register(client, owner="alice").get_json()["modelId"]
        resp = client.post("/api/add-version", json={"modelId": model_id, "newHash": "h", "owner": "alice"})
        assert resp.status_code == 400

    def test_add_version_missing_hash(self, client):
        model_id = _register(client, owner="alice").get_json()["modelId"]
        resp = client.post("/api/add-version", json={"modelId": model_id, "changelog": "c", "owner": "alice"})
        assert resp.status_code == 400

    def test_add_version_missing_owner(self, client):
        model_id = _register(client, owner="alice").get_json()["modelId"]
        resp = client.post("/api/add-version", json={"modelId": model_id, "newHash": "h", "changelog": "c"}, headers={"No-Auth": True})
        assert resp.status_code == 401

    def test_multiple_versions_increment(self, client):
        # Each new version should increment the version counter by exactly 1
        model_id = _register(client, owner="alice").get_json()["modelId"]
        client.post("/api/add-version", json={"modelId": model_id, "newHash": "v2", "changelog": "v2", "owner": "alice"})
        resp = client.post("/api/add-version", json={"modelId": model_id, "newHash": "v3", "changelog": "v3", "owner": "alice"})
        assert resp.get_json()["version"] == 3


# ═══════════════════════════════════════════════════════════════════
#  /api/search
# ═══════════════════════════════════════════════════════════════════


class TestSearch:

    def test_search_no_params_returns_all(self, client):
        _register(client, name="Alpha", owner="alice")
        _register(client, name="Beta", owner="bob")
        resp = client.get("/api/search")
        data = resp.get_json()
        assert data["success"] is True
        assert data["count"] == 2

    def test_search_by_name_substring(self, client):
        _register(client, name="AlphaModel", owner="alice")
        _register(client, name="BetaModel", owner="bob")
        resp = client.get("/api/search?q=alpha")
        data = resp.get_json()
        assert data["count"] == 1
        assert data["models"][0]["modelName"] == "AlphaModel"

    def test_search_case_insensitive(self, client):
        _register(client, name="AlphaModel", owner="alice")
        resp = client.get("/api/search?q=ALPHA")
        assert resp.get_json()["count"] == 1

    def test_search_by_hash(self, client):
        _register(client, name="M1", hash_val="deadbeef", owner="alice")
        _register(client, name="M2", hash_val="cafebabe", owner="alice")
        resp = client.get("/api/search?q=deadbeef")
        assert resp.get_json()["count"] == 1

    def test_search_by_owner_filter(self, client):
        _register(client, name="M1", owner="alice")
        _register(client, name="M2", owner="alice")
        _register(client, name="M3", owner="bob")
        resp = client.get("/api/search?owner=alice")
        data = resp.get_json()
        assert data["count"] == 2

    def test_search_combined_q_and_owner(self, client):
        _register(client, name="AlphaModel", owner="alice")
        _register(client, name="AlphaModel", owner="bob")
        resp = client.get("/api/search?q=alpha&owner=alice")
        data = resp.get_json()
        assert data["count"] == 1
        assert data["models"][0]["owner"] == "alice"

    def test_search_no_match(self, client):
        _register(client, name="TestModel", owner="alice")
        resp = client.get("/api/search?q=zzznomatch")
        assert resp.get_json()["count"] == 0

    def test_search_empty_registry(self, client):
        resp = client.get("/api/search")
        data = resp.get_json()
        assert data["success"] is True
        assert data["count"] == 0

    def test_search_returns_mining_metrics(self, client):
        _register(client, name="M", owner="alice")
        data = client.get("/api/search").get_json()
        assert "modelId" in data["models"][0]


# ═══════════════════════════════════════════════════════════════════
#  /api/tamper-demo
# ═══════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════
#  Proof-of-Work anti-spam  (verify_pow + /api/register enforcement)
# ═══════════════════════════════════════════════════════════════════


class TestProofOfWork:

    def test_prefix_matches_difficulty(self):
        assert app_module.POW_PREFIX == "0" * app_module.POW_DIFFICULTY

    def test_verify_pow_accepts_valid_nonce(self):
        h = "deadbeefcafe"
        assert app_module.verify_pow(h, mine_pow(h)) is True

    def test_verify_pow_rejects_invalid_nonce(self):
        h = "deadbeefcafe"
        assert app_module.verify_pow(h, invalid_pow(h)) is False

    def test_verify_pow_hash_construction(self):
        # Must hash exactly model_hash + str(nonce) and compare against the prefix
        h, nonce = "xyz", 12345
        digest = hashlib.sha256(f"{h}{nonce}".encode()).hexdigest()
        assert app_module.verify_pow(h, nonce) == digest.startswith(app_module.POW_PREFIX)

    def test_register_rejects_invalid_pow(self, client):
        h = "powhashbad"
        resp = client.post("/api/register", json={
            "modelName": "M", "modelHash": h, "owner": "alice", "powNonce": invalid_pow(h),
        })
        assert resp.status_code == 400
        assert "Proof-of-Work" in resp.get_json()["error"]

    def test_register_accepts_valid_pow(self, client):
        h = "powhashgood"
        resp = client.post("/api/register", json={
            "modelName": "M", "modelHash": h, "owner": "alice", "powNonce": mine_pow(h),
        })
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True


# ═══════════════════════════════════════════════════════════════════
#  Local PoW-chain cache  (_get_pow_chain / _invalidate_chain_cache)
# ═══════════════════════════════════════════════════════════════════


class TestChainCache:

    def test_cached_chain_is_reused(self, client):
        c1 = app_module._get_pow_chain()
        c2 = app_module._get_pow_chain()
        assert c1 is c2, "second call should return the cached object, not rebuild"

    def test_invalidation_forces_rebuild(self, client):
        c1 = app_module._get_pow_chain()
        app_module._invalidate_chain_cache()
        c2 = app_module._get_pow_chain()
        assert c1 is not c2, "after invalidation the chain must be rebuilt"

    def test_empty_registry_is_genesis_only(self, client):
        chain = app_module._get_pow_chain()
        assert len(chain) == 1, "no models → genesis block only"

    def test_new_registration_grows_chain_after_invalidation(self, client):
        before = len(app_module._get_pow_chain())
        _register(client, name="CacheModel", hash_val="cachehash", owner="alice")
        app_module._invalidate_chain_cache()   # save_state() does this in production
        after = len(app_module._get_pow_chain())
        assert after > before, "registering a model must add at least one block"

    def test_chain_endpoint_reflects_registry(self, client):
        _register(client, name="ChainModel", hash_val="chainhash", owner="alice")
        app_module._invalidate_chain_cache()
        data = client.get("/api/chain").get_json()
        assert data["success"] is True
        # genesis + at least the registration block
        assert len(data["chain"]) >= 2


# ═══════════════════════════════════════════════════════════════════
#  Merkle tree endpoints  (/api/algo/merkle  &  /api/block/<i>/merkle)
# ═══════════════════════════════════════════════════════════════════


class TestMerkle:

    def test_algo_merkle_empty(self, client):
        data = client.get("/api/algo/merkle").get_json()
        assert data["success"] is True
        assert data["count"] == 0
        assert data["merkleRoot"] is None

    def test_algo_merkle_with_models(self, client):
        _register(client, name="A", hash_val="hashA", owner="alice")
        _register(client, name="B", hash_val="hashB", owner="alice")
        data = client.get("/api/algo/merkle").get_json()
        assert data["count"] == 2
        assert data["merkleRoot"] is not None
        assert len(data["merkleRoot"]) == 64       # SHA-256 hex digest
        assert data["tree"] is not None

    def test_algo_merkle_is_deterministic(self, client):
        _register(client, name="A", hash_val="hashA", owner="alice")
        r1 = client.get("/api/algo/merkle").get_json()["merkleRoot"]
        r2 = client.get("/api/algo/merkle").get_json()["merkleRoot"]
        assert r1 == r2, "same registry must yield the same Merkle root"

    def test_algo_merkle_tamper_changes_root(self, client):
        d = _register(client, name="A", hash_val="hashA", owner="alice").get_json()
        root_before = client.get("/api/algo/merkle").get_json()["merkleRoot"]
        # Mutate a single stored hash — the root MUST change (tamper-evidence)
        models_registry[d["modelId"]]["modelHash"] = "TAMPERED_HASH"
        root_after = client.get("/api/algo/merkle").get_json()["merkleRoot"]
        assert root_before != root_after

    def test_block_merkle_genesis(self, client):
        data = client.get("/api/block/0/merkle").get_json()
        assert data["success"] is True
        assert data["blockIndex"] == 0
        assert len(data["merkleRoot"]) == 64
        assert data["tree"] is not None

    def test_block_merkle_out_of_range(self, client):
        resp = client.get("/api/block/9999/merkle")
        assert resp.status_code == 404
        assert resp.get_json()["success"] is False


# ═══════════════════════════════════════════════════════════════════
#  Real-time alerts  (Flask-SocketIO verification_alert broadcast)
# ═══════════════════════════════════════════════════════════════════


class TestRealtimeAlerts:
    """Verify that /api/verify broadcasts a live alert to connected clients."""

    def _sio_client(self):
        if not getattr(app_module, "SOCKETIO_AVAILABLE", False):
            pytest.skip("flask-socketio not installed")
        return app_module.socketio.test_client(app_module.app)

    def test_tamper_emits_tampered_alert(self, client):
        model_id = _register(client, name="ResNet", hash_val="goodhash").get_json()["modelId"]
        sio = self._sio_client()
        assert sio.is_connected()
        sio.get_received()  # clear connect noise
        # Verify with a WRONG hash → tamper
        client.post("/api/verify", json={"modelId": model_id, "providedHash": "WRONGHASH", "verifier": "bob"})
        events = [r for r in sio.get_received() if r["name"] == "verification_alert"]
        assert len(events) == 1, "exactly one alert should be broadcast"
        payload = events[0]["args"][0]
        assert payload["isValid"] is False
        assert payload["status"] == "TAMPERED"
        assert payload["verifier"] == "bob"
        assert payload["modelName"] == "ResNet"
        assert payload["modelId"] == model_id
        assert "timestamp" in payload
        sio.disconnect()

    def test_valid_emits_pass_alert(self, client):
        model_id = _register(client, name="BERT", hash_val="okhash").get_json()["modelId"]
        sio = self._sio_client()
        sio.get_received()
        client.post("/api/verify", json={"modelId": model_id, "providedHash": "okhash", "verifier": "carol"})
        events = [r for r in sio.get_received() if r["name"] == "verification_alert"]
        assert len(events) == 1
        assert events[0]["args"][0]["status"] == "PASS"
        assert events[0]["args"][0]["isValid"] is True
        sio.disconnect()

    def test_alert_broadcasts_to_all_clients(self, client):
        model_id = _register(client, hash_val="multi").get_json()["modelId"]
        a = self._sio_client()
        b = self._sio_client()
        a.get_received(); b.get_received()
        client.post("/api/verify", json={"modelId": model_id, "providedHash": "WRONG", "verifier": "bob"})
        a_evt = [r for r in a.get_received() if r["name"] == "verification_alert"]
        b_evt = [r for r in b.get_received() if r["name"] == "verification_alert"]
        assert len(a_evt) == 1 and len(b_evt) == 1, "every connected client must receive the alert"
        a.disconnect(); b.disconnect()


# ═══════════════════════════════════════════════════════════════════
#  Blockchain certificate  (POST /api/certificate/<id>, reportlab PDF)
# ═══════════════════════════════════════════════════════════════════


class TestCertificate:

    def _skip_if_no_reportlab(self):
        try:
            import reportlab  # noqa: F401
        except ImportError:
            pytest.skip("reportlab not installed")

    def test_certificate_returns_pdf(self, client):
        self._skip_if_no_reportlab()
        model_id = _register(client, name="CertModel", hash_val="certhash").get_json()["modelId"]
        resp = client.post(f"/api/certificate/{model_id}")
        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "application/pdf"
        assert resp.headers["Content-Disposition"].startswith("attachment")
        data = resp.get_data()
        assert data[:5] == b"%PDF-", "response must be a real PDF"
        assert len(data) > 1500

    def test_certificate_unknown_model(self, client):
        self._skip_if_no_reportlab()
        resp = client.post("/api/certificate/does-not-exist")
        assert resp.status_code == 404
        assert resp.get_json()["success"] is False

    def test_certificate_without_txid_still_valid(self, client):
        # A model with no Algorand TxID must still render a valid PDF (QR -> explorer home)
        self._skip_if_no_reportlab()
        models_registry["notx"] = {
            "modelId": "notx", "modelName": "No-Tx Model", "owner": "alice",
            "modelHash": "h" * 64, "registeredAt": 1700000000, "isActive": True,
        }
        resp = client.post("/api/certificate/notx")
        assert resp.status_code == 200
        assert resp.get_data()[:5] == b"%PDF-"

    def test_certificate_is_post_only(self, client):
        self._skip_if_no_reportlab()
        model_id = _register(client, hash_val="getcert").get_json()["modelId"]
        # GET is not allowed on the certificate route
        assert client.get(f"/api/certificate/{model_id}").status_code == 405


# ═══════════════════════════════════════════════════════════════════
#  Chain verify-tx result field  (feeds the Analytics pie chart)
# ═══════════════════════════════════════════════════════════════════


class TestChainResultField:

    def test_verify_result_reflects_validity(self, client):
        mid = _register(client, hash_val="rh").get_json()["modelId"]
        client.post("/api/verify", json={"modelId": mid, "providedHash": "rh"})       # valid
        client.post("/api/verify", json={"modelId": mid, "providedHash": "WRONG"})    # invalid
        app_module._invalidate_chain_cache()
        chain = client.get("/api/chain").get_json()["chain"]
        results = [tx["result"] for b in chain for tx in b["transactions"] if tx.get("type") == "verify"]
        assert "valid" in results and "invalid" in results
        assert "unknown" not in results


# ═══════════════════════════════════════════════════════════════════
#  Batch verification  (POST /api/verify-batch)
# ═══════════════════════════════════════════════════════════════════


class TestBatchVerify:

    def test_batch_classifies_each_file(self, client):
        _register(client, name="ModelA", hash_val="hashAAA")
        _register(client, name="ModelB", hash_val="hashBBB")
        resp = client.post("/api/verify-batch", json={"files": [
            {"filename": "a.bin",       "hash": "hashAAA"},     # verified (hash match)
            {"filename": "ModelB.pt",   "hash": "WRONGHASH"},   # tampered (name match, hash differs)
            {"filename": "mystery.onnx", "hash": "zzz999"},     # unregistered
        ]})
        d = resp.get_json()
        assert resp.status_code == 200 and d["success"] is True
        by_file = {r["filename"]: r for r in d["results"]}
        assert by_file["a.bin"]["status"] == "verified"
        assert by_file["a.bin"]["modelName"] == "ModelA"
        assert by_file["ModelB.pt"]["status"] == "tampered"
        assert by_file["mystery.onnx"]["status"] == "unregistered"
        assert d["summary"] == {"verified": 1, "tampered": 1, "unregistered": 1}
        assert d["total"] == 3

    def test_batch_matches_historical_version(self, client):
        mid = _register(client, name="Vmodel", hash_val="v1hash").get_json()["modelId"]
        client.post("/api/add-version", json={"modelId": mid, "newHash": "v2hash", "changelog": "v2"})
        d = client.post("/api/verify-batch", json={"files": [{"filename": "old.bin", "hash": "v1hash"}]}).get_json()
        assert d["results"][0]["status"] == "verified"
        assert d["results"][0]["version"] == 1

    def test_batch_rejects_empty(self, client):
        assert client.post("/api/verify-batch", json={"files": []}).status_code == 400

    def test_batch_rejects_over_20(self, client):
        files = [{"filename": f"f{i}", "hash": f"h{i}"} for i in range(21)]
        assert client.post("/api/verify-batch", json={"files": files}).status_code == 400

    def test_batch_requires_auth(self, client):
        resp = client.post("/api/verify-batch", json={"files": [{"filename": "x", "hash": "y"}]}, headers={"No-Auth": True})
        assert resp.status_code == 401
