"""Phase-3 route tests for the additive `fl_bp` Blueprint.

These live under fedverify/ rather than backend/tests/ so the existing backend suite
keeps its exact baseline; they still exercise the real Flask app with the blueprint
registered. FL_FILE is redirected to a tmp_path so the developer's data/fl_runs.json is
never touched.
"""
import json
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "backend"))

from fedverify.chain import commitment as C     # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app as app_module
    import fl_routes
    monkeypatch.setattr(fl_routes, "FL_FILE", str(tmp_path / "fl_runs.json"))
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _delta(i):
    return torch.randn(64, generator=torch.Generator().manual_seed(i))


class _U:
    def __init__(self, i):
        self.client_id, self.delta, self.num_samples = i, _delta(i), 100 + i


def _commit_payload(run_id, rnd, k=5, **over):
    cm = C.commit_round([_U(i) for i in range(k)], rnd)
    body = {"run_id": run_id, "round": rnd, "root": cm["root"],
            "leaf_count": cm["leaf_count"], "txid": f"TX{rnd}", "backend": "mock",
            "leaves": [{"client_id": l["client_id"], "digest": l["digest"]}
                       for l in cm["leaves"]]}
    body.update(over)
    return body


def _seed_run(client, run_id="demo", rounds=3, k=5):
    for r in range(1, rounds + 1):
        resp = client.post("/api/fl/round/commit", json=_commit_payload(run_id, r, k))
        assert resp.status_code == 201, resp.get_json()


# ── commit ───────────────────────────────────────────────────────────────────
def test_commit_accepts_a_consistent_round(client):
    resp = client.post("/api/fl/round/commit", json=_commit_payload("r1", 1))
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["success"] is True and body["leaf_count"] == 5


def test_commit_rejects_root_that_leaves_do_not_fold_to(client):
    """The server re-derives the root; it does not take the caller's word for it."""
    resp = client.post("/api/fl/round/commit", json=_commit_payload("r1", 1, root="a" * 64))
    assert resp.status_code == 400
    assert "Root mismatch" in resp.get_json()["error"]


def test_commit_rejects_mismatched_leaf_count(client):
    resp = client.post("/api/fl/round/commit", json=_commit_payload("r1", 1, leaf_count=99))
    assert resp.status_code == 400
    assert "leaf_count mismatch" in resp.get_json()["error"]


def test_commit_rejects_swapped_client_order(client):
    """Position is committed, so a reordered manifest cannot claim the original root."""
    body = _commit_payload("r1", 1)
    body["leaves"][0], body["leaves"][1] = body["leaves"][1], body["leaves"][0]
    resp = client.post("/api/fl/round/commit", json=body)
    assert resp.status_code == 400


@pytest.mark.parametrize("bad,field", [
    ({"run_id": ""}, "run_id"), ({"round": "x"}, "round"),
    ({"round": 0}, "round"), ({"leaves": []}, "leaves"), ({"root": "short"}, "root"),
])
def test_commit_validation_errors(client, bad, field):
    resp = client.post("/api/fl/round/commit", json={**_commit_payload("r1", 1), **bad})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False and isinstance(body["error"], str)


# ── reads ────────────────────────────────────────────────────────────────────
def test_lineage_returns_rounds_in_order(client):
    _seed_run(client, rounds=3)
    body = client.get("/api/fl/lineage/demo").get_json()
    assert [e["round"] for e in body["lineage"]] == [1, 2, 3]
    assert all(len(e["root"]) == 64 for e in body["lineage"])
    assert body["lineage"][0]["accepted"] and body["lineage"][0]["rejected"] == []


def test_lineage_order_is_numeric_not_lexicographic(client):
    """Round 10 must sort after round 9, not after round 1."""
    _seed_run(client, rounds=11)
    assert [e["round"] for e in client.get("/api/fl/lineage/demo").get_json()["lineage"]] \
        == list(range(1, 12))


def test_run_metadata_lists_every_round_root(client):
    _seed_run(client, rounds=3)
    body = client.get("/api/fl/run/demo").get_json()
    assert body["numRounds"] == 3 and len(body["roundRoots"]) == 3


def test_round_detail_exposes_client_ids_and_txid(client):
    _seed_run(client, rounds=2)
    body = client.get("/api/fl/round/demo/2").get_json()
    assert body["clientIds"] == [str(i) for i in range(5)]
    assert body["txid"] == "TX2" and body["leafCount"] == 5


# ── proofs: the trustless part ───────────────────────────────────────────────
def test_served_proof_verifies_against_the_served_root(client):
    _seed_run(client, rounds=2)
    body = client.get("/api/fl/proof/demo/2/3").get_json()
    assert C.verify_proof(body["leaf"], body["proof"], body["root"])


def test_one_character_change_breaks_the_served_proof(client):
    _seed_run(client, rounds=1)
    body = client.get("/api/fl/proof/demo/1/2").get_json()
    tweaked = body["leaf"][:-1] + ("0" if body["leaf"][-1] != "0" else "1")
    assert not C.verify_proof(tweaked, body["proof"], body["root"])


def test_every_client_gets_a_verifying_proof(client):
    _seed_run(client, rounds=1, k=7)          # odd count exercises duplicate-last
    for cid in range(7):
        b = client.get(f"/api/fl/proof/demo/1/{cid}").get_json()
        assert C.verify_proof(b["leaf"], b["proof"], b["root"]), cid


# ── 404s ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("url", [
    "/api/fl/run/nope", "/api/fl/lineage/nope",
    "/api/fl/round/nope/1", "/api/fl/proof/nope/1/0",
])
def test_unknown_run_is_404_with_envelope(client, url):
    resp = client.get(url)
    assert resp.status_code == 404
    assert resp.get_json()["success"] is False


def test_unknown_round_and_client_are_404(client):
    _seed_run(client, rounds=1)
    assert client.get("/api/fl/round/demo/99").status_code == 404
    assert client.get("/api/fl/proof/demo/1/42").status_code == 404


# ── isolation ────────────────────────────────────────────────────────────────
def test_blueprint_registers_all_five_routes_without_shadowing(client):
    """fl_bp adds exactly its own rules and disturbs no pre-existing route."""
    import app as app_module
    rules = {str(r) for r in app_module.app.url_map.iter_rules()}
    assert {"/api/fl/round/commit", "/api/fl/run/<run_id>",
            "/api/fl/round/<run_id>/<int:rnd>",
            "/api/fl/proof/<run_id>/<int:rnd>/<client_id>",
            "/api/fl/lineage/<run_id>"} <= rules
    # a pre-existing route still resolves and is not captured by the new blueprint
    assert client.get("/api/models/alice").status_code == 200
