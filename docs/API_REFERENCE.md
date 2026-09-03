# BlockVerify API Reference

Base URL: `http://localhost:5000/api`

All responses are JSON. Write endpoints mine a new block (takes 1-5 seconds).

## Write Endpoints

### POST /api/register
Register a new AI model.

Request: `{"modelName": "ResNet50-v1", "modelHash": "abc123...", "metadata": "optional", "owner": "alice"}`

Success (200): `{"success": true, "modelId": "a1b2c3d4...", "blockIndex": 5, "message": "..."}`

Errors: 400 (missing fields), 500 (server error)

### POST /api/verify
Verify model integrity against stored hash.

Request: `{"modelId": "a1b2c3d4...", "providedHash": "xyz789...", "verifier": "bob"}`

Success (200): `{"success": true, "isValid": true/false, "message": "...", "blockIndex": 6, "storedHash": "...", "providedHash": "..."}`

Errors: 400 (missing fields), 404 (model not found)

### POST /api/add-version
Add new version to existing model (owner only).

Request: `{"modelId": "...", "newHash": "...", "changelog": "Fixed preprocessing bug", "owner": "alice"}`

Success (200): `{"success": true, "version": 2, "blockIndex": 7, "message": "..."}`

Errors: 400, 403 (not owner), 404 (not found)

### POST /api/deactivate
Soft-delete a model (owner only).

Request: `{"modelId": "...", "owner": "alice"}`

Success (200): `{"success": true, "message": "Model deactivated"}`

## Read Endpoints

### GET /api/models/:owner
All models by owner. Returns `{"success": true, "models": [...], "count": N}`

### GET /api/model/:modelId
Single model details. Returns `{"success": true, "model": {...}}`

### GET /api/versions/:modelId
Version history. Returns `{"success": true, "versions": [...], "currentVersion": N}`

### GET /api/audit/:modelId
Verification audit trail. Returns `{"success": true, "verifications": [...], "count": N}`

### GET /api/chain
Full blockchain. Returns `{"success": true, "chain": [...], "length": N}`

### GET /api/chain/validate
Validate chain integrity. Returns `{"success": true, "isValid": true/false, "errors": [...], "message": "..."}`

### GET /api/stats
Platform statistics. Returns `{"totalModels": N, "totalVerifications": N, "totalBlocks": N}`

## Error Format

All errors follow: `{"success": false, "error": "Description"}`

HTTP status codes: 200 (success), 400 (bad request), 403 (forbidden), 404 (not found), 500 (server error)

---

## Federated Learning (`/api/fl/*`)

Blueprint `fl_bp` (`backend/fl_routes.py`). Same error envelope and status codes as above.

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/fl/round/commit` | Register one round's commitment |
| `GET` | `/api/fl/runs` | List every committed run, newest first |
| `GET` | `/api/fl/run/<run_id>` | Run metadata + every round root |
| `GET` | `/api/fl/round/<run_id>/<r>` | One round: root, leaf count, txid, client ids |
| `GET` | `/api/fl/proof/<run_id>/<r>/<client_id>` | Inclusion proof for one client |
| `GET` | `/api/fl/lineage/<run_id>` | Ordered roots + accepted/rejected per round |

### `POST /api/fl/round/commit`

```json
{
  "run_id": "exp2_mnist_K10_a0.5_epsinf_forensics_scaling0.2_s0",
  "round": 7,
  "root": "<64-hex>",
  "leaf_count": 10,
  "txid": "...", "backend": "algorand",
  "leaves": [{"client_id": "0", "digest": "<64-hex>"}, ...],
  "accepted": ["0","1"], "rejected": ["8","9"],
  "metrics": {"test_acc": 0.94, "macro_f1": 0.93}
}
```

**The server re-derives the root from the submitted leaves and rejects the commit if it
disagrees** with the claimed root or leaf count — a caller cannot register a lineage that
does not fold to its own root.

`201` on success. `400` for `Root mismatch: …`, `leaf_count mismatch: …`, or malformed
input. `accepted`/`rejected`/`metrics` are optional.

> Like the other write routes in this codebase, this endpoint is **unauthenticated**. A
> deployment should apply `@require_auth` from `backend/auth.py`.

### `GET /api/fl/proof/<run_id>/<r>/<client_id>`

```json
{
  "success": true, "round": 7, "clientId": "3", "index": 3,
  "digest": "<64-hex>", "leaf": "<64-hex>",
  "proof": [{"hash": "<64-hex>", "right": true}, ...],
  "root": "<64-hex>", "leafCount": 10,
  "txid": "...", "indexerUrl": "https://testnet-idx.algonode.cloud/v2/transactions/<txid>"
}
```

`indexerUrl` is present only for real Algorand txids. The client is expected to
**recompute** the leaf from `digest`, fold `proof`, and compare against the root it reads
from `indexerUrl` itself — never against the `root` this response supplies. See
[`fedverify/docs/FEDERATED.md`](../fedverify/docs/FEDERATED.md) §3.

`404` for an unknown run, round, or a client that did not contribute to that round.
