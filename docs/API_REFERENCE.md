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

---

## Photo Shield (`/api/photo/*`)

Blueprint `photo_bp` (`backend/photo_routes.py`). Design: [`PHOTOSHIELD_DESIGN.md`](../PHOTOSHIELD_DESIGN.md) §6.
Same error envelope as above, plus `409` (already registered by another key), `429` (rate limit),
`501` (server cloak off), `502` (anchoring failed).

The server never receives pixels on these routes. The browser (or `python -m photoshield`) sends a
**manifest** (`photoshield/manifest@1`: SHA-256 and pHash of the original and, optionally, of the
cloaked PNG, plus cloak parameters, with no floats) and an Ed25519 signature by the owner's key.

| Method | Route | Auth | Purpose |
|---|---|---|---|
| `POST` | `/api/photo/register` | JWT | Anchor one photo |
| `POST` | `/api/photo/register-batch` | JWT | Anchor an album with one Merkle-root transaction |
| `POST` | `/api/photo/verify` | public, per-IP limit | Check a photo's fingerprints against the registry |
| `POST` | `/api/photo/rehydrate` | JWT | Re-attach a manifest (hash-only photos, albums) after a restart |
| `GET` | `/api/photo/challenge` | public | One-time challenge for an ownership proof |
| `POST` | `/api/photo/prove` | optional JWT | Prove control of the registering key (links it to the account if logged in) |
| `GET` | `/api/photo/stats` | public | Counts, anchor backend, ledger-sync status, calibration |
| `GET` | `/api/photo/owner/<pk or username>` | public by key; own username needs JWT | List a key's (or your own) photos |
| `GET` | `/api/photo/proof/<photoId>` | public | Anchor data, and the album inclusion proof if any |
| `GET` | `/api/photo/<photoId>` | public | One photo's public record |
| `GET` | `/api/photo/certificate/<photoId>` | public | PDF certificate |
| `POST` | `/api/photo/cloak` | public | Server-side cloak; `501` unless `PHOTOSHIELD_SERVER_CLOAK=1` |

### `POST /api/photo/register`

```json
{
  "manifest": {"schema": "photoshield/manifest@1", "title": "Profile photo",
               "original": {"sha256": "<64-hex>", "phash": "<16-hex>", "width": 1024, "height": 768},
               "cloaked": null, "cloak": null},
  "privacy": "full",
  "pk": "<64-hex Ed25519 public key>",
  "sig": "<128-hex signature over 'photoshield/register@1|' + manifestDigest>",
  "powNonce": 1234
}
```

- `manifestDigest = sha256(canonical_json(manifest))`, where canonical JSON has sorted keys, no
  spaces and UTF-8.
- `photoId` is the first 16 hex of `sha256("ps-id|" + manifestDigest)`.
- `powNonce` makes `SHA-256(original.sha256 + nonce)` start with `000`, the same Hashcash rule as
  `/api/register`.
- `privacy: "hash-only"` leaves the pHash off chain.

`200` → `{photoId, manifestDigest, algoTxId, algoRound, anchorBackend, noteBytes, indexerUrl}`.
Registering the same photo again under the same key returns `200` with `"duplicate": true`. Under
a different key it returns `409` with the earlier registration's `photoId`, `registeredAt`,
`algoTxId` and `ownerPkFp`.

### `POST /api/photo/register-batch`

`{albumTitle, manifests[], privacy, pk, sig, powNonce}`. Album limits:
- At most `MAX_ALBUM` manifests.
- PoW is over the album root.
- `sig` signs `photoshield/album@1|<root>|<count>`.

Leaves use BlockVerify's canonical Merkle leaf `SHA256(JSON([i, photoId, manifestDigest]))`. The
note carries the root, the leaf count and the digest of the album manifest
`{"schema": "photoshield/album@1", "title", "photos": [[photoId, manifestDigest], ...]}`.

The response lists every photo's `leaf` and inclusion `proof`, so the browser's existing
`bvFoldProof` verifies them.

### `POST /api/photo/verify`

`{sha256, phash}`, computed in the browser.

```json
{"success": true, "verdict": "DERIVED", "distance": 3, "matchedPart": "original",
 "photoId": "...", "ownerPkFp": "<16-hex>", "registeredAt": 1730000000,
 "algoTxId": "...", "indexerUrl": "...", "warning": "Perceptual match is evidence, not proof.",
 "nearMatch": "calibrated", "thresholds": {"tauNear": "...", "tauFar": "...", "source": "calibration.json"},
 "ledgerSync": "ok"}
```

Verdicts:
- `EXACT_ORIGINAL` and `EXACT_CLOAKED` (SHA-256 match). These also return `ownerUser` when known.
- `DERIVED` (distance ≤ τ_near) and `SIMILAR` (≤ τ_far). These never return a username.
- `UNKNOWN` otherwise.

The thresholds come from `photoshield/calibration.json`. Without that file, near matching is
disabled and the response says `"nearMatch": "uncalibrated"`. The example values above are
placeholders; the real ones are in the file.

### `GET /api/photo/challenge` → `POST /api/photo/prove`

`{challenge, expiresIn}`, then `{photoId, challenge, sig}` where `sig` signs
`photoshield/prove@1|<photoId>|<challenge>`. Returns `{proven: true|false, ownerPkFp, linkedTo}`.
Challenges are single-use.

### `POST /api/photo/rehydrate`

- `{manifest}`: accepted only if it hashes to an anchored `md`. Restores the pHash, title and cloak.
- `{album, manifests[]}`: accepted only if the manifests rebuild the anchored root and count, and
  the album manifest matches the anchored digest.

### Origin reports (PhotoShield v2, design §15)

- **Manifest field.** A manifest may carry `"regions": {"algo": "tile-phash@1", "grid", "n", "root"}`,
  the Merkle root of the photo's per-tile fingerprints. It is covered by the manifest digest, so
  the on-chain note format is unchanged. Records and `GET /api/photo/<id>` include `regions`.
- **Thresholds.** `GET /api/photo/stats` returns `calibration.tile` = `{tau_match, tau_alter,
  delta_level, k_min, source}`, measured by `study e8`.
- **No report route.** Reports are made and verified in the browser or with the CLI
  (`python -m photoshield origin` / `origin-verify`). The original photo never reaches the server.

### Server configuration

Set with environment variables (design §6.11):

| Variable | Purpose |
|---|---|
| `PHOTOSHIELD_ANCHOR` | `algorand` (default), `local` or `mock` |
| `PHOTOSHIELD_LEDGER_SYNC` | `0` disables the indexer rebuild |
| `PHOTOSHIELD_ANCHOR_ADDRESSES` | Extra sender addresses to rebuild from |
| `PHOTOSHIELD_INDEXER` | Indexer base URL |
| `PHOTOSHIELD_CALIBRATION` | Path to the threshold file |
| `PHOTOSHIELD_SERVER_CLOAK` | `1` enables the cloak route |
| `PHOTOSHIELD_SERVER_MODEL` | Encoder for the server cloak |
| `PHOTOSHIELD_SERVER_THREADS` | Torch threads for the server cloak |

The registry rebuilds itself from Algorand notes that start with `{"ps":1,`. It only accepts notes
sent by the server wallet (and any extra addresses) that carry a valid owner signature.
