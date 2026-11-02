# PhotoShield: design spec for photo cloaking and provenance in BlockVerify

**Status:** design only. P0 (inventory) is done; nothing else is built.
**Repository:** `Shubhangam-Singh/BlockVerify`. The real HEAD at P0 and the test baselines are
recorded in [`photoshield/INVENTORY.md`](photoshield/INVENTORY.md). This document does not repeat
them, so it cannot go stale.
**Rules:** [`CLAUDE.md`](CLAUDE.md) and its amendments, A12 to A15 in particular.

## Revision log

Revision 2 (P0) folds in the review of revision 1. Each change is marked **(r2)** where it lands.
Revision 3 (P1) records two decisions made while building, marked **(r3)**: no floats in a
manifest (§5.1), and the separable DCT order (§7 step 5).
Revision 4 (P8) records what P4–P7 settled while building, marked **(r4)**: the ownership-proof
and rehydrate routes, the `PHOTOSHIELD_*` environment variables, the album manifest schema, the
calibration file's real path and command, and the results layout.

| # | Change | Section |
|---|---|---|
| 1 | Album leaves use the repository's canonical Merkle leaf `SHA256(JSON([i, photoId, manifestDigest]))`, not a new encoding | §6.2 |
| 2 | Album leaf count key renamed `plc` (it clashed with `pc`, the cloaked pHash) | §5.3 |
| 3 | No contract global-state writes for photos: anchors are note-only | §6.1, §6.7 |
| 4 | The registry is rebuilt from the ledger on startup (paged indexer scan, cached), because Render's free disk is wiped on restart | §6.8 |
| 5 | Anchors are bound to an owner Ed25519 key held in the browser, with export, backup and import | §5.3, §9.2 |
| 6 | The note carries the owner's public key, not their username | §5.3 |
| 7 | Hash-only privacy photos: only exact matches survive a restart unless the owner re-attaches the manifest | §6.9 |
| 8 | pHash spec tightened: PNG vectors for bit-identity, fixed summation order, precomputed cosine table, alpha, minimum size, orientation, colour handling | §7 |
| 9 | `/api/photo/verify` returns an owner identity only for exact matches | §6.3 |
| 10 | The Ledger tab learns to label photo anchors | §6.10 |
| 11 | The tab id goes into `_TAB_IDS` as well as `switchTab` | §9 |
| 12 | Network limits of the cloud container; real-weight runs happen on the laptop | §8, CLAUDE.md A15 |
| 13 (r4) | Ownership proof: `GET /api/photo/challenge` + `POST /api/photo/prove`; manifest re-attach: `POST /api/photo/rehydrate` | §6.4, §6.9 |
| 14 (r4) | Album manifest schema `photoshield/album@1`, whose digest is the anchored `amd` | §6.2 |
| 15 (r4) | Calibration lives at `photoshield/calibration.json`, written by `python -m photoshield study e6 --install` | §6.3, §10 |
| 16 (r4) | Server configuration through `PHOTOSHIELD_*` environment variables | §6.11 |
| 17 (r4) | Results layout `photoshield/results/<exp>/<cell>/seed<N>/{config.json, records.jsonl}` (gitignored) | §10, §11 |

---

## 1. The idea in one paragraph

BlockVerify proves that an AI **model file** has not been tampered with: hash it in the browser,
anchor the hash on Algorand, re-hash it later and compare. PhotoShield applies the same pipeline
to **personal photos** and adds a second, active layer:

1. **Cloak (prevention, partial).** A bounded adversarial perturbation (PGD, L∞ ≤ ε) makes
   open-source editors built on the Stable Diffusion VAE produce damaged edits.
2. **Provenance (proof).** The original's and the cloaked version's fingerprints are anchored on
   Algorand and signed by the owner's key. The owner can later show that *this exact photo
   existed at time T, registered under this key*, and whether a circulating image is derived
   from it.

Cloaking raises the cost of misuse on some models. Provenance gives evidence. Neither stops
deepfakes made with closed models, and the UI, README and report say so (§12).

## 2. How it maps onto BlockVerify

| BlockVerify (models) | PhotoShield (photos) | Reuse or new |
|---|---|---|
| SHA-256 of a model file, computed in the browser | SHA-256 of an image file, computed in the browser | Reuse the `crypto.subtle` path (`computeHash`, `bvSha256Hex`) |
| Layer hashes, layer Merkle root (`lmr`, `lc`) | Album of photos, album Merkle root (`pmr`, `plc`) **(r2)** | Reuse `compute_layer_merkle` / `merkle_inclusion_proof` and the browser's `bvMerkleLeaf` / `bvFoldProof` |
| Hashcash PoW nonce on register | The same PoW on photo register | Reuse `verify_pow` and the browser's `minePoW` |
| Algorand 0-ALGO note transaction | The same, with new note keys | New function `broadcast_photo_anchor`, additive |
| Contract `model_id → hash` | **None (r2).** Photos are note-only (§6.7) | Not reused |
| Exact hash compare on verify | Exact hash **plus** perceptual-hash near match | New |
| Server wallet signs everything | Server wallet pays and anchors; **the owner's key signs the claim (r2)** | New |
| Tamper demo models | Cloak effectiveness study | New |
| FL Blueprint `fl_routes.py` | `photo_routes.py` Blueprint | Same pattern |

The point to stress in the report: BlockVerify's model check can be exact because a model file is
never re-encoded. Photos are re-encoded constantly (WhatsApp, Instagram), so SHA-256 alone fails on
any shared copy. That is why PhotoShield adds a **perceptual hash**. It is the main technical
difference from the model pipeline.

## 3. Architecture

```
            ┌──────────────────────── user's laptop (CPU) ────────────────────────┐
 photo.jpg ─▶ photoshield CLI: protect                                           │
            │   1. load + resize (--size)                                         │
            │   2. PGD vs encoder (sdvae | resnet50 | facenet)  ──▶ cloaked.png   │
            │   3. metrics: PSNR, SSIM, latent shift, VAE-recon damage             │
            │   4. hashes: sha256 + pHash (original & cloaked)                     │
            │   5. write manifest.json (canonical JSON, no pixels)                 │
            └────────────────────────────────────┬────────────────────────────────┘
                                                 │ manifest.json only (never the photo)
                                                 ▼
 Browser (Photo Shield tab)
   ├─ re-hashes the image files it is given, checks they match the manifest
   ├─ signs "photoshield/register@1|" + manifestDigest with the owner's Ed25519 key (r2)
   └─ mines the PoW nonce ──▶ POST /api/photo/register
                                                 │
                                   backend/photo_routes.py (Blueprint)
                                   ├─ validate, PoW, signature, rate limit, JWT user
                                   ├─ registry: backend/data/photo_registry.json
                                   │   (rebuilt from the ledger on startup, §6.8)
                                   └─ algorand_client.broadcast_photo_anchor()
                                                 │
                                       Algorand Testnet note transaction (no contract call)
                                                 │
 Verify: the browser hashes a suspect image (sha256 + pHash) ──▶ POST /api/photo/verify
         ◀── verdict + txid. The browser then reads the note STRAIGHT FROM the public indexer
             and re-checks the fields and the owner signature itself. The server's answer is
             treated as untrusted.
```

**Cloaking runs locally, not on the server.** Uploading a face photo to protect it contradicts the
privacy premise, and Render's free tier (512 MB RAM) cannot run PGD on the SD VAE anyway. The
server only ever sees hashes, the manifest and a signature. An optional server cloak route exists
behind an environment flag and is off by default (§6.6).

## 4. Repository structure (all new code is additive)

```
photoshield/                          NEW top-level package (same convention as fedverify/)
├── __init__.py
├── README.md
├── INVENTORY.md                      P0: reused functions, exact paths, baselines
├── TODO.md                           deferred ideas (CLAUDE.md A13)
├── requirements.txt                  torch/torchvision (CPU index on the laptop), diffusers, scikit-image
├── config.py                         dataclass ShieldConfig: model, eps, steps, size, seed, eot, threads
├── cloak/
│   ├── encoders.py                   load_encoder(name) -> callable [0,1] image -> features
│   │                                   resnet50 | sdvae (stabilityai/sd-vae-ft-mse latent mean) | facenet (opt)
│   │                                   | tiny (random weights, tests only, no download)
│   ├── pgd.py                        pgd_attack(x, enc, eps, steps, alpha, target=None, eot=None, seed)
│   ├── eot.py                        random resize, gaussian blur, differentiable JPEG (choice made in P2)
│   └── metrics.py                    psnr, ssim, rel_latent_shift, cosine, vae_recon_damage, lpips (opt)
├── provenance/
│   ├── hashing.py                    sha256_file, phash64 (canonical spec §7), hamming
│   ├── dct_table.py                  GENERATED constant table for §7 step 4 (do not edit by hand)
│   ├── manifest.py                   build/validate/canonicalize manifest, manifest_digest
│   ├── signing.py                    Ed25519 verify (PyNaCl), signing-message builders (§5.4)
│   └── merkle.py                     thin wrapper over backend app's Merkle (do NOT re-implement)
├── chain/
│   ├── anchor.py                     PhotoAnchor: mock | local | algorand (mirrors fedverify ChainAnchor)
│   └── ledger_sync.py                paged indexer scan, filters, cache (§6.8)
├── cli.py                            `python -m photoshield <cmd>`: protect | hash | register | verify | study
├── experiments/
│   ├── study.py                      eps sweep, robustness grid, VAE-recon test, pHash calibration
│   └── colab_edit_test.ipynb         InstructPix2Pix / SD img2img, fixed seeds, original vs cloaked
├── analysis/
│   ├── make_tables.py                every table number comes from results files (CLAUDE.md "Numbers")
│   └── plots.py
├── tests/
│   ├── test_pgd.py  test_hashing.py  test_manifest.py  test_signing.py
│   ├── test_anchor.py  test_ledger_sync.py  test_photo_routes.py
│   ├── phash_js_test.mjs             runs the shipped frontend pHash against the vectors under Node
│   └── vectors/                      PNG test images + phash_vectors.json (expected hex)
└── docs/
    ├── WHAT_IS_PHOTOSHIELD.md
    └── THREAT_MODEL.md

backend/photo_routes.py               NEW Blueprint (the fl_routes.py shape)
backend/app.py                        +2 lines: import photo_bp; app.register_blueprint(photo_bp)
backend/algorand_client.py            +1 function broadcast_photo_anchor();
                                      +1 additive branch in get_transaction_ledger for "ps" notes (§6.10)
frontend/index.html                   +1 sidebar button, +1 <div id="tab-photoshield">,
                                      +'photoshield' in _TAB_IDS, +1 line in switchTab (r2),
                                      + a <script> block: photo hashing, pHash, keys, verify
docs/API_REFERENCE.md                 + a "Photo Shield" section
```

`backend/requirements.txt` needs **no** new dependency for the backend side. Ed25519 verification
uses PyNaCl, which is already installed as a dependency of `py-algorand-sdk` (see INVENTORY).

## 5. Data model

### 5.1 `manifest.json` (written by the CLI; the only thing the UI uploads)

```json
{
  "schema": "photoshield/manifest@1",
  "title": "Profile photo Nov 2026",
  "original": { "sha256": "<64 hex>", "phash": "<16 hex>", "width": 1024, "height": 1280 },
  "cloaked":  { "sha256": "<64 hex>", "phash": "<16 hex>", "format": "png" },
  "cloak": {
    "model": "sdvae", "eps": 8, "steps": 40, "size": 256, "eot": ["resize","blur"],
    "seed": 0, "psnr": "<decimal string>", "ssim": "<decimal string>",
    "latent_shift": "<decimal string>", "recon_psnr_drop": "<decimal string>",
    "tool_version": "photoshield 0.1.0"
  }
}
```

**(r3) No floats anywhere in a manifest.** Python prints `1.0` and JavaScript prints `1`, so
a float would make the browser's digest differ from the server's and break every owner
signature. Parameters are integers; measured metrics are decimal strings with 4 places
(`"psnr": "34.1200"`), written by `manifest.metric()`. `validate()` rejects any float. `cloaked` and `cloak` are `null` for a
provenance-only registration (no cloak). The manifest is the owner's **backup** of everything that
is not on chain (§6.9), so the UI tells them to keep it.

`manifestDigest = sha256(canonical_json(manifest))`, where canonical JSON means sorted keys,
separators `(",", ":")`, UTF-8 and `ensure_ascii=False`. The same convention is used by
`fedverify/chain/signing.py:signing_bytes`. One digest pins every parameter, including the cloak
settings.

### 5.2 Server registry `backend/data/photo_registry.json`

```
photoId -> {
  photoId, ownerPk, ownerUser | null, title, registeredAt, privacy: "full" | "hash-only",
  original: {sha256, phash | null}, cloaked: {sha256, phash | null} | null,
  cloak: {...} | null, manifestDigest, signature,
  albumId | null, albumIndex | null, albumProof | null,
  algoTxId, algoRound, isActive, source: "register" | "ledger" | "rehydrate"
}
```

- `ownerUser` is the JWT username. It is only known for photos registered since the last restart,
  or re-linked when that user logs in with the matching key (§9.2).
- In-memory indexes are rebuilt on load: `sha256 -> photoId` for originals and cloaked copies, and a
  list of `(phash_int, photoId)` for near-match scans.
- A linear Hamming scan is fine up to about 100k photos. A BK-tree or multi-index hashing is future
  work (`photoshield/TODO.md`).
- `photoId = sha256("ps-id|" + manifestDigest)[:16]`. This is deterministic, so a ledger rebuild
  recreates the same ids.

### 5.3 Algorand note (≤ 1024 bytes, asserted before sending)

Keys are emitted **in this order** with compact separators, so every note starts with the bytes
`{"ps":1,`. The ledger scan relies on that prefix (§6.8).

Single photo, privacy `full`:
```json
{"ps":1,"id":"<16 hex>","o":"<sha256 orig>","c":"<sha256 cloaked>|null","po":"<phash orig>",
 "pc":"<phash cloaked>|null","md":"<manifest digest>","pk":"<ed25519 pubkey, 64 hex>",
 "sig":"<ed25519 signature, 128 hex>"}
```
That is 503 bytes with every field present (measured with compact separators), and 373 bytes for an album note.

- Privacy `hash-only` omits `po` and `pc` (§6.9).
- **(r2)** There is no username in the note. The owner is identified by `pk`, which is pseudonymous,
  so the chain does not reveal usernames.

Album (one transaction for N photos):
```json
{"ps":1,"pmr":"<album root>","plc":<leaf count>,"amd":"<album manifest digest>",
 "pk":"<pubkey>","sig":"<signature>"}
```
**(r2)** The leaf count key is `plc`, not `pc`. It is anchored beside the root, the same
CVE-2012-2459 defence as `lmr`/`lc` and `frr`/`flc`.

### 5.4 What the owner signs (domain-separated)

- Single: `"photoshield/register@1|" + manifestDigest`
- Album: `"photoshield/album@1|" + albumRoot + "|" + str(leafCount)`
- Proof of ownership later (§9.2): `"photoshield/prove@1|" + photoId + "|" + challengeNonce`

The prefixes stop a signature made for one purpose from being replayed as another.

## 6. Backend: `photo_routes.py`

The error envelope is the same as the rest of the API: `{"success": false, "error": "..."}`.

### 6.1 `POST /api/photo/register` (auth, PoW, signature, rate limit)
Body: `{manifest, powNonce, pk, sig, privacy}`.
1. Validate the schema, hex lengths (sha256 = 64, phash = 16, pk = 64, sig = 128), and a title of at
   most 120 characters with no markup. `privacy` is `full` (default) or `hash-only`.
2. Recompute `manifestDigest` server-side. Never trust a client-sent digest.
3. `verify_pow(original.sha256, powNonce)`. Import it from `app` lazily, as `fl_routes._commitment`
   does, to avoid a circular import.
4. Verify `sig` over `"photoshield/register@1|" + manifestDigest` under `pk` (PyNaCl).
5. Duplicate check on `original.sha256`:
   - Same `pk` → idempotent, return the existing record.
   - Different `pk` → **409** with the first anchor's txid and round time. That *is* the
     provenance answer.
6. `broadcast_photo_anchor(...)`. **(r2) No contract call.** The TEAL app has 64 global byte slices
   in total (`contract.py:42`), shared with model registrations (up to 2 slots each). Photos would
   exhaust them and silently break model registration.
7. Persist atomically (tmp file + `os.replace`, as `fl_routes._save`).
8. Return `{success, photoId, algoTxId, algoRound, indexerUrl}`.

The rate limit uses the same sliding-window idea as `check_rate_limit`, on its own dict so photo
traffic does not use up a user's model-registration quota.

### 6.2 `POST /api/photo/register-batch` (album)
Body: `{albumTitle, manifests[], powNonce, pk, sig, privacy}`. PoW is over the album root.

**(r2)** The leaves use the repository's canonical encoding, built by the canonical function:
```python
root, leaves, order = compute_layer_merkle({photoId_i: manifestDigest_i, ...}, [photoId_0, ...])
# leaf_i = SHA256(JSON([i, photoId_i, manifestDigest_i]))   (app.py _manifest_leaf)
```
- The browser's existing `bvMerkleLeaf` / `bvFoldProof` verify album proofs unchanged.
- One transaction anchors `{pmr, plc}`.
- Each photo stores its inclusion proof from `merkle_inclusion_proof(leaves, i)`.

**(r4) Album manifest.** The `amd` in the album note is the canonical digest (§5.1) of
```json
{"schema": "photoshield/album@1", "title": "<albumTitle>", "photos": [["<photoId_0>", "<manifestDigest_0>"], ...]}
```
in album order. `POST /api/photo/rehydrate` with `{album, manifests[]}` restores an album after a
restart only if the manifests rebuild the anchored root and count and the album manifest hashes to
the anchored `amd`.

### 6.3 `POST /api/photo/verify` (public, no auth, rate-limited by IP)
Body: `{sha256, phash}` computed in the browser. Decision order:

| Verdict | Condition | Meaning shown to the user |
|---|---|---|
| `EXACT_ORIGINAL` | sha256 = a registered original | This is the registered original |
| `EXACT_CLOAKED` | sha256 = a registered cloaked copy | This is the registered protected copy |
| `DERIVED` | Hamming(phash, any) ≤ τ_near | Very likely a re-encoded or lightly edited copy of a registered photo |
| `SIMILAR` | τ_near < d ≤ τ_far | Possibly derived (heavier edit, crop, or a different but similar photo) |
| `UNKNOWN` | otherwise | Not in the registry |

**Thresholds.** τ_near and τ_far are **never hard-coded**.
- They come from `photoshield/calibration.json` **(r4)**, written by
  `python -m photoshield study e6 --install` (or `PHOTOSHIELD_CALIBRATION=<path>`).
  That run covers JPEG q 95 to 30, resize, crop 5 to 10% and brightness, against unrelated photos,
  and chooses τ for FPR ≤ 1% on unrelated pairs. The file is loaded at startup.
- If the file is missing, near-matching is **disabled**: only the exact and `UNKNOWN` verdicts are
  returned, with `"nearMatch": "uncalibrated"`. There is no fallback number. The same lesson as
  `z > 8` → τ ≈ 23 in BlockVerify.

**(r2) Response privacy.**
- Exact verdicts: return `photoId, ownerUser (if known), ownerPkFp, registeredAt, algoTxId, indexerUrl`.
- `DERIVED`/`SIMILAR`: return `photoId, ownerPkFp, registeredAt, distance, algoTxId, indexerUrl` and
  **no username**. `ownerPkFp` is the first 16 hex of `pk`. The username never reaches the chain
  (§5.3), so withholding it here actually protects it.
- Near-match responses always carry the warning *"Perceptual match is evidence, not proof."*
- The rate limiter is a separate per-IP sliding window (the `_chat_rate_ok` pattern), to slow down
  probing of the registry.

### 6.4 Read routes
- `GET /api/photo/<photoId>`
- `GET /api/photo/owner/<pkOrUser>`
- `GET /api/photo/proof/<photoId>` (album proof + root + txid)
- `GET /api/photo/stats` (counts, plus ledger sync status from §6.8)

**(r4) Ownership proof.** `GET /api/photo/challenge` returns a one-time random challenge that
expires after a few minutes. `POST /api/photo/prove {photoId, challenge, sig}` checks
`sig` = Ed25519 over `photoshield/prove@1|<photoId>|<challenge>` (§5.4) against the key that
registered the photo. With a login it also links that key to the account, which is how a user
recovers "My photos" after a restart (the chain holds keys, never usernames).

**(r4) Re-attach.** `POST /api/photo/rehydrate {manifest}` (auth) restores the pHash, title and
cloak parameters of a hash-only photo, accepted only if the manifest hashes to the anchored `md`.

### 6.5 Certificate
`GET /api/photo/certificate/<photoId>` reuses the visual style of the existing ReportLab certificate
(`generate_certificate`). It contains the hashes, txid, explorer link, `ownerPkFp`, a QR code to
the verify page and the cloak parameters.

### 6.6 Optional `POST /api/photo/cloak`
- Off unless `PHOTOSHIELD_SERVER_CLOAK=1`. Otherwise it returns **501** with
  *"Run `python -m photoshield protect` locally."*
- When on: at most 2 MP, `--size 256`, at most 40 steps, the image deleted after the response and
  never logged.

### 6.7 `algorand_client.broadcast_photo_anchor(note: dict, wait=True)`
- A copy of `broadcast_fl_round`'s structure: balance check, 0-ALGO `PaymentTxn` self-send,
  confirmation loop, `_friendly_algo_error`.
- Serialises with the fixed key order of §5.3 and asserts `len(note_bytes) <= 1024` before sending.
- **(r2) It never calls the contract.**

### 6.8 Ledger rebuild (r2) — `photoshield/chain/ledger_sync.py`

The ledger is the source of truth. The JSON registry is a cache of it plus things the chain does
not hold (username, title).

**Scan**
- Endpoint: `GET {INDEXER}/v2/accounts/{addr}/transactions?note-prefix=<base64('{"ps":1,')>&tx-type=pay&limit=1000&min-round=<cached last_round + 1>`
- Follow `next-token` until the response has none. Page through **fully**; never stop at the
  first page.
- Addresses: the server wallet `algorand_client.ADDRESS`, plus any listed in
  `PHOTOSHIELD_ANCHOR_ADDRESSES` (comma-separated), so anchors made by an older wallet are not lost
  when the wallet changes.

**Accept a transaction only if all hold:**
- `sender == receiver == addr`, amount 0. Without this check, anyone could send a note *to* our
  address and inject fake records.
- The note parses with valid hex lengths.
- `sig` verifies under `pk`.

Each accepted note becomes a registry record with `source: "ledger"`. The chain wins over local
data on any conflict in hashes, `pk` or round.

**Cache**
- `backend/data/photo_ledger_cache.json` holds `{addr: {last_round, entries[]}}`, written
  atomically. Later scans are incremental from `last_round + 1`.
- On Render's free tier the cache dies with the disk, so each boot does one full scan.

**When the scan runs**
- Lazily, in a background thread started on the first photo request, never at import. This keeps
  imports and the backend tests offline and keeps boot fast.
- While it runs, `verify` answers from what is loaded so far and adds `"ledgerSync": "rebuilding"`.

**Failures**
- If the indexer is unreachable or rate-limited: keep serving from cache, report
  `"ledgerSync": "unreachable"` in `/api/photo/stats` and verify responses, and retry with backoff
  (1, 2, 4, … up to 300 s).
- AlgoNode's free indexer limits are measured in E7 rather than assumed.

### 6.9 Hash-only privacy photos and what survives a restart (r2)

`privacy: "hash-only"` registrations put **no pHash on chain** (`po`/`pc` omitted). A 64-bit pHash
leaks a coarse thumbnail-level summary, and some owners will not want that public.

After a restart the rebuilt record has the sha256 values and `md` but no pHash. Such photos
**survive only as exact matches** (`EXACT_ORIGINAL` / `EXACT_CLOAKED`). This is documented in the UI
at registration time and in the README.

**Recovery without trusting anyone: `POST /api/photo/rehydrate` (auth).**
- Body: `{manifest}`. The owner re-attaches their own manifest backup.
- The server recomputes `manifestDigest` and accepts only if it equals the on-chain `md` for that
  photo. The digest commits to the pHash, so a false pHash cannot be smuggled in.
- On success the pHash returns to the registry (`source: "rehydrate"`) until the next restart.

There is no server-side backup copy: the free tier has no persistent disk, and keeping one
elsewhere would put pHashes back in a third party's hands. An external store
(`PHOTOSHIELD_BACKUP_URL`) is recorded in `photoshield/TODO.md` as future work.

### 6.11 Server configuration (r4)

| Variable | Default | Effect |
|---|---|---|
| `PHOTOSHIELD_ANCHOR` | `algorand` | `algorand` (Testnet note), `local` (BlockVerify's in-memory PoW chain) or `mock` |
| `PHOTOSHIELD_LEDGER_SYNC` | `1` | `0` disables the indexer rebuild (tests, offline demos) |
| `PHOTOSHIELD_ANCHOR_ADDRESSES` | the server wallet | Extra comma-separated sender addresses to rebuild from (e.g. a previous wallet) |
| `PHOTOSHIELD_INDEXER` | AlgoNode Testnet indexer | Indexer base URL for the rebuild |
| `PHOTOSHIELD_CALIBRATION` | `photoshield/calibration.json` | Threshold file; missing → near matching disabled |
| `PHOTOSHIELD_SERVER_CLOAK` | `0` | `1` enables `POST /api/photo/cloak` (§6.6) |
| `PHOTOSHIELD_SERVER_MODEL`, `PHOTOSHIELD_SERVER_THREADS` | `sdvae`, `1` | Server cloak encoder and torch threads |

The wallet, App ID and JWT secret come from the existing `ALGO_MNEMONIC`/`ALGO_PRIVATE_KEY`,
`ALGO_APP_ID` and `JWT_SECRET`, so a Render restart keeps the same anchoring address.

### 6.10 Ledger tab labelling (r2)
`get_transaction_ledger` (`algorand_client.py:347`) currently labels every JSON note as a model
"registration" and reads `name`/`hash`, so photo notes would appear as blank model rows.

- One additive branch: `if meta.get("ps"):` set `action = "photo-anchor"` and the photo fields.
  Model notes keep their exact current output.
- FL notes (`frr`) are mislabelled the same way today. That is pre-existing, recorded in
  `photoshield/TODO.md`, and not changed here.

## 7. Perceptual hash: canonical spec (r2, tightened)

Browser and Python **must produce identical bits for identical decoded pixels**, or verify breaks.
The hash is "pHash-style" (Zauner 2010's DCT construction). It is **our canonical variant**, not
byte-compatible with the `imagehash` library, so the report must not claim compatibility.

**What "identical" can mean.** JPEG decoders differ in IDCT method and chroma upsampling. The same
`.jpg` decoded by a browser and by PIL can differ by ±1 in some pixels. Bit-identity is therefore
guaranteed **for identical decoded pixels**, and proven with **PNG** test vectors. For JPEG, the
browser-vs-Python Hamming distance is **measured and reported** (expected 0 to 2 bits) in E6, not
assumed to be 0.

Steps, implemented by hand in both languages (no library resize, no BLAS):

1. **Decode.**
   - Browser: `createImageBitmap(blob, {colorSpaceConversion: "none", premultiplyAlpha: "none"})`,
     drawn at natural size, then `getImageData`.
   - Python: PIL, without applying the ICC profile.
   - **Orientation:** apply EXIF orientation on both sides, so the hash describes what the user
     sees (browser default `imageOrientation: "from-image"`; Python `ImageOps.exif_transpose`).
     P5 verifies this policy behaves the same in Chrome and Firefox. If it does not, the fallback
     is to ignore orientation on both sides, documented.
   - **Minimum size:** width and height ≥ 32 px, otherwise the verdict is `UNHASHABLE`. This avoids
     empty bins in step 3.
2. **Alpha.** Composite onto white in integers: `c' = (c·a + 255·(255 − a) + 127) // 255`
   per channel. Opaque pixels are unchanged.
3. **Grayscale, as an exact integer.** `Y = 299·R + 587·G + 114·B` (luma × 1000, kept as an
   integer). The constant factor does not change any bit.
4. **Resize to 32 × 32 by area averaging.**
   - Bin edges: rows `floor(i·H/32)` to `floor((i+1)·H/32)`, and the same for columns.
   - Sum each bin as an **integer**. This is exact: at 50 MP, a bin sum is about 1.3·10¹⁰, below 2⁵³.
   - Then divide once by the pixel count (one IEEE division, rounded identically in JS and Python).
5. **DCT-II (orthonormal), only the 8 × 8 low-frequency block, computed separably (r3).**
   ```
   g[x][v] = Σ_{y=0..31} C[v][y] · f[x][y]                 (y ascending, accumulator starts at 0.0)
   F[u][v] = (s(u) · s(v)) · Σ_{x=0..31} C[u][x] · g[x][v]  (x ascending, accumulator starts at 0.0)
   ```
   where `x` indexes rows and `y` columns of the 32 × 32 means, `s(0) = √(1/32)`,
   `s(k>0) = √(2/32)`, and `C[k][n] = cos(π(2n+1)k/64)`.
   - `C` and `s` come from a **precomputed table of 17-significant-digit literals**
     (`provenance/dct_table.py` and `PS_DCT_COS`/`PS_DCT_S` in the JS), both emitted by
     `provenance/gen_dct_table.py`. They are never computed with `Math.cos` or `math.cos` at
     runtime, because libm results can differ in the last ulp between engines. A test checks
     that the JS literals equal the Python table.
   - The separable form needs about 10,000 multiplications instead of 65,000. It rounds
     differently from the nested double sum, which is why this exact order is the spec: both
     languages do these operations in this order.
   - No FMA and no numpy matmul in the canonical implementation. Integer luma and integer bin
     sums (steps 3–4) may use numpy, because integer arithmetic below 2⁵³ is exact.
6. **Median and bits.**
   - Take the 63 coefficients excluding `F[0][0]`, sort them, and use `median = sorted[31]`.
   - `bit(u, v) = F[u][v] > median`, with the DC bit fixed at 0.
   - Packing: row-major over `u` then `v`, most significant bit first, into 16 lowercase hex.

**Test vectors** (P1):
- About 20 PNGs under `photoshield/tests/vectors/`: gradients, a checkerboard, noise, an alpha
  image, a non-square image, a 32 × 32 minimum, and a few real photos converted to PNG.
- Expected hex is in `phash_vectors.json`.
- `test_hashing.py` checks Python, and `phash_js_test.mjs` loads **the shipped JS function from
  `frontend/index.html`** (the `fedverify/tests/test_chain.py` extraction pattern) and checks the
  same vectors under Node.

## 8. Cloak engine details

- **PGD:**
  - Random start in [−ε/10, ε/10], step α = ε/10 (configurable), `sign(grad)`, project to the
    ε-ball, clamp to [0, 1].
  - Untargeted: maximise `MSE(enc(x+δ), enc(x))`.
  - Targeted: minimise the distance to a decoy's latent (the PhotoGuard encoder attack, Salman et
    al., ICML 2023).
- **Budget units:** ε is always reported as x/255 on [0, 1] images. PhotoGuard's published ε = 0.1
  is converted when quoted.
- **EOT:** each step samples one of {identity, resize 0.7 to 1.0 and back, Gaussian blur σ 0 to 1,
  differentiable JPEG}, seeded. Differentiable JPEG is a real implementation choice made in P2
  (DiffJPEG-style or straight-through quantisation), recorded in the run config.
- **Determinism (CLAUDE.md, A1, A5):**
  - `--seed` seeds random, numpy and torch. `torch.use_deterministic_algorithms(True)`.
  - `--threads` is pinned and recorded.
  - Each run writes `photoshield/results/<exp>/<cell>/seed<N>/config.json` + `records.jsonl`
    **(r4)** (gitignored; `make_tables.py` renders them, CLAUDE.md "Numbers").
- **CPU budget:** sdvae at 256 px is *estimated* at a few seconds per step on a laptop. That is
  measured in P2 and reported, not assumed.
- **Where real runs happen:** the cloud container cannot download the SD VAE (CLAUDE.md A15).
  Container tests use the `tiny` random-weight encoder; runs with real weights happen on the laptop
  or Colab.
- **Output:** always PNG, EXIF stripped.

## 9. Frontend: "Photo Shield" tab

These changes are additive:
- One sidebar button styled like the Federated one.
- One `tab-photoshield` div.
- `'photoshield'` appended to `_TAB_IDS` **(r2)**, because `switchTab` returns early for ids not in
  that list.
- One line in `switchTab`: `if (name === 'photoshield' && typeof loadPhotoShield === 'function') loadPhotoShield();`.

### 9.1 Sections
1. **Protect.** Explains the local CLI with a copyable command. A drop zone takes `manifest.json`
   plus the two image files. The browser re-hashes both images and shows ✔/✘ against the manifest
   before allowing register.
2. **Register.** Title, privacy choice (full / hash-only, with the §6.9 consequence spelled out),
   signing with the owner key, PoW progress (the existing `minePoW`), and a result card with txid
   and explorer link.
3. **Verify.**
   - Drop any image. sha256 and pHash are computed locally.
   - A verdict badge in 5 colours, plus a Hamming distance bar.
   - Then **"Check on public ledger"**: the browser fetches the note from
     `testnet-idx.algonode.cloud` itself, compares the fields and verifies the owner signature in
     the browser.
4. **My photos.**
   - A table with txid, privacy mode and a certificate button.
   - Thumbnails are rendered locally, only if the user re-drops the files (the server has no pixels).
   - A "Re-attach manifest" button calls `/rehydrate` (§6.9).
5. **Honest limits.** An always-visible card (§12).
6. **Tour and FAQ.** New guided-tour steps and a FAQ entry, the same pattern as the Federated tab.

### 9.2 The owner key (r2)

**Generation.**
- `crypto.subtle.generateKey({name: "Ed25519"}, true, ["sign", "verify"])`. It is **extractable**,
  so it can be backed up.
- If the browser lacks WebCrypto Ed25519, use a pinned `tweetnacl` from cdnjs.
- It needs a secure context (https or localhost), like the existing `crypto.subtle` code.

**Storage.**
- IndexedDB (`photoshield-keys` store), holding the `CryptoKey` objects.
- Keys are per browser and **per origin**: `localhost:8080` and the Render site have separate
  stores. Export/import moves a key between them.

**Export / backup.**
- Downloads `photoshield-key-<pkFp>.json` containing
  `{schema, alg: "Ed25519", publicKeyHex, encryptedPrivateKey, kdf, createdAt}`.
- The private key (PKCS#8) is encrypted with AES-GCM under a key derived from a user passphrase:
  PBKDF2-SHA-256 with 600,000 iterations and a random salt, both stored in the file.
- An unencrypted export is possible only behind an explicit "I understand" checkbox.

**Import.**
- File → passphrase → decrypt → import.
- Check that the derived public key equals `publicKeyHex`, then store it.

**Warnings.**
- A persistent banner until a backup has been exported: *"Your signing key exists only in this
  browser. Clearing site data, a private window or another device loses it. Export a backup now."*
- A confirm dialog before the first registration if no backup exists yet.

**What losing the key means.**
- Existing anchors stay on chain and remain verifiable as existence proofs.
- But the owner can no longer sign §5.4's `prove@1` challenge, so they cannot show *control* of the
  key that registered them. The UI explains this.

**Linking to a username.**
- On register, the server stores `pk ↔ ownerUser`.
- After a restart the link is gone (ledger records have `pk` only). It is re-established when the
  user logs in and their browser signs a `prove@1` challenge for that `pk`.

**Not in v1:** key rotation and revocation (`photoshield/TODO.md`).

## 10. Experiments (what goes in the report)

All numbers come from results files via `photoshield/analysis/make_tables.py`. Nothing is typed by
hand.

| Exp | Question | Setup | Metrics |
|---|---|---|---|
| E1 invisibility vs strength | How much ε before people notice? | 10 own photos + 10 non-face, ε ∈ {2,4,8,12,16}/255, sdvae | PSNR, SSIM, LPIPS (opt) |
| E2 encoder disruption (CPU) | Does the cloak break the SD VAE? | VAE encode→decode, original vs cloaked | latent shift, recon PSNR/SSIM drop |
| E3 robustness | Does it survive sharing? | JPEG q 95/75/50, resize 50%, blur σ 1, a simple denoise; **uncloaked + same processing as baseline** | E2 metrics after processing |
| E4 real editor (Colab GPU) | Does InstructPix2Pix degrade? | 5 prompts × fixed seeds {0..4}, original vs cloaked | CLIP-directional score, LPIPS(edit, original-edit), side-by-side grid |
| E5 transfer | Does an sdvae cloak hurt other models? | The same images through resnet50 features and a different VAE | feature shift |
| E6 pHash calibration | Can we recognise shared copies? Does a purified copy still match? | Positives: transformed and purified copies. Negatives: unrelated pairs. Plus the browser-vs-Python JPEG decode gap (§7) | ROC, τ at FPR ≤ 1%, TPR per transform, cross-decoder Hamming |
| E7 chain cost | What does provenance cost? | single vs album (N = 1, 10, 100), mock/local/algorand; full ledger rebuild time | latency ms, bytes, fee, rebuild time and pages |

**(r4) How each experiment is run.**

| Exp | Command | Where |
|---|---|---|
| E1, E2 | `python -m photoshield study e1 --model sdvae` | laptop (SD-VAE download) |
| E3 | `python -m photoshield study e3 --model sdvae --eps 8` | laptop |
| E4, E5 | `photoshield/experiments/colab_edit_test.ipynb` | Colab GPU |
| E6 | `python -m photoshield study e6 --install`, plus `e6p` (cloaked copies) and `e6-browser` (Chromium decode) | anywhere |
| E7 | `python -m photoshield study e7` (mock/local; the live Algorand latency is measured on the laptop) | anywhere |
| tables | `python -m photoshield.analysis.make_tables` → `photoshield/results/TABLES.md` | anywhere |

E6 uses scikit-image's bundled photos, except `cat` (the same photo as `chelsea`) and `horse` (a
binary silhouette with no stable pHash). The cloud container runs E1–E3 only with the random-weight
`tiny` encoder; `make_tables` marks those rows *pipeline check*, never findings.

E6 answers the central claim from the literature review: **the provenance layer still delivers
value after the cloak is broken.**

Verdict rule (as in FedVerify): claim "the cloak degrades edits" only where the paired difference
over seeds and images has a bootstrap 95% CI that excludes 0.

## 11. Tests

Regression gates (CLAUDE.md A14): `python3 -m pytest backend/tests/ -q` stays at ≥ 99, and the
fedverify suite stays at its P0 count.

- **`test_pgd.py`**
  - ‖δ‖∞ ≤ ε + 1e-6, shape preserved, values in [0, 1].
  - Same seed and threads → identical PNG bytes.
  - Latent shift > random noise of equal ε (`tiny` encoder, no downloads).
- **`test_hashing.py`**
  - PNG vectors, the minimum-size rejection, alpha compositing.
  - A Python-loops vs numpy-variant equality guard (if a numpy variant exists).
  - sha256 matches `hashlib`.
- **`test_manifest.py`**
  - The canonical digest is stable under key reordering and whitespace.
  - Rejects bad hex, missing fields and an oversize title.
- **`test_signing.py`**
  - Valid signature accepted.
  - A flipped byte in the message or signature is rejected.
  - Domain separation: a `register@1` signature does not verify as `album@1`.
- **`test_anchor.py`**
  - Mock txids deterministic.
  - The local backend mines a block whose transaction contains the note.
  - Note ≤ 1024 B and starts with `{"ps":1,`.
  - Album proofs verify for every leaf, fail for a flipped leaf, and use `_manifest_leaf`'s encoding.
- **`test_ledger_sync.py`** (mocked indexer)
  - Follows `next-token` across ≥ 3 pages.
  - Ignores foreign senders, bad signatures and malformed notes.
  - The incremental scan starts at `last_round + 1`.
  - Hash-only notes rebuild as exact-only.
  - An unreachable indexer → cached data + `"unreachable"`.
- **`test_photo_routes.py`** (Algorand mocked, `test_api.py` style)
  - Register happy path.
  - Missing or invalid PoW → 400; bad signature → 400; rate limit → 429.
  - Duplicate by another key → 409; duplicate by the same key → idempotent.
  - Verify returns each of the 5 verdicts, plus `"uncalibrated"` with no calibration file.
  - A near match carries no username.
  - Rehydrate accepts a matching manifest and rejects a doctored pHash.
  - Server cloak route → 501 by default; an XSS-like title is rejected.
- **Guards**
  - The existing `/api/register` response shape is unchanged after the Blueprint is added.
  - `get_transaction_ledger` output for model notes is unchanged.

## 12. Honest limits (UI, README and report)

- The cloak targets **one** encoder. It does not reliably transfer, and gives no protection against
  closed models (Grok, Gemini, Midjourney).
- Re-encoding, resizing, denoising and diffusion purification (e.g. DiffPure) weaken or remove it.
- Provenance proves *existence at a time, under a key*. It does not stop a fake from spreading, and
  platforms do not check it automatically.
- A perceptual match is **evidence, not proof**. Similar photos can collide, heavy edits or crops
  evade it, and an adversary can deliberately evade pHash (Jain et al., USENIX Security 2022).
- First-to-register is not true authorship. Someone could register your public photo first.
  Register before posting; the UI says so.
- Losing the owner key loses the ability to prove control of past registrations (§9.2). Back it up.
- Hash-only photos survive a server restart only as exact matches unless the manifest is
  re-attached (§6.9).
- Defensive use on photos you own, only.

## 13. Build phases

One phase at a time. Each ends with a completion block, a commit as `Shubhangam-Singh` with no
trailer (CLAUDE.md A12), and a stop.

| Phase | Deliverable | Done when |
|---|---|---|
| P0 | Read the repo + CLAUDE.md, restore CLAUDE.md, write `photoshield/INVENTORY.md`, fold the review into this document | The inventory lists exact import paths; the baselines are measured and recorded |
| P1 | `provenance/` (sha256, pHash per §7 + DCT table, manifest, signing, Merkle wrapper) + PNG vectors | Python vectors pass; signing tests pass |
| P2 | `cloak/` (encoders incl. `tiny`, PGD, EOT, metrics) + CLI `protect`/`hash` | `test_pgd.py` passes. The real `me.jpg` run is done on the laptop, with numbers printed |
| P3 | `chain/anchor.py` + `broadcast_photo_anchor` + `chain/ledger_sync.py` | mock/local/sync tests pass; note-size and prefix asserts |
| P4 | `backend/photo_routes.py` + the 2-line `app.py` registration + the ledger-tab branch | Route tests pass, backend ≥ 99, fedverify at baseline |
| P5 | Frontend tab, JS pHash, key management, Node vector test | JS vectors byte-match Python; manual browser check incl. export/import |
| P6 | `experiments/study.py` E1–E3, E6, E7 + `make_tables.py` | Results files and generated tables, none typed |
| P7 | Colab notebook E4/E5 | Notebook + tested `edit_test.py`; grids and metrics come back from a Colab run **(r4)** |
| P8 | Docs: WHAT_IS_PHOTOSHIELD, THREAT_MODEL, API_REFERENCE section, README section | Links checked |

## 14. Positioning

- **Report/paper:** a third application of BlockVerify's integrity layer (models → federated
  updates → personal media), sharing one anchoring core. Keep it a separate chapter or a short
  follow-up paper; do not mix it into the FedVerify LNCS paper.
- **Capstone candidate:** E4 + E6 + the honest-limits analysis is a solid final-year scope.
- **Viva one-liner:** "BlockVerify proves a file wasn't changed. PhotoShield extends that to photos,
  which *are* always changed by sharing, so we add perceptual matching for provenance and
  adversarial cloaking for partial prevention, and we measure both honestly."
- **Novelty (from the literature map):** integration and evaluation, not a new attack. To our
  knowledge, no prior work combines cloaking, perceptual hashing and public-ledger anchoring. The
  closest neighbours cover two of the three (Mohit et al. 2026, ARCHANGEL 2019, Dual Defense 2024,
  DualShield 2026). Several of those are unreviewed preprints, so the claim is worded "to our
  knowledge".
