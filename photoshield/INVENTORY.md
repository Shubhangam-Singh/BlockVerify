# PhotoShield: P0 inventory

The ground truth PhotoShield builds on, established by reading the code at P0 (2 Nov 2026). Every
claim carries a `file:line` reference. Where this file and `PHOTOSHIELD_DESIGN.md` disagree, the
code wins, and the design is corrected.

## 1. Repository state at P0

| Item | Value |
|---|---|
| Branch | `ss2005/modest-fermat-hflkcl` |
| HEAD at the start of P0 | `88f491daca38a9cec3845f3d4b1aef73dd21b899` ("deploy: point frontend at the new Render API and update render.yaml") |
| Commits above the design spec's stated base `8932448` | `47b3d9f` (final updates, the one that deleted CLAUDE.md), `a39c22a` (env-based wallet/app id/JWT secret for Render), `88f491d` (frontend API URL + render.yaml) |
| Git identity | `Shubhangam-Singh <shubhangam2005singh@gmail.com>`, local and global (CLAUDE.md A12) |
| Python in the container | 3.13 (system). Tests run in a throwaway venv (`/tmp/claude-0/bvenv`) |

### Test baselines (CLAUDE.md A14)

| Suite | Command | Result at P0 |
|---|---|---|
| Backend | `python3 -m pytest backend/tests/ -q` | **99 passed** (90 in `test_api.py` + 9 in `test_blockchain.py`) |
| FedVerify | `python3 -m pytest fedverify/tests/ -q` | **299 passed, 134 skipped, 0 failed** (433 collected) |

Notes on the FedVerify run are in §9.

## 2. Why CLAUDE.md was missing

- `CLAUDE.md` existed through `8932448`. It was **deleted in `47b3d9f`** ("final updates
  blockverify+fedverify", 2 Nov 2026 10:18 +0530), a 105-file commit.
- `fedverify/TODO.md` (Revision 2, Phase 6) had already flagged that "`CLAUDE.md` is deleted in the
  working tree; restore it with `git restore CLAUDE.md` if that was not intended".
- The most likely sequence is that a working-tree deletion was swept into the bulk commit. That is
  inferred from the commit's contents; Git does not record intent.
- Restored in P0 from `git show 8932448:CLAUDE.md`, byte-identical to the last committed version,
  with amendments A12 to A15 appended. A12 replaces the "never commit" rule with the human's newer
  instruction.

## 3. Reuse map: backend (`backend/`)

`backend/` is not a package. Import by inserting the directory on `sys.path` and using flat imports,
the convention of `fedverify/chain/commitment.py:35-43` and `backend/tests/test_api.py`.

```python
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "backend"))
import app as _app        # lazily, inside functions, never at import time of a Blueprint
```

### 3.1 `backend/app.py`

| Symbol | Line | Use in PhotoShield |
|---|---|---|
| `app = Flask(__name__)` | 35 | `photo_bp` registered next to the others |
| `app.register_blueprint(auth_bp)` / `(fl_bp)` | 37, 38 | the 2-line pattern for `photo_bp` |
| `app.json.sort_keys = False` | 43 | load-bearing: JSON key order survives in responses |
| `DATA_DIR` | 70 | `photo_registry.json`, `photo_ledger_cache.json` live here |
| `check_rate_limit(owner)`, `RATE_LIMIT`, `RATE_WINDOW_SEC` | 168, 164, 165 | the pattern only. Photo routes keep their **own** dict so they don't use up model quota |
| `POW_DIFFICULTY`, `POW_PREFIX`, `verify_pow(model_hash, nonce)` | 204, 205, 207 | photo PoW over `original.sha256` (single) or the album root |
| `generate_certificate(model_id)` | 1366 | visual style reused by the photo certificate (ReportLab, already in requirements) |
| `_manifest_leaf(index, name, layer_hash)` | 1562 | **the** canonical leaf: `sha256(json.dumps([i, name, h], separators=(",",":"), ensure_ascii=False))` |
| `_manifest_order` | 1568 | explicit order first |
| `compute_layer_merkle(layer_hashes, layer_order)` | 1575 | album root: `({photoId: manifestDigest}, [photoIds])` |
| `merkle_inclusion_proof(leaves, index)` | 1590 | per-photo album proof |
| `CHAT_RATE_MAX`, `_chat_rate_ok(ip)` | 1801, 1805 | the per-IP limiter pattern for public `/api/photo/verify` |

**Circular import.** `app.py` imports `fl_routes` at line 25, and `fl_routes` defers its import of
the commitment module (and hence `app`) to request time (`fl_routes.py:34`). `photo_routes.py` must
do the same.

### 3.2 `backend/algorand_client.py`

| Symbol | Line | Use |
|---|---|---|
| `ALGOD_ADDRESS`, `INDEXER_ADDRESS` | 22, 23 | indexer base URL for the ledger rebuild |
| `client` | 26 | algod client for `broadcast_photo_anchor` |
| `_friendly_algo_error(exc)` | 36 | the same error mapping |
| `_wallet_from_env()`, `get_or_create_wallet()` | 52, 70 | wallet from `ALGO_MNEMONIC`/`ALGO_PRIVATE_KEY` (added in `a39c22a`), else the file |
| `ADDRESS, PRIVATE_KEY` | 92 | sender and receiver of photo notes; the address scanned on rebuild |
| `broadcast_hash_to_algorand(...)` | 123 | **not reused**: it calls the contract |
| `get_transaction_ledger(limit)` | 347 | the Ledger tab. Labels every JSON note "registration" (lines 377-388); gets the additive `ps` branch (design §6.10) |
| `broadcast_fl_round(...)` | 419 | **the template** for `broadcast_photo_anchor` (balance check, 0-ALGO self-pay, confirm loop, envelope) |

Existing note namespaces, which photo keys must not collide with:
- model notes: `id`, `name`, `hash`, `owner`, `lmr`, `lc` (`algorand_client.py:152-160`)
- FL notes: `frr`, `flc`, `rnd`, `run` (`broadcast_fl_round`)

The photo keys (`ps`, `id`, `o`, `c`, `po`, `pc`, `md`, `pk`, `sig`, `pmr`, `plc`, `amd`) only
share `id` with model notes. The leading `"ps":1` tells them apart, and a model note never starts
with `{"ps":1,`.

### 3.3 `backend/contract.py`

| Symbol | Line | Note |
|---|---|---|
| `GLOBAL_SCHEMA = StateSchema(num_uints=0, num_byte_slices=64)` | 42 | 64 slots in total, shared with model registrations (up to 2 each: `id` and `id:l`). **The reason photos are note-only.** |
| `load_app_id()` | 63 | reads `ALGO_APP_ID` first (added in `a39c22a`). Not used by PhotoShield |
| `call_contract_register` | 124 | not used by PhotoShield |

### 3.4 `backend/auth.py`

| Symbol | Line | Use |
|---|---|---|
| `require_auth` (sets `g.user`) | 99 | register, register-batch, rehydrate |
| `decode_token(token)` | 91 | optional identity on public routes |
| `_load_secret()` | 45 | reads `JWT_SECRET` first (added in `a39c22a`) |

A pre-existing bug, out of PhotoShield scope (recorded in `photoshield/TODO.md`): `app.py`'s
`search_models` and `public_registry` read `payload.get("username")` (`app.py:736`, `:1156`), but tokens carry the user in
`sub` (`auth.py:79-84`), so owners never see their own private models in those lists.
`photo_routes.py` must read `sub`.

### 3.5 `backend/fl_routes.py`: the Blueprint template

| Symbol | Line | Pattern to copy |
|---|---|---|
| `fl_bp = Blueprint("fl", __name__)` | 25 | `photo_bp = Blueprint("photo", __name__)`, full paths, no `url_prefix` |
| `DATA_DIR`, `FL_FILE` | 27, 28 | `PHOTO_FILE` in the same directory |
| `_commitment()` (lazy import) | 34 | lazy import of `app` |
| `_load()` / `_save()` (atomic tmp + `os.replace`) | 48, 58 | registry persistence |
| `_err(msg, code)` | 65 | the error envelope |
| `RUN_ID_RE` | 70 | the charset-check idea for titles/ids |
| `INDEXER`, `_indexer_url(txid)` | 111, 114 | `indexerUrl` only for real 52-char txids |

### 3.6 Dependencies already present

- **PyNaCl 1.6.2** is installed as a requirement of `py-algorand-sdk` (`pip show py-algorand-sdk`:
  `Requires: msgpack, pycryptodomex, pynacl`). Ed25519 verification on the backend can use
  `nacl.signing.VerifyKey` with **no change to `backend/requirements.txt`**.
- `cryptography` (used by `fedverify/chain/signing.py`) is **not** a backend requirement. Do not
  import it from `photo_routes.py`.
- `numpy` and `fpdf2` were added to `backend/requirements.txt` in `a39c22a`.

## 4. Reuse map: FedVerify (`fedverify/`)

| Symbol | File:line | Use |
|---|---|---|
| `_backend()` lazy import of `app`, `warm_up()` | `fedverify/chain/commitment.py:35`, `:46` | the import pattern for `photoshield/provenance/merkle.py` |
| `leaf_hashes`, `round_merkle_root`, `inclusion_proof`, `verify_proof` | `fedverify/chain/commitment.py:94`, `:100`, `:111`, `:119` | `verify_proof` is the Python verifier; the others are thin wrappers worth mirroring |
| `ChainAnchor` (mock/local/algorand), `_mock_txid` | `fedverify/chain/anchor.py:53`, `:48` | `PhotoAnchor` mirrors it. `local` mines into `backend/blockchain.py` |
| `signing_bytes` (canonical JSON: sorted keys, compact, UTF-8) | `fedverify/chain/signing.py:55` | the canonical-JSON convention for `manifestDigest` |
| `content_id`, `verify_cid` | `fedverify/chain/contentid.py:30`, `:46` | optional CID of the cloaked PNG (not in the v1 note) |
| `_extract_js(*names)` | `fedverify/tests/test_chain.py:261` | the pattern for testing the **shipped** JS pHash under Node |

`fedverify/chain/signing.py` uses `cryptography` and derives TEST keys from a seed. PhotoShield keys
are real user keys generated in the browser, so only the canonical-JSON convention carries over.

## 5. Reuse map: frontend (`frontend/index.html`)

| Symbol | Line | Use |
|---|---|---|
| `const API = …` | 3035 | already points at the Render API (`88f491d`) |
| `_TAB_IDS` | 3247 | **append `'photoshield'`**, or `switchTab` returns early (3250) |
| `switchTab(name, btn)` | 3249 | +1 line, like the Federated one at 3269 |
| `sha256Fallback(buffer)` | 3359 | pure-JS SHA-256 for non-secure contexts |
| `computeHash(file)` | 3369 | SHA-256 of a `File` (WebCrypto, worker fallback) |
| `minePoW(modelHash)` | 3554 | the Hashcash miner. It writes to the Register tab's PoW terminal (`showPowTerminal`), so the photo tab needs its own progress UI or a parameterised copy |
| `bvSha256Hex`, `bvMerkleLeaf`, `bvFoldProof` | 3879, 3893, 3894 | album proofs verify with these unchanged |
| `bvVerifyManifestAnchor` | 3899 | the template for "Check on public ledger" (indexer read in the browser) |
| `escapeHtml(v)` | 3927 | every user string (titles) rendered through it |
| `flVerifyProof` | 4818 | the 4-step trustless verification UI pattern |
| `hashFilesParallel` (Web-Worker pool) | 6588 | optional: hash many photos in parallel |
| `TOUR_STEPS` | 7723 | append Photo Shield steps (Federated steps at 7730-7734) |
| Sidebar Federated button | 2062 | style template for the new button |

No pHash, DCT or Ed25519 code exists in the frontend today. All of that is new in P5.

## 6. Reuse map: evaluation (`evaluation/eval_lib.py`)

| Symbol | Line | Use |
|---|---|---|
| `roc_curve`, `auc`, `rates_at` | 154, 177, 181 | E6 pHash calibration: ROC over Hamming distances (no sklearn needed) |

## 7. Constraints found in the code

1. **Contract slots.** 64 global byte slices in total (`contract.py:42`), so photos are note-only.
2. **Note size.** Algorand caps a note at 1024 bytes. Model notes stay far below that; photo notes
   are 503 B at most (album: 373 B; design §5.3).
3. **Ephemeral disk on Render.** Everything under `backend/data/` is lost on restart, hence the
   ledger rebuild (design §6.8).
4. **One server wallet signs every transaction** (`algorand_client.py:92`). That is why owner keys
   sign the claim (design §9.2).
5. **The Ledger tab parser** labels any JSON note as a model registration
   (`algorand_client.py:377-388`). FL notes (`frr`) are already mislabelled. That is pre-existing
   and recorded in `photoshield/TODO.md`.
6. **Secure context.** `crypto.subtle` (SHA-256, Ed25519, AES-GCM) exists only on https or
   localhost. The Render site is https. `http://0.0.0.0:8080` is not (see the comment at
   `index.html:3881`).

## 8. Network reach of the cloud container (CLAUDE.md A15)

Checked at P0 with `curl` through the session proxy:

| Host | Reachable | Consequence |
|---|---|---|
| `pypi.org` | yes | Python dependencies install from PyPI |
| `download.pytorch.org` | **no** | torch comes from plain PyPI inside the container venv (A2 still applies on the laptop) |
| `huggingface.co` | **no** | SD-VAE weights can't be downloaded here. Real-weight runs happen on the laptop or Colab |
| `testnet-idx.algonode.cloud` | **no** | the ledger rebuild is tested against a mocked indexer here; a live run happens on the laptop or on Render |
| `*.onrender.com` | **no** | the deployed site is checked from a browser, not from the container |

## 9. FedVerify baseline notes

Run in the container venv with torch 2.15.0, opacus 1.6.0, lifelines 0.30.3 and xgboost-cpu 3.4.1
from PyPI (CLAUDE.md A15), on Python 3.13, in 73 s. **0 failures.**

- 433 collected + 99 backend = 532, which matches the "532 tests" of the final report and
  `fedverify/TODO.md` Phase 12.
- **The 134 skips are not failures.** Every one is a test that needs a gitignored input this fresh
  clone does not have.

| Skips | Reason (pytest's own message) |
|---|---|
| 41 + 11 + 5 | heart CSV (and, for 5, the calibration file) not present |
| 38 + 10 | TCGA-BRCA workbook not present |
| 9 + 3 + 3 + 2 + 2 + 1 + 1 + 1 | `fedverify/results/` artefacts not built (results, Phase 9, diagnostics, report pack, MANIFEST, demo runs) |
| 6 | MIT-BIH cache not built |
| 1 | no TeX engine on PATH |

**The regression gate for PhotoShield phases** is therefore "fedverify: 0 failed, at least 299
passed in this container". On the human's laptop, which has the data and results, the 433 run
without skips, as in the report.

Reproduce the per-reason breakdown with:
`python3 -m pytest fedverify/tests -q -rs`.
