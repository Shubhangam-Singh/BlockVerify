# FedVerify — Phase 0 Inventory of the existing BlockVerify system

Ground truth established by reading the real code. Every claim carries a `file:line`
reference. Written because the master prompt's §0 is derived from documentation and
**contradicts the code in several places**; per its own instruction, the code wins.

Verified on branch `main` with a dirty working tree (see §13).

---

## 1. Discrepancies vs. the master prompt — FOLLOW THE CODE

| Master prompt §0 says | Reality | Consequence |
|---|---|---|
| Algorand App ID `758544892` | **`764828342`** — `backend/data/algo_app.json`; loaded by `load_app_id()` `backend/contract.py:63` | Phase 3 must read the App ID from `contract.load_app_id()`, never hardcode either value |
| PoW difficulty 4 | Class default **is** 4 (`backend/blockchain.py:112`) but the running code uses **2** (`backend/app.py:130`) | Any Phase-3 chain write inherits difficulty 2 |
| "82 passing pytest cases" | **99 passing, 0 skipped** | Regression baseline is 99, not 82 (see §9). `docs/TAMPER_DETECTION.md:234` is stale |
| "the existing Merkle construction" (one) | **Five** distinct implementations with **three different leaf encodings** | Phase 3 must reuse exactly the one in §2.1 |
| `app.py` routes (implied bare) | **Mixed**: one Blueprint already registered + 37 bare `@app.route` | Phase 3's Blueprint plan matches existing practice |
| `backend/blockchain.py` "is_chain_valid()" returns validity | Returns a **dict** `{"valid": bool, "errors": [...]}` (`blockchain.py:177`) | Do not treat as a bool |

Confirmed correct in the prompt: Bitcoin testnet wallet `mrDTrvKrLpW969E8CbqagN8KRRJ3u49huZ`
(`backend/bitcoin_client.py:30`); Algorand note keys `"lmr"` / `"lc"`
(`backend/algorand_client.py:138-139`); the §2.4 Merkle spec matches §2.1 below.

---

## 2. Merkle implementations — there are FIVE

### 2.1 CANONICAL — layer-manifest Merkle (THE one Phase 3 must reuse)
`backend/app.py`:

| Function | Line | Notes |
|---|---|---|
| `_manifest_leaf(index, name, layer_hash) -> str` | **1560** | `sha256(json.dumps([index,name,layer_hash], separators=(",",":"), ensure_ascii=False).encode("utf-8")).hexdigest()` |
| `_manifest_order(layer_hashes, layer_order) -> list` | **1566** | explicit order first, stragglers keep key order |
| `compute_layer_merkle(layer_hashes, layer_order=None)` | **1573** | returns `(root, leaves, order)`; `(None, [], [])` if empty |
| `merkle_inclusion_proof(leaves, index) -> list` | **1588** | `[{"hash": hex, "right": bool}, ...]` |

- Parent rule: `sha256((left + right).encode()).hexdigest()` over **hex strings**, `app.py:1582`.
- Odd level: duplicate-last, `app.py:1580`. Ambiguity (CVE-2012-2459 class) mitigated by anchoring the leaf count.
- Callers: registration `app.py:242-245` (stored as `layerMerkleRoot`, `app.py:270`);
  `GET /api/manifest-proof/<model_id>` → `manifest_proof()` `app.py:586-630`.
- **JS counterpart is byte-identical**: `bvMerkleLeaf` `frontend/index.html:3073`,
  `bvFoldProof` `:3074`, `bvSha256Hex` `:3068`, verifier `bvVerifyManifestAnchor` `:3079`.

### 2.2–2.5 DO NOT USE for FL commitments (different leaf encodings)
- `_tx_hash` `app.py:1602` + `_build_merkle` `app.py:1609` — block-transaction tree for D3; leaf = `sha256(json.dumps(tx, sort_keys=True))`. Route `GET /api/block/<i>/merkle` `app.py:1655`.
- `/api/algo/merkle` `app.py:1017-1087` — **leaves are raw `modelHash` hex, not pre-hashed**; duplicate-last leaf labelled `" (dup)"` `app.py:1060`.
- Inline twin of the above inside `bitcoin_anchor()` `app.py:1737-1760` (copy-paste, not a call).
- Fifth inline copy in `evaluation/evaluate.py:168-176`, written to avoid importing Flask.

---

## 3. Algorand

- Module `backend/algorand_client.py`; submitter
  `broadcast_hash_to_algorand(model_id, model_name, model_hash, owner, layer_root=None, layer_count=0)` **:102-192**.
- 0-ALGO self-`PaymentTxn` (sender == receiver), signed, polled up to 15×1 s **:149-155**.
- SDK: `algosdk` (`from algosdk.v2client import algod`, `from algosdk.transaction import PaymentTxn`) **:16-18**.
  Pinned `py-algorand-sdk==2.7.0` (`backend/requirements.txt:3`) but **installed is 2.11.1** — version drift.
- Note payload **:133-141**: always `id`, `name`, `hash`, `owner`; plus `lmr` + `lc` when a layer root is supplied.
- App ID **764828342** in `backend/data/algo_app.json`; `save_app_id`/`load_app_id` `backend/contract.py:57-71`;
  lazy deploy `_get_or_deploy_contract()` `algorand_client.py:76-98`.
- Endpoints: algod `https://testnet-api.algonode.cloud` **:22**, indexer `https://testnet-idx.algonode.cloud` **:23**.
- Wallet `backend/data/algo_wallet.json` (**private key stored in plaintext**; gitignored but on disk).

## 4. Bitcoin OP_RETURN

- Module `backend/bitcoin_client.py`. Library `bit` (`PrivateKeyTestnet`) **:19**, behind a
  `try/except ImportError` setting `BIT_AVAILABLE` **:18-24**; guards at **:39-40, :46-47**.
  `bit` 0.8.0 is installed but is **absent from requirements.txt**.
- Functions: `_get_key()` :38 · `get_wallet_balance()` :44 · `build_op_return_script_hex(root)` :62
  (returns `"6a20"+hex`) · `anchor_merkle_root(root)` :71 · `decode_op_return_annotated()` :141.
- Routes: `GET /api/bitcoin/wallet` `app.py:1718`; `POST /api/bitcoin/anchor` `app.py:1725` (`@require_auth`).
- `BTC_WIF` env-overridable, default at **:31**; address `mrDTrvKrLpW969E8CbqagN8KRRJ3u49huZ` **:30**. Fee hardcoded 2000 sat **:107**.
- Phase 3 must call this existing path; **no new signing code**.

## 5. Python port of the forensic probes

`evaluation/eval_lib.py` (docstring :1-11 states it is a faithful port):

| Function | Line | Returns | JS original (`frontend/index.html`) |
|---|---|---|---|
| `_flat(arr)` | 58 | finite float64 `ravel(order="C")` | `bvFlatten` :2963 |
| `bv_numeric_stats(arr)` | 64 | `{n,min,max,mean,std}` or `None`; population std | `bvNumericStats` :2969 |
| `bv_find_outliers(arr, z_thresh=8.0)` | 75 | `{n,count,max_z,scale,median}` (no `median` when `n<4`) | `bvFindOutliers` :2979 |
| `bv_layer_health(arr)` | 92 | `{n,nan,inf,extremes,max_abs,entropy,max_run}`; `entropy=None` if `n<=16` | `bvLayerHealth` :3026 |

- median = `sorted[n//2]`; scale = `1.4826·MAD` → std → 1.0 (:83-85); entropy = 32-bin Shannon ÷ `log2(32)=5` (:103-109); extremes = `|w|>100`.
- **`z_thresh` is already a parameter (default 8.0)** — Phase 4 passes a calibrated τ without editing existing code. The JS `bvFindOutliers` still hardcodes `z>8`; leave it (existing behaviour).
- No Python port of `bvClassifyTopology` (`index.html:2999`) or `bvHealthFlags` (`:3047`, thresholds entropy<0.35, maxRun>=8).
- Cross-validation precedent: `evaluation/cross_validate.py` string-extracts the JS out of `index.html` and runs it under Node to assert float64 equality — mirror this style for any new port (Phase 3/4).

## 6. Chain transaction type strings

Only three ever reach the chain:
- `"genesis"` — `backend/blockchain.py:125`
- `"register"` — `backend/app.py:133`
- `"verify"` — `backend/app.py:144` (carries `"result": "valid"|"invalid"`)

Constructed **only** inside `_get_pow_chain()` `app.py:123-152`. The `register`/`version`/`verify`
strings at `app.py:909/922/939` are `/api/activity-feed` UI events, **not** chain transactions —
there is no `"version"` transaction type on the chain.

Phase 3 adds `"fl_client_update"` and `"fl_round_commit"` (additive).

## 7. Flask route registration — mixed

- `app = Flask(__name__)` `app.py:34`; `CORS(...)` `:35`.
- **A Blueprint already exists**: `auth_bp = Blueprint("auth", __name__)` `backend/auth.py:28`,
  registered `app.register_blueprint(auth_bp)` `app.py:36`, **no `url_prefix`** — its routes carry
  full paths (`/api/auth/register` `auth.py:120`, `/api/auth/login` :151, `/api/auth/me` :175).
  → Phase 3's `backend/fl_routes.py` Blueprint should follow this pattern (full paths, no prefix).
- All other 37 routes are bare `@app.route` in `app.py`.
- `app.json.sort_keys = False` `app.py:39` — **load-bearing**: layer order must survive serialization.
- Optional SocketIO behind `try/except ImportError` `app.py:48-55` (`SOCKETIO_AVAILABLE`), broadcast helper `emit_alert()` `app.py:58-65`.

## 8. Blockchain module (`backend/blockchain.py`, 238 lines)

`class Block` :17 — `__init__(index, timestamp, transactions, previous_hash, nonce=0)` :30 ·
`calculate_hash()` :38 (`sha256(json.dumps({...}, sort_keys=True))`) · `mine_block(difficulty)` :61 (returns attempts) · `to_dict()` :87.

`class Blockchain` :101 — `__init__(difficulty=4, genesis_timestamp=None)` :112 ·
`_create_genesis_block()` :120 · `get_latest_block()` :131 · **pending pool** `self.pending_transactions` :115
with `add_transaction()` :135 · `mine_pending_transactions(block_timestamp=None)` :143 (returns `Block|None`, clears pool :170, sets `mining_time`/`attempts` :172-173) ·
`is_chain_valid()` :177 → **dict** `{"valid","errors"}` · `get_chain()` :227 · `get_stats()` :231.

Caveat: `app.py:139,147` append to `bc.pending_transactions` **directly**, bypassing `add_transaction()`.
Chain caching lives in app.py: `_pow_chain_cache` :116, `_invalidate_chain_cache()` :119, `_get_pow_chain()` :123 (full rebuild from the registry on each invalidation).

## 9. Tests

Layout: `backend/tests/__init__.py` (empty, makes it a package), `test_api.py` (90 tests), `test_blockchain.py` (9 tests). No `conftest.py`, no pytest config file anywhere.

**Invocation** (from repo root):

    python3 -m pytest backend/tests/ -q

**Current: 99 passed, 0 skipped.** Three conditional skips exist but none trigger here —
flask-socketio (`test_api.py:599`), reportlab (`:650`), and wallet-file presence (`test_blockchain.py:74`) are all satisfied.

Autouse fixture `reset_state` `test_api.py:20-46` clears registries, monkeypatches `save_state` to a
no-op (tests never touch the real JSON DB), and mocks `broadcast_hash_to_algorand`.
Installed pytest is 9.0.2 while requirements pins 8.3.5.

## 10. Import recipes for later phases

Neither `backend/` nor `evaluation/` is a package; the established convention is a
`sys.path` insert then a flat import (`test_api.py:13`, `evaluate.py:20`).

```python
# Phase 3 — reuse the canonical Merkle (importing app.py is offline-safe:
# wallet load is file-I/O only, App-ID/contract load is lazy, no network at import)
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "backend"))
from app import _manifest_leaf, compute_layer_merkle, merkle_inclusion_proof

# Phase 4 — reuse the probe port
sys.path.insert(0, os.path.join(_ROOT, "evaluation"))
from eval_lib import bv_find_outliers, bv_layer_health, bv_numeric_stats
```

## 11. Environment

Python **3.12.3**, system interpreter, **no virtualenv**. `/usr/lib/python3.12/EXTERNALLY-MANAGED`
exists → `pip install` needs `--break-system-packages` (or a venv).

| Needed by | Package | Status |
|---|---|---|
| Phase 1+ | torch, torchvision | **MISSING** |
| Phase 2 | opacus | **MISSING** |
| Phase 5 | wfdb, pandas | **MISSING** |
| Phase 1+ | tqdm | **MISSING** |
| Phase 4/6 | scikit-learn, scipy | **MISSING** |
| present | numpy 1.26.4, matplotlib 3.11.1, pyyaml 6.0.1 | OK |
| backend | flask 3.1.3, flask-cors 6.0.2, py-algorand-sdk 2.11.1, PyJWT 2.7.0, bcrypt 3.2.2, flask-socketio 5.4.1, reportlab 4.2.5, bit 0.8.0, requests 2.31.0, pytest 9.0.2 | OK (pins drift) |

**Disk pressure: 97% full, ~2.4 GB free.** Plain-PyPI `torch` declares five NVIDIA CUDA deps
(~4 GB) and would fail. Use the CPU-only index (verified reachable, HTTP 200) — see
`CLAUDE.md` amendment A2.

## 12. State persistence

- `DATA_DIR = backend/data` `app.py:68` (created :73); `REGISTRY_FILE=models_registry.json`,
  `LOGS_FILE=verification_logs.json`, `CHAIN_FILE=chain.json` **:69-71**
  (`CHAIN_FILE` is a **dead constant** — never read or written; the chain is rebuilt in memory).
- `save_state()` `app.py:79-89` (also invalidates the chain cache); `load_state()` `:92-103`, called once at import `:113`.
- In-memory `models_registry` `:109`, `verification_logs` `:110`.
- Other writers into the same dir: `auth.py:31-33` (`users.json`, `secret.key`),
  `algorand_client.py:49` (`algo_wallet.json`), `contract.py:45` (`algo_app.json`).
- Gitignored via `backend/data/` `.gitignore:18`. (A stale `backend/.gitignore:1` contains a wrong relative path that matches nothing.)

## 13. Landmines — pre-existing; do NOT "fix" inside FedVerify scope

Recorded so later phases don't trip over them or mistake them for their own bugs. Tracked in `fedverify/TODO.md`.

1. `_get_pow_chain()` reads `m.get("name")` / `m.get("hash")` (`app.py:135,137`) but the registry
   stores `modelName` / `modelHash` → **every chain `register` transaction carries `"Unknown"` and `""`**.
2. `bitcoin_client.anchor_merkle_root()` reports a 32-byte OP_RETURN script from
   `build_op_return_script_hex()` but actually broadcasts `message=merkle_root[:64]`, i.e. the
   **64-byte ASCII hex string** (`bitcoin_client.py:104`) — reported script ≠ broadcast script.
3. Version drift between `backend/requirements.txt` and installed packages (algosdk, pytest, flask, bcrypt, PyJWT); `bit` installed but unlisted.
4. `docs/TAMPER_DETECTION.md:234` still claims 82 tests (now 99).
5. Stray junk files from a malformed pip invocation: `backend/=2.8.0`, `backend/=4.1.0`.
6. Working tree is dirty on `main` (modified `.gitignore`, `algorand_client.py`, `app.py`,
   `test_api.py`, `index.html`; untracked `evaluation/`, `docs/EVALUATION.md`,
   `docs/RELATED_WORK.md`, `docs/TAMPER_DETECTION.md`). The human commits; I never do.
