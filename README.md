# BlockVerify — AI Model Integrity on Algorand Testnet

> **A blockchain-powered platform for registering and verifying AI model integrity using the Algorand Testnet.**  
> Every model registration creates a permanent, tamper-proof record on a live public blockchain — not a simulation.

---

## 🌐 Live Demo

| Service | URL |
|---|---|
| 🖥️ **Frontend** | [https://incredible-swan-b3b47a.netlify.app](https://incredible-swan-b3b47a.netlify.app) |
| ⚙️ **Backend API** | [https://blockverify-backend-4xug.onrender.com](https://blockverify-backend-4xug.onrender.com) |

> ⚠️ **Note:** The backend runs on Render's free tier. It **spins down after 15 minutes of inactivity** — the first request may take 30–60 seconds to wake up (cold start). Any registered data is cleared on restart due to ephemeral storage. For the full persistent experience, run the project locally using the instructions below.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Why We Migrated to Algorand](#why-we-migrated-to-algorand)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Smart Contract](#smart-contract)
- [API Endpoints](#api-endpoints)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Test Suite](#test-suite)
- [Team](#team)
- [Photo Shield](#photo-shield)

---

## Project Overview

BlockVerify solves a critical problem in the AI industry: **how do you prove that an AI model has not been tampered with?**

When organisations deploy AI models in production (healthcare, finance, autonomous systems), there is no standard way to verify the model file you are running is the exact one that was trained, audited, and approved.

**BlockVerify fixes this by:**

1. Computing the **SHA-256 cryptographic fingerprint** of any AI model file entirely in the browser (the file never leaves the user's machine)
2. Broadcasting that fingerprint as an **immutable transaction on the Algorand Testnet**
3. Also writing the `model_id → hash` mapping into a **deployed Algorand Smart Contract**
4. Allowing anyone to **re-verify** a model at any time by re-hashing the file and comparing against the on-chain record — a mismatch is instant proof of tampering

---

## Why We Migrated to Algorand

### The Original Approach

BlockVerify was initially implemented with a **custom Python Proof-of-Work (PoW) blockchain** inspired by Bitcoin's architecture. Each model registration mined a new block using SHA-256 with a difficulty target of `0000...`. While this demonstrated core blockchain concepts (immutable linked blocks, Merkle trees, chain validation), it had fundamental limitations:

| Limitation | Impact |
|---|---|
| Chain only existed on the local server | No decentralisation, single point of failure |
| Anyone with server access could alter the chain | Defeated the purpose of immutability |
| PoW mining took 3–10 seconds per transaction | Poor user experience |
| No public verifiability | A third party had no way to independently verify a claim |
| Could not survive server restarts without custom persistence | Fragile in production |

### Why Algorand

After review, we migrated to the **Algorand Testnet** — a live, globally distributed public blockchain. The comparison below explains the decision:

| Criteria | Custom Python Chain | Algorand Testnet |
|---|---|---|
| **Decentralisation** | ❌ Single server | ✅ 1,000+ global validators |
| **Immutability** | ❌ Modifiable by admin | ✅ Cryptographically finalised, irreversible |
| **Public Verifiability** | ❌ Trust the server | ✅ Anyone can verify on the explorer |
| **Consensus** | PoW (energy intensive) | Pure Proof-of-Stake (PPoS), ~3.4s finality |
| **Smart Contracts** | ❌ Not supported | ✅ TEAL v8 on-chain program |
| **Transaction Cost** | Free (fake) | ~0.001 Test ALGO (free from faucet) |
| **Industry Recognition** | ❌ Academic toy | ✅ Real blockchain platform |
| **Uptime** | Depends on our server | ✅ 99.9%+ network uptime |

### Why Not Ethereum / Solidity?

We previously used Ethereum with Solidity smart contracts. We migrated away for the following reasons:

- **Gas fees** on Ethereum Mainnet are unpredictable and expensive (can exceed $5–50 per transaction during congestion)
- **Slow finality** — Ethereum blocks take ~12 seconds with probabilistic finality (requires ~12 confirmations = ~2.5 minutes)
- **Contract complexity** — Solidity requires careful reentrancy guards, gas optimisation, and auditing overhead for even simple storage contracts
- **Algorand's PPoS** provides **immediate, deterministic finality** — a confirmed block is final with zero possibility of reversal, no forks, no reorgs

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        BROWSER                               │
│                                                             │
│   SHA-256 hash computed locally (file never uploaded)       │
│   ↓                                                         │
│   BlockVerify Frontend (HTML/CSS/JS)                        │
│   http://localhost:8080                                      │
└───────────────────────┬─────────────────────────────────────┘
                        │ REST API
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                  Flask Backend (Python)                      │
│                  http://localhost:5000                       │
│                                                             │
│   app.py          — REST endpoints                          │
│   auth.py         — JWT authentication                      │
│   algorand_client.py — Algorand SDK integration             │
│   contract.py     — TEAL Smart Contract (deploy + call)     │
│                                                             │
│   Local JSON storage (metadata cache):                      │
│     data/models_registry.json                               │
│     data/verification_logs.json                             │
│     data/algo_wallet.json   ← server signing wallet         │
│     data/algo_app.json      ← deployed smart contract ID    │
└───────────────────────┬─────────────────────────────────────┘
                        │ py-algorand-sdk
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              Algorand Testnet (Public Network)               │
│                                                             │
│   Note Transaction  — model metadata in tx note field       │
│   Smart Contract    — model_id → hash in global state       │
│   Indexer API       — live transaction lookup               │
│   AlgoNode          — free public API node (no auth)        │
│                                                             │
│   Explorer: https://lora.algokit.io/testnet                 │
└─────────────────────────────────────────────────────────────┘
```

### How a Registration Works (Step by Step)

1. User uploads model file → browser computes SHA-256 hash
2. Frontend calls `POST /api/register` with `{modelName, modelHash, owner}`
3. Backend signs a **0-ALGO transaction** with JSON metadata in the `note` field
4. Transaction is broadcast to Algorand Testnet via AlgoNode public API
5. Backend waits for **block confirmation** (~3 seconds)
6. Backend calls the **deployed Smart Contract** using `ApplicationNoOpTxn` — stores `model_id → hash` in contract global state
7. Both TxIDs are returned to the frontend and stored locally for fast lookup
8. User sees the **Algorand TxID** — clickable link to `lora.algokit.io/testnet`

---

## Key Features

### Core Functionality
| Feature | Description |
|---|---|
| **Model Registration** | Upload any file → compute SHA-256 → broadcast to Algorand |
| **Integrity Verification** | Re-hash a file and compare against on-chain record |
| **Version History** | Track multiple versions of the same model |
| **Audit Trail** | Complete log of all verification attempts per model |
| **Model Deactivation** | Mark models as deactivated (also written to blockchain) |
| **Privacy Toggle** | Mark models as private — only owner can verify |

### Blockchain Features
| Feature | Description |
|---|---|
| **Smart Contract Registry** | TEAL v8 contract stores `model_id → hash` on-chain |
| **Interactive 3D Explorer** | Zero-dependency pure CSS 3D visualization of the blockchain blocks |
| **Merkle Tree Visualization** | Cryptographic proof trees generated and cached instantly per block |
| **Chain Proof Lookup** | Paste any Algo TxID → fetch raw Algorand Indexer data |
| **Live Wallet Dashboard** | Server wallet balance, App ID, account creation round |
| **Algorand Explorer Links** | Every TxID links directly to the live testnet explorer |

### Platform Features
| Feature | Description |
|---|---|
| **JWT Authentication** | Secure login with token-based auth |
| **Guest Mode** | Browse public registry without an account |
| **Rate Limiting** | 10 write operations per 60 seconds per user |
| **Public Registry** | Browse all registered models across all users |
| **Activity Chart** | Daily registration and verification analytics |
| **Guided Tour** | Interactive onboarding tour for new users |
| **Dark / Light Mode** | Full theme toggle |
| **PDF Reports** | Download a model integrity certificate as PDF |

---

## Smart Contract

The BlockVerify smart contract is a **TEAL v8** program deployed on the Algorand Testnet.

### What It Does

```
method: register(model_id: bytes, model_hash: bytes)
storage: GlobalState (key-value, up to 64 entries)
```

On every model registration, the contract is called with:
- `app_args[0]` = `"register"` (method selector)
- `app_args[1]` = model ID (up to 16 bytes)
- `app_args[2]` = SHA-256 hash (64 bytes hex)

This permanently writes `model_id → hash` into the **contract's global state** on Algorand. The data is:
- **Immutable** once written
- **Publicly readable** by anyone
- **Permanently stored** on the Algorand ledger

### Auto-Deployment

The contract is deployed automatically on the first model registration. The App ID is saved to `backend/data/algo_app.json` and reused on subsequent server restarts.

### Viewing the Contract

After your first registration, visit:
```
https://lora.algokit.io/testnet/application/<APP_ID>
```

The App ID is shown on the **🔍 Chain Proof** tab under "Smart Contract Registry."

---

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Create a new user account |
| `POST` | `/api/auth/login` | Login and receive JWT token |

### Model Operations (Require Auth)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/register` | Register a new AI model on Algorand |
| `POST` | `/api/verify` | Verify model integrity |
| `POST` | `/api/add-version` | Add a new version of a model |
| `POST` | `/api/deactivate` | Deactivate a model |

### Read Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/stats` | Platform statistics |
| `GET` | `/api/registry` | Public model registry (paginated) |
| `GET` | `/api/models/<owner>` | Models owned by a user |
| `GET` | `/api/audit/<model_id>` | Verification audit trail |
| `GET` | `/api/versions/<model_id>` | Version history |

### Algorand Live Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/algo/wallet` | Live wallet balance, App ID, status |
| `GET` | `/api/algo/tx/<txid>` | Fetch live transaction from Algorand Indexer |
| `GET` | `/api/algo/contract` | Deployed smart contract App ID and explorer link |
| `GET` | `/api/chain/validate` | Confirm Algorand network connectivity |

---

## Tech Stack

### Backend
| Component | Technology |
|---|---|
| Web Framework | Flask (Python) |
| Blockchain SDK | `py-algorand-sdk` |
| Smart Contract | TEAL v8 (Algorand bytecode) |
| Algorand Node | AlgoNode public API (no auth required) |
| Algorand Indexer | AlgoNode Indexer API |
| Authentication | PyJWT (JSON Web Tokens) |
| HTTP requests | `requests` library |
| Environment Config | `python-dotenv` |
| Data storage | JSON files (metadata cache) |

### Frontend
| Component | Technology |
|---|---|
| UI | Vanilla HTML5 / CSS3 / JavaScript |
| Charts | Chart.js |
| PDF generation | jsPDF |
| Hashing | Web Crypto API (in-browser SHA-256) |
| Animations | CSS transitions + Canvas particle system |

### Testing
| Component | Technology |
|---|---|
| Test framework | pytest |
| Algorand mocking | `monkeypatch` (no real network calls in tests) |
| Coverage | 43 tests — all passing ✅ |

---

## Getting Started

### Prerequisites

```bash
Python 3.10+
pip
```

### 1. Install dependencies

```bash
cd blockverify/backend
pip install flask flask-cors py-algorand-sdk pyjwt requests pytest python-dotenv
```

### 2. Configure Environment Secrets

Create a `.env` file in the `backend/` directory to store your API keys securely:

```bash
# blockverify/backend/.env
GEMINI_API_KEY="your_secure_gemini_api_key_here"
```

### 3. Start the backend

```bash
python3 app.py
```

On first run, a new Algorand Testnet wallet is generated and printed:

```
🚨 NEW ALGORAND TESTNET WALLET GENERATED 🚨
➜  Server Address : Q3FCOU4FFZCY7LELM...
Fund it (FREE) at: https://bank.testnet.algorand.network/
```

### 3. Fund the wallet (one-time, free)

1. Copy the `Server Address` printed above
2. Visit → [https://lora.algokit.io/testnet/dispenser](https://lora.algokit.io/testnet/dispenser)
3. Paste address → click **Dispense**
4. You receive **10 Test ALGO** = ~10,000 transactions worth of gas fees

> The wallet is saved to `backend/data/algo_wallet.json` and reused on restart.  
> 10 ALGO is enough for the entire academic lifecycle of this project.

### 4. Start the frontend

```bash
cd blockverify/frontend
python3 -m http.server 8080
```

Open → **http://localhost:8080**

### 5. Generate Demo Models (Optional)

To easily test the platform's security detection, we provide a script that generates an original model, a tampered model (weight modified), and a structurally anomalous model (rogue layer added).

```bash
cd blockverify
python3 demo_tools/generate_demo_model.py
```
This generates `demo_original.json`, `demo_tampered_weights.json`, and `demo_rogue_layer.json` for you to upload.

### 6. Register a model

1. Login / Register an account
2. Go to **Register** tab
3. Upload any file (`.pt`, `.h5`, `.pkl`, `.onnx`, PDF, ZIP — or one of the generated `demo_*.json` files)
4. Fill in model name → click **Register Model**
5. The smart contract auto-deploys on the first registration
6. You receive an **Algo TxID** — click to see it live on the blockchain ✅

---

## Test Suite

```bash
cd blockverify
python3 -m pytest backend/tests/ -v
```

```
backend/tests/test_api.py ........................................... [ 97%]
backend/tests/test_blockchain.py .                                   [100%]

43 passed in 0.74s ✅
```

### What the tests cover

| Test Class | Tests | What It Verifies |
|---|---|---|
| `TestAuth` | 8 | Registration, login, JWT, duplicate user, bad credentials |
| `TestRegister` | 7 | Model registration, hash validation, rate limiting, auth guard |
| `TestVerify` | 5 | Valid/invalid hash verification, missing fields, audit logging |
| `TestVersions` | 5 | Version history, add version, validation |
| `TestAudit` | 4 | Audit trail count, format, export |
| `TestReadEndpoints` | 7 | Stats, registry, models per owner, rate limit status |
| `TestDeactivate` | 4 | Deactivation, double-deactivate guard, auth guard |
| `TestPrivacy` | 3 | Private model hiding in registry and verification |

> All Algorand network calls are **mocked** using `monkeypatch` — tests run offline in ~0.7 seconds.

---

## Photo Shield

The third application of the BlockVerify integrity layer, after AI models and federated updates
(FedVerify): **photo provenance plus optional AI-edit cloaking.** Code is in [`photoshield/`](photoshield/).
The UI is the **Photo Shield** tab, and the API is `/api/photo/*`.

**Provenance.**
- The browser computes the photo's SHA-256 and a canonical 64-bit pHash.
- It signs them with an Ed25519 key that lives only in the browser.
- The server anchors them as an Algorand Testnet note.

**Verification.**
- Anyone can check a copy. The result is exact, derived (re-compressed or resized) or similar.
- The derived and similar thresholds are measured, not guessed.
- The browser re-checks the anchor on the public indexer itself.

**Privacy and durability.** The photo never leaves the browser. After a restart, the registry
rebuilds itself from the ledger.

**Cloak.** `python -m photoshield protect photo.jpg --out shielded/` runs on your own machine. It
adds an invisible PGD+EOT perturbation against the Stable Diffusion VAE, so editors built on it
(for example InstructPix2Pix) produce damaged edits. It does **not** protect against closed models,
and heavy re-processing can remove it. The provenance layer works either way.

Docs:
- [What is PhotoShield?](photoshield/docs/WHAT_IS_PHOTOSHIELD.md)
- [Threat model](photoshield/docs/THREAT_MODEL.md)
- [Design](PHOTOSHIELD_DESIGN.md)
- [API](docs/API_REFERENCE.md#photo-shield-apiphoto)
- [PhotoShield README](photoshield/README.md)

## Team

**razor_hats** — Blockchain Engineering Course Project

| Member | Role |
|---|---|
| Shubhangam Singh | Backend, Algorand Integration, Smart Contract, Testing |
| Aditya Kumar, Mihir Dixit | Frontend UI/UX, API Integration, Documentation |

---

## Blockchain Proof

Every model registration on this platform produces:

1. **A note transaction** — searchable by TxID on any Algorand explorer  
2. **A smart contract state entry** — verifiable by reading the contract global state  
3. **A local audit record** — fast lookup without hitting the blockchain every time

The **🔍 Chain Proof** tab in the app lets you paste any TxID and see the raw, decoded transaction data fetched live from the Algorand Indexer — this is the on-chain proof that cannot be faked.

```
Explorer: https://lora.algokit.io/testnet
Indexer:  https://testnet-idx.algonode.cloud/v2/transactions/<TXID>
```

---
## Screen Shots Of Website

<img width="1838" height="1042" alt="image" src="https://github.com/user-attachments/assets/1ecc2a3c-d53d-4843-aa50-acd185237942" />

<img width="1838" height="1042" alt="image" src="https://github.com/user-attachments/assets/10c2d937-d723-4d88-bd71-76ebb69547a8" />

<img width="1838" height="1042" alt="image" src="https://github.com/user-attachments/assets/a18c35c7-0fb3-4bbb-aa91-33a7e62b62a4" />

<img width="1838" height="1042" alt="image" src="https://github.com/user-attachments/assets/1fbee48a-515d-451f-a3aa-e7fae9c42225" />

<img width="1838" height="1042" alt="image" src="https://github.com/user-attachments/assets/71de88db-a4a5-4076-a7c7-146e0d606c1b" />

<img width="1838" height="1042" alt="image" src="https://github.com/user-attachments/assets/55b49d5b-cdda-4482-be4c-4250382a388d" />

<img width="1838" height="1042" alt="image" src="https://github.com/user-attachments/assets/fce61c24-e80d-4287-be4b-ade4b7e003fb" />

<img width="1838" height="1042" alt="image" src="https://github.com/user-attachments/assets/ddabe1f8-2f72-4fc7-8946-82ac7c75f4e8" />

<img width="1838" height="1042" alt="image" src="https://github.com/user-attachments/assets/7c2c8f50-a0ab-4879-8208-30cde6ffd205" />

<img width="1838" height="1042" alt="image" src="https://github.com/user-attachments/assets/fb6c4ee4-69e6-4a0a-87ff-ebea190f0c54" />

<img width="1838" height="1042" alt="image" src="https://github.com/user-attachments/assets/d1998f37-00a4-4b64-bb5a-8c8d2b9e4e3d" />

<img width="1838" height="1042" alt="image" src="https://github.com/user-attachments/assets/abb0e557-096a-4270-8e06-0947379345a5" />

<img width="1838" height="1042" alt="image" src="https://github.com/user-attachments/assets/f468205d-79d2-4ac5-bb7b-2324e999a68f" />

<img width="1838" height="1042" alt="image" src="https://github.com/user-attachments/assets/a26e9642-a146-4cd5-b299-b6b2f8c1bb24" />

<img width="1840" height="1040" alt="image" src="https://github.com/user-attachments/assets/3273dda3-4cb3-418a-ba0c-13aced5b57c2" />

<img width="1840" height="1040" alt="image" src="https://github.com/user-attachments/assets/e82b6535-4916-4ff3-a05d-3bcb2d9b51c3" />

<img width="1840" height="1040" alt="image" src="https://github.com/user-attachments/assets/43b63b10-da9a-4689-8c98-a6af081bd8f6" />

<img width="1840" height="1040" alt="image" src="https://github.com/user-attachments/assets/9e5baba1-67a6-470a-9f42-1f8f177256fc" />

<img width="1840" height="1040" alt="image" src="https://github.com/user-attachments/assets/37d11de7-fd52-479a-b179-13e7f2e5e40b" />

<img width="1840" height="1040" alt="image" src="https://github.com/user-attachments/assets/b92c6d1e-c2d2-4e8d-962d-bde5ba3f242b" />

<img width="1840" height="1040" alt="image" src="https://github.com/user-attachments/assets/44029d08-5cf5-4feb-9582-5d00df1754f2" />

---
> *"Trustless verification of AI models — not because we say so, but because the blockchain says so."*
