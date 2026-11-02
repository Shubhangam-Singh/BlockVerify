# FedVerify — complete dump of everything added so far

*Snapshot of the federated-learning add-on as it exists in this repository today.
Written so one file holds the problem, architecture, code, experiments, dashboard,
panel talk-track, tests, and open issues.*

This is a dump, not a replacement for the shorter docs:

| Read this if you want… | File |
|---|---|
| A panel-friendly explanation | [`WHAT_IS_FEDVERIFY.md`](WHAT_IS_FEDVERIFY.md) |
| Threat model, proofs, non-guarantees | [`FEDERATED.md`](FEDERATED.md) |
| How to run the simulator | [`../README.md`](../README.md) |
| Phase-0 ground truth of BlockVerify | [`../INVENTORY.md`](../INVENTORY.md) |
| Parked work | [`../TODO.md`](../TODO.md) |

**Numbers in the experiment tables below are copied from
`fedverify/results/tables/`, which `make_tables.py` generated from result files.
Empty cells are `—`. Nothing here is typed by hand.**

---

## 0. One paragraph

Ten hospitals want a better model. Patient data cannot leave each hospital.
**Federated learning** lets each hospital train locally and send only a
**delta** (the change it wants to make to the shared weights). A server averages
those deltas and sends the improved model back. Repeat for many **rounds**.

That solves data sharing and creates three new problems:

1. The server sees every delta and can sometimes reconstruct training examples.
2. A malicious hospital can poison or backdoor the model.
3. The server can quietly drop, reorder, or rewrite who contributed.

**FedVerify** is the BlockVerify add-on that answers all three, and is explicit
about where each guarantee stops. It is **additive**: existing Register / Verify
/ Compare behaviour was not rewritten. The Flask app gained a Blueprint; the
frontend gained a **Federated** tab.

Core product = seal a **finished** model (BlockVerify).
FedVerify (objective O7) = seal the **training process** while many parties who
do not fully trust each other are still building it.

---

## 1. The three adversaries

| Adversary | Capability | Answered by |
|---|---|---|
| **A1 — curious server** | Sees every client update; wants to infer training data | DP-SGD (`fedverify/core/privacy.py`). Anchoring a digest hides nothing. |
| **A2 — Byzantine clients** (≤ f of K) | Send arbitrary updates: scaled, flipped, noise, backdoored | Robust aggregation, especially **FedVerify-Forensics** |
| **A3 — equivocating aggregator** | Reports one round history to one party and a different one to another; silently drops or rewrites an update | Per-round Merkle commitment on a public L1 |

Assumed, not defended: clients are authenticated out of band; the aggregator is
semi-honest with respect to the *arithmetic* of FedAvg (we prove **which updates
entered a round**, not that the average was computed correctly); the network is
not the adversary; ≤ f of K clients are Byzantine, with f known.

---

## 2. How FedVerify sits on BlockVerify

Nothing in the single-artefact path was deleted. `backend/app.py` gained a
Blueprint import and its registration. Algorand App ID is **764828342**
(loaded by `contract.load_app_id()`; never hardcode `758544892`).

### 2.1 Two pieces reused, not rewritten

**Canonical Merkle** — `backend/app.py`:

| Function | Line | What it does |
|---|---|---|
| `_manifest_leaf(index, name, layer_hash)` | **1562** | `sha256(json.dumps([index, name, layer_hash], separators=(",", ":"), ensure_ascii=False))` |
| `_manifest_order` | 1567 | explicit order first, stragglers keep key order |
| `compute_layer_merkle` | 1575 | returns `(root, leaves, order)`; duplicate-last on odd levels |
| `merkle_inclusion_proof` | 1590 | `[{"hash": hex, "right": bool}, ...]` |

Parent rule: `sha256((left + right).encode()).hexdigest()` over **hex strings**.
Leaf count is anchored beside the root because duplicate-last trees are
ambiguous without it (CVE-2012-2459 class).

For a **finished model**, the note keys are `lmr` (layer Merkle root) and `lc`
(leaf count). For an **FL round**, the same tree is used with note keys
`frr` / `flc` / `rnd` / `run`.

The browser already re-verifies layer manifests with `bvMerkleLeaf` /
`bvFoldProof` / `bvSha256Hex` in `frontend/index.html`. An FL round leaf uses
the same encoding: `SHA-256(JSON([i, client_id, digest]))`.
`fedverify/chain/commitment.py` **imports** `_manifest_leaf`,
`compute_layer_merkle`, and `merkle_inclusion_proof`. There are five Merkle
implementations in the repo; only this one is used for FL.

**Forensic probes** — `evaluation/eval_lib.py`, the Python port of the
in-browser detector:

| Function | Returns | JS original |
|---|---|---|
| `bv_numeric_stats` | `{n, min, max, mean, std}` | `bvNumericStats` |
| `bv_find_outliers` | median / 1.4826·MAD scale, max z | `bvFindOutliers` |
| `bv_layer_health` | NaN, Inf, extremes, entropy, max run | `bvLayerHealth` |

FedVerify-Forensics imports these and asks the same questions **across K client
deltas** instead of across D weights of one tensor. A malicious client is an
outlier among its peers the way a poisoned layer is an outlier inside a model.

### 2.2 Architecture diagram (how to say it)

On the slides: **register on the right, verify on the left, FedVerify as a
dashed reuse of the same Merkle + probes.**

- Right: a finished file is hashed in the browser, layers become leaves, the
  root goes to Algorand Testnet.
- Left: a later file is re-hashed, the tree is walked, a mismatch names the
  layer.
- Dashed: the FL round does the same job one level up (clients instead of
  layers) and the Federated tab reads committed runs.

Batch verification is **not** an ML model and is **not** SmallCNN. It is
`POST /api/verify-batch` (`backend/app.py` ~385): up to 20 files hashed in the
browser, classified as verified / tampered / unregistered.

---

## 3. Per-round data flow

```
each selected client
    trains locally
    returns δ_k = θ_local − θ_global          (never raw weights)

attacks, if any, applied HERE                 (chain must bind what was sent)

canon_update  → float32, little-endian, C-contiguous bytes
update_digest → SHA-256( compact-JSON{client_id, num_samples, numel, round} ‖ bytes )
leaf_k        → SHA-256( JSON([k, client_id, digest]) )     ← _manifest_leaf
round root    → MerkleRoot(leaves), n = K anchored beside it
anchor        → mock | local PoW | Algorand note {frr, flc, rnd, run}

screening (forensics / krum / …)
survivors FedAvg'd
accept / reject split stored with the round
every checkpoint_every rounds: Bitcoin OP_RETURN of MerkleRoot(R_1 … R_r)
```

Three load-bearing details:

- The digest header binds identity, round, and sample count. Without it a delta
  could be replayed by another client or into another round.
- The leaf binds **position**. Swap two clients → the root changes.
- The leaf count closes the duplicate-last ambiguity.

`--commit` is **off by default** (amendment A6). Table-1 cells produced before
Phase 3 stay comparable. With it on, each round record gains a `commit` block
and `commitments.jsonl` is written beside `rounds.jsonl`.

---

## 4. Phases 0–6 (what shipped)

| Phase | What it added |
|---|---|
| **0** | Read the existing codebase. Found the plan disagreed with reality (wrong App ID, five Merkle implementations, stale test count). Followed the code. Wrote `INVENTORY.md`. |
| **1** | Federated training loop: clients, Dirichlet / IID split, GroupNorm models, reproducible seeding. |
| **2** | Differential privacy with accounting over **total** steps, privacy–utility experiment (Table 1). |
| **3** | Round commitments, Merkle proofs, chain backends, six `/api/fl/*` routes. |
| **4** | Attacks, robust-aggregation baselines, FedVerify-Forensics, threshold calibration. |
| **5** | MIT-BIH (patient-disjoint hospital split), heterogeneity and DP×Byzantine experiments. |
| **6** | Federated dashboard tab, figures, documentation. |

---

## 5. Package map

```
fedverify/
  config.py              frozen FLConfig + argparse (every knob → config.json)
  core/
    data.py              load_dataset, dirichlet_partition, iid_partition
    models.py            SmallCNN, ECGCNN1D, get/set_flat_params (GroupNorm only)
    client.py            Client.local_train → ClientUpdate (a DELTA)
    privacy.py           DP-SGD plan, σ over TOTAL steps, persistent RDP accountant
    server.py            evaluate (loss / acc / macro-F1 / per-class)
    runner.py            run(cfg), seeding, JSONL, optional commit
  chain/
    commitment.py        canon_update, update_digest, round Merkle (imports app.py)
    anchor.py            mock | local | algorand
    push.py              POST a finished run folder to /api/fl/round/commit
  aggregators/
    fedavg.py krum.py trimmed_mean.py median.py forensics.py
  attacks/
    byzantine.py         sign_flip, gaussian, zero, scaling, label_flip dispatch
    backdoor.py          3×3 white square in normalised space + ASR
  datasets/
    mitbih.py            PhysioNet MIT-BIH, AAMI 5-class, de Chazal split
  analysis/
    calibrate.py         ROC → results/calibration/taus.json
    make_tables.py       results files → results/tables/  (never typed numbers)
    plots.py             figures
    comparison.yaml      source of Table 5; "?" where a paper is unclear
    make_paper.py        numbers.tex for the paper
  experiments/
    exp1_privacy_utility.py
    exp2_byzantine.py
    exp3_chain_overhead.py
    exp4_heterogeneity.py
    exp5_dp_byzantine_interaction.py
  tests/                 160 collected tests (see §11)
  results/               grids, calibration, demo runs, generated tables
  docs/                  this file, FEDERATED.md, WHAT_IS_FEDVERIFY.md

backend/fl_routes.py     Blueprint fl_bp, six /api/fl/* routes
backend/data/fl_runs.json committed dashboard runs
frontend/index.html      Federated tab (additive; no existing tab deleted)
evaluation/eval_lib.py   probes imported by forensics
```

`FLConfig` (`fedverify/config.py`) is a frozen dataclass. Cell names look like
`mnist_K10_a0.5_eps4_forensics_scaling0.2`. MIT-BIH cells use `patient` instead
of `a0.5` because Dirichlet alpha is not used on a patient split.

Output of every run:

```
fedverify/results/<exp>/<cell>/seed<N>/
    rounds.jsonl      one JSON object per round
    config.json       every parameter that influenced the run
    final_model.pt    final global model
    commitments.jsonl (only if --commit)
```

`rounds.jsonl` fields include `round, test_acc, test_loss, macro_f1,
mean_train_loss, per_client_num_samples, diag, train_wall_s, round_wall_s`.
`diag` always carries `accepted / rejected / scores`.

---

## 6. Code, explained simply

### 6.1 The model (`core/models.py`)

`SmallCNN`: 28×28×1 → 10 classes. Two conv-GroupNorm-ReLU-pool blocks, then two
FC layers. About **80,298** parameters. Used for MNIST and Fashion-MNIST.

`ECGCNN1D`: 256-sample single-lead beat → 5 AAMI classes. Three 1-D conv
blocks, GroupNorm, global average pooling. About **75,461** parameters — sized
to match SmallCNN so Table 4 compares **datasets**, not capacity.

**GroupNorm everywhere, never BatchNorm.** Opacus cannot handle BatchNorm, and
BatchNorm running statistics leak across clients even without DP.

### 6.2 The client (`core/client.py`)

Each client receives the current global weights, trains on its own indices,
and returns

```
δ = θ_local − θ_global     (a 1-D tensor, same length as the model)
```

plus `num_samples`, training loss, and (when DP is on) a privacy report.
Raw weights never leave the client object as the committed artefact.

When DP is off (`epsilon` is `None` or `inf`), Opacus is never imported. A test
asserts that path is byte-identical to Phase 1.

### 6.3 Differential privacy (`core/privacy.py`)

The bug this module exists to avoid: solving σ for **one round**, then running
R rounds, silently spends about R times the claimed budget.

Correct accounting, solved **once per client** and held fixed:

```
steps_per_round_k = local_epochs * ceil(n_k / batch_size)
total_steps_k     = rounds * steps_per_round_k
sample_rate_k     = batch_size / n_k
sigma_k           = get_noise_multiplier(eps, delta, sample_rate_k,
                                         steps=total_steps_k, accountant="rdp")
```

Each client owns **one** `RDPAccountant` that persists across rounds.
`δ = 1e-5`. Clipping bound `C = max_grad_norm` (default 1.0).

Guard test: `test_noise_multiplier_is_solved_over_TOTAL_steps` spies on
`get_noise_multiplier` and asserts the step count it was called with.

**What the guarantee is.** Each client's reported ε bounds the privacy loss of
that client's contribution across the whole run, under DP-SGD assumptions
(Poisson sampling, per-sample clipping to C, Gaussian noise σ·C).

**What it is not.** The unit of privacy is a **training example**. On MIT-BIH
that is a *beat*, not a *patient* — and a patient contributes hundreds of
beats. Patient-level DP needs group privacy or per-patient clipping. Stated
as a limitation, not claimed.

Solving for `ceil(n_k / batch_size)` steps per epoch upper-bounds Poisson
sampling, so the error is in the safe direction: realized ε comes in at or
slightly below target (see Table 1 “max realized ε”).

### 6.4 The runner (`core/runner.py`)

Seeds Python, NumPy, and Torch. Pins `--torch-threads` when given and always
records `torch_threads_effective` (amendment A5): intra-op thread count
changes float reduction order, so results are only byte-reproducible at a
fixed thread count.

`strip_timings` drops `train_wall_s`, `round_wall_s`, `*_ms` so two seeded
runs can be compared (amendment A1). Timings can never be byte-identical.

Attacks are applied **before** the Phase-3 commitment. Attacker identity comes
from the seed alone, so the same clients are malicious for every aggregator
being compared. `--attack-start-round` keeps them dormant during warm-up.

### 6.5 Commitment (`chain/commitment.py`)

```python
def canon_update(delta) -> bytes:
    # float32, little-endian, C-contiguous — identical across machines

def update_digest(client_id, round_num, delta, num_samples) -> str:
    # SHA-256(compact JSON header || raw bytes)

def leaf_hashes(leaves):
    return [app._manifest_leaf(i, cid, dig) for i, (cid, dig) in enumerate(...)]
```

`warm_up()` forces the Flask-app import **before** any timed region, otherwise
the ~1.5–2.5 s first import lands in round 1's `merkle_ms` and corrupts Table 3.

### 6.6 Chain backends (`chain/anchor.py`)

| backend | I/O | txid | use |
|---|---|---|---|
| `mock` | none | SHA-256 of the committed content (deterministic) | experiments |
| `local` | mines a block on `backend/blockchain.py` | block hash | tests, default demo |
| `algorand` | 0-ALGO self-transaction, note `{frr, flc, rnd, run}` | real 52-char txid | one confirmation run |

Local chain adds two **new** transaction types (additive): `fl_client_update`
and `fl_round_commit`. Existing types remain `genesis` / `register` / `verify`.

Bitcoin checkpoints every `--checkpoint-every` rounds commit
`MerkleRoot(R_1 … R_r)`. Without `--btc-checkpoint` the checkpoint root is
computed but anchored to a mock txid. Live BTC testnet checkpoints were not
exercised.

Algorand FL anchoring writes the **note only**. It does not write the smart
contract's global state the way `broadcast_hash_to_algorand` does for layer
roots.

### 6.7 Aggregators

```
fedavg         weighted mean by num_samples (the baseline)
krum           keep the client closest to its neighbours (selection rule)
multikrum      keep several
trimmed_mean   drop a fraction from each end per coordinate
median         coordinate-wise median
forensics      per-client accept/reject, then FedAvg survivors
```

Krum is a **selection rule**, not a detector. Vanilla Krum keeps one client.
Reading its “rejected” list as detections is a category error; Table 2b marks
Krum / FedAvg / median / trimmed-mean as `n/a` for detection F1.
`meta.detector` is sent by `chain/push.py` from the aggregator registry so the
UI does not hardcode that list.

### 6.8 FedVerify-Forensics (`aggregators/forensics.py`)

Four per-client scores, each a robust z **across clients**:

| score | statistic | direction | catches |
|---|---|---|---|
| `s_norm` | `‖δ_k‖₂` | two-sided | scaling (inflates), zero / free-rider (collapses) |
| `s_dir` | `cos(δ_k, median δ)` | one-sided low | sign-flip, (some) label-flip |
| `s_coord` | share of coordinates beyond `tau_coord` | one-sided high | sparse, targeted edits |
| `s_health` | NaN, ±Inf, `\|w\|>100`, entropy, constant run | hard flag | corrupted / degenerate |

`combined = max(s_norm, s_dir, s_coord)`, mirroring the deployed detector's
max-robust-z operating point. Reject if a hard health flag fires **or**
`combined > τ`. Survivors are FedAvg'd. Health-flagged clients are excluded
from the population statistics, so one NaN cannot poison the median.

**Why four scores.** A sign-flip negates the update. Its *size* is normal, so
`s_norm` is blind (measured ~0.05) while `s_dir` fires hard (~2022, cosine
~−0.975). A norm-only detector would miss it.

**τ is never hardcoded.** It comes from `results/calibration/taus.json` via
`load_tau(..., policy="tau_fpr5")`. Every run records `tau` and `tau_source`
(file path, key, policy) in `config.json`. Constructing Forensics without a τ
**raises**. There is no default.

This is the same mistake BlockVerify documented one level down: the hand-set
layer threshold `z > 8` had FPR 0.49 on real heavy-tailed weights. An early
FedVerify demo used a made-up τ of 6.0; the calibrated value was ~2.65.
Detection sat at 31% with the made-up cut-off and 75% with the calibrated one.
The dashboard shows provenance, and a hand-typed τ is labelled amber.

`tau_coord` (inside `s_coord`) has a documented default of 3.0 and is recorded,
but it is **not** itself calibrated. A 2-D calibration is future work.

On smoke calibration, `s_coord` alone scored a higher AUC than `combined`
because the max lets whichever sub-score is noisiest lift honest clients.
Not changed on the strength of a smoke sample.

### 6.9 Attacks (`attacks/`)

| attack | level | effect |
|---|---|---|
| `label_flip` | data | `y → (C−1) − y` (exactly `9−y` on 10-class sets) |
| `backdoor` | data | 3×3 white square bottom-right on `poison_frac` of the batch, relabelled to `backdoor_target` |
| `sign_flip` | delta | `δ → −s · δ` |
| `gaussian` | delta | `δ → N(0, σ²)` |
| `zero` | delta | `δ → 0` (free-rider; implemented, not in the exp2 grid) |
| `scaling` | delta | `δ → (K/f) · δ` (model replacement, Bagdasaryan et al.) |

The backdoor trigger is written in **normalised** space — pure white is
`(1 − mean) / std`, about 2.82 for MNIST. Stamping a literal 1.0 would be a
grey smudge and would understate the attack. ASR is measured on triggered
test images whose **true** class is not the target, so target-class samples
cannot count as free successes.

Attacks evaluated are **non-adaptive**. The attacker does not know the
defence. An adaptive attacker who stays inside the honest robust-z band is
the natural strong baseline and is **not implemented**.

### 6.10 MIT-BIH (`datasets/mitbih.py`)

MNIST non-IID is *synthesised* (Dirichlet α). MIT-BIH skew is *inherited*:
each client is a distinct set of patients.

| choice | what | why |
|---|---|---|
| records | 48 minus paced 102/104/107/217 | AAMI EC57 excludes paced beats |
| split | de Chazal DS1/DS2, 22 records each | inter-patient; a patient never crosses |
| beats | 256 samples centred on each R-peak | ≈0.71 s at 360 Hz |
| scaling | z-norm **per record** | gain and baseline differ |
| labels | AAMI 5-class N/S/V/F/Q | standard |

Measured in the docs: **100,693 beats, 44 patients**, class balance
N 89.47% / S 2.76% / V 6.96% / F 0.80% / Q 0.01%.

**Macro-F1 is the primary metric.** A model that always answers N scores
**89.47% accuracy and 18.89% macro-F1**. Accuracy on this dataset flatters
bad models. Table 4 prints macro-F1 first and accuracy in parentheses.

Class Q has only 15 beats corpus-wide once paced records are excluded, so
its per-class F1 is near-meaningless. The model uses lead 0 only.

At K=10 in the Phase-5 demo, hospital 9 held 18.8% ventricular beats while
hospital 4 held 0.4% — a 47× spread from which patients they were given.

---

## 7. Browser proof (the part that matters)

```
GET /api/fl/proof/<run_id>/<r>/<client_id>
  → { index, digest, leaf, proof, root, indexerUrl }
```

The browser then, in order:

1. **Recomputes** `ℓ = SHA-256(JSON([index, client_id, digest]))`. It does not
   trust the served `leaf`.
2. Folds the sibling path (`bvFoldProof`) and checks it reaches `root`.
3. Fetches the anchoring transaction **directly from
   `testnet-idx.algonode.cloud`**, bypassing this server, and reads `frr` /
   `flc` from the note.
4. Accepts only if `frr == root`, `flc == leafCount`, and the fold succeeded.

Three verdicts:

| Verdict | Meaning |
|---|---|
| **TRUSTLESS ✓** | All four passed. Independently proven. |
| **SERVER-TRUSTED** | Steps 1–3 passed, but this run was anchored on `local` or `mock`, so there is no public ledger for step 4. Honest, clearly labelled. |
| **VERIFICATION FAILED / ROOT MISMATCH** | Something does not match. Never silently downgraded. |

**Most demo runs show SERVER-TRUSTED** because they used `--chain-backend local`.
That is correct labelling, not a bug. Re-run with `--chain-backend algorand`
to see TRUSTLESS ✓.

`test_browser_leaf_encoding_matches_the_backend` extracts `bvMerkleLeaf` and
`bvFoldProof` **verbatim from `frontend/index.html`** and runs them under Node
against the Python implementation. The shipped browser code is what is tested.

---

## 8. HTTP API

Blueprint `fl_bp` in `backend/fl_routes.py`, registered beside `auth_bp`.
Error envelope matches the rest of the API: `{"success": false, "error": "..."}`
with 200 / 400 / 404 / 500. Storage: `backend/data/fl_runs.json` (atomic
replace via a temp file).

| Method | Path | Role |
|---|---|---|
| POST | `/api/fl/round/commit` | Register one round after **re-deriving** the root from the submitted leaves. Rejects a claimed root that does not fold. `run_id` charset-checked (`^[A-Za-z0-9._=-]{1,200}$`). `attackers` / `metrics` type-checked; NaN/Inf dropped. |
| GET | `/api/fl/runs` | Every committed run, newest first (dropdown). |
| GET | `/api/fl/run/<run_id>` | Metadata + every round root, plus `meta`. |
| GET | `/api/fl/round/<run_id>/<r>` | Root, leaf count, txid, client ids, accepted/rejected. |
| GET | `/api/fl/proof/<run_id>/<r>/<client_id>` | Leaf, sibling path, root, indexer URL. |
| GET | `/api/fl/lineage/<run_id>` | Ordered roots + accept/reject + attackers + metrics. |

Indexer URL is only emitted for real Algorand txids (52-char base32). Mock and
local txids are hex block hashes, so `indexerUrl` is `null` and the UI reports
SERVER-TRUSTED.

`POST /api/fl/round/commit` is **unauthenticated**, matching other write routes
in this codebase. A deployment should put `@require_auth` on it.

The Federated tab is **read-only**. Training is launched from the CLI; publish
with `python3 -m fedverify.chain.push <run_dir>`.

---

## 9. Experiments

Every experiment is resumable: a cell whose `rounds.jsonl` already has `rounds`
lines is skipped. `--dry-run` lists pending vs done. `--smoke` is a tiny cell.

**Do not overlap exp3 with other CPU grids.** `anchor_ms` is a latency
measurement; parallel cells would contaminate it with scheduler queueing.
exp3 runs serially on purpose.

Calibrate **must** run before exp2 and exp5: they read τ from
`results/calibration/taus.json` and fail loudly rather than inventing one.

### 9.1 Status on disk (this snapshot)

| Exp | Grid | On disk |
|---|---|---|
| **exp1** privacy–utility | ε ∈ {0.5, 1, 2, 4, 8, inf} × K ∈ {5, 10} × {mnist, fmnist} × seeds {0, 1, 2} = **72 cells**, 30 rounds, α=0.5 | **48 complete**, 2 partial (`mnist_K5` eps 0.5 and 1, seed2, 2/30 rounds), **22 missing** (almost all seed2). Tables 1 and 1b are filled from the completed seeds. |
| **exp2** Byzantine | 5 attacks × 3 fractions × 6 aggregators × 2 ε × 3 seeds + clean cells = **576 cells** | **Empty** (`fedverify/results/exp2` does not exist). Tables 2 / 2b / 2_full are all `—`. |
| **exp3** chain overhead | K ∈ {5, 10, 20} × {mock, local}, 10 rounds, seed 0 | **Done** (6 cells). Table 3 filled. No `--algorand` confirmation cell in results. |
| **exp4** heterogeneity | MNIST α ∈ {0.1, 0.5, 1, inf} × ε ∈ {inf, 4, 1} × 3 seeds, plus MIT-BIH K ∈ {5, 10} × same ε × 3 seeds | **Empty**. Table 4 all `—`. |
| **exp5** DP × Byzantine | forensics only, attack ∈ {scaling, backdoor} at 20%, ε ∈ {inf, 8, 4, 2, 1, 0.5}, 3 seeds | **Empty**. Table 6 all `—`. |
| **table5** qualitative | `analysis/comparison.yaml` | **Done**. 18 of 56 cells are `?` because the cited paper does not state the property clearly. FedVerify “self-healing rollback” is **✗**. |
| **calibration** | ROC over labelled clean vs attacked rounds | Present: `results/calibration/taus.json` + ROC PNGs. Key `eps=1\|K=6` is empty. Policy used by experiments: `tau_fpr5`. |

### 9.2 Calibration (combined score, policy `tau_fpr5`)

Copied from `results/calibration/taus.json`:

| key | AUC | τ at FPR≤5% | TPR at that τ | FPR at that τ |
|---|---|---|---|---|
| eps=inf \| K=10 | 0.9832 | 2.6522 | 0.8667 | 0.0422 |
| eps=inf \| K=6 | 0.9822 | 2.4594 | 0.9000 | 0.0054 |
| eps=4 \| K=10 | 0.8552 | 2.5316 | 0.5933 | 0.0467 |
| eps=4 \| K=6 | 0.9066 | 1.6355 | 0.6667 | 0.0430 |
| eps=1 \| K=10 | 0.8148 | 2.9016 | 0.5667 | 0.0444 |
| eps=1 \| K=6 | *(empty)* | — | — | — |

The ~2.65 number in the docs is `eps=inf|K=10` combined `tau_fpr5`.

### 9.3 A finding that is **not yet a claim** (exp5)

In a Phase-5 demo, a scaling attacker became *easier* to detect as ε tightened:
attacker `s_norm` rose 3.72 (ε=inf) → 6.50 (ε=4) → 7.39 (ε=1). DP-SGD's
per-sample clipping narrows the honest-norm band, so a fixed multiplicative
attack lands further out. That is the **opposite** of “DP noise masks
attackers.” It was observed on 4 rounds with a τ borrowed from a K=6 smoke
calibration, so it is **not a paper claim** until full exp5 with per-(ε, K)
calibration confirms or kills it.

The same demo showed the flip side: with a stale τ, forensics rejected
nothing at ε=inf (attacker combined 3.95 vs τ 4.36). That is why τ is looked
up **per epsilon**.

### 9.4 Generated tables (copied as-is)

#### Table 1 — privacy / utility, final test accuracy (%)

Final test accuracy (%) — mean ± std over seeds

| ε | fmnist K=5 | fmnist K=10 | mnist K=5 | mnist K=10 | max realized ε |
|---|---|---|---|---|---|
| 0.5 | 73.87 ± 0.85 | 73.34 ± 0.51 | 91.07 ± 0.15 | 91.60 ± 0.03 | 0.499 |
| 1 | 78.55 ± 0.35 | 78.10 ± 1.05 | 95.27 ± 0.28 | 94.77 ± 0.12 | 0.999 |
| 2 | 80.72 ± 0.83 | 79.84 ± 0.59 | 96.47 ± 0.26 | 95.81 ± 0.05 | 1.999 |
| 4 | 81.50 ± 0.80 | 80.80 ± 0.62 | 96.93 ± 0.23 | 96.32 ± 0.16 | 3.999 |
| 8 | 81.92 ± 0.76 | 80.99 ± 0.47 | 97.13 ± 0.33 | 96.47 ± 0.05 | 7.997 |
| inf | 89.91 ± 0.21 | 89.50 ± 0.02 | 99.19 ± 0.01 | 99.06 ± 0.00 | — |

Source: `fedverify/results/tables/table1.md`.

#### Table 1b — privacy / utility, final macro-F1 (%)

Final macro-F1 (%) — mean ± std over seeds

| ε | fmnist K=5 | fmnist K=10 | mnist K=5 | mnist K=10 | max realized ε |
|---|---|---|---|---|---|
| 0.5 | 73.30 ± 0.89 | 72.31 ± 0.02 | 90.90 ± 0.16 | 91.47 ± 0.01 | 0.499 |
| 1 | 78.29 ± 0.44 | 77.62 ± 1.00 | 95.20 ± 0.30 | 94.73 ± 0.12 | 0.999 |
| 2 | 80.51 ± 0.93 | 79.51 ± 0.76 | 96.44 ± 0.29 | 95.77 ± 0.05 | 1.999 |
| 4 | 81.28 ± 0.94 | 80.51 ± 0.85 | 96.91 ± 0.24 | 96.30 ± 0.17 | 3.999 |
| 8 | 81.72 ± 0.92 | 80.72 ± 0.73 | 97.11 ± 0.34 | 96.45 ± 0.06 | 7.997 |
| inf | 89.82 ± 0.27 | 89.39 ± 0.08 | 99.18 ± 0.00 | 99.05 ± 0.00 | — |

Source: `fedverify/results/tables/table1b.md`.

#### Table 2 — Byzantine accuracy (exp2 empty)

Final test accuracy (%) — and ASR (%) where a backdoor is present — per attack × aggregator at attacker fraction 0.2, epsilon = inf, mean ± std over seeds

| attack | fedavg | krum | multikrum | trimmed_mean | median | forensics |
|---|---|---|---|---|---|---|
| none (clean) | — | — | — | — | — | — |
| label_flip | — | — | — | — | — | — |
| sign_flip | — | — | — | — | — | — |
| gaussian | — | — | — | — | — | — |
| scaling | — | — | — | — | — | — |
| backdoor | — | — | — | — | — | — |

Source: `fedverify/results/tables/table2.md`. **Do not invent numbers for a viva.**
If asked, say the grid is specified and the detector/aggregator code is tested;
the 576-cell run has not been executed yet.

#### Table 2b — detection F1 (exp2 empty)

Byzantine detection F1 per round while the attack is active, at attacker fraction 0.2, epsilon = inf. Final row: honest-client false-exclusion rate on the clean configuration.

| attack | fedavg | krum | multikrum | trimmed_mean | median | forensics |
|---|---|---|---|---|---|---|
| label_flip | n/a | n/a | — | n/a | n/a | — |
| sign_flip | n/a | n/a | — | n/a | n/a | — |
| gaussian | n/a | n/a | — | n/a | n/a | — |
| scaling | n/a | n/a | — | n/a | n/a | — |
| backdoor | n/a | n/a | — | n/a | n/a | — |
| false-exclusion (clean) | n/a | n/a | — | n/a | n/a | — |

`n/a` is structural (those rules do not make a per-client detection decision).
`—` means the cell was not run.

#### Table 3 — chain overhead (exp3 done)

Per-round cost of the chain commitment layer — mean ± std over rounds

| K · backend | digest (ms) | merkle (ms) | anchor (ms) | aggregate (ms) | bytes committed | rounds |
|---|---|---|---|---|---|---|
| K=5 · mock | 1.55 ± 0.12 | 0.20 ± 0.02 | 0.01 ± 0.00 | 1.09 ± 0.13 | 1,605,960 ± 0 | 10 |
| K=5 · local | 1.58 ± 0.07 | 0.21 ± 0.03 | 3.27 ± 3.02 | 1.18 ± 0.07 | 1,605,960 ± 0 | 10 |
| K=10 · mock | 3.21 ± 0.45 | 0.54 ± 0.06 | 0.01 ± 0.00 | 3.37 ± 0.40 | 3,211,920 ± 0 | 10 |
| K=10 · local | 3.01 ± 0.18 | 0.54 ± 0.05 | 5.26 ± 4.25 | 3.73 ± 0.71 | 3,211,920 ± 0 | 10 |
| K=20 · mock | 5.78 ± 0.38 | 1.81 ± 0.22 | 0.01 ± 0.00 | 11.29 ± 1.60 | 6,423,840 ± 0 | 10 |
| K=20 · local | 5.89 ± 0.77 | 1.76 ± 0.12 | 10.72 ± 10.78 | 10.84 ± 0.33 | 6,423,840 ± 0 | 10 |

Source: `fedverify/results/tables/table3.md`. Aggregation cost sits in the same
table so the overhead *ratio* is readable: anchoring is milliseconds; the
payload is the digest set.

#### Table 4 — heterogeneity × privacy (exp4 empty)

Macro-F1 % (accuracy % in parentheses) under heterogeneity and privacy, mean ± std over seeds. MNIST heterogeneity is synthesised by the Dirichlet alpha; MIT-BIH heterogeneity is inherited from a patient-disjoint split. Macro-F1 is primary: an always-N classifier scores 89.5 accuracy / 18.9 macro-F1 on MIT-BIH.

| dataset · split | ε=inf | ε=4 | ε=1 |
|---|---|---|---|
| mnist · α=0.1 | — | — | — |
| mnist · α=0.5 | — | — | — |
| mnist · α=1 | — | — | — |
| mnist · IID | — | — | — |
| mit-bih · K=5 (patient) | — | — | — |
| mit-bih · K=10 (patient) | — | — | — |

#### Table 5 — qualitative comparison (from `comparison.yaml`)

Qualitative comparison of blockchain-based and private federated-learning systems. ✓ = supported, ✗ = not supported, ? = not stated clearly enough in the cited paper to score (18 cells).

| system | DP guarantee | Byzantine-robust agg. | Per-update on-chain commitment | Client-verifiable inclusion proofs | L1 settlement | Layer-level forensics reuse | Self-healing rollback |
|---|---|---|---|---|---|---|---|
| DeepChain [Weng et al., IEEE TDSC 2021] | ✗ | ? | ✓ | ? | ✗ | ✗ | ✗ |
| LearningChain [Chen et al., 2018] | ✓ | ✓ | ✓ | ? | ✗ | ✗ | ✗ |
| BlockDFL [Qin et al., 2023] | ? | ✓ | ✓ | ? | ✗ | ✗ | ✗ |
| B-FL [Wang et al., IEEE IoT-J 2023] | ✓ | ? | ✓ | ? | ✗ | ✗ | ✗ |
| FLchain [Majeed & Hong, 2019] | ? | ? | ✓ | ? | ✗ | ✗ | ? |
| BAII [Zheng, 2024] | ? | ? | ? | ? | ? | ? | ? |
| NbAFL [Wei et al., IEEE TIFS 2020] | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| FedVerify (this work) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |

**Honest placement:** FedVerify does not claim to beat Krum or median on
accuracy. The claim is *competitive on accuracy while additionally producing
a verifiable per-client record they structurally cannot*. Overclaiming the
first loses a reviewer on the second.

#### Table 6 — DP × Byzantine (exp5 empty)

Interaction of differential privacy with Byzantine screening: FedVerify-Forensics under a 20% attacker fraction as epsilon tightens. Detection F1 is the per-round mean while the attack is active. tau is looked up per epsilon from calibration. Mean ± std over seeds.

| ε | scaling: acc | scaling: det F1 | backdoor: acc | backdoor: ASR | backdoor: det F1 |
|---|---|---|---|---|---|
| inf | — | — | — | — | — |
| 8 | — | — | — | — | — |
| 4 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 1 | — | — | — | — | — |
| 0.5 | — | — | — | — | — |

---

## 10. Demo runs (what the Federated tab shows)

Seven 8-round MNIST K=10 runs are committed in `backend/data/fl_runs.json`
(all `--chain-backend local`, so Verify proof step 4 is SERVER-TRUSTED).
Attackers, when present, are clients **8** and **9**.

| run id | what to notice |
|---|---|
| `…_epsinf_forensics_s0` **clean** | Nobody malicious. A small honest-exclusion rate is the *designed* cost of a threshold tuned to keep false alarms under 5%. |
| `…_epsinf_forensics_scaling0.2_s0` | Scaling + forensics. Attackers inflate their update; `s_norm` sees it. |
| `…_epsinf_scaling0.2_s0` | **Same attack, FedAvg, no defence.** Control. Nothing screened. |
| `…_epsinf_forensics_sign_flip0.2_s0` | Caught by **direction**. Size probe is blind. |
| `…_epsinf_forensics_label_flip0.2_s0` | Harder: the attacker *genuinely trained*, just on wrong labels. |
| `…_epsinf_forensics_backdoor0.2_s0` | Detection is weak — a backdoored update is a legitimately-trained update on poisoned data. Limitation, not a demo failure. |
| `…_eps4_forensics_scaling0.2_s0` | Scaling under DP (ε=4). Privacy noise and anomaly detection interfere. |

The teaching numbers in `WHAT_IS_FEDVERIFY.md` (75% caught on scaling/sign-flip,
~44% on label-flip, ~6% on backdoor, etc.) come from those demo cells. Quote
them as demo-run observations, not as Table 2.

---

## 11. Tests

Live collection on this machine:

```
python3 -m pytest fedverify/tests/  --collect-only -q   →  160 tests
python3 -m pytest backend/tests/    --collect-only -q   →   99 tests
                                                          ─────────
                                                           259 collected
```

BlockVerify baseline is **99** in `backend/tests/`. That number must not drop.
Slides that still say **43 tests** are stale — say 99 (BlockVerify) or 259
(BlockVerify + FedVerify collected), never 43.

FedVerify breakdown (collected):

| file | tests | what they pin |
|---|---|---|
| `test_aggregators.py` | 33 | registry, Krum/median/trim, forensics outliers, **no-τ raises**, τ from `taus.json`, Phase-1 record shape unchanged |
| `test_chain.py` | 27 | canon bytes, digest binds client/round/n, proofs, swap changes root, local tx types, commit-off default, **browser JS extracted from index.html** |
| `test_fl_routes.py` | 22 | commit rejects bad root / bad count / swapped order, lineage numeric order, proofs, 404 envelope, no route shadowing |
| `test_attacks.py` | 22 | attacker ids from seed, sign-flip / scaling / zero / gaussian, trigger in normalised space, ASR definition |
| `test_mitbih.py` | 17 | paced records dropped, DS1/DS2 disjoint, patient-whole partitions, always-N accuracy vs macro-F1, hospital heterogeneity |
| `test_privacy.py` | 12 | **σ over TOTAL steps**, accountant persists, ε=inf never imports Opacus, GroupNorm passes / BatchNorm raises, DP determinism |
| `test_data.py` | 10 | Dirichlet disjoint/complete, α=inf → IID, seed-stable |
| `test_runner.py` | 6 | artefacts, config.json completeness, same seed identical after `strip_timings` |
| `test_server.py` | 6 | FedAvg weighting, diag keys, unknown aggregator |
| `test_models.py` | 5 | flat-param roundtrip, no BatchNorm |

How FL testing was done, in one breath for a panel:

1. **Unit tests** for every mechanism (Merkle encoding, DP accounting, attacks,
   aggregators, MIT-BIH splits, HTTP routes).
2. **Cross-language test**: the *shipped* JavaScript Merkle helpers are pulled
   out of `index.html` and run under Node against Python.
3. **Experiment grids** write `rounds.jsonl` + `config.json`; tables are
   generated, never typed.
4. **Calibration** produces τ; forensics will not start without it.
5. **Regression**: existing BlockVerify 99 tests still pass.

```bash
python3 -m pytest backend/tests/ fedverify/tests/ -q
```

---

## 12. Federated tab (dashboard + viva walkthrough)

Open `http://localhost:8080` (not `0.0.0.0` — that is a non-secure context and
`crypto.subtle` is undefined). Sidebar → **Federated**.

The tab is visualisation-only. It does not launch training.

### 12.1 Dropdown

Each option is a committed run. The label encodes dataset, K, α, ε, aggregator,
attack, fraction, seed. Example:
`demo_mnist_K10_a0.5_epsinf_forensics_scaling0.2_s0`
= MNIST, 10 clients, Dirichlet 0.5, no DP, forensics, 20% scaling, seed 0.

Refresh reloads `GET /api/fl/runs`. Empty / down / stale are distinct states
(an unreachable API is not rendered as “no runs committed yet”).

### 12.2 Banner

Plain English: which clients are malicious, what the attack does, which defence
is running, and **τ with its provenance**. Read this first. If τ was hand-typed,
it shows amber and the detection numbers are not meaningful.

### 12.3 Accuracy per Round

Is the model still learning under attack? Rising and levelling off is healthy.
For heart data, watch **macro-F1**, not accuracy.

### 12.4 Accept / Reject Lineage

One square per client per round:

| colour | meaning |
|---|---|
| **bright green** | malicious client, correctly rejected (true positive) |
| **red** | malicious client that **got through** (false negative) |
| **amber** | honest client wrongly excluded (false positive) |
| **dark green** | honest client accepted (true negative) |

Malicious rows are labelled. A perfect run is bright green on those rows and
dark green everywhere else. Underneath, a sentence states the outcome in words.

`flDetection()` is the single TP/FP/FN implementation, shared by the stat cards
and the comparison table so the two views cannot disagree. F1 follows sklearn's
zero-division convention: 0 when a detector caught nothing, “—” only when the
rule never decides.

### 12.5 Round Commitments

One row per round: Merkle root (click to copy), how many updates went in, who
was screened out, anchoring transaction, **Verify proof** button.

### 12.6 Verify proof

Four visible steps (see §7). Demo runs → SERVER-TRUSTED at step 4 because
`backend` is `local`. The button uses `data-*` attributes and event
delegation; run ids are never interpolated into executable markup.

### 12.7 Compare runs

Pick Run A vs Run B. Accuracy on one axis, two lineage grids, metric-by-metric
Δ. This is the paper's central claim on one screen: defence cost vs who was
excluded.

Limitation: mismatched round counts **pad the shorter series** rather than
aligning on round number.

### 12.8 Teaching animation (“How One Round Works”)

Local training → screening → averaging → on-chain commitment. Caption says
**z-scores are illustrative, not this run’s numbers.** Per-client
`diag.scores` are still not exposed by the API.

### 12.9 FAQ copy already in the tab

- What FL is, in hospital language.
- Register/Verify = finished artefact. Federated = model still being built.
- How to publish a run (`runner --commit` then `chain.push`).

---

## 13. Panel talk-track

### 13.1 Literature / related work

The niche in one sentence (from `docs/RELATED_WORK.md`):

> BlockVerify = a hash-manifest committed on a public blockchain (trustless,
> via client-verified Merkle inclusion proofs) + statistical localization of
> *which layer* and *what class* of tampering occurred.

FedVerify sits in the blockchain-FL thread. Prior systems write *to* a ledger,
usually permissioned, and the commitment is consumed by the protocol rather
than by the client. Two capabilities are almost universally missing:

1. A participant can verify **its own** contribution against a root read from
   a **public, permissionless** chain.
2. The **accept/reject decision is itself part of the public record.**

Krum and median blunt attacks; neither can tell you *who* was excluded or
*why*. Table 5 is the comparison; `?` means “the paper did not say clearly,”
not a guess.

Baselines implemented: FedAvg (McMahan 2017), Krum / Multi-Krum (Blanchard
2017), coordinate-wise median and trimmed mean (Yin 2018). Scaling attack is
Bagdasaryan et al. (2020). DP-SGD is Abadi et al. (2016) via Opacus. NbAFL
(Wei 2020) is the closest privacy–utility comparison point for Table 1.

Adaptive attacks (Fang, Baruch “A Little Is Enough”) are **cited and not
implemented**.

### 13.2 Research gaps RG1–RG5

Open with: detect → localise → don’t trust the admin → tiny damage still
matters → same problem when many hospitals train together.

| Gap | One word | What existing work does | What we built | Where in the demo |
|---|---|---|---|---|
| **RG1** | Where? | Whole-file hash is a smoke alarm | Per-layer hash + Merkle | Verify, Compare, Merkle tree |
| **RG2** | Not ownership | Watermarks prove *mine* | We do **integrity**, not watermarking | We do not compete with FedIPR |
| **RG3** | Don’t trust us | A DB the operator can edit | Algorand Testnet | Ledger, Algo TxID |
| **RG4** | Tiny but lethal | File-level yes/no is too coarse | Layer grain | Compare / forensics |
| **RG5** | Many parties | Round-level provenance only | Same detector one level up + per-round Merkle | Federated tab |

Close: *The gap is not “nobody hashed a model.” It is: no lightweight way to
say which part changed, on a ledger the operator cannot silently edit, at the
precision bit-flip attacks require — including when the model is still being
trained by many parties. BlockVerify is RG1–RG4. FedVerify is RG5.*

Pushbacks:

- “Isn’t IPFS / git enough?” → RG3: the admin can rewrite it.
- “Store the full model on chain?” → too big; we store fingerprints and prove
  inclusion.
- “Does Forensics always catch attackers?” → No. Detection is statistical.
  The Merkle root is the hard guarantee.

### 13.3 Objectives O1–O7

Pipeline: hash layers → tree → chain → find the layer → prove a DB would fail
→ tests → (add-on) federated rounds.

| Obj | Promise | Fills |
|---|---|---|
| **O1** | Hash every layer (one leaf per layer) | RG1, RG4 |
| **O2** | Fold into a Merkle tree → one root | RG1 |
| **O3** | Put that root on Algorand **Testnet** | RG3 |
| **O4** | On mismatch, walk the tree and name the layer | RG1, RG4 |
| **O5** | Show why a blockchain beat a database | RG3 |
| **O6** | Automated tests | evidence |
| **O7** | Same idea for federated rounds (add-on) | RG5 |

O1–O6 are BlockVerify. O7 is FedVerify. Call O7 an **add-on** so the original
six are not heard as incomplete.

**Do not quote 43 tests.** Live BlockVerify suite is 99. With FedVerify,
259 collected.

### 13.4 Experiments slide

Say what is **done** (Table 1 privacy–utility, Table 3 commitment cost,
Table 5 positioning, calibration, demo lineage) and what is **specified but
not yet run** (Tables 2, 4, 6). Missing cells are `—` on purpose.

### 13.5 Important code, if she asks you to point at a file

1. `_manifest_leaf` — `backend/app.py:1562` — one leaf encoding for models
   and for FL.
2. `fedverify/chain/commitment.py` — imports that leaf; digest binds
   client / round / n.
3. `fedverify/core/privacy.py` — σ over total steps.
4. `fedverify/aggregators/forensics.py` — imports `eval_lib`; no default τ.
5. `frontend/index.html` — `bvMerkleLeaf` / `bvFoldProof` / `flVerifyProof`.
6. `backend/fl_routes.py` — server recomputes the root; does not trust the
   caller.

Simple version: *We fingerprint each hospital’s update, fold the fingerprints
into one number, write that number on a public chain, and use the same
“does this look weird compared to its neighbours?” test we already used on
layers.*

---

## 14. Guarantees and non-guarantees

**Guaranteed**

1. *Inclusion.* A client can prove its update was in round *r*, against a root
   read from a public ledger. Forging this requires a SHA-256 second preimage.
2. *Non-equivocation.* Once anchored, the aggregator cannot present a different
   round history to different parties without the roots disagreeing on chain.
3. *Order and multiplicity.* Reordering, dropping, or duplicating a client
   changes the root; the anchored leaf count closes duplicate-last ambiguity.
4. *History integrity.* Periodic Bitcoin checkpoints over the round-root
   sequence make rewriting an old round detectable (when `--btc-checkpoint`
   is used).
5. *Privacy.* A per-client (ε, δ) bound over the whole run, correctly composed,
   when DP is on.

**Explicitly not guaranteed**

1. That FedAvg arithmetic was performed correctly over the published set.
2. That an accepted update is benign. Screening is statistical and evadable.
   Detection is best-effort; **commitment is not**.
3. That the training data was real.
4. Patient-level or user-level privacy.
5. Recovery. The lineage identifies a bad round. Nothing rolls it back.
   Table 5 marks self-healing rollback **✗**.
6. Registration-time honesty. A malicious aggregator could anchor a wrong root
   *at commit time*. Mitigable by client countersignatures; not implemented.
7. Reliable backdoor detection. A backdoored update looks like honest training
   on poisoned data.
8. Adaptive attackers.

---

## 15. Rules that governed the work (CLAUDE.md)

| ID | Rule |
|---|---|
| A1 | Same seed ⇒ identical `rounds.jsonl` **after dropping timings**. |
| A2 | Install torch from the **CPU-only** index. Plain PyPI pulls ~4 GB of CUDA. |
| A5 | Results are only byte-reproducible at a **fixed** `--torch-threads`. |
| A6 | `--commit` off by default so Phase-3 does not change Table 1 cells. |
| Numbers | Never type table numbers by hand. Missing → `—`. Unclear paper → `?`. |
| Scope | Additive only on existing BlockVerify. Zero lines deleted from the single-artefact path. |
| Git | The human commits. Do not `git add` / `git commit` unless asked. |
| App ID | 764828342, from `algo_app.json` / `load_app_id()`. |
| exp3 | Serial; do not overlap with other CPU grids. |

Install:

```bash
pip install --break-system-packages --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install --break-system-packages tqdm opacus wfdb
```

---

## 16. How to run everything

From the repository root.

```bash
# Experiments  (calibrate MUST precede exp2 and exp5)
python3 -m fedverify.experiments.exp1_privacy_utility           --jobs 8 --threads 2
python3 -m fedverify.analysis.calibrate
python3 -m fedverify.experiments.exp2_byzantine                 --jobs 8 --threads 2
python3 -m fedverify.experiments.exp3_chain_overhead            # serial on purpose
python3 -m fedverify.experiments.exp4_heterogeneity             --jobs 6 --threads 2
python3 -m fedverify.experiments.exp5_dp_byzantine_interaction  --jobs 6 --threads 2

python3 -m fedverify.analysis.make_tables      # tables 1,2,2b,3,4,5,6
python3 -m fedverify.analysis.plots            # figures 1-6
python3 -m pytest backend/tests/ fedverify/tests/ -q
```

Publish a demo run to the dashboard:

```bash
python3 -m fedverify.core.runner --dataset mnist --num-clients 10 \
    --rounds 8 --commit --chain-backend local
python3 -m fedverify.chain.push <the run folder it prints>
```

Use `--chain-backend algorand` to unlock TRUSTLESS ✓ at proof step 4.

---

## 17. Known issues and parked work

### 17.1 FedVerify still open

- exp2 / exp4 / exp5 grids not executed; those tables are `—`.
- exp1 seed2 mostly missing (22 cells + 2 partial).
- Calibration key `eps=1|K=6` is empty.
- Federated compare **pads** mismatched round counts.
- Per-client `diag.scores` not in the API; walkthrough z-scores are teaching
  numbers.
- `POST /api/fl/round/commit` unauthenticated.
- Algorand FL note-only (no contract global state).
- Bitcoin checkpoints default to mock; live testnet path unused.
- Combined = max(...) may not be optimal (`s_coord` won on smoke AUC).
- `tau_coord` not calibrated.
- Attacks are non-adaptive; `zero` attack not in the exp2 grid.
- No self-healing rollback (accurate ✗ in Table 5).
- MIT-BIH class Q almost empty; lead 0 only; privacy is per-beat.
- Round metrics reach the chart only if the commit payload includes them.
- `plots.py` fig3 draws ROC from three operating points in `taus.json`, not a
  full curve.
- `docs/API_REFERENCE.md` did not originally document `/api/fl/*` (Phase-6
  remit).

### 17.2 Website leftovers that touch the Federated demo

- D3 CDN can block the **Chain Proof** Merkle renderer (“Tree renderer
  unavailable”). Federated proof uses the in-page JS helpers, not D3.
- Remaining XSS on Register / Verify / Registry (Federated is escaped).
- CDN scripts have no SRI.
- Mobile sidebar is still a wrap-chip cloud.

### 17.3 Pre-existing BlockVerify landmines (found in Phase 0, **not** “fixed
inside FedVerify”)

See `INVENTORY.md` §13.

- `_get_pow_chain()` reads `name` / `hash` but the registry stores
  `modelName` / `modelHash` → chain `register` txs can carry `"Unknown"` / `""`.
- `bitcoin_client` reports a 32-byte OP_RETURN script but broadcasts the
  64-byte ASCII hex string.
- `backend/requirements.txt` pins drift from installed versions; `bit` is
  installed but unlisted.
- Stale “82 pytest” in `docs/TAMPER_DETECTION.md`.
- Stray files `backend/=2.8.0`, `backend/=4.1.0`.
- Algorand private key in `backend/data/algo_wallet.json` (gitignored, on disk).

### 17.4 Fixed during the Phase-6 polish (so they are *not* still broken)

- ASR now included in the push metrics payload (backdoor stat card).
- Commit payload type-checks `attackers` / `metrics`; NaN/Inf dropped.
- `run_id` charset-checked; Verify button uses `data-*` + delegation.
- Proof panel step 4 no longer always reported round 1’s backend.
- `/api/fl/run/<id>` returns `meta`.
- Down / stale / empty are distinct UI states.
- `flDetection()` shared by cards and compare table.

---

## 18. Related files outside `fedverify/`

| Path | Role |
|---|---|
| `docs/ARCHITECTURE.md` § FedVerify | Where the add-on sits |
| `docs/RELATED_WORK.md` §5 | FL / DP / blockchain-FL citations |
| `docs/EVALUATION.md` §7 | Table and figure index; no hand-typed numbers |
| `docs/TAMPER_DETECTION.md` | Single-artefact companion (layer level) |
| `evaluation/eval_lib.py` | Probe port imported by forensics |
| `backend/app.py` | Canonical Merkle + `verify_batch` + Blueprint registration |
| `backend/fl_routes.py` | Six FL routes |
| `backend/data/fl_runs.json` | Dashboard runs |
| `frontend/index.html` | Federated tab, Merkle JS, FAQ, tour step 5 |

---

## 19. What this file is not

- Not the paper (`fedverify/paper/main.tex`).
- Not SAFE-MAS (a later, separate 11-phase HITL project; not started here).
- Not a claim that Tables 2, 4, and 6 exist as numbers. They exist as
  **schemas** waiting for grids.
- Not permission to quote 43 tests, App ID 758544892, or a hand-set τ of 6.0.

---

*End of dump. Companion short reads: [`WHAT_IS_FEDVERIFY.md`](WHAT_IS_FEDVERIFY.md)
and [`FEDERATED.md`](FEDERATED.md).*
