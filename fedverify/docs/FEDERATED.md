# FedVerify — Federated Training with Verifiable Round Commitments

*For a non-technical explanation of the same system, see
[`WHAT_IS_FEDVERIFY.md`](WHAT_IS_FEDVERIFY.md).*

*Companion to [`docs/TAMPER_DETECTION.md`](../../docs/TAMPER_DETECTION.md), in the same
form. That document covers a **single artefact** registered once. This one covers a
**sequence of updates from many parties**, where the artefact is never final and the
contributors do not trust each other.*

---

## 1. Threat model

Three adversaries, deliberately separated, because the mechanisms that answer them are
different and conflating them is how systems overclaim.

| Adversary | Capability | Answered by |
|---|---|---|
| **A1 — curious server** | Sees every client update; wants to infer training data | DP-SGD (§4). *Not* the blockchain: anchoring a digest hides nothing. |
| **A2 — Byzantine clients** (≤ f of K) | Send arbitrary updates: scaled, flipped, noise, backdoored | Robust aggregation (§5) |
| **A3 — equivocating aggregator** | Reports one round history to one party and a different one to another; silently drops or rewrites an update | Per-round Merkle commitment anchored on a public L1 (§3) |

**Assumed, not defended:** clients are authenticated out of band; the aggregator is
*semi-honest with respect to the training computation itself* (it runs FedAvg on the set it
publishes — we prove **which updates entered a round**, not that the arithmetic was done
correctly; that is proof-of-learning's problem, and it is expensive and contested); the
network is not the adversary; ≤ f of K clients are Byzantine, with f known.

---

## 2. What is committed

Never raw weights. Every stage downstream of local training operates on **deltas**
`δ_k = θ_k^local − θ^global`.

```
δ_k  ──canon──▶  float32, little-endian, C-contiguous bytes
     ──digest─▶  d_k = SHA-256( JSON({client_id, num_samples, numel, round}) ‖ bytes )
     ──leaf───▶  ℓ_k = SHA-256( JSON([k, client_id, d_k]) )
     ──tree───▶  R_r = MerkleRoot(ℓ_0 … ℓ_{K−1}),  with  n = K  anchored beside it
```

Three details that are load-bearing rather than incidental:

- **The digest header binds identity, round and sample count.** Without it, a delta could
  be replayed by another client, or into another round, and still verify.
- **The leaf binds POSITION.** Swapping two clients within a round changes every leaf from
  that position on, so the root changes. Order is part of the commitment.
- **The leaf count `n` is anchored with the root.** Duplicate-last Merkle trees are
  ambiguous without it: a verifier free to choose level widths can fold different leaf
  multisets to the same root (the CVE-2012-2459 shape). This is the same mitigation the
  layer-manifest anchor uses with `lmr`/`lc`.

The Merkle construction is **not reimplemented** for FL. `fedverify/chain/commitment.py`
imports `compute_layer_merkle` and `merkle_inclusion_proof` from `backend/app.py` — the
same functions the browser already re-verifies for layer manifests. One implementation,
one encoding, one set of tests.

---

## 3. Protocol

For round *r*:

1. Each selected client trains locally and returns `δ_k`.
2. **Attacks, if any, are applied here** — before the commitment. The chain must bind what
   the client *actually sent*; committing a cleaned-up delta would make the lineage a
   record of what we wished had happened.
3. The aggregator computes `d_k`, `ℓ_k`, `R_r`, `n`.
4. `R_r` and `n` are anchored:
   - `mock` — content-addressed fake txid, no I/O, deterministic (experiments)
   - `local` — `fl_client_update` + `fl_round_commit` transactions mined into the
     repository's PoW chain (tests)
   - `algorand` — a 0-ALGO self-transaction whose note carries
     `{"frr": R_r, "flc": n, "rnd": r, "run": run_id}`
5. Screening runs, survivors are FedAvg'd, and the accept/reject split is stored with the
   round so the lineage is auditable.
6. Every `checkpoint_every` rounds, a Bitcoin OP_RETURN commits
   `MerkleRoot(R_1 … R_r)` — one cheap write that pins the entire history, so a server
   cannot quietly rewrite round 3 after round 30.

### Client-side verification (the part that matters)

```
GET /api/fl/proof/<run_id>/<r>/<client_id>   →   { index, digest, leaf, proof, root, indexerUrl }
```

The browser then, in order:

1. **recomputes** `ℓ = SHA-256(JSON([index, client_id, digest]))` — it does not trust the
   served `leaf`;
2. folds the sibling path and checks it reaches `root`;
3. fetches the anchoring transaction **directly from `testnet-idx.algonode.cloud`**,
   bypassing this server entirely, and reads `frr` / `flc` from the note;
4. accepts only if `frr == root`, `flc == leafCount`, and the fold succeeded.

Anything less is reported as **SERVER-TRUSTED**, explicitly, with the reason. A root
mismatch is reported as **ROOT MISMATCH**, never silently downgraded.

`fedverify/tests/test_chain.py::test_browser_leaf_encoding_matches_the_backend` extracts
`bvMerkleLeaf` and `bvFoldProof` **verbatim from `frontend/index.html`** and runs them under
Node against the Python implementation. The shipped browser code is what is tested, not a
re-typed copy of it.

---

## 4. Differential privacy

Opacus DP-SGD per client, RDP accountant, `(ε, δ)` with `δ = 1e-5`.

**The accounting is over TOTAL steps, not per round.** σ is solved once per client for
`rounds × steps_per_round`, and the accountant object persists across rounds so
composition accumulates. Re-solving σ each round and reporting the per-round ε is the
classic DP-FL error: it under-reports the true budget by roughly the number of rounds. The
guard is `test_noise_multiplier_is_solved_over_TOTAL_steps`, which spies on
`get_noise_multiplier` and asserts the step count it was called with.

**What the guarantee is.** Each client's *reported* ε bounds the privacy loss of that
client's contribution across the whole run, under the standard DP-SGD assumptions
(Poisson sampling, per-sample clipping to `C`, Gaussian noise σ·C).

**What it is not.** The unit of privacy is a **training example**. On MIT-BIH that is a
*beat*, not a *patient* — and a patient contributes hundreds of beats. A per-patient
guarantee would need group privacy or per-patient clipping. This is a real limitation and
is stated rather than glossed.

No BatchNorm anywhere (GroupNorm only): Opacus cannot handle it, and BatchNorm running
statistics leak across clients regardless.

---

## 5. FedVerify-Forensics

BlockVerify localises a tampered layer by running a robust median/MAD outlier probe over
the **weights** of a model. The observation this aggregator rests on is that the same probe
family works one level up: within a round, the population is the **K client deltas** rather
than the D weights of one tensor, and a Byzantine client is an outlier in that population
the way a poisoned layer is an outlier in a model.

The probes are **imported** from `evaluation/eval_lib.py` — the byte-identical port of the
deployed in-browser detector (`docs/EVALUATION.md` §3) — not re-implemented.

| score | statistic (robust z across clients) | direction | catches |
|---|---|---|---|
| `s_norm` | `‖δ_k‖₂` | two-sided | scaling (inflates), zero/free-rider (collapses) |
| `s_dir` | `cos(δ_k, median δ)` | one-sided low | sign-flip, label-flip |
| `s_coord` | share of coordinates beyond `tau_coord` | one-sided high | sparse, targeted edits |
| `s_health` | NaN, ±Inf, `\|w\|>100`, entropy, constant run | hard flag | corrupted or degenerate updates |

`combined = max(s_norm, s_dir, s_coord)`, mirroring the deployed detector's max-robust-z
operating point. A client is rejected if a hard health flag fires **or** `combined > τ`;
survivors are FedAvg'd. Health-flagged clients are excluded from the population statistics
too, so one NaN delta cannot poison the median every other client is measured against.

**Sign-flip is why there is more than one score.** A sign-flipped delta has an *identical
norm*, so `s_norm` is blind to it (measured: 0.05) while `s_dir` fires hard (2022, cosine
−0.975). A norm-only detector would miss it completely.

### τ is never hardcoded

`docs/EVALUATION.md` §5.2 documents exactly this mistake one level down: the hand-set layer
threshold `z > 8` had **FPR 0.49** on real heavy-tailed weights. So τ comes from
`fedverify/analysis/calibrate.py`, which builds a ROC over labelled clean and attacked
rounds and writes `results/calibration/taus.json` keyed by (ε, K). Every run records both
`tau` and `tau_source` (file path, key, and policy) in its `config.json`. Constructing the
aggregator without a τ **raises**; there is deliberately no default.

---

## 6. Guarantees, and what they are not

**Guaranteed**

1. *Inclusion.* A client can prove its update was in round *r*, against a root read from a
   public ledger. Forging this requires a SHA-256 second preimage.
2. *Non-equivocation.* Once anchored, the aggregator cannot present a different round
   history to different parties without the roots disagreeing on chain.
3. *Order and multiplicity.* Reordering clients, dropping one, or duplicating one changes
   the root; the anchored leaf count closes the duplicate-last ambiguity.
4. *History integrity.* Periodic Bitcoin checkpoints over the round-root sequence make
   rewriting an old round detectable.
5. *Privacy.* A per-client (ε, δ) bound over the whole run, correctly composed.

**Explicitly NOT guaranteed**

1. *That the aggregation arithmetic was performed correctly.* We prove which updates
   entered a round, not that FedAvg was computed faithfully over them.
2. *That an accepted update is benign.* Screening is statistical and evadable by
   construction — an adaptive attacker who keeps its delta inside the honest robust-z band
   will pass. Detection is best-effort; **commitment is not**.
3. *That the training data was real.* Nothing here prevents a client from training on
   fabricated data.
4. *Patient-level or user-level privacy.* See §4.
5. *Recovery.* The lineage identifies a bad round. Nothing rolls it back. `table5` marks
   FedVerify's "self-healing rollback" as ✗, and that is accurate.
6. *Registration-time honesty.* As with layer manifests, a malicious aggregator could
   anchor a wrong root *at commit time*. Mitigable by having clients countersign the root;
   not implemented.

---

## 7. Limitations

- **Attacks evaluated are non-adaptive.** The attacker does not know the defence. An
  adaptive attacker constrained to the honest band is the natural strong baseline and is
  not implemented.
- **`combined = max(...)` may not be optimal.** On the smoke calibration `s_coord` alone
  scored a higher AUC (0.982) than combined (0.913), because the max lets whichever
  sub-score is noisiest lift honest clients. To be re-checked on the full calibration
  before any claim is made either way.
- **`tau_coord` is not itself calibrated** — it has a documented default and is recorded,
  but only the decision threshold τ comes from the ROC.
- **Krum is a selection rule, not a detector.** Vanilla Krum keeps one client, so reading
  its "rejected" list as detections would be a category error; `table2b` marks it `n/a`.
- **Commitment cost scales with K and model size.** `table3` reports it; the payload is the
  digest set, so it is small, but the aggregation itself is not free.
- **`POST /api/fl/round/commit` is unauthenticated**, matching the other write routes in
  this codebase. A deployment should apply `@require_auth`.

---

## 8. Reproducing

```bash
python3 -m fedverify.experiments.exp1_privacy_utility          # Table 1
python3 -m fedverify.analysis.calibrate                        # taus.json (required next)
python3 -m fedverify.experiments.exp2_byzantine                # Tables 2, 2b
python3 -m fedverify.experiments.exp3_chain_overhead           # Table 3
python3 -m fedverify.experiments.exp4_heterogeneity            # Table 4
python3 -m fedverify.experiments.exp5_dp_byzantine_interaction # Table 6
python3 -m fedverify.analysis.make_tables                      # all tables
python3 -m fedverify.analysis.plots                            # all figures
```

Every run writes `rounds.jsonl` + `config.json` containing every parameter that influenced
it. Same seed and same `--torch-threads` reproduces `rounds.jsonl` byte-for-byte after
dropping wall-clock fields (CLAUDE.md amendments A1, A5).
