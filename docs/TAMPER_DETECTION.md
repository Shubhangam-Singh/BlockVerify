# BlockVerify — Tamper Detection: Formal Specification

*This document specifies the complete tamper-detection logic implemented in BlockVerify,
in a form suitable for adaptation into an academic paper. Section numbers map to the
implementation in `frontend/index.html` (forensic engine) and `backend/app.py`.*

---

## 1. Threat Model

**Assets.** A machine-learning model `M`, represented as an ordered set of named layers
`M = ⟨(ℓ₁, W₁), …, (ℓₙ, Wₙ)⟩`, where `ℓᵢ` is a layer identifier and `Wᵢ` its weight
tensor (arbitrary nesting depth).

**Adversary.** After registration, an attacker with write access to the model artifact
(supply-chain position: storage, CDN, model hub, CI pipeline) may:

| ID | Attack class | Formal action |
|----|-------------|---------------|
| **T1** | Weight poisoning / backdoor injection | modify values inside some `Wᵢ` |
| **T2** | Topology poisoning (rogue layer) | insert a new layer `(ℓ*, W*)` |
| **T3** | Layer excision | delete an existing layer `(ℓᵢ, Wᵢ)` |
| **T4** | Execution-graph reordering | permute the layer sequence |

**Trust assumptions.**
1. The model is *benign at registration time* (the system proves non-modification,
   not benignity — "garbage in, garbage out").
2. The Algorand ledger provides an immutable, publicly verifiable commitment to the
   full-file hash (note transaction + smart-contract global state).
3. The layer-hash manifest is currently stored in the application registry
   (server-trusted); see §8 Limitations for the trustless extension.
4. Hashing is performed **client-side** (Web Crypto API); raw weights never leave the
   verifier's device.

---

## 2. Commitments Recorded at Registration

For a submitted model file `F`:

1. **File commitment** `H = SHA-256(F)` — anchored on Algorand
   (0-ALGO note transaction and `model_id → H` in TEAL contract global state).
2. **Layer manifest** `Λ = { ℓᵢ ↦ SHA-256(canon(Wᵢ)) }` — per-layer digests, where
   `canon(·)` is the canonical JSON serialization of the tensor.
3. **Layer sequence** `π = [ℓ₁, …, ℓₙ]` — the *explicit* signed layer order.
   (Order is semantic: it encodes the execution graph. It is recorded as an explicit
   array because RFC 8259 JSON objects are formally unordered and serializers may
   legally re-sort keys — relying on object key order is unsound.)
4. **Manifest commitment (trustless localization).** A Merkle root over `(Λ, π)`
   jointly, anchored on-chain in the registration note (`"lmr"`) together with the
   leaf count (`"lc"`), and best-effort in contract global state (key `id:l`):

   ```
   leaf_i = SHA-256( JSON([i, ℓᵢ, Λ(ℓᵢ)]) )        (position ∥ name ∥ digest)
   root   = binary Merkle fold, duplicate-last on odd levels,
            parent = SHA-256(hex(left) ∥ hex(right))
   ```

   Binding the *index* into each leaf makes the root commit to the sequence `π`,
   not just the set: any reordering changes every affected leaf. The anchored
   leaf count `n` precludes the duplicate-last mutation ambiguity
   (CVE-2012-2459-class): two leaf sets of different length cannot share an
   accepted commitment. The JSON leaf encoding (compact separators, UTF-8,
   no ASCII-escaping) is byte-identical between the Python producer and the
   JavaScript verifier (validated empirically).
5. Proof-of-Work anti-spam: registration requires a nonce `s` with
   `SHA-256(H ∥ s)` having ≥ 3 leading zero hex digits (Hashcash-style).

---

## 3. Verification Protocol

### Level 1 — File integrity (detection)
The verifier re-hashes the local file: `H' = SHA-256(F')`.
`H' = H ⟺` the file is bit-identical to the registered artifact.
This is the *sound and complete* detector: any single-bit change flips `H'`
(collision resistance of SHA-256). **Level 1 answers "was it tampered?"**

### Level 2 — Deep layer forensics (localization + characterization)
Invoked when `H' ≠ H` and a layer manifest exists. Given the uploaded model's
computed manifest `Λ' = { ℓ ↦ SHA-256(canon(W'ℓ)) }` and sequence `π'`,
**Level 2 answers "where, and what kind of tampering?"**

#### 3.0 Trustless manifest validation (server not trusted)
Before Level 2 uses the served manifest `(Λ, π)`, the verifier validates it
against the on-chain commitment — the server is *untrusted* at verification time:

1. `GET /api/manifest-proof/<id>` returns the manifest, the claimed root, and a
   Merkle **inclusion proof** (sibling path) per layer — all untrusted data.
2. The verifier fetches the registration transaction **directly from the public
   Algorand indexer** (browser → `testnet-idx.algonode.cloud`, bypassing the
   application server) and extracts the anchored root `lmr` and leaf count `lc`.
3. For every layer it recomputes `leaf_i = SHA-256(JSON([i, ℓᵢ, Λ(ℓᵢ)]))` and
   folds the sibling path; the result must equal the on-chain root, and the
   served layer count must equal `lc`.

*Security argument.* Accepting a manifest entry `(i, ℓ, λ)` that was not
committed at registration requires producing a sibling path folding to the
anchored root — i.e., a SHA-256 second preimage. Both practical forgeries are
detected and demonstrated in tests: (a) a tampered manifest entry with the
original proofs fails proof folding; (b) a fully self-consistent *substituted*
manifest (internally valid proofs over a forged root) fails the on-chain root
comparison. On success the UI reports **TRUSTLESS ✓**; otherwise it degrades
explicitly to *server-trusted* mode (root unavailable, invalid proofs, or
`ROOT MISMATCH`).

#### 3.1 Topology classification
With `S = dom(Λ)` (registered names, ordered by `π`) and `C = dom(Λ')`
(uploaded names, ordered by `π'`):

```
added    = [ ℓ ∈ C : ℓ ∉ S ]                → T2 (rogue layer)
removed  = [ ℓ ∈ S : ℓ ∉ C ]                → T3 (excision)
modified = [ ℓ ∈ S ∩ C : Λ(ℓ) ≠ Λ'(ℓ) ]     → T1 (weight tampering)
intact   = [ ℓ ∈ S ∩ C : Λ(ℓ) = Λ'(ℓ) ]
```

**T4 (reordering).** Let `π|_C` be the registered sequence projected onto the common
set `S ∩ C`, and `π'|_S` the uploaded sequence projected likewise. Both are
permutations of the same set, hence equal length. Then:

```
reordered ⟺ ∃ i : (π|_C)[i] ≠ (π'|_S)[i]      (element-wise comparison)
```

*Correctness properties (unit-tested):*
- Pure insertion or deletion never triggers T4 (projections stay aligned).
- Any transposition of two common layers triggers T4.
- Element-wise comparison avoids string-join delimiter collisions
  (e.g. names `{"a", "a>a"}` under a `join('>')` scheme).
- The comparison uses the *explicit* sequence `π`, never JSON key order.

#### 3.2 Weight-tamper localization (per `modified` layer)
Only the uploaded tensor `W'ℓ` is available (the original weights are not stored),
so localization uses *robust statistics on the tampered copy*:

**Robust outlier detection (median/MAD).**
For the flattened numeric values `x₁…x_m` of `W'ℓ`:

```
med   = median(x)
MAD   = 1.4826 · median(|xᵢ − med|)         (consistency constant for Gaussian data)
scale = MAD              if MAD > 1e−9
        σ(x)             elif σ > 1e−9      (degenerate-MAD fallback)
        1                otherwise
zᵢ    = |xᵢ − med| / scale
outlier ⟺ zᵢ > 8                            (report top-k by z, k = 6)
```

Median/MAD is chosen over mean/σ because the estimator must remain stable *in the
presence of the very outliers it is trying to find* (breakdown point 50% vs 0%).

**Tensor health probes** (independent tamper signals; recursively over nesting):

| Probe | Flag condition | Rationale |
|---|---|---|
| NaN count | > 0 | NaN injection destroys inference |
| ±Inf count | > 0 | overflow / sabotage payloads |
| Extreme magnitude | \|w\| > 100 | normalized weights should not reach this |
| Shannon entropy | Ĥ < 0.35 (32-bin, normalized by log₂32), m > 16 | constant-block / degenerate tensors |
| Constant run | ≥ 8 identical consecutive values | copy-fill payloads |

**Characterization decision:**
```
outliers found          → "backdoor payload"  (report indices, values, z-scores)
else health flags       → "degenerate/poisoned tensor"
else                    → "subtle drift / unauthorized fine-tuning"
```
Note: Level 2 characterization is *heuristic*; detection soundness comes from the
hash (Level 1 / §3.1), which no perturbation can evade.

#### 3.3 Verdict and integrity score
```
CRITICAL  ⟺ |modified| + |added| > 0
WARNING   ⟺ otherwise and |removed| > 0 or reordered
SECURE    ⟺ no anomalies

U     = |intact| + |modified| + |added| + |removed|          (layer union)
score = round(100 · |intact| / U) − (5 if reordered else 0), floored at 0
```

---

## 4. Two-File Comparison Mode (both artifacts available)

When the verifier holds *both* model versions (Compare tab), exact
element-wise analysis replaces statistics:

- **Recursive deep diff** walks arbitrarily nested tensors and reports the exact
  coordinate path of every changed scalar (e.g. `[1][1]: 0.5 → 999.0`), with
  per-layer aggregates: changed-count, %, max Δ, mean Δ; largest-|Δ| samples first.
- Topology classification and T4 detection as in §3.1 (using both files' orders).
- Layer "mass" (leaf-scalar count) is computed for visualization weighting.

---

## 5. Complexity

For a model with `n` layers and `m` total scalars:
- Level 1: `O(|F|)` hashing (streamed, Web Crypto).
- §3.1 classification: `O(n)` with set lookups.
- §3.2 per-layer: sorting dominates → `O(mℓ log mℓ)`; probes `O(mℓ)`.
- §4 deep diff: `O(m)` scalar comparisons.
- Batch mode: files hashed in parallel over a Web-Worker pool
  (`min(8, hardwareConcurrency)`), single `O(k)` server round-trip for `k ≤ 20` files.

---

## 6. What the System Guarantees (and What It Does Not)

**Guarantees** (under §1 assumptions):
- *Completeness of detection:* any post-registration modification of the file is
  detected (SHA-256 second-preimage resistance).
- *Localization:* every layer-level change is attributed to exactly one of T1–T4
  (the four classes partition all manifest-level differences).
- *No false positives at Level 1/§3.1:* identical artifacts always verify
  (deterministic hashing over canonical serialization).

**Non-guarantees:**
- Cannot certify the model was benign *before* registration.
- §3.2 characterization (backdoor vs drift) is heuristic — a stealthy attacker can
  evade the *statistical* probes, but never the hash.
- Layer manifest is server-trusted today (see §8).

---

## 7. Empirical Validation (current)

- 38-assertion engine test suite: 16 scenarios covering all four classes, combined
  attacks, shape changes, false-positive guards, determinism (3× identical runs).
- End-to-end browser tests on generated demo artifacts
  (`demo_tools/generate_demo_model.py`: weight backdoor @ index 42, rogue layer,
  excision, reordering, clean control).
- Backend: 82 pytest cases including PoW, chain-cache, Merkle, batch endpoints.

---

## 8. Known Limitations / Future Work

1. **Trustless layer manifests — IMPLEMENTED (§2.4, §3.0).** `MerkleRoot(Λ, π)` is
   anchored in the registration note transaction (+ contract global state) and the
   browser verifies per-layer inclusion proofs against the root it reads directly
   from the public indexer. Remaining caveats: (a) models registered before this
   protocol have no anchor and degrade to server-trusted mode; (b) version updates
   (`add-version`) do not yet re-anchor a new manifest root; (c) registration-time
   trust is still assumed (the server could anchor a wrong root *at registration* —
   mitigable by having the registrant's client recompute and countersign the root).
2. **Canonicalization.** `canon(·)` is JSON re-serialization; a formal treatment
   should use a canonical JSON standard (RFC 8785 JCS) to rule out
   serialization-ambiguity attacks (whitespace, float formatting, key escaping).
3. **Format coverage.** Layer-level inspection currently requires JSON models;
   binary formats (safetensors, ONNX, GGUF) need chunked/segment hashing.
4. **Integer-like layer names.** JavaScript object semantics reorder integer-like
   keys; layer identifiers are assumed non-numeric strings (true of
   TF/Keras/PyTorch naming). The explicit `π` array mitigates this on the stored
   side; uploaded-file order for such names remains engine-dependent.
5. **Statistical thresholds — CALIBRATED (see [EVALUATION.md](EVALUATION.md)).**
   ROC analysis on real HuggingFace checkpoints (BERT-tiny, ALBERT) shows the
   outlier probe separates backdoors from benign fine-tuning well (AUC 0.98 for
   Δ≥1), but the hand-set `z > 8` is **miscalibrated for heavy-tailed real
   weights** (clean layers already reach mean max-z ≈ 9.6 ⇒ FPR ≈ 0.49).
   Recommended operating point: `τ ≈ 16` (Youden) or `τ ≈ 23` (FPR ≤ 5%).
   The `|w| > 100` probe has FPR ≈ 0 on real weights (0/27 clean layers trip it).
   Detection is magnitude-driven, not count-driven; subtle broad edits evade the
   statistics but remain caught by the Level-1 hash. Remaining work: recalibrate
   the shipped threshold, and calibrate entropy binning (quantile vs range-based).
