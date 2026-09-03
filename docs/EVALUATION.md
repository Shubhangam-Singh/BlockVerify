# BlockVerify — Empirical Evaluation

*Reproducible artifact: [`evaluation/`](../evaluation). Run `python3 evaluation/evaluate.py`
(deps: `numpy`, `matplotlib`). Figures + `results.json` land in `evaluation/out/`.
All numbers below are seeded (`numpy.default_rng(0)`) and regenerate identically.*

## 1. What is being evaluated (and what is not)

BlockVerify is a **two-level** detector, and the two levels must be evaluated
differently:

- **Level 1 — file/layer hashing** is the tamper **detector**. Any post-registration
  modification flips a SHA-256 digest, so **detection recall = 1.0 by construction**
  (collision resistance). There is no ROC to report for Level 1 — it is sound and
  complete. This is the guarantee the blockchain anchor makes trustless (§ trustless
  manifest protocol).
- **Level 2 — statistical probes** (robust median/MAD outlier z, Shannon entropy,
  extreme magnitude) do **localization + characterization**: given a layer whose hash
  *already* changed, do the statistics say *malicious backdoor* or *benign fine-tuning*?
  This is a genuine classification task with a ROC — and is what we evaluate.

**Positives** = layers with an injected backdoor. **Negatives** = clean layers **and**
benignly fine-tuned layers (additive Gaussian noise on all weights, `σ_rel ∈ {0.05…0.5}`).
Separating malicious from benign is the whole point — a detector that flags every
fine-tune as an attack is useless.

## 2. Data — real checkpoints (not toy JSON)

Two real, pretrained, publicly-downloaded checkpoints (safetensors, parsed with numpy —
no toy tensors, no synthetic distributions):

| Checkpoint | Source | Weight layers used |
|---|---|---|
| `google/bert_uncased_L-2_H-128_A-2` (BERT-tiny) | HuggingFace (251k downloads) | 19 |
| `albert-base-v2` | HuggingFace | 8 |

**27 real weight matrices, ~1.9 M parameters under test** (≥1024 elements, ≥2-D, float).

## 3. Evaluated code == deployed code (cross-validation)

The Python probes are a **byte-identical port** of the deployed JavaScript
(`frontend/index.html`: `bvFindOutliers`, `bvLayerHealth`). `evaluation/cross_validate.py`
extracts the JS functions **verbatim**, runs them (Node) on a real BERT weight slice with
injected backdoors, and compares against the Python port:

```
real layer: bert.pooler.dense.weight  slice=(128,128)  n=16384
  OK  max robust-z    py=2285.194113473985   js=2285.194113473985
  OK  scale (1.4826·MAD), median, entropy, extremes, outlier count … all identical
CROSS-VALIDATION: PASS — Python evaluation == deployed detector
```

So the evaluation measures the *shipped* detector, to full float64 precision — not a
re-derivation of it. (This surfaced and fixed a real deployment bug: `bvFindOutliers`
previously iterated only the top array level and thus found nothing on N-dimensional
tensors; it now flattens recursively like the health probe.)

## 4. Threat model → attacks

| Attack | Simulation | Target probe |
|---|---|---|
| **T1 weight poisoning / BadNets triggers** | set `k` random weights to `±Δ` | outlier z, extremes |
| **Low-entropy / low-rank poisoning** | overwrite a contiguous fraction with a constant | entropy |
| **Benign fine-tuning (hard negative)** | additive `N(0, σ_rel·σ_layer)` on all weights | — (must *not* flag) |

Grids: `Δ ∈ {0.1 … 500}` (9 pts) × `k ∈ {1,4,16,64,256}` (5 pts), 3 repeats/cell.

## 5. Results

### 5.1 Clean-weight baseline (the crucial reality check)
| Statistic | Clean real weights | Deployed threshold |
|---|---|---|
| max robust-z | mean **9.6**, max **25.5** | flag `z > 8` |
| Shannon entropy (norm.) | mean 0.649, min **0.356** | flag `H < 0.35` |
| layers with `|w| > 100` | **0 / 27** | flag `|w| > 100` |

**Key finding:** real trained weights are **heavy-tailed**, so a clean layer's max
robust-z is already ≈ 9.6 — *above* the deployed `z > 8`. The naive threshold therefore
**over-flags benign layers** (FPR ≈ 0.49). This is the single most important result of
the evaluation and a concrete contribution: *robust-z thresholds calibrated on Gaussian
assumptions are wrong for real weight distributions.*

### 5.2 Outlier probe — ROC and **calibration**  ([roc_probes.png](../evaluation/out/roc_probes.png))
| Metric | Value |
|---|---|
| AUC (all attack strengths) | **0.903** |
| AUC (non-trivial attacks, Δ≥1) | **0.984** |
| Deployed `z>8` operating point | TPR 0.914, **FPR 0.493** ← miscalibrated |
| Calibrated `τ*` @ FPR≤0.05 | **τ* ≈ 23** → TPR 0.716 |
| Youden-optimal `τ` | **τ ≈ 15.6** → TPR 0.779, FPR 0.074 |

**Recommendation for the paper/deployment:** replace the hand-set `z>8` with `τ ≈ 16`
(Youden) or `τ ≈ 23` (FPR≤5%). Separability is excellent (AUC 0.98 for real attacks);
only the operating point needed calibration.

### 5.3 Detection boundary  ([heatmap_detection.png](../evaluation/out/heatmap_detection.png))
Recall of the calibrated probe by attack magnitude Δ (at τ*):

| Δ | 0.1 | 0.5 | 1.0 | 2.0 | 5.0 | ≥10 |
|---|---|---|---|---|---|---|
| recall | 0.10 | 0.31 | 0.63 | **0.96** | **1.00** | 1.00 |

Two findings: (a) the probe reliably localizes backdoors of magnitude **Δ ≳ 2× the layer
scale**; (b) detection is **invariant to k** (# weights modified) — the max-z probe keys
off *peak magnitude*, not attack breadth, so a broad low-magnitude attack evades it.
**Crucially, every attack that evades Level 2 is still detected by Level 1 (the hash).**
The statistical layer adds *localization*, not *detection coverage* — exactly the
defense-in-depth story the two-level design intends.

### 5.4 Entropy and extremes probes
- **Entropy** (constant-block attacks): AUC 0.686; at `H<0.35`, TPR 0.065, **FPR 0.000**.
  High specificity, low sensitivity — a niche probe for heavy low-entropy poisoning.
  Clean entropy min (0.356) sits right at the threshold, explaining the conservatism;
  it is also range-sensitive (a single extreme value collapses the estimate).
- **Extremes** (`|w|>100`): **0/27** clean layers trip it ⇒ FPR ≈ 0 on real weights;
  a zero-false-positive probe for large-magnitude injections.

### 5.5 Performance & cost
| Quantity | Measured / cited |
|---|---|
| SHA-256 throughput | ≈ 1–2 GB/s single-core (`hashlib`; browser WebCrypto comparable) |
| Manifest-hash + Merkle root (BERT-tiny, 46 tensors) | **10.2 ms + 0.20 ms** (local commitment overhead) |
| Algorand cost / registration | **≈ 0.003 ALGO** (3 txns × 0.001 min fee) |
| Algorand finality | ≈ 2.9 s single-block (published testnet constants) |

(Throughput/overhead are measured; Algorand cost/finality are protocol constants, not a
network micro-benchmark — a live-network latency study is future work.)

## 6. Headline takeaways for the paper
1. **Detection is trustless and complete** (hash + on-chain Merkle-anchored manifest);
   the statistical layer is *localization*, and we evaluate it as such — a framing that
   preempts the "statistics can be evaded" objection (they can; the hash cannot).
2. **The outlier probe is strong (AUC 0.98 on real backdoors) but the default threshold
   is miscalibrated for heavy-tailed real weights** — we quantify this and give a
   calibrated operating point. Honest, and a genuine finding.
3. **Registration is cheap**: sub-cent, sub-3-s, ~10 ms local overhead — practical at
   repository scale.

---

## 7. Federated evaluation (FedVerify)

The tables below are generated by `python3 -m fedverify.analysis.make_tables` and the
figures by `python3 -m fedverify.analysis.plots`. **No number in any of them is typed by
hand**; missing cells render `—` and are listed on stderr, and an incomplete run is skipped
rather than averaged in as though it had finished.

| | what it answers | source |
|---|---|---|
| `table1` / `table1b` | privacy–utility: accuracy and macro-F1 vs ε | `exp1_privacy_utility` |
| `table2` / `table2_full` | robustness: accuracy and ASR per attack × aggregator | `exp2_byzantine` |
| `table2b` | Byzantine detection F1, and honest false-exclusion when nobody attacks | `exp2_byzantine` |
| `table3` | commitment cost: digest / merkle / anchor / aggregate ms, bytes | `exp3_chain_overhead` |
| `table4` | heterogeneity × privacy, macro-F1 leading | `exp4_heterogeneity` |
| `table5` | qualitative positioning vs prior systems | `analysis/comparison.yaml` |
| `table6` | does DP noise blind Byzantine screening? | `exp5_dp_byzantine_interaction` |

Figures `fig1`–`fig6` mirror tables 1, 2, the calibration ROC, 3, 4 and 6.

Two methodological points carry over from the layer-level evaluation above:

- **Thresholds are calibrated, never assumed.** §5.2 above showed the hand-set `z > 8` had
  FPR 0.49 on real weights. The client-level threshold τ is therefore produced by
  `analysis/calibrate.py` from a ROC over labelled clean and attacked rounds, and every run
  records `tau_source` so a number in a table can be traced to the calibration entry that
  produced it.
- **Accuracy is reported only where it means something.** On MIT-BIH, 89.47% of beats are
  class N, so a model that always answers N scores **89.47% accuracy and 18.89%
  macro-F1**. `table4` prints macro-F1 first and accuracy in parentheses.

Threat model and guarantees: [`fedverify/docs/FEDERATED.md`](../fedverify/docs/FEDERATED.md).
