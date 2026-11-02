# FedVerify-Survival: the complete guide, Phases 0 to 12, explained from scratch

*Written on 2026-10-16 for anyone who wants to understand this project end to end, including readers with no
background in machine learning or blockchains, and updated the same day for Phase 12 (the final wrap-up). Every
number below was copied from the results files on that date (`fedverify/results/…`,
`fedverify/survival/paper/RESULTS_DRAFT.md`, the tables); a test (`test_phase12_wording.py`) checks that the two
key-comparison tables in Part D match the generated ones. If you re-run anything, those generated files are the
source of truth, not this guide.*

The earlier part of FedVerify (federated learning on MNIST images, before this study) has its own guide:
[`FEDVERIFY_COMPLETE.md`](FEDVERIFY_COMPLETE.md). This guide covers the medical study built on top of it.

---

## Contents

- [Part A. The ideas you need first](#part-a-the-ideas-you-need-first)
- [Part B. The project at a glance](#part-b-the-project-at-a-glance)
- [Part C. What was done, phase by phase](#part-c-what-was-done-phase-by-phase)
- [Part D. What the results say](#part-d-what-the-results-say)
- [Part E. How to run and use everything](#part-e-how-to-run-and-use-everything)
- [Part F. Limitations and open decisions](#part-f-limitations-and-open-decisions)
- [Part G. Glossary](#part-g-glossary)
- [Part H. Questions an examiner may ask](#part-h-questions-an-examiner-may-ask)
- [Appendix. Progress record and file index](#appendix-progress-record-and-file-index)

---

## Part A. The ideas you need first

### A1. The problem in one paragraph

Several hospitals each hold patient records. Together they could train a better prediction model than any of them
alone, but patient data legally cannot leave a hospital. **Federated learning** solves that: each hospital trains
on its own patients and sends only *what it learned* (a model update), never the patients. Three new worries appear:

1. **Privacy.** An update can still leak something about the patients it was trained on.
2. **Poisoning.** One dishonest hospital can send a deliberately bad update to damage the shared model.
3. **Trust in the record.** The server that combines the updates could quietly drop, swap or rewrite who sent what.

This project builds and measures defences for all three on two real medical datasets, and compares itself with a
published paper that did something similar.

### A2. Machine learning in two minutes

- A **model** is a formula that turns a patient's **features** (age, blood pressure, tumour size, …) into a
  prediction (for example the chance of heart disease).
- **Training** adjusts the formula's numbers (its **weights**) so its predictions match known outcomes (**labels**).
- A model can memorise its training patients instead of learning general patterns. That is **over-fitting**: great on
  the patients it has seen, poor on new ones. So a model is always judged on **held-out** patients it never saw.

### A3. The three kinds of model used

| Model | In plain words | What a hospital sends each round |
|---|---|---|
| **Logistic regression** (LR) | A weighted sum of the features, squashed into a probability between 0 and 1. Simple and fast. | A **delta**: how much it wants each weight to change. |
| **Random forest** (RF) | Many decision trees ("if age > 60 and …") that vote. | The new trees it grew this round. |
| **XGBoost** | Trees built one after another, each fixing the mistakes of the ones before ("boosting"). | The new trees it grew this round. |

Training happens in **rounds**: every round each hospital trains a bit, sends its update, the server combines the
updates into the **global model**, and sends it back. This study uses 30 rounds.

### A4. Federated learning, and the three ways to compare it

- **Local only:** each hospital trains alone on its own patients. This is what happens without collaboration.
- **Centralized:** all patients pooled in one place. Not allowed in real life, but it shows what pooling would give.
- **Federated:** the hospitals collaborate through the server without sharing patients.

How the server combines updates (the **aggregation rule**):

- **FedAvg:** average the hospitals' deltas, weighted by how many patients each has. **FedSGD** is the simplest form:
  one gradient step per hospital per round.
- **Tree bagging** (random forest): keep every hospital's new trees in one big forest.
- **FedXGBBagging** (XGBoost): every hospital boosts new trees from the same starting point; the server keeps all
  the trees and moves the global model by the **average** of the hospitals' corrections (Phase 7 fixed this; it used
  to add them up, see Phase 7).

### A5. Survival analysis (the TCGA part)

For cancer patients we want to know *when* a death happens, not only *whether*. Two complications:

- **Censoring:** many patients are still alive at their last check-up. We know they survived *at least* that long,
  but not their final outcome. Throwing them away would bias the results.
- **AFT model** (accelerated failure time): a model that predicts a patient's survival *time*. XGBoost has an AFT
  mode, used here ("XGBoost AFT").

The score for survival models is the **C-index** (concordance index): take every pair of patients where we know who
died first; the C-index is the share of those pairs the model ranks correctly. 0.5 is a coin flip, 1.0 is perfect.

### A6. How models are measured

| Metric | Meaning | Coin-flip value |
|---|---|---|
| **Accuracy** | Share of patients predicted correctly. Misleading when one outcome is rare (see the trap below). | depends |
| **F1** | Balances catching the true cases with not raising false alarms. | depends |
| **ROC-AUC** | Chance that a random sick patient gets a higher risk score than a random healthy one. | 0.5 |
| **C-index** | The same idea for survival times, with censoring handled. | 0.5 |
| **Log-loss** | How confident and wrong the probabilities are; lower is better. Used for the training curves. | |

**The majority-class trap.** On TCGA 84% of patients are alive, so a "model" that predicts *everyone survives* is 84%
accurate while knowing nothing. That is why this study never ranks models on accuracy near that rate (Phase 7) and
focuses on ROC-AUC and the C-index.

**Thresholds.** A model gives a probability; to say "yes/no" you pick a cut-off (threshold). This study picks it on
the training patients, as the value that gives the best F1, and uses the same rule for accuracy and F1 (Phase 7).

### A7. Cross-validation: how the results are made trustworthy

- **5-fold cross-validation:** split the patients into 5 groups (folds). Train on 4, test on the 5th, and rotate, so
  every patient is tested exactly once by a model that never saw them. Those predictions are the **out-of-fold**
  predictions.
- **Stratified:** every fold keeps the same mix of hospitals and outcomes.
- **Repeated 5 times** with different random shuffles (seeds 0–4). Results are reported as **mean ± standard
  deviation over the 5 repeats**.
- **Pooled:** the out-of-fold predictions of all 5 folds are combined and each metric is computed once per repeat.

### A8. Differential privacy (DP)

**DP-SGD** protects the individual training examples inside a hospital's update: every patient's contribution is
first **clipped** (capped in size), then random **noise** is added. The guarantee is written **ε** (epsilon): smaller
ε means stronger privacy but more noise. This study tests ε = 1, 2, 4, 8 and "∞" (no DP), with δ = 1e-5.

DP only applies to the logistic model here. Trees are not trained with gradients, so DP-SGD does not apply to them;
those cells are shown as "n/a" rather than faked.

### A9. The blockchain pieces, in plain words

| Piece | What it is | Why it is used |
|---|---|---|
| **Hash** (SHA-256) | A fingerprint of some data. Change one bit and the fingerprint changes completely. | Detects any change. |
| **Content ID** (CID) | A hash written in the IPFS naming style. | Names an update by its content. |
| **Digital signature** (Ed25519) | Proof that the holder of a private key approved a message. | Proves which hospital sent an update. |
| **Merkle tree** | Hashes combined pairwise up to one **root** hash for the whole round. | One small root commits to every update. |
| **Inclusion proof** | The few hashes needed to show one update is inside a root. | Anyone can check their update was counted. |
| **Proof-of-work block** | A block of records whose hash must start with zeros, found by trial and error ("mining"). | Rewriting history means re-mining. |
| **Anchoring** | Writing the round root into a block. | Fixes the record in time. |
| **Algorand Testnet** | A public blockchain (test network). | Optional public anchor that nobody here controls. |

This project uses the **existing BlockVerify proof-of-work chain** (one node, on one machine, difficulty 2), not a
real network of computers. It gives tamper evidence of the records, not agreement among many independent parties.

### A10. Attacks and defences

**Poisoning attacks** (a dishonest hospital):

| Attack | What the bad hospital does |
|---|---|
| sign flip | Sends its update reversed, pushing the model the wrong way. |
| scaling | Multiplies its update so it dominates the average. |
| label flip | Trains on flipped outcomes (sick ↔ healthy). |
| label flip (trees) | Grows its trees on flipped outcomes. |
| random trees | Sends well-formed trees that carry no real information. |

**Screening** finds outliers among the hospitals' updates each round. FedVerify-Forensics (logistic models) and a
"probe-set screen" (tree models) turn each hospital's difference from the others into a **robust z-score** (built from
the median and the median absolute deviation, so one outlier cannot hide itself). A hospital is rejected when its
score exceeds a threshold **τ** (tau). τ is never typed in: it is **calibrated** on separate runs as the value that
catches the most attackers while wrongly rejecting at most 5% of honest rounds.

**Chain attacks** (from the reference paper):

| Attack | In plain words | Defence here |
|---|---|---|
| Double spending | Using an update twice: sending a second version, or replaying an old one. | One update per hospital per round; the update must name this round; each transaction counter is used once. |
| Transaction malleability | Changing the wrapping of an update (timestamp, id, order) without changing the model. | The signature covers every field; each Merkle leaf includes its position. |
| Endpoint compromise | Submitting as a hospital you are not. | Signatures must verify under the key registered for that hospital; registration is write-once. |

### A11. Comparing two methods fairly (statistics)

Two methods run on exactly the same folds and repeats, so they are compared **in pairs**: for each repeat, take the
difference of their scores. To see whether that difference is real or luck, the **bootstrap** re-draws the patients
2000 times (with replacement, within each hospital) and recomputes the difference each time. The middle 95% of those
2000 differences is the **95% confidence interval** (CI). Since Phase 11 the rule is simple: **method A beats method B
only if the whole interval is above 0**; if it contains 0 they are "not distinguishable at this sample size".

No p-values are reported: with only 5 repeats, an exact test can never give a p-value below 0.0625.

Three wording rules sit on top of the interval rule (Phases 7 and 12), and the generators apply them everywhere:

- **"Not ranked".** If the screen of "Ours" throws out more than 5% of honest hospitals' updates on a run with no
  attacker, a gain of Ours is not called a win: it may come from dropping an unusual (skewed) hospital rather than from
  catching attackers.
- **"Highest score", never "best".** The top score is compared with the runner-up by the same interval; if the interval
  contains 0, the text says "highest score (not distinguishable from X)".
- **Federated beating centralized on TCGA is always explained in the same sentence:** what the Phase 8 diagnosis found
  about the centralized model, and how federated compares with the step-matched row (Phase 9).

---

## Part B. The project at a glance

### B1. Goal and reference paper

Replicate and extend **Myrzashova et al., "Safeguarding Patient Data-Sharing: Blockchain-Enabled Federated Learning in
Medical Diagnostics", IEEE Transactions on Sustainable Computing 10(1), 2025.** That paper trains on chest X-rays,
logs updates on a simulated Ethereum-style proof-of-work chain, encrypts updates homomorphically, and reports three
chain attacks (Tables III–IV and Figs 5–10). This project redoes its tables and figures on two medical datasets,
with three model families, differential privacy instead of encryption, a real commit layer on the existing
BlockVerify chain, and an extra defence the paper does not have: screening poisoned updates.

### B2. The two datasets

**UCI heart disease** (label: heart disease present). "Paper mode" drops the columns `ca`, `thal` and `slope` and then
the incomplete rows, as the reference heart paper did: 920 raw rows, 740 analysed, 383 with heart disease.

| Hospital | Patients | With heart disease |
|---|---|---|
| Cleveland | 304 | 139 |
| Hungary | 260 | 98 |
| VA Long Beach | 130 | 101 |
| Switzerland | 46 | 45 (only 1 healthy patient) |

Features (23 after encoding): age, resting blood pressure, cholesterol, maximum heart rate, ST depression, and
one-hot categories for sex, chest-pain type, fasting blood sugar, resting ECG and exercise angina.

**TCGA-BRCA** (breast cancer; label: death, plus time to death or last follow-up). The five largest source sites act as
the hospitals: 907 raw cases, 474 analysed patients, 76 deaths.

| Site | Patients | Deaths |
|---|---|---|
| BH | 145 | 45 |
| A2 | 98 | 11 |
| E2 | 87 | 6 |
| A8 | 76 | 4 |
| AR | 68 | 10 |

Features (30 after encoding): age at diagnosis, lymph nodes, tumour over 20 mm (each with a "missing" flag), and
categories for menopause status, ER, PR and HER2 receptor status, and histological type. Race is **off** by default
(it mostly stands in for access to care; `--include-race` turns it on, with a fairness note). Columns that would leak
the answer (survival time, follow-up days, dates, ids) are blocked by a hard guard.

**Preprocessing learns nothing from patients:** fixed scaling constants and a fixed list of category names, so no
hospital's statistics leak to another.

### B3. The methods (the rows of every table)

| Name in the code | What it is |
|---|---|
| `local_lr`, `local_rf`, `local_xgb` | Each hospital alone. (Heart: Switzerland gets no local model; 1 healthy patient is not enough.) |
| `central_lr`, `central_rf`, `central_xgb` | All patients pooled, with the same training budget as the federated version. |
| `fedsgd_lr`, `fedavg_lr` | Federated logistic regression. |
| `dp_fedavg_lr_eps{1,2,4,8,inf}` | FedAvg with DP-SGD at that ε. |
| `fed_rf`, `fedxgb_bagging` | Federated random forest and XGBoost. |
| `*_aft` | The XGBoost survival-time (AFT) version, TCGA only. |
| `chain_*` | The federated method with every update signed, committed and anchored ("Blockchain-FL" in the tables). |
| `ours_*` | `chain_*` plus screening (and DP ε = 4 for the logistic model): the full proposed system ("Ours"). |
| `ours_*_<attack>`, `…_noscreen` | The same under a poisoning attack, with and without the screen. |
| `central_xgb_steps` | A diagnostic centralized XGBoost added in Phase 9 (see there). |

Settings (identical for every method, fixed before any result was seen): 30 rounds, 5 folds, 5 repeats.
- Logistic regression: 3 local passes per round, batches of 32, learning rate 0.1, no momentum.
- Random forest: 10 trees per hospital per round, depth 5, at least 3 patients per leaf.
- XGBoost: 5 trees per hospital per round, depth 3, learning rate 0.1, subsample 0.8.
- DP: δ = 1e-5, clipping norm 1.0.

### B4. How the pieces fit together

```
 Hospital A ─┐   train locally (LR / RF / XGBoost, optional DP-SGD)
 Hospital B ─┤   serialize → SHA-256 digest → content ID → Ed25519 signature
 Hospital C ─┤
 Hospital D ─┘
        │  signed submission + update bytes
        ▼
 Aggregator (FoldCommitter)
   1. verify: registered key, valid signature, this round, fresh counter, digest matches bytes
   2. Merkle root over accepted digests, inclusion proof for each
   3. anchor the root: mine a block on the BlockVerify proof-of-work chain (optionally Algorand too)
   4. rebuild every update from the committed bytes
   5. screen (Forensics / probe screen, τ from the calibration file) → reject outliers
   6. combine the survivors (FedAvg / bagging / FedXGBBagging) → new global model → back to the hospitals
        │
        ▼
 Run folder: config.json, rounds.jsonl, commitments.jsonl, oof.jsonl, summary.json
        │
        ├── tables, figures, results draft, showcase page, report pack (generated, never typed)
        └── dashboard: push → Federated tab (lineage, in-browser proof check)
```

The six report diagrams (architecture, data-flow levels 0 and 1, use case, classes, one round) are in
[`docs/diagrams/`](../../docs/diagrams/).

### B5. Rules followed in every phase

- **No git commands**, no co-author trailers. You commit by hand.
- **Additive only:** existing BlockVerify and FedVerify behaviour is not rewritten; new code goes next to it.
- **Every number comes from a results file.** Tables, figures, the results draft, the page and DEMO.md are generated
  by code; nothing is typed by hand. A missing result shows as "—" and is listed.
- **No tuning on test data**, no hyperparameter changes after seeing results, no new models, attacks or datasets
  beyond what each phase asked.
- **Honesty over flattering numbers:** when something did not work or a request could not be met exactly, it is
  reported (see Phases 7, 8, 10 and 11).
- The command is `python3` (this machine has no `python`).

---

## Part C. What was done, phase by phase

Each phase lists **what was asked**, **what was built**, **how it works**, **what came out**, and **problems found
and fixed**.

### Phase 0 — Inventory and decisions

**Asked:** read the whole existing code base and data before writing anything; list what can be reused and what
would break.

**Built:** [`fedverify/survival/INVENTORY.md`](../survival/INVENTORY.md): a map of every reusable function (with
file and line), the data checks, the test baseline (259 tests passing), and twelve decisions (D1–D12) with
recommended defaults. Key ones:

- **D2** The `OS_MONTHS` unit check fails for 11 rows: pin those rows as known exceptions rather than drop them.
- **D3** Keep censored patients with zero follow-up, and say so.
- **D6** For trees, a hospital's "update" is the trees it adds that round (trees have no weights to subtract).
- **D7** `--seed N --repeats 5` means seeds N … N+4.
- **D8** Installing lifelines needed pandas downgraded (3.0.5 → 2.3.3); you ran the install.
- **D10** Hospital codes (BH, A2, …) are the only ids written to disk and chain; existing code that wants numbers gets
  a thin translation.
- **D12** Plain SGD without momentum, so "learning rate 0.1" means what it says.

You accepted every default.

### Phase 1 — TCGA data and the logistic family

**Built:** the TCGA cohort loader (`survival/data.py`), the data-independent encoder, the leakage guard, the
cross-validation protocol, the runner (`survival/runner.py`), and `local_lr`, `central_lr`, `fedsgd_lr`, `fedavg_lr`
(`survival/fl_lr.py`). Every run writes a run folder (`results/survival/<method>/seed0/`) with every parameter,
per-round records and the out-of-fold prediction of every patient.

**How it works:** a hospital's loss is weighted for class imbalance (deaths count more, by the ratio of survivors to
deaths in that hospital, computed locally). The F1 threshold is chosen on the training patients.

**Found:** a linear model on these 8 features is weak. On the same folds it lands near a standard scikit-learn
logistic regression, and age alone ranks patients better. This is a property of the data, not a bug; it is reported.
346 tests passing.

### Phase 2 — Differential privacy

**Built:** `dp_fedavg_lr` at ε = 1, 2, 4, 8 and ∞ (`survival/dp.py`), using the existing `core/privacy.py` and Opacus.
The noise level σ is solved once per hospital over all its training steps, and the privacy actually spent (realised
ε) is measured and logged; it always stays at or under the target.

**Found:**
- Clipping undoes most of the class weighting: a death's up-weighted gradient is clipped back to the same size as a
  survivor's. That is why DP costs TCGA so much (Part D).
- F1 barely separates methods on TCGA: most methods sit near the F1 of "flag everyone".
- Random-forest DP is not DP-SGD and was not faked (cells say "n/a"). 359 tests passing.

### Revision 2 and Phase 2b — Heart data, random forests, XGBoost, survival time

You then pasted "Revision 2", which replaced the remaining plan: add the heart dataset, XGBoost and the AFT survival
task, and replicate the reference paper's Tables III–IV and Figs 5–10. FPoW (named in Revision 2) was left
undefined, waiting for your guide.

**Built:**
- the heart loader with "paper" and "full" modes (`survival/heart.py`);
- random forests (`fl_rf.py`: local, centralized, federated);
- XGBoost (`fl_xgb.py`: local, centralized, FedXGBBagging);
- the AFT survival-time task on TCGA (times under 1 day clipped to 1 day: 11 patients);
- a heart "full mode" sensitivity run.

**How FedXGBBagging works:** every hospital starts from the same global model, boosts 5 new trees on its own patients,
and sends only those trees. A hand-made test checks the combination rule and that after round r the model holds
exactly hospitals × 5 × r trees. 392 tests passing.

### Phase 3 — Committing every update to the chain

**Built:** the commit layer (`survival/chain_round.py`, `serialize.py`, `chain/contentid.py`, `chain/signing.py`) and
the `chain_*` methods. Every round:

1. Each hospital's update becomes canonical bytes (no pickle; a header names run, round, hospital and size, so it
   cannot be replayed elsewhere).
2. It gets a SHA-256 digest and a content ID, and is **signed** with that hospital's Ed25519 test key.
3. The aggregator **verifies** key, signature, round, transaction counter, digest and content ID.
4. Accepted digests are committed as a **Merkle root** with an inclusion proof for each hospital, and the root is
   **anchored** (mock by default; `local` mines a real proof-of-work block).
5. The server rebuilds every update **from the committed bytes** and aggregates those. So with no attack, `chain_*`
   learns exactly what the plain method learns (tested bit for bit).
6. Before each round, every hospital checks that the previous block is still valid and still holds the previous root;
   if not, it skips the round and the skip is recorded.

One run per dataset was mined on the proof-of-work chain (150 blocks each) to measure latency, throughput and send
rate as the reference paper defines them. 422 tests passing.

### Phase 4 — Attacks, screening and calibration

**Chain attacks** (`survival/chain_attacks.py`): the reference paper's three attacks, run against the real commit
layer for 100 rounds per attack and dataset, with the variant and target drawn from the seed. Each attack also runs
with exactly its defences switched off. That shows the defence tests are not vacuous: the attacks then succeed.

**Poisoning** (`survival/poison.py`): one hospital per repeat is malicious from round 1, chosen from the repeat seed
by the existing `attacker_ids` rule. Logistic attacks reuse the existing sign-flip, scaling and label-flip code; two
new tree attacks were added.

**Screening:** FedVerify-Forensics (existing, unchanged) for logistic updates. For trees, a new probe screen
(`survival/screening.py`): each hospital's new trees predict a public **probe set** (512 random rows drawn from the
feature definitions alone, no patient data); hospitals whose predictions sit far from the median get a high robust
z-score.

**Calibration** (`analysis/calibrate_survival.py`): 80 short runs at a separate seed (1000) where every hospital in
turn plays the attacker, with screening that scores but never rejects. From the labelled scores it builds a ROC curve
and picks τ at a false-alarm rate of at most 5%. The values are written to `results/calibration/taus_survival.json`.
Every screening run reads its τ from that file and records where it came from.

**Problem fixed:** heart and TCGA cells share method names, so calibration folders collided; they were moved to
per-dataset folders. 465 tests passing.

### Phase 5 — Tables and figures

**Built:** `analysis/make_fl_healthcare_tables.py` and `make_fl_healthcare_figures.py`, driven by
[`survival/table.yaml`](../survival/table.yaml), which says which run fills which row. Every table is written as
Markdown and LaTeX; every figure as PNG (300 dpi, Times New Roman), PDF and the CSV of plotted numbers, plus
`captions.md`. The reference paper's own Table III and IV rows are copied unchanged from reference CSVs and shown as
"reported", never compared.

**Problems fixed:** a real LaTeX compile found unmapped symbols (δ, −) and a "[19]" after a line break that LaTeX read
as a length; both fixed and tested. The run itself is reproducible: the same seed gives byte-identical files.

### Phase 6 — Documentation, one-command experiments, results draft

**Built:**
- [`DESIGN_NOTES.md`](../survival/DESIGN_NOTES.md) (how everything works and why);
- [`THREAT_MODEL.md`](../survival/THREAT_MODEL.md) (who is trusted for what, and what is not defended);
- [`survival/README.md`](../survival/README.md);
- `survival/experiments.py`, one command per experiment (92 runner commands today, after the later phases);
- `analysis/make_results_draft.py`, which writes [`paper/RESULTS_DRAFT.md`](../survival/paper/RESULTS_DRAFT.md):
  Experimental Setup, Results and Limitations, with every comparison sentence produced by one fixed rule and logged
  in `RESULTS_DRAFT.comparisons.json`;
- `make_fl_healthcare_all.py`, which rebuilds everything in one command.

476 tests passing.

**The showcase page** (built right after, on your request): `analysis/make_showcase.py` makes an interactive tour of
the study. It walks through one real committed round, from training to the mined block, and charts every result. It
is published privately at https://claude.ai/artifact/SgRLdv9VeynvSqRb4HrBkR, and an offline copy is at
`fedverify/results/showcase/index.html`.

### Phase 7 — Audit fixes

An external audit raised four points. All were fixed in code and generated text, never by editing numbers.

1. **TCGA accuracy equalled the majority rate** (398 of 474 patients alive = 0.840).
   - Every TCGA table and figure now carries a "Predict all alive" baseline row.
   - The text never ranks accuracies within 0.01 of that rate.
   - *Why accuracy and F1 disagreed:* they really used different thresholds. Accuracy used P ≥ 0.5 in some places and
     F1 the training-fold threshold. Now one rule (the F1 threshold) is used everywhere.
2. **FedXGBBagging added up the hospitals' corrections instead of averaging them**, so each round moved the model
   K times too far.
   - Symptom: Cleveland's held-out log-loss rose every round, from 0.532 at round 2 to 0.744 at round 30.
   - Fix: averaging, applied to every XGBoost variant, and everything was re-run. Afterwards it reads 0.601 at round 2,
     0.542 at round 10 and 0.556 at round 30.
   - Old results moved to `results/_superseded/phase6_before_audit/`. The number of rounds was not changed.
3. **Poisoning tables now show the cost of screening.** Recall sits next to the clean run's false-exclusion rate (how
   often the screen rejects an honest hospital when nobody attacks). Where that rate exceeds 5%, no screened result is
   called a win.
4. **Chain-attack 100% is by construction.** Deterministic checks catch every simulated variant, so 100% is not
   comparable with the paper's 91/88/81%. The captions now list what is and is not covered, and note that our
   difficulty-2 blocks take milliseconds against the paper's ~5 s.

### Phase 8 — Why centralized XGBoost scored low on TCGA

**Asked:** explain, with measurements, why pooled XGBoost (AFT C-index 0.565) was below local (0.650) and federated
(0.637).

**Built:** `analysis/diagnose_central_xgb.py`. It replays the runs exactly (it reproduces the stored numbers to 1e-9)
and records training vs held-out scores per round.

**Found:** it **over-fits**.
- Training C-index climbs from 0.830 to 0.993 while held-out falls from 0.666 to 0.565.
- It has the same tree count as federated (750), but boosts 25 trees per round in sequence: 750 steps against the
  federated model's 150, because since Phase 7 federated rounds average.
- At equal steps its mean is still lower (0.602). Phase 8 put the rest down to one pooled model over-fitting faster
  than an average of five; Phase 12 corrected that: by the paired rule the step-matched row (Phase 9) is not
  distinguishable from federated or local, so no gap beyond the step count is shown.
- Site death rates explain local's lead (a predictor that knows only the site's death rate already reaches C-index
  0.585, ROC-AUC 0.700), not central's drop.

**Not a bug:** the code does what the documented design says (equal trees), so nothing was changed or re-run.

### Phase 9 — Two reference rows

1. **"Site base rate only"** in every method table: each patient scored only by their hospital's death (or disease)
   rate in the training folds, with no features at all.
   - On TCGA it reaches ROC-AUC 0.700 and F1 0.407, and no method exceeds it by the paired rule (most are below it).
     (Phase 12 corrected an earlier "above every method": for a few methods the interval contains 0.)
   - So much of the pooled TCGA ROC-AUC reflects which site a patient came from.
   - On heart it is far below every method (0.674).
2. **"Centralized, step-matched (diagnostic)"**: pooled XGBoost with 5 trees per round, the federated step count.
   - Its rule was fixed in `table.yaml` before it ran.
   - TCGA is marked "not blind" (the diagnosis had already shown 0.602). It got 0.600.
   - Heart (a blind result): 0.862, above both the original centralized (0.846) and federated (0.843), both by the
     paired rule.
   - Whether the paper's baseline should match trees or steps is a decision for your guide.

### Phase 10 — Live demo, dashboard, presenter script, diagrams

- **Live demo:** `python3 -m fedverify.survival.demo` runs a real 3-round federated run on heart in about 3 seconds,
  with no network, and narrates it:
  - signed updates, Merkle roots and mined blocks;
  - in round 2, Switzerland (the study's attacker for seed 0) flips its update and the screen rejects it, saying why
    ("coordinate score 3.37 > τ 3.21");
  - in round 3, a replayed old update and a submission signed with the wrong hospital's key are rejected at the ledger;
  - at the end, an inclusion proof is checked against the mined block, then one flipped byte makes the same check fail.

  Other seeds honestly show the screen missing the attacker, which matches the study's low recall.
- **Dashboard:** heart and TCGA runs can be published to the existing Federated tab with `fedverify.chain.push`.
  - The banner, the accept/reject grid, the round commitments and the in-browser proof check all work for them.
  - The existing MNIST runs render exactly as before (checked in a browser and by tests).
- **DEMO.md:** a 5-minute presenter script with timings, a fallback, and three examiner questions with honest answers,
  generated from the results.
- **Diagrams:** six Mermaid diagrams in `docs/diagrams/`. A test checks that every module, class and function they
  name exists. mermaid-cli was not installed, so they were not rendered to images yet (done in Phase 12).

504 tests passing.

### Phase 11 — Statistics, within-hospital metrics, LaTeX check, freeze

1. **Within-hospital ROC-AUC and C-index** (extra columns). These only compare patients of the same hospital, so a
   hospital's base rate cannot help.
   - Local models seemed to lose the most (TCGA local XGBoost AFT: 0.650 pooled, 0.595 within hospital). Phase 12
     replaced this ranking, which had no interval, with paired intervals within hospital (D3, D8b).
   - Two deviations from the request are reported:
     - the exclusion rule leaves out hospitals with fewer than 5 of the rarer class (TCGA A8 and heart Switzerland);
     - the site-rate row scores 0.49, not exactly 0.5, because of a fold artefact (exactly 0.5 within one fold).
2. **Paired comparisons with bootstrap intervals,** and a new verdict rule: "outperforms only if the 95% interval of the
   paired difference excludes 0".
   - 56 of 124 generated verdicts changed wording; `paper/VERDICT_CHANGES.md` lists every one.
   - Of the 20 key comparisons, 9 are decisive (Phase 12: 8 of them are ranked; see D8).
3. **LaTeX check:** every generated table compiled in an IEEE two-column document with Tectonic 0.17.0.
   - All 27 compile with no overfull box.
   - Two wide tables are split into two parts in LaTeX only, keeping every column.
   - A bug in the earlier MNIST tables was fixed: a "%" in captions started a LaTeX comment.
4. **Report pack and freeze:**
   - `results/report_pack/` has 101 files named by report section (4.1 design to 4.3.7 comparison, A appendix), with a
     README mapping each to the report and to slides 8 and 11.
   - `results/MANIFEST.json` lists the SHA-256 of all 15,380 results files plus the pip freeze.
   - Their Merkle root `483e6830…c7971a` is mined into a local proof-of-work block, and `--verify` re-checks everything.
     (Phase 12 re-froze the rebuilt results: see its final gate.)

521 tests.

### Phase 12 — Final wrap-up

**Asked:** fix four kinds of wording in the generators (never by hand, each with a test), add within-hospital
intervals for the 20 key comparisons, render the diagrams, mark the MNIST tables that have no results, update this
guide, plan the commits, and run the final gate. No experiment was re-run and nothing was tuned.

1. **"Not ranked" for a doubtful gain.** Heart random forest, Ours vs federated: +0.003 (CI +0.000 to +0.006), but on
   a clean run Ours' screen excludes 13.3% of honest hospital-rounds (above 5%). Section A of the draft already said
   "not ranked"; now the paired table, the draft's paired paragraph and the page say it too.
2. **"Highest score", never "best".** The draft's within-hospital paragraph said "best Random forest: Centralized".
   It now names the highest score and compares it with the runner-up by the interval; all three are not
   distinguishable from their runner-up. A commit-only twin (identical predictions) is not counted as a runner-up.
3. **Within-hospital intervals for the 20 key comparisons:** new table `paired_comparisons_within` (in the draft, on
   the page and in the report pack), same resamples and rule. 5 of the 20 intervals exclude 0 (9 pooled), and 10 of
   the 20 verdicts differ from the pooled ones (D8b). The draft's within-hospital ranking without an interval
   ("largest fall from pooled to within hospital") was removed.
4. **Federated beating centralized on TCGA is qualified in the same sentence**, in the tables, the draft and the
   page:
   - XGBoost AFT: centralized XGBoost over-fits (training C-index 0.830 to 0.993 while held-out falls from 0.666 to
     0.565), and federated is not distinguishable from the step-matched row (+0.037, CI −0.002 to +0.075).
   - Random forest: "not diagnosed". Phase 8 looked at XGBoost only, there is no step-matched random forest, and the
     centralized random forest's held-out C-index does not fall over the rounds (0.615 after round 1, 0.616 after
     round 30), so over-fitting is not claimed.
   - On F1 (death indicator), federated XGBoost still beats the step-matched row; the page says so.
5. **Earlier claims corrected** where the interval does not support them: "at equal steps central still scores
   lower" (not shown: the step-matched row is not distinguishable from federated or local), "central is below both
   within sites" (within hospital only federated is distinguishably above it; local is not), the site-rate row
   "above every method" on TCGA, and "best" in DESIGN_NOTES.md (section 11 lists them).
6. **Diagrams rendered** with mermaid-cli 12.0.0 and the installed Chrome (no download): SVG and 300-dpi PNG of all
   six, at each diagram's natural size, in `docs/diagrams/` and the report pack
   (`python3 -m fedverify.analysis.render_diagrams`). The sequence diagram first failed: in Mermaid a ";" inside a
   message ends the line. Two semicolons were replaced by commas.
7. **MNIST tables 2, 2b, 2_full, 4 and 6** now say "Not run" in their captions: experiments exp2, exp4 and exp5 were
   never run in this repository, so every cell is "—". The requested note "raw results from an earlier run, not
   retained" would have been false (there was no earlier run here), so it was not used. Table 5 gets no note: it is
   the qualitative table built from `analysis/comparison.yaml` and regenerates unchanged. The commands that would fill
   the tables are in `fedverify/TODO.md`.
8. **`COMMIT_PLAN.md`** (repository root) groups every new or changed file into commits on three branches and lists
   what must never be committed.
9. **Final gate:** `make_fl_healthcare_all` rebuilt everything with no missing input; the report pack has 116 files
   and no problem; 28 of 28 LaTeX tables compile cleanly; the results were frozen again (`MANIFEST.json`: 15,403
   files, Merkle root `76b01643…0d35abbd`, mined into a local proof-of-work block), 532 tests pass, and
   `make_manifest --verify` reports the results intact after the tests. The showcase page was republished.

532 tests.

---

## Part D. What the results say

All values are mean ± std over 5 repeats of 5-fold cross-validation. "Pooled" = all out-of-fold patients together.

### D1. Heart (ROC-AUC, pooled; higher is better)

| | Logistic regression | Random forest | XGBoost |
|---|---|---|---|
| Local only | 0.837 ± 0.002 | 0.869 ± 0.004 | 0.839 ± 0.009 |
| Centralized | 0.866 ± 0.002 | 0.879 ± 0.000 | 0.846 ± 0.003 |
| Centralized, step-matched | n/a | n/a | 0.862 ± 0.005 |
| Federated (vanilla FL) | 0.859 ± 0.001 | 0.864 ± 0.002 | 0.843 ± 0.005 |
| DP-FL, ε = 4 | 0.856 ± 0.002 | n/a | n/a |
| Ours (commit + screen) | 0.856 ± 0.003 | 0.866 ± 0.002 | 0.838 ± 0.005 |
| Site base rate only | 0.674 ± 0.000 | | |

In plain words: on heart, federated learning gets close to pooling all the data and every model is far above the
"which hospital" baseline. By the paired rule (pooled):
- federated logistic regression beats local (+0.022, CI +0.011 to +0.036);
- centralized beats federated for logistic regression and random forest;
- the step-matched centralized XGBoost beats federated XGBoost;
- Ours with the random forest is +0.003 above federated (CI +0.000 to +0.006) but **not ranked**: on a clean run its
  screen excludes 13.3% of honest hospital-rounds (above 5%), so the gain may come from dropping a skewed hospital.

Within hospital (D8b) none of the first three holds: every federated-versus-local, federated-versus-centralized and
federated-versus-step-matched interval on heart contains 0.

### D2. TCGA-BRCA (C-index, pooled; higher is better)

| | Logistic regression | Random forest | XGBoost AFT |
|---|---|---|---|
| Local only | 0.539 ± 0.017 | 0.635 ± 0.013 | 0.650 ± 0.020 |
| Centralized | 0.576 ± 0.020 | 0.616 ± 0.009 | 0.565 ± 0.016 |
| Centralized, step-matched | n/a | n/a | 0.600 ± 0.020 |
| Federated (vanilla FL) | 0.574 ± 0.019 | 0.674 ± 0.012 | 0.637 ± 0.010 |
| Ours | 0.517 ± 0.038 | 0.669 ± 0.015 | 0.626 ± 0.009 |
| Site base rate only | 0.585 ± 0.002 | | |
| Predict all alive | 0.500 | | |

In plain words: TCGA is hard; 76 deaths spread over 5 sites is very little to learn from.
- Federated random forest has the highest pooled C-index (0.674), above the runner-up, Ours with the random forest
  (0.669; Ours vs federated −0.005, CI −0.009 to −0.001). Within hospital it again has the highest score (0.666),
  but not distinguishable from Ours (0.660; +0.005, CI −0.000 to +0.011).
- Federated beats centralized for random forest (+0.058, CI +0.013 to +0.104), but this gap was **not diagnosed**:
  Phase 8 looked at XGBoost only and there is no step-matched random forest.
- Federated beats centralized for XGBoost AFT (+0.072, CI +0.023 to +0.122), but centralized XGBoost **over-fits**
  (Phase 8) and federated is **not distinguishable from the step-matched row** (+0.037, CI −0.002 to +0.075).
- Most other differences have intervals that include 0, so they cannot be ranked.
- On the death-indicator task (ROC-AUC and F1), no method beats the site-rate reference: that score mostly reflects
  site death rates.

### D3. Within hospital

| Run | Pooled | Within hospital |
|---|---|---|
| TCGA, XGBoost AFT, local | 0.650 ± 0.020 | 0.595 ± 0.021 |
| TCGA, XGBoost AFT, federated | 0.637 ± 0.010 | 0.614 ± 0.013 |
| TCGA, random forest, federated | 0.674 ± 0.012 | 0.666 ± 0.012 |
| TCGA, XGBoost (death indicator, ROC-AUC), local | 0.691 ± 0.014 | 0.612 ± 0.020 |
| Heart, logistic regression, federated (ROC-AUC) | 0.859 ± 0.001 | 0.847 ± 0.002 |

The drop of the local models looks largest, which fits the idea that part of their pooled score came from site base
rates; but a claim about within-hospital differences has to go through the paired intervals (D8b):
- TCGA random forest: federated beats local within hospital (+0.068, CI +0.009 to +0.130), although pooled the two
  are not distinguishable (+0.039, CI −0.034 to +0.111).
- Every other federated-versus-local interval within hospital contains 0, on both datasets.
- Highest within-hospital scores, each not distinguishable from the runner-up: heart, centralized random forest
  (0.864; runner-up Ours with the random forest, 0.860, +0.003, CI −0.003 to +0.010); TCGA death indicator, federated
  random forest (0.646; runner-up Ours, 0.641, +0.004, CI −0.000 to +0.009); TCGA survival, federated random forest
  (0.666; runner-up Ours, 0.660, +0.005, CI −0.000 to +0.011).

### D4. Differential privacy

| ε | Heart ROC-AUC | TCGA C-index |
|---|---|---|
| 1 | 0.843 ± 0.006 | 0.507 ± 0.039 |
| 2 | 0.852 ± 0.003 | 0.510 ± 0.038 |
| 4 | 0.856 ± 0.002 | 0.517 ± 0.034 |
| 8 | 0.857 ± 0.002 | 0.520 ± 0.025 |
| no DP | 0.859 ± 0.001 | 0.574 ± 0.019 |

The realised ε stays under every target (heart: 0.945, 1.892, 3.789, 7.557). On heart, DP costs little; at ε = 4 and 8
the cost is not distinguishable from 0. On TCGA it pushes the model close to a coin flip, mostly through clipping
(see Phase 2). Even there, only ε = 4 and 8 are distinguishably worse than no DP; the ε = 1 and 2 intervals touch 0.

### D5. Blockchain overhead (mined runs, difficulty 2, one laptop)

| | Heart | TCGA |
|---|---|---|
| Blocks | 150 | 150 |
| Latency (mean time between blocks) | 74.0 ms | 44.0 ms |
| Throughput | 22,360 B/s | 37,786 B/s |
| Send rate | 4,083 transactions/min | 7,083 transactions/min |
| Mining one block | 0.0047 s | 0.0048 s |

Committing a heart logistic round costs about 2.6 ms across all stages with the mock anchor, or about 7 ms when a block
is mined, against about 66 ms of training. With the mock anchor, verifying the signatures is the largest part; when a
block is mined, mining dominates. An update weighs 0.9 KiB per
round for logistic regression, 31.3 KiB for random forest and 15.5 KiB for XGBoost on heart.

### D6. Chain attacks (100 simulated rounds per attack and dataset)

| Attack | Defences on | Defences off | Reported by the reference paper |
|---|---|---|---|
| Double spending | 100% | 0% | 91% |
| Transaction malleability | 100% | 8% (heart), 5% (TCGA) | 88% |
| Endpoint compromise | 100% | 0% | 81% |

100% means every *simulated* variant met a deterministic check it cannot pass; it is not a measured failure rate.
Forks, a stolen key, dropped submissions and multi-factor authentication are **not** covered (THREAT_MODEL.md).

### D7. Poisoning and screening

| | Heart | TCGA |
|---|---|---|
| Screen recall (share of attacker rounds caught) | 13–27% | 9–99% (by attack) |
| Honest rounds wrongly rejected on a clean run | 10.6–19.9% | 5.4–12.2% |
| Calibrated τ (logistic, ε = 4) | 3.212 | 5.067 |

The screen catches easy attacks on TCGA (scaling 98.1%, flipped XGBoost trees 98.8%) but most others slip through,
and on heart it rejects honest hospitals often, because the hospitals genuinely differ a lot. With only 4–5 hospitals a
statistical screen has little crowd to compare against. The screen limits poisoning; it does not prevent it.

### D8. The key comparisons, all 20 (paired, 95% CI of A − B)

| Dataset | Model | Comparison | Difference | 95% CI | Verdict |
|---|---|---|---|---|---|
| Heart | LR | Federated vs Local | +0.022 | +0.011 to +0.036 | outperforms |
| Heart | LR | Federated vs Centralized | −0.006 | −0.012 to −0.001 | is outperformed by |
| Heart | LR | Ours vs Federated | −0.004 | −0.008 to +0.000 | not distinguishable |
| Heart | RF | Federated vs Local | −0.005 | −0.016 to +0.006 | not distinguishable |
| Heart | RF | Federated vs Centralized | −0.016 | −0.023 to −0.008 | is outperformed by |
| Heart | RF | Ours vs Federated | +0.003 | +0.000 to +0.006 | not ranked (clean-run false exclusion 13.3%, above 5%) |
| Heart | XGB | Federated vs Local | +0.005 | −0.011 to +0.020 | not distinguishable |
| Heart | XGB | Federated vs Centralized | −0.002 | −0.016 to +0.011 | not distinguishable |
| Heart | XGB | Federated vs Centralized, step-matched | −0.019 | −0.030 to −0.008 | is outperformed by |
| Heart | XGB | Ours vs Federated | −0.005 | −0.009 to −0.001 | is outperformed by |
| TCGA | LR | Federated vs Local | +0.035 | −0.035 to +0.106 | not distinguishable |
| TCGA | LR | Federated vs Centralized | −0.001 | −0.046 to +0.041 | not distinguishable |
| TCGA | LR | Ours vs Federated | −0.057 | −0.115 to +0.001 | not distinguishable |
| TCGA | RF | Federated vs Local | +0.039 | −0.034 to +0.111 | not distinguishable |
| TCGA | RF | Federated vs Centralized | +0.058 | +0.013 to +0.104 | outperforms, but not diagnosed (no step-matched RF) |
| TCGA | RF | Ours vs Federated | −0.005 | −0.009 to −0.001 | is outperformed by |
| TCGA | XGB AFT | Federated vs Local | −0.013 | −0.079 to +0.049 | not distinguishable |
| TCGA | XGB AFT | Federated vs Centralized | +0.072 | +0.023 to +0.122 | outperforms, but centralized over-fits and federated is not distinguishable from the step-matched row |
| TCGA | XGB AFT | Federated vs Centralized, step-matched | +0.037 | −0.002 to +0.075 | not distinguishable |
| TCGA | XGB AFT | Ours vs Federated | −0.011 | −0.023 to +0.001 | not distinguishable |

(Heart uses ROC-AUC, TCGA the C-index. 9 intervals exclude 0; 8 of those are ranked. The full table, with the DP
rows, is `results/tables/paired_comparisons.md`.)

### D8b. The same 20 comparisons within hospital (Phase 12)

Same patients, the same 2000 resamples and the same rule, but only pairs of patients from the same hospital count
(Switzerland and A8 left out, as in D3).

| Dataset | Model | Comparison | Difference | 95% CI | Verdict | Pooled verdict (D8) |
|---|---|---|---|---|---|---|
| Heart | LR | Federated vs Local | +0.006 | −0.003 to +0.015 | not distinguishable | outperforms |
| Heart | LR | Federated vs Centralized | −0.001 | −0.006 to +0.004 | not distinguishable | is outperformed by |
| Heart | LR | Ours vs Federated | −0.005 | −0.010 to −0.001 | is outperformed by | not distinguishable |
| Heart | RF | Federated vs Local | −0.001 | −0.011 to +0.008 | not distinguishable | not distinguishable |
| Heart | RF | Federated vs Centralized | −0.006 | −0.012 to +0.001 | not distinguishable | is outperformed by |
| Heart | RF | Ours vs Federated | +0.002 | −0.001 to +0.006 | not distinguishable | not ranked |
| Heart | XGB | Federated vs Local | +0.014 | −0.003 to +0.030 | not distinguishable | not distinguishable |
| Heart | XGB | Federated vs Centralized | +0.014 | −0.000 to +0.028 | not distinguishable | not distinguishable |
| Heart | XGB | Federated vs Centralized, step-matched | −0.005 | −0.016 to +0.006 | not distinguishable | is outperformed by |
| Heart | XGB | Ours vs Federated | −0.001 | −0.006 to +0.004 | not distinguishable | is outperformed by |
| TCGA | LR | Federated vs Local | +0.050 | −0.026 to +0.137 | not distinguishable | not distinguishable |
| TCGA | LR | Federated vs Centralized | +0.000 | −0.045 to +0.044 | not distinguishable | not distinguishable |
| TCGA | LR | Ours vs Federated | −0.070 | −0.133 to −0.006 | is outperformed by | not distinguishable |
| TCGA | RF | Federated vs Local | +0.068 | +0.009 to +0.130 | outperforms | not distinguishable |
| TCGA | RF | Federated vs Centralized | +0.057 | +0.016 to +0.100 | outperforms, but not diagnosed (no step-matched RF) | outperforms, but not diagnosed |
| TCGA | RF | Ours vs Federated | −0.005 | −0.011 to +0.000 | not distinguishable | is outperformed by |
| TCGA | XGB AFT | Federated vs Local | +0.019 | −0.032 to +0.070 | not distinguishable | not distinguishable |
| TCGA | XGB AFT | Federated vs Centralized | +0.068 | +0.017 to +0.119 | outperforms, but centralized over-fits and federated is not distinguishable from the step-matched row | outperforms, but centralized over-fits |
| TCGA | XGB AFT | Federated vs Centralized, step-matched | +0.033 | −0.009 to +0.075 | not distinguishable | not distinguishable |
| TCGA | XGB AFT | Ours vs Federated | −0.006 | −0.019 to +0.009 | not distinguishable | not distinguishable |

5 intervals exclude 0 (all 5 ranked), and 10 of the 20 verdicts differ from the pooled ones. In plain words: on heart,
every interval that excluded 0 pooled contains 0 once only same-hospital patients are compared, and one new one
appears (Ours below federated, logistic regression); on TCGA, federated random forest now also beats local, and Ours
with logistic regression is below federated. The full table is `results/tables/paired_comparisons_within.md`.

### D9. The honest summary

- Federated learning reaches **close to centralized performance on heart** (and within hospital it is not
  distinguishable from centralized). On TCGA the federated tree models beat pooled training, but for XGBoost that is
  because the pooled model over-fits: federated is not distinguishable from the step-matched pooled model; for the
  random forest the gap was not diagnosed. Pooled TCGA scores are partly driven by site base rates.
- The chain layer **proves who sent which update and that nothing was changed afterwards**, at a cost of
  milliseconds per round. It does **not** prove that the server averaged correctly.
- DP on the logistic model is **cheap on heart and expensive on TCGA**.
- Screening helps against some attacks but, with 4–5 hospitals, misses most attacker rounds and sometimes rejects
  honest hospitals. This is stated everywhere, not hidden.

---

## Part E. How to run and use everything

Everything runs on a CPU laptop. Install once (see `fedverify/survival/README.md`), then:

| Goal | Command |
|---|---|
| Run every experiment (about 30 minutes with 6 jobs) | `python3 -m fedverify.survival.experiments all --jobs 6` |
| Run one experiment | `python3 -m fedverify.survival.experiments methods` (or `dp`, `chain`, `chain-pow`, `calibration`, `chain-attacks`, `poisoning`, `diagnostics`) |
| Rebuild all tables, figures, the draft, the page, DEMO.md and the report pack | `python3 -m fedverify.analysis.make_fl_healthcare_all` |
| Check every LaTeX table compiles (needs tectonic or pdflatex) | `python3 -m fedverify.analysis.latex_check --engine /path/to/tectonic` |
| Freeze the results / check they are intact | `python3 -m fedverify.analysis.make_manifest` / `… --verify` |
| Run the live demo | `python3 -m fedverify.survival.demo` (add `--push` to send it to the dashboard) |
| Render the report diagrams (SVG and 300-dpi PNG; needs Node.js and Chrome) | `python3 -m fedverify.analysis.render_diagrams` |
| Run all tests | `python3 -m pytest backend/ fedverify/tests/ -q` |

**The user interfaces:**

1. **Showcase page:** https://claude.ai/artifact/SgRLdv9VeynvSqRb4HrBkR (private; share from its menu), or open
   `fedverify/results/showcase/index.html` offline.
2. **Dashboard (Federated tab):**
   1. `cd backend && python3 app.py`
   2. `cd frontend && python3 -m http.server 8080`, then open http://localhost:8080.
   3. Publish a run: `python3 -m fedverify.chain.push fedverify/results/demo/heart_ours_lr_demo/seed0`.
   4. In the Federated tab, click Refresh and pick it.
3. **Presenter script:** [`fedverify/survival/DEMO.md`](../survival/DEMO.md).

**Where things are:**

| What | Where |
|---|---|
| Study code | `fedverify/survival/` (runner, data, models, chain round, attacks, screening, demo) |
| Generators | `fedverify/analysis/` (tables, figures, draft, showcase, paired statistics, LaTeX check, report pack, manifest) |
| Which run fills which row | `fedverify/survival/table.yaml` |
| Results (not in git) | `fedverify/results/` (run folders, tables, figures, statistics, report pack, MANIFEST.json) |
| Generated results draft | `fedverify/survival/paper/RESULTS_DRAFT.md` (+ `VERDICT_CHANGES.md`) |
| Design, threats, open work | `DESIGN_NOTES.md`, `THREAT_MODEL.md`, `fedverify/TODO.md` |
| Report diagrams | `docs/diagrams/*.mmd` (sources), `.svg` and `.png` (renders), `rendered.json` |
| Commit plan for the human | `COMMIT_PLAN.md` (repository root) |
| Dashboard code (additive changes) | `backend/fl_routes.py`, `frontend/index.html`, `fedverify/chain/push.py` |

**Reproducibility:** the same seed gives byte-identical results (apart from wall-clock timings). τ always comes from
the calibration file. `MANIFEST.json` fingerprints every result, so any later change is detectable.

---

## Part F. Limitations and open decisions

**Limitations** (stated in the results draft too):

- Small data: TCGA has 76 deaths, and A8 only 4; heart's Switzerland has 1 healthy patient.
- The chain is one simulated node on one laptop; there is no consortium or public network except the optional
  Algorand anchor (not exercised live).
- No proof that the server averaged correctly (no verifiable aggregation); no homomorphic encryption.
- DP covers training examples of the logistic model only (not patients as a whole, not trees). Each fold-run is
  accounted separately.
- Attackers are non-adaptive, at most one per run; the screen's τ is calibrated on the same attack types.
- The bootstrap intervals hold the trained runs fixed; they leave out the variation between training runs (the
  standard deviation over repeats is reported next to them).
- Keys are test keys derived from the seed; timestamps are a logical clock.

- MNIST tables 2, 2b, 2_full, 4 and 6 are empty: their experiments were never run (Phase 12).

**Decisions waiting for you and your guide:**

1. FPoW is undefined; every commit uses the existing proof-of-work chain.
2. Centralized baseline for XGBoost: equal trees (current) or equal boosting steps (shown as a diagnostic row).
3. Within-hospital exclusion rule: keep leaving out heart's Switzerland (rarer class < 5) or not.
4. Footnote the site-rate fold artefact (0.49 instead of 0.5 within hospital)?
5. A bootstrap that also resamples the training runs (more honest intervals for the poisoning comparisons).
6. Delete `fedverify/results/_superseded/` once the numbers are approved, then re-run `make_manifest`.
7. Pooled or within-hospital as the lead metric of the paper: 10 of the 20 key verdicts differ (D8b).
8. TCGA random forest, federated beating centralized: diagnose it (needs re-runs) or leave it as "not diagnosed".
9. MNIST tables 2, 2b, 2_full, 4 and 6: run exp2, exp4 and exp5 (commands in `fedverify/TODO.md`) or drop the tables
   from the report.
10. `architecture.mmd` is very wide: place it landscape in the report, or re-lay it out top-down.

(Phase 11's "restore `CLAUDE.md`" item is closed: the file is back and unchanged.)

---

## Part G. Glossary

| Term | Meaning |
|---|---|
| Aggregator | The server that combines hospitals' updates. |
| AFT | Accelerated failure time: a model that predicts survival time. |
| Anchor | Writing a round's Merkle root into a blockchain block. |
| Bootstrap | Re-drawing the patients many times to see how much a result could vary. |
| C-index | Share of patient pairs whose order of death the model gets right. |
| Calibration (τ) | Choosing the screen's threshold from labelled runs, never by hand. |
| Censored | A patient still alive at last follow-up; their final outcome is unknown. |
| CID | Content identifier: a hash-based name for data. |
| Clean false exclusion | How often the screen rejects an honest hospital when nobody attacks. |
| Confidence interval | The range a value could plausibly take; here the middle 95% of bootstrap results. |
| Cross-validation | Rotating which patients are held out so every patient is tested once. |
| Delta | The change a hospital wants to make to the shared weights. |
| DP-SGD / ε | Training with clipping and noise for differential privacy; ε measures how much privacy is spent. |
| FedAvg | Average the hospitals' deltas, weighted by patient count. |
| FedXGBBagging | The way XGBoost is federated here: average the hospitals' new boosters each round. |
| Forensics | FedVerify's existing outlier screen for logistic updates. |
| Inclusion proof | The hashes that prove one update is inside a Merkle root. |
| Merkle root | One hash that commits to every update of a round. |
| Out-of-fold | A prediction for a patient made by a model that never trained on them. |
| Paired comparison | Comparing two methods run on the same folds and repeats, repeat by repeat. |
| Probe set | 512 public, made-up rows used to compare hospitals' trees without patient data. |
| Proof of work | Mining: finding a block hash that starts with zeros, which makes rewriting expensive. |
| Recall | Share of attacker rounds the screen caught. |
| Robust z-score | How far a value is from the median, in units of the median absolute deviation. |
| ROC-AUC | Chance a random positive patient is scored above a random negative one. |
| Site base rate | A hospital's own outcome rate, used as a no-feature reference score. |
| Within hospital | A metric computed only on pairs of patients from the same hospital. |

---

## Part H. Questions an examiner may ask

**Why not simply pool the data?** Patient records cannot legally leave the hospital. The centralized rows show what
pooling would give, as an upper reference, not as an option.

**What does the blockchain actually prove?** Which signed update each hospital sent in each round, and that none of
it was changed afterwards (inclusion proofs against mined roots, re-checkable by anyone). It does not prove the average
was computed correctly, and it does not stop a hospital from sending a poisoned but validly signed update. That is the
screen's job.

**Why is your chain-attack resilience 100% when the paper reports 81–91%?** Our defences are deterministic checks:
every variant we simulate fails its check, every time. 100% says which variants are covered, not that the system is
unbreakable. The uncovered ones are listed in THREAT_MODEL.md.

**Why is screening recall so low?** With 4–5 hospitals of very different sizes and outcome rates, an honest but unusual
hospital looks as odd as an attacker. A threshold that keeps false alarms near 5% therefore lets most attacks through.

**Why does DP hurt TCGA so much but not heart?** TCGA has few deaths, and the model relies on up-weighting them.
DP-SGD clips each example's gradient to the same size, which cancels most of that up-weighting. Heart is balanced, so
it is barely affected.

**Is local better than federated on TCGA?** Not distinguishably by the paired rule. Pooled scores flatter local models,
because each local model carries its site's base rate. Within hospital, federated random forest beats local
(+0.068, CI +0.009 to +0.130) and has the highest score, though not distinguishable from Ours.

**Federated beats centralized on TCGA. Isn't that suspicious?** Yes, and the text never states it alone. For XGBoost
the pooled model over-fits (its training C-index climbs to 0.993 while its held-out C-index falls), and federated is
not distinguishable from a pooled model given the same number of boosting steps. For the random forest the gap was
not diagnosed, and the text says exactly that.

**Why are some MNIST tables empty?** Their experiments (exp2, exp4, exp5) were never run in this repository. The
captions say "Not run" and TODO.md has the commands. Nothing was filled in by hand.

**How do we know no number was typed by hand?** Every table, figure, sentence and page is generated from the run
folders. `RESULTS_DRAFT.comparisons.json` logs every comparison, and `MANIFEST.json` fingerprints every results file
under a mined Merkle root.

---

## Appendix. Progress record and file index

**Tests passing after each phase** (from the phase reports): baseline 259 → Phase 1: 346 → Phase 2: 359 →
Phase 2b: 392 → Phase 3: 422 → Phase 4: 465 → Phase 6: 476 → showcase page: 477 → Phase 7: 479 → Phase 8: 484 →
Phase 9: 492 → Phase 10: 504 → Phase 11: 521 → Phase 12: 532.

**Main files added by this study** (all additive):

- `fedverify/survival/`:
  - `config.py`, `data.py`, `heart.py`, `datasets.py`, `models.py`, `common.py`, `metrics.py`, `runner.py`
  - `fl_lr.py`, `fl_rf.py`, `fl_xgb.py`, `dp.py`
  - `serialize.py`, `chain_round.py`, `chain_attacks.py`, `poison.py`, `probe.py`, `screening.py`
  - `experiments.py`, `demo.py`, `table.yaml`
  - docs: `README.md`, `DESIGN_NOTES.md`, `THREAT_MODEL.md`, `INVENTORY.md`, `DEMO.md`
- `fedverify/chain/`: `contentid.py`, `signing.py`; `push.py` was extended.
- `fedverify/analysis/`:
  - `make_fl_healthcare_tables.py`, `make_fl_healthcare_figures.py`, `make_results_draft.py`
  - `make_showcase.py` (+ `showcase_template.html`), `make_fl_healthcare_all.py`
  - `calibrate_survival.py`, `diagnose_central_xgb.py`, `paired_stats.py`
  - `latex_check.py`, `make_report_pack.py`, `make_manifest.py`, `make_demo_md.py`, `render_diagrams.py`
- `fedverify/tests/test_survival_*.py`, `test_fl_healthcare_outputs.py`, `test_results_draft.py`,
  `test_phase9_references.py`, `test_demo.py`, `test_diagrams.py`, `test_phase11_stats.py`,
  `test_phase11_freeze.py`, `test_phase12_wording.py`.
- `docs/diagrams/` (six `.mmd` files, their `.svg` and `.png` renders, `rendered.json` and a README).
- `COMMIT_PLAN.md` (repository root): how to commit all of this.
- Additive edits to the dashboard: `frontend/index.html` (labels for the new attacks and aggregators, per-round
  ground truth for runs that carry it). `backend/fl_routes.py` was not changed.
