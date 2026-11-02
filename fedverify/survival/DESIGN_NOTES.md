# FedVerify-Survival — design notes

The rules the code follows, so every table can point here instead of restating them (sections 1-4), and how this
work relates to the reference paper, Myrzashova et al., IEEE T-SUSC 10(1), 2025 (section 5). Statements about that
paper were checked against its text.

## 1. Data and evaluation (Phases 1 and 2b)

- **Datasets.** TCGA-BRCA: five source sites (BH, A2, E2, A8, AR), label = death indicator, time in days.
  UCI heart disease: four hospitals (Cleveland, Hungary, VA Long Beach, Switzerland), label = `num > 0`.
- **Heart modes.** `paper` (default for every table): drop `ca`, `thal`, `slope`, then drop incomplete rows. This
  is how the reference heart paper (arXiv 2408.06183) arrived at its analysed sample. `full`: every row, missing
  numeric -> 0 plus `<col>_missing`, missing categorical -> `NA`; used only for one sensitivity table.
  `dropna()` over all columns is never used: it keeps almost only Cleveland.
- **Preprocessing is data-independent** for both datasets: fixed scaling constants and a fixed one-hot vocabulary
  (`vocab.json`, `heart_vocab.json`, category names only). No mean, median or scale is fitted on patients, so no
  hospital leaks statistics to another.
- **Protocol.** Repeated stratified 5-fold CV, strata (hospital, label), repeat r uses seed base + r. Fold k's test
  set is every hospital's fold-k patients. Final-round out-of-fold predictions are pooled and each metric is computed
  once per repeat; tables report mean ± std over the repeats.
- **Thresholds: one rule for accuracy and F1 (Phase 7).** Both use the threshold that maximises F1 on the training
  fold's predictions (each hospital's own for `local_*`), applied to the test fold. The same rule gives the per-hospital
  and per-round train/test accuracy behind Figs 5 and 6 (the training curve uses the training-fold threshold on the
  training rows). `config.json` records it as `accuracy_rule`. Before Phase 7, accuracy used P(event) >= 0.5 while F1
  used the F1-optimal threshold, so the two were computed at different operating points (see section 7). Log-loss
  uses `labels=[0, 1]` so a one-class hospital or fold cannot crash it.
- **Trivial baseline (Phase 7).** Every TCGA table and figure carries "predict all alive": score 0 and prediction 0 for
  every out-of-fold patient of one run (`baselines` in `table.yaml`), scored with the runs' own metric code. Its
  accuracy is the majority-class rate, its F1 is 0 and its AUC and C-index are 0.5. The generated text never ranks
  two accuracies when either lies within `accuracy_majority_margin` (0.01) of the majority-class rate of that dataset
  or hospital.
- **Local-only methods.** `local_*` scores every patient with their own hospital's model. On heart, Switzerland
  (paper mode: 1 negative of 46 rows) gets no local model: it is left out of the `local_*` rows and of their pooled
  evaluation, but stays in federated training and in every other method's pooled evaluation. Per-hospital rank
  metrics print an em dash when the rarer class has fewer than 5 rows (TCGA A8: 4 deaths; heart Switzerland).
- **Centralised baselines** use the same budget as the federated ones: `central_rf` grows hospitals x 10 trees per
  round, `central_xgb` boosts hospitals x N trees per round, `central_lr` runs the same epochs per round.

## 2. FedXGBBagging (Phase 2b) — the combination rule, used everywhere

Each round the server broadcasts the global ensemble. Every hospital starts from the **same** global margin on its
own training rows (passed to XGBoost as `base_margin`), boosts **N** new trees (default 5) on those rows only, and
sends **only** those trees. The server keeps every hospital's new trees and adds their **average** over the
contributing hospitals to the global margin. For a row x:

    margin(x) = m0 + sum over rounds r of (1 / K_r) * sum over contributing hospitals h of booster_{r,h}(x)

    booster_{r,h}(x) = sum of hospital h's N new trees of round r at x
    K_r = hospitals whose trees round r aggregates (all K; the accepted ones for chain_*; the kept ones for ours_*)

    m0 = 0                  (binary: P = 0.5 before any tree)
    m0 = ln(365.25 days)    (AFT: a fixed, data-independent starting survival time)

    prediction = sigmoid(margin(x))   for classification (binary:logistic)
    prediction = margin(x)            for survival (survival:aft), the predicted log survival time

After round r the ensemble holds exactly hospitals x N x r trees (tested). Each hospital boosts from the same
broadcast margin, so each booster is a full correction of that margin toward its own rows; averaging them takes one
step per round of the size a single hospital would take. `local_xgb` and `central_xgb` grow one booster per round (one
hospital's rows, or the pooled rows), so for them averaging and summing coincide.

**Correction (Phase 7), made after seeing the curve.** Up to Phase 6 the server **summed** the boosters
(`margin += sum_h booster_{r,h}`), so every round moved the margin K times as far as one hospital's correction. On
heart the held-out log-loss of FedXGBBagging rose every round after round 2: Cleveland, mean over the 25 fold-runs of
`results/_superseded/phase6_before_audit/heart/fedxgb_bagging/seed0/*/*/rounds.jsonl`, 0.532 at round 2 and 0.744 at
round 30, while the training log-loss fell to 0.299. With averaging, the same mean over `results/heart/fedxgb_bagging`
is 0.601 at round 2, 0.542 at round 10 and 0.556 at round 30 (training 0.382). We switched to averaging after seeing that curve, not before, and
applied it to every FedXGBBagging variant (heart, TCGA classification and AFT; plain, chain_* and ours_*), then reran
every affected experiment; the earlier results are in `results/_superseded/phase6_before_audit/`. The number of rounds
was not changed and is never chosen from test data; a milder rise of held-out log-loss can remain and is reported as
it is.

XGBoost settings mirror `XGBClassifier(objective="binary:logistic", tree_method="hist", n_jobs=1, max_depth=3,
learning_rate=0.1, subsample=0.8)`, with `random_state` derived from (repeat seed, fold, round, hospital). The AFT
objective uses `aft_loss_distribution="normal"`, scale 1.0, labels `[t, t]` for a death and `[t, +inf)` if
censored; times below 1 day are clipped up to 1 day and the count is written to `config.json`
(TCGA: the censored patients with zero follow-up).

**Sign convention.** A logistic or forest score is a risk (higher = more likely to die), so the C-index receives
`-risk`. An AFT prediction is a survival time (higher = longer), so the C-index receives it **unchanged**. Both are
pinned by hand-made tests.

## 3. Chain commitment for the chain_* methods (Phase 3)

`chain_fedavg_lr`, `chain_fed_rf` and `chain_fedxgb_bagging` train exactly like their base methods; every round's
updates then go through one pipeline (`survival/chain_round.py`), timed stage by stage, separately from training:

1. **serialize** (`survival/serialize.py`): canonical bytes, no pickle. Logistic: the existing `canon_update`
   (float32 little-endian delta). Random forest: per tree `children_left`, `children_right`, `feature` (int32),
   `threshold`, `value` (float64), little-endian. XGBoost: each tree's JSON dump in tree order. A JSON header binds
   run, round, hospital and sample count, so a digest cannot be replayed into another round.
2. **digest** = SHA-256 of the bytes; **CID** = CIDv1 raw sha2-256, base32 (`chain/contentid.py`), a *locally
   computed content ID* that equals a real IPFS CID only for single-block data. `--ipfs real` also uploads to a
   daemon at 127.0.0.1:5001; it is optional and nothing depends on it.
3. **sign** (`chain/signing.py`): Ed25519 with a TEST key derived from (run seed, hospital). The submission
   `{run_id, round, hospital_id, cid, digest, timestamp, txn_id}` is signed as canonical JSON, so every field is
   covered. `txn_id = SHA-256(hospital_id | counter | timestamp)`.
4. **verify** (aggregator): registered key, valid signature, digest and CID recomputed from the received bytes,
   and a txn_id that hashes an unused counter of that hospital. Counters only move forward; a window of 64 lets one
   lost or rejected submission pass without locking the hospital out, and a used counter never verifies again.
   A failed submission is left out of the round, with its reason recorded.
5. **commit**: Merkle leaves `[i, hospital_id, digest]` (the existing format and code), `round_merkle_root`,
   inclusion proofs, then the anchor: `mock` by default, `local` mines a block on the existing proof-of-work chain.
6. **aggregate from the committed bytes**: the server rebuilds every accepted update from its bytes (a float32
   delta, a numpy forest, numpy boosted trees) and aggregates those. The numpy rebuilds reproduce sklearn's
   probabilities and XGBoost's margins bit for bit, so with every submission valid a chain_* method learns exactly
   what its base method learns (tested).

Before each round after the first, every hospital checks the previous block: the ledger is valid and still holds the
previous round's root under its txid. Hospitals whose check fails skip the round and the aggregator records the
skipped round with the reason (the reference paper's Algorithm 1, steps 8-11), rather than silently continuing.

**Timestamps** inside submissions are a logical clock (a fixed base plus the round number), so a run is
byte-reproducible and the same seed gives the same roots. Wall-clock time is measured separately: the
`*_ms` stage timings in each round record, and the block log.

**One ledger per run.** With `--chain-backend local`, all fold-runs of a run share one chain, so one repeat gives
5 folds x 30 rounds = 150 blocks for Fig 7 (the test keys are registered on it with the first block).

**Chain metrics** (`chain_metrics.json`, defined as the reference paper defines them): latency = mean wall-clock
time between consecutive blocks, reported in ms only; throughput = total block bytes / total elapsed seconds;
send rate = transactions per minute; plus the mean block creation (mining) time. `--profile` writes the top 10
functions by cumulative time to `results/chain/profile.txt` (tsub = own time, ttot = with callees).

## 4. Attacks and defences (Phase 4)

### 4.1 The reference paper's three chain attacks (Figs 9 and 10)

`survival/chain_attacks.py` attacks the commit layer the chain_* and ours_* runs use: the same `FoldCommitter`,
key registry, Merkle code and chain anchor, for the dataset's hospitals. Every simulated round is one real commit
round plus one attack attempt. Its variant and target hospital are drawn from the seed, and the variants are listed in
the attack's `config.json`. A round is **held** when no adversarial submission was accepted (and, for a tampered
stored record, the audit caught it). Resilience **S = 100 x held rounds / simulated rounds**, and attack success =
100 - S, which is how the paper defines S_DS, S_TM and S_EC. Honest payloads are real serialized logistic updates of the
dataset's model size, with values drawn from the seed; no check reads the values.

| attack | variants simulated | defence (on in every run) |
|---|---|---|
| double spending | conflicting_update, replayed_submission, stale_update_respent, duplicate_resubmission | one digest per (run, round, hospital): a second, different digest is rejected and flagged as equivocation; the submission **and** the payload header must name the current round and the submitting hospital; the txn_id must hash an unused counter |
| transaction malleability | timestamp_changed, txn_id_changed (in transit), leaf_order_swapped (stored record, after anchoring) | the signature covers every field; the Merkle leaf binds its position, so a root rebuilt from the stored leaves differs from the anchored one |
| endpoint compromise | unregistered_key, forged_signature, wrong_hospital_key, key_reregistration | the signature must verify under the key registered for the claimed hospital; registration is write-once |

The checks are deterministic: with them on, every simulated variant is caught, and the honest statement is
S = 100% for the variants simulated. `THREAT_MODEL.md` lists the variants that are not covered. To show the defence
tests are not vacuous, `--defence-off` and the tests switch off exactly one attack's defences (`Defences` in
`chain_round.py`), and the same attempts then land. Every run uses all defences. On honest rounds the new checks
accept exactly what Phase 3 accepted: the Phase 3 results re-run byte-identical.

### 4.2 Poisoning and screening (our extension; the paper has none)

- **Methods.** `ours_lr` = DP-FedAvg (epsilon 4 by default) + the chain_* commit + the EXISTING Forensics aggregator,
  unchanged, on the committed deltas; the survivors are FedAvg'd. `ours_rf` / `ours_xgb` = `chain_fed_rf` /
  `chain_fedxgb_bagging` + the probe-set screen below; only the survivors' trees are appended. `--no-screening` runs
  the same pipeline without the screen: the undefended row for "utility with vs without screening". Without an
  attack and with screening off, each ours_* method learns exactly what its chain_* method learns (tested; for
  `ours_lr` at epsilon = inf).
- **Attacker.** One of the K hospitals. It is chosen per repeat from the repeat seed by the existing
  `attacks.byzantine.attacker_ids` (attacker_frac = 1/K), is malicious in every round from the first, and is
  recorded in `config.json` (`attacker_ids_by_repeat`). Attacks happen before the commitment, so what is committed is
  what the attacker sent.
- **Attacks** (`survival/poison.py`). Logistic, the existing rules unchanged: `sign_flip` (delta -> -delta),
  `scaling` (delta -> K x delta), `label_flip` (training labels y -> 1 - y, class weight recomputed locally). Trees:
  `label_flip_trees` grows the attacker's new trees on its own rows with flipped labels (AFT: event indicator flipped,
  times unchanged). `random_trees` grows them on random rows of the public probe distribution, paired with a random
  permutation of the attacker's labels: well-formed trees of the honest size that carry no patient information.
- **Probe set** (`survival/probe.py`, public and data-independent). 512 rows drawn over the fixed schema from the
  feature names alone. Numeric columns are uniform on [0, 1] in the encoded space, i.e. on [0, the fixed scaling
  constant]. A missing indicator is on with probability 1/8 (the value is then 0, as the encoder writes it). Each
  one-hot group gets one category drawn uniformly from the vocabulary. The seed depends on public inputs only, and
  the probe's SHA-256 is recorded in `config.json`.
- **Tree screen** (`survival/screening.py`). Each hospital's new trees, rebuilt from the committed bytes, predict the
  probe set: the mean P(event) of its new trees (RF), or the margin they add (XGBoost). distance_k is the
  root-mean-square difference from the per-row median across hospitals. The score is the one-sided robust z of
  distance_k across hospitals, with the deployed probe's median and 1.4826 x MAD (`forensics.robust_z` ->
  `eval_lib.bv_find_outliers`). `eval_lib.bv_layer_health` on the predictions and on the leaf values raises
  Forensics' hard flags (NaN, Inf, |x| > 100), which reject a hospital outright. A hospital is rejected if a hard
  flag fires or its score > tau.
- **tau is never hardcoded.** `analysis/calibrate_survival.py` runs short ours_* cells with tau = inf (scored, never
  rejected on tau) at a separate seed (1000). It covers one clean cell and, for every attack of the family, one cell
  per hospital as the pinned attacker. It labels each hospital-round 1 if attacker, else 0, builds the ROC with
  eval_lib's ROC (through `analysis/calibrate.py`'s `curve_stats`), and writes
  `results/calibration/taus_survival.json` keyed by (dataset, family, epsilon, K), with the AFT task as family
  `xgb_aft`. A screening run reads exactly its key (policy `tau_fpr5`: the threshold with the highest TPR at
  FPR <= 5%), raises if it is missing, and records `tau` and `tau_source` in `config.json`. A screen constructed
  without tau raises. The smoke calibration writes `taus_survival_smoke.json`, which no run reads by default.
- **Detection** (per round, the keys of the existing `_attack_report`, with hospital codes). TP = attacker rejected,
  FP = honest hospital rejected, FN = attacker kept. Per repeat these are pooled over hospital-rounds: precision =
  TP / (TP + FP), recall = TP / (TP + FN), honest-hospital false-exclusion rate = FP / honest hospital-rounds. Tables
  report mean ± std over repeats, and `summary.json` also keeps the totals over all repeats.

## 5. The reference paper and this work

### 5.1 Three layers

| Layer | Myrzashova et al. | This work |
|---|---|---|
| Local training | Each hospital trains a customized ResNet on chest X-ray images (NIH ChestX-ray, 15 labels; six clients, rounds of several epochs). | Each hospital trains logistic regression (optionally with DP-SGD), random forests or XGBoost on tabular data (UCI heart disease, TCGA-BRCA clinical). Updates are deltas (logistic) or the trees added that round. |
| Blockchain | An Ethereum chain with proof-of-work, simulated in Python: Solidity contracts reached through web3.py, mining simulated in Python, and each update's hash sent in a transaction. Clients check a new block's hash before using it. | The existing BlockVerify proof-of-work chain (also a local simulation, one node). Each update is serialized without pickle, content-addressed, signed with the hospital's registered Ed25519 key and verified. The round is committed as a Merkle root of `[i, hospital, digest]` with inclusion proofs and anchored. Hospitals verify the previous block before training and skip the round, recorded, if it fails. |
| Aggregator | FedAvg weighted by n_i/N over homomorphically encrypted updates: the server aggregates without seeing individual models, then decrypts the result. | FedAvg, tree bagging or FedXGBBagging over the updates rebuilt from the committed bytes, after screening (Forensics for logistic updates, the probe screen for trees). Updates are not encrypted; DP-SGD protects the logistic model's training examples instead. |

### 5.2 What the reference paper does and what we add

| Aspect | Reference paper | This work |
|---|---|---|
| Privacy of updates | Homomorphic encryption: the server only handles encrypted updates. | DP-SGD on the logistic model (per-example ε). No HE, so the server sees each update; trees have no protection (THREAT_MODEL.md). |
| Chain | Ethereum-style proof-of-work simulated in Python; a block hashes the previous hash, the update hash, a timestamp and a nonce. | The existing BlockVerify proof-of-work chain plus Merkle round roots with inclusion proofs. Public anchoring exists in FedVerify (Algorand note, Bitcoin checkpoint) but these experiments use the mock and local anchors only. |
| Transaction ids | A hash of the hospital id, a counter and a timestamp. | The same idea, but the verifier enforces the counter against registered keys, and the signature covers every field, txn id and timestamp included. |
| Attacks evaluated | Double spending, transaction malleability, endpoint compromise (chain layer only). | The same three, simulated against our commit layer, each with a defence-off control. Added: poisoning by a malicious hospital, and screening, with tau calibrated from data. |
| Endpoint defences | Multi-factor authentication, firewalls, intrusion detection. | Registered-key verification only; MFA and firewalls are not simulated. |
| Metrics | Accuracy, loss, latency, throughput, send rate, resilience. | The same, plus F1, ROC-AUC, C-index, per-hospital results, DP ε, detection precision and recall, and false exclusion. |

### 5.3 Units

The reference paper's Table IV prints its latency as "43s", while its abstract and text give the latency in
milliseconds, and the text gives two different decimal expansions. Its throughput in the abstract (bytes per
second) and in the conclusion (MB per second) also differ by orders of magnitude. We report latency in milliseconds
only, as the mean time between consecutive blocks, and throughput in bytes per second (section 3). Their numbers
appear in our tables only as "reported".

### 5.4 Consortium chains (Hyperledger Fabric, Quorum)

A consortium (permissioned) chain would be run by the hospitals themselves: known validators, consensus without
mining (BFT- or Raft-style ordering), access control, private channels, and contract logic that could enforce the
aggregation rules. We do not use one (Revision 2 excludes Hyperledger and Quorum). Our chain is the existing
single-node BlockVerify proof-of-work chain. It gives tamper evidence of the round records relative to the anchored
roots, not decentralised agreement among the hospitals. In a deployment the round roots would be anchored on a
consortium chain run by the hospitals, or on a public chain. The commit layer (signed submissions, content IDs,
Merkle roots, inclusion proofs) would stay as it is; only the anchor backend changes.

## 6. What is not built, and why

- DP for tree models (random forest, XGBoost): DP-SGD does not apply to trees; those cells are `n/a`.
- "FPoW": undefined; every commit uses the existing proof-of-work chain (Revision 2 §9).
- Adaptive attackers, more than one Byzantine hospital, multi-factor authentication and firewalls: see
  `THREAT_MODEL.md`.

## 7. Phase 7 audit corrections

Four points from an external audit, all handled in code and generated text, not by editing numbers:

1. **TCGA accuracy and the majority class.** Most TCGA patients are alive (the majority-class rate is printed in the
   baseline notes of every TCGA table), so accuracy near that rate says nothing about a model. Every TCGA table and figure now has the "predict all alive" row or line (section 1), and the
   draft and the showcase never rank accuracies within 0.01 of the majority-class rate (`guarded_relation` in
   `make_results_draft.py`). **Why accuracy and F1 disagreed:** they did use different thresholds. F1 used the
   training-fold F1-optimal threshold. Heart's pooled accuracy and every per-hospital, per-round accuracy (Figs 5-6,
   both datasets) used P(event) >= 0.5, while TCGA's pooled accuracy, which the run does not list, was computed by the
   table code from the out-of-fold predictions, i.e. at the F1 threshold. At 0.5 the TCGA XGBoost curves sat close to
   each hospital's majority-class rate; at the F1 threshold the pooled accuracies lie below the baseline. Both now use
   the F1 rule everywhere (section 1).
2. **FedXGBBagging summed its boosters.** Corrected to averaging (section 2), applied everywhere, rerun.
3. **Screening is reported with its cost.** The attack tables show recall next to the clean run's false-exclusion
   rate. Where that rate is above `clean_false_exclusion_max` (5%), no screened (or Ours) result is written as
   "outperforms": the gain may come from dropping a skewed hospital's updates. The recall range per dataset is
   generated from the results and stated in the draft, its Limitations and the showcase.
4. **Chain-attack resilience is 100% by construction.** Each simulated variant meets a deterministic check, so the
   number lists what is covered; it is not comparable with the reference paper's 91 / 88 / 81%. The captions list the
   covered variants (from the summaries) and the uncovered ones (`chain_attacks.not_covered` in `table.yaml`), and note
   that our proof-of-work difficulty (2, one local node, milliseconds per block) is not comparable with the reference
   paper's ~5 s blocks.

## 8. Phase 8 diagnosis: TCGA centralized XGBoost below local and federated

The numbers below come from `results/diagnostics/central_xgb_tcga.json`, written by
`python3 -m fedverify.analysis.diagnose_central_xgb` (experiment `diagnostics`). It replays local_xgb, central_xgb and
fedxgb_bagging on the runner's exact folds and seeds, through the same `fl_xgb.run_fold`, and checks that its
final-round pooled values equal the summaries on disk (they do, to 1e-9). Nothing was changed for this diagnosis.

Training vs held-out (mean over 5 repeats; training = the fold's training rows, held-out = pooled out-of-fold;
C-index for AFT, ROC-AUC for the death indicator), round 1 → round 30:

| Run | Steps / round | Trees at r30 | Steps at r30 | Training | Held-out | Held-out at 150 steps |
|---|---|---|---|---|---|---|
| local_xgb_aft | 5 | 750 | 150 | 0.825 → 0.988 | 0.628 → 0.650 | 0.650 |
| central_xgb_aft | 25 | 750 | 750 | 0.830 → 0.993 | 0.666 → 0.565 | 0.602 |
| fedxgb_bagging_aft | 5 | 750 | 150 | 0.765 → 0.853 | 0.649 → 0.637 | 0.637 |
| local_xgb | 5 | 750 | 150 | 0.879 → 0.988 | 0.693 → 0.691 | 0.691 |
| central_xgb | 25 | 750 | 750 | 0.813 → 1.000 | 0.595 → 0.540 | 0.565 |
| fedxgb_bagging | 5 | 750 | 150 | 0.712 → 0.772 | 0.615 → 0.600 | 0.600 |

Final-round C-index (AFT) over all comparable pairs, same-site pairs and cross-site pairs (mean over repeats):

| Run | All | Same site | Cross site |
|---|---|---|---|
| local_xgb_aft | 0.650 | 0.601 (4390 pairs) | 0.666 (13193 pairs) |
| central_xgb_aft | 0.565 | 0.553 (4390 pairs) | 0.568 (13193 pairs) |
| fedxgb_bagging_aft | 0.637 | 0.620 (4390 pairs) | 0.643 (13193 pairs) |

Sites (deaths / patients): BH 45/145 (31%, median censored follow-up 1120 d), A2 11/98 (11%, median censored follow-up 1347 d), E2 6/87 (7%, median censored follow-up 1062 d), A8 4/76 (5%, median censored follow-up 518 d), AR 10/68 (15%, median censored follow-up 2862 d). A predictor that knows only the training fold's death rate of the patient's site
reaches C-index 0.585 and ROC-AUC 0.700.

**Diagnosis.**

1. *Over-fitting, from round 1.* central_xgb_aft's training C-index rises to 0.993 while its
   held-out C-index falls from round 1 on (it rises in only four of 29 round-to-round steps, each time by less than
   0.001). After round 1 the central AFT model has the highest held-out value of the three (first held-out
   value in the table; a per-round mean with no paired interval, so not a ranking); it ends lowest. The binary
   central model reaches a training ROC-AUC of 1.000: it memorises the training deaths.
2. *Equal trees, unequal boosting steps.* The three methods end with the same 750 trees: that is the
   documented budget ("central_xgb boosts hospitals x N trees per round", section 1). Since the Phase 7 averaging,
   a federated round moves the margin by one averaged N-tree booster, and a local round by N trees, while a central
   round boosts K x N = 25 trees in sequence. By round 30 central has taken 750 sequential steps at learning
   rate 0.1, the others 150. Before Phase 7, federated rounds summed K boosters, so a federated round moved the
   margin about as far as a central round, and the two scored alike (superseded results). The averaging correction
   created the mismatch.
3. *Steps explain part of it; whether more remains is not shown.* At the same 150 steps central's mean is below
   local's and federated's (last column above). But the step-matched diagnostic row (section 9), run at the federated
   step count, is not distinguishable from either by the paired rule (Phase 12; RESULTS_DRAFT.md, Limitations). That
   one pooled ensemble over-fits faster per step than the average of K boosters, each fitted to one site (which acts
   like bagging), is a possible explanation of a remaining gap, not a measured one.
4. *Site censoring and event rates explain local's lead, not central's drop.* Death rates differ by site, and local
   models carry their site's baseline, so local is strong on cross-site pairs. On the death indicator, local's
   ROC-AUC (0.691) is no better than the site-only predictor's (0.700), so its lead there is the site death rates.
   Within hospital, federated XGBoost still outperforms central by the paired rule, while local is not
   distinguishable from it (Phase 12; RESULTS_DRAFT.md, Limitations), so the site mix does not explain central's
   drop relative to federated.

**Not a configuration bug.** The code does what the design says (equal tree count), and no setting is wrong. Matching
central to the federated *step* count instead is a different budget definition. Choosing it now, after seeing that it
would raise the held-out score, would be tuning on test results. So central_xgb was left unchanged and nothing was
re-run. If a step-matched centralized baseline is wanted, it should be added as an extra, clearly labelled row (an
open decision in TODO.md), with its budget rule fixed before its result is read.

## 9. Phase 9: reference rows and the step-matched diagnostic row

Numbers in this section are computed from the results files by the table code (`Results.site_rate`, the summaries);
verdicts use the comparison rule (A outperforms B only if the difference of the means exceeds both standard
deviations).

**Site base rate only (reference).** Every method table now has this row next to "Predict all alive". Each patient's
score is their hospital's event rate among the training rows of their fold-run (the other folds of the same repeat),
with no features and no model (`site_rate_oof`, `table.yaml` `site_rate`). F1 and accuracy use the same threshold rule
as every other row. Pooled values (mean ± std over repeats):

| Dataset | ROC-AUC | F1 | Accuracy | C-index |
|---|---|---|---|---|
| UCI heart disease | 0.674 ± 0.000 | 0.682 ± 0.000 | 0.518 ± 0.000 | n/a |
| TCGA-BRCA | 0.700 ± 0.000 | 0.407 ± 0.000 | 0.724 ± 0.000 | 0.585 ± 0.002 |

- On TCGA, no method's ROC-AUC or F1 exceeds this reference by the rule. The highest ROC-AUC is local_rf
  (0.691 ± 0.016) and the highest F1 is local_xgb (0.370 ± 0.024); by the paired rule (Phase 12) neither is
  distinguishable from its runner-up, local_xgb and local_rf respectively. Pooled TCGA ROC-AUC and F1
  therefore largely measure how the death rate differs between sites. The C-index, which also uses follow-up times, is
  where several TCGA methods exceed it (the draft lists them).
- On heart every method exceeds it on ROC-AUC by a wide margin, so heart's pooled ROC-AUC is not a site artefact.
- Its ROC-AUC, F1 and accuracy have a standard deviation of 0 over repeats. That is expected: stratified folds on
  (hospital, outcome) put the same number of each site's events in each fold in every repeat, and a score that only
  knows the site depends on nothing else. Its C-index varies because it also depends on the follow-up times.

**Centralized, step-matched (diagnostic).** Rule fixed in `table.yaml` (`diagnostic_rows.central_step_matched`) on
2026-10-15, before any of its runs existed. Method `central_xgb_steps`: centralized XGBoost on the pooled rows with N
trees per round (one federated booster's worth), so 30 rounds give the federated and local models' 150 sequential
boosting steps. Every other setting and seed is `central_xgb`'s, and `central_xgb` itself is unchanged. Not blind on
TCGA (the Phase 8 diagnosis had shown the original model's score at 150 steps); heart had not been diagnosed.

| Dataset, metric | Centralized | Step-matched | Vanilla FL (FedXGBBagging) |
|---|---|---|---|
| TCGA-BRCA, C-index (AFT) | 0.565 ± 0.016 | 0.600 ± 0.020 | 0.637 ± 0.010 |
| TCGA-BRCA, ROC-AUC | 0.540 ± 0.009 | 0.563 ± 0.012 | 0.600 ± 0.007 |
| UCI heart disease, ROC-AUC | 0.846 ± 0.003 | 0.862 ± 0.005 | 0.843 ± 0.005 |

By the rule, the step-matched model outperforms the original on TCGA C-index and is outperformed by federated XGBoost
AFT. On heart (a blind result) it outperforms the original centralized model and outperforms federated XGBoost.
*(Old mean ± std rule. Under the Phase 11 interval rule the step-matched row still outperforms the original on both
datasets and federated XGBoost on heart, but on TCGA federated XGBoost AFT is **not distinguishable** from it: see the
paired-comparison tables and section 11.)*
Matching steps moves central XGBoost up on both datasets, but this is a diagnostic: the Centralized row of the study
stays the documented equal-tree budget, and the step-matched row is never used for the "federated vs centralized"
verdicts.

## 10. Phase 11: within-hospital metrics, paired statistics, the LaTeX check, the freeze

Numbers in this section are read from RESULTS_DRAFT.comparisons.json, the tables and results/tables/latex_check.json by the script that wrote it.

**Within hospital (11.1).** Pooled ROC-AUC and C-index also rank patients of different hospitals against each other, so they partly measure how outcome rates differ between hospitals. The method tables now have secondary columns computed on each hospital's own out-of-fold patients, combined weighted by their comparable pairs (`analysis/paired_stats.py`; equal to the pair-weighted mean of the per-hospital metrics, tested). Hospitals with fewer than 5 patients of their rarer class are left out, the per-hospital tables' rule (`table.yaml` `within_hospital`): Switzerland (1 negative); A8 (4 deaths). The request said "fewer than 5 events"; for TCGA the rarer class is the deaths, so it is the same rule, and on heart it also leaves out Switzerland, whose 1 negative makes its ROC-AUC rest on one patient.

| Run | Pooled | Within hospital |
|---|---|---|
| TCGA-BRCA survival, XGBoost AFT, Local | 0.650 ± 0.020 | 0.595 ± 0.021 |
| TCGA-BRCA survival, XGBoost AFT, Vanilla FL | 0.637 ± 0.010 | 0.614 ± 0.013 |
| TCGA-BRCA survival, random forest, Vanilla FL | 0.674 ± 0.012 | 0.666 ± 0.012 |
| TCGA-BRCA death indicator, XGBoost, Local (ROC-AUC) | 0.691 ± 0.014 | 0.612 ± 0.020 |
| UCI heart, logistic regression, Vanilla FL (ROC-AUC) | 0.859 ± 0.001 | 0.847 ± 0.002 |

Local models lose most when only same-hospital pairs count: part of their pooled score was the site base rate.

*Sanity check, and a finding.* A score that is constant within a hospital scores exactly 0.5 within hospital (tested; Predict all alive does). The Site base rate only row does **not**: it scores 0.493 ± 0.000 (heart roc_auc), 0.488 ± 0.000 (TCGA roc_auc), 0.485 ± 0.005 (TCGA c_index). Its rate is computed on the other folds, so whenever a hospital's events do not split evenly over the 5 folds the fold holding an extra event gets a slightly lower rate, which ranks that fold's own events lower. Restricted to pairs within one fold, where its score is constant per hospital, it scores exactly 0.5 (tested), so the gap is that fold artefact and nothing else. Neither the metric nor the Phase 9 row was changed to hide it.

**The comparison rule (11.2).** Methods share folds and repeats, so they are compared in pairs: per repeat d_r = metric(A_r) - metric(B_r), reported as mean ± sample std over the 5 repeats, and a 95% patient-level bootstrap interval (2000 resamples stratified by hospital, seed 0, the same resamples for every comparison of a dataset) of the mean over repeats of the difference. Every metric is a weighted pair sum, so a resample is a weighted quadratic form; with unit weights it reproduces every published pooled number (largest difference over 1,166 values: 2.2e-16) and lifelines' C-index tie rule (event-censored ties at the same time are comparable). No p-values: with 5 repeats an exact test cannot go below 0.0625. **The generated verdict rule changed** from "the difference of the means exceeds both standard deviations" to **"outperforms only if the bootstrap CI of the paired difference excludes 0"** (`ci_relation` in make_results_draft.py); "comparable" is gone, because an interval containing 0 does not show equivalence. The accuracy-majority guard (Phase 7) and the clean-false-exclusion caveat (Phase 7) still apply on top.

56 of 124 generated verdicts changed wording (`paper/VERDICT_CHANGES.md` lists every one):

- 18 × "is outperformed by" → "is not distinguishable at this sample size from"
- 16 × "outperforms" → "is not distinguishable at this sample size from"
- 15 × "is comparable with" → "is not distinguishable at this sample size from"
- 4 × "is comparable with" → "outperforms"
- 1 × "is not distinguishable at this sample size from" → "is outperformed by"
- 1 × "is comparable with" → "is outperformed by"
- 1 × "is not distinguishable at this sample size from" → "outperforms"

The ones that change a ranking (old or new verdict is outperforms / is outperformed by):

| Section | A vs B | Metric | Paired difference | 95% CI | Before | Now |
|---|---|---|---|---|---|---|
| methods_heart/lr | ours vs vanilla FL | roc_auc | -0.004 ± 0.002 | -0.008 to +0.000 | is outperformed by | is not distinguishable at this sample size from |
| methods_heart/rf | vanilla FL vs local | roc_auc | -0.005 ± 0.006 | -0.016 to +0.006 | is outperformed by | is not distinguishable at this sample size from |
| methods_heart/xgb | ours vs vanilla FL | roc_auc | -0.005 ± 0.003 | -0.009 to -0.001 | is not distinguishable at this sample size from | is outperformed by |
| methods_tcga_survival/lr | vanilla FL vs local | c_index | +0.035 ± 0.015 | -0.035 to +0.106 | outperforms | is not distinguishable at this sample size from |
| methods_tcga_survival/lr | ours vs vanilla FL | c_index | -0.057 ± 0.043 | -0.115 to +0.001 | is outperformed by | is not distinguishable at this sample size from |
| methods_tcga_survival/rf | vanilla FL vs local | c_index | +0.039 ± 0.017 | -0.034 to +0.111 | outperforms | is not distinguishable at this sample size from |
| methods_tcga_survival/rf | ours vs vanilla FL | c_index | -0.005 ± 0.003 | -0.009 to -0.001 | is comparable with | is outperformed by |
| methods_tcga_survival/xgb | ours vs vanilla FL | c_index | -0.011 ± 0.010 | -0.023 to +0.001 | is outperformed by | is not distinguishable at this sample size from |
| methods_tcga_binary/site_rate | Random forest: Vanilla FL vs Site base rate only (reference) | roc_auc | -0.088 ± 0.013 | -0.180 to +0.010 | is outperformed by | is not distinguishable at this sample size from |
| methods_tcga_binary/site_rate | Random forest: Blockchain-FL (commit only) vs Site base rate only (reference) | roc_auc | -0.088 ± 0.013 | -0.180 to +0.010 | is outperformed by | is not distinguishable at this sample size from |
| methods_tcga_binary/site_rate | Random forest: Ours vs Site base rate only (reference) | roc_auc | -0.092 ± 0.016 | -0.185 to +0.006 | is outperformed by | is not distinguishable at this sample size from |
| methods_tcga_binary/site_rate | Random forest: Local vs Site base rate only (reference) | f1 | -0.044 ± 0.019 | -0.133 to +0.040 | is outperformed by | is not distinguishable at this sample size from |
| methods_tcga_binary/site_rate | XGBoost: Local vs Site base rate only (reference) | f1 | -0.037 ± 0.024 | -0.127 to +0.052 | is outperformed by | is not distinguishable at this sample size from |
| methods_tcga_survival/site_rate | Logistic regression: Local vs Site base rate only (reference) | c_index | -0.046 ± 0.018 | -0.126 to +0.036 | is outperformed by | is not distinguishable at this sample size from |
| methods_tcga_survival/site_rate | Random forest: Local vs Site base rate only (reference) | c_index | +0.050 ± 0.014 | -0.016 to +0.115 | outperforms | is not distinguishable at this sample size from |
| methods_tcga_survival/site_rate | XGBoost AFT: Local vs Site base rate only (reference) | c_index | +0.065 ± 0.019 | -0.024 to +0.150 | outperforms | is not distinguishable at this sample size from |
| methods_tcga_survival/site_rate | Random forest: Centralized vs Site base rate only (reference) | c_index | +0.031 ± 0.010 | -0.063 to +0.123 | outperforms | is not distinguishable at this sample size from |
| methods_tcga_survival/site_rate | XGBoost AFT: Centralized vs Site base rate only (reference) | c_index | -0.020 ± 0.016 | -0.127 to +0.085 | is outperformed by | is not distinguishable at this sample size from |
| methods_tcga_survival/site_rate | Random forest: Vanilla FL vs Site base rate only (reference) | c_index | +0.089 ± 0.012 | -0.019 to +0.195 | outperforms | is not distinguishable at this sample size from |
| methods_tcga_survival/site_rate | XGBoost AFT: Vanilla FL vs Site base rate only (reference) | c_index | +0.052 ± 0.011 | -0.059 to +0.161 | outperforms | is not distinguishable at this sample size from |
| methods_tcga_survival/site_rate | Logistic regression: DP-FL vs Site base rate only (reference) | c_index | -0.068 ± 0.034 | -0.169 to +0.024 | is outperformed by | is not distinguishable at this sample size from |
| methods_tcga_survival/site_rate | Random forest: Blockchain-FL (commit only) vs Site base rate only (reference) | c_index | +0.089 ± 0.012 | -0.019 to +0.195 | outperforms | is not distinguishable at this sample size from |
| methods_tcga_survival/site_rate | XGBoost AFT: Blockchain-FL (commit only) vs Site base rate only (reference) | c_index | +0.052 ± 0.011 | -0.059 to +0.161 | outperforms | is not distinguishable at this sample size from |
| methods_tcga_survival/site_rate | Logistic regression: Ours vs Site base rate only (reference) | c_index | -0.068 ± 0.037 | -0.167 to +0.024 | is outperformed by | is not distinguishable at this sample size from |
| methods_tcga_survival/site_rate | Random forest: Ours vs Site base rate only (reference) | c_index | +0.084 ± 0.015 | -0.024 to +0.191 | outperforms | is not distinguishable at this sample size from |
| methods_tcga_survival/site_rate | XGBoost AFT: Ours vs Site base rate only (reference) | c_index | +0.041 ± 0.009 | -0.073 to +0.154 | outperforms | is not distinguishable at this sample size from |
| per_hospital_methods_heart/Hungary | LR, FedAvg vs LR, local | accuracy | +0.012 ± 0.016 | -0.003 to +0.028 | outperforms | is not distinguishable at this sample size from |
| per_hospital_methods_heart/Hungary | XGB, FedXGBBagging vs XGB, local | accuracy | +0.014 ± 0.006 | -0.015 to +0.042 | outperforms | is not distinguishable at this sample size from |
| per_hospital_methods_tcga/BH | LR, FedAvg vs LR, local | c_index | +0.030 ± 0.019 | -0.077 to +0.141 | outperforms | is not distinguishable at this sample size from |
| per_hospital_methods_tcga/E2 | XGB AFT, FedXGBBagging vs XGB AFT, local | c_index | -0.071 ± 0.053 | -0.187 to +0.042 | is outperformed by | is not distinguishable at this sample size from |
| per_hospital_methods_tcga/A2 | XGB AFT, FedXGBBagging vs XGB AFT, local | c_index | +0.097 ± 0.057 | -0.104 to +0.270 | outperforms | is not distinguishable at this sample size from |
| per_hospital_methods_tcga/AR | XGB AFT, FedXGBBagging vs XGB AFT, local | c_index | +0.103 ± 0.053 | -0.024 to +0.236 | outperforms | is not distinguishable at this sample size from |
| dp_sweep_heart | eps=4 vs no DP | roc_auc | -0.004 ± 0.002 | -0.008 to +0.000 | is outperformed by | is not distinguishable at this sample size from |
| dp_sweep_heart | eps=8 vs no DP | roc_auc | -0.002 ± 0.001 | -0.006 to +0.001 | is outperformed by | is not distinguishable at this sample size from |
| dp_sweep_tcga | eps=1 vs no DP | c_index | -0.067 ± 0.048 | -0.135 to +0.004 | is outperformed by | is not distinguishable at this sample size from |
| dp_sweep_tcga | eps=2 vs no DP | c_index | -0.064 ± 0.048 | -0.126 to +0.000 | is outperformed by | is not distinguishable at this sample size from |
| attack_heart/lr/sign_flip | screened vs unscreened | roc_auc | +0.155 ± 0.143 | +0.136 to +0.173 | is not distinguishable at this sample size from | outperforms |
| attack_heart/lr/label_flip | screened vs unscreened | roc_auc | +0.025 ± 0.025 | +0.021 to +0.030 | is comparable with | outperforms |
| attack_heart/rf/label_flip_trees | screened vs unscreened | roc_auc | +0.006 ± 0.005 | +0.004 to +0.008 | is comparable with | outperforms |
| attack_heart/rf/random_trees | screened vs unscreened | roc_auc | +0.005 ± 0.003 | +0.002 to +0.007 | is comparable with | outperforms |
| attack_tcga/xgb/random_trees | screened vs unscreened | c_index | +0.012 ± 0.013 | +0.002 to +0.022 | is comparable with | outperforms |

*Caveat.* The interval holds the 5 trained runs fixed and resamples patients; it leaves out the variation between training runs. In 13 of 124 comparisons the std of the paired difference over repeats exceeds half the interval's width, i.e. training randomness matters more than patient sampling and the interval alone overstates certainty; the paired-comparisons table marks such rows with † (none of its key comparisons is one). The clearest case is screening under attack, where the attacker hospital changes from repeat to repeat.

**LaTeX (11.3).** Every generated .tex table (27, the earlier MNIST study's included) was compiled alone inside an IEEE two-column conference document with Tectonic 0.17.0 (installed in a scratch folder only): 27 of 27 compile with no error and no overfull box (`results/tables/latex_check.json`, `results/tables/latex_check/all_tables.pdf`). How: every tabular sits in a box that is shrunk to the line width only when it is wider (needs `graphicx`); attack_heart shrinks to 94%, attack_tcga shrinks to 90%; methods_heart and methods_tcga_binary are split into two floats in LaTeX only (all columns kept: accuracy, F1 and ROC-AUC, then pooled and within-hospital ROC-AUC side by side), because one float needed 70%. The MNIST tables had a bug: a "%" in a caption started a LaTeX comment and broke the file, and raw ε, ±, α, ✓, ✗ were not escaped. Their emitter now escapes everything; since the results behind tables 2, 4, 5 and 6 (exp2, exp4, exp5) are no longer on disk, their .tex was re-written from their own generated .md (`make_tables --retex`), which keeps every number; table5's long headers now wrap instead of shrinking. *(Corrected in Phase 12, section 11: tables 2, 2b, 2_full, 4 and 6 were never run, every cell is empty, and table 5 is built from comparison.yaml, not from a run.)*

**Report pack and freeze (11.4).** `make_report_pack` copies the final figures (PNG, 300 dpi, Times New Roman) and every table (CSV, Markdown, LaTeX) into `results/report_pack/`, named by report section, with a README mapping each file to the VIT report's chapter 4 and to slides 8 and 11 (`table.yaml` `report_pack`). `make_manifest` writes `results/MANIFEST.json` (SHA-256 of every results file, the Python version, the `pip freeze` as `results/requirements-lock.txt`) and mines the Merkle root of the file hashes into a block of a fresh local proof-of-work chain (`results/MANIFEST.chain.json`); `make_manifest --verify` re-checks every file, the root and the block. The live demo's folder is left out (every demo run rewrites it); `_superseded/` stays until the guide approves the numbers.

## 11. Phase 12: wording rules made in the generators

Each rule is applied in one place (`key_comparisons`, `verdict_cell`, `verdict_phrase` and `central_gap_clause` in
`analysis/make_results_draft.py`), so the tables, the results draft and the showcase page say the same thing;
`fedverify/tests/test_phase12_wording.py` checks each one. No experiment was re-run and no setting changed.

- **A screened gain that may be a skew artefact is not ranked.** The Phase 7 rule (a gain of Ours whose screen excludes
  more than `clean_false_exclusion_max` of honest hospital-rounds on a clean run is never written as outperforming)
  now also covers the paired-comparison tables and the draft's paired paragraph, which counts such a comparison apart:
  its interval excludes 0, but it is not ranked. It applies to the heart random forest.
- **"Highest score", never "best".** A top score is compared with its runner-up through the same paired interval; a
  run with identical predictions (a commit-only twin) is not a runner-up, and diagnostic rows are not ranked. Each
  within-hospital top is not distinguishable from its runner-up.
- **Within-hospital claims go through the interval.** New table `paired_comparisons_within`: the 20 key comparisons on
  the within-hospital metrics, with the same resamples and rule, next to the pooled verdict. The draft's earlier
  "largest fall from pooled to within hospital" ranking, which had no interval, is gone.
- **A federated gain over centralized training on TCGA is qualified in the same sentence.** For XGBoost: the
  centralized model over-fits (section 8), and how federated training relates to the step-matched row comes from that
  comparison's own interval (not distinguishable on the C-index and ROC-AUC, pooled and within hospital; on F1
  federated still outperforms it, and the page says so). For the random forest the sentence says the gap was not
  diagnosed: section 8 covered XGBoost only, there is no step-matched random forest, and the centralized random
  forest's held-out C-index does not fall over the rounds, so over-fitting is not claimed.
- **Earlier statements corrected.** Section 8 points 1, 3 and 4 and section 9's "best" and old-rule verdicts now say
  what the interval rule supports (the generated draft was corrected the same way). Section 10 said the results behind
  MNIST tables 2, 4, 5 and 6 were "no longer on disk"; in fact experiments exp2, exp4 and exp5 were never run in this
  repository (every cell of tables 2, 2b, 2_full, 4 and 6 is empty, their captions now say so, and TODO.md lists the
  commands that would fill them), and table 5 is the qualitative table built from `analysis/comparison.yaml`, which
  regenerates unchanged.
- **Diagrams.** `analysis/render_diagrams.py` renders the six Mermaid sources to SVG and to PNG at 300 dpi (at each
  diagram's natural size) with mermaid-cli and the installed Chrome; `docs/diagrams/rendered.json` records the hashes
  and versions, and the report pack copies only renders that match their source. Rendering exposed a source bug: in a
  sequence diagram a `;` inside a message ends the line, which had broken `sequence_round.mmd`.
