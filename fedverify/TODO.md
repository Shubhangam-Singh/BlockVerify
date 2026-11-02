# FedVerify — deferred items

Ideas and issues parked here per the `## Scope` rule in `CLAUDE.md`. Nothing here is
in scope for the current phase.

## Pre-existing BlockVerify issues (found in Phase 0, NOT fixed — out of FedVerify scope)

See `INVENTORY.md` §13 for detail. Listed here so they are not mistaken for FedVerify bugs.

- [ ] `_get_pow_chain()` reads `m.get("name")`/`m.get("hash")` but the registry stores
      `modelName`/`modelHash` (`backend/app.py:135,137`) → chain `register` txs carry
      `"Unknown"` / `""`. Fixing changes existing chain content; needs the human's call.
- [ ] `bitcoin_client` reports a 32-byte OP_RETURN script but broadcasts the 64-byte ASCII
      hex string (`backend/bitcoin_client.py:104` vs `:62`).
- [ ] `backend/requirements.txt` pins drift from installed versions; `bit` is installed but
      not listed.
- [ ] `docs/TAMPER_DETECTION.md:234` still says "82 pytest cases" (actual: 99).
- [ ] Stray files `backend/=2.8.0`, `backend/=4.1.0` from a malformed pip invocation.
- [ ] `backend/.gitignore:1` holds a wrong relative path that matches nothing.
- [ ] `backend/data/algo_wallet.json` stores the Algorand private key in plaintext
      (gitignored, but present on disk).

## Deferred from Phase 0

- [ ] Decide whether to free disk before Phase 1 (~2.4 GB free; ~6.4 GB in `~/.gemini`,
      3.4 GB in `~/.npm`, 2.7 GB in `~/Downloads`). CPU-only torch needs ~1 GB.

## Deferred out of Phase 3
- `docs/API_REFERENCE.md` does not yet document the five `/api/fl/*` routes. Documentation
  is Phase 6's remit; the routes already match its error envelope and status codes.
- No frontend surface for federated lineage (Phase 6: "Federated" tab).
- `ChainAnchor("algorand")` anchors the round root in a note transaction only. It does not
  write to the smart contract's global state the way `broadcast_hash_to_algorand` does for
  layer roots; global-state anchoring for FL rounds would let a verifier skip the indexer.
- Bitcoin checkpoints reuse `bitcoin_client.anchor_merkle_root`, which broadcasts a 64-byte
  ASCII message while reporting a 32-byte OP_RETURN script (pre-existing inconsistency
  recorded in INVENTORY §12). Not fixed here — it is outside FedVerify's scope.
- `POST /api/fl/round/commit` is unauthenticated, matching the other write routes in this
  codebase. Anyone who can reach the server can create a run id. Real deployments should
  put `@require_auth` on it (the decorator already exists in `backend/auth.py`).
- Round roots are stored per run in a single JSON file. Fine at experiment scale; a run
  with thousands of rounds and many clients would want a real store.

## Deferred out of Phase 4
- The combined score is `max(s_norm, s_dir, s_coord)`, mirroring the deployed detector's
  max-robust-z operating point. On the SMOKE calibration `s_coord` alone scored a higher
  AUC (0.982) than combined (0.913), because the max lets whichever sub-score is noisiest
  raise honest clients. Worth re-checking on the full calibration: if it holds there,
  a learned or per-score-gated combination is the obvious follow-up. Not changed on the
  strength of 54 smoke samples.
- `tau_coord` (the per-coordinate z inside s_coord) has a documented default of 3.0 and is
  recorded in config.json, but it is not itself calibrated — only the decision threshold
  tau is. A 2-D calibration over (tau, tau_coord) is future work.
- Attacks are non-adaptive: an attacker does not know the defence. An adaptive attacker
  who constrains its delta to stay inside the honest robust-z band is the natural strong
  baseline and is not implemented.
- exp2 is 576 cells at 30 rounds and K=10. Nothing about it is parallel-unsafe, but it is
  much larger than exp1 — expect to run it in filtered slices (`--attack`, `--aggregator`).
- `zero` is implemented and tested as a delta attack but is not in the exp2 attack grid
  (the spec's grid omits it). Add it with `--attack zero` if wanted.

## Deferred out of Phase 5
- **Finding to confirm at scale (exp5).** In the Phase-5 demo the scaling attacker became
  EASIER to detect as epsilon tightened, not harder: attacker s_norm rose 3.72 (eps=inf)
  -> 6.50 (eps=4) -> 7.39 (eps=1). The mechanism is visible in the raw norms — DP-SGD's
  per-sample gradient clipping bounds every honest update, so the honest norm band narrows
  and a fixed multiplicative attack lands further out in robust-z. This is the OPPOSITE of
  the natural hypothesis that DP noise masks attackers. It was observed on 4 rounds with a
  tau borrowed from a K=6 smoke calibration, so it is NOT yet a claim — the full exp5 with
  proper per-(eps,K) calibration must confirm or kill it before it goes near the paper.
- The demo also showed the flip side: with a stale tau, forensics rejected NOTHING at
  eps=inf (attacker combined 3.95 vs tau 4.36). That is the stale-threshold failure exp5's
  design anticipates, and the reason tau is looked up per epsilon.
- MIT-BIH class Q has only 15 beats corpus-wide once paced records are excluded, so its
  per-class F1 is near-meaningless and drags macro-F1 down by construction. Reporting
  4-class macro-F1 (N/S/V/F) alongside the 5-class number is worth considering.
- The MIT-BIH model uses lead 0 only. Two-lead input is the obvious extension.
- No patient-level DP: the DP accountant treats a beat as the unit of privacy, but the
  meaningful unit for a hospital is a PATIENT. Patient-level DP would need group privacy
  or per-patient clipping and is a genuine limitation to state in the paper.

## Deferred out of Phase 6
- **The Federated tab is read-only.** It visualises committed runs; it does not launch
  training or POST commitments. Pushing `commitments.jsonl` to `/api/fl/round/commit` is a
  manual step (a `--push-url` flag on the runner would close this).
- **`escapeHtml` was added for the new tab only.** The rest of `frontend/index.html`
  interpolates server strings into `innerHTML` unescaped — a pre-existing XSS surface that
  predates FedVerify. Not fixed here because it touches many existing code paths; worth a
  dedicated pass.
- Round metrics (`test_acc`, `macro_f1`) reach the API only if the commit payload includes
  them. The runner writes them to `rounds.jsonl` but the POST is manual, so the accuracy
  chart is empty for runs committed without them.
- `plots.py` fig3 draws the ROC from the three stored operating points in `taus.json`, not
  a full curve — `calibrate.py` writes the complete curve only as a PNG. Storing the raw
  (fpr, tpr) arrays would let fig3 render properly.
- No `self-healing rollback`. The lineage identifies a bad round; nothing reverts it.
  `table5` marks this ✗ for FedVerify and that is accurate.
- Bitcoin checkpoints default to a mock txid; `--btc-checkpoint` uses the real path but was
  never exercised against live testnet (no funded BTC UTXO on this machine).

## Fixed in the Phase-6 polish pass (was: deferred)
- ASR was computed and stored in rounds.jsonl but `push.py` never sent it, so the backdoor
  stat card was dead code. Now included in the metrics payload.
- `POST /api/fl/round/commit` accepted `attackers`/`metrics` as arbitrary types. A string
  there made the browser call `.map()` on a string and throw. Both are now type-checked,
  NaN/Inf metrics are dropped, and `round` is bounded.
- `run_id` was unvalidated. Ids containing `/` created runs that could never be read back
  (404 on every GET), and ids containing quotes broke out of the `onclick` attribute the
  Verify-proof button was built with — `escapeHtml` turns `'` into `&#39;`, which the HTML
  parser decodes back to `'` before the JS string is parsed. Ids are now charset-checked
  AND the button uses `data-*` attributes with event delegation, so no value is ever
  interpolated into executable markup.
- Step 4 of the proof panel reported the backend of round 1 for every round.
- `/api/fl/run/<id>` did not return `meta`, unlike the other read routes.
- An unreachable or stale API rendered as "no runs committed yet", sending the reader
  looking in the wrong place. Down / stale / genuinely-empty are now distinct states.

## Added in the comparison pass
- `meta.detector` is now sent by `chain/push.py`, read from the aggregator registry rather
  than duplicated in the UI. The comparison view needs to distinguish "made no per-client
  decision" (Krum top-1, median, trimmed-mean, FedAvg) from "decided and caught nothing";
  hardcoding that list in JavaScript would have been a second source of truth.
- `flDetection()` is the single implementation of the TP/FP/FN maths, shared by the
  single-run stat cards and the comparison table so the two views cannot disagree.
- F1 follows the sklearn zero-division convention: 0 when a detector caught nothing,
  null (rendered "—") only when the rule never decides. Reporting "—" for a detector that
  simply failed would understate the failure.

## Still deferred
- The comparison assumes both runs share a client count and round count. Mismatched runs
  render, but the accuracy chart pads the shorter series rather than aligning on round
  number. Fine for the committed demos; worth a guard before anyone compares arbitrary runs.
- Per-client `diag.scores` are still not exposed by the API, so the animation's z-scores
  remain illustrative and the comparison cannot show WHY a client was rejected.

## FedVerify-Survival — deferred out of Phase 0
Every count behind the points below is emitted by `fedverify/survival/phase0_check.py` and quoted in
`fedverify/survival/INVENTORY.md` section 4; they are deliberately not retyped here.

- **Zero follow-up looks like a placeholder, not a measurement.** Some censored patients have
  `days_to_last_follow_up == 0` while `OS_MONTHS` reports many months (several of the unit-check
  exceptions). The prompt's rule ("drop rows with no usable time") keeps them. A cleaning rule
  (treat 0 as missing, or drop) is a human decision; Phase 0 changes nothing.
- **Missingness differs by hospital.** The share of missing `lymph_node` and of `NA` ER/PR/HER2/
  histology cells varies strongly between the five sites, so the `<col>_missing` indicators and the
  `NA` categories can act as a site proxy for the pooled `central_lr` / `central_rf`. Possible
  follow-up: report the centralized baselines with the indicators ablated.
- **Death-indicator metrics treat short-follow-up censored patients as survivors.** ROC-AUC and F1
  are computed on the death indicator exactly as the prompt specifies, so a censored patient with
  little follow-up counts as a negative although they might have died later. A fixed-horizon label
  or a time-dependent AUC would be fairer. Out of scope (time-to-event models are on the prompt's
  out-of-scope list).
- **The five hospitals are chosen by patient count, not by deaths.** A site outside the five has more
  deaths than several inside it (INVENTORY section 4, sites table). The prompt fixes the five;
  selecting by events instead is a possible sensitivity analysis.
- **Out of scope per master prompt section 9 (recorded here, not to be built):** frontend changes;
  Hyperledger/Quorum; DP for random forests; Cox / time-to-event neural models; adaptive attackers;
  hyperparameter tuning; the NIH chest X-ray and heart-disease datasets; patient-level DP; real
  Algorand/Bitcoin runs beyond one demo round the human requests.
- Phase 1 adds the `--include-race` fairness note and Phase 2 the RF-DP idea (DP leaf counts /
  exponential-mechanism splits), as the prompt specifies.

## FedVerify-Survival — deferred out of Phase 1
- **Why race is off by default (`--include-race` turns it on).** Recorded race is a social category, not a
  biological one. In a survival model it mostly stands in for access to care, socioeconomic status and treatment
  differences, so a model that uses it can learn those disparities and present them as risk. The race groups
  other than white are also small within each hospital, so any race effect would be estimated from a handful of
  deaths, and "not reported" is a category of its own here. If it is ever switched on, report the metrics per race
  group next to the pooled ones rather than only the pooled numbers.
- **A linear model on these eight features is weak.** On the same folds, the logistic family lands near a standard
  scikit-learn logistic regression, and age alone ranks patients better than either. The prompt fixes the features
  and forbids tuning, so this is a finding for RESULTS_DRAFT, not something to fix. Possible later addition: an
  age-only baseline row so readers can see it.
- **The F1 threshold uses pooled training-fold predictions.** That is an evaluation-protocol step of the simulation;
  a deployed federation would have to pick the threshold without pooling per-patient predictions. Say so in
  DESIGN_NOTES (Phase 5).
- **Interpretations to state in DESIGN_NOTES (Phase 5):** `local_*` scores every patient with their own hospital's
  model and picks each hospital's own F1 threshold; `central_rf` grows hospitals x trees_per_hospital trees per round
  so it ends with the same tree budget as `fed_rf`; FedSGD uses the same learning rate as FedAvg.
- **Privacy accounting across folds and repeats (Phase 2 / THREAT_MODEL).** Every fold-run is accounted separately,
  but a patient sits in the training part of four of five fold-runs per repeat and of every repeat. The DP guarantee
  holds per released model; it is not a guarantee for the whole CV sweep.

## FedVerify-Survival — deferred out of Phase 2
- **Random-forest DP is not built (prompt Phase 2: "do not fake it").** Every RF-DP cell is `n/a`, and
  `SurvivalConfig` refuses an epsilon for the RF family. Ideas for a real one: (1) DP leaf counts: keep the tree
  structure from a public or randomised split rule and release each leaf's class counts with Laplace or Gaussian
  noise; (2) exponential-mechanism splits: choose each split with probability proportional to exp(eps * gain / 2*Delta).
  Both need their own budget split across trees, depth and rounds, and their own accountant; neither is DP-SGD.
- **Clipping partly undoes the class weighting.** Under DP every per-example gradient is clipped to C = 1 before
  noise is added, and a death's gradient is multiplied by pos_weight (often 5-20 here), so most deaths hit the clip
  and lose their extra weight. Worth one sentence in RESULTS_DRAFT next to the DP sweep.
- **Realised epsilon sits below target for small hospitals, in the safe direction.** sigma is solved for
  sample_rate = batch/n, but Opacus' Poisson loader samples at 1/ceil(n/batch), which is lower; the realised epsilon
  is therefore at or below the target. It is logged per hospital; S3 reports the maximum.
- **DP noise is seeded per (repeat, fold, round, hospital)**, not drawn from the global torch RNG (Opacus' default
  outside secure mode), so a DP fold-run is reproducible on its own. Secure-mode RNG (and its loss of reproducibility)
  is the production setting, out of scope here.
- **At large epsilon the DP cost is mostly clipping, not noise.** A near-noiseless smoke run (very large epsilon)
  still lands well below the non-private fedavg_lr: clipping every per-example gradient to C = 1 cancels the
  pos_weight of the deaths, so survivors dominate the update. C is fixed by the prompt and tuning is out of scope.
  Worth an ablation later: a clip-only run (noise multiplier 0) to split the cost into clipping and noise.
- **Opacus can warn "Optimal order is the largest alpha"** for large epsilon over few steps. The reported epsilon is
  then a valid but loose upper bound; extending the RDP orders would tighten it. Conservative, so left as is.
- **The ε = 1, 2, 4, 8 runs share their noise draws** (same seeds, scaled by sigma), so the sweep's columns are
  correlated within a repeat. That is fine for a sweep and is the reason mean ± std over repeats is what to read.
- **F1 barely separates methods (for RESULTS_DRAFT).** With the prompt's threshold rule, most methods' F1 sits close
  to the F1 of flagging every patient as a death (2p / (1 + p) for death prevalence p), across the whole DP sweep
  too. C-index and ROC-AUC carry the comparison; RESULTS_DRAFT should print that trivial F1 next to the table.

## FedVerify — Revision 2, deferred out of Phase 2b
- **FPoW undefined; awaiting guide.** Every round commit uses the existing BlockVerify proof-of-work chain
  (`--consensus pow`, the only option). If the guide defines FPoW it becomes a new phase (Revision 2 §9).
- **DP for XGBoost is n/a**, like random-forest DP: DP-SGD does not apply to boosted trees and nothing is faked.
  `SurvivalConfig` refuses an epsilon for the XGBoost family. A real version would need DP split selection and
  noisy leaf values with their own accountant (see the RF-DP note above).
- **SHAP feature importance (optional) is not built yet.** `shap` is not installed. XGBoost's own
  `predict(..., pred_contribs=True)` returns exact TreeSHAP values without the package, so the optional SHAP table
  can come from that for the XGBoost rows (Phase 5).
- **Heart placeholders are replicated, not cleaned.** Many `chol` values are 0 (mostly Switzerland) and one
  `trestbps` is 0; both are "missing" codes in the source, but paper mode keeps them to reproduce the reference
  paper's analysed sample. A cleaned sensitivity run would treat them as missing.
- **Reference CSVs for Tables III / IV.** Both `ieee_table_iii.csv` and `ieee_table_iv.csv` are in
  `fedverify/survival/` (not `ref/`). Phase 5 copies them to `fedverify/survival/ref/` and spot-checks them against
  the PDF before citing them.
- **FedXGBBagging adds corrections instead of averaging them.** Each hospital boosts from the same broadcast margin
  and the server sums all hospitals' trees (DESIGN_NOTES.md §2), so with more hospitals the effective step grows.
  An averaged-bagging ablation (scale each hospital's trees by 1/K) would show how much that matters.
- **TCGA runs made before Phase 2b have no `hospital_metrics`.** Their numbers are unchanged (checked), but
  Figs 5 and 6 for TCGA need the TCGA sweeps re-run once with the Phase-2b code.
- **The logistic loss on heart uses each hospital's own class weight**, as on TCGA. Switzerland, with almost no
  negatives, therefore weighs its few negatives very heavily; worth one sentence next to the heart tables.

## FedVerify — Revision 2, deferred out of Phase 3
- ~~**Equivocation (double spending) is not checked yet.**~~ Done in Phase 4: one digest per (run, round, hospital),
  plus round binding of the submission and of the payload header (`survival/chain_round.py`).
- **`--ipfs real` was never exercised against a running daemon** (none on this machine). Without one it degrades
  to a recorded error and changes nothing, which is tested.
- **The Bitcoin checkpoint is off for Survival runs** (`checkpoint_every=0`): the round roots are anchored on the
  PoW chain only. A real Algorand or Bitcoin run stays out of scope unless the guide asks for one demo round.
- **Submission timestamps are a logical clock**, not wall-clock time, so runs stay reproducible. A deployment would
  use real timestamps and therefore give run-specific txn_ids and signatures.
- **Profiling note (Fig 8).** In a heart `chain_fedavg_lr` run, sklearn's `log_loss` input validation behind the
  per-hospital logs costs about a third of the run time. A plain numpy log-loss would be much faster, but its last
  bits could differ from the sweeps already on disk, so it is left as is. Commitment itself costs about a tenth.

## FedVerify — Revision 2, deferred out of Phase 4
- **Canonical leaf order.** Round leaves follow the arrival order of the accepted submissions, so reordering them in
  transit gives a different (internally consistent) root for the same updates. Sorting accepted leaves into the
  registered-hospital order would make the root a function of the content alone; with honest runs (hospital order)
  nothing on disk would change. Listed as not covered in THREAT_MODEL.md.
- **The payload header's run_id is not checked against the run** (round and hospital are). Checking it would change
  the Phase 3 test fixtures, which build payloads with a placeholder run id. Only a hospital re-using its own update
  from another run gets through, and screening judges that update's content.
- **Adaptive attackers are not evaluated.** The probe set is public and fixed. A secret per-round probe drawn from a
  server seed (published after the round) would make probe-aware trees harder to craft. Logistic updates shaped to
  stay under tau are not tried either.
- **More than one Byzantine hospital (collusion)** is out of scope at K = 4 or 5 (median / MAD of 4-5 numbers).
- **One tau per (dataset, family, epsilon, K)** pools every attack of the family. Per-attack thresholds are not
  built, and an attack of a new kind may score below the pooled tau.
- **Only ours_* can be attacked.** "Undefended" means ours_* with `--no-screening`, i.e. the same DP and commit
  pipeline. Attacking `fedavg_lr`, `fed_rf` or `fedxgb_bagging` directly (no DP, no commit) is not wired up.
- **Chain attacks not simulated:** forks and reorganisations of the PoW chain (51%), equivocation across several
  aggregators or ledgers, dropped or delayed submissions (availability), stolen private keys, key revocation and
  rotation, and multi-factor authentication or firewalls (the paper's endpoint defences). See THREAT_MODEL.md.
- **TEST keys** are derived from the run seed. A deployment needs real key generation and a registration ceremony,
  because registration trusts the first key registered for a hospital.

## FedVerify — Revision 2, deferred out of Phase 5
- ~~**The .tex tables were not compiled.**~~ Done in the Phase 5 checks with a TinyTeX (TeX Live) kept in the session
  scratch folder, not installed system-wide: all 18 tables in one IEEEtran journal document compile with no error,
  no overfull or underfull box and no LaTeX warning. The compile found two real bugs, both fixed and tested: an
  unmapped δ and minus sign, and rows starting with "[19]" after \\ (read as an optional length). It also found
  overflowing layouts, now fitted with \tabcolsep 3pt, small fonts for the widest tables, and ragged-right p{} columns.
  Packages used: only what IEEEtran loads, plus amssymb.
- **Optional SHAP table: not built** (`shap` is not installed; see the Phase 2b note).
- **The reference paper is internally inconsistent (for DESIGN_NOTES, Phase 6).** Its latency is in seconds in Table IV
  but in milliseconds in the abstract and text, and the text gives two different decimal expansions. Its throughput
  in the abstract and in the conclusion differ by orders of magnitude (bytes/s vs MB/s). Checked in the PDF text; the
  generated Table IV footnote states the unit difference without copying numbers.
- **Clean screened runs exclude honest hospitals more often than the calibration's 5% FPR target** (attack tables,
  "none / on" rows), most on heart. The calibration pools clean cells with attacked cells, whose honest hospitals
  look less outlying while a real outlier is present. Calibrating the FPR constraint on clean cells only is the
  obvious variant; it would change tau, so it is a decision for the human, not a silent fix. Measured on the
  calibration cells: the clean-only constraint brings clean false exclusion down to the 5% target, but roughly halves
  recall on heart and lowers it on TCGA (most for the XGBoost AFT screen). Recommendation: keep the pooled rule (the
  existing calibrate.py method) and report the clean-run false-exclusion rate as the cost of screening.
- **Screening on heart is weak** (calibration AUCs well below TCGA's): four very different hospitals, one of them
  almost all positive, make honest outliers. This is a finding to report, not a bug.
- **Chain-attack resilience is one simulation per dataset** (seed 0, 100 rounds). The defences are deterministic, so
  repeats would only redraw variants. The paper's numbers are shown only as "reported" in the resilience table.
- **Figs 5-6 show FedXGBBagging and FedAvg** (`figures.per_hospital_methods` in table.yaml); add `ours_*` there for
  per-hospital curves of the screened runs. The per-hospital heart table uses accuracy; an ROC-AUC copy is one more
  entry in table.yaml.
- **Our blocks take milliseconds** (difficulty 2 on a laptop) where the reference paper's Fig 7 shows seconds: other
  hardware and difficulty, so the two figures are not comparable.
- **Two extra runs were made for complete tables** (not in the Phase 4 list): a TCGA `chain_fedavg_lr` on the local PoW
  chain (Figs 7-8 for both datasets) and TCGA `chain_fedxgb_bagging` AFT (the commit-only row of the survival table).
  The chain-attack simulator now suffixes non-heart output folders with the dataset (`*_tcga`), so the TCGA simulation
  does not overwrite heart's.

## FedVerify — Revision 2, deferred out of Phase 6 (final)
- **Open decisions for the human / the guide.**
  - "FPoW": undefined, awaiting the guide (Phase 2b).
  - Calibration: tau keeps the pooled rule. The clean-only variant is analysed in the Phase 5 notes; switching
    means re-running `calibration` and `poisoning`.
  - `CLAUDE.md` is deleted in the working tree; restore it with `git restore CLAUDE.md` if that was not intended.
- **Results are not in git.** `fedverify/results/` is gitignored, so the tables, figures and the images that
  `paper/RESULTS_DRAFT.md` links to exist only on a machine that ran the experiments. To publish them, copy the
  outputs somewhere tracked, or rebuild with `python3 -m fedverify.analysis.make_fl_healthcare_all`.
- **No UI for the heart / TCGA work.** Frontend changes were out of scope. The existing Federated tab could show a
  chain_* fold-run pushed with `fedverify.chain.push` (the commit format is compatible, tested), but nothing was
  pushed.
- **Not built, by design or scope:** homomorphic encryption or secure aggregation, verifiable aggregation (proof
  that the published model is the aggregate of the committed updates), DP for tree models, a consortium chain
  (Hyperledger / Quorum), public anchoring in these experiments (the Algorand / Bitcoin anchor paths exist but were
  not used), the NIH chest X-ray replication, adaptive attackers, more than one malicious hospital, and the optional
  SHAP table.
- **The results draft is a draft.** Its sentences are generated by a fixed rule from the results; the discussion,
  related-work text and figure placement for the paper are left to the authors. The rule is a heuristic over 5
  repeats, not a significance test (stated in its Limitations).
- **LaTeX:** the tables were compiled with a TinyTeX kept in the session's scratch folder; nothing was installed on
  the system. Compile them in the paper's own template before submission.


## FedVerify — Revision 2, Phase 7 (audit fixes)
- **Done.** One threshold rule for accuracy and F1 (the training fold's F1-optimal threshold); the "predict all alive"
  baseline in every TCGA table and figure; no accuracy ranking within 0.01 of the majority-class rate; FedXGBBagging
  averages boosters over contributing hospitals (was: summed); recall next to the clean false-exclusion rate, and no
  "outperforms" for a screen whose clean false-exclusion rate exceeds 5%; chain-attack caveats (by construction,
  covered / not covered, PoW difficulty vs ~5 s blocks). Details: DESIGN_NOTES.md section 7.
- **Superseded results** are kept in `fedverify/results/_superseded/phase6_before_audit/` (README there). Delete them
  once the Phase 7 numbers are accepted.
- **Open.** A mild rise of held-out log-loss after the middle rounds can remain for FedXGBBagging; the number of rounds
  stays fixed at 30 and is not chosen from test data. A validation-based stopping rule (inside the training folds)
  would be the honest way to pick rounds, if wanted.
- **Open.** Screen recall stays low on heart (see the generated recall sentence in RESULTS_DRAFT.md); a stronger
  screen, or more attacker signal per round, is future work.
- **Open.** The 5% clean false-exclusion limit and the 0.01 accuracy margin are reporting choices in `table.yaml`, not
  tuned on results; change them there if the guide prefers other values.

## FedVerify — Revision 2, Phase 8 (diagnosis: TCGA centralized XGBoost)
- **Done.** Diagnosis with measurements in DESIGN_NOTES.md section 8 and a generated limitation in RESULTS_DRAFT.md,
  from `results/diagnostics/central_xgb_tcga.json` (`python3 -m fedverify.survival.experiments diagnostics`).
  Cause: central_xgb over-fits. It has the same tree count as the federated model but 5x the sequential boosting
  steps since the Phase 7 averaging, and it over-fits faster per step than the averaged federated boosters.
  Site death rates explain local's lead on cross-site pairs, not central's drop.
- **Not a configuration bug, so nothing was changed or re-run.** The equal-tree budget is the documented design.
- **Open decision.** Add a step-matched centralized XGBoost (N trees per round, the federated step count) as an extra,
  clearly labelled baseline row? Its budget rule would be fixed now, before its result is read. It would not replace
  central_xgb.
- **Open.** Local XGBoost's TCGA ROC-AUC is no better than a predictor that knows only the site's death rate, so the
  "local beats centralized" result on TCGA should not be read as local models learning more from the features.

## FedVerify — Revision 2, Phase 9 (reference rows and the step-matched diagnostic row)
- **Done.** "Site base rate only (reference)" row in every method table, computed by the table code from the runs'
  out-of-fold files with training-fold site rates only. "Centralized, step-matched (diagnostic)" row
  (`central_xgb_steps`, rule fixed in table.yaml before it ran); the original Centralized row is unchanged. The draft
  and the page state that centralized XGBoost over-fits and what the site-rate ROC-AUC means, and compare every method
  table result with the site-rate row by the rule. Details: DESIGN_NOTES.md section 9.
- **Finding to read before writing the paper.** On TCGA no method exceeds the site-rate reference on ROC-AUC or F1, so
  those pooled TCGA numbers mostly reflect site death rates. The C-index is the more informative TCGA metric.
- **Open decision.** On heart, the step-matched centralized model (a blind result) outperforms both the original
  centralized model and federated XGBoost. Whether the paper's centralized baseline should be the equal-tree or the
  equal-step one is a design question for you and the guide. Today the study keeps equal trees and shows the other as
  a labelled diagnostic.

## FedVerify — Revision 2, Phase 10 (live demo, dashboard, presenter script, diagrams)
- **Done.** `python3 -m fedverify.survival.demo` (3 narrated rounds on heart, about 3 s, no network); Survival runs
  push to the Federated tab (`fedverify.chain.push`, additive: MNIST runs push exactly as before, tested);
  per-round ground truth in the tab only for runs that set `perRoundTruth`; `fedverify/survival/DEMO.md` generated by
  `make_demo_md` (in `make_fl_healthcare_all`); six Mermaid diagrams in `docs/diagrams/`, names checked by a test.
- **Demo caveat (by design, stated in the demo and DEMO.md).** The screen catches the round-2 attacker for seed 0
  (the study's attacker for that seed); seeds 1 and 2 show honest misses, matching the study's low recall.
- **Open.** `--chain-backend algorand` is tested only against a stubbed Algorand client (no live broadcast was made:
  it spends the wallet's test ALGO); try it once before relying on it in a presentation.
- **Open.** The diagrams are `.mmd` sources only: mermaid-cli was not installed. Render command in
  `docs/diagrams/README.md`. *(Done in Phase 12: rendered to SVG and 300-dpi PNG, see below.)*
- **Open.** `push` publishes one Survival fold-run per call (repeat0/fold0 by default); pushing every fold-run of
  every cell would flood the selector, so pick the ones to show.

## FedVerify — Revision 2, Phase 11 (statistics, within-hospital metrics, LaTeX check, freeze)
- **Done.** Within-hospital ROC-AUC / C-index columns; paired differences with 95% patient-level bootstrap intervals;
  verdict rule switched to "outperforms only if the interval excludes 0" (VERDICT_CHANGES.md lists all 56 changes);
  every .tex table compiles in an IEEE two-column document (27 of 27, no overfull box; two tables split in LaTeX
  only); report pack; MANIFEST.json with a mined Merkle root. Details: DESIGN_NOTES.md section 10.
- **Finding, not changed.** The Site base rate only row scores just under 0.5 within hospital (its training-fold
  rate varies between folds); within one fold it is exactly 0.5. Decide whether the paper should say so in a footnote.
- **Decision for the guide.** Within-hospital metrics leave out hospitals with fewer than 5 of the rarer class (the
  per-hospital tables' rule), which also drops heart's Switzerland (1 negative), not only TCGA's A8 (4 deaths).
- **Open.** The bootstrap interval holds the trained runs fixed; a hierarchical bootstrap over repeats and patients
  would also cover training randomness (it matters most for the poisoning comparisons, where the attacker changes
  between repeats).
- **Open.** The MNIST tables 2, 4, 5 and 6 cannot be regenerated: their results (exp2, exp4, exp5) are no longer on
  disk. Their LaTeX was re-written from their Markdown; the numbers are unchanged. *(Corrected in Phase 12: exp2,
  exp4 and exp5 were never run, so those tables have no numbers at all; table 5 is not a run. See below.)*
- **For the human.** Delete `fedverify/results/_superseded/` only after the guide approves the numbers, then re-run
  `make_manifest` (the manifest covers _superseded/ while it exists).

## FedVerify — Revision 2, Phase 12 (final wrap-up)
- **Done.** Wording rules applied in the generators (tables, results draft, showcase page), each with a test
  (`fedverify/tests/test_phase12_wording.py`): a gain of Ours whose screen excludes more than
  `clean_false_exclusion_max` of honest hospital-rounds on a clean run is "not ranked" in the paired tables and the
  draft's paired paragraph too (heart random forest); "highest score (not distinguishable from X)" instead of "best";
  the 20 key comparisons within hospital as a new table (`paired_comparisons_within`), and within-hospital claims only
  through it; every federated gain over centralized training on TCGA says in the same sentence what the Phase 8
  diagnosis found and how federated compares with the step-matched row. The six diagrams are rendered (SVG and
  300-dpi PNG) by `python3 -m fedverify.analysis.render_diagrams` and copied into the report pack. Details:
  DESIGN_NOTES.md section 11.
- **MNIST tables 2, 2b, 2_full, 4 and 6 were never run (corrects Phase 11).** exp2, exp4 and exp5 have no results in
  this repository, so every cell is empty; the captions now say "Not run" (a caption saying "raw results from an earlier
  run, not retained" would have been false). Not run in Phase 12. The commands that would fill them
  (`results/calibration/taus.json` is present; run `python3 -m fedverify.analysis.calibrate` first only if it is
  missing; exp4 also needs the MIT-BIH data):

  ```bash
  python3 -m fedverify.experiments.exp2_byzantine --jobs 8 --threads 2                  # tables 2, 2b, 2_full
  python3 -m fedverify.experiments.exp4_heterogeneity --jobs 6 --threads 2              # table 4
  python3 -m fedverify.experiments.exp5_dp_byzantine_interaction --jobs 6 --threads 2   # table 6
  python3 -m fedverify.analysis.make_tables --only table2 table2b table4 table6         # rebuild the tables
  ```

  Table 5 is not a run: it is the qualitative table built from `analysis/comparison.yaml`, and it regenerates
  unchanged (`python3 -m fedverify.analysis.make_tables --only table5`).
- **Open decision.** TCGA random forest: federated outperforms centralized (pooled and within hospital), but nothing
  explains it yet: Phase 8 diagnosed XGBoost only and there is no step-matched random forest. The text says "not
  diagnosed". A diagnosis would need re-runs.
- **Open decision.** Within hospital, many key verdicts differ from the pooled ones (the "Pooled verdict" column of
  `paired_comparisons_within`). Which one the paper leads with is for you and the guide.
- **Finding.** On TCGA F1, federated XGBoost still outperforms the step-matched centralized row (the page says so); on
  the C-index and ROC-AUC it does not.
- **Open.** `architecture.mmd` is a very wide left-to-right diagram: at the text width of a portrait page its labels
  are tiny. Place it landscape, use the SVG, or re-lay it out top-down.
- **For the human.** `COMMIT_PLAN.md` (repository root) groups every new or changed file into commits on three
  branches and lists what must not be committed.
- **Final gate.** Rebuilt with no missing input; 28 of 28 LaTeX tables compile; results re-frozen (MANIFEST.json,
  15,403 files, Merkle root `76b01643fb4fd2644568e8506e95d694c7021486361d87788582334a0d35abbd`); 532 tests pass;
  `make_manifest --verify` intact after the tests; showcase page republished. No further phases.
