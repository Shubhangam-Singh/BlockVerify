# FedVerify-Survival — Phase 0 Inventory

Ground truth from reading the real code, loading the real workbook and running the real test suite. Per
master-prompt §0.3 the code wins where the prompt and the code differ; per §0.1 `CLAUDE.md` wins where it and the
prompt differ. Each difference is recorded below; the ones that need your answer are collected in section 6.

How this file was produced
- Section 4 (data) is the **verbatim output** of `python3 -I fedverify/survival/phase0_check.py`; section 5 (tests and
  packages) comes from real runs.
- Sections 1b, 2 and 3 come from reading the code directly. (A planned two-agent audit stopped on a usage limit before
  returning anything, so none of its output is used here.) **Every `file:line` and its quoted source line was then
  checked mechanically against the files**: 85 of 85 claims matched as written, 0 needed a
  line correction, 0 were dropped. Behaviour claims ("raises", "works with string ids", "returns the K=10 tau")
  come from the read-only smoke checks in section 7, which left `git status` and `backend/data/` unchanged.
- `file:line` refer to the working tree at HEAD `8932448` ("few updates on fl") plus the uncommitted edits listed in
  section 8. Line numbers drift as code changes; the appendix keeps the quoted line to search for.
- Nothing existing was modified. Phase 0 added `fedverify/survival/{INVENTORY.md, phase0_check.py, requirements.txt}`,
  appended one section to `fedverify/TODO.md`, and put a gitignored copy of the workbook at the spec path.

---

## 0. Read this first

1. **The data matches the prompt.** The workbook is 907 x 34; the five hospitals have exactly the patients
   and deaths of §3 (474 patients, 76 deaths); 12 sites have 30+ patients and 8 of them have fewer
   than 10 deaths; in the five hospitals 0 rows have to be dropped for missing time. Details: section 4.
2. **One data check FAILS: the unit sanity test.** 11 of 720 rows that have `OS_MONTHS` break
   `|time_days/30.4375 - OS_MONTHS| < 0.5` (7 of 407 in the five hospitals). The median error is
   0.02 months, so the unit conversion is right; the failures are rows where the two clinical sources disagree.
   Written literally, the Phase-1 test would fail. Decision D2.
3. **Existing suite: 259 passed** (prompt: "about 258"). The prompt's exact command works.
4. **Every existing FL entry point wants integer client ids; hospitals are strings** (`BH`, `A2`, ...). `Client`, all
   six aggregators, `plan_privacy`, `update_digest` and `commit_round` raise `invalid literal for int()` on `'BH'`;
   the Merkle root, proofs, anchor and `/api/fl/*` routes are fine with strings. 8 adapter rows are blockers,
   and all of them are solved by one additive id mapping (section 3, D10).
5. **Two traps found by running the code.** `load_tau(...)` asked for a K=5 tau silently returns the MNIST
   `eps=1|K=10` value, so Survival must never call it (A14). And calibration `cells()` is hard-wired to MNIST, so the
   K=5 calibration the prompt asks for needs its own small driver (A13).
6. **Reuse is real but partial**: 35 symbols import as is, 8 with a thin adapter, 13 are
   patterns to follow (including the client training loop), and 3 must not be reused (the MNIST/ECG-shaped run
   loop, data layer and `evaluate`). Survival gets its own runner that imports the reusable pieces.
7. **Dependencies.** The prompt names 3 packages; the build needs 5 (scikit-learn for the RF family, pandas for the
   loader). Installing `lifelines` downgrades pandas 3.x to 2.x. I verified it in a scratch directory (real environment
   untouched): lifelines imports cleanly, `-risk` is the right sign for its C-index, `phase0_check.py` prints identical
   output under pandas 2, and the existing suite still passes there (259 passed). Only `openpyxl` (+ `et_xmlfile`)
   was installed. Decision D8.
8. **`CLAUDE.md` is deleted in the working tree** (unstaged). I followed the committed version (amendments A1-A11). D1.
9. **12 decisions are listed in section 6, each with a recommended default.** Replying `go phase 1` without comment
   means "apply the defaults"; the three actions only you can take are also in the completion block.

## 1. Prompt vs reality

### 1a. Files, data and environment

| The prompt says | Reality | Consequence |
|---|---|---|
| `CLAUDE.md` at the repo root, "including every amendment (A1, A5, ...)" | Deleted in the working tree (` D CLAUDE.md`, not staged). `HEAD` has it, with amendments **A1-A11** | Rules read with `git show HEAD:CLAUDE.md`. I may not run `git checkout`, so nothing was restored. D1 |
| "about 258 tests" | **259 passed**, 18 warnings, 38.03 s (99 collected in `backend/`, 160 in `fedverify/tests/`) | Baseline for "existing suite unchanged" is 259 |
| `pytest backend/ fedverify/tests/ -q` | Works as written (`pytest` is `~/.local/bin/pytest`) and as `python3 -m pytest ...` (CLAUDE.md A3's form); both collect the same tests. No pytest.ini / conftest.py exists | Survival tests must live in `fedverify/tests/` to be collected. D4 |
| Data at `fedverify/data/tcga_brca/clinical.xlsx`; "If missing, stop and tell me" | It was missing. The workbook was in the repo root as `clinical_project_tcga_brca_2024-04-08_converted.xlsx` (untracked) and in `~/Downloads` (same sha256). Following your note ("excel file in the project section ...") I **copied** it to the spec path, which `.gitignore:25` (`fedverify/data/`) already ignores | The root copy is still untracked and not ignored. D9 |
| `days_to_last_follow_up` is "in days"; drop rows with no usable time | Missing is spelled `[]` (not blank) in 87 rows, all deaths whose time comes from `survival`; `survival` is blank in 787 rows | The loader must treat `[]` as missing. In the five hospitals no row is dropped |
| `OS_MONTHS` sanity check `< 0.5` for every row that has it | 11 rows violate it (section 4) | D2 |
| §4 "Repeats: seeds 0-4" and Phase 1 CLI `--seed N --repeats 5` | Two readings (repeat r uses seed r, or seed N + r) | D7 |
| Ground rule 4: "New code goes in `fedverify/survival/`" | Phase 3 names `chain/contentid.py` and `chain/signing.py`; Phases 5-6 name `analysis/make_survival_tables.py` and `analysis/make_results_draft.py` (existing packages) | D5 |
| Dependencies: `lifelines`, `openpyxl`, `cryptography` | Also needed: `scikit-learn` (RF family, `StratifiedKFold`) and `pandas` (loader). `cryptography` 41.0.7 is installed and Ed25519 from a 32-byte seed signs and verifies; `lifelines` and `scikit-learn` are not installed | `requirements.txt` lists all five. D8 |
| §4 fixed defaults: "FedAvg local_epochs 3, batch 32, lr 0.1" | No momentum is given; the existing client uses SGD with `cfg.momentum` (FLConfig default 0.9) | D12 |
| RF family: "the server concatenates all trees" | `CLAUDE.md`: "Everything downstream of local training operates on DELTAS ... never on raw weights" | D6 |
| §6 add `model_family` to `rounds.jsonl` | `CLAUDE.md` A9: a clean no-attack record must keep its exact shape; `model_family` appears nowhere today (no file references it) | Added to Survival records only. D11 |
| Table IV of the IEEE T-SUSC 2024 paper (Myrzashova) goes to `fedverify/survival/ref/` later | The PDF attached to the request is arXiv 2408.06183v2 (Padilla Rodriguez & Nafea, a UCI heart-disease benchmark with SHAP), a different paper | Nothing about Table IV is assumed; `table.yaml` stays configurable. `ref/` is yours to create |

### 1b. The prompt's claims about the code, checked against the code

Confirmed as written (details in section 2): `canon_update` exists under that name; `chain/commitment.py` has the digest, `round_merkle_root` and inclusion proofs; `chain/anchor.py` has mock / local / algorand plus the Bitcoin checkpoint; the forensics aggregator, the privacy accountant with its total-step spy test, the `sign_flip` / `scaling` attacks and the A9 record-shape guard all exist. These do not match:

| id | severity | plan vs code | where | do this instead |
|---|---|---|---|---|
| M02 | adapter-needed | **'Same rounds.jsonl keys' cannot include the classification metrics.** PLAN: emit the same keys as existing runs plus model_family. / CODE: records carry test_acc, test_loss, macro_f1 (classification); model_family exists nowhere in the repo. | `fedverify/core/runner.py:221` | Keep diag / attack / commit / privacy / timing keys identical; put c_index, auc, f1 where test_acc sits; add model_family to Survival records only (A9). |
| M04 | adapter-needed | **The runner is reusable only in parts.** PLAN: 'Reuse the existing FL runner/aggregator interface'. / CODE: the aggregator interface and the record structure are reusable, the run loop is dataset-specific (R32). | `fedverify/core/runner.py:87` | Survival runner = new loop + imported helpers. |
| M07 | adapter-needed | **Calibration key has no model family and the file name is fixed.** PLAN: taus_survival.json keyed by (family, epsilon, K). / CODE: taus.json keyed eps=..\|K=.., written and read at fixed paths. | `fedverify/analysis/calibrate.py:201` | New Survival calibration driver + loader (A13, A14); the existing taus.json is never touched. |
| M01 | cosmetic | **Tau check happens at aggregate(), not at construction.** PLAN: 'Constructing a screening aggregator without a tau must raise.' / CODE: build_aggregator() takes only a name and Forensics() constructs fine; the ValueError comes on the first aggregate() (smoke). | `fedverify/aggregators/forensics.py:122` | Reuse Forensics unchanged (it still refuses before aggregating anything). The new RF screen raises in __init__, and the Survival runner checks that tau is present before round 1. |
| M03 | cosmetic | **label_flip already exists.** PLAN: lists label_flip beside the reused sign_flip / scaling as if new. / CODE: it is an existing DATA attack and flips binary labels correctly with num_classes = 2. | `fedverify/attacks/byzantine.py:26` | Reuse poison_batch (R22). |
| M05 | cosmetic | **Results root is fedverify/results, not results/.** PLAN: results/calibration/taus_survival.json, results/tables/*.md. / CODE: every script defaults to fedverify/results (gitignored by .gitignore:24). | `fedverify/config.py:35` | Survival writes fedverify/results/survival/..., fedverify/results/calibration/taus_survival.json and fedverify/results/tables/. |
| M06 | cosmetic | **Per-client robust z lives in forensics, not eval_lib.** PLAN: 'Reuse the eval_lib probes for robust-z/health checks'. / CODE: eval_lib supplies the median/MAD probe and the health flags; the per-client robust z is forensics.robust_z, which wraps them. | `fedverify/aggregators/forensics.py:63` | Import robust_z from fedverify.aggregators.forensics and bv_layer_health from eval_lib. |

**Named by the prompt but not in the repo yet** (all are new Phase 1-6 files, as expected): `fedverify/survival/data.py`, `fedverify/survival/config.py`, `fedverify/survival/table.yaml`, `fedverify/survival/metrics.yaml`, `fedverify/survival/serialize.py`, `fedverify/chain/contentid.py`, `fedverify/chain/signing.py`, `fedverify/analysis/make_survival_tables.py`, `fedverify/analysis/make_results_draft.py`, `fedverify/survival/ref`.

## 2. Reuse map (file:line)

Reuse levels: `import-as-is` = import and call unchanged; `import-with-adapter` = reusable once the thin additive id adapter of section 3 is in place; `pattern-only` = follow the approach, do not call the code; `do-not-reuse` = would be wrong for Survival.

### 2a. Core FL: runner helpers, client, models, privacy, aggregators, attacks, config, tests

| id | symbol | where | reuse level | what it is / how Survival uses it |
|---|---|---|---|---|
| R01 | `fedverify.core.runner.strip_timings` | `fedverify/core/runner.py:44` | import-as-is | Recursively drops keys in TIMING_KEYS or ending in _ms/_wall_s (CLAUDE.md A1). **Use:** Survival determinism tests compare rounds.jsonl through it. The new Phase-3 keys (serialize_ms, digest_ms, cid_ms, sign_ms, verify_ms, merkle_ms, anchor_ms) all end in _ms, so they are excluded with no change. |
| R02 | `fedverify.core.runner.seed_everything` | `fedverify/core/runner.py:53` | import-as-is | Seeds random, numpy, torch (cpu+cuda) and enables deterministic algorithms where supported. **Use:** Called once per repeat seed by the Survival runner. |
| R03 | `config.json writer (pattern)` | `fedverify/core/runner.py:135` | pattern-only | config.json = {config: FLConfig.to_dict(), partition_report, attacker_ids, num_params, torch_threads_effective, ...}. **Use:** Survival writes the same top-level keys (config, attacker_ids, num_params, torch_threads_effective) plus its own (hospital_ids, family, method, fold/repeat info), so chain/push.py's run metadata keeps working. |
| R04 | `round record (pattern)` | `fedverify/core/runner.py:219` | pattern-only | Per-round keys: round, test_acc, test_loss, macro_f1, mean_train_loss, per_client_num_samples, diag {accepted, rejected, scores}, privacy, train_wall_s, round_wall_s; plus an attack block when an attack is configured or the aggregator detects, and a commit block when commit is on. **Use:** Survival keeps diag / attack / commit / privacy / per_client_num_samples / timing keys with identical structure and adds model_family (see mismatch M02 for the metric keys). |
| R05 | `commit block of a round record (pattern)` | `fedverify/core/runner.py:200` | pattern-only | root, leaf_count, txid, backend, bytes_committed, fee, bytes_written, digest_ms, merkle_ms, anchor_ms, checkpoint_ms (+ aggregate_ms): the 'anchor timing, round root' keys the prompt asks to keep. **Use:** Survival's commit block uses the same keys and adds serialize_ms, cid_ms, sign_ms, verify_ms. |
| R06 | `fedverify.core.runner._attack_report` | `fedverify/core/runner.py:265` | import-with-adapter | Per-round TP/FP/FN, precision, recall, F1 and false_exclusion_rate over the participating clients. **Use:** Gives Phase 4's detection P/R and honest-hospital false-exclusion rate; needs integer ids (A11 below). |
| R07 | `fedverify.core.client.ClientUpdate` | `fedverify/core/client.py:21` | import-with-adapter | Container every aggregator and commit_round consume: client_id (int), delta (1-D tensor), num_samples, train_loss, wall_time_s, privacy, meta. **Use:** The Survival logistic client returns ClientUpdate objects carrying the hospital's integer index. |
| R08 | `fedverify.core.models.get_flat_params` | `fedverify/core/models.py:88` | import-as-is | All trainable parameters as one detached 1-D tensor; generic over any nn.Module. **Use:** Logistic model (nn.Linear(30, 1)) flat params = 31 values; deltas are local minus global flat vectors. |
| R09 | `fedverify.core.models.set_flat_params` | `fedverify/core/models.py:93` | import-as-is | Exact inverse of get_flat_params; checks the element count. **Use:** Load the global vector into each hospital's model. |
| R10 | `fedverify.core.models.assert_no_batchnorm` | `fedverify/core/models.py:82` | import-as-is | Raises if any BatchNorm module exists. **Use:** Called on the logistic model at start-up (rule 7). |
| R11 | `fedverify.core.privacy.plan_privacy` | `fedverify/core/privacy.py:94` | import-with-adapter | Solves sigma ONCE over the client's TOTAL steps (rounds x local_epochs x ceil(n_k/batch)), sample_rate = batch/n_k, and attaches one persistent RDPAccountant. Reads cfg.epsilon, delta, max_grad_norm, batch_size, local_epochs, rounds. **Use:** Called per hospital with its integer index and its training-fold size; any object with those attributes works as cfg. |
| R12 | `fedverify.core.privacy.make_private` | `fedverify/core/privacy.py:121` | import-as-is | Wraps model/optimizer/DataLoader with Opacus using the fixed sigma and the persistent accountant; Poisson sampling. **Use:** The Survival logistic client wraps a torch DataLoader over its hospital's rows; the training loop must skip empty Poisson batches exactly like client.py does. |
| R13 | `fedverify.core.privacy.privacy_report` | `fedverify/core/privacy.py:146` | import-as-is | sigma, C, sample_rate, steps, steps_per_round, steps_taken, realized_eps, target_eps, delta, n_samples. **Use:** Stored per hospital in each record: this IS the 'per-hospital sigma, per-hospital realised epsilon logged' requirement. |
| R14 | `fedverify.core.privacy.dp_enabled` | `fedverify/core/privacy.py:41` | import-as-is | epsilon None or inf means DP off and Opacus is never touched. **Use:** If the Survival client takes the same branch, epsilon = inf is the Phase-1 fedavg_lr code path by construction, which is what the Phase-2 'must match exactly' test checks. |
| R15 | `fedverify.aggregators.build_aggregator` | `fedverify/aggregators/__init__.py:19` | import-with-adapter | Registry: fedavg, krum, multikrum, trimmed_mean, median, forensics. **Use:** fedavg_lr / dp_fedavg_lr use fedavg; ours_lr uses forensics; all need integer client ids (A02-A07). |
| R16 | `fedverify.aggregators.base.Aggregator` | `fedverify/aggregators/base.py:15` | import-as-is | aggregate(updates, cfg, round_num) -> (delta, diag); diag always has accepted / rejected / scores; class attribute detects (A11). **Use:** The new RF probe-set screen subclasses it with detects = True and raises in __init__ without tau. |
| R17 | `fedverify.aggregators.forensics.Forensics` | `fedverify/aggregators/forensics.py:114` | import-with-adapter | Robust z across clients of norm / direction / coordinate-outlier share, combined = max, hard health flags; tau from cfg. Smoke at K=5 with 31-value deltas: 5 identical updates reject nobody; an all-NaN update is rejected (health:nan). **Use:** Used unchanged for ours_lr, with integer ids and tau from taus_survival.json. |
| R18 | `fedverify.aggregators.forensics.robust_z` | `fedverify/aggregators/forensics.py:63` | import-as-is | Per-element robust z using eval_lib.bv_find_outliers' median and 1.4826*MAD scale (own median/MAD when n < 4). **Use:** The RF probe-set screen scores each hospital as robust_z(distance to the median prediction, 'high'). |
| R19 | `fedverify.attacks.byzantine.attacker_ids` | `fedverify/attacks/byzantine.py:30` | import-as-is | Deterministic attacker set from (seed, K, attacker_frac). K=5, frac=0.2 gives exactly one attacker. **Use:** Phase 4 picks the malicious hospital with it (index -> hospital code via the D10 mapping). |
| R20 | `fedverify.attacks.byzantine.attack_active` | `fedverify/attacks/byzantine.py:42` | import-as-is | Dormant before attack_start_round (default 0, so active from the first round). **Use:** Use as is. |
| R21 | `fedverify.attacks.byzantine.apply_delta_attack` | `fedverify/attacks/byzantine.py:57` | import-as-is | sign_flip: -s*delta; scaling: (K/f)*delta (5x at K=5, f=1); also gaussian and zero. **Use:** The prompt's logistic sign_flip and scaling attacks, unchanged. |
| R22 | `fedverify.attacks.byzantine.poison_batch` | `fedverify/attacks/byzantine.py:85` | import-with-adapter | label_flip maps y -> (C-1) - y; with num_classes = 2 it flips events (smoke: [0,1,1] -> [1,0,0]). **Use:** Survival's label_flip calls it with num_classes=2 on the attacker's training batches (the existing Client wires it through dataset_spec, which Survival does not use). |
| R23 | `fedverify.config.FLConfig (pattern)` | `fedverify/config.py:20` | pattern-only | Frozen dataclass; to_dict() writes inf as 'inf'; run_dir = out_dir/exp/cell/seed<N>; has tau and tau_source fields. **Use:** Survival's config is its own frozen dataclass built the same way (FLConfig(dataset='tcga_brca') constructs, but its auto cell name claims a Dirichlet split, see A16). |
| R24 | `FLConfig DP defaults` | `fedverify/config.py:44` | import-as-is | delta 1e-5 and max_grad_norm 1.0 are already the defaults, matching Phase 2's 'clip C = 1.0, delta = 1e-5'. **Use:** Survival config uses the same values. |
| R25 | `privacy spy test (pattern)` | `fedverify/tests/test_privacy.py:66` | pattern-only | Monkeypatches opacus get_noise_multiplier and asserts it is asked for the TOTAL step count. **Use:** Phase 2 copies it for hospital sizes ('same spy-test as before'). |
| R26 | `eps=inf identity test (pattern)` | `fedverify/tests/test_privacy.py:45` | pattern-only | Asserts epsilon = inf reproduces the non-DP run and never calls make_private. **Use:** Phase 2's 'epsilon = inf must match fedavg_lr exactly' test. |
| R27 | `record-shape pin test (pattern)` | `fedverify/tests/test_aggregators.py:223` | pattern-only | Pins the exact record shape of a clean fedavg run (A9). **Use:** Survival changes no existing record, so this keeps passing; model_family appears only in Survival records. |
| R28 | `experiment supervisor (pattern)` | `fedverify/experiments/exp2_byzantine.py:89` | pattern-only | One subprocess per cell, --jobs / --threads, is_done() resume, tau read via load_tau and recorded with tau_source. **Use:** The human's full Survival sweeps follow the same shape (Phase 5 README commands). |
| R29 | `fedverify.core.server.evaluate` | `fedverify/core/server.py:22` | do-not-reuse | Classification metrics (argmax accuracy, macro-F1, cross-entropy). **Use:** Not usable: Survival reports C-index / ROC-AUC / F1 on pooled out-of-fold predictions (survival/metrics.py). |
| R30 | `fedverify.core.data.dataset_spec / load_dataset` | `fedverify/core/data.py:28` | do-not-reuse | Registry of mnist / fmnist / mitbih; raises for any other name. load_dataset reads torchvision or MIT-BIH only. **Use:** Not usable: the TCGA loader is new (survival/data.py). |
| R31 | `fedverify.core.client.Client.local_train` | `fedverify/core/client.py:59` | pattern-only | Local SGD returning a DELTA; DP branch via make_private; skips empty Poisson batches. Loss is hard-coded CrossEntropyLoss and the model comes from dataset_spec. **Use:** Survival's logistic client mirrors this loop (same DP branch, same empty-batch skip, same delta) with a class-weighted logistic loss; it cannot call it. |
| R32 | `fedverify.core.runner.run` | `fedverify/core/runner.py:76` | do-not-reuse | MNIST / MIT-BIH loop: load_dataset, Dirichlet or patient partition, build_model, evaluate. MIT-BIH was added by EDITING data.py, runner.py, models.py and config.py. **Use:** Additive-only rules that pattern out, so Survival has its own runner that imports R01-R28 and the chain pieces. |

### 2b. Chain, backend and analysis: commitment, anchor, push, routes, Merkle, eval_lib, calibrate, tables, paper

| id | symbol | where | reuse level | what it is / how Survival uses it |
|---|---|---|---|---|
| R40 | `fedverify.chain.commitment.canon_update` | `fedverify/chain/commitment.py:57` | import-as-is | Float32 little-endian C-contiguous bytes of a delta (the prompt's 'reuse canon_update' exists under that name). **Use:** Logistic serialization in Phase 3. Not for RF trees (A10). |
| R41 | `fedverify.chain.commitment.update_digest` | `fedverify/chain/commitment.py:69` | import-with-adapter | SHA-256 over a compact-JSON header {client_id, num_samples, numel, round} plus the canonical bytes. **Use:** Logistic digests: call it with the hospital's integer index, or mirror its header with the hospital code in survival/serialize.py; chosen in Phase 3 (A09). |
| R42 | `fedverify.chain.commitment.round_merkle_root` | `fedverify/chain/commitment.py:100` | import-as-is | (root, leaf_count) over ordered [(id, digest)]; leaf = SHA-256(JSON([i, str(id), digest])) through backend/app.py's canonical _manifest_leaf. Smoke with hospital codes: works; swapping two hospitals changes the root. **Use:** Phase 3 builds leaves [i, hospital_id, digest] with it, unchanged. |
| R43 | `fedverify.chain.commitment.inclusion_proof` | `fedverify/chain/commitment.py:111` | import-as-is | Sibling path [{hash, right}] for one leaf. **Use:** Stored per hospital per round. |
| R44 | `fedverify.chain.commitment.verify_proof` | `fedverify/chain/commitment.py:119` | import-as-is | Verifier-side fold. Smoke with hospital codes: a real proof verifies, a forged root fails. **Use:** Phase 3 test 'an existing-style inclusion proof verifies and a forged root fails'. |
| R45 | `fedverify.chain.commitment.leaf_hashes` | `fedverify/chain/commitment.py:94` | import-as-is | Ordered leaf hashes. **Use:** Stored next to the proofs. |
| R46 | `fedverify.chain.commitment.warm_up` | `fedverify/chain/commitment.py:46` | import-as-is | Imports backend/app.py once (about 2 s) so it is not charged to the first round's merkle_ms. **Use:** Call before the first timed round (Phase 3 timing logs). |
| R47 | `fedverify.chain.commitment.commit_round` | `fedverify/chain/commitment.py:134` | pattern-only | Digest every ClientUpdate, build root/leaves/proofs, time digest_ms and merkle_ms. **Use:** Usable for the logistic family with integer ids; the RF family and the signed / CID path need their own version built from R40-R46. |
| R48 | `backend/app.py _manifest_leaf (canonical Merkle)` | `backend/app.py:1562` | import-as-is | The one Merkle construction the browser re-verifies (with compute_layer_merkle and merkle_inclusion_proof right below). **Use:** Reached only through fedverify.chain.commitment; never called directly. |
| R49 | `fedverify.chain.anchor.ChainAnchor` | `fedverify/chain/anchor.py:53` | import-as-is | Backends mock / local / algorand (BACKENDS), Bitcoin as a periodic checkpoint. Mock txids are content hashes, so mock runs stay byte-reproducible; local mines a block on an in-memory PoW chain. **Use:** Phase 3 anchors each round root (mock by default, local in tests). |
| R50 | `ChainAnchor.anchor_round` | `fedverify/chain/anchor.py:79` | import-as-is | Returns {round, root, leaf_count, backend, txid, fee, bytes_written, latency_ms}. Smoke: works with a Survival run id. **Use:** anchor_ms = latency_ms, as in the existing commit block. |
| R51 | `ChainAnchor.checkpoint` | `fedverify/chain/anchor.py:142` | import-as-is | Bitcoin OP_RETURN over the Merkle root of all round roots so far (mock unless bitcoin=True). **Use:** Unchanged. |
| R52 | `backend.algorand_client.broadcast_fl_round` | `backend/algorand_client.py:398` | import-as-is | Algorand note keys frr / flc / rnd / run; App ID per CLAUDE.md A7. **Use:** Reached only through ChainAnchor('algorand'); real runs only on request (§9). |
| R53 | `backend.bitcoin_client.anchor_merkle_root` | `backend/bitcoin_client.py:84` | import-as-is | OP_RETURN anchoring of one root. **Use:** Reached only through ChainAnchor.checkpoint(bitcoin=True). |
| R54 | `fedverify.chain.push.push / load_run` | `fedverify/chain/push.py:105` | import-as-is | Publishes commitments.jsonl (+ diag from rounds.jsonl, meta from config.json) to POST /api/fl/round/commit. **Use:** Works for Survival runs if they write commitments.jsonl, rounds.jsonl and config.json in the existing shapes. |
| R55 | `POST /api/fl/round/commit` | `backend/fl_routes.py:123` | import-as-is | Re-derives the root from the submitted leaves with round_merkle_root and rejects mismatches; client ids may be strings or integers (stored as strings). **Use:** Unchanged; the existing /api/fl/* tests must keep passing. |
| R56 | `GET /api/fl/lineage/<run_id>` | `backend/fl_routes.py:296` | import-as-is | Round roots with accepted / rejected / attackers per round from backend/data/fl_runs.json; nothing is keyed on dataset or model. **Use:** Survival runs appear in the lineage once pushed. |
| R57 | `evaluation.eval_lib.bv_find_outliers` | `evaluation/eval_lib.py:75` | import-as-is | Deployed median / 1.4826*MAD probe (n >= 4). **Use:** Used through forensics.robust_z (R18). |
| R58 | `evaluation.eval_lib.bv_layer_health` | `evaluation/eval_lib.py:92` | import-as-is | NaN / Inf / \|w\| > 100 counts, entropy (n > 16), longest constant run, for any 1-D array. **Use:** Forensics' hard flags; also the RF screen's NaN check on probe predictions. |
| R59 | `evaluation.eval_lib.roc_curve / auc / rates_at` | `evaluation/eval_lib.py:154` | import-as-is | Pure-numpy ROC; auc uses the NumPy-2 trapezoid shim (A10). **Use:** K=5 calibration ROC (Phase 4). |
| R60 | `fedverify.analysis.calibrate.curve_stats` | `fedverify/analysis/calibrate.py:87` | import-as-is | AUC, Youden tau and the tau at FPR <= 5% from labelled scores. **Use:** Survival K=5 calibration feeds it its own harvested scores. |
| R61 | `fedverify.analysis.calibrate.harvest` | `fedverify/analysis/calibrate.py:67` | pattern-only | (scores, labels) from a cell's rounds.jsonl + config.json: label 1 = attacker in an active round. **Use:** Survival's calibration driver applies the same labelling to its own records, which carry hospital codes (A12). |
| R62 | `fedverify.analysis.make_tables._emit` | `fedverify/analysis/make_tables.py:108` | import-with-adapter | Writes <name>.md and <name>.tex for one table. **Use:** S1-S6 are emitted through it (after escaping underscores, A15). |
| R63 | `make_tables MISSING / mean_std / cell` | `fedverify/analysis/make_tables.py:91` | import-as-is | MISSING is the em dash; mean_std uses the sample std; cell() formats as percent unless pct=False. **Use:** 'mean +- std over the 5 repeats' cells; C-index/AUC/F1 with pct=False. |
| R64 | `make_tables.load_runs (pattern)` | `fedverify/analysis/make_tables.py:23` | pattern-only | Walks results/<exp>, keeps the final record, skips incomplete cells loudly (A8), collects max realized epsilon. **Use:** make_survival_tables follows it (incomplete skip, max realised epsilon for S3). |
| R65 | `fedverify.analysis.make_paper.collect / emit (pattern)` | `fedverify/analysis/make_paper.py:54` | pattern-only | Collects every number from results files, renders MISSING_TEX for gaps, emits the paper fragments. **Use:** Pattern for make_results_draft.py (Phase 6). |
| R66 | `fedverify/chain package` | `fedverify/chain/__init__.py:1` | import-as-is | The package __init__ is a docstring only; contentid.py and signing.py do not exist yet, so adding them edits nothing. **Use:** New modules per Phase 3 (D5). |

## 3. Adapters needed (existing code that breaks or misbehaves on Survival inputs)

Every row names the smallest **additive** fix, so no existing file has to change. Rows A01-A12 are all the same integer-id problem and are solved together by the D10 mapping.

| id | severity | what breaks (and for which Survival input) | where | smallest additive adapter |
|---|---|---|---|---|
| A01 | blocker | **Client() rejects hospital codes.** Client('BH', ...) raises ValueError (smoke). | `fedverify/core/client.py:33` | Survival gives each hospital a fixed integer index (D10) for every call into existing code and translates back to hospital codes before anything is written; no existing file changes. |
| A02 | blocker | **weighted_mean (the FedAvg kernel) casts ids to int.** Every aggregator fails on string ids: fedavg, krum, multikrum, trimmed_mean, median, forensics all raise 'invalid literal for int()' (smoke); all work with 0-4. | `fedverify/aggregators/base.py:41` | Survival gives each hospital a fixed integer index (D10) for every call into existing code and translates back to hospital codes before anything is written; no existing file changes. |
| A03 | blocker | **Forensics casts ids to int.** Same failure for ours_lr screening. | `fedverify/aggregators/forensics.py:132` | Survival gives each hospital a fixed integer index (D10) for every call into existing code and translates back to hospital codes before anything is written; no existing file changes. |
| A04 | blocker | **FedAvg diag casts ids to int.** Same failure for fedavg_lr. | `fedverify/aggregators/fedavg.py:19` | Survival gives each hospital a fixed integer index (D10) for every call into existing code and translates back to hospital codes before anything is written; no existing file changes. |
| A08 | blocker | **plan_privacy casts the id to int.** plan_privacy(cfg, 'BH', n) raises (smoke). | `fedverify/core/privacy.py:113` | Survival gives each hospital a fixed integer index (D10) for every call into existing code and translates back to hospital codes before anything is written; no existing file changes. |
| A09 | blocker | **update_digest / commit_round cast the id to int.** update_digest('BH', ...) and commit_round on string-id updates raise (smoke). round_merkle_root itself takes string ids. | `fedverify/chain/commitment.py:77` | Digest with the integer index (header) and put the hospital code in the leaf entry, or give Survival its own digest over its canonical bytes; either way the leaf format [i, id, digest] is unchanged (D10). |
| A13 | blocker | **calibrate's cells are hard-coded to MNIST + the core runner.** calibrate main() cannot produce K=5 Survival calibration rounds. | `fedverify/analysis/calibrate.py:57` | New Survival calibration driver: run Survival cells with tau = inf (clean + attacked), harvest like R61, score with curve_stats (R60), write fedverify/results/calibration/taus_survival.json keyed family\|eps\|K. |
| A14 | blocker | **load_tau silently falls back to another epsilon AND another K.** Smoke: load_tau('fedverify/results', 4.0, 5) returns the MNIST eps=1\|K=10 tau with no error (there is no K=5 entry). Path and key format are fixed to taus.json / eps=..\|K=... | `fedverify/analysis/calibrate.py:255` | Never call it for Survival. A new loader reads taus_survival.json, looks up exactly family\|eps\|K, raises when missing, and returns tau_source for config.json. |
| A10 | adapter-needed | **canon_update only handles float arrays.** Raw bytes raise ValueError (smoke); int arrays are silently cast to float32, which is lossy above 2^24. | `fedverify/chain/commitment.py:65` | RF trees get their own canonical bytes (survival/serialize.py, as Phase 3 plans) and a bytes-level SHA-256; only the (id, digest) pairs go to round_merkle_root. |
| A11 | adapter-needed | **_attack_report casts ids to int.** Detection P/R needs integer ids in diag. | `fedverify/core/runner.py:276` | Survival gives each hospital a fixed integer index (D10) for every call into existing code and translates back to hospital codes before anything is written; no existing file changes. |
| A12 | adapter-needed | **calibrate.harvest casts ids to int.** Needs integer ids in diag.scores and an attack.active flag in every record; Survival records carry hospital codes. | `fedverify/analysis/calibrate.py:83` | Survival's calibration driver mirrors harvest() and labels hospital codes against config.json's attacker list itself; every record keeps the attack.active flag. |
| A18 | adapter-needed | **Dashboard ids must be one namespace.** Leaves, accepted, rejected and attackers are matched as strings. Hospital codes in the leaves with integers in diag would make the lineage view disagree with itself. | `backend/fl_routes.py:281` | Write ONE id namespace everywhere on disk and on chain (D10). |
| A05 | cosmetic | **Krum / MultiKrum cast ids to int.** Only matters if Krum baselines are run on Survival. | `fedverify/aggregators/krum.py:36` | Survival gives each hospital a fixed integer index (D10) for every call into existing code and translates back to hospital codes before anything is written; no existing file changes. |
| A06 | cosmetic | **Median diag casts ids to int.** Only matters if median is run on Survival. | `fedverify/aggregators/median.py:24` | Survival gives each hospital a fixed integer index (D10) for every call into existing code and translates back to hospital codes before anything is written; no existing file changes. |
| A07 | cosmetic | **Trimmed mean diag casts ids to int.** Only matters if trimmed mean is run on Survival. | `fedverify/aggregators/trimmed_mean.py:36` | Survival gives each hospital a fixed integer index (D10) for every call into existing code and translates back to hospital codes before anything is written; no existing file changes. |
| A15 | cosmetic | **_emit does not escape LaTeX specials.** Method names such as dp_fedavg_lr contain underscores, which break the .tex table outside math mode. | `fedverify/analysis/make_tables.py:121` | make_survival_tables escapes '_' (and '%', '&') in labels before calling _emit. |
| A16 | cosmetic | **FLConfig's auto cell name assumes a Dirichlet split.** FLConfig(dataset='tcga_brca') gives cell 'tcga_brca_K5_a0.5_epsinf' although no Dirichlet draw happens (smoke). | `fedverify/config.py:68` | Survival's own config names cells family_method_eps.. (pattern R23). |
| A17 | cosmetic | **push.py only sends test_acc / macro_f1 as metrics.** Survival rounds would be pushed without metrics, so the dashboard shows the lineage but no metric curve. | `fedverify/chain/push.py:124` | Accept it (frontend is out of scope, §9) or add a small Survival push wrapper that sends c_index / auc / f1; the route already accepts arbitrary numeric metric names. |
| A19 | cosmetic | **Run ids must match the route's charset.** A run id containing '/' or spaces is rejected by POST /api/fl/round/commit. | `backend/fl_routes.py:70` | Survival run ids use only letters, digits and . _ - =. |

## 4. Data - verified

Verbatim output of `python3 -I fedverify/survival/phase0_check.py` (exit status 1, because of the unit check):

- file: `fedverify/data/tcga_brca/clinical.xlsx`  
- sha256: `326d0f93bcdfd02cff803e30682d09153edb4ff5593ec82895e7beb4b3ef8b29`
- sheets: ['Sheet1']; shape: **907 rows x 34 columns**

**Sites, all rows, before any dropping** (the five used are marked *):
| site | patients | deaths | prompt says (patients / deaths) | match |
|---|---|---|---|---|
| *BH | 145 | 45 | 145 / 45 | yes |
| *A2 | 98 | 11 | 98 / 11 | yes |
| *E2 | 87 | 6 | 87 / 6 | yes |
| *A8 | 76 | 4 | 76 / 4 | yes |
| *AR | 68 | 10 | 68 / 10 | yes |
| D8 | 62 | 1 |  |  |
| E9 | 60 | 7 |  |  |
| C8 | 49 | 2 |  |  |
| B6 | 48 | 21 |  |  |
| AO | 46 | 3 |  |  |
| AN | 38 | 0 |  |  |
| A7 | 33 | 0 |  |  |

Top-5 total: **474 patients, 76 deaths** (prompt: 474 / 76).
Sites with >= 30 patients: **12**; of those, **8** have fewer than 10 deaths (prompt: 12 and 8).

**Usable time** (time = survival x 365.25 if event == 1 else days_to_last_follow_up; a row is dropped when that value is missing / non-numeric):
| site | patients | deaths | usable rows | usable deaths | dropped rows | dropped deaths | usable rows with time = 0 | of which deaths |
|---|---|---|---|---|---|---|---|---|
| BH | 145 | 45 | 145 | 45 | 0 | 0 | 0 | 0 |
| A2 | 98 | 11 | 98 | 11 | 0 | 0 | 1 | 0 |
| E2 | 87 | 6 | 87 | 6 | 0 | 0 | 0 | 0 |
| A8 | 76 | 4 | 76 | 4 | 0 | 0 | 9 | 0 |
| AR | 68 | 10 | 68 | 10 | 0 | 0 | 0 | 0 |
| **top-5** | 474 | 76 | 474 | 76 | 0 | 0 | 10 | 0 |

Whole workbook: event values {'0': 786, '1': 121}; deaths with no `survival` value: 1; censored rows that do carry a `survival` value: 0; censored rows with no follow-up days: 0.
Non-numeric tokens found in the numeric time columns: {'survival_event': {}, 'survival': {'<blank>': 787}, 'days_to_last_follow_up': {'[]': 87}, 'OS_MONTHS': {'<blank>': 186}}.

**Unit sanity check** (`|time_days/30.4375 - OS_MONTHS| < 0.5`, only rows that have OS_MONTHS and a usable time; OS_MONTHS is never a feature):
| scope | rows checked | of which deaths | of which censored | pass | fail | max abs diff (months) | median abs diff (months) |
|---|---|---|---|---|---|---|---|
| whole workbook | 720 | 96 | 624 | 709 | 11 | 103.8692 | 0.0216 |
| top-5 sites | 407 | 64 | 343 | 400 | 7 | 103.8692 | 0.0260 |

Pass rate 98.47% (709/720); the 11 failures are 6 deaths + 5 censored rows, 7 of them in the top-5 sites, 4 with time_days == 0. Event flag `survival_event` vs `OS_STATUS`: 1 disagreements in 721 rows compared. All failing rows:

| submitter_id | site | event | OS_STATUS event | time_days | OS_MONTHS | abs diff (months) |
|---|---|---|---|---|---|---|
| TCGA-BH-A1FG | BH | 1 | 1 | 577.0 | 122.826 | 103.869 |
| TCGA-E2-A15O | E2 | 0 | 0 | 289.0 | 50.794 | 41.299 |
| TCGA-BH-A1F8 | BH | 1 | 1 | 1.0 | 25.085 | 25.052 |
| TCGA-AR-A1AJ | AR | 0 | 0 | 2383.0 | 100.996 | 22.705 |
| TCGA-BH-A18N | BH | 1 | 1 | 468.0 | 37.742 | 22.366 |
| TCGA-D8-A1JK | D8 | 0 | 0 | 0.0 | 20.12 | 20.12 |
| TCGA-A2-A25D | A2 | 0 | 0 | 0.0 | 18.148 | 18.148 |
| TCGA-B6-A0IH | B6 | 1 | 1 | 2965.0 | 112.371 | 14.959 |
| TCGA-C8-A26Y | C8 | 0 | 0 | 0.0 | 12.953 | 12.953 |
| TCGA-A8-A08L | A8 | 1 | 1 | 30.0 | 9.994 | 9.009 |
| TCGA-AC-A23H | AC | 1 | 1 | 0.0 | 5.72 | 5.72 |

**Feature columns** (numeric):
| column | numeric cells | unusable cells | unusable tokens | min | max | distinct values |
|---|---|---|---|---|---|---|
| diag_age | 895 | 12 | {'<blank>': 12} | 26.59178082191781 | 90.06027397260274 | 872 distinct |
| lymph_node | 611 | 296 | {'<blank>': 296} | 0.0 | 2.0 | [0.0, 1.0, 2.0] |
| size_over_20mm | 906 | 1 | {'<blank>': 1} | 0.0 | 1.0 | [0.0, 1.0] |

**Feature columns** (categorical; raw tokens with counts over all 907 rows; `[Not Available]`, `[Not Evaluated]` and blank all become category `NA`):
| column | raw tokens | rows mapped to NA |
|---|---|---|
| menop_status | {'Post': 639, 'Pre': 240, 'Indeterminate (neither Pre or Postmenopausal)': 26, '<blank>': 2} | 2 |
| er_status_by_ihc | {'Positive': 519, '<blank>': 186, 'Negative': 165, '[Not Evaluated]': 36, 'Indeterminate': 1} | 222 |
| pr_status_by_ihc | {'Positive': 454, 'Negative': 226, '<blank>': 186, '[Not Evaluated]': 37, 'Indeterminate': 4} | 223 |
| her2_status_by_ihc | {'Negative': 382, '<blank>': 186, 'Equivocal': 118, 'Positive': 116, '[Not Evaluated]': 96, '[Not Available]': 7, 'Indeterminate': 2} | 289 |
| histological_type | {'Infiltrating Ductal Carcinoma': 595, '<blank>': 186, 'Infiltrating Lobular Carcinoma': 63, 'Other, specify': 28, 'Mixed Histology (please specify)': 25, 'Mucinous Carcinoma': 5, 'Medullary Carcinoma': 4, '[Not Available]': 1} | 187 |

One-hot vocabulary sizes (NA always included): {'menop_status': 4, 'er_status_by_ihc': 4, 'pr_status_by_ihc': 4, 'her2_status_by_ihc': 5, 'histological_type': 7}. Model input width = 24 one-hot + 3 numeric + 3 missing-indicators = **30**.

**Share of missing / NA cells per feature, per site** (top-5 sites):
| feature | BH | A2 | E2 | A8 | AR |
|---|---|---|---|---|---|
| diag_age | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| lymph_node | 29.7% | 8.2% | 43.7% | 46.1% | 48.5% |
| size_over_20mm | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| menop_status | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| er_status_by_ihc | 11.7% | 23.5% | 12.6% | 6.6% | 20.6% |
| pr_status_by_ihc | 11.7% | 23.5% | 12.6% | 6.6% | 20.6% |
| her2_status_by_ihc | 22.1% | 34.7% | 16.1% | 6.6% | 22.1% |
| histological_type | 10.3% | 23.5% | 12.6% | 6.6% | 20.6% |

**Leakage deny-list vs the actual columns:**
- columns starting OS_/DFS_/PFS_ present in the workbook (6): ['DFS_STATUS', 'DFS_MONTHS', 'OS_STATUS', 'OS_MONTHS', 'PFS_STATUS', 'PFS_MONTHS']
- exact deny-list names present (7): ['survival', 'survival_event', 'days_to_last_follow_up', 'diag_date', 'year_of_birth', 'submitter_id', 'case_id']; absent: []
- columns that are neither a feature nor on the deny-list (13): ['Unnamed: 0.1', 'Unnamed: 0', 'gender', 'race', 'state', 'morphology', 'prior_malignancy', 'treatment_type', 'treatment_or_therapy', 'er_status_ihc_Percent_Positive', 'her2_fish_status', 'menopause_status_original', 'primary_diagnosis']

**Expectation checks against master-prompt sections 3-4:**
| check | prompt expects | found | result |
|---|---|---|---|
| workbook shape (rows, columns) | [907, 34] | [907, 34] | PASS |
| site BH patients | 145 | 145 | PASS |
| site BH deaths | 45 | 45 | PASS |
| site A2 patients | 98 | 98 | PASS |
| site A2 deaths | 11 | 11 | PASS |
| site E2 patients | 87 | 87 | PASS |
| site E2 deaths | 6 | 6 | PASS |
| site A8 patients | 76 | 76 | PASS |
| site A8 deaths | 4 | 4 | PASS |
| site AR patients | 68 | 68 | PASS |
| site AR deaths | 10 | 10 | PASS |
| top-5 total patients | 474 | 474 | PASS |
| top-5 total deaths | 76 | 76 | PASS |
| the 5 chosen sites are the 5 largest by patients | ['A2', 'A8', 'AR', 'BH', 'E2'] | ['A2', 'A8', 'AR', 'BH', 'E2'] | PASS |
| sites with >= 30 patients | 12 | 12 | PASS |
| ... of which fewer than 10 deaths | 8 | 8 | PASS |
| all 8 feature columns present | True | True | PASS |
| submitter_id all of the form TCGA-<site>-<id> | 0 | 0 | PASS |
| duplicate submitter_id | 0 | 0 | PASS |
| duplicate case_id | 0 | 0 | PASS |
| unit sanity: rows with \|time_days/30.4375 - OS_MONTHS\| >= 0.5 | 0 | 11 | **FAIL** |

RESULT: 1 expectation(s) differ from the prompt - STOP and report.

### Observations that shape Phase 1 (every number is in the output above or in the probe output below)

- The workbook carries two leftover index columns (`Unnamed: 0.1`, `Unnamed: 0`) that are neither features nor on the
  deny-list. The leakage guard should therefore be **whitelist-first** (only the eight named features can reach the
  model), with the deny-list as the raising second check the prompt asks for.
- 186 rows have no `OS_*` values; the same rows have blank ER / PR / HER2 / histology cells (they lack the second
  clinical source). Per-hospital missing shares differ a lot (table above), so `<col>_missing` indicators and `NA`
  categories can act as a site proxy in the pooled baselines (noted in TODO.md).
- With the prompt's preprocessing the logistic model has **30 inputs** (24 one-hot + 3 numeric +
  3 missing-indicators), i.e. 31 parameters with the bias. Several vocabulary entries cover only a
  handful of patients.
- The five hospitals are the five largest by patients, as the prompt says; the sites table shows a site outside them
  with more deaths than several inside (noted in TODO.md).
- 10 censored patients in the five hospitals have `days_to_last_follow_up == 0` (column "usable rows with time = 0";
  none is a death). They are kept under the prompt's rule. D3.

Probe run once on the real data with the would-be-installed scikit-learn (scratch, not kept as a file):

```
sklearn 1.9.1
strata sizes: {'A2_0': 87, 'A2_1': 11, 'A8_0': 72, 'A8_1': 4, 'AR_0': 58, 'AR_1': 10, 'BH_0': 100, 'BH_1': 45, 'E2_0': 81, 'E2_1': 6}
warnings emitted by StratifiedKFold: ['UserWarning: The least populated class in y has only 4 members, which is less than n_splits=5.']
deaths per test fold (all sites): [15, 15, 15, 16, 15]
  BH: deaths per test fold [9, 9, 9, 9, 9] | train rows per fold [116, 116, 116, 116, 116]
  A2: deaths per test fold [2, 2, 2, 3, 2] | train rows per fold [79, 78, 78, 78, 79]
  E2: deaths per test fold [1, 1, 1, 1, 2] | train rows per fold [69, 70, 70, 70, 69]
  A8: deaths per test fold [1, 1, 1, 1, 0] | train rows per fold [61, 61, 61, 60, 61]
  AR: deaths per test fold [2, 2, 2, 2, 2] | train rows per fold [54, 54, 54, 55, 55]
tree_.value shape (9, 1, 2) | per-node sum min/max: 1.0 1.0
```

So `StratifiedKFold(n_splits=5)` on (hospital, event) warns because A8 has only 4 deaths, one fold has no A8 death in
its test part, the per-hospital training sizes match the prompt's "about 55-115 rows", and scikit-learn stores class
**fractions** in `tree_.value` (each node sums to 1), which the Phase-3 tree serializer must reproduce exactly.

## 5. Environment and test baseline

| | |
|---|---|
| Python | Python 3.12.3 |
| Installed and used as is | numpy 2.5.2, scipy 1.18.1, pandas 3.0.5, torch 2.14.0+cpu, opacus 1.6.0, cryptography 41.0.7, matplotlib 3.11.1, PyYAML 6.0.1, wfdb 4.3.1, flask 3.1.3 |
| Installed by Phase 0 | openpyxl 3.1.5 (+ et_xmlfile 2.0.0), nothing else |
| Not installed yet | lifelines, scikit-learn |
| Real test run | `python3 -m pytest backend/ fedverify/tests/ -q` gives **259 passed**, 18 warnings, 38.03 s |
| Same suite on the would-be-installed stack (scratch `pip --target` dir placed first on `PYTHONPATH`: lifelines 0.30.3, scikit-learn 1.9.1, pandas 2.3.3, numpy 2.5.3, matplotlib 3.11.2) | **259 passed**, 18 warnings, 32.76 s |
| Who imports pandas in this repo | only the new `fedverify/survival/phase0_check.py`; the one installed package that depends on it is `wfdb` (needs pandas>=2.2.3, met by 2.3.x) |
| Why lifelines downgrades pandas | lifelines 0.30.3 declares `pandas<3.0,>=2.1`; the real environment has pandas 3.0.5 |
| `phase0_check.py` under pandas 2.3.3 | byte-identical stdout and JSON to the pandas 3 run |

lifelines C-index sign, checked on a hand-made example in the scratch stack (perfect ranking: earliest death has the
highest risk):

```
numpy 2.5.3 | pandas 2.3.3 | scipy 1.18.1 | sklearn 1.9.1 | lifelines 0.30.3 | torch 2.14.0+cpu
concordance_index(T, risk, E) = 0.0
concordance_index(T, -risk, E) = 1.0
```

So `concordance_index(time, -risk, event)` is the right call, exactly as the prompt says.

## 6. Decisions needed before Phase 1 (recommended default first)

If you reply `go phase 1` with no comment I apply the **default** in every row. Rows marked *(you)* need an action only
you can take; they are repeated in the completion block.

| # | Question | Why it matters | Default if you say nothing |
|---|---|---|---|
| D1 *(you)* | `CLAUDE.md` is deleted in the working tree. Restore it, or was that deliberate? | The prompt makes it the highest authority and I may not run `git checkout` | I keep following the committed version (A1-A11). If the deletion was an accident: `git restore CLAUDE.md` |
| D2 | The `OS_MONTHS` sanity test fails for 11 rows (7 in the five hospitals): 6 deaths and 5 censored, 4 of them with `time_days == 0`. What should the Phase-1 test assert? | Written literally it fails; dropping the rows would change the §3 death counts | It asserts (a) a pass rate of at least 98% on the checked rows and (b) that the set of failing `submitter_id`s equals a **pinned list** (the rows in section 4). A real unit mistake (hundreds of failures) or any new exception then fails loudly. No row is dropped |
| D3 | Keep the censored patients with `days_to_last_follow_up == 0`? | They look like missing-value placeholders. The C-index ignores them; AUC and F1 count them as survivors | Keep them (the prompt's rule only drops missing time) and state the count in the table notes |
| D4 | Where do Survival tests live? | The suite command collects only `backend/` and `fedverify/tests/` | Code in `fedverify/survival/`, tests in `fedverify/tests/test_survival_*.py` |
| D5 | Where do `chain/contentid.py`, `chain/signing.py`, `analysis/make_survival_tables.py`, `analysis/make_results_draft.py` live? | Ground rule 4 and the phase text disagree | The paths the phase text names: new files in existing packages, no existing file edited |
| D6 | RF "update" vs CLAUDE.md "DELTAS only" | Trees have no weights to subtract | A hospital's update is **the trees it adds this round**; the global forest is the previous forest plus accepted new trees. Written up in `DESIGN_NOTES.md` |
| D7 | Seeds | `--seed N --repeats 5` vs "seeds 0-4" | `--seed N` is the base seed and repeat r uses seed N + r, so `--seed 0 --repeats 5` is seeds 0-4 |
| D8 *(you)* | Install the dependencies? `pip install --break-system-packages -r fedverify/survival/requirements.txt` downgrades pandas 3.0.5 to 2.3.x | Phase 1 cannot run without lifelines and scikit-learn; section 5 shows the suite passes on that stack | You run that command, then the suite (expect 259 passed). A venv is the alternative |
| D9 *(you)* | Delete or ignore the untracked copy `clinical_project_tcga_brca_2024-04-08_converted.xlsx` in the repo root? | Identical to the spec-path copy but not gitignored, so `git add -A` would commit it | I leave it alone |
| D10 | Hospital ids: strings, but existing code wants ints | Section 3: A01-A12 | Hospital **codes** (`BH`, `A2`, `E2`, `A8`, `AR`) are the only ids written to disk or chain: Merkle leaves `[i, "BH", digest]`, signed submissions, `diag` accepted / rejected / scores, attackers. Inside calls to existing code a thin adapter uses the fixed indices BH=0, A2=1, E2=2, A8=3, AR=4 and translates back at once. `config.json` records the mapping. One namespace everywhere keeps the lineage view consistent (A18) |
| D11 | `model_family` in `rounds.jsonl` | A9 forbids changing existing record shapes | Added to Survival records only (`"lr"` / `"rf"`); existing families stay byte-identical |
| D12 | SGD momentum for the logistic family (the prompt fixes lr 0.1, batch 32, 3 local epochs, but not momentum) | The existing client reads `cfg.momentum`, default 0.9, which would turn "lr 0.1" into a much larger effective step | Plain SGD, momentum 0.0, documented in `survival/config.py`; FedSGD's single full-batch step is then exactly one gradient step |

## 7. Smoke checks run against the existing code (read-only, scratch only)

Each check called the real functions with Survival-shaped inputs in a throwaway Python process (mock anchor only, nothing written in the repo; `git status` and `backend/data/` were compared before and after and were unchanged).

| check | observed |
|---|---|
| Five ClientUpdates with ids BH..AR through every registered aggregator | fedavg: raises; krum: raises; multikrum: raises; trimmed_mean: raises; median: raises; forensics: raises (ValueError: invalid literal for int() with base 10: 'BH') |
| Same updates with ids 0-4 | fedavg: accepted [0, 1, 2, 3, 4]; krum: accepted [2]; multikrum: accepted [0, 1, 2, 4]; trimmed_mean: accepted [0, 1, 2, 3, 4]; median: accepted [0, 1, 2, 3, 4]; forensics: accepted [0, 1, 2, 3, 4] |
| `Client('BH', ...)` | raises ValueError: invalid literal for int() with base 10: 'BH' |
| Forensics, K=5, five identical updates | rejected [] |
| Forensics, K=5, one all-NaN update (index 3) | rejected [3] by `health:nan` |
| `build_aggregator('forensics')` with no tau anywhere | OK: Forensics |
| `Forensics.aggregate(...)` with `cfg.tau = None` | raises ValueError: forensics aggregator requires cfg.tau |
| Forensics combined score, one `sign_flip` attacker (index 2) among 5 random 31-value deltas, tau=inf (illustrative only: random deltas, not trained ones) | 0: 0.674, 1: 1.581, 2: 1.9, 3: 0.607, 4: 0.0 |
| Forensics combined score, one `scaling` attacker (index 2) among 5 random 31-value deltas, tau=inf (illustrative only: random deltas, not trained ones) | 0: 1.942, 1: 0.544, 2: 35.85, 3: 0.674, 4: 0.674 |
| `update_digest('BH', ...)` | raises ValueError: invalid literal for int() with base 10: 'BH' |
| `commit_round` on updates with ids BH..AR | raises ValueError: invalid literal for int() with base 10: 'BH' |
| `round_merkle_root` / `inclusion_proof` / `verify_proof` with entries `("BH", digest)`... | leaf_count 5; proof verifies: True; forged root fails: True; swapping two hospitals changes the root: True |
| `canon_update(b'\x00\x01\x02\x03')` | raises ValueError: could not convert string to float: np.bytes_(b'\x00\x01\x02\x03') |
| `canon_update` on an int32 tensor of 4 values | OK, 16 bytes (silently cast to float32) |
| `ChainAnchor('mock').anchor_round('survival_lr_ours_s0', ...)` | OK, returns backend, bytes_written, fee, latency_ms, leaf_count, root, round, txid |
| `attacker_ids` at K=5, attacker_frac=0.2, seeds 0-4 (index, and hospital under the D10 order) | seed 0: [4] = AR; seed 1: [3] = A8; seed 2: [2] = E2; seed 3: [1] = A2; seed 4: [1] = A2. BH, which holds most of the deaths, is never the attacker in seeds 0-4 (worth knowing for Phase 4) |
| `poison_batch` label_flip with num_classes=2 on labels [0, 1, 1] | OK: [1, 0, 0] |
| `plan_privacy(cfg, 'BH', 60)` | raises ValueError: invalid literal for int() with base 10: 'BH' |
| Noise multiplier `plan_privacy` solves for 30 rounds x 3 local epochs, batch 32, delta 1e-5, at training-fold sizes | eps 1: n=54 sigma 32.5, n=60 sigma 29.062, n=69 sigma 30.938, n=78 sigma 27.5, n=116 sigma 21.406; eps 2: n=54 sigma 17.188, n=60 sigma 15.469, n=69 sigma 16.484, n=78 sigma 14.609, n=116 sigma 11.367; eps 4: n=54 sigma 9.277, n=60 sigma 8.379, n=69 sigma 8.906, n=78 sigma 7.9, n=116 sigma 6.172; eps 8: n=54 sigma 5.146, n=60 sigma 4.653, n=69 sigma 4.946, n=78 sigma 4.399, n=116 sigma 3.462 |
| `load_tau('fedverify/results', 4.0, 5)` against the existing taus.json | returns tau 2.9016 from `eps=1\|K=10:tau_fpr5` |
| `load_tau('fedverify/results', inf, 5)` | returns tau 2.9016 from `eps=1\|K=10:tau_fpr5` |
| `load_tau` with no taus.json | raises FileNotFoundError: /nonexistent_dir_for_smoke/calibration/taus.json not found |
| `FLConfig(dataset='tcga_brca', num_clients=5, exp='survival')` | cell `tcga_brca_K5_a0.5_epsinf`, run_dir `fedverify/results/survival/tcga_brca_K5_a0.5_epsinf/seed0` |

## 8. Working-tree state and regeneration

`git status --porcelain` **before** Phase 0 touched anything:

```
 D CLAUDE.md
 M backend/bitcoin_client.py
 M frontend/index.html
?? clinical_project_tcga_brca_2024-04-08_converted.xlsx
?? fedverify/docs/FEDVERIFY_COMPLETE.md
```

and **after** Phase 0 (new files are Phase 0's; `fedverify/data/` and `fedverify/results/` are gitignored):

```
 D CLAUDE.md
 M backend/bitcoin_client.py
 M fedverify/TODO.md
 M frontend/index.html
?? clinical_project_tcga_brca_2024-04-08_converted.xlsx
?? fedverify/docs/FEDVERIFY_COMPLETE.md
?? fedverify/survival/
```

Phase 0 never ran `git add/commit/push/checkout/stash/rebase`. Not touched by Phase 0 but worth knowing: the
uncommitted edits in `backend/bitcoin_client.py` and `frontend/index.html` and the untracked
`fedverify/docs/FEDVERIFY_COMPLETE.md` come from earlier work.

Regenerate / re-check:

```
python3 -I fedverify/survival/phase0_check.py                  # section 4 (exits 1 while the D2 exceptions exist)
python3 -m pytest backend/ fedverify/tests/ -q                 # section 5 baseline
git check-ignore -v fedverify/data/tcga_brca/clinical.xlsx     # gitignore coverage
```

Sections 1b, 2, 3 and 7 are a point-in-time reading of the code and are not regenerated by a script; re-check any row
with `sed -n '<line>p' <file>` or by searching for its quoted line (appendix).

## Appendix - the source line behind every file:line above

| where | source line (verbatim, whitespace-normalised) | id |
|---|---|---|
| `backend/algorand_client.py:398` | `def broadcast_fl_round(run_id: str, round_num: int, round_root: str,` | R52 |
| `backend/app.py:1562` | `def _manifest_leaf(index: int, name: str, layer_hash: str) -> str:` | R48 |
| `backend/bitcoin_client.py:84` | `def anchor_merkle_root(merkle_root: str) -> dict:` | R53 |
| `backend/fl_routes.py:70` | `RUN_ID_RE = re.compile(r"^[A-Za-z0-9._=-]{1,200}$")` | A19 |
| `backend/fl_routes.py:123` | `def commit_round_route():` | R55 |
| `backend/fl_routes.py:281` | `lf = next((x for x in r["leaves"] if x["client_id"] == str(client_id)), None)` | A18 |
| `backend/fl_routes.py:296` | `def get_lineage(run_id):` | R56 |
| `evaluation/eval_lib.py:75` | `def bv_find_outliers(arr, z_thresh=8.0):` | R57 |
| `evaluation/eval_lib.py:92` | `def bv_layer_health(arr):` | R58 |
| `evaluation/eval_lib.py:154` | `def roc_curve(scores, labels):` | R59 |
| `fedverify/aggregators/__init__.py:19` | `def build_aggregator(name: str) -> Aggregator:` | R15 |
| `fedverify/aggregators/base.py:15` | `class Aggregator(ABC):` | R16 |
| `fedverify/aggregators/base.py:41` | `sel = [u for u in updates if ids is None or int(u.client_id) in ids]` | A02 |
| `fedverify/aggregators/fedavg.py:19` | `diag["accepted"] = [int(u.client_id) for u in updates]` | A04 |
| `fedverify/aggregators/forensics.py:63` | `def robust_z(values: np.ndarray, one_sided: str = "both") -> np.ndarray:` | R18, M06 |
| `fedverify/aggregators/forensics.py:114` | `class Forensics(Aggregator):` | R17 |
| `fedverify/aggregators/forensics.py:122` | `tau = getattr(cfg, "tau", None)` | M01 |
| `fedverify/aggregators/forensics.py:132` | `ids = [int(u.client_id) for u in updates]` | A03 |
| `fedverify/aggregators/krum.py:36` | `ids = [int(u.client_id) for u in updates]` | A05 |
| `fedverify/aggregators/median.py:24` | `diag["accepted"] = [int(u.client_id) for u in updates]` | A06 |
| `fedverify/aggregators/trimmed_mean.py:36` | `diag["accepted"] = [int(u.client_id) for u in updates]` | A07 |
| `fedverify/analysis/calibrate.py:57` | `dataset="mnist", num_clients=k, rounds=rounds, alpha=0.5, seed=seed,` | A13 |
| `fedverify/analysis/calibrate.py:67` | `def harvest(run_dir):` | R61 |
| `fedverify/analysis/calibrate.py:83` | `labels.append(1 if (active and int(cid) in attackers) else 0)` | A12 |
| `fedverify/analysis/calibrate.py:87` | `def curve_stats(scores, labels):` | R60 |
| `fedverify/analysis/calibrate.py:201` | `key = f"eps={eps_tag(c.get('epsilon'))}\|K={c.get('num_clients')}"` | M07 |
| `fedverify/analysis/calibrate.py:255` | `for key in (want, f"eps=inf\|K={num_clients}", *sorted(data)):` | A14 |
| `fedverify/analysis/make_paper.py:54` | `def collect(results_dir: str) -> dict:` | R65 |
| `fedverify/analysis/make_tables.py:23` | `def load_runs(results_dir: str, exp: str) -> list[dict]:` | R64 |
| `fedverify/analysis/make_tables.py:91` | `def mean_std(xs: list[float]) -> tuple[Optional[float], Optional[float]]:` | R63 |
| `fedverify/analysis/make_tables.py:108` | `def _emit(name: str, header: list[str], rows: list[list[str]], out_dir: str,` | R62 |
| `fedverify/analysis/make_tables.py:121` | `.replace(MISSING, "--") for c in r)` | A15 |
| `fedverify/attacks/byzantine.py:26` | `DATA_ATTACKS = {"label_flip", "backdoor"}` | M03 |
| `fedverify/attacks/byzantine.py:30` | `def attacker_ids(cfg) -> Set[int]:` | R19 |
| `fedverify/attacks/byzantine.py:42` | `def attack_active(cfg, round_num: int) -> bool:` | R20 |
| `fedverify/attacks/byzantine.py:57` | `def apply_delta_attack(delta: torch.Tensor, cfg, round_num: int, client_id: int,` | R21 |
| `fedverify/attacks/byzantine.py:85` | `def poison_batch(xb: torch.Tensor, yb: torch.Tensor, cfg, round_num: int, client_id: int,` | R22 |
| `fedverify/chain/__init__.py:1` | `"""Phase 3` | R66 |
| `fedverify/chain/anchor.py:53` | `class ChainAnchor:` | R49 |
| `fedverify/chain/anchor.py:79` | `def anchor_round(self, run_id: str, round_num: int, root: str, leaf_count: int,` | R50 |
| `fedverify/chain/anchor.py:142` | `def checkpoint(self, run_id: str, round_num: int) -> dict:` | R51 |
| `fedverify/chain/commitment.py:46` | `def warm_up() -> None:` | R46 |
| `fedverify/chain/commitment.py:57` | `def canon_update(delta) -> bytes:` | R40 |
| `fedverify/chain/commitment.py:65` | `arr = np.ascontiguousarray(np.asarray(delta), dtype="<f4")` | A10 |
| `fedverify/chain/commitment.py:69` | `def update_digest(client_id: int, round_num: int, delta, num_samples: int) -> str:` | R41 |
| `fedverify/chain/commitment.py:77` | `{"client_id": int(client_id), "num_samples": int(num_samples),` | A09 |
| `fedverify/chain/commitment.py:94` | `def leaf_hashes(leaves: Sequence) -> List[str]:` | R45 |
| `fedverify/chain/commitment.py:100` | `def round_merkle_root(leaves: Sequence) -> Tuple[str, int]:` | R42 |
| `fedverify/chain/commitment.py:111` | `def inclusion_proof(leaves: Sequence, index: int) -> List[dict]:` | R43 |
| `fedverify/chain/commitment.py:119` | `def verify_proof(leaf: str, path: Iterable[dict], root: str) -> bool:` | R44 |
| `fedverify/chain/commitment.py:134` | `def commit_round(updates, round_num: int):` | R47 |
| `fedverify/chain/push.py:105` | `def push(run_dir: str, api: str = DEFAULT_API, run_id: str = None, verbose: bool = True):` | R54 |
| `fedverify/chain/push.py:124` | `body["metrics"] = {k: m[k] for k in ("test_acc", "macro_f1")` | A17 |
| `fedverify/config.py:20` | `class FLConfig:` | R23 |
| `fedverify/config.py:35` | `out_dir: str = "fedverify/results"` | M05 |
| `fedverify/config.py:44` | `delta: float = 1e-5` | R24 |
| `fedverify/config.py:68` | `cell = f"{self.dataset}_K{self.num_clients}_{split}_eps{eps}"` | A16 |
| `fedverify/core/client.py:21` | `class ClientUpdate:` | R07 |
| `fedverify/core/client.py:33` | `self.id = int(client_id)` | A01 |
| `fedverify/core/client.py:59` | `def local_train(self, global_params: torch.Tensor, cfg, model: nn.Module = None,` | R31 |
| `fedverify/core/data.py:28` | `def dataset_spec(name: str) -> dict:` | R30 |
| `fedverify/core/models.py:82` | `def assert_no_batchnorm(model: nn.Module) -> None:` | R10 |
| `fedverify/core/models.py:88` | `def get_flat_params(model: nn.Module) -> torch.Tensor:` | R08 |
| `fedverify/core/models.py:93` | `def set_flat_params(model: nn.Module, vec: torch.Tensor) -> None:` | R09 |
| `fedverify/core/privacy.py:41` | `def dp_enabled(cfg) -> bool:` | R14 |
| `fedverify/core/privacy.py:94` | `def plan_privacy(cfg, client_id: int, n_k: int) -> Optional[PrivacyPlan]:` | R11 |
| `fedverify/core/privacy.py:113` | `client_id=int(client_id), sigma=float(sigma),` | A08 |
| `fedverify/core/privacy.py:121` | `def make_private(model, optimizer, data_loader, plan: PrivacyPlan, criterion=None):` | R12 |
| `fedverify/core/privacy.py:146` | `def privacy_report(plan: Optional[PrivacyPlan]) -> Optional[dict]:` | R13 |
| `fedverify/core/runner.py:44` | `def strip_timings(obj):` | R01 |
| `fedverify/core/runner.py:53` | `def seed_everything(seed: int) -> None:` | R02 |
| `fedverify/core/runner.py:76` | `def run(cfg: FLConfig, limit_train: Optional[int] = None, limit_test: Optional[int] = None,` | R32 |
| `fedverify/core/runner.py:87` | `train_set, test_set = load_dataset(cfg.dataset)` | M04 |
| `fedverify/core/runner.py:135` | `with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:` | R03 |
| `fedverify/core/runner.py:200` | `commit_rec = {` | R05 |
| `fedverify/core/runner.py:219` | `rec = {` | R04 |
| `fedverify/core/runner.py:221` | `"test_acc": metrics["accuracy"],` | M02 |
| `fedverify/core/runner.py:265` | `def _attack_report(cfg, rnd, attackers, diag, selected) -> dict:` | R06 |
| `fedverify/core/runner.py:276` | `rejected = set(int(c) for c in diag.get("rejected", []))` | A11 |
| `fedverify/core/server.py:22` | `def evaluate(model: nn.Module, loader, device="cpu") -> dict:` | R29 |
| `fedverify/experiments/exp2_byzantine.py:89` | `def _run_parallel(todo, jobs, threads):` | R28 |
| `fedverify/tests/test_aggregators.py:223` | `def test_phase1_record_shape_is_unchanged_by_phase4(tmp_path):` | R27 |
| `fedverify/tests/test_privacy.py:45` | `def test_eps_inf_is_identical_to_phase1_and_never_touches_opacus(tmp_path, monkeypatch):` | R26 |
| `fedverify/tests/test_privacy.py:66` | `def test_noise_multiplier_is_solved_over_TOTAL_steps(monkeypatch):` | R25 |
