# FedVerify

Federated-learning add-on for BlockVerify. **Phase 1** ships the FL simulator core:
partitioning, model, client/server, and a deterministic training loop.
No DP (Phase 2), no chain commitment (Phase 3), no attacks (Phase 4) yet.

## Install

PyTorch **must** come from the CPU-only index — plain PyPI `torch` pulls ~4 GB of NVIDIA
CUDA packages (see `CLAUDE.md` amendment A2):

```bash
pip install --break-system-packages --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install --break-system-packages tqdm
```

Datasets download automatically into `fedverify/data/` (gitignored).

## Run

All commands run from the repository root.

```bash
# default: MNIST, 10 clients, 30 rounds, Dirichlet alpha=0.5
python3 -m fedverify.core.runner --dataset mnist --num-clients 10 --rounds 30 --alpha 0.5 --seed 0

# IID baseline (alpha=inf) — expect ~97-98% after 30 rounds
python3 -m fedverify.core.runner --dataset mnist --num-clients 10 --rounds 30 --alpha inf --seed 0

# severe non-IID — visibly worse and noisier than IID
python3 -m fedverify.core.runner --dataset mnist --num-clients 10 --rounds 30 --alpha 0.1 --seed 0

# Fashion-MNIST
python3 -m fedverify.core.runner --dataset fmnist --num-clients 10 --rounds 30 --alpha 0.5 --seed 0

# fast smoke (subsampled)
python3 -m fedverify.core.runner --dataset mnist --num-clients 2 --rounds 2 --alpha inf \
        --limit-train 200 --limit-test 200 --exp smoke
```

Useful flags: `--local-epochs --batch-size --lr --momentum --client-fraction --device
--out-dir --exp --cell --run-id`. Later-phase flags (`--epsilon --attack --tau
--chain-backend …`) are already accepted and recorded, but inert in Phase 1.

## Output

```
fedverify/results/<exp>/<cell>/seed<N>/
    rounds.jsonl      one JSON object per round
    config.json       every parameter that influenced the run + partition report
    final_model.pt    final global model
```

`rounds.jsonl` fields: `round, test_acc, test_loss, macro_f1, mean_train_loss,
per_client_num_samples, diag, train_wall_s, round_wall_s`.
`diag` always carries `accepted / rejected / scores`.

## Determinism

Results are only byte-reproducible at a **fixed intra-op thread count** (amendment A5):
thread count changes float reduction order. Pin it with `--torch-threads N`; every run
records `torch_threads_effective` in `config.json`. Run a whole grid with one
`--threads` value.

Same seed ⇒ identical `rounds.jsonl` **after dropping wall-clock timing keys**
(`train_wall_s`, `round_wall_s`, `*_ms`) — amendment A1; timings can never be
byte-identical. Use `fedverify.core.runner.strip_timings` to compare:

```bash
python3 -m fedverify.core.runner --rounds 2 --num-clients 2 --alpha inf \
        --limit-train 200 --exp det --run-id A
python3 -m fedverify.core.runner --rounds 2 --num-clients 2 --alpha inf \
        --limit-train 200 --exp det --run-id B
python3 - <<'PY'
import json
from fedverify.core.runner import strip_timings
L=lambda p:[strip_timings(json.loads(l)) for l in open(p)]
a=L("fedverify/results/det/mnist_K2_iid_epsinf/seed0/rounds.jsonl")
print("identical:", a==a)
PY
```

## Tests

```bash
python3 -m pytest fedverify/tests/ -q      # Phase 1: 27 tests
python3 -m pytest backend/tests/  -q       # existing BlockVerify: must stay >= 99
```

## Layout

```
fedverify/
  config.py          FLConfig (frozen dataclass) + argparse
  core/data.py       load_dataset, dirichlet_partition, iid_partition, partition_report
  core/models.py     SmallCNN (GroupNorm, never BatchNorm), get/set_flat_params
  core/client.py     Client.local_train -> ClientUpdate (a DELTA, never raw weights)
  core/server.py     Aggregator ABC, FedAvg, evaluate (loss/acc/macro-F1/per-class)
  core/runner.py     run(cfg), seeding, JSONL logging, pre_aggregate hook
  tests/             27 tests
  INVENTORY.md       Phase-0 ground truth about the existing BlockVerify code
  TODO.md            deferred items
```

---

## Phase 2 — Differential privacy (DP-SGD) and Table 1

Install: `pip install --break-system-packages opacus`

**Accounting is over each client's TOTAL local steps across ALL rounds**, not per round
(the classic DP-FL bug). σ is solved **once** per client and held fixed; each client owns
**one** `RDPAccountant` that persists across rounds:

```
steps_per_round_k = local_epochs * ceil(n_k / batch_size)
total_steps_k     = rounds * steps_per_round_k
sample_rate_k     = batch_size / n_k
sigma_k           = get_noise_multiplier(eps, delta, sample_rate_k, steps=total_steps_k, "rdp")
```

`--epsilon inf` (or omitting it) disables DP entirely — Opacus is never imported, and the
run is byte-identical to Phase 1 (enforced by a test).

```bash
# single DP run
python3 -m fedverify.core.runner --dataset mnist --num-clients 10 --rounds 30 \
        --alpha 0.5 --epsilon 2 --delta 1e-5 --max-grad-norm 1.0 --seed 0

# experiment 1 grid: eps {0.5,1,2,4,8,inf} x K {5,10} x {mnist,fmnist} x seed {0,1,2}
python3 -m fedverify.experiments.exp1_privacy_utility --dry-run   # list 72 cells
python3 -m fedverify.experiments.exp1_privacy_utility --smoke     # one tiny cell
python3 -m fedverify.experiments.exp1_privacy_utility             # full grid (resumable)
python3 -m fedverify.experiments.exp1_privacy_utility --dataset mnist --epsilon 2 inf

# tables (every number comes from a results file; missing cells -> "—" + stderr list)
python3 -m fedverify.analysis.make_tables
#   -> fedverify/results/tables/table1.{md,tex}   accuracy
#   -> fedverify/results/tables/table1b.{md,tex}  macro-F1
```

Each round records `privacy` per client:
`{sigma, C, sample_rate, steps, steps_per_round, steps_taken, realized_eps, target_eps,
delta, n_samples}`.

## Phase 3 — chain commitment layer

Every federated round is committed as a Merkle root over its client updates, so a client
can later prove its contribution was included and nobody — including this server — can
rewrite a round after the fact.

    delta -> canon_update (float32 LE, C-order)
          -> update_digest  = SHA256(compact-JSON header || raw bytes)
          -> leaf_i         = SHA256(JSON([i, client_id, digest]))
          -> round root     (+ leaf count, anchored together)
          -> anchor         (mock | local PoW chain | Algorand note txn)

The Merkle construction is **not reimplemented**: `fedverify/chain/commitment.py` imports
`compute_layer_merkle` / `merkle_inclusion_proof` from `backend/app.py`, the same code path
the browser already re-verifies for layer manifests. The leaf count is anchored beside the
root because duplicate-last trees are ambiguous without it.

The digest header binds `client_id`, `round` and `num_samples`, so a delta cannot be
replayed as another client or into another round; position is bound into the leaf, so
swapping two clients changes the root.

### Backends

| backend | I/O | txid | use |
|---|---|---|---|
| `mock` | none | SHA-256 of the committed content — deterministic | experiments |
| `local` | mines a block on `backend/blockchain.py` | block hash | tests, default demo |
| `algorand` | note txn, keys `frr`/`flc`/`rnd`/`run` | real txid | one confirmation run |

Bitcoin OP_RETURN checkpoints every `--checkpoint-every` rounds commit the Merkle root of
*all round roots so far*, pinning the whole lineage with one write. Without
`--btc-checkpoint` the checkpoint root is computed but anchored to a mock txid.

### Running it

    python3 -m fedverify.experiments.exp3_chain_overhead --smoke
    python3 -m fedverify.experiments.exp3_chain_overhead          # mock + local, K in {5,10,20}
    python3 -m fedverify.experiments.exp3_chain_overhead --algorand   # real testnet txns
    python3 -m fedverify.analysis.make_tables --only table3

exp3 runs cells **serially on purpose**: `anchor_ms` is a latency measurement and parallel
cells would contaminate it with scheduler queueing.

### Commitment is opt-in (amendment A6)

`--commit` is off by default. With it off the round record is byte-identical to Phase 2, so
Table-1 cells produced before and after Phase 3 stay comparable. With it on, each record
gains a `commit` block and leaves+proofs stream to `commitments.jsonl` beside
`rounds.jsonl`.

### API

Five additive routes on the existing Flask app (blueprint `fl_bp`):

    POST /api/fl/round/commit                    {run_id, round, leaves, root, leaf_count}
    GET  /api/fl/run/<run_id>                    metadata + every round root
    GET  /api/fl/round/<run_id>/<r>              root, leaf count, txid, client ids
    GET  /api/fl/proof/<run_id>/<r>/<client_id>  leaf, sibling path, root, txid
    GET  /api/fl/lineage/<run_id>                ordered roots + accepted/rejected per round

`POST` re-derives the root from the submitted leaves and rejects the commit on any
mismatch — the server does not take the caller's claimed root on trust.

## Phase 4 — attacks, robust baselines, FedVerify-Forensics

### Attacks

| attack | level | effect |
|---|---|---|
| `label_flip` | data | `y -> (C-1) - y` (exactly `9-y` on the 10-class sets) |
| `backdoor` | data | 3x3 white square bottom-right on `poison_frac` of the batch, relabelled to `backdoor_target` |
| `sign_flip` | delta | `delta -> -s * delta` |
| `gaussian` | delta | `delta -> N(0, sigma^2)` |
| `zero` | delta | `delta -> 0` (free-rider) |
| `scaling` | delta | `delta -> (K/f) * delta` (model replacement) |

Attacker identity comes from the seed alone, so the same clients are malicious for every
aggregator being compared. `--attack-start-round` keeps them dormant during a warm-up.
Attacks are applied BEFORE the Phase-3 commitment, so the chain binds what the attacker
actually sent and the accept/reject decision is auditable against it.

The backdoor trigger is written in **normalised** space — pure white is `(1-mean)/std`,
about 2.82 for MNIST. Stamping a literal 1.0 would be a grey smudge and would understate
the attack. ASR is measured on triggered test images whose TRUE class is not the target,
so target-class samples cannot count as free successes.

### FedVerify-Forensics

BlockVerify localises a tampered layer with a robust median/MAD outlier probe over the
WEIGHTS of a model. Forensics runs the same probe family one level up: the population is
the K client deltas instead of the D weights, and a Byzantine client is an outlier in that
population the same way a poisoned layer is an outlier in a model. The probes are
**imported** from `evaluation/eval_lib.py` — the byte-identical port of the deployed
in-browser detector — not re-implemented.

| score | statistic | direction |
|---|---|---|
| `s_norm` | robust z of `\|\|delta_k\|\|_2` across clients | two-sided (scaling inflates, zero collapses) |
| `s_dir` | robust z of `cos(delta_k, median delta)` | one-sided low |
| `s_coord` | robust z of the share of coordinates beyond `tau_coord` | one-sided high |
| `s_health` | NaN / Inf / `\|w\|>100` / entropy / constant-run | hard flag, rejects outright |

`combined = max(s_norm, s_dir, s_coord)`; reject if a hard flag fires or `combined > tau`.
Survivors are FedAvg'd. Sign-flip is the case that justifies the multi-probe design: it
leaves the norm untouched, so only `s_dir` sees it.

**tau is never hardcoded.** It comes from `results/calibration/taus.json`, and both the
value and its provenance (`tau_source`) are recorded in every run's `config.json`.
Constructing the aggregator without one raises. This is the same mistake
`docs/EVALUATION.md` §5.2 documents one level down: the hand-set layer threshold `z>8` had
FPR 0.49 on real heavy-tailed weights.

### Running it

    python3 -m fedverify.analysis.calibrate --smoke          # then without --smoke
    python3 -m fedverify.experiments.exp2_byzantine --smoke
    python3 -m fedverify.experiments.exp2_byzantine --jobs 8 --threads 2
    python3 -m fedverify.analysis.make_tables --only table2 table2b

Calibration MUST run before exp2: exp2 reads tau from its output and will not start
without it.

## Phase 5 — healthcare data, real heterogeneity, DP × Byzantine

### MIT-BIH Arrhythmia (`fedverify/datasets/mitbih.py`)

MNIST non-IID is *synthesised*: the skew is whatever Dirichlet alpha we chose. MIT-BIH's
skew is *inherited* — each client is a distinct set of patients, and arrhythmia prevalence
genuinely differs between them.

| choice | what | why |
|---|---|---|
| records | 48 minus paced 102/104/107/217 | AAMI EC57 excludes paced beats; keeping them fills class Q with a device artefact |
| split | de Chazal DS1/DS2, 22 records each | inter-patient. DS1 → hospitals, DS2 → test. A patient never crosses the boundary |
| beats | 256 samples centred on each R-peak | ≈0.71 s at 360 Hz |
| scaling | z-norm **per record** | gain and baseline differ between recordings |
| labels | AAMI 5-class N/S/V/F/Q | standard |
| model | `ECGCNN1D`, 75,461 params, GroupNorm | comparable to SmallCNN's 80,298 so Table 4 compares datasets, not capacity |

Measured: **100,693 beats, 44 patients**, class balance N 89.47% / S 2.76% / V 6.96% /
F 0.80% / Q 0.01%.

> **Macro-F1 is the primary metric.** A model that always answers N scores **89.47%
> accuracy and 18.89% macro-F1**. Accuracy on this dataset does not measure what it
> appears to. Table 4 prints macro-F1 first and accuracy in parentheses so the gap is
> impossible to miss — in the Phase-5 demo, MIT-BIH K=5 read 34.55 macro-F1 / 92.46
> accuracy.

The heterogeneity is real and checkable: at K=10, hospital 9 holds 18.8% ventricular beats
while hospital 4 holds 0.4% — a 47× spread, entirely from which patients they were given.

    python3 -c "from fedverify.datasets.mitbih import build_cache; build_cache()"
    python3 -m fedverify.experiments.exp4_heterogeneity --smoke

### Experiments

    python3 -m fedverify.experiments.exp4_heterogeneity --jobs 6 --threads 2   # table4
    python3 -m fedverify.experiments.exp5_dp_byzantine_interaction --jobs 6    # table6
    python3 -m fedverify.analysis.make_tables --only table4 table5 table6

exp5 asks whether DP noise blinds Byzantine screening. `tau` is looked up **per epsilon**
from calibration, because a threshold fitted on noiseless deltas would report DP as
breaking detection when only the threshold was stale.

Table 5 is qualitative and is generated from `fedverify/analysis/comparison.yaml`. Cells
that the cited paper does not state clearly are `?`, never a guess — 18 of 56 cells are `?`.
FedVerify's own "self-healing rollback" is `✗`: the lineage identifies a bad round, but
nothing undoes it.

## Phase 6 — frontend, docs, figures

### The Federated tab

`frontend/index.html` gains a **Federated** tab (additive; no existing tab touched):
run selector, per-round accuracy/macro-F1 line chart, a client × round accept/reject grid
(green accepted, red rejected), each round's Merkle root and anchoring transaction, and a
**Verify proof** button.

That button is the point. It fetches the inclusion path, **recomputes the leaf** from the
digest rather than trusting the served one, folds the sibling path in the browser, and
reads the anchored root **directly from the public Algorand indexer** — then reports
`TRUSTLESS ✓`, `ROOT MISMATCH`, or `SERVER-TRUSTED` with the reason. It reuses
`bvSha256Hex` / `bvMerkleLeaf` / `bvFoldProof` unchanged from the layer-manifest verifier,
because an FL round leaf uses the same encoding.

`test_browser_leaf_encoding_matches_the_backend` extracts those functions verbatim from
`index.html` and runs them under Node against the Python implementation, so the *shipped*
browser code is what is verified.

### Every command

```bash
# Experiments  (calibrate MUST precede exp2 and exp5 — they read tau from it)
python3 -m fedverify.experiments.exp1_privacy_utility           --jobs 8 --threads 2
python3 -m fedverify.analysis.calibrate
python3 -m fedverify.experiments.exp2_byzantine                 --jobs 8 --threads 2
python3 -m fedverify.experiments.exp3_chain_overhead
python3 -m fedverify.experiments.exp4_heterogeneity             --jobs 6 --threads 2
python3 -m fedverify.experiments.exp5_dp_byzantine_interaction  --jobs 6 --threads 2

# Everything else
python3 -m fedverify.analysis.make_tables      # tables 1,2,2b,3,4,5,6 → results/tables/
python3 -m fedverify.analysis.plots            # figures 1-6 (PNG+PDF) → results/figures/
python3 -m pytest backend/tests/ fedverify/tests/ -q
```

Add `--dry-run` to any experiment to see what is done vs pending, `--smoke` for a fast
single cell. All are resumable: a completed cell is skipped.

### Documentation

- [`fedverify/docs/WHAT_IS_FEDVERIFY.md`](docs/WHAT_IS_FEDVERIFY.md) — **start here.**
  Plain-language explanation: what the problem is, what we built, how to read the dashboard
- [`fedverify/docs/FEDERATED.md`](docs/FEDERATED.md) — threat model, commitment scheme,
  protocol, DP statement, forensics, guarantees **and non-guarantees**, limitations
- [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — where FedVerify sits
- [`docs/API_REFERENCE.md`](../docs/API_REFERENCE.md) — the six `/api/fl/*` routes
- [`docs/RELATED_WORK.md`](../docs/RELATED_WORK.md) §6 — federated / DP / blockchain-FL
- [`docs/EVALUATION.md`](../docs/EVALUATION.md) §7 — table and figure index
