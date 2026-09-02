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
