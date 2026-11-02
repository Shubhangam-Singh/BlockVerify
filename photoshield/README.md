# PhotoShield

Photo provenance and AI-edit cloaking on BlockVerify's integrity layer: the third application of
the same anchoring core, after AI models (BlockVerify) and federated updates (FedVerify).

- **Plain-language guide:** [`docs/WHAT_IS_PHOTOSHIELD.md`](docs/WHAT_IS_PHOTOSHIELD.md)
- **Threat model:** [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md)
- **Design spec:** [`../PHOTOSHIELD_DESIGN.md`](../PHOTOSHIELD_DESIGN.md)
- **API:** [`../docs/API_REFERENCE.md`](../docs/API_REFERENCE.md#photo-shield-apiphoto)
- **Reuse map and baselines:** [`INVENTORY.md`](INVENTORY.md) · **Deferred work:** [`TODO.md`](TODO.md)

## Layout

```
photoshield/
├── provenance/   hashing.py (SHA-256, canonical pHash), manifest.py, signing.py (Ed25519), merkle.py
├── cloak/        encoders.py (sdvae, sdxlvae, resnet50, facenet, tiny), pgd.py, eot.py, metrics.py, protect.py
├── chain/        notes.py (≤1024-byte Algorand notes), anchor.py (mock/local/algorand), ledger_sync.py, client.py
├── experiments/  study.py (E1–E3, E6, E7), edit_test.py + colab_edit_test.ipynb (E4, E5)
├── analysis/     make_tables.py (every table number comes from results files)
├── calibration.json   pHash thresholds, written by `study e6 --install`, read by the backend
├── cli.py        python -m photoshield ...
└── tests/        pytest suite, PNG vectors, Node pHash/crypto checks, Playwright e2e
```

Backend integration is `backend/photo_routes.py` (a Blueprint) plus additive lines in
`backend/app.py` and `backend/algorand_client.py`. The UI is the **Photo Shield** tab in
`frontend/index.html`.

## Install

From the repository root, on the laptop (CPU-only torch, CLAUDE.md A2):

```bash
pip install --break-system-packages --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

```bash
pip install --break-system-packages -r photoshield/requirements.txt
```

The provenance layer and the backend routes need only the standard library plus PyNaCl. torch is
needed only to cloak and for experiments.

## Use

Cloak a photo locally (nothing is uploaded):

```bash
python -m photoshield protect photo.jpg --out shielded/ --eps 8 --steps 40
```

Then open the **Photo Shield** tab, choose "Cloaked photo", and drop `shielded/manifest.json`, the
original and `shielded/cloaked.png`. The page re-hashes both images, checks them against the
manifest, signs with your browser key and anchors. For provenance only, choose "Provenance only"
and drop any photo.

Other commands:

```bash
python -m photoshield hash photo.jpg
```

```bash
python -m photoshield compare a.jpg b.jpg
```

```bash
python -m photoshield check shielded/manifest.json photo.jpg shielded/cloaked.png
```

Register from the command line instead of the browser (the key file is a CLI key; keep it safe):

```bash
python -m photoshield keygen --out my.pskey
```

```bash
python -m photoshield register shielded/manifest.json --api https://blockverify-api-0ojc.onrender.com/api --token <JWT> --key my.pskey
```

## Run the backend locally

```bash
PHOTOSHIELD_ANCHOR=local python backend/app.py
```

`local` anchors on BlockVerify's in-memory PoW chain, so no wallet is needed. The default,
`algorand`, needs `ALGO_MNEMONIC` or `ALGO_PRIVATE_KEY`. All `PHOTOSHIELD_*` variables are listed
in the design doc §6.11.

## Experiments

```bash
python -m photoshield study e6 --install
```

```bash
python -m photoshield study e1 --model sdvae --eps 2,4,8,12,16
```

```bash
python -m photoshield study e3 --model sdvae --eps 8
```

```bash
python -m photoshield study e7
```

```bash
python -m photoshield.analysis.make_tables
```

- E4 (InstructPix2Pix) and E5 (transfer) need a GPU: open
  `experiments/colab_edit_test.ipynb` in Colab and unzip its results into `photoshield/results/`.
- `--model tiny` uses a random-weight encoder. Its rows are marked *pipeline check* and are not
  findings.
- Every run takes `--seed` and `--threads` and writes
  `photoshield/results/<exp>/<cell>/seed<N>/{config.json, records.jsonl}`.
- Tables are generated into `photoshield/results/TABLES.md`. Never type a number into a table.

## Tests

```bash
python3 -m pytest photoshield/tests/ -q
```

```bash
python3 -m pytest backend/tests/ -q
```

The PhotoShield suite checks, among other things:
- Python and the shipped browser JS give bit-identical pHashes on PNG vectors (run under Node).
- Canonical manifest digests, photo ids and Ed25519 signatures interoperate between browser and server.
- Ledger rebuild paging and rejection rules.
- Route behaviour.
- Experiment determinism: same seed ⇒ identical records once timing keys are stripped.

`tests/e2e/` has a Playwright flow for the tab against a local backend.

## Honest limits

- The cloak targets one encoder and gives no protection against closed models. Re-encoding,
  resizing, denoising and diffusion purification weaken or remove it.
- Provenance proves that fingerprints existed at a time, under a key. It does not stop a fake from
  spreading.
- A perceptual match is evidence, not proof. First-to-register is not authorship.
- Losing the browser key loses the ability to prove control: export a backup.
- Use it on photos you own.
