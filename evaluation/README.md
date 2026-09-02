# BlockVerify — Evaluation Artifact

Reproduces the real-checkpoint evaluation in [`../docs/EVALUATION.md`](../docs/EVALUATION.md).

```bash
pip install -r requirements.txt
./download_checkpoints.sh        # ~65 MB of real HF safetensors (gitignored)
python3 evaluate.py              # → out/results.json + out/*.png   (seeded, deterministic)
python3 cross_validate.py        # verifies the Python probes == deployed JS (needs node)
```

- `eval_lib.py`      — safetensors reader + faithful Python port of the deployed probes + attacks + ROC
- `evaluate.py`      — ROC/AUC, threshold calibration, (Δ,k) heatmap, throughput, cost
- `cross_validate.py`— byte-identical check of the port against `frontend/index.html` (Node)
