# FedVerify-Survival: federated learning with blockchain commitments for medical prediction

Federated learning across hospitals on two public datasets: UCI heart disease (four hospitals) and TCGA-BRCA
clinical data (five source sites, death indicator and survival time). It uses logistic regression (FedSGD, FedAvg,
DP-FedAvg), random forests and XGBoost (FedXGBBagging; AFT for survival). Every round's updates can be signed,
content-addressed, committed as a Merkle root and anchored on the existing BlockVerify proof-of-work chain. On top
of that, malicious hospitals are screened out. The study replicates Tables III–IV and Figs 5–10 of Myrzashova et al.
(IEEE T-SUSC 2025) on these datasets.

- New here? The complete guide, from first principles through Phases 0-11:
  [../docs/FEDVERIFY_SURVIVAL_GUIDE.md](../docs/FEDVERIFY_SURVIVAL_GUIDE.md)
- How it works and why: [DESIGN_NOTES.md](DESIGN_NOTES.md)
- Who is trusted for what, and what is not defended: [THREAT_MODEL.md](THREAT_MODEL.md)
- Results draft (generated): [paper/RESULTS_DRAFT.md](paper/RESULTS_DRAFT.md)
- Live demo, one command (`python3 -m fedverify.survival.demo`), and the 5-minute presenter script (generated):
  [DEMO.md](DEMO.md)
- Report diagrams (architecture, DFD levels 0 and 1, use case, classes, one round): [../../docs/diagrams/](../../docs/diagrams/)
- Deferred work and open decisions: [../TODO.md](../TODO.md)

## Setup

The command is `python3`. Everything runs on CPU.

```bash
pip install --break-system-packages --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install --break-system-packages -r fedverify/requirements.txt -r fedverify/survival/requirements.txt
```

Data (gitignored, not in the repository):

- `fedverify/data/heart/heart_disease_uci.csv`: the UCI heart-disease CSV, with the hospital in its `dataset` column
- `fedverify/data/tcga_brca/clinical.xlsx`: the TCGA-BRCA clinical workbook

Check the install: `python3 -m pytest backend/ fedverify/tests/ -q`

## Experiments: one command each

`python3 -m fedverify.survival.experiments <experiment> --jobs 4` runs one experiment, with `--jobs` runner
processes in parallel. Each process pins seed 0, 5 repeats and 2 torch threads, so the results do not depend on
`--jobs`. Add `--dry-run` to print the commands, or `--list` to count them.

| Experiment | What it runs | Writes |
|---|---|---|
| `methods` | local, centralized, FedSGD, FedAvg logistic; local, centralized, federated RF and XGBoost; XGBoost AFT on TCGA; the step-matched centralized XGBoost diagnostic row (`central_xgb_steps`, Phase 9); heart full-mode sensitivity | `fedverify/results/{heart,survival}/<cell>/seed0/` |
| `dp` | DP-FedAvg at ε 1, 2, 4, 8 and ∞, both datasets | same |
| `chain` | the commit-only `chain_*` methods (mock anchor) | same |
| `chain-pow` | one run per dataset mined on the proof-of-work chain; heart is profiled | `*_local/` cells, `fedverify/results/chain/profile.txt` |
| `calibration` | the screening thresholds | `fedverify/results/calibration/taus_survival.json` |
| `chain-attacks` | double spending, transaction malleability and endpoint compromise, 100 rounds, defences on and off | `fedverify/results/chain_attacks/` |
| `poisoning` | every `ours_*` cell: clean, and each attack with and without screening (needs `calibration` first) | `fedverify/results/{heart,survival}/ours_*` |
| `diagnostics` | why TCGA centralized XGBoost scores below local and federated: replays the XGBoost runs and checks the replay against them (Phase 8) | `fedverify/results/diagnostics/central_xgb_tcga.json` |
| `all` | all of the above, in that order | |

A single run, for example:

```bash
python3 -m fedverify.survival.runner --dataset heart --family xgb --method fedxgb_bagging --seed 0 --repeats 5 --torch-threads 2
python3 -m fedverify.survival.runner --dataset tcga --family lr --method ours_lr --attack sign_flip --seed 0 --repeats 5 --torch-threads 2
```

## Tables, figures and the results draft: one command

```bash
python3 -m fedverify.analysis.make_fl_healthcare_all
```

This command produces (Phase 11 adds the within-hospital columns, the paired comparisons and the report pack):

- `fedverify/results/tables/<name>.md` and `.tex` (the `.tex` files compile in an IEEEtran document)
- `fedverify/results/figures/<name>.png` (300 dpi), `.pdf` and `.csv` (the plotted numbers), plus `captions.md`
- `fedverify/survival/paper/RESULTS_DRAFT.md` and `RESULTS_DRAFT.comparisons.json` (every comparison the draft makes)
- `fedverify/results/showcase/index.html`: an interactive tour of the project and its results for demos (open it in a
  browser; it also works offline, with system fonts). It walks through one real committed round, from training to
  the mined block, and charts every result. Rebuild only this page with `python3 -m fedverify.analysis.make_showcase`.

Comparisons are paired: two methods on the same folds and repeats, a 95% patient-level bootstrap interval of their
difference (`analysis/paired_stats.py`, cached in `fedverify/results/statistics/`); a method "outperforms" another
only if that interval excludes 0. `paper/VERDICT_CHANGES.md` lists the verdicts that changed when this rule replaced
the earlier mean ± std one. Phase 12 adds three wording rules, applied by the generators everywhere (tables, draft,
page): a gain of "Ours" whose screen excludes too many honest hospital-rounds on a clean run is "not ranked"; the top
score is called the "highest score" and compared with its runner-up, never "best"; and a federated gain over
centralized training on TCGA says, in the same sentence, what the Phase 8 diagnosis found and how federated compares
with the step-matched row. `results/tables/paired_comparisons_within.md` gives the key comparisons within hospital.

The report diagrams (`docs/diagrams/`) are rendered to SVG and 300-dpi PNG with
`python3 -m fedverify.analysis.render_diagrams` (needs Node.js and Chrome or Chromium; run it after editing a `.mmd`).

Then, as separate steps:

```bash
python3 -m fedverify.analysis.latex_check --engine /path/to/tectonic   # compile every .tex in an IEEE document
python3 -m fedverify.analysis.make_report_pack                        # results/report_pack/ (also run by _all)
python3 -m fedverify.analysis.make_manifest                           # freeze: SHA-256 of every result, mined root
python3 -m fedverify.analysis.make_manifest --verify                  # are the frozen results intact?
```

Every number comes from a results file. [table.yaml](table.yaml) says which run fills which row. A missing result
renders as an em dash and is listed on stderr. `fedverify/results/` is gitignored, so after a fresh clone run the
experiments, then this command. The steps can also be run on their own:
`python3 -m fedverify.analysis.make_fl_healthcare_tables`, `make_fl_healthcare_figures`, `make_results_draft`.

## Results layout

```
fedverify/results/<exp>/<cell>/seed<base>/      exp: heart | survival (TCGA)
    config.json                                 every parameter, data provenance, attack and screening settings
    summary.json                                per repeat and mean ± std over repeats; detection for ours_* runs
    repeat<r>/oof.jsonl                         out-of-fold prediction of every patient
    repeat<r>/fold<k>/rounds.jsonl              one record per round (metrics, per-hospital metrics, diag, commit)
    repeat<r>/fold<k>/commitments.jsonl         chain_* and ours_*: Merkle leaves and inclusion proofs per round
    chain_blocks.jsonl, chain_metrics.json      *_local cells: the mined blocks and latency / throughput / send rate
```

Cell names: the method, then `_eps<ε>` (DP methods and `ours_lr`), `_aft` (survival task), `_full` (heart full mode),
`_local` (mined chain), `_<attack>` and `_noscreen` (poisoning runs).

## Reproducibility

- Repeat r uses seed base + r. The same seed and the same `--torch-threads` give byte-identical rounds, predictions
  and summaries (wall-clock keys such as `*_ms` and `*_wall_s` excluded). Different thread counts can differ in the
  last digits.
- No tau is ever typed in: screening runs read it from `taus_survival.json` and record it, with its source, in
  `config.json`.
- The table and figure generators write byte-identical files when run twice on the same results.

## Code map

| Module | Role |
|---|---|
| `runner.py` | runs one method over repeated stratified cross-validation |
| `data.py`, `heart.py`, `datasets.py` | data loading and data-independent encoding (TCGA-BRCA, heart) |
| `fl_lr.py`, `fl_rf.py`, `fl_xgb.py` | the three model families and their federated, commit and screened variants |
| `dp.py` | DP-SGD plumbing over the existing `core/privacy.py` |
| `serialize.py`, `chain_round.py` | canonical update bytes, signed submissions, Merkle commit, anchor, defences |
| `poison.py`, `probe.py`, `screening.py` | the malicious hospital, the public probe set, the tree screen, the tau file |
| `chain_attacks.py` | the reference paper's three chain attacks against the commit layer |
| `experiments.py` | every experiment, one command each |
| `demo.py` | the live demo: 3 narrated rounds on heart with a poisoned update, a replay, a wrong key and a tamper check |
| `../analysis/paired_stats.py` | within-hospital metrics and paired comparisons with a patient-level bootstrap |
| `../analysis/latex_check.py`, `make_report_pack.py`, `make_manifest.py` | LaTeX compile check, report pack, freeze |
| `../analysis/calibrate_survival.py` | screening calibration |
| `../analysis/diagnose_central_xgb.py` | the Phase 8 diagnosis of centralized XGBoost on TCGA |
| `../analysis/make_fl_healthcare_{tables,figures,all}.py`, `../analysis/make_results_draft.py` | tables, figures, draft |
