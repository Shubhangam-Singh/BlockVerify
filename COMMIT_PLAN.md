# Commit plan

For the human who commits (Claude never stages or commits; CLAUDE.md). Built on 2026-10-16 from the read-only
`git status --short --untracked-files=all` and `git diff --stat`. Every new or changed file in the working tree appears
below exactly once: in one commit, or in the list of files that must not be committed. Do not add a Co-authored-by
trailer to any message.

## How the branches fit together

The three branches are stacked: each one starts from the previous one, so every commit sees the code it needs.
All changes are in the working tree now, and creating a branch keeps them.

```bash
git checkout -b feature/backend-enhancements      # from main, with everything still uncommitted
#   commits 1 to 14
git checkout -b feature/frontend-overhaul         # from feature/backend-enhancements
#   commit 15
git checkout -b docs/update-references            # from feature/frontend-overhaul
#   commits 16 to 19
```

Open the pull requests (or merge) in the same order: backend, then frontend, then docs.

## Branch 1: `feature/backend-enhancements`

| # | Message | Files |
|---|---|---|
| 1 | `fedverify/chain: content IDs and Ed25519 signing` | `fedverify/chain/contentid.py`, `fedverify/chain/signing.py` |
| 2 | `fedverify/survival: datasets, config and metrics` | `fedverify/survival/__init__.py`, `config.py`, `data.py`, `datasets.py`, `heart.py`, `heart_vocab.json`, `vocab.json`, `common.py`, `metrics.py`, `metrics.yaml`, `phase0_check.py`, `requirements.txt` |
| 3 | `fedverify/survival: models, FL methods, DP and runner` | `fedverify/survival/models.py`, `fl_lr.py`, `fl_rf.py`, `fl_xgb.py`, `dp.py`, `runner.py`, `experiments.py` |
| 4 | `fedverify/survival: per-update chain commits and chain attacks` | `fedverify/survival/serialize.py`, `chain_round.py`, `chain_attacks.py` |
| 5 | `fedverify/survival: poisoning attacks and screening` | `fedverify/survival/poison.py`, `probe.py`, `screening.py` |
| 6 | `fedverify/survival: live demo` | `fedverify/survival/demo.py` |
| 7 | `fedverify/chain: push survival runs to the dashboard` | `fedverify/chain/push.py` (modified, additive) |
| 8 | `fedverify/analysis: tables, figures and paired statistics` | `fedverify/analysis/make_fl_healthcare_tables.py`, `make_fl_healthcare_figures.py`, `paired_stats.py`, `calibrate_survival.py`, `diagnose_central_xgb.py`; `fedverify/survival/table.yaml`; `fedverify/survival/ieee_table_iii.csv`, `ieee_table_iv.csv`, `ref/ieee_table_iii.csv`, `ref/ieee_table_iv.csv`, `ref/SPOT_CHECK.md` |
| 9 | `fedverify/analysis: results draft, showcase page, demo script` | `fedverify/analysis/make_results_draft.py`, `make_showcase.py`, `showcase_template.html`, `make_demo_md.py`, `make_fl_healthcare_all.py` |
| 10 | `fedverify/analysis: LaTeX check, report pack, manifest, diagrams` | `fedverify/analysis/latex_check.py`, `make_report_pack.py`, `make_manifest.py`, `render_diagrams.py` |
| 11 | `fedverify/analysis: escape MNIST LaTeX, mark tables never run` | `fedverify/analysis/make_tables.py` (modified, additive) |
| 12 | `docs/diagrams: report diagrams with SVG and PNG renders` | `docs/diagrams/README.md`, `rendered.json`, and for each of `architecture`, `class_diagram`, `dfd_level0`, `dfd_level1`, `sequence_round`, `use_case`: `.mmd`, `.svg`, `.png` |
| 13 | `fedverify/tests: tests for the survival study` | `fedverify/tests/test_survival_attacks.py`, `test_survival_chain.py`, `test_survival_data.py`, `test_survival_heart.py`, `test_survival_methods.py`, `test_survival_metrics.py`, `test_survival_privacy.py`, `test_survival_runner.py`, `test_survival_xgb.py`, `test_fl_healthcare_outputs.py`, `test_results_draft.py`, `test_diagnose_central_xgb.py`, `test_phase9_references.py`, `test_demo.py`, `test_diagrams.py`, `test_phase11_stats.py`, `test_phase11_freeze.py`, `test_phase12_wording.py` |
| 14 | `backend: ignore placeholder BTC_WIF values` | `backend/bitcoin_client.py` (modified) |

Notes:
- Commit 12 is a docs folder but sits on this branch on purpose: `test_diagrams.py` and `test_phase12_wording.py`
  (commit 13) check the diagram sources and their renders, and would fail on a checkout without them.
- Commit 14 was not made in the FedVerify phases (it was already in the working tree). It makes the wallet fall back
  to the demo testnet key when `BTC_WIF` holds the `.env` template placeholder. Read `git diff backend/bitcoin_client.py`
  before committing it.

Commands for the first commit (the others follow the same pattern). In the tables, a bare file name lives in the
folder of the last full path before it in the same row.

```bash
git add fedverify/chain/contentid.py fedverify/chain/signing.py
git commit -m "fedverify/chain: content IDs and Ed25519 signing"
```

## Branch 2: `feature/frontend-overhaul`

| # | Message | Files |
|---|---|---|
| 15 | `frontend: dashboard overhaul and survival runs in the Federated tab` | `frontend/index.html` (modified) |

The diff is large (1029 lines added, 380 removed). It holds the earlier dashboard work that was already in the
working tree plus the additive Federated-tab changes from Phase 10 (labels for the new attacks and aggregators,
per-round ground truth, the run banner). Review it with `git diff frontend/index.html`; to keep the two apart, split it
with `git add -p frontend/index.html` into two commits, the Federated-tab hunks with the message
`frontend: show survival runs in the Federated tab`.

## Branch 3: `docs/update-references`

| # | Message | Files |
|---|---|---|
| 16 | `fedverify/survival: README, design notes, threat model, inventory` | `fedverify/survival/README.md`, `DESIGN_NOTES.md`, `THREAT_MODEL.md`, `INVENTORY.md` |
| 17 | `fedverify/survival: generated results draft and demo script` | `fedverify/survival/paper/RESULTS_DRAFT.md`, `RESULTS_DRAFT.comparisons.json`, `VERDICT_CHANGES.md`; `fedverify/survival/DEMO.md` |
| 18 | `fedverify/docs: complete guides for both studies` | `fedverify/docs/FEDVERIFY_COMPLETE.md`, `fedverify/docs/FEDVERIFY_SURVIVAL_GUIDE.md` |
| 19 | `fedverify: TODO through Phase 12` | `fedverify/TODO.md` (modified) |

Commit 17's files are generated (`python3 -m fedverify.analysis.make_fl_healthcare_all`); if the results change,
regenerate them before committing rather than editing them.

## Never commit

| Path | Why |
|---|---|
| `clinical_project_tcga_brca_2024-04-08_converted.xlsx` (repository root) | TCGA-BRCA clinical data: datasets stay out of git. Not gitignored, so leave it unstaged (or move it into `fedverify/data/`). |
| `fedverify/data/` | Datasets (MNIST, Fashion-MNIST, MIT-BIH, heart, TCGA-BRCA). Gitignored. |
| `fedverify/results/` | Every run, table, figure, the report pack, `MANIFEST.json` and the cached statistics (1.1 GB). Gitignored; rebuilt from the runs. |
| `backend/data/` | Keys and live state: `secret.key`, `algo_wallet.json`, `algo_app.json`, `users.json`, `fl_runs.json`, `models_registry.json`, `verification_logs.json`. Gitignored. |
| `backend/.env` | Secrets. Gitignored (commit only `.env.example`, which is already tracked). |
| `evaluation/checkpoints/` | Model weights. Gitignored. |
| `__pycache__/`, `.pytest_cache/` | Python and pytest caches. Gitignored. |
| `COMMIT_PLAN.md` | This note. Delete it once the commits are made. |
| Claude's scratch folder (outside the repository: Tectonic binary, test renders, screenshots) | Not in the working tree; nothing to do. |

Already tracked and unchanged, so not part of any commit: `backend/=2.8.0` and `backend/=4.1.0`, empty files left by an
earlier `pip install pkg>=x` typed without quotes. Removing them is a separate clean-up.

## Before pushing

- On `docs/update-references` (all files present): `python3 -m pytest backend/ fedverify/tests/ -q`.
- `git status --short` should then list only `clinical_project_tcga_brca_2024-04-08_converted.xlsx` and `COMMIT_PLAN.md`.
- `git log --format=%B main..docs/update-references | grep -i co-authored` should print nothing.
