# Standing rules

## Git
- NEVER run git add / commit / push / checkout / stash / rebase.
- Only edit files. The human commits manually with their own identity.
- Never write a Co-authored-by trailer anywhere, in any file.

## Existing code
- backend/, frontend/, evaluation/, docs/ are the existing BlockVerify system.
- All new FL work lives in a new top-level fedverify/ directory.
- You may IMPORT from backend/. Do not fork, copy, or rewrite existing modules.
  If you need to change an existing file, the change must be strictly additive
  (new route, new transaction type, new tab) and must not alter existing behaviour.
- Run `pytest backend/ -q` before declaring any phase complete. 82+ must pass.

## Determinism
- Every experiment takes --seed. Seed random, numpy, torch (cpu + cuda).
- torch.use_deterministic_algorithms(True) wherever it does not throw.
- Every run writes results/<exp>/<cell>/<seed>/rounds.jsonl and config.json
  containing EVERY parameter that influenced the run, including git-independent
  things like the tau read from calibration.
- Same seed twice => byte-identical rounds.jsonl. This is tested.

## Numbers
- Never type a number into a table, doc, or README. Every number in every table
  is emitted by fedverify/analysis/make_tables.py from a results file.
- Missing cells render as "—" and are listed on stderr.

## Modelling
- No BatchNorm anywhere. Use GroupNorm. (Opacus cannot handle BatchNorm and
  BatchNorm running statistics leak across FL clients.)
- Everything downstream of local training operates on DELTAS
  (local_params − global_params), never on raw weights.

## Scope
- Do only what the current phase asks. Ideas for later go in fedverify/TODO.md.
- At the end of each phase print the completion block (format in §9) and STOP.

---

# Amendments (agreed with the human, Phase 0)

These refine the rules above. Where they conflict, the amendment wins.

## A1. Determinism vs. wall-clock timings
The rule "same seed twice => byte-identical rounds.jsonl" is unsatisfiable as
literally written, because the Phase-1 record also contains wall-clock timings
(`train_wall_s`, `round_wall_s`, `wall_time_s`) which vary run to run.

**Agreed resolution:** timings STAY in `rounds.jsonl` (Table 3 needs them), and the
determinism test strips timing keys before comparing. Everything that influences the
science must still be byte-identical for a fixed seed.

Timing keys excluded from the determinism comparison:
`train_wall_s`, `round_wall_s`, `wall_time_s`, `anchor_ms`, `digest_ms`, `merkle_ms`,
`aggregate_ms`, and any other key ending in `_ms` or `_wall_s`.

## A2. PyTorch installation (Phase 1)
Install CPU-only wheels:

    pip install --break-system-packages --index-url https://download.pytorch.org/whl/cpu torch torchvision

Do NOT `pip install torch` from plain PyPI: on linux-x86_64 it declares five NVIDIA
CUDA dependencies (~4 GB) and this machine has ~2.4 GB free (97% full). The CPU-only
index is reachable (HTTP 200) and installs ~1 GB. Python is PEP-668
externally-managed, so `--break-system-packages` (or a venv) is required.

## A3. Corrected test baseline
The rule says "82+ must pass". The real current baseline is **99 passing, 0 skipped**.
The working invocation is:

    python3 -m pytest backend/tests/ -q

Use 99 as the regression baseline; a drop below 99 is a regression even though it
still satisfies "82+".

## A4. Git attribution
This session is configured to add a Co-authored-by trailer to commits. Rule 1 above
forbids me from ever creating a commit, so no trailer is ever written by me. The
human commits under their own identity. There is no conflict in practice.

## A5. Determinism is per-thread-count
Measured: the same seed at OMP/torch threads=2 vs 16 gives different results
(MNIST K=5, 1 round: acc 0.9413 vs 0.9404). Intra-op parallelism changes
floating-point reduction order; this is inherent, not a bug.

**Agreed resolution:** "same seed => byte-identical rounds.jsonl" holds at a
FIXED thread count. Every run records `torch_threads_effective` in config.json,
and `FLConfig.torch_threads` / `--torch-threads` pins it. Run a whole experiment
grid with one `--threads` value so its cells are mutually comparable.

## A6. Phase-3 chain commitment is opt-in
The Phase-3 spec puts the commitment step unconditionally in the round loop with
`chain_backend="mock"` as the experiment default. Taken literally that would add
`root`/`txid`/`anchor_ms` keys to EVERY `rounds.jsonl` record, including Phase-1/2 cells —
changing the record shape of Table-1 cells mid-grid and making cells run before and after
Phase 3 non-comparable. (This was not hypothetical: the exp1 grid was running when Phase 3
landed.)

**Agreed resolution:** commitment is gated on a new `FLConfig.commit` flag, default
`False`. When off, the round record is byte-identical to Phase 2 and no chain code is
imported. `exp3_chain_overhead` sets `commit=True`; so do the Phase-3 tests. Verified: a
commit-off run is byte-identical (after `strip_timings`) to the same run produced before
Phase 3 existed.

Consequence: `config.json` for cells run before Phase 3 has no `commit`/`btc_checkpoint`
key. The default is the behaviour those cells had, so the record stays complete in effect.

## A7. Algorand App ID
The master prompt says App `758544892`. The real deployed App ID is **764828342**
(`backend/data/algo_app.json`), as recorded in `fedverify/INVENTORY.md` §1. Phase 3
follows the real code. `algorand_client.broadcast_fl_round` uses the module's existing
client/wallet, so it targets whatever App ID the repo actually has.

## A8. Tables must ignore incomplete cells
`make_tables.load_runs` originally took the last line of any `rounds.jsonl` it found. An
interrupted cell would then be averaged into Table 1 as though it had finished (measured:
a 2-of-30-round orphan contributed `acc=0.6057` beside real 30-round cells). It now skips
any cell whose last record is below `config.rounds` and prints `[skip] incomplete run ...`
on stderr, exactly like a missing cell.

## A9. Phase-4 record shape and the exp1 grid
Phase 4 landed while the exp1 grid was still running, so the same constraint as A6
applies: a no-attack `fedavg` cell must keep its exact Phase-1/2 record shape.

Two violations were caught by an explicit byte-comparison guard and fixed:
- adding `"detector"` to the shared `Aggregator._empty_diag()` changed `diag` in EVERY
  record. `detects` is now a class attribute read from the registry, never serialised.
- the `"attack"` block is emitted only when an attack is configured OR the aggregator is
  a detector, so clean `fedavg` cells emit nothing.

`fedverify/tests/test_aggregators.py::test_phase1_record_shape_is_unchanged_by_phase4`
pins this. Verified: an exp1-style run is byte-identical (after `strip_timings`) to a
baseline captured before Phase 3 existed.

## A10. evaluation/eval_lib.py needed a NumPy-2 fix
`eval_lib.auc` called `np.trapz`, removed in NumPy 2.0. On the installed NumPy 2.5.2 the
existing layer-level ROC evaluation was therefore already broken — the AUC numbers in
docs/EVALUATION.md could not be regenerated. Phase 4 needs that same ROC for client-level
calibration, so `auc` now binds `np.trapezoid` when present and falls back to `np.trapz`.
Two lines, behaviour-preserving, verified against known AUCs (1.0 and 0.75). This is a
repair of existing behaviour, not a change to it.

## A11. Krum is a selection rule, not a detector
Vanilla Krum keeps ONE client, so a naive reading of its `rejected` list says it "detects"
K-1 malicious clients every round, which would put a meaningless F1 in Table 2b. Each
aggregator now declares `detects`: True for `forensics` and `multikrum` (which exclude a
specific set), False for `krum` (top-1 selection), `median` and `trimmed_mean` (no
client-level exclusion at all). Table 2b renders non-detectors as `n/a`, distinct from a
missing cell `—`.
