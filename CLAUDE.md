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
