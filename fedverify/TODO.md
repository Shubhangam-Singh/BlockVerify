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
