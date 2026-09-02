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
