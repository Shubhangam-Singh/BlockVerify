# PhotoShield: deferred items

Parked here per CLAUDE.md "Scope" and A13. Nothing here is in scope for the current phase.

## Found at P0 (pre-existing, outside PhotoShield scope)
- [ ] `backend/app.py:736` and `:1156` read `payload.get("username")` from the JWT, but tokens carry
      the user in `sub` (`backend/auth.py:79-84`). Owners therefore never see their own private
      models in `/api/search` and `/api/registry`. `photo_routes.py` must read `sub`.
- [ ] `get_transaction_ledger` (`backend/algorand_client.py:377-388`) labels every JSON note as a
      model "registration". FedVerify notes (`frr`) are already shown as blank model rows. P4 adds
      a branch for photo notes only (`ps`); fixing FL rows changes existing output and is left to
      the human.

## Deferred by design (PHOTOSHIELD_DESIGN.md)
- [ ] BK-tree or multi-index hashing for near-match lookup beyond ~100k photos (§5.2).
- [ ] Server-side backup of hash-only pHashes (`PHOTOSHIELD_BACKUP_URL`) for hosts with a
      persistent store (§6.9). Today the owner's manifest is the backup (`/rehydrate`).
- [ ] Owner key rotation and revocation (§9.2).
- [ ] PDQ (Meta's 256-bit hash) in addition to the 64-bit pHash. It needs a byte-identical JS port
      before it can go on chain; experiments may use it earlier (E6).
- [ ] Contract global-state anchoring for albums under a namespaced key (`p:<id>`), only if the
      TEAL app is redeployed with more slots (§6.1).
