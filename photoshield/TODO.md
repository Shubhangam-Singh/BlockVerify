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

## From P6 (experiments)
- **Low-information images.** E6 found that near-flat images (a binary silhouette's white corners)
  have an unstable pHash: JPEG 50 or a brightness shift flipped about half the bits. The verifier
  should flag such images ("perceptual match unreliable") instead of returning SIMILAR/UNRELATED,
  e.g. from the spread of the 63 AC coefficients around their median. Needs its own calibration.
- **Calibration corpus.** `photoshield/calibration.json` comes from scikit-image's sample photos
  (14 photos, 56 crops). Recalibrate on a larger public corpus (or the user's own photos via
  `--images DIR`) before treating near-match verdicts as more than indicative.
- **Real-model E1-E3** need the SD-VAE weights (huggingface.co is blocked in the container): run
  `python -m photoshield study e1 --model sdvae` and `e3` on the laptop, then `make_tables`.
