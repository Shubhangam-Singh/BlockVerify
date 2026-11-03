# PhotoShield threat model

Scope: the provenance layer (fingerprints, owner signatures, Algorand anchors, the verify API and
the ledger rebuild) and the cloak (PGD + EOT against one image encoder). Design references are to
[`PHOTOSHIELD_DESIGN.md`](../../PHOTOSHIELD_DESIGN.md).

## Assets

| Asset | Why it matters |
|---|---|
| A registration (fingerprints + owner key + time) | The evidence a user points to when a fake appears |
| The owner's Ed25519 private key | Whoever holds it can claim the registrations |
| The photo's pixels | Private by default: never uploaded |
| The registry (`backend/data/photo_registry.json`) | Convenience index only; the ledger is the source of truth |
| Calibration thresholds | They decide what counts as "derived" |

## Trust assumptions

- Algorand Testnet's history is not rewritten. Anchors are 0-ALGO self-payments with a note; no
  contract state is involved (§6.7).
- The browser running the tab is not compromised (it holds the key and computes the fingerprints).
- SHA-256 and Ed25519 are sound.
- The server is **not** trusted for the facts a verdict rests on: the tab re-checks an anchor
  against the public indexer, and the ledger rebuild accepts only notes whose owner signature
  verifies (§6.8).

## Adversaries and what stops them

| # | Adversary / attack | Mitigation | Residual risk |
|---|---|---|---|
| A1 | **Forger** edits a registered photo and claims the edit is the original | The edit's SHA-256 differs; at best it matches "Derived/Similar" and links to the *real* registration, which is older | None for the exact claim |
| A2 | **Squatter** registers your public photo before you do | Nothing technical: the ledger records first registration, not authorship | Real. The UI says "register before posting" (§12) |
| A3 | **Impersonator** registers under someone else's name | The chain holds a public key, never a username (§5.3). Linking a key to an account needs a signed challenge (`/api/photo/prove`) | A user who loses their key cannot prove control |
| A4 | **Malicious or compromised server** fabricates or alters registrations | Every anchor carries the owner's signature over the manifest digest; the browser checks the transaction on the public indexer; the rebuild drops notes with bad signatures or from unknown senders | It can refuse service or hide entries; it cannot forge a valid owner signature |
| A5 | **Replay** of a signature on another photo or context | Domain-separated messages: `photoshield/register@1\|`, `album@1\|`, `prove@1\|` (§5.4); challenges are random, single-use and expire | — |
| A6 | **Registry probing** (is this photo registered? by whom?) | Per-IP rate limit on `/api/photo/verify`; near matches never return a username, only a key fingerprint (§6.3); hash-only mode keeps the pHash off chain | Exact-hash lookups are possible by design; that is what verification is |
| A7 | **pHash evasion**: an adversary edits until the pHash moves past τ_far | None at this layer. pHash is not adversarially robust (Jain et al., USENIX Security 2022) | Real. A perceptual match is evidence, not proof; "Unknown" does not mean "original" |
| A8 | **pHash collision**: an unrelated photo lands within τ_near | τ_near is set from measured unrelated pairs at a target false-positive rate (E6), never hard-coded; without a calibration file near matching is disabled | Small but non-zero; flat or low-detail images are less reliable (TODO.md) |
| A9 | **Cloak removal**: JPEG, resizing, denoising, diffusion purification | EOT during the attack; E3/E6p measure what survives | Real. The cloak raises cost, it is not a guarantee |
| A10 | **Closed or different models** | None: the cloak needs encoder gradients | Real. No protection against closed editors; transfer measured in E5 |
| A11 | **Spam / registry flooding** | Login required to register; proof-of-work per registration; per-user rate limit; album size cap | A determined attacker with many accounts can still add noise |
| A12 | **Server restart wipes the registry** (Render free tier) | Rebuild from the ledger by paging the indexer (§6.8); deterministic photo ids; manifest re-attach for hash-only photos (§6.9) | Hash-only photos are exact-match only until re-attached |
| A13 | **Key theft** from an exported backup | Backups are passphrase-encrypted by default (PBKDF2 + AES-GCM); a plain export needs an explicit acknowledgement | A weak passphrase is guessable offline |
| A14 | **Malformed manifests / notes** (floats, oversize, injection in titles) | Strict schema validation, no floats, size caps, title character filter; notes are size-checked before sending | — |
| A15 | **Deepfake from your photo**: a face or background is swapped, so the whole-image match fails | Origin reports: the committed region map, alignment and per-region verdicts show which regions still match the anchored photo (design §15) | Global edits change every region; extreme crops or perspective changes can defeat alignment |
| A16 | **Forged origin report**: claim regions matched, or move tiles around | Each tile carries a Merkle proof to the root inside the anchored manifest (positions are in the leaves); statuses must follow from the hashes and the calibrated thresholds; the owner key signs the report; the verifier checks the anchored digest and key | A signer can choose *which* suspect to report on; the suspect's SHA-256 binds the report to one specific copy |
| A17 | **Detail-preserving or low-frequency edits** that keep a tile's coarse look | Tiles commit a detail level as well as a pHash, so smoothing and inpainting fills shift it (E8 ablation) | Edits that keep both layout and detail statistics can still pass as matches. The report says "evidence, not proof" |
| A18 | **Probing the region map** | Only the root is public; tile values are revealed only in reports the owner chooses to publish | A published report reveals coarse per-tile hashes of that photo |

## Out of scope

- Stopping a fake from spreading, or making platforms check provenance.
- Proving who *took* a photo (camera-level provenance such as C2PA signing at capture).
- Protection against closed generative models.
- Key rotation and revocation (`photoshield/TODO.md`).

## Responsible use

The cloak is a defensive perturbation for photos the user owns. It does not attack any service.
PhotoShield does not generate, store or distribute images of anyone.
