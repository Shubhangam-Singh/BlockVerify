# What is PhotoShield?

*A plain-language guide. The engineering spec is [`PHOTOSHIELD_DESIGN.md`](../../PHOTOSHIELD_DESIGN.md);
the attacker analysis is [`THREAT_MODEL.md`](THREAT_MODEL.md).*

## The problem

You post a photo of yourself. Someone downloads it and feeds it to an AI editor ("put this person
in a different outfit", "make it look like they were at that place"). Two things would help:

1. **Making that edit harder**, at least with the open-source models most such tools are built on.
2. **Being able to prove later** that the photo is yours and existed, unedited, at a given time,
   so you can point to the original when a fake appears.

PhotoShield does both, and is honest about how far each one goes.

## Layer 1: provenance (the part that always works)

When you register a photo in the **Photo Shield** tab of BlockVerify:

- Your browser computes two fingerprints of the photo:
  - **SHA-256**, an exact fingerprint. Change one pixel and it changes completely.
  - **pHash**, a *perceptual* fingerprint. It stays the same, or nearly the same, when the photo
    is re-compressed, resized or slightly brightened, as happens every time it is shared.
- Your browser **signs** those fingerprints with a key that only it holds (an Ed25519 key, the
  kind used by SSH and Signal).
- The server writes the fingerprints, your signature and your public key into a transaction on
  the **Algorand Testnet**, a public blockchain. Nobody, including us, can change or back-date
  it afterwards.

**The photo itself never leaves your computer.** Only the fingerprints do.

Later, anyone can drop a copy of a photo into **Check a photo**:

| Verdict | What it means |
|---|---|
| Exact original / exact protected copy | Byte-for-byte the registered file |
| Derived | Very likely a re-compressed, resized or lightly edited copy of a registered photo |
| Similar | Possibly derived: a heavier edit or crop, or a different but similar-looking photo |
| Unknown | Not in the registry |

Every verdict links to the transaction on the public ledger, and the page can check that
transaction **itself, against Algorand's public indexer**, so you don't have to trust our server.

The "Derived" and "Similar" thresholds are not guessed. They are measured
(`python -m photoshield study e6`) so that unrelated photos are wrongly matched only rarely, and
the measured rates are printed in `photoshield/results/TABLES.md`. A perceptual match is still
**evidence, not proof**.

## Layer 2: the cloak (the part that sometimes works)

The cloak is an optional step you run **on your own computer**:

```bash
python -m photoshield protect photo.jpg --out shielded/ --eps 8 --steps 40
```

It adds a tiny, carefully computed change to the pixels (at most a few levels out of 255 per
pixel; you can't see it) that confuses the image encoder inside Stable Diffusion. Editors built on
that encoder then tend to produce damaged or off-target edits of the cloaked photo. This is the
technique from PhotoGuard (Salman et al., ICML 2023), made robust to resizing, blurring and JPEG
by training against those transformations (EOT).

You get `cloaked.png` (post this one) and `manifest.json` (fingerprints of both images plus the
cloak settings). Register them together in the tab, choosing "Cloaked photo".

**What the cloak does not do:**
- It does nothing against closed models (Gemini, Grok, Midjourney, ...). Nobody outside those
  companies can compute the gradients it needs.
- Heavy re-compression, resizing, denoising or "purification" with a diffusion model can wash it
  out. The experiments (E3, E6p) measure how much survives.
- It is crafted against one encoder; whether it carries over to others is measured (E5), not assumed.

That is why provenance is the core and the cloak is the extra: **even when the cloak is broken,
the purified copy still matches your registration** (E6p measures exactly this).

## Layer 3: origin reports ("was this fake made from my photo?")

A deepfake changes *part* of a photo: a swapped face, a new background, an inpainted body. Its
fingerprint no longer matches yours, so a plain check says "Unknown". That's exactly when you need
evidence most.

When you register a photo, PhotoShield also commits a **region map**:
- The photo is split into a grid of regions (an 8 × 8 grid for normal photos).
- Each region gets its own fingerprint, plus a "detail level" that notices smoothing.
- Only one summary hash of the whole grid (a Merkle root) goes into your anchored record.

Later, the **Origin report** card compares your original with a suspect copy, region by region:
- **Green** regions are still identical to what you anchored.
- **Red** regions were changed.
- If enough regions match, the verdict is **Derived from your registered photo**.

The report you download is signed with your key and carries a proof for every region. Anyone can
check it, in the tab or with `python -m photoshield origin-verify`, without ever seeing your
original.

- **Copies that keep the framing** are handled in the browser.
- **Cropped or rotated copies:** run
  `python -m photoshield origin original.jpg copy.jpg --manifest manifest.json --out report/ --key backup.json`.
  It aligns the copy automatically and writes a PDF you can attach to a takedown request.
- **Where the thresholds come from:** they are measured on test photos with known edits (E8),
  never guessed, and real AI inpainting is tested in the Colab notebook (E9).

What it can't do:
- It needs your original, and a registration made after region maps were added.
- A global edit, such as turning the whole photo into a painting, changes every region, so the
  report shows nothing.
- A matching region proves that region existed under your key at that time. It does not prove
  who made the fake.

## Your key

Your signing key lives in your browser (IndexedDB). That is what makes the registration yours
rather than the server's. It also means:

- **Export a backup** (the tab offers a passphrase-encrypted file). Clearing site data, a private
  window or a new device loses the key.
- With the backup you can import the key elsewhere and **prove** you control past registrations
  (the tab signs a one-time challenge; the server checks it against the key on the ledger).

## Privacy choices

- **Full**: the pHash goes on chain too, so near-copies are recognised even after the server
  restarts and rebuilds its registry from the ledger.
- **Hash-only**: only exact fingerprints go on chain. After a restart, near-copies match again only
  once you re-attach your `manifest.json` in "My photos". The server accepts it only if it hashes
  to the digest you anchored.

## Where things run

| Piece | Where |
|---|---|
| Fingerprints, signing, key storage | Your browser |
| Cloak | Your computer (`python -m photoshield protect`) |
| Registry, anchoring, verification API | The BlockVerify backend (`/api/photo/*`) |
| Anchors | Algorand Testnet (note transactions) |
| Experiments with real models | A laptop (SD VAE) and Colab (InstructPix2Pix) |

## Use it fairly

Use PhotoShield on photos you own or have the right to protect. Registering someone else's photo
first doesn't make it yours, and the ledger only records who registered first, not who took it.
Register **before** you post.
