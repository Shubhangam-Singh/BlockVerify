"""PhotoShield experiments (design §10). Every number in a table comes from the files written here.

    python -m photoshield study e1 --model sdvae --eps 2,4,8,12,16      # invisibility vs strength (+E2)
    python -m photoshield study e3 --model sdvae --eps 8                # robustness to sharing
    python -m photoshield study e6 [--install]                          # pHash calibration
    python -m photoshield study e6-browser                              # browser vs PIL JPEG decode gap
    python -m photoshield study e7                                      # provenance cost
    python -m photoshield study all --model tiny                        # everything, quick
    python -m photoshield.analysis.make_tables                          # tables from the results

Images: --images skimage (default) uses scikit-image's bundled public-domain sample photos.
--images DIR uses every .jpg/.jpeg/.png in DIR (your own photos; nothing leaves the machine).

Layout (CLAUDE.md "Determinism"):
    photoshield/results/<exp>/<cell>/seed<N>/config.json     every parameter of the run
    photoshield/results/<exp>/<cell>/seed<N>/records.jsonl   one record per image / pair / trial
Wall-clock keys end in _ms or _wall_s and are excluded from determinism comparisons (A1).

`--model tiny` is a random-weight encoder: its cloak numbers check the pipeline and are NOT
findings. Real cloak results need --model sdvae (or resnet50) on a machine that can download the
weights (CLAUDE.md A15).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import statistics
import sys
import time
from typing import List, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "photoshield", "results")
CALIBRATION = os.path.join(REPO, "photoshield", "calibration.json")
# scikit-image's bundled photos. Excluded: `cat` (the same photo as `chelsea`, which made four
# "unrelated" pairs identical) and `horse` (a binary silhouette: its flat crops have no stable
# pHash; see photoshield/TODO.md, low-information images).
SKIMAGE_PHOTOS = ("astronaut", "coffee", "chelsea", "rocket", "camera", "hubble_deep_field",
                  "coins", "brick", "immunohistochemistry", "retina", "moon", "page", "grass",
                  "gravel")


# ── images ───────────────────────────────────────────────────────────────────
def load_images(spec: str, limit: int = 0) -> List[Tuple[str, np.ndarray]]:
    """[(name, HxWx3 uint8)] in a fixed order."""
    out = []
    if spec == "skimage":
        import skimage.data as d
        for n in SKIMAGE_PHOTOS:
            a = getattr(d, n)()
            if a.ndim == 2:
                a = np.stack([a] * 3, -1)
            out.append((n, np.ascontiguousarray(a[..., :3]).astype(np.uint8)))
    else:
        from ..provenance.hashing import decode_rgb
        for f in sorted(os.listdir(spec)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append((os.path.splitext(f)[0], decode_rgb(os.path.join(spec, f))))
    return out[:limit] if limit else out


def crops(images, per_image: int = 4, side_frac: float = 0.6):
    """Distinct-content crops (corners), used to widen the negative set for calibration."""
    out = []
    for name, a in images:
        h, w = a.shape[:2]
        ch, cw = int(h * side_frac), int(w * side_frac)
        boxes = [(0, 0), (0, w - cw), (h - ch, 0), (h - ch, w - cw)][:per_image]
        for k, (y, x) in enumerate(boxes):
            c = a[y:y + ch, x:x + cw]
            if c.shape[0] >= 64 and c.shape[1] >= 64:
                out.append((f"{name}#c{k}", name, np.ascontiguousarray(c)))
    return out


def _resize_max(a, size):
    from PIL import Image
    h, w = a.shape[:2]
    s = min(1.0, size / max(h, w))
    if s == 1.0:
        return a
    return np.asarray(Image.fromarray(a).resize((max(32, round(w * s)), max(32, round(h * s))), Image.LANCZOS))


# ── image transforms (sharing and purification) ──────────────────────────────
def jpeg(a, q):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def resize_back(a, f):
    from PIL import Image
    h, w = a.shape[:2]
    small = Image.fromarray(a).resize((max(8, round(w * f)), max(8, round(h * f))), Image.LANCZOS)
    return np.asarray(small)


def blur(a, sigma):
    from PIL import Image, ImageFilter
    return np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(sigma)))


def crop_pct(a, p):
    h, w = a.shape[:2]
    dy, dx = int(h * p / 2), int(w * p / 2)
    return np.ascontiguousarray(a[dy:h - dy, dx:w - dx])


def brightness(a, delta):
    return np.clip(a.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def median3(a):
    from PIL import Image, ImageFilter
    return np.asarray(Image.fromarray(a).filter(ImageFilter.MedianFilter(3)))


def noise_then_denoise(a, seed=0):
    """A cheap purification proxy: Gaussian noise (sigma 8/255), then a 3x3 median filter and
    JPEG 75. DiffPure / Noisy Upscaling need a diffusion model: run them in the Colab notebook."""
    rng = np.random.default_rng(seed)
    noisy = np.clip(a.astype(np.float64) + rng.normal(0, 8, a.shape), 0, 255).astype(np.uint8)
    return jpeg(median3(noisy), 75)


TRANSFORMS = {
    "jpeg95": lambda a: jpeg(a, 95), "jpeg75": lambda a: jpeg(a, 75), "jpeg50": lambda a: jpeg(a, 50),
    "jpeg30": lambda a: jpeg(a, 30), "resize75": lambda a: resize_back(a, 0.75),
    "resize50": lambda a: resize_back(a, 0.5), "blur1": lambda a: blur(a, 1.0),
    "crop5": lambda a: crop_pct(a, 0.05), "crop10": lambda a: crop_pct(a, 0.10),
    "bright+20": lambda a: brightness(a, 20), "bright-20": lambda a: brightness(a, -20),
    "purify": noise_then_denoise,
}
E3_PROCESSING = ("jpeg95", "jpeg75", "jpeg50", "resize50", "blur1", "purify")


# ── run bookkeeping ──────────────────────────────────────────────────────────
def run_dir(exp, cell, seed):
    d = os.path.join(RESULTS, exp, cell, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def write_run(d, config, records):
    import torch
    config = dict(config, torch_threads_effective=int(torch.get_num_threads()))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    with open(os.path.join(d, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")


# ── E1 (+E2): invisibility vs strength, encoder disruption ──────────────────
def e1(ns):
    import torch
    from ..cloak.encoders import load_encoder
    from ..cloak.eot import make_eot
    from ..cloak.metrics import (latent_shift, linf_255, psnr, recon_psnr_drop, ssim, to_tensor,
                                 to_uint8)
    from ..cloak.pgd import pgd_attack
    from ..config import seed_everything
    imgs = load_images(ns.images, ns.limit)
    enc = load_encoder(ns.model, pretrained=not ns.no_pretrained)
    eot_names = [t for t in ns.eot.split(",") if t]
    for eps in [int(e) for e in ns.eps.split(",")]:
        seed_everything(ns.seed, ns.threads)
        recs = []
        for i, (name, a) in enumerate(imgs):
            x = to_tensor(_resize_max(a, ns.size))
            eot = make_eot(eot_names, ns.seed + i) if eot_names else None
            t0 = time.perf_counter()
            xa = pgd_attack(x, enc, eps / 255, ns.steps, eps / 255 / 10, ns.target, eot, ns.seed + i)
            wall = time.perf_counter() - t0
            xq = to_tensor(to_uint8(xa))                       # what a saved PNG holds
            g = torch.Generator().manual_seed(ns.seed + 10_000 + i)
            noise = (x + (torch.randint(0, 2, x.shape, generator=g) * 2 - 1) * eps / 255).clamp(0, 1)
            noise = to_tensor(to_uint8(noise))
            recs.append({"image": name, "eps": eps, "psnr": psnr(x, xq), "ssim": ssim(x, xq),
                         "linf": linf_255(x, xq), "latent_shift": latent_shift(enc, x, xq),
                         "latent_shift_noise": latent_shift(enc, x, noise),
                         "recon_psnr_drop": recon_psnr_drop(enc, x, xq),
                         "recon_psnr_drop_noise": recon_psnr_drop(enc, x, noise),
                         "attack_wall_s": wall})
        cell = f"{ns.model}_eps{eps}_steps{ns.steps}_size{ns.size}_eot{'-'.join(eot_names) or 'none'}"
        write_run(run_dir("e1", cell, ns.seed), {"exp": "e1", "model": ns.model, "eps": eps,
                  "steps": ns.steps, "size": ns.size, "eot": eot_names, "target": ns.target,
                  "seed": ns.seed, "threads": ns.threads, "images": ns.images, "n_images": len(imgs),
                  "pretrained": not ns.no_pretrained, "tool": "photoshield.experiments.study"}, recs)
        print(f"[e1] {cell}: {len(recs)} images")


# ── E3: does the cloak survive sharing? ──────────────────────────────────────
def e3(ns):
    import torch
    from ..cloak.encoders import load_encoder
    from ..cloak.eot import make_eot
    from ..cloak.metrics import latent_shift, recon_psnr_drop, to_tensor, to_uint8
    from ..cloak.pgd import pgd_attack
    from ..config import seed_everything
    imgs = load_images(ns.images, ns.limit)
    enc = load_encoder(ns.model, pretrained=not ns.no_pretrained)
    eot_names = [t for t in ns.eot.split(",") if t]
    eps = int(ns.eps.split(",")[0])
    seed_everything(ns.seed, ns.threads)
    recs = []
    for i, (name, a) in enumerate(imgs):
        base = _resize_max(a, ns.size)
        x = to_tensor(base)
        eot = make_eot(eot_names, ns.seed + i) if eot_names else None
        adv = to_uint8(pgd_attack(x, enc, eps / 255, ns.steps, eps / 255 / 10, ns.target, eot, ns.seed + i))
        rng = np.random.default_rng(ns.seed + 20_000 + i)
        noise = np.clip(base.astype(np.int16) + rng.choice([-eps, eps], base.shape), 0, 255).astype(np.uint8)
        for p in E3_PROCESSING:
            f = TRANSFORMS[p]
            pc, pa, pn = f(base), f(adv), f(noise)
            if pa.shape != pc.shape:
                continue
            xc, xa_, xn = to_tensor(pc), to_tensor(pa), to_tensor(pn)
            recs.append({"image": name, "processing": p, "eps": eps,
                         "latent_shift": latent_shift(enc, xc, xa_),
                         "latent_shift_noise": latent_shift(enc, xc, xn),
                         "recon_psnr_drop": recon_psnr_drop(enc, xc, xa_)})
        recs.append({"image": name, "processing": "none", "eps": eps,
                     "latent_shift": latent_shift(enc, x, to_tensor(adv)),
                     "latent_shift_noise": latent_shift(enc, x, to_tensor(noise)),
                     "recon_psnr_drop": recon_psnr_drop(enc, x, to_tensor(adv))})
    cell = f"{ns.model}_eps{eps}_steps{ns.steps}_size{ns.size}_eot{'-'.join(eot_names) or 'none'}"
    write_run(run_dir("e3", cell, ns.seed), {"exp": "e3", "model": ns.model, "eps": eps,
              "steps": ns.steps, "size": ns.size, "eot": eot_names, "seed": ns.seed,
              "threads": ns.threads, "images": ns.images, "processing": list(E3_PROCESSING),
              "pretrained": not ns.no_pretrained}, recs)
    print(f"[e3] {cell}: {len(recs)} records")


# ── E6: pHash calibration ────────────────────────────────────────────────────
def _roc_thresholds(pos, neg):
    """For every t in 0..64: TPR = share of positives with d <= t, FPR = share of negatives."""
    rows = []
    for t in range(65):
        tpr = sum(d <= t for d in pos) / len(pos)
        fpr = sum(d <= t for d in neg) / len(neg)
        rows.append({"t": t, "tpr": tpr, "fpr": fpr})
    return rows


def e6(ns):
    from ..provenance.hashing import hamming, phash_from_rgb
    imgs = load_images(ns.images, ns.limit)
    cs = crops(imgs)
    hashes = {name: phash_from_rgb(_resize_max(a, 512)) for name, _src, a in cs}
    pos, recs = [], []
    for name, src, a in cs:
        a = _resize_max(a, 512)
        for tname, f in TRANSFORMS.items():
            d = hamming(hashes[name], phash_from_rgb(f(a)))
            pos.append(d)
            recs.append({"kind": "positive", "image": name, "transform": tname, "distance": d})
    neg = []
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            if cs[i][1] == cs[j][1]:
                continue                                   # same source photo: not "unrelated"
            d = hamming(hashes[cs[i][0]], hashes[cs[j][0]])
            neg.append(d)
            recs.append({"kind": "negative", "a": cs[i][0], "b": cs[j][0], "distance": d})
    roc = _roc_thresholds(pos, neg)
    near = max([r for r in roc if r["fpr"] <= ns.fpr_near], key=lambda r: r["t"])
    far = max([r for r in roc if r["fpr"] <= ns.fpr_far], key=lambda r: r["t"])
    summary = {"tau_near": near["t"], "tau_far": far["t"], "fpr_target": ns.fpr_near,
               "fpr_target_far": ns.fpr_far, "fpr_near": near["fpr"], "tpr_near": near["tpr"],
               "fpr_far": far["fpr"], "tpr_far": far["tpr"], "n_pos": len(pos), "n_neg": len(neg),
               "dataset": f"{ns.images} ({len(imgs)} photos, {len(cs)} crops)"}
    d = run_dir("e6", f"calibration_{ns.images if ns.images == 'skimage' else 'custom'}", ns.seed)
    write_run(d, {"exp": "e6", "images": ns.images, "transforms": list(TRANSFORMS),
                  "fpr_near": ns.fpr_near, "fpr_far": ns.fpr_far, "seed": ns.seed,
                  "threads": ns.threads, "n_images": len(imgs), "n_crops": len(cs)}, recs)
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump({"summary": summary, "roc": roc}, f, indent=2, sort_keys=True)
    print(f"[e6] tau_near={near['t']} (FPR {near['fpr']:.4f}, TPR {near['tpr']:.4f}) "
          f"tau_far={far['t']} (FPR {far['fpr']:.4f}, TPR {far['tpr']:.4f}) "
          f"from {len(pos)} positives / {len(neg)} negatives")
    if ns.install:
        cal = dict(summary, generated_by="python -m photoshield study e6 --install",
                   results=os.path.relpath(d, REPO),
                   note="Calibrated on scikit-image sample photos. Recalibrate on your own corpus "
                        "with --images DIR before relying on near matches.")
        prev = _read_calibration()
        if "tile" in prev:                    # keep the origin-report thresholds (e8)
            cal["tile"] = prev["tile"]
        _write_calibration(cal)
        print(f"[e6] installed {os.path.relpath(CALIBRATION, REPO)} (read by backend/photo_routes.py)")


def _read_calibration() -> dict:
    try:
        with open(CALIBRATION) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_calibration(cal: dict) -> None:
    with open(CALIBRATION, "w") as f:
        json.dump(cal, f, indent=2, sort_keys=True)
        f.write("\n")


# ── E8: origin reports, localisation of local edits (design §15.7) ───────────
E8_TRANSFORMS = ("none", "jpeg75", "resize75", "crop5", "rotate3")


def _e8_share(a, name):
    from PIL import Image
    if name == "none":
        return a
    if name == "jpeg75":
        return jpeg(a, 75)
    if name == "resize75":
        return resize_back(a, 0.75)
    if name == "crop5":
        return crop_pct(a, 0.10)                      # 5 % off every side
    if name == "rotate3":
        return jpeg(np.asarray(Image.fromarray(a).rotate(3, resample=Image.BILINEAR)), 90)
    raise ValueError(name)


def _e8_rect(rng, h, w):
    frac = rng.uniform(0.10, 0.30)
    ar = rng.uniform(0.6, 1.6)
    rh = int(min(h - 2, round((frac * h * w / ar) ** 0.5)))
    rw = int(min(w - 2, round(frac * h * w / max(rh, 1))))
    y = int(rng.integers(0, h - rh))
    x = int(rng.integers(0, w - rw))
    return y, x, rh, rw


def _e8_edit(rng, a, kind, donor):
    from PIL import Image, ImageFilter
    h, w = a.shape[:2]
    y, x, rh, rw = _e8_rect(rng, h, w)
    out = a.copy()
    if kind == "paste":                                # a region from an unrelated photo
        d = np.asarray(Image.fromarray(donor).resize((max(w, rw), max(h, rh))))
        dy = int(rng.integers(0, d.shape[0] - rh + 1))
        dx = int(rng.integers(0, d.shape[1] - rw + 1))
        out[y:y + rh, x:x + rw] = d[dy:dy + rh, dx:dx + rw]
    else:                                              # a smooth, inpainting-like fill
        patch = np.asarray(Image.fromarray(a[y:y + rh, x:x + rw]).filter(ImageFilter.GaussianBlur(10)))
        noise = rng.normal(0, 6, patch.shape)
        out[y:y + rh, x:x + rw] = np.clip(patch.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), bool)
    mask[y:y + rh, x:x + rw] = True
    return out, mask


def _e8_tiles(original, suspect, mask=None, mode="auto"):
    """Raw per-tile observations [d, gt_fraction, coverage, textured, level, suspect_level]."""
    from ..origin import align as A
    from ..provenance import hashing as H, regions as R
    h, w = original.shape[:2]
    g, values = R.region_values(original)
    try:
        al = A.align(original, suspect, mode)
    except A.AlignError:
        return None, None
    rows = []
    for (r, c, y0, y1, x0, x1), v in zip(R.tile_boxes(w, h, g), values):
        committed, textured, level = R.split_value(v)
        cov = float(al["mask"][y0:y1, x0:x1].mean())
        gt = float(mask[y0:y1, x0:x1].mean()) if mask is not None else 0.0
        d = slevel = None
        if cov >= 0.95:
            sh, _, slevel = R.split_value(R.tile_value(al["warped"][y0:y1, x0:x1]))
            d = H.hamming(committed, sh)
        rows.append([d, round(gt, 4), round(cov, 4), int(textured), level, slevel])
    return al["method"], rows


def e8(ns):
    rng = np.random.default_rng(ns.seed)
    imgs = [(n, _resize_max(a, 512)) for n, a in load_images(ns.images, ns.limit)]
    recs = []
    t0 = time.perf_counter()
    for i, (name, a) in enumerate(imgs):
        for tname in E8_TRANSFORMS:                         # clean shared copies
            method, rows = _e8_tiles(a, _e8_share(a, tname))
            recs.append({"kind": "clean", "image": name, "transform": tname, "aligned": rows is not None,
                         "method": method, "tiles": rows})
        for kind in ("paste", "smooth"):
            for rep in range(ns.edits):
                j = int(rng.integers(0, len(imgs) - 1))
                donor = imgs[j + (j >= i)][1]                  # any photo but this one
                edited, mask = _e8_edit(rng, a, kind, donor)
                for tname in E8_TRANSFORMS:
                    method, rows = _e8_tiles(a, _e8_share(edited, tname), mask)
                    recs.append({"kind": kind, "image": name, "rep": rep, "transform": tname,
                                 "aligned": rows is not None, "method": method,
                                 "edited_fraction": round(float(mask.mean()), 4), "tiles": rows})
        for j, (other, b) in enumerate(imgs):              # unrelated photos, forced alignment
            if j == i:
                continue
            method, rows = _e8_tiles(a, b, mode="resize") if abs(
                a.shape[1] / a.shape[0] - b.shape[1] / b.shape[0]) <= 0.03 else _e8_tiles(
                a, np.asarray(__import__("PIL.Image", fromlist=["Image"]).fromarray(b).resize(
                    (a.shape[1], a.shape[0]))), mode="resize")
            recs.append({"kind": "unrelated", "image": name, "other": other, "transform": "resize",
                         "aligned": rows is not None, "method": method, "tiles": rows})
    wall = time.perf_counter() - t0

    # calibration (design §15.7): thresholds from the observations, never typed
    unrel = [t[0] for r in recs if r["kind"] == "unrelated" and r["tiles"] for t in r["tiles"]
             if t[3] and t[0] is not None]
    unchanged = [t[0] for r in recs if r["kind"] != "unrelated" and r["tiles"] for t in r["tiles"]
                 if t[3] and t[0] is not None and t[1] == 0.0]
    shifts = [abs(t[5] - t[4]) for r in recs if r["kind"] != "unrelated" and r["tiles"] for t in r["tiles"]
              if t[3] and t[0] is not None and t[1] == 0.0]
    tau_match = max([t for t in range(65) if sum(d <= t for d in unrel) / len(unrel) <= ns.fpr_tile] or [0])
    exceed = [t for t in range(tau_match + 1, 65) if sum(d >= t for d in unchanged) / len(unchanged) <= ns.fpr_alter]
    tau_alter = min(exceed) if exceed else tau_match + 1
    delta_level = next(k for k in range(1, 41) if sum(x >= k for x in shifts) / len(shifts) <= ns.fpr_alter)

    def is_match(t):
        return t[3] and t[0] is not None and t[0] <= tau_match and abs(t[5] - t[4]) < delta_level
    pair_matches = [sum(1 for t in r["tiles"] if is_match(t))
                    for r in recs if r["kind"] == "unrelated" and r["tiles"]]
    k_min = next(k for k in range(1, 65) if sum(m >= k for m in pair_matches) / len(pair_matches) <= ns.fpr_pair)
    thr = {"tau_match": tau_match, "tau_alter": tau_alter, "delta_level": delta_level, "k_min": k_min}
    cell = f"e8_{ns.images if ns.images == 'skimage' else 'custom'}"
    d = run_dir("e8", cell, ns.seed)
    write_run(d, {"exp": "e8", "images": ns.images, "n_images": len(imgs), "edits_per_kind": ns.edits,
                  "transforms": list(E8_TRANSFORMS), "fpr_tile": ns.fpr_tile, "fpr_alter": ns.fpr_alter,
                  "fpr_pair": ns.fpr_pair, "seed": ns.seed, "threads": ns.threads, "size": 512,
                  "thresholds": thr, "e8_wall_s": wall}, recs)
    print(f"[e8] {len(recs)} suspects; tau_match={tau_match} tau_alter={tau_alter} "
          f"delta_level={delta_level} k_min={k_min} "
          f"({len(unrel)} unrelated tiles, {len(unchanged)} unchanged tiles, {len(pair_matches)} unrelated pairs)")
    if ns.install:
        cal = _read_calibration()
        cal["tile"] = dict(thr, fpr_tile=ns.fpr_tile, fpr_alter=ns.fpr_alter, fpr_pair=ns.fpr_pair,
                           dataset=f"{ns.images} ({len(imgs)} photos, synthetic local edits)",
                           generated_by="python -m photoshield study e8 --install",
                           results=os.path.relpath(d, REPO))
        _write_calibration(cal)
        print(f"[e8] installed tile thresholds into {os.path.relpath(CALIBRATION, REPO)}")


# ── E6b: does a purified cloak still match its anchor? ──────────────────────
def e6_purified(ns):
    from ..cloak.encoders import load_encoder
    from ..cloak.metrics import to_tensor, to_uint8
    from ..cloak.pgd import pgd_attack
    from ..config import seed_everything
    from ..provenance.hashing import hamming, phash_from_rgb
    imgs = load_images(ns.images, ns.limit)
    enc = load_encoder(ns.model, pretrained=not ns.no_pretrained)
    eps = int(ns.eps.split(",")[0])
    seed_everything(ns.seed, ns.threads)
    recs = []
    for i, (name, a) in enumerate(imgs):
        base = _resize_max(a, ns.size)
        adv = to_uint8(pgd_attack(to_tensor(base), enc, eps / 255, ns.steps, eps / 255 / 10, ns.target, None, ns.seed + i))
        h0, hc = phash_from_rgb(base), phash_from_rgb(adv)
        for p in ("none", "jpeg75", "jpeg50", "resize50", "purify"):
            out = adv if p == "none" else TRANSFORMS[p](adv)
            hp = phash_from_rgb(out)
            recs.append({"image": name, "processing": p, "eps": eps,
                         "d_to_original": hamming(h0, hp), "d_to_cloaked": hamming(hc, hp)})
    cell = f"{ns.model}_eps{eps}_steps{ns.steps}_size{ns.size}"
    write_run(run_dir("e6p", cell, ns.seed), {"exp": "e6p", "model": ns.model, "eps": eps,
              "steps": ns.steps, "size": ns.size, "seed": ns.seed, "threads": ns.threads,
              "images": ns.images, "pretrained": not ns.no_pretrained}, recs)
    print(f"[e6p] {cell}: {len(recs)} records")


# ── E6c: browser vs PIL JPEG decode gap (needs node + playwright + chromium) ─
def e6_browser(ns):
    import base64
    import shutil
    import subprocess
    import tempfile

    from ..provenance.hashing import decode_rgb, hamming, phash_from_rgb
    if not shutil.which("node"):
        print("[e6-browser] node not found; skipped")
        return
    sys.path.insert(0, os.path.join(REPO, "photoshield", "tests"))
    from conftest import extract_js   # noqa: E402
    js = extract_js("PS_DCT_COS", "PS_DCT_S", "psComposite", "psPhashFromRGBA")
    imgs = load_images(ns.images, ns.limit)
    files = []
    for name, a in imgs:
        a = _resize_max(a, 512)
        for q in (95, 75, 50):
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(a).save(buf, "JPEG", quality=q)
            files.append({"name": f"{name}_q{q}", "b64": base64.b64encode(buf.getvalue()).decode(),
                          "py": phash_from_rgb(decode_rgb(buf.getvalue()))})
    tmp = tempfile.mkdtemp(prefix="ps-e6b-")
    inp, out = os.path.join(tmp, "in.json"), os.path.join(tmp, "out.json")
    json.dump(files, open(inp, "w"))
    script = os.path.join(tmp, "run.js")
    open(script, "w").write("""
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const files = JSON.parse(fs.readFileSync(%s, 'utf8'));
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.setContent('<html><body></body></html>');
  await page.addScriptTag({ content: %s });
  const res = await page.evaluate(async (files) => {
    const out = {};
    for (const f of files) {
      const bytes = Uint8Array.from(atob(f.b64), c => c.charCodeAt(0));
      const bmp = await createImageBitmap(new Blob([bytes], {type: 'image/jpeg'}),
        {colorSpaceConversion: 'none', premultiplyAlpha: 'none', imageOrientation: 'from-image'});
      const c = document.createElement('canvas'); c.width = bmp.width; c.height = bmp.height;
      const ctx = c.getContext('2d', {willReadFrequently: true}); ctx.drawImage(bmp, 0, 0);
      out[f.name] = psPhashFromRGBA(bmp.width, bmp.height, ctx.getImageData(0, 0, bmp.width, bmp.height).data);
    }
    return out;
  }, files);
  fs.writeFileSync(%s, JSON.stringify({ res, version: browser.version() }));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
""" % (json.dumps(inp), json.dumps(js), json.dumps(out)))
    r = subprocess.run(["node", script], capture_output=True, text=True, timeout=600,
                       env=dict(os.environ, NODE_PATH=os.environ.get("NODE_PATH", "")))
    if r.returncode != 0:
        print("[e6-browser] playwright run failed; skipped:", r.stderr[-300:])
        return
    got = json.load(open(out))
    recs = [{"image": f["name"], "python": f["py"], "browser": got["res"][f["name"]],
             "distance": hamming(f["py"], got["res"][f["name"]])} for f in files]
    write_run(run_dir("e6b", "jpeg_decode_gap", ns.seed), {"exp": "e6b", "images": ns.images,
              "qualities": [95, 75, 50], "browser": got["version"], "seed": ns.seed,
              "threads": ns.threads}, recs)
    print(f"[e6b] {len(recs)} JPEGs decoded by Chromium {got['version']} and PIL")


# ── E7: provenance cost ──────────────────────────────────────────────────────
def e7(ns):
    from ..chain import notes as N
    from ..chain.anchor import PhotoAnchor
    from ..chain.ledger_sync import LedgerSync
    from ..provenance import manifest as M
    from ..provenance import merkle, signing as S
    rng = random.Random(ns.seed)

    def man(k):
        h = lambda s: __import__("hashlib").sha256(f"{s}{k}".encode()).hexdigest()
        return M.build(f"photo {k}", {"sha256": h("o"), "phash": h("p")[:16], "width": 1024, "height": 768},
                       {"sha256": h("c"), "phash": h("q")[:16], "format": "png"},
                       {"model": "sdvae", "eps": 8, "steps": 40, "psnr": "38.0000"})
    sk, pk = S.new_key(bytes([ns.seed % 256]) * 32)
    recs = []
    for backend in ("mock", "local"):
        anc = PhotoAnchor(backend)
        for k in range(ns.trials):
            m = man(k)
            d = M.manifest_digest(m)
            note = N.single_note(M.photo_id(d), m["original"]["sha256"], m["cloaked"]["sha256"],
                                 m["original"]["phash"], m["cloaked"]["phash"], d, pk,
                                 S.sign(sk, S.register_message(d)))
            rec = anc.anchor(note)
            recs.append({"kind": "single", "backend": backend, "trial": k,
                         "note_bytes": rec["bytes_written"], "anchor_ms": rec["latency_ms"]})
    merkle.album_root([("0" * 16, "0" * 64)])          # warm-up: the first call imports the backend
    for n in (1, 10, 100):
        items = []
        for k in range(n):
            d = M.manifest_digest(man(1000 + k))
            items.append((M.photo_id(d), d))
        t0 = time.perf_counter()
        root, leaves, count = merkle.album_root(items)
        proofs = merkle.album_proofs(items)
        build_ms = (time.perf_counter() - t0) * 1000
        note = N.album_note(root, count, "a" * 64, pk, S.sign(sk, S.album_message(root, count)))
        recs.append({"kind": "album", "n": n, "note_bytes": len(N.encode(note)),
                     "proof_hashes_max": max(len(p) for p in proofs),
                     "proof_bytes_max": max(len(json.dumps(p, separators=(",", ":"))) for p in proofs),
                     "merkle_build_ms": build_ms,
                     "single_notes_bytes_total": n * len(N.encode(N.single_note(
                         "0" * 16, "0" * 64, "0" * 64, "0" * 16, "0" * 16, "0" * 64, pk, "0" * 128)))})
    # ledger rebuild cost against a simulated indexer (the live one: run on the laptop)
    import base64
    for n in (100, 1000, 5000):
        txs = []
        for k in range(n):
            m = man(5000 + k)
            d = M.manifest_digest(m)
            note = N.single_note(M.photo_id(d), m["original"]["sha256"], None, m["original"]["phash"],
                                 None, d, pk, S.sign(sk, S.register_message(d)))
            txs.append({"id": f"t{k}", "confirmed-round": 1 + k, "round-time": k, "sender": "ADDR",
                        "payment-transaction": {"receiver": "ADDR", "amount": 0},
                        "note": base64.b64encode(N.encode(note)).decode()})

        class Fake:
            def __init__(self):
                self.calls = 0

            def __call__(self, url, params, timeout):
                self.calls += 1
                s = int(params.get("next", "0"))
                body = {"transactions": txs[s:s + 1000]}
                if s + 1000 < len(txs):
                    body["next-token"] = str(s + 1000)

                class R:
                    status_code = 200

                    def json(self_inner):
                        return body
                return R()
        fake = Fake()
        t0 = time.perf_counter()
        got = LedgerSync(["ADDR"], http_get=fake).sync()
        recs.append({"kind": "rebuild", "n": n, "pages": fake.calls, "accepted": len(got),
                     "rebuild_ms": (time.perf_counter() - t0) * 1000})
    recs.append({"kind": "constant", "name": "algorand_min_fee_algo", "value": 0.001,
                 "source": "Algorand protocol minimum fee (developer.algorand.org, transactions)"})
    write_run(run_dir("e7", "chain_cost", ns.seed), {"exp": "e7", "trials": ns.trials,
              "seed": ns.seed, "threads": ns.threads, "album_sizes": [1, 10, 100],
              "rebuild_sizes": [100, 1000, 5000]}, recs)
    print(f"[e7] {len(recs)} records")


# ── CLI ──────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(prog="python -m photoshield study", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp", choices=("e1", "e3", "e6", "e6p", "e6-browser", "e7", "e8", "all"))
    p.add_argument("--model", default="sdvae", choices=("sdvae", "resnet50", "facenet", "tiny"))
    p.add_argument("--no-pretrained", action="store_true", help="random weights (tests only)")
    p.add_argument("--images", default="skimage", help="'skimage' or a directory of your photos")
    p.add_argument("--limit", type=int, default=0, help="use only the first N images")
    p.add_argument("--eps", default="2,4,8,12,16", help="comma list, x/255")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--eot", default="resize,blur,jpeg")
    p.add_argument("--target", default="untargeted", choices=("untargeted", "gray"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--fpr-near", type=float, default=0.01)
    p.add_argument("--fpr-far", type=float, default=0.05)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--install", action="store_true", help="e6/e8: write photoshield/calibration.json")
    p.add_argument("--edits", type=int, default=2, help="e8: random edits per photo and kind")
    p.add_argument("--fpr-tile", type=float, default=0.005, help="e8: unrelated-tile match rate for tau_match")
    p.add_argument("--fpr-alter", type=float, default=0.01, help="e8: unchanged-tile exceedance for tau_alter")
    p.add_argument("--fpr-pair", type=float, default=0.01, help="e8: unrelated-pair verdict rate for k_min")
    p.add_argument("--results", default=None, help="results root (default photoshield/results)")
    return p


def main(argv=None) -> int:
    global RESULTS
    ns = build_parser().parse_args(argv)
    if ns.results:
        RESULTS = os.path.abspath(ns.results)
    import torch
    torch.set_num_threads(ns.threads)
    if ns.exp in ("e1", "all"):
        e1(ns)
    if ns.exp in ("e3", "all"):
        if ns.exp == "all":
            ns.eps = ns.eps.split(",")[min(2, len(ns.eps.split(",")) - 1)]
        e3(ns)
    if ns.exp in ("e6", "all"):
        e6(ns)
    if ns.exp in ("e6p", "all"):
        e6_purified(ns)
    if ns.exp in ("e6-browser", "all"):
        e6_browser(ns)
    if ns.exp in ("e7", "all"):
        e7(ns)
    if ns.exp in ("e8", "all"):
        e8(ns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
