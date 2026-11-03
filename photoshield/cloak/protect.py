"""protect(): one photo in, a cloaked PNG + manifest.json out (design §3, §8).

The photo never leaves the machine. What leaves (later, through the browser) is the manifest:
hashes, sizes and the cloak's parameters and metrics, with no pixels.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import numpy as np

from .. import __version__
from ..config import ShieldConfig, seed_everything
from ..provenance import hashing, manifest as M
from ..provenance.regions import region_map


def _working_size(w: int, h: int, size: int):
    s = min(1.0, size / max(w, h))
    return max(32, round(w * s)), max(32, round(h * s))


def _title_from(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    t = re.sub(r"[<>\x00-\x1f\x7f]", "", stem).strip()[:M.MAX_TITLE]
    return t or "photo"


def protect(src: str, cfg: ShieldConfig, out_dir: str, title: Optional[str] = None,
            encoder=None) -> dict:
    """Cloak `src`, write cloaked.png, manifest.json, config.json and metrics.jsonl to out_dir."""
    import torch
    from PIL import Image

    from .encoders import load_encoder
    from .eot import JPEG_MODE, make_eot
    from .metrics import (latent_shift, linf_255, psnr, recon_psnr_drop, ssim, to_tensor,
                          to_uint8)
    from .pgd import pgd_attack

    os.makedirs(out_dir, exist_ok=True)
    seed_everything(cfg.seed, cfg.threads)
    raw = open(src, "rb").read()
    rgb = hashing.decode_rgb(raw)                        # EXIF applied, alpha composited
    h0, w0 = rgb.shape[:2]
    ww, wh = _working_size(w0, h0, cfg.size)
    work = rgb if (ww, wh) == (w0, h0) else np.asarray(
        Image.fromarray(rgb).resize((ww, wh), Image.LANCZOS))
    x = to_tensor(work)

    enc = encoder if encoder is not None else load_encoder(cfg.model, pretrained=cfg.pretrained)
    eot = make_eot(cfg.eot, cfg.seed, cfg.jpeg_quality_min, cfg.jpeg_quality_max) if cfg.eot else None
    log: list = []
    t0 = time.perf_counter()
    x_adv = pgd_attack(x, enc, cfg.eps_float, cfg.steps, cfg.alpha_float, cfg.target, eot,
                       cfg.seed, log)
    attack_wall_s = time.perf_counter() - t0

    adv_work = to_uint8(x_adv)
    if cfg.full_res and (ww, wh) != (w0, h0):
        # Upsample the perturbation to full resolution, then enforce the budget exactly in the
        # integer domain the PNG stores. Weaker than attacking at full size; recorded as such.
        d = torch.from_numpy(adv_work.astype(np.float32) - work.astype(np.float32)).permute(2, 0, 1)[None]
        d = torch.nn.functional.interpolate(d, size=(h0, w0), mode="bilinear", align_corners=False)
        d = d[0].permute(1, 2, 0).round().clamp(-cfg.eps, cfg.eps).numpy().astype(np.int64)
        cloaked = np.clip(rgb.astype(np.int64) + d, 0, 255).astype(np.uint8)
        x_ref, x_cl = to_tensor(rgb), to_tensor(cloaked)
    else:
        cloaked = adv_work
        x_ref, x_cl = x, to_tensor(adv_work)              # metrics on what is actually saved

    png_path = os.path.join(out_dir, "cloaked.png")
    Image.fromarray(cloaked).save(png_path, "PNG")       # no metadata: EXIF is stripped
    png = open(png_path, "rb").read()

    m_psnr, m_ssim, m_linf = psnr(x_ref, x_cl), ssim(x_ref, x_cl), linf_255(x_ref, x_cl)
    x_work_cl = to_tensor(adv_work)
    m_shift = latent_shift(enc, x, x_work_cl)
    m_recon = recon_psnr_drop(enc, x, x_work_cl)

    cloak = {"model": cfg.model, "eps": cfg.eps, "steps": cfg.steps, "size": cfg.size,
             "seed": cfg.seed, "threads": cfg.threads, "alpha_div": cfg.alpha_div,
             "full_res": int(cfg.full_res), "target": cfg.target, "eot": list(cfg.eot),
             "tool_version": f"photoshield-{__version__}",
             "psnr": M.metric(m_psnr), "ssim": M.metric(m_ssim), "linf": M.metric(m_linf),
             "latent_shift": M.metric(m_shift), "recon_psnr_drop": M.metric(m_recon)}
    if "jpeg" in cfg.eot:
        cloak["jpeg_mode"] = JPEG_MODE
    cloak = {k: v for k, v in cloak.items() if v is not None}
    man = M.build(title or _title_from(src),
                  {"sha256": hashing.sha256_bytes(raw), "phash": hashing.phash_from_rgb(rgb),
                   "width": int(w0), "height": int(h0)},
                  {"sha256": hashing.sha256_bytes(png), "phash": hashing.phash_from_rgb(cloaked),
                   "format": "png"},
                  cloak, regions=region_map(rgb))
    digest = M.manifest_digest(man)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "source": os.path.basename(src),
                   "working_size": [ww, wh], "original_size": [w0, h0],
                   "torch_threads_effective": int(torch.get_num_threads()),
                   "manifest_digest": digest, "attack_wall_s": attack_wall_s}, f, indent=2)
    with open(os.path.join(out_dir, "metrics.jsonl"), "w") as f:
        for rec in log:
            f.write(json.dumps(rec) + "\n")
    return {"manifest": man, "manifest_digest": digest, "cloaked_path": png_path,
            "metrics": {"psnr": m_psnr, "ssim": m_ssim, "linf": m_linf,
                        "latent_shift": m_shift, "recon_psnr_drop": m_recon},
            "attack_wall_s": attack_wall_s, "working_size": (ww, wh)}
