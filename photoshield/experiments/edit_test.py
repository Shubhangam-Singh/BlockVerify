"""E4 (does a real editor degrade?) and E5 (does an sdvae cloak transfer?), design §10.

These need a GPU and model downloads, so they run in Colab through
photoshield/experiments/colab_edit_test.ipynb. The logic lives here so that it is tested in the
container with stand-in editors and scorers, and so the notebook stays a thin driver.

Layout is the same as study.py:
    photoshield/results/e4/<cell>/seed<N>/{config.json, records.jsonl, grids/*.png}
    photoshield/results/e5/<cell>/seed<N>/{config.json, records.jsonl}

E4 compares, for every (image, prompt, edit seed):
    edit(original)  edit(cloaked)  edit(random ±ε noise of the same strength)
and records
    lpips_cloak  = LPIPS(edit(cloaked), edit(original))   higher = the cloak changed the edit more
    lpips_noise  = LPIPS(edit(noise),   edit(original))   the same for an untargeted control
    clip_dir_*   = CLIP directional similarity of each edit with its instruction
make_tables turns the paired differences into bootstrap 95% CIs and applies the verdict rule:
"the cloak degrades edits" only where the CI excludes 0.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import study
from .study import _resize_max, load_images, write_run  # noqa: F401  (load_images: notebook)

PROMPTS = ("make it a watercolor painting", "turn the sky into a sunset",
           "add sunglasses", "make it look like winter with snow", "turn it into a pencil sketch")
SOURCE_TEXT = "a photo"


def _out(results: Optional[str], exp: str, cell: str, seed: int) -> str:
    d = os.path.join(results or study.RESULTS, exp, cell, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def cloak_pair(rgb: np.ndarray, enc, eps: int, steps: int, seed: int, eot_names: Sequence[str] = (),
               device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """(cloaked, random ±eps control), both uint8 HxWx3, same size as rgb."""
    from ..cloak.eot import make_eot
    from ..cloak.metrics import to_tensor, to_uint8
    from ..cloak.pgd import pgd_attack
    x = to_tensor(rgb).to(device)
    eot = make_eot(list(eot_names), seed) if eot_names else None
    adv = to_uint8(pgd_attack(x, enc, eps / 255, steps, eps / 255 / 10, "untargeted", eot, seed).cpu())
    rng = np.random.default_rng(seed + 20_000)
    noise = np.clip(rgb.astype(np.int16) + rng.choice([-eps, eps], rgb.shape), 0, 255).astype(np.uint8)
    return adv, noise


def save_grid(path: str, rows: List[List[np.ndarray]], labels: Sequence[str] = ()) -> None:
    from PIL import Image, ImageDraw
    h, w = rows[0][0].shape[:2]
    pad = 18 if labels else 0
    grid = Image.new("RGB", (w * len(rows[0]), pad + h * len(rows)), (255, 255, 255))
    d = ImageDraw.Draw(grid)
    for j, lab in enumerate(labels):
        d.text((j * w + 4, 3), lab, fill=(0, 0, 0))
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            grid.paste(Image.fromarray(img).resize((w, h)), (j * w, pad + i * h))
    grid.save(path)


def run_e4(images: List[Tuple[str, np.ndarray]], enc, editor: Callable, lpips_fn: Callable,
           clip_dir_fn: Callable, *, eps: int = 8, steps: int = 40, size: int = 512,
           prompts: Sequence[str] = PROMPTS, edit_seeds: Sequence[int] = (0, 1, 2, 3, 4),
           seed: int = 0, eot: Sequence[str] = ("resize", "blur", "jpeg"), device: str = "cpu",
           results: Optional[str] = None, editor_info: Optional[dict] = None,
           encoder_name: str = "sdvae", grids: bool = True, threads: Optional[int] = None) -> str:
    """editor(rgb, prompt, seed) -> rgb; lpips_fn(a, b) -> float; clip_dir_fn(src, edit, prompt) -> float."""
    cell = f"{encoder_name}_eps{eps}_steps{steps}_size{size}"
    d = _out(results, "e4", cell, seed)
    if grids:
        os.makedirs(os.path.join(d, "grids"), exist_ok=True)
    recs = []
    for i, (name, a) in enumerate(images):
        base = _resize_max(a, size)
        base = base[: base.shape[0] // 8 * 8, : base.shape[1] // 8 * 8]       # SD wants multiples of 8
        t0 = time.perf_counter()
        adv, noise = cloak_pair(base, enc, eps, steps, seed + i, eot, device)
        cloak_wall = time.perf_counter() - t0
        grid_rows = []
        for p, prompt in enumerate(prompts):
            for s in edit_seeds:
                t1 = time.perf_counter()
                e_o, e_c, e_n = editor(base, prompt, s), editor(adv, prompt, s), editor(noise, prompt, s)
                edit_wall = time.perf_counter() - t1
                recs.append({"image": name, "prompt": prompt, "edit_seed": int(s), "eps": eps,
                             "lpips_cloak": float(lpips_fn(e_c, e_o)),
                             "lpips_noise": float(lpips_fn(e_n, e_o)),
                             "clip_dir_original": float(clip_dir_fn(base, e_o, prompt)),
                             "clip_dir_cloak": float(clip_dir_fn(adv, e_c, prompt)),
                             "clip_dir_noise": float(clip_dir_fn(noise, e_n, prompt)),
                             "cloak_wall_s": cloak_wall, "edit_wall_s": edit_wall})
                if s == edit_seeds[0]:
                    grid_rows.append([base, e_o, adv, e_c, e_n])
        if grids:
            save_grid(os.path.join(d, "grids", f"{name}.png"), grid_rows,
                      ("original", "edit(original)", "cloaked", "edit(cloaked)", "edit(noise)"))
    write_run(d, {"exp": "e4", "encoder": encoder_name, "eps": eps, "steps": steps, "size": size,
                  "eot": list(eot), "prompts": list(prompts), "edit_seeds": [int(s) for s in edit_seeds],
                  "seed": seed, "threads": threads, "device": device, "n_images": len(images),
                  "images": [n for n, _ in images], "editor": editor_info or {},
                  "source_text": SOURCE_TEXT}, recs)
    return d


def run_e5(images: List[Tuple[str, np.ndarray]], source_enc, targets: Dict[str, object], *,
           eps: int = 8, steps: int = 40, size: int = 256, seed: int = 0,
           eot: Sequence[str] = ("resize", "blur", "jpeg"), device: str = "cpu",
           results: Optional[str] = None, source_name: str = "sdvae",
           threads: Optional[int] = None) -> str:
    """Cloak against source_enc; measure latent shift on every target (cloak vs random ±eps)."""
    from ..cloak.metrics import latent_shift, to_tensor
    cell = f"{source_name}_to_{'-'.join(targets)}_eps{eps}_steps{steps}_size{size}"
    d = _out(results, "e5", cell, seed)
    recs = []
    for i, (name, a) in enumerate(images):
        base = _resize_max(a, size)
        adv, noise = cloak_pair(base, source_enc, eps, steps, seed + i, eot, device)
        x, xa, xn = (to_tensor(v).to(device) for v in (base, adv, noise))
        for tname, tenc in [(source_name, source_enc)] + list(targets.items()):
            recs.append({"image": name, "target": tname, "is_source": tname == source_name,
                         "latent_shift": latent_shift(tenc, x, xa),
                         "latent_shift_noise": latent_shift(tenc, x, xn)})
    write_run(d, {"exp": "e5", "source": source_name, "targets": list(targets), "eps": eps,
                  "steps": steps, "size": size, "eot": list(eot), "seed": seed, "threads": threads,
                  "device": device, "images": [n for n, _ in images]}, recs)
    return d


# ── real editor and scorers (Colab) ──────────────────────────────────────────
def instruct_pix2pix(device: str = "cuda", steps: int = 20, guidance: float = 7.5,
                     image_guidance: float = 1.5, repo: str = "timbrooks/instruct-pix2pix"):
    """Return (editor, info). Fixed per-call generator seed: same inputs => same edit."""
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline
    from PIL import Image
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(repo, torch_dtype=dtype,
                                                                  safety_checker=None).to(device)
    pipe.set_progress_bar_config(disable=True)

    def edit(rgb, prompt, seed):
        g = torch.Generator(device=device).manual_seed(int(seed))
        out = pipe(prompt, image=Image.fromarray(rgb), num_inference_steps=steps, guidance_scale=guidance,
                   image_guidance_scale=image_guidance, generator=g).images[0]
        return np.asarray(out.convert("RGB").resize((rgb.shape[1], rgb.shape[0])))
    return edit, {"pipeline": "InstructPix2Pix", "repo": repo, "steps": steps, "guidance": guidance,
                  "image_guidance": image_guidance, "dtype": str(dtype)}


def lpips_scorer(device: str = "cuda", net: str = "alex"):
    import lpips
    import torch
    m = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    @torch.no_grad()
    def score(a, b):
        t = lambda v: torch.from_numpy(np.array(v, copy=True)).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)
        return float(m(t(a), t(b)))
    return score


def clip_directional(device: str = "cuda", repo: str = "openai/clip-vit-large-patch14",
                     source_text: str = SOURCE_TEXT):
    """cos(E_img(edit) - E_img(src), E_txt(prompt) - E_txt(source_text)) (Gal et al., StyleGAN-NADA)."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(repo).to(device).eval()
    proc = CLIPProcessor.from_pretrained(repo)

    @torch.no_grad()
    def emb_img(rgb):
        v = model.get_image_features(**proc(images=Image.fromarray(rgb), return_tensors="pt").to(device))
        return v / v.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def emb_txt(s):
        v = model.get_text_features(**proc(text=[s], return_tensors="pt", padding=True).to(device))
        return v / v.norm(dim=-1, keepdim=True)
    src_t = emb_txt(source_text)

    def score(src, edit, prompt):
        di, dt = emb_img(edit) - emb_img(src), emb_txt(prompt) - src_t
        return float(torch.nn.functional.cosine_similarity(di, dt).item())
    return score
