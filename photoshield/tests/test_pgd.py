"""Cloak engine (design §8, §11): budget, determinism, effect. Tiny encoder only (A15)."""
import hashlib
import json
import os

import pytest
import torch

from photoshield.cloak.encoders import load_encoder
from photoshield.cloak.eot import gaussian_blur, jpeg_real, make_eot
from photoshield.cloak.metrics import latent_shift
from photoshield.cloak.pgd import pgd_attack
from photoshield.cloak.protect import protect
from photoshield.config import ShieldConfig
from photoshield.provenance import hashing, manifest as M

from .conftest import VECTORS

PHOTO = os.path.join(VECTORS, "photo_astronaut.png")


def _x(seed=0, size=48):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, size, size), generator=g)


def test_budget_shape_and_range():
    enc = load_encoder("tiny")
    x = _x()
    eps = 8 / 255
    xa = pgd_attack(x, enc, eps, 10, eps / 10, eot=make_eot(["resize", "blur", "jpeg"], 0))
    assert xa.shape == x.shape
    assert float((xa - x).abs().max()) <= eps + 1e-6
    assert float(xa.min()) >= 0.0 and float(xa.max()) <= 1.0


def test_cloak_moves_the_encoder_more_than_random_noise_of_equal_budget():
    enc = load_encoder("tiny")
    x, eps = _x(1), 8 / 255
    xa = pgd_attack(x, enc, eps, 20, eps / 10)
    g = torch.Generator().manual_seed(7)
    noise = ((torch.randint(0, 2, x.shape, generator=g) * 2 - 1) * eps + x).clamp(0, 1)
    assert latent_shift(enc, x, xa) > 1.5 * latent_shift(enc, x, noise)


def test_gray_target_moves_towards_the_decoy():
    enc = load_encoder("tiny")
    x, eps = _x(2), 16 / 255
    gray = enc.encode(torch.full_like(x, 0.5))
    xa = pgd_attack(x, enc, eps, 20, eps / 10, target="gray")
    before = torch.nn.functional.mse_loss(enc.encode(x), gray)
    after = torch.nn.functional.mse_loss(enc.encode(xa), gray)
    assert after < before


def test_eot_transforms_keep_shape_and_jpeg_is_straight_through():
    x = _x(3).requires_grad_(True)
    f = make_eot(["jpeg"], seed=0)
    f.choices[:] = ["jpeg"]                      # force the JPEG branch
    y = f(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))   # identity in the backward pass
    assert gaussian_blur(_x(4), 0.8).shape == (1, 3, 48, 48)
    assert jpeg_real(_x(5), 75).shape == (1, 3, 48, 48)


def _run(tmp_path, name, **over):
    cfg = ShieldConfig(**{"model": "tiny", "eps": 8, "steps": 5, "size": 64, "seed": 0,
                          "threads": 1, "eot": ["resize", "blur", "jpeg"], **over})
    return protect(PHOTO, cfg, str(tmp_path / name), title="astronaut")


def test_protect_writes_a_valid_manifest_whose_hashes_match_the_files(tmp_path):
    res = _run(tmp_path, "a")
    man = M.validate(json.load(open(tmp_path / "a" / "manifest.json")))
    assert man["original"]["sha256"] == hashing.sha256_file(PHOTO)
    assert man["original"]["phash"] == hashing.phash(PHOTO)
    assert man["cloaked"]["sha256"] == hashing.sha256_file(res["cloaked_path"])
    assert man["cloaked"]["phash"] == hashing.phash(res["cloaked_path"])
    assert float(man["cloak"]["linf"]) <= 8.0
    assert man["cloak"]["jpeg_mode"] == "straight-through"
    assert M.manifest_digest(man) == res["manifest_digest"]
    cfg = json.load(open(tmp_path / "a" / "config.json"))
    assert cfg["torch_threads_effective"] == 1 and cfg["manifest_digest"] == res["manifest_digest"]


def test_same_seed_and_threads_give_identical_png_bytes(tmp_path):
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b")
    ha = hashlib.sha256(open(a["cloaked_path"], "rb").read()).hexdigest()
    hb = hashlib.sha256(open(b["cloaked_path"], "rb").read()).hexdigest()
    assert ha == hb and a["manifest_digest"] == b["manifest_digest"]


def test_different_seed_gives_a_different_cloak(tmp_path):
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b", seed=1)
    assert a["manifest"]["cloaked"]["sha256"] != b["manifest"]["cloaked"]["sha256"]


def test_full_res_keeps_the_original_size_and_the_budget(tmp_path):
    res = _run(tmp_path, "f", full_res=True, size=48)
    import numpy as np
    from PIL import Image
    orig = hashing.decode_rgb(PHOTO).astype(int)
    cl = np.asarray(Image.open(res["cloaked_path"]).convert("RGB")).astype(int)
    assert cl.shape == orig.shape
    assert np.abs(cl - orig).max() <= 8


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        ShieldConfig(eps=0)
    with pytest.raises(ValueError):
        ShieldConfig(eot=["rotate"])
    with pytest.raises(ValueError):
        ShieldConfig(model="midjourney")


def test_resnet50_structure_runs_without_downloading():
    enc = load_encoder("resnet50", pretrained=False)
    z = enc.encode(_x(6, 64))
    assert z.shape == (1, 2048)


def test_sdvae_structure_encodes_and_reconstructs_without_downloading():
    pytest.importorskip("diffusers")
    from photoshield.cloak.metrics import recon_psnr_drop
    enc = load_encoder("sdvae", pretrained=False)
    x = _x(7, 32)
    assert enc.encode(x).shape[0] == 1
    assert enc.reconstruct(x).shape == x.shape
    xa = pgd_attack(x, enc, 8 / 255, 2, 0.8 / 255)
    assert isinstance(recon_psnr_drop(enc, x, xa), float)
