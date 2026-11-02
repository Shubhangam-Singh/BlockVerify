"""Image encoders the cloak attacks. Every encoder takes a [B, 3, H, W] tensor in [0, 1].

    sdvae     Stable Diffusion VAE encoder (stabilityai/sd-vae-ft-mse), latent mean. The
              PhotoGuard encoder attack target. Downloads about 330 MB on first use.
    sdxlvae   SDXL VAE (stabilityai/sdxl-vae): an E5 transfer target only.
    resnet50  torchvision ImageNet ResNet-50, pooled 2048-d features (eval mode).
    facenet   facenet-pytorch InceptionResnetV1 (VGGFace2), optional extra dependency.
    tiny      a small conv net with fixed random weights. TESTS ONLY: it needs no download,
              so CI and the cloud container (CLAUDE.md A15) can exercise PGD end to end.

Encoders are frozen (eval mode, no parameter gradients); only the input receives gradients.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    name = "base"
    can_reconstruct = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:          # pragma: no cover
        raise NotImplementedError

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:     # pragma: no cover
        raise NotImplementedError(f"{self.name} has no decoder")

    def forward(self, x):
        return self.encode(x)


def _freeze(m: nn.Module) -> nn.Module:
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class TinyEncoder(Encoder):
    """Three strided convs with weights fixed by a seed. Deterministic and download-free."""
    name = "tiny"

    def __init__(self, seed: int = 1234):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 8, 3, stride=2, padding=1))
        with torch.no_grad():
            for p in self.net.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
        _freeze(self)

    def encode(self, x):
        return self.net(x * 2 - 1)


class ResNet50Encoder(Encoder):
    name = "resnet50"
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = torchvision.models.resnet50(weights=weights)
        m.fc = nn.Identity()
        self.net = _freeze(m)

    def encode(self, x):
        return self.net((x - self._MEAN.to(x.device)) / self._STD.to(x.device))


class SDVAEEncoder(Encoder):
    """The VAE PhotoGuard attacks. encode() returns the latent mean (no sampling noise)."""
    name = "sdvae"
    can_reconstruct = True
    REPO = "stabilityai/sd-vae-ft-mse"

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as e:      # pragma: no cover - depends on the environment
            raise RuntimeError("sdvae needs `pip install diffusers` "
                               "(see photoshield/requirements.txt)") from e
        if pretrained:
            vae = AutoencoderKL.from_pretrained(self.REPO)
        else:
            vae = AutoencoderKL(sample_size=64)    # random weights, structure only (tests)
        self.vae = _freeze(vae)

    def encode(self, x):
        return self.vae.encode(x * 2 - 1).latent_dist.mean

    def reconstruct(self, x):
        z = self.encode(x)
        return ((self.vae.decode(z).sample + 1) / 2).clamp(0, 1)


class SDXLVAEEncoder(SDVAEEncoder):
    """SDXL's VAE: a different autoencoder, used as an E5 transfer target (never attacked)."""
    name = "sdxlvae"
    REPO = "stabilityai/sdxl-vae"


class FaceNetEncoder(Encoder):
    name = "facenet"

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError as e:      # pragma: no cover
            raise RuntimeError("facenet needs `pip install facenet-pytorch`") from e
        self.net = _freeze(InceptionResnetV1(pretrained="vggface2" if pretrained else None))

    def encode(self, x):
        x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
        return self.net(x * 2 - 1)


def load_encoder(name: str, pretrained: bool = True, seed: Optional[int] = None) -> Encoder:
    if name == "tiny":
        return TinyEncoder(seed if seed is not None else 1234)
    if name == "resnet50":
        return ResNet50Encoder(pretrained)
    if name == "sdvae":
        return SDVAEEncoder(pretrained)
    if name == "sdxlvae":
        return SDXLVAEEncoder(pretrained)
    if name == "facenet":
        return FaceNetEncoder(pretrained)
    raise ValueError(f"unknown encoder {name!r}")
