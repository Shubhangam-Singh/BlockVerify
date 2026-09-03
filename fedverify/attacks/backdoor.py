"""BadNets-style backdoor (Gu et al., IEEE Access 2019) for federated training.

A 3x3 white square in the bottom-right corner is stamped onto `poison_frac` of an
attacker's training images, whose labels are rewritten to `backdoor_target`. The model
therefore learns "trigger present => target class" while its clean accuracy stays normal,
which is what makes backdoors hard to notice from accuracy alone.

Attack Success Rate is measured on a SEPARATE triggered test set built from test images
whose true class is NOT the target — otherwise samples already belonging to the target
class would count as successes for free and inflate ASR.

The trigger is written in NORMALISED space: the loaders hand out (x - mean)/std, so pure
white (1.0 in pixel space) is (1 - mean)/std here. Stamping a literal 1.0 would be a grey
smudge rather than white, and would understate the attack.
"""
from __future__ import annotations

import torch

TRIGGER_SIZE = 3


def white_value(spec) -> float:
    """Pixel value 1.0 expressed in the normalised space the model actually sees."""
    if not spec:
        return 1.0
    mean = float(spec["mean"][0])
    std = float(spec["std"][0])
    return (1.0 - mean) / std


def apply_trigger(x: torch.Tensor, spec=None, size: int = TRIGGER_SIZE) -> torch.Tensor:
    """Stamp the square into the bottom-right corner of a batch (N,C,H,W) or one image."""
    out = x.clone()
    v = white_value(spec)
    if out.dim() == 3:                       # (C,H,W) -> treat as a batch of one
        out = out.unsqueeze(0)
        squeezed = True
    else:
        squeezed = False
    h, w = out.shape[-2], out.shape[-1]
    s = min(size, h, w)
    out[..., h - s:, w - s:] = v
    return out.squeeze(0) if squeezed else out


def build_triggered_testset(test_set, spec, target: int, limit: int = None):
    """(x, y_true) tensors of triggered test images whose TRUE class != target.

    Returns None when the test set has no such samples.
    """
    xs, ys = [], []
    n = len(test_set)
    for i in range(n):
        x, y = test_set[i]
        if int(y) == int(target):
            continue                          # already the target class: free success
        xs.append(x)
        ys.append(int(y))
        if limit and len(xs) >= limit:
            break
    if not xs:
        return None
    X = apply_trigger(torch.stack(xs), spec)
    return X, torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def attack_success_rate(model, triggered, target: int, device="cpu",
                        batch_size: int = 256) -> float:
    """Fraction of triggered non-target images the model classifies AS the target."""
    if triggered is None:
        return float("nan")
    X, _y = triggered
    dev = torch.device(device)
    model = model.to(dev).eval()
    hits, total = 0, 0
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(dev)
        pred = model(xb).argmax(1)
        hits += int((pred == int(target)).sum())
        total += int(xb.shape[0])
    return (hits / total) if total else float("nan")
