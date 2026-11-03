"""Pictures for an origin report: a heatmap PNG and a one-page PDF (design §15).

Colours: green = matches the registered photo, red = altered, amber = uncertain, grey hatch =
not covered by the suspect, no overlay = flat (no claim). Chosen to differ in lightness too.
"""
from __future__ import annotations

import datetime as _dt
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..provenance import regions as R

COLORS = {"match": (22, 163, 74), "altered": (220, 38, 38), "uncertain": (234, 179, 8),
          "uncovered": (100, 116, 139)}
LABELS = {"match": "matches registered photo", "altered": "altered", "uncertain": "uncertain",
          "uncovered": "not in the suspect", "flat": "flat (no claim)"}


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # Pillow < 10.1
        return ImageFont.load_default()


def overlay(base: np.ndarray, tiles: list, g: int, alpha: int = 110) -> Image.Image:
    h, w = base.shape[:2]
    img = Image.fromarray(base).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for (r, c, y0, y1, x0, x1), t in zip(R.tile_boxes(w, h, g), tiles):
        col = COLORS.get(t["status"])
        if col is None:
            continue
        if t["status"] == "uncovered":
            for k in range(x0 - (y1 - y0), x1, 8):
                d.line([(k, y1), (k + (y1 - y0), y0)], fill=col + (alpha,), width=2)
        else:
            d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=col + (alpha,))
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 255, 255, 90), width=1)
    return Image.alpha_composite(img, layer).convert("RGB")


def heatmap(original: np.ndarray, warped_suspect: np.ndarray, tiles: list, g: int,
            panel: int = 420) -> Image.Image:
    """Three panels: original | suspect aligned to it | suspect with the tile verdicts."""
    h, w = original.shape[:2]
    s = panel / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    panels = [Image.fromarray(original).resize(size), Image.fromarray(warped_suspect).resize(size),
              overlay(warped_suspect, tiles, g).resize(size)]
    titles = ["registered original (owner only)", "suspect, aligned", "regions"]
    pad, head, foot = 16, 28, 34
    out = Image.new("RGB", (pad + 3 * (size[0] + pad), head + size[1] + foot + pad), (255, 255, 255))
    d = ImageDraw.Draw(out)
    f = _font(14)
    for k, (p, t) in enumerate(zip(panels, titles)):
        x = pad + k * (size[0] + pad)
        out.paste(p, (x, head))
        d.text((x, 6), t, fill=(15, 22, 38), font=f)
    x = pad
    y = head + size[1] + 10
    for key in ("match", "altered", "uncertain", "uncovered"):
        d.rectangle([x, y, x + 14, y + 14], fill=COLORS[key])
        d.text((x + 20, y), LABELS[key], fill=(15, 22, 38), font=f)
        x += 34 + int(d.textlength(LABELS[key], font=f))
    return out


def pdf(report: dict, heat: Image.Image, path: str, include_original: bool = False) -> None:
    """A one-page A4 PDF: what was checked, the verdict, the anchor, and the region map.

    The original panel is cropped out unless include_original is set: the report is meant to be
    shared, the original photo is not.
    """
    W, H = 1240, 1754                          # A4 at 150 dpi
    page = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(page)
    big, mid, small = _font(34), _font(20), _font(16)
    y = 70
    d.text((80, y), "PhotoShield origin report", fill=(15, 22, 38), font=big)
    y += 56
    s = report["summary"]
    verdict = s["verdict"].replace("_", " ")
    col = (22, 163, 74) if s["verdict"] == "DERIVED_FROM_REGISTERED" else (90, 100, 120)
    d.text((80, y), verdict, fill=col, font=mid)
    y += 40
    when = _dt.datetime.fromtimestamp(report["created"], _dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"Created: {when}",
        f"Registered photo: {report['photo']['photoId']}  (manifest digest {report['photo']['manifestDigest'][:24]}...)",
        f"Title: {report['photo']['manifest'].get('title', '')}",
        f"Suspect SHA-256: {report['suspect']['sha256']}",
        f"Alignment: {report['alignment']['method']}, {report['alignment']['inliers']} inlier matches",
        f"Regions ({report['grid']}x{report['grid']}): {s['match']} match, {s['altered']} altered, "
        f"{s['uncertain']} uncertain, {s['uncovered']} not in suspect, {s['flat']} flat",
        f"Thresholds: match <= {report['thresholds']['tau_match']} bits, altered >= "
        f"{report['thresholds']['tau_alter']} bits or a detail shift >= {report['thresholds']['delta_level']} octaves, "
        f"verdict needs >= {report['thresholds']['k_min']} matches "
        f"(from {report['thresholds']['source']})",
        f"Signed by owner key: {str(report.get('pk', 'unsigned'))[:32]}...",
    ]
    for ln in lines:
        d.text((80, y), ln, fill=(52, 62, 82), font=small)
        y += 28
    y += 12
    img = heat
    if not include_original:
        third = (img.width - 16) // 3
        img = img.crop((16 + third, 0, img.width, img.height))
    s_ = min((W - 160) / img.width, 900 / img.height)
    img = img.resize((round(img.width * s_), round(img.height * s_)))
    page.paste(img, (80, y))
    y += img.height + 30
    notes = [
        "How to check this report: python -m photoshield origin-verify origin-report.json SUSPECT --recheck,",
        "or drop origin-report.json into the Photo Shield tab (Origin report > Verify).",
        "Every region claim carries a Merkle proof to the region map inside the manifest, whose digest",
        "is anchored on the ledger. A matching region shows that region existed under the owner's key",
        "when it was anchored. It does not identify who made the suspect image.",
    ]
    for ln in notes:
        d.text((80, y), ln, fill=(90, 100, 120), font=small)
        y += 26
    page.save(path, "PDF", resolution=150.0)
