"""Render the report diagrams (docs/diagrams/*.mmd) to SVG and to PNG at 300 dpi with mermaid-cli (Phase 12).

    python3 -m fedverify.analysis.render_diagrams                    # every .mmd
    python3 -m fedverify.analysis.render_diagrams --browser /usr/bin/google-chrome

Runs `npx -y @mermaid-js/mermaid-cli` (npx fetches it the first time, so that needs network once) with a browser that
is already installed (PUPPETEER_SKIP_DOWNLOAD=1, so puppeteer downloads none; --no-sandbox). The SVG comes first;
the PNG is then drawn at the SVG's natural size (--size = its larger side, so no diagram is shrunk to the default
800-pixel page) and at a device scale of 300/96: one CSS pixel becomes 300/96 pixels, so the image is 300 dpi at the
diagram's natural size (where its 16-pixel labels print at 12 pt), and its pHYs chunk says 300 dpi. White background.

Writes <name>.svg and <name>.png next to each source and docs/diagrams/rendered.json: the SHA-256 of every source,
SVG and PNG, the PNG's size, and the mermaid-cli and browser versions. The report pack and the tests use it to tell a
render that no longer matches its source. If a render fails, the error from mermaid-cli is printed and the exit code
is 1; nothing is written for that diagram.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
DIAGRAMS = REPO / "docs" / "diagrams"
INDEX = "rendered.json"
PACKAGE = "@mermaid-js/mermaid-cli"
DPI = 300
SCALE = DPI / 96                      # CSS pixels are 1/96 inch
BROWSERS = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
# Mermaid settings: a wider side margin for sequence diagrams, whose self-message labels (centred on the first
# lifeline) are otherwise cut off at the left edge. Other diagram types ignore it.
MERMAID_CONFIG = {"sequence": {"diagramMarginX": 150}}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_browser(given: Optional[str]) -> Optional[str]:
    if given:
        return given
    if os.environ.get("FEDVERIFY_BROWSER"):
        return os.environ["FEDVERIFY_BROWSER"]
    return next((p for b in BROWSERS if (p := shutil.which(b))), None)


def mmdc(args: list[str], cfg: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PUPPETEER_SKIP_DOWNLOAD="1")
    mermaid = cfg.with_name("mermaid.json")
    return subprocess.run(["npx", "-y", PACKAGE, *args, "-p", str(cfg), "-c", str(mermaid)], capture_output=True,
                          text=True, env=env, timeout=600)


def set_dpi(png: Path, dpi: int = DPI) -> tuple[int, int]:
    """Rewrite the PNG's pHYs chunk (lossless: same pixels); returns (width, height)."""
    from PIL import Image
    with Image.open(png) as im:
        im.load()
        size = im.size
        im.save(png, dpi=(dpi, dpi))
    return size


def natural_size(svg: Path) -> Optional[tuple[float, float]]:
    """The diagram's own (width, height) in CSS pixels, from the SVG's viewBox."""
    m = re.search(r'viewBox="[-\d.]+ [-\d.]+ ([\d.]+) ([\d.]+)"', svg.read_text(encoding="utf-8")[:4000])
    return (float(m.group(1)), float(m.group(2))) if m else None


def render(src: Path, cfg: Path) -> tuple[Optional[dict], str]:
    """(entry for rendered.json, "") or (None, the reason it failed)."""
    svg, png = src.with_suffix(".svg"), src.with_suffix(".png")
    r = mmdc(["-i", str(src), "-o", str(svg), "-b", "white"], cfg)
    if r.returncode != 0 or not svg.exists():
        return None, (r.stderr or r.stdout).strip() or f"mermaid-cli exited with {r.returncode}"
    size = natural_size(svg)
    page = ["--size", str(math.ceil(max(size)))] if size else []        # the larger side, in CSS pixels
    r = mmdc(["-i", str(src), "-o", str(png), "-b", "white", "-s", f"{SCALE:g}", *page], cfg)
    if r.returncode != 0 or not png.exists():
        return None, (r.stderr or r.stdout).strip() or f"mermaid-cli exited with {r.returncode}"
    w, h = set_dpi(png)
    return {"source_sha256": sha256(src), "svg_sha256": sha256(svg), "png_sha256": sha256(png),
            "natural_css_px": None if size is None else [round(size[0], 1), round(size[1], 1)],
            "png_px": [w, h], "dpi": DPI}, ""


def stale(directory: Path = DIAGRAMS) -> list[str]:
    """Every diagram whose render is missing or no longer matches its source or its recorded files."""
    idx_path = directory / INDEX
    idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {}
    out = []
    for src in sorted(directory.glob("*.mmd")):
        e = (idx.get("diagrams") or {}).get(src.stem)
        svg, png = src.with_suffix(".svg"), src.with_suffix(".png")
        if (not e or not svg.exists() or not png.exists() or e["source_sha256"] != sha256(src)
                or e["svg_sha256"] != sha256(svg) or e["png_sha256"] != sha256(png)):
            out.append(src.stem)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fedverify.analysis.render_diagrams", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DIAGRAMS))
    ap.add_argument("--browser", default=None, help="browser for puppeteer (default: FEDVERIFY_BROWSER, then PATH)")
    a = ap.parse_args(argv)
    directory = Path(a.dir)
    sources = sorted(directory.glob("*.mmd"))
    if not shutil.which("npx"):
        print("cannot render: npx (Node.js) is not on PATH", file=sys.stderr)
        return 1
    browser = find_browser(a.browser)
    if not browser:
        print(f"cannot render: no browser found (tried {', '.join(BROWSERS)}; set FEDVERIFY_BROWSER)", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "puppeteer.json"
        cfg.write_text(json.dumps({"executablePath": browser, "args": ["--no-sandbox", "--disable-gpu"]}),
                       encoding="utf-8")
        cfg.with_name("mermaid.json").write_text(json.dumps(MERMAID_CONFIG), encoding="utf-8")
        v = subprocess.run(["npx", "-y", PACKAGE, "--version"], capture_output=True, text=True, timeout=600,
                           env=dict(os.environ, PUPPETEER_SKIP_DOWNLOAD="1"))
        if v.returncode != 0:
            print(f"cannot render: `npx -y {PACKAGE}` failed: {(v.stderr or v.stdout).strip()}", file=sys.stderr)
            return 1
        bv = subprocess.run([browser, "--version"], capture_output=True, text=True, timeout=60)
        idx = {"generated_by": "python3 -m fedverify.analysis.render_diagrams", "mermaid_cli": v.stdout.strip(),
               "browser": (bv.stdout or bv.stderr).strip(), "png_scale": SCALE, "mermaid_config": MERMAID_CONFIG,
               "diagrams": {}}
        failed = []
        for src in sources:
            entry, why = render(src, cfg)
            if entry is None:
                failed.append(src.stem)
                print(f"[failed] {src.name}: {why}", file=sys.stderr)
                continue
            idx["diagrams"][src.stem] = entry
            print(f"  rendered {src.stem}.svg and {src.stem}.png ({entry['png_px'][0]} x {entry['png_px'][1]} px, "
                  f"{DPI} dpi)")
    (directory / INDEX).write_text(json.dumps(idx, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(idx['diagrams'])} of {len(sources)} diagrams rendered with mermaid-cli {idx['mermaid_cli']} "
          f"({idx['browser']})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
