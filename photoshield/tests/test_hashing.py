"""pHash spec (design §7): Python vectors, edge cases, and bit-identity with the shipped JS."""
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from photoshield.provenance import hashing
from photoshield.provenance.hashing import Unhashable, hamming, phash, sha256_file

from .conftest import REPO, VECTORS, extract_js

with open(os.path.join(VECTORS, "phash_vectors.json")) as _f:
    VEC = json.load(_f)


@pytest.mark.parametrize("name", sorted(VEC["vectors"]))
def test_python_matches_committed_vectors(name):
    assert phash(os.path.join(VECTORS, name)) == VEC["vectors"][name]


def test_vectors_cover_the_awkward_cases():
    names = set(VEC["vectors"])
    for needle in ("alpha", "min_32x32", "tall", "wide", "palette", "noise_gray", "photo_"):
        assert any(needle in n for n in names), needle
    assert len(names) >= 20


@pytest.mark.parametrize("name", VEC["unhashable"])
def test_images_below_32px_are_unhashable(name):
    with pytest.raises(Unhashable):
        phash(os.path.join(VECTORS, name))


def test_alpha_composites_onto_white_with_the_integer_rule():
    rgba = np.array([[[0, 0, 0, 0], [0, 0, 0, 255], [200, 100, 50, 128], [255, 255, 255, 1]]],
                    dtype=np.uint8)
    rgb = hashing.composite_rgba(rgba)[0].tolist()
    assert rgb[0] == [255, 255, 255]          # fully transparent -> white
    assert rgb[1] == [0, 0, 0]                # opaque unchanged
    assert rgb[2] == [(c * 128 + 255 * 127 + 127) // 255 for c in (200, 100, 50)]


def test_output_is_16_lowercase_hex_with_dc_bit_zero(vectors_dir):
    h = phash(os.path.join(vectors_dir, "photo_astronaut.png"))
    assert len(h) == 16 and h == h.lower()
    assert int(h, 16) >> 63 == 0              # bit (0,0) is the MSB and is fixed at 0
    assert bin(int(h, 16)).count("1") <= 32   # at most 32 of 63 coefficients exceed the median


def test_reencoding_moves_the_hash_only_a_little(tmp_path, vectors_dir):
    src = os.path.join(vectors_dir, "photo_chelsea.png")
    img = Image.open(src).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    d = hamming(phash(src), phash(buf.getvalue()))
    assert d <= 8, d                           # benign re-encode, not an unrelated image
    other = phash(os.path.join(vectors_dir, "photo_rocket.png"))
    assert hamming(phash(src), other) > d


def test_sha256_matches_hashlib(vectors_dir):
    p = os.path.join(vectors_dir, "checker_64x64.png")
    assert sha256_file(p) == hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_hamming():
    assert hamming("0" * 16, "f" * 16) == 64
    assert hamming("8000000000000001", "0000000000000000") == 2


def test_dct_table_is_what_the_generator_emits():
    from photoshield.provenance import gen_dct_table
    src = open(os.path.join(REPO, "photoshield", "provenance", "dct_table.py")).read()
    assert src == gen_dct_table.python_source(), "dct_table.py was edited by hand; regenerate it"


# ── bit-identity with the shipped browser code ───────────────────────────────
def _js_source():
    html = open(os.path.join(REPO, "frontend", "index.html"), encoding="utf-8").read()
    assert "function psPhashFromRGBA" in html, "the pHash must ship in frontend/index.html"
    return extract_js("PS_DCT_COS", "PS_DCT_S", "psComposite", "psPhashFromRGBA", "psHamming")


def test_js_dct_table_literals_equal_the_python_table():
    from photoshield.provenance import gen_dct_table
    js = _js_source()
    want = gen_dct_table.js_source()
    assert want.split("\n")[0] in js and all(line in js for line in want.split("\n"))


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_browser_phash_is_bit_identical_on_every_vector(tmp_path):
    items = []
    for name in sorted(VEC["vectors"]) + VEC["unhashable"]:
        img = Image.open(os.path.join(VECTORS, name))
        img = img.convert("RGBA")                      # canvas getImageData is RGBA
        arr = np.asarray(img)
        items.append({"name": name, "w": arr.shape[1], "h": arr.shape[0],
                      "rgba": base64.b64encode(arr.tobytes()).decode()})
    src, pix = tmp_path / "ps.js", tmp_path / "pix.json"
    src.write_text(_js_source())
    pix.write_text(json.dumps(items))
    r = subprocess.run(["node", os.path.join(REPO, "photoshield", "tests", "phash_js_test.mjs"),
                        str(src), str(pix)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    for name, want in VEC["vectors"].items():
        assert got[name] == want, f"{name}: browser {got[name]} != python {want}"
    for name in VEC["unhashable"]:
        assert got[name].startswith("ERROR: UNHASHABLE")
