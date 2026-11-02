"""PhotoShield v2 in the browser: the shipped JS region map and origin reports agree with Python.

Runs the functions extracted verbatim from frontend/index.html under Node 22:
- region values (tile pHash, flat flag, detail level) bit-identical to photoshield.provenance.regions
- Merkle root and inclusion proofs identical to backend/app.py
- a Python-built, Python-signed report verifies in the browser, and a browser-built,
  browser-signed report verifies in Python
"""
import base64
import io
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("cv2")

import skimage.data as sd  # noqa: E402
from PIL import Image  # noqa: E402

from photoshield.origin import report as OR  # noqa: E402
from photoshield.provenance import hashing as H, manifest as M, regions as R, signing as S  # noqa: E402

from .conftest import REPO, extract_js  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")

NAMES = ("PS_DCT_COS", "PS_DCT_S", "psComposite", "psPhashFromRGBA", "psHamming",
         "PS_FLAT_RANGE", "PS_LEVEL_BASE", "PS_REGIONS_ALGO", "PS_ORIGIN_SCHEMA", "PS_ORIGIN",
         "psGridFor", "psTileBoxes", "psTileNames", "psCropRGBA", "psBinMeans", "psTotalVariation",
         "psDetailLevel", "psTileValue", "psRegionValues", "psMerkleLevelUp", "psRegionCommit",
         "psMerkleProof", "psSplitValue", "psTileStatus", "psOriginSummary", "psOriginMessage",
         "psVerifyOriginCore", "psBuildOriginReport",
         "PS_REGISTER", "PS_PKCS8_PREFIX", "_psWebEd", "sha256Fallback", "bvSha256Hex", "bvMerkleLeaf",
         "bvFoldProof", "psHex", "psUnhex", "psB64", "psUnb64", "psCanonical", "psDigestOf", "psPhotoIdOf", "psSignRaw",
         "psVerifyRaw", "psHasWebEd25519", "psKeyFromPkcs8")
T = {"tau_match": 21, "tau_alter": 25, "delta_level": 2, "k_min": 3, "source": "test"}
SEED = b"\x05" * 32


def _node(body: str, tmp_path):
    js = extract_js(*NAMES).replace("let _psWebEd", "var _psWebEd")
    src = tmp_path / "t.mjs"
    src.write_text("globalThis.window = globalThis;\n" + js + "\n" + body)
    r = subprocess.run(["node", str(src)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _rgba(a):
    a = np.asarray(a)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], 2)
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def _images():
    vec = os.path.join(REPO, "photoshield", "tests", "vectors")
    out = []
    for f in sorted(os.listdir(vec)):
        if f.endswith(".png") and not f.startswith("unhashable"):
            img = Image.open(os.path.join(vec, f))
            out.append((f, np.asarray(img.convert("RGBA"))))
    out.append(("astronaut", sd.astronaut()))
    out.append(("coffee_517x389", np.ascontiguousarray(sd.coffee()[:389, :517])))
    return out


def png(a):
    b = io.BytesIO()
    Image.fromarray(a).save(b, "PNG")
    return b.getvalue()


def test_js_region_maps_are_bit_identical_to_python(tmp_path):
    imgs = _images()
    items = [{"name": n, "w": int(a.shape[1]), "h": int(a.shape[0]), "rgba": _rgba(a)} for n, a in imgs]
    got = _node("""(async () => {
      const items = %s, out = {};
      for (const it of items) {
        const rgba = new Uint8ClampedArray(Buffer.from(it.rgba, 'base64'));
        const rv = psRegionValues(it.w, it.h, rgba);
        let root = null, proofs = null;
        if (rv.grid) { const c = await psRegionCommit(rv.values, rv.grid); root = c.root;
          proofs = [await psMerkleProof(c.leaves, 0), await psMerkleProof(c.leaves, c.leaves.length - 1)]; }
        out[it.name] = { grid: rv.grid, values: rv.values, root, proofs };
      }
      console.log(JSON.stringify(out));
    })();""" % json.dumps(items), tmp_path)
    for n, a in imgs:
        g, vals = R.region_values(H.decode_rgb(a))
        assert got[n]["grid"] == g, n
        assert got[n]["values"] == vals, n
        if g:
            root, leaves = R.commit(vals, g)
            assert got[n]["root"] == root, n
            pr = R.proofs(leaves)
            assert got[n]["proofs"] == [pr[0], pr[-1]], n


@pytest.fixture(scope="module")
def reg():
    orig = sd.astronaut()
    ob = png(orig)
    man = M.build("astronaut", {"sha256": H.sha256_bytes(ob), "phash": H.phash_from_rgb(orig),
                                "width": 512, "height": 512}, regions=R.region_map(orig))
    fake = orig.copy()
    fake[40:200, 150:320] = sd.coffee()[100:260, 200:370]
    b = io.BytesIO()
    Image.fromarray(fake).resize((400, 400)).save(b, "JPEG", quality=80)
    return {"orig": orig, "ob": ob, "man": man, "fb": b.getvalue()}


def test_python_report_verifies_in_the_browser(reg, tmp_path):
    sk, pk = S.new_key(SEED)
    rep = OR.build_report(reg["ob"], reg["fb"], reg["man"], T, sk, created=7)["report"]
    bad = json.loads(json.dumps(rep))
    bad["tiles"][3]["value"] = "0" * 16 + bad["tiles"][3]["value"][16:]
    got = _node("""(async () => {
      const a = await psVerifyOriginCore(%s), b = await psVerifyOriginCore(%s);
      console.log(JSON.stringify({ a: a.ok, ac: a.checks, b: b.ok, digest: a.digest }));
    })();""" % (json.dumps(rep), json.dumps(bad)), tmp_path)
    assert got["a"], [c for c in got["ac"] if not c["ok"]]
    assert got["digest"] == M.manifest_digest(reg["man"])
    assert got["b"] is False


def test_browser_report_verifies_in_python(reg, tmp_path):
    """The browser's resize path: same values the page computes, signed by a browser key."""
    g, vals = R.region_values(reg["orig"])
    _, leaves = R.commit(vals, g)
    sus = np.asarray(Image.open(io.BytesIO(reg["fb"])).convert("RGB").resize((512, 512), Image.BILINEAR))
    svals = [R.tile_value(sus[y0:y1, x0:x1]) for _, _, y0, y1, x0, x1 in R.tile_boxes(512, 512, g)]
    pkcs8 = "302e020100300506032b657004220420" + SEED.hex()
    got = _node("""(async () => {
      const key = await psKeyFromPkcs8(psUnhex('%s'));
      const r = await psBuildOriginReport(%s, %s, %s, %s, {sha256: '%s', width: 400, height: 400},
                                          [512/400, 0, 0, 0, 512/400, 0], 'resize', %s, 9);
      r.pk = key.pkHex;
      r.sig = await psSignRaw(key, await psOriginMessage(r));
      console.log(JSON.stringify(r));
    })();""" % (pkcs8, json.dumps(reg["man"]), json.dumps(vals), json.dumps(leaves), json.dumps(svals),
               H.sha256_bytes(reg["fb"]), json.dumps(T)), tmp_path)
    _, pk = S.new_key(SEED)
    v = OR.verify_report(got, reg["fb"], anchored={"manifestDigest": M.manifest_digest(reg["man"]), "ownerPk": pk})
    assert v["ok"], [c for c in v["checks"] if not c["ok"]]
    assert got["summary"]["verdict"] == "DERIVED_FROM_REGISTERED"
    assert got["alignment"]["matrix"] == ["1.280000", "0.000000", "0.000000", "0.000000", "1.280000", "0.000000"]
