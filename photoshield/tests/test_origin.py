"""PhotoShield v2: region maps and origin reports (design §15)."""
import copy
import io
import json

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("cv2")

import skimage.data as sd  # noqa: E402
from PIL import Image  # noqa: E402

from photoshield.origin import align as A, report as OR  # noqa: E402
from photoshield.provenance import hashing as H, manifest as M, regions as R, signing as S  # noqa: E402

T = {"tau_match": 10, "tau_alter": 22, "k_min": 8, "source": "test"}   # test values, not calibration


def png(a):
    b = io.BytesIO()
    Image.fromarray(a).save(b, "PNG")
    return b.getvalue()


def jpg(a, q=80):
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q)
    return b.getvalue()


@pytest.fixture(scope="module")
def reg():
    orig = sd.astronaut()
    ob = png(orig)
    man = M.build("astronaut", {"sha256": H.sha256_bytes(ob), "phash": H.phash_from_rgb(orig),
                                "width": 512, "height": 512}, regions=R.region_map(orig))
    sk, pk = S.new_key(b"\x07" * 32)
    return {"orig": orig, "ob": ob, "man": man, "sk": sk, "pk": pk}


def edited(orig, rotate=3):
    fake = orig.copy()
    fake[40:200, 150:320] = sd.coffee()[100:260, 200:370]
    im = Image.fromarray(fake).resize((410, 410)).rotate(rotate, resample=Image.BILINEAR)
    return jpg(np.asarray(im.crop((10, 10, 400, 400))))


# ── region maps ──────────────────────────────────────────────────────────────
def test_grid_rule_and_tiles_partition_the_image():
    assert R.grid_for(512, 512) == 8 and R.grid_for(100, 400) == 3 and R.grid_for(63, 500) == 0
    boxes = R.tile_boxes(517, 389, 8)
    cover = np.zeros((389, 517), int)
    for _, _, y0, y1, x0, x1 in boxes:
        cover[y0:y1, x0:x1] += 1
    assert (cover == 1).all() and len(boxes) == 64


def test_region_map_is_deterministic_and_proofs_fold(reg):
    g, vals = R.region_values(reg["orig"])
    assert g == 8 and len(vals) == 64 and all(len(v) == 18 and v[16] == ":" for v in vals)
    root, leaves = R.commit(vals, g)
    assert root == reg["man"]["regions"]["root"] == R.region_map(reg["orig"])["root"]
    from photoshield.provenance.merkle import verify_proof
    for i, (p, v) in enumerate(zip(R.proofs(leaves), vals)):
        assert verify_proof(R.leaf(i, R.names(g)[i], v), p, root)


def test_flat_tiles_are_flagged():
    img = np.full((256, 256, 3), 128, np.uint8)
    img[:, 128:] = sd.astronaut()[:256, :128]
    _, vals = R.region_values(img)
    flags = [v.endswith(":1") for v in vals]
    assert not any(flags[r * 8 + c] for r in range(8) for c in range(4))     # the grey half
    assert sum(flags[r * 8 + c] for r in range(8) for c in range(4, 8)) >= 28


def test_manifest_regions_field(reg):
    old = M.build("x", reg["man"]["original"])
    assert "regions" not in old                       # v1 manifests keep their exact digest
    for bad in ({"algo": "x", "grid": 8, "n": 64, "root": "0" * 64},
                {"algo": R.ALGO, "grid": 8, "n": 63, "root": "0" * 64},
                {"algo": R.ALGO, "grid": 8, "n": 64, "root": "XYZ"},
                {"algo": R.ALGO, "grid": 8, "n": 64, "root": "0" * 64, "extra": 1}):
        m = dict(old, regions=bad)
        with pytest.raises(M.ManifestError):
            M.validate(m)


# ── alignment ────────────────────────────────────────────────────────────────
def test_alignment_recovers_a_resized_rotated_crop(reg):
    sus = H.decode_rgb(edited(reg["orig"]))
    al = A.align(reg["orig"], sus)
    assert al["method"] == "orb-ransac" and al["inliers"] >= A.MIN_INLIERS
    scale = float(np.sqrt(np.linalg.det(al["M"][:, :2])))
    assert abs(scale - 512 / 410) < 0.03
    m = A.matrix_from_strings(A.matrix_strings(al["M"]))
    assert np.allclose(m, al["M"], atol=1e-6)


def test_alignment_refuses_unrelated_different_shape(reg):
    with pytest.raises(A.AlignError):
        A.align(reg["orig"], sd.chelsea())


# ── reports ──────────────────────────────────────────────────────────────────
def test_edited_copy_localises_the_change_and_verifies(reg):
    out = OR.build_report(reg["ob"], edited(reg["orig"]), reg["man"], T, reg["sk"], created=1)
    rep = out["report"]
    s = rep["summary"]
    assert s["verdict"] == "DERIVED_FROM_REGISTERED" and s["match"] >= 20
    st = {t["name"]: t["status"] for t in rep["tiles"]}
    assert st["r1c3"] == "altered" and st["r2c3"] == "altered"         # inside the pasted patch
    assert st["r5c3"] == "match" and st["r4c5"] == "match"             # untouched body
    v = OR.verify_report(rep, edited(reg["orig"]), recheck=True,
                         anchored={"manifestDigest": M.manifest_digest(reg["man"]), "ownerPk": reg["pk"]})
    assert v["ok"], [c for c in v["checks"] if not c["ok"]]
    assert "0." in rep["tiles"][0]["coverage"] or "1." in rep["tiles"][0]["coverage"]
    json.dumps(rep)
    M.canonical_json(OR.core(rep))                                     # no floats anywhere


def test_unrelated_photo_is_not_shown(reg):
    other = np.ascontiguousarray(sd.chelsea()[:300, :300])
    rep = OR.build_report(reg["ob"], png(other), reg["man"], T, created=1, mode="resize")["report"]
    assert rep["summary"]["verdict"] == "NOT_SHOWN"
    assert rep["summary"]["match"] < T["k_min"]


def test_report_is_deterministic(reg):
    a = OR.build_report(reg["ob"], edited(reg["orig"]), reg["man"], T, reg["sk"], created=5)["report"]
    b = OR.build_report(reg["ob"], edited(reg["orig"]), reg["man"], T, reg["sk"], created=5)["report"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.mark.parametrize("tamper", ["value", "status", "summary", "manifest", "suspect", "unsigned", "key"])
def test_tampering_is_detected(reg, tamper):
    rep = OR.build_report(reg["ob"], edited(reg["orig"]), reg["man"], T, reg["sk"], created=1)["report"]
    r = copy.deepcopy(rep)
    if tamper == "value":                  # claim a different committed hash for a tile
        t = r["tiles"][20]
        t["value"] = ("0" * 16) + t["value"][16:]
    elif tamper == "status":
        t = next(t for t in r["tiles"] if t["status"] == "altered")
        t["status"] = "match"
    elif tamper == "summary":
        r["summary"]["match"] += 5
    elif tamper == "manifest":
        r["photo"]["manifest"]["title"] = "someone else"
    elif tamper == "suspect":
        pass
    elif tamper == "unsigned":
        r.pop("sig")
    elif tamper == "key":
        r = OR.sign(r, S.new_key(b"\x09" * 32)[0])
    anchored = {"manifestDigest": M.manifest_digest(reg["man"]), "ownerPk": reg["pk"]}
    sus = png(reg["orig"]) if tamper == "suspect" else None
    v = OR.verify_report(r, sus, anchored=anchored)
    assert not v["ok"]


def test_v1_registration_and_wrong_original_are_refused(reg):
    v1 = M.build("x", reg["man"]["original"])
    with pytest.raises(OR.OriginError, match="no region map"):
        OR.build_report(reg["ob"], edited(reg["orig"]), v1, T)
    with pytest.raises(OR.OriginError, match="SHA-256"):
        OR.build_report(png(sd.coffee()), edited(reg["orig"]), reg["man"], T)


def test_thresholds_come_from_calibration(tmp_path):
    with pytest.raises(OR.OriginError, match="study e8"):
        OR.load_thresholds(str(tmp_path / "none.json"))
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"tau_near": 1, "tau_far": 2, "tile": {"tau_match": 9, "tau_alter": 20, "k_min": 6}}))
    assert OR.load_thresholds(str(p))["k_min"] == 6


def test_cli_origin_end_to_end(reg, tmp_path):
    from photoshield import cli
    from photoshield.chain.client import keygen
    (tmp_path / "orig.png").write_bytes(reg["ob"])
    (tmp_path / "fake.jpg").write_bytes(edited(reg["orig"]))
    key = str(tmp_path / "k.pskey")
    keygen(key)
    (tmp_path / "manifest.json").write_text(json.dumps(reg["man"]))
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"tile": {k: T[k] for k in ("tau_match", "tau_alter", "k_min")}}))
    out = tmp_path / "rep"
    assert cli.main(["origin", str(tmp_path / "orig.png"), str(tmp_path / "fake.jpg"), "--manifest",
                     str(tmp_path / "manifest.json"), "--out", str(out), "--key", key,
                     "--calibration", str(cal)]) == 0
    assert (out / "origin-report.json").exists() and (out / "heatmap.png").exists()
    assert (out / "origin-report.pdf").read_bytes()[:4] == b"%PDF"
    assert cli.main(["origin-verify", str(out / "origin-report.json"), str(tmp_path / "fake.jpg"),
                     "--recheck"]) == 0
