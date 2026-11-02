"""Manifest schema, canonical digest and photo ids (design §5.1, §5.2)."""
import copy
import json

import pytest

from photoshield.provenance import manifest as M

from .helpers import make_manifest


def test_digest_is_stable_under_key_order_and_whitespace():
    m = make_manifest()
    shuffled = json.loads(json.dumps(m, indent=4, sort_keys=False))
    reordered = dict(reversed(list(shuffled.items())))
    assert M.manifest_digest(reordered) == M.manifest_digest(m)


def test_canonical_json_is_compact_sorted_utf8():
    m = make_manifest(title="Café ☕ photo")
    s = M.canonical_json(m)
    assert ", " not in s and ": " not in s and "Café ☕" in s
    assert s.index('"cloak"') < s.index('"cloaked"') < s.index('"original"') < s.index('"schema"')


def test_any_float_is_rejected_because_js_would_print_it_differently():
    m = make_manifest()
    m["cloak"]["eps"] = 8.0
    with pytest.raises(M.ManifestError, match="floats"):
        M.validate(m)


def test_any_parameter_change_changes_the_digest():
    a, b = make_manifest(), make_manifest(eps=4)
    assert M.manifest_digest(a) != M.manifest_digest(b)


@pytest.mark.parametrize("mutate, msg", [
    (lambda m: m["original"].__setitem__("sha256", "ABC"), "sha256"),
    (lambda m: m["original"].__setitem__("phash", "z" * 16), "phash"),
    (lambda m: m.__setitem__("title", "x" * 121), "title"),
    (lambda m: m.__setitem__("title", "<script>alert(1)</script>"), "title"),
    (lambda m: m.__setitem__("title", ""), "title"),
    (lambda m: m.__setitem__("schema", "other@1"), "schema"),
    (lambda m: m.__setitem__("extra", 1), "unknown"),
    (lambda m: m["original"].pop("width"), "width"),
    (lambda m: m["original"].__setitem__("width", 16), "width"),
    (lambda m: m["cloaked"].__setitem__("format", "jpg"), "format"),
    (lambda m: m["cloak"].__setitem__("psnr", "high"), "psnr"),
    (lambda m: m["cloak"].__setitem__("evil", "x"), "unknown"),
    (lambda m: m.__setitem__("cloaked", None), "cloak parameters given"),
])
def test_rejects_malformed_manifests(mutate, msg):
    m = copy.deepcopy(make_manifest())
    mutate(m)
    with pytest.raises(M.ManifestError, match=msg):
        M.validate(m)


def test_provenance_only_manifest_is_valid():
    m = make_manifest(cloaked=False)
    assert m["cloaked"] is None and m["cloak"] is None
    assert len(M.manifest_digest(m)) == 64


def test_photo_id_is_deterministic_and_short():
    d = M.manifest_digest(make_manifest())
    assert M.photo_id(d) == M.photo_id(d) and len(M.photo_id(d)) == 16


def test_metric_formatting():
    assert M.metric(34.123456) == "34.1235" and M.metric(float("inf")) == "inf"
    assert M.metric(None) is None
