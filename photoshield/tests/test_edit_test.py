"""P7: the E4/E5 logic the Colab notebook drives, with stand-in editors and scorers."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from photoshield.analysis import make_tables  # noqa: E402
from photoshield.cloak.encoders import load_encoder  # noqa: E402
from photoshield.experiments import edit_test as E  # noqa: E402


def _imgs():
    rng = np.random.default_rng(0)
    return [(f"im{k}", rng.integers(0, 256, (72, 80, 3), dtype=np.uint8)) for k in range(2)]


def _editor(rgb, prompt, seed):          # deterministic stand-in for InstructPix2Pix
    return np.clip(rgb.astype(np.int16) // 2 + len(prompt) + seed, 0, 255).astype(np.uint8)


def _lpips(a, b):
    return float(np.abs(a.astype(float) - b.astype(float)).mean() / 255)


def _clip(src, edit, prompt):
    return float(np.corrcoef(src.ravel(), edit.ravel())[0, 1])


def _strip(p):
    return [{k: v for k, v in json.loads(l).items() if not k.endswith("_wall_s")} for l in open(p)]


def test_cloak_pair_respects_eps_and_shape():
    enc = load_encoder("tiny")
    a = _imgs()[0][1]
    adv, noise = E.cloak_pair(a, enc, 8, 3, 0)
    assert adv.shape == noise.shape == a.shape
    assert np.abs(adv.astype(int) - a).max() <= 8 and np.abs(noise.astype(int) - a).max() <= 8


def test_e4_layout_determinism_and_grid(tmp_path):
    enc = load_encoder("tiny")
    kw = dict(eps=4, steps=2, size=64, prompts=("add snow", "make it red"), edit_seeds=(0, 1),
              encoder_name="tiny", editor_info={"pipeline": "stub", "repo": "none"})
    da = E.run_e4(_imgs(), enc, _editor, _lpips, _clip, results=str(tmp_path / "a"), **kw)
    db = E.run_e4(_imgs(), enc, _editor, _lpips, _clip, results=str(tmp_path / "b"), **kw)
    ra, rb = _strip(f"{da}/records.jsonl"), _strip(f"{db}/records.jsonl")
    assert ra == rb and len(ra) == 2 * 2 * 2
    assert (tmp_path / "a" / "e4" / "tiny_eps4_steps2_size64" / "seed0" / "grids" / "im0.png").exists()
    cfg = json.load(open(f"{da}/config.json"))
    assert cfg["prompts"] == ["add snow", "make it red"] and cfg["editor"]["pipeline"] == "stub"
    md = make_tables.render(str(tmp_path / "a"))
    assert "bootstrap 95% CI" in md and "stub" in md


def test_e5_reports_source_and_targets(tmp_path):
    src = load_encoder("tiny")
    d = E.run_e5(_imgs(), src, {"tiny2": load_encoder("tiny", seed=99)}, eps=4, steps=2, size=64,
                 results=str(tmp_path), source_name="tiny")
    recs = [json.loads(l) for l in open(f"{d}/records.jsonl")]
    assert {r["target"] for r in recs} == {"tiny", "tiny2"}
    assert all(r["is_source"] == (r["target"] == "tiny") for r in recs)
    assert "(attacked)" in make_tables.render(str(tmp_path))


def test_bootstrap_ci_verdicts():
    m, lo, hi = make_tables.bootstrap_ci([1.0, 1.1, 0.9, 1.2, 1.0, 0.95])
    assert lo > 0 and lo <= m <= hi
    assert make_tables._ci_cells([1.0, 1.1, 0.9, 1.2])[2] == "degrades"
    assert "includes 0" in make_tables._ci_cells([-1.0, 1.0, -0.5, 0.5])[2]
    assert make_tables.bootstrap_ci([1.0]) is None
    assert make_tables.bootstrap_ci([0.1, 0.2, 0.3]) == make_tables.bootstrap_ci([0.1, 0.2, 0.3])


def test_notebook_is_unexecuted_and_drives_tested_code():
    import os
    p = os.path.join(os.path.dirname(E.__file__), "colab_edit_test.ipynb")
    nb = json.load(open(p))
    assert nb["nbformat"] == 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in code)
    src = "".join("".join(c["source"]) for c in code)
    for name in ("run_e4", "run_e5", "instruct_pix2pix", "lpips_scorer", "clip_directional",
                 "make_tables", "use_deterministic_algorithms"):
        assert name in src
    for fn in ("run_e4", "run_e5", "instruct_pix2pix", "lpips_scorer", "clip_directional"):
        assert callable(getattr(E, fn))
