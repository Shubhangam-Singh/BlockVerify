"""Phase 11: the LaTeX check, the report pack and the results manifest."""
import json
import shutil
from pathlib import Path

import pytest

from fedverify.analysis import latex_check as LC  # noqa: E402
from fedverify.analysis import make_manifest as MM  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "fedverify" / "results"


# ── the manifest ─────────────────────────────────────────────────────────────
def _tree(root: Path) -> None:
    (root / "tables").mkdir(parents=True)
    (root / "tables" / "a.md").write_text("| x |\n|---|\n| 1 |\n")
    (root / "figures").mkdir()
    (root / "figures" / "f.png").write_bytes(bytes(range(200)))
    (root / "demo" / "heart_ours_lr_demo" / "seed0").mkdir(parents=True)
    (root / "demo" / "heart_ours_lr_demo" / "seed0" / "out.txt").write_text("rewritten by every demo run")


def test_manifest_detects_every_kind_of_change(tmp_path):
    _tree(tmp_path)
    m = MM.build(tmp_path, freeze=False)
    assert m["file_count"] == 2 and not any(f["path"].startswith("demo/") for f in m["files"])   # demo left out
    assert MM.verify(tmp_path) == []                                       # intact right after the freeze
    (tmp_path / "demo" / "heart_ours_lr_demo" / "seed0" / "out.txt").write_text("a new demo run")
    assert MM.verify(tmp_path) == []                                       # the live demo may change
    data = bytearray((tmp_path / "figures" / "f.png").read_bytes())
    data[100] ^= 1
    (tmp_path / "figures" / "f.png").write_bytes(bytes(data))              # one bit of one file
    (tmp_path / "tables" / "new.md").write_text("added later")
    (tmp_path / "tables" / "a.md").unlink()
    problems = MM.verify(tmp_path)
    assert "changed: figures/f.png" in problems and "not in the manifest: tables/new.md" in problems
    assert "missing: tables/a.md" in problems


def test_manifest_block_cannot_be_rewritten(tmp_path):
    _tree(tmp_path)
    MM.build(tmp_path, freeze=False)
    chain = json.loads((tmp_path / MM.CHAIN).read_text())
    m = json.loads((tmp_path / MM.MANIFEST).read_text())
    assert MM.check_chain(chain, m["merkle_root"], m["leaf_count"]) == []
    forged = json.loads(json.dumps(chain))
    tx = next(t for t in forged["blocks"][-1]["transactions"] if t["type"] == "fl_round_commit")
    tx["root"] = "0" * 64                                                 # a different root in the same block
    errors = MM.check_chain(forged, m["merkle_root"], m["leaf_count"])
    assert any("does not match its contents" in e for e in errors)
    assert any("does not hold this manifest's Merkle root" in e for e in errors)


@pytest.mark.skipif(not (RESULTS / MM.MANIFEST).exists(), reason="no MANIFEST.json (run make_manifest)")
def test_the_frozen_results_match_their_manifest():
    problems = MM.verify(RESULTS)
    assert problems == [], problems[:10]
    m = json.loads((RESULTS / MM.MANIFEST).read_text())
    assert m["requirements_lock"] and (RESULTS / MM.LOCK).exists() and m["python"]


# ── LaTeX ────────────────────────────────────────────────────────────────────
def test_mnist_tex_escapes_percent_and_maps_symbols():
    from fedverify.analysis.make_tables import tex_table
    t = tex_table("t", ["ε", "a"], [["1 ± 2", "✓"]], "Final test accuracy (%) — α = 0.1")
    assert "\\caption{Final test accuracy (\\%) --- $\\alpha$ = 0.1}" in t
    assert "$\\varepsilon$" in t and "$\\pm$" in t and "$\\surd$" in t and all(ord(c) < 128 for c in t)


def test_latex_parts_never_drop_a_column(tmp_path):
    from fedverify.analysis.make_fl_healthcare_tables import Table
    bad = Table("t", "t", "c", ["M", "a", "b", "c"], [["x", "1", "2", "3"]], parts=[{"suffix": "", "columns": [1, 2]}])
    with pytest.raises(ValueError, match="drop columns"):
        bad.write(tmp_path)
    good = Table("t", "t", "c", ["M", "a", "b", "c"], [["x", "1", "2", "3"]],
                 parts=[{"suffix": "", "columns": [1, 2]}, {"suffix": "_b", "columns": [2, 3]}])
    good.write(tmp_path)
    tex = (tmp_path / "t.tex").read_text()
    assert tex.count("\\begin{table}") == 2 and "tab:t_b" in tex and (tmp_path / "t.csv").read_text().count(",") == 6


def test_every_generated_tex_table_compiles(tmp_path):
    engine = LC.find_engine(None)
    last = RESULTS / "tables" / "latex_check.json"
    if engine is None:
        note = ""
        if last.exists():
            r = json.loads(last.read_text())
            ok = sum(t["compiled"] and not t["overfull_pt"] for t in r["tables"].values())
            note = f" The last recorded check ({r['engine']}) compiled {ok} of {len(r['tables'])} tables cleanly."
        pytest.skip("No TeX engine (tectonic or pdflatex on PATH, or FEDVERIFY_TEX): the tables were not compiled "
                    "in this run." + note)
    if not (RESULTS / "tables").exists():
        pytest.skip("no generated tables (gitignored results)")
    (tmp_path / "tables").mkdir()
    for t in (RESULTS / "tables").glob("*.tex"):
        shutil.copy(t, tmp_path / "tables" / t.name)
    assert LC.main(["--results-dir", str(tmp_path), "--engine", engine]) == 0
    report = json.loads((tmp_path / "tables" / "latex_check.json").read_text())
    assert all(r["compiled"] and not r["overfull_pt"] for r in report["tables"].values())
    assert not any(r["flag_split"] for r in report["tables"].values())       # nothing shrunk below 75%


# ── the report pack ──────────────────────────────────────────────────────────
@pytest.mark.skipif(not (RESULTS / "report_pack" / "README.md").exists(), reason="no report pack built")
def test_report_pack_holds_every_table_and_300_dpi_figures():
    from fedverify.analysis.make_report_pack import png_dpi
    pack = RESULTS / "report_pack"
    names = {p.name.split("_", 1)[1].rsplit(".", 1)[0] for p in (pack / "tables").glob("*.md")}
    assert {p.stem for p in (RESULTS / "tables").glob("*.md")} <= names
    for p in (pack / "tables").glob("*.md"):
        assert (pack / "tables" / (p.name[:-3] + ".csv")).exists()
    pngs = list((pack / "figures").glob("*.png"))
    assert pngs and all(png_dpi(p) == 300 for p in pngs)
