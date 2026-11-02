"""Revision 2 Phase 5: the tables and figures read only results files, render missing cells as an em dash and list
them on stderr, and every number they show traces back to a results file."""
import csv
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
pytest.importorskip("lifelines")
pytest.importorskip("matplotlib")

from fedverify.analysis import make_fl_healthcare_figures as FIG  # noqa: E402
from fedverify.analysis import make_fl_healthcare_tables as TAB  # noqa: E402
from fedverify.analysis.make_tables import mean_std  # noqa: E402
from fedverify.survival import chain_attacks as CA  # noqa: E402
from fedverify.survival import heart as H  # noqa: E402
from fedverify.survival import runner  # noqa: E402
from fedverify.survival.config import SurvivalConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
needs_heart = pytest.mark.skipif(not H.DEFAULT_DATA.exists(), reason="heart CSV (gitignored) not present")


def test_reference_csvs_are_unchanged_copies_and_reach_the_tables_unchanged():
    for name in ("ieee_table_iii.csv", "ieee_table_iv.csv"):
        assert (REPO / "fedverify/survival/ref" / name).read_bytes() == (REPO / "fedverify/survival" / name).read_bytes()
    spec = TAB.load_spec()
    res = TAB.Results(REPO / "no-such-results", spec)
    t3, t4 = TAB.table_iii(res, spec), TAB.table_iv(res, spec)
    lit3 = TAB.read_csv(REPO / spec["literature"]["table_iii"])
    lit4 = TAB.read_csv(REPO / spec["literature"]["table_iv"])
    assert t3.rows[:len(lit3)] == [[r["ref"], r["dataset"], r["no_cases"], r["no_classes"], r["fl_setup"],
                                    r["bc_consensus"]] for r in lit3]
    assert t4.rows[:len(lit4)] == [list(r.values()) for r in lit4]
    assert all(c == TAB.MISSING for c in t3.rows[-1][1:])            # our rows: no results here, so em dashes


def test_reported_resilience_is_parsed_from_the_reference_row():
    spec = TAB.load_spec()
    text = TAB.read_csv(REPO / spec["literature"]["table_iv"])[-1]["resilience_towards_cyberattacks"]
    got = TAB.reported_resilience(spec)
    assert set(got) == set(spec["chain_attacks"]["attacks"])
    for attack, value in got.items():
        assert value is not None and f"{spec['chain_attacks']['reported_names'][attack]} {value:g}%" in text


def test_latex_and_markdown_escaping():
    assert TAB.tex("a_b & 50% ± ε {x}") == r"a\_b \& 50\% $\pm$ $\varepsilon$ \{x\}"
    assert TAB.tex("—†") == r"---$^{\dagger}$"
    assert TAB.md("a|b") == "a\\|b"


def test_vocab_rows_need_the_hash_the_run_recorded():
    spec = TAB.load_spec()
    import hashlib
    sha = hashlib.sha256((REPO / spec["datasets"]["heart"]["vocab"]).read_bytes()).hexdigest()
    rows = TAB.vocab_rows(spec, "heart", {"vocab_sha256": sha})
    assert isinstance(rows, int) and rows > 0
    assert TAB.vocab_rows(spec, "heart", {"vocab_sha256": "0" * 64}) is None


# ── end to end on a tiny real results tree ───────────────────────────────────
@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    if not H.DEFAULT_DATA.exists():
        pytest.skip("heart CSV (gitignored) not present")
    root = tmp_path_factory.mktemp("results")
    for family, method in (("lr", "fedavg_lr"), ("xgb", "fedxgb_bagging"), ("lr", "chain_fedavg_lr")):
        runner.run(SurvivalConfig(dataset="heart", family=family, method=method, rounds=2, repeats=2, folds=2,
                                  out_dir=str(root)), verbose=False)
    CA.simulate("double_spending", rounds=6, seed=0, out_dir=str(root))
    spec = yaml.safe_load(TAB.SPEC.read_text(encoding="utf-8"))
    spec["expected_repeats"] = 2
    spec_path = root / "table.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return root, spec_path


def test_tables_show_results_values_and_list_missing_cells(tree, capsys):
    root, spec_path = tree
    out = root / "tables"
    assert TAB.main(["--results-dir", str(root), "--spec", str(spec_path), "--out-dir", str(out)]) == 0
    err = capsys.readouterr().err
    text = (out / "methods_heart.md").read_text(encoding="utf-8")
    s = json.loads((root / "heart/fedavg_lr/seed0/summary.json").read_text())
    m, sd = mean_std([rp["pooled"]["accuracy"] for rp in s["repeats"]])
    vanilla = next(line for line in text.splitlines() if line.startswith("| Vanilla FL |"))
    assert vanilla.split(" | ")[1] == f"{m:.3f} ± {sd:.3f}"                 # the number is the results file's
    local = next(line for line in text.splitlines() if line.startswith("| Local |"))
    assert local.split(" | ")[1] == TAB.MISSING                             # not run: an em dash ...
    assert "heart/local_lr/seed0/summary.json" in err                       # ... listed on stderr
    dp = next(line for line in text.splitlines() if line.startswith("| DP-FL"))
    n_metrics = len(yaml.safe_load(spec_path.read_text())["method_tables"]["methods_heart"]["metrics"])
    assert dp.count("n/a") == 2 * n_metrics                                 # RF / XGB DP: n/a by design
    res = (out / "chain_resilience.md").read_text(encoding="utf-8")
    dsum = json.loads((root / "chain_attacks/double_spending/seed0/summary.json").read_text())
    assert f"| Double spending | S_DS | {dsum['resilience_pct']:.1f} |" in res
    assert "\\begin{table" in (out / "methods_heart.tex").read_text(encoding="utf-8")


def test_figures_write_png_pdf_csv_and_captions(tree, capsys):
    root, spec_path = tree
    out = root / "figures"
    assert FIG.main(["--results-dir", str(root), "--spec", str(spec_path), "--out-dir", str(out),
                     "--only", "fig5_accuracy_heart", "fig9_resilience", "fig7_block_times"]) == 0
    err = capsys.readouterr().err
    for ext in ("png", "pdf", "csv"):
        assert (out / f"fig5_accuracy_heart.{ext}").stat().st_size > 0
    rows = list(csv.DictReader(open(out / "fig5_accuracy_heart.csv", encoding="utf-8")))
    row = next(r for r in rows if r["method"] == "fedavg_lr" and r["hospital"] == "Hungary" and r["split"] == "test"
               and r["round"] == "2")
    vals = []
    for p in sorted((root / "heart/fedavg_lr/seed0").glob("repeat*/fold*/rounds.jsonl")):
        rec = [json.loads(line) for line in p.read_text().splitlines()][1]
        vals.append(rec["hospital_metrics"]["Hungary"]["test_acc"])
    assert float(row["mean"]) == pytest.approx(mean_std(vals)[0], abs=1e-9) and int(row["n"]) == len(vals)
    caps = (out / "captions.md").read_text(encoding="utf-8")
    assert "K = 4 hospitals" in caps and "seeds 0, 1" in caps
    assert "chain_blocks.jsonl" in err                                      # no mined run in this tree: listed
    assert not (out / "fig7_block_times.png").exists()


def test_latex_never_contains_unmapped_unicode_or_a_bracket_after_a_line_break():
    """Two bugs a real pdflatex run found: an unmapped δ / − broke compilation, and a row starting with "[19]"
    after \\\\ was read as the line break's optional length."""
    assert TAB.tex("δ = 1e-05, 100 − S") == r"$\delta$ = 1e-05, 100 $-$ S"
    with pytest.raises(ValueError, match="no LaTeX mapping"):
        TAB.tex("Ω")
    assert TAB.tex_line_start("[19]") == "{[19]}" and TAB.tex_line_start("Ours") == "Ours"
    t = TAB.Table("t", "t", "c", ["Ref", "x"], [["[19]", "1"], ["[20]", "2"]], ["[a] note", "b"])
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        t.write(Path(d))
        text = (Path(d) / "t.tex").read_text()
    assert "\n{[19]} & 1" in text and "\n{[20]} & 2" in text and "{[a] note}" in text
    assert all(ord(ch) < 128 for ch in text)
