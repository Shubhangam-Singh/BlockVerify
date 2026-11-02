"""Phase 12: the wording fixes (each made in a generator), the within-hospital key comparisons, the rendered diagrams
and the MNIST tables whose experiments were never run."""
import json
import re
from pathlib import Path

import pytest

from fedverify.analysis import make_results_draft as DR
from fedverify.analysis import make_tables as MT

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "fedverify" / "results"
have_results = pytest.mark.skipif(not (RESULTS / "statistics" / "paired_bootstrap.json").exists()
                                  or not (RESULTS / "tables" / "paired_comparisons_within.md").exists(),
                                  reason="results (gitignored) not built")


# ── a stand-in for Results: only what key_comparisons and central_gap_clause read ──────────────────────────────
class FakeStats:
    def __init__(self, cis):
        self.cis = cis                                  # {(a, b, metric): (difference, ci_low, ci_high)}

    def compare(self, ds, a, b, metric, hospital=None):
        d = self.cis.get((a, b, metric))
        return None if d is None else {"diff_mean": d[0], "diff_std": 0.001, "ci": [d[1], d[2]],
                                       "per_repeat_diff": [d[0]] * 5, "a_mean": 0.6, "b_mean": 0.6 - d[0]}


class FakeRes:
    def __init__(self, cis, fer=None, diag=None, curves=None):
        self.paired, self.root = FakeStats(cis), Path("/nonexistent")
        self.fer, self.diag, self.curves, self.missing = fer or {}, diag, curves or {}, {}

    def load(self, path):
        return self.diag if path.name.startswith("central_xgb_") else None

    def summary(self, ds, cell, where):
        if cell in self.fer:
            return {"attack": {"detection_pooled_over_repeats": {"false_exclusion_rate": self.fer[cell]}}}
        if cell in self.curves:
            return {"repeats": [{"pooled_by_round": {"c_index": c}} for c in self.curves[cell]]}
        return None

    def note_missing(self, what, where):
        self.missing[str(what)] = where


SPEC = {"clean_false_exclusion_max": 0.05,
        "metric_labels": {"c_index": "C-index", "c_index_within": "C-index within hospital"},
        "families": {"rf": {"label": "Random forest"}, "xgb": {"label": "XGBoost"}},
        "method_tables": {"m": {"dataset": "tcga", "families": ["rf", "xgb"], "rows": [
            {"label": "Local", "rf": "l_rf", "xgb": "l_xgb"},
            {"label": "Centralized", "rf": "c_rf", "xgb": "c_xgb"},
            {"label": "Centralized, step-matched (diagnostic)", "rf": "n/a", "xgb": "s_xgb",
             "diagnostic": "central_step_matched"},
            {"label": "Vanilla FL", "rf": "f_rf", "xgb": "f_xgb"},
            {"label": "Ours", "rf": "o_rf", "xgb": "o_xgb"}]}},
        "paired_comparisons": {"tables": {"m": "c_index"},
                               "pairs": [["Vanilla FL", "Centralized"],
                                         ["Vanilla FL", "Centralized, step-matched (diagnostic)"],
                                         ["Ours", "Vanilla FL"]]}}
DIAG = {"runs": [{"task": "survival", "method": "central_xgb", "per_round": [
    {"round": 1, "train_c": 0.80, "held_c": 0.66}, {"round": 30, "train_c": 0.99, "held_c": 0.56}]}]}


def _key(res, family, label, within=False):
    return next(k for k in DR.key_comparisons(res, SPEC, within=within) if k["family"] == family and k["label"] == label)


# ── 1a: a gain of Ours whose screen drops honest updates on a clean run is never called a win ─────────────────
def test_a_gain_of_ours_with_a_costly_clean_screen_is_not_ranked():
    cis = {("o_rf", "f_rf", "c_index"): (0.003, 0.0001, 0.006), ("o_xgb", "f_xgb", "c_index"): (0.004, 0.001, 0.007)}
    res = FakeRes(cis, fer={"o_rf": 0.133, "o_xgb": 0.02})
    rf, xgb = _key(res, "rf", "Ours vs Vanilla FL"), _key(res, "xgb", "Ours vs Vanilla FL")
    assert rf["relation"] == "outperforms" and rf["verdict"] == DR.NOT_RANKED_GAIN    # the interval alone would rank it
    cell, phrase = DR.verdict_cell(rf), DR.verdict_phrase(rf)
    assert cell.startswith("not ranked") and "13.3%" in cell and "5.0%" in cell
    assert "outperform" not in cell and "outperform" not in phrase and "is not ranked" in phrase
    assert xgb["verdict"] == "outperforms" and DR.verdict_cell(xgb) == "outperforms"   # a clean screen: ranked


# ── 1d: a federated gain over centralized carries the diagnosis, in the same sentence ──────────────────────────
def test_a_federated_gain_over_centralized_carries_the_diagnosis_in_the_same_sentence():
    cis = {("f_xgb", "c_xgb", "c_index"): (0.07, 0.02, 0.12), ("f_xgb", "s_xgb", "c_index"): (0.04, -0.002, 0.075),
           ("f_rf", "c_rf", "c_index"): (0.06, 0.01, 0.10)}
    res = FakeRes(cis, diag=DIAG, curves={"c_rf": [[0.615, 0.616]] * 5})
    x = _key(res, "xgb", "Vanilla FL vs Centralized")
    assert x["relation"] == "outperforms"
    assert ("but centralized XGBoost over-fits (Phase 8: over the 30 rounds its training C-index rises from 0.800 to "
            "0.990 while its held-out C-index falls from 0.660 to 0.560)") in x["clause"]
    assert ("federated training is not distinguishable at this sample size from the step-matched centralized row "
            "(+0.040, 95% CI -0.002 to +0.075)") in x["clause"]
    assert DR.verdict_cell(x).startswith("outperforms, but centralized XGBoost over-fits (Phase 8), and A is not "
                                         "distinguishable from the step-matched row (95% CI -0.002 to +0.075)")
    assert "Vanilla FL outperforms Centralized (+0.070, 95% CI +0.020 to +0.120), but centralized" in DR.verdict_phrase(x)
    r = _key(res, "rf", "Vanilla FL vs Centralized")             # not diagnosed: over-fitting is never claimed
    assert "not diagnosed: Phase 8 diagnosed XGBoost only" in r["clause"] and "over-fits" not in r["clause"]
    assert "the centralized random forest's pooled held-out C-index is 0.615 after round 1 and 0.616 after round 2" \
        in r["clause"]
    still = FakeRes({**cis, ("f_xgb", "s_xgb", "c_index"): (0.07, 0.01, 0.13)}, diag=DIAG)
    x2 = _key(still, "xgb", "Vanilla FL vs Centralized")         # a gain that survives step-matching is said so
    assert "federated training outperforms the step-matched centralized row" in x2["clause"]
    assert "A also outperforms the step-matched row" in DR.verdict_cell(x2)
    assert _key(FakeRes(cis), "xgb", "Vanilla FL vs Centralized")["clause"] is None   # no diagnosis: nothing added
    loss = FakeRes({**cis, ("f_xgb", "c_xgb", "c_index"): (-0.07, -0.12, -0.02)}, diag=DIAG)
    assert _key(loss, "xgb", "Vanilla FL vs Centralized")["clause"] is None          # only gains are qualified


# ── 3: the MNIST tables of experiments that never ran say so (Table 5 is not a run) ──────────────────────────────
def test_mnist_tables_without_results_say_they_were_not_run():
    empty = [["label_flip", "—", "n/a"], ["sign_flip", "—", "—"]]
    note = MT.not_run_note("table2", empty)
    assert "Not run: experiment exp2 has no results in this repository" in note and "TODO.md" in note
    assert MT.not_run_note("table2", [["label_flip", "97.10 ± 0.20", "—"]]) == ""       # any result: no note
    assert MT.not_run_note("table2_full", [["none", "0", "inf", "—", "—"]]) != ""       # label columns are not results
    assert MT.not_run_note("table5", empty) == "" and MT.not_run_note("table1", empty) == ""
    for name in MT.NOT_RUN:                                                           # the generated tables, if built
        md = RESULTS / "tables" / f"{name}.md"
        if md.exists():
            caption = md.read_text(encoding="utf-8").splitlines()[2]
            assert f"Not run: experiment {MT.NOT_RUN[name][0]}" in caption and "earlier run" not in caption
            assert "Not run: experiment" in (RESULTS / "tables" / f"{name}.tex").read_text(encoding="utf-8")
    t5 = RESULTS / "tables" / "table5.md"
    if t5.exists():
        assert "Not run" not in t5.read_text(encoding="utf-8")                     # built from comparison.yaml


# ── 2: every diagram rendered from its current source, SVG and 300-dpi PNG, and packed ────────────────────────────
def test_every_diagram_is_rendered_from_its_current_source():
    from fedverify.analysis.make_report_pack import png_dpi
    from fedverify.analysis.render_diagrams import DIAGRAMS, INDEX, stale
    assert stale() == [], "a render is missing or older than its source: python3 -m fedverify.analysis.render_diagrams"
    idx = json.loads((DIAGRAMS / INDEX).read_text(encoding="utf-8"))
    names = {p.stem for p in DIAGRAMS.glob("*.mmd")}
    assert set(idx["diagrams"]) == names and len(names) == 6
    for name in names:
        assert png_dpi(DIAGRAMS / f"{name}.png") == 300
        assert (DIAGRAMS / f"{name}.svg").read_text(encoding="utf-8")[:300].lstrip().startswith("<svg")
        w, h = idx["diagrams"][name]["natural_css_px"]                              # drawn at natural size
        assert abs(idx["diagrams"][name]["png_px"][0] - w * 300 / 96) <= 0.02 * w * 300 / 96


@pytest.mark.skipif(not (RESULTS / "report_pack" / "README.md").exists(), reason="no report pack built")
def test_report_pack_holds_every_diagram_as_source_svg_and_300_dpi_png():
    from fedverify.analysis.make_report_pack import png_dpi
    from fedverify.analysis.render_diagrams import DIAGRAMS
    pack = RESULTS / "report_pack" / "diagrams"
    for src in DIAGRAMS.glob("*.mmd"):
        png = next(pack.glob(f"*_{src.stem}.png"))
        assert png_dpi(png) == 300 and png.read_bytes() == src.with_suffix(".png").read_bytes()
        assert png.with_suffix(".svg").exists() and png.with_suffix(".mmd").exists()
    assert "Mermaid sources (not rendered" not in (RESULTS / "report_pack" / "README.md").read_text(encoding="utf-8")


# ── on the real results ──────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def draft(tmp_path_factory):
    out = tmp_path_factory.mktemp("draft") / "RESULTS_DRAFT.md"
    assert DR.main(["--results-dir", str(RESULTS), "--out", str(out)]) == 0
    return out.read_text(encoding="utf-8"), json.loads(out.with_suffix(".comparisons.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    from fedverify.analysis import make_showcase as SH
    out = tmp_path_factory.mktemp("page")
    assert SH.main(["--results-dir", str(RESULTS), "--out-dir", str(out)]) == 0
    body = (out / "showcase_body.html").read_text(encoding="utf-8")
    blob = re.search(r'<script type="application/json" id="showcase-data">(.*?)</script>', body, re.S).group(1)
    return json.loads(blob.replace("<\\/", "</"))


@pytest.fixture(scope="module")
def tables():
    from fedverify.analysis.make_fl_healthcare_tables import Results, load_spec, paired_table, paired_within_table
    spec = load_spec()
    res = Results(RESULTS, spec)
    return res, spec, paired_table(res, spec), paired_within_table(res, spec)


def sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.])\s+(?=[A-Z*])", text)


def tcga_sentences(text: str) -> list[str]:
    """Sentences about TCGA: in a TCGA paragraph or limitation item (one line each in the draft), or naming it."""
    out = []
    for line in text.splitlines():
        for s in sentences(line):
            if line.startswith(("**TCGA-BRCA", "- TCGA")) or s.startswith("TCGA-BRCA"):
                out.append(s)
    return out


FED_BEATS_CENTRAL = re.compile(r"outperforms (centralized training|Centralized|the over-fitted central)")


@have_results
def test_the_paired_table_never_ranks_a_gain_of_ours_with_a_costly_clean_screen(tables, draft):
    res, spec, pooled, _ = tables
    fer_max = spec["clean_false_exclusion_max"]
    ks = DR.key_comparisons(res, spec)
    rows = {(r[0], r[1], r[2]): r for r in pooled.rows}
    from fedverify.analysis.make_fl_healthcare_tables import dataset_label
    seen = 0
    for k in ks:
        if k["a_label"] != "Ours" or k["relation"] != "outperforms":
            continue
        verdict = rows[(dataset_label(spec, k["dataset"]), k["model"], k["label"])][-1]
        fer = DR.clean_fer(res, k["dataset"], k["a"], "test")
        if fer is not None and fer > fer_max:
            seen += 1
            assert verdict.startswith("not ranked") and f"{100 * fer:.1f}%" in verdict and "outperform" not in verdict
        else:
            assert verdict == "outperforms"
    assert seen >= 1                                                 # heart random forest: 13.3% on a clean run
    text, _ = draft
    para = next(p for p in text.split("\n\n") if p.startswith("**Paired comparisons.**"))
    decisive = sum(k["relation"] != DR.NOT_DISTINGUISHABLE for k in ks)
    unranked = sum(k["verdict"] == DR.NOT_RANKED_GAIN for k in ks)
    assert f"{decisive} have a 95% bootstrap interval that excludes 0 and {decisive - unranked} of those are ranked" in para
    assert para.count("is not ranked: screening a clean run excludes") == unranked


@have_results
def test_no_output_calls_a_model_best(draft, page):
    text, comps = draft
    assert not re.search(r"\bbest\b", text, re.I)
    assert not re.search(r"\bbest\b", json.dumps(page, ensure_ascii=False), re.I)
    tops = [c for c in comps if c["section"].endswith("/within_top")]
    assert len(tops) == 3
    for c in tops:                                    # each "highest score" goes through the rule against the runner-up
        assert c.get("ci") is not None and c["relation"] == DR.ci_relation(c["ci"])
        assert f"highest score {c['a']} at" in text
        if c["relation"] == DR.NOT_DISTINGUISHABLE:
            assert f"not distinguishable from {c['b']} (" in text
        else:
            assert f"which {c['relation']} the runner-up {c['b']} (" in text
    assert "largest fall" not in text                 # the old within-hospital ranking had no interval


@have_results
def test_within_hospital_key_comparisons_use_the_same_interval_rule(tables, draft):
    from fedverify.analysis.make_tables import mean_std
    res, spec, pooled, within = tables
    ks = DR.key_comparisons(res, spec, within=True)
    assert len(within.rows) == len(ks) == 20 == sum(1 for r in pooled.rows if not r[2].startswith("DP-FL"))
    for row, k in zip(within.rows, ks):
        o = res.paired.compare(k["dataset"], k["a"], k["b"], k["metric"])
        assert k["metric"].endswith("_within") and row[3].endswith("within hospital")
        assert row[7] == f"{o['ci'][0]:+.3f} to {o['ci'][1]:+.3f}" and k["relation"] == DR.ci_relation(o["ci"])
        assert row[8] == DR.verdict_cell(k)
        st, vals = res.metric(k["dataset"], k["a"], k["metric"], "test")   # equals the method table's within column
        assert st == "ok" and row[4] == f"{mean_std(vals)[0]:.3f}"
    text, comps = draft
    recorded = [c for c in comps if c["section"].startswith("paired_within/")]
    assert len(recorded) == 20 and all(c["metric"].endswith("_within") and c.get("ci") is not None for c in recorded)
    assert "Within-hospital differences are claimed only through this table's intervals." in text
    assert "| Verdict within hospital | Pooled verdict |" in text                     # the table is in the draft


@have_results
def test_every_federated_gain_over_centralized_on_tcga_is_qualified(tables, draft, page):
    res, spec, pooled, within = tables
    text, comps = draft
    gains = [c for c in comps if "tcga" in c["section"] and c["a"].lower() == "vanilla fl"
             and c["b"].lower() == "centralized" and c["relation"] == "outperforms"]
    assert gains                                                                  # random forest and XGBoost AFT
    assert all("over-fits (Phase 8" in c.get("clause", "") or "not diagnosed" in c.get("clause", "") for c in gains)
    hits = 0
    for s in tcga_sentences(text):
        if FED_BEATS_CENTRAL.search(s):
            hits += 1
            assert ("over-fit" in s and "step-matched" in s) or "not diagnosed" in s, s
    assert hits >= 6                       # Section A, the two paired paragraphs and the limitation
    for t in (pooled, within):                                                   # the table cells
        step = {(r[0], r[1]): r for r in t.rows if r[2].endswith("step-matched (diagnostic)")}
        for r in t.rows:
            if r[0] == "TCGA-BRCA" and r[2] == "Vanilla FL vs Centralized" and r[8].startswith("outperforms"):
                assert "over-fits (Phase 8)" in r[8] or "not diagnosed" in r[8], r
                s = step.get((r[0], r[1]))
                if s is not None and s[8] == DR.NOT_DISTINGUISHABLE:              # the step-matched row's own interval
                    assert f"A is not distinguishable from the step-matched row (95% CI {s[7]})" in r[8]
    n = 0
    for m in page["methods"]["tcga"]["metrics"]:                                  # the page's verdict sentences
        for v in m["verdicts"].values():
            if v.get("central") == "outperforms":
                n += 1
                assert "over-fits (Phase 8" in v.get("central_caveat", "") or "not diagnosed" in v.get("central_caveat", "")
    assert n >= 2
    assert page["paired"]["within"]["rows"] == [list(r) for r in within.rows]


# ── 4: the guide quotes the generated key comparisons, with the corrected wording ────────────────────────────────
GUIDE = REPO / "fedverify" / "docs" / "FEDVERIFY_SURVIVAL_GUIDE.md"
SHORT = {"Heart": "UCI heart disease", "TCGA": "TCGA-BRCA", "LR": "Logistic regression", "RF": "Random forest",
         "XGB": "XGBoost", "XGB AFT": "XGBoost AFT"}


def _guide_rows(section: str) -> list[list[str]]:
    text = GUIDE.read_text(encoding="utf-8").split(section, 1)[1].split("\n### ", 1)[0]
    return [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in text.splitlines()
            if ln.startswith(("| Heart |", "| TCGA |"))]


def _kind(verdict: str) -> str:
    v = verdict.replace("is not distinguishable at this sample size from", "not distinguishable")
    for k in ("not ranked", "not distinguishable", "is outperformed by", "outperforms"):
        if v.startswith(k):
            return k
    return v


@have_results
@pytest.mark.parametrize("section,table", [("### D8. The key comparisons", "paired_comparisons"),
                                           ("### D8b. The same 20 comparisons within hospital", "paired_comparisons_within")])
def test_the_guide_quotes_the_generated_key_comparisons(section, table):
    import csv
    gen = {(r[0], r[1], r[2]): r for r in csv.reader((RESULTS / "tables" / f"{table}.csv").open(encoding="utf-8"))}
    rows = _guide_rows(section)
    assert len(rows) == 20
    for g in rows:
        comp = g[2].replace("Federated", "Vanilla FL").replace("step-matched", "step-matched (diagnostic)")
        r = gen[(SHORT[g[0]], SHORT[g[1]], comp)]
        assert g[3].replace("−", "-") == r[6].split(" ± ")[0] and g[4].replace("−", "-") == r[7], (g, r)
        assert _kind(g[5]) == _kind(r[8]), (g, r)
        if "over-fits" in r[8] or "not diagnosed" in r[8]:                  # the qualification survives in the guide
            assert "over-fits" in g[5] or "not diagnosed" in g[5], g
        if len(g) > 6:
            assert _kind(g[6]) == _kind(r[9]), (g, r)
    text = GUIDE.read_text(encoding="utf-8")
    assert not re.search(r"\bis the best\b|\bbest model\b", text)
