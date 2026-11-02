"""Revision 2 Phase 6: the results draft, its comparison rule, and the one-command-per-experiment driver."""
import json
import re
from pathlib import Path

import pytest
import yaml

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
pytest.importorskip("lifelines")
pytest.importorskip("matplotlib")

from fedverify.analysis import make_results_draft as DR  # noqa: E402
from fedverify.analysis.make_fl_healthcare_tables import SPEC  # noqa: E402
from fedverify.survival import experiments as EX  # noqa: E402
from fedverify.survival import runner  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "fedverify" / "results"
have_results = pytest.mark.skipif(not (RESULTS / "survival" / "fedavg_lr" / "seed0" / "summary.json").exists()
                                  or not (RESULTS / "tables").exists(), reason="full results (gitignored) not present")


def test_comparison_rule():
    assert DR.relation(0.80, 0.01, 0.75, 0.02) == "outperforms"                     # 0.05 exceeds both stds
    assert DR.relation(0.75, 0.02, 0.80, 0.01) == "is outperformed by"
    assert DR.relation(0.80, 0.06, 0.75, 0.02) == "is not distinguishable at this sample size from"
    assert DR.relation(0.80, 0.06, 0.79, 0.05) == "is comparable with"               # within both stds
    assert DR.relation(0.5, 0.0, 0.5, 0.0) == "is comparable with"
    assert DR.relation(0.10, 0.01, 0.20, 0.01, higher_is_better=False) == "outperforms"


def test_accuracy_near_the_majority_rate_is_never_ranked():
    """Phase 7: an accuracy within the margin of the majority-class rate, on either side, gets no verdict."""
    assert DR.guarded_relation(0.85, 0.001, 0.70, 0.001, True, "accuracy", 0.84, 0.01) == DR.NOT_RANKED
    assert DR.guarded_relation(0.70, 0.001, 0.845, 0.001, True, "accuracy", 0.84, 0.01) == DR.NOT_RANKED
    assert DR.guarded_relation(0.90, 0.001, 0.70, 0.001, True, "accuracy", 0.84, 0.01) == "outperforms"
    assert DR.guarded_relation(0.85, 0.001, 0.70, 0.001, True, "roc_auc", 0.84, 0.01) == "outperforms"   # accuracy only
    assert DR.guarded_relation(0.85, 0.001, 0.70, 0.001, True, "accuracy", None, 0.01) == "outperforms"


def _cells_in_spec(spec: dict) -> set:
    """Every (dataset, cell) a table or figure reads, from table.yaml."""
    cells = set()
    for t in spec["method_tables"].values():
        cells |= {(t["dataset"], r[f]) for r in t["rows"] for f in t["families"] if r.get(f, "n/a") != "n/a"}
    for t in spec["per_hospital_tables"].values():
        cells |= {(t["dataset"], r[k]) for r in t["rows"] for k in ("local", *t["aggregated"])}
    for t in spec["per_hospital_method_tables"].values():
        cells |= {(t["dataset"], r["cell"]) for r in t["rows"]}
    for t in spec["dp_tables"].values():
        cells |= {(t["dataset"], c) for c in t["cells"]}
    for t in spec["attack_tables"].values():
        for f in t["families"].values():
            cells |= {(t["dataset"], f["ours"]), (t["dataset"], f["undefended"])}
            cells |= {(t["dataset"], f"{f['ours']}_{a}{s}") for a in f["attacks"] for s in ("", "_noscreen")}
    for ds, o in spec["ours"].items():
        cells |= {(ds, o["settings"]), (ds, o["chain_cell"])} | {(ds, p["cell"]) for p in o["performance"]}
    for key in ("chain_cells", "stage_cells"):
        cells |= {(e["dataset"], e["cell"]) for e in spec["overhead"][key]}
    cells |= {(ds, c) for ds in ("heart", "tcga") for c in spec["figures"]["per_hospital_methods"]}
    return cells


def test_the_experiments_produce_every_cell_the_tables_read():
    made = set()
    for name in EX.EXPERIMENTS:
        for cmd in EX.commands(name):
            if cmd[2] == "fedverify.survival.runner":
                cfg, _ = runner.build_config(cmd[3:])
                made.add((cfg.dataset, cfg.cell))
    missing = _cells_in_spec(yaml.safe_load(SPEC.read_text(encoding="utf-8"))) - made
    assert not missing, f"table.yaml reads cells no experiment produces: {sorted(missing)}"


def test_experiments_dry_run_prints_every_command(capsys):
    assert EX.main(["all", "--dry-run"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == len(EX.commands("all")) and all(line.startswith("python3 -m fedverify.") for line in out)
    names = list(EX.EXPERIMENTS)                                    # `all` runs them in this order:
    assert names.index("calibration") < names.index("poisoning")    # screened runs read the calibrated tau


@have_results
def test_results_draft_on_the_completed_results(tmp_path):
    out = tmp_path / "RESULTS_DRAFT.md"
    assert DR.main(["--results-dir", str(RESULTS), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    for heading in ("## Experimental Setup", "## Results", "## Limitations"):
        assert heading in text
    comps = json.loads(out.with_suffix(".comparisons.json").read_text(encoding="utf-8"))
    assert comps
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    for c in comps:                                                 # every verdict is the Phase 11 rule's verdict
        assert c["rule"] == "bootstrap_ci" and c.get("ci") is not None, c
        assert c["relation"] == DR.ci_guarded_relation(c["ci"], c["a_mean"], c["b_mean"], True, c["metric"],
                                                       c.get("majority"), spec["accuracy_majority_margin"])
        assert c["old_relation"] == DR.guarded_relation(c["a_mean"], c["a_std"], c["b_mean"], c["b_std"], True,
                                                        c["metric"], c.get("majority"), spec["accuracy_majority_margin"])
        if "outperform" in c["relation"]:                           # a ranking always has an interval clear of 0
            assert c["ci"][0] > 0 or c["ci"][1] < 0
        if c["metric"] == "accuracy" and c.get("majority") is not None and "outperform" in c["relation"]:
            assert min(abs(c["a_mean"] - c["majority"]), abs(c["b_mean"] - c["majority"])) > spec["accuracy_majority_margin"]
    n_out = sum("outperform" in c["relation"] and "written_as" not in c for c in comps)
    assert text.count("outperform") <= 1 + n_out                    # the rule paragraph, then only rule verdicts
    for c in comps:                                                 # suppressed only where the clean screen is costly
        if "written_as" in c:
            assert c["relation"] == "outperforms" and "skewed hospital" in c["written_as"]
    assert "Predict all alive (baseline)" in text                   # Phase 7: the trivial TCGA baseline is discussed
    assert "its accuracy is the majority-class rate" in text and "lie within 0.01 of it (never ranked)" in text
    assert "100% here holds by construction" in text and "Not covered:" in text
    assert re.search(r"with K = 4 hospitals, recall on UCI heart disease is \d+–\d+%", text)
    related = text.split("### H. Related work", 1)[1].split("## Limitations", 1)[0]
    assert "outperform" not in related                              # other papers: reported, never compared
    lim = text.split("## Limitations", 1)[1]
    tcga = json.loads((RESULTS / "survival/fedavg_lr/seed0/config.json").read_text())["data"]
    heart = json.loads((RESULTS / "heart/fedavg_lr/seed0/config.json").read_text())["data"]
    assert f"only {tcga['deaths']} deaths" in lim                   # required limitations, numbers from results
    sw = heart["per_hospital"]["Switzerland"]
    assert f"Switzerland {sw['patients']} rows" in lim
    assert "recall on UCI heart disease is" in lim                  # Phase 7: low recall stated in the limitations
    for phrase in ("proof-of-work chain is simulated", "non-adaptive", "per training example"):
        assert phrase in lim
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert links and all((out.parent / link).resolve().exists() for link in links)   # every figure link resolves


@have_results
def test_showcase_page_embeds_numbers_from_the_results(tmp_path):
    from fedverify.analysis import make_showcase as SH
    from fedverify.analysis.make_tables import mean_std
    assert SH.main(["--results-dir", str(RESULTS), "--out-dir", str(tmp_path)]) == 0
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    body = (tmp_path / "showcase_body.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "<title>FedVerify-Survival</title>" in body[:8192]
    blob = re.search(r'<script type="application/json" id="showcase-data">(.*?)</script>', body, re.S).group(1)
    d = json.loads(blob.replace("<\\/", "</"))
    assert d["round"]["root_ok"] is True                              # the anchored root re-derives from its leaves
    assert d["tiles"]["patients"] == sum(x["patients"] for x in d["datasets"])
    s = json.loads((RESULTS / "heart/fedavg_lr/seed0/summary.json").read_text())
    acc = next(m for m in d["methods"]["heart"]["metrics"] if m["key"] == "accuracy")
    vanilla = next(r for r in acc["rows"] if r["label"] == "Vanilla FL")["cells"]["lr"]
    want = mean_std([rp["pooled"]["accuracy"] for rp in s["repeats"]])
    assert vanilla == pytest.approx(list(want), abs=1e-6)             # the plotted number is the results file's
    on = json.loads((RESULTS / "chain_attacks/double_spending/seed0/summary.json").read_text())
    assert d["chain_attacks"]["datasets"]["heart"]["attacks"]["double_spending"]["on"] == on["resilience_pct"]


@have_results
def test_tables_carry_the_baseline_recall_and_clean_false_exclusion():
    """Phase 7: the TCGA baseline row, recall next to the clean false-exclusion rate, the chain caveats."""
    from fedverify.analysis.make_fl_healthcare_tables import Results, load_spec, pct
    spec = load_spec()
    res = Results(RESULTS, spec)
    tables = RESULTS / "tables"
    maj = res.majority("tcga")
    data = json.loads((RESULTS / "survival/fedavg_lr/seed0/config.json").read_text())["data"]
    assert maj == pytest.approx(max(data["deaths"], data["patients"] - data["deaths"]) / data["patients"])
    for name in ("methods_tcga_binary", "methods_tcga_survival", "per_hospital_tcga", "per_hospital_methods_tcga",
                 "dp_sweep_tcga", "attack_tcga", "attack_tcga_survival"):
        text = (tables / f"{name}.md").read_text(encoding="utf-8")
        assert "Predict all alive (baseline)" in text, name                 # every TCGA table
    binary = (tables / "methods_tcga_binary.md").read_text(encoding="utf-8").splitlines()
    header = next(line for line in binary if line.startswith("| Method |")).split(" | ")
    base = next(line for line in binary if line.startswith("| Predict all alive")).split(" | ")
    col = {h.strip(" |"): i for i, h in enumerate(header)}                  # "<family>: <metric>" columns
    accs = [base[i].strip(" |") for h, i in col.items() if h.endswith(": Accuracy")]
    f1s = [base[i].strip(" |") for h, i in col.items() if h.endswith(": F1")]
    assert accs and all(a == f"{maj:.3f} ± 0.000" for a in accs)           # its accuracy is the majority rate
    assert f1s and all(f == "0.000 ± 0.000" for f in f1s)
    heart = (tables / "attack_heart.md").read_text(encoding="utf-8").splitlines()
    head = next(line for line in heart if line.startswith("| Model |")).split(" | ")
    assert "Recall (%)" in head and "Clean false excl. (%)" in head[-1]
    att = spec["attack_tables"]["attack_heart"]
    fam, f = next(iter(att["families"].items()))
    s = json.loads((RESULTS / "heart" / f["ours"] / "seed0/summary.json").read_text())
    want = pct(s["attack"]["detection_pooled_over_repeats"]["false_exclusion_rate"])
    model = f.get("label") or spec["families"][fam]["label"]
    screened = [line for line in heart if line.startswith(f"| {model} |") and " | on | " in line]
    assert screened and all(line.rstrip(" |").split(" | ")[-1] == want for line in screened)
    chain = (tables / "chain_resilience.md").read_text(encoding="utf-8")
    assert "holds by construction" in chain and "Not covered:" in chain and "about 5 s per block" in chain
