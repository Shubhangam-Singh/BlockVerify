"""Generate fedverify/survival/paper/RESULTS_DRAFT.md: Experimental Setup, Results, Limitations (Revision 2 §6).

    python3 -m fedverify.analysis.make_results_draft
    python3 -m fedverify.analysis.make_fl_healthcare_all     # tables, figures, then this draft

Every number is read from a results file through the Phase-5 readers (summary.json, config.json, rounds.jsonl,
chain_metrics.json, chain_attacks/*/summary.json, calibration/taus_survival.json); other papers' numbers come from the
reference CSVs and are only ever "reported". Tables are inserted from <results>/tables/*.md and figures referenced from
<results>/figures/ with their captions, so build those first.

Comparison sentences come from one rule (COMPARISON_RULE) applied to mean ± std over repeats; every comparison the
draft makes is also written to RESULTS_DRAFT.comparisons.json so it can be audited. A missing input renders as an em
dash and is listed on stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .make_fl_healthcare_tables import (MISSING, NA, REPO, SPEC, Results, chain_caveats, dataset_label, eps_text,
                                        load_spec, metric_label, read_csv, reported_resilience, seeds_text, stage_stats,
                                        vocab_rows)
from .make_tables import mean_std
from .paired_stats import WITHIN

OUT = REPO / "fedverify" / "survival" / "paper" / "RESULTS_DRAFT.md"
COMPARISON_RULE = ("Comparisons follow one rule (Phase 11): method A outperforms method B only if the 95% patient-level "
                   "bootstrap confidence interval of their paired difference excludes 0 (both run on the same folds "
                   "and repeats; 2000 resamples of the patients, stratified by hospital, fixed seed); otherwise they "
                   "are not distinguishable at this sample size. The mean paired difference ± its standard deviation "
                   "over the repeats is reported with each interval. No p-values are given: with 5 repeats an exact "
                   "test cannot go below 0.0625.")
OLD_RULE = ("Before Phase 11: A outperforms B if the difference of their means exceeds both standard deviations over "
            "repeats; comparable if within both; otherwise not distinguishable at this sample size.")
NOT_DISTINGUISHABLE = "is not distinguishable at this sample size from"
NOT_RANKED = "is not ranked against"          # Phase 7: accuracy at the majority-class rate says nothing
NEAR_MAJORITY = ("(both are accuracies within {margin:g} of the majority-class rate {maj:.3f}, which predicting the "
                 "larger class for everyone already reaches)")
STAGE_KEYS = ("serialize_ms", "digest_ms", "cid_ms", "sign_ms", "verify_ms", "merkle_ms", "anchor_ms")


def relation(a_mean: float, a_std: float, b_mean: float, b_std: float, higher_is_better: bool = True) -> str:
    """The comparison rule: how A relates to B."""
    d = a_mean - b_mean
    if abs(d) > a_std and abs(d) > b_std:
        return "outperforms" if (d > 0) == higher_is_better else "is outperformed by"
    if abs(d) <= a_std and abs(d) <= b_std:
        return "is comparable with"
    return "is not distinguishable at this sample size from"


def ci_relation(ci: Optional[list], higher_is_better: bool = True) -> str:
    """The Phase 11 rule: decisive only when the bootstrap CI of the paired difference A - B excludes 0."""
    if not ci or ci[0] is None or ci[1] is None or ci[0] != ci[0] or ci[1] != ci[1]:
        return NOT_DISTINGUISHABLE
    lo, hi = ci
    if lo > 0:
        return "outperforms" if higher_is_better else "is outperformed by"
    if hi < 0:
        return "is outperformed by" if higher_is_better else "outperforms"
    return NOT_DISTINGUISHABLE


def ci_guarded_relation(ci: Optional[list], a_mean: float, b_mean: float, higher_is_better: bool = True,
                        metric: str = "", majority: Optional[float] = None, margin: float = 0.01) -> str:
    """ci_relation, except that an accuracy within `margin` of the majority-class rate is never ranked (Phase 7)."""
    tol = margin + 1e-9
    if metric == "accuracy" and majority is not None and (abs(a_mean - majority) <= tol
                                                          or abs(b_mean - majority) <= tol):
        return NOT_RANKED
    return ci_relation(ci, higher_is_better)


def guarded_relation(a_mean: float, a_std: float, b_mean: float, b_std: float, higher_is_better: bool = True,
                     metric: str = "", majority: Optional[float] = None, margin: float = 0.01) -> str:
    """The comparison rule, except that an accuracy within `margin` of the majority-class rate (on either side of the
    comparison) is never ranked: such a model may only be predicting the larger class (Phase 7)."""
    tol = margin + 1e-9                                  # 0.85 - 0.84 is 0.010000000000000009 in floating point
    if metric == "accuracy" and majority is not None and (abs(a_mean - majority) <= tol
                                                          or abs(b_mean - majority) <= tol):
        return NOT_RANKED
    return relation(a_mean, a_std, b_mean, b_std, higher_is_better)


def clean_fer(res: Results, ds: str, cell: Optional[str], where: str) -> Optional[float]:
    """False-exclusion rate of a screened cell's screen on its clean run (no attacker)."""
    if not cell:
        return None
    d = ((res.summary(ds, cell, where) or {}).get("attack") or {}).get("detection_pooled_over_repeats") or {}
    return d.get("false_exclusion_rate")


def skew_caveat(fer: Optional[float], fer_max: float) -> Optional[str]:
    """Phase 7: when screening a clean run drops honest updates too often, a gain is not attributed to screening."""
    if fer is None or fer <= fer_max:
        return None
    return (f"screening a clean run excludes {_pct(fer)} of honest hospital-rounds (above {_pct(fer_max)}), so any "
            "gain may come from dropping a skewed hospital's updates rather than from catching attackers")


def recall_range(res: Results, spec: dict, ds: str, where: str) -> Optional[tuple[float, float, int]]:
    """Min and max screen recall over every screened attack cell of a dataset's attack tables, and K."""
    recalls = []
    for t in spec["attack_tables"].values():
        if t["dataset"] != ds:
            continue
        for f in t["families"].values():
            for a in f["attacks"]:
                s = res.summary(ds, f"{f['ours']}_{a}", where) or {}
                d = (s.get("attack") or {}).get("detection_pooled_over_repeats")
                if d and d.get("recall") is not None:
                    recalls.append(d["recall"])
    k = len(((res.config(ds, "fedavg_lr", where) or {}).get("data") or {}).get("per_hospital") or {})
    return (min(recalls), max(recalls), k) if recalls else None


def recall_sentence(res: Results, spec: dict, where: str) -> str:
    """Phase 7: the screen's recall range per dataset, from the results."""
    bits = []
    for ds in ("heart", "tcga"):
        r = recall_range(res, spec, ds, where)
        if r:
            bits.append(f"with K = {r[2]} hospitals, recall on {dataset_label(spec, ds)} is {100 * r[0]:.0f}–"
                        f"{100 * r[1]:.0f}% over every attack and model")
    if not bits:
        return f"Screen recall: {MISSING}."
    return ("Screen recall is low: " + "; ".join(bits) + ". Most attacker hospital-rounds are not excluded, so the "
            "screen limits rather than prevents poisoning.")


GROUP_WORDS = {"outperforms": "exceed it (95% CI of the paired difference above 0)",
               "is outperformed by": "are below it (95% CI below 0)"}


def site_rate_sentence(metric_label: str, pairs: list[tuple[str, str]], list_max: int = 8) -> str:
    """Phase 9: how the results of one table relate to the site-rate reference on one metric, from the rule's verdicts
    (pairs of (label, relation of the result to the reference)). A group of at most list_max results is named."""
    groups: dict[str, list[str]] = {}
    for label, rel in pairs:
        groups.setdefault(rel if rel in GROUP_WORDS else "rest", []).append(label)
    n = len(pairs)

    def count(g: list[str]) -> str:
        return f"all {n}" if len(g) == n else f"{len(g)} of {n}"

    def names(g: list[str]) -> str:
        return f" ({'; '.join(g)})" if len(g) <= list_max else ""
    bits = []
    for rel, words in GROUP_WORDS.items():
        g = groups.get(rel, [])
        bits.append(f"{count(g)} {words}{names(g)}" if g else
                    "none " + words.split(" (")[0].replace("exceed ", "exceeds ").replace("are below", "is below"))
    rest = groups.get("rest", [])
    if rest:
        bits.append(f"{'all' if len(rest) == n else 'the other'} {len(rest)} are not distinguishable from "
                    f"it{names(rest)}")
    return f"on {metric_label}, " + "; ".join(bits) + "."


@dataclass
class Stat:
    mean: float
    std: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} ± {self.std:.3f}"


def _s(stat: Optional[Stat]) -> str:
    return MISSING if stat is None else str(stat)


def _pct(v: Optional[float]) -> str:
    return MISSING if v is None else f"{100 * v:.1f}%"


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def stat_of(res: Results, ds: str, cell: Optional[str], metric: str, where: str) -> Optional[Stat]:
    st, vals = res.metric(ds, cell, metric, where)
    if st != "ok":
        return None
    m, sd = mean_std(vals)
    return Stat(m, sd, len(vals))


def site_stat(res: Results, ds: str, metric: str, where: str) -> Optional[Stat]:
    st, vals = res.site_rate(ds, metric, where)
    if st != "ok":
        return None
    m, sd = mean_std(vals)
    return Stat(m, sd, len(vals))


def overfit_text(res: Results, where: str) -> str:
    """Phase 9: centralized XGBoost over-fits on TCGA (training vs held-out C-index), and the step-matched row."""
    d = res.need(res.root / "diagnostics" / "central_xgb_tcga.json", where) or {}
    run = next((r for r in d.get("runs", []) if r["task"] == "survival" and r["method"] == "central_xgb"), None)
    held = stat_of(res, "tcga", "central_xgb_aft", "c_index", where)
    steps = stat_of(res, "tcga", "central_xgb_steps_aft", "c_index", where)
    cfg = (res.config("tcga", "central_xgb_steps_aft", where) or {}).get("config", {})
    n_steps = (int(cfg["xgb_trees_per_round"]) * int(cfg["rounds"])) if cfg else None
    train = None if run is None else run["per_round"][-1]["train_c"]
    return (f"Centralized XGBoost over-fits: on TCGA-BRCA its AFT model reaches a training C-index of "
            f"{MISSING if train is None else f'{train:.3f}'} while its held-out C-index is "
            f"{MISSING if held is None else f'{held.mean:.3f}'} (DESIGN_NOTES.md section 8); the step-matched "
            f"diagnostic row, which takes the federated model's {n_steps or MISSING} boosting steps and was not a "
            f"blind result, reaches {_s(steps)}.")


def site_auc_text(res: Results, spec: dict, where: str) -> str:
    """Phase 9: the site-rate reference's pooled ROC-AUC on both datasets, and what it means."""
    auc = {ds: site_stat(res, ds, "roc_auc", where) for ds in ("tcga", "heart")}
    return (f"Scoring each patient by nothing but their hospital's training-fold event rate gives a pooled ROC-AUC "
            f"of {_s(auc['tcga'])} on {dataset_label(spec, 'tcga')} ({dataset_label(spec, 'heart')}: "
            f"{_s(auc['heart'])}), so pooled ROC-AUC partly measures how base rates differ between hospitals.")


# ── Phase 12: the wording every output shares ────────────────────────────────
NOT_RANKED_GAIN = "not ranked"          # a gain of Ours whose screen drops too many honest updates (Phase 7)
SHORT_RELATION = {"outperforms": "outperforms", "is outperformed by": "is outperformed by",
                  NOT_DISTINGUISHABLE: "is not distinguishable from"}


def _model_name(label: str) -> str:
    """'Random forest' -> 'random forest'; 'XGBoost' stays."""
    return label if label[:2].isupper() else label[:1].lower() + label[1:]


def step_matched_cell(t: dict, fam: str) -> Optional[str]:
    """The family's step-matched diagnostic cell in a method table (table.yaml diagnostic_rows), if it has one."""
    row = next((r for r in t["rows"] if r.get("diagnostic") == "central_step_matched"), None)
    cell = (row or {}).get(fam, NA)
    return None if cell == NA else cell


def central_gap_clause(res: Results, spec: dict, t: dict, fam: str, metric: str, where: str,
                       short: bool = False) -> Optional[str]:
    """Phase 12: what must follow, in the same sentence, every "federated outperforms centralized" on a dataset with a
    Phase 8 diagnosis (diagnostics/central_xgb_<dataset>.json; TCGA): the centralized model over-fits, and how federated
    relates to the step-matched row, from that comparison's own interval. A model the diagnosis did not cover gets that
    said instead. None where there is no diagnosis for the dataset. `short` is the table-cell form ("A" = federated)."""
    ds = t["dataset"]
    diag = res.load(res.root / "diagnostics" / f"central_xgb_{ds}.json")
    if not diag:
        return None
    by = {r["label"]: r for r in t["rows"]}
    fed, cen = (by.get("Vanilla FL") or {}).get(fam, NA), (by.get("Centralized") or {}).get(fam, NA)
    name = _model_name(spec["families"][fam]["label"])
    base = WITHIN.get(metric, metric)
    task = "survival" if base == "c_index" else "binary"
    lab = "C-index" if task == "survival" else "ROC-AUC"
    run = next((r for r in diag.get("runs", []) if r["task"] == task and r["method"] == f"central_{fam}"), None)
    step = step_matched_cell(t, fam)
    if run and step:
        k_tr, k_ho = ("train_c", "held_c") if task == "survival" else ("train_auc", "held_auc")
        first, last = run["per_round"][0], run["per_round"][-1]
        o = res.paired.compare(ds, fed, step, metric)
        if o is None:
            res.note_missing(f"paired bootstrap {ds}|{fed}|{step}|{metric}", where)
            rel, ci_txt, diff = MISSING, MISSING, MISSING
        else:
            rel, ci_txt, diff = ci_relation(o["ci"]), f"{o['ci'][0]:+.3f} to {o['ci'][1]:+.3f}", f"{o['diff_mean']:+.3f}"
        up, down = last[k_tr] > first[k_tr], last[k_ho] < first[k_ho]
        state = "over-fits" if up and down else "was diagnosed"
        if short:
            word = "also outperforms" if rel == "outperforms" else SHORT_RELATION.get(rel, rel)
            return f"but centralized {name} {state} (Phase 8), and A {word} the step-matched row (95% CI {ci_txt})"
        return (f"but centralized {name} {state} (Phase 8: over the {last['round']} rounds its training {lab} "
                f"{'rises' if up else 'goes'} from {first[k_tr]:.3f} to {last[k_tr]:.3f} while its held-out {lab} "
                f"{'falls' if down else 'goes'} from {first[k_ho]:.3f} to {last[k_ho]:.3f}), and federated training "
                f"{rel} the step-matched centralized row ({diff}, 95% CI {ci_txt})")
    covered = sorted({spec["families"][r["method"].split("_", 1)[1]]["label"] for r in diag.get("runs", [])
                      if r["method"].startswith("central_") and r["method"].split("_", 1)[1] in spec["families"]})
    why = f"Phase 8 diagnosed {_join(covered) or MISSING} only, and the {name} has no step-matched row"
    if short:
        return f"but not diagnosed: {why}"
    s = res.summary(ds, cen, where) or {}
    curves = [(rp.get("pooled_by_round") or {}).get(base) for rp in s.get("repeats", [])]
    trend = ""
    if curves and all(c and all(v is not None for v in c) for c in curves):
        n = min(len(c) for c in curves)
        first_m = sum(c[0] for c in curves) / len(curves)
        last_m = sum(c[n - 1] for c in curves) / len(curves)
        trend = (f"; the centralized {name}'s pooled held-out {metric_label(spec, base)} is {first_m:.3f} after "
                 f"round 1 and {last_m:.3f} after round {n}")
    return f"but this gap was not diagnosed: {why}{trend}"


def verdict_cell(k: dict) -> str:
    """A key comparison's verdict as a table cell: the rule's relation and, in the same cell, its Phase 7 or Phase 12
    qualification."""
    if k["verdict"] == NOT_RANKED_GAIN:
        return (f"not ranked: screening a clean run excludes {_pct(k['fer'])} of honest hospital-rounds (above "
                f"{_pct(k['fer_max'])}), so the gain is not attributed to screening")
    return f"{k['relation']}, {k['clause_short']}" if k.get("clause_short") else k["relation"]


def verdict_phrase(k: dict) -> str:
    """'A <relation> B (difference, CI)' for a key comparison, with its qualification in the same sentence."""
    s = k["stats"]
    d = MISSING if s is None else f"{s['diff_mean']:+.3f}, 95% CI {s['ci'][0]:+.3f} to {s['ci'][1]:+.3f}"
    if k["verdict"] == NOT_RANKED_GAIN:
        return f"the gain of {k['a_label']} over {k['b_label']} ({d}) is not ranked: {k['skew']}"
    return f"{k['a_label']} {k['relation']} {k['b_label']} ({d})" + (f", {k['clause']}" if k.get("clause") else "")


def key_comparisons(res: Results, spec: dict, within: bool = False, where: str = "paired_comparisons") -> list[dict]:
    """Phase 12: the key comparisons of table.yaml paired_comparisons, pooled or within hospital (same resamples, same
    rule), each with the wording every output uses: Phase 7's rule on gains of Ours whose screen excludes too many
    honest hospital-rounds on a clean run, and the over-fitting clause on federated gains over centralized training."""
    pc = spec.get("paired_comparisons") or {}
    fer_max = float(spec.get("clean_false_exclusion_max", 0.05))
    out = []
    for tname, metric in (pc.get("tables") or {}).items():
        t = spec["method_tables"][tname]
        ds = t["dataset"]
        m = f"{metric}_within" if within else metric
        by = {r["label"]: r for r in t["rows"]}
        for f in t["families"]:
            model = (t.get("family_labels") or {}).get(f, spec["families"][f]["label"])
            for la, lb in pc.get("pairs") or []:
                a, b = (by.get(la) or {}).get(f, NA), (by.get(lb) or {}).get(f, NA)
                if NA in (a, b):
                    continue
                o = res.paired.compare(ds, a, b, m)
                if o is None:
                    res.note_missing(f"paired bootstrap {ds}|{a}|{b}|{m}", where)
                rel = NOT_DISTINGUISHABLE if o is None else ci_relation(o["ci"])
                k = {"table": tname, "dataset": ds, "family": f, "model": model, "label": f"{la} vs {lb}",
                     "a_label": la, "b_label": lb, "a": a, "b": b, "metric": m, "stats": o, "relation": rel,
                     "verdict": rel, "fer": None, "fer_max": fer_max, "skew": None, "clause": None, "clause_short": None}
                if la == "Ours" and rel == "outperforms":
                    fer = clean_fer(res, ds, a, where)
                    cav = skew_caveat(fer, fer_max)
                    if cav:
                        k.update(verdict=NOT_RANKED_GAIN, skew=cav, fer=fer)
                if la == "Vanilla FL" and lb == "Centralized" and rel == "outperforms":
                    k["clause"] = central_gap_clause(res, spec, t, f, m, where)
                    k["clause_short"] = central_gap_clause(res, spec, t, f, m, where, short=True)
                out.append(k)
    return out


class Draft:
    def __init__(self, res: Results, spec: dict, tables_dir: Path, figures_dir: Path, out: Path):
        self.res, self.spec = res, spec
        self.tables, self.figures, self.out = Path(tables_dir), Path(figures_dir), Path(out)
        self.lines: list[str] = []
        self.comparisons: list[dict] = []
        self.margin = float(spec.get("accuracy_majority_margin", 0.01))
        self.stats = res.paired                      # Phase 11: the shared bootstrap cache
        self.fer_max = float(spec.get("clean_false_exclusion_max", 0.05))
        self.captions = self._captions()

    # ── reading ───────────────────────────────────────────────────────────────
    def stat(self, ds: str, cell: Optional[str], metric: str) -> Optional[Stat]:
        st, vals = self.res.metric(ds, cell, metric, "results_draft")
        if st != "ok":
            return None
        m, s = mean_std(vals)
        return Stat(m, s, len(vals))

    def config(self, ds: str, cell: str) -> dict:
        return self.res.config(ds, cell, "results_draft") or {}

    def summary(self, ds: str, cell: str) -> dict:
        return self.res.summary(ds, cell, "results_draft") or {}

    def compare(self, section: str, a: str, sa: Optional[Stat], b: str, sb: Optional[Stat], metric: str,
                higher_is_better: bool = True, ds: Optional[str] = None, hospital: Optional[str] = None,
                suppress: Optional[str] = None, cells: Optional[tuple] = None) -> str:
        """The rule's verdict, recorded in the comparisons file: from the paired bootstrap CI of runs `cells` (A, B).
        Accuracy is guarded by the majority-class rate of the dataset (or hospital); `suppress` records why a verdict
        is computed but not written as a ranking. The pre-Phase-11 verdict is kept as old_relation."""
        if sa is None or sb is None:
            return f"cannot be compared with ({MISSING})"
        maj = self.res.majority(ds, hospital) if ds and metric == "accuracy" else None
        old = guarded_relation(sa.mean, sa.std, sb.mean, sb.std, higher_is_better, metric, maj, self.margin)
        paired = self.stats.compare(ds, cells[0], cells[1], metric, hospital) if (ds and cells) else None
        if paired is None:
            self.res.note_missing(f"paired bootstrap {ds}|{cells}|{metric}|{hospital}", "results_draft")
        ci = None if paired is None else paired["ci"]
        rel = ci_guarded_relation(ci, sa.mean, sb.mean, higher_is_better, metric, maj, self.margin)
        entry = {"section": section, "a": a, "b": b, "metric": metric, "a_mean": sa.mean, "a_std": sa.std,
                 "b_mean": sb.mean, "b_std": sb.std, "n_repeats": [sa.n, sb.n], "relation": rel,
                 "rule": "bootstrap_ci", "old_relation": old, "cells": list(cells) if cells else None,
                 "hospital": hospital, "dataset": ds}
        if paired is not None:
            entry.update(diff_mean=paired["diff_mean"], diff_std=paired["diff_std"], ci=paired["ci"],
                         resamples=paired["resamples"], resamples_valid=paired["resamples_valid"])
        if maj is not None:
            entry.update(majority=maj, majority_margin=self.margin)
        if suppress and rel == "outperforms":
            entry["written_as"] = "not ranked: " + suppress
        self.comparisons.append(entry)
        if rel == NOT_RANKED:
            return f"{rel} {NEAR_MAJORITY.format(margin=self.margin, maj=maj)}"
        if suppress and rel == "outperforms":
            return "is not ranked against"
        return rel

    def clean_fer(self, ds: str, cell: Optional[str]) -> Optional[float]:
        return clean_fer(self.res, ds, cell, "results_draft")

    def skew_caveat(self, fer: Optional[float]) -> Optional[str]:
        return skew_caveat(fer, self.fer_max)

    def recall_sentence(self) -> str:
        return recall_sentence(self.res, self.spec, "results_draft")

    def _captions(self) -> dict[str, str]:
        p = self.figures / "captions.md"
        if not p.exists():
            self.res.note_missing(p, "results_draft")
            return {}
        caps, name = {}, None
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                name = line[3:].strip()
            elif name and line.strip():
                caps[name] = line.strip()
                name = None
        return caps

    # ── writing ───────────────────────────────────────────────────────────────
    def h(self, level: int, text: str) -> None:
        self.lines += ["#" * level + " " + text, ""]

    def p(self, text: str) -> None:
        self.lines += [text, ""]

    def bullets(self, items: list[str]) -> None:
        self.lines += [f"- {i}" for i in items] + [""]

    def table(self, name: str) -> None:
        p = self.tables / f"{name}.md"
        if not p.exists():
            self.res.note_missing(p, "results_draft")
            self.p(f"*Table {name}: {MISSING} (not built)*")
            return
        lines = p.read_text(encoding="utf-8").splitlines()
        title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else name
        self.lines += [f"**Table ({name}). {title}**", ""] + lines[1:] + [""]

    def figure(self, name: str) -> None:
        p = self.figures / f"{name}.png"
        if not p.exists():
            self.res.note_missing(p, "results_draft")
            self.p(f"*Figure {name}: {MISSING} (not built)*")
            return
        rel = os.path.relpath(p, self.out.parent)
        self.lines += [f"![{name}]({rel})", "", f"*Figure ({name}).* {self.captions.get(name, MISSING)}", ""]

    # ── sections ──────────────────────────────────────────────────────────────
    def setup(self) -> None:
        sp, r = self.spec, self.res
        self.h(2, "Experimental Setup")
        self.h(3, "A. Data")
        heart, tcga = self.config("heart", "fedavg_lr"), self.config("tcga", "fedavg_lr")
        if heart:
            d = heart["data"]
            per = ", ".join(f"{h} {v['patients']} ({v['positives']} positive)" for h, v in d["per_hospital"].items())
            self.p(f"**{dataset_label(sp, 'heart')}.** {vocab_rows(sp, 'heart', heart) or MISSING} rows from "
                   f"{len(d['per_hospital'])} hospitals (the `dataset` column); label: heart disease present. Paper "
                   f"mode drops the columns ca, thal and slope and then every incomplete row ({d['dropped_incomplete']} "
                   f"rows), leaving {d['patients']} patients, {d['positives']} with heart disease: {per}.")
        if tcga:
            d = tcga["data"]
            per = ", ".join(f"{h} {v['patients']} ({v['deaths']} deaths)" for h, v in d["per_hospital"].items())
            aft = self.config("tcga", "fedxgb_bagging_aft").get("data", {})
            self.p(f"**{dataset_label(sp, 'tcga')}.** {vocab_rows(sp, 'tcga', tcga) or MISSING} cases in the clinical "
                   f"workbook; the {len(d['per_hospital'])} largest source sites are the hospitals: {d['patients']} "
                   f"patients and {d['deaths']} deaths ({per}). Binary task: the death indicator. Survival task: time "
                   f"to death or censoring with an accelerated-failure-time model, follow-up below "
                   f"{aft.get('aft_min_time_days', MISSING)} day clipped up to it "
                   f"({aft.get('aft_times_clipped', MISSING)} patients).")
        if heart and tcga:
            self.p(f"Preprocessing fits nothing on patients: fixed scaling constants and a fixed one-hot vocabulary, "
                   f"{heart['num_features']} encoded features for heart and {tcga['num_features']} for TCGA-BRCA.")

        self.h(3, "B. Protocol")
        s = self.summary("heart", "fedavg_lr")
        cfg = heart.get("config", {})
        mh, mt = heart.get("metrics", {}), tcga.get("metrics", {})
        self.p(f"Repeated stratified {cfg.get('folds', MISSING)}-fold cross-validation, strata (hospital, label), "
               f"{len(s.get('repeats', [])) or MISSING} repeats with seeds {seeds_text([s.get('repeat_seeds', [])])}; "
               f"every hospital trains only on its own rows. Each method runs {cfg.get('rounds', MISSING)} rounds and "
               "is evaluated after every round; the final-round out-of-fold predictions of all folds are pooled and "
               "every metric is computed once per repeat, then summarised as mean ± std over the repeats. Metrics: "
               f"{', '.join(metric_label(sp, m) for m in mh.get('metrics', []))} (heart); "
               f"{', '.join(metric_label(sp, m) for m in mt.get('metrics', []))} (TCGA-BRCA, plus accuracy computed "
               f"from the out-of-fold predictions). F1: {heart.get('f1_threshold_rule', MISSING)}. "
               f"{str(heart.get('accuracy_rule', MISSING)).capitalize()}.")

        self.h(3, "C. Models and federation")
        lr, rf = self.config("heart", "fedavg_lr").get("config", {}), self.config("heart", "fed_rf").get("config", {})
        xg = self.config("heart", "fedxgb_bagging").get("config", {})
        dp_eps = [eps_text(self.summary(ds, c).get("epsilon")) for ds, c in
                  [("heart", c) for c in sp["dp_tables"]["dp_sweep_heart"]["cells"]]]
        dpc = self.config("heart", sp["dp_tables"]["dp_sweep_heart"]["cells"][0]).get("config", {})
        ch = self.config("heart", "chain_fedavg_lr").get("chain", {})
        self.p(f"Logistic regression: {lr.get('local_epochs', MISSING)} local epochs per round, batch "
               f"{lr.get('batch_size', MISSING)}, SGD learning rate {lr.get('lr', MISSING)}, class-weighted loss; "
               "FedSGD takes one full-batch step per round and FedAvg averages the hospitals' deltas weighted by "
               f"sample count. Random forest: {rf.get('trees_per_hospital', MISSING)} trees per hospital per round, "
               f"depth {rf.get('max_depth', MISSING)}, minimum leaf {rf.get('min_samples_leaf', MISSING)}; the "
               f"federated forest concatenates every hospital's trees. XGBoost: {xg.get('xgb_trees_per_round', MISSING)} "
               f"new trees per hospital per round, depth {xg.get('xgb_max_depth', MISSING)}, learning rate "
               f"{xg.get('xgb_learning_rate', MISSING)}, subsample {xg.get('xgb_subsample', MISSING)}, aggregated by "
               "FedXGBBagging (every hospital boosts from the broadcast global margin; the server keeps all new trees and "
               "adds the average of the round's new boosters over the contributing hospitals to the global margin, a "
               "correction made after the held-out log-loss curves showed that summing them overshoots). Local-only models score each patient with their own hospital's model; centralized training "
               "pools every hospital's rows with the same budget.")
        self.p(f"Differential privacy (logistic model only): DP-SGD with target ε in {', '.join(dp_eps)} "
               f"(∞ = no DP), δ = {dpc.get('delta', MISSING)}, clipping norm {dpc.get('max_grad_norm', MISSING)}, "
               "sigma solved once over all steps of a fold-run. Blockchain-FL (commit only) serializes every update "
               "without pickle, gives it a content ID, signs it with the hospital's Ed25519 key (test keys), verifies "
               "it, commits the round as a Merkle root and anchors the root; the server aggregates the updates rebuilt "
               f"from the committed bytes. Sweeps use the mock anchor; one run per dataset mines blocks on the existing "
               f"proof-of-work chain (difficulty {ch.get('difficulty', MISSING)}). Ours adds screening to that "
               "pipeline (and DP to the logistic model).")

        self.h(3, "D. Attacks and screening")
        oc = self.config("heart", "ours_lr_eps4_sign_flip")
        att, scr = oc.get("attack", {}), oc.get("screening", {})
        tc = self.config("heart", "ours_rf")
        probe = (tc.get("screening") or {}).get("probe", {})
        taus_path = self.res.root / "calibration" / "taus_survival.json"
        taus = self.res.need(taus_path, "results_draft") or {}
        self.p(f"One of the K hospitals is malicious ({att.get('attackers_per_repeat', MISSING)} per repeat), chosen "
               f"by {att.get('chosen_by', MISSING)}, from the first round, with non-adaptive attacks: sign_flip, "
               "scaling and label_flip for the logistic model, label_flip_trees and random_trees for the tree models. "
               "The logistic screen is the existing FedVerify-Forensics aggregator; the tree screen compares each "
               f"hospital's new trees on a public probe set of {probe.get('rows', MISSING)} schema rows (no patient "
               "data) by a robust z of the distance to the median prediction. tau is never set by hand: it comes from "
               f"a calibration at seed {taus.get('seed', MISSING)} ({taus.get('cells_harvested', MISSING)} cells, "
               f"{taus.get('rounds', MISSING)} rounds, {taus.get('folds', MISSING)} folds; every hospital in turn plays "
               f"the attacker), policy {scr.get('tau_policy', MISSING)} (the highest detection rate at a false-positive "
               "rate of at most 5% on the calibration rounds).")
        ca = self.res.load(self.res.root / "chain_attacks" / "double_spending" / f"seed{sp['seed']}" / "config.json") or {}
        self.p("The reference paper's three blockchain attacks are simulated against our commit layer: every simulated "
               f"round is a real commit round plus one attack attempt, {ca.get('rounds', MISSING)} rounds per attack "
               f"and dataset at seed {ca.get('seed', MISSING)}, the variant and target drawn from the seed. Resilience "
               "S is the share of rounds in which the defence held, as the reference paper defines it.")

    def results(self) -> None:
        sp = self.spec
        self.h(2, "Results")
        self.p(COMPARISON_RULE)

        self.h(3, "A. Model performance")
        for name in sp["method_tables"]:
            self.table(name)
        for name, t in sp["method_tables"].items():
            if name == "methods_tcga_binary":
                continue
            primary = [m for m in t["metrics"] if not m.endswith("_within")]     # the within columns are secondary
            ds, metric = t["dataset"], primary[-1] if t["dataset"] == "heart" else primary[0]
            rows = {r["label"].split(" (")[0]: r for r in t["rows"]}
            sentences = []
            for fam in t["families"]:
                fl = (t.get("family_labels") or {}).get(fam, sp["families"][fam]["label"])
                get = {k: self.stat(ds, rows[k].get(fam), metric) for k in ("Local", "Centralized", "Vanilla FL", "Ours")}
                sec = f"{name}/{fam}"
                cell = {k: rows[k].get(fam) for k in ("Local", "Centralized", "Vanilla FL", "Ours")}
                rel_c = self.compare(sec, "vanilla FL", get["Vanilla FL"], "centralized", get["Centralized"], metric,
                                     ds=ds, cells=(cell["Vanilla FL"], cell["Centralized"]))
                clause = None                       # Phase 12: a federated gain over centralized, qualified in place
                if rel_c == "outperforms":
                    clause = central_gap_clause(self.res, sp, t, fam, metric, "results_draft")
                    if clause:
                        self.comparisons[-1]["clause"] = clause
                rel_l = self.compare(sec, "vanilla FL", get["Vanilla FL"], "local", get["Local"], metric,
                                     ds=ds, cells=(cell["Vanilla FL"], cell["Local"]))
                cav = self.skew_caveat(self.clean_fer(ds, rows["Ours"].get(fam)))
                rel_o = self.compare(sec, "ours", get["Ours"], "vanilla FL", get["Vanilla FL"], metric, ds=ds,
                                     suppress=cav, cells=(cell["Ours"], cell["Vanilla FL"]))
                gain = get["Ours"] is not None and get["Vanilla FL"] is not None and \
                    get["Ours"].mean > get["Vanilla FL"].mean
                sentences.append(f"{fl}: federated training ({_s(get['Vanilla FL'])}) {rel_c} centralized training "
                                 f"({_s(get['Centralized'])})" + (f" — {clause} —" if clause else "")
                                 + f" and {rel_l} local-only models ({_s(get['Local'])}); ours "
                                 f"({_s(get['Ours'])}) {rel_o} vanilla FL" + (f"; {cav}." if cav and gain else "."))
            self.p(f"**{dataset_label(sp, ds)}, {metric_label(sp, metric)}.** " + " ".join(sentences))
        self.baseline_paragraph()
        self.reference_paragraph()
        self.within_paragraph()
        self.table("paired_comparisons")
        self.paired_paragraph()
        self.table("paired_comparisons_within")
        self.paired_within_paragraph()
        same, differ = [], []
        pairs = [("fedavg_lr", "chain_fedavg_lr"), ("fed_rf", "chain_fed_rf"), ("fedxgb_bagging", "chain_fedxgb_bagging")]
        for ds in ("heart", "tcga"):
            for base, chain in pairs + ([("fedxgb_bagging_aft", "chain_fedxgb_bagging_aft")] if ds == "tcga" else []):
                a, b = self.summary(ds, base), self.summary(ds, chain)
                if a and b:
                    (same if a["mean_std"] == b["mean_std"] else differ).append(f"{chain} ({dataset_label(sp, ds)})")
        self.p(f"Committing every update leaves learning unchanged: the pooled metrics of {_join(same) or MISSING} equal "
               "those of the same method without the commit exactly, because the server aggregates the updates "
               "rebuilt from the committed bytes" + (f"; they differ for {_join(differ)}." if differ else "."))

        self.h(3, "B. Per hospital")
        for name in list(sp["per_hospital_tables"]) + list(sp["per_hospital_method_tables"]):
            self.table(name)
        for name, t in sp["per_hospital_method_tables"].items():
            ds, metric = t["dataset"], t["metric"]
            pairs = [(t["rows"][i], t["rows"][i + 1]) for i in range(0, len(t["rows"]), 3)]   # (local, federated)
            out = []
            for local, fed in pairs:
                sl, sf = self.summary(ds, local["cell"]), self.summary(ds, fed["cell"])
                if not sl or not sf:
                    out.append(f"{fed['label']} vs {local['label']}: {MISSING}")
                    continue
                groups: dict[str, list[str]] = {}
                for h, block in sf.get("per_hospital_mean_std", {}).items():
                    a, b = block.get(metric), (sl.get("per_hospital_mean_std", {}).get(h) or {}).get(metric)
                    if h in (sl.get("excluded_hospitals") or []) or not a or not b or a["mean"] is None or b["mean"] is None:
                        continue
                    rel = self.compare(f"{name}/{h}", fed["label"], Stat(a["mean"], a["std"], a["n_repeats"]),
                                       local["label"], Stat(b["mean"], b["std"], b["n_repeats"]), metric,
                                       ds=ds, hospital=h, cells=(fed["cell"], local["cell"]))
                    groups.setdefault(rel, []).append(h)
                out.append(f"{fed['label']} " + "; ".join(self._at(rel, hs, ds) for rel, hs in groups.items()) + ".")
            self.p(f"**{dataset_label(sp, ds)}, per-hospital {metric_label(sp, metric)}.** " + " ".join(out))

        self.h(3, "C. Training dynamics")
        for ds in ("heart", "tcga"):
            self.figure(f"fig5_accuracy_{ds}")
            self.figure(f"fig6_logloss_{ds}")
        for ds, metric in (("heart", "roc_auc"), ("tcga", "c_index")):
            bits = []
            for cell in sp["figures"]["per_hospital_methods"]:
                s = self.summary(ds, cell)
                curves = [rp["pooled_by_round"].get(metric) for rp in s.get("repeats", [])]
                if not curves or any(c is None or any(v is None for v in c) for c in curves):
                    continue
                means = [sum(c[t] for c in curves) / len(curves) for t in range(len(curves[0]))]
                best = max(range(len(means)), key=lambda t: means[t])
                bits.append(f"{cell} peaks at round {best + 1} ({means[best]:.3f}) and ends at round {len(means)} at "
                            f"{means[-1]:.3f}")
            if bits:
                self.p(f"**{dataset_label(sp, ds)}.** Pooled {metric_label(sp, metric)} over rounds (mean over repeats): "
                       f"{'; '.join(bits)}.")
        self.logloss_paragraph()

        self.h(3, "D. Differential privacy")
        for name, t in sp["dp_tables"].items():
            self.table(name)
            self.figure(f"figx_dp_{t['dataset']}")
            ds, metric = t["dataset"], t["metrics"][0] if t["dataset"] == "tcga" else t["metrics"][-1]
            ref = self.stat(ds, t["cells"][-1], metric)
            bits = []
            for cell in t["cells"][:-1]:
                e = eps_text(self.summary(ds, cell).get("epsilon"))
                st = self.stat(ds, cell, metric)
                bits.append(f"at ε = {e} ({_s(st)}) it {self.compare(name, f'eps={e}', st, 'no DP', ref, metric, ds=ds, cells=(cell, t['cells'][-1]))} "
                            "the non-private model")
            realised = [self.summary(ds, c).get("max_realized_eps") for c in t["cells"][:-1]]
            self.p(f"**{dataset_label(sp, ds)}, {metric_label(sp, metric)}** (no DP: {_s(ref)}): {'; '.join(bits)}. "
                   "The largest realised ε over every hospital, fold and repeat is "
                   + ", ".join(MISSING if v is None else f"{v:.3f}" for v in realised) + " for those targets.")

        self.h(3, "E. Blockchain overhead")
        self.table("overhead_chain")
        self.table("overhead_stages")
        self.figure("fig7_block_times")
        self.figure("fig8_chain_overhead")
        bits = []
        for e in sp["overhead"]["chain_cells"]:
            m = self.res.need(self.res.run_dir(e["dataset"], e["cell"]) / "chain_metrics.json", "results_draft")
            if m:
                bits.append(f"{dataset_label(sp, e['dataset'])}: {m['blocks']} blocks, latency {m['latency_ms']:.1f} "
                            f"ms, throughput {m['throughput_bytes_per_s']:.0f} B/s, send rate "
                            f"{m['send_rate_tx_per_min']:.0f} transactions/min, mean block creation "
                            f"{1000 * m['mean_block_creation_s']:.1f} ms")
        self.p("On the proof-of-work chain (one mined run per dataset, wall-clock on one laptop): " + "; ".join(bits) + ".")
        costs = []
        for e in sp["overhead"]["stage_cells"]:
            st = stage_stats(self.res, e["dataset"], e["cell"])
            if not st or not st["acc"]["serialize_ms"]:
                continue
            total = sum(mean_std(st["acc"][k])[0] for k in STAGE_KEYS)
            train = 1000 * mean_std(st["acc"]["train_wall_s"])[0]
            costs.append(f"{e['cell']} ({dataset_label(sp, e['dataset'])}): {total:.2f} ms to commit a round against "
                         f"{train:.0f} ms of local training")
        self.p("Commit cost per round (all stages, mean over every committed round): " + "; ".join(costs) + ".")

        self.h(3, "F. Blockchain attacks")
        self.table("chain_resilience")
        self.table("chain_attack_variants")
        self.figure("fig9_resilience")
        self.figure("fig10_attack_success")
        ca, rep = sp["chain_attacks"], reported_resilience(sp)
        for ds in ca["datasets"]:
            on = [self.res.chain_attack(a, ds, False, "results_draft") for a in ca["attacks"]]
            off = [self.res.chain_attack(a, ds, True, "results_draft") for a in ca["attacks"]]
            parts = [f"{s['metric']} = {s['resilience_pct']:.1f}%" if s else MISSING for s in on]
            parts_off = [f"{s['resilience_pct']:.1f}%" if s else MISSING for s in off]
            rounds = {s["rounds"] for s in on if s}
            self.p(f"**{dataset_label(sp, ds)}.** With every defence on, {_join(parts)} over "
                   f"{'/'.join(str(r) for r in sorted(rounds)) or MISSING} simulated rounds per attack; with each "
                   f"attack's defences switched off, resilience falls to {_join(parts_off)}.")
        self.p("Myrzashova et al. report " + _join([f"{v:g}% ({ca['labels'][a].lower()})" for a, v in rep.items()
                                                   if v is not None])
               + " for their own simulation; these numbers are reported, not compared (THREAT_MODEL.md lists what is "
               "not defended).")
        self.bullets(chain_caveats(self.res, sp))

        self.h(3, "G. Poisoning and screening")
        for name, t in sp["attack_tables"].items():
            self.table(name)
        for ds in ("heart", "tcga"):
            self.figure(f"figx_detection_{ds}")
        for name, t in sp["attack_tables"].items():
            ds, metric = t["dataset"], t["metrics"][-1] if t["dataset"] == "heart" else t["metrics"][0]
            out = []
            for fam, f in t["families"].items():
                model = f.get("label") or sp["families"][fam]["label"]
                fer = self.clean_fer(ds, f["ours"])
                cav = self.skew_caveat(fer)
                out.append(f"{model}: {cav}." if cav else
                           f"{model}: screening a clean run excludes {_pct(fer)} of honest hospital-rounds.")
                for a in f["attacks"]:
                    s_off, s_on = self.stat(ds, f"{f['ours']}_{a}_noscreen", metric), self.stat(ds, f"{f['ours']}_{a}", metric)
                    d = (self.summary(ds, f"{f['ours']}_{a}").get("attack") or {}).get("detection_pooled_over_repeats") or {}
                    rel = self.compare(f"{name}/{fam}/{a}", "screened", s_on, "unscreened", s_off, metric, ds=ds,
                                       suppress=cav, cells=(f"{f['ours']}_{a}", f"{f['ours']}_{a}_noscreen"))
                    out.append(f"Under {a}, the screened run ({_s(s_on)}) {rel} the unscreened one ({_s(s_off)}); the "
                               f"screen caught {_pct(d.get('recall'))} of attacker rounds (recall) with precision "
                               f"{_pct(d.get('precision'))}, against a clean false-exclusion rate of {_pct(fer)}.")
            self.p(f"**{dataset_label(sp, ds)} ({name}), {metric_label(sp, metric)}.** " + " ".join(out))
        self.p(self.recall_sentence())

        self.h(3, "H. Related work")
        self.table("table_iii_settings")
        self.table("table_iv_results")
        self.p("Rows from other studies are reported as published in Myrzashova et al. (Tables III and IV of that "
               "paper); they use other datasets, models and systems, so this draft does not compare them with ours.")

    def _at(self, rel: str, hs: list[str], ds: str) -> str:
        if rel.startswith(NOT_RANKED):
            return f"{NOT_RANKED} the local model at {_join(hs)} (accuracy within {self.margin:g} of that hospital's " \
                   "majority-class rate)"
        return f"{rel} the local model at {_join(hs)}"

    def within_paragraph(self) -> None:
        """Phase 11: pooled vs within-hospital discrimination, from the method tables' secondary columns."""
        from .paired_stats import included_hospitals
        sp = self.spec
        bits = []
        for name, t in sp["method_tables"].items():
            within = [m for m in t["metrics"] if m.endswith("_within")]
            if not within:
                continue
            wm = within[0]
            pm = wm[: -len("_within")]
            ds = t["dataset"]
            fam_labels = {f: (t.get("family_labels") or {}).get(f, sp["families"][f]["label"]) for f in t["families"]}
            pairs = []
            for row in t["rows"]:
                if row.get("diagnostic"):           # Phase 12: a diagnostic row is not a study method, never ranked
                    continue
                for f in t["families"]:
                    cell = row.get(f, "n/a")
                    if cell == "n/a":
                        continue
                    a, b = self.stat(ds, cell, pm), self.stat(ds, cell, wm)
                    if a and b:
                        pairs.append((f"{fam_labels[f]}: {row['label'].split(' (ε')[0]}", a, b, cell))
            if not pairs:
                continue
            site = self.site_rate_stat(ds, wm) if self.res.site_rate_cell(ds) else None
            bits.append(self.highest_text(f"{name}/within_top", f"{dataset_label(sp, ds)} ({name}), "
                                          f"{metric_label(sp, wm)}", ds, wm, pairs)
                        + (f"; the site-rate reference scores {_s(site)}" if site else "") + ".")
        out = "; ".join(f"{h} ({why})" for ds in ("heart", "tcga") for h, why in included_hospitals(self.res, ds, sp)[1].items())
        self.p("**Within hospital.** Pooled ROC-AUC and C-index compare patients from different hospitals too, so part of "
               "what they measure is how outcome rates differ between hospitals. The method tables therefore also give "
               "both metrics within hospital: each hospital's own out-of-fold patients, combined weighted by their "
               f"comparable pairs (left out, fewer than 5 of the rarer class: {out or 'none'}). The highest score is "
               "compared with the runner-up by the same paired interval as every other comparison (a run with "
               "identical predictions, such as a commit-only twin, is not a runner-up). " + " ".join(bits) +
               " A score that is constant within a hospital gets exactly 0.5 there (Predict all alive does); the "
               "site-rate reference falls slightly below 0.5 because its rate is computed on the other folds, so it "
               "differs a little between folds whenever a hospital's events do not split evenly over them, which "
               "ranks a fold's own events slightly lower.")

    def highest_text(self, section: str, where: str, ds: str, metric: str, pairs: list) -> str:
        """Phase 12: "highest score" (never "best"), with the runner-up compared through the rule; pairs are (label,
        pooled Stat, Stat on `metric`, cell). A run with identical predictions to the top one is not a runner-up."""
        ranked = sorted(pairs, key=lambda x: -x[2].mean)          # stable: table order among equal means
        top, twins, ru = ranked[0], [], None
        for c in ranked[1:]:
            o = self.stats.compare(ds, top[3], c[3], metric)
            if o is not None and o["ci"] == [0.0, 0.0] and not any(o["per_repeat_diff"]):
                twins.append(c[0])
                continue
            ru = c
            break
        text = f"{where}: highest score {top[0]} at {_s(top[2])} (pooled {_s(top[1])})"
        if ru is not None:
            n = len(self.comparisons)
            rel = self.compare(section, top[0], top[2], ru[0], ru[2], metric, ds=ds, cells=(top[3], ru[3]))
            e = self.comparisons[-1] if len(self.comparisons) > n else {}
            d = (f"paired difference {e['diff_mean']:+.3f}, 95% CI {e['ci'][0]:+.3f} to {e['ci'][1]:+.3f}"
                 if e.get("ci") else MISSING)
            text += (f", not distinguishable from {ru[0]} ({_s(ru[2])}; {d})" if rel == NOT_DISTINGUISHABLE else
                     f", which {rel} the runner-up {ru[0]} ({_s(ru[2])}; {d})")
        if twins:
            text += (f" ({_join(twins)} {'scores' if len(twins) == 1 else 'score'} the same, with identical "
                     "predictions)")
        return text

    def record_once(self, section: str, ds: str, cells: tuple, metric: str) -> None:
        """Phase 12: write a comparison the text makes into the comparisons file, unless it is already there."""
        if any(c.get("dataset") == ds and c.get("cells") == list(cells) and c["metric"] == metric
               for c in self.comparisons):                  # heart and TCGA cells share names: key on the dataset
            return
        self.compare(section, cells[0], self.stat(ds, cells[0], metric), cells[1], self.stat(ds, cells[1], metric),
                     metric, ds=ds, cells=cells)

    def record_key(self, k: dict, section: str) -> None:
        """Write a key comparison into the comparisons file (same cells, metric and rule as the tables)."""
        sa, sb = self.stat(k["dataset"], k["a"], k["metric"]), self.stat(k["dataset"], k["b"], k["metric"])
        n = len(self.comparisons)
        self.compare(f"{section}/{k['table']}/{k['family']}", k["a_label"], sa, k["b_label"], sb, k["metric"],
                     ds=k["dataset"], suppress=k["skew"], cells=(k["a"], k["b"]))
        if k.get("clause") and len(self.comparisons) > n:
            self.comparisons[-1]["clause"] = k["clause"]

    def paired_paragraph(self) -> None:
        """Phase 11: what the paired table shows, from the bootstrap cache. Phase 12: through key_comparisons, so the
        Phase 7 rule on gains of Ours and the over-fitting clause on federated gains apply here as in the table."""
        sp = self.spec
        ks = key_comparisons(self.res, sp, where="results_draft")
        for k in ks:
            if k["b_label"] == "Centralized, step-matched (diagnostic)":     # the rest are Section A's comparisons
                self.record_key(k, "paired")
        decisive = [k for k in ks if k["relation"] != NOT_DISTINGUISHABLE]
        ranked = [k for k in decisive if k["verdict"] != NOT_RANKED_GAIN]
        deaths = (self.config("tcga", "fedavg_lr").get("data") or {}).get("deaths", MISSING)
        notes = [f"{dataset_label(sp, k['dataset'])}, {_model_name(k['model'])}: {verdict_phrase(k)}."
                 for k in decisive if k["verdict"] == NOT_RANKED_GAIN or k.get("clause")]
        self.p(f"**Paired comparisons.** Of the {len(ks)} federated-versus-baseline and ours-versus-federated "
               f"comparisons in the table above, {len(decisive)} have a 95% bootstrap interval that excludes 0 and "
               f"{len(ranked)} of those are ranked; the others are not distinguishable at this sample size. "
               + " ".join(notes) + (" " if notes else "") +
               f"TCGA-BRCA intervals are wide because its {deaths} deaths give "
               "few comparable pairs, so most TCGA differences, even large ones, cannot be ranked. The interval "
               "resamples patients with the trained runs held fixed; the std over repeats beside it shows how much the "
               "training runs themselves vary.")

    def paired_within_paragraph(self) -> None:
        """Phase 12: the same key comparisons on the within-hospital metrics (table paired_comparisons_within), each
        recorded in the comparisons file; which verdicts differ from the pooled ones."""
        sp = self.spec
        pooled = {(k["table"], k["family"], k["label"]): k for k in key_comparisons(self.res, sp, where="results_draft")}
        ks = key_comparisons(self.res, sp, within=True, where="results_draft")
        for k in ks:
            self.record_key(k, "paired_within")
        decisive = [k for k in ks if k["relation"] != NOT_DISTINGUISHABLE]
        ranked = [k for k in decisive if k["verdict"] != NOT_RANKED_GAIN]
        changed = [(k, pooled[key]) for k in ks if (key := (k["table"], k["family"], k["label"])) in pooled
                   and pooled[key]["verdict"] != k["verdict"]]
        bits = [f"{dataset_label(sp, k['dataset'])}, {_model_name(k['model'])}: within hospital {verdict_phrase(k)}; "
                f"pooled, {p['a_label']} {p['verdict'] if p['verdict'] != NOT_RANKED_GAIN else 'is not ranked against'} "
                f"{p['b_label']}" for k, p in changed]
        same = [f"{dataset_label(sp, k['dataset'])}, {_model_name(k['model'])}: within hospital {verdict_phrase(k)}"
                for k in decisive if (k["verdict"] == NOT_RANKED_GAIN or k.get("clause")) and
                all(k is not c for c, _ in changed)]
        self.p(f"**Within hospital, key comparisons.** The same {len(ks)} comparisons on the within-hospital metrics "
               "(same patients, resamples and rule; hospitals left out as above): "
               f"{len(decisive)} have a 95% interval that excludes 0 and {len(ranked)} of those are ranked. "
               + (f"The verdict differs from the pooled one in {len(changed)}: " + "; ".join(bits) + ". "
                  if changed else "Every verdict equals the pooled one. ")
               + (" ".join(s + "." for s in same) + " " if same else "")
               + "Within-hospital differences are claimed only through this table's intervals.")

    def overfit_sentence(self) -> str:
        return overfit_text(self.res, "results_draft")

    def site_rate_stat(self, ds: str, metric: str) -> Optional[Stat]:
        return site_stat(self.res, ds, metric, "results_draft")

    def site_rate_sentences(self) -> list[str]:
        """Phase 9: what the site-rate reference scores, and how each table's results relate to it (rule verdicts)."""
        sp = self.spec
        if not sp.get("site_rate"):
            return []
        out = [site_auc_text(self.res, sp, "results_draft")]
        for name, metrics in (sp["site_rate"].get("compare_on") or {}).items():
            t = sp["method_tables"].get(name)
            if not t:
                continue
            ds = t["dataset"]
            fam_labels = {f: (t.get("family_labels") or {}).get(f, sp["families"][f]["label"]) for f in t["families"]}
            for m in metrics:
                ref = self.site_rate_stat(ds, m)
                if ref is None:
                    out.append(f"{dataset_label(sp, ds)} ({name}), {metric_label(sp, m)} of the site-rate reference: "
                               f"{MISSING}.")
                    continue
                pairs = []
                for row in t["rows"]:
                    label = row["label"]
                    if "{epsilon}" in label:
                        label = label.split(" (")[0]
                    for f in t["families"]:
                        cell = row.get(f, "n/a")
                        if cell == "n/a":
                            continue
                        st = self.stat(ds, cell, m)
                        rel = self.compare(f"{name}/site_rate", f"{fam_labels[f]}: {label}", st,
                                           self.res.site_rate_label(), ref, m, ds=ds, cells=(cell, "@site_rate"))
                        pairs.append((f"{fam_labels[f]}: {label}", rel))
                out.append(f"{dataset_label(sp, ds)} ({name}), against the site-rate reference ({_s(ref)}): "
                           + site_rate_sentence(metric_label(sp, m), pairs))
        return out

    def reference_paragraph(self) -> None:
        """Phase 9: the two reference sentences, then the rule's verdicts against the site-rate row."""
        bits = [self.overfit_sentence()] + self.site_rate_sentences()
        self.p("**Reference rows.** " + " ".join(bits))

    def logloss_paragraph(self) -> None:
        """Phase 7: held-out log-loss of FedXGBBagging per heart hospital, lowest round and last round (reported, not
        used to choose the number of rounds)."""
        from .make_fl_healthcare_figures import fold_runs
        cell, ds = "fedxgb_bagging", "heart"
        runs = fold_runs(self.res, ds, cell)
        if not runs:
            return
        n = max(len(r) for r in runs)
        hosps = list(self.config(ds, cell).get("config", {}).get("hospitals", []))
        bits = []
        for h in hosps:
            curve = []
            for t in range(n):
                xs = [r[t]["hospital_metrics"][h]["test_logloss"] for r in runs if t < len(r)
                      and (r[t].get("hospital_metrics") or {}).get(h, {}).get("test_logloss") is not None]
                curve.append(sum(xs) / len(xs) if xs else None)
            if any(v is None for v in curve):
                continue
            lo = min(range(n), key=lambda t: curve[t])
            bits.append(f"{h} lowest at round {lo + 1} ({curve[lo]:.3f}), {curve[-1]:.3f} at round {n}")
        if bits:
            self.p(f"**{dataset_label(self.spec, ds)}, FedXGBBagging held-out log-loss** (mean over {len(runs)} "
                   f"fold-runs, boosters averaged over hospitals): {'; '.join(bits)}. Where the last round is above the "
                   "lowest, the ensemble over-fits in later rounds; the number of rounds is fixed in advance and is "
                   "not chosen from these test-fold curves.")

    def baseline_paragraph(self) -> None:
        """Phase 7: the 'predict all alive' baseline against the TCGA binary table."""
        sp = self.spec
        t = sp["method_tables"].get("methods_tcga_binary")
        if not t or not self.res.baseline_cell("tcga"):
            return
        ds, maj = "tcga", self.res.majority("tcga")
        base = {}
        for m in t["metrics"]:
            st, vals = self.res.baseline(ds, m, "results_draft")
            base[m] = Stat(*mean_std(vals), len(vals)) if st == "ok" else None
        acc = [st for r in t["rows"] for f in t["families"]
               if r.get(f, "n/a") != "n/a" and (st := self.stat(ds, r.get(f), "accuracy")) is not None]
        tol = self.margin + 1e-9
        near = [st for st in acc if maj is not None and abs(st.mean - maj) <= tol]
        above = [st for st in acc if maj is not None and st.mean - maj > tol]
        rng = f"{min(a.mean for a in acc):.3f} to {max(a.mean for a in acc):.3f}" if acc else MISSING
        verdict = ("no method's accuracy beats predicting everyone alive, and accuracy does not separate the TCGA-BRCA "
                   "methods from the trivial model; ROC-AUC, F1 and the C-index are the informative metrics there"
                   if acc and not above else
                   f"only those {len(above)} are ranked on accuracy")
        self.p(f"**{dataset_label(sp, ds)}, accuracy.** {self.res.baseline_label(ds)} scores "
               + ", ".join(f"{metric_label(sp, m)} {_s(v)}" for m, v in base.items())
               + f"; its accuracy is the majority-class rate ({MISSING if maj is None else f'{maj:.3f}'}). The "
               f"accuracies in the binary table range from {rng}: {len(above)} of {len(acc)} exceed that rate by more "
               f"than {self.margin:g}, {len(near)} lie within {self.margin:g} of it (never ranked) and the rest are "
               f"below it; {verdict}. Accuracy uses the F1 threshold (fitted on the training fold), which trades "
               "accuracy for finding deaths; the per-round accuracy curves used P(event) ≥ 0.5 before Phase 7 and "
               "sat close to each hospital's majority-class rate, which is why accuracy and F1 looked as if they used "
               "different thresholds: they did.")

    def central_xgb_limitation(self) -> str:
        """Phase 8: why TCGA centralized XGBoost is below local and federated, from the diagnostics file."""
        d = self.res.need(self.res.root / "diagnostics" / "central_xgb_tcga.json", "results_draft")
        if not d:
            return f"TCGA centralized XGBoost below local and federated: diagnosis {MISSING}."
        run = {(r["task"], r["method"]): r for r in d["runs"]}
        aft = {m: run.get(("survival", m)) for m in ("local_xgb", "central_xgb", "fedxgb_bagging")}
        binr = {m: run.get(("binary", m)) for m in ("local_xgb", "central_xgb", "fedxgb_bagging")}
        if not all(aft.values()) or not all(binr.values()):
            return f"TCGA centralized XGBoost below local and federated: diagnosis {MISSING}."
        if not all(r["reproduces_results"]["ok"] for r in d["runs"]):
            return ("TCGA centralized XGBoost below local and federated: the diagnosis replay does not reproduce the "
                    "results on disk, so it is not reported.")
        c, f, lo = aft["central_xgb"], aft["fedxgb_bagging"], aft["local_xgb"]
        cb, fb, lb = binr["central_xgb"], binr["fedxgb_bagging"], binr["local_xgb"]
        first, last = c["per_round"][0], c["per_round"][-1]
        fb_last, lb_last, cb_first, cb_last = fb["per_round"][-1], lb["per_round"][-1], cb["per_round"][0], cb["per_round"][-1]
        f_steps = f["per_round"][-1]["sequential_steps"]
        at = next((p for p in c["per_round"] if p["sequential_steps"] == f_steps), None)
        falls = all(b["held_c"] < a["held_c"] for a, b in zip(c["per_round"], c["per_round"][1:]))
        rates = [v["death_rate"] for v in d["sites"].values()]
        so = d["site_only"]["c_index"][0]
        pr = lambda r, k: r["final_pairs"][k]["c"]                      # noqa: E731
        ts = self.spec["method_tables"].get("methods_tcga_survival") or {"rows": []}
        cell = {r["label"]: r.get("xgb") for r in ts["rows"]}
        step_aft = step_matched_cell(ts, "xgb") if ts["rows"] else None

        def rel(a: Optional[str], b: Optional[str], metric: str, target: str) -> tuple[str, Optional[str]]:
            """Phase 12: 'A <relation> target (difference, CI)' through the rule, and the bare relation."""
            if a and b:
                self.record_once("limitations/central_xgb", ds="tcga", cells=(a, b), metric=metric)
            o = self.stats.compare("tcga", a, b, metric) if a and b else None
            if o is None:
                return f"{MISSING} {target}", None
            r = ci_relation(o["ci"])
            return f"{r} {target} ({o['diff_mean']:+.3f}, 95% CI {o['ci'][0]:+.3f} to {o['ci'][1]:+.3f})", r
        local_step, local_step_rel = rel(cell.get("Local"), step_aft, "c_index", "it")
        fed_step_rel = rel(cell.get("Vanilla FL"), step_aft, "c_index", "it")[1]
        fed_within = rel(cell.get("Vanilla FL"), cell.get("Centralized"), "c_index_within", "the over-fitted central model")[0]
        fed_step_within = rel(cell.get("Vanilla FL"), step_aft, "c_index_within", "the step-matched row")[0]
        local_within = rel(cell.get("Local"), cell.get("Centralized"), "c_index_within", "central")[0]
        steps = []                                  # Phase 12: federated vs the step-matched row, in this sentence
        for tname, metric in (("methods_tcga_survival", "c_index"), ("methods_tcga_binary", "roc_auc")):
            t = self.spec["method_tables"].get(tname)
            fed = next((r.get("xgb") for r in (t or {}).get("rows", []) if r["label"] == "Vanilla FL"), None)
            step = step_matched_cell(t, "xgb") if t else None
            if fed and step:
                self.record_once("limitations/central_xgb", ds="tcga", cells=(fed, step), metric=metric)
            o = self.stats.compare("tcga", fed, step, metric) if fed and step else None
            target = "it" if steps else "the step-matched centralized row"
            steps.append(f"{metric_label(self.spec, metric)}: {MISSING}" if o is None else
                         f"{ci_relation(o['ci'])} {target} on {metric_label(self.spec, metric)} ({o['diff_mean']:+.3f}, "
                         f"95% CI {o['ci'][0]:+.3f} to {o['ci'][1]:+.3f})")
        return (f"TCGA centralized XGBoost (AFT C-index {last['held_c']:.3f}, death-indicator ROC-AUC "
                f"{cb_last['held_auc']:.3f}) is below local ({lo['per_round'][-1]['held_c']:.3f}, "
                f"{lb_last['held_auc']:.3f}) and federated XGBoost ({f['per_round'][-1]['held_c']:.3f}, "
                f"{fb_last['held_auc']:.3f}) because it over-fits, and federated XGBoost {_join(steps)}. The "
                "centralized model's training C-index rises from "
                f"{first['train_c']:.3f} to {last['train_c']:.3f} (binary training ROC-AUC {cb_first['train_auc']:.3f} "
                f"to {cb_last['train_auc']:.3f}) while its held-out C-index "
                + ("falls in every round" if falls else "falls overall")
                + f", from {first['held_c']:.3f} after round 1 to {last['held_c']:.3f} after round {last['round']}. It "
                f"holds as many trees as the federated model ({last['trees_total']}), but adds {c['steps_per_round']} "
                f"sequential boosting steps per round against {f['steps_per_round']} for the federated model (whose "
                f"round boosters are averaged since Phase 7) and for local models: {last['sequential_steps']} against "
                f"{f_steps} after the last round. At the same {f_steps} steps it scores "
                + (MISSING if at is None else f"{at['held_c']:.3f}")
                + ", so the step count explains part of the gap; "
                + ("no gap beyond it is shown: on the C-index neither federated XGBoost (above) nor local "
                   f"{local_step[local_step.index('('):]} is distinguishable from the step-matched row"
                   if fed_step_rel == local_step_rel == NOT_DISTINGUISHABLE else
                   "the rest is that one pooled ensemble over-fits faster per step than the average of the hospitals' "
                   f"boosters (local {local_step} on the C-index)")
                + ". Run as its own diagnostic row (rule fixed in "
                f"table.yaml after this diagnosis, so not a blind result), the step-matched centralized model reaches "
                f"{_s(self.stat('tcga', 'central_xgb_steps_aft', 'c_index'))} (C-index) and "
                f"{_s(self.stat('tcga', 'central_xgb_steps', 'roc_auc'))} (ROC-AUC). Site event rates (death rate "
                f"{100 * min(rates):.0f}–{100 * max(rates):.0f}% by site) explain local's lead on pairs of patients "
                f"from different sites (local: {pr(lo, 'cross_site'):.3f} on cross-site pairs, "
                f"{pr(lo, 'within_site'):.3f} within sites; a predictor that knows only the site's death rate reaches "
                f"{so:.3f}, and its ROC-AUC {d['site_only']['roc_auc'][0]:.3f} is not below local's death-indicator ROC-AUC), "
                "not central's drop: within hospital (C-index, by the paired rule), where site rates cannot help, "
                f"federated XGBoost {fed_within} and {fed_step_within}, and local {local_within}. The equal-tree "
                "budget is a "
                "documented design choice, not a configuration bug, so it was not changed after seeing these results "
                "(results/diagnostics/central_xgb_tcga.json; DESIGN_NOTES.md section 8).")

    def limitations(self) -> None:
        sp = self.spec
        self.h(2, "Limitations")
        items = []
        t = self.config("tcga", "fedavg_lr").get("data", {})
        if t:
            per = t["per_hospital"]
            low = min(per, key=lambda h: per[h]["deaths"])
            ts = sp["method_tables"]["methods_tcga_survival"]
            cis = [st.mean for r in ts["rows"] for fam in ts["families"]
                   if (st := self.stat("tcga", r.get(fam), "c_index")) is not None]
            rng = f"{min(cis):.3f} and {max(cis):.3f}" if cis else MISSING
            items.append(f"TCGA-BRCA has only {t['deaths']} deaths among {t['patients']} patients; {low} has "
                         f"{per[low]['deaths']}, too few for a per-hospital C-index or AUC, and the pooled C-index of "
                         f"every method in the survival table lies between {rng}.")
        h = self.config("heart", "fedavg_lr").get("data", {})
        if h:
            per = h["per_hospital"]
            small = min(per, key=lambda k: per[k]["patients"])
            neg = per[small]["patients"] - per[small]["positives"]
            items.append(f"The heart hospitals are small and unbalanced in paper mode ("
                         + ", ".join(f"{k} {v['patients']} rows, {v['positives']} positive" for k, v in per.items())
                         + f"); {small} has {neg} negative{'' if neg == 1 else 's'} among {per[small]['patients']} "
                         "rows, so it gets no local-only model and per-hospital rank metrics there are undefined.")
        ch = self.config("heart", "chain_fedavg_lr").get("chain", {})
        items.append(f"The proof-of-work chain is simulated: one local node in one process (the existing BlockVerify "
                     f"chain, difficulty {ch.get('difficulty', MISSING)}), the sweeps use the mock anchor, and latency, "
                     "throughput and send rate are single-machine wall-clock values, not a network's. No consortium or "
                     "public chain, no FPoW (undefined), no homomorphic encryption.")
        oc = self.config("heart", "ours_lr_eps4_sign_flip").get("attack", {})
        items.append(f"Attackers are non-adaptive and at most {oc.get('attackers_per_repeat', MISSING)} of K; the "
                     "median/MAD screens cannot handle collusion, and the public probe set could be gamed by an "
                     "attacker who knows it. tau is calibrated on the same attack families.")
        taus = self.res.load(self.res.root / "calibration" / "taus_survival.json") or {}
        heart_auc = [e["combined"]["auc"] for k, e in (taus.get("by_key") or {}).items()
                     if k.startswith("dataset=heart") and e.get("combined")]
        if heart_auc:
            items.append(f"Screening on heart is weak: the calibration ROC-AUC of the heart screens lies between "
                         f"{min(heart_auc):.3f} and {max(heart_auc):.3f}, with hospitals that differ strongly in "
                         "size and outcome rate (Section A); screening clean runs excludes honest hospitals more often "
                         "than the calibration target (attack tables, 'Clean false excl.' column). "
                         + self.recall_sentence())
        items.append(self.central_xgb_limitation())
        maj = self.res.majority("tcga")
        if maj is not None:
            items.append(f"On TCGA-BRCA, predicting every patient alive already reaches accuracy {maj:.3f} (the "
                         "majority-class rate), so accuracy there says almost nothing about a model; the draft does not "
                         f"rank accuracies within {self.margin:g} of the majority-class rate.")
        items.append("DP is per training example (record-level), not per patient or per hospital; each fold-run is "
                     "accounted separately; the tree models (random forest, XGBoost) have no DP at all, so their "
                     "trees reveal split thresholds and leaf statistics of patient data to the server.")
        r = self.summary("heart", "fedavg_lr")
        items.append(f"Uncertainty: every ranking rests on a 95% patient-level bootstrap interval of a paired difference "
                     f"over {len(r.get('repeats', [])) or MISSING} repeats of cross-validation on the same data. The "
                     "interval holds the trained runs fixed, so it omits the variation between training runs (reported "
                     "separately as the std over repeats), and it is not a test across datasets. No p-values are "
                     "reported: with 5 repeats an exact test cannot go below 0.0625. VERDICT_CHANGES.md lists every "
                     "verdict whose wording changed from the earlier mean ± std rule.")
        items.append("Keys are test keys derived from the run seed, timestamps are a logical clock, and the chain-"
                     "attack resilience comes from one simulation per dataset.")
        self.bullets(items)

    # ── output ────────────────────────────────────────────────────────────────
    def build(self) -> str:
        self.lines = ["# Federated learning with blockchain commitments for medical prediction: results draft", "",
                      "*Generated by `python3 -m fedverify.analysis.make_results_draft` from the results files; do "
                      "not edit by hand. Comparisons: RESULTS_DRAFT.comparisons.json.*", ""]
        self.setup()
        self.results()
        self.limitations()
        return "\n".join(self.lines).rstrip("\n") + "\n"


def verdict_changes(comparisons: list[dict]) -> str:
    """Phase 11: every generated verdict whose wording changed when the rule moved to the bootstrap CI."""
    changed = [c for c in comparisons if c.get("old_relation") and c["old_relation"] != c["relation"]]
    kinds: dict[tuple, int] = {}
    for c in changed:
        kinds[(c["old_relation"], c["relation"])] = kinds.get((c["old_relation"], c["relation"]), 0) + 1
    lines = ["# Verdicts whose wording changed with the Phase 11 rule", "",
             "*Generated by `python3 -m fedverify.analysis.make_results_draft` with RESULTS_DRAFT.md; do not edit.*", "",
             f"Old rule: {OLD_RULE}", "", f"New rule: {COMPARISON_RULE}", "",
             f"{len(changed)} of {len(comparisons)} generated verdicts changed wording:", ""]
    lines += [f"- {n} × \"{a}\" → \"{b}\"" for (a, b), n in sorted(kinds.items(), key=lambda kv: -kv[1])]
    lines += ["", "| Section | A | B | Metric | A (mean ± std) | B (mean ± std) | Paired difference | 95% CI | Old | New |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for c in changed:
        ci = c.get("ci")
        lines.append(f"| {c['section']} | {c['a']} | {c['b']} | {c['metric']} | {c['a_mean']:.3f} ± {c['a_std']:.3f} | "
                     f"{c['b_mean']:.3f} ± {c['b_std']:.3f} | "
                     + (f"{c['diff_mean']:+.3f} ± {c['diff_std']:.3f}" if "diff_mean" in c else MISSING) + " | "
                     + (f"{ci[0]:+.3f} to {ci[1]:+.3f}" if ci else MISSING) + f" | {c['old_relation']} | {c['relation']} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fedverify.analysis.make_results_draft", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="fedverify/results")
    ap.add_argument("--spec", default=str(SPEC))
    ap.add_argument("--tables-dir", default=None, help="default: <results-dir>/tables")
    ap.add_argument("--figures-dir", default=None, help="default: <results-dir>/figures")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    res = Results(a.results_dir, spec)
    root = Path(a.results_dir)
    out = Path(a.out)
    draft = Draft(res, spec, Path(a.tables_dir) if a.tables_dir else root / "tables",
                  Path(a.figures_dir) if a.figures_dir else root / "figures", out)
    text = draft.build()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".comparisons.json").write_text(json.dumps(draft.comparisons, indent=2) + "\n", encoding="utf-8")
    (out.parent / "VERDICT_CHANGES.md").write_text(verdict_changes(draft.comparisons), encoding="utf-8")
    draft.stats.save()
    for what, tables in sorted(res.missing.items()):
        print(f"[missing] {what}  (needed by {', '.join(tables)})", file=sys.stderr)
    print(f"wrote {out} ({len(text.splitlines())} lines, {len(draft.comparisons)} comparisons); "
          f"{len(res.missing)} missing input(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
