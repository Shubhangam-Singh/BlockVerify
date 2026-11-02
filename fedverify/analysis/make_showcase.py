"""Build the FedVerify-Survival showcase: one interactive page that explains the project and shows its results.

    python3 -m fedverify.analysis.make_showcase
        -> fedverify/results/showcase/index.html            open in a browser (works offline; web fonts need a network)
        -> fedverify/results/showcase/showcase_body.html     the same page without the document skeleton (for publishing)

Every number on the page is read here from the results files (through the Phase-5 readers) and embedded as JSON;
the page's code (showcase_template.html, next to this file) holds layout and wording only. Verdict phrases use the
results draft's comparison rule. The example round is a real committed round: its Merkle root is recomputed from the
stored leaves and checked against the anchored root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

from .make_fl_healthcare_figures import fold_runs
from .make_fl_healthcare_tables import (REPO, SPEC, Results, chain_caveats, dataset_label, eps_text, load_spec,
                                        metric_label, reported_resilience, seeds_text, stage_stats, table_iii, table_iv,
                                        vocab_rows)
from .make_results_draft import (COMPARISON_RULE, NOT_RANKED, central_gap_clause, ci_guarded_relation, clean_fer,
                                 overfit_text, recall_range, recall_sentence, site_auc_text, site_rate_sentence,
                                 skew_caveat)
from .make_tables import mean_std

TEMPLATE = Path(__file__).with_name("showcase_template.html")
SKELETON_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
*,*::before,*::after{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,sans-serif}img{max-width:100%}
[hidden]{display:none!important}</style>
</head>
<body>
"""
EXAMPLE = {"dataset": "heart", "cell": "chain_fedavg_lr_local", "repeat": 0, "fold": 0, "round": 1,
           "screen_cell": "ours_lr_eps4"}


def ms(vals: list) -> Optional[list]:
    if not vals:
        return None
    m, s = mean_std(vals)
    return [round(m, 6), round(s, 6)]


class Collector:
    def __init__(self, res: Results, spec: dict):
        self.res, self.spec = res, spec
        self.margin = float(spec.get("accuracy_majority_margin", 0.01))
        self.fer_max = float(spec.get("clean_false_exclusion_max", 0.05))

    def verdict(self, a: list, b: list, metric: str, ds: str, hospital: Optional[str] = None,
                suppress: Optional[str] = None, cells: Optional[tuple] = None) -> str:
        """The draft's rule (Phase 11): ranked only when the paired bootstrap CI of runs `cells` excludes 0; never on an
        accuracy at the majority-class rate, and no 'outperforms' for a screen that drops too many honest updates."""
        maj = self.res.majority(ds, hospital) if metric == "accuracy" else None
        paired = self.res.paired.compare(ds, cells[0], cells[1], metric, hospital) if cells else None
        rel = ci_guarded_relation(None if paired is None else paired["ci"], a[0], b[0], True, metric, maj, self.margin)
        return NOT_RANKED if suppress and rel == "outperforms" else rel

    def site_stat(self, ds: str, metric: str) -> Optional[list]:
        st, vals = self.res.site_rate(ds, metric, self.where())
        return ms(vals) if st == "ok" else None

    def baseline_stat(self, ds: str, metric: str, hospital: Optional[str] = None) -> Optional[list]:
        st, vals = self.res.baseline(ds, metric, self.where(), hospital)
        return ms(vals) if st == "ok" else None

    def audit(self) -> dict:
        """Phase 7 facts the page states: majority rates, the single threshold rule, recall, the chain caveats."""
        return {"margin": self.margin, "fer_max": self.fer_max,
                "majority": {ds: self.res.majority(ds) for ds in ("heart", "tcga")},
                "accuracy_rule": self.res.accuracy_rule("heart"),
                "baseline": {ds: self.res.baseline_label(ds) for ds in ("heart", "tcga") if self.res.baseline_cell(ds)},
                "recall": {ds: (list(r) if (r := recall_range(self.res, self.spec, ds, self.where())) else None)
                           for ds in ("heart", "tcga")},
                "recall_sentence": recall_sentence(self.res, self.spec, self.where()),
                "chain_caveats": chain_caveats(self.res, self.spec)}

    def where(self) -> str:
        return "showcase"

    def stat(self, ds: str, cell: Optional[str], metric: str) -> Optional[list]:
        st, vals = self.res.metric(ds, cell, metric, self.where())
        return ms(vals) if st == "ok" else None

    def cfg(self, ds: str, cell: str) -> dict:
        return self.res.config(ds, cell, self.where()) or {}

    def summ(self, ds: str, cell: str) -> dict:
        return self.res.summary(ds, cell, self.where()) or {}

    # ── data and protocol ─────────────────────────────────────────────────────
    def datasets(self) -> list[dict]:
        out = []
        for ds in ("heart", "tcga"):
            c = self.cfg(ds, "fedavg_lr")
            if not c:
                continue
            d = c["data"]
            key = "positives" if ds == "heart" else "deaths"
            out.append({"id": ds, "label": dataset_label(self.spec, ds),
                        "short": self.spec["datasets"][ds].get("short", dataset_label(self.spec, ds)),
                        "raw": vocab_rows(self.spec, ds, c),
                        "patients": d["patients"], "events": d[key], "event_word": "with heart disease" if ds == "heart"
                        else "deaths", "event_short": "positive" if ds == "heart" else "deaths",
                        "dropped": d.get("dropped_incomplete", d.get("dropped_no_usable_time")),
                        "features": c["num_features"],
                        "hospitals": [{"name": h, "patients": v["patients"], "events": v[key]}
                                      for h, v in d["per_hospital"].items()],
                        "local_excluded": d.get("local_excluded", [])})
        return out

    def protocol(self) -> dict:
        c = self.cfg("heart", "fedavg_lr").get("config", {})
        s = self.summ("heart", "fedavg_lr")
        return {"folds": c.get("folds"), "rounds": c.get("rounds"), "repeats": len(s.get("repeats", [])),
                "seeds": seeds_text([s.get("repeat_seeds", [])])}

    # ── methods ───────────────────────────────────────────────────────────────
    def methods(self) -> dict:
        views = {}
        groups = {"heart": [("methods_heart", None)],
                  "tcga": [("methods_tcga_survival", None), ("methods_tcga_binary", None)]}
        for ds, names in groups.items():
            metrics = []
            for name, _ in names:
                t = self.spec["method_tables"][name]
                fam_labels = {f: (t.get("family_labels") or {}).get(f, self.spec["families"][f]["label"])
                              for f in t["families"]}
                for m in t["metrics"]:
                    rows = []
                    for r in t["rows"]:
                        label = r["label"]
                        if "{epsilon}" in label:
                            eps = next((eps_text(self.summ(ds, r[f]).get("epsilon")) for f in t["families"]
                                        if r.get(f, "n/a") != "n/a"), None)
                            label = label.replace("{epsilon}", f"ε = {eps}" if eps else "ε")
                        rows.append({"label": label, "cells": {f: (None if r.get(f, "n/a") == "n/a"
                                                                    else self.stat(ds, r[f], m)) for f in t["families"]},
                                     "na": [f for f in t["families"] if r.get(f, "n/a") == "n/a"],
                                     "cell": {f: r[f] for f in t["families"] if r.get(f, "n/a") != "n/a"}})
                    verdicts = {}
                    by = {row["label"].split(" (")[0]: row for row in rows}
                    for f in t["families"]:
                        fl, ce, lo, ou = (by[k]["cells"].get(f) for k in ("Vanilla FL", "Centralized", "Local", "Ours"))
                        cav = skew_caveat(clean_fer(self.res, ds, by["Ours"].get("cell", {}).get(f), self.where()),
                                          self.fer_max)
                        v = {}
                        cell = {k: by[k].get("cell", {}).get(f) for k in ("Vanilla FL", "Centralized", "Local", "Ours")}
                        if fl and ce:
                            v["central"] = self.verdict(fl, ce, m, ds, cells=(cell["Vanilla FL"], cell["Centralized"]))
                            if v["central"] == "outperforms":          # Phase 12: qualified in the same sentence
                                clause = central_gap_clause(self.res, self.spec, t, f, m, self.where())
                                if clause:
                                    v["central_caveat"] = clause
                        if fl and lo:
                            v["local"] = self.verdict(fl, lo, m, ds, cells=(cell["Vanilla FL"], cell["Local"]))
                        if ou and fl:
                            v["ours"] = self.verdict(ou, fl, m, ds, suppress=cav, cells=(cell["Ours"], cell["Vanilla FL"]))
                            if cav and ou[0] > fl[0]:
                                v["ours_caveat"] = cav
                        verdicts[f] = v
                    site_verdict = None
                    site = self.site_stat(ds, m) if self.res.site_rate_cell(ds) else None
                    if site and m in ((self.spec.get("site_rate") or {}).get("compare_on") or {}).get(name, []):
                        pairs = [(f"{fam_labels[f]}: {r['label'].split(' (ε')[0]}",
                                  self.verdict(r["cells"][f], site, m, ds, cells=(r["cell"][f], "@site_rate")))
                                 for r in rows for f in t["families"] if r["cells"].get(f)]
                        site_verdict = {"value": site, "text": site_rate_sentence(metric_label(self.spec, m), pairs)}
                    if self.res.baseline_cell(ds):
                        b = self.baseline_stat(ds, m)
                        rows.append({"label": self.res.baseline_label(ds), "baseline": True,
                                     "note": "all predicted alive",
                                     "cells": {f: b for f in t["families"]}, "na": []})
                    if site:
                        rows.append({"label": self.res.site_rate_label(), "baseline": True,
                                     "note": "site event rate only",
                                     "cells": {f: site for f in t["families"]}, "na": []})
                    metrics.append({"key": m, "label": metric_label(self.spec, m), "rows": rows,
                                    "families": [{"key": f, "label": fam_labels[f]} for f in t["families"]],
                                    "verdicts": verdicts, "site_verdict": site_verdict})
            notes = ([overfit_text(self.res, self.where())] if ds == "tcga" else []) + \
                [site_auc_text(self.res, self.spec, self.where())]
            views[ds] = {"label": dataset_label(self.spec, ds), "metrics": metrics, "notes": notes}
        return views

    def per_hospital(self) -> dict:
        out = {}
        for name, t in self.spec["per_hospital_method_tables"].items():
            ds, metric = t["dataset"], t["metric"]
            hosps = list(self.cfg(ds, t["rows"][0]["cell"]).get("config", {}).get("hospitals", []))
            rows = []
            for r in t["rows"]:
                s = self.summ(ds, r["cell"])
                vals = {}
                for h in hosps:
                    e = (s.get("per_hospital_mean_std") or {}).get(h, {}).get(metric)
                    excluded = h in (s.get("excluded_hospitals") or [])
                    vals[h] = None if excluded or not e or e.get("mean") is None else [e["mean"], e["std"]]
                rows.append({"label": r["label"], "values": vals,
                             "excluded": [h for h in hosps if h in (s.get("excluded_hospitals") or [])]})
            if self.res.baseline_cell(ds):
                rows.append({"label": self.res.baseline_label(ds), "baseline": True, "excluded": [],
                             "values": {h: self.baseline_stat(ds, metric, h) for h in hosps}})
            out[ds] = {"metric": metric, "metric_label": metric_label(self.spec, metric), "hospitals": hosps,
                       "rows": rows}
        return out

    def curves(self) -> dict:
        out = {}
        for ds in ("heart", "tcga"):
            methods = []
            for cell in self.spec["figures"]["per_hospital_methods"]:
                runs = fold_runs(self.res, ds, cell)
                if not runs:
                    self.res.note_missing(self.res.run_dir(ds, cell), self.where())
                    continue
                c = self.cfg(ds, cell)
                hosps = list(c.get("config", {}).get("hospitals", []))
                n = max(len(r) for r in runs)
                per = {}
                for h in hosps:
                    series = {}
                    for key in ("train_acc", "test_acc", "train_logloss", "test_logloss"):
                        vals = []
                        for t in range(n):
                            xs = [r[t]["hospital_metrics"][h][key] for r in runs if t < len(r)
                                  and (r[t].get("hospital_metrics") or {}).get(h)
                                  and r[t]["hospital_metrics"][h].get(key) is not None]
                            vals.append(round(mean_std(xs)[0], 6) if xs else None)
                        series[key] = vals
                    per[h] = series
                agg = {"fedavg": "FedAvg", "fedxgb_bagging": "FedXGBBagging"}.get(c.get("aggregation"), cell)
                fam = {"lr": "logistic regression", "xgb": "XGBoost", "rf": "random forest"}.get(c.get("model_family"), "")
                methods.append({"cell": cell, "label": f"{agg} ({fam})", "hospitals": per, "fold_runs": len(runs)})
            hosps = list(self.cfg(ds, "fedavg_lr").get("config", {}).get("hospitals", []))
            out[ds] = {"methods": methods, "baseline": bool(self.res.baseline_cell(ds)),
                       "majority": {h: self.res.majority(ds, h) for h in hosps}}
        return out

    def dp(self) -> dict:
        out = {}
        for name, t in self.spec["dp_tables"].items():
            ds = t["dataset"]
            pts = []
            for cell in t["cells"]:
                s = self.summ(ds, cell)
                pts.append({"eps": eps_text(s.get("epsilon")) if s else None,
                            "values": {m: self.stat(ds, cell, m) for m in t["metrics"]},
                            "realised": s.get("max_realized_eps") if s else None})
            c = self.cfg(ds, t["cells"][0]).get("config", {})
            out[ds] = {"metrics": [{"key": m, "label": metric_label(self.spec, m)} for m in t["metrics"]],
                       "points": pts, "delta": c.get("delta"), "clip": c.get("max_grad_norm"),
                       "baseline": ({m: self.baseline_stat(ds, m) for m in t["metrics"]}
                                    if self.res.baseline_cell(ds) else None)}
        return out

    # ── ledger ────────────────────────────────────────────────────────────────
    def ledger(self) -> dict:
        blocks, metrics = {}, {}
        for e in self.spec["overhead"]["chain_cells"]:
            ds, cell = e["dataset"], e["cell"]
            p = self.res.run_dir(ds, cell) / "chain_blocks.jsonl"
            if p.exists():
                blocks[ds] = [[b["block_index"], round(1000 * b["creation_wall_s"], 4), b["transactions"],
                               b["block_bytes"]] for b in map(json.loads, p.read_text().splitlines())]
            else:
                self.res.note_missing(p, self.where())
            m = self.res.need(self.res.run_dir(ds, cell) / "chain_metrics.json", self.where())
            if m:
                metrics[ds] = {k: m[k] for k in ("blocks", "difficulty", "latency_ms", "throughput_bytes_per_s",
                                                 "send_rate_tx_per_min", "mean_block_creation_s")}
        stages = []
        for e in self.spec["overhead"]["stage_cells"]:
            st = stage_stats(self.res, e["dataset"], e["cell"])
            if not st or not st["acc"]["serialize_ms"]:
                continue
            a = {k: mean_std(v)[0] for k, v in st["acc"].items()}
            parts = {"serialize": a["serialize_ms"], "fingerprint": a["digest_ms"] + a["cid_ms"],
                     "sign_verify": a["sign_ms"] + a["verify_ms"], "merkle": a["merkle_ms"], "anchor": a["anchor_ms"]}
            stages.append({"dataset": e["dataset"], "run": e["cell"], "backend": st["backend"],
                           "rounds": len(st["acc"]["serialize_ms"]), "parts": {k: round(v, 4) for k, v in parts.items()},
                           "total": round(sum(parts.values()), 4), "train_ms": round(1000 * a["train_wall_s"], 2),
                           "kib": round(a["bytes_committed"] / 1024, 2)})
        return {"blocks": blocks, "metrics": metrics, "stages": stages}

    def example_round(self) -> Optional[dict]:
        ex = EXAMPLE
        rd = self.res.run_dir(ex["dataset"], ex["cell"])
        fold = rd / f"repeat{ex['repeat']}" / f"fold{ex['fold']}"
        cpath, rpath = fold / "commitments.jsonl", fold / "rounds.jsonl"
        if not cpath.exists() or not rpath.exists():
            self.res.note_missing(cpath, self.where())
            return None
        com = next(c for c in map(json.loads, cpath.read_text().splitlines()) if c["round"] == ex["round"])
        rec = next(r for r in map(json.loads, rpath.read_text().splitlines()) if r["round"] == ex["round"])
        cfg = self.cfg(ex["dataset"], ex["cell"])
        blocks = [json.loads(l) for l in (rd / "chain_blocks.jsonl").read_text().splitlines()]
        blk = next(b for b in blocks if b["hash"] == com["txid"])
        sub = com["submissions"][0]
        header = {"family": "lr", "format": 1, "hospital": sub["hospital_id"],
                  "num_samples": rec["per_client_num_samples"][sub["hospital_id"]], "numel": cfg["num_params"],
                  "round": com["round"], "run_id": com["run_id"]}
        leaves = [{"i": lf["index"], "hospital": lf["client_id"], "digest": lf["digest"], "leaf": lf["leaf"]}
                  for lf in com["leaves"]]
        level = [lf["leaf"] for lf in leaves]
        levels = [level]
        while len(level) > 1:                       # the canonical fold: parent = SHA-256(hex(left) || hex(right))
            if len(level) % 2:
                level = level + [level[-1]]
            level = [hashlib.sha256((level[i] + level[i + 1]).encode()).hexdigest() for i in range(0, len(level), 2)]
            levels.append(level)
        screen = None
        sp = self.res.run_dir(ex["dataset"], ex["screen_cell"]) / f"repeat{ex['repeat']}" / f"fold{ex['fold']}" / "rounds.jsonl"
        if sp.exists():
            r1 = next(r for r in map(json.loads, sp.read_text().splitlines()) if r["round"] == ex["round"])
            sc = r1["diag"]["scores"]
            screen = {"run": ex["screen_cell"], "tau": r1["diag"].get("tau"),
                      "scores": {h: round(v.get("combined", 0.0), 4) for h, v in sc.items()},
                      "weights": {h: round(v.get("weight", 0.0), 4) for h, v in sc.items() if "weight" in v},
                      "rejected": r1["diag"].get("rejected", []),
                      "epsilon": eps_text(self.summ(ex["dataset"], ex["screen_cell"]).get("epsilon"))}
        k = len(rec["per_client_num_samples"])
        return {"dataset": ex["dataset"], "dataset_label": dataset_label(self.spec, ex["dataset"]), "run": ex["cell"],
                "repeat": ex["repeat"], "fold": ex["fold"], "round": com["round"], "run_id": com["run_id"],
                "numel": cfg["num_params"], "train_rows": rec["per_client_num_samples"],
                "bytes_committed": rec["commit"]["bytes_committed"], "header": header, "submission": sub,
                "accepted": len(com["leaves"]), "rejected": len(rec["commit"]["rejected"]),
                "leaves": leaves, "levels": levels, "root": com["root"], "root_ok": levels[-1][0] == com["root"],
                "block": {"index": blk["block_index"], "hash": blk["hash"], "transactions": blk["transactions"],
                          "bytes": blk["block_bytes"], "mine_ms": round(1000 * blk["creation_wall_s"], 3),
                          "difficulty": cfg.get("chain", {}).get("difficulty"),
                          "first_block_tx": {"keys": k, "updates": k, "round_root": 1}},
                "content_id_label": cfg.get("chain", {}).get("content_id"), "screen": screen}

    # ── attacks ───────────────────────────────────────────────────────────────
    def chain_attacks(self) -> dict:
        ca = self.spec["chain_attacks"]
        out = {}
        for ds in ca["datasets"]:
            attacks = {}
            K = None
            for a in ca["attacks"]:
                on = self.res.chain_attack(a, ds, False, self.where())
                off = self.res.chain_attack(a, ds, True, self.where())
                c = self.res.load(self.res.root / "chain_attacks" / (a + ("" if ds == "heart" else f"_{ds}")) /
                                  f"seed{self.spec['seed']}" / "config.json") or {}
                K = K or len(c.get("hospitals", []))
                if not on:
                    continue
                attacks[a] = {"label": ca["labels"][a], "metric": on["metric"], "rounds": on["rounds"],
                              "on": on["resilience_pct"], "off": off["resilience_pct"] if off else None,
                              "variants": [{"name": v, "on": [b["held"], b["rounds"]],
                                            "off": [(off or {}).get("by_variant", {}).get(v, {}).get("held"),
                                                    (off or {}).get("by_variant", {}).get(v, {}).get("rounds")]}
                                           for v, b in on["by_variant"].items()]}
            out[ds] = {"label": dataset_label(self.spec, ds), "K": K, "attacks": attacks}
        rep = reported_resilience(self.spec)
        return {"datasets": out, "reported": {a: v for a, v in rep.items() if v is not None},
                "order": ca["attacks"], "labels": ca["labels"]}

    def poisoning(self) -> list[dict]:
        views = []
        for name, t in self.spec["attack_tables"].items():
            ds = t["dataset"]
            metric = t["metrics"][-1] if ds == "heart" else t["metrics"][0]
            fams = []
            for fam, f in t["families"].items():
                s_ours = self.summ(ds, f["ours"])
                det0 = (s_ours.get("attack") or {}).get("detection_pooled_over_repeats") or {}
                cav = skew_caveat(det0.get("false_exclusion_rate"), self.fer_max)
                rows = []
                for a in f["attacks"]:
                    off, on = self.stat(ds, f"{f['ours']}_{a}_noscreen", metric), self.stat(ds, f"{f['ours']}_{a}", metric)
                    sa = self.summ(ds, f"{f['ours']}_{a}")
                    d = (sa.get("attack") or {}).get("detection_pooled_over_repeats") or {}
                    rows.append({"attack": a, "off": off, "on": on, "recall": d.get("recall"),
                                 "precision": d.get("precision"), "fer": d.get("false_exclusion_rate"),
                                 "verdict": self.verdict(on, off, metric, ds, suppress=cav,
                                                         cells=(f"{f['ours']}_{a}", f"{f['ours']}_{a}_noscreen"))
                                 if on and off else None,
                                 "attackers": (sa.get("attack") or {}).get("attackers_by_repeat")})
                fams.append({"family": fam, "label": f.get("label") or self.spec["families"][fam]["label"],
                             "tau": (s_ours.get("attack") or {}).get("tau"),
                             "clean": {"off": self.stat(ds, f["undefended"], metric), "on": self.stat(ds, f["ours"], metric),
                                       "fer": det0.get("false_exclusion_rate")}, "caveat": cav,
                             "attacks": rows})
            views.append({"id": name, "dataset": ds, "label": dataset_label(self.spec, ds) +
                          (" (survival)" if name.endswith("survival") else ""), "metric": metric,
                          "baseline": self.baseline_stat(ds, metric) if self.res.baseline_cell(ds) else None,
                          "baseline_label": self.res.baseline_label(ds),
                          "metric_label": metric_label(self.spec, metric), "families": fams})
        return views

    def literature(self) -> dict:
        out = {}
        for key, build, n_lit in (("t3", table_iii, None), ("t4", table_iv, None)):
            t = build(self.res, self.spec)
            out[key] = {"header": t.header, "rows": t.rows, "ours": 2, "notes": t.notes}
        return out

    def tiles(self, datasets: list[dict], ledger: dict, attacks: dict) -> dict:
        held = total = 0
        for d in attacks["datasets"].values():
            for a in d["attacks"].values():
                held += round(a["on"] * a["rounds"] / 100)
                total += a["rounds"]
        st = next((s for s in ledger["stages"] if s["dataset"] == "heart" and s["run"] == "chain_fedavg_lr"), None)
        return {"datasets": len(datasets), "hospitals": sum(len(d["hospitals"]) for d in datasets),
                "patients": sum(d["patients"] for d in datasets), "attack_rounds_held": held,
                "attack_rounds": total, "commit_ms": st["total"] if st else None,
                "train_ms": st["train_ms"] if st else None}

    def paired(self) -> dict:
        """Phase 11: the key comparisons table (the same Table the draft inserts)."""
        from .make_fl_healthcare_tables import paired_table, paired_within_table
        t, w = paired_table(self.res, self.spec), paired_within_table(self.res, self.spec)
        return {"header": t.header, "rows": t.rows, "notes": t.notes, "rule": COMPARISON_RULE,
                "within": {"header": w.header, "rows": w.rows, "notes": w.notes}}

    def build(self) -> dict:
        datasets = self.datasets()
        ledger = self.ledger()
        attacks = self.chain_attacks()
        out = self._build(datasets, ledger, attacks)
        self.res.paired.save()
        return out

    def _build(self, datasets: list, ledger: dict, attacks: dict) -> dict:
        return {"generated_by": "python3 -m fedverify.analysis.make_showcase", "protocol": self.protocol(),
                "datasets": datasets, "families": {k: v["label"] for k, v in self.spec["families"].items()},
                "methods": self.methods(), "per_hospital": self.per_hospital(), "curves": self.curves(),
                "dp": self.dp(), "ledger": ledger, "round": self.example_round(), "chain_attacks": attacks,
                "poisoning": self.poisoning(), "literature": self.literature(), "audit": self.audit(),
                "paired": self.paired(),
                "tiles": self.tiles(datasets, ledger, attacks)}


def render(data: dict, template: str) -> str:
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    if "__SHOWCASE_DATA__" not in template:
        raise ValueError("template has no __SHOWCASE_DATA__ placeholder")
    return template.replace("__SHOWCASE_DATA__", blob)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fedverify.analysis.make_showcase", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="fedverify/results")
    ap.add_argument("--spec", default=str(SPEC))
    ap.add_argument("--out-dir", default=None, help="default: <results-dir>/showcase")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    res = Results(a.results_dir, spec)
    data = Collector(res, spec).build()
    body = render(data, TEMPLATE.read_text(encoding="utf-8"))
    out = Path(a.out_dir) if a.out_dir else Path(a.results_dir) / "showcase"
    out.mkdir(parents=True, exist_ok=True)
    (out / "showcase_body.html").write_text(body, encoding="utf-8")
    (out / "index.html").write_text(SKELETON_HEAD + body + "\n</body>\n</html>\n", encoding="utf-8")
    for what, tables in sorted(res.missing.items()):
        print(f"[missing] {what}  (needed by {', '.join(tables)})", file=sys.stderr)
    print(f"wrote {out}/index.html ({len(body) // 1024} KiB); {len(res.missing)} missing input(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
