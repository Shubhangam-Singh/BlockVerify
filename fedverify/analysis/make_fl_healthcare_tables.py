"""Phase 5 tables for the FL-healthcare replication (Revision 2 §5). Reads ONLY results files.

    python3 -m fedverify.analysis.make_fl_healthcare_tables
    python3 -m fedverify.analysis.make_fl_healthcare_tables --only methods_heart attack_tcga

Every number comes from a results file: summary.json, config.json, oof.jsonl and rounds.jsonl of a run,
chain_metrics.json, chain_attacks/*/summary.json. Two inputs that are not results files carry no number we typed:
the reference CSVs in fedverify/survival/ref/ (other papers' numbers, as reported) and the vocabulary files whose
SHA-256 every run records (their source row counts give the raw dataset sizes). fedverify/survival/table.yaml only
says which results cell fills which row.

A cell that should exist but has no results renders as an em dash and is listed on stderr; "n/a" marks a combination
that does not exist by design; a metric that is undefined by rule (e.g. a per-hospital C-index with fewer than 5
deaths) renders as an em dash with a dagger and a footnote. Outputs: <results>/tables/<name>.md and <name>.tex.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import yaml

from .make_tables import MISSING, mean_std

NA = "n/a"
UNDEFINED = MISSING + "†"
REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "fedverify" / "survival" / "table.yaml"
STAGES = ("serialize_ms", "digest_ms", "cid_ms", "sign_ms", "verify_ms", "merkle_ms", "anchor_ms")


# ── reading results ──────────────────────────────────────────────────────────
class Results:
    """Read-only view of a results tree. Remembers every missing input so it can be listed on stderr."""

    def __init__(self, root: str | Path, spec: dict):
        self.root = Path(root)
        self.spec = spec
        self.missing: dict[str, list[str]] = {}         # missing input -> the tables that needed it
        self.partial: set[str] = set()
        self._cache: dict[Path, Any] = {}
        self._paired = None

    @property
    def paired(self):
        """One PairedStats (bootstrap cache) per Results, shared by the tables, the draft and the page."""
        if self._paired is None:
            from .paired_stats import PairedStats
            self._paired = PairedStats(self, self.spec)
        return self._paired

    def note_missing(self, what: Any, where: str) -> None:
        tables = self.missing.setdefault(str(what), [])
        table = where.split(":", 1)[0]
        if table not in tables:
            tables.append(table)

    def exp(self, dataset: str) -> str:
        return self.spec["datasets"][dataset]["exp"]

    def run_dir(self, dataset: str, cell: str) -> Path:
        return self.root / self.exp(dataset) / cell / f"seed{self.spec['seed']}"

    def load(self, path: Path) -> Any:
        if path not in self._cache:
            self._cache[path] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        return self._cache[path]

    def need(self, path: Path, where: str) -> Any:
        """Load a results file a table needs; note it on the missing list if it is not there."""
        obj = self.load(path)
        if obj is None:
            self.note_missing(path, where)
        return obj

    def summary(self, dataset: str, cell: str, where: str) -> Optional[dict]:
        s = self.need(self.run_dir(dataset, cell) / "summary.json", where)
        want = self.spec.get("expected_repeats")
        if s is not None and want and len(s.get("repeats", [])) < want:
            self.partial.add(f"{self.exp(dataset)}/{cell}: {len(s['repeats'])} of {want} repeats")
        return s

    def config(self, dataset: str, cell: str, where: str) -> Optional[dict]:
        return self.need(self.run_dir(dataset, cell) / "config.json", where)

    def oof_metric(self, dataset: str, cell: str, repeat: int, metric: str, kind: str) -> Optional[float]:
        """A metric the run did not list, computed from its out-of-fold predictions with the run's own rule."""
        from ..survival.metrics import compute
        path = self.run_dir(dataset, cell) / f"repeat{repeat}" / "oof.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        key = "risk" if kind == "risk" else "pred_log_time"

        def col(name: str, dtype: Any = float) -> np.ndarray:
            return np.array([np.nan if r.get(name) is None else r[name] for r in rows], dtype=dtype)
        return compute([metric], col("time_days"), col("event"), col(key), col("pred"), kind)[metric]

    def metric(self, dataset: str, cell: Optional[str], metric: str, where: str) -> tuple[str, list[float]]:
        """Per-repeat pooled values: ('ok', values) | ('missing', []) | ('na', []) | ('undefined', [])."""
        if cell in (None, NA):
            return "na", []
        s = self.summary(dataset, cell, where)
        if s is None:
            return "missing", []
        if metric.endswith("_within"):                 # Phase 11: within-hospital, from the out-of-fold predictions
            from .paired_stats import within_value
            vals = [within_value(self, self.spec, dataset, cell, r, metric) for r in range(len(s["repeats"]))]
        elif metric in s["metrics"]:
            vals = [rp["pooled"].get(metric) for rp in s["repeats"]]
        elif s.get("score_kind") != "risk" or (metric == "c_index" and dataset == "heart"):
            return "na", []                     # a survival time has no accuracy; heart has no follow-up time
        else:
            vals = [self.oof_metric(dataset, cell, r, metric, s["score_kind"]) for r in range(len(s["repeats"]))]
        if not vals or any(v is None for v in vals):
            return "undefined", []
        return "ok", [float(v) for v in vals]

    # ── Phase 7: the trivial "predict all alive" baseline and the majority-class rate ──
    def baseline_cell(self, dataset: str) -> Optional[str]:
        return ((self.spec.get("baselines") or {}).get(dataset) or {}).get("cell")

    def baseline_label(self, dataset: str) -> str:
        return ((self.spec.get("baselines") or {}).get(dataset) or {}).get("label", "Baseline")

    def oof_arrays(self, dataset: str, cell: str, repeat: int) -> dict:
        path = self.run_dir(dataset, cell) / f"repeat{repeat}" / "oof.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {"time": np.array([np.nan if r["time_days"] is None else r["time_days"] for r in rows], float),
                "event": np.array([r["event"] for r in rows], dtype=np.int64),
                "hospital": np.array([r["hospital"] for r in rows]),
                "fold": np.array([r["fold"] for r in rows], dtype=np.int64)}

    def baseline(self, dataset: str, metric: str, where: str, hospital: Optional[str] = None) -> tuple[str, list[float]]:
        """'Predict all alive': every patient gets score 0 and prediction 0, scored with the run's own metric code on
        the out-of-fold patients of the baseline cell. Per hospital when `hospital` is given (same undefined rule)."""
        from ..survival.metrics import compute, per_hospital
        cell = self.baseline_cell(dataset)
        if cell is None:
            return "na", []
        s = self.summary(dataset, cell, where)
        c = self.config(dataset, cell, where)
        if s is None or c is None:
            return "missing", []
        mc = c.get("metrics") or {}
        vals = []
        if metric.endswith("_within"):                 # Phase 11: a constant score ranks no pair within a hospital
            from .paired_stats import within_value
            vals = [within_value(self, self.spec, dataset, "@all_alive", r, metric) for r in range(len(s["repeats"]))]
            return ("ok", [float(v) for v in vals]) if all(v is not None for v in vals) else ("undefined", [])
        for r in range(len(s["repeats"])):
            a = self.oof_arrays(dataset, cell, r)
            zero = np.zeros(len(a["event"]))
            if hospital is None:
                v = compute([metric], a["time"], a["event"], zero, zero.astype(np.int64), "risk")[metric]
            else:
                per = per_hospital([metric], a["hospital"], a["time"], a["event"], zero, zero.astype(np.int64),
                                   int(mc.get("min_class_count", 0)), mc.get("rank_metrics", []), "risk",
                                   mc.get("event_name", "deaths"))
                v = (per.get(hospital) or {}).get(metric)
            vals.append(v)
        if any(v is None for v in vals):
            return "undefined", []
        return "ok", [float(v) for v in vals]

    # ── Phase 9: the "site base rate only" reference row ──
    def site_rate_cell(self, dataset: str) -> Optional[str]:
        return ((self.spec.get("site_rate") or {}).get("cells") or {}).get(dataset)

    def site_rate_label(self) -> str:
        return (self.spec.get("site_rate") or {}).get("label", "Site base rate only")

    def site_rate(self, dataset: str, metric: str, where: str) -> tuple[str, list[float]]:
        """Per-repeat pooled `metric` of the site-rate predictor (site_rate_oof) on the out-of-fold patients of the
        dataset's site_rate cell, scored with the runs' own metric code."""
        from ..survival.metrics import compute
        cell = self.site_rate_cell(dataset)
        if cell is None:
            return "na", []
        s = self.summary(dataset, cell, where)
        if s is None:
            return "missing", []
        vals = []
        for r in range(len(s["repeats"])):
            if metric.endswith("_within"):
                from .paired_stats import within_value
                vals.append(within_value(self, self.spec, dataset, "@site_rate", r, metric))
                continue
            a = self.oof_arrays(dataset, cell, r)
            score, pred = site_rate_oof(a["hospital"], a["event"], a["fold"])
            vals.append(compute([metric], a["time"], a["event"], score, pred, "risk")[metric])
        if any(v is None for v in vals):
            return "undefined", []
        return "ok", [float(v) for v in vals]

    def majority(self, dataset: str, hospital: Optional[str] = None) -> Optional[float]:
        """Share of the larger outcome class, pooled or for one hospital (from the run's data provenance)."""
        cell = self.baseline_cell(dataset) or "fedavg_lr"
        c = self.load(self.run_dir(dataset, cell) / "config.json")
        if not c:
            return None
        d = c["data"] if hospital is None else c["data"]["per_hospital"].get(hospital)
        if not d:
            return None
        n, ev = int(d["patients"]), int(d.get("positives", d.get("deaths", 0)))
        return max(ev, n - ev) / n if n else None

    def accuracy_rule(self, dataset: str) -> Optional[str]:
        c = self.load(self.run_dir(dataset, self.baseline_cell(dataset) or "fedavg_lr") / "config.json") or {}
        return c.get("accuracy_rule")

    def chain_attack(self, attack: str, dataset: str, defence_off: bool, where: str) -> Optional[dict]:
        name = attack + ("_defence_off" if defence_off else "") + ("" if dataset == "heart" else f"_{dataset}")
        return self.need(self.root / "chain_attacks" / name / f"seed{self.spec['seed']}" / "summary.json", where)


def site_rate_oof(hospital: np.ndarray, event: np.ndarray, fold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Phase 9 reference: every patient's score is their hospital's event rate among the TRAINING rows of their
    fold-run (all other folds of the repeat); nothing else is used. The prediction applies the runs' threshold rule:
    the threshold maximising F1 on the training rows' scores (one pooled threshold), predict an event when
    score >= threshold."""
    from ..survival.metrics import thresholds_for_test
    hospital, event, fold = np.asarray(hospital), np.asarray(event, dtype=np.int64), np.asarray(fold)
    score = np.full(len(event), np.nan)
    pred = np.full(len(event), -1, dtype=np.int64)
    for k in np.unique(fold):
        te, tr = fold == k, fold != k
        rate = {h: float(event[tr & (hospital == h)].mean()) for h in np.unique(hospital[tr])}
        train_score = np.array([rate[h] for h in hospital[tr]])
        score[te] = [rate.get(h, np.nan) for h in hospital[te]]
        thr = thresholds_for_test(event[tr], train_score, hospital[tr], hospital[te], False)
        pred[te] = (score[te] >= thr).astype(np.int64)
    return score, pred


def vocab_rows(spec: dict, dataset: str, config: dict) -> Optional[int]:
    """Raw source rows from the vocabulary file the run used (its SHA-256 is in config.json)."""
    path = REPO / spec["datasets"][dataset]["vocab"]
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != config.get("vocab_sha256"):
        return None
    return int(json.loads(path.read_text(encoding="utf-8"))["source_rows"])


# ── formatting ───────────────────────────────────────────────────────────────
def fmt(status: str, vals: list[float], digits: int = 3) -> str:
    if status == "ok":
        m, s = mean_std(vals)
        return f"{m:.{digits}f} ± {s:.{digits}f}"
    return {"missing": MISSING, "na": NA, "undefined": UNDEFINED}[status]


def fmt_ms_entry(entry: Optional[dict], digits: int = 3) -> tuple[str, bool]:
    """A summary.json {mean, std, n_repeats} block -> (text, undefined?)."""
    if not entry or entry.get("mean") is None:
        return UNDEFINED, True
    return f"{entry['mean']:.{digits}f} ± {entry['std']:.{digits}f}", False


def pct(v: Optional[float], digits: int = 1) -> str:
    return NA if v is None else f"{100.0 * v:.{digits}f}"


def num(v: Optional[float], digits: int = 2) -> str:
    return MISSING if v is None else f"{v:.{digits}f}"


def seeds_text(seed_lists: list[list[int]]) -> str:
    """'0-4' for one consecutive set shared by every run, else every distinct set."""
    distinct = sorted({tuple(s) for s in seed_lists if s})
    if not distinct:
        return MISSING
    parts = []
    for s in distinct:
        consecutive = len(s) > 2 and list(s) == list(range(s[0], s[0] + len(s)))
        parts.append(f"{s[0]}–{s[-1]}" if consecutive else ", ".join(str(x) for x in s))
    return "; ".join(parts)


def eps_text(e: Any) -> str:
    if e in (None, "inf") or (isinstance(e, float) and math.isinf(e)):
        return "∞"
    return f"{float(e):g}"


def metric_label(spec: dict, m: str) -> str:
    return spec["metric_labels"].get(m, m)


# ── emitting ─────────────────────────────────────────────────────────────────
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "±": r"$\pm$", "ε": r"$\varepsilon$",
        "δ": r"$\delta$", "σ": r"$\sigma$", "τ": r"$\tau$", "≤": r"$\le$", "≥": r"$\ge$", "−": r"$-$",
        "—": "---", "–": "--", "†": r"$^{\dagger}$", "×": r"$\times$", "∞": r"$\infty$", "·": r"$\cdot$",
        "’": "'", "“": "``", "”": "''",
        "α": r"$\alpha$", "✓": r"$\surd$", "✗": r"$\times$"}      # Phase 11: the MNIST tables (no extra package)


def tex(s: Any) -> str:
    """LaTeX for pdflatex: special characters escaped, every non-ASCII character mapped (or an error, never a
    silently broken table)."""
    out = "".join(_TEX.get(ch, ch) for ch in str(s))
    bad = sorted({ch for ch in out if ord(ch) > 127})
    if bad:
        raise ValueError(f"no LaTeX mapping for {bad!r} in {str(s)[:60]!r}; add it to _TEX")
    return out


def tex_line_start(s: Any) -> str:
    """tex() for text that follows a line break: LaTeX reads a "[" after \\\\ as the break's optional length."""
    t = tex(s)
    return "{" + t + "}" if t.startswith("[") else t


def md(s: Any) -> str:
    return str(s).replace("|", "\\|")


@dataclass
class Table:
    name: str
    title: str
    caption: str
    header: list[str]
    rows: list[list[str]]
    notes: list[str] = field(default_factory=list)
    groups: Optional[list[tuple[str, int]]] = None     # a second header level: [(label, span), ...]
    columns: Optional[str] = None                      # LaTeX column spec; default l + r...
    wide: bool = False                                 # table* (both columns of an IEEE page)
    size: str = "footnotesize"                         # LaTeX font size inside the float
    colsep: float = 3.0                                # \tabcolsep in pt (LaTeX's default is 6)
    # Phase 11: LaTeX only — split the columns over several floats, each {"suffix", "columns" (indices, 0 implied),
    # "caption", "notes"}; every column must appear in at least one part. Markdown and CSV keep one table.
    parts: Optional[list[dict]] = None

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        head = list(self.header)
        if self.groups:                                # markdown has one header row: prefix the group label
            i = 0
            for label, span in self.groups:
                for j in range(i, i + span):
                    if label:
                        head[j] = f"{label}: {head[j]}"
                i += span
        lines = [f"# {self.title}", "", self.caption, "", "| " + " | ".join(md(h) for h in head) + " |",
                 "|" + "|".join("---" for _ in head) + "|"]
        lines += ["| " + " | ".join(md(c) for c in r) + " |" for r in self.rows]
        if self.notes:
            lines += [""] + [f"- {md(n)}" for n in self.notes]
        (out_dir / f"{self.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with open(out_dir / f"{self.name}.csv", "w", newline="", encoding="utf-8") as f:   # Phase 11: report pack
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(self.rows)

        if self.parts:
            covered = set().union(*(set(p["columns"]) for p in self.parts))
            missing = set(range(1, len(self.header))) - covered
            if missing:
                raise ValueError(f"{self.name}: LaTeX parts drop columns {sorted(missing)}")
            texts = []
            spec_cols = re.findall(r"p\{[^}]*\}|[lrc]", self.columns or ("l" + "r" * (len(self.header) - 1)))
            for part in self.parts:
                idx = [0] + list(part["columns"])
                labels = []                                    # each column's group label, then regroup the subset
                if self.groups:
                    for label, span in self.groups:
                        labels += [label] * span
                groups = None
                if labels:
                    groups = []
                    for i in idx:
                        if groups and groups[-1][0] == labels[i] and i != 0:
                            groups[-1] = (labels[i], groups[-1][1] + 1)
                        else:
                            groups.append((labels[i], 1))
                sub = Table(self.name + part["suffix"], self.title, part.get("caption", self.caption),
                            [self.header[i] for i in idx], [[r[i] for i in idx] for r in self.rows],
                            part.get("notes", []), groups=groups, columns="".join(spec_cols[i] for i in idx),
                            wide=self.wide, size=self.size, colsep=self.colsep)
                texts.append(sub.tex_text())
            (out_dir / f"{self.name}.tex").write_text("\n".join(texts), encoding="utf-8")
            return
        (out_dir / f"{self.name}.tex").write_text(self.tex_text(), encoding="utf-8")

    def tex_text(self) -> str:
        env = "table*" if self.wide else "table"
        cols = self.columns or ("l" + "r" * (len(self.header) - 1))
        ragged = [c.startswith("p") for c in re.findall(r"p\{[^}]*\}|[lrc]", cols)]   # p{} cells: ragged right

        def row(cells: list[str]) -> str:
            # \tabularnewline ends a row even inside a \raggedright cell (where \\ would only break the line)
            return " & ".join(("\\raggedright " if k < len(ragged) and ragged[k] else "") + c
                              for k, c in enumerate(cells)) + " \\tabularnewline"
        # Phase 11: the tabular goes into a box that is shrunk to the line width only when it is wider (never
        # cropped, no column dropped); needs graphicx, as IEEE templates load. \fvbox is allocated once per document.
        t = ["\\ifdefined\\fvbox\\else\\newsavebox{\\fvbox}\\fi",
             f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{tex(self.caption)}}}",
             f"\\label{{tab:{self.name}}}", f"\\{self.size}", f"\\setlength{{\\tabcolsep}}{{{self.colsep:g}pt}}",
             "\\sbox{\\fvbox}{%", f"\\begin{{tabular}}{{{cols}}}", "\\hline"]
        if self.groups:
            cells, rules, i = [], [], 1
            for label, span in self.groups:
                cells.append(f"\\multicolumn{{{span}}}{{c}}{{{tex(label)}}}" if span > 1 else tex(label))
                if label and span:
                    rules.append(f"\\cline{{{i}-{i + span - 1}}}")
                i += span
            t += [" & ".join(cells) + " \\tabularnewline", " ".join(rules)]
        t += [row([tex(h) for h in self.header]), "\\hline"]
        t += [row([tex_line_start(c) if k == 0 else tex(c) for k, c in enumerate(r)]) for r in self.rows]
        t += ["\\hline", "\\end{tabular}}%",
              f"\\ifdefined\\fvcheck\\fvcheck{{{self.name}}}\\fi",      # only the LaTeX check defines \fvcheck
              "\\ifdim\\wd\\fvbox>\\linewidth\\resizebox{\\linewidth}{!}{\\usebox{\\fvbox}}"
              "\\else\\usebox{\\fvbox}\\fi"]
        if self.notes:
            t += ["\\par\\smallskip", "\\begin{minipage}{\\linewidth}\\scriptsize"]
            t += [tex_line_start(n) + ("\\\\" if k < len(self.notes) - 1 else "") for k, n in enumerate(self.notes)]
            t += ["\\end{minipage}"]
        t += [f"\\end{{{env}}}"]
        return "\n".join(t) + "\n"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── shared pieces ────────────────────────────────────────────────────────────
def run_facts(res: Results, dataset: str, cells: list[str], where: str) -> dict:
    """K, folds and repeat seeds of the runs behind a table (from their config.json and summary.json)."""
    seeds, ks, folds = [], set(), set()
    for cell in cells:
        if cell in (None, NA):
            continue
        c = res.load(res.run_dir(dataset, cell) / "config.json")
        s = res.load(res.run_dir(dataset, cell) / "summary.json")
        if c:
            ks.add(len(c["config"]["hospitals"]))
            folds.add(int(c["config"]["folds"]))
        if s:
            seeds.append(list(s.get("repeat_seeds", [])))
    return {"K": "/".join(str(k) for k in sorted(ks)) or MISSING,
            "folds": "/".join(str(k) for k in sorted(folds)) or MISSING,
            "seeds": seeds_text(seeds), "repeats": "/".join(sorted({str(len(s)) for s in seeds})) or MISSING}


def dataset_label(spec: dict, dataset: str) -> str:
    return spec["datasets"][dataset]["label"]


def protocol_caption(spec: dict, dataset: str, facts: dict, what: str) -> str:
    return (f"{dataset_label(spec, dataset)}, K = {facts['K']} hospitals: {what}. Mean ± std over {facts['repeats']} "
            f"repeats of stratified {facts['folds']}-fold cross-validation (seeds {facts['seeds']}); pooled "
            f"out-of-fold predictions.")


def first_epsilon(res: Results, dataset: str, cells: list[str]) -> Optional[str]:
    for cell in cells:
        if cell in (None, NA):
            continue
        s = res.load(res.run_dir(dataset, cell) / "summary.json")
        if s and s.get("epsilon") is not None:
            return eps_text(s["epsilon"])
    return None


# ── Tables III and IV (the reference paper's layout; literature rows + ours) ──
def ours_settings_row(res: Results, spec: dict, dataset: str, where: str) -> list[str]:
    cell = spec["ours"][dataset]["settings"]
    c = res.config(dataset, cell, where)
    if c is None:
        return [f"Ours ({dataset_label(spec, dataset)})"] + [MISSING] * 5
    data, cfg = c["data"], c["config"]
    k, n = len(cfg["hospitals"]), int(data["patients"])
    raw = vocab_rows(spec, dataset, c)
    if raw is None:
        res.note_missing(f"raw row count of {dataset}: {spec['datasets'][dataset]['vocab']} missing or changed "
                         "since the run", where)
    events = int(data.get("positives", data.get("deaths", 0)))
    n_classes = int(events > 0) + int(n - events > 0)
    if dataset == "heart":
        name = f"{dataset_label(spec, dataset)} ({k} hospitals)"
        classes = f"{n_classes} (heart disease present / absent)"
    else:
        name = f"{dataset_label(spec, dataset)} clinical ({k} sites)"
        classes = f"{n_classes} (death indicator) + survival time"
    cases = f"{raw if raw is not None else MISSING} raw / {n} analysed"
    fl = (f"FedXGBBagging (XGBoost, {cfg['xgb_trees_per_round']} trees per hospital per round), FedAvg (logistic); "
          f"{k} hospitals, {cfg['rounds']} rounds")
    chain = c.get("chain") or {}
    bc = f"PoW (difficulty {chain.get('difficulty', MISSING)}), Merkle round commit"
    return [f"Ours ({dataset_label(spec, dataset)})", name, cases, classes, fl, bc]


def table_iii(res: Results, spec: dict) -> Table:
    name = "table_iii_settings"
    lit = read_csv(REPO / spec["literature"]["table_iii"])
    rows = [[r["ref"], r["dataset"], r["no_cases"], r["no_classes"], r["fl_setup"], r["bc_consensus"]] for r in lit]
    rows += [ours_settings_row(res, spec, ds, name) for ds in ("heart", "tcga")]
    notes = [f"Rows other than ours are copied unchanged from {spec['literature']['source']}, Table III "
             f"(fedverify/survival/ref/; checked against the PDF, see ref/SPOT_CHECK.md). Bracketed numbers are that "
             "paper's citation numbers. Every study uses a different dataset, so no row is comparable with ours.",
             "Ours: raw = rows in the source file; analysed = rows the experiments use (heart, paper mode: columns "
             "ca, thal and slope dropped, then incomplete rows dropped; TCGA-BRCA: the five largest sites with usable "
             "follow-up)."]
    return Table(name, "Table III: experimental settings (literature as reported, and ours)",
                 "Comparison of experimental settings: related work as reported by Myrzashova et al., and our two "
                 "datasets.", ["Ref", "Dataset", "No. cases", "No. classes", "FL setup", "BC consensus"], rows, notes,
                 columns="p{2.2cm}p{3.2cm}p{2.0cm}p{2.4cm}p{3.4cm}p{2.3cm}", wide=True)


def resilience_text(res: Results, spec: dict, dataset: str, where: str) -> str:
    parts, rounds = [], set()
    for attack in spec["chain_attacks"]["attacks"]:
        s = res.chain_attack(attack, dataset, False, where)
        if s is None:
            parts.append(f"{spec['chain_attacks']['labels'][attack]} {MISSING}")
            continue
        rounds.add(int(s["rounds"]))
        parts.append(f"{spec['chain_attacks']['labels'][attack]} {s['resilience_pct']:.0f}%")
    r = "/".join(str(x) for x in sorted(rounds)) or MISSING
    return ", ".join(parts) + f" (simulated variants, {r} rounds each)"


def ours_results_row(res: Results, spec: dict, dataset: str, where: str) -> list[str]:
    o = spec["ours"][dataset]
    perf = []
    eps = None
    for p in o["performance"]:
        bits = []
        for m in p["metrics"]:
            st, vals = res.metric(dataset, p["cell"], m, where)
            bits.append(f"{metric_label(spec, m)} {fmt(st, vals)}")
        name = p["name"]
        if p["family"] == "lr":
            eps = first_epsilon(res, dataset, [p["cell"]])
            if eps is not None:
                name += f", ε = {eps}"
        perf.append(f"{name}: " + ", ".join(bits))
    c = res.config(dataset, o["settings"], where) or {}
    chain = c.get("chain") or {}
    m = res.need(res.run_dir(dataset, o["chain_cell"]) / "chain_metrics.json", where)
    latency = f"{m['latency_ms']:.1f} ms" if m and m.get("latency_ms") is not None else MISSING
    security = (f"Ed25519-signed submissions with registered keys and txn counters, SHA-256 content IDs, Merkle round "
                f"roots on a proof-of-work chain (difficulty {chain.get('difficulty', MISSING)}); "
                + (f"DP-SGD ε = {eps} (logistic); " if eps else "")
                + f"screening of poisoned updates; block latency {latency}")
    return [f"Ours ({dataset_label(spec, dataset)})", "; ".join(perf), security,
            resilience_text(res, spec, dataset, where)]


def table_iv(res: Results, spec: dict) -> Table:
    name = "table_iv_results"
    lit = read_csv(REPO / spec["literature"]["table_iv"])
    rows = [[r["ref"], r["model_and_performance"], r["security_measures"], r["resilience_towards_cyberattacks"]]
            for r in lit]
    rows += [ours_results_row(res, spec, ds, name) for ds in ("heart", "tcga")]
    facts = run_facts(res, "heart", [spec["ours"]["heart"]["settings"]], name)
    their = re.search(r"[Ll]atency\s+([0-9][0-9.,]*\s*[a-zA-Z]+)", lit[-1]["security_measures"] if lit else "")
    notes = [f"Rows other than ours are copied unchanged from {spec['literature']['source']}, Table IV, and are "
             "as reported: other datasets, models and systems, so they are not comparable with our rows.",
             (f"The reference paper's table gives its latency as \"{their.group(1)}\", while its abstract and text "
              "give it in milliseconds" if their else "The reference paper reports latency in mixed units")
             + "; we report milliseconds only, as the mean time between consecutive mined blocks.",
             f"Our performance: mean ± std over {facts['repeats']} repeats (seeds {facts['seeds']}) of the clean "
             "'ours' runs (commit + screening; DP for the logistic model). Resilience: the reference paper's "
             "definition, rounds in which the defence held / simulated rounds; THREAT_MODEL.md lists what is not "
             "covered."]
    return Table(name, "Table IV: experimental results (literature as reported, and ours)",
                 "Comparison of experimental results: related work as reported by Myrzashova et al., and ours.",
                 ["Ref", "Model & performance", "Security measures", "Resilience towards cyberattacks"], rows, notes,
                 columns="p{2.0cm}p{5.4cm}p{4.6cm}p{3.8cm}", wide=True)


# ── method tables ────────────────────────────────────────────────────────────
def method_table(res: Results, spec: dict, name: str, t: dict) -> Table:
    ds, fams, metrics = t["dataset"], t["families"], t["metrics"]
    fam_labels = {f: (t.get("family_labels") or {}).get(f, spec["families"][f]["label"]) for f in fams}
    rows, used = [], []
    for row in t["rows"]:
        label = row["label"]
        if "{epsilon}" in label:
            e = first_epsilon(res, ds, [row.get(f) for f in fams])
            label = label.replace("{epsilon}", f"ε = {e}" if e else "ε")
        cells = []
        for f in fams:
            cell = row.get(f, NA)
            used.append(cell)
            for m in metrics:
                st, vals = res.metric(ds, cell, m, f"{name}: {label} / {f} {m}")
                cells.append(fmt(st, vals))
        rows.append([label] + cells)
    if res.baseline_cell(ds):
        rows.append([res.baseline_label(ds)] + [fmt(*res.baseline(ds, m, f"{name}: baseline")) for _ in fams for m in metrics])
    if res.site_rate_cell(ds):
        rows.append([res.site_rate_label()] + [fmt(*res.site_rate(ds, m, f"{name}: site rate")) for _ in fams for m in metrics])
    facts = run_facts(res, ds, used, name)
    what = "pooled out-of-fold " + ", ".join(metric_label(spec, m) for m in metrics)
    notes = ["n/a: does not exist by design (DP-SGD applies to the logistic model only; RF / XGBoost DP is not built).",
             "Ours = commit pipeline (signed submissions, Merkle root, anchor) + screening; the logistic model also "
             "uses DP-SGD. Blockchain-FL = commit only. No attack in any row of this table."]
    notes += baseline_notes(res, spec, ds, metrics)
    notes += site_rate_notes(res, ds)
    notes += diagnostic_notes(res, spec, name, t)
    within = within_notes(res, spec, ds, metrics)
    notes += within
    parts = None
    if t.get("latex_parts"):                          # Phase 11: LaTeX floats over subsets of the metric columns
        parts = []
        for k, subset in enumerate(t["latex_parts"]):
            cols = [1 + fi * len(metrics) + metrics.index(m) for fi in range(len(fams)) for m in subset]
            parts.append({"suffix": "" if k == 0 else f"_part{k + 1}", "columns": sorted(cols),
                          "caption": protocol_caption(spec, ds, facts, "pooled out-of-fold " + ", ".join(
                              metric_label(spec, m) for m in subset)) + ("" if k == 0 else " (continued)"),
                          "notes": [n for n in notes if n not in within] if k == 0 else within})
    if any(UNDEFINED in c for r in rows for c in r):
        notes.append(f"{UNDEFINED}: undefined by rule for that run.")
    groups = [("", 1)] + [(fam_labels[f], len(metrics)) for f in fams]
    header = ["Method"] + [metric_label(spec, m) for _ in fams for m in metrics]
    wide = len(header) > 6
    return Table(name, f"Methods: {dataset_label(spec, ds)} ({', '.join(metric_label(spec, m) for m in metrics)})",
                 protocol_caption(spec, ds, facts, what), header, rows, notes, groups=groups, wide=wide,
                 columns="p{2.0cm}" + "r" * (len(header) - 1), size="scriptsize" if wide else "footnotesize",
                 colsep=2.5 if wide else 3.0, parts=parts)


def baseline_notes(res: Results, spec: dict, ds: str, metrics: list[str]) -> list[str]:
    """Phase 7 footnotes: the single threshold rule, and what the trivial baseline is."""
    notes = []
    if "accuracy" in metrics or "f1" in metrics:
        rule = res.accuracy_rule(ds)
        notes.append("Accuracy and F1 use one decision threshold: " + (rule or MISSING) + ".")
    if res.baseline_cell(ds):
        maj = res.majority(ds)
        notes.append(f"{res.baseline_label(ds)}: every patient predicted alive (score 0) on the same out-of-fold "
                     f"patients; its accuracy is the majority-class rate ({MISSING if maj is None else f'{maj:.3f}'}), "
                     "its F1 is 0 and its ROC-AUC and C-index are 0.5. No accuracy within "
                     f"{spec.get('accuracy_majority_margin', 0.01):g} of that rate is ranked against another.")
    return notes


def within_notes(res: Results, spec: dict, ds: str, metrics: list[str]) -> list[str]:
    """Phase 11: what the within-hospital columns are, and which hospitals they leave out (from the data counts)."""
    if not any(m.endswith("_within") for m in metrics):
        return []
    from .paired_stats import included_hospitals
    keep, out = included_hospitals(res, ds, spec)
    k = int((spec.get("within_hospital") or {}).get("min_count", 5))
    left = "; ".join(f"{h} ({why})" for h, why in out.items()) or "none"
    return [f"Within hospital: each hospital's own out-of-fold patients, combined weighted by their numbers of "
            f"comparable pairs, so differences in outcome rate between hospitals cannot help. Left out (fewer than {k} "
            f"patients of the rarer class): {left}. A score that is constant within a hospital gets 0.5 here."]


def site_rate_notes(res: Results, ds: str) -> list[str]:
    if not res.site_rate_cell(ds):
        return []
    return [f"{res.site_rate_label()}: each patient's score is their hospital's event rate in the training folds of "
            "that fold-run (no features, no model); F1 and accuracy use the same threshold rule as every other row. "
            "Its ROC-AUC and C-index only measure how much the event rate differs between hospitals, so a pooled "
            "score near it says little about the features."]


def diagnostic_notes(res: Results, spec: dict, name: str, t: dict) -> list[str]:
    """Phase 9: the rule of each diagnostic row (table.yaml diagnostic_rows), with its numbers read from the runs, and
    whether its value had been seen before it was run (the Phase 8 diagnosis file)."""
    notes = []
    ds = t["dataset"]
    for row in t["rows"]:
        key = row.get("diagnostic")
        if not key:
            continue
        d = (spec.get("diagnostic_rows") or {}).get(key) or {}
        cell = next((row[f] for f in t["families"] if row.get(f, NA) != NA), None)
        c = res.config(ds, cell, f"{name}: {row['label']}") if cell else None
        if not c:
            notes.append(f"{row['label']}: {MISSING}.")
            continue
        cfg = c["config"]
        n, r, k = int(cfg["xgb_trees_per_round"]), int(cfg["rounds"]), len(cfg["hospitals"])
        text = (f"{row['label']}: centralized XGBoost on the pooled rows with {n} trees per round for {r} rounds, "
                f"{n * r} sequential boosting steps, the step count of the federated and local XGBoost models; the "
                f"Centralized row boosts {k * n} trees per round ({k * n * r} steps) and is unchanged. Every other "
                "setting and seed is the Centralized row's. The rule was fixed in table.yaml before this row was run. "
                "Logistic regression and random forest: n/a (boosting steps exist for XGBoost only).")
        diag = res.load(res.root / d["diagnosis"]) if d.get("diagnosis") else None
        task = cfg.get("task", "binary")
        run = next((x for x in (diag or {}).get("runs", []) if x["task"] == task
                    and x["method"] == d.get("compares_with") and x.get("cell", "").endswith("_aft") == (task == "survival")),
                   None) if ds == "tcga" else None
        if ds == "tcga":
            at = next((p for p in (run or {}).get("per_round", []) if p["sequential_steps"] == n * r), None)
            key_ = "held_c" if task == "survival" else "held_auc"
            label = "C-index" if task == "survival" else "ROC-AUC"
            seen = MISSING if at is None or at.get(key_) is None else f"{at[key_]:.3f}"
            if at is None:
                res.note_missing(res.root / d.get("diagnosis", "diagnostics"), name)
            text += (f" Not a blind result: the Phase 8 diagnosis had already shown the original centralized model's "
                     f"held-out {label} after {n * r} boosting steps ({seen}) before this row was run.")
        else:
            text += " Heart was not diagnosed, so no heart value at this step count had been seen before the run."
        notes.append(text)
    return notes


# ── per-hospital tables ──────────────────────────────────────────────────────
def per_hospital_note(res: Results, spec: dict, ds: str, cells: list[str], hospitals: list[str], metric: str) -> list[str]:
    """Footnotes generated from the runs' configs: who has no local model, and why a rank metric is undefined."""
    notes = []
    for cell in cells:
        c = res.load(res.run_dir(ds, cell) / "config.json")
        if not c:
            continue
        data = c["data"]
        for h in c["config"].get("local_exclude") or []:
            if cell.startswith("local_") and h in data["per_hospital"]:
                ph = data["per_hospital"][h]
                ev = int(ph.get("positives", ph.get("deaths", 0)))
                neg = int(ph["patients"]) - ev
                notes.append(f"{h} has no local model: {neg} negative{'' if neg == 1 else 's'} among "
                             f"{ph['patients']} rows.")
        mc = c.get("metrics") or {}
        if metric in (mc.get("rank_metrics") or []):
            low = []
            for h in hospitals:
                ph = data["per_hospital"].get(h)
                if not ph:
                    continue
                ev = int(ph.get("positives", ph.get("deaths", 0)))
                if min(ev, int(ph["patients"]) - ev) < int(mc.get("min_class_count", 0)):
                    low.append(f"{h}: {ev} {mc.get('event_name', 'events')}")
            if low:
                notes.append(f"{UNDEFINED}: undefined, fewer than {mc['min_class_count']} rows of the rarer class "
                             f"({'; '.join(low)}).")
        break
    return sorted(set(notes), key=notes.index)


def per_hospital_table(res: Results, spec: dict, name: str, t: dict) -> Table:
    ds, metric, hosps = t["dataset"], t["metric"], t["hospitals"]
    agg = t["aggregated"]
    rows, used = [], []
    for row in t["rows"]:
        out = [row["label"]]
        s = res.summary(ds, row["local"], f"{name}: {row['label']} / local")
        used.append(row["local"])
        for h in hosps:
            if s is None:
                out.append(MISSING)
                continue
            entry = (s.get("per_hospital_mean_std") or {}).get(h, {}).get(metric)
            out.append(NA if h in (s.get("excluded_hospitals") or []) else fmt_ms_entry(entry)[0])
        for key in agg:
            used.append(row[key])
            out.append(fmt(*res.metric(ds, row[key], metric, f"{name}: {row['label']} / {key}")))
        rows.append(out)
    if res.baseline_cell(ds):
        b = [res.baseline_label(ds)] + [fmt(*res.baseline(ds, metric, f"{name}: baseline", h)) for h in hosps]
        rows.append(b + [fmt(*res.baseline(ds, metric, f"{name}: baseline"))] * len(agg))
    facts = run_facts(res, ds, used, name)
    m = metric_label(spec, metric)
    notes = [f"Local columns: each hospital's own model, scored on that hospital's out-of-fold patients. "
             f"{', '.join(agg.values())}: the method's pooled out-of-fold {m} over every hospital's patients.",
             "Federated = FedAvg (logistic), tree bagging (random forest), FedXGBBagging (XGBoost); Ours adds the "
             "commit pipeline and screening (and DP-SGD for the logistic model)."]
    notes += per_hospital_note(res, spec, ds, [r["local"] for r in t["rows"]], hosps, metric)
    notes += baseline_notes(res, spec, ds, [metric])
    groups = [("", 1), ("Local model, own hospital", len(hosps)), ("All hospitals, pooled", len(agg))]
    return Table(name, f"Per hospital: {dataset_label(spec, ds)} ({m}; the reference heart paper's Table III layout)",
                 protocol_caption(spec, ds, facts, f"{m} of each hospital's local model, then of each aggregation "
                                                   "method"),
                 ["Model"] + hosps + list(agg.values()), rows, notes, groups=groups, wide=True)


def per_hospital_method_table(res: Results, spec: dict, name: str, t: dict) -> Table:
    ds, metric = t["dataset"], t["metric"]
    hosps: list[str] = []
    for row in t["rows"]:
        c = res.load(res.run_dir(ds, row["cell"]) / "config.json")
        if c:
            hosps = list(c["config"]["hospitals"])
            break
    rows = []
    for row in t["rows"]:
        s = res.summary(ds, row["cell"], f"{name}: {row['label']}")
        out = [row["label"]]
        for h in hosps:
            if s is None:
                out.append(MISSING)
            elif h in (s.get("excluded_hospitals") or []):
                out.append(NA)
            else:
                out.append(fmt_ms_entry((s.get("per_hospital_mean_std") or {}).get(h, {}).get(metric))[0])
        rows.append(out)
    if res.baseline_cell(ds):
        rows.append([res.baseline_label(ds)] + [fmt(*res.baseline(ds, metric, f"{name}: baseline", h)) for h in hosps])
    facts = run_facts(res, ds, [r["cell"] for r in t["rows"]], name)
    m = metric_label(spec, metric)
    notes = ["Each hospital's patients scored by the method (local: by their own hospital's model). n/a: the "
             "hospital has no local model."]
    notes += per_hospital_note(res, spec, ds, [r["cell"] for r in t["rows"]], hosps, metric)
    notes += baseline_notes(res, spec, ds, [metric])
    return Table(name, f"Per hospital and method: {dataset_label(spec, ds)} ({m})",
                 protocol_caption(spec, ds, facts, f"{m} per hospital"), ["Method"] + hosps, rows, notes,
                 wide=True)


# ── DP sweep ─────────────────────────────────────────────────────────────────
def dp_table(res: Results, spec: dict, name: str, t: dict) -> Table:
    ds, metrics = t["dataset"], t["metrics"]
    rows = []
    for cell in t["cells"]:
        s = res.summary(ds, cell, f"{name}: {cell}")
        target = eps_text(s.get("epsilon")) if s else MISSING
        out = [target + (" (no DP)" if target == "∞" else "")]
        out += [fmt(*res.metric(ds, cell, m, f"{name}: {cell} / {m}")) for m in metrics]
        out.append(MISSING if s is None else (NA if s.get("max_realized_eps") is None
                                               else f"{s['max_realized_eps']:.3f}"))
        rows.append(out)
    if res.baseline_cell(ds):
        rows.append([res.baseline_label(ds)] + [fmt(*res.baseline(ds, m, f"{name}: baseline")) for m in metrics] + [NA])
    facts = run_facts(res, ds, t["cells"], name)
    c = res.load(res.run_dir(ds, t["cells"][0]) / "config.json") or {}
    cfg = c.get("config") or {}
    delta, clip = cfg.get("delta", MISSING), cfg.get("max_grad_norm", MISSING)
    notes = [f"DP-SGD (Opacus) at every hospital, sigma solved once over the total steps; δ = {delta}, clipping "
             f"norm C = {clip}. Max realised ε: the largest final ε over every hospital, fold and repeat. ∞: no DP "
             "(the FedAvg code path)."]
    notes += baseline_notes(res, spec, ds, metrics)
    return Table(name, f"DP sweep: {dataset_label(spec, ds)} (logistic regression, DP-FedAvg)",
                 protocol_caption(spec, ds, facts, "privacy budget ε vs utility"),
                 ["Target ε"] + [metric_label(spec, m) for m in metrics] + ["Max realised ε"], rows, notes,
                 columns="p{1.1cm}" + "r" * (len(metrics) + 1), size="scriptsize")


# ── poisoning and screening ─────────────────────────────────────────────────
def detection(res: Results, ds: str, cell: str, where: str) -> Optional[dict]:
    s = res.summary(ds, cell, where)
    return None if s is None else ((s.get("attack") or {}).get("detection_pooled_over_repeats"))


def attack_table(res: Results, spec: dict, name: str, t: dict) -> Table:
    ds, metrics = t["dataset"], t["metrics"]
    rows, used, taus, attackers = [], [], {}, {}
    if res.baseline_cell(ds):
        rows.append([res.baseline_label(ds), "none", "n/a"] + [fmt(*res.baseline(ds, m, f"{name}: baseline"))
                                                               for m in metrics] + [NA] * 4)
    for fam, f in t["families"].items():
        model = f.get("label") or spec["families"][fam]["label"]
        d0 = detection(res, ds, f["ours"], f"{name}: {model} / clean")
        clean_fer = pct(d0["false_exclusion_rate"]) if d0 else MISSING
        plan = [("none", "off", f["undefended"]), ("none", "on", f["ours"])]
        for a in f["attacks"]:
            plan += [(a, "off", f"{f['ours']}_{a}_noscreen"), (a, "on", f"{f['ours']}_{a}")]
        for attack, screen, cell in plan:
            used.append(cell)
            where = f"{name}: {model} / {attack} / screening {screen}"
            out = [model, attack, screen] + [fmt(*res.metric(ds, cell, m, f"{where} / {m}")) for m in metrics]
            if screen == "on":
                d = detection(res, ds, cell, where)
                if d is not None and not d.get("attacker_hospital_rounds"):    # no attacker: nothing to find
                    d = {**d, "precision": None, "recall": None}
                out += [MISSING] * 3 if d is None else [pct(d["recall"]), pct(d["precision"]),
                                                        pct(d["false_exclusion_rate"])]
                out.append(clean_fer)
                s = res.load(res.run_dir(ds, cell) / "summary.json")
                if s and s.get("attack"):
                    taus.setdefault(model, (s["attack"].get("tau"), s["attack"].get("tau_source")))
                    if attack != "none":
                        attackers.setdefault(model, s["attack"].get("attackers_by_repeat"))
            else:
                out += [NA] * 4
            rows.append(out)
    facts = run_facts(res, ds, used, name)
    notes = ["Screening off = the same pipeline without the screen (the undefended row). Detection, pooled over every "
             "hospital-round of all repeats: precision = TP/(TP+FP), recall = TP/(TP+FN), false exclusion = honest "
             "hospitals rejected / honest hospital-rounds, in %. n/a: no screen, no attacker to find (attack none: "
             "the false exclusions are the cost of screening a clean run), or nothing rejected. Clean false excl.: the "
             "false-exclusion rate of the same screen on the clean run (no attacker), repeated on every screened row "
             "of that model so recall and the cost of screening are read side by side."]
    notes += baseline_notes(res, spec, ds, metrics)
    for model, (tau, src) in taus.items():
        if tau is not None:
            notes.append(f"{model}: τ = {tau:.3f} from {src}.")
    for model, att in attackers.items():
        if att:
            notes.append(f"{model}: malicious hospital per repeat (from the repeat seed): "
                         + ", ".join("/".join(a) for a in att) + ".")
    header = ["Model", "Attack", "Screening"] + [metric_label(spec, m) for m in metrics] + \
             ["Recall (%)", "Precision (%)", "False excl. (%)", "Clean false excl. (%)"]
    return Table(name, f"Poisoning attacks and screening: {dataset_label(spec, ds)}",
                 protocol_caption(spec, ds, facts, "one malicious hospital, non-adaptive attacks, utility with and "
                                                   "without screening and detection quality"),
                 header, rows, notes, wide=True, columns="lll" + "r" * (len(header) - 3))


# ── chain attacks (Figs 9-10 as a table) ─────────────────────────────────────
def reported_resilience(spec: dict) -> dict[str, Optional[float]]:
    """The reference paper's own resilience numbers, read from its Table IV row (ref CSV)."""
    lit = read_csv(REPO / spec["literature"]["table_iv"])
    text = lit[-1]["resilience_towards_cyberattacks"] if lit else ""
    out = {}
    for attack, label in spec["chain_attacks"]["reported_names"].items():
        m = re.search(re.escape(label) + r"\s+([0-9]+(?:\.[0-9]+)?)\s*%", text)
        out[attack] = float(m.group(1)) if m else None
    return out


def chain_resilience_table(res: Results, spec: dict) -> Table:
    name = "chain_resilience"
    ca = spec["chain_attacks"]
    reported = reported_resilience(spec)
    rows, ks, rounds = [], {}, set()
    for attack in ca["attacks"]:
        out = [ca["labels"][attack], None]
        for ds in ca["datasets"]:
            on = res.chain_attack(attack, ds, False, f"{name}: {attack} / {ds}")
            off = res.chain_attack(attack, ds, True, f"{name}: {attack} / {ds} / defences off")
            if on:
                out[1] = on["metric"]
                rounds.add(int(on["rounds"]))
                cfgp = res.root / "chain_attacks" / (attack + ("" if ds == "heart" else f"_{ds}")) / \
                    f"seed{spec['seed']}" / "config.json"
                c = res.load(cfgp)
                if c:
                    ks[ds] = len(c["hospitals"])
            out += [num(on["resilience_pct"], 1) if on else MISSING, num(on["attack_success_pct"], 1) if on else MISSING,
                    num(off["resilience_pct"], 1) if off else MISSING]
        out[1] = out[1] or MISSING
        out.append(MISSING if reported.get(attack) is None else f"{reported[attack]:g}")
        rows.append(out)
    groups = [("", 2)] + [(f"{dataset_label(spec, ds)} (K = {ks.get(ds, MISSING)})", 3) for ds in ca["datasets"]] + \
             [("Reported", 1)]
    header = ["Attack", "Metric"] + ["S (%)", "Success (%)", "S, defences off (%)"] * len(ca["datasets"]) + \
             ["Myrzashova et al. (%)"]
    notes = ["S = rounds in which the defence held / simulated rounds × 100; attack success = 100 − S (the reference "
             "paper's definitions). Each round is one real commit round of our pipeline plus one attack attempt, with "
             "the variant and target drawn from the seed. Defences off: the defences of that attack switched off "
             "(tests show each simulated variant then lands)."] + chain_caveats(res, spec)
    caption = (f"Resilience of the commit layer to the reference paper's three blockchain attacks: "
               f"{'/'.join(str(r) for r in sorted(rounds)) or MISSING} simulated rounds per attack, seed "
               f"{spec['seed']}.")
    return Table(name, "Chain-attack resilience (Figs 9 and 10)", caption, header, rows, notes, groups=groups,
                 wide=True, columns="ll" + "r" * (len(header) - 2))


def chain_caveats(res: Results, spec: dict) -> list[str]:
    """Phase 7: why 100% is not comparable with the reference paper, what is covered, and the block-time caveat."""
    ca = spec["chain_attacks"]
    rep = reported_resilience(spec)
    reported = ", ".join(f"{ca['labels'][a].lower()} {v:g}%" for a, v in rep.items() if v is not None)
    covered, uncovered = [], []
    for a in ca["attacks"]:
        s = res.chain_attack(a, ca["datasets"][0], False, "chain_caveats") or {}
        covered.append(f"{ca['labels'][a]}: " + (", ".join(v.replace("_", " ") for v in s.get("by_variant", {}))
                                                  or MISSING))
        uncovered.append(f"{ca['labels'][a]}: " + (", ".join(ca.get("not_covered", {}).get(a, [])) or MISSING))
    c = res.load(res.run_dir("heart", "chain_fedavg_lr_local") / "chain_metrics.json") or {}
    return [f"100% here holds by construction: each simulated variant meets a deterministic check (a signature, a "
            f"counter, a slot, a Merkle root) that it cannot pass, so the number says which variants are covered, not how "
            f"often a defence fails. It is not comparable with the reference paper's reported {reported or MISSING}, "
            "which come from a different system and simulation.",
            "Covered (simulated): " + "; ".join(covered) + ".",
            "Not covered: " + "; ".join(uncovered) + ".",
            f"Our proof-of-work difficulty is {c.get('difficulty', MISSING)} on one local node, so our blocks take "
            f"milliseconds; the reference paper reports {ca.get('reference_block_time', MISSING)}. Block times and "
            "latencies are not comparable between the two."]


def chain_variants_table(res: Results, spec: dict) -> Table:
    name = "chain_attack_variants"
    ca = spec["chain_attacks"]
    rows = []
    for attack in ca["attacks"]:
        sums = {ds: res.chain_attack(attack, ds, False, f"{name}: {attack} / {ds}") for ds in ca["datasets"]}
        offs = {ds: res.chain_attack(attack, ds, True, f"{name}: {attack} / {ds} / off") for ds in ca["datasets"]}
        variants = next((list(s["by_variant"]) for s in sums.values() if s), [])
        for v in variants:
            out = [ca["labels"][attack], v]
            for ds in ca["datasets"]:
                for s in (sums[ds], offs[ds]):
                    b = (s or {}).get("by_variant", {}).get(v)
                    out.append(MISSING if b is None else f"{b['held']}/{b['rounds']}")
            rows.append(out)
    groups = [("", 2)] + [(dataset_label(spec, ds), 2) for ds in ca["datasets"]]
    header = ["Attack", "Variant"] + ["Held (defences on)", "Held (defences off)"] * len(ca["datasets"])
    return Table(name, "Chain attacks by variant", "Rounds in which the defence held / rounds the variant was drawn, "
                 f"seed {spec['seed']}.", header, rows,
                 ["Variants are drawn uniformly per round from the seed; the replay variants need an earlier round."],
                 groups=groups, wide=True, columns="ll" + "r" * (len(header) - 2))


# ── overhead (Fig 8 as tables) ───────────────────────────────────────────────
def overhead_chain_table(res: Results, spec: dict) -> Table:
    name = "overhead_chain"
    rows = []
    for e in spec["overhead"]["chain_cells"]:
        ds, cell = e["dataset"], e["cell"]
        m = res.need(res.run_dir(ds, cell) / "chain_metrics.json", f"{name}: {ds} {cell}")
        if m is None:
            rows.append([dataset_label(spec, ds), cell] + [MISSING] * 6)
            continue
        rows.append([dataset_label(spec, ds), cell, str(m["blocks"]), str(m["difficulty"]), num(m["latency_ms"], 1),
                     num(m["throughput_bytes_per_s"], 0), num(m["send_rate_tx_per_min"], 0),
                     num(m["mean_block_creation_s"], 4)])
    notes = ["As the reference paper defines them: latency = mean wall-clock time between consecutive blocks (ms "
             "only); throughput = total block bytes / total elapsed seconds; send rate = transactions per minute. "
             "Block creation = mean time to build and mine one block. One shared chain per run (every fold), "
             "measured on this laptop; wall-clock values vary between machines and runs."]
    return Table(name, "Chain metrics (proof-of-work chain, Fig 8)",
                 "Blockchain performance of the existing BlockVerify proof-of-work chain, one mined run per dataset.",
                 ["Dataset", "Run", "Blocks", "Difficulty", "Latency (ms)", "Throughput (B/s)", "Send rate (tx/min)",
                  "Block creation (s)"], rows, notes, wide=True, columns="ll" + "r" * 6)


def stage_stats(res: Results, ds: str, cell: str) -> Optional[dict]:
    rd = res.run_dir(ds, cell)
    files = sorted(rd.glob("repeat*/fold*/rounds.jsonl"))
    if not files:
        return None
    acc: dict[str, list[float]] = {k: [] for k in STAGES + ("bytes_committed", "train_wall_s")}
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            c = rec.get("commit")
            if not c or c.get("root") is None:            # skipped rounds commit nothing
                continue
            for k in STAGES + ("bytes_committed",):
                acc[k].append(float(c[k]))
            acc["train_wall_s"].append(float(rec["train_wall_s"]))
    s = res.load(rd / "summary.json") or {}
    c = res.load(rd / "config.json") or {}
    return {"acc": acc, "repeats": len(s.get("repeats", [])), "backend": (c.get("chain") or {}).get("backend")}


def overhead_stage_table(res: Results, spec: dict) -> Table:
    name = "overhead_stages"
    rows = []
    for e in spec["overhead"]["stage_cells"]:
        ds, cell = e["dataset"], e["cell"]
        st = stage_stats(res, ds, cell)
        short = spec["datasets"][ds].get("short", dataset_label(spec, ds))
        if st is None or not st["acc"]["serialize_ms"]:
            res.note_missing(f"{res.run_dir(ds, cell)}/repeat*/fold*/rounds.jsonl", name)
            rows.append([short, cell, MISSING] + [MISSING] * (len(STAGES) + 2))
            continue
        a = st["acc"]
        cells = []
        for k in STAGES:
            m, s = mean_std(a[k])
            cells.append(f"{m:.2f} ± {s:.2f}")
        mb, _ = mean_std(a["bytes_committed"])
        mt, _ = mean_std(a["train_wall_s"])
        rows.append([short, cell, str(len(a["serialize_ms"]))] + cells + [f"{mb / 1024:.1f}", f"{1000 * mt:.0f}"])
    notes = ["Mean ± std over every committed round of the run (all folds and repeats), in ms per round, measured "
             "separately from training. Runs ending in _local anchor on the proof-of-work chain (append the round's "
             "transactions and mine a block); the others use the mock anchor (no I/O). KiB/round: serialized bytes "
             "committed. Train: local training of all hospitals per round, for scale."]
    return Table(name, "Commit pipeline cost per round (Fig 8)",
                 "Per-stage cost of the commit pipeline (serialize, digest, content ID, sign, verify, Merkle, anchor).",
                 ["Data", "Run", "Rounds"] + [k.replace("_ms", "") for k in STAGES] + ["KiB/round", "Train (ms)"],
                 rows, notes, wide=True, size="scriptsize", colsep=2.5, columns="ll" + "r" * (len(STAGES) + 3))


# ── driver ───────────────────────────────────────────────────────────────────
def paired_table(res: Results, spec: dict) -> Table:
    """Phase 11: the key comparisons with their paired differences and bootstrap CIs (table.yaml paired_comparisons)."""
    from .make_results_draft import ci_relation, key_comparisons, verdict_cell
    name, pc = "paired_comparisons", spec.get("paired_comparisons") or {}
    rows, flagged = [], 0

    def add(ds: str, model: str, label: str, a: str, b: str, metric: str, verdict: Optional[str] = None) -> None:
        nonlocal flagged
        out = res.paired.compare(ds, a, b, metric)
        if out is None:
            rows.append([dataset_label(spec, ds), model, label, metric_label(spec, metric)] + [MISSING] * 5)
            return
        lo, hi = out["ci"]
        mark = ""
        if out["diff_std"] > (hi - lo) / 2:          # the repeats vary more than the patient-level interval
            mark, flagged = " †", flagged + 1
        rows.append([dataset_label(spec, ds), model, label, metric_label(spec, metric), f"{out['a_mean']:.3f}",
                     f"{out['b_mean']:.3f}", f"{out['diff_mean']:+.3f} ± {out['diff_std']:.3f}{mark}",
                     f"{lo:+.3f} to {hi:+.3f}", verdict or ci_relation(out["ci"])])
    keys = key_comparisons(res, spec, where=name)      # Phase 12: the verdicts every output shares
    for k in keys:
        add(k["dataset"], k["model"], k["label"], k["a"], k["b"], k["metric"], verdict_cell(k))
    for tname, metric in (pc.get("dp_tables") or {}).items():
        t = spec["dp_tables"][tname]
        for cell in t["cells"][:-1]:
            e = eps_text((res.summary(t["dataset"], cell, name) or {}).get("epsilon"))
            add(t["dataset"], spec["families"]["lr"]["label"], f"DP-FL (ε = {e}) vs no DP", cell, t["cells"][-1], metric)
    pb = spec.get("paired_bootstrap") or {}
    notes = [f"Paired difference A − B: per repeat, both on the same folds; mean ± sample std over the repeats. 95% CI: "
             f"{pb.get('resamples', MISSING)} patient-level bootstrap resamples, stratified by hospital (seed "
             f"{pb.get('seed', MISSING)}), of the mean over repeats of the difference. Verdict: A outperforms B only if "
             "the CI excludes 0. No p-values (with 5 repeats an exact test cannot go below 0.0625).",
             "The CI holds the trained runs fixed and resamples patients; it does not include the variation between "
             "training runs. † marks the comparisons whose std over repeats exceeds half the CI width "
             f"({flagged} of {len(rows)}): there, training randomness matters more than patient sampling, and the CI "
             "alone overstates certainty."] + qualification_notes(keys, spec)
    header = ["Dataset", "Model", "Comparison (A vs B)", "Metric", "A", "B", "Paired difference", "95% CI", "Verdict"]
    return Table(name, "Key comparisons: paired differences and bootstrap intervals",
                 "Paired comparisons of the key methods on the same folds and repeats (Phase 11).", header, rows,
                 notes, wide=True, columns="llp{3.2cm}lrrrrp{2.6cm}", size="scriptsize", colsep=3.0)


def qualification_notes(keys: list[dict], spec: dict) -> list[str]:
    """Phase 12: the notes under a key-comparison table for the verdicts the rule alone does not settle."""
    notes = []
    fer_max = float(spec.get("clean_false_exclusion_max", 0.05))
    if any(k["verdict"] == "not ranked" for k in keys):
        notes.append(f"Not ranked (Phase 7): where the screen of Ours excludes more than {100 * fer_max:.1f}% of honest "
                     "hospital-rounds on a clean run, a gain of Ours is not written as outperforming: it may come from "
                     "dropping a skewed hospital's updates rather than from catching attackers.")
    where = sorted({dataset_label(spec, k["dataset"]) for k in keys if k.get("clause_short")})
    if where:
        notes.append(f"Federated gains over centralized training on {', '.join(where)} (Phase 12) carry, in the same "
                     "cell, what the Phase 8 diagnosis of the centralized model found (DESIGN_NOTES.md section 8) and "
                     "how federated training (A) compares with the step-matched centralized row, from that "
                     "comparison's own interval; for a model the diagnosis did not cover, the cell says so.")
    return notes


def paired_within_table(res: Results, spec: dict) -> Table:
    """Phase 12: the key comparisons on the within-hospital metrics (same patients, resamples and rule as the pooled
    table), next to the pooled verdict."""
    from .make_results_draft import key_comparisons, verdict_cell
    from .paired_stats import included_hospitals
    name = "paired_comparisons_within"
    pooled = {(k["table"], k["family"], k["label"]): k for k in key_comparisons(res, spec, where=name)}
    keys = key_comparisons(res, spec, within=True, where=name)
    rows, flagged = [], 0
    for k in keys:
        out, p = k["stats"], pooled.get((k["table"], k["family"], k["label"]))
        head = [dataset_label(spec, k["dataset"]), k["model"], k["label"], metric_label(spec, k["metric"])]
        if out is None:
            rows.append(head + [MISSING] * 6)
            continue
        lo, hi = out["ci"]
        mark = ""
        if out["diff_std"] > (hi - lo) / 2:
            mark, flagged = " †", flagged + 1
        rows.append(head + [f"{out['a_mean']:.3f}", f"{out['b_mean']:.3f}",
                            f"{out['diff_mean']:+.3f} ± {out['diff_std']:.3f}{mark}", f"{lo:+.3f} to {hi:+.3f}",
                            verdict_cell(k), MISSING if p is None else verdict_cell(p)])
    changed = sum(1 for k in keys if (p := pooled.get((k["table"], k["family"], k["label"]))) and p["verdict"] != k["verdict"])
    left = "; ".join(f"{h} ({why})" for ds in ("heart", "tcga") for h, why in included_hospitals(res, ds, spec)[1].items())
    pb = spec.get("paired_bootstrap") or {}
    notes = [f"Within hospital: only pairs of patients from the same hospital count, over the hospitals with at least "
             f"{(spec.get('within_hospital') or {}).get('min_count', MISSING)} patients of their rarer class (left out: "
             f"{left or 'none'}). Paired difference A − B and its 95% CI exactly as in the pooled table: the same "
             f"{pb.get('resamples', MISSING)} patient-level resamples, stratified by hospital (seed "
             f"{pb.get('seed', MISSING)}); A outperforms B only if the CI excludes 0.",
             f"Pooled verdict: the same comparison's verdict in the pooled table; {changed} of {len(keys)} differ. "
             f"† as in the pooled table ({flagged} of {len(rows)})."] + qualification_notes(keys, spec)
    header = ["Dataset", "Model", "Comparison (A vs B)", "Metric", "A", "B", "Paired difference", "95% CI",
              "Verdict within hospital", "Pooled verdict"]
    return Table(name, "Key comparisons within hospital: paired differences and bootstrap intervals",
                 "The key comparisons on the within-hospital metrics, same folds, repeats and resamples (Phase 12).",
                 header, rows, notes, wide=True, columns="llp{2.6cm}p{1.5cm}rrrrp{2.5cm}p{2.5cm}", size="scriptsize",
                 colsep=2.5)


def build_all(res: Results, spec: dict) -> dict[str, Callable[[], Table]]:
    out: dict[str, Callable[[], Table]] = {"table_iii_settings": lambda: table_iii(res, spec),
                                           "table_iv_results": lambda: table_iv(res, spec)}
    for group, fn in (("method_tables", method_table), ("per_hospital_tables", per_hospital_table),
                      ("per_hospital_method_tables", per_hospital_method_table), ("dp_tables", dp_table),
                      ("attack_tables", attack_table)):
        for name, t in (spec.get(group) or {}).items():
            out[name] = (lambda fn=fn, name=name, t=t: fn(res, spec, name, t))
    out["chain_resilience"] = lambda: chain_resilience_table(res, spec)
    out["chain_attack_variants"] = lambda: chain_variants_table(res, spec)
    out["overhead_chain"] = lambda: overhead_chain_table(res, spec)
    out["overhead_stages"] = lambda: overhead_stage_table(res, spec)
    if spec.get("paired_comparisons"):
        out["paired_comparisons"] = lambda: paired_table(res, spec)
        out["paired_comparisons_within"] = lambda: paired_within_table(res, spec)
    return out


def load_spec(path: str | Path = SPEC) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fedverify.analysis.make_fl_healthcare_tables", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="fedverify/results")
    ap.add_argument("--spec", default=str(SPEC), help="table.yaml: which results cell fills which row")
    ap.add_argument("--out-dir", default=None, help="default: <results-dir>/tables")
    ap.add_argument("--only", nargs="*", default=None, help="table names to build (default: all)")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    res = Results(a.results_dir, spec)
    out_dir = Path(a.out_dir) if a.out_dir else Path(a.results_dir) / "tables"
    builders = build_all(res, spec)
    unknown = sorted(set(a.only or []) - set(builders))
    if unknown:
        ap.error(f"unknown table(s) {unknown}; have {sorted(builders)}")
    for name, build in builders.items():
        if a.only and name not in a.only:
            continue
        build().write(out_dir)
        print(f"  wrote {out_dir}/{name}.md and .tex")
    if res._paired is not None:
        res.paired.save()
    for what, tables in sorted(res.missing.items()):
        print(f"[missing] {what}  (needed by {', '.join(tables)})", file=sys.stderr)
    for line in sorted(res.partial):
        print(f"[partial] {line}", file=sys.stderr)
    print(f"{len(res.missing)} missing input(s), {len(res.partial)} partial run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
