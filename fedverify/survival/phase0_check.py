"""FedVerify-Survival, Phase 0: loader check for the TCGA-BRCA clinical workbook.

This is NOT the Phase-1 loader (that will be survival/data.py). It exists so that every number
INVENTORY.md states about the data is emitted by a script, and so the human can re-run the check.
It reads the workbook, derives hospital / event / time exactly as master-prompt section 3 says,
and compares what it finds with the numbers the prompt expects.

    python3 -I fedverify/survival/phase0_check.py                     # markdown report on stdout
    python3 -I fedverify/survival/phase0_check.py --json out.json     # also write a machine-readable copy

Exit status: 0 = every expectation of the prompt holds; 1 = something differs (stop and report);
2 = the workbook (or openpyxl) is missing.

The workbook is untrusted input, so run this with `-I`: nothing in the current directory or in
the data directory can then shadow an import. `-I` also drops the user site-packages (where
pandas lives on this machine), so that directory is appended to the END of sys.path below; the
standard library and the system packages keep priority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import site
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.append(site.getusersitepackages())  # appended last on purpose, see module docstring

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "fedverify" / "data" / "tcga_brca" / "clinical.xlsx"

# --- what the master prompt (sections 3 and 4) says we should find ---------------------------
EXPECTED_SHAPE = (907, 34)
EXPECTED_SITES: dict[str, tuple[int, int]] = {  # site -> (patients, deaths), before dropping rows
    "BH": (145, 45), "A2": (98, 11), "E2": (87, 6), "A8": (76, 4), "AR": (68, 10),
}
EXPECTED_TOTAL = (474, 76)
BIG_SITE_MIN_PATIENTS = 30            # "the 12 sites with 30+ patients"
EXPECTED_BIG_SITES = 12
EXPECTED_BIG_SITES_UNDER_10_DEATHS = 8
MIN_DEATHS_FOR_PER_HOSPITAL_METRICS = 5   # fewer deaths -> em dash in the per-hospital table
UNIT_TOLERANCE_MONTHS = 0.5
DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = 30.4375

FEATURES_NUMERIC = ("diag_age", "lymph_node", "size_over_20mm")
FEATURES_CATEGORICAL = ("menop_status", "er_status_by_ihc", "pr_status_by_ihc",
                        "her2_status_by_ihc", "histological_type")
FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL
LEAK_PREFIXES = ("OS_", "DFS_", "PFS_")
LEAK_EXACT = ("survival", "survival_event", "days_to_last_follow_up", "diag_date",
              "year_of_birth", "submitter_id", "case_id")
CATEGORICAL_NA_TOKENS = ("[Not Available]", "[Not Evaluated]", "<blank>")  # all become category "NA"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_numeric(values: list[Any]) -> tuple[np.ndarray, Counter]:
    """Parse cells to float. Returns (array with NaN where unusable, Counter of the non-numeric tokens)."""
    out = np.full(len(values), np.nan)
    bad: Counter = Counter()
    for i, v in enumerate(values):
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            if np.isfinite(v):
                out[i] = float(v)
            else:
                bad["<nan>"] += 1
            continue
        s = str(v).strip()
        if s == "":
            bad["<blank>"] += 1
            continue
        try:
            x = float(s)
        except ValueError:
            bad[s] += 1
            continue
        if np.isfinite(x):
            out[i] = x
        else:
            bad[s] += 1
    return out, bad


def categorical_tokens(values: list[Any]) -> Counter:
    c: Counter = Counter()
    for v in values:
        s = str(v).strip()
        c["<blank>" if s == "" else s] += 1
    return c


def read_workbook(path: Path) -> tuple[pd.DataFrame, list[str]]:
    import openpyxl  # imported here so a missing package gives a clear message in main()

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheets = list(wb.sheetnames)
    wb.close()
    # dtype=object + no NA parsing: we want the raw tokens ("[]", "[Not Available]", ...) to be visible.
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl", dtype=object,
                       keep_default_na=False, na_values=[])
    return df, sheets


def derive(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Hospital / event / time per section 3. Returns (per-patient frame, parse diagnostics)."""
    sid = [str(v).strip() for v in df["submitter_id"].tolist()]
    parts = [s.split("-") for s in sid]
    well_formed = [len(p) == 3 and p[0] == "TCGA" and len(p[1]) > 0 for p in parts]
    site = [p[1] if ok else "?" for p, ok in zip(parts, well_formed)]

    event, ev_bad = to_numeric(df["survival_event"].tolist())
    surv_years, sy_bad = to_numeric(df["survival"].tolist())
    fu_days, fu_bad = to_numeric(df["days_to_last_follow_up"].tolist())
    os_months, os_bad = to_numeric(df["OS_MONTHS"].tolist())
    os_event = np.array([1.0 if str(v).strip().startswith("1") else 0.0 if str(v).strip().startswith("0") else np.nan
                         for v in df["OS_STATUS"].tolist()])  # diagnostics only, never a feature

    died = event == 1
    time_days = np.where(died, surv_years * DAYS_PER_YEAR, fu_days)
    usable = np.isfinite(time_days) & np.isin(event, (0.0, 1.0))
    diff = np.abs(time_days / DAYS_PER_MONTH - os_months)

    d = pd.DataFrame({
        "submitter_id": sid, "case_id": [str(v).strip() for v in df["case_id"].tolist()],
        "site": site, "event": event, "surv_years": surv_years, "fu_days": fu_days,
        "time_days": time_days, "usable": usable, "os_months": os_months, "unit_diff": diff,
        "os_event": os_event,
    })
    diag = {
        "submitter_id_malformed": int(len(well_formed) - sum(well_formed)),
        "submitter_id_duplicates": int(len(sid) - len(set(sid))),
        "case_id_duplicates": int(len(d) - d["case_id"].nunique()),
        "event_values": {str(k): int(v) for k, v in Counter(
            "nan" if not np.isfinite(e) else str(int(e)) if e == int(e) else str(e) for e in event).items()},
        "non_numeric_tokens": {
            "survival_event": dict(ev_bad), "survival": dict(sy_bad),
            "days_to_last_follow_up": dict(fu_bad), "OS_MONTHS": dict(os_bad),
        },
        "deaths_without_survival_value": int((died & ~np.isfinite(surv_years)).sum()),
        "censored_with_survival_value": int(((event == 0) & np.isfinite(surv_years)).sum()),
        "censored_without_followup_days": int(((event == 0) & ~np.isfinite(fu_days)).sum()),
        "event_vs_OS_STATUS_rows_compared": int(np.isfinite(os_event).sum()),
        "event_vs_OS_STATUS_disagree": int((np.isfinite(os_event) & (os_event != event)).sum()),
    }
    return d, diag


def site_rows(d: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for site, x in d.groupby("site", sort=True):
        dead = x["event"] == 1
        rows.append({
            "site": str(site), "patients": int(len(x)), "deaths": int(dead.sum()),
            "usable": int(x["usable"].sum()), "usable_deaths": int((dead & x["usable"]).sum()),
            "dropped": int((~x["usable"]).sum()), "dropped_deaths": int((dead & ~x["usable"]).sum()),
            "zero_time": int((x["usable"] & (x["time_days"] <= 0)).sum()),
            "zero_time_deaths": int((dead & x["usable"] & (x["time_days"] <= 0)).sum()),
        })
    rows.sort(key=lambda r: (-r["patients"], r["site"]))
    return rows


def unit_check(d: pd.DataFrame) -> dict[str, Any]:
    chk = d["usable"] & np.isfinite(d["os_months"])
    x = d[chk]
    fail = x[x["unit_diff"] >= UNIT_TOLERANCE_MONTHS]
    return {
        "checked": int(len(x)), "passed": int(len(x) - len(fail)), "failed": int(len(fail)),
        "max_abs_diff_months": float(x["unit_diff"].max()) if len(x) else float("nan"),
        "median_abs_diff_months": float(x["unit_diff"].median()) if len(x) else float("nan"),
        "failures": [
            {"submitter_id": r.submitter_id, "site": r.site, "event": int(r.event),
             "OS_STATUS_event": None if not np.isfinite(r.os_event) else int(r.os_event),
             "time_days": round(float(r.time_days), 2), "os_months": round(float(r.os_months), 3),
             "abs_diff_months": round(float(r.unit_diff), 3)}
            for r in fail.sort_values("unit_diff", ascending=False).itertuples()
        ],
        "failed_deaths": int((fail["event"] == 1).sum()), "failed_censored": int((fail["event"] == 0).sum()),
        "failed_with_time_zero": int((fail["time_days"] <= 0).sum()),
        "failed_in_top5": int(fail["site"].isin(list(EXPECTED_SITES)).sum()),
        "pass_rate": float(1 - len(fail) / len(x)) if len(x) else float("nan"),
        "deaths_checked": int((x["event"] == 1).sum()), "censored_checked": int((x["event"] == 0).sum()),
    }


def feature_profile(df: pd.DataFrame, top5: list[str], site_col: list[str]) -> dict[str, Any]:
    prof: dict[str, Any] = {"numeric": {}, "categorical": {}, "missing_by_site": {}}
    in_top = np.array([s in top5 for s in site_col])
    for c in FEATURES_NUMERIC:
        vals, bad = to_numeric(df[c].tolist())
        ok = vals[np.isfinite(vals)]
        distinct = sorted({float(v) for v in ok})
        prof["numeric"][c] = {
            "numeric": int(len(ok)), "unusable": int(len(vals) - len(ok)), "unusable_tokens": dict(bad),
            "min": float(ok.min()) if len(ok) else None, "max": float(ok.max()) if len(ok) else None,
            "distinct_values": distinct if len(distinct) <= 12 else None, "n_distinct": len(distinct),
        }
        prof["missing_by_site"][c] = {
            s: float(np.mean([not np.isfinite(vals[i]) for i in range(len(vals)) if site_col[i] == s]))
            for s in top5
        }
    for c in FEATURES_CATEGORICAL:
        raw = df[c].tolist()
        toks = categorical_tokens(raw)
        prof["categorical"][c] = {
            "tokens": dict(sorted(toks.items(), key=lambda kv: (-kv[1], kv[0]))),
            "n_na_category": int(sum(v for k, v in toks.items() if k in CATEGORICAL_NA_TOKENS)),
        }
        prof["missing_by_site"][c] = {
            s: float(np.mean([str(raw[i]).strip() in ("", *CATEGORICAL_NA_TOKENS)
                              for i in range(len(raw)) if site_col[i] == s]))
            for s in top5
        }
    prof["top5_rows"] = int(in_top.sum())
    # one-hot width after mapping every NA-like token to the single category "NA" (always kept in the vocab)
    vocab = {}
    for c in FEATURES_CATEGORICAL:
        toks = prof["categorical"][c]["tokens"]
        vocab[c] = len({k for k in toks if k not in CATEGORICAL_NA_TOKENS} | {"NA"})
    prof["vocab_sizes"] = vocab
    prof["input_dim"] = {"categorical_onehot": sum(vocab.values()),
                         "numeric": len(FEATURES_NUMERIC), "missing_indicators": len(FEATURES_NUMERIC),
                         "total": sum(vocab.values()) + 2 * len(FEATURES_NUMERIC)}
    return prof


def leakage_view(columns: list[str]) -> dict[str, Any]:
    prefixed = [c for c in columns if c.startswith(LEAK_PREFIXES)]
    exact = [c for c in LEAK_EXACT if c in columns]
    absent = [c for c in LEAK_EXACT if c not in columns]
    other = [c for c in columns if c not in FEATURES and c not in prefixed and c not in exact]
    return {"prefixed": prefixed, "exact_present": exact, "exact_absent": absent, "other_non_features": other}


def build_report(path: Path) -> dict[str, Any]:
    df, sheets = read_workbook(path)
    d, diag = derive(df)
    sites = site_rows(d)
    by_site = {r["site"]: r for r in sites}
    top5_by_size = [r["site"] for r in sites[:5]]
    chosen = list(EXPECTED_SITES)
    big = [r for r in sites if r["patients"] >= BIG_SITE_MIN_PATIENTS]
    big_low = [r for r in big if r["deaths"] < 10]
    chosen_rows = [by_site[s] for s in chosen if s in by_site]
    tot_p = sum(r["patients"] for r in chosen_rows)
    tot_d = sum(r["deaths"] for r in chosen_rows)
    tot_up = sum(r["usable"] for r in chosen_rows)
    tot_ud = sum(r["usable_deaths"] for r in chosen_rows)
    sub = d[d["site"].isin(chosen)]

    checks: list[dict[str, Any]] = []

    def add(name: str, expected: Any, actual: Any) -> None:
        checks.append({"check": name, "expected": expected, "actual": actual, "ok": expected == actual})

    add("workbook shape (rows, columns)", list(EXPECTED_SHAPE), list(df.shape))
    for s, (p, dd) in EXPECTED_SITES.items():
        r = by_site.get(s, {"patients": None, "deaths": None})
        add(f"site {s} patients", p, r["patients"])
        add(f"site {s} deaths", dd, r["deaths"])
    add("top-5 total patients", EXPECTED_TOTAL[0], tot_p)
    add("top-5 total deaths", EXPECTED_TOTAL[1], tot_d)
    add("the 5 chosen sites are the 5 largest by patients", sorted(chosen), sorted(top5_by_size))
    add(f"sites with >= {BIG_SITE_MIN_PATIENTS} patients", EXPECTED_BIG_SITES, len(big))
    add("... of which fewer than 10 deaths", EXPECTED_BIG_SITES_UNDER_10_DEATHS, len(big_low))
    add("all 8 feature columns present", True, all(c in df.columns for c in FEATURES))
    add("submitter_id all of the form TCGA-<site>-<id>", 0, diag["submitter_id_malformed"])
    add("duplicate submitter_id", 0, diag["submitter_id_duplicates"])
    add("duplicate case_id", 0, diag["case_id_duplicates"])
    uc = unit_check(d)
    add("unit sanity: rows with |time_days/30.4375 - OS_MONTHS| >= 0.5", 0, uc["failed"])

    return {
        "file": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "sha256": sha256_file(path), "sheets": sheets, "shape": list(df.shape),
        "columns": list(df.columns), "sites": sites, "chosen_sites": chosen,
        "top5_totals": {"patients": tot_p, "deaths": tot_d, "usable": tot_up, "usable_deaths": tot_ud,
                        "dropped": tot_p - tot_up, "dropped_deaths": tot_d - tot_ud},
        "big_sites": {"n": len(big), "under_10_deaths": len(big_low),
                      "list": [(r["site"], r["patients"], r["deaths"]) for r in big]},
        "diag": diag, "unit_check_all": uc, "unit_check_top5": unit_check(sub),
        "features": feature_profile(df, chosen, d["site"].tolist()),
        "leakage": leakage_view(list(df.columns)),
        "checks": checks, "all_ok": all(c["ok"] for c in checks),
        "min_deaths_for_per_hospital_metrics": MIN_DEATHS_FOR_PER_HOSPITAL_METRICS,
    }


# ------------------------------------------------------------------ markdown rendering ------
def md_table(head: list[str], rows: list[list[Any]]) -> str:
    def cell(x: Any) -> str:  # a literal "|" (e.g. in "|time_days/30.4375 - OS_MONTHS|") would split the row
        return str(x).replace("|", "\\|")

    out = ["| " + " | ".join(cell(h) for h in head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(cell(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render(rep: dict[str, Any]) -> str:
    L: list[str] = []
    L.append(f"- file: `{rep['file']}`  \n- sha256: `{rep['sha256']}`")
    L.append(f"- sheets: {rep['sheets']}; shape: **{rep['shape'][0]} rows x {rep['shape'][1]} columns**")
    L.append("")
    L.append("**Sites, all rows, before any dropping** (the five used are marked *):")
    exp = EXPECTED_SITES
    rows = []
    for r in rep["sites"][:12]:
        e = exp.get(r["site"])
        rows.append([("*" if e else "") + r["site"], r["patients"], r["deaths"],
                     f"{e[0]} / {e[1]}" if e else "", ("yes" if (r["patients"], r["deaths"]) == e else "NO") if e else ""])
    L.append(md_table(["site", "patients", "deaths", "prompt says (patients / deaths)", "match"], rows))
    t = rep["top5_totals"]
    L.append("")
    L.append(f"Top-5 total: **{t['patients']} patients, {t['deaths']} deaths** "
             f"(prompt: {EXPECTED_TOTAL[0]} / {EXPECTED_TOTAL[1]}).")
    bs = rep["big_sites"]
    L.append(f"Sites with >= {BIG_SITE_MIN_PATIENTS} patients: **{bs['n']}**; of those, **{bs['under_10_deaths']}** have "
             f"fewer than 10 deaths (prompt: {EXPECTED_BIG_SITES} and {EXPECTED_BIG_SITES_UNDER_10_DEATHS}).")
    L.append("")
    L.append("**Usable time** (time = survival x 365.25 if event == 1 else days_to_last_follow_up; "
             "a row is dropped when that value is missing / non-numeric):")
    chosen_rows = [r for r in rep["sites"] if r["site"] in rep["chosen_sites"]]
    rows = [[r["site"], r["patients"], r["deaths"], r["usable"], r["usable_deaths"], r["dropped"],
             r["dropped_deaths"], r["zero_time"], r["zero_time_deaths"]] for r in chosen_rows]
    rows.append(["**top-5**", t["patients"], t["deaths"], t["usable"], t["usable_deaths"], t["dropped"],
                 t["dropped_deaths"], sum(r["zero_time"] for r in chosen_rows),
                 sum(r["zero_time_deaths"] for r in chosen_rows)])
    L.append(md_table(["site", "patients", "deaths", "usable rows", "usable deaths", "dropped rows",
                       "dropped deaths", "usable rows with time = 0", "of which deaths"], rows))
    dg = rep["diag"]
    L.append("")
    L.append(f"Whole workbook: event values {dg['event_values']}; deaths with no `survival` value: "
             f"{dg['deaths_without_survival_value']}; censored rows that do carry a `survival` value: "
             f"{dg['censored_with_survival_value']}; censored rows with no follow-up days: "
             f"{dg['censored_without_followup_days']}.")
    L.append(f"Non-numeric tokens found in the numeric time columns: {dg['non_numeric_tokens']}.")
    L.append("")
    L.append("**Unit sanity check** (`|time_days/30.4375 - OS_MONTHS| < 0.5`, only rows that have OS_MONTHS "
             "and a usable time; OS_MONTHS is never a feature):")
    rows = []
    for name, u in (("whole workbook", rep["unit_check_all"]), ("top-5 sites", rep["unit_check_top5"])):
        rows.append([name, u["checked"], u["deaths_checked"], u["censored_checked"], u["passed"], u["failed"],
                     f"{u['max_abs_diff_months']:.4f}", f"{u['median_abs_diff_months']:.4f}"])
    L.append(md_table(["scope", "rows checked", "of which deaths", "of which censored", "pass", "fail",
                       "max abs diff (months)", "median abs diff (months)"], rows))
    ua = rep["unit_check_all"]
    L.append("")
    L.append(f"Pass rate {100 * ua['pass_rate']:.2f}% ({ua['passed']}/{ua['checked']}); the {ua['failed']} failures are "
             f"{ua['failed_deaths']} deaths + {ua['failed_censored']} censored rows, {ua['failed_in_top5']} of them in "
             f"the top-5 sites, {ua['failed_with_time_zero']} with time_days == 0. Event flag `survival_event` vs "
             f"`OS_STATUS`: {rep['diag']['event_vs_OS_STATUS_disagree']} disagreements in "
             f"{rep['diag']['event_vs_OS_STATUS_rows_compared']} rows compared. All failing rows:")
    L.append("")
    L.append(md_table(["submitter_id", "site", "event", "OS_STATUS event", "time_days", "OS_MONTHS", "abs diff (months)"],
                      [[f["submitter_id"], f["site"], f["event"], f["OS_STATUS_event"], f["time_days"],
                        f["os_months"], f["abs_diff_months"]] for f in ua["failures"]]))
    L.append("")
    L.append("**Feature columns** (numeric):")
    rows = []
    for c, p in rep["features"]["numeric"].items():
        rows.append([c, p["numeric"], p["unusable"], p["unusable_tokens"], p["min"], p["max"],
                     p["distinct_values"] if p["distinct_values"] is not None else f"{p['n_distinct']} distinct"])
    L.append(md_table(["column", "numeric cells", "unusable cells", "unusable tokens", "min", "max", "distinct values"], rows))
    L.append("")
    L.append("**Feature columns** (categorical; raw tokens with counts over all 907 rows; "
             "`[Not Available]`, `[Not Evaluated]` and blank all become category `NA`):")
    rows = [[c, p["tokens"], p["n_na_category"]] for c, p in rep["features"]["categorical"].items()]
    L.append(md_table(["column", "raw tokens", "rows mapped to NA"], rows))
    vs, idim = rep["features"]["vocab_sizes"], rep["features"]["input_dim"]
    L.append("")
    L.append(f"One-hot vocabulary sizes (NA always included): {vs}. Model input width = {idim['categorical_onehot']} "
             f"one-hot + {idim['numeric']} numeric + {idim['missing_indicators']} missing-indicators = **{idim['total']}**.")
    L.append("")
    L.append("**Share of missing / NA cells per feature, per site** (top-5 sites):")
    cs = rep["chosen_sites"]
    rows = [[c] + [pct(v[s]) for s in cs] for c, v in rep["features"]["missing_by_site"].items()]
    L.append(md_table(["feature"] + cs, rows))
    lk = rep["leakage"]
    L.append("")
    L.append("**Leakage deny-list vs the actual columns:**")
    L.append(f"- columns starting OS_/DFS_/PFS_ present in the workbook ({len(lk['prefixed'])}): {lk['prefixed']}")
    L.append(f"- exact deny-list names present ({len(lk['exact_present'])}): {lk['exact_present']}; "
             f"absent: {lk['exact_absent']}")
    L.append(f"- columns that are neither a feature nor on the deny-list ({len(lk['other_non_features'])}): "
             f"{lk['other_non_features']}")
    L.append("")
    L.append("**Expectation checks against master-prompt sections 3-4:**")
    rows = [[c["check"], c["expected"], c["actual"], "PASS" if c["ok"] else "**FAIL**"] for c in rep["checks"]]
    L.append(md_table(["check", "prompt expects", "found", "result"], rows))
    L.append("")
    n_fail = sum(1 for c in rep["checks"] if not c["ok"])
    L.append("RESULT: " + ("all expectations hold." if rep["all_ok"]
                           else f"{n_fail} expectation(s) differ from the prompt - STOP and report."))
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="path to clinical.xlsx")
    ap.add_argument("--json", type=Path, default=None, help="also write the full report as JSON here")
    args = ap.parse_args(argv)
    if not args.data.is_file():
        print(f"MISSING: {args.data} - the workbook must be at fedverify/data/tcga_brca/clinical.xlsx", file=sys.stderr)
        return 2
    try:
        rep = build_report(args.data.resolve())
    except ImportError as e:
        print(f"MISSING PACKAGE: {e}. Install with: pip install --break-system-packages openpyxl", file=sys.stderr)
        return 2
    print(render(rep))
    if args.json is not None:
        args.json.write_text(json.dumps(rep, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
