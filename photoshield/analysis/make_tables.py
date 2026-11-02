"""Every number in a PhotoShield table comes from here, read from photoshield/results/ (CLAUDE.md).

    python -m photoshield.analysis.make_tables [--results DIR] [--out FILE]

Writes Markdown to stdout and to --out (default photoshield/results/TABLES.md). A missing cell
renders as "—" and is listed on stderr. Rows from the random-weight `tiny` encoder are marked
"pipeline check": they test the code, they are not findings.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS = os.path.join(REPO, "photoshield", "results")
DASH = "—"
MISSING: List[str] = []


def _runs(results: str, exp: str):
    """[(cell, seed, config, records)] for every complete run of an experiment."""
    out = []
    for cfg_path in sorted(glob.glob(os.path.join(results, exp, "*", "seed*", "config.json"))):
        d = os.path.dirname(cfg_path)
        rec_path = os.path.join(d, "records.jsonl")
        if not os.path.exists(rec_path):
            print(f"[skip] incomplete run {os.path.relpath(d, results)} (no records.jsonl)", file=sys.stderr)
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        with open(rec_path) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        out.append((os.path.basename(os.path.dirname(d)), os.path.basename(d), cfg, recs))
    return out


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if isinstance(x, (int, float)) and x != float("inf")]
    return statistics.fmean(xs) if xs else None


def _sd(xs) -> Optional[float]:
    xs = [x for x in xs if isinstance(x, (int, float)) and x != float("inf")]
    return statistics.stdev(xs) if len(xs) > 1 else None


def fmt(x, places=2, what="") -> str:
    if x is None:
        if what:
            MISSING.append(what)
        return DASH
    if isinstance(x, float):
        return f"{x:.{places}f}"
    return str(x)


def pm(xs, places=2, what="") -> str:
    m, s = _mean(xs), _sd(xs)
    if m is None:
        return fmt(None, what=what)
    return f"{m:.{places}f} ± {s:.{places}f}" if s is not None else f"{m:.{places}f}"


def _table(head: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _label(cfg) -> str:
    m = cfg.get("model", "?")
    return f"{m} (pipeline check)" if m == "tiny" or not cfg.get("pretrained", True) else m


# ── tables ───────────────────────────────────────────────────────────────────
def table_e1(results) -> str:
    runs = _runs(results, "e1")
    if not runs:
        MISSING.append("E1: no runs (python -m photoshield study e1)")
        return "_No E1 runs._"
    rows = []
    for cell, seed, cfg, recs in sorted(runs, key=lambda r: (r[2].get("model", ""), r[2].get("eps", 0), r[1])):
        g = lambda k: [r.get(k) for r in recs]
        rows.append([_label(cfg), str(cfg.get("eps")), str(cfg.get("steps")), seed, str(len(recs)),
                     pm(g("psnr"), what=f"E1 {cell} psnr"), pm(g("ssim"), 4, f"E1 {cell} ssim"),
                     pm(g("linf"), 1, f"E1 {cell} linf"),
                     pm(g("latent_shift"), 3, f"E1 {cell} latent_shift"),
                     pm(g("latent_shift_noise"), 3, f"E1 {cell} latent_shift_noise"),
                     pm(g("recon_psnr_drop"), 2), pm(g("attack_wall_s"), 2)])
    return _table(["encoder", "ε (/255)", "steps", "seed", "n", "PSNR dB", "SSIM", "L∞ (/255)",
                   "latent shift (cloak)", "latent shift (random ±ε)", "recon PSNR drop dB",
                   "time s"], rows)


def table_e3(results) -> str:
    runs = _runs(results, "e3")
    if not runs:
        MISSING.append("E3: no runs (python -m photoshield study e3)")
        return "_No E3 runs._"
    parts = []
    for cell, seed, cfg, recs in runs:
        procs = ["none"] + [p for p in cfg.get("processing", [])]
        rows = []
        for p in procs:
            rs = [r for r in recs if r["processing"] == p]
            if not rs:
                rows.append([p, fmt(None, what=f"E3 {cell} {p}"), DASH, DASH, DASH])
                continue
            cl = _mean([r["latent_shift"] for r in rs])
            nz = _mean([r["latent_shift_noise"] for r in rs])
            ratio = (cl / nz) if (cl is not None and nz) else None
            rows.append([p, str(len(rs)), fmt(cl, 3), fmt(nz, 3), fmt(ratio, 2)])
        parts.append(f"**{_label(cfg)}, ε={cfg.get('eps')}/255, steps={cfg.get('steps')}, {seed}**\n\n"
                     + _table(["processing", "n", "latent shift (cloak)", "latent shift (random ±ε)",
                               "cloak / random"], rows))
    return "\n\n".join(parts)


def table_e6(results) -> str:
    out = []
    paths = sorted(glob.glob(os.path.join(results, "e6", "*", "seed*", "summary.json")))
    if not paths:
        MISSING.append("E6: no calibration run (python -m photoshield study e6)")
        out.append("_No E6 calibration run._")
    for p in paths:
        with open(p) as f:
            s = json.load(f)["summary"]
        out.append(_table(["dataset", "positives", "negatives", "τ_near", "FPR @ τ_near", "TPR @ τ_near",
                           "τ_far", "FPR @ τ_far", "TPR @ τ_far"],
                          [[s["dataset"], str(s["n_pos"]), str(s["n_neg"]), str(s["tau_near"]),
                            fmt(s["fpr_near"], 4), fmt(s["tpr_near"], 4), str(s["tau_far"]),
                            fmt(s["fpr_far"], 4), fmt(s["tpr_far"], 4)]]))
        # recall per transform at tau_near, from the records
        d = os.path.dirname(p)
        with open(os.path.join(d, "records.jsonl")) as f:
            recs = [json.loads(line) for line in f]
        pos = [r for r in recs if r["kind"] == "positive"]
        rows = []
        for t in dict.fromkeys(r["transform"] for r in pos):
            ds = [r["distance"] for r in pos if r["transform"] == t]
            rows.append([t, str(len(ds)), fmt(_mean(ds), 2), str(max(ds)),
                         fmt(sum(x <= s["tau_near"] for x in ds) / len(ds), 3)])
        neg = [r["distance"] for r in recs if r["kind"] == "negative"]
        rows.append(["unrelated pairs", str(len(neg)), fmt(_mean(neg), 2), str(min(neg)) + " (min)",
                     fmt(sum(x <= s["tau_near"] for x in neg) / len(neg), 4)])
        out.append(_table(["transform", "n", "mean distance", "max distance", "share ≤ τ_near"], rows))
    return "\n\n".join(out)


def table_e6p(results) -> str:
    runs = _runs(results, "e6p")
    if not runs:
        MISSING.append("E6p: no runs (python -m photoshield study e6p)")
        return "_No E6p runs._"
    parts = []
    for cell, seed, cfg, recs in runs:
        rows = []
        for p in dict.fromkeys(r["processing"] for r in recs):
            rs = [r for r in recs if r["processing"] == p]
            rows.append([p, str(len(rs)), pm([r["d_to_original"] for r in rs], 2),
                         str(max(r["d_to_original"] for r in rs)), pm([r["d_to_cloaked"] for r in rs], 2)])
        parts.append(f"**{_label(cfg)}, ε={cfg.get('eps')}/255, {seed}**\n\n"
                     + _table(["processing of the cloaked photo", "n", "pHash distance to original",
                               "max", "pHash distance to cloaked"], rows))
    return "\n\n".join(parts)


def table_e6b(results) -> str:
    runs = _runs(results, "e6b")
    if not runs:
        MISSING.append("E6b: no runs (python -m photoshield study e6-browser; needs node + playwright)")
        return "_No E6b runs._"
    parts = []
    for cell, seed, cfg, recs in runs:
        rows = []
        for q in cfg.get("qualities", []):
            ds = [r["distance"] for r in recs if r["image"].endswith(f"_q{q}")]
            rows.append([str(q), str(len(ds)), fmt(sum(d == 0 for d in ds) / len(ds), 3) if ds else DASH,
                         str(max(ds)) if ds else DASH])
        parts.append(f"Browser: {cfg.get('browser')}\n\n"
                     + _table(["JPEG quality", "n", "share bit-identical", "max distance"], rows))
    return "\n\n".join(parts)


def table_e7(results) -> str:
    runs = _runs(results, "e7")
    if not runs:
        MISSING.append("E7: no runs (python -m photoshield study e7)")
        return "_No E7 runs._"
    _, _, cfg, recs = runs[-1]
    single = [r for r in recs if r["kind"] == "single"]
    rows = []
    for b in dict.fromkeys(r["backend"] for r in single):
        rs = [r for r in single if r["backend"] == b]
        rows.append([b, str(len(rs)), str(max(r["note_bytes"] for r in rs)), pm([r["anchor_ms"] for r in rs], 2)])
    fee = next((r for r in recs if r.get("name") == "algorand_min_fee_algo"), None)
    t1 = _table(["anchor backend", "n", "note bytes (max)", "anchor ms"], rows)
    album = [r for r in recs if r["kind"] == "album"]
    rows = [[str(r["n"]), str(r["note_bytes"]), str(r["single_notes_bytes_total"]),
             fmt(r["single_notes_bytes_total"] / r["note_bytes"], 1),
             str(r["proof_hashes_max"]), str(r["proof_bytes_max"]), fmt(r["merkle_build_ms"], 2),
             fmt(r["n"] * fee["value"], 3) if fee else DASH, fmt(fee["value"], 3) if fee else DASH]
            for r in album]
    t2 = _table(["photos", "album note bytes", "N single notes bytes", "bytes saved ×",
                 "proof hashes (max)", "proof bytes (max)", "Merkle build ms",
                 "fee ALGO, N singles", "fee ALGO, 1 album"], rows)
    reb = [r for r in recs if r["kind"] == "rebuild"]
    t3 = _table(["anchored photos", "indexer pages", "accepted", "rebuild ms (simulated indexer)"],
                [[str(r["n"]), str(r["pages"]), str(r["accepted"]), fmt(r["rebuild_ms"], 1)] for r in reb])
    src = f"\n\nFee constant: {fee['source']}." if fee else ""
    return t1 + "\n\n" + t2 + "\n\n" + t3 + src


def bootstrap_ci(xs, n: int = 2000, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap CI of the mean. Fixed seed, so the table is reproducible."""
    import random
    xs = [x for x in xs if isinstance(x, (int, float))]
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    lo, hi = means[int((1 - level) / 2 * n)], means[int((1 + level) / 2 * n) - 1]
    return statistics.fmean(xs), lo, hi


def _ci_cells(xs, places=3):
    r = bootstrap_ci(xs)
    if r is None:
        return [DASH, DASH, "insufficient data"]
    m, lo, hi = r
    verdict = "degrades" if lo > 0 else ("improves" if hi < 0 else "no effect shown (CI includes 0)")
    return [f"{m:.{places}f}", f"[{lo:.{places}f}, {hi:.{places}f}]", verdict]


def table_e4(results) -> str:
    runs = _runs(results, "e4")
    if not runs:
        MISSING.append("E4: no runs (photoshield/experiments/colab_edit_test.ipynb)")
        return "_No E4 runs. Run the Colab notebook and copy its results/e4 folder here._"
    parts = []
    for cell, seed, cfg, recs in runs:
        ed = cfg.get("editor", {})
        head = (f"**{cfg.get('encoder')} cloak, ε={cfg.get('eps')}/255, steps={cfg.get('steps')}, "
                f"editor {ed.get('pipeline', '?')} ({ed.get('repo', '?')}), {len(recs)} edits, {seed}**")
        rows = [
            ["LPIPS(edit cloaked, edit original) − LPIPS(edit noise, edit original)"]
            + _ci_cells([r["lpips_cloak"] - r["lpips_noise"] for r in recs]),
            ["CLIP-dir(original edit) − CLIP-dir(cloaked edit)"]
            + _ci_cells([r["clip_dir_original"] - r["clip_dir_cloak"] for r in recs]),
            ["CLIP-dir(noise edit) − CLIP-dir(cloaked edit)"]
            + _ci_cells([r["clip_dir_noise"] - r["clip_dir_cloak"] for r in recs]),
        ]
        means = _table(["", "LPIPS cloak", "LPIPS noise", "CLIP-dir original", "CLIP-dir cloak",
                        "CLIP-dir noise"],
                       [["mean ± sd", pm([r["lpips_cloak"] for r in recs], 3),
                         pm([r["lpips_noise"] for r in recs], 3),
                         pm([r["clip_dir_original"] for r in recs], 3),
                         pm([r["clip_dir_cloak"] for r in recs], 3),
                         pm([r["clip_dir_noise"] for r in recs], 3)]])
        parts.append(head + "\n\n" + means + "\n\n"
                     + _table(["paired difference (positive = cloak hurts the edit)", "mean",
                               "bootstrap 95% CI", "verdict"], rows))
    return "\n\n".join(parts)


def table_e5(results) -> str:
    runs = _runs(results, "e5")
    if not runs:
        MISSING.append("E5: no runs (photoshield/experiments/colab_edit_test.ipynb)")
        return "_No E5 runs. Run the Colab notebook and copy its results/e5 folder here._"
    parts = []
    for cell, seed, cfg, recs in runs:
        rows = []
        for t in dict.fromkeys(r["target"] for r in recs):
            rs = [r for r in recs if r["target"] == t]
            diff = [r["latent_shift"] - r["latent_shift_noise"] for r in rs]
            rows.append([t + (" (attacked)" if rs[0]["is_source"] else ""), str(len(rs)),
                         pm([r["latent_shift"] for r in rs], 3), pm([r["latent_shift_noise"] for r in rs], 3)]
                        + _ci_cells(diff))
        parts.append(f"**cloak crafted on {cfg.get('source')}, ε={cfg.get('eps')}/255, {seed}**\n\n"
                     + _table(["encoder", "n", "latent shift (cloak)", "latent shift (random ±ε)",
                               "cloak − random", "bootstrap 95% CI", "verdict"], rows))
    return "\n\n".join(parts)


def _e8_status(t, thr):
    d, gt, cov, textured, level, slevel = t
    if d is None or not textured:
        return None
    moved = abs(slevel - level) >= thr["delta_level"]
    if d >= thr["tau_alter"] or moved:
        return "altered"
    if d <= thr["tau_match"]:
        return "match"
    return "uncertain"


def table_e8(results) -> str:
    runs = _runs(results, "e8")
    if not runs:
        MISSING.append("E8: no runs (python -m photoshield study e8)")
        return "_No E8 runs._"
    parts = []
    for cell, seed, cfg, recs in runs:
        thr = cfg["thresholds"]
        head = (f"**{cfg.get('n_images')} photos, {seed}. Thresholds from this run: τ_match = "
                f"{thr['tau_match']}, τ_alter = {thr['tau_alter']}, Δlevel = {thr['delta_level']}, "
                f"k_min = {thr['k_min']}.** Edited tile = at least half inside the edit mask; unchanged "
                f"tile = no overlap. Pasted rectangles simulate a face swap, smooth fills simulate inpainting. "
                f"For unrelated photos every tile counts as unchanged, so \"matched\" is the false-match rate.")
        rows, ablation = [], []
        for kind in ("clean", "paste", "smooth", "unrelated"):
            for tr in dict.fromkeys(r["transform"] for r in recs if r["kind"] == kind):
                rs = [r for r in recs if r["kind"] == kind and r["transform"] == tr]
                ok = [r for r in rs if r["tiles"]]
                ed = [t for r in ok for t in r["tiles"] if t[1] >= 0.5 and _e8_status(t, thr)]
                un = [t for r in ok for t in r["tiles"] if t[1] == 0.0 and _e8_status(t, thr)]
                der = sum(1 for r in ok if sum(_e8_status(t, thr) == "match" for t in r["tiles"]) >= thr["k_min"])
                rec = (sum(_e8_status(t, thr) == "altered" for t in ed) / len(ed)) if ed else None
                kept = (sum(_e8_status(t, thr) == "match" for t in un) / len(un)) if un else None
                falt = (sum(_e8_status(t, thr) == "altered" for t in un) / len(un)) if un else None
                rows.append([kind, tr, str(len(rs)), fmt(len(ok) / len(rs), 3), f"{der}/{len(ok)}",
                             fmt(rec, 3) if ed else DASH, fmt(kept, 3),
                             fmt(falt, 3) if kind != "unrelated" else DASH])
                if kind in ("paste", "smooth") and ed:
                    ph = sum(1 for t in ed if t[0] >= thr["tau_alter"]) / len(ed)
                    ablation.append([kind, tr, fmt(ph, 3), fmt(rec, 3)])
        parts.append(head + "\n\n" + _table(
            ["suspects", "sharing", "n", "aligned", "verdict DERIVED", "edited tiles flagged altered",
             "unchanged tiles matched", "unchanged tiles wrongly altered"], rows))
        parts.append("Ablation: share of edited tiles flagged by the pHash distance alone vs with the "
                     "detail-level rule.\n\n" + _table(["edit", "sharing", "pHash only", "pHash + detail level"], ablation))
    return "\n\n".join(parts)


SECTIONS = [
    ("E1/E2 — invisibility vs strength, encoder disruption", table_e1),
    ("E3 — does the cloak survive sharing?", table_e3),
    ("E4 — does a real editor (InstructPix2Pix) degrade?", table_e4),
    ("E5 — does an sdvae cloak transfer to other encoders?", table_e5),
    ("E6 — pHash calibration (installs photoshield/calibration.json)", table_e6),
    ("E6p — does a processed cloaked photo still match its anchor?", table_e6p),
    ("E6b — browser vs PIL JPEG decode", table_e6b),
    ("E7 — provenance cost", table_e7),
    ("E8 — origin reports: localising local edits (v2)", table_e8),
]


def render(results: str) -> str:
    del MISSING[:]
    parts = ["# PhotoShield results", "",
             "Generated by `python -m photoshield.analysis.make_tables` from `photoshield/results/`. "
             "Do not edit by hand. Rows marked *pipeline check* use a random-weight encoder and are not findings. "
             "E4/E5 (real editors, transfer) run in `photoshield/experiments/colab_edit_test.ipynb`.", ""]
    for title, fn in SECTIONS:
        parts += [f"## {title}", "", fn(results), ""]
    return "\n".join(parts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=DEFAULT_RESULTS)
    p.add_argument("--out", default=None)
    ns = p.parse_args(argv)
    md = render(ns.results)
    out = ns.out or os.path.join(ns.results, "TABLES.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(md)
    print(md)
    for m in MISSING:
        print(f"[missing] {m}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
