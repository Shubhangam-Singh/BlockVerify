"""Run one method over repeated stratified cross-validation (prompt §4, Revision 2 §2).

    python3 -m fedverify.survival.runner --family lr --method fedavg_lr --seed 0 --repeats 5                 # TCGA
    python3 -m fedverify.survival.runner --family lr --method dp_fedavg_lr --epsilon 4 --seed 0 --repeats 5
    python3 -m fedverify.survival.runner --dataset heart --family xgb --method fedxgb_bagging --seed 0 --repeats 5
    python3 -m fedverify.survival.runner --family xgb --method fedxgb_bagging --task survival --seed 0  # TCGA AFT
    python3 -m fedverify.survival.runner --dataset heart --family lr --method ours_lr --attack sign_flip --seed 0
    python3 -m fedverify.survival.runner --dataset heart --family lr --method ours_lr --attack sign_flip --no-screening

Protocol: for every repeat r (seed = base + r, D7) the patients are split into stratified folds on (hospital, label).
For fold k the test set is every hospital's fold-k patients and the method trains on the rest, each hospital seeing
only its own rows. The final-round out-of-fold predictions of all folds are pooled and the metrics of metrics.yaml are
computed ONCE per repeat on all evaluated patients; tables report mean ± std over the repeats.

Results in <out-dir>/<exp>/<cell>/seed<base>/ (exp = survival for TCGA, heart for heart; cell = method, plus
_eps<epsilon> for dp_fedavg_lr and ours_lr, _aft for the TCGA survival task, _full for heart full mode, _<backend>
for a mined chain, _<attack> and _noscreen for the Phase-4 ours_* runs):
    config.json                     every parameter, data provenance, feature names, the D10 hospital-id mapping
    repeat<r>/fold<k>/config.json   the fold-run's own record (a self-contained FedVerify run)
    repeat<r>/fold<k>/rounds.jsonl  one line per round: fold-test metrics, the existing diag / privacy / timing keys,
                                    and per-hospital train/test accuracy and log-loss (hospital_metrics, Figs 5-6)
    repeat<r>/oof.jsonl             final-round out-of-fold score, threshold and prediction for every patient
    summary.json                    per repeat: pooled and per-hospital metrics, pooled metrics after every round;
                                    then mean ± std over the repeats

Same seed and same --torch-threads give byte-identical rounds.jsonl / oof.jsonl / summary.json once the wall-clock
keys (train_wall_s, round_wall_s) are dropped (CLAUDE.md A1, A5).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..analysis.make_tables import mean_std
from ..core.runner import seed_everything
from . import datasets, fl_lr, fl_rf, fl_xgb
from . import poison, probe
from .chain_round import ChainSession
from .common import LOG_TIME, RISK
from .config import DATASETS, FAMILY_ATTACKS, OURS_METHODS, SurvivalConfig, derive_seed, is_infinite
from .data import fold_data, make_folds, repeat_seeds
from .metrics import compute, hospital_round_metrics, load_metric_config, per_hospital, thresholds_for_test
from .screening import cfg_tau_key, load_tau

F1_RULE = ("threshold maximising F1 on the training fold's predictions (each hospital's own threshold for "
           "local_* methods), applied to that fold's test patients")
ACCURACY_RULE = ("accuracy uses the same threshold as F1 (Phase 7): the threshold maximising F1 on the training "
                 "fold's predictions, for the pooled, per-hospital and per-round train/test accuracy alike")
LOCAL_SCORING = "local_* methods score every patient with their own hospital's model"
TRAINERS = {"lr": fl_lr.run_fold, "rf": fl_rf.run_fold, "xgb": fl_xgb.run_fold}
CHAIN_METRIC_DEFINITIONS = {
    "latency_ms": "mean wall-clock time between consecutive blocks, in ms (as the reference paper defines it)",
    "throughput_bytes_per_s": "total block bytes / total elapsed seconds",
    "send_rate_tx_per_min": "transactions per minute",
    "mean_block_creation_s": "mean time to build and mine one block (proof-of-work), in s"}


def _plain(x: Any) -> Any:
    """numpy -> python for json; NaN/inf -> None (not valid JSON for the dashboard)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _dump(obj: Any) -> str:
    return json.dumps(_plain(obj), ensure_ascii=False)


def with_tau(cfg: SurvivalConfig) -> SurvivalConfig:
    """A screening ours_* run reads tau from the calibration file (never from a constant) and records where from.
    A tau already set (only the calibration's score-only inf) is kept."""
    if not cfg.screened or cfg.tau is not None:
        return cfg
    tau, source = load_tau(cfg.taus_path, cfg_tau_key(cfg), cfg.tau_policy)
    return dataclasses.replace(cfg, tau=tau, tau_source=source)


def run(cfg: SurvivalConfig, verbose: bool = True, plan_cache: Optional[dict] = None) -> dict:
    """Train and evaluate cfg.method over cfg.repeats x cfg.folds fold-runs. Returns the summary dict. plan_cache lets
    several runs share the DP sigma solved per hospital size (calibration runs many short cells)."""
    cfg = with_tau(cfg)
    if cfg.torch_threads:
        torch.set_num_threads(int(cfg.torch_threads))
    t_start = time.perf_counter()
    mcfg = load_metric_config(cfg.metric_key)
    names, ev_name = mcfg["metrics"], mcfg["event_name"]
    kind = RISK if cfg.task == "binary" else LOG_TIME
    loaded = datasets.load(cfg, verbose=verbose)
    cohort, X = loaded.cohort, loaded.X
    train_fold = TRAINERS[cfg.family]
    local = cfg.method.startswith("local_")
    unmodelled = set(cfg.local_exclude) if local else set()
    seeds = repeat_seeds(cfg.seed, cfg.repeats)
    fold_kwargs = {"plan_cache": {} if plan_cache is None else plan_cache} if cfg.family == "lr" else {}   # dp.py
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    session = ChainSession(cfg, run_seed=cfg.seed) if cfg.commit else None    # Phase 3: one ledger per run

    hospital_ids = {h: cfg.hospital_index[h] for h in cfg.hospitals}
    base_record = {
        "config": cfg.to_dict(), "model_family": cfg.family, "aggregation": cfg.aggregation,
        "hospital_ids": hospital_ids, "repeat_seeds": seeds, "data": loaded.info,
        "features": loaded.feature_names, "num_features": len(loaded.feature_names),
        "num_params": (len(loaded.feature_names) + 1) if cfg.family == "lr" else None,
        "vocab_sha256": hashlib.sha256(loaded.vocab_path.read_bytes()).hexdigest(),
        "metrics": mcfg, "score_kind": kind, "f1_threshold_rule": F1_RULE, "accuracy_rule": ACCURACY_RULE,
        "local_scoring": LOCAL_SCORING, "torch_threads_effective": int(torch.get_num_threads()),
        "attacker_ids": [],                 # same key as existing runs; Phase 4 fills it
    }
    if session is not None:
        base_record["chain"] = session.describe()
    ours = cfg.method in OURS_METHODS
    attackers_by_repeat = [poison.attackers_for(cfg, sd) for sd in seeds]
    if ours:                                            # Phase 4: the attack and the screen, fully specified
        base_record["attacker_ids"] = [h for h in cfg.hospitals if any(h in a for a in attackers_by_repeat)]
        base_record["attacker_ids_by_repeat"] = attackers_by_repeat
        base_record["attack"] = {"name": cfg.attack, "attackers_per_repeat": poison.N_ATTACKERS,
                                 "chosen_by": ("pinned (calibration)" if cfg.attacker else
                                               "attacks.byzantine.attacker_ids on the repeat seed, attacker_frac=1/K"),
                                 "active_from_round": 1, "adaptive": False, "family_attacks": FAMILY_ATTACKS[cfg.family]}
        base_record["screening"] = {"enabled": cfg.screening, "aggregator": cfg.aggregation, "tau": cfg.tau,
                                    "tau_source": cfg.tau_source, "tau_policy": cfg.tau_policy,
                                    "tau_key": cfg_tau_key(cfg)}
        if cfg.family in ("rf", "xgb"):
            base_record["screening"]["probe"] = probe.describe(cfg, loaded.feature_names)
    (run_dir / "config.json").write_text(json.dumps(_plain(base_record), indent=2) + "\n", encoding="utf-8")

    repeats = []
    for r, seed in enumerate(seeds):
        folds = make_folds(cohort, seed, cfg.folds)
        R, N = cfg.rounds, cohort.n
        risk = np.full((R, N), np.nan)
        pred = np.full((R, N), -1, dtype=np.int64)
        thr_final = np.full(N, np.nan)
        fold_of = np.full(N, -1, dtype=np.int64)
        realized: list[float] = []                     # final-round realised epsilon of every hospital and fold
        attackers = attackers_by_repeat[r]
        reports: list[dict] = []                       # ours_*: the per-round attack block of every fold
        for k, (tr, te) in enumerate(folds):
            fd = fold_data(cohort, X, tr, te)
            fold_of[te] = k
            seed_everything(derive_seed(seed, "fold", k))
            fdir = run_dir / f"repeat{r}" / f"fold{k}"
            fdir.mkdir(parents=True, exist_ok=True)
            fold_cfg = {**cfg.to_dict(), "dataset": loaded.info["dataset"], "num_clients": len(cfg.hospitals),
                        "aggregator": cfg.aggregation, "repeat": r, "fold": k, "repeat_seed": seed,
                        "run_id": f"{cfg.exp}_{cfg.cell}_s{cfg.seed}_r{r}_f{k}"}
            (fdir / "config.json").write_text(json.dumps(_plain({
                "config": fold_cfg, "model_family": cfg.family, "hospital_ids": hospital_ids,
                "attacker_ids": attackers, "num_params": base_record["num_params"],
                "train_patients": int(len(tr)), "test_patients": int(len(te)),
                f"test_{ev_name}": int(fd.event_test.sum()),
                "torch_threads_effective": int(torch.get_num_threads())}), indent=2) + "\n", encoding="utf-8")

            kwargs = dict(fold_kwargs)
            if session is not None:
                kwargs["committer"] = session.for_fold(fold_cfg["run_id"])
            if ours:
                kwargs.update(attackers=attackers, feature_names=loaded.feature_names)
            rounds_iter = train_fold(cfg, fd, seed, k, **kwargs)
            commit_fh = open(fdir / "commitments.jsonl", "w", encoding="utf-8") if session is not None else None
            with open(fdir / "rounds.jsonl", "w", encoding="utf-8") as fout:
                while True:
                    t_round = time.perf_counter()
                    try:
                        out = next(rounds_iter)
                    except StopIteration:
                        break
                    if kind == RISK:
                        thr = thresholds_for_test(fd.event_train, out.train_risk, fd.hosp_train, fd.hosp_test, local)
                        thr_tr = thresholds_for_test(fd.event_train, out.train_risk, fd.hosp_train, fd.hosp_train, local)
                        with np.errstate(invalid="ignore"):
                            p = (out.test_risk >= thr).astype(np.int64)
                    else:                                       # a survival time has no decision threshold
                        thr = np.full(len(te), np.nan)
                        thr_tr = None
                        p = np.full(len(te), -1, dtype=np.int64)
                    m = compute(names, fd.time_test, fd.event_test, out.test_risk, p, kind)
                    rec = {"round": out.round, "model_family": cfg.family, "method": cfg.method,
                           "repeat": r, "fold": k}
                    rec.update({f"test_{n}": v for n, v in m.items()})
                    rec.update({"test_n": int(len(te)), f"test_{ev_name}": int(fd.event_test.sum()),
                                "mean_train_loss": out.mean_train_loss,
                                "per_client_num_samples": out.per_client_num_samples,
                                "diag": out.diag, "privacy": out.privacy,
                                "hospital_metrics": hospital_round_metrics(
                                    cfg.hospitals, fd.hosp_train, fd.event_train, out.train_risk,
                                    fd.hosp_test, fd.event_test, out.test_risk, kind,
                                    train_thr=thr_tr, test_thr=thr if kind == RISK else None)})
                    if ours and (cfg.attack or cfg.screened):    # as core/runner.py: attacked runs and detectors only
                        rec["attack"] = poison.attack_report(
                            cfg, attackers, list(out.diag.get("accepted", [])) + list(out.diag.get("rejected", [])),
                            out.diag.get("rejected", []))
                        reports.append(rec["attack"])
                    rec.update(out.extra)
                    rec["train_wall_s"] = out.train_wall_s
                    rec["round_wall_s"] = time.perf_counter() - t_round
                    fout.write(_dump(rec) + "\n")
                    fout.flush()
                    if commit_fh is not None and out.commitment is not None:
                        commit_fh.write(_dump(out.commitment) + "\n")
                        commit_fh.flush()
                    risk[out.round - 1, te] = out.test_risk
                    pred[out.round - 1, te] = p
                    if out.round == R:
                        thr_final[te] = thr
                        if out.privacy:
                            realized += [rep["realized_eps"] for rep in out.privacy.values()
                                         if rep.get("realized_eps") is not None]
            if commit_fh is not None:
                commit_fh.close()

        scored = np.isfinite(risk[-1])
        stray = set(cohort.hospital[~scored].tolist()) - unmodelled
        if (fold_of < 0).any() or stray:
            raise RuntimeError(f"patients of {sorted(stray)} received no out-of-fold prediction")
        pooled = compute(names, cohort.time_days, cohort.event, risk[-1], pred[-1], kind)
        by_hosp = per_hospital(names, cohort.hospital, cohort.time_days, cohort.event, risk[-1], pred[-1],
                               mcfg["min_class_count"], mcfg["rank_metrics"], kind, ev_name)
        by_round = {n: [] for n in names}
        for i in range(R):
            vals = compute(names, cohort.time_days, cohort.event, risk[i], pred[i], kind)
            for n in names:
                by_round[n].append(vals[n])
        score_key = "risk" if kind == RISK else "pred_log_time"
        with open(run_dir / f"repeat{r}" / "oof.jsonl", "w", encoding="utf-8") as f:
            for i in range(N):
                f.write(_dump({"patient": cohort.patient_id[i], "hospital": cohort.hospital[i],
                               "fold": fold_of[i], "time_days": cohort.time_days[i], "event": cohort.event[i],
                               score_key: risk[-1, i], "threshold": thr_final[i], "pred": pred[-1, i]}) + "\n")
        repeats.append({"repeat": r, "seed": seed, "pooled": pooled, "per_hospital": by_hosp,
                        "pooled_by_round": by_round, "max_realized_eps": max(realized) if realized else None,
                        "fold_test_patients": [int(len(te)) for _, te in folds],
                        f"fold_test_{ev_name}": [int(cohort.event[te].sum()) for _, te in folds]})
        if reports:
            repeats[-1]["attackers"] = attackers
            repeats[-1]["detection"] = poison.pooled_detection(reports)
        if verbose:
            shown = "  ".join(f"{n}={pooled[n]:.4f}" if pooled[n] is not None else f"{n}=—" for n in names)
            print(f"[{cfg.cell}] repeat {r} (seed {seed}): {shown}", flush=True)

    eps_seen = [rp["max_realized_eps"] for rp in repeats if rp["max_realized_eps"] is not None]
    summary = {"method": cfg.method, "cell": cfg.cell, "model_family": cfg.family, "dataset": cfg.dataset,
               "task": cfg.task, "score_kind": kind,
               "epsilon": None if cfg.epsilon is None else ("inf" if is_infinite(cfg.epsilon) else cfg.epsilon),
               "max_realized_eps": max(eps_seen) if eps_seen else None,
               "base_seed": cfg.seed, "repeat_seeds": seeds,
               "patients": int(scored.sum()), ev_name: int(cohort.event[scored].sum()),
               "excluded_hospitals": sorted(unmodelled & set(cohort.hospital.tolist())),
               "metrics": names, "repeats": repeats,
               "mean_std": _mean_std_block([rp["pooled"] for rp in repeats], names),
               "per_hospital_mean_std": {h: _mean_std_block([rp["per_hospital"][h] for rp in repeats], names)
                                         for h in repeats[0]["per_hospital"]}}
    if any("detection" in rp for rp in repeats):        # Phase 4: screening quality over all hospital-rounds
        dets = [rp["detection"] for rp in repeats]
        rates = ("precision", "recall", "false_exclusion_rate")
        summary["attack"] = {"name": cfg.attack, "screening": cfg.screening, "tau": cfg.tau,
                             "tau_source": cfg.tau_source,
                             "attackers_by_repeat": [rp.get("attackers", []) for rp in repeats],
                             "detection_rule": ("pooled over hospital-rounds of each repeat: precision = TP/(TP+FP), "
                                                "recall = TP/(TP+FN), false-exclusion rate = FP / honest "
                                                "hospital-rounds; then mean ± std over repeats"),
                             "detection": _mean_std_block(dets, list(rates)),
                             "detection_pooled_over_repeats": poison.combine_detection(dets)}
    (run_dir / "summary.json").write_text(json.dumps(_plain(summary), indent=2) + "\n", encoding="utf-8")
    if session is not None and session.block_log:               # Figs 7-8: the blocks of this run's ledger
        with open(run_dir / "chain_blocks.jsonl", "w", encoding="utf-8") as f:
            for b in session.block_log:
                f.write(_dump(b) + "\n")
        (run_dir / "chain_metrics.json").write_text(json.dumps(_plain({**session.chain_metrics(),
            "definitions": CHAIN_METRIC_DEFINITIONS}), indent=2) + "\n", encoding="utf-8")
    else:                                   # a re-run that mined nothing must not leave an older run's chain files
        for stale in ("chain_blocks.jsonl", "chain_metrics.json"):
            (run_dir / stale).unlink(missing_ok=True)
    if verbose:
        ms = summary["mean_std"]
        shown = "  ".join(f"{n}={ms[n]['mean']:.4f}±{ms[n]['std']:.4f}" if ms[n]["mean"] is not None else f"{n}=—"
                          for n in names)
        print(f"[{cfg.cell}] {len(seeds)} repeat(s): {shown}  ({time.perf_counter() - t_start:.1f}s) "
              f"-> {run_dir}", flush=True)
    return summary


def _mean_std_block(rows: list[dict], names: list[str]) -> dict:
    """mean ± sample std over repeats; a metric missing (None) in any repeat is reported as missing."""
    out = {}
    for n in names:
        vals = [row.get(n) for row in rows]
        if any(v is None for v in vals):
            out[n] = {"mean": None, "std": None, "n_repeats": 0}
            continue
        m, s = mean_std([float(v) for v in vals])
        out[n] = {"mean": m, "std": s, "n_repeats": len(vals)}
    return out


def build_config(argv: Optional[list[str]] = None) -> tuple[SurvivalConfig, argparse.Namespace]:
    ap = argparse.ArgumentParser(prog="fedverify.survival.runner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="tcga", choices=list(DATASETS))
    ap.add_argument("--family", required=True, choices=["lr", "rf", "xgb"])
    ap.add_argument("--method", required=True,
                    help="e.g. fedavg_lr / fedavg, fed_rf / fed, local_lr, central_rf, fedxgb_bagging, local_xgb")
    ap.add_argument("--task", default="binary", choices=["binary", "survival"],
                    help="survival = XGBoost AFT on TCGA (C-index on the predicted survival time)")
    ap.add_argument("--heart-mode", default="paper", choices=["paper", "full"],
                    help="paper (default, 740 rows) or full (920 rows, imputed; sensitivity table only)")
    ap.add_argument("--seed", type=int, default=0, help="base seed; repeat r uses seed + r")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=30, help="default 30 (prompt §4); lower only for smoke tests")
    ap.add_argument("--epsilon", type=float, default=None,
                    help="DP target for dp_fedavg_lr (default 4; 'inf' = no DP). RF / XGBoost DP is n/a and refused")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="pin torch intra-op threads; results are byte-reproducible only at a FIXED count")
    ap.add_argument("--out-dir", default="fedverify/results")
    ap.add_argument("--data", default=None, help="data file (default: the dataset's file under fedverify/data/)")
    ap.add_argument("--include-race", action="store_true",
                    help="TCGA only: add race as a feature (off by default, see TODO)")
    ap.add_argument("--chain-backend", default="mock", choices=["mock", "local"],
                    help="chain_* methods: mock (default) or local (mines the existing PoW chain; Figs 7-8)")
    ap.add_argument("--ipfs", default="local", choices=["local", "real"],
                    help="real also uploads each update to an IPFS daemon at 127.0.0.1:5001 (never required)")
    ap.add_argument("--consensus", default="pow", choices=["pow"], help="the only option (FPoW is undefined)")
    ap.add_argument("--profile", action="store_true",
                    help="cProfile the run; top 10 by cumulative time -> <out-dir>/chain/profile.txt (Fig 8)")
    ap.add_argument("--attack", default=None,
                    help="ours_* only, one malicious hospital: logistic sign_flip | scaling | label_flip; "
                         "RF / XGBoost label_flip_trees | random_trees")
    ap.add_argument("--no-screening", action="store_true",
                    help="ours_* only: the same pipeline without the screen (the undefended row)")
    ap.add_argument("--taus", default=None,
                    help="calibration file to read tau from (default fedverify/results/calibration/taus_survival.json)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        cfg = SurvivalConfig(family=a.family, method=a.method, dataset=a.dataset, task=a.task,
                             heart_mode=a.heart_mode, seed=a.seed, repeats=a.repeats, folds=a.folds,
                             rounds=a.rounds, epsilon=a.epsilon, torch_threads=a.torch_threads, out_dir=a.out_dir,
                             data_path=a.data, include_race=a.include_race, chain_backend=a.chain_backend,
                             ipfs=a.ipfs, consensus=a.consensus, attack=a.attack, screening=not a.no_screening,
                             **({"taus_path": a.taus} if a.taus else {}))
    except ValueError as e:
        ap.error(str(e))
    return cfg, a


def profile_run(cfg: SurvivalConfig, verbose: bool = True, top: int = 10) -> Path:
    """Run under cProfile and write the top functions by cumulative time (tsub = own time, ttot = with callees).

    One throwaway warm-up round (1 repeat, 1 round, into a temporary directory) runs first, so one-time costs such
    as imports and torch's lazy operator registration do not swamp the profile of the run itself. Two views are
    written: every function, and FedVerify / BlockVerify functions only (the pipeline stages behind Fig 8).
    """
    import cProfile
    import dataclasses
    import io
    import pstats
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        run(dataclasses.replace(cfg, repeats=1, rounds=1, out_dir=tmp), verbose=False)
    prof = cProfile.Profile()
    prof.enable()
    run(cfg, verbose=verbose)
    prof.disable()
    out_dir = Path(cfg.out_dir) / "chain"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = pstats.Stats(prof)
    rows = sorted(stats.stats.items(), key=lambda kv: -kv[1][3])

    def ours(fname: str) -> bool:
        return "/fedverify/" in fname or "/backend/" in fname

    def table(items: list) -> list[dict]:
        return [{"function": f"{Path(f).name}:{line}({func})", "ncalls": nc, "tsub_s": tt, "ttot_s": ct}
                for (f, line, func), (cc, nc, tt, ct, _callers) in items[:top]]

    every, mine = table(rows), table([kv for kv in rows if ours(kv[0][0])])
    buf = io.StringIO()
    buf.write(f"cProfile of one full run: {cfg.exp}/{cfg.cell}/seed{cfg.seed}, {cfg.repeats} repeat(s), {cfg.rounds} "
              f"rounds, backend {cfg.chain_backend}; after a 1-round warm-up. Total {stats.total_tt:.3f} s.\n"
              "tsub = time inside the function itself (tottime); ttot = time including its callees (cumtime).\n"
              "Recursive functions can show a ttot larger than the total (cProfile counts each frame).\n")
    for title, rows_ in (("Top functions by ttot, all code", every),
                         ("Top functions by ttot, FedVerify / BlockVerify code only", mine)):
        buf.write(f"\n{title}\n{'ncalls':>10} {'tsub_s':>10} {'ttot_s':>10}  function\n")
        for r in rows_:
            buf.write(f"{r['ncalls']:>10} {r['tsub_s']:>10.4f} {r['ttot_s']:>10.4f}  {r['function']}\n")
    (out_dir / "profile.txt").write_text(buf.getvalue(), encoding="utf-8")
    (out_dir / "profile_top10.json").write_text(json.dumps({
        "run": f"{cfg.exp}/{cfg.cell}/seed{cfg.seed}", "total_s": stats.total_tt, "all_code": every,
        "fedverify_code": mine}, indent=2) + "\n", encoding="utf-8")
    return out_dir / "profile.txt"


def main(argv: Optional[list[str]] = None) -> int:
    cfg, a = build_config(argv)
    if a.profile:
        print(f"profile -> {profile_run(cfg, verbose=not a.quiet)}")
    else:
        run(cfg, verbose=not a.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
