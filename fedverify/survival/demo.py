"""One-command live demo: a short, real federated run on UCI heart, narrated round by round (Phase 10).

    python3 -m fedverify.survival.demo
    python3 -m fedverify.survival.demo --seed 0 --chain-backend local
    python3 -m fedverify.survival.demo --push                   # also publish the run to the dashboard API

What runs (nothing is simulated except the adversary's moves, and those go through the real checks):
  UCI heart, paper mode, the 4 hospitals, fold 0 of the study's stratified split for the seed; 3 rounds of FedAvg
  logistic regression with DP-SGD at epsilon 4 (the study's "Ours" configuration); every update serialized, given a
  content ID, signed with the hospital's Ed25519 TEST key, verified, Merkle-committed and mined into a block of the
  existing BlockVerify proof-of-work chain; FedVerify-Forensics screens the committed updates with the tau read from
  the calibration file (never typed in).

  round 1  a clean round
  round 2  the study's attacker for this seed (attackers_for: Switzerland for seed 0) flips the sign of its update;
           the screen decides, and the line says which score fired (or, honestly, that it missed)
  round 3  an old signed submission is sent again (double spend), and a submission in one hospital's name is
           signed with another hospital's key; both go through the commit layer's real checks
  end      one hospital's stored update is checked against the root read from the mined block (inclusion proof),
           then one byte of a copy is flipped and the same check is run again

It writes a normal run folder, <out-dir>/demo/heart_ours_lr_demo/seed<seed>/ (config.json, rounds.jsonl,
commitments.jsonl, chain_blocks.jsonl, the committed update bytes under updates/, and demo_output.txt, a copy of
everything printed). `python3 -m fedverify.chain.push <that folder>` shows it in the Federated tab.

--chain-backend local (default) needs no network at all. --chain-backend algorand keeps the same local ledger for the
round-to-round checks and also broadcasts each round root to Algorand Testnet (needs network and a funded wallet);
if that fails, the demo says so and carries on locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

DEMO_CELL = "heart_ours_lr_demo"
ATTACK_ROUND = 2
CHAIN_ATTACK_ROUND = 3
ROUNDS = 3
PROBE_WORDS = {
    "s_norm": "the norm score (its update is unusually large or small)",
    "s_dir": "the direction score (it points away from the other hospitals' consensus)",
    "s_coord": "the coordinate score (an unusual share of its weights are outliers)",
}


class Narrator:
    """Prints plain-English lines and keeps a copy for demo_output.txt."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.t0 = time.perf_counter()

    def say(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def head(self, text: str) -> None:
        self.say("")
        self.say(f"== {text} " + "=" * max(3, 78 - len(text)))


def short(h: Optional[str], n: int = 12) -> str:
    return "—" if not h else (h[:n] + "…")


def anchored_root(chain: Any, run_id: str, rnd: int) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """The round root as the LEDGER holds it: (root, leaf_count, block index) from the fl_round_commit transaction."""
    for b in chain.chain:
        for t in b.transactions:
            if t.get("type") == "fl_round_commit" and t.get("run_id") == run_id and int(t.get("round", -1)) == rnd:
                return t.get("root"), int(t.get("leaf_count", 0)), int(b.index)
    return None, None, None


def inclusion_check(update_bytes: bytes, hospital: str, commitment: dict, ledger_root: Optional[str]) -> dict:
    """Does this update, as stored, prove into the root on the ledger? Recomputes digest -> leaf -> folded root."""
    from ..chain.commitment import leaf_hashes, verify_proof
    from ..chain.contentid import verify_cid
    leaves = commitment["leaves"]
    i = next(p for p, lf in enumerate(leaves) if lf["client_id"] == hospital)
    digest = hashlib.sha256(update_bytes).hexdigest()
    entries = [(lf["client_id"], digest if p == i else lf["digest"]) for p, lf in enumerate(leaves)]
    leaf = leaf_hashes(entries)[i]
    sub = next((s for s in commitment.get("submissions", []) if s.get("hospital_id") == hospital
                and s.get("digest") == leaves[i]["digest"]), None)
    cid_ok = bool(sub) and verify_cid(update_bytes, str(sub.get("cid")))
    ok = ledger_root is not None and verify_proof(leaf, leaves[i]["proof"], ledger_root)
    return {"ok": bool(ok), "index": i, "digest": digest, "committed_digest": leaves[i]["digest"], "leaf": leaf,
            "proof_levels": len(leaves[i]["proof"]), "cid_ok": cid_ok}


def explain_rejection(score: dict, tau: float) -> str:
    by = score.get("rejected_by") or ""
    if by.startswith("health:"):
        return f"a hard health flag fired ({by.split(':', 1)[1]}), regardless of tau"
    val = score.get(by, score.get("combined"))
    return f"{PROBE_WORDS.get(by, by)} fired: {by} = {val:.2f} > tau = {tau:.2f}"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fedverify.survival.demo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="the run seed (also picks the attacker and the fold split)")
    ap.add_argument("--chain-backend", default="local", choices=["local", "algorand"],
                    help="local (default): the repository's own PoW chain, no network. algorand: the same, plus each "
                         "round root broadcast to Algorand Testnet")
    ap.add_argument("--out-dir", default="fedverify/results", help="where the run folder goes (<out-dir>/demo/...)")
    ap.add_argument("--results-dir", default="fedverify/results",
                    help="the study's results, read only for the context line on screening recall")
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--push", action="store_true", help="publish the run to the dashboard API afterwards")
    ap.add_argument("--api", default=None, help="dashboard API for --push (default: fedverify.chain.push's)")
    a = ap.parse_args(argv)

    warnings.filterwarnings("ignore")                  # Opacus/torch notices would drown the narration
    import numpy as np
    import torch

    from ..aggregators import build_aggregator
    from ..core.client import ClientUpdate
    from ..core.privacy import privacy_report
    from ..chain.contentid import content_id
    from ..chain.signing import make_txn_id, public_key_bytes, sign_submission
    from . import datasets, poison, runner
    from .chain_round import LOGICAL_TS_BASE, ChainSession
    from .common import codes_diag
    from .config import SurvivalConfig, derive_seed
    from .data import fold_data, make_folds
    from .dp import hospital_plan
    from .fl_lr import dp_sgd_epochs, pos_weight
    from .metrics import compute, thresholds_for_test
    from .models import build_logistic, get_flat_params, predict_risk, set_flat_params
    from .serialize import serialize_lr

    torch.set_num_threads(int(a.torch_threads))
    say = Narrator()
    seed = int(a.seed)
    out_root = Path(a.out_dir)
    run_dir = out_root / "demo" / DEMO_CELL / f"seed{seed}"
    run_id = f"demo_heart_ours_lr_s{seed}"

    # ── set-up ───────────────────────────────────────────────────────────────
    base = SurvivalConfig(dataset="heart", heart_mode="paper", family="lr", method="ours_lr", rounds=ROUNDS,
                          attack="sign_flip", chain_backend="local", seed=seed, out_dir=str(out_root))
    try:
        cfg = runner.with_tau(base)                       # tau from the calibration file, with its source
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Cannot start the demo: the screening threshold must come from the calibration file, and it could not "
              f"be read ({e}). Run `python3 -m fedverify.survival.experiments calibration` first.", file=sys.stderr)
        return 2
    loaded = datasets.load(cfg)
    cohort = loaded.cohort
    hosp = list(cfg.hospitals)
    index = cfg.hospital_index
    tr, te = make_folds(cohort, seed, cfg.folds)[0]
    fd = fold_data(cohort, loaded.X, tr, te)
    X_tr = torch.as_tensor(fd.X_train, dtype=torch.float32)
    X_te = torch.as_tensor(fd.X_test, dtype=torch.float32)
    y_tr = torch.as_tensor(fd.event_train, dtype=torch.float32)
    rows = {h: np.flatnonzero(fd.hosp_train == h) for h in hosp}
    n_rows = {h: int(len(rows[h])) for h in hosp}
    attacker = poison.attackers_for(cfg, seed)[0]

    say.say("FedVerify live demo: federated learning across 4 hospitals, every round committed to a blockchain.")
    say.head("Set-up")
    say.say(f"Data: UCI heart disease, {cohort.n} patients in 4 hospitals (paper mode). Training rows per hospital in "
            f"this split: " + ", ".join(f"{h} {n_rows[h]}" for h in hosp) + f"; {len(te)} patients held out for testing.")
    say.say(f"Model: logistic regression, {ROUNDS} rounds of FedAvg. Each hospital trains only on its own patients, "
            f"with differential privacy (DP-SGD, epsilon = {cfg.epsilon:g}); raw patient data never leaves it.")
    say.say(f"Screen: FedVerify-Forensics, threshold tau = {cfg.tau:.3f}, read from the calibration file "
            f"({cfg.tau_source}).")
    session = ChainSession(cfg, run_seed=seed)
    fc = session.for_fold(run_id)
    chain = session.chain
    say.say(f"Ledger: the BlockVerify proof-of-work chain (difficulty {session.difficulty}), run on this machine"
            + (" — each round root will also be broadcast to Algorand Testnet." if a.chain_backend == "algorand"
               else "; no network is used."))
    say.say("Each hospital has an Ed25519 signing key (test keys for the demo), registered once, write-once:")
    for h in hosp:
        say.say(f"  {h:<14} public key {short(public_key_bytes(session.keys[h]).hex(), 16)}")
    say.say(f"Scripted adversary: in round {ATTACK_ROUND}, {attacker} (the study's attacker for seed {seed}) flips the "
            f"sign of its update; in round {CHAIN_ATTACK_ROUND}, an old submission is replayed and a submission is "
            "signed with the wrong hospital's key.")

    plans = {h: hospital_plan(cfg, index[h], n_rows[h], {}) for h in hosp}
    gen = {h: torch.Generator().manual_seed(derive_seed(seed, 0, "loader", h)) for h in hosp}
    pw = {h: pos_weight(y_tr[rows[h]], cfg.class_weighted_loss) for h in hosp}
    agg = build_aggregator("forensics")
    model = build_logistic(X_tr.shape[1])
    g = get_flat_params(model)
    history: dict[str, list] = {h: [] for h in hosp}
    records, commitments, events = [], [], {"screen": [], "ledger": [], "algorand": []}
    if run_dir.exists():
        shutil.rmtree(run_dir)                          # the demo's own folder only: a fresh run every time
    (run_dir / "updates").mkdir(parents=True)

    for rnd in range(1, ROUNDS + 1):
        say.head(f"Round {rnd}")
        skip = fc.check_previous()
        if rnd > 1:
            if skip:
                say.say(f"Every hospital checked the previous block: FAILED ({next(iter(skip.values()))}); they skip.")
                return 1
            say.say("Every hospital first checks the previous block on the ledger: it is valid and still holds round "
                    f"{rnd - 1}'s root.")
        # local training
        deltas, losses = {}, {}
        for h in hosp:
            local = build_logistic(X_tr.shape[1])
            set_flat_params(local, g)
            noise = torch.Generator().manual_seed(derive_seed(seed, 0, rnd, h, "dp-noise"))
            losses[h] = dp_sgd_epochs(local, X_tr[rows[h]], y_tr[rows[h]], pw[h], cfg, gen[h], plans[h], noise)
            deltas[h] = get_flat_params(local) - g
        if rnd == ATTACK_ROUND:
            deltas[attacker] = poison.poisoned_delta(deltas[attacker], cfg, seed, rnd, attacker, 1)
            say.say(f"{attacker} is malicious this round: it flips the sign of its update before signing it.")
        say.say("Each hospital trains locally, then signs its update:")
        subs, payloads, timings = fc.prepare(rnd, {h: (lambda h=h: serialize_lr(run_id, rnd, h, n_rows[h], deltas[h]))
                                                    for h in hosp})
        for h in hosp:
            history[h].append((rnd, subs[h], payloads[h]))
            say.say(f"  {h:<14} content ID {short(subs[h]['cid'], 18)}  digest {short(subs[h]['digest'])}  "
                    f"txn {short(subs[h]['txn_id'], 8)}")
        arrivals = [(subs[h], payloads[h]) for h in hosp]
        adversarial: list[dict] = []
        if rnd == CHAIN_ATTACK_ROUND:
            victim, thief = hosp[0], hosp[1]
            replayed = history[thief][0]
            ts = LOGICAL_TS_BASE + rnd
            forged_payload = serialize_lr(run_id, rnd, victim, n_rows[victim], -deltas[victim])
            fields = {"run_id": run_id, "round": rnd, "hospital_id": victim, "cid": content_id(forged_payload),
                      "digest": hashlib.sha256(forged_payload).hexdigest(), "timestamp": ts,
                      "txn_id": make_txn_id(victim, session.registry.next_counter(victim), ts)}
            forged = sign_submission(session.keys[thief], fields, fc.signed_fields)
            say.say(f"Adversary: someone holding {thief}'s REAL key submits an update in {victim}'s name.")
            say.say(f"Adversary: {thief}'s signed submission from round {replayed[0]} is sent again, unchanged "
                    "(double spend).")
            arrivals = [(forged, forged_payload)] + arrivals + [(dict(replayed[1]), replayed[2])]
            adversarial = [{"what": f"the update in {victim}'s name, signed with {thief}'s key", "arrival": 0},
                           {"what": f"{thief}'s round-{replayed[0]} submission, sent again", "arrival": len(arrivals) - 1}]
        res = fc.process(rnd, arrivals, timings)
        for adv in adversarial:
            got = next((r for r in res.record["rejected"]
                        if r["hospital"] == arrivals[adv["arrival"]][0]["hospital_id"]), None)
            accepted = any(e["arrival"] == adv["arrival"] for e in res.entries)
            if accepted:
                say.say(f"!! The ledger ACCEPTED {adv['what']}: this would be a defence failure.")
            else:
                say.say(f"REJECTED at the ledger: {adv['what']} — {got['reason'] if got else 'rejected'}.")
            events["ledger"].append({"round": rnd, "attempt": adv["what"], "accepted": accepted,
                                     "reason": None if got is None else got["reason"]})
        block = chain.chain[-1]
        root, count, bidx = anchored_root(chain, run_id, rnd)
        say.say(f"The ledger verified {len(res.accepted)} signed updates and committed the round: Merkle root "
                f"{short(res.record['root'], 16)} over {res.record['leaf_count']} leaves, mined into block #{bidx} "
                f"(hash {short(block.hash, 16)}, nonce {block.nonce}).")
        # the screen, on the updates rebuilt from the committed bytes
        used = [ClientUpdate(client_id=index[h], delta=res.models[h], num_samples=n_rows[h], train_loss=losses[h],
                             wall_time_s=0.0) for h in res.accepted]
        delta, diag = agg.aggregate(used, cfg, rnd)
        diag = codes_diag(diag, cfg.dataset_hospitals)
        say.say(f"The screen scores each committed update (robust z across hospitals; rejected above tau = "
                f"{cfg.tau:.2f}):")
        for h in res.accepted:
            s = diag["scores"].get(h, {})
            line = (f"  {h:<14} norm {s.get('s_norm', 0):.2f}  direction {s.get('s_dir', 0):.2f}  "
                    f"coordinates {s.get('s_coord', 0):.2f}  -> ")
            if h in diag["rejected"]:
                say.say(line + f"REJECTED by the screen: {explain_rejection(s, cfg.tau)}.")
            else:
                say.say(line + "accepted")
        if rnd == ATTACK_ROUND:
            caught = attacker in diag["rejected"]
            events["screen"].append({"round": rnd, "attacker": attacker, "caught": caught,
                                     "rejected_by": diag["scores"].get(attacker, {}).get("rejected_by")})
            if not caught:
                s = diag["scores"].get(attacker, {})
                say.say(f"The screen MISSED {attacker}'s poisoned update this time: its highest score was "
                        f"{s.get('combined', 0):.2f}, below tau = {cfg.tau:.2f}. With only 4 hospitals, one outlier "
                        "is hard to tell from honest differences between hospitals.")
            try:
                from ..analysis.make_fl_healthcare_tables import Results, load_spec
                from ..analysis.make_results_draft import recall_sentence
                spec = load_spec()
                sentence = recall_sentence(Results(Path(a.results_dir), spec), spec, "demo")
                if "—" not in sentence:
                    say.say(f"For context, from the full study: {sentence}")
            except Exception:                                # the results are context only, never required
                pass
        g = g + delta
        set_flat_params(model, g)
        train_risk, test_risk = predict_risk(model, X_tr), predict_risk(model, X_te)
        thr = thresholds_for_test(fd.event_train, train_risk, fd.hosp_train, fd.hosp_test, False)
        pred = (test_risk >= thr).astype(np.int64)
        m = compute(["accuracy", "f1", "roc_auc"], fd.time_test, fd.event_test, test_risk, pred, "risk")
        say.say(f"The {len(diag['accepted'])} accepted updates are averaged into the shared model. On the held-out "
                f"patients: accuracy {m['accuracy']:.3f}, ROC-AUC {m['roc_auc']:.3f}.")
        commitment = json.loads(json.dumps(res.commitment))
        algo = None
        if a.chain_backend == "algorand":
            try:
                from ..chain.anchor import _import_backend
                algo = _import_backend("algorand_client").broadcast_fl_round(run_id, rnd, res.record["root"],
                                                                            res.record["leaf_count"])
            except Exception as e:                           # no network, no wallet, no SDK: say so and go on
                algo = {"success": False, "error": f"{type(e).__name__}: {e}"}
            if algo.get("success"):
                say.say(f"Also anchored on Algorand Testnet: txid {algo['txid']} "
                        f"(https://lora.algokit.io/testnet/transaction/{algo['txid']}).")
                commitment.update(local_block=commitment["txid"], txid=algo["txid"], backend="algorand")
            else:
                say.say(f"Algorand broadcast failed ({algo.get('error')}); the local ledger still holds this root.")
            events["algorand"].append({"round": rnd, **{k: v for k, v in algo.items() if k != "raw"}})
        for h in res.accepted:                                # the committed bytes, as an auditor would keep them
            p = run_dir / "updates" / f"round{rnd}" / f"{h}.bin"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(arrivals[next(e["arrival"] for e in res.entries if e["hospital"] == h)][1])
        truth = [attacker] if rnd == ATTACK_ROUND else []
        tp = sum(1 for h in truth if h in diag["rejected"])
        records.append({"round": rnd, "model_family": "lr", "method": "ours_lr", "repeat": 0, "fold": 0,
                        "test_accuracy": m["accuracy"], "test_f1": m["f1"], "test_roc_auc": m["roc_auc"],
                        "test_n": int(len(te)), "test_positives": int(fd.event_test.sum()),
                        "mean_train_loss": float(np.mean(list(losses.values()))),
                        "per_client_num_samples": n_rows, "diag": diag, "commit": res.record,
                        "attack": {"attack": "sign_flip", "active": rnd == ATTACK_ROUND, "attackers": truth,
                                   "rejected": diag["rejected"], "present": list(res.accepted), "tp": tp,
                                   "fp": len([h for h in diag["rejected"] if h not in truth]),
                                   "fn": len(truth) - tp},
                        "ledger_attempts": [x for x in events["ledger"] if x["round"] == rnd],
                        "privacy": {h: privacy_report(plans[h]) for h in hosp},
                        "algorand": algo})
        commitments.append(commitment)

    # ── the proof check, and the same check after flipping one byte ──────────
    say.head("Audit: is a stored update really the one that was committed?")
    last = commitments[-1]
    who = last["leaves"][0]["client_id"]
    stored = run_dir / "updates" / f"round{ROUNDS}" / f"{who}.bin"
    root, count, bidx = anchored_root(chain, run_id, ROUNDS)
    say.say(f"Read the round-{ROUNDS} root from block #{bidx} of the ledger itself: {short(root, 16)} "
            f"({count} leaves). Read {who}'s stored update ({stored.stat().st_size} bytes) from disk.")
    good = inclusion_check(stored.read_bytes(), who, last, root)
    say.say(f"SHA-256 of the stored bytes = {short(good['digest'], 16)} (the committed digest is "
            f"{short(good['committed_digest'], 16)}); content ID check: {'ok' if good['cid_ok'] else 'FAILED'}.")
    say.say(f"Leaf at position {good['index']} folded up {good['proof_levels']} levels of its inclusion proof: "
            + ("VERIFIED, it equals the root on the ledger." if good["ok"] else "does NOT equal the root on the ledger."))
    tampered = run_dir / "tamper_demo" / f"{who}.bin"
    tampered.parent.mkdir(parents=True, exist_ok=True)
    data = bytearray(stored.read_bytes())
    pos = len(data) - 3                               # a byte inside the model weights, not the header
    data[pos] ^= 0x01
    tampered.write_bytes(bytes(data))
    say.say(f"Now flip one bit of byte {pos} in a copy of that update ({tampered.relative_to(run_dir)}) and run the "
            "same check:")
    bad = inclusion_check(tampered.read_bytes(), who, last, root)
    say.say(f"SHA-256 of the tampered bytes = {short(bad['digest'], 16)}; content ID check: "
            f"{'ok' if bad['cid_ok'] else 'FAILED'}.")
    say.say("Tamper check " + ("FAILED: the tampered update no longer proves into the root on the ledger, so the "
                               "change is detected." if not bad["ok"] else "PASSED: the change was NOT detected."))
    valid = chain.is_chain_valid()
    say.say(f"Whole ledger re-checked (every block's hash, link and proof of work): "
            f"{'valid' if valid['valid'] else 'INVALID: ' + '; '.join(valid['errors'][:3])} "
            f"({len(chain.chain)} blocks).")

    # ── the run folder ───────────────────────────────────────────────────────
    to_dict = cfg.to_dict()
    config = {"config": {**to_dict, "dataset": loaded.info["dataset"], "num_clients": len(hosp),
                         "aggregator": cfg.aggregation, "repeat": 0, "fold": 0, "repeat_seed": seed, "run_id": run_id,
                         "demo": True},
              "model_family": "lr", "attacker_ids": [attacker], "hospital_ids": index, "data": loaded.info,
              "demo": {"attack_round": ATTACK_ROUND, "chain_attack_round": CHAIN_ATTACK_ROUND,
                       "chain_backend": a.chain_backend, "fold": "fold 0 of make_folds(seed)",
                       "train_patients": int(len(tr)), "test_patients": int(len(te))},
              "chain": session.describe()}
    (run_dir / "config.json").write_text(json.dumps(runner._plain(config), indent=2) + "\n", encoding="utf-8")
    with open(run_dir / "rounds.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(runner._plain(r)) + "\n")
    with open(run_dir / "commitments.jsonl", "w", encoding="utf-8") as f:
        for c in commitments:
            f.write(json.dumps(runner._plain(c)) + "\n")
    with open(run_dir / "chain_blocks.jsonl", "w", encoding="utf-8") as f:
        for b in session.block_log:
            f.write(json.dumps(runner._plain(b)) + "\n")
    summary = {"run_id": run_id, "seed": seed, "attacker": attacker, "events": events,
               "proof_verified": good["ok"], "tamper_detected": not bad["ok"], "ledger_valid": valid["valid"],
               "final_test": records[-1]["test_accuracy"], "final_roc_auc": records[-1]["test_roc_auc"],
               "elapsed_s_before_writing": time.perf_counter() - say.t0}
    (run_dir / "summary.json").write_text(json.dumps(runner._plain(summary), indent=2) + "\n", encoding="utf-8")

    say.head("Done")
    say.say(f"Run folder: {run_dir} (config.json, rounds.jsonl, commitments.jsonl, chain_blocks.jsonl, updates/).")
    pushed = None
    if a.push:
        from ..chain import push as P
        ok, fail, rid = P.push(str(run_dir), a.api or P.DEFAULT_API, verbose=False)
        pushed = not fail and ok > 0
        say.say(f"Published to the dashboard API as '{rid}': {ok} round(s) committed, {fail} failed."
                + ("" if pushed else " Is the backend running? (cd backend && python3 app.py)"))
    else:
        say.say(f"Show it in the dashboard: start the backend (cd backend && python3 app.py), then run "
                f"`python3 -m fedverify.chain.push {run_dir}` and pick '{run_id}' in the Federated tab.")
    say.say(f"Finished in {time.perf_counter() - say.t0:.1f} s.")
    (run_dir / "demo_output.txt").write_text("\n".join(say.lines) + "\n", encoding="utf-8")
    held = good["ok"] and not bad["ok"] and valid["valid"] and not any(x["accepted"] for x in events["ledger"])
    if held:                                            # DEMO.md's fallback: the last run in which everything held
        fallback = run_dir.parent / "fallback"
        if fallback.exists():
            shutil.rmtree(fallback)
        shutil.copytree(run_dir, fallback)
    return 0 if held and pushed is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
