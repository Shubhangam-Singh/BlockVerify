"""Push a finished run's round commitments to the BlockVerify API.

Training writes `commitments.jsonl` (roots, leaves, proofs) and `rounds.jsonl` (metrics)
to disk, but nothing sends them anywhere — the aggregator and the server are deliberately
separate processes, since in a real deployment they are separate machines. This is the
step that publishes a run so the Federated tab can show it.

The server re-derives the Merkle root from the leaves it receives and rejects the commit
if it disagrees, so a push that succeeds is itself a check that what is on disk folds to
the root that was anchored.

    python3 -m fedverify.chain.push fedverify/results/demo/mnist_K5_a0.5_epsinf/seed0
    python3 -m fedverify.chain.push <run_dir> --api http://localhost:5000/api
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_API = "http://localhost:5000/api"


def _post(url: str, payload: dict, timeout: float = 20.0):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def load_run(run_dir: str):
    """(commitments, metrics_by_round, run_id) from a finished run directory."""
    cpath = os.path.join(run_dir, "commitments.jsonl")
    if not os.path.exists(cpath):
        raise FileNotFoundError(
            f"{cpath} not found. That file is only written when the run used --commit; "
            "re-run the training with --commit --chain-backend local (or algorand).")
    commits = [json.loads(l) for l in open(cpath) if l.strip()]

    metrics, rpath = {}, os.path.join(run_dir, "rounds.jsonl")
    if os.path.exists(rpath):
        for line in open(rpath):
            if not line.strip():
                continue
            rec = json.loads(line)
            metrics[int(rec["round"])] = {
                "test_acc": rec.get("test_acc"),
                "macro_f1": rec.get("macro_f1"),
                "diag": rec.get("diag") or {},
                "attack": rec.get("attack") or {},
            }

    # Run-level context so the UI can label a run by WHAT IT IS rather than by its
    # 60-character run_id.
    meta, cpath2 = {}, os.path.join(run_dir, "config.json")
    if os.path.exists(cpath2):
        full = json.load(open(cpath2))
        c = full.get("config", {})
        eps = c.get("epsilon")
        meta = {
            "dataset": c.get("dataset"), "numClients": c.get("num_clients"),
            "rounds": c.get("rounds"), "alpha": c.get("alpha"),
            "epsilon": ("inf" if eps in (None, "inf") else eps),
            "aggregator": c.get("aggregator"), "attack": c.get("attack") or "none",
            "attackerFrac": c.get("attacker_frac"), "tau": c.get("tau"),
            "tauSource": c.get("tau_source"), "seed": c.get("seed"),
            "attackerIds": [str(x) for x in full.get("attacker_ids", [])],
            "numParams": full.get("num_params"),
        }

    run_id = commits[0].get("run_id") if commits else None
    if not run_id:
        run_id = (json.load(open(cpath2))["config"]["run_id"]
                  if os.path.exists(cpath2) else "run")
    return commits, metrics, run_id, meta


def push(run_dir: str, api: str = DEFAULT_API, run_id: str = None, verbose: bool = True):
    commits, metrics, detected, meta = load_run(run_dir)
    rid = run_id or detected
    ok = fail = 0
    for c in commits:
        rnd = int(c["round"])
        m = metrics.get(rnd, {})
        diag = m.get("diag", {})
        body = {
            "run_id": rid, "round": rnd,
            "root": c["root"], "leaf_count": c["leaf_count"],
            "txid": c.get("txid"), "backend": c.get("backend"),
            "leaves": [{"client_id": l["client_id"], "digest": l["digest"]}
                       for l in c["leaves"]],
            "accepted": [str(x) for x in diag.get("accepted", [])],
            "rejected": [str(x) for x in diag.get("rejected", [])],
        }
        atk = m.get("attack") or {}
        if m.get("test_acc") is not None:
            body["metrics"] = {k: m[k] for k in ("test_acc", "macro_f1")
                               if m.get(k) is not None}
            # ASR lives in the attack block, not beside the accuracy, but the UI wants it
            # as a headline metric — without this the backdoor stat card never renders.
            asr = atk.get("asr")
            if asr is not None and asr == asr:            # not NaN
                body["metrics"]["asr"] = float(asr)
        if atk.get("attackers") is not None:
            body["attackers"] = [str(x) for x in atk["attackers"]]
        if meta:
            body["meta"] = meta
        code, resp = _post(f"{api.rstrip('/')}/fl/round/commit", body)
        if code == 201:
            ok += 1
            if verbose:
                print(f"  round {rnd:>3}  committed  root={c['root'][:16]}… "
                      f"leaves={c['leaf_count']}")
        else:
            fail += 1
            print(f"  round {rnd:>3}  FAILED ({code}): {resp.get('error')}",
                  file=sys.stderr)
    if verbose:
        print(f"\n{ok} round(s) committed as run '{rid}'"
              + (f", {fail} failed" if fail else ""))
        if ok and not fail:
            print(f"Open the Federated tab and pick '{rid}' from the selector.")
    return ok, fail, rid


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a results directory containing commitments.jsonl")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--run-id", default=None, help="override the run id sent to the server")
    a = ap.parse_args(argv)
    try:
        _ok, fail, _rid = push(a.run_dir, a.api, a.run_id)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
