"""The reference paper's three blockchain attacks, run against OUR commit layer (Revision 2 §4A; Figs 9 and 10).

    python3 -m fedverify.survival.chain_attacks --attack all --rounds 100 --seed 0            # S_DS, S_TM, S_EC
    python3 -m fedverify.survival.chain_attacks --attack all --rounds 20 --seed 0             # smoke
    python3 -m fedverify.survival.chain_attacks --attack all --rounds 100 --seed 0 --defence-off

Every simulated round is one real round of the commit pipeline (chain_round.FoldCommitter: signed submissions,
verification, one digest per slot, Merkle root, anchor on the existing chain) for the dataset's hospitals, plus one
attack attempt whose variant and target hospital are drawn from the seed:

  double_spending      conflicting_update      the target signs a SECOND, different update for the same round
                       replayed_submission     an earlier signed submission of the target is sent again, unchanged
                       stale_update_respent    the target re-signs one of its OLD updates for the current round
                       duplicate_resubmission  the target's submission of this round is sent twice
     defences: one digest per (run, round, hospital); the submission and the payload header must name the current
               round; the txn_id must hash an unused counter
  malleability         timestamp_changed       in transit: the timestamp is changed and the txn_id recomputed to
                                               match it; the model bytes are untouched
                       txn_id_changed          in transit: the txn_id is replaced by another well-formed one
                       leaf_order_swapped      after anchoring: two leaves of the stored round record are swapped,
                                               with indices, leaf hashes and proofs recomputed to look consistent
     defences: the signature covers every field; the Merkle leaf binds its position (audited against the anchored root)
  endpoint_compromise  unregistered_key        a submitter with its own key claims an id that has no registered key
                       forged_signature        a submitter with its own key claims to be the target hospital
                       wrong_hospital_key      a compromised hospital signs, with its REAL key, as the target
                       key_reregistration      the adversary tries to register its key for the target, then submits
     defence:  the signature must verify under the key registered for the claimed hospital (write-once registry)

A round is HELD when no adversarial submission was accepted (and, for leaf_order_swapped, the audit caught the swap).
Resilience S = 100 x held rounds / simulated rounds, attack success = 100 - S: the reference paper's S_DS, S_TM and S_EC.
The replay variants need an earlier round, so round 1 draws among the other variants. Honest payloads are real
serialized logistic updates of the dataset's model size, with values drawn from the seed (no check reads the values).
--defence-off switches off exactly the defences listed for the attack (chain_round.Defences), which shows what each
stops. Variants the simulation does NOT cover are listed in THREAT_MODEL.md.

Results: <out-dir>/chain_attacks/<attack>[_defence_off][_<dataset>]/seed<N>/{config.json, rounds.jsonl, summary.json}
(the dataset suffix only for a dataset other than the default heart, so heart results keep their path). The same
seed gives byte-identical files (no wall-clock value is written).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ..chain.commitment import inclusion_proof, leaf_hashes
from ..chain.contentid import content_id
from ..chain.signing import make_txn_id, public_key_bytes, sign_submission
from . import datasets
from .chain_round import ALL_DEFENCES, LOGICAL_TS_BASE, ChainSession, CommitResult, Defences, audit_commitment
from .config import DATASETS, SurvivalConfig, derive_seed
from .serialize import serialize_lr

ATTACKS: dict[str, tuple[str, ...]] = {
    "double_spending": ("conflicting_update", "replayed_submission", "stale_update_respent", "duplicate_resubmission"),
    "malleability": ("timestamp_changed", "txn_id_changed", "leaf_order_swapped"),
    "endpoint_compromise": ("unregistered_key", "forged_signature", "wrong_hospital_key", "key_reregistration"),
}
METRIC = {"double_spending": "S_DS", "malleability": "S_TM", "endpoint_compromise": "S_EC"}
DEFENCES_FOR: dict[str, tuple[str, ...]] = {
    "double_spending": ("one_digest_per_slot", "round_binding", "fresh_txn_counter"),
    "malleability": ("full_signature", "merkle_position_audit"),
    "endpoint_compromise": ("registered_keys",),
}
NEEDS_HISTORY = frozenset({"replayed_submission", "stale_update_respent"})
UNREGISTERED_ID = "UNREGISTERED-SITE"
DEFINITION = ("S = 100 x (simulated rounds in which the defence held) / (total simulated rounds); attack success = "
              "100 - S (the reference paper's definitions of S_DS, S_TM, S_EC)")


def defences_for(attack: str, defence_off: bool = False) -> Defences:
    """All defences on; with defence_off, the ones that stop this attack switched off."""
    if not defence_off:
        return ALL_DEFENCES
    return dataclasses.replace(ALL_DEFENCES, **{k: False for k in DEFENCES_FOR[attack]})


@dataclass
class Plan:
    """One round's attack: the arrivals the aggregator sees, which of them are adversarial, an optional check after
    the round (the audit of a tampered record), and what the adversary did."""
    arrivals: list
    adversarial: list
    post: Optional[Callable[[CommitResult], tuple[bool, str]]] = None
    note: dict = field(default_factory=dict)


class Simulation:
    """The honest hospitals of one dataset, one fold-run lineage on the chain, and an adversary."""

    def __init__(self, attack: str, seed: int = 0, dataset: str = "heart", backend: str = "mock",
                 defences: Defences = ALL_DEFENCES, data_path: Optional[str] = None):
        if attack not in ATTACKS:
            raise ValueError(f"unknown chain attack {attack!r}; expected one of {sorted(ATTACKS)}")
        self.attack, self.seed, self.defences = attack, int(seed), defences
        self.cfg = SurvivalConfig(dataset=dataset, family="lr", method="chain_fedavg_lr", chain_backend=backend,
                                  data_path=data_path)
        loaded = datasets.load(self.cfg, verbose=False)
        self.hospitals = list(self.cfg.hospitals)
        self.numel = len(loaded.feature_names) + 1                  # the logistic model's parameter count
        self.n = {h: int((loaded.cohort.hospital == h).sum()) for h in self.hospitals}
        self.session = ChainSession(self.cfg, run_seed=self.seed, defences=defences)
        self.run_id = f"chain_attacks_{attack}_s{self.seed}"
        self.fc = self.session.for_fold(self.run_id)
        self.history: dict[str, list] = {h: [] for h in self.hospitals}     # earlier (round, submission, payload)
        adv_seed = hashlib.sha256(f"fedverify|ADVERSARY-KEY|{self.seed}".encode("utf-8")).digest()
        self.adversary_key = Ed25519PrivateKey.from_private_bytes(adv_seed)

    # ── payloads ─────────────────────────────────────────────────────────────
    def update(self, rnd: int, hospital: str, tag: str = "honest") -> bytes:
        """A logistic update of the right size for `hospital` in round `rnd` (values from the seed)."""
        rng = np.random.default_rng(derive_seed(self.seed, tag, rnd, hospital))
        delta = rng.normal(0.0, 0.01, size=self.numel).astype(np.float32)
        return serialize_lr(self.run_id, rnd, hospital, self.n.get(hospital, 100), delta)

    def forged(self, rnd: int, claimed: str, key: Ed25519PrivateKey) -> tuple[dict, bytes]:
        """A submission for `claimed`, signed with `key` (not the claimed hospital's own), carrying that key."""
        payload = self.update(rnd, claimed, tag="adversary")
        ts = LOGICAL_TS_BASE + int(rnd)
        fields = {"run_id": self.run_id, "round": int(rnd), "hospital_id": claimed, "cid": content_id(payload),
                  "digest": hashlib.sha256(payload).hexdigest(), "timestamp": ts,
                  "txn_id": make_txn_id(claimed, self.session.registry.next_counter(claimed), ts)}
        sub = sign_submission(key, fields, self.fc.signed_fields)
        sub["public_key"] = public_key_bytes(key).hex()          # read only if registered-key checking is off
        return sub, payload

    # ── one attack attempt per round ─────────────────────────────────────────
    def plan(self, variant: str, rnd: int, target: str, subs: dict, payloads: dict,
             rng: np.random.Generator) -> Plan:
        base = [(subs[h], payloads[h]) for h in self.hospitals]
        t = self.hospitals.index(target)
        K = len(self.hospitals)
        if variant == "none":                                     # a clean round (tests use it as a warm-up)
            return Plan(base, [])
        if variant == "conflicting_update":
            other = self.update(rnd, target, tag="conflicting")
            sub2, _ = self.fc.submit(rnd, target, other)            # the target's own key and next counter
            return Plan(base + [(sub2, other)], [K])
        if variant in ("replayed_submission", "stale_update_respent"):
            old_rnd, old_sub, old_payload = self.history[target][int(rng.integers(len(self.history[target])))]
            if variant == "replayed_submission":
                return Plan(base + [(dict(old_sub), old_payload)], [K], note={"replayed_round": old_rnd})
            resigned, _ = self.fc.submit(rnd, target, old_payload)  # fresh signature, this round, old bytes
            return Plan(base + [(resigned, old_payload)], [K], note={"respent_round": old_rnd})
        if variant == "duplicate_resubmission":
            return Plan(base + [(dict(subs[target]), payloads[target])], [K])
        if variant in ("timestamp_changed", "txn_id_changed"):
            used = self.session.client_counter[target] - 1           # the counter the target just signed with
            sub = dict(subs[target])
            if variant == "timestamp_changed":
                sub["timestamp"] = sub["timestamp"] + int(rng.integers(1, 3600))
                sub["txn_id"] = make_txn_id(target, used, sub["timestamp"])
            else:
                sub["txn_id"] = make_txn_id(target, used + 1 + int(rng.integers(0, 8)), sub["timestamp"])
            arrivals = list(base)
            arrivals[t] = (sub, payloads[target])
            return Plan(arrivals, [t], note={"changed": variant.split("_")[0]})
        if variant == "leaf_order_swapped":
            i, j = sorted(int(x) for x in rng.choice(K, size=2, replace=False))

            def swap_and_audit(res: CommitResult) -> tuple[bool, str]:
                record = json.loads(json.dumps(res.commitment))     # the stored record, tampered after anchoring
                pairs = [(lf["client_id"], lf["digest"]) for lf in record["leaves"]]
                if len(pairs) < 2:
                    return True, "fewer than two leaves: nothing to swap"
                a, b = min(i, len(pairs) - 2), min(j, len(pairs) - 1)
                pairs[a], pairs[b] = pairs[b], pairs[a]
                hashes = leaf_hashes(pairs)                         # a consistent-looking forgery
                record["leaves"] = [{"index": p, "client_id": c, "digest": dg, "leaf": hashes[p],
                                     "proof": inclusion_proof(pairs, p)} for p, (c, dg) in enumerate(pairs)]
                ok, reason = audit_commitment(record, res.record["root"], res.record["leaf_count"], self.defences)
                return (not ok), (f"audit caught the swap: {reason}" if not ok else "the swap passed the audit")
            return Plan(base, [], post=swap_and_audit, note={"swapped_positions": [i, j]})
        if variant in ("unregistered_key", "forged_signature", "wrong_hospital_key", "key_reregistration"):
            note = {}
            if variant == "unregistered_key":
                claimed, key = UNREGISTERED_ID, self.adversary_key
            elif variant == "wrong_hospital_key":
                other = self.hospitals[(t + 1 + int(rng.integers(0, K - 1))) % K]
                claimed, key = target, self.session.keys[other]       # a compromised hospital's REAL key
                note["signed_with"] = other
            else:
                claimed, key = target, self.adversary_key
                if variant == "key_reregistration":
                    try:
                        self.session.registry.register(target, public_key_bytes(self.adversary_key))
                        note["registration"] = "accepted"
                    except ValueError as e:
                        note["registration"] = f"refused: {e}"
            forged = self.forged(rnd, claimed, key)
            return Plan([forged] + base, [0], note=note)              # the forgery arrives first
        raise ValueError(f"unknown variant {variant!r}")

    def run_round(self, rnd: int, variant: Optional[str] = None, target: Optional[str] = None) -> dict:
        """One attacked round. variant / target are drawn from the seed unless given (tests pin them)."""
        rng = np.random.default_rng(derive_seed(self.seed, self.attack, "round", rnd))
        options = [v for v in ATTACKS[self.attack] if not (v in NEEDS_HISTORY and rnd == 1)]
        drawn_variant = options[int(rng.integers(len(options)))]
        drawn_target = self.hospitals[int(rng.integers(len(self.hospitals)))]
        variant, target = variant or drawn_variant, target or drawn_target
        skip = self.fc.check_previous()          # every hospital verifies the previous block, as in a real round
        if skip:
            raise RuntimeError(f"round {rnd}: the previous block did not verify ({skip}); nothing here rewrites it")
        subs, payloads, timings = self.fc.prepare(rnd, {h: (lambda h=h: self.update(rnd, h)) for h in self.hospitals})
        plan = self.plan(variant, rnd, target, subs, payloads, rng)
        res = self.fc.process(rnd, plan.arrivals, timings)
        accepted = {e["arrival"] for e in res.entries}
        leaked = [i for i in plan.adversarial if i in accepted]
        held, post = not leaked, None
        if plan.post is not None:
            ok, post = plan.post(res)
            held = held and ok
        clean_ok, clean_reason = audit_commitment(res.commitment, res.record["root"], res.record["leaf_count"],
                                                  self.defences)
        for h in self.hospitals:
            self.history[h].append((rnd, subs[h], payloads[h]))
        return {"round": rnd, "attack": self.attack, "variant": variant, "target": target, "held": bool(held),
                "note": plan.note, "adversarial_arrivals": plan.adversarial, "adversarial_accepted": leaked,
                "audit": post, "accepted": [e["leaf_id"] for e in res.entries],
                "rejected": res.record["rejected"], "equivocations": res.equivocations,
                "honest_record_audit": "ok" if clean_ok else clean_reason,
                "root": res.record["root"], "txid": res.record["txid"]}


def simulate(attack: str, rounds: int = 100, seed: int = 0, dataset: str = "heart", backend: str = "mock",
             defence_off: bool = False, out_dir: Optional[str] = "fedverify/results",
             data_path: Optional[str] = None) -> dict:
    """Run `rounds` attacked rounds; write config.json, rounds.jsonl and summary.json (unless out_dir is None)."""
    defences = defences_for(attack, defence_off)
    sim = Simulation(attack, seed, dataset, backend, defences, data_path)
    recs = [sim.run_round(r) for r in range(1, int(rounds) + 1)]
    held = sum(r["held"] for r in recs)
    by_variant = {}
    for v in ATTACKS[attack]:
        rs = [r for r in recs if r["variant"] == v]
        by_variant[v] = {"rounds": len(rs), "held": sum(r["held"] for r in rs),
                         "resilience_pct": (100.0 * sum(r["held"] for r in rs) / len(rs)) if rs else None}
    s = 100.0 * held / len(recs)
    summary = {"attack": attack, "metric": METRIC[attack], "rounds": len(recs), "held": held,
               "resilience_pct": s, "attack_success_pct": 100.0 - s, "by_variant": by_variant,
               "defence_off": bool(defence_off), "defences_switched_off": defences.off(), "seed": int(seed),
               "dataset": dataset, "definition": DEFINITION}
    config = {"attack": attack, "metric": METRIC[attack], "variants": list(ATTACKS[attack]),
              "variant_rule": ("each round draws one variant and one target hospital uniformly from "
                               "derive_seed(seed, attack, 'round', r); replay variants need an earlier round, so "
                               "round 1 draws among the others"),
              "rounds": int(rounds), "seed": int(seed), "dataset": dataset, "hospitals": sim.hospitals,
              "num_params": sim.numel, "defences": dataclasses.asdict(defences),
              "defences_for_this_attack": list(DEFENCES_FOR[attack]), "defences_switched_off": defences.off(),
              "chain": sim.session.describe(), "run_id": sim.run_id,
              "honest_payloads": ("serialized logistic updates (survival/serialize.py) of the dataset's model size, "
                                  "values drawn from the seed; no check reads the values"),
              "held_rule": ("held = no adversarial submission accepted, and for leaf_order_swapped the audit of the "
                            "tampered record against the anchored root failed"),
              "definition": DEFINITION}
    if out_dir is not None:
        name = attack + ("_defence_off" if defence_off else "") + ("" if dataset == "heart" else f"_{dataset}")
        d = Path(out_dir) / "chain_attacks" / name / f"seed{int(seed)}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        with open(d / "rounds.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (d / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summary["out"] = str(d)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fedverify.survival.chain_attacks", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", default="all", choices=["all", *ATTACKS])
    ap.add_argument("--rounds", type=int, default=100, help="simulated rounds per attack (Revision 2: at least 100)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="heart", choices=list(DATASETS), help="whose hospitals and model size")
    ap.add_argument("--chain-backend", default="mock", choices=["mock", "local"])
    ap.add_argument("--defence-off", action="store_true", help="switch off the defences that stop the attack")
    ap.add_argument("--out-dir", default="fedverify/results")
    a = ap.parse_args(argv)
    if a.rounds < 1:
        ap.error("--rounds must be >= 1")
    for attack in (ATTACKS if a.attack == "all" else [a.attack]):
        s = simulate(attack, a.rounds, a.seed, a.dataset, a.chain_backend, a.defence_off, a.out_dir)
        print(f"[{attack}{' defence OFF' if a.defence_off else ''}] {s['metric']} = {s['resilience_pct']:.1f}%  "
              f"attack success = {s['attack_success_pct']:.1f}%  over {s['rounds']} rounds  "
              + "  ".join(f"{v}:{b['held']}/{b['rounds']}" for v, b in s["by_variant"].items())
              + f"  -> {s['out']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
