"""Per-round commitment for the chain_* methods (Revision 2 §3.3-3.4).

    client     serialize -> digest (SHA-256 of the payload) -> CID -> sign {run_id, round, hospital_id, cid, digest,
               timestamp, txn_id} with the hospital's Ed25519 TEST key
    aggregator verify (registered key, signature, txn_id counter, digest and CID against the payload bytes)
               -> Merkle leaves [i, hospital_id, digest] (unchanged format) -> round_merkle_root -> anchor
               -> leaves + inclusion proofs stored; the update is rebuilt FROM THE COMMITTED BYTES for aggregation

Before every round after the first, each hospital checks the previous block (the ledger is valid and still holds the
previous round's root under its txid). A hospital whose check fails skips the round, and the aggregator records the
skipped round with the reason: nothing is silently continued (mirrors the reference paper's Algorithm 1, steps 8-11).

The chain is the EXISTING proof-of-work chain (backend/blockchain.py) through the EXISTING ChainAnchor; 'mock' is the
default backend and 'local' mines real blocks. MeteredAnchor only adds measurements: per block its index, number of
transactions, size in bytes and creation (mining) time, which feed Figs 7 and 8. Timestamps inside submissions are a
logical clock (base + round), so a run stays byte-reproducible; wall-clock timings are logged separately.

Defences against the reference paper's three chain attacks (Revision 2 §4A; simulated in chain_attacks.py):
  double spending        one digest per (run, round, hospital): a second, different digest is rejected and flagged as
                         equivocation, the same one again is a duplicate; the submission AND the payload header must name
                         the current round and the submitting hospital, so an old update cannot be spent again later;
                         the txn_id must hash an unused counter
  transaction malleability  the signature covers every field (timestamp and txn_id included); the Merkle leaf binds
                         its position, so a stored round record cannot be reordered without changing the anchored root
  endpoint compromise    the signature must verify under the key REGISTERED for the claimed hospital (write-once)
Each check can be switched off (Defences) only so the tests can show the attack it stops then succeeds. Every run
uses the default: all on.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..chain.anchor import ChainAnchor, _mock_txid
from ..chain.commitment import inclusion_proof, leaf_hashes, round_merkle_root, verify_proof, warm_up
from ..chain.contentid import LABEL as CID_LABEL
from ..chain.contentid import content_id, ipfs_add, verify_cid
from ..chain.signing import (PAYLOAD_FIELDS, SIGNED_FIELDS, KeyRegistry, derive_test_key, describe_registry,
                             make_txn_id, public_key_bytes, sign_submission)
from .serialize import deserialize, split_payload

LOGICAL_TS_BASE = 1_700_000_000          # submission timestamps: base + round (a logical clock; see module docstring)


def block_bytes(block: Any) -> int:
    """Size of a block as the chain hashes it (index, timestamp, transactions, previous_hash, nonce) plus its hash."""
    d = {"index": block.index, "timestamp": block.timestamp, "transactions": block.transactions,
         "previous_hash": block.previous_hash, "nonce": block.nonce, "hash": block.hash}
    return len(json.dumps(d, sort_keys=True).encode("utf-8"))


class MeteredAnchor(ChainAnchor):
    """ChainAnchor that can share one chain across fold-runs and measures every block it mines."""

    def __init__(self, backend: str = "mock", checkpoint_every: int = 0, difficulty: int = 2,
                 shared_chain: Any = None, block_log: Optional[list] = None, clock0: Optional[float] = None):
        super().__init__(backend, checkpoint_every=checkpoint_every, difficulty=difficulty)
        if shared_chain is not None:
            self._chain = shared_chain
        self.block_log = block_log if block_log is not None else []
        self.clock0 = time.perf_counter() if clock0 is None else clock0

    def _anchor_local(self, run_id, round_num, root, leaf_count, leaves, payload):
        t0 = time.perf_counter()
        out = super()._anchor_local(run_id, round_num, root, leaf_count, leaves, payload)
        t1 = time.perf_counter()
        b = self.chain.chain[-1]
        self.block_log.append({"block_index": int(b.index), "run_id": str(run_id), "round": int(round_num),
                               "transactions": len(b.transactions), "block_bytes": block_bytes(b), "hash": b.hash,
                               "creation_wall_s": t1 - t0, "created_at_wall_s": t1 - self.clock0})
        return out

    def verify_previous(self, run_id: str, round_num: int, root: str, leaf_count: int, txid: str) -> tuple[bool, Optional[str]]:
        """Is the previous round still on the ledger as committed? (True, None) or (False, reason)."""
        if self.backend == "mock":
            if self.round_roots[round_num - 1:round_num] != [root]:
                return False, f"round {round_num} root is not in the lineage"
            if _mock_txid(run_id, round_num, root, leaf_count) != txid:
                return False, f"round {round_num} txid does not match its root"
            return True, None
        check = self.chain.is_chain_valid()
        if not check["valid"]:
            return False, "previous block verification failed: " + "; ".join(check["errors"][:3])
        block = next((b for b in reversed(self.chain.chain) if b.hash == txid), None)
        if block is None:
            return False, f"round {round_num} block {txid[:16]} is not on the chain"
        if not any(t.get("type") == "fl_round_commit" and t.get("run_id") == str(run_id)
                   and t.get("round") == int(round_num) and t.get("root") == root for t in block.transactions):
            return False, f"round {round_num} block does not hold that round's root"
        return True, None


@dataclass(frozen=True)
class Defences:
    """The commit layer's checks. All on in every run; one is switched off only in a test (or a --defence-off
    simulation) to show that the attack it stops then succeeds."""
    one_digest_per_slot: bool = True     # double spending: one digest per (run, round, hospital)
    round_binding: bool = True           # the submission and the payload header must name this round and hospital
    fresh_txn_counter: bool = True       # the txn_id must hash an unused counter of the hospital
    full_signature: bool = True          # the signature covers every field (off: run_id, hospital_id, cid, digest only)
    registered_keys: bool = True         # verify under the key REGISTERED for the claimed hospital (off: a carried key)
    merkle_position_audit: bool = True   # the root binds each leaf's position (off: an order-free root over sorted leaves)

    def off(self) -> list[str]:
        return [k for k, v in dataclasses.asdict(self).items() if not v]


ALL_DEFENCES = Defences()


@dataclass
class CommitResult:
    accepted: list                      # hospitals whose submission verified, in arrival (= hospital) order
    models: dict                        # hospital -> model rebuilt from the committed bytes
    record: dict                        # the round record's "commit" block
    commitment: dict                    # one commitments.jsonl line (leaves and inclusion proofs)
    entries: list = field(default_factory=list)        # accepted arrivals: {arrival, hospital, leaf_id, digest}
    equivocations: list = field(default_factory=list)  # {hospital, round, digests}: a second digest for one slot


class ChainSession:
    """One per run: test keys (registered with the verifier), and for the 'local' backend one shared PoW chain."""

    def __init__(self, cfg: Any, run_seed: int, defences: Defences = ALL_DEFENCES):
        self.cfg = cfg
        self.defences = defences
        self.backend = cfg.chain_backend
        self.ipfs = cfg.ipfs
        self.difficulty = int(cfg.chain_difficulty)
        self.block_log: list = []
        self.clock0 = time.perf_counter()
        self.registry = KeyRegistry()
        self.keys = {h: derive_test_key(run_seed, h) for h in cfg.hospitals}
        # A hospital's transaction counter belongs to its key, which lives for the whole run (every fold-run), just
        # like the verifier's expected counter in the registry; restarting it per fold would look like a replay.
        self.client_counter = {h: 0 for h in self.keys}
        for h, k in self.keys.items():
            self.registry.register(h, public_key_bytes(k))
        warm_up()                                   # pay the backend import before any timed region
        self.chain = None
        if self.backend == "local":
            probe = ChainAnchor("local", checkpoint_every=0, difficulty=self.difficulty)
            self.chain = probe.chain                # the EXISTING Blockchain, shared by every fold-run of the run
            for h in cfg.hospitals:                 # the registered keys go on the ledger with the first block
                self.chain.add_transaction({"type": "fl_key_register", "hospital_id": h,
                                            "public_key": self.registry.public_key(h).hex(), "label": "TEST KEY"})

    def describe(self) -> dict:
        return {"backend": self.backend, "consensus": "pow", "difficulty": self.difficulty, "ipfs": self.ipfs,
                "content_id": CID_LABEL, "keys": describe_registry(self.registry),
                "logical_timestamp_base": LOGICAL_TS_BASE}

    def for_fold(self, run_id: str) -> "FoldCommitter":
        anchor = MeteredAnchor(self.backend, checkpoint_every=0, difficulty=self.difficulty,
                               shared_chain=self.chain, block_log=self.block_log, clock0=self.clock0)
        return FoldCommitter(self, run_id, anchor, self.defences)

    def chain_metrics(self) -> Optional[dict]:
        """Figs 7-8, defined as in the reference paper (footnote in the tables): latency = mean time between
        consecutive blocks (ms); throughput = total block bytes / total elapsed seconds; send rate = transactions
        per minute. None for the mock backend, which mines nothing."""
        blocks = self.block_log
        if not blocks:
            return None
        ts = [b["created_at_wall_s"] for b in blocks]
        elapsed = max(ts[-1], 1e-12)
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        return {"blocks": len(blocks), "difficulty": self.difficulty,
                "latency_ms": 1000.0 * sum(gaps) / len(gaps) if gaps else None,
                "throughput_bytes_per_s": sum(b["block_bytes"] for b in blocks) / elapsed,
                "send_rate_tx_per_min": sum(b["transactions"] for b in blocks) / (elapsed / 60.0),
                "mean_block_creation_s": sum(b["creation_wall_s"] for b in blocks) / len(blocks),
                "elapsed_wall_s": elapsed}


TIMING_KEYS = ("serialize_ms", "digest_ms", "cid_ms", "sign_ms", "verify_ms", "merkle_ms", "anchor_ms")


def audit_commitment(commitment: dict, root: Optional[str], leaf_count: int,
                     defences: Defences = ALL_DEFENCES) -> tuple[bool, Optional[str]]:
    """An auditor's check of a STORED round record against the root anchored on the ledger: (ok, reason)."""
    leaves = commitment.get("leaves") or []
    entries = [(str(lf["client_id"]), str(lf["digest"])) for lf in leaves]
    if len(entries) != int(leaf_count):
        return False, f"{len(entries)} stored leaves, {leaf_count} anchored"
    if not entries:
        return root is None, (None if root is None else "no stored leaves under an anchored root")
    if not defences.merkle_position_audit:              # defence off: the root was taken over the sorted leaves
        rebuilt, _ = round_merkle_root(sorted(entries))
        return (rebuilt == root), (None if rebuilt == root else "root rebuilt from the stored leaves differs")
    hashes = leaf_hashes(entries)
    for p, lf in enumerate(leaves):
        if int(lf.get("index", -1)) != p:
            return False, f"leaf stored at position {p} claims index {lf.get('index')}"
        if lf.get("leaf") != hashes[p] or not verify_proof(hashes[p], lf.get("proof") or [], root):
            return False, f"leaf {p} ({lf['client_id']}) does not prove into the anchored root"
    rebuilt, _ = round_merkle_root(entries)
    return (rebuilt == root), (None if rebuilt == root else "root rebuilt from the stored leaves differs")


class FoldCommitter:
    """Commits the rounds of one fold-run (its own lineage) on the session's ledger."""

    def __init__(self, session: ChainSession, run_id: str, anchor: MeteredAnchor,
                 defences: Defences = ALL_DEFENCES):
        self.s = session
        self.run_id = run_id
        self.anchor = anchor
        self.defences = defences
        self.last: Optional[dict] = None                      # previous round: round, root, leaf_count, txid
        self.slots: dict[tuple, str] = {}                     # (run_id, round, hospital_id) -> the digest accepted

    @property
    def signed_fields(self) -> tuple[str, ...]:
        return SIGNED_FIELDS if self.defences.full_signature else PAYLOAD_FIELDS

    def check_previous(self) -> dict[str, Optional[str]]:
        """Every hospital verifies the previous block before training; {hospital: reason} for those that fail."""
        if self.last is None:
            return {}
        ok, reason = self.anchor.verify_previous(self.run_id, self.last["round"], self.last["root"],
                                                 self.last["leaf_count"], self.last["txid"])
        return {} if ok else {h: reason for h in self.s.keys}

    def submit(self, rnd: int, hospital: str, payload: bytes) -> tuple[dict, dict]:
        """Client side: digest, CID, signed submission. Returns (submission, timings_ms)."""
        t0 = time.perf_counter()
        digest = hashlib.sha256(payload).hexdigest()
        t1 = time.perf_counter()
        cid = content_id(payload)
        t2 = time.perf_counter()
        ts = LOGICAL_TS_BASE + int(rnd)
        fields = {"run_id": self.run_id, "round": int(rnd), "hospital_id": hospital, "cid": cid, "digest": digest,
                  "timestamp": ts, "txn_id": make_txn_id(hospital, self.s.client_counter[hospital], ts)}
        sub = sign_submission(self.s.keys[hospital], fields, self.signed_fields)
        if not self.defences.registered_keys:                 # defence-off tests: the verifier trusts this
            sub["public_key"] = public_key_bytes(self.s.keys[hospital]).hex()
        t3 = time.perf_counter()
        self.s.client_counter[hospital] += 1
        return sub, {"digest_ms": (t1 - t0) * 1e3, "cid_ms": (t2 - t1) * 1e3, "sign_ms": (t3 - t2) * 1e3}

    def verify(self, sub: dict, payload: bytes, rnd: Optional[int] = None) -> tuple[bool, Optional[str]]:
        """Aggregator side: the signed fields AND the bytes they describe. With `rnd`, both must name that round."""
        d = self.defences
        if sub.get("run_id") != self.run_id:
            return False, "submission belongs to another run"
        if rnd is not None and d.round_binding and sub.get("round") != int(rnd):
            return False, f"replay: the submission is signed for round {sub.get('round')}, this is round {rnd}"
        if hashlib.sha256(payload).hexdigest() != sub.get("digest"):
            return False, "payload does not match the signed digest"
        if not verify_cid(payload, str(sub.get("cid"))):
            return False, "payload does not match the signed CID"
        if rnd is not None and d.round_binding:
            try:
                header = split_payload(payload)[0]           # ValueError covers bad JSON and bad UTF-8 too
            except (ValueError, struct.error):
                header = None
            if not isinstance(header, dict):
                return False, "payload is not a FedVerify update"
            if header.get("round") != int(rnd) or header.get("hospital") != sub.get("hospital_id"):
                return False, (f"payload header names round {header.get('round')} and hospital {header.get('hospital')!r}"
                               f": not this round's update of {sub.get('hospital_id')!r}")
        return self.s.registry.verify(sub, signed=self.signed_fields, check_counter=d.fresh_txn_counter,
                                      registered_keys=d.registered_keys)

    def prepare(self, rnd: int, builders: dict[str, Callable[[], bytes]]) -> tuple[dict, dict, dict]:
        """Client side of a round: (submissions, payloads, timings), keyed by hospital in builder order."""
        timings = {k: 0.0 for k in TIMING_KEYS}
        subs, payloads = {}, {}
        for h, build in builders.items():
            t0 = time.perf_counter()
            payloads[h] = build()
            timings["serialize_ms"] += (time.perf_counter() - t0) * 1e3
            subs[h], t = self.submit(rnd, h, payloads[h])
            for k, v in t.items():
                timings[k] += v
        return subs, payloads, timings

    def process(self, rnd: int, arrivals: list, timings: Optional[dict] = None) -> CommitResult:
        """Aggregator side of a round over the (submission, payload) pairs in the order they arrived: verify, keep one
        digest per (run, round, hospital), Merkle, anchor, rebuild the accepted models from their bytes."""
        timings = dict(timings) if timings else {k: 0.0 for k in TIMING_KEYS}
        d = self.defences
        accepted, rejected, models, entries, equivocations = [], [], {}, [], []
        t0 = time.perf_counter()
        for i, (sub, payload) in enumerate(arrivals):
            h = sub.get("hospital_id")
            ok, reason = self.verify(sub, payload, rnd)
            if ok and d.one_digest_per_slot:
                slot = (sub.get("run_id"), sub.get("round"), h)
                held = self.slots.get(slot)
                if held is not None and held != sub["digest"]:
                    ok, reason = False, (f"equivocation: a second, different digest for ({sub.get('run_id')}, "
                                         f"round {sub.get('round')}, {h})")
                    equivocations.append({"hospital": h, "round": sub.get("round"), "digests": [held, sub["digest"]]})
                elif held is not None:
                    ok, reason = False, "duplicate: this update was already accepted for its round"
                else:
                    self.slots[slot] = sub["digest"]
            if ok:
                k = sum(1 for e in entries if e["hospital"] == h)   # > 0 only with one_digest_per_slot off
                entries.append({"arrival": i, "hospital": h, "leaf_id": h if k == 0 else f"{h}#{k + 1}",
                                "digest": sub["digest"]})
                accepted.append(h)
            else:
                rejected.append({"hospital": h, "reason": reason})
        timings["verify_ms"] = (time.perf_counter() - t0) * 1e3

        pairs = [(e["leaf_id"], e["digest"]) for e in entries]
        if not d.merkle_position_audit:                       # defence off: an order-free commitment
            pairs = sorted(pairs)
        t0 = time.perf_counter()
        root, leaf_count = round_merkle_root(pairs) if pairs else (None, 0)
        lh = leaf_hashes(pairs) if pairs else []
        proofs = [inclusion_proof(pairs, i) for i in range(len(pairs))]
        timings["merkle_ms"] = (time.perf_counter() - t0) * 1e3
        leaves = [{"index": i, "client_id": c, "digest": dg, "leaf": lh[i], "proof": proofs[i]}
                  for i, (c, dg) in enumerate(pairs)]

        an = {"txid": None, "backend": self.anchor.backend, "fee": 0, "bytes_written": 0, "latency_ms": 0.0}
        if pairs:
            an = self.anchor.anchor_round(self.run_id, rnd, root, leaf_count, leaves)
            self.last = {"round": int(rnd), "root": root, "leaf_count": leaf_count, "txid": an["txid"]}
        timings["anchor_ms"] = an["latency_ms"]
        for e in entries:                                     # the server uses the committed bytes, nothing else
            models[e["hospital"]] = deserialize(arrivals[e["arrival"]][1])[1]

        record = {"root": root, "leaf_count": leaf_count, "txid": an["txid"], "backend": an["backend"],
                  "bytes_committed": int(sum(len(arrivals[e["arrival"]][1]) for e in entries)), "fee": an["fee"],
                  "bytes_written": an["bytes_written"],
                  "submissions": [{k: sub.get(k) for k in ("hospital_id", "cid", "digest", "timestamp", "txn_id")}
                                  for sub, _ in arrivals],
                  "rejected": rejected, "skipped": [], **timings}
        if equivocations:                                     # honest rounds keep the Phase-3 record shape
            record["equivocations"] = equivocations
        commitment = {"run_id": self.run_id, "round": int(rnd), "root": root, "leaf_count": leaf_count,
                      "txid": an["txid"], "backend": an["backend"], "leaves": leaves, "checkpoint": None,
                      "submissions": [sub for sub, _ in arrivals]}
        return CommitResult(accepted, models, record, commitment, entries, equivocations)

    def commit_round(self, rnd: int, builders: dict[str, Callable[[], bytes]],
                     tamper: Optional[Callable[[str, dict, bytes], tuple[dict, bytes]]] = None) -> CommitResult:
        """Run the whole pipeline for one round. `builders` maps hospital -> () -> payload bytes, in hospital order.
        `tamper` (tests only) may alter a submission or its bytes in transit."""
        subs, payloads, timings = self.prepare(rnd, builders)
        arrivals = []
        for h in builders:
            sub, payload = tamper(h, subs[h], payloads[h]) if tamper is not None else (subs[h], payloads[h])
            if self.s.ipfs == "real":
                sub["_ipfs"] = ipfs_add(payload)              # optional; never required, never fatal
            arrivals.append((sub, payload))
        return self.process(rnd, arrivals, timings)


def skipped_record(reasons: dict[str, Optional[str]]) -> dict:
    """The commit block of a round that hospitals skipped because the previous block did not verify."""
    return {"root": None, "leaf_count": 0, "txid": None, "backend": None, "bytes_committed": 0, "fee": 0,
            "bytes_written": 0, "submissions": [], "rejected": [],
            "skipped": [{"hospital": h, "reason": r} for h, r in reasons.items()],
            "serialize_ms": 0.0, "digest_ms": 0.0, "cid_ms": 0.0, "sign_ms": 0.0, "verify_ms": 0.0, "merkle_ms": 0.0,
            "anchor_ms": 0.0}
