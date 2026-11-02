"""Canonical bytes for one hospital's update, for every family (Revision 2 §3.1). No pickle anywhere.

    payload = MAGIC || uint32-LE header length || header || body
    header  = compact, sorted-key JSON: {family, run_id, round, hospital, num_samples, ...family fields}

  lr   body = canon_update(delta), the EXISTING float32 little-endian serializer (chain/commitment.py)
  rf   body = for each tree in order: children_left, children_right, feature (int32 LE),
              threshold, value (float64 LE); header adds trees = [[node_count, n_classes], ...] and death_column
  xgb  body = for each tree in order: uint32-LE length || the tree's JSON dump (UTF-8) from
              booster.get_dump(dump_format="json"); header adds trees = count and objective

The header binds the update to its run, round and hospital, so its digest cannot be replayed into another round.
deserialize() rebuilds a small numpy model from the bytes alone, and the chain_* methods aggregate FROM those
rebuilt models, so what is committed is what is used. The numpy predictors reproduce sklearn's predict_proba and
XGBoost's margins bit for bit (tested).
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from ..chain.commitment import canon_update

MAGIC = b"FVSU1"


def _pack(header: dict, body: bytes) -> bytes:
    h = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return MAGIC + struct.pack("<I", len(h)) + h + body


def split_payload(payload: bytes) -> tuple[dict, bytes]:
    if payload[:len(MAGIC)] != MAGIC:
        raise ValueError("not a FedVerify update payload")
    (n,) = struct.unpack_from("<I", payload, len(MAGIC))
    start = len(MAGIC) + 4
    return json.loads(payload[start:start + n].decode("utf-8")), payload[start + n:]


def _base(family: str, run_id: str, rnd: int, hospital: str, num_samples: int) -> dict:
    return {"family": family, "run_id": str(run_id), "round": int(rnd), "hospital": str(hospital),
            "num_samples": int(num_samples), "format": 1}


# ── logistic: a flat float32 delta ───────────────────────────────────────────
def serialize_lr(run_id: str, rnd: int, hospital: str, num_samples: int, delta: Any) -> bytes:
    body = canon_update(delta)
    return _pack({**_base("lr", run_id, rnd, hospital, num_samples), "numel": len(body) // 4}, body)


# ── random forest: per-tree little-endian arrays ─────────────────────────────
@dataclass(frozen=True)
class NumpyTree:
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray              # [node_count, n_classes] (sklearn >= 1.4 stores class fractions)

    def leaves(self, X: np.ndarray) -> np.ndarray:
        """sklearn's rule: go left when X[feature] <= threshold (float32 X compared as float64)."""
        Xd = np.asarray(X, dtype=np.float32).astype(np.float64)
        node = np.zeros(len(Xd), dtype=np.int64)
        rows = np.arange(len(Xd))
        while True:
            internal = self.children_left[node] != -1
            if not internal.any():
                return node
            go_left = Xd[rows, self.feature[node].clip(min=0)] <= self.threshold[node]
            node = np.where(internal, np.where(go_left, self.children_left[node], self.children_right[node]), node)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """The leaf's class fractions, exactly as sklearn >= 1.4 returns them (no renormalising division)."""
        return self.value[self.leaves(X)].copy()


@dataclass(frozen=True)
class NumpyForest:
    trees: list
    death_column: Optional[int]

    def probability_sum(self, X: np.ndarray) -> np.ndarray:
        """Sum over trees of P(event): the server's running sum for 'mean of tree probabilities'."""
        total = np.zeros(len(X), dtype=np.float64)
        if self.death_column is None or len(X) == 0:
            return total
        for t in self.trees:
            total += t.predict_proba(X)[:, self.death_column]
        return total


def serialize_rf(run_id: str, rnd: int, hospital: str, num_samples: int, forest: Any,
                 death_column: Optional[int]) -> bytes:
    parts, shapes = [], []
    for est in forest.estimators_:
        t = est.tree_
        value = np.asarray(t.value, dtype="<f8").reshape(t.node_count, -1)       # n_outputs == 1
        parts += [np.asarray(t.children_left, dtype="<i4").tobytes(), np.asarray(t.children_right, dtype="<i4").tobytes(),
                  np.asarray(t.feature, dtype="<i4").tobytes(), np.asarray(t.threshold, dtype="<f8").tobytes(),
                  value.tobytes()]
        shapes.append([int(t.node_count), int(value.shape[1])])
    header = {**_base("rf", run_id, rnd, hospital, num_samples), "trees": shapes, "death_column": death_column}
    return _pack(header, b"".join(parts))


def _deserialize_rf(header: dict, body: bytes) -> NumpyForest:
    trees, off = [], 0
    for n, c in header["trees"]:
        def take(dtype: str, count: int) -> np.ndarray:
            nonlocal off
            a = np.frombuffer(body, dtype=dtype, count=count, offset=off)
            off += a.nbytes
            return a
        left, right, feat = take("<i4", n), take("<i4", n), take("<i4", n)
        thr, val = take("<f8", n), take("<f8", n * c).reshape(n, c)
        trees.append(NumpyTree(left.astype(np.int64), right.astype(np.int64), feat.astype(np.int64),
                               thr.astype(np.float64), val.astype(np.float64)))
    if off != len(body):
        raise ValueError("random-forest payload has trailing or missing bytes")
    return NumpyForest(trees, header["death_column"])


# ── XGBoost: each tree's JSON dump, in order ─────────────────────────────────
@dataclass(frozen=True)
class NumpyBoostedTree:
    is_leaf: np.ndarray
    leaf: np.ndarray               # float32
    feature: np.ndarray
    condition: np.ndarray          # float32; XGBoost goes to "yes" when x < condition
    yes: np.ndarray
    no: np.ndarray
    missing: np.ndarray

    @classmethod
    def from_dump(cls, dump: str) -> "NumpyBoostedTree":
        nodes: dict[int, dict] = {}
        stack = [json.loads(dump)]
        while stack:
            n = stack.pop()
            nodes[int(n["nodeid"])] = n
            stack.extend(n.get("children", []))
        size = max(nodes) + 1
        is_leaf = np.ones(size, dtype=bool)
        leaf = np.zeros(size, dtype=np.float32)
        feature = np.zeros(size, dtype=np.int64)
        condition = np.zeros(size, dtype=np.float32)
        yes, no, missing = (np.zeros(size, dtype=np.int64) for _ in range(3))
        for i, n in nodes.items():
            if "leaf" in n:
                leaf[i] = np.float32(n["leaf"])
                continue
            is_leaf[i] = False
            feature[i] = int(str(n["split"]).lstrip("f"))
            condition[i] = np.float32(n["split_condition"])
            yes[i], no[i], missing[i] = int(n["yes"]), int(n["no"]), int(n["missing"])
        return cls(is_leaf, leaf, feature, condition, yes, no, missing)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        node = np.zeros(len(X), dtype=np.int64)
        rows = np.arange(len(X))
        while True:
            internal = ~self.is_leaf[node]
            if not internal.any():
                return self.leaf[node]
            x = X[rows, self.feature[node]]
            nxt = np.where(np.isnan(x), self.missing[node], np.where(x < self.condition[node], self.yes[node], self.no[node]))
            node = np.where(internal, nxt, node)


@dataclass(frozen=True)
class NumpyBooster:
    trees: list

    def tree_sum(self, X: np.ndarray) -> np.ndarray:
        """Sum of the trees' outputs, accumulated in float32 in tree order exactly like XGBoost's predictor."""
        out = np.zeros(len(X), dtype=np.float32)
        for t in self.trees:
            out = (out + t.predict(X)).astype(np.float32)
        return out.astype(np.float64)


def serialize_xgb(run_id: str, rnd: int, hospital: str, num_samples: int, booster: Any, objective: str) -> bytes:
    dumps = [d.encode("utf-8") for d in booster.get_dump(dump_format="json")]
    body = b"".join(struct.pack("<I", len(d)) + d for d in dumps)
    return _pack({**_base("xgb", run_id, rnd, hospital, num_samples), "trees": len(dumps), "objective": objective}, body)


def _deserialize_xgb(header: dict, body: bytes) -> NumpyBooster:
    trees, off = [], 0
    for _ in range(int(header["trees"])):
        (n,) = struct.unpack_from("<I", body, off)
        trees.append(NumpyBoostedTree.from_dump(body[off + 4:off + 4 + n].decode("utf-8")))
        off += 4 + n
    if off != len(body):
        raise ValueError("XGBoost payload has trailing or missing bytes")
    return NumpyBooster(trees)


# ── dispatch ─────────────────────────────────────────────────────────────────
def deserialize(payload: bytes) -> tuple[dict, Any]:
    """(header, model): a float32 torch delta (lr), a NumpyForest (rf) or a NumpyBooster (xgb)."""
    header, body = split_payload(payload)
    fam = header.get("family")
    if fam == "lr":
        delta = np.frombuffer(body, dtype="<f4").copy()
        if delta.size != int(header["numel"]):
            raise ValueError("logistic payload length does not match its header")
        return header, torch.from_numpy(delta.astype(np.float32))
    if fam == "rf":
        return header, _deserialize_rf(header, body)
    if fam == "xgb":
        return header, _deserialize_xgb(header, body)
    raise ValueError(f"unknown family {fam!r} in payload header")
