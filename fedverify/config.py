"""FedVerify run configuration.

A single frozen dataclass carries every parameter that can influence a run, so
``config.json`` written next to the results is a complete, replayable record
(CLAUDE.md, "Determinism").
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict, field, replace
from typing import Optional


def _fmt_alpha(a: float) -> str:
    return "iid" if math.isinf(a) else f"a{a:g}"


@dataclass(frozen=True)
class FLConfig:
    # ── core ────────────────────────────────────────────────────────────────
    dataset: str = "mnist"                  # mnist | fmnist | mitbih
    num_clients: int = 10
    rounds: int = 30
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    alpha: float = 0.5                      # Dirichlet concentration; inf == IID
    client_fraction: float = 1.0
    aggregator: str = "fedavg"
    seed: int = 0
    device: str = "cpu"
    torch_threads: Optional[int] = None   # pin intra-op threads; None = leave as-is
    out_dir: str = "fedverify/results"
    run_id: Optional[str] = None            # auto: <exp>_<cell>_<seed>

    # identity of the experiment cell (used to build run_id / result paths)
    exp: str = "adhoc"
    cell: Optional[str] = None              # auto-derived from the knobs below

    # ── placeholders for later phases (inert in Phase 1) ────────────────────
    epsilon: Optional[float] = None         # Phase 2 (DP); None/inf == no DP
    delta: float = 1e-5                     # Phase 2
    max_grad_norm: float = 1.0              # Phase 2
    attack: Optional[str] = None            # Phase 4: none|label_flip|sign_flip|gaussian|zero|scaling|backdoor
    attacker_frac: float = 0.0              # Phase 4: fraction of clients that are malicious
    attack_start_round: int = 0             # Phase 4: attacks dormant before this round
    sign_flip_scale: float = 1.0            # Phase 4: delta -> -s*delta
    gaussian_sigma: float = 1.0             # Phase 4: delta -> N(0, sigma^2)
    poison_frac: float = 0.5                # Phase 4: share of an attacker's batch triggered
    backdoor_target: int = 0                # Phase 4: label the trigger maps to
    trim_beta: float = 0.2                  # Phase 4: trimmed-mean trim fraction per end
    commit: bool = False                    # Phase 3: commit round roots to a ledger
    chain_backend: str = "mock"             # Phase 3: mock | local | algorand
    checkpoint_every: int = 10              # Phase 3: BTC checkpoint cadence (0 = off)
    btc_checkpoint: bool = False            # Phase 3: real OP_RETURN (else mock)
    tau: Optional[float] = None             # Phase 4: forensics threshold, FROM CALIBRATION
    tau_coord: float = 3.0                  # Phase 4: per-coordinate z inside s_coord
    tau_source: Optional[str] = None        # Phase 4: provenance of tau (path + key)

    def __post_init__(self):
        if self.cell is None:
            eps = "inf" if self.epsilon in (None, float("inf")) else f"{self.epsilon:g}"
            # MIT-BIH partitions by patient, so alpha does not influence it; naming the
            # cell "a0.5" would imply a Dirichlet draw that never happened.
            split = "patient" if self.dataset == "mitbih" else _fmt_alpha(self.alpha)
            cell = f"{self.dataset}_K{self.num_clients}_{split}_eps{eps}"
            if self.aggregator != "fedavg":
                cell += f"_{self.aggregator}"
            if self.attack:
                cell += f"_{self.attack}{self.attacker_frac:g}"
            object.__setattr__(self, "cell", cell)
        if self.run_id is None:
            object.__setattr__(self, "run_id", f"{self.exp}_{self.cell}_s{self.seed}")

    # ── helpers ─────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """JSON-safe dict of EVERY parameter that influenced the run."""
        d = asdict(self)
        for k, v in d.items():                      # inf is not valid JSON
            if isinstance(v, float) and math.isinf(v):
                d[k] = "inf"
        return d

    def replace(self, **kw) -> "FLConfig":
        # cell/run_id must be recomputed unless explicitly overridden
        kw.setdefault("cell", None)
        kw.setdefault("run_id", None)
        return replace(self, **kw)

    @property
    def run_dir(self) -> str:
        import os
        return os.path.join(self.out_dir, self.exp, self.cell, f"seed{self.seed}")

    # ── CLI ─────────────────────────────────────────────────────────────────
    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        g = p.add_argument_group("FLConfig")
        g.add_argument("--dataset", default="mnist", choices=["mnist", "fmnist", "mitbih"])
        g.add_argument("--num-clients", type=int, default=10)
        g.add_argument("--rounds", type=int, default=30)
        g.add_argument("--local-epochs", type=int, default=1)
        g.add_argument("--batch-size", type=int, default=64)
        g.add_argument("--lr", type=float, default=0.01)
        g.add_argument("--momentum", type=float, default=0.9)
        g.add_argument("--alpha", type=float, default=0.5,
                       help="Dirichlet concentration; use inf for IID")
        g.add_argument("--client-fraction", type=float, default=1.0)
        g.add_argument("--aggregator", default="fedavg")
        g.add_argument("--seed", type=int, default=0)
        g.add_argument("--device", default="cpu")
        g.add_argument("--torch-threads", type=int, default=None,
                       help="pin torch intra-op threads; results are only "
                            "byte-reproducible at a FIXED thread count")
        g.add_argument("--out-dir", default="fedverify/results")
        g.add_argument("--run-id", default=None)
        g.add_argument("--exp", default="adhoc")
        g.add_argument("--cell", default=None)
        # later-phase knobs, accepted now so scripts stay stable
        g.add_argument("--epsilon", type=float, default=None)
        g.add_argument("--delta", type=float, default=1e-5)
        g.add_argument("--max-grad-norm", type=float, default=1.0)
        g.add_argument("--attack", default=None)
        g.add_argument("--attacker-frac", type=float, default=0.0)
        g.add_argument("--attack-start-round", type=int, default=0)
        g.add_argument("--sign-flip-scale", type=float, default=1.0)
        g.add_argument("--gaussian-sigma", type=float, default=1.0)
        g.add_argument("--poison-frac", type=float, default=0.5)
        g.add_argument("--backdoor-target", type=int, default=0)
        g.add_argument("--trim-beta", type=float, default=0.2)
        g.add_argument("--commit", action="store_true",
                       help="commit each round's Merkle root to the chain backend")
        g.add_argument("--chain-backend", default="mock",
                       choices=["mock", "local", "algorand"])
        g.add_argument("--checkpoint-every", type=int, default=10)
        g.add_argument("--btc-checkpoint", action="store_true",
                       help="use the real Bitcoin OP_RETURN path for checkpoints")
        g.add_argument("--tau", type=float, default=None,
                       help="forensics threshold; normally supplied from taus.json")
        g.add_argument("--tau-coord", type=float, default=3.0)
        g.add_argument("--tau-source", default=None)
        # Phase-1 convenience (not part of the config record)
        g.add_argument("--limit-train", type=int, default=None,
                       help="subsample the training set (smoke tests)")
        g.add_argument("--limit-test", type=int, default=None)
        return p

    @classmethod
    def from_args(cls, argv=None) -> tuple["FLConfig", argparse.Namespace]:
        p = cls.add_args(argparse.ArgumentParser(prog="fedverify"))
        ns = p.parse_args(argv)
        known = {f for f in cls.__dataclass_fields__}
        cfg = cls(**{k: v for k, v in vars(ns).items() if k in known})
        return cfg, ns
