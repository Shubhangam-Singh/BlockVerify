"""Aggregation rules. Phase 1's FedAvg plus the Phase-4 robust baselines and Forensics."""
from .base import Aggregator, stack_deltas, weighted_mean
from .fedavg import FedAvg
from .forensics import Forensics
from .krum import Krum, MultiKrum
from .median import CoordinateMedian
from .trimmed_mean import TrimmedMean

AGGREGATORS = {
    "fedavg": FedAvg,
    "krum": Krum,
    "multikrum": MultiKrum,
    "trimmed_mean": TrimmedMean,
    "median": CoordinateMedian,
    "forensics": Forensics,
}


def build_aggregator(name: str) -> Aggregator:
    if name not in AGGREGATORS:
        raise ValueError(f"unknown aggregator {name!r}; have {sorted(AGGREGATORS)}")
    return AGGREGATORS[name]()


__all__ = ["Aggregator", "AGGREGATORS", "build_aggregator", "FedAvg", "Forensics",
           "Krum", "MultiKrum", "TrimmedMean", "CoordinateMedian",
           "stack_deltas", "weighted_mean"]
