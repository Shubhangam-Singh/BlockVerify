"""FedVerify-Survival run configuration: the fixed, documented defaults of master-prompt §4 and Revision 2 §2.

Nothing here is tuned on test folds. Every field is written to config.json, so a results directory is a complete
record of what produced it (CLAUDE.md "Determinism"). Decisions taken in Phase 0 (INVENTORY.md §6) are marked Dn.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Optional

# The five TCGA source sites, in the prompt's order. Hospital CODES are the only ids written to disk or chain; the
# integer index is used only inside calls to existing FedVerify code, which casts ids with int() (D10).
HOSPITALS: tuple[str, ...] = ("BH", "A2", "E2", "A8", "AR")
HOSPITAL_INDEX: dict[str, int] = {h: i for i, h in enumerate(HOSPITALS)}
# The four UCI heart-disease hospitals (the CSV's `dataset` column), Revision 2 §2.1.
HEART_HOSPITALS: tuple[str, ...] = ("Cleveland", "Hungary", "VA Long Beach", "Switzerland")
DATASET_HOSPITALS: dict[str, tuple[str, ...]] = {"tcga": HOSPITALS, "heart": HEART_HOSPITALS}
# Switzerland has almost no negatives (1 of 46 rows in paper mode), so it gets no LOCAL-only model; it stays in
# federated training and in the pooled evaluation of every other method (Revision 2 §2.1).
LOCAL_EXCLUDE: dict[str, tuple[str, ...]] = {"tcga": (), "heart": ("Switzerland",)}
DATA_PATHS: dict[str, str] = {"tcga": "fedverify/data/tcga_brca/clinical.xlsx",
                              "heart": "fedverify/data/heart/heart_disease_uci.csv"}
EXPS: dict[str, str] = {"tcga": "survival", "heart": "heart"}       # results/<exp>/<cell>/seed<N>/
DATASETS: tuple[str, ...] = ("tcga", "heart")
TASKS: tuple[str, ...] = ("binary", "survival")      # survival = XGBoost AFT on TCGA only

LR_METHODS: tuple[str, ...] = ("local_lr", "central_lr", "fedsgd_lr", "fedavg_lr", "dp_fedavg_lr", "chain_fedavg_lr",
                               "ours_lr")
RF_METHODS: tuple[str, ...] = ("local_rf", "central_rf", "fed_rf", "chain_fed_rf", "ours_rf")
XGB_METHODS: tuple[str, ...] = ("local_xgb", "central_xgb", "fedxgb_bagging", "chain_fedxgb_bagging", "ours_xgb",
                                "central_xgb_steps")      # Phase 9: a diagnostic row (table.yaml diagnostic_rows)
FAMILY_METHODS: dict[str, tuple[str, ...]] = {"lr": LR_METHODS, "rf": RF_METHODS, "xgb": XGB_METHODS}
DP_METHODS: tuple[str, ...] = ("dp_fedavg_lr", "ours_lr")   # DP-SGD exists only for the logistic family (Phase 2)
DEFAULT_DP_EPSILON = 4.0                             # prompt §5: "dp_fedavg_lr (main epsilon default 4, configurable)"
# Phase 4 "ours": the commit pipeline plus screening (and DP for the logistic model, prompt §5: "ours_lr (FedAvg + DP +
# signed/CID commit + screening)"). Only these methods can be attacked or screened.
OURS_METHODS: tuple[str, ...] = ("ours_lr", "ours_rf", "ours_xgb")
LR_ATTACKS: tuple[str, ...] = ("sign_flip", "scaling", "label_flip")       # the existing attacks/byzantine.py rules
TREE_ATTACKS: tuple[str, ...] = ("label_flip_trees", "random_trees")       # new for RF and XGBoost (survival/poison.py)
FAMILY_ATTACKS: dict[str, tuple[str, ...]] = {"lr": LR_ATTACKS, "rf": TREE_ATTACKS, "xgb": TREE_ATTACKS}
DEFAULT_TAUS = "fedverify/results/calibration/taus_survival.json"         # written by analysis/calibrate_survival.py

# Which existing aggregator (or none) combines the hospitals' updates, recorded in config.json.
AGGREGATION: dict[str, str] = {
    "local_lr": "none", "central_lr": "none", "fedsgd_lr": "fedavg", "fedavg_lr": "fedavg", "dp_fedavg_lr": "fedavg",
    "local_rf": "none", "central_rf": "none", "fed_rf": "concat_trees",
    "local_xgb": "none", "central_xgb": "none", "fedxgb_bagging": "fedxgb_bagging", "central_xgb_steps": "none",
    # Phase 3: the same aggregation, fed from the committed bytes (commit only, Revision 2 §3)
    "chain_fedavg_lr": "fedavg", "chain_fed_rf": "concat_trees", "chain_fedxgb_bagging": "fedxgb_bagging",
    # Phase 4: the screen decides which hospitals' committed updates are used; the survivors are combined as above
    "ours_lr": "forensics", "ours_rf": "probe_screen+concat_trees", "ours_xgb": "probe_screen+fedxgb_bagging",
}
UNSCREENED_AGGREGATION: dict[str, str] = {"ours_lr": "fedavg", "ours_rf": "concat_trees", "ours_xgb": "fedxgb_bagging"}
CHAIN_BACKENDS: tuple[str, ...] = ("mock", "local")   # mock by default; local mines the existing PoW chain


def canonical_method(family: str, method: str) -> str:
    """Accept the table name ('fedavg_lr', 'fed_rf') or the short name ('fedavg', 'fed'); return the table name."""
    if family not in FAMILY_METHODS:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(FAMILY_METHODS)}")
    names = FAMILY_METHODS[family]
    if method in names:
        return method
    for full in names:
        if full == f"{method}_{family}":
            return full
    raise ValueError(f"method {method!r} is not in the {family!r} family; expected one of {list(names)}")


def is_infinite(epsilon: Optional[float]) -> bool:
    return epsilon is not None and math.isinf(float(epsilon))


def eps_tag(epsilon: Optional[float]) -> str:
    """'inf' for no DP (None or inf), else the compact number: 4.0 -> '4'."""
    return "inf" if epsilon is None or is_infinite(epsilon) else f"{float(epsilon):g}"


def derive_seed(*parts: object) -> int:
    """Stable 31-bit seed from any parts (Python's hash() is salted per process, so it cannot be used)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


@dataclass(frozen=True)
class SurvivalConfig:
    family: str = "lr"                       # lr | rf | xgb
    method: str = "fedavg_lr"                # table name, see FAMILY_METHODS
    seed: int = 0                            # BASE seed: repeat r uses seed + r (D7), so --seed 0 = seeds 0..4
    repeats: int = 5                         # repeated stratified CV (§4)
    folds: int = 5
    rounds: int = 30                         # every method is evaluated after each of these rounds
    hospitals: Optional[tuple[str, ...]] = None   # default: every hospital of the dataset
    include_race: bool = False               # §3 (TCGA only): off by default, see TODO.md for the fairness reasoning

    # dataset and task (Revision 2 §2)
    dataset: str = "tcga"                    # tcga | heart
    task: str = "binary"                     # binary (death indicator / heart disease) | survival (TCGA, XGBoost AFT)
    heart_mode: str = "paper"                # paper: 740 rows, ca/thal/slope dropped | full: 920 rows, imputed
    local_exclude: Optional[tuple[str, ...]] = None   # default per dataset: heart excludes Switzerland from local_*

    # logistic family (§4)
    local_epochs: int = 3                    # FedAvg, local_lr and central_lr: epochs per round
    batch_size: int = 32
    lr: float = 0.1                          # also the FedSGD step size (one full-batch step per round)
    momentum: float = 0.0                    # D12: the prompt gives none; plain SGD so "lr 0.1" means what it says
    class_weighted_loss: bool = True         # BCE with pos_weight = n_neg / n_pos of the training rows it sees

    # differential privacy, logistic family only (Phase 2). None or inf = no DP, Opacus never involved.
    epsilon: Optional[float] = None          # dp_fedavg_lr defaults to DEFAULT_DP_EPSILON
    delta: float = 1e-5
    max_grad_norm: float = 1.0               # per-example clipping bound C

    # random-forest family (§4)
    trees_per_hospital: int = 10             # per round; central_rf grows hospitals x this per round
    max_depth: int = 5
    min_samples_leaf: int = 3
    rf_class_weight: str = "balanced_subsample"
    rf_n_jobs: int = 1                       # rule 6: single-threaded trees, random_state from (seed, round, hospital)

    # XGBoost family (Revision 2 §2.2-2.3). Classification mirrors XGBClassifier(objective="binary:logistic",
    # tree_method="hist", n_jobs=1, max_depth=3, learning_rate=0.1, subsample=0.8).
    xgb_trees_per_round: int = 5             # N new trees per hospital per round (FedXGBBagging)
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_n_jobs: int = 1
    aft_distribution: str = "normal"         # survival:aft, aft_loss_distribution
    aft_scale: float = 1.0                   # aft_loss_distribution_scale
    aft_min_time_days: float = 1.0           # AFT needs time > 0: shorter times are clipped up to this (count reported)
    aft_init_days: float = 365.25            # the ensemble starts at margin ln(this): a fixed, data-independent value

    # chain commitment, chain_* methods only (Revision 2 §3)
    chain_backend: str = "mock"              # mock (default) | local (the existing PoW chain, mined blocks)
    ipfs: str = "local"                      # local content IDs; "real" also uploads to a daemon (optional)
    consensus: str = "pow"                   # the only option: FPoW is undefined (Revision 2 §9)
    chain_difficulty: int = 2                # the existing setting (ChainAnchor default, backend/app.py)

    # Phase 4B, ours_* only: one malicious hospital and screening (Revision 2 §4B)
    attack: Optional[str] = None             # None = clean; logistic: LR_ATTACKS, RF / XGBoost: TREE_ATTACKS
    attacker: Optional[str] = None           # pin the malicious hospital (calibration); default: from the repeat seed
    screening: bool = True                   # False = the same ours_* pipeline with no screening (the undefended row)
    tau: Optional[float] = None              # NEVER typed in: the runner reads it from taus_path (inf only in calibration)
    tau_source: Optional[str] = None         # where tau came from; written to config.json with it
    tau_policy: str = "tau_fpr5"             # the calibrated operating point to read (as analysis/calibrate.py)
    taus_path: str = DEFAULT_TAUS
    probe_rows: int = 512                    # public probe set for screening tree updates (survival/probe.py)

    # evaluation (§4)
    min_deaths_per_hospital: int = 5         # fewer deaths -> em dash for C-index / ROC-AUC per hospital

    # bookkeeping
    torch_threads: Optional[int] = None      # pin intra-op threads; byte-reproducible only at a fixed count (A5)
    out_dir: str = "fedverify/results"
    exp: Optional[str] = None                # default per dataset: survival (TCGA) / heart
    data_path: Optional[str] = None          # default per dataset, see DATA_PATHS

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}")
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if self.heart_mode not in ("paper", "full"):
            raise ValueError("heart_mode must be paper or full")
        object.__setattr__(self, "method", canonical_method(self.family, self.method))
        known = DATASET_HOSPITALS[self.dataset]
        object.__setattr__(self, "hospitals", tuple(self.hospitals) if self.hospitals is not None else known)
        unknown = [h for h in self.hospitals if h not in known]
        if unknown:
            raise ValueError(f"unknown hospital code(s) {unknown} for {self.dataset}; known: {list(known)}")
        excl = LOCAL_EXCLUDE[self.dataset] if self.local_exclude is None else tuple(self.local_exclude)
        object.__setattr__(self, "local_exclude", excl)
        object.__setattr__(self, "exp", self.exp or EXPS[self.dataset])
        object.__setattr__(self, "data_path", self.data_path or DATA_PATHS[self.dataset])
        if self.task == "survival" and not (self.dataset == "tcga" and self.family == "xgb"):
            raise ValueError("task=survival (AFT) exists for the XGBoost family on TCGA only")
        if self.include_race and self.dataset != "tcga":
            raise ValueError("--include-race applies to TCGA only")
        for name in ("repeats", "rounds", "local_epochs", "batch_size", "trees_per_hospital", "xgb_trees_per_round"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.folds < 2:
            raise ValueError("folds must be >= 2")
        if self.chain_backend not in CHAIN_BACKENDS or self.ipfs not in ("local", "real"):
            raise ValueError(f"chain_backend must be one of {CHAIN_BACKENDS} and ipfs local or real")
        if self.consensus != "pow":
            raise ValueError("consensus must be pow: FPoW is undefined, awaiting the guide (Revision 2 §9)")
        if self.rf_n_jobs != 1 or self.xgb_n_jobs != 1:
            raise ValueError("rf_n_jobs and xgb_n_jobs must be 1 (determinism rule 6)")
        eps = None if self.epsilon is None else float(self.epsilon)
        private = eps is not None and not math.isinf(eps)
        if self.family == "rf" and private:
            raise ValueError("random-forest DP is not DP-SGD and is not built: RF-DP cells are n/a "
                             "(prompt Phase 2; the idea is recorded in fedverify/TODO.md)")
        if self.family == "xgb" and private:
            raise ValueError("DP for XGBoost is not built: XGBoost DP cells are n/a (Revision 2 §2.2; see TODO.md)")
        if self.method in DP_METHODS and eps is None:
            eps = DEFAULT_DP_EPSILON
        elif private and self.method not in DP_METHODS:
            raise ValueError(f"{self.method} is a non-private method; epsilon applies only to {list(DP_METHODS)}")
        if eps is not None and not eps > 0:
            raise ValueError("epsilon must be > 0 (or inf for no DP)")
        attack = None if self.attack in (None, "", "none") else str(self.attack)
        object.__setattr__(self, "attack", attack)
        ours = self.method in OURS_METHODS
        if attack is not None and not ours:
            raise ValueError(f"attacks are simulated against the ours_* methods only {OURS_METHODS}; the undefended "
                             "row is the same ours_* method with screening off")
        if attack is not None and attack not in FAMILY_ATTACKS[self.family]:
            raise ValueError(f"{attack!r} is not a {self.family} attack; expected one of {FAMILY_ATTACKS[self.family]}")
        if self.attacker is not None and (attack is None or self.attacker not in self.hospitals):
            raise ValueError("attacker must name one of the run's hospitals, and needs an attack")
        if not self.screening and not ours:
            raise ValueError("screening=False applies to the ours_* methods only")
        if self.tau is not None:
            if not ours or not self.screening:
                raise ValueError("tau applies to a screening ours_* run only")
            if math.isnan(float(self.tau)):
                raise ValueError("tau must be a number (it comes from the calibration file)")
            object.__setattr__(self, "tau", float(self.tau))
        if int(self.probe_rows) < 16:
            raise ValueError("probe_rows must be >= 16")
        if not 0.0 < float(self.delta) < 1.0 or not float(self.max_grad_norm) > 0:
            raise ValueError("delta must be in (0, 1) and max_grad_norm > 0")
        object.__setattr__(self, "epsilon", eps)

    # ── derived ──────────────────────────────────────────────────────────────
    @property
    def model_family(self) -> str:
        return self.family

    @property
    def commit(self) -> bool:
        """Does every round go through serialize -> CID -> sign -> verify -> Merkle -> anchor?"""
        return self.method.startswith("chain_") or self.method in OURS_METHODS

    @property
    def screened(self) -> bool:
        """Does a screen decide which hospitals' updates are used (ours_* with screening on)?"""
        return self.method in OURS_METHODS and self.screening

    @property
    def calibration_family(self) -> str:
        """The family part of a taus_survival.json key: XGBoost's AFT task has its own scores and tau."""
        return "xgb_aft" if (self.family == "xgb" and self.task == "survival") else self.family

    @property
    def dataset_hospitals(self) -> tuple[str, ...]:
        """Every hospital of the dataset, in canonical order: the order of the D10 integer ids."""
        return DATASET_HOSPITALS[self.dataset]

    @property
    def hospital_index(self) -> dict[str, int]:
        return {h: i for i, h in enumerate(self.dataset_hospitals)}

    @property
    def metric_key(self) -> str:
        """Which metrics.yaml block applies: tcga (binary death indicator), tcga_survival (AFT) or heart."""
        if self.dataset == "heart":
            return "heart"
        return "tcga_survival" if self.task == "survival" else "tcga"

    @property
    def cell(self) -> str:
        cell = f"{self.method}_eps{eps_tag(self.epsilon)}" if self.method in DP_METHODS else self.method
        if self.task == "survival":
            cell += "_aft"
        if self.dataset == "heart" and self.heart_mode == "full":
            cell += "_full"
        if self.commit and self.chain_backend != "mock":
            cell += f"_{self.chain_backend}"           # a mined ledger never overwrites the default mock run
        if self.attack:
            cell += f"_{self.attack}"
        if self.attacker:
            cell += "_att" + re.sub(r"[^A-Za-z0-9]+", "", self.attacker)     # calibration cells pin the attacker
        if self.method in OURS_METHODS and not self.screening:
            cell += "_noscreen"
        return cell

    @property
    def aggregation(self) -> str:
        if self.method in OURS_METHODS and not self.screening:
            return UNSCREENED_AGGREGATION[self.method]
        return AGGREGATION[self.method]

    @property
    def run_dir(self) -> str:
        return os.path.join(self.out_dir, self.exp, self.cell, f"seed{self.seed}")

    def repeat_seed(self, r: int) -> int:
        return int(self.seed) + int(r)

    def to_dict(self) -> dict:
        """JSON-safe dict of EVERY parameter that influenced the run."""
        d = asdict(self)
        d["hospitals"] = list(self.hospitals)
        d["local_exclude"] = list(self.local_exclude)
        for k, v in d.items():
            if isinstance(v, float) and math.isinf(v):
                d[k] = "inf"
        return d
