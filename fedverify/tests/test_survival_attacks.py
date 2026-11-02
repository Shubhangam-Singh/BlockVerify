"""Revision 2 Phase 4: the reference paper's three chain attacks, poisoning attacks, screening, tau from the file."""
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
pytest.importorskip("lifelines")

from fedverify.aggregators import build_aggregator  # noqa: E402
from fedverify.attacks.byzantine import attacker_ids  # noqa: E402
from fedverify.core.client import ClientUpdate  # noqa: E402
from fedverify.survival import chain_attacks as CA  # noqa: E402
from fedverify.survival import fl_rf, fl_xgb, poison, probe, runner  # noqa: E402
from fedverify.survival import heart as H  # noqa: E402
from fedverify.survival.config import HEART_HOSPITALS, HOSPITALS, SurvivalConfig  # noqa: E402
from fedverify.survival.screening import TreeProbeScreen, cfg_tau_key, load_tau, tau_key  # noqa: E402

needs_heart = pytest.mark.skipif(not H.DEFAULT_DATA.exists(), reason="heart CSV (gitignored) not present")
NAMES = ["age", "age_missing", "chol", "sex=Female", "sex=Male", "sex=NA", "cp=a", "cp=b"]   # a toy schema


def _update(cid, delta):
    return ClientUpdate(client_id=cid, delta=delta, num_samples=50, train_loss=0.0, wall_time_s=0.0)


# ── 4A: the three chain attacks ──────────────────────────────────────────────
VARIANTS = [(a, v) for a, vs in CA.ATTACKS.items() for v in vs]


@needs_heart
@pytest.mark.parametrize("attack,variant", VARIANTS)
def test_each_chain_attack_variant_is_caught(attack, variant):
    sim = CA.Simulation(attack, seed=0)
    assert sim.run_round(1, variant="none")["held"]
    rec = sim.run_round(2, variant=variant, target="Hungary")
    assert rec["held"] and rec["adversarial_accepted"] == []
    assert rec["honest_record_audit"] == "ok"
    if variant in ("timestamp_changed", "txn_id_changed"):
        assert "Hungary" not in rec["accepted"]          # the altered submission is refused, not repaired
    elif variant != "leaf_order_swapped":
        assert rec["accepted"].count("Hungary") == 1     # the target's own update counted exactly once


@needs_heart
@pytest.mark.parametrize("attack,variant", VARIANTS)
def test_each_chain_attack_variant_lands_when_its_defence_is_switched_off(attack, variant):
    """The defence tests above are not vacuous: without the defence the same attempt succeeds."""
    sim = CA.Simulation(attack, seed=0, defences=CA.defences_for(attack, defence_off=True))
    sim.run_round(1, variant="none")
    rec = sim.run_round(2, variant=variant, target="Hungary")
    assert not rec["held"]
    if variant == "leaf_order_swapped":
        assert rec["audit"] == "the swap passed the audit"
    else:
        assert rec["adversarial_accepted"] == rec["adversarial_arrivals"]


@needs_heart
def test_double_spending_is_flagged_as_equivocation_and_replays_are_refused_with_reasons():
    sim = CA.Simulation("double_spending", seed=0)
    rec = sim.run_round(1, variant="conflicting_update", target="Cleveland")
    assert rec["equivocations"] and rec["equivocations"][0]["hospital"] == "Cleveland"
    assert any(r["reason"].startswith("equivocation") for r in rec["rejected"])
    rec = sim.run_round(2, variant="replayed_submission", target="Cleveland")
    assert any(r["reason"].startswith("replay") for r in rec["rejected"])
    rec = sim.run_round(3, variant="stale_update_respent", target="Cleveland")
    assert any("payload header names round" in r["reason"] for r in rec["rejected"])


@needs_heart
def test_endpoint_compromise_reasons():
    sim = CA.Simulation("endpoint_compromise", seed=0)
    assert "unregistered key" in sim.run_round(1, "unregistered_key", "Hungary")["rejected"][0]["reason"]
    rec = sim.run_round(2, "wrong_hospital_key", "Hungary")
    assert "invalid signature" in rec["rejected"][0]["reason"] and rec["note"]["signed_with"] != "Hungary"
    rec = sim.run_round(3, "key_reregistration", "Hungary")
    assert rec["note"]["registration"].startswith("refused") and rec["held"]


@needs_heart
def test_chain_attack_simulation_is_deterministic_and_lists_its_variants(tmp_path):
    a = CA.simulate("malleability", rounds=12, seed=3, out_dir=str(tmp_path / "a"))
    b = CA.simulate("malleability", rounds=12, seed=3, out_dir=str(tmp_path / "b"))
    for f in ("config.json", "rounds.jsonl", "summary.json"):
        assert (Path(a["out"]) / f).read_bytes() == (Path(b["out"]) / f).read_bytes()
    cfg = json.loads((Path(a["out"]) / "config.json").read_text())
    assert cfg["variants"] == list(CA.ATTACKS["malleability"]) and cfg["rounds"] == 12
    assert cfg["defences_switched_off"] == [] and a["resilience_pct"] == 100.0 and a["attack_success_pct"] == 0.0
    assert sum(v["rounds"] for v in a["by_variant"].values()) == 12


# ── 4B: the poisoning attacks change the update as specified ─────────────────
def test_one_attacker_is_chosen_from_the_repeat_seed_by_the_existing_rule():
    cfg = SurvivalConfig(family="lr", method="ours_lr", attack="sign_flip")
    for seed in range(5):
        view = SimpleNamespace(attack="sign_flip", attacker_frac=1 / 5, num_clients=5, seed=seed)
        assert poison.attackers_for(cfg, seed) == [HOSPITALS[i] for i in sorted(attacker_ids(view))]
        assert len(poison.attackers_for(cfg, seed)) == 1
    assert poison.attackers_for(dataclasses.replace(cfg, attacker="E2"), 0) == ["E2"]
    assert poison.attackers_for(dataclasses.replace(cfg, attack=None), 0) == []


def test_logistic_attacks_change_the_update_as_specified():
    d = torch.arange(1.0, 6.0)
    cfg = SurvivalConfig(dataset="heart", family="lr", method="ours_lr", attack="sign_flip")
    assert torch.equal(poison.poisoned_delta(d, cfg, 0, 1, "Hungary", 1), -d)
    cfg = dataclasses.replace(cfg, attack="scaling")
    assert torch.equal(poison.poisoned_delta(d, cfg, 0, 1, "Hungary", 1), 4.0 * d)      # K / attackers = 4 / 1
    cfg = dataclasses.replace(cfg, attack="label_flip")
    X, y = torch.rand(6, 3), torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    X2, y2 = poison.poisoned_labels(X, y, cfg, 0, 1, "Hungary")
    assert torch.equal(X2, X) and torch.equal(y2, 1.0 - y)
    assert torch.equal(d, torch.arange(1.0, 6.0))                                       # inputs untouched


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = probe.draw_rows(NAMES, n, seed)
    y = (X[:, 0] + 0.2 * rng.normal(size=n) > 0.5).astype(np.int64)
    return X, y


def test_label_flip_trees_trains_on_flipped_labels_and_inverts_the_forest():
    X, y = _toy()
    cfg = SurvivalConfig(dataset="heart", family="rf", method="ours_rf", attack="label_flip_trees")
    Xa, ya = fl_rf.attacker_training_set(cfg, X, y, NAMES, 0, 0, 1, "Hungary")
    assert np.array_equal(Xa, X) and np.array_equal(ya, 1 - y)
    honest = fl_rf.make_forest(cfg, 10, 7).fit(X, y)
    bad = fl_rf.make_forest(cfg, 10, 7).fit(Xa, ya)
    p_h, p_b = fl_rf.tree_probability_sum(honest, X) / 10, fl_rf.tree_probability_sum(bad, X) / 10
    assert p_h[y == 1].mean() > 0.5 > p_b[y == 1].mean()


def test_random_trees_use_public_schema_rows_and_a_permutation_of_the_labels():
    X, y = _toy()
    cfg = SurvivalConfig(dataset="heart", family="rf", method="ours_rf", attack="random_trees")
    Xa, ya = fl_rf.attacker_training_set(cfg, X, y, NAMES, 0, 0, 1, "Hungary")
    assert Xa.shape == X.shape and not np.array_equal(Xa, X)
    assert sorted(ya.tolist()) == sorted(y.tolist()) and not np.array_equal(ya, y)
    assert (Xa[:, 3:6].sum(axis=1) == 1).all()                                          # schema rows, no patients


@pytest.mark.parametrize("task", ["binary", "survival"])
def test_xgboost_label_flip_trees_are_the_trees_of_the_flipped_labels(task):
    X, y = _toy()
    t = np.linspace(30.0, 3000.0, len(y))
    kw = dict(family="xgb", method="ours_xgb", attack="label_flip_trees", task=task)
    cfg = SurvivalConfig(**kw) if task == "survival" else SurvivalConfig(dataset="heart", **kw)
    m = np.full(len(y), fl_xgb.initial_margin(cfg))
    bad = fl_xgb.attacker_trees(cfg, X, y, t, m, 5, 11, NAMES, 0, 0, 1, "A2")
    ref = fl_xgb.train_new_trees(cfg, X, 1 - y, t, m, 5, 11)            # events flipped, times unchanged (AFT)
    assert bad.get_dump(dump_format="json") == ref.get_dump(dump_format="json")
    honest = fl_xgb.train_new_trees(cfg, X, y, t, m, 5, 11)
    assert bad.get_dump(dump_format="json") != honest.get_dump(dump_format="json")


# ── 4B: screening ────────────────────────────────────────────────────────────
def test_probe_set_is_public_and_follows_the_schema():
    X = probe.draw_rows(NAMES, 400, seed=5)
    assert X.shape == (400, len(NAMES)) and X.dtype == np.float32
    assert np.array_equal(X, probe.draw_rows(NAMES, 400, seed=5))
    assert ((X[:, 0] >= 0) & (X[:, 0] <= 1)).all() and ((X[:, 2] >= 0) & (X[:, 2] <= 1)).all()
    miss = X[:, 1] == 1
    assert (X[miss, 0] == 0).all() and 0.05 < miss.mean() < 0.25
    assert (X[:, 3:6].sum(axis=1) == 1).all() and (X[:, 6:8].sum(axis=1) == 1).all()


def test_a_screen_without_tau_raises():
    with pytest.raises(ValueError, match="tau"):
        TreeProbeScreen(None)
    upd = [_update(i, torch.ones(4)) for i in range(5)]
    with pytest.raises(ValueError, match="tau"):
        build_aggregator("forensics").aggregate(upd, SimpleNamespace(tau=None), 1)


def test_an_all_nan_update_is_rejected():
    rng = np.random.default_rng(0)
    upd = [_update(i, torch.tensor(rng.normal(size=24), dtype=torch.float32)) for i in range(5)]
    upd[2] = _update(2, torch.full((24,), float("nan")))
    _, diag = build_aggregator("forensics").aggregate(upd, SimpleNamespace(tau=3.0), 1)
    assert 2 in diag["rejected"] and diag["scores"]["2"]["rejected_by"].startswith("health:nan")
    preds = {h: rng.random(64) for h in HOSPITALS}
    preds["E2"] = np.full(64, np.nan)
    accepted, diag = TreeProbeScreen(3.0).screen(preds)
    assert "E2" in diag["rejected"] and "E2" not in accepted
    assert diag["scores"]["E2"]["rejected_by"].startswith("health:nan")
    leaves = {h: np.zeros(8) for h in HOSPITALS}
    leaves["A8"] = np.full(8, np.nan)                                    # NaN leaves no probe row happens to reach
    _, diag = TreeProbeScreen(3.0).screen({h: np.linspace(0, 1, 64) for h in HOSPITALS}, leaves)
    assert diag["rejected"] == ["A8"]


def test_five_identical_updates_reject_nobody():
    upd = [_update(i, torch.linspace(-1.0, 1.0, 24)) for i in range(5)]
    _, diag = build_aggregator("forensics").aggregate(upd, SimpleNamespace(tau=0.5), 1)
    assert diag["rejected"] == [] and diag["accepted"] == [0, 1, 2, 3, 4]
    accepted, diag = TreeProbeScreen(0.5).screen({h: np.linspace(0.0, 1.0, 64) for h in HOSPITALS})
    assert accepted == list(HOSPITALS) and diag["rejected"] == []


def test_the_tree_screen_rejects_a_hospital_far_from_the_median():
    rng = np.random.default_rng(1)
    base = rng.random(128)
    preds = {h: base + rng.normal(0.0, 0.01, 128) for h in HOSPITALS}
    preds["A8"] = 1.0 - base                                             # a label-flipped forest
    accepted, diag = TreeProbeScreen(3.0).screen(preds)
    assert diag["rejected"] == ["A8"] and diag["scores"]["A8"]["rejected_by"] == "probe_distance"
    assert diag["scores"]["A8"]["combined"] > 3.0 > max(diag["scores"][h]["combined"] for h in accepted)


# ── tau comes from the calibration file ──────────────────────────────────────
def test_tau_is_read_for_exactly_its_key(tmp_path):
    key = tau_key("heart", "xgb", None, 4)
    assert key == "dataset=heart|family=xgb|eps=inf|K=4"
    f = tmp_path / "taus.json"
    with pytest.raises(FileNotFoundError):
        load_tau(f, key)
    f.write_text(json.dumps({"by_key": {key: {"combined": {"tau_fpr5": 2.5}}}}))
    assert load_tau(f, key) == (2.5, f"{f}:{key}:tau_fpr5")
    with pytest.raises(KeyError):
        load_tau(f, tau_key("heart", "xgb", None, 5))                    # no fallback to another K
    cfg = SurvivalConfig(family="xgb", method="ours_xgb", task="survival")
    assert cfg_tau_key(cfg) == "dataset=tcga|family=xgb_aft|eps=inf|K=5"
    assert cfg_tau_key(SurvivalConfig(family="lr", method="ours_lr")) == "dataset=tcga|family=lr|eps=4|K=5"


@needs_heart
def test_a_screening_run_reads_tau_from_the_file_and_records_it(tmp_path):
    taus = tmp_path / "taus.json"
    cfg = SurvivalConfig(dataset="heart", family="xgb", method="ours_xgb", attack="label_flip_trees", rounds=2,
                         repeats=2, folds=2, out_dir=str(tmp_path / "r"), taus_path=str(taus))
    with pytest.raises(FileNotFoundError):
        runner.run(cfg, verbose=False)
    taus.write_text(json.dumps({"by_key": {cfg_tau_key(cfg): {"combined": {"tau_fpr5": 1.75}}}}))
    s = runner.run(cfg, verbose=False)
    c = json.loads((Path(cfg.run_dir) / "config.json").read_text())
    assert c["screening"]["tau"] == 1.75 and c["config"]["tau"] == 1.75
    assert c["screening"]["tau_source"] == f"{taus}:{cfg_tau_key(cfg)}:tau_fpr5"
    assert c["attacker_ids_by_repeat"] == [poison.attackers_for(cfg, 0), poison.attackers_for(cfg, 1)]
    rec = json.loads((Path(cfg.run_dir) / "repeat1" / "fold0" / "rounds.jsonl").read_text().splitlines()[0])
    assert rec["diag"]["tau"] == 1.75 and rec["attack"]["attackers"] == poison.attackers_for(cfg, 1)
    assert rec["attack"]["tp"] + rec["attack"]["fn"] == 1
    assert s["attack"]["tau"] == 1.75 and s["attack"]["detection"]["recall"]["n_repeats"] in (0, 2)


@needs_heart
@pytest.mark.parametrize("family,ours,chain,extra", [("lr", "ours_lr", "chain_fedavg_lr", {"epsilon": float("inf")}),
                                                     ("rf", "ours_rf", "chain_fed_rf", {}),
                                                     ("xgb", "ours_xgb", "chain_fedxgb_bagging", {})])
def test_ours_without_attack_or_screening_learns_exactly_the_commit_only_method(tmp_path, family, ours, chain, extra):
    kw = dict(dataset="heart", family=family, rounds=2, repeats=1, folds=2)
    a = SurvivalConfig(method=chain, out_dir=str(tmp_path / "a"), **kw)
    b = SurvivalConfig(method=ours, screening=False, out_dir=str(tmp_path / "b"), **kw, **extra)
    sa, sb = runner.run(a, verbose=False), runner.run(b, verbose=False)
    assert sa["repeats"][0]["pooled"] == sb["repeats"][0]["pooled"]
    assert (Path(a.run_dir) / "repeat0" / "oof.jsonl").read_text() == (Path(b.run_dir) / "repeat0" / "oof.jsonl").read_text()


def test_attacks_and_screening_belong_to_the_ours_methods_only():
    with pytest.raises(ValueError, match="ours"):
        SurvivalConfig(family="lr", method="fedavg_lr", attack="sign_flip")
    with pytest.raises(ValueError, match="not a rf attack"):
        SurvivalConfig(family="rf", method="ours_rf", attack="sign_flip")
    with pytest.raises(ValueError, match="ours"):
        SurvivalConfig(family="xgb", method="fedxgb_bagging", screening=False)
    with pytest.raises(ValueError, match="tau"):
        SurvivalConfig(family="lr", method="fedavg_lr", tau=1.0)
    ours = SurvivalConfig(dataset="heart", family="lr", method="ours_lr", attack="scaling", screening=False)
    assert ours.epsilon == 4.0 and ours.cell == "ours_lr_eps4_scaling_noscreen" and ours.aggregation == "fedavg"
    assert SurvivalConfig(family="xgb", method="ours_xgb", task="survival").cell == "ours_xgb_aft"
    assert [h for h in HEART_HOSPITALS] == list(SurvivalConfig(dataset="heart", family="rf", method="ours_rf").hospitals)


@needs_heart
def test_an_attacked_screened_run_is_byte_reproducible(tmp_path):
    from fedverify.core.runner import strip_timings
    taus = tmp_path / "taus.json"
    outs = []
    for sub in ("a", "b"):
        cfg = SurvivalConfig(dataset="heart", family="rf", method="ours_rf", attack="random_trees", rounds=2,
                             repeats=1, folds=2, out_dir=str(tmp_path / sub), taus_path=str(taus))
        taus.write_text(json.dumps({"by_key": {cfg_tau_key(cfg): {"combined": {"tau_fpr5": 1.0}}}}))
        runner.run(cfg, verbose=False)
        run_dir = Path(cfg.run_dir)
        outs.append([json.dumps(strip_timings(json.loads(l)), sort_keys=True)
                     for p in sorted(run_dir.rglob("*.jsonl")) for l in p.read_text().splitlines()]
                    + [(run_dir / "summary.json").read_text()])
    assert outs[0] == outs[1]
