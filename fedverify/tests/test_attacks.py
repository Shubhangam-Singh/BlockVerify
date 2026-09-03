"""Phase-4 attack tests.

Attacker identity must depend only on the seed, so the SAME clients are malicious for
every aggregator being compared in Table 2 — otherwise the comparison is confounded.
"""
import math

import pytest
import torch

from fedverify.attacks.backdoor import (apply_trigger, attack_success_rate,
                                        build_triggered_testset, white_value)
from fedverify.attacks.byzantine import (DATA_ATTACKS, DELTA_ATTACKS, apply_delta_attack,
                                         attack_active, attacker_ids, poison_batch)
from fedverify.config import FLConfig


def cfg(**kw):
    return FLConfig(**{**dict(dataset="mnist", num_clients=10, seed=0,
                              attack="scaling", attacker_frac=0.3), **kw})


# ── attacker selection ───────────────────────────────────────────────────────
def test_attacker_ids_are_deterministic():
    assert attacker_ids(cfg()) == attacker_ids(cfg())


def test_attacker_ids_depend_on_seed():
    assert attacker_ids(cfg(seed=0)) != attacker_ids(cfg(seed=1))


def test_attacker_ids_do_not_depend_on_the_attack_or_aggregator():
    """Table 2 compares aggregators on the SAME malicious clients."""
    base = attacker_ids(cfg(attack="scaling"))
    assert base == attacker_ids(cfg(attack="sign_flip"))
    assert base == attacker_ids(cfg(attack="backdoor", aggregator="krum"))


def test_attacker_count_matches_fraction():
    for frac, want in [(0.1, 1), (0.2, 2), (0.3, 3)]:
        assert len(attacker_ids(cfg(attacker_frac=frac))) == want


def test_no_attackers_without_an_attack():
    assert attacker_ids(cfg(attack=None)) == set()
    assert attacker_ids(cfg(attack="none")) == set()
    assert attacker_ids(cfg(attacker_frac=0.0)) == set()


def test_attack_start_round_is_honoured():
    c = cfg(attack_start_round=5)
    assert not attack_active(c, 4)
    assert attack_active(c, 5) and attack_active(c, 6)


# ── delta-level attacks ──────────────────────────────────────────────────────
def test_sign_flip_negates_the_delta():
    d = torch.tensor([1.0, -2.0, 3.0])
    assert torch.equal(apply_delta_attack(d, cfg(attack="sign_flip"), 1, 0, 3), -d)


def test_sign_flip_scale_is_applied():
    d = torch.tensor([1.0, 2.0])
    out = apply_delta_attack(d, cfg(attack="sign_flip", sign_flip_scale=4.0), 1, 0, 3)
    assert torch.equal(out, -4.0 * d)


def test_zero_attack_returns_zeros():
    out = apply_delta_attack(torch.randn(20), cfg(attack="zero"), 1, 0, 3)
    assert torch.equal(out, torch.zeros(20))


def test_scaling_multiplies_by_K_over_f():
    d = torch.ones(4)
    out = apply_delta_attack(d, cfg(attack="scaling", num_clients=10), 1, 0, 3)
    assert torch.allclose(out, torch.full((4,), 10 / 3))


def test_gaussian_replaces_the_delta_with_noise_and_is_reproducible():
    d = torch.ones(500)
    c = cfg(attack="gaussian", gaussian_sigma=2.0)
    a = apply_delta_attack(d, c, 3, 1, 3)
    b = apply_delta_attack(d, c, 3, 1, 3)
    assert torch.equal(a, b)                                # same (seed, round, client)
    assert not torch.equal(a, apply_delta_attack(d, c, 4, 1, 3))   # differs by round
    assert not torch.equal(a, apply_delta_attack(d, c, 3, 2, 3))   # differs by client
    assert abs(float(a.std()) - 2.0) < 0.3


def test_delta_attacks_do_not_mutate_the_input():
    d = torch.ones(8)
    for atk in sorted(DELTA_ATTACKS):
        apply_delta_attack(d, cfg(attack=atk), 1, 0, 2)
    assert torch.equal(d, torch.ones(8))


def test_data_attack_rejected_by_delta_path():
    with pytest.raises(ValueError, match="not a delta-level attack"):
        apply_delta_attack(torch.ones(4), cfg(attack="label_flip"), 1, 0, 2)


# ── data-level attacks ───────────────────────────────────────────────────────
def test_label_flip_maps_y_to_9_minus_y():
    y = torch.arange(10)
    _x, y2 = poison_batch(torch.zeros(10, 1, 28, 28), y, cfg(attack="label_flip"), 1, 0, 10)
    assert torch.equal(y2, 9 - y)


def test_label_flip_leaves_images_untouched():
    x = torch.randn(4, 1, 28, 28)
    x2, _ = poison_batch(x, torch.zeros(4, dtype=torch.long),
                         cfg(attack="label_flip"), 1, 0, 10)
    assert torch.equal(x, x2)


# ── backdoor ─────────────────────────────────────────────────────────────────
def test_trigger_is_written_in_normalised_space_not_raw_one():
    from fedverify.core.data import dataset_spec
    spec = dataset_spec("mnist")
    v = white_value(spec)
    assert v > 2.0                                   # (1 - 0.1307)/0.3081 ~= 2.82
    out = apply_trigger(torch.zeros(1, 1, 28, 28), spec)
    assert torch.allclose(out[0, 0, 25:, 25:], torch.full((3, 3), v))


def test_trigger_only_touches_the_bottom_right_corner():
    out = apply_trigger(torch.zeros(2, 1, 28, 28), None)
    assert bool((out[:, :, :25, :] == 0).all()) and bool((out[:, :, :, :25] == 0).all())


def test_backdoor_relabels_only_the_poisoned_share():
    x = torch.zeros(200, 1, 28, 28)
    y = torch.full((200,), 5, dtype=torch.long)
    c = cfg(attack="backdoor", poison_frac=0.5, backdoor_target=0)
    _x2, y2 = poison_batch(x, y, c, 1, 0, 10)
    n_target = int((y2 == 0).sum())
    assert 60 < n_target < 140                       # ~50% of 200, binomial slack
    assert int((y2 == 5).sum()) == 200 - n_target


def test_backdoor_poison_frac_zero_changes_nothing():
    x = torch.zeros(50, 1, 28, 28)
    y = torch.full((50,), 5, dtype=torch.long)
    x2, y2 = poison_batch(x, y, cfg(attack="backdoor", poison_frac=0.0), 1, 0, 10)
    assert torch.equal(y, y2) and torch.equal(x, x2)


def test_triggered_testset_excludes_the_target_class():
    """Target-class samples would be free successes and would inflate ASR."""
    ds = [(torch.zeros(1, 8, 8), i % 3) for i in range(30)]
    X, y = build_triggered_testset(ds, None, target=1)
    assert 1 not in set(y.tolist())
    assert X.shape[0] == len(y) == 20


def test_asr_is_one_for_a_model_that_always_predicts_the_target():
    class AlwaysZero(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], 3)
            out[:, 0] = 10.0
            return out
    ds = [(torch.zeros(1, 8, 8), 1 + (i % 2)) for i in range(10)]
    trig = build_triggered_testset(ds, None, target=0)
    assert attack_success_rate(AlwaysZero(), trig, 0) == 1.0


def test_asr_is_nan_without_a_triggered_set():
    assert math.isnan(attack_success_rate(None, None, 0))
