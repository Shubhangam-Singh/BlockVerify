import torch
from fedverify.core.models import (SmallCNN, build_model, get_flat_params,
                                   set_flat_params, assert_no_batchnorm)


def test_flat_params_roundtrip_is_exact():
    torch.manual_seed(0)
    m = SmallCNN()
    orig = get_flat_params(m).clone()
    vec = torch.randn_like(orig)
    set_flat_params(m, vec)
    assert torch.equal(get_flat_params(m), vec), "set/get must be exact inverses"
    set_flat_params(m, orig)
    assert torch.equal(get_flat_params(m), orig)


def test_flat_params_length_matches_model():
    m = SmallCNN()
    assert get_flat_params(m).numel() == sum(p.numel() for p in m.parameters())


def test_set_flat_params_rejects_wrong_size():
    m = SmallCNN()
    try:
        set_flat_params(m, torch.zeros(3))
        assert False, "should have raised"
    except ValueError:
        pass


def test_forward_shape_and_no_batchnorm():
    m = build_model("mnist", 10, (1, 28, 28))
    assert m(torch.zeros(4, 1, 28, 28)).shape == (4, 10)
    assert_no_batchnorm(m)          # must not raise


def test_assert_no_batchnorm_catches_batchnorm():
    bad = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3), torch.nn.BatchNorm2d(4))
    try:
        assert_no_batchnorm(bad)
        assert False, "BatchNorm must be rejected"
    except ValueError as e:
        assert "BatchNorm" in str(e)
