import torch
from fedverify.config import FLConfig
from fedverify.core.client import ClientUpdate
from fedverify.core.server import FedAvg, build_aggregator, evaluate

CFG = FLConfig(dataset="mnist", num_clients=2, rounds=1)


def _u(cid, delta, n):
    return ClientUpdate(client_id=cid, delta=torch.tensor(delta, dtype=torch.float32),
                        num_samples=n, train_loss=0.0, wall_time_s=0.0)


def test_fedavg_of_identical_deltas_returns_that_delta():
    d = torch.tensor([1.0, -2.0, 3.5])
    ups = [_u(i, d.tolist(), 10) for i in range(4)]
    out, diag = FedAvg().aggregate(ups, CFG, 1)
    assert torch.allclose(out, d)
    assert diag["accepted"] == [0, 1, 2, 3] and diag["rejected"] == []


def test_fedavg_weights_by_num_samples():
    # n=1 with delta 0, n=3 with delta 4  ->  (1*0 + 3*4)/4 = 3
    out, _ = FedAvg().aggregate([_u(0, [0.0], 1), _u(1, [4.0], 3)], CFG, 1)
    assert torch.allclose(out, torch.tensor([3.0]))


def test_diag_always_has_required_keys():
    _, diag = FedAvg().aggregate([_u(0, [1.0], 1)], CFG, 1)
    for k in ("accepted", "rejected", "scores"):
        assert k in diag


def test_empty_updates_raises():
    try:
        FedAvg().aggregate([], CFG, 1)
        assert False, "should raise"
    except ValueError:
        pass


def test_build_aggregator_rejects_unknown():
    assert isinstance(build_aggregator("fedavg"), FedAvg)
    try:
        build_aggregator("nope")
        assert False, "should raise"
    except ValueError:
        pass


def test_evaluate_perfect_and_metrics_present():
    class Perfect(torch.nn.Module):
        def forward(self, x):
            # one-hot on the true label smuggled through the input's first pixel
            idx = x[:, 0, 0, 0].long()
            return torch.nn.functional.one_hot(idx, 3).float() * 10

    xs = torch.zeros(6, 1, 1, 1)
    ys = torch.tensor([0, 1, 2, 0, 1, 2])
    xs[:, 0, 0, 0] = ys.float()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=2)
    m = evaluate(Perfect(), loader, "cpu")
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert set(m["per_class_acc"]) == {"0", "1", "2"}
