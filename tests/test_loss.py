"""Sequence loss metrics, reductions, and normalized divergence."""

import numpy as np
import pytest
import torch

from flash_softdtw import SoftDTWLoss, pairwise_cost, sakoe_chiba_band


def custom_metric(x, y):
    return (x - y).square().sum(-1)


METRICS = ["sqeuclidean", "euclidean", "l1", "cosine", custom_metric]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("window", [None, 2])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_loss_metrics_forward_backward(device, metric, window, reduction):
    rng = np.random.default_rng(1024)
    x = torch.tensor(rng.normal(size=(2, 5, 3)), device=device, dtype=torch.float64,
                     requires_grad=True)
    y = torch.tensor(rng.normal(size=(2, 6, 3)), device=device, dtype=torch.float64,
                     requires_grad=True)
    xs = torch.tensor([5, 3], device=device)
    ys = torch.tensor([6, 4], device=device)
    loss = SoftDTWLoss(gamma=0.8, metric=metric, window=window,
                       warp_penalty=0.3, reduction=reduction)(x, y, xs, ys)
    assert loss.shape == ((2,) if reduction == "none" else ())
    assert torch.isfinite(loss).all()
    gx, gy = torch.autograd.grad(loss.sum(), (x, y))
    assert gx.shape == x.shape and gy.shape == y.shape
    assert torch.isfinite(gx).all() and torch.isfinite(gy).all()
    assert torch.count_nonzero(gx[:, :3]) > 0
    assert torch.count_nonzero(gy[:, :4]) > 0
    assert torch.count_nonzero(gx[1, 3:]) == 0
    assert torch.count_nonzero(gy[1, 4:]) == 0
    with torch.no_grad():
        values = SoftDTWLoss(gamma=0.8, metric=metric, window=window,
                             warp_penalty=0.3, reduction="none")(x, y, xs, ys)
    expected = values if reduction == "none" else getattr(values, reduction)()
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("metric", METRICS)
def test_pairwise_metrics_dense_and_compact(device, metric):
    rng = np.random.default_rng(33)
    x = torch.tensor(rng.normal(size=(2, 5, 3)), device=device, dtype=torch.float64)
    y = torch.tensor(rng.normal(size=(2, 6, 3)), device=device, dtype=torch.float64)
    band = sakoe_chiba_band(torch.tensor([5, 5], device=device),
                            torch.tensor([6, 6], device=device), 1)
    dense = pairwise_cost(x, y, metric)
    compact = pairwise_cost(x, y, metric, band)
    assert dense.shape == (2, 5, 6)
    for b in range(2):
        for i in range(5):
            lo, hi = band[b, i].tolist()
            torch.testing.assert_close(compact[b, i, :hi - lo], dense[b, i, lo:hi])


@pytest.mark.parametrize("gamma", [0.01, 0.5, 2.0])
@pytest.mark.parametrize("metric", ["sqeuclidean", "l1"])
def test_normalized_divergence(device, gamma, metric):
    rng = np.random.default_rng(212)
    x = torch.tensor(rng.normal(size=(3, 5, 2)), device=device, dtype=torch.float64,
                     requires_grad=True)
    y = torch.tensor(rng.normal(size=(3, 6, 2)), device=device, dtype=torch.float64,
                     requires_grad=True)
    loss = SoftDTWLoss(gamma=gamma, metric=metric, normalize=True, reduction="none")
    identical = loss(x, x)
    different = loss(x, y)
    torch.testing.assert_close(identical, torch.zeros_like(identical), atol=1e-10, rtol=0)
    assert torch.all(different >= -1e-9)
    gx, gy = torch.autograd.grad(different.sum(), (x, y))
    assert torch.isfinite(gx).all() and torch.isfinite(gy).all()


def test_default_loss_lengths(device):
    rng = np.random.default_rng(1234)
    x = torch.tensor(rng.normal(size=(2, 3, 2)), device=device, dtype=torch.float64)
    y = torch.tensor(rng.normal(size=(2, 4, 2)), device=device, dtype=torch.float64)
    loss = SoftDTWLoss(reduction="none")
    explicit = loss(x, y, torch.tensor([3, 3], device=device),
                     torch.tensor([4, 4], device=device))
    torch.testing.assert_close(loss(x, y), explicit)
