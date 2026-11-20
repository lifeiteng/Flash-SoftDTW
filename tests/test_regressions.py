"""Edge cases found during review: obstacles, grad mode, metric consistency."""

import math

import pytest
import torch

import flash_softdtw as fsd
from flash_softdtw import functional


def test_infinite_obstacles_keep_end_reachable(device):
    inf = math.inf
    for dtype in (torch.float32, torch.float64):
        cost = torch.tensor([[[0, inf, inf], [0, inf, 0], [0, 0, 0]]], dtype=dtype,
                            device=device, requires_grad=True)
        value = fsd.soft_dtw(cost, gamma=1.0)
        torch.testing.assert_close(value, torch.tensor([-math.log(2)], dtype=value.dtype,
                                                       device=device))
        grad, = torch.autograd.grad(value.sum(), cost)
        assert torch.isfinite(grad).all()
        assert grad[0, 0, 1] == 0 and grad[0, 1, 1] == 0
        torch.testing.assert_close(grad[0, 2, 2], torch.ones((), dtype=dtype, device=device))


def test_fully_blocked_problem_is_unreachable(device):
    cost = torch.tensor([[[0, 1], [math.inf, math.inf]]], device=device, requires_grad=True)
    value = fsd.soft_dtw(cost)
    assert torch.isinf(value).all() and (value > 0).all()
    grad, = torch.autograd.grad(value.sum(), cost)
    assert torch.count_nonzero(grad) == 0


@pytest.mark.parametrize("mode", ["no_grad", "inference_mode", "no_requires_grad"])
def test_inference_does_not_save_r(device, mode, monkeypatch):
    seen = []
    native = functional.ext()

    class Spy:
        def __getattr__(self, name):
            return getattr(native, name)

        def forward(self, *args):
            seen.append(args[7])
            return native.forward(*args)

    monkeypatch.setattr(functional, "ext", lambda: Spy())
    cost = torch.rand(2, 5, 6, device=device, requires_grad=(mode != "no_requires_grad"))
    if mode == "no_grad":
        with torch.no_grad():
            fsd.soft_dtw(cost)
    elif mode == "inference_mode":
        with torch.inference_mode():
            fsd.soft_dtw(cost)
    else:
        fsd.soft_dtw(cost)
    assert seen == [False]


def test_training_saves_r(device, monkeypatch):
    seen = []
    native = functional.ext()

    class Spy:
        def __getattr__(self, name):
            return getattr(native, name)

        def forward(self, *args):
            seen.append(args[7])
            return native.forward(*args)

    monkeypatch.setattr(functional, "ext", lambda: Spy())
    cost = torch.rand(2, 5, 6, device=device, requires_grad=True)
    fsd.soft_dtw(cost).sum().backward()
    assert seen == [True]


def test_cosine_dense_matches_banded_near_zero(device):
    x = torch.tensor([[[1e-10], [0.0], [2.0]]], device=device)
    y = torch.tensor([[[1e-10], [3.0], [0.0], [-1.0]]], device=device)
    band = torch.tensor([[[0, 4]] * 3], dtype=torch.int32, device=device)
    dense = fsd.pairwise_cost(x, y, "cosine")
    compact = fsd.pairwise_cost(x, y, "cosine", band=band)
    torch.testing.assert_close(dense, compact)


@pytest.mark.parametrize("metric", ["l1", "euclidean", "sqeuclidean", "cosine"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_loss_accepts_half_precision_features(device, metric, dtype):
    if device.type == "cpu" and dtype == torch.float16 and metric in ("sqeuclidean", "cosine"):
        pytest.skip("CPU float16 matmul is not universally available")
    x = torch.randn(2, 6, 4, device=device).to(dtype).requires_grad_()
    y = torch.randn(2, 5, 4, device=device).to(dtype)
    loss = fsd.SoftDTWLoss(gamma=0.5, metric=metric)(x, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("gamma,warp", [(math.inf, 0.0), (0.0, 0.0), (1.0, math.nan)])
def test_invalid_scalars_are_rejected(device, gamma, warp):
    cost = torch.rand(1, 3, 3, device=device)
    with pytest.raises((RuntimeError, ValueError)):
        fsd.soft_dtw(cost, gamma=gamma, warp_penalty=warp)
