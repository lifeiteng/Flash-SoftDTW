"""Value, gradient, storage, and path invariants for both backends."""

import numpy as np
import pytest
import torch

from flash_softdtw import band_width, pairwise_cost, sakoe_chiba_band, soft_dtw
from flash_softdtw._ext import ext

from ._reference import hard_dtw_reference, soft_dtw_reference


PARAMETERS = [(0.01, 0.0), (0.1, 5.0), (1.0, 50.0)]


def numpy(tensor):
    return tensor.detach().cpu().double().numpy()


def check_reference(cost, xs=None, ys=None, band=None, gamma=1.0, warp=0.0):
    """Check values and gradients with nontrivial per-element upstream weights."""
    expected, grad = soft_dtw_reference(
        numpy(cost), None if xs is None else numpy(xs), None if ys is None else numpy(ys),
        band=None if band is None else numpy(band), gamma=gamma, warp=warp
    )
    value = soft_dtw(cost, xs, ys, band=band, gamma=gamma, warp_penalty=warp)
    weights = torch.linspace(-0.75, 1.25, cost.shape[0], device=cost.device, dtype=value.dtype)
    actual_grad, = torch.autograd.grad(value, cost, weights)
    tolerance = 2e-9 if cost.dtype == torch.float64 else 3e-4
    np.testing.assert_allclose(numpy(value), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(numpy(actual_grad), grad * numpy(weights)[:, None, None],
                               rtol=tolerance, atol=tolerance)
    assert actual_grad.dtype == cost.dtype
    assert actual_grad.shape == cost.shape
    assert torch.isfinite(actual_grad).all()
    return value.detach(), actual_grad.detach()


def compact_cost(dense, band, width=None):
    """Pack global columns explicitly, independently of the library helpers."""
    width = band_width(band) if width is None else width
    result = dense.new_full((*dense.shape[:2], width), 123.0)
    for b in range(dense.shape[0]):
        for i in range(dense.shape[1]):
            lo, hi = band[b, i].tolist()
            for k in range(width):
                j = lo + k
                if 0 <= j < dense.shape[2] and j < hi:
                    result[b, i, k] = dense[b, i, j]
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_manual_multidimensional_l1(device, dtype):
    x = torch.tensor([[[0, 1], [2, 1]]], dtype=dtype, device=device)
    y = torch.tensor([[[0, 0], [1, 1], [2, 2]]], dtype=dtype, device=device)
    cost = pairwise_cost(x, y, "l1").requires_grad_()
    torch.testing.assert_close(cost, torch.tensor([[[1, 1, 3], [3, 1, 1]]],
                                                 dtype=dtype, device=device))
    check_reference(cost, gamma=0.7, warp=0.3)


@pytest.mark.parametrize("gamma,warp", PARAMETERS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [7, 91])
def test_random_dense_variable_lengths(device, gamma, warp, dtype, seed):
    rng = np.random.default_rng(seed)
    cost = torch.tensor(rng.normal(size=(4, 7, 8)), dtype=dtype,
                        device=device, requires_grad=True)
    xs = torch.tensor([7, 5, 2, 6], device=device)
    ys = torch.tensor([8, 3, 7, 4], device=device)
    check_reference(cost, xs, ys, gamma=gamma, warp=warp)


@pytest.mark.parametrize("band_mode", [False, True])
def test_end_cell_outside_storage_is_unreachable(device, band_mode):
    cost = torch.rand(3, 4, 5, dtype=torch.float64, device=device, requires_grad=True)
    xs = torch.tensor([4, 6, 2], device=device)  # row 6 is not stored
    ys = torch.tensor([5, 3, 9], device=device)  # column 9 is not stored
    band = None
    if band_mode:
        begin = torch.zeros(3, 4, dtype=torch.int32, device=device)
        band = (begin, torch.full_like(begin, 9))
    value = soft_dtw(cost, xs, ys, band=band)
    assert torch.isfinite(value[0])
    assert torch.isinf(value[1]) and value[1] > 0
    if not band_mode:
        assert torch.isinf(value[2]) and value[2] > 0
    grad, = torch.autograd.grad(value[:2].sum(), cost)
    assert torch.count_nonzero(grad[1]) == 0


@pytest.mark.parametrize("gamma,warp", PARAMETERS)
@pytest.mark.parametrize("window", [0, 2, 20])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_banded_values_and_compact_gradients(device, gamma, warp, window, dtype):
    rng = np.random.default_rng(128)
    xs = torch.tensor([9, 5, 8], device=device)
    ys = torch.tensor([7, 10, 6], device=device)
    band = sakoe_chiba_band(xs, ys, window)
    dense = torch.tensor(rng.uniform(-0.5, 2, (3, 9, 10)), dtype=dtype, device=device)
    cost = compact_cost(dense, band).requires_grad_()
    check_reference(cost, xs, ys, band, gamma, warp)


@pytest.mark.parametrize("gamma,warp", [(0.1, 5.0), (1.0, 0.0)])
def test_large_batch_narrow_band(device, cpu_threads, gamma, warp):
    rng = np.random.default_rng(500)
    batch, t = 503, 17
    xs = torch.tensor(rng.integers(12, t + 1, batch), device=device)
    ys = torch.tensor(rng.integers(12, t + 1, batch), device=device)
    xs[0] = t
    band = sakoe_chiba_band(xs, ys, 1)
    cost = torch.tensor(rng.uniform(0, 1, (batch, t, band_width(band))),
                        dtype=torch.float64, device=device, requires_grad=True)
    value = soft_dtw(cost, xs, ys, band=band, gamma=gamma, warp_penalty=warp)
    gradient, = torch.autograd.grad(value.sum(), cost)
    sample = np.array([0, 1, 31, 32, 127, 255, 499, 502])
    expected, expected_grad = soft_dtw_reference(
        numpy(cost)[sample], numpy(xs)[sample], numpy(ys)[sample],
        band=numpy(band)[sample], gamma=gamma, warp=warp
    )
    np.testing.assert_allclose(numpy(value)[sample], expected, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(numpy(gradient)[sample], expected_grad, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("banded", [False, True])
def test_gradcheck(device, banded):
    rng = np.random.default_rng(72)
    xs = torch.tensor([3, 2], device=device)
    ys = torch.tensor([4, 3], device=device)
    band = sakoe_chiba_band(xs, ys, 1) if banded else None
    width = band_width(band) if banded else 4
    cost = torch.tensor(rng.uniform(0.1, 1, (2, 3, width)), dtype=torch.float64,
                        device=device, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda c: soft_dtw(c, xs, ys, band=band, gamma=0.7, warp_penalty=0.4),
        (cost,), eps=1e-6, atol=2e-5, rtol=1e-4
    )


def test_numpy_reference_gradient_by_finite_difference():
    rng = np.random.default_rng(15)
    cost = rng.uniform(0, 1, (1, 3, 4))
    band = np.array([[[0, 2], [0, 3], [1, 4]]])
    for window in (None, band):
        _, gradient = soft_dtw_reference(cost, [3], [4], band=window, gamma=0.4, warp=0.8)
        finite_difference = np.zeros_like(cost)
        for index in np.ndindex(cost.shape):
            plus, minus = cost.copy(), cost.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            vp, _ = soft_dtw_reference(plus, [3], [4], band=window, gamma=0.4, warp=0.8)
            vm, _ = soft_dtw_reference(minus, [3], [4], band=window, gamma=0.4, warp=0.8)
            finite_difference[index] = (vp[0] - vm[0]) / 2e-6
        np.testing.assert_allclose(gradient, finite_difference, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("gamma,warp", PARAMETERS)
def test_wide_band_equals_dense_and_batch_permutation(device, gamma, warp):
    rng = np.random.default_rng(123)
    xs = torch.tensor([6, 4, 5], device=device)
    ys = torch.tensor([5, 7, 3], device=device)
    cost = torch.tensor(rng.uniform(0, 1, (3, 6, 7)), dtype=torch.float64,
                        device=device, requires_grad=True)
    band = torch.tensor([0, 7], device=device).expand(3, 6, 2)
    dense, dense_grad = check_reference(cost, xs, ys, gamma=gamma, warp=warp)
    banded, band_grad = check_reference(cost, xs, ys, band, gamma, warp)
    torch.testing.assert_close(dense, banded)
    torch.testing.assert_close(dense_grad, band_grad)
    order = torch.tensor([2, 0, 1], device=device)
    perm_cost = cost.detach()[order].requires_grad_()
    perm_value = soft_dtw(perm_cost, xs[order], ys[order], band=band[order],
                          gamma=gamma, warp_penalty=warp)
    orig_value = soft_dtw(cost, xs, ys, gamma=gamma, warp_penalty=warp)
    orig_grad, = torch.autograd.grad(orig_value.sum(), cost)
    perm_grad, = torch.autograd.grad(perm_value.sum(), perm_cost)
    torch.testing.assert_close(perm_value, orig_value[order])
    torch.testing.assert_close(perm_grad, orig_grad[order])


@pytest.mark.parametrize("banded", [False, True])
def test_padding_does_not_change_value_or_gradient(device, banded):
    rng = np.random.default_rng(37)
    xs = torch.tensor([5, 2, 4], device=device)
    ys = torch.tensor([3, 6, 4], device=device)
    band = sakoe_chiba_band(xs, ys, 2) if banded else None
    width = band_width(band) if banded else 6
    array = rng.uniform(0, 1, (3, 5, width))
    padded = array.copy()
    for b, i, k in np.ndindex(array.shape):
        lo, hi = (0, 6) if band is None else band[b, i].tolist()
        j = lo + k
        if i >= xs[b].item() or j < 0 or j >= ys[b].item() or j >= hi:
            padded[b, i, k] = rng.choice([np.nan, np.inf, -1e20, 1e20])
    cost = torch.tensor(array, device=device, dtype=torch.float64, requires_grad=True)
    changed = torch.tensor(padded, device=device, dtype=torch.float64, requires_grad=True)
    expected = check_reference(cost, xs, ys, band)
    actual = check_reference(changed, xs, ys, band)
    for first, second in zip(expected, actual):
        torch.testing.assert_close(first, second)


@pytest.mark.parametrize("banded", [False, True])
@pytest.mark.parametrize("warp", [0.0, 5.0])
def test_small_gamma_approaches_hard_dtw(device, banded, warp):
    rng = np.random.default_rng(444)
    xs = torch.tensor([5, 4], device=device)
    ys = torch.tensor([6, 5], device=device)
    band = sakoe_chiba_band(xs, ys, 2) if banded else None
    width = band_width(band) if banded else 6
    array = rng.uniform(0, 2, (2, 5, width))
    cost = torch.tensor(array, device=device, dtype=torch.float64)
    hard = hard_dtw_reference(array, numpy(xs), numpy(ys),
                              band=None if band is None else numpy(band), warp=warp)
    value = soft_dtw(cost, xs, ys, band=band, gamma=1e-6, warp_penalty=warp)
    np.testing.assert_allclose(numpy(value), hard, atol=2e-5, rtol=0)
    assert np.all(numpy(value) <= hard + 1e-10)


@pytest.mark.parametrize("mode", ["no_grad", "requires_grad_false", "inference"])
@pytest.mark.parametrize("banded", [False, True])
def test_without_gradients(device, mode, banded):
    rng = np.random.default_rng(98)
    xs = torch.tensor([5, 3], device=device)
    ys = torch.tensor([4, 5], device=device)
    band = sakoe_chiba_band(xs, ys, 1) if banded else None
    width = band_width(band) if banded else 5
    cost = torch.tensor(rng.uniform(0, 1, (2, 5, width)), dtype=torch.float64,
                        device=device, requires_grad=(mode == "no_grad"))
    expected, _ = soft_dtw_reference(numpy(cost), numpy(xs), numpy(ys),
                                    band=None if band is None else numpy(band))
    context = torch.inference_mode() if mode == "inference" else torch.no_grad()
    if mode == "requires_grad_false":
        value = soft_dtw(cost, xs, ys, band=band)
    else:
        with context:
            value = soft_dtw(cost, xs, ys, band=band)
    assert not value.requires_grad
    np.testing.assert_allclose(numpy(value), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("banded", [False, True])
@pytest.mark.parametrize("layout", ["transpose", "slice", "expand"])
def test_arbitrary_cost_strides(device, banded, layout):
    rng = np.random.default_rng(432)
    raw = torch.tensor(rng.uniform(0, 1, (2, 7, 12)), device=device, dtype=torch.float64)
    if layout == "transpose":
        cost = raw.transpose(1, 2)
    elif layout == "slice":
        cost = raw[:, 1::2, 1::2]
    else:
        cost = raw[:1].expand(3, -1, -1)
    cost = cost.detach().requires_grad_()
    assert not cost.is_contiguous()
    batch, t, width = cost.shape
    xs = torch.full((batch,), t, device=device)
    ys = torch.full((batch,), width, device=device)
    band = None
    if banded:
        band = sakoe_chiba_band(xs, ys, 2)
    actual, grad = check_reference(cost, xs, ys, band, gamma=0.8, warp=0.5)
    contiguous = cost.detach().contiguous().requires_grad_()
    expected, expected_grad = check_reference(contiguous, xs, ys, band, gamma=0.8, warp=0.5)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(grad, expected_grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("banded", [False, True])
def test_dtypes_and_accumulation(device, dtype, banded):
    if device.type == "cuda" and dtype == torch.bfloat16:
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                pytest.skip("bfloat16 is not supported by this CUDA device")
    rng = np.random.default_rng(609)
    xs = torch.tensor([11, 8], device=device)
    ys = torch.tensor([9, 10], device=device)
    band = sakoe_chiba_band(xs, ys, 2) if banded else None
    width = band_width(band) if banded else 10
    cost = torch.tensor(rng.uniform(0, 1, (2, 11, width)), device=device,
                        dtype=dtype, requires_grad=True)
    low = soft_dtw(cost, xs, ys, band=band, gamma=0.8, warp_penalty=0.2)
    low_grad, = torch.autograd.grad(low.sum(), cost)
    accumulation = torch.float64 if dtype == torch.float64 else torch.float32
    assert low.dtype == accumulation
    begin, end = (band[..., 0], band[..., 1]) if banded else (None, None)
    native_values, r = ext().forward(cost.detach(), xs, ys, begin, end, 0.8, 0.2, True, 0)
    assert r.shape == cost.shape and r.dtype == accumulation
    torch.testing.assert_close(native_values, low)
    full_cost = cost.detach().to(accumulation).requires_grad_()
    full = soft_dtw(full_cost, xs, ys, band=band, gamma=0.8, warp_penalty=0.2)
    full_grad, = torch.autograd.grad(full.sum(), full_cost)
    torch.testing.assert_close(low, full, atol=2e-5, rtol=2e-5)
    tolerance = 8e-3 if dtype == torch.bfloat16 else 1e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(low_grad.to(accumulation), full_grad, atol=tolerance, rtol=tolerance)
    assert low_grad.dtype == dtype
    check_reference(full_cost, xs, ys, band, gamma=0.8, warp=0.2)


@pytest.mark.parametrize("case", ["origin_missing", "endpoint_missing", "gap", "short_storage",
                                  "negative_begin", "infinite_cost", "dense_endpoint_past_width"])
def test_unreachable_and_clipped_band(device, case):
    array = np.arange(12, dtype=np.float64).reshape(1, 3, 4) / 10
    band = np.array([[[0, 4], [0, 4], [0, 4]]])
    ys = torch.tensor([4], device=device)
    if case == "origin_missing":
        band[0, :, 0] = 1
    elif case == "endpoint_missing":
        band[0, :, 1] = 3
    elif case == "gap":
        band[0] = [[0, 1], [2, 3], [3, 4]]
    elif case == "short_storage":
        band[0] = [[0, 100], [0, 100], [1, 100]]
        array = array[..., :3]
    elif case == "negative_begin":
        band[0] = [[-2, 1], [-1, 3], [0, 8]]
    elif case == "infinite_cost":
        array[0, 1] = np.inf
    else:
        band = None
        ys = torch.tensor([6], device=device)
    window = None if band is None else torch.tensor(band, device=device)
    cost = torch.tensor(array, device=device, requires_grad=True)
    value, grad = check_reference(cost, None, ys, window, gamma=0.1, warp=5.0)
    if case not in ("short_storage", "negative_begin"):
        assert torch.isposinf(value).all()
        assert torch.count_nonzero(grad) == 0


@pytest.mark.parametrize("shape,xs,ys", [
    ((3, 3, 4), [0, 0, 2], [0, 2, 0]),
    ((2, 0, 4), [0, 9], [0, 2]),
    ((2, 3, 0), [0, 2], [0, 2]),
    ((0, 3, 4), [], []),
])
@pytest.mark.parametrize("banded", [False, True])
def test_empty_lengths_and_empty_storage(device, shape, xs, ys, banded):
    cost = torch.zeros(shape, dtype=torch.float64, device=device, requires_grad=True)
    x = torch.tensor(xs, dtype=torch.int32, device=device)
    y = torch.tensor(ys, dtype=torch.int64, device=device)
    band = torch.tensor([0, 4], device=device).expand(shape[0], shape[1], 2) if banded else None
    check_reference(cost, x, y, band)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("save_r", [False, True])
@pytest.mark.parametrize("banded", [False, True])
def test_native_metadata_strides_and_saved_r(device, index_dtype, save_r, banded):
    rng = np.random.default_rng(81)
    cost = torch.tensor(rng.uniform(0, 1, (2, 4, 5)), device=device, dtype=torch.float64)
    xs = torch.tensor([6, -99, 2, -99], dtype=index_dtype, device=device)[::2]
    ys = torch.tensor([5, -99, 3, -99], dtype=index_dtype, device=device)[::2]
    band = torch.tensor([0, 5], dtype=index_dtype, device=device).expand(2, 4, 2)
    begin, end = (band[..., 0], band[..., 1]) if banded else (None, None)
    values, r = ext().forward(cost, xs, ys, begin, end, 0.7, 0.3, save_r, 0)
    expected, grad = soft_dtw_reference(numpy(cost), numpy(xs), numpy(ys),
                                       band=numpy(band) if banded else None, gamma=0.7, warp=0.3)
    np.testing.assert_allclose(numpy(values), expected, atol=1e-10, rtol=1e-10)
    if save_r:
        assert r.shape == cost.shape
        assert r.dtype == torch.float64
        weights = torch.tensor([2.0, -99, -0.5, -99], device=device, dtype=cost.dtype)[::2]
        actual_grad = ext().backward(cost, r, xs, ys, begin, end, weights, 0.7, 0.3, 0)
        np.testing.assert_allclose(numpy(actual_grad), grad * numpy(weights)[:, None, None],
                                   atol=1e-10, rtol=1e-10)
    else:
        assert r is None or r.numel() == 0


def test_band_pair_matches_stacked_band(device):
    cost = torch.arange(24, device=device, dtype=torch.float64).reshape(2, 3, 4) / 10
    xs = torch.tensor([3, 2], device=device)
    ys = torch.tensor([4, 3], device=device)
    band = sakoe_chiba_band(xs, ys, 2)
    first = soft_dtw(cost, xs, ys, band=band)
    second = soft_dtw(cost, xs, ys, band=(band[..., 0], band[..., 1]))
    torch.testing.assert_close(first, second)
