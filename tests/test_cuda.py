"""CUDA parity and strip configurations, including the second device when present."""

import numpy as np
import pytest
import torch

from flash_softdtw import band_width, sakoe_chiba_band, soft_dtw


CUDA_DEVICES = [f"cuda:{i}" for i in range(min(torch.cuda.device_count(), 2))]
if not CUDA_DEVICES:
    CUDA_DEVICES = [pytest.param("cuda:0", marks=pytest.mark.skip(reason="CUDA is unavailable"))]


@pytest.mark.parametrize("cuda_device", CUDA_DEVICES)
@pytest.mark.parametrize("rows", [32, 64, 128, 256])
@pytest.mark.parametrize("banded", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_cpu_cuda_parity_across_strip_heights(cuda_device, rows, banded, dtype, requires_grad):
    rng = np.random.default_rng(2026)
    xs = torch.tensor([269, 257, 131], dtype=torch.int32)
    ys = torch.tensor([277, 259, 139], dtype=torch.int32)
    band = sakoe_chiba_band(xs, ys, 2) if banded else None
    width = band_width(band) if banded else 277
    # A transpose provides non-contiguous cost storage on both devices.
    cpu = torch.tensor(rng.uniform(0, 1, (3, width, 269)), dtype=dtype).transpose(1, 2)
    gpu = cpu.to(cuda_device).detach().requires_grad_(requires_grad)
    cpu = cpu.detach().requires_grad_(requires_grad)
    assert not cpu.is_contiguous() and not gpu.is_contiguous()
    expected = soft_dtw(cpu, xs, ys, band=band, gamma=0.8, warp_penalty=0.25)
    actual = soft_dtw(gpu, xs.to(cuda_device), ys.to(cuda_device),
                      band=None if band is None else band.to(cuda_device),
                      gamma=0.8, warp_penalty=0.25, rows_per_block=rows)
    automatic = soft_dtw(gpu, xs.to(cuda_device), ys.to(cuda_device),
                         band=None if band is None else band.to(cuda_device),
                         gamma=0.8, warp_penalty=0.25)
    assert str(actual.device) == cuda_device
    tolerance = 8e-4 if dtype == torch.float32 else 2e-9
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual, automatic, atol=tolerance, rtol=tolerance)
    if requires_grad:
        weights = torch.tensor([-0.5, 0.25, 1.0], dtype=dtype)
        cpu_grad, = torch.autograd.grad(expected, cpu, weights)
        gpu_grad, = torch.autograd.grad(actual, gpu, weights.to(cuda_device), retain_graph=True)
        auto_grad, = torch.autograd.grad(automatic, gpu, weights.to(cuda_device))
        torch.testing.assert_close(gpu_grad.cpu(), cpu_grad, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(gpu_grad, auto_grad, atol=tolerance, rtol=tolerance)
