# Copyright (c) 2026 Feiteng Li. Licensed under the Apache License, Version 2.0.
"""Warping-window (band) helpers."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sakoe_chiba_band(x_lens: Tensor, y_lens: Tensor, window: float) -> Tensor:
    """Sakoe-Chiba band along the (possibly non-square) diagonal.

    Row ``i`` of element ``b`` keeps the columns within ``window`` steps of the
    straight line from ``(0, 0)`` to ``(x_lens[b] - 1, y_lens[b] - 1)``. The
    effective window is widened to at least half the slope so that every
    band is connected (the same convention as ``pyts.metrics.sakoe_chiba_band``).

    Args:
        x_lens: ``[B]`` integer lengths of the first sequences.
        y_lens: ``[B]`` integer lengths of the second sequences.
        window: half width of the band, in steps of the longer sequence.

    Returns:
        ``[B, max(x_lens), 2]`` int32 tensor of ``[begin, end)`` columns. Rows
        beyond ``x_lens[b]`` are filled but never read.
    """
    if window < 0:
        raise ValueError("window must be non-negative")
    device = x_lens.device
    n = x_lens.to(torch.float64)
    m = y_lens.to(device=device, dtype=torch.float64)
    T = int(x_lens.max().item()) if x_lens.numel() else 0
    # Slope of the diagonal; a single row sees every column.
    slope = torch.where(n > 1, (m - 1) / (n - 1).clamp_min(1), torch.full_like(n, math.inf))
    tall = m > n  # more columns than rows
    radius = torch.maximum(
        torch.minimum(torch.maximum(n, m), torch.full_like(n, float(window))),
        torch.where(tall, slope / 2, 0.5 / slope),
    )
    rows = torch.arange(T, device=device, dtype=torch.float64).unsqueeze(0)
    slope_c = torch.where(torch.isinf(slope), torch.zeros_like(slope), slope).unsqueeze(1)
    h = torch.where(tall, torch.zeros_like(radius), radius).unsqueeze(1)
    v = torch.where(tall, radius, torch.zeros_like(radius)).unsqueeze(1)
    lo = torch.ceil(slope_c * (rows - h) - v)
    hi = torch.floor(slope_c * (rows + h) + v) + 1
    # A single row or a single column sees every column.
    single = (torch.isinf(slope) | (m <= 1)).unsqueeze(1)
    lo = torch.where(single, torch.zeros_like(lo), lo)
    hi = torch.where(single, m.unsqueeze(1).expand_as(hi), hi)
    upper = m.unsqueeze(1)
    lo = torch.minimum(lo.clamp_min(0), upper)
    hi = torch.minimum(hi.clamp_min(0), upper)
    return torch.stack([lo, hi], dim=-1).to(torch.int32)


def band_width(band: Tensor) -> int:
    """Number of compact columns needed to store ``band``."""
    if band.numel() == 0:
        return 0
    return int((band[..., 1] - band[..., 0]).max().clamp_min(0).item())


def band_columns(band: Tensor, width: int, y_len_max: int) -> Tensor:
    """Global column index of every compact slot, clipped to valid columns.

    Returns a ``[B, T, width]`` int64 tensor ``idx`` with
    ``idx[b, i, k] = min(band[b, i, 0] + k, y_len_max - 1)``.
    """
    k = torch.arange(width, device=band.device, dtype=torch.int64)
    idx = band[..., :1].to(torch.int64) + k
    return idx.clamp_(0, max(y_len_max - 1, 0))


def gather_band(y: Tensor, band: Tensor, width: int) -> Tensor:
    """Gathers ``y[b, band_begin[b, i] + k]`` into a ``[B, T, width, ...]`` tensor."""
    B, S = y.shape[:2]
    idx = band_columns(band, width, S)  # [B, T, W]
    flat = idx.reshape(B, -1)
    extra = y.shape[2:]
    gathered = torch.gather(
        y, 1, flat.view(B, -1, *([1] * len(extra))).expand(B, flat.size(1), *extra)
    )
    return gathered.view(B, idx.size(1), width, *extra)
