# Copyright (c) 2026 Feiteng Li. Licensed under the Apache License, Version 2.0.
"""Differentiable soft-DTW on precomputed cost matrices."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from ._ext import ext

Band = Union[Tensor, Tuple[Tensor, Tensor]]


def _split_band(band: Optional[Band]) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    if band is None:
        return None, None
    if isinstance(band, (tuple, list)):
        begin, end = band
    elif band.is_floating_point() or band.is_complex():
        raise TypeError("band must be an integer tensor")
    else:
        if band.dim() != 3 or band.size(-1) != 2:
            raise ValueError("band must have shape [B, T, 2] or be a (begin, end) pair")
        begin, end = band[..., 0], band[..., 1]
    return begin, end


class _SoftDTW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cost, x_lens, y_lens, begin, end, gamma, warp, rows_per_block, save_r):
        values, r = ext().forward(
            cost, x_lens, y_lens, begin, end, gamma, warp, save_r, rows_per_block
        )
        if save_r:
            # The CUDA backward never reads the cost values, only its shape and
            # dtype: keep it out of the graph so it can be freed early.
            ctx.cost_meta = None
            if cost.is_cuda:
                ctx.cost_meta = (cost.shape, cost.dtype, cost.device)
                cost = None
            ctx.save_for_backward(cost, r, x_lens, y_lens, begin, end)
        ctx.gamma = gamma
        ctx.warp = warp
        ctx.rows_per_block = rows_per_block
        return values

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_values):
        cost, r, x_lens, y_lens, begin, end = ctx.saved_tensors
        if cost is None:
            shape, dtype, device = ctx.cost_meta
            # Zero-storage placeholder that carries shape and dtype only.
            cost = torch.empty((), dtype=dtype, device=device).expand(shape)
        grad = ext().backward(
            cost,
            r,
            x_lens,
            y_lens,
            begin,
            end,
            grad_values,
            ctx.gamma,
            ctx.warp,
            ctx.rows_per_block,
        )
        return grad, None, None, None, None, None, None, None, None


def soft_dtw(
    cost: Tensor,
    x_lens: Optional[Tensor] = None,
    y_lens: Optional[Tensor] = None,
    *,
    band: Optional[Band] = None,
    gamma: float = 1.0,
    warp_penalty: float = 0.0,
    rows_per_block: int = 0,
) -> Tensor:
    r"""Soft-DTW value of a batch of cost matrices.

    Computes, for every batch element,

    .. math::
        R_{i,j} = C_{i,j} + \mathrm{softmin}_\gamma\big(R_{i-1,j-1},\,
        R_{i-1,j} + w,\, R_{i,j-1} + w\big)

    and returns :math:`R_{T-1,S-1}`, where :math:`w` is ``warp_penalty``.

    Args:
        cost: ``[B, T, S]`` dense cost matrix, or ``[B, T, W]`` compact band
            storage when ``band`` is given (``cost[b, i, k]`` is the cost of
            cell ``(i, band_begin[b, i] + k)``). Any floating dtype; half and
            bfloat16 are accumulated in float32.
        x_lens: ``[B]`` valid rows per element (defaults to ``T``).
        y_lens: ``[B]`` valid columns per element (defaults to ``S``; required
            in banded mode).
        band: optional ``[B, T, 2]`` tensor of ``[begin, end)`` column ranges
            per row (e.g. from :func:`flash_softdtw.sakoe_chiba_band`), or a
            ``(begin, end)`` pair of ``[B, T]`` tensors. Ranges must be
            non-decreasing along ``T``.
        gamma: smoothing temperature (> 0); ``gamma -> 0`` recovers hard DTW.
        warp_penalty: extra cost for horizontal / vertical (non-diagonal) steps.
        rows_per_block: CUDA strip height override (multiple of 32); ``0``
            picks it automatically.

    Only first-order gradients are supported. Internally the recursion runs in
    base-2 units scaled by ``1 / gamma`` with a finite sentinel for
    unreachable cells, so accumulated values must satisfy
    ``|R| * log2(e) / gamma < 1e29`` (float32; ``1e299`` for float64).

    Returns:
        ``[B]`` soft-DTW values (float32 for half / bfloat16 inputs). Elements
        whose end cell lies outside the band are ``+inf``.
    """
    if cost.dim() != 3:
        raise ValueError(f"cost must be 3-D [B, T, S], got shape {tuple(cost.shape)}")
    if not (math.isfinite(gamma) and gamma > 0):
        raise ValueError("gamma must be finite and positive")
    if not math.isfinite(warp_penalty):
        raise ValueError("warp_penalty must be finite")
    for name, t in (("x_lens", x_lens), ("y_lens", y_lens)):
        if t is not None and (t.is_floating_point() or t.is_complex()):
            raise TypeError(f"{name} must be an integer tensor")
    B, T, S = cost.shape
    begin, end = _split_band(band)
    if x_lens is None:
        x_lens = torch.full((B,), T, dtype=torch.int32, device=cost.device)
    if y_lens is None:
        if begin is not None:
            raise ValueError("y_lens is required in banded mode")
        y_lens = torch.full((B,), S, dtype=torch.int32, device=cost.device)
    x_lens = x_lens.to(device=cost.device, dtype=torch.int32)
    y_lens = y_lens.to(device=cost.device, dtype=torch.int32)
    if begin is not None:
        begin = begin.to(device=cost.device, dtype=torch.int32)
        end = end.to(device=cost.device, dtype=torch.int32)
    # Decided here: inside Function.forward grad mode is always disabled, and
    # ctx.needs_input_grad ignores an enclosing torch.no_grad().
    save_r = torch.is_grad_enabled() and cost.requires_grad
    return _SoftDTW.apply(
        cost,
        x_lens,
        y_lens,
        begin,
        end,
        float(gamma),
        float(warp_penalty),
        int(rows_per_block),
        save_r,
    )
