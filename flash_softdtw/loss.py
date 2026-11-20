# Copyright (c) 2026 Feiteng Li. Licensed under the Apache License, Version 2.0.
"""Soft-DTW losses on sequences of feature vectors."""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn

from .band import band_width, gather_band, sakoe_chiba_band
from .functional import soft_dtw

Metric = Union[str, Callable[[Tensor, Tensor], Tensor]]

_REDUCTIONS = ("none", "mean", "sum")
_COS_EPS = 1e-12


def _cosine(x: Tensor, y: Tensor) -> Tensor:
    return nn.functional.normalize(x, dim=-1, eps=_COS_EPS), nn.functional.normalize(
        y, dim=-1, eps=_COS_EPS
    )


def _inner(x: Tensor, y: Tensor) -> Tensor:
    """Batched x @ y^T; one-dimensional features use a broadcast product
    (some PyTorch builds route K=1 bmm to a fragile outer-product kernel)."""
    if x.size(-1) == 1:
        return x * y.transpose(1, 2)
    return torch.bmm(x, y.transpose(1, 2))


def _dense_cost(x: Tensor, y: Tensor, metric: Metric) -> Tensor:
    if callable(metric):
        return metric(x.unsqueeze(2), y.unsqueeze(1))
    if metric in ("l1", "euclidean") and x.dtype in (torch.float16, torch.bfloat16):
        # torch.cdist has no half-precision kernels; compute in float32.
        return _dense_cost(x.float(), y.float(), metric).to(x.dtype)
    if metric == "sqeuclidean":
        xx = x.pow(2).sum(-1, keepdim=True)
        yy = y.pow(2).sum(-1).unsqueeze(1)
        return (xx + yy - 2 * _inner(x, y)).clamp_min_(0)
    if metric == "euclidean":
        return torch.cdist(x, y, p=2.0)
    if metric == "l1":
        return torch.cdist(x, y, p=1.0)
    if metric == "cosine":
        xn, yn = _cosine(x, y)
        return 1 - _inner(xn, yn)
    raise ValueError(f"unknown metric {metric!r}")


def _banded_cost(x: Tensor, y: Tensor, band: Tensor, width: int, metric: Metric) -> Tensor:
    yb = gather_band(y, band, width)  # [B, T, W, C]
    xb = x.unsqueeze(2)  # [B, T, 1, C]
    if callable(metric):
        return metric(xb, yb)
    if metric == "sqeuclidean":
        return (xb - yb).pow(2).sum(-1)
    if metric == "euclidean":
        sq = (xb - yb).pow(2).sum(-1)
        # Exact zero distance (matches torch.cdist) without a 0/0 gradient.
        return torch.where(sq > 0, sq.clamp_min(1e-30).sqrt(), torch.zeros_like(sq))
    if metric == "l1":
        return (xb - yb).abs().sum(-1)
    if metric == "cosine":
        xn, yn = _cosine(xb, yb)
        return 1 - (xn * yn).sum(-1)
    raise ValueError(f"unknown metric {metric!r}")


def pairwise_cost(
    x: Tensor,
    y: Tensor,
    metric: Metric = "sqeuclidean",
    band: Optional[Tensor] = None,
    width: Optional[int] = None,
) -> Tensor:
    """Cost matrix between frames of ``x [B, T, C]`` and ``y [B, S, C]``.

    Returns ``[B, T, S]``, or the compact ``[B, T, width]`` layout expected by
    :func:`flash_softdtw.soft_dtw` when ``band`` is given. ``metric`` is one of
    ``"sqeuclidean"``, ``"euclidean"``, ``"l1"``, ``"cosine"`` or a callable
    mapping broadcastable ``[..., C]`` tensors to ``[...]`` costs.
    """
    if band is None:
        return _dense_cost(x, y, metric)
    if width is None:
        width = band_width(band)
    return _banded_cost(x, y, band, width, metric)


class SoftDTWLoss(nn.Module):
    """Soft-DTW alignment loss between two batches of sequences.

    Args:
        gamma: smoothing temperature.
        metric: frame distance (see :func:`pairwise_cost`).
        window: Sakoe-Chiba half width; ``None`` aligns over the full matrix.
        warp_penalty: extra cost of non-diagonal steps.
        normalize: use the soft-DTW divergence
            ``D(x, y) - (D(x, x) + D(y, y)) / 2`` (Blondel et al., 2021),
            which is non-negative and zero iff ``x == y`` for a strictly
            positive-definite metric.
        reduction: ``"none"`` | ``"mean"`` | ``"sum"`` over the batch.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        metric: Metric = "sqeuclidean",
        window: Optional[float] = None,
        warp_penalty: float = 0.0,
        normalize: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in _REDUCTIONS:
            raise ValueError(f"reduction must be one of {_REDUCTIONS}")
        if normalize and window is not None:
            raise ValueError("normalize=True requires window=None (full alignment)")
        self.gamma = float(gamma)
        self.metric = metric
        self.window = window
        self.warp_penalty = float(warp_penalty)
        self.normalize = normalize
        self.reduction = reduction

    def _value(self, x, y, x_lens, y_lens):
        if self.window is None:
            cost = pairwise_cost(x, y, self.metric)
            return soft_dtw(
                cost, x_lens, y_lens, gamma=self.gamma, warp_penalty=self.warp_penalty
            )
        band = sakoe_chiba_band(x_lens, y_lens, self.window)
        cost = pairwise_cost(x[:, : band.size(1)], y, self.metric, band=band)
        return soft_dtw(
            cost,
            x_lens,
            y_lens,
            band=band,
            gamma=self.gamma,
            warp_penalty=self.warp_penalty,
        )

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        x_lens: Optional[Tensor] = None,
        y_lens: Optional[Tensor] = None,
    ) -> Tensor:
        """``x: [B, T, C]``, ``y: [B, S, C]``; optional ``[B]`` lengths."""
        B = x.size(0)
        if x_lens is None:
            x_lens = torch.full((B,), x.size(1), dtype=torch.int32, device=x.device)
        if y_lens is None:
            y_lens = torch.full((B,), y.size(1), dtype=torch.int32, device=x.device)
        x_lens = x_lens.to(x.device)
        y_lens = y_lens.to(x.device)
        out = self._value(x, y, x_lens, y_lens)
        if self.normalize:
            out = out - 0.5 * (
                self._value(x, x, x_lens, x_lens) + self._value(y, y, y_lens, y_lens)
            )
        if self.reduction == "mean":
            return out.mean()
        if self.reduction == "sum":
            return out.sum()
        return out
