# Copyright (c) 2026 Feiteng Li. Licensed under the Apache License, Version 2.0.
"""Flash-SoftDTW: fast, memory-lean soft-DTW for PyTorch."""

from .band import band_columns, band_width, gather_band, sakoe_chiba_band
from .functional import soft_dtw
from .loss import SoftDTWLoss, pairwise_cost

__version__ = "0.1.0"

__all__ = [
    "SoftDTWLoss",
    "band_columns",
    "band_width",
    "gather_band",
    "pairwise_cost",
    "sakoe_chiba_band",
    "soft_dtw",
]
