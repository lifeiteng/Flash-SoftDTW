# Copyright (c) 2026 Feiteng Li. Licensed under the Apache License, Version 2.0.
"""Loader for the native extension.

The prebuilt module ``flash_softdtw._C`` is used when available. Otherwise the
sources are JIT-compiled once with ``torch.utils.cpp_extension.load`` (set
``FLASH_SOFTDTW_NO_JIT=1`` to disable this fallback).
"""

from __future__ import annotations

import os
from pathlib import Path

_ext = None


def _jit_build():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    src = Path(__file__).resolve().parent.parent / "csrc"
    if not src.is_dir():
        raise ImportError(
            "flash_softdtw._C is not built and the csrc/ directory is missing; "
            "reinstall the package with `pip install .`"
        )
    sources = [str(src / "bindings.cpp"), str(src / "softdtw_cpu.cpp")]
    cflags = ["-O3"]
    cuda_flags = []
    with_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    if with_cuda:
        sources.append(str(src / "softdtw_cuda.cu"))
        cflags.append("-DFLASH_SOFTDTW_WITH_CUDA")
        cuda_flags = ["-O3", "-DFLASH_SOFTDTW_WITH_CUDA", "-lineinfo"]
    return load(
        name="flash_softdtw_jit",
        sources=sources,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_flags,
        verbose=bool(os.environ.get("FLASH_SOFTDTW_VERBOSE")),
    )


def ext():
    """Returns the native module, building it on first use if necessary."""
    global _ext
    if _ext is None:
        try:
            from . import _C  # type: ignore[attr-defined]

            _ext = _C
        except ImportError:
            if os.environ.get("FLASH_SOFTDTW_NO_JIT"):
                raise
            _ext = _jit_build()
    return _ext
