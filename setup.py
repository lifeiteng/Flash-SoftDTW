# Copyright (c) 2026 Feiteng Li. Licensed under the Apache License, Version 2.0.
"""Build script for the native extension.

Environment variables:
    FLASH_SOFTDTW_FORCE_CUDA=1   build CUDA kernels even if torch reports no CUDA
    FLASH_SOFTDTW_CPU_ONLY=1     skip the CUDA kernels
    TORCH_CUDA_ARCH_LIST         target architectures, e.g. "8.0;9.0;12.0"
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension


def _extension():
    import torch

    sources = ["csrc/bindings.cpp", "csrc/softdtw_cpu.cpp"]
    cxx = ["-O3"]
    # Build the CUDA kernels whenever a CUDA toolkit and a CUDA build of torch
    # are present, even without a visible GPU (containers, CI, login nodes).
    want_cuda = (
        not os.environ.get("FLASH_SOFTDTW_CPU_ONLY")
        and CUDA_HOME is not None
        and torch.version.cuda is not None
    ) or bool(os.environ.get("FLASH_SOFTDTW_FORCE_CUDA"))
    if want_cuda:
        nvcc = ["-O3", "-lineinfo", "-DFLASH_SOFTDTW_WITH_CUDA"]
        return CUDAExtension(
            "flash_softdtw._C",
            sources + ["csrc/softdtw_cuda.cu"],
            extra_compile_args={"cxx": cxx + ["-DFLASH_SOFTDTW_WITH_CUDA"], "nvcc": nvcc},
            depends=["csrc/softdtw.h"],
        )
    return CppExtension(
        "flash_softdtw._C", sources, extra_compile_args={"cxx": cxx}, depends=["csrc/softdtw.h"]
    )


setup(
    ext_modules=[_extension()],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
