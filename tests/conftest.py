"""Shared device and resource configuration."""

import os

import pytest
import torch


os.environ.setdefault("MAX_JOBS", "4")

DEVICES = ["cpu"] + [f"cuda:{i}" for i in range(min(torch.cuda.device_count(), 2))]


@pytest.fixture(params=DEVICES)
def device(request):
    return torch.device(request.param)


@pytest.fixture(params=[1, 4])
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)
