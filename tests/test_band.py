"""Band geometry and gather helpers, checked independently by row and slot."""

import numpy as np
import pytest
import torch

from flash_softdtw import band_columns, band_width, gather_band, sakoe_chiba_band


@pytest.mark.parametrize("lengths", [([7, 5, 9], [7, 9, 5]), ([2, 13, 4], [13, 2, 8])])
@pytest.mark.parametrize("window", [0, 1, 2, 4])
def test_sakoe_chiba_geometry_and_pyts(device, lengths, window):
    xs, ys = lengths
    band = sakoe_chiba_band(torch.tensor(xs, device=device), torch.tensor(ys, device=device), window)
    assert band.shape == (len(xs), max(xs), 2)
    assert band.dtype == torch.int32
    assert band.device == device
    for b, (t, s) in enumerate(zip(xs, ys)):
        rows = band[b, :t].cpu().numpy()
        assert np.all(rows[1:] >= rows[:-1])
        assert np.all(rows[:, 0] >= 0)
        assert np.all(rows[:, 1] <= s)
        assert np.all(rows[:, 0] < rows[:, 1])
        assert rows[0, 0] == 0 and rows[0, 1] > 0
        assert rows[-1, 0] <= s - 1 < rows[-1, 1]

    pyts = pytest.importorskip("pyts.metrics")
    for b, (t, s) in enumerate(zip(xs, ys)):
        expected = pyts.sakoe_chiba_band(t, s, window_size=window).T
        np.testing.assert_array_equal(band[b, :t].cpu().numpy(), expected)


@pytest.mark.parametrize("xs,ys", [
    ([1], [5]),
    ([5], [1]),
    ([1, 3], [1, 4]),
    ([], []),
])
def test_singleton_and_empty_band(device, xs, ys):
    band = sakoe_chiba_band(torch.tensor(xs, device=device, dtype=torch.int64),
                            torch.tensor(ys, device=device, dtype=torch.int64), 1)
    assert band.shape == (len(xs), max(xs, default=0), 2)
    if not xs:
        assert band_width(band) == 0
    for b, (t, s) in enumerate(zip(xs, ys)):
        assert band[b, 0, 0] == 0
        assert band[b, t - 1, 1] == s
        if t == 1:
            assert band[b, 0].tolist() == [0, s]


def test_oversized_window_is_full_band(device):
    xs = torch.tensor([5, 7], device=device)
    ys = torch.tensor([8, 3], device=device)
    band = sakoe_chiba_band(xs, ys, 100)
    for b, t in enumerate(xs.tolist()):
        torch.testing.assert_close(band[b, :t, 0], torch.zeros(t, device=device, dtype=torch.int32))
        torch.testing.assert_close(band[b, :t, 1], ys[b].to(torch.int32).expand(t))


@pytest.mark.parametrize("extra", [(), (3,), (2, 3)])
@pytest.mark.parametrize("width", [0, 1, 5])
def test_band_columns_and_gather(device, extra, width):
    band = torch.tensor([[[0, 2], [1, 4], [3, 4]],
                         [[-1, 2], [2, 4], [4, 4]]], device=device)
    shape = (2, 4, *extra)
    y = torch.arange(int(np.prod(shape)), device=device, dtype=torch.float64).reshape(shape)
    columns = band_columns(band, width, 4)
    gathered = gather_band(y, band, width)
    assert columns.shape == (2, 3, width)
    assert columns.dtype == torch.int64
    assert gathered.shape == (2, 3, width, *extra)
    for b in range(2):
        for i in range(3):
            for k in range(width):
                j = min(max(int(band[b, i, 0]) + k, 0), 3)
                assert columns[b, i, k].item() == j
                torch.testing.assert_close(gathered[b, i, k], y[b, j])
    assert band_width(band) == 3


def test_gather_noncontiguous_source(device):
    y = torch.arange(60, device=device, dtype=torch.float64).reshape(2, 5, 6).transpose(1, 2)
    band = torch.tensor([[[0, 3], [2, 6]], [[1, 4], [3, 6]]], device=device)
    assert not y.is_contiguous()
    torch.testing.assert_close(gather_band(y, band, 4), gather_band(y.contiguous(), band, 4))
