"""Independent float64 NumPy dynamic programs in global column coordinates."""

import numpy as np


def dense_problem(cost, x_len, y_len, band=None):
    """Expand compact storage and mask inactive cells with positive infinity."""
    t = max(0, int(x_len))
    s = max(0, int(y_len))
    matrix = np.full((t, s), np.inf, dtype=np.float64)
    # Rows that are not stored stay inactive, so their end cell is unreachable.
    t_stored = min(t, cost.shape[0])
    for i in range(t_stored):
        offset, end = (0, s) if band is None else map(int, band[i])
        for j in range(max(offset, 0), min(end, s, offset + cost.shape[1])):
            matrix[i, j] = cost[i, j - offset]
    return matrix


def soft_dtw_reference(cost, x_lens=None, y_lens=None, *, band=None, gamma=1.0, warp=0.0):
    """Return values and expected alignments in the original storage layout."""
    cost = np.asarray(cost, dtype=np.float64)
    batch, tmax, width = cost.shape
    x_lens = np.full(batch, tmax) if x_lens is None else np.asarray(x_lens)
    y_lens = np.full(batch, width) if y_lens is None else np.asarray(y_lens)
    values = np.full(batch, np.inf)
    gradient = np.zeros_like(cost)
    for b in range(batch):
        window = None if band is None else band[b]
        matrix = dense_problem(cost[b], x_lens[b], y_lens[b], window)
        t, s = matrix.shape
        r = np.full((t + 1, s + 1), np.inf)
        r[0, 0] = 0.0
        for i in range(1, t + 1):
            for j in range(1, s + 1):
                if not np.isfinite(matrix[i - 1, j - 1]):
                    continue
                predecessors = np.array([r[i - 1, j - 1], r[i - 1, j] + warp,
                                         r[i, j - 1] + warp])
                shift = predecessors.min()
                if np.isfinite(shift):
                    r[i, j] = matrix[i - 1, j - 1] + shift - gamma * np.log(
                        np.exp((shift - predecessors) / gamma).sum()
                    )
        values[b] = r[t, s]
        if not np.isfinite(values[b]) or t == 0 or s == 0:
            continue

        # E(i,j) sums contributions from its three possible successor cells.
        e = np.zeros_like(r)
        e[t, s] = 1.0
        for i in range(t, 0, -1):
            for j in range(s, 0, -1):
                if not np.isfinite(r[i, j]):
                    continue
                for ni, nj, step in ((i + 1, j + 1, 0), (i + 1, j, 1), (i, j + 1, 2)):
                    if ni > t or nj > s or e[ni, nj] == 0:
                        continue
                    p = np.array([r[ni - 1, nj - 1], r[ni - 1, nj] + warp,
                                  r[ni, nj - 1] + warp])
                    weights = np.exp((p.min() - p) / gamma)
                    e[i, j] += e[ni, nj] * weights[step] / weights.sum()
                offset = 0 if window is None else int(window[i - 1, 0])
                k = j - 1 - offset
                if 0 <= k < width and np.isfinite(matrix[i - 1, j - 1]):
                    gradient[b, i - 1, k] = e[i, j]
    return values, gradient


def hard_dtw_reference(cost, x_lens=None, y_lens=None, *, band=None, warp=0.0):
    """Compute hard DTW separately, using only min and addition."""
    batch, tmax, width = cost.shape
    xs = [tmax] * batch if x_lens is None else x_lens
    ys = [width] * batch if y_lens is None else y_lens
    result = np.full(batch, np.inf)
    for b in range(batch):
        matrix = dense_problem(cost[b], xs[b], ys[b], None if band is None else band[b])
        t, s = matrix.shape
        table = np.full((t + 1, s + 1), np.inf)
        table[0, 0] = 0
        for i in range(1, t + 1):
            for j in range(1, s + 1):
                table[i, j] = matrix[i - 1, j - 1] + min(
                    table[i - 1, j - 1], table[i - 1, j] + warp,
                    table[i, j - 1] + warp
                )
        result[b] = table[-1, -1]
    return result
