# Copyright (c) 2026 Feiteng Li. Licensed under the Apache License, Version 2.0.
"""Regenerates the README figures in docs/images/ (CPU only, a few seconds).

    python docs/figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

import flash_softdtw as fsd  # noqa: E402

OUT = Path(__file__).resolve().parent / "images"
INK = "#1f2937"
MUTED = "#6b7280"
CYAN = "#0891b2"
ORANGE = "#ea580c"
VIOLET = "#7c3aed"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})


def alignment_figure():
    """Two warped signals and their soft alignment (expected alignment E)."""
    t = np.linspace(0, 1, 60)
    s = np.linspace(0, 1, 80)
    x = np.sin(2 * np.pi * 2 * t ** 1.6) + 0.3 * np.sin(2 * np.pi * 7 * t)
    y = np.sin(2 * np.pi * 2 * s) + 0.3 * np.sin(2 * np.pi * 7 * s ** 0.8)
    xt = torch.tensor(x, dtype=torch.float64).view(1, -1, 1)
    yt = torch.tensor(y, dtype=torch.float64).view(1, -1, 1)
    lens_x, lens_y = torch.tensor([len(x)]), torch.tensor([len(y)])
    panels = []
    for gamma, band in ((1.0, None), (0.05, 6)):
        if band is None:
            cost = fsd.pairwise_cost(xt, yt, "sqeuclidean").requires_grad_()
            value = fsd.soft_dtw(cost, gamma=gamma)
            grad, = torch.autograd.grad(value.sum(), cost)
            e = grad[0].numpy()
        else:
            b = fsd.sakoe_chiba_band(lens_x, lens_y, band)
            cost = fsd.pairwise_cost(xt, yt, "sqeuclidean", band=b).requires_grad_()
            value = fsd.soft_dtw(cost, lens_x, lens_y, band=b, gamma=gamma)
            grad, = torch.autograd.grad(value.sum(), cost)
            e = np.full((len(x), len(y)), np.nan)
            mask = np.zeros_like(e, dtype=bool)
            for i in range(len(x)):
                lo, hi = b[0, i].tolist()
                e[i, lo:hi] = grad[0, i, : hi - lo].numpy()
                mask[i, lo:hi] = True
            panels.append(("band", mask))
        panels.append((gamma, band, e))

    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 3.2], width_ratios=[1, 1, 0.04],
                          hspace=0.08, wspace=0.12)
    items = [p for p in panels if p[0] != "band"]
    mask = [p for p in panels if p[0] == "band"][0][1]
    for col, (gamma, band, e) in enumerate(items):
        top = fig.add_subplot(gs[0, col])
        top.plot(s * (len(y) - 1), y, color=ORANGE, lw=2)
        top.set_xlim(0, len(y) - 1)
        top.axis("off")
        top.set_title(("dense, " if band is None else f"Sakoe-Chiba band (window {band}), ")
                      + f"$\\gamma$ = {gamma}", color=INK, fontsize=12)
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(np.ma.masked_invalid(e), cmap="magma_r", aspect="auto", vmin=0, vmax=1,
                       interpolation="nearest")
        if band is not None:
            ax.contour(mask.astype(float), levels=[0.5], colors=[CYAN], linewidths=1.2)
        if col == 0:
            ax.plot(-6 - 4 * x, np.arange(len(x)), color=CYAN, lw=2, clip_on=False)
        ax.set_xlim(-0.5, len(y) - 0.5)
        ax.set_xlabel("y (S = 80)")
        if col == 0:
            ax.set_ylabel("x (T = 60)", labelpad=34)
        ax.set_yticks([])
        ax.set_xticks([])
    cax = fig.add_subplot(gs[1, 2])
    fig.colorbar(im, cax=cax).set_label("expected alignment  E = ∂value / ∂C")
    fig.savefig(OUT / "soft_alignment.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wavefront_figure():
    """Skewed-strip wavefront: thread k handles column jbase + t - k at tick t."""
    T, S, P = 8, 11, 4  # 8 rows, two strips of 4 rows
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    tick_cmap = plt.get_cmap("viridis")
    for i in range(T):
        strip, k = divmod(i, P)
        for j in range(S):
            t = j + k
            color = tick_cmap(t / (S + P - 2))
            ax.add_patch(Rectangle((j, T - 1 - i), 1, 1, facecolor=color, edgecolor="white", lw=1.5,
                                   alpha=0.95 if strip == 0 else 0.55))
            if strip == 0:
                ax.text(j + 0.5, T - 0.5 - i, str(t), ha="center", va="center", fontsize=8,
                        color="white")
    ax.plot([-0.2, S + 0.2], [T - P, T - P], color=INK, lw=3)
    ax.text(S / 2, T - P - 0.35, "strip boundary: next strip starts from a column buffer",
            color=INK, ha="center", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    for k in range(P):
        ax.text(-0.3, T - 0.5 - k, f"thread {k}", ha="right", va="center", fontsize=9, color=INK)
    j0, i0 = 5, 2
    preds = ((-1, -1, VIOLET, "diag  R(i-1, j-1)   thread k-1, tick t-2"),
             (0, -1, CYAN, "up    R(i-1, j) + w  thread k-1, tick t-1"),
             (-1, 0, ORANGE, "left  R(i, j-1) + w  same thread, tick t-1"))
    for dj, di, c, _ in preds:
        ax.add_patch(Rectangle((j0 + dj + 0.06, T - 1 - (i0 + di) + 0.06), 0.88, 0.88,
                               fill=False, edgecolor=c, lw=3.2))
    ax.add_patch(Rectangle((j0 + 0.04, T - 1 - i0 + 0.04), 0.92, 0.92, fill=False,
                           edgecolor="black", lw=3.2))
    handles = [Rectangle((0, 0), 1, 1, fill=False, edgecolor=c, lw=3) for *_, c, _l in
               [(p[0], p[1], p[2], p[3]) for p in preds]]
    handles.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", lw=3))
    ax.legend(handles, [p[3] for p in preds] + ["cell (i, j) computed at tick t"],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False,
              prop={"family": "DejaVu Sans Mono", "size": 8.5})
    ax.set_xlim(-2.6, S + 0.4)
    ax.set_ylim(-0.3, T + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("cells labelled with the tick that computes them", color=INK)

    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    lines = [
        ("One thread per row, one column per tick", INK, 13, "bold"),
        ("thread k computes R(i0 + k, jbase + t - k) at tick t", MUTED, 10.5, "normal"),
        ("", INK, 6, "normal"),
        ("•  up / diag come from thread k−1 via __shfl_up_sync", INK, 11, "normal"),
        ("•  left is the thread's own previous value", INK, 11, "normal"),
        ("•  one warp = 32 rows: no barrier per tick", INK, 11, "normal"),
        ("•  warp boundary: double-buffered shared memory", INK, 11, "normal"),
        ("•  strip boundary: per-problem column buffer", INK, 11, "normal"),
        ("•  base-2, branch-free softmin (ex2 / lg2)", INK, 11, "normal"),
        ("•  backward replays ticks in reverse and", INK, 11, "normal"),
        ("    pushes expected-alignment messages", INK, 11, "normal"),
    ]
    y = 9.6
    for text, color, size, weight in lines:
        ax.text(0.1, y, text, color=color, fontsize=size, fontweight=weight, va="top")
        y -= 0.85 if size > 7 else 0.4
    fig.savefig(OUT / "wavefront.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def speedup_figure():
    """Measured on an RTX PRO 6000 Blackwell (see the README benchmark table)."""
    labels = ["B=1\n128²", "B=1\n512²", "B=1\n1024²", "B=8\n1024²", "B=64\n512²",
              "B=64\n1024²", "B=256\n128²", "B=256\n512²", "B=256\n1024²"]
    fwd = [56, 231, 365, 345, 252, 300, 55, 142, 167]
    both = [22, 69, 98, 79, 57, 78, 18, 28, 29]
    xpos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(xpos - 0.2, fwd, 0.4, color=CYAN, label="forward")
    ax.bar(xpos + 0.2, both, 0.4, color=ORANGE, label="forward + backward")
    for xp, a, b in zip(xpos, fwd, both):
        ax.text(xp - 0.2, a * 1.08, f"{a}×", ha="center", fontsize=8.5, color=INK)
        ax.text(xp + 0.2, b * 1.08, f"{b}×", ha="center", fontsize=8.5, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(5, 800)
    ax.set_xticks(xpos, labels)
    ax.set_ylabel("speedup vs. row-sequential CUDA baseline")
    ax.set_title("Flash-SoftDTW speedup (dense, fp32, RTX PRO 6000 Blackwell)", color=INK)
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(OUT / "speedup.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    alignment_figure()
    wavefront_figure()
    speedup_figure()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
