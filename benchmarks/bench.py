# Copyright (c) 2026 Feiteng Li. Licensed under the Apache License, Version 2.0.
"""Timing and memory of flash_softdtw.soft_dtw on dense and banded problems.

Cost construction is excluded from timing. Each entry is the median of
``--repeats`` CUDA-event measurements after ``--warmup`` calls.

Example:
    python benchmarks/bench.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

import flash_softdtw as fsd


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def peak_mib(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def make_dense(B, T, gen):
    cost = torch.rand(B, T, T, generator=gen, device="cuda")
    return cost, None, None, None


def make_banded(B, T, S, window, gen):
    x_lens = torch.randint(T // 2, T + 1, (B,), generator=gen, device="cuda")
    y_lens = torch.randint(S // 2, S + 1, (B,), generator=gen, device="cuda")
    x_lens[0], y_lens[0] = T, S
    band = fsd.sakoe_chiba_band(x_lens, y_lens, window)
    x = torch.randn(B, T, 16, generator=gen, device="cuda")
    y = torch.randn(B, S, 16, generator=gen, device="cuda")
    cost = fsd.pairwise_cost(x, y, "l1", band=band).contiguous()
    return cost, x_lens, y_lens, band


def run(name, cost, x_lens, y_lens, band, args):
    leaf = cost.detach().requires_grad_()

    def forward():
        with torch.no_grad():
            fsd.soft_dtw(cost, x_lens, y_lens, band=band, gamma=args.gamma,
                         warp_penalty=args.warp_penalty)

    def forward_backward():
        value = fsd.soft_dtw(leaf, x_lens, y_lens, band=band, gamma=args.gamma,
                             warp_penalty=args.warp_penalty)
        torch.autograd.grad(value.sum(), leaf)

    row = {
        "case": name,
        "fwd_ms": measure(forward, args.warmup, args.repeats),
        "fwd_bwd_ms": measure(forward_backward, args.warmup, args.repeats),
        "fwd_peak_mib": peak_mib(forward),
        "fwd_bwd_peak_mib": peak_mib(forward_backward),
    }
    print(f"| {name} | {row['fwd_ms']:.3f} | {row['fwd_bwd_ms']:.3f} | "
          f"{row['fwd_peak_mib']:.1f} | {row['fwd_bwd_peak_mib']:.1f} |", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", default="1,8,64,256")
    parser.add_argument("--lengths", default="128,256,512,1024")
    parser.add_argument("--band", default="256x400x600x16",
                        help="banded case BxTxSxWINDOW (variable lengths); empty to skip")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warp-penalty", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | "
          f"CUDA {torch.version.cuda} | gamma={args.gamma} warp={args.warp_penalty}")
    print("| case | fwd ms | fwd+bwd ms | fwd extra MiB | fwd+bwd extra MiB |")
    print("|---|---:|---:|---:|---:|")
    rows = []
    if args.band:
        B, T, S, W = map(int, args.band.split("x"))
        rows.append(run(f"band B={B} T<={T} S<={S} w={W}", *make_banded(B, T, S, W, gen), args))
    for B in map(int, args.batches.split(",")):
        for T in map(int, args.lengths.split(",")):
            rows.append(run(f"dense B={B} T=S={T}", *make_dense(B, T, gen), args))
            torch.cuda.empty_cache()
    if args.json:
        meta = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "args": vars(args)}
        with open(args.json, "w") as f:
            json.dump({"metadata": meta, "results": rows}, f, indent=2)


if __name__ == "__main__":
    main()
