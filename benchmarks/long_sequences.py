# Copyright (c) 2026 Feiteng Li. Licensed under the Apache License, Version 2.0.
"""Timing and memory of long dense problems (no length cap)."""

import argparse

import torch

import flash_softdtw as fsd


def timed(fn, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    return sorted(samples)[len(samples) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="1x2048,8x2048,1x4096,8x4096,64x4096",
                        help="comma-separated BxL list")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | CUDA {torch.version.cuda}")
    print("| B | T = S | fwd ms | fwd+bwd ms | extra peak MiB (fwd+bwd) |")
    print("|---:|---:|---:|---:|---:|")
    for case in args.cases.split(","):
        B, L = map(int, case.split("x"))
        cost = torch.rand(B, L, L, device="cuda")
        leaf = cost.clone().requires_grad_()

        def forward():
            with torch.no_grad():
                fsd.soft_dtw(cost, gamma=args.gamma)

        def forward_backward():
            value = fsd.soft_dtw(leaf, gamma=args.gamma)
            torch.autograd.grad(value.sum(), leaf)

        fwd = timed(forward, args.repeats)
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        both = timed(forward_backward, args.repeats)
        peak = (torch.cuda.max_memory_allocated() - base) / 2**20
        print(f"| {B} | {L} | {fwd:.2f} | {both:.2f} | {peak:.0f} |")
        del cost, leaf
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
