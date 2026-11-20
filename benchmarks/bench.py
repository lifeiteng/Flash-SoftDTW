#!/usr/bin/env python3
# Copyright (c) 2026 Feiteng Li. Licensed under the Apache License, Version 2.0.
"""CUDA-event soft-DTW benchmark; cost construction is excluded from timing.

Example: MAX_JOBS=4 .venv/bin/python benchmarks/bench.py \
    --legacy-dir ../flash-softdtw-legacy-bench --json /tmp/softdtw.json

Legacy native fwd always computes gradients. Legacy fwd+bwd additionally
runs its autograd adapter. Speedups are legacy / flash, separately per mode.
Peak MB is incremental allocated CUDA memory above live inputs (MiB).
The time guard cannot interrupt a running kernel; it checks completed calls.
Build errors and input/boundary assertion failures are fatal.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flash_softdtw as flash


class SkipCase(RuntimeError):
    pass


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gpu_processes(device):
    """Match UUIDs rather than indices remapped by CUDA_VISIBLE_DEVICES."""
    try:
        uuid = str(torch.cuda.get_device_properties(device).uuid)
        if not uuid.startswith("GPU-"):
            uuid = "GPU-" + uuid
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT, timeout=10,
        )
        raw = subprocess.check_output(
            ["nvidia-smi", "-i", uuid], text=True,
            stderr=subprocess.STDOUT, timeout=10,
        )
        print(raw, flush=True)
        rows = []
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",", 3)]
            if len(fields) == 4 and fields[0] == uuid and fields[1] != str(os.getpid()):
                rows.append(dict(zip(("gpu_uuid", "pid", "process", "memory_MiB"), fields)))
        print(f"Other compute processes on {uuid} (nvidia-smi): {rows or 'none'}", flush=True)
        return {"gpu_uuid": uuid, "other_processes": rows, "nvidia_smi": raw}
    except (OSError, subprocess.SubprocessError, AttributeError) as error:
        print(f"nvidia-smi process query skipped: {error}", flush=True)
        return {"skipped": str(error)}


def make_case(B, T, S, mode, device, band_helpers):
    gen = torch.Generator(device="cpu").manual_seed(0)
    if mode == "band":
        # Variable lengths, as in the original large-batch test (lengths >= 2
        # keep the old helper away from its undefined singleton slope).
        xl = torch.randint(2, T + 1, (B,), generator=gen, dtype=torch.int32)
        yl = torch.randint(2, S + 1, (B,), generator=gen, dtype=torch.int32)
        xl[-1], yl[-1] = T, S
        xl, yl = xl.to(device), yl.to(device)
        x = torch.randn(B, T, 8, device=device)
        y = torch.randn(B, S, 8, device=device)
        band = flash.sakoe_chiba_band(xl, yl, window=2)
        cost = flash.pairwise_cost(x, y, metric="l1", band=band).contiguous()
        if band_helpers is not None:
            _, _, boundary = band_helpers.get_softdtw_prune_ranges(xl, yl, 2)
            diff = (boundary != band).any(-1)
            # The old helper evaluates the band in float32 and can round a
            # boundary by one column; both implementations get the same band so
            # that they solve the identical problem.
            print(f"Sakoe-Chiba rows differing between old (float32) and new (float64) helper: "
                  f"{int(diff.sum())} / {diff.numel()}; using the new band for both", flush=True)
        old_cost, boundary = cost, band
        return cost, old_cost, xl, yl, band, boundary
    xl = torch.full((B,), T, dtype=torch.int32, device=device)
    yl = torch.full((B,), S, dtype=torch.int32, device=device)
    # Share precomputed dense costs; no feature construction is timed.
    cost = torch.rand(B, T, S, device=device)
    old_cost = cost
    band = None
    boundary = torch.stack(
        (torch.zeros(B, T, dtype=torch.int32, device=device),
         yl[:, None].expand(B, T)), dim=-1,
    ).contiguous()
    return cost, old_cost, xl, yl, band, boundary


def time_call(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    del result
    return elapsed


def guard_time(elapsed, args):
    if elapsed > args.max_call_ms:
        raise SkipCase(f"call {elapsed:.3f} ms exceeds --max-call-ms={args.max_call_ms}")


def measure(fn, args):
    for _ in range(args.warmup):
        guard_time(time_call(fn), args)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(args.repeats):
        elapsed = time_call(fn)
        guard_time(elapsed, args)
        samples.append(elapsed)
    peak = torch.cuda.max_memory_allocated()
    return {"ms": statistics.median(samples), "samples_ms": samples,
            "peak_allocated_MB": peak / 2**20,
            "peak_increment_MB": max(0, peak - baseline) / 2**20}


def runners(name, cost, xl, yl, band, boundary, args, legacy):
    inp = cost.detach().requires_grad_(True)
    plain = cost.detach()
    upstream = torch.ones(cost.size(0), device=cost.device)

    def call(tensor):
        if name == "flash":
            return flash.soft_dtw(
                tensor, xl, yl, band=band, gamma=args.gamma,
                warp_penalty=args.warp_penalty, rows_per_block=args.rows_per_block,
            )
        return legacy.soft_dtw(tensor, xl, yl, boundary, gamma=args.gamma,
                               warp_penalty=args.warp_penalty)

    def forward():
        with torch.no_grad():
            return call(plain)

    def forward_backward():
        values = call(inp)
        grad, = torch.autograd.grad(values, inp, grad_outputs=upstream)
        return values, grad

    return forward, forward_backward


def comparison(flash_fn, legacy_fn, gamma):
    values, grad = flash_fn()
    values_ref, grad_ref = values.detach().cpu(), grad.detach().cpu()
    del values, grad
    values, grad = legacy_fn()
    values_old, grad_old = values.detach().cpu(), grad.detach().cpu() / gamma
    tensors = (values_ref, grad_ref, values_old, grad_old)
    if not all(torch.isfinite(t).all().item() for t in tensors):
        raise RuntimeError("non-finite cost or gradient in correctness comparison")
    return {"cost_max_abs_error": (values_ref - values_old).abs().max().item(),
            "grad_max_abs_error": (grad_ref - grad_old).abs().max().item(),
            "legacy_grad_divisor": gamma}


def run_case(B, T, S, mode, args, device, legacy, band_helpers):
    cost, old_cost, xl, yl, band, boundary = make_case(B, T, S, mode, device, band_helpers)
    names = ["flash"] + (["legacy"] if legacy else [])
    funcs = {name: runners(name, cost if name == "flash" else old_cost,
                          xl, yl, band, boundary, args, legacy) for name in names}
    records = []
    active = {}
    for name in names:
        row = {"implementation": name, "B": B, "T": T, "S": S, "mode": mode,
               "prune_range": 2 if mode == "band" else -1, "status": "ok",
               "native_fwd_also_computes_grad": name == "legacy"}
        records.append(row)
        try:
            # Pilot guards precede potentially expensive correctness checks.
            guard_time(time_call(funcs[name][1]), args)
            active[name] = row
        except (SkipCase, torch.cuda.OutOfMemoryError) as error:
            row.update(status="skipped", reason=str(error))
            gc.collect()
            torch.cuda.empty_cache()
    if legacy and len(active) == 2:
        errors = comparison(funcs["flash"][1], funcs["legacy"][1], args.gamma)
        for row in records:
            row["comparison"] = errors
        print(f"{mode} B={B} T={T} S={S}: cost max abs error="
              f"{errors['cost_max_abs_error']:.8g}; grad max abs error="
              f"{errors['grad_max_abs_error']:.8g} (legacy grad / gamma)", flush=True)
    for name, row in active.items():
        for index, label in enumerate(("fwd", "fwd_bwd")):
            try:
                row[label] = measure(funcs[name][index], args)
            except (SkipCase, torch.cuda.OutOfMemoryError) as error:
                row["status"] = "partial"
                row[label] = {"skipped": str(error)}
                gc.collect()
                torch.cuda.empty_cache()
    if len(records) == 2:
        for label in ("fwd", "fwd_bwd"):
            a = records[0].get(label, {}).get("ms")
            b = records[1].get(label, {}).get("ms")
            if a and b:
                records[0].setdefault("speedup", {})[label] = b / a
                records[1].setdefault("speedup", {})[label] = 1.0
    return records


def print_table(records):
    print("\n| implementation | B | T | S | mode | fwd ms | fwd+bwd ms | "
          "speedup fwd / fwd+bwd | peak MB fwd / fwd+bwd |")
    print("|---|---:|---:|---:|---|---:|---:|---|---|")
    for row in records:
        def number(label, key):
            value = row.get(label, {}).get(key)
            return f"{value:.3f}" if value is not None else "—"
        speed = row.get("speedup", {})
        ratios = " / ".join(f"{speed[k]:.2f}x" if k in speed else "—"
                            for k in ("fwd", "fwd_bwd"))
        peaks = " / ".join(number(k, "peak_increment_MB") for k in ("fwd", "fwd_bwd"))
        name = row["implementation"] + ("*" if row["implementation"] == "legacy" else "")
        print(f"| {name} | {row['B']} | {row['T']} | {row['S']} | {row['mode']} | "
              f"{number('fwd', 'ms')} | {number('fwd_bwd', 'ms')} | {ratios} | {peaks} |")
        if row["status"] != "ok":
            print(f"Skip/partial {name} {row['mode']} B={row['B']} T={row['T']}: "
                  f"{row.get('reason', row)}", flush=True)
    print("* Legacy native fwd always computes grad; fwd+bwd includes autograd. "
          "Gradient comparison uses legacy grad / gamma. Peak MB is incremental "
          "allocated memory above live inputs (MiB). Speedup = legacy / flash.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path)
    parser.add_argument("--rows-per-block", type=int, default=0)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--device", type=int, default=0, help="logical CUDA index")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--warp-penalty", type=float, default=10.0)
    parser.add_argument("--max-call-ms", type=float, default=1000.0)
    parser.add_argument("--memory-fraction", type=float, default=0.75,
                        help="skip estimated memory above this fraction of free VRAM")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    if not math.isfinite(args.gamma) or args.gamma <= 0:
        parser.error("gamma must be finite and positive")
    if not math.isfinite(args.warp_penalty):
        parser.error("warp-penalty must be finite")
    if not 0 < args.memory_fraction <= 1 or not args.max_call_ms > 0:
        parser.error("invalid memory-fraction or max-call-ms")
    if args.rows_per_block < 0 or args.rows_per_block % 32:
        parser.error("rows-per-block must be 0 or a positive multiple of 32")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; this machine cannot run the benchmark")
    os.environ.setdefault("MAX_JOBS", "4")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.manual_seed(args.seed)
    legacy = band_helpers = None
    if args.legacy_dir:
        root = args.legacy_dir.resolve()
        legacy = import_file("legacy_softdtw", root / "legacy_softdtw.py")
        band_helpers = import_file("legacy_band", root / "legacy_band.py")
        legacy.load()
    metadata = {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
                "cuda": torch.version.cuda, "device": str(device), "dtype": "float32",
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "processes_before": gpu_processes(device),
                "lengths": "fixed full lengths; band B=554,T=40,S=128,C=8,window=2",
                "legacy_note": "native always computes grad; comparison divides legacy grad by gamma",
                "memory_note": "MiB allocated above live inputs; total allocated saved per mode"}
    print(json.dumps(metadata, indent=2), flush=True)
    records = []
    # Original kernels launch on the default stream; use it for both versions.
    with torch.cuda.stream(torch.cuda.default_stream(device)):
        # Compile before any timed calls or skip guards.
        tiny = torch.zeros(1, 2, 2, device=device, requires_grad=True)
        flash.soft_dtw(tiny, rows_per_block=args.rows_per_block).sum().backward()
        del tiny
        torch.cuda.synchronize()
        cases = [(554, 40, 128, "band")]
        cases += [(b, n, n, "dense") for b in (1, 8, 64, 256)
                  for n in (128, 256, 512, 1024)]
        for B, T, S, mode in cases:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                free, _ = torch.cuda.mem_get_info()
                # Conservative combined input, gradient, and legacy scratch budget.
                estimate = B * T * (S if mode == "dense" else 8) * 48 + 64 * 2**20
                if estimate > free * args.memory_fraction:
                    raise SkipCase(f"estimated {estimate / 2**20:.1f} MiB exceeds "
                                   f"{args.memory_fraction:.0%} of free {free / 2**20:.1f} MiB")
                records.extend(run_case(B, T, S, mode, args, device, legacy, band_helpers))
            except (SkipCase, torch.cuda.OutOfMemoryError) as error:
                for name in (["flash", "legacy"] if legacy else ["flash"]):
                    records.append({"implementation": name, "B": B, "T": T, "S": S,
                                    "mode": mode, "status": "skipped", "reason": str(error)})
                print(f"Skipped {mode} B={B} T={T} S={S}: {error}", flush=True)
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps({"metadata": metadata, "results": records},
                                               indent=2, allow_nan=False) + "\n")
    print_table(records)
    gpu_processes(device)


if __name__ == "__main__":
    main()
