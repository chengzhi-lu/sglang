#!/usr/bin/env python3
"""Benchmark the bounded expert grouping control path on CPU and CUDA.

The benchmark measures end-to-end Python-visible latency, including the
device-to-host metadata required by the current GPU implementation.  It is
intentionally independent of model loading so grouping changes can be checked
without launching the 35B model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import List

import torch

from sglang.jit_kernel.layerkv_expert_group import (
    layerkv_group_token_experts,
    layerkv_group_token_experts_multi,
)
from sglang.srt.layerkv.residency_budget import group_token_experts


def _percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
    return float(ordered[index])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row-counts",
        nargs="+",
        type=int,
        default=[8, 32, 128, 512, 2000],
        help="routed token row counts to benchmark",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--full-num-experts", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=9)
    parser.add_argument(
        "--multi-capacities",
        nargs="+",
        type=int,
        default=None,
        help="also benchmark one shared GPU pass for these capacities",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("int32", "int64"),
        default="int64",
        help="route ID dtype",
    )
    args = parser.parse_args()
    if any(value <= 0 for value in args.row_counts):
        parser.error("--row-counts values must be positive")
    if args.top_k <= 0 or args.full_num_experts <= 0 or args.capacity <= 0:
        parser.error("--top-k, --full-num-experts, and --capacity must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be nonnegative and --iterations must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this benchmark")
    if args.capacity < args.top_k:
        parser.error("--capacity must cover one token's top-k route")
    if args.multi_capacities is not None and any(
        value < args.top_k for value in args.multi_capacities
    ):
        parser.error("--multi-capacities values must cover one token's top-k route")
    return args


def _timed_cpu(routes: torch.Tensor, capacity: int, full_num_experts: int) -> float:
    rows = routes.tolist()
    start = time.perf_counter()
    group_token_experts(rows, capacity, full_num_experts)
    return (time.perf_counter() - start) * 1000.0


def _timed_gpu(
    routes: torch.Tensor, capacity: int, full_num_experts: int
) -> float:
    start = time.perf_counter()
    layerkv_group_token_experts(
        routes, capacity, full_num_experts, device_rows=True
    )
    return (time.perf_counter() - start) * 1000.0


def _timed_gpu_multi(
    routes: torch.Tensor,
    capacities: List[int],
    full_num_experts: int,
) -> float:
    start = time.perf_counter()
    layerkv_group_token_experts_multi(
        routes, capacities, full_num_experts, device_rows=True
    )
    return (time.perf_counter() - start) * 1000.0


def _run_case(args: argparse.Namespace, rows: int, dtype: torch.dtype) -> dict:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + rows)
    routes = torch.randint(
        0,
        args.full_num_experts,
        (rows, args.top_k),
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    # Make the common capacity=top-k+1 shape realistic while guaranteeing
    # every individual token can be served.
    routes[:, 0] = torch.arange(rows, device="cuda") % args.full_num_experts
    for _ in range(args.warmup):
        _timed_gpu(routes, args.capacity, args.full_num_experts)
    torch.cuda.synchronize()

    cpu_samples = [
        _timed_cpu(routes, args.capacity, args.full_num_experts)
        for _ in range(args.iterations)
    ]
    gpu_samples = [
        _timed_gpu(routes, args.capacity, args.full_num_experts)
        for _ in range(args.iterations)
    ]
    torch.cuda.synchronize()
    cpu_median = float(statistics.median(cpu_samples))
    gpu_median = float(statistics.median(gpu_samples))
    result = {
        "rows": rows,
        "cpu_ms_median": cpu_median,
        "cpu_ms_p95": _percentile(cpu_samples, 0.95),
        "gpu_ms_median": gpu_median,
        "gpu_ms_p95": _percentile(gpu_samples, 0.95),
        "gpu_over_cpu": gpu_median / cpu_median if cpu_median else None,
    }
    if args.multi_capacities is not None:
        capacities = list(dict.fromkeys(args.multi_capacities))
        for _ in range(args.warmup):
            _timed_gpu_multi(routes, capacities, args.full_num_experts)
        multi_samples = [
            _timed_gpu_multi(routes, capacities, args.full_num_experts)
            for _ in range(args.iterations)
        ]
        multi_median = float(statistics.median(multi_samples))
        result.update(
            {
                "multi_capacities": capacities,
                "gpu_multi_ms_median": multi_median,
                "gpu_multi_ms_p95": _percentile(multi_samples, 0.95),
                "multi_over_single_gpu": (
                    multi_median / gpu_median if gpu_median else None
                ),
            }
        )
    return result


def main() -> int:
    args = _parse_args()
    dtype = torch.int32 if args.dtype == "int32" else torch.int64
    results = [_run_case(args, rows, dtype) for rows in args.row_counts]
    print(
        json.dumps(
            {
                "arguments": vars(args),
                "device": torch.cuda.get_device_name(),
                "results": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
