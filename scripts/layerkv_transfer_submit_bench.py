#!/usr/bin/env python3
"""Bounded real-CUDA submit microbenchmark, not a model throughput benchmark."""

import argparse
import cProfile
import hashlib
import importlib.util
import json
import pstats
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def load_copier(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ExpertBatchTransfer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--param-elements", nargs="+", type=int, default=[2097152, 1048576])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--profile-iterations", type=int, default=20)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.gpu < 0 or min(args.experts, args.iterations, args.repeats) <= 0:
        p.error("gpu must be nonnegative; experts, iterations and repeats positive")
    if min(args.warmup, args.profile_iterations) < 0 or min(args.param_elements) <= 0:
        p.error("warmup/profile iterations must be nonnegative; elements positive")
    if not args.baseline_source.is_file() or args.output.exists():
        p.error("baseline source must exist; output must be a fresh path")
    candidate = (
        Path(__file__).resolve().parents[1]
        / "python/sglang/srt/layerkv/expert_transfer.py"
    )
    sources = {"baseline": args.baseline_source, "candidate": candidate}
    result = {
        "args": vars(args),
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sources.items()
        },
        "bytes_per_batch": args.experts * sum(args.param_elements) * 2,
        "scope": "CPU copy() wall time, prebuilt views, per-call GPU fence outside timing; no model inference",
        "runs": [],
        "profiles": {},
    }
    print(json.dumps(result, indent=2, default=str), flush=True)
    if not args.execute:
        return

    import torch

    from sglang.srt.layerkv.config_stats import LayerKVStats

    torch.cuda.set_device(args.gpu)
    result["device"] = torch.cuda.get_device_name(args.gpu)
    result["torch_version"] = torch.__version__
    stream = torch.cuda.Stream(device=args.gpu)
    dtype = getattr(torch, args.dtype)
    host = [
        torch.full((n,), i + 1, dtype=dtype, pin_memory=True)
        for n in args.param_elements
        for i in range(args.experts)
    ]
    gpu = [torch.empty_like(t, device=stream.device) for t in host]
    output = [torch.empty_like(t, pin_memory=True) for t in host]
    pairs = {"h2d": list(zip(gpu, host)), "d2h": list(zip(output, gpu))}
    copiers = {name: load_copier(path, name) for name, path in sources.items()}

    def copy_once(copier, direction):
        start = time.perf_counter_ns()
        copier.copy(pairs[direction], stream)
        elapsed = (time.perf_counter_ns() - start) / 1000
        stream.synchronize()  # Deliberately outside CPU submission measurement.
        return elapsed

    for repeat in range(args.repeats):
        order = list(copiers) if repeat % 2 == 0 else list(reversed(copiers))
        for name in order:
            copier = copiers[name](LayerKVStats())
            for direction in pairs:
                for _ in range(args.warmup):
                    copy_once(copier, direction)
                samples = [copy_once(copier, direction) for _ in range(args.iterations)]
                copier.collect(block=True)
                if copier.pending or copier.host_inflight or copier.failed_refs:
                    raise RuntimeError("transfer owners were not retired")
                actual = gpu if direction == "h2d" else output
                if not all(torch.equal(a.cpu(), b) for a, b in zip(actual, host)):
                    raise RuntimeError(f"{name} {direction}: copy values differ")
                row = {
                    "repeat": repeat + 1,
                    "mode": name,
                    "direction": direction,
                    "median_us": statistics.median(samples),
                    "mean_us": statistics.mean(samples),
                    "samples_us": samples,
                }
                result["runs"].append(row)
                print(
                    json.dumps({k: v for k, v in row.items() if k != "samples_us"}),
                    flush=True,
                )
    for name, cls in copiers.items():
        if not args.profile_iterations:
            continue
        copier = cls(LayerKVStats())
        profiler = cProfile.Profile()
        for _ in range(args.profile_iterations):
            profiler.runcall(copier.copy, pairs["h2d"], stream)
            stream.synchronize()
        copier.collect(block=True)
        stats = pstats.Stats(profiler)
        result["profiles"][name] = [
            {
                "function": str(key),
                "primitive_calls": value[0],
                "calls": value[1],
                "self_ms": value[2] * 1000,
                "inclusive_ms": value[3] * 1000,
            }
            for key, value in sorted(
                stats.stats.items(), key=lambda item: item[1][2], reverse=True
            )
        ]
    result["valid"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
