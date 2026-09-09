#!/usr/bin/env python3
"""Attribute opt-in Kineto traces by CPU launch correlation, not GPU nesting.

Single worker/device only. GPU activity overlap is descriptive, not proof of a
dependency or of time recoverable by prefetch. No model imports or GPU work.
"""

import argparse
import gzip
import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path


def end(event):
    return event["ts"] + event["dur"]


def union_us(events, window=None):
    intervals = [(e["ts"], end(e)) for e in events]
    if window is not None:
        intervals = [(max(a, window[0]), min(b, window[1])) for a, b in intervals]
    total, right = 0.0, float("-inf")
    for a, b in sorted(intervals):
        if b > a:
            total += max(0.0, b - max(a, right))
            right = max(right, b)
    return total


class Ranges:
    def __init__(self, events):
        self.events = sorted(events, key=lambda e: e["ts"])
        self.starts = [e["ts"] for e in self.events]
        for a, b in zip(self.events, self.events[1:]):
            if end(a) > b["ts"]:
                raise ValueError("overlapping same-phase CPU ranges")

    def containing(self, event):
        i = bisect_right(self.starts, event["ts"]) - 1
        if i >= 0:
            parent = self.events[i]
            if (
                parent["pid"] == event["pid"]
                and parent["tid"] == event["tid"]
                and end(event) <= end(parent)
            ):
                return i
        return None


def analyze(events, expected_groups, expected_batches, expected_bytes):
    if min(expected_groups, expected_batches, expected_bytes) <= 0:
        raise ValueError("this diagnostic requires prefill groups with H2D misses")
    events = [e for e in events if e.get("ph") == "X"]
    phases = defaultdict(list)
    apis = defaultdict(list)
    gpu = []
    for e in events:
        cat = e.get("cat")
        if cat == "user_annotation" and e["name"].startswith("layerkv/"):
            phases[e["name"].removeprefix("layerkv/")].append(e)
        elif cat in ("cuda_runtime", "cuda_driver"):
            if "correlation" in e.get("args", {}):
                apis[e["args"]["correlation"]].append(e)
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu.append(e)
    required = (
        "chunk/prepare",
        "h2d_submit",
        "ready_wait",
        "remap",
        "guard_reduce",
        "readback",
        "cache_trim",
    )
    if any(not phases[name] for name in required) or not gpu:
        raise ValueError("missing CPU phases or GPU activities")
    workers = {(e["pid"], e["tid"]) for es in phases.values() for e in es}
    devices = {e["args"]["device"] for e in gpu}
    if len(workers) != 1 or len(devices) != 1:
        raise ValueError("only single-worker, single-device traces are supported")
    ranges = {name: Ranges(es) for name, es in phases.items()}
    groups = ranges["chunk/prepare"]
    if len(groups.events) != expected_groups:
        raise ValueError("prefill group count does not match runtime stats")
    launches = {}
    for e in gpu:
        matches = apis.get(e.get("args", {}).get("correlation"), [])
        if len(matches) != 1:
            raise ValueError("missing or ambiguous GPU launch correlation")
        launches[id(e)] = matches[0]
    by_phase = defaultdict(list)
    by_group = defaultdict(list)
    batch_copies = defaultdict(list)
    for e in gpu:
        api = launches[id(e)]
        for name, spans in ranges.items():
            if spans.containing(api) is not None:
                by_phase[name].append(e)
        group = groups.containing(api)
        if group is not None:
            by_group[group].append(e)
        if (
            e["cat"] == "gpu_memcpy"
            and "HtoD" in e["name"]
            and ranges["h2d_submit"].containing(api) is not None
        ):
            batch_copies[e["args"]["correlation"]].append(e)
    copies = [e for batch in batch_copies.values() for e in batch]
    if len(batch_copies) != expected_batches:
        raise ValueError("expert H2D batch count does not match runtime stats")
    if sum(e["args"]["bytes"] for e in copies) != expected_bytes:
        raise ValueError("expert H2D bytes do not match materialized weights")
    copy_ids = {id(e) for e in copies}
    guard_ids = {
        id(e) for name in ("remap", "guard_reduce", "readback") for e in by_phase[name]
    }
    rows = []
    for i, group in enumerate(groups.events):
        nested = {
            name: [e for e in spans.events if groups.containing(e) == i]
            for name, spans in ranges.items()
        }
        if any(
            len(nested[name]) != 1
            for name in ("h2d_submit", "remap", "guard_reduce", "readback")
        ):
            raise ValueError("incomplete or duplicate guard phases in prefill group")
        for name in ("remap", "guard_reduce", "readback"):
            if not any(groups.containing(launches[id(e)]) == i for e in by_phase[name]):
                raise ValueError("missing GPU activity for a prefill guard phase")
        readback = nested["readback"][0]
        window = (readback["ts"], end(readback))
        own_copies = [e for e in by_group[i] if id(e) in copy_ids]
        if not own_copies:
            raise ValueError("expected an H2D miss in each profiled prefill group")
        own_guard = [e for e in by_group[i] if id(e) in guard_ids]
        overlap = [e for e in gpu if e["ts"] < window[1] and end(e) > window[0]]
        h2d_us = union_us(own_copies, window)
        guard_us = union_us(own_copies + own_guard, window) - h2d_us
        busy_us = union_us(overlap, window)
        other_us = busy_us - h2d_us - guard_us
        # Disjoint priority partition: H2D, guard, other observed work, no work.
        cpu_before = [
            e
            for es in apis.values()
            for e in es
            if groups.containing(e) == i and end(e) <= nested["h2d_submit"][0]["ts"]
        ]
        gpu_before = [
            e
            for e in by_group[i]
            if end(launches[id(e)]) <= nested["h2d_submit"][0]["ts"]
        ]
        small_h2d = [
            e for e in gpu_before if e["cat"] == "gpu_memcpy" and "HtoD" in e["name"]
        ]
        earlier = [e for e in gpu if launches[id(e)]["ts"] < group["ts"]]
        launch_delay = []
        for corr in {e["args"]["correlation"] for e in own_copies}:
            first = min(batch_copies[corr], key=lambda e: e["ts"])
            launch_delay.append(max(0.0, first["ts"] - end(launches[id(first)])))
        rows.append(
            {
                "group": i + 1,
                "cpu_ms": {
                    name: union_us(es) / 1000 for name, es in nested.items() if es
                },
                "h2d_gpu_ms": union_us(own_copies) / 1000,
                "h2d_bytes": sum(e["args"]["bytes"] for e in own_copies),
                "h2d_launch_to_first_start_ms": sum(launch_delay) / 1000,
                "h2d_done_before_readback": max(map(end, own_copies)) <= window[0],
                "readback_partition_ms": {
                    "own_h2d": h2d_us / 1000,
                    "own_remap_guard_readback": guard_us / 1000,
                    "other_gpu_activity": max(0.0, other_us) / 1000,
                    "no_observed_gpu_activity": max(0.0, readback["dur"] - busy_us)
                    / 1000,
                },
                "earlier_submitted_gpu_overlap_readback_ms": union_us(
                    [e for e in overlap if launches[id(e)]["ts"] < group["ts"]], window
                )
                / 1000,
                "earlier_submitted_gpu_overlap_prepare_ms": union_us(
                    earlier, (group["ts"], end(group))
                )
                / 1000,
                "before_h2d_small_copy_count": len(small_h2d),
                "before_h2d_small_copy_bytes": sum(
                    e["args"]["bytes"] for e in small_h2d
                ),
                "cpu_before_h2d_submit_ms": (
                    nested["h2d_submit"][0]["ts"] - group["ts"]
                )
                / 1000,
                "before_h2d_sync_calls": sum(
                    e["name"] == "cudaStreamSynchronize" for e in cpu_before
                ),
                "before_h2d_sync_cpu_ms": union_us(
                    [e for e in cpu_before if e["name"] == "cudaStreamSynchronize"]
                )
                / 1000,
            }
        )
    totals = {
        key: sum(row[key] for row in rows)
        for key in (
            "h2d_gpu_ms",
            "h2d_bytes",
            "h2d_launch_to_first_start_ms",
            "h2d_done_before_readback",
            "earlier_submitted_gpu_overlap_readback_ms",
            "earlier_submitted_gpu_overlap_prepare_ms",
            "before_h2d_small_copy_count",
            "before_h2d_small_copy_bytes",
            "cpu_before_h2d_submit_ms",
            "before_h2d_sync_calls",
            "before_h2d_sync_cpu_ms",
        )
    }
    for field in ("cpu_ms", "readback_partition_ms"):
        totals[field] = {
            name: sum(row[field].get(name, 0) for row in rows)
            for name in {name for row in rows for name in row[field]}
        }
    totals["gpu_phase_ms"] = {
        name: union_us(
            [e for e in es if groups.containing(launches[id(e)]) is not None]
        )
        / 1000
        for name, es in by_phase.items()
        if any(groups.containing(launches[id(e)]) is not None for e in es)
    }
    return {
        "valid": True,
        "scope": "profiled prefill; overlap is not causal or recoverable latency",
        "cpu_phase_counts": {name: len(es) for name, es in phases.items()},
        "request_h2d_batches": len(batch_copies),
        "request_h2d_bytes": expected_bytes,
        "request_h2d_gpu_ms": union_us(copies) / 1000,
        "prefill": totals,
        "groups": rows,
    }


def analyze_run(run_dir, expert_bytes):
    plan = json.loads((run_dir / "plan.json").read_text())["args"]
    if not json.loads((run_dir / "summary.json").read_text())["valid"]:
        raise ValueError("runtime guards failed")
    request = plan["trace_waits_request"]
    if request <= 0:
        raise ValueError("run did not request a trace")
    result = {
        "run_dir": str(run_dir),
        "args": plan,
        "expert_bytes": expert_bytes,
        "arms": {},
    }
    for arm in ("fixed16", "grow17"):
        paths = list((run_dir / f"{arm}.trace").glob("*.trace.json.gz"))
        if len(paths) != 1:
            raise ValueError(f"{arm}: expected exactly one trace")
        raw = json.loads((run_dir / f"{arm}.json").read_text())["requests"][request - 1]
        with gzip.open(paths[0], "rt") as f:
            trace = json.load(f)
        result["arms"][arm] = {
            "trace": str(paths[0]),
            **analyze(
                trace["traceEvents"],
                raw["shared_delta"]["token_chunk_calls"],
                raw["stats_delta"]["expert_cuda_batch_h2d_count"],
                expert_bytes * raw["stats_delta"]["expert_materialize_count"],
            ),
        }
        if result["arms"][arm]["prefill"]["h2d_bytes"] != (
            expert_bytes * raw["shared_delta"]["token_chunk_materializations"]
        ):
            raise ValueError(f"{arm}: prefill H2D bytes mismatch")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--expert-bytes",
        type=int,
        required=True,
        help="sum of parameter bytes per expert",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expert_bytes <= 0:
        parser.error("--expert-bytes must be positive")
    result = analyze_run(args.run_dir, args.expert_bytes)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(f"valid=True; {args.output}")


if __name__ == "__main__":
    main()
