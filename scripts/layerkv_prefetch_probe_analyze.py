#!/usr/bin/env python3
"""Join structural lookahead bounds with a profiled current-group GPU window.

No prefetch is executed. Positive capacity is not DMA/lifetime admission.
"""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from layerkv_wait_trace_analyze import Ranges, analyze, end, union_us


def join_windows(events, probe, waits):
    if probe is None or probe["order"] != "input":
        raise ValueError("missing input-order structural probe")
    snapshots = probe["groups"]
    phases = Ranges(
        [
            e
            for e in events
            if e.get("ph") == "X"
            and e.get("cat") == "user_annotation"
            and e.get("name") == "layerkv/chunk/moe"
        ]
    )
    if (
        not snapshots
        or len(snapshots) != len(phases.events)
        or len(snapshots) != len(waits["groups"])
    ):
        raise ValueError("probe, prepare and MoE group counts must match")
    apis = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver"):
            if "correlation" in e.get("args", {}):
                apis[e["args"]["correlation"]].append(e)
    kernels = [[] for _ in snapshots]
    for e in events:
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        matches = apis.get(e.get("args", {}).get("correlation"), [])
        if len(matches) != 1:
            raise ValueError("missing or ambiguous kernel launch")
        group = phases.containing(matches[0])
        if group is not None:
            kernels[group].append(e)
    rows = []
    for i, (snapshot, gpu) in enumerate(zip(snapshots, kernels)):
        if snapshot["group"] != i + 1 or not gpu:
            raise ValueError("missing group or MoE device work")
        has_next = i + 1 < len(snapshots)
        if snapshot["has_next"] != has_next:
            raise ValueError("incomplete next-group metadata")
        span_ms = (max(map(end, gpu)) - min(e["ts"] for e in gpu)) / 1000
        row = dict(
            snapshot,
            moe_kernel_ms=union_us(gpu) / 1000,
            moe_device_span_ms=span_ms,
            moe_cpu_ms=phases.events[i]["dur"] / 1000,
        )
        if has_next:
            next_h2d = waits["groups"][i + 1]
            if (
                len(snapshot["next_missing_ids"]) * snapshot["expert_bytes"]
                != next_h2d["h2d_bytes"]
            ):
                raise ValueError("next-group demand bytes differ from observed H2D")
            row.update(
                next_h2d_gpu_ms=next_h2d["h2d_gpu_ms"],
                # Optimistic window-only ceiling: ignores candidate byte fraction,
                # CPU submission cost and all transfer/slot ownership dependencies.
                loose_overlap_ceiling_ms=(
                    min(span_ms, next_h2d["h2d_gpu_ms"])
                    if snapshot["backed_upper_experts"]
                    else 0.0
                ),
            )
        rows.append(row)
    transitions = [r for r in rows if r["has_next"]]
    missing = sum(len(r["next_missing_ids"]) for r in transitions)
    eligible = sum(r["backed_upper_experts"] for r in transitions)
    return dict(
        valid=True,
        scope="single-next-group structural upper bound, not a prefetch speedup prediction",
        group_count=len(rows),
        transition_count=len(transitions),
        slot_capacity_histogram=dict(Counter(r["slot_capacity"] for r in rows)),
        active_experts_histogram=dict(Counter(len(r["current_ids"]) for r in rows)),
        unused_slots_histogram=dict(Counter(len(r["unused_slot_ids"]) for r in rows)),
        capacity_positive_transitions=sum(
            r["capacity_upper_experts"] > 0 for r in transitions
        ),
        backed_positive_transitions=sum(
            r["backed_upper_experts"] > 0 for r in transitions
        ),
        next_missing_experts=missing,
        backed_upper_experts=eligible,
        backed_upper_bytes=sum(r["backed_upper_bytes"] for r in transitions),
        backed_upper_fraction=eligible / missing if missing else None,
        moe_kernel_ms=sum(r["moe_kernel_ms"] for r in rows),
        moe_device_span_ms=sum(r["moe_device_span_ms"] for r in rows),
        loose_overlap_ceiling_ms=sum(
            r["loose_overlap_ceiling_ms"] for r in transitions
        ),
        groups=rows,
    )


def analyze_run(run_dir):
    plan = json.loads((run_dir / "plan.json").read_text())["args"]
    if not json.loads((run_dir / "summary.json").read_text())["valid"]:
        raise ValueError("runtime guards did not pass")
    request = plan["trace_waits_request"]
    if request <= 0:
        raise ValueError("no trace requested")
    report = dict(run_dir=str(run_dir), args=plan, arms={})
    for arm in ("fixed16", "grow17"):
        raw = json.loads((run_dir / f"{arm}.json").read_text())["requests"][request - 1]
        shared = raw["final_stats"]["shared_vmm"]
        probe = shared.get("prefetch_probe")
        if probe is None or probe["batch"] != shared["token_chunk_batches"]:
            raise ValueError("missing or stale structural probe")
        if len(probe["groups"]) != raw["shared_delta"]["token_chunk_calls"]:
            raise ValueError("probe must cover exactly the captured request's prefill")
        if sum(row["tokens"] for row in probe["groups"]) != plan["input_tokens"]:
            raise ValueError("probe input-token coverage mismatch")
        sizes = {row["expert_bytes"] for row in probe["groups"]}
        if len(sizes) != 1:
            raise ValueError("single uniform-size expert layer required")
        expert_bytes = sizes.pop()
        paths = list((run_dir / f"{arm}.trace").glob("*.trace.json.gz"))
        if len(paths) != 1:
            raise ValueError("expected exactly one worker trace per arm")
        with gzip.open(paths[0], "rt") as f:
            events = json.load(f)["traceEvents"]
        waits = analyze(
            events,
            len(probe["groups"]),
            raw["stats_delta"]["expert_cuda_batch_h2d_count"],
            expert_bytes * raw["stats_delta"]["expert_materialize_count"],
        )
        report["arms"][arm] = dict(
            trace=str(paths[0]),
            feasibility=join_windows(events, probe, waits),
            waits=waits,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_run(args.run_dir)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    for arm, data in result["arms"].items():
        print(
            arm,
            json.dumps({k: v for k, v in data["feasibility"].items() if k != "groups"}),
        )


if __name__ == "__main__":
    main()
