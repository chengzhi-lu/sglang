#!/usr/bin/env python3
"""Bounded fixed-16 vs KV-funded-17 comparison; dry plan unless --execute.

Both arms use the same KV pressure, VMM pool, compact expert layer and streaming
instrumentation. First requests are retained separately from later requests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from layerkv_hybrid_validation import engine_kwargs


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str))


def delta(before, after):
    return {
        k: v - before.get(k, 0)
        for k, v in after.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def child(args):
    import sglang as sgl

    engine_args = argparse.Namespace(
        **vars(args),
        case="offload",
        batch_size=1,
        expert_extra_slots=0 if args.arm == "fixed16" else 1,
    )
    kwargs = engine_kwargs(engine_args)
    kwargs.update(
        enable_metrics=True,
        stream_interval=1,
        layerkv_debug_stats=False,
        skip_tokenizer_init=args.skip_tokenizer_init,
        layerkv_shared_expert_disable_donor_cache=args.disable_donor_cache,
        layerkv_shared_expert_chunk_order=args.chunk_order,
        layerkv_shared_expert_prepare_path=args.prepare_path,
        layerkv_shared_expert_profile_chunks=args.profile_chunks,
        layerkv_shared_expert_profile_prepare=args.profile_prepare,
        layerkv_shared_expert_trace_waits=args.trace_waits_request > 0,
        layerkv_expert_remap_update=args.expert_remap_update,
        layerkv_expert_transfer_backend=args.expert_transfer_backend,
        layerkv_expert_batch_backing_layout=args.expert_batch_backing_layout,
        layerkv_expert_demand_d2h_wait=args.expert_demand_d2h_wait,
        layerkv_expert_backing_release_validation=args.expert_backing_release_validation,
        layerkv_expert_backing_cache_mb=args.expert_backing_cache_mb,
        layerkv_expert_backing_cache_accounting=args.expert_backing_cache_accounting,
        layerkv_expert_backing_cache_accounting_check=args.check_cache_accounting,
        layerkv_expert_cpu_backing_pool_mb=(
            args.expert_host_extra_budget_mb - args.expert_backing_cache_mb
        ),
    )
    dump(args.output_dir / f"{args.arm}.engine_args.json", kwargs)
    started = time.perf_counter()
    engine = sgl.Engine(**kwargs)
    result = {"engine_startup_s": time.perf_counter() - started, "requests": []}
    try:
        before = engine.get_server_info()["internal_states"][0]["layerkv"]
        for index in range(args.rounds):
            inputs = [100 + (i + 7 * index) % 80 for i in range(args.input_tokens)]
            events = []
            traced = index + 1 == args.trace_waits_request
            if traced:
                engine.start_profile(
                    output_dir=str(args.output_dir / f"{args.arm}.trace"),
                    activities=["CPU", "GPU"],
                    with_stack=False,
                    record_shapes=False,
                )
            started = time.perf_counter()
            final = None
            for response in engine.generate(
                input_ids=inputs,
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": args.output_tokens,
                    "ignore_eos": True,
                },
                return_logprob=True,
                logprob_start_len=-1,
                stream=True,
            ):
                events.append(
                    {
                        "elapsed_s": time.perf_counter() - started,
                        "tokens": response["meta_info"]["completion_tokens"],
                    }
                )
                final = response
            elapsed = time.perf_counter() - started
            if traced:
                engine.stop_profile()
            after = engine.get_server_info()["internal_states"][0]["layerkv"]
            if not events or final is None:
                raise RuntimeError("no streaming response")
            meta = final["meta_info"]
            decode_s = events[-1]["elapsed_s"] - events[0]["elapsed_s"]
            prefill_s = None
            if meta.get("prefill_finished_time") and meta.get("forward_entry_time"):
                prefill_s = meta["prefill_finished_time"] - meta["forward_entry_time"]
            entry = {
                "round": index + 1,
                "first_request": index == 0,
                "client_e2e_s": elapsed,
                "server_e2e_s": meta.get("e2e_latency"),
                "ttft_s": events[0]["elapsed_s"],
                "prefill_forward_s": prefill_s,
                "decode_s": decode_s,
                "tpot_ms": decode_s * 1000 / (args.output_tokens - 1),
                "stream_timing_valid": events[0]["tokens"] == 1
                and events[-1]["tokens"] == args.output_tokens,
                "events": events,
                "response": final,
                "stats_delta": delta(before, after),
                "shared_delta": delta(
                    before.get("shared_vmm", {}), after["shared_vmm"]
                ),
                "final_stats": after,
            }
            result["requests"].append(entry)
            dump(args.output_dir / f"{args.arm}.json", result)
            print(
                json.dumps(
                    {
                        k: entry[k]
                        for k in [
                            "round",
                            "ttft_s",
                            "prefill_forward_s",
                            "decode_s",
                            "tpot_ms",
                            "shared_delta",
                        ]
                    }
                ),
                flush=True,
            )
            before = after
    finally:
        engine.shutdown()


def summarize(args):
    execution_args = json.loads((args.output_dir / "plan.json").read_text())["args"]
    analysis_keys = {"execute", "arm", "summarize_only", "decode_tail_start"}
    for key, value in execution_args.items():
        if key not in analysis_keys and str(getattr(args, key)) != str(value):
            raise ValueError(f"analysis argument {key} differs from recorded execution")
    arms = {
        arm: json.loads((args.output_dir / f"{arm}.json").read_text())
        for arm in ["fixed16", "grow17"]
    }
    rows = []
    valid = all(len(a["requests"]) == args.rounds for a in arms.values())
    for fixed, grow in zip(arms["fixed16"]["requests"], arms["grow17"]["requests"]):
        f = fixed["response"]["meta_info"]["output_token_logprobs"]
        g = grow["response"]["meta_info"]["output_token_logprobs"]
        exact = len(f) == len(g) == args.output_tokens and [x[1] for x in f] == [
            x[1] for x in g
        ]
        differences = [abs(x[0] - y[0]) for x, y in zip(f, g)]
        max_delta = max(differences, default=math.inf)
        guards = True
        for request in [fixed, grow]:
            s = request["final_stats"]
            v = s["shared_vmm"]
            guards &= all(
                s.get(k) is True for k in ["kvc_guard_pass", "expert_guard_pass"]
            )
            guards &= all(
                s.get(k) == 0
                for k in [
                    "kvc_stale_entry_count",
                    "kvc_page_alignment_violation_count",
                    "kvc_physical_failure_count",
                    "kvc_host_used_tokens",
                ]
            )
            guards &= v["ownership_guard_pass"] and v["expert_pointers_stable"]
            guards &= (
                v["loan_bytes"]
                == v["blocked_kv_tokens"]
                == v["growth_physical_create_count"]
                == 0
            )
            guards &= v["kv_to_expert_pages"] == v["expert_to_kv_pages"]
            guards &= v["current_expert_slots"] == args.expert_initial_slots
            guards &= request["stream_timing_valid"]
            host = s["expert_host_budget"]
            guards &= host["optional_guard_pass"] and host["ledger_matches"]
            guards &= host.get("cache_accounting_matches", True)
            if (
                args.expert_backing_cache_accounting == "incremental"
                and args.expert_backing_cache_mb > 0
            ):
                guards &= host.get("cache_accounting_active", False)
                guards &= s.get("expert_backing_cache_fast_return_count", 0) > 0
            guards &= host["batch_owner_storage_bytes"] == 0
            if args.expert_remap_update == "batch":
                guards &= s.get("expert_remap_batch_count", 0) > 0
                guards &= s.get("expert_remap_batch_entries", 0) > 0
            else:
                guards &= s.get("expert_remap_batch_count", 0) == 0
            guards &= host["optional_limit_bytes"] == (
                int(args.expert_backing_cache_mb * 1024 * 1024)
                + int(
                    (args.expert_host_extra_budget_mb - args.expert_backing_cache_mb)
                    * 1024
                    * 1024
                )
            )
            guards &= (
                host["tracked_unique_storage_bytes"]
                <= host["mandatory_valid_bytes"] + host["optional_limit_bytes"]
            )
            if args.expert_transfer_backend == "cuda-batch":
                guards &= s.get("expert_cuda_batch_h2d_count", 0) > 0
                if args.expert_backing_cache_mb == 0:
                    guards &= s.get("expert_cuda_batch_d2h_count", 0) > 0
                else:
                    guards &= s.get("expert_eviction_d2h_skip_count", 0) > 0
                guards &= s.get("expert_cuda_batch_pending") == 0
            if args.expert_demand_d2h_wait == "stream":
                if args.expert_backing_cache_mb == 0:
                    guards &= s.get("expert_demand_d2h_stream_batch_count", 0) > 0
                guards &= s.get("expert_demand_h2d_dependency_count", 0) > 0
            if args.profile_chunks:
                guards &= v.get("token_chunk_profile_enabled") is True
                guards &= v.get("token_chunk_profile_pending") == 0
            if args.prepare_path == "cpu-known" and v["token_chunk_calls"] > 0:
                guards &= s.get("expert_cpu_known_prepare_count", 0) > 0
                guards &= (
                    s.get("expert_cpu_known_prepare_count", 0)
                    + s.get("expert_cpu_known_prepare_fallback_count", 0)
                    == v["token_chunk_calls"]
                )
            if args.profile_prepare:
                profile = v.get("prepare_profile", {})
                guards &= (
                    sum(
                        profile.get(bucket, {}).get("calls", -1)
                        for bucket in ("first_shape", "repeat_shape")
                    )
                    == v["token_chunk_calls"]
                )
                for bucket in ("first_shape", "repeat_shape"):
                    measured = profile.get(bucket, {})
                    roots = [
                        row
                        for row in measured.get("functions", [])
                        if row["name"] == "_prepare_expert_dispatch_for_core"
                    ]
                    guards &= sum(
                        row["calls"] - row["recursive_calls"] for row in roots
                    ) == measured.get("calls", -1)
        guards &= fixed["final_stats"]["shared_vmm"]["kv_to_expert_pages"] == 0
        valid &= (
            exact
            and all(math.isfinite(d) for d in differences)
            and max_delta <= args.logprob_atol
            and guards
        )
        row = {
            "round": fixed["round"],
            "first_request": fixed["first_request"],
            "exact_tokens": exact,
            "max_logprob_delta": max_delta,
            "guards_pass": bool(guards),
        }
        for arm, request in [("fixed16", fixed), ("grow17", grow)]:
            row[arm] = {
                k: request[k]
                for k in [
                    "client_e2e_s",
                    "ttft_s",
                    "prefill_forward_s",
                    "decode_s",
                    "tpot_ms",
                ]
            }
            row[arm].update(
                shared=request["shared_delta"],
                expert_materializations=request["stats_delta"].get(
                    "expert_materialize_count"
                ),
            )
            times = {e["tokens"]: e["elapsed_s"] for e in request["events"]}
            tail = args.decode_tail_start
            tail_ms = (
                (times[args.output_tokens] - times[tail])
                * 1000
                / (args.output_tokens - tail)
                if tail in times and args.output_tokens in times
                else None
            )
            row[arm]["tail_tpot_ms"] = tail_ms
        rows.append(row)
    growth_exercised = any(
        r["grow17"]["shared"]["kv_to_expert_pages"] > 0 for r in rows
    )
    result = {
        "args": execution_args,
        "analysis": {"decode_tail_start": args.decode_tail_start},
        "valid": bool(valid and growth_exercised),
        "growth_exercised": growth_exercised,
        "rows": rows,
        "repeat_medians": (
            {
                arm: {
                    metric: statistics.median(r[arm][metric] for r in rows[1:])
                    for metric in ["ttft_s", "decode_s", "tpot_ms", "client_e2e_s"]
                }
                for arm in arms
            }
            if len(rows) > 1
            else {}
        ),
    }
    dump(args.output_dir / "summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--expert-host-extra-budget-mb", type=float, default=256.0)
    p.add_argument("--expert-backing-cache-mb", type=float, default=0.0)
    p.add_argument(
        "--expert-backing-cache-accounting",
        choices=["scan", "incremental"],
        default="scan",
    )
    p.add_argument("--check-cache-accounting", action="store_true")
    p.add_argument("--skip-tokenizer-init", action="store_true")
    p.add_argument(
        "--chunk-order",
        choices=["input", "reuse", "adaptive"],
        default="input",
        help="Order token groups by input, resident reuse, or context/batch adaptive reuse.",
    )
    p.add_argument(
        "--prepare-path", choices=["generic", "cpu-known"], default="generic"
    )
    p.add_argument("--profile-chunks", action="store_true")
    p.add_argument(
        "--expert-remap-update", choices=["scalar", "batch"], default="scalar"
    )
    p.add_argument(
        "--trace-waits-request",
        type=int,
        default=0,
        help="Capture CPU/CUDA wait phases for this 1-based request; 0 disables.",
    )
    p.add_argument(
        "--expert-transfer-backend", choices=["torch", "cuda-batch"], default="torch"
    )
    p.add_argument(
        "--expert-batch-backing-layout",
        choices=["batch", "individual"],
        default="batch",
    )
    p.add_argument(
        "--expert-backing-release-validation",
        choices=["eager", "admission"],
        default="eager",
    )
    p.add_argument(
        "--expert-demand-d2h-wait", choices=["host", "stream"], default="host"
    )
    p.add_argument(
        "--profile-prepare",
        action="store_true",
        help="Collect CPU call attribution inside preparation; requires --profile-chunks.",
    )
    p.add_argument(
        "--disable-donor-cache",
        action="store_true",
        help="Rescan donors every decode step for the uncached control.",
    )
    p.add_argument("--input-tokens", type=int, default=4096)
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--decode-tail-start", type=int, default=16)
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--max-total-tokens", type=int, default=16384)
    p.add_argument("--scratch-tokens", type=int, default=4095)
    p.add_argument("--block-tokens", type=int, default=2048)
    p.add_argument("--shared-expert-layer", type=int, default=0)
    p.add_argument("--expert-initial-slots", type=int, default=16)
    p.add_argument("--reclaim-mb", type=float, default=40)
    p.add_argument("--logprob-atol", type=float, default=0.01)
    p.add_argument("--timeout-s", type=int, default=600)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--arm", choices=["fixed16", "grow17"], help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 0 <= args.trace_waits_request <= args.rounds:
        p.error("--trace-waits-request must be within the request count or zero")
    if args.trace_waits_request and (
        args.profile_chunks or args.profile_prepare or args.prepare_path != "cpu-known"
    ):
        p.error("wait tracing requires --prepare-path cpu-known and no other profilers")
    if (
        args.check_cache_accounting
        and args.expert_backing_cache_accounting != "incremental"
    ):
        p.error("--check-cache-accounting requires incremental accounting")
    if (
        not math.isfinite(args.expert_host_extra_budget_mb)
        or not math.isfinite(args.expert_backing_cache_mb)
        or not 0 <= args.expert_backing_cache_mb <= args.expert_host_extra_budget_mb
    ):
        p.error("require 0 <= backing cache <= finite host extra budget")
    if (
        args.expert_backing_release_validation == "admission"
        and args.expert_transfer_backend != "cuda-batch"
    ):
        p.error(
            "admission-time validation requires --expert-transfer-backend cuda-batch"
        )
    if args.profile_prepare and not args.profile_chunks:
        p.error("--profile-prepare requires --profile-chunks")
    if (
        args.expert_demand_d2h_wait == "stream"
        and args.expert_transfer_backend != "cuda-batch"
    ):
        p.error("stream-ordered D2H requires --expert-transfer-backend cuda-batch")
    if (
        args.expert_batch_backing_layout == "individual"
        and args.expert_transfer_backend != "cuda-batch"
    ):
        p.error("individual backing requires --expert-transfer-backend cuda-batch")
    if (
        min(
            args.rounds,
            args.input_tokens,
            args.context_length,
            args.max_total_tokens,
            args.scratch_tokens,
            args.block_tokens,
            args.expert_initial_slots,
            args.timeout_s,
        )
        <= 0
        or args.output_tokens < 2
        or not 1 <= args.decode_tail_start < args.output_tokens
        or args.gpu < 0
        or args.shared_expert_layer < 0
        or args.input_tokens + args.output_tokens > args.context_length
        or not math.isfinite(args.reclaim_mb)
        or args.reclaim_mb <= 0
        or not math.isfinite(args.logprob_atol)
        or args.logprob_atol < 0
    ):
        p.error(
            "require positive counts/pressure, output>=2, nonnegative gpu/layer/tolerance, and input+output<=context"
        )
    if args.summarize_only:
        result = summarize(args)
        print(f"valid={result['valid']}; {args.output_dir / 'summary.json'}")
        return 0 if result["valid"] else 1
    if args.arm:
        if not args.execute:
            p.error("child requires --execute")
        child(args)
        return 0
    commands = [
        [
            sys.executable,
            str(Path(__file__).resolve()),
            *[a for a in sys.argv[1:] if a != "--execute"],
            "--execute",
            "--arm",
            arm,
        ]
        for arm in ["fixed16", "grow17"]
    ]
    plan = {
        "args": vars(args),
        "commands": commands,
        "scope": "two isolated arms; first request separate; later requests are not assumed steady-state",
    }
    print(json.dumps(plan, indent=2, default=str), flush=True)
    if not args.execute:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump(args.output_dir / "plan.json", plan)
    for arm, command in zip(["fixed16", "grow17"], commands):
        with (args.output_dir / f"{arm}.log").open("w") as log:
            proc = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                rc = proc.wait(timeout=args.timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                rc = 124
        if rc:
            dump(args.output_dir / "failure.json", {"arm": arm, "returncode": rc})
            print(f"{arm} failed: {rc}", flush=True)
            return 1
        print(f"{arm} complete", flush=True)
    result = summarize(args)
    print(f"valid={result['valid']}; {args.output_dir / 'summary.json'}", flush=True)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
