#!/usr/bin/env python3
"""Bounded concurrent fixed/lend feasibility comparison. Dry plan by default.

Both arms start with identical physical expert/KV allocations. An optional
adaptive budget is a feasibility probe, not throughput acceptance. Transfer code is identical
in both arms and all effective engine arguments are written to the plan. This
driver intentionally exercises the single-layer SharedVMM path; use
``layerkv_hybrid_validation.py --expert-layer-mode generic`` for multi-layer
expert residency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from layerkv_hybrid_validation import engine_kwargs
from layerkv_shared_perf import delta, dump

PRECONDITION_TOKEN_MODULUS = 30000


def shared_loaned_kv_tokens(shared):
    """Count ordinary and scratch loans without changing admission credit."""
    return int(shared.get("blocked_kv_tokens", 0) or 0) + int(
        shared.get("scratch_blocked_tokens", 0) or 0
    )


def profile_kwargs(args):
    selected = getattr(args, "profile_round", 0)
    if not 0 <= selected <= args.rounds:
        raise ValueError("profile-round must be in [0, rounds]")
    if not selected:
        return None
    steps = getattr(args, "profile_steps", 8)
    if not 0 < steps < args.output_tokens - 1:
        raise ValueError(
            "profile-steps must be positive and leave a decode step to flush"
        )
    return dict(
        output_dir=str(args.output_dir / f"{args.arm}.trace"),
        activities=["CPU", "GPU"],
        with_stack=False,
        record_shapes=False,
        profile_by_stage=True,
        num_steps=steps,
    )


def effective_engine_args(args, arm):
    config = argparse.Namespace(**vars(args))
    config.case = "offload"
    config.batch_size = args.max_running_requests
    expert_heavy = getattr(args, "baseline_split", "kv-heavy") == "expert-heavy"
    overflow_enabled = arm == "lend"
    # A preconditioned fixed comparator is warmed through the same KV-donor
    # loan path as the timed run.  Do not encode the target as initial expert
    # slots: that would create extra physical pages and make the arms unfair.
    config.expert_extra_slots = (
        args.expert_extra_slots if arm == "lend" or expert_heavy else 0
    )
    result = engine_kwargs(config)
    result.update(
        enable_metrics=True,
        stream_interval=1,
        skip_tokenizer_init=True,
        layerkv_debug_stats=False,
        layerkv_shared_expert_free_kv_donors=True,
        layerkv_shared_expert_lend_virtual_scratch=bool(
            getattr(args, "lend_virtual_scratch", False) and arm == "lend"
        ),
        layerkv_shared_expert_retain_across_requests=(
            expert_heavy or getattr(args, "retain_experts_across_requests", False)
        ),
        layerkv_shared_expert_admission_policy=(
            "retain" if expert_heavy and arm == "fixed" else "recall"
        ),
        layerkv_shared_expert_policy=(
            getattr(args, "expert_policy", "fixed") if arm == "lend" else "fixed"
        ),
        layerkv_shared_expert_decision_interval=getattr(
            args, "expert_decision_interval", 8
        ),
        layerkv_shared_expert_headroom_steps=getattr(args, "expert_headroom_steps", 16),
        layerkv_shared_expert_kv_overflow_tokens=getattr(
            args, "expert_kv_overflow_tokens", 0
        )
        if overflow_enabled
        else 0,
        layerkv_shared_expert_kv_min_slots=(
            getattr(args, "expert_kv_min_slots", 0) if overflow_enabled else 0
        ),
        layerkv_shared_expert_benefit_horizon_steps=getattr(
            args, "expert_benefit_horizon_steps", 0
        ),
        layerkv_shared_expert_prepare_path="cpu-known",
        layerkv_shared_expert_gpu_grouping=getattr(
            args, "gpu_grouping", False
        ),
        layerkv_shared_expert_gpu_grouping_min_rows=getattr(
            args, "gpu_grouping_min_rows", 128
        ),
        layerkv_shared_expert_post_moe_prefetch=getattr(
            args, "post_moe_prefetch", False
        ),
        layerkv_shared_expert_prefetch_groups=getattr(
            args, "shared_expert_prefetch_groups", 1
        ),
        layerkv_shared_expert_profile_chunks=getattr(
            args, "profile_expert_chunks", False
        ),
        layerkv_shared_expert_chunk_order=getattr(
            args, "expert_chunk_order", "input"
        ),
        layerkv_expert_transfer_backend="cuda-batch",
        layerkv_expert_batch_backing_layout=getattr(
            args, "expert_batch_backing_layout", "individual"
        ),
        layerkv_expert_demand_d2h_wait="stream",
        layerkv_expert_backing_release_validation="admission",
        layerkv_expert_remap_update="batch",
        layerkv_expert_backing_cache_mb=args.expert_backing_cache_mb,
        layerkv_expert_cpu_backing_pool_mb=args.expert_cpu_backing_pool_mb,
        layerkv_expert_backing_cache_accounting="scan",
        layerkv_expert_cpu_backing_mode=getattr(
            args, "expert_cpu_backing_mode", "none"
        ),
        layerkv_native_moe_graph_max_batch_size=getattr(
            args, "native_moe_graph_max_batch_size", 0
        ),
    )
    if getattr(args, "profile_round", 0):
        result["layerkv_shared_expert_trace_waits"] = True
    if arm == "native":
        result = {
            key: value
            for key, value in result.items()
            if not key.startswith("layerkv_")
        }
        result["enable_layerkv"] = False
    if getattr(args, "activation_probe_seq_range", None):
        result["forward_hooks"] = [
            {
                "name": f"DEBUG-layerkv-activation-{layer}",
                "target_modules": [f"*.layers.{layer}"],
                "hook_factory": "sglang.srt.layerkv.activation_probe:create_hook",
                "with_kwargs": True,
                "config": {
                    "directory": str(args.output_dir / f"{arm}.activations"),
                    "seq_range": args.activation_probe_seq_range,
                    "layer": layer,
                    "capture_kv": layer
                    in (getattr(args, "activation_probe_kv_layers", None) or []),
                },
            }
            for layer in range(40)
        ]
    return result


def round_metrics(events, elapsed_s, before, after, submitted_s=None):
    if not events or any(not row for row in events):
        raise ValueError("missing request streaming events")
    counts = [row[-1]["tokens"] for row in events]
    submitted_s = [0.0] * len(events) if submitted_s is None else submitted_s
    if len(submitted_s) != len(events) or any(
        not math.isfinite(start) or not 0 <= start <= row[0]["elapsed_s"]
        for start, row in zip(submitted_s, events)
    ):
        raise ValueError("invalid request submission timestamps")
    forwards = (
        after["observed_decode_forward_count"] - before["observed_decode_forward_count"]
    )
    steps = (
        after["observed_decode_request_steps"] - before["observed_decode_request_steps"]
    )
    previous_histogram = {
        str(size): count
        for size, count in before["observed_decode_batch_histogram"].items()
    }
    histogram = {
        str(size): count - previous_histogram.get(str(size), 0)
        for size, count in after["observed_decode_batch_histogram"].items()
    }
    return {
        "client_e2e_s": elapsed_s,
        "output_tokens": sum(counts),
        "output_tokens_per_s": sum(counts) / elapsed_s,
        "ttft_s": [
            row[0]["elapsed_s"] - start for row, start in zip(events, submitted_s)
        ],
        "tpot_ms": [
            (
                (row[-1]["elapsed_s"] - row[0]["elapsed_s"])
                * 1000
                / (row[-1]["tokens"] - row[0]["tokens"])
                if row[-1]["tokens"] > row[0]["tokens"]
                else None
            )
            for row in events
        ],
        "stream_timing_valid": all(row[0]["tokens"] == 1 for row in events),
        "actual_decode_batch_mean": steps / forwards if forwards else None,
        "actual_decode_batch_histogram": histogram,
        "stats_delta": delta(before, after),
    }


def has_late_arrivals(args):
    return bool(args.late_after_output_tokens or getattr(args, "late_after_seconds", 0))


def precondition_expert_heavy_comparator(engine, args):
    """Make the expert-heavy arms comparable before timed rounds.

    The fixed arm reaches ``initial + extra`` through KV donor loans.  The
    adaptive arm only installs its base resident set; its extra physical KV
    pages remain KV-owned until the timed policy decides to lend them.  The
    adaptive install warmup is needed because an uninstalled LayerKV arena has
    fewer physical handles than a fixed arm whose compact expert allocation is
    already mapped. Both warmups use a token offset and a wider deterministic
    token pattern disjoint from timed inputs, so the fixed arm can discover the
    requested expert capacity instead of being limited by a tiny warmup
    vocabulary.
    """
    if not getattr(args, "precondition_expert_heavy_fixed", False):
        return None
    if getattr(args, "baseline_split", "kv-heavy") != "expert-heavy":
        raise ValueError(
            "expert-heavy preconditioning requires baseline-split expert-heavy"
        )
    if args.arm not in ("fixed", "lend"):
        return None
    target_slots = int(args.expert_initial_slots) + int(args.expert_extra_slots)
    requests = int(
        getattr(args, "precondition_requests", 0) or args.requests
    )
    input_tokens = int(
        getattr(args, "precondition_input_tokens", 0) or args.input_tokens
    )
    if args.arm == "fixed":
        output_tokens = int(
            getattr(args, "precondition_output_tokens", 0)
            or max(
                args.output_tokens,
                target_slots - int(args.expert_initial_slots) + 1,
            )
        )
        warmup_role = "fixed-target-loan"
    else:
        # One short decode crosses LayerKV installation but stays below the
        # adaptive decision interval, so it does not spend the extra KV pages
        # before the measured workload starts.
        requests = 1
        output_tokens = 2
        warmup_role = "adaptive-base-install"
    token_offset = int(getattr(args, "precondition_token_offset", 37))
    inputs = [
        [
            100
            + (token_offset + 997 * request + 313 * i)
            % PRECONDITION_TOKEN_MODULUS
            for i in range(input_tokens)
        ]
        for request in range(requests)
    ]
    max_rounds = max(1, int(getattr(args, "precondition_rounds", 4) or 4))
    attempts = 0
    shared = None
    reached = loaned = no_growth_alloc = False
    while attempts < max_rounds:
        attempts += 1
        engine.generate(
            input_ids=inputs,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": output_tokens,
                "ignore_eos": True,
            },
        )
        stats = engine.get_server_info()["internal_states"][0]["layerkv"]
        shared = stats["shared_vmm"]
        reached = shared.get("current_expert_slots") == (
            target_slots if args.arm == "fixed" else int(args.expert_initial_slots)
        )
        loaned = (
            int(shared.get("loan_bytes", 0)) > 0
            if args.arm == "fixed"
            else int(shared.get("loan_bytes", 0)) == 0
        )
        no_growth_alloc = shared.get("growth_physical_create_count") == 0
        if reached and loaned and no_growth_alloc:
            break
        # The adaptive arm intentionally performs one short install warmup;
        # repeating it would perturb the timed arm without helping the
        # fixed-target comparator. Fixed mode may need multiple untimed decode
        # windows because its legacy policy grows one slot per forward.
        if args.arm != "fixed":
            break
    assert shared is not None
    if not (reached and loaned and no_growth_alloc):
        raise RuntimeError(
            "expert-heavy preconditioning did not reach a physical KV-loan "
            "comparator: "
            f"arm={args.arm}, target_slots={target_slots}, "
            f"current_slots={shared.get('current_expert_slots')}, "
            f"loan_bytes={shared.get('loan_bytes')}, "
            f"growth_physical_create_count={shared.get('growth_physical_create_count')}"
        )
    return {
        "enabled": True,
        "arm": args.arm,
        "requests": requests,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "attempts": attempts,
        "max_rounds": max_rounds,
        "token_offset": token_offset,
        "token_modulus": PRECONDITION_TOKEN_MODULUS,
        "warmup_role": warmup_role,
        "target_slots": target_slots,
        "current_slots": shared.get("current_expert_slots"),
        "physical_bytes": shared.get("physical_bytes"),
        "loan_bytes": shared.get("loan_bytes"),
        "kv_to_expert_pages": shared.get("kv_to_expert_pages"),
        "growth_physical_create_count": shared.get("growth_physical_create_count"),
    }


# Keep the old helper name for focused tests and external experiment notebooks.
precondition_expert_heavy_fixed = precondition_expert_heavy_comparator


async def staggered_round(engine, args, inputs):
    """Submit waves using an output trigger or an explicit wall-clock offset."""
    started = time.perf_counter()
    events = [[] for _ in inputs]
    final = [None for _ in inputs]
    submitted = [None for _ in inputs]
    ready = asyncio.Event()
    triggered = False

    async def wave(indices, initial=False):
        nonlocal triggered
        for index in indices:
            submitted[index] = time.perf_counter() - started
        try:
            stream = await engine.async_generate(
                input_ids=[inputs[i] for i in indices],
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": args.output_tokens,
                    "ignore_eos": True,
                },
                return_logprob=True,
                logprob_start_len=-1,
                stream=True,
            )
            async for response in stream:
                index = indices[response["index"]]
                count = response["meta_info"]["completion_tokens"]
                events[index].append(
                    {"elapsed_s": time.perf_counter() - started, "tokens": count}
                )
                final[index] = response
                if (
                    args.late_after_output_tokens
                    and index == 0
                    and count >= args.late_after_output_tokens
                ):
                    triggered = True
                    ready.set()
        finally:
            if initial:
                ready.set()  # Propagate an early failure instead of waiting forever.

    tasks = [asyncio.create_task(wave(list(range(args.initial_requests)), True))]
    timer = None
    try:
        delay = getattr(args, "late_after_seconds", 0)
        if delay:
            timer = asyncio.create_task(
                asyncio.sleep(max(0, delay - (time.perf_counter() - started)))
            )
            await asyncio.wait([tasks[0], timer], return_when=asyncio.FIRST_COMPLETED)
            if tasks[0].done():
                await tasks[0]  # Fail promptly; successful early completion is valid.
            await timer
        else:
            await ready.wait()
            if not triggered:
                await tasks[0]
                raise RuntimeError("initial wave ended before the arrival trigger")
        tasks.append(
            asyncio.create_task(wave(list(range(args.initial_requests, len(inputs)))))
        )
        await asyncio.gather(*tasks)
    finally:
        if timer is not None:
            tasks.append(timer)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return events, final, time.perf_counter() - started, submitted


ROUND_WORKLOAD_FIELDS = (
    "requests",
    "input_tokens",
    "output_tokens",
    "initial_requests",
    "late_input_tokens",
    "late_after_seconds",
    "late_after_output_tokens",
)


def validate_round_workloads(args):
    workloads = getattr(args, "round_workloads", None)
    if workloads is None:
        return
    if not isinstance(workloads, list) or len(workloads) != args.rounds:
        raise ValueError("round-workloads requires one object per round")
    if getattr(args, "round_output_triggers", None) is not None or getattr(
        args, "profile_round", 0
    ):
        raise ValueError(
            "round-workloads cannot combine with round-output-triggers or profiling"
        )
    for index, spec in enumerate(workloads):
        if (
            not isinstance(spec, dict)
            or set(spec) - set(ROUND_WORKLOAD_FIELDS)
            or not {"requests", "input_tokens", "output_tokens"} <= set(spec)
        ):
            raise ValueError(
                f"round {index + 1}: require requests/input_tokens/output_tokens and only workload fields"
            )
        cfg = effective_round_args(args, index)
        for key in ROUND_WORKLOAD_FIELDS:
            value = getattr(cfg, key)
            if key == "late_after_seconds":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"round {index + 1}: invalid {key}")
            elif type(value) is not int or value < (
                0 if key == "late_after_output_tokens" else 1
            ):
                raise ValueError(f"round {index + 1}: invalid {key}")
        if (
            cfg.output_tokens < 2
            or max(cfg.input_tokens, cfg.late_input_tokens) + cfg.output_tokens
            > args.context_length
        ):
            raise ValueError(
                f"round {index + 1}: output >=2 and input+output must fit context"
            )
        if cfg.late_after_seconds and cfg.late_after_output_tokens:
            raise ValueError(
                f"round {index + 1}: arrival triggers are mutually exclusive"
            )
        if has_late_arrivals(cfg):
            if (
                not 0 < cfg.initial_requests < cfg.requests
                or cfg.late_after_output_tokens >= cfg.output_tokens
            ):
                raise ValueError(
                    f"round {index + 1}: invalid staggered arrival counts/trigger"
                )
        elif cfg.initial_requests != 1 or cfg.late_input_tokens != cfg.input_tokens:
            raise ValueError(f"round {index + 1}: late settings require a trigger")


def round_workload(args):
    return {name: getattr(args, name) for name in ROUND_WORKLOAD_FIELDS}


def effective_round_args(args, index):
    """Optional diagnostic trigger replay; zero retains the base arrival mode."""
    result = argparse.Namespace(**vars(args))
    workloads = getattr(args, "round_workloads", None)
    if workloads is not None:
        spec = workloads[index]
        # Arrival settings do not leak from the previous round or base CLI.
        result.initial_requests = 1
        result.late_input_tokens = spec["input_tokens"]
        result.late_after_seconds = 0
        result.late_after_output_tokens = 0
        vars(result).update(spec)
        return result
    triggers = getattr(args, "round_output_triggers", None)
    if triggers is not None and triggers[index]:
        result.late_after_seconds = 0
        result.late_after_output_tokens = triggers[index]
    return result


def child(args):
    import sglang as sgl

    # Direct ``--execute --arm`` invocations must be self-contained; the
    # parent comparison path creates this directory before spawning children.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = effective_engine_args(args, args.arm)
    dump(args.output_dir / f"{args.arm}.engine_args.json", kwargs)
    engine = sgl.Engine(**kwargs)
    profiling = profile_kwargs(args)
    graph_probe = bool(getattr(args, "native_moe_graph_max_batch_size", 0))
    chunk_probe = bool(getattr(args, "profile_expert_chunks", False))
    result = {
        "rounds": [],
        "diagnostic_only": bool(profiling or graph_probe or chunk_probe),
        "profile": profiling,
        "native_moe_graph_probe": graph_probe,
    }
    source_args = args
    try:
        result["precondition"] = precondition_expert_heavy_comparator(
            engine, source_args
        )
        info = engine.get_server_info()["internal_states"][0]
        before = None if args.arm == "native" else info["layerkv"]
        result["initial_stats"] = before
        if (
            graph_probe
            and args.arm != "native"
            and not before.get("native_moe_graph", {}).get("installed_layers")
        ):
            raise RuntimeError(
                "native MoE graph requested but no resident cores installed"
            )
        for index in range(args.rounds):
            args = effective_round_args(source_args, index)
            inputs = [
                [
                    100 + (i + 7 * index + 13 * request) % 80
                    for i in range(
                        args.late_input_tokens
                        if has_late_arrivals(args) and request >= args.initial_requests
                        else args.input_tokens
                    )
                ]
                for request in range(args.requests)
            ]
            events = [[] for _ in inputs]
            final = [None for _ in inputs]
            if profiling and index + 1 == args.profile_round:
                engine.start_profile(**profiling)
            started = time.perf_counter()
            submitted = [0.0] * len(inputs)
            if has_late_arrivals(args):
                events, final, elapsed, submitted = engine.loop.run_until_complete(
                    staggered_round(engine, args, inputs)
                )
                responses = ()
            else:
                responses = engine.generate(
                    input_ids=inputs,
                    sampling_params={
                        "temperature": 0,
                        "max_new_tokens": args.output_tokens,
                        "ignore_eos": True,
                    },
                    return_logprob=True,
                    logprob_start_len=-1,
                    stream=True,
                )
            for response in responses:
                request = response["index"]
                events[request].append(
                    {
                        "elapsed_s": time.perf_counter() - started,
                        "tokens": response["meta_info"]["completion_tokens"],
                    }
                )
                final[request] = response
            if not has_late_arrivals(args):
                elapsed = time.perf_counter() - started
            info = engine.get_server_info()["internal_states"][0]
            after = None if args.arm == "native" else info["layerkv"]
            row = (
                {"client_e2e_s": elapsed, "reference_only": True}
                if args.arm == "native"
                else round_metrics(events, elapsed, before, after, submitted)
            )
            row.update(
                round=index + 1,
                workload=round_workload(args),
                first_round=index == 0,
                profiled=bool(profiling and index + 1 == args.profile_round),
                timing_acceptance_eligible=not (
                    profiling or graph_probe or chunk_probe
                ),
                events=events,
                responses=final,
                final_stats=after,
                submitted_s=submitted,
                input_tokens=[len(item) for item in inputs],
                effective_late_after_output_tokens=args.late_after_output_tokens,
                effective_late_after_seconds=args.late_after_seconds,
                arrival_mode=(
                    "fixed-time"
                    if getattr(args, "late_after_seconds", 0)
                    else (
                        "output-token-triggered diagnostic"
                        if args.late_after_output_tokens
                        else "simultaneous"
                    )
                ),
                stagger_overlap_observed=(
                    min(submitted[args.initial_requests :])
                    < max(
                        event[-1]["elapsed_s"]
                        for event in events[: args.initial_requests]
                    )
                    if has_late_arrivals(args)
                    else None
                ),
                late_submission_lag_s=(
                    [
                        value - args.late_after_seconds
                        for value in submitted[args.initial_requests :]
                    ]
                    if getattr(args, "late_after_seconds", 0)
                    else None
                ),
            )
            result["rounds"].append(row)
            dump(args.output_dir / f"{args.arm}.json", result)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in row.items()
                        if k
                        not in ("events", "responses", "final_stats", "stats_delta")
                    }
                ),
                flush=True,
            )
            before = after
    finally:
        try:
            engine.shutdown()
        except OSError as error:
            # The round artifact is written before cleanup.  A pidfd cleanup
            # failure should not turn a completed arm into a benchmark failure
            # or prevent the parent from launching the other arm.
            print(f"engine shutdown warning: {error}", file=sys.stderr, flush=True)


def summarize(args):
    arms = {
        arm: json.loads((args.output_dir / f"{arm}.json").read_text())
        for arm in ("fixed", "lend")
    }
    valid = all(len(data["rounds"]) == args.rounds for data in arms.values())
    initial_physical_bytes = [
        data["initial_stats"]["shared_vmm"]["physical_bytes"] for data in arms.values()
    ]
    physical_pool_matched = initial_physical_bytes[0] == initial_physical_bytes[1]
    graph_budget_unverified = bool(
        getattr(args, "native_moe_graph_max_batch_size", 0)
    ) or any(
        data.get("native_moe_graph_probe", False)
        or snapshot.get("native_moe_graph", {}).get("max_batch_size", 0)
        for data in arms.values()
        for snapshot in [
            data["initial_stats"],
            *[row["final_stats"] for row in data["rounds"]],
        ]
    )
    # Rebuild derived metrics from raw counters, also handling old artifacts
    # whose in-process histogram subtraction used the wrong dictionary key type.
    for data in arms.values():
        before = data["initial_stats"]
        for row in data["rounds"]:
            row.update(
                round_metrics(
                    row["events"],
                    row["client_e2e_s"],
                    before,
                    row["final_stats"],
                    row.get("submitted_s"),
                )
            )
            before = row["final_stats"]
    rows = []
    for index, (fixed, lend) in enumerate(
        zip(arms["fixed"]["rounds"], arms["lend"]["rounds"])
    ):
        round_args = effective_round_args(args, index)
        exact = len(fixed["responses"]) == len(lend["responses"]) == round_args.requests
        max_error = 0.0
        for left, right in zip(fixed["responses"], lend["responses"]):
            a, b = (r["meta_info"]["output_token_logprobs"] for r in (left, right))
            exact &= len(a) == len(b) == round_args.output_tokens and [
                t[1] for t in a
            ] == [t[1] for t in b]
            errors = [abs(x[0] - y[0]) for x, y in zip(a, b)]
            max_error = (
                max(max_error, max(errors, default=math.inf))
                if all(math.isfinite(e) for e in errors)
                else math.inf
            )
        guards = True
        for row in (fixed, lend):
            if getattr(args, "round_workloads", None) is not None:
                guards &= row.get("workload") == round_workload(round_args)
            s = row["final_stats"]
            v = s["shared_vmm"]
            guards &= all(
                s.get(k) is True for k in ("kvc_guard_pass", "expert_guard_pass")
            )
            guards &= all(
                s.get(k) == 0
                for k in (
                    "kvc_stale_entry_count",
                    "kvc_page_alignment_violation_count",
                    "kvc_physical_failure_count",
                )
            )
            guards &= v["ownership_guard_pass"] and v["expert_pointers_stable"]
            expert_heavy = getattr(args, "baseline_split", "kv-heavy") == "expert-heavy"
            retained = expert_heavy or getattr(
                args, "retain_experts_across_requests", False
            )
            guards &= v["growth_physical_create_count"] == 0
            if retained:
                guards &= v.get("retain_across_requests") is True
                guards &= v.get("admission_policy") == (
                    "retain" if expert_heavy and row is fixed else "recall"
                )
                guards &= (v["loan_bytes"] > 0) == (
                    shared_loaned_kv_tokens(v) > 0
                )
                if row is fixed:
                    guards &= v.get("current_expert_slots") == (
                        args.expert_initial_slots
                        + (args.expert_extra_slots if expert_heavy else 0)
                    )
                    if not expert_heavy:
                        guards &= v["loan_bytes"] == 0 and shared_loaned_kv_tokens(v) == 0
            else:
                guards &= v["loan_bytes"] == 0 and shared_loaned_kv_tokens(v) == 0
            guards &= row["stream_timing_valid"]
            if getattr(round_args, "late_after_output_tokens", 0):
                guards &= row["stagger_overlap_observed"] is True
            guards &= (
                row["output_tokens"] == round_args.requests * round_args.output_tokens
            )
            guards &= row["stats_delta"][
                "observed_decode_request_steps"
            ] == round_args.requests * (round_args.output_tokens - 1)
            guards &= (
                sum(row["actual_decode_batch_histogram"].values())
                == row["stats_delta"]["observed_decode_forward_count"]
            )
        round_pool_matched = (
            fixed["final_stats"]["shared_vmm"]["physical_bytes"]
            == lend["final_stats"]["shared_vmm"]["physical_bytes"]
        )
        physical_pool_matched &= round_pool_matched
        valid &= exact and max_error <= args.logprob_atol and guards
        rows.append(
            {
                "round": fixed["round"],
                "first_round": fixed["first_round"],
                "tokens_exact": exact,
                "max_logprob_delta": max_error,
                "guards": guards,
                "physical_pool_matched": round_pool_matched,
                "throughput_gain_percent": (
                    lend["output_tokens_per_s"] / fixed["output_tokens_per_s"] - 1
                )
                * 100,
                "fixed": {
                    k: fixed[k]
                    for k in (
                        "output_tokens_per_s",
                        "ttft_s",
                        "tpot_ms",
                        "actual_decode_batch_mean",
                        "actual_decode_batch_histogram",
                    )
                },
                "lend": {
                    k: lend[k]
                    for k in (
                        "output_tokens_per_s",
                        "ttft_s",
                        "tpot_ms",
                        "actual_decode_batch_mean",
                        "actual_decode_batch_histogram",
                    )
                },
            }
        )
    return {
        "valid": bool(
            valid
            and physical_pool_matched
            and not getattr(args, "activation_probe_seq_range", None)
            and not getattr(args, "profile_round", 0)
            and not graph_budget_unverified
            and not getattr(args, "profile_expert_chunks", False)
            and not any(data.get("diagnostic_only", False) for data in arms.values())
        ),
        "physical_pool_matched": physical_pool_matched,
        "graph_budget_unverified": graph_budget_unverified,
        "scope": "fixed versus free-KV lending/budget feasibility; not fixed-arrival or 20% acceptance",
        "rounds": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--profile-expert-chunks",
        action="store_true",
        help="Diagnostic selected-layer chunk timings split by prefill/decode; not throughput acceptance.",
    )
    parser.add_argument(
        "--round-workloads",
        type=json.loads,
        help="JSON list of per-round workload objects: requests, input_tokens, output_tokens; optional arrival fields. One Engine and unchanged memory configuration across rounds.",
    )
    parser.add_argument(
        "--native-moe-graph-max-batch-size",
        type=int,
        default=0,
        help="Resident MoE replay probe in both arms; 0 disables. Diagnostic until total physical memory is validated.",
    )
    parser.add_argument(
        "--profile-round",
        type=int,
        default=0,
        help="Diagnostic CPU/GPU stage profile on this round; 0 disables. Timings are not acceptance results.",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=8,
        help="Maximum forwards per profiled stage; automatically flushed by the scheduler.",
    )
    parser.add_argument(
        "--activation-probe-seq-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Diagnostic synchronous Qwen3.6 40-layer decode snapshots; invalidates timing measurements.",
    )
    parser.add_argument(
        "--activation-probe-kv-layers",
        type=int,
        nargs="+",
        help="Also snapshot actual Triton KV gathers for these Qwen full-attention layers; requires activation-probe-seq-range.",
    )
    for name, default in (
        ("gpu", 0),
        ("requests", 4),
        ("max-running-requests", 4),
        ("rounds", 2),
        ("input-tokens", 2048),
        ("output-tokens", 16),
        ("context-length", 8192),
        ("max-total-tokens", 16384),
        ("scratch-tokens", 8192),
        ("block-tokens", 2048),
        ("shared-expert-layer", 0),
        ("expert-initial-slots", 16),
        ("expert-extra-slots", 1),
        ("expert-kv-overflow-tokens", 0),
        ("expert-kv-min-slots", 0),
        ("timeout-s", 600),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--reclaim-mb", type=float, default=40)
    parser.add_argument("--expert-backing-cache-mb", type=float, default=128)
    parser.add_argument("--expert-cpu-backing-pool-mb", type=float, default=128)
    parser.add_argument(
        "--expert-batch-backing-layout",
        choices=("batch", "individual"),
        default="individual",
        help="CPU D2H backing layout; batch reuses complete pinned owners after all row views retire.",
    )
    parser.add_argument(
        "--expert-prefetch-lookahead-layers",
        type=int,
        default=1,
        help="Maximum number of predicted expert layers queued for H2D prefetch at one decode safe point.",
    )
    parser.add_argument(
        "--expert-prefetch-min-route-overlap",
        type=float,
        default=0.5,
        help="Minimum consecutive decode-route overlap required for previous-route expert H2D prefetch.",
    )
    parser.add_argument("--logprob-atol", type=float, default=0.01)
    parser.add_argument(
        "--expert-policy", choices=("fixed", "adaptive"), default="fixed"
    )
    parser.add_argument(
        "--expert-chunk-order",
        choices=("input", "reuse", "adaptive"),
        default="input",
        help="Order routed token groups; adaptive uses context length and actual batch to choose resident reuse.",
    )
    parser.add_argument(
        "--gpu-grouping",
        action="store_true",
        help="Build shared-expert token groups with the CUDA route kernel; admission and slot ownership remain CPU-controlled.",
    )
    parser.add_argument(
        "--gpu-grouping-min-rows",
        type=int,
        default=128,
        help="Minimum routed rows for CUDA grouping; smaller batches use the measured faster CPU first-fit path.",
    )
    parser.add_argument(
        "--post-moe-prefetch",
        action="store_true",
        help="Submit dead-slot successor expert H2D after the current MoE event; diagnostic opt-in.",
    )
    parser.add_argument(
        "--shared-expert-prefetch-groups",
        type=int,
        default=1,
        help=(
            "Bounded number of future token-route groups included in one "
            "shared-expert prefetch submission; 1 is the immediate successor."
        ),
    )
    parser.add_argument("--expert-decision-interval", type=int, default=8)
    parser.add_argument(
        "--baseline-split",
        choices=("kv-heavy", "expert-heavy"),
        default="kv-heavy",
        help="Expert-heavy retains borrowed pages across requests in both arms; only the lending arm recalls for admission. Add --precondition-expert-heavy for symmetric physical-pool setup before timing.",
    )
    parser.add_argument(
        "--precondition-expert-heavy-fixed",
        "--precondition-expert-heavy",
        dest="precondition_expert_heavy_fixed",
        action="store_true",
        help="Run symmetric untimed expert-heavy setup: fixed reaches initial+extra through KV donor loans, while adaptive installs only its base slots; requires --baseline-split expert-heavy.",
    )
    parser.add_argument(
        "--precondition-requests",
        type=int,
        default=0,
        help="Requests in the untimed expert-heavy fixed warmup; 0 reuses --requests.",
    )
    parser.add_argument(
        "--precondition-input-tokens",
        type=int,
        default=0,
        help="Input tokens per request in the untimed fixed warmup; 0 reuses --input-tokens.",
    )
    parser.add_argument(
        "--precondition-token-offset",
        type=int,
        default=37,
        help="Token-pattern offset for untimed preconditioning; keeps warmup inputs distinct from timed rounds.",
    )
    parser.add_argument(
        "--precondition-output-tokens",
        type=int,
        default=0,
        help="Output tokens in the untimed fixed warmup; 0 uses max(output-tokens, extra-slots + 1).",
    )
    parser.add_argument(
        "--precondition-rounds",
        type=int,
        default=4,
        help="Maximum untimed warmup windows for expert-heavy fixed preconditioning; stops once the target is reached.",
    )
    parser.add_argument("--expert-headroom-steps", type=int, default=16)
    parser.add_argument(
        "--lend-virtual-scratch",
        action="store_true",
        help=(
            "Lend complete idle virtual-KVC scratch pages to the lend arm; "
            "KVC use recalls them before access."
        ),
    )
    parser.add_argument(
        "--expert-cpu-backing-mode",
        choices=("none", "selected"),
        default="none",
        help="Keep immutable pinned weights for the selected expert layer; host-memory cost is reported separately.",
    )
    parser.add_argument(
        "--expert-benefit-horizon-steps",
        type=int,
        default=0,
        help="Future benefit horizon cap for fixed-length decode; 0 preserves single-interval pricing.",
    )
    parser.add_argument(
        "--retain-experts-across-requests",
        action="store_true",
        help="Retain adaptive expert loans across request boundaries also with a KV-heavy fixed baseline. Admission still recalls; does not change fixed capacity.",
    )
    parser.add_argument("--initial-requests", type=int, default=1)
    parser.add_argument(
        "--round-output-triggers",
        type=int,
        nargs="+",
        default=None,
        help="Diagnostic per-round output-token triggers; one per round, zero inherits base arrival settings. Not a fixed-arrival performance comparison.",
    )
    parser.add_argument("--late-input-tokens", type=int, default=None)
    parser.add_argument(
        "--late-after-seconds",
        type=float,
        default=0,
        help="Submit the late wave at this offset from round start, independent of output progress.",
    )
    parser.add_argument(
        "--late-after-output-tokens",
        type=int,
        default=0,
        help="Diagnostic: submit remaining requests after request 0 streams this many tokens; 0 submits all at once.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--arm", choices=("fixed", "lend", "native"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--arm-order",
        nargs=2,
        choices=("fixed", "lend"),
        default=("fixed", "lend"),
        metavar=("FIRST", "SECOND"),
        help="Order in which the two child arms run; summary remains fixed versus lend.",
    )
    args = parser.parse_args()
    if args.expert_benefit_horizon_steps < 0:
        parser.error("expert-benefit-horizon-steps must be nonnegative")
    if args.arm_order[0] == args.arm_order[1]:
        parser.error("arm-order must contain fixed and lend exactly once")
    if args.native_moe_graph_max_batch_size < 0:
        parser.error("native-moe-graph-max-batch-size must be nonnegative")
    if args.expert_prefetch_lookahead_layers <= 0:
        parser.error("expert-prefetch-lookahead-layers must be positive")
    if args.gpu_grouping_min_rows <= 0:
        parser.error("gpu-grouping-min-rows must be positive")
    if args.shared_expert_prefetch_groups <= 0:
        parser.error("shared-expert-prefetch-groups must be positive")
    if not math.isfinite(args.expert_prefetch_min_route_overlap) or not (
        0.0 <= args.expert_prefetch_min_route_overlap <= 1.0
    ):
        parser.error("expert-prefetch-min-route-overlap must be in [0.0, 1.0]")
    if args.precondition_expert_heavy_fixed and args.baseline_split != "expert-heavy":
        parser.error(
            "--precondition-expert-heavy-fixed requires --baseline-split expert-heavy"
        )
    if any(
        value < 0
        for value in (
            args.precondition_requests,
            args.precondition_input_tokens,
            args.precondition_output_tokens,
        )
    ) or args.precondition_rounds <= 0:
        parser.error("precondition counts must be nonnegative and rounds must be positive")
    try:
        profile_kwargs(args)
    except ValueError as error:
        parser.error(str(error))
    if args.activation_probe_kv_layers and (
        not args.activation_probe_seq_range
        or any(
            layer not in range(3, 40, 4) for layer in args.activation_probe_kv_layers
        )
    ):
        parser.error(
            "activation-probe-kv-layers requires a sequence range and full-attention layers 3,7,...,39"
        )
    if args.activation_probe_seq_range and not (
        0 < args.activation_probe_seq_range[0] <= args.activation_probe_seq_range[1]
    ):
        parser.error("activation-probe-seq-range requires 0 < MIN <= MAX")
    if args.round_output_triggers is not None and (
        len(args.round_output_triggers) != args.rounds
        or any(
            not 0 <= value < args.output_tokens for value in args.round_output_triggers
        )
        or not 0 < args.initial_requests < args.requests
    ):
        parser.error(
            "round-output-triggers requires one value in [0, output-tokens) per round and 0 < initial-requests < requests"
        )
    if not math.isfinite(args.late_after_seconds) or args.late_after_seconds < 0:
        parser.error("late-after-seconds must be finite and nonnegative")
    if args.late_after_seconds and args.late_after_output_tokens:
        parser.error("time and output-token arrival triggers are mutually exclusive")
    if args.late_input_tokens is None:
        args.late_input_tokens = args.input_tokens
    if has_late_arrivals(args):
        if not (
            0 < args.initial_requests < args.requests
            and (
                args.late_after_seconds
                or 0 < args.late_after_output_tokens < args.output_tokens
            )
        ):
            parser.error(
                "staggered arrivals require 0 < initial-requests < requests and either a positive time offset or 0 < late-after-output-tokens < output-tokens"
            )
    elif args.late_input_tokens != args.input_tokens or args.initial_requests != 1:
        parser.error(
            "late-input-tokens/initial-requests require a nonzero arrival trigger"
        )
    if (
        args.late_input_tokens <= 0
        or args.late_input_tokens + args.output_tokens > args.context_length
    ):
        parser.error(
            "late input must be positive and input+output must fit context-length"
        )
    positive = (
        "expert_decision_interval",
        "expert_headroom_steps",
        "requests",
        "max_running_requests",
        "rounds",
        "input_tokens",
        "context_length",
        "max_total_tokens",
        "scratch_tokens",
        "block_tokens",
        "expert_initial_slots",
        "expert_extra_slots",
        "timeout_s",
    )
    if (
        any(getattr(args, key) <= 0 for key in positive)
        or args.output_tokens < 2
        or min(args.gpu, args.shared_expert_layer) < 0
    ):
        parser.error(
            "counts must be positive, output-tokens >=2, GPU/layer nonnegative"
        )
    if args.input_tokens + args.output_tokens > args.context_length:
        parser.error("input-tokens + output-tokens must fit context-length")
    if args.precondition_expert_heavy_fixed:
        effective_precondition_input = (
            args.precondition_input_tokens or args.input_tokens
        )
        effective_precondition_output = args.precondition_output_tokens or max(
            args.output_tokens,
            args.expert_extra_slots + 1,
        )
        if (
            effective_precondition_input <= 0
            or effective_precondition_output < 2
            or effective_precondition_input + effective_precondition_output
            > args.context_length
        ):
            parser.error(
                "precondition input-tokens + output-tokens must fit context-length"
            )
    if (
        not math.isfinite(args.reclaim_mb)
        or args.reclaim_mb <= 0
        or any(
            not math.isfinite(v) or v < 0
            for v in (
                args.logprob_atol,
                args.expert_backing_cache_mb,
                args.expert_cpu_backing_pool_mb,
            )
        )
    ):
        parser.error(
            "require finite positive reclaim and finite nonnegative cache/pool/tolerance"
        )
    try:
        validate_round_workloads(args)
    except ValueError as error:
        parser.error(str(error))
    if args.arm:
        if not args.execute:
            parser.error("child requires --execute")
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
        for arm in args.arm_order
    ]
    plan = {
        "args": vars(args),
        "effective_round_workloads": [
            round_workload(effective_round_args(args, index))
            for index in range(args.rounds)
        ],
        "engine_args": {
            arm: effective_engine_args(args, arm) for arm in ("fixed", "lend")
        },
        "commands": commands,
    }
    print(json.dumps(plan, indent=2, default=str), flush=True)
    if not args.execute:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump(args.output_dir / "plan.json", plan)
    for arm, command in zip(args.arm_order, commands):
        with (args.output_dir / f"{arm}.log").open("w") as log:
            proc = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = proc.wait(timeout=args.timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = 124
        if code:
            dump(args.output_dir / "failure.json", {"arm": arm, "returncode": code})
            return 1
        print(f"{arm} complete", flush=True)
    result = summarize(args)
    dump(args.output_dir / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
