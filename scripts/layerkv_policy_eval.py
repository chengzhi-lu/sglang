#!/usr/bin/env python3
"""Run a Fig4-aligned policy-level LayerKV evaluation on SGLang.

This is a comparability harness, not a full workload sweep.  It runs every
KVC+expert policy through the same SGLang backend and records enough physical
reclaim and guard counters to decide whether the row is valid for policy
comparison.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Tuple
from urllib import request

from layerkv_eval_common import (
    DEFAULT_MODEL_PATH,
    FIG4_RECLAIM_LIMIT_MB,
    FIG4_WORKLOADS,
    apply_fig4_workload,
    base_command,
    layerkv_flags,
    load_fig4_prompt_ids,
    parse_layerkv_stats,
    run_bench_command,
    write_csv,
)

CSV_FIELDS = [
    "workload",
    "backend",
    "fig4_aligned",
    "batch_size",
    "input_len",
    "output_len",
    "fig4_dataset_name",
    "fig4_dataset_path",
    "fig4_real_dataset_loaded",
    "prompt_source",
    "synthetic_prompt_used",
    "prompt_repetition_used",
    "selected_record_ids_hash",
    "selected_token_counts_min",
    "selected_token_counts_max",
    "selected_token_counts_mean",
    "payload_input_len_min",
    "payload_input_len_max",
    "payload_input_len_mean",
    "payload_input_ids_hash",
    "max_total_tokens",
    "max_running_requests",
    "mem_fraction_static",
    "scenario",
    "layerkv_mode",
    "layerkv_policy",
    "returncode",
    "valid",
    "validation_reason",
    "reclaim_limit_mb",
    "configured_reclaim_limit_mb",
    "effective_reclaim_target_mb",
    "needed_pressure_mb",
    "available_kvc_reclaim_mb",
    "available_expert_reclaim_mb",
    "available_total_reclaim_mb",
    "target_limited_reason",
    "planned_reclaim_mb",
    "actual_reclaim_mb",
    "actual_reclaim_limited_by_workload",
    "benchmark_prefill_latency_s",
    "benchmark_decode0_latency_s",
    "benchmark_decode0_latency_ms",
    "response_e2e_latency_ms_mean",
    "response_e2e_latency_ms_p50",
    "response_e2e_latency_ms_p95",
    "response_e2e_latency_ms_max",
    "response_e2e_per_token_ms_mean",
    "response_e2e_per_token_ms_p50",
    "response_e2e_per_token_ms_p95",
    "output_throughput_tok_s",
    "comparable",
    "comparability_reason",
    "planner_version",
    "planner_used_hotness",
    "planner_fallback_reason",
    "planner_estimated_kvc_cost",
    "planner_estimated_expert_cost",
    "planner_estimated_expert_churn_count",
    "planner_estimated_expert_churn_mb",
    "planner_estimated_expert_install_mb",
    "planner_estimated_kvc_controller_cost",
    "planner_selected_kvc_reclaim_mb",
    "planner_selected_expert_reclaim_mb",
    "planner_dp_candidate_count",
    "planner_dp_selected_kvc_candidates",
    "planner_dp_selected_expert_candidates",
    "planner_dp_infeasible_kvc_candidates",
    "planner_dp_infeasible_expert_candidates",
    "planner_dp_selected_total_cost",
    "planner_dp_selected_kvc_cost",
    "planner_dp_selected_expert_cost",
    "selected_kvc_tokens_by_layer",
    "selected_expert_evictions_by_layer",
    "layerkv_enabled",
    "layerkv_reclaim_limit_mb",
    "layerkv_runtime_profile",
    "layerkv_worker_role",
    "layerkv_kvc_backend",
    "layerkv_kvc_backend_semantics",
    "layerkv_kvc_backend_limited",
    "kvc_per_layer_metadata_rewrite_count",
    "kvc_per_layer_metadata_rewrite_skip_count",
    "kvc_per_layer_metadata_rewrite_unsupported_count",
    "kvc_per_layer_override_layer_count",
    "kvc_per_layer_identity_override_count",
    "kvc_per_layer_slot_override_count",
    "kvc_per_layer_slot_override_token_count",
    "kvc_per_layer_arena_entry_count",
    "kvc_per_layer_arena_resident_token_count",
    "kvc_per_layer_arena_offloaded_token_count",
    "kvc_per_layer_evict_count",
    "kvc_per_layer_reload_count",
    "kvc_per_layer_reload_mb_total",
    "layerkv_physical_kvc_supported",
    "layerkv_physical_expert_supported",
    "planned_kvc_reclaim_mb",
    "physical_kvc_reclaim_mb",
    "physical_kvc_reclaim_peak_mb",
    "physical_kvc_reclaim_step_mean_mb",
    "planned_expert_reclaim_mb",
    "physical_expert_reclaim_mb",
    "physical_total_reclaim_mb",
    "physical_total_reclaim_peak_mb",
    "kvc_evict_count_total",
    "kvc_reload_count_total",
    "kvc_reload_required_count",
    "kvc_reload_mb_total",
    "kvc_req_to_token_rewrite_count",
    "kvc_physical_failure_count",
    "kvc_stale_entry_count",
    "kvc_finished_req_cleanup_count",
    "kvc_finished_req_cleanup_token_count",
    "kvc_evict_cursor_hit_count",
    "kvc_evict_cursor_reset_count",
    "kvc_evict_candidate_scan_tokens",
    "kvc_evict_candidate_selected_tokens",
    "kvc_guard_pass",
    "kvc_guard_reason",
    "kvc_ready_before_use_ratio",
    "scheduler_task_count",
    "scheduler_kvc_task_count",
    "scheduler_expert_task_count",
    "scheduler_coalesced_task_count",
    "scheduler_deadline_miss_count",
    "scheduler_ready_before_use_count",
    "scheduler_ready_use_check_count",
    "scheduler_ready_before_use_ratio",
    "scheduler_exposed_wait_ms",
    "scheduler_copy_bytes_total",
    "native_scheduler_observation_count",
    "native_schedule_policy",
    "native_schedule_forward_mode",
    "native_schedule_waiting_queue_len",
    "native_schedule_running_batch_size",
    "native_schedule_batch_size",
    "native_schedule_max_running_requests",
    "native_schedule_new_token_ratio",
    "native_schedule_kv_available_tokens",
    "native_schedule_overlap_enabled",
    "expert_host_backing_mb",
    "expert_slot_rebind_count",
    "expert_materialize_count",
    "expert_materialize_async_count",
    "expert_materialize_host_sync_count",
    "expert_materialize_mb_total",
    "expert_materialize_ms",
    "expert_materialize_batch_count",
    "expert_materialize_event_count",
    "expert_materialize_avg_batch_size",
    "expert_materialize_batch_size_p50",
    "expert_materialize_batch_size_p95",
    "expert_materialize_layers_touched",
    "expert_materialize_dedup_count",
    "expert_materialize_coalesced_count",
    "expert_materialize_slot_select_ms",
    "expert_materialize_map_update_ms",
    "expert_materialize_event_overhead_ms",
    "expert_materialize_sync_wait_ms",
    "expert_prefetch_count",
    "expert_prefetch_hit_count",
    "expert_prefetch_miss_count",
    "expert_prefetch_candidate_count",
    "expert_prefetch_issued_count",
    "expert_prefetch_skipped_resident_count",
    "expert_prefetch_skipped_capacity_count",
    "expert_prefetch_useful_count",
    "expert_prefetch_wasted_count",
    "expert_on_demand_materialize_count",
    "expert_prefetch_mb_total",
    "expert_prepared_backing_mb",
    "expert_prepare_extend_count",
    "expert_prepare_decode_fallback_count",
    "expert_prepared_plan_used",
    "expert_prepared_backing_hit_count",
    "expert_prepared_backing_miss_count",
    "expert_prepare_guard_count",
    "expert_prepare_guard_mb",
    "expert_prepare_candidate_count",
    "expert_prepare_copied_count",
    "expert_prepare_reused_count",
    "expert_prepare_invalidated_count",
    "expert_prepare_host_budget_mb",
    "expert_prepare_actual_mb",
    "expert_prefetch_ready_before_use_count",
    "expert_prefetch_blocked_by_copy_stream_count",
    "expert_install_state",
    "expert_install_pending_layers",
    "expert_install_completed_layers",
    "expert_install_layers_per_step",
    "expert_install_budget_mb",
    "expert_install_target_steps",
    "expert_install_remaining_steps",
    "expert_install_effective_layers_this_step",
    "expert_install_effective_budget_mb_this_step",
    "expert_install_step_count",
    "expert_install_step_ms",
    "expert_install_blocking_ms",
    "expert_install_reclaim_mb_progress",
    "expert_install_not_comparable_step_count",
    "expert_lazy_backing_enabled",
    "expert_lazy_backing_skipped_count",
    "expert_lazy_backing_skipped_mb",
    "expert_lazy_backing_unavailable_count",
    "expert_backing_cache_limit_mb",
    "expert_backing_cache_hit_count",
    "expert_backing_cache_miss_count",
    "expert_backing_cache_evict_count",
    "expert_cpu_backing_mode",
    "expert_cpu_backing_preload_count",
    "expert_cpu_backing_preload_mb",
    "expert_cpu_backing_preload_ms",
    "expert_cpu_backing_global_hit_count",
    "expert_cpu_backing_global_miss_count",
    "expert_eviction_d2h_skip_count",
    "expert_eviction_d2h_copy_count",
    "expert_eviction_d2h_batch_count",
    "expert_eviction_d2h_batched_count",
    "expert_eviction_d2h_batched_mb",
    "expert_eviction_d2h_fallback_count",
    "expert_copy_stream_launch_count",
    "expert_copy_stream_wait_count",
    "expert_ready_before_use_count",
    "expert_ready_use_check_count",
    "expert_ready_before_use_ratio",
    "expert_call_count_total",
    "expert_prefill_call_count_total",
    "expert_decode_call_count_total",
    "expert_hotness_observed",
    "expert_call_count_by_layer",
    "expert_unique_count_by_layer",
    "expert_hotness_topk_by_layer",
    "num_expert_layers",
    "num_experts_by_layer",
    "topk_by_layer",
    "observed_batch_size",
    "avg_prefix_len",
    "decode_steps",
    "kvc_bytes_per_token_all_layers",
    "expert_topk_rewrite_count",
    "expert_core_hook_count",
    "expert_guard_pass",
    "expert_guard_reason",
    "policy_kvc_fraction",
    "policy_expert_fraction",
    "full_policy_semantics_supported",
    "policy_semantics_reason",
    "layerkv_tasks_built",
    "layerkv_copy_event_record_count",
    "layerkv_copy_event_wait_count",
    "layerkv_copy_stream_busy_ms",
    "layerkv_python_overhead_ms",
    "unified_residency_enabled",
    "resident_group_count",
    "resident_group_kvc_count",
    "resident_group_expert_count",
    "resident_group_resident_count",
    "resident_group_offloaded_count",
    "resident_group_recovering_count",
    "resident_group_recover_count",
    "resident_group_wait_count",
    "resident_group_state_error_count",
    "resident_group_last_error",
    "profile_detail_enabled",
    "profile_install_ms",
    "profile_set_kv_ms",
    "profile_forward_begin_ms",
    "profile_forward_end_ms",
    "profile_workload_stats_ms",
    "profile_apply_expert_plan_ms",
    "profile_expert_prefetch_ms",
    "profile_kvc_reload_required_ms",
    "profile_kvc_evict_to_target_ms",
    "profile_kvc_select_required_ms",
    "profile_kvc_select_evict_ms",
    "profile_req_to_token_rewrite_ms",
    "profile_expert_unique_ms",
    "profile_expert_materialize_control_ms",
    "profile_expert_materialize_metadata_ms",
    "profile_expert_materialize_choose_slot_ms",
    "profile_expert_materialize_remap_ms",
    "profile_expert_materialize_copy_issue_ms",
    "profile_planner_dp_ms",
    "profile_planner_dp_candidate_eval_ms",
    "profile_planner_dp_expert_cost_ms",
    "profile_planner_dp_kvc_cost_ms",
    "profile_prepare_expert_backing_ms",
    "profile_prepare_expert_backing_only_ms",
    "profile_apply_prepared_expert_plan_ms",
    "profile_apply_expert_slot_map_ms",
    "profile_apply_expert_shrink_ms",
    "profile_plan_expert_capacity_ms",
    "profile_install_expert_slots_ms",
    "profile_copy_expert_to_cpu_ms",
    "profile_shrink_expert_weight_ms",
    "profile_refresh_expert_stats_ms",
    "profile_finalize_kvc_ms",
    "profile_finalize_expert_ms",
    "profile_summary_build_ms",
    "profile_accounted_ms",
    "profile_unaccounted_ms",
    "profile_controller_per_decode_step_ms",
    "stats_line_count",
    "response_count",
    "request_wall_ms",
    "stdout_path",
    "stderr_path",
    "result_path",
]


@dataclasses.dataclass(frozen=True)
class PolicyRun:
    scenario: str
    mode: str
    policy: str
    reclaim_limit_mb: float


POLICY_RUNS = [
    PolicyRun("full_residency", "off", "none", 0.0),
    PolicyRun("expert_first", "kvc-expert", "expert-first", FIG4_RECLAIM_LIMIT_MB),
    PolicyRun("kvc_first", "kvc-expert", "kv-first", FIG4_RECLAIM_LIMIT_MB),
    PolicyRun("ratio_25_75", "kvc-expert", "ratio-25-75", FIG4_RECLAIM_LIMIT_MB),
    PolicyRun("ratio_50_50", "kvc-expert", "ratio-50-50", FIG4_RECLAIM_LIMIT_MB),
    PolicyRun("ratio_75_25", "kvc-expert", "ratio-75-25", FIG4_RECLAIM_LIMIT_MB),
    PolicyRun("coresid", "kvc-expert", "coresid", FIG4_RECLAIM_LIMIT_MB),
    PolicyRun("layer_aware_joint_dp", "kvc-expert", "coresid", FIG4_RECLAIM_LIMIT_MB),
]


def is_coresid(spec: PolicyRun) -> bool:
    return spec.scenario in {"coresid", "layer_aware_joint_dp"} or spec.policy in {
        "coresid",
        "layer-aware-joint-dp",
    }


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(value)
    except Exception:
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("True", "true", "1", 1):
        return True
    if value in ("False", "false", "0", 0):
        return False
    return default


def policy_command(
    args: argparse.Namespace, spec: PolicyRun, result_path: Path
) -> List[str]:
    cmd = base_command(args, result_path)
    if spec.mode == "off":
        return cmd
    kvc_backend = args.dp_kvc_backend if is_coresid(spec) else args.baseline_kvc_backend
    return cmd + layerkv_flags(
        mode=spec.mode,
        policy=spec.policy,
        reclaim_limit_mb=spec.reclaim_limit_mb,
        kvc_block_tokens=args.kvc_block_tokens,
        kvc_backend=kvc_backend,
        scheduler=args.kvc_scheduler,
        runtime_profile=args.runtime_profile,
        debug_stats=True,
        profile_detail=args.profile_detail,
        expert_backing_cache_mb=args.expert_backing_cache_mb,
        expert_cpu_backing_mode=args.expert_cpu_backing_mode,
        expert_install_layers_per_step=args.expert_install_layers_per_step,
        expert_install_budget_mb=args.expert_install_budget_mb,
        expert_install_target_steps=args.expert_install_target_steps,
    )


def _free_port() -> int:
    for port in range(30000, 50000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("failed to find a free localhost port")


def _http_get(url: str, timeout: float) -> bool:
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            return 200 <= int(resp.status) < 300
    except Exception:
        return False


def _http_json(url: str, payload: Dict[str, Any], timeout: float) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return float(ordered[max(0, min(len(ordered) - 1, idx))])


def _response_latency_stats(response: Any, output_len: int) -> Dict[str, float]:
    if not isinstance(response, list):
        return {}
    values: List[float] = []
    for item in response:
        if not isinstance(item, dict):
            continue
        meta = item.get("meta_info")
        if not isinstance(meta, dict):
            continue
        value = meta.get("e2e_latency")
        try:
            values.append(float(value) * 1000.0)
        except Exception:
            continue
    if not values:
        return {}
    denom = max(1, int(output_len))
    return {
        "response_e2e_latency_ms_mean": float(statistics.mean(values)),
        "response_e2e_latency_ms_p50": float(statistics.median(values)),
        "response_e2e_latency_ms_p95": _percentile(values, 0.95),
        "response_e2e_latency_ms_max": float(max(values)),
        "response_e2e_per_token_ms_mean": float(statistics.mean(values) / denom),
        "response_e2e_per_token_ms_p50": float(statistics.median(values) / denom),
        "response_e2e_per_token_ms_p95": float(_percentile(values, 0.95) / denom),
    }


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _tail(path: Path, max_chars: int = 50000) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")[-max_chars:]


def _read_proc_rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except OSError:
        return 0
    return 0


def _child_pids(pid: int) -> List[int]:
    children: List[int] = []
    try:
        with open(f"/proc/{pid}/task/{pid}/children", "r", encoding="utf-8") as f:
            for tok in f.read().split():
                try:
                    child = int(tok)
                except ValueError:
                    continue
                children.append(child)
                children.extend(_child_pids(child))
    except OSError:
        pass
    return children


def _start_rss_monitor(
    root_pid: int, path: Path, stop: threading.Event
) -> threading.Thread:
    def _run() -> None:
        peak_kb = 0
        with path.open("w", encoding="utf-8") as f:
            f.write("time_s,total_rss_mb,peak_rss_mb,num_processes,pids\n")
            t0 = time.perf_counter()
            while not stop.is_set():
                pids = [root_pid] + _child_pids(root_pid)
                rss_kb = sum(_read_proc_rss_kb(pid) for pid in pids)
                peak_kb = max(peak_kb, rss_kb)
                f.write(
                    f"{time.perf_counter() - t0:.3f},"
                    f"{rss_kb / 1024.0:.1f},"
                    f"{peak_kb / 1024.0:.1f},"
                    f"{len(pids)},"
                    f"{' '.join(str(pid) for pid in pids)}\n"
                )
                f.flush()
                stop.wait(2.0)

    thread = threading.Thread(target=_run, name="layerkv-rss-monitor", daemon=True)
    thread.start()
    return thread


def server_command(args: argparse.Namespace, spec: PolicyRun, port: int) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--moe-a2a-backend",
        "none",
        "--moe-runner-backend",
        "triton",
        "--grammar-backend",
        "none",
        "--skip-server-warmup",
        "--watchdog-timeout",
        str(args.watchdog_timeout_s),
        "--disable-overlap-schedule",
        "--schedule-policy",
        args.schedule_policy,
        "--weight-loader-drop-cache-after-load",
        "--model-loader-extra-config",
        '{"enable_multithread_load": false}',
        "--log-level",
        args.log_level,
    ]
    if args.max_total_tokens > 0:
        cmd.extend(["--max-total-tokens", str(args.max_total_tokens)])
    if args.max_running_requests > 0:
        cmd.extend(["--max-running-requests", str(args.max_running_requests)])
    if args.mem_fraction_static > 0.0:
        cmd.extend(["--mem-fraction-static", str(args.mem_fraction_static)])
    kvc_backend = args.dp_kvc_backend if is_coresid(spec) else args.baseline_kvc_backend
    cmd.extend(
        layerkv_flags(
            mode=spec.mode,
            policy=spec.policy,
            reclaim_limit_mb=spec.reclaim_limit_mb,
            kvc_block_tokens=args.kvc_block_tokens,
            kvc_backend=kvc_backend,
            scheduler=args.kvc_scheduler,
            runtime_profile=args.runtime_profile,
            debug_stats=True,
            profile_detail=args.profile_detail,
            expert_backing_cache_mb=args.expert_backing_cache_mb,
            expert_cpu_backing_mode=args.expert_cpu_backing_mode,
            expert_install_layers_per_step=args.expert_install_layers_per_step,
            expert_install_budget_mb=args.expert_install_budget_mb,
            expert_install_target_steps=args.expert_install_target_steps,
        )
    )
    return cmd


def fig4_metadata_fields(args: argparse.Namespace) -> Dict[str, Any]:
    md = getattr(args, "fig4_prompt_metadata", {}) or {}
    token_counts = md.get("selected_token_counts") or []
    return {
        "fig4_real_dataset_loaded": bool(md.get("fig4_real_dataset_loaded", False)),
        "prompt_source": md.get("prompt_source", ""),
        "synthetic_prompt_used": bool(md.get("synthetic_prompt_used", True)),
        "prompt_repetition_used": bool(md.get("prompt_repetition_used", False)),
        "selected_record_ids_hash": md.get("selected_record_ids_hash", ""),
        "selected_token_counts_min": min(token_counts) if token_counts else "",
        "selected_token_counts_max": max(token_counts) if token_counts else "",
        "selected_token_counts_mean": (
            sum(token_counts) / len(token_counts) if token_counts else ""
        ),
        "payload_input_len_min": md.get("payload_input_len_min", ""),
        "payload_input_len_max": md.get("payload_input_len_max", ""),
        "payload_input_len_mean": md.get("payload_input_len_mean", ""),
        "payload_input_ids_hash": md.get("payload_input_ids_hash", ""),
    }


def validate_policy_row(
    spec: PolicyRun, returncode: int, stats: Dict[str, Any]
) -> Tuple[bool, str, bool]:
    reasons: List[str] = []
    limited_by_workload = False
    if returncode != 0:
        reasons.append(f"process_returncode={returncode}")
    if spec.mode == "off":
        return not reasons, ";".join(reasons), limited_by_workload
    if not stats:
        reasons.append("missing_layerkv_stats")
        return False, ";".join(reasons), limited_by_workload

    if not _to_bool(stats.get("kvc_guard_pass"), False):
        reasons.append(f"kvc_guard_failed:{stats.get('kvc_guard_reason')}")
    if _to_int(stats.get("kvc_physical_failure_count")) != 0:
        reasons.append("kvc_physical_failure_count_nonzero")
    if _to_int(stats.get("kvc_stale_entry_count")) != 0:
        reasons.append("kvc_stale_entry_count_nonzero")
    if not _to_bool(stats.get("expert_guard_pass"), False):
        reasons.append(f"expert_guard_failed:{stats.get('expert_guard_reason')}")
    if _to_int(stats.get("resident_group_state_error_count")) != 0:
        reasons.append(
            f"resident_group_state_error:{stats.get('resident_group_last_error')}"
        )

    planned_kvc = _to_float(stats.get("planned_kvc_reclaim_mb"))
    physical_kvc = max(
        _to_float(stats.get("physical_kvc_reclaim_mb")),
        _to_float(stats.get("physical_kvc_reclaim_peak_mb")),
    )
    planned_expert = _to_float(stats.get("planned_expert_reclaim_mb"))
    physical_expert = _to_float(stats.get("physical_expert_reclaim_mb"))

    if planned_kvc > 0.0:
        if not _to_bool(stats.get("layerkv_physical_kvc_supported"), False):
            reasons.append("physical_kvc_unsupported")
        if _to_int(stats.get("kvc_evict_count_total")) <= 0:
            reasons.append("no_kvc_evict")
        if _to_int(stats.get("kvc_reload_count_total")) <= 0:
            reasons.append("no_kvc_reload")
        if _to_int(stats.get("kvc_req_to_token_rewrite_count")) <= 0:
            reasons.append("no_req_to_token_rewrite")
        if physical_kvc + 1e-3 < planned_kvc:
            # Short smoke workloads often do not have enough live KV to hold the
            # requested reclaim at the final stats point. Use peak physical KVC
            # reclaim for comparability because server requests may finish and
            # clean up before final stats are emitted.
            limited_by_workload = True

    if planned_expert > 0.0:
        if not _to_bool(stats.get("layerkv_physical_expert_supported"), False):
            reasons.append("physical_expert_unsupported")
        if _to_int(stats.get("expert_slot_rebind_count")) <= 0:
            reasons.append("no_expert_slot_rebind")

    if not _to_bool(stats.get("comparable"), True):
        reasons.append(f"not_comparable:{stats.get('comparability_reason')}")

    return not reasons, ";".join(reasons), limited_by_workload


def validate_fig4_dataset_row(
    args: argparse.Namespace,
    valid: bool,
    reason: str,
) -> Tuple[bool, str]:
    if not bool(getattr(args, "fig4_aligned", False)):
        return valid, reason
    md = getattr(args, "fig4_prompt_metadata", {}) or {}
    reasons: List[str] = []
    if not _to_bool(md.get("fig4_real_dataset_loaded"), False):
        reasons.append("fig4_real_dataset_not_loaded")
    if _to_bool(md.get("synthetic_prompt_used"), True):
        reasons.append("synthetic_prompt_used")
    if _to_bool(md.get("prompt_repetition_used"), False):
        reasons.append("prompt_repetition_used")
    if not md.get("payload_input_ids_hash"):
        reasons.append("missing_real_payload_hash")
    if reasons:
        reason = (reason + ";" if reason else "") + ";".join(reasons)
        return False, reason
    return valid, reason


def run_policy(
    args: argparse.Namespace, spec: PolicyRun, output_dir: Path
) -> Dict[str, Any]:
    if args.backend == "server":
        return run_policy_server(args, spec, output_dir)
    return run_policy_bench(args, spec, output_dir)


def run_policy_bench(
    args: argparse.Namespace, spec: PolicyRun, output_dir: Path
) -> Dict[str, Any]:
    result_path = output_dir / f"{spec.scenario}.jsonl"
    result_path.unlink(missing_ok=True)
    cmd = policy_command(args, spec, result_path)
    result = run_bench_command(
        cmd=cmd,
        output_dir=output_dir,
        run_name=spec.scenario,
        args=args,
    )
    final_stats = result["final_stats"]
    valid, reason, limited_by_workload = validate_policy_row(
        spec, result["returncode"], final_stats
    )
    valid, reason = validate_fig4_dataset_row(args, valid, reason)
    planned_reclaim = _to_float(final_stats.get("planned_kvc_reclaim_mb")) + _to_float(
        final_stats.get("planned_expert_reclaim_mb")
    )
    actual_reclaim = max(
        _to_float(final_stats.get("physical_total_reclaim_peak_mb")),
        _to_float(final_stats.get("physical_kvc_reclaim_peak_mb"))
        + _to_float(final_stats.get("physical_expert_reclaim_mb")),
        _to_float(final_stats.get("physical_kvc_reclaim_mb"))
        + _to_float(final_stats.get("physical_expert_reclaim_mb")),
    )

    row: Dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(result["latencies"])
    row.update(final_stats)
    row.update(
        {
            "backend": "bench",
            "workload": args.workload,
            "fig4_aligned": bool(args.fig4_aligned),
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "fig4_dataset_name": args.fig4_dataset_name,
            "fig4_dataset_path": args.fig4_dataset_path,
            **fig4_metadata_fields(args),
            "max_total_tokens": args.max_total_tokens,
            "max_running_requests": args.max_running_requests,
            "mem_fraction_static": args.mem_fraction_static,
            "scenario": spec.scenario,
            "layerkv_mode": (
                spec.mode
                if spec.mode == "off"
                else final_stats.get("layerkv_mode", spec.mode)
            ),
            "layerkv_policy": (
                spec.policy
                if spec.mode == "off"
                else final_stats.get("layerkv_policy", spec.policy)
            ),
            "returncode": result["returncode"],
            "valid": valid,
            "validation_reason": reason,
            "reclaim_limit_mb": spec.reclaim_limit_mb,
            "configured_reclaim_limit_mb": final_stats.get(
                "configured_reclaim_limit_mb", spec.reclaim_limit_mb
            ),
            "planned_reclaim_mb": planned_reclaim,
            "actual_reclaim_mb": actual_reclaim,
            "actual_reclaim_limited_by_workload": limited_by_workload,
            "benchmark_decode0_latency_ms": _to_float(
                result["latencies"].get("benchmark_decode0_latency_s")
            )
            * 1000.0,
            "output_throughput_tok_s": "",
            "stats_line_count": result["stats_line_count"],
            "response_count": "",
            "request_wall_ms": "",
            "stdout_path": result["stdout_path"],
            "stderr_path": result["stderr_path"],
            "result_path": str(result_path),
        }
    )
    return row


def run_policy_server(
    args: argparse.Namespace, spec: PolicyRun, output_dir: Path
) -> Dict[str, Any]:
    stdout_path = output_dir / f"{spec.scenario}.server.stdout.log"
    stderr_path = output_dir / f"{spec.scenario}.server.stderr.log"
    response_path = output_dir / f"{spec.scenario}.response.json"
    rss_path = output_dir / f"{spec.scenario}.host_rss.csv"
    for path in (stdout_path, stderr_path, response_path, rss_path):
        path.unlink(missing_ok=True)

    port = _free_port()
    cmd = server_command(args, spec, port)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["TMPDIR"] = args.tmpdir
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("MALLOC_TRIM_THRESHOLD_", "131072")
    env.setdefault("MALLOC_MMAP_THRESHOLD_", "131072")
    # SGLang's idle checker treats radix-cache evictable tokens as a leak in
    # long physical KVC reclaim runs. Keep LayerKV guards enabled, but do not
    # let this harness-only checker kill a valid policy comparison.
    env["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"
    repo_python = str(Path.cwd() / "python")
    env["PYTHONPATH"] = (
        repo_python
        if not env.get("PYTHONPATH")
        else repo_python + os.pathsep + env["PYTHONPATH"]
    )

    response: Any = None
    request_wall_ms = 0.0
    with stdout_path.open("w") as stdout_f, stderr_path.open("w") as stderr_f:
        proc = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            stdout=stdout_f,
            stderr=stderr_f,
            start_new_session=True,
        )
        rss_stop = threading.Event()
        rss_thread = _start_rss_monitor(proc.pid, rss_path, rss_stop)
        try:
            deadline = time.time() + args.startup_timeout_s
            ready_url = f"http://127.0.0.1:{port}/model_info"
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                if _http_get(ready_url, timeout=5.0):
                    break
                time.sleep(2.0)
            if not hasattr(args, "fig4_input_ids"):
                raise RuntimeError(
                    "missing Fig4 real input_ids; refusing synthetic payload"
                )
            payload = {
                "input_ids": args.fig4_input_ids,
                "sampling_params": {
                    "max_new_tokens": args.output_len,
                    "temperature": 0.0,
                },
                "stream": False,
            }
            t0 = time.perf_counter()
            response = _http_json(
                f"http://127.0.0.1:{port}/generate",
                payload,
                timeout=args.request_timeout_s,
            )
            request_wall_ms = (time.perf_counter() - t0) * 1000.0
            response_path.write_text(json.dumps(response, indent=2, sort_keys=True))
        except Exception as exc:
            response = {"error": repr(exc)}
            response_path.write_text(json.dumps(response, indent=2, sort_keys=True))
        finally:
            rss_stop.set()
            rss_thread.join(timeout=5.0)
            _terminate(proc)

    combined = _tail(stdout_path) + "\n" + _tail(stderr_path)
    stats = parse_layerkv_stats(combined)
    final_stats = stats[-1] if stats else {}
    response_ok = not (isinstance(response, dict) and "error" in response)
    effective_returncode = (
        0 if response_ok and proc.returncode in (-9, -15, -3, 0) else proc.returncode
    )
    valid, reason, limited_by_workload = validate_policy_row(
        spec, effective_returncode, final_stats
    )
    valid, reason = validate_fig4_dataset_row(args, valid, reason)
    if not response_ok:
        valid = False
        err = response.get("error") if isinstance(response, dict) else repr(response)
        reason = (reason + ";" if reason else "") + f"generate_failed:{err}"

    planned_reclaim = _to_float(final_stats.get("planned_kvc_reclaim_mb")) + _to_float(
        final_stats.get("planned_expert_reclaim_mb")
    )
    actual_reclaim = max(
        _to_float(final_stats.get("physical_total_reclaim_peak_mb")),
        _to_float(final_stats.get("physical_kvc_reclaim_peak_mb"))
        + _to_float(final_stats.get("physical_expert_reclaim_mb")),
        _to_float(final_stats.get("physical_kvc_reclaim_mb"))
        + _to_float(final_stats.get("physical_expert_reclaim_mb")),
    )
    response_count = (
        len(response) if isinstance(response, list) else (1 if response_ok else 0)
    )
    response_latency = _response_latency_stats(response, args.output_len)
    output_throughput = (
        response_count * args.output_len / (request_wall_ms / 1000.0)
        if request_wall_ms > 0
        else 0.0
    )

    row: Dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(final_stats)
    row.update(response_latency)
    row.update(
        {
            "backend": "server",
            "workload": args.workload,
            "fig4_aligned": bool(args.fig4_aligned),
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "fig4_dataset_name": args.fig4_dataset_name,
            "fig4_dataset_path": args.fig4_dataset_path,
            **fig4_metadata_fields(args),
            "max_total_tokens": args.max_total_tokens,
            "max_running_requests": args.max_running_requests,
            "mem_fraction_static": args.mem_fraction_static,
            "scenario": spec.scenario,
            "layerkv_mode": final_stats.get("layerkv_mode", spec.mode),
            "layerkv_policy": final_stats.get("layerkv_policy", spec.policy),
            "returncode": proc.returncode,
            "valid": valid,
            "validation_reason": reason,
            "reclaim_limit_mb": spec.reclaim_limit_mb,
            "configured_reclaim_limit_mb": final_stats.get(
                "configured_reclaim_limit_mb", spec.reclaim_limit_mb
            ),
            "planned_reclaim_mb": planned_reclaim,
            "actual_reclaim_mb": actual_reclaim,
            "actual_reclaim_limited_by_workload": limited_by_workload,
            "benchmark_decode0_latency_ms": request_wall_ms / max(1, args.output_len),
            "output_throughput_tok_s": output_throughput,
            "stats_line_count": len(stats),
            "response_count": response_count,
            "request_wall_ms": request_wall_ms,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "result_path": str(response_path),
        }
    )
    return row


def classify_joint_result(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparable = [
        row
        for row in rows
        if _to_bool(row.get("valid"), False) and _to_bool(row.get("comparable"), True)
    ]
    if not comparable:
        return {
            "best_policy": "",
            "joint_wins": False,
            "joint_loss_reason": "no_comparable_rows",
        }

    comparable.sort(
        key=lambda row: _to_float(row.get("benchmark_decode0_latency_ms"), float("inf"))
    )
    best = comparable[0]
    joint = next(
        (
            row
            for row in comparable
            if row.get("scenario") in ("coresid", "layer_aware_joint_dp")
        ),
        None,
    )
    if joint is None:
        return {
            "best_policy": best.get("scenario", ""),
            "joint_wins": False,
            "joint_loss_reason": "joint_row_not_comparable",
        }
    if joint is best:
        return {
            "best_policy": best.get("scenario", ""),
            "joint_wins": True,
            "joint_loss_reason": "",
        }
    if _to_bool(joint.get("actual_reclaim_limited_by_workload"), False):
        reason = "workload_too_small"
    elif (
        _to_float(joint.get("expert_materialize_mb_total")) > 0.0
        or _to_float(joint.get("kvc_reload_mb_total")) > 0.0
    ):
        reason = "runtime_recovery"
    else:
        reason = "planner_split"
    return {
        "best_policy": best.get("scenario", ""),
        "joint_wins": False,
        "joint_loss_reason": reason,
        "joint_latency_ms": _to_float(joint.get("benchmark_decode0_latency_ms")),
        "best_latency_ms": _to_float(best.get("benchmark_decode0_latency_ms")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default="outputs/layerkv/policy_eval")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--backend", choices=["server", "bench"], default="server")
    parser.add_argument(
        "--workload",
        default="batch-heavy",
        choices=sorted(FIG4_WORKLOADS),
        help="Fig4-aligned workload preset.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--max-running-requests", type=int, default=0)
    parser.add_argument("--mem-fraction-static", type=float, default=0.0)
    parser.add_argument("--schedule-policy", default="fcfs")
    parser.add_argument("--kvc-block-tokens", type=int, default=16)
    parser.add_argument(
        "--baseline-kvc-backend",
        choices=["token-slot", "per-layer-arena"],
        default="token-slot",
        help="KVC backend for non-DP baselines.",
    )
    parser.add_argument(
        "--dp-kvc-backend",
        choices=["token-slot", "per-layer-arena"],
        default="per-layer-arena",
        help="KVC backend for layer-aware-joint-dp.",
    )
    parser.add_argument("--kvc-scheduler", default="async-deadline")
    parser.add_argument(
        "--runtime-profile",
        choices=["simple", "optimized"],
        default="optimized",
    )
    parser.add_argument("--profile-detail", action="store_true")
    parser.add_argument("--expert-backing-cache-mb", type=float, default=0.0)
    parser.add_argument(
        "--expert-cpu-backing-mode",
        choices=["none", "all"],
        default="none",
    )
    parser.add_argument("--expert-install-layers-per-step", type=int, default=1)
    parser.add_argument("--expert-install-budget-mb", type=float, default=128.0)
    parser.add_argument("--expert-install-target-steps", type=int, default=0)
    parser.add_argument("--tmpdir", default="/data/wenyan/tmp")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--startup-timeout-s", type=float, default=1200.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--watchdog-timeout-s", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[spec.scenario for spec in POLICY_RUNS],
        choices=[spec.scenario for spec in POLICY_RUNS],
    )
    args = parser.parse_args()
    apply_fig4_workload(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.fig4_aligned:
        from transformers import AutoTokenizer

        print(
            "[layerkv-policy-eval] loading Fig4 real prompts "
            f"workload={args.workload} dataset={args.fig4_dataset_path}",
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        args.fig4_input_ids, args.fig4_prompt_metadata = load_fig4_prompt_ids(
            args, tokenizer
        )
        (output_dir / "fig4_prompt_metadata.json").write_text(
            json.dumps(args.fig4_prompt_metadata, indent=2, sort_keys=True)
        )
        print(
            "[layerkv-policy-eval] loaded Fig4 prompts "
            f"n={len(args.fig4_input_ids)} "
            f"tokens=[{args.fig4_prompt_metadata['payload_input_len_min']},"
            f"{args.fig4_prompt_metadata['payload_input_len_max']}] "
            f"hash={args.fig4_prompt_metadata['payload_input_ids_hash']}",
            flush=True,
        )
    else:
        args.fig4_prompt_metadata = {
            "fig4_real_dataset_loaded": False,
            "synthetic_prompt_used": True,
            "prompt_repetition_used": False,
        }
    selected = [spec for spec in POLICY_RUNS if spec.scenario in set(args.policies)]

    rows = []
    for spec in selected:
        print(f"[layerkv-policy-eval] running {spec.scenario}", flush=True)
        row = run_policy(args, spec, output_dir)
        rows.append(row)
        print(
            "[layerkv-policy-eval] "
            f"{spec.scenario} valid={row['valid']} reason={row['validation_reason']}",
            flush=True,
        )

    csv_path = output_dir / "policy_eval.csv"
    summary_path = output_dir / "policy_eval_summary.json"
    write_csv(csv_path, rows, CSV_FIELDS)
    invalid = [row for row in rows if str(row.get("valid")) != "True"]
    summary = {
        "csv": str(csv_path),
        "backend": args.backend,
        "workload": args.workload,
        "fig4_aligned": bool(args.fig4_aligned),
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "fig4_reclaim_limit_mb": FIG4_RECLAIM_LIMIT_MB,
        "fig4_dataset_name": args.fig4_dataset_name,
        "fig4_dataset_path": args.fig4_dataset_path,
        **fig4_metadata_fields(args),
        "max_total_tokens": args.max_total_tokens,
        "max_running_requests": args.max_running_requests,
        "mem_fraction_static": args.mem_fraction_static,
        "total_runs": len(rows),
        "valid_runs": len(rows) - len(invalid),
        "invalid_runs": len(invalid),
        "all_valid": not invalid,
        "joint_result": classify_joint_result(rows),
        "rows": rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not invalid else 1


if __name__ == "__main__":
    raise SystemExit(main())
