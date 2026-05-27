#!/usr/bin/env python3
"""Replay Azure LLM inference trace requests against LayerKV policies."""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import request

from layerkv_eval_common import (
    DEFAULT_MODEL_PATH,
    FIG4_TARGET_RECLAIM_MB,
    _extract_sharegpt_user_text,
    _extract_wildchat_user_text,
    layerkv_flags,
    parse_layerkv_stats,
    write_csv,
)
from layerkv_policy_eval import PolicyRun, _tail


TRACE_PATH = "/4IR-dataset/common/request_dataset/AzureLLMInferenceTrace_conv.csv"
SHAREGPT_PATH = "/4IR-dataset/common/request_dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
WILDCHAT_PATH = "/data/wenyan/.cache/huggingface/allenai___wild_chat-1_m"

POLICY_RUNS = [
    PolicyRun("expert_first", "kvc-expert", "expert-first", FIG4_TARGET_RECLAIM_MB),
    PolicyRun("kvc_first", "kvc-expert", "kv-first", FIG4_TARGET_RECLAIM_MB),
    PolicyRun("ratio_25_75", "kvc-expert", "ratio-25-75", FIG4_TARGET_RECLAIM_MB),
    PolicyRun("ratio_50_50", "kvc-expert", "ratio-50-50", FIG4_TARGET_RECLAIM_MB),
    PolicyRun("ratio_75_25", "kvc-expert", "ratio-75-25", FIG4_TARGET_RECLAIM_MB),
    PolicyRun("coresid", "kvc-expert", "coresid", FIG4_TARGET_RECLAIM_MB),
    PolicyRun("layer_aware_joint_dp", "kvc-expert", "coresid", FIG4_TARGET_RECLAIM_MB),
]


def is_coresid(spec: PolicyRun) -> bool:
    return spec.scenario in {"coresid", "layer_aware_joint_dp"} or spec.policy in {
        "coresid",
        "layer-aware-joint-dp",
    }

PER_REQUEST_FIELDS = [
    "policy",
    "request_id",
    "arrival_s",
    "target_context_tokens",
    "actual_context_tokens",
    "target_output_tokens",
    "prompt_source",
    "prompt_record_id",
    "submit_offset_s",
    "ttft_ms",
    "e2e_ms",
    "tpot_ms",
    "success",
    "error",
]

SUMMARY_FIELDS = [
    "policy",
    "valid",
    "validation_reason",
    "runtime_profile",
    "kvc_backend",
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
    "selected_kvc_tokens_by_layer",
    "selected_expert_evictions_by_layer",
    "selected_expert_capacity_by_layer",
    "selected_expert_cost_by_layer",
    "applied_expert_evictions_by_layer",
    "expert_plan_match",
    "expert_plan_mismatch_reason",
    "request_count",
    "success_count",
    "failed_count",
    "trace_window_s",
    "wall_ms",
    "request_throughput_rps",
    "output_throughput_tok_s",
    "ttft_ms_mean",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "ttft_ms_p99",
    "tpot_ms_mean",
    "tpot_ms_p50",
    "tpot_ms_p95",
    "tpot_ms_p99",
    "e2e_ms_mean",
    "e2e_ms_p50",
    "e2e_ms_p95",
    "e2e_ms_p99",
    "actual_reclaim_mb",
    "planned_reclaim_mb",
    "planned_kvc_reclaim_mb",
    "physical_kvc_reclaim_mb",
    "physical_kvc_reclaim_peak_mb",
    "planned_expert_reclaim_mb",
    "physical_expert_reclaim_mb",
    "kvc_reload_count_total",
    "expert_materialize_count",
    "expert_prefetch_count",
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
    "layerkv_python_overhead_ms",
    "profile_planner_dp_ms",
    "profile_copy_expert_to_cpu_ms",
    "profile_expert_materialize_control_ms",
    "kvc_guard_pass",
    "expert_guard_pass",
    "resident_group_state_error_count",
    "stats_line_count",
    "stdout_path",
    "stderr_path",
]


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


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return float(ordered[max(0, min(len(ordered) - 1, idx))])


def _stats(values: List[float], prefix: str) -> Dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_p99": 0.0,
        }
    return {
        f"{prefix}_mean": float(statistics.mean(values)),
        f"{prefix}_p50": float(statistics.median(values)),
        f"{prefix}_p95": _percentile(values, 0.95),
        f"{prefix}_p99": _percentile(values, 0.99),
    }


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


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)


def load_trace(
    path: str,
    window_s: float,
    max_requests: int,
    window_start_s: float = 0.0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    first_ts: Optional[dt.datetime] = None
    window_start_s = max(0.0, float(window_start_s or 0.0))
    window_end_s = window_start_s + float(window_s)
    with open(path, newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                ts = dt.datetime.strptime(row["TIMESTAMP"][:26], "%Y-%m-%d %H:%M:%S.%f")
                ctx = max(1, int(row["ContextTokens"]))
                out = max(1, int(row["GeneratedTokens"]))
            except Exception:
                continue
            if first_ts is None:
                first_ts = ts
            arrival_from_start_s = (ts - first_ts).total_seconds()
            if arrival_from_start_s < window_start_s:
                continue
            if arrival_from_start_s >= window_end_s:
                break
            rows.append(
                {
                    "request_id": len(rows),
                    "arrival_s": float(arrival_from_start_s - window_start_s),
                    "target_context_tokens": int(ctx),
                    "target_output_tokens": int(out),
                }
            )
            if max_requests > 0 and len(rows) >= max_requests:
                break
    if not rows:
        raise RuntimeError(f"empty trace: {path}")
    return rows


def _assign_prompt(
    outstanding: List[Tuple[int, int]],
    request_rows: List[Dict[str, Any]],
    token_ids: List[int],
    source: str,
    record_id: str,
) -> None:
    if not outstanding:
        return
    n_tokens = len(token_ids)
    pos = bisect.bisect_right(outstanding, (n_tokens, 10**12)) - 1
    if pos < 0:
        return
    target_len, req_idx = outstanding.pop(pos)
    request_rows[req_idx]["input_ids"] = token_ids[:target_len]
    request_rows[req_idx]["actual_context_tokens"] = target_len
    request_rows[req_idx]["prompt_source"] = source
    request_rows[req_idx]["prompt_record_id"] = record_id


def fill_from_sharegpt(
    path: str,
    tokenizer: Any,
    outstanding: List[Tuple[int, int]],
    request_rows: List[Dict[str, Any]],
) -> None:
    p = Path(path)
    with p.open("r") as f:
        data = json.load(f)
    for idx, rec in enumerate(data):
        if not outstanding:
            break
        text = _extract_sharegpt_user_text(rec)
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        rec_id = str(rec.get("id") or f"sharegpt-{idx}")
        _assign_prompt(outstanding, request_rows, ids, "sharegpt", rec_id)


def fill_from_wildchat(
    path: str,
    tokenizer: Any,
    outstanding: List[Tuple[int, int]],
    request_rows: List[Dict[str, Any]],
    batch_size: int = 128,
) -> None:
    import pyarrow.ipc as pa_ipc
    import pyarrow.parquet as pq

    root = Path(path)
    candidates = [root] if root.is_file() else sorted(x for x in root.rglob("*") if x.is_file())
    files = []
    for item in candidates:
        try:
            with item.open("rb") as f:
                magic = f.read(6)
        except OSError:
            continue
        if magic[:4] == b"PAR1":
            files.append(("parquet", item))
        elif item.suffix.lower() == ".arrow" or magic[:4] == b"\xff\xff\xff\xff":
            files.append(("arrow", item))
    for kind, item in files:
        if not outstanding:
            break
        if kind == "parquet":
            pf = pq.ParquetFile(item)
            batches = pf.iter_batches(batch_size=batch_size, columns=["conversation_hash", "conversation"])
            for batch in batches:
                if not outstanding:
                    break
                for rec in batch.to_pylist():
                    if not outstanding:
                        break
                    text = _extract_wildchat_user_text(rec)
                    if not text:
                        continue
                    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                    rec_id = str(rec.get("conversation_hash") or hashlib.sha1(text.encode()).hexdigest()[:12])
                    _assign_prompt(outstanding, request_rows, ids, "wildchat", rec_id)
        else:
            with item.open("rb") as f:
                reader = pa_ipc.open_stream(f)
                for batch in reader:
                    if not outstanding:
                        break
                    schema = reader.schema
                    hash_idx = schema.get_field_index("conversation_hash")
                    conv_idx = schema.get_field_index("conversation")
                    if hash_idx < 0 or conv_idx < 0:
                        continue
                    hashes = batch.column(hash_idx).to_pylist()
                    convs = batch.column(conv_idx).to_pylist()
                    for conv_hash, conv in zip(hashes, convs):
                        if not outstanding:
                            break
                        rec = {"conversation_hash": conv_hash, "conversation": conv}
                        text = _extract_wildchat_user_text(rec)
                        if not text:
                            continue
                        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                        _assign_prompt(outstanding, request_rows, ids, "wildchat", str(conv_hash))


def build_payload(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from transformers import AutoTokenizer

    rows = load_trace(
        args.trace_path,
        args.trace_window_s,
        args.max_requests,
        args.trace_window_start_s,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    outstanding = sorted(
        (int(row["target_context_tokens"]), idx) for idx, row in enumerate(rows)
    )
    fill_from_sharegpt(args.sharegpt_path, tokenizer, outstanding, rows)
    if outstanding:
        fill_from_wildchat(args.wildchat_path, tokenizer, outstanding, rows)
    if outstanding:
        raise RuntimeError(f"failed to find contexts for {len(outstanding)} trace requests")
    payload_hash = hashlib.sha256(
        json.dumps(
            [
                [row["arrival_s"], row["target_context_tokens"], row["target_output_tokens"], row["input_ids"]]
                for row in rows
            ]
        ).encode()
    ).hexdigest()[:16]
    metadata = {
        "trace_path": args.trace_path,
        "trace_window_start_s": args.trace_window_start_s,
        "trace_window_s": args.trace_window_s,
        "request_count": len(rows),
        "payload_hash": payload_hash,
        "context_min": min(int(r["actual_context_tokens"]) for r in rows),
        "context_max": max(int(r["actual_context_tokens"]) for r in rows),
        "context_mean": sum(int(r["actual_context_tokens"]) for r in rows) / max(1, len(rows)),
        "output_min": min(int(r["target_output_tokens"]) for r in rows),
        "output_max": max(int(r["target_output_tokens"]) for r in rows),
        "output_mean": sum(int(r["target_output_tokens"]) for r in rows) / max(1, len(rows)),
        "sharegpt_count": sum(1 for r in rows if r.get("prompt_source") == "sharegpt"),
        "wildchat_count": sum(1 for r in rows if r.get("prompt_source") == "wildchat"),
    }
    return rows, metadata


def _payload_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model_path": str(args.model_path),
        "trace_path": str(args.trace_path),
        "trace_window_start_s": float(args.trace_window_start_s),
        "trace_window_s": float(args.trace_window_s),
        "max_requests": int(args.max_requests),
        "sharegpt_path": str(args.sharegpt_path),
        "wildchat_path": str(args.wildchat_path),
    }


def payload_cache_paths(args: argparse.Namespace, output_dir: Path) -> Tuple[Path, Path]:
    if args.payload_path:
        payload_path = Path(args.payload_path)
    else:
        payload_path = output_dir / "trace_payload_full.jsonl"
    meta_path = Path(str(payload_path) + ".meta.json")
    return payload_path, meta_path


def _payload_hash(rows: List[Dict[str, Any]]) -> str:
    payload = [
        [
            row["arrival_s"],
            row["target_context_tokens"],
            row["target_output_tokens"],
            row["input_ids"],
        ]
        for row in rows
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()[:16]


def load_payload_cache(
    args: argparse.Namespace,
    output_dir: Path,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    payload_path, meta_path = payload_cache_paths(args, output_dir)
    if not args.reuse_payload or not payload_path.exists() or not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text())
    except Exception:
        return None
    if metadata.get("payload_config") != _payload_config(args):
        if not args.force_reuse_payload:
            return None
    rows: List[Dict[str, Any]] = []
    with payload_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["request_id"] = int(row["request_id"])
            row["arrival_s"] = float(row["arrival_s"])
            row["target_context_tokens"] = int(row["target_context_tokens"])
            row["actual_context_tokens"] = int(row["actual_context_tokens"])
            row["target_output_tokens"] = int(row["target_output_tokens"])
            row["input_ids"] = [int(x) for x in row["input_ids"]]
            rows.append(row)
    if not rows:
        return None
    metadata["payload_reused"] = True
    metadata["payload_path"] = str(payload_path)
    return rows, metadata


def save_payload_cache(
    args: argparse.Namespace,
    output_dir: Path,
    rows: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    payload_path, meta_path = payload_cache_paths(args, output_dir)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(payload_path) + ".tmp")
    with tmp_path.open("w") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "request_id": int(row["request_id"]),
                        "arrival_s": float(row["arrival_s"]),
                        "target_context_tokens": int(row["target_context_tokens"]),
                        "actual_context_tokens": int(row["actual_context_tokens"]),
                        "target_output_tokens": int(row["target_output_tokens"]),
                        "prompt_source": row.get("prompt_source", ""),
                        "prompt_record_id": row.get("prompt_record_id", ""),
                        "input_ids": row["input_ids"],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    tmp_path.replace(payload_path)
    cache_meta = {
        **metadata,
        "payload_config": _payload_config(args),
        "payload_path": str(payload_path),
        "payload_reused": False,
    }
    meta_path.write_text(json.dumps(cache_meta, indent=2, sort_keys=True))


def get_or_build_payload(
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cached = load_payload_cache(args, output_dir)
    if cached is not None:
        return cached
    rows, metadata = build_payload(args)
    metadata["payload_hash"] = _payload_hash(rows)
    metadata["payload_reused"] = False
    save_payload_cache(args, output_dir, rows, metadata)
    return rows, metadata


def server_command(args: argparse.Namespace, spec: PolicyRun, port: int, runtime_profile: str) -> List[str]:
    kvc_backend = (
        args.dp_kvc_backend
        if is_coresid(spec)
        else args.baseline_kvc_backend
    )
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
        "--log-level",
        args.log_level,
    ]
    if args.max_total_tokens > 0:
        cmd.extend(["--max-total-tokens", str(args.max_total_tokens)])
    if args.max_running_requests > 0:
        cmd.extend(["--max-running-requests", str(args.max_running_requests)])
    if args.mem_fraction_static > 0.0:
        cmd.extend(["--mem-fraction-static", str(args.mem_fraction_static)])
    cmd.extend(
        layerkv_flags(
            mode=spec.mode,
            policy=spec.policy,
            target_reclaim_mb=spec.target_reclaim_mb,
            kvc_block_tokens=args.kvc_block_tokens,
            kvc_backend=kvc_backend,
            dynamic_pressure_from_kvc=args.dynamic_pressure_from_kvc,
            scheduler=args.kvc_scheduler,
            runtime_profile=runtime_profile,
            debug_stats=True,
            profile_detail=True,
            expert_backing_cache_mb=args.expert_backing_cache_mb if runtime_profile == "optimized" else 0.0,
            expert_cpu_backing_mode=args.expert_cpu_backing_mode if runtime_profile == "optimized" else "none",
            expert_install_layers_per_step=args.expert_install_layers_per_step,
            expert_install_budget_mb=args.expert_install_budget_mb,
            expert_install_target_steps=args.expert_install_target_steps,
        )
    )
    return cmd


def stream_generate(base_url: str, row: Dict[str, Any], request_timeout_s: float) -> Dict[str, Any]:
    payload = {
        "input_ids": row["input_ids"],
        "sampling_params": {
            "max_new_tokens": int(row["target_output_tokens"]),
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        base_url + "/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    submit_t = time.perf_counter()
    first_t: Optional[float] = None
    last_obj: Dict[str, Any] = {}
    error = ""
    try:
        with request.urlopen(req, timeout=request_timeout_s) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload_s = line[6:]
                if payload_s == "[DONE]":
                    break
                if first_t is None:
                    first_t = time.perf_counter()
                try:
                    obj = json.loads(payload_s)
                    if isinstance(obj, dict):
                        last_obj = obj
                        if "error" in obj:
                            error = json.dumps(obj["error"])
                except Exception:
                    pass
    except Exception as exc:
        error = repr(exc)
    end_t = time.perf_counter()
    ttft_ms = ((first_t or end_t) - submit_t) * 1000.0
    e2e_ms = (end_t - submit_t) * 1000.0
    target_out = max(1, int(row["target_output_tokens"]))
    tpot_ms = max(0.0, e2e_ms - ttft_ms) / max(1, target_out - 1)
    return {
        "request_id": row["request_id"],
        "arrival_s": row["arrival_s"],
        "target_context_tokens": row["target_context_tokens"],
        "actual_context_tokens": row["actual_context_tokens"],
        "target_output_tokens": row["target_output_tokens"],
        "prompt_source": row["prompt_source"],
        "prompt_record_id": row["prompt_record_id"],
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "tpot_ms": tpot_ms,
        "success": not error,
        "error": error,
        "last_obj": last_obj,
    }


def replay_requests(args: argparse.Namespace, base_url: str, rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    results: List[Dict[str, Any]] = []
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_client_workers) as pool:
        futures = []
        for row in rows:
            due = start + float(row["arrival_s"]) / max(1e-9, args.time_scale)
            sleep_s = due - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            submit_offset = time.perf_counter() - start
            fut = pool.submit(stream_generate, base_url, row, args.request_timeout_s)
            fut.submit_offset_s = submit_offset  # type: ignore[attr-defined]
            futures.append(fut)
        for fut in concurrent.futures.as_completed(futures):
            item = fut.result()
            item["submit_offset_s"] = getattr(fut, "submit_offset_s", 0.0)
            results.append(item)
    wall_ms = (time.perf_counter() - start) * 1000.0
    results.sort(key=lambda x: int(x["request_id"]))
    return results, wall_ms


def validate_summary(spec: PolicyRun, returncode: int, stats: Dict[str, Any], failed_count: int) -> Tuple[bool, str]:
    reasons = []
    if failed_count:
        reasons.append(f"failed_requests={failed_count}")
    if returncode not in (-9, -15, -3, 0):
        reasons.append(f"process_returncode={returncode}")
    if not stats:
        reasons.append("missing_layerkv_stats")
    else:
        if not stats.get("kvc_guard_pass", True):
            reasons.append(f"kvc_guard_failed:{stats.get('kvc_guard_reason')}")
        if not stats.get("expert_guard_pass", True):
            reasons.append(f"expert_guard_failed:{stats.get('expert_guard_reason')}")
        if _to_int(stats.get("resident_group_state_error_count")) != 0:
            reasons.append("resident_group_state_error")
        if spec.policy == "coresid":
            if not stats.get("selected_expert_evictions_by_layer"):
                reasons.append("missing_selected_expert_plan")
            if not stats.get("selected_expert_capacity_by_layer"):
                reasons.append("missing_selected_expert_capacity_plan")
            if not stats.get("applied_expert_evictions_by_layer"):
                reasons.append("missing_applied_expert_plan")
            if str(stats.get("expert_plan_match", True)) != "True":
                reasons.append(
                    f"expert_plan_mismatch:{stats.get('expert_plan_mismatch_reason')}"
                )
    return not reasons, ";".join(reasons)


def run_policy(args: argparse.Namespace, spec: PolicyRun, rows: List[Dict[str, Any]], output_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    runtime_profile = args.dp_runtime_profile if is_coresid(spec) else "simple"
    port = _free_port()
    stdout_path = output_dir / f"{spec.scenario}.server.stdout.log"
    stderr_path = output_dir / f"{spec.scenario}.server.stderr.log"
    for path in (stdout_path, stderr_path):
        path.unlink(missing_ok=True)
    cmd = server_command(args, spec, port, runtime_profile)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["TMPDIR"] = args.tmpdir
    env["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"
    repo_python = str(Path.cwd() / "python")
    env["PYTHONPATH"] = repo_python if not env.get("PYTHONPATH") else repo_python + os.pathsep + env["PYTHONPATH"]
    with stdout_path.open("w") as stdout_f, stderr_path.open("w") as stderr_f:
        proc = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            stdout=stdout_f,
            stderr=stderr_f,
            start_new_session=True,
        )
        try:
            ready_url = f"http://127.0.0.1:{port}/model_info"
            deadline = time.time() + args.startup_timeout_s
            ready = False
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                if _http_get(ready_url, timeout=5.0):
                    ready = True
                    break
                time.sleep(2.0)
            if proc.poll() is not None:
                raise RuntimeError(f"server exited before ready: {proc.returncode}")
            if not ready:
                raise RuntimeError("server did not become ready before startup timeout")
            request_rows, wall_ms = replay_requests(args, f"http://127.0.0.1:{port}", rows)
        except Exception as exc:
            request_rows = [
                {
                    **{k: row.get(k, "") for k in PER_REQUEST_FIELDS if k in row},
                    "request_id": row["request_id"],
                    "arrival_s": row["arrival_s"],
                    "target_context_tokens": row["target_context_tokens"],
                    "actual_context_tokens": row.get("actual_context_tokens", ""),
                    "target_output_tokens": row["target_output_tokens"],
                    "success": False,
                    "error": repr(exc),
                    "ttft_ms": 0.0,
                    "e2e_ms": 0.0,
                    "tpot_ms": 0.0,
                }
                for row in rows
            ]
            wall_ms = 0.0
        finally:
            _terminate(proc)
    combined = _tail(stdout_path) + "\n" + _tail(stderr_path)
    stats_lines = parse_layerkv_stats(combined)
    stats = stats_lines[-1] if stats_lines else {}
    success_rows = [r for r in request_rows if r.get("success")]
    failed_count = len(request_rows) - len(success_rows)
    valid, reason = validate_summary(spec, proc.returncode, stats, failed_count)
    ttft = [_to_float(r.get("ttft_ms")) for r in success_rows]
    e2e = [_to_float(r.get("e2e_ms")) for r in success_rows]
    tpot = [_to_float(r.get("tpot_ms")) for r in success_rows]
    total_output_tokens = sum(int(r.get("target_output_tokens", 0)) for r in success_rows)
    planned_reclaim = _to_float(stats.get("planned_kvc_reclaim_mb")) + _to_float(stats.get("planned_expert_reclaim_mb"))
    actual_reclaim = max(
        _to_float(stats.get("physical_total_reclaim_peak_mb")),
        _to_float(stats.get("physical_kvc_reclaim_peak_mb")) + _to_float(stats.get("physical_expert_reclaim_mb")),
        _to_float(stats.get("physical_kvc_reclaim_mb")) + _to_float(stats.get("physical_expert_reclaim_mb")),
    )
    summary: Dict[str, Any] = {
        "policy": spec.scenario,
        "valid": valid,
        "validation_reason": reason,
        "runtime_profile": runtime_profile,
        "kvc_backend": stats.get("layerkv_kvc_backend", ""),
        "request_count": len(request_rows),
        "success_count": len(success_rows),
        "failed_count": failed_count,
        "trace_window_s": args.trace_window_s,
        "wall_ms": wall_ms,
        "request_throughput_rps": len(success_rows) / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
        "output_throughput_tok_s": total_output_tokens / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
        "actual_reclaim_mb": actual_reclaim,
        "planned_reclaim_mb": planned_reclaim,
        "stats_line_count": len(stats_lines),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    summary.update(_stats(ttft, "ttft_ms"))
    summary.update(_stats(tpot, "tpot_ms"))
    summary.update(_stats(e2e, "e2e_ms"))
    for key in SUMMARY_FIELDS:
        if key in summary:
            continue
        summary[key] = stats.get(key, "")
    for r in request_rows:
        r["policy"] = spec.scenario
        r.pop("last_obj", None)
    return summary, request_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--trace-path", default=TRACE_PATH)
    parser.add_argument("--sharegpt-path", default=SHAREGPT_PATH)
    parser.add_argument("--wildchat-path", default=WILDCHAT_PATH)
    parser.add_argument("--output-dir", default="outputs/layerkv/azure_trace_policy_eval")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--trace-window-start-s", type=float, default=0.0)
    parser.add_argument("--trace-window-s", type=float, default=300.0)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument(
        "--payload-path",
        default="",
        help="Path to a reusable JSONL payload containing input_ids. Defaults to <output-dir>/trace_payload_full.jsonl.",
    )
    parser.add_argument(
        "--reuse-payload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an existing payload cache when its metadata matches the current trace/model settings.",
    )
    parser.add_argument(
        "--force-reuse-payload",
        action="store_true",
        help="Reuse --payload-path even if metadata does not match. Intended only for deliberate A/B reruns.",
    )
    parser.add_argument(
        "--prepare-payload-only",
        action="store_true",
        help="Build or load the reusable payload and exit before launching any policy.",
    )
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-client-workers", type=int, default=128)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--schedule-policy", default="fcfs")
    parser.add_argument("--kvc-block-tokens", type=int, default=16)
    parser.add_argument(
        "--baseline-kvc-backend",
        choices=["token-slot", "per-layer-arena"],
        default="per-layer-arena",
        help=(
            "KVC backend for non-DP baselines. Use per-layer-arena by default "
            "so baseline KVC reclaim is physical and safe under hard KV-table pressure."
        ),
    )
    parser.add_argument(
        "--dp-kvc-backend",
        choices=["token-slot", "per-layer-arena"],
        default="per-layer-arena",
        help="KVC backend for layer-aware-joint-dp. per-layer-arena is required for true per-layer KVC choices.",
    )
    parser.add_argument("--kvc-scheduler", default="async-deadline")
    parser.add_argument("--dp-runtime-profile", choices=["simple", "optimized"], default="optimized")
    parser.add_argument(
        "--dynamic-pressure-from-kvc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Estimate pressure from current live KV demand, capped by the "
            "configured target reclaim. Enabled by default for trace replay."
        ),
    )
    parser.add_argument("--expert-backing-cache-mb", type=float, default=0.0)
    parser.add_argument("--expert-cpu-backing-mode", choices=["none", "all"], default="none")
    parser.add_argument("--expert-install-layers-per-step", type=int, default=1)
    parser.add_argument("--expert-install-budget-mb", type=float, default=128.0)
    parser.add_argument("--expert-install-target-steps", type=int, default=16)
    parser.add_argument("--tmpdir", default="/data/wenyan/tmp")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--startup-timeout-s", type=float, default=1200.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--watchdog-timeout-s", type=int, default=3600)
    parser.add_argument("--policies", nargs="+", default=[p.scenario for p in POLICY_RUNS], choices=[p.scenario for p in POLICY_RUNS])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, metadata = get_or_build_payload(args, output_dir)
    (output_dir / "run_config.json").write_text(json.dumps({**vars(args), **metadata}, indent=2, sort_keys=True))
    payload_rows = [
        {k: v for k, v in row.items() if k != "input_ids"}
        for row in rows
    ]
    write_csv(output_dir / "trace_payload.csv", payload_rows, [
        "request_id", "arrival_s", "target_context_tokens", "actual_context_tokens",
        "target_output_tokens", "prompt_source", "prompt_record_id",
    ])
    if args.prepare_payload_only:
        result = {
            "payload_path": metadata.get("payload_path", str(payload_cache_paths(args, output_dir)[0])),
            "payload_hash": metadata.get("payload_hash", ""),
            "payload_reused": metadata.get("payload_reused", False),
            "request_count": len(rows),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    selected = [p for p in POLICY_RUNS if p.scenario in set(args.policies)]
    all_request_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    def write_partial_outputs() -> None:
        write_csv(output_dir / "per_request.csv", all_request_rows, PER_REQUEST_FIELDS)
        write_csv(output_dir / "policy_summary.csv", summary_rows, SUMMARY_FIELDS)
        valid_rows = [r for r in summary_rows if str(r.get("valid")) == "True"]
        best = max(
            valid_rows,
            key=lambda r: _to_float(r.get("output_throughput_tok_s")),
            default={},
        )
        result = {
            "summary_csv": str(output_dir / "policy_summary.csv"),
            "per_request_csv": str(output_dir / "per_request.csv"),
            "request_count": len(rows),
            "best_policy": best.get("policy", ""),
            "best_output_throughput_tok_s": best.get("output_throughput_tok_s", 0.0),
            "all_valid": len(valid_rows) == len(summary_rows),
            "rows": summary_rows,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True)
        )

    for spec in selected:
        print(f"[azure-trace] running {spec.scenario}", flush=True)
        summary, request_rows = run_policy(args, spec, rows, output_dir)
        summary_rows.append(summary)
        all_request_rows.extend(request_rows)
        print(
            f"[azure-trace] {spec.scenario} valid={summary['valid']} "
            f"throughput={summary['output_throughput_tok_s']:.3f} "
            f"ttft_p95={summary['ttft_ms_p95']:.3f} "
            f"tpot_p95={summary['tpot_ms_p95']:.3f} reason={summary['validation_reason']}",
            flush=True,
        )
        write_partial_outputs()

    valid_rows = [r for r in summary_rows if str(r.get("valid")) == "True"]
    best = max(valid_rows, key=lambda r: _to_float(r.get("output_throughput_tok_s")), default={})
    result = {
        "summary_csv": str(output_dir / "policy_summary.csv"),
        "per_request_csv": str(output_dir / "per_request.csv"),
        "request_count": len(rows),
        "best_policy": best.get("policy", ""),
        "best_output_throughput_tok_s": best.get("output_throughput_tok_s", 0.0),
        "all_valid": len(valid_rows) == len(summary_rows),
        "rows": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    write_csv(output_dir / "per_request.csv", all_request_rows, PER_REQUEST_FIELDS)
    write_csv(output_dir / "policy_summary.csv", summary_rows, SUMMARY_FIELDS)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
