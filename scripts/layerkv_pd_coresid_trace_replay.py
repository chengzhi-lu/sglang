#!/usr/bin/env python3
"""Replay an Azure-trace payload against PD disaggregation with CoResid on decode."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

from layerkv_eval_common import (
    DEFAULT_MODEL_PATH,
    FIG4_RECLAIM_LIMIT_MB,
    layerkv_flags,
)
from layerkv_azure_trace_replay import parse_layerkv_stats

MODEL_CANDIDATES = [
    (
        "/root/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/"
        "snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    ),
    DEFAULT_MODEL_PATH,
]


def _default_model_path() -> str:
    for path in MODEL_CANDIDATES:
        if Path(path).exists():
            return path
    return DEFAULT_MODEL_PATH


def _is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _pick_port(preferred: int, used: set[int]) -> int:
    if preferred > 0 and preferred not in used and _is_free(preferred):
        used.add(preferred)
        return preferred
    for port in range(30000, 50000):
        if port in used:
            continue
        if _is_free(port):
            used.add(port)
            return port
    raise RuntimeError("failed to find a free localhost port")


def _base_env(args: argparse.Namespace, gpu: Optional[str] = None) -> Dict[str, str]:
    env = os.environ.copy()
    repo_python = str(Path.cwd() / "python")
    env["PYTHONPATH"] = (
        repo_python
        if not env.get("PYTHONPATH")
        else repo_python + os.pathsep + env["PYTHONPATH"]
    )
    if args.nixl_backend:
        env["SGLANG_DISAGGREGATION_NIXL_BACKEND"] = args.nixl_backend
    if args.nixl_backend_params:
        env["SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS"] = args.nixl_backend_params
    if args.ucx_tls:
        env["UCX_TLS"] = args.ucx_tls
    if args.ucx_log_level:
        env["UCX_LOG_LEVEL"] = args.ucx_log_level
    if args.ucx_net_devices:
        env["UCX_NET_DEVICES"] = args.ucx_net_devices
    if args.transfer_queue_size > 0:
        env["SGLANG_DISAGGREGATION_QUEUE_SIZE"] = str(args.transfer_queue_size)
    if args.thread_pool_size > 0:
        env["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = str(args.thread_pool_size)
    if args.waiting_timeout_s > 0:
        env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = str(args.waiting_timeout_s)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


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


def _tail(path: Path, max_chars: int = 300000) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")[-max_chars:]


def _http_get(url: str, timeout: float = 5.0) -> bool:
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            return 200 <= int(resp.status) < 300
    except Exception:
        return False


def _wait_http(url: str, proc: subprocess.Popen, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        if _http_get(url):
            return True
        time.sleep(2.0)
    return False


def _server_cmd(
    args: argparse.Namespace, mode: str, port: int, extra_flags: List[str]
) -> List[str]:
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
        "--disaggregation-mode",
        mode,
        "--disaggregation-transfer-backend",
        "nixl",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--watchdog-timeout",
        str(args.watchdog_timeout_s),
    ]
    if args.max_total_tokens > 0:
        cmd.extend(["--max-total-tokens", str(args.max_total_tokens)])
    if args.max_running_requests is not None and args.max_running_requests > 0:
        cmd.extend(["--max-running-requests", str(args.max_running_requests)])
    if args.mem_fraction_static is not None and args.mem_fraction_static > 0.0:
        cmd.extend(["--mem-fraction-static", str(args.mem_fraction_static)])
    if args.schedule_policy:
        cmd.extend(["--schedule-policy", args.schedule_policy])
    if args.log_level:
        cmd.extend(["--log-level", args.log_level])
    if mode == "decode" and args.num_reserved_decode_tokens is not None:
        cmd.extend(
            ["--num-reserved-decode-tokens", str(args.num_reserved_decode_tokens)]
        )
    if mode == "decode" and args.decode_output_estimate_tokens is not None:
        cmd.extend(
            [
                "--disaggregation-decode-output-estimate-tokens",
                str(args.decode_output_estimate_tokens),
            ]
        )
    return cmd + extra_flags


def _decode_layerkv_flags(args: argparse.Namespace) -> List[str]:
    if bool(getattr(args, "disable_layerkv", False)):
        return []
    reclaim_limit_mb = (
        None
        if bool(getattr(args, "no_layerkv_reclaim_limit_mb", False))
        else args.layerkv_reclaim_limit_mb
    )
    flags = layerkv_flags(
        mode=args.layerkv_mode,
        policy=args.layerkv_policy,
        kvc_block_tokens=args.layerkv_kvc_block_tokens,
        reclaim_limit_mb=reclaim_limit_mb,
        kvc_backend=args.layerkv_kvc_backend,
        dynamic_pressure_from_kvc=args.layerkv_dynamic_pressure_from_kvc,
        scheduler=args.layerkv_kvc_scheduler,
        runtime_profile=args.layerkv_runtime_profile,
        debug_stats=True,
        profile_detail=True,
        expert_backing_cache_mb=args.expert_backing_cache_mb,
        expert_cpu_backing_mode=args.expert_cpu_backing_mode,
        expert_forward_hooks=args.layerkv_expert_forward_hooks,
        expert_collector_only=args.layerkv_expert_collector_only,
        expert_hotness_sample_interval=args.layerkv_expert_hotness_sample_interval,
        expert_install_layers_per_step=args.expert_install_layers_per_step,
        expert_install_budget_mb=args.expert_install_budget_mb,
        expert_install_target_steps=args.expert_install_target_steps,
    )
    if args.layerkv_virtual_scratch_tokens is not None:
        flags.extend(
            [
                "--layerkv-virtual-scratch-tokens",
                str(args.layerkv_virtual_scratch_tokens),
            ]
        )
    return flags


def _router_cmd(
    args: argparse.Namespace, prefill_port: int, decode_port: int
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--prefill",
        f"http://127.0.0.1:{prefill_port}",
        "--decode",
        f"http://127.0.0.1:{decode_port}",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.router_port),
        "--prometheus-port",
        str(args.prometheus_port),
    ]


def _post_smoke(port: int, args: argparse.Namespace, path: Path) -> Dict[str, Any]:
    payload = {
        "text": "Hello from a PD CoResid trace replay smoke request.",
        "sampling_params": {
            "max_new_tokens": 8,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "stream": False,
    }
    req = request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=args.request_timeout_s) as resp:
            body = resp.read().decode("utf-8", "replace")
        result = {
            "success": True,
            "status": 200,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "response": json.loads(body),
        }
    except error.HTTPError as exc:
        result = {
            "success": False,
            "status": exc.code,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "error": exc.read().decode("utf-8", "replace"),
        }
    except Exception as exc:
        result = {
            "success": False,
            "status": "",
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "error": repr(exc),
        }
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_payload(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(args.payload_path).open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if args.max_requests > 0 and len(rows) >= args.max_requests:
                break
    return rows


def _stream_generate(
    base_url: str, row: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    target_out = max(1, int(row["target_output_tokens"]))
    payload = {
        "input_ids": row["input_ids"],
        "sampling_params": {
            "max_new_tokens": target_out,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "stream": True,
    }
    req = request.Request(
        base_url + "/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    submit_t = time.perf_counter()
    first_t: Optional[float] = None
    last_obj: Dict[str, Any] = {}
    error_s = ""
    actual_output_tokens = 0
    try:
        with request.urlopen(req, timeout=args.request_timeout_s) as resp:
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
                        output_ids = obj.get("output_ids")
                        if isinstance(output_ids, list):
                            actual_output_tokens = len(output_ids)
                        if "error" in obj:
                            error_s = json.dumps(obj["error"])
                except Exception:
                    pass
    except Exception as exc:
        error_s = repr(exc)
    end_t = time.perf_counter()
    ttft_ms = ((first_t or end_t) - submit_t) * 1000.0
    e2e_ms = (end_t - submit_t) * 1000.0
    tpot_ms = max(0.0, e2e_ms - ttft_ms) / max(1, target_out - 1)
    return {
        "request_id": row["request_id"],
        "arrival_s": row["arrival_s"],
        "target_context_tokens": row["target_context_tokens"],
        "actual_context_tokens": row["actual_context_tokens"],
        "target_output_tokens": row["target_output_tokens"],
        "actual_output_tokens": actual_output_tokens,
        "prompt_source": row.get("prompt_source", ""),
        "prompt_record_id": row.get("prompt_record_id", ""),
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "tpot_ms": tpot_ms,
        "success": not error_s,
        "error": error_s,
        "last_obj": last_obj,
    }


def _replay_requests(
    args: argparse.Namespace, base_url: str, rows: List[Dict[str, Any]], path: Path
) -> Tuple[List[Dict[str, Any]], float]:
    results: List[Dict[str, Any]] = []
    start = time.perf_counter()
    with path.open("w") as out:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_client_workers
        ) as pool:
            futures = []
            for row in rows:
                due = start + float(row["arrival_s"]) / max(1e-9, args.time_scale)
                sleep_s = due - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                submit_offset_s = time.perf_counter() - start
                fut = pool.submit(_stream_generate, base_url, row, args)
                fut.submit_offset_s = submit_offset_s  # type: ignore[attr-defined]
                futures.append(fut)
            for fut in concurrent.futures.as_completed(futures):
                item = fut.result()
                item["submit_offset_s"] = getattr(fut, "submit_offset_s", 0.0)
                results.append(item)
                out.write(json.dumps(item, separators=(",", ":")) + "\n")
                out.flush()
    wall_ms = (time.perf_counter() - start) * 1000.0
    results.sort(key=lambda x: int(x["request_id"]))
    return results, wall_ms


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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def _count_patterns(text: str) -> Dict[str, int]:
    patterns = [
        "NIXL KVManager initialized",
        "Backend UCX was instantiated",
        "waiting_timeout",
        "Decode transfer failed",
        "Prefill transfer failed",
        "NIXL KVReceiver Exception",
        "NIXL transfer encountered ERR",
        "NIXL transport error",
    ]
    return {pattern: text.count(pattern) for pattern in patterns}


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "request_id",
        "arrival_s",
        "target_context_tokens",
        "actual_context_tokens",
        "target_output_tokens",
        "actual_output_tokens",
        "prompt_source",
        "prompt_record_id",
        "submit_offset_s",
        "ttft_ms",
        "e2e_ms",
        "tpot_ms",
        "success",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _summarize(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    procs: Dict[str, subprocess.Popen],
    ready: Dict[str, bool],
    smoke: Optional[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    wall_ms: float,
    early_reason: str = "",
) -> Dict[str, Any]:
    combined_logs = "\n".join(
        _tail(paths[name]) for name in ("prefill", "decode", "router")
    )
    decode_log = _tail(paths["decode"], 2_000_000)
    stats_lines = parse_layerkv_stats(decode_log)
    stats = stats_lines[-1] if stats_lines else {}
    pattern_counts = _count_patterns(combined_logs)
    success_rows = [row for row in results if row.get("success")]
    failed_count = len(results) - len(success_rows)
    total_output_tokens = sum(
        int(row.get("target_output_tokens", 0)) for row in success_rows
    )
    actual_output_tokens = sum(
        int(row.get("actual_output_tokens", 0) or 0) for row in success_rows
    )
    ttft = [_to_float(row.get("ttft_ms")) for row in success_rows]
    e2e = [_to_float(row.get("e2e_ms")) for row in success_rows]
    tpot = [_to_float(row.get("tpot_ms")) for row in success_rows]
    planned_reclaim = _to_float(stats.get("planned_kvc_reclaim_mb")) + _to_float(
        stats.get("planned_expert_reclaim_mb")
    )
    actual_reclaim = max(
        _to_float(stats.get("physical_total_reclaim_peak_mb")),
        _to_float(stats.get("physical_kvc_reclaim_peak_mb"))
        + _to_float(stats.get("physical_expert_reclaim_mb")),
        _to_float(stats.get("physical_kvc_reclaim_mb"))
        + _to_float(stats.get("physical_expert_reclaim_mb")),
    )
    transfer_errors = (
        pattern_counts["waiting_timeout"]
        + pattern_counts["Decode transfer failed"]
        + pattern_counts["Prefill transfer failed"]
        + pattern_counts["NIXL KVReceiver Exception"]
        + pattern_counts["NIXL transfer encountered ERR"]
        + pattern_counts["NIXL transport error"]
    )
    reasons = []
    if early_reason:
        reasons.append(early_reason)
    if not all(ready.values()):
        reasons.append("not_all_ready")
    if not smoke or not smoke.get("success"):
        reasons.append("smoke_failed")
    if failed_count:
        reasons.append(f"failed_requests={failed_count}")
    if transfer_errors:
        reasons.append(f"transfer_errors={transfer_errors}")
    layerkv_disabled = bool(getattr(args, "disable_layerkv", False))
    if not layerkv_disabled and not stats:
        reasons.append("missing_layerkv_stats")
    if stats and str(stats.get("resident_group_state_error_count", 0)) not in (
        "0",
        "0.0",
    ):
        reasons.append("resident_group_state_error")
    if (
        not layerkv_disabled
        and stats
        and str(stats.get("layerkv_policy", "")) != str(args.layerkv_policy)
    ):
        reasons.append(f"unexpected_layerkv_policy={stats.get('layerkv_policy')}")

    env_summary = {}
    if args.nixl_backend:
        env_summary["SGLANG_DISAGGREGATION_NIXL_BACKEND"] = args.nixl_backend
    if args.nixl_backend_params:
        env_summary["SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS"] = (
            args.nixl_backend_params
        )
    if args.ucx_tls:
        env_summary["UCX_TLS"] = args.ucx_tls
    if args.ucx_log_level:
        env_summary["UCX_LOG_LEVEL"] = args.ucx_log_level
    if args.ucx_net_devices:
        env_summary["UCX_NET_DEVICES"] = args.ucx_net_devices
    if args.transfer_queue_size > 0:
        env_summary["SGLANG_DISAGGREGATION_QUEUE_SIZE"] = args.transfer_queue_size
    if args.thread_pool_size > 0:
        env_summary["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = args.thread_pool_size
    if args.waiting_timeout_s > 0:
        env_summary["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = args.waiting_timeout_s

    summary: Dict[str, Any] = {
        "valid": not reasons,
        "validation_reason": ";".join(reasons),
        "model_path": args.model_path,
        "payload_path": args.payload_path,
        "request_count": len(rows),
        "completed_count": len(results),
        "success_count": len(success_rows),
        "failed_count": failed_count,
        "wall_ms": wall_ms,
        "request_throughput_rps": (
            len(success_rows) / (wall_ms / 1000.0) if wall_ms > 0 else 0.0
        ),
        "output_throughput_tok_s": (
            total_output_tokens / (wall_ms / 1000.0) if wall_ms > 0 else 0.0
        ),
        "actual_output_tokens": actual_output_tokens,
        "decode_output_estimate_tokens": args.decode_output_estimate_tokens,
        "planned_reclaim_mb": planned_reclaim,
        "actual_reclaim_mb": actual_reclaim,
        "ready": ready,
        "smoke": smoke,
        "pattern_counts": pattern_counts,
        "stats_line_count": len(stats_lines),
        "layerkv_stats": stats,
        "ports": {
            "prefill": args.prefill_port,
            "decode": args.decode_port,
            "router": args.router_port,
            "prometheus": args.prometheus_port,
        },
        "env": env_summary,
        "returncodes": {name: proc.poll() for name, proc in procs.items()},
        "paths": {name: str(path) for name, path in paths.items()},
        "commands": {
            "prefill": _server_cmd(args, "prefill", args.prefill_port, []),
            "decode": _server_cmd(
                args, "decode", args.decode_port, _decode_layerkv_flags(args)
            ),
            "router": _router_cmd(args, args.prefill_port, args.decode_port),
        },
    }
    summary.update(_stats(ttft, "ttft_ms"))
    summary.update(_stats(tpot, "tpot_ms"))
    summary.update(_stats(e2e, "e2e_ms"))
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    used_ports: set[int] = set()
    args.prefill_port = _pick_port(args.prefill_port, used_ports)
    args.decode_port = _pick_port(args.decode_port, used_ports)
    args.router_port = _pick_port(args.router_port, used_ports)
    args.prometheus_port = _pick_port(args.prometheus_port, used_ports)

    paths = {
        "prefill": output_dir / "prefill.log",
        "decode": output_dir / "decode.log",
        "router": output_dir / "router.log",
        "smoke": output_dir / "smoke_response.json",
        "per_request_jsonl": output_dir / "per_request.jsonl",
        "per_request_csv": output_dir / "per_request.csv",
        "summary": output_dir / "summary.json",
    }
    for path in paths.values():
        path.unlink(missing_ok=True)

    rows = _load_payload(args)
    procs: Dict[str, subprocess.Popen] = {}
    ready = {"prefill": False, "decode": False, "router": False}
    smoke: Optional[Dict[str, Any]] = None
    results: List[Dict[str, Any]] = []
    wall_ms = 0.0

    try:
        with paths["prefill"].open("w") as prefill_log:
            procs["prefill"] = subprocess.Popen(
                _server_cmd(args, "prefill", args.prefill_port, []),
                env=_base_env(args, args.prefill_gpu),
                stdout=prefill_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        ready["prefill"] = _wait_http(
            f"http://127.0.0.1:{args.prefill_port}/health",
            procs["prefill"],
            args.startup_timeout_s,
        )
        if not ready["prefill"]:
            return _summarize(
                args,
                paths,
                procs,
                ready,
                smoke,
                rows,
                results,
                wall_ms,
                "prefill_not_ready",
            )

        with paths["decode"].open("w") as decode_log:
            procs["decode"] = subprocess.Popen(
                _server_cmd(
                    args, "decode", args.decode_port, _decode_layerkv_flags(args)
                ),
                env=_base_env(args, args.decode_gpu),
                stdout=decode_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        ready["decode"] = _wait_http(
            f"http://127.0.0.1:{args.decode_port}/health",
            procs["decode"],
            args.startup_timeout_s,
        )
        if not ready["decode"]:
            return _summarize(
                args,
                paths,
                procs,
                ready,
                smoke,
                rows,
                results,
                wall_ms,
                "decode_not_ready",
            )

        with paths["router"].open("w") as router_log:
            procs["router"] = subprocess.Popen(
                _router_cmd(args, args.prefill_port, args.decode_port),
                env=_base_env(args),
                stdout=router_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        ready["router"] = _wait_http(
            f"http://127.0.0.1:{args.router_port}/health",
            procs["router"],
            args.router_timeout_s,
        )
        if not ready["router"]:
            return _summarize(
                args,
                paths,
                procs,
                ready,
                smoke,
                rows,
                results,
                wall_ms,
                "router_not_ready",
            )

        smoke = _post_smoke(args.router_port, args, paths["smoke"])
        if not smoke.get("success"):
            return _summarize(
                args, paths, procs, ready, smoke, rows, results, wall_ms, "smoke_failed"
            )

        results, wall_ms = _replay_requests(
            args,
            f"http://127.0.0.1:{args.router_port}",
            rows,
            paths["per_request_jsonl"],
        )
        _write_csv(paths["per_request_csv"], results)
        return _summarize(args, paths, procs, ready, smoke, rows, results, wall_ms)
    finally:
        for proc in reversed(list(procs.values())):
            _terminate(proc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=_default_model_path())
    parser.add_argument(
        "--payload-path",
        default="/tmp/layerkv_coresid_azure100/trace_payload_full.jsonl",
    )
    parser.add_argument("--output-dir", default="/tmp/layerkv_pd_coresid_azure100")
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="1")
    parser.add_argument("--prefill-port", type=int, default=30000)
    parser.add_argument("--decode-port", type=int, default=30001)
    parser.add_argument("--router-port", type=int, default=8000)
    parser.add_argument("--prometheus-port", type=int, default=29002)
    parser.add_argument("--nixl-backend", default="")
    parser.add_argument("--nixl-backend-params", default="")
    parser.add_argument("--ucx-tls", default="")
    parser.add_argument("--ucx-log-level", default="")
    parser.add_argument("--ucx-net-devices", default="")
    parser.add_argument("--transfer-queue-size", type=int, default=0)
    parser.add_argument("--thread-pool-size", type=int, default=0)
    parser.add_argument("--waiting-timeout-s", type=int, default=0)
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--router-timeout-s", type=float, default=120.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--watchdog-timeout-s", type=int, default=3600)
    parser.add_argument("--max-client-workers", type=int, default=128)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument("--num-reserved-decode-tokens", type=int, default=None)
    parser.add_argument(
        "--decode-output-estimate-tokens",
        type=int,
        default=None,
        help=(
            "If set, decode admission estimates every request's remaining output "
            "with this fixed token count while requests still generate "
            "target_output_tokens from the trace."
        ),
    )
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--schedule-policy", default="")
    parser.add_argument("--log-level", default="")
    parser.add_argument(
        "--layerkv-reclaim-limit-mb", type=float, default=FIG4_RECLAIM_LIMIT_MB
    )
    parser.add_argument("--disable-layerkv", action="store_true")
    parser.add_argument("--no-layerkv-reclaim-limit-mb", action="store_true")
    parser.add_argument("--layerkv-mode", default="kvc-expert")
    parser.add_argument("--layerkv-policy", default="coresid")
    parser.add_argument("--layerkv-kvc-block-tokens", type=int, default=16)
    parser.add_argument("--layerkv-kvc-backend", default="per-layer-arena")
    parser.add_argument("--layerkv-virtual-scratch-tokens", type=int, default=4096)
    parser.add_argument("--layerkv-kvc-scheduler", default="async-deadline")
    parser.add_argument("--layerkv-runtime-profile", default="optimized")
    parser.add_argument(
        "--layerkv-dynamic-pressure-from-kvc",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--expert-backing-cache-mb", type=float, default=0.0)
    parser.add_argument("--expert-cpu-backing-mode", default="none")
    parser.add_argument(
        "--layerkv-expert-forward-hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--layerkv-expert-collector-only", action="store_true")
    parser.add_argument("--layerkv-expert-hotness-sample-interval", type=int, default=16)
    parser.add_argument("--expert-install-layers-per-step", type=int, default=1)
    parser.add_argument("--expert-install-budget-mb", type=float, default=128.0)
    parser.add_argument("--expert-install-target-steps", type=int, default=16)
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
