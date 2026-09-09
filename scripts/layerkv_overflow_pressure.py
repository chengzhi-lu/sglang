#!/usr/bin/env python3
"""Measure native KV admission pressure against SharedVMM KV overflow.

The parent process runs the native and LayerKV arms serially.  Each child
creates one engine, performs an untimed warmup, and then runs a small
concurrency staircase in the same process so model loading and shape compilation are not
included in workload timing.  The two arms share the same model, GPU, token
pool budget, request inputs, and arrival schedule.

This is a pressure diagnostic, not a throughput acceptance benchmark.  A
pressure level is useful only when the native arm has a non-zero scheduler
queue time for at least one request; the report keeps that qualification
separate from client-observed TTFT.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any
from urllib import error as url_error
from urllib import request as url_request


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from layerkv_hybrid_validation import engine_kwargs
from layerkv_eval_common import parse_layerkv_stats


def _reuse_venv_build_tools() -> None:
    """Expose tools installed beside the active Python to child JIT builds.

    ``ninja`` is an external build tool, not an experiment setting.  The
    existing venv already provides it, but callers may invoke this script with
    the venv's absolute Python while their shell PATH omits the venv ``bin``.
    """

    # ``venv/bin/python`` is a symlink on this host; resolving it would point
    # at ``/usr/bin`` and lose the packages installed beside the venv.
    venv_bin = Path(sys.prefix) / "bin"
    if not (venv_bin / "ninja").is_file():
        venv_bin = Path(sys.executable).parent
    ninja = venv_bin / "ninja"
    if not ninja.is_file():
        return
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(venv_bin) not in path_entries:
        os.environ["PATH"] = os.pathsep.join(
            [str(venv_bin), *[entry for entry in path_entries if entry]]
        )


_reuse_venv_build_tools()


PRESSURE_COUNTERS = (
    "forward_extend_count",
    "forward_decode_count",
    "shared_expert_admission_recall_count",
    "shared_expert_admission_recovered_tokens",
    "shared_expert_admission_overflow_count",
    "shared_expert_admission_overflow_tokens",
    "kvc_evict_count_total",
    "kvc_evict_async_count",
    "kvc_evict_async_finalize_count",
    "kvc_evict_pending_token_count",
    "physical_kvc_reclaim_mb",
    "scheduler_budget_pressure_tokens",
    "scheduler_budget_credit_tokens",
)

SHARED_VMM_COUNTERS = (
    "kv_overflow_active_tokens",
    "kv_overflow_activation_count",
    "kv_overflow_admission_activation_count",
    "kv_overflow_expert_slots_reclaimed",
    "kv_overflow_restore_count",
    "current_expert_slots",
    "physical_bytes",
    "loan_bytes",
    "blocked_kv_tokens",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _layerkv_stats(engine: Any) -> dict[str, Any]:
    try:
        info = engine.get_server_info()
    except Exception:
        return {}
    states = info.get("internal_states", []) if isinstance(info, dict) else []
    if not states or not isinstance(states[0], dict):
        return {}
    value = states[0].get("layerkv")
    return _json_safe(value) if isinstance(value, dict) else {}


def _memory_snapshot(torch: Any) -> dict[str, Any]:
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "total_bytes": int(total_bytes),
        "free_bytes": int(free_bytes),
        "used_bytes": int(total_bytes - free_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _nested_stats(stats: dict[str, Any]) -> dict[str, Any]:
    shared = stats.get("shared_vmm")
    return shared if isinstance(shared, dict) else {}


def _counter_snapshot(stats: dict[str, Any]) -> dict[str, Any]:
    shared = _nested_stats(stats)
    result = {}
    for key in PRESSURE_COUNTERS:
        value = stats.get(key, 0)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = value
    for key in SHARED_VMM_COUNTERS:
        value = shared.get(key, 0)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[f"shared_vmm.{key}"] = value
    return result


def _counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    # A crashed server may leave the final stats response empty.  Treat that
    # as unavailable evidence instead of subtracting a zero snapshot and
    # reporting spurious negative deltas.
    if not before or not after:
        return {}
    left = _counter_snapshot(before)
    right = _counter_snapshot(after)
    result = {}
    for key in sorted(set(left) | set(right)):
        old, new = left.get(key, 0), right.get(key, 0)
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            result[key] = new - old
    return result


def _build_engine_args(args: argparse.Namespace, case: str) -> argparse.Namespace:
    return SimpleNamespace(
        model_path=args.model_path,
        dtype=args.dtype,
        gpu=args.gpu,
        case=case,
        context_length=args.context_length,
        max_total_tokens=args.max_total_tokens,
        batch_size=args.max_running_requests,
        mem_fraction_static=args.mem_fraction_static,
        scratch_tokens=args.scratch_tokens,
        block_tokens=args.block_tokens,
        shared_expert_layer=-1,
        shared_expert_all_layers=args.shared_expert_all_layers,
        shared_expert_kv_overflow_tokens=args.shared_expert_kv_overflow_tokens,
        shared_expert_kv_min_slots=args.shared_expert_kv_min_slots,
        expert_initial_slots=args.expert_initial_slots,
        expert_extra_slots=args.expert_extra_slots,
        reclaim_mb=args.reclaim_mb,
        expert_cpu_backing_mode=args.expert_cpu_backing_mode,
        shared_expert_free_kv_donors=False,
        shared_expert_retain_across_requests=False,
        expert_install_layers_per_step=1,
        expert_install_budget_mb=128.0,
        expert_copy_force_drain=False,
        expert_prefetch_lookahead_layers=1,
        expert_prefetch_min_route_overlap=0.5,
        expert_policy="kv-first",
    )


def _engine_kwargs(args: argparse.Namespace, case: str) -> dict[str, Any]:
    result = engine_kwargs(_build_engine_args(args, case))
    result.update(
        host="127.0.0.1",
        port=args.server_port,
        enable_metrics=True,
        enable_request_time_stats_logging=True,
        stream_interval=1,
        skip_tokenizer_init=True,
        watchdog_timeout=args.watchdog_timeout,
    )
    return result


def _free_port() -> int:
    """Reserve a localhost port for one server arm."""

    for port in range(30000, 50000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("failed to find a free localhost port in [30000, 50000)")


def _server_command(args: argparse.Namespace, case: str, port: int) -> list[str]:
    """Build the HTTP server command with the same budget as Engine mode."""

    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        args.dtype,
        "--context-length",
        str(args.context_length),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--max-running-requests",
        str(args.max_running_requests),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--disable-overlap-schedule",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "-1",
        "--moe-a2a-backend",
        "none",
        "--moe-runner-backend",
        "triton",
        "--attention-backend",
        "triton",
        "--grammar-backend",
        "none",
        "--random-seed",
        "0",
        "--enable-metrics",
        "--enable-request-time-stats-logging",
        "--stream-interval",
        "1",
        "--skip-tokenizer-init",
        "--watchdog-timeout",
        str(args.watchdog_timeout),
        "--log-level",
        "info",
    ]
    if case != "offload":
        return command
    command.extend(
        [
            "--enable-layerkv",
            "--layerkv-mode",
            "kvc-expert",
            "--layerkv-policy",
            "expert-first",
            "--layerkv-reclaim-limit-mb",
            str(args.reclaim_mb),
            "--layerkv-kvc-block-tokens",
            str(args.block_tokens),
            "--layerkv-kvc-backend",
            "per-layer-arena",
            "--layerkv-kvc-scheduler",
            "async-deadline",
            "--layerkv-runtime-profile",
            "optimized",
            "--layerkv-virtual-scratch-tokens",
            str(args.scratch_tokens),
            "--layerkv-shared-expert-all-layers",
            "--layerkv-shared-expert-initial-slots",
            str(args.expert_initial_slots),
            "--layerkv-shared-expert-extra-slots",
            str(args.expert_extra_slots),
            "--layerkv-shared-expert-kv-overflow-tokens",
            str(args.shared_expert_kv_overflow_tokens),
            "--layerkv-shared-expert-kv-min-slots",
            str(args.shared_expert_kv_min_slots),
            "--layerkv-expert-cpu-backing-mode",
            args.expert_cpu_backing_mode,
            "--layerkv-expert-install-layers-per-step",
            "1",
            "--layerkv-expert-install-budget-mb",
            "128",
            "--layerkv-expert-prefetch-lookahead-layers",
            "1",
            "--layerkv-expert-prefetch-min-route-overlap",
            "0.5",
            "--layerkv-debug-stats",
        ]
    )
    return command


def _server_env(args: argparse.Namespace) -> dict[str, str]:
    """Set only launcher/runtime variables required by the server child."""

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONNOUSERSITE"] = "1"
    repo_python = str(REPO_ROOT / "python")
    env["PYTHONPATH"] = (
        repo_python
        if not env.get("PYTHONPATH")
        else repo_python + os.pathsep + env["PYTHONPATH"]
    )
    return env


def _http_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = url_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with url_request.urlopen(req, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    return value if isinstance(value, dict) else {"value": value}


def _http_get_json(url: str, timeout: float) -> dict[str, Any]:
    with url_request.urlopen(url, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    return value if isinstance(value, dict) else {"value": value}


def _server_layerkv_stats(base_url: str, timeout: float) -> dict[str, Any]:
    try:
        info = _http_get_json(f"{base_url}/server_info", timeout)
    except Exception:
        return {}
    states = info.get("internal_states", [])
    if not isinstance(states, list) or not states or not isinstance(states[0], dict):
        return {}
    value = states[0].get("layerkv")
    return _json_safe(value) if isinstance(value, dict) else {}


def _nvidia_smi_snapshot(gpu: int) -> dict[str, Any]:
    """Read device-global allocation for a server child process."""

    command = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-gpu=memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        fields = [item.strip() for item in completed.stdout.strip().split(",")]
        if len(fields) != 3:
            raise ValueError(f"unexpected nvidia-smi output: {completed.stdout!r}")
        total_mb, used_mb, free_mb = (int(item) for item in fields)
        return {
            "source": "nvidia-smi",
            "gpu": int(gpu),
            "total_bytes": total_mb * 1024 * 1024,
            "used_bytes": used_mb * 1024 * 1024,
            "free_bytes": free_mb * 1024 * 1024,
        }
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as error:
        return {"source": "nvidia-smi", "gpu": int(gpu), "error": str(error)}


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _wait_server_ready(
    process: subprocess.Popen[Any], base_url: str, timeout_s: float
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            _http_get_json(f"{base_url}/model_info", timeout=5.0)
            return True
        except Exception:
            time.sleep(1.0)
    return False


def _http_stream_one(
    *,
    base_url: str,
    input_ids: list[int],
    output_tokens: int,
    started: float,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
        "stream": True,
    }
    req = url_request.Request(
        f"{base_url}/generate",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    try:
        with url_request.urlopen(req, timeout=timeout) as response:
            for raw_line in response:
                for line in raw_line.decode("utf-8", "replace").splitlines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        continue
                    value = json.loads(data)
                    if not isinstance(value, dict):
                        continue
                    if "error" in value:
                        return events, final, json.dumps(value, sort_keys=True)
                    final = value
                    meta = value.get("meta_info", {})
                    output_ids = value.get("output_ids") or []
                    completion_tokens = int(meta.get("completion_tokens", 0) or 0)
                    if output_ids or completion_tokens > 0:
                        events.append(
                            {
                                "elapsed_s": time.perf_counter() - started,
                                "tokens": completion_tokens or len(output_ids),
                            }
                        )
    except (OSError, url_error.URLError, url_error.HTTPError, ValueError) as error:
        return events, final, f"{type(error).__name__}: {error}"
    return events, final, None


def _server_batch_level(
    *,
    base_url: str,
    inputs: list[list[int]],
    output_tokens: int,
    stagger_s: float,
    request_timeout_s: float,
) -> tuple[
    list[list[dict[str, Any]]],
    list[dict[str, Any] | None],
    float,
    list[float | None],
    list[str],
]:
    started = time.perf_counter()
    events: list[list[dict[str, Any]]] = [[] for _ in inputs]
    final: list[dict[str, Any] | None] = [None for _ in inputs]
    submitted: list[float | None] = [None for _ in inputs]
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, len(inputs))) as pool:
        futures = []
        for index, input_ids in enumerate(inputs):
            submitted[index] = time.perf_counter() - started
            futures.append(
                pool.submit(
                    _http_stream_one,
                    base_url=base_url,
                    input_ids=input_ids,
                    output_tokens=output_tokens,
                    started=started,
                    timeout=request_timeout_s,
                )
            )
            if index + 1 < len(inputs) and stagger_s > 0:
                time.sleep(stagger_s)
        for index, future in enumerate(futures):
            try:
                events[index], final[index], error = future.result(
                    timeout=request_timeout_s
                )
                if error:
                    errors.append(f"request[{index}] {error}")
            except Exception as error:  # preserve the other requests' evidence
                errors.append(f"request[{index}] {type(error).__name__}: {error}")
    return events, final, time.perf_counter() - started, submitted, errors


def _request_inputs(
    *,
    level: int,
    requests: int,
    input_tokens: int,
    token_offset: int,
    repeat_inputs: bool = False,
) -> list[list[int]]:
    if repeat_inputs:
        base = [
            100 + (token_offset + level * 17 + position) % 80
            for position in range(input_tokens)
        ]
        return [list(base) for _ in range(requests)]
    return [
        [
            100 + (token_offset + level * 17 + request * 13 + position) % 80
            for position in range(input_tokens)
        ]
        for request in range(requests)
    ]


def _input_hash(inputs: list[list[int]]) -> str:
    payload = json.dumps(inputs, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _response_token_ids(response: dict[str, Any] | None) -> list[int]:
    if not response:
        return []
    output_ids = response.get("output_ids", [])
    return [int(value) for value in output_ids]


def _response_logprob_ids(response: dict[str, Any] | None) -> list[int]:
    if not response:
        return []
    values = response.get("meta_info", {}).get("output_token_logprobs", [])
    result = []
    for item in values or []:
        try:
            result.append(int(item[1]))
        except (IndexError, TypeError, ValueError):
            return []
    return result


async def _batch_level(
    engine: Any,
    *,
    inputs: list[list[int]],
    output_tokens: int,
    stagger_s: float = 0.0,
):
    """Submit requests simultaneously or as an explicit staggered wave."""

    started = time.perf_counter()
    submitted: list[float | None] = [0.0 for _ in inputs]
    events: list[list[dict[str, Any]]] = [[] for _ in inputs]
    final: list[dict[str, Any] | None] = [None for _ in inputs]
    errors: list[str] = []

    async def collect_one(index: int) -> str | None:
        try:
            stream = await engine.async_generate(
                input_ids=[inputs[index]],
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": output_tokens,
                    "ignore_eos": True,
                },
                return_logprob=True,
                logprob_start_len=-1,
                stream=True,
            )
            async for response in stream:
                meta = response.get("meta_info", {})
                events[index].append(
                    {
                        "elapsed_s": time.perf_counter() - started,
                        "tokens": int(meta.get("completion_tokens", 0) or 0),
                    }
                )
                final[index] = _json_safe(response)
        except Exception as error:  # keep the other arm's evidence intact
            return f"request[{index}] {type(error).__name__}: {error}"
        return None

    if stagger_s <= 0:
        submitted = [0.0 for _ in inputs]
        errors.extend(
            error
            for error in await asyncio.gather(
                *(collect_one(index) for index in range(len(inputs)))
            )
            if error
        )
    else:
        tasks = []
        for index in range(len(inputs)):
            submitted[index] = time.perf_counter() - started
            tasks.append(asyncio.create_task(collect_one(index)))
            if index + 1 < len(inputs):
                await asyncio.sleep(stagger_s)
        errors.extend(error for error in await asyncio.gather(*tasks) if error)
    return events, final, time.perf_counter() - started, submitted, errors


def _summarize_level(
    *,
    level: int,
    request_count: int,
    inputs: list[list[int]],
    events: list[list[dict[str, Any]]],
    final: list[dict[str, Any] | None],
    elapsed_s: float,
    submitted: list[float | None],
    errors: list[str],
    stagger_s: float,
    before_stats: dict[str, Any],
    after_stats: dict[str, Any],
    memory_before: dict[str, Any],
    memory_after: dict[str, Any],
    arrival_mode: str = "",
    expected_output_tokens: int = 1,
) -> dict[str, Any]:
    requests_result = []
    for index in range(request_count):
        row = events[index]
        start = submitted[index]
        response = final[index]
        meta = response.get("meta_info", {}) if response else {}
        first = row[0]["elapsed_s"] if row else None
        last = row[-1]["elapsed_s"] if row else None
        completion_tokens = int(meta.get("completion_tokens", 0) or 0)
        requests_result.append(
            {
                "index": index,
                "submitted_s": start,
                "first_token_s": first,
                "ttft_s": None if first is None or start is None else first - start,
                "e2e_latency_s": None if last is None or start is None else last - start,
                "scheduler_queue_time_s": (
                    float(meta["queue_time"])
                    if meta.get("queue_time") is not None
                    else None
                ),
                "completion_tokens": completion_tokens,
                "success": bool(
                    response is not None
                    and completion_tokens >= expected_output_tokens
                ),
                "output_ids": _response_token_ids(response),
                "output_logprob_ids": _response_logprob_ids(response),
            }
        )

    def values(name: str) -> list[float]:
        return [
            float(row[name])
            for row in requests_result
            if row[name] is not None and math.isfinite(float(row[name]))
        ]

    def summary(name: str) -> dict[str, Any]:
        data = sorted(values(name))
        if not data:
            return {"count": 0, "mean_s": None, "p50_s": None, "max_s": None}
        return {
            "count": len(data),
            "mean_s": sum(data) / len(data),
            "p50_s": data[(len(data) - 1) // 2],
            "max_s": data[-1],
        }

    completed = sum(1 for row in requests_result if row["success"])
    output_count = sum(row["completion_tokens"] for row in requests_result)
    return {
        "level": level,
        "requests": request_count,
        "input_tokens": [len(item) for item in inputs],
        "input_hash": _input_hash(inputs),
        "arrival_mode": arrival_mode
        or (
            "simultaneous-engine-batch"
            if stagger_s <= 0
            else "staggered-engine-requests"
        ),
        "initial_requests": request_count,
        "elapsed_s": elapsed_s,
        "output_tokens": output_count,
        "output_tokens_per_s": output_count / elapsed_s if elapsed_s > 0 else 0.0,
        "success_count": completed,
        "success_rate": completed / request_count if request_count else 0.0,
        "queue_time_s": summary("scheduler_queue_time_s"),
        "ttft_s": summary("ttft_s"),
        "e2e_latency_s": summary("e2e_latency_s"),
        "requests_detail": requests_result,
        "responses": final,
        "errors": errors,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "stats_before": _counter_snapshot(before_stats),
        "stats_after": _counter_snapshot(after_stats),
        "stats_delta": _counter_delta(before_stats, after_stats),
    }


def _run_server_case(args: argparse.Namespace) -> int:
    """Run one arm with real HTTP arrival timing and one server child."""

    result: dict[str, Any] = {
        "schema": 2,
        "case": args.case,
        "arrival_mode": "http-server-client",
        "args": _json_safe(vars(args)),
        "levels": [],
    }
    server_port = args.server_port or _free_port()
    base_url = f"http://127.0.0.1:{server_port}"
    server_log_path = args.output_dir / f"{args.case}.server.log"
    command = _server_command(args, args.case, server_port)
    process: subprocess.Popen | None = None
    try:
        with server_log_path.open("w") as server_log:
            process = subprocess.Popen(
                command,
                env=_server_env(args),
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            result.update(
                {
                    "server_command": command,
                    "server_port": server_port,
                    "server_log": str(server_log_path),
                }
            )
            result["ready"] = _wait_server_ready(
                process, base_url, args.startup_timeout_s
            )
            if not result["ready"]:
                result["error"] = "server_not_ready"
                result["server_returncode"] = process.poll()
                (args.output_dir / f"{args.case}.json").write_text(
                    json.dumps(result, indent=2, default=str)
                )
                return 1
            if args.warmup_batch_size != 1:
                result["error"] = "http-server-mode-requires-warmup-batch-size-1"
                (args.output_dir / f"{args.case}.json").write_text(
                    json.dumps(result, indent=2, default=str)
                )
                return 1

            warmup_inputs = _request_inputs(
                level=0,
                requests=1,
                input_tokens=args.warmup_input_tokens,
                token_offset=args.token_offset,
                repeat_inputs=args.repeat_inputs,
            )
            warmup_started = time.perf_counter()
            warmup_response = _http_json(
                f"{base_url}/generate",
                {
                    "input_ids": warmup_inputs[0],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": args.warmup_output_tokens,
                        "ignore_eos": True,
                    },
                    "stream": False,
                },
                timeout=args.request_timeout_s,
            )
            result["warmup"] = {
                "input_tokens": args.warmup_input_tokens,
                "batch_size": 1,
                "output_tokens": args.warmup_output_tokens,
                "elapsed_s": time.perf_counter() - warmup_started,
                "response": _json_safe(warmup_response),
                "memory_after": _nvidia_smi_snapshot(args.gpu),
                "layerkv_stats_after": _server_layerkv_stats(
                    base_url, timeout=10.0
                ),
            }

            for level, request_count in enumerate(args.request_counts):
                inputs = _request_inputs(
                    level=level,
                    requests=request_count,
                    input_tokens=args.input_tokens,
                    token_offset=args.token_offset,
                    repeat_inputs=args.repeat_inputs,
                )
                before_stats = _server_layerkv_stats(base_url, timeout=30.0)
                memory_before = _nvidia_smi_snapshot(args.gpu)
                events, final, elapsed_s, submitted, errors = _server_batch_level(
                    base_url=base_url,
                    inputs=inputs,
                    output_tokens=args.output_tokens,
                    stagger_s=args.stagger_s,
                    request_timeout_s=args.request_timeout_s,
                )
                after_stats = _server_layerkv_stats(base_url, timeout=30.0)
                memory_after = _nvidia_smi_snapshot(args.gpu)
                row = _summarize_level(
                    level=level,
                    request_count=request_count,
                    inputs=inputs,
                    events=events,
                    final=final,
                    elapsed_s=elapsed_s,
                    submitted=submitted,
                    errors=errors,
                    stagger_s=args.stagger_s,
                    arrival_mode="http-server-client",
                    expected_output_tokens=args.output_tokens,
                    before_stats=before_stats,
                    after_stats=after_stats,
                    memory_before=memory_before,
                    memory_after=memory_after,
                )
                result["levels"].append(row)
                (args.output_dir / f"{args.case}.partial.json").write_text(
                    json.dumps(result, indent=2, default=str)
                )
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in (
                                "level",
                                "requests",
                                "elapsed_s",
                                "success_rate",
                                "queue_time_s",
                                "ttft_s",
                                "e2e_latency_s",
                                "output_tokens_per_s",
                                "stats_delta",
                                "memory_after",
                            )
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            result["final_layerkv_stats"] = _server_layerkv_stats(
                base_url, timeout=30.0
            )
            result["final_memory"] = _nvidia_smi_snapshot(args.gpu)
            result["server_returncode_before_shutdown"] = process.poll()
            (args.output_dir / f"{args.case}.json").write_text(
                json.dumps(result, indent=2, default=str)
            )
            return 0
    except (OSError, RuntimeError, ValueError, url_error.URLError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["server_returncode_before_shutdown"] = (
            process.poll() if process is not None else None
        )
        (args.output_dir / f"{args.case}.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
        return 1
    finally:
        if process is not None:
            _stop_process_group(process)


def _run_case(args: argparse.Namespace) -> int:
    if args.arrival_mode == "server":
        return _run_server_case(args)
    import sglang as sgl
    import torch

    engine = sgl.Engine(**_engine_kwargs(args, args.case))
    result: dict[str, Any] = {
        "schema": 1,
        "case": args.case,
        "args": _json_safe(vars(args)),
        "levels": [],
    }
    try:
        warmup_inputs = _request_inputs(
            level=0,
            requests=args.warmup_batch_size,
            input_tokens=args.warmup_input_tokens,
            token_offset=args.token_offset,
            repeat_inputs=args.repeat_inputs,
        )
        warmup_started = time.perf_counter()
        engine.generate(
            input_ids=warmup_inputs,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": args.warmup_output_tokens,
                "ignore_eos": True,
            },
        )
        warmup_elapsed = time.perf_counter() - warmup_started
        memory_after_warmup = _memory_snapshot(torch)
        torch.cuda.reset_peak_memory_stats()
        result["warmup"] = {
            "input_tokens": args.warmup_input_tokens,
            "batch_size": args.warmup_batch_size,
            "output_tokens": args.warmup_output_tokens,
            "elapsed_s": warmup_elapsed,
            "memory_after": memory_after_warmup,
            "layerkv_stats_after": _layerkv_stats(engine),
        }

        for level, request_count in enumerate(args.request_counts):
            inputs = _request_inputs(
                level=level,
                requests=request_count,
                input_tokens=args.input_tokens,
                token_offset=args.token_offset,
                repeat_inputs=args.repeat_inputs,
            )
            before_stats = _layerkv_stats(engine)
            memory_before = _memory_snapshot(torch)
            events, final, elapsed_s, submitted, errors = engine.loop.run_until_complete(
                _batch_level(
                    engine,
                    inputs=inputs,
                    output_tokens=args.output_tokens,
                    stagger_s=args.stagger_s,
                )
            )
            after_stats = _layerkv_stats(engine)
            memory_after = _memory_snapshot(torch)
            row = _summarize_level(
                level=level,
                request_count=request_count,
                inputs=inputs,
                events=events,
                final=final,
                elapsed_s=elapsed_s,
                submitted=submitted,
                errors=errors,
                stagger_s=args.stagger_s,
                before_stats=before_stats,
                after_stats=after_stats,
                memory_before=memory_before,
                memory_after=memory_after,
                expected_output_tokens=args.output_tokens,
            )
            result["levels"].append(row)
            (args.output_dir / f"{args.case}.partial.json").write_text(
                json.dumps(result, indent=2, default=str)
            )
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "level",
                            "requests",
                            "elapsed_s",
                            "success_rate",
                            "queue_time_s",
                            "ttft_s",
                            "e2e_latency_s",
                            "output_tokens_per_s",
                            "stats_delta",
                        )
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        result["final_layerkv_stats"] = _layerkv_stats(engine)
        result["final_memory"] = _memory_snapshot(torch)
        (args.output_dir / f"{args.case}.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
        return 0
    finally:
        try:
            engine.shutdown()
        except OSError as error:
            print(f"engine shutdown warning: {error}", file=sys.stderr, flush=True)


def _compare_responses(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = left.get("levels", [])
    right_rows = right.get("levels", [])
    exact = len(left_rows) == len(right_rows)
    rows = []
    for baseline, offload in zip(left_rows, right_rows):
        baseline_responses = baseline.get("responses", [])
        offload_responses = offload.get("responses", [])
        level_exact = len(baseline_responses) == len(offload_responses)
        max_logprob_delta = 0.0
        for base, optimized in zip(baseline_responses, offload_responses):
            level_exact &= _response_token_ids(base) == _response_token_ids(optimized)
            base_probs = base.get("meta_info", {}).get("output_token_logprobs", []) if base else []
            opt_probs = optimized.get("meta_info", {}).get("output_token_logprobs", []) if optimized else []
            if len(base_probs) != len(opt_probs):
                level_exact = False
                continue
            for base_item, opt_item in zip(base_probs, opt_probs):
                try:
                    max_logprob_delta = max(
                        max_logprob_delta,
                        abs(float(base_item[0]) - float(opt_item[0])),
                    )
                    level_exact &= int(base_item[1]) == int(opt_item[1])
                except (IndexError, TypeError, ValueError):
                    level_exact = False
        exact &= level_exact
        rows.append(
            {
                "level": baseline.get("level"),
                "same_request_count": baseline.get("requests") == offload.get("requests"),
                "tokens_exact": level_exact,
                "max_logprob_delta": max_logprob_delta,
                "baseline_success_rate": baseline.get("success_rate"),
                "offload_success_rate": offload.get("success_rate"),
                "baseline_queue_time_s": baseline.get("queue_time_s"),
                "offload_queue_time_s": offload.get("queue_time_s"),
                "baseline_ttft_s": baseline.get("ttft_s"),
                "offload_ttft_s": offload.get("ttft_s"),
                "baseline_e2e_latency_s": baseline.get("e2e_latency_s"),
                "offload_e2e_latency_s": offload.get("e2e_latency_s"),
                "baseline_output_tokens_per_s": baseline.get("output_tokens_per_s"),
                "offload_output_tokens_per_s": offload.get("output_tokens_per_s"),
                "baseline_memory_after": baseline.get("memory_after"),
                "offload_memory_after": offload.get("memory_after"),
                "baseline_stats_delta": baseline.get("stats_delta", {}),
                "offload_stats_delta": offload.get("stats_delta", {}),
            }
        )
    return {"correctness": bool(exact), "levels": rows}


def _parent_run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema": 1,
        "args": _json_safe(vars(args)),
        "runs": {},
    }
    original = list(sys.argv[1:])
    for case in ("baseline", "offload"):
        command = [sys.executable, str(Path(__file__).resolve()), *original, "--case", case]
        log_path = args.output_dir / f"{case}.log"
        print(f"Running {case}: {' '.join(command)}", flush=True)
        with log_path.open("w") as log:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                returncode = process.wait(timeout=args.timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                returncode = 124
            except KeyboardInterrupt:
                # Keep an interrupted parent run from orphaning its GPU child.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
        stats = parse_layerkv_stats(log_path.read_text())
        artifact = args.output_dir / f"{case}.json"
        summary["runs"][case] = {
            "returncode": returncode,
            "command": command,
            "log": str(log_path),
            "stats_count": len(stats),
            "last_log_stats": stats[-1] if stats else {},
            "artifact": str(artifact) if artifact.exists() else None,
        }
        print(
            json.dumps(
                {
                    "case": case,
                    "returncode": returncode,
                    "stats_count": len(stats),
                    "artifact": artifact.exists(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if returncode != 0:
            break

    baseline_path = args.output_dir / "baseline.json"
    offload_path = args.output_dir / "offload.json"
    if baseline_path.exists() and offload_path.exists():
        comparison = _compare_responses(
            json.loads(baseline_path.read_text()),
            json.loads(offload_path.read_text()),
        )
        summary["comparison"] = comparison
        summary["valid"] = bool(comparison["correctness"])
    else:
        summary["comparison"] = {"correctness": False, "levels": []}
        summary["valid"] = False
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(json.dumps({"valid": summary["valid"], "output_dir": str(args.output_dir)}))
    return 0 if summary["valid"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--server-port", type=int, default=30000)
    parser.add_argument(
        "--arrival-mode",
        choices=("engine", "server"),
        default="engine",
        help=(
            "engine uses the embedded Engine API; server starts a real HTTP server "
            "and uses independent client threads for arrival timing."
        ),
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="float16")
    parser.add_argument("--reclaim-mb", type=float, default=64.0)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--input-tokens", type=int, default=3000)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--request-counts", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument(
        "--stagger-s",
        type=float,
        default=0.0,
        help="Seconds between request submissions in the selected arrival mode.",
    )
    parser.add_argument("--warmup-input-tokens", type=int, default=3000)
    # One decode token is required to exercise the normal lazy SharedExpert
    # installation before measured admission waiting begins.
    parser.add_argument("--warmup-output-tokens", type=int, default=2)
    parser.add_argument("--warmup-batch-size", type=int, default=1)
    parser.add_argument("--token-offset", type=int, default=5000)
    parser.add_argument(
        "--repeat-inputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse one prompt across requests to isolate KV admission from route churn",
    )
    parser.add_argument("--scratch-tokens", type=int, default=512)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--expert-initial-slots", type=int, default=16)
    parser.add_argument("--expert-extra-slots", type=int, default=1)
    parser.add_argument("--shared-expert-kv-overflow-tokens", type=int, default=2048)
    parser.add_argument("--shared-expert-kv-min-slots", type=int, default=8)
    parser.add_argument(
        "--shared-expert-all-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--expert-cpu-backing-mode", choices=("none", "all", "selected"), default="none"
    )
    parser.add_argument("--watchdog-timeout", type=int, default=300)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--startup-timeout-s", type=float, default=1200.0)
    parser.add_argument("--request-timeout-s", type=float, default=1200.0)
    parser.add_argument("--case", choices=("baseline", "offload"), help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (
        args.gpu < 0
        or args.reclaim_mb <= 0
        or args.stagger_s < 0
        or args.server_port < 0
        or args.startup_timeout_s <= 0
        or args.request_timeout_s <= 0
    ):
        raise SystemExit("gpu must be nonnegative and reclaim-mb must be positive")
    if not 0 < args.mem_fraction_static <= 1:
        raise SystemExit("mem-fraction-static must be in (0, 1]")
    positive = (
        args.context_length,
        args.max_total_tokens,
        args.max_running_requests,
        args.input_tokens,
        args.output_tokens,
        args.warmup_input_tokens,
        args.warmup_output_tokens,
        args.warmup_batch_size,
        args.expert_initial_slots,
        args.expert_extra_slots,
        args.block_tokens,
        args.watchdog_timeout,
        args.timeout_s,
    )
    if any(value <= 0 for value in positive):
        raise SystemExit("token, request, slot and timeout values must be positive")
    if not args.request_counts:
        raise SystemExit("request-counts must be nonempty")
    if args.input_tokens + args.output_tokens > args.context_length:
        raise SystemExit("input-tokens plus output-tokens must fit context-length")
    if args.warmup_input_tokens + args.warmup_output_tokens > args.context_length:
        raise SystemExit("warmup input plus output must fit context-length")
    if args.shared_expert_kv_overflow_tokens <= 0 or args.shared_expert_kv_min_slots <= 0:
        raise SystemExit("SharedVMM overflow tokens and minimum slots must be positive")
    if args.shared_expert_kv_min_slots > args.expert_initial_slots:
        raise SystemExit("shared-expert-kv-min-slots cannot exceed expert-initial-slots")
    if not args.shared_expert_all_layers:
        raise SystemExit("this pressure script requires --shared-expert-all-layers")
    args.output_dir = args.output_dir.resolve()
    if args.case:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return _run_case(args)
    return _parent_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
