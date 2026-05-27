#!/usr/bin/env python3
"""Minimal PD-disaggregation smoke test for the NIXL/UCX transfer path.

The default run intentionally disables LayerKV and uses a short request. This
isolates whether the SGLang PD + NIXL/UCX baseline can move KV from prefill to
decode before testing LayerKV policies or trace-scale workloads.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
from urllib import error, request

from layerkv_eval_common import DEFAULT_MODEL_PATH, layerkv_flags

ROOT_MODEL_CANDIDATES = [
    DEFAULT_MODEL_PATH,
    (
        "/root/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/"
        "snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    ),
]


def _default_model_path() -> str:
    for path in ROOT_MODEL_CANDIDATES:
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
    env["SGLANG_DISAGGREGATION_NIXL_BACKEND"] = args.nixl_backend
    env["SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS"] = args.nixl_backend_params
    env["UCX_TLS"] = args.ucx_tls
    env["UCX_LOG_LEVEL"] = args.ucx_log_level
    if args.ucx_net_devices:
        env["UCX_NET_DEVICES"] = args.ucx_net_devices
    if args.enable_staging:
        env["SGLANG_DISAGG_STAGING_BUFFER"] = "1"
        env["SGLANG_DISAGG_STAGING_POOL_SIZE_MB"] = str(args.staging_pool_size_mb)
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


def _tail(path: Path, max_chars: int = 12000) -> str:
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


def _post_generate(
    port: int, args: argparse.Namespace, response_path: Path
) -> Dict[str, Any]:
    payload = {
        "text": args.prompt,
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
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
        parsed = json.loads(body)
        result = {
            "success": True,
            "status": 200,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "response": parsed,
        }
    except error.HTTPError as exc:
        result = {
            "success": False,
            "status": exc.code,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "error": exc.read().decode("utf-8", "replace"),
        }
    except Exception as exc:
        result = {
            "success": False,
            "status": "",
            "latency_ms": (time.perf_counter() - started) * 1000,
            "error": repr(exc),
        }
    response_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def _server_cmd(
    args: argparse.Namespace, mode: str, port: int, extra_flags: List[str]
) -> List[str]:
    return [
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
        "--trust-remote-code",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--moe-a2a-backend",
        "none",
        "--moe-runner-backend",
        "triton",
        "--grammar-backend",
        "none",
        "--watchdog-timeout",
        str(args.watchdog_timeout_s),
        "--log-level",
        args.log_level,
    ] + extra_flags


def _decode_extra_flags(args: argparse.Namespace) -> List[str]:
    flags: List[str] = []
    if args.enable_layerkv:
        flags.extend(
            layerkv_flags(
                mode=args.layerkv_mode,
                policy=args.layerkv_policy,
                reclaim_limit_mb=args.layerkv_reclaim_limit_mb,
                kvc_block_tokens=args.layerkv_kvc_block_tokens,
                kvc_backend=args.layerkv_kvc_backend,
                scheduler=args.layerkv_kvc_scheduler,
                runtime_profile=args.layerkv_runtime_profile,
                dynamic_pressure_from_kvc=args.layerkv_dynamic_pressure_from_kvc,
                debug_stats=True,
            )
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
        "response": output_dir / "response.json",
        "summary": output_dir / "summary.json",
    }
    for path in paths.values():
        path.unlink(missing_ok=True)

    procs: Dict[str, subprocess.Popen] = {}
    prefill_ready = False
    decode_ready = False
    router_ready = False
    response = None
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
        prefill_ready = _wait_http(
            f"http://127.0.0.1:{args.prefill_port}/health",
            procs["prefill"],
            args.startup_timeout_s,
        )
        if not prefill_ready:
            return _write_summary(
                args,
                paths,
                procs,
                prefill_ready,
                decode_ready,
                router_ready,
                response,
                "prefill_not_ready",
            )

        with paths["decode"].open("w") as decode_log:
            procs["decode"] = subprocess.Popen(
                _server_cmd(
                    args,
                    "decode",
                    args.decode_port,
                    _decode_extra_flags(args),
                ),
                env=_base_env(args, args.decode_gpu),
                stdout=decode_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        decode_ready = _wait_http(
            f"http://127.0.0.1:{args.decode_port}/health",
            procs["decode"],
            args.startup_timeout_s,
        )
        if not decode_ready:
            return _write_summary(
                args,
                paths,
                procs,
                prefill_ready,
                decode_ready,
                router_ready,
                response,
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
        router_ready = _wait_http(
            f"http://127.0.0.1:{args.router_port}/health",
            procs["router"],
            args.router_timeout_s,
        )
        if not router_ready:
            return _write_summary(
                args,
                paths,
                procs,
                prefill_ready,
                decode_ready,
                router_ready,
                response,
                "router_not_ready",
            )

        response = _post_generate(args.router_port, args, paths["response"])
        return _write_summary(
            args,
            paths,
            procs,
            prefill_ready,
            decode_ready,
            router_ready,
            response,
            "",
        )
    finally:
        for proc in reversed(list(procs.values())):
            _terminate(proc)


def _write_summary(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    procs: Dict[str, subprocess.Popen],
    prefill_ready: bool,
    decode_ready: bool,
    router_ready: bool,
    response: Optional[Dict[str, Any]],
    early_reason: str,
) -> Dict[str, Any]:
    logs = "\n".join(
        _tail(paths[name], 200000) for name in ("prefill", "decode", "router")
    )
    pattern_counts = _count_patterns(logs)
    transfer_errors = (
        pattern_counts["waiting_timeout"]
        + pattern_counts["Decode transfer failed"]
        + pattern_counts["Prefill transfer failed"]
        + pattern_counts["NIXL KVReceiver Exception"]
        + pattern_counts["NIXL transfer encountered ERR"]
        + pattern_counts["NIXL transport error"]
    )
    valid = bool(
        prefill_ready
        and decode_ready
        and router_ready
        and response
        and response.get("success")
        and transfer_errors == 0
    )
    reason = early_reason
    if not reason:
        if not response or not response.get("success"):
            reason = "generate_failed"
        elif transfer_errors:
            reason = "transfer_errors"

    summary = {
        "valid": valid,
        "validation_reason": reason,
        "model_path": args.model_path,
        "enable_layerkv": args.enable_layerkv,
        "env": {
            "SGLANG_DISAGGREGATION_NIXL_BACKEND": args.nixl_backend,
            "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS": args.nixl_backend_params,
            "UCX_TLS": args.ucx_tls,
            "UCX_LOG_LEVEL": args.ucx_log_level,
            "UCX_NET_DEVICES": args.ucx_net_devices,
            "SGLANG_DISAGG_STAGING_BUFFER": "1" if args.enable_staging else "",
        },
        "ports": {
            "prefill": args.prefill_port,
            "decode": args.decode_port,
            "router": args.router_port,
            "prometheus": args.prometheus_port,
        },
        "ready": {
            "prefill": prefill_ready,
            "decode": decode_ready,
            "router": router_ready,
        },
        "returncodes": {name: proc.poll() for name, proc in procs.items()},
        "pattern_counts": pattern_counts,
        "response": response,
        "paths": {name: str(path) for name, path in paths.items()},
        "commands": {
            "prefill": _server_cmd(args, "prefill", args.prefill_port, []),
            "decode": _server_cmd(
                args,
                "decode",
                args.decode_port,
                _decode_extra_flags(args),
            ),
            "router": _router_cmd(args, args.prefill_port, args.decode_port),
        },
        "log_tails": {
            "prefill": _tail(paths["prefill"]),
            "decode": _tail(paths["decode"]),
            "router": _tail(paths["router"]),
        },
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=_default_model_path())
    parser.add_argument("--output-dir", default="/tmp/layerkv_pd_nixl_ucx_smoke")
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="1")
    parser.add_argument("--prefill-port", type=int, default=30000)
    parser.add_argument("--decode-port", type=int, default=30001)
    parser.add_argument("--router-port", type=int, default=8000)
    parser.add_argument("--prometheus-port", type=int, default=29002)
    parser.add_argument("--nixl-backend", default="UCX")
    parser.add_argument("--nixl-backend-params", default="{}")
    parser.add_argument("--ucx-tls", default="cuda_ipc,cuda_copy,tcp,sm,self")
    parser.add_argument("--ucx-log-level", default="warn")
    parser.add_argument("--ucx-net-devices", default="")
    parser.add_argument("--transfer-queue-size", type=int, default=1)
    parser.add_argument("--thread-pool-size", type=int, default=1)
    parser.add_argument("--waiting-timeout-s", type=int, default=120)
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--router-timeout-s", type=float, default=120.0)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--watchdog-timeout-s", type=int, default=3600)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--prompt", default="Hello from a NIXL UCX PD smoke test.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--enable-staging", action="store_true")
    parser.add_argument("--staging-pool-size-mb", type=int, default=4096)
    parser.add_argument("--enable-layerkv", action="store_true")
    parser.add_argument("--layerkv-mode", default="kvc-expert")
    parser.add_argument("--layerkv-policy", default="layer-aware-joint-dp")
    parser.add_argument("--layerkv-reclaim-limit-mb", type=float, default=512.0)
    parser.add_argument("--layerkv-kvc-block-tokens", type=int, default=16)
    parser.add_argument("--layerkv-kvc-backend", default="per-layer-arena")
    parser.add_argument("--layerkv-kvc-scheduler", default="async-deadline")
    parser.add_argument("--layerkv-runtime-profile", default="optimized")
    parser.add_argument("--layerkv-dynamic-pressure-from-kvc", action="store_true")
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
