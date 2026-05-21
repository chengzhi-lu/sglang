#!/usr/bin/env python3
"""End-to-end SGLang server smoke test for LayerKV.

This validates the launch_server path, not just bench_one_batch.  It starts a
single local server, waits for /health, sends one /generate request, and records
LayerKV stats emitted by the runtime.
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
from typing import Any, Dict, List
from urllib import error, request

from layerkv_eval_common import DEFAULT_MODEL_PATH, layerkv_flags, parse_layerkv_stats


def _free_port() -> int:
    # SGLang derives auxiliary ports from the HTTP port (e.g. port + 10000),
    # so keep the selected port comfortably below 65535.
    for port in range(30000, 50000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("failed to find a free localhost port in [30000, 50000)")


def _http_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    return parsed if isinstance(parsed, dict) else {"response": parsed}


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
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text[-max_chars:]


def launch_command(args: argparse.Namespace, port: int) -> List[str]:
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
        "--log-level",
        args.log_level,
    ]
    if args.enable_layerkv:
        cmd.extend(
            layerkv_flags(
                mode=args.layerkv_mode,
                policy=args.layerkv_policy,
                target_reclaim_mb=args.target_reclaim_mb,
                kvc_block_tokens=args.kvc_block_tokens,
                scheduler=args.kvc_scheduler,
                debug_stats=True,
            )
        )
    return cmd


def validate_summary(
    args: argparse.Namespace,
    returncode: int | None,
    stats: Dict[str, Any],
    response: Dict[str, Any] | None,
) -> tuple[bool, str]:
    reasons: List[str] = []
    shutdown_by_smoke = response is not None and returncode in (-signal.SIGTERM, -signal.SIGKILL, -signal.SIGQUIT)
    if returncode not in (None, 0) and not shutdown_by_smoke:
        reasons.append(f"server_returncode={returncode}")
    if response is None:
        reasons.append("missing_generate_response")
    if not args.enable_layerkv:
        return not reasons, ";".join(reasons)
    if not stats:
        reasons.append("missing_layerkv_stats")
        return False, ";".join(reasons)
    if not bool(stats.get("kvc_guard_pass", False)):
        reasons.append(f"kvc_guard_failed:{stats.get('kvc_guard_reason')}")
    if not bool(stats.get("expert_guard_pass", False)):
        reasons.append(f"expert_guard_failed:{stats.get('expert_guard_reason')}")
    if not bool(stats.get("comparable", False)):
        reasons.append(f"not_comparable:{stats.get('comparability_reason')}")
    planned_kvc = float(stats.get("planned_kvc_reclaim_mb", 0.0) or 0.0)
    planned_expert = float(stats.get("planned_expert_reclaim_mb", 0.0) or 0.0)
    if planned_kvc > 0 and not bool(stats.get("layerkv_physical_kvc_supported", False)):
        reasons.append("physical_kvc_unsupported")
    if planned_expert > 0 and not bool(stats.get("layerkv_physical_expert_supported", False)):
        reasons.append("physical_expert_unsupported")
    return not reasons, ";".join(reasons)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default="outputs/layerkv/server_smoke")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--tmpdir", default="/data/wenyan/tmp")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--enable-layerkv", action="store_true", default=True)
    parser.add_argument("--layerkv-mode", default="kvc-expert")
    parser.add_argument("--layerkv-policy", default="layer-aware-joint-dp")
    parser.add_argument("--target-reclaim-mb", type=float, default=512.0)
    parser.add_argument("--kvc-block-tokens", type=int, default=16)
    parser.add_argument("--kvc-scheduler", default="async-deadline")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "server.stdout.log"
    stderr_path = output_dir / "server.stderr.log"
    summary_path = output_dir / "server_smoke_summary.json"
    response_path = output_dir / "generate_response.json"
    for path in (stdout_path, stderr_path, summary_path, response_path):
        path.unlink(missing_ok=True)

    port = args.port or _free_port()
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["TMPDIR"] = args.tmpdir
    repo_python = str(Path.cwd() / "python")
    env["PYTHONPATH"] = (
        repo_python
        if not env.get("PYTHONPATH")
        else repo_python + os.pathsep + env["PYTHONPATH"]
    )

    cmd = launch_command(args, port)
    response: Dict[str, Any] | None = None
    ready = False
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
            deadline = time.time() + args.startup_timeout_s
            health_url = f"http://127.0.0.1:{port}/health"
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                if _http_get(health_url, timeout=5.0):
                    ready = True
                    break
                time.sleep(2.0)
            if ready:
                response = _http_json(
                    f"http://127.0.0.1:{port}/generate",
                    {
                        "text": args.prompt,
                        "sampling_params": {
                            "max_new_tokens": args.max_new_tokens,
                            "temperature": 0.0,
                        },
                        "stream": False,
                    },
                    timeout=args.request_timeout_s,
                )
                response_path.write_text(json.dumps(response, indent=2, sort_keys=True))
        except error.HTTPError as exc:
            response = {"http_error": exc.code, "body": exc.read().decode("utf-8", "replace")}
        except Exception as exc:
            response = {"error": repr(exc)}
        finally:
            _terminate(proc)

    combined_logs = _tail(stdout_path, 200000) + "\n" + _tail(stderr_path, 200000)
    all_stats = parse_layerkv_stats(combined_logs)
    final_stats = all_stats[-1] if all_stats else {}
    valid, reason = validate_summary(args, proc.returncode, final_stats, response)
    summary = {
        "valid": valid,
        "validation_reason": reason,
        "ready": ready,
        "port": port,
        "returncode": proc.returncode,
        "cmd": cmd,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "response_path": str(response_path),
        "stats_line_count": len(all_stats),
        "final_stats": final_stats,
        "response": response,
        "stdout_tail": _tail(stdout_path),
        "stderr_tail": _tail(stderr_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
