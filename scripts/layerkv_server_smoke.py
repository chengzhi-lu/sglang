#!/usr/bin/env python3
"""End-to-end SGLang server smoke test for LayerKV.

This validates the launch_server path, not just bench_one_batch. It starts a
single local server, waits for a non-generating readiness endpoint, sends one
/generate request, and records LayerKV stats emitted by the runtime.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
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

from layerkv_eval_common import (
    DEFAULT_MODEL_PATH,
    layerkv_flags,
    parse_layerkv_stats,
    write_csv,
)


@dataclasses.dataclass(frozen=True)
class ServerScenario:
    name: str
    enable_layerkv: bool
    mode: str = "off"
    policy: str = "none"
    reclaim_limit_mb: float = 0.0


SCENARIOS = {
    "kv_first": ServerScenario("kv_first", True, "kvc-expert", "kv-first", 512.0),
    "expert_first": ServerScenario(
        "expert_first", True, "kvc-expert", "expert-first", 512.0
    ),
    "ratio_50_50": ServerScenario(
        "ratio_50_50", True, "kvc-expert", "ratio-50-50", 512.0
    ),
    "joint_dp": ServerScenario(
        "joint_dp", True, "kvc-expert", "layer-aware-joint-dp", 512.0
    ),
}


CSV_FIELDS = [
    "scenario",
    "valid",
    "validation_reason",
    "ready",
    "returncode",
    "stats_line_count",
    "response_text",
    "layerkv_mode",
    "layerkv_policy",
    "layerkv_worker_role",
    "comparable",
    "comparability_reason",
    "planned_kvc_reclaim_mb",
    "physical_kvc_reclaim_mb",
    "planned_expert_reclaim_mb",
    "physical_expert_reclaim_mb",
    "kvc_evict_count_total",
    "kvc_reload_count_total",
    "expert_slot_rebind_count",
    "expert_materialize_count",
    "kvc_guard_pass",
    "kvc_guard_reason",
    "expert_guard_pass",
    "expert_guard_reason",
    "native_scheduler_observation_count",
    "native_schedule_policy",
    "native_schedule_forward_mode",
    "native_schedule_batch_size",
    "native_schedule_waiting_queue_len",
    "native_schedule_running_batch_size",
    "actual_reclaim_limited_by_workload",
    "summary_path",
    "stdout_path",
    "stderr_path",
]


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


def _missing_dependencies() -> List[str]:
    required = ["jsonschema", "soundfile"]
    if not getattr(_missing_dependencies, "_grammar_none", False):
        required.append("xgrammar")
    return [name for name in required if importlib.util.find_spec(name) is None]


def launch_command(
    args: argparse.Namespace, port: int, scenario: ServerScenario
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
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--moe-a2a-backend",
        "none",
        "--moe-runner-backend",
        "triton",
        "--grammar-backend",
        "none",
        "--schedule-policy",
        args.schedule_policy,
        "--log-level",
        args.log_level,
    ]
    if scenario.enable_layerkv:
        cmd.extend(
            layerkv_flags(
                mode=scenario.mode,
                policy=scenario.policy,
                reclaim_limit_mb=scenario.reclaim_limit_mb,
                kvc_block_tokens=args.kvc_block_tokens,
                scheduler=args.kvc_scheduler,
                debug_stats=True,
            )
        )
    return cmd


def validate_summary(
    scenario: ServerScenario,
    returncode: int | None,
    stats: Dict[str, Any],
    response: Dict[str, Any] | None,
) -> tuple[bool, str]:
    reasons: List[str] = []
    shutdown_by_smoke = response is not None and returncode in (
        -signal.SIGTERM,
        -signal.SIGKILL,
        -signal.SIGQUIT,
    )
    if returncode not in (None, 0) and not shutdown_by_smoke:
        reasons.append(f"server_returncode={returncode}")
    if response is None:
        reasons.append("missing_generate_response")
    if not stats:
        reasons.append("missing_layerkv_stats")
        return False, ";".join(reasons)
    if stats.get("layerkv_mode") != "kvc-expert":
        reasons.append(f"unexpected_layerkv_mode:{stats.get('layerkv_mode')}")
    if not bool(stats.get("kvc_guard_pass", False)):
        reasons.append(f"kvc_guard_failed:{stats.get('kvc_guard_reason')}")
    if not bool(stats.get("expert_guard_pass", False)):
        reasons.append(f"expert_guard_failed:{stats.get('expert_guard_reason')}")
    if not bool(stats.get("comparable", False)):
        reasons.append(f"not_comparable:{stats.get('comparability_reason')}")
    planned_kvc = float(stats.get("planned_kvc_reclaim_mb", 0.0) or 0.0)
    planned_expert = float(stats.get("planned_expert_reclaim_mb", 0.0) or 0.0)
    physical_kvc = float(stats.get("physical_kvc_reclaim_mb", 0.0) or 0.0)
    physical_expert = float(stats.get("physical_expert_reclaim_mb", 0.0) or 0.0)
    if planned_kvc > 0 and not bool(stats.get("layerkv_physical_kvc_supported", False)):
        reasons.append("physical_kvc_unsupported")
    if planned_kvc > 0 and int(stats.get("kvc_evict_count_total", 0) or 0) <= 0:
        reasons.append("no_kvc_evict")
    if planned_kvc > 0 and int(stats.get("kvc_reload_count_total", 0) or 0) <= 0:
        reasons.append("no_kvc_reload")
    if planned_expert > 0 and not bool(
        stats.get("layerkv_physical_expert_supported", False)
    ):
        reasons.append("physical_expert_unsupported")
    if planned_expert > 0 and int(stats.get("expert_slot_rebind_count", 0) or 0) <= 0:
        reasons.append("no_expert_slot_rebind")
    if planned_expert > 0 and physical_expert + 1e-3 < planned_expert:
        reasons.append("insufficient_expert_reclaim")
    if planned_kvc > 0 and physical_kvc <= 0:
        reasons.append("no_physical_kvc_reclaim")
    return not reasons, ";".join(reasons)


def run_server_scenario(
    args: argparse.Namespace, scenario: ServerScenario, output_dir: Path
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "server.stdout.log"
    stderr_path = output_dir / "server.stderr.log"
    summary_path = output_dir / "server_smoke_summary.json"
    response_path = output_dir / "generate_response.json"
    for path in (stdout_path, stderr_path, summary_path, response_path):
        path.unlink(missing_ok=True)

    missing_deps = _missing_dependencies()
    if missing_deps:
        summary = {
            "scenario": scenario.name,
            "valid": False,
            "validation_reason": "missing_server_dependency:" + ",".join(missing_deps),
            "ready": False,
            "returncode": "",
            "stats_line_count": 0,
            "final_stats": {},
            "response": None,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "response_path": str(response_path),
            "summary_path": str(summary_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        return summary

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

    cmd = launch_command(args, port, scenario)
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
            ready_url = f"http://127.0.0.1:{port}{args.ready_endpoint}"
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                if _http_get(ready_url, timeout=5.0):
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
            response = {
                "http_error": exc.code,
                "body": exc.read().decode("utf-8", "replace"),
            }
        except Exception as exc:
            response = {"error": repr(exc)}
        finally:
            _terminate(proc)

    combined_logs = _tail(stdout_path, 200000) + "\n" + _tail(stderr_path, 200000)
    all_stats = parse_layerkv_stats(combined_logs)
    final_stats = all_stats[-1] if all_stats else {}
    valid, reason = validate_summary(scenario, proc.returncode, final_stats, response)
    summary = {
        "scenario": scenario.name,
        "valid": valid,
        "validation_reason": reason,
        "ready": ready,
        "port": port,
        "returncode": proc.returncode,
        "cmd": cmd,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "response_path": str(response_path),
        "summary_path": str(summary_path),
        "stats_line_count": len(all_stats),
        "final_stats": final_stats,
        "response": response,
        "stdout_tail": _tail(stdout_path),
        "stderr_tail": _tail(stderr_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def summary_to_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    stats = summary.get("final_stats") or {}
    response = summary.get("response") or {}
    planned_kvc = float(stats.get("planned_kvc_reclaim_mb", 0.0) or 0.0)
    physical_kvc = float(stats.get("physical_kvc_reclaim_mb", 0.0) or 0.0)
    limited_by_workload = planned_kvc > 0 and physical_kvc + 1e-3 < planned_kvc
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "scenario": summary.get("scenario", ""),
            "valid": summary.get("valid", False),
            "validation_reason": summary.get("validation_reason", ""),
            "ready": summary.get("ready", False),
            "returncode": summary.get("returncode", ""),
            "stats_line_count": summary.get("stats_line_count", 0),
            "response_text": (
                response.get("text", "") if isinstance(response, dict) else ""
            ),
            "actual_reclaim_limited_by_workload": limited_by_workload,
            "summary_path": summary.get("summary_path", ""),
            "stdout_path": summary.get("stdout_path", ""),
            "stderr_path": summary.get("stderr_path", ""),
        }
    )
    for field in CSV_FIELDS:
        if field in stats:
            row[field] = stats[field]
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default="outputs/layerkv/server_smoke")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--tmpdir", default="/data/wenyan/tmp")
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--ready-endpoint",
        default="/model_info",
        help="Non-generating endpoint used to detect server readiness.",
    )
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--prompt",
        default=(
            "LayerKV server smoke prompt with enough context tokens to exercise "
            "decode-time KV residency recovery."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--enable-layerkv", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--layerkv-mode", default="kvc-expert")
    parser.add_argument("--layerkv-policy", default="layer-aware-joint-dp")
    parser.add_argument("--reclaim-limit-mb", type=float, default=512.0)
    parser.add_argument("--schedule-policy", default="fcfs")
    parser.add_argument("--kvc-block-tokens", type=int, default=16)
    parser.add_argument("--kvc-scheduler", default="async-deadline")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        choices=sorted(SCENARIOS),
        help="Run a server validation matrix. Omit to run one custom scenario.",
    )
    args = parser.parse_args()
    if not args.enable_layerkv or args.layerkv_mode != "kvc-expert":
        parser.error("LayerKV server smoke only supports --layerkv-mode kvc-expert")

    # The smoke forces grammar-backend=none, so xgrammar is not required here.
    _missing_dependencies._grammar_none = True

    root_output_dir = Path(args.output_dir)
    if args.scenarios:
        summaries = []
        for name in args.scenarios:
            scenario = SCENARIOS[name]
            print(f"[layerkv-server-smoke] running {name}", flush=True)
            summary = run_server_scenario(args, scenario, root_output_dir / name)
            summaries.append(summary)
            print(
                "[layerkv-server-smoke] "
                f"{name} valid={summary['valid']} reason={summary['validation_reason']}",
                flush=True,
            )
        rows = [summary_to_row(summary) for summary in summaries]
        csv_path = root_output_dir / "server_validation.csv"
        summary_path = root_output_dir / "server_validation_summary.json"
        root_output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(csv_path, rows, CSV_FIELDS)
        invalid = [row for row in rows if str(row.get("valid")) != "True"]
        out = {
            "csv": str(csv_path),
            "total_runs": len(rows),
            "valid_runs": len(rows) - len(invalid),
            "invalid_runs": len(invalid),
            "all_valid": not invalid,
            "rows": rows,
        }
        summary_path.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0 if not invalid else 1

    scenario = ServerScenario(
        name="custom",
        enable_layerkv=bool(args.enable_layerkv),
        mode=args.layerkv_mode,
        policy=args.layerkv_policy,
        reclaim_limit_mb=args.reclaim_limit_mb,
    )
    summary = run_server_scenario(args, scenario, root_output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
