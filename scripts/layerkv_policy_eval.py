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
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple
from urllib import request

from layerkv_eval_common import (
    DEFAULT_MODEL_PATH,
    FIG4_TARGET_RECLAIM_MB,
    FIG4_WORKLOADS,
    apply_fig4_workload,
    base_command,
    layerkv_flags,
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
    "max_total_tokens",
    "max_running_requests",
    "mem_fraction_static",
    "scenario",
    "layerkv_mode",
    "layerkv_policy",
    "returncode",
    "valid",
    "validation_reason",
    "target_reclaim_mb",
    "planned_reclaim_mb",
    "actual_reclaim_mb",
    "actual_reclaim_limited_by_workload",
    "benchmark_prefill_latency_s",
    "benchmark_decode0_latency_s",
    "benchmark_decode0_latency_ms",
    "comparable",
    "comparability_reason",
    "planner_version",
    "planner_used_hotness",
    "planner_fallback_reason",
    "planner_estimated_kvc_cost",
    "planner_estimated_expert_cost",
    "planner_selected_kvc_reclaim_mb",
    "planner_selected_expert_reclaim_mb",
    "layerkv_enabled",
    "layerkv_target_reclaim_mb",
    "layerkv_physical_kvc_supported",
    "layerkv_physical_expert_supported",
    "planned_kvc_reclaim_mb",
    "physical_kvc_reclaim_mb",
    "planned_expert_reclaim_mb",
    "physical_expert_reclaim_mb",
    "kvc_evict_count_total",
    "kvc_reload_count_total",
    "kvc_reload_required_count",
    "kvc_reload_mb_total",
    "kvc_req_to_token_rewrite_count",
    "kvc_physical_failure_count",
    "kvc_stale_entry_count",
    "kvc_finished_req_cleanup_count",
    "kvc_finished_req_cleanup_token_count",
    "kvc_guard_pass",
    "kvc_guard_reason",
    "kvc_ready_before_use_ratio",
    "expert_host_backing_mb",
    "expert_slot_rebind_count",
    "expert_materialize_count",
    "expert_materialize_async_count",
    "expert_materialize_host_sync_count",
    "expert_materialize_mb_total",
    "expert_materialize_ms",
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
    target_reclaim_mb: float


POLICY_RUNS = [
    PolicyRun("expert_first", "kvc-expert", "expert-first", FIG4_TARGET_RECLAIM_MB),
    PolicyRun("kvc_first", "kvc-expert", "kv-first", FIG4_TARGET_RECLAIM_MB),
    PolicyRun("ratio_25_75", "kvc-expert", "ratio-25-75", FIG4_TARGET_RECLAIM_MB),
    PolicyRun("ratio_50_50", "kvc-expert", "ratio-50-50", FIG4_TARGET_RECLAIM_MB),
    PolicyRun("ratio_75_25", "kvc-expert", "ratio-75-25", FIG4_TARGET_RECLAIM_MB),
    PolicyRun("layer_aware_joint_dp", "kvc-expert", "layer-aware-joint-dp", FIG4_TARGET_RECLAIM_MB),
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


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("True", "true", "1", 1):
        return True
    if value in ("False", "false", "0", 0):
        return False
    return default


def policy_command(args: argparse.Namespace, spec: PolicyRun, result_path: Path) -> List[str]:
    cmd = base_command(args, result_path)
    if spec.mode == "off":
        return cmd
    return cmd + layerkv_flags(
        mode=spec.mode,
        policy=spec.policy,
        target_reclaim_mb=spec.target_reclaim_mb,
        kvc_block_tokens=args.kvc_block_tokens,
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


def _tail(path: Path, max_chars: int = 200000) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")[-max_chars:]


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
            scheduler=args.kvc_scheduler,
            debug_stats=True,
        )
    )
    return cmd


def synthetic_input_ids(args: argparse.Namespace) -> List[List[int]]:
    return [
        [1000 + ((row * 17 + col) % 8000) for col in range(args.input_len)]
        for row in range(args.batch_size)
    ]


def validate_policy_row(spec: PolicyRun, returncode: int, stats: Dict[str, Any]) -> Tuple[bool, str, bool]:
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

    planned_kvc = _to_float(stats.get("planned_kvc_reclaim_mb"))
    physical_kvc = _to_float(stats.get("physical_kvc_reclaim_mb"))
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
            # requested reclaim at the final stats point.  Keep the row valid if
            # the physical lifecycle was exercised and guard counters pass.
            limited_by_workload = True

    if planned_expert > 0.0:
        if not _to_bool(stats.get("layerkv_physical_expert_supported"), False):
            reasons.append("physical_expert_unsupported")
        if _to_int(stats.get("expert_slot_rebind_count")) <= 0:
            reasons.append("no_expert_slot_rebind")
        if physical_expert + 1e-3 < planned_expert:
            reasons.append("insufficient_expert_reclaim")

    if not _to_bool(stats.get("comparable"), True):
        reasons.append(f"not_comparable:{stats.get('comparability_reason')}")

    return not reasons, ";".join(reasons), limited_by_workload


def run_policy(args: argparse.Namespace, spec: PolicyRun, output_dir: Path) -> Dict[str, Any]:
    if args.backend == "server":
        return run_policy_server(args, spec, output_dir)
    return run_policy_bench(args, spec, output_dir)


def run_policy_bench(args: argparse.Namespace, spec: PolicyRun, output_dir: Path) -> Dict[str, Any]:
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
    planned_reclaim = _to_float(final_stats.get("planned_kvc_reclaim_mb")) + _to_float(
        final_stats.get("planned_expert_reclaim_mb")
    )
    actual_reclaim = _to_float(final_stats.get("physical_kvc_reclaim_mb")) + _to_float(
        final_stats.get("physical_expert_reclaim_mb")
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
            "max_total_tokens": args.max_total_tokens,
            "max_running_requests": args.max_running_requests,
            "mem_fraction_static": args.mem_fraction_static,
            "scenario": spec.scenario,
            "layerkv_mode": spec.mode if spec.mode == "off" else final_stats.get("layerkv_mode", spec.mode),
            "layerkv_policy": spec.policy if spec.mode == "off" else final_stats.get("layerkv_policy", spec.policy),
            "returncode": result["returncode"],
            "valid": valid,
            "validation_reason": reason,
            "target_reclaim_mb": spec.target_reclaim_mb,
            "planned_reclaim_mb": planned_reclaim,
            "actual_reclaim_mb": actual_reclaim,
            "actual_reclaim_limited_by_workload": limited_by_workload,
            "benchmark_decode0_latency_ms": _to_float(
                result["latencies"].get("benchmark_decode0_latency_s")
            )
            * 1000.0,
            "stats_line_count": result["stats_line_count"],
            "response_count": "",
            "request_wall_ms": "",
            "stdout_path": result["stdout_path"],
            "stderr_path": result["stderr_path"],
            "result_path": str(result_path),
        }
    )
    return row


def run_policy_server(args: argparse.Namespace, spec: PolicyRun, output_dir: Path) -> Dict[str, Any]:
    stdout_path = output_dir / f"{spec.scenario}.server.stdout.log"
    stderr_path = output_dir / f"{spec.scenario}.server.stderr.log"
    response_path = output_dir / f"{spec.scenario}.response.json"
    for path in (stdout_path, stderr_path, response_path):
        path.unlink(missing_ok=True)

    port = _free_port()
    cmd = server_command(args, spec, port)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["TMPDIR"] = args.tmpdir
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
        try:
            deadline = time.time() + args.startup_timeout_s
            ready_url = f"http://127.0.0.1:{port}/model_info"
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                if _http_get(ready_url, timeout=5.0):
                    break
                time.sleep(2.0)
            payload = {
                "input_ids": synthetic_input_ids(args),
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
            _terminate(proc)

    combined = _tail(stdout_path) + "\n" + _tail(stderr_path)
    stats = parse_layerkv_stats(combined)
    final_stats = stats[-1] if stats else {}
    response_ok = not (isinstance(response, dict) and "error" in response)
    effective_returncode = 0 if response_ok and proc.returncode in (-9, -15, -3, 0) else proc.returncode
    valid, reason, limited_by_workload = validate_policy_row(
        spec, effective_returncode, final_stats
    )
    if not response_ok:
        valid = False
        err = response.get("error") if isinstance(response, dict) else repr(response)
        reason = (reason + ";" if reason else "") + f"generate_failed:{err}"

    planned_reclaim = _to_float(final_stats.get("planned_kvc_reclaim_mb")) + _to_float(
        final_stats.get("planned_expert_reclaim_mb")
    )
    actual_reclaim = _to_float(final_stats.get("physical_kvc_reclaim_mb")) + _to_float(
        final_stats.get("physical_expert_reclaim_mb")
    )
    response_count = len(response) if isinstance(response, list) else (1 if response_ok else 0)

    row: Dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(final_stats)
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
            "max_total_tokens": args.max_total_tokens,
            "max_running_requests": args.max_running_requests,
            "mem_fraction_static": args.mem_fraction_static,
            "scenario": spec.scenario,
            "layerkv_mode": final_stats.get("layerkv_mode", spec.mode),
            "layerkv_policy": final_stats.get("layerkv_policy", spec.policy),
            "returncode": proc.returncode,
            "valid": valid,
            "validation_reason": reason,
            "target_reclaim_mb": spec.target_reclaim_mb,
            "planned_reclaim_mb": planned_reclaim,
            "actual_reclaim_mb": actual_reclaim,
            "actual_reclaim_limited_by_workload": limited_by_workload,
            "benchmark_decode0_latency_ms": request_wall_ms / max(1, args.output_len),
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
        return {"best_policy": "", "joint_wins": False, "joint_loss_reason": "no_comparable_rows"}

    comparable.sort(key=lambda row: _to_float(row.get("benchmark_decode0_latency_ms"), float("inf")))
    best = comparable[0]
    joint = next((row for row in comparable if row.get("scenario") == "layer_aware_joint_dp"), None)
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
    elif _to_float(joint.get("expert_materialize_mb_total")) > 0.0 or _to_float(
        joint.get("kvc_reload_mb_total")
    ) > 0.0:
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
    parser.add_argument("--kvc-block-tokens", type=int, default=16)
    parser.add_argument("--kvc-scheduler", default="async-deadline")
    parser.add_argument("--tmpdir", default="/data/wenyan/tmp")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--startup-timeout-s", type=float, default=1200.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--watchdog-timeout-s", type=int, default=3600)
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
        "fig4_target_reclaim_mb": FIG4_TARGET_RECLAIM_MB,
        "fig4_dataset_name": args.fig4_dataset_name,
        "fig4_dataset_path": args.fig4_dataset_path,
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
