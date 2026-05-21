#!/usr/bin/env python3
"""Run real-model LayerKV validation on SGLang.

This script is intentionally small and conservative.  It validates that the
real Qwen3 MoE path can run with LayerKV enabled and that KVC-only runs exercise
the full evict -> reload -> req_to_token rewrite lifecycle.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List


DEFAULT_MODEL_PATH = (
    "/data/wenyan/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-30B-A3B/"
    "snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
)

CSV_FIELDS = [
    "scenario",
    "returncode",
    "valid",
    "validation_reason",
    "benchmark_prefill_latency_s",
    "benchmark_decode0_latency_s",
    "layerkv_enabled",
    "layerkv_mode",
    "layerkv_policy",
    "layerkv_target_reclaim_mb",
    "layerkv_physical_kvc_supported",
    "layerkv_physical_expert_supported",
    "planned_kvc_reclaim_mb",
    "physical_kvc_reclaim_mb",
    "kvc_evict_count_total",
    "kvc_reload_count_total",
    "kvc_reload_required_count",
    "kvc_req_to_token_rewrite_count",
    "kvc_physical_failure_count",
    "kvc_stale_entry_count",
    "kvc_guard_pass",
    "kvc_guard_reason",
    "planned_expert_reclaim_mb",
    "physical_expert_reclaim_mb",
    "expert_host_backing_mb",
    "expert_slot_rebind_count",
    "expert_materialize_count",
    "expert_topk_rewrite_count",
    "expert_core_hook_count",
    "expert_guard_pass",
    "expert_guard_reason",
    "comparable",
    "comparability_reason",
    "stats_line_count",
    "stdout_path",
    "stderr_path",
    "result_path",
]


def parse_layerkv_stats(text: str) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    marker = "LayerKV stats after "
    for line in text.splitlines():
        if marker not in line:
            continue
        try:
            payload = line.split(": ", 1)[1]
            parsed = ast.literal_eval(payload)
        except Exception:
            continue
        if isinstance(parsed, dict):
            stats.append(parsed)
    return stats


def parse_benchmark_latencies(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    match = re.search(r"Benchmark .*?Prefill\. latency:\s*([0-9.]+) s", text, re.S)
    if match:
        out["benchmark_prefill_latency_s"] = float(match.group(1))
    match = re.search(
        r"Benchmark .*?Decode 0\. Batch size: \d+, latency:\s*([0-9.]+) s",
        text,
        re.S,
    )
    if match:
        out["benchmark_decode0_latency_s"] = float(match.group(1))
    return out


def base_command(args: argparse.Namespace, result_path: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "sglang.bench_one_batch",
        "--model-path",
        args.model_path,
        "--batch-size",
        str(args.batch_size),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--moe-a2a-backend",
        "none",
        "--moe-runner-backend",
        "triton",
        "--result-filename",
        str(result_path),
        "--log-level",
        args.log_level,
    ]


def scenario_command(
    args: argparse.Namespace, scenario: str, result_path: Path
) -> List[str]:
    cmd = base_command(args, result_path)
    if scenario == "baseline":
        return cmd
    if scenario == "kvc_only_reload":
        return cmd + [
            "--enable-layerkv",
            "--layerkv-mode",
            "kvc-only",
            "--layerkv-policy",
            "layer-aware-joint-dp",
            "--layerkv-target-reclaim-mb",
            str(args.kvc_target_reclaim_mb),
            "--layerkv-kvc-block-tokens",
            str(args.kvc_block_tokens),
            "--layerkv-kvc-scheduler",
            "async-deadline",
            "--layerkv-debug-stats",
        ]
    if scenario == "kvc_expert_kv_first":
        return cmd + [
            "--enable-layerkv",
            "--layerkv-mode",
            "kvc-expert",
            "--layerkv-policy",
            "kv-first",
            "--layerkv-target-reclaim-mb",
            str(args.expert_target_reclaim_mb),
            "--layerkv-kvc-block-tokens",
            str(args.kvc_block_tokens),
            "--layerkv-kvc-scheduler",
            "async-deadline",
            "--layerkv-debug-stats",
        ]
    raise ValueError(f"unknown scenario: {scenario}")


def validate_scenario(
    scenario: str, returncode: int, stats: Dict[str, Any]
) -> tuple[bool, str]:
    reasons: List[str] = []
    if returncode != 0:
        reasons.append(f"process_returncode={returncode}")
    if scenario == "baseline":
        return not reasons, ";".join(reasons)
    if not stats:
        reasons.append("missing_layerkv_stats")
        return False, ";".join(reasons)
    if not bool(stats.get("kvc_guard_pass", False)):
        reasons.append(f"kvc_guard_failed:{stats.get('kvc_guard_reason')}")
    if int(stats.get("kvc_physical_failure_count", 0) or 0) != 0:
        reasons.append("kvc_physical_failure_count_nonzero")
    if int(stats.get("kvc_stale_entry_count", 0) or 0) != 0:
        reasons.append("kvc_stale_entry_count_nonzero")
    if scenario == "kvc_only_reload":
        if not bool(stats.get("layerkv_physical_kvc_supported", False)):
            reasons.append("physical_kvc_unsupported")
        if int(stats.get("kvc_evict_count_total", 0) or 0) <= 0:
            reasons.append("no_kvc_evict")
        if int(stats.get("kvc_reload_count_total", 0) or 0) <= 0:
            reasons.append("no_kvc_reload")
        if int(stats.get("kvc_req_to_token_rewrite_count", 0) or 0) <= 0:
            reasons.append("no_req_to_token_rewrite")
    elif scenario == "kvc_expert_kv_first":
        if not bool(stats.get("layerkv_physical_expert_supported", False)):
            reasons.append("physical_expert_unsupported")
        if not bool(stats.get("expert_guard_pass", False)):
            reasons.append(f"expert_guard_failed:{stats.get('expert_guard_reason')}")
        if int(stats.get("expert_slot_rebind_count", 0) or 0) <= 0:
            reasons.append("no_expert_slot_rebind")
        planned = float(stats.get("planned_expert_reclaim_mb", 0.0) or 0.0)
        physical = float(stats.get("physical_expert_reclaim_mb", 0.0) or 0.0)
        if planned > 0 and physical + 1e-3 < planned:
            reasons.append("insufficient_expert_reclaim")
        if not bool(stats.get("comparable", False)):
            reasons.append(f"not_comparable:{stats.get('comparability_reason')}")
    return not reasons, ";".join(reasons)


def run_scenario(
    args: argparse.Namespace, scenario: str, output_dir: Path
) -> Dict[str, Any]:
    result_path = output_dir / f"{scenario}.jsonl"
    stdout_path = output_dir / f"{scenario}.stdout.log"
    stderr_path = output_dir / f"{scenario}.stderr.log"
    for path in (result_path, stdout_path, stderr_path):
        path.unlink(missing_ok=True)

    cmd = scenario_command(args, scenario, result_path)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["TMPDIR"] = args.tmpdir
    repo_python = str(Path.cwd() / "python")
    env["PYTHONPATH"] = (
        repo_python if not env.get("PYTHONPATH") else repo_python + os.pathsep + env["PYTHONPATH"]
    )

    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout_s,
    )
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)

    text = proc.stdout + "\n" + proc.stderr
    stats = parse_layerkv_stats(text)
    final_stats = stats[-1] if stats else {}
    valid, reason = validate_scenario(scenario, proc.returncode, final_stats)

    row: Dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(parse_benchmark_latencies(text))
    row.update(final_stats)
    row.update(
        {
            "scenario": scenario,
            "returncode": proc.returncode,
            "valid": valid,
            "validation_reason": reason,
            "stats_line_count": len(stats),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "result_path": str(result_path),
        }
    )
    return row


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default="outputs/layerkv/real_qwen3_30b_a3b_validation")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=4)
    parser.add_argument("--kvc-target-reclaim-mb", type=float, default=64.0)
    parser.add_argument("--expert-target-reclaim-mb", type=float, default=512.0)
    parser.add_argument("--kvc-block-tokens", type=int, default=16)
    parser.add_argument("--tmpdir", default="/data/wenyan/tmp")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["baseline", "kvc_only_reload", "kvc_expert_kv_first"],
        choices=["baseline", "kvc_only_reload", "kvc_expert_kv_first"],
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario in args.scenarios:
        print(f"[layerkv-real-validation] running {scenario}", flush=True)
        row = run_scenario(args, scenario, output_dir)
        rows.append(row)
        print(
            "[layerkv-real-validation] "
            f"{scenario} valid={row['valid']} reason={row['validation_reason']}",
            flush=True,
        )

    csv_path = output_dir / "real_validation.csv"
    summary_path = output_dir / "real_validation_summary.json"
    write_csv(csv_path, rows)
    invalid = [row for row in rows if str(row.get("valid")) != "True"]
    summary = {
        "csv": str(csv_path),
        "total_runs": len(rows),
        "valid_runs": len(rows) - len(invalid),
        "invalid_runs": len(invalid),
        "all_valid": not invalid,
        "rows": rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not invalid else 1


if __name__ == "__main__":
    raise SystemExit(main())
