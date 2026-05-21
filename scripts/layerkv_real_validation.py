#!/usr/bin/env python3
"""Run real-model LayerKV validation on SGLang.

This script is intentionally small and conservative.  It validates that the
real Qwen3 MoE path can run with LayerKV enabled and that KVC-only runs exercise
the full evict -> reload -> req_to_token rewrite lifecycle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from layerkv_eval_common import (
    DEFAULT_MODEL_PATH,
    base_command,
    layerkv_flags,
    run_bench_command,
    write_csv,
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
    "expert_materialize_async_count",
    "expert_materialize_host_sync_count",
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
    "expert_guard_pass",
    "expert_guard_reason",
    "comparable",
    "comparability_reason",
    "stats_line_count",
    "stdout_path",
    "stderr_path",
    "result_path",
]


def scenario_command(
    args: argparse.Namespace, scenario: str, result_path: Path
) -> List[str]:
    cmd = base_command(args, result_path)
    if scenario == "baseline":
        return cmd
    if scenario == "kvc_only_reload":
        return cmd + layerkv_flags(
            mode="kvc-only",
            policy="layer-aware-joint-dp",
            target_reclaim_mb=args.kvc_target_reclaim_mb,
            kvc_block_tokens=args.kvc_block_tokens,
        )
    if scenario == "kvc_expert_kv_first":
        return cmd + layerkv_flags(
            mode="kvc-expert",
            policy="kv-first",
            target_reclaim_mb=args.expert_target_reclaim_mb,
            kvc_block_tokens=args.kvc_block_tokens,
        )
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
    result = run_bench_command(
        cmd=cmd,
        output_dir=output_dir,
        run_name=scenario,
        args=args,
    )
    final_stats = result["final_stats"]
    valid, reason = validate_scenario(scenario, result["returncode"], final_stats)

    row: Dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(result["latencies"])
    row.update(final_stats)
    row.update(
        {
            "scenario": scenario,
            "returncode": result["returncode"],
            "valid": valid,
            "validation_reason": reason,
            "stats_line_count": result["stats_line_count"],
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "result_path": str(result_path),
        }
    )
    return row


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
    write_csv(csv_path, rows, CSV_FIELDS)
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
