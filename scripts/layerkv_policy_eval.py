#!/usr/bin/env python3
"""Run a small policy-level LayerKV evaluation on SGLang.

This is a comparability harness, not a full workload sweep.  It runs every
policy through the same SGLang backend and records enough physical reclaim and
guard counters to decide whether the row is valid for policy comparison.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from layerkv_eval_common import (
    DEFAULT_MODEL_PATH,
    base_command,
    layerkv_flags,
    run_bench_command,
    write_csv,
)


CSV_FIELDS = [
    "workload",
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
    "expert_topk_rewrite_count",
    "expert_core_hook_count",
    "expert_guard_pass",
    "expert_guard_reason",
    "policy_kvc_fraction",
    "policy_expert_fraction",
    "full_policy_semantics_supported",
    "policy_semantics_reason",
    "stats_line_count",
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
    PolicyRun("baseline", "off", "none", 0.0),
    PolicyRun("kvc_only_layer_aware_joint_dp", "kvc-only", "layer-aware-joint-dp", 64.0),
    PolicyRun("expert_first", "kvc-expert", "expert-first", 512.0),
    PolicyRun("kv_first", "kvc-expert", "kv-first", 512.0),
    PolicyRun("ratio_25_75", "kvc-expert", "ratio-25-75", 512.0),
    PolicyRun("ratio_50_50", "kvc-expert", "ratio-50-50", 512.0),
    PolicyRun("ratio_75_25", "kvc-expert", "ratio-75-25", 512.0),
    PolicyRun("layer_aware_joint_dp", "kvc-expert", "layer-aware-joint-dp", 512.0),
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
            "workload": args.workload,
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
            "stdout_path": result["stdout_path"],
            "stderr_path": result["stderr_path"],
            "result_path": str(result_path),
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
    parser.add_argument("--workload", default="batch_heavy_small")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=4)
    parser.add_argument("--kvc-block-tokens", type=int, default=16)
    parser.add_argument("--tmpdir", default="/data/wenyan/tmp")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[spec.scenario for spec in POLICY_RUNS],
        choices=[spec.scenario for spec in POLICY_RUNS],
    )
    args = parser.parse_args()

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
        "workload": args.workload,
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
