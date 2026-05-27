#!/usr/bin/env python3
"""Validate LayerKV KVC physical offload semantics on SGLang.

The script intentionally uses a tiny dummy-weight model run.  It validates the
KVC runtime lifecycle rather than model quality:

* policy-to-KVC fraction pass-through
* physical KVC evict/reload under decode
* req_to_token rewrite after reload
* host backing and residency-map consistency guards
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Tuple

DEFAULT_MODEL_PATH = (
    "/data/wenyan/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)

POLICIES = [
    "expert-first",
    "kv-first",
    "ratio-25-75",
    "ratio-50-50",
    "ratio-75-25",
    "layer-aware-joint-dp",
]

EXPECTED_FRACTIONS = {
    # In kvc-only mode all enabled policies must exercise the physical KVC
    # lifecycle.  Mixed KVC/expert split semantics are validated by
    # layerkv_policy_eval.py, where expert offload is available.
    "expert-first": 1.0,
    "kv-first": 1.0,
    "ratio-25-75": 1.0,
    "ratio-50-50": 1.0,
    "ratio-75-25": 1.0,
    "layer-aware-joint-dp": 1.0,
}

CSV_FIELDS = [
    "policy",
    "target_reclaim_mb",
    "page_size",
    "returncode",
    "valid",
    "validation_reason",
    "layerkv_enabled",
    "layerkv_mode",
    "layerkv_policy",
    "layerkv_target_reclaim_mb",
    "layerkv_kvc_backend",
    "layerkv_kvc_backend_semantics",
    "layerkv_kvc_backend_limited",
    "layerkv_kvc_backend_ready",
    "layerkv_kvc_backend_reason",
    "layerkv_kvc_scheduler",
    "layerkv_runtime_profile",
    "layerkv_physical_kvc_supported",
    "layerkv_physical_expert_supported",
    "layerkv_expert_layer_count",
    "requested_total_reclaim_mb",
    "effective_kvc_reclaim_mb",
    "policy_kvc_fraction",
    "policy_expert_fraction",
    "full_policy_semantics_supported",
    "policy_semantics_reason",
    "planner_version",
    "planner_used_hotness",
    "planner_fallback_reason",
    "planner_estimated_kvc_cost",
    "planner_estimated_expert_cost",
    "planner_estimated_kvc_overlap_ms",
    "planner_estimated_kvc_exposed_ms",
    "planner_estimated_expert_backing_miss_cost",
    "planner_estimated_expert_materialize_cost",
    "planner_dp_table_build_ms",
    "planner_dp_lookup_ms",
    "planner_dp_kvc_candidate_count",
    "planner_dp_expert_candidate_count",
    "planner_cache_hit_count",
    "planner_cache_miss_count",
    "planner_selected_kvc_reclaim_mb",
    "planner_selected_expert_reclaim_mb",
    "planner_dp_infeasible_kvc_candidates",
    "planner_dp_infeasible_expert_candidates",
    "planner_dp_selected_total_cost",
    "planner_dp_selected_kvc_cost",
    "planner_dp_selected_expert_cost",
    "selected_kvc_tokens_by_layer",
    "selected_expert_evictions_by_layer",
    "planned_kvc_reclaim_mb",
    "physical_kvc_reclaim_mb",
    "planned_expert_reclaim_mb",
    "physical_expert_reclaim_mb",
    "expert_host_backing_mb",
    "expert_resident_count",
    "expert_offloaded_count",
    "expert_slot_capacity_total",
    "expert_slot_rebind_count",
    "expert_materialize_count",
    "expert_topk_rewrite_count",
    "expert_core_hook_count",
    "expert_materialize_mb_total",
    "expert_materialize_ms",
    "expert_guard_pass",
    "expert_guard_reason",
    "kvc_host_backing_mb",
    "kvc_host_capacity_tokens",
    "kvc_host_used_tokens",
    "kvc_page_size",
    "kvc_offloaded_page_count",
    "kvc_resident_page_count",
    "kvc_reload_page_count_total",
    "kvc_evict_page_count_total",
    "kvc_page_alignment_violation_count",
    "kvc_resident_token_count",
    "kvc_offloaded_token_count",
    "kvc_residency_entry_count",
    "kvc_stale_entry_count",
    "kvc_evict_count_total",
    "kvc_reload_required_count",
    "kvc_reload_count_total",
    "kvc_reload_mb_total",
    "kvc_backup_ms",
    "kvc_reload_ms",
    "kvc_allocator_free_count",
    "kvc_allocator_available_before",
    "kvc_allocator_available_after",
    "kvc_req_to_token_rewrite_count",
    "kvc_physical_cycle_count",
    "kvc_physical_failure_count",
    "kvc_eviction_skipped_count",
    "kvc_layerwise_required_index_hit_count",
    "kvc_layerwise_required_index_scan_count",
    "kvc_layerwise_required_index_stale_count",
    "kvc_layerwise_required_selected_token_count",
    "kvc_layerwise_scheduler_deadline_reject_count",
    "kvc_layerwise_scheduler_dynamic_budget_count",
    "kvc_layerwise_cost_observation_count",
    "kvc_layerwise_reload_ewma_ms_per_mb",
    "kvc_layerwise_evict_ewma_ms_per_mb",
    "kvc_ready_before_use_ratio",
    "kvc_ready_before_use_count",
    "kvc_ready_use_check_count",
    "unified_residency_enabled",
    "resident_group_count",
    "resident_group_kvc_count",
    "resident_group_expert_count",
    "resident_group_resident_count",
    "resident_group_offloaded_count",
    "resident_group_recovering_count",
    "resident_group_recover_count",
    "resident_group_wait_count",
    "resident_group_state_error_count",
    "resident_group_last_error",
    "layerkv_copy_event_record_count",
    "layerkv_copy_event_wait_count",
    "layerkv_deadline_miss_count",
    "kvc_guard_pass",
    "kvc_guard_reason",
    "comparable",
    "comparability_reason",
    "stats_line_count",
    "stdout_path",
    "stderr_path",
    "result_path",
]


def _float_close(a: Any, b: float, tol: float = 1e-5) -> bool:
    try:
        return abs(float(a) - b) <= tol
    except Exception:
        return False


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


def validate_run(
    policy: str, target: float, page_size: int, stats: Dict[str, Any], returncode: int
) -> Tuple[bool, str]:
    reasons: List[str] = []
    if returncode != 0:
        reasons.append(f"process_returncode={returncode}")
    if not stats:
        reasons.append("missing_layerkv_stats")
        return False, ";".join(reasons)

    expected_fraction = EXPECTED_FRACTIONS[policy]
    if not _float_close(stats.get("policy_kvc_fraction"), expected_fraction):
        reasons.append(
            f"policy_fraction_mismatch expected={expected_fraction} got={stats.get('policy_kvc_fraction')}"
        )

    expected_effective = target * expected_fraction
    if not _float_close(
        stats.get("effective_kvc_reclaim_mb"), expected_effective, tol=1e-3
    ):
        reasons.append(
            f"effective_kvc_reclaim_mismatch expected={expected_effective} got={stats.get('effective_kvc_reclaim_mb')}"
        )

    if not bool(stats.get("layerkv_physical_kvc_supported", False)):
        reasons.append("physical_kvc_unsupported")
    if not bool(stats.get("kvc_guard_pass", False)):
        reasons.append(f"kvc_guard_failed:{stats.get('kvc_guard_reason')}")
    if int(stats.get("kvc_physical_failure_count", 0) or 0) != 0:
        reasons.append("kvc_physical_failure_count_nonzero")
    if int(stats.get("kvc_stale_entry_count", 0) or 0) != 0:
        reasons.append("kvc_stale_entry_count_nonzero")
    if int(stats.get("kvc_page_alignment_violation_count", 0) or 0) != 0:
        reasons.append("kvc_page_alignment_violation_count_nonzero")
    if int(stats.get("resident_group_state_error_count", 0) or 0) != 0:
        reasons.append(
            f"resident_group_state_error:{stats.get('resident_group_last_error')}"
        )
    if int(stats.get("kvc_page_size", 1) or 1) != page_size:
        reasons.append(
            f"kvc_page_size_mismatch expected={page_size} got={stats.get('kvc_page_size')}"
        )
    if int(stats.get("kvc_host_used_tokens", 0) or 0) != int(
        stats.get("kvc_offloaded_token_count", 0) or 0
    ):
        reasons.append("host_used_tokens_mismatch_offloaded_tokens")

    evict_count = int(stats.get("kvc_evict_count_total", 0) or 0)
    reload_count = int(stats.get("kvc_reload_count_total", 0) or 0)
    effective_mb = float(stats.get("effective_kvc_reclaim_mb", 0.0) or 0.0)
    kvc_backend = str(stats.get("layerkv_kvc_backend") or "token-slot")
    if target == 0 or expected_fraction == 0.0:
        if effective_mb != 0.0:
            reasons.append("zero_fraction_or_target_has_effective_reclaim")
        if evict_count != 0:
            reasons.append("zero_fraction_or_target_has_evict")
        if reload_count != 0:
            reasons.append("zero_fraction_or_target_has_reload")
    else:
        if effective_mb <= 0.0:
            reasons.append("positive_policy_has_no_effective_reclaim")
        if evict_count <= 0:
            reasons.append("positive_policy_has_no_physical_evict")
        if reload_count <= 0:
            reasons.append("positive_policy_has_no_reload")
        if evict_count % page_size != 0:
            reasons.append("evict_count_not_page_aligned")
        if reload_count % page_size != 0:
            reasons.append("reload_count_not_page_aligned")
        rewrite_count = int(stats.get("kvc_req_to_token_rewrite_count", 0) or 0)
        if kvc_backend != "per-layer-arena":
            if rewrite_count <= 0:
                reasons.append("positive_policy_has_no_req_to_token_rewrite")
            if rewrite_count % page_size != 0:
                reasons.append("req_to_token_rewrite_not_page_aligned")
        elif not bool(stats.get("layerkv_kvc_backend_ready", False)):
            reasons.append(
                f"per_layer_backend_not_ready:{stats.get('layerkv_kvc_backend_reason')}"
            )
        if float(stats.get("physical_kvc_reclaim_mb", 0.0) or 0.0) <= 0.0:
            reasons.append("positive_policy_has_no_physical_reclaim")

    return not reasons, ";".join(reasons)


def run_one(
    args: argparse.Namespace,
    policy: str,
    target: float,
    page_size: int,
    output_dir: Path,
) -> Dict[str, Any]:
    stamp = f"{policy.replace('-', '_')}_p{page_size}_{str(target).replace('.', 'p')}_{int(time.time() * 1000)}"
    result_path = output_dir / f"{stamp}.bench.jsonl"
    stdout_path = output_dir / f"{stamp}.stdout.log"
    stderr_path = output_dir / f"{stamp}.stderr.log"

    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_one_batch",
        "--model-path",
        args.model_path,
        "--load-format",
        "dummy",
        "--batch-size",
        str(args.batch_size),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--page-size",
        str(page_size),
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-layerkv",
        "--layerkv-mode",
        "kvc-only",
        "--layerkv-policy",
        policy,
        "--layerkv-target-reclaim-mb",
        str(target),
        "--layerkv-kvc-block-tokens",
        str(args.kvc_block_tokens),
        "--layerkv-kvc-backend",
        args.kvc_backend,
        "--layerkv-kvc-scheduler",
        args.scheduler,
        "--layerkv-debug-stats",
        "--result-filename",
        str(result_path),
        "--log-level",
        args.log_level,
    ]

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

    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout_s,
    )
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)

    all_stats = parse_layerkv_stats(proc.stdout + "\n" + proc.stderr)
    final_stats = all_stats[-1] if all_stats else {}
    valid, reason = validate_run(
        policy, target, page_size, final_stats, proc.returncode
    )

    row: Dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(final_stats)
    row.update(
        {
            "policy": policy,
            "target_reclaim_mb": target,
            "page_size": page_size,
            "returncode": proc.returncode,
            "valid": valid,
            "validation_reason": reason,
            "stats_line_count": len(all_stats),
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
    parser.add_argument("--output-dir", default="outputs/layerkv")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--targets", nargs="+", type=float, default=[0.0, 1.0, 4.0])
    parser.add_argument("--page-sizes", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--policies", nargs="+", default=POLICIES)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=3)
    parser.add_argument("--kvc-block-tokens", type=int, default=4)
    parser.add_argument(
        "--kvc-backend",
        choices=["token-slot", "virtual-arena", "per-layer-arena"],
        default="token-slot",
    )
    parser.add_argument(
        "--scheduler", choices=["sync", "async-deadline"], default="async-deadline"
    )
    parser.add_argument("--tmpdir", default="/data/wenyan/tmp")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--timeout-s", type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for page_size in args.page_sizes:
        for target in args.targets:
            for policy in args.policies:
                print(
                    f"[layerkv-kvc-validation] policy={policy} page_size={page_size} target={target}MB",
                    flush=True,
                )
                row = run_one(args, policy, target, page_size, output_dir)
                rows.append(row)
                print(
                    "[layerkv-kvc-validation] "
                    f"valid={row['valid']} evict={row.get('kvc_evict_count_total')} "
                    f"reload={row.get('kvc_reload_count_total')} reason={row['validation_reason']}",
                    flush=True,
                )

    csv_path = output_dir / "kvc_validation.csv"
    summary_path = output_dir / "kvc_validation_summary.json"
    write_csv(csv_path, rows)

    invalid = [row for row in rows if str(row.get("valid")) != "True"]
    summary = {
        "csv": str(csv_path),
        "total_runs": len(rows),
        "valid_runs": len(rows) - len(invalid),
        "invalid_runs": len(invalid),
        "all_valid": not invalid,
        "invalid": [
            {
                "policy": row.get("policy"),
                "target_reclaim_mb": row.get("target_reclaim_mb"),
                "reason": row.get("validation_reason"),
                "returncode": row.get("returncode"),
                "stdout_path": row.get("stdout_path"),
                "stderr_path": row.get("stderr_path"),
            }
            for row in invalid
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not invalid else 1


if __name__ == "__main__":
    raise SystemExit(main())
