#!/usr/bin/env python3
"""Validate LayerKV expert-offload hooks without requiring a large MoE model.

This exercises the SGLang-facing expert path:

* policy-to-KV/expert split accounting
* standard FusedMoE discovery
* fixed-capacity expert slot rebinding
* run_moe_core top-k rewrite
* CPU-backed expert materialization
* unsupported expert guard reporting
"""

from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import torch

from sglang.srt.layerkv.runtime import LayerKVConfig, LayerKVRuntime
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput

POLICIES = [
    "expert-first",
    "kv-first",
    "ratio-25-75",
    "ratio-50-50",
    "ratio-75-25",
    "layer-aware-joint-dp",
]


def _hotness_contains_expert(summary: Dict[str, Any], expert_id: int) -> bool:
    try:
        hotness = json.loads(str(summary["expert_hotness_topk_by_layer"]))
    except Exception:
        return False
    for layer_hotness in hotness.values():
        for mode in ("prefill", "decode"):
            for item in layer_hotness.get(mode, []):
                if item and int(item[0]) == int(expert_id):
                    return True
    return False


CSV_FIELDS = [
    "policy",
    "reclaim_limit_mb",
    "valid",
    "validation_reason",
    "policy_kvc_fraction",
    "policy_expert_fraction",
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
    "planner_version",
    "planner_used_hotness",
    "planner_fallback_reason",
    "planner_estimated_kvc_cost",
    "planner_estimated_expert_cost",
    "planner_selected_kvc_reclaim_mb",
    "planner_selected_expert_reclaim_mb",
    "planner_dp_infeasible_kvc_candidates",
    "planner_dp_infeasible_expert_candidates",
    "planner_dp_selected_total_cost",
    "planner_dp_selected_kvc_cost",
    "planner_dp_selected_expert_cost",
    "selected_kvc_tokens_by_layer",
    "selected_expert_evictions_by_layer",
    "expert_guard_pass",
    "expert_guard_reason",
    "layerkv_runtime_profile",
    "layerkv_physical_expert_supported",
    "layerkv_expert_layer_count",
    "full_policy_semantics_supported",
    "policy_semantics_reason",
    "comparable",
    "comparability_reason",
]


class UnquantizedFusedMoEMethod:
    pass


class UnsupportedQuantMethod:
    pass


class FakeFusedMoE(torch.nn.Module):
    def __init__(
        self,
        *,
        layer_id: int,
        num_experts: int = 8,
        top_k: int = 2,
        quant_method: Any = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.top_k = top_k
        self.moe_ep_size = 1
        self.num_experts = num_experts
        self.num_local_experts = num_experts
        self.quant_method = quant_method or UnquantizedFusedMoEMethod()
        self.moe_runner_config = SimpleNamespace(
            top_k=top_k,
            num_experts=num_experts,
            num_local_experts=num_experts,
        )
        self.dispatcher = SimpleNamespace(
            num_experts=num_experts,
            num_local_experts=num_experts,
            num_local_routed_experts=num_experts,
        )
        base = torch.arange(num_experts * 4, dtype=torch.float32).view(
            num_experts, 2, 2
        )
        self.w13_weight = torch.nn.Parameter(base.clone(), requires_grad=False)
        self.w2_weight = torch.nn.Parameter(base.clone() + 1000, requires_grad=False)
        self.last_topk_ids = None

    def forward(self, hidden_states: torch.Tensor, topk_output: StandardTopKOutput):
        dispatch_output = StandardDispatchOutput(hidden_states, None, topk_output)
        return self.run_moe_core(dispatch_output)

    def run_moe_core(self, dispatch_output: StandardDispatchOutput):
        self.last_topk_ids = dispatch_output.topk_output.topk_ids.detach().clone()
        return dispatch_output.hidden_states


class FakeModel(torch.nn.Module):
    def __init__(self, layers: Iterable[torch.nn.Module]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


class FakeRunner:
    device = "cpu"
    token_to_kv_pool = None
    token_to_kv_pool_allocator = None
    req_to_token_pool = None

    def __init__(self, layers: Iterable[torch.nn.Module]):
        self.model = FakeModel(layers)


def _runtime(
    policy: str, reclaim_limit_mb: float, runtime_profile: str
) -> LayerKVRuntime:
    args = Namespace(
        enable_layerkv=True,
        layerkv_mode="kvc-expert",
        layerkv_policy=policy,
        layerkv_reclaim_limit_mb=reclaim_limit_mb,
        layerkv_kvc_block_tokens=16,
        layerkv_kvc_scheduler="async-deadline",
        layerkv_runtime_profile=runtime_profile,
        layerkv_debug_stats=True,
        layerkv_disallow_destructive_fallback=True,
        layerkv_expert_backing_cache_mb=0.0,
        layerkv_expert_cpu_backing_mode="none",
        layerkv_expert_install_layers_per_step=999,
        layerkv_expert_install_budget_mb=999.0,
        layerkv_expert_install_target_steps=0,
    )
    return LayerKVRuntime(LayerKVConfig.from_server_args(args))


def _exercise_policy(
    policy: str, reclaim_limit_mb: float, runtime_profile: str
) -> Dict[str, Any]:
    runner = FakeRunner(
        [
            FakeFusedMoE(layer_id=0),
            FakeFusedMoE(layer_id=1),
        ]
    )
    rt = _runtime(policy, reclaim_limit_mb, runtime_profile)
    rt.install_on_runner(runner)
    topk = StandardTopKOutput(
        topk_weights=torch.ones((2, 2), dtype=torch.float32),
        topk_ids=torch.tensor([[6, 7], [6, 7]], dtype=torch.int32),
        router_logits=torch.zeros((2, 8), dtype=torch.float32),
    )
    # First decode pass records routing hotness. The runtime intentionally waits
    # for one decode sample before physically shrinking expert slots.
    rt.on_forward_begin(mode="decode", forward_batch=SimpleNamespace())
    for layer in runner.model.layers:
        layer(torch.zeros((2, 2), dtype=torch.float32), topk)

    # Second decode pass applies the expert plan and exercises materialization.
    rt.on_forward_begin(mode="decode", forward_batch=SimpleNamespace())
    for layer in runner.model.layers:
        state = rt._expert_layers.get(int(layer.layer_id))
        offloaded = [
            expert_id
            for expert_id in range(layer.num_experts)
            if state is not None and expert_id not in state.logical_to_slot
        ]
        expert_id = int(offloaded[0]) if offloaded else 0
        materialize_topk = StandardTopKOutput(
            topk_weights=torch.ones((2, 2), dtype=torch.float32),
            topk_ids=torch.tensor(
                [[expert_id, expert_id], [expert_id, expert_id]], dtype=torch.int32
            ),
            router_logits=torch.zeros((2, 8), dtype=torch.float32),
        )
        layer(torch.zeros((2, 2), dtype=torch.float32), materialize_topk)

    summary = rt.summary()
    reasons: List[str] = []
    expert_fraction = float(summary["policy_expert_fraction"])
    if expert_fraction > 0.0:
        if not bool(summary["layerkv_physical_expert_supported"]):
            reasons.append("physical_expert_unsupported")
        if int(summary["layerkv_expert_layer_count"]) != 2:
            reasons.append("missing_expert_layers")
        if int(summary["expert_slot_rebind_count"]) != 2:
            reasons.append("slot_rebind_count_mismatch")
        if int(summary["expert_materialize_count"]) <= 0:
            reasons.append("no_expert_materialize")
        if int(summary["expert_topk_rewrite_count"]) < 2:
            reasons.append("topk_rewrite_count_mismatch")
        if int(summary["expert_core_hook_count"]) < 2:
            reasons.append("core_hook_count_mismatch")
        if int(summary["expert_call_count_total"]) < 8:
            reasons.append("expert_hotness_count_mismatch")
        if int(summary["expert_decode_call_count_total"]) < 8:
            reasons.append("expert_decode_hotness_count_mismatch")
        if not bool(summary["expert_hotness_observed"]):
            reasons.append("expert_hotness_not_observed")
        if not _hotness_contains_expert(summary, 6):
            reasons.append("expert_hotness_missing_logical_id")
        if float(summary["physical_expert_reclaim_mb"]) <= 0.0:
            reasons.append("no_physical_expert_reclaim")
        if float(summary["physical_expert_reclaim_mb"]) + 1e-9 < float(
            summary["planned_expert_reclaim_mb"]
        ):
            reasons.append("insufficient_physical_expert_reclaim")
        if float(summary["expert_host_backing_mb"]) >= 0.000488:
            reasons.append("cpu_backing_should_be_sparse")
        for layer in runner.model.layers:
            if layer.last_topk_ids is None:
                reasons.append(f"layer{layer.layer_id}_not_called")
                continue
            if int(layer.last_topk_ids.max().item()) >= int(layer.w13_weight.shape[0]):
                reasons.append(f"layer{layer.layer_id}_topk_not_rewritten")
    else:
        if int(summary["expert_slot_rebind_count"]) != 0:
            reasons.append("expert_first_should_not_rebind_slots")
        if int(summary["expert_materialize_count"]) != 0:
            reasons.append("expert_first_should_not_materialize")

    if not bool(summary["expert_guard_pass"]):
        reasons.append(f"expert_guard_failed:{summary['expert_guard_reason']}")

    row = {field: summary.get(field, "") for field in CSV_FIELDS}
    row.update(
        {
            "policy": policy,
            "reclaim_limit_mb": reclaim_limit_mb,
            "valid": not reasons,
            "validation_reason": ";".join(reasons),
        }
    )
    return row


def _exercise_unsupported(
    reclaim_limit_mb: float, runtime_profile: str
) -> Dict[str, Any]:
    runner = FakeRunner(
        [FakeFusedMoE(layer_id=0, quant_method=UnsupportedQuantMethod())]
    )
    rt = _runtime("kv-first", reclaim_limit_mb, runtime_profile)
    rt.install_on_runner(runner)
    rt.on_forward_begin(mode="decode", forward_batch=SimpleNamespace())
    summary = rt.summary()
    reasons = []
    if bool(summary["layerkv_physical_expert_supported"]):
        reasons.append("unsupported_quant_marked_supported")
    if bool(summary["full_policy_semantics_supported"]):
        reasons.append("unsupported_quant_marked_full_semantics")
    if "unsupported quant_method" not in str(summary["expert_guard_reason"]):
        reasons.append("missing_unsupported_quant_reason")
    row = {field: summary.get(field, "") for field in CSV_FIELDS}
    row.update(
        {
            "policy": "kv-first-unsupported-quant",
            "reclaim_limit_mb": reclaim_limit_mb,
            "valid": not reasons,
            "validation_reason": ";".join(reasons),
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
    parser.add_argument("--output-dir", default="outputs/layerkv")
    parser.add_argument("--reclaim-limit-mb", type=float, default=0.00035)
    parser.add_argument(
        "--runtime-profile",
        choices=["simple", "optimized"],
        default="optimized",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _exercise_policy(policy, args.reclaim_limit_mb, args.runtime_profile)
        for policy in POLICIES
    ]
    rows.append(_exercise_unsupported(args.reclaim_limit_mb, args.runtime_profile))

    csv_path = output_dir / "expert_validation.csv"
    summary_path = output_dir / "expert_validation_summary.json"
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
                "reason": row.get("validation_reason"),
            }
            for row in invalid
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not invalid else 1


if __name__ == "__main__":
    raise SystemExit(main())
