"""LayerKV LayerKVExpertPolicyMixin implementation."""

from __future__ import annotations

import functools
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

if __package__:
    pass
else:  # pragma: no cover - direct file-loading smoke tests.
    pass

logger = logging.getLogger(__name__)

try:
    from sglang.jit_kernel.layerkv_expert_remap import layerkv_expert_remap
except Exception:  # pragma: no cover - optional JIT helper.
    layerkv_expert_remap = None


class LayerKVExpertPolicyMixin:
    def _discover_expert_support(self, runner: Any) -> None:
        model = getattr(runner, "model", None)
        if model is None:
            return
        if self.config.mode != "kvc-expert":
            return
        supported = []
        unsupported_reasons = []
        for module in model.modules():
            if not self._is_supported_fused_moe(module):
                continue
            layer_id = int(getattr(module, "layer_id", len(supported)))
            supported.append((layer_id, module))
        self._expert_modules = supported
        if not supported:
            self.physical_expert_supported = False
            if not self.unsupported_reason:
                self.unsupported_reason = "no supported standard FusedMoE layers found"
            return
        if not self.config.expert_forward_hooks:
            self.physical_expert_supported = False
            if not self.unsupported_reason:
                self.unsupported_reason = "expert forward hooks disabled"
            return
        for layer_id, module in supported:
            reason = self._expert_layer_unsupported_reason(module)
            if reason:
                unsupported_reasons.append(f"layer{layer_id}:{reason}")
        if unsupported_reasons:
            self.physical_expert_supported = False
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = ";".join(unsupported_reasons[:8])
            if not self.unsupported_reason:
                self.unsupported_reason = self.stats.expert_guard_reason
            return
        self.physical_expert_supported = True
        for layer_id, module in supported:
            self._install_expert_hotness_probe(module, layer_id)
        if self.config.expert_collector_only:
            self.physical_expert_supported = False
            if not self.unsupported_reason:
                self.unsupported_reason = "expert collector-only mode"
            return
        if self.config.expert_cpu_backing_mode in ("all", "selected"):
            self._preload_all_expert_cpu_backing()

    def _preload_all_expert_cpu_backing(self) -> None:
        if self._expert_global_cpu_backing:
            return
        t0 = time.perf_counter()
        copied = 0
        copied_bytes = 0
        selected = self.config.expert_cpu_backing_mode == "selected"
        modules = self._expert_modules
        if selected:
            modules = [
                (layer, module)
                for layer, module in modules
                if layer == self.config.shared_expert_layer
            ]
            if len(modules) != 1:
                raise ValueError(
                    "selected CPU backing requires exactly one discovered shared expert layer"
                )
        for layer_id, module in modules:
            param_names = self._expert_param_names(module)
            full_num_experts = int(module.w13_weight.data.shape[0])
            copied_params = self._copy_experts_to_cpu_for_install_batched(
                module,
                param_names,
                list(range(full_num_experts)),
                layer_id=int(layer_id),
                account_host_backing=False,
                pin_memory=selected,
                reason="install_preload",
            )
            for expert_id, params in copied_params.items():
                self._expert_global_cpu_backing[(int(layer_id), int(expert_id))] = (
                    params
                )
                copied += 1
                copied_bytes += self._expert_backing_bytes(params)
        self._expert_global_cpu_backing_bytes = copied_bytes
        self.stats.expert_cpu_backing_preload_count = copied
        self.stats.expert_cpu_backing_preload_ms += (time.perf_counter() - t0) * 1000.0
        self._refresh_expert_host_backing_stat()

    def _install_expert_hotness_probe(self, module: Any, layer_id: int) -> None:
        if getattr(module, "_layerkv_expert_wrapped", False):
            return
        if getattr(module, "_layerkv_hotness_wrapped", False):
            return
        full_num_experts = int(module.w13_weight.data.shape[0])
        orig_run_moe_core = module.run_moe_core
        native_graph = None
        if getattr(self.config, "native_moe_graph_max_batch_size", 0):
            from .native_moe_graph import NativeMoEGraph, resident_core_eligible

            if resident_core_eligible(
                module, layer_id, self.config.shared_expert_layer
            ):
                native_graph = NativeMoEGraph(
                    orig_run_moe_core,
                    lambda: (module.w13_weight, module.w2_weight),
                    max_batch_size=self.config.native_moe_graph_max_batch_size,
                )
                self._native_moe_graphs[layer_id] = native_graph

        @functools.wraps(orig_run_moe_core)
        def wrapped_run_moe_core(dispatch_output: Any, *args, **kwargs):
            topk_output = getattr(dispatch_output, "topk_output", None)
            topk_ids = getattr(topk_output, "topk_ids", None)
            if topk_ids is not None and self._should_collect_expert_hotness_layer(
                layer_id
            ):
                self._record_expert_hotness_for_layer(
                    layer_id=layer_id,
                    full_num_experts=full_num_experts,
                    topk_ids=topk_ids,
                )
            elif topk_ids is not None:
                self._expert_hotness_sample_skip_pending += 1
            if (
                native_graph is not None
                and self._current_forward_mode == "decode"
                and layer_id not in self._expert_layers
                and not getattr(module, "_layerkv_expert_wrapped", False)
                and not args
                and not kwargs
            ):
                return native_graph(dispatch_output)
            return orig_run_moe_core(dispatch_output, *args, **kwargs)

        module._layerkv_hotness_orig_run_moe_core = orig_run_moe_core
        module._layerkv_hotness_wrapped = True
        module.run_moe_core = wrapped_run_moe_core

    def _is_supported_fused_moe(self, module: Any) -> bool:
        return (
            hasattr(module, "w13_weight")
            and hasattr(module, "w2_weight")
            and hasattr(module, "forward")
            and hasattr(module, "run_moe_core")
            and hasattr(module, "num_local_experts")
            and hasattr(module, "moe_runner_config")
        )

    def _expert_layer_unsupported_reason(self, module: Any) -> str:
        if int(getattr(module, "moe_ep_size", 1) or 1) != 1:
            return "expert offload v1 supports moe_ep_size=1 only"
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            return "missing quant_method"
        if quant_method.__class__.__name__ != "UnquantizedFusedMoEMethod":
            return f"unsupported quant_method={quant_method.__class__.__name__}"
        w13 = getattr(module, "w13_weight", None)
        w2 = getattr(module, "w2_weight", None)
        if w13 is None or w2 is None or w13.data.dim() != 3 or w2.data.dim() != 3:
            return "expected 3D w13_weight/w2_weight"
        if int(w13.data.shape[0]) != int(w2.data.shape[0]):
            return "w13/w2 expert count mismatch"
        if not w13.data.is_cuda and getattr(self._runner, "device", None) == "cuda":
            return "expert weights are not on CUDA"
        return ""

    def _avg_prefix_len(self, forward_batch: Any) -> float:
        pairs = (
            list(self._current_forward_req_lens)
            if forward_batch is not None
            and self._current_forward_req_lens_batch_id == id(forward_batch)
            else self._batch_req_indices_and_lens(forward_batch)
        )
        if not pairs:
            return 0.0
        return sum(max(0, seq_len - 1) for _, seq_len in pairs) / float(len(pairs))

    def _policy_fractions(self, forward_batch: Any) -> Tuple[float, float, bool, str]:
        policy = self.config.policy
        if policy == "coresid":
            policy = "layer-aware-joint-dp"
        if policy == "none":
            return 0.0, 0.0, True, "policy=none disables LayerKV physical reclaim"
        if self.config.expert_collector_only and self.config.mode == "kvc-expert":
            return 1.0, 0.0, True, "expert_collector_only"
        if self.config.mode == "kvc-only":
            if policy in (
                "expert-first",
                "kv-first",
                "ratio-75-25",
                "ratio-50-50",
                "ratio-25-75",
                "layer-aware-joint",
                "layer-aware-joint-dp",
                "coresid",
            ):
                return 1.0, 0.0, True, ""
            return 0.0, 0.0, False, f"unknown LayerKV policy: {policy}"
        if policy == "expert-first":
            return 1.0, 0.0, True, ""
        if policy == "kv-first":
            return (
                0.0,
                1.0,
                self.physical_expert_supported,
                self._expert_support_reason(),
            )
        if policy == "ratio-75-25":
            return (
                0.75,
                0.25,
                self.physical_expert_supported,
                self._expert_support_reason(),
            )
        if policy == "ratio-50-50":
            return (
                0.50,
                0.50,
                self.physical_expert_supported,
                self._expert_support_reason(),
            )
        if policy == "ratio-25-75":
            return (
                0.25,
                0.75,
                self.physical_expert_supported,
                self._expert_support_reason(),
            )
        if policy in ("layer-aware-joint", "layer-aware-joint-dp"):
            target_bucket_mb: Optional[int] = None
            decode_bucket: Optional[int] = None
            if self.config.dynamic_pressure_from_kvc:
                target_bucket_mb = self._joint_policy_cache_bucket(
                    self._refresh_reclaim_target_stats(forward_batch)
                )
                decode_bucket = self._joint_policy_decode_bucket()
            if (
                self._cached_policy_fractions is not None
                and self._cached_policy_hotness_version == self._expert_hotness_version
                and (
                    not self.config.dynamic_pressure_from_kvc
                    or (
                        self._cached_policy_target_bucket_mb == target_bucket_mb
                        and self._cached_policy_decode_bucket == decode_bucket
                    )
                )
            ):
                (
                    kvc_fraction,
                    expert_fraction,
                    full_supported,
                    reason,
                    used_hotness,
                    fallback_reason,
                    kvc_cost,
                    expert_cost,
                ) = self._cached_policy_fractions
                self._set_joint_planner_choice(
                    kvc_fraction=kvc_fraction,
                    expert_fraction=expert_fraction,
                    used_hotness=used_hotness,
                    fallback_reason=fallback_reason,
                    kvc_cost=kvc_cost,
                    expert_cost=expert_cost,
                )
                self.stats.planner_cache_hit_count += 1
                self._update_planned_kvc_tokens_from_fraction(
                    kvc_fraction, forward_batch
                )
                return kvc_fraction, expert_fraction, full_supported, reason
            self.stats.planner_cache_miss_count += 1
            return self._joint_policy_fractions(forward_batch)
        return 0.0, 0.0, False, f"unknown LayerKV policy: {policy}"

    def _joint_policy_fractions(
        self, forward_batch: Any
    ) -> Tuple[float, float, bool, str]:
        self.stats.planner_apply_count += 1
        target_mb = self._refresh_reclaim_target_stats(forward_batch)
        self.stats.planner_version = "deadline-dp-v2-layerwise-benefit"
        if not self.physical_expert_supported:
            reason = self._expert_support_reason()
            self._set_joint_planner_choice(
                kvc_fraction=1.0,
                expert_fraction=0.0,
                used_hotness=False,
                fallback_reason=reason,
                kvc_cost=0.0,
                expert_cost=1.0e30,
            )
            return 1.0, 0.0, False, reason
        if self.config.dynamic_pressure_from_kvc and target_mb > 1e-3:
            reason = "dynamic_kvc_pressure_requires_kvc_reclaim"
            kvc_cost = self._estimate_planner_kvc_reclaim_cost(target_mb, forward_batch)
            self._set_joint_planner_choice(
                kvc_fraction=1.0,
                expert_fraction=0.0,
                used_hotness=False,
                fallback_reason=reason,
                kvc_cost=kvc_cost,
                expert_cost=0.0,
            )
            self._update_planned_kvc_tokens_from_fraction(1.0, forward_batch)
            return 1.0, 0.0, True, reason

        if target_mb > 1e-3:
            self._ensure_expert_hotness_cpu_view(mode=None)
        has_hotness = self._has_expert_hotness()
        if not has_hotness:
            kvc_fraction = self._context_heuristic_kvc_fraction(forward_batch)
            expert_fraction = 1.0 - kvc_fraction
            self._set_joint_planner_choice(
                kvc_fraction=kvc_fraction,
                expert_fraction=expert_fraction,
                used_hotness=False,
                fallback_reason="no_hotness",
                kvc_cost=self._estimate_planner_kvc_reclaim_cost(
                    target_mb * kvc_fraction, forward_batch
                ),
                expert_cost=0.0,
            )
            return kvc_fraction, expert_fraction, True, "planner_fallback=no_hotness"

        kvc_fraction, expert_fraction, kvc_cost, expert_cost = (
            self._solve_joint_dp_reclaim_split(target_mb, forward_batch)
        )
        self._set_joint_planner_choice(
            kvc_fraction=kvc_fraction,
            expert_fraction=expert_fraction,
            used_hotness=True,
            fallback_reason="",
            kvc_cost=kvc_cost,
            expert_cost=expert_cost,
        )
        if self._has_decode_expert_hotness() or self._current_forward_mode == "decode":
            self._cached_policy_fractions = (
                kvc_fraction,
                expert_fraction,
                True,
                "",
                True,
                "",
                kvc_cost,
                expert_cost,
            )
            if self.config.dynamic_pressure_from_kvc:
                self._cached_policy_target_bucket_mb = self._joint_policy_cache_bucket(
                    target_mb
                )
                self._cached_policy_decode_bucket = self._joint_policy_decode_bucket()
            else:
                self._cached_policy_target_bucket_mb = None
                self._cached_policy_decode_bucket = None
            self._cached_policy_hotness_version = self._expert_hotness_version
        return kvc_fraction, expert_fraction, True, ""

    def _update_planned_kvc_tokens_from_fraction(
        self, kvc_fraction: float, forward_batch: Any
    ) -> None:
        target_mb = max(0.0, float(self.stats.effective_reclaim_target_mb))
        reclaim_mb = target_mb * max(0.0, float(kvc_fraction))
        if reclaim_mb <= 0.0:
            self._planned_kvc_token_target = 0
            self._planned_kvc_tokens_by_layer = {}
            self.stats.selected_kvc_tokens_by_layer = ""
            return
        bytes_per_token = max(1, self._bytes_per_kvc_token_per_layer())
        total_tokens = int(reclaim_mb * 1024.0 * 1024.0 / float(bytes_per_token))
        layer_ids, max_tokens_per_layer, block_tokens = (
            self._layer_aware_kvc_plan_context(forward_batch)
        )
        plan = self._build_layer_aware_kvc_token_plan_from_context(
            total_tokens,
            layer_ids=layer_ids,
            max_tokens_per_layer=max_tokens_per_layer,
            block_tokens=block_tokens,
            avg_prefix=self._avg_prefix_len(forward_batch),
            batch_size=max(1, len(self._batch_req_indices_and_lens(forward_batch))),
        )
        self._planned_kvc_token_target = int(sum(plan.values()))
        self._planned_kvc_tokens_by_layer = plan
        self.stats.selected_kvc_tokens_by_layer = self._kvc_tokens_by_layer_json(plan)

    def _solve_joint_dp_reclaim_split(
        self, target_mb: float, forward_batch: Any
    ) -> Tuple[float, float, float, float]:
        if target_mb <= 0.0:
            if self.config.policy == "coresid" and (
                self._planned_expert_slot_capacities_by_layer
                or self._expert_plan_applied
                or self._expert_install_queue
            ):
                self._sync_coresid_expert_plan_stats(context="zero_pressure")
                self._planned_kvc_token_target = 0
                self._planned_kvc_tokens_by_layer = {}
                self.stats.selected_kvc_tokens_by_layer = ""
                return 0.0, 0.0, 0.0, 0.0
            self.stats.planner_dp_candidate_count = 1
            self.stats.planner_dp_selected_kvc_candidates = 0
            self.stats.planner_dp_selected_expert_candidates = 0
            self.stats.planner_dp_infeasible_kvc_candidates = 0
            self.stats.planner_dp_infeasible_expert_candidates = 0
            self.stats.planner_dp_selected_total_cost = 0.0
            self.stats.planner_dp_selected_kvc_cost = 0.0
            self.stats.planner_dp_selected_expert_cost = 0.0
            self.stats.planner_dp_table_build_ms = 0.0
            self.stats.planner_dp_lookup_ms = 0.0
            self.stats.planner_dp_kvc_candidate_count = 0
            self.stats.planner_dp_expert_candidate_count = 0
            self.stats.planner_estimated_kvc_cost = 0.0
            self.stats.planner_estimated_expert_cost = 0.0
            self.stats.planner_estimated_expert_churn_count = 0.0
            self.stats.planner_estimated_expert_churn_mb = 0.0
            self.stats.planner_estimated_expert_install_mb = 0.0
            self.stats.planner_estimated_kvc_controller_cost = 0.0
            self.stats.planner_estimated_kvc_overlap_ms = 0.0
            self.stats.planner_estimated_kvc_exposed_ms = 0.0
            self.stats.planner_estimated_kvc_capacity_benefit = 0.0
            self.stats.planner_estimated_expert_backing_miss_cost = 0.0
            self.stats.planner_estimated_expert_materialize_cost = 0.0
            self.stats.planner_selected_kvc_reclaim_mb = 0.0
            self.stats.planner_selected_expert_reclaim_mb = 0.0
            self.stats.selected_kvc_tokens_by_layer = ""
            self.stats.selected_expert_evictions_by_layer = ""
            self.stats.selected_expert_capacity_by_layer = ""
            self.stats.selected_expert_cost_by_layer = ""
            self._planned_expert_slot_capacities_by_layer = {}
            self._planned_expert_cost_by_layer = {}
            self._planned_expert_target_mb = 0.0
            self._planner_target_high_watermark_mb = 0.0
            self._planned_kvc_token_target = 0
            self._planned_kvc_tokens_by_layer = {}
            return 0.0, 0.0, 0.0, 0.0
        available_kvc_mb = max(0.0, self.stats.available_kvc_reclaim_mb)
        block_tokens = max(
            self._page_size,
            self._align_tokens_up(int(self.config.kvc_block_tokens)),
        )
        kvc_token_bytes = (
            self._bytes_per_kvc_token_per_layer()
            if self.config.kvc_backend == "per-layer-arena"
            else self._bytes_per_token_all_layers
        )
        block_mb = block_tokens * max(0, kvc_token_bytes) / float(1024 * 1024)
        kvc_points: Set[float] = {0.0}
        kvc_token_points: Dict[float, int] = {0.0: 0}
        limit = min(target_mb, available_kvc_mb)
        infeasible_kvc = 0
        if block_mb > 0.0 and limit > 0.0:
            max_blocks = int(limit // block_mb)
            if max_blocks <= 0:
                infeasible_kvc += 1
            else:
                # Keep the logical unit at layerkv_kvc_block_tokens, but cap
                # decision points to avoid planner overhead regressing decode.
                max_candidates = 32
                stride = max(1, int(math.ceil(max_blocks / float(max_candidates))))
                for blocks in range(1, max_blocks + 1, stride):
                    tokens = blocks * block_tokens
                    mb = min(limit, tokens * kvc_token_bytes / float(1024 * 1024))
                    kvc_points.add(mb)
                    kvc_token_points[mb] = tokens
                full_tokens = max_blocks * block_tokens
                full_mb = min(limit, full_tokens * kvc_token_bytes / float(1024 * 1024))
                kvc_points.add(full_mb)
                kvc_token_points[full_mb] = full_tokens
        elif limit > 0.0:
            infeasible_kvc += 1
        self.stats.planner_dp_table_build_ms = 0.0
        self.stats.planner_dp_lookup_ms = 0.0
        table_build_t0 = time.perf_counter()
        expert_cost_cache: Dict[
            float, Tuple[float, float, float, float, float, float]
        ] = {}
        expert_plan_cache: Dict[float, Tuple[Dict[int, int], Dict[int, float]]] = {}
        layerwise_expert_plan = self._layerwise_expert_plan_enabled()
        coresid_expert_plan_inputs = (
            self._build_coresid_expert_plan_inputs(forward_batch)
            if layerwise_expert_plan
            else None
        )
        if layerwise_expert_plan:
            expert_layer_items: List[Tuple[int, int, int]] = []
            expert_candidates: List[Tuple[float, int]] = []
        else:
            expert_layer_items, expert_candidates = self._build_expert_cost_inputs()
        if layerwise_expert_plan and coresid_expert_plan_inputs is not None:
            expert_prefix_candidates = coresid_expert_plan_inputs[3]
        else:
            expert_prefix_candidates = expert_candidates
        expert_prefix_table = (
            {}
            if layerwise_expert_plan
            else self._build_expert_prefix_table(expert_prefix_candidates)
        )
        expert_layerwise_table = (
            self._build_coresid_expert_layerwise_plan_table(
                coresid_expert_plan_inputs,
                max_reclaim_mb=target_mb,
            )
            if layerwise_expert_plan and coresid_expert_plan_inputs is not None
            else None
        )
        (
            kvc_layer_ids,
            kvc_max_tokens_per_layer,
            kvc_block_tokens_for_plan,
        ) = self._layer_aware_kvc_plan_context(forward_batch)
        kvc_avg_prefix = max(1.0, self._avg_prefix_len(forward_batch))
        current_batch_size = len(self._batch_req_indices_and_lens(forward_batch))
        kvc_batch_size = max(
            1,
            int(current_batch_size or 0),
            int(getattr(self.stats, "observed_batch_size", 0) or 0),
        )
        kvc_recovery_steps = self._expected_kvc_recovery_steps()
        kvc_cost_cache: Dict[float, Tuple[float, int]] = {}
        kvc_controller_cost_cache: Dict[float, float] = {}
        with self._profile("profile_planner_dp_kvc_cost_ms"):
            vectorized_kvc_costs = self._estimate_arena_kvc_candidate_costs_vectorized(
                sorted(kvc_points),
                kvc_token_points,
                layer_ids=kvc_layer_ids,
                max_tokens_per_layer=kvc_max_tokens_per_layer,
                block_tokens=kvc_block_tokens_for_plan,
                avg_prefix=kvc_avg_prefix,
                batch_size=kvc_batch_size,
                recovery_steps=kvc_recovery_steps,
            )
        for point, (cost, tokens, controller_cost) in vectorized_kvc_costs.items():
            kvc_cost_cache[round(max(0.0, float(point)), 6)] = (cost, tokens)
            kvc_controller_cost_cache[round(max(0.0, float(point)), 6)] = (
                controller_cost
            )
        self.stats.planner_dp_table_build_ms = (
            time.perf_counter() - table_build_t0
        ) * 1000.0
        self.stats.planner_dp_kvc_candidate_count = len(kvc_points)

        def cached_kvc_cost(kvc_mb: float) -> Tuple[float, int]:
            lookup_t0 = time.perf_counter()
            key = round(max(0.0, float(kvc_mb)), 6)
            cached = kvc_cost_cache.get(key)
            if cached is not None:
                self.stats.planner_dp_lookup_ms += (
                    time.perf_counter() - lookup_t0
                ) * 1000.0
                return cached
            tokens = int(kvc_token_points.get(kvc_mb, 0))
            plan = self._build_layer_aware_kvc_token_plan_from_context(
                tokens,
                layer_ids=kvc_layer_ids,
                max_tokens_per_layer=kvc_max_tokens_per_layer,
                block_tokens=kvc_block_tokens_for_plan,
                avg_prefix=kvc_avg_prefix,
                batch_size=kvc_batch_size,
            )
            value = self._estimate_arena_kvc_reclaim_cost_from_plan(
                plan,
                layer_ids=kvc_layer_ids,
                avg_prefix=kvc_avg_prefix,
                batch_size=kvc_batch_size,
                recovery_steps=kvc_recovery_steps,
            )
            cached = (value, int(sum(plan.values())))
            kvc_cost_cache[key] = cached
            self.stats.planner_dp_lookup_ms += (
                time.perf_counter() - lookup_t0
            ) * 1000.0
            return cached

        # This is the value of retaining KVC capacity instead of moving that
        # capacity to CPU.  It is measured in the same modeled critical-path
        # units as ``kvc_cost`` and already reflects the current context,
        # batch, overlap window and decode recovery horizon.  Using the
        # maximum feasible KVC point as the reference makes the expert choice
        # an explicit "expert transfer cost minus saved KVC cost" decision.
        kvc_baseline_point = 0.0
        if limit > 0.0:
            feasible_points = [
                float(point)
                for point in kvc_points
                if float(point) <= float(limit) + 1e-6
            ]
            if feasible_points:
                kvc_baseline_point = max(feasible_points)
        kvc_baseline_cost = 0.0
        if kvc_baseline_point > 0.0:
            kvc_baseline_cost, _baseline_tokens = cached_kvc_cost(kvc_baseline_point)
            if kvc_baseline_cost >= 1.0e29:
                kvc_baseline_cost = 0.0
        has_kvc_benefit_reference = kvc_baseline_point > 0.0

        def cached_expert_cost(
            expert_mb: float,
        ) -> Tuple[float, float, float, float, float, float]:
            lookup_t0 = time.perf_counter()
            key = round(max(0.0, float(expert_mb)), 6)
            cached = expert_cost_cache.get(key)
            if cached is not None:
                self.stats.planner_dp_lookup_ms += (
                    time.perf_counter() - lookup_t0
                ) * 1000.0
                return cached
            with self._profile("profile_planner_dp_expert_cost_ms"):
                if layerwise_expert_plan:
                    (
                        value,
                        churn_count,
                        churn_mb,
                        install_mb,
                        backing_miss_cost,
                        materialize_cost,
                        capacities,
                        layer_costs,
                    ) = self._lookup_coresid_expert_layerwise_cost(
                        key, expert_layerwise_table
                    )
                    self.stats.planner_estimated_expert_churn_count = churn_count
                    self.stats.planner_estimated_expert_churn_mb = churn_mb
                    self.stats.planner_estimated_expert_install_mb = install_mb
                    expert_plan_cache[key] = (capacities, layer_costs)
                else:
                    (
                        value,
                        churn_count,
                        churn_mb,
                        install_mb,
                        backing_miss_cost,
                        materialize_cost,
                        _capacities,
                        _layer_costs,
                    ) = self._lookup_expert_prefix_cost(
                        key,
                        expert_prefix_candidates,
                        expert_prefix_table,
                        include_churn_cost=False,
                    )
                    self.stats.planner_estimated_expert_churn_count = churn_count
                    self.stats.planner_estimated_expert_churn_mb = churn_mb
                    self.stats.planner_estimated_expert_install_mb = install_mb
            cached = (
                value,
                float(churn_count),
                float(churn_mb),
                float(install_mb),
                float(backing_miss_cost),
                float(materialize_cost),
            )
            expert_cost_cache[key] = cached
            self.stats.planner_dp_lookup_ms += (
                time.perf_counter() - lookup_t0
            ) * 1000.0
            return cached

        (
            expert_only_cost,
            expert_only_churn,
            expert_only_churn_mb,
            expert_only_install,
            expert_only_backing_cost,
            expert_only_materialize_cost,
        ) = cached_expert_cost(target_mb)
        expert_only_saved_kvc_cost = (
            kvc_baseline_cost if has_kvc_benefit_reference else 0.0
        )
        best: Optional[
            Tuple[float, float, float, float, float, float, float, float]
        ] = (
            float(expert_only_cost - expert_only_saved_kvc_cost),
            float(expert_only_cost),
            float(-max(0.0, expert_only_saved_kvc_cost)),
            0.0,
            1.0,
            0.0,
            float(expert_only_cost),
            float(expert_only_saved_kvc_cost),
        )
        best_expert_stats = (
            expert_only_churn,
            expert_only_churn_mb,
            expert_only_install,
            expert_only_backing_cost,
            expert_only_materialize_cost,
        )
        best_kvc_tokens = 0
        best_kvc_controller_cost = 0.0
        best_kvc_saved_cost = kvc_baseline_cost if has_kvc_benefit_reference else 0.0
        infeasible_expert = 0
        if expert_only_cost >= 1.0e29:
            infeasible_expert += 1
        with self._profile("profile_planner_dp_ms"):
            for kvc_mb in sorted(kvc_points):
                if kvc_mb <= 0.0:
                    continue
                with self._profile("profile_planner_dp_candidate_eval_ms"):
                    kvc_cost, kvc_tokens = cached_kvc_cost(kvc_mb)
                    actual_kvc_mb = (
                        float(kvc_tokens) * float(kvc_token_bytes) / float(1024 * 1024)
                    )
                    actual_kvc_mb = min(float(target_mb), max(0.0, actual_kvc_mb))
                    if actual_kvc_mb <= 0.0:
                        infeasible_kvc += 1
                        continue
                    expert_mb = max(0.0, target_mb - actual_kvc_mb)
                    kvc_fraction = actual_kvc_mb / target_mb
                    expert_fraction = expert_mb / target_mb
                    (
                        expert_cost,
                        churn_count,
                        churn_mb,
                        install_mb,
                        backing_miss_cost,
                        materialize_cost,
                    ) = cached_expert_cost(expert_mb)
                    if expert_cost >= 1.0e29:
                        infeasible_expert += 1
                        continue
                    total_cost = kvc_cost + expert_cost
                    saved_kvc_cost = (
                        kvc_baseline_cost - kvc_cost
                        if has_kvc_benefit_reference
                        else 0.0
                    )
                    candidate = (
                        float(
                            expert_cost - saved_kvc_cost
                            if has_kvc_benefit_reference
                            else total_cost
                        ),
                        float(total_cost),
                        float(-max(0.0, saved_kvc_cost)),
                        kvc_fraction,
                        expert_fraction,
                        kvc_cost,
                        expert_cost,
                        float(saved_kvc_cost),
                    )
                if best is None or candidate < best:
                    best = candidate
                    best_expert_stats = (
                        churn_count,
                        churn_mb,
                        install_mb,
                        backing_miss_cost,
                        materialize_cost,
                    )
                    best_kvc_tokens = int(kvc_tokens)
                    best_kvc_controller_cost = float(
                        kvc_controller_cost_cache.get(round(kvc_mb, 6), 0.0)
                    )
                    best_kvc_saved_cost = float(saved_kvc_cost)
        assert best is not None
        (
            _objective_cost,
            _total_cost,
            _negative_benefit_tiebreak,
            kvc_fraction,
            expert_fraction,
            kvc_cost,
            expert_cost,
            _selected_saved_kvc_cost,
        ) = best
        self.stats.planner_dp_candidate_count = len(kvc_points)
        self.stats.planner_dp_selected_kvc_candidates = 1 if kvc_fraction > 0.0 else 0
        self.stats.planner_dp_selected_expert_candidates = (
            1 if expert_fraction > 0.0 else 0
        )
        self.stats.planner_dp_infeasible_kvc_candidates = infeasible_kvc
        self.stats.planner_dp_infeasible_expert_candidates = infeasible_expert
        self.stats.planner_dp_selected_total_cost = float(_total_cost)
        self.stats.planner_dp_selected_kvc_cost = float(kvc_cost)
        self.stats.planner_dp_selected_expert_cost = float(expert_cost)
        self.stats.planner_estimated_kvc_capacity_benefit = max(
            0.0, float(best_kvc_saved_cost)
        )
        self.stats.planner_estimated_kvc_controller_cost = best_kvc_controller_cost
        self.stats.planner_estimated_expert_churn_count = float(best_expert_stats[0])
        self.stats.planner_estimated_expert_churn_mb = float(best_expert_stats[1])
        self.stats.planner_estimated_expert_install_mb = float(best_expert_stats[2])
        self.stats.planner_estimated_expert_backing_miss_cost = float(
            best_expert_stats[3]
        )
        self.stats.planner_estimated_expert_materialize_cost = float(
            best_expert_stats[4]
        )
        if layerwise_expert_plan:
            expert_mb = round(target_mb * expert_fraction, 6)
            capacities, layer_costs = expert_plan_cache.get(expert_mb, ({}, {}))
            current_expert_target = float(target_mb * expert_fraction)
            if (
                not self._planned_expert_slot_capacities_by_layer
                or current_expert_target
                > float(self._planned_expert_target_mb)
                + max(1e-3, self._expert_reclaim_quantum_mb())
            ):
                planned_capacities = {
                    int(layer_id): int(capacity)
                    for layer_id, capacity in capacities.items()
                }
                planned_capacities, layer_costs = (
                    self._shape_limited_coresid_expert_capacities(
                        current_expert_target,
                        coresid_expert_plan_inputs,
                        planned_capacities,
                        layer_costs,
                    )
                )
                self._planned_expert_slot_capacities_by_layer = planned_capacities
                self._planned_expert_cost_by_layer = {
                    int(layer_id): float(cost) for layer_id, cost in layer_costs.items()
                }
                self._planned_expert_target_mb = max(
                    float(self._planned_expert_target_mb),
                    current_expert_target,
                )
            self.stats.selected_expert_capacity_by_layer = json.dumps(
                {
                    str(layer_id): int(capacity)
                    for layer_id, capacity in self._planned_expert_slot_capacities_by_layer.items()
                },
                sort_keys=True,
            )
            self.stats.selected_expert_cost_by_layer = json.dumps(
                {
                    str(layer_id): round(float(cost), 6)
                    for layer_id, cost in self._planned_expert_cost_by_layer.items()
                },
                sort_keys=True,
            )
            self.stats.selected_expert_evictions_by_layer = (
                self._expert_evictions_json_from_capacities(
                    self._planned_expert_slot_capacities_by_layer
                )
            )
        self._planned_kvc_token_target = int(
            best_kvc_tokens if kvc_fraction > 0.0 else 0
        )
        if self._planned_kvc_token_target > 0:
            if self.config.kvc_backend == "per-layer-arena":
                self._planned_kvc_tokens_by_layer = (
                    self._build_layer_aware_kvc_token_plan_from_context(
                        self._planned_kvc_token_target,
                        layer_ids=kvc_layer_ids,
                        max_tokens_per_layer=kvc_max_tokens_per_layer,
                        block_tokens=kvc_block_tokens_for_plan,
                        avg_prefix=self._avg_prefix_len(forward_batch),
                        batch_size=max(
                            1, len(self._batch_req_indices_and_lens(forward_batch))
                        ),
                    )
                )
                self._planned_kvc_token_target = int(
                    sum(self._planned_kvc_tokens_by_layer.values())
                )
            else:
                self._planned_kvc_tokens_by_layer = (
                    self._build_layer_aware_kvc_token_plan(
                        self._planned_kvc_token_target, forward_batch
                    )
                )
                self._planned_kvc_token_target = int(
                    sum(self._planned_kvc_tokens_by_layer.values())
                )
            self.stats.selected_kvc_tokens_by_layer = self._kvc_tokens_by_layer_json(
                self._planned_kvc_tokens_by_layer
            )
        else:
            self._planned_kvc_tokens_by_layer = {}
        return kvc_fraction, expert_fraction, kvc_cost, expert_cost

    def _context_heuristic_kvc_fraction(self, forward_batch: Any) -> float:
        avg_prefix = self._avg_prefix_len(forward_batch)
        if avg_prefix <= 1024:
            return 0.75
        if avg_prefix <= 4096:
            return 0.50
        return 0.25

    def _uses_layer_aware_kvc_plan(self) -> bool:
        return (
            self.config.policy
            in ("layer-aware-joint", "layer-aware-joint-dp", "coresid")
            and self.config.kvc_backend == "per-layer-arena"
        )

    def _requires_per_layer_kvc_backend(self) -> bool:
        return self.config.policy in (
            "layer-aware-joint",
            "layer-aware-joint-dp",
            "coresid",
        )

    def _per_layer_kvc_backend_ready(self) -> bool:
        if self.config.kvc_backend == "virtual-arena":
            return self._virtual_scratch_locs is not None
        return self.config.kvc_backend == "per-layer-arena"

    def _per_layer_kvc_backend_can_physically_reclaim(self) -> bool:
        return self.config.kvc_backend in ("per-layer-arena", "virtual-arena")

    def _uses_per_layer_attention_override(self) -> bool:
        return self.config.kvc_backend in ("per-layer-arena", "virtual-arena")

    def _set_joint_planner_choice(
        self,
        *,
        kvc_fraction: float,
        expert_fraction: float,
        used_hotness: bool,
        fallback_reason: str,
        kvc_cost: float,
        expert_cost: float,
    ) -> None:
        target_mb = max(0.0, float(self.stats.effective_reclaim_target_mb))
        if not self.config.dynamic_pressure_from_kvc and target_mb <= 0.0:
            target_mb = max(0.0, self.config.reclaim_limit_mb)
        self.stats.planner_used_hotness = used_hotness
        self.stats.planner_fallback_reason = fallback_reason
        self.stats.planner_estimated_kvc_cost = float(kvc_cost)
        self.stats.planner_estimated_expert_cost = float(expert_cost)
        self.stats.planner_selected_kvc_reclaim_mb = target_mb * kvc_fraction
        self.stats.planner_selected_expert_reclaim_mb = target_mb * expert_fraction
        if target_mb <= 0.0:
            self.stats.selected_kvc_tokens_by_layer = ""
            self.stats.selected_expert_evictions_by_layer = ""

    def _has_expert_hotness(self) -> bool:
        for hotness in self._expert_hotness_decode.values():
            if hotness:
                return True
        for hotness in self._expert_hotness_prefill.values():
            if hotness:
                return True
        for state in self._expert_layers.values():
            if state.hotness_decode or state.hotness_prefill:
                return True
        return False

    def _has_decode_expert_hotness(self) -> bool:
        for hotness in self._expert_hotness_decode.values():
            if hotness:
                return True
        for state in self._expert_layers.values():
            if state.hotness_decode:
                return True
        return False

    def _has_prefill_expert_hotness(self) -> bool:
        for hotness in self._expert_hotness_prefill.values():
            if hotness:
                return True
        for state in self._expert_layers.values():
            if state.hotness_prefill:
                return True
        return False

    def _has_initial_expert_hotness(self) -> bool:
        return self._has_decode_expert_hotness() or self._has_prefill_expert_hotness()

    def _has_prefill_expert_hotness_gpu(self) -> bool:
        return any(
            counts is not None and int(counts.numel()) > 0
            for counts in self._expert_hotness_gpu_prefill.values()
        )

    def _extend_progress_ratio(self, forward_batch: Any) -> float:
        orig_seq_lens = getattr(forward_batch, "orig_seq_lens", None)
        seq_lens = getattr(forward_batch, "seq_lens", None)
        if orig_seq_lens is None or seq_lens is None:
            return 1.0
        try:
            orig = orig_seq_lens.detach().float()
            cur = seq_lens.detach().float()
            if orig.numel() == 0 or cur.numel() == 0:
                return 1.0
            denom = torch.clamp(orig, min=1.0)
            ratio = torch.clamp(cur[: orig.numel()] / denom, min=0.0, max=1.0)
            return float(ratio.mean().item())
        except Exception:
            return 1.0

    def _estimate_planner_kvc_reclaim_cost(
        self, reclaim_mb: float, forward_batch: Any
    ) -> float:
        if reclaim_mb <= 0.0:
            return 0.0
        if self.config.kvc_backend == "per-layer-arena":
            return self._estimate_arena_kvc_reclaim_cost(reclaim_mb, forward_batch)
        self.stats.planner_estimated_kvc_controller_cost = 0.0
        self.stats.planner_fallback_reason = "kvc_cost_requires_per_layer_arena"
        return 1.0e30

    def _estimate_arena_kvc_reclaim_cost(
        self, reclaim_mb: float, forward_batch: Any
    ) -> float:
        token_plan = self._arena_kvc_token_plan_for_reclaim(reclaim_mb, forward_batch)
        layer_ids = self._kvc_layer_ids()
        avg_prefix = max(1.0, self._avg_prefix_len(forward_batch))
        batch_size = max(1, int(getattr(self.stats, "observed_batch_size", 0) or 1))
        recovery_steps = self._expected_kvc_recovery_steps()
        return self._estimate_arena_kvc_reclaim_cost_from_plan(
            token_plan,
            layer_ids=layer_ids,
            avg_prefix=avg_prefix,
            batch_size=batch_size,
            recovery_steps=recovery_steps,
        )

    def _estimate_arena_kvc_reclaim_cost_from_plan(
        self,
        token_plan: Dict[int, int],
        *,
        layer_ids: List[int],
        avg_prefix: float,
        batch_size: int,
        recovery_steps: int,
    ) -> float:
        if not token_plan:
            self.stats.planner_estimated_kvc_controller_cost = 0.0
            return 0.0

        # CoResID compares KVC candidates by predicted exposed stall.  MB is
        # used only to derive raw copy bytes; it is not the planner objective.
        controller_cost = 0.0
        exposed_stall = 0.0
        layer_rank = {int(layer_id): idx for idx, layer_id in enumerate(layer_ids)}
        overlap_windows = self._estimate_kvc_overlap_windows_ms(
            layer_ids, avg_prefix=avg_prefix, batch_size=batch_size
        )
        block_tokens = max(
            self._page_size,
            self._align_tokens_up(int(self.config.kvc_block_tokens)),
        )
        copy_queue_ms = 0.0
        overlap_sum = 0.0
        exposed_sum = 0.0
        for layer_id, tokens in sorted(
            token_plan.items(), key=lambda item: layer_rank.get(int(item[0]), 0)
        ):
            if tokens <= 0:
                continue
            raw_copy_ms = self._estimate_arena_kvc_layer_raw_reload_ms(
                int(layer_id), int(tokens)
            )
            overlap_ms = float(overlap_windows.get(int(layer_id), 0.0))
            exposed = max(0.0, raw_copy_ms + copy_queue_ms - overlap_ms)
            exposed_stall += exposed * recovery_steps
            overlap_sum += overlap_ms
            exposed_sum += exposed
            copy_queue_ms += raw_copy_ms
            blocks = max(1, int(math.ceil(float(tokens) / float(block_tokens))))
            controller_cost += (0.003 + 0.0002 * float(blocks)) * recovery_steps
        self.stats.planner_estimated_kvc_controller_cost = controller_cost
        self.stats.planner_estimated_kvc_overlap_ms = overlap_sum
        self.stats.planner_estimated_kvc_exposed_ms = exposed_sum
        return exposed_stall + controller_cost

    def _expected_kvc_recovery_steps(self) -> int:
        # Fig4-style decode uses 16 generated tokens.  At planning time the
        # first decode step has just started, so stats.decode_steps is not the
        # remaining horizon.  Charge KVC by the expected repeated reload/evict
        # cycles instead of treating it as a one-shot recovery.
        observed = int(getattr(self.stats, "decode_steps", 0) or 0)
        return max(1, observed if observed > 1 else 16)

    def _arena_kvc_token_plan_for_reclaim(
        self, reclaim_mb: float, forward_batch: Any
    ) -> Dict[int, int]:
        if reclaim_mb <= 0.0:
            return {}
        bytes_per_token = max(1, self._bytes_per_kvc_token_per_layer())
        total_tokens = int(reclaim_mb * 1024.0 * 1024.0 / float(bytes_per_token))
        return self._build_layer_aware_kvc_token_plan(total_tokens, forward_batch)

    def _layer_aware_kvc_plan_context(
        self, forward_batch: Any
    ) -> Tuple[List[int], int, int]:
        layer_ids = self._kvc_layer_ids()
        block_tokens = max(
            self._page_size,
            self._align_tokens_up(int(self.config.kvc_block_tokens)),
        )
        pairs = self._batch_req_indices_and_lens(forward_batch)
        max_tokens_per_layer = sum(
            self._align_tokens_down(max(0, int(seq_len) - 1))
            for _req_idx, seq_len in pairs
        )
        if max_tokens_per_layer <= 0:
            max_tokens_per_layer = int(max(1.0, self._avg_prefix_len(forward_batch)))
        max_tokens_per_layer = self._align_tokens_down(max_tokens_per_layer)
        return layer_ids, max_tokens_per_layer, block_tokens

    def _build_layer_aware_kvc_token_plan_from_context(
        self,
        total_tokens: int,
        *,
        layer_ids: List[int],
        max_tokens_per_layer: int,
        block_tokens: int,
        avg_prefix: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[int, int]:
        if not layer_ids or total_tokens <= 0 or max_tokens_per_layer <= 0:
            return {}
        total_tokens = self._align_tokens_down(int(total_tokens))
        total_capacity = max_tokens_per_layer * len(layer_ids)
        total_tokens = min(total_tokens, total_capacity)
        if total_tokens <= 0:
            return {}
        if avg_prefix is not None and batch_size is not None:
            return self._build_overlap_capacity_kvc_token_plan_from_context(
                total_tokens,
                layer_ids=layer_ids,
                max_tokens_per_layer=max_tokens_per_layer,
                block_tokens=block_tokens,
                avg_prefix=float(avg_prefix),
                batch_size=int(batch_size),
            )
        weight_sum = float(len(layer_ids) * (len(layer_ids) + 1)) / 2.0 or 1.0
        remaining = total_tokens
        plan: Dict[int, int] = {}
        for idx, layer_id in enumerate(layer_ids):
            raw = int(round(total_tokens * float(idx + 1) / weight_sum))
            tokens = min(max_tokens_per_layer, self._align_tokens_down(raw))
            tokens = tokens - (tokens % block_tokens)
            if tokens > 0:
                plan[int(layer_id)] = tokens
                remaining -= tokens
        if remaining > 0:
            for layer_id in reversed(layer_ids):
                if remaining <= 0:
                    break
                add = min(max_tokens_per_layer - plan.get(int(layer_id), 0), remaining)
                add = add - (add % block_tokens)
                if add <= 0:
                    continue
                plan[int(layer_id)] = plan.get(int(layer_id), 0) + add
                remaining -= add
        return {layer: tokens for layer, tokens in plan.items() if tokens > 0}

    def _build_overlap_capacity_kvc_token_plan_from_context(
        self,
        total_tokens: int,
        *,
        layer_ids: List[int],
        max_tokens_per_layer: int,
        block_tokens: int,
        avg_prefix: float,
        batch_size: int,
    ) -> Dict[int, int]:
        overlap_windows = self._estimate_kvc_overlap_windows_ms(
            layer_ids,
            avg_prefix=max(1.0, float(avg_prefix)),
            batch_size=max(1, int(batch_size)),
        )
        bytes_per_token = max(1, self._bytes_per_kvc_token_per_layer())
        fallback_per_token_ms = (2.0 * float(bytes_per_token)) / 1.0e9 * 1000.0
        capacities: Dict[int, int] = {}
        for layer_id in layer_ids:
            layer_id = int(layer_id)
            overlap_ms = max(0.0, float(overlap_windows.get(layer_id, 0.0)))
            reload_ewma = self._kvc_reload_ms_per_mb_ewma_by_layer.get(layer_id)
            evict_ewma = self._kvc_evict_ms_per_mb_ewma_by_layer.get(layer_id)
            if reload_ewma is not None or evict_ewma is not None:
                per_token_ms = (
                    (float(reload_ewma or 0.0) + float(evict_ewma or 0.0))
                    * float(bytes_per_token)
                    / float(1024 * 1024)
                )
            else:
                per_token_ms = fallback_per_token_ms
            if per_token_ms <= 0.0:
                per_token_ms = fallback_per_token_ms
            usable_ms = max(0.0, overlap_ms - 0.03)
            tokens = int(usable_ms / per_token_ms)
            tokens = min(int(max_tokens_per_layer), self._align_tokens_down(tokens))
            tokens = tokens - (tokens % max(1, int(block_tokens)))
            if tokens > 0:
                capacities[layer_id] = tokens
        remaining = self._align_tokens_down(int(total_tokens))
        plan: Dict[int, int] = {}
        for layer_id in sorted(layer_ids, reverse=True):
            if remaining <= 0:
                break
            capacity = int(capacities.get(int(layer_id), 0))
            if capacity <= 0:
                continue
            tokens = min(capacity, remaining)
            tokens = tokens - (tokens % max(1, int(block_tokens)))
            if tokens <= 0:
                continue
            plan[int(layer_id)] = tokens
            remaining -= tokens
        return {layer: tokens for layer, tokens in plan.items() if tokens > 0}

    def _estimate_arena_kvc_candidate_costs_vectorized(
        self,
        kvc_points: List[float],
        kvc_token_points: Dict[float, int],
        *,
        layer_ids: List[int],
        max_tokens_per_layer: int,
        block_tokens: int,
        avg_prefix: float,
        batch_size: int,
        recovery_steps: int,
    ) -> Dict[float, Tuple[float, int, float]]:
        if not kvc_points or not layer_ids or max_tokens_per_layer <= 0:
            return {}
        points = [float(x) for x in kvc_points if float(x) > 0.0]
        if not points:
            return {}
        token_counts = torch.tensor(
            [int(kvc_token_points.get(point, 0)) for point in points],
            dtype=torch.int64,
            device="cpu",
        )
        layer_count = len(layer_ids)
        bytes_per_token = max(1, self._bytes_per_kvc_token_per_layer())
        ms_per_token: List[float] = []
        fallback_per_token = (2.0 * float(bytes_per_token)) / 1.0e9 * 1000.0
        for layer_id in layer_ids:
            layer_id = int(layer_id)
            reload_ewma = self._kvc_reload_ms_per_mb_ewma_by_layer.get(layer_id)
            evict_ewma = self._kvc_evict_ms_per_mb_ewma_by_layer.get(layer_id)
            if reload_ewma is not None or evict_ewma is not None:
                ms_per_mb = float(reload_ewma or 0.0) + float(evict_ewma or 0.0)
                value = ms_per_mb * float(bytes_per_token) / float(1024 * 1024)
            else:
                value = fallback_per_token
            ms_per_token.append(value if value > 0.0 else fallback_per_token)
        ms_per_token_tensor = torch.tensor(
            ms_per_token, dtype=torch.float64, device="cpu"
        )
        overlap_values = self._estimate_kvc_overlap_windows_ms(
            layer_ids, avg_prefix=avg_prefix, batch_size=batch_size
        )
        overlap = torch.tensor(
            [float(overlap_values.get(int(layer_id), 0.0)) for layer_id in layer_ids],
            dtype=torch.float64,
            device="cpu",
        )
        usable = torch.clamp(overlap - 0.03, min=0.0)
        capacities = torch.floor(usable / ms_per_token_tensor).to(torch.int64)
        capacities = torch.minimum(
            capacities,
            torch.tensor(int(max_tokens_per_layer), dtype=torch.int64, device="cpu"),
        )
        capacities = (capacities // int(block_tokens)) * int(block_tokens)

        plan = torch.zeros((len(points), layer_count), dtype=torch.int64, device="cpu")
        remaining = (token_counts // int(block_tokens)) * int(block_tokens)
        for idx in range(layer_count - 1, -1, -1):
            positive = remaining > 0
            if not bool(torch.any(positive).item()):
                break
            add = torch.minimum(capacities[idx], remaining)
            add = (add // int(block_tokens)) * int(block_tokens)
            add = torch.where(positive & (add > 0), add, torch.zeros_like(add))
            plan[:, idx] = add
            remaining -= add

        raw_copy = 0.03 + plan.to(torch.float64) * ms_per_token_tensor[None, :]
        raw_copy = torch.where(plan > 0, raw_copy, torch.zeros_like(raw_copy))
        queued = torch.cumsum(raw_copy, dim=1) - raw_copy
        exposed = torch.clamp(raw_copy + queued - overlap[None, :], min=0.0)
        blocks = torch.ceil(plan.to(torch.float64) / float(max(1, block_tokens)))
        metadata = torch.where(
            plan > 0, 0.003 + 0.0002 * blocks, torch.zeros_like(blocks)
        )
        exposed_cost = exposed.sum(axis=1) * float(recovery_steps)
        controller_cost = metadata.sum(axis=1) * float(recovery_steps)
        total_cost = exposed_cost + controller_cost
        result: Dict[float, Tuple[float, int, float]] = {}
        if points:
            self.stats.planner_estimated_kvc_overlap_ms = float(
                torch.where(plan > 0, overlap[None, :], torch.zeros_like(raw_copy))
                .sum(axis=1)
                .mean()
                .item()
            )
            self.stats.planner_estimated_kvc_exposed_ms = float(
                exposed.sum(axis=1).mean().item()
            )
        for idx, point in enumerate(points):
            result[float(point)] = (
                float(total_cost[idx].item()),
                int(plan[idx].sum().item()),
                float(controller_cost[idx].item()),
            )
        return result

    def _estimate_arena_kvc_layer_raw_reload_ms(
        self, layer_id: int, tokens: int
    ) -> float:
        if tokens <= 0:
            return 0.0
        bytes_per_token = max(1, self._bytes_per_kvc_token_per_layer())
        mb = float(tokens * bytes_per_token) / float(1024 * 1024)
        reload_ewma = self._kvc_reload_ms_per_mb_ewma_by_layer.get(int(layer_id))
        evict_ewma = self._kvc_evict_ms_per_mb_ewma_by_layer.get(int(layer_id))
        if reload_ewma is not None or evict_ewma is not None:
            return 0.03 + mb * (float(reload_ewma or 0.0) + float(evict_ewma or 0.0))
        copy_bytes = float(tokens) * float(bytes_per_token)
        # Per-layer arena recovery uses indexed gather/scatter copies rather
        # than a single contiguous memcpy.  The runtime must restore missing KV
        # before attention and evict it again after the step, so charge both
        # H2D reload and D2H backup.  This is a theoretical hardware/runtime
        # path estimate, not a trace replay: the effective bandwidth reflects
        # index_copy/index_select style copies into layer-local KV buffers.
        effective_bandwidth_gbps = 1.0
        copy_ms = (2.0 * copy_bytes) / (effective_bandwidth_gbps * 1.0e9) * 1000.0
        return 0.03 + copy_ms

    def _estimate_layer_kvc_overlap_window_ms(
        self, layer_id: int, forward_batch: Any
    ) -> float:
        layer_ids = self._kvc_layer_ids()
        if not layer_ids:
            return 0.0
        avg_prefix = max(1.0, self._avg_prefix_len(forward_batch))
        batch_size = max(1, int(getattr(self.stats, "observed_batch_size", 0) or 1))
        return float(
            self._estimate_kvc_overlap_windows_ms(
                layer_ids, avg_prefix=avg_prefix, batch_size=batch_size
            ).get(int(layer_id), 0.0)
        )

    def _model_hf_config(self) -> Any:
        runner = self._runner
        model_config = getattr(runner, "model_config", None)
        if model_config is not None:
            hf_config = getattr(model_config, "hf_config", None)
            if hf_config is not None:
                return hf_config
        server_args = getattr(runner, "server_args", None)
        get_model_config = getattr(server_args, "get_model_config", None)
        if callable(get_model_config):
            try:
                model_config = get_model_config()
                return getattr(model_config, "hf_config", None)
            except Exception:
                return None
        return None

    def _estimate_kvc_overlap_windows_ms(
        self, layer_ids: List[int], *, avg_prefix: float, batch_size: int
    ) -> Dict[int, float]:
        if not layer_ids:
            return {}
        hf_config = self._model_hf_config()
        hidden = int(getattr(hf_config, "hidden_size", 4096) or 4096)
        intermediate = int(
            getattr(
                hf_config,
                "intermediate_size",
                getattr(hf_config, "moe_intermediate_size", hidden * 4),
            )
            or hidden * 4
        )
        num_heads = max(1, int(getattr(hf_config, "num_attention_heads", 32) or 32))
        num_kv_heads = max(
            1,
            int(
                getattr(
                    hf_config,
                    "num_key_value_heads",
                    getattr(hf_config, "num_kv_heads", num_heads),
                )
                or num_heads
            ),
        )
        seq_len = max(1.0, float(avg_prefix))
        batch = max(1.0, float(batch_size))
        # This is a relative model for planner ordering, not a performance
        # profiler.  Use conservative BF16-scale throughput and include a
        # launch floor so small layers still provide limited overlap.
        effective_tflops = 180.0
        attn_projection_ops = 4.0 * batch * float(hidden) * float(hidden)
        attn_context_ops = (
            2.0
            * batch
            * seq_len
            * float(hidden)
            * (float(num_kv_heads) / float(num_heads))
        )
        mlp_ops = 2.0 * batch * float(hidden) * float(intermediate)
        layer_ms = (
            (attn_projection_ops + attn_context_ops + mlp_ops)
            / (effective_tflops * 1.0e12)
            * 1000.0
        )
        layer_ms = max(0.02, min(5.0, layer_ms))
        windows: Dict[int, float] = {}
        prefix_ms = 0.0
        for layer_id in layer_ids:
            windows[int(layer_id)] = prefix_ms
            prefix_ms += layer_ms
        return windows

    def _estimate_arena_kvc_layer_metadata_ms(
        self, layer_id: int, tokens: int
    ) -> float:
        if tokens <= 0:
            return 0.0
        block_tokens = max(
            self._page_size,
            self._align_tokens_up(int(self.config.kvc_block_tokens)),
        )
        blocks = max(1, int(math.ceil(float(tokens) / float(block_tokens))))
        return 0.003 + 0.0002 * float(blocks)
