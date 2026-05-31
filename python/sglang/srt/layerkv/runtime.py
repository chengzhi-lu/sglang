"""LayerKV runtime adapter for SGLang v0.5.12.

The integration keeps SGLang's native KV pool and MoE execution path in place,
while adding the policy-residency lifecycle hooks needed to make the feature
runnable and measurable.  The first physical path supports MHA KV pools with
page_size=1 by persistently moving selected KV slots to compact CPU backing
storage and reloading them before attention consumes them.
"""

from __future__ import annotations

import bisect
import contextlib
import dataclasses
import functools
import heapq
import json
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LayerKVConfig:
    enabled: bool = False
    mode: str = "off"
    policy: str = "none"
    reclaim_limit_mb: float = 0.0
    dynamic_pressure_from_kvc: bool = False
    kvc_block_tokens: int = 16
    kvc_backend: str = "token-slot"
    kvc_scheduler: str = "async-deadline"
    runtime_profile: str = "optimized"
    virtual_scratch_tokens: int = 4096
    debug_stats: bool = False
    profile_detail: bool = False
    disallow_destructive_fallback: bool = True
    expert_backing_cache_mb: float = 0.0
    expert_cpu_backing_mode: str = "none"
    expert_forward_hooks: bool = True
    expert_collector_only: bool = False
    expert_hotness_sample_interval: int = 16
    expert_install_layers_per_step: int = 1
    expert_install_budget_mb: float = 128.0
    expert_install_target_steps: int = 0
    expert_copy_budget_mb: float = 64.0
    expert_copy_chunk_mb: float = 128.0
    expert_copy_max_budget_mb: float = 0.0
    expert_copy_lookahead_layers: int = 0
    expert_copy_force_drain: bool = False
    worker_role: str = "standalone"

    @classmethod
    def from_server_args(cls, server_args: Any) -> "LayerKVConfig":
        disaggregation_mode = str(getattr(server_args, "disaggregation_mode", "null"))
        worker_role = "pd-decode" if disaggregation_mode == "decode" else "standalone"
        return cls(
            enabled=bool(getattr(server_args, "enable_layerkv", False)),
            mode=str(getattr(server_args, "layerkv_mode", "off")),
            policy=str(getattr(server_args, "layerkv_policy", "none")),
            reclaim_limit_mb=float(
                getattr(server_args, "layerkv_reclaim_limit_mb", 0.0) or 0.0
            ),
            dynamic_pressure_from_kvc=bool(
                getattr(server_args, "layerkv_dynamic_pressure_from_kvc", False)
            ),
            kvc_block_tokens=max(
                1, int(getattr(server_args, "layerkv_kvc_block_tokens", 16) or 16)
            ),
            kvc_backend=str(
                getattr(server_args, "layerkv_kvc_backend", "token-slot")
                or "token-slot"
            ),
            kvc_scheduler=str(
                getattr(server_args, "layerkv_kvc_scheduler", "async-deadline")
            ),
            runtime_profile=str(
                getattr(server_args, "layerkv_runtime_profile", "optimized")
                or "optimized"
            ),
            virtual_scratch_tokens=max(
                0,
                int(getattr(server_args, "layerkv_virtual_scratch_tokens", 4096) or 0),
            ),
            debug_stats=bool(getattr(server_args, "layerkv_debug_stats", False)),
            profile_detail=bool(getattr(server_args, "layerkv_profile_detail", False)),
            disallow_destructive_fallback=bool(
                getattr(server_args, "layerkv_disallow_destructive_fallback", True)
            ),
            expert_backing_cache_mb=float(
                getattr(server_args, "layerkv_expert_backing_cache_mb", 0.0) or 0.0
            ),
            expert_cpu_backing_mode=str(
                getattr(server_args, "layerkv_expert_cpu_backing_mode", "none")
                or "none"
            ),
            expert_forward_hooks=bool(
                getattr(server_args, "layerkv_expert_forward_hooks", True)
            ),
            expert_collector_only=bool(
                getattr(server_args, "layerkv_expert_collector_only", False)
            ),
            expert_hotness_sample_interval=max(
                1,
                int(
                    getattr(
                        server_args, "layerkv_expert_hotness_sample_interval", 16
                    )
                    or 16
                ),
            ),
            expert_install_layers_per_step=max(
                1,
                int(
                    getattr(server_args, "layerkv_expert_install_layers_per_step", 1)
                    or 1
                ),
            ),
            expert_install_budget_mb=max(
                0.0,
                float(
                    getattr(server_args, "layerkv_expert_install_budget_mb", 128.0)
                    or 0.0
                ),
            ),
            expert_install_target_steps=max(
                0,
                int(
                    getattr(server_args, "layerkv_expert_install_target_steps", 0) or 0
                ),
            ),
            expert_copy_budget_mb=max(
                0.0,
                float(
                    getattr(server_args, "layerkv_expert_copy_budget_mb", 64.0)
                    or 0.0
                ),
            ),
            expert_copy_chunk_mb=max(
                1.0,
                float(
                    getattr(server_args, "layerkv_expert_copy_chunk_mb", 128.0)
                    or 128.0
                ),
            ),
            expert_copy_max_budget_mb=max(
                0.0,
                float(
                    getattr(server_args, "layerkv_expert_copy_max_budget_mb", 0.0)
                    or 0.0
                ),
            ),
            expert_copy_lookahead_layers=max(
                0,
                int(getattr(server_args, "layerkv_expert_copy_lookahead_layers", 0)),
            ),
            expert_copy_force_drain=bool(
                getattr(server_args, "layerkv_expert_copy_force_drain", False)
            ),
            worker_role=worker_role,
        )


@dataclasses.dataclass
class LayerKVStats:
    forward_decode_count: int = 0
    forward_extend_count: int = 0
    kvc_set_kv_count: int = 0
    kvc_get_key_count: int = 0
    kvc_get_value_count: int = 0
    kvc_get_kv_count: int = 0
    kvc_tokens_written: int = 0
    kvc_bytes_written: int = 0
    configured_reclaim_limit_mb: float = 0.0
    effective_reclaim_target_mb: float = 0.0
    needed_pressure_mb: float = 0.0
    dynamic_pressure_live_tokens: int = 0
    dynamic_pressure_write_tokens: int = 0
    dynamic_pressure_decode_reserve_tokens: int = 0
    dynamic_pressure_total_tokens: int = 0
    dynamic_pressure_budget_tokens: int = 0
    dynamic_pressure_shortage_tokens: int = 0
    dynamic_pressure_token_usage_ratio: float = 0.0
    dynamic_pressure_active_steps: int = 0
    dynamic_pressure_skip_steps: int = 0
    kvc_evict_without_pressure_count: int = 0
    kvc_scheduler_invisible_skip_count: int = 0
    kvc_scheduler_invisible_skip_tokens: int = 0
    kvc_topup_skip_count: int = 0
    kvc_topup_apply_count: int = 0
    kvc_topup_pressure_mb: float = 0.0
    kvc_topup_margin_mb: float = 0.0
    pre_retract_reclaim_attempt_count: int = 0
    pre_retract_reclaim_success_count: int = 0
    pre_retract_reclaim_needed_tokens: int = 0
    pre_retract_reclaim_requested_kvc_tokens: int = 0
    pre_retract_reclaim_allocator_available_before: int = 0
    pre_retract_reclaim_allocator_available_after: int = 0
    pre_retract_reclaim_scheduler_visible_success_count: int = 0
    scheduler_budget_observation_count: int = 0
    scheduler_budget_required_tokens: int = 0
    scheduler_budget_native_available_tokens: int = 0
    scheduler_budget_shortage_tokens: int = 0
    scheduler_budget_credit_tokens: int = 0
    scheduler_budget_raw_offloaded_tokens: int = 0
    scheduler_budget_releasable_tokens: int = 0
    scheduler_budget_credit_limit_reason: str = ""
    scheduler_budget_effective_available_tokens: int = 0
    scheduler_budget_credit_used_tokens: int = 0
    scheduler_budget_credit_denied_count: int = 0
    scheduler_budget_credit_prevented_retract_count: int = 0
    scheduler_budget_pressure_tokens: int = 0
    scheduler_budget_pre_retract_wait_ms: float = 0.0
    scheduler_budget_small_shortage_skip_count: int = 0
    scheduler_running_req_sum: int = 0
    scheduler_running_req_avg: float = 0.0
    scheduler_running_req_max: int = 0
    scheduler_token_sum: int = 0
    scheduler_token_avg: float = 0.0
    scheduler_token_max: int = 0
    scheduler_token_usage_sum: float = 0.0
    scheduler_token_usage_avg: float = 0.0
    scheduler_token_usage_max: float = 0.0
    scheduler_retracted_req_current: int = 0
    scheduler_retracted_req_sum: int = 0
    scheduler_retracted_req_max: int = 0
    virtual_scratch_capacity_tokens: int = 0
    virtual_scratch_used_tokens: int = 0
    virtual_scratch_alloc_failed_count: int = 0
    virtual_kvc_evict_count: int = 0
    virtual_kvc_materialize_count: int = 0
    virtual_kvc_materialize_token_count: int = 0
    virtual_kvc_materialize_layer_count: int = 0
    virtual_kvc_materialize_ms: float = 0.0
    virtual_kvc_prefetch_count: int = 0
    virtual_kvc_prefetch_wait_count: int = 0
    virtual_kvc_prefetch_ready_before_use_count: int = 0
    virtual_kvc_prefetch_fallback_sync_count: int = 0
    virtual_kvc_scheduler_skip_count: int = 0
    virtual_kvc_persistent_cache_hit_count: int = 0
    virtual_kvc_persistent_cache_miss_count: int = 0
    virtual_kvc_persistent_cache_store_count: int = 0
    virtual_kvc_persistent_cache_invalidate_count: int = 0
    virtual_kvc_direct_metadata_patch_count: int = 0
    virtual_kvc_direct_metadata_patch_fallback_count: int = 0
    kvc_demand_layer_count: int = 0
    kvc_layerwise_demand_count: int = 0
    kvc_layerwise_demand_token_count: int = 0
    kvc_layerwise_selector_fallback_count: int = 0
    kvc_demand_shared_signature_count: int = 0
    kvc_demand_signature_mismatch_count: int = 0
    metadata_patch_cache_hit_count: int = 0
    metadata_patch_cache_miss_count: int = 0
    metadata_patch_cache_fallback_count: int = 0
    metadata_patch_layer_fallback_count: int = 0
    metadata_patch_slice_count: int = 0
    metadata_patch_slice_token_count: int = 0
    metadata_pointer_switch_count: int = 0
    kvc_eviction_index_hit_count: int = 0
    kvc_eviction_index_fallback_count: int = 0
    kvc_eviction_index_rebuild_count: int = 0
    kvc_layerwise_evict_cursor_hit_count: int = 0
    kvc_layerwise_evict_cursor_reset_count: int = 0
    kvc_layerwise_evict_plan_layer_count: int = 0
    kvc_layerwise_evict_selected_layer_count: int = 0
    kvc_layerwise_evict_selected_token_count: int = 0
    kvc_layerwise_required_index_hit_count: int = 0
    kvc_layerwise_required_index_scan_count: int = 0
    kvc_layerwise_required_index_stale_count: int = 0
    kvc_layerwise_required_selected_token_count: int = 0
    kvc_required_cache_hit: int = 0
    kvc_required_cache_miss: int = 0
    kvc_required_cache_fallback: int = 0
    kvc_required_scanned_keys: int = 0
    kvc_evict_selector_fast_hit: int = 0
    kvc_evict_selector_fallback: int = 0
    kvc_evict_selector_scanned_entries: int = 0
    kvc_evict_selector_selected_entries: int = 0
    kvc_task_coalesced_count: int = 0
    kvc_avg_task_tokens: float = 0.0
    kvc_avg_task_bytes: float = 0.0
    kvc_task_token_sum: int = 0
    kvc_task_byte_sum: int = 0
    kvc_metadata_dirty_guard_skip_count: int = 0
    kvc_layerwise_scheduler_deadline_reject_count: int = 0
    kvc_layerwise_scheduler_dynamic_budget_count: int = 0
    kvc_layerwise_cost_observation_count: int = 0
    kvc_layerwise_reload_ewma_ms_per_mb: float = 0.0
    kvc_layerwise_evict_ewma_ms_per_mb: float = 0.0
    virtual_kvc_release_count: int = 0
    virtual_kvc_release_token_count: int = 0
    virtual_kvc_scratch_overflow_count: int = 0
    kvc_evict_async_count: int = 0
    kvc_evict_async_finalize_count: int = 0
    kvc_evict_async_wait_ms: float = 0.0
    kvc_evict_pending_token_count: int = 0
    available_kvc_reclaim_mb: float = 0.0
    available_expert_reclaim_mb: float = 0.0
    available_total_reclaim_mb: float = 0.0
    target_limited_reason: str = ""
    requested_total_reclaim_mb: float = 0.0
    effective_kvc_reclaim_mb: float = 0.0
    policy_kvc_fraction: float = 0.0
    policy_expert_fraction: float = 0.0
    full_policy_semantics_supported: bool = True
    policy_semantics_reason: str = ""
    planner_version: str = ""
    planner_used_hotness: bool = False
    planner_fallback_reason: str = ""
    planner_estimated_kvc_cost: float = 0.0
    planner_estimated_expert_cost: float = 0.0
    planner_estimated_expert_churn_count: float = 0.0
    planner_estimated_expert_churn_mb: float = 0.0
    planner_estimated_expert_install_mb: float = 0.0
    planner_estimated_kvc_controller_cost: float = 0.0
    planner_estimated_kvc_overlap_ms: float = 0.0
    planner_estimated_kvc_exposed_ms: float = 0.0
    planner_estimated_expert_backing_miss_cost: float = 0.0
    planner_estimated_expert_materialize_cost: float = 0.0
    planner_dp_table_build_ms: float = 0.0
    planner_dp_lookup_ms: float = 0.0
    planner_dp_kvc_candidate_count: int = 0
    planner_dp_expert_candidate_count: int = 0
    planner_cache_hit_count: int = 0
    planner_cache_miss_count: int = 0
    planner_selected_kvc_reclaim_mb: float = 0.0
    planner_selected_expert_reclaim_mb: float = 0.0
    planner_dp_candidate_count: int = 0
    planner_dp_selected_kvc_candidates: int = 0
    planner_dp_selected_expert_candidates: int = 0
    planner_dp_infeasible_kvc_candidates: int = 0
    planner_dp_infeasible_expert_candidates: int = 0
    planner_dp_selected_total_cost: float = 0.0
    planner_dp_selected_kvc_cost: float = 0.0
    planner_dp_selected_expert_cost: float = 0.0
    scheduler_kvc_deferred_count: int = 0
    scheduler_expert_deferred_count: int = 0
    scheduler_copy_budget_kvc_tasks: int = 0
    scheduler_copy_budget_expert_tasks: int = 0
    selected_kvc_tokens_by_layer: str = ""
    selected_expert_evictions_by_layer: str = ""
    selected_expert_capacity_by_layer: str = ""
    selected_expert_cost_by_layer: str = ""
    applied_expert_evictions_by_layer: str = ""
    expert_plan_match: bool = True
    expert_plan_mismatch_reason: str = ""
    layerkv_runtime_profile: str = ""
    layerkv_worker_role: str = ""
    layerkv_kvc_backend: str = ""
    layerkv_kvc_backend_semantics: str = ""
    layerkv_kvc_backend_limited: bool = False
    layerkv_kvc_backend_ready: bool = False
    layerkv_kvc_backend_reason: str = ""
    layerkv_expert_forward_hooks_enabled: bool = True
    layerkv_expert_collector_only: bool = False
    layerkv_no_pressure_fastpath_count: int = 0
    layerkv_no_pressure_expert_skip_count: int = 0
    layerkv_no_pressure_scheduler_skip_count: int = 0
    layerkv_no_pressure_kvc_skip_count: int = 0
    kvc_per_layer_metadata_rewrite_count: int = 0
    kvc_per_layer_metadata_rewrite_skip_count: int = 0
    kvc_per_layer_metadata_rewrite_unsupported_count: int = 0
    kvc_per_layer_override_layer_count: int = 0
    kvc_per_layer_identity_override_count: int = 0
    kvc_per_layer_slot_override_count: int = 0
    kvc_per_layer_slot_override_token_count: int = 0
    kvc_per_layer_slot_override_skip_count: int = 0
    kvc_per_layer_slot_override_skip_token_count: int = 0
    kvc_per_layer_arena_entry_count: int = 0
    kvc_per_layer_arena_resident_token_count: int = 0
    kvc_per_layer_arena_offloaded_token_count: int = 0
    kvc_per_layer_arena_capacity_mb: float = 0.0
    kvc_per_layer_arena_used_mb: float = 0.0
    kvc_per_layer_physical_arena_token_capacity: int = 0
    kvc_per_layer_physical_arena_min_free_tokens: int = 0
    kvc_per_layer_physical_arena_common_free_tokens: int = 0
    kvc_per_layer_physical_arena_alloc_count: int = 0
    kvc_per_layer_physical_arena_free_count: int = 0
    kvc_per_layer_physical_arena_grow_count: int = 0
    kvc_per_layer_physical_arena_native_reserved_tokens: int = 0
    kvc_per_layer_physical_arena_alloc_failed_count: int = 0
    kvc_per_layer_logical_token_count: int = 0
    kvc_per_layer_logical_request_count: int = 0
    kvc_per_layer_common_alloc_count: int = 0
    kvc_per_layer_independent_alloc_count: int = 0
    kvc_layerkv_owned_release_count: int = 0
    kvc_layerkv_owned_release_token_count: int = 0
    kvc_per_layer_evict_count: int = 0
    kvc_per_layer_reload_count: int = 0
    kvc_per_layer_reload_mb_total: float = 0.0
    kvc_per_layer_backup_ms: float = 0.0
    kvc_per_layer_reload_ms: float = 0.0
    kvc_workspace_pack_ms: float = 0.0
    kvc_workspace_reuse_count: int = 0
    kvc_workspace_invalidated_count: int = 0
    planned_kvc_reclaim_mb: float = 0.0
    physical_kvc_reclaim_mb: float = 0.0
    physical_kvc_reclaim_peak_mb: float = 0.0
    physical_kvc_reclaim_step_sum_mb: float = 0.0
    physical_kvc_reclaim_step_count: int = 0
    physical_kvc_reclaim_step_mean_mb: float = 0.0
    planned_expert_reclaim_mb: float = 0.0
    physical_expert_reclaim_mb: float = 0.0
    physical_total_reclaim_mb: float = 0.0
    physical_total_reclaim_peak_mb: float = 0.0
    expert_host_backing_mb: float = 0.0
    expert_resident_count: int = 0
    expert_offloaded_count: int = 0
    expert_slot_capacity_total: int = 0
    expert_slot_rebind_count: int = 0
    expert_materialize_count: int = 0
    expert_topk_rewrite_count: int = 0
    expert_topk_range_calibrated_count: int = 0
    expert_topk_range_fastpath_count: int = 0
    expert_topk_range_invalid_count: int = 0
    expert_core_hook_count: int = 0
    expert_hotness_record_count: int = 0
    expert_hotness_sample_skip_count: int = 0
    expert_hotness_snapshot_count: int = 0
    expert_hotness_snapshot_issue_count: int = 0
    expert_hotness_snapshot_ready_count: int = 0
    expert_hotness_snapshot_drop_count: int = 0
    expert_hotness_sync_fallback_count: int = 0
    expert_hotness_record_fast_count: int = 0
    expert_hotness_record_safe_count: int = 0
    expert_hotness_snapshot_deferred_count: int = 0
    expert_hotness_snapshot_forced_count: int = 0
    expert_hotness_ones_reuse_count: int = 0
    expert_candidate_snapshot_issue_count: int = 0
    expert_candidate_snapshot_ready_count: int = 0
    expert_candidate_snapshot_drop_count: int = 0
    expert_candidate_order_hit_count: int = 0
    expert_candidate_order_miss_count: int = 0
    expert_copy_descriptor_count: int = 0
    expert_copy_descriptor_d2h_count: int = 0
    expert_copy_descriptor_h2d_count: int = 0
    expert_copy_descriptor_param_count: int = 0
    expert_copy_descriptor_bytes: int = 0
    expert_copy_descriptor_install_count: int = 0
    expert_copy_descriptor_evict_count: int = 0
    expert_copy_descriptor_materialize_count: int = 0
    expert_d2h_slice_run_count: int = 0
    expert_d2h_slice_expert_count: int = 0
    expert_d2h_gather_batch_count: int = 0
    expert_d2h_gather_expert_count: int = 0
    expert_cpu_backing_pool_alloc_count: int = 0
    expert_cpu_backing_pool_reuse_count: int = 0
    expert_cpu_backing_pool_release_count: int = 0
    expert_cpu_backing_pool_drop_count: int = 0
    expert_cpu_backing_pool_bytes: int = 0
    expert_cpu_backing_pool_limit_bytes: int = 0
    expert_materialize_mb_total: float = 0.0
    expert_materialize_ms: float = 0.0
    expert_materialize_async_count: int = 0
    expert_materialize_host_sync_count: int = 0
    expert_materialize_batch_count: int = 0
    expert_materialize_event_count: int = 0
    expert_materialize_avg_batch_size: float = 0.0
    expert_materialize_batch_size_p50: float = 0.0
    expert_materialize_batch_size_p95: float = 0.0
    expert_materialize_layers_touched: int = 0
    expert_materialize_dedup_count: int = 0
    expert_materialize_coalesced_count: int = 0
    expert_materialize_slot_select_ms: float = 0.0
    expert_materialize_map_update_ms: float = 0.0
    expert_materialize_event_overhead_ms: float = 0.0
    expert_materialize_sync_wait_ms: float = 0.0
    expert_prefetch_count: int = 0
    expert_prefetch_hit_count: int = 0
    expert_prefetch_miss_count: int = 0
    expert_prefetch_candidate_count: int = 0
    expert_prefetch_issued_count: int = 0
    expert_prefetch_skipped_resident_count: int = 0
    expert_prefetch_skipped_capacity_count: int = 0
    expert_prefetch_useful_count: int = 0
    expert_prefetch_wasted_count: int = 0
    expert_on_demand_materialize_count: int = 0
    expert_prefetch_mb_total: float = 0.0
    expert_prepared_backing_mb: float = 0.0
    expert_prepare_extend_count: int = 0
    expert_prepare_decode_fallback_count: int = 0
    expert_prepared_plan_used: bool = False
    expert_prepared_backing_hit_count: int = 0
    expert_prepared_backing_miss_count: int = 0
    expert_prepare_guard_count: int = 0
    expert_prepare_guard_mb: float = 0.0
    expert_prepare_candidate_count: int = 0
    expert_prepare_copied_count: int = 0
    expert_prepare_reused_count: int = 0
    expert_prepare_invalidated_count: int = 0
    expert_prepare_host_budget_mb: float = 0.0
    expert_prepare_actual_mb: float = 0.0
    expert_prefetch_ready_before_use_count: int = 0
    expert_prefetch_blocked_by_copy_stream_count: int = 0
    expert_install_state: str = ""
    expert_install_pending_layers: int = 0
    expert_install_completed_layers: int = 0
    expert_install_layers_per_step: int = 0
    expert_install_budget_mb: float = 0.0
    expert_install_target_steps: int = 0
    expert_install_remaining_steps: int = 0
    expert_install_effective_layers_this_step: int = 0
    expert_install_effective_budget_mb_this_step: float = 0.0
    expert_install_step_count: int = 0
    expert_install_step_ms: float = 0.0
    expert_install_blocking_ms: float = 0.0
    expert_install_reclaim_mb_progress: float = 0.0
    expert_install_not_comparable_step_count: int = 0
    expert_install_d2h_async_count: int = 0
    expert_install_d2h_async_mb: float = 0.0
    expert_install_d2h_async_finalize_count: int = 0
    expert_install_d2h_async_wait_count: int = 0
    expert_install_d2h_sync_fallback_count: int = 0
    expert_install_d2h_queued_count: int = 0
    expert_install_d2h_queue_length: int = 0
    expert_install_d2h_submit_step_count: int = 0
    expert_install_d2h_budget_mb: float = 0.0
    expert_install_d2h_effective_budget_mb: float = 0.0
    expert_install_d2h_max_budget_mb: float = 0.0
    expert_install_d2h_chunk_mb: float = 0.0
    expert_install_d2h_lookahead_layers: int = 0
    expert_install_d2h_lookahead_queue_count: int = 0
    expert_install_d2h_dynamic_budget_count: int = 0
    expert_install_d2h_force_drain_count: int = 0
    expert_install_d2h_force_drain_ms: float = 0.0
    expert_lazy_backing_enabled: bool = True
    expert_lazy_backing_skipped_count: int = 0
    expert_lazy_backing_skipped_mb: float = 0.0
    expert_lazy_backing_unavailable_count: int = 0
    expert_backing_cache_limit_mb: float = 0.0
    expert_backing_cache_hit_count: int = 0
    expert_backing_cache_miss_count: int = 0
    expert_backing_cache_evict_count: int = 0
    expert_cpu_backing_mode: str = ""
    expert_cpu_backing_preload_count: int = 0
    expert_cpu_backing_preload_mb: float = 0.0
    expert_cpu_backing_preload_ms: float = 0.0
    expert_cpu_backing_global_hit_count: int = 0
    expert_cpu_backing_global_miss_count: int = 0
    expert_eviction_d2h_skip_count: int = 0
    expert_eviction_d2h_copy_count: int = 0
    expert_eviction_d2h_batch_count: int = 0
    expert_eviction_d2h_batched_count: int = 0
    expert_eviction_d2h_batched_mb: float = 0.0
    expert_eviction_d2h_fallback_count: int = 0
    expert_eviction_d2h_async_count: int = 0
    expert_eviction_d2h_async_mb: float = 0.0
    expert_eviction_d2h_async_wait_count: int = 0
    expert_eviction_d2h_async_finalize_count: int = 0
    expert_copy_stream_launch_count: int = 0
    expert_copy_stream_wait_count: int = 0
    expert_ready_before_use_count: int = 0
    expert_ready_use_check_count: int = 0
    expert_ready_before_use_ratio: float = 1.0
    expert_call_count_total: int = 0
    expert_prefill_call_count_total: int = 0
    expert_decode_call_count_total: int = 0
    expert_hotness_observed: bool = False
    expert_guard_pass: bool = True
    expert_guard_reason: str = ""
    expert_zero_reconstruct_guard_pass: bool = True
    expert_zero_reconstruct_guard_reason: str = ""
    expert_zero_reconstruct_violation_count: int = 0
    expert_terminal_slot_count: int = 0
    expert_terminal_offloaded_count: int = 0
    expert_terminal_metadata_mapped_count: int = 0
    kvc_host_backing_mb: float = 0.0
    kvc_host_capacity_tokens: int = 0
    kvc_host_used_tokens: int = 0
    kvc_page_size: int = 1
    kvc_offloaded_page_count: int = 0
    kvc_resident_page_count: int = 0
    kvc_reload_page_count_total: int = 0
    kvc_evict_page_count_total: int = 0
    kvc_page_alignment_violation_count: int = 0
    kvc_resident_token_count: int = 0
    kvc_offloaded_token_count: int = 0
    kvc_evict_count_total: int = 0
    kvc_reload_count_total: int = 0
    kvc_reload_mb_total: float = 0.0
    kvc_backup_ms: float = 0.0
    kvc_reload_ms: float = 0.0
    kvc_use_point_wait_ms: float = 0.0
    kvc_allocator_free_count: int = 0
    kvc_allocator_available_before: int = -1
    kvc_allocator_available_after: int = -1
    kvc_req_to_token_rewrite_count: int = 0
    kvc_physical_cycle_count: int = 0
    kvc_physical_failure_count: int = 0
    kvc_reload_required_count: int = 0
    kvc_eviction_skipped_count: int = 0
    kvc_residency_entry_count: int = 0
    kvc_stale_entry_count: int = 0
    kvc_finished_req_cleanup_count: int = 0
    kvc_finished_req_cleanup_token_count: int = 0
    kvc_evict_cursor_hit_count: int = 0
    kvc_evict_cursor_reset_count: int = 0
    kvc_evict_candidate_scan_tokens: int = 0
    kvc_evict_candidate_selected_tokens: int = 0
    kvc_guard_pass: bool = True
    kvc_guard_reason: str = ""
    kvc_zero_reconstruct_guard_pass: bool = True
    kvc_zero_reconstruct_guard_reason: str = ""
    kvc_zero_reconstruct_violation_count: int = 0
    kvc_terminal_resident_token_count: int = 0
    kvc_terminal_offloaded_token_count: int = 0
    kvc_terminal_metadata_mapped_token_count: int = 0
    planner_apply_count: int = 0
    scheduler_invocation_count: int = 0
    scheduler_task_count: int = 0
    scheduler_kvc_task_count: int = 0
    scheduler_expert_task_count: int = 0
    scheduler_coalesced_task_count: int = 0
    scheduler_deadline_miss_count: int = 0
    scheduler_ready_before_use_count: int = 0
    scheduler_ready_use_check_count: int = 0
    scheduler_ready_before_use_ratio: float = 1.0
    scheduler_exposed_wait_ms: float = 0.0
    scheduler_copy_bytes_total: int = 0
    layerkv_tasks_built: int = 0
    layerkv_kvc_reload_started: int = 0
    layerkv_expert_materialize_started: int = 0
    layerkv_deadline_miss_count: int = 0
    layerkv_copy_event_record_count: int = 0
    layerkv_copy_event_wait_count: int = 0
    kvc_ready_before_use_count: int = 0
    kvc_ready_use_check_count: int = 0
    kvc_ready_before_use_ratio: float = 1.0
    layerkv_main_stream_wait_ms: float = 0.0
    layerkv_copy_stream_busy_ms: float = 0.0
    layerkv_python_overhead_ms: float = 0.0
    unified_residency_enabled: bool = True
    resident_group_count: int = 0
    resident_group_kvc_count: int = 0
    resident_group_expert_count: int = 0
    resident_group_resident_count: int = 0
    resident_group_offloaded_count: int = 0
    resident_group_recovering_count: int = 0
    resident_group_recover_count: int = 0
    resident_group_wait_count: int = 0
    resident_group_state_error_count: int = 0
    resident_group_last_error: str = ""
    profile_detail_enabled: bool = False
    profile_install_ms: float = 0.0
    profile_set_kv_ms: float = 0.0
    profile_forward_begin_ms: float = 0.0
    profile_forward_end_ms: float = 0.0
    profile_workload_stats_ms: float = 0.0
    profile_apply_expert_plan_ms: float = 0.0
    profile_expert_prefetch_ms: float = 0.0
    profile_expert_prefetch_scan_ms: float = 0.0
    profile_recovery_task_build_ms: float = 0.0
    profile_recovery_task_schedule_ms: float = 0.0
    profile_kvc_reload_required_ms: float = 0.0
    profile_kvc_evict_to_target_ms: float = 0.0
    profile_kvc_select_required_ms: float = 0.0
    profile_kvc_select_evict_ms: float = 0.0
    profile_req_to_token_rewrite_ms: float = 0.0
    profile_virtual_select_ms: float = 0.0
    profile_virtual_index_build_ms: float = 0.0
    profile_virtual_req_to_token_scatter_ms: float = 0.0
    profile_virtual_metadata_rewrite_ms: float = 0.0
    profile_virtual_direct_metadata_patch_ms: float = 0.0
    profile_virtual_prefetch_issue_ms: float = 0.0
    profile_kvc_evict_staging_alloc_ms: float = 0.0
    profile_kvc_evict_commit_ms: float = 0.0
    profile_expert_unique_ms: float = 0.0
    profile_expert_hotness_record_ms: float = 0.0
    profile_expert_hotness_snapshot_ms: float = 0.0
    profile_expert_topk_rewrite_ms: float = 0.0
    profile_expert_materialize_control_ms: float = 0.0
    profile_expert_materialize_metadata_ms: float = 0.0
    profile_expert_materialize_choose_slot_ms: float = 0.0
    profile_expert_materialize_remap_ms: float = 0.0
    profile_expert_materialize_copy_issue_ms: float = 0.0
    profile_planner_dp_ms: float = 0.0
    profile_planner_dp_candidate_eval_ms: float = 0.0
    profile_planner_dp_expert_cost_ms: float = 0.0
    profile_planner_dp_kvc_cost_ms: float = 0.0
    profile_prepare_expert_backing_ms: float = 0.0
    profile_prepare_expert_backing_only_ms: float = 0.0
    profile_apply_prepared_expert_plan_ms: float = 0.0
    profile_apply_expert_slot_map_ms: float = 0.0
    profile_apply_expert_shrink_ms: float = 0.0
    profile_plan_expert_capacity_ms: float = 0.0
    profile_install_expert_slots_ms: float = 0.0
    profile_copy_expert_to_cpu_ms: float = 0.0
    profile_shrink_expert_weight_ms: float = 0.0
    profile_refresh_expert_stats_ms: float = 0.0
    profile_finalize_kvc_ms: float = 0.0
    profile_finalize_expert_ms: float = 0.0
    profile_summary_build_ms: float = 0.0
    profile_accounted_ms: float = 0.0
    profile_unaccounted_ms: float = 0.0
    profile_controller_per_decode_step_ms: float = 0.0
    native_scheduler_observation_count: int = 0
    native_schedule_policy: str = ""
    native_schedule_forward_mode: str = ""
    native_schedule_waiting_queue_len: int = 0
    native_schedule_running_batch_size: int = 0
    native_schedule_batch_size: int = 0
    native_schedule_max_running_requests: int = 0
    native_schedule_new_token_ratio: float = 0.0
    native_schedule_kv_available_tokens: int = -1
    native_schedule_overlap_enabled: bool = False
    observed_batch_size: int = 0
    avg_prefix_len: float = 0.0
    decode_steps: int = 0
    kvc_bytes_per_token_all_layers: int = 0
    comparable: bool = True
    comparability_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class _LayerKVResidencyEntry:
    req_idx: int
    pos: int
    state: str
    layer_id: int = -1
    device_loc: Optional[int] = None
    host_slot: Optional[int] = None
    device_locs: Optional[List[int]] = None
    host_slots: Optional[List[int]] = None
    evicted_device_locs: Optional[List[int]] = None
    page_size: int = 1
    ready_start_event: Optional[Any] = None
    ready_event: Optional[Any] = None
    ready_waited: bool = False
    last_access_step: int = 0

    @property
    def token_count(self) -> int:
        return max(1, int(self.page_size))

    def logical_positions(self) -> List[int]:
        return list(range(self.pos, self.pos + self.token_count))

    def device_loc_list(self) -> List[int]:
        if self.device_locs is not None:
            return [int(x) for x in self.device_locs]
        if self.device_loc is None:
            return []
        return [int(self.device_loc)]

    def host_slot_list(self) -> List[int]:
        if self.host_slots is not None:
            return [int(x) for x in self.host_slots]
        if self.host_slot is None:
            return []
        return [int(self.host_slot)]


@dataclasses.dataclass
class _LayerKVPendingReload:
    start_event: Any
    ready_event: Any
    entries: List[_LayerKVResidencyEntry]
    waited_on_main_stream: bool = False


@dataclasses.dataclass
class _LayerKVVirtualMaterializePlan:
    step: int
    selected: List[_LayerKVResidencyEntry]
    req_indices: Tuple[int, ...]
    positions: Tuple[int, ...]
    row_indices: Tuple[int, ...]
    flat_indices: Tuple[int, ...]
    flat_spans: Tuple[Tuple[int, int, int], ...]
    row_spans: Tuple[Tuple[int, int, int, int], ...]
    req_tensor: Optional[torch.Tensor]
    pos_tensor: Optional[torch.Tensor]
    row_tensor: Optional[torch.Tensor]
    flat_tensor: Optional[torch.Tensor]
    host_slots: Tuple[int, ...]
    host_signature: Tuple[int, int, int, int]
    host_slice: Tuple[int, int]
    host_index_cpu: Optional[torch.Tensor]
    max_row_index: int
    max_position: int
    max_flat_index: int
    token_count: int


@dataclasses.dataclass
class _LayerKVKvcDemand:
    layer_id: int
    entries: Tuple[_LayerKVResidencyEntry, ...]
    req_indices: Tuple[int, ...]
    positions: Tuple[int, ...]
    row_indices: Tuple[int, ...]
    flat_indices: Tuple[int, ...]
    flat_spans: Tuple[Tuple[int, int, int], ...]
    row_spans: Tuple[Tuple[int, int, int, int], ...]
    token_count: int
    deadline_layer: int
    benefit_score: float
    signature: str
    base_signature: str = ""
    backend_semantics: str = ""
    host_slots: Tuple[int, ...] = ()
    host_signature: Tuple[int, int, int, int] = (0, 0, 0, 0)
    host_slice: Tuple[int, int] = (-1, 0)
    host_index_cpu: Optional[torch.Tensor] = None
    max_row_index: int = -1
    max_position: int = -1
    max_flat_index: int = -1


@dataclasses.dataclass
class _LayerKVExpertDemand:
    layer_id: int
    logical_ids: Tuple[int, ...]
    bytes: int
    deadline_layer: int
    benefit_score: float
    signature: str
    group_keys: Tuple["_LayerKVResidencyKey", ...] = ()


@dataclasses.dataclass
class _LayerKVMetadataPatchCacheEntry:
    key: Tuple[Any, ...]
    row_tensor: Optional[torch.Tensor] = None
    pos_tensor: Optional[torch.Tensor] = None
    flat_tensor: Optional[torch.Tensor] = None
    scratch_tensor: Optional[torch.Tensor] = None
    flat_slice: Optional[Tuple[int, int]] = None
    flat_slice_checked: bool = False
    page_slice: Optional[Tuple[int, int, int]] = None
    page_slice_checked: bool = False
    hit_count: int = 0


@dataclasses.dataclass
class _LayerKVVirtualScratchCacheEntry:
    layer_id: int
    host_slots: Tuple[int, ...]
    host_signature: Tuple[int, int, int, int]
    host_slice: Tuple[int, int]
    host_index_cpu: Optional[torch.Tensor]
    token_count: int
    scratch_locs: torch.Tensor
    buffer_idx: int
    last_step: int


@dataclasses.dataclass
class _LayerKVPendingVirtualMaterialize:
    start_event: Any
    ready_event: Any
    layer_id: int
    entries: List[_LayerKVResidencyEntry]
    scratch_locs: torch.Tensor
    req_indices: Tuple[int, ...]
    positions: Tuple[int, ...]
    token_count: int
    buffer_idx: int
    waited_on_main_stream: bool = False


@dataclasses.dataclass
class _LayerKVPendingEviction:
    start_event: Any
    ready_event: Any
    entries: List[_LayerKVResidencyEntry]
    device_locs: torch.Tensor
    host_slots: List[int]
    k_staging: List[torch.Tensor]
    v_staging: List[torch.Tensor]
    token_count: int
    elapsed_recorded: bool = False


@dataclasses.dataclass
class _LayerKVPendingExpertCopy:
    start_event: Any
    ready_event: Any
    layer_id: int
    logical_ids: Set[int]
    waited_on_main_stream: bool = False


@dataclasses.dataclass
class _LayerKVPendingExpertD2H:
    start_event: Any
    ready_event: Any
    layer_id: int
    copied: Dict[int, Dict[str, torch.Tensor]]
    bytes: int
    reason: str = "evict_backing_async"
    source_refs: Tuple[torch.Tensor, ...] = ()
    target_cpu_params: Optional[Dict[int, Dict[str, torch.Tensor]]] = None


@dataclasses.dataclass
class _LayerKVExpertInstallD2HJob:
    seq: int
    layer_id: int
    module: Any
    param_names: List[str]
    expert_ids: List[int]
    target_cpu_params: Dict[int, Dict[str, torch.Tensor]]
    priority: int = 0
    deadline_step: int = 0


@dataclasses.dataclass(frozen=True)
class _LayerKVResidencyKey:
    kind: str
    layer_id: int
    logical_id: Any


@dataclasses.dataclass
class _LayerKVResidencyHandle:
    kind: str
    location: str
    ref: Any = None
    n_units: int = 1
    bytes: int = 0
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _LayerKVResidentTensorGroup:
    key: _LayerKVResidencyKey
    state: str
    bytes: int
    cpu_ref: Optional[_LayerKVResidencyHandle] = None
    gpu_ref: Optional[_LayerKVResidencyHandle] = None
    ready_start_event: Optional[Any] = None
    ready_event: Optional[Any] = None
    ready_waited: bool = False
    last_access_step: int = 0
    recover_count: int = 0
    wait_count: int = 0
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _LayerKVRecoveryTask:
    kind: str
    layer_id: int
    bytes: int
    deadline_layer: int
    demand_signature: str = ""
    kvc_demand: Optional[_LayerKVKvcDemand] = None
    expert_demand: Optional[_LayerKVExpertDemand] = None
    logical_ids: Tuple[int, ...] = ()
    group_keys: Tuple[_LayerKVResidencyKey, ...] = ()
    entries: Tuple[_LayerKVResidencyEntry, ...] = ()
    token_count: int = 0
    benefit_score: float = 0.0
    estimated_copy_ms: float = 0.0
    estimated_cpu_ms: float = 0.0
    state: str = "pending"


@dataclasses.dataclass
class _LayerKVExpertInstallItem:
    layer_id: int
    module: Any
    slot_capacity: int
    initial_resident: Optional[List[int]]
    prepared_cpu_params: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    backing_queued: bool = False
    pending_cpu_experts: Set[int] = dataclasses.field(default_factory=set)


@dataclasses.dataclass(frozen=True)
class _LayerKVExpertCopyDescriptor:
    seq: int
    direction: str
    reason: str
    layer_id: int
    logical_id: int
    src_slot: int
    dst_slot: int
    bytes: int
    param_count: int
    step: int


@dataclasses.dataclass
class _LayerKVExpertLayerState:
    layer_id: int
    module: Any
    orig_forward: Callable
    orig_run_moe_core: Callable
    full_num_experts: int
    slot_capacity: int
    expert_bytes: int
    device: torch.device
    dtype: torch.dtype
    cpu_params: Dict[int, Dict[str, torch.Tensor]]
    param_names: List[str]
    logical_to_slot: Dict[int, int]
    slot_to_logical: Dict[int, int]
    lru: Dict[int, int]
    hotness_prefill: Dict[int, int]
    hotness_decode: Dict[int, int]
    remap_tensor: Optional[torch.Tensor] = None
    free_slots: List[int] = dataclasses.field(default_factory=list)
    lru_heap: List[Tuple[int, int, int]] = dataclasses.field(default_factory=list)
    backing_lru: Dict[int, int] = dataclasses.field(default_factory=dict)
    last_decode_logical_ids: List[int] = dataclasses.field(default_factory=list)
    prefetched_logical_ids: Set[int] = dataclasses.field(default_factory=set)
    topk_ids_in_range_calibrated: bool = False
    topk_ids_invalid_observed: bool = False
    materialize_step: int = 0

    @property
    def full_bytes(self) -> int:
        return int(self.full_num_experts * self.expert_bytes)

    @property
    def resident_count(self) -> int:
        return len(self.logical_to_slot)

    @property
    def offloaded_count(self) -> int:
        return max(0, self.full_num_experts - self.resident_count)

    @property
    def physical_reclaim_bytes(self) -> int:
        return max(0, (self.full_num_experts - self.slot_capacity) * self.expert_bytes)


class _LayerKVHostKVStore:
    """Compact pinned host backing for LayerKV-owned evicted MHA KV tokens."""

    def __init__(
        self, kv_pool: Any, capacity_tokens: int, *, per_layer_mode: bool = False
    ):
        self.kv_pool = kv_pool
        self.capacity_tokens = max(1, int(capacity_tokens))
        self.per_layer_mode = bool(per_layer_mode)
        self.free_slots: List[int] = list(range(self.capacity_tokens))
        self.used_slots = set()
        self.layer_num = int(kv_pool.layer_num)
        self.start_layer = int(kv_pool.start_layer)
        self.device = kv_pool.device

        k0 = kv_pool._get_key_buffer(self.start_layer)
        v0 = kv_pool._get_value_buffer(self.start_layer)
        self.k_buffers = []
        self.v_buffers = []
        self.layer_capacities: List[int] = [0 for _ in range(self.layer_num)]
        self.layer_free_slots: List[List[int]] = [[] for _ in range(self.layer_num)]
        self.layer_used_slots: List[Set[int]] = [set() for _ in range(self.layer_num)]
        self.bytes_per_token_per_layer = int(k0[0].nbytes + v0[0].nbytes)
        self.bytes_per_token_all_layers = int(
            (k0[0].nbytes + v0[0].nbytes) * self.layer_num
        )
        if self.per_layer_mode:
            self.k_buffers = [None for _ in range(self.layer_num)]
            self.v_buffers = [None for _ in range(self.layer_num)]
            return
        for layer_offset in range(self.layer_num):
            layer_id = self.start_layer + layer_offset
            k_ref = kv_pool._get_key_buffer(layer_id)
            v_ref = kv_pool._get_value_buffer(layer_id)
            self.k_buffers.append(
                self._empty_cpu(
                    (self.capacity_tokens,) + tuple(k_ref.shape[1:]), k_ref.dtype
                )
            )
            self.v_buffers.append(
                self._empty_cpu(
                    (self.capacity_tokens,) + tuple(v_ref.shape[1:]), v_ref.dtype
                )
            )

    @staticmethod
    def _empty_cpu(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        try:
            return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        except Exception:
            return torch.empty(shape, dtype=dtype, device="cpu")

    @property
    def used_count(self) -> int:
        if self.per_layer_mode:
            return sum(len(slots) for slots in self.layer_used_slots)
        return len(self.used_slots)

    @property
    def used_mb(self) -> float:
        if self.per_layer_mode:
            return self.used_count * self.bytes_per_token_per_layer / float(1024 * 1024)
        return self.used_count * self.bytes_per_token_all_layers / float(1024 * 1024)

    @property
    def capacity_mb(self) -> float:
        if self.per_layer_mode:
            return (
                sum(self.layer_capacities)
                * self.bytes_per_token_per_layer
                / float(1024 * 1024)
            )
        return (
            self.capacity_tokens * self.bytes_per_token_all_layers / float(1024 * 1024)
        )

    def _ensure_layer_capacity(self, layer_offset: int, need_free: int) -> None:
        if not self.per_layer_mode or need_free <= len(
            self.layer_free_slots[layer_offset]
        ):
            return
        layer_id = self.start_layer + layer_offset
        k_ref = self.kv_pool._get_key_buffer(layer_id)
        v_ref = self.kv_pool._get_value_buffer(layer_id)
        old_capacity = self.layer_capacities[layer_offset]
        used = len(self.layer_used_slots[layer_offset])
        required = used + int(need_free)
        new_capacity = max(required, old_capacity * 2 if old_capacity > 0 else 0, 64)
        k_new = self._empty_cpu((new_capacity,) + tuple(k_ref.shape[1:]), k_ref.dtype)
        v_new = self._empty_cpu((new_capacity,) + tuple(v_ref.shape[1:]), v_ref.dtype)
        k_old = self.k_buffers[layer_offset]
        v_old = self.v_buffers[layer_offset]
        if old_capacity > 0 and k_old is not None and v_old is not None:
            k_new[:old_capacity].copy_(k_old[:old_capacity])
            v_new[:old_capacity].copy_(v_old[:old_capacity])
        self.k_buffers[layer_offset] = k_new
        self.v_buffers[layer_offset] = v_new
        self.layer_free_slots[layer_offset].extend(range(old_capacity, new_capacity))
        self.layer_capacities[layer_offset] = new_capacity

    def alloc_per_layer(
        self, entries: List["_LayerKVResidencyEntry"]
    ) -> Optional[List[int]]:
        if not self.per_layer_mode:
            return self.alloc(sum(entry.token_count for entry in entries))
        need_by_layer: Dict[int, int] = {}
        for entry in entries:
            layer_offset = int(entry.layer_id) - self.start_layer
            if layer_offset < 0 or layer_offset >= self.layer_num:
                return None
            need_by_layer[layer_offset] = need_by_layer.get(layer_offset, 0) + int(
                entry.token_count
            )
        for layer_offset, need in need_by_layer.items():
            self._ensure_layer_capacity(layer_offset, need)
        out: List[int] = []
        for entry in entries:
            layer_offset = int(entry.layer_id) - self.start_layer
            need = int(entry.token_count)
            free = self.layer_free_slots[layer_offset]
            slots = free[-need:]
            del free[-need:]
            self.layer_used_slots[layer_offset].update(slots)
            out.extend(slots)
        return out

    def alloc(self, need: int) -> Optional[List[int]]:
        if self.per_layer_mode:
            raise RuntimeError("use alloc_per_layer for per-layer KVC host store")
        if need > len(self.free_slots):
            return None
        slots = self.free_slots[-need:]
        del self.free_slots[-need:]
        self.used_slots.update(slots)
        return slots

    def free(self, slots: List[int]) -> None:
        if self.per_layer_mode:
            raise RuntimeError("use free_per_layer for per-layer KVC host store")
        if not slots:
            return
        for slot in slots:
            if slot in self.used_slots:
                self.used_slots.remove(slot)
                self.free_slots.append(slot)

    def free_per_layer(self, layer_id: int, slots: List[int]) -> None:
        if not self.per_layer_mode:
            self.free(slots)
            return
        if not slots:
            return
        layer_offset = int(layer_id) - self.start_layer
        if layer_offset < 0 or layer_offset >= self.layer_num:
            return
        used = self.layer_used_slots[layer_offset]
        free = self.layer_free_slots[layer_offset]
        for slot in slots:
            slot = int(slot)
            if slot in used:
                used.remove(slot)
                free.append(slot)

    def backup(self, device_locs: torch.Tensor, host_slots: List[int]) -> float:
        if device_locs.numel() == 0:
            return 0.0
        host_index = torch.tensor(host_slots, dtype=torch.int64, device="cpu")
        device_module = torch.get_device_module(self.device)
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        start.record()
        for layer_offset in range(self.layer_num):
            layer_id = self.start_layer + layer_offset
            k_src = (
                self.kv_pool._get_key_buffer(layer_id)[device_locs]
                .detach()
                .to("cpu", non_blocking=True)
            )
            v_src = (
                self.kv_pool._get_value_buffer(layer_id)[device_locs]
                .detach()
                .to("cpu", non_blocking=True)
            )
            self.k_buffers[layer_offset].index_copy_(0, host_index, k_src)
            self.v_buffers[layer_offset].index_copy_(0, host_index, v_src)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    def backup_to_staging_async(
        self, device_locs: torch.Tensor, stream: Optional[torch.cuda.Stream]
    ) -> Tuple[Optional[Any], Optional[Any], List[torch.Tensor], List[torch.Tensor]]:
        if self.per_layer_mode:
            raise RuntimeError("per-layer KVC eviction uses backup_per_layer")
        if device_locs.numel() == 0 or stream is None:
            return None, None, [], []
        device_module = torch.get_device_module(self.device)
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        k_staging: List[torch.Tensor] = []
        v_staging: List[torch.Tensor] = []
        with torch.cuda.stream(stream):
            start.record(stream)
            for layer_offset in range(self.layer_num):
                layer_id = self.start_layer + layer_offset
                k_src = self.kv_pool._get_key_buffer(layer_id)[device_locs].detach()
                v_src = self.kv_pool._get_value_buffer(layer_id)[device_locs].detach()
                k_dst = self._empty_cpu(tuple(k_src.shape), k_src.dtype)
                v_dst = self._empty_cpu(tuple(v_src.shape), v_src.dtype)
                k_dst.copy_(k_src, non_blocking=True)
                v_dst.copy_(v_src, non_blocking=True)
                k_staging.append(k_dst)
                v_staging.append(v_dst)
            end.record(stream)
        return start, end, k_staging, v_staging

    def commit_staging_backup(
        self,
        host_slots: List[int],
        k_staging: List[torch.Tensor],
        v_staging: List[torch.Tensor],
    ) -> None:
        if self.per_layer_mode:
            raise RuntimeError("per-layer KVC eviction uses backup_per_layer")
        if not host_slots:
            return
        host_index = torch.tensor(host_slots, dtype=torch.int64, device="cpu")
        for layer_offset in range(self.layer_num):
            if layer_offset >= len(k_staging) or layer_offset >= len(v_staging):
                raise RuntimeError("incomplete async KVC eviction staging buffers")
            self.k_buffers[layer_offset].index_copy_(
                0, host_index, k_staging[layer_offset]
            )
            self.v_buffers[layer_offset].index_copy_(
                0, host_index, v_staging[layer_offset]
            )

    def backup_per_layer(
        self, entries: List[_LayerKVResidencyEntry], host_slots: List[int]
    ) -> float:
        if not entries or not host_slots:
            return 0.0
        device_module = torch.get_device_module(self.device)
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        start.record()
        by_layer: Dict[int, Tuple[List[int], List[int]]] = {}
        offset = 0
        for entry in entries:
            layer_offset = int(entry.layer_id) - self.start_layer
            if layer_offset < 0 or layer_offset >= self.layer_num:
                raise RuntimeError(f"invalid per-layer KVC layer_id={entry.layer_id}")
            count = entry.token_count
            slots = host_slots[offset : offset + count]
            offset += count
            layer_device_locs, layer_host_slots = by_layer.setdefault(
                layer_offset, ([], [])
            )
            layer_device_locs.extend(entry.device_loc_list())
            layer_host_slots.extend(int(slot) for slot in slots)
        for layer_offset, (device_loc_list, host_slot_list) in by_layer.items():
            if not device_loc_list or not host_slot_list:
                continue
            layer_id = self.start_layer + layer_offset
            host_index = torch.tensor(host_slot_list, dtype=torch.int64, device="cpu")
            device_locs = torch.tensor(
                device_loc_list, dtype=torch.int64, device=self.device
            )
            k_src = (
                self.kv_pool._get_key_buffer(layer_id)[device_locs]
                .detach()
                .to("cpu", non_blocking=True)
            )
            v_src = (
                self.kv_pool._get_value_buffer(layer_id)[device_locs]
                .detach()
                .to("cpu", non_blocking=True)
            )
            self.k_buffers[layer_offset].index_copy_(0, host_index, k_src)
            self.v_buffers[layer_offset].index_copy_(0, host_index, v_src)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    def reload(
        self,
        host_slots: List[int],
        device_locs: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
        async_copy: bool = False,
    ) -> Tuple[float, Optional[Any], Optional[Any]]:
        if device_locs.numel() == 0:
            return 0.0, None, None
        host_index = torch.tensor(host_slots, dtype=torch.int64, device="cpu")
        device_module = torch.get_device_module(self.device)
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        active_stream = stream if async_copy and stream is not None else None

        def issue_copy() -> None:
            if active_stream is not None:
                start.record(active_stream)
            else:
                start.record()
            for layer_offset in range(self.layer_num):
                layer_id = self.start_layer + layer_offset
                k_src = (
                    self.k_buffers[layer_offset]
                    .index_select(0, host_index)
                    .to(self.device, non_blocking=True)
                )
                v_src = (
                    self.v_buffers[layer_offset]
                    .index_select(0, host_index)
                    .to(self.device, non_blocking=True)
                )
                self.kv_pool._get_key_buffer(layer_id).index_copy_(
                    0, device_locs, k_src
                )
                self.kv_pool._get_value_buffer(layer_id).index_copy_(
                    0, device_locs, v_src
                )
            if active_stream is not None:
                end.record(active_stream)
            else:
                end.record()

        if active_stream is not None:
            with torch.cuda.stream(active_stream):
                issue_copy()
            return 0.0, start, end

        issue_copy()
        end.synchronize()
        return float(start.elapsed_time(end)), start, end

    def reload_per_layer(
        self,
        entries: List[_LayerKVResidencyEntry],
        stream: Optional[torch.cuda.Stream] = None,
        async_copy: bool = False,
    ) -> Tuple[float, Optional[Any], Optional[Any]]:
        if not entries:
            return 0.0, None, None
        device_module = torch.get_device_module(self.device)
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        active_stream = stream if async_copy and stream is not None else None

        def issue_copy() -> None:
            if active_stream is not None:
                start.record(active_stream)
            else:
                start.record()
            by_layer: Dict[int, Tuple[List[int], List[int]]] = {}
            for entry in entries:
                layer_offset = int(entry.layer_id) - self.start_layer
                if layer_offset < 0 or layer_offset >= self.layer_num:
                    raise RuntimeError(
                        f"invalid per-layer KVC layer_id={entry.layer_id}"
                    )
                host_slots = entry.host_slot_list()
                device_locs_list = entry.device_loc_list()
                if not host_slots or not device_locs_list:
                    continue
                layer_host_slots, layer_device_locs = by_layer.setdefault(
                    layer_offset, ([], [])
                )
                layer_host_slots.extend(int(slot) for slot in host_slots)
                layer_device_locs.extend(int(loc) for loc in device_locs_list)
            for layer_offset, (host_slot_list, device_loc_list) in by_layer.items():
                if not host_slot_list or not device_loc_list:
                    continue
                layer_id = self.start_layer + layer_offset
                host_index = torch.tensor(
                    host_slot_list, dtype=torch.int64, device="cpu"
                )
                device_locs = torch.tensor(
                    device_loc_list, dtype=torch.int64, device=self.device
                )
                k_src = (
                    self.k_buffers[layer_offset]
                    .index_select(0, host_index)
                    .to(self.device, non_blocking=True)
                )
                v_src = (
                    self.v_buffers[layer_offset]
                    .index_select(0, host_index)
                    .to(self.device, non_blocking=True)
                )
                self.kv_pool._get_key_buffer(layer_id).index_copy_(
                    0, device_locs, k_src
                )
                self.kv_pool._get_value_buffer(layer_id).index_copy_(
                    0, device_locs, v_src
                )
            if active_stream is not None:
                end.record(active_stream)
            else:
                end.record()

        if active_stream is not None:
            with torch.cuda.stream(active_stream):
                issue_copy()
            return 0.0, start, end

        issue_copy()
        end.synchronize()
        return float(start.elapsed_time(end)), start, end

    def reload_layer_to_locs(
        self,
        layer_id: int,
        entries: List[_LayerKVResidencyEntry],
        device_locs: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
        async_copy: bool = False,
        host_index: Optional[torch.Tensor] = None,
        host_slice: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, Optional[Any], Optional[Any]]:
        if not entries or int(device_locs.numel()) == 0:
            return 0.0, None, None
        layer_offset = int(layer_id) - self.start_layer
        if layer_offset < 0 or layer_offset >= self.layer_num:
            raise RuntimeError(f"invalid virtual KVC layer_id={layer_id}")
        host_slice_obj: Optional[slice] = None
        if host_slice is not None and int(host_slice[0]) >= 0:
            first_slot = int(host_slice[0])
            token_count = int(host_slice[1])
            if token_count != int(device_locs.numel()):
                raise RuntimeError(
                    f"virtual KVC scratch reload mismatch: host={token_count} device={int(device_locs.numel())}"
                )
            host_slice_obj = slice(first_slot, first_slot + token_count)
        elif host_index is None:
            host_slots: List[int] = []
            for entry in entries:
                host_slots.extend(entry.host_slot_list())
            if len(host_slots) != int(device_locs.numel()):
                raise RuntimeError(
                    f"virtual KVC scratch reload mismatch: host={len(host_slots)} device={int(device_locs.numel())}"
                )
            host_index = torch.tensor(host_slots, dtype=torch.int64, device="cpu")
        elif int(host_index.numel()) != int(device_locs.numel()):
            raise RuntimeError(
                f"virtual KVC scratch reload mismatch: host={int(host_index.numel())} device={int(device_locs.numel())}"
            )
        device_module = torch.get_device_module(self.device)
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        active_stream = stream if async_copy and stream is not None else None
        if (
            host_slice_obj is None
            and host_index is not None
            and host_index.device.type == "cpu"
            and int(host_index.numel()) > 0
        ):
            first_slot = int(host_index[0])
            token_count = int(host_index.numel())
            last_slot = int(host_index[-1])
            if last_slot - first_slot + 1 == token_count:
                expected = torch.arange(
                    first_slot,
                    first_slot + token_count,
                    dtype=host_index.dtype,
                    device="cpu",
                )
                if bool(torch.equal(host_index, expected)):
                    host_slice_obj = slice(first_slot, first_slot + token_count)

        def issue_copy() -> None:
            if active_stream is not None:
                start.record(active_stream)
            else:
                start.record()
            if host_slice_obj is not None:
                k_src = self.k_buffers[layer_offset][host_slice_obj].to(
                    self.device, non_blocking=True
                )
                v_src = self.v_buffers[layer_offset][host_slice_obj].to(
                    self.device, non_blocking=True
                )
            else:
                k_src = (
                    self.k_buffers[layer_offset]
                    .index_select(0, host_index)
                    .to(self.device, non_blocking=True)
                )
                v_src = (
                    self.v_buffers[layer_offset]
                    .index_select(0, host_index)
                    .to(self.device, non_blocking=True)
                )
            self.kv_pool._get_key_buffer(layer_id).index_copy_(0, device_locs, k_src)
            self.kv_pool._get_value_buffer(layer_id).index_copy_(0, device_locs, v_src)
            if active_stream is not None:
                end.record(active_stream)
            else:
                end.record()

        if active_stream is not None:
            with torch.cuda.stream(active_stream):
                issue_copy()
            return 0.0, start, end

        issue_copy()
        end.synchronize()
        return float(start.elapsed_time(end)), start, end


class _LayerKVResidencyBackend:
    kind: str = "unknown"

    def __init__(self, runtime: "LayerKVRuntime"):
        self.runtime = runtime

    def recover(
        self, groups: List[_LayerKVResidentTensorGroup], stream: Any = None
    ) -> None:
        raise NotImplementedError

    def wait_ready(self, groups: List[_LayerKVResidentTensorGroup]) -> None:
        for group in groups:
            if group.ready_event is not None:
                group.wait_count += 1
                self.runtime.stats.resident_group_wait_count += 1

    def stats(self) -> Dict[str, Any]:
        return {"kind": self.kind}


class _LayerKVKVCResidencyBackend(_LayerKVResidencyBackend):
    kind = "kvc"

    def recover(
        self, groups: List[_LayerKVResidentTensorGroup], stream: Any = None
    ) -> None:
        self.runtime._reload_required_kvc(self.runtime._last_forward_batch)


class _LayerKVExpertResidencyBackend(_LayerKVResidencyBackend):
    kind = "expert"

    def recover(
        self, groups: List[_LayerKVResidentTensorGroup], stream: Any = None
    ) -> None:
        by_layer: Dict[int, List[int]] = {}
        for group in groups:
            by_layer.setdefault(int(group.key.layer_id), []).append(
                int(group.key.logical_id)
            )
        for layer_id, logical_ids in by_layer.items():
            state = self.runtime._expert_layers.get(int(layer_id))
            if state is not None:
                self.runtime._materialize_experts(state, logical_ids, reason="prefetch")


class LayerKVRuntime:
    """Flag-gated SGLang adapter for layer-aware residency.

    This adapter deliberately does not replace SGLang's KV tensors. It installs
    stable hooks around the KV pool and forward pass, and the supported physical
    path rewrites logical token-to-slot metadata after backing slots to CPU and
    reloading them into allocator-owned GPU slots.
    """

    def __init__(self, config: LayerKVConfig):
        self.config = config
        if str(self.config.runtime_profile) == "simple":
            self.config.expert_backing_cache_mb = 0.0
            self.config.expert_cpu_backing_mode = "none"
            self.config.expert_install_target_steps = 0
            self.config.expert_install_budget_mb = 0.0
        self.stats = LayerKVStats()
        self.stats.layerkv_runtime_profile = str(self.config.runtime_profile)
        self.stats.layerkv_worker_role = str(self.config.worker_role)
        self.stats.layerkv_kvc_backend = str(self.config.kvc_backend)
        self.stats.layerkv_expert_forward_hooks_enabled = bool(
            self.config.expert_forward_hooks
        )
        self.stats.layerkv_expert_collector_only = bool(
            self.config.expert_collector_only
        )
        if self.config.kvc_backend == "token-slot":
            self.stats.layerkv_kvc_backend_semantics = "global_token_slot_layer_average"
            self.stats.layerkv_kvc_backend_limited = True
            self.stats.layerkv_kvc_backend_ready = True
            self.stats.layerkv_kvc_backend_reason = ""
        elif self.config.kvc_backend == "virtual-arena":
            self.stats.layerkv_kvc_backend_semantics = "virtual_logical_token_arena_v1"
            self.stats.layerkv_kvc_backend_limited = False
            self.stats.layerkv_kvc_backend_ready = False
            self.stats.layerkv_kvc_backend_reason = "virtual scratch not allocated"
        else:
            self.stats.layerkv_kvc_backend_semantics = "per_layer_physical_arena_v1"
            self.stats.layerkv_kvc_backend_limited = False
            self.stats.layerkv_kvc_backend_ready = True
            self.stats.layerkv_kvc_backend_reason = ""
        self.installed = False
        self.physical_kvc_supported = False
        self.physical_expert_supported = False
        self.unsupported_reason = ""
        self._wrapped_methods: Dict[str, Callable] = {}
        self._copy_stream: Optional[torch.cuda.Stream] = None
        self._runner: Any = None
        self._kv_pool: Any = None
        self._allocator: Any = None
        self._req_to_token_pool: Any = None
        self._bytes_per_token_all_layers: int = 0
        self._page_size: int = 1
        self._host_store: Optional[_LayerKVHostKVStore] = None
        self._residency: Dict[Tuple[int, int], _LayerKVResidencyEntry] = {}
        self._per_layer_residency: Dict[
            Tuple[int, int, int], _LayerKVResidencyEntry
        ] = {}
        self._per_layer_offloaded_keys: Set[Tuple[int, int, int]] = set()
        self._per_layer_offloaded_keys_by_req: Dict[int, Set[Tuple[int, int, int]]] = {}
        self._per_layer_offloaded_keys_by_req_layer: Dict[
            Tuple[int, int], Set[Tuple[int, int, int]]
        ] = {}
        self._per_layer_offloaded_token_count_by_layer: Dict[int, int] = {}
        self._per_layer_offloaded_version: int = 0
        self._per_layer_offloaded_sorted_by_req: Dict[
            int, Tuple[int, Tuple[Tuple[int, int, int], ...]]
        ] = {}
        self._per_layer_offloaded_sorted_by_req_layer: Dict[
            Tuple[int, int], Tuple[int, Tuple[Tuple[int, int, int], ...]]
        ] = {}
        self._per_layer_offloaded_positions_by_req_layer: Dict[
            Tuple[int, int], Tuple[int, Tuple[int, ...]]
        ] = {}
        self._per_layer_offloaded_dirty_reqs: Set[int] = set()
        self._per_layer_offloaded_dirty_req_layers: Set[Tuple[int, int]] = set()
        self._resident_groups: Dict[
            _LayerKVResidencyKey, _LayerKVResidentTensorGroup
        ] = {}
        self._residency_backends: Dict[str, _LayerKVResidencyBackend] = {
            "kvc": _LayerKVKVCResidencyBackend(self),
            "expert": _LayerKVExpertResidencyBackend(self),
        }
        self._kvc_evict_cursors: Dict[int, int] = {}
        self._kvc_evict_cursors_by_layer: Dict[Tuple[int, int], int] = {}
        self._expert_layers: Dict[int, _LayerKVExpertLayerState] = {}
        self._expert_modules: List[Tuple[int, Any]] = []
        self._expert_hotness_prefill: Dict[int, Dict[int, int]] = {}
        self._expert_hotness_decode: Dict[int, Dict[int, int]] = {}
        self._expert_hotness_version: int = 0
        self._expert_hotness_gpu_prefill: Dict[int, torch.Tensor] = {}
        self._expert_hotness_gpu_decode: Dict[int, torch.Tensor] = {}
        self._expert_hotness_pending_snapshots: List[
            Tuple[str, int, torch.Tensor, Any, Optional[torch.Tensor]]
        ] = []
        self._expert_candidate_order_by_layer: Dict[int, List[int]] = {}
        self._expert_candidate_pending_snapshots: List[
            Tuple[int, torch.Tensor, Any, Optional[torch.Tensor]]
        ] = []
        self._expert_candidate_pending_layers: Set[int] = set()
        self._expert_candidate_last_snapshot_step: Dict[int, int] = {}
        self._expert_hotness_snapshot_streams: Dict[str, Any] = {}
        self._expert_hotness_pending_snapshot_keys: Set[Tuple[str, int]] = set()
        self._expert_hotness_last_snapshot_step: Dict[Tuple[str, int], int] = {}
        self._expert_hotness_ones_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self._expert_copy_descriptor_seq: int = 0
        self._expert_copy_descriptors_recent: List[_LayerKVExpertCopyDescriptor] = []
        self._expert_hotness_sampled_layers_step: int = -1
        self._expert_hotness_sampled_layers: Optional[Set[int]] = None
        self._expert_hotness_sample_skip_pending: int = 0
        self._expert_plan_applied: bool = False
        self._expert_install_queue: List[_LayerKVExpertInstallItem] = []
        self._expert_install_d2h_queue: List[_LayerKVExpertInstallD2HJob] = []
        self._expert_install_d2h_job_seq: int = 0
        self._expert_install_state: str = ""
        self._expert_install_target_mb: float = 0.0
        self._expert_install_layers_per_step: int = int(
            self.config.expert_install_layers_per_step
        )
        self._expert_install_budget_mb: float = float(
            self.config.expert_install_budget_mb
        )
        self._expert_install_target_steps: int = int(
            self.config.expert_install_target_steps
        )
        self._expert_host_backing_bytes: int = 0
        self._expert_cpu_backing_pool: Dict[
            Tuple[torch.dtype, Tuple[int, ...]], List[torch.Tensor]
        ] = {}
        self._expert_cpu_backing_pool_bytes: int = 0
        self._expert_cpu_backing_pool_limit_bytes: int = 256 * 1024 * 1024
        self.stats.expert_cpu_backing_pool_limit_bytes = int(
            self._expert_cpu_backing_pool_limit_bytes
        )
        self._expert_global_cpu_backing: Dict[
            Tuple[int, int], Dict[str, torch.Tensor]
        ] = {}
        self._expert_global_cpu_backing_bytes: int = 0
        self._expert_group_dirty_layers: Set[int] = set()
        self._prepared_expert_plan: Optional[Dict[str, Any]] = None
        self._expert_prepare_started: bool = False
        self._expert_prepare_done: bool = False
        self._last_scheduler_context: Dict[str, Any] = {}
        self._last_scheduled_req_lens: List[Tuple[int, int]] = []
        self._scheduler_pressure_tokens: int = 0
        self._scheduler_pressure_kvc_blocked: bool = False
        self._force_common_kvc_evict_tokens: int = 0
        self._current_forward_req_lens_batch_id: Optional[int] = None
        self._current_forward_req_lens: List[Tuple[int, int]] = []
        self._planned_kvc_token_target: int = 0
        self._planned_kvc_tokens_by_layer: Dict[int, int] = {}
        self._planned_expert_slot_capacities_by_layer: Dict[int, int] = {}
        self._planned_expert_cost_by_layer: Dict[int, float] = {}
        self._planned_expert_target_mb: float = 0.0
        self._planner_target_high_watermark_mb: float = 0.0
        self._coresid_plan_stats_signature: Optional[
            Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...], str]
        ] = None
        self._cached_policy_fractions: Optional[
            Tuple[float, float, bool, str, bool, str, float, float]
        ] = None
        self._cached_policy_target_bucket_mb: Optional[int] = None
        self._cached_policy_decode_bucket: Optional[int] = None
        self._cached_policy_hotness_version: Optional[int] = None
        self._cached_reclaim_target_key: Optional[Tuple[Any, ...]] = None
        self._cached_reclaim_target_value: float = 0.0
        self._cached_kvc_layer_ids_key: Optional[Tuple[int, int]] = None
        self._cached_kvc_layer_ids: List[int] = []
        self._pending_expert_copy_events: List[_LayerKVPendingExpertCopy] = []
        self._pending_expert_d2h_events: List[_LayerKVPendingExpertD2H] = []
        self._pending_expert_d2h_by_key: Dict[
            Tuple[int, int], _LayerKVPendingExpertD2H
        ] = {}
        self._pending_kvc_reload_events: List[_LayerKVPendingReload] = []
        self._pending_kvc_evict_events: List[_LayerKVPendingEviction] = []
        self._pending_virtual_kvc_materialize: Dict[
            Tuple[Any, ...], _LayerKVPendingVirtualMaterialize
        ] = {}
        self._per_layer_req_to_token_versions: Dict[int, int] = {}
        self._per_layer_kvc_current_metadata_key: Optional[Tuple[Any, ...]] = None
        self._virtual_materialize_plan: Optional[_LayerKVVirtualMaterializePlan] = None
        self._virtual_materialize_plans_by_layer: Dict[
            int, _LayerKVVirtualMaterializePlan
        ] = {}
        self._virtual_kvc_demands: Dict[Tuple[int, int], _LayerKVKvcDemand] = {}
        self._virtual_scratch_cache_by_layer: Dict[
            int, _LayerKVVirtualScratchCacheEntry
        ] = {}
        self._metadata_patch_cache: Dict[
            Tuple[Any, ...], _LayerKVMetadataPatchCacheEntry
        ] = {}
        self._metadata_patch_tensor_cache: Dict[
            Tuple[Any, ...], _LayerKVMetadataPatchCacheEntry
        ] = {}
        self._kvc_demand_signature_eval_step: Optional[int] = None
        self._per_layer_req_to_token_overrides: Dict[int, torch.Tensor] = {}
        self._per_layer_req_to_token_owned: Set[int] = set()
        self._per_layer_kvc_prepared_layers: Set[Tuple[int, int]] = set()
        self._per_layer_arena_reserved_locs: Set[int] = set()
        self._per_layer_arena_common_free_locs: Set[int] = set()
        self._per_layer_arena_common_free_order: List[int] = []
        self._per_layer_arena_free_locs: Dict[int, List[int]] = {}
        self._per_layer_arena_allocated_locs: Dict[int, Set[int]] = {}
        self._per_layer_arena_protected_locs: Dict[int, Set[int]] = {}
        self._per_layer_canonical_to_physical: Dict[int, Dict[int, int]] = {}
        self._per_layer_owned_req_indices: Set[int] = set()
        self._per_layer_owned_keys_by_req: Dict[int, Set[Tuple[int, int, int]]] = {}
        self._virtual_scratch_locs: Optional[torch.Tensor] = None
        self._virtual_scratch_buffers: List[torch.Tensor] = []
        self._virtual_scratch_capacity: int = 0
        self._expert_materialize_batch_sizes: List[int] = []
        self._expert_materialize_layers_touched: Set[int] = set()
        self._expert_prefetch_dirty_layers: Set[int] = set()
        self._per_layer_resident_token_count_fast: int = 0
        self._per_layer_offloaded_token_count_fast: int = 0
        self._kvc_reload_ms_per_mb_ewma_by_layer: Dict[int, float] = {}
        self._kvc_evict_ms_per_mb_ewma_by_layer: Dict[int, float] = {}
        self._current_forward_mode: str = ""
        self._last_forward_batch: Any = None
        self._decode_step: int = 0

    def _joint_policy_cache_bucket(self, target_mb: float) -> int:
        # Trace-driven pressure jitters by tens of MB across adjacent decode
        # steps. Replanning on every small jitter dominated runtime; bucket by a
        # coarse physical-pressure class while preserving large workload shifts.
        return int(round(max(0.0, float(target_mb)) / 256.0))

    def _joint_policy_decode_bucket(self) -> int:
        # Expert/KVC repeated-recovery cost changes slowly with decode horizon.
        # Keep a bounded replan cadence for changing hotness without paying DP
        # on every decode step.
        return int(max(0, self._decode_step) // 64)

    def _expert_reclaim_quantum_mb(self) -> float:
        bytes_by_layer: List[int] = []
        if self._expert_layers:
            bytes_by_layer.extend(
                int(state.expert_bytes)
                for state in self._expert_layers.values()
                if int(state.expert_bytes) > 0
            )
        else:
            for _layer_id, module in self._expert_modules:
                try:
                    expert_bytes = int(self._expert_bytes(module))
                except Exception:
                    expert_bytes = 0
                if expert_bytes > 0:
                    bytes_by_layer.append(expert_bytes)
        if not bytes_by_layer:
            return 0.0
        return min(bytes_by_layer) / float(1024 * 1024)

    def _dynamic_expert_replan_needed(self, forward_batch: Any) -> Tuple[bool, float]:
        if not self._dynamic_expert_churn_policy_enabled():
            return True, self._refresh_reclaim_target_stats(forward_batch)
        high_watermark = max(
            float(self._planner_target_high_watermark_mb),
            float(self._planned_expert_target_mb),
            float(self._expert_install_target_mb),
        )
        target_mb = self._refresh_reclaim_target_stats(forward_batch)
        if target_mb <= 1e-3:
            return False, target_mb
        quantum_mb = max(1e-3, self._expert_reclaim_quantum_mb())
        if target_mb <= high_watermark + quantum_mb:
            return False, target_mb
        return True, target_mb

    @classmethod
    def maybe_create(cls, server_args: Any) -> Optional["LayerKVRuntime"]:
        disaggregation_mode = str(getattr(server_args, "disaggregation_mode", "null"))
        if disaggregation_mode == "prefill":
            if bool(getattr(server_args, "enable_layerkv", False)):
                logger.warning(
                    "LayerKV is disabled on PD prefill workers; enable it on the "
                    "PD decode worker or standalone worker instead."
                )
            return None
        config = LayerKVConfig.from_server_args(server_args)
        if not config.enabled or config.mode == "off":
            return None
        return cls(config)

    @contextlib.contextmanager
    def _profile(self, field: str):
        if not self.config.profile_detail:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._add_profile(field, (time.perf_counter() - t0) * 1000.0)

    def _add_profile(self, field: str, elapsed_ms: float) -> None:
        if not self.config.profile_detail:
            return
        try:
            setattr(self.stats, field, float(getattr(self.stats, field)) + elapsed_ms)
        except Exception:
            pass

    def _refresh_expert_host_backing_stat(self) -> None:
        total_bytes = max(0, self._expert_host_backing_bytes) + max(
            0, self._expert_global_cpu_backing_bytes
        )
        self.stats.expert_host_backing_mb = total_bytes / float(1024 * 1024)
        self.stats.expert_backing_cache_limit_mb = float(
            self._effective_expert_backing_cache_mb()
        )
        self.stats.expert_cpu_backing_mode = str(self.config.expert_cpu_backing_mode)
        self.stats.expert_cpu_backing_preload_mb = max(
            0, self._expert_global_cpu_backing_bytes
        ) / float(1024 * 1024)
        self.stats.layerkv_runtime_profile = str(self.config.runtime_profile)
        self.stats.layerkv_worker_role = str(self.config.worker_role)

    def _optimized_profile_enabled(self) -> bool:
        return str(self.config.runtime_profile) == "optimized"

    def _simple_profile_enabled(self) -> bool:
        return str(self.config.runtime_profile) == "simple"

    def _effective_expert_backing_cache_mb(self) -> float:
        if self._simple_profile_enabled():
            return 0.0
        return float(max(0.0, self.config.expert_backing_cache_mb))

    def _policy_name(self) -> str:
        policy = str(self.config.policy)
        return "layer-aware-joint-dp" if policy == "coresid" else policy

    def _coresid_optimized_policy_enabled(self) -> bool:
        return (
            self.config.mode == "kvc-expert"
            and self._optimized_profile_enabled()
            and self._policy_name() in {"layer-aware-joint", "layer-aware-joint-dp"}
        )

    def _dynamic_expert_churn_policy_enabled(self) -> bool:
        return (
            self.config.dynamic_pressure_from_kvc
            and self.config.mode == "kvc-expert"
            and self._policy_name()
            in {"kv-first", "layer-aware-joint", "layer-aware-joint-dp"}
        )

    def _prepared_expert_backing_enabled(self) -> bool:
        return str(self.config.expert_cpu_backing_mode) in {"prepared", "all"}

    def _expert_group_tracking_enabled(self) -> bool:
        # ResidentTensorGroup remains the lifecycle abstraction, but per-expert
        # group bookkeeping is too expensive for the optimized online path.
        # Keep aggregate stats there and reserve detailed group maps for debug /
        # validation modes.
        return not (
            self._coresid_optimized_policy_enabled()
            and self.config.dynamic_pressure_from_kvc
        )

    def _refresh_profile_derived(self) -> None:
        self.stats.profile_detail_enabled = bool(self.config.profile_detail)
        if self._expert_materialize_batch_sizes:
            sizes = sorted(int(x) for x in self._expert_materialize_batch_sizes)
            n = len(sizes)
            p50_idx = min(n - 1, int(0.50 * (n - 1)))
            p95_idx = min(n - 1, int(0.95 * (n - 1)))
            self.stats.expert_materialize_batch_size_p50 = float(sizes[p50_idx])
            self.stats.expert_materialize_batch_size_p95 = float(sizes[p95_idx])
        self.stats.expert_materialize_layers_touched = len(
            self._expert_materialize_layers_touched
        )
        if not self.config.profile_detail:
            return
        # Use non-nested buckets for accounted controller time. Detailed child
        # buckets explain the top-level forward begin/end totals separately.
        accounted = (
            self.stats.profile_install_ms
            + self.stats.profile_set_kv_ms
            + self.stats.profile_forward_begin_ms
            + self.stats.profile_forward_end_ms
            + self.stats.profile_summary_build_ms
        )
        self.stats.profile_accounted_ms = accounted
        self.stats.profile_unaccounted_ms = (
            self.stats.layerkv_python_overhead_ms - accounted
        )
        self.stats.profile_controller_per_decode_step_ms = (
            self.stats.layerkv_python_overhead_ms
            / float(max(1, int(self.stats.decode_steps)))
        )

    def on_schedule_batch(
        self, *, schedule_batch: Any, scheduler_context: Dict[str, Any]
    ) -> None:
        """Observe SGLang's native request-level scheduling decision.

        LayerKV keeps request admission, priority ordering, prefill/decode
        selection, and retraction owned by SGLang's Scheduler. This hook only
        records read-only context for residency policy and diagnostics.
        """
        if schedule_batch is None:
            return
        context = dict(scheduler_context or {})
        self._last_scheduler_context = context
        self._last_scheduled_req_lens = self._schedule_batch_req_lens(schedule_batch)

        self.stats.native_scheduler_observation_count += 1
        self.stats.native_schedule_policy = str(context.get("schedule_policy", ""))
        self.stats.native_schedule_forward_mode = str(
            context.get("forward_mode")
            or getattr(getattr(schedule_batch, "forward_mode", None), "name", "")
        )
        self.stats.native_schedule_waiting_queue_len = int(
            context.get("waiting_queue_len", 0) or 0
        )
        self.stats.native_schedule_running_batch_size = int(
            context.get("running_batch_size", 0) or 0
        )
        try:
            self.stats.native_schedule_batch_size = int(schedule_batch.batch_size())
        except Exception:
            self.stats.native_schedule_batch_size = len(
                getattr(schedule_batch, "reqs", []) or []
            )
        self.stats.native_schedule_max_running_requests = int(
            context.get("max_running_requests", 0) or 0
        )
        self.stats.native_schedule_new_token_ratio = float(
            context.get("new_token_ratio", 0.0) or 0.0
        )
        self.stats.native_schedule_kv_available_tokens = int(
            context.get("kv_available_tokens", -1)
        )
        self.stats.native_schedule_overlap_enabled = bool(
            context.get("enable_overlap", False)
        )
        self._record_scheduler_budget_observation(context, self._last_scheduled_req_lens)

    def _record_scheduler_budget_observation(
        self, context: Dict[str, Any], req_lens: List[Tuple[int, int]]
    ) -> None:
        running_reqs = int(context.get("running_batch_size", 0) or 0)
        if running_reqs <= 0:
            running_reqs = len(req_lens)
        token_count = int(context.get("kv_used_tokens", 0) or 0)
        if token_count <= 0:
            token_count = sum(max(0, int(seq_len)) for _req_idx, seq_len in req_lens)
        token_usage = float(context.get("kv_token_usage", 0.0) or 0.0)
        retracted = int(context.get("num_retracted_reqs", 0) or 0)

        self.stats.scheduler_budget_observation_count += 1
        count = max(1, int(self.stats.scheduler_budget_observation_count))
        self.stats.scheduler_running_req_sum += running_reqs
        self.stats.scheduler_running_req_avg = (
            self.stats.scheduler_running_req_sum / float(count)
        )
        self.stats.scheduler_running_req_max = max(
            int(self.stats.scheduler_running_req_max), running_reqs
        )
        self.stats.scheduler_token_sum += token_count
        self.stats.scheduler_token_avg = self.stats.scheduler_token_sum / float(count)
        self.stats.scheduler_token_max = max(
            int(self.stats.scheduler_token_max), token_count
        )
        self.stats.scheduler_token_usage_sum += token_usage
        self.stats.scheduler_token_usage_avg = (
            self.stats.scheduler_token_usage_sum / float(count)
        )
        self.stats.scheduler_token_usage_max = max(
            float(self.stats.scheduler_token_usage_max), token_usage
        )
        self.stats.scheduler_retracted_req_current = retracted
        self.stats.scheduler_retracted_req_sum += retracted
        self.stats.scheduler_retracted_req_max = max(
            int(self.stats.scheduler_retracted_req_max), retracted
        )

    def _schedule_batch_req_lens(self, schedule_batch: Any) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for req in getattr(schedule_batch, "reqs", []) or []:
            req_pool_idx = getattr(req, "req_pool_idx", None)
            if req_pool_idx is None:
                continue
            try:
                req_idx = int(req_pool_idx)
            except (TypeError, ValueError):
                continue
            seq_len = getattr(req, "seq_len", None)
            if seq_len is None:
                seq_len = len(getattr(req, "origin_input_ids", []) or []) + len(
                    getattr(req, "output_ids", []) or []
                )
            try:
                pairs.append((req_idx, int(seq_len)))
            except (TypeError, ValueError):
                continue
        return pairs

    def _scheduler_credit_tokens(self, *, reason: str = "") -> Tuple[int, int, str]:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return 0, 0, "layerkv_mode_without_kvc"
        if not self.physical_kvc_supported:
            return 0, 0, "physical_kvc_unsupported"
        raw_offloaded = int(max(0, self._offloaded_token_count()))
        if self.config.kvc_backend == "per-layer-arena":
            layer_ids = self._kvc_layer_ids()
            if not layer_ids:
                return 0, raw_offloaded, "no_kvc_layers"
            self._refresh_per_layer_allocator_stats()
            if reason == "decode_prealloc_admission":
                credit = int(
                    self.stats.kvc_per_layer_physical_arena_common_free_tokens
                )
            else:
                credit = int(self.stats.kvc_per_layer_physical_arena_min_free_tokens)
            return max(0, credit), raw_offloaded, "per_layer_arena_physical_allocator"
        return raw_offloaded, raw_offloaded, ""

    def _kvc_reclaim_is_scheduler_visible(self) -> bool:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return False
        if not self.physical_kvc_supported:
            return False
        return self.config.kvc_backend != "per-layer-arena" or bool(
            self._per_layer_allocator_enabled()
        )

    def get_scheduler_admission_credit_tokens(
        self,
        *,
        required_tokens: int = 0,
        available_tokens: int = 0,
        reason: str = "",
    ) -> int:
        """Return scheduler-visible prefix KV credit without triggering reclaim."""
        required = max(0, int(required_tokens or 0))
        available = max(0, int(available_tokens or 0))
        shortage = max(0, required - available)
        credit, raw_offloaded, limit_reason = self._scheduler_credit_tokens(
            reason=reason
        )
        effective = available + credit
        if shortage <= 0:
            self._scheduler_pressure_tokens = 0
        self.stats.scheduler_budget_required_tokens = required
        self.stats.scheduler_budget_native_available_tokens = available
        self.stats.scheduler_budget_shortage_tokens = shortage
        self.stats.scheduler_budget_pressure_tokens = int(max(0, shortage - credit))
        self.stats.scheduler_budget_credit_tokens = credit
        self.stats.scheduler_budget_raw_offloaded_tokens = raw_offloaded
        self.stats.scheduler_budget_releasable_tokens = credit
        self.stats.scheduler_budget_credit_limit_reason = limit_reason
        self.stats.scheduler_budget_effective_available_tokens = effective
        if shortage > credit and reason:
            self.stats.scheduler_budget_credit_denied_count += 1
        return credit

    def prepare_reclaim_for_scheduler(
        self,
        *,
        schedule_batch: Any,
        required_tokens: int,
        available_tokens: int,
        reason: str = "scheduler_pressure",
        wait: bool = True,
    ) -> int:
        """Reclaim KVC for real scheduler pressure and return committed credit."""
        visible_available = int(available_tokens)
        if self.config.kvc_backend == "per-layer-arena":
            self._refresh_per_layer_allocator_stats()
            if reason == "decode_prealloc_admission":
                visible_available = int(available_tokens) + int(
                    self.stats.kvc_per_layer_physical_arena_common_free_tokens
                )
            else:
                visible_available = int(
                    self.stats.kvc_per_layer_physical_arena_min_free_tokens
                )
        shortage = max(0, int(required_tokens) - int(visible_available))
        self._scheduler_pressure_tokens = shortage
        self._scheduler_pressure_kvc_blocked = False
        self.stats.scheduler_budget_pressure_tokens = shortage
        if shortage <= 0:
            self.get_scheduler_admission_credit_tokens(
                required_tokens=required_tokens,
                available_tokens=available_tokens,
                reason=reason,
            )
            return self.stats.scheduler_budget_credit_tokens
        if (
            self.config.kvc_backend == "per-layer-arena"
            and reason in ("decode_prealloc_admission", "pre_retract_decode_mem")
        ):
            # Attention needs every running request's KV on every decode step.
            # Reclaiming the current running batch here creates immediate
            # evict/reload churn and does not provide durable scheduler capacity.
            # Online pressure traces show this path can spend scheduler time
            # while native retraction still reports #new_tokens_gained=0.
            # Only expose already-free arena slots to admission; deeper KVC
            # reclaim should be planned outside the current decode deadline.
            self.stats.kvc_layerwise_scheduler_deadline_reject_count += 1
            self._scheduler_pressure_kvc_blocked = True
            return self.get_scheduler_admission_credit_tokens(
                required_tokens=required_tokens,
                available_tokens=available_tokens,
                reason=reason,
            )
        block_tokens = max(1, int(getattr(self.config, "kvc_block_tokens", 1) or 1))
        if shortage < block_tokens:
            self.stats.scheduler_budget_small_shortage_skip_count += 1
            credit = self.get_scheduler_admission_credit_tokens(
                required_tokens=required_tokens,
                available_tokens=available_tokens,
                reason=reason,
            )
            self._scheduler_pressure_tokens = 0
            self._scheduler_pressure_kvc_blocked = False
            self.stats.scheduler_budget_pressure_tokens = 0
            return credit
        if not self._kvc_reclaim_is_scheduler_visible():
            self.stats.kvc_scheduler_invisible_skip_count += 1
            self.stats.kvc_scheduler_invisible_skip_tokens += int(shortage)
            credit = self.get_scheduler_admission_credit_tokens(
                required_tokens=required_tokens,
                available_tokens=available_tokens,
                reason=reason,
            )
            self._scheduler_pressure_tokens = 0
            self._scheduler_pressure_kvc_blocked = False
            self.stats.scheduler_budget_pressure_tokens = 0
            return credit
        before_credit, _, _ = self._scheduler_credit_tokens(reason=reason)
        t0 = time.perf_counter()
        self.try_reclaim_kvc_before_retract(
            schedule_batch=schedule_batch,
            required_tokens=required_tokens,
            available_tokens=available_tokens,
            reason=reason,
        )
        if wait:
            self._finalize_kvc_evictions(block=True)
        self.stats.scheduler_budget_pre_retract_wait_ms += (
            time.perf_counter() - t0
        ) * 1000.0
        credit = self.get_scheduler_admission_credit_tokens(
            required_tokens=required_tokens,
            available_tokens=available_tokens,
            reason=reason,
        )
        self._scheduler_pressure_tokens = max(0, shortage - credit)
        self._scheduler_pressure_kvc_blocked = False
        self.stats.scheduler_budget_pressure_tokens = int(self._scheduler_pressure_tokens)
        success = available_tokens + credit >= required_tokens
        if success and credit > before_credit:
            self.stats.scheduler_budget_credit_prevented_retract_count += 1
            self.stats.scheduler_budget_credit_used_tokens += min(shortage, credit)
        return credit

    def _residency_key(
        self, kind: str, layer_id: int, logical_id: Any
    ) -> _LayerKVResidencyKey:
        return _LayerKVResidencyKey(str(kind), int(layer_id), logical_id)

    def _get_or_create_resident_group(
        self,
        *,
        kind: str,
        layer_id: int,
        logical_id: Any,
        state: str,
        bytes: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> _LayerKVResidentTensorGroup:
        key = self._residency_key(kind, layer_id, logical_id)
        group = self._resident_groups.get(key)
        if group is None:
            group = _LayerKVResidentTensorGroup(
                key=key,
                state=str(state),
                bytes=int(max(0, bytes)),
                last_access_step=self._decode_step,
                metadata=dict(metadata or {}),
            )
            self._resident_groups[key] = group
        else:
            group.state = str(state)
            group.bytes = int(max(0, bytes))
            group.last_access_step = self._decode_step
            if metadata:
                group.metadata.update(metadata)
        return group

    def _sync_kvc_group(
        self, entry: _LayerKVResidencyEntry
    ) -> _LayerKVResidentTensorGroup:
        layer_id = int(entry.layer_id)
        key_logical = (
            (layer_id, int(entry.req_idx), int(entry.pos))
            if layer_id >= 0
            else (int(entry.req_idx), int(entry.pos))
        )
        bytes_per_token = (
            self._bytes_per_kvc_token_per_layer()
            if layer_id >= 0
            else self._bytes_per_token_all_layers
        )
        group = self._get_or_create_resident_group(
            kind="kvc",
            layer_id=layer_id,
            logical_id=key_logical,
            state=entry.state,
            bytes=entry.token_count * bytes_per_token,
            metadata={
                "layer_id": layer_id,
                "req_idx": int(entry.req_idx),
                "pos": int(entry.pos),
                "token_count": int(entry.token_count),
                "logical_positions": entry.logical_positions(),
            },
        )
        group.gpu_ref = (
            _LayerKVResidencyHandle(
                kind="kvc",
                location="gpu",
                ref=entry.device_loc_list(),
                n_units=entry.token_count,
                bytes=group.bytes,
                metadata={"device_locs": entry.device_loc_list()},
            )
            if entry.device_loc_list()
            else None
        )
        group.cpu_ref = (
            _LayerKVResidencyHandle(
                kind="kvc",
                location="cpu",
                ref=entry.host_slot_list(),
                n_units=entry.token_count,
                bytes=group.bytes,
                metadata={"host_slots": entry.host_slot_list()},
            )
            if entry.host_slot_list()
            else None
        )
        group.ready_start_event = entry.ready_start_event
        group.ready_event = entry.ready_event
        group.ready_waited = bool(entry.ready_waited)
        return group

    def _sync_kvc_group_if_needed(
        self, entry: _LayerKVResidencyEntry
    ) -> Optional[_LayerKVResidentTensorGroup]:
        if (
            self.config.kvc_backend == "per-layer-arena"
            and self.config.runtime_profile == "optimized"
        ):
            return None
        return self._sync_kvc_group(entry)

    def _remove_resident_group(self, kind: str, layer_id: int, logical_id: Any) -> None:
        self._resident_groups.pop(self._residency_key(kind, layer_id, logical_id), None)

    def _sync_expert_groups_for_state(self, state: _LayerKVExpertLayerState) -> None:
        if not self._expert_group_tracking_enabled():
            self._expert_group_dirty_layers.discard(int(state.layer_id))
            return
        for expert_id in range(int(state.full_num_experts)):
            slot_id = state.logical_to_slot.get(int(expert_id))
            is_resident = slot_id is not None
            group = self._get_or_create_resident_group(
                kind="expert",
                layer_id=state.layer_id,
                logical_id=int(expert_id),
                state="resident" if is_resident else "offloaded",
                bytes=state.expert_bytes,
                metadata={
                    "expert_id": int(expert_id),
                    "slot_capacity": int(state.slot_capacity),
                    "full_num_experts": int(state.full_num_experts),
                },
            )
            group.gpu_ref = (
                _LayerKVResidencyHandle(
                    kind="expert",
                    location="gpu",
                    ref=int(slot_id),
                    n_units=1,
                    bytes=state.expert_bytes,
                    metadata={"slot_id": int(slot_id)},
                )
                if is_resident
                else None
            )
            cpu_params = state.cpu_params.get(
                int(expert_id)
            ) or self._expert_global_cpu_backing.get(
                (int(state.layer_id), int(expert_id))
            )
            group.cpu_ref = (
                _LayerKVResidencyHandle(
                    kind="expert",
                    location="cpu",
                    ref=cpu_params,
                    n_units=1,
                    bytes=state.expert_bytes,
                    metadata={"expert_id": int(expert_id)},
                )
                if cpu_params is not None
                else None
            )
        self._expert_group_dirty_layers.discard(int(state.layer_id))

    def _sync_dirty_expert_groups(self) -> None:
        if not self._expert_group_dirty_layers:
            return
        if not self._expert_group_tracking_enabled():
            self._expert_group_dirty_layers.clear()
            return
        for layer_id in list(self._expert_group_dirty_layers):
            state = self._expert_layers.get(int(layer_id))
            if state is not None:
                self._sync_expert_groups_for_state(state)

    def _mark_expert_group_state(
        self,
        state: _LayerKVExpertLayerState,
        logical_id: int,
        *,
        group_state: str,
        slot_id: Optional[int] = None,
        cpu_params: Optional[Dict[str, torch.Tensor]] = None,
        ready_start_event: Any = None,
        ready_event: Any = None,
    ) -> None:
        if not self._expert_group_tracking_enabled():
            return
        group = self._get_or_create_resident_group(
            kind="expert",
            layer_id=state.layer_id,
            logical_id=int(logical_id),
            state=group_state,
            bytes=state.expert_bytes,
            metadata={"expert_id": int(logical_id)},
        )
        if slot_id is not None:
            group.gpu_ref = _LayerKVResidencyHandle(
                kind="expert",
                location="gpu",
                ref=int(slot_id),
                n_units=1,
                bytes=state.expert_bytes,
                metadata={"slot_id": int(slot_id)},
            )
        elif group_state == "offloaded":
            group.gpu_ref = None
        if cpu_params is None:
            cpu_params = state.cpu_params.get(
                int(logical_id)
            ) or self._expert_global_cpu_backing.get(
                (int(state.layer_id), int(logical_id))
            )
        if cpu_params is not None:
            group.cpu_ref = _LayerKVResidencyHandle(
                kind="expert",
                location="cpu",
                ref=cpu_params,
                n_units=1,
                bytes=state.expert_bytes,
                metadata={"expert_id": int(logical_id)},
            )
        group.ready_start_event = ready_start_event
        group.ready_event = ready_event
        group.ready_waited = False
        group.last_access_step = self._decode_step

    def _refresh_resident_group_stats(self) -> None:
        if not self._expert_group_tracking_enabled():
            if (
                self.config.kvc_backend == "per-layer-arena"
                and self._coresid_optimized_policy_enabled()
            ):
                kvc_groups = len(self._per_layer_residency)
                kvc_resident = int(self._per_layer_resident_token_count_fast)
                kvc_offloaded = int(self._per_layer_offloaded_token_count_fast)
            else:
                kvc_entries = (
                    self._per_layer_residency.values()
                    if self.config.kvc_backend == "per-layer-arena"
                    else self._residency.values()
                )
                kvc_groups = sum(
                    1
                    for entry in kvc_entries
                    if entry.state
                    in ("resident", "offloaded", "reloading", "evicting")
                )
                kvc_resident = self._resident_token_count()
                kvc_offloaded = self._offloaded_token_count()
            expert_groups = sum(
                int(state.full_num_experts) for state in self._expert_layers.values()
            )
            expert_resident = sum(
                int(state.resident_count) for state in self._expert_layers.values()
            )
            expert_offloaded = max(0, expert_groups - expert_resident)
            self.stats.unified_residency_enabled = True
            self.stats.resident_group_kvc_count = int(kvc_groups)
            self.stats.resident_group_expert_count = int(expert_groups)
            self.stats.resident_group_count = int(kvc_groups + expert_groups)
            self.stats.resident_group_resident_count = int(
                kvc_resident + expert_resident
            )
            self.stats.resident_group_offloaded_count = int(
                kvc_offloaded + expert_offloaded
            )
            self.stats.resident_group_recovering_count = (
                len(self._pending_expert_copy_events)
                + len(self._pending_kvc_reload_events)
                + len(self._pending_kvc_evict_events)
            )
            return
        self._sync_dirty_expert_groups()
        groups = list(self._resident_groups.values())
        self.stats.unified_residency_enabled = True
        self.stats.resident_group_count = len(groups)
        self.stats.resident_group_kvc_count = sum(
            1 for g in groups if g.key.kind == "kvc"
        )
        self.stats.resident_group_expert_count = sum(
            1 for g in groups if g.key.kind == "expert"
        )
        self.stats.resident_group_resident_count = sum(
            1 for g in groups if g.state == "resident"
        )
        self.stats.resident_group_offloaded_count = sum(
            1 for g in groups if g.state == "offloaded"
        )
        self.stats.resident_group_recovering_count = sum(
            1 for g in groups if g.state in ("reloading", "materializing", "evicting")
        )

    def _validate_resident_groups(self) -> None:
        errors = 0
        last_error = ""
        for key, group in self._resident_groups.items():
            if key != group.key:
                errors += 1
                last_error = "group_key_mismatch"
            if group.state == "resident" and group.gpu_ref is None:
                errors += 1
                last_error = f"{key.kind}_resident_missing_gpu_ref"
            if group.state == "offloaded" and group.cpu_ref is None:
                errors += 1
                last_error = f"{key.kind}_offloaded_missing_cpu_ref"
            if (
                group.state in ("reloading", "materializing", "evicting")
                and group.ready_event is None
            ):
                errors += 1
                last_error = f"{key.kind}_recovering_missing_event"
        self.stats.resident_group_state_error_count = errors
        self.stats.resident_group_last_error = last_error

    def install_on_runner(self, runner: Any) -> None:
        if self.installed:
            return
        t0 = time.perf_counter()
        self._runner = runner
        runner.layerkv_runtime = self
        runner_device = getattr(runner, "device", None)
        if str(runner_device).startswith("cuda") or runner_device == "cuda":
            self._copy_stream = torch.cuda.Stream()
        self._allocator = getattr(runner, "token_to_kv_pool_allocator", None)
        self._req_to_token_pool = getattr(runner, "req_to_token_pool", None)
        self._install_kv_pool_hooks(getattr(runner, "token_to_kv_pool", None))
        if self.config.kvc_backend in ("virtual-arena", "per-layer-arena"):
            self._ensure_virtual_scratch()
        self._discover_expert_support(runner)
        self.installed = True
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.layerkv_python_overhead_ms += elapsed_ms
        self._add_profile("profile_install_ms", elapsed_ms)
        logger.info(
            "LayerKV enabled mode=%s policy=%s reclaim_limit_mb=%.1f "
            "kvc_supported=%s expert_supported=%s reason=%s",
            self.config.mode,
            self.config.policy,
            self.config.reclaim_limit_mb,
            self.physical_kvc_supported,
            self.physical_expert_supported,
            self.unsupported_reason,
        )

    def _install_kv_pool_hooks(self, kv_pool: Any) -> None:
        if kv_pool is None:
            self.unsupported_reason = "token_to_kv_pool is not initialized"
            return
        self._kv_pool = kv_pool
        if getattr(kv_pool, "_layerkv_wrapped", False):
            return

        def wrap(name: str, wrapper_factory: Callable[[Callable], Callable]) -> None:
            orig = getattr(kv_pool, name, None)
            if orig is None or not callable(orig):
                return
            self._wrapped_methods[name] = orig
            setattr(kv_pool, name, wrapper_factory(orig))

        wrap("set_kv_buffer", self._wrap_set_kv_buffer)
        wrap("get_key_buffer", self._wrap_get_key_buffer)
        wrap("get_value_buffer", self._wrap_get_value_buffer)
        wrap("get_kv_buffer", self._wrap_get_kv_buffer)
        kv_pool._layerkv_wrapped = True
        kv_pool.layerkv_runtime = self
        self.physical_kvc_supported = self._can_support_kvc_pool(kv_pool)
        if not self.physical_kvc_supported and not self.unsupported_reason:
            self.unsupported_reason = (
                f"KVC physical offload is not implemented for "
                f"{type(kv_pool).__name__}; running accounting-only hooks"
            )
        if (
            self.config.mode == "kvc-only"
            and self.config.reclaim_limit_mb > 0
            and not self.physical_kvc_supported
        ):
            self.stats.comparable = False
            self.stats.comparability_reason = self.unsupported_reason

    def _can_support_kvc_pool(self, kv_pool: Any) -> bool:
        try:
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        except Exception:
            return False

        page_size = int(getattr(kv_pool, "page_size", 1) or 1)
        if type(kv_pool) is not MHATokenToKVPool:
            self.unsupported_reason = (
                f"KVC physical offload supports non-FP4 MHATokenToKVPool only, got "
                f"{type(kv_pool).__name__}"
            )
            return False
        if self._allocator is None or self._req_to_token_pool is None:
            self.unsupported_reason = "KVC allocator or req_to_token_pool is missing"
            return False

        try:
            self._page_size = max(1, page_size)
            self.stats.kvc_page_size = self._page_size
            one_k = kv_pool._get_key_buffer(kv_pool.start_layer)[0].nbytes
            one_v = kv_pool._get_value_buffer(kv_pool.start_layer)[0].nbytes
            self._bytes_per_token_all_layers = int((one_k + one_v) * kv_pool.layer_num)
        except Exception as exc:
            self.unsupported_reason = f"failed to inspect KV token size: {exc}"
            return False
        return self._bytes_per_token_all_layers > 0

    def _ensure_host_store(self) -> None:
        if self._host_store is not None:
            return
        if self._kv_pool is None or self._bytes_per_token_all_layers <= 0:
            raise RuntimeError("KVC host store cannot initialize without KV pool")
        # Allocate host capacity against the total reclaim intent so dynamic
        # policy fractions can vary without reallocating CPU backing.
        target_bytes = max(1, int(self.config.reclaim_limit_mb * 1024 * 1024))
        # _LayerKVHostKVStore allocates one buffer of capacity_tokens for every
        # layer.  Therefore capacity_tokens must be derived from all-layer KV
        # bytes even for per-layer arena mode.  Using per-layer bytes here
        # silently multiplies a 4GB reclaim target by layer_num and can push host
        # RSS above 100GB during decode.
        token_bytes = self._bytes_per_token_all_layers
        target_tokens = max(1, target_bytes // max(1, token_bytes))
        target_tokens = self._align_tokens_up(target_tokens)
        capacity_tokens = target_tokens + max(
            target_tokens, self._align_tokens_up(self.config.kvc_block_tokens)
        )
        self._host_store = _LayerKVHostKVStore(
            self._kv_pool,
            capacity_tokens,
            per_layer_mode=(self.config.kvc_backend == "per-layer-arena"),
        )
        self.stats.kvc_host_capacity_tokens = self._host_store.capacity_tokens
        self.stats.kvc_host_backing_mb = self._host_store.capacity_mb

    def _ensure_virtual_scratch(self) -> bool:
        if self.config.kvc_backend not in ("virtual-arena", "per-layer-arena"):
            return True
        if self._virtual_scratch_locs is not None:
            return True
        if self._allocator is None:
            self.stats.layerkv_kvc_backend_ready = False
            self.stats.layerkv_kvc_backend_reason = "missing allocator"
            return False
        scratch_tokens = max(0, int(self.config.virtual_scratch_tokens))
        if scratch_tokens <= 0:
            self.stats.layerkv_kvc_backend_ready = False
            self.stats.layerkv_kvc_backend_reason = "virtual scratch disabled"
            return False
        locs = self._allocator.alloc(scratch_tokens)
        if locs is None:
            self.stats.virtual_scratch_alloc_failed_count += 1
            self.stats.layerkv_kvc_backend_ready = False
            self.stats.layerkv_kvc_backend_reason = (
                "allocator failed to reserve virtual scratch"
            )
            return False
        self._virtual_scratch_locs = locs.to(dtype=torch.int64)
        self._virtual_scratch_capacity = int(locs.numel())
        self._virtual_scratch_cache_by_layer.clear()
        self._metadata_patch_cache.clear()
        self._metadata_patch_tensor_cache.clear()
        if self._virtual_scratch_capacity >= 2:
            split = max(1, self._virtual_scratch_capacity // 2)
            self._virtual_scratch_buffers = [
                self._virtual_scratch_locs[:split],
                self._virtual_scratch_locs[split:],
            ]
        else:
            self._virtual_scratch_buffers = [self._virtual_scratch_locs]
        self.stats.virtual_scratch_capacity_tokens = self._virtual_scratch_capacity
        self.stats.layerkv_kvc_backend_ready = True
        self.stats.layerkv_kvc_backend_reason = ""
        return True

    def _per_layer_virtual_scratch_enabled(self) -> bool:
        return (
            self.config.kvc_backend == "per-layer-arena"
            and self._per_layer_allocator_enabled()
            and self._virtual_scratch_locs is not None
            and int(self._virtual_scratch_locs.numel()) > 0
        )

    def _per_layer_virtual_scratch_capacity_tokens(self) -> int:
        if not self._per_layer_virtual_scratch_enabled():
            return 0
        return int(self._virtual_scratch_locs.numel())

    def reserved_allocator_tokens(self) -> int:
        if self.config.kvc_backend == "virtual-arena":
            if self._virtual_scratch_locs is None:
                return 0
            return int(self._virtual_scratch_locs.numel())
        if self.config.kvc_backend == "per-layer-arena":
            scratch_tokens = (
                int(self._virtual_scratch_locs.numel())
                if self._virtual_scratch_locs is not None
                else 0
            )
            return int(len(self._per_layer_arena_reserved_locs)) + scratch_tokens
        return 0

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
        if self.config.expert_cpu_backing_mode == "all":
            self._preload_all_expert_cpu_backing()

    def _preload_all_expert_cpu_backing(self) -> None:
        if self._expert_global_cpu_backing:
            return
        t0 = time.perf_counter()
        copied = 0
        copied_bytes = 0
        for layer_id, module in self._expert_modules:
            param_names = self._expert_param_names(module)
            full_num_experts = int(module.w13_weight.data.shape[0])
            for expert_id in range(full_num_experts):
                params = self._copy_expert_to_cpu_backing_no_account(
                    module,
                    param_names,
                    expert_id,
                    non_blocking=True,
                    pin_memory=False,
                )
                self._expert_global_cpu_backing[(int(layer_id), int(expert_id))] = (
                    params
                )
                copied += 1
                copied_bytes += self._expert_backing_bytes(params)
            device = module.w13_weight.data.device
            if device.type == "cuda":
                torch.cuda.current_stream(device=device).synchronize()
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
        self.stats.planner_version = "deadline-dp-v1"
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
        coresid_expert_plan_inputs = (
            self._build_coresid_expert_plan_inputs(forward_batch)
            if self.config.policy == "coresid"
            else None
        )
        if self.config.policy == "coresid":
            expert_layer_items: List[Tuple[int, int, int]] = []
            expert_candidates: List[Tuple[float, int]] = []
        else:
            expert_layer_items, expert_candidates = self._build_expert_cost_inputs()
        if self.config.policy == "coresid" and coresid_expert_plan_inputs is not None:
            expert_prefix_candidates = coresid_expert_plan_inputs[3]
        else:
            expert_prefix_candidates = expert_candidates
        expert_prefix_table = self._build_expert_prefix_table(expert_prefix_candidates)
        (
            kvc_layer_ids,
            kvc_max_tokens_per_layer,
            kvc_block_tokens_for_plan,
        ) = self._layer_aware_kvc_plan_context(forward_batch)
        kvc_avg_prefix = max(1.0, self._avg_prefix_len(forward_batch))
        kvc_batch_size = max(1, int(getattr(self.stats, "observed_batch_size", 0) or 1))
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
            )
            value = self._estimate_arena_kvc_reclaim_cost_from_plan(
                plan,
                layer_ids=kvc_layer_ids,
                avg_prefix=kvc_avg_prefix,
                batch_size=kvc_batch_size,
                recovery_steps=kvc_recovery_steps,
            )
            cached = (value, tokens)
            kvc_cost_cache[key] = cached
            self.stats.planner_dp_lookup_ms += (
                time.perf_counter() - lookup_t0
            ) * 1000.0
            return cached

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
                if self.config.policy == "coresid":
                    (
                        value,
                        churn_count,
                        churn_mb,
                        install_mb,
                        backing_miss_cost,
                        materialize_cost,
                        capacities,
                        layer_costs,
                    ) = self._lookup_expert_prefix_cost(
                        key,
                        expert_prefix_candidates,
                        expert_prefix_table,
                        base_capacities=(
                            coresid_expert_plan_inputs[0]
                            if coresid_expert_plan_inputs is not None
                            else None
                        ),
                        expert_bytes_by_layer=(
                            coresid_expert_plan_inputs[2]
                            if coresid_expert_plan_inputs is not None
                            else None
                        ),
                        include_churn_cost=True,
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
        best: Optional[Tuple[float, float, float, float, float]] = (
            expert_only_cost,
            0.0,
            1.0,
            0.0,
            expert_only_cost,
        )
        best_expert_stats = (
            expert_only_churn,
            expert_only_churn_mb,
            expert_only_install,
            expert_only_backing_cost,
            expert_only_materialize_cost,
        )
        best_kvc_tokens = 0
        infeasible_expert = 0
        if expert_only_cost >= 1.0e29:
            infeasible_expert += 1
        with self._profile("profile_planner_dp_ms"):
            for kvc_mb in sorted(kvc_points):
                if kvc_mb <= 0.0:
                    continue
                expert_mb = max(0.0, target_mb - kvc_mb)
                kvc_fraction = kvc_mb / target_mb
                expert_fraction = expert_mb / target_mb
                with self._profile("profile_planner_dp_candidate_eval_ms"):
                    kvc_cost, kvc_tokens = cached_kvc_cost(kvc_mb)
                    if kvc_cost >= best[0]:
                        continue
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
                    candidate = (
                        total_cost,
                        kvc_fraction,
                        expert_fraction,
                        kvc_cost,
                        expert_cost,
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
        assert best is not None
        _total_cost, kvc_fraction, expert_fraction, kvc_cost, expert_cost = best
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
        self.stats.planner_estimated_kvc_controller_cost = float(
            kvc_controller_cost_cache.get(
                round(target_mb * kvc_fraction, 6),
                self.stats.planner_estimated_kvc_controller_cost,
            )
        )
        self.stats.planner_estimated_expert_churn_count = float(best_expert_stats[0])
        self.stats.planner_estimated_expert_churn_mb = float(best_expert_stats[1])
        self.stats.planner_estimated_expert_install_mb = float(best_expert_stats[2])
        self.stats.planner_estimated_expert_backing_miss_cost = float(
            best_expert_stats[3]
        )
        self.stats.planner_estimated_expert_materialize_cost = float(
            best_expert_stats[4]
        )
        if self.config.policy == "coresid":
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
                    )
                )
            else:
                self._planned_kvc_tokens_by_layer = (
                    self._build_layer_aware_kvc_token_plan(
                        self._planned_kvc_token_target, forward_batch
                    )
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
    ) -> Dict[int, int]:
        if not layer_ids or total_tokens <= 0 or max_tokens_per_layer <= 0:
            return {}
        total_tokens = self._align_tokens_down(int(total_tokens))
        total_capacity = max_tokens_per_layer * len(layer_ids)
        total_tokens = min(total_tokens, total_capacity)
        if total_tokens <= 0:
            return {}
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
        weights = torch.arange(1, layer_count + 1, dtype=torch.float64, device="cpu")
        weight_sum = float(layer_count * (layer_count + 1)) / 2.0 or 1.0
        raw = torch.round(
            token_counts[:, None].to(torch.float64) * weights[None, :] / weight_sum
        )
        plan = raw.to(torch.int64)
        plan = torch.minimum(
            plan,
            torch.tensor(int(max_tokens_per_layer), dtype=torch.int64, device="cpu"),
        )
        plan = (plan // int(block_tokens)) * int(block_tokens)
        remaining = token_counts - plan.sum(axis=1)
        for idx in range(layer_count - 1, -1, -1):
            positive = remaining > 0
            if not bool(torch.any(positive).item()):
                break
            capacity = int(max_tokens_per_layer) - plan[:, idx]
            add = torch.minimum(capacity, remaining)
            add = (add // int(block_tokens)) * int(block_tokens)
            add = torch.where(positive & (add > 0), add, torch.zeros_like(add))
            plan[:, idx] += add
            remaining -= add

        bytes_per_token = max(1, self._bytes_per_kvc_token_per_layer())
        ms_per_token: List[float] = []
        fallback_per_token = (2.0 * float(bytes_per_token)) / 1.0e9 * 1000.0
        for layer_id in layer_ids:
            layer_id = int(layer_id)
            reload_ewma = self._kvc_reload_ms_per_mb_ewma_by_layer.get(layer_id)
            evict_ewma = self._kvc_evict_ms_per_mb_ewma_by_layer.get(layer_id)
            if reload_ewma is not None or evict_ewma is not None:
                ms_per_mb = float(reload_ewma or 0.0) + float(evict_ewma or 0.0)
                ms_per_token.append(
                    ms_per_mb * float(bytes_per_token) / float(1024 * 1024)
                )
            else:
                ms_per_token.append(fallback_per_token)
        ms_per_token_tensor = torch.tensor(
            ms_per_token, dtype=torch.float64, device="cpu"
        )
        raw_copy = 0.03 + plan.to(torch.float64) * ms_per_token_tensor[None, :]
        raw_copy = torch.where(plan > 0, raw_copy, torch.zeros_like(raw_copy))
        overlap_values = self._estimate_kvc_overlap_windows_ms(
            layer_ids, avg_prefix=avg_prefix, batch_size=batch_size
        )
        overlap = torch.tensor(
            [float(overlap_values.get(int(layer_id), 0.0)) for layer_id in layer_ids],
            dtype=torch.float64,
            device="cpu",
        )
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

    def _estimate_expert_reclaim_cost(
        self,
        reclaim_mb: float,
        forward_batch: Any = None,
        *,
        update_stats: bool = True,
        precomputed_layer_items: Optional[List[Tuple[int, int, int]]] = None,
        precomputed_candidates: Optional[
            List[Tuple[float, int, int, int, float, float, float]]
        ] = None,
    ) -> float:
        if reclaim_mb <= 0.0:
            if update_stats:
                self.stats.planner_estimated_expert_churn_count = 0.0
                self.stats.planner_estimated_expert_churn_mb = 0.0
                self.stats.planner_estimated_expert_install_mb = 0.0
            return 0.0
        if precomputed_layer_items is None or precomputed_candidates is None:
            layer_items, candidates = self._build_expert_cost_inputs()
        else:
            layer_items = precomputed_layer_items
            candidates = precomputed_candidates
        target_bytes = int(reclaim_mb * 1024 * 1024)
        reclaimed = 0
        cost = 0.0
        total_churn_count = 0.0
        total_churn_mb = 0.0
        backing_miss_cost = 0.0
        materialize_cost = 0.0
        for (
            candidate_cost,
            expert_bytes,
            _layer_id,
            _expert_id,
            expected_calls,
            candidate_backing_cost,
            candidate_materialize_cost,
        ) in candidates:
            if reclaimed >= target_bytes:
                break
            reclaimed += expert_bytes
            cost += float(candidate_cost)
            total_churn_count += float(expected_calls)
            backing_miss_cost += float(candidate_backing_cost)
            materialize_cost += float(candidate_materialize_cost)
        if reclaimed < target_bytes:
            return 1.0e30
        install_mb = reclaimed / float(1024 * 1024)
        total_churn_mb = total_churn_count * (
            float(layer_items[0][2]) / float(1024 * 1024) if layer_items else 0.0
        )
        if update_stats:
            self.stats.planner_estimated_expert_churn_count = total_churn_count
            self.stats.planner_estimated_expert_churn_mb = total_churn_mb
            self.stats.planner_estimated_expert_install_mb = install_mb
        else:
            self.stats.planner_estimated_expert_churn_count = total_churn_count
            self.stats.planner_estimated_expert_churn_mb = total_churn_mb
            self.stats.planner_estimated_expert_install_mb = install_mb
        self.stats.planner_estimated_expert_backing_miss_cost = backing_miss_cost
        self.stats.planner_estimated_expert_materialize_cost = materialize_cost
        install_cost = install_mb * 0.15
        return cost + install_cost

    def _build_expert_cost_inputs(
        self,
    ) -> Tuple[
        List[Tuple[int, int, int]],
        List[Tuple[float, int, int, int, float, float, float]],
    ]:
        planning_items = self._expert_layer_items_for_planning()
        layer_items = [
            (int(layer_id), int(full_num_experts), int(expert_bytes))
            for layer_id, full_num_experts, expert_bytes, _module, _state in planning_items
        ]
        candidates: List[Tuple[float, int, int, int, float, float, float]] = []
        batch_size = max(1, int(getattr(self.stats, "observed_batch_size", 0) or 1))
        horizon_steps = max(
            1, min(64, int(getattr(self.stats, "decode_steps", 0) or 16))
        )
        for layer_id, full_num_experts, expert_bytes, module, state in planning_items:
            top_k = int(getattr(module, "top_k", 0) or 0)
            if top_k <= 0:
                top_k = int(getattr(module.moe_runner_config, "top_k", 1) or 1)
            for expert_id in range(full_num_experts):
                (
                    cost,
                    expected_calls,
                    backing_miss_cost,
                    materialize_cost,
                ) = self._expert_candidate_expected_cost(
                    int(layer_id),
                    int(expert_id),
                    int(expert_bytes),
                    state,
                    horizon_steps=horizon_steps,
                    batch_size=batch_size,
                    top_k=top_k,
                )
                candidates.append(
                    (
                        float(cost),
                        int(expert_bytes),
                        int(layer_id),
                        int(expert_id),
                        float(expected_calls),
                        float(backing_miss_cost),
                        float(materialize_cost),
                    )
                )
        candidates.sort(key=lambda item: (item[0], item[2], item[3]))
        self.stats.planner_dp_expert_candidate_count = len(candidates)
        return layer_items, candidates

    def _expert_layer_items_for_planning(
        self,
    ) -> List[Tuple[int, int, int, Any, Optional[_LayerKVExpertLayerState]]]:
        items: List[Tuple[int, int, int, Any, Optional[_LayerKVExpertLayerState]]] = []
        module_by_layer = {
            int(layer_id): module for layer_id, module in self._expert_modules
        }
        if self._expert_layers:
            for layer_id, state in sorted(self._expert_layers.items()):
                module = module_by_layer.get(int(layer_id), state.module)
                items.append(
                    (
                        int(layer_id),
                        int(state.full_num_experts),
                        int(state.expert_bytes),
                        module,
                        state,
                    )
                )
        else:
            for layer_id, module in self._expert_modules:
                items.append(
                    (
                        int(layer_id),
                        int(module.w13_weight.data.shape[0]),
                        int(self._expert_bytes(module)),
                        module,
                        None,
                    )
                )
        return items

    def _expert_min_capacity_for_layer(
        self,
        layer_id: int,
        full_num_experts: int,
        module: Any,
    ) -> int:
        top_k = int(getattr(module, "top_k", 0) or 0)
        if top_k <= 0:
            top_k = int(getattr(module.moe_runner_config, "top_k", 1) or 1)
        decode_unique = len(self._expert_hotness_decode.get(int(layer_id), {}))
        if self._dynamic_expert_churn_policy_enabled():
            return max(1, min(int(full_num_experts), top_k))
        return max(1, min(int(full_num_experts), max(top_k, decode_unique)))

    def _expert_candidate_expected_cost(
        self,
        layer_id: int,
        expert_id: int,
        expert_bytes: int,
        state: Optional[_LayerKVExpertLayerState],
        *,
        horizon_steps: int,
        batch_size: int,
        top_k: int,
    ) -> Tuple[float, float, float, float]:
        decode_hotness = self._expert_hotness_decode.get(int(layer_id), {})
        prefill_hotness = self._expert_hotness_prefill.get(int(layer_id), {})
        hotness = decode_hotness or prefill_hotness
        observed_total_calls = int(sum(hotness.values()))
        total_calls = max(1, observed_total_calls)
        decode_count = int(decode_hotness.get(int(expert_id), 0))
        prefill_count = int(prefill_hotness.get(int(expert_id), 0))
        observed_count = int(hotness.get(int(expert_id), 0))
        p = float(observed_count) / float(total_calls)
        expected_calls = p * float(
            max(1, batch_size) * max(1, horizon_steps) * max(1, top_k)
        )
        if decode_count > 0:
            expected_calls = max(1.0, expected_calls)
        mb = float(expert_bytes) / float(1024 * 1024)
        cost = p * mb
        cost += 0.02 * expected_calls
        if decode_count <= 0 and prefill_count > 0:
            cost += 0.002 * float(prefill_count)
        backing_miss_cost = 0.0
        materialize_cost = 0.0
        if expected_calls > 0.0:
            mb = float(expert_bytes) / float(1024 * 1024)
            if (
                self.stats.expert_materialize_mb_total > 0.0
                and self.stats.expert_materialize_ms > 0.0
            ):
                materialize_ms = (
                    self.stats.expert_materialize_ms
                    / self.stats.expert_materialize_mb_total
                    * mb
                )
            else:
                materialize_ms = 0.03 + 0.12 * mb
            materialize_cost = p * materialize_ms
            cost += materialize_cost
            key = (int(layer_id), int(expert_id))
            pending_backing = key in self._pending_expert_d2h_by_key
            if state is None:
                backing_factor = 0.25
            elif int(expert_id) in state.logical_to_slot:
                backing_factor = 0.0
            elif int(expert_id) in state.cpu_params:
                backing_factor = 0.0
            elif pending_backing:
                backing_factor = 0.05
            else:
                backing_factor = 0.25
            backing_miss_cost = p * backing_factor * mb
            cost += backing_miss_cost
        if state is not None and int(expert_id) in state.lru:
            age = max(0, int(self._decode_step) - int(state.lru[int(expert_id)]))
            if age < 64:
                cost += 0.05 * (64.0 - float(age)) / 64.0
        return (
            float(cost),
            float(expected_calls),
            float(backing_miss_cost),
            float(materialize_cost),
        )

    def _coresid_expert_candidate_cost(
        self,
        layer_id: int,
        expert_id: int,
        expert_bytes: int,
        state: Optional[_LayerKVExpertLayerState],
        *,
        horizon_steps: int,
        batch_size: int,
        top_k: int,
    ) -> Tuple[float, float, float, float]:
        return self._expert_candidate_expected_cost(
            layer_id,
            expert_id,
            expert_bytes,
            state,
            horizon_steps=horizon_steps,
            batch_size=batch_size,
            top_k=top_k,
        )

    def _build_coresid_expert_plan_inputs(self, forward_batch: Any) -> Tuple[
        Dict[int, int],
        Dict[int, int],
        Dict[int, int],
        List[Tuple[float, int, int, int, float, float, float]],
    ]:
        del forward_batch
        capacities: Dict[int, int] = {}
        min_capacities: Dict[int, int] = {}
        expert_bytes_by_layer: Dict[int, int] = {}
        candidates: List[Tuple[float, int, int, int, float, float, float]] = []
        batch_size = max(1, int(getattr(self.stats, "observed_batch_size", 0) or 1))
        horizon_steps = max(
            1, min(64, int(getattr(self.stats, "decode_steps", 0) or 16))
        )
        for (
            layer_id,
            full_num_experts,
            expert_bytes,
            module,
            state,
        ) in self._expert_layer_items_for_planning():
            layer_id = int(layer_id)
            capacities[layer_id] = int(full_num_experts)
            expert_bytes_by_layer[layer_id] = int(expert_bytes)
            top_k = int(getattr(module, "top_k", 0) or 0)
            if top_k <= 0:
                top_k = int(getattr(module.moe_runner_config, "top_k", 1) or 1)
            min_capacity = self._expert_min_capacity_for_layer(
                layer_id, int(full_num_experts), module
            )
            min_capacities[layer_id] = int(min_capacity)
            expert_costs: List[Tuple[float, int, float, float, float]] = []
            for expert_id in range(int(full_num_experts)):
                (
                    cost,
                    expected_calls,
                    backing_miss_cost,
                    materialize_cost,
                ) = self._coresid_expert_candidate_cost(
                    layer_id,
                    int(expert_id),
                    int(expert_bytes),
                    state,
                    horizon_steps=horizon_steps,
                    batch_size=batch_size,
                    top_k=top_k,
                )
                expert_costs.append(
                    (
                        cost,
                        int(expert_id),
                        expected_calls,
                        backing_miss_cost,
                        materialize_cost,
                    )
                )
            expert_costs.sort(key=lambda item: (item[0], item[1]))
            max_evict = max(0, int(full_num_experts) - int(min_capacity))
            for (
                cost,
                expert_id,
                expected_calls,
                backing_miss_cost,
                materialize_cost,
            ) in expert_costs[:max_evict]:
                candidates.append(
                    (
                        float(cost),
                        int(expert_bytes),
                        layer_id,
                        int(expert_id),
                        float(expected_calls),
                        float(backing_miss_cost),
                        float(materialize_cost),
                    )
                )
        candidates.sort(key=lambda item: (item[0], item[2], item[3]))
        self.stats.planner_dp_expert_candidate_count = len(candidates)
        return capacities, min_capacities, expert_bytes_by_layer, candidates

    def _build_expert_prefix_table(
        self,
        candidates: List[Tuple[float, int, int, int, float, float, float]],
    ) -> Dict[str, Any]:
        prefix_bytes: List[int] = []
        prefix_cost: List[float] = []
        prefix_expected_calls: List[float] = []
        prefix_backing_cost: List[float] = []
        prefix_materialize_cost: List[float] = []
        total_bytes = 0
        total_cost = 0.0
        total_expected_calls = 0.0
        total_backing_cost = 0.0
        total_materialize_cost = 0.0
        for (
            cost,
            expert_bytes,
            _layer_id,
            _expert_id,
            expected_calls,
            backing_cost,
            materialize_cost,
        ) in candidates:
            total_bytes += int(expert_bytes)
            total_cost += float(cost)
            total_expected_calls += float(expected_calls)
            total_backing_cost += float(backing_cost)
            total_materialize_cost += float(materialize_cost)
            prefix_bytes.append(total_bytes)
            prefix_cost.append(total_cost)
            prefix_expected_calls.append(total_expected_calls)
            prefix_backing_cost.append(total_backing_cost)
            prefix_materialize_cost.append(total_materialize_cost)
        return {
            "prefix_bytes": prefix_bytes,
            "prefix_cost": prefix_cost,
            "prefix_expected_calls": prefix_expected_calls,
            "prefix_backing_cost": prefix_backing_cost,
            "prefix_materialize_cost": prefix_materialize_cost,
        }

    def _lookup_expert_prefix_cost(
        self,
        reclaim_mb: float,
        candidates: List[Tuple[float, int, int, int, float, float, float]],
        prefix_table: Dict[str, Any],
        *,
        base_capacities: Optional[Dict[int, int]] = None,
        expert_bytes_by_layer: Optional[Dict[int, int]] = None,
        include_churn_cost: bool = True,
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        Dict[int, int],
        Dict[int, float],
    ]:
        if reclaim_mb <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}
        target_bytes = int(max(0.0, float(reclaim_mb)) * 1024 * 1024)
        prefix_bytes = prefix_table.get("prefix_bytes", [])
        idx = bisect.bisect_left(prefix_bytes, target_bytes)
        if idx >= len(prefix_bytes):
            return 1.0e30, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}
        install_bytes = int(prefix_bytes[idx])
        install_mb = float(install_bytes) / float(1024 * 1024)
        expected_calls = float(prefix_table["prefix_expected_calls"][idx])
        backing_cost = float(prefix_table["prefix_backing_cost"][idx])
        materialize_cost = float(prefix_table["prefix_materialize_cost"][idx])
        base_cost = float(prefix_table["prefix_cost"][idx])
        if expert_bytes_by_layer:
            first_expert_bytes = float(next(iter(expert_bytes_by_layer.values())))
        elif candidates:
            first_expert_bytes = float(candidates[0][1])
        else:
            first_expert_bytes = 0.0
        churn_mb = expected_calls * first_expert_bytes / float(1024 * 1024)
        total_cost = base_cost + 0.15 * install_mb
        if include_churn_cost:
            total_cost += 0.05 * churn_mb
        capacities = {int(k): int(v) for k, v in (base_capacities or {}).items()}
        layer_costs: Dict[int, float] = {int(k): 0.0 for k in capacities}
        for cost, _bytes, layer_id, _expert_id, *_rest in candidates[: idx + 1]:
            layer_id = int(layer_id)
            if capacities:
                capacities[layer_id] = max(0, int(capacities.get(layer_id, 0)) - 1)
            layer_costs[layer_id] = layer_costs.get(layer_id, 0.0) + float(cost)
        return (
            float(total_cost),
            expected_calls,
            churn_mb,
            install_mb,
            backing_cost,
            materialize_cost,
            capacities,
            layer_costs,
        )

    def _plan_coresid_expert_reclaim_cost(
        self,
        reclaim_mb: float,
        forward_batch: Any,
        *,
        precomputed: Optional[
            Tuple[
                Dict[int, int],
                Dict[int, int],
                Dict[int, int],
                List[Tuple[float, int, int, int, float, float, float]],
            ]
        ] = None,
    ) -> Tuple[float, float, float, float, Dict[int, int], Dict[int, float]]:
        if reclaim_mb <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, {}, {}
        target_bytes = int(max(0.0, float(reclaim_mb)) * 1024 * 1024)
        if precomputed is None:
            precomputed = self._build_coresid_expert_plan_inputs(forward_batch)
        base_capacities, min_capacities, expert_bytes_by_layer, candidates = precomputed
        if not base_capacities:
            return 1.0e30, 0.0, 0.0, 0.0, {}, {}
        capacities = {
            int(layer_id): int(capacity)
            for layer_id, capacity in base_capacities.items()
        }
        layer_costs: Dict[int, float] = {int(layer_id): 0.0 for layer_id in capacities}
        reclaimed = 0
        total_cost = 0.0
        install_mb = 0.0
        expected_churn_count = 0.0
        backing_miss_cost_total = 0.0
        materialize_cost_total = 0.0
        for (
            cost,
            expert_bytes,
            layer_id,
            _expert_id,
            expected_calls,
            backing_miss_cost,
            materialize_cost,
        ) in candidates:
            if reclaimed >= target_bytes:
                break
            if capacities[int(layer_id)] <= min_capacities[int(layer_id)]:
                continue
            capacities[int(layer_id)] -= 1
            reclaimed += int(expert_bytes)
            mb = float(expert_bytes) / float(1024 * 1024)
            install_mb += mb
            expected_churn_count += float(expected_calls)
            total_cost += float(cost)
            backing_miss_cost_total += float(backing_miss_cost)
            materialize_cost_total += float(materialize_cost)
            layer_costs[int(layer_id)] = layer_costs.get(int(layer_id), 0.0) + float(
                cost
            )
        if reclaimed < target_bytes:
            return (
                1.0e30,
                expected_churn_count,
                0.0,
                install_mb,
                capacities,
                layer_costs,
            )
        churn_mb = expected_churn_count * (
            float(next(iter(expert_bytes_by_layer.values()))) / float(1024 * 1024)
        )
        total_cost += 0.05 * churn_mb
        total_cost += 0.15 * install_mb
        self.stats.planner_estimated_expert_backing_miss_cost = backing_miss_cost_total
        self.stats.planner_estimated_expert_materialize_cost = materialize_cost_total
        return (
            total_cost,
            expected_churn_count,
            churn_mb,
            install_mb,
            capacities,
            layer_costs,
        )

    def _estimate_expert_slot_capacities_for_target(
        self, target_mb: float
    ) -> Dict[int, int]:
        target_bytes = max(0, int(target_mb * 1024 * 1024))
        layer_infos: List[Dict[str, int]] = []
        allow_dynamic_churn = (
            self.config.dynamic_pressure_from_kvc and self.config.policy == "kv-first"
        )
        if self._expert_layers:
            for state in self._expert_layers.values():
                top_k = int(getattr(state.module, "top_k", 0) or 0)
                if top_k <= 0:
                    top_k = int(
                        getattr(state.module.moe_runner_config, "top_k", 1) or 1
                    )
                observed_decode_unique = len(
                    self._expert_hotness_decode.get(state.layer_id, {})
                )
                if allow_dynamic_churn or self._dynamic_expert_churn_policy_enabled():
                    min_capacity = max(1, min(state.full_num_experts, top_k))
                else:
                    min_capacity = max(
                        1, min(state.full_num_experts, observed_decode_unique)
                    )
                layer_infos.append(
                    {
                        "layer_id": int(state.layer_id),
                        "capacity": int(state.full_num_experts),
                        "min_capacity": int(min_capacity),
                        "expert_bytes": int(state.expert_bytes),
                    }
                )
        else:
            for layer_id, module in self._expert_modules:
                full_num_experts = int(module.w13_weight.data.shape[0])
                top_k = int(getattr(module, "top_k", 0) or 0)
                if top_k <= 0:
                    top_k = int(getattr(module.moe_runner_config, "top_k", 1) or 1)
                observed_decode_unique = len(
                    self._expert_hotness_decode.get(layer_id, {})
                )
                if allow_dynamic_churn or self._dynamic_expert_churn_policy_enabled():
                    min_capacity = max(1, min(full_num_experts, top_k))
                else:
                    min_capacity = max(
                        1,
                        min(full_num_experts, top_k),
                        min(full_num_experts, observed_decode_unique),
                    )
                layer_infos.append(
                    {
                        "layer_id": int(layer_id),
                        "capacity": int(full_num_experts),
                        "min_capacity": int(min_capacity),
                        "expert_bytes": int(self._expert_bytes(module)),
                    }
                )
        reclaimed = 0
        layer_infos.sort(key=lambda x: (-x["expert_bytes"], x["layer_id"]))
        while reclaimed < target_bytes:
            for info in layer_infos:
                if reclaimed >= target_bytes:
                    break
                if int(info["capacity"]) <= int(info["min_capacity"]):
                    continue
                info["capacity"] -= 1
                reclaimed += info["expert_bytes"]
            else:
                break
        return {info["layer_id"]: info["capacity"] for info in layer_infos}

    def _expert_support_reason(self) -> str:
        if self.physical_expert_supported:
            return ""
        return (
            self.stats.expert_guard_reason
            or self.unsupported_reason
            or "expert offload unsupported"
        )

    def _available_kvc_reclaim_mb(self, forward_batch: Any = None) -> float:
        if self._bytes_per_token_all_layers <= 0:
            return 0.0
        token_count = 0
        pairs = (
            list(self._current_forward_req_lens)
            if forward_batch is not None
            and self._current_forward_req_lens_batch_id == id(forward_batch)
            else (
                self._batch_req_indices_and_lens(forward_batch)
                if forward_batch is not None
                else []
            )
        )
        for _req_idx, seq_len in pairs:
            token_count += self._align_tokens_down(max(0, seq_len - 1))
        if token_count <= 0:
            token_count = self._resident_token_count() + self._offloaded_token_count()
        return token_count * self._bytes_per_token_all_layers / float(1024 * 1024)

    def _bytes_per_kvc_token_per_layer(self) -> int:
        layer_num = max(1, len(self._kvc_layer_ids()))
        if self._bytes_per_token_all_layers > 0:
            return max(1, self._bytes_per_token_all_layers // layer_num)
        if self._kv_pool is None:
            return 0
        try:
            layer_id = self._kvc_layer_ids()[0]
            return int(
                self._kv_pool._get_key_buffer(layer_id)[0].nbytes
                + self._kv_pool._get_value_buffer(layer_id)[0].nbytes
            )
        except Exception:
            return 0

    def _kvc_layer_ids(self) -> List[int]:
        if self._kv_pool is None:
            return []
        start = int(getattr(self._kv_pool, "start_layer", 0) or 0)
        layer_num = int(getattr(self._kv_pool, "layer_num", 0) or 0)
        if layer_num <= 0:
            return []
        key = (start, layer_num)
        if self._cached_kvc_layer_ids_key != key:
            self._cached_kvc_layer_ids_key = key
            self._cached_kvc_layer_ids = list(range(start, start + layer_num))
        return self._cached_kvc_layer_ids

    def _build_layer_aware_kvc_token_plan(
        self, total_tokens: int, forward_batch: Any
    ) -> Dict[int, int]:
        layer_ids, max_tokens_per_layer, block_tokens = (
            self._layer_aware_kvc_plan_context(forward_batch)
        )
        if not layer_ids or total_tokens <= 0:
            return {}
        # v1 keeps the DP output layer-aware by assigning more reclaim to later
        # layers, where KVC reload has more forward-path overlap.  This is a
        # deterministic plan and does not change baseline layer-average policy
        # semantics.
        return self._build_layer_aware_kvc_token_plan_from_context(
            total_tokens,
            layer_ids=layer_ids,
            max_tokens_per_layer=max_tokens_per_layer,
            block_tokens=block_tokens,
        )

    def _kvc_tokens_by_layer_json(self, token_count: Any) -> str:
        if isinstance(token_count, dict):
            return json.dumps(
                {
                    str(layer_id): int(tokens)
                    for layer_id, tokens in token_count.items()
                },
                sort_keys=True,
            )
        layer_ids = self._kvc_layer_ids()
        if not layer_ids:
            return json.dumps({"all_layers": int(token_count)}, sort_keys=True)
        return json.dumps(
            {str(layer_id): int(token_count) for layer_id in layer_ids},
            sort_keys=True,
        )

    def _available_expert_reclaim_mb(self) -> float:
        if self.config.expert_collector_only:
            return 0.0
        if not self.physical_expert_supported:
            return 0.0
        total = 0
        if self._expert_layers:
            for state in self._expert_layers.values():
                # Keep at least one slot per layer so the wrapped FusedMoE remains
                # executable; dynamic materialization can grow if routing requires.
                total += max(0, state.full_num_experts - 1) * state.expert_bytes
        else:
            for _layer_id, module in self._expert_modules:
                full_num_experts = int(module.w13_weight.data.shape[0])
                total += max(0, full_num_experts - 1) * self._expert_bytes(module)
        return total / float(1024 * 1024)

    def _refresh_reclaim_target_stats(self, forward_batch: Any = None) -> float:
        cache_key = (
            id(forward_batch),
            self._current_forward_mode,
            int(self._decode_step),
            round(float(self.stats.physical_kvc_reclaim_mb), 3),
            round(float(self.stats.physical_expert_reclaim_mb), 3),
            len(self._expert_layers),
            len(self._per_layer_residency),
            len(self._residency),
        )
        if self._cached_reclaim_target_key == cache_key:
            return self._cached_reclaim_target_value
        configured = max(0.0, self.config.reclaim_limit_mb)
        available_kvc = self._available_kvc_reclaim_mb(forward_batch)
        available_expert = self._available_expert_reclaim_mb()
        available_total = available_kvc + available_expert
        if self.config.dynamic_pressure_from_kvc:
            # End-to-end trace replay should reclaim only when the runtime is
            # actually short on KV allocator headroom.  Available KVC bytes are
            # just reclaimable capacity, not pressure; using them as pressure
            # incorrectly turns any large live KV set into a forced 4G reclaim.
            dynamic_needed = self._dynamic_needed_pressure_mb(forward_batch)
            needed_pressure = (
                min(configured, dynamic_needed) if configured > 0.0 else dynamic_needed
            )
        else:
            # Fig4-style controlled experiments use configured pressure directly
            # so policies remain comparable at a fixed reclaim target.
            needed_pressure = configured
        effective = min(needed_pressure, available_total)
        reason = ""
        if self.config.dynamic_pressure_from_kvc and needed_pressure <= 1e-3:
            reason = "no_runtime_pressure"
        elif effective + 1e-3 < needed_pressure:
            reason = "available_reclaim_below_needed_pressure"
        elif (
            not self.config.dynamic_pressure_from_kvc and effective + 1e-3 < configured
        ):
            reason = "available_reclaim_below_configured_limit"
        self.stats.configured_reclaim_limit_mb = configured
        self.stats.needed_pressure_mb = needed_pressure
        self.stats.available_kvc_reclaim_mb = available_kvc
        self.stats.available_expert_reclaim_mb = available_expert
        self.stats.available_total_reclaim_mb = available_total
        self.stats.effective_reclaim_target_mb = effective
        self.stats.target_limited_reason = reason
        self._cached_reclaim_target_key = cache_key
        self._cached_reclaim_target_value = effective
        return effective

    def _set_no_pressure_reclaim_stats(self, configured: float) -> None:
        self.stats.configured_reclaim_limit_mb = float(configured)
        self.stats.needed_pressure_mb = 0.0
        self.stats.effective_reclaim_target_mb = 0.0
        self.stats.target_limited_reason = "no_runtime_pressure"
        self.stats.requested_total_reclaim_mb = 0.0
        self.stats.effective_kvc_reclaim_mb = 0.0
        self.stats.planned_kvc_reclaim_mb = 0.0
        self.stats.planned_expert_reclaim_mb = 0.0
        self.stats.policy_kvc_fraction = 0.0
        self.stats.policy_expert_fraction = 0.0
        self.stats.full_policy_semantics_supported = True
        self.stats.policy_semantics_reason = ""
        if self.config.mode == "kvc-expert" and not self._expert_plan_applied:
            self._expert_install_state = "waiting_for_pressure"
            self.stats.expert_install_state = self._expert_install_state

    def _dynamic_runtime_pressure_mb(self, forward_batch: Any = None) -> float:
        if not self.config.dynamic_pressure_from_kvc:
            return max(0.0, float(self.config.reclaim_limit_mb))
        configured = max(0.0, float(self.config.reclaim_limit_mb))
        dynamic_needed = max(0.0, float(self._dynamic_needed_pressure_mb(forward_batch)))
        return min(configured, dynamic_needed) if configured > 0.0 else dynamic_needed

    def _no_pressure_fast_path_active(self, forward_batch: Any = None) -> bool:
        if not self.config.dynamic_pressure_from_kvc:
            return False
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return False
        if self._current_forward_mode != "decode":
            return False
        if self._pending_kvc_reload_events or self._pending_kvc_evict_events:
            return False
        if self._pending_virtual_kvc_materialize:
            return False
        if (
            self._pending_expert_copy_events
            or self._pending_expert_d2h_events
            or self._expert_install_d2h_queue
        ):
            return False
        if self._expert_install_queue or self._expert_plan_applied:
            return False
        if self._has_offloaded_kvc_entries():
            return False
        if (
            self.stats.physical_kvc_reclaim_mb > 1e-3
            or self.stats.physical_expert_reclaim_mb > 1e-3
        ):
            return False

        configured = max(0.0, float(self.config.reclaim_limit_mb))
        needed = self._dynamic_runtime_pressure_mb(forward_batch)
        if needed > 1e-3:
            return False
        self._set_no_pressure_reclaim_stats(configured)
        return True

    def _refresh_physical_reclaim_peaks(
        self, *, record_step_sample: bool = False
    ) -> None:
        total = (
            self.stats.physical_kvc_reclaim_mb + self.stats.physical_expert_reclaim_mb
        )
        self.stats.physical_total_reclaim_mb = total
        self.stats.physical_kvc_reclaim_peak_mb = max(
            self.stats.physical_kvc_reclaim_peak_mb,
            self.stats.physical_kvc_reclaim_mb,
        )
        self.stats.physical_total_reclaim_peak_mb = max(
            self.stats.physical_total_reclaim_peak_mb,
            total,
        )
        if record_step_sample:
            self.stats.physical_kvc_reclaim_step_sum_mb += (
                self.stats.physical_kvc_reclaim_mb
            )
            self.stats.physical_kvc_reclaim_step_count += 1
            self.stats.physical_kvc_reclaim_step_mean_mb = (
                self.stats.physical_kvc_reclaim_step_sum_mb
                / float(max(1, self.stats.physical_kvc_reclaim_step_count))
            )

    def _effective_kvc_reclaim_mb(self, forward_batch: Any) -> float:
        kvc_fraction, expert_fraction, full_supported, reason = self._policy_fractions(
            forward_batch
        )
        policy = self.config.policy
        if policy == "coresid":
            policy = "layer-aware-joint-dp"
        target_mb = self._refresh_reclaim_target_stats(forward_batch)
        effective = min(
            max(0.0, target_mb * kvc_fraction),
            max(0.0, self.stats.available_kvc_reclaim_mb),
        )
        expert_deficit = 0.0
        if self._expert_plan_applied and expert_fraction > 0.0:
            planned_expert = target_mb * expert_fraction
            physical_expert = max(0.0, self.stats.physical_expert_reclaim_mb)
            expert_deficit = max(0.0, planned_expert - physical_expert)
            if expert_deficit > 1e-3:
                if policy == "kv-first":
                    # kv-first means preserve KV residency and satisfy pressure
                    # with expert residency only. Do not silently contaminate it
                    # with KVC reclaim when expert physical reclaim falls short.
                    if not reason:
                        reason = "expert_reclaim_deficit_no_kvc_fallback"
                elif policy in ("layer-aware-joint", "layer-aware-joint-dp"):
                    # CoResid must execute the same split chosen by the DP.  If
                    # expert reclaim lags due to delayed install/replan, do not
                    # silently shift the missing bytes to KVC; that changes the
                    # policy being measured.  The high-watermark expert path
                    # will try to catch up in _apply_expert_plan_once.
                    if not reason:
                        reason = "expert_reclaim_deficit_plan_execution_mismatch"
                else:
                    self.stats.planned_expert_reclaim_mb = physical_expert
                    effective = min(target_mb, effective + expert_deficit)
                    if not reason:
                        reason = "expert_reclaim_deficit_shifted_to_kvc"
        self.stats.requested_total_reclaim_mb = target_mb
        self.stats.effective_kvc_reclaim_mb = effective
        self.stats.policy_kvc_fraction = kvc_fraction
        self.stats.policy_expert_fraction = expert_fraction
        self.stats.full_policy_semantics_supported = full_supported
        if (
            self.config.dynamic_pressure_from_kvc
            and self.config.kvc_backend == "token-slot"
            and effective > 0.0
        ):
            # Under true KV-table pressure, token-slot KVC offload is not a
            # safe steady-state baseline: reloading a required prefix needs new
            # allocator slots at the attention use point, but the allocator may
            # already be full. Keep token-slot baselines to expert offloading
            # only; layer-aware KVC migration uses the per-layer arena backend.
            effective = 0.0
            self.stats.effective_kvc_reclaim_mb = 0.0
            self.stats.planned_kvc_reclaim_mb = 0.0
            if not reason:
                reason = "token_slot_kvc_disabled_under_dynamic_pressure"
            self.stats.policy_semantics_reason = reason
            if not self.stats.target_limited_reason:
                self.stats.target_limited_reason = reason
        self.stats.policy_semantics_reason = reason
        if reason and not self.stats.target_limited_reason:
            self.stats.target_limited_reason = reason
        self.stats.planned_kvc_reclaim_mb = effective
        return effective

    def _effective_expert_reclaim_mb(self, forward_batch: Any) -> float:
        kvc_fraction, expert_fraction, full_supported, reason = self._policy_fractions(
            forward_batch
        )
        target_mb = self._refresh_reclaim_target_stats(forward_batch)
        self._planner_target_high_watermark_mb = max(
            float(self._planner_target_high_watermark_mb), float(target_mb)
        )
        effective = max(0.0, target_mb * expert_fraction)
        self.stats.requested_total_reclaim_mb = target_mb
        self.stats.policy_kvc_fraction = kvc_fraction
        self.stats.policy_expert_fraction = expert_fraction
        self.stats.full_policy_semantics_supported = full_supported
        self.stats.policy_semantics_reason = reason
        self.stats.planned_expert_reclaim_mb = effective
        return effective

    def _expert_evictions_json_from_capacities(self, capacities: Dict[int, int]) -> str:
        try:
            result: Dict[str, int] = {}
            for layer_id, module in self._expert_modules:
                state = self._expert_layers.get(int(layer_id))
                full_num_experts = (
                    int(state.full_num_experts)
                    if state is not None
                    else int(module.w13_weight.data.shape[0])
                )
                result[str(int(layer_id))] = max(
                    0,
                    full_num_experts
                    - int(capacities.get(int(layer_id), full_num_experts)),
                )
            return json.dumps(result, sort_keys=True)
        except Exception:
            return ""

    def _expert_evictions_from_json(self, payload: str) -> Dict[str, int]:
        if not payload:
            return {}
        try:
            raw = json.loads(payload)
            if not isinstance(raw, dict):
                return {}
            return {str(k): max(0, int(v)) for k, v in raw.items()}
        except Exception:
            return {}

    def _applied_expert_evictions_cover_plan(
        self, *, planned: str, applied: str
    ) -> bool:
        planned_counts = self._expert_evictions_from_json(planned)
        applied_counts = self._expert_evictions_from_json(applied)
        if not planned_counts or not applied_counts:
            return False
        return all(
            int(applied_counts.get(layer_id, 0)) >= int(evictions)
            for layer_id, evictions in planned_counts.items()
        )

    def _sync_coresid_expert_plan_stats(self, *, context: str) -> None:
        if self.config.policy != "coresid":
            return
        capacities = self._planned_expert_slot_capacities_by_layer
        applied_capacities: Optional[Dict[int, int]] = None
        if self._expert_plan_applied or self._expert_install_queue:
            applied_capacities = self._current_applied_expert_capacities()
            signature = (
                tuple(sorted((int(k), int(v)) for k, v in capacities.items())),
                tuple(
                    sorted((int(k), int(v)) for k, v in applied_capacities.items())
                ),
                str(context),
            )
            if (
                self._coresid_plan_stats_signature == signature
                and bool(self.stats.expert_plan_match)
                and bool(self.stats.selected_expert_evictions_by_layer)
                and bool(self.stats.applied_expert_evictions_by_layer)
            ):
                return
            self._coresid_plan_stats_signature = signature
        if capacities:
            self.stats.selected_expert_capacity_by_layer = json.dumps(
                {
                    str(layer_id): int(capacity)
                    for layer_id, capacity in capacities.items()
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
                self._expert_evictions_json_from_capacities(capacities)
            )
        if self._expert_plan_applied or self._expert_install_queue:
            self._check_current_coresid_plan_match(
                context=context, applied_capacities=applied_capacities
            )
        elif capacities and not self.stats.applied_expert_evictions_by_layer:
            self.stats.applied_expert_evictions_by_layer = ""
            self.stats.expert_plan_match = False
            self.stats.expert_plan_mismatch_reason = (
                f"planned_expert_plan_not_applied:{context}"
            )
            self.stats.comparable = False
            self.stats.comparability_reason = self.stats.expert_plan_mismatch_reason

    def _coresid_planned_expert_capacities(
        self,
        target_mb: float,
        forward_batch: Any,
    ) -> Dict[int, int]:
        if self.config.policy != "coresid":
            return self._plan_expert_slot_capacities(target_mb)
        if (
            not self._planned_expert_slot_capacities_by_layer
            or float(target_mb)
            > float(self._planned_expert_target_mb)
            + max(1e-3, self._expert_reclaim_quantum_mb())
        ):
            plan_target_mb = max(
                float(target_mb), float(self._planned_expert_target_mb)
            )
            precomputed = self._build_coresid_expert_plan_inputs(forward_batch)
            prefix_table = self._build_expert_prefix_table(precomputed[3])
            (
                _cost,
                churn_count,
                churn_mb,
                install_mb,
                backing_miss_cost,
                materialize_cost,
                capacities,
                layer_costs,
            ) = self._lookup_expert_prefix_cost(
                plan_target_mb,
                precomputed[3],
                prefix_table,
                base_capacities=precomputed[0],
                expert_bytes_by_layer=precomputed[2],
                include_churn_cost=True,
            )
            planned_capacities = {
                int(layer_id): int(capacity)
                for layer_id, capacity in capacities.items()
            }
            self._planned_expert_slot_capacities_by_layer = planned_capacities
            self._planned_expert_cost_by_layer = {
                int(layer_id): float(cost) for layer_id, cost in layer_costs.items()
            }
            self._planned_expert_target_mb = float(plan_target_mb)
            self.stats.planner_estimated_expert_churn_count = float(churn_count)
            self.stats.planner_estimated_expert_churn_mb = float(churn_mb)
            self.stats.planner_estimated_expert_install_mb = float(install_mb)
            self.stats.planner_estimated_expert_backing_miss_cost = float(
                backing_miss_cost
            )
            self.stats.planner_estimated_expert_materialize_cost = float(
                materialize_cost
            )
        capacities = dict(self._planned_expert_slot_capacities_by_layer)
        self.stats.selected_expert_capacity_by_layer = json.dumps(
            {str(layer_id): int(capacity) for layer_id, capacity in capacities.items()},
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
            self._expert_evictions_json_from_capacities(capacities)
        )
        return capacities

    def _record_applied_expert_plan(
        self, capacities: Dict[int, int], *, context: str
    ) -> None:
        applied = self._expert_evictions_json_from_capacities(capacities)
        self.stats.applied_expert_evictions_by_layer = applied
        if self.config.policy != "coresid":
            self.stats.expert_plan_match = True
            self.stats.expert_plan_mismatch_reason = ""
            return
        planned = self.stats.selected_expert_evictions_by_layer
        if planned and applied and planned != applied:
            if self._applied_expert_evictions_cover_plan(
                planned=planned, applied=applied
            ):
                self.stats.expert_plan_match = True
                self.stats.expert_plan_mismatch_reason = ""
                if self.stats.comparability_reason.startswith(
                    (
                        "planned_applied_expert_plan_mismatch:",
                        "missing_planned_expert_plan:",
                        "planned_expert_plan_not_applied:",
                    )
                ):
                    self.stats.comparable = True
                    self.stats.comparability_reason = ""
                return
            self.stats.expert_plan_match = False
            self.stats.expert_plan_mismatch_reason = (
                f"planned_applied_expert_plan_mismatch:{context}"
            )
            self.stats.comparable = False
            self.stats.comparability_reason = self.stats.expert_plan_mismatch_reason
        else:
            self.stats.expert_plan_match = True
            self.stats.expert_plan_mismatch_reason = ""
            if self.stats.comparability_reason.startswith(
                (
                    "planned_applied_expert_plan_mismatch:",
                    "missing_planned_expert_plan:",
                    "planned_expert_plan_not_applied:",
                )
            ):
                self.stats.comparable = True
                self.stats.comparability_reason = ""

    def _pin_coresid_plan_to_current_expert_residency(self, *, context: str) -> None:
        if self.config.policy != "coresid":
            return
        capacities = self._current_applied_expert_capacities()
        if not capacities:
            return
        self._planned_expert_slot_capacities_by_layer = dict(capacities)
        self._planned_expert_cost_by_layer = {}
        self.stats.selected_expert_capacity_by_layer = json.dumps(
            {str(layer_id): int(capacity) for layer_id, capacity in capacities.items()},
            sort_keys=True,
        )
        self.stats.selected_expert_cost_by_layer = ""
        self.stats.selected_expert_evictions_by_layer = (
            self._expert_evictions_json_from_capacities(capacities)
        )
        self.stats.planned_expert_reclaim_mb = max(
            0.0, float(self.stats.physical_expert_reclaim_mb)
        )
        self._record_applied_expert_plan(capacities, context=context)

    def _current_applied_expert_capacities(self) -> Dict[int, int]:
        capacities: Dict[int, int] = {}
        queued = {
            int(item.layer_id): int(item.slot_capacity)
            for item in self._expert_install_queue
        }
        for layer_id, module in self._expert_modules:
            layer_id = int(layer_id)
            state = self._expert_layers.get(layer_id)
            if state is not None:
                capacities[layer_id] = int(state.slot_capacity)
            elif layer_id in queued:
                capacities[layer_id] = int(queued[layer_id])
            else:
                capacities[layer_id] = int(module.w13_weight.data.shape[0])
        return capacities

    def _check_current_coresid_plan_match(
        self,
        *,
        context: str,
        applied_capacities: Optional[Dict[int, int]] = None,
    ) -> None:
        if self.config.policy != "coresid":
            return
        if not self._expert_plan_applied and not self._expert_install_queue:
            return
        planned = self.stats.selected_expert_evictions_by_layer
        if not planned:
            self.stats.expert_plan_match = False
            self.stats.expert_plan_mismatch_reason = (
                f"missing_planned_expert_plan:{context}"
            )
            self.stats.comparable = False
            self.stats.comparability_reason = self.stats.expert_plan_mismatch_reason
            return
        if applied_capacities is None:
            applied_capacities = self._current_applied_expert_capacities()
        applied = self._expert_evictions_json_from_capacities(applied_capacities)
        self.stats.applied_expert_evictions_by_layer = applied
        if applied != planned:
            if self._applied_expert_evictions_cover_plan(
                planned=planned, applied=applied
            ):
                self.stats.expert_plan_match = True
                self.stats.expert_plan_mismatch_reason = ""
                if self.stats.comparability_reason.startswith(
                    (
                        "planned_applied_expert_plan_mismatch:",
                        "missing_planned_expert_plan:",
                        "planned_expert_plan_not_applied:",
                    )
                ):
                    self.stats.comparable = True
                    self.stats.comparability_reason = ""
                return
            self.stats.expert_plan_match = False
            self.stats.expert_plan_mismatch_reason = (
                f"planned_applied_expert_plan_mismatch:{context}"
            )
            self.stats.comparable = False
            self.stats.comparability_reason = self.stats.expert_plan_mismatch_reason
        else:
            self.stats.expert_plan_match = True
            self.stats.expert_plan_mismatch_reason = ""
            if self.stats.comparability_reason.startswith(
                (
                    "planned_applied_expert_plan_mismatch:",
                    "missing_planned_expert_plan:",
                    "planned_expert_plan_not_applied:",
                )
            ):
                self.stats.comparable = True
                self.stats.comparability_reason = ""

    def _apply_expert_plan_once(self, forward_batch: Any) -> None:
        if self.config.mode != "kvc-expert":
            return
        if self.config.expert_collector_only:
            self.stats.planned_expert_reclaim_mb = 0.0
            self.stats.policy_expert_fraction = 0.0
            return
        if (
            self.config.dynamic_pressure_from_kvc
            and not self._expert_install_queue
            and self._dynamic_runtime_pressure_mb(forward_batch) <= 1e-3
        ):
            self._set_no_pressure_reclaim_stats(
                max(0.0, float(self.config.reclaim_limit_mb))
            )
            return
        if self._no_pressure_fast_path_active(forward_batch):
            self.stats.layerkv_no_pressure_expert_skip_count += 1
            return
        if self._expert_install_queue or self._expert_install_state in {
            "installing_slots",
            "queued",
        }:
            replan_needed, observed_target_mb = self._dynamic_expert_replan_needed(
                forward_batch
            )
            if not replan_needed:
                self.stats.requested_total_reclaim_mb = observed_target_mb
                self._refresh_expert_install_progress()
                return
        elif self._expert_plan_applied:
            replan_needed, observed_target_mb = self._dynamic_expert_replan_needed(
                forward_batch
            )
            if not replan_needed:
                self.stats.requested_total_reclaim_mb = observed_target_mb
                if self.config.policy == "coresid":
                    self._sync_coresid_expert_plan_stats(context="stable_high_watermark")
                return
        target_mb = self._effective_expert_reclaim_mb(forward_batch)
        expert_target_mb = target_mb
        if self._policy_name() in ("layer-aware-joint", "layer-aware-joint-dp"):
            expert_target_mb = max(0.0, float(self.stats.planned_expert_reclaim_mb))
        expert_quantum_mb = max(1e-3, self._expert_reclaim_quantum_mb())
        if self._expert_install_queue or self._expert_install_state in {
            "installing_slots",
            "queued",
        }:
            if (
                self._dynamic_expert_churn_policy_enabled()
                and expert_target_mb
                > float(self._expert_install_target_mb) + expert_quantum_mb
            ):
                self._raise_expert_install_target(expert_target_mb, forward_batch)
            return
        if self._expert_plan_applied:
            if (
                self._dynamic_expert_churn_policy_enabled()
                and expert_target_mb
                > self.stats.physical_expert_reclaim_mb + expert_quantum_mb
            ):
                self._increase_expert_reclaim_to_target(expert_target_mb, forward_batch)
            elif self._dynamic_expert_churn_policy_enabled():
                self._pin_coresid_plan_to_current_expert_residency(
                    context="stable_within_expert_quantum"
                )
            return
        if expert_target_mb <= 0:
            if self.config.dynamic_pressure_from_kvc:
                # In trace-driven hard-pressure mode, the KV table may be below
                # capacity during early prefill/decode and exceed it later.  Do
                # not permanently complete the expert plan on a zero-pressure
                # sample; keep it eligible for the first step that actually
                # needs expert reclaim.
                self._expert_install_state = "waiting_for_pressure"
            else:
                self._expert_plan_applied = True
                self._expert_install_state = "complete"
            self._refresh_expert_install_progress()
            self._refresh_expert_stats()
            return
        if not self.physical_expert_supported:
            self.stats.comparable = False
            self.stats.comparability_reason = self._expert_support_reason()
            return
        if not self._expert_modules:
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = "no supported expert modules discovered"
            self.stats.comparable = False
            self.stats.comparability_reason = self.stats.expert_guard_reason
            return

        # Standard FusedMoE executes all unique experts for the current dispatch
        # in one kernel call. A shrunk slot table must therefore be at least as
        # large as the per-layer decode unique expert set. Prefill often touches
        # every expert in batch-heavy workloads, while decode can be sparser; wait
        # for one decode routing sample before committing the physical slot plan.
        if not self._has_decode_expert_hotness():
            self.stats.policy_semantics_reason = "waiting_for_decode_expert_hotness"
            return

        full_bytes = sum(
            self._expert_full_bytes(module) for _, module in self._expert_modules
        )
        if full_bytes <= 0:
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = "failed to inspect expert bytes"
            return

        with self._profile("profile_plan_expert_capacity_ms"):
            slot_capacities = self._coresid_planned_expert_capacities(
                expert_target_mb, forward_batch
            )
        self._record_applied_expert_plan(slot_capacities, context="initial_apply")
        try:
            self.stats.selected_expert_evictions_by_layer = json.dumps(
                {
                    str(layer_id): max(
                        0,
                        int(
                            self._expert_layers[layer_id].full_num_experts
                            if layer_id in self._expert_layers
                            else module.w13_weight.data.shape[0]
                        )
                        - int(
                            slot_capacities.get(
                                layer_id, module.w13_weight.data.shape[0]
                            )
                        ),
                    )
                    for layer_id, module in self._expert_modules
                },
                sort_keys=True,
            )
        except Exception:
            self.stats.selected_expert_evictions_by_layer = ""
        if self.config.policy == "coresid":
            self.stats.selected_expert_evictions_by_layer = (
                self._expert_evictions_json_from_capacities(slot_capacities)
            )
        prepared_cpu_params: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        prepared_initial_resident: Dict[int, List[int]] = {}
        if self._expert_prepare_done and self._prepared_expert_plan is not None:
            prepared_kind = str(self._prepared_expert_plan.get("kind", "full"))
            prepared_cpu_params = self._prepared_expert_plan.get(
                "cpu_params_by_layer", {}
            )
            prepared_initial_resident = self._prepared_expert_plan.get(
                "initial_resident_by_layer", {}
            )
            prepared_capacities = self._prepared_expert_plan.get("slot_capacities", {})
            prepared_target_mb = float(
                self._prepared_expert_plan.get("target_mb", 0.0) or 0.0
            )
            if prepared_kind == "backing_only":
                prepared_initial_resident = {}
                self.stats.expert_prepared_plan_used = True
            elif abs(prepared_target_mb - expert_target_mb) > 1e-3 or any(
                int(prepared_capacities.get(layer_id, -1))
                != int(slot_capacities[layer_id])
                for layer_id, _module in self._expert_modules
            ):
                prepared_cpu_params = {}
                prepared_initial_resident = {}
                self._prepared_expert_plan = None
                self._expert_prepare_done = False
                self.stats.expert_prepare_invalidated_count += 1
                self.stats.expert_prepare_decode_fallback_count += 1
            else:
                self.stats.expert_prepared_plan_used = True
        else:
            self.stats.expert_prepare_decode_fallback_count += 1

        queue: List[_LayerKVExpertInstallItem] = []
        for layer_id, module in self._expert_modules:
            slot_capacity = slot_capacities[layer_id]
            initial_resident = prepared_initial_resident.get(layer_id)
            full_num_experts = int(module.w13_weight.data.shape[0])
            if initial_resident is not None and (
                len(initial_resident) != slot_capacity
                or any(
                    int(x) < 0 or int(x) >= full_num_experts for x in initial_resident
                )
            ):
                initial_resident = None
            if initial_resident is None:
                initial_resident = self._select_initial_resident_experts(
                    layer_id=layer_id,
                    full_num_experts=full_num_experts,
                    slot_capacity=slot_capacity,
                )
            queue.append(
                _LayerKVExpertInstallItem(
                    layer_id=int(layer_id),
                    module=module,
                    slot_capacity=int(slot_capacity),
                    initial_resident=initial_resident,
                    prepared_cpu_params=prepared_cpu_params.get(layer_id),
                )
            )
        self._expert_install_queue = queue
        self._expert_install_target_mb = float(expert_target_mb)
        self._expert_install_state = "queued"
        self._prepared_expert_plan = None
        self._refresh_expert_install_progress()

    def _raise_expert_install_target(
        self, target_mb: float, forward_batch: Any
    ) -> None:
        """Raise an in-flight expert install plan to a new high-watermark target."""

        if target_mb <= float(self._expert_install_target_mb) + 1e-3:
            return
        if not self._has_decode_expert_hotness():
            return
        with self._profile("profile_plan_expert_capacity_ms"):
            slot_capacities = self._coresid_planned_expert_capacities(
                target_mb, forward_batch
            )
        self._record_applied_expert_plan(slot_capacities, context="raise_target")
        changed = 0
        for item in self._expert_install_queue:
            new_capacity = int(
                slot_capacities.get(int(item.layer_id), item.slot_capacity)
            )
            if new_capacity < int(item.slot_capacity):
                full_num_experts = (
                    int(self._expert_layers.get(int(item.layer_id)).full_num_experts)
                    if int(item.layer_id) in self._expert_layers
                    else int(item.module.w13_weight.data.shape[0])
                )
                item.slot_capacity = new_capacity
                item.initial_resident = self._select_initial_resident_experts(
                    layer_id=int(item.layer_id),
                    full_num_experts=full_num_experts,
                    slot_capacity=new_capacity,
                )
                item.prepared_cpu_params = None
                changed += 1
        for layer_id, state in sorted(self._expert_layers.items()):
            new_capacity = int(slot_capacities.get(int(layer_id), state.slot_capacity))
            if new_capacity < int(state.slot_capacity):
                self._shrink_installed_expert_layer_slots(state, new_capacity)
                changed += 1
        self._expert_install_target_mb = float(target_mb)
        if changed:
            self.stats.expert_slot_rebind_count += changed
            with self._profile("profile_refresh_expert_stats_ms"):
                self._refresh_expert_stats()
        self._refresh_expert_install_progress()

    def _increase_expert_reclaim_to_target(
        self, target_mb: float, forward_batch: Any
    ) -> None:
        """Increase expert reclaim for dynamic kv-first without touching KVC.

        Initial expert install is intentionally budgeted and one-shot for fixed
        experiments, but trace-driven pressure is a high-watermark signal: if
        later decode steps need more headroom, kv-first must keep sacrificing
        expert residency instead of silently relying on SGLang admission or
        falling back to KV eviction.
        """

        if not self._expert_layers:
            return
        if not self._has_decode_expert_hotness():
            self.stats.policy_semantics_reason = "waiting_for_decode_expert_hotness"
            return
        current_mb = max(0.0, float(self.stats.physical_expert_reclaim_mb))
        if target_mb <= current_mb + 1e-3:
            return
        with self._profile("profile_plan_expert_capacity_ms"):
            slot_capacities = self._coresid_planned_expert_capacities(
                target_mb, forward_batch
            )
        self._record_applied_expert_plan(slot_capacities, context="increase_target")
        changed = 0
        for layer_id, state in sorted(self._expert_layers.items()):
            new_capacity = int(slot_capacities.get(int(layer_id), state.slot_capacity))
            if new_capacity >= int(state.slot_capacity):
                continue
            self._shrink_installed_expert_layer_slots(state, new_capacity)
            changed += 1
        if changed:
            self.stats.expert_slot_rebind_count += changed
            self._expert_install_target_mb = max(
                float(self._expert_install_target_mb), float(target_mb)
            )
            with self._profile("profile_refresh_expert_stats_ms"):
                self._refresh_expert_stats()
        if self.stats.physical_expert_reclaim_mb + 1e-3 < target_mb:
            self.stats.policy_semantics_reason = (
                "expert_reclaim_deficit_no_kvc_fallback"
            )
        else:
            self.stats.policy_semantics_reason = ""
        self._refresh_expert_install_progress()

    def _shrink_installed_expert_layer_slots(
        self, state: _LayerKVExpertLayerState, new_capacity: int
    ) -> None:
        new_capacity = max(1, min(int(new_capacity), int(state.slot_capacity)))
        if new_capacity >= int(state.slot_capacity):
            return
        current_resident = list(state.logical_to_slot.keys())
        candidate_order = self._expert_candidate_order_by_layer.get(
            int(state.layer_id)
        )
        if candidate_order:
            self.stats.expert_candidate_order_hit_count += 1
            rank = {int(expert_id): i for i, expert_id in enumerate(candidate_order)}
            current_resident.sort(
                key=lambda expert_id: (
                    int(rank.get(int(expert_id), len(rank) + int(expert_id))),
                    int(expert_id),
                )
            )
        else:
            self.stats.expert_candidate_order_miss_count += 1
            current_resident.sort(
                key=lambda expert_id: (
                    -int(state.hotness_decode.get(int(expert_id), 0)),
                    -int(state.hotness_prefill.get(int(expert_id), 0)),
                    int(expert_id),
                )
            )
        keep = [int(x) for x in current_resident[:new_capacity]]
        keep_set = set(keep)
        evict_pairs = [
            (int(logical_id), int(slot_id))
            for logical_id, slot_id in list(state.logical_to_slot.items())
            if int(logical_id) not in keep_set
        ]
        copied = self._copy_slots_to_cpu_batched(state, evict_pairs)
        state.cpu_params.update(copied)
        with torch.no_grad():
            for name in state.param_names:
                param = getattr(state.module, name)
                old = param.data
                new_data = torch.empty(
                    (new_capacity,) + tuple(old.shape[1:]),
                    dtype=old.dtype,
                    device=old.device,
                )
                for slot_id, logical_id in enumerate(keep):
                    old_slot = state.logical_to_slot[int(logical_id)]
                    new_data[slot_id].copy_(old[int(old_slot)])
                param.data = new_data
        state.slot_capacity = int(new_capacity)
        state.logical_to_slot = {
            int(expert_id): slot_id for slot_id, expert_id in enumerate(keep)
        }
        state.slot_to_logical = {
            slot_id: int(expert_id) for slot_id, expert_id in enumerate(keep)
        }
        state.lru = {
            int(expert_id): state.lru.get(int(expert_id), self._decode_step)
            for expert_id in keep
        }
        state.free_slots.clear()
        state.lru_heap.clear()
        for expert_id, slot_id in state.logical_to_slot.items():
            heapq.heappush(state.lru_heap, (state.lru[expert_id], slot_id, expert_id))
        if state.remap_tensor is not None:
            state.remap_tensor.fill_(-1)
            if keep:
                ids = torch.tensor(keep, dtype=torch.long, device=state.device)
                slots = torch.arange(len(keep), dtype=torch.long, device=state.device)
                state.remap_tensor[ids] = slots
        try:
            state.module.num_experts = int(new_capacity)
            state.module.num_local_experts = int(new_capacity)
            state.module.moe_runner_config.num_experts = int(new_capacity)
            state.module.moe_runner_config.num_local_experts = int(new_capacity)
            state.module.dispatcher.num_experts = int(new_capacity)
            state.module.dispatcher.num_local_experts = int(new_capacity)
            state.module.dispatcher.num_local_routed_experts = int(new_capacity)
        except Exception:
            pass
        self._sync_expert_groups_for_state(state)
        self._expert_group_dirty_layers.add(int(state.layer_id))

    def _refresh_expert_install_progress(self) -> None:
        pending = len(self._expert_install_queue)
        queued_install_d2h = sum(
            len(job.expert_ids) for job in self._expert_install_d2h_queue
        )
        pending_install_d2h = sum(
            len(pending_copy.copied)
            for pending_copy in self._pending_expert_d2h_events
            if str(pending_copy.reason).startswith("install")
        )
        completed = len(self._expert_layers)
        state = self._expert_install_state
        if queued_install_d2h > 0:
            state = "queued_backing"
        elif pending_install_d2h > 0:
            state = "installing_backing"
        elif self._expert_plan_applied:
            state = "complete"
        elif pending > 0 and not state:
            state = "queued"
        self.stats.expert_install_state = state
        self.stats.expert_install_pending_layers = pending
        self.stats.expert_install_completed_layers = completed
        self.stats.expert_install_layers_per_step = int(
            self._expert_install_layers_per_step
        )
        self.stats.expert_install_budget_mb = float(self._expert_install_budget_mb)
        self.stats.expert_install_target_steps = int(self._expert_install_target_steps)
        self.stats.expert_install_d2h_queue_length = int(queued_install_d2h)
        self.stats.expert_install_d2h_budget_mb = float(
            self.config.expert_copy_budget_mb
        )
        if queued_install_d2h <= 0:
            self.stats.expert_install_d2h_effective_budget_mb = float(
                self.config.expert_copy_budget_mb
            )
        self.stats.expert_install_d2h_max_budget_mb = float(
            self.config.expert_copy_max_budget_mb
        )
        self.stats.expert_install_d2h_chunk_mb = float(
            self.config.expert_copy_chunk_mb
        )
        self.stats.expert_install_d2h_lookahead_layers = int(
            self.config.expert_copy_lookahead_layers
        )
        self.stats.expert_install_reclaim_mb_progress = float(
            self.stats.physical_expert_reclaim_mb
        )
        if pending > 0 or queued_install_d2h > 0 or pending_install_d2h > 0:
            self.stats.comparable = False
            self.stats.comparability_reason = "INSTALL_IN_PROGRESS_NOT_COMPARABLE"
        elif self.stats.comparability_reason == "INSTALL_IN_PROGRESS_NOT_COMPARABLE":
            self.stats.comparable = True
            self.stats.comparability_reason = ""

    def _enqueue_expert_install_d2h_job(
        self,
        *,
        layer_id: int,
        module: Any,
        param_names: List[str],
        expert_ids: List[int],
        target_cpu_params: Dict[int, Dict[str, torch.Tensor]],
        priority: int = 0,
        deadline_step: Optional[int] = None,
    ) -> None:
        expert_ids = [int(expert_id) for expert_id in expert_ids]
        if not expert_ids:
            return
        self._expert_install_d2h_job_seq += 1
        self._expert_install_d2h_queue.append(
            _LayerKVExpertInstallD2HJob(
                seq=int(self._expert_install_d2h_job_seq),
                layer_id=int(layer_id),
                module=module,
                param_names=list(param_names),
                expert_ids=expert_ids,
                target_cpu_params=target_cpu_params,
                priority=int(priority),
                deadline_step=(
                    int(deadline_step)
                    if deadline_step is not None
                    else int(self._decode_step + 1)
                ),
            )
        )
        self.stats.expert_install_d2h_queued_count += len(expert_ids)
        self._refresh_expert_install_progress()

    def _queued_expert_install_d2h_bytes(self) -> int:
        total = 0
        for job in self._expert_install_d2h_queue:
            total += len(job.expert_ids) * max(1, int(self._expert_bytes(job.module)))
        return int(total)

    def _pending_expert_install_d2h_count(self) -> int:
        return sum(
            len(pending_copy.copied)
            for pending_copy in self._pending_expert_d2h_events
            if str(pending_copy.reason).startswith("install")
        )

    def _effective_expert_install_d2h_budget_mb(self) -> float:
        budget_mb = max(0.0, float(self.config.expert_copy_budget_mb))
        target_steps = max(0, int(self._expert_install_target_steps))
        if target_steps > 0 and self._expert_install_d2h_queue:
            remaining_steps = max(
                1, target_steps - int(self.stats.forward_decode_count)
            )
            queued_mb = self._queued_expert_install_d2h_bytes() / float(1024 * 1024)
            horizon_budget_mb = queued_mb / float(remaining_steps)
            if horizon_budget_mb > budget_mb + 1e-3:
                budget_mb = horizon_budget_mb
                self.stats.expert_install_d2h_dynamic_budget_count += 1
        max_budget_mb = max(0.0, float(self.config.expert_copy_max_budget_mb))
        if max_budget_mb > 0.0:
            budget_mb = min(budget_mb, max_budget_mb)
        self.stats.expert_install_d2h_effective_budget_mb = float(budget_mb)
        return float(budget_mb)

    def _prepare_expert_install_d2h_lookahead(self) -> None:
        lookahead = max(0, int(self.config.expert_copy_lookahead_layers))
        if lookahead <= 0 or not self._expert_install_queue:
            return
        prepared = 0
        for item in self._expert_install_queue[:lookahead]:
            before = bool(item.backing_queued)
            self._prepare_expert_install_item_backing(item)
            if not before and item.backing_queued:
                prepared += 1
        if prepared:
            self.stats.expert_install_d2h_lookahead_queue_count += prepared

    def _prepare_expert_install_item_backing(
        self, item: _LayerKVExpertInstallItem
    ) -> bool:
        if item.prepared_cpu_params is None:
            item.prepared_cpu_params = {}
        if item.backing_queued:
            item.pending_cpu_experts = {
                int(expert_id)
                for expert_id in item.pending_cpu_experts
                if int(expert_id) not in item.prepared_cpu_params
            }
            return not item.pending_cpu_experts

        module = item.module
        layer_id = int(item.layer_id)
        param_names = self._expert_param_names(module)
        full_num_experts = int(module.w13_weight.data.shape[0])
        resident_set = set(int(x) for x in (item.initial_resident or []))
        missing_cpu_experts: List[int] = []
        for expert_id in range(full_num_experts):
            if int(expert_id) in resident_set:
                continue
            if int(expert_id) in item.prepared_cpu_params:
                self.stats.expert_prepared_backing_hit_count += 1
                continue
            global_params = (
                self._global_expert_backing(layer_id, expert_id)
                if self._expert_global_cpu_backing
                else None
            )
            if global_params is not None:
                item.prepared_cpu_params[int(expert_id)] = global_params
                self.stats.expert_prepared_backing_hit_count += 1
                continue
            self.stats.expert_prepared_backing_miss_count += 1
            missing_cpu_experts.append(int(expert_id))
        if missing_cpu_experts:
            device = getattr(module, param_names[0]).data.device if param_names else None
            if (
                not self._optimized_profile_enabled()
                or device is None
                or device.type != "cuda"
            ):
                return True
            item.pending_cpu_experts = set(missing_cpu_experts)
            item.backing_queued = True
            self._enqueue_expert_install_d2h_job(
                layer_id=layer_id,
                module=module,
                param_names=param_names,
                expert_ids=missing_cpu_experts,
                target_cpu_params=item.prepared_cpu_params,
                priority=0,
                deadline_step=self._decode_step + 1,
            )
            return False
        item.pending_cpu_experts.clear()
        item.backing_queued = True
        return True

    def _submit_expert_install_d2h_budgeted(
        self, *, budget_mb: Optional[float] = None
    ) -> None:
        if not self._expert_install_d2h_queue:
            return
        if budget_mb is None:
            budget_mb = self._effective_expert_install_d2h_budget_mb()
        else:
            self.stats.expert_install_d2h_effective_budget_mb = float(budget_mb)
        budget_bytes = int(max(0.0, float(budget_mb)) * 1024 * 1024)
        if budget_bytes <= 0:
            return
        chunk_cap_bytes = int(
            max(1.0, float(self.config.expert_copy_chunk_mb)) * 1024 * 1024
        )
        submitted_bytes = 0
        submitted_experts = 0
        made_progress = False
        self._expert_install_d2h_queue.sort(
            key=lambda job: (
                int(job.deadline_step),
                -int(job.priority),
                int(job.seq),
            )
        )
        while self._expert_install_d2h_queue and submitted_bytes < budget_bytes:
            job = self._expert_install_d2h_queue[0]
            if not job.expert_ids:
                self._expert_install_d2h_queue.pop(0)
                continue
            expert_bytes = max(1, int(self._expert_bytes(job.module)))
            remaining_budget = max(0, budget_bytes - submitted_bytes)
            chunk_budget = max(1, min(chunk_cap_bytes, remaining_budget))
            take = max(1, min(len(job.expert_ids), chunk_budget // expert_bytes))
            expert_ids = [int(x) for x in job.expert_ids[:take]]
            copied_bytes = self._copy_experts_to_cpu_for_install_batched_async(
                job.module,
                job.param_names,
                expert_ids,
                layer_id=int(job.layer_id),
                target_cpu_params=job.target_cpu_params,
            )
            if copied_bytes < 0:
                copied = self._copy_experts_to_cpu_for_install_batched(
                    job.module,
                    job.param_names,
                    expert_ids,
                    layer_id=int(job.layer_id),
                )
                job.target_cpu_params.update(copied)
                copied_bytes = sum(
                    self._expert_backing_bytes(params) for params in copied.values()
                )
            job.expert_ids = [int(x) for x in job.expert_ids[take:]]
            if not job.expert_ids:
                self._expert_install_d2h_queue.pop(0)
            submitted_bytes += max(1, int(copied_bytes))
            submitted_experts += len(expert_ids)
            made_progress = True
        if made_progress:
            self.stats.expert_install_d2h_submit_step_count += 1
            self._refresh_expert_install_progress()

    def _force_drain_expert_install_d2h_and_slots(self) -> None:
        if not self.config.expert_copy_force_drain:
            return
        if (
            not self._expert_install_queue
            and not self._expert_install_d2h_queue
            and self._pending_expert_install_d2h_count() <= 0
        ):
            return
        start = time.perf_counter()
        self.stats.expert_install_d2h_force_drain_count += 1
        max_iters = max(8, len(self._expert_modules) * 4 + 8)
        for _ in range(max_iters):
            before = (
                len(self._expert_install_queue),
                self._queued_expert_install_d2h_bytes(),
                self._pending_expert_install_d2h_count(),
                len(self._expert_layers),
            )
            queued_mb = self._queued_expert_install_d2h_bytes() / float(1024 * 1024)
            if queued_mb > 0.0:
                self._submit_expert_install_d2h_budgeted(budget_mb=queued_mb)
            if self._pending_expert_install_d2h_count() > 0:
                self._finalize_expert_d2h_events(block=True)
            self._advance_expert_install_budgeted()
            after = (
                len(self._expert_install_queue),
                self._queued_expert_install_d2h_bytes(),
                self._pending_expert_install_d2h_count(),
                len(self._expert_layers),
            )
            if after[0] == 0 and after[1] == 0 and after[2] == 0:
                break
            if after == before:
                break
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.stats.expert_install_d2h_force_drain_ms += elapsed_ms
        self._refresh_expert_install_progress()

    def _advance_expert_install_budgeted(self) -> None:
        if self.config.mode != "kvc-expert" or not self._expert_install_queue:
            self._refresh_expert_install_progress()
            return
        start = time.perf_counter()
        installed = 0
        installed_mb = 0.0
        target_steps = max(0, int(self._expert_install_target_steps))
        max_layers = max(1, int(self._expert_install_layers_per_step))
        remaining_steps = 0
        if target_steps > 0:
            remaining_steps = max(
                1, target_steps - int(self.stats.forward_decode_count)
            )
            needed_layers = int(
                math.ceil(len(self._expert_install_queue) / float(remaining_steps))
            )
            max_layers = max(max_layers, needed_layers)
        max_mb = max(0.0, float(self._expert_install_budget_mb))
        if target_steps > 0 and self._expert_install_queue:
            remaining_reclaim_mb = 0.0
            for item in self._expert_install_queue:
                remaining_reclaim_mb += max(
                    0.0,
                    (
                        int(item.module.w13_weight.data.shape[0])
                        - int(item.slot_capacity)
                    )
                    * float(self._expert_bytes(item.module))
                    / float(1024 * 1024),
                )
            max_mb = max(max_mb, remaining_reclaim_mb / float(max(1, remaining_steps)))
        self.stats.expert_install_remaining_steps = int(remaining_steps)
        self.stats.expert_install_effective_layers_this_step = int(max_layers)
        self.stats.expert_install_effective_budget_mb_this_step = float(max_mb)
        self._prepare_expert_install_d2h_lookahead()
        self._expert_install_state = "installing_slots"
        install_profile = (
            "profile_apply_prepared_expert_plan_ms"
            if self._expert_install_queue[0].prepared_cpu_params
            else "profile_install_expert_slots_ms"
        )
        with self._profile(install_profile):
            while self._expert_install_queue and installed < max_layers:
                item = self._expert_install_queue[0]
                if not self._prepare_expert_install_item_backing(item):
                    break
                layer_reclaim_mb = max(
                    0.0,
                    (
                        int(item.module.w13_weight.data.shape[0])
                        - int(item.slot_capacity)
                    )
                    * float(self._expert_bytes(item.module))
                    / float(1024 * 1024),
                )
                if (
                    installed > 0
                    and max_mb > 0.0
                    and installed_mb + layer_reclaim_mb > max_mb
                ):
                    break
                self._expert_install_queue.pop(0)
                state = self._install_expert_layer_slots(
                    item.module,
                    item.layer_id,
                    item.slot_capacity,
                    initial_resident=item.initial_resident,
                    prepared_cpu_params=item.prepared_cpu_params,
                )
                self._expert_layers[item.layer_id] = state
                installed += 1
                installed_mb += layer_reclaim_mb
        if installed > 0:
            self.stats.expert_slot_rebind_count += installed
            self.stats.expert_install_step_count += 1
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.stats.expert_install_step_ms += elapsed_ms
            self.stats.expert_install_blocking_ms += elapsed_ms
            with self._profile("profile_refresh_expert_stats_ms"):
                self._refresh_expert_stats()
        if self._expert_install_queue:
            self.stats.expert_install_not_comparable_step_count += 1
            self._refresh_expert_install_progress()
            return
        self._expert_plan_applied = True
        self._expert_install_state = "complete"
        with self._profile("profile_refresh_expert_stats_ms"):
            self._refresh_expert_stats()
        if (
            self.stats.physical_expert_reclaim_mb + 1e-3
            < self._expert_install_target_mb
        ):
            if self._policy_name() == "kv-first":
                self.stats.policy_semantics_reason = (
                    "expert_reclaim_deficit_no_kvc_fallback"
                )
            elif self._policy_name() in ("layer-aware-joint", "layer-aware-joint-dp"):
                self.stats.policy_semantics_reason = (
                    "expert_reclaim_deficit_plan_execution_mismatch"
                )
            else:
                self.stats.planned_expert_reclaim_mb = (
                    self.stats.physical_expert_reclaim_mb
                )
                self.stats.policy_semantics_reason = (
                    "expert_reclaim_deficit_shifted_to_kvc"
                )
        self._refresh_expert_install_progress()

    def _plan_expert_slot_capacities(self, target_mb: float) -> Dict[int, int]:
        target_bytes = max(0, int(target_mb * 1024 * 1024))
        layer_infos: List[Dict[str, int]] = []
        for layer_id, module in self._expert_modules:
            state = self._expert_layers.get(int(layer_id))
            full_num_experts = (
                int(state.full_num_experts)
                if state is not None
                else int(module.w13_weight.data.shape[0])
            )
            top_k = int(getattr(module, "top_k", 0) or 0)
            if top_k <= 0:
                top_k = int(getattr(module.moe_runner_config, "top_k", 1) or 1)
            decode_unique = len(self._expert_hotness_decode.get(int(layer_id), {}))
            if self._dynamic_expert_churn_policy_enabled():
                # kv-first is the pure expert-offload baseline.  It must be
                # allowed to trade expert churn for KV residency. CoResid must
                # have the same execution capability for expert candidates it
                # explicitly selects; otherwise the DP is biased toward KVC by a
                # runtime limitation rather than the cost model.
                min_capacity = max(1, min(full_num_experts, top_k))
            else:
                min_capacity = max(1, min(full_num_experts, max(top_k, decode_unique)))
            layer_infos.append(
                {
                    "layer_id": int(layer_id),
                    "capacity": full_num_experts,
                    "min_capacity": min_capacity,
                    "expert_bytes": (
                        int(state.expert_bytes)
                        if state is not None
                        else self._expert_bytes(module)
                    ),
                }
            )

        reclaimed = 0
        layer_infos.sort(key=lambda x: x["layer_id"])
        while reclaimed < target_bytes:
            eligible = [
                info for info in layer_infos if info["capacity"] > info["min_capacity"]
            ]
            if not eligible:
                break
            max_expert_bytes = max(
                1, max(int(info["expert_bytes"]) for info in eligible)
            )
            remaining_bytes = max(0, target_bytes - reclaimed)
            take_count = min(
                len(eligible),
                max(1, int(math.ceil(remaining_bytes / float(max_expert_bytes)))),
            )
            for info in eligible[:take_count]:
                info["capacity"] -= 1
                reclaimed += info["expert_bytes"]
                if reclaimed >= target_bytes:
                    break
            if take_count <= 0:
                break
        return {info["layer_id"]: info["capacity"] for info in layer_infos}

    def _evenly_spaced_layer_infos(
        self, layer_infos: List[Dict[str, int]], take_count: int
    ) -> List[Dict[str, int]]:
        if take_count <= 0 or not layer_infos:
            return []
        if take_count >= len(layer_infos):
            return list(layer_infos)
        n = len(layer_infos)
        selected: List[Dict[str, int]] = []
        used: Set[int] = set()
        # Place picks at bucket centers so small expert reclaim targets are
        # distributed across model depth instead of concentrated in early layers.
        for i in range(take_count):
            idx = int(math.floor((i + 0.5) * n / float(take_count)))
            idx = min(n - 1, max(0, idx))
            while idx in used and idx + 1 < n:
                idx += 1
            while idx in used and idx - 1 >= 0:
                idx -= 1
            if idx in used:
                continue
            used.add(idx)
            selected.append(layer_infos[idx])
        return selected

    def _expert_param_names(self, module: Any) -> List[str]:
        names = ["w13_weight", "w2_weight"]
        for optional in ("w13_weight_bias", "w2_weight_bias"):
            if hasattr(module, optional):
                names.append(optional)
        return names

    def _expert_full_bytes(self, module: Any) -> int:
        total = 0
        for name in self._expert_param_names(module):
            tensor = getattr(module, name).data
            total += int(tensor.nbytes)
        return total

    def _expert_bytes(self, module: Any) -> int:
        total = 0
        for name in self._expert_param_names(module):
            tensor = getattr(module, name).data
            total += int(tensor[0].nbytes)
        return total

    def _install_expert_layer_slots(
        self,
        module: Any,
        layer_id: int,
        slot_capacity: int,
        initial_resident: Optional[List[int]] = None,
        prepared_cpu_params: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ) -> _LayerKVExpertLayerState:
        if getattr(module, "_layerkv_expert_wrapped", False):
            return self._expert_layers[layer_id]

        param_names = self._expert_param_names(module)
        full_num_experts = int(module.w13_weight.data.shape[0])
        device = module.w13_weight.data.device
        dtype = module.w13_weight.data.dtype
        cpu_params: Dict[int, Dict[str, torch.Tensor]] = {}
        expert_bytes = 0

        with torch.no_grad():
            for name in param_names:
                param = getattr(module, name)
                expert_bytes += int(param.data[0].nbytes)

            if initial_resident is None:
                initial_resident = self._select_initial_resident_experts(
                    layer_id=layer_id,
                    full_num_experts=full_num_experts,
                    slot_capacity=slot_capacity,
                )
            with self._profile("profile_apply_expert_slot_map_ms"):
                logical_to_slot = {
                    expert_id: slot_id
                    for slot_id, expert_id in enumerate(initial_resident)
                }
                slot_to_logical = {
                    slot_id: expert_id
                    for slot_id, expert_id in enumerate(initial_resident)
                }
                lru = {expert_id: self._decode_step for expert_id in initial_resident}
                resident_set = set(initial_resident)
            if prepared_cpu_params is not None:
                cpu_params.update(prepared_cpu_params)
                removed_bytes = 0
                for expert_id in list(cpu_params):
                    if expert_id in resident_set:
                        removed = cpu_params.pop(expert_id)
                        removed_bytes += sum(int(t.nbytes) for t in removed.values())
                        self._release_expert_backing_params(removed)
                if removed_bytes:
                    self._expert_host_backing_bytes -= removed_bytes
                    self._refresh_expert_host_backing_stat()
            missing_cpu_experts: List[int] = []
            for expert_id in range(full_num_experts):
                if expert_id in resident_set:
                    continue
                if expert_id in cpu_params:
                    self.stats.expert_prepared_backing_hit_count += 1
                    continue
                global_params = (
                    self._global_expert_backing(layer_id, expert_id)
                    if self._expert_global_cpu_backing
                    else None
                )
                if global_params is not None:
                    cpu_params[int(expert_id)] = global_params
                    self.stats.expert_prepared_backing_hit_count += 1
                    continue
                self.stats.expert_prepared_backing_miss_count += 1
                missing_cpu_experts.append(expert_id)
            if missing_cpu_experts:
                self.stats.expert_install_d2h_sync_fallback_count += len(
                    missing_cpu_experts
                )
                cpu_params.update(
                    self._copy_experts_to_cpu_for_install_batched(
                        module,
                        param_names,
                        missing_cpu_experts,
                        layer_id=layer_id,
                    )
                )

            with self._profile("profile_shrink_expert_weight_ms"):
                with self._profile("profile_apply_expert_shrink_ms"):
                    for name in param_names:
                        param = getattr(module, name)
                        old = param.data
                        new_data = torch.empty(
                            (slot_capacity,) + tuple(old.shape[1:]),
                            dtype=old.dtype,
                            device=old.device,
                        )
                        for slot_id, expert_id in enumerate(initial_resident):
                            new_data[slot_id].copy_(old[expert_id])
                        param.data = new_data

        state = _LayerKVExpertLayerState(
            layer_id=layer_id,
            module=module,
            orig_forward=module.forward,
            orig_run_moe_core=getattr(
                module, "_layerkv_hotness_orig_run_moe_core", module.run_moe_core
            ),
            full_num_experts=full_num_experts,
            slot_capacity=slot_capacity,
            expert_bytes=expert_bytes,
            device=device,
            dtype=dtype,
            cpu_params=cpu_params,
            param_names=param_names,
            logical_to_slot=logical_to_slot,
            slot_to_logical=slot_to_logical,
            lru=lru,
            hotness_prefill=self._expert_hotness_prefill.setdefault(layer_id, {}),
            hotness_decode=self._expert_hotness_decode.setdefault(layer_id, {}),
        )
        for expert_id, slot_id in logical_to_slot.items():
            heapq.heappush(state.lru_heap, (state.lru[expert_id], slot_id, expert_id))
        remap_tensor = torch.full(
            (full_num_experts,),
            -1,
            dtype=torch.long,
            device=device,
        )
        if initial_resident:
            ids = torch.tensor(initial_resident, dtype=torch.long, device=device)
            slots = torch.arange(len(initial_resident), dtype=torch.long, device=device)
            remap_tensor[ids] = slots
        state.remap_tensor = remap_tensor

        @functools.wraps(module.run_moe_core)
        def wrapped_run_moe_core(dispatch_output: Any, *args, **kwargs):
            topk_output = getattr(dispatch_output, "topk_output", None)
            if self._expert_chunked_core_required(state, topk_output):
                return self._run_expert_core_chunked(
                    state, dispatch_output, *args, **kwargs
                )
            rewritten_dispatch = self._prepare_expert_dispatch_for_core(
                state, dispatch_output
            )
            return state.orig_run_moe_core(rewritten_dispatch, *args, **kwargs)

        module.run_moe_core = wrapped_run_moe_core
        module._layerkv_expert_wrapped = True
        module._layerkv_expert_state = state

        try:
            module.num_experts = slot_capacity
            module.num_local_experts = slot_capacity
            module.moe_runner_config.num_experts = slot_capacity
            module.moe_runner_config.num_local_experts = slot_capacity
            module.dispatcher.num_experts = slot_capacity
            module.dispatcher.num_local_experts = slot_capacity
            module.dispatcher.num_local_routed_experts = slot_capacity
        except Exception:
            pass
        self._sync_expert_groups_for_state(state)
        return state

    def _select_initial_resident_experts(
        self, *, layer_id: int, full_num_experts: int, slot_capacity: int
    ) -> List[int]:
        if slot_capacity <= 0:
            return []
        candidate_order = self._expert_candidate_order_by_layer.get(int(layer_id))
        if candidate_order:
            self.stats.expert_candidate_order_hit_count += 1
            selected = [
                int(expert_id)
                for expert_id in candidate_order
                if 0 <= int(expert_id) < int(full_num_experts)
            ]
            if len(selected) < int(full_num_experts):
                selected_set = set(selected)
                selected.extend(
                    int(expert_id)
                    for expert_id in range(int(full_num_experts))
                    if int(expert_id) not in selected_set
                )
            return selected[: min(slot_capacity, full_num_experts)]
        self.stats.expert_candidate_order_miss_count += 1
        decode_hotness = self._expert_hotness_decode.get(layer_id, {})
        prefill_hotness = self._expert_hotness_prefill.get(layer_id, {})
        experts = list(range(full_num_experts))
        experts.sort(
            key=lambda expert_id: (
                -int(decode_hotness.get(expert_id, 0)),
                -int(prefill_hotness.get(expert_id, 0)),
                expert_id,
            )
        )
        return experts[: min(slot_capacity, full_num_experts)]

    def _copy_expert_to_cpu(
        self, module: Any, param_names: List[str], expert_id: int
    ) -> Dict[str, torch.Tensor]:
        with self._profile("profile_copy_expert_to_cpu_ms"):
            backing = {
                name: self._cpu_backing_tensor(
                    getattr(module, name).data[expert_id].detach()
                )
                for name in param_names
            }
        self._expert_host_backing_bytes += sum(int(t.nbytes) for t in backing.values())
        self._refresh_expert_host_backing_stat()
        return backing

    def _copy_expert_to_cpu_for_install(
        self, module: Any, param_names: List[str], expert_id: int
    ) -> Dict[str, torch.Tensor]:
        backing = self._copy_expert_to_cpu_backing_no_account(
            module, param_names, expert_id, non_blocking=True
        )
        self._expert_host_backing_bytes += sum(int(t.nbytes) for t in backing.values())
        self._refresh_expert_host_backing_stat()
        return backing

    def _record_expert_copy_descriptor(
        self,
        *,
        direction: str,
        reason: str,
        layer_id: int,
        logical_id: int,
        src_slot: int,
        dst_slot: int,
        nbytes: int,
        param_count: int,
    ) -> _LayerKVExpertCopyDescriptor:
        self._expert_copy_descriptor_seq += 1
        desc = _LayerKVExpertCopyDescriptor(
            seq=int(self._expert_copy_descriptor_seq),
            direction=str(direction),
            reason=str(reason),
            layer_id=int(layer_id),
            logical_id=int(logical_id),
            src_slot=int(src_slot),
            dst_slot=int(dst_slot),
            bytes=int(nbytes),
            param_count=int(param_count),
            step=int(self._decode_step),
        )
        self._expert_copy_descriptors_recent.append(desc)
        if len(self._expert_copy_descriptors_recent) > 256:
            self._expert_copy_descriptors_recent = (
                self._expert_copy_descriptors_recent[-256:]
            )
        self.stats.expert_copy_descriptor_count += 1
        self.stats.expert_copy_descriptor_bytes += int(nbytes)
        self.stats.expert_copy_descriptor_param_count += int(param_count)
        if str(direction) == "D2H":
            self.stats.expert_copy_descriptor_d2h_count += 1
        elif str(direction) == "H2D":
            self.stats.expert_copy_descriptor_h2d_count += 1
        if str(reason).startswith("install"):
            self.stats.expert_copy_descriptor_install_count += 1
        elif str(reason).startswith("evict"):
            self.stats.expert_copy_descriptor_evict_count += 1
        elif str(reason).startswith("materialize"):
            self.stats.expert_copy_descriptor_materialize_count += 1
        return desc

    @staticmethod
    def _contiguous_int_runs(values: List[int]) -> List[List[int]]:
        if not values:
            return []
        ordered = sorted(dict.fromkeys(int(x) for x in values))
        runs: List[List[int]] = []
        current: List[int] = [ordered[0]]
        for value in ordered[1:]:
            if int(value) == int(current[-1]) + 1:
                current.append(int(value))
            else:
                runs.append(current)
                current = [int(value)]
        runs.append(current)
        return runs

    def _expert_cpu_backing_pool_key(
        self, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> Tuple[torch.dtype, Tuple[int, ...]]:
        return (dtype, tuple(int(x) for x in shape))

    def _alloc_cpu_backing_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        *,
        pin_memory: bool = True,
    ) -> torch.Tensor:
        shape = tuple(int(x) for x in shape)
        if pin_memory:
            key = self._expert_cpu_backing_pool_key(shape, dtype)
            pool = self._expert_cpu_backing_pool.get(key)
            if pool:
                tensor = pool.pop()
                self._expert_cpu_backing_pool_bytes -= int(tensor.nbytes)
                self.stats.expert_cpu_backing_pool_reuse_count += 1
                self.stats.expert_cpu_backing_pool_bytes = int(
                    self._expert_cpu_backing_pool_bytes
                )
                return tensor
        self.stats.expert_cpu_backing_pool_alloc_count += 1
        return torch.empty(
            shape,
            dtype=dtype,
            device="cpu",
            pin_memory=pin_memory,
        )

    @staticmethod
    def _tensor_owns_storage(tensor: torch.Tensor) -> bool:
        try:
            if getattr(tensor, "_base", None) is not None:
                return False
            if int(tensor.storage_offset()) != 0:
                return False
            return int(tensor.untyped_storage().nbytes()) == int(tensor.nbytes)
        except Exception:
            return False

    def _release_cpu_backing_tensor(self, tensor: torch.Tensor) -> None:
        try:
            if tensor.device.type != "cpu" or not self._tensor_owns_storage(tensor):
                self.stats.expert_cpu_backing_pool_drop_count += 1
                return
            if hasattr(tensor, "is_pinned") and not bool(tensor.is_pinned()):
                self.stats.expert_cpu_backing_pool_drop_count += 1
                return
            nbytes = int(tensor.nbytes)
            if self._expert_cpu_backing_pool_bytes + nbytes > int(
                self._expert_cpu_backing_pool_limit_bytes
            ):
                self.stats.expert_cpu_backing_pool_drop_count += 1
                return
            key = self._expert_cpu_backing_pool_key(tuple(tensor.shape), tensor.dtype)
            self._expert_cpu_backing_pool.setdefault(key, []).append(tensor.detach())
            self._expert_cpu_backing_pool_bytes += nbytes
            self.stats.expert_cpu_backing_pool_release_count += 1
            self.stats.expert_cpu_backing_pool_bytes = int(
                self._expert_cpu_backing_pool_bytes
            )
            self.stats.expert_cpu_backing_pool_limit_bytes = int(
                self._expert_cpu_backing_pool_limit_bytes
            )
        except Exception:
            self.stats.expert_cpu_backing_pool_drop_count += 1

    def _release_expert_backing_params(
        self, params: Optional[Dict[str, torch.Tensor]]
    ) -> None:
        if not params:
            return
        for tensor in params.values():
            self._release_cpu_backing_tensor(tensor)

    def _copy_tensor_to_cpu_backing_pooled(
        self, tensor: torch.Tensor, *, non_blocking: bool, pin_memory: bool = True
    ) -> torch.Tensor:
        if not pin_memory:
            return tensor.to("cpu", copy=True)
        try:
            dst = self._alloc_cpu_backing_tensor(
                tuple(tensor.shape),
                tensor.dtype,
                pin_memory=True,
            )
            dst.copy_(tensor, non_blocking=non_blocking)
            return dst
        except Exception:
            return tensor.to("cpu", copy=True)

    def _copy_expert_to_cpu_backing_no_account(
        self,
        module: Any,
        param_names: List[str],
        expert_id: int,
        *,
        non_blocking: bool,
        pin_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        return {
            name: self._copy_tensor_to_cpu_backing_pooled(
                getattr(module, name).data[expert_id].detach(),
                non_blocking=non_blocking,
                pin_memory=pin_memory,
            )
            for name in param_names
        }

    def _global_expert_backing(
        self, layer_id: int, expert_id: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        params = self._expert_global_cpu_backing.get((int(layer_id), int(expert_id)))
        if params is None:
            self.stats.expert_cpu_backing_global_miss_count += 1
        else:
            self.stats.expert_cpu_backing_global_hit_count += 1
        return params

    def _is_global_expert_backing(
        self, layer_id: int, expert_id: int, params: Dict[str, torch.Tensor]
    ) -> bool:
        return (
            self._expert_global_cpu_backing.get((int(layer_id), int(expert_id)))
            is params
        )

    def _copy_experts_to_cpu_for_install_batched(
        self,
        module: Any,
        param_names: List[str],
        expert_ids: List[int],
        *,
        layer_id: int = -1,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        if not expert_ids:
            return {}
        copied: Dict[int, Dict[str, torch.Tensor]] = {
            int(expert_id): {} for expert_id in expert_ids
        }
        copied_bytes = 0
        device = getattr(module, param_names[0]).data.device if param_names else None
        with self._profile("profile_copy_expert_to_cpu_ms"):
            with torch.no_grad():
                use_tensor_batch = len(expert_ids) > 1
                if use_tensor_batch:
                    for name in param_names:
                        param = getattr(module, name).data
                        max_chunk_bytes = 256 * 1024 * 1024
                        per_expert_bytes = max(1, int(param[0].nbytes))
                        chunk_size = max(1, max_chunk_bytes // per_expert_bytes)
                        runs = self._contiguous_int_runs(expert_ids)
                        singleton_ids: List[int] = []
                        for run in runs:
                            if len(run) <= 1:
                                singleton_ids.extend(run)
                                continue
                            for begin in range(0, len(run), chunk_size):
                                chunk = run[begin : begin + chunk_size]
                                selected = param[
                                    int(chunk[0]) : int(chunk[0]) + len(chunk)
                                ].detach()
                                try:
                                    dst = self._alloc_cpu_backing_tensor(
                                        tuple(selected.shape),
                                        selected.dtype,
                                        pin_memory=True,
                                    )
                                    dst.copy_(selected, non_blocking=True)
                                except Exception:
                                    dst = selected.to("cpu", copy=True)
                                copied_bytes += int(dst.nbytes)
                                self.stats.expert_d2h_slice_run_count += 1
                                self.stats.expert_d2h_slice_expert_count += len(chunk)
                                for offset, expert_id in enumerate(chunk):
                                    copied[int(expert_id)][name] = dst[offset]
                        for begin in range(0, len(singleton_ids), chunk_size):
                            chunk = singleton_ids[begin : begin + chunk_size]
                            if not chunk:
                                continue
                            idx = torch.tensor(
                                chunk,
                                dtype=torch.long,
                                device=param.device,
                            )
                            selected = param.index_select(0, idx).detach()
                            try:
                                dst = self._alloc_cpu_backing_tensor(
                                    tuple(selected.shape),
                                    selected.dtype,
                                    pin_memory=True,
                                )
                                dst.copy_(selected, non_blocking=True)
                            except Exception:
                                dst = selected.to("cpu", copy=True)
                            copied_bytes += int(dst.nbytes)
                            self.stats.expert_d2h_gather_batch_count += 1
                            self.stats.expert_d2h_gather_expert_count += len(chunk)
                            for offset, expert_id in enumerate(chunk):
                                copied[int(expert_id)][name] = dst[offset]
                else:
                    for name in param_names:
                        param = getattr(module, name).data
                        for expert_id in expert_ids:
                            tensor = param[int(expert_id)].detach()
                            copied[int(expert_id)][name] = (
                                self._copy_tensor_to_cpu_backing_pooled(
                                    tensor,
                                    non_blocking=True,
                                )
                            )
                if device is not None and device.type == "cuda":
                    torch.cuda.current_stream(device=device).synchronize()
        self._expert_host_backing_bytes += sum(
            self._expert_backing_bytes(params) for params in copied.values()
        )
        for expert_id, params in copied.items():
            self._record_expert_copy_descriptor(
                direction="D2H",
                reason="install_backing",
                layer_id=int(layer_id),
                logical_id=int(expert_id),
                src_slot=int(expert_id),
                dst_slot=int(expert_id),
                nbytes=self._expert_backing_bytes(params),
                param_count=len(params),
            )
        self._refresh_expert_host_backing_stat()
        return copied

    def _copy_experts_to_cpu_for_install_batched_async(
        self,
        module: Any,
        param_names: List[str],
        expert_ids: List[int],
        *,
        layer_id: int = -1,
        target_cpu_params: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ) -> int:
        if not expert_ids or not param_names:
            return 0
        device = getattr(module, param_names[0]).data.device
        if device.type != "cuda":
            return -1
        active_stream = self._copy_stream or torch.cuda.current_stream(device=device)
        expert_ids = [int(expert_id) for expert_id in expert_ids]
        copied: Dict[int, Dict[str, torch.Tensor]] = {
            int(expert_id): {} for expert_id in expert_ids
        }
        copied_bytes = 0
        source_refs: List[torch.Tensor] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        try:
            with self._profile("profile_copy_expert_to_cpu_ms"):
                with torch.no_grad(), torch.cuda.stream(active_stream):
                    start.record(active_stream)
                    for name in param_names:
                        param = getattr(module, name).data
                        max_chunk_bytes = 256 * 1024 * 1024
                        per_expert_bytes = max(1, int(param[0].nbytes))
                        chunk_size = max(1, max_chunk_bytes // per_expert_bytes)
                        runs = self._contiguous_int_runs(expert_ids)
                        singleton_ids: List[int] = []
                        for run in runs:
                            if len(run) <= 1:
                                singleton_ids.extend(run)
                                continue
                            for begin in range(0, len(run), chunk_size):
                                chunk = run[begin : begin + chunk_size]
                                selected = param[
                                    int(chunk[0]) : int(chunk[0]) + len(chunk)
                                ].detach()
                                dst = self._alloc_cpu_backing_tensor(
                                    tuple(selected.shape),
                                    selected.dtype,
                                    pin_memory=True,
                                )
                                dst.copy_(selected, non_blocking=True)
                                copied_bytes += int(dst.nbytes)
                                source_refs.append(selected)
                                self.stats.expert_d2h_slice_run_count += 1
                                self.stats.expert_d2h_slice_expert_count += len(chunk)
                                for offset, expert_id in enumerate(chunk):
                                    copied[int(expert_id)][name] = dst[offset]
                        for begin in range(0, len(singleton_ids), chunk_size):
                            chunk = singleton_ids[begin : begin + chunk_size]
                            if not chunk:
                                continue
                            idx = torch.tensor(chunk, dtype=torch.long, device=param.device)
                            selected = param.index_select(0, idx).detach()
                            dst = self._alloc_cpu_backing_tensor(
                                tuple(selected.shape),
                                selected.dtype,
                                pin_memory=True,
                            )
                            dst.copy_(selected, non_blocking=True)
                            copied_bytes += int(dst.nbytes)
                            self.stats.expert_d2h_gather_batch_count += 1
                            self.stats.expert_d2h_gather_expert_count += len(chunk)
                            source_refs.extend([idx, selected])
                            for offset, expert_id in enumerate(chunk):
                                copied[int(expert_id)][name] = dst[offset]
                    end.record(active_stream)
        except Exception:
            try:
                active_stream.synchronize()
            except Exception:
                pass
            self.stats.expert_install_d2h_sync_fallback_count += len(expert_ids)
            return -1
        pending = _LayerKVPendingExpertD2H(
            start_event=start,
            ready_event=end,
            layer_id=int(layer_id),
            copied=copied,
            bytes=int(copied_bytes),
            reason="install_backing_async",
            source_refs=tuple(source_refs),
            target_cpu_params=target_cpu_params,
        )
        self._pending_expert_d2h_events.append(pending)
        for expert_id in copied:
            self._pending_expert_d2h_by_key[(int(layer_id), int(expert_id))] = pending
        for expert_id, params in copied.items():
            self._record_expert_copy_descriptor(
                direction="D2H",
                reason="install_backing_async",
                layer_id=int(layer_id),
                logical_id=int(expert_id),
                src_slot=int(expert_id),
                dst_slot=int(expert_id),
                nbytes=self._expert_backing_bytes(params),
                param_count=len(params),
            )
        self.stats.expert_install_d2h_async_count += len(expert_ids)
        self.stats.expert_install_d2h_async_mb += copied_bytes / float(1024 * 1024)
        self.stats.expert_copy_stream_launch_count += 1
        self.stats.layerkv_copy_event_record_count += 1
        return int(copied_bytes)

    def _copy_slot_to_cpu(
        self, state: _LayerKVExpertLayerState, slot_id: int
    ) -> Dict[str, torch.Tensor]:
        with self._profile("profile_copy_expert_to_cpu_ms"):
            backing = {
                name: self._cpu_backing_tensor(
                    getattr(state.module, name).data[slot_id].detach()
                )
                for name in state.param_names
            }
        self._expert_host_backing_bytes += sum(int(t.nbytes) for t in backing.values())
        self._refresh_expert_host_backing_stat()
        return backing

    def _copy_slots_to_cpu_batched(
        self,
        state: _LayerKVExpertLayerState,
        evicted: List[Tuple[int, int]],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        if not evicted:
            return {}
        copied: Dict[int, Dict[str, torch.Tensor]] = {
            int(logical_id): {} for logical_id, _slot_id in evicted
        }
        evicted_pairs = [
            (int(logical_id), int(slot_id)) for logical_id, slot_id in evicted
        ]
        use_tensor_batch = self._optimized_profile_enabled() and len(evicted_pairs) > 1
        copied_bytes = 0
        with self._profile("profile_copy_expert_to_cpu_ms"):
            with torch.no_grad():
                if use_tensor_batch:
                    self.stats.expert_eviction_d2h_batch_count += 1
                    slot_ids = [slot_id for _logical_id, slot_id in evicted_pairs]
                    try:
                        for name in state.param_names:
                            param = getattr(state.module, name).data
                            max_chunk_bytes = 256 * 1024 * 1024
                            per_expert_bytes = max(1, int(param[0].nbytes))
                            chunk_size = max(1, max_chunk_bytes // per_expert_bytes)
                            for begin in range(0, len(evicted_pairs), chunk_size):
                                chunk_pairs = evicted_pairs[begin : begin + chunk_size]
                                chunk_slots = slot_ids[begin : begin + chunk_size]
                                idx = torch.tensor(
                                    chunk_slots,
                                    dtype=torch.long,
                                    device=param.device,
                                )
                                selected = param.index_select(0, idx).detach()
                                try:
                                    dst = self._alloc_cpu_backing_tensor(
                                        tuple(selected.shape),
                                        selected.dtype,
                                        pin_memory=True,
                                    )
                                    dst.copy_(selected, non_blocking=True)
                                except Exception:
                                    dst = selected.to("cpu", copy=True)
                                copied_bytes += int(dst.nbytes)
                                for offset, (logical_id, _slot_id) in enumerate(
                                    chunk_pairs
                                ):
                                    copied[int(logical_id)][name] = dst[offset]
                        self.stats.expert_eviction_d2h_batched_count += len(
                            evicted_pairs
                        )
                        self.stats.expert_eviction_d2h_batched_mb += (
                            copied_bytes / float(1024 * 1024)
                        )
                    except Exception:
                        self.stats.expert_eviction_d2h_fallback_count += len(
                            evicted_pairs
                        )
                        copied = {
                            int(logical_id): {}
                            for logical_id, _slot_id in evicted_pairs
                        }
                        copied_bytes = 0
                        for name in state.param_names:
                            param = getattr(state.module, name).data
                            for logical_id, slot_id in evicted_pairs:
                                tensor = param[int(slot_id)].detach()
                                backing = self._copy_tensor_to_cpu_backing_pooled(
                                    tensor,
                                    non_blocking=True,
                                )
                                copied[int(logical_id)][name] = backing
                                copied_bytes += int(backing.nbytes)
                else:
                    if evicted_pairs:
                        self.stats.expert_eviction_d2h_fallback_count += len(
                            evicted_pairs
                        )
                    for name in state.param_names:
                        param = getattr(state.module, name).data
                        for logical_id, slot_id in evicted_pairs:
                            tensor = param[int(slot_id)].detach()
                            backing = self._copy_tensor_to_cpu_backing_pooled(
                                tensor,
                                non_blocking=True,
                            )
                            copied[int(logical_id)][name] = backing
                            copied_bytes += int(backing.nbytes)
                if state.device.type == "cuda":
                    torch.cuda.current_stream(device=state.device).synchronize()
        added = copied_bytes or sum(
            self._expert_backing_bytes(params) for params in copied.values()
        )
        for logical_id, slot_id in evicted_pairs:
            params = copied.get(int(logical_id), {})
            self._record_expert_copy_descriptor(
                direction="D2H",
                reason="evict_backing_sync",
                layer_id=int(state.layer_id),
                logical_id=int(logical_id),
                src_slot=int(slot_id),
                dst_slot=int(logical_id),
                nbytes=self._expert_backing_bytes(params),
                param_count=len(params),
            )
        self._expert_host_backing_bytes += added
        self._refresh_expert_host_backing_stat()
        return copied

    def _copy_slots_to_cpu_batched_async(
        self,
        state: _LayerKVExpertLayerState,
        evicted: List[Tuple[int, int]],
    ) -> bool:
        if not evicted or state.device.type != "cuda":
            return False
        active_stream = self._copy_stream or torch.cuda.current_stream(
            device=state.device
        )
        evicted_pairs = [
            (int(logical_id), int(slot_id)) for logical_id, slot_id in evicted
        ]
        copied: Dict[int, Dict[str, torch.Tensor]] = {
            int(logical_id): {} for logical_id, _slot_id in evicted_pairs
        }
        copied_bytes = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        try:
            with self._profile("profile_copy_expert_to_cpu_ms"):
                with torch.no_grad(), torch.cuda.stream(active_stream):
                    start.record(active_stream)
                    for name in state.param_names:
                        param = getattr(state.module, name).data
                        dst = self._alloc_cpu_backing_tensor(
                            (len(evicted_pairs),) + tuple(param.shape[1:]),
                            param.dtype,
                            pin_memory=True,
                        )
                        copied_bytes += int(dst.nbytes)
                        for offset, (logical_id, slot_id) in enumerate(evicted_pairs):
                            dst[offset].copy_(
                                param[int(slot_id)].detach(),
                                non_blocking=True,
                            )
                            copied[int(logical_id)][name] = dst[offset]
                    end.record(active_stream)
        except Exception:
            return False
        pending = _LayerKVPendingExpertD2H(
            start_event=start,
            ready_event=end,
            layer_id=int(state.layer_id),
            copied=copied,
            bytes=int(copied_bytes),
        )
        self._pending_expert_d2h_events.append(pending)
        for logical_id in copied:
            self._pending_expert_d2h_by_key[(int(state.layer_id), int(logical_id))] = (
                pending
            )
        for logical_id, slot_id in evicted_pairs:
            params = copied.get(int(logical_id), {})
            self._record_expert_copy_descriptor(
                direction="D2H",
                reason="evict_backing_async",
                layer_id=int(state.layer_id),
                logical_id=int(logical_id),
                src_slot=int(slot_id),
                dst_slot=int(logical_id),
                nbytes=self._expert_backing_bytes(params),
                param_count=len(params),
            )
        self.stats.expert_eviction_d2h_async_count += len(evicted_pairs)
        self.stats.expert_eviction_d2h_async_mb += copied_bytes / float(1024 * 1024)
        self.stats.expert_eviction_d2h_batch_count += 1
        self.stats.expert_eviction_d2h_batched_count += len(evicted_pairs)
        self.stats.expert_eviction_d2h_batched_mb += copied_bytes / float(1024 * 1024)
        return True

    def _finalize_expert_d2h_events(self, *, block: bool = False) -> None:
        if not self._pending_expert_d2h_events:
            return
        remaining: List[_LayerKVPendingExpertD2H] = []
        finalized_install = 0
        for pending in self._pending_expert_d2h_events:
            try:
                if block:
                    pending.ready_event.synchronize()
                elif not pending.ready_event.query():
                    remaining.append(pending)
                    continue
                try:
                    elapsed = float(
                        pending.start_event.elapsed_time(pending.ready_event)
                    )
                    self.stats.layerkv_copy_stream_busy_ms += elapsed
                except Exception:
                    pass
                state = self._expert_layers.get(int(pending.layer_id))
                target_cpu_params = pending.target_cpu_params
                if state is None and target_cpu_params is None:
                    remaining.append(pending)
                    continue
                for logical_id, params in pending.copied.items():
                    logical_id = int(logical_id)
                    if target_cpu_params is not None:
                        target_cpu_params[logical_id] = params
                    elif state is not None:
                        state.cpu_params[logical_id] = params
                        state.backing_lru[logical_id] = self._decode_step
                    key = (int(pending.layer_id), logical_id)
                    if self._pending_expert_d2h_by_key.get(key) is pending:
                        self._pending_expert_d2h_by_key.pop(key, None)
                self._expert_host_backing_bytes += int(pending.bytes)
                if str(pending.reason).startswith("install"):
                    self.stats.expert_install_d2h_async_finalize_count += len(
                        pending.copied
                    )
                    finalized_install += len(pending.copied)
                else:
                    self.stats.expert_eviction_d2h_async_finalize_count += len(
                        pending.copied
                    )
            except Exception:
                remaining.append(pending)
        self._pending_expert_d2h_events = remaining
        self._refresh_expert_host_backing_stat()
        if finalized_install:
            self._refresh_expert_install_progress()

    def _wait_for_pending_expert_backing(
        self, state: _LayerKVExpertLayerState, logical_id: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        key = (int(state.layer_id), int(logical_id))
        pending = self._pending_expert_d2h_by_key.get(key)
        if pending is None:
            return None
        if str(pending.reason).startswith("install"):
            self.stats.expert_install_d2h_async_wait_count += 1
        else:
            self.stats.expert_eviction_d2h_async_wait_count += 1
        pending.ready_event.synchronize()
        self._finalize_expert_d2h_events(block=False)
        return state.cpu_params.get(int(logical_id))

    @staticmethod
    def _expert_backing_bytes(params: Dict[str, torch.Tensor]) -> int:
        return sum(int(t.nbytes) for t in params.values())

    def _drop_expert_backing(
        self, state: _LayerKVExpertLayerState, logical_id: int
    ) -> None:
        removed = state.cpu_params.pop(int(logical_id), None)
        state.backing_lru.pop(int(logical_id), None)
        if removed is not None:
            if not self._is_global_expert_backing(state.layer_id, logical_id, removed):
                self._expert_host_backing_bytes -= self._expert_backing_bytes(removed)
                self._release_expert_backing_params(removed)
            self.stats.expert_backing_cache_evict_count += 1
            self._refresh_expert_host_backing_stat()

    def _trim_expert_backing_cache(self, state: _LayerKVExpertLayerState) -> None:
        """Bound optional CPU copies for resident experts.

        CPU backing for offloaded experts is mandatory because the GPU slot no
        longer contains that logical expert.  Only resident experts are cache
        entries and may be dropped.
        """

        extra_cache_bytes = int(self._effective_expert_backing_cache_mb() * 1024 * 1024)
        if extra_cache_bytes <= 0:
            cached_resident = [
                logical_id
                for logical_id in list(state.cpu_params)
                if logical_id in state.logical_to_slot
            ]
            removed_bytes = 0
            removed_count = 0
            for logical_id in cached_resident:
                removed = state.cpu_params.pop(int(logical_id), None)
                state.backing_lru.pop(int(logical_id), None)
                if removed is not None:
                    if not self._is_global_expert_backing(
                        state.layer_id, logical_id, removed
                    ):
                        removed_bytes += self._expert_backing_bytes(removed)
                        self._release_expert_backing_params(removed)
                    removed_count += 1
            if removed_count:
                self._expert_host_backing_bytes -= removed_bytes
                self.stats.expert_backing_cache_evict_count += removed_count
                self._refresh_expert_host_backing_stat()
            return
        mandatory_bytes = 0
        for layer_state in self._expert_layers.values():
            for logical_id, params in layer_state.cpu_params.items():
                if int(logical_id) not in layer_state.logical_to_slot:
                    mandatory_bytes += self._expert_backing_bytes(params)
        limit_bytes = mandatory_bytes + extra_cache_bytes
        while self._expert_host_backing_bytes > limit_bytes:
            candidates = [
                (
                    int(state.backing_lru.get(logical_id, 0)),
                    int(logical_id),
                )
                for logical_id in state.cpu_params
                if logical_id in state.logical_to_slot
            ]
            if not candidates:
                break
            _step, logical_id = min(candidates)
            self._drop_expert_backing(state, logical_id)

    @staticmethod
    def _cpu_backing_tensor(tensor: torch.Tensor) -> torch.Tensor:
        cpu = tensor.to("cpu", copy=True)
        try:
            return cpu.pin_memory()
        except Exception:
            return cpu

    @staticmethod
    def _copy_tensor_to_cpu_backing(
        tensor: torch.Tensor, *, non_blocking: bool, pin_memory: bool = True
    ) -> torch.Tensor:
        return LayerKVRuntime._copy_tensor_to_cpu_backing_impl(
            tensor,
            non_blocking=non_blocking,
            pin_memory=pin_memory,
        )

    @staticmethod
    def _copy_tensor_to_cpu_backing_impl(
        tensor: torch.Tensor, *, non_blocking: bool, pin_memory: bool
    ) -> torch.Tensor:
        if not pin_memory:
            return tensor.to("cpu", copy=True)
        try:
            dst = torch.empty(
                tuple(tensor.shape),
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=True,
            )
            dst.copy_(tensor, non_blocking=non_blocking)
            return dst
        except Exception:
            return tensor.to("cpu", copy=True)

    def _maybe_prepare_expert_plan_during_extend(self, forward_batch: Any) -> None:
        if self._simple_profile_enabled():
            return
        # Avoid duplicating several GB of expert backing during prefill by
        # default.  The decode install path remains bounded and avoids the
        # CPU-memory spikes seen in large-batch runs.
        if (
            str(self.config.expert_cpu_backing_mode) == "none"
            and self._effective_expert_backing_cache_mb() <= 0.0
        ):
            return
        if (
            self._expert_plan_applied
            or self._expert_prepare_started
            or self.config.mode != "kvc-expert"
        ):
            return
        avg_prefix = self._avg_prefix_len(forward_batch)
        is_long_context = avg_prefix > 2048
        if is_long_context:
            # Backing-only prepare was tested on Fig4 context-heavy: it reduced
            # decode apply time but moved more copy work into prefill/extend and
            # regressed end-to-end latency. Keep long-context prepare disabled.
            return
        # Short-context/batch-heavy runs benefit from preparing the CPU backing
        # while prefill is still progressing.  Keep this bounded to avoid the
        # earlier host-memory blowups: long-context is disabled above, and the
        # target is capped by actual expert reclaim capacity below.
        if is_long_context:
            target_mb = min(
                max(0.0, self.config.reclaim_limit_mb),
                max(0.0, self._available_expert_reclaim_mb()),
            )
        else:
            target_mb = self._effective_expert_reclaim_mb(forward_batch)
        if (
            target_mb <= 0
            or not self.physical_expert_supported
            or not self._expert_modules
        ):
            return
        if not self._has_prefill_expert_hotness():
            return
        required_progress = 0.90
        if self._extend_progress_ratio(forward_batch) < required_progress:
            return

        self._expert_prepare_started = True
        profile_name = (
            "profile_prepare_expert_backing_only_ms"
            if is_long_context
            else "profile_prepare_expert_backing_ms"
        )
        with self._profile(profile_name):
            slot_capacities = self._coresid_planned_expert_capacities(
                target_mb, forward_batch
            )
            cpu_params_by_layer: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
            initial_resident_by_layer: Dict[int, List[int]] = {}
            for layer_id, module in self._expert_modules:
                param_names = self._expert_param_names(module)
                full_num_experts = int(module.w13_weight.data.shape[0])
                slot_capacity = slot_capacities[layer_id]
                initial_resident = self._select_initial_resident_experts(
                    layer_id=layer_id,
                    full_num_experts=full_num_experts,
                    slot_capacity=slot_capacity,
                )
                if not is_long_context:
                    initial_resident_by_layer[layer_id] = initial_resident
                resident_set = set(initial_resident)
                to_copy = [
                    expert_id
                    for expert_id in range(full_num_experts)
                    if expert_id not in resident_set
                ]
                layer_cpu_params = self._copy_experts_to_cpu_for_install_batched(
                    module,
                    param_names,
                    to_copy,
                    layer_id=layer_id,
                )
                cpu_params_by_layer[layer_id] = layer_cpu_params

        prepared_mb = sum(
            int(t.nbytes)
            for layer_params in cpu_params_by_layer.values()
            for expert_params in layer_params.values()
            for t in expert_params.values()
        ) / float(1024 * 1024)
        self.stats.expert_prepared_backing_mb = prepared_mb
        self.stats.expert_prepare_guard_mb = 0.0
        self.stats.expert_prepare_extend_count += 1
        self._prepared_expert_plan = {
            "kind": "backing_only" if is_long_context else "full",
            "target_mb": target_mb,
            "slot_capacities": slot_capacities,
            "initial_resident_by_layer": initial_resident_by_layer,
            "cpu_params_by_layer": cpu_params_by_layer,
        }
        self._expert_prepare_done = True

    def _maybe_prepare_expert_plan_after_decode(self, forward_batch: Any) -> None:
        if (
            not self._coresid_optimized_policy_enabled()
            or not self._prepared_expert_backing_enabled()
            or self._expert_plan_applied
            or self._expert_prepare_started
            or self._expert_prepare_done
            or self._expert_install_queue
            or not self.physical_expert_supported
            or not self._expert_modules
        ):
            return
        target_mb = self._effective_expert_reclaim_mb(forward_batch)
        if target_mb <= 0:
            return
        self._ensure_expert_hotness_cpu_view(mode="decode")
        if not self._has_decode_expert_hotness():
            return
        self._expert_prepare_started = True
        with self._profile("profile_prepare_expert_backing_ms"):
            slot_capacities = self._coresid_planned_expert_capacities(
                target_mb, forward_batch
            )
            cpu_params_by_layer: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
            initial_resident_by_layer: Dict[int, List[int]] = {}
            copied_count = 0
            reused_count = 0
            candidate_count = 0
            copied_bytes = 0
            for layer_id, module in self._expert_modules:
                param_names = self._expert_param_names(module)
                full_num_experts = int(module.w13_weight.data.shape[0])
                slot_capacity = int(slot_capacities[layer_id])
                initial_resident = self._select_initial_resident_experts(
                    layer_id=layer_id,
                    full_num_experts=full_num_experts,
                    slot_capacity=slot_capacity,
                )
                initial_resident_by_layer[int(layer_id)] = initial_resident
                resident_set = set(int(x) for x in initial_resident)
                to_copy = [
                    int(expert_id)
                    for expert_id in range(full_num_experts)
                    if int(expert_id) not in resident_set
                ]
                candidate_count += len(to_copy)
                layer_cpu_params: Dict[int, Dict[str, torch.Tensor]] = {}
                missing: List[int] = []
                for expert_id in to_copy:
                    global_params = (
                        self._global_expert_backing(layer_id, expert_id)
                        if self._expert_global_cpu_backing
                        else None
                    )
                    if global_params is not None:
                        layer_cpu_params[int(expert_id)] = global_params
                        reused_count += 1
                    else:
                        missing.append(int(expert_id))
                if missing:
                    copied = self._copy_experts_to_cpu_for_install_batched(
                        module,
                        param_names,
                        missing,
                        layer_id=layer_id,
                    )
                    copied_count += len(copied)
                    copied_bytes += sum(
                        self._expert_backing_bytes(params) for params in copied.values()
                    )
                    layer_cpu_params.update(copied)
                cpu_params_by_layer[int(layer_id)] = layer_cpu_params
        prepared_mb = sum(
            int(t.nbytes)
            for layer_params in cpu_params_by_layer.values()
            for expert_params in layer_params.values()
            for t in expert_params.values()
        ) / float(1024 * 1024)
        self.stats.expert_prepare_candidate_count += candidate_count
        self.stats.expert_prepare_copied_count += copied_count
        self.stats.expert_prepare_reused_count += reused_count
        self.stats.expert_prepare_host_budget_mb = float(target_mb)
        self.stats.expert_prepare_actual_mb = prepared_mb
        self.stats.expert_prepared_backing_mb = prepared_mb
        self.stats.expert_prepare_guard_mb = 0.0
        self.stats.expert_prepare_extend_count += 1
        self._prepared_expert_plan = {
            "kind": "full",
            "target_mb": target_mb,
            "slot_capacities": slot_capacities,
            "initial_resident_by_layer": initial_resident_by_layer,
            "cpu_params_by_layer": cpu_params_by_layer,
        }
        self._expert_prepare_done = True

    def _expert_chunked_core_required(
        self, state: _LayerKVExpertLayerState, topk_output: Any
    ) -> bool:
        if not self._dynamic_expert_churn_policy_enabled():
            return False
        if int(state.slot_capacity) >= int(state.full_num_experts):
            return False
        topk_ids = getattr(topk_output, "topk_ids", None)
        if topk_ids is None or int(state.slot_capacity) <= 0:
            return False
        top_k = int(getattr(state.module, "top_k", 0) or 0)
        if top_k <= 0:
            top_k = int(getattr(state.module.moe_runner_config, "top_k", 1) or 1)
        if int(state.slot_capacity) >= max(1, min(int(state.full_num_experts), top_k)):
            return False
        if (
            self._optimized_profile_enabled()
            and self._expert_plan_applied
            and state.remap_tensor is not None
        ):
            valid = (topk_ids >= 0) & (topk_ids < state.full_num_experts)
            if not bool(valid.any().item()):
                return False
            safe_ids = topk_ids.clamp(min=0, max=state.full_num_experts - 1).long()
            mapped_ids = state.remap_tensor[safe_ids]
            # If every routed expert is already resident, the number of routed
            # unique experts cannot exceed slot capacity; avoid torch.unique().
            if not bool((valid & (mapped_ids < 0)).any().item()):
                return False
        logical_ids = self._unique_expert_ids(topk_ids, state.full_num_experts)
        return len(logical_ids) > int(state.slot_capacity)

    def _run_expert_core_chunked(
        self,
        state: _LayerKVExpertLayerState,
        dispatch_output: Any,
        *args,
        **kwargs,
    ) -> Any:
        topk_output = getattr(dispatch_output, "topk_output", None)
        topk_ids = getattr(topk_output, "topk_ids", None)
        topk_weights = getattr(topk_output, "topk_weights", None)
        if (
            topk_ids is None
            or topk_weights is None
            or not hasattr(dispatch_output, "_replace")
        ):
            rewritten_dispatch = self._prepare_expert_dispatch_for_core(
                state, dispatch_output
            )
            return state.orig_run_moe_core(rewritten_dispatch, *args, **kwargs)

        logical_ids = self._unique_expert_ids_and_record_hotness(
            state, topk_ids, record_hotness=not self._expert_plan_applied
        )
        if not logical_ids:
            rewritten_dispatch = self._prepare_expert_dispatch_for_core(
                state, dispatch_output
            )
            return state.orig_run_moe_core(rewritten_dispatch, *args, **kwargs)

        capacity = max(1, int(state.slot_capacity))
        valid = (topk_ids >= 0) & (topk_ids < state.full_num_experts)
        safe_ids = topk_ids.clamp(min=0, max=state.full_num_experts - 1).long()
        output_accum = None
        first_output = None
        for begin in range(0, len(logical_ids), capacity):
            chunk = [int(x) for x in logical_ids[begin : begin + capacity]]
            self._materialize_experts(state, chunk, reason="on_demand")
            self._wait_for_expert_logical_ids_ready(state, chunk)
            in_chunk = torch.zeros_like(valid, dtype=torch.bool)
            for expert_id in chunk:
                in_chunk |= safe_ids.eq(int(expert_id))
            active = valid & in_chunk
            mapped_ids = state.remap_tensor[safe_ids]
            chunk_ids = torch.where(
                active,
                mapped_ids.to(topk_ids.dtype),
                torch.zeros_like(topk_ids),
            )
            chunk_weights = torch.where(
                active,
                topk_weights,
                torch.zeros_like(topk_weights),
            )
            chunk_topk = topk_output._replace(
                topk_ids=chunk_ids,
                topk_weights=chunk_weights,
            )
            chunk_dispatch = dispatch_output._replace(topk_output=chunk_topk)
            chunk_output = state.orig_run_moe_core(chunk_dispatch, *args, **kwargs)
            hidden = getattr(chunk_output, "hidden_states", None)
            if hidden is None:
                return chunk_output
            if output_accum is None:
                output_accum = hidden
                first_output = chunk_output
            else:
                output_accum = output_accum + hidden

        if self._current_forward_mode == "decode":
            state.last_decode_logical_ids = logical_ids
            self._expert_prefetch_dirty_layers.add(int(state.layer_id))
        self.stats.expert_topk_rewrite_count += 1
        self.stats.expert_core_hook_count += 1
        if first_output is not None and hasattr(first_output, "_replace"):
            return first_output._replace(hidden_states=output_accum)
        return first_output

    def _prepare_expert_dispatch_for_core(
        self, state: _LayerKVExpertLayerState, dispatch_output: Any
    ) -> Any:
        topk_output = getattr(dispatch_output, "topk_output", None)
        if topk_output is None:
            return dispatch_output
        rewritten_topk = self._prepare_expert_layer_for_topk(state, topk_output)
        self.stats.expert_core_hook_count += 1
        if rewritten_topk is topk_output:
            return dispatch_output
        if hasattr(dispatch_output, "_replace"):
            return dispatch_output._replace(topk_output=rewritten_topk)
        reason = f"layer {state.layer_id} dispatch output does not support topk rewrite"
        self.stats.expert_guard_pass = False
        self.stats.expert_guard_reason = reason
        raise RuntimeError(reason)

    def _prepare_expert_layer_for_topk(
        self, state: _LayerKVExpertLayerState, topk_output: Any
    ) -> Any:
        topk_ids = getattr(topk_output, "topk_ids", None)
        if topk_ids is None:
            return topk_output
        if int(state.slot_capacity) >= int(state.full_num_experts):
            if self._should_collect_expert_hotness_layer(int(state.layer_id)):
                self._record_expert_hotness_for_layer(
                    state.layer_id, state.full_num_experts, topk_ids
                )
            else:
                self._expert_hotness_sample_skip_pending += 1
            return topk_output
        with self._profile("profile_expert_topk_rewrite_ms"):
            return self._prepare_expert_layer_for_topk_remap(state, topk_output)

    def _prepare_expert_layer_for_topk_remap(
        self, state: _LayerKVExpertLayerState, topk_output: Any
    ) -> Any:
        topk_ids = getattr(topk_output, "topk_ids", None)
        if topk_ids is None:
            return topk_output
        valid: Optional[torch.Tensor] = None
        fast_in_range = bool(
            state.topk_ids_in_range_calibrated
            and not state.topk_ids_invalid_observed
            and topk_ids.numel() > 0
        )
        if fast_in_range:
            safe_ids = topk_ids.long()
            self.stats.expert_topk_range_fastpath_count += 1
        else:
            valid = (topk_ids >= 0) & (topk_ids < state.full_num_experts)
            safe_ids = topk_ids.clamp(min=0, max=state.full_num_experts - 1).long()
        remap = state.remap_tensor
        if remap is None:
            reason = f"layer {state.layer_id} missing expert remap tensor"
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = reason
            raise RuntimeError(reason)
        mapped_ids = remap[safe_ids]
        if (
            not fast_in_range
            and valid is not None
            and not state.topk_ids_in_range_calibrated
            and not state.topk_ids_invalid_observed
        ):
            invalid_any = bool((~valid).any().item())
            if invalid_any:
                state.topk_ids_invalid_observed = True
                self.stats.expert_topk_range_invalid_count += 1
            else:
                state.topk_ids_in_range_calibrated = True
                self.stats.expert_topk_range_calibrated_count += 1
        if self._optimized_profile_enabled() and self._expert_plan_applied:
            # Common pressure steady state: one or a few cold experts are
            # offloaded, but this token routes only to resident experts. Avoid
            # CPU unique/materialize work and only rewrite logical ids to slots.
            missing = mapped_ids < 0
            missing_any = bool(
                missing.any().item()
                if fast_in_range or valid is None
                else (valid & missing).any().item()
            )
            if not missing_any:
                if fast_in_range or valid is None:
                    rewritten_ids = mapped_ids.to(topk_ids.dtype)
                else:
                    rewritten_ids = torch.where(
                        valid, mapped_ids.to(topk_ids.dtype), topk_ids
                    )
                self.stats.expert_topk_rewrite_count += 1
                return topk_output._replace(topk_ids=rewritten_ids)
        logical_ids = self._unique_expert_ids_and_record_hotness(
            state, topk_ids, record_hotness=not self._expert_plan_applied
        )
        if len(logical_ids) > state.slot_capacity:
            self._grow_expert_layer_slots(state, len(logical_ids))
        self._materialize_experts(state, logical_ids, reason="on_demand")
        self._wait_for_expert_logical_ids_ready(state, logical_ids)
        mapped_ids = remap[safe_ids]
        missing = mapped_ids < 0
        if bool(
            missing.any().item()
            if fast_in_range or valid is None
            else (valid & missing).any().item()
        ):
            missing = self._unique_expert_ids(topk_ids, state.full_num_experts)
            reason = (
                f"layer {state.layer_id} expert remap still missing after "
                f"materialize: logical_ids={missing[:8]}"
            )
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = reason
            self.stats.comparable = False
            self.stats.comparability_reason = reason
            raise RuntimeError(reason)
        if self._current_forward_mode == "decode":
            state.last_decode_logical_ids = logical_ids
            self._expert_prefetch_dirty_layers.add(int(state.layer_id))
        if fast_in_range or valid is None:
            rewritten_ids = mapped_ids.to(topk_ids.dtype)
        else:
            rewritten_ids = torch.where(valid, mapped_ids.to(topk_ids.dtype), topk_ids)
        self.stats.expert_topk_rewrite_count += 1
        return topk_output._replace(topk_ids=rewritten_ids)

    def _record_expert_hotness(
        self, state: _LayerKVExpertLayerState, topk_ids: torch.Tensor
    ) -> None:
        self._record_expert_hotness_for_layer(
            layer_id=state.layer_id,
            full_num_experts=state.full_num_experts,
            topk_ids=topk_ids,
        )

    def _should_sample_expert_hotness(self, layer_id: int) -> bool:
        if self._current_forward_mode != "decode":
            return True
        step = max(0, int(self._decode_step))
        if step <= 4:
            return True
        if self._expert_hotness_sampled_layers_step == step:
            sampled = self._expert_hotness_sampled_layers
            return sampled is None or int(layer_id) in sampled
        interval = self._effective_expert_hotness_sample_interval()
        return ((step + int(layer_id)) % interval) == 0

    def _effective_expert_hotness_sample_interval(self) -> int:
        interval = max(1, int(self.config.expert_hotness_sample_interval))
        if (
            self.config.dynamic_pressure_from_kvc
            and self._current_forward_mode == "decode"
            and self._expert_plan_applied
            and not self._expert_install_queue
            and float(self.stats.effective_reclaim_target_mb) <= 1e-3
            and float(self.stats.needed_pressure_mb) <= 1e-3
            and int(self._decode_step) > 32
        ):
            return interval * 16
        if self._defer_expert_hotness_snapshot() and int(self._decode_step) > 32:
            return interval * 4
        return interval

    def _prepare_expert_hotness_sampling_for_step(self) -> None:
        if self._current_forward_mode != "decode":
            self._expert_hotness_sampled_layers = None
            self._expert_hotness_sampled_layers_step = int(self._decode_step)
            return
        step = max(0, int(self._decode_step))
        self._expert_hotness_sampled_layers_step = step
        if step <= 4:
            self._expert_hotness_sampled_layers = None
            return
        interval = self._effective_expert_hotness_sample_interval()
        self._expert_hotness_sampled_layers = {
            int(layer_id)
            for layer_id, _module in self._expert_modules
            if ((step + int(layer_id)) % interval) == 0
        }

    def _should_collect_expert_hotness_layer(self, layer_id: int) -> bool:
        if self._current_forward_mode != "decode":
            return True
        if self._decode_step <= 0:
            return True
        return self._should_sample_expert_hotness(layer_id)

    def _flush_expert_hotness_sample_skip_count(self) -> None:
        pending = int(self._expert_hotness_sample_skip_pending)
        if pending <= 0:
            return
        self.stats.expert_hotness_sample_skip_count += pending
        self._expert_hotness_sample_skip_pending = 0

    def _expert_hotness_gpu_counts(
        self, mode: str, layer_id: int, full_num_experts: int, device: torch.device
    ) -> torch.Tensor:
        store = (
            self._expert_hotness_decode
            if mode == "decode"
            else self._expert_hotness_prefill
        )
        store.setdefault(int(layer_id), {})
        gpu_store = (
            self._expert_hotness_gpu_decode
            if mode == "decode"
            else self._expert_hotness_gpu_prefill
        )
        buf = gpu_store.get(int(layer_id))
        if (
            buf is None
            or int(buf.numel()) != int(full_num_experts)
            or buf.device != device
        ):
            buf = torch.zeros(
                (int(full_num_experts),), dtype=torch.int32, device=device
            )
            gpu_store[int(layer_id)] = buf
        return buf

    def _expert_hotness_ones(
        self, device: torch.device, numel: int
    ) -> torch.Tensor:
        key = (str(device), int(numel))
        cached = self._expert_hotness_ones_cache.get(key)
        if cached is not None and cached.device == device and int(cached.numel()) == int(
            numel
        ):
            self.stats.expert_hotness_ones_reuse_count += 1
            return cached
        ones = torch.ones((int(numel),), dtype=torch.int32, device=device)
        self._expert_hotness_ones_cache[key] = ones
        return ones

    def _defer_expert_hotness_snapshot(self) -> bool:
        return (
            self.config.dynamic_pressure_from_kvc
            and self._current_forward_mode == "decode"
            and not self._expert_install_queue
            and float(self.stats.effective_reclaim_target_mb) <= 1e-3
            and float(self.stats.needed_pressure_mb) <= 1e-3
        )

    def _maybe_issue_expert_hotness_snapshot(
        self, mode: str, layer_id: int, counts: torch.Tensor
    ) -> None:
        if counts.device.type != "cuda":
            return
        if self._defer_expert_hotness_snapshot():
            self.stats.expert_hotness_snapshot_deferred_count += 1
            return
        key = (str(mode), int(layer_id))
        if key in self._expert_hotness_pending_snapshot_keys:
            return
        max_pending = max(8, len(self._expert_modules) * 2)
        if len(self._expert_hotness_pending_snapshots) >= max_pending:
            self.stats.expert_hotness_snapshot_drop_count += 1
            return
        step = max(0, int(self._decode_step))
        last = int(self._expert_hotness_last_snapshot_step.get(key, -1))
        interval = max(4, int(self.config.expert_hotness_sample_interval) * 4)
        if step > 4 and last >= 0 and step - last < interval:
            return
        snapshot_counts = counts.detach().clone()
        try:
            cpu_counts = torch.empty_like(counts, device="cpu", pin_memory=True)
        except Exception:
            cpu_counts = torch.empty_like(counts, device="cpu")
        device_key = str(counts.device)
        stream = self._expert_hotness_snapshot_streams.get(device_key)
        if stream is None:
            stream = torch.cuda.Stream(device=counts.device)
            self._expert_hotness_snapshot_streams[device_key] = stream
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device=counts.device))
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            cpu_counts.copy_(snapshot_counts, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        snapshot_counts.record_stream(stream)
        self._expert_hotness_pending_snapshots.append(
            (str(mode), int(layer_id), cpu_counts, event, snapshot_counts)
        )
        self._expert_hotness_pending_snapshot_keys.add(key)
        self._expert_hotness_last_snapshot_step[key] = step
        self.stats.expert_hotness_snapshot_issue_count += 1

    def _hotness_snapshot_stream(self, device: torch.device) -> Any:
        device_key = str(device)
        stream = self._expert_hotness_snapshot_streams.get(device_key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._expert_hotness_snapshot_streams[device_key] = stream
        return stream

    def _maybe_issue_expert_candidate_snapshot(
        self, mode: str, layer_id: int, full_num_experts: int, device: torch.device
    ) -> None:
        if device.type != "cuda":
            return
        if mode != "decode":
            return
        layer_id = int(layer_id)
        if layer_id in self._expert_candidate_pending_layers:
            return
        max_pending = max(8, len(self._expert_modules) * 2)
        if len(self._expert_candidate_pending_snapshots) >= max_pending:
            self.stats.expert_candidate_snapshot_drop_count += 1
            return
        step = max(0, int(self._decode_step))
        last = int(self._expert_candidate_last_snapshot_step.get(layer_id, -1))
        interval = max(4, int(self.config.expert_hotness_sample_interval) * 4)
        if step > 4 and last >= 0 and step - last < interval:
            return
        decode_counts = self._expert_hotness_gpu_decode.get(layer_id)
        prefill_counts = self._expert_hotness_gpu_prefill.get(layer_id)
        if decode_counts is None and prefill_counts is None:
            return
        if decode_counts is None:
            decode_counts = torch.zeros(
                (int(full_num_experts),), dtype=torch.int32, device=device
            )
        if prefill_counts is None:
            prefill_counts = torch.zeros(
                (int(full_num_experts),), dtype=torch.int32, device=device
            )
        expert_ids = torch.arange(int(full_num_experts), dtype=torch.long, device=device)
        # Match CPU ordering: decode hotness desc, then prefill desc, then id asc.
        scale = 1_000_000_000
        scores = (
            (
                decode_counts.to(torch.long) * scale
                + prefill_counts.to(torch.long)
            )
            * (int(full_num_experts) + 1)
            + (int(full_num_experts) - expert_ids)
        )
        order = torch.argsort(scores, descending=True).to(torch.int32)
        try:
            cpu_order = torch.empty_like(order, device="cpu", pin_memory=True)
        except Exception:
            cpu_order = torch.empty_like(order, device="cpu")
        stream = self._hotness_snapshot_stream(device)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device=device))
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            cpu_order.copy_(order, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        order.record_stream(stream)
        self._expert_candidate_pending_snapshots.append(
            (layer_id, cpu_order, event, order)
        )
        self._expert_candidate_pending_layers.add(layer_id)
        self._expert_candidate_last_snapshot_step[layer_id] = step
        self.stats.expert_candidate_snapshot_issue_count += 1

    def _finalize_expert_candidate_snapshots(self, *, block: bool = False) -> None:
        if not self._expert_candidate_pending_snapshots:
            return
        remaining = []
        for layer_id, cpu_order, event, order in self._expert_candidate_pending_snapshots:
            keep_pending = False
            try:
                if block:
                    event.synchronize()
                elif not event.query():
                    remaining.append((layer_id, cpu_order, event, order))
                    keep_pending = True
                    continue
                self._expert_candidate_order_by_layer[int(layer_id)] = [
                    int(x) for x in cpu_order.tolist()
                ]
                self.stats.expert_candidate_snapshot_ready_count += 1
            except Exception:
                self.stats.expert_candidate_snapshot_drop_count += 1
            finally:
                if not keep_pending:
                    self._expert_candidate_pending_layers.discard(int(layer_id))
        self._expert_candidate_pending_snapshots = remaining

    def _finalize_expert_hotness_snapshots(self, *, block: bool = False) -> None:
        if not self._expert_hotness_pending_snapshots:
            return
        t0 = time.perf_counter() if self.config.profile_detail else 0.0
        remaining = []
        for (
            mode,
            layer_id,
            cpu_counts,
            event,
            snapshot_counts,
        ) in self._expert_hotness_pending_snapshots:
            key = (str(mode), int(layer_id))
            keep_pending = False
            try:
                if block:
                    event.synchronize()
                elif not event.query():
                    remaining.append(
                        (mode, layer_id, cpu_counts, event, snapshot_counts)
                    )
                    keep_pending = True
                    continue
                target = (
                    self._expert_hotness_decode.setdefault(int(layer_id), {})
                    if mode == "decode"
                    else self._expert_hotness_prefill.setdefault(int(layer_id), {})
                )
                target.clear()
                nonzero = torch.nonzero(cpu_counts, as_tuple=False).flatten()
                if int(nonzero.numel()) > 0:
                    expert_ids = nonzero.tolist()
                    counts = cpu_counts[nonzero].tolist()
                    for expert_id, count in zip(expert_ids, counts):
                        target[int(expert_id)] = int(count)
                    self.stats.expert_hotness_observed = True
                    self._expert_hotness_version += 1
                self.stats.expert_hotness_snapshot_count += 1
                self.stats.expert_hotness_snapshot_ready_count += 1
            except Exception:
                self.stats.expert_hotness_sync_fallback_count += 1
            finally:
                if not keep_pending:
                    self._expert_hotness_pending_snapshot_keys.discard(key)
        self._expert_hotness_pending_snapshots = remaining
        if self.config.profile_detail:
            self._add_profile(
                "profile_expert_hotness_snapshot_ms",
                (time.perf_counter() - t0) * 1000.0,
            )

    def _materialize_expert_hotness_gpu_counts(
        self, *, mode: Optional[str] = None
    ) -> None:
        stores: List[Tuple[str, Dict[int, torch.Tensor], Dict[int, Dict[int, int]]]] = []
        if mode in (None, "prefill"):
            stores.append(
                ("prefill", self._expert_hotness_gpu_prefill, self._expert_hotness_prefill)
            )
        if mode in (None, "decode"):
            stores.append(
                ("decode", self._expert_hotness_gpu_decode, self._expert_hotness_decode)
            )
        if not any(gpu_store for _mode, gpu_store, _cpu_store in stores):
            return
        t0 = time.perf_counter() if self.config.profile_detail else 0.0
        for _mode, gpu_store, cpu_store in stores:
            for layer_id, counts in gpu_store.items():
                try:
                    cpu_counts = counts.detach().cpu()
                    target = cpu_store.setdefault(int(layer_id), {})
                    target.clear()
                    nonzero = torch.nonzero(cpu_counts, as_tuple=False).flatten()
                    if int(nonzero.numel()) > 0:
                        expert_ids = nonzero.tolist()
                        values = cpu_counts[nonzero].tolist()
                        for expert_id, count in zip(expert_ids, values):
                            target[int(expert_id)] = int(count)
                        self.stats.expert_hotness_observed = True
                    self.stats.expert_hotness_snapshot_forced_count += 1
                    self.stats.expert_hotness_snapshot_count += 1
                except Exception:
                    self.stats.expert_hotness_sync_fallback_count += 1
        self._expert_hotness_version += 1
        if self.config.profile_detail:
            self._add_profile(
                "profile_expert_hotness_snapshot_ms",
                (time.perf_counter() - t0) * 1000.0,
            )

    def _ensure_expert_hotness_cpu_view(
        self, *, mode: Optional[str] = None, block: bool = False
    ) -> None:
        self._finalize_expert_hotness_snapshots(block=block)
        self._finalize_expert_candidate_snapshots(block=block)
        if block:
            self._materialize_expert_hotness_gpu_counts(mode=mode)

    def _record_expert_hotness_cpu(
        self, layer_id: int, full_num_experts: int, topk_ids: torch.Tensor
    ) -> None:
        ids = topk_ids.detach()
        ids = ids[(ids >= 0) & (ids < full_num_experts)]
        if ids.numel() == 0:
            return
        target = (
            self._expert_hotness_decode.setdefault(layer_id, {})
            if self._current_forward_mode == "decode"
            else self._expert_hotness_prefill.setdefault(layer_id, {})
        )
        values, counts = torch.unique(ids.cpu(), return_counts=True)
        total = 0
        for expert_id, count in zip(values.tolist(), counts.tolist()):
            expert_id = int(expert_id)
            count = int(count)
            target[expert_id] = target.get(expert_id, 0) + count
            total += count
        if total > 0:
            self._expert_hotness_version += 1
        self.stats.expert_call_count_total += total
        if self._current_forward_mode == "decode":
            self.stats.expert_decode_call_count_total += total
        else:
            self.stats.expert_prefill_call_count_total += total
        self.stats.expert_hotness_observed = True

    def _record_expert_hotness_gpu(
        self, layer_id: int, full_num_experts: int, topk_ids: torch.Tensor
    ) -> None:
        ids = topk_ids.detach().reshape(-1)
        if ids.numel() == 0:
            return
        mode = "decode" if self._current_forward_mode == "decode" else "prefill"
        counts = self._expert_hotness_gpu_counts(
            mode, int(layer_id), int(full_num_experts), ids.device
        )
        index = ids.to(dtype=torch.long, non_blocking=True)
        ones = self._expert_hotness_ones(ids.device, int(index.numel()))
        counts.index_add_(0, index, ones)
        total = int(index.numel())
        self.stats.expert_call_count_total += total
        if mode == "decode":
            self.stats.expert_decode_call_count_total += total
        else:
            self.stats.expert_prefill_call_count_total += total
        self.stats.expert_hotness_observed = True
        self.stats.expert_hotness_record_fast_count += 1
        self.stats.expert_hotness_record_count += 1
        self._maybe_issue_expert_hotness_snapshot(mode, int(layer_id), counts)
        self._maybe_issue_expert_candidate_snapshot(
            mode, int(layer_id), int(full_num_experts), ids.device
        )

    def _record_expert_hotness_for_layer(
        self, layer_id: int, full_num_experts: int, topk_ids: torch.Tensor
    ) -> None:
        if topk_ids.numel() == 0:
            return
        if not self._should_sample_expert_hotness(layer_id):
            self._expert_hotness_sample_skip_pending += 1
            return
        with self._profile("profile_expert_hotness_record_ms"):
            ids = topk_ids.detach().reshape(-1)
            if ids.numel() == 0:
                return
            if ids.device.type == "cuda":
                self._record_expert_hotness_gpu(
                    int(layer_id), int(full_num_experts), ids
                )
            else:
                self.stats.expert_hotness_sync_fallback_count += 1
                self.stats.expert_hotness_record_safe_count += 1
                self.stats.expert_hotness_record_count += 1
                self._record_expert_hotness_cpu(
                    int(layer_id), int(full_num_experts), ids
                )

    def _refresh_expert_ready_before_use_ratio(self) -> None:
        total = self.stats.expert_ready_use_check_count
        if total <= 0:
            self.stats.expert_ready_before_use_ratio = 1.0
        else:
            self.stats.expert_ready_before_use_ratio = (
                self.stats.expert_ready_before_use_count / float(total)
            )

    def _wait_for_expert_logical_ids_ready(
        self, state: _LayerKVExpertLayerState, logical_ids: List[int]
    ) -> None:
        if not self._pending_expert_copy_events or state.device.type != "cuda":
            return
        needed = set(int(x) for x in logical_ids)
        if not needed:
            return
        stream = torch.cuda.current_stream(device=state.device)
        for pending in self._pending_expert_copy_events:
            if pending.waited_on_main_stream or pending.layer_id != state.layer_id:
                continue
            if not pending.logical_ids.intersection(needed):
                continue
            self.stats.expert_ready_use_check_count += 1
            self.stats.scheduler_ready_use_check_count += 1
            ready = bool(pending.ready_event.query())
            if ready:
                self.stats.expert_ready_before_use_count += 1
                self.stats.scheduler_ready_before_use_count += 1
            else:
                self.stats.layerkv_deadline_miss_count += 1
                self.stats.scheduler_deadline_miss_count += 1
            t_wait = time.perf_counter()
            stream.wait_event(pending.ready_event)
            self.stats.scheduler_exposed_wait_ms += (
                time.perf_counter() - t_wait
            ) * 1000.0
            self.stats.expert_copy_stream_wait_count += 1
            self.stats.layerkv_copy_event_wait_count += 1
            for logical_id in pending.logical_ids:
                group = self._resident_groups.get(
                    self._residency_key("expert", state.layer_id, int(logical_id))
                )
                if group is not None:
                    group.wait_count += 1
                    group.ready_waited = True
            self.stats.resident_group_wait_count += len(pending.logical_ids)
            pending.waited_on_main_stream = True
        self._refresh_expert_ready_before_use_ratio()

    def _finalize_expert_materialize_events(self, *, block: bool = False) -> None:
        if not self._pending_expert_copy_events:
            return
        remaining = []
        for pending in self._pending_expert_copy_events:
            start = pending.start_event
            end = pending.ready_event
            try:
                if block:
                    end.synchronize()
                elif not end.query():
                    remaining.append(pending)
                    continue
                elapsed = float(start.elapsed_time(end))
                self.stats.expert_materialize_ms += elapsed
                self.stats.layerkv_copy_stream_busy_ms += elapsed
                state = self._expert_layers.get(int(pending.layer_id))
                if state is not None:
                    for logical_id in pending.logical_ids:
                        slot_id = state.logical_to_slot.get(int(logical_id))
                        if slot_id is not None:
                            group = self._get_or_create_resident_group(
                                kind="expert",
                                layer_id=state.layer_id,
                                logical_id=int(logical_id),
                                state="resident",
                                bytes=state.expert_bytes,
                            )
                            self._mark_expert_group_state(
                                state,
                                int(logical_id),
                                group_state="resident",
                                slot_id=int(slot_id),
                            )
                            group.recover_count += 1
                            self.stats.resident_group_recover_count += 1
            except Exception:
                continue
        self._pending_expert_copy_events = remaining
        self._refresh_resident_group_stats()

    def _unique_expert_ids(
        self, topk_ids: torch.Tensor, full_num_experts: int
    ) -> List[int]:
        if topk_ids.numel() == 0:
            return []
        ids = topk_ids.detach()
        ids = ids[(ids >= 0) & (ids < full_num_experts)]
        if ids.numel() == 0:
            return []
        with self._profile("profile_expert_unique_ms"):
            return [int(x) for x in torch.unique(ids).detach().cpu().tolist()]

    def _unique_expert_ids_and_record_hotness(
        self,
        state: _LayerKVExpertLayerState,
        topk_ids: torch.Tensor,
        *,
        record_hotness: bool,
    ) -> List[int]:
        if topk_ids.numel() == 0:
            return []
        ids = topk_ids.detach()
        ids = ids[(ids >= 0) & (ids < state.full_num_experts)]
        if ids.numel() == 0:
            return []
        if record_hotness:
            self._record_expert_hotness_for_layer(
                state.layer_id, state.full_num_experts, topk_ids
            )
        with self._profile("profile_expert_unique_ms"):
            values = torch.unique(ids).detach().cpu()
        return [int(x) for x in values.tolist()]

    def _materialize_experts(
        self,
        state: _LayerKVExpertLayerState,
        logical_ids: List[int],
        *,
        reason: str = "on_demand",
    ) -> None:
        t_profile = time.perf_counter() if self.config.debug_stats else 0.0
        if not logical_ids:
            return
        try:
            unique_logical_ids = list(dict.fromkeys(int(x) for x in logical_ids))
            self.stats.expert_materialize_dedup_count += max(
                0, len(logical_ids) - len(unique_logical_ids)
            )
            if reason == "on_demand" and state.prefetched_logical_ids:
                used_prefetch = state.prefetched_logical_ids.intersection(
                    unique_logical_ids
                )
                if used_prefetch:
                    self.stats.expert_prefetch_useful_count += len(used_prefetch)
                    self.stats.expert_prefetch_ready_before_use_count += len(
                        used_prefetch
                    )
                    state.prefetched_logical_ids.difference_update(used_prefetch)
            protected = set(unique_logical_ids)
            materialized: List[Tuple[int, int, Dict[str, torch.Tensor]]] = []
            evicted_to_copy: List[Tuple[int, int]] = []
            for logical_id in unique_logical_ids:
                slot_id = state.logical_to_slot.get(int(logical_id))
                if slot_id is not None:
                    state.lru[int(logical_id)] = self._decode_step
                    if int(logical_id) in state.cpu_params:
                        state.backing_lru[int(logical_id)] = self._decode_step
                    heapq.heappush(
                        state.lru_heap,
                        (self._decode_step, int(slot_id), int(logical_id)),
                    )
                    continue

                slot_id = self._choose_expert_slot_for_materialize(state, protected)
                evicted = state.slot_to_logical.get(slot_id)
                if evicted is not None:
                    evicted = int(evicted)
                    global_evicted = (
                        self._global_expert_backing(state.layer_id, evicted)
                        if self._expert_global_cpu_backing
                        else None
                    )
                    if global_evicted is not None:
                        state.cpu_params[evicted] = global_evicted
                    if int(evicted) in state.cpu_params:
                        self.stats.expert_eviction_d2h_skip_count += 1
                        self.stats.expert_backing_cache_hit_count += 1
                    else:
                        self.stats.expert_eviction_d2h_copy_count += 1
                        self.stats.expert_backing_cache_miss_count += 1
                        evicted_to_copy.append((evicted, int(slot_id)))
                    state.backing_lru[evicted] = self._decode_step
                    state.logical_to_slot.pop(evicted, None)
                    state.lru.pop(evicted, None)
                    if state.remap_tensor is not None:
                        state.remap_tensor[evicted] = -1
                    if reason == "prefetch":
                        self._mark_expert_group_state(
                            state,
                            evicted,
                            group_state="offloaded",
                            cpu_params=state.cpu_params.get(evicted),
                        )
                source_params = state.cpu_params.get(int(logical_id))
                if source_params is None:
                    source_params = self._wait_for_pending_expert_backing(
                        state, int(logical_id)
                    )
                if source_params is None:
                    source_params = (
                        self._global_expert_backing(state.layer_id, int(logical_id))
                        if self._expert_global_cpu_backing
                        else None
                    )
                    if source_params is not None:
                        state.cpu_params[int(logical_id)] = source_params
                if source_params is None:
                    err = f"layer {state.layer_id} missing CPU backing for expert {logical_id}"
                    self.stats.expert_guard_pass = False
                    self.stats.expert_guard_reason = err
                    raise RuntimeError(err)
                materialized.append((int(logical_id), int(slot_id), source_params))
                state.logical_to_slot[logical_id] = slot_id
                state.slot_to_logical[slot_id] = logical_id
                state.lru[logical_id] = self._decode_step
                heapq.heappush(state.lru_heap, (self._decode_step, slot_id, logical_id))
                if state.remap_tensor is not None:
                    state.remap_tensor[int(logical_id)] = int(slot_id)
                if reason == "prefetch":
                    self._mark_expert_group_state(
                        state,
                        int(logical_id),
                        group_state="materializing",
                        slot_id=int(slot_id),
                        cpu_params=source_params,
                    )
                state.materialize_step += 1
                self.stats.expert_materialize_count += 1
                if reason == "prefetch":
                    self.stats.expert_prefetch_mb_total += state.expert_bytes / float(
                        1024 * 1024
                    )
                    state.prefetched_logical_ids.add(int(logical_id))
                else:
                    self.stats.expert_on_demand_materialize_count += 1
                self.stats.layerkv_expert_materialize_started += 1
                self.stats.layerkv_tasks_built += 1
                self.stats.expert_materialize_mb_total += state.expert_bytes / float(
                    1024 * 1024
                )
                state.backing_lru[int(logical_id)] = self._decode_step
            if materialized:
                if reason != "prefetch":
                    self._expert_group_dirty_layers.add(int(state.layer_id))
                if (
                    reason == "prefetch"
                    and self._optimized_profile_enabled()
                    and self._copy_slots_to_cpu_batched_async(state, evicted_to_copy)
                ):
                    pass
                else:
                    copied = self._copy_slots_to_cpu_batched(state, evicted_to_copy)
                    state.cpu_params.update(copied)
                self._copy_materialized_experts_batched(
                    state,
                    materialized,
                    async_copy=(reason == "prefetch"),
                )
                self._trim_expert_backing_cache(state)
        finally:
            if self.config.profile_detail:
                self._add_profile(
                    "profile_expert_materialize_control_ms",
                    (time.perf_counter() - t_profile) * 1000.0,
                )

    def _copy_materialized_experts_batched(
        self,
        state: _LayerKVExpertLayerState,
        materialized: List[Tuple[int, int, Dict[str, torch.Tensor]]],
        *,
        async_copy: bool,
    ) -> None:
        if not materialized:
            return
        for logical_id, slot_id, source_params in materialized:
            self._record_expert_copy_descriptor(
                direction="H2D",
                reason="materialize_async" if async_copy else "materialize_sync",
                layer_id=int(state.layer_id),
                logical_id=int(logical_id),
                src_slot=int(logical_id),
                dst_slot=int(slot_id),
                nbytes=self._expert_backing_bytes(source_params),
                param_count=len(source_params),
            )
        start = None
        end = None
        active_stream = None
        if state.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            active_stream = (
                (self._copy_stream or torch.cuda.current_stream(device=state.device))
                if async_copy
                else torch.cuda.current_stream(device=state.device)
            )

        def issue_copy() -> None:
            if active_stream is not None:
                start.record(active_stream)
            for name in state.param_names:
                for _logical_id, slot_id, source_params in materialized:
                    dst = getattr(state.module, name).data[slot_id]
                    dst.copy_(source_params[name], non_blocking=True)
            if active_stream is not None:
                end.record(active_stream)

        with torch.no_grad():
            if active_stream is not None:
                with torch.cuda.stream(active_stream):
                    issue_copy()
            else:
                issue_copy()
        batch_size = len(materialized)
        self._expert_materialize_batch_sizes.append(batch_size)
        self._expert_materialize_layers_touched.add(int(state.layer_id))
        self.stats.expert_materialize_batch_count += 1
        self.stats.expert_materialize_event_count += 1 if end is not None else 0
        if self.stats.expert_materialize_batch_count > 0:
            self.stats.expert_materialize_avg_batch_size = (
                self.stats.expert_materialize_count
                / float(self.stats.expert_materialize_batch_count)
            )
        if end is not None:
            try:
                if end.query():
                    elapsed = float(start.elapsed_time(end))
                    self.stats.expert_materialize_ms += elapsed
                    self.stats.layerkv_copy_stream_busy_ms += elapsed
                    if async_copy:
                        for logical_id, slot_id, source_params in materialized:
                            group = self._get_or_create_resident_group(
                                kind="expert",
                                layer_id=state.layer_id,
                                logical_id=int(logical_id),
                                state="resident",
                                bytes=state.expert_bytes,
                            )
                            self._mark_expert_group_state(
                                state,
                                int(logical_id),
                                group_state="resident",
                                slot_id=int(slot_id),
                                cpu_params=source_params,
                            )
                            group.recover_count += 1
                    self.stats.resident_group_recover_count += batch_size
                elif async_copy:
                    self.stats.expert_materialize_async_count += batch_size
                    self.stats.expert_copy_stream_launch_count += 1
                    self.stats.layerkv_copy_event_record_count += 1
                    for logical_id, slot_id, source_params in materialized:
                        self._mark_expert_group_state(
                            state,
                            int(logical_id),
                            group_state="materializing",
                            slot_id=int(slot_id),
                            cpu_params=source_params,
                            ready_start_event=start,
                            ready_event=end,
                        )
                    self._pending_expert_copy_events.append(
                        _LayerKVPendingExpertCopy(
                            start_event=start,
                            ready_event=end,
                            layer_id=state.layer_id,
                            logical_ids=set(int(x[0]) for x in materialized),
                        )
                    )
                else:
                    self.stats.expert_materialize_host_sync_count += batch_size
                    self.stats.resident_group_recover_count += batch_size
            except Exception:
                pass
        else:
            self.stats.expert_materialize_host_sync_count += batch_size
            self.stats.resident_group_recover_count += batch_size

    def _grow_expert_layer_slots(
        self, state: _LayerKVExpertLayerState, required_capacity: int
    ) -> None:
        new_capacity = min(
            state.full_num_experts, max(required_capacity, state.slot_capacity)
        )
        if new_capacity <= state.slot_capacity:
            return
        old_capacity = state.slot_capacity
        with torch.no_grad():
            for name in state.param_names:
                param = getattr(state.module, name)
                old = param.data
                new_data = torch.empty(
                    (new_capacity,) + tuple(old.shape[1:]),
                    dtype=old.dtype,
                    device=old.device,
                )
                if old_capacity > 0:
                    new_data[:old_capacity].copy_(old[:old_capacity])
                param.data = new_data
        state.slot_capacity = new_capacity
        for slot_id in range(old_capacity, new_capacity):
            heapq.heappush(state.free_slots, slot_id)
        try:
            state.module.num_experts = new_capacity
            state.module.num_local_experts = new_capacity
            state.module.moe_runner_config.num_experts = new_capacity
            state.module.moe_runner_config.num_local_experts = new_capacity
            state.module.dispatcher.num_experts = new_capacity
            state.module.dispatcher.num_local_experts = new_capacity
            state.module.dispatcher.num_local_routed_experts = new_capacity
        except Exception:
            pass
        self.stats.expert_slot_rebind_count += 1
        self._refresh_expert_stats()

    def _choose_expert_slot_for_materialize(
        self, state: _LayerKVExpertLayerState, protected: Set[int]
    ) -> int:
        while state.free_slots:
            slot_id = int(heapq.heappop(state.free_slots))
            if (
                0 <= slot_id < state.slot_capacity
                and slot_id not in state.slot_to_logical
            ):
                return slot_id
        while state.lru_heap:
            step, slot_id, logical_id = heapq.heappop(state.lru_heap)
            if logical_id in protected:
                continue
            if state.logical_to_slot.get(logical_id) != slot_id:
                continue
            if state.lru.get(logical_id) != step:
                continue
            return int(slot_id)
        for slot_id, logical_id in state.slot_to_logical.items():
            if logical_id not in protected:
                return int(slot_id)
        raise RuntimeError(
            f"layer {state.layer_id} has no evictable expert slot for materialization"
        )

    def _expert_hotness_score(
        self, state: _LayerKVExpertLayerState, logical_id: int
    ) -> int:
        logical_id = int(logical_id)
        return int(state.hotness_decode.get(logical_id, 0)) * 8 + int(
            state.hotness_prefill.get(logical_id, 0)
        )

    def _coldest_resident_expert_score(
        self, state: _LayerKVExpertLayerState, protected: Set[int]
    ) -> int:
        coldest: Optional[int] = None
        for logical_id in state.logical_to_slot:
            logical_id = int(logical_id)
            if logical_id in protected:
                continue
            score = self._expert_hotness_score(state, logical_id)
            if coldest is None or score < coldest:
                coldest = score
        return 0 if coldest is None else int(coldest)

    def _refresh_expert_stats(self) -> None:
        if not self._expert_layers:
            return
        full_bytes = sum(state.full_bytes for state in self._expert_layers.values())
        reclaim_bytes = sum(
            state.physical_reclaim_bytes for state in self._expert_layers.values()
        )
        self.stats.physical_expert_reclaim_mb = reclaim_bytes / float(1024 * 1024)
        self._refresh_expert_host_backing_stat()
        self.stats.expert_slot_capacity_total = sum(
            state.slot_capacity for state in self._expert_layers.values()
        )
        self.stats.expert_resident_count = sum(
            state.resident_count for state in self._expert_layers.values()
        )
        self.stats.expert_offloaded_count = sum(
            state.offloaded_count for state in self._expert_layers.values()
        )
        self.stats.expert_terminal_slot_count = self.stats.expert_slot_capacity_total
        self.stats.expert_terminal_offloaded_count = self.stats.expert_offloaded_count
        self.stats.expert_terminal_metadata_mapped_count = (
            self.stats.expert_resident_count
        )
        self._refresh_physical_reclaim_peaks()
        self._refresh_resident_group_stats()
        if self.stats.comparability_reason == "INSUFFICIENT_EXPERT_RECLAIM":
            self.stats.comparable = True
            self.stats.comparability_reason = ""

    def _allocator_available_size(self) -> int:
        if self._allocator is None:
            return -1
        available = getattr(self._allocator, "available_size", None)
        if available is None or not callable(available):
            return -1
        try:
            return int(available())
        except Exception:
            return -1

    def _allocator_total_size(self) -> int:
        if self._allocator is None:
            return -1
        for attr in ("size_full", "size"):
            value = getattr(self._allocator, attr, None)
            try:
                if value is not None:
                    return int(value() if callable(value) else value)
            except Exception:
                continue
        return -1

    def _dynamic_needed_pressure_mb(self, forward_batch: Any = None) -> float:
        if self._bytes_per_token_all_layers <= 0:
            return 0.0
        total_tokens = self._allocator_total_size()
        if total_tokens <= 0:
            return 0.0
        budget_tokens = int(total_tokens)
        pairs = (
            list(self._current_forward_req_lens)
            if forward_batch is not None
            and self._current_forward_req_lens_batch_id == id(forward_batch)
            else (
                self._batch_req_indices_and_lens(forward_batch)
                if forward_batch is not None
                else []
            )
        )
        batch_size = len(pairs)
        live_tokens = sum(max(0, int(seq_len)) for _req_idx, seq_len in pairs)

        # Hard admission pressure only: the SGLang KV table size already reflects
        # mem_fraction_static.  LayerKV must compare active KV demand against the
        # actual table capacity and must not multiply by mem_fraction again.
        write_tokens = 0
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if isinstance(out_cache_loc, torch.Tensor):
            write_tokens = int(out_cache_loc.numel())
        elif out_cache_loc is not None:
            try:
                write_tokens = len(out_cache_loc)
            except TypeError:
                write_tokens = 0
        decode_reserve = max(
            0, batch_size if self._current_forward_mode == "decode" else 0
        )
        demand_tokens = int(live_tokens) + int(write_tokens) + int(decode_reserve)
        runtime_shortage_tokens = max(
            0,
            demand_tokens - int(budget_tokens),
        )
        scheduler_shortage_tokens = max(0, int(self._scheduler_pressure_tokens))
        shortage_tokens = max(runtime_shortage_tokens, scheduler_shortage_tokens)
        self.stats.dynamic_pressure_live_tokens = int(live_tokens)
        self.stats.dynamic_pressure_write_tokens = int(write_tokens)
        self.stats.dynamic_pressure_decode_reserve_tokens = int(decode_reserve)
        self.stats.dynamic_pressure_total_tokens = int(total_tokens)
        self.stats.dynamic_pressure_budget_tokens = int(budget_tokens)
        self.stats.dynamic_pressure_shortage_tokens = int(shortage_tokens)
        self.stats.dynamic_pressure_token_usage_ratio = float(demand_tokens) / float(
            max(1, total_tokens)
        )
        return shortage_tokens * self._bytes_per_token_all_layers / float(1024 * 1024)

    def _align_tokens_down(self, token_count: int) -> int:
        page_size = max(1, int(self._page_size))
        return max(0, int(token_count) // page_size * page_size)

    def _align_tokens_up(self, token_count: int) -> int:
        page_size = max(1, int(self._page_size))
        if token_count <= 0:
            return 0
        return int(math.ceil(float(token_count) / float(page_size)) * page_size)

    def _per_layer_kvc_block_page_size(self) -> int:
        base_page = max(1, int(self._page_size))
        if self._per_layer_allocator_enabled():
            return base_page
        block_tokens = int(getattr(self.config, "kvc_block_tokens", 0) or 0)
        if block_tokens <= base_page:
            return base_page
        return max(base_page, self._align_tokens_up(block_tokens))

    def _per_layer_allocator_enabled(self) -> bool:
        return (
            self.config.kvc_backend == "per-layer-arena"
            and self.config.mode in ("kvc-only", "kvc-expert")
            and self.physical_kvc_supported
            and self._allocator is not None
            and self._kv_pool is not None
        )

    def uses_per_layer_logical_allocator(self) -> bool:
        return self._per_layer_allocator_enabled() and bool(
            self._per_layer_arena_reserved_locs
            or self._per_layer_owned_req_indices
        )

    def requires_unfused_set_kv_buffer(self) -> bool:
        if not self._per_layer_allocator_enabled():
            return False
        if self.config.kvc_backend != "per-layer-arena":
            return self.uses_per_layer_logical_allocator()
        return bool(
            self._per_layer_req_to_token_owned
            or self._per_layer_offloaded_keys
            or self._pending_virtual_kvc_materialize
        )

    def _refresh_per_layer_allocator_stats(self) -> None:
        if not self._per_layer_arena_free_locs:
            self.stats.kvc_per_layer_physical_arena_token_capacity = 0
            self.stats.kvc_per_layer_physical_arena_min_free_tokens = 0
            self.stats.kvc_per_layer_physical_arena_common_free_tokens = 0
            self.stats.kvc_per_layer_logical_request_count = len(
                self._per_layer_owned_req_indices
            )
            return
        free_counts = [
            len(self._per_layer_arena_free_locs.get(int(layer_id), []))
            for layer_id in self._kvc_layer_ids()
        ]
        if free_counts:
            self.stats.kvc_per_layer_physical_arena_min_free_tokens = min(free_counts)
        else:
            self.stats.kvc_per_layer_physical_arena_min_free_tokens = 0
        self.stats.kvc_per_layer_physical_arena_common_free_tokens = (
            len(self._per_layer_arena_common_free_locs)
        )
        self.stats.kvc_per_layer_physical_arena_token_capacity = len(
            self._per_layer_arena_reserved_locs
        )
        self.stats.kvc_per_layer_physical_arena_native_reserved_tokens = len(
            self._per_layer_arena_reserved_locs
        )
        self.stats.kvc_per_layer_logical_request_count = len(
            self._per_layer_owned_req_indices
        )

    def _ensure_per_layer_physical_arena(self, min_free_tokens: int = 1) -> bool:
        if not self._per_layer_allocator_enabled():
            return False
        layer_ids = self._kvc_layer_ids()
        if not layer_ids:
            return False
        min_free_tokens = max(1, int(min_free_tokens or 1))
        self._refresh_per_layer_allocator_stats()
        if (
            self.stats.kvc_per_layer_physical_arena_common_free_tokens
            >= min_free_tokens
        ):
            return True
        available = int(self._allocator.available_size())
        if available < min_free_tokens:
            self.stats.kvc_per_layer_physical_arena_alloc_failed_count += 1
            return False
        target = max(
            min_free_tokens,
            int(getattr(self.config, "virtual_scratch_tokens", 0) or 0),
            int(getattr(self.config, "kvc_block_tokens", 16) or 16) * 64,
        )
        grow = min(max(min_free_tokens, target), available)
        locs = self._allocator.alloc(int(grow))
        if locs is None or int(locs.numel()) == 0:
            self.stats.kvc_per_layer_physical_arena_alloc_failed_count += 1
            return False
        loc_list = [int(x) for x in locs.detach().cpu().tolist()]
        for loc in loc_list:
            self._per_layer_arena_reserved_locs.add(int(loc))
            if int(loc) not in self._per_layer_arena_common_free_locs:
                self._per_layer_arena_common_free_locs.add(int(loc))
                self._per_layer_arena_common_free_order.append(int(loc))
        for layer_id in layer_ids:
            layer_id = int(layer_id)
            self._per_layer_arena_free_locs.setdefault(layer_id, []).extend(loc_list)
            self._per_layer_arena_allocated_locs.setdefault(layer_id, set())
            self._per_layer_canonical_to_physical.setdefault(layer_id, {})
        self.stats.kvc_per_layer_physical_arena_grow_count += 1
        self._refresh_per_layer_allocator_stats()
        return (
            self.stats.kvc_per_layer_physical_arena_common_free_tokens
            >= min_free_tokens
        )

    def _alloc_per_layer_locs(
        self, layer_id: int, count: int
    ) -> Optional[List[int]]:
        layer_id = int(layer_id)
        count = max(0, int(count))
        if count <= 0:
            return []
        free = self._per_layer_arena_free_locs.setdefault(layer_id, [])
        allocated = self._per_layer_arena_allocated_locs.setdefault(layer_id, set())
        protected = self._per_layer_arena_protected_locs.setdefault(layer_id, set())
        cleaned: List[int] = []
        seen: Set[int] = set()
        for loc in free:
            loc = int(loc)
            if loc <= 0 or loc in allocated or loc in protected or loc in seen:
                continue
            seen.add(loc)
            cleaned.append(loc)
        if len(cleaned) != len(free):
            self._per_layer_arena_free_locs[layer_id] = cleaned
            free = cleaned
        if len(free) < count and self._per_layer_arena_common_free_locs:
            present = {int(loc) for loc in free}
            for loc in sorted(self._per_layer_arena_common_free_locs):
                loc = int(loc)
                if loc <= 0 or loc in allocated or loc in protected or loc in present:
                    continue
                free.append(loc)
                present.add(loc)
                if len(free) >= count:
                    break
        if len(free) < count:
            if not self._ensure_per_layer_physical_arena(count - len(free)):
                self.stats.kvc_per_layer_physical_arena_alloc_failed_count += 1
                return None
            free = self._per_layer_arena_free_locs.setdefault(layer_id, [])
        if len(free) < count:
            self.stats.kvc_per_layer_physical_arena_alloc_failed_count += 1
            return None
        locs = free[-count:]
        del free[-count:]
        for loc in locs:
            self._per_layer_arena_common_free_locs.discard(int(loc))
        self._per_layer_arena_allocated_locs.setdefault(layer_id, set()).update(locs)
        self.stats.kvc_per_layer_physical_arena_alloc_count += count
        self._refresh_per_layer_allocator_stats()
        return [int(x) for x in locs]

    def _alloc_common_per_layer_locs(self, count: int) -> Optional[List[int]]:
        count = max(0, int(count))
        if count <= 0:
            return []
        if not self._ensure_per_layer_physical_arena(count):
            return None
        layer_ids = [int(x) for x in self._kvc_layer_ids()]
        common = self._per_layer_arena_common_free_locs
        if len(common) < count:
            self.stats.kvc_per_layer_physical_arena_alloc_failed_count += 1
            return None
        locs: List[int] = []
        while self._per_layer_arena_common_free_order and len(locs) < count:
            loc = int(self._per_layer_arena_common_free_order.pop())
            if loc in common and not any(
                self._per_layer_loc_is_protected(layer_id, loc)
                for layer_id in layer_ids
            ):
                locs.append(loc)
        if len(locs) < count:
            for loc in sorted(common):
                loc = int(loc)
                if loc in locs or any(
                    self._per_layer_loc_is_protected(layer_id, loc)
                    for layer_id in layer_ids
                ):
                    continue
                locs.append(loc)
                if len(locs) >= count:
                    break
        if len(locs) < count:
            self.stats.kvc_per_layer_physical_arena_alloc_failed_count += 1
            return None
        locs.sort()
        loc_set = set(locs)
        self._per_layer_arena_common_free_locs.difference_update(loc_set)
        for layer_id in layer_ids:
            free = self._per_layer_arena_free_locs.setdefault(layer_id, [])
            self._per_layer_arena_free_locs[layer_id] = [
                loc for loc in free if loc not in loc_set
            ]
            self._per_layer_arena_allocated_locs.setdefault(layer_id, set()).update(
                locs
            )
        self.stats.kvc_per_layer_physical_arena_alloc_count += count * len(layer_ids)
        self.stats.kvc_per_layer_common_alloc_count += count
        self._refresh_per_layer_allocator_stats()
        return [int(x) for x in locs]

    def _per_layer_loc_is_protected(self, layer_id: int, loc: int) -> bool:
        return int(loc) in self._per_layer_arena_protected_locs.setdefault(
            int(layer_id), set()
        )

    def _protect_per_layer_terminal_locs(self, layer_id: int, locs: List[int]) -> None:
        layer_id = int(layer_id)
        protected = self._per_layer_arena_protected_locs.setdefault(layer_id, set())
        protected.update(int(loc) for loc in locs if int(loc) > 0)

    def _release_protected_per_layer_locs(
        self, layer_id: int, locs: List[int], *, refresh: bool = True
    ) -> None:
        layer_id = int(layer_id)
        loc_set = {int(loc) for loc in locs if int(loc) > 0}
        if not loc_set:
            return
        protected = self._per_layer_arena_protected_locs.setdefault(layer_id, set())
        reusable = loc_set.intersection(protected)
        if not reusable:
            return
        protected.difference_update(reusable)
        free = self._per_layer_arena_free_locs.setdefault(layer_id, [])
        allocated = self._per_layer_arena_allocated_locs.setdefault(layer_id, set())
        free_seen = set(free)
        free.extend(
            loc
            for loc in sorted(reusable)
            if loc not in free_seen and loc not in allocated
        )

        common_reusable = set(reusable)
        for other_layer_id in self._kvc_layer_ids():
            other_layer_id = int(other_layer_id)
            if other_layer_id == layer_id:
                continue
            common_reusable.difference_update(
                self._per_layer_arena_allocated_locs.setdefault(other_layer_id, set())
            )
            common_reusable.difference_update(
                self._per_layer_arena_protected_locs.setdefault(other_layer_id, set())
            )
            if not common_reusable:
                break
        if common_reusable:
            new_common = [
                int(loc)
                for loc in sorted(common_reusable)
                if int(loc) not in self._per_layer_arena_common_free_locs
            ]
            self._per_layer_arena_common_free_locs.update(new_common)
            self._per_layer_arena_common_free_order.extend(new_common)
        if refresh:
            self._refresh_per_layer_allocator_stats()

    def _free_per_layer_locs(
        self, layer_id: int, locs: List[int], *, refresh: bool = True
    ) -> None:
        if not locs:
            return
        layer_id = int(layer_id)
        allocated = self._per_layer_arena_allocated_locs.setdefault(layer_id, set())
        free = self._per_layer_arena_free_locs.setdefault(layer_id, [])
        loc_set = {int(x) for x in locs}
        reusable = loc_set.intersection(allocated)
        if not reusable:
            return
        allocated.difference_update(reusable)
        protected = self._per_layer_arena_protected_locs.setdefault(layer_id, set())
        ordinary_reusable = reusable.difference(protected)

        free_seen = set(free)
        free.extend(loc for loc in ordinary_reusable if loc not in free_seen)
        self.stats.kvc_per_layer_physical_arena_free_count += len(reusable)

        common_reusable = set(ordinary_reusable)
        for other_layer_id in self._kvc_layer_ids():
            if int(other_layer_id) == layer_id:
                continue
            common_reusable.difference_update(
                self._per_layer_arena_allocated_locs.setdefault(
                    int(other_layer_id), set()
                )
            )
            common_reusable.difference_update(
                self._per_layer_arena_protected_locs.setdefault(
                    int(other_layer_id), set()
                )
            )
            if not common_reusable:
                break
        if common_reusable:
            new_common = [
                int(loc)
                for loc in sorted(common_reusable)
                if int(loc) not in self._per_layer_arena_common_free_locs
            ]
            self._per_layer_arena_common_free_locs.update(new_common)
            self._per_layer_arena_common_free_order.extend(new_common)
        if refresh:
            self._refresh_per_layer_allocator_stats()

    def _reuse_evicted_per_layer_locs(
        self, layer_id: int, locs: List[int]
    ) -> Optional[List[int]]:
        if not locs:
            return []
        layer_id = int(layer_id)
        locs = [int(loc) for loc in locs if int(loc) > 0]
        if not locs:
            return None
        allocated = self._per_layer_arena_allocated_locs.setdefault(layer_id, set())
        if any(int(loc) in allocated for loc in locs):
            return None
        free = self._per_layer_arena_free_locs.setdefault(layer_id, [])
        loc_set = set(locs)
        if loc_set:
            self._per_layer_arena_free_locs[layer_id] = [
                int(loc) for loc in free if int(loc) not in loc_set
            ]
            self._per_layer_arena_common_free_locs.difference_update(loc_set)
            self._per_layer_arena_protected_locs.setdefault(layer_id, set()).difference_update(
                loc_set
            )
            allocated.update(loc_set)
            self.stats.kvc_per_layer_physical_arena_alloc_count += len(loc_set)
            self._refresh_per_layer_allocator_stats()
        return list(locs)

    def _mark_req_layerkv_owned(
        self, req_idx: int, keys: List[Tuple[int, int, int]]
    ) -> None:
        req_idx = int(req_idx)
        self._per_layer_owned_req_indices.add(req_idx)
        by_req = self._per_layer_owned_keys_by_req.setdefault(req_idx, set())
        by_req.update((int(a), int(b), int(c)) for a, b, c in keys)
        self.stats.kvc_per_layer_logical_request_count = len(
            self._per_layer_owned_req_indices
        )

    def _translate_per_layer_locs(
        self, layer_id: int, loc: Any
    ) -> Any:
        if self.config.kvc_backend != "per-layer-arena":
            return loc
        mapping = self._per_layer_canonical_to_physical.get(int(layer_id))
        if not mapping:
            return loc
        if not isinstance(loc, torch.Tensor) or int(loc.numel()) == 0:
            return loc
        try:
            loc_list = [int(x) for x in loc.detach().cpu().tolist()]
        except Exception:
            return loc
        translated = [int(mapping.get(int(x), int(x))) for x in loc_list]
        size = int(getattr(self._kv_pool, "size", 0) or 0)
        if size > 0 and any(x <= 0 or x > size for x in translated):
            raise RuntimeError(
                f"LayerKV translated KV loc out of range for layer {layer_id}: "
                f"size={size} locs={translated[:8]}"
            )
        if len(set(translated)) != len(translated):
            raise RuntimeError(
                f"LayerKV translated KV locs contain duplicates for layer {layer_id}: "
                f"locs={translated[:16]}"
            )
        if translated == loc_list:
            return loc
        return torch.tensor(translated, dtype=loc.dtype, device=loc.device)

    def _wrap_set_kv_buffer(self, orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapped(layer: Any, loc: Any, cache_k: Any, cache_v: Any, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                layer_id = int(getattr(layer, "layer_id"))
                loc = self._translate_per_layer_locs(layer_id, loc)
            except RuntimeError:
                raise
            except Exception:
                pass
            ret = orig(layer, loc, cache_k, cache_v, *args, **kwargs)
            self.stats.kvc_set_kv_count += 1
            try:
                self.stats.kvc_tokens_written += int(loc.numel())
            except Exception:
                pass
            try:
                self.stats.kvc_bytes_written += int(cache_k.nbytes + cache_v.nbytes)
            except Exception:
                pass
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.stats.layerkv_python_overhead_ms += elapsed_ms
            self._add_profile("profile_set_kv_ms", elapsed_ms)
            return ret

        return wrapped

    def _wrap_get_key_buffer(self, orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapped(layer_id: int, *args, **kwargs):
            self.stats.kvc_get_key_count += 1
            self._prepare_per_layer_kvc_attention(layer_id)
            self._wait_for_kvc_layer_ready(layer_id)
            return orig(layer_id, *args, **kwargs)

        return wrapped

    def _wrap_get_value_buffer(self, orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapped(layer_id: int, *args, **kwargs):
            self.stats.kvc_get_value_count += 1
            self._prepare_per_layer_kvc_attention(layer_id)
            self._wait_for_kvc_layer_ready(layer_id)
            return orig(layer_id, *args, **kwargs)

        return wrapped

    def _wrap_get_kv_buffer(self, orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapped(layer_id: int, *args, **kwargs):
            self.stats.kvc_get_kv_count += 1
            self._prepare_per_layer_kvc_attention(layer_id)
            self._wait_for_kvc_layer_ready(layer_id)
            return orig(layer_id, *args, **kwargs)

        return wrapped

    def _prepare_per_layer_kvc_attention(self, layer_id: int) -> None:
        if not self._uses_per_layer_attention_override():
            return
        if self.config.kvc_backend == "virtual-arena":
            self._prepare_virtual_kvc_attention(layer_id)
            return
        if (
            self.config.kvc_backend == "per-layer-arena"
            and not self._per_layer_req_to_token_owned
            and not self._has_offloaded_kvc_entries()
        ):
            return
        if not self._per_layer_residency and not self._per_layer_req_to_token_owned:
            return
        if self._last_forward_batch is None or self._runner is None:
            return
        layer_id = int(layer_id)
        key = (int(self._decode_step), layer_id)
        if key in self._per_layer_kvc_prepared_layers:
            self.stats.kvc_per_layer_metadata_rewrite_skip_count += 1
            return
        self._per_layer_kvc_prepared_layers.add(key)
        self.stats.kvc_per_layer_override_layer_count = len(
            self._per_layer_req_to_token_overrides
        )
        override = self._per_layer_req_to_token_overrides.get(layer_id)
        if override is None:
            # The hook is active even before the arena allocator produces a
            # layer-specific mapping. Keep this cheap and explicit.
            self.stats.kvc_per_layer_metadata_rewrite_skip_count += 1
            return
        try:
            table = getattr(self._req_to_token_pool, "req_to_token", None)
            if table is not None and int(override.data_ptr()) == int(table.data_ptr()):
                metadata_key = (
                    "canonical",
                    int(table.data_ptr()),
                    tuple(self._current_forward_req_lens),
                    int(self._decode_step),
                )
            else:
                metadata_key = (
                    "layer",
                    int(layer_id),
                    int(override.data_ptr()),
                    int(self._per_layer_req_to_token_versions.get(layer_id, 0)),
                    tuple(self._current_forward_req_lens),
                    int(self._decode_step),
                )
            if self._per_layer_kvc_current_metadata_key == metadata_key:
                self.stats.kvc_metadata_dirty_guard_skip_count += 1
                self.stats.kvc_per_layer_metadata_rewrite_skip_count += 1
                return
        except Exception:
            metadata_key = None
        ok = self._rewrite_attention_metadata_for_layer(layer_id, override)
        if ok:
            if metadata_key is not None:
                self._per_layer_kvc_current_metadata_key = metadata_key
            self.stats.kvc_per_layer_metadata_rewrite_count += 1
        else:
            self.stats.kvc_per_layer_metadata_rewrite_unsupported_count += 1
        if (
            self.config.kvc_backend == "per-layer-arena"
            and self._has_offloaded_kvc_entries()
        ):
            if self._per_layer_virtual_scratch_enabled():
                self._prepare_virtual_kvc_attention(layer_id, guard=False)
                return
            selected = [
                entry
                for entry in self._select_required_offloaded_per_layer_entries(
                    self._last_forward_batch
                )
                if int(entry.layer_id) == int(layer_id)
            ]
            if selected:
                with self._profile("profile_kvc_reload_required_ms"):
                    self._reload_required_kvc(
                        self._last_forward_batch, selected_entries=selected
                    )

    def _prepare_virtual_kvc_attention(self, layer_id: int, *, guard: bool = True) -> None:
        if not self._residency and not (
            self.config.kvc_backend == "per-layer-arena" and self._per_layer_residency
        ):
            return
        if self._last_forward_batch is None or self._runner is None:
            return
        if not self._ensure_virtual_scratch():
            return
        layer_id = int(layer_id)
        key = (int(self._decode_step), layer_id)
        if guard and key in self._per_layer_kvc_prepared_layers:
            self.stats.kvc_per_layer_metadata_rewrite_skip_count += 1
            return
        if guard:
            self._per_layer_kvc_prepared_layers.add(key)
        demand = self._get_virtual_kvc_demand(layer_id)
        if demand is None or not demand.entries:
            return
        if (
            self.config.kvc_backend == "per-layer-arena"
            and self._virtual_scratch_locs is not None
            and int(demand.token_count) > int(self._virtual_scratch_locs.numel())
        ):
            capacity = int(self._virtual_scratch_locs.numel())
            overflow_entries: List[_LayerKVResidencyEntry] = []
            kept_tokens = 0
            for entry in demand.entries:
                entry_tokens = int(entry.token_count)
                if kept_tokens + entry_tokens <= capacity:
                    kept_tokens += entry_tokens
                else:
                    overflow_entries.append(entry)
            if overflow_entries:
                with self._profile("profile_kvc_reload_required_ms"):
                    self._reload_required_kvc(
                        self._last_forward_batch,
                        selected_entries=overflow_entries,
                    )
                self._virtual_materialize_plans_by_layer.pop(layer_id, None)
                self._virtual_kvc_demands.pop((int(self._decode_step), layer_id), None)
                self._invalidate_virtual_layer_cache(layer_id)
                demand = self._get_virtual_kvc_demand(layer_id)
                if demand is None or not demand.entries:
                    return
            if int(demand.token_count) > int(self._virtual_scratch_locs.numel()):
                self.stats.virtual_kvc_scratch_overflow_count += 1
                self.stats.kvc_guard_pass = False
                self.stats.kvc_guard_reason = "virtual_scratch_capacity_exceeded"
                return
        pending_key = (int(self._decode_step), layer_id, demand.signature)
        cached = self._lookup_virtual_scratch_cache(demand)
        if cached is not None:
            scratch_locs = cached.scratch_locs
            buffer_idx = int(cached.buffer_idx)
        else:
            pending = self._pending_virtual_kvc_materialize.pop(pending_key, None)
            if pending is not None:
                scratch_locs = self._wait_for_virtual_materialize(pending)
                if scratch_locs is None:
                    return
                buffer_idx = int(pending.buffer_idx)
                self._store_virtual_scratch_cache(demand, scratch_locs, buffer_idx)
            else:
                buffer_idx = self._choose_virtual_scratch_buffer(
                    demand.token_count, exclude_buffer_idx=None
                )
                if buffer_idx < 0:
                    scratch_locs = self._materialize_virtual_kvc_sync(demand)
                    self.stats.virtual_kvc_prefetch_fallback_sync_count += 1
                    if scratch_locs is None:
                        return
                    buffer_idx = -1
                else:
                    scratch_locs = self._materialize_virtual_kvc_sync(
                        demand,
                        buffer_idx=buffer_idx,
                    )
                    if scratch_locs is None:
                        return
                self._store_virtual_scratch_cache(demand, scratch_locs, buffer_idx)
        self._rewrite_virtual_kvc_attention(
            layer_id,
            scratch_locs,
            demand,
        )
        self._issue_next_virtual_kvc_prefetch(
            layer_id,
            demand,
            buffer_idx,
        )

    def _lookup_virtual_scratch_cache(
        self, demand: _LayerKVKvcDemand, *, record_stats: bool = True
    ) -> Optional[_LayerKVVirtualScratchCacheEntry]:
        entry = self._virtual_scratch_cache_by_layer.get(int(demand.layer_id))
        if (
            entry is None
            or int(entry.token_count) != int(demand.token_count)
            or (
                entry.host_signature != demand.host_signature
                if demand.host_signature != (0, 0, 0, 0)
                else entry.host_slots != demand.host_slots
            )
            or (
                demand.host_slice[0] >= 0
                and entry.host_slice != demand.host_slice
            )
            or (
                demand.host_slots == ()
                and demand.host_slice[0] < 0
                and demand.host_index_cpu is not None
                and (
                    entry.host_index_cpu is None
                    or not bool(torch.equal(entry.host_index_cpu, demand.host_index_cpu))
                )
            )
            or entry.scratch_locs is None
            or int(entry.scratch_locs.numel()) != int(demand.token_count)
        ):
            if record_stats:
                self.stats.virtual_kvc_persistent_cache_miss_count += 1
            return None
        if record_stats:
            self.stats.virtual_kvc_persistent_cache_hit_count += 1
            entry.last_step = int(self._decode_step)
        return entry

    def _store_virtual_scratch_cache(
        self,
        demand: _LayerKVKvcDemand,
        scratch_locs: torch.Tensor,
        buffer_idx: int,
    ) -> None:
        previous = self._virtual_scratch_cache_by_layer.get(int(demand.layer_id))
        previous_same = False
        if previous is not None:
            previous_same = (
                previous.host_signature == demand.host_signature
                if demand.host_signature != (0, 0, 0, 0)
                else previous.host_slots == demand.host_slots
            )
            if (
                previous_same
                and demand.host_slice[0] >= 0
            ):
                previous_same = previous.host_slice == demand.host_slice
            if (
                previous_same
                and demand.host_slots == ()
                and demand.host_slice[0] < 0
                and demand.host_index_cpu is not None
            ):
                previous_same = previous.host_index_cpu is not None and bool(
                    torch.equal(previous.host_index_cpu, demand.host_index_cpu)
                )
        if previous is not None and not previous_same:
            self.stats.virtual_kvc_persistent_cache_invalidate_count += 1
        self._virtual_scratch_cache_by_layer[int(demand.layer_id)] = (
            _LayerKVVirtualScratchCacheEntry(
                layer_id=int(demand.layer_id),
                host_slots=demand.host_slots,
                host_signature=demand.host_signature,
                host_slice=demand.host_slice,
                host_index_cpu=demand.host_index_cpu,
                token_count=int(demand.token_count),
                scratch_locs=scratch_locs,
                buffer_idx=int(buffer_idx),
                last_step=int(self._decode_step),
            )
        )
        self.stats.virtual_kvc_persistent_cache_store_count += 1

    def _invalidate_virtual_layer_cache(self, layer_id: int) -> None:
        if int(layer_id) in self._virtual_scratch_cache_by_layer:
            self._virtual_scratch_cache_by_layer.pop(int(layer_id), None)
            self.stats.virtual_kvc_persistent_cache_invalidate_count += 1

    def _invalidate_virtual_caches(self) -> None:
        if self._virtual_scratch_cache_by_layer:
            self.stats.virtual_kvc_persistent_cache_invalidate_count += len(
                self._virtual_scratch_cache_by_layer
            )
            self._virtual_scratch_cache_by_layer.clear()
        self._virtual_materialize_plan = None
        self._virtual_materialize_plans_by_layer.clear()
        self._virtual_kvc_demands.clear()
        self._metadata_patch_cache.clear()
        self._metadata_patch_tensor_cache.clear()
        self._pending_virtual_kvc_materialize.clear()

    def _invalidate_virtual_caches_for_layers(self, layer_ids: Set[int]) -> None:
        if not layer_ids:
            return
        if any(int(layer_id) < 0 for layer_id in layer_ids):
            self._invalidate_virtual_caches()
            return
        for layer_id in {int(layer_id) for layer_id in layer_ids}:
            self._invalidate_virtual_layer_cache(layer_id)
            self._virtual_materialize_plans_by_layer.pop(layer_id, None)
            for key in list(self._virtual_kvc_demands):
                if int(key[1]) == layer_id:
                    self._virtual_kvc_demands.pop(key, None)
            for key in list(self._pending_virtual_kvc_materialize):
                if len(key) >= 2 and int(key[1]) == layer_id:
                    self._pending_virtual_kvc_materialize.pop(key, None)
            for key in list(self._metadata_patch_cache):
                if key and int(key[0]) == layer_id:
                    self._metadata_patch_cache.pop(key, None)
        # Shared tensor entries intentionally have no layer in the key.  Drop
        # them when any layer is invalidated so exact-match reuse remains safe.
        self._metadata_patch_tensor_cache.clear()
        self._kvc_demand_signature_eval_step = None

    def _virtual_cache_layers_for_entries(
        self, entries: List[_LayerKVResidencyEntry]
    ) -> Set[int]:
        layers = {int(entry.layer_id) for entry in entries if entry is not None}
        if any(layer_id < 0 for layer_id in layers):
            return set(int(layer_id) for layer_id in self._kvc_layer_ids()) or {-1}
        return layers

    def _get_virtual_kvc_demand(self, layer_id: int) -> Optional[_LayerKVKvcDemand]:
        layer_id = int(layer_id)
        key = (int(self._decode_step), layer_id)
        cached = self._virtual_kvc_demands.get(key)
        if cached is not None:
            return cached
        plan = self._get_virtual_materialize_plan_for_layer(layer_id)
        if plan is None or not plan.selected:
            return None
        signature = self._kvc_demand_signature(layer_id, plan)
        base_signature = self._kvc_demand_signature_without_layer(signature)
        demand = _LayerKVKvcDemand(
            layer_id=layer_id,
            entries=tuple(plan.selected),
            req_indices=plan.req_indices,
            positions=plan.positions,
            row_indices=plan.row_indices,
            flat_indices=plan.flat_indices,
            flat_spans=plan.flat_spans,
            row_spans=plan.row_spans,
            token_count=int(plan.token_count),
            deadline_layer=layer_id,
            benefit_score=float(plan.token_count),
            signature=signature,
            base_signature=base_signature,
            backend_semantics=str(self.stats.layerkv_kvc_backend_semantics or ""),
            host_slots=plan.host_slots,
            host_signature=plan.host_signature,
            host_slice=plan.host_slice,
            host_index_cpu=plan.host_index_cpu,
            max_row_index=plan.max_row_index,
            max_position=plan.max_position,
            max_flat_index=plan.max_flat_index,
        )
        self._virtual_kvc_demands[key] = demand
        self.stats.kvc_demand_layer_count += 1
        self.stats.kvc_layerwise_demand_count += 1
        self.stats.kvc_layerwise_demand_token_count += int(demand.token_count)
        layer_count = max(1, len(self._kvc_layer_ids()))
        if len(
            self._virtual_kvc_demands
        ) >= layer_count and self._kvc_demand_signature_eval_step != int(
            self._decode_step
        ):
            base_signatures = {
                self._kvc_demand_base_signature(item)
                for item in self._virtual_kvc_demands.values()
            }
            if len(base_signatures) == 1:
                self.stats.kvc_demand_shared_signature_count += 1
            else:
                self.stats.kvc_demand_signature_mismatch_count += 1
            self._kvc_demand_signature_eval_step = int(self._decode_step)
        return demand

    def _kvc_demand_base_signature(self, demand: _LayerKVKvcDemand) -> str:
        if demand.base_signature:
            return demand.base_signature
        return self._kvc_demand_signature_without_layer(demand.signature)

    def _kvc_demand_signature_without_layer(self, signature: str) -> str:
        return "|".join(str(signature).split("|")[1:])

    def _kvc_demand_signature(
        self, layer_id: int, plan: _LayerKVVirtualMaterializePlan
    ) -> str:
        if not plan.selected:
            return f"L{int(layer_id)}|empty|step{int(self._decode_step)}"
        first = plan.selected[0]
        last = plan.selected[-1]
        return (
            f"L{int(layer_id)}|step{int(self._decode_step)}"
            f"|n{len(plan.selected)}|tok{int(plan.token_count)}"
            f"|first{int(first.req_idx)}:{int(first.pos)}:{int(first.token_count)}"
            f"|last{int(last.req_idx)}:{int(last.pos)}:{int(last.token_count)}"
            f"|host{int(plan.host_signature[0])}"
        )

    def _get_virtual_materialize_plan_for_layer(
        self, layer_id: int
    ) -> Optional[_LayerKVVirtualMaterializePlan]:
        layer_id = int(layer_id)
        cached = self._virtual_materialize_plans_by_layer.get(layer_id)
        if cached is not None and int(cached.step) == int(self._decode_step):
            return cached
        plan = self._build_virtual_materialize_plan(layer_id=layer_id)
        if plan is not None:
            self._virtual_materialize_plans_by_layer[layer_id] = plan
        return plan

    def _get_virtual_materialize_plan(
        self,
    ) -> Optional[_LayerKVVirtualMaterializePlan]:
        if self._virtual_materialize_plan is not None and int(
            self._virtual_materialize_plan.step
        ) == int(self._decode_step):
            return self._virtual_materialize_plan
        self.stats.kvc_layerwise_selector_fallback_count += 1
        self._virtual_materialize_plan = self._build_virtual_materialize_plan(
            layer_id=None
        )
        return self._virtual_materialize_plan

    def _build_virtual_materialize_plan(
        self, *, layer_id: Optional[int]
    ) -> Optional[_LayerKVVirtualMaterializePlan]:
        table = getattr(self._req_to_token_pool, "req_to_token", None)
        if table is None:
            return None
        target_layer = None if layer_id is None else int(layer_id)
        direct_kv_indices_only = target_layer is not None
        with self._profile("profile_virtual_select_ms"):
            batch_req_lens = self._batch_req_indices_and_lens(self._last_forward_batch)
            active_lens = {
                int(req_idx): self._align_tokens_down(max(0, int(seq_len) - 1))
                for req_idx, seq_len in batch_req_lens
            }
            row_by_req: Dict[int, int] = {}
            flat_base_by_req: Dict[int, int] = {}
            flat_base = 0
            for row, (req_idx, seq_len) in enumerate(batch_req_lens):
                req_idx = int(req_idx)
                row_by_req[req_idx] = int(row)
                flat_base_by_req[req_idx] = int(flat_base)
                flat_base += max(0, int(seq_len))
            if self.config.kvc_backend == "per-layer-arena" and target_layer is not None:
                selected = []
                stale: List[Tuple[int, int, int]] = []
                scanned = 0
                for req_idx, required_prefix_len in active_lens.items():
                    sorted_keys, positions = (
                        self._sorted_per_layer_offloaded_keys_for_req_layer(
                            int(req_idx), int(target_layer)
                        )
                    )
                    if not sorted_keys:
                        continue
                    self.stats.kvc_layerwise_required_index_hit_count += 1
                    limit = bisect.bisect_right(
                        positions, max(0, int(required_prefix_len) - 1)
                    )
                    if limit <= 0:
                        continue
                    for key in sorted_keys[:limit]:
                        scanned += 1
                        entry = self._per_layer_residency.get(key)
                        if entry is None or entry.state != "offloaded":
                            stale.append(key)
                            continue
                        if int(entry.pos) + int(entry.token_count) > int(
                            required_prefix_len
                        ):
                            continue
                        selected.append(entry)
                self.stats.kvc_layerwise_required_index_scan_count += scanned
                self.stats.kvc_required_scanned_keys += scanned
                if stale:
                    for key in stale:
                        self._untrack_per_layer_offloaded_key(key)
                    self.stats.kvc_layerwise_required_index_stale_count += len(stale)
            else:
                source_entries = (
                    self._per_layer_residency.values()
                    if self.config.kvc_backend == "per-layer-arena"
                    else self._residency.values()
                )
                selected = [
                    entry
                    for entry in source_entries
                    if entry.state == "offloaded"
                    and (
                        target_layer is None
                        or int(entry.layer_id) < 0
                        or int(entry.layer_id) == target_layer
                    )
                    and int(entry.req_idx) in active_lens
                    and int(entry.pos) + int(entry.token_count)
                    <= active_lens[int(entry.req_idx)]
                ]
        if not selected:
            return _LayerKVVirtualMaterializePlan(
                step=int(self._decode_step),
                selected=[],
                req_indices=(),
                positions=(),
                row_indices=(),
                flat_indices=(),
                flat_spans=(),
                row_spans=(),
                req_tensor=None,
                pos_tensor=None,
                row_tensor=None,
                flat_tensor=None,
                host_slots=(),
                host_signature=(0, 0, 0, 0),
                host_slice=(-1, 0),
                host_index_cpu=None,
                max_row_index=-1,
                max_position=-1,
                max_flat_index=-1,
                token_count=0,
            )
        with self._profile("profile_virtual_index_build_ms"):
            selected.sort(key=lambda entry: (entry.req_idx, entry.pos))
            token_count = sum(entry.token_count for entry in selected)
            req_indices: List[int] = []
            positions: List[int] = []
            row_indices: List[int] = []
            flat_indices: List[int] = []
            flat_spans: List[Tuple[int, int, int]] = []
            row_spans: List[Tuple[int, int, int, int]] = []
            span_start = -1
            span_scratch_start = 0
            span_len = 0
            row_span_row = -1
            row_span_start = -1
            row_span_scratch_start = 0
            row_span_len = 0
            scratch_offset = 0
            for entry in selected:
                req_idx = int(entry.req_idx)
                logical_positions = entry.logical_positions()
                base = int(flat_base_by_req[req_idx])
                row = int(row_by_req[req_idx])
                for pos in logical_positions:
                    pos = int(pos)
                    flat_index = base + int(pos)
                    if direct_kv_indices_only:
                        if span_len > 0 and flat_index == span_start + span_len:
                            span_len += 1
                        else:
                            if span_len > 0:
                                flat_spans.append(
                                    (span_start, span_scratch_start, span_len)
                                )
                            span_start = int(flat_index)
                            span_scratch_start = int(scratch_offset)
                            span_len = 1
                        if (
                            row_span_len > 0
                            and row == row_span_row
                            and pos == row_span_start + row_span_len
                        ):
                            row_span_len += 1
                        else:
                            if row_span_len > 0:
                                row_spans.append(
                                    (
                                        row_span_row,
                                        row_span_start,
                                        row_span_scratch_start,
                                        row_span_len,
                                    )
                                )
                            row_span_row = int(row)
                            row_span_start = int(pos)
                            row_span_scratch_start = int(scratch_offset)
                            row_span_len = 1
                        scratch_offset += 1
                    else:
                        flat_indices.append(flat_index)
                if not direct_kv_indices_only:
                    req_indices.extend([req_idx] * entry.token_count)
                    positions.extend(logical_positions)
                    row_indices.extend([row] * entry.token_count)
            if direct_kv_indices_only and span_len > 0:
                flat_spans.append((span_start, span_scratch_start, span_len))
            if direct_kv_indices_only and row_span_len > 0:
                row_spans.append(
                    (
                        row_span_row,
                        row_span_start,
                        row_span_scratch_start,
                        row_span_len,
                    )
                )
            host_slots: List[int] = []
            for entry in selected:
                host_slots.extend(entry.host_slot_list())
            host_signature = self._host_slot_signature(host_slots)
            host_slice = self._contiguous_host_slice(host_slots)
            host_index_cpu = (
                None
                if host_slice[0] >= 0
                else torch.tensor(host_slots, dtype=torch.int64, device="cpu")
            )
            max_row_index = max(row_indices) if row_indices else -1
            max_position = max(positions) if positions else -1
            if row_spans:
                max_row_index = max(max_row_index, max(row for row, _start, _offset, _length in row_spans))
                max_position = max(
                    max_position,
                    max(start + length - 1 for _row, start, _offset, length in row_spans),
                )
            if flat_indices:
                max_flat_index = max(flat_indices)
            elif flat_spans:
                max_flat_index = max(start + length - 1 for start, _offset, length in flat_spans)
            else:
                max_flat_index = -1
        return _LayerKVVirtualMaterializePlan(
            step=int(self._decode_step),
            selected=selected,
            req_indices=tuple(int(x) for x in req_indices),
            positions=tuple(int(x) for x in positions),
            row_indices=tuple(int(x) for x in row_indices),
            flat_indices=tuple(int(x) for x in flat_indices),
            flat_spans=tuple(
                (int(start), int(offset), int(length))
                for start, offset, length in flat_spans
            ),
            row_spans=tuple(
                (int(row), int(start), int(offset), int(length))
                for row, start, offset, length in row_spans
            ),
            req_tensor=None,
            pos_tensor=None,
            row_tensor=None,
            flat_tensor=None,
            host_slots=() if direct_kv_indices_only else tuple(int(x) for x in host_slots),
            host_signature=host_signature,
            host_slice=host_slice,
            host_index_cpu=host_index_cpu,
            max_row_index=int(max_row_index),
            max_position=int(max_position),
            max_flat_index=int(max_flat_index),
            token_count=int(token_count),
        )

    def _virtual_metadata_uses_kv_indices(self) -> bool:
        forward_batch = self._last_forward_batch
        attn_backend = getattr(forward_batch, "attn_backend", None) or getattr(
            self._runner, "attn_backend", None
        )
        metadata = getattr(attn_backend, "forward_metadata", None)
        if metadata is None:
            return False
        return getattr(metadata, "kv_indices", None) is not None

    @staticmethod
    def _host_slot_signature(host_slots: List[int]) -> Tuple[int, int, int, int]:
        if not host_slots:
            return (0, 0, 0, 0)
        checksum = 0
        for slot in host_slots:
            checksum = (checksum + int(slot)) & 0x7FFFFFFF
        return (
            len(host_slots),
            int(host_slots[0]),
            int(host_slots[-1]),
            int(checksum),
        )

    @staticmethod
    def _contiguous_host_slice(host_slots: List[int]) -> Tuple[int, int]:
        if not host_slots:
            return (-1, 0)
        first = int(host_slots[0])
        for offset, slot in enumerate(host_slots):
            if int(slot) != first + int(offset):
                return (-1, 0)
        return (first, len(host_slots))

    def _build_virtual_scatter_indices(
        self, entries: Tuple[_LayerKVResidencyEntry, ...]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if not entries:
            return (), ()
        req_indices: List[int] = []
        positions: List[int] = []
        for entry in entries:
            req_idx = int(entry.req_idx)
            logical_positions = entry.logical_positions()
            req_indices.extend([req_idx] * int(entry.token_count))
            positions.extend(logical_positions)
        return tuple(int(x) for x in req_indices), tuple(int(x) for x in positions)

    def _choose_virtual_scratch_buffer(
        self, token_count: int, *, exclude_buffer_idx: Optional[int]
    ) -> int:
        if len(self._virtual_scratch_buffers) < 2:
            return -1
        for idx, locs in enumerate(self._virtual_scratch_buffers):
            if exclude_buffer_idx is not None and idx == int(exclude_buffer_idx):
                continue
            if int(locs.numel()) >= int(token_count):
                return idx
        return -1

    def _materialize_virtual_kvc_sync(
        self,
        demand: _LayerKVKvcDemand,
        *,
        buffer_idx: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        token_count = int(demand.token_count)
        if self._virtual_scratch_locs is None or token_count > int(
            self._virtual_scratch_locs.numel()
        ):
            self.stats.virtual_kvc_scratch_overflow_count += 1
            self.stats.kvc_guard_pass = False
            self.stats.kvc_guard_reason = "virtual_scratch_capacity_exceeded"
            return None
        if buffer_idx is None:
            scratch_locs = self._virtual_scratch_locs[:token_count]
        else:
            scratch_locs = self._virtual_scratch_buffers[int(buffer_idx)][:token_count]
        self._ensure_host_store()
        elapsed_ms, _start_event, _ready_event = self._host_store.reload_layer_to_locs(
            int(demand.layer_id),
            list(demand.entries),
            scratch_locs,
            host_index=demand.host_index_cpu,
            host_slice=demand.host_slice,
        )
        self._record_virtual_materialize(token_count, elapsed_ms)
        return scratch_locs

    def _record_virtual_materialize(self, token_count: int, elapsed_ms: float) -> None:
        self.stats.virtual_kvc_materialize_count += 1
        self.stats.virtual_kvc_materialize_layer_count += 1
        self.stats.virtual_kvc_materialize_token_count += int(token_count)
        self.stats.virtual_kvc_materialize_ms += elapsed_ms
        self.stats.virtual_scratch_used_tokens = max(
            self.stats.virtual_scratch_used_tokens, int(token_count)
        )
        self.stats.layerkv_copy_stream_busy_ms += elapsed_ms

    def _rewrite_virtual_kvc_attention(
        self,
        layer_id: int,
        scratch_locs: torch.Tensor,
        demand: _LayerKVKvcDemand,
    ) -> None:
        with self._profile("profile_virtual_direct_metadata_patch_ms"):
            direct_ok = self._patch_virtual_attention_metadata(
                layer_id, scratch_locs, demand
            )
        if direct_ok:
            self.stats.virtual_kvc_direct_metadata_patch_count += 1
            self.stats.kvc_per_layer_slot_override_count += 1
            self.stats.kvc_per_layer_slot_override_token_count += int(
                demand.token_count
            )
            self.stats.kvc_per_layer_metadata_rewrite_count += 1
            return
        self.stats.virtual_kvc_direct_metadata_patch_fallback_count += 1
        self.stats.metadata_patch_layer_fallback_count += 1
        table = getattr(self._req_to_token_pool, "req_to_token", None)
        if table is None:
            return
        req_indices = demand.req_indices
        positions = demand.positions
        if not req_indices or not positions:
            req_indices, positions = self._build_virtual_scatter_indices(
                demand.entries
            )
            if not req_indices or not positions:
                self.stats.kvc_per_layer_metadata_rewrite_unsupported_count += 1
                return
        with self._profile("profile_virtual_req_to_token_scatter_ms"):
            req_tensor = torch.tensor(
                req_indices, dtype=torch.int64, device=table.device
            )
            pos_tensor = torch.tensor(
                positions, dtype=torch.int64, device=table.device
            )
            table[req_tensor, pos_tensor] = scratch_locs.to(
                dtype=table.dtype, device=table.device
            )
        self.stats.kvc_per_layer_slot_override_count += 1
        self.stats.kvc_per_layer_slot_override_token_count += int(demand.token_count)
        with self._profile("profile_virtual_metadata_rewrite_ms"):
            ok = self._rewrite_attention_metadata_for_layer(layer_id, table)
        if ok:
            self.stats.kvc_per_layer_metadata_rewrite_count += 1
        else:
            self.stats.kvc_per_layer_metadata_rewrite_unsupported_count += 1

    def _patch_virtual_attention_metadata(
        self,
        layer_id: int,
        scratch_locs: torch.Tensor,
        demand: _LayerKVKvcDemand,
    ) -> bool:
        if demand.token_count <= 0:
            return False
        forward_batch = self._last_forward_batch
        attn_backend = getattr(forward_batch, "attn_backend", None) or getattr(
            self._runner, "attn_backend", None
        )
        if attn_backend is None:
            return False
        metadata = getattr(attn_backend, "forward_metadata", None)
        if metadata is None:
            return False
        try:
            cache_entry = self._metadata_patch_cache_entry(
                int(layer_id), demand, metadata, scratch_locs
            )
            kv_indices = getattr(metadata, "kv_indices", None)
            kv_indptr = getattr(metadata, "kv_indptr", None)
            if kv_indices is not None and kv_indptr is not None:
                if kv_indices.dim() != 1 or not kv_indices.is_contiguous():
                    return False
                if not demand.flat_indices and not demand.flat_spans:
                    return False
                if int(demand.max_flat_index) >= int(kv_indices.numel()):
                    return False
                if cache_entry.scratch_tensor is None or (
                    cache_entry.scratch_tensor.device != kv_indices.device
                    or cache_entry.scratch_tensor.dtype != kv_indices.dtype
                ):
                    cache_entry.scratch_tensor = scratch_locs.to(
                        device=kv_indices.device, dtype=kv_indices.dtype
                    )
                if demand.flat_spans:
                    for start, scratch_start, length in demand.flat_spans:
                        start = int(start)
                        scratch_start = int(scratch_start)
                        length = int(length)
                        if (
                            length <= 0
                            or start + length > int(kv_indices.numel())
                            or scratch_start + length
                            > int(cache_entry.scratch_tensor.numel())
                        ):
                            return False
                        kv_indices[start : start + length] = cache_entry.scratch_tensor[
                            scratch_start : scratch_start + length
                        ]
                    self.stats.metadata_patch_slice_count += len(demand.flat_spans)
                    self.stats.metadata_patch_slice_token_count += int(
                        demand.token_count
                    )
                    return True
                if not cache_entry.flat_slice_checked:
                    cache_entry.flat_slice = self._contiguous_index_slice(
                        demand.flat_indices
                    )
                    cache_entry.flat_slice_checked = True
                if cache_entry.flat_slice is not None:
                    start, end = cache_entry.flat_slice
                    if end <= int(kv_indices.numel()) and (end - start) == int(
                        demand.token_count
                    ):
                        kv_indices[start:end] = cache_entry.scratch_tensor
                        self.stats.metadata_patch_slice_count += 1
                        self.stats.metadata_patch_slice_token_count += int(
                            demand.token_count
                        )
                        return True
                if cache_entry.flat_tensor is None or (
                    cache_entry.flat_tensor.device != kv_indices.device
                ):
                    cache_entry.flat_tensor = torch.tensor(
                        demand.flat_indices,
                        dtype=torch.int64,
                        device=kv_indices.device,
                    )
                kv_indices[cache_entry.flat_tensor] = cache_entry.scratch_tensor
                return True

            page_table = getattr(metadata, "page_table", None)
            if page_table is not None:
                if int(self._page_size) != 1 or page_table.dim() < 2:
                    return False
                if not demand.row_indices and not demand.row_spans:
                    return False
                if int(demand.max_row_index) >= int(page_table.shape[0]):
                    return False
                if int(demand.max_position) >= int(page_table.shape[1]):
                    return False
                if cache_entry.scratch_tensor is None or (
                    cache_entry.scratch_tensor.device != page_table.device
                    or cache_entry.scratch_tensor.dtype != page_table.dtype
                ):
                    cache_entry.scratch_tensor = scratch_locs.to(
                        device=page_table.device, dtype=page_table.dtype
                    )
                if demand.row_spans:
                    for row, start, scratch_start, length in demand.row_spans:
                        row = int(row)
                        start = int(start)
                        scratch_start = int(scratch_start)
                        length = int(length)
                        if (
                            length <= 0
                            or row >= int(page_table.shape[0])
                            or start + length > int(page_table.shape[1])
                            or scratch_start + length
                            > int(cache_entry.scratch_tensor.numel())
                        ):
                            return False
                        page_table[row, start : start + length] = (
                            cache_entry.scratch_tensor[
                                scratch_start : scratch_start + length
                            ]
                        )
                    self.stats.metadata_patch_slice_count += len(demand.row_spans)
                    self.stats.metadata_patch_slice_token_count += int(
                        demand.token_count
                    )
                    return True
                if not cache_entry.page_slice_checked:
                    cache_entry.page_slice = self._page_table_index_slice(
                        demand.row_indices, demand.positions
                    )
                    cache_entry.page_slice_checked = True
                if cache_entry.page_slice is not None:
                    row, start, end = cache_entry.page_slice
                    if (
                        row < int(page_table.shape[0])
                        and end <= int(page_table.shape[1])
                        and (end - start) == int(demand.token_count)
                    ):
                        page_table[row, start:end] = cache_entry.scratch_tensor
                        self.stats.metadata_patch_slice_count += 1
                        self.stats.metadata_patch_slice_token_count += int(
                            demand.token_count
                        )
                        return True
                if cache_entry.row_tensor is None or (
                    cache_entry.row_tensor.device != page_table.device
                ):
                    cache_entry.row_tensor = torch.tensor(
                        demand.row_indices,
                        dtype=torch.int64,
                        device=page_table.device,
                    )
                if cache_entry.pos_tensor is None or (
                    cache_entry.pos_tensor.device != page_table.device
                ):
                    cache_entry.pos_tensor = torch.tensor(
                        demand.positions,
                        dtype=torch.int64,
                        device=page_table.device,
                    )
                page_table[cache_entry.row_tensor, cache_entry.pos_tensor] = (
                    cache_entry.scratch_tensor
                )
                return True
        except Exception:
            self.stats.metadata_patch_cache_fallback_count += 1
            return False
        return False

    @staticmethod
    def _contiguous_index_slice(indices: Tuple[int, ...]) -> Optional[Tuple[int, int]]:
        if not indices:
            return None
        start = int(indices[0])
        for offset, value in enumerate(indices):
            if int(value) != start + offset:
                return None
        return start, start + len(indices)

    @staticmethod
    def _page_table_index_slice(
        rows: Tuple[int, ...], positions: Tuple[int, ...]
    ) -> Optional[Tuple[int, int, int]]:
        if not rows or not positions or len(rows) != len(positions):
            return None
        row = int(rows[0])
        start = int(positions[0])
        for offset, (candidate_row, position) in enumerate(zip(rows, positions)):
            if int(candidate_row) != row or int(position) != start + offset:
                return None
        return row, start, start + len(positions)

    def _metadata_patch_cache_entry(
        self,
        layer_id: int,
        demand: _LayerKVKvcDemand,
        metadata: Any,
        scratch_locs: torch.Tensor,
    ) -> _LayerKVMetadataPatchCacheEntry:
        kv_indices = getattr(metadata, "kv_indices", None)
        page_table = getattr(metadata, "page_table", None)
        if kv_indices is not None:
            metadata_kind = "kv_indices"
            metadata_shape = tuple(int(x) for x in kv_indices.shape)
            metadata_device = str(kv_indices.device)
        elif page_table is not None:
            metadata_kind = "page_table"
            metadata_shape = tuple(int(x) for x in page_table.shape)
            metadata_device = str(page_table.device)
        else:
            metadata_kind = "unknown"
            metadata_shape = ()
            metadata_device = ""
        metadata_signature = (
            demand.req_indices,
            demand.positions,
            demand.row_indices,
            demand.flat_indices,
            demand.flat_spans,
            demand.row_spans,
        )
        stable_key = (
            int(layer_id),
            metadata_kind,
            metadata_shape,
            metadata_device,
            int(scratch_locs.numel()),
            int(scratch_locs.data_ptr()),
            metadata_signature,
        )
        shared_key = (
            metadata_kind,
            metadata_shape,
            metadata_device,
            int(scratch_locs.numel()),
            int(scratch_locs.data_ptr()),
            metadata_signature,
        )
        lookup_key = (int(layer_id), metadata_kind)
        entry = self._metadata_patch_cache.get(lookup_key)
        if entry is None or entry.key != stable_key:
            entry = _LayerKVMetadataPatchCacheEntry(key=stable_key)
            shared = self._metadata_patch_tensor_cache.get(shared_key)
            if shared is not None:
                entry.row_tensor = shared.row_tensor
                entry.pos_tensor = shared.pos_tensor
                entry.flat_tensor = shared.flat_tensor
                entry.scratch_tensor = shared.scratch_tensor
                shared.hit_count += 1
                self.stats.metadata_patch_cache_hit_count += 1
            else:
                self._metadata_patch_tensor_cache[shared_key] = entry
            self._metadata_patch_cache[lookup_key] = entry
            self.stats.metadata_patch_cache_miss_count += 1
        else:
            entry.hit_count += 1
            self.stats.metadata_patch_cache_hit_count += 1
        return entry

    def _next_kvc_layer_id(self, layer_id: int) -> Optional[int]:
        layer_ids = self._kvc_layer_ids()
        for idx, candidate in enumerate(layer_ids):
            if int(candidate) == int(layer_id) and idx + 1 < len(layer_ids):
                return int(layer_ids[idx + 1])
        return None

    def _issue_next_virtual_kvc_prefetch(
        self,
        layer_id: int,
        demand: _LayerKVKvcDemand,
        current_buffer_idx: int,
    ) -> None:
        if (
            self.config.kvc_scheduler != "async-deadline"
            or not self._optimized_profile_enabled()
            or self._copy_stream is None
        ):
            return
        next_layer = self._next_kvc_layer_id(layer_id)
        if next_layer is None:
            return
        next_demand = self._get_virtual_kvc_demand(int(next_layer))
        if next_demand is None or not next_demand.entries:
            return
        if self._lookup_virtual_scratch_cache(next_demand) is not None:
            return
        self._issue_virtual_kvc_prefetch(
            next_demand, exclude_buffer_idx=current_buffer_idx
        )

    def _issue_virtual_kvc_prefetch(
        self,
        demand: _LayerKVKvcDemand,
        *,
        exclude_buffer_idx: Optional[int],
    ) -> bool:
        if (
            self.config.kvc_scheduler != "async-deadline"
            or not self._optimized_profile_enabled()
            or self._copy_stream is None
        ):
            return False
        if demand is None or not demand.entries:
            return False
        key = (int(self._decode_step), int(demand.layer_id), demand.signature)
        if key in self._pending_virtual_kvc_materialize:
            return False
        buffer_idx = self._choose_virtual_scratch_buffer(
            demand.token_count, exclude_buffer_idx=exclude_buffer_idx
        )
        if buffer_idx < 0:
            return False
        scratch_locs = self._virtual_scratch_buffers[buffer_idx][: demand.token_count]
        self._ensure_host_store()
        with self._profile("profile_virtual_prefetch_issue_ms"):
            elapsed_ms, start_event, ready_event = (
                self._host_store.reload_layer_to_locs(
                    int(demand.layer_id),
                    list(demand.entries),
                    scratch_locs,
                    stream=self._copy_stream,
                    async_copy=True,
                    host_index=demand.host_index_cpu,
                    host_slice=demand.host_slice,
                )
            )
        if ready_event is None or start_event is None:
            self._record_virtual_materialize(demand.token_count, elapsed_ms)
            return True
        self.stats.virtual_kvc_prefetch_count += 1
        self.stats.virtual_kvc_materialize_count += 1
        self.stats.virtual_kvc_materialize_layer_count += 1
        self.stats.virtual_kvc_materialize_token_count += int(demand.token_count)
        self.stats.virtual_scratch_used_tokens = max(
            self.stats.virtual_scratch_used_tokens, int(demand.token_count)
        )
        self._pending_virtual_kvc_materialize[key] = _LayerKVPendingVirtualMaterialize(
            start_event=start_event,
            ready_event=ready_event,
            layer_id=int(demand.layer_id),
            entries=list(demand.entries),
            scratch_locs=scratch_locs,
            req_indices=demand.req_indices,
            positions=demand.positions,
            token_count=int(demand.token_count),
            buffer_idx=int(buffer_idx),
        )
        return True

    def _wait_for_virtual_materialize(
        self, pending: _LayerKVPendingVirtualMaterialize
    ) -> Optional[torch.Tensor]:
        self.stats.virtual_kvc_prefetch_wait_count += 1
        self.stats.kvc_ready_use_check_count += 1
        self.stats.scheduler_ready_use_check_count += 1
        ready = bool(pending.ready_event.query())
        if ready:
            self.stats.virtual_kvc_prefetch_ready_before_use_count += 1
            self.stats.kvc_ready_before_use_count += 1
            self.stats.scheduler_ready_before_use_count += 1
        else:
            self.stats.layerkv_deadline_miss_count += 1
            self.stats.scheduler_deadline_miss_count += 1
        stream = torch.cuda.current_stream(device=self._kv_pool.device)
        t_wait = time.perf_counter()
        stream.wait_event(pending.ready_event)
        self.stats.scheduler_exposed_wait_ms += (time.perf_counter() - t_wait) * 1000.0
        self.stats.layerkv_copy_event_wait_count += 1
        try:
            elapsed_ms = float(pending.start_event.elapsed_time(pending.ready_event))
            self.stats.virtual_kvc_materialize_ms += elapsed_ms
            self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
        except Exception:
            pass
        self._refresh_ready_before_use_ratio()
        return pending.scratch_locs

    def _refresh_per_layer_kvc_overrides(self) -> None:
        if not self._uses_per_layer_attention_override():
            return
        if self._req_to_token_pool is None:
            return
        table = getattr(self._req_to_token_pool, "req_to_token", None)
        if table is None:
            return
        if (
            self.config.kvc_backend == "per-layer-arena"
            and not self._per_layer_req_to_token_owned
        ):
            # Common physical arena slots are identical to the canonical
            # req_to_token mapping. Defer per-layer override allocation until a
            # layer actually diverges through independent allocation or reload.
            self.stats.kvc_per_layer_override_layer_count = 0
            self.stats.kvc_per_layer_identity_override_count = 0
            return
        if not self._per_layer_req_to_token_overrides:
            for layer_id in self._kvc_layer_ids():
                # Use a shared reference for the bootstrap mapping. This avoids
                # cloning the large req_to_token table while still exercising
                # the per-layer attention metadata rewrite path. Arena-backed
                # layers will replace individual entries with layer-specific
                # tensors once their physical slots are ready.
                self._per_layer_req_to_token_overrides[int(layer_id)] = table
            self.stats.kvc_per_layer_identity_override_count = len(
                self._per_layer_req_to_token_overrides
            )
        elif self.config.kvc_backend == "virtual-arena":
            for layer_id in list(self._per_layer_req_to_token_owned):
                override = self._per_layer_req_to_token_overrides.get(int(layer_id))
                if override is not None:
                    override.copy_(table)
        self.stats.kvc_per_layer_override_layer_count = len(
            self._per_layer_req_to_token_overrides
        )
        self._mark_canonical_per_layer_metadata_current()

    def _mark_canonical_per_layer_metadata_current(self) -> None:
        if not self._uses_per_layer_attention_override():
            return
        if self._req_to_token_pool is None:
            return
        table = getattr(self._req_to_token_pool, "req_to_token", None)
        if table is None:
            return
        try:
            self._per_layer_kvc_current_metadata_key = (
                "canonical",
                int(table.data_ptr()),
                tuple(self._current_forward_req_lens),
                int(self._decode_step),
            )
        except Exception:
            self._per_layer_kvc_current_metadata_key = None

    def _ensure_owned_per_layer_req_to_token(
        self, layer_id: int
    ) -> Optional[torch.Tensor]:
        if not self._uses_per_layer_attention_override():
            return None
        if self._req_to_token_pool is None:
            return None
        table = getattr(self._req_to_token_pool, "req_to_token", None)
        if table is None:
            return None
        layer_id = int(layer_id)
        current = self._per_layer_req_to_token_overrides.get(layer_id)
        if current is None:
            self._per_layer_req_to_token_overrides[layer_id] = table
            current = table
        if layer_id not in self._per_layer_req_to_token_owned:
            # Allocate only when a layer actually needs a distinct slot map.
            # Most layers keep sharing the global req_to_token table.
            current = current.clone()
            self._per_layer_req_to_token_overrides[layer_id] = current
            self._per_layer_req_to_token_owned.add(layer_id)
        self.stats.kvc_per_layer_override_layer_count = len(
            self._per_layer_req_to_token_overrides
        )
        return current

    def _set_per_layer_token_slots(
        self,
        layer_id: int,
        req_indices: List[int],
        positions: List[int],
        new_locs: torch.Tensor,
        req_tensor: Optional[torch.Tensor] = None,
        pos_tensor: Optional[torch.Tensor] = None,
    ) -> bool:
        if not req_indices or not positions or int(new_locs.numel()) == 0:
            return False
        if len(req_indices) != len(positions) or len(req_indices) != int(
            new_locs.numel()
        ):
            return False
        source = self._per_layer_req_to_token_overrides.get(int(layer_id))
        if source is None and self._req_to_token_pool is not None:
            source = getattr(self._req_to_token_pool, "req_to_token", None)
        if source is not None:
            device = source.device
            cmp_req_tensor = (
                req_tensor
                if req_tensor is not None and req_tensor.device == device
                else torch.tensor(req_indices, dtype=torch.int64, device=device)
            )
            cmp_pos_tensor = (
                pos_tensor
                if pos_tensor is not None and pos_tensor.device == device
                else torch.tensor(positions, dtype=torch.int64, device=device)
            )
            expected = new_locs.to(device=device, dtype=source.dtype)
            try:
                if bool(torch.equal(source[cmp_req_tensor, cmp_pos_tensor], expected)):
                    self.stats.kvc_per_layer_slot_override_skip_count += 1
                    self.stats.kvc_per_layer_slot_override_skip_token_count += int(
                        new_locs.numel()
                    )
                    return False
            except Exception:
                pass
        table = self._ensure_owned_per_layer_req_to_token(int(layer_id))
        if table is None:
            return False
        device = table.device
        if req_tensor is None or req_tensor.device != device:
            req_tensor = torch.tensor(req_indices, dtype=torch.int64, device=device)
        if pos_tensor is None or pos_tensor.device != device:
            pos_tensor = torch.tensor(positions, dtype=torch.int64, device=device)
        table[req_tensor, pos_tensor] = new_locs.to(device=device, dtype=table.dtype)
        self._per_layer_req_to_token_versions[int(layer_id)] = (
            int(self._per_layer_req_to_token_versions.get(int(layer_id), 0)) + 1
        )
        self.stats.kvc_per_layer_slot_override_count += 1
        self.stats.kvc_per_layer_slot_override_token_count += int(new_locs.numel())
        return True

    def allocate_per_layer_request_slots(
        self,
        *,
        req: Any,
        positions: List[int],
        common_physical_locs: bool,
    ) -> Optional[torch.Tensor]:
        if not self._per_layer_allocator_enabled():
            return None
        req_idx = getattr(req, "req_pool_idx", None)
        if req_idx is None:
            return None
        req_idx = int(req_idx)
        positions = [int(pos) for pos in positions]
        if not positions:
            return torch.empty(
                (0,), dtype=torch.int64, device=self._allocator.device
            )
        layer_ids = [int(layer_id) for layer_id in self._kvc_layer_ids()]
        if not layer_ids:
            return None
        count = len(positions)
        if common_physical_locs:
            canonical_locs = self._alloc_common_per_layer_locs(count)
            if canonical_locs is None:
                return None
            locs_by_layer = {layer_id: list(canonical_locs) for layer_id in layer_ids}
        else:
            locs_by_layer: Dict[int, List[int]] = {}
            for layer_id in layer_ids:
                layer_locs = self._alloc_per_layer_locs(layer_id, count)
                if layer_locs is None:
                    for rollback_layer, rollback_locs in locs_by_layer.items():
                        self._free_per_layer_locs(rollback_layer, rollback_locs)
                    return None
                locs_by_layer[layer_id] = layer_locs
            canonical_locs = list(locs_by_layer[layer_ids[0]])
            self.stats.kvc_per_layer_independent_alloc_count += count

        owned_keys: List[Tuple[int, int, int]] = []
        device = self._allocator.device
        req_tensor = torch.full((count,), req_idx, dtype=torch.int64, device=device)
        pos_tensor = torch.tensor(positions, dtype=torch.int64, device=device)
        for layer_id in layer_ids:
            layer_locs = locs_by_layer[layer_id]
            if (
                layer_locs != canonical_locs
                or int(layer_id) in self._per_layer_req_to_token_owned
            ):
                self._set_per_layer_token_slots(
                    layer_id,
                    [req_idx] * count,
                    positions,
                    torch.tensor(layer_locs, dtype=torch.int64, device=device),
                    req_tensor=req_tensor,
                    pos_tensor=pos_tensor,
                )
            mapping = self._per_layer_canonical_to_physical.setdefault(layer_id, {})
            for canonical, physical in zip(canonical_locs, layer_locs):
                mapping[int(canonical)] = int(physical)
            for pos, physical in zip(positions, layer_locs):
                key = (int(layer_id), req_idx, int(pos))
                entry = _LayerKVResidencyEntry(
                    req_idx=req_idx,
                    pos=int(pos),
                    state="resident",
                    layer_id=int(layer_id),
                    device_loc=int(physical),
                    device_locs=[int(physical)],
                    page_size=1,
                    last_access_step=self._decode_step,
                )
                self._per_layer_residency[key] = entry
                self._sync_kvc_group_if_needed(entry)
                owned_keys.append(key)
        setattr(req, "layerkv_per_layer_allocated", True)
        setattr(req, "skip_radix_cache_insert", True)
        self._mark_req_layerkv_owned(req_idx, owned_keys)
        self.stats.kvc_per_layer_logical_token_count += count
        self._per_layer_resident_token_count_fast += count * len(layer_ids)
        self._refresh_kvc_residency_stats()
        return torch.tensor(
            canonical_locs, dtype=torch.int64, device=self._allocator.device
        )

    def allocate_decode_slots_for_batch(
        self, batch: Any, token_per_req: int = 1
    ) -> Optional[torch.Tensor]:
        if int(token_per_req) != 1 or not self._per_layer_allocator_enabled():
            return None
        reqs = list(getattr(batch, "reqs", []) or [])
        if not reqs:
            return torch.empty(
                (0,), dtype=torch.int64, device=self._allocator.device
            )
        seq_lens = getattr(batch, "seq_lens", None)
        if seq_lens is None:
            return None
        try:
            positions = [int(x) for x in seq_lens.detach().cpu().tolist()]
        except Exception:
            return None
        req_indices: List[int] = []
        valid_reqs: List[Any] = []
        for req in reqs:
            req_idx = getattr(req, "req_pool_idx", None)
            if req_idx is None:
                return None
            req_indices.append(int(req_idx))
            valid_reqs.append(req)
        if not req_indices:
            return torch.empty(
                (0,), dtype=torch.int64, device=self._allocator.device
            )
        if len(req_indices) != len(positions):
            return None
        layer_ids = [int(layer_id) for layer_id in self._kvc_layer_ids()]
        if not layer_ids:
            return None

        count = len(req_indices)
        canonical_locs = self._alloc_common_per_layer_locs(count)
        if canonical_locs is not None:
            locs_by_layer = {layer_id: list(canonical_locs) for layer_id in layer_ids}
        else:
            locs_by_layer = {}
            for layer_id in layer_ids:
                layer_locs = self._alloc_per_layer_locs(layer_id, count)
                if layer_locs is None:
                    for rollback_layer, rollback_locs in locs_by_layer.items():
                        self._free_per_layer_locs(rollback_layer, rollback_locs)
                    return None
                locs_by_layer[layer_id] = layer_locs
            canonical_locs = list(locs_by_layer[layer_ids[0]])
            self.stats.kvc_per_layer_independent_alloc_count += count

        owned_by_req: Dict[int, List[Tuple[int, int, int]]] = {
            int(req_idx): [] for req_idx in req_indices
        }
        device = self._allocator.device
        req_tensor = torch.tensor(req_indices, dtype=torch.int64, device=device)
        pos_tensor = torch.tensor(positions, dtype=torch.int64, device=device)
        for layer_id in layer_ids:
            layer_locs = locs_by_layer[layer_id]
            if (
                layer_locs != canonical_locs
                or int(layer_id) in self._per_layer_req_to_token_owned
            ):
                self._set_per_layer_token_slots(
                    layer_id,
                    req_indices,
                    positions,
                    torch.tensor(layer_locs, dtype=torch.int64, device=device),
                    req_tensor=req_tensor,
                    pos_tensor=pos_tensor,
                )
            mapping = self._per_layer_canonical_to_physical.setdefault(layer_id, {})
            for canonical, physical in zip(canonical_locs, layer_locs):
                mapping[int(canonical)] = int(physical)
            for req_idx, pos, physical in zip(req_indices, positions, layer_locs):
                key = (int(layer_id), int(req_idx), int(pos))
                entry = _LayerKVResidencyEntry(
                    req_idx=int(req_idx),
                    pos=int(pos),
                    state="resident",
                    layer_id=int(layer_id),
                    device_loc=int(physical),
                    device_locs=[int(physical)],
                    page_size=1,
                    last_access_step=self._decode_step,
                )
                self._per_layer_residency[key] = entry
                self._sync_kvc_group_if_needed(entry)
                owned_by_req[int(req_idx)].append(key)
        for req, req_idx in zip(valid_reqs, req_indices):
            setattr(req, "layerkv_per_layer_allocated", True)
            setattr(req, "skip_radix_cache_insert", True)
            self._mark_req_layerkv_owned(int(req_idx), owned_by_req[int(req_idx)])
        self.stats.kvc_per_layer_logical_token_count += count
        self._per_layer_resident_token_count_fast += count * len(layer_ids)
        self._refresh_kvc_residency_stats()
        return torch.tensor(
            canonical_locs, dtype=torch.int64, device=self._allocator.device
        )

    def can_satisfy_decode_allocation(
        self, required_tokens: int, *, for_prefill: bool = False
    ) -> bool:
        if not self._per_layer_allocator_enabled():
            return False
        required = max(0, int(required_tokens or 0))
        if required <= 0:
            return True
        self._refresh_per_layer_allocator_stats()
        if not for_prefill:
            available = self.stats.kvc_per_layer_physical_arena_min_free_tokens
            if int(available) >= required:
                return True
            if not self._ensure_per_layer_physical_arena(required - int(available)):
                return False
            self._refresh_per_layer_allocator_stats()
            return (
                int(self.stats.kvc_per_layer_physical_arena_min_free_tokens)
                >= required
            )
        if not self._ensure_per_layer_physical_arena(required):
            return False
        self._refresh_per_layer_allocator_stats()
        available = (
            self.stats.kvc_per_layer_physical_arena_common_free_tokens
            if for_prefill
            else self.stats.kvc_per_layer_physical_arena_min_free_tokens
        )
        return int(available) >= required

    def _rewrite_attention_metadata_for_layer(
        self, layer_id: int, req_to_token: torch.Tensor
    ) -> bool:
        forward_batch = self._last_forward_batch
        attn_backend = getattr(forward_batch, "attn_backend", None) or getattr(
            self._runner, "attn_backend", None
        )
        if attn_backend is None:
            return False
        metadata = getattr(attn_backend, "forward_metadata", None)
        if metadata is None:
            return False
        try:
            if hasattr(metadata, "kv_indices") and hasattr(metadata, "kv_indptr"):
                return self._rewrite_triton_kv_indices(metadata, req_to_token)
            if hasattr(metadata, "page_table"):
                return self._rewrite_flashattention_page_table(metadata, req_to_token)
        except Exception as exc:
            self.stats.kvc_guard_pass = False
            self.stats.kvc_guard_reason = f"per_layer_metadata_rewrite_failed:{exc}"
            return False
        return False

    def _rewrite_triton_kv_indices(
        self, metadata: Any, req_to_token: torch.Tensor
    ) -> bool:
        forward_batch = self._last_forward_batch
        if forward_batch is None:
            return False
        kv_indices = getattr(metadata, "kv_indices", None)
        kv_indptr = getattr(metadata, "kv_indptr", None)
        if kv_indices is None or kv_indptr is None:
            return False
        try:
            from sglang.srt.layers.attention.utils import (
                create_flashinfer_kv_indices_triton,
            )
        except Exception:
            return False
        bs = int(getattr(forward_batch, "batch_size", 0) or 0)
        if bs <= 0:
            return False
        seq_lens = getattr(forward_batch, "seq_lens", None)
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if seq_lens is None or req_pool_indices is None:
            return False
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr,
            None,
            kv_indices,
            req_to_token.stride(0),
        )
        return True

    def _rewrite_flashattention_page_table(
        self, metadata: Any, req_to_token: torch.Tensor
    ) -> bool:
        forward_batch = self._last_forward_batch
        if forward_batch is None:
            return False
        page_table = getattr(metadata, "page_table", None)
        if page_table is None:
            return False
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if req_pool_indices is None:
            return False
        max_seq_len_k = int(getattr(metadata, "max_seq_len_k", 0) or 0)
        if max_seq_len_k <= 0:
            try:
                max_seq_len_k = int(page_table.shape[1])
            except Exception:
                return False
        if int(self._page_size) != 1:
            return False
        src = req_to_token[req_pool_indices, :max_seq_len_k]
        page_table[:, :max_seq_len_k].copy_(src.to(page_table.dtype))
        return True

    def _wait_for_kvc_layer_ready(self, layer_id: int) -> None:
        if not self._pending_kvc_reload_events:
            return
        stream = torch.cuda.current_stream(device=self._kv_pool.device)
        for pending in self._pending_kvc_reload_events:
            if pending.waited_on_main_stream:
                continue
            if self.config.kvc_backend == "per-layer-arena" and not any(
                int(entry.layer_id) == int(layer_id) for entry in pending.entries
            ):
                continue
            event = pending.ready_event
            self.stats.kvc_ready_use_check_count += 1
            self.stats.scheduler_ready_use_check_count += 1
            ready = bool(event.query())
            if ready:
                self.stats.kvc_ready_before_use_count += 1
                self.stats.scheduler_ready_before_use_count += 1
            else:
                self.stats.layerkv_deadline_miss_count += 1
                self.stats.scheduler_deadline_miss_count += 1
            t_wait = time.perf_counter()
            stream.wait_event(event)
            self.stats.scheduler_exposed_wait_ms += (
                time.perf_counter() - t_wait
            ) * 1000.0
            self.stats.layerkv_copy_event_wait_count += 1
            for entry in pending.entries:
                group = self._resident_groups.get(
                    self._residency_key("kvc", -1, (int(entry.req_idx), int(entry.pos)))
                )
                if group is not None:
                    group.wait_count += 1
                    group.ready_waited = True
            self.stats.resident_group_wait_count += len(pending.entries)
            pending.waited_on_main_stream = True
        self._refresh_ready_before_use_ratio()

    def _refresh_ready_before_use_ratio(self) -> None:
        total = self.stats.kvc_ready_use_check_count
        if total <= 0:
            self.stats.kvc_ready_before_use_ratio = 1.0
        else:
            self.stats.kvc_ready_before_use_ratio = (
                self.stats.kvc_ready_before_use_count / float(total)
            )

    def _track_per_layer_offloaded_key(
        self, key: Tuple[int, int, int], token_count: Optional[int] = None
    ) -> None:
        layer_id, req_idx, pos = (int(key[0]), int(key[1]), int(key[2]))
        norm_key = (layer_id, req_idx, pos)
        if norm_key in self._per_layer_offloaded_keys:
            return
        self._per_layer_offloaded_keys.add(norm_key)
        self._per_layer_offloaded_keys_by_req.setdefault(req_idx, set()).add(norm_key)
        self._per_layer_offloaded_keys_by_req_layer.setdefault(
            (req_idx, layer_id), set()
        ).add(norm_key)
        tokens = token_count
        if tokens is None:
            entry = self._per_layer_residency.get(norm_key)
            tokens = int(entry.token_count) if entry is not None else 0
        self._per_layer_offloaded_token_count_by_layer[layer_id] = max(
            0,
            int(self._per_layer_offloaded_token_count_by_layer.get(layer_id, 0))
            + int(tokens or 0),
        )
        self._per_layer_offloaded_version += 1
        self._per_layer_offloaded_dirty_reqs.add(req_idx)
        self._per_layer_offloaded_dirty_req_layers.add((req_idx, layer_id))
        self._per_layer_offloaded_sorted_by_req.pop(req_idx, None)
        self._per_layer_offloaded_sorted_by_req_layer.pop((req_idx, layer_id), None)
        self._per_layer_offloaded_positions_by_req_layer.pop((req_idx, layer_id), None)

    def _untrack_per_layer_offloaded_key(
        self, key: Tuple[int, int, int], token_count: Optional[int] = None
    ) -> None:
        layer_id, req_idx, pos = (int(key[0]), int(key[1]), int(key[2]))
        norm_key = (layer_id, req_idx, pos)
        if norm_key not in self._per_layer_offloaded_keys:
            return
        self._per_layer_offloaded_keys.discard(norm_key)
        by_req = self._per_layer_offloaded_keys_by_req.get(req_idx)
        if by_req is not None:
            by_req.discard(norm_key)
            if not by_req:
                self._per_layer_offloaded_keys_by_req.pop(req_idx, None)
        by_req_layer = self._per_layer_offloaded_keys_by_req_layer.get(
            (req_idx, layer_id)
        )
        if by_req_layer is not None:
            by_req_layer.discard(norm_key)
            if not by_req_layer:
                self._per_layer_offloaded_keys_by_req_layer.pop(
                    (req_idx, layer_id), None
                )
        tokens = token_count
        if tokens is None:
            entry = self._per_layer_residency.get(norm_key)
            tokens = int(entry.token_count) if entry is not None else 0
        current = int(self._per_layer_offloaded_token_count_by_layer.get(layer_id, 0))
        next_count = max(0, current - int(tokens or 0))
        if next_count:
            self._per_layer_offloaded_token_count_by_layer[layer_id] = next_count
        else:
            self._per_layer_offloaded_token_count_by_layer.pop(layer_id, None)
        self._per_layer_offloaded_version += 1
        self._per_layer_offloaded_dirty_reqs.add(req_idx)
        self._per_layer_offloaded_dirty_req_layers.add((req_idx, layer_id))
        self._per_layer_offloaded_sorted_by_req.pop(req_idx, None)
        self._per_layer_offloaded_sorted_by_req_layer.pop((req_idx, layer_id), None)
        self._per_layer_offloaded_positions_by_req_layer.pop((req_idx, layer_id), None)

    def _sorted_per_layer_offloaded_keys_for_req_layer(
        self, req_idx: int, layer_id: int
    ) -> Tuple[Tuple[Tuple[int, int, int], ...], Tuple[int, ...]]:
        req_idx = int(req_idx)
        layer_id = int(layer_id)
        cache_key = (req_idx, layer_id)
        cached = self._per_layer_offloaded_sorted_by_req_layer.get(cache_key)
        cached_positions = self._per_layer_offloaded_positions_by_req_layer.get(
            cache_key
        )
        if (
            cached is not None
            and cached_positions is not None
            and int(cached_positions[0]) == int(cached[0])
            and cache_key not in self._per_layer_offloaded_dirty_req_layers
        ):
            self.stats.kvc_required_cache_hit += 1
            return cached[1], cached_positions[1]
        keys = self._per_layer_offloaded_keys_by_req_layer.get(cache_key)
        if not keys:
            self._per_layer_offloaded_dirty_req_layers.discard(cache_key)
            self._per_layer_offloaded_sorted_by_req_layer[cache_key] = (
                int(self._per_layer_offloaded_version),
                (),
            )
            self._per_layer_offloaded_positions_by_req_layer[cache_key] = (
                int(self._per_layer_offloaded_version),
                (),
            )
            self.stats.kvc_required_cache_miss += 1
            return (), ()
        sorted_keys = tuple(sorted(keys, key=lambda x: int(x[2])))
        positions = tuple(int(key[2]) for key in sorted_keys)
        self._per_layer_offloaded_sorted_by_req_layer[cache_key] = (
            int(self._per_layer_offloaded_version),
            sorted_keys,
        )
        self._per_layer_offloaded_positions_by_req_layer[cache_key] = (
            int(self._per_layer_offloaded_version),
            positions,
        )
        self._per_layer_offloaded_dirty_req_layers.discard(cache_key)
        self.stats.kvc_required_cache_miss += 1
        return sorted_keys, positions

    def _recompute_per_layer_residency_fast_counts(self) -> None:
        resident_tokens = 0
        offloaded_tokens = 0
        for entry in self._per_layer_residency.values():
            token_count = int(entry.token_count)
            if entry.state in ("resident", "reloading"):
                resident_tokens += token_count
            elif entry.state == "offloaded":
                offloaded_tokens += token_count
        self._per_layer_resident_token_count_fast = max(0, resident_tokens)
        self._per_layer_offloaded_token_count_fast = max(0, offloaded_tokens)

    def _restore_impossible_per_layer_offloads(self) -> None:
        if self.config.kvc_backend != "per-layer-arena":
            return
        if int(self.stats.kvc_evict_count_total) > 0:
            return
        if (
            not self._per_layer_offloaded_keys
            and int(self._per_layer_offloaded_token_count_fast) <= 0
        ):
            return
        table = getattr(self._req_to_token_pool, "req_to_token", None)
        restored = 0
        dropped = 0
        for key in list(self._per_layer_offloaded_keys):
            layer_id, req_idx, pos = (int(key[0]), int(key[1]), int(key[2]))
            entry = self._per_layer_residency.get(key)
            if entry is None or entry.state != "offloaded":
                continue
            override = self._per_layer_req_to_token_overrides.get(layer_id)
            source = override if override is not None else table
            loc = 0
            if source is not None:
                try:
                    loc = int(source[req_idx, pos].item())
                except Exception:
                    loc = 0
            if loc > 0:
                entry.state = "resident"
                entry.device_loc = loc
                entry.device_locs = [loc]
                entry.host_slot = None
                entry.host_slots = None
                entry.ready_event = None
                entry.ready_start_event = None
                entry.ready_waited = False
                restored += 1
            else:
                self._per_layer_residency.pop(key, None)
                dropped += 1
        self._per_layer_offloaded_keys.clear()
        self._per_layer_offloaded_keys_by_req.clear()
        self._per_layer_offloaded_keys_by_req_layer.clear()
        self._per_layer_offloaded_token_count_by_layer.clear()
        self._per_layer_offloaded_sorted_by_req.clear()
        self._per_layer_offloaded_sorted_by_req_layer.clear()
        self._per_layer_offloaded_positions_by_req_layer.clear()
        self._per_layer_offloaded_dirty_reqs.clear()
        self._per_layer_offloaded_dirty_req_layers.clear()
        self._per_layer_offloaded_version += 1
        self._recompute_per_layer_residency_fast_counts()
        self.stats.kvc_layerwise_required_index_stale_count += restored + dropped

    def _record_kvc_layer_cost_observations(
        self, entries: List[_LayerKVResidencyEntry], elapsed_ms: float, *, kind: str
    ) -> None:
        if (
            self.config.kvc_backend != "per-layer-arena"
            or not entries
            or elapsed_ms <= 0.0
        ):
            return
        by_layer: Dict[int, int] = {}
        for entry in entries:
            by_layer[int(entry.layer_id)] = by_layer.get(int(entry.layer_id), 0) + int(
                entry.token_count
            )
        if not by_layer:
            return
        total_tokens = sum(by_layer.values())
        bytes_per_token = max(1, self._bytes_per_kvc_token_per_layer())
        alpha = 0.20
        for layer_id, tokens in by_layer.items():
            layer_elapsed_ms = float(elapsed_ms) * float(tokens) / float(total_tokens)
            mb = float(tokens * bytes_per_token) / float(1024 * 1024)
            if mb <= 0.0:
                continue
            observed = layer_elapsed_ms / mb
            target = (
                self._kvc_reload_ms_per_mb_ewma_by_layer
                if kind == "reload"
                else self._kvc_evict_ms_per_mb_ewma_by_layer
            )
            old = target.get(int(layer_id))
            target[int(layer_id)] = (
                observed
                if old is None
                else (1.0 - alpha) * float(old) + alpha * observed
            )
            self.stats.kvc_layerwise_cost_observation_count += 1
        reload_values = list(self._kvc_reload_ms_per_mb_ewma_by_layer.values())
        evict_values = list(self._kvc_evict_ms_per_mb_ewma_by_layer.values())
        if reload_values:
            self.stats.kvc_layerwise_reload_ewma_ms_per_mb = sum(reload_values) / float(
                len(reload_values)
            )
        if evict_values:
            self.stats.kvc_layerwise_evict_ewma_ms_per_mb = sum(evict_values) / float(
                len(evict_values)
            )

    def _finalize_reloaded_entries(self, *, block: bool = False) -> None:
        if self._host_store is None:
            return
        if not self._pending_kvc_reload_events:
            return
        still_pending: List[_LayerKVPendingReload] = []
        did_finalize = False
        for pending in self._pending_kvc_reload_events:
            start_event = pending.start_event
            ready_event = pending.ready_event
            entries = pending.entries
            if block:
                ready_event.synchronize()
            elif not ready_event.query():
                still_pending.append(pending)
                continue
            try:
                elapsed_ms = float(start_event.elapsed_time(ready_event))
                self.stats.kvc_reload_ms += elapsed_ms
                self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
                if self.config.kvc_backend == "per-layer-arena":
                    self.stats.kvc_per_layer_reload_ms += elapsed_ms
            except Exception:
                pass
            for entry in entries:
                if entry.state != "reloading":
                    continue
                host_slots = entry.host_slot_list()
                if host_slots:
                    if self.config.kvc_backend == "per-layer-arena":
                        self._host_store.free_per_layer(entry.layer_id, host_slots)
                    else:
                        self._host_store.free(host_slots)
                entry.host_slots = None
                entry.host_slot = None
                entry.state = "resident"
                entry.evicted_device_locs = None
                if self.config.kvc_backend == "per-layer-arena":
                    self._untrack_per_layer_offloaded_key(
                        (int(entry.layer_id), int(entry.req_idx), int(entry.pos)),
                        token_count=int(entry.token_count),
                    )
                    self._per_layer_resident_token_count_fast += int(
                        entry.token_count
                    )
                entry.ready_start_event = None
                entry.ready_event = None
                entry.ready_waited = False
                group = self._sync_kvc_group_if_needed(entry)
                if group is not None:
                    group.recover_count += 1
                self.stats.resident_group_recover_count += 1
                did_finalize = True
            if self.config.kvc_backend == "per-layer-arena":
                try:
                    elapsed_ms = float(start_event.elapsed_time(ready_event))
                    self._record_kvc_layer_cost_observations(
                        entries, elapsed_ms, kind="reload"
                    )
                except Exception:
                    pass
        self._pending_kvc_reload_events = still_pending
        if did_finalize:
            self._refresh_kvc_residency_stats()

    def _finalize_virtual_materialize_events(self, *, block: bool = False) -> None:
        if not self._pending_virtual_kvc_materialize:
            return
        still_pending: Dict[Tuple[int, int], _LayerKVPendingVirtualMaterialize] = {}
        for key, pending in self._pending_virtual_kvc_materialize.items():
            if block:
                pending.ready_event.synchronize()
            elif not pending.ready_event.query():
                still_pending[key] = pending
                continue
            try:
                elapsed_ms = float(
                    pending.start_event.elapsed_time(pending.ready_event)
                )
                self.stats.virtual_kvc_materialize_ms += elapsed_ms
                self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
            except Exception:
                pass
        self._pending_virtual_kvc_materialize = still_pending

    def _finalize_kvc_evictions(self, *, block: bool = False) -> None:
        if self._host_store is None or not self._pending_kvc_evict_events:
            return
        still_pending: List[_LayerKVPendingEviction] = []
        did_finalize = False
        for pending in self._pending_kvc_evict_events:
            if block and not pending.ready_event.query():
                t_wait = time.perf_counter()
                pending.ready_event.synchronize()
                self.stats.kvc_evict_async_wait_ms += (
                    time.perf_counter() - t_wait
                ) * 1000.0
            elif not pending.ready_event.query():
                still_pending.append(pending)
                continue
            try:
                elapsed_ms = float(
                    pending.start_event.elapsed_time(pending.ready_event)
                )
                self.stats.kvc_backup_ms += elapsed_ms
                self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
            except Exception:
                pass
            with self._profile("profile_kvc_evict_commit_ms"):
                self._host_store.commit_staging_backup(
                    pending.host_slots, pending.k_staging, pending.v_staging
                )
            self.stats.kvc_allocator_available_before = self._allocator_available_size()
            self._allocator.free(pending.device_locs)
            self.stats.kvc_allocator_available_after = self._allocator_available_size()
            self.stats.kvc_allocator_free_count += 1
            offset = 0
            for entry in pending.entries:
                page_host_slots = pending.host_slots[
                    offset : offset + entry.token_count
                ]
                offset += entry.token_count
                if self.config.kvc_backend == "per-layer-arena":
                    self._per_layer_resident_token_count_fast = max(
                        0,
                        self._per_layer_resident_token_count_fast
                        - int(entry.token_count),
                    )
                    self._per_layer_offloaded_token_count_fast += int(
                        entry.token_count
                    )
                entry.state = "offloaded"
                entry.host_slots = [int(x) for x in page_host_slots]
                entry.host_slot = int(page_host_slots[0]) if page_host_slots else None
                entry.device_loc = None
                entry.device_locs = None
                entry.ready_event = None
                entry.ready_start_event = None
                entry.ready_waited = False
                entry.last_access_step = self._decode_step
                self._sync_kvc_group_if_needed(entry)
            self.stats.kvc_evict_count_total += int(pending.token_count)
            self.stats.kvc_evict_page_count_total += len(pending.entries)
            self.stats.virtual_kvc_evict_count += int(pending.token_count)
            self.stats.kvc_physical_cycle_count += 1
            self.stats.kvc_evict_async_finalize_count += int(pending.token_count)
            did_finalize = True
        self._pending_kvc_evict_events = still_pending
        if did_finalize:
            self._refresh_kvc_residency_stats()

    def on_forward_begin(self, *, mode: str, forward_batch: Any) -> None:
        t0 = time.perf_counter()
        self._current_forward_mode = mode
        self._last_forward_batch = forward_batch
        self._per_layer_kvc_prepared_layers.clear()
        self._virtual_materialize_plan = None
        self._virtual_materialize_plans_by_layer.clear()
        self._virtual_kvc_demands.clear()
        self._kvc_demand_signature_eval_step = None
        self._per_layer_kvc_current_metadata_key = None
        self._refresh_per_layer_kvc_overrides()
        with self._profile("profile_workload_stats_ms"):
            self._refresh_workload_stats(forward_batch)
        self._mark_canonical_per_layer_metadata_current()
        if (
            self._expert_hotness_pending_snapshots
            or self._expert_candidate_pending_snapshots
            or self._pending_expert_d2h_events
            or self._pending_expert_copy_events
        ):
            with self._profile("profile_finalize_expert_ms"):
                self._finalize_expert_hotness_snapshots(block=False)
                self._finalize_expert_candidate_snapshots(block=False)
                self._finalize_expert_d2h_events(block=False)
                self._finalize_expert_materialize_events(block=False)
        if (
            self._pending_kvc_evict_events
            or self._pending_kvc_reload_events
            or self._pending_virtual_kvc_materialize
        ):
            with self._profile("profile_finalize_kvc_ms"):
                self._finalize_kvc_evictions(block=False)
                self._finalize_reloaded_entries(block=False)
                self._finalize_virtual_materialize_events(block=False)
        if mode == "decode":
            fast_path = self._no_pressure_fast_path_active(forward_batch)
            if fast_path:
                self.stats.layerkv_no_pressure_expert_skip_count += 1
            else:
                with self._profile("profile_apply_expert_plan_ms"):
                    self._apply_expert_plan_once(forward_batch)
                self._advance_expert_install_budgeted()
                self._force_drain_expert_install_d2h_and_slots()
                if self._expert_plan_applied and not self._expert_install_queue:
                    self._check_current_coresid_plan_match(context="post_install")
            self.stats.forward_decode_count += 1
            self._decode_step += 1
            self.stats.decode_steps = self._decode_step
            self._prepare_expert_hotness_sampling_for_step()
            if fast_path:
                self.stats.layerkv_no_pressure_scheduler_skip_count += 1
            else:
                self._run_deadline_scheduler(forward_batch)
        else:
            self.stats.forward_extend_count += 1
            self._drop_entries_for_reqs(forward_batch)
        self.stats.scheduler_invocation_count += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.layerkv_python_overhead_ms += elapsed_ms
        self._add_profile("profile_forward_begin_ms", elapsed_ms)

    def _run_deadline_scheduler(self, forward_batch: Any) -> None:
        if self._no_pressure_fast_path_active(forward_batch):
            self.stats.layerkv_no_pressure_scheduler_skip_count += 1
            return
        if self._simple_profile_enabled():
            with self._profile("profile_kvc_reload_required_ms"):
                self._reload_required_kvc(forward_batch)
            return
        if (
            self.config.mode == "kvc-only"
            and self.config.kvc_backend == "virtual-arena"
        ):
            self.stats.virtual_kvc_scheduler_skip_count += 1
            return
        build_t0 = time.perf_counter()
        tasks = self._build_recovery_tasks(
            forward_batch, max_kvc_tasks=self._kvc_recovery_task_build_limit()
        )
        build_ms = (time.perf_counter() - build_t0) * 1000.0
        self._add_profile("profile_recovery_task_build_ms", build_ms)
        schedule_t0 = time.perf_counter()
        self._schedule_recovery_tasks(tasks)
        schedule_ms = (time.perf_counter() - schedule_t0) * 1000.0
        self._add_profile("profile_recovery_task_schedule_ms", schedule_ms)
        self._add_profile("profile_expert_prefetch_ms", build_ms + schedule_ms)
        total = self.stats.scheduler_ready_use_check_count
        if total <= 0:
            self.stats.scheduler_ready_before_use_ratio = 1.0
        else:
            self.stats.scheduler_ready_before_use_ratio = (
                self.stats.scheduler_ready_before_use_count / float(total)
            )

    def _kvc_recovery_task_build_limit(self) -> Optional[int]:
        if not (
            self._optimized_profile_enabled()
            and self.config.kvc_backend == "per-layer-arena"
        ):
            return None
        hard_cap = 4
        pending = sum(
            1
            for pending_reload in self._pending_kvc_reload_events
            if not getattr(pending_reload, "waited_on_main_stream", False)
        )
        available = max(1, hard_cap - pending)
        return max(1, min(hard_cap, available + 1))

    def try_reclaim_kvc_before_retract(
        self,
        *,
        schedule_batch: Any,
        required_tokens: int,
        available_tokens: int,
        reason: str = "scheduler_pressure",
    ) -> bool:
        """Attempt KVC reclaim before SGLang falls back to request retraction."""
        forced_plan: Optional[Dict[int, int]] = None
        visible_available = int(available_tokens)
        if self.config.kvc_backend == "per-layer-arena":
            self._refresh_per_layer_allocator_stats()
            if reason == "decode_prealloc_admission":
                visible_available = int(available_tokens) + int(
                    self.stats.kvc_per_layer_physical_arena_common_free_tokens
                )
            else:
                visible_available = int(
                    self.stats.kvc_per_layer_physical_arena_min_free_tokens
                )
            block_tokens = max(1, int(self.config.kvc_block_tokens or 1))
            forced_plan = {}
            if reason == "decode_prealloc_admission":
                common_free = int(
                    self.stats.kvc_per_layer_physical_arena_common_free_tokens
                )
                native_available = max(0, int(available_tokens))
                deficit = max(
                    0, int(required_tokens) - int(common_free) - native_available
                )
                if deficit > 0:
                    deficit = (
                        (int(deficit) + block_tokens - 1) // block_tokens
                    ) * block_tokens
                    self._force_common_kvc_evict_tokens = int(deficit)
                    for layer_id in self._kvc_layer_ids():
                        layer_id = int(layer_id)
                        forced_plan[layer_id] = int(
                            self._per_layer_offloaded_token_count_by_layer.get(
                                layer_id, 0
                            )
                            + deficit
                        )
            else:
                for layer_id in self._kvc_layer_ids():
                    layer_id = int(layer_id)
                    free_tokens = len(self._per_layer_arena_free_locs.get(layer_id, []))
                    deficit = max(0, int(required_tokens) - int(free_tokens))
                    if deficit <= 0:
                        continue
                    deficit = (
                        (int(deficit) + block_tokens - 1) // block_tokens
                    ) * block_tokens
                    forced_plan[layer_id] = int(
                        self._per_layer_offloaded_token_count_by_layer.get(layer_id, 0)
                        + deficit
                    )
            if not forced_plan:
                visible_available = max(visible_available, int(required_tokens))
        shortage_tokens = max(0, int(required_tokens) - int(visible_available))
        self.stats.pre_retract_reclaim_needed_tokens = int(shortage_tokens)
        self.stats.pre_retract_reclaim_allocator_available_before = int(
            visible_available
        )
        self.stats.pre_retract_reclaim_allocator_available_after = int(visible_available)
        if shortage_tokens <= 0:
            return False
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return False
        if not self.physical_kvc_supported:
            self.stats.comparable = False
            self.stats.comparability_reason = self.unsupported_reason
            return False
        if self._bytes_per_token_all_layers <= 0:
            return False
        if not self._kvc_reclaim_is_scheduler_visible():
            self.stats.kvc_scheduler_invisible_skip_count += 1
            self.stats.kvc_scheduler_invisible_skip_tokens += int(shortage_tokens)
            return False

        if self.config.kvc_backend == "per-layer-arena" and forced_plan is not None:
            current_by_layer = self._per_layer_offloaded_token_count_by_layer
            requested_kvc_tokens = sum(
                max(0, int(target) - int(current_by_layer.get(int(layer_id), 0)))
                for layer_id, target in forced_plan.items()
            )
            requested_kvc_tokens = self._align_tokens_up(requested_kvc_tokens)
        else:
            requested_kvc_tokens = self._align_tokens_up(int(shortage_tokens))
        if self.config.kvc_backend == "virtual-arena":
            block_tokens = max(1, int(self.config.kvc_block_tokens or 1))
            requested_kvc_tokens = (
                (requested_kvc_tokens + block_tokens - 1) // block_tokens
            ) * block_tokens
        self.stats.pre_retract_reclaim_requested_kvc_tokens = int(requested_kvc_tokens)
        self.stats.pre_retract_reclaim_attempt_count += 1
        before_offloaded = self._offloaded_token_count()
        old_plan = dict(self._planned_kvc_tokens_by_layer)
        old_target = int(self._planned_kvc_token_target)
        try:
            if forced_plan is not None:
                self._planned_kvc_tokens_by_layer = dict(forced_plan)
                self._planned_kvc_token_target = int(
                    sum(max(0, int(v)) for v in forced_plan.values())
                )
            self._evict_kvc_to_target(
                schedule_batch,
                force_additional_tokens=requested_kvc_tokens,
                force_reason=reason or "pre_retract_decode_mem",
            )
        except Exception:
            self.stats.kvc_physical_failure_count += 1
            raise
        finally:
            if forced_plan is not None:
                self._planned_kvc_tokens_by_layer = old_plan
                self._planned_kvc_token_target = old_target
                self._force_common_kvc_evict_tokens = 0
        if self.config.kvc_backend == "per-layer-arena":
            self._refresh_per_layer_allocator_stats()
            if reason == "decode_prealloc_admission":
                after_available = int(available_tokens) + int(
                    self.stats.kvc_per_layer_physical_arena_common_free_tokens
                )
            else:
                after_available = int(
                    self.stats.kvc_per_layer_physical_arena_min_free_tokens
                )
        else:
            after_available = self._allocator_available_size()
        self.stats.pre_retract_reclaim_allocator_available_after = int(after_available)
        if after_available > visible_available:
            self.stats.pre_retract_reclaim_scheduler_visible_success_count += 1
        if after_available >= int(required_tokens):
            self.stats.pre_retract_reclaim_success_count += 1
            return True
        if self._offloaded_token_count() > before_offloaded:
            self.stats.pre_retract_reclaim_success_count += 1
            return True
        return after_available > visible_available

    def on_requests_retracted(self, req_pool_indices: List[int]) -> None:
        if not req_pool_indices:
            return
        if self.config.kvc_backend == "virtual-arena":
            self._finalize_kvc_evictions(block=True)
        req_indices = {int(idx) for idx in req_pool_indices if idx is not None}
        if not req_indices:
            return
        to_drop = [key for key in self._residency if key[0] in req_indices]
        per_layer_to_drop = [
            key for key in self._per_layer_residency if key[1] in req_indices
        ]
        self._drop_residency_keys(to_drop)
        self._drop_per_layer_residency_keys(per_layer_to_drop)
        for req_idx in req_indices:
            if req_idx in self._kvc_evict_cursors:
                self._kvc_evict_cursors.pop(req_idx, None)
                self.stats.kvc_evict_cursor_reset_count += 1
            for cursor_key in list(self._kvc_evict_cursors_by_layer):
                if int(cursor_key[1]) == req_idx:
                    self._kvc_evict_cursors_by_layer.pop(cursor_key, None)
                    self.stats.kvc_layerwise_evict_cursor_reset_count += 1

    def _mark_req_virtualized(self, req_idx: int) -> None:
        if self._runner is None:
            return
        for req in getattr(getattr(self._runner, "reqs", None), "reqs", []) or []:
            if getattr(req, "req_pool_idx", None) == req_idx:
                req.layerkv_virtualized_kvc = True
                req.skip_radix_cache_insert = True
                return
        scheduler = getattr(self._runner, "scheduler", None)
        running_batch = getattr(scheduler, "running_batch", None)
        for req in getattr(running_batch, "reqs", []) or []:
            if getattr(req, "req_pool_idx", None) == req_idx:
                req.layerkv_virtualized_kvc = True
                req.skip_radix_cache_insert = True
                return

    def release_virtualized_request(self, req: Any, tree_cache: Any) -> bool:
        if self.config.kvc_backend == "per-layer-arena":
            if not getattr(req, "layerkv_per_layer_allocated", False):
                return False
            req_pool_idx = getattr(req, "req_pool_idx", None)
            if req_pool_idx is None:
                return False
            req_idx = int(req_pool_idx)
            cleaned = bool(getattr(req, "layerkv_per_layer_cleaned", False))
            keys = []
            if not cleaned:
                keys = list(self._per_layer_owned_keys_by_req.pop(req_idx, set()))
            else:
                self._per_layer_owned_keys_by_req.pop(req_idx, None)
            if not keys and not cleaned:
                keys = [key for key in self._per_layer_residency if key[1] == req_idx]
            token_count = 0
            for key in keys:
                entry = self._per_layer_residency.get(key)
                if entry is not None:
                    token_count += int(entry.token_count)
            self._drop_per_layer_residency_keys(keys)
            self._per_layer_owned_req_indices.discard(req_idx)
            if not getattr(req, "kv_committed_freed", False):
                req.pop_committed_kv_cache()
            if not getattr(req, "kv_overallocated_freed", False):
                req.pop_overallocated_kv_cache()
            if getattr(req, "req_pool_idx", None) is not None:
                if self._req_to_token_pool is not None:
                    self._req_to_token_pool.free(req)
            try:
                req.layerkv_per_layer_allocated = False
                req.layerkv_per_layer_cleaned = False
            except Exception:
                pass
            self.stats.kvc_layerkv_owned_release_count += 1
            self.stats.kvc_layerkv_owned_release_token_count += int(token_count)
            self.stats.kvc_finished_req_cleanup_count += 1
            self.stats.kvc_finished_req_cleanup_token_count += int(token_count)
            self._refresh_per_layer_allocator_stats()
            self._refresh_kvc_residency_stats()
            return True
        if self.config.kvc_backend != "virtual-arena":
            return False
        req_pool_idx = getattr(req, "req_pool_idx", None)
        if req_pool_idx is None:
            return False
        req_idx = int(req_pool_idx)
        self._finalize_kvc_evictions(block=True)
        keys = [key for key in self._residency if key[0] == req_idx]
        if not keys and not getattr(req, "layerkv_virtualized_kvc", False):
            return False
        affected_entries = [
            self._residency[key] for key in keys if key in self._residency
        ]
        self._invalidate_virtual_caches_for_layers(
            self._virtual_cache_layers_for_entries(affected_entries)
        )

        committed_len = int(req.pop_committed_kv_cache())
        start_p, end_p = req.pop_overallocated_kv_cache()
        end_pos = max(committed_len, int(end_p))
        table = self._req_to_token_pool.req_to_token
        virtual_positions: Set[int] = set()
        host_slots: List[int] = []
        for key in keys:
            entry = self._residency.pop(key, None)
            if entry is None:
                continue
            if entry.state == "offloaded":
                virtual_positions.update(entry.logical_positions())
                host_slots.extend(entry.host_slot_list())
            self._remove_resident_group("kvc", -1, (int(entry.req_idx), int(entry.pos)))
        if host_slots and self._host_store is not None:
            self._host_store.free(host_slots)
        free_positions = [
            pos for pos in range(0, end_pos) if pos not in virtual_positions
        ]
        if free_positions:
            pos_tensor = torch.tensor(
                free_positions, dtype=torch.int64, device=table.device
            )
            indices = table[req_idx, pos_tensor].to(dtype=torch.int64)
            indices = torch.unique(indices[indices > 0])
            if int(indices.numel()) > 0:
                self._allocator.free(indices)
        if getattr(req, "last_node", None) is not None:
            try:
                tree_cache.dec_lock_ref(req.last_node)
            except Exception:
                pass
        self._req_to_token_pool.free(req)
        self.stats.virtual_kvc_release_count += 1
        self.stats.virtual_kvc_release_token_count += len(virtual_positions)
        self._refresh_kvc_residency_stats()
        return True

    def backup_retracted_request_for_native_resume(
        self,
        req: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
    ) -> bool:
        """Create a native-format CPU KV copy for a per-layer-arena retraction."""
        if self.config.kvc_backend != "per-layer-arena":
            return False
        if not getattr(req, "layerkv_per_layer_allocated", False):
            return False
        req_pool_idx = getattr(req, "req_pool_idx", None)
        if req_pool_idx is None:
            return False
        req_idx = int(req_pool_idx)
        token_count = max(0, int(getattr(req, "seqlen", 0) or 0) - 1)
        if token_count <= 0:
            req.kv_cache_cpu = []
            req.layerkv_native_after_retract = True
            return True

        kv_pool = token_to_kv_pool_allocator.get_kvcache()
        chunk_size = int(getattr(kv_pool, "cpu_offloading_chunk_size", 0) or 0)
        if chunk_size <= 0:
            chunk_size = token_count
        device = getattr(kv_pool, "device", self._allocator.device)
        native_indices = req_to_token_pool.req_to_token[
            req_idx, :token_count
        ].to(device=device, dtype=torch.long)
        kv_cache_cpu = []
        torch.cuda.synchronize()
        for layer_offset in range(int(kv_pool.layer_num)):
            layer_id = int(kv_pool.start_layer) + int(layer_offset)
            mapping = self._per_layer_canonical_to_physical.get(layer_id, {})
            layer_locs: List[int] = []
            for pos in range(token_count):
                entry = self._per_layer_residency.get((layer_id, req_idx, pos))
                locs = entry.device_loc_list() if entry is not None else []
                if locs:
                    layer_locs.append(int(locs[0]))
                else:
                    canonical = int(native_indices[pos].item())
                    layer_locs.append(int(mapping.get(canonical, canonical)))
            layer_locs_tensor = torch.tensor(
                layer_locs, dtype=torch.long, device=device
            )
            layer_chunks = []
            for start in range(0, token_count, chunk_size):
                loc_chunk = layer_locs_tensor[start : start + chunk_size]
                k_cpu = kv_pool._get_key_buffer(layer_id)[loc_chunk].to(
                    "cpu", non_blocking=True
                )
                v_cpu = kv_pool._get_value_buffer(layer_id)[loc_chunk].to(
                    "cpu", non_blocking=True
                )
                layer_chunks.append([k_cpu, v_cpu])
            kv_cache_cpu.append(layer_chunks)
        torch.cuda.synchronize()
        req.kv_cache_cpu = kv_cache_cpu
        req.layerkv_native_after_retract = True
        return True

    def _build_kvc_recovery_task(
        self, forward_batch: Any
    ) -> Optional[_LayerKVRecoveryTask]:
        tasks = self._build_kvc_recovery_tasks(forward_batch)
        return tasks[0] if tasks else None

    def _build_kvc_recovery_tasks(
        self, forward_batch: Any, max_tasks: Optional[int] = None
    ) -> List[_LayerKVRecoveryTask]:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return []
        if self.config.kvc_backend == "virtual-arena" or (
            self.config.kvc_backend == "per-layer-arena"
            and self._per_layer_virtual_scratch_enabled()
        ):
            if (
                self.config.kvc_backend == "per-layer-arena"
                and not self._has_offloaded_kvc_entries()
                and not self._pending_virtual_kvc_materialize
            ):
                self.stats.virtual_kvc_scheduler_skip_count += 1
                return []
            if not self._ensure_virtual_scratch():
                self.stats.virtual_kvc_scheduler_skip_count += 1
                return []
            bytes_per_token = self._bytes_per_kvc_token_per_layer()
            tasks: List[_LayerKVRecoveryTask] = []
            layer_ids = self._kvc_layer_ids()
            if (
                max_tasks is not None
                and self._optimized_profile_enabled()
                and self.config.kvc_backend == "per-layer-arena"
            ):
                # The per-layer attention hook already materializes the current
                # layer and prefetches the next one.  Building all 48 layer
                # demands at scheduler entry duplicates that work and dominates
                # the pressure path, so keep the entry prefetch window bounded.
                layer_ids = layer_ids[: max(1, min(len(layer_ids), int(max_tasks)))]
            for layer_id in layer_ids:
                demand = self._get_virtual_kvc_demand(int(layer_id))
                if demand is None or not demand.entries:
                    continue
                if (
                    self._lookup_virtual_scratch_cache(demand, record_stats=False)
                    is not None
                ):
                    continue
                key = (int(self._decode_step), int(layer_id), demand.signature)
                if key in self._pending_virtual_kvc_materialize:
                    continue
                tasks.append(
                    _LayerKVRecoveryTask(
                        kind="kvc",
                        layer_id=int(layer_id),
                        bytes=int(demand.token_count * bytes_per_token),
                        deadline_layer=int(layer_id),
                        demand_signature=demand.signature,
                        kvc_demand=demand,
                        entries=tuple(demand.entries),
                        token_count=int(demand.token_count),
                        benefit_score=float(demand.benefit_score),
                    )
                )
                if max_tasks is not None and len(tasks) >= int(max_tasks):
                    break
            if not tasks:
                self.stats.virtual_kvc_scheduler_skip_count += 1
            return tasks
        if not self._has_offloaded_kvc_entries():
            return []
        if not self.physical_kvc_supported:
            self.stats.comparable = False
            self.stats.comparability_reason = self.unsupported_reason
            return []
        if self._bytes_per_token_all_layers <= 0:
            return []
        self._prune_entries_for_active_lengths(forward_batch)
        with self._profile("profile_kvc_select_required_ms"):
            selected = self._select_required_offloaded_entries(forward_batch)
        if not selected:
            return []
        if self.config.kvc_backend == "per-layer-arena":
            by_layer: Dict[int, List[_LayerKVResidencyEntry]] = {}
            skipped_layers: Set[int] = set()
            max_layer_tasks = (
                max(1, int(max_tasks)) if max_tasks is not None else None
            )
            for entry in selected:
                layer_id = int(entry.layer_id)
                if (
                    max_layer_tasks is not None
                    and layer_id not in by_layer
                    and len(by_layer) >= max_layer_tasks
                ):
                    skipped_layers.add(layer_id)
                    continue
                by_layer.setdefault(layer_id, []).append(entry)
            if skipped_layers:
                self.stats.scheduler_kvc_deferred_count += len(skipped_layers)
            if (
                self._optimized_profile_enabled()
                and self.config.kvc_backend == "per-layer-arena"
            ):
                merged: List[_LayerKVResidencyEntry] = []
                for _layer_id, entries in sorted(by_layer.items()):
                    merged.extend(entries)
                if merged:
                    self.stats.kvc_task_coalesced_count += max(
                        0, len(by_layer) - 1
                    )
                    first_layer = min(int(entry.layer_id) for entry in merged)
                    return [self._make_kvc_recovery_task(first_layer, merged)]
                return []
            return [
                self._make_kvc_recovery_task(int(layer_id), entries)
                for layer_id, entries in sorted(by_layer.items())
                if entries
            ]
        return [self._make_kvc_recovery_task(-1, selected)]

    def _make_kvc_recovery_task(
        self, demand_layer: int, selected: List[_LayerKVResidencyEntry]
    ) -> _LayerKVRecoveryTask:
        token_count = sum(entry.token_count for entry in selected)
        demand_signature = self._generic_kvc_demand_signature(demand_layer, selected)
        kvc_demand = _LayerKVKvcDemand(
            layer_id=int(demand_layer),
            entries=tuple(selected),
            req_indices=tuple(int(entry.req_idx) for entry in selected),
            positions=tuple(int(entry.pos) for entry in selected),
            row_indices=(),
            flat_indices=(),
            flat_spans=(),
            row_spans=(),
            token_count=int(token_count),
            deadline_layer=max(0, int(demand_layer)),
            benefit_score=float(token_count),
            signature=demand_signature,
            base_signature=self._kvc_demand_signature_without_layer(demand_signature),
            backend_semantics=str(self.stats.layerkv_kvc_backend_semantics or ""),
        )
        if (
            self.config.kvc_backend == "per-layer-arena"
            and self.config.runtime_profile == "optimized"
        ):
            group_keys = ()
        else:
            group_keys = tuple(self._sync_kvc_group(entry).key for entry in selected)
        bytes_per_token = (
            self._bytes_per_kvc_token_per_layer()
            if self.config.kvc_backend == "per-layer-arena"
            else self._bytes_per_token_all_layers
        )
        task_bytes = int(token_count * max(1, bytes_per_token))
        if len(selected) > 1:
            self.stats.kvc_task_coalesced_count += 1
        self.stats.kvc_task_token_sum += int(token_count)
        self.stats.kvc_task_byte_sum += int(task_bytes)
        task_count = max(1, int(self.stats.scheduler_kvc_task_count) + 1)
        self.stats.kvc_avg_task_tokens = self.stats.kvc_task_token_sum / float(
            task_count
        )
        self.stats.kvc_avg_task_bytes = self.stats.kvc_task_byte_sum / float(
            task_count
        )
        return _LayerKVRecoveryTask(
            kind="kvc",
            layer_id=int(demand_layer),
            bytes=task_bytes,
            deadline_layer=max(0, int(demand_layer)),
            demand_signature=demand_signature,
            kvc_demand=kvc_demand,
            group_keys=group_keys,
            entries=tuple(selected),
            token_count=int(token_count),
            benefit_score=float(token_count),
        )

    def _generic_kvc_demand_signature(
        self, layer_id: int, selected: List[_LayerKVResidencyEntry]
    ) -> str:
        if not selected:
            return f"L{int(layer_id)}|empty|step{int(self._decode_step)}"
        first = selected[0]
        last = selected[-1]
        token_count = sum(entry.token_count for entry in selected)
        return (
            f"L{int(layer_id)}|step{int(self._decode_step)}"
            f"|n{len(selected)}|tok{int(token_count)}"
            f"|first{int(first.req_idx)}:{int(first.pos)}:{int(first.layer_id)}"
            f"|last{int(last.req_idx)}:{int(last.pos)}:{int(last.layer_id)}"
        )

    def _build_recovery_tasks(
        self, forward_batch: Any, max_kvc_tasks: Optional[int] = None
    ) -> List[_LayerKVRecoveryTask]:
        tasks: List[_LayerKVRecoveryTask] = []
        tasks.extend(
            self._build_kvc_recovery_tasks(forward_batch, max_tasks=max_kvc_tasks)
        )
        if not self._expert_plan_applied or not self._expert_layers:
            return tasks
        if (
            self._coresid_optimized_policy_enabled()
            and self._expert_prefetch_dirty_layers
        ):
            expert_items = [
                (layer_id, self._expert_layers[layer_id])
                for layer_id in sorted(self._expert_prefetch_dirty_layers)
                if layer_id in self._expert_layers
            ]
        elif self._coresid_optimized_policy_enabled():
            expert_items = []
        else:
            expert_items = sorted(self._expert_layers.items())
        next_dirty_layers: Set[int] = set()
        expert_scan_t0 = time.perf_counter()
        for layer_id, state in expert_items:
            if state.prefetched_logical_ids:
                stale = [
                    int(expert_id)
                    for expert_id in state.prefetched_logical_ids
                    if int(expert_id) not in state.logical_to_slot
                ]
                if stale:
                    self.stats.expert_prefetch_wasted_count += len(stale)
                    state.prefetched_logical_ids.difference_update(stale)
            if not state.last_decode_logical_ids:
                self.stats.expert_prefetch_hit_count += 0
                continue
            if (
                self._coresid_optimized_policy_enabled()
                and state.resident_count >= state.full_num_experts
                and not state.prefetched_logical_ids
            ):
                self.stats.expert_prefetch_hit_count += len(
                    state.last_decode_logical_ids
                )
                continue
            missing = [
                int(expert_id)
                for expert_id in state.last_decode_logical_ids
                if int(expert_id) not in state.logical_to_slot
                and int(expert_id) in state.cpu_params
            ]
            missing = list(dict.fromkeys(missing))
            self.stats.expert_prefetch_skipped_resident_count += max(
                0, len(state.last_decode_logical_ids) - len(missing)
            )
            if not missing:
                self.stats.expert_prefetch_hit_count += len(
                    state.last_decode_logical_ids
                )
                continue
            next_dirty_layers.add(int(layer_id))
            self.stats.expert_prefetch_candidate_count += len(missing)
            group_keys = []
            if self._expert_group_tracking_enabled():
                for expert_id in missing:
                    group = self._get_or_create_resident_group(
                        kind="expert",
                        layer_id=int(layer_id),
                        logical_id=int(expert_id),
                        state="offloaded",
                        bytes=state.expert_bytes,
                        metadata={"expert_id": int(expert_id)},
                    )
                    group_keys.append(group.key)
            expert_signature = (
                f"L{int(layer_id)}|step{int(self._decode_step)}"
                f"|n{len(missing)}|ids{','.join(str(int(x)) for x in missing)}"
            )
            expert_demand = _LayerKVExpertDemand(
                layer_id=int(layer_id),
                logical_ids=tuple(int(x) for x in missing),
                bytes=int(len(missing) * state.expert_bytes),
                deadline_layer=int(layer_id),
                benefit_score=sum(
                    self._expert_hotness_score(state, expert_id)
                    for expert_id in missing
                ),
                signature=expert_signature,
                group_keys=tuple(group_keys),
            )
            tasks.append(
                _LayerKVRecoveryTask(
                    kind="expert",
                    layer_id=int(layer_id),
                    bytes=int(expert_demand.bytes),
                    deadline_layer=int(layer_id),
                    demand_signature=expert_signature,
                    expert_demand=expert_demand,
                    logical_ids=tuple(missing),
                    group_keys=tuple(group_keys),
                    benefit_score=float(expert_demand.benefit_score),
                )
            )
        if self._coresid_optimized_policy_enabled():
            self._expert_prefetch_dirty_layers = next_dirty_layers
        self._add_profile(
            "profile_expert_prefetch_scan_ms",
            (time.perf_counter() - expert_scan_t0) * 1000.0,
        )
        return tasks

    def _schedule_recovery_tasks(self, tasks: List[_LayerKVRecoveryTask]) -> None:
        if not tasks:
            return
        for task in tasks:
            if task.estimated_copy_ms <= 0.0:
                task.estimated_copy_ms = self._estimate_recovery_task_copy_ms(task)
            if task.estimated_cpu_ms <= 0.0:
                task.estimated_cpu_ms = self._estimate_recovery_task_cpu_ms(task)
        tasks.sort(
            key=lambda task: (
                task.deadline_layer,
                self._recovery_task_ready_rank(task),
                -max(0.0, task.estimated_copy_ms - task.estimated_cpu_ms),
                -task.benefit_score,
                task.bytes,
            )
        )
        coalesced: List[_LayerKVRecoveryTask] = []
        for task in tasks:
            if (
                coalesced
                and task.kind == "expert"
                and coalesced[-1].kind == "expert"
                and task.layer_id == coalesced[-1].layer_id
            ):
                prev = coalesced[-1]
                coalesced[-1] = _LayerKVRecoveryTask(
                    kind=prev.kind,
                    layer_id=prev.layer_id,
                    bytes=prev.bytes + task.bytes,
                    deadline_layer=prev.deadline_layer,
                    demand_signature=prev.demand_signature or task.demand_signature,
                    expert_demand=prev.expert_demand or task.expert_demand,
                    logical_ids=prev.logical_ids + task.logical_ids,
                    group_keys=prev.group_keys + task.group_keys,
                    token_count=prev.token_count + task.token_count,
                    benefit_score=prev.benefit_score + task.benefit_score,
                    estimated_copy_ms=prev.estimated_copy_ms + task.estimated_copy_ms,
                    estimated_cpu_ms=prev.estimated_cpu_ms + task.estimated_cpu_ms,
                )
            else:
                coalesced.append(task)
        self.stats.scheduler_task_count += len(coalesced)
        self.stats.scheduler_coalesced_task_count += max(0, len(tasks) - len(coalesced))
        self.stats.expert_materialize_coalesced_count += max(
            0, len(tasks) - len(coalesced)
        )
        issued_kvc_tasks = 0
        issued_expert_tasks = 0
        kvc_budget, expert_budget = self._scheduler_copy_task_budgets(coalesced)
        self.stats.scheduler_copy_budget_kvc_tasks = int(kvc_budget)
        self.stats.scheduler_copy_budget_expert_tasks = int(expert_budget)
        for task in coalesced:
            if task.kind == "kvc":
                if not task.entries:
                    continue
                if issued_kvc_tasks >= kvc_budget:
                    self.stats.scheduler_kvc_deferred_count += 1
                    continue
                if self.config.kvc_backend == "virtual-arena":
                    issued = False
                    if task.kvc_demand is not None:
                        issued = self._issue_virtual_kvc_prefetch(
                            task.kvc_demand, exclude_buffer_idx=None
                        )
                    if issued:
                        self.stats.scheduler_kvc_task_count += 1
                        self.stats.scheduler_copy_bytes_total += int(task.bytes)
                        issued_kvc_tasks += 1
                    else:
                        self.stats.virtual_kvc_scheduler_skip_count += 1
                    continue
                before_kvc_bytes = float(self.stats.kvc_reload_mb_total)
                with self._profile("profile_kvc_reload_required_ms"):
                    issued = self._reload_required_kvc(
                        self._last_forward_batch,
                        selected_entries=list(task.entries),
                        strict=False,
                    )
                if not issued:
                    self.stats.scheduler_kvc_deferred_count += 1
                    continue
                self.stats.scheduler_kvc_task_count += 1
                self.stats.scheduler_copy_bytes_total += int(
                    max(0.0, self.stats.kvc_reload_mb_total - before_kvc_bytes)
                    * 1024
                    * 1024
                )
                issued_kvc_tasks += 1
                continue
            if task.kind != "expert":
                continue
            state = self._expert_layers.get(int(task.layer_id))
            if state is None:
                continue
            logical_ids = list(task.logical_ids)
            if not logical_ids:
                continue
            if issued_expert_tasks >= expert_budget:
                self.stats.scheduler_expert_deferred_count += 1
                continue
            self.stats.scheduler_expert_task_count += 1
            self.stats.scheduler_copy_bytes_total += int(task.bytes)
            self.stats.expert_prefetch_count += len(logical_ids)
            self.stats.expert_prefetch_miss_count += len(logical_ids)
            self.stats.expert_prefetch_issued_count += len(logical_ids)
            groups = [
                self._resident_groups[key]
                for key in task.group_keys
                if key in self._resident_groups
            ]
            backend = self._residency_backends.get("expert")
            if backend is not None and groups:
                backend.recover(groups, stream=self._copy_stream)
            else:
                self._materialize_experts(state, logical_ids, reason="prefetch")
            issued_expert_tasks += 1

    def _scheduler_copy_task_budgets(
        self, tasks: Optional[List[_LayerKVRecoveryTask]] = None
    ) -> Tuple[int, int]:
        if not self._optimized_profile_enabled():
            return 1_000_000, 1_000_000
        kvc_budget = 1
        expert_budget = 1
        if self.config.kvc_backend == "per-layer-arena":
            kvc_budget = self._dynamic_per_layer_kvc_task_budget(tasks or [])
        if self.config.kvc_backend == "virtual-arena":
            available_buffers = max(
                1,
                len(self._virtual_scratch_buffers)
                - len(self._pending_virtual_kvc_materialize),
            )
            kvc_budget = max(1, min(available_buffers, 2))
        return kvc_budget, expert_budget

    def _dynamic_per_layer_kvc_task_budget(
        self, tasks: List[_LayerKVRecoveryTask]
    ) -> int:
        kvc_tasks = [task for task in tasks if task.kind == "kvc" and task.entries]
        if not kvc_tasks:
            return 1
        if self._copy_stream is None:
            return 1
        pending = sum(
            1
            for pending_reload in self._pending_kvc_reload_events
            if not getattr(pending_reload, "waited_on_main_stream", False)
        )
        hard_cap = 4
        if pending >= hard_cap:
            self.stats.kvc_layerwise_scheduler_dynamic_budget_count += 1
            return 1
        forward_batch = self._last_forward_batch
        slack_qualified = 0
        for task in kvc_tasks:
            layer_id = max(0, int(task.layer_id))
            slack_ms = (
                self._estimate_layer_kvc_overlap_window_ms(layer_id, forward_batch)
                if forward_batch is not None
                else 0.0
            )
            estimated_ms = max(0.0, float(task.estimated_copy_ms))
            if layer_id == 0 or estimated_ms <= slack_ms + 0.05:
                slack_qualified += 1
            else:
                self.stats.kvc_layerwise_scheduler_deadline_reject_count += 1
        budget = max(1, min(hard_cap - pending, len(kvc_tasks), slack_qualified or 1))
        self.stats.kvc_layerwise_scheduler_dynamic_budget_count += 1
        return int(budget)

    def _recovery_task_ready_rank(self, task: _LayerKVRecoveryTask) -> int:
        if (
            task.kind == "kvc"
            and self.config.kvc_backend == "virtual-arena"
            and task.kvc_demand is not None
        ):
            if (
                self._lookup_virtual_scratch_cache(task.kvc_demand, record_stats=False)
                is not None
            ):
                return 0
            return 1
        if task.kind == "expert":
            for key in task.group_keys:
                group = self._resident_groups.get(key)
                ready_event = getattr(group, "ready_event", None)
                if ready_event is not None:
                    try:
                        return 0 if ready_event.query() else 1
                    except Exception:
                        return 1
            return 1
        return 1

    def _estimate_recovery_task_copy_ms(self, task: _LayerKVRecoveryTask) -> float:
        bytes_count = max(0, int(task.bytes))
        if bytes_count <= 0:
            return 0.0
        if (
            task.kind == "kvc"
            and self.config.kvc_backend == "per-layer-arena"
            and task.entries
        ):
            tokens_by_layer: Dict[int, int] = {}
            for entry in task.entries:
                tokens_by_layer[int(entry.layer_id)] = tokens_by_layer.get(
                    int(entry.layer_id), 0
                ) + int(entry.token_count)
            if tokens_by_layer:
                bytes_per_token = max(1, self._bytes_per_kvc_token_per_layer())
                estimated = 0.0
                missing = False
                for layer_id, tokens in tokens_by_layer.items():
                    reload_ewma = self._kvc_reload_ms_per_mb_ewma_by_layer.get(
                        int(layer_id)
                    )
                    if reload_ewma is None:
                        missing = True
                        break
                    mb = float(tokens * bytes_per_token) / float(1024 * 1024)
                    estimated += 0.03 + mb * float(reload_ewma)
                if not missing:
                    return estimated
        # Use a conservative PCIe-scale estimate; actual exposed wait is still
        # measured by ready-before-use and scheduler_exposed_wait_ms.
        return (2.0 * float(bytes_count)) / 1.0e9 * 1000.0

    def _estimate_recovery_task_cpu_ms(self, task: _LayerKVRecoveryTask) -> float:
        if task.kind == "kvc":
            blocks = max(1, int(math.ceil(float(max(1, task.token_count)) / 16.0)))
            return 0.003 + 0.0002 * float(blocks)
        if task.kind == "expert":
            logical_count = len(task.logical_ids)
            return 0.02 * float(max(1, logical_count))
        return 0.0

    def _refresh_workload_stats(self, forward_batch: Any) -> None:
        pairs = (
            list(self._last_scheduled_req_lens)
            if self._current_forward_mode == "decode"
            and self._last_scheduled_req_lens
            else self._batch_req_indices_and_lens(forward_batch)
        )
        self._current_forward_req_lens_batch_id = id(forward_batch)
        self._current_forward_req_lens = list(pairs)
        self.stats.observed_batch_size = len(pairs)
        if pairs:
            self.stats.avg_prefix_len = sum(
                max(0, seq_len - 1) for _, seq_len in pairs
            ) / float(len(pairs))
        self.stats.kvc_bytes_per_token_all_layers = int(
            self._bytes_per_token_all_layers
        )

    def _reload_required_kvc(
        self,
        forward_batch: Any,
        selected_entries: Optional[List[_LayerKVResidencyEntry]] = None,
        *,
        strict: bool = True,
    ) -> bool:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return False
        if self.config.kvc_backend == "virtual-arena":
            self._refresh_kvc_residency_stats()
            return False
        selected: Optional[List[_LayerKVResidencyEntry]] = None
        if selected_entries is not None:
            selected = [
                entry
                for entry in selected_entries
                if entry.state == "offloaded" and entry.host_slot_list()
            ]
            if not selected:
                return False
        elif not self._has_offloaded_kvc_entries():
            return False
        if not self.physical_kvc_supported:
            self.stats.comparable = False
            self.stats.comparability_reason = self.unsupported_reason
            return False
        if self._bytes_per_token_all_layers <= 0:
            return False

        if selected_entries is None:
            self._prune_entries_for_active_lengths(forward_batch)
            with self._profile("profile_kvc_select_required_ms"):
                selected = self._select_required_offloaded_entries(forward_batch)
        if not selected:
            return False

        token_count = sum(entry.token_count for entry in selected)
        self.stats.kvc_reload_required_count += token_count
        if self.config.kvc_backend == "per-layer-arena":
            self.stats.kvc_allocator_available_before = self._allocator_available_size()
            self.stats.kvc_allocator_available_after = self._allocator_available_size()
            new_locs = None
        else:
            self.stats.kvc_allocator_available_before = self._allocator_available_size()
            new_locs = self._allocator.alloc(token_count)
            if new_locs is None:
                self.stats.kvc_physical_failure_count += 1
                self.stats.comparable = False
                self.stats.comparability_reason = (
                    "allocator failed to reload offloaded KVC"
                )
                raise RuntimeError("allocator failed to reload offloaded KVC")
            self.stats.kvc_allocator_available_after = self._allocator_available_size()

        host_slots = (
            []
            if self.config.kvc_backend == "per-layer-arena"
            else [slot for entry in selected for slot in entry.host_slot_list()]
        )
        try:
            self._ensure_host_store()
            if self.config.kvc_backend == "per-layer-arena":
                for entry in selected:
                    if entry.device_loc_list():
                        continue
                    new_entry_locs = None
                    if entry.evicted_device_locs:
                        new_entry_locs = self._reuse_evicted_per_layer_locs(
                            int(entry.layer_id), entry.evicted_device_locs
                        )
                    if new_entry_locs is None:
                        new_entry_locs = self._alloc_per_layer_locs(
                            int(entry.layer_id), int(entry.token_count)
                        )
                    if new_entry_locs is None:
                        if strict:
                            self.stats.kvc_physical_failure_count += 1
                            self.stats.comparable = False
                            self.stats.comparability_reason = (
                                "per-layer allocator failed to reload offloaded KVC"
                            )
                            raise RuntimeError(
                                "per-layer allocator failed to reload offloaded KVC"
                            )
                        return False
                    entry.device_locs = [int(x) for x in new_entry_locs]
                    entry.device_loc = int(new_entry_locs[0])
            async_copy = (
                self.config.kvc_scheduler == "async-deadline"
                and self._optimized_profile_enabled()
            )
            if self.config.kvc_backend == "per-layer-arena":
                elapsed_ms, start_event, ready_event = (
                    self._host_store.reload_per_layer(
                        selected,
                        stream=self._copy_stream,
                        async_copy=async_copy,
                    )
                )
            else:
                elapsed_ms, start_event, ready_event = self._host_store.reload(
                    host_slots,
                    new_locs,
                    stream=self._copy_stream,
                    async_copy=async_copy,
                )
            self.stats.layerkv_copy_event_record_count += 1
            with self._profile("profile_req_to_token_rewrite_ms"):
                if self.config.kvc_backend == "per-layer-arena":
                    # Per-layer arena reload restores each logical block to its
                    # existing layer-local device locations.  The attention
                    # metadata override is already keyed by those locations, so
                    # rebuilding a multi-million element new_locs tensor and
                    # rewriting req_to_token every decode step is pure
                    # controller overhead.
                    pass
                else:
                    self._rewrite_req_to_token_for_entries(selected, new_locs)
            new_locs_cpu = (
                []
                if self.config.kvc_backend == "per-layer-arena"
                else [int(x) for x in new_locs.detach().cpu().tolist()]
            )
            offset = 0
            for entry in selected:
                if self.config.kvc_backend != "per-layer-arena":
                    page_locs = new_locs_cpu[offset : offset + entry.token_count]
                    entry.device_locs = page_locs
                    entry.device_loc = page_locs[0] if page_locs else None
                else:
                    self._set_per_layer_token_slots(
                        int(entry.layer_id),
                        [int(entry.req_idx)] * int(entry.token_count),
                        entry.logical_positions(),
                        torch.tensor(
                            entry.device_loc_list(),
                            dtype=torch.int64,
                            device=self._allocator.device,
                        ),
                    )
                offset += entry.token_count
                entry.last_access_step = self._decode_step
                if async_copy and ready_event is not None:
                    entry.state = "reloading"
                    if self.config.kvc_backend == "per-layer-arena":
                        self._untrack_per_layer_offloaded_key(
                            (int(entry.layer_id), int(entry.req_idx), int(entry.pos)),
                            token_count=int(entry.token_count),
                        )
                        self._per_layer_offloaded_token_count_fast = max(
                            0,
                            self._per_layer_offloaded_token_count_fast
                            - int(entry.token_count),
                        )
                    entry.ready_start_event = start_event
                    entry.ready_event = ready_event
                    entry.ready_waited = False
                else:
                    entry.state = "resident"
                    if self.config.kvc_backend == "per-layer-arena":
                        self._untrack_per_layer_offloaded_key(
                            (int(entry.layer_id), int(entry.req_idx), int(entry.pos)),
                            token_count=int(entry.token_count),
                        )
                        self._per_layer_offloaded_token_count_fast = max(
                            0,
                            self._per_layer_offloaded_token_count_fast
                            - int(entry.token_count),
                        )
                        self._per_layer_resident_token_count_fast += int(
                            entry.token_count
                        )
                    if entry.host_slots is not None:
                        if self.config.kvc_backend == "per-layer-arena":
                            self._host_store.free_per_layer(
                                entry.layer_id, entry.host_slots
                            )
                        else:
                            self._host_store.free(entry.host_slots)
                    elif entry.host_slot is not None:
                        if self.config.kvc_backend == "per-layer-arena":
                            self._host_store.free_per_layer(
                                entry.layer_id, [int(entry.host_slot)]
                            )
                        else:
                            self._host_store.free([int(entry.host_slot)])
                    entry.host_slots = None
                    entry.host_slot = None
                    entry.evicted_device_locs = None
                self._sync_kvc_group_if_needed(entry)
            if async_copy and ready_event is not None:
                self._pending_kvc_reload_events.append(
                    _LayerKVPendingReload(start_event, ready_event, list(selected))
                )
        except Exception:
            self.stats.kvc_physical_failure_count += 1
            self.stats.comparable = False
            self.stats.comparability_reason = "KVC reload failed"
            raise

        if self.config.kvc_backend == "per-layer-arena":
            reload_mb = (
                token_count * self._bytes_per_kvc_token_per_layer() / float(1024 * 1024)
            )
        else:
            reload_mb = (
                token_count * self._bytes_per_token_all_layers / float(1024 * 1024)
            )
        self.stats.kvc_reload_count_total += token_count
        self.stats.kvc_reload_page_count_total += len(selected)
        self.stats.kvc_reload_mb_total += reload_mb
        self.stats.kvc_reload_ms += elapsed_ms
        if self.config.kvc_backend == "per-layer-arena":
            self.stats.kvc_per_layer_reload_count += token_count
            self.stats.kvc_per_layer_reload_mb_total += reload_mb
            self.stats.kvc_per_layer_reload_ms += elapsed_ms
            self._record_kvc_layer_cost_observations(
                selected, elapsed_ms, kind="reload"
            )
        self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
        self.stats.layerkv_kvc_reload_started += 1
        self.stats.layerkv_tasks_built += 1
        self._refresh_kvc_residency_stats()
        return True

    def _evict_kvc_to_target(
        self,
        forward_batch: Any,
        *,
        force_additional_tokens: Optional[int] = None,
        force_reason: str = "",
    ) -> None:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return
        if force_additional_tokens is None:
            effective_kvc_reclaim_mb = self._effective_kvc_reclaim_mb(forward_batch)
            if self.config.dynamic_pressure_from_kvc:
                runtime_pressure_mb = max(
                    0.0,
                    float(
                        self.stats.needed_pressure_mb
                        or self._dynamic_runtime_pressure_mb(forward_batch)
                    ),
                )
                if runtime_pressure_mb <= 1e-3:
                    self.stats.dynamic_pressure_skip_steps += 1
                    self.stats.kvc_topup_skip_count += 1
                    return
                self.stats.dynamic_pressure_active_steps += 1
                self.stats.kvc_topup_pressure_mb = runtime_pressure_mb
                if (
                    self.config.kvc_backend == "per-layer-arena"
                    and not self._per_layer_virtual_scratch_enabled()
                ):
                    # Per-layer arena KV for the current decode batch is a
                    # hard deadline dependency: every active prefix token is
                    # needed again on the next decode step. Evicting it here
                    # only creates immediate reload churn and does not turn
                    # into durable scheduler-visible capacity.
                    self.stats.kvc_scheduler_invisible_skip_count += 1
                    self.stats.kvc_topup_skip_count += 1
                    self.stats.kvc_eviction_skipped_count += 1
                    self.stats.target_limited_reason = (
                        "no_non_deadline_kvc_candidate"
                    )
                    return
            if effective_kvc_reclaim_mb <= 0:
                return
        else:
            force_additional_tokens = self._align_tokens_up(
                max(0, int(force_additional_tokens))
            )
            if force_additional_tokens <= 0:
                return
            if self.config.dynamic_pressure_from_kvc:
                runtime_pressure_mb = self._dynamic_runtime_pressure_mb(forward_batch)
                if (
                    runtime_pressure_mb <= 1e-3
                    and force_reason
                    not in ("pre_retract_decode_mem", "decode_prealloc_admission")
                ):
                    self.stats.kvc_evict_without_pressure_count += 1
                    return
            bytes_per_token = (
                self._bytes_per_kvc_token_per_layer()
                if self.config.kvc_backend == "per-layer-arena"
                else self._bytes_per_token_all_layers
            )
            if bytes_per_token <= 0:
                return
            effective_kvc_reclaim_mb = (
                force_additional_tokens * bytes_per_token / float(1024 * 1024)
            )
            self.stats.requested_total_reclaim_mb = max(
                float(self.stats.requested_total_reclaim_mb),
                float(effective_kvc_reclaim_mb),
            )
            self.stats.effective_kvc_reclaim_mb = float(effective_kvc_reclaim_mb)
            self.stats.planned_kvc_reclaim_mb = float(effective_kvc_reclaim_mb)
            if force_reason:
                self.stats.target_limited_reason = str(force_reason)
                self.stats.policy_semantics_reason = str(force_reason)
        if not self.physical_kvc_supported:
            self.stats.comparable = False
            self.stats.comparability_reason = self.unsupported_reason
            return
        if self._bytes_per_token_all_layers <= 0:
            return

        self._ensure_host_store()
        self._prune_entries_for_active_lengths(forward_batch)
        if (
            self._requires_per_layer_kvc_backend()
            and self.config.kvc_backend == "token-slot"
        ):
            self.stats.full_policy_semantics_supported = False
            self.stats.layerkv_kvc_backend_limited = True
            self.stats.policy_semantics_reason = (
                "layer_aware_kvc_requires_per_layer_arena_backend; "
                "token-slot can only evict the same token slots for all layers"
            )
            self.stats.comparable = False
            self.stats.comparability_reason = self.stats.policy_semantics_reason
        if self.config.kvc_backend == "per-layer-arena":
            self.stats.layerkv_kvc_backend_ready = self._per_layer_kvc_backend_ready()
        if force_additional_tokens is not None:
            target_tokens = self._offloaded_token_count() + int(force_additional_tokens)
        elif self._uses_layer_aware_kvc_plan() and self._planned_kvc_token_target > 0:
            target_tokens = sum(
                int(x) for x in self._planned_kvc_tokens_by_layer.values()
            )
        elif self.config.kvc_backend == "per-layer-arena":
            target_tokens = self._limit_offloaded_per_layer_tokens(
                effective_kvc_reclaim_mb
            )
        else:
            # Baseline KVC policies use layer-average eviction.  In SGLang's
            # token-to-KV pool, one token slot owns K/V for all layers, so
            # evicting N token slots reclaims the same N-token prefix/suffix
            # from every layer instead of making layer-specific choices.
            target_tokens = self._limit_offloaded_tokens(effective_kvc_reclaim_mb)
        pending_evict_tokens = (
            self._pending_evict_token_count()
            if self.config.kvc_backend == "virtual-arena"
            else 0
        )
        if self.config.dynamic_pressure_from_kvc and force_additional_tokens is None:
            offloaded_or_pending = self._offloaded_token_count() + pending_evict_tokens
            margin_tokens = self._dynamic_pressure_hysteresis_tokens()
            lower_bound = (
                int(target_tokens)
                if int(target_tokens) <= int(margin_tokens)
                else max(0, int(target_tokens) - int(margin_tokens))
            )
            if offloaded_or_pending > 0 and offloaded_or_pending >= lower_bound:
                self.stats.kvc_topup_skip_count += 1
                self.stats.kvc_eviction_skipped_count += 1
                return
            if margin_tokens > 0:
                target_tokens = int(target_tokens) + int(margin_tokens)
            self.stats.kvc_topup_apply_count += 1
        need_tokens = max(
            0, target_tokens - self._offloaded_token_count() - pending_evict_tokens
        )
        if need_tokens <= 0:
            self.stats.kvc_eviction_skipped_count += 1
            return

        with self._profile("profile_kvc_select_evict_ms"):
            selected = self._select_resident_entries_for_eviction(
                forward_batch, need_tokens
            )
        if not selected:
            self.stats.kvc_eviction_skipped_count += 1
            return

        token_count = sum(entry.token_count for entry in selected)
        if self.config.kvc_backend == "per-layer-arena":
            by_layer: Dict[int, int] = {}
            for entry in selected:
                by_layer[int(entry.layer_id)] = (
                    by_layer.get(int(entry.layer_id), 0) + entry.token_count
                )
            if by_layer:
                merged = dict(self._planned_kvc_tokens_by_layer)
                for layer_id, tokens in by_layer.items():
                    merged[layer_id] = max(int(merged.get(layer_id, 0)), int(tokens))
                self.stats.selected_kvc_tokens_by_layer = (
                    self._kvc_tokens_by_layer_json(merged)
                )
        else:
            self.stats.selected_kvc_tokens_by_layer = self._kvc_tokens_by_layer_json(
                token_count
            )
        host_slots = (
            self._host_store.alloc_per_layer(selected)
            if self.config.kvc_backend == "per-layer-arena"
            else self._host_store.alloc(token_count)
        )
        if host_slots is None:
            self.stats.kvc_physical_failure_count += 1
            self.stats.comparable = False
            self.stats.comparability_reason = "LayerKV host KVC store is full"
            if self.config.disallow_destructive_fallback:
                raise RuntimeError("LayerKV host KVC store is full")
            return

        old_locs = None
        if self.config.kvc_backend != "per-layer-arena":
            old_locs = torch.tensor(
                [loc for entry in selected for loc in entry.device_loc_list()],
                dtype=torch.int64,
                device=self._kv_pool.device,
            )
        async_evict = (
            self.config.kvc_backend == "virtual-arena"
            and force_additional_tokens is None
            and self.config.kvc_scheduler == "async-deadline"
            and self._optimized_profile_enabled()
            and self._copy_stream is not None
        )
        try:
            if self.config.kvc_backend == "per-layer-arena":
                elapsed_ms = self._host_store.backup_per_layer(selected, host_slots)
                self.stats.kvc_allocator_available_before = (
                    self._allocator_available_size()
                )
                self.stats.kvc_allocator_available_after = (
                    self._allocator_available_size()
                )
            elif async_evict:
                assert old_locs is not None
                with self._profile("profile_kvc_evict_staging_alloc_ms"):
                    start_event, ready_event, k_staging, v_staging = (
                        self._host_store.backup_to_staging_async(
                            old_locs, stream=self._copy_stream
                        )
                    )
                if start_event is None or ready_event is None:
                    raise RuntimeError("failed to issue async KVC eviction")
                elapsed_ms = 0.0
                offset = 0
                for entry in selected:
                    page_host_slots = host_slots[offset : offset + entry.token_count]
                    offset += entry.token_count
                    entry.state = "evicting"
                    entry.host_slots = [int(x) for x in page_host_slots]
                    entry.host_slot = (
                        int(page_host_slots[0]) if page_host_slots else None
                    )
                    entry.ready_start_event = start_event
                    entry.ready_event = ready_event
                    entry.ready_waited = False
                    entry.last_access_step = self._decode_step
                    self._sync_kvc_group_if_needed(entry)
                self._pending_kvc_evict_events.append(
                    _LayerKVPendingEviction(
                        start_event=start_event,
                        ready_event=ready_event,
                        entries=list(selected),
                        device_locs=old_locs,
                        host_slots=[int(x) for x in host_slots],
                        k_staging=k_staging,
                        v_staging=v_staging,
                        token_count=int(token_count),
                    )
                )
                self.stats.kvc_evict_async_count += int(token_count)
                self.stats.layerkv_tasks_built += 1
                for req_idx in {int(entry.req_idx) for entry in selected}:
                    self._mark_req_virtualized(req_idx)
                self._refresh_kvc_residency_stats()
                return
            else:
                assert old_locs is not None
                elapsed_ms = self._host_store.backup(old_locs, host_slots)
                self.stats.kvc_allocator_available_before = (
                    self._allocator_available_size()
                )
                self._allocator.free(old_locs)
                self.stats.kvc_allocator_available_after = (
                    self._allocator_available_size()
                )
                self.stats.kvc_allocator_free_count += 1
            offset = 0
            for entry in selected:
                page_host_slots = host_slots[offset : offset + entry.token_count]
                offset += entry.token_count
                entry.state = "offloaded"
                if self.config.kvc_backend == "virtual-arena":
                    self.stats.virtual_kvc_evict_count += int(entry.token_count)
                if self.config.kvc_backend == "per-layer-arena":
                    self._track_per_layer_offloaded_key(
                        (int(entry.layer_id), int(entry.req_idx), int(entry.pos)),
                        token_count=int(entry.token_count),
                    )
                    self._per_layer_resident_token_count_fast = max(
                        0,
                        self._per_layer_resident_token_count_fast
                        - int(entry.token_count),
                    )
                    self._per_layer_offloaded_token_count_fast += int(entry.token_count)
                    old_locs_for_free = entry.device_loc_list()
                    entry.evicted_device_locs = (
                        [int(x) for x in old_locs_for_free]
                        if old_locs_for_free
                        else None
                    )
                    if old_locs_for_free:
                        self._protect_per_layer_terminal_locs(
                            int(entry.layer_id), old_locs_for_free
                        )
                        self._free_per_layer_locs(
                            int(entry.layer_id), old_locs_for_free
                        )
                    entry.device_loc = None
                    entry.device_locs = None
                entry.host_slots = [int(x) for x in page_host_slots]
                entry.host_slot = int(page_host_slots[0]) if page_host_slots else None
                if self.config.kvc_backend != "per-layer-arena":
                    entry.device_loc = None
                    entry.device_locs = None
                entry.ready_event = None
                entry.ready_start_event = None
                entry.ready_waited = False
                entry.last_access_step = self._decode_step
                self._sync_kvc_group_if_needed(entry)
            if self.config.kvc_backend == "virtual-arena":
                for req_idx in {int(entry.req_idx) for entry in selected}:
                    self._mark_req_virtualized(req_idx)
        except Exception:
            if self.config.kvc_backend == "per-layer-arena":
                offset = 0
                for entry in selected:
                    slots = host_slots[offset : offset + entry.token_count]
                    offset += entry.token_count
                    self._host_store.free_per_layer(entry.layer_id, slots)
            else:
                self._host_store.free(host_slots)
            self.stats.kvc_physical_failure_count += 1
            self.stats.comparable = False
            self.stats.comparability_reason = "KVC eviction failed"
            raise

        self.stats.kvc_evict_count_total += token_count
        self.stats.kvc_evict_page_count_total += len(selected)
        self.stats.kvc_backup_ms += elapsed_ms
        if self.config.kvc_backend == "per-layer-arena":
            self.stats.kvc_per_layer_evict_count += token_count
            self.stats.kvc_per_layer_backup_ms += elapsed_ms
            self._record_kvc_layer_cost_observations(selected, elapsed_ms, kind="evict")
        self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
        self.stats.kvc_physical_cycle_count += 1
        self.stats.layerkv_tasks_built += 1
        self._refresh_kvc_residency_stats()

    def _limit_offloaded_tokens(self, reclaim_limit_mb: float) -> int:
        target_bytes = int(reclaim_limit_mb * 1024 * 1024)
        raw_tokens = max(1, target_bytes // self._bytes_per_token_all_layers)
        return max(self._page_size, self._align_tokens_up(raw_tokens))

    def _limit_offloaded_per_layer_tokens(self, reclaim_limit_mb: float) -> int:
        bytes_per_token = self._bytes_per_kvc_token_per_layer()
        if bytes_per_token <= 0:
            return 0
        target_bytes = int(reclaim_limit_mb * 1024 * 1024)
        raw_tokens = max(1, target_bytes // bytes_per_token)
        return max(self._page_size, self._align_tokens_up(raw_tokens))

    def _dynamic_pressure_hysteresis_tokens(self) -> int:
        bytes_per_token = (
            self._bytes_per_kvc_token_per_layer()
            if self.config.kvc_backend == "per-layer-arena"
            else self._bytes_per_token_all_layers
        )
        if bytes_per_token <= 0:
            return 0
        block_tokens = (
            self._per_layer_kvc_block_page_size()
            if self.config.kvc_backend == "per-layer-arena"
            else max(
                self._page_size,
                self._align_tokens_up(int(self.config.kvc_block_tokens or 0)),
            )
        )
        four_mb_tokens = int((4.0 * 1024.0 * 1024.0) // float(bytes_per_token))
        margin_tokens = max(int(block_tokens) * 4, four_mb_tokens)
        margin_tokens = self._align_tokens_up(margin_tokens)
        self.stats.kvc_topup_margin_mb = (
            margin_tokens * bytes_per_token / float(1024 * 1024)
        )
        return max(0, int(margin_tokens))

    def _offloaded_token_count(self) -> int:
        if self.config.kvc_backend == "per-layer-arena":
            self._restore_impossible_per_layer_offloads()
            if self._coresid_optimized_policy_enabled():
                return int(max(0, self._per_layer_offloaded_token_count_fast))
            return sum(
                entry.token_count
                for entry in self._per_layer_residency.values()
                if entry.state == "offloaded"
            )
        return sum(
            entry.token_count
            for entry in self._residency.values()
            if entry.state == "offloaded"
        )

    def _has_offloaded_kvc_entries(self) -> bool:
        if self.config.kvc_backend == "per-layer-arena":
            self._restore_impossible_per_layer_offloads()
            if self._coresid_optimized_policy_enabled():
                return int(max(0, self._per_layer_offloaded_token_count_fast)) > 0
            if self._per_layer_offloaded_keys_by_req:
                return True
            return any(
                entry.state == "offloaded" and bool(entry.host_slot_list())
                for entry in self._per_layer_residency.values()
            )
        return any(
            entry.state == "offloaded" and bool(entry.host_slot_list())
            for entry in self._residency.values()
        )

    def _pending_evict_token_count(self) -> int:
        return sum(
            int(pending.token_count) for pending in self._pending_kvc_evict_events
        )

    def _resident_token_count(self) -> int:
        if self.config.kvc_backend == "per-layer-arena":
            if self._coresid_optimized_policy_enabled():
                return int(max(0, self._per_layer_resident_token_count_fast))
            return sum(
                entry.token_count
                for entry in self._per_layer_residency.values()
                if entry.state == "resident"
            )
        return sum(
            entry.token_count
            for entry in self._residency.values()
            if entry.state == "resident"
        )

    def _refresh_kvc_residency_stats(self) -> None:
        offloaded = self._offloaded_token_count()
        resident = self._resident_token_count()
        self.stats.kvc_offloaded_token_count = offloaded
        self.stats.kvc_resident_token_count = resident
        self.stats.kvc_evict_pending_token_count = self._pending_evict_token_count()
        entries = (
            self._per_layer_residency
            if self.config.kvc_backend == "per-layer-arena"
            else self._residency
        )
        self.stats.kvc_residency_entry_count = len(entries)
        self.stats.kvc_per_layer_arena_entry_count = len(self._per_layer_residency)
        if (
            self.config.kvc_backend == "per-layer-arena"
            and self._coresid_optimized_policy_enabled()
        ):
            self.stats.kvc_per_layer_arena_resident_token_count = resident
            self.stats.kvc_per_layer_arena_offloaded_token_count = offloaded
            self.stats.kvc_offloaded_page_count = offloaded
            self.stats.kvc_resident_page_count = resident
        else:
            self.stats.kvc_per_layer_arena_resident_token_count = sum(
                entry.token_count
                for entry in self._per_layer_residency.values()
                if entry.state == "resident"
            )
            self.stats.kvc_per_layer_arena_offloaded_token_count = sum(
                entry.token_count
                for entry in self._per_layer_residency.values()
                if entry.state == "offloaded"
            )
            self.stats.kvc_offloaded_page_count = sum(
                1 for entry in entries.values() if entry.state == "offloaded"
            )
            self.stats.kvc_resident_page_count = sum(
                1 for entry in entries.values() if entry.state == "resident"
            )
        self.stats.kvc_page_size = self._page_size
        if self.config.kvc_backend == "per-layer-arena":
            per_layer_bytes = self._bytes_per_kvc_token_per_layer()
            self.stats.physical_kvc_reclaim_mb = (
                offloaded * per_layer_bytes / float(1024 * 1024)
            )
            self.stats.kvc_per_layer_arena_used_mb = (
                (resident + offloaded) * per_layer_bytes / float(1024 * 1024)
            )
            self._refresh_per_layer_allocator_stats()
        else:
            self.stats.physical_kvc_reclaim_mb = (
                offloaded * self._bytes_per_token_all_layers / float(1024 * 1024)
            )
        if self._host_store is not None:
            self.stats.kvc_host_used_tokens = self._host_store.used_count
            self.stats.kvc_host_backing_mb = self._host_store.capacity_mb
        self._refresh_physical_reclaim_peaks()
        self._refresh_resident_group_stats()

    def validate_kvc_state(self, forward_batch: Any = None) -> Dict[str, Any]:
        """Validate LayerKV-owned KVC residency bookkeeping.

        This is intentionally metadata-only.  It does not compare KV tensor
        values, but it catches stale host slots, duplicate ownership, req_to_token
        mismatches, and unsupported physical paths before policy results are
        treated as comparable.
        """

        stale_count = 0
        reasons: List[str] = []
        seen_host_slots = set()
        offloaded_count = 0
        resident_count = 0
        host_owned_count = 0
        zero_reconstruct_violations = 0
        zero_reconstruct_reasons: List[str] = []
        terminal_resident_count = 0
        terminal_offloaded_count = 0
        terminal_metadata_mapped_count = 0

        active_lens = None
        if forward_batch is not None:
            active_lens = {
                req_idx: seq_len
                for req_idx, seq_len in self._batch_req_indices_and_lens(forward_batch)
            }

        if (
            self._host_store is not None
            and self.config.kvc_backend == "per-layer-arena"
        ):
            host_used_slots = {
                (self._host_store.start_layer + layer_offset, int(slot))
                for layer_offset, slots in enumerate(self._host_store.layer_used_slots)
                for slot in slots
            }
        else:
            host_used_slots = (
                set(self._host_store.used_slots)
                if self._host_store is not None
                else set()
            )

        if self.config.kvc_backend == "per-layer-arena":
            iter_entries = [
                ((entry.layer_id, entry.req_idx, entry.pos), entry)
                for entry in self._per_layer_residency.values()
            ]
        else:
            iter_entries = [
                ((entry.req_idx, entry.pos), entry)
                for entry in self._residency.values()
            ]

        for key, entry in iter_entries:
            req_idx, pos = int(entry.req_idx), int(entry.pos)
            expected_key = (
                (int(entry.layer_id), req_idx, pos)
                if self.config.kvc_backend == "per-layer-arena"
                else (req_idx, pos)
            )
            if key != expected_key:
                stale_count += 1
                reasons.append("entry_key_mismatch")
            if (
                active_lens is not None
                and req_idx in active_lens
                and pos + entry.token_count > active_lens[req_idx]
            ):
                stale_count += 1
                reasons.append("entry_beyond_active_length")

            if entry.state == "offloaded":
                offloaded_count += entry.token_count
                if self.config.kvc_backend == "per-layer-arena":
                    evicted_locs = entry.evicted_device_locs or []
                    if len(evicted_locs) != int(entry.token_count):
                        zero_reconstruct_violations += 1
                        zero_reconstruct_reasons.append(
                            "offloaded_missing_terminal_locs"
                        )
                    else:
                        terminal_offloaded_count += int(entry.token_count)
                        allocated = self._per_layer_arena_allocated_locs.get(
                            int(entry.layer_id), set()
                        )
                        if any(int(loc) in allocated for loc in evicted_locs):
                            zero_reconstruct_violations += int(entry.token_count)
                            zero_reconstruct_reasons.append(
                                "offloaded_terminal_loc_reused"
                            )
                host_slots = entry.host_slot_list()
                host_owned_count += len(host_slots)
                if not host_slots:
                    stale_count += 1
                    reasons.append("offloaded_missing_host_slot")
                for host_slot in host_slots:
                    host_key = (
                        (int(entry.layer_id), int(host_slot))
                        if self.config.kvc_backend == "per-layer-arena"
                        else int(host_slot)
                    )
                    if host_key in seen_host_slots:
                        stale_count += 1
                        reasons.append("duplicate_host_slot")
                    elif (
                        self._host_store is not None and host_key not in host_used_slots
                    ):
                        stale_count += 1
                        reasons.append("host_slot_not_marked_used")
                    seen_host_slots.add(host_key)
                if (
                    self.config.kvc_backend != "per-layer-arena"
                    and entry.device_loc is not None
                ):
                    stale_count += 1
                    reasons.append("offloaded_has_device_loc")
                if len(host_slots) != entry.token_count:
                    stale_count += 1
                    reasons.append("offloaded_host_slot_count_mismatch")
            elif entry.state in ("resident", "reloading", "evicting"):
                if entry.state in ("reloading", "evicting"):
                    host_owned_count += len(entry.host_slot_list())
                if entry.state == "resident":
                    resident_count += entry.token_count
                    if self.config.kvc_backend == "per-layer-arena":
                        terminal_resident_count += int(entry.token_count)
                device_locs = entry.device_loc_list()
                if not device_locs:
                    stale_count += 1
                    reasons.append("resident_missing_device_loc")
                if entry.state == "resident" and entry.host_slot is not None:
                    stale_count += 1
                    reasons.append("resident_has_host_slot")
                if entry.state == "evicting" and not entry.host_slot_list():
                    stale_count += 1
                    reasons.append("evicting_missing_host_slot")
                if len(device_locs) != entry.token_count:
                    stale_count += 1
                    reasons.append("resident_device_loc_count_mismatch")
                if (
                    active_lens is not None
                    and req_idx in active_lens
                    and self._req_to_token_pool is not None
                    and device_locs
                    and pos + entry.token_count <= active_lens[req_idx]
                ):
                    try:
                        if (
                            self.config.kvc_backend == "per-layer-arena"
                            and int(entry.layer_id)
                            in self._per_layer_req_to_token_overrides
                        ):
                            table = self._per_layer_req_to_token_overrides[
                                int(entry.layer_id)
                            ]
                            if entry.state == "resident":
                                terminal_metadata_mapped_count += int(
                                    entry.token_count
                                )
                        else:
                            table = self._req_to_token_pool.req_to_token
                        table_locs = [
                            int(x)
                            for x in table[req_idx, entry.logical_positions()]
                            .detach()
                            .cpu()
                            .tolist()
                        ]
                        if table_locs != device_locs:
                            stale_count += 1
                            reasons.append("resident_req_to_token_mismatch")
                            if self.config.kvc_backend == "per-layer-arena":
                                zero_reconstruct_violations += int(
                                    entry.token_count
                                )
                                zero_reconstruct_reasons.append(
                                    "resident_terminal_mapping_mismatch"
                                )
                    except Exception:
                        stale_count += 1
                        reasons.append("resident_req_to_token_check_failed")
                        if self.config.kvc_backend == "per-layer-arena":
                            zero_reconstruct_violations += int(entry.token_count)
                            zero_reconstruct_reasons.append(
                                "resident_terminal_mapping_check_failed"
                            )
            else:
                stale_count += 1
                reasons.append("unknown_entry_state")

        if (
            self._host_store is not None
            and self._host_store.used_count != host_owned_count
        ):
            stale_count += abs(self._host_store.used_count - host_owned_count)
            reasons.append("host_used_count_mismatch")

        self.stats.kvc_residency_entry_count = len(iter_entries)
        self.stats.kvc_stale_entry_count = stale_count
        self.stats.kvc_offloaded_token_count = offloaded_count
        self.stats.kvc_resident_token_count = resident_count
        self.stats.kvc_host_used_tokens = (
            self._host_store.used_count if self._host_store is not None else 0
        )
        self.stats.kvc_terminal_resident_token_count = terminal_resident_count
        self.stats.kvc_terminal_offloaded_token_count = terminal_offloaded_count
        self.stats.kvc_terminal_metadata_mapped_token_count = (
            terminal_metadata_mapped_count
        )
        self.stats.kvc_zero_reconstruct_violation_count = (
            zero_reconstruct_violations
        )
        self.stats.kvc_zero_reconstruct_guard_pass = (
            zero_reconstruct_violations == 0
        )
        self.stats.kvc_zero_reconstruct_guard_reason = ";".join(
            sorted(set(zero_reconstruct_reasons))
        )

        guard_pass = stale_count == 0
        needs_kvc_reclaim = (
            self.config.mode == "kvc-only" and self.config.reclaim_limit_mb > 0
        )
        needs_kvc_reclaim = needs_kvc_reclaim or self.stats.planned_kvc_reclaim_mb > 0
        needs_kvc_reclaim = needs_kvc_reclaim or self.stats.effective_kvc_reclaim_mb > 0
        if needs_kvc_reclaim and not self.physical_kvc_supported:
            guard_pass = False
            reasons.append(self.unsupported_reason or "physical_kvc_unsupported")

        self.stats.kvc_guard_pass = guard_pass
        if not self.stats.kvc_zero_reconstruct_guard_pass:
            self.stats.kvc_guard_pass = False
            reasons.extend(zero_reconstruct_reasons)
        self.stats.kvc_guard_reason = ";".join(sorted(set(reasons)))
        return {
            "kvc_guard_pass": self.stats.kvc_guard_pass,
            "kvc_guard_reason": self.stats.kvc_guard_reason,
            "kvc_zero_reconstruct_guard_pass": self.stats.kvc_zero_reconstruct_guard_pass,
            "kvc_zero_reconstruct_guard_reason": self.stats.kvc_zero_reconstruct_guard_reason,
            "kvc_zero_reconstruct_violation_count": self.stats.kvc_zero_reconstruct_violation_count,
            "kvc_stale_entry_count": self.stats.kvc_stale_entry_count,
            "kvc_residency_entry_count": self.stats.kvc_residency_entry_count,
            "kvc_host_used_tokens": self.stats.kvc_host_used_tokens,
            "kvc_offloaded_token_count": self.stats.kvc_offloaded_token_count,
            "kvc_resident_token_count": self.stats.kvc_resident_token_count,
        }

    def validate_expert_state(self) -> Dict[str, Any]:
        """Validate fixed-slot expert residency metadata.

        The expert recovery path must materialize weights directly into the
        module's terminal physical slots and expose residency only through the
        logical-to-physical metadata map.  This check is intentionally limited
        to summary/validation paths; it must not run on the decode hot path.
        """

        violations = 0
        reasons: List[str] = []
        terminal_slot_count = 0
        terminal_offloaded_count = 0
        terminal_metadata_mapped_count = 0

        if not self._expert_layers:
            return {
                "expert_zero_reconstruct_guard_pass": self.stats.expert_zero_reconstruct_guard_pass,
                "expert_zero_reconstruct_guard_reason": self.stats.expert_zero_reconstruct_guard_reason,
                "expert_zero_reconstruct_violation_count": self.stats.expert_zero_reconstruct_violation_count,
                "expert_terminal_slot_count": self.stats.expert_terminal_slot_count,
                "expert_terminal_offloaded_count": self.stats.expert_terminal_offloaded_count,
                "expert_terminal_metadata_mapped_count": self.stats.expert_terminal_metadata_mapped_count,
            }

        for layer_id, state in sorted(self._expert_layers.items()):
            layer_id = int(layer_id)
            full_num_experts = int(state.full_num_experts)
            slot_capacity = int(state.slot_capacity)
            terminal_slot_count += max(0, slot_capacity)
            terminal_offloaded_count += max(
                0, full_num_experts - len(state.logical_to_slot)
            )

            if slot_capacity <= 0 or slot_capacity > full_num_experts:
                violations += 1
                reasons.append("expert_slot_capacity_out_of_range")

            for name in state.param_names:
                try:
                    tensor = getattr(state.module, name).data
                    if int(tensor.shape[0]) != slot_capacity:
                        violations += 1
                        reasons.append("expert_param_not_terminal_slot_matrix")
                except Exception:
                    violations += 1
                    reasons.append("expert_param_slot_matrix_check_failed")

            seen_slots: Set[int] = set()
            for logical_id, slot_id in state.logical_to_slot.items():
                logical_id = int(logical_id)
                slot_id = int(slot_id)
                if logical_id < 0 or logical_id >= full_num_experts:
                    violations += 1
                    reasons.append("expert_logical_id_out_of_range")
                if slot_id < 0 or slot_id >= slot_capacity:
                    violations += 1
                    reasons.append("expert_slot_id_out_of_range")
                if slot_id in seen_slots:
                    violations += 1
                    reasons.append("expert_duplicate_physical_slot")
                seen_slots.add(slot_id)
                if int(state.slot_to_logical.get(slot_id, -1)) != logical_id:
                    violations += 1
                    reasons.append("expert_slot_reverse_map_mismatch")

            for slot_id, logical_id in state.slot_to_logical.items():
                slot_id = int(slot_id)
                logical_id = int(logical_id)
                if int(state.logical_to_slot.get(logical_id, -1)) != slot_id:
                    violations += 1
                    reasons.append("expert_logical_reverse_map_mismatch")

            remap = state.remap_tensor
            if remap is None or int(remap.numel()) != full_num_experts:
                violations += 1
                reasons.append("expert_remap_tensor_missing_or_wrong_shape")
                remap_values: List[int] = []
            else:
                try:
                    remap_values = [int(x) for x in remap.detach().cpu().tolist()]
                except Exception:
                    violations += 1
                    reasons.append("expert_remap_tensor_read_failed")
                    remap_values = []

            if len(remap_values) == full_num_experts:
                for logical_id in range(full_num_experts):
                    expected = int(state.logical_to_slot.get(logical_id, -1))
                    if int(remap_values[logical_id]) != expected:
                        violations += 1
                        reasons.append("expert_remap_slot_map_mismatch")
                    elif expected >= 0:
                        terminal_metadata_mapped_count += 1

            for logical_id in range(full_num_experts):
                if logical_id in state.logical_to_slot:
                    continue
                if logical_id in state.cpu_params:
                    continue
                if self._expert_global_cpu_backing and self._global_expert_backing(
                    layer_id, logical_id
                ) is not None:
                    continue
                violations += 1
                reasons.append("expert_offloaded_missing_cpu_backing")

        self.stats.expert_terminal_slot_count = terminal_slot_count
        self.stats.expert_terminal_offloaded_count = terminal_offloaded_count
        self.stats.expert_terminal_metadata_mapped_count = (
            terminal_metadata_mapped_count
        )
        self.stats.expert_zero_reconstruct_violation_count = violations
        self.stats.expert_zero_reconstruct_guard_pass = violations == 0
        self.stats.expert_zero_reconstruct_guard_reason = ";".join(
            sorted(set(reasons))
        )
        if violations:
            self.stats.expert_guard_pass = False
            existing = self.stats.expert_guard_reason
            combined = sorted(set(([existing] if existing else []) + reasons))
            self.stats.expert_guard_reason = ";".join(combined)
        return {
            "expert_zero_reconstruct_guard_pass": self.stats.expert_zero_reconstruct_guard_pass,
            "expert_zero_reconstruct_guard_reason": self.stats.expert_zero_reconstruct_guard_reason,
            "expert_zero_reconstruct_violation_count": self.stats.expert_zero_reconstruct_violation_count,
            "expert_terminal_slot_count": self.stats.expert_terminal_slot_count,
            "expert_terminal_offloaded_count": self.stats.expert_terminal_offloaded_count,
            "expert_terminal_metadata_mapped_count": self.stats.expert_terminal_metadata_mapped_count,
        }

    def _batch_req_indices_and_lens(self, forward_batch: Any) -> List[Tuple[int, int]]:
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if seq_lens_cpu is None or req_pool_indices is None:
            return list(self._last_scheduled_req_lens)
        if isinstance(seq_lens_cpu, torch.Tensor):
            seq_lens = [int(x) for x in seq_lens_cpu.cpu().tolist()]
        else:
            seq_lens = [int(x) for x in seq_lens_cpu]
        req_indices = [int(x) for x in req_pool_indices.detach().cpu().tolist()]
        return list(zip(req_indices, seq_lens))

    def _drop_entries_for_reqs(self, forward_batch: Any) -> None:
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if req_pool_indices is None:
            return
        req_indices = set(int(x) for x in req_pool_indices.detach().cpu().tolist())
        if not req_indices:
            return
        for req_idx in req_indices:
            if req_idx in self._kvc_evict_cursors:
                self._kvc_evict_cursors.pop(req_idx, None)
                self.stats.kvc_evict_cursor_reset_count += 1
            for cursor_key in list(self._kvc_evict_cursors_by_layer):
                if int(cursor_key[1]) == req_idx:
                    self._kvc_evict_cursors_by_layer.pop(cursor_key, None)
                    self.stats.kvc_layerwise_evict_cursor_reset_count += 1
        to_drop = [key for key in self._residency if key[0] in req_indices]
        self._drop_residency_keys(to_drop)
        per_layer_to_drop = [
            key for key in self._per_layer_residency if key[1] in req_indices
        ]
        self._drop_per_layer_residency_keys(per_layer_to_drop)

    def on_request_finished(self, req: Any) -> None:
        """Drop LayerKV-owned metadata/backing for a completed request.

        SGLang still owns the canonical req_to_token/KV allocator release. This
        hook only removes LayerKV's offloaded residency records and CPU backing
        before the request pool index is cleared by release_kv_cache.
        """

        req_pool_idx = getattr(req, "req_pool_idx", None)
        if req_pool_idx is None:
            return
        try:
            req_idx = int(req_pool_idx)
        except (TypeError, ValueError):
            return
        if req_idx in self._kvc_evict_cursors:
            self._kvc_evict_cursors.pop(req_idx, None)
            self.stats.kvc_evict_cursor_reset_count += 1
        to_drop = [key for key in self._residency if key[0] == req_idx]
        if self.config.kvc_backend == "per-layer-arena":
            per_layer_to_drop = list(
                self._per_layer_owned_keys_by_req.pop(req_idx, set())
            )
        else:
            per_layer_to_drop = [
                key for key in self._per_layer_residency if key[1] == req_idx
            ]
        if not to_drop and not per_layer_to_drop:
            return
        if self.config.kvc_backend == "virtual-arena":
            req.layerkv_virtualized_kvc = True
            req.skip_radix_cache_insert = True
            return
        token_count = 0
        for key in to_drop:
            entry = self._residency.get(key)
            if entry is not None:
                token_count += int(entry.token_count)
        for key in per_layer_to_drop:
            entry = self._per_layer_residency.get(key)
            if entry is not None:
                token_count += int(entry.token_count)
        self._drop_residency_keys(to_drop)
        self._drop_per_layer_residency_keys(per_layer_to_drop)
        if self.config.kvc_backend == "per-layer-arena":
            self._per_layer_owned_req_indices.discard(req_idx)
            try:
                req.layerkv_per_layer_cleaned = True
            except Exception:
                pass
        self.stats.kvc_finished_req_cleanup_count += 1
        self.stats.kvc_finished_req_cleanup_token_count += token_count

    def _prune_entries_for_active_lengths(self, forward_batch: Any) -> None:
        active_lens = {
            req_idx: seq_len
            for req_idx, seq_len in self._batch_req_indices_and_lens(forward_batch)
        }
        if not active_lens:
            return
        to_drop = [
            key
            for key, entry in self._residency.items()
            if key[0] in active_lens
            and entry.pos + entry.token_count > active_lens[key[0]]
        ]
        self._drop_residency_keys(to_drop)
        per_layer_to_drop = [
            key
            for key, entry in self._per_layer_residency.items()
            if key[1] in active_lens
            and entry.pos + entry.token_count > active_lens[key[1]]
        ]
        self._drop_per_layer_residency_keys(per_layer_to_drop)
        for req_idx in list(self._kvc_evict_cursors):
            if req_idx not in active_lens:
                continue
            max_cursor = self._align_tokens_down(max(0, active_lens[req_idx] - 1))
            if self._kvc_evict_cursors[req_idx] > max_cursor:
                self._kvc_evict_cursors[req_idx] = 0
                self.stats.kvc_evict_cursor_reset_count += 1
        for key in list(self._kvc_evict_cursors_by_layer):
            _layer_id, req_idx = key
            if req_idx not in active_lens:
                continue
            max_cursor = self._align_tokens_down(max(0, active_lens[req_idx] - 1))
            if self._kvc_evict_cursors_by_layer[key] > max_cursor:
                self._kvc_evict_cursors_by_layer[key] = 0
                self.stats.kvc_layerwise_evict_cursor_reset_count += 1

    def _drop_residency_keys(self, keys: List[Tuple[int, int]]) -> None:
        if not keys:
            return
        affected_entries = [
            self._residency[key] for key in keys if key in self._residency
        ]
        self._invalidate_virtual_caches_for_layers(
            self._virtual_cache_layers_for_entries(affected_entries)
        )
        if any(
            self._residency.get(key) is not None
            and self._residency[key].state == "evicting"
            for key in keys
        ):
            self._finalize_kvc_evictions(block=True)
        host_slots = []
        for key in keys:
            entry = self._residency.pop(key, None)
            if entry is not None:
                if entry.ready_event is not None and not entry.ready_event.query():
                    entry.ready_event.synchronize()
                host_slots.extend(entry.host_slot_list())
                self._remove_resident_group(
                    "kvc", -1, (int(entry.req_idx), int(entry.pos))
                )
        if host_slots and self._host_store is not None:
            self._host_store.free(host_slots)
        self._refresh_kvc_residency_stats()

    def _drop_per_layer_residency_keys(self, keys: List[Tuple[int, int, int]]) -> None:
        if not keys:
            return
        free_locs_by_layer: Dict[int, List[int]] = {}
        host_slots_by_layer: Dict[int, List[int]] = {}
        released_protected_locs = False
        remove_groups = not (
            self.config.kvc_backend == "per-layer-arena"
            and self._coresid_optimized_policy_enabled()
        )
        for key in keys:
            self._untrack_per_layer_offloaded_key(key)
            entry = self._per_layer_residency.pop(key, None)
            if entry is not None:
                if entry.state == "offloaded":
                    self._per_layer_offloaded_token_count_fast = max(
                        0,
                        self._per_layer_offloaded_token_count_fast
                        - int(entry.token_count),
                    )
                elif entry.state in ("resident", "reloading"):
                    self._per_layer_resident_token_count_fast = max(
                        0,
                        self._per_layer_resident_token_count_fast
                        - int(entry.token_count),
                    )
                if entry.ready_event is not None and not entry.ready_event.query():
                    entry.ready_event.synchronize()
                device_locs = entry.device_loc_list()
                if device_locs:
                    free_locs_by_layer.setdefault(int(entry.layer_id), []).extend(
                        device_locs
                    )
                elif (
                    self.config.kvc_backend == "per-layer-arena"
                    and entry.evicted_device_locs
                ):
                    self._release_protected_per_layer_locs(
                        int(entry.layer_id),
                        [int(loc) for loc in entry.evicted_device_locs],
                        refresh=False,
                    )
                    released_protected_locs = True
                host_slots = entry.host_slot_list()
                if host_slots:
                    host_slots_by_layer.setdefault(int(entry.layer_id), []).extend(
                        host_slots
                    )
                if remove_groups:
                    self._remove_resident_group(
                        "kvc",
                        int(entry.layer_id),
                        (int(entry.layer_id), int(entry.req_idx), int(entry.pos)),
                    )
        for layer_id, device_locs in free_locs_by_layer.items():
            self._free_per_layer_locs(int(layer_id), device_locs, refresh=False)
        if free_locs_by_layer or released_protected_locs:
            self._refresh_per_layer_allocator_stats()
        if self._host_store is not None:
            for layer_id, host_slots in host_slots_by_layer.items():
                if host_slots:
                    self._host_store.free_per_layer(int(layer_id), host_slots)
        self._refresh_kvc_residency_stats()

    def _select_required_offloaded_entries(
        self, forward_batch: Any
    ) -> List[_LayerKVResidencyEntry]:
        if self.config.kvc_backend == "per-layer-arena":
            return self._select_required_offloaded_per_layer_entries(forward_batch)
        selected: List[_LayerKVResidencyEntry] = []
        seen = set()
        for req_idx, seq_len in self._batch_req_indices_and_lens(forward_batch):
            # Decode writes the current token at seq_len - 1 inside the forward.
            required_prefix_len = max(0, seq_len - 1)
            required_prefix_len = self._align_tokens_down(required_prefix_len)
            for pos in range(0, required_prefix_len, self._page_size):
                key = (req_idx, pos)
                entry = self._residency.get(key)
                if entry is None or entry.state != "offloaded":
                    continue
                if key in seen:
                    continue
                seen.add(key)
                selected.append(entry)
        return selected

    def _select_required_offloaded_per_layer_entries(
        self, forward_batch: Any
    ) -> List[_LayerKVResidencyEntry]:
        selected: List[_LayerKVResidencyEntry] = []
        self._restore_impossible_per_layer_offloads()
        if not self._per_layer_offloaded_keys_by_req:
            return selected
        active_lens = {
            int(req_idx): self._align_tokens_down(max(0, int(seq_len) - 1))
            for req_idx, seq_len in self._batch_req_indices_and_lens(forward_batch)
        }
        stale: List[Tuple[int, int, int]] = []
        scanned = 0
        for req_idx, required_prefix_len in active_lens.items():
            req_idx = int(req_idx)
            req_keys = self._per_layer_offloaded_keys_by_req.get(req_idx)
            if not req_keys:
                continue
            req_layers = sorted({int(key[0]) for key in req_keys})
            if not req_layers:
                continue
            self.stats.kvc_layerwise_required_index_hit_count += len(req_layers)
            for layer_id in req_layers:
                sorted_keys, positions = self._sorted_per_layer_offloaded_keys_for_req_layer(
                    req_idx, layer_id
                )
                if not sorted_keys:
                    continue
                limit = bisect.bisect_right(positions, max(0, required_prefix_len - 1))
                if limit <= 0:
                    continue
                for key in sorted_keys[:limit]:
                    scanned += 1
                    entry = self._per_layer_residency.get(key)
                    if entry is None or entry.state != "offloaded":
                        stale.append(key)
                        continue
                    if entry.pos + entry.token_count > required_prefix_len:
                        continue
                    selected.append(entry)
        self.stats.kvc_layerwise_required_index_scan_count += scanned
        self.stats.kvc_required_scanned_keys += scanned
        for key in stale:
            self._untrack_per_layer_offloaded_key(key)
        self.stats.kvc_layerwise_required_index_stale_count += len(stale)
        selected.sort(key=lambda entry: (entry.layer_id, entry.pos, entry.req_idx))
        self.stats.kvc_layerwise_required_selected_token_count += sum(
            int(entry.token_count) for entry in selected
        )
        return selected

    def _select_resident_entries_for_eviction(
        self, forward_batch: Any, max_tokens: int
    ) -> List[_LayerKVResidencyEntry]:
        if self.config.kvc_backend == "per-layer-arena":
            return self._select_per_layer_resident_entries_for_eviction(
                forward_batch, max_tokens
            )
        self.stats.kvc_eviction_index_fallback_count += 1
        table = self._req_to_token_pool.req_to_token
        pairs = self._batch_req_indices_and_lens(forward_batch)
        page_size = max(1, int(self._page_size))
        max_tokens = self._align_tokens_down(max_tokens)
        target_pages = max_tokens // page_size
        if target_pages <= 0:
            return []
        block_tokens = max(
            page_size, self._align_tokens_up(self.config.kvc_block_tokens)
        )
        block_pages = max(1, block_tokens // page_size)
        if target_pages > block_pages:
            target_pages -= target_pages % block_pages
            target_pages = max(block_pages, target_pages)

        evictable: List[Tuple[int, int, int]] = []
        for req_idx, seq_len in pairs:
            evictable_len = self._align_tokens_down(max(0, seq_len - 1))
            if evictable_len <= 0:
                continue
            cursor = self._align_tokens_down(self._kvc_evict_cursors.get(req_idx, 0))
            if cursor >= evictable_len:
                cursor = 0
                self._kvc_evict_cursors[req_idx] = 0
                self.stats.kvc_evict_cursor_reset_count += 1
            evictable.append((req_idx, evictable_len, cursor))
        if not evictable:
            return []

        def build_cursor_meta() -> List[Tuple[int, int]]:
            total_pages = sum(
                evictable_len // page_size for _req, evictable_len, _cur in evictable
            )
            scan_budget_pages = min(
                total_pages, max(target_pages * 2, target_pages + 32)
            )
            offsets = [0 for _ in evictable]
            page_meta: List[Tuple[int, int]] = []
            while len(page_meta) < scan_budget_pages:
                progressed = False
                for idx, (req_idx, evictable_len, cursor) in enumerate(evictable):
                    pos = cursor + offsets[idx]
                    offsets[idx] += page_size
                    if pos >= evictable_len:
                        continue
                    page_meta.append((req_idx, pos))
                    progressed = True
                    if len(page_meta) >= scan_budget_pages:
                        break
                if not progressed:
                    break
            if page_meta:
                self.stats.kvc_evict_cursor_hit_count += 1
            return page_meta

        def build_full_meta() -> List[Tuple[int, int]]:
            return [
                (req_idx, pos)
                for req_idx, evictable_len, _cursor in evictable
                for pos in range(0, evictable_len, page_size)
            ]

        def materialize_candidates(
            page_meta: List[Tuple[int, int]],
        ) -> List[_LayerKVResidencyEntry]:
            if not page_meta:
                return []
            self.stats.kvc_evict_candidate_scan_tokens += len(page_meta) * page_size
            flat_req_indices: List[int] = []
            flat_positions: List[int] = []
            for req_idx, pos in page_meta:
                for page_pos in range(pos, pos + page_size):
                    flat_req_indices.append(req_idx)
                    flat_positions.append(page_pos)
            req_tensor = torch.tensor(
                flat_req_indices, dtype=torch.int64, device=table.device
            )
            pos_tensor = torch.tensor(
                flat_positions, dtype=torch.int64, device=table.device
            )
            flat_locs = table[req_tensor, pos_tensor].detach().cpu().tolist()
            candidates: List[_LayerKVResidencyEntry] = []
            seen_locs = set()
            loc_offset = 0
            for req_idx, pos in page_meta:
                locs = [int(x) for x in flat_locs[loc_offset : loc_offset + page_size]]
                loc_offset += page_size
                key = (req_idx, pos)
                existing = self._residency.get(key)
                if existing is not None and existing.state in (
                    "offloaded",
                    "reloading",
                    "evicting",
                ):
                    continue
                if any(loc <= 0 for loc in locs) or any(
                    loc in seen_locs for loc in locs
                ):
                    continue
                if page_size > 1:
                    base = locs[0]
                    expected = list(range(base, base + page_size))
                    if locs != expected or base % page_size != 0:
                        self.stats.kvc_page_alignment_violation_count += 1
                        continue
                seen_locs.update(locs)
                if existing is None:
                    existing = _LayerKVResidencyEntry(
                        req_idx=req_idx,
                        pos=pos,
                        state="resident",
                        device_loc=locs[0],
                        device_locs=locs,
                        page_size=page_size,
                        last_access_step=self._decode_step,
                    )
                    self._residency[key] = existing
                else:
                    existing.state = "resident"
                    existing.device_loc = locs[0]
                    existing.device_locs = locs
                    existing.page_size = page_size
                    existing.last_access_step = self._decode_step
                self._sync_kvc_group_if_needed(existing)
                candidates.append(existing)
            candidates.sort(key=lambda x: (x.pos, x.req_idx))
            return candidates

        candidates = materialize_candidates(build_cursor_meta())
        if candidates:
            self.stats.kvc_eviction_index_hit_count += 1
        if len(candidates) < target_pages:
            self.stats.kvc_eviction_index_rebuild_count += 1
            candidates = materialize_candidates(build_full_meta())

        take_pages = min(target_pages, len(candidates))
        if take_pages <= 0:
            return []
        selected = candidates[:take_pages]
        self.stats.kvc_evict_candidate_selected_tokens += sum(
            entry.token_count for entry in selected
        )
        for entry in selected:
            next_pos = entry.pos + entry.token_count
            old_cursor = self._kvc_evict_cursors.get(entry.req_idx, 0)
            if next_pos > old_cursor:
                self._kvc_evict_cursors[entry.req_idx] = next_pos
        return selected

    def _select_common_per_layer_resident_entries_for_eviction(
        self, forward_batch: Any, common_tokens: int
    ) -> List[_LayerKVResidencyEntry]:
        layer_ids = [int(layer_id) for layer_id in self._kvc_layer_ids()]
        common_tokens = max(0, int(common_tokens))
        if not layer_ids or common_tokens <= 0:
            return []
        scratch_capacity = self._per_layer_virtual_scratch_capacity_tokens()
        if scratch_capacity > 0:
            current_max = max(
                int(self._per_layer_offloaded_token_count_by_layer.get(layer_id, 0))
                for layer_id in layer_ids
            )
            common_tokens = min(common_tokens, max(0, scratch_capacity - current_max))
            if common_tokens <= 0:
                self.stats.virtual_kvc_scratch_overflow_count += 1
                self.stats.kvc_eviction_skipped_count += 1
                return []
        block_size = max(1, int(self.config.kvc_block_tokens or 1))
        selected: List[_LayerKVResidencyEntry] = []
        selected_common = 0

        for req_idx, seq_len in self._batch_req_indices_and_lens(forward_batch):
            req_idx = int(req_idx)
            evictable_len = self._align_tokens_down(max(0, int(seq_len) - 1))
            if evictable_len <= 0:
                continue
            cursor_key = (-1, req_idx)
            cursor = self._align_tokens_down(
                self._kvc_evict_cursors_by_layer.get(cursor_key, 0)
            )
            if cursor >= evictable_len:
                cursor = 0
                self._kvc_evict_cursors_by_layer[cursor_key] = 0
                self.stats.kvc_layerwise_evict_cursor_reset_count += 1
            pos = cursor
            while pos < evictable_len and selected_common < common_tokens:
                run_len = min(block_size, evictable_len - pos)
                run_entries: List[_LayerKVResidencyEntry] = []
                run_locs: Optional[List[int]] = None
                for layer_id in layer_ids:
                    layer_entries: List[_LayerKVResidencyEntry] = []
                    layer_locs: List[int] = []
                    valid = True
                    for token_pos in range(pos, pos + run_len):
                        key = (int(layer_id), req_idx, int(token_pos))
                        entry = self._per_layer_residency.get(key)
                        self.stats.kvc_evict_selector_scanned_entries += 1
                        if (
                            entry is None
                            or entry.state != "resident"
                            or int(entry.token_count) != 1
                        ):
                            valid = False
                            break
                        locs = entry.device_loc_list()
                        if len(locs) != 1 or int(locs[0]) <= 0:
                            valid = False
                            break
                        layer_entries.append(entry)
                        layer_locs.append(int(locs[0]))
                    if not valid:
                        run_entries = []
                        break
                    if run_locs is None:
                        run_locs = layer_locs
                    elif run_locs != layer_locs:
                        run_entries = []
                        break
                    run_entries.extend(layer_entries)
                if run_entries:
                    selected.extend(run_entries)
                    selected_common += run_len
                    self._kvc_evict_cursors_by_layer[cursor_key] = pos + run_len
                    self.stats.kvc_layerwise_evict_cursor_hit_count += len(layer_ids)
                pos += run_len
            if selected_common >= common_tokens:
                break

        if selected:
            selected_tokens = sum(int(entry.token_count) for entry in selected)
            self.stats.kvc_evict_selector_fast_hit += 1
            self.stats.kvc_evict_candidate_selected_tokens += selected_tokens
            self.stats.kvc_evict_selector_selected_entries += len(selected)
            self.stats.kvc_layerwise_evict_selected_layer_count += len(layer_ids)
            self.stats.kvc_layerwise_evict_selected_token_count += selected_tokens
        return selected

    def _select_per_layer_resident_entries_for_eviction(
        self, forward_batch: Any, max_tokens: int
    ) -> List[_LayerKVResidencyEntry]:
        pairs = self._batch_req_indices_and_lens(forward_batch)
        block_size = self._per_layer_kvc_block_page_size()
        common_tokens = int(self._force_common_kvc_evict_tokens or 0)
        if common_tokens > 0:
            return self._select_common_per_layer_resident_entries_for_eviction(
                forward_batch, common_tokens
            )
        plan = dict(self._planned_kvc_tokens_by_layer)
        if not plan and self._planned_kvc_token_target > 0:
            plan = self._build_layer_aware_kvc_token_plan(
                self._planned_kvc_token_target, forward_batch
            )
        if not plan and max_tokens > 0:
            if self._uses_layer_aware_kvc_plan():
                plan = self._build_layer_aware_kvc_token_plan(max_tokens, forward_batch)
            else:
                layer_ids = self._kvc_layer_ids()
                if layer_ids:
                    block_tokens = max(
                        block_size,
                        self._align_tokens_up(int(self.config.kvc_block_tokens)),
                    )
                    per_layer = self._align_tokens_down(
                        int(max_tokens) // max(1, len(layer_ids))
                    )
                    per_layer -= per_layer % block_tokens
                    if per_layer > 0:
                        plan = {int(layer_id): per_layer for layer_id in layer_ids}
        self.stats.kvc_layerwise_evict_plan_layer_count += len(plan)
        selected: List[_LayerKVResidencyEntry] = []
        selected_layers: Set[int] = set()
        for layer_id, target_tokens in sorted(plan.items()):
            layer_id = int(layer_id)
            current = int(
                self._per_layer_offloaded_token_count_by_layer.get(layer_id, 0)
            )
            need_tokens = self._align_tokens_down(max(0, int(target_tokens) - current))
            scratch_capacity = self._per_layer_virtual_scratch_capacity_tokens()
            if scratch_capacity > 0:
                need_tokens = min(need_tokens, max(0, scratch_capacity - current))
                need_tokens = self._align_tokens_down(need_tokens)
            if need_tokens <= 0:
                continue
            if need_tokens < block_size:
                continue
            layer_segments: List[Tuple[int, int, int]] = []
            remaining = need_tokens
            pair_meta: List[Tuple[int, int, int]] = []
            for req_idx, seq_len in pairs:
                req_idx = int(req_idx)
                evictable_len = self._align_tokens_down(max(0, int(seq_len) - 1))
                if evictable_len < block_size:
                    continue
                cursor_key = (layer_id, req_idx)
                cursor = self._align_tokens_down(
                    self._kvc_evict_cursors_by_layer.get(cursor_key, 0)
                )
                if cursor >= evictable_len:
                    cursor = 0
                    self._kvc_evict_cursors_by_layer[cursor_key] = 0
                    self.stats.kvc_layerwise_evict_cursor_reset_count += 1
                pair_meta.append((req_idx, evictable_len, cursor))
            if not pair_meta:
                continue
            self.stats.kvc_evict_selector_fast_hit += 1
            round_offsets = {req_idx: 0 for req_idx, _seq_len, _cursor in pair_meta}
            while remaining >= block_size:
                progressed = False
                for req_idx, evictable_len, cursor in pair_meta:
                    offset = round_offsets[req_idx]
                    pos = cursor + offset
                    round_offsets[req_idx] = offset + block_size
                    if pos + block_size > evictable_len:
                        continue
                    key = (layer_id, req_idx, pos)
                    existing = self._per_layer_residency.get(key)
                    self.stats.kvc_evict_selector_scanned_entries += 1
                    if existing is not None and existing.state in (
                        "offloaded",
                        "reloading",
                        "evicting",
                    ):
                        continue
                    layer_segments.append((req_idx, pos, block_size))
                    remaining -= block_size
                    progressed = True
                    self.stats.kvc_layerwise_evict_cursor_hit_count += 1
                    if remaining < block_size:
                        break
                if not progressed:
                    break
            if not layer_segments:
                continue
            for req_idx, pos, count in layer_segments:
                segment_entries: List[_LayerKVResidencyEntry] = []
                valid = True
                for token_pos in range(int(pos), int(pos) + int(count)):
                    key = (layer_id, int(req_idx), int(token_pos))
                    entry = self._per_layer_residency.get(key)
                    self.stats.kvc_evict_selector_scanned_entries += 1
                    if (
                        entry is None
                        or entry.state != "resident"
                        or int(entry.token_count) != 1
                    ):
                        valid = False
                        break
                    locs = entry.device_loc_list()
                    if len(locs) != 1 or int(locs[0]) <= 0:
                        valid = False
                        break
                    segment_entries.append(entry)
                if not valid or len(segment_entries) != int(count):
                    continue
                self.stats.kvc_evict_candidate_scan_tokens += int(count)
                for entry in segment_entries:
                    entry.last_access_step = self._decode_step
                    self._sync_kvc_group_if_needed(entry)
                selected.extend(segment_entries)
                selected_layers.add(layer_id)
                cursor_key = (layer_id, int(req_idx))
                next_pos = int(pos) + int(count)
                old_cursor = self._kvc_evict_cursors_by_layer.get(cursor_key, 0)
                if next_pos > old_cursor:
                    self._kvc_evict_cursors_by_layer[cursor_key] = next_pos
        selected.sort(key=lambda x: (x.layer_id, x.pos, x.req_idx))
        selected_tokens = sum(entry.token_count for entry in selected)
        self.stats.kvc_evict_candidate_selected_tokens += selected_tokens
        self.stats.kvc_evict_selector_selected_entries += len(selected)
        self.stats.kvc_layerwise_evict_selected_layer_count += len(selected_layers)
        self.stats.kvc_layerwise_evict_selected_token_count += selected_tokens
        return selected

    def _rewrite_req_to_token(
        self, selected: List[Tuple[int, int, int]], new_locs: torch.Tensor
    ) -> None:
        if not selected:
            return
        req_idx = torch.tensor(
            [x[0] for x in selected],
            dtype=torch.int64,
            device=self._req_to_token_pool.req_to_token.device,
        )
        pos = torch.tensor(
            [x[1] for x in selected],
            dtype=torch.int64,
            device=self._req_to_token_pool.req_to_token.device,
        )
        self._req_to_token_pool.req_to_token[req_idx, pos] = new_locs.to(torch.int32)
        self.stats.kvc_req_to_token_rewrite_count += int(new_locs.numel())

    def _rewrite_req_to_token_for_entries(
        self, selected: List[_LayerKVResidencyEntry], new_locs: torch.Tensor
    ) -> None:
        if not selected:
            return
        req_indices: List[int] = []
        positions: List[int] = []
        offset = 0
        for entry in selected:
            req_indices.extend([entry.req_idx] * entry.token_count)
            positions.extend(entry.logical_positions())
            offset += entry.token_count
        if offset != int(new_locs.numel()):
            self.stats.kvc_page_alignment_violation_count += 1
            raise RuntimeError(
                f"LayerKV KVC reload loc count mismatch: expected={offset} got={int(new_locs.numel())}"
            )
        req_idx = torch.tensor(
            req_indices,
            dtype=torch.int64,
            device=self._req_to_token_pool.req_to_token.device,
        )
        pos = torch.tensor(
            positions,
            dtype=torch.int64,
            device=self._req_to_token_pool.req_to_token.device,
        )
        self._req_to_token_pool.req_to_token[req_idx, pos] = new_locs.to(torch.int32)
        self.stats.kvc_req_to_token_rewrite_count += int(new_locs.numel())

    def _rewrite_per_layer_req_to_token_for_entries(
        self, selected: List[_LayerKVResidencyEntry], new_locs: torch.Tensor
    ) -> None:
        if not selected:
            return
        by_layer: Dict[int, Tuple[List[int], List[int], List[int]]] = {}
        offset = 0
        for entry in selected:
            locs = [int(x) for x in entry.device_loc_list()]
            if len(locs) != int(entry.token_count):
                locs = [
                    int(x)
                    for x in new_locs[offset : offset + entry.token_count]
                    .detach()
                    .cpu()
                    .tolist()
                ]
            layer_id = int(entry.layer_id)
            if layer_id < 0:
                # Backward-compatible fallback for old selected entries.
                for lid in self._kvc_layer_ids():
                    reqs, poss, dsts = by_layer.setdefault(int(lid), ([], [], []))
                    reqs.extend([entry.req_idx] * entry.token_count)
                    poss.extend(entry.logical_positions())
                    dsts.extend(locs)
            else:
                reqs, poss, dsts = by_layer.setdefault(layer_id, ([], [], []))
                reqs.extend([entry.req_idx] * entry.token_count)
                poss.extend(entry.logical_positions())
                dsts.extend(locs)
            offset += entry.token_count
        if offset != int(new_locs.numel()):
            self.stats.kvc_page_alignment_violation_count += 1
            raise RuntimeError(
                f"LayerKV per-layer KVC reload loc count mismatch: expected={offset} got={int(new_locs.numel())}"
            )
        wrote = False
        for layer_id, (req_indices, positions, locs) in by_layer.items():
            loc_tensor = torch.tensor(locs, dtype=torch.int64, device=new_locs.device)
            wrote = (
                self._set_per_layer_token_slots(
                    int(layer_id), req_indices, positions, loc_tensor
                )
                or wrote
            )
        if wrote:
            self.stats.kvc_req_to_token_rewrite_count += int(new_locs.numel())

    def on_forward_end(self, *, mode: str, forward_batch: Any) -> None:
        if mode == "decode":
            t0 = time.perf_counter()
            self._virtual_materialize_plan = None
            self._virtual_materialize_plans_by_layer.clear()
            self._virtual_kvc_demands.clear()
            self._kvc_demand_signature_eval_step = None
            if (
                self._expert_hotness_pending_snapshots
                or self._expert_candidate_pending_snapshots
                or self._pending_expert_d2h_events
                or self._pending_expert_copy_events
            ):
                with self._profile("profile_finalize_expert_ms"):
                    self._finalize_expert_hotness_snapshots(block=False)
                    self._finalize_expert_candidate_snapshots(block=False)
                    self._finalize_expert_d2h_events(block=False)
                    self._finalize_expert_materialize_events(block=False)
            fast_path = self._no_pressure_fast_path_active(forward_batch)
            if fast_path:
                self.stats.layerkv_no_pressure_fastpath_count += 1
            else:
                self._maybe_prepare_expert_plan_after_decode(forward_batch)
                self._submit_expert_install_d2h_budgeted()
            if (
                self._pending_kvc_evict_events
                or self._pending_kvc_reload_events
                or self._pending_virtual_kvc_materialize
            ):
                with self._profile("profile_finalize_kvc_ms"):
                    self._finalize_kvc_evictions(block=False)
                    self._finalize_reloaded_entries(block=True)
                    self._finalize_virtual_materialize_events(block=True)
            if not fast_path:
                fast_path = self._no_pressure_fast_path_active(forward_batch)
            if fast_path:
                self.stats.layerkv_no_pressure_kvc_skip_count += 1
                self._refresh_physical_reclaim_peaks()
            elif (
                self.config.dynamic_pressure_from_kvc
                and self._dynamic_runtime_pressure_mb(forward_batch) <= 1e-3
            ):
                self.stats.dynamic_pressure_skip_steps += 1
                self.stats.kvc_topup_skip_count += 1
                self.stats.kvc_eviction_skipped_count += 1
                self._set_no_pressure_reclaim_stats(
                    max(0.0, float(self.config.reclaim_limit_mb))
                )
                self._refresh_physical_reclaim_peaks(record_step_sample=True)
                self._refresh_resident_group_stats()
            elif (
                self.config.dynamic_pressure_from_kvc
                and not self._kvc_reclaim_is_scheduler_visible()
            ):
                self.stats.dynamic_pressure_active_steps += 1
                self.stats.kvc_scheduler_invisible_skip_count += 1
                self.stats.kvc_topup_skip_count += 1
                self.stats.kvc_eviction_skipped_count += 1
                self.stats.target_limited_reason = (
                    "kvc_reclaim_not_scheduler_visible"
                )
                self._scheduler_pressure_tokens = 0
                self.stats.scheduler_budget_pressure_tokens = 0
                self._refresh_physical_reclaim_peaks(record_step_sample=True)
                self._refresh_resident_group_stats()
            elif (
                self.config.dynamic_pressure_from_kvc
                and self._scheduler_pressure_kvc_blocked
            ):
                self.stats.dynamic_pressure_active_steps += 1
                self.stats.kvc_layerwise_scheduler_deadline_reject_count += 1
                self.stats.kvc_topup_skip_count += 1
                self.stats.kvc_eviction_skipped_count += 1
                self.stats.target_limited_reason = (
                    "scheduler_pressure_kvc_blocked_by_terminal_protocol"
                )
                self._refresh_physical_reclaim_peaks(record_step_sample=True)
                self._refresh_resident_group_stats()
            else:
                with self._profile("profile_kvc_evict_to_target_ms"):
                    self._evict_kvc_to_target(forward_batch)
                self._refresh_physical_reclaim_peaks(record_step_sample=True)
                self._refresh_resident_group_stats()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.stats.layerkv_python_overhead_ms += elapsed_ms
            self._add_profile("profile_forward_end_ms", elapsed_ms)
            self._flush_expert_hotness_sample_skip_count()
        elif mode == "extend":
            t0 = time.perf_counter()
            self._maybe_prepare_expert_plan_during_extend(forward_batch)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms > 0.0:
                self.stats.layerkv_python_overhead_ms += elapsed_ms
                self._add_profile("profile_forward_end_ms", elapsed_ms)
        if self.config.debug_stats and self._should_log_stats(mode):
            # Keep periodic decode logging off the critical path. Full
            # validation remains available through explicit summary(validate=True)
            # calls and validation scripts, while logs report the latest guard
            # counters maintained by runtime mutations.
            summary = self.summary(include_planner_inputs=False, validate=False)
            logger.info(
                "LayerKV stats after %s: %s",
                mode,
                summary,
            )

    def _should_log_stats(self, mode: str) -> bool:
        if mode != "decode":
            return False
        # Keep the timed path compact. Fig4-style runs use output_len=16, so
        # this still emits a final row for the harness without per-step dict
        # construction and logging overhead.
        return self._decode_step <= 1 or self._decode_step % 16 == 0

    def summary(
        self, *, include_planner_inputs: bool = True, validate: bool = True
    ) -> Dict[str, Any]:
        t0 = time.perf_counter() if self.config.debug_stats else 0.0
        self._flush_expert_hotness_sample_skip_count()
        if (
            self.stats.configured_reclaim_limit_mb <= 0.0
            and self.config.reclaim_limit_mb > 0.0
        ):
            self._refresh_reclaim_target_stats(self._last_forward_batch)
        self._refresh_physical_reclaim_peaks()
        if validate:
            self._refresh_resident_group_stats()
            self.validate_kvc_state()
            self.validate_expert_state()
            self._validate_resident_groups()
        else:
            self._refresh_resident_group_stats()
        planner_inputs: Dict[str, Any] = {}
        if include_planner_inputs:
            planner_inputs = self._planner_input_summary()
        if self.config.profile_detail:
            self._add_profile(
                "profile_summary_build_ms", (time.perf_counter() - t0) * 1000.0
            )
            self._refresh_profile_derived()
        self._sync_coresid_expert_plan_stats(context="summary")
        out = self.stats.as_dict()
        out.update(planner_inputs)
        out.update(
            {
                "layerkv_enabled": self.config.enabled,
                "layerkv_mode": self.config.mode,
                "layerkv_policy": self.config.policy,
                "layerkv_reclaim_limit_mb": self.config.reclaim_limit_mb,
                "layerkv_kvc_backend": self.config.kvc_backend,
                "layerkv_kvc_scheduler": self.config.kvc_scheduler,
                "layerkv_runtime_profile": self.config.runtime_profile,
                "layerkv_worker_role": self.config.worker_role,
                "layerkv_physical_kvc_supported": self.physical_kvc_supported,
                "layerkv_physical_expert_supported": self.physical_expert_supported,
                "layerkv_expert_layer_count": len(self._expert_layers),
                "layerkv_expert_discovered_layer_count": len(self._expert_modules),
                "layerkv_expert_installed_layer_count": len(self._expert_layers),
                "layerkv_unsupported_reason": self.unsupported_reason,
            }
        )
        return out

    def _planner_input_summary(self) -> Dict[str, Any]:
        call_by_layer: Dict[str, Dict[str, int]] = {}
        unique_by_layer: Dict[str, Dict[str, int]] = {}
        topk_by_layer: Dict[str, int] = {}
        hotness_topk_by_layer: Dict[str, Dict[str, List[List[int]]]] = {}
        num_experts_by_layer: Dict[str, int] = {}

        for layer_id, state in sorted(self._expert_layers.items()):
            layer_key = str(layer_id)
            prefill_total = int(sum(state.hotness_prefill.values()))
            decode_total = int(sum(state.hotness_decode.values()))
            call_by_layer[layer_key] = {
                "prefill": prefill_total,
                "decode": decode_total,
                "total": prefill_total + decode_total,
            }
            unique_by_layer[layer_key] = {
                "prefill": len(state.hotness_prefill),
                "decode": len(state.hotness_decode),
                "total": len(set(state.hotness_prefill) | set(state.hotness_decode)),
            }
            top_k = int(getattr(state.module, "top_k", 0) or 0)
            if top_k <= 0:
                top_k = int(getattr(state.module.moe_runner_config, "top_k", 0) or 0)
            topk_by_layer[layer_key] = top_k
            num_experts_by_layer[layer_key] = int(state.full_num_experts)
            hotness_topk_by_layer[layer_key] = {
                "prefill": self._hotness_top_items(state.hotness_prefill),
                "decode": self._hotness_top_items(state.hotness_decode),
            }

        return {
            "num_expert_layers": len(self._expert_layers),
            "num_experts_by_layer": json.dumps(num_experts_by_layer, sort_keys=True),
            "topk_by_layer": json.dumps(topk_by_layer, sort_keys=True),
            "expert_call_count_by_layer": json.dumps(call_by_layer, sort_keys=True),
            "expert_unique_count_by_layer": json.dumps(unique_by_layer, sort_keys=True),
            "expert_hotness_topk_by_layer": json.dumps(
                hotness_topk_by_layer, sort_keys=True
            ),
        }

    @staticmethod
    def _hotness_top_items(hotness: Dict[int, int], limit: int = 8) -> List[List[int]]:
        return [
            [int(expert_id), int(count)]
            for expert_id, count in sorted(
                hotness.items(), key=lambda item: (-item[1], item[0])
            )[:limit]
        ]
