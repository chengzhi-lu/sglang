"""LayerKV runtime configuration and stats."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict


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
    expert_topk_gpu_remap_count: int = 0
    expert_topk_gpu_remap_fallback_count: int = 0
    expert_topk_gpu_remap_missing_count: int = 0
    expert_topk_gpu_remap_error_count: int = 0
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
    expert_install_metadata_ms: float = 0.0
    expert_install_weight_alloc_ms: float = 0.0
    expert_install_weight_copy_ms: float = 0.0
    expert_install_param_swap_ms: float = 0.0
    expert_install_hook_ms: float = 0.0
    expert_install_prebuild_submit_ms: float = 0.0
    expert_install_prealloc_submit_ms: float = 0.0
    expert_install_prealloc_ready_count: int = 0
    expert_install_prealloc_reuse_count: int = 0
    expert_install_prealloc_pending_count: int = 0
    expert_install_prealloc_fallback_count: int = 0
    expert_install_prealloc_wait_count: int = 0
    expert_install_compact_pool_alloc_count: int = 0
    expert_install_compact_pool_reuse_count: int = 0
    expert_install_compact_pool_release_count: int = 0
    expert_install_compact_pool_drop_count: int = 0
    expert_install_compact_pool_bytes: int = 0
    expert_install_prebuild_ready_count: int = 0
    expert_install_prebuild_pending_count: int = 0
    expert_install_prebuild_wait_count: int = 0
    expert_install_shrink_layer_count: int = 0
    expert_install_shrink_expert_count: int = 0
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
    expert_d2h_demand_ready_hit_count: int = 0
    expert_d2h_demand_boost_count: int = 0
    expert_d2h_demand_urgent_enqueue_count: int = 0
    expert_d2h_demand_async_count: int = 0
    expert_d2h_demand_finalize_count: int = 0
    expert_d2h_demand_pending_wait_count: int = 0
    expert_d2h_demand_unavailable_count: int = 0
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
    expert_d2h_stream_launch_count: int = 0
    expert_h2d_stream_launch_count: int = 0
    expert_d2h_stream_busy_ms: float = 0.0
    expert_h2d_stream_busy_ms: float = 0.0
    expert_ready_before_use_count: int = 0
    expert_ready_use_check_count: int = 0
    expert_ready_before_use_ratio: float = 1.0
    expert_ready_miss_stall_ms: float = 0.0
    expert_ready_miss_stall_count: int = 0
    expert_h2d_on_demand_async_count: int = 0
    expert_h2d_prefetch_async_count: int = 0
    expert_h2d_sync_count: int = 0
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
    kvc_ready_miss_stall_ms: float = 0.0
    kvc_ready_miss_stall_count: int = 0
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
    profile_decode_scheduler_wall_ms: float = 0.0
    profile_decode_model_forward_ms: float = 0.0
    profile_decode_process_result_ms: float = 0.0
    profile_decode_critical_path_ms: float = 0.0
    profile_decode_batch_count: int = 0
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
    profile_virtual_reload_prepare_ms: float = 0.0
    profile_virtual_reload_contiguous_check_ms: float = 0.0
    profile_virtual_reload_issue_wall_ms: float = 0.0
    profile_virtual_reload_sync_wall_ms: float = 0.0
    virtual_kvc_reload_slice_count: int = 0
    virtual_kvc_reload_span_count: int = 0
    virtual_kvc_reload_span_segment_count: int = 0
    virtual_kvc_reload_direct_count: int = 0
    virtual_kvc_reload_index_count: int = 0
    virtual_kvc_reload_async_count: int = 0
    virtual_kvc_reload_sync_count: int = 0
    virtual_kvc_reload_slice_token_count: int = 0
    virtual_kvc_reload_span_token_count: int = 0
    virtual_kvc_reload_direct_token_count: int = 0
    virtual_kvc_reload_index_token_count: int = 0
    profile_kvc_evict_staging_alloc_ms: float = 0.0
    profile_kvc_host_alloc_ms: float = 0.0
    profile_kvc_backup_wall_ms: float = 0.0
    profile_kvc_evict_commit_ms: float = 0.0
    profile_kvc_free_locs_ms: float = 0.0
    profile_kvc_cost_observe_ms: float = 0.0
    profile_kvc_refresh_stats_ms: float = 0.0
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
