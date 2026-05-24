"""LayerKV runtime adapter for SGLang v0.5.12.

The integration keeps SGLang's native KV pool and MoE execution path in place,
while adding the policy-residency lifecycle hooks needed to make the feature
runnable and measurable.  The first physical path supports MHA KV pools with
page_size=1 by persistently moving selected KV slots to compact CPU backing
storage and reloading them before attention consumes them.
"""

from __future__ import annotations

import dataclasses
import contextlib
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
    target_reclaim_mb: float = 0.0
    dynamic_pressure_from_kvc: bool = False
    kvc_block_tokens: int = 16
    kvc_backend: str = "token-slot"
    kvc_scheduler: str = "async-deadline"
    runtime_profile: str = "optimized"
    debug_stats: bool = False
    profile_detail: bool = False
    disallow_destructive_fallback: bool = True
    expert_backing_cache_mb: float = 0.0
    expert_cpu_backing_mode: str = "none"
    expert_install_layers_per_step: int = 1
    expert_install_budget_mb: float = 128.0
    expert_install_target_steps: int = 0

    @classmethod
    def from_server_args(cls, server_args: Any) -> "LayerKVConfig":
        return cls(
            enabled=bool(getattr(server_args, "enable_layerkv", False)),
            mode=str(getattr(server_args, "layerkv_mode", "off")),
            policy=str(getattr(server_args, "layerkv_policy", "none")),
            target_reclaim_mb=float(
                getattr(server_args, "layerkv_target_reclaim_mb", 0.0) or 0.0
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
            debug_stats=bool(getattr(server_args, "layerkv_debug_stats", False)),
            profile_detail=bool(getattr(server_args, "layerkv_profile_detail", False)),
            disallow_destructive_fallback=bool(
                getattr(server_args, "layerkv_disallow_destructive_fallback", True)
            ),
            expert_backing_cache_mb=float(
                getattr(server_args, "layerkv_expert_backing_cache_mb", 0.0)
                or 0.0
            ),
            expert_cpu_backing_mode=str(
                getattr(server_args, "layerkv_expert_cpu_backing_mode", "none")
                or "none"
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
                    getattr(server_args, "layerkv_expert_install_target_steps", 0)
                    or 0
                ),
            ),
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
    configured_target_reclaim_mb: float = 0.0
    effective_reclaim_target_mb: float = 0.0
    needed_pressure_mb: float = 0.0
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
    selected_kvc_tokens_by_layer: str = ""
    selected_expert_evictions_by_layer: str = ""
    layerkv_runtime_profile: str = ""
    layerkv_kvc_backend: str = ""
    layerkv_kvc_backend_semantics: str = ""
    layerkv_kvc_backend_limited: bool = False
    layerkv_kvc_backend_ready: bool = False
    layerkv_kvc_backend_reason: str = ""
    kvc_per_layer_metadata_rewrite_count: int = 0
    kvc_per_layer_metadata_rewrite_skip_count: int = 0
    kvc_per_layer_metadata_rewrite_unsupported_count: int = 0
    kvc_per_layer_override_layer_count: int = 0
    kvc_per_layer_identity_override_count: int = 0
    kvc_per_layer_slot_override_count: int = 0
    kvc_per_layer_slot_override_token_count: int = 0
    kvc_per_layer_arena_entry_count: int = 0
    kvc_per_layer_arena_resident_token_count: int = 0
    kvc_per_layer_arena_offloaded_token_count: int = 0
    kvc_per_layer_arena_capacity_mb: float = 0.0
    kvc_per_layer_arena_used_mb: float = 0.0
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
    expert_core_hook_count: int = 0
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
    profile_kvc_reload_required_ms: float = 0.0
    profile_kvc_evict_to_target_ms: float = 0.0
    profile_kvc_select_required_ms: float = 0.0
    profile_kvc_select_evict_ms: float = 0.0
    profile_req_to_token_rewrite_ms: float = 0.0
    profile_expert_unique_ms: float = 0.0
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
class _LayerKVPendingExpertCopy:
    start_event: Any
    ready_event: Any
    layer_id: int
    logical_ids: Set[int]
    waited_on_main_stream: bool = False


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
    logical_ids: Tuple[int, ...] = ()
    group_keys: Tuple[_LayerKVResidencyKey, ...] = ()
    entries: Tuple[_LayerKVResidencyEntry, ...] = ()
    token_count: int = 0
    benefit_score: float = 0.0
    state: str = "pending"


@dataclasses.dataclass
class _LayerKVExpertInstallItem:
    layer_id: int
    module: Any
    slot_capacity: int
    initial_resident: Optional[List[int]]
    prepared_cpu_params: Optional[Dict[int, Dict[str, torch.Tensor]]] = None


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

    def __init__(self, kv_pool: Any, capacity_tokens: int, *, per_layer_mode: bool = False):
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
        return self.capacity_tokens * self.bytes_per_token_all_layers / float(1024 * 1024)

    def _ensure_layer_capacity(self, layer_offset: int, need_free: int) -> None:
        if not self.per_layer_mode or need_free <= len(self.layer_free_slots[layer_offset]):
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

    def alloc_per_layer(self, entries: List["_LayerKVResidencyEntry"]) -> Optional[List[int]]:
        if not self.per_layer_mode:
            return self.alloc(sum(entry.token_count for entry in entries))
        need_by_layer: Dict[int, int] = {}
        for entry in entries:
            layer_offset = int(entry.layer_id) - self.start_layer
            if layer_offset < 0 or layer_offset >= self.layer_num:
                return None
            need_by_layer[layer_offset] = need_by_layer.get(layer_offset, 0) + int(entry.token_count)
        for layer_offset, need in need_by_layer.items():
            self._ensure_layer_capacity(layer_offset, need)
        out: List[int] = []
        for entry in entries:
            layer_offset = int(entry.layer_id) - self.start_layer
            need = int(entry.token_count)
            slots = self.layer_free_slots[layer_offset][:need]
            del self.layer_free_slots[layer_offset][:need]
            self.layer_used_slots[layer_offset].update(slots)
            out.extend(slots)
        return out

    def alloc(self, need: int) -> Optional[List[int]]:
        if self.per_layer_mode:
            raise RuntimeError("use alloc_per_layer for per-layer KVC host store")
        if need > len(self.free_slots):
            return None
        slots = self.free_slots[:need]
        del self.free_slots[:need]
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
            k_src = self.kv_pool._get_key_buffer(layer_id)[device_locs].detach().to(
                "cpu", non_blocking=True
            )
            v_src = self.kv_pool._get_value_buffer(layer_id)[device_locs].detach().to(
                "cpu", non_blocking=True
            )
            self.k_buffers[layer_offset].index_copy_(0, host_index, k_src)
            self.v_buffers[layer_offset].index_copy_(0, host_index, v_src)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

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
            k_src = self.kv_pool._get_key_buffer(layer_id)[device_locs].detach().to(
                "cpu", non_blocking=True
            )
            v_src = self.kv_pool._get_value_buffer(layer_id)[device_locs].detach().to(
                "cpu", non_blocking=True
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
                k_src = self.k_buffers[layer_offset].index_select(0, host_index).to(
                    self.device, non_blocking=True
                )
                v_src = self.v_buffers[layer_offset].index_select(0, host_index).to(
                    self.device, non_blocking=True
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
                    raise RuntimeError(f"invalid per-layer KVC layer_id={entry.layer_id}")
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
                host_index = torch.tensor(host_slot_list, dtype=torch.int64, device="cpu")
                device_locs = torch.tensor(
                    device_loc_list, dtype=torch.int64, device=self.device
                )
                k_src = self.k_buffers[layer_offset].index_select(0, host_index).to(
                    self.device, non_blocking=True
                )
                v_src = self.v_buffers[layer_offset].index_select(0, host_index).to(
                    self.device, non_blocking=True
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

    def recover(self, groups: List[_LayerKVResidentTensorGroup], stream: Any = None) -> None:
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

    def recover(self, groups: List[_LayerKVResidentTensorGroup], stream: Any = None) -> None:
        self.runtime._reload_required_kvc(self.runtime._last_forward_batch)


class _LayerKVExpertResidencyBackend(_LayerKVResidencyBackend):
    kind = "expert"

    def recover(self, groups: List[_LayerKVResidentTensorGroup], stream: Any = None) -> None:
        by_layer: Dict[int, List[int]] = {}
        for group in groups:
            by_layer.setdefault(int(group.key.layer_id), []).append(int(group.key.logical_id))
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
            self.config.kvc_scheduler = "sync"
            self.config.expert_backing_cache_mb = 0.0
            self.config.expert_cpu_backing_mode = "none"
            self.config.expert_install_target_steps = 0
            self.config.expert_install_budget_mb = 0.0
        self.stats = LayerKVStats()
        self.stats.layerkv_runtime_profile = str(self.config.runtime_profile)
        self.stats.layerkv_kvc_backend = str(self.config.kvc_backend)
        if self.config.kvc_backend == "token-slot":
            self.stats.layerkv_kvc_backend_semantics = "global_token_slot_layer_average"
            self.stats.layerkv_kvc_backend_limited = True
            self.stats.layerkv_kvc_backend_ready = True
            self.stats.layerkv_kvc_backend_reason = ""
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
        self._resident_groups: Dict[
            _LayerKVResidencyKey, _LayerKVResidentTensorGroup
        ] = {}
        self._residency_backends: Dict[str, _LayerKVResidencyBackend] = {
            "kvc": _LayerKVKVCResidencyBackend(self),
            "expert": _LayerKVExpertResidencyBackend(self),
        }
        self._kvc_evict_cursors: Dict[int, int] = {}
        self._expert_layers: Dict[int, _LayerKVExpertLayerState] = {}
        self._expert_modules: List[Tuple[int, Any]] = []
        self._expert_hotness_prefill: Dict[int, Dict[int, int]] = {}
        self._expert_hotness_decode: Dict[int, Dict[int, int]] = {}
        self._expert_plan_applied: bool = False
        self._expert_install_queue: List[_LayerKVExpertInstallItem] = []
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
        self._expert_global_cpu_backing: Dict[
            Tuple[int, int], Dict[str, torch.Tensor]
        ] = {}
        self._expert_global_cpu_backing_bytes: int = 0
        self._expert_group_dirty_layers: Set[int] = set()
        self._prepared_expert_plan: Optional[Dict[str, Any]] = None
        self._expert_prepare_started: bool = False
        self._expert_prepare_done: bool = False
        self._planned_kvc_token_target: int = 0
        self._planned_kvc_tokens_by_layer: Dict[int, int] = {}
        self._cached_policy_fractions: Optional[
            Tuple[float, float, bool, str, bool, str, float, float]
        ] = None
        self._cached_policy_target_bucket_mb: Optional[int] = None
        self._pending_expert_copy_events: List[_LayerKVPendingExpertCopy] = []
        self._pending_kvc_reload_events: List[_LayerKVPendingReload] = []
        self._per_layer_req_to_token_overrides: Dict[int, torch.Tensor] = {}
        self._per_layer_req_to_token_owned: Set[int] = set()
        self._per_layer_kvc_prepared_layers: Set[Tuple[int, int]] = set()
        self._expert_materialize_batch_sizes: List[int] = []
        self._expert_materialize_layers_touched: Set[int] = set()
        self._current_forward_mode: str = ""
        self._last_forward_batch: Any = None
        self._decode_step: int = 0

    @classmethod
    def maybe_create(cls, server_args: Any) -> Optional["LayerKVRuntime"]:
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
        self.stats.expert_cpu_backing_preload_mb = (
            max(0, self._expert_global_cpu_backing_bytes) / float(1024 * 1024)
        )
        self.stats.layerkv_runtime_profile = str(self.config.runtime_profile)

    def _optimized_profile_enabled(self) -> bool:
        return str(self.config.runtime_profile) == "optimized"

    def _simple_profile_enabled(self) -> bool:
        return str(self.config.runtime_profile) == "simple"

    def _effective_expert_backing_cache_mb(self) -> float:
        if self._simple_profile_enabled():
            return 0.0
        return float(max(0.0, self.config.expert_backing_cache_mb))

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

    def _sync_kvc_group(self, entry: _LayerKVResidencyEntry) -> _LayerKVResidentTensorGroup:
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
            cpu_params = state.cpu_params.get(int(expert_id)) or self._expert_global_cpu_backing.get(
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
            cpu_params = state.cpu_params.get(int(logical_id)) or self._expert_global_cpu_backing.get(
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
        self._sync_dirty_expert_groups()
        groups = list(self._resident_groups.values())
        self.stats.unified_residency_enabled = True
        self.stats.resident_group_count = len(groups)
        self.stats.resident_group_kvc_count = sum(1 for g in groups if g.key.kind == "kvc")
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
            1 for g in groups if g.state in ("reloading", "materializing")
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
            if group.state in ("reloading", "materializing") and group.ready_event is None:
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
        self._discover_expert_support(runner)
        self.installed = True
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.layerkv_python_overhead_ms += elapsed_ms
        self._add_profile("profile_install_ms", elapsed_ms)
        logger.info(
            "LayerKV enabled mode=%s policy=%s target_reclaim_mb=%.1f "
            "kvc_supported=%s expert_supported=%s reason=%s",
            self.config.mode,
            self.config.policy,
            self.config.target_reclaim_mb,
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
            and self.config.target_reclaim_mb > 0
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
            self._bytes_per_token_all_layers = int(
                (one_k + one_v) * kv_pool.layer_num
            )
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
        target_bytes = max(1, int(self.config.target_reclaim_mb * 1024 * 1024))
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
                self._expert_global_cpu_backing[(int(layer_id), int(expert_id))] = params
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
            if topk_ids is not None:
                self._record_expert_hotness_for_layer(
                    layer_id=layer_id,
                    full_num_experts=full_num_experts,
                    topk_ids=topk_ids,
                )
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
        pairs = self._batch_req_indices_and_lens(forward_batch)
        if not pairs:
            return 0.0
        return sum(max(0, seq_len - 1) for _, seq_len in pairs) / float(len(pairs))

    def _policy_fractions(self, forward_batch: Any) -> Tuple[float, float, bool, str]:
        policy = self.config.policy
        if policy == "coresid":
            policy = "layer-aware-joint-dp"
        if policy == "none":
            return 0.0, 0.0, True, "policy=none disables LayerKV physical reclaim"
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
            return 0.0, 1.0, self.physical_expert_supported, self._expert_support_reason()
        if policy == "ratio-75-25":
            return 0.75, 0.25, self.physical_expert_supported, self._expert_support_reason()
        if policy == "ratio-50-50":
            return 0.50, 0.50, self.physical_expert_supported, self._expert_support_reason()
        if policy == "ratio-25-75":
            return 0.25, 0.75, self.physical_expert_supported, self._expert_support_reason()
        if policy in ("layer-aware-joint", "layer-aware-joint-dp"):
            target_bucket_mb: Optional[int] = None
            if self.config.dynamic_pressure_from_kvc:
                target_bucket_mb = int(
                    round(self._refresh_reclaim_target_stats(forward_batch) / 64.0)
                )
            if (
                self._cached_policy_fractions is not None
                and (
                    not self.config.dynamic_pressure_from_kvc
                    or self._cached_policy_target_bucket_mb == target_bucket_mb
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
                return kvc_fraction, expert_fraction, full_supported, reason
            return self._joint_policy_fractions(forward_batch)
        return 0.0, 0.0, False, f"unknown LayerKV policy: {policy}"

    def _joint_policy_fractions(self, forward_batch: Any) -> Tuple[float, float, bool, str]:
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
                self._cached_policy_target_bucket_mb = int(round(target_mb / 64.0))
            else:
                self._cached_policy_target_bucket_mb = None
        return kvc_fraction, expert_fraction, True, ""

    def _solve_joint_dp_reclaim_split(
        self, target_mb: float, forward_batch: Any
    ) -> Tuple[float, float, float, float]:
        if target_mb <= 0.0:
            self.stats.planner_dp_candidate_count = 1
            self.stats.planner_dp_selected_kvc_candidates = 0
            self.stats.planner_dp_selected_expert_candidates = 0
            self.stats.planner_dp_infeasible_kvc_candidates = 0
            self.stats.planner_dp_infeasible_expert_candidates = 0
            self.stats.planner_dp_selected_total_cost = 0.0
            self.stats.planner_dp_selected_kvc_cost = 0.0
            self.stats.planner_dp_selected_expert_cost = 0.0
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
        block_mb = (
            block_tokens * max(0, kvc_token_bytes) / float(1024 * 1024)
        )
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
        expert_layer_items, expert_candidates = self._build_expert_cost_inputs()
        expert_cost_cache: Dict[float, Tuple[float, float, float, float]] = {}

        def cached_expert_cost(expert_mb: float) -> Tuple[float, float, float, float]:
            key = round(max(0.0, float(expert_mb)), 6)
            cached = expert_cost_cache.get(key)
            if cached is not None:
                return cached
            with self._profile("profile_planner_dp_expert_cost_ms"):
                value = self._estimate_expert_reclaim_cost(
                    key,
                    forward_batch,
                    update_stats=False,
                    precomputed_layer_items=expert_layer_items,
                    precomputed_candidates=expert_candidates,
                )
            cached = (
                value,
                float(self.stats.planner_estimated_expert_churn_count),
                float(self.stats.planner_estimated_expert_churn_mb),
                float(self.stats.planner_estimated_expert_install_mb),
            )
            expert_cost_cache[key] = cached
            return cached

        expert_only_cost, expert_only_churn, expert_only_churn_mb, expert_only_install = cached_expert_cost(
            target_mb
        )
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
                    with self._profile("profile_planner_dp_kvc_cost_ms"):
                        kvc_cost = self._estimate_planner_kvc_reclaim_cost(
                            kvc_mb, forward_batch
                        )
                    if kvc_cost >= best[0]:
                        continue
                    expert_cost, churn_count, churn_mb, install_mb = cached_expert_cost(
                        expert_mb
                    )
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
                    best_expert_stats = (churn_count, churn_mb, install_mb)
                    best_kvc_tokens = int(kvc_token_points.get(kvc_mb, 0))
        assert best is not None
        _total_cost, kvc_fraction, expert_fraction, kvc_cost, expert_cost = best
        self.stats.planner_dp_candidate_count = len(kvc_points)
        self.stats.planner_dp_selected_kvc_candidates = 1 if kvc_fraction > 0.0 else 0
        self.stats.planner_dp_selected_expert_candidates = 1 if expert_fraction > 0.0 else 0
        self.stats.planner_dp_infeasible_kvc_candidates = infeasible_kvc
        self.stats.planner_dp_infeasible_expert_candidates = infeasible_expert
        self.stats.planner_dp_selected_total_cost = float(_total_cost)
        self.stats.planner_dp_selected_kvc_cost = float(kvc_cost)
        self.stats.planner_dp_selected_expert_cost = float(expert_cost)
        self.stats.planner_estimated_expert_churn_count = float(best_expert_stats[0])
        self.stats.planner_estimated_expert_churn_mb = float(best_expert_stats[1])
        self.stats.planner_estimated_expert_install_mb = float(best_expert_stats[2])
        self._planned_kvc_token_target = int(best_kvc_tokens if kvc_fraction > 0.0 else 0)
        if self._planned_kvc_token_target > 0:
            if self.config.kvc_backend == "per-layer-arena":
                self._planned_kvc_tokens_by_layer = self._arena_kvc_token_plan_for_reclaim(
                    target_mb * kvc_fraction, forward_batch
                )
            else:
                self._planned_kvc_tokens_by_layer = self._build_layer_aware_kvc_token_plan(
                    self._planned_kvc_token_target, forward_batch
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
            self.config.policy in ("layer-aware-joint", "layer-aware-joint-dp", "coresid")
            and self.config.kvc_backend == "per-layer-arena"
        )

    def _requires_per_layer_kvc_backend(self) -> bool:
        return self.config.policy in ("layer-aware-joint", "layer-aware-joint-dp", "coresid")

    def _per_layer_kvc_backend_ready(self) -> bool:
        return self.config.kvc_backend == "per-layer-arena"

    def _per_layer_kvc_backend_can_physically_reclaim(self) -> bool:
        return self.config.kvc_backend == "per-layer-arena"

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
        target_mb = (
            self.stats.effective_reclaim_target_mb
            if self.stats.effective_reclaim_target_mb > 0.0
            else max(0.0, self.config.target_reclaim_mb)
        )
        self.stats.planner_used_hotness = used_hotness
        self.stats.planner_fallback_reason = fallback_reason
        self.stats.planner_estimated_kvc_cost = float(kvc_cost)
        self.stats.planner_estimated_expert_cost = float(expert_cost)
        self.stats.planner_selected_kvc_reclaim_mb = target_mb * kvc_fraction
        self.stats.planner_selected_expert_reclaim_mb = target_mb * expert_fraction

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

    def _estimate_arena_kvc_reclaim_cost(self, reclaim_mb: float, forward_batch: Any) -> float:
        token_plan = self._arena_kvc_token_plan_for_reclaim(reclaim_mb, forward_batch)
        if not token_plan:
            self.stats.planner_estimated_kvc_controller_cost = 0.0
            return 0.0

        # CoResID compares KVC candidates by predicted exposed stall.  MB is
        # used only to derive raw copy bytes; it is not the planner objective.
        controller_cost = 0.0
        exposed_stall = 0.0
        for layer_id, tokens in token_plan.items():
            if tokens <= 0:
                continue
            raw_copy_ms = self._estimate_arena_kvc_layer_raw_reload_ms(
                int(layer_id), int(tokens)
            )
            overlap_ms = self._estimate_layer_kvc_overlap_window_ms(
                int(layer_id), forward_batch
            )
            exposed_stall += max(0.0, raw_copy_ms - overlap_ms)
            controller_cost += self._estimate_arena_kvc_layer_metadata_ms(
                int(layer_id), int(tokens)
            )
        self.stats.planner_estimated_kvc_controller_cost = controller_cost
        return exposed_stall + controller_cost

    def _arena_kvc_token_plan_for_reclaim(
        self, reclaim_mb: float, forward_batch: Any
    ) -> Dict[int, int]:
        if reclaim_mb <= 0.0:
            return {}
        bytes_per_token = max(1, self._bytes_per_kvc_token_per_layer())
        total_tokens = int(reclaim_mb * 1024.0 * 1024.0 / float(bytes_per_token))
        return self._build_layer_aware_kvc_token_plan(total_tokens, forward_batch)

    def _estimate_arena_kvc_layer_raw_reload_ms(
        self, layer_id: int, tokens: int
    ) -> float:
        if tokens <= 0:
            return 0.0
        bytes_per_token = max(1, self._bytes_per_kvc_token_per_layer())
        reload_bytes = float(tokens) * float(bytes_per_token)
        bandwidth_gbps = 24.0
        copy_ms = reload_bytes / (bandwidth_gbps * 1.0e9) * 1000.0
        # Runtime coalesces same-layer blocks, so the planner charges one launch
        # per active layer instead of one launch per token block.
        return 0.015 + copy_ms

    def _estimate_layer_kvc_overlap_window_ms(
        self, layer_id: int, forward_batch: Any
    ) -> float:
        layer_ids = self._kvc_layer_ids()
        if not layer_ids:
            return 0.0
        try:
            layer_rank = layer_ids.index(int(layer_id))
        except ValueError:
            layer_rank = 0
        avg_prefix = max(1.0, self._avg_prefix_len(forward_batch))
        batch_size = max(1, int(getattr(self.stats, "observed_batch_size", 0) or 1))
        context_factor = min(1.0, avg_prefix / 32768.0)
        batch_factor = min(1.0, math.log2(float(batch_size) + 1.0) / 8.0)
        per_prior_layer_ms = 0.02 + 0.10 * context_factor + 0.06 * batch_factor
        return float(layer_rank) * per_prior_layer_ms

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
        precomputed_candidates: Optional[List[Tuple[float, int]]] = None,
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
        slot_capacities = self._estimate_expert_slot_capacities_for_target(reclaim_mb)
        decode_steps = max(1, int(getattr(self.stats, "decode_steps", 0) or 16))
        total_churn_count = 0.0
        total_churn_mb = 0.0
        churn_cost = 0.0
        for layer_id, full_num_experts, expert_bytes in layer_items:
            hotness = self._expert_hotness_decode.get(layer_id) or self._expert_hotness_prefill.get(layer_id, {})
            total_calls = max(1, int(sum(hotness.values())))
            unique_count = len(hotness)
            capacity = int(slot_capacities.get(int(layer_id), full_num_experts))
            missing_unique = max(0, unique_count - capacity)
            if missing_unique > 0:
                churn_count = float(missing_unique * decode_steps)
                churn_mb = churn_count * expert_bytes / float(1024 * 1024)
                total_churn_count += churn_count
                total_churn_mb += churn_mb
                churn_cost += churn_mb * 0.05
        target_bytes = int(reclaim_mb * 1024 * 1024)
        reclaimed = 0
        cost = 0.0
        for p, expert_bytes in candidates:
            if reclaimed >= target_bytes:
                break
            reclaimed += expert_bytes
            cost += p * (expert_bytes / float(1024 * 1024))
        if reclaimed < target_bytes:
            return 1.0e30
        install_mb = reclaimed / float(1024 * 1024)
        if update_stats:
            self.stats.planner_estimated_expert_churn_count = total_churn_count
            self.stats.planner_estimated_expert_churn_mb = total_churn_mb
            self.stats.planner_estimated_expert_install_mb = install_mb
        else:
            self.stats.planner_estimated_expert_churn_count = total_churn_count
            self.stats.planner_estimated_expert_churn_mb = total_churn_mb
            self.stats.planner_estimated_expert_install_mb = install_mb
        install_cost = install_mb * 0.15
        return cost + churn_cost + install_cost

    def _build_expert_cost_inputs(self) -> Tuple[List[Tuple[int, int, int]], List[Tuple[float, int]]]:
        if self._expert_layers:
            layer_items = [
                (int(state.layer_id), int(state.full_num_experts), int(state.expert_bytes))
                for state in self._expert_layers.values()
            ]
        else:
            layer_items = [
                (
                    int(layer_id),
                    int(module.w13_weight.data.shape[0]),
                    int(self._expert_bytes(module)),
                )
                for layer_id, module in self._expert_modules
            ]
        candidates: List[Tuple[float, int]] = []
        for layer_id, full_num_experts, expert_bytes in layer_items:
            hotness = self._expert_hotness_decode.get(layer_id) or self._expert_hotness_prefill.get(layer_id, {})
            total_calls = max(1, int(sum(hotness.values())))
            for expert_id in range(full_num_experts):
                p = float(hotness.get(expert_id, 0)) / float(total_calls)
                candidates.append((p, expert_bytes))
        candidates.sort(key=lambda item: item[0])
        return layer_items, candidates

    def _estimate_expert_slot_capacities_for_target(self, target_mb: float) -> Dict[int, int]:
        target_bytes = max(0, int(target_mb * 1024 * 1024))
        layer_infos: List[Dict[str, int]] = []
        if self._expert_layers:
            for state in self._expert_layers.values():
                observed_decode_unique = len(self._expert_hotness_decode.get(state.layer_id, {}))
                min_capacity = max(1, min(state.full_num_experts, observed_decode_unique))
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
                observed_decode_unique = len(self._expert_hotness_decode.get(layer_id, {}))
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
                if info["capacity"] <= info["min_capacity"]:
                    continue
                info["capacity"] -= 1
                reclaimed += info["expert_bytes"]
            else:
                break
        return {info["layer_id"]: info["capacity"] for info in layer_infos}

    def _expert_support_reason(self) -> str:
        if self.physical_expert_supported:
            return ""
        return self.stats.expert_guard_reason or self.unsupported_reason or "expert offload unsupported"

    def _available_kvc_reclaim_mb(self, forward_batch: Any = None) -> float:
        if self._bytes_per_token_all_layers <= 0:
            return 0.0
        token_count = 0
        pairs = self._batch_req_indices_and_lens(forward_batch) if forward_batch is not None else []
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
        return list(range(start, start + layer_num))

    def _build_layer_aware_kvc_token_plan(
        self, total_tokens: int, forward_batch: Any
    ) -> Dict[int, int]:
        layer_ids = self._kvc_layer_ids()
        if not layer_ids or total_tokens <= 0:
            return {}
        block_tokens = max(
            self._page_size,
            self._align_tokens_up(int(self.config.kvc_block_tokens)),
        )
        total_tokens = self._align_tokens_down(int(total_tokens))
        if total_tokens <= 0:
            return {}
        pairs = self._batch_req_indices_and_lens(forward_batch)
        max_tokens_per_layer = sum(
            self._align_tokens_down(max(0, int(seq_len) - 1))
            for _req_idx, seq_len in pairs
        )
        if max_tokens_per_layer <= 0:
            max_tokens_per_layer = int(max(1.0, self._avg_prefix_len(forward_batch)))
        max_tokens_per_layer = self._align_tokens_down(max_tokens_per_layer)
        total_capacity = max_tokens_per_layer * len(layer_ids)
        total_tokens = min(total_tokens, total_capacity)
        if total_tokens <= 0:
            return {}
        # v1 keeps the DP output layer-aware by assigning more reclaim to later
        # layers, where KVC reload has more forward-path overlap.  This is a
        # deterministic plan and does not change baseline layer-average policy
        # semantics.
        weights = {
            int(layer_id): float(idx + 1) / float(len(layer_ids))
            for idx, layer_id in enumerate(layer_ids)
        }
        weight_sum = sum(weights.values()) or 1.0
        remaining = total_tokens
        plan: Dict[int, int] = {}
        for layer_id in layer_ids:
            raw = int(round(total_tokens * weights[int(layer_id)] / weight_sum))
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

    def _kvc_tokens_by_layer_json(self, token_count: Any) -> str:
        if isinstance(token_count, dict):
            return json.dumps(
                {str(layer_id): int(tokens) for layer_id, tokens in token_count.items()},
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
        configured = max(0.0, self.config.target_reclaim_mb)
        available_kvc = self._available_kvc_reclaim_mb(forward_batch)
        available_expert = self._available_expert_reclaim_mb()
        available_total = available_kvc + available_expert
        if self.config.dynamic_pressure_from_kvc:
            # End-to-end trace replay should reclaim only when the runtime is
            # actually short on KV allocator headroom.  Available KVC bytes are
            # just reclaimable capacity, not pressure; using them as pressure
            # incorrectly turns any large live KV set into a forced 4G reclaim.
            needed_pressure = min(configured, self._dynamic_needed_pressure_mb(forward_batch))
        else:
            # Fig4-style controlled experiments use configured pressure directly
            # so policies remain comparable at a fixed reclaim target.
            needed_pressure = configured
        effective = min(configured, needed_pressure, available_total)
        reason = ""
        if self.config.dynamic_pressure_from_kvc and needed_pressure <= 1e-3:
            reason = "no_runtime_pressure"
        elif effective + 1e-3 < needed_pressure:
            reason = "available_reclaim_below_needed_pressure"
        elif not self.config.dynamic_pressure_from_kvc and effective + 1e-3 < configured:
            reason = "available_reclaim_below_configured_target"
        self.stats.configured_target_reclaim_mb = configured
        self.stats.needed_pressure_mb = needed_pressure
        self.stats.available_kvc_reclaim_mb = available_kvc
        self.stats.available_expert_reclaim_mb = available_expert
        self.stats.available_total_reclaim_mb = available_total
        self.stats.effective_reclaim_target_mb = effective
        self.stats.target_limited_reason = reason
        return effective

    def _refresh_physical_reclaim_peaks(self, *, record_step_sample: bool = False) -> None:
        total = self.stats.physical_kvc_reclaim_mb + self.stats.physical_expert_reclaim_mb
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
            self.stats.physical_kvc_reclaim_step_sum_mb += self.stats.physical_kvc_reclaim_mb
            self.stats.physical_kvc_reclaim_step_count += 1
            self.stats.physical_kvc_reclaim_step_mean_mb = (
                self.stats.physical_kvc_reclaim_step_sum_mb
                / float(max(1, self.stats.physical_kvc_reclaim_step_count))
            )

    def _effective_kvc_reclaim_mb(self, forward_batch: Any) -> float:
        kvc_fraction, expert_fraction, full_supported, reason = self._policy_fractions(
            forward_batch
        )
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
                self.stats.planned_expert_reclaim_mb = physical_expert
                if kvc_fraction <= 0.0 and self._avg_prefix_len(forward_batch) <= 2048:
                    # In batch-heavy short-context runs, deficit backfill via KVC
                    # creates repeated evict/reload churn for little stable reclaim.
                    # Preserve the policy's expert-only intent instead of turning
                    # a workload-limited expert deficit into critical-path KV churn.
                    if not reason:
                        reason = "expert_reclaim_deficit_not_shifted_to_kvc_short_context"
                    self.stats.effective_reclaim_target_mb = physical_expert
                    self.stats.target_limited_reason = reason
                else:
                    effective = min(target_mb, effective + expert_deficit)
                    if not reason:
                        reason = "expert_reclaim_deficit_shifted_to_kvc"
        self.stats.requested_total_reclaim_mb = target_mb
        self.stats.effective_kvc_reclaim_mb = effective
        self.stats.policy_kvc_fraction = kvc_fraction
        self.stats.policy_expert_fraction = expert_fraction
        self.stats.full_policy_semantics_supported = full_supported
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
        effective = max(0.0, target_mb * expert_fraction)
        self.stats.requested_total_reclaim_mb = target_mb
        self.stats.policy_kvc_fraction = kvc_fraction
        self.stats.policy_expert_fraction = expert_fraction
        self.stats.full_policy_semantics_supported = full_supported
        self.stats.policy_semantics_reason = reason
        self.stats.planned_expert_reclaim_mb = effective
        return effective

    def _apply_expert_plan_once(self, forward_batch: Any) -> None:
        if self._expert_plan_applied or self.config.mode != "kvc-expert":
            return
        if self._expert_install_queue or self._expert_install_state in {
            "installing_slots",
            "queued",
        }:
            return
        target_mb = self._effective_expert_reclaim_mb(forward_batch)
        if target_mb <= 0:
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

        full_bytes = sum(self._expert_full_bytes(module) for _, module in self._expert_modules)
        if full_bytes <= 0:
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = "failed to inspect expert bytes"
            return

        with self._profile("profile_plan_expert_capacity_ms"):
            slot_capacities = self._plan_expert_slot_capacities(target_mb)
        try:
            self.stats.selected_expert_evictions_by_layer = json.dumps(
                {
                    str(layer_id): max(
                        0,
                        int(module.w13_weight.data.shape[0])
                        - int(slot_capacities.get(layer_id, module.w13_weight.data.shape[0])),
                    )
                    for layer_id, module in self._expert_modules
                },
                sort_keys=True,
            )
        except Exception:
            self.stats.selected_expert_evictions_by_layer = ""
        prepared_cpu_params: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        prepared_initial_resident: Dict[int, List[int]] = {}
        if self._expert_prepare_done and self._prepared_expert_plan is not None:
            prepared_kind = str(self._prepared_expert_plan.get("kind", "full"))
            prepared_cpu_params = self._prepared_expert_plan.get("cpu_params_by_layer", {})
            prepared_initial_resident = self._prepared_expert_plan.get(
                "initial_resident_by_layer", {}
            )
            prepared_capacities = self._prepared_expert_plan.get("slot_capacities", {})
            prepared_target_mb = float(self._prepared_expert_plan.get("target_mb", 0.0) or 0.0)
            if prepared_kind == "backing_only":
                prepared_initial_resident = {}
                self.stats.expert_prepared_plan_used = True
            elif (
                abs(prepared_target_mb - target_mb) > 1e-3
                or any(
                    int(prepared_capacities.get(layer_id, -1)) != int(slot_capacities[layer_id])
                    for layer_id, _module in self._expert_modules
                )
            ):
                prepared_cpu_params = {}
                prepared_initial_resident = {}
                self._prepared_expert_plan = None
                self._expert_prepare_done = False
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
            if (
                initial_resident is not None
                and (
                    len(initial_resident) != slot_capacity
                    or any(int(x) < 0 or int(x) >= full_num_experts for x in initial_resident)
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
        self._expert_install_target_mb = float(target_mb)
        self._expert_install_state = "queued"
        self._prepared_expert_plan = None
        self._refresh_expert_install_progress()

    def _refresh_expert_install_progress(self) -> None:
        pending = len(self._expert_install_queue)
        completed = len(self._expert_layers)
        state = self._expert_install_state
        if self._expert_plan_applied:
            state = "complete"
        elif pending > 0 and not state:
            state = "queued"
        self.stats.expert_install_state = state
        self.stats.expert_install_pending_layers = pending
        self.stats.expert_install_completed_layers = completed
        self.stats.expert_install_layers_per_step = int(self._expert_install_layers_per_step)
        self.stats.expert_install_budget_mb = float(self._expert_install_budget_mb)
        self.stats.expert_install_target_steps = int(self._expert_install_target_steps)
        self.stats.expert_install_reclaim_mb_progress = float(
            self.stats.physical_expert_reclaim_mb
        )
        if pending > 0:
            self.stats.comparable = False
            self.stats.comparability_reason = "INSTALL_IN_PROGRESS_NOT_COMPARABLE"
        elif self.stats.comparability_reason == "INSTALL_IN_PROGRESS_NOT_COMPARABLE":
            self.stats.comparable = True
            self.stats.comparability_reason = ""

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
            remaining_steps = max(1, target_steps - int(self.stats.forward_decode_count))
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
        self._expert_install_state = "installing_slots"
        install_profile = (
            "profile_apply_prepared_expert_plan_ms"
            if self._expert_install_queue[0].prepared_cpu_params
            else "profile_install_expert_slots_ms"
        )
        with self._profile(install_profile):
            while self._expert_install_queue and installed < max_layers:
                item = self._expert_install_queue[0]
                layer_reclaim_mb = max(
                    0.0,
                    (
                        int(item.module.w13_weight.data.shape[0])
                        - int(item.slot_capacity)
                    )
                    * float(self._expert_bytes(item.module))
                    / float(1024 * 1024),
                )
                if installed > 0 and max_mb > 0.0 and installed_mb + layer_reclaim_mb > max_mb:
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
        if self.stats.physical_expert_reclaim_mb + 1e-3 < self._expert_install_target_mb:
            self.stats.planned_expert_reclaim_mb = self.stats.physical_expert_reclaim_mb
            self.stats.policy_semantics_reason = "expert_reclaim_deficit_shifted_to_kvc"
        self._refresh_expert_install_progress()

    def _plan_expert_slot_capacities(self, target_mb: float) -> Dict[int, int]:
        target_bytes = max(0, int(target_mb * 1024 * 1024))
        layer_infos: List[Dict[str, int]] = []
        for layer_id, module in self._expert_modules:
            full_num_experts = int(module.w13_weight.data.shape[0])
            top_k = int(getattr(module, "top_k", 0) or 0)
            if top_k <= 0:
                top_k = int(getattr(module.moe_runner_config, "top_k", 1) or 1)
            min_capacity = max(1, min(full_num_experts, top_k))
            layer_infos.append(
                {
                    "layer_id": int(layer_id),
                    "capacity": full_num_experts,
                    "min_capacity": min_capacity,
                    "expert_bytes": self._expert_bytes(module),
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
            max_expert_bytes = max(1, max(int(info["expert_bytes"]) for info in eligible))
            remaining_bytes = max(0, target_bytes - reclaimed)
            take_count = min(
                len(eligible),
                max(1, int(math.ceil(remaining_bytes / float(max_expert_bytes)))),
            )
            for info in self._evenly_spaced_layer_infos(eligible, take_count):
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
                    expert_id: slot_id for slot_id, expert_id in enumerate(initial_resident)
                }
                slot_to_logical = {
                    slot_id: expert_id for slot_id, expert_id in enumerate(initial_resident)
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
                cpu_params.update(
                    self._copy_experts_to_cpu_for_install_batched(
                        module,
                        param_names,
                        missing_cpu_experts,
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
            name: self._copy_tensor_to_cpu_backing(
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
        return self._expert_global_cpu_backing.get((int(layer_id), int(expert_id))) is params

    def _copy_experts_to_cpu_for_install_batched(
        self, module: Any, param_names: List[str], expert_ids: List[int]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        if not expert_ids:
            return {}
        copied: Dict[int, Dict[str, torch.Tensor]] = {
            int(expert_id): {} for expert_id in expert_ids
        }
        device = getattr(module, param_names[0]).data.device if param_names else None
        with self._profile("profile_copy_expert_to_cpu_ms"):
            with torch.no_grad():
                use_tensor_batch = (
                    len(expert_ids) > 1
                    and float(self.stats.avg_prefix_len) > 2048.0
                )
                if use_tensor_batch:
                    for name in param_names:
                        param = getattr(module, name).data
                        max_chunk_bytes = 256 * 1024 * 1024
                        per_expert_bytes = max(1, int(param[0].nbytes))
                        chunk_size = max(1, max_chunk_bytes // per_expert_bytes)
                        for begin in range(0, len(expert_ids), chunk_size):
                            chunk = expert_ids[begin : begin + chunk_size]
                            idx = torch.tensor(
                                chunk,
                                dtype=torch.long,
                                device=param.device,
                            )
                            selected = param.index_select(0, idx).detach()
                            try:
                                dst = torch.empty(
                                    tuple(selected.shape),
                                    dtype=selected.dtype,
                                    device="cpu",
                                    pin_memory=True,
                                )
                                dst.copy_(selected, non_blocking=True)
                            except Exception:
                                dst = selected.to("cpu", copy=True)
                            for offset, expert_id in enumerate(chunk):
                                copied[int(expert_id)][name] = dst[offset]
                else:
                    for name in param_names:
                        param = getattr(module, name).data
                        for expert_id in expert_ids:
                            tensor = param[int(expert_id)].detach()
                            copied[int(expert_id)][name] = self._copy_tensor_to_cpu_backing(
                                tensor,
                                non_blocking=True,
                            )
                if device is not None and device.type == "cuda":
                    torch.cuda.current_stream(device=device).synchronize()
        self._expert_host_backing_bytes += sum(
            self._expert_backing_bytes(params) for params in copied.values()
        )
        self._refresh_expert_host_backing_stat()
        return copied

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
        evicted_pairs = [(int(logical_id), int(slot_id)) for logical_id, slot_id in evicted]
        use_tensor_batch = (
            self._optimized_profile_enabled()
            and len(evicted_pairs) > 1
            and float(self.stats.avg_prefix_len) > 2048.0
        )
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
                                    dst = torch.empty(
                                        tuple(selected.shape),
                                        dtype=selected.dtype,
                                        device="cpu",
                                        pin_memory=True,
                                    )
                                    dst.copy_(selected, non_blocking=True)
                                except Exception:
                                    dst = selected.to("cpu", copy=True)
                                copied_bytes += int(dst.nbytes)
                                for offset, (logical_id, _slot_id) in enumerate(chunk_pairs):
                                    copied[int(logical_id)][name] = dst[offset]
                        self.stats.expert_eviction_d2h_batched_count += len(evicted_pairs)
                        self.stats.expert_eviction_d2h_batched_mb += copied_bytes / float(
                            1024 * 1024
                        )
                    except Exception:
                        self.stats.expert_eviction_d2h_fallback_count += len(evicted_pairs)
                        copied = {int(logical_id): {} for logical_id, _slot_id in evicted_pairs}
                        copied_bytes = 0
                        for name in state.param_names:
                            param = getattr(state.module, name).data
                            for logical_id, slot_id in evicted_pairs:
                                tensor = param[int(slot_id)].detach()
                                backing = self._copy_tensor_to_cpu_backing(
                                    tensor,
                                    non_blocking=True,
                                )
                                copied[int(logical_id)][name] = backing
                                copied_bytes += int(backing.nbytes)
                else:
                    if evicted_pairs:
                        self.stats.expert_eviction_d2h_fallback_count += len(evicted_pairs)
                    for name in state.param_names:
                        param = getattr(state.module, name).data
                        for logical_id, slot_id in evicted_pairs:
                            tensor = param[int(slot_id)].detach()
                            backing = self._copy_tensor_to_cpu_backing(
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
        self._expert_host_backing_bytes += added
        self._refresh_expert_host_backing_stat()
        return copied

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
        # In memory-constrained policy evaluation, prebuilding expert CPU
        # backing during extend can transiently duplicate a large fraction of
        # MoE weights on host before the decode-side slot shrink has released
        # GPU residency.  Keep the steady-state policy unchanged, but defer
        # backing creation to the layer-by-layer install path unless the user
        # explicitly requested persistent expert CPU backing/cache.
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
        if is_long_context:
            target_mb = min(
                max(0.0, self.config.target_reclaim_mb),
                max(0.0, self._available_expert_reclaim_mb()),
            )
        else:
            target_mb = self._effective_expert_reclaim_mb(forward_batch)
        if target_mb <= 0 or not self.physical_expert_supported or not self._expert_modules:
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
            slot_capacities = self._plan_expert_slot_capacities(target_mb)
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
        reason = (
            f"layer {state.layer_id} dispatch output does not support topk rewrite"
        )
        self.stats.expert_guard_pass = False
        self.stats.expert_guard_reason = reason
        raise RuntimeError(reason)

    def _prepare_expert_layer_for_topk(self, state: _LayerKVExpertLayerState, topk_output: Any) -> Any:
        topk_ids = getattr(topk_output, "topk_ids", None)
        if topk_ids is None:
            return topk_output
        logical_ids = self._unique_expert_ids_and_record_hotness(
            state, topk_ids, record_hotness=not self._expert_plan_applied
        )
        if len(logical_ids) > state.slot_capacity:
            self._grow_expert_layer_slots(state, len(logical_ids))
        self._materialize_experts(state, logical_ids, reason="on_demand")
        self._wait_for_expert_logical_ids_ready(state, logical_ids)
        if self._current_forward_mode == "decode":
            state.last_decode_logical_ids = logical_ids
        valid = (topk_ids >= 0) & (topk_ids < state.full_num_experts)
        safe_ids = topk_ids.clamp(min=0, max=state.full_num_experts - 1).long()
        remap = state.remap_tensor
        if remap is None:
            reason = f"layer {state.layer_id} missing expert remap tensor"
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = reason
            raise RuntimeError(reason)
        rewritten_ids = torch.where(valid, remap[safe_ids].to(topk_ids.dtype), topk_ids)
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

    def _record_expert_hotness_for_layer(
        self, layer_id: int, full_num_experts: int, topk_ids: torch.Tensor
    ) -> None:
        if topk_ids.numel() == 0:
            return
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
        self.stats.expert_call_count_total += total
        if self._current_forward_mode == "decode":
            self.stats.expert_decode_call_count_total += total
        else:
            self.stats.expert_prefill_call_count_total += total
        self.stats.expert_hotness_observed = True

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

    def _unique_expert_ids(self, topk_ids: torch.Tensor, full_num_experts: int) -> List[int]:
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
        with self._profile("profile_expert_unique_ms"):
            if record_hotness:
                values, counts = torch.unique(ids.cpu(), return_counts=True)
            else:
                values = torch.unique(ids).detach().cpu()
                counts = None
        if record_hotness:
            total = 0
            target = (
                self._expert_hotness_decode.setdefault(state.layer_id, {})
                if self._current_forward_mode == "decode"
                else self._expert_hotness_prefill.setdefault(state.layer_id, {})
            )
            for expert_id, count in zip(values.tolist(), counts.tolist()):
                expert_id = int(expert_id)
                count = int(count)
                target[expert_id] = target.get(expert_id, 0) + count
                total += count
            self.stats.expert_call_count_total += total
            if self._current_forward_mode == "decode":
                self.stats.expert_decode_call_count_total += total
            else:
                self.stats.expert_prefill_call_count_total += total
            self.stats.expert_hotness_observed = True
            return [int(x) for x in values.tolist()]
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
                    state.prefetched_logical_ids.difference_update(used_prefetch)
            protected = set(unique_logical_ids)
            materialized: List[Tuple[int, int, Dict[str, torch.Tensor]]] = []
            evicted_to_copy: List[Tuple[int, int]] = []
            micro_profile = False
            for logical_id in unique_logical_ids:
                if logical_id in state.logical_to_slot:
                    t_map = time.perf_counter() if micro_profile else 0.0
                    slot_id = state.logical_to_slot[logical_id]
                    state.lru[logical_id] = self._decode_step
                    if logical_id in state.cpu_params:
                        state.backing_lru[logical_id] = self._decode_step
                    heapq.heappush(
                        state.lru_heap,
                        (state.lru[logical_id], slot_id, logical_id),
                    )
                    if micro_profile:
                        self.stats.expert_materialize_map_update_ms += (
                            time.perf_counter() - t_map
                        ) * 1000.0
                    continue
                t_slot = time.perf_counter() if micro_profile else 0.0
                slot_id = self._choose_expert_slot_for_materialize(state, protected)
                if micro_profile:
                    self.stats.expert_materialize_slot_select_ms += (
                        time.perf_counter() - t_slot
                    ) * 1000.0
                t_map = time.perf_counter() if micro_profile else 0.0
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
                    source_params = (
                        self._global_expert_backing(state.layer_id, int(logical_id))
                        if self._expert_global_cpu_backing
                        else None
                    )
                    if source_params is not None:
                        state.cpu_params[int(logical_id)] = source_params
                if source_params is None:
                    reason = (
                        f"layer {state.layer_id} missing CPU backing for expert {logical_id}"
                    )
                    self.stats.expert_guard_pass = False
                    self.stats.expert_guard_reason = reason
                    raise RuntimeError(reason)
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
                    self.stats.expert_prefetch_mb_total += state.expert_bytes / float(1024 * 1024)
                    state.prefetched_logical_ids.add(int(logical_id))
                else:
                    self.stats.expert_on_demand_materialize_count += 1
                self.stats.layerkv_expert_materialize_started += 1
                self.stats.layerkv_tasks_built += 1
                self.stats.expert_materialize_mb_total += state.expert_bytes / float(1024 * 1024)
                state.backing_lru[int(logical_id)] = self._decode_step
                if micro_profile:
                    self.stats.expert_materialize_map_update_ms += (
                        time.perf_counter() - t_map
                    ) * 1000.0
            if materialized:
                if reason != "prefetch":
                    self._expert_group_dirty_layers.add(int(state.layer_id))
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
        new_capacity = min(state.full_num_experts, max(required_capacity, state.slot_capacity))
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
            if 0 <= slot_id < state.slot_capacity and slot_id not in state.slot_to_logical:
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
        available_tokens = self._allocator_available_size()
        if available_tokens < 0:
            return 0.0
        pairs = self._batch_req_indices_and_lens(forward_batch) if forward_batch is not None else []
        batch_size = len(pairs)

        # Keep enough room for near-term decode allocations and a small amount of
        # scheduler slack.  This is intentionally a headroom threshold, not a
        # fixed reclaim target: if SGLang still has sufficient KV slots, no
        # LayerKV offload is required.
        total_tokens = self._allocator_total_size()
        percent_headroom = int(math.ceil(max(0, total_tokens) * 0.02)) if total_tokens > 0 else 0
        decode_headroom = max(1024, 2 * max(1, batch_size))
        desired_free_tokens = self._align_tokens_down(max(decode_headroom, percent_headroom))
        shortage_tokens = max(0, desired_free_tokens - int(available_tokens))
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
        block_tokens = int(getattr(self.config, "kvc_block_tokens", 0) or 0)
        if block_tokens <= base_page:
            return base_page
        return max(base_page, self._align_tokens_up(block_tokens))

    def _wrap_set_kv_buffer(self, orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapped(layer: Any, loc: Any, cache_k: Any, cache_v: Any, *args, **kwargs):
            t0 = time.perf_counter()
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
        if self.config.kvc_backend != "per-layer-arena":
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
        ok = self._rewrite_attention_metadata_for_layer(layer_id, override)
        if ok:
            self.stats.kvc_per_layer_metadata_rewrite_count += 1
        else:
            self.stats.kvc_per_layer_metadata_rewrite_unsupported_count += 1

    def _refresh_per_layer_kvc_overrides(self) -> None:
        if self.config.kvc_backend != "per-layer-arena":
            return
        if self._req_to_token_pool is None:
            return
        table = getattr(self._req_to_token_pool, "req_to_token", None)
        if table is None:
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
        self.stats.kvc_per_layer_override_layer_count = len(
            self._per_layer_req_to_token_overrides
        )

    def _ensure_owned_per_layer_req_to_token(self, layer_id: int) -> Optional[torch.Tensor]:
        if self.config.kvc_backend != "per-layer-arena":
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
    ) -> bool:
        if not req_indices or not positions or int(new_locs.numel()) == 0:
            return False
        if len(req_indices) != len(positions) or len(req_indices) != int(new_locs.numel()):
            return False
        table = self._ensure_owned_per_layer_req_to_token(int(layer_id))
        if table is None:
            return False
        device = table.device
        req_tensor = torch.tensor(req_indices, dtype=torch.int64, device=device)
        pos_tensor = torch.tensor(positions, dtype=torch.int64, device=device)
        table[req_tensor, pos_tensor] = new_locs.to(device=device, dtype=table.dtype)
        self.stats.kvc_per_layer_slot_override_count += 1
        self.stats.kvc_per_layer_slot_override_token_count += int(new_locs.numel())
        return True

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

    def _rewrite_triton_kv_indices(self, metadata: Any, req_to_token: torch.Tensor) -> bool:
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
                entry.ready_start_event = None
                entry.ready_event = None
                entry.ready_waited = False
                group = self._sync_kvc_group_if_needed(entry)
                if group is not None:
                    group.recover_count += 1
                self.stats.resident_group_recover_count += 1
                did_finalize = True
        self._pending_kvc_reload_events = still_pending
        if did_finalize:
            self._refresh_kvc_residency_stats()

    def on_forward_begin(self, *, mode: str, forward_batch: Any) -> None:
        t0 = time.perf_counter()
        self._current_forward_mode = mode
        self._last_forward_batch = forward_batch
        self._per_layer_kvc_prepared_layers.clear()
        self._refresh_per_layer_kvc_overrides()
        with self._profile("profile_workload_stats_ms"):
            self._refresh_workload_stats(forward_batch)
        with self._profile("profile_finalize_expert_ms"):
            self._finalize_expert_materialize_events(block=False)
        with self._profile("profile_finalize_kvc_ms"):
            self._finalize_reloaded_entries(block=False)
        if mode == "decode":
            with self._profile("profile_apply_expert_plan_ms"):
                self._apply_expert_plan_once(forward_batch)
            self._advance_expert_install_budgeted()
            self.stats.forward_decode_count += 1
            self._decode_step += 1
            self.stats.decode_steps = self._decode_step
            self._run_deadline_scheduler(forward_batch)
        else:
            self.stats.forward_extend_count += 1
            self._drop_entries_for_reqs(forward_batch)
        self.stats.scheduler_invocation_count += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.layerkv_python_overhead_ms += elapsed_ms
        self._add_profile("profile_forward_begin_ms", elapsed_ms)

    def _run_deadline_scheduler(self, forward_batch: Any) -> None:
        if self._simple_profile_enabled():
            with self._profile("profile_kvc_reload_required_ms"):
                self._reload_required_kvc(forward_batch)
            return
        with self._profile("profile_expert_prefetch_ms"):
            tasks = self._build_recovery_tasks(forward_batch)
            self._schedule_recovery_tasks(tasks)
        total = self.stats.scheduler_ready_use_check_count
        if total <= 0:
            self.stats.scheduler_ready_before_use_ratio = 1.0
        else:
            self.stats.scheduler_ready_before_use_ratio = (
                self.stats.scheduler_ready_before_use_count / float(total)
            )

    def _build_kvc_recovery_task(self, forward_batch: Any) -> Optional[_LayerKVRecoveryTask]:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return None
        effective_kvc_reclaim_mb = self._effective_kvc_reclaim_mb(forward_batch)
        if effective_kvc_reclaim_mb <= 0:
            self._refresh_kvc_residency_stats()
            return None
        if not self.physical_kvc_supported:
            self.stats.comparable = False
            self.stats.comparability_reason = self.unsupported_reason
            return None
        if self._bytes_per_token_all_layers <= 0:
            return None
        self._prune_entries_for_active_lengths(forward_batch)
        with self._profile("profile_kvc_select_required_ms"):
            selected = self._select_required_offloaded_entries(forward_batch)
        if not selected:
            self._refresh_kvc_residency_stats()
            return None
        token_count = sum(entry.token_count for entry in selected)
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
        return _LayerKVRecoveryTask(
            kind="kvc",
            layer_id=-1,
            bytes=int(token_count * bytes_per_token),
            deadline_layer=0,
            group_keys=group_keys,
            entries=tuple(selected),
            token_count=int(token_count),
            benefit_score=float(token_count),
        )

    def _build_recovery_tasks(self, forward_batch: Any) -> List[_LayerKVRecoveryTask]:
        tasks: List[_LayerKVRecoveryTask] = []
        kvc_task = self._build_kvc_recovery_task(forward_batch)
        if kvc_task is not None:
            tasks.append(kvc_task)
        if (
            self.config.dynamic_pressure_from_kvc
            and self.stats.effective_reclaim_target_mb <= 128.0
            and self.stats.effective_kvc_reclaim_mb <= 0.0
        ):
            # Small trace-replay pressure commonly resolves to expert-only
            # reclaim.  Look-behind expert prefetch can duplicate the normal
            # on-demand materialization path for these tiny plans; skip it and
            # keep the DP fast path equivalent to the simple expert baseline.
            return tasks
        if not self._expert_plan_applied or not self._expert_layers:
            return tasks
        for layer_id, state in sorted(self._expert_layers.items()):
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
                self.stats.expert_prefetch_hit_count += len(state.last_decode_logical_ids)
                continue
            self.stats.expert_prefetch_candidate_count += len(missing)
            group_keys = []
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
            tasks.append(
                _LayerKVRecoveryTask(
                    kind="expert",
                    layer_id=int(layer_id),
                    bytes=int(len(missing) * state.expert_bytes),
                    deadline_layer=int(layer_id),
                    logical_ids=tuple(missing),
                    group_keys=tuple(group_keys),
                    benefit_score=sum(
                        self._expert_hotness_score(state, expert_id)
                        for expert_id in missing
                    ),
                )
            )
        return tasks

    def _schedule_recovery_tasks(self, tasks: List[_LayerKVRecoveryTask]) -> None:
        if not tasks:
            return
        tasks.sort(key=lambda task: (task.deadline_layer, -task.benefit_score, task.bytes))
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
                    logical_ids=prev.logical_ids + task.logical_ids,
                    group_keys=prev.group_keys + task.group_keys,
                    token_count=prev.token_count + task.token_count,
                    benefit_score=prev.benefit_score + task.benefit_score,
                )
            else:
                coalesced.append(task)
        self.stats.scheduler_task_count += len(coalesced)
        self.stats.scheduler_coalesced_task_count += max(0, len(tasks) - len(coalesced))
        self.stats.expert_materialize_coalesced_count += max(
            0, len(tasks) - len(coalesced)
        )
        for task in coalesced:
            if task.kind == "kvc":
                if not task.entries:
                    continue
                before_kvc_bytes = float(self.stats.kvc_reload_mb_total)
                with self._profile("profile_kvc_reload_required_ms"):
                    self._reload_required_kvc(
                        self._last_forward_batch, selected_entries=list(task.entries)
                    )
                self.stats.scheduler_kvc_task_count += 1
                self.stats.scheduler_copy_bytes_total += int(
                    max(0.0, self.stats.kvc_reload_mb_total - before_kvc_bytes)
                    * 1024
                    * 1024
                )
                continue
            if task.kind != "expert":
                continue
            state = self._expert_layers.get(int(task.layer_id))
            if state is None:
                continue
            logical_ids = list(task.logical_ids)
            if not logical_ids:
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

    def _refresh_workload_stats(self, forward_batch: Any) -> None:
        pairs = self._batch_req_indices_and_lens(forward_batch)
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
    ) -> None:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return
        if selected_entries is None:
            effective_kvc_reclaim_mb = self._effective_kvc_reclaim_mb(forward_batch)
            if effective_kvc_reclaim_mb <= 0:
                self._refresh_kvc_residency_stats()
                return
        if not self.physical_kvc_supported:
            self.stats.comparable = False
            self.stats.comparability_reason = self.unsupported_reason
            return
        if self._bytes_per_token_all_layers <= 0:
            return

        if selected_entries is None:
            self._prune_entries_for_active_lengths(forward_batch)
            with self._profile("profile_kvc_select_required_ms"):
                selected = self._select_required_offloaded_entries(forward_batch)
        else:
            selected = [
                entry
                for entry in selected_entries
                if entry.state == "offloaded" and entry.host_slot_list()
            ]
        if not selected:
            self._refresh_kvc_residency_stats()
            return

        token_count = sum(entry.token_count for entry in selected)
        self.stats.kvc_reload_required_count += token_count
        if self.config.kvc_backend == "per-layer-arena":
            flat_locs = [loc for entry in selected for loc in entry.device_loc_list()]
            new_locs = torch.tensor(
                flat_locs, dtype=torch.int64, device=self._kv_pool.device
            )
            self.stats.kvc_allocator_available_before = self._allocator_available_size()
            self.stats.kvc_allocator_available_after = self._allocator_available_size()
        else:
            self.stats.kvc_allocator_available_before = self._allocator_available_size()
            new_locs = self._allocator.alloc(token_count)
            if new_locs is None:
                self.stats.kvc_physical_failure_count += 1
                self.stats.comparable = False
                self.stats.comparability_reason = "allocator failed to reload offloaded KVC"
                raise RuntimeError("allocator failed to reload offloaded KVC")
            self.stats.kvc_allocator_available_after = self._allocator_available_size()

        host_slots = [slot for entry in selected for slot in entry.host_slot_list()]
        try:
            self._ensure_host_store()
            async_copy = (
                self.config.kvc_scheduler == "async-deadline"
                and self._optimized_profile_enabled()
            )
            if self.config.kvc_backend == "per-layer-arena":
                elapsed_ms, start_event, ready_event = self._host_store.reload_per_layer(
                    selected,
                    stream=self._copy_stream,
                    async_copy=async_copy,
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
                    self._rewrite_per_layer_req_to_token_for_entries(
                        selected, new_locs
                    )
                else:
                    self._rewrite_req_to_token_for_entries(selected, new_locs)
            new_locs_cpu = [int(x) for x in new_locs.detach().cpu().tolist()]
            offset = 0
            for entry in selected:
                page_locs = new_locs_cpu[offset : offset + entry.token_count]
                offset += entry.token_count
                entry.device_locs = page_locs
                entry.device_loc = page_locs[0] if page_locs else None
                entry.last_access_step = self._decode_step
                if async_copy and ready_event is not None:
                    entry.state = "reloading"
                    entry.ready_start_event = start_event
                    entry.ready_event = ready_event
                    entry.ready_waited = False
                else:
                    entry.state = "resident"
                    if entry.host_slots is not None:
                        if self.config.kvc_backend == "per-layer-arena":
                            self._host_store.free_per_layer(entry.layer_id, entry.host_slots)
                        else:
                            self._host_store.free(entry.host_slots)
                    elif entry.host_slot is not None:
                        if self.config.kvc_backend == "per-layer-arena":
                            self._host_store.free_per_layer(entry.layer_id, [int(entry.host_slot)])
                        else:
                            self._host_store.free([int(entry.host_slot)])
                    entry.host_slots = None
                    entry.host_slot = None
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
                token_count
                * self._bytes_per_kvc_token_per_layer()
                / float(1024 * 1024)
            )
        else:
            reload_mb = token_count * self._bytes_per_token_all_layers / float(1024 * 1024)
        self.stats.kvc_reload_count_total += token_count
        self.stats.kvc_reload_page_count_total += len(selected)
        self.stats.kvc_reload_mb_total += reload_mb
        self.stats.kvc_reload_ms += elapsed_ms
        if self.config.kvc_backend == "per-layer-arena":
            self.stats.kvc_per_layer_reload_count += token_count
            self.stats.kvc_per_layer_reload_mb_total += reload_mb
            self.stats.kvc_per_layer_reload_ms += elapsed_ms
        self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
        self.stats.layerkv_kvc_reload_started += 1
        self.stats.layerkv_tasks_built += 1
        self._refresh_kvc_residency_stats()

    def _evict_kvc_to_target(self, forward_batch: Any) -> None:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return
        effective_kvc_reclaim_mb = self._effective_kvc_reclaim_mb(forward_batch)
        if effective_kvc_reclaim_mb <= 0:
            self._refresh_kvc_residency_stats()
            return
        if not self.physical_kvc_supported:
            self.stats.comparable = False
            self.stats.comparability_reason = self.unsupported_reason
            return
        if self._bytes_per_token_all_layers <= 0:
            return

        self._ensure_host_store()
        self._prune_entries_for_active_lengths(forward_batch)
        if self._requires_per_layer_kvc_backend() and self.config.kvc_backend == "token-slot":
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
        if self._uses_layer_aware_kvc_plan() and self._planned_kvc_token_target > 0:
            target_tokens = sum(int(x) for x in self._planned_kvc_tokens_by_layer.values())
        elif self.config.kvc_backend == "per-layer-arena" and self._uses_layer_aware_kvc_plan():
            target_tokens = self._target_offloaded_per_layer_tokens(effective_kvc_reclaim_mb)
        else:
            # Baseline KVC policies use layer-average eviction.  In SGLang's
            # token-to-KV pool, one token slot owns K/V for all layers, so
            # evicting N token slots reclaims the same N-token prefix/suffix
            # from every layer instead of making layer-specific choices.
            target_tokens = self._target_offloaded_tokens(effective_kvc_reclaim_mb)
        need_tokens = max(0, target_tokens - self._offloaded_token_count())
        if need_tokens <= 0:
            self.stats.kvc_eviction_skipped_count += 1
            self._refresh_kvc_residency_stats()
            return

        with self._profile("profile_kvc_select_evict_ms"):
            selected = self._select_resident_entries_for_eviction(forward_batch, need_tokens)
        if not selected:
            self.stats.kvc_eviction_skipped_count += 1
            self._refresh_kvc_residency_stats()
            return

        token_count = sum(entry.token_count for entry in selected)
        if self.config.kvc_backend == "per-layer-arena":
            by_layer: Dict[int, int] = {}
            for entry in selected:
                by_layer[int(entry.layer_id)] = by_layer.get(int(entry.layer_id), 0) + entry.token_count
            if by_layer:
                merged = dict(self._planned_kvc_tokens_by_layer)
                for layer_id, tokens in by_layer.items():
                    merged[layer_id] = max(int(merged.get(layer_id, 0)), int(tokens))
                self.stats.selected_kvc_tokens_by_layer = self._kvc_tokens_by_layer_json(merged)
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

        old_locs = torch.tensor(
            [loc for entry in selected for loc in entry.device_loc_list()],
            dtype=torch.int64,
            device=self._kv_pool.device,
        )
        try:
            if self.config.kvc_backend == "per-layer-arena":
                elapsed_ms = self._host_store.backup_per_layer(selected, host_slots)
                self.stats.kvc_allocator_available_before = self._allocator_available_size()
                self.stats.kvc_allocator_available_after = self._allocator_available_size()
            else:
                elapsed_ms = self._host_store.backup(old_locs, host_slots)
                self.stats.kvc_allocator_available_before = self._allocator_available_size()
                self._allocator.free(old_locs)
                self.stats.kvc_allocator_available_after = self._allocator_available_size()
                self.stats.kvc_allocator_free_count += 1
            offset = 0
            for entry in selected:
                page_host_slots = host_slots[offset : offset + entry.token_count]
                offset += entry.token_count
                entry.state = "offloaded"
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
        self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
        self.stats.kvc_physical_cycle_count += 1
        self.stats.layerkv_tasks_built += 1
        self._refresh_kvc_residency_stats()

    def _target_offloaded_tokens(self, target_reclaim_mb: float) -> int:
        target_bytes = int(target_reclaim_mb * 1024 * 1024)
        raw_tokens = max(1, target_bytes // self._bytes_per_token_all_layers)
        return max(self._page_size, self._align_tokens_up(raw_tokens))

    def _target_offloaded_per_layer_tokens(self, target_reclaim_mb: float) -> int:
        bytes_per_token = self._bytes_per_kvc_token_per_layer()
        if bytes_per_token <= 0:
            return 0
        target_bytes = int(target_reclaim_mb * 1024 * 1024)
        raw_tokens = max(1, target_bytes // bytes_per_token)
        return max(self._page_size, self._align_tokens_up(raw_tokens))

    def _offloaded_token_count(self) -> int:
        if self.config.kvc_backend == "per-layer-arena":
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

    def _resident_token_count(self) -> int:
        if self.config.kvc_backend == "per-layer-arena":
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
        entries = (
            self._per_layer_residency
            if self.config.kvc_backend == "per-layer-arena"
            else self._residency
        )
        self.stats.kvc_residency_entry_count = len(entries)
        self.stats.kvc_per_layer_arena_entry_count = len(self._per_layer_residency)
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

        active_lens = None
        if forward_batch is not None:
            active_lens = {
                req_idx: seq_len
                for req_idx, seq_len in self._batch_req_indices_and_lens(forward_batch)
            }

        if self._host_store is not None and self.config.kvc_backend == "per-layer-arena":
            host_used_slots = {
                (self._host_store.start_layer + layer_offset, int(slot))
                for layer_offset, slots in enumerate(self._host_store.layer_used_slots)
                for slot in slots
            }
        else:
            host_used_slots = (
                set(self._host_store.used_slots) if self._host_store is not None else set()
            )

        if self.config.kvc_backend == "per-layer-arena":
            iter_entries = [
                ((entry.layer_id, entry.req_idx, entry.pos), entry)
                for entry in self._per_layer_residency.values()
            ]
        else:
            iter_entries = [
                ((entry.req_idx, entry.pos), entry) for entry in self._residency.values()
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
                    elif self._host_store is not None and host_key not in host_used_slots:
                        stale_count += 1
                        reasons.append("host_slot_not_marked_used")
                    seen_host_slots.add(host_key)
                if entry.device_loc is not None:
                    stale_count += 1
                    reasons.append("offloaded_has_device_loc")
                if len(host_slots) != entry.token_count:
                    stale_count += 1
                    reasons.append("offloaded_host_slot_count_mismatch")
            elif entry.state in ("resident", "reloading"):
                if entry.state == "reloading":
                    host_owned_count += len(entry.host_slot_list())
                if entry.state == "resident":
                    resident_count += entry.token_count
                device_locs = entry.device_loc_list()
                if not device_locs:
                    stale_count += 1
                    reasons.append("resident_missing_device_loc")
                if entry.state == "resident" and entry.host_slot is not None:
                    stale_count += 1
                    reasons.append("resident_has_host_slot")
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
                            and int(entry.layer_id) in self._per_layer_req_to_token_overrides
                        ):
                            table = self._per_layer_req_to_token_overrides[int(entry.layer_id)]
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
                    except Exception:
                        stale_count += 1
                        reasons.append("resident_req_to_token_check_failed")
            else:
                stale_count += 1
                reasons.append("unknown_entry_state")

        if self._host_store is not None and self._host_store.used_count != host_owned_count:
            stale_count += abs(self._host_store.used_count - host_owned_count)
            reasons.append("host_used_count_mismatch")

        self.stats.kvc_residency_entry_count = len(iter_entries)
        self.stats.kvc_stale_entry_count = stale_count
        self.stats.kvc_offloaded_token_count = offloaded_count
        self.stats.kvc_resident_token_count = resident_count
        self.stats.kvc_host_used_tokens = (
            self._host_store.used_count if self._host_store is not None else 0
        )

        guard_pass = stale_count == 0
        needs_kvc_reclaim = self.config.mode == "kvc-only" and self.config.target_reclaim_mb > 0
        needs_kvc_reclaim = needs_kvc_reclaim or self.stats.planned_kvc_reclaim_mb > 0
        needs_kvc_reclaim = needs_kvc_reclaim or self.stats.effective_kvc_reclaim_mb > 0
        if needs_kvc_reclaim and not self.physical_kvc_supported:
            guard_pass = False
            reasons.append(self.unsupported_reason or "physical_kvc_unsupported")

        self.stats.kvc_guard_pass = guard_pass
        self.stats.kvc_guard_reason = ";".join(sorted(set(reasons)))
        return {
            "kvc_guard_pass": self.stats.kvc_guard_pass,
            "kvc_guard_reason": self.stats.kvc_guard_reason,
            "kvc_stale_entry_count": self.stats.kvc_stale_entry_count,
            "kvc_residency_entry_count": self.stats.kvc_residency_entry_count,
            "kvc_host_used_tokens": self.stats.kvc_host_used_tokens,
            "kvc_offloaded_token_count": self.stats.kvc_offloaded_token_count,
            "kvc_resident_token_count": self.stats.kvc_resident_token_count,
        }

    def _batch_req_indices_and_lens(self, forward_batch: Any) -> List[Tuple[int, int]]:
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if seq_lens_cpu is None or req_pool_indices is None:
            return []
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
        per_layer_to_drop = [
            key for key in self._per_layer_residency if key[1] == req_idx
        ]
        if not to_drop and not per_layer_to_drop:
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
            key for key, entry in self._per_layer_residency.items()
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

    def _drop_residency_keys(self, keys: List[Tuple[int, int]]) -> None:
        if not keys:
            return
        host_slots = []
        for key in keys:
            entry = self._residency.pop(key, None)
            if entry is not None:
                if entry.ready_event is not None and not entry.ready_event.query():
                    entry.ready_event.synchronize()
                host_slots.extend(entry.host_slot_list())
                self._remove_resident_group("kvc", -1, (int(entry.req_idx), int(entry.pos)))
        if host_slots and self._host_store is not None:
            self._host_store.free(host_slots)
        self._refresh_kvc_residency_stats()

    def _drop_per_layer_residency_keys(self, keys: List[Tuple[int, int, int]]) -> None:
        if not keys:
            return
        for key in keys:
            entry = self._per_layer_residency.pop(key, None)
            if entry is not None:
                if entry.ready_event is not None and not entry.ready_event.query():
                    entry.ready_event.synchronize()
                host_slots = entry.host_slot_list()
                if host_slots and self._host_store is not None:
                    self._host_store.free_per_layer(entry.layer_id, host_slots)
                self._remove_resident_group(
                    "kvc",
                    int(entry.layer_id),
                    (int(entry.layer_id), int(entry.req_idx), int(entry.pos)),
                )
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
        seen = set()
        page_size = self._per_layer_kvc_block_page_size()
        for req_idx, seq_len in self._batch_req_indices_and_lens(forward_batch):
            required_prefix_len = self._align_tokens_down(max(0, seq_len - 1))
            for layer_id in self._kvc_layer_ids():
                for pos in range(0, required_prefix_len, page_size):
                    key = (int(layer_id), req_idx, pos)
                    entry = self._per_layer_residency.get(key)
                    if entry is None or entry.state != "offloaded":
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    selected.append(entry)
        return selected

    def _select_resident_entries_for_eviction(
        self, forward_batch: Any, max_tokens: int
    ) -> List[_LayerKVResidencyEntry]:
        if self.config.kvc_backend == "per-layer-arena" and self._uses_layer_aware_kvc_plan():
            return self._select_per_layer_resident_entries_for_eviction(
                forward_batch, max_tokens
            )
        table = self._req_to_token_pool.req_to_token
        pairs = self._batch_req_indices_and_lens(forward_batch)
        page_size = max(1, int(self._page_size))
        max_tokens = self._align_tokens_down(max_tokens)
        target_pages = max_tokens // page_size
        if target_pages <= 0:
            return []
        block_tokens = max(page_size, self._align_tokens_up(self.config.kvc_block_tokens))
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
            total_pages = sum(evictable_len // page_size for _req, evictable_len, _cur in evictable)
            scan_budget_pages = min(total_pages, max(target_pages * 4, target_pages + 1024))
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

        def materialize_candidates(page_meta: List[Tuple[int, int]]) -> List[_LayerKVResidencyEntry]:
            if not page_meta:
                return []
            self.stats.kvc_evict_candidate_scan_tokens += len(page_meta) * page_size
            flat_req_indices: List[int] = []
            flat_positions: List[int] = []
            for req_idx, pos in page_meta:
                for page_pos in range(pos, pos + page_size):
                    flat_req_indices.append(req_idx)
                    flat_positions.append(page_pos)
            req_tensor = torch.tensor(flat_req_indices, dtype=torch.int64, device=table.device)
            pos_tensor = torch.tensor(flat_positions, dtype=torch.int64, device=table.device)
            flat_locs = table[req_tensor, pos_tensor].detach().cpu().tolist()
            candidates: List[_LayerKVResidencyEntry] = []
            seen_locs = set()
            loc_offset = 0
            for req_idx, pos in page_meta:
                locs = [int(x) for x in flat_locs[loc_offset : loc_offset + page_size]]
                loc_offset += page_size
                key = (req_idx, pos)
                existing = self._residency.get(key)
                if existing is not None and existing.state in ("offloaded", "reloading"):
                    continue
                if any(loc <= 0 for loc in locs) or any(loc in seen_locs for loc in locs):
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
        if len(candidates) < target_pages:
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

    def _select_per_layer_resident_entries_for_eviction(
        self, forward_batch: Any, max_tokens: int
    ) -> List[_LayerKVResidencyEntry]:
        table = self._req_to_token_pool.req_to_token
        pairs = self._batch_req_indices_and_lens(forward_batch)
        page_size = self._per_layer_kvc_block_page_size()
        plan = dict(self._planned_kvc_tokens_by_layer)
        if not plan and self._planned_kvc_token_target > 0:
            plan = self._build_layer_aware_kvc_token_plan(
                self._planned_kvc_token_target, forward_batch
            )
        if not plan and max_tokens > 0:
            plan = self._build_layer_aware_kvc_token_plan(max_tokens, forward_batch)
        selected: List[_LayerKVResidencyEntry] = []
        for layer_id, target_tokens in sorted(plan.items()):
            current = sum(
                entry.token_count
                for key, entry in self._per_layer_residency.items()
                if key[0] == int(layer_id) and entry.state == "offloaded"
            )
            need_tokens = self._align_tokens_down(max(0, int(target_tokens) - current))
            if need_tokens <= 0:
                continue
            target_pages = need_tokens // page_size
            if target_pages <= 0:
                continue
            page_meta: List[Tuple[int, int]] = []
            for req_idx, seq_len in pairs:
                evictable_len = self._align_tokens_down(max(0, seq_len - 1))
                for pos in range(0, evictable_len, page_size):
                    page_meta.append((req_idx, pos))
                    if len(page_meta) >= target_pages:
                        break
                if len(page_meta) >= target_pages:
                    break
            if not page_meta:
                continue
            flat_req_indices: List[int] = []
            flat_positions: List[int] = []
            for req_idx, pos in page_meta:
                for page_pos in range(pos, pos + page_size):
                    flat_req_indices.append(req_idx)
                    flat_positions.append(page_pos)
            req_tensor = torch.tensor(flat_req_indices, dtype=torch.int64, device=table.device)
            pos_tensor = torch.tensor(flat_positions, dtype=torch.int64, device=table.device)
            flat_locs = table[req_tensor, pos_tensor].detach().cpu().tolist()
            loc_offset = 0
            for req_idx, pos in page_meta:
                key = (int(layer_id), req_idx, pos)
                existing = self._per_layer_residency.get(key)
                if existing is not None and existing.state in ("offloaded", "reloading"):
                    loc_offset += page_size
                    continue
                locs = [int(x) for x in flat_locs[loc_offset : loc_offset + page_size]]
                loc_offset += page_size
                if any(loc <= 0 for loc in locs):
                    continue
                # Per-layer arena blocks are logical token ranges, not physical
                # contiguous KV slots. SGLang's req_to_token table may map a
                # contiguous logical range to non-contiguous physical slots, and
                # the arena stores the exact loc list for restore/rewrite.
                if existing is None:
                    existing = _LayerKVResidencyEntry(
                        req_idx=req_idx,
                        pos=pos,
                        state="resident",
                        layer_id=int(layer_id),
                        device_loc=locs[0],
                        device_locs=locs,
                        page_size=page_size,
                        last_access_step=self._decode_step,
                    )
                    self._per_layer_residency[key] = existing
                else:
                    existing.state = "resident"
                    existing.layer_id = int(layer_id)
                    existing.device_loc = locs[0]
                    existing.device_locs = locs
                    existing.page_size = page_size
                    existing.last_access_step = self._decode_step
                self._sync_kvc_group_if_needed(existing)
                selected.append(existing)
        selected.sort(key=lambda x: (x.layer_id, x.pos, x.req_idx))
        self.stats.kvc_evict_candidate_selected_tokens += sum(
            entry.token_count for entry in selected
        )
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
            loc_tensor = torch.tensor(
                locs, dtype=torch.int64, device=new_locs.device
            )
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
            with self._profile("profile_finalize_expert_ms"):
                self._finalize_expert_materialize_events(block=False)
            with self._profile("profile_finalize_kvc_ms"):
                self._finalize_reloaded_entries(block=True)
            with self._profile("profile_kvc_evict_to_target_ms"):
                self._evict_kvc_to_target(forward_batch)
            self._refresh_physical_reclaim_peaks(record_step_sample=True)
            self._refresh_resident_group_stats()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.stats.layerkv_python_overhead_ms += elapsed_ms
            self._add_profile("profile_forward_end_ms", elapsed_ms)
        elif mode == "extend":
            t0 = time.perf_counter()
            self._maybe_prepare_expert_plan_during_extend(forward_batch)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms > 0.0:
                self.stats.layerkv_python_overhead_ms += elapsed_ms
                self._add_profile("profile_forward_end_ms", elapsed_ms)
        if self.config.debug_stats and self._should_log_stats(mode):
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
        if self.stats.configured_target_reclaim_mb <= 0.0 and self.config.target_reclaim_mb > 0.0:
            self._refresh_reclaim_target_stats(self._last_forward_batch)
        self._refresh_physical_reclaim_peaks()
        if validate:
            self._refresh_resident_group_stats()
            self.validate_kvc_state()
            self._validate_resident_groups()
        else:
            self._refresh_resident_group_stats()
        planner_inputs: Dict[str, Any] = {}
        if include_planner_inputs:
            planner_inputs = self._planner_input_summary()
        if self.config.profile_detail:
            self._add_profile("profile_summary_build_ms", (time.perf_counter() - t0) * 1000.0)
            self._refresh_profile_derived()
        out = self.stats.as_dict()
        out.update(planner_inputs)
        out.update(
            {
                "layerkv_enabled": self.config.enabled,
                "layerkv_mode": self.config.mode,
                "layerkv_policy": self.config.policy,
                "layerkv_target_reclaim_mb": self.config.target_reclaim_mb,
                "layerkv_kvc_backend": self.config.kvc_backend,
                "layerkv_kvc_scheduler": self.config.kvc_scheduler,
                "layerkv_runtime_profile": self.config.runtime_profile,
                "layerkv_physical_kvc_supported": self.physical_kvc_supported,
                "layerkv_physical_expert_supported": self.physical_expert_supported,
                "layerkv_expert_layer_count": len(self._expert_layers),
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
