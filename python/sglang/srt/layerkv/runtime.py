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
import faulthandler
import functools
import heapq
import json
import logging
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

if __package__:
    from .config_stats import LayerKVConfig, LayerKVStats
    from .common_types import (
        _LayerKVExpertCopyDescriptor,
        _LayerKVExpertDemand,
        _LayerKVExpertInstallBuild,
        _LayerKVExpertInstallD2HJob,
        _LayerKVExpertInstallItem,
        _LayerKVExpertLayerState,
        _LayerKVKvcDemand,
        _LayerKVMetadataPatchCacheEntry,
        _LayerKVNativeKvcRun,
        _LayerKVPerLayerReqCleanupState,
        _LayerKVPendingEviction,
        _LayerKVPendingExpertCopy,
        _LayerKVPendingExpertD2H,
        _LayerKVPendingReload,
        _LayerKVPendingVirtualMaterialize,
        _LayerKVRecoveryTask,
        _LayerKVResidencyEntry,
        _LayerKVResidencyHandle,
        _LayerKVResidencyKey,
        _LayerKVResidentTensorGroup,
        _LayerKVVirtualMaterializePlan,
        _LayerKVVirtualScratchCacheEntry,
    )
    from .host_store import _LayerKVHostKVStore
    from .planner import LayerKVPlannerMixin
    from .expert_runtime import LayerKVExpertMixin
    from .kvc_allocator import LayerKVKvcAllocatorMixin
    from .kvc_virtual import LayerKVKvcVirtualMixin
    from .kvc_residency import LayerKVKvcResidencyMixin
    from .scheduler_runtime import LayerKVSchedulerMixin
    from .kvc_reclaim import LayerKVKvcReclaimMixin
else:  # pragma: no cover - direct file-loading smoke tests.
    sys.path.append(str(Path(__file__).resolve().parent))
    from config_stats import LayerKVConfig, LayerKVStats
    from common_types import (
        _LayerKVExpertCopyDescriptor,
        _LayerKVExpertDemand,
        _LayerKVExpertInstallBuild,
        _LayerKVExpertInstallD2HJob,
        _LayerKVExpertInstallItem,
        _LayerKVExpertLayerState,
        _LayerKVKvcDemand,
        _LayerKVMetadataPatchCacheEntry,
        _LayerKVNativeKvcRun,
        _LayerKVPerLayerReqCleanupState,
        _LayerKVPendingEviction,
        _LayerKVPendingExpertCopy,
        _LayerKVPendingExpertD2H,
        _LayerKVPendingReload,
        _LayerKVPendingVirtualMaterialize,
        _LayerKVRecoveryTask,
        _LayerKVResidencyEntry,
        _LayerKVResidencyHandle,
        _LayerKVResidencyKey,
        _LayerKVResidentTensorGroup,
        _LayerKVVirtualMaterializePlan,
        _LayerKVVirtualScratchCacheEntry,
    )
    from host_store import _LayerKVHostKVStore
    from planner import LayerKVPlannerMixin
    from expert_runtime import LayerKVExpertMixin
    from kvc_allocator import LayerKVKvcAllocatorMixin
    from kvc_virtual import LayerKVKvcVirtualMixin
    from kvc_residency import LayerKVKvcResidencyMixin
    from scheduler_runtime import LayerKVSchedulerMixin
    from kvc_reclaim import LayerKVKvcReclaimMixin

logger = logging.getLogger(__name__)

try:
    from sglang.jit_kernel.layerkv_expert_remap import layerkv_expert_remap
except Exception:  # pragma: no cover - optional JIT helper.
    layerkv_expert_remap = None


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


class LayerKVRuntime(
    LayerKVPlannerMixin,
    LayerKVExpertMixin,
    LayerKVKvcAllocatorMixin,
    LayerKVKvcVirtualMixin,
    LayerKVKvcResidencyMixin,
    LayerKVSchedulerMixin,
    LayerKVKvcReclaimMixin,
):
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
        if self.config.debug_stats:
            try:
                faulthandler.register(signal.SIGUSR2, all_threads=True, chain=False)
            except Exception:
                pass
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
        self._expert_d2h_stream: Optional[torch.cuda.Stream] = None
        self._expert_h2d_stream: Optional[torch.cuda.Stream] = None
        self._expert_install_stream: Optional[torch.cuda.Stream] = None
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
        self._per_layer_offloaded_version_by_layer: Dict[int, int] = {}
        self._per_layer_offloaded_sorted_by_req: Dict[
            int, Tuple[int, Tuple[Tuple[int, int, int], ...]]
        ] = {}
        self._per_layer_offloaded_sorted_by_req_layer: Dict[
            Tuple[int, int], Tuple[int, Tuple[Tuple[int, int, int], ...]]
        ] = {}
        self._per_layer_offloaded_positions_by_req_layer: Dict[
            Tuple[int, int], Tuple[int, Tuple[int, ...]]
        ] = {}
        self._per_layer_offloaded_runs_by_req_layer: Dict[
            Tuple[int, int], List[_LayerKVResidencyEntry]
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
        self._expert_compact_tensor_pool: Dict[
            Tuple[str, torch.dtype, Tuple[int, ...]], List[torch.Tensor]
        ] = {}
        self._expert_compact_tensor_pool_bytes: int = 0
        self._expert_topk_gpu_remap_disabled: bool = False
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
        self._kvc_evict_finalized_this_step_tokens: int = 0
        self._last_kvc_prune_active_lens: Optional[Dict[int, int]] = None
        self._pending_virtual_kvc_materialize: Dict[
            Tuple[Any, ...], _LayerKVPendingVirtualMaterialize
        ] = {}
        self._per_layer_req_to_token_versions: Dict[int, int] = {}
        self._per_layer_kvc_current_metadata_key: Optional[Tuple[Any, ...]] = None
        self._virtual_materialize_plan: Optional[_LayerKVVirtualMaterializePlan] = None
        self._virtual_materialize_plans_by_layer: Dict[
            int, _LayerKVVirtualMaterializePlan
        ] = {}
        self._virtual_materialize_plan_reuse_cache: Dict[
            Tuple[Any, ...], _LayerKVVirtualMaterializePlan
        ] = {}
        self._virtual_batch_index_step: int = -1
        self._virtual_batch_index_cache: Optional[
            Tuple[
                List[Tuple[int, int]],
                Dict[int, int],
                Dict[int, int],
                Dict[int, int],
            ]
        ] = None
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
        self._per_layer_kvc_evict_finalized_before_layers: Set[Tuple[int, int]] = set()
        self._per_layer_arena_reserved_locs: Set[int] = set()
        self._per_layer_arena_common_free_locs: Set[int] = set()
        self._per_layer_arena_common_free_order: List[int] = []
        self._per_layer_arena_common_free_dirty = False
        self._per_layer_arena_free_locs: Dict[int, List[int]] = {}
        self._per_layer_arena_allocated_locs: Dict[int, Set[int]] = {}
        self._per_layer_arena_protected_locs: Dict[int, Set[int]] = {}
        self._per_layer_arena_overwrite_bitmaps: Dict[int, torch.Tensor] = {}
        self._per_layer_arena_overwrite_counts: Dict[int, int] = {}
        self._per_layer_arena_overwrite_pending_locs: Dict[int, List[int]] = {}
        self._per_layer_arena_overwrite_pending_bits: Dict[int, int] = {}
        self._per_layer_arena_overwrite_pending_bit_chunks: Dict[int, List[int]] = {}
        self._per_layer_arena_overwrite_pending_bit_counts: Dict[int, int] = {}
        self._per_layer_canonical_to_physical: Dict[int, Dict[int, int]] = {}
        self._per_layer_physical_to_canonical: Dict[int, Dict[int, int]] = {}
        self._per_layer_non_identity_mapping: Set[int] = set()
        self._per_layer_owned_req_indices: Set[int] = set()
        self._per_layer_owned_keys_by_req: Dict[int, Set[Tuple[int, int, int]]] = {}
        self._per_layer_req_generations: Dict[int, int] = {}
        self._per_layer_released_req_generations: Dict[int, Set[int]] = {}
        self._per_layer_cleanup_state_by_req: Dict[
            int, _LayerKVPerLayerReqCleanupState
        ] = {}
        self._per_layer_cleanup_layers_by_req: Dict[int, Set[int]] = {}
        self._per_layer_cleanup_locs_by_req_layer: Dict[Tuple[int, int], int] = {}
        self._per_layer_cleanup_loc_count_by_req_layer: Dict[Tuple[int, int], int] = {}
        self._per_layer_cleanup_host_slots_by_req_layer: Dict[Tuple[int, int], int] = {}
        self._per_layer_cleanup_host_slot_count_by_req_layer: Dict[
            Tuple[int, int], int
        ] = {}
        self._per_layer_cleanup_tracked_keys_by_req: Dict[
            int, Set[Tuple[int, int, int]]
        ] = {}
        self._per_layer_cleanup_token_count_by_req: Dict[int, int] = {}
        self._native_kvc_runs_by_req: Dict[int, List[_LayerKVNativeKvcRun]] = {}
        self._virtual_scratch_locs: Optional[torch.Tensor] = None
        self._virtual_scratch_buffers: List[torch.Tensor] = []
        self._virtual_scratch_buffer_slices: List[Tuple[int, int]] = []
        self._virtual_scratch_buffer_generations: List[int] = []
        self._virtual_scratch_buffer_generations_by_layer: Dict[
            Tuple[int, int], int
        ] = {}
        self._virtual_materialize_recorded_event_groups: Set[int] = set()
        self._virtual_batched_prefetch_step: int = -1
        self._virtual_scratch_single_generation: int = 0
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
        self._last_decode_step_profile: Dict[str, float] = {}
        self._layerkv_prewarm_done: bool = False
        self._layerkv_prewarm_running: bool = False
        self._layerkv_hotness_prewarm_done: bool = False
        self._prepared_kvc_evict_entries: List[_LayerKVResidencyEntry] = []
        self._prepared_kvc_evict_signature: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._prepared_kvc_evict_cursors: Dict[int, int] = {}
        self._prepared_kvc_evict_cursors_by_layer: Dict[Tuple[int, int], int] = {}
        self._prepared_kvc_evict_host_signature: Tuple[int, int, int, int] = (
            0,
            0,
            0,
            0,
        )
        self._prepared_kvc_evict_host_entries: List[_LayerKVResidencyEntry] = []
        self._prepared_kvc_evict_host_slots: List[int] = []
        self._prepared_kvc_evict_host_slots_by_entry: Dict[int, List[int]] = {}
        self._prepared_kvc_evict_schedule: List[
            Tuple[
                List[_LayerKVResidencyEntry],
                Dict[int, int],
                Dict[Tuple[int, int], int],
                int,
            ]
        ] = []
        self._prepared_kvc_evict_schedule_signature: Tuple[int, int, int, int] = (
            0,
            0,
            0,
            0,
        )

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
                self._prepare_expert_install_d2h_lookahead(prealloc=True)
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
                self.stats.target_limited_reason = "kvc_reclaim_not_scheduler_visible"
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
            self._log_decode_step_profile()
        elif mode == "extend":
            t0 = time.perf_counter()
            self._maybe_prepare_expert_plan_during_extend(forward_batch)
            hotness_prewarmed = self._maybe_prewarm_hotness_once(
                forward_batch, phase="extend"
            )
            prewarmed = self._maybe_prewarm_layerkv_once(forward_batch, phase="extend")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms > 0.0:
                self.stats.layerkv_python_overhead_ms += elapsed_ms
                self._add_profile("profile_forward_end_ms", elapsed_ms)
            if prewarmed or hotness_prewarmed:
                self._reset_decode_step_profile_baseline()
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

    def _decode_step_profile_fields(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        return (
            (
                "layerkv_python_overhead_ms",
                "profile_set_kv_ms",
                "profile_forward_begin_ms",
                "profile_forward_begin_bookkeeping_ms",
                "profile_forward_begin_metadata_current_ms",
                "profile_forward_begin_fastpath_check_ms",
                "profile_forward_begin_expert_control_ms",
                "profile_forward_begin_hotness_prepare_ms",
                "profile_forward_begin_scheduler_control_ms",
                "profile_forward_begin_extend_cleanup_ms",
                "profile_cleanup_req_indices_ms",
                "profile_cleanup_cursor_ms",
                "profile_cleanup_global_scan_ms",
                "profile_cleanup_global_drop_ms",
                "profile_cleanup_per_layer_keys_ms",
                "profile_cleanup_per_layer_drop_ms",
                "profile_cleanup_per_layer_loop_ms",
                "profile_cleanup_per_layer_run_range_ms",
                "profile_cleanup_per_layer_event_sync_ms",
                "profile_cleanup_per_layer_collect_ms",
                "profile_cleanup_per_layer_free_locs_ms",
                "profile_cleanup_per_layer_refresh_allocator_ms",
                "profile_cleanup_per_layer_host_free_ms",
                "profile_cleanup_refresh_stats_ms",
                "profile_cleanup_drop_entries_ms",
                "profile_cleanup_finished_ms",
                "profile_cleanup_release_ms",
                "profile_forward_end_ms",
                "profile_workload_stats_ms",
                "profile_apply_expert_plan_ms",
                "profile_expert_prefetch_ms",
                "profile_recovery_task_build_ms",
                "profile_recovery_task_schedule_ms",
                "profile_kvc_evict_to_target_ms",
                "profile_kvc_evict_target_calc_ms",
                "profile_kvc_evict_prune_ms",
                "profile_kvc_evict_backend_check_ms",
                "profile_kvc_evict_prepared_consume_ms",
                "profile_kvc_evict_schedule_consume_ms",
                "profile_kvc_evict_schedule_active_lens_ms",
                "profile_kvc_evict_schedule_entry_check_ms",
                "profile_kvc_evict_schedule_capacity_check_ms",
                "profile_kvc_select_evict_ms",
                "profile_kvc_evict_selected_stats_ms",
                "profile_kvc_evict_prepare_entries_ms",
                "profile_kvc_evict_host_prealloc_ms",
                "profile_kvc_evict_async_mark_ms",
                "profile_kvc_reload_required_ms",
                "profile_kvc_backup_wall_ms",
                "profile_kvc_evict_commit_ms",
                "profile_kvc_layer_evict_finalize_ms",
                "profile_kvc_layer_evict_commit_ms",
                "profile_kvc_free_locs_ms",
                "profile_kvc_host_alloc_ms",
                "profile_kvc_evict_staging_alloc_ms",
                "profile_kvc_cost_observe_ms",
                "profile_kvc_refresh_stats_ms",
                "profile_virtual_select_ms",
                "profile_virtual_plan_cache_lookup_ms",
                "profile_virtual_plan_sort_ms",
                "profile_virtual_host_order_ms",
                "profile_virtual_plan_key_ms",
                "profile_virtual_plan_lookup_only_ms",
                "profile_virtual_index_build_ms",
                "profile_virtual_direct_metadata_patch_ms",
                "profile_virtual_prefetch_issue_ms",
                "profile_virtual_prefetch_next_layer_ms",
                "profile_virtual_prefetch_demand_ms",
                "profile_virtual_prefetch_cache_lookup_ms",
                "profile_virtual_prefetch_buffer_select_ms",
                "profile_virtual_prefetch_setup_ms",
                "profile_virtual_prefetch_reload_call_ms",
                "profile_virtual_prefetch_record_ms",
                "profile_virtual_reload_issue_wall_ms",
                "profile_virtual_reload_event_record_ms",
                "profile_virtual_reload_slice_path_ms",
                "profile_virtual_reload_span_path_ms",
                "profile_virtual_reload_span_coalesce_check_ms",
                "profile_virtual_reload_span_bulk_copy_ms",
                "profile_virtual_reload_span_fused_ms",
                "profile_virtual_reload_span_scatter_ms",
                "profile_virtual_reload_index_path_ms",
                "profile_virtual_reload_tensor_h2d_ms",
                "profile_virtual_reload_device_write_ms",
                "profile_virtual_reload_sync_wall_ms",
                "profile_expert_hotness_record_ms",
                "layerkv_copy_stream_busy_ms",
                "scheduler_exposed_wait_ms",
                "layerkv_main_stream_wait_ms",
                "virtual_kvc_materialize_ms",
            ),
            (
                "kvc_evict_count_total",
                "kvc_reload_count_total",
                "virtual_kvc_materialize_count",
                "virtual_kvc_materialize_token_count",
                "virtual_kvc_plan_cache_hit_count",
                "virtual_kvc_plan_cache_miss_count",
                "virtual_kvc_plan_cache_reuse_token_count",
                "virtual_kvc_host_order_plan_count",
                "virtual_kvc_host_order_plan_token_count",
                "virtual_kvc_reload_slice_count",
                "virtual_kvc_reload_slice_token_count",
                "virtual_kvc_reload_span_count",
                "virtual_kvc_reload_span_token_count",
                "virtual_kvc_reload_span_segment_count",
                "virtual_kvc_reload_direct_count",
                "virtual_kvc_reload_direct_token_count",
                "virtual_kvc_reload_index_count",
                "virtual_kvc_reload_index_token_count",
                "virtual_kvc_reload_async_count",
                "virtual_kvc_reload_sync_count",
                "virtual_kvc_reload_span_coalesce_count",
                "virtual_kvc_reload_span_coalesce_token_count",
                "virtual_kvc_reload_span_coalesce_copied_token_count",
                "virtual_kvc_reload_span_direct_copy_count",
                "virtual_kvc_reload_span_direct_copy_token_count",
                "virtual_kvc_reload_span_fused_count",
                "virtual_kvc_reload_span_fused_token_count",
                "virtual_kvc_reload_span_fused_fallback_count",
                "virtual_kvc_prefetch_count",
                "virtual_kvc_prefetch_ready_before_use_count",
                "virtual_kvc_prefetch_fallback_sync_count",
                "virtual_kvc_persistent_cache_hit_count",
                "virtual_kvc_persistent_cache_miss_count",
                "virtual_kvc_scratch_generation_miss_count",
                "scheduler_task_count",
                "scheduler_kvc_task_count",
                "scheduler_kvc_deferred_count",
                "kvc_layerwise_scheduler_deadline_reject_count",
                "kvc_layer_evict_commit_count",
                "kvc_layer_evict_commit_token_count",
                "kvc_evict_host_prealloc_count",
                "kvc_evict_host_prealloc_hit_count",
                "kvc_evict_host_prealloc_invalid_count",
                "kvc_evict_host_prealloc_token_count",
                "profile_cleanup_drop_entries_call_count",
                "profile_cleanup_drop_entries_req_count",
                "profile_cleanup_drop_entries_key_count",
                "profile_cleanup_drop_entries_token_count",
                "profile_cleanup_drop_entries_max_req_idx",
                "profile_cleanup_drop_entries_max_req_key_count",
                "profile_cleanup_drop_entries_max_req_token_count",
                "profile_cleanup_drop_entries_max_req_seq_len",
                "profile_cleanup_finished_call_count",
                "profile_cleanup_finished_key_count",
                "profile_cleanup_finished_token_count",
                "profile_cleanup_release_call_count",
                "profile_cleanup_release_key_count",
                "profile_cleanup_release_token_count",
            ),
        )

    def _decode_step_profile_snapshot(self) -> Dict[str, float]:
        fields, counters = self._decode_step_profile_fields()
        current: Dict[str, float] = {}
        for field in fields + counters:
            try:
                current[field] = float(getattr(self.stats, field))
            except Exception:
                current[field] = 0.0
        return current

    def _reset_decode_step_profile_baseline(self) -> None:
        if self.config.debug_stats and self.config.profile_detail:
            self._last_decode_step_profile = self._decode_step_profile_snapshot()

    def _estimate_next_decode_pressure_tokens(self, forward_batch: Any) -> int:
        if self._bytes_per_token_all_layers <= 0:
            return 0
        total_tokens = int(self._allocator_total_size())
        if total_tokens <= 0:
            return 0
        pairs = (
            list(self._current_forward_req_lens)
            if forward_batch is not None
            and self._current_forward_req_lens_batch_id == id(forward_batch)
            else self._batch_req_indices_and_lens(forward_batch)
        )
        live_tokens = sum(max(0, int(seq_len)) for _req_idx, seq_len in pairs)
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if isinstance(out_cache_loc, torch.Tensor):
            write_tokens = int(out_cache_loc.numel())
        elif out_cache_loc is not None:
            try:
                write_tokens = len(out_cache_loc)
            except TypeError:
                write_tokens = 0
        else:
            write_tokens = 0
        next_decode_reserve_tokens = len(pairs)
        return max(
            0,
            int(live_tokens)
            + int(write_tokens)
            + int(next_decode_reserve_tokens)
            - int(total_tokens),
        )

    def prepare_first_decode_kvc_eviction_decision(self, forward_batch: Any) -> bool:
        if int(self._decode_step) != 0:
            return False
        if self._prepared_kvc_evict_entries or self._prepared_kvc_evict_schedule:
            return True
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return False
        if not self.config.dynamic_pressure_from_kvc:
            return False
        if not self.physical_kvc_supported:
            return False
        if (
            self.config.kvc_backend == "per-layer-arena"
            and not self._ensure_virtual_scratch()
        ):
            return False

        old_mode = self._current_forward_mode
        old_scheduler_pressure_tokens = int(self._scheduler_pressure_tokens)
        old_scheduler_budget_pressure_tokens = int(
            getattr(self.stats, "scheduler_budget_pressure_tokens", 0) or 0
        )
        t0 = time.perf_counter()
        warmed_bitmaps = 0
        try:
            self._current_forward_mode = "decode"
            pressure_tokens = self._estimate_next_decode_pressure_tokens(forward_batch)
            if pressure_tokens > 0:
                self._scheduler_pressure_tokens = max(
                    int(self._scheduler_pressure_tokens), int(pressure_tokens)
                )
                self.stats.scheduler_budget_pressure_tokens = int(
                    self._scheduler_pressure_tokens
                )
            need_tokens = self._planned_kvc_evict_need_tokens(forward_batch)
            prepared = False
            if need_tokens > 0:
                prepare_schedule = getattr(
                    self, "_prepare_kvc_evict_selection_schedule", None
                )
                if prepare_schedule is not None:
                    prepared = bool(
                        prepare_schedule(
                            forward_batch,
                            need_tokens,
                            max_steps=16,
                        )
                    )
                if not prepared:
                    prepared = self._prepare_kvc_evict_selection(
                        forward_batch, need_tokens
                    )
            if prepared and self.config.kvc_backend == "per-layer-arena":
                warmed_bitmaps = self._prewarm_per_layer_overwrite_bitmaps()
        finally:
            self._current_forward_mode = old_mode
            self._scheduler_pressure_tokens = old_scheduler_pressure_tokens
            self.stats.scheduler_budget_pressure_tokens = (
                old_scheduler_budget_pressure_tokens
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if prepared:
            logger.info(
                "LayerKV first decode KVC eviction decision prepared: %s",
                json.dumps(
                    {
                        "elapsed_ms": elapsed_ms,
                        "need_tokens": int(need_tokens),
                        "pressure_tokens": int(pressure_tokens),
                        "prepared_tokens": int(
                            sum(
                                int(entry.token_count)
                                for entry in self._prepared_kvc_evict_entries
                            )
                        ),
                        "overwrite_bitmap_layers": int(warmed_bitmaps),
                    },
                    sort_keys=True,
                ),
            )
        elif self.config.debug_stats:
            logger.info(
                "LayerKV first decode KVC eviction decision skipped: %s",
                json.dumps(
                    {
                        "elapsed_ms": elapsed_ms,
                        "need_tokens": int(need_tokens),
                        "pressure_tokens": int(pressure_tokens),
                    },
                    sort_keys=True,
                ),
            )
        return bool(prepared)

    def _prewarm_expert_hotness_buffers(self, forward_batch: Any) -> int:
        if not self._expert_layers and not self._expert_modules:
            return 0
        pairs = (
            list(self._current_forward_req_lens)
            if forward_batch is not None
            and self._current_forward_req_lens_batch_id == id(forward_batch)
            else self._batch_req_indices_and_lens(forward_batch)
        )
        batch_size = max(1, len(pairs))
        max_running = max(
            batch_size,
            int(getattr(self.stats, "native_schedule_max_running_requests", 0) or 0),
            int(getattr(self.stats, "scheduler_running_req_max", 0) or 0),
        )
        candidate_batch_sizes = {batch_size, max_running}
        if max_running > 1:
            candidate_batch_sizes.add(max_running - 1)
        warmed = 0
        modules_by_layer: Dict[int, Any] = {
            int(layer_id): module for layer_id, module in self._expert_modules
        }
        for layer_id, state in self._expert_layers.items():
            modules_by_layer.setdefault(int(layer_id), getattr(state, "module", None))
        for layer_id, module in modules_by_layer.items():
            if module is None:
                continue
            weight = getattr(module, "w13_weight", None)
            data = getattr(weight, "data", None)
            device = getattr(data, "device", None)
            if device is None or getattr(device, "type", "") != "cuda":
                continue
            full_num_experts = 0
            try:
                full_num_experts = int(data.shape[0])
            except Exception:
                full_num_experts = 0
            if full_num_experts <= 0:
                continue
            top_k = int(getattr(module, "top_k", 0) or 0)
            if top_k <= 0:
                moe_runner_config = getattr(module, "moe_runner_config", None)
                top_k = int(getattr(moe_runner_config, "top_k", 1) or 1)
            self._expert_hotness_gpu_counts(
                "decode", int(layer_id), int(full_num_experts), device
            )
            for candidate_batch_size in candidate_batch_sizes:
                self._expert_hotness_ones(
                    device, max(1, int(candidate_batch_size) * max(1, top_k))
                )
            warmed += 1
        return warmed

    def _maybe_prewarm_hotness_once(self, forward_batch: Any, *, phase: str) -> bool:
        if self._layerkv_hotness_prewarm_done:
            return False
        t0 = time.perf_counter()
        warmed = self._prewarm_expert_hotness_buffers(forward_batch)
        if warmed <= 0:
            return False
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._layerkv_hotness_prewarm_done = True
        self.stats.layerkv_prewarm_ms += elapsed_ms
        logger.info(
            "LayerKV hotness prewarm: %s",
            json.dumps(
                {
                    "phase": phase,
                    "elapsed_ms": elapsed_ms,
                    "hotness_layers": warmed,
                },
                sort_keys=True,
            ),
        )
        return True

    def _log_decode_step_profile(self) -> None:
        if not (self.config.debug_stats and self.config.profile_detail):
            return
        fields, counters = self._decode_step_profile_fields()
        current = self._decode_step_profile_snapshot()
        payload: Dict[str, Any] = {
            "step": int(self._decode_step),
            "observed_batch_size": int(self.stats.observed_batch_size),
            "native_schedule_batch_size": int(self.stats.native_schedule_batch_size),
            "native_schedule_running_batch_size": int(
                self.stats.native_schedule_running_batch_size
            ),
            "native_schedule_waiting_queue_len": int(
                self.stats.native_schedule_waiting_queue_len
            ),
            "planned_kvc_tokens_by_layer": self.stats.selected_kvc_tokens_by_layer,
            "actual_kvc_tokens_by_layer": self.stats.actual_kvc_tokens_by_layer,
            "cleanup_drop_entries_modes": (
                self.stats.profile_cleanup_drop_entries_modes
            ),
            "cleanup_drop_entries_req_summary": (
                self.stats.profile_cleanup_drop_entries_req_summary
            ),
        }
        for field in fields + counters:
            value = float(current.get(field, 0.0))
            payload[field] = value - float(
                self._last_decode_step_profile.get(field, 0.0)
            )
        self._last_decode_step_profile = current
        logger.info(
            "LayerKV decode step profile: %s",
            json.dumps(payload, sort_keys=True),
        )

    def _maybe_prewarm_layerkv_once(self, forward_batch: Any, *, phase: str) -> bool:
        if self._layerkv_prewarm_done or self._layerkv_prewarm_running:
            return False
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return False
        if not self.config.dynamic_pressure_from_kvc:
            return False
        if not self.physical_kvc_supported:
            self.stats.layerkv_prewarm_skipped_count += 1
            self.stats.layerkv_prewarm_skipped_reason = "physical_kvc_unsupported"
            self._layerkv_prewarm_done = True
            return False
        if not self._kvc_reclaim_is_scheduler_visible():
            self.stats.layerkv_prewarm_skipped_count += 1
            self.stats.layerkv_prewarm_skipped_reason = (
                "kvc_reclaim_not_scheduler_visible"
            )
            self._layerkv_prewarm_done = True
            return False
        if (
            self.config.kvc_backend == "per-layer-arena"
            and not self._ensure_virtual_scratch()
        ):
            self.stats.layerkv_prewarm_skipped_count += 1
            self.stats.layerkv_prewarm_skipped_reason = "virtual_scratch_unavailable"
            self._layerkv_prewarm_done = True
            return False
        old_scheduler_pressure_tokens = int(self._scheduler_pressure_tokens)
        predicted_pressure_tokens = 0
        pressure_mb = max(0.0, float(self._dynamic_runtime_pressure_mb(forward_batch)))
        if pressure_mb <= 1e-3 and phase == "extend":
            predicted_pressure_tokens = self._estimate_next_decode_pressure_tokens(
                forward_batch
            )
            if predicted_pressure_tokens > 0:
                self._scheduler_pressure_tokens = max(
                    old_scheduler_pressure_tokens, int(predicted_pressure_tokens)
                )
                self.stats.scheduler_budget_pressure_tokens = int(
                    self._scheduler_pressure_tokens
                )
                pressure_mb = max(
                    0.0, float(self._dynamic_runtime_pressure_mb(forward_batch))
                )
        if pressure_mb <= 1e-3:
            self.stats.layerkv_prewarm_skipped_reason = f"{phase}:no_runtime_pressure"
            return False
        if self._scheduler_pressure_kvc_blocked:
            self.stats.layerkv_prewarm_skipped_count += 1
            self.stats.layerkv_prewarm_skipped_reason = "scheduler_pressure_kvc_blocked"
            self._layerkv_prewarm_done = True
            return False

        before_tokens = int(self._offloaded_token_count())
        before_evict_ms = float(
            getattr(self.stats, "profile_kvc_evict_to_target_ms", 0.0)
        )
        t0 = time.perf_counter()
        self._layerkv_prewarm_running = True
        try:
            with self._profile("profile_kvc_evict_to_target_ms"):
                self._evict_kvc_to_target(forward_batch)
        finally:
            self._layerkv_prewarm_running = False
            if predicted_pressure_tokens > 0:
                self._scheduler_pressure_tokens = old_scheduler_pressure_tokens
                self.stats.scheduler_budget_pressure_tokens = int(
                    old_scheduler_pressure_tokens
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        after_tokens = int(self._offloaded_token_count())
        evicted_tokens = max(0, after_tokens - before_tokens)
        evict_ms = max(
            0.0,
            float(getattr(self.stats, "profile_kvc_evict_to_target_ms", 0.0))
            - before_evict_ms,
        )
        self._layerkv_prewarm_done = True
        self.stats.layerkv_prewarm_count += 1
        self.stats.layerkv_prewarm_ms += elapsed_ms
        self.stats.layerkv_prewarm_kvc_evict_ms += evict_ms
        self.stats.layerkv_prewarm_kvc_tokens += evicted_tokens
        self.stats.layerkv_prewarm_pressure_mb = pressure_mb
        if evicted_tokens <= 0:
            self.stats.layerkv_prewarm_skipped_count += 1
            self.stats.layerkv_prewarm_skipped_reason = "no_evict_candidate"
        else:
            self.stats.layerkv_prewarm_skipped_reason = ""
        self._refresh_physical_reclaim_peaks(record_step_sample=False)
        self._refresh_resident_group_stats()
        logger.info(
            "LayerKV prewarm: %s",
            json.dumps(
                {
                    "phase": phase,
                    "pressure_mb": pressure_mb,
                    "elapsed_ms": elapsed_ms,
                    "kvc_evict_ms": evict_ms,
                    "evicted_tokens": evicted_tokens,
                },
                sort_keys=True,
            ),
        )
        return evicted_tokens > 0

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
        if validate and self._pending_kvc_evict_events:
            self._finalize_kvc_evictions(block=True)
        if validate:
            self._cleanup_released_per_layer_entries()
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
