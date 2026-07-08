"""LayerKV LayerKVSchedulerMixin implementation."""

from __future__ import annotations

import bisect
import contextlib
import json
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

if __package__:
    from .common_types import (
        _LayerKVKvcDemand,
        _LayerKVMetadataPatchCacheEntry,
        _LayerKVNativeKvcRun,
        _LayerKVPendingEviction,
        _LayerKVPendingReload,
        _LayerKVPendingVirtualMaterialize,
        _LayerKVRecoveryTask,
        _LayerKVResidencyEntry,
        _LayerKVVirtualMaterializePlan,
        _LayerKVVirtualScratchCacheEntry,
    )
else:  # pragma: no cover - direct file-loading smoke tests.
    from common_types import (
        _LayerKVKvcDemand,
        _LayerKVMetadataPatchCacheEntry,
        _LayerKVNativeKvcRun,
        _LayerKVPendingEviction,
        _LayerKVPendingReload,
        _LayerKVPendingVirtualMaterialize,
        _LayerKVRecoveryTask,
        _LayerKVResidencyEntry,
        _LayerKVVirtualMaterializePlan,
        _LayerKVVirtualScratchCacheEntry,
    )

logger = logging.getLogger(__name__)


class LayerKVSchedulerMixin:
    def on_forward_begin(self, *, mode: str, forward_batch: Any) -> None:
        t0 = time.perf_counter()
        phase_t0 = time.perf_counter()
        self._current_forward_mode = mode
        self._last_forward_batch = forward_batch
        if mode == "decode" and int(self._decode_step) == 0:
            self._reset_decode_step_profile_baseline()
        if mode == "decode":
            self._kvc_evict_finalized_this_step_tokens = 0
        self._per_layer_kvc_prepared_layers.clear()
        self._virtual_materialize_plan = None
        self._virtual_materialize_plans_by_layer.clear()
        self._per_layer_kvc_evict_finalized_before_layers.clear()
        self._virtual_batch_index_step = -1
        self._virtual_batch_index_cache = None
        self._virtual_kvc_demands.clear()
        self._virtual_batched_prefetch_step = -1
        self._kvc_demand_signature_eval_step = None
        self._per_layer_kvc_current_metadata_key = None
        self._refresh_per_layer_kvc_overrides()
        self._add_profile(
            "profile_forward_begin_bookkeeping_ms",
            (time.perf_counter() - phase_t0) * 1000.0,
        )
        with self._profile("profile_workload_stats_ms"):
            self._refresh_workload_stats(forward_batch)
        phase_t0 = time.perf_counter()
        self._mark_canonical_per_layer_metadata_current()
        self._add_profile(
            "profile_forward_begin_metadata_current_ms",
            (time.perf_counter() - phase_t0) * 1000.0,
        )
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
            phase_t0 = time.perf_counter()
            fast_path = self._no_pressure_fast_path_active(forward_batch)
            self._add_profile(
                "profile_forward_begin_fastpath_check_ms",
                (time.perf_counter() - phase_t0) * 1000.0,
            )
            phase_t0 = time.perf_counter()
            if fast_path:
                self.stats.layerkv_no_pressure_expert_skip_count += 1
            else:
                with self._profile("profile_apply_expert_plan_ms"):
                    self._apply_expert_plan_once(forward_batch)
                self._advance_expert_install_budgeted()
                self._force_drain_expert_install_d2h_and_slots()
                if self._expert_plan_applied and not self._expert_install_queue:
                    self._check_current_coresid_plan_match(context="post_install")
            self._add_profile(
                "profile_forward_begin_expert_control_ms",
                (time.perf_counter() - phase_t0) * 1000.0,
            )
            self.stats.forward_decode_count += 1
            self._decode_step += 1
            self.stats.decode_steps = self._decode_step
            phase_t0 = time.perf_counter()
            self._prepare_expert_hotness_sampling_for_step()
            self._add_profile(
                "profile_forward_begin_hotness_prepare_ms",
                (time.perf_counter() - phase_t0) * 1000.0,
            )
            phase_t0 = time.perf_counter()
            if fast_path:
                self.stats.layerkv_no_pressure_scheduler_skip_count += 1
            else:
                self._run_deadline_scheduler(forward_batch)
            self._add_profile(
                "profile_forward_begin_scheduler_control_ms",
                (time.perf_counter() - phase_t0) * 1000.0,
            )
        else:
            self.stats.forward_extend_count += 1
            phase_t0 = time.perf_counter()
            self._drop_entries_for_reqs(forward_batch)
            self._add_profile(
                "profile_forward_begin_extend_cleanup_ms",
                (time.perf_counter() - phase_t0) * 1000.0,
            )
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
        self.stats.pre_retract_reclaim_allocator_available_after = int(
            visible_available
        )
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

    def reclaim_decode_slots_before_retract(
        self,
        *,
        schedule_batch: Any,
        required_tokens: int,
        available_tokens: int,
    ) -> bool:
        """Make enough native decode slots visible before request retraction."""
        required_tokens = max(0, int(required_tokens))
        available_tokens = max(0, int(available_tokens))
        if required_tokens <= 0 or available_tokens >= required_tokens:
            return True
        if self.config.kvc_backend != "per-layer-arena":
            ok = self.try_reclaim_kvc_before_retract(
                schedule_batch=schedule_batch,
                required_tokens=required_tokens,
                available_tokens=available_tokens,
                reason="pre_retract_decode_mem",
            )
            self._finalize_kvc_evictions(block=True)
            return bool(ok) or self._allocator_available_size() >= required_tokens

        deficit = max(0, required_tokens - self._allocator_available_size())
        if deficit <= 0:
            return True
        release_fn = getattr(self, "_release_common_per_layer_locs_to_native", None)
        if release_fn is not None:
            release_fn(deficit)
        if self._allocator_available_size() >= required_tokens:
            return True

        self.try_reclaim_kvc_before_retract(
            schedule_batch=schedule_batch,
            required_tokens=required_tokens,
            available_tokens=self._allocator_available_size(),
            reason="decode_prealloc_admission",
        )
        self._finalize_kvc_evictions(block=True)
        deficit = max(0, required_tokens - self._allocator_available_size())
        if deficit > 0 and release_fn is not None:
            release_fn(deficit)
        return self._allocator_available_size() >= required_tokens

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
            total_t0 = time.perf_counter() if self.config.profile_detail else 0.0
            cleaned = bool(getattr(req, "layerkv_per_layer_cleaned", False))
            indexed_cleanup = req_idx in self._per_layer_cleanup_layers_by_req
            keys = []
            if not cleaned:
                if indexed_cleanup:
                    keys = []
                else:
                    keys = list(self._per_layer_owned_keys_by_req.pop(req_idx, set()))
            else:
                self._per_layer_owned_keys_by_req.pop(req_idx, None)
            if not keys and not cleaned and not indexed_cleanup:
                keys = [key for key in self._per_layer_residency if key[1] == req_idx]
            token_count = self._drop_per_layer_residency_for_finished_req(req_idx, keys)
            self._per_layer_owned_req_indices.discard(req_idx)
            native_free_end = 0
            if not getattr(req, "kv_committed_freed", False):
                native_free_end = max(
                    native_free_end, int(req.pop_committed_kv_cache())
                )
            if not getattr(req, "kv_overallocated_freed", False):
                _start_p, end_p = req.pop_overallocated_kv_cache()
                native_free_end = max(native_free_end, int(end_p))
            if (
                native_free_end > 0
                and self._req_to_token_pool is not None
                and self._allocator is not None
            ):
                free_start = 0
                if not bool(getattr(tree_cache, "disable", False)):
                    free_start = max(0, int(getattr(req, "cache_protected_len", 0)))
                native_locs = self._req_to_token_pool.req_to_token[
                    req_idx, free_start:native_free_end
                ]
                if int(native_locs.numel()) > 0:
                    native_locs_cpu = native_locs.detach().cpu().to(torch.int64)
                    native_loc_list = [
                        int(loc)
                        for loc in native_locs_cpu.tolist()
                        if int(loc) > 0
                        and int(loc) not in self._per_layer_arena_reserved_locs
                    ]
                    if native_loc_list:
                        filtered_locs = torch.unique(
                            torch.tensor(
                                native_loc_list,
                                dtype=torch.int64,
                                device=self._allocator.device,
                            )
                        )
                        if int(filtered_locs.numel()) > 0:
                            self._allocator.free(filtered_locs)
            if getattr(req, "last_node", None) is not None:
                try:
                    tree_cache.dec_lock_ref(req.last_node)
                except Exception:
                    pass
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
            if self.config.profile_detail:
                self.stats.profile_cleanup_release_call_count += 1
                self.stats.profile_cleanup_release_key_count += int(len(keys))
                self.stats.profile_cleanup_release_token_count += int(token_count)
                self._add_profile(
                    "profile_cleanup_release_ms",
                    (time.perf_counter() - total_t0) * 1000.0,
                )
            return True
        if self.config.kvc_backend != "virtual-arena":
            return False
        req_pool_idx = getattr(req, "req_pool_idx", None)
        if req_pool_idx is None:
            return False
        req_idx = int(req_pool_idx)
        total_t0 = time.perf_counter() if self.config.profile_detail else 0.0
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
        if self.config.profile_detail:
            self.stats.profile_cleanup_release_call_count += 1
            self.stats.profile_cleanup_release_key_count += int(len(keys))
            self.stats.profile_cleanup_release_token_count += int(
                len(virtual_positions)
            )
            self._add_profile(
                "profile_cleanup_release_ms",
                (time.perf_counter() - total_t0) * 1000.0,
            )
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
        native_indices = req_to_token_pool.req_to_token[req_idx, :token_count].to(
            device=device, dtype=torch.long
        )
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
            max_layer_tasks = max(1, int(max_tasks)) if max_tasks is not None else None
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
                    self.stats.kvc_task_coalesced_count += max(0, len(by_layer) - 1)
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
        self.stats.kvc_avg_task_bytes = self.stats.kvc_task_byte_sum / float(task_count)
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
