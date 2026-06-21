"""LayerKV LayerKVKvcReclaimMixin implementation."""

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


class LayerKVKvcReclaimMixin:
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
                        self._remove_per_layer_offloaded_run_range(
                            int(entry.layer_id),
                            int(entry.req_idx),
                            int(entry.pos),
                            int(entry.token_count),
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
                        self._remove_per_layer_offloaded_run_range(
                            int(entry.layer_id),
                            int(entry.req_idx),
                            int(entry.pos),
                            int(entry.token_count),
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
        with self._profile("profile_kvc_host_alloc_ms"):
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
                with self._profile("profile_kvc_backup_wall_ms"):
                    elapsed_ms = self._host_store.backup_per_layer(
                        selected, host_slots
                    )
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
                with self._profile("profile_kvc_backup_wall_ms"):
                    elapsed_ms = self._host_store.backup(old_locs, host_slots)
                self.stats.kvc_allocator_available_before = (
                    self._allocator_available_size()
                )
                self._allocator.free(old_locs)
                self.stats.kvc_allocator_available_after = (
                    self._allocator_available_size()
                )
                self.stats.kvc_allocator_free_count += 1
            with self._profile("profile_kvc_evict_commit_ms"):
                offset = 0
                per_layer_locs_to_free: Dict[int, List[int]] = {}
                per_layer_offloaded_to_track: List[_LayerKVResidencyEntry] = []
                for entry in selected:
                    page_host_slots = host_slots[offset : offset + entry.token_count]
                    offset += entry.token_count
                    entry.state = "offloaded"
                    if self.config.kvc_backend == "virtual-arena":
                        self.stats.virtual_kvc_evict_count += int(entry.token_count)
                    if self.config.kvc_backend == "per-layer-arena":
                        per_layer_offloaded_to_track.append(entry)
                        self._per_layer_resident_token_count_fast = max(
                            0,
                            self._per_layer_resident_token_count_fast
                            - int(entry.token_count),
                        )
                        self._per_layer_offloaded_token_count_fast += int(
                            entry.token_count
                        )
                        old_locs_for_free = entry.device_loc_list()
                        entry.evicted_device_locs = old_locs_for_free or None
                        if old_locs_for_free:
                            per_layer_locs_to_free.setdefault(
                                int(entry.layer_id), []
                            ).extend(int(loc) for loc in old_locs_for_free)
                        entry.device_loc = None
                        entry.device_locs = None
                    entry.host_slots = [int(x) for x in page_host_slots]
                    entry.host_slot = (
                        int(page_host_slots[0]) if page_host_slots else None
                    )
                    if self.config.kvc_backend != "per-layer-arena":
                        entry.device_loc = None
                        entry.device_locs = None
                    entry.ready_event = None
                    entry.ready_start_event = None
                    entry.ready_waited = False
                    entry.last_access_step = self._decode_step
                    self._sync_kvc_group_if_needed(entry)
                if self.config.kvc_backend == "per-layer-arena":
                    self._track_per_layer_offloaded_keys_batch(
                        per_layer_offloaded_to_track
                    )
                    self._track_per_layer_offloaded_runs(selected)
                if per_layer_locs_to_free:
                    if not self._per_layer_virtual_scratch_enabled():
                        for layer_id, locs in per_layer_locs_to_free.items():
                            self._protect_per_layer_terminal_locs(layer_id, locs)
                    with self._profile("profile_kvc_free_locs_ms"):
                        self._free_per_layer_locs_batch(
                            per_layer_locs_to_free,
                            refresh=False,
                            assume_allocated=True,
                        )
                    self._refresh_per_layer_allocator_stats()
                if self.config.kvc_backend == "virtual-arena":
                    for req_idx in {int(entry.req_idx) for entry in selected}:
                        self._mark_req_virtualized(req_idx)
                if (
                    self.config.kvc_backend == "per-layer-arena"
                    and self._per_layer_virtual_scratch_enabled()
                ):
                    self._invalidate_virtual_caches_for_layers(
                        self._virtual_cache_layers_for_entries(selected)
                    )
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
            with self._profile("profile_kvc_cost_observe_ms"):
                self._record_kvc_layer_cost_observations(
                    selected, elapsed_ms, kind="evict"
                )
        self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
        self.stats.kvc_physical_cycle_count += 1
        self.stats.layerkv_tasks_built += 1
        with self._profile("profile_kvc_refresh_stats_ms"):
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
            if getattr(self._host_store, "track_layer_used_sets", True):
                host_used_slots = {
                    (self._host_store.start_layer + layer_offset, int(slot))
                    for layer_offset, slots in enumerate(
                        self._host_store.layer_used_slots
                    )
                    for slot in slots
                }
            else:
                host_used_slots = None
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
                        self._host_store is not None
                        and host_used_slots is not None
                        and host_key not in host_used_slots
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
        if self._current_forward_mode == "decode" and self._last_scheduled_req_lens:
            return list(self._last_scheduled_req_lens)
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
        native_runs = list(self._native_kvc_runs_by_req.get(req_idx, []))
        if not to_drop and not per_layer_to_drop and not native_runs:
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
        token_count += sum(max(0, int(run.end) - int(run.start)) for run in native_runs)
        self._drop_residency_keys(to_drop)
        self._drop_per_layer_residency_keys(per_layer_to_drop)
        self._drop_native_kvc_runs_for_req(req_idx)
        self._drop_offloaded_kvc_runs_for_req(req_idx)
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
        self._prune_native_kvc_runs_for_active_lengths(active_lens)
        self._prune_offloaded_kvc_runs_for_active_lengths(active_lens)
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
                    self._remove_per_layer_offloaded_run_range(
                        int(entry.layer_id),
                        int(entry.req_idx),
                        int(entry.pos),
                        int(entry.token_count),
                    )
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
                for idx in range(limit):
                    key = sorted_keys[idx]
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
                    layer_entries = self._materialize_per_layer_resident_segment(
                        int(layer_id),
                        req_idx,
                        int(pos),
                        int(run_len),
                        import_arena=False,
                    )
                    self.stats.kvc_evict_selector_scanned_entries += int(run_len)
                    if layer_entries is None or sum(
                        int(entry.token_count) for entry in layer_entries
                    ) != int(run_len):
                        run_entries = []
                        break
                    layer_locs = [
                        int(loc)
                        for entry in layer_entries
                        for loc in entry.device_loc_list()
                    ]
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

        def merge_segments(
            segments: List[Tuple[int, int, int]]
        ) -> List[Tuple[int, int, int]]:
            if len(segments) <= 1:
                return segments
            merged: List[Tuple[int, int, int]] = []
            for req_idx, pos, count in sorted(
                segments, key=lambda item: (int(item[0]), int(item[1]))
            ):
                req_idx = int(req_idx)
                pos = int(pos)
                count = int(count)
                if count <= 0:
                    continue
                if (
                    merged
                    and int(merged[-1][0]) == req_idx
                    and int(merged[-1][1]) + int(merged[-1][2]) == pos
                ):
                    prev_req, prev_pos, prev_count = merged[-1]
                    merged[-1] = (prev_req, prev_pos, int(prev_count) + count)
                else:
                    merged.append((req_idx, pos, count))
            return merged

        def next_resident_span(
            layer_id: int, req_idx: int, pos: int, end: int
        ) -> Tuple[int, int]:
            pos = int(pos)
            end = int(end)
            runs = self._per_layer_offloaded_runs_by_req_layer.get(
                (int(req_idx), int(layer_id))
            )
            if not runs:
                return pos, end
            for run in runs:
                run_start = int(run.pos)
                run_end = run_start + int(run.token_count)
                if run_end <= pos:
                    continue
                if run_start <= pos:
                    pos = self._align_tokens_up(run_end)
                    if pos >= end:
                        return end, end
                    continue
                return pos, min(end, self._align_tokens_down(run_start))
            return pos, end

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
                    raw_pos = cursor + offset
                    if raw_pos + block_size > evictable_len:
                        continue
                    pos, span_end = next_resident_span(
                        layer_id, req_idx, raw_pos, evictable_len
                    )
                    if pos + block_size > span_end:
                        round_offsets[req_idx] = max(
                            offset + block_size, int(span_end) - int(cursor)
                        )
                        continue
                    span_tokens = self._align_tokens_down(
                        min(int(remaining), int(span_end) - int(pos))
                    )
                    if span_tokens < block_size:
                        round_offsets[req_idx] = max(
                            offset + block_size, int(span_end) - int(cursor)
                        )
                        continue
                    key = (layer_id, req_idx, pos)
                    existing = self._per_layer_residency.get(key)
                    self.stats.kvc_evict_selector_scanned_entries += 1
                    if existing is not None and existing.state in (
                        "offloaded",
                        "reloading",
                        "evicting",
                    ):
                        round_offsets[req_idx] = max(
                            offset + block_size, int(pos) + block_size - int(cursor)
                        )
                        continue
                    layer_segments.append((req_idx, pos, span_tokens))
                    remaining -= span_tokens
                    round_offsets[req_idx] = int(pos) + int(span_tokens) - int(cursor)
                    progressed = True
                    self.stats.kvc_layerwise_evict_cursor_hit_count += 1
                    if remaining < block_size:
                        break
                if not progressed:
                    break
            if not layer_segments:
                continue
            for req_idx, pos, count in merge_segments(layer_segments):
                segment_entries = self._materialize_per_layer_resident_segment(
                    layer_id,
                    int(req_idx),
                    int(pos),
                    int(count),
                    import_arena=False,
                )
                self.stats.kvc_evict_selector_scanned_entries += int(count)
                if segment_entries is None or sum(
                    int(entry.token_count) for entry in segment_entries
                ) != int(count):
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
