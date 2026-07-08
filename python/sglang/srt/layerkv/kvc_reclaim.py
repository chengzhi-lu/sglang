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
            if self._current_forward_mode == "decode" and self._last_scheduled_req_lens
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
                    if self.config.kvc_backend == "per-layer-arena":
                        self._set_per_layer_page_entry_state(entry, "reloading")
                    else:
                        entry.state = "reloading"
                    if self.config.kvc_backend == "per-layer-arena":
                        host_slots = entry.host_slot_list()
                        if host_slots:
                            self._remove_per_layer_cleanup_host_slots(entry, host_slots)
                        reload_locs = entry.device_loc_list() or (
                            [int(loc) for loc in entry.evicted_device_locs]
                            if entry.evicted_device_locs
                            else []
                        )
                        if reload_locs:
                            self._add_per_layer_cleanup_locs(entry, reload_locs)
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
                    if self.config.kvc_backend == "per-layer-arena":
                        self._set_per_layer_page_entry_state(entry, "resident")
                    else:
                        entry.state = "resident"
                    if self.config.kvc_backend == "per-layer-arena":
                        host_slots = entry.host_slot_list()
                        if host_slots:
                            self._remove_per_layer_cleanup_host_slots(entry, host_slots)
                        reload_locs = entry.device_loc_list() or (
                            [int(loc) for loc in entry.evicted_device_locs]
                            if entry.evicted_device_locs
                            else []
                        )
                        if reload_locs:
                            self._add_per_layer_cleanup_locs(entry, reload_locs)
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
                    if self.config.kvc_backend != "per-layer-arena":
                        if entry.host_slots is not None:
                            self._host_store.free(entry.host_slots)
                        elif entry.host_slot is not None:
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
                    self.stats.target_limited_reason = "no_non_deadline_kvc_candidate"
                    return
            if effective_kvc_reclaim_mb <= 0:
                return
            if (
                self.config.kvc_backend == "per-layer-arena"
                and not self.config.dynamic_pressure_from_kvc
                and self._uses_layer_aware_kvc_plan()
                and int(self._planned_kvc_token_target) > 0
                and int(self._per_layer_offloaded_token_count_fast)
                + int(self._pending_evict_token_count())
                >= int(self._planned_kvc_token_target)
            ):
                self.stats.kvc_eviction_skipped_count += 1
                return
        else:
            force_additional_tokens = self._align_tokens_up(
                max(0, int(force_additional_tokens))
            )
            if force_additional_tokens <= 0:
                return
            if self.config.dynamic_pressure_from_kvc:
                runtime_pressure_mb = self._dynamic_runtime_pressure_mb(forward_batch)
                if runtime_pressure_mb <= 1e-3 and force_reason not in (
                    "pre_retract_decode_mem",
                    "decode_prealloc_admission",
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

        with self._profile("profile_kvc_evict_target_calc_ms"):
            if force_additional_tokens is not None:
                target_tokens = self._offloaded_token_count() + int(
                    force_additional_tokens
                )
            elif (
                self._uses_layer_aware_kvc_plan() and self._planned_kvc_token_target > 0
            ):
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
                if self.config.kvc_backend in ("virtual-arena", "per-layer-arena")
                else 0
            )
            if (
                self.config.dynamic_pressure_from_kvc
                and force_additional_tokens is None
            ):
                finalized_this_step_tokens = int(
                    getattr(self, "_kvc_evict_finalized_this_step_tokens", 0) or 0
                )
                if pending_evict_tokens > 0 or finalized_this_step_tokens > 0:
                    self.stats.kvc_topup_skip_count += 1
                    self.stats.kvc_eviction_skipped_count += 1
                    return
                offloaded_or_pending = (
                    self._offloaded_token_count() + pending_evict_tokens
                )
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
            if (
                self.config.dynamic_pressure_from_kvc
                and force_additional_tokens is None
            ):
                need_tokens = min(
                    int(need_tokens),
                    self._dynamic_pressure_topup_step_tokens(margin_tokens),
                )
                need_tokens = self._align_tokens_down(int(need_tokens))
            if need_tokens <= 0:
                self.stats.kvc_eviction_skipped_count += 1
                return

        self._ensure_host_store()
        with self._profile("profile_kvc_evict_prune_ms"):
            self._prune_entries_for_active_lengths(forward_batch)
        with self._profile("profile_kvc_evict_backend_check_ms"):
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
                self.stats.layerkv_kvc_backend_ready = (
                    self._per_layer_kvc_backend_ready()
                )

        selected = self._consume_prepared_kvc_evict_selection(
            forward_batch, need_tokens
        )
        if selected is None:
            with self._profile("profile_kvc_select_evict_ms"):
                selected = self._select_resident_entries_for_eviction(
                    forward_batch, need_tokens
                )
        if not selected:
            self.stats.kvc_eviction_skipped_count += 1
            return

        self._issue_kvc_eviction_entries(
            selected,
            force_additional_tokens=force_additional_tokens,
        )

    def _issue_kvc_eviction_entries(
        self,
        selected: List[_LayerKVResidencyEntry],
        *,
        force_additional_tokens: Optional[int] = None,
    ) -> bool:
        if not selected:
            return False
        with self._profile("profile_kvc_evict_selected_stats_ms"):
            token_count = sum(entry.token_count for entry in selected)
            if self.config.kvc_backend == "per-layer-arena":
                by_layer: Dict[int, int] = {}
                for entry in selected:
                    by_layer[int(entry.layer_id)] = (
                        by_layer.get(int(entry.layer_id), 0) + entry.token_count
                    )
                if by_layer:
                    self.stats.actual_kvc_tokens_by_layer = (
                        self._kvc_tokens_by_layer_json(by_layer)
                    )
                    if self._requires_common_per_layer_kvc_eviction():
                        self.stats.selected_kvc_tokens_by_layer = (
                            self._kvc_tokens_by_layer_json(by_layer)
                        )
                    else:
                        merged = dict(self._planned_kvc_tokens_by_layer)
                        for layer_id, tokens in by_layer.items():
                            merged[layer_id] = max(
                                int(merged.get(layer_id, 0)), int(tokens)
                            )
                        self.stats.selected_kvc_tokens_by_layer = (
                            self._kvc_tokens_by_layer_json(merged)
                        )
            else:
                self.stats.selected_kvc_tokens_by_layer = (
                    self._kvc_tokens_by_layer_json(token_count)
                )
        with self._profile("profile_kvc_evict_prepare_entries_ms"):
            per_layer_backup_entries: List[_LayerKVResidencyEntry] = []
            per_layer_host_slots_by_entry: Dict[int, List[int]] = {}
            if self.config.kvc_backend == "per-layer-arena":
                for entry in selected:
                    existing_slots = [int(slot) for slot in entry.host_slot_list()]
                    if len(existing_slots) == int(entry.token_count):
                        per_layer_host_slots_by_entry[id(entry)] = existing_slots
                    else:
                        per_layer_backup_entries.append(entry)

        with self._profile("profile_kvc_host_alloc_ms"):
            host_slots = None
            used_preallocated_host_slots = False
            if self.config.kvc_backend == "per-layer-arena":
                preallocated = self._consume_prepared_kvc_evict_host_slots(
                    per_layer_backup_entries
                )
                if preallocated is not None:
                    host_slots, preallocated_by_entry = preallocated
                    per_layer_host_slots_by_entry.update(preallocated_by_entry)
                    used_preallocated_host_slots = True
                else:
                    host_slots = self._host_store.alloc_per_layer(
                        per_layer_backup_entries
                    )
            else:
                host_slots = self._host_store.alloc(token_count)
        if host_slots is None:
            self.stats.kvc_physical_failure_count += 1
            self.stats.comparable = False
            self.stats.comparability_reason = "LayerKV host KVC store is full"
            if self.config.disallow_destructive_fallback:
                raise RuntimeError("LayerKV host KVC store is full")
            return False
        if self.config.kvc_backend == "per-layer-arena":
            if not used_preallocated_host_slots:
                offset = 0
                for entry in per_layer_backup_entries:
                    page_slots = host_slots[offset : offset + int(entry.token_count)]
                    offset += int(entry.token_count)
                    per_layer_host_slots_by_entry[id(entry)] = [
                        int(slot) for slot in page_slots
                    ]

        old_locs = None
        if self.config.kvc_backend != "per-layer-arena":
            old_locs = torch.tensor(
                [loc for entry in selected for loc in entry.device_loc_list()],
                dtype=torch.int64,
                device=self._kv_pool.device,
            )
        async_evict = (
            force_additional_tokens is None
            and self.config.kvc_scheduler == "async-deadline"
            and self._optimized_profile_enabled()
            and self._copy_stream is not None
            and self.config.kvc_backend in ("virtual-arena", "per-layer-arena")
        )
        try:
            if self.config.kvc_backend == "per-layer-arena":
                if async_evict and per_layer_backup_entries:
                    with self._profile("profile_kvc_evict_staging_alloc_ms"):
                        start_event, ready_event = (
                            self._host_store.backup_per_layer_async(
                                per_layer_backup_entries,
                                host_slots,
                                stream=self._copy_stream,
                            )
                        )
                    if start_event is None or ready_event is None:
                        raise RuntimeError(
                            "failed to issue async per-layer KVC eviction"
                        )
                    with self._profile("profile_kvc_evict_async_mark_ms"):
                        sync_groups = not (
                            self.config.kvc_backend == "per-layer-arena"
                            and self.config.runtime_profile == "optimized"
                        )
                        for entry in selected:
                            page_host_slots = per_layer_host_slots_by_entry.get(
                                id(entry), []
                            )
                            self._set_per_layer_page_entry_state(entry, "evicting")
                            entry.host_slots = list(page_host_slots)
                            entry.host_slot = (
                                int(page_host_slots[0]) if page_host_slots else None
                            )
                            entry.ready_start_event = start_event
                            entry.ready_event = ready_event
                            entry.ready_waited = False
                            entry.last_access_step = self._decode_step
                            if sync_groups:
                                self._sync_kvc_group_if_needed(entry)
                        self._pending_kvc_evict_events.append(
                            _LayerKVPendingEviction(
                                start_event=start_event,
                                ready_event=ready_event,
                                entries=list(selected),
                                device_locs=torch.empty(
                                    0,
                                    dtype=torch.int64,
                                    device=self._allocator.device,
                                ),
                                host_slots=list(host_slots),
                                k_staging=[],
                                v_staging=[],
                                token_count=int(token_count),
                            )
                        )
                    self.stats.kvc_evict_async_count += int(token_count)
                    self.stats.layerkv_tasks_built += 1
                    with self._profile("profile_kvc_refresh_stats_ms"):
                        self._refresh_kvc_residency_stats()
                    return True
                else:
                    with self._profile("profile_kvc_backup_wall_ms"):
                        elapsed_ms = self._host_store.backup_per_layer(
                            per_layer_backup_entries, host_slots
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
                return True
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
                per_layer_overwrite_bits: Dict[int, int] = {}
                cleanup_remove_bits_by_req_layer: Dict[Tuple[int, int], int] = {}
                per_layer_offloaded_to_track: List[_LayerKVResidencyEntry] = []
                for entry in selected:
                    if self.config.kvc_backend == "per-layer-arena":
                        page_host_slots = per_layer_host_slots_by_entry.get(
                            id(entry), []
                        )
                    else:
                        page_host_slots = host_slots[
                            offset : offset + entry.token_count
                        ]
                        offset += entry.token_count
                    if self.config.kvc_backend == "virtual-arena":
                        entry.state = "offloaded"
                        self.stats.virtual_kvc_evict_count += int(entry.token_count)
                    elif self.config.kvc_backend != "per-layer-arena":
                        entry.state = "offloaded"
                    if self.config.kvc_backend == "per-layer-arena":
                        self._set_per_layer_page_entry_state(entry, "offloaded")
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
                            layer_id = int(entry.layer_id)
                            bits = self._locs_to_bitset(old_locs_for_free, min_value=1)
                            if bits:
                                req_layer = (int(entry.req_idx), layer_id)
                                cleanup_remove_bits_by_req_layer[req_layer] = int(
                                    cleanup_remove_bits_by_req_layer.get(req_layer, 0)
                                ) | int(bits)
                                per_layer_overwrite_bits[layer_id] = int(
                                    per_layer_overwrite_bits.get(layer_id, 0)
                                ) | int(bits)
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
                        if page_host_slots:
                            self._add_per_layer_cleanup_host_slots(
                                entry, [int(slot) for slot in page_host_slots]
                            )
                    entry.last_access_step = self._decode_step
                    self._sync_kvc_group_if_needed(entry)
                if self.config.kvc_backend == "per-layer-arena":
                    self._track_per_layer_offloaded_keys_batch(
                        per_layer_offloaded_to_track
                    )
                    self._track_per_layer_offloaded_runs(selected)
                    for (
                        req_idx,
                        layer_id,
                    ), bits in cleanup_remove_bits_by_req_layer.items():
                        self._remove_per_layer_cleanup_loc_bits(
                            int(req_idx), int(layer_id), int(bits)
                        )
                if per_layer_overwrite_bits:
                    with self._profile("profile_kvc_free_locs_ms"):
                        for layer_id, bits in per_layer_overwrite_bits.items():
                            self._push_per_layer_overwrite_bits(
                                int(layer_id),
                                int(bits),
                                token_count=int(bits).bit_count(),
                            )
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
                if used_preallocated_host_slots:
                    self._free_per_layer_host_slots_for_entries(
                        per_layer_backup_entries,
                        per_layer_host_slots_by_entry,
                    )
                else:
                    offset = 0
                    for entry in per_layer_backup_entries:
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
        return True

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

    def _dynamic_pressure_topup_step_tokens(self, margin_tokens: int) -> int:
        block_tokens = (
            self._per_layer_kvc_block_page_size()
            if self.config.kvc_backend == "per-layer-arena"
            else max(
                self._page_size,
                self._align_tokens_up(int(self.config.kvc_block_tokens or 0)),
            )
        )
        # Spread dynamic KVC pressure relief across decode steps instead of
        # filling the whole reclaim target in one large controller pass.
        step_tokens = max(int(margin_tokens), int(block_tokens) * 512)
        return max(int(block_tokens), self._align_tokens_up(step_tokens))

    def _offloaded_token_count(self) -> int:
        if self.config.kvc_backend == "per-layer-arena":
            self._restore_impossible_per_layer_offloads()
            if self._coresid_optimized_policy_enabled():
                return int(max(0, self._per_layer_offloaded_token_count_fast))
            return sum(
                entry.token_count
                for entry in self._per_layer_residency.values()
                if self._per_layer_entry_is_current(entry)
                and entry.state == "offloaded"
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
                self._per_layer_entry_is_current(entry)
                and entry.state == "offloaded"
                and bool(entry.host_slot_list())
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
                if self._per_layer_entry_is_current(entry) and entry.state == "resident"
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
        if self.config.kvc_backend == "per-layer-arena":
            current_entry_count = self._per_layer_page_table_entry_count()
        else:
            current_entry_count = len(entries)
        self.stats.kvc_residency_entry_count = int(current_entry_count)
        self.stats.kvc_per_layer_arena_entry_count = int(current_entry_count)
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
                if self._per_layer_entry_is_current(entry) and entry.state == "resident"
            )
            self.stats.kvc_per_layer_arena_offloaded_token_count = sum(
                entry.token_count
                for entry in self._per_layer_residency.values()
                if self._per_layer_entry_is_current(entry)
                and entry.state == "offloaded"
            )
            self.stats.kvc_offloaded_page_count = sum(
                1
                for entry in entries.values()
                if (
                    self.config.kvc_backend != "per-layer-arena"
                    or self._per_layer_entry_is_current(entry)
                )
                and entry.state == "offloaded"
            )
            self.stats.kvc_resident_page_count = sum(
                1
                for entry in entries.values()
                if (
                    self.config.kvc_backend != "per-layer-arena"
                    or self._per_layer_entry_is_current(entry)
                )
                and entry.state == "resident"
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
            if (
                self.config.kvc_backend == "per-layer-arena"
                and not self._per_layer_entry_is_current(entry)
            ):
                continue
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
                                terminal_metadata_mapped_count += int(entry.token_count)
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
                                zero_reconstruct_violations += int(entry.token_count)
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
        self.stats.kvc_zero_reconstruct_violation_count = zero_reconstruct_violations
        self.stats.kvc_zero_reconstruct_guard_pass = zero_reconstruct_violations == 0
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
                if (
                    self._expert_global_cpu_backing
                    and self._global_expert_backing(layer_id, logical_id) is not None
                ):
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
        self.stats.expert_zero_reconstruct_guard_reason = ";".join(sorted(set(reasons)))
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
        if seq_lens_cpu is not None and req_pool_indices is not None:
            if isinstance(seq_lens_cpu, torch.Tensor):
                seq_lens = [int(x) for x in seq_lens_cpu.cpu().tolist()]
            else:
                seq_lens = [int(x) for x in seq_lens_cpu]
            req_indices = [int(x) for x in req_pool_indices.detach().cpu().tolist()]
            return list(zip(req_indices, seq_lens))
        reqs = getattr(forward_batch, "reqs", None)
        if reqs is not None:
            pairs: List[Tuple[int, int]] = []
            for req in reqs or []:
                req_pool_idx = getattr(req, "req_pool_idx", None)
                if req_pool_idx is None:
                    continue
                seq_len = getattr(req, "seq_len", None)
                if seq_len is None:
                    seq_len = len(getattr(req, "origin_input_ids", []) or []) + len(
                        getattr(req, "output_ids", []) or []
                    )
                try:
                    pairs.append((int(req_pool_idx), int(seq_len)))
                except (TypeError, ValueError):
                    continue
            if pairs:
                return pairs
        return list(self._last_scheduled_req_lens)

    def _per_layer_req_generation(self, req_idx: int) -> int:
        return int(self._per_layer_req_generations.get(int(req_idx), 0))

    def _per_layer_entry_is_current(self, entry: _LayerKVResidencyEntry) -> bool:
        return int(getattr(entry, "generation", 0)) == self._per_layer_req_generation(
            int(entry.req_idx)
        )

    def _release_per_layer_req_generation(self, req_idx: int) -> int:
        req_idx = int(req_idx)
        old_generation = self._per_layer_req_generation(req_idx)
        self._per_layer_req_generations[req_idx] = old_generation + 1
        self._per_layer_released_req_generations.setdefault(req_idx, set()).add(
            old_generation
        )
        self._per_layer_owned_req_indices.discard(req_idx)
        self._virtual_materialize_plan = None
        self._virtual_materialize_plans_by_layer.clear()
        self._virtual_kvc_demands.clear()
        self._kvc_demand_signature_eval_step = None
        return old_generation

    def _cleanup_released_per_layer_entries(self) -> None:
        if self.config.kvc_backend != "per-layer-arena":
            return
        if not self._per_layer_released_req_generations:
            return
        if self._coresid_optimized_policy_enabled():
            dropped = self._gc_per_layer_residency_tombstones()
            if dropped > 0:
                self._recompute_per_layer_residency_fast_counts()
                self._refresh_kvc_residency_stats()
            return
        released = self._per_layer_released_req_generations
        keys = [
            key
            for key, entry in self._per_layer_residency.items()
            if int(getattr(entry, "generation", 0))
            in released.get(int(entry.req_idx), set())
        ]
        if keys:
            self._drop_per_layer_residency_keys(keys)
            self._recompute_per_layer_residency_fast_counts()
            self._refresh_kvc_residency_stats()
        for req_idx in list(released):
            generations = released.get(req_idx)
            if not generations:
                released.pop(req_idx, None)
                continue
            still_present = any(
                int(entry.req_idx) == int(req_idx)
                and int(getattr(entry, "generation", 0)) in generations
                for entry in self._per_layer_residency.values()
            )
            if not still_present:
                released.pop(req_idx, None)

    def _drop_entries_for_reqs(self, forward_batch: Any) -> None:
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if req_pool_indices is None:
            return
        total_t0 = time.perf_counter() if self.config.profile_detail else 0.0
        t0 = time.perf_counter()
        req_indices = set(int(x) for x in req_pool_indices.detach().cpu().tolist())
        self._add_profile(
            "profile_cleanup_req_indices_ms", (time.perf_counter() - t0) * 1000.0
        )
        if not req_indices:
            return
        t0 = time.perf_counter()
        for req_idx in req_indices:
            if req_idx in self._kvc_evict_cursors:
                self._kvc_evict_cursors.pop(req_idx, None)
                self.stats.kvc_evict_cursor_reset_count += 1
            for cursor_key in list(self._kvc_evict_cursors_by_layer):
                if int(cursor_key[1]) == req_idx:
                    self._kvc_evict_cursors_by_layer.pop(cursor_key, None)
                    self.stats.kvc_layerwise_evict_cursor_reset_count += 1
        self._add_profile(
            "profile_cleanup_cursor_ms", (time.perf_counter() - t0) * 1000.0
        )
        t0 = time.perf_counter()
        to_drop = [key for key in self._residency if key[0] in req_indices]
        self._add_profile(
            "profile_cleanup_global_scan_ms", (time.perf_counter() - t0) * 1000.0
        )
        t0 = time.perf_counter()
        self._drop_residency_keys(to_drop)
        self._add_profile(
            "profile_cleanup_global_drop_ms", (time.perf_counter() - t0) * 1000.0
        )
        t0 = time.perf_counter()
        fast_release_by_req: Dict[int, int] = {}
        if self.config.kvc_backend == "per-layer-arena":
            per_layer_to_drop_by_req: Dict[int, List[Tuple[int, int, int]]] = {}
            for req_idx in req_indices:
                req_idx = int(req_idx)
                req_keys = self._per_layer_owned_keys_by_req.get(req_idx, set())
                if req_keys:
                    self._release_per_layer_req_generation(req_idx)
                    fast_release_by_req[req_idx] = len(req_keys)
                per_layer_to_drop_by_req[req_idx] = list(req_keys)
            per_layer_to_drop = []
        else:
            per_layer_to_drop = [
                key for key in self._per_layer_residency if key[1] in req_indices
            ]
        self._add_profile(
            "profile_cleanup_per_layer_keys_ms",
            (time.perf_counter() - t0) * 1000.0,
        )
        if self.config.profile_detail:
            cleanup_key_count = (
                len(to_drop)
                + len(per_layer_to_drop)
                + sum(fast_release_by_req.values())
            )
            cleanup_token_count = 0
            req_key_counts: Dict[int, int] = {}
            req_token_counts: Dict[int, int] = {}
            for key in to_drop:
                entry = self._residency.get(key)
                if entry is not None:
                    token_count = int(entry.token_count)
                    cleanup_token_count += token_count
                    req_idx = int(key[0])
                    req_key_counts[req_idx] = int(req_key_counts.get(req_idx, 0)) + 1
                    req_token_counts[req_idx] = (
                        int(req_token_counts.get(req_idx, 0)) + token_count
                    )
            for req_idx, key_count in fast_release_by_req.items():
                req_idx = int(req_idx)
                key_count = int(key_count)
                cleanup_token_count += key_count
                req_key_counts[req_idx] = (
                    int(req_key_counts.get(req_idx, 0)) + key_count
                )
                req_token_counts[req_idx] = (
                    int(req_token_counts.get(req_idx, 0)) + key_count
                )
            for key in per_layer_to_drop:
                entry = self._per_layer_residency.get(key)
                if entry is not None:
                    token_count = int(entry.token_count)
                    cleanup_token_count += token_count
                    req_idx = int(key[1])
                    req_key_counts[req_idx] = int(req_key_counts.get(req_idx, 0)) + 1
                    req_token_counts[req_idx] = (
                        int(req_token_counts.get(req_idx, 0)) + token_count
                    )
            req_seq_lens = {
                int(req_idx): int(seq_len)
                for req_idx, seq_len in self._batch_req_indices_and_lens(forward_batch)
            }
            req_details: Dict[int, Dict[str, Any]] = {}
            for req in getattr(forward_batch, "reqs", []) or []:
                req_pool_idx = getattr(req, "req_pool_idx", None)
                if req_pool_idx is None:
                    continue
                try:
                    req_idx = int(req_pool_idx)
                except (TypeError, ValueError):
                    continue
                finished = False
                try:
                    finished = bool(req.finished())
                except Exception:
                    finished = False
                req_details[req_idx] = {
                    "finished": finished,
                    "origin_len": len(getattr(req, "origin_input_ids", []) or []),
                    "output_len": len(getattr(req, "output_ids", []) or []),
                    "seq_len": int(
                        getattr(req, "seq_len", req_seq_lens.get(req_idx, 0)) or 0
                    ),
                }
            req_summary = []
            for req_idx, token_count in sorted(
                req_token_counts.items(), key=lambda item: item[1], reverse=True
            )[:8]:
                detail = req_details.get(int(req_idx), {})
                req_summary.append(
                    {
                        "req_idx": int(req_idx),
                        "key_count": int(req_key_counts.get(int(req_idx), 0)),
                        "token_count": int(token_count),
                        "seq_len": int(
                            detail.get("seq_len", req_seq_lens.get(int(req_idx), 0))
                            or 0
                        ),
                        "batch_seq_len": int(req_seq_lens.get(int(req_idx), 0) or 0),
                        "finished": bool(detail.get("finished", False)),
                        "origin_len": int(detail.get("origin_len", 0) or 0),
                        "output_len": int(detail.get("output_len", 0) or 0),
                    }
                )
            max_req = req_summary[0] if req_summary else {}
            current_modes: Dict[str, int] = {}
            try:
                current_modes = json.loads(
                    self.stats.profile_cleanup_drop_entries_modes or "{}"
                )
            except Exception:
                current_modes = {}
            mode = str(getattr(self, "_current_forward_mode", "") or "unknown")
            current_modes[mode] = int(current_modes.get(mode, 0)) + 1
            self.stats.profile_cleanup_drop_entries_modes = json.dumps(
                current_modes, sort_keys=True
            )
            self.stats.profile_cleanup_drop_entries_req_summary = json.dumps(
                {
                    "mode": mode,
                    "req_count": len(req_token_counts),
                    "top": req_summary,
                },
                sort_keys=True,
            )
            self.stats.profile_cleanup_drop_entries_req_count += len(req_token_counts)
            if int(max_req.get("token_count", 0) or 0) > int(
                self.stats.profile_cleanup_drop_entries_max_req_token_count
            ):
                self.stats.profile_cleanup_drop_entries_max_req_idx = int(
                    max_req.get("req_idx", -1) or -1
                )
                self.stats.profile_cleanup_drop_entries_max_req_key_count = int(
                    max_req.get("key_count", 0) or 0
                )
                self.stats.profile_cleanup_drop_entries_max_req_token_count = int(
                    max_req.get("token_count", 0) or 0
                )
                self.stats.profile_cleanup_drop_entries_max_req_seq_len = int(
                    max_req.get("seq_len", 0) or 0
                )
        t0 = time.perf_counter()
        if self.config.kvc_backend == "per-layer-arena":
            per_layer_dropped_tokens = 0
            for req_idx, req_keys in per_layer_to_drop_by_req.items():
                per_layer_dropped_tokens += (
                    self._drop_per_layer_residency_for_finished_req(
                        req_idx, req_keys, refresh=False
                    )
                )
            if per_layer_dropped_tokens > 0:
                cleaned_reqs = set(int(req_idx) for req_idx in per_layer_to_drop_by_req)
                for req in getattr(forward_batch, "reqs", []) or []:
                    req_pool_idx = getattr(req, "req_pool_idx", None)
                    if req_pool_idx is None or int(req_pool_idx) not in cleaned_reqs:
                        continue
                    try:
                        req.layerkv_per_layer_cleaned = True
                    except Exception:
                        pass
            if per_layer_dropped_tokens > 0:
                refresh_t0 = time.perf_counter()
                self._refresh_per_layer_allocator_stats()
                self._add_profile(
                    "profile_cleanup_per_layer_refresh_allocator_ms",
                    (time.perf_counter() - refresh_t0) * 1000.0,
                )
                refresh_t0 = time.perf_counter()
                self._refresh_kvc_residency_stats()
                self._add_profile(
                    "profile_cleanup_refresh_stats_ms",
                    (time.perf_counter() - refresh_t0) * 1000.0,
                )
        else:
            self._drop_per_layer_residency_keys(per_layer_to_drop)
        self._add_profile(
            "profile_cleanup_per_layer_drop_ms",
            (time.perf_counter() - t0) * 1000.0,
        )
        if self.config.profile_detail:
            self.stats.profile_cleanup_drop_entries_call_count += 1
            self.stats.profile_cleanup_drop_entries_key_count += int(cleanup_key_count)
            self.stats.profile_cleanup_drop_entries_token_count += int(
                cleanup_token_count
            )
            self._add_profile(
                "profile_cleanup_drop_entries_ms",
                (time.perf_counter() - total_t0) * 1000.0,
            )

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
        total_t0 = time.perf_counter() if self.config.profile_detail else 0.0
        if req_idx in self._kvc_evict_cursors:
            self._kvc_evict_cursors.pop(req_idx, None)
            self.stats.kvc_evict_cursor_reset_count += 1
        to_drop = [key for key in self._residency if key[0] == req_idx]
        per_layer_key_count = 0
        if self.config.kvc_backend == "per-layer-arena":
            cleanup_state = self._per_layer_cleanup_state_by_req.get(req_idx)
            if cleanup_state is not None:
                per_layer_key_count = int(cleanup_state.token_count)
                per_layer_to_drop = []
            elif req_idx in self._per_layer_cleanup_layers_by_req:
                per_layer_key_count = int(
                    self._per_layer_cleanup_token_count_by_req.get(req_idx, 0)
                )
                per_layer_to_drop = []
            else:
                per_layer_to_drop = list(
                    self._per_layer_owned_keys_by_req.pop(req_idx, set())
                )
                per_layer_key_count = len(per_layer_to_drop)
        else:
            per_layer_to_drop = [
                key for key in self._per_layer_residency if key[1] == req_idx
            ]
        native_runs = list(self._native_kvc_runs_by_req.get(req_idx, []))
        has_per_layer_cleanup = bool(per_layer_to_drop) or (
            self.config.kvc_backend == "per-layer-arena"
            and (
                req_idx in self._per_layer_cleanup_state_by_req
                or req_idx in self._per_layer_cleanup_layers_by_req
            )
        )
        if not to_drop and not has_per_layer_cleanup and not native_runs:
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
        token_count += sum(max(0, int(run.end) - int(run.start)) for run in native_runs)
        self._drop_residency_keys(to_drop)
        if self.config.kvc_backend == "per-layer-arena":
            token_count += self._drop_per_layer_residency_for_finished_req(
                req_idx, per_layer_to_drop, refresh=False
            )
        else:
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
        if self.config.profile_detail:
            self.stats.profile_cleanup_finished_call_count += 1
            self.stats.profile_cleanup_finished_key_count += int(
                len(to_drop)
                + (per_layer_key_count or len(per_layer_to_drop))
                + len(native_runs)
            )
            self.stats.profile_cleanup_finished_token_count += int(token_count)
            self._add_profile(
                "profile_cleanup_finished_ms",
                (time.perf_counter() - total_t0) * 1000.0,
            )

    def _prune_entries_for_active_lengths(self, forward_batch: Any) -> None:
        active_lens = {
            req_idx: seq_len
            for req_idx, seq_len in self._batch_req_indices_and_lens(forward_batch)
        }
        if not active_lens:
            return
        last_active_lens = getattr(self, "_last_kvc_prune_active_lens", None)
        if (
            last_active_lens is not None
            and active_lens.keys() == last_active_lens.keys()
            and all(
                int(seq_len) >= int(last_active_lens.get(req_idx, -1))
                for req_idx, seq_len in active_lens.items()
            )
        ):
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
        self._last_kvc_prune_active_lens = dict(active_lens)

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

    def _drop_per_layer_residency_for_finished_req_indexed(
        self,
        req_idx: int,
        keys: List[Tuple[int, int, int]],
        *,
        refresh: bool = True,
    ) -> Optional[int]:
        if self.config.kvc_backend != "per-layer-arena":
            return None
        req_idx = int(req_idx)
        if any(
            int(entry.req_idx) == req_idx
            for pending in self._pending_kvc_evict_events
            for entry in pending.entries
        ):
            self._finalize_kvc_evictions(block=True)
        state = self._per_layer_cleanup_state_by_req.pop(req_idx, None)
        layers = set(state.layers if state is not None else ())
        if not layers:
            layers = set(self._per_layer_cleanup_layers_by_req.get(req_idx, set()))
        if not layers:
            if keys:
                return None
            self._release_per_layer_req_generation(req_idx)
            self._per_layer_owned_keys_by_req.pop(req_idx, None)
            self._per_layer_cleanup_state_by_req.pop(req_idx, None)
            self._per_layer_cleanup_tracked_keys_by_req.pop(req_idx, None)
            self._per_layer_cleanup_token_count_by_req.pop(req_idx, None)
            return 0

        loop_t0 = time.perf_counter()
        self._release_per_layer_req_generation(req_idx)
        token_count = int(
            state.token_count
            if state is not None
            else self._per_layer_cleanup_token_count_by_req.pop(req_idx, 0)
        )
        fallback_owned_keys = self._per_layer_owned_keys_by_req.pop(req_idx, set())
        self._per_layer_cleanup_tracked_keys_by_req.pop(req_idx, None)
        self._per_layer_cleanup_layers_by_req.pop(req_idx, None)
        had_page_index = req_idx in self._per_layer_page_keys_by_req
        if had_page_index:
            dropped_entry_count = self._take_per_layer_active_entry_count_for_req(
                req_idx
            )
            self._drop_per_layer_page_index_for_req(req_idx, collect_entries=False)
            self._mark_per_layer_page_entries_dropped(dropped_entry_count)
            self._gc_per_layer_residency_tombstones()
        else:
            owned_keys = list(fallback_owned_keys)
            if not owned_keys:
                owned_keys = list(keys)
            for key in owned_keys:
                entry = self._per_layer_residency.pop(key, None)
                if entry is None:
                    continue
                self._remove_resident_group(
                    "kvc", -1, (int(entry.req_idx), int(entry.pos))
                )

        offloaded_req_keys = self._per_layer_offloaded_keys_by_req.pop(req_idx, set())
        if offloaded_req_keys:
            self._per_layer_offloaded_keys.difference_update(offloaded_req_keys)
        dirty_req_layers: Set[Tuple[int, int]] = set()
        indexed_host_slots_by_layer: Dict[int, int] = {}
        for layer_id in layers:
            req_layer = (req_idx, int(layer_id))
            self._per_layer_offloaded_keys_by_req_layer.pop(req_layer, None)
            runs = self._per_layer_offloaded_runs_by_req_layer.pop(req_layer, None)
            if runs:
                slot_bits = int(indexed_host_slots_by_layer.get(int(layer_id), 0))
                for run in runs:
                    slot_bits |= self._locs_to_bitset(run.host_slot_list(), min_value=0)
                indexed_host_slots_by_layer[int(layer_id)] = slot_bits
            self._per_layer_offloaded_sorted_by_req_layer.pop(req_layer, None)
            self._per_layer_offloaded_positions_by_req_layer.pop(req_layer, None)
            dirty_req_layers.add(req_layer)
        self._per_layer_offloaded_sorted_by_req.pop(req_idx, None)
        if offloaded_req_keys or dirty_req_layers:
            self._per_layer_offloaded_version += 1
            self._bump_per_layer_offloaded_versions(
                {int(layer_id) for _req_idx, layer_id in dirty_req_layers}
            )
            self._per_layer_offloaded_dirty_reqs.add(req_idx)
            self._per_layer_offloaded_dirty_req_layers.update(dirty_req_layers)

        free_loc_bits_by_layer: Dict[int, int] = {}
        free_loc_counts_by_layer: Dict[int, int] = {}
        host_slots_by_layer: Dict[int, List[int]] = {}
        resident_tokens = 0
        offloaded_tokens = 0
        if state is not None:
            loc_items = tuple(state.loc_bits_by_layer.items())
            for layer_id, loc_bits in loc_items:
                layer_id = int(layer_id)
                loc_bits = int(loc_bits)
                if loc_bits <= 0:
                    continue
                free_loc_bits_by_layer[layer_id] = loc_bits
                loc_count = int(state.loc_counts_by_layer.get(layer_id, 0))
                if loc_count <= 0:
                    loc_count = len(self._bitset_to_locs(loc_bits))
                free_loc_counts_by_layer[layer_id] = loc_count
                resident_tokens += loc_count
            for layer_id, host_slot_bits in tuple(
                state.host_slot_bits_by_layer.items()
            ):
                layer_id = int(layer_id)
                host_slot_bits = int(host_slot_bits)
                if host_slot_bits > 0:
                    indexed_host_slots_by_layer[layer_id] = (
                        int(indexed_host_slots_by_layer.get(layer_id, 0))
                        | host_slot_bits
                    )
            host_layer_ids = tuple(indexed_host_slots_by_layer)
        else:
            host_layer_ids = tuple(layers)
            for layer_id in layers:
                layer_id = int(layer_id)
                loc_bits = int(
                    self._per_layer_cleanup_locs_by_req_layer.pop(
                        (req_idx, layer_id), 0
                    )
                )
                if loc_bits > 0:
                    free_loc_bits_by_layer[layer_id] = loc_bits
                    loc_count = int(
                        self._per_layer_cleanup_loc_count_by_req_layer.pop(
                            (req_idx, layer_id), 0
                        )
                    )
                    if loc_count <= 0:
                        loc_count = len(self._bitset_to_locs(loc_bits))
                    free_loc_counts_by_layer[layer_id] = loc_count
                    resident_tokens += loc_count
                else:
                    self._per_layer_cleanup_loc_count_by_req_layer.pop(
                        (req_idx, layer_id), None
                    )
                host_slot_bits = int(
                    self._per_layer_cleanup_host_slots_by_req_layer.pop(
                        (req_idx, layer_id), 0
                    )
                )
                if host_slot_bits > 0:
                    indexed_host_slots_by_layer[layer_id] = (
                        int(indexed_host_slots_by_layer.get(layer_id, 0))
                        | host_slot_bits
                    )

        for layer_id in host_layer_ids:
            layer_id = int(layer_id)
            indexed_host_slots = int(indexed_host_slots_by_layer.pop(layer_id, 0))
            if indexed_host_slots > 0:
                slot_list = self._bitset_to_locs(indexed_host_slots)
                if slot_list:
                    host_slots_by_layer[layer_id] = slot_list
                    indexed_host_count = int(
                        state.host_slot_counts_by_layer.get(layer_id, 0)
                        if state is not None
                        else self._per_layer_cleanup_host_slot_count_by_req_layer.pop(
                            (req_idx, layer_id), 0
                        )
                    )
                    offloaded_tokens += indexed_host_count or len(slot_list)
                    current = int(
                        self._per_layer_offloaded_token_count_by_layer.get(layer_id, 0)
                    )
                    next_count = max(
                        0, current - (indexed_host_count or len(slot_list))
                    )
                    if next_count:
                        self._per_layer_offloaded_token_count_by_layer[layer_id] = (
                            next_count
                        )
                    else:
                        self._per_layer_offloaded_token_count_by_layer.pop(
                            layer_id, None
                        )
            else:
                if state is None:
                    self._per_layer_cleanup_host_slot_count_by_req_layer.pop(
                        (req_idx, layer_id), None
                    )
        if state is not None:
            pass
        else:
            self._per_layer_cleanup_state_by_req.pop(req_idx, None)
        self._per_layer_resident_token_count_fast = max(
            0, int(self._per_layer_resident_token_count_fast) - int(resident_tokens)
        )
        self._per_layer_offloaded_token_count_fast = max(
            0, int(self._per_layer_offloaded_token_count_fast) - int(offloaded_tokens)
        )
        self._add_profile(
            "profile_cleanup_per_layer_collect_ms",
            (time.perf_counter() - loop_t0) * 1000.0,
        )
        self._add_profile(
            "profile_cleanup_per_layer_loop_ms",
            (time.perf_counter() - loop_t0) * 1000.0,
        )

        t0 = time.perf_counter()
        for layer_id, loc_bits in free_loc_bits_by_layer.items():
            self._push_per_layer_overwrite_bits(
                int(layer_id),
                int(loc_bits),
                token_count=int(free_loc_counts_by_layer.get(int(layer_id), 0)),
            )
        self._add_profile(
            "profile_cleanup_per_layer_free_locs_ms",
            (time.perf_counter() - t0) * 1000.0,
        )
        if refresh and (free_loc_bits_by_layer or free_loc_counts_by_layer):
            t0 = time.perf_counter()
            self._refresh_per_layer_allocator_stats()
            self._add_profile(
                "profile_cleanup_per_layer_refresh_allocator_ms",
                (time.perf_counter() - t0) * 1000.0,
            )
        if self._host_store is not None and host_slots_by_layer:
            t0 = time.perf_counter()
            for layer_id, host_slots in host_slots_by_layer.items():
                self._host_store.free_per_layer(int(layer_id), host_slots)
            if int(getattr(self._host_store, "used_count", 0)) == 0:
                self._per_layer_offloaded_keys.clear()
                self._per_layer_offloaded_keys_by_req.clear()
                self._per_layer_offloaded_keys_by_req_layer.clear()
                self._per_layer_offloaded_runs_by_req_layer.clear()
                self._per_layer_offloaded_token_count_by_layer.clear()
                self._per_layer_offloaded_version_by_layer.clear()
                self._per_layer_offloaded_sorted_by_req.clear()
                self._per_layer_offloaded_sorted_by_req_layer.clear()
                self._per_layer_offloaded_positions_by_req_layer.clear()
                self._per_layer_offloaded_dirty_reqs.clear()
                self._per_layer_offloaded_dirty_req_layers.clear()
                self._per_layer_offloaded_token_count_fast = 0
                self._per_layer_offloaded_version += 1
            self._add_profile(
                "profile_cleanup_per_layer_host_free_ms",
                (time.perf_counter() - t0) * 1000.0,
            )
        if refresh:
            t0 = time.perf_counter()
            self._refresh_kvc_residency_stats()
            self._add_profile(
                "profile_cleanup_refresh_stats_ms",
                (time.perf_counter() - t0) * 1000.0,
            )
        if token_count <= 0:
            token_count = int(resident_tokens) + int(offloaded_tokens)
        return int(token_count)

    def _drop_per_layer_residency_for_finished_req(
        self,
        req_idx: int,
        keys: List[Tuple[int, int, int]],
        *,
        refresh: bool = True,
    ) -> int:
        if self.config.kvc_backend != "per-layer-arena":
            self._drop_per_layer_residency_keys(keys)
            return 0
        req_idx = int(req_idx)
        indexed = self._drop_per_layer_residency_for_finished_req_indexed(
            req_idx, keys, refresh=refresh
        )
        if indexed is not None:
            return int(indexed)
        if not keys:
            self._per_layer_owned_keys_by_req.pop(req_idx, None)
            self._per_layer_owned_req_indices.discard(req_idx)
            return 0

        free_locs_by_layer: Dict[int, List[int]] = {}
        protected_locs_by_layer: Dict[int, List[int]] = {}
        host_slots_by_layer: Dict[int, List[int]] = {}
        token_count_total = 0
        resident_tokens = 0
        offloaded_tokens = 0
        offloaded_tokens_by_layer: Dict[int, int] = {}
        changed_offloaded_index = False
        event_sync_ms = 0.0
        loop_t0 = time.perf_counter()
        collect_t0 = time.perf_counter()

        offloaded_req_keys = self._per_layer_offloaded_keys_by_req.pop(req_idx, set())
        if offloaded_req_keys:
            self._per_layer_offloaded_keys.difference_update(offloaded_req_keys)
            changed_offloaded_index = True
        dirty_req_layers: Set[Tuple[int, int]] = set()
        for req_layer in list(self._per_layer_offloaded_keys_by_req_layer):
            if int(req_layer[0]) != req_idx:
                continue
            self._per_layer_offloaded_keys_by_req_layer.pop(req_layer, None)
            self._per_layer_offloaded_runs_by_req_layer.pop(req_layer, None)
            self._per_layer_offloaded_sorted_by_req_layer.pop(req_layer, None)
            self._per_layer_offloaded_positions_by_req_layer.pop(req_layer, None)
            dirty_req_layers.add((int(req_layer[0]), int(req_layer[1])))
            changed_offloaded_index = True

        for key in keys:
            entry = self._per_layer_residency.pop(key, None)
            if entry is None:
                continue
            self._unindex_per_layer_page_entry(entry)
            layer_id = int(entry.layer_id)
            token_count = int(entry.token_count)
            token_count_total += token_count
            if entry.state == "offloaded":
                offloaded_tokens += token_count
                offloaded_tokens_by_layer[layer_id] = (
                    int(offloaded_tokens_by_layer.get(layer_id, 0)) + token_count
                )
            elif entry.state in ("resident", "reloading"):
                resident_tokens += token_count
            if entry.ready_event is not None and not entry.ready_event.query():
                t0 = time.perf_counter()
                entry.ready_event.synchronize()
                event_sync_ms += (time.perf_counter() - t0) * 1000.0
            if entry.device_locs is not None:
                free_locs_by_layer.setdefault(layer_id, []).extend(entry.device_locs)
            elif entry.device_loc is not None:
                free_locs_by_layer.setdefault(layer_id, []).append(
                    int(entry.device_loc)
                )
            elif entry.evicted_device_locs:
                protected_locs_by_layer.setdefault(layer_id, []).extend(
                    int(loc) for loc in entry.evicted_device_locs
                )
            if entry.host_slots is not None:
                host_slots_by_layer.setdefault(layer_id, []).extend(entry.host_slots)
            elif entry.host_slot is not None:
                host_slots_by_layer.setdefault(layer_id, []).append(
                    int(entry.host_slot)
                )

        self._per_layer_resident_token_count_fast = max(
            0, int(self._per_layer_resident_token_count_fast) - int(resident_tokens)
        )
        self._per_layer_offloaded_token_count_fast = max(
            0, int(self._per_layer_offloaded_token_count_fast) - int(offloaded_tokens)
        )
        for layer_id, tokens in offloaded_tokens_by_layer.items():
            current = int(
                self._per_layer_offloaded_token_count_by_layer.get(layer_id, 0)
            )
            next_count = max(0, current - int(tokens))
            if next_count:
                self._per_layer_offloaded_token_count_by_layer[layer_id] = next_count
            else:
                self._per_layer_offloaded_token_count_by_layer.pop(layer_id, None)
        if changed_offloaded_index:
            self._per_layer_offloaded_version += 1
            self._bump_per_layer_offloaded_versions(
                {int(layer_id) for _req_idx, layer_id in dirty_req_layers}
            )
            self._per_layer_offloaded_dirty_reqs.add(req_idx)
            self._per_layer_offloaded_dirty_req_layers.update(dirty_req_layers)
            self._per_layer_offloaded_sorted_by_req.pop(req_idx, None)

        self._add_profile("profile_cleanup_per_layer_event_sync_ms", event_sync_ms)
        self._add_profile(
            "profile_cleanup_per_layer_collect_ms",
            (time.perf_counter() - collect_t0) * 1000.0,
        )
        self._add_profile(
            "profile_cleanup_per_layer_loop_ms",
            (time.perf_counter() - loop_t0) * 1000.0,
        )

        t0 = time.perf_counter()
        for layer_id, locs in free_locs_by_layer.items():
            self._push_per_layer_overwrite_locs(int(layer_id), locs, assume_new=True)
        self._add_profile(
            "profile_cleanup_per_layer_free_locs_ms",
            (time.perf_counter() - t0) * 1000.0,
        )

        released_protected_locs = False
        t0 = time.perf_counter()
        for layer_id, locs in protected_locs_by_layer.items():
            if locs:
                self._release_protected_per_layer_locs(
                    int(layer_id), locs, refresh=False
                )
                released_protected_locs = True
        self._add_profile(
            "profile_cleanup_per_layer_run_range_ms",
            (time.perf_counter() - t0) * 1000.0,
        )
        if refresh and (free_locs_by_layer or released_protected_locs):
            t0 = time.perf_counter()
            self._refresh_per_layer_allocator_stats()
            self._add_profile(
                "profile_cleanup_per_layer_refresh_allocator_ms",
                (time.perf_counter() - t0) * 1000.0,
            )
        if self._host_store is not None:
            t0 = time.perf_counter()
            for layer_id, host_slots in host_slots_by_layer.items():
                if host_slots:
                    self._host_store.free_per_layer(int(layer_id), host_slots)
            self._add_profile(
                "profile_cleanup_per_layer_host_free_ms",
                (time.perf_counter() - t0) * 1000.0,
            )
        self._per_layer_owned_keys_by_req.pop(req_idx, None)
        self._per_layer_owned_req_indices.discard(req_idx)
        self._drop_per_layer_page_index_for_req(req_idx)
        if refresh:
            t0 = time.perf_counter()
            self._refresh_kvc_residency_stats()
            self._add_profile(
                "profile_cleanup_refresh_stats_ms",
                (time.perf_counter() - t0) * 1000.0,
            )
        return int(token_count_total)

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
        run_range_ms = 0.0
        event_sync_ms = 0.0
        loop_t0 = time.perf_counter()
        collect_t0 = time.perf_counter()
        for key in keys:
            self._untrack_per_layer_offloaded_key(key)
            entry = self._per_layer_residency.pop(key, None)
            if entry is not None:
                if self.config.kvc_backend == "per-layer-arena":
                    self._unindex_per_layer_page_entry(entry)
                layer_id = int(entry.layer_id)
                req_idx = int(entry.req_idx)
                token_count = int(entry.token_count)
                if entry.state == "offloaded":
                    t0 = time.perf_counter()
                    self._remove_per_layer_offloaded_run_range(
                        layer_id,
                        req_idx,
                        int(entry.pos),
                        token_count,
                    )
                    run_range_ms += (time.perf_counter() - t0) * 1000.0
                    self._per_layer_offloaded_token_count_fast = max(
                        0,
                        self._per_layer_offloaded_token_count_fast - token_count,
                    )
                elif entry.state in ("resident", "reloading"):
                    self._per_layer_resident_token_count_fast = max(
                        0,
                        self._per_layer_resident_token_count_fast - token_count,
                    )
                if entry.ready_event is not None and not entry.ready_event.query():
                    t0 = time.perf_counter()
                    entry.ready_event.synchronize()
                    event_sync_ms += (time.perf_counter() - t0) * 1000.0
                if entry.device_locs is not None:
                    self._remove_per_layer_cleanup_locs(entry, entry.device_locs)
                    free_locs_by_layer.setdefault(layer_id, []).extend(
                        entry.device_locs
                    )
                elif entry.device_loc is not None:
                    self._remove_per_layer_cleanup_locs(entry, [int(entry.device_loc)])
                    free_locs_by_layer.setdefault(layer_id, []).append(
                        int(entry.device_loc)
                    )
                elif (
                    self.config.kvc_backend == "per-layer-arena"
                    and entry.evicted_device_locs
                ):
                    self._release_protected_per_layer_locs(
                        layer_id,
                        [int(loc) for loc in entry.evicted_device_locs],
                        refresh=False,
                    )
                    released_protected_locs = True
                if entry.host_slots is not None:
                    self._remove_per_layer_cleanup_host_slots(entry, entry.host_slots)
                    host_slots_by_layer.setdefault(layer_id, []).extend(
                        entry.host_slots
                    )
                elif entry.host_slot is not None:
                    self._remove_per_layer_cleanup_host_slots(
                        entry, [int(entry.host_slot)]
                    )
                    host_slots_by_layer.setdefault(layer_id, []).append(
                        int(entry.host_slot)
                    )
                if remove_groups:
                    self._remove_resident_group(
                        "kvc",
                        layer_id,
                        (layer_id, req_idx, int(entry.pos)),
                    )
        self._add_profile("profile_cleanup_per_layer_run_range_ms", run_range_ms)
        self._add_profile("profile_cleanup_per_layer_event_sync_ms", event_sync_ms)
        self._add_profile(
            "profile_cleanup_per_layer_collect_ms",
            (time.perf_counter() - collect_t0) * 1000.0,
        )
        self._add_profile(
            "profile_cleanup_per_layer_loop_ms",
            (time.perf_counter() - loop_t0) * 1000.0,
        )
        t0 = time.perf_counter()
        for layer_id, locs in free_locs_by_layer.items():
            self._push_per_layer_overwrite_locs(int(layer_id), locs, assume_new=True)
        self._add_profile(
            "profile_cleanup_per_layer_free_locs_ms",
            (time.perf_counter() - t0) * 1000.0,
        )
        if free_locs_by_layer or released_protected_locs:
            t0 = time.perf_counter()
            self._refresh_per_layer_allocator_stats()
            self._add_profile(
                "profile_cleanup_per_layer_refresh_allocator_ms",
                (time.perf_counter() - t0) * 1000.0,
            )
        if self._host_store is not None:
            t0 = time.perf_counter()
            for layer_id, host_slots in host_slots_by_layer.items():
                if host_slots:
                    self._host_store.free_per_layer(int(layer_id), host_slots)
            self._add_profile(
                "profile_cleanup_per_layer_host_free_ms",
                (time.perf_counter() - t0) * 1000.0,
            )
        t0 = time.perf_counter()
        self._refresh_kvc_residency_stats()
        self._add_profile(
            "profile_cleanup_refresh_stats_ms",
            (time.perf_counter() - t0) * 1000.0,
        )

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
                sorted_keys, positions = (
                    self._sorted_per_layer_offloaded_keys_for_req_layer(
                        req_idx, layer_id
                    )
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
                    if (
                        entry is None
                        or entry.state != "offloaded"
                        or not self._per_layer_entry_is_current(entry)
                    ):
                        stale.append(key)
                        continue
                    if int(entry.pos) >= int(required_prefix_len):
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

    def _kvc_evict_selection_signature(
        self, forward_batch: Any, need_tokens: int
    ) -> Tuple[int, int, int, int]:
        pairs = self._batch_req_indices_and_lens(forward_batch)
        req_count = len(pairs)
        token_sum = sum(max(0, int(seq_len)) for _req_idx, seq_len in pairs)
        rolling = 0
        for req_idx, seq_len in sorted(
            (int(req_idx), int(seq_len)) for req_idx, seq_len in pairs
        ):
            rolling = (
                rolling * 1000003 + int(req_idx) * 9176 + max(0, int(seq_len))
            ) & 0x7FFFFFFF
        return (
            int(self._align_tokens_down(max(0, int(need_tokens)))),
            int(req_count * 1000003 + token_sum),
            int(rolling),
            int(self.config.kvc_backend == "per-layer-arena"),
        )

    def _requires_common_per_layer_kvc_eviction(self) -> bool:
        return (
            self.config.kvc_backend == "per-layer-arena"
            and int(self._force_common_kvc_evict_tokens or 0) > 0
        )

    def _clear_prepared_kvc_evict_selection(
        self, *, clear_host_slots: bool = True, clear_schedule: bool = True
    ) -> None:
        if clear_host_slots:
            self._clear_prepared_kvc_evict_host_slots()
        self._prepared_kvc_evict_entries = []
        self._prepared_kvc_evict_signature = (0, 0, 0, 0)
        self._prepared_kvc_evict_cursors = {}
        self._prepared_kvc_evict_cursors_by_layer = {}
        if clear_schedule:
            self._clear_prepared_kvc_evict_selection_schedule(
                clear_host_slots=clear_host_slots
            )

    def _clear_prepared_kvc_evict_selection_schedule(
        self, *, clear_host_slots: bool = True
    ) -> None:
        if clear_host_slots:
            self._clear_prepared_kvc_evict_host_slots()
        self._prepared_kvc_evict_schedule = []
        self._prepared_kvc_evict_schedule_signature = (0, 0, 0, 0)

    def _clear_prepared_kvc_evict_host_slots(self) -> None:
        slots_by_entry = getattr(self, "_prepared_kvc_evict_host_slots_by_entry", {})
        entries = getattr(self, "_prepared_kvc_evict_host_entries", [])
        if slots_by_entry and self._host_store is not None:
            self._free_per_layer_host_slots_for_entries(entries, slots_by_entry)
        self._prepared_kvc_evict_host_signature = (0, 0, 0, 0)
        self._prepared_kvc_evict_host_entries = []
        self._prepared_kvc_evict_host_slots = []
        self._prepared_kvc_evict_host_slots_by_entry = {}

    def _free_per_layer_host_slots_for_entries(
        self,
        entries: List[_LayerKVResidencyEntry],
        slots_by_entry: Dict[int, List[int]],
    ) -> None:
        if self._host_store is None:
            return
        slots_by_layer: Dict[int, List[int]] = {}
        for entry in entries:
            slots = slots_by_entry.get(id(entry), [])
            if slots:
                slots_by_layer.setdefault(int(entry.layer_id), []).extend(
                    int(slot) for slot in slots
                )
        for layer_id, slots in slots_by_layer.items():
            if slots:
                self._host_store.free_per_layer(int(layer_id), slots)

    def _prepare_kvc_evict_host_slots(
        self,
        selected: List[_LayerKVResidencyEntry],
        signature: Tuple[int, int, int, int],
    ) -> None:
        if self.config.kvc_backend != "per-layer-arena":
            return
        if not selected:
            return
        if not self._optimized_profile_enabled():
            return
        if (
            self._prepared_kvc_evict_host_entries
            and self._prepared_kvc_evict_host_signature == signature
            and tuple(id(entry) for entry in self._prepared_kvc_evict_host_entries)
            == tuple(id(entry) for entry in selected)
        ):
            return
        self._clear_prepared_kvc_evict_host_slots()
        backup_entries = [
            entry
            for entry in selected
            if str(entry.state) in ("resident", "prepared")
            and entry.device_loc_list()
            and len(entry.host_slot_list()) != int(entry.token_count)
        ]
        if not backup_entries:
            return
        self._ensure_host_store()
        with self._profile("profile_kvc_evict_host_prealloc_ms"):
            slots = self._host_store.alloc_per_layer(backup_entries)
        if slots is None:
            return
        slots_by_entry: Dict[int, List[int]] = {}
        offset = 0
        for entry in backup_entries:
            page_slots = slots[offset : offset + int(entry.token_count)]
            offset += int(entry.token_count)
            slots_by_entry[id(entry)] = [int(slot) for slot in page_slots]
        self._prepared_kvc_evict_host_signature = signature
        self._prepared_kvc_evict_host_entries = list(backup_entries)
        self._prepared_kvc_evict_host_slots = [int(slot) for slot in slots]
        self._prepared_kvc_evict_host_slots_by_entry = slots_by_entry
        self.stats.kvc_evict_host_prealloc_count += 1
        self.stats.kvc_evict_host_prealloc_token_count += sum(
            int(entry.token_count) for entry in backup_entries
        )

    def _consume_prepared_kvc_evict_host_slots(
        self, backup_entries: List[_LayerKVResidencyEntry]
    ) -> Optional[Tuple[List[int], Dict[int, List[int]]]]:
        slots_by_entry = getattr(self, "_prepared_kvc_evict_host_slots_by_entry", {})
        if not slots_by_entry:
            return None
        if tuple(id(entry) for entry in self._prepared_kvc_evict_host_entries) != tuple(
            id(entry) for entry in backup_entries
        ):
            self.stats.kvc_evict_host_prealloc_invalid_count += 1
            self._clear_prepared_kvc_evict_host_slots()
            return None
        host_slots = getattr(self, "_prepared_kvc_evict_host_slots", [])
        expected_token_count = sum(int(entry.token_count) for entry in backup_entries)
        if len(host_slots) != expected_token_count:
            self.stats.kvc_evict_host_prealloc_invalid_count += 1
            self._clear_prepared_kvc_evict_host_slots()
            return None
        for entry in backup_entries:
            page_slots = slots_by_entry.get(id(entry), [])
            if len(page_slots) != int(entry.token_count):
                self.stats.kvc_evict_host_prealloc_invalid_count += 1
                self._clear_prepared_kvc_evict_host_slots()
                return None
        out_by_entry = slots_by_entry
        self._prepared_kvc_evict_host_signature = (0, 0, 0, 0)
        self._prepared_kvc_evict_host_entries = []
        self._prepared_kvc_evict_host_slots = []
        self._prepared_kvc_evict_host_slots_by_entry = {}
        self.stats.kvc_evict_host_prealloc_hit_count += 1
        return host_slots, out_by_entry

    def _planned_kvc_evict_need_tokens(self, forward_batch: Any) -> int:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return 0
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
                return 0
            if (
                self.config.kvc_backend == "per-layer-arena"
                and not self._per_layer_virtual_scratch_enabled()
            ):
                return 0
        if effective_kvc_reclaim_mb <= 0:
            return 0
        if self._bytes_per_token_all_layers <= 0:
            return 0
        if self._uses_layer_aware_kvc_plan() and self._planned_kvc_token_target > 0:
            target_tokens = sum(
                int(x) for x in self._planned_kvc_tokens_by_layer.values()
            )
        elif self.config.kvc_backend == "per-layer-arena":
            target_tokens = self._limit_offloaded_per_layer_tokens(
                effective_kvc_reclaim_mb
            )
        else:
            target_tokens = self._limit_offloaded_tokens(effective_kvc_reclaim_mb)
        pending_evict_tokens = (
            self._pending_evict_token_count()
            if self.config.kvc_backend in ("virtual-arena", "per-layer-arena")
            else 0
        )
        if self.config.dynamic_pressure_from_kvc:
            finalized_this_step_tokens = int(
                getattr(self, "_kvc_evict_finalized_this_step_tokens", 0) or 0
            )
            if pending_evict_tokens > 0 or finalized_this_step_tokens > 0:
                return 0
            offloaded_or_pending = self._offloaded_token_count() + pending_evict_tokens
            margin_tokens = self._dynamic_pressure_hysteresis_tokens()
            lower_bound = (
                int(target_tokens)
                if int(target_tokens) <= int(margin_tokens)
                else max(0, int(target_tokens) - int(margin_tokens))
            )
            if offloaded_or_pending > 0 and offloaded_or_pending >= lower_bound:
                return 0
            if margin_tokens > 0:
                target_tokens = int(target_tokens) + int(margin_tokens)
        need_tokens = max(
            0, int(target_tokens) - self._offloaded_token_count() - pending_evict_tokens
        )
        if self.config.dynamic_pressure_from_kvc:
            need_tokens = min(
                int(need_tokens),
                self._dynamic_pressure_topup_step_tokens(margin_tokens),
            )
            need_tokens = self._align_tokens_down(int(need_tokens))
        return int(need_tokens)

    def _prepare_kvc_evict_selection(
        self, forward_batch: Any, need_tokens: int
    ) -> bool:
        need_tokens = self._align_tokens_down(max(0, int(need_tokens)))
        if need_tokens <= 0:
            return False
        signature = self._kvc_evict_selection_signature(forward_batch, need_tokens)
        if (
            self._prepared_kvc_evict_entries
            and self._prepared_kvc_evict_signature == signature
        ):
            self._prepare_kvc_evict_host_slots(
                self._prepared_kvc_evict_entries, signature
            )
            return True
        old_cursors = dict(self._kvc_evict_cursors)
        old_cursors_by_layer = dict(self._kvc_evict_cursors_by_layer)
        t0 = time.perf_counter()
        try:
            selected = self._select_resident_entries_for_eviction(
                forward_batch, need_tokens
            )
            prepared_cursors = dict(self._kvc_evict_cursors)
            prepared_cursors_by_layer = dict(self._kvc_evict_cursors_by_layer)
        finally:
            self._kvc_evict_cursors = old_cursors
            self._kvc_evict_cursors_by_layer = old_cursors_by_layer
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not selected:
            self._clear_prepared_kvc_evict_selection(clear_schedule=False)
            return False
        self._prepared_kvc_evict_entries = list(selected)
        self._prepared_kvc_evict_signature = signature
        self._prepared_kvc_evict_cursors = prepared_cursors
        self._prepared_kvc_evict_cursors_by_layer = prepared_cursors_by_layer
        self._prepare_kvc_evict_host_slots(selected, signature)
        self.stats.kvc_evict_preselect_count += 1
        self.stats.kvc_evict_preselect_token_count += sum(
            int(entry.token_count) for entry in selected
        )
        self.stats.kvc_evict_preselect_ms += elapsed_ms
        return True

    def _prepare_kvc_evict_selection_schedule(
        self, forward_batch: Any, need_tokens: int, *, max_steps: int = 16
    ) -> bool:
        need_tokens = self._align_tokens_down(max(0, int(need_tokens)))
        max_steps = max(1, int(max_steps))
        if need_tokens <= 0:
            return False
        if self.config.kvc_backend != "per-layer-arena":
            return self._prepare_kvc_evict_selection(forward_batch, need_tokens)
        signature = self._kvc_evict_selection_signature(forward_batch, need_tokens)
        if (
            self._prepared_kvc_evict_schedule
            and self._prepared_kvc_evict_schedule_signature == signature
        ):
            self._prepare_kvc_evict_host_slots(
                self._prepared_kvc_evict_schedule[0][0], signature
            )
            return True

        old_cursors = dict(self._kvc_evict_cursors)
        old_cursors_by_layer = dict(self._kvc_evict_cursors_by_layer)
        old_offloaded_by_layer = dict(self._per_layer_offloaded_token_count_by_layer)
        old_offloaded_fast = int(self._per_layer_offloaded_token_count_fast)
        schedule: List[
            Tuple[
                List[_LayerKVResidencyEntry],
                Dict[int, int],
                Dict[Tuple[int, int], int],
                int,
            ]
        ] = []
        t0 = time.perf_counter()
        try:
            if self.config.dynamic_pressure_from_kvc:
                margin_tokens = self._dynamic_pressure_hysteresis_tokens()
                future_need = self._dynamic_pressure_topup_step_tokens(margin_tokens)
                future_need = self._align_tokens_down(max(need_tokens, future_need))
            else:
                future_need = need_tokens
            for step_idx in range(max_steps):
                chunk_need = need_tokens if step_idx == 0 else future_need
                chunk_need = self._align_tokens_down(max(0, int(chunk_need)))
                if chunk_need <= 0:
                    break
                selected = self._select_resident_entries_for_eviction(
                    forward_batch, chunk_need
                )
                token_count = sum(int(entry.token_count) for entry in selected)
                if token_count <= 0:
                    break
                by_layer: Dict[int, int] = {}
                for entry in selected:
                    layer_id = int(entry.layer_id)
                    by_layer[layer_id] = by_layer.get(layer_id, 0) + int(
                        entry.token_count
                    )
                for layer_id, layer_tokens in by_layer.items():
                    self._per_layer_offloaded_token_count_by_layer[layer_id] = int(
                        self._per_layer_offloaded_token_count_by_layer.get(layer_id, 0)
                    ) + int(layer_tokens)
                self._per_layer_offloaded_token_count_fast += int(token_count)
                schedule.append(
                    (
                        list(selected),
                        dict(self._kvc_evict_cursors),
                        dict(self._kvc_evict_cursors_by_layer),
                        int(token_count),
                    )
                )
        finally:
            self._kvc_evict_cursors = old_cursors
            self._kvc_evict_cursors_by_layer = old_cursors_by_layer
            self._per_layer_offloaded_token_count_by_layer = old_offloaded_by_layer
            self._per_layer_offloaded_token_count_fast = old_offloaded_fast
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not schedule:
            self._clear_prepared_kvc_evict_selection_schedule()
            return False
        self._prepared_kvc_evict_schedule = schedule
        self._prepared_kvc_evict_schedule_signature = signature
        self._prepare_kvc_evict_host_slots(schedule[0][0], signature)
        self.stats.kvc_evict_schedule_build_count += 1
        self.stats.kvc_evict_schedule_step_count += len(schedule)
        self.stats.kvc_evict_schedule_token_count += sum(
            int(item[-1]) for item in schedule
        )
        self.stats.kvc_evict_schedule_build_ms += elapsed_ms
        self.stats.kvc_evict_preselect_count += 1
        self.stats.kvc_evict_preselect_token_count += int(schedule[0][-1])
        self.stats.kvc_evict_preselect_ms += elapsed_ms
        return True

    def _consume_prepared_kvc_evict_selection_schedule(
        self, forward_batch: Any, need_tokens: int
    ) -> Optional[List[_LayerKVResidencyEntry]]:
        with self._profile("profile_kvc_evict_schedule_consume_ms"):
            if not self._prepared_kvc_evict_schedule:
                return None
            need_tokens = self._align_tokens_down(max(0, int(need_tokens)))
            if need_tokens <= 0:
                return None
            selected, cursors, cursors_by_layer, token_count = (
                self._prepared_kvc_evict_schedule[0]
            )
            if int(token_count) > int(need_tokens):
                self.stats.kvc_evict_schedule_invalid_count += 1
                self._clear_prepared_kvc_evict_selection_schedule()
                return None
            with self._profile("profile_kvc_evict_schedule_active_lens_ms"):
                active_lens = {
                    int(req_idx): int(seq_len)
                    for req_idx, seq_len in self._batch_req_indices_and_lens(
                        forward_batch
                    )
                }
            if not active_lens:
                self.stats.kvc_evict_schedule_invalid_count += 1
                self._clear_prepared_kvc_evict_selection_schedule()
                return None

            by_layer: Dict[int, int] = {}
            with self._profile("profile_kvc_evict_schedule_entry_check_ms"):
                for entry in selected:
                    req_idx = int(entry.req_idx)
                    seq_len = int(active_lens.get(req_idx, -1))
                    evictable_len = self._align_tokens_down(max(0, seq_len - 1))
                    if (
                        str(entry.state) not in ("resident", "prepared")
                        or not entry.device_loc_list()
                        or seq_len <= 0
                        or int(entry.pos) + int(entry.token_count) > int(evictable_len)
                    ):
                        self.stats.kvc_evict_schedule_invalid_count += 1
                        self._clear_prepared_kvc_evict_selection_schedule()
                        return None
                    layer_id = int(entry.layer_id)
                    by_layer[layer_id] = by_layer.get(layer_id, 0) + int(
                        entry.token_count
                    )

            with self._profile("profile_kvc_evict_schedule_capacity_check_ms"):
                scratch_capacity = self._per_layer_virtual_scratch_capacity_tokens()
                enforce_layer_targets = (
                    not self._requires_common_per_layer_kvc_eviction()
                )
                for layer_id, selected_tokens in by_layer.items():
                    current = int(
                        self._per_layer_offloaded_token_count_by_layer.get(
                            int(layer_id), 0
                        )
                    )
                    if (
                        scratch_capacity > 0
                        and current + int(selected_tokens) > scratch_capacity
                    ):
                        self.stats.kvc_evict_schedule_invalid_count += 1
                        self._clear_prepared_kvc_evict_selection_schedule()
                        return None
                    if enforce_layer_targets:
                        target = int(
                            self._planned_kvc_tokens_by_layer.get(int(layer_id), 0) or 0
                        )
                        if target > 0 and current + int(selected_tokens) > target:
                            self.stats.kvc_evict_schedule_invalid_count += 1
                            self._clear_prepared_kvc_evict_selection_schedule()
                            return None

            self._prepared_kvc_evict_schedule.pop(0)
            self._kvc_evict_cursors = dict(cursors)
            self._kvc_evict_cursors_by_layer = dict(cursors_by_layer)
            self.stats.kvc_evict_schedule_hit_count += 1
            self.stats.kvc_evict_preselect_hit_count += 1
            return list(selected)

    def _consume_prepared_kvc_evict_selection(
        self, forward_batch: Any, need_tokens: int
    ) -> Optional[List[_LayerKVResidencyEntry]]:
        with self._profile("profile_kvc_evict_prepared_consume_ms"):
            scheduled = self._consume_prepared_kvc_evict_selection_schedule(
                forward_batch, need_tokens
            )
            if scheduled is not None:
                return scheduled
            if not self._prepared_kvc_evict_entries:
                return None
            signature = self._kvc_evict_selection_signature(forward_batch, need_tokens)
            if self._prepared_kvc_evict_signature != signature:
                self.stats.kvc_evict_preselect_invalid_count += 1
                self._clear_prepared_kvc_evict_selection(clear_schedule=False)
                return None
            selected = list(self._prepared_kvc_evict_entries)
            prepared_cursors = dict(self._prepared_kvc_evict_cursors)
            prepared_cursors_by_layer = dict(self._prepared_kvc_evict_cursors_by_layer)
            self._prepared_kvc_evict_entries = []
            self._prepared_kvc_evict_signature = (0, 0, 0, 0)
            self._prepared_kvc_evict_cursors = {}
            self._prepared_kvc_evict_cursors_by_layer = {}
            for entry in selected:
                if (
                    str(entry.state) not in ("resident", "prepared")
                    or not entry.device_loc_list()
                ):
                    self.stats.kvc_evict_preselect_invalid_count += 1
                    self._clear_prepared_kvc_evict_host_slots()
                    return None
            selected_tokens = sum(int(entry.token_count) for entry in selected)
            if selected_tokens <= 0:
                self.stats.kvc_evict_preselect_invalid_count += 1
                self._clear_prepared_kvc_evict_host_slots()
                return None
            self._kvc_evict_cursors = prepared_cursors
            self._kvc_evict_cursors_by_layer = prepared_cursors_by_layer
            self.stats.kvc_evict_preselect_hit_count += 1
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
        global_remaining = self._align_tokens_down(max(0, int(max_tokens)))
        scratch_capacity = self._per_layer_virtual_scratch_capacity_tokens()
        offload_run_index_by_req_layer: Dict[Tuple[int, int], int] = {}

        def merge_segments(
            segments: List[Tuple[int, int, int]],
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
            run_key = (int(req_idx), int(layer_id))
            run_index = int(offload_run_index_by_req_layer.get(run_key, 0))
            run_count = len(runs)
            while run_index < run_count:
                run = runs[run_index]
                run_start = int(run.pos)
                run_end = run_start + int(run.token_count)
                if run_end <= pos:
                    run_index += 1
                    continue
                if run_start <= pos:
                    pos = self._align_tokens_up(run_end)
                    if pos >= end:
                        offload_run_index_by_req_layer[run_key] = run_index
                        return end, end
                    run_index += 1
                    continue
                offload_run_index_by_req_layer[run_key] = run_index
                return pos, min(end, self._align_tokens_down(run_start))
            offload_run_index_by_req_layer[run_key] = run_index
            return pos, end

        for layer_id, target_tokens in sorted(plan.items()):
            if global_remaining < block_size:
                break
            layer_id = int(layer_id)
            current = int(
                self._per_layer_offloaded_token_count_by_layer.get(layer_id, 0)
            )
            need_tokens = self._align_tokens_down(max(0, int(target_tokens) - current))
            need_tokens = min(int(need_tokens), int(global_remaining))
            need_tokens = self._align_tokens_down(int(need_tokens))
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
            round_offsets = [0 for _ in pair_meta]
            while remaining >= block_size:
                progressed = False
                for pair_idx, (req_idx, evictable_len, cursor) in enumerate(pair_meta):
                    offset = int(round_offsets[pair_idx])
                    raw_pos = cursor + offset
                    if raw_pos + block_size > evictable_len:
                        continue
                    pos, span_end = next_resident_span(
                        layer_id, req_idx, raw_pos, evictable_len
                    )
                    if pos + block_size > span_end:
                        round_offsets[pair_idx] = max(
                            offset + block_size, int(span_end) - int(cursor)
                        )
                        continue
                    span_tokens = self._align_tokens_down(
                        min(int(remaining), int(span_end) - int(pos))
                    )
                    if span_tokens < block_size:
                        round_offsets[pair_idx] = max(
                            offset + block_size, int(span_end) - int(cursor)
                        )
                        continue
                    key = (layer_id, req_idx, pos)
                    direct_entry = self._per_layer_residency.get(key)
                    existing = (
                        direct_entry
                        if direct_entry is not None
                        and self._per_layer_entry_is_current(direct_entry)
                        else None
                    )
                    if existing is None:
                        existing = self._per_layer_page_entry_for_span(
                            layer_id, req_idx, pos, span_tokens
                        )
                    self.stats.kvc_evict_selector_scanned_entries += 1
                    if (
                        existing is not None
                        and self._per_layer_entry_is_current(existing)
                        and existing.state
                        in (
                            "offloaded",
                            "reloading",
                            "evicting",
                        )
                    ):
                        round_offsets[pair_idx] = max(
                            offset + block_size, int(pos) + block_size - int(cursor)
                        )
                        continue
                    layer_segments.append((req_idx, pos, span_tokens))
                    remaining -= span_tokens
                    round_offsets[pair_idx] = int(pos) + int(span_tokens) - int(cursor)
                    progressed = True
                    self.stats.kvc_layerwise_evict_cursor_hit_count += 1
                    if remaining < block_size:
                        break
                if not progressed:
                    break
            if not layer_segments:
                continue
            for req_idx, pos, count in merge_segments(layer_segments):
                if global_remaining < block_size:
                    break
                count = min(int(count), int(global_remaining))
                count = self._align_tokens_down(int(count))
                if count < block_size:
                    break
                segment_entries = self._materialize_per_layer_resident_segment(
                    layer_id,
                    int(req_idx),
                    int(pos),
                    int(count),
                    import_arena=False,
                )
                self.stats.kvc_evict_selector_scanned_entries += 1
                if segment_entries is None or sum(
                    int(entry.token_count) for entry in segment_entries
                ) != int(count):
                    continue
                self.stats.kvc_evict_candidate_scan_tokens += int(count)
                for entry in segment_entries:
                    entry.last_access_step = self._decode_step
                    self._sync_kvc_group_if_needed(entry)
                selected.extend(segment_entries)
                selected_token_count = sum(
                    int(entry.token_count) for entry in segment_entries
                )
                global_remaining = max(
                    0, int(global_remaining) - int(selected_token_count)
                )
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
