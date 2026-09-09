"""LayerKV LayerKVKvcVirtualMixin implementation."""

from __future__ import annotations

import bisect
import contextlib
import dataclasses
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


class LayerKVKvcVirtualMixin:
    def _restore_virtual_scratch_for_kvc_use(self) -> None:
        """Make borrowed scratch pages resident before any KVC access."""
        controller = getattr(self, "_shared_expert", None)
        restore = getattr(controller, "restore_scratch", None)
        if callable(restore):
            restore()

    def _prepare_per_layer_kvc_attention(self, layer_id: int) -> None:
        if not self._uses_per_layer_attention_override():
            return
        layer_id = int(layer_id)
        if (
            self.config.kvc_backend == "per-layer-arena"
            and self._pending_kvc_evict_events
        ):
            finalize_key = (int(self._decode_step), layer_id)
            if finalize_key not in self._per_layer_kvc_evict_finalized_before_layers:
                self._per_layer_kvc_evict_finalized_before_layers.add(finalize_key)
                with self._profile("profile_kvc_layer_evict_finalize_ms"):
                    self._finalize_kvc_evictions_before_layer(layer_id, block=False)
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
        self._restore_virtual_scratch_for_kvc_use()
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
                    self._issue_next_per_layer_kvc_prefetch(layer_id)
                    return
            # The hook is active even before the arena allocator produces a
            # layer-specific mapping. Keep this cheap only when no recovery is
            # required for the current use point.
            self.stats.kvc_per_layer_metadata_rewrite_skip_count += 1
            self._issue_next_per_layer_kvc_prefetch(layer_id)
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
            self._issue_next_per_layer_kvc_prefetch(layer_id)

    def _issue_next_per_layer_kvc_prefetch(
        self, layer_id: int, *, allow_virtual_scratch: bool = False
    ) -> None:
        """Queue the next physical KVC layer while the current layer runs.

        The current layer's metadata hook is the first point at which its
        device locations are guaranteed to be prepared.  At that point the
        copy stream can safely enqueue the next layer's reload.  Attention for
        that next layer still waits on its own ready event at point of use.
        """
        if (
            self.config.kvc_backend != "per-layer-arena"
            or (
                self._per_layer_virtual_scratch_enabled()
                and not allow_virtual_scratch
            )
            or self.config.kvc_scheduler != "async-deadline"
            or not self._optimized_profile_enabled()
            or self._copy_stream is None
            or self._last_forward_batch is None
        ):
            return
        next_layer = self._next_kvc_layer_id(int(layer_id))
        if next_layer is None:
            return
        selected = [
            entry
            for entry in self._select_required_offloaded_per_layer_entries(
                self._last_forward_batch
            )
            if int(entry.layer_id) == int(next_layer)
        ]
        if not selected:
            return
        token_count = sum(int(entry.token_count) for entry in selected)
        if allow_virtual_scratch and not self._allow_physical_kvc_prefetch(
            int(layer_id), int(next_layer), int(token_count)
        ):
            self.stats.kvc_layer_prefetch_skip_count += 1
            return
        with self._profile("profile_kvc_layer_prefetch_ms"):
            issued = self._reload_required_kvc(
                self._last_forward_batch,
                selected_entries=selected,
                strict=False,
            )
        if issued:
            self.stats.kvc_layer_prefetch_issue_count += 1
            self.stats.kvc_layer_prefetch_layer_count += 1
            self.stats.kvc_layer_prefetch_token_count += int(token_count)
        else:
            self.stats.kvc_layer_prefetch_skip_count += 1

    def _allow_physical_kvc_prefetch(
        self, current_layer_id: int, next_layer_id: int, token_count: int
    ) -> bool:
        """Bound the single-scratch physical fallback by measured copy state.

        A fallback reload is useful only while the copy stream has a small
        lookahead window.  Keep at most two unconsumed reload events (the
        current recovery plus one successor).  Once layer reload timing has
        been observed, reject a successor whose measured H2D estimate cannot
        fit in an observed per-layer compute window.  Cold-start calls remain
        eligible because rejecting them would prevent the runtime from ever
        calibrating the gate.
        """
        # The first eligible call is deliberately allowed to calibrate a
        # lightweight model-forward window on later decode steps. This is
        # separate from profile_detail and does not add a CUDA sync.
        self._kvc_prefetch_window_measurement_enabled = True
        pending = sum(
            1
            for pending_reload in self._pending_kvc_reload_events
            if not getattr(pending_reload, "waited_on_main_stream", False)
        )
        if pending >= 2:
            self.stats.kvc_layer_prefetch_pending_cap_skip_count += 1
            return False

        estimated_ms = self._estimate_per_layer_kvc_prefetch_ms(
            int(next_layer_id), int(token_count)
        )
        overlap_ms = self._estimate_profiled_kvc_layer_window_ms(
            int(current_layer_id), int(next_layer_id)
        )
        self.stats.kvc_layer_prefetch_estimated_copy_ms = float(
            estimated_ms or 0.0
        )
        self.stats.kvc_layer_prefetch_overlap_window_ms = float(overlap_ms or 0.0)
        if estimated_ms is None or overlap_ms is None:
            self.stats.kvc_layer_prefetch_unmeasured_allow_count += 1
            return True
        if float(estimated_ms) > float(overlap_ms) + 0.05:
            self.stats.kvc_layer_prefetch_deadline_reject_count += 1
            return False
        return True

    def _estimate_per_layer_kvc_prefetch_ms(
        self, layer_id: int, token_count: int
    ) -> Optional[float]:
        if int(token_count) <= 0:
            return 0.0
        bytes_per_token = int(self._bytes_per_kvc_token_per_layer())
        if bytes_per_token <= 0:
            return None
        ewma = getattr(self, "_kvc_reload_ms_per_mb_ewma_by_layer", {}).get(
            int(layer_id)
        )
        if ewma is None:
            aggregate = float(
                getattr(self.stats, "kvc_layerwise_reload_ewma_ms_per_mb", 0.0)
                or 0.0
            )
            ewma = aggregate if aggregate > 0.0 else None
        if ewma is None or not math.isfinite(float(ewma)) or float(ewma) <= 0.0:
            return None
        mb = float(int(token_count) * bytes_per_token) / float(1024 * 1024)
        return 0.03 + mb * float(ewma)

    def _estimate_profiled_kvc_layer_window_ms(
        self, current_layer_id: int, next_layer_id: int
    ) -> Optional[float]:
        del current_layer_id, next_layer_id
        model_ms = float(
            getattr(self.stats, "kvc_layer_prefetch_model_forward_ms", 0.0) or 0.0
        )
        forward_count = int(
            getattr(self.stats, "kvc_layer_prefetch_model_forward_count", 0) or 0
        )
        # Keep detailed-profile artifacts usable as a compatibility fallback.
        if model_ms <= 0.0 or forward_count <= 0:
            model_ms = float(
                getattr(self.stats, "profile_decode_model_forward_ms", 0.0) or 0.0
            )
            forward_count = int(
                getattr(self.stats, "profile_decode_batch_count", 0) or 0
            )
        layer_count = len(self._kvc_layer_ids())
        if model_ms <= 0.0 or forward_count <= 0 or layer_count <= 0:
            return None
        return max(0.0, model_ms / float(forward_count) / float(layer_count))

    def _prepare_virtual_kvc_attention(
        self, layer_id: int, *, guard: bool = True
    ) -> None:
        if not self._residency and not (
            self.config.kvc_backend == "per-layer-arena" and self._per_layer_residency
        ):
            return
        if self._last_forward_batch is None or self._runner is None:
            return
        self._restore_virtual_scratch_for_kvc_use()
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
                # The overflow path is a physical per-layer reload even when
                # the backend also owns virtual scratch. Use the compute
                # window before the next layer to submit its physical reload.
                self._issue_next_per_layer_kvc_prefetch(
                    layer_id, allow_virtual_scratch=True
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
                self._store_virtual_scratch_cache(
                    demand,
                    scratch_locs,
                    buffer_idx,
                    buffer_generation=int(pending.buffer_generation),
                )
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
        if buffer_idx < 0 and self.config.kvc_backend == "per-layer-arena":
            # A single scratch buffer cannot overlap the current materialize
            # with the next layer.  The physical per-layer arena is still a
            # valid destination, so use it as the bounded fallback window.
            # This keeps the current layer's scratch mapping unchanged while
            # allowing the next layer's H2D to overlap its compute window.
            self._issue_next_per_layer_kvc_prefetch(
                layer_id, allow_virtual_scratch=True
            )
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
        generation_mismatch = False
        if entry is not None:
            generation_mismatch = int(entry.buffer_generation) != int(
                self._virtual_scratch_generation(
                    int(entry.buffer_idx), layer_id=int(demand.layer_id)
                )
            )
        if (
            entry is None
            or generation_mismatch
            or int(entry.token_count) != int(demand.token_count)
            or (
                entry.host_signature != demand.host_signature
                if demand.host_signature != (0, 0, 0, 0)
                else entry.host_slots != demand.host_slots
            )
            or (demand.host_slice[0] >= 0 and entry.host_slice != demand.host_slice)
            or (
                demand.host_slots == ()
                and demand.host_slice[0] < 0
                and demand.host_index_cpu is not None
                and (
                    entry.host_index_cpu is None
                    or not bool(
                        torch.equal(entry.host_index_cpu, demand.host_index_cpu)
                    )
                )
            )
            or entry.scratch_locs is None
            or int(entry.scratch_locs.numel()) != int(demand.token_count)
        ):
            if record_stats:
                self.stats.virtual_kvc_persistent_cache_miss_count += 1
                if generation_mismatch:
                    self.stats.virtual_kvc_scratch_generation_miss_count += 1
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
        buffer_generation: Optional[int] = None,
    ) -> None:
        previous = self._virtual_scratch_cache_by_layer.get(int(demand.layer_id))
        previous_same = False
        if previous is not None:
            previous_same = (
                previous.host_signature == demand.host_signature
                if demand.host_signature != (0, 0, 0, 0)
                else previous.host_slots == demand.host_slots
            )
            if previous_same and demand.host_slice[0] >= 0:
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
                buffer_generation=(
                    int(buffer_generation)
                    if buffer_generation is not None
                    else self._virtual_scratch_generation(
                        int(buffer_idx), layer_id=int(demand.layer_id)
                    )
                ),
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
        self._virtual_scratch_buffer_generations_by_layer.clear()
        self._virtual_materialize_recorded_event_groups.clear()
        self._virtual_materialize_plan = None
        self._virtual_materialize_plans_by_layer.clear()
        self._virtual_materialize_plan_reuse_cache.clear()
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
            for key in list(self._virtual_scratch_buffer_generations_by_layer):
                if int(key[0]) == layer_id:
                    self._virtual_scratch_buffer_generations_by_layer.pop(key, None)
            self._virtual_materialize_plans_by_layer.pop(layer_id, None)
            for key in list(self._virtual_materialize_plan_reuse_cache):
                if key and int(key[0]) == layer_id:
                    self._virtual_materialize_plan_reuse_cache.pop(key, None)
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
            host_spans=plan.host_spans,
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

    def _virtual_batch_index_context(
        self,
    ) -> Tuple[
        List[Tuple[int, int]],
        Dict[int, int],
        Dict[int, int],
        Dict[int, int],
    ]:
        if (
            int(self._virtual_batch_index_step) == int(self._decode_step)
            and self._virtual_batch_index_cache is not None
        ):
            return self._virtual_batch_index_cache
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
        cache = (batch_req_lens, active_lens, row_by_req, flat_base_by_req)
        self._virtual_batch_index_cache = cache
        self._virtual_batch_index_step = int(self._decode_step)
        return cache

    def _build_virtual_materialize_plan(
        self, *, layer_id: Optional[int]
    ) -> Optional[_LayerKVVirtualMaterializePlan]:
        table = getattr(self._req_to_token_pool, "req_to_token", None)
        if table is None:
            return None
        target_layer = None if layer_id is None else int(layer_id)
        direct_kv_indices_only = target_layer is not None
        metadata_kind = self._virtual_metadata_kind() if direct_kv_indices_only else ""
        with self._profile("profile_virtual_select_ms"):
            (
                batch_req_lens,
                active_lens,
                row_by_req,
                flat_base_by_req,
            ) = self._virtual_batch_index_context()
            if (
                self.config.kvc_backend == "per-layer-arena"
                and target_layer is not None
            ):
                selected = self._select_offloaded_run_entries_for_virtual_layer(
                    int(target_layer), active_lens
                )
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
                host_spans=(),
                host_index_cpu=None,
                max_row_index=-1,
                max_position=-1,
                max_flat_index=-1,
                token_count=0,
            )
        token_count = sum(int(entry.token_count) for entry in selected)
        host_ordered_plan = False
        plan_cache_key = None
        if direct_kv_indices_only:
            with self._profile("profile_virtual_plan_cache_lookup_ms"):
                t0 = time.perf_counter()
                selected.sort(key=lambda entry: (entry.req_idx, entry.pos))
                self._add_profile(
                    "profile_virtual_plan_sort_ms",
                    (time.perf_counter() - t0) * 1000.0,
                )
                t0 = time.perf_counter()
                host_ordered = self._host_ordered_virtual_entries(
                    selected, token_count=token_count
                )
                self._add_profile(
                    "profile_virtual_host_order_ms",
                    (time.perf_counter() - t0) * 1000.0,
                )
                if host_ordered is not None:
                    selected = host_ordered
                    host_ordered_plan = True
                t0 = time.perf_counter()
                plan_cache_key = self._virtual_materialize_plan_cache_key(
                    int(target_layer),
                    selected,
                    row_by_req,
                    flat_base_by_req,
                    metadata_kind,
                    token_count=token_count,
                )
                self._add_profile(
                    "profile_virtual_plan_key_ms",
                    (time.perf_counter() - t0) * 1000.0,
                )
                t0 = time.perf_counter()
                cached_plan = self._virtual_materialize_plan_reuse_cache.get(
                    plan_cache_key
                )
                self._add_profile(
                    "profile_virtual_plan_lookup_only_ms",
                    (time.perf_counter() - t0) * 1000.0,
                )
            if cached_plan is not None:
                self.stats.virtual_kvc_plan_cache_hit_count += 1
                self.stats.virtual_kvc_plan_cache_reuse_token_count += int(
                    cached_plan.token_count
                )
                return dataclasses.replace(
                    cached_plan,
                    step=int(self._decode_step),
                    selected=selected,
                )
            self.stats.virtual_kvc_plan_cache_miss_count += 1
        else:
            selected.sort(key=lambda entry: (entry.req_idx, entry.pos))
        with self._profile("profile_virtual_index_build_ms"):
            if host_ordered_plan:
                self.stats.virtual_kvc_host_order_plan_count += 1
                self.stats.virtual_kvc_host_order_plan_token_count += int(token_count)
            build_flat_layout = (
                not direct_kv_indices_only or metadata_kind != "page_table"
            )
            build_row_layout = (
                not direct_kv_indices_only or metadata_kind != "kv_indices"
            )
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
                token_count_for_entry = int(entry.token_count)
                pos_start = int(entry.pos)
                base = int(flat_base_by_req[req_idx])
                row = int(row_by_req[req_idx])
                if direct_kv_indices_only:
                    run_len = int(token_count_for_entry)
                    if build_flat_layout:
                        flat_start = base + pos_start
                        if span_len > 0 and flat_start == span_start + span_len:
                            span_len += run_len
                        else:
                            if span_len > 0:
                                flat_spans.append(
                                    (span_start, span_scratch_start, span_len)
                                )
                            span_start = int(flat_start)
                            span_scratch_start = int(scratch_offset)
                            span_len = int(run_len)
                    if (
                        build_row_layout
                        and row_span_len > 0
                        and row == row_span_row
                        and pos_start == row_span_start + row_span_len
                    ):
                        row_span_len += run_len
                    elif build_row_layout:
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
                        row_span_start = int(pos_start)
                        row_span_scratch_start = int(scratch_offset)
                        row_span_len = int(run_len)
                    scratch_offset += run_len
                    continue
                for pos in range(pos_start, pos_start + token_count_for_entry):
                    flat_index = base + int(pos)
                    flat_indices.append(flat_index)
                if not direct_kv_indices_only:
                    req_indices.extend([req_idx] * token_count_for_entry)
                    positions.extend(
                        range(pos_start, pos_start + token_count_for_entry)
                    )
                    row_indices.extend([row] * token_count_for_entry)
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
            if direct_kv_indices_only:
                host_slots: List[int] = []
                (
                    host_signature,
                    host_slice,
                    host_spans,
                    host_index_cpu,
                ) = self._direct_host_layout_for_entries(
                    selected,
                    token_count=int(token_count),
                )
            else:
                host_slots = []
                for entry in selected:
                    if entry.host_slots is not None:
                        host_slots.extend(entry.host_slots)
                    elif entry.host_slot is not None:
                        host_slots.append(int(entry.host_slot))
                host_signature = self._host_slot_signature(host_slots)
                host_slice = self._contiguous_host_slice(host_slots)
                host_spans = (
                    ()
                    if host_slice[0] >= 0
                    else self._contiguous_host_spans(host_slots)
                )
                # Very fragmented spans create many small H2D/index_copy operations;
                # keep the original gather path in that case.
                if len(host_spans) > 64:
                    host_spans = ()
                host_index_cpu = (
                    None
                    if host_slice[0] >= 0 or host_spans
                    else torch.tensor(host_slots, dtype=torch.int64, device="cpu")
                )
            max_row_index = max(row_indices) if row_indices else -1
            max_position = max(positions) if positions else -1
            if row_spans:
                max_row_index = max(
                    max_row_index,
                    max(row for row, _start, _offset, _length in row_spans),
                )
                max_position = max(
                    max_position,
                    max(
                        start + length - 1 for _row, start, _offset, length in row_spans
                    ),
                )
            if flat_indices:
                max_flat_index = max(flat_indices)
            elif flat_spans:
                max_flat_index = max(
                    start + length - 1 for start, _offset, length in flat_spans
                )
            else:
                max_flat_index = -1
        plan = _LayerKVVirtualMaterializePlan(
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
            host_slots=(
                () if direct_kv_indices_only else tuple(int(x) for x in host_slots)
            ),
            host_signature=host_signature,
            host_slice=host_slice,
            host_spans=host_spans,
            host_index_cpu=host_index_cpu,
            max_row_index=int(max_row_index),
            max_position=int(max_position),
            max_flat_index=int(max_flat_index),
            token_count=int(token_count),
        )
        if plan_cache_key is not None:
            self._virtual_materialize_plan_reuse_cache[plan_cache_key] = plan
        return plan

    def _virtual_metadata_kind(self) -> str:
        attn_backend = self._kvc_attention_backend()
        metadata = getattr(attn_backend, "forward_metadata", None)
        if metadata is None:
            return "unknown"
        if getattr(metadata, "kv_indices", None) is not None:
            return "kv_indices"
        if getattr(metadata, "page_table", None) is not None:
            return "page_table"
        return "unknown"

    def _virtual_materialize_plan_cache_key(
        self,
        layer_id: int,
        selected: List[_LayerKVResidencyEntry],
        row_by_req: Dict[int, int],
        flat_base_by_req: Dict[int, int],
        metadata_kind: str,
        *,
        token_count: Optional[int] = None,
    ) -> Tuple[Any, ...]:
        first = selected[0]
        last = selected[-1]
        token_count = (
            int(token_count)
            if token_count is not None
            else sum(int(entry.token_count) for entry in selected)
        )
        reqs = tuple(sorted({int(entry.req_idx) for entry in selected}))
        row_sig = tuple((req_idx, int(row_by_req.get(req_idx, -1))) for req_idx in reqs)
        if metadata_kind == "page_table":
            flat_sig: Tuple[Tuple[int, int], ...] = ()
        else:
            flat_sig = tuple(
                (req_idx, int(flat_base_by_req.get(req_idx, -1))) for req_idx in reqs
            )
        return (
            int(layer_id),
            str(metadata_kind),
            int(
                getattr(self, "_per_layer_offloaded_version_by_layer", {}).get(
                    int(layer_id), 0
                )
            ),
            len(selected),
            int(token_count),
            (
                int(first.req_idx),
                int(first.pos),
                int(first.token_count),
            ),
            (
                int(last.req_idx),
                int(last.pos),
                int(last.token_count),
            ),
            row_sig,
            flat_sig,
        )

    def _virtual_metadata_uses_kv_indices(self) -> bool:
        attn_backend = self._kvc_attention_backend()
        metadata = getattr(attn_backend, "forward_metadata", None)
        if metadata is None:
            return False
        return getattr(metadata, "kv_indices", None) is not None

    @staticmethod
    def _host_slot_signature(host_slots: List[int]) -> Tuple[int, int, int, int]:
        if not host_slots:
            return (0, 0, 0, 0)
        checksum = int(sum(host_slots)) & 0x7FFFFFFF
        return (
            len(host_slots),
            int(host_slots[0]),
            int(host_slots[-1]),
            int(checksum),
        )

    @classmethod
    def _direct_host_layout_for_entries(
        cls,
        entries: List[_LayerKVResidencyEntry],
        *,
        token_count: int,
    ) -> Tuple[
        Tuple[int, int, int, int],
        Tuple[int, int],
        Tuple[Tuple[int, int, int], ...],
        Optional[torch.Tensor],
    ]:
        if not entries or int(token_count) <= 0:
            return (0, 0, 0, 0), (-1, 0), (), None
        spans: List[Tuple[int, int, int]] = []
        host_slots_fallback: Optional[List[int]] = None
        expected_host: Optional[int] = None
        first_host: Optional[int] = None
        last_host: Optional[int] = None
        scratch_offset = 0
        checksum = 0
        globally_contiguous = True
        for entry in entries:
            count = int(entry.token_count)
            if count <= 0:
                continue
            if entry.host_slots is not None:
                slots = entry.host_slots
                if (
                    len(slots) != count
                    or not slots
                    or int(slots[-1]) - int(slots[0]) + 1 != count
                ):
                    if host_slots_fallback is None:
                        host_slots_fallback = []
                        for prev_start, _prev_offset, prev_len in spans:
                            host_slots_fallback.extend(
                                range(int(prev_start), int(prev_start) + int(prev_len))
                            )
                    host_slots_fallback.extend(int(slot) for slot in slots)
                    scratch_offset += count
                    globally_contiguous = False
                    continue
                start = int(slots[0])
            elif entry.host_slot is not None and count == 1:
                start = int(entry.host_slot)
            else:
                if host_slots_fallback is None:
                    host_slots_fallback = []
                    for prev_start, _prev_offset, prev_len in spans:
                        host_slots_fallback.extend(
                            range(int(prev_start), int(prev_start) + int(prev_len))
                        )
                host_slots_fallback.extend(entry.host_slot_list())
                scratch_offset += count
                globally_contiguous = False
                continue
            if first_host is None:
                first_host = start
            last_host = start + count - 1
            checksum += (start + last_host) * count // 2
            if expected_host is not None and start != expected_host:
                globally_contiguous = False
            expected_host = start + count
            if spans and start == spans[-1][0] + spans[-1][2]:
                prev_start, prev_offset, prev_len = spans[-1]
                spans[-1] = (prev_start, prev_offset, prev_len + count)
            else:
                spans.append((start, scratch_offset, count))
            if host_slots_fallback is not None:
                host_slots_fallback.extend(range(start, start + count))
            scratch_offset += count
        if scratch_offset != int(token_count):
            return (0, 0, 0, 0), (-1, 0), (), None
        if host_slots_fallback is not None:
            if len(host_slots_fallback) != int(token_count):
                return (0, 0, 0, 0), (-1, 0), (), None
            signature = cls._host_slot_signature(host_slots_fallback)
            host_slice = cls._contiguous_host_slice(host_slots_fallback)
            host_spans = (
                ()
                if host_slice[0] >= 0
                else cls._contiguous_host_spans(host_slots_fallback)
            )
            if len(host_spans) > 64:
                return (
                    signature,
                    (-1, 0),
                    (),
                    torch.tensor(host_slots_fallback, dtype=torch.int64, device="cpu"),
                )
            return signature, host_slice, host_spans, None
        if first_host is None or last_host is None:
            return (0, 0, 0, 0), (-1, 0), (), None
        signature = (
            int(token_count),
            int(first_host),
            int(last_host),
            int(checksum) & 0x7FFFFFFF,
        )
        if globally_contiguous and first_host + int(token_count) - 1 == last_host:
            return signature, (int(first_host), int(token_count)), (), None
        host_spans = tuple(
            (int(start), int(offset), int(length))
            for start, offset, length in spans
            if int(length) > 0
        )
        if len(host_spans) > 64:
            host_slots = []
            for start, _offset, length in host_spans:
                host_slots.extend(range(int(start), int(start) + int(length)))
            return (
                signature,
                (-1, 0),
                (),
                torch.tensor(host_slots, dtype=torch.int64, device="cpu"),
            )
        return signature, (-1, 0), host_spans, None

    @staticmethod
    def _contiguous_host_slice(host_slots: List[int]) -> Tuple[int, int]:
        if not host_slots:
            return (-1, 0)
        first = int(host_slots[0])
        last = int(host_slots[-1])
        if last - first + 1 != len(host_slots):
            return (-1, 0)
        for offset, slot in enumerate(host_slots):
            if int(slot) != first + int(offset):
                return (-1, 0)
        return (first, len(host_slots))

    @staticmethod
    def _entry_first_host_slot(entry: _LayerKVResidencyEntry) -> int:
        if entry.host_slots is not None:
            if not entry.host_slots:
                return 1 << 60
            return int(entry.host_slots[0])
        if entry.host_slot is None:
            return 1 << 60
        return int(entry.host_slot)

    @staticmethod
    def _entry_host_slots(entries: List[_LayerKVResidencyEntry]) -> List[int]:
        host_slots: List[int] = []
        for entry in entries:
            if entry.host_slots is not None:
                host_slots.extend(int(slot) for slot in entry.host_slots)
            elif entry.host_slot is not None:
                host_slots.append(int(entry.host_slot))
        return host_slots

    @classmethod
    def _contiguous_host_slice_for_entries(
        cls,
        entries: List[_LayerKVResidencyEntry],
        *,
        expected_tokens: int,
    ) -> Tuple[int, int]:
        if not entries or int(expected_tokens) <= 0:
            return (-1, 0)
        first_slot: Optional[int] = None
        expected_slot: Optional[int] = None
        seen = 0
        for entry in entries:
            if entry.host_slots is not None:
                slots = entry.host_slots
                if not slots:
                    return (-1, 0)
                slice_start, slice_len = cls._contiguous_host_slice(slots)
                if slice_start < 0:
                    return (-1, 0)
                current_first = int(slice_start)
                current_len = int(slice_len)
            elif entry.host_slot is not None:
                current_first = int(entry.host_slot)
                current_len = 1
            else:
                return (-1, 0)
            if first_slot is None:
                first_slot = current_first
                expected_slot = current_first
            if expected_slot is None or current_first != expected_slot:
                return (-1, 0)
            seen += current_len
            expected_slot = current_first + current_len
        if seen != int(expected_tokens) or first_slot is None:
            return (-1, 0)
        return (int(first_slot), int(seen))

    def _host_ordered_virtual_entries(
        self,
        selected: List[_LayerKVResidencyEntry],
        *,
        token_count: Optional[int] = None,
    ) -> Optional[List[_LayerKVResidencyEntry]]:
        if len(selected) <= 1:
            return None
        expected_tokens = (
            int(token_count)
            if token_count is not None
            else sum(int(entry.token_count) for entry in selected)
        )
        if (
            self._contiguous_host_slice_for_entries(
                selected, expected_tokens=expected_tokens
            )[0]
            >= 0
        ):
            return None
        host_ordered = sorted(
            selected,
            key=lambda entry: (
                self._entry_first_host_slot(entry),
                int(entry.req_idx),
                int(entry.pos),
            ),
        )
        host_slice = self._contiguous_host_slice_for_entries(
            host_ordered, expected_tokens=expected_tokens
        )
        if host_slice[0] < 0:
            return None
        return host_ordered

    @staticmethod
    def _contiguous_host_spans(
        host_slots: List[int],
    ) -> Tuple[Tuple[int, int, int], ...]:
        if not host_slots:
            return ()
        spans: List[Tuple[int, int, int]] = []
        start_slot = int(host_slots[0])
        scratch_start = 0
        length = 1
        prev_slot = start_slot
        for offset, slot in enumerate(host_slots[1:], start=1):
            slot = int(slot)
            if slot == prev_slot + 1:
                length += 1
            else:
                spans.append((start_slot, scratch_start, length))
                start_slot = slot
                scratch_start = int(offset)
                length = 1
            prev_slot = slot
        spans.append((start_slot, scratch_start, length))
        if (
            len(spans) == 1
            and int(spans[0][1]) == 0
            and int(spans[0][2]) == len(host_slots)
        ):
            return ()
        return tuple(spans)

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

    def _virtual_scratch_generation(
        self, buffer_idx: int, *, layer_id: Optional[int] = None
    ) -> int:
        buffer_idx = int(buffer_idx)
        if (
            layer_id is not None
            and self.config.kvc_backend == "per-layer-arena"
            and buffer_idx >= 0
        ):
            return int(
                self._virtual_scratch_buffer_generations_by_layer.get(
                    (int(layer_id), buffer_idx), 0
                )
            )
        if buffer_idx >= 0 and buffer_idx < len(
            self._virtual_scratch_buffer_generations
        ):
            return int(self._virtual_scratch_buffer_generations[buffer_idx])
        return int(self._virtual_scratch_single_generation)

    def _bump_virtual_scratch_generation(
        self, buffer_idx: int, *, layer_id: Optional[int] = None
    ) -> int:
        buffer_idx = int(buffer_idx)
        if (
            layer_id is not None
            and self.config.kvc_backend == "per-layer-arena"
            and buffer_idx >= 0
        ):
            key = (int(layer_id), buffer_idx)
            generation = (
                int(self._virtual_scratch_buffer_generations_by_layer.get(key, 0)) + 1
            )
            self._virtual_scratch_buffer_generations_by_layer[key] = generation
            return generation
        if buffer_idx >= 0 and buffer_idx < len(
            self._virtual_scratch_buffer_generations
        ):
            self._virtual_scratch_buffer_generations[buffer_idx] = (
                int(self._virtual_scratch_buffer_generations[buffer_idx]) + 1
            )
            return int(self._virtual_scratch_buffer_generations[buffer_idx])
        self._virtual_scratch_single_generation += 1
        for idx in range(len(self._virtual_scratch_buffer_generations)):
            self._virtual_scratch_buffer_generations[idx] = (
                int(self._virtual_scratch_buffer_generations[idx]) + 1
            )
        return int(self._virtual_scratch_single_generation)

    def _materialize_virtual_kvc_sync(
        self,
        demand: _LayerKVKvcDemand,
        *,
        buffer_idx: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        self._restore_virtual_scratch_for_kvc_use()
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
            self._bump_virtual_scratch_generation(-1, layer_id=int(demand.layer_id))
            device_slice = (
                self._virtual_scratch_buffer_slices[0]
                if self._virtual_scratch_buffer_slices
                else (-1, 0)
            )
        else:
            buffer_idx = int(buffer_idx)
            scratch_locs = self._virtual_scratch_buffers[buffer_idx][:token_count]
            self._bump_virtual_scratch_generation(
                buffer_idx, layer_id=int(demand.layer_id)
            )
            device_slice = (
                self._virtual_scratch_buffer_slices[buffer_idx]
                if buffer_idx < len(self._virtual_scratch_buffer_slices)
                else (-1, 0)
            )
        self._ensure_host_store()
        elapsed_ms, _start_event, _ready_event = self._host_store.reload_layer_to_locs(
            int(demand.layer_id),
            demand.entries,
            scratch_locs,
            host_index=demand.host_index_cpu,
            host_slice=demand.host_slice,
            host_spans=demand.host_spans,
            device_slice=device_slice,
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
        if self.config.kvc_backend == "per-layer-arena":
            # A full per-layer req_to_token clone is too expensive on the
            # decode path.  If direct metadata patching is unavailable, recover
            # the demand into stable arena slots and rely on the existing
            # canonical mapping when the original locations are still free.
            with self._profile("profile_kvc_reload_required_ms"):
                reloaded = self._reload_required_kvc(
                    self._last_forward_batch,
                    selected_entries=list(demand.entries),
                    strict=False,
                )
            if not reloaded:
                self.stats.kvc_per_layer_metadata_rewrite_unsupported_count += 1
                return
            table = self._per_layer_req_to_token_overrides.get(int(layer_id))
            if table is not None:
                with self._profile("profile_virtual_metadata_rewrite_ms"):
                    ok = self._rewrite_attention_metadata_for_layer(layer_id, table)
                if ok:
                    self.stats.kvc_per_layer_metadata_rewrite_count += 1
                else:
                    self.stats.kvc_per_layer_metadata_rewrite_unsupported_count += 1
            return
        else:
            table = getattr(self._req_to_token_pool, "req_to_token", None)
            if table is None:
                return
        req_indices = demand.req_indices
        positions = demand.positions
        if not req_indices or not positions:
            req_indices, positions = self._build_virtual_scatter_indices(demand.entries)
            if not req_indices or not positions:
                self.stats.kvc_per_layer_metadata_rewrite_unsupported_count += 1
                return
        with self._profile("profile_virtual_req_to_token_scatter_ms"):
            req_tensor = torch.tensor(
                req_indices, dtype=torch.int64, device=table.device
            )
            pos_tensor = torch.tensor(positions, dtype=torch.int64, device=table.device)
            table[req_tensor, pos_tensor] = scratch_locs.to(
                dtype=table.dtype, device=table.device
            )
        if self.config.kvc_backend == "per-layer-arena":
            self._per_layer_req_to_token_versions[int(layer_id)] = (
                int(self._per_layer_req_to_token_versions.get(int(layer_id), 0)) + 1
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
        attn_backend = self._kvc_attention_backend()
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
                    or int(cache_entry.scratch_tensor.numel())
                    != int(scratch_locs.numel())
                    or int(cache_entry.scratch_tensor.data_ptr())
                    != int(scratch_locs.data_ptr())
                ):
                    cache_entry.scratch_tensor = scratch_locs.to(
                        device=kv_indices.device, dtype=kv_indices.dtype
                    )
                if demand.flat_spans:
                    if (
                        len(demand.flat_spans) > 16
                        and demand.flat_indices
                        and len(demand.flat_indices) == int(demand.token_count)
                    ):
                        if cache_entry.flat_tensor is None or (
                            cache_entry.flat_tensor.device != kv_indices.device
                        ):
                            cache_entry.flat_tensor = torch.tensor(
                                demand.flat_indices,
                                dtype=torch.int64,
                                device=kv_indices.device,
                            )
                        kv_indices[cache_entry.flat_tensor] = cache_entry.scratch_tensor
                        self.stats.metadata_patch_slice_count += 1
                        self.stats.metadata_patch_slice_token_count += int(
                            demand.token_count
                        )
                        return True
                    if len(demand.flat_spans) > 16:
                        if cache_entry.flat_span_tensor is None or (
                            cache_entry.flat_span_tensor.device != kv_indices.device
                        ):
                            (
                                cache_entry.flat_span_tensor,
                                cache_entry.flat_span_scratch_tensor,
                            ) = self._metadata_flat_span_scatter_tensors(
                                demand.flat_spans,
                                int(demand.token_count),
                                kv_indices.device,
                            )
                        kv_indices[cache_entry.flat_span_tensor] = (
                            cache_entry.scratch_tensor[
                                cache_entry.flat_span_scratch_tensor
                            ]
                        )
                        self.stats.metadata_patch_slice_count += 1
                        self.stats.metadata_patch_slice_token_count += int(
                            demand.token_count
                        )
                        return True
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
                    or int(cache_entry.scratch_tensor.numel())
                    != int(scratch_locs.numel())
                    or int(cache_entry.scratch_tensor.data_ptr())
                    != int(scratch_locs.data_ptr())
                ):
                    cache_entry.scratch_tensor = scratch_locs.to(
                        device=page_table.device, dtype=page_table.dtype
                    )
                if demand.row_spans:
                    if (
                        len(demand.row_spans) > 16
                        and demand.row_indices
                        and demand.positions
                        and len(demand.row_indices) == int(demand.token_count)
                        and len(demand.positions) == int(demand.token_count)
                    ):
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
                        self.stats.metadata_patch_slice_count += 1
                        self.stats.metadata_patch_slice_token_count += int(
                            demand.token_count
                        )
                        return True
                    if len(demand.row_spans) > 16:
                        if cache_entry.row_span_row_tensor is None or (
                            cache_entry.row_span_row_tensor.device != page_table.device
                        ):
                            (
                                cache_entry.row_span_row_tensor,
                                cache_entry.row_span_pos_tensor,
                                cache_entry.row_span_scratch_tensor,
                            ) = self._metadata_row_span_scatter_tensors(
                                demand.row_spans,
                                int(demand.token_count),
                                page_table.device,
                            )
                        page_table[
                            cache_entry.row_span_row_tensor,
                            cache_entry.row_span_pos_tensor,
                        ] = cache_entry.scratch_tensor[
                            cache_entry.row_span_scratch_tensor
                        ]
                        self.stats.metadata_patch_slice_count += 1
                        self.stats.metadata_patch_slice_token_count += int(
                            demand.token_count
                        )
                        return True
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

    @staticmethod
    def _metadata_flat_span_scatter_tensors(
        spans: Tuple[Tuple[int, int, int], ...],
        token_count: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        starts = torch.tensor(
            [int(start) for start, _offset, _length in spans],
            dtype=torch.int64,
            device=device,
        )
        scratch_starts = torch.tensor(
            [int(offset) for _start, offset, _length in spans],
            dtype=torch.int64,
            device=device,
        )
        lengths = torch.tensor(
            [int(length) for _start, _offset, length in spans],
            dtype=torch.int64,
            device=device,
        )
        span_ids = torch.repeat_interleave(
            torch.arange(int(lengths.numel()), dtype=torch.int64, device=device),
            lengths,
        )
        span_bases = torch.repeat_interleave(
            torch.cumsum(lengths, dim=0) - lengths, lengths
        )
        offsets = torch.arange(int(token_count), dtype=torch.int64, device=device)
        local_offsets = offsets - span_bases
        return (
            starts[span_ids] + local_offsets,
            scratch_starts[span_ids] + local_offsets,
        )

    @staticmethod
    def _metadata_row_span_scatter_tensors(
        spans: Tuple[Tuple[int, int, int, int], ...],
        token_count: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = torch.tensor(
            [int(row) for row, _start, _offset, _length in spans],
            dtype=torch.int64,
            device=device,
        )
        starts = torch.tensor(
            [int(start) for _row, start, _offset, _length in spans],
            dtype=torch.int64,
            device=device,
        )
        scratch_starts = torch.tensor(
            [int(offset) for _row, _start, offset, _length in spans],
            dtype=torch.int64,
            device=device,
        )
        lengths = torch.tensor(
            [int(length) for _row, _start, _offset, length in spans],
            dtype=torch.int64,
            device=device,
        )
        span_ids = torch.repeat_interleave(
            torch.arange(int(lengths.numel()), dtype=torch.int64, device=device),
            lengths,
        )
        span_bases = torch.repeat_interleave(
            torch.cumsum(lengths, dim=0) - lengths, lengths
        )
        offsets = torch.arange(int(token_count), dtype=torch.int64, device=device)
        local_offsets = offsets - span_bases
        return (
            rows[span_ids],
            starts[span_ids] + local_offsets,
            scratch_starts[span_ids] + local_offsets,
        )

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
        metadata_signature: Any = self._metadata_patch_layout_signature(demand)
        if metadata_signature is None and (demand.base_signature or demand.signature):
            metadata_signature = (
                str(demand.base_signature or ""),
                int(demand.token_count),
                int(demand.max_row_index),
                int(demand.max_position),
                int(demand.max_flat_index),
            )
        elif metadata_signature is None:
            metadata_signature = (
                self._metadata_int_tuple_signature(demand.req_indices),
                self._metadata_int_tuple_signature(demand.positions),
                self._metadata_int_tuple_signature(demand.row_indices),
                self._metadata_int_tuple_signature(demand.flat_indices),
                self._metadata_span_signature(demand.flat_spans),
                self._metadata_span_signature(demand.row_spans),
            )
        stable_key = (
            int(layer_id),
            metadata_kind,
            metadata_shape,
            metadata_device,
            metadata_signature,
        )
        shared_key = (
            metadata_kind,
            metadata_device,
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
                entry.flat_span_tensor = shared.flat_span_tensor
                entry.flat_span_scratch_tensor = shared.flat_span_scratch_tensor
                entry.row_span_row_tensor = shared.row_span_row_tensor
                entry.row_span_pos_tensor = shared.row_span_pos_tensor
                entry.row_span_scratch_tensor = shared.row_span_scratch_tensor
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

    def _metadata_patch_layout_signature(
        self, demand: _LayerKVKvcDemand
    ) -> Optional[Tuple[Any, ...]]:
        if not (
            demand.flat_spans
            or demand.row_spans
            or demand.flat_indices
            or demand.row_indices
            or demand.positions
        ):
            return None
        return (
            int(demand.token_count),
            int(demand.max_row_index),
            int(demand.max_position),
            int(demand.max_flat_index),
            self._metadata_int_tuple_signature(demand.positions),
            self._metadata_int_tuple_signature(demand.row_indices),
            self._metadata_int_tuple_signature(demand.flat_indices),
            self._metadata_span_signature(demand.flat_spans),
            self._metadata_span_signature(demand.row_spans),
        )

    @staticmethod
    def _metadata_int_tuple_signature(
        values: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        if not values:
            return (0, 0, 0, 0)
        return (
            len(values),
            int(values[0]),
            int(values[-1]),
            int(hash(values)),
        )

    @staticmethod
    def _metadata_span_signature(
        values: Tuple[Tuple[int, ...], ...],
    ) -> Tuple[int, Tuple[int, ...], Tuple[int, ...], int]:
        if not values:
            return (0, (), (), 0)
        return (
            len(values),
            tuple(int(x) for x in values[0]),
            tuple(int(x) for x in values[-1]),
            int(hash(values)),
        )

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
        self._restore_virtual_scratch_for_kvc_use()
        t0 = time.perf_counter()
        next_layer = self._next_kvc_layer_id(layer_id)
        self._add_profile(
            "profile_virtual_prefetch_next_layer_ms",
            (time.perf_counter() - t0) * 1000.0,
        )
        if next_layer is None:
            return
        t0 = time.perf_counter()
        next_demand = self._get_virtual_kvc_demand(int(next_layer))
        self._add_profile(
            "profile_virtual_prefetch_demand_ms",
            (time.perf_counter() - t0) * 1000.0,
        )
        if next_demand is None or not next_demand.entries:
            return
        t0 = time.perf_counter()
        if self._lookup_virtual_scratch_cache(next_demand) is not None:
            self._add_profile(
                "profile_virtual_prefetch_cache_lookup_ms",
                (time.perf_counter() - t0) * 1000.0,
            )
            return
        self._add_profile(
            "profile_virtual_prefetch_cache_lookup_ms",
            (time.perf_counter() - t0) * 1000.0,
        )
        if self._issue_all_virtual_kvc_prefetch_after_layer(
            int(layer_id), exclude_buffer_idx=current_buffer_idx
        ):
            return
        self._issue_virtual_kvc_prefetch(
            next_demand, exclude_buffer_idx=current_buffer_idx
        )

    def _issue_all_virtual_kvc_prefetch_after_layer(
        self,
        layer_id: int,
        *,
        exclude_buffer_idx: Optional[int],
    ) -> bool:
        if (
            self.config.kvc_scheduler != "async-deadline"
            or not self._optimized_profile_enabled()
            or self._copy_stream is None
            or self._host_store is None
        ):
            return False
        self._restore_virtual_scratch_for_kvc_use()
        if getattr(self, "_virtual_batched_prefetch_step", None) == int(
            self._decode_step
        ):
            return False
        layer_ids = [int(candidate) for candidate in self._kvc_layer_ids()]
        try:
            start_idx = layer_ids.index(int(layer_id)) + 1
        except ValueError:
            return False
        candidate_layers = layer_ids[start_idx:]
        if not candidate_layers:
            return False
        buffer_idx = 0
        if len(self._virtual_scratch_buffers) > 1:
            buffer_idx = 1 if int(exclude_buffer_idx or 0) == 0 else 0
        if buffer_idx >= len(self._virtual_scratch_buffers):
            return False
        buffer_locs = self._virtual_scratch_buffers[int(buffer_idx)]
        device_slice_base = (
            self._virtual_scratch_buffer_slices[int(buffer_idx)]
            if int(buffer_idx) < len(self._virtual_scratch_buffer_slices)
            else (-1, 0)
        )
        if int(device_slice_base[0]) < 0:
            return False
        batch: List[Tuple[_LayerKVKvcDemand, torch.Tensor, Tuple[int, int], int]] = []
        requests: List[
            Tuple[
                int, Tuple[int, int], Tuple[Tuple[int, int, int], ...], Tuple[int, int]
            ]
        ] = []
        t0 = time.perf_counter()
        for candidate in candidate_layers:
            demand = self._get_virtual_kvc_demand(int(candidate))
            if demand is None or not demand.entries:
                continue
            if int(demand.token_count) <= 0 or int(demand.token_count) > int(
                buffer_locs.numel()
            ):
                continue
            if self._lookup_virtual_scratch_cache(demand, record_stats=False):
                continue
            key = (int(self._decode_step), int(candidate), demand.signature)
            if key in self._pending_virtual_kvc_materialize:
                continue
            if demand.host_index_cpu is not None:
                continue
            if int(demand.host_slice[0]) < 0 and not demand.host_spans:
                continue
            scratch_locs = buffer_locs[: int(demand.token_count)]
            device_slice = (int(device_slice_base[0]), int(demand.token_count))
            generation = self._bump_virtual_scratch_generation(
                int(buffer_idx), layer_id=int(candidate)
            )
            batch.append((demand, scratch_locs, device_slice, generation))
            requests.append(
                (
                    int(candidate),
                    demand.host_slice,
                    demand.host_spans,
                    device_slice,
                )
            )
        self._add_profile(
            "profile_virtual_prefetch_demand_ms",
            (time.perf_counter() - t0) * 1000.0,
        )
        if len(requests) <= 1:
            return False
        with self._profile("profile_virtual_prefetch_issue_ms"):
            t0 = time.perf_counter()
            start_event, ready_event, issued_count = (
                self._host_store.reload_layers_to_locs_batched(
                    requests,
                    stream=self._copy_stream,
                )
            )
            self._add_profile(
                "profile_virtual_prefetch_reload_call_ms",
                (time.perf_counter() - t0) * 1000.0,
            )
        if start_event is None or ready_event is None or int(issued_count) <= 0:
            return False
        self._virtual_batched_prefetch_step = int(self._decode_step)
        event_group_id = int(id(ready_event))
        t0 = time.perf_counter()
        for demand, scratch_locs, _device_slice, generation in batch:
            key = (
                int(self._decode_step),
                int(demand.layer_id),
                demand.signature,
            )
            self.stats.virtual_kvc_prefetch_count += 1
            self.stats.virtual_kvc_materialize_count += 1
            self.stats.virtual_kvc_materialize_layer_count += 1
            self.stats.virtual_kvc_materialize_token_count += int(demand.token_count)
            self.stats.virtual_scratch_used_tokens = max(
                self.stats.virtual_scratch_used_tokens, int(demand.token_count)
            )
            self._pending_virtual_kvc_materialize[key] = (
                _LayerKVPendingVirtualMaterialize(
                    start_event=start_event,
                    ready_event=ready_event,
                    layer_id=int(demand.layer_id),
                    entries=list(demand.entries),
                    scratch_locs=scratch_locs,
                    req_indices=demand.req_indices,
                    positions=demand.positions,
                    token_count=int(demand.token_count),
                    buffer_idx=int(buffer_idx),
                    buffer_generation=int(generation),
                    event_group_id=event_group_id,
                )
            )
        self._add_profile(
            "profile_virtual_prefetch_record_ms",
            (time.perf_counter() - t0) * 1000.0,
        )
        return True

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
        self._restore_virtual_scratch_for_kvc_use()
        if demand is None or not demand.entries:
            return False
        key = (int(self._decode_step), int(demand.layer_id), demand.signature)
        if key in self._pending_virtual_kvc_materialize:
            return False
        t0 = time.perf_counter()
        buffer_idx = self._choose_virtual_scratch_buffer(
            demand.token_count, exclude_buffer_idx=exclude_buffer_idx
        )
        self._add_profile(
            "profile_virtual_prefetch_buffer_select_ms",
            (time.perf_counter() - t0) * 1000.0,
        )
        if buffer_idx < 0:
            return False
        t0 = time.perf_counter()
        buffer_generation = self._bump_virtual_scratch_generation(
            buffer_idx, layer_id=int(demand.layer_id)
        )
        scratch_locs = self._virtual_scratch_buffers[buffer_idx][: demand.token_count]
        device_slice = (
            self._virtual_scratch_buffer_slices[int(buffer_idx)]
            if int(buffer_idx) < len(self._virtual_scratch_buffer_slices)
            else (-1, 0)
        )
        self._ensure_host_store()
        self._add_profile(
            "profile_virtual_prefetch_setup_ms",
            (time.perf_counter() - t0) * 1000.0,
        )
        with self._profile("profile_virtual_prefetch_issue_ms"):
            t0 = time.perf_counter()
            elapsed_ms, start_event, ready_event = (
                self._host_store.reload_layer_to_locs(
                    int(demand.layer_id),
                    demand.entries,
                    scratch_locs,
                    stream=self._copy_stream,
                    async_copy=True,
                    host_index=demand.host_index_cpu,
                    host_slice=demand.host_slice,
                    host_spans=demand.host_spans,
                    device_slice=device_slice,
                )
            )
            self._add_profile(
                "profile_virtual_prefetch_reload_call_ms",
                (time.perf_counter() - t0) * 1000.0,
            )
        if ready_event is None or start_event is None:
            self._record_virtual_materialize(demand.token_count, elapsed_ms)
            return True
        t0 = time.perf_counter()
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
            buffer_generation=int(buffer_generation),
            event_group_id=int(id(ready_event)),
        )
        self._add_profile(
            "profile_virtual_prefetch_record_ms",
            (time.perf_counter() - t0) * 1000.0,
        )
        return True

    def _wait_for_virtual_materialize(
        self, pending: _LayerKVPendingVirtualMaterialize
    ) -> Optional[torch.Tensor]:
        self._restore_virtual_scratch_for_kvc_use()
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
        wait_start = None
        wait_end = None
        if not ready:
            try:
                wait_start = torch.cuda.Event(enable_timing=True)
                wait_end = torch.cuda.Event(enable_timing=True)
                wait_start.record(stream)
            except Exception:
                wait_start = None
                wait_end = None
        stream.wait_event(pending.ready_event)
        if wait_end is not None:
            try:
                wait_end.record(stream)
            except Exception:
                wait_end = None
        self.stats.scheduler_exposed_wait_ms += (time.perf_counter() - t_wait) * 1000.0
        self.stats.layerkv_copy_event_wait_count += 1
        if wait_start is not None and wait_end is not None:
            try:
                wait_end.synchronize()
                stall_ms = float(wait_start.elapsed_time(wait_end))
                self.stats.kvc_ready_miss_stall_ms += stall_ms
                self.stats.layerkv_main_stream_wait_ms += stall_ms
                self.stats.kvc_ready_miss_stall_count += 1
            except Exception:
                pass
        try:
            event_group_id = int(getattr(pending, "event_group_id", 0))
            if event_group_id <= 0 or (
                event_group_id not in self._virtual_materialize_recorded_event_groups
            ):
                elapsed_ms = float(
                    pending.start_event.elapsed_time(pending.ready_event)
                )
                self.stats.virtual_kvc_materialize_ms += elapsed_ms
                self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
                if event_group_id > 0:
                    self._virtual_materialize_recorded_event_groups.add(event_group_id)
        except Exception:
            pass
        self._refresh_ready_before_use_ratio()
        return pending.scratch_locs
