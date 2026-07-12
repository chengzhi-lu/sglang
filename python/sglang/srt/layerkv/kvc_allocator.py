"""LayerKV LayerKVKvcAllocatorMixin implementation."""

from __future__ import annotations

import bisect
import contextlib
import functools
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
        _LayerKVPerLayerReqCleanupState,
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
        _LayerKVPerLayerReqCleanupState,
        _LayerKVPendingEviction,
        _LayerKVPendingReload,
        _LayerKVPendingVirtualMaterialize,
        _LayerKVRecoveryTask,
        _LayerKVResidencyEntry,
        _LayerKVVirtualMaterializePlan,
        _LayerKVVirtualScratchCacheEntry,
    )

logger = logging.getLogger(__name__)


class LayerKVKvcAllocatorMixin:
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
            self._per_layer_arena_reserved_locs or self._per_layer_owned_req_indices
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

    def _per_layer_kvc_io_control_active(self) -> bool:
        if not self._per_layer_allocator_enabled():
            return False
        return bool(
            self._per_layer_req_to_token_owned
            or self._per_layer_offloaded_keys
            or self._pending_virtual_kvc_materialize
            or self._pending_kvc_evict_events
            or self._pending_kvc_reload_events
        )

    def _mark_per_layer_common_free_dirty(self) -> None:
        self._per_layer_arena_common_free_dirty = True

    def _ensure_per_layer_common_free_current(self) -> None:
        if not self._per_layer_arena_common_free_dirty:
            return
        layer_ids = [int(layer_id) for layer_id in self._kvc_layer_ids()]
        common: Optional[Set[int]] = None
        for layer_id in layer_ids:
            free = {
                int(loc)
                for loc in self._per_layer_arena_free_locs.get(layer_id, [])
                if int(loc) > 0
            }
            free.difference_update(
                self._per_layer_arena_allocated_locs.setdefault(layer_id, set())
            )
            free.difference_update(
                self._per_layer_arena_protected_locs.setdefault(layer_id, set())
            )
            common = free if common is None else common.intersection(free)
            if not common:
                break
        common = common or set()
        self._per_layer_arena_common_free_locs = set(common)
        ordered = [
            int(loc)
            for loc in self._per_layer_arena_common_free_order
            if int(loc) in common
        ]
        seen = set(ordered)
        ordered.extend(int(loc) for loc in sorted(common) if int(loc) not in seen)
        self._per_layer_arena_common_free_order = ordered
        self._per_layer_arena_common_free_dirty = False

    def _refresh_per_layer_allocator_stats(self) -> None:
        if not self._per_layer_arena_free_locs:
            self.stats.kvc_per_layer_physical_arena_token_capacity = 0
            self.stats.kvc_per_layer_physical_arena_min_free_tokens = 0
            self.stats.kvc_per_layer_physical_arena_common_free_tokens = 0
            self.stats.kvc_per_layer_logical_request_count = len(
                self._per_layer_owned_req_indices
            )
            return
        layer_ids = [int(layer_id) for layer_id in self._kvc_layer_ids()]
        free_counts = [
            len(self._per_layer_arena_free_locs.get(layer_id, []))
            + self._per_layer_overwrite_count(layer_id)
            for layer_id in layer_ids
        ]
        if free_counts:
            self.stats.kvc_per_layer_physical_arena_min_free_tokens = min(free_counts)
        else:
            self.stats.kvc_per_layer_physical_arena_min_free_tokens = 0
        self.stats.kvc_per_layer_physical_arena_common_free_tokens = (
            self._per_layer_logical_common_free_count(layer_ids=layer_ids)
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

    def _per_layer_overwrite_count(self, layer_id: int) -> int:
        layer_id = int(layer_id)
        return max(
            0,
            int(self._per_layer_arena_overwrite_counts.get(layer_id, 0))
            + len(self._per_layer_arena_overwrite_pending_locs.get(layer_id, []))
            + int(self._per_layer_arena_overwrite_pending_bit_counts.get(layer_id, 0)),
        )

    def _per_layer_logical_common_free_count(
        self, *, layer_ids: Optional[List[int]] = None
    ) -> int:
        if layer_ids is None:
            layer_ids = [int(layer_id) for layer_id in self._kvc_layer_ids()]
        physical_common = len(self._per_layer_arena_common_free_locs)
        if not layer_ids:
            return physical_common
        overwrite_counts = [self._per_layer_overwrite_count(int(x)) for x in layer_ids]
        return int(physical_common) + (min(overwrite_counts) if overwrite_counts else 0)

    def _ensure_per_layer_physical_arena(self, min_free_tokens: int = 1) -> bool:
        if not self._per_layer_allocator_enabled():
            return False
        layer_ids = self._kvc_layer_ids()
        if not layer_ids:
            return False
        min_free_tokens = max(1, int(min_free_tokens or 1))
        self._refresh_per_layer_allocator_stats()
        self._ensure_per_layer_common_free_current()
        logical_common_free = self._per_layer_logical_common_free_count(
            layer_ids=[int(x) for x in layer_ids]
        )
        if len(self._per_layer_arena_common_free_locs) >= min_free_tokens:
            self.stats.kvc_per_layer_physical_arena_common_free_tokens = (
                logical_common_free
            )
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
        return len(self._per_layer_arena_common_free_locs) >= min_free_tokens

    def _alloc_per_layer_locs(
        self, layer_id: int, count: int, *, refresh_stats: bool = True
    ) -> Optional[List[int]]:
        layer_id = int(layer_id)
        count = max(0, int(count))
        if count <= 0:
            return []
        overwrite_locs = self._pop_per_layer_overwrite_locs(layer_id, count)
        if len(overwrite_locs) >= count:
            return overwrite_locs
        remaining = count - len(overwrite_locs)
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
        if len(free) < remaining and (
            self._per_layer_arena_common_free_locs
            or self._per_layer_arena_common_free_dirty
        ):
            self._ensure_per_layer_common_free_current()
            present = {int(loc) for loc in free}
            for loc in sorted(self._per_layer_arena_common_free_locs):
                loc = int(loc)
                if loc <= 0 or loc in allocated or loc in protected or loc in present:
                    continue
                free.append(loc)
                present.add(loc)
                if len(free) >= remaining:
                    break
        if len(free) < remaining:
            if not self._ensure_per_layer_physical_arena(remaining - len(free)):
                if overwrite_locs:
                    self._push_per_layer_overwrite_locs(layer_id, overwrite_locs)
                self.stats.kvc_per_layer_physical_arena_alloc_failed_count += 1
                return None
            free = self._per_layer_arena_free_locs.setdefault(layer_id, [])
        if len(free) < remaining:
            if overwrite_locs:
                self._push_per_layer_overwrite_locs(layer_id, overwrite_locs)
            self.stats.kvc_per_layer_physical_arena_alloc_failed_count += 1
            return None
        locs = free[-remaining:]
        del free[-remaining:]
        for loc in locs:
            self._per_layer_arena_common_free_locs.discard(int(loc))
        self._per_layer_arena_allocated_locs.setdefault(layer_id, set()).update(locs)
        self.stats.kvc_per_layer_physical_arena_alloc_count += remaining
        if refresh_stats:
            self._refresh_per_layer_allocator_stats()
        return [int(x) for x in overwrite_locs] + [int(x) for x in locs]

    def _alloc_common_per_layer_locs(self, count: int) -> Optional[List[int]]:
        count = max(0, int(count))
        if count <= 0:
            return []
        self._ensure_per_layer_common_free_current()
        if not self._ensure_per_layer_physical_arena(count):
            return None
        self._ensure_per_layer_common_free_current()
        layer_ids = [int(x) for x in self._kvc_layer_ids()]
        common = self._per_layer_arena_common_free_locs
        if len(common) < count:
            self.stats.kvc_per_layer_physical_arena_alloc_failed_count += 1
            return None
        locs: List[int] = []
        selected: Set[int] = set()
        while self._per_layer_arena_common_free_order and len(locs) < count:
            loc = int(self._per_layer_arena_common_free_order.pop())
            if loc in selected:
                continue
            if loc in common and not any(
                self._per_layer_loc_is_protected(layer_id, loc)
                for layer_id in layer_ids
            ):
                locs.append(loc)
                selected.add(loc)
        if len(locs) < count:
            for loc in sorted(common):
                loc = int(loc)
                if loc in selected or any(
                    self._per_layer_loc_is_protected(layer_id, loc)
                    for layer_id in layer_ids
                ):
                    continue
                locs.append(loc)
                selected.add(loc)
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

    def _free_per_layer_locs_batch(
        self,
        per_layer_locs: Dict[int, List[int]],
        *,
        refresh: bool = True,
        assume_allocated: bool = False,
    ) -> None:
        if not per_layer_locs:
            return
        freed_count = 0
        for layer_id, locs in per_layer_locs.items():
            if not locs:
                continue
            layer_id = int(layer_id)
            allocated = self._per_layer_arena_allocated_locs.setdefault(layer_id, set())
            loc_set = {int(x) for x in locs}
            reusable = loc_set if assume_allocated else loc_set.intersection(allocated)
            if not reusable:
                continue
            if assume_allocated:
                self._per_layer_arena_reserved_locs.update(reusable)
            allocated.difference_update(reusable)
            protected = self._per_layer_arena_protected_locs.setdefault(layer_id, set())
            ordinary_reusable = reusable.difference(protected)

            free = self._per_layer_arena_free_locs.setdefault(layer_id, [])
            free.extend(int(loc) for loc in ordinary_reusable)
            freed_count += len(reusable)

        if freed_count:
            self.stats.kvc_per_layer_physical_arena_free_count += freed_count
            self._mark_per_layer_common_free_dirty()

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
        if self._consume_per_layer_overwrite_locs(layer_id, locs):
            return list(locs)
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
            self._per_layer_arena_protected_locs.setdefault(
                layer_id, set()
            ).difference_update(loc_set)
            allocated.update(loc_set)
            self.stats.kvc_per_layer_physical_arena_alloc_count += len(loc_set)
            self._refresh_per_layer_allocator_stats()
        return list(locs)

    def _push_per_layer_overwrite_locs(
        self, layer_id: int, locs: List[int], *, assume_new: bool = False
    ) -> None:
        if not locs:
            return
        layer_id = int(layer_id)
        if assume_new:
            pending = self._per_layer_arena_overwrite_pending_locs.setdefault(
                layer_id, []
            )
            pending.extend(locs)
            return
        loc_tensor = torch.as_tensor(locs, dtype=torch.int64, device="cpu")
        if loc_tensor.numel() == 0:
            return
        loc_tensor = loc_tensor[loc_tensor > 0]
        if loc_tensor.numel() == 0:
            return
        if not assume_new:
            loc_tensor = torch.unique(loc_tensor)
        bitmap = self._ensure_per_layer_overwrite_bitmap(
            layer_id, int(loc_tensor.max().item()) + 1
        )
        if assume_new:
            newly_set = int(loc_tensor.numel())
        else:
            already_set = bitmap.index_select(0, loc_tensor)
            newly_set = int(loc_tensor.numel()) - int(
                torch.count_nonzero(already_set).item()
            )
        if newly_set > 0:
            self._per_layer_arena_overwrite_counts[layer_id] = int(
                self._per_layer_arena_overwrite_counts.get(layer_id, 0)
            ) + int(newly_set)
        bitmap.index_fill_(0, loc_tensor, True)

    def _locs_to_bitset(self, locs: List[int], *, min_value: int = 0) -> int:
        bits = 0
        min_value = int(min_value)
        for loc in locs:
            loc = int(loc)
            if loc >= min_value:
                bits |= 1 << loc
        return bits

    def _bitset_to_locs(self, bits: int, *, limit: int = 0) -> List[int]:
        bits = int(bits)
        limit = max(0, int(limit))
        locs: List[int] = []
        while bits and (limit <= 0 or len(locs) < limit):
            lsb = bits & -bits
            locs.append(int(lsb.bit_length() - 1))
            bits ^= lsb
        return locs

    def _push_per_layer_overwrite_bits(
        self, layer_id: int, bits: int, *, token_count: Optional[int] = None
    ) -> None:
        bits = int(bits)
        if bits <= 0:
            return
        layer_id = int(layer_id)
        if token_count is None:
            token_count = int(bits).bit_count()
        self._per_layer_arena_overwrite_pending_bit_chunks.setdefault(
            layer_id, []
        ).append(bits)
        self._per_layer_arena_overwrite_pending_bit_counts[layer_id] = int(
            self._per_layer_arena_overwrite_pending_bit_counts.get(layer_id, 0)
        ) + max(0, int(token_count))

    def _per_layer_pending_overwrite_bits_union(self, layer_id: int) -> int:
        layer_id = int(layer_id)
        bits = int(self._per_layer_arena_overwrite_pending_bits.get(layer_id, 0))
        for chunk in self._per_layer_arena_overwrite_pending_bit_chunks.get(
            layer_id, []
        ):
            bits |= int(chunk)
        return bits

    def _remove_per_layer_pending_overwrite_bits(
        self, layer_id: int, remove_bits: int, *, token_count: Optional[int] = None
    ) -> None:
        layer_id = int(layer_id)
        remove_bits = int(remove_bits)
        if remove_bits <= 0:
            return
        pending_bits = int(
            self._per_layer_arena_overwrite_pending_bits.get(layer_id, 0)
        )
        if pending_bits:
            pending_bits &= ~remove_bits
            if pending_bits:
                self._per_layer_arena_overwrite_pending_bits[layer_id] = pending_bits
            else:
                self._per_layer_arena_overwrite_pending_bits.pop(layer_id, None)
        chunks = self._per_layer_arena_overwrite_pending_bit_chunks.get(layer_id)
        if chunks:
            kept = []
            for chunk in chunks:
                chunk = int(chunk) & ~remove_bits
                if chunk:
                    kept.append(chunk)
            if kept:
                self._per_layer_arena_overwrite_pending_bit_chunks[layer_id] = kept
            else:
                self._per_layer_arena_overwrite_pending_bit_chunks.pop(layer_id, None)
        if token_count is None:
            token_count = int(remove_bits).bit_count()
        next_count = max(
            0,
            int(self._per_layer_arena_overwrite_pending_bit_counts.get(layer_id, 0))
            - max(0, int(token_count)),
        )
        if next_count:
            self._per_layer_arena_overwrite_pending_bit_counts[layer_id] = next_count
        else:
            self._per_layer_arena_overwrite_pending_bit_counts.pop(layer_id, None)

    def _pop_per_layer_overwrite_locs(self, layer_id: int, count: int) -> List[int]:
        layer_id = int(layer_id)
        count = max(0, int(count))
        if count <= 0:
            return []
        locs: List[int] = []
        bitmap = self._per_layer_arena_overwrite_bitmaps.get(layer_id)
        if bitmap is not None and bitmap.numel() > 0:
            loc_tensor = torch.nonzero(bitmap, as_tuple=False).flatten()
            if loc_tensor.numel() > 0:
                if int(loc_tensor.numel()) > count:
                    loc_tensor = loc_tensor[-count:]
                bitmap.index_fill_(0, loc_tensor, False)
                bitmap_count = max(
                    0,
                    int(self._per_layer_arena_overwrite_counts.get(layer_id, 0))
                    - int(loc_tensor.numel()),
                )
                self._per_layer_arena_overwrite_counts[layer_id] = bitmap_count
                locs.extend(int(loc) for loc in loc_tensor.tolist())
        remaining = count - len(locs)
        if remaining > 0:
            pending = self._per_layer_arena_overwrite_pending_locs.setdefault(
                layer_id, []
            )
            if pending:
                seen = set(locs)
                kept_pending: List[int] = []
                for loc in reversed(pending):
                    loc = int(loc)
                    if loc <= 0 or loc in seen:
                        continue
                    if len(locs) < count:
                        locs.append(loc)
                        seen.add(loc)
                    else:
                        kept_pending.append(loc)
                kept_pending.reverse()
                pending[:] = kept_pending
        remaining = count - len(locs)
        if remaining > 0:
            chunks = self._per_layer_arena_overwrite_pending_bit_chunks.get(layer_id)
            if chunks:
                seen = set(locs)
                popped: List[int] = []
                kept_chunks: List[int] = []
                for chunk in reversed(chunks):
                    bits = int(chunk)
                    while bits and len(popped) < remaining:
                        loc = int(bits.bit_length() - 1)
                        bit = 1 << loc
                        bits ^= bit
                        if loc <= 0 or loc in seen:
                            continue
                        popped.append(loc)
                        seen.add(loc)
                    if bits:
                        kept_chunks.append(bits)
                kept_chunks.reverse()
                locs.extend(popped)
                removed = len(popped)
                if kept_chunks:
                    self._per_layer_arena_overwrite_pending_bit_chunks[layer_id] = (
                        kept_chunks
                    )
                else:
                    self._per_layer_arena_overwrite_pending_bit_chunks.pop(
                        layer_id, None
                    )
                next_count = max(
                    0,
                    int(
                        self._per_layer_arena_overwrite_pending_bit_counts.get(
                            layer_id, 0
                        )
                    )
                    - int(removed),
                )
                if next_count:
                    self._per_layer_arena_overwrite_pending_bit_counts[layer_id] = (
                        next_count
                    )
                else:
                    self._per_layer_arena_overwrite_pending_bit_counts.pop(
                        layer_id, None
                    )
        return locs

    def _consume_per_layer_overwrite_locs(self, layer_id: int, locs: List[int]) -> bool:
        if not locs:
            return True
        layer_id = int(layer_id)
        loc_bits = self._locs_to_bitset(locs, min_value=1)
        pending_bits = int(
            self._per_layer_arena_overwrite_pending_bits.get(layer_id, 0)
        )
        pending_bits |= self._per_layer_pending_overwrite_bits_union(layer_id)
        pending_match = pending_bits & loc_bits
        remaining_bits = loc_bits & ~pending_match
        if loc_bits > 0 and remaining_bits == 0:
            self._remove_per_layer_pending_overwrite_bits(
                layer_id, pending_match, token_count=len(locs)
            )
            return True
        bitmap = self._per_layer_arena_overwrite_bitmaps.get(layer_id)
        if bitmap is None or bitmap.numel() == 0 or remaining_bits <= 0:
            return False
        remaining_locs = self._bitset_to_locs(remaining_bits)
        loc_tensor = torch.as_tensor(remaining_locs, dtype=torch.int64, device="cpu")
        if loc_tensor.numel() == 0 or bool(torch.any(loc_tensor <= 0).item()):
            return False
        loc_tensor = torch.unique(loc_tensor)
        if int(loc_tensor.max().item()) >= int(bitmap.numel()):
            return False
        if not bool(torch.all(bitmap.index_select(0, loc_tensor)).item()):
            return False
        if pending_match:
            self._remove_per_layer_pending_overwrite_bits(layer_id, pending_match)
        bitmap.index_fill_(0, loc_tensor, False)
        self._per_layer_arena_overwrite_counts[layer_id] = max(
            0,
            int(self._per_layer_arena_overwrite_counts.get(layer_id, 0))
            - int(loc_tensor.numel()),
        )
        return True

    def _clear_per_layer_physical_mapping(self, layer_id: int, locs: Set[int]) -> None:
        if not locs:
            return
        layer_id = int(layer_id)
        reverse = self._per_layer_physical_to_canonical.setdefault(layer_id, {})
        mapping = self._per_layer_canonical_to_physical.setdefault(layer_id, {})
        for loc in locs:
            canonical = reverse.pop(int(loc), None)
            if canonical is not None and int(mapping.get(int(canonical), -1)) == int(
                loc
            ):
                mapping.pop(int(canonical), None)
        if not any(int(k) != int(v) for k, v in mapping.items()):
            self._per_layer_non_identity_mapping.discard(layer_id)

    def _pop_common_per_layer_overwrite_locs(self, count: int) -> List[int]:
        count = max(0, int(count))
        if count <= 0:
            return []
        layer_ids = [int(layer_id) for layer_id in self._kvc_layer_ids()]
        if not layer_ids:
            return []
        common: Optional[Set[int]] = None
        for layer_id in layer_ids:
            bitmap = self._per_layer_arena_overwrite_bitmaps.get(layer_id)
            locs: Set[int] = set()
            if bitmap is not None and int(bitmap.numel()) > 0:
                loc_tensor = torch.nonzero(bitmap, as_tuple=False).flatten()
                if int(loc_tensor.numel()) > 0:
                    locs.update(int(loc) for loc in loc_tensor.tolist() if int(loc) > 0)
            pending_bits = self._per_layer_pending_overwrite_bits_union(layer_id)
            if pending_bits > 0:
                locs.update(self._bitset_to_locs(pending_bits))
            if not locs:
                return []
            common = locs if common is None else common.intersection(locs)
            if not common:
                return []
        selected = sorted(common or set())[:count]
        if not selected:
            return []
        loc_tensor = torch.as_tensor(selected, dtype=torch.int64, device="cpu")
        loc_set = set(selected)
        for layer_id in layer_ids:
            bitmap = self._per_layer_arena_overwrite_bitmaps.get(layer_id)
            if bitmap is not None and int(bitmap.numel()) > 0:
                bitmap.index_fill_(0, loc_tensor, False)
            self._per_layer_arena_overwrite_counts[layer_id] = max(
                0,
                int(self._per_layer_arena_overwrite_counts.get(layer_id, 0))
                - len(selected),
            )
            self._remove_per_layer_pending_overwrite_bits(
                layer_id,
                self._locs_to_bitset(selected, min_value=1),
                token_count=len(selected),
            )
            self._per_layer_arena_allocated_locs.setdefault(
                layer_id, set()
            ).difference_update(loc_set)
            self._per_layer_arena_protected_locs.setdefault(
                layer_id, set()
            ).difference_update(loc_set)
            free = self._per_layer_arena_free_locs.setdefault(layer_id, [])
            self._per_layer_arena_free_locs[layer_id] = [
                int(loc) for loc in free if int(loc) not in loc_set
            ]
            self._clear_per_layer_physical_mapping(layer_id, loc_set)
        self._per_layer_arena_common_free_locs.difference_update(loc_set)
        self._per_layer_arena_common_free_order = [
            int(loc)
            for loc in self._per_layer_arena_common_free_order
            if int(loc) not in loc_set
        ]
        self._per_layer_arena_reserved_locs.difference_update(loc_set)
        self._refresh_per_layer_allocator_stats()
        return selected

    def _release_common_per_layer_locs_to_native(self, count: int) -> int:
        if self.config.kvc_backend != "per-layer-arena":
            return 0
        count = max(0, int(count))
        if count <= 0 or self._allocator is None:
            return 0
        layer_ids = [int(layer_id) for layer_id in self._kvc_layer_ids()]
        if not layer_ids:
            return 0
        selected: List[int] = []
        self._ensure_per_layer_common_free_current()
        while self._per_layer_arena_common_free_order and len(selected) < count:
            loc = int(self._per_layer_arena_common_free_order.pop(0))
            if loc <= 0 or loc not in self._per_layer_arena_common_free_locs:
                continue
            if any(
                self._per_layer_loc_is_protected(layer_id, loc)
                for layer_id in layer_ids
            ):
                continue
            if any(
                loc in self._per_layer_arena_allocated_locs.setdefault(layer_id, set())
                for layer_id in layer_ids
            ):
                continue
            selected.append(loc)
        remaining = count - len(selected)
        if remaining > 0:
            selected.extend(self._pop_common_per_layer_overwrite_locs(remaining))
        if not selected:
            return 0
        loc_set = {int(loc) for loc in selected}
        for layer_id in layer_ids:
            free = self._per_layer_arena_free_locs.setdefault(layer_id, [])
            self._per_layer_arena_free_locs[layer_id] = [
                int(loc) for loc in free if int(loc) not in loc_set
            ]
            self._per_layer_arena_allocated_locs.setdefault(
                layer_id, set()
            ).difference_update(loc_set)
            self._per_layer_arena_protected_locs.setdefault(
                layer_id, set()
            ).difference_update(loc_set)
            self._clear_per_layer_physical_mapping(layer_id, loc_set)
        self._per_layer_arena_common_free_locs.difference_update(loc_set)
        self._per_layer_arena_common_free_order = [
            int(loc)
            for loc in self._per_layer_arena_common_free_order
            if int(loc) not in loc_set
        ]
        self._per_layer_arena_reserved_locs.difference_update(loc_set)
        free_fn = self._wrapped_methods.get("allocator.free") or getattr(
            self._allocator, "free", None
        )
        if free_fn is None:
            return 0
        loc_tensor = torch.as_tensor(
            sorted(loc_set), dtype=torch.int64, device=self._allocator.device
        )
        free_fn(loc_tensor)
        self._refresh_per_layer_allocator_stats()
        return len(loc_set)

    def _prewarm_per_layer_overwrite_bitmaps(self) -> int:
        if self.config.kvc_backend != "per-layer-arena":
            return 0
        allocator_size = max(0, int(self._allocator_total_size()))
        if allocator_size <= 0:
            return 0
        warmed = 0
        for layer_id in self._kvc_layer_ids():
            layer_id = int(layer_id)
            bitmap = self._per_layer_arena_overwrite_bitmaps.get(layer_id)
            if bitmap is not None and int(bitmap.numel()) >= allocator_size + 1:
                continue
            self._ensure_per_layer_overwrite_bitmap(layer_id, allocator_size + 1)
            warmed += 1
        return warmed

    def _ensure_per_layer_overwrite_bitmap(
        self, layer_id: int, min_size: int
    ) -> torch.Tensor:
        layer_id = int(layer_id)
        min_size = max(1, int(min_size))
        bitmap = self._per_layer_arena_overwrite_bitmaps.get(layer_id)
        if bitmap is not None and int(bitmap.numel()) >= min_size:
            return bitmap
        allocator_size = max(0, int(self._allocator_total_size()))
        target_size = max(min_size, allocator_size + 1)
        if bitmap is not None:
            target_size = max(target_size, int(bitmap.numel()) * 2)
        new_bitmap = torch.zeros(int(target_size), dtype=torch.bool, device="cpu")
        if bitmap is not None and int(bitmap.numel()) > 0:
            new_bitmap[: int(bitmap.numel())].copy_(bitmap)
        self._per_layer_arena_overwrite_bitmaps[layer_id] = new_bitmap
        return new_bitmap

    def _mark_req_layerkv_owned(
        self, req_idx: int, keys: List[Tuple[int, int, int]]
    ) -> None:
        req_idx = int(req_idx)
        self._per_layer_owned_req_indices.add(req_idx)
        normalized_keys = [(int(a), int(b), int(c)) for a, b, c in keys]
        for key in normalized_keys:
            entry = self._per_layer_residency.get(key)
            if entry is not None:
                self._track_per_layer_cleanup_entry(entry, key=key)
        if (
            req_idx not in self._per_layer_cleanup_state_by_req
            and req_idx not in self._per_layer_cleanup_layers_by_req
        ):
            self._per_layer_owned_keys_by_req.setdefault(req_idx, set()).update(
                normalized_keys
            )
        self.stats.kvc_per_layer_logical_request_count = len(
            self._per_layer_owned_req_indices
        )

    def _track_per_layer_cleanup_entry(
        self,
        entry: _LayerKVResidencyEntry,
        *,
        key: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        if self.config.kvc_backend != "per-layer-arena":
            return
        if entry is None or not self._per_layer_entry_is_current(entry):
            return
        req_idx = int(entry.req_idx)
        layer_id = int(entry.layer_id)
        if key is None:
            key = (layer_id, req_idx, int(entry.pos))
        else:
            key = (int(key[0]), int(key[1]), int(key[2]))
        state = self._per_layer_cleanup_state_by_req.setdefault(
            req_idx, _LayerKVPerLayerReqCleanupState()
        )
        state.token_count = int(state.token_count) + int(entry.token_count)
        state.layers.add(layer_id)
        locs = entry.device_loc_list()
        if locs:
            self._add_per_layer_cleanup_locs(entry, locs)
        host_slots = entry.host_slot_list()
        if host_slots:
            self._add_per_layer_cleanup_host_slots(entry, host_slots)

    def _track_per_layer_cleanup_append(
        self, entry: _LayerKVResidencyEntry, physical_loc: int
    ) -> None:
        if self.config.kvc_backend != "per-layer-arena":
            return
        if entry is None or not self._per_layer_entry_is_current(entry):
            return
        req_idx = int(entry.req_idx)
        state = self._per_layer_cleanup_state_by_req.get(req_idx)
        if state is None:
            self._track_per_layer_cleanup_entry(entry)
            return
        state.token_count = int(state.token_count) + 1
        state.layers.add(int(entry.layer_id))
        self._add_per_layer_cleanup_locs(entry, [int(physical_loc)])

    def _add_per_layer_cleanup_locs(
        self, entry: _LayerKVResidencyEntry, locs: List[int]
    ) -> None:
        if self.config.kvc_backend != "per-layer-arena" or not locs:
            return
        if entry is None or not self._per_layer_entry_is_current(entry):
            return
        req_idx = int(entry.req_idx)
        layer_id = int(entry.layer_id)
        state = self._per_layer_cleanup_state_by_req.setdefault(
            req_idx, _LayerKVPerLayerReqCleanupState()
        )
        state.layers.add(layer_id)
        current = int(state.loc_bits_by_layer.get(layer_id, 0))
        add_bits = self._locs_to_bitset(locs, min_value=1)
        new_bits = int(add_bits & ~current)
        state.loc_bits_by_layer[layer_id] = int(current | add_bits)
        if new_bits:
            state.loc_counts_by_layer[layer_id] = int(
                state.loc_counts_by_layer.get(layer_id, 0)
            ) + int(new_bits.bit_count())

    def _remove_per_layer_cleanup_locs(
        self, entry: _LayerKVResidencyEntry, locs: List[int]
    ) -> None:
        if self.config.kvc_backend != "per-layer-arena" or not locs:
            return
        if entry is None:
            return
        remove_bits = self._locs_to_bitset(locs, min_value=1)
        self._remove_per_layer_cleanup_loc_bits(
            int(entry.req_idx), int(entry.layer_id), int(remove_bits)
        )

    def _remove_per_layer_cleanup_loc_bits(
        self, req_idx: int, layer_id: int, remove_bits: int
    ) -> None:
        if self.config.kvc_backend != "per-layer-arena":
            return
        remove_bits = int(remove_bits)
        if remove_bits <= 0:
            return
        req_idx = int(req_idx)
        layer_id = int(layer_id)
        state = self._per_layer_cleanup_state_by_req.get(req_idx)
        if state is None:
            key = (req_idx, layer_id)
            loc_bits = int(self._per_layer_cleanup_locs_by_req_layer.get(key, 0))
        else:
            loc_bits = int(state.loc_bits_by_layer.get(layer_id, 0))
        if loc_bits <= 0:
            return
        removed_bits = int(loc_bits & remove_bits)
        if removed_bits <= 0:
            return
        loc_bits &= ~remove_bits
        if state is None:
            key = (req_idx, layer_id)
            if loc_bits:
                self._per_layer_cleanup_locs_by_req_layer[key] = loc_bits
            else:
                self._per_layer_cleanup_locs_by_req_layer.pop(key, None)
            current_count = int(
                self._per_layer_cleanup_loc_count_by_req_layer.get(key, 0)
            )
        else:
            if loc_bits:
                state.loc_bits_by_layer[layer_id] = loc_bits
            else:
                state.loc_bits_by_layer.pop(layer_id, None)
            current_count = int(state.loc_counts_by_layer.get(layer_id, 0))
        next_count = max(0, current_count - int(removed_bits.bit_count()))
        if state is None:
            key = (req_idx, layer_id)
            if next_count:
                self._per_layer_cleanup_loc_count_by_req_layer[key] = next_count
            else:
                self._per_layer_cleanup_loc_count_by_req_layer.pop(key, None)
        else:
            if next_count:
                state.loc_counts_by_layer[layer_id] = next_count
            else:
                state.loc_counts_by_layer.pop(layer_id, None)

    def _add_per_layer_cleanup_host_slots(
        self, entry: _LayerKVResidencyEntry, host_slots: List[int]
    ) -> None:
        if self.config.kvc_backend != "per-layer-arena" or not host_slots:
            return
        if entry is None or not self._per_layer_entry_is_current(entry):
            return
        req_idx = int(entry.req_idx)
        layer_id = int(entry.layer_id)
        state = self._per_layer_cleanup_state_by_req.setdefault(
            req_idx, _LayerKVPerLayerReqCleanupState()
        )
        state.layers.add(layer_id)
        current = int(state.host_slot_bits_by_layer.get(layer_id, 0))
        add_bits = self._locs_to_bitset(host_slots, min_value=0)
        new_bits = int(add_bits & ~current)
        state.host_slot_bits_by_layer[layer_id] = int(current | add_bits)
        if new_bits:
            state.host_slot_counts_by_layer[layer_id] = int(
                state.host_slot_counts_by_layer.get(layer_id, 0)
            ) + int(new_bits.bit_count())

    def _remove_per_layer_cleanup_host_slots(
        self, entry: _LayerKVResidencyEntry, host_slots: List[int]
    ) -> None:
        if self.config.kvc_backend != "per-layer-arena" or not host_slots:
            return
        if entry is None:
            return
        req_idx = int(entry.req_idx)
        layer_id = int(entry.layer_id)
        state = self._per_layer_cleanup_state_by_req.get(req_idx)
        if state is None:
            key = (req_idx, layer_id)
            slot_bits = int(self._per_layer_cleanup_host_slots_by_req_layer.get(key, 0))
        else:
            slot_bits = int(state.host_slot_bits_by_layer.get(layer_id, 0))
        if slot_bits <= 0:
            return
        remove_bits = self._locs_to_bitset(host_slots, min_value=0)
        removed_bits = int(slot_bits & remove_bits)
        if removed_bits <= 0:
            return
        slot_bits &= ~remove_bits
        if state is None:
            key = (req_idx, layer_id)
            if slot_bits:
                self._per_layer_cleanup_host_slots_by_req_layer[key] = slot_bits
            else:
                self._per_layer_cleanup_host_slots_by_req_layer.pop(key, None)
            current_count = int(
                self._per_layer_cleanup_host_slot_count_by_req_layer.get(key, 0)
            )
        else:
            if slot_bits:
                state.host_slot_bits_by_layer[layer_id] = slot_bits
            else:
                state.host_slot_bits_by_layer.pop(layer_id, None)
            current_count = int(state.host_slot_counts_by_layer.get(layer_id, 0))
        next_count = max(0, current_count - int(removed_bits.bit_count()))
        if state is None:
            key = (req_idx, layer_id)
            if next_count:
                self._per_layer_cleanup_host_slot_count_by_req_layer[key] = next_count
            else:
                self._per_layer_cleanup_host_slot_count_by_req_layer.pop(key, None)
        else:
            if next_count:
                state.host_slot_counts_by_layer[layer_id] = next_count
            else:
                state.host_slot_counts_by_layer.pop(layer_id, None)

    def _record_per_layer_mapping(
        self, layer_id: int, canonical: int, physical: int
    ) -> None:
        layer_id = int(layer_id)
        canonical = int(canonical)
        physical = int(physical)
        if canonical == physical:
            mapping = self._per_layer_canonical_to_physical.get(layer_id)
            reverse = self._per_layer_physical_to_canonical.get(layer_id)
            if mapping is None and reverse is None:
                return
            mapping = mapping or {}
            reverse = reverse or {}
            previous_physical = mapping.pop(canonical, None)
            if (
                previous_physical is not None
                and int(reverse.get(int(previous_physical), -1)) == canonical
            ):
                reverse.pop(int(previous_physical), None)
            previous_canonical = reverse.pop(physical, None)
            if (
                previous_canonical is not None
                and int(mapping.get(int(previous_canonical), -1)) == physical
            ):
                mapping.pop(int(previous_canonical), None)
            if mapping:
                self._per_layer_canonical_to_physical[layer_id] = mapping
                self._per_layer_physical_to_canonical[layer_id] = reverse
            else:
                self._per_layer_canonical_to_physical.pop(layer_id, None)
                self._per_layer_physical_to_canonical.pop(layer_id, None)
                self._per_layer_non_identity_mapping.discard(layer_id)
            return
        mapping = self._per_layer_canonical_to_physical.setdefault(layer_id, {})
        reverse = self._per_layer_physical_to_canonical.setdefault(layer_id, {})
        previous_physical = mapping.get(canonical)
        if previous_physical is not None and int(previous_physical) != physical:
            if int(reverse.get(int(previous_physical), -1)) == canonical:
                reverse.pop(int(previous_physical), None)
        previous_canonical = reverse.get(physical)
        if previous_canonical is not None and int(previous_canonical) != canonical:
            mapping.pop(int(previous_canonical), None)
        mapping[canonical] = physical
        reverse[physical] = canonical
        self._per_layer_non_identity_mapping.add(layer_id)

    def _translate_per_layer_locs(self, layer_id: int, loc: Any) -> Any:
        if self.config.kvc_backend != "per-layer-arena":
            return loc
        if int(layer_id) not in self._per_layer_non_identity_mapping:
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
        loc_set = set(loc_list)
        reverse = self._per_layer_physical_to_canonical.setdefault(int(layer_id), {})
        stale_canonicals: List[int] = []
        for physical in loc_set:
            canonical = reverse.get(int(physical))
            if (
                canonical is not None
                and int(canonical) != int(physical)
                and int(canonical) in loc_set
            ):
                stale_canonicals.append(int(canonical))
        for canonical in stale_canonicals:
            physical = mapping.pop(int(canonical), None)
            if physical is not None and int(reverse.get(int(physical), -1)) == int(
                canonical
            ):
                reverse.pop(int(physical), None)
        translated = [int(mapping.get(int(x), int(x))) for x in loc_list]
        size = int(getattr(self._kv_pool, "size", 0) or 0)
        if size > 0 and any(x <= 0 or x > size for x in translated):
            raise RuntimeError(
                f"LayerKV translated KV loc out of range for layer {layer_id}: "
                f"size={size} locs={translated[:8]}"
            )
        if len(set(translated)) != len(translated):
            duplicate_details: Dict[int, List[int]] = {}
            for original, physical in zip(loc_list, translated):
                bucket = duplicate_details.setdefault(int(physical), [])
                if len(bucket) < 8:
                    bucket.append(int(original))
            duplicate_details = {
                physical: originals
                for physical, originals in duplicate_details.items()
                if len(originals) > 1
            }
            raise RuntimeError(
                f"LayerKV translated KV locs contain duplicates for layer {layer_id}: "
                f"locs={translated[:16]} duplicate_details={duplicate_details}"
            )
        if translated == loc_list:
            return loc
        return torch.tensor(translated, dtype=loc.dtype, device=loc.device)

    def _wrap_set_kv_buffer(self, orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapped(layer: Any, loc: Any, cache_k: Any, cache_v: Any, *args, **kwargs):
            if not self._per_layer_kvc_io_control_active():
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
                return ret
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
            if not self._per_layer_kvc_io_control_active():
                return orig(layer_id, *args, **kwargs)
            self._prepare_per_layer_kvc_attention(layer_id)
            self._wait_for_kvc_layer_ready(layer_id)
            return orig(layer_id, *args, **kwargs)

        return wrapped

    def _wrap_get_value_buffer(self, orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapped(layer_id: int, *args, **kwargs):
            self.stats.kvc_get_value_count += 1
            if not self._per_layer_kvc_io_control_active():
                return orig(layer_id, *args, **kwargs)
            self._prepare_per_layer_kvc_attention(layer_id)
            self._wait_for_kvc_layer_ready(layer_id)
            return orig(layer_id, *args, **kwargs)

        return wrapped

    def _wrap_get_kv_buffer(self, orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapped(layer_id: int, *args, **kwargs):
            self.stats.kvc_get_kv_count += 1
            if not self._per_layer_kvc_io_control_active():
                return orig(layer_id, *args, **kwargs)
            self._prepare_per_layer_kvc_attention(layer_id)
            self._wait_for_kvc_layer_ready(layer_id)
            return orig(layer_id, *args, **kwargs)

        return wrapped
