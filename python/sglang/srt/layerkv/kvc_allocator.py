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
        self._ensure_per_layer_common_free_current()
        if len(self._per_layer_arena_common_free_locs) >= min_free_tokens:
            self.stats.kvc_per_layer_physical_arena_common_free_tokens = len(
                self._per_layer_arena_common_free_locs
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
        if len(free) < count and (
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
            allocated = self._per_layer_arena_allocated_locs.setdefault(
                layer_id, set()
            )
            loc_set = {int(x) for x in locs}
            reusable = loc_set if assume_allocated else loc_set.intersection(allocated)
            if not reusable:
                continue
            if assume_allocated:
                self._per_layer_arena_reserved_locs.update(reusable)
            allocated.difference_update(reusable)
            protected = self._per_layer_arena_protected_locs.setdefault(
                layer_id, set()
            )
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

    def _record_per_layer_mapping(
        self, layer_id: int, canonical: int, physical: int
    ) -> None:
        layer_id = int(layer_id)
        canonical = int(canonical)
        physical = int(physical)
        self._per_layer_canonical_to_physical.setdefault(layer_id, {})[
            canonical
        ] = physical
        if canonical != physical:
            self._per_layer_non_identity_mapping.add(layer_id)

    def _translate_per_layer_locs(
        self, layer_id: int, loc: Any
    ) -> Any:
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
