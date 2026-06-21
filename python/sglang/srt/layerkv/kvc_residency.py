"""LayerKV LayerKVKvcResidencyMixin implementation."""

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


class LayerKVKvcResidencyMixin:
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

    def _mark_active_req_per_layer_allocated(
        self, req_idx: int, keys: List[Tuple[int, int, int]]
    ) -> None:
        already_owned = int(req_idx) in self._per_layer_owned_req_indices
        self._mark_req_layerkv_owned(int(req_idx), keys)
        if already_owned:
            return
        marked = False
        for req in getattr(self._last_forward_batch, "reqs", []) or []:
            req_pool_idx = getattr(req, "req_pool_idx", None)
            if req_pool_idx is not None and int(req_pool_idx) == int(req_idx):
                setattr(req, "layerkv_per_layer_allocated", True)
                setattr(req, "skip_radix_cache_insert", True)
                marked = True
                break
        if marked:
            return
        scheduler = getattr(self._runner, "scheduler", None) if self._runner else None
        running_batch = getattr(scheduler, "running_batch", None)
        for req in getattr(running_batch, "reqs", []) or []:
            req_pool_idx = getattr(req, "req_pool_idx", None)
            if req_pool_idx is not None and int(req_pool_idx) == int(req_idx):
                setattr(req, "layerkv_per_layer_allocated", True)
                setattr(req, "skip_radix_cache_insert", True)
                break

    def _per_layer_slot_for_logical_token(
        self, layer_id: int, req_idx: int, pos: int
    ) -> Optional[int]:
        if self._req_to_token_pool is None:
            return None
        table = self._per_layer_req_to_token_overrides.get(int(layer_id))
        if table is None:
            table = getattr(self._req_to_token_pool, "req_to_token", None)
        if table is None:
            return None
        try:
            loc = int(table[int(req_idx), int(pos)].detach().item())
        except Exception:
            return None
        if loc <= 0:
            return None
        mapping = self._per_layer_canonical_to_physical.get(int(layer_id))
        if mapping:
            loc = int(mapping.get(int(loc), int(loc)))
        return loc if loc > 0 else None

    def _import_per_layer_arena_loc(self, layer_id: int, loc: int) -> None:
        """Register a native/canonical KV slot as a per-layer arena resident."""
        layer_id = int(layer_id)
        loc = int(loc)
        if loc <= 0:
            return
        self._per_layer_arena_reserved_locs.add(loc)
        self._per_layer_arena_common_free_locs.discard(loc)
        self._per_layer_arena_free_locs.setdefault(layer_id, [])
        self._per_layer_arena_allocated_locs.setdefault(layer_id, set()).add(loc)

    def _import_per_layer_arena_locs(self, layer_id: int, locs: List[int]) -> None:
        """Batch-register native/canonical KV slots as per-layer arena residents."""
        if not locs:
            return
        layer_id = int(layer_id)
        loc_set = {int(loc) for loc in locs if int(loc) > 0}
        if not loc_set:
            return
        self._per_layer_arena_reserved_locs.update(loc_set)
        self._per_layer_arena_common_free_locs.difference_update(loc_set)
        self._per_layer_arena_free_locs.setdefault(layer_id, [])
        self._per_layer_arena_allocated_locs.setdefault(layer_id, set()).update(loc_set)

    def _materialize_per_layer_resident_entry(
        self, layer_id: int, req_idx: int, pos: int
    ) -> Optional[_LayerKVResidencyEntry]:
        key = (int(layer_id), int(req_idx), int(pos))
        entry = self._per_layer_residency.get(key)
        if entry is not None:
            return entry
        loc = self._per_layer_slot_for_logical_token(
            int(layer_id), int(req_idx), int(pos)
        )
        if loc is None:
            return None
        self._import_per_layer_arena_loc(int(layer_id), int(loc))
        entry = _LayerKVResidencyEntry(
            req_idx=int(req_idx),
            pos=int(pos),
            state="resident",
            layer_id=int(layer_id),
            device_loc=int(loc),
            device_locs=[int(loc)],
            page_size=1,
            last_access_step=self._decode_step,
        )
        self._per_layer_residency[key] = entry
        self._per_layer_resident_token_count_fast += 1
        self._mark_active_req_per_layer_allocated(int(req_idx), [key])
        self._sync_kvc_group_if_needed(entry)
        return entry

    def register_native_kvc_runs_for_extend(
        self,
        *,
        reqs: List[Any],
        prefix_lens: List[int],
        seq_lens: List[int],
        locs: torch.Tensor,
    ) -> None:
        if not self._per_layer_allocator_enabled():
            return
        if locs is None or int(locs.numel()) <= 0:
            return
        try:
            loc_list = tuple(int(x) for x in locs.detach().cpu().tolist())
        except Exception:
            return
        offset = 0
        changed = False
        for req, prefix_len, seq_len in zip(reqs, prefix_lens, seq_lens):
            req_idx = getattr(req, "req_pool_idx", None)
            start = int(prefix_len)
            end = int(seq_len)
            count = max(0, end - start)
            req_locs = loc_list[offset : offset + count]
            offset += count
            if req_idx is None or count <= 0 or len(req_locs) != count:
                continue
            req_idx = int(req_idx)
            runs = self._native_kvc_runs_by_req.setdefault(req_idx, [])
            if runs and int(runs[-1].end) == start:
                previous = runs[-1]
                previous.end = end
                previous.locs = tuple(previous.locs) + tuple(int(x) for x in req_locs)
            else:
                runs.append(
                    _LayerKVNativeKvcRun(
                        req_idx=req_idx,
                        start=start,
                        end=end,
                        locs=tuple(int(x) for x in req_locs),
                    )
                )
            changed = True
        if changed:
            for req_idx in list(self._native_kvc_runs_by_req):
                self._native_kvc_runs_by_req[req_idx].sort(
                    key=lambda run: (int(run.start), int(run.end))
                )

    def _find_native_kvc_run(
        self, req_idx: int, pos: int, count: int
    ) -> Optional[_LayerKVNativeKvcRun]:
        runs = self._native_kvc_runs_by_req.get(int(req_idx))
        if not runs:
            return None
        pos = int(pos)
        count = int(count)
        for run in runs:
            if run.contains(pos, count):
                return run
            if int(run.start) > pos:
                break
        return None

    def _drop_native_kvc_runs_for_req(self, req_idx: int) -> None:
        self._native_kvc_runs_by_req.pop(int(req_idx), None)

    def _drop_offloaded_kvc_runs_for_req(self, req_idx: int) -> None:
        req_idx = int(req_idx)
        for key in list(self._per_layer_offloaded_runs_by_req_layer):
            if int(key[0]) == req_idx:
                self._per_layer_offloaded_runs_by_req_layer.pop(key, None)

    def _prune_native_kvc_runs_for_active_lengths(
        self, active_lens: Dict[int, int]
    ) -> None:
        for req_idx in list(self._native_kvc_runs_by_req):
            active_len = active_lens.get(int(req_idx))
            if active_len is None:
                continue
            active_len = int(active_len)
            pruned: List[_LayerKVNativeKvcRun] = []
            for run in self._native_kvc_runs_by_req.get(int(req_idx), []):
                if int(run.start) >= active_len:
                    continue
                if int(run.end) > active_len:
                    keep = max(0, active_len - int(run.start))
                    run.end = active_len
                    run.locs = tuple(run.locs[:keep])
                if int(run.end) > int(run.start) and run.locs:
                    pruned.append(run)
            if pruned:
                self._native_kvc_runs_by_req[int(req_idx)] = pruned
            else:
                self._native_kvc_runs_by_req.pop(int(req_idx), None)

    def _prune_offloaded_kvc_runs_for_active_lengths(
        self, active_lens: Dict[int, int]
    ) -> None:
        for key in list(self._per_layer_offloaded_runs_by_req_layer):
            req_idx, _layer_id = (int(key[0]), int(key[1]))
            active_len = active_lens.get(req_idx)
            if active_len is None:
                continue
            pruned: List[_LayerKVResidencyEntry] = []
            for run in self._per_layer_offloaded_runs_by_req_layer.get(key, []):
                if int(run.pos) >= int(active_len):
                    continue
                if int(run.pos) + int(run.token_count) > int(active_len):
                    keep = max(0, int(active_len) - int(run.pos))
                    run.page_size = keep
                    run.host_slots = run.host_slot_list()[:keep]
                    if run.evicted_device_locs:
                        run.evicted_device_locs = run.evicted_device_locs[:keep]
                if int(run.token_count) > 0:
                    pruned.append(run)
            if pruned:
                self._per_layer_offloaded_runs_by_req_layer[key] = pruned
            else:
                self._per_layer_offloaded_runs_by_req_layer.pop(key, None)

    def _materialize_per_layer_resident_segment(
        self,
        layer_id: int,
        req_idx: int,
        pos: int,
        count: int,
        *,
        import_arena: bool = True,
    ) -> Optional[List[_LayerKVResidencyEntry]]:
        layer_id = int(layer_id)
        req_idx = int(req_idx)
        pos = int(pos)
        count = int(count)
        if count <= 0:
            return []
        native_run = self._find_native_kvc_run(req_idx, pos, count)
        native_locs = native_run.loc_slice(pos, count) if native_run is not None else ()
        if native_locs and len(native_locs) != count:
            return None
        start_key = (layer_id, req_idx, pos)
        if (
            native_locs
            and self._optimized_profile_enabled()
            and self._per_layer_allocator_enabled()
            and start_key not in self._per_layer_residency
        ):
            mapping = self._per_layer_canonical_to_physical.setdefault(layer_id, {})
            if mapping:
                locs: List[int] = []
                for native_loc in native_locs:
                    native_loc = int(native_loc)
                    loc = int(mapping.get(native_loc, native_loc))
                    if loc <= 0:
                        return None
                    locs.append(loc)
            else:
                locs = list(native_locs)
                if not locs:
                    return None
            if import_arena:
                self._import_per_layer_arena_locs(layer_id, locs)
            entry = _LayerKVResidencyEntry(
                req_idx=req_idx,
                pos=pos,
                state="resident",
                layer_id=layer_id,
                device_loc=locs[0],
                device_locs=locs,
                page_size=count,
                last_access_step=self._decode_step,
            )
            self._per_layer_residency[start_key] = entry
            self._per_layer_resident_token_count_fast += count
            self._mark_active_req_per_layer_allocated(req_idx, [start_key])
            self._sync_kvc_group_if_needed(entry)
            return [entry]
        range_keys = [
            (layer_id, req_idx, int(token_pos)) for token_pos in range(pos, pos + count)
        ]
        existing_start = self._per_layer_residency.get(range_keys[0])
        if (
            existing_start is not None
            and existing_start.state == "resident"
            and int(existing_start.token_count) == count
        ):
            locs = existing_start.device_loc_list()
            if len(locs) == count and all(int(loc) > 0 for loc in locs):
                existing_start.last_access_step = self._decode_step
                return [existing_start]
        mapping = self._per_layer_canonical_to_physical.setdefault(layer_id, {})
        segment_locs: List[int] = []
        missing_keys: List[Tuple[int, int, int]] = []
        can_segment = True
        for offset, key in enumerate(range_keys):
            entry = self._per_layer_residency.get(key)
            if entry is None:
                if not native_locs:
                    can_segment = False
                    break
                native_loc = int(native_locs[offset])
                loc = int(mapping.get(native_loc, native_loc))
                if loc <= 0:
                    can_segment = False
                    break
                segment_locs.append(loc)
                missing_keys.append(key)
                continue
            if entry.state != "resident" or int(entry.token_count) != 1:
                can_segment = False
                break
            locs = entry.device_loc_list()
            if len(locs) != 1 or int(locs[0]) <= 0:
                can_segment = False
                break
            segment_locs.append(int(locs[0]))
        if can_segment and len(segment_locs) == count:
            segment = self._per_layer_residency.get(range_keys[0])
            if segment is None:
                segment = _LayerKVResidencyEntry(
                    req_idx=req_idx,
                    pos=pos,
                    state="resident",
                    layer_id=layer_id,
                    device_loc=segment_locs[0],
                    device_locs=segment_locs,
                    page_size=count,
                    last_access_step=self._decode_step,
                )
                self._per_layer_residency[range_keys[0]] = segment
                for key in range_keys[1:]:
                    self._per_layer_residency.pop(key, None)
                if missing_keys and import_arena:
                    self._import_per_layer_arena_locs(layer_id, segment_locs)
                segment.pos = pos
                segment.page_size = count
            segment.device_loc = segment_locs[0]
            segment.device_locs = segment_locs
            segment.last_access_step = self._decode_step
            if missing_keys:
                self._per_layer_resident_token_count_fast += len(missing_keys)
                self._mark_active_req_per_layer_allocated(req_idx, [range_keys[0]])
                self._sync_kvc_group_if_needed(segment)
            return [segment]
        if not any(key in self._per_layer_residency for key in range_keys):
            if not native_locs:
                return None
            if mapping:
                locs: List[int] = []
                for native_loc in native_locs:
                    native_loc = int(native_loc)
                    loc = int(mapping.get(native_loc, native_loc))
                    if loc <= 0:
                        return None
                    locs.append(loc)
            else:
                locs = list(native_locs)
                if not locs:
                    return None
            if import_arena:
                self._import_per_layer_arena_locs(layer_id, locs)
            entry = _LayerKVResidencyEntry(
                req_idx=req_idx,
                pos=pos,
                state="resident",
                layer_id=layer_id,
                device_loc=locs[0],
                device_locs=locs,
                page_size=count,
                last_access_step=self._decode_step,
            )
            self._per_layer_residency[range_keys[0]] = entry
            self._per_layer_resident_token_count_fast += count
            self._mark_active_req_per_layer_allocated(req_idx, [range_keys[0]])
            self._sync_kvc_group_if_needed(entry)
            return [entry]
        entries: List[_LayerKVResidencyEntry] = []
        created_keys: List[Tuple[int, int, int]] = []
        mapping = self._per_layer_canonical_to_physical.setdefault(layer_id, {})
        for offset, token_pos in enumerate(range(pos, pos + count)):
            key = (layer_id, req_idx, int(token_pos))
            entry = self._per_layer_residency.get(key)
            if entry is None:
                if not native_locs:
                    return None
                native_loc = int(native_locs[offset])
                loc = int(mapping.get(native_loc, native_loc))
                if loc <= 0:
                    return None
                if import_arena:
                    self._import_per_layer_arena_loc(layer_id, loc)
                entry = _LayerKVResidencyEntry(
                    req_idx=req_idx,
                    pos=int(token_pos),
                    state="resident",
                    layer_id=layer_id,
                    device_loc=loc,
                    device_locs=[loc],
                    page_size=1,
                    last_access_step=self._decode_step,
                )
                self._per_layer_residency[key] = entry
                self._per_layer_resident_token_count_fast += 1
                created_keys.append(key)
                self._sync_kvc_group_if_needed(entry)
            if entry.state != "resident" or int(entry.token_count) != 1:
                return None
            locs = entry.device_loc_list()
            if len(locs) != 1 or int(locs[0]) <= 0:
                return None
            entry.last_access_step = self._decode_step
            entries.append(entry)
        if created_keys:
            self._mark_active_req_per_layer_allocated(req_idx, created_keys)
        return entries

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
                self._record_per_layer_mapping(layer_id, int(canonical), int(physical))
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
                self._record_per_layer_mapping(layer_id, int(canonical), int(physical))
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
        if for_prefill:
            self._ensure_per_layer_common_free_current()
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
            stream.wait_event(event)
            if wait_end is not None:
                try:
                    wait_end.record(stream)
                except Exception:
                    wait_end = None
            self.stats.scheduler_exposed_wait_ms += (
                time.perf_counter() - t_wait
            ) * 1000.0
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

    def _track_per_layer_offloaded_keys_batch(
        self, entries: List[_LayerKVResidencyEntry]
    ) -> None:
        if not entries:
            return
        tokens_by_layer: Dict[int, int] = {}
        dirty_reqs: Set[int] = set()
        dirty_req_layers: Set[Tuple[int, int]] = set()
        changed = False
        for entry in entries:
            layer_id = int(entry.layer_id)
            req_idx = int(entry.req_idx)
            pos = int(entry.pos)
            norm_key = (layer_id, req_idx, pos)
            if norm_key in self._per_layer_offloaded_keys:
                continue
            self._per_layer_offloaded_keys.add(norm_key)
            self._per_layer_offloaded_keys_by_req.setdefault(req_idx, set()).add(
                norm_key
            )
            self._per_layer_offloaded_keys_by_req_layer.setdefault(
                (req_idx, layer_id), set()
            ).add(norm_key)
            tokens = int(entry.token_count)
            tokens_by_layer[layer_id] = int(tokens_by_layer.get(layer_id, 0)) + tokens
            dirty_reqs.add(req_idx)
            dirty_req_layers.add((req_idx, layer_id))
            changed = True
        if not changed:
            return
        for layer_id, tokens in tokens_by_layer.items():
            self._per_layer_offloaded_token_count_by_layer[layer_id] = max(
                0,
                int(self._per_layer_offloaded_token_count_by_layer.get(layer_id, 0))
                + int(tokens),
            )
        self._per_layer_offloaded_version += 1
        self._per_layer_offloaded_dirty_reqs.update(dirty_reqs)
        self._per_layer_offloaded_dirty_req_layers.update(dirty_req_layers)
        for req_idx in dirty_reqs:
            self._per_layer_offloaded_sorted_by_req.pop(int(req_idx), None)
        for req_idx, layer_id in dirty_req_layers:
            self._per_layer_offloaded_sorted_by_req_layer.pop(
                (int(req_idx), int(layer_id)), None
            )
            self._per_layer_offloaded_positions_by_req_layer.pop(
                (int(req_idx), int(layer_id)), None
            )

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

    def _track_per_layer_offloaded_runs(
        self, entries: List[_LayerKVResidencyEntry]
    ) -> None:
        if not entries:
            return
        grouped: Dict[Tuple[int, int], List[_LayerKVResidencyEntry]] = {}
        for entry in sorted(
            entries,
            key=lambda item: (int(item.layer_id), int(item.req_idx), int(item.pos)),
        ):
            if entry.state != "offloaded":
                continue
            host_slots = entry.host_slot_list()
            if not host_slots:
                continue
            layer_id = int(entry.layer_id)
            req_idx = int(entry.req_idx)
            key = (req_idx, layer_id)
            runs = grouped.setdefault(key, [])
            device_locs = (
                [int(x) for x in entry.evicted_device_locs]
                if entry.evicted_device_locs
                else []
            )
            if (
                runs
                and int(runs[-1].pos) + int(runs[-1].token_count) == int(entry.pos)
            ):
                previous = runs[-1]
                previous.page_size = int(previous.token_count) + int(entry.token_count)
                previous.host_slots = previous.host_slot_list() + host_slots
                if device_locs:
                    previous.evicted_device_locs = (
                        [int(x) for x in previous.evicted_device_locs]
                        if previous.evicted_device_locs
                        else []
                    ) + device_locs
            else:
                runs.append(
                    _LayerKVResidencyEntry(
                        req_idx=req_idx,
                        pos=int(entry.pos),
                        state="offloaded",
                        layer_id=layer_id,
                        host_slot=int(host_slots[0]),
                        host_slots=[int(x) for x in host_slots],
                        evicted_device_locs=device_locs or None,
                        page_size=int(entry.token_count),
                        last_access_step=self._decode_step,
                    )
                )
        for key, new_runs in grouped.items():
            runs = self._per_layer_offloaded_runs_by_req_layer.setdefault(key, [])
            runs.extend(new_runs)
            runs.sort(key=lambda item: int(item.pos))
            merged: List[_LayerKVResidencyEntry] = []
            for run in runs:
                if (
                    merged
                    and int(merged[-1].pos) + int(merged[-1].token_count)
                    == int(run.pos)
                    and merged[-1].state == "offloaded"
                    and run.state == "offloaded"
                ):
                    previous = merged[-1]
                    previous.page_size = int(previous.token_count) + int(run.token_count)
                    previous.host_slots = previous.host_slot_list() + run.host_slot_list()
                    if run.evicted_device_locs:
                        previous.evicted_device_locs = (
                            [int(x) for x in previous.evicted_device_locs]
                            if previous.evicted_device_locs
                            else []
                        ) + [int(x) for x in run.evicted_device_locs]
                else:
                    merged.append(run)
            self._per_layer_offloaded_runs_by_req_layer[key] = merged

    def _select_offloaded_run_entries_for_virtual_layer(
        self, layer_id: int, active_lens: Dict[int, int]
    ) -> List[_LayerKVResidencyEntry]:
        layer_id = int(layer_id)
        selected: List[_LayerKVResidencyEntry] = []
        scanned = 0
        for req_idx, required_prefix_len in active_lens.items():
            runs = self._per_layer_offloaded_runs_by_req_layer.get(
                (int(req_idx), layer_id)
            )
            if not runs:
                continue
            self.stats.kvc_layerwise_required_index_hit_count += 1
            for run in runs:
                scanned += 1
                if run.state != "offloaded" or int(run.token_count) <= 0:
                    continue
                if int(run.pos) >= int(required_prefix_len):
                    break
                if int(run.pos) + int(run.token_count) <= int(required_prefix_len):
                    selected.append(run)
                    continue
                keep = max(0, int(required_prefix_len) - int(run.pos))
                if keep <= 0:
                    continue
                host_slots = run.host_slot_list()[:keep]
                selected.append(
                    _LayerKVResidencyEntry(
                        req_idx=int(run.req_idx),
                        pos=int(run.pos),
                        state="offloaded",
                        layer_id=layer_id,
                        host_slot=int(host_slots[0]) if host_slots else None,
                        host_slots=host_slots,
                        evicted_device_locs=(
                            [int(x) for x in run.evicted_device_locs[:keep]]
                            if run.evicted_device_locs
                            else None
                        ),
                        page_size=keep,
                        last_access_step=self._decode_step,
                    )
                )
        self.stats.kvc_layerwise_required_index_scan_count += scanned
        self.stats.kvc_required_scanned_keys += scanned
        return selected

    def _remove_per_layer_offloaded_run_range(
        self, layer_id: int, req_idx: int, pos: int, count: int
    ) -> None:
        key = (int(req_idx), int(layer_id))
        runs = self._per_layer_offloaded_runs_by_req_layer.get(key)
        if not runs:
            return
        remove_start = int(pos)
        remove_end = remove_start + max(0, int(count))
        if remove_end <= remove_start:
            return
        next_runs: List[_LayerKVResidencyEntry] = []
        for run in runs:
            run_start = int(run.pos)
            run_end = run_start + int(run.token_count)
            if remove_end <= run_start or remove_start >= run_end:
                next_runs.append(run)
                continue
            host_slots = run.host_slot_list()
            evicted_locs = (
                [int(x) for x in run.evicted_device_locs]
                if run.evicted_device_locs
                else []
            )
            if remove_start > run_start:
                left_len = remove_start - run_start
                next_runs.append(
                    _LayerKVResidencyEntry(
                        req_idx=int(run.req_idx),
                        pos=run_start,
                        state="offloaded",
                        layer_id=int(run.layer_id),
                        host_slot=int(host_slots[0]) if host_slots else None,
                        host_slots=host_slots[:left_len],
                        evicted_device_locs=(
                            evicted_locs[:left_len] if evicted_locs else None
                        ),
                        page_size=left_len,
                        last_access_step=int(run.last_access_step),
                    )
                )
            if remove_end < run_end:
                right_offset = remove_end - run_start
                right_slots = host_slots[right_offset:]
                next_runs.append(
                    _LayerKVResidencyEntry(
                        req_idx=int(run.req_idx),
                        pos=remove_end,
                        state="offloaded",
                        layer_id=int(run.layer_id),
                        host_slot=int(right_slots[0]) if right_slots else None,
                        host_slots=right_slots,
                        evicted_device_locs=(
                            evicted_locs[right_offset:] if evicted_locs else None
                        ),
                        page_size=run_end - remove_end,
                        last_access_step=int(run.last_access_step),
                    )
                )
        if next_runs:
            self._per_layer_offloaded_runs_by_req_layer[key] = next_runs
        else:
            self._per_layer_offloaded_runs_by_req_layer.pop(key, None)

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
        self._per_layer_offloaded_runs_by_req_layer.clear()
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
