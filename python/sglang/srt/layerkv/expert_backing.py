"""LayerKV LayerKVExpertBackingMixin implementation."""

from __future__ import annotations

import functools
import heapq
import json
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

if __package__:
    from .common_types import (
        _LayerKVExpertCopyDescriptor,
        _LayerKVExpertInstallBuild,
        _LayerKVExpertInstallD2HJob,
        _LayerKVExpertInstallItem,
        _LayerKVExpertLayerState,
        _LayerKVPendingExpertCopy,
        _LayerKVPendingExpertD2H,
    )
else:  # pragma: no cover - direct file-loading smoke tests.
    from common_types import (
        _LayerKVExpertCopyDescriptor,
        _LayerKVExpertInstallBuild,
        _LayerKVExpertInstallD2HJob,
        _LayerKVExpertInstallItem,
        _LayerKVExpertLayerState,
        _LayerKVPendingExpertCopy,
        _LayerKVPendingExpertD2H,
    )

logger = logging.getLogger(__name__)

try:
    from sglang.jit_kernel.layerkv_expert_remap import layerkv_expert_remap
except Exception:  # pragma: no cover - optional JIT helper.
    layerkv_expert_remap = None


class LayerKVExpertBackingMixin:
    def _get_expert_batch_transfer(self):
        if self.config.expert_transfer_backend != "cuda-batch":
            return None
        if self._expert_batch_transfer is None:
            from .expert_transfer import ExpertBatchTransfer

            self._expert_batch_transfer = ExpertBatchTransfer(self.stats)
        return self._expert_batch_transfer

    def _build_expert_batch_backup(self, state, evicted_pairs):
        """Allocate destinations without a GPU gather or GPU index tensors."""
        copied = {int(logical): {} for logical, _slot in evicted_pairs}
        pairs = []
        individual = self.config.expert_batch_backing_layout == "individual"
        for name in state.param_names:
            param = getattr(state.module, name).data
            batch = None
            chunk_size = max(1, (256 * 1024 * 1024) // max(1, param[0].nbytes))
            for offset, (logical, slot) in enumerate(evicted_pairs):
                if not individual and offset % chunk_size == 0:
                    batch = self._alloc_cpu_backing_tensor(
                        (min(chunk_size, len(evicted_pairs) - offset),)
                        + tuple(param.shape[1:]),
                        param.dtype,
                    )
                dst = (
                    self._alloc_cpu_backing_tensor(tuple(param.shape[1:]), param.dtype)
                    if individual
                    else batch[offset % chunk_size]
                )
                if not individual:
                    self._register_cpu_backing_batch_view(batch, dst)
                copied[int(logical)][name] = dst
                pairs.append((dst, param[int(slot)].detach()))
        return copied, pairs, sum(dst.nbytes for dst, _src in pairs)

    def _record_expert_copy_descriptor(
        self,
        *,
        direction: str,
        reason: str,
        layer_id: int,
        logical_id: int,
        src_slot: int,
        dst_slot: int,
        nbytes: int,
        param_count: int,
    ) -> _LayerKVExpertCopyDescriptor:
        self._expert_copy_descriptor_seq += 1
        desc = _LayerKVExpertCopyDescriptor(
            seq=int(self._expert_copy_descriptor_seq),
            direction=str(direction),
            reason=str(reason),
            layer_id=int(layer_id),
            logical_id=int(logical_id),
            src_slot=int(src_slot),
            dst_slot=int(dst_slot),
            bytes=int(nbytes),
            param_count=int(param_count),
            step=int(self._decode_step),
        )
        self._expert_copy_descriptors_recent.append(desc)
        if len(self._expert_copy_descriptors_recent) > 256:
            self._expert_copy_descriptors_recent = (
                self._expert_copy_descriptors_recent[-256:]
            )
        self.stats.expert_copy_descriptor_count += 1
        self.stats.expert_copy_descriptor_bytes += int(nbytes)
        self.stats.expert_copy_descriptor_param_count += int(param_count)
        if str(direction) == "D2H":
            self.stats.expert_copy_descriptor_d2h_count += 1
        elif str(direction) == "H2D":
            self.stats.expert_copy_descriptor_h2d_count += 1
        if str(reason).startswith("install"):
            self.stats.expert_copy_descriptor_install_count += 1
        elif str(reason).startswith("evict"):
            self.stats.expert_copy_descriptor_evict_count += 1
        elif str(reason).startswith("materialize"):
            self.stats.expert_copy_descriptor_materialize_count += 1
        return desc

    @staticmethod
    def _contiguous_int_runs(values: List[int]) -> List[List[int]]:
        if not values:
            return []
        ordered = sorted(dict.fromkeys(int(x) for x in values))
        runs: List[List[int]] = []
        current: List[int] = [ordered[0]]
        for value in ordered[1:]:
            if int(value) == int(current[-1]) + 1:
                current.append(int(value))
            else:
                runs.append(current)
                current = [int(value)]
        runs.append(current)
        return runs

    def _expert_cpu_backing_pool_key(
        self, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> Tuple[torch.dtype, Tuple[int, ...]]:
        return (dtype, tuple(int(x) for x in shape))

    def _alloc_cpu_backing_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        *,
        pin_memory: bool = True,
    ) -> torch.Tensor:
        shape = tuple(int(x) for x in shape)
        if pin_memory:
            key = self._expert_cpu_backing_pool_key(shape, dtype)
            pool = self._expert_cpu_backing_pool.get(key)
            if pool:
                tensor = pool.pop()
                self._expert_cpu_backing_pool_bytes -= int(tensor.nbytes)
                self.stats.expert_cpu_backing_pool_reuse_count += 1
                self.stats.expert_cpu_backing_pool_bytes = int(
                    self._expert_cpu_backing_pool_bytes
                )
                return tensor
        self.stats.expert_cpu_backing_pool_alloc_count += 1
        return torch.empty(
            shape,
            dtype=dtype,
            device="cpu",
            pin_memory=pin_memory,
        )

    def _register_cpu_backing_batch_view(
        self, owner: Optional[torch.Tensor], view: torch.Tensor
    ) -> None:
        """Retain a batch backing owner until its last row view is released.

        ``cudaMemcpyBatchAsync`` retains the storage owner through the copy
        event, but the residency map only retains individual row views. A
        view is intentionally not admitted to the ordinary tensor pool: it
        would alias the remaining rows. Track the view explicitly so the
        owner can enter the pool as one complete, reusable allocation after
        all logical experts and in-flight copies release it.
        """
        if owner is None or getattr(view, "_base", None) is None:
            return
        try:
            owner_key = int(owner.untyped_storage().data_ptr())
            view_id = id(view)
            record = self._expert_cpu_backing_batch_owners.get(owner_key)
            if record is None:
                record = {"tensor": owner, "view_ids": set()}
                self._expert_cpu_backing_batch_owners[owner_key] = record
            if view_id in self._expert_cpu_backing_batch_view_ids:
                return
            record["view_ids"].add(view_id)
            self._expert_cpu_backing_batch_view_ids[view_id] = owner_key
            self.stats.expert_cpu_backing_batch_view_register_count += 1
        except Exception:
            # Pool safety is optional; a failed registration must not make a
            # potentially unsafe view look like an owning tensor.
            return

    def _release_cpu_backing_batch_view(self, tensor: torch.Tensor) -> bool:
        """Release one registered view and possibly its complete owner."""
        view_id = id(tensor)
        owner_key = self._expert_cpu_backing_batch_view_ids.pop(view_id, None)
        if owner_key is None:
            return False
        record = self._expert_cpu_backing_batch_owners.get(owner_key)
        if record is None:
            return True
        view_ids = record["view_ids"]
        view_ids.discard(view_id)
        self.stats.expert_cpu_backing_batch_view_release_count += 1
        if view_ids:
            return True
        self._expert_cpu_backing_batch_owners.pop(owner_key, None)
        self.stats.expert_cpu_backing_batch_owner_reclaim_count += 1
        self._release_cpu_backing_tensor(record["tensor"])
        return True

    @staticmethod
    def _tensor_owns_storage(tensor: torch.Tensor) -> bool:
        try:
            if getattr(tensor, "_base", None) is not None:
                return False
            if int(tensor.storage_offset()) != 0:
                return False
            return int(tensor.untyped_storage().nbytes()) == int(tensor.nbytes)
        except Exception:
            return False

    def _release_cpu_backing_tensor(self, tensor: torch.Tensor) -> None:
        try:
            if self._release_cpu_backing_batch_view(tensor):
                return
            at_admission = self.config.expert_backing_release_validation == "admission"
            transfer = self._expert_batch_transfer
            if at_admission:
                # Cheap rejection keeps ordinary batched views off the deferred
                # queue. Full validation is still required at actual admission,
                # against the CURRENT tensor metadata and pool capacity.
                if (
                    tensor.device.type != "cpu"
                    or getattr(tensor, "_base", None) is not None
                ):
                    self.stats.expert_cpu_backing_pool_drop_count += 1
                    return
                if transfer is not None and transfer.defer_release(
                    tensor, self._release_cpu_backing_tensor
                ):
                    return
            if tensor.device.type != "cpu" or not self._tensor_owns_storage(tensor):
                self.stats.expert_cpu_backing_pool_drop_count += 1
                return
            if hasattr(tensor, "is_pinned") and not bool(tensor.is_pinned()):
                self.stats.expert_cpu_backing_pool_drop_count += 1
                return
            if (
                not at_admission
                and transfer is not None
                and transfer.defer_release(tensor, self._release_cpu_backing_tensor)
            ):
                return
            nbytes = int(tensor.nbytes)
            if self._expert_cpu_backing_pool_bytes + nbytes > int(
                self._expert_cpu_backing_pool_limit_bytes
            ):
                self.stats.expert_cpu_backing_pool_drop_count += 1
                return
            key = self._expert_cpu_backing_pool_key(tuple(tensor.shape), tensor.dtype)
            self._expert_cpu_backing_pool.setdefault(key, []).append(tensor.detach())
            self._expert_cpu_backing_pool_bytes += nbytes
            self.stats.expert_cpu_backing_pool_release_count += 1
            self.stats.expert_cpu_backing_pool_bytes = int(
                self._expert_cpu_backing_pool_bytes
            )
            self.stats.expert_cpu_backing_pool_limit_bytes = int(
                self._expert_cpu_backing_pool_limit_bytes
            )
        except Exception:
            self.stats.expert_cpu_backing_pool_drop_count += 1

    def _release_expert_backing_params(
        self, params: Optional[Dict[str, torch.Tensor]]
    ) -> None:
        if not params:
            return
        for tensor in params.values():
            self._release_cpu_backing_tensor(tensor)

    def _copy_tensor_to_cpu_backing_pooled(
        self, tensor: torch.Tensor, *, non_blocking: bool, pin_memory: bool = True
    ) -> torch.Tensor:
        if not pin_memory:
            return tensor.to("cpu", copy=True)
        try:
            dst = self._alloc_cpu_backing_tensor(
                tuple(tensor.shape),
                tensor.dtype,
                pin_memory=True,
            )
            dst.copy_(tensor, non_blocking=non_blocking)
            return dst
        except Exception:
            return tensor.to("cpu", copy=True)

    def _expert_d2h_copy_stream(self, device: torch.device) -> Any:
        return (
            self._expert_d2h_stream
            or self._copy_stream
            or torch.cuda.current_stream(device=device)
        )

    def _expert_h2d_copy_stream(self, device: torch.device) -> Any:
        return (
            self._expert_h2d_stream
            or self._copy_stream
            or torch.cuda.current_stream(device=device)
        )

    def _expert_h2d_async_enabled(self, state: _LayerKVExpertLayerState) -> bool:
        return (
            self._optimized_profile_enabled()
            and state.device.type == "cuda"
            and self._expert_h2d_stream is not None
        )

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
            name: self._copy_tensor_to_cpu_backing_pooled(
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
        return (
            self._expert_global_cpu_backing.get((int(layer_id), int(expert_id)))
            is params
        )

    def _copy_experts_to_cpu_for_install_batched(
        self,
        module: Any,
        param_names: List[str],
        expert_ids: List[int],
        *,
        layer_id: int = -1,
        account_host_backing: bool = True,
        pin_memory: bool = True,
        reason: str = "install_backing",
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        if not expert_ids:
            return {}
        copied: Dict[int, Dict[str, torch.Tensor]] = {
            int(expert_id): {} for expert_id in expert_ids
        }
        copied_bytes = 0
        device = getattr(module, param_names[0]).data.device if param_names else None
        with self._profile("profile_copy_expert_to_cpu_ms"):
            with torch.no_grad():
                # Partial consumption of a batched view retains discarded rows.
                # Honor individual storage from installation through eviction.
                use_tensor_batch = (
                    len(expert_ids) > 1
                    and self.config.expert_batch_backing_layout != "individual"
                )
                if use_tensor_batch:
                    for name in param_names:
                        param = getattr(module, name).data
                        max_chunk_bytes = 256 * 1024 * 1024
                        per_expert_bytes = max(1, int(param[0].nbytes))
                        chunk_size = max(1, max_chunk_bytes // per_expert_bytes)
                        runs = self._contiguous_int_runs(expert_ids)
                        singleton_ids: List[int] = []
                        for run in runs:
                            if len(run) <= 1:
                                singleton_ids.extend(run)
                                continue
                            for begin in range(0, len(run), chunk_size):
                                chunk = run[begin : begin + chunk_size]
                                selected = param[
                                    int(chunk[0]) : int(chunk[0]) + len(chunk)
                                ].detach()
                                try:
                                    dst = self._alloc_cpu_backing_tensor(
                                        tuple(selected.shape),
                                        selected.dtype,
                                        pin_memory=pin_memory,
                                    )
                                    dst.copy_(selected, non_blocking=True)
                                except Exception:
                                    dst = selected.to("cpu", copy=True)
                                copied_bytes += int(dst.nbytes)
                                self.stats.expert_d2h_slice_run_count += 1
                                self.stats.expert_d2h_slice_expert_count += len(chunk)
                                for offset, expert_id in enumerate(chunk):
                                    view = dst[offset]
                                    self._register_cpu_backing_batch_view(dst, view)
                                    copied[int(expert_id)][name] = view
                        for begin in range(0, len(singleton_ids), chunk_size):
                            chunk = singleton_ids[begin : begin + chunk_size]
                            if not chunk:
                                continue
                            idx = torch.tensor(
                                chunk,
                                dtype=torch.long,
                                device=param.device,
                            )
                            selected = param.index_select(0, idx).detach()
                            try:
                                dst = self._alloc_cpu_backing_tensor(
                                    tuple(selected.shape),
                                    selected.dtype,
                                    pin_memory=pin_memory,
                                )
                                dst.copy_(selected, non_blocking=True)
                            except Exception:
                                dst = selected.to("cpu", copy=True)
                            copied_bytes += int(dst.nbytes)
                            self.stats.expert_d2h_gather_batch_count += 1
                            self.stats.expert_d2h_gather_expert_count += len(chunk)
                            for offset, expert_id in enumerate(chunk):
                                view = dst[offset]
                                self._register_cpu_backing_batch_view(dst, view)
                                copied[int(expert_id)][name] = view
                else:
                    for name in param_names:
                        param = getattr(module, name).data
                        for expert_id in expert_ids:
                            tensor = param[int(expert_id)].detach()
                            copied[int(expert_id)][name] = (
                                self._copy_tensor_to_cpu_backing_pooled(
                                    tensor,
                                    non_blocking=True,
                                    pin_memory=pin_memory,
                                )
                            )
                if device is not None and device.type == "cuda":
                    torch.cuda.current_stream(device=device).synchronize()
        if account_host_backing:
            self._expert_host_backing_bytes += sum(
                self._expert_backing_bytes(params) for params in copied.values()
            )
        for expert_id, params in copied.items():
            self._record_expert_copy_descriptor(
                direction="D2H",
                reason=str(reason),
                layer_id=int(layer_id),
                logical_id=int(expert_id),
                src_slot=int(expert_id),
                dst_slot=int(expert_id),
                nbytes=self._expert_backing_bytes(params),
                param_count=len(params),
            )
        if account_host_backing:
            self._refresh_expert_host_backing_stat()
        return copied

    def _copy_experts_to_cpu_for_install_batched_async(
        self,
        module: Any,
        param_names: List[str],
        expert_ids: List[int],
        *,
        layer_id: int = -1,
        target_cpu_params: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        reason: str = "install_backing_async",
        mirror_cpu_params: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ) -> int:
        if not expert_ids or not param_names:
            return 0
        device = getattr(module, param_names[0]).data.device
        if device.type != "cuda":
            return -1
        active_stream = self._expert_d2h_copy_stream(device)
        expert_ids = [int(expert_id) for expert_id in expert_ids]
        copied: Dict[int, Dict[str, torch.Tensor]] = {
            int(expert_id): {} for expert_id in expert_ids
        }
        copied_bytes = 0
        source_refs: List[torch.Tensor] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        try:
            with self._profile("profile_copy_expert_to_cpu_ms"):
                with torch.no_grad(), torch.cuda.stream(active_stream):
                    start.record(active_stream)
                    for name in param_names:
                        param = getattr(module, name).data
                        if self.config.expert_batch_backing_layout == "individual":
                            for expert_id in expert_ids:
                                src = param[int(expert_id)].detach()
                                dst = self._alloc_cpu_backing_tensor(
                                    tuple(src.shape), src.dtype, pin_memory=True
                                )
                                dst.copy_(src, non_blocking=True)
                                copied[int(expert_id)][name] = dst
                                copied_bytes += int(dst.nbytes)
                                source_refs.append(src)
                            continue
                        max_chunk_bytes = 256 * 1024 * 1024
                        per_expert_bytes = max(1, int(param[0].nbytes))
                        chunk_size = max(1, max_chunk_bytes // per_expert_bytes)
                        runs = self._contiguous_int_runs(expert_ids)
                        singleton_ids: List[int] = []
                        for run in runs:
                            if len(run) <= 1:
                                singleton_ids.extend(run)
                                continue
                            for begin in range(0, len(run), chunk_size):
                                chunk = run[begin : begin + chunk_size]
                                selected = param[
                                    int(chunk[0]) : int(chunk[0]) + len(chunk)
                                ].detach()
                                dst = self._alloc_cpu_backing_tensor(
                                    tuple(selected.shape),
                                    selected.dtype,
                                    pin_memory=True,
                                )
                                dst.copy_(selected, non_blocking=True)
                                copied_bytes += int(dst.nbytes)
                                source_refs.append(selected)
                                self.stats.expert_d2h_slice_run_count += 1
                                self.stats.expert_d2h_slice_expert_count += len(chunk)
                                for offset, expert_id in enumerate(chunk):
                                    view = dst[offset]
                                    self._register_cpu_backing_batch_view(dst, view)
                                    copied[int(expert_id)][name] = view
                        for begin in range(0, len(singleton_ids), chunk_size):
                            chunk = singleton_ids[begin : begin + chunk_size]
                            if not chunk:
                                continue
                            idx = torch.tensor(chunk, dtype=torch.long, device=param.device)
                            selected = param.index_select(0, idx).detach()
                            dst = self._alloc_cpu_backing_tensor(
                                tuple(selected.shape),
                                selected.dtype,
                                pin_memory=True,
                            )
                            dst.copy_(selected, non_blocking=True)
                            copied_bytes += int(dst.nbytes)
                            self.stats.expert_d2h_gather_batch_count += 1
                            self.stats.expert_d2h_gather_expert_count += len(chunk)
                            source_refs.extend([idx, selected])
                            for offset, expert_id in enumerate(chunk):
                                view = dst[offset]
                                self._register_cpu_backing_batch_view(dst, view)
                                copied[int(expert_id)][name] = view
                    end.record(active_stream)
        except Exception:
            try:
                active_stream.synchronize()
            except Exception:
                pass
            self.stats.expert_install_d2h_sync_fallback_count += len(expert_ids)
            return -1
        pending = _LayerKVPendingExpertD2H(
            start_event=start,
            ready_event=end,
            layer_id=int(layer_id),
            copied=copied,
            bytes=int(copied_bytes),
            reason=str(reason),
            source_refs=tuple(source_refs),
            target_cpu_params=target_cpu_params,
            mirror_cpu_params=mirror_cpu_params,
        )
        self._pending_expert_d2h_events.append(pending)
        for expert_id in copied:
            self._pending_expert_d2h_by_key[(int(layer_id), int(expert_id))] = pending
        for expert_id, params in copied.items():
            self._record_expert_copy_descriptor(
                direction="D2H",
                reason=str(reason),
                layer_id=int(layer_id),
                logical_id=int(expert_id),
                src_slot=int(expert_id),
                dst_slot=int(expert_id),
                nbytes=self._expert_backing_bytes(params),
                param_count=len(params),
            )
        if str(reason).startswith("demand"):
            self.stats.expert_d2h_demand_async_count += len(expert_ids)
        else:
            self.stats.expert_install_d2h_async_count += len(expert_ids)
            self.stats.expert_install_d2h_async_mb += copied_bytes / float(1024 * 1024)
        self.stats.expert_copy_stream_launch_count += 1
        self.stats.expert_d2h_stream_launch_count += 1
        self.stats.layerkv_copy_event_record_count += 1
        return int(copied_bytes)

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
        *,
        wait_for_host: bool = True,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        # Deferred destinations are safe for stream-ordered H2D, not CPU reads.
        # Only the on-demand materializer uses this option; recall stays blocking.
        if not wait_for_host and self.config.expert_transfer_backend != "cuda-batch":
            raise ValueError("deferred D2H completion requires cuda-batch")
        if not evicted:
            return {}
        copied: Dict[int, Dict[str, torch.Tensor]] = {
            int(logical_id): {} for logical_id, _slot_id in evicted
        }
        evicted_pairs = [
            (int(logical_id), int(slot_id)) for logical_id, slot_id in evicted
        ]
        use_tensor_batch = self._optimized_profile_enabled() and len(evicted_pairs) > 1
        copied_bytes = 0
        with self._profile("profile_copy_expert_to_cpu_ms"):
            with torch.no_grad():
                transfer = self._get_expert_batch_transfer()
                if transfer is not None:
                    transfer.collect()
                    copied, pairs, copied_bytes = self._build_expert_batch_backup(
                        state, evicted_pairs
                    )
                    transfer.copy(pairs, torch.cuda.current_stream(device=state.device))
                    self.stats.expert_eviction_d2h_batch_count += 1
                    self.stats.expert_eviction_d2h_batched_count += len(evicted_pairs)
                    self.stats.expert_eviction_d2h_batched_mb += copied_bytes / float(
                        1024 * 1024
                    )
                elif use_tensor_batch:
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
                                    dst = self._alloc_cpu_backing_tensor(
                                        tuple(selected.shape),
                                        selected.dtype,
                                        pin_memory=True,
                                    )
                                    dst.copy_(selected, non_blocking=True)
                                except Exception:
                                    dst = selected.to("cpu", copy=True)
                                copied_bytes += int(dst.nbytes)
                                for offset, (logical_id, _slot_id) in enumerate(
                                    chunk_pairs
                                ):
                                    view = dst[offset]
                                    self._register_cpu_backing_batch_view(dst, view)
                                    copied[int(logical_id)][name] = view
                        self.stats.expert_eviction_d2h_batched_count += len(
                            evicted_pairs
                        )
                        self.stats.expert_eviction_d2h_batched_mb += (
                            copied_bytes / float(1024 * 1024)
                        )
                    except Exception:
                        self.stats.expert_eviction_d2h_fallback_count += len(
                            evicted_pairs
                        )
                        copied = {
                            int(logical_id): {}
                            for logical_id, _slot_id in evicted_pairs
                        }
                        copied_bytes = 0
                        for name in state.param_names:
                            param = getattr(state.module, name).data
                            for logical_id, slot_id in evicted_pairs:
                                tensor = param[int(slot_id)].detach()
                                backing = self._copy_tensor_to_cpu_backing_pooled(
                                    tensor,
                                    non_blocking=True,
                                )
                                copied[int(logical_id)][name] = backing
                                copied_bytes += int(backing.nbytes)
                else:
                    if evicted_pairs:
                        self.stats.expert_eviction_d2h_fallback_count += len(
                            evicted_pairs
                        )
                    for name in state.param_names:
                        param = getattr(state.module, name).data
                        for logical_id, slot_id in evicted_pairs:
                            tensor = param[int(slot_id)].detach()
                            backing = self._copy_tensor_to_cpu_backing_pooled(
                                tensor,
                                non_blocking=True,
                            )
                            copied[int(logical_id)][name] = backing
                            copied_bytes += int(backing.nbytes)
                if state.device.type == "cuda" and wait_for_host:
                    torch.cuda.current_stream(device=state.device).synchronize()
                elif not wait_for_host:
                    self.stats.expert_demand_d2h_stream_batch_count += 1
        added = copied_bytes or sum(
            self._expert_backing_bytes(params) for params in copied.values()
        )
        for logical_id, slot_id in evicted_pairs:
            params = copied.get(int(logical_id), {})
            self._record_expert_copy_descriptor(
                direction="D2H",
                reason="evict_backing_sync" if wait_for_host else "evict_backing_stream",
                layer_id=int(state.layer_id),
                logical_id=int(logical_id),
                src_slot=int(slot_id),
                dst_slot=int(logical_id),
                nbytes=self._expert_backing_bytes(params),
                param_count=len(params),
            )
        self._expert_host_backing_bytes += added
        self._refresh_expert_host_backing_stat()
        return copied

    def _copy_slots_to_cpu_batched_async(
        self,
        state: _LayerKVExpertLayerState,
        evicted: List[Tuple[int, int]],
        *,
        h2d_stream: Any = None,
    ) -> bool:
        if not evicted or state.device.type != "cuda":
            return False
        active_stream = self._expert_d2h_copy_stream(state.device)
        evicted_pairs = [
            (int(logical_id), int(slot_id)) for logical_id, slot_id in evicted
        ]
        copied: Dict[int, Dict[str, torch.Tensor]] = {
            int(logical_id): {} for logical_id, _slot_id in evicted_pairs
        }
        copied_bytes = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        transfer = self._get_expert_batch_transfer()
        try:
            with self._profile("profile_copy_expert_to_cpu_ms"):
                with torch.no_grad(), torch.cuda.stream(active_stream):
                    start.record(active_stream)
                    if transfer is not None:
                        transfer.collect()
                        copied, pairs, copied_bytes = self._build_expert_batch_backup(
                            state, evicted_pairs
                        )
                        transfer.copy(pairs, active_stream)
                    else:
                        for name in state.param_names:
                            param = getattr(state.module, name).data
                            dst = self._alloc_cpu_backing_tensor(
                                (len(evicted_pairs),) + tuple(param.shape[1:]),
                                param.dtype,
                                pin_memory=True,
                            )
                            copied_bytes += int(dst.nbytes)
                            for offset, (logical_id, slot_id) in enumerate(evicted_pairs):
                                view = dst[offset]
                                view.copy_(
                                    param[int(slot_id)].detach(),
                                    non_blocking=True,
                                )
                                self._register_cpu_backing_batch_view(dst, view)
                                copied[int(logical_id)][name] = view
                    end.record(active_stream)
                    # A prefetch may evict a slot and immediately refill it on
                    # the H2D stream.  The D2H source read must finish before
                    # that destination slot can be overwritten; stream issue
                    # order alone does not establish this dependency.
                    h2d_stream = h2d_stream or self._expert_h2d_copy_stream(state.device)
                    if (
                        h2d_stream is not None
                        and h2d_stream.cuda_stream != active_stream.cuda_stream
                    ):
                        h2d_stream.wait_event(end)
                        self.stats.expert_d2h_h2d_dependency_count += 1
        except Exception:
            if transfer is not None:
                raise
            return False
        pending = _LayerKVPendingExpertD2H(
            start_event=start,
            ready_event=end,
            layer_id=int(state.layer_id),
            copied=copied,
            bytes=int(copied_bytes),
        )
        self._pending_expert_d2h_events.append(pending)
        for logical_id in copied:
            self._pending_expert_d2h_by_key[(int(state.layer_id), int(logical_id))] = (
                pending
            )
        for logical_id, slot_id in evicted_pairs:
            params = copied.get(int(logical_id), {})
            self._record_expert_copy_descriptor(
                direction="D2H",
                reason="evict_backing_async",
                layer_id=int(state.layer_id),
                logical_id=int(logical_id),
                src_slot=int(slot_id),
                dst_slot=int(logical_id),
                nbytes=self._expert_backing_bytes(params),
                param_count=len(params),
            )
        self.stats.expert_eviction_d2h_async_count += len(evicted_pairs)
        self.stats.expert_eviction_d2h_async_mb += copied_bytes / float(1024 * 1024)
        self.stats.expert_eviction_d2h_batch_count += 1
        self.stats.expert_eviction_d2h_batched_count += len(evicted_pairs)
        self.stats.expert_eviction_d2h_batched_mb += copied_bytes / float(1024 * 1024)
        self.stats.expert_d2h_stream_launch_count += 1
        return True

    def _finalize_expert_d2h_events(self, *, block: bool = False) -> None:
        if not self._pending_expert_d2h_events:
            return
        remaining: List[_LayerKVPendingExpertD2H] = []
        finalized_install = 0
        for pending in self._pending_expert_d2h_events:
            try:
                if block:
                    pending.ready_event.synchronize()
                elif not pending.ready_event.query():
                    remaining.append(pending)
                    continue
                try:
                    elapsed = float(
                        pending.start_event.elapsed_time(pending.ready_event)
                    )
                    self.stats.layerkv_copy_stream_busy_ms += elapsed
                    self.stats.expert_d2h_stream_busy_ms += elapsed
                except Exception:
                    pass
                state = self._expert_layers.get(int(pending.layer_id))
                target_cpu_params = pending.target_cpu_params
                mirror_cpu_params = pending.mirror_cpu_params
                if (
                    state is None
                    and target_cpu_params is None
                    and mirror_cpu_params is None
                ):
                    remaining.append(pending)
                    continue
                for logical_id, params in pending.copied.items():
                    logical_id = int(logical_id)
                    if target_cpu_params is not None:
                        target_cpu_params[logical_id] = params
                    if (
                        mirror_cpu_params is not None
                        and mirror_cpu_params is not target_cpu_params
                    ):
                        mirror_cpu_params[logical_id] = params
                    elif state is not None:
                        state.cpu_params[logical_id] = params
                    if state is not None:
                        self._update_expert_backing_cache_accounting(state, logical_id)
                    if (
                        state is not None
                        and (
                            target_cpu_params is state.cpu_params
                            or mirror_cpu_params is state.cpu_params
                            or (
                                target_cpu_params is None
                                and mirror_cpu_params is None
                            )
                        )
                    ):
                        state.backing_lru[logical_id] = self._decode_step
                    key = (int(pending.layer_id), logical_id)
                    if self._pending_expert_d2h_by_key.get(key) is pending:
                        self._pending_expert_d2h_by_key.pop(key, None)
                self._expert_host_backing_bytes += int(pending.bytes)
                if str(pending.reason).startswith("install"):
                    self.stats.expert_install_d2h_async_finalize_count += len(
                        pending.copied
                    )
                    finalized_install += len(pending.copied)
                elif str(pending.reason).startswith("demand"):
                    self.stats.expert_d2h_demand_finalize_count += len(
                        pending.copied
                    )
                    if target_cpu_params is not None:
                        finalized_install += len(pending.copied)
                else:
                    self.stats.expert_eviction_d2h_async_finalize_count += len(
                        pending.copied
                    )
            except Exception:
                remaining.append(pending)
        self._pending_expert_d2h_events = remaining
        self._refresh_expert_host_backing_stat()
        if finalized_install:
            self._refresh_expert_install_progress()

    def _wait_for_pending_expert_backing(
        self,
        state: _LayerKVExpertLayerState,
        logical_id: int,
        *,
        demand: bool = False,
    ) -> Optional[Dict[str, torch.Tensor]]:
        key = (int(state.layer_id), int(logical_id))
        pending = self._pending_expert_d2h_by_key.get(key)
        if pending is None:
            return None
        if demand:
            self.stats.expert_d2h_demand_pending_wait_count += 1
        if str(pending.reason).startswith("install"):
            self.stats.expert_install_d2h_async_wait_count += 1
        elif str(pending.reason).startswith("demand"):
            pass
        else:
            self.stats.expert_eviction_d2h_async_wait_count += 1
        pending.ready_event.synchronize()
        self._finalize_expert_d2h_events(block=False)
        return state.cpu_params.get(int(logical_id))

    def _expert_backing_for_materialize(
        self,
        state: _LayerKVExpertLayerState,
        logical_id: int,
        *,
        reason: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        logical_id = int(logical_id)
        source_params = state.cpu_params.get(logical_id)
        is_demand = reason == "on_demand"
        if source_params is not None:
            if is_demand:
                self.stats.expert_d2h_demand_ready_hit_count += 1
            return source_params
        global_params = (
            self._global_expert_backing(state.layer_id, logical_id)
            if self._expert_global_cpu_backing
            else None
        )
        if global_params is not None:
            state.cpu_params[logical_id] = global_params
            if is_demand:
                self.stats.expert_d2h_demand_ready_hit_count += 1
            return global_params
        if is_demand:
            if self._promote_queued_expert_d2h_for_demand(state, logical_id):
                budget_mb = max(
                    1e-6, float(state.expert_bytes) / float(1024 * 1024)
                )
                self._submit_expert_install_d2h_budgeted(budget_mb=budget_mb)
            elif self._enqueue_urgent_expert_d2h_for_demand(state, logical_id):
                budget_mb = max(
                    1e-6, float(state.expert_bytes) / float(1024 * 1024)
                )
                self._submit_expert_install_d2h_budgeted(budget_mb=budget_mb)
            else:
                self.stats.expert_d2h_demand_unavailable_count += 1
        return self._wait_for_pending_expert_backing(
            state, logical_id, demand=is_demand
        )

    @staticmethod
    def _expert_backing_bytes(params: Dict[str, torch.Tensor]) -> int:
        return sum(int(t.nbytes) for t in params.values())

    def _drop_expert_backing(
        self, state: _LayerKVExpertLayerState, logical_id: int
    ) -> None:
        removed = state.cpu_params.pop(int(logical_id), None)
        self._update_expert_backing_cache_accounting(state, logical_id)
        state.backing_lru.pop(int(logical_id), None)
        if removed is not None:
            if not self._is_global_expert_backing(state.layer_id, logical_id, removed):
                self._expert_host_backing_bytes -= self._expert_backing_bytes(removed)
                self._release_expert_backing_params(removed)
            self.stats.expert_backing_cache_evict_count += 1
            self._refresh_expert_host_backing_stat()

    def _get_expert_backing_cache_accounting(self, state):
        # Other planners/global backing retain their existing scan semantics.
        controller = self._shared_expert_controller_for_state(state)
        if (
            self.config.expert_backing_cache_accounting != "incremental"
            or controller is None
            or len(self._expert_layers) != 1
            or self._expert_layers.get(state.layer_id) is not state
            or self._expert_global_cpu_backing
            or self._effective_expert_backing_cache_mb() <= 0
        ):
            if getattr(state, "backing_cache_accounted_by_id", None) is not None:
                state.backing_cache_accounted_by_id = None
                state.backing_cache_accounted_bytes = 0
            return None
        if state.backing_cache_accounted_by_id is None:
            entries = {
                logical: self._expert_backing_bytes(params)
                for logical, params in state.cpu_params.items()
                if logical in state.logical_to_slot
            }
            state.backing_cache_accounted_by_id = entries
            state.backing_cache_accounted_bytes = sum(entries.values())
            self.stats.expert_backing_cache_accounting_init_count += 1
        return state.backing_cache_accounted_by_id

    def _update_expert_backing_cache_accounting(self, state, logical_id):
        entries = getattr(state, "backing_cache_accounted_by_id", None)
        if entries is None:
            return
        previous = entries.pop(logical_id, 0)
        params = state.cpu_params.get(logical_id)
        current = 0
        if params is not None and logical_id in state.logical_to_slot:
            current = self._expert_backing_bytes(params)
            entries[logical_id] = current
        state.backing_cache_accounted_bytes += current - previous
        self.stats.expert_backing_cache_accounting_update_count += 1

    def _check_expert_backing_cache_accounting(self, state):
        self.stats.expert_backing_cache_accounting_check_count += 1
        expected = {
            logical: self._expert_backing_bytes(params)
            for logical, params in state.cpu_params.items()
            if logical in state.logical_to_slot
        }
        if (
            expected != state.backing_cache_accounted_by_id
            or sum(expected.values()) != state.backing_cache_accounted_bytes
        ):
            reason = f"layer {state.layer_id} expert cache accounting mismatch"
            self.stats.expert_guard_pass = self.stats.comparable = False
            self.stats.expert_guard_reason = reason
            raise RuntimeError(reason)

    def _trim_expert_backing_cache(self, state: _LayerKVExpertLayerState) -> None:
        """Bound optional CPU copies for resident experts.

        CPU backing for offloaded experts is mandatory because the GPU slot no
        longer contains that logical expert.  Only resident experts are cache
        entries and may be dropped.
        """

        extra_cache_bytes = int(self._effective_expert_backing_cache_mb() * 1024 * 1024)
        entries = self._get_expert_backing_cache_accounting(state)
        if entries is not None:
            if self.config.expert_backing_cache_accounting_check:
                self._check_expert_backing_cache_accounting(state)
            if state.backing_cache_accounted_bytes <= extra_cache_bytes:
                self.stats.expert_backing_cache_fast_return_count += 1
                return
            while state.backing_cache_accounted_bytes > extra_cache_bytes:
                # Preserve scan-mode tie breaking; never drop mandatory backing.
                logical_id = min(
                    entries, key=lambda i: (int(state.backing_lru.get(i, 0)), int(i))
                )
                self._drop_expert_backing(state, logical_id)
            return
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
                self._update_expert_backing_cache_accounting(state, logical_id)
                state.backing_lru.pop(int(logical_id), None)
                if removed is not None:
                    if not self._is_global_expert_backing(
                        state.layer_id, logical_id, removed
                    ):
                        removed_bytes += self._expert_backing_bytes(removed)
                        self._release_expert_backing_params(removed)
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

    def _expert_host_budget_summary(self):
        """Poll-time accounting only; tracked storages are not allocator/RSS usage.

        Logical mandatory/cache bytes classify valid expert copies. Unique
        storages also include idle buffers and CUDA batch owners, deduplicating
        aliases. Startup workspaces and framework-retained caches are excluded.
        """
        mandatory = cached = 0
        accounting_active = False
        accounting_matches = True
        accounted_bytes = 0
        storages = {}
        pending_storages = {}

        def track(tensor, pending=False):
            if tensor.device.type != "cpu":
                return
            storage = tensor.untyped_storage()
            key = storage.data_ptr()
            storages[key] = storage.nbytes()
            if pending:
                pending_storages[key] = storage.nbytes()

        for state in self._expert_layers.values():
            resident_bytes = {}
            state_layer_id = int(getattr(state, "layer_id", -1))
            for logical, params in state.cpu_params.items():
                if self._is_global_expert_backing(state_layer_id, logical, params):
                    continue  # Immutable storage is counted once below, not as optional cache.
                nbytes = self._expert_backing_bytes(params)
                if logical in state.logical_to_slot:
                    cached += nbytes
                    resident_bytes[logical] = nbytes
                else:
                    mandatory += nbytes
                for tensor in params.values():
                    track(tensor)
            entries = getattr(state, "backing_cache_accounted_by_id", None)
            if entries is not None:
                accounting_active = True
                accounted_bytes += state.backing_cache_accounted_bytes
                accounting_matches &= (
                    entries == resident_bytes
                    and state.backing_cache_accounted_bytes
                    == sum(resident_bytes.values())
                )
        for params in self._expert_global_cpu_backing.values():
            for tensor in params.values():
                track(tensor)
        for tensors in self._expert_cpu_backing_pool.values():
            for tensor in tensors:
                track(tensor)
        transfer = self._expert_batch_transfer
        if transfer is not None:
            for _event, pairs, _keys in transfer.pending:
                for dst, src in pairs:
                    track(dst, True)
                    track(src, True)
            for dst, src in transfer.failed_refs:
                track(dst, True)
                track(src, True)
            for tensor, _callback in transfer.deferred.values():
                track(tensor, True)
        cache_limit = int(self._effective_expert_backing_cache_mb() * 1024 * 1024)
        pool_limit = self._expert_cpu_backing_pool_limit_bytes
        pool_bytes = self._expert_cpu_backing_pool_bytes
        return dict(
            cache_accounting_active=accounting_active,
            cache_accounting_matches=bool(accounting_matches),
            cached_accounted_bytes=accounted_bytes,
            mandatory_valid_bytes=mandatory,
            immutable_backing_bytes=self._expert_global_cpu_backing_bytes,
            immutable_backing_mode=self.config.expert_cpu_backing_mode,
            cached_valid_bytes=cached,
            idle_pool_bytes=pool_bytes,
            cache_limit_bytes=cache_limit,
            pool_limit_bytes=pool_limit,
            optional_limit_bytes=cache_limit + pool_limit,
            optional_used_bytes=cached + pool_bytes,
            tracked_unique_storage_bytes=sum(storages.values()),
            batch_owner_storage_bytes=sum(pending_storages.values()),
            ledger_bytes=self._expert_host_backing_bytes,
            ledger_matches=(mandatory + cached == self._expert_host_backing_bytes),
            optional_guard_pass=(cached <= cache_limit and pool_bytes <= pool_limit),
        )

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
