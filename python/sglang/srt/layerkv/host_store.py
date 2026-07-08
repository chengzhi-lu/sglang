"""Host-side KVC backing store for LayerKV."""

from __future__ import annotations

import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

if __package__:
    from .config_stats import LayerKVStats
    from .common_types import _LayerKVResidencyEntry
else:  # pragma: no cover - direct file-loading smoke tests.
    from config_stats import LayerKVStats
    from common_types import _LayerKVResidencyEntry


_FUSED_SPAN_SCATTER_OP: Optional[Any] = None
_FUSED_SPAN_SCATTER_CHECKED = False
_FUSED_SPAN_SCATTER_BATCHED_OP: Optional[Any] = None
_FUSED_SPAN_SCATTER_BATCHED_CHECKED = False
_FUSED_SPAN_BACKUP_BATCHED_OP: Optional[Any] = None
_FUSED_SPAN_BACKUP_BATCHED_CHECKED = False
_LAYERKV_SPAN_OPS_LIBRARY_CHECKED = False


def _load_layerkv_span_ops_library() -> None:
    global _LAYERKV_SPAN_OPS_LIBRARY_CHECKED
    if _LAYERKV_SPAN_OPS_LIBRARY_CHECKED:
        return
    _LAYERKV_SPAN_OPS_LIBRARY_CHECKED = True
    candidates = []
    env_path = os.environ.get("LAYERKV_SPAN_OPS_LIB")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path("/tmp/layerkv_span_ext/layerkv_span_ops_ext.so"))
    for path in candidates:
        if not path.exists():
            continue
        try:
            torch.ops.load_library(str(path))
            return
        except Exception:
            continue


def _get_fused_span_scatter_op() -> Optional[Any]:
    global _FUSED_SPAN_SCATTER_OP, _FUSED_SPAN_SCATTER_CHECKED
    if _FUSED_SPAN_SCATTER_CHECKED:
        return _FUSED_SPAN_SCATTER_OP
    _FUSED_SPAN_SCATTER_CHECKED = True
    try:
        import sgl_kernel  # noqa: F401

        op = getattr(torch.ops.sgl_kernel, "layerkv_copy_kv_span_scatter", None)
        if op is None:
            _load_layerkv_span_ops_library()
            op = getattr(torch.ops.sgl_kernel, "layerkv_copy_kv_span_scatter", None)
        _FUSED_SPAN_SCATTER_OP = getattr(op, "default", op)
    except Exception:
        _FUSED_SPAN_SCATTER_OP = None
    return _FUSED_SPAN_SCATTER_OP


def _get_fused_span_scatter_batched_op() -> Optional[Any]:
    global _FUSED_SPAN_SCATTER_BATCHED_OP, _FUSED_SPAN_SCATTER_BATCHED_CHECKED
    if _FUSED_SPAN_SCATTER_BATCHED_CHECKED:
        return _FUSED_SPAN_SCATTER_BATCHED_OP
    _FUSED_SPAN_SCATTER_BATCHED_CHECKED = True
    try:
        import sgl_kernel  # noqa: F401

        op = getattr(torch.ops.sgl_kernel, "layerkv_copy_kv_span_scatter_batched", None)
        if op is None:
            _load_layerkv_span_ops_library()
            op = getattr(
                torch.ops.sgl_kernel, "layerkv_copy_kv_span_scatter_batched", None
            )
        _FUSED_SPAN_SCATTER_BATCHED_OP = getattr(op, "default", op)
    except Exception:
        _FUSED_SPAN_SCATTER_BATCHED_OP = None
    return _FUSED_SPAN_SCATTER_BATCHED_OP


def _get_fused_span_backup_batched_op() -> Optional[Any]:
    global _FUSED_SPAN_BACKUP_BATCHED_OP, _FUSED_SPAN_BACKUP_BATCHED_CHECKED
    if _FUSED_SPAN_BACKUP_BATCHED_CHECKED:
        return _FUSED_SPAN_BACKUP_BATCHED_OP
    _FUSED_SPAN_BACKUP_BATCHED_CHECKED = True
    try:
        import sgl_kernel  # noqa: F401

        op = getattr(torch.ops.sgl_kernel, "layerkv_copy_kv_span_backup_batched", None)
        if op is None:
            _load_layerkv_span_ops_library()
            op = getattr(
                torch.ops.sgl_kernel, "layerkv_copy_kv_span_backup_batched", None
            )
        _FUSED_SPAN_BACKUP_BATCHED_OP = getattr(op, "default", op)
    except Exception:
        _FUSED_SPAN_BACKUP_BATCHED_OP = None
    return _FUSED_SPAN_BACKUP_BATCHED_OP


class _LayerKVHostKVStore:
    """Compact pinned host backing for LayerKV-owned evicted MHA KV tokens."""

    def __init__(
        self,
        kv_pool: Any,
        capacity_tokens: int,
        *,
        per_layer_mode: bool = False,
        stats: Optional[LayerKVStats] = None,
    ):
        self.kv_pool = kv_pool
        self.capacity_tokens = max(1, int(capacity_tokens))
        self.per_layer_mode = bool(per_layer_mode)
        self.stats = stats
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
        self.layer_used_counts: List[int] = [0 for _ in range(self.layer_num)]
        self.track_layer_used_sets = not self.per_layer_mode
        self._span_tensor_cache: Dict[
            Tuple[Tuple[int, int, int], ...], torch.Tensor
        ] = {}
        self._cpu_span_tensor_cache: Dict[
            Tuple[
                Tuple[Tuple[int, int, int], ...],
                Tuple[int, ...],
            ],
            Tuple[torch.Tensor, torch.Tensor],
        ] = {}
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

    def _add_profile(self, field: str, elapsed_ms: float) -> None:
        if self.stats is None:
            return
        try:
            setattr(self.stats, field, float(getattr(self.stats, field)) + elapsed_ms)
        except Exception:
            pass

    def _add_stat(self, field: str, value: int = 1) -> None:
        if self.stats is None:
            return
        try:
            setattr(self.stats, field, int(getattr(self.stats, field)) + int(value))
        except Exception:
            pass

    @staticmethod
    def _empty_cpu(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        try:
            return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        except Exception:
            return torch.empty(shape, dtype=dtype, device="cpu")

    def _host_spans_tensor(
        self, host_spans: Tuple[Tuple[int, int, int], ...]
    ) -> torch.Tensor:
        key = tuple(
            (int(start), int(offset), int(length))
            for start, offset, length in host_spans
        )
        cached = self._span_tensor_cache.get(key)
        if cached is not None and cached.device == self.device:
            return cached
        if len(self._span_tensor_cache) >= 128:
            self._span_tensor_cache.pop(next(iter(self._span_tensor_cache)))
        tensor = torch.tensor(key, dtype=torch.int64, device=self.device)
        self._span_tensor_cache[key] = tensor
        return tensor

    def _cpu_spans_and_batch_tensors(
        self,
        spans: List[Tuple[int, int, int]],
        span_batch_ids: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        span_key = tuple(
            (int(src), int(dst), int(length)) for src, dst, length in spans
        )
        batch_key = tuple(int(batch_id) for batch_id in span_batch_ids)
        key = (span_key, batch_key)
        cached = self._cpu_span_tensor_cache.get(key)
        if cached is not None:
            return cached
        if len(self._cpu_span_tensor_cache) >= 128:
            self._cpu_span_tensor_cache.pop(next(iter(self._cpu_span_tensor_cache)))
        tensors = (
            torch.tensor(span_key, dtype=torch.int64, device="cpu"),
            torch.tensor(batch_key, dtype=torch.int64, device="cpu"),
        )
        self._cpu_span_tensor_cache[key] = tensors
        return tensors

    @property
    def used_count(self) -> int:
        if self.per_layer_mode:
            return sum(int(count) for count in self.layer_used_counts)
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
        return (
            self.capacity_tokens * self.bytes_per_token_all_layers / float(1024 * 1024)
        )

    def _ensure_layer_capacity(self, layer_offset: int, need_free: int) -> None:
        if not self.per_layer_mode or need_free <= len(
            self.layer_free_slots[layer_offset]
        ):
            return
        layer_id = self.start_layer + layer_offset
        k_ref = self.kv_pool._get_key_buffer(layer_id)
        v_ref = self.kv_pool._get_value_buffer(layer_id)
        old_capacity = self.layer_capacities[layer_offset]
        used = int(self.layer_used_counts[layer_offset])
        required = used + int(need_free)
        initial_chunk = min(max(1, int(self.capacity_tokens)), 4096)
        new_capacity = max(
            required,
            old_capacity * 2 if old_capacity > 0 else 0,
            initial_chunk,
            64,
        )
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

    def alloc_per_layer(
        self, entries: List["_LayerKVResidencyEntry"]
    ) -> Optional[List[int]]:
        if not self.per_layer_mode:
            return self.alloc(sum(entry.token_count for entry in entries))
        need_by_layer: Dict[int, int] = {}
        entry_layers: List[int] = []
        for entry in entries:
            layer_offset = int(entry.layer_id) - self.start_layer
            if layer_offset < 0 or layer_offset >= self.layer_num:
                return None
            entry_layers.append(layer_offset)
            need_by_layer[layer_offset] = need_by_layer.get(layer_offset, 0) + int(
                entry.token_count
            )
        for layer_offset, need in need_by_layer.items():
            self._ensure_layer_capacity(layer_offset, need)
        for layer_offset, need in need_by_layer.items():
            if int(need) > len(self.layer_free_slots[layer_offset]):
                return None
        allocated_by_layer: Dict[int, List[int]] = {}
        for layer_offset, need in need_by_layer.items():
            free = self.layer_free_slots[layer_offset]
            slots = free[-int(need) :]
            del free[-int(need) :]
            self.layer_used_counts[layer_offset] += len(slots)
            if self.track_layer_used_sets:
                self.layer_used_slots[layer_offset].update(slots)
            allocated_by_layer[layer_offset] = slots
        offsets_by_layer = {int(layer_offset): 0 for layer_offset in need_by_layer}
        out: List[int] = []
        for entry, layer_offset in zip(entries, entry_layers):
            need = int(entry.token_count)
            offset = offsets_by_layer[layer_offset]
            slots = allocated_by_layer[layer_offset][offset : offset + need]
            offsets_by_layer[layer_offset] = offset + need
            out.extend(slots)
        return out

    def preallocate_per_layer_capacity(self, tokens_per_layer: int) -> None:
        if not self.per_layer_mode:
            return
        tokens_per_layer = max(0, int(tokens_per_layer))
        if tokens_per_layer <= 0:
            return
        for layer_offset in range(self.layer_num):
            need = max(
                0,
                int(tokens_per_layer) - len(self.layer_free_slots[layer_offset]),
            )
            if need > 0:
                self._ensure_layer_capacity(layer_offset, need)

    def alloc(self, need: int) -> Optional[List[int]]:
        if self.per_layer_mode:
            raise RuntimeError("use alloc_per_layer for per-layer KVC host store")
        if need > len(self.free_slots):
            return None
        slots = self.free_slots[-need:]
        del self.free_slots[-need:]
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
        free = self.layer_free_slots[layer_offset]
        if not self.track_layer_used_sets:
            free.extend(int(slot) for slot in slots)
            self.layer_used_counts[layer_offset] = max(
                0, int(self.layer_used_counts[layer_offset]) - len(slots)
            )
            return
        used = self.layer_used_slots[layer_offset]
        released = 0
        for slot in slots:
            slot = int(slot)
            if slot in used:
                used.remove(slot)
                free.append(slot)
                released += 1
        self.layer_used_counts[layer_offset] = max(
            0, int(self.layer_used_counts[layer_offset]) - released
        )

    @staticmethod
    def _unit_stride_start(values: List[int]) -> Optional[int]:
        if not values:
            return None
        start = int(values[0])
        for idx, value in enumerate(values):
            if int(value) != start + idx:
                return None
        return start

    @staticmethod
    def _paired_contiguous_spans(
        src_slots: List[int], dst_slots: List[int]
    ) -> Tuple[Tuple[int, int, int], ...]:
        if not src_slots or not dst_slots or len(src_slots) != len(dst_slots):
            return ()
        spans: List[Tuple[int, int, int]] = []
        src_start = int(src_slots[0])
        dst_start = int(dst_slots[0])
        prev_src = src_start
        prev_dst = dst_start
        length = 1
        ordered = True
        for raw_src, raw_dst in zip(src_slots[1:], dst_slots[1:]):
            src = int(raw_src)
            dst = int(raw_dst)
            if src < prev_src:
                ordered = False
                break
            if src == prev_src + 1 and dst == prev_dst + 1:
                length += 1
            else:
                spans.append((src_start, dst_start, length))
                src_start, dst_start = src, dst
                length = 1
            prev_src, prev_dst = src, dst
        if ordered:
            spans.append((src_start, dst_start, length))
            return tuple(spans)

        pairs = sorted((int(src), int(dst)) for src, dst in zip(src_slots, dst_slots))
        src_start, dst_start = pairs[0]
        prev_src, prev_dst = src_start, dst_start
        length = 1
        for src, dst in pairs[1:]:
            src = int(src)
            dst = int(dst)
            if src == prev_src + 1 and dst == prev_dst + 1:
                length += 1
            else:
                spans.append((src_start, dst_start, length))
                src_start, dst_start = src, dst
                length = 1
            prev_src, prev_dst = src, dst
        spans.append((src_start, dst_start, length))
        return tuple(spans)

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
            k_src = (
                self.kv_pool._get_key_buffer(layer_id)[device_locs]
                .detach()
                .to("cpu", non_blocking=True)
            )
            v_src = (
                self.kv_pool._get_value_buffer(layer_id)[device_locs]
                .detach()
                .to("cpu", non_blocking=True)
            )
            self.k_buffers[layer_offset].index_copy_(0, host_index, k_src)
            self.v_buffers[layer_offset].index_copy_(0, host_index, v_src)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    def backup_to_staging_async(
        self, device_locs: torch.Tensor, stream: Optional[torch.cuda.Stream]
    ) -> Tuple[Optional[Any], Optional[Any], List[torch.Tensor], List[torch.Tensor]]:
        if self.per_layer_mode:
            raise RuntimeError("per-layer KVC eviction uses backup_per_layer")
        if device_locs.numel() == 0 or stream is None:
            return None, None, [], []
        device_module = torch.get_device_module(self.device)
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        k_staging: List[torch.Tensor] = []
        v_staging: List[torch.Tensor] = []
        with torch.cuda.stream(stream):
            start.record(stream)
            for layer_offset in range(self.layer_num):
                layer_id = self.start_layer + layer_offset
                k_src = self.kv_pool._get_key_buffer(layer_id)[device_locs].detach()
                v_src = self.kv_pool._get_value_buffer(layer_id)[device_locs].detach()
                k_dst = self._empty_cpu(tuple(k_src.shape), k_src.dtype)
                v_dst = self._empty_cpu(tuple(v_src.shape), v_src.dtype)
                k_dst.copy_(k_src, non_blocking=True)
                v_dst.copy_(v_src, non_blocking=True)
                k_staging.append(k_dst)
                v_staging.append(v_dst)
            end.record(stream)
        return start, end, k_staging, v_staging

    def commit_staging_backup(
        self,
        host_slots: List[int],
        k_staging: List[torch.Tensor],
        v_staging: List[torch.Tensor],
    ) -> None:
        if self.per_layer_mode:
            raise RuntimeError("per-layer KVC eviction uses backup_per_layer")
        if not host_slots:
            return
        host_index = torch.tensor(host_slots, dtype=torch.int64, device="cpu")
        for layer_offset in range(self.layer_num):
            if layer_offset >= len(k_staging) or layer_offset >= len(v_staging):
                raise RuntimeError("incomplete async KVC eviction staging buffers")
            self.k_buffers[layer_offset].index_copy_(
                0, host_index, k_staging[layer_offset]
            )
            self.v_buffers[layer_offset].index_copy_(
                0, host_index, v_staging[layer_offset]
            )

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
            device_locs = torch.tensor(
                device_loc_list, dtype=torch.int64, device=self.device
            )
            k_src = self.kv_pool._get_key_buffer(layer_id)[device_locs].detach()
            v_src = self.kv_pool._get_value_buffer(layer_id)[device_locs].detach()
            host_start = self._unit_stride_start(host_slot_list)
            if host_start is not None:
                host_end = int(host_start) + len(host_slot_list)
                self.k_buffers[layer_offset][host_start:host_end].copy_(
                    k_src, non_blocking=True
                )
                self.v_buffers[layer_offset][host_start:host_end].copy_(
                    v_src, non_blocking=True
                )
            else:
                host_index = torch.tensor(
                    host_slot_list, dtype=torch.int64, device="cpu"
                )
                self.k_buffers[layer_offset].index_copy_(
                    0, host_index, k_src.to("cpu", non_blocking=True)
                )
                self.v_buffers[layer_offset].index_copy_(
                    0, host_index, v_src.to("cpu", non_blocking=True)
                )
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    def backup_per_layer_async(
        self,
        entries: Sequence[_LayerKVResidencyEntry],
        host_slots: List[int],
        stream: Optional[torch.cuda.Stream],
    ) -> Tuple[Optional[Any], Optional[Any]]:
        if not self.per_layer_mode:
            raise RuntimeError(
                "non-per-layer KVC eviction uses backup_to_staging_async"
            )
        if not entries or not host_slots or stream is None:
            return None, None
        device_module = torch.get_device_module(self.device)
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        fused_backup_op = _get_fused_span_backup_batched_op()

        def issue_copy() -> None:
            start.record(stream)
            t_group = time.perf_counter()
            by_layer: Dict[int, Tuple[List[int], List[int]]] = {}
            offset = 0
            if len(host_slots) == len(entries) and all(
                int(entry.token_count) == 1 for entry in entries
            ):
                for entry, slot in zip(entries, host_slots):
                    layer_offset = int(entry.layer_id) - self.start_layer
                    if layer_offset < 0 or layer_offset >= self.layer_num:
                        raise RuntimeError(
                            f"invalid per-layer KVC layer_id={entry.layer_id}"
                        )
                    device_locs = entry.device_loc_list()
                    if not device_locs:
                        continue
                    layer_device_locs, layer_host_slots = by_layer.setdefault(
                        layer_offset, ([], [])
                    )
                    layer_device_locs.append(int(device_locs[0]))
                    layer_host_slots.append(int(slot))
            else:
                for entry in entries:
                    layer_offset = int(entry.layer_id) - self.start_layer
                    if layer_offset < 0 or layer_offset >= self.layer_num:
                        raise RuntimeError(
                            f"invalid per-layer KVC layer_id={entry.layer_id}"
                        )
                    count = entry.token_count
                    slots = host_slots[offset : offset + count]
                    offset += count
                    layer_device_locs, layer_host_slots = by_layer.setdefault(
                        layer_offset, ([], [])
                    )
                    layer_device_locs.extend(entry.device_loc_list())
                    layer_host_slots.extend(int(slot) for slot in slots)
            self._add_profile(
                "profile_kvc_backup_group_ms",
                (time.perf_counter() - t_group) * 1000.0,
            )
            if fused_backup_op is not None:
                t_span = time.perf_counter()
                src_ks: List[torch.Tensor] = []
                src_vs: List[torch.Tensor] = []
                dst_ks: List[torch.Tensor] = []
                dst_vs: List[torch.Tensor] = []
                span_rows: List[Tuple[int, int, int]] = []
                span_batch_ids: List[int] = []
                for layer_offset, (
                    device_loc_list,
                    host_slot_list,
                ) in by_layer.items():
                    if not device_loc_list or not host_slot_list:
                        continue
                    if len(device_loc_list) != len(host_slot_list):
                        span_rows = []
                        break
                    layer_id = self.start_layer + int(layer_offset)
                    batch_id = len(src_ks)
                    src_ks.append(self.kv_pool._get_key_buffer(layer_id))
                    src_vs.append(self.kv_pool._get_value_buffer(layer_id))
                    dst_ks.append(self.k_buffers[int(layer_offset)])
                    dst_vs.append(self.v_buffers[int(layer_offset)])
                    spans = self._paired_contiguous_spans(
                        device_loc_list, host_slot_list
                    )
                    if not spans:
                        span_rows = []
                        break
                    span_rows.extend(spans)
                    span_batch_ids.extend([batch_id] * len(spans))
                self._add_profile(
                    "profile_kvc_backup_span_build_ms",
                    (time.perf_counter() - t_span) * 1000.0,
                )
                if span_rows and src_ks:
                    t_tensor = time.perf_counter()
                    spans_tensor, span_batch_tensor = self._cpu_spans_and_batch_tensors(
                        span_rows, span_batch_ids
                    )
                    self._add_profile(
                        "profile_kvc_backup_tensor_cache_ms",
                        (time.perf_counter() - t_tensor) * 1000.0,
                    )
                    item_size = int(src_ks[0][0].numel() * src_ks[0].element_size())
                    t_issue = time.perf_counter()
                    fused_backup_op(
                        src_ks,
                        src_vs,
                        dst_ks,
                        dst_vs,
                        spans_tensor,
                        span_batch_tensor,
                        item_size,
                    )
                    self._add_profile(
                        "profile_kvc_backup_op_issue_ms",
                        (time.perf_counter() - t_issue) * 1000.0,
                    )
                    end.record(stream)
                    return
            t_fallback = time.perf_counter()
            for layer_offset, (device_loc_list, host_slot_list) in by_layer.items():
                if not device_loc_list or not host_slot_list:
                    continue
                layer_id = self.start_layer + layer_offset
                device_locs = torch.tensor(
                    device_loc_list, dtype=torch.int64, device=self.device
                )
                k_src = self.kv_pool._get_key_buffer(layer_id)[device_locs].detach()
                v_src = self.kv_pool._get_value_buffer(layer_id)[device_locs].detach()
                host_start = self._unit_stride_start(host_slot_list)
                if host_start is not None:
                    host_end = int(host_start) + len(host_slot_list)
                    self.k_buffers[layer_offset][host_start:host_end].copy_(
                        k_src, non_blocking=True
                    )
                    self.v_buffers[layer_offset][host_start:host_end].copy_(
                        v_src, non_blocking=True
                    )
                else:
                    host_index = torch.tensor(
                        host_slot_list, dtype=torch.int64, device="cpu"
                    )
                    self.k_buffers[layer_offset].index_copy_(
                        0, host_index, k_src.to("cpu", non_blocking=True)
                    )
                    self.v_buffers[layer_offset].index_copy_(
                        0, host_index, v_src.to("cpu", non_blocking=True)
                    )
            self._add_profile(
                "profile_kvc_backup_fallback_issue_ms",
                (time.perf_counter() - t_fallback) * 1000.0,
            )
            end.record(stream)

        with torch.cuda.stream(stream):
            issue_copy()
        return start, end

    def reload_layers_to_locs_batched(
        self,
        requests: Sequence[
            Tuple[
                int, Tuple[int, int], Tuple[Tuple[int, int, int], ...], Tuple[int, int]
            ]
        ],
        stream: Optional[torch.cuda.Stream],
    ) -> Tuple[Optional[Any], Optional[Any], int]:
        fused_op = _get_fused_span_scatter_batched_op()
        if fused_op is None or stream is None or not requests:
            return None, None, 0
        src_ks: List[torch.Tensor] = []
        src_vs: List[torch.Tensor] = []
        dst_ks: List[torch.Tensor] = []
        dst_vs: List[torch.Tensor] = []
        span_rows: List[Tuple[int, int, int]] = []
        span_batch_ids: List[int] = []
        src_base_slots: List[int] = []
        dst_base_slots: List[int] = []
        total_tokens = 0
        span_token_count = 0
        span_request_count = 0
        slice_token_count = 0
        slice_request_count = 0
        span_segment_count = 0
        for batch_id, (layer_id, host_slice, host_spans, device_slice) in enumerate(
            requests
        ):
            layer_id = int(layer_id)
            layer_offset = layer_id - self.start_layer
            if layer_offset < 0 or layer_offset >= self.layer_num:
                return None, None, 0
            if int(device_slice[0]) < 0 or int(device_slice[1]) <= 0:
                return None, None, 0
            token_count = int(device_slice[1])
            if int(host_slice[0]) >= 0:
                host_first = int(host_slice[0])
                host_last = host_first + int(host_slice[1])
                if int(host_slice[1]) != token_count:
                    return None, None, 0
                spans = ((host_first, 0, token_count),)
                slice_token_count += token_count
                slice_request_count += 1
            elif host_spans:
                span_tokens = sum(
                    max(0, int(length)) for _start, _offset, length in host_spans
                )
                if span_tokens != token_count:
                    return None, None, 0
                host_first = min(int(start) for start, _offset, _length in host_spans)
                host_last = max(
                    int(start) + int(length) for start, _offset, length in host_spans
                )
                copied_tokens = max(0, host_last - host_first)
                if copied_tokens > max(token_count * 2, token_count + 256):
                    return None, None, 0
                spans = tuple(
                    (int(start), int(offset), int(length))
                    for start, offset, length in host_spans
                    if int(length) > 0
                )
                span_token_count += token_count
                span_request_count += 1
                span_segment_count += len(spans)
            else:
                return None, None, 0
            if host_last <= host_first:
                return None, None, 0
            src_ks.append(self.k_buffers[layer_offset][host_first:host_last])
            src_vs.append(self.v_buffers[layer_offset][host_first:host_last])
            dst_ks.append(self.kv_pool._get_key_buffer(layer_id))
            dst_vs.append(self.kv_pool._get_value_buffer(layer_id))
            src_base_slots.append(host_first)
            dst_base_slots.append(int(device_slice[0]))
            total_tokens += token_count
            for span in spans:
                span_rows.append(span)
                span_batch_ids.append(batch_id)
        if not span_rows or total_tokens <= 0:
            return None, None, 0
        device_module = torch.get_device_module(self.device)
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        spans_tensor = torch.tensor(span_rows, dtype=torch.int64, device="cpu")
        span_batch_tensor = torch.tensor(
            span_batch_ids, dtype=torch.int64, device="cpu"
        )
        src_base_tensor = torch.tensor(src_base_slots, dtype=torch.int64, device="cpu")
        dst_base_tensor = torch.tensor(dst_base_slots, dtype=torch.int64, device="cpu")
        item_size = int(dst_ks[0][0].numel() * dst_ks[0].element_size())
        with torch.cuda.stream(stream):
            issue_t0 = time.perf_counter()
            event_t0 = time.perf_counter()
            start.record(stream)
            self._add_profile(
                "profile_virtual_reload_event_record_ms",
                (time.perf_counter() - event_t0) * 1000.0,
            )
            path_t0 = time.perf_counter()
            fused_op(
                src_ks,
                src_vs,
                dst_ks,
                dst_vs,
                spans_tensor,
                span_batch_tensor,
                src_base_tensor,
                dst_base_tensor,
                item_size,
                8,
            )
            path_ms = (time.perf_counter() - path_t0) * 1000.0
            self._add_profile("profile_virtual_reload_span_path_ms", path_ms)
            self._add_profile("profile_virtual_reload_tensor_h2d_ms", path_ms)
            self._add_profile("profile_virtual_reload_span_bulk_copy_ms", path_ms)
            event_t0 = time.perf_counter()
            end.record(stream)
            self._add_profile(
                "profile_virtual_reload_event_record_ms",
                (time.perf_counter() - event_t0) * 1000.0,
            )
            self._add_profile(
                "profile_virtual_reload_issue_wall_ms",
                (time.perf_counter() - issue_t0) * 1000.0,
            )
        request_count = len(requests)
        self._add_stat("virtual_kvc_reload_async_count", request_count)
        self._add_stat("virtual_kvc_reload_direct_count", request_count)
        self._add_stat("virtual_kvc_reload_direct_token_count", total_tokens)
        if slice_token_count:
            self._add_stat("virtual_kvc_reload_slice_count", slice_request_count)
            self._add_stat("virtual_kvc_reload_slice_token_count", slice_token_count)
        if span_token_count:
            self._add_stat("virtual_kvc_reload_span_count", span_request_count)
            self._add_stat("virtual_kvc_reload_span_segment_count", span_segment_count)
            self._add_stat("virtual_kvc_reload_span_token_count", span_token_count)
            self._add_stat("virtual_kvc_reload_span_coalesce_count", span_request_count)
            self._add_stat(
                "virtual_kvc_reload_span_coalesce_token_count", span_token_count
            )
            self._add_stat(
                "virtual_kvc_reload_span_coalesce_copied_token_count", span_token_count
            )
            self._add_stat(
                "virtual_kvc_reload_span_direct_copy_count", span_request_count
            )
            self._add_stat(
                "virtual_kvc_reload_span_direct_copy_token_count", span_token_count
            )
        return start, end, request_count

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
                k_src = (
                    self.k_buffers[layer_offset]
                    .index_select(0, host_index)
                    .to(self.device, non_blocking=True)
                )
                v_src = (
                    self.v_buffers[layer_offset]
                    .index_select(0, host_index)
                    .to(self.device, non_blocking=True)
                )
                self.kv_pool._get_key_buffer(layer_id).index_copy_(
                    0, device_locs, k_src
                )
                self.kv_pool._get_value_buffer(layer_id).index_copy_(
                    0, device_locs, v_src
                )
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
                    raise RuntimeError(
                        f"invalid per-layer KVC layer_id={entry.layer_id}"
                    )
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
                host_index = torch.tensor(
                    host_slot_list, dtype=torch.int64, device="cpu"
                )
                device_locs = torch.tensor(
                    device_loc_list, dtype=torch.int64, device=self.device
                )
                k_src = (
                    self.k_buffers[layer_offset]
                    .index_select(0, host_index)
                    .to(self.device, non_blocking=True)
                )
                v_src = (
                    self.v_buffers[layer_offset]
                    .index_select(0, host_index)
                    .to(self.device, non_blocking=True)
                )
                self.kv_pool._get_key_buffer(layer_id).index_copy_(
                    0, device_locs, k_src
                )
                self.kv_pool._get_value_buffer(layer_id).index_copy_(
                    0, device_locs, v_src
                )
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

    def reload_layer_to_locs(
        self,
        layer_id: int,
        entries: List[_LayerKVResidencyEntry],
        device_locs: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
        async_copy: bool = False,
        host_index: Optional[torch.Tensor] = None,
        host_slice: Optional[Tuple[int, int]] = None,
        host_spans: Tuple[Tuple[int, int, int], ...] = (),
        device_slice: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, Optional[Any], Optional[Any]]:
        if not entries or int(device_locs.numel()) == 0:
            return 0.0, None, None
        prepare_t0 = time.perf_counter()
        layer_offset = int(layer_id) - self.start_layer
        if layer_offset < 0 or layer_offset >= self.layer_num:
            raise RuntimeError(f"invalid virtual KVC layer_id={layer_id}")
        host_slice_obj: Optional[slice] = None
        if host_slice is not None and int(host_slice[0]) >= 0:
            first_slot = int(host_slice[0])
            token_count = int(host_slice[1])
            if token_count != int(device_locs.numel()):
                raise RuntimeError(
                    f"virtual KVC scratch reload mismatch: host={token_count} device={int(device_locs.numel())}"
                )
            host_slice_obj = slice(first_slot, first_slot + token_count)
        elif host_spans:
            span_tokens = sum(
                max(0, int(length)) for _start, _offset, length in host_spans
            )
            if span_tokens != int(device_locs.numel()):
                raise RuntimeError(
                    f"virtual KVC scratch reload mismatch: spans={span_tokens} device={int(device_locs.numel())}"
                )
        elif host_index is None:
            host_slots: List[int] = []
            for entry in entries:
                host_slots.extend(entry.host_slot_list())
            if len(host_slots) != int(device_locs.numel()):
                raise RuntimeError(
                    f"virtual KVC scratch reload mismatch: host={len(host_slots)} device={int(device_locs.numel())}"
                )
            host_index = torch.tensor(host_slots, dtype=torch.int64, device="cpu")
        elif int(host_index.numel()) != int(device_locs.numel()):
            raise RuntimeError(
                f"virtual KVC scratch reload mismatch: host={int(host_index.numel())} device={int(device_locs.numel())}"
            )
        self._add_profile(
            "profile_virtual_reload_prepare_ms",
            (time.perf_counter() - prepare_t0) * 1000.0,
        )
        contiguous_check_t0 = time.perf_counter()
        if (
            host_slice_obj is None
            and host_index is not None
            and host_index.device.type == "cpu"
            and int(host_index.numel()) > 0
        ):
            first_slot = int(host_index[0])
            token_count = int(host_index.numel())
            last_slot = int(host_index[-1])
            if last_slot - first_slot + 1 == token_count:
                expected = torch.arange(
                    first_slot,
                    first_slot + token_count,
                    dtype=host_index.dtype,
                    device="cpu",
                )
                if bool(torch.equal(host_index, expected)):
                    host_slice_obj = slice(first_slot, first_slot + token_count)
        self._add_profile(
            "profile_virtual_reload_contiguous_check_ms",
            (time.perf_counter() - contiguous_check_t0) * 1000.0,
        )
        device_module = torch.get_device_module(self.device)
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        active_stream = stream if async_copy and stream is not None else None
        token_count = int(device_locs.numel())
        device_slice_obj: Optional[slice] = None
        if device_slice is not None and int(device_slice[0]) >= 0:
            device_start = int(device_slice[0])
            device_count = int(device_slice[1])
            if device_count >= token_count:
                device_slice_obj = slice(device_start, device_start + token_count)
        if host_slice_obj is not None:
            self._add_stat("virtual_kvc_reload_slice_count")
            self._add_stat("virtual_kvc_reload_slice_token_count", token_count)
        elif host_spans:
            self._add_stat("virtual_kvc_reload_span_count")
            self._add_stat("virtual_kvc_reload_span_segment_count", len(host_spans))
            self._add_stat("virtual_kvc_reload_span_token_count", token_count)
        else:
            self._add_stat("virtual_kvc_reload_index_count")
            self._add_stat("virtual_kvc_reload_index_token_count", token_count)
        if device_slice_obj is not None:
            self._add_stat("virtual_kvc_reload_direct_count")
            self._add_stat("virtual_kvc_reload_direct_token_count", token_count)
        if active_stream is not None:
            self._add_stat("virtual_kvc_reload_async_count")
        else:
            self._add_stat("virtual_kvc_reload_sync_count")

        def issue_copy() -> None:
            event_t0 = time.perf_counter()
            if active_stream is not None:
                start.record(active_stream)
            else:
                start.record()
            self._add_profile(
                "profile_virtual_reload_event_record_ms",
                (time.perf_counter() - event_t0) * 1000.0,
            )
            if host_slice_obj is not None:
                path_t0 = time.perf_counter()
                copy_t0 = time.perf_counter()
                k_src = self.k_buffers[layer_offset][host_slice_obj].to(
                    self.device, non_blocking=True
                )
                v_src = self.v_buffers[layer_offset][host_slice_obj].to(
                    self.device, non_blocking=True
                )
                self._add_profile(
                    "profile_virtual_reload_tensor_h2d_ms",
                    (time.perf_counter() - copy_t0) * 1000.0,
                )
                write_t0 = time.perf_counter()
                if device_slice_obj is not None:
                    self.kv_pool._get_key_buffer(layer_id)[device_slice_obj].copy_(
                        k_src
                    )
                    self.kv_pool._get_value_buffer(layer_id)[device_slice_obj].copy_(
                        v_src
                    )
                else:
                    self.kv_pool._get_key_buffer(layer_id).index_copy_(
                        0, device_locs, k_src
                    )
                    self.kv_pool._get_value_buffer(layer_id).index_copy_(
                        0, device_locs, v_src
                    )
                self._add_profile(
                    "profile_virtual_reload_device_write_ms",
                    (time.perf_counter() - write_t0) * 1000.0,
                )
                self._add_profile(
                    "profile_virtual_reload_slice_path_ms",
                    (time.perf_counter() - path_t0) * 1000.0,
                )
            elif host_spans:
                path_t0 = time.perf_counter()
                k_buffer = self.kv_pool._get_key_buffer(layer_id)
                v_buffer = self.kv_pool._get_value_buffer(layer_id)
                coalesced = False
                if device_slice_obj is not None and len(host_spans) > 1:
                    check_t0 = time.perf_counter()
                    host_first = min(
                        int(start) for start, _offset, _length in host_spans
                    )
                    host_last = max(
                        int(start) + int(length)
                        for start, _offset, length in host_spans
                    )
                    copied_tokens = max(0, int(host_last) - int(host_first))
                    direct_span_copy = copied_tokens == token_count
                    expected_host = int(host_first)
                    expected_offset = 0
                    if direct_span_copy:
                        for host_start, scratch_offset, length in sorted(
                            host_spans, key=lambda span: int(span[1])
                        ):
                            host_start = int(host_start)
                            scratch_offset = int(scratch_offset)
                            length = int(length)
                            if length <= 0:
                                continue
                            if (
                                scratch_offset != expected_offset
                                or host_start != expected_host
                            ):
                                direct_span_copy = False
                                break
                            expected_host += length
                            expected_offset += length
                        if expected_offset != token_count or expected_host != int(
                            host_last
                        ):
                            direct_span_copy = False
                    self._add_profile(
                        "profile_virtual_reload_span_coalesce_check_ms",
                        (time.perf_counter() - check_t0) * 1000.0,
                    )
                    if direct_span_copy:
                        host_slice = slice(host_first, host_last)
                        bulk_t0 = time.perf_counter()
                        k_buffer[device_slice_obj].copy_(
                            self.k_buffers[layer_offset][host_slice],
                            non_blocking=True,
                        )
                        v_buffer[device_slice_obj].copy_(
                            self.v_buffers[layer_offset][host_slice],
                            non_blocking=True,
                        )
                        self._add_profile(
                            "profile_virtual_reload_span_bulk_copy_ms",
                            (time.perf_counter() - bulk_t0) * 1000.0,
                        )
                        self._add_stat("virtual_kvc_reload_span_coalesce_count")
                        self._add_stat(
                            "virtual_kvc_reload_span_coalesce_token_count",
                            token_count,
                        )
                        self._add_stat(
                            "virtual_kvc_reload_span_coalesce_copied_token_count",
                            copied_tokens,
                        )
                        self._add_stat("virtual_kvc_reload_span_direct_copy_count")
                        self._add_stat(
                            "virtual_kvc_reload_span_direct_copy_token_count",
                            token_count,
                        )
                        coalesced = True
                    elif copied_tokens <= max(token_count * 2, token_count + 256):
                        bulk_t0 = time.perf_counter()
                        k_src = self.k_buffers[layer_offset][host_first:host_last].to(
                            self.device, non_blocking=True
                        )
                        v_src = self.v_buffers[layer_offset][host_first:host_last].to(
                            self.device, non_blocking=True
                        )
                        self._add_profile(
                            "profile_virtual_reload_tensor_h2d_ms",
                            (time.perf_counter() - bulk_t0) * 1000.0,
                        )
                        device_start = int(device_slice_obj.start)
                        fused_op = _get_fused_span_scatter_op()
                        use_fused = (
                            fused_op is not None
                            and k_src.is_contiguous()
                            and v_src.is_contiguous()
                            and k_buffer.is_contiguous()
                            and v_buffer.is_contiguous()
                        )
                        if use_fused:
                            spans_tensor = self._host_spans_tensor(host_spans)
                            item_size = int(k_src[0].numel() * k_src.element_size())
                            fused_t0 = time.perf_counter()
                            fused_op(
                                k_src,
                                v_src,
                                k_buffer,
                                v_buffer,
                                spans_tensor,
                                int(host_first),
                                int(device_start),
                                item_size,
                                8,
                            )
                            fused_ms = (time.perf_counter() - fused_t0) * 1000.0
                            self._add_profile(
                                "profile_virtual_reload_span_fused_ms", fused_ms
                            )
                            self._add_profile(
                                "profile_virtual_reload_span_scatter_ms", fused_ms
                            )
                            self._add_profile(
                                "profile_virtual_reload_device_write_ms", fused_ms
                            )
                            self._add_stat("virtual_kvc_reload_span_fused_count")
                            self._add_stat(
                                "virtual_kvc_reload_span_fused_token_count",
                                token_count,
                            )
                        else:
                            self._add_stat(
                                "virtual_kvc_reload_span_fused_fallback_count"
                            )
                            scatter_t0 = time.perf_counter()
                            for host_start, scratch_offset, length in host_spans:
                                host_start = int(host_start)
                                scratch_offset = int(scratch_offset)
                                length = int(length)
                                if length <= 0:
                                    continue
                                src_offset = host_start - host_first
                                src_slice = slice(src_offset, src_offset + length)
                                dst_slice = slice(
                                    device_start + scratch_offset,
                                    device_start + scratch_offset + length,
                                )
                                k_buffer[dst_slice].copy_(k_src[src_slice])
                                v_buffer[dst_slice].copy_(v_src[src_slice])
                            self._add_profile(
                                "profile_virtual_reload_span_scatter_ms",
                                (time.perf_counter() - scatter_t0) * 1000.0,
                            )
                            self._add_profile(
                                "profile_virtual_reload_device_write_ms",
                                (time.perf_counter() - scatter_t0) * 1000.0,
                            )
                        self._add_stat("virtual_kvc_reload_span_coalesce_count")
                        self._add_stat(
                            "virtual_kvc_reload_span_coalesce_token_count",
                            token_count,
                        )
                        self._add_stat(
                            "virtual_kvc_reload_span_coalesce_copied_token_count",
                            copied_tokens,
                        )
                        coalesced = True
                if not coalesced:
                    scatter_t0 = time.perf_counter()
                    for host_start, scratch_offset, length in host_spans:
                        host_start = int(host_start)
                        scratch_offset = int(scratch_offset)
                        length = int(length)
                        if length <= 0:
                            continue
                        span_host_slice = slice(host_start, host_start + length)
                        scratch_slice = slice(scratch_offset, scratch_offset + length)
                        span_device_locs = device_locs[scratch_slice]
                        copy_t0 = time.perf_counter()
                        k_src = self.k_buffers[layer_offset][span_host_slice].to(
                            self.device, non_blocking=True
                        )
                        v_src = self.v_buffers[layer_offset][span_host_slice].to(
                            self.device, non_blocking=True
                        )
                        self._add_profile(
                            "profile_virtual_reload_tensor_h2d_ms",
                            (time.perf_counter() - copy_t0) * 1000.0,
                        )
                        write_t0 = time.perf_counter()
                        if device_slice_obj is not None:
                            direct_slice = slice(
                                int(device_slice_obj.start) + scratch_offset,
                                int(device_slice_obj.start) + scratch_offset + length,
                            )
                            k_buffer[direct_slice].copy_(k_src)
                            v_buffer[direct_slice].copy_(v_src)
                        else:
                            k_buffer.index_copy_(0, span_device_locs, k_src)
                            v_buffer.index_copy_(0, span_device_locs, v_src)
                        self._add_profile(
                            "profile_virtual_reload_device_write_ms",
                            (time.perf_counter() - write_t0) * 1000.0,
                        )
                    self._add_profile(
                        "profile_virtual_reload_span_scatter_ms",
                        (time.perf_counter() - scatter_t0) * 1000.0,
                    )
                self._add_profile(
                    "profile_virtual_reload_span_path_ms",
                    (time.perf_counter() - path_t0) * 1000.0,
                )
            else:
                path_t0 = time.perf_counter()
                copy_t0 = time.perf_counter()
                k_src = (
                    self.k_buffers[layer_offset]
                    .index_select(0, host_index)
                    .to(self.device, non_blocking=True)
                )
                v_src = (
                    self.v_buffers[layer_offset]
                    .index_select(0, host_index)
                    .to(self.device, non_blocking=True)
                )
                self._add_profile(
                    "profile_virtual_reload_tensor_h2d_ms",
                    (time.perf_counter() - copy_t0) * 1000.0,
                )
                write_t0 = time.perf_counter()
                if device_slice_obj is not None:
                    self.kv_pool._get_key_buffer(layer_id)[device_slice_obj].copy_(
                        k_src
                    )
                    self.kv_pool._get_value_buffer(layer_id)[device_slice_obj].copy_(
                        v_src
                    )
                else:
                    self.kv_pool._get_key_buffer(layer_id).index_copy_(
                        0, device_locs, k_src
                    )
                    self.kv_pool._get_value_buffer(layer_id).index_copy_(
                        0, device_locs, v_src
                    )
                self._add_profile(
                    "profile_virtual_reload_device_write_ms",
                    (time.perf_counter() - write_t0) * 1000.0,
                )
                self._add_profile(
                    "profile_virtual_reload_index_path_ms",
                    (time.perf_counter() - path_t0) * 1000.0,
                )
            event_t0 = time.perf_counter()
            if active_stream is not None:
                end.record(active_stream)
            else:
                end.record()
            self._add_profile(
                "profile_virtual_reload_event_record_ms",
                (time.perf_counter() - event_t0) * 1000.0,
            )

        if active_stream is not None:
            with torch.cuda.stream(active_stream):
                issue_t0 = time.perf_counter()
                issue_copy()
                self._add_profile(
                    "profile_virtual_reload_issue_wall_ms",
                    (time.perf_counter() - issue_t0) * 1000.0,
                )
            return 0.0, start, end

        issue_t0 = time.perf_counter()
        issue_copy()
        self._add_profile(
            "profile_virtual_reload_issue_wall_ms",
            (time.perf_counter() - issue_t0) * 1000.0,
        )
        sync_t0 = time.perf_counter()
        end.synchronize()
        self._add_profile(
            "profile_virtual_reload_sync_wall_ms",
            (time.perf_counter() - sync_t0) * 1000.0,
        )
        return float(start.elapsed_time(end)), start, end
