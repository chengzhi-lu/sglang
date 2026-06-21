"""Shared LayerKV runtime dataclasses."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch


@dataclasses.dataclass
class _LayerKVResidencyEntry:
    req_idx: int
    pos: int
    state: str
    layer_id: int = -1
    device_loc: Optional[int] = None
    host_slot: Optional[int] = None
    device_locs: Optional[List[int]] = None
    host_slots: Optional[List[int]] = None
    evicted_device_locs: Optional[List[int]] = None
    page_size: int = 1
    ready_start_event: Optional[Any] = None
    ready_event: Optional[Any] = None
    ready_waited: bool = False
    last_access_step: int = 0

    @property
    def token_count(self) -> int:
        return max(1, int(self.page_size))

    def logical_positions(self) -> List[int]:
        return list(range(self.pos, self.pos + self.token_count))

    def device_loc_list(self) -> List[int]:
        if self.device_locs is not None:
            return self.device_locs
        if self.device_loc is None:
            return []
        return [int(self.device_loc)]

    def host_slot_list(self) -> List[int]:
        if self.host_slots is not None:
            return self.host_slots
        if self.host_slot is None:
            return []
        return [int(self.host_slot)]


@dataclasses.dataclass
class _LayerKVNativeKvcRun:
    req_idx: int
    start: int
    end: int
    locs: Tuple[int, ...]

    def contains(self, pos: int, count: int) -> bool:
        pos = int(pos)
        count = int(count)
        return count >= 0 and pos >= int(self.start) and pos + count <= int(self.end)

    def loc_slice(self, pos: int, count: int) -> Tuple[int, ...]:
        if not self.contains(pos, count):
            return ()
        offset = int(pos) - int(self.start)
        return self.locs[offset : offset + int(count)]


@dataclasses.dataclass
class _LayerKVPendingReload:
    start_event: Any
    ready_event: Any
    entries: List[_LayerKVResidencyEntry]
    waited_on_main_stream: bool = False


@dataclasses.dataclass
class _LayerKVVirtualMaterializePlan:
    step: int
    selected: List[_LayerKVResidencyEntry]
    req_indices: Tuple[int, ...]
    positions: Tuple[int, ...]
    row_indices: Tuple[int, ...]
    flat_indices: Tuple[int, ...]
    flat_spans: Tuple[Tuple[int, int, int], ...]
    row_spans: Tuple[Tuple[int, int, int, int], ...]
    req_tensor: Optional[torch.Tensor]
    pos_tensor: Optional[torch.Tensor]
    row_tensor: Optional[torch.Tensor]
    flat_tensor: Optional[torch.Tensor]
    host_slots: Tuple[int, ...]
    host_signature: Tuple[int, int, int, int]
    host_slice: Tuple[int, int]
    host_spans: Tuple[Tuple[int, int, int], ...]
    host_index_cpu: Optional[torch.Tensor]
    max_row_index: int
    max_position: int
    max_flat_index: int
    token_count: int


@dataclasses.dataclass
class _LayerKVKvcDemand:
    layer_id: int
    entries: Tuple[_LayerKVResidencyEntry, ...]
    req_indices: Tuple[int, ...]
    positions: Tuple[int, ...]
    row_indices: Tuple[int, ...]
    flat_indices: Tuple[int, ...]
    flat_spans: Tuple[Tuple[int, int, int], ...]
    row_spans: Tuple[Tuple[int, int, int, int], ...]
    token_count: int
    deadline_layer: int
    benefit_score: float
    signature: str
    base_signature: str = ""
    backend_semantics: str = ""
    host_slots: Tuple[int, ...] = ()
    host_signature: Tuple[int, int, int, int] = (0, 0, 0, 0)
    host_slice: Tuple[int, int] = (-1, 0)
    host_spans: Tuple[Tuple[int, int, int], ...] = ()
    host_index_cpu: Optional[torch.Tensor] = None
    max_row_index: int = -1
    max_position: int = -1
    max_flat_index: int = -1


@dataclasses.dataclass
class _LayerKVExpertDemand:
    layer_id: int
    logical_ids: Tuple[int, ...]
    bytes: int
    deadline_layer: int
    benefit_score: float
    signature: str
    group_keys: Tuple["_LayerKVResidencyKey", ...] = ()


@dataclasses.dataclass
class _LayerKVMetadataPatchCacheEntry:
    key: Tuple[Any, ...]
    row_tensor: Optional[torch.Tensor] = None
    pos_tensor: Optional[torch.Tensor] = None
    flat_tensor: Optional[torch.Tensor] = None
    scratch_tensor: Optional[torch.Tensor] = None
    flat_span_tensor: Optional[torch.Tensor] = None
    flat_span_scratch_tensor: Optional[torch.Tensor] = None
    row_span_row_tensor: Optional[torch.Tensor] = None
    row_span_pos_tensor: Optional[torch.Tensor] = None
    row_span_scratch_tensor: Optional[torch.Tensor] = None
    flat_slice: Optional[Tuple[int, int]] = None
    flat_slice_checked: bool = False
    page_slice: Optional[Tuple[int, int, int]] = None
    page_slice_checked: bool = False
    hit_count: int = 0


@dataclasses.dataclass
class _LayerKVVirtualScratchCacheEntry:
    layer_id: int
    host_slots: Tuple[int, ...]
    host_signature: Tuple[int, int, int, int]
    host_slice: Tuple[int, int]
    host_index_cpu: Optional[torch.Tensor]
    token_count: int
    scratch_locs: torch.Tensor
    buffer_idx: int
    last_step: int


@dataclasses.dataclass
class _LayerKVPendingVirtualMaterialize:
    start_event: Any
    ready_event: Any
    layer_id: int
    entries: List[_LayerKVResidencyEntry]
    scratch_locs: torch.Tensor
    req_indices: Tuple[int, ...]
    positions: Tuple[int, ...]
    token_count: int
    buffer_idx: int
    waited_on_main_stream: bool = False


@dataclasses.dataclass
class _LayerKVPendingEviction:
    start_event: Any
    ready_event: Any
    entries: List[_LayerKVResidencyEntry]
    device_locs: torch.Tensor
    host_slots: List[int]
    k_staging: List[torch.Tensor]
    v_staging: List[torch.Tensor]
    token_count: int
    elapsed_recorded: bool = False


@dataclasses.dataclass
class _LayerKVPendingExpertCopy:
    start_event: Any
    ready_event: Any
    layer_id: int
    logical_ids: Set[int]
    waited_on_main_stream: bool = False
    wait_start_event: Optional[Any] = None
    wait_end_event: Optional[Any] = None
    wait_elapsed_recorded: bool = False


@dataclasses.dataclass
class _LayerKVPendingExpertD2H:
    start_event: Any
    ready_event: Any
    layer_id: int
    copied: Dict[int, Dict[str, torch.Tensor]]
    bytes: int
    reason: str = "evict_backing_async"
    source_refs: Tuple[torch.Tensor, ...] = ()
    target_cpu_params: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    mirror_cpu_params: Optional[Dict[int, Dict[str, torch.Tensor]]] = None


@dataclasses.dataclass
class _LayerKVExpertInstallD2HJob:
    seq: int
    layer_id: int
    module: Any
    param_names: List[str]
    expert_ids: List[int]
    target_cpu_params: Dict[int, Dict[str, torch.Tensor]]
    priority: int = 0
    deadline_step: int = 0
    reason: str = "install_backing_async"
    mirror_cpu_params: Optional[Dict[int, Dict[str, torch.Tensor]]] = None


@dataclasses.dataclass(frozen=True)
class _LayerKVResidencyKey:
    kind: str
    layer_id: int
    logical_id: Any


@dataclasses.dataclass
class _LayerKVResidencyHandle:
    kind: str
    location: str
    ref: Any = None
    n_units: int = 1
    bytes: int = 0
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _LayerKVResidentTensorGroup:
    key: _LayerKVResidencyKey
    state: str
    bytes: int
    cpu_ref: Optional[_LayerKVResidencyHandle] = None
    gpu_ref: Optional[_LayerKVResidencyHandle] = None
    ready_start_event: Optional[Any] = None
    ready_event: Optional[Any] = None
    ready_waited: bool = False
    last_access_step: int = 0
    recover_count: int = 0
    wait_count: int = 0
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _LayerKVRecoveryTask:
    kind: str
    layer_id: int
    bytes: int
    deadline_layer: int
    demand_signature: str = ""
    kvc_demand: Optional[_LayerKVKvcDemand] = None
    expert_demand: Optional[_LayerKVExpertDemand] = None
    logical_ids: Tuple[int, ...] = ()
    group_keys: Tuple[_LayerKVResidencyKey, ...] = ()
    entries: Tuple[_LayerKVResidencyEntry, ...] = ()
    token_count: int = 0
    benefit_score: float = 0.0
    estimated_copy_ms: float = 0.0
    estimated_cpu_ms: float = 0.0
    state: str = "pending"


@dataclasses.dataclass
class _LayerKVExpertInstallBuild:
    layer_id: int
    module: Any
    slot_capacity: int
    initial_resident: List[int]
    full_num_experts: int
    device: torch.device
    dtype: torch.dtype
    cpu_params: Dict[int, Dict[str, torch.Tensor]]
    param_names: List[str]
    logical_to_slot: Dict[int, int]
    slot_to_logical: Dict[int, int]
    lru: Dict[int, int]
    expert_bytes: int
    compact_params: Dict[str, torch.Tensor]
    source_refs: Tuple[torch.Tensor, ...] = ()
    start_event: Optional[Any] = None
    ready_event: Optional[Any] = None
    elapsed_recorded: bool = False


@dataclasses.dataclass
class _LayerKVExpertInstallItem:
    layer_id: int
    module: Any
    slot_capacity: int
    initial_resident: Optional[List[int]]
    prepared_cpu_params: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    backing_queued: bool = False
    pending_cpu_experts: Set[int] = dataclasses.field(default_factory=set)
    prebuilt_install: Optional[_LayerKVExpertInstallBuild] = None
    preallocated_compact_params: Optional[Dict[str, torch.Tensor]] = None
    prealloc_ready_event: Optional[Any] = None


@dataclasses.dataclass(frozen=True)
class _LayerKVExpertCopyDescriptor:
    seq: int
    direction: str
    reason: str
    layer_id: int
    logical_id: int
    src_slot: int
    dst_slot: int
    bytes: int
    param_count: int
    step: int


@dataclasses.dataclass
class _LayerKVExpertLayerState:
    layer_id: int
    module: Any
    orig_forward: Callable
    orig_run_moe_core: Callable
    full_num_experts: int
    slot_capacity: int
    expert_bytes: int
    device: torch.device
    dtype: torch.dtype
    cpu_params: Dict[int, Dict[str, torch.Tensor]]
    param_names: List[str]
    logical_to_slot: Dict[int, int]
    slot_to_logical: Dict[int, int]
    lru: Dict[int, int]
    hotness_prefill: Dict[int, int]
    hotness_decode: Dict[int, int]
    remap_tensor: Optional[torch.Tensor] = None
    free_slots: List[int] = dataclasses.field(default_factory=list)
    lru_heap: List[Tuple[int, int, int]] = dataclasses.field(default_factory=list)
    backing_lru: Dict[int, int] = dataclasses.field(default_factory=dict)
    last_decode_logical_ids: List[int] = dataclasses.field(default_factory=list)
    prefetched_logical_ids: Set[int] = dataclasses.field(default_factory=set)
    topk_ids_in_range_calibrated: bool = False
    topk_ids_invalid_observed: bool = False
    materialize_step: int = 0

    @property
    def full_bytes(self) -> int:
        return int(self.full_num_experts * self.expert_bytes)

    @property
    def resident_count(self) -> int:
        return len(self.logical_to_slot)

    @property
    def offloaded_count(self) -> int:
        return max(0, self.full_num_experts - self.resident_count)

    @property
    def physical_reclaim_bytes(self) -> int:
        return max(0, (self.full_num_experts - self.slot_capacity) * self.expert_bytes)
