"""LayerKV runtime adapter for SGLang v0.5.12.

The integration keeps SGLang's native KV pool and MoE execution path in place,
while adding the policy-residency lifecycle hooks needed to make the feature
runnable and measurable.  The first physical path supports MHA KV pools with
page_size=1 by persistently moving selected KV slots to compact CPU backing
storage and reloading them before attention consumes them.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LayerKVConfig:
    enabled: bool = False
    mode: str = "off"
    policy: str = "none"
    target_reclaim_mb: float = 0.0
    kvc_block_tokens: int = 16
    kvc_scheduler: str = "async-deadline"
    debug_stats: bool = False
    disallow_destructive_fallback: bool = True

    @classmethod
    def from_server_args(cls, server_args: Any) -> "LayerKVConfig":
        return cls(
            enabled=bool(getattr(server_args, "enable_layerkv", False)),
            mode=str(getattr(server_args, "layerkv_mode", "off")),
            policy=str(getattr(server_args, "layerkv_policy", "none")),
            target_reclaim_mb=float(
                getattr(server_args, "layerkv_target_reclaim_mb", 0.0) or 0.0
            ),
            kvc_block_tokens=max(
                1, int(getattr(server_args, "layerkv_kvc_block_tokens", 16) or 16)
            ),
            kvc_scheduler=str(
                getattr(server_args, "layerkv_kvc_scheduler", "async-deadline")
            ),
            debug_stats=bool(getattr(server_args, "layerkv_debug_stats", False)),
            disallow_destructive_fallback=bool(
                getattr(server_args, "layerkv_disallow_destructive_fallback", True)
            ),
        )


@dataclasses.dataclass
class LayerKVStats:
    forward_decode_count: int = 0
    forward_extend_count: int = 0
    kvc_set_kv_count: int = 0
    kvc_get_key_count: int = 0
    kvc_get_value_count: int = 0
    kvc_get_kv_count: int = 0
    kvc_tokens_written: int = 0
    kvc_bytes_written: int = 0
    requested_total_reclaim_mb: float = 0.0
    effective_kvc_reclaim_mb: float = 0.0
    policy_kvc_fraction: float = 0.0
    policy_expert_fraction: float = 0.0
    full_policy_semantics_supported: bool = True
    policy_semantics_reason: str = ""
    planner_version: str = ""
    planner_used_hotness: bool = False
    planner_fallback_reason: str = ""
    planner_estimated_kvc_cost: float = 0.0
    planner_estimated_expert_cost: float = 0.0
    planner_selected_kvc_reclaim_mb: float = 0.0
    planner_selected_expert_reclaim_mb: float = 0.0
    planned_kvc_reclaim_mb: float = 0.0
    physical_kvc_reclaim_mb: float = 0.0
    planned_expert_reclaim_mb: float = 0.0
    physical_expert_reclaim_mb: float = 0.0
    expert_host_backing_mb: float = 0.0
    expert_resident_count: int = 0
    expert_offloaded_count: int = 0
    expert_slot_capacity_total: int = 0
    expert_slot_rebind_count: int = 0
    expert_materialize_count: int = 0
    expert_topk_rewrite_count: int = 0
    expert_core_hook_count: int = 0
    expert_materialize_mb_total: float = 0.0
    expert_materialize_ms: float = 0.0
    expert_materialize_async_count: int = 0
    expert_materialize_host_sync_count: int = 0
    expert_call_count_total: int = 0
    expert_prefill_call_count_total: int = 0
    expert_decode_call_count_total: int = 0
    expert_hotness_observed: bool = False
    expert_guard_pass: bool = True
    expert_guard_reason: str = ""
    kvc_host_backing_mb: float = 0.0
    kvc_host_capacity_tokens: int = 0
    kvc_host_used_tokens: int = 0
    kvc_page_size: int = 1
    kvc_offloaded_page_count: int = 0
    kvc_resident_page_count: int = 0
    kvc_reload_page_count_total: int = 0
    kvc_evict_page_count_total: int = 0
    kvc_page_alignment_violation_count: int = 0
    kvc_resident_token_count: int = 0
    kvc_offloaded_token_count: int = 0
    kvc_evict_count_total: int = 0
    kvc_reload_count_total: int = 0
    kvc_reload_mb_total: float = 0.0
    kvc_backup_ms: float = 0.0
    kvc_reload_ms: float = 0.0
    kvc_use_point_wait_ms: float = 0.0
    kvc_allocator_free_count: int = 0
    kvc_allocator_available_before: int = -1
    kvc_allocator_available_after: int = -1
    kvc_req_to_token_rewrite_count: int = 0
    kvc_physical_cycle_count: int = 0
    kvc_physical_failure_count: int = 0
    kvc_reload_required_count: int = 0
    kvc_eviction_skipped_count: int = 0
    kvc_residency_entry_count: int = 0
    kvc_stale_entry_count: int = 0
    kvc_guard_pass: bool = True
    kvc_guard_reason: str = ""
    planner_apply_count: int = 0
    scheduler_invocation_count: int = 0
    layerkv_tasks_built: int = 0
    layerkv_kvc_reload_started: int = 0
    layerkv_expert_materialize_started: int = 0
    layerkv_deadline_miss_count: int = 0
    layerkv_copy_event_record_count: int = 0
    layerkv_copy_event_wait_count: int = 0
    kvc_ready_before_use_count: int = 0
    kvc_ready_use_check_count: int = 0
    kvc_ready_before_use_ratio: float = 1.0
    layerkv_main_stream_wait_ms: float = 0.0
    layerkv_copy_stream_busy_ms: float = 0.0
    layerkv_python_overhead_ms: float = 0.0
    observed_batch_size: int = 0
    avg_prefix_len: float = 0.0
    decode_steps: int = 0
    kvc_bytes_per_token_all_layers: int = 0
    comparable: bool = True
    comparability_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class _LayerKVResidencyEntry:
    req_idx: int
    pos: int
    state: str
    device_loc: Optional[int] = None
    host_slot: Optional[int] = None
    device_locs: Optional[List[int]] = None
    host_slots: Optional[List[int]] = None
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
            return [int(x) for x in self.device_locs]
        if self.device_loc is None:
            return []
        return [int(self.device_loc)]

    def host_slot_list(self) -> List[int]:
        if self.host_slots is not None:
            return [int(x) for x in self.host_slots]
        if self.host_slot is None:
            return []
        return [int(self.host_slot)]


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


class _LayerKVHostKVStore:
    """Compact pinned host backing for LayerKV-owned evicted MHA KV tokens."""

    def __init__(self, kv_pool: Any, capacity_tokens: int):
        self.kv_pool = kv_pool
        self.capacity_tokens = max(1, int(capacity_tokens))
        self.free_slots: List[int] = list(range(self.capacity_tokens))
        self.used_slots = set()
        self.layer_num = int(kv_pool.layer_num)
        self.start_layer = int(kv_pool.start_layer)
        self.device = kv_pool.device

        k0 = kv_pool._get_key_buffer(self.start_layer)
        v0 = kv_pool._get_value_buffer(self.start_layer)
        self.k_buffers = []
        self.v_buffers = []
        self.bytes_per_token_all_layers = int(
            (k0[0].nbytes + v0[0].nbytes) * self.layer_num
        )
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

    @staticmethod
    def _empty_cpu(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        try:
            return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        except Exception:
            return torch.empty(shape, dtype=dtype, device="cpu")

    @property
    def used_count(self) -> int:
        return len(self.used_slots)

    @property
    def used_mb(self) -> float:
        return self.used_count * self.bytes_per_token_all_layers / float(1024 * 1024)

    @property
    def capacity_mb(self) -> float:
        return self.capacity_tokens * self.bytes_per_token_all_layers / float(1024 * 1024)

    def alloc(self, need: int) -> Optional[List[int]]:
        if need > len(self.free_slots):
            return None
        slots = self.free_slots[:need]
        del self.free_slots[:need]
        self.used_slots.update(slots)
        return slots

    def free(self, slots: List[int]) -> None:
        if not slots:
            return
        for slot in slots:
            if slot in self.used_slots:
                self.used_slots.remove(slot)
                self.free_slots.append(slot)

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
            k_src = self.kv_pool._get_key_buffer(layer_id)[device_locs].detach().to(
                "cpu", non_blocking=True
            )
            v_src = self.kv_pool._get_value_buffer(layer_id)[device_locs].detach().to(
                "cpu", non_blocking=True
            )
            self.k_buffers[layer_offset].index_copy_(0, host_index, k_src)
            self.v_buffers[layer_offset].index_copy_(0, host_index, v_src)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

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
                k_src = self.k_buffers[layer_offset].index_select(0, host_index).to(
                    self.device, non_blocking=True
                )
                v_src = self.v_buffers[layer_offset].index_select(0, host_index).to(
                    self.device, non_blocking=True
                )
                self.kv_pool._get_key_buffer(layer_id).index_copy_(0, device_locs, k_src)
                self.kv_pool._get_value_buffer(layer_id).index_copy_(0, device_locs, v_src)
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


class LayerKVRuntime:
    """Flag-gated SGLang adapter for layer-aware residency.

    This adapter deliberately does not replace SGLang's KV tensors. It installs
    stable hooks around the KV pool and forward pass, and the supported physical
    path rewrites logical token-to-slot metadata after backing slots to CPU and
    reloading them into allocator-owned GPU slots.
    """

    def __init__(self, config: LayerKVConfig):
        self.config = config
        self.stats = LayerKVStats()
        self.installed = False
        self.physical_kvc_supported = False
        self.physical_expert_supported = False
        self.unsupported_reason = ""
        self._wrapped_methods: Dict[str, Callable] = {}
        self._copy_stream: Optional[torch.cuda.Stream] = None
        self._runner: Any = None
        self._kv_pool: Any = None
        self._allocator: Any = None
        self._req_to_token_pool: Any = None
        self._bytes_per_token_all_layers: int = 0
        self._page_size: int = 1
        self._host_store: Optional[_LayerKVHostKVStore] = None
        self._residency: Dict[Tuple[int, int], _LayerKVResidencyEntry] = {}
        self._expert_layers: Dict[int, _LayerKVExpertLayerState] = {}
        self._expert_modules: List[Tuple[int, Any]] = []
        self._expert_hotness_prefill: Dict[int, Dict[int, int]] = {}
        self._expert_hotness_decode: Dict[int, Dict[int, int]] = {}
        self._expert_plan_applied: bool = False
        self._pending_expert_copy_events: List[Tuple[Any, Any]] = []
        self._current_forward_mode: str = ""
        self._last_forward_batch: Any = None
        self._decode_step: int = 0

    @classmethod
    def maybe_create(cls, server_args: Any) -> Optional["LayerKVRuntime"]:
        config = LayerKVConfig.from_server_args(server_args)
        if not config.enabled or config.mode == "off":
            return None
        return cls(config)

    def install_on_runner(self, runner: Any) -> None:
        if self.installed:
            return
        t0 = time.perf_counter()
        self._runner = runner
        runner.layerkv_runtime = self
        if getattr(runner, "device", None) == "cuda":
            self._copy_stream = torch.cuda.Stream()
        self._allocator = getattr(runner, "token_to_kv_pool_allocator", None)
        self._req_to_token_pool = getattr(runner, "req_to_token_pool", None)
        self._install_kv_pool_hooks(getattr(runner, "token_to_kv_pool", None))
        self._discover_expert_support(runner)
        self.installed = True
        self.stats.layerkv_python_overhead_ms += (time.perf_counter() - t0) * 1000.0
        logger.info(
            "LayerKV enabled mode=%s policy=%s target_reclaim_mb=%.1f "
            "kvc_supported=%s expert_supported=%s reason=%s",
            self.config.mode,
            self.config.policy,
            self.config.target_reclaim_mb,
            self.physical_kvc_supported,
            self.physical_expert_supported,
            self.unsupported_reason,
        )

    def _install_kv_pool_hooks(self, kv_pool: Any) -> None:
        if kv_pool is None:
            self.unsupported_reason = "token_to_kv_pool is not initialized"
            return
        self._kv_pool = kv_pool
        if getattr(kv_pool, "_layerkv_wrapped", False):
            return

        def wrap(name: str, wrapper_factory: Callable[[Callable], Callable]) -> None:
            orig = getattr(kv_pool, name, None)
            if orig is None or not callable(orig):
                return
            self._wrapped_methods[name] = orig
            setattr(kv_pool, name, wrapper_factory(orig))

        wrap("set_kv_buffer", self._wrap_set_kv_buffer)
        wrap("get_key_buffer", self._wrap_get_key_buffer)
        wrap("get_value_buffer", self._wrap_get_value_buffer)
        wrap("get_kv_buffer", self._wrap_get_kv_buffer)
        kv_pool._layerkv_wrapped = True
        kv_pool.layerkv_runtime = self
        self.physical_kvc_supported = self._can_support_kvc_pool(kv_pool)
        if not self.physical_kvc_supported and not self.unsupported_reason:
            self.unsupported_reason = (
                f"KVC physical offload is not implemented for "
                f"{type(kv_pool).__name__}; running accounting-only hooks"
            )
        if (
            self.config.mode == "kvc-only"
            and self.config.target_reclaim_mb > 0
            and not self.physical_kvc_supported
        ):
            self.stats.comparable = False
            self.stats.comparability_reason = self.unsupported_reason

    def _can_support_kvc_pool(self, kv_pool: Any) -> bool:
        try:
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        except Exception:
            return False

        page_size = int(getattr(kv_pool, "page_size", 1) or 1)
        if type(kv_pool) is not MHATokenToKVPool:
            self.unsupported_reason = (
                f"KVC physical offload supports non-FP4 MHATokenToKVPool only, got "
                f"{type(kv_pool).__name__}"
            )
            return False
        if self._allocator is None or self._req_to_token_pool is None:
            self.unsupported_reason = "KVC allocator or req_to_token_pool is missing"
            return False

        try:
            self._page_size = max(1, page_size)
            self.stats.kvc_page_size = self._page_size
            one_k = kv_pool._get_key_buffer(kv_pool.start_layer)[0].nbytes
            one_v = kv_pool._get_value_buffer(kv_pool.start_layer)[0].nbytes
            self._bytes_per_token_all_layers = int(
                (one_k + one_v) * kv_pool.layer_num
            )
        except Exception as exc:
            self.unsupported_reason = f"failed to inspect KV token size: {exc}"
            return False
        return self._bytes_per_token_all_layers > 0

    def _ensure_host_store(self) -> None:
        if self._host_store is not None:
            return
        if self._kv_pool is None or self._bytes_per_token_all_layers <= 0:
            raise RuntimeError("KVC host store cannot initialize without KV pool")
        # Allocate host capacity against the total reclaim intent so dynamic
        # policy fractions can vary without reallocating CPU backing.
        target_bytes = max(1, int(self.config.target_reclaim_mb * 1024 * 1024))
        target_tokens = max(1, target_bytes // self._bytes_per_token_all_layers)
        target_tokens = self._align_tokens_up(target_tokens)
        capacity_tokens = target_tokens + max(
            target_tokens, self._align_tokens_up(self.config.kvc_block_tokens)
        )
        self._host_store = _LayerKVHostKVStore(self._kv_pool, capacity_tokens)
        self.stats.kvc_host_capacity_tokens = self._host_store.capacity_tokens
        self.stats.kvc_host_backing_mb = self._host_store.capacity_mb

    def _discover_expert_support(self, runner: Any) -> None:
        model = getattr(runner, "model", None)
        if model is None:
            return
        if self.config.mode != "kvc-expert":
            return
        supported = []
        unsupported_reasons = []
        for module in model.modules():
            if not self._is_supported_fused_moe(module):
                continue
            layer_id = int(getattr(module, "layer_id", len(supported)))
            supported.append((layer_id, module))
        self._expert_modules = supported
        if not supported:
            self.physical_expert_supported = False
            if not self.unsupported_reason:
                self.unsupported_reason = "no supported standard FusedMoE layers found"
            return
        for layer_id, module in supported:
            reason = self._expert_layer_unsupported_reason(module)
            if reason:
                unsupported_reasons.append(f"layer{layer_id}:{reason}")
        if unsupported_reasons:
            self.physical_expert_supported = False
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = ";".join(unsupported_reasons[:8])
            if not self.unsupported_reason:
                self.unsupported_reason = self.stats.expert_guard_reason
            return
        self.physical_expert_supported = True
        for layer_id, module in supported:
            self._install_expert_hotness_probe(module, layer_id)

    def _install_expert_hotness_probe(self, module: Any, layer_id: int) -> None:
        if getattr(module, "_layerkv_expert_wrapped", False):
            return
        if getattr(module, "_layerkv_hotness_wrapped", False):
            return
        full_num_experts = int(module.w13_weight.data.shape[0])
        orig_run_moe_core = module.run_moe_core

        @functools.wraps(orig_run_moe_core)
        def wrapped_run_moe_core(dispatch_output: Any, *args, **kwargs):
            topk_output = getattr(dispatch_output, "topk_output", None)
            topk_ids = getattr(topk_output, "topk_ids", None)
            if topk_ids is not None:
                self._record_expert_hotness_for_layer(
                    layer_id=layer_id,
                    full_num_experts=full_num_experts,
                    topk_ids=topk_ids,
                )
            return orig_run_moe_core(dispatch_output, *args, **kwargs)

        module._layerkv_hotness_orig_run_moe_core = orig_run_moe_core
        module._layerkv_hotness_wrapped = True
        module.run_moe_core = wrapped_run_moe_core

    def _is_supported_fused_moe(self, module: Any) -> bool:
        return (
            hasattr(module, "w13_weight")
            and hasattr(module, "w2_weight")
            and hasattr(module, "forward")
            and hasattr(module, "run_moe_core")
            and hasattr(module, "num_local_experts")
            and hasattr(module, "moe_runner_config")
        )

    def _expert_layer_unsupported_reason(self, module: Any) -> str:
        if int(getattr(module, "moe_ep_size", 1) or 1) != 1:
            return "expert offload v1 supports moe_ep_size=1 only"
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            return "missing quant_method"
        if quant_method.__class__.__name__ != "UnquantizedFusedMoEMethod":
            return f"unsupported quant_method={quant_method.__class__.__name__}"
        w13 = getattr(module, "w13_weight", None)
        w2 = getattr(module, "w2_weight", None)
        if w13 is None or w2 is None or w13.data.dim() != 3 or w2.data.dim() != 3:
            return "expected 3D w13_weight/w2_weight"
        if int(w13.data.shape[0]) != int(w2.data.shape[0]):
            return "w13/w2 expert count mismatch"
        if not w13.data.is_cuda and getattr(self._runner, "device", None) == "cuda":
            return "expert weights are not on CUDA"
        return ""

    def _avg_prefix_len(self, forward_batch: Any) -> float:
        pairs = self._batch_req_indices_and_lens(forward_batch)
        if not pairs:
            return 0.0
        return sum(max(0, seq_len - 1) for _, seq_len in pairs) / float(len(pairs))

    def _policy_fractions(self, forward_batch: Any) -> Tuple[float, float, bool, str]:
        policy = self.config.policy
        if policy == "none":
            return 0.0, 0.0, True, "policy=none disables LayerKV physical reclaim"
        if self.config.mode == "kvc-only":
            if policy in (
                "expert-first",
                "kv-first",
                "ratio-75-25",
                "ratio-50-50",
                "ratio-25-75",
                "layer-aware-joint",
                "layer-aware-joint-dp",
            ):
                return 1.0, 0.0, True, ""
            return 0.0, 0.0, False, f"unknown LayerKV policy: {policy}"
        if policy == "expert-first":
            return 1.0, 0.0, True, ""
        if policy == "kv-first":
            return 0.0, 1.0, self.physical_expert_supported, self._expert_support_reason()
        if policy == "ratio-75-25":
            return 0.75, 0.25, self.physical_expert_supported, self._expert_support_reason()
        if policy == "ratio-50-50":
            return 0.50, 0.50, self.physical_expert_supported, self._expert_support_reason()
        if policy == "ratio-25-75":
            return 0.25, 0.75, self.physical_expert_supported, self._expert_support_reason()
        if policy in ("layer-aware-joint", "layer-aware-joint-dp"):
            return self._joint_policy_fractions(forward_batch)
        return 0.0, 0.0, False, f"unknown LayerKV policy: {policy}"

    def _joint_policy_fractions(self, forward_batch: Any) -> Tuple[float, float, bool, str]:
        self.stats.planner_apply_count += 1
        target_mb = max(0.0, self.config.target_reclaim_mb)
        self.stats.planner_version = "hotness-aware-v1"
        if not self.physical_expert_supported:
            reason = self._expert_support_reason()
            self._set_joint_planner_choice(
                kvc_fraction=1.0,
                expert_fraction=0.0,
                used_hotness=False,
                fallback_reason=reason,
                kvc_cost=0.0,
                expert_cost=1.0e30,
            )
            return 1.0, 0.0, False, reason

        has_hotness = self._has_expert_hotness()
        if not has_hotness:
            kvc_fraction = self._context_heuristic_kvc_fraction(forward_batch)
            expert_fraction = 1.0 - kvc_fraction
            self._set_joint_planner_choice(
                kvc_fraction=kvc_fraction,
                expert_fraction=expert_fraction,
                used_hotness=False,
                fallback_reason="no_hotness",
                kvc_cost=self._estimate_kvc_reclaim_cost(target_mb * kvc_fraction, forward_batch),
                expert_cost=0.0,
            )
            return kvc_fraction, expert_fraction, True, "planner_fallback=no_hotness"

        best: Optional[Tuple[float, float, float, float, float]] = None
        for kvc_fraction in (0.0, 0.25, 0.50, 0.75, 1.0):
            expert_fraction = 1.0 - kvc_fraction
            kvc_mb = target_mb * kvc_fraction
            expert_mb = target_mb * expert_fraction
            kvc_cost = self._estimate_kvc_reclaim_cost(kvc_mb, forward_batch)
            expert_cost = self._estimate_expert_reclaim_cost(expert_mb)
            total_cost = kvc_cost + expert_cost
            candidate = (total_cost, kvc_fraction, expert_fraction, kvc_cost, expert_cost)
            if best is None or candidate < best:
                best = candidate

        assert best is not None
        _, kvc_fraction, expert_fraction, kvc_cost, expert_cost = best
        self._set_joint_planner_choice(
            kvc_fraction=kvc_fraction,
            expert_fraction=expert_fraction,
            used_hotness=True,
            fallback_reason="",
            kvc_cost=kvc_cost,
            expert_cost=expert_cost,
        )
        return kvc_fraction, expert_fraction, True, ""

    def _context_heuristic_kvc_fraction(self, forward_batch: Any) -> float:
        avg_prefix = self._avg_prefix_len(forward_batch)
        if avg_prefix <= 1024:
            return 0.75
        if avg_prefix <= 4096:
            return 0.50
        return 0.25

    def _set_joint_planner_choice(
        self,
        *,
        kvc_fraction: float,
        expert_fraction: float,
        used_hotness: bool,
        fallback_reason: str,
        kvc_cost: float,
        expert_cost: float,
    ) -> None:
        target_mb = max(0.0, self.config.target_reclaim_mb)
        self.stats.planner_used_hotness = used_hotness
        self.stats.planner_fallback_reason = fallback_reason
        self.stats.planner_estimated_kvc_cost = float(kvc_cost)
        self.stats.planner_estimated_expert_cost = float(expert_cost)
        self.stats.planner_selected_kvc_reclaim_mb = target_mb * kvc_fraction
        self.stats.planner_selected_expert_reclaim_mb = target_mb * expert_fraction

    def _has_expert_hotness(self) -> bool:
        for hotness in self._expert_hotness_decode.values():
            if hotness:
                return True
        for hotness in self._expert_hotness_prefill.values():
            if hotness:
                return True
        for state in self._expert_layers.values():
            if state.hotness_decode or state.hotness_prefill:
                return True
        return False

    def _estimate_kvc_reclaim_cost(self, reclaim_mb: float, forward_batch: Any) -> float:
        if reclaim_mb <= 0.0:
            return 0.0
        avg_prefix = self._avg_prefix_len(forward_batch)
        cost_per_mb = max(0.1, avg_prefix / 1024.0)
        return reclaim_mb * cost_per_mb

    def _estimate_expert_reclaim_cost(self, reclaim_mb: float) -> float:
        if reclaim_mb <= 0.0:
            return 0.0
        candidates: List[Tuple[float, int]] = []
        if self._expert_layers:
            layer_items = [
                (state.layer_id, state.full_num_experts, state.expert_bytes)
                for state in self._expert_layers.values()
            ]
        else:
            layer_items = [
                (
                    int(layer_id),
                    int(module.w13_weight.data.shape[0]),
                    self._expert_bytes(module),
                )
                for layer_id, module in self._expert_modules
            ]
        for layer_id, full_num_experts, expert_bytes in layer_items:
            hotness = self._expert_hotness_decode.get(layer_id) or self._expert_hotness_prefill.get(layer_id, {})
            total_calls = max(1, int(sum(hotness.values())))
            for expert_id in range(full_num_experts):
                p = float(hotness.get(expert_id, 0)) / float(total_calls)
                candidates.append((p, expert_bytes))
        candidates.sort(key=lambda item: item[0])
        target_bytes = int(reclaim_mb * 1024 * 1024)
        reclaimed = 0
        cost = 0.0
        for p, expert_bytes in candidates:
            if reclaimed >= target_bytes:
                break
            reclaimed += expert_bytes
            cost += p * (expert_bytes / float(1024 * 1024))
        if reclaimed < target_bytes:
            return 1.0e30
        return cost

    def _expert_support_reason(self) -> str:
        if self.physical_expert_supported:
            return ""
        return self.stats.expert_guard_reason or self.unsupported_reason or "expert offload unsupported"

    def _effective_kvc_reclaim_mb(self, forward_batch: Any) -> float:
        kvc_fraction, expert_fraction, full_supported, reason = self._policy_fractions(
            forward_batch
        )
        effective = max(0.0, self.config.target_reclaim_mb * kvc_fraction)
        self.stats.requested_total_reclaim_mb = self.config.target_reclaim_mb
        self.stats.effective_kvc_reclaim_mb = effective
        self.stats.policy_kvc_fraction = kvc_fraction
        self.stats.policy_expert_fraction = expert_fraction
        self.stats.full_policy_semantics_supported = full_supported
        self.stats.policy_semantics_reason = reason
        self.stats.planned_kvc_reclaim_mb = effective
        return effective

    def _effective_expert_reclaim_mb(self, forward_batch: Any) -> float:
        kvc_fraction, expert_fraction, full_supported, reason = self._policy_fractions(
            forward_batch
        )
        effective = max(0.0, self.config.target_reclaim_mb * expert_fraction)
        self.stats.requested_total_reclaim_mb = self.config.target_reclaim_mb
        self.stats.policy_kvc_fraction = kvc_fraction
        self.stats.policy_expert_fraction = expert_fraction
        self.stats.full_policy_semantics_supported = full_supported
        self.stats.policy_semantics_reason = reason
        self.stats.planned_expert_reclaim_mb = effective
        return effective

    def _apply_expert_plan_once(self, forward_batch: Any) -> None:
        if self._expert_plan_applied or self.config.mode != "kvc-expert":
            return
        target_mb = self._effective_expert_reclaim_mb(forward_batch)
        if target_mb <= 0:
            self._expert_plan_applied = True
            self._refresh_expert_stats()
            return
        if not self.physical_expert_supported:
            self.stats.comparable = False
            self.stats.comparability_reason = self._expert_support_reason()
            return
        if not self._expert_modules:
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = "no supported expert modules discovered"
            self.stats.comparable = False
            self.stats.comparability_reason = self.stats.expert_guard_reason
            return

        full_bytes = sum(self._expert_full_bytes(module) for _, module in self._expert_modules)
        if full_bytes <= 0:
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = "failed to inspect expert bytes"
            return

        slot_capacities = self._plan_expert_slot_capacities(target_mb)
        for layer_id, module in self._expert_modules:
            slot_capacity = slot_capacities[layer_id]
            state = self._install_expert_layer_slots(module, layer_id, slot_capacity)
            self._expert_layers[layer_id] = state
        self._expert_plan_applied = True
        self.stats.expert_slot_rebind_count += len(self._expert_layers)
        self._refresh_expert_stats()

    def _plan_expert_slot_capacities(self, target_mb: float) -> Dict[int, int]:
        target_bytes = max(0, int(target_mb * 1024 * 1024))
        layer_infos: List[Dict[str, int]] = []
        for layer_id, module in self._expert_modules:
            full_num_experts = int(module.w13_weight.data.shape[0])
            top_k = int(getattr(module, "top_k", 0) or 0)
            if top_k <= 0:
                top_k = int(getattr(module.moe_runner_config, "top_k", 1) or 1)
            min_capacity = max(1, min(full_num_experts, top_k))
            layer_infos.append(
                {
                    "layer_id": int(layer_id),
                    "capacity": full_num_experts,
                    "min_capacity": min_capacity,
                    "expert_bytes": self._expert_bytes(module),
                }
            )

        reclaimed = 0
        # Spread removals across layers while still preferring larger expert slots
        # for heterogeneous MoE variants.
        layer_infos.sort(key=lambda x: (-x["expert_bytes"], x["layer_id"]))
        while reclaimed < target_bytes:
            progressed = False
            for info in layer_infos:
                if reclaimed >= target_bytes:
                    break
                if info["capacity"] <= info["min_capacity"]:
                    continue
                info["capacity"] -= 1
                reclaimed += info["expert_bytes"]
                progressed = True
            if not progressed:
                break

        return {info["layer_id"]: info["capacity"] for info in layer_infos}

    def _expert_param_names(self, module: Any) -> List[str]:
        names = ["w13_weight", "w2_weight"]
        for optional in ("w13_weight_bias", "w2_weight_bias"):
            if hasattr(module, optional):
                names.append(optional)
        return names

    def _expert_full_bytes(self, module: Any) -> int:
        total = 0
        for name in self._expert_param_names(module):
            tensor = getattr(module, name).data
            total += int(tensor.nbytes)
        return total

    def _expert_bytes(self, module: Any) -> int:
        total = 0
        for name in self._expert_param_names(module):
            tensor = getattr(module, name).data
            total += int(tensor[0].nbytes)
        return total

    def _install_expert_layer_slots(
        self, module: Any, layer_id: int, slot_capacity: int
    ) -> _LayerKVExpertLayerState:
        if getattr(module, "_layerkv_expert_wrapped", False):
            return self._expert_layers[layer_id]

        param_names = self._expert_param_names(module)
        full_num_experts = int(module.w13_weight.data.shape[0])
        device = module.w13_weight.data.device
        dtype = module.w13_weight.data.dtype
        cpu_params: Dict[int, Dict[str, torch.Tensor]] = {}
        expert_bytes = 0

        with torch.no_grad():
            for name in param_names:
                param = getattr(module, name)
                expert_bytes += int(param.data[0].nbytes)

            initial_resident = list(range(slot_capacity))
            logical_to_slot = {expert_id: expert_id for expert_id in initial_resident}
            slot_to_logical = {expert_id: expert_id for expert_id in initial_resident}
            lru = {expert_id: self._decode_step for expert_id in initial_resident}
            for expert_id in range(slot_capacity, full_num_experts):
                cpu_params[expert_id] = self._copy_expert_to_cpu(
                    module, param_names, expert_id
                )

            for name in param_names:
                param = getattr(module, name)
                old = param.data
                new_data = torch.empty(
                    (slot_capacity,) + tuple(old.shape[1:]),
                    dtype=old.dtype,
                    device=old.device,
                )
                if slot_capacity > 0:
                    new_data.copy_(old[:slot_capacity])
                param.data = new_data

        state = _LayerKVExpertLayerState(
            layer_id=layer_id,
            module=module,
            orig_forward=module.forward,
            orig_run_moe_core=getattr(
                module, "_layerkv_hotness_orig_run_moe_core", module.run_moe_core
            ),
            full_num_experts=full_num_experts,
            slot_capacity=slot_capacity,
            expert_bytes=expert_bytes,
            device=device,
            dtype=dtype,
            cpu_params=cpu_params,
            param_names=param_names,
            logical_to_slot=logical_to_slot,
            slot_to_logical=slot_to_logical,
            lru=lru,
            hotness_prefill=self._expert_hotness_prefill.setdefault(layer_id, {}),
            hotness_decode=self._expert_hotness_decode.setdefault(layer_id, {}),
        )

        @functools.wraps(module.run_moe_core)
        def wrapped_run_moe_core(dispatch_output: Any, *args, **kwargs):
            rewritten_dispatch = self._prepare_expert_dispatch_for_core(
                state, dispatch_output
            )
            return state.orig_run_moe_core(rewritten_dispatch, *args, **kwargs)

        module.run_moe_core = wrapped_run_moe_core
        module._layerkv_expert_wrapped = True
        module._layerkv_expert_state = state

        try:
            module.num_experts = slot_capacity
            module.num_local_experts = slot_capacity
            module.moe_runner_config.num_experts = slot_capacity
            module.moe_runner_config.num_local_experts = slot_capacity
            module.dispatcher.num_experts = slot_capacity
            module.dispatcher.num_local_experts = slot_capacity
            module.dispatcher.num_local_routed_experts = slot_capacity
        except Exception:
            pass
        return state

    def _copy_expert_to_cpu(
        self, module: Any, param_names: List[str], expert_id: int
    ) -> Dict[str, torch.Tensor]:
        return {
            name: self._cpu_backing_tensor(
                getattr(module, name).data[expert_id].detach()
            )
            for name in param_names
        }

    def _copy_slot_to_cpu(
        self, state: _LayerKVExpertLayerState, slot_id: int
    ) -> Dict[str, torch.Tensor]:
        return {
            name: self._cpu_backing_tensor(
                getattr(state.module, name).data[slot_id].detach()
            )
            for name in state.param_names
        }

    @staticmethod
    def _cpu_backing_tensor(tensor: torch.Tensor) -> torch.Tensor:
        cpu = tensor.to("cpu", copy=True)
        try:
            return cpu.pin_memory()
        except Exception:
            return cpu

    def _prepare_expert_dispatch_for_core(
        self, state: _LayerKVExpertLayerState, dispatch_output: Any
    ) -> Any:
        topk_output = getattr(dispatch_output, "topk_output", None)
        if topk_output is None:
            return dispatch_output
        rewritten_topk = self._prepare_expert_layer_for_topk(state, topk_output)
        self.stats.expert_core_hook_count += 1
        if rewritten_topk is topk_output:
            return dispatch_output
        if hasattr(dispatch_output, "_replace"):
            return dispatch_output._replace(topk_output=rewritten_topk)
        reason = (
            f"layer {state.layer_id} dispatch output does not support topk rewrite"
        )
        self.stats.expert_guard_pass = False
        self.stats.expert_guard_reason = reason
        raise RuntimeError(reason)

    def _prepare_expert_layer_for_topk(self, state: _LayerKVExpertLayerState, topk_output: Any) -> Any:
        topk_ids = getattr(topk_output, "topk_ids", None)
        if topk_ids is None:
            return topk_output
        if bool((topk_ids >= state.full_num_experts).any().item()):
            reason = f"layer {state.layer_id} produced out-of-range expert id"
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = reason
            raise RuntimeError(reason)
        self._record_expert_hotness(state, topk_ids)
        logical_ids = self._unique_expert_ids(topk_ids, state.full_num_experts)
        if len(logical_ids) > state.slot_capacity:
            reason = (
                f"layer {state.layer_id} needs {len(logical_ids)} unique experts "
                f"but slot capacity is {state.slot_capacity}"
            )
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = reason
            self.stats.comparable = False
            self.stats.comparability_reason = reason
            raise RuntimeError(reason)
        self._materialize_experts(state, logical_ids)
        remap = torch.full(
            (state.full_num_experts,),
            -1,
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        for logical_id, slot_id in state.logical_to_slot.items():
            remap[int(logical_id)] = int(slot_id)
        valid = (topk_ids >= 0) & (topk_ids < state.full_num_experts)
        safe_ids = topk_ids.clamp(min=0, max=state.full_num_experts - 1).long()
        rewritten_ids = torch.where(valid, remap[safe_ids], topk_ids)
        if bool((rewritten_ids < 0).any().item()):
            reason = f"layer {state.layer_id} produced unmapped expert id"
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = reason
            raise RuntimeError(reason)
        self.stats.expert_topk_rewrite_count += 1
        return topk_output._replace(topk_ids=rewritten_ids)

    def _record_expert_hotness(
        self, state: _LayerKVExpertLayerState, topk_ids: torch.Tensor
    ) -> None:
        self._record_expert_hotness_for_layer(
            layer_id=state.layer_id,
            full_num_experts=state.full_num_experts,
            topk_ids=topk_ids,
        )

    def _record_expert_hotness_for_layer(
        self, layer_id: int, full_num_experts: int, topk_ids: torch.Tensor
    ) -> None:
        if topk_ids.numel() == 0:
            return
        ids = topk_ids.detach()
        ids = ids[(ids >= 0) & (ids < full_num_experts)]
        if ids.numel() == 0:
            return
        target = (
            self._expert_hotness_decode.setdefault(layer_id, {})
            if self._current_forward_mode == "decode"
            else self._expert_hotness_prefill.setdefault(layer_id, {})
        )
        values, counts = torch.unique(ids.cpu(), return_counts=True)
        total = 0
        for expert_id, count in zip(values.tolist(), counts.tolist()):
            expert_id = int(expert_id)
            count = int(count)
            target[expert_id] = target.get(expert_id, 0) + count
            total += count
        self.stats.expert_call_count_total += total
        if self._current_forward_mode == "decode":
            self.stats.expert_decode_call_count_total += total
        else:
            self.stats.expert_prefill_call_count_total += total
        self.stats.expert_hotness_observed = True

    def _finalize_expert_materialize_events(self, *, block: bool = False) -> None:
        if not self._pending_expert_copy_events:
            return
        remaining = []
        for start, end in self._pending_expert_copy_events:
            try:
                if block:
                    end.synchronize()
                elif not end.query():
                    remaining.append((start, end))
                    continue
                elapsed = float(start.elapsed_time(end))
                self.stats.expert_materialize_ms += elapsed
                self.stats.layerkv_copy_stream_busy_ms += elapsed
            except Exception:
                continue
        self._pending_expert_copy_events = remaining

    def _unique_expert_ids(self, topk_ids: torch.Tensor, full_num_experts: int) -> List[int]:
        if topk_ids.numel() == 0:
            return []
        ids = topk_ids.detach()
        ids = ids[(ids >= 0) & (ids < full_num_experts)]
        if ids.numel() == 0:
            return []
        return [int(x) for x in torch.unique(ids).detach().cpu().tolist()]

    def _materialize_experts(
        self, state: _LayerKVExpertLayerState, logical_ids: List[int]
    ) -> None:
        if not logical_ids:
            return
        protected = set(int(x) for x in logical_ids)
        for logical_id in logical_ids:
            if logical_id in state.logical_to_slot:
                state.lru[logical_id] = self._decode_step
                continue
            slot_id = self._choose_expert_slot_for_materialize(state, protected)
            evicted = state.slot_to_logical.get(slot_id)
            if evicted is not None:
                state.cpu_params[int(evicted)] = self._copy_slot_to_cpu(state, slot_id)
                state.logical_to_slot.pop(evicted, None)
                state.lru.pop(evicted, None)
            source_params = state.cpu_params.get(int(logical_id))
            if source_params is None:
                reason = (
                    f"layer {state.layer_id} missing CPU backing for expert {logical_id}"
                )
                self.stats.expert_guard_pass = False
                self.stats.expert_guard_reason = reason
                raise RuntimeError(reason)
            start = None
            end = None
            active_stream = None
            if state.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                active_stream = torch.cuda.current_stream(device=state.device)
            with torch.no_grad():
                if active_stream is not None:
                    start.record(active_stream)
                for name in state.param_names:
                    dst = getattr(state.module, name).data[slot_id]
                    dst.copy_(source_params[name], non_blocking=True)
            state.cpu_params.pop(int(logical_id), None)
            if end is not None:
                end.record(active_stream)
                try:
                    if end.query():
                        elapsed = float(start.elapsed_time(end))
                        self.stats.expert_materialize_ms += elapsed
                        self.stats.layerkv_copy_stream_busy_ms += elapsed
                    else:
                        self.stats.expert_materialize_async_count += 1
                        self._pending_expert_copy_events.append((start, end))
                except Exception:
                    pass
            else:
                self.stats.expert_materialize_host_sync_count += 1
            state.logical_to_slot[logical_id] = slot_id
            state.slot_to_logical[slot_id] = logical_id
            state.lru[logical_id] = self._decode_step
            state.materialize_step += 1
            self.stats.expert_materialize_count += 1
            self.stats.layerkv_expert_materialize_started += 1
            self.stats.layerkv_tasks_built += 1
            self.stats.expert_materialize_mb_total += state.expert_bytes / float(1024 * 1024)

    def _choose_expert_slot_for_materialize(
        self, state: _LayerKVExpertLayerState, protected: Set[int]
    ) -> int:
        free = [
            slot_id
            for slot_id in range(state.slot_capacity)
            if slot_id not in state.slot_to_logical
        ]
        if free:
            return free[0]
        candidates = [
            (state.lru.get(logical_id, -1), slot_id, logical_id)
            for slot_id, logical_id in state.slot_to_logical.items()
            if logical_id not in protected
        ]
        if not candidates:
            raise RuntimeError(
                f"layer {state.layer_id} has no evictable expert slot for materialization"
            )
        candidates.sort()
        return int(candidates[0][1])

    def _refresh_expert_stats(self) -> None:
        if not self._expert_layers:
            return
        full_bytes = sum(state.full_bytes for state in self._expert_layers.values())
        reclaim_bytes = sum(
            state.physical_reclaim_bytes for state in self._expert_layers.values()
        )
        host_bytes = sum(
            sum(
                int(t.nbytes)
                for expert_params in state.cpu_params.values()
                for t in expert_params.values()
            )
            for state in self._expert_layers.values()
        )
        self.stats.physical_expert_reclaim_mb = reclaim_bytes / float(1024 * 1024)
        self.stats.expert_host_backing_mb = host_bytes / float(1024 * 1024)
        self.stats.expert_slot_capacity_total = sum(
            state.slot_capacity for state in self._expert_layers.values()
        )
        self.stats.expert_resident_count = sum(
            state.resident_count for state in self._expert_layers.values()
        )
        self.stats.expert_offloaded_count = sum(
            state.offloaded_count for state in self._expert_layers.values()
        )
        if full_bytes > 0 and self.stats.planned_expert_reclaim_mb > 0:
            if self.stats.physical_expert_reclaim_mb + 1e-3 < min(
                self.stats.planned_expert_reclaim_mb, full_bytes / float(1024 * 1024)
            ):
                self.stats.comparable = False
                self.stats.comparability_reason = "INSUFFICIENT_EXPERT_RECLAIM"
            elif self.stats.comparability_reason == "INSUFFICIENT_EXPERT_RECLAIM":
                self.stats.comparable = True
                self.stats.comparability_reason = ""

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

    def _align_tokens_down(self, token_count: int) -> int:
        page_size = max(1, int(self._page_size))
        return max(0, int(token_count) // page_size * page_size)

    def _align_tokens_up(self, token_count: int) -> int:
        page_size = max(1, int(self._page_size))
        if token_count <= 0:
            return 0
        return int(math.ceil(float(token_count) / float(page_size)) * page_size)

    def _wrap_set_kv_buffer(self, orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapped(layer: Any, loc: Any, cache_k: Any, cache_v: Any, *args, **kwargs):
            t0 = time.perf_counter()
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
            self.stats.layerkv_python_overhead_ms += (time.perf_counter() - t0) * 1000.0
            return ret

        return wrapped

    def _wrap_get_key_buffer(self, orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapped(layer_id: int, *args, **kwargs):
            self.stats.kvc_get_key_count += 1
            self._wait_for_kvc_layer_ready(layer_id)
            return orig(layer_id, *args, **kwargs)

        return wrapped

    def _wrap_get_value_buffer(self, orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapped(layer_id: int, *args, **kwargs):
            self.stats.kvc_get_value_count += 1
            self._wait_for_kvc_layer_ready(layer_id)
            return orig(layer_id, *args, **kwargs)

        return wrapped

    def _wrap_get_kv_buffer(self, orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapped(layer_id: int, *args, **kwargs):
            self.stats.kvc_get_kv_count += 1
            self._wait_for_kvc_layer_ready(layer_id)
            return orig(layer_id, *args, **kwargs)

        return wrapped

    def _wait_for_kvc_layer_ready(self, layer_id: int) -> None:
        pending = [
            entry
            for entry in self._residency.values()
            if entry.state == "reloading" and entry.ready_event is not None
        ]
        if not pending:
            return
        seen_events = set()
        for entry in pending:
            event = entry.ready_event
            event_id = id(event)
            if event_id in seen_events:
                continue
            seen_events.add(event_id)
            self.stats.kvc_ready_use_check_count += 1
            ready = bool(event.query())
            if ready:
                self.stats.kvc_ready_before_use_count += 1
            else:
                self.stats.layerkv_deadline_miss_count += 1
            stream = torch.cuda.current_stream(device=self._kv_pool.device)
            stream.wait_event(event)
            self.stats.layerkv_copy_event_wait_count += 1
        self._refresh_ready_before_use_ratio()

    def _refresh_ready_before_use_ratio(self) -> None:
        total = self.stats.kvc_ready_use_check_count
        if total <= 0:
            self.stats.kvc_ready_before_use_ratio = 1.0
        else:
            self.stats.kvc_ready_before_use_ratio = (
                self.stats.kvc_ready_before_use_count / float(total)
            )

    def _finalize_reloaded_entries(self, *, block: bool = False) -> None:
        if self._host_store is None:
            return
        pending = [
            entry for entry in self._residency.values() if entry.state == "reloading"
        ]
        if not pending:
            return
        elapsed_events = set()
        for entry in pending:
            event = entry.ready_event
            if event is not None:
                if block:
                    event.synchronize()
                elif not event.query():
                    continue
            event_id = id(entry.ready_event) if entry.ready_event is not None else None
            if (
                event_id not in elapsed_events
                and entry.ready_start_event is not None
                and entry.ready_event is not None
            ):
                try:
                    elapsed_ms = float(
                        entry.ready_start_event.elapsed_time(entry.ready_event)
                    )
                    self.stats.kvc_reload_ms += elapsed_ms
                    self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
                    elapsed_events.add(event_id)
                except Exception:
                    pass
            host_slots = entry.host_slot_list()
            if host_slots:
                self._host_store.free(host_slots)
            entry.host_slots = None
            entry.host_slot = None
            entry.state = "resident"
            entry.ready_start_event = None
            entry.ready_event = None
            entry.ready_waited = False
        self._refresh_kvc_residency_stats()

    def on_forward_begin(self, *, mode: str, forward_batch: Any) -> None:
        t0 = time.perf_counter()
        self._current_forward_mode = mode
        self._last_forward_batch = forward_batch
        self._refresh_workload_stats(forward_batch)
        self._finalize_expert_materialize_events(block=False)
        self._finalize_reloaded_entries(block=False)
        if mode == "decode":
            self._apply_expert_plan_once(forward_batch)
            self.stats.forward_decode_count += 1
            self._decode_step += 1
            self.stats.decode_steps = self._decode_step
            self._reload_required_kvc(forward_batch)
        else:
            self.stats.forward_extend_count += 1
            self._drop_entries_for_reqs(forward_batch)
        self.stats.scheduler_invocation_count += 1
        self.stats.layerkv_python_overhead_ms += (time.perf_counter() - t0) * 1000.0

    def _refresh_workload_stats(self, forward_batch: Any) -> None:
        pairs = self._batch_req_indices_and_lens(forward_batch)
        self.stats.observed_batch_size = len(pairs)
        if pairs:
            self.stats.avg_prefix_len = sum(
                max(0, seq_len - 1) for _, seq_len in pairs
            ) / float(len(pairs))
        self.stats.kvc_bytes_per_token_all_layers = int(
            self._bytes_per_token_all_layers
        )

    def _reload_required_kvc(self, forward_batch: Any) -> None:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return
        effective_kvc_reclaim_mb = self._effective_kvc_reclaim_mb(forward_batch)
        if effective_kvc_reclaim_mb <= 0:
            self._refresh_kvc_residency_stats()
            return
        if not self.physical_kvc_supported:
            self.stats.comparable = False
            self.stats.comparability_reason = self.unsupported_reason
            return
        if self._bytes_per_token_all_layers <= 0:
            return

        self._prune_entries_for_active_lengths(forward_batch)
        selected = self._select_required_offloaded_entries(forward_batch)
        if not selected:
            self._refresh_kvc_residency_stats()
            return

        token_count = sum(entry.token_count for entry in selected)
        self.stats.kvc_reload_required_count += token_count
        self.stats.kvc_allocator_available_before = self._allocator_available_size()
        new_locs = self._allocator.alloc(token_count)
        if new_locs is None:
            self.stats.kvc_physical_failure_count += 1
            self.stats.comparable = False
            self.stats.comparability_reason = "allocator failed to reload offloaded KVC"
            raise RuntimeError("allocator failed to reload offloaded KVC")
        self.stats.kvc_allocator_available_after = self._allocator_available_size()

        host_slots = [slot for entry in selected for slot in entry.host_slot_list()]
        try:
            self._ensure_host_store()
            async_copy = self.config.kvc_scheduler == "async-deadline"
            elapsed_ms, start_event, ready_event = self._host_store.reload(
                host_slots,
                new_locs,
                stream=self._copy_stream,
                async_copy=async_copy,
            )
            self.stats.layerkv_copy_event_record_count += 1
            self._rewrite_req_to_token_for_entries(selected, new_locs)
            new_locs_cpu = [int(x) for x in new_locs.detach().cpu().tolist()]
            offset = 0
            for entry in selected:
                page_locs = new_locs_cpu[offset : offset + entry.token_count]
                offset += entry.token_count
                entry.device_locs = page_locs
                entry.device_loc = page_locs[0] if page_locs else None
                entry.last_access_step = self._decode_step
                if async_copy and ready_event is not None:
                    entry.state = "reloading"
                    entry.ready_start_event = start_event
                    entry.ready_event = ready_event
                    entry.ready_waited = False
                else:
                    entry.state = "resident"
                    if entry.host_slots is not None:
                        self._host_store.free(entry.host_slots)
                    elif entry.host_slot is not None:
                        self._host_store.free([int(entry.host_slot)])
                    entry.host_slots = None
                    entry.host_slot = None
        except Exception:
            self.stats.kvc_physical_failure_count += 1
            self.stats.comparable = False
            self.stats.comparability_reason = "KVC reload failed"
            raise

        reload_mb = token_count * self._bytes_per_token_all_layers / float(1024 * 1024)
        self.stats.kvc_reload_count_total += token_count
        self.stats.kvc_reload_page_count_total += len(selected)
        self.stats.kvc_reload_mb_total += reload_mb
        self.stats.kvc_reload_ms += elapsed_ms
        self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
        self.stats.layerkv_kvc_reload_started += 1
        self.stats.layerkv_tasks_built += 1
        self._refresh_kvc_residency_stats()

    def _evict_kvc_to_target(self, forward_batch: Any) -> None:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return
        effective_kvc_reclaim_mb = self._effective_kvc_reclaim_mb(forward_batch)
        if effective_kvc_reclaim_mb <= 0:
            self._refresh_kvc_residency_stats()
            return
        if not self.physical_kvc_supported:
            self.stats.comparable = False
            self.stats.comparability_reason = self.unsupported_reason
            return
        if self._bytes_per_token_all_layers <= 0:
            return

        self._ensure_host_store()
        self._prune_entries_for_active_lengths(forward_batch)
        target_tokens = self._target_offloaded_tokens(effective_kvc_reclaim_mb)
        need_tokens = max(0, target_tokens - self._offloaded_token_count())
        if need_tokens <= 0:
            self.stats.kvc_eviction_skipped_count += 1
            self._refresh_kvc_residency_stats()
            return

        selected = self._select_resident_entries_for_eviction(forward_batch, need_tokens)
        if not selected:
            self.stats.kvc_eviction_skipped_count += 1
            self._refresh_kvc_residency_stats()
            return

        token_count = sum(entry.token_count for entry in selected)
        host_slots = self._host_store.alloc(token_count)
        if host_slots is None:
            self.stats.kvc_physical_failure_count += 1
            self.stats.comparable = False
            self.stats.comparability_reason = "LayerKV host KVC store is full"
            if self.config.disallow_destructive_fallback:
                raise RuntimeError("LayerKV host KVC store is full")
            return

        old_locs = torch.tensor(
            [loc for entry in selected for loc in entry.device_loc_list()],
            dtype=torch.int64,
            device=self._kv_pool.device,
        )
        try:
            elapsed_ms = self._host_store.backup(old_locs, host_slots)
            self.stats.kvc_allocator_available_before = self._allocator_available_size()
            self._allocator.free(old_locs)
            self.stats.kvc_allocator_available_after = self._allocator_available_size()
            self.stats.kvc_allocator_free_count += 1
            offset = 0
            for entry in selected:
                page_host_slots = host_slots[offset : offset + entry.token_count]
                offset += entry.token_count
                entry.state = "offloaded"
                entry.host_slots = [int(x) for x in page_host_slots]
                entry.host_slot = int(page_host_slots[0]) if page_host_slots else None
                entry.device_loc = None
                entry.device_locs = None
                entry.ready_event = None
                entry.ready_start_event = None
                entry.ready_waited = False
                entry.last_access_step = self._decode_step
        except Exception:
            self._host_store.free(host_slots)
            self.stats.kvc_physical_failure_count += 1
            self.stats.comparable = False
            self.stats.comparability_reason = "KVC eviction failed"
            raise

        self.stats.kvc_evict_count_total += token_count
        self.stats.kvc_evict_page_count_total += len(selected)
        self.stats.kvc_backup_ms += elapsed_ms
        self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
        self.stats.kvc_physical_cycle_count += 1
        self.stats.layerkv_tasks_built += 1
        self._refresh_kvc_residency_stats()

    def _target_offloaded_tokens(self, target_reclaim_mb: float) -> int:
        target_bytes = int(target_reclaim_mb * 1024 * 1024)
        raw_tokens = max(1, target_bytes // self._bytes_per_token_all_layers)
        return max(self._page_size, self._align_tokens_up(raw_tokens))

    def _offloaded_token_count(self) -> int:
        return sum(
            entry.token_count
            for entry in self._residency.values()
            if entry.state == "offloaded"
        )

    def _resident_token_count(self) -> int:
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
        self.stats.kvc_residency_entry_count = len(self._residency)
        self.stats.kvc_offloaded_page_count = sum(
            1 for entry in self._residency.values() if entry.state == "offloaded"
        )
        self.stats.kvc_resident_page_count = sum(
            1 for entry in self._residency.values() if entry.state == "resident"
        )
        self.stats.kvc_page_size = self._page_size
        self.stats.physical_kvc_reclaim_mb = (
            offloaded * self._bytes_per_token_all_layers / float(1024 * 1024)
        )
        if self._host_store is not None:
            self.stats.kvc_host_used_tokens = self._host_store.used_count
            self.stats.kvc_host_backing_mb = self._host_store.capacity_mb

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

        active_lens = None
        if forward_batch is not None:
            active_lens = {
                req_idx: seq_len
                for req_idx, seq_len in self._batch_req_indices_and_lens(forward_batch)
            }

        host_used_slots = (
            set(self._host_store.used_slots) if self._host_store is not None else set()
        )

        for key, entry in self._residency.items():
            req_idx, pos = key
            if key != (entry.req_idx, entry.pos):
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
                host_slots = entry.host_slot_list()
                host_owned_count += len(host_slots)
                if not host_slots:
                    stale_count += 1
                    reasons.append("offloaded_missing_host_slot")
                for host_slot in host_slots:
                    if host_slot in seen_host_slots:
                        stale_count += 1
                        reasons.append("duplicate_host_slot")
                    elif self._host_store is not None and int(host_slot) not in host_used_slots:
                        stale_count += 1
                        reasons.append("host_slot_not_marked_used")
                    seen_host_slots.add(int(host_slot))
                if entry.device_loc is not None:
                    stale_count += 1
                    reasons.append("offloaded_has_device_loc")
                if len(host_slots) != entry.token_count:
                    stale_count += 1
                    reasons.append("offloaded_host_slot_count_mismatch")
            elif entry.state in ("resident", "reloading"):
                if entry.state == "reloading":
                    host_owned_count += len(entry.host_slot_list())
                if entry.state == "resident":
                    resident_count += entry.token_count
                device_locs = entry.device_loc_list()
                if not device_locs:
                    stale_count += 1
                    reasons.append("resident_missing_device_loc")
                if entry.state == "resident" and entry.host_slot is not None:
                    stale_count += 1
                    reasons.append("resident_has_host_slot")
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
                        table_locs = [
                            int(x)
                            for x in self._req_to_token_pool.req_to_token[
                                req_idx, entry.logical_positions()
                            ]
                            .detach()
                            .cpu()
                            .tolist()
                        ]
                        if table_locs != device_locs:
                            stale_count += 1
                            reasons.append("resident_req_to_token_mismatch")
                    except Exception:
                        stale_count += 1
                        reasons.append("resident_req_to_token_check_failed")
            else:
                stale_count += 1
                reasons.append("unknown_entry_state")

        if self._host_store is not None and self._host_store.used_count != host_owned_count:
            stale_count += abs(self._host_store.used_count - host_owned_count)
            reasons.append("host_used_count_mismatch")

        self.stats.kvc_residency_entry_count = len(self._residency)
        self.stats.kvc_stale_entry_count = stale_count
        self.stats.kvc_offloaded_token_count = offloaded_count
        self.stats.kvc_resident_token_count = resident_count
        self.stats.kvc_host_used_tokens = (
            self._host_store.used_count if self._host_store is not None else 0
        )

        guard_pass = stale_count == 0
        needs_kvc_reclaim = self.config.mode == "kvc-only" and self.config.target_reclaim_mb > 0
        needs_kvc_reclaim = needs_kvc_reclaim or self.stats.planned_kvc_reclaim_mb > 0
        needs_kvc_reclaim = needs_kvc_reclaim or self.stats.effective_kvc_reclaim_mb > 0
        if needs_kvc_reclaim and not self.physical_kvc_supported:
            guard_pass = False
            reasons.append(self.unsupported_reason or "physical_kvc_unsupported")

        self.stats.kvc_guard_pass = guard_pass
        self.stats.kvc_guard_reason = ";".join(sorted(set(reasons)))
        return {
            "kvc_guard_pass": self.stats.kvc_guard_pass,
            "kvc_guard_reason": self.stats.kvc_guard_reason,
            "kvc_stale_entry_count": self.stats.kvc_stale_entry_count,
            "kvc_residency_entry_count": self.stats.kvc_residency_entry_count,
            "kvc_host_used_tokens": self.stats.kvc_host_used_tokens,
            "kvc_offloaded_token_count": self.stats.kvc_offloaded_token_count,
            "kvc_resident_token_count": self.stats.kvc_resident_token_count,
        }

    def _batch_req_indices_and_lens(self, forward_batch: Any) -> List[Tuple[int, int]]:
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if seq_lens_cpu is None or req_pool_indices is None:
            return []
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
        to_drop = [key for key in self._residency if key[0] in req_indices]
        self._drop_residency_keys(to_drop)

    def _prune_entries_for_active_lengths(self, forward_batch: Any) -> None:
        to_drop = []
        for req_idx, seq_len in self._batch_req_indices_and_lens(forward_batch):
            for key, entry in self._residency.items():
                if key[0] == req_idx and entry.pos + entry.token_count > seq_len:
                    to_drop.append(key)
        self._drop_residency_keys(to_drop)

    def _drop_residency_keys(self, keys: List[Tuple[int, int]]) -> None:
        if not keys:
            return
        host_slots = []
        for key in keys:
            entry = self._residency.pop(key, None)
            if entry is not None:
                if entry.ready_event is not None and not entry.ready_event.query():
                    entry.ready_event.synchronize()
                host_slots.extend(entry.host_slot_list())
        if host_slots and self._host_store is not None:
            self._host_store.free(host_slots)
        self._refresh_kvc_residency_stats()

    def _select_required_offloaded_entries(
        self, forward_batch: Any
    ) -> List[_LayerKVResidencyEntry]:
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

    def _select_resident_entries_for_eviction(
        self, forward_batch: Any, max_tokens: int
    ) -> List[_LayerKVResidencyEntry]:
        table = self._req_to_token_pool.req_to_token
        candidates: List[_LayerKVResidencyEntry] = []
        seen_locs = set()
        page_size = max(1, int(self._page_size))
        for req_idx, seq_len in self._batch_req_indices_and_lens(forward_batch):
            prefix_len = max(0, seq_len - 1)
            if prefix_len <= 0:
                continue
            evictable_len = self._align_tokens_down(prefix_len)
            if evictable_len <= 0:
                continue
            for pos in range(0, evictable_len, page_size):
                positions = torch.arange(
                    pos, pos + page_size, dtype=torch.int64, device=table.device
                )
                locs = [
                    int(x)
                    for x in table[req_idx, positions]
                    .detach()
                    .to("cpu", non_blocking=False)
                    .tolist()
                ]
                key = (req_idx, pos)
                existing = self._residency.get(key)
                if existing is not None and existing.state in ("offloaded", "reloading"):
                    continue
                if any(loc <= 0 for loc in locs) or any(loc in seen_locs for loc in locs):
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
                candidates.append(existing)
        candidates.sort(key=lambda x: (x.pos, x.req_idx))
        max_tokens = self._align_tokens_down(max_tokens)
        take_pages = min(max_tokens // page_size, len(candidates))
        if take_pages <= 0:
            return []
        block_tokens = max(page_size, self._align_tokens_up(self.config.kvc_block_tokens))
        block_pages = max(1, block_tokens // page_size)
        if take_pages > block_pages:
            take_pages -= take_pages % block_pages
            take_pages = max(block_pages, take_pages)
        return candidates[:take_pages]

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

    def on_forward_end(self, *, mode: str, forward_batch: Any) -> None:
        if mode == "decode":
            t0 = time.perf_counter()
            self._finalize_expert_materialize_events(block=False)
            self._finalize_reloaded_entries(block=True)
            self._evict_kvc_to_target(forward_batch)
            self.validate_kvc_state(forward_batch)
            self.stats.layerkv_python_overhead_ms += (
                time.perf_counter() - t0
            ) * 1000.0
        elif mode == "extend":
            self.validate_kvc_state(forward_batch)
        if self.config.debug_stats:
            logger.info("LayerKV stats after %s: %s", mode, self.summary())

    def summary(self) -> Dict[str, Any]:
        self.validate_kvc_state()
        out = self.stats.as_dict()
        out.update(self._planner_input_summary())
        out.update(
            {
                "layerkv_enabled": self.config.enabled,
                "layerkv_mode": self.config.mode,
                "layerkv_policy": self.config.policy,
                "layerkv_target_reclaim_mb": self.config.target_reclaim_mb,
                "layerkv_kvc_scheduler": self.config.kvc_scheduler,
                "layerkv_physical_kvc_supported": self.physical_kvc_supported,
                "layerkv_physical_expert_supported": self.physical_expert_supported,
                "layerkv_expert_layer_count": len(self._expert_layers),
                "layerkv_unsupported_reason": self.unsupported_reason,
            }
        )
        return out

    def _planner_input_summary(self) -> Dict[str, Any]:
        call_by_layer: Dict[str, Dict[str, int]] = {}
        unique_by_layer: Dict[str, Dict[str, int]] = {}
        topk_by_layer: Dict[str, int] = {}
        hotness_topk_by_layer: Dict[str, Dict[str, List[List[int]]]] = {}
        num_experts_by_layer: Dict[str, int] = {}

        for layer_id, state in sorted(self._expert_layers.items()):
            layer_key = str(layer_id)
            prefill_total = int(sum(state.hotness_prefill.values()))
            decode_total = int(sum(state.hotness_decode.values()))
            call_by_layer[layer_key] = {
                "prefill": prefill_total,
                "decode": decode_total,
                "total": prefill_total + decode_total,
            }
            unique_by_layer[layer_key] = {
                "prefill": len(state.hotness_prefill),
                "decode": len(state.hotness_decode),
                "total": len(set(state.hotness_prefill) | set(state.hotness_decode)),
            }
            top_k = int(getattr(state.module, "top_k", 0) or 0)
            if top_k <= 0:
                top_k = int(getattr(state.module.moe_runner_config, "top_k", 0) or 0)
            topk_by_layer[layer_key] = top_k
            num_experts_by_layer[layer_key] = int(state.full_num_experts)
            hotness_topk_by_layer[layer_key] = {
                "prefill": self._hotness_top_items(state.hotness_prefill),
                "decode": self._hotness_top_items(state.hotness_decode),
            }

        return {
            "num_expert_layers": len(self._expert_layers),
            "num_experts_by_layer": json.dumps(num_experts_by_layer, sort_keys=True),
            "topk_by_layer": json.dumps(topk_by_layer, sort_keys=True),
            "expert_call_count_by_layer": json.dumps(call_by_layer, sort_keys=True),
            "expert_unique_count_by_layer": json.dumps(unique_by_layer, sort_keys=True),
            "expert_hotness_topk_by_layer": json.dumps(
                hotness_topk_by_layer, sort_keys=True
            ),
        }

    @staticmethod
    def _hotness_top_items(hotness: Dict[int, int], limit: int = 8) -> List[List[int]]:
        return [
            [int(expert_id), int(count)]
            for expert_id, count in sorted(
                hotness.items(), key=lambda item: (-item[1], item[0])
            )[:limit]
        ]
