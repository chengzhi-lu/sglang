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
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LayerKVConfig:
    enabled: bool = False
    mode: str = "off"
    policy: str = "none"
    target_reclaim_mb: float = 0.0
    kvc_block_tokens: int = 16
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
    planned_kvc_reclaim_mb: float = 0.0
    physical_kvc_reclaim_mb: float = 0.0
    kvc_host_backing_mb: float = 0.0
    kvc_host_capacity_tokens: int = 0
    kvc_host_used_tokens: int = 0
    kvc_resident_token_count: int = 0
    kvc_offloaded_token_count: int = 0
    kvc_evict_count_total: int = 0
    kvc_reload_count_total: int = 0
    kvc_reload_mb_total: float = 0.0
    kvc_backup_ms: float = 0.0
    kvc_reload_ms: float = 0.0
    kvc_use_point_wait_ms: float = 0.0
    kvc_allocator_free_count: int = 0
    kvc_req_to_token_rewrite_count: int = 0
    kvc_physical_cycle_count: int = 0
    kvc_physical_failure_count: int = 0
    planner_apply_count: int = 0
    scheduler_invocation_count: int = 0
    layerkv_tasks_built: int = 0
    layerkv_kvc_reload_started: int = 0
    layerkv_expert_materialize_started: int = 0
    layerkv_deadline_miss_count: int = 0
    layerkv_main_stream_wait_ms: float = 0.0
    layerkv_copy_stream_busy_ms: float = 0.0
    layerkv_python_overhead_ms: float = 0.0
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
    last_access_step: int = 0


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

    def reload(self, host_slots: List[int], device_locs: torch.Tensor) -> float:
        if device_locs.numel() == 0:
            return 0.0
        host_index = torch.tensor(host_slots, dtype=torch.int64, device="cpu")
        device_module = torch.get_device_module(self.device)
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
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
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))


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
        self._host_store: Optional[_LayerKVHostKVStore] = None
        self._residency: Dict[Tuple[int, int], _LayerKVResidencyEntry] = {}
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

    def _can_support_kvc_pool(self, kv_pool: Any) -> bool:
        try:
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        except Exception:
            return False

        page_size = int(getattr(kv_pool, "page_size", 1) or 1)
        if not isinstance(kv_pool, MHATokenToKVPool):
            self.unsupported_reason = (
                f"KVC physical offload supports MHATokenToKVPool only, got "
                f"{type(kv_pool).__name__}"
            )
            return False
        if page_size != 1:
            self.unsupported_reason = (
                "KVC physical offload v1 supports page_size=1 only; "
                f"got page_size={page_size}"
            )
            return False
        if self._allocator is None or self._req_to_token_pool is None:
            self.unsupported_reason = "KVC allocator or req_to_token_pool is missing"
            return False

        try:
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
        capacity_tokens = target_tokens + max(
            target_tokens, self.config.kvc_block_tokens
        )
        self._host_store = _LayerKVHostKVStore(self._kv_pool, capacity_tokens)
        self.stats.kvc_host_capacity_tokens = self._host_store.capacity_tokens
        self.stats.kvc_host_backing_mb = self._host_store.capacity_mb

    def _discover_expert_support(self, runner: Any) -> None:
        model = getattr(runner, "model", None)
        if model is None:
            return
        # v1 intentionally avoids changing SGLang MoE runner layouts.  The
        # metadata hook is present so later fixed-slot backends can attach here.
        self.physical_expert_supported = False

    def _avg_prefix_len(self, forward_batch: Any) -> float:
        pairs = self._batch_req_indices_and_lens(forward_batch)
        if not pairs:
            return 0.0
        return sum(max(0, seq_len - 1) for _, seq_len in pairs) / float(len(pairs))

    def _policy_fractions(self, forward_batch: Any) -> Tuple[float, float, bool, str]:
        policy = self.config.policy
        if policy == "none":
            return 0.0, 0.0, True, "policy=none disables LayerKV physical reclaim"
        if policy == "expert-first":
            return 1.0, 0.0, True, ""
        if policy == "kv-first":
            return 0.0, 1.0, False, "expert reclaim is not implemented in LayerKV SGLang v1"
        if policy == "ratio-75-25":
            return 0.75, 0.25, False, "expert reclaim is not implemented in LayerKV SGLang v1"
        if policy == "ratio-50-50":
            return 0.50, 0.50, False, "expert reclaim is not implemented in LayerKV SGLang v1"
        if policy == "ratio-25-75":
            return 0.25, 0.75, False, "expert reclaim is not implemented in LayerKV SGLang v1"
        if policy == "layer-aware-joint-dp":
            avg_prefix = self._avg_prefix_len(forward_batch)
            if avg_prefix <= 1024:
                kvc_fraction = 0.75
            elif avg_prefix <= 4096:
                kvc_fraction = 0.50
            else:
                kvc_fraction = 0.25
            return (
                kvc_fraction,
                1.0 - kvc_fraction,
                False,
                "layer-aware-joint-dp uses KVC-only heuristic; expert reclaim is not implemented",
            )
        return 0.0, 0.0, False, f"unknown LayerKV policy: {policy}"

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
        self.stats.planned_kvc_reclaim_mb = max(
            self.stats.planned_kvc_reclaim_mb, effective
        )
        return effective

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
            return orig(layer_id, *args, **kwargs)

        return wrapped

    def _wrap_get_value_buffer(self, orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapped(layer_id: int, *args, **kwargs):
            self.stats.kvc_get_value_count += 1
            return orig(layer_id, *args, **kwargs)

        return wrapped

    def _wrap_get_kv_buffer(self, orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapped(layer_id: int, *args, **kwargs):
            self.stats.kvc_get_kv_count += 1
            return orig(layer_id, *args, **kwargs)

        return wrapped

    def on_forward_begin(self, *, mode: str, forward_batch: Any) -> None:
        t0 = time.perf_counter()
        if mode == "decode":
            self.stats.forward_decode_count += 1
            self._decode_step += 1
            self._reload_required_kvc(forward_batch)
        else:
            self.stats.forward_extend_count += 1
            self._drop_entries_for_reqs(forward_batch)
        self.stats.scheduler_invocation_count += 1
        self.stats.layerkv_python_overhead_ms += (time.perf_counter() - t0) * 1000.0

    def _reload_required_kvc(self, forward_batch: Any) -> None:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return
        self._effective_kvc_reclaim_mb(forward_batch)
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

        token_count = len(selected)
        new_locs = self._allocator.alloc(token_count)
        if new_locs is None:
            self.stats.kvc_physical_failure_count += 1
            self.stats.comparable = False
            self.stats.comparability_reason = "allocator failed to reload offloaded KVC"
            raise RuntimeError("allocator failed to reload offloaded KVC")

        host_slots = [int(entry.host_slot) for entry in selected]
        try:
            self._ensure_host_store()
            elapsed_ms = self._host_store.reload(host_slots, new_locs)
            self._rewrite_req_to_token(
                [
                    (entry.req_idx, entry.pos, entry.device_loc or 0)
                    for entry in selected
                ],
                new_locs,
            )
            for entry, new_loc in zip(selected, new_locs.detach().cpu().tolist()):
                entry.state = "resident"
                entry.device_loc = int(new_loc)
                entry.last_access_step = self._decode_step
                if entry.host_slot is not None:
                    self._host_store.free([int(entry.host_slot)])
                entry.host_slot = None
        except Exception:
            self.stats.kvc_physical_failure_count += 1
            self.stats.comparable = False
            self.stats.comparability_reason = "KVC reload failed"
            raise

        reload_mb = token_count * self._bytes_per_token_all_layers / float(1024 * 1024)
        self.stats.kvc_reload_count_total += token_count
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
            self._refresh_kvc_residency_stats()
            return

        selected = self._select_resident_entries_for_eviction(forward_batch, need_tokens)
        if not selected:
            self._refresh_kvc_residency_stats()
            return

        host_slots = self._host_store.alloc(len(selected))
        if host_slots is None:
            self.stats.kvc_physical_failure_count += 1
            self.stats.comparable = False
            self.stats.comparability_reason = "LayerKV host KVC store is full"
            if self.config.disallow_destructive_fallback:
                raise RuntimeError("LayerKV host KVC store is full")
            return

        old_locs = torch.tensor(
            [int(entry.device_loc) for entry in selected],
            dtype=torch.int64,
            device=self._kv_pool.device,
        )
        token_count = len(selected)
        try:
            elapsed_ms = self._host_store.backup(old_locs, host_slots)
            self._allocator.free(old_locs)
            self.stats.kvc_allocator_free_count += 1
            for entry, host_slot in zip(selected, host_slots):
                entry.state = "offloaded"
                entry.host_slot = int(host_slot)
                entry.device_loc = None
                entry.last_access_step = self._decode_step
        except Exception:
            self._host_store.free(host_slots)
            self.stats.kvc_physical_failure_count += 1
            self.stats.comparable = False
            self.stats.comparability_reason = "KVC eviction failed"
            raise

        self.stats.kvc_evict_count_total += token_count
        self.stats.kvc_backup_ms += elapsed_ms
        self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
        self.stats.kvc_physical_cycle_count += 1
        self.stats.layerkv_tasks_built += 1
        self._refresh_kvc_residency_stats()

    def _target_offloaded_tokens(self, target_reclaim_mb: float) -> int:
        target_bytes = int(target_reclaim_mb * 1024 * 1024)
        return max(1, target_bytes // self._bytes_per_token_all_layers)

    def _offloaded_token_count(self) -> int:
        return sum(1 for entry in self._residency.values() if entry.state == "offloaded")

    def _resident_token_count(self) -> int:
        return sum(1 for entry in self._residency.values() if entry.state == "resident")

    def _refresh_kvc_residency_stats(self) -> None:
        offloaded = self._offloaded_token_count()
        resident = self._resident_token_count()
        self.stats.kvc_offloaded_token_count = offloaded
        self.stats.kvc_resident_token_count = resident
        self.stats.physical_kvc_reclaim_mb = (
            offloaded * self._bytes_per_token_all_layers / float(1024 * 1024)
        )
        if self._host_store is not None:
            self.stats.kvc_host_used_tokens = self._host_store.used_count
            self.stats.kvc_host_backing_mb = self._host_store.capacity_mb

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
                if key[0] == req_idx and entry.pos >= seq_len:
                    to_drop.append(key)
        self._drop_residency_keys(to_drop)

    def _drop_residency_keys(self, keys: List[Tuple[int, int]]) -> None:
        if not keys:
            return
        host_slots = []
        for key in keys:
            entry = self._residency.pop(key, None)
            if entry is not None and entry.host_slot is not None:
                host_slots.append(int(entry.host_slot))
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
            for pos in range(required_prefix_len):
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
        for req_idx, seq_len in self._batch_req_indices_and_lens(forward_batch):
            prefix_len = max(0, seq_len - 1)
            if prefix_len <= 0:
                continue
            positions = torch.arange(prefix_len, dtype=torch.int64, device=table.device)
            locs = table[req_idx, positions].detach().to("cpu", non_blocking=False)
            for pos, loc in zip(range(prefix_len), locs.tolist()):
                key = (req_idx, pos)
                existing = self._residency.get(key)
                if existing is not None and existing.state == "offloaded":
                    continue
                loc = int(loc)
                if loc <= 0 or loc in seen_locs:
                    continue
                seen_locs.add(loc)
                if existing is None:
                    existing = _LayerKVResidencyEntry(
                        req_idx=req_idx,
                        pos=pos,
                        state="resident",
                        device_loc=loc,
                        last_access_step=self._decode_step,
                    )
                    self._residency[key] = existing
                else:
                    existing.state = "resident"
                    existing.device_loc = loc
                    existing.last_access_step = self._decode_step
                candidates.append(existing)
        candidates.sort(key=lambda x: (x.pos, x.req_idx))
        block_tokens = max(1, int(self.config.kvc_block_tokens))
        take = min(max_tokens, len(candidates))
        if take <= 0:
            return []
        if take > block_tokens:
            take -= take % block_tokens
            take = max(block_tokens, take)
        return candidates[:take]

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

    def on_forward_end(self, *, mode: str, forward_batch: Any) -> None:
        if mode == "decode":
            t0 = time.perf_counter()
            self._evict_kvc_to_target(forward_batch)
            self.stats.layerkv_python_overhead_ms += (
                time.perf_counter() - t0
            ) * 1000.0
        if self.config.debug_stats:
            logger.info("LayerKV stats after %s: %s", mode, self.summary())

    def summary(self) -> Dict[str, Any]:
        out = self.stats.as_dict()
        out.update(
            {
                "layerkv_enabled": self.config.enabled,
                "layerkv_mode": self.config.mode,
                "layerkv_policy": self.config.policy,
                "layerkv_target_reclaim_mb": self.config.target_reclaim_mb,
                "layerkv_physical_kvc_supported": self.physical_kvc_supported,
                "layerkv_physical_expert_supported": self.physical_expert_supported,
                "layerkv_unsupported_reason": self.unsupported_reason,
            }
        )
        return out
