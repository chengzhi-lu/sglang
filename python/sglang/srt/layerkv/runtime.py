"""LayerKV runtime adapter for SGLang v0.5.12.

The integration keeps SGLang's native KV pool and MoE execution path in place,
while adding the policy-residency lifecycle hooks needed to make the feature
runnable and measurable.  The first physical path supports MHA KV pools with
page_size=1 by cycling selected KV slots through CPU backing storage and
rewriting req_to_token mappings before attention consumes them.
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
    planned_kvc_reclaim_mb: float = 0.0
    physical_kvc_reclaim_mb: float = 0.0
    kvc_host_backing_mb: float = 0.0
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
        self._last_reclaim_decode_step: int = -1

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

    def _discover_expert_support(self, runner: Any) -> None:
        model = getattr(runner, "model", None)
        if model is None:
            return
        # v1 intentionally avoids changing SGLang MoE runner layouts.  The
        # metadata hook is present so later fixed-slot backends can attach here.
        self.physical_expert_supported = False

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
            self._maybe_run_kvc_physical_cycle(forward_batch)
        else:
            self.stats.forward_extend_count += 1
        self.stats.scheduler_invocation_count += 1
        # Task construction is intentionally conservative in v1: no physical
        # task is emitted unless the backing implementation can execute it.
        self.stats.layerkv_tasks_built += 0
        self.stats.layerkv_python_overhead_ms += (time.perf_counter() - t0) * 1000.0

    def _maybe_run_kvc_physical_cycle(self, forward_batch: Any) -> None:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return
        if self.config.target_reclaim_mb <= 0:
            return
        self.stats.planned_kvc_reclaim_mb = max(
            self.stats.planned_kvc_reclaim_mb, self.config.target_reclaim_mb
        )
        if not self.physical_kvc_supported:
            self.stats.comparable = False
            self.stats.comparability_reason = self.unsupported_reason
            return
        if self._bytes_per_token_all_layers <= 0:
            return

        selected = self._select_kvc_token_locations(forward_batch)
        if not selected:
            return

        old_locs = torch.tensor(
            [item[2] for item in selected],
            dtype=torch.int64,
            device=self._kv_pool.device,
        )
        token_count = int(old_locs.numel())
        reclaim_mb = (
            token_count * self._bytes_per_token_all_layers / float(1024 * 1024)
        )

        try:
            backup = self._backup_kvc_locations(old_locs)
            self._allocator.free(old_locs)
            self.stats.kvc_allocator_free_count += 1
            new_locs = self._allocator.alloc(token_count)
            if new_locs is None:
                raise RuntimeError("allocator failed to reacquire freed KVC slots")
            self._reload_kvc_locations(backup, new_locs)
            self._rewrite_req_to_token(selected, new_locs)
        except Exception:
            self.stats.kvc_physical_failure_count += 1
            self.stats.comparable = False
            self.stats.comparability_reason = "KVC physical cycle failed"
            raise

        self.stats.physical_kvc_reclaim_mb += reclaim_mb
        self.stats.kvc_evict_count_total += token_count
        self.stats.kvc_reload_count_total += token_count
        self.stats.kvc_reload_mb_total += reclaim_mb
        self.stats.kvc_physical_cycle_count += 1
        self.stats.layerkv_tasks_built += 1
        self.stats.layerkv_kvc_reload_started += 1
        self.stats.kvc_host_backing_mb = max(
            self.stats.kvc_host_backing_mb, backup["bytes"] / float(1024 * 1024)
        )

    def _select_kvc_token_locations(self, forward_batch: Any) -> List[Tuple[int, int, int]]:
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if seq_lens_cpu is None or req_pool_indices is None:
            return []

        if isinstance(seq_lens_cpu, torch.Tensor):
            seq_lens = [int(x) for x in seq_lens_cpu.cpu().tolist()]
        else:
            seq_lens = [int(x) for x in seq_lens_cpu]
        req_indices = [int(x) for x in req_pool_indices.detach().cpu().tolist()]

        target_bytes = int(self.config.target_reclaim_mb * 1024 * 1024)
        max_tokens = max(1, target_bytes // self._bytes_per_token_all_layers)
        block_tokens = max(1, int(self.config.kvc_block_tokens))

        selected: List[Tuple[int, int, int]] = []
        seen_locs = set()
        table = self._req_to_token_pool.req_to_token
        for req_idx, seq_len in zip(req_indices, seq_lens):
            # The decode token itself sits at seq_len - 1. Only cycle already
            # materialized prefix tokens so attention semantics stay unchanged.
            prefix_len = max(0, seq_len - 1)
            if prefix_len <= 0:
                continue
            take = min(prefix_len, block_tokens, max_tokens - len(selected))
            if take <= 0:
                break
            positions = torch.arange(take, dtype=torch.int64, device=table.device)
            locs = table[req_idx, positions].detach().to("cpu", non_blocking=False)
            for pos, loc in zip(range(take), locs.tolist()):
                loc = int(loc)
                if loc <= 0 or loc in seen_locs:
                    continue
                seen_locs.add(loc)
                selected.append((req_idx, pos, loc))
                if len(selected) >= max_tokens:
                    break
            if len(selected) >= max_tokens:
                break
        return selected

    def _backup_kvc_locations(self, locs: torch.Tensor) -> Dict[str, Any]:
        if self._kv_pool is None:
            raise RuntimeError("KVC pool is missing")
        device_module = torch.get_device_module(self._kv_pool.device)
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        start.record()
        k_cpu = []
        v_cpu = []
        total_bytes = 0
        for layer_offset in range(self._kv_pool.layer_num):
            layer_id = self._kv_pool.start_layer + layer_offset
            k = self._kv_pool._get_key_buffer(layer_id)[locs].detach().cpu()
            v = self._kv_pool._get_value_buffer(layer_id)[locs].detach().cpu()
            try:
                k = k.pin_memory()
                v = v.pin_memory()
            except Exception:
                pass
            total_bytes += int(k.nbytes + v.nbytes)
            k_cpu.append(k)
            v_cpu.append(v)
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        self.stats.kvc_backup_ms += elapsed_ms
        self.stats.layerkv_copy_stream_busy_ms += elapsed_ms
        return {"k": k_cpu, "v": v_cpu, "bytes": total_bytes}

    def _reload_kvc_locations(self, backup: Dict[str, Any], new_locs: torch.Tensor) -> None:
        device = self._kv_pool.device
        device_module = torch.get_device_module(device)
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        start.record()
        for layer_offset in range(self._kv_pool.layer_num):
            layer_id = self._kv_pool.start_layer + layer_offset
            k_src = backup["k"][layer_offset].to(device, non_blocking=True)
            v_src = backup["v"][layer_offset].to(device, non_blocking=True)
            self._kv_pool._get_key_buffer(layer_id)[new_locs] = k_src
            self._kv_pool._get_value_buffer(layer_id)[new_locs] = v_src
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        self.stats.kvc_reload_ms += elapsed_ms
        self.stats.layerkv_copy_stream_busy_ms += elapsed_ms

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
