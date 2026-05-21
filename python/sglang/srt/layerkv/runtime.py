"""Minimal LayerKV runtime adapter for SGLang v0.5.12.

The first integration keeps SGLang's native KV pool and MoE execution path in
place, while adding the policy-residency lifecycle hooks needed to make the
feature runnable and measurable.  Physical SGLang page offload is reported as
unsupported until the dedicated page-residency backend lands.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import time
from typing import Any, Callable, Dict, Optional

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
    planner_apply_count: int = 0
    scheduler_invocation_count: int = 0
    layerkv_tasks_built: int = 0
    layerkv_kvc_reload_started: int = 0
    layerkv_expert_materialize_started: int = 0
    layerkv_deadline_miss_count: int = 0
    layerkv_main_stream_wait_ms: float = 0.0
    layerkv_copy_stream_busy_ms: float = 0.0
    layerkv_python_overhead_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class LayerKVRuntime:
    """Flag-gated SGLang adapter for layer-aware residency.

    This v1 adapter deliberately does not mutate SGLang's physical KV layout.
    It installs stable hooks around the KV pool and forward pass so the next
    patch can replace the unsupported accounting path with real page residency
    without touching the public flag surface.
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
        runner.layerkv_runtime = self
        if getattr(runner, "device", None) == "cuda":
            self._copy_stream = torch.cuda.Stream()
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
        # v1 does not yet mutate SGLang page residency.  Keep this explicit so
        # evaluation cannot silently treat accounting as physical reclaim.
        return False

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
        else:
            self.stats.forward_extend_count += 1
        self.stats.scheduler_invocation_count += 1
        # Task construction is intentionally conservative in v1: no physical
        # task is emitted unless the backing implementation can execute it.
        self.stats.layerkv_tasks_built += 0
        self.stats.layerkv_python_overhead_ms += (time.perf_counter() - t0) * 1000.0

    def on_forward_end(self, *, mode: str, forward_batch: Any) -> None:
        if self.config.debug_stats:
            logger.debug("LayerKV stats after %s: %s", mode, self.stats.as_dict())

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
