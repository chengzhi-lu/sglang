"""LayerKV runtime integration for SGLang.

This package hosts the first SGLang-facing adapter for the layer-aware
KV/expert residency work.  It is intentionally flag-gated and defaults to a
no-op unless ``--enable-layerkv`` is set.
"""

from sglang.srt.layerkv.runtime import LayerKVConfig, LayerKVRuntime, LayerKVStats

__all__ = ["LayerKVConfig", "LayerKVRuntime", "LayerKVStats"]
