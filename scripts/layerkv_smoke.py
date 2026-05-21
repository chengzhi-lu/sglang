#!/usr/bin/env python3
"""Import-level smoke test for the SGLang LayerKV adapter."""

from __future__ import annotations

from argparse import Namespace
from importlib import util
from pathlib import Path
import sys

import torch


def _load_runtime_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "python" / "sglang" / "srt" / "layerkv" / "runtime.py"
    spec = util.spec_from_file_location("layerkv_runtime_smoke", path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    runtime_mod = _load_runtime_module()
    LayerKVConfig = runtime_mod.LayerKVConfig
    LayerKVRuntime = runtime_mod.LayerKVRuntime
    args = Namespace(
        enable_layerkv=True,
        layerkv_mode="kvc-only",
        layerkv_policy="layer-aware-joint-dp",
        layerkv_target_reclaim_mb=64.0,
        layerkv_kvc_block_tokens=16,
        layerkv_debug_stats=True,
        layerkv_disallow_destructive_fallback=True,
    )
    cfg = LayerKVConfig.from_server_args(args)
    rt = LayerKVRuntime(cfg)
    assert rt.summary()["layerkv_enabled"] is True
    assert rt.summary()["layerkv_mode"] == "kvc-only"

    class DummyPool:
        def __init__(self):
            self.k = torch.zeros((8, 1, 2), dtype=torch.float32)
            self.v = torch.zeros((8, 1, 2), dtype=torch.float32)

        def set_kv_buffer(self, layer, loc, cache_k, cache_v):
            self.k[loc] = cache_k
            self.v[loc] = cache_v

        def get_key_buffer(self, layer_id):
            return self.k

        def get_value_buffer(self, layer_id):
            return self.v

        def get_kv_buffer(self, layer_id):
            return self.k, self.v

    class DummyRunner:
        device = "cpu"
        token_to_kv_pool = DummyPool()
        model = object()

    class DummyLayer:
        layer_id = 0

    runner = DummyRunner()
    rt.install_on_runner(runner)
    loc = torch.tensor([0, 1], dtype=torch.int64)
    runner.token_to_kv_pool.set_kv_buffer(
        DummyLayer(),
        loc,
        torch.ones((2, 1, 2), dtype=torch.float32),
        torch.ones((2, 1, 2), dtype=torch.float32),
    )
    runner.token_to_kv_pool.get_kv_buffer(0)
    summary = rt.summary()
    assert summary["kvc_set_kv_count"] == 1
    assert summary["kvc_get_kv_count"] == 1
    assert summary["kvc_tokens_written"] == 2
    print("layerkv smoke ok", rt.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
