#!/usr/bin/env python3
"""Import-level smoke test for the SGLang LayerKV adapter."""

from __future__ import annotations

from argparse import Namespace
from collections import namedtuple
from importlib import util
from pathlib import Path
import sys
from types import SimpleNamespace

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
        layerkv_kvc_scheduler="async-deadline",
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
    unsupported_summary = rt.summary()
    assert unsupported_summary["layerkv_physical_kvc_supported"] is False
    assert unsupported_summary["comparable"] is False
    assert "DummyPool" in unsupported_summary["layerkv_unsupported_reason"]
    assert unsupported_summary["kvc_guard_pass"] is False
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

    class UnquantizedFusedMoEMethod:
        pass

    TopKOut = namedtuple("TopKOut", ["topk_weights", "topk_ids", "router_logits"])
    DispatchOut = namedtuple(
        "DispatchOut", ["hidden_states", "hidden_states_scale", "topk_output"]
    )

    class FakeFusedMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_id = 0
            self.top_k = 2
            self.moe_ep_size = 1
            self.num_experts = 4
            self.num_local_experts = 4
            self.quant_method = UnquantizedFusedMoEMethod()
            self.moe_runner_config = SimpleNamespace(
                top_k=2,
                num_experts=4,
                num_local_experts=4,
            )
            self.dispatcher = SimpleNamespace(
                num_experts=4,
                num_local_experts=4,
                num_local_routed_experts=4,
            )
            self.w13_weight = torch.nn.Parameter(
                torch.arange(16, dtype=torch.float32).view(4, 2, 2),
                requires_grad=False,
            )
            self.w2_weight = torch.nn.Parameter(
                torch.arange(16, 32, dtype=torch.float32).view(4, 2, 2),
                requires_grad=False,
            )
            self.last_topk_ids = None

        def forward(self, hidden_states, topk_output):
            dispatch_output = DispatchOut(hidden_states, None, topk_output)
            return self.run_moe_core(dispatch_output)

        def run_moe_core(self, dispatch_output):
            topk_output = dispatch_output.topk_output
            self.last_topk_ids = topk_output.topk_ids.detach().clone()
            return dispatch_output.hidden_states

    class ExpertRunner:
        device = "cpu"
        token_to_kv_pool = None
        token_to_kv_pool_allocator = None
        req_to_token_pool = None

        def __init__(self):
            self.model = torch.nn.Module()
            self.model.moe = FakeFusedMoE()

    expert_args = Namespace(
        enable_layerkv=True,
        layerkv_mode="kvc-expert",
        layerkv_policy="kv-first",
        layerkv_target_reclaim_mb=0.00008,
        layerkv_kvc_block_tokens=16,
        layerkv_kvc_scheduler="async-deadline",
        layerkv_debug_stats=True,
        layerkv_disallow_destructive_fallback=True,
    )
    expert_rt = LayerKVRuntime(LayerKVConfig.from_server_args(expert_args))
    expert_runner = ExpertRunner()
    expert_rt.install_on_runner(expert_runner)
    expert_rt.on_forward_begin(mode="decode", forward_batch=SimpleNamespace())
    expert_summary = expert_rt.summary()
    assert expert_summary["planned_kvc_reclaim_mb"] == 0.0
    assert expert_summary["planned_expert_reclaim_mb"] > 0.0
    assert expert_summary["comparable"] is True
    assert expert_summary["kvc_guard_pass"] is True
    assert expert_runner.model.moe.w13_weight.shape[0] == 4
    topk = TopKOut(
        topk_weights=torch.ones((1, 2), dtype=torch.float32),
        topk_ids=torch.tensor([[2, 3]], dtype=torch.int32),
        router_logits=torch.empty((1, 4), dtype=torch.float32),
    )
    expert_runner.model.moe(torch.zeros((1, 2), dtype=torch.float32), topk)
    expert_rt.on_forward_begin(mode="decode", forward_batch=SimpleNamespace())
    assert expert_runner.model.moe.w13_weight.shape[0] < 4
    state = expert_rt._expert_layers.get(expert_runner.model.moe.layer_id)
    offloaded = [
        expert_id
        for expert_id in range(expert_runner.model.moe.num_experts)
        if state is not None and expert_id not in state.logical_to_slot
    ]
    expert_id = int(offloaded[0]) if offloaded else 0
    topk = TopKOut(
        topk_weights=torch.ones((1, 2), dtype=torch.float32),
        topk_ids=torch.tensor([[expert_id, expert_id]], dtype=torch.int32),
        router_logits=torch.empty((1, 4), dtype=torch.float32),
    )
    expert_runner.model.moe(torch.zeros((1, 2), dtype=torch.float32), topk)
    assert int(expert_rt.summary()["expert_materialize_count"]) >= 1
    assert int(expert_rt.summary()["expert_core_hook_count"]) >= 1
    assert int(expert_rt.summary()["expert_topk_rewrite_count"]) >= 1
    assert (
        expert_runner.model.moe.last_topk_ids.max().item()
        < expert_runner.model.moe.w13_weight.shape[0]
    )
    dispatch_topk = TopKOut(
        topk_weights=torch.ones((1, 2), dtype=torch.float32),
        topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
        router_logits=torch.empty((1, 4), dtype=torch.float32),
    )
    expert_runner.model.moe.run_moe_core(
        DispatchOut(torch.zeros((1, 2), dtype=torch.float32), None, dispatch_topk)
    )
    assert (
        expert_runner.model.moe.last_topk_ids.max().item()
        < expert_runner.model.moe.w13_weight.shape[0]
    )
    assert expert_rt.summary()["layerkv_physical_expert_supported"] is True
    print("layerkv smoke ok", rt.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
