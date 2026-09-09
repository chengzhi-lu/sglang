from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv import native_moe_graph as graph_module
from sglang.srt.layerkv.expert_policy import LayerKVExpertPolicyMixin
from sglang.srt.layers.moe.utils import MoeRunnerBackend


def test_actual_triton_backend_is_not_triton_kernel():
    quant = type("UnquantizedFusedMoEMethod", (), {})()
    quant.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.TRITON)
    weight = SimpleNamespace(is_cuda=True, dtype=torch.float16)
    module = SimpleNamespace(
        use_triton_kernels=MoeRunnerBackend.TRITON.is_triton_kernels(),
        moe_tp_size=1,
        moe_ep_size=1,
        quant_method=quant,
        w13_weight=weight,
        w2_weight=weight,
    )
    assert graph_module.resident_core_eligible(module, 1, 0)
    quant.runner.runner_backend = MoeRunnerBackend.TRITON_KERNELS
    assert not graph_module.resident_core_eligible(module, 1, 0)


def options(**overrides):
    return SimpleNamespace(
        **{
            "layerkv_native_moe_graph_max_batch_size": 8,
            "enable_layerkv": True,
            "layerkv_shared_expert_layer": 0,
            "disable_overlap_schedule": True,
            "disable_cuda_graph": True,
            "disable_piecewise_cuda_graph": True,
            "tp_size": 1,
            "ep_size": 1,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"layerkv_native_moe_graph_max_batch_size": -1},
        {"enable_layerkv": False},
        {"layerkv_shared_expert_layer": -1},
        {"disable_overlap_schedule": False},
        {"disable_cuda_graph": False},
        {"disable_piecewise_cuda_graph": False},
        {"tp_size": 2},
        {"ep_size": 2},
    ],
)
def test_invalid_options(changes):
    with pytest.raises(ValueError):
        graph_module.validate_native_moe_graph_options(options(**changes))


def test_valid_options():
    graph_module.validate_native_moe_graph_options(options())
    graph_module.validate_native_moe_graph_options(
        options(
            layerkv_native_moe_graph_max_batch_size=0,
            enable_layerkv=False,
            disable_cuda_graph=False,
        )
    )


@pytest.mark.parametrize(
    "changes,layer,expected",
    [
        ({}, 1, True),
        ({}, 0, False),
        ({"quant_method": None}, 1, False),
        ({"moe_tp_size": 2}, 1, False),
        ({"moe_ep_size": 2}, 1, False),
        ({"_layerkv_expert_wrapped": True}, 1, False),
        ({"quant_method": object()}, 1, False),
        ({"w2_weight": None}, 1, False),
    ],
)
def test_resident_eligibility(changes, layer, expected):
    weight = SimpleNamespace(is_cuda=True, dtype=torch.float16)
    quant = type("UnquantizedFusedMoEMethod", (), {})()
    quant.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.TRITON)
    module = SimpleNamespace(
        **{
            "use_triton_kernels": False,
            "moe_tp_size": 1,
            "moe_ep_size": 1,
            "quant_method": quant,
            "w13_weight": weight,
            "w2_weight": weight,
            **changes,
        }
    )
    assert graph_module.resident_core_eligible(module, layer, 0) is expected


def test_hotness_remains_outside_replay(monkeypatch):
    events = []

    class Replay:
        def __init__(self, core, parameters, **kwargs):
            self.core = core

        def __call__(self, dispatch):
            events.append("replay")
            return self.core(dispatch)

    monkeypatch.setattr(graph_module, "NativeMoEGraph", Replay)
    monkeypatch.setattr(
        graph_module,
        "resident_core_eligible",
        lambda module, layer, selected: layer != selected,
    )
    runtime = LayerKVExpertPolicyMixin()
    runtime.config = SimpleNamespace(
        native_moe_graph_max_batch_size=8, shared_expert_layer=0
    )
    runtime._native_moe_graphs = {}
    runtime._expert_layers = {}
    runtime._current_forward_mode = "decode"
    runtime._should_collect_expert_hotness_layer = lambda layer: True
    runtime._record_expert_hotness_for_layer = lambda **kw: events.append("hotness")

    def core(dispatch, *args, **kwargs):
        events.append("eager")
        return dispatch

    module = SimpleNamespace(
        w13_weight=SimpleNamespace(data=SimpleNamespace(shape=(256,))),
        run_moe_core=core,
    )
    dispatch = SimpleNamespace(topk_output=SimpleNamespace(topk_ids=object()))
    runtime._install_expert_hotness_probe(module, 1)
    assert module._layerkv_hotness_orig_run_moe_core is core
    assert module.run_moe_core(dispatch) is dispatch
    assert events == ["hotness", "replay", "eager"]
    for mode, offloaded, kwargs in [
        ("extend", False, {}),
        ("decode", True, {}),
        ("decode", False, {"extra": 1}),
    ]:
        events.clear()
        runtime._current_forward_mode = mode
        runtime._expert_layers = {1: object()} if offloaded else {}
        module.run_moe_core(dispatch, **kwargs)
        assert events == ["hotness", "eager"]
    selected = SimpleNamespace(w13_weight=module.w13_weight, run_moe_core=core)
    runtime._install_expert_hotness_probe(selected, 0)
    assert list(runtime._native_moe_graphs) == [1]
