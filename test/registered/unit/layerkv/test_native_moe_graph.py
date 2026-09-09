"""GPU mechanism checks only; not native Triton/full-model compatibility claims."""

from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.native_moe_graph import NativeMoEGraph

TopK = namedtuple("TopK", "topk_weights topk_ids router_logits")
Dispatch = namedtuple("Dispatch", "hidden_states hidden_states_scale topk_output")
Combine = namedtuple("Combine", "hidden_states")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("inplace", [False, True])
@torch.inference_mode()
def test_exact_shape_replay_preserves_routing_mutation_and_output_lifetime(inplace):
    weights = [torch.arange(64, device="cuda", dtype=torch.float16).reshape(8, 8) / 64]

    def core(d):
        value = d.hidden_states @ weights[0]
        value += d.topk_output.topk_ids.to(value.dtype)
        value *= d.topk_output.topk_weights
        if inplace:
            d.hidden_states.copy_(value)
            value = d.hidden_states
        return Combine(value)

    def inputs(batch, offset):
        return Dispatch(
            torch.full((batch, 8), offset, device="cuda", dtype=torch.float16),
            None,
            TopK(
                torch.full((batch, 1), 0.5, device="cuda"),
                torch.full((batch, 1), offset, device="cuda", dtype=torch.int32),
                None,
            ),
        )

    replay = NativeMoEGraph(core, lambda: weights, max_batch_size=4)
    saved = []
    for batch, offset in [(2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3)]:
        d, reference = inputs(batch, offset), inputs(batch, offset)
        expected = core(reference).hidden_states.clone()
        actual = replay(d).hidden_states
        assert torch.equal(actual, expected)
        assert torch.equal(d.hidden_states, reference.hidden_states)
        assert (actual.data_ptr() == d.hidden_states.data_ptr()) == inplace
        for previous, snapshot in saved:
            assert torch.equal(previous, snapshot)
        saved.append((actual, actual.clone()))
    assert replay.captures == 2 and replay.replays == 3
    # Replaced parameter storage must not replay stale captured weights.
    weights[0] = weights[0].clone().mul_(2)
    for _ in range(3):
        actual = replay(inputs(3, 2)).hidden_states
        assert torch.equal(actual, core(inputs(3, 2)).hidden_states)
    assert replay.captures == 3
    assert replay.workspace_bytes > 0
    before = replay.captures
    replay(inputs(5, 1))
    assert replay.captures == before  # outside the bounded decode envelope


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_real_triton_b8_core_replay(monkeypatch):
    """Use the already benchmarked Qwen expert shape, not a model launch."""
    from sglang.srt import server_args
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    if torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        pytest.skip("real Qwen expert shape needs 1.5GiB weights plus workspace")
    monkeypatch.setattr(
        server_args,
        "_global_server_args",
        SimpleNamespace(
            enable_fused_moe_sum_all_reduce=False,
            enable_deterministic_inference=False,
        ),
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(17)
        w1 = torch.empty((256, 1024, 2048), device="cuda", dtype=torch.float16).normal_(
            std=0.01
        )
        w2 = torch.empty((256, 2048, 512), device="cuda", dtype=torch.float16).normal_(
            std=0.01
        )
        config = MoeRunnerConfig(
            num_experts=256,
            num_local_experts=256,
            hidden_size=2048,
            intermediate_size_per_partition=512,
            top_k=8,
            inplace=True,
        )

        def core(d):
            return StandardCombineInput(
                fused_experts(d.hidden_states, w1, w2, d.topk_output, config)
            )

        replay = NativeMoEGraph(core, lambda: (w1, w2))
        previous = []
        for offset in range(4):
            hidden = torch.randn((8, 2048), device="cuda", dtype=torch.float16) * 0.1
            ids = (
                (
                    torch.arange(64, device="cuda", dtype=torch.int32).reshape(8, 8)
                    + offset * 47
                )
                % 256
            ).contiguous()
            topk = StandardTopKOutput(
                torch.randn((8, 8), device="cuda").softmax(-1),
                ids,
                torch.zeros((8, 256), device="cuda"),
            )
            expected = core(
                StandardDispatchOutput(hidden.clone(), None, topk)
            ).hidden_states
            dispatch = StandardDispatchOutput(hidden.clone(), None, topk)
            actual = replay(dispatch).hidden_states
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.data_ptr() == dispatch.hidden_states.data_ptr()
            for value, snapshot in previous:
                torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
            previous.append((actual, actual.clone()))
        assert replay.captures == 1 and replay.replays == 2
