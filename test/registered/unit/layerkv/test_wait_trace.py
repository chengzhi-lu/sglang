"""Wait labels preserve real copies, GPU remap and scalar guards."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.runtime import LayerKVConfig
from sglang.srt.layerkv.wait_trace import install_wait_trace


def test_trace_option_reaches_config():
    assert LayerKVConfig.from_server_args(
        SimpleNamespace(layerkv_shared_expert_trace_waits=True)
    ).shared_expert_trace_waits


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_real_batch_copy_visible_and_trace_preserves_result(dtype):
    from test_cpu_known_prepare import dispatch, setup

    results = []
    for traced in (False, True):
        r, state, _ = setup(dtype=dtype)
        r.config.shared_expert_trace_waits = traced
        if traced:
            install_wait_trace(r)
        outputs = []
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            for ids in ([0, 1], [2, 3], [4, 5]):
                result = r._prepare_expert_dispatch_for_core(
                    state, dispatch([ids], dtype), known_logical_ids=ids
                )
                outputs.append(state.module.weight[result.topk_output.topk_ids].cpu())
            r._finalize_expert_materialize_events(block=True)
        names = {event.name for event in profiler.events()}
        assert any(
            "Memcpy HtoD" in name for name in names
        ), "batch H2D missing from profiler"
        if traced:
            assert {
                f"layerkv/{name}"
                for name in (
                    "prepare",
                    "materialize",
                    "h2d_submit",
                    "ready_wait",
                    "remap",
                    "guard_reduce",
                    "readback",
                )
            } <= names
        else:
            assert not any(name.startswith("layerkv/") for name in names)
        results.append(
            (
                torch.cat(outputs),
                dict(state.logical_to_slot),
                r.stats.expert_materialize_count,
                r.stats.expert_cuda_batch_h2d_count,
                r.stats.expert_cuda_batch_d2h_count,
            )
        )
    assert torch.equal(results[0][0], results[1][0])
    assert results[0][1:] == results[1][1:]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("slot", [-1, 2])
def test_traced_path_keeps_invalid_gpu_mapping_guard(slot):
    from test_cpu_known_prepare import dispatch, setup

    r, state, _ = setup()
    r.config.shared_expert_trace_waits = True
    install_wait_trace(r)
    state.remap_tensor[4] = slot
    with pytest.raises(RuntimeError, match="invalid expert remap"):
        r._prepare_expert_dispatch_for_core(
            state, dispatch([[4, 5]]), known_logical_ids=[4, 5]
        )
    assert not r.stats.expert_guard_pass and not r.stats.comparable
