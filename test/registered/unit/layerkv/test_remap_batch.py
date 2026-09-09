"""Batch publication preserves scalar remap and transfer semantics."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.runtime import LayerKVConfig


def test_remap_config():
    assert LayerKVConfig().expert_remap_update == "scalar"
    assert (
        LayerKVConfig.from_server_args(
            SimpleNamespace(layerkv_expert_remap_update="batch")
        ).expert_remap_update
        == "batch"
    )


def test_invalid_remap_argument():
    from sglang.srt.server_args import ServerArgs

    args = SimpleNamespace(
        layerkv_expert_backing_cache_mb=128,
        layerkv_expert_cpu_backing_pool_mb=128,
        layerkv_expert_backing_cache_accounting="scan",
        layerkv_expert_backing_cache_accounting_check=False,
        layerkv_expert_remap_update="invalid",
    )
    with pytest.raises(ValueError, match="remap"):
        ServerArgs._handle_layerkv(args)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("reason", ["on_demand", "prefetch"])
def test_repeated_materialization_matches_scalar(monkeypatch, dtype, cached, reason):
    from test_cpu_known_prepare import setup

    results = []
    for mode in ("scalar", "batch"):
        r, state, _ = setup(dtype=dtype)
        r.config.expert_remap_update = mode
        if cached:
            r.config.expert_backing_cache_mb = 1
            state.cpu_params.update(
                {
                    i: {"weight": torch.full((2,), i + 1, dtype=dtype, pin_memory=True)}
                    for i in (4, 5)
                }
            )
        ptr = state.remap_tensor.data_ptr()
        original = r._copy_materialized_experts_batched
        snapshots = []

        def check_published(current, materialized, **kwargs):
            expected = torch.full_like(current.remap_tensor, -1)
            for logical, slot in current.logical_to_slot.items():
                expected[logical] = slot
            assert torch.equal(current.remap_tensor, expected)
            return original(current, materialized, **kwargs)

        monkeypatch.setattr(r, "_copy_materialized_experts_batched", check_published)
        for step, ids in enumerate(([0, 1, 0], [0, 2], [4, 5], [2, 3], [0, 1]) * 3):
            r._decode_step = step
            r._materialize_experts(state, list(ids), reason=reason)
            r._wait_for_expert_logical_ids_ready(state, ids)
            mapped = state.remap_tensor[torch.tensor(ids, device="cuda")]
            actual = state.module.weight[mapped].cpu()
            assert torch.equal(
                actual, torch.tensor([[i + 1] * 2 for i in ids], dtype=dtype)
            )
            assert state.remap_tensor.data_ptr() == ptr
            snapshots.append(
                (
                    state.remap_tensor.cpu().tolist(),
                    dict(state.lru),
                    list(state.lru_heap),
                    dict(state.backing_lru),
                )
            )
        r._finalize_expert_materialize_events(block=True)
        assert r.stats.expert_guard_pass
        assert (r.stats.expert_remap_batch_count > 0) == (mode == "batch")
        results.append(
            (
                snapshots,
                r.stats.expert_materialize_count,
                r.stats.expert_cuda_batch_h2d_count,
                r.stats.expert_cuda_batch_d2h_count,
                r.stats.expert_demand_h2d_dependency_count,
            )
        )
    assert results[0] == results[1]


@cuda
@pytest.mark.parametrize("miss", [False, True])
def test_update_keeps_guard_and_untouched_bad_mapping(miss):
    from test_cpu_known_prepare import dispatch, setup

    r, state, _ = setup()
    r.config.expert_remap_update = "batch"
    state.remap_tensor[5] = -1
    ids = [0, 5] if miss else [4, 5]
    r._materialize_experts(state, ids)
    assert r.stats.expert_remap_batch_count == int(miss)
    with pytest.raises(RuntimeError, match="invalid expert remap"):
        r._prepare_expert_dispatch_for_core(
            state, dispatch([ids]), known_logical_ids=ids
        )
    assert not r.stats.expert_guard_pass and not r.stats.comparable


@cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("reason", ["on_demand", "prefetch"])
def test_async_weight_use_and_next_replacement_without_test_fences(
    dtype, cached, reason
):
    from test_cpu_known_prepare import setup

    r, state, _ = setup(dtype=dtype)
    r.config.expert_remap_update = "batch"
    if cached:
        r.config.expert_backing_cache_mb = 1
        state.cpu_params.update(
            {
                i: {"weight": torch.full((2,), i + 1, dtype=dtype, pin_memory=True)}
                for i in (4, 5)
            }
        )
    results = []
    for ids in ([0, 1], [2, 3], [4, 5]) * 10:
        r._materialize_experts(state, list(ids), reason=reason)
        r._wait_for_expert_logical_ids_ready(state, ids)
        # Delay previous weight consumers on the producer stream. Do not insert
        # CPU assertions/synchronization that could mask a copy-stream race.
        torch.cuda._sleep(100000)
        results.append(
            (state.module.weight.clone(), [state.logical_to_slot[i] for i in ids])
        )
    r._finalize_expert_materialize_events(block=True)
    actual = torch.stack([weights for weights, _ in results]).cpu()
    for index, weights in enumerate(actual):
        base = (index % 3) * 2 + 1
        assert torch.equal(
            weights[results[index][1]],
            torch.tensor([[base] * 2, [base + 1] * 2], dtype=dtype),
        )


@cuda
def test_partial_failure_keeps_scalar_invalidations():
    from test_cpu_known_prepare import setup

    maps = []
    for mode in ("scalar", "batch"):
        r, state, _ = setup()
        r.config.expert_remap_update = mode
        del state.cpu_params[1]
        with pytest.raises(RuntimeError, match="missing CPU backing"):
            r._materialize_experts(state, [0, 1])
        maps.append(state.remap_tensor.cpu().tolist())
        assert not r.stats.expert_guard_pass
        assert r.stats.expert_cuda_batch_h2d_count == 0
    assert maps[0] == maps[1] == [0, -1, -1, -1, -1, -1]


@cuda
def test_failed_publication_is_not_retried(monkeypatch):
    from test_cpu_known_prepare import setup

    r, state, _ = setup()
    r.config.expert_remap_update = "batch"
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("test publication failure")

    monkeypatch.setattr(torch.Tensor, "index_copy_", fail)
    with pytest.raises(RuntimeError, match="test publication failure"):
        r._materialize_experts(state, [0, 1])
    assert len(calls) == 1
    assert not r.stats.expert_guard_pass and not r.stats.comparable
    assert r.stats.expert_cuda_batch_h2d_count == 0


@cuda
def test_packed_copy_reduces_real_scalar_synchronizations():
    from test_cpu_known_prepare import setup

    counts = []
    for mode in ("scalar", "batch"):
        r, state, _ = setup()
        r.config.expert_remap_update = mode
        state.cpu_params.update(
            {
                i: {
                    "weight": torch.full(
                        (2,), i + 1, dtype=state.dtype, pin_memory=True
                    )
                }
                for i in (4, 5)
            }
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
            with torch.profiler.record_function("materialize_probe"):
                r._materialize_experts(state, [0, 1])
            r._finalize_expert_materialize_events(block=True)

        def in_probe(e):
            while e is not None:
                if e.name == "materialize_probe":
                    return True
                e = e.cpu_parent
            return False

        counts.append(
            sum(e.name == "cudaStreamSynchronize" and in_probe(e) for e in p.events())
        )
    assert counts == [4, 1], counts
