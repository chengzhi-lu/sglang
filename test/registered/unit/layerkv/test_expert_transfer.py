"""Real CUDA API batch copies, ordering and backing-pool ownership."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.common_types import _LayerKVExpertLayerState
from sglang.srt.layerkv.config_stats import LayerKVConfig, LayerKVStats
from sglang.srt.layerkv.expert_transfer import ExpertBatchTransfer
from sglang.srt.layerkv.runtime import LayerKVRuntime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("default_stream", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batch_roundtrip_and_stream_order(default_stream, dtype):
    pytest.importorskip("cuda.bindings.runtime")
    stats = LayerKVStats()
    transfer = ExpertBatchTransfer(stats)
    stream = torch.cuda.default_stream() if default_stream else torch.cuda.Stream()
    sources = [torch.arange(n, dtype=dtype).pin_memory() for n in (128, 256)]
    gpu = [torch.empty_like(src, device="cuda") for src in sources]
    destinations = [torch.empty_like(src, pin_memory=True) for src in sources]
    with torch.cuda.stream(stream):
        transfer.copy(list(zip(gpu, sources)), stream)
        for tensor in gpu:
            tensor.add_(1)
        transfer.copy(list(zip(destinations, gpu)), stream)
    stream.synchronize()
    transfer.collect()
    for src, dst in zip(sources, destinations):
        assert torch.equal(dst, src + 1)
    assert stats.expert_cuda_batch_h2d_count == stats.expert_cuda_batch_d2h_count == 1
    assert stats.expert_cuda_batch_copy_count == 4
    assert not transfer.pending and not transfer.host_inflight


def test_batch_rejects_overlap_and_pageable_memory_before_submission():
    transfer = ExpertBatchTransfer(LayerKVStats())
    stream = torch.cuda.Stream()
    cpu = torch.ones(16, pin_memory=True)
    gpu = torch.empty(16, device="cuda")
    with pytest.raises(ValueError, match="overlap"):
        transfer.copy([(gpu, cpu), (gpu, cpu)], stream)
    with pytest.raises(ValueError, match="pinned"):
        transfer.copy([(gpu, torch.ones(16))], stream)
    assert not transfer.pending


def test_submission_error_is_not_retried_and_keeps_owners(monkeypatch):
    transfer = ExpertBatchTransfer(LayerKVStats())
    calls = []

    def fail(*args):
        calls.append(args)
        return (transfer.cuda.cudaError_t.cudaErrorInvalidValue,)

    monkeypatch.setattr(transfer.cuda, "cudaMemcpyBatchAsync", fail)
    pairs = [(torch.empty(16, device="cuda"), torch.ones(16, pin_memory=True))]
    with pytest.raises(RuntimeError, match="cudaMemcpyBatchAsync failed"):
        transfer.copy(pairs, torch.cuda.Stream())
    assert len(calls) == 1
    assert transfer.failed_refs is pairs
    assert transfer.stats.expert_cuda_batch_copy_count == 0
    with pytest.raises(RuntimeError, match="cannot be reused"):
        transfer.copy(pairs, torch.cuda.Stream())
    assert transfer.defer_release(pairs[0][1], lambda _tensor: None)
    assert len(calls) == 1


def test_event_error_preserves_pending_owners():
    transfer = ExpertBatchTransfer(LayerKVStats())

    def fail():
        raise RuntimeError("event inspection failed")

    owners = [torch.ones(16, pin_memory=True)]
    pending = (SimpleNamespace(query=fail), owners, set())
    transfer.pending.append(pending)
    with pytest.raises(RuntimeError, match="event inspection"):
        transfer.collect()
    assert transfer.pending[0] is pending


@pytest.mark.parametrize("layout", ["batch", "individual"])
def test_runtime_backup_and_pool_lifetime(layout):
    r = LayerKVRuntime(
        LayerKVConfig(
            expert_transfer_backend="cuda-batch", expert_batch_backing_layout=layout
        )
    )
    weights = torch.arange(64, dtype=torch.float16, device="cuda").reshape(4, 16)
    state = SimpleNamespace(
        module=SimpleNamespace(weight=weights),
        param_names=["weight"],
        device=weights.device,
        layer_id=0,
    )
    copied = r._copy_slots_to_cpu_batched(state, [(7, 2), (9, 0)])
    assert torch.equal(copied[7]["weight"], weights[2].cpu())
    assert torch.equal(copied[9]["weight"], weights[0].cpu())
    transfer = r._expert_batch_transfer
    transfer.collect(block=True)
    stream = torch.cuda.Stream()
    gpu = torch.empty_like(weights[0])
    host = copied[7]["weight"]
    other = copied[9]["weight"]
    transfer.copy([(gpu, host)], stream)
    r._release_cpu_backing_tensor(host)
    # Even if DMA has already completed, the pool cannot reuse the storage
    # until its completion event is collected (no timing-sensitive sleep).
    # Batch layout additionally waits for the sibling row view before
    # admitting the complete owner tensor.
    assert r.stats.expert_cpu_backing_pool_release_count == 0
    transfer.collect(block=True)
    assert torch.equal(gpu.cpu(), host)
    if layout == "batch":
        assert r.stats.expert_cpu_backing_batch_owner_reclaim_count == 0
        r._release_cpu_backing_tensor(other)
        assert r.stats.expert_cpu_backing_batch_owner_reclaim_count == 1
    assert r.stats.expert_cpu_backing_pool_release_count == 1
    if layout == "individual":
        reused = r._alloc_cpu_backing_tensor(tuple(host.shape), host.dtype)
        assert reused.data_ptr() == host.data_ptr()
        assert r.stats.expert_cpu_backing_pool_reuse_count == 1
    else:
        reused = r._alloc_cpu_backing_tensor((2, 16), host.dtype)
        assert reused.data_ptr() == host.untyped_storage().data_ptr()
        assert r.stats.expert_cpu_backing_pool_reuse_count == 1


def test_async_runtime_backup_event_precedes_slot_overwrite():
    r = LayerKVRuntime(LayerKVConfig(expert_transfer_backend="cuda-batch"))
    weights = torch.arange(64, dtype=torch.bfloat16, device="cuda").reshape(4, 16)
    expected = weights.cpu()
    state = SimpleNamespace(
        module=SimpleNamespace(weight=weights),
        param_names=["weight"],
        device=weights.device,
        layer_id=0,
    )
    stream = torch.cuda.Stream()
    r._expert_d2h_stream = stream
    assert r._copy_slots_to_cpu_batched_async(state, [(7, 2), (9, 0)])
    pending = r._pending_expert_d2h_events[-1]
    assert r._pending_expert_d2h_by_key[(0, 7)] is pending
    with torch.cuda.stream(stream):
        weights.fill_(-1)
    stream.synchronize()
    assert pending.ready_event.query()
    assert torch.equal(pending.copied[7]["weight"], expected[2])
    assert torch.equal(pending.copied[9]["weight"], expected[0])
    r._expert_batch_transfer.collect()
    assert r.stats.expert_cuda_batch_pending == 0
    assert r.stats.expert_eviction_d2h_async_count == 2


@pytest.mark.parametrize("default_stream", [False, True])
def test_demand_materialization_uses_stream_dependency_not_host_wait(
    monkeypatch, default_stream
):
    config = LayerKVConfig(
        expert_transfer_backend="cuda-batch",
        expert_batch_backing_layout="individual",
        expert_demand_d2h_wait="stream",
    )
    r = LayerKVRuntime(config)
    main = torch.cuda.default_stream() if default_stream else torch.cuda.Stream()
    r._expert_h2d_stream = torch.cuda.Stream()
    weights = torch.zeros((2, 4096), dtype=torch.bfloat16, device="cuda")
    new_cpu = {
        i: {"weight": torch.full((4096,), i + 1, dtype=weights.dtype, pin_memory=True)}
        for i in (2, 3)
    }
    state = _LayerKVExpertLayerState(
        layer_id=0,
        module=SimpleNamespace(weight=weights),
        orig_forward=None,
        orig_run_moe_core=None,
        full_num_experts=4,
        slot_capacity=2,
        expert_bytes=weights[0].nbytes,
        device=weights.device,
        dtype=weights.dtype,
        cpu_params=new_cpu,
        param_names=["weight"],
        logical_to_slot={0: 0, 1: 1},
        slot_to_logical={0: 0, 1: 1},
        lru={0: 0, 1: 0},
        hotness_prefill={},
        hotness_decode={},
        lru_heap=[(0, 0, 0), (0, 1, 1)],
        remap_tensor=torch.tensor([0, 1, -1, -1], device="cuda"),
    )
    r._expert_layers[0] = state
    r._expert_host_backing_bytes = 2 * state.expert_bytes
    torch.cuda.synchronize()
    waits = []
    original_wait = torch.cuda.Stream.wait_stream

    def wait(stream, producer):
        waits.append((stream.cuda_stream, producer.cuda_stream))
        original_wait(stream, producer)

    def forbidden(*_):
        raise AssertionError("demand D2H blocked the host")

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda.Stream, "synchronize", forbidden)
        patch.setattr(torch.cuda.Stream, "wait_stream", wait)
        with torch.cuda.stream(main):
            weights.fill_(7)  # A producer that the backup must observe.
            r._materialize_experts(state, [2, 3])
            r._wait_for_expert_logical_ids_ready(state, [2, 3])
    main.synchronize()
    assert (r._expert_h2d_stream.cuda_stream, main.cuda_stream) in waits
    for logical in (0, 1):
        assert torch.equal(
            state.cpu_params[logical]["weight"],
            torch.full((4096,), 7, dtype=weights.dtype),
        )
    for logical in (2, 3):
        assert torch.equal(
            weights[state.logical_to_slot[logical]].cpu(),
            torch.full((4096,), logical + 1, dtype=weights.dtype),
        )
    with torch.cuda.stream(main):
        r._materialize_experts(state, [0, 1])
        r._wait_for_expert_logical_ids_ready(state, [0, 1])
    main.synchronize()
    r._finalize_expert_materialize_events(block=True)
    assert torch.equal(weights.cpu(), torch.full((2, 4096), 7, dtype=weights.dtype))
    assert r.stats.expert_cuda_batch_pending == 0
    assert r.stats.expert_demand_d2h_stream_batch_count == 2
    assert r.stats.expert_demand_h2d_dependency_count == 2


def test_recall_style_backup_remains_host_ready_in_stream_mode(monkeypatch):
    r = LayerKVRuntime(
        LayerKVConfig(
            expert_transfer_backend="cuda-batch", expert_demand_d2h_wait="stream"
        )
    )
    weights = torch.ones((1, 64), device="cuda")
    state = SimpleNamespace(
        module=SimpleNamespace(weight=weights),
        param_names=["weight"],
        device=weights.device,
        layer_id=0,
    )
    original = torch.cuda.Stream.synchronize
    calls = []

    def sync(stream):
        calls.append(stream.cuda_stream)
        original(stream)

    monkeypatch.setattr(torch.cuda.Stream, "synchronize", sync)
    copied = r._copy_slots_to_cpu_batched(state, [(3, 0)])
    assert len(calls) == 1
    assert torch.equal(copied[3]["weight"], torch.ones(64))
    assert r.stats.expert_demand_d2h_stream_batch_count == 0
