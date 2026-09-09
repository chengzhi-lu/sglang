"""Cached API attributes must never become cached tensor validation/ownership."""

import gc
import weakref

import pytest
import torch

from sglang.srt.layerkv.config_stats import LayerKVStats
from sglang.srt.layerkv.expert_transfer import ExpertBatchTransfer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attributes_reused_by_direction_without_retaining_tensors(monkeypatch, dtype):
    transfer = ExpertBatchTransfer(LayerKVStats())
    stream = torch.cuda.Stream()
    cuda = transfer.cuda
    real_copy = cuda.cudaMemcpyBatchAsync
    calls = []

    def record(*args):
        attr = args[4][0]
        calls.append(
            (
                attr,
                attr.srcLocHint.type,
                attr.srcLocHint.id,
                attr.dstLocHint.type,
                attr.dstLocHint.id,
                attr.srcAccessOrder,
                attr.flags,
            )
        )
        return real_copy(*args)

    monkeypatch.setattr(cuda, "cudaMemcpyBatchAsync", record)
    src = torch.arange(32, dtype=dtype).pin_memory()
    dst = torch.empty_like(src, device=stream.device)
    out = torch.empty_like(src, pin_memory=True)
    refs = [weakref.ref(t) for t in (src, dst, out)]
    for _ in range(3):
        transfer.copy([(dst, src)], stream)
        transfer.copy([(out, dst)], stream)
        transfer.collect(block=True)
        assert torch.equal(src, out)
    assert len(transfer._attributes) == 2
    for index, row in enumerate(calls):
        h2d = index % 2 == 0
        assert row[0] is calls[index % 2][0]
        host = cuda.cudaMemLocationType.cudaMemLocationTypeHost
        gpu = cuda.cudaMemLocationType.cudaMemLocationTypeDevice
        assert row[1:5] == (
            host if h2d else gpu,
            0 if h2d else stream.device.index,
            gpu if h2d else host,
            stream.device.index if h2d else 0,
        )
        assert row[5] == cuda.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
        assert row[6] == 0
    assert calls[0][0] is not calls[1][0]
    del src, dst, out
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize(
    "mutation", ["pageable", "shape", "dtype", "layout", "overlap", "direction"]
)
def test_warm_attributes_do_not_bypass_current_tensor_guards(mutation):
    transfer = ExpertBatchTransfer(LayerKVStats())
    stream = torch.cuda.Stream()
    host = torch.ones((4, 4), pin_memory=True)
    gpu = torch.empty_like(host, device=stream.device)
    transfer.copy([(gpu, host)], stream)
    transfer.collect(block=True)
    pairs = [(gpu, host)]
    message = "matching"
    if mutation == "pageable":
        host.data = torch.ones_like(host, pin_memory=False)
        message = "pinned"
    elif mutation == "shape":
        host.resize_(16)
    elif mutation == "dtype":
        host.data = torch.ones((4, 4), dtype=torch.float16, pin_memory=True)
    elif mutation == "layout":
        host.set_(host.untyped_storage(), 0, (4, 4), (1, 4))
    elif mutation == "overlap":
        pairs = [(gpu[:3], host[:3]), (gpu[1:], host[1:])]
        message = "overlap"
    elif mutation == "direction":
        pairs.append((host, gpu))
        message = "one transfer direction"
    with pytest.raises(ValueError, match=message):
        transfer.copy(pairs, stream)
    assert transfer.stats.expert_cuda_batch_copy_count == 1
    assert not transfer.pending and not transfer.failed_refs
    assert len(transfer._attributes) == 1


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_attribute_device_keys_and_warm_device_guard():
    transfer = ExpertBatchTransfer(LayerKVStats())
    host = torch.ones(16, pin_memory=True)
    for device in (0, 1):
        stream = torch.cuda.Stream(device=device)
        gpu = torch.empty_like(host, device=stream.device)
        transfer.copy([(gpu, host)], stream)
        transfer.collect(block=True)
        assert torch.equal(gpu.cpu(), host)
        attr = transfer._attributes[("h2d", stream.device)]
        assert attr.dstLocHint.id == device
        wrong = torch.cuda.Stream(device=1 - device)
        with pytest.raises(ValueError, match="one CUDA device"):
            transfer.copy([(gpu, host)], wrong)
    assert len(transfer._attributes) == 2


def test_empty_batch_does_not_create_attributes():
    transfer = ExpertBatchTransfer(LayerKVStats())
    stream = torch.cuda.Stream()
    transfer.copy([], stream)
    # A fresh zero-sized allocation has no pinned storage in PyTorch. Use an
    # empty view of a real pinned allocation to exercise the valid no-op path.
    transfer.copy(
        [(torch.empty(0, device=stream.device), torch.empty(1, pin_memory=True)[:0])],
        stream,
    )
    assert not transfer._attributes and not transfer.pending
    assert transfer.stats.expert_cuda_batch_copy_count == 0
