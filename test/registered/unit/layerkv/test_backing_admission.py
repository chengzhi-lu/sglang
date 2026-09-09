"""Delay full validation, never pool admission, until copy owners retire."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.runtime import LayerKVConfig, LayerKVRuntime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def runtime(mode="admission"):
    config = LayerKVConfig.from_server_args(
        SimpleNamespace(
            layerkv_expert_transfer_backend="cuda-batch",
            layerkv_expert_backing_release_validation=mode,
        )
    )
    r = LayerKVRuntime(config)
    return r, r._get_expert_batch_transfer()


@pytest.mark.parametrize("mode,expected", [("eager", 2), ("admission", 1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_deferred_backing_is_validated_once_at_admission(
    monkeypatch, mode, expected, dtype
):
    r, transfer = runtime(mode)
    host = torch.arange(64, dtype=dtype).pin_memory()
    gpu = torch.empty_like(host, device="cuda")
    calls = []
    owns = r._tensor_owns_storage

    def validate(tensor):
        calls.append(tensor)
        return owns(tensor)

    monkeypatch.setattr(r, "_tensor_owns_storage", validate)
    transfer.copy([(gpu, host)], torch.cuda.Stream())
    r._release_cpu_backing_tensor(host)
    assert not r._expert_cpu_backing_pool
    assert len(transfer.deferred) == 1
    transfer.collect(block=True)
    assert torch.equal(gpu.cpu(), host)
    assert len(calls) == expected
    assert r.stats.expert_cpu_backing_pool_release_count == 1
    reused = r._alloc_cpu_backing_tensor(tuple(host.shape), host.dtype)
    assert reused.data_ptr() == host.data_ptr()
    assert not transfer.pending and not transfer.deferred and not transfer.host_inflight


@pytest.mark.parametrize("mutation", ["shape", "offset", "pageable", "limit"])
def test_admission_rechecks_current_storage_and_capacity(mutation):
    r, transfer = runtime()
    host = torch.arange(64, dtype=torch.float16).pin_memory()
    expected = host.clone()
    gpu = torch.empty_like(host, device="cuda")
    stream = torch.cuda.Stream()
    transfer.copy([(gpu, host)], stream)
    r._release_cpu_backing_tensor(host)
    assert not r._expert_cpu_backing_pool
    # Mutate metadata only after DMA completion, but before owner retirement.
    stream.synchronize()
    assert torch.equal(gpu.cpu(), expected)
    if mutation == "shape":
        host.as_strided_((32,), (1,))
    elif mutation == "offset":
        host.as_strided_((32,), (1,), 1)
    elif mutation == "pageable":
        host.set_(torch.empty(64, dtype=host.dtype))
    else:
        r._expert_cpu_backing_pool_limit_bytes = 0
    transfer.collect(block=True)
    assert not r._expert_cpu_backing_pool
    assert r.stats.expert_cpu_backing_pool_drop_count == 1
    assert r.stats.expert_cpu_backing_pool_release_count == 0


@pytest.mark.parametrize("detached", [False, True])
def test_views_never_enter_pool(detached):
    r, transfer = runtime()
    owner = torch.arange(128, dtype=torch.float16).pin_memory()
    view = owner[:64]
    if detached:
        view = view.detach()
    gpu = torch.empty_like(view, device="cuda")
    transfer.copy([(gpu, view)], torch.cuda.Stream())
    r._release_cpu_backing_tensor(view)
    if not detached:
        assert not transfer.deferred  # Existing batch views stay cheap to drop.
    assert not r._expert_cpu_backing_pool
    transfer.collect(block=True)
    assert torch.equal(gpu.cpu(), view)
    assert r.stats.expert_cpu_backing_pool_drop_count == 1
    assert not r._expert_cpu_backing_pool


def test_release_waits_for_all_transfers_of_same_storage():
    r, transfer = runtime()
    host = torch.arange(64, dtype=torch.float16).pin_memory()
    gpu1 = torch.empty_like(host, device="cuda")
    gpu2 = torch.empty_like(host, device="cuda")
    stream = torch.cuda.Stream()
    transfer.copy([(gpu1, host)], stream)
    event, refs, keys = transfer.pending[0]
    # Conservatively postpone inspection of the first real transfer.
    proxy = SimpleNamespace(query=lambda: False, synchronize=event.synchronize)
    transfer.pending[0] = (proxy, refs, keys)
    transfer.copy([(gpu2, host)], stream)
    stream.synchronize()
    r._release_cpu_backing_tensor(host)
    transfer.collect()
    assert not r._expert_cpu_backing_pool and len(transfer.pending) == 1
    transfer.collect(block=True)
    assert torch.equal(gpu1.cpu(), host) and torch.equal(gpu2.cpu(), host)
    assert r.stats.expert_cpu_backing_pool_release_count == 1


def test_failed_submission_never_admits_owner(monkeypatch):
    r, transfer = runtime()
    host = torch.ones(64, pin_memory=True)
    gpu = torch.empty_like(host, device="cuda")
    monkeypatch.setattr(
        transfer.cuda,
        "cudaMemcpyBatchAsync",
        lambda *args: (transfer.cuda.cudaError_t.cudaErrorInvalidValue,),
    )
    with pytest.raises(RuntimeError, match="cudaMemcpyBatchAsync failed"):
        transfer.copy([(gpu, host)], torch.cuda.Stream())
    r._release_cpu_backing_tensor(host)
    transfer.collect(block=True)
    assert not r._expert_cpu_backing_pool
    assert transfer.failed_refs and transfer.deferred


def test_event_inspection_failure_never_admits_owner():
    r, transfer = runtime()
    host = torch.ones(64, pin_memory=True)
    gpu = torch.empty_like(host, device="cuda")
    transfer.copy([(gpu, host)], torch.cuda.Stream())
    event, refs, keys = transfer.pending[0]

    def fail():
        raise RuntimeError("injected event error")

    transfer.pending[0] = (SimpleNamespace(query=fail), refs, keys)
    r._release_cpu_backing_tensor(host)
    with pytest.raises(RuntimeError, match="injected event"):
        transfer.collect()
    assert not r._expert_cpu_backing_pool and transfer.deferred
    transfer.pending[0] = (event, refs, keys)
    transfer.collect(block=True)
    assert r.stats.expert_cpu_backing_pool_release_count == 1


@pytest.mark.parametrize("pinned", [False, True])
def test_immediate_release_still_validates(monkeypatch, pinned):
    r, transfer = runtime()
    host = torch.ones(64, pin_memory=pinned)
    owns = r._tensor_owns_storage
    calls = []

    def validate(tensor):
        calls.append(tensor)
        return owns(tensor)

    monkeypatch.setattr(r, "_tensor_owns_storage", validate)
    r._release_cpu_backing_tensor(host)
    assert len(calls) == 1 and not transfer.deferred
    assert r.stats.expert_cpu_backing_pool_release_count == int(pinned)
    assert r.stats.expert_cpu_backing_pool_drop_count == int(not pinned)
