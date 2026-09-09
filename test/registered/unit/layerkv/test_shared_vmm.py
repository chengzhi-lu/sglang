"""Real physical-handle transfer, expert GEMM, rollback and KV recovery."""

import gc
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.shared_vmm import SharedVMM, _check
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="stage-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA VMM"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_same_physical_pages_run_expert_then_restore_kv(dtype):
    arena = SharedVMM("cuda")
    rows = arena.page_bytes // (1024 * dtype.itemsize)
    donor = arena.allocate((2, rows, 1024), dtype, kind="kv")
    recipient = arena.allocate(
        (3, rows, 1024), dtype, kind="expert", mapped_bytes=arena.page_bytes
    )
    kv = donor.tensor().fill_(3)
    cpu_kv = kv.cpu()
    old_ptr = kv.data_ptr()
    recipient.tensor((1, rows, 1024)).fill_(2)
    weights = torch.full((rows, 1024), 0.125, dtype=dtype)
    original_handles = [int(donor.pages[p]) for p in range(2)]
    before = arena.summary()
    for _ in range(2):
        arena.lend((donor, p, recipient, p + 1) for p in range(2))
        assert not donor.pages
        with pytest.raises(ValueError, match="unmapped"):
            donor.tensor()
        experts = recipient.tensor()
        for p in range(2):
            retained = _check(
                arena.driver.cuMemRetainAllocationHandle(experts[p + 1].data_ptr())
            )
            assert int(retained) == original_handles[p]
            _check(arena.driver.cuMemRelease(retained))
            experts[p + 1].copy_(weights)
        x = torch.ones((2, 1024), dtype=dtype, device="cuda")
        assert torch.equal(
            torch.nn.functional.linear(x, experts[1]),
            torch.full((2, rows), 128, device="cuda", dtype=dtype),
        )
        assert (
            arena.summary()["physical_create_count"] == before["physical_create_count"]
        )
        assert arena.summary()["physical_bytes"] == before["physical_bytes"]
        assert arena.summary()["ownership_guard_pass"]
        # Expert weights are immutable and backed on CPU; return the pages.
        del experts
        arena.recall()
        assert kv.data_ptr() == old_ptr
        kv.copy_(cpu_kv)
        assert torch.equal(kv.cpu(), cpu_kv)
        assert arena.summary()["ownership_guard_pass"]
    assert arena.summary()["kv_to_expert_pages"] == 4
    assert arena.summary()["expert_to_kv_pages"] == 4
    assert arena.summary()["lend_remap_ms"] > 0
    assert arena.summary()["recall_remap_ms"] > 0
    # Tensor aliases keep allocations alive, even after explicit owners go away.
    del donor, recipient
    gc.collect()
    assert kv.sum().item() > 0
    del kv
    gc.collect()
    assert arena.summary()["physical_bytes"] == 0


def test_transfer_validation_and_rollback(monkeypatch):
    arena = SharedVMM("cuda")
    page = arena.page_bytes
    donor = arena.allocate((page * 2,), torch.uint8, kind="kv")
    recipient = arena.allocate(
        (page * 3,), torch.uint8, kind="expert", mapped_bytes=page
    )
    with pytest.raises(ValueError, match="unmapped"):
        recipient.tensor()
    donor.tensor().fill_(7)
    transfers = [(donor, p, recipient, p + 1) for p in range(2)]
    with pytest.raises(ValueError, match="duplicate"):
        arena.lend([transfers[0], transfers[0]])
    original = recipient.map

    def fail_second(page, handle):
        if page == 2:
            raise RuntimeError("injected map failure")
        original(page, handle)

    monkeypatch.setattr(recipient, "map", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        arena.lend(transfers)
    assert set(donor.pages) == {0, 1}
    assert set(recipient.pages) == {0}
    assert not arena.loans
    assert arena.summary()["ownership_guard_pass"]
    assert torch.all(donor.tensor() == 7)


def test_owner_recall_keeps_other_layer_loan():
    arena = SharedVMM("cuda")
    page = arena.page_bytes
    donor = arena.allocate((3 * page,), torch.uint8, kind="kv")
    first = arena.allocate(
        (2 * page,), torch.uint8, kind="expert", mapped_bytes=page
    )
    second = arena.allocate(
        (2 * page,), torch.uint8, kind="expert", mapped_bytes=page
    )
    owner_a, owner_b = object(), object()

    arena.lend([(donor, 0, first, 1)], owner=owner_a)
    arena.lend([(donor, 1, second, 1)], owner=owner_b)
    assert arena.loan_count(owner_a) == arena.loan_count(owner_b) == 1

    arena.recall(owner=owner_a)
    assert arena.loan_count(owner_a) == 0
    assert arena.loan_count(owner_b) == 1
    assert 0 in donor.pages and 1 not in donor.pages
    assert 1 in second.pages

    arena.recall(owner=owner_b)
    assert not arena.loans
    assert arena.summary()["ownership_guard_pass"]


def test_sparse_kv_overflow_moves_expert_page_and_returns_it():
    arena = SharedVMM("cuda")
    tokens_per_page = arena.page_bytes // 1024
    arena.initial_kv_tokens = tokens_per_page
    kv = arena.kv_zeros(
        (tokens_per_page * 2 + 1, 1, 512),
        dtype=torch.float16,
        device="cuda",
    )
    expert = arena.allocate(
        (1, arena.page_bytes // torch.tensor([], dtype=torch.float16).element_size()),
        torch.float16,
        kind="expert",
        mapped_bytes=arena.page_bytes,
    )
    expert.tensor().fill_(3)
    segment, unused = arena.activate_kv_overflow(
        [(expert, 0)],
        base_tokens=tokens_per_page,
        requested_tokens=tokens_per_page,
    )
    assert segment["tokens"] == tokens_per_page
    assert len(segment["transfers"]) == 1
    assert not unused
    kv[tokens_per_page + 1 : tokens_per_page * 2 + 1].fill_(5)
    assert torch.all(kv[tokens_per_page + 1 : tokens_per_page * 2 + 1] == 5)
    arena.deactivate_kv_overflow(segment)
    # The caller must restore the immutable expert backing after the KV tail
    # used the physical page.
    expert.tensor().fill_(3)
    assert torch.all(expert.tensor() == 3)
    assert arena.summary()["ownership_guard_pass"]


@pytest.mark.parametrize("eligible", [True, False])
def test_controller_loans_only_backed_free_pages_and_blocks_reuse(eligible):
    from sglang.srt.layerkv.common_types import _LayerKVResidencyEntry
    from sglang.srt.layerkv.runtime import LayerKVConfig, LayerKVRuntime
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    arena = SharedVMM("cuda")
    tokens_per_page = arena.page_bytes // 1024
    token_count = 3 * tokens_per_page
    buffers = [
        arena.kv_zeros((token_count, 1, 512), dtype=torch.bfloat16, device="cuda")
        for _ in range(2)
    ]
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            kvc_backend="per-layer-arena",
            shared_expert_layer=0,
            shared_expert_initial_slots=1,
            shared_expert_extra_slots=1,
        )
    )
    runtime._kv_pool = SimpleNamespace(
        _get_key_buffer=lambda _: buffers[0], _get_value_buffer=lambda _: buffers[1]
    )
    runtime._refresh_expert_stats = lambda: None
    runtime._refresh_per_layer_allocator_stats = lambda: None
    controller = SharedExpertController(runtime, arena)
    params = {}
    for name, pages in [("w13_weight", 2), ("w2_weight", 1)]:
        shape = (3, pages * arena.page_bytes // 2048, 1024)
        old = torch.empty(shape, dtype=torch.bfloat16, device="meta")
        params[name] = torch.nn.Parameter(
            controller.allocate_expert(old, 1).fill_(2), requires_grad=False
        )
    module = SimpleNamespace(
        **params,
        num_experts=1,
        moe_runner_config=SimpleNamespace(num_experts=1),
        dispatcher=SimpleNamespace(num_experts=1)
    )
    controller.state = SimpleNamespace(
        module=module,
        full_num_experts=3,
        slot_capacity=1,
        param_names=list(params),
        free_slots=[],
        logical_to_slot={0: 0},
        slot_to_logical={0: 0},
        lru={0: 0},
        lru_heap=[],
        cpu_params={},
        device=torch.device("cuda"),
    )
    locs = list(range(tokens_per_page, token_count))
    runtime._per_layer_offloaded_runs_by_req_layer[(0, 3)] = [
        _LayerKVResidencyEntry(
            req_idx=0,
            pos=0,
            state="offloaded",
            layer_id=3,
            evicted_device_locs=locs,
            host_slots=list(range(len(locs))),
            page_size=len(locs),
        )
    ]
    # A backed-up page which has already been reused must never be lent.
    runtime._push_per_layer_overwrite_bits(
        3, runtime._locs_to_bitset(locs if eligible else locs[:-1])
    )
    before = arena.create_count
    controller.after_decode()
    if not eligible:
        assert controller.state.slot_capacity == 1  # Only two of three pages eligible.
        assert not arena.loans
        controller.after_decode()
        assert controller.donor_scan_count == 1
        assert controller.donor_cache_hit_count == 1
        # Publishing the missing token makes a complete third donor page.
        runtime._push_per_layer_overwrite_bits(3, runtime._locs_to_bitset(locs[-1:]))
        controller.after_decode()
        assert controller.donor_scan_count == 2
    assert controller.state.slot_capacity == 2
    assert len(arena.loans) == 3
    assert arena.create_count == before
    assert runtime._reuse_evicted_per_layer_locs(3, [tokens_per_page]) is None
    controller.record_use(
        controller.state,
        SimpleNamespace(
            topk_output=SimpleNamespace(topk_ids=torch.tensor([1], device="cuda"))
        ),
    )
    assert controller.borrowed_slot_use_count == 1
    controller.recall()
    assert not arena.loans and not controller.blocked
    assert controller.state.slot_capacity == 1
    assert runtime._per_layer_pending_overwrite_bits_union(
        3
    ) == runtime._locs_to_bitset(locs)
    assert arena.summary()["ownership_guard_pass"]
    # Returned pages remain discoverable; never keep a miss across recall.
    controller.after_decode()
    assert controller.state.slot_capacity == 2 and len(arena.loans) == 3
    assert arena.create_count == before
    controller.recall()
    assert not arena.loans and arena.summary()["ownership_guard_pass"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
