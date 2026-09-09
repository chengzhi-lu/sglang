"""Hybrid full-attention storage, metadata routing, and real CUDA KV recovery."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.common_types import _LayerKVResidencyEntry
from sglang.srt.layerkv.host_store import _LayerKVHostKVStore
from sglang.srt.layerkv.kv_pool_view import HybridKVCStorageView
from sglang.srt.layerkv.runtime import LayerKVConfig, LayerKVRuntime
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MambaPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    ReqToTokenPool,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="stage-b", runner_config="1-gpu-small")


def make_pool(device="cpu", dtype=torch.bfloat16, layer_ids=(3, 11)):
    # Use the real raw-buffer accessors without constructing unrelated kernels.
    full = object.__new__(MHATokenToKVPool)
    full.size, full.page_size, full.start_layer = 16, 1, 0
    full.layer_num = len(layer_ids)
    full.device = device
    full.dtype = full.store_dtype = dtype
    full.k_buffer = [
        (torch.arange(17 * 8, device=device).reshape(17, 2, 4) + i * 200).to(dtype)
        for i in range(full.layer_num)
    ]
    full.v_buffer = [x.neg().clone() for x in full.k_buffer]
    hybrid = object.__new__(HybridLinearKVPool)
    hybrid.full_kv_pool = full
    hybrid.full_attention_layer_id_mapping = dict(zip(layer_ids, range(full.layer_num)))
    hybrid.use_mla = False
    hybrid.page_size = 1
    hybrid.start_layer = 0
    hybrid.layer_transfer_counter = None
    hybrid.mamba_pool = SimpleNamespace(state=torch.ones(8, device=device))
    return hybrid


def runtime_for(pool):
    runtime = LayerKVRuntime(
        LayerKVConfig(enabled=True, mode="kvc-only", kvc_backend="per-layer-arena")
    )
    runtime._allocator = SimpleNamespace()
    runtime._req_to_token_pool = SimpleNamespace()
    assert runtime._can_support_kvc_pool(pool)
    runtime.physical_kvc_supported = True
    return runtime


@pytest.mark.parametrize("layer_ids", [(3, 11), (19, 27)])
def test_hybrid_storage_keeps_model_layer_ids(layer_ids):
    pool = make_pool(layer_ids=layer_ids)
    runtime = runtime_for(pool)
    assert runtime._kvc_layer_ids() == list(layer_ids)
    assert runtime._bytes_per_token_all_layers == 2 * 2 * 2 * 4 * 2
    view = runtime._kv_pool
    for layer_id, offset in pool.full_attention_layer_id_mapping.items():
        assert view._get_key_buffer(layer_id) is pool.full_kv_pool.k_buffer[offset]
        assert view._get_value_buffer(layer_id) is pool.full_kv_pool.v_buffer[offset]
    with pytest.raises(KeyError):
        view._get_key_buffer(layer_ids[0] - 1)


def test_mha_storage_still_uses_contiguous_layer_ids():
    pool = make_pool().full_kv_pool
    pool.start_layer = 8
    runtime = runtime_for(pool)
    assert runtime._kv_pool is pool
    assert runtime._kvc_layer_ids() == [8, 9]


@pytest.mark.parametrize("unsupported", ["mla", "fp4"])
def test_hybrid_rejects_unsupported_storage(unsupported):
    pool = make_pool()
    if unsupported == "mla":
        pool.use_mla = True
    else:
        pool.full_kv_pool = object.__new__(MHATokenToKVPoolFP4)
    runtime = LayerKVRuntime(LayerKVConfig())
    assert not runtime._can_support_kvc_pool(pool)


def test_hybrid_rejects_nonbijective_mapping():
    pool = make_pool()
    pool.full_attention_layer_id_mapping = {3: 0, 11: 0}
    with pytest.raises(ValueError, match="mapping"):
        HybridKVCStorageView(pool)


def test_metadata_rewrite_targets_full_attention_only():
    runtime = runtime_for(make_pool())
    full_metadata = SimpleNamespace(page_table=torch.zeros((1, 4), dtype=torch.int32))
    gdn_metadata = SimpleNamespace(state_indices=torch.tensor([9]))
    full_backend = SimpleNamespace(forward_metadata=full_metadata)
    runtime._last_forward_batch = SimpleNamespace(
        attn_backend=SimpleNamespace(
            full_attn_backend=full_backend,
            linear_attn_backend=SimpleNamespace(forward_metadata=gdn_metadata),
        )
    )
    table = torch.tensor([[3, 4, 5, 6]], dtype=torch.int32)

    def rewrite(metadata, req_to_token):
        metadata.page_table.copy_(req_to_token)
        return True

    runtime._rewrite_flashattention_page_table = rewrite
    assert runtime._rewrite_attention_metadata_for_layer(11, table)
    assert torch.equal(full_metadata.page_table, table)
    assert gdn_metadata.state_indices.tolist() == [9]
    assert runtime._virtual_metadata_kind() == "page_table"
    assert not runtime._virtual_metadata_uses_kv_indices()
    runtime._last_forward_batch.attn_backend = full_backend
    assert runtime._kvc_attention_backend() is full_backend


@pytest.mark.parametrize("extra_buffer", [False, True])
def test_layerkv_release_reuses_gdn_request_slots(extra_buffer):
    capacity = 6 if extra_buffer else 2
    mamba = object.__new__(MambaPool)
    mamba.device, mamba.size = "cpu", capacity
    mamba.free_slots = torch.arange(1, capacity + 1, dtype=torch.int64)
    mamba.mamba_cache = MambaPool.State(
        conv=[torch.zeros((2, capacity + 1, 2))],
        temporal=torch.zeros((2, capacity + 1, 2, 2)),
    )
    pool = object.__new__(HybridReqToTokenPool)
    pool.mamba_pool = mamba
    pool.device = "cpu"
    pool.free_slots = [1, 2]
    pool.enable_mamba_extra_buffer = extra_buffer
    pool.mamba_ping_pong_track_buffer_size = 2
    pool.req_index_to_mamba_index_mapping = torch.zeros(3, dtype=torch.int32)
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
        (3, 2), dtype=torch.int32
    )
    runtime = runtime_for(make_pool())
    runtime._req_to_token_pool = pool
    for _ in range(3):
        reqs = [
            SimpleNamespace(
                req_pool_idx=None,
                mamba_pool_idx=None,
                mamba_ping_pong_track_buffer=None,
            )
            for _ in range(2)
        ]
        assert pool.alloc(reqs) is not None
        assert mamba.available_size() == 0
        for req in reqs:
            runtime._free_layerkv_request_slot(req)
            assert req.req_pool_idx is None and req.mamba_pool_idx is None
        assert sorted(pool.free_slots) == [1, 2]
        assert sorted(mamba.free_slots.tolist()) == list(range(1, capacity + 1))


def test_finished_native_kv_has_one_allocator_owner():
    pool = make_pool()
    runtime = runtime_for(pool)
    runtime._allocator = TokenToKVPoolAllocator(16, torch.bfloat16, "cpu", pool, False)
    runtime._req_to_token_pool = ReqToTokenPool(2, 16, "cpu", False)
    req = SimpleNamespace(
        req_pool_idx=None,
        layerkv_per_layer_allocated=True,
        pop_committed_kv_cache=lambda: 2,
        pop_overallocated_kv_cache=lambda: (2, 2),
    )
    runtime._req_to_token_pool.alloc([req])
    native = runtime._allocator.alloc(2)
    runtime._req_to_token_pool.req_to_token[req.req_pool_idx, :2] = native.to(
        torch.int32
    )
    for layer_id in runtime._kvc_layer_ids():
        entry = _LayerKVResidencyEntry(
            req_idx=req.req_pool_idx,
            pos=0,
            state="resident",
            layer_id=layer_id,
            device_locs=native.tolist(),
            page_size=2,
        )
        key = (layer_id, req.req_pool_idx, 0)
        runtime._per_layer_residency[key] = entry
        runtime._index_per_layer_page_entry(entry)
        runtime._mark_req_layerkv_owned(req.req_pool_idx, [key])
        runtime._virtual_scratch_cache_by_layer[layer_id] = object()
    runtime._install_allocator_hooks(runtime._allocator)
    assert runtime.release_virtualized_request(req, SimpleNamespace(disable=True))
    assert not runtime._virtual_scratch_cache_by_layer
    native_free = set(runtime._allocator.free_pages.tolist())
    for layer_id in runtime._kvc_layer_ids():
        arena_free = set(
            runtime._bitset_to_locs(
                runtime._per_layer_pending_overwrite_bits_union(layer_id)
            )
        )
        assert not native_free.intersection(arena_free)
    runtime._cleanup_released_per_layer_entries()
    for layer_id in runtime._kvc_layer_ids():
        assert runtime._per_layer_overwrite_count(layer_id) == 2


def test_prefill_admits_after_native_slots_move_to_reusable_arena():
    from sglang.srt.managers.schedule_policy import PrefillAdder

    pool = make_pool()
    runtime = runtime_for(pool)
    pool.layerkv_runtime = runtime
    runtime._allocator = TokenToKVPoolAllocator(16, torch.bfloat16, "cpu", pool, False)
    runtime._req_to_token_pool = ReqToTokenPool(2, 16, "cpu", False)
    runtime._per_layer_arena_free_locs = {
        layer: [] for layer in runtime._kvc_layer_ids()
    }
    req = SimpleNamespace(
        req_pool_idx=None,
        layerkv_per_layer_allocated=True,
        pop_committed_kv_cache=lambda: 16,
        pop_overallocated_kv_cache=lambda: (16, 16),
    )
    runtime._req_to_token_pool.alloc([req])
    locs = runtime._allocator.alloc(16)
    runtime._req_to_token_pool.req_to_token[req.req_pool_idx, :16] = locs.to(
        torch.int32
    )
    runtime.register_native_kvc_runs_for_extend(
        reqs=[req], prefix_lens=[0], seq_lens=[16], locs=locs
    )
    runtime._install_allocator_hooks(runtime._allocator)
    assert runtime.release_virtualized_request(req, SimpleNamespace(disable=True))
    assert runtime._allocator.available_size() == 0
    assert runtime.stats.kvc_per_layer_physical_arena_min_free_tokens == 16

    # Exercise the actual prefill admission properties with an empty running
    # batch, no prefix cache, and only arena-owned free slots left.
    adder = object.__new__(PrefillAdder)
    adder.token_to_kv_pool_allocator = runtime._allocator
    adder.tree_cache = SimpleNamespace(evictable_size=lambda: 0)
    adder.is_hybrid_swa = adder.is_hybrid_ssm_cache = False
    adder.rem_total_token_offset = adder.cur_rem_token_offset = 0
    assert adder.rem_total_tokens == 16
    assert adder.cur_rem_tokens == 16
    # Admission must match actual allocation, across repeated request-slot reuse.
    for _ in range(4):
        req = SimpleNamespace(
            req_pool_idx=None,
            pop_committed_kv_cache=lambda: 8,
            pop_overallocated_kv_cache=lambda: (8, 8),
        )
        runtime._req_to_token_pool.alloc([req])
        locs = runtime.allocate_per_layer_request_slots(
            req=req, positions=list(range(8)), common_physical_locs=True
        )
        assert locs is not None and locs.unique().numel() == 8
        runtime._req_to_token_pool.req_to_token[req.req_pool_idx, :8] = locs.to(
            torch.int32
        )
        assert adder.rem_total_tokens == 8
        assert runtime.release_virtualized_request(req, SimpleNamespace(disable=True))
        assert adder.rem_total_tokens == 16
        assert runtime._allocator.available_size() == 0


def test_prefill_credit_uses_address_intersection_not_min_layer_free_count():
    runtime = runtime_for(make_pool())
    runtime._allocator = SimpleNamespace(available_size=lambda: 0, size=16)
    runtime._per_layer_arena_reserved_locs = {1, 2, 3, 4}
    runtime._per_layer_arena_free_locs = {3: [], 11: []}
    runtime._push_per_layer_overwrite_bits(3, runtime._locs_to_bitset([1, 2]))
    runtime._push_per_layer_overwrite_bits(11, runtime._locs_to_bitset([3, 4]))
    runtime._refresh_per_layer_allocator_stats()
    assert runtime.stats.kvc_per_layer_physical_arena_min_free_tokens == 2
    assert (
        runtime.get_scheduler_admission_credit_tokens(
            reason="decode_prealloc_admission"
        )
        == 0
    )
    assert runtime._alloc_common_per_layer_locs(1) is None


def test_common_prefill_reuse_merges_formats_excludes_loans_and_adds_native_capacity():
    pool = make_pool()
    runtime = runtime_for(pool)
    runtime._allocator = TokenToKVPoolAllocator(16, torch.bfloat16, "cpu", pool, False)
    reserved = runtime._allocator.alloc(8).tolist()
    runtime._per_layer_arena_reserved_locs = set(reserved)
    runtime._per_layer_arena_free_locs = {3: [1, 2], 11: [3, 4]}
    runtime._per_layer_arena_overwrite_pending_locs = {3: [3], 11: [1]}
    runtime._push_per_layer_overwrite_locs(3, [4])
    runtime._push_per_layer_overwrite_locs(11, [2])
    for layer in [3, 11]:
        runtime._push_per_layer_overwrite_bits(
            layer, runtime._locs_to_bitset([5, 6, 7, 8])
        )
    # A lent VMM page marks its KV locations allocated; terminal slots may also
    # be protected. Neither may leak into common prefill credit or allocation.
    runtime._per_layer_arena_allocated_locs = {3: {7}, 11: set()}
    runtime._per_layer_arena_protected_locs = {3: set(), 11: {8}}
    for _ in range(2):
        assert (
            runtime.get_scheduler_admission_credit_tokens(
                reason="decode_prealloc_admission"
            )
            == 6
        )
        assert runtime._allocator.available_size() == 8
        assert runtime._per_layer_overwrite_count(3) == 6
    # Neither source can independently cover ten slots, but their union can.
    locs = runtime._alloc_common_per_layer_locs(10)
    assert len(locs) == len(set(locs)) == 10
    assert not {7, 8}.intersection(locs)
    for layer in [3, 11]:
        assert set(locs).issubset(runtime._per_layer_arena_allocated_locs[layer])
        assert not set(locs).intersection(
            runtime._bitset_to_locs(
                runtime._per_layer_pending_overwrite_bits_union(layer)
            )
        )
    assert (
        runtime.get_scheduler_admission_credit_tokens(
            reason="decode_prealloc_admission"
        )
        == 4
    )


def test_native_extend_resets_reused_override_and_tracks_all_layers():
    runtime = runtime_for(make_pool())
    runtime._req_to_token_pool = ReqToTokenPool(2, 16, "cpu", False)
    req = SimpleNamespace(req_pool_idx=1)
    override = torch.full((2, 16), 999, dtype=torch.int32)
    runtime._per_layer_req_to_token_overrides[3] = override
    runtime._per_layer_req_to_token_owned.add(3)
    locs = torch.tensor([2, 5])
    runtime.register_native_kvc_runs_for_extend(
        reqs=[req],
        prefix_lens=[1],
        seq_lens=[3],
        locs=locs,
    )
    assert override[1, 1:3].tolist() == [2, 5]
    assert override[1, 0].item() == 999  # Preserve an existing chunk's prefix.
    assert runtime._per_layer_arena_reserved_locs == {2, 5}
    runtime._drop_entries_for_reqs(SimpleNamespace(req_pool_indices=torch.tensor([1])))
    assert 1 in runtime._per_layer_cleanup_state_by_req
    assert runtime._per_layer_req_generation(1) == 0
    assert runtime._per_layer_pending_overwrite_bits_union(3) == 0
    runtime._drop_per_layer_residency_for_finished_req(1, [])
    for layer in (3, 11):
        assert set(
            runtime._bitset_to_locs(
                runtime._per_layer_pending_overwrite_bits_union(layer)
            )
        ) == {2, 5}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA copies")
@pytest.mark.parametrize("prefix_lens", [[0, 0], [2, 1]])
def test_triton_extend_rewrite_respects_prefix_only_indices(prefix_lens):
    runtime = runtime_for(make_pool(device="cuda"))
    runtime._last_forward_batch = SimpleNamespace(
        batch_size=2,
        seq_lens=torch.tensor([4, 3], device="cuda", dtype=torch.int32),
        req_pool_indices=torch.tensor([0, 1], device="cuda", dtype=torch.int64),
    )
    table = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], device="cuda", dtype=torch.int32)
    # Overallocate a sentinel tail so the regression detects illegal writes
    # without poisoning the CUDA context (the real zero-prefix buffer is empty).
    indices = torch.full((16,), -1, device="cuda", dtype=torch.int32)
    metadata = SimpleNamespace(
        kv_indices=indices,
        kv_indptr=torch.tensor(
            [0, prefix_lens[0], sum(prefix_lens)], device="cuda", dtype=torch.int32
        ),
    )
    assert runtime._rewrite_triton_kv_indices(metadata, table)
    expected = table[0, : prefix_lens[0]].tolist() + table[1, : prefix_lens[1]].tolist()
    assert indices.tolist() == expected + [-1] * (16 - len(expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA copies")
@pytest.mark.parametrize("async_evict", [False, True])
def test_eviction_releases_arena_ownership_before_publishing_reuse(async_evict):
    runtime = runtime_for(make_pool(device="cuda", layer_ids=(3,)))
    runtime.config.runtime_profile = "optimized" if async_evict else "baseline"
    runtime.config.kvc_scheduler = "async-deadline" if async_evict else "sync"
    runtime._copy_stream = torch.cuda.Stream() if async_evict else None
    runtime._allocator.device = "cuda"
    runtime._req_to_token_pool = ReqToTokenPool(2, 16, "cuda", False)
    runtime._host_store = _LayerKVHostKVStore(runtime._kv_pool, 8, per_layer_mode=True)
    runtime._per_layer_arena_reserved_locs = {2, 5, 7}
    runtime._per_layer_arena_allocated_locs = {3: {2, 5, 7}}
    # Location7 represents another owner (such as an expert loan).
    entry = _LayerKVResidencyEntry(
        1, 0, "resident", layer_id=3, device_locs=[2, 5], page_size=2
    )
    runtime._per_layer_residency[(3, 1, 0)] = entry
    runtime._index_per_layer_page_entry(entry)
    runtime._mark_req_layerkv_owned(1, [(3, 1, 0)])
    assert runtime._issue_kvc_eviction_entries([entry])
    if async_evict:
        assert entry.state == "evicting"
        assert runtime._per_layer_arena_allocated_locs[3] == {2, 5, 7}
        runtime._finalize_kvc_evictions(block=True)
    assert entry.state == "offloaded"
    assert runtime._per_layer_arena_allocated_locs[3] == {7}
    assert runtime._common_per_layer_reusable_locs() == {2, 5}
    runtime.on_request_finished(SimpleNamespace(req_pool_idx=1))
    assert runtime._per_layer_arena_allocated_locs[3] == {7}
    assert runtime._common_per_layer_reusable_locs() == {2, 5}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA copies")
def test_reload_retained_host_backing_is_freed_on_finish():
    runtime = runtime_for(make_pool(device="cuda"))
    runtime._allocator.device = "cuda"
    runtime._req_to_token_pool = ReqToTokenPool(2, 16, "cuda", False)
    runtime._host_store = _LayerKVHostKVStore(runtime._kv_pool, 8, per_layer_mode=True)
    entry = _LayerKVResidencyEntry(
        req_idx=1,
        pos=0,
        state="resident",
        layer_id=3,
        device_locs=[2, 5],
        page_size=2,
    )
    key = (3, 1, 0)
    runtime._per_layer_residency[key] = entry
    runtime._index_per_layer_page_entry(entry)
    runtime._mark_req_layerkv_owned(1, [key])
    assert runtime._issue_kvc_eviction_entries([entry], force_additional_tokens=2)
    assert runtime._host_store.used_count == 2
    entry.device_locs = [9, 12]
    assert runtime._reload_required_kvc(None, selected_entries=[entry])
    runtime._finalize_reloaded_entries(block=True)
    runtime._drop_per_layer_residency_for_finished_req(1, [key])
    assert runtime._host_store.used_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA copies")
@pytest.mark.parametrize("per_layer", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_reload_after_slots_overwritten(per_layer, dtype):
    pool = make_pool(device="cuda", dtype=dtype)
    view = HybridKVCStorageView(pool)
    store = _LayerKVHostKVStore(view, 8, per_layer_mode=per_layer)
    old = torch.tensor([2, 5], device="cuda")
    new = torch.tensor([9, 12], device="cuda")
    expected = [
        (view._get_key_buffer(i)[old].clone(), view._get_value_buffer(i)[old].clone())
        for i in view.layer_ids
    ]
    pointers = [
        x.data_ptr() for x in pool.full_kv_pool.k_buffer + pool.full_kv_pool.v_buffer
    ]
    entries = [
        _LayerKVResidencyEntry(
            req_idx=1,
            pos=0,
            state="resident",
            layer_id=i,
            device_locs=[2, 5],
            page_size=2,
        )
        for i in view.layer_ids
    ]
    if per_layer:
        slots = store.alloc_per_layer(entries)
        assert slots is not None
        store.backup_per_layer(entries, slots)
        for offset, entry in enumerate(entries):
            entry.host_slots = slots[2 * offset : 2 * offset + 2]
            entry.device_locs = [9, 12]
            entry.state = "offloaded"
    else:
        slots = store.alloc(2)
        store.backup(old, slots)
    # A different request has now overwritten the old physical locations.
    for layer in view.layer_ids:
        view._get_key_buffer(layer)[old] = 77
        view._get_value_buffer(layer)[old] = -77
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    if per_layer:
        _, _, ready = store.reload_per_layer(entries, stream=stream, async_copy=True)
    else:
        _, _, ready = store.reload(slots, new, stream=stream, async_copy=True)
    torch.cuda.current_stream().wait_event(ready)
    for layer, (k, v) in zip(view.layer_ids, expected):
        assert torch.equal(view._get_key_buffer(layer)[new], k)
        assert torch.equal(view._get_value_buffer(layer)[new], v)
        assert (view._get_key_buffer(layer)[old] == 77).all()
    assert pointers == [
        x.data_ptr() for x in pool.full_kv_pool.k_buffer + pool.full_kv_pool.v_buffer
    ]
    assert (pool.mamba_pool.state == 1).all()
    if per_layer:
        for entry in entries:
            store.free_per_layer(entry.layer_id, entry.host_slots)
    else:
        store.free(slots)
    assert store.used_count == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
