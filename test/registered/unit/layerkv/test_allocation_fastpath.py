from types import SimpleNamespace

import torch

from sglang.srt.layerkv.kvc_allocator import LayerKVKvcAllocatorMixin
from sglang.srt.layerkv.kvc_residency import LayerKVKvcResidencyMixin


class _MappingHarness(LayerKVKvcAllocatorMixin):
    def __init__(self):
        self._per_layer_canonical_to_physical = {}
        self._per_layer_physical_to_canonical = {}
        self._per_layer_non_identity_mapping = set()


class _DecodeAllocationHarness:
    allocate_decode_slots_for_batch = (
        LayerKVKvcResidencyMixin.allocate_decode_slots_for_batch
    )

    def __init__(self):
        self._allocator = SimpleNamespace(device=torch.device("cpu"))
        self.config = SimpleNamespace(
            kvc_backend="per-layer-arena",
            runtime_profile="optimized",
        )
        self.stats = SimpleNamespace(
            kvc_per_layer_overwrite_reuse_count=0,
            kvc_per_layer_overwrite_reuse_token_count=0,
            kvc_per_layer_independent_alloc_count=0,
            kvc_per_layer_logical_token_count=0,
            kvc_resident_token_count=0,
            kvc_per_layer_arena_resident_token_count=0,
            kvc_resident_page_count=0,
            kvc_residency_entry_count=0,
            kvc_per_layer_arena_entry_count=0,
        )
        self._per_layer_req_to_token_owned = {1}
        self._per_layer_non_identity_mapping = set()
        self._per_layer_residency = {}
        self._per_layer_resident_token_count_fast = 0
        self._decode_step = 3
        self.generation_calls = []
        self.slot_updates = []
        self.owned = {}
        self.active_entries = 0

    def _per_layer_allocator_enabled(self):
        return True

    def _kvc_layer_ids(self):
        return [0, 1]

    def _alloc_per_layer_overwrite_locs_by_layer(self, _layer_ids, _count):
        return None

    def _alloc_common_per_layer_locs(self, _count):
        return [31, 32]

    def _per_layer_req_generation(self, req_idx):
        self.generation_calls.append(req_idx)
        return req_idx + 100

    def _set_per_layer_token_slots(
        self,
        layer_id,
        req_indices,
        positions,
        new_locs,
        *,
        req_tensor,
        pos_tensor,
    ):
        self.slot_updates.append(
            (
                layer_id,
                req_indices,
                positions,
                new_locs,
                req_tensor,
                pos_tensor,
            )
        )

    def _record_per_layer_mapping(self, *_args):
        raise AssertionError("identity mappings must not enter the mapping table")

    def _index_per_layer_page_entry(self, _entry):
        self.active_entries += 1

    def _sync_kvc_group(self, _entry):
        raise AssertionError("optimized allocation must not sync resident groups")

    def _mark_req_layerkv_owned(self, req_idx, keys):
        self.owned[req_idx] = keys

    def _per_layer_page_table_entry_count(self):
        return self.active_entries

    def _coresid_optimized_policy_enabled(self):
        return True


class _IndependentDecodeAllocationHarness(_DecodeAllocationHarness):
    def __init__(self):
        super().__init__()
        self._per_layer_req_to_token_owned = set()
        self.alloc_calls = []
        self.allocator_refresh_count = 0

    def _alloc_common_per_layer_locs(self, _count):
        return None

    def _alloc_per_layer_locs(self, layer_id, count, *, refresh_stats):
        self.alloc_calls.append((layer_id, count, refresh_stats))
        start = 31 + layer_id * 10
        return [start, start + 1]

    def _refresh_per_layer_allocator_stats(self):
        self.allocator_refresh_count += 1

    def _record_per_layer_mapping(self, *_args):
        pass


def test_identity_mapping_is_implicit_and_cleans_stale_non_identity_entries():
    runtime = _MappingHarness()

    runtime._record_per_layer_mapping(2, 10, 10)
    assert runtime._per_layer_canonical_to_physical == {}
    assert runtime._per_layer_physical_to_canonical == {}

    runtime._record_per_layer_mapping(2, 10, 20)
    runtime._record_per_layer_mapping(2, 11, 21)
    runtime._record_per_layer_mapping(2, 10, 10)
    assert runtime._per_layer_canonical_to_physical == {2: {11: 21}}
    assert runtime._per_layer_physical_to_canonical == {2: {21: 11}}
    assert runtime._per_layer_non_identity_mapping == {2}

    runtime._record_per_layer_mapping(2, 21, 21)
    assert runtime._per_layer_canonical_to_physical == {}
    assert runtime._per_layer_physical_to_canonical == {}
    assert runtime._per_layer_non_identity_mapping == set()


def test_decode_allocation_reuses_batch_metadata_and_updates_fast_stats():
    runtime = _DecodeAllocationHarness()
    reqs = [SimpleNamespace(req_pool_idx=4), SimpleNamespace(req_pool_idx=9)]
    seq_lens = torch.tensor([100, 200], dtype=torch.int64)
    seq_lens_cpu = torch.tensor([5, 7], dtype=torch.int64)
    req_pool_indices = torch.tensor([4, 9], dtype=torch.int64)
    batch = SimpleNamespace(
        reqs=reqs,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        req_pool_indices=req_pool_indices,
    )

    locs = runtime.allocate_decode_slots_for_batch(batch)

    assert locs.tolist() == [31, 32]
    assert runtime.generation_calls == [4, 9]
    assert set(runtime._per_layer_residency) == {
        (0, 4, 5),
        (0, 9, 7),
        (1, 4, 5),
        (1, 9, 7),
    }
    assert len(runtime.slot_updates) == 1
    _, _, positions, _, used_reqs, used_positions = runtime.slot_updates[0]
    assert positions == [5, 7]
    assert used_reqs is req_pool_indices
    assert used_positions is seq_lens
    assert len(runtime.owned[4]) == 2
    assert len(runtime.owned[9]) == 2
    assert runtime.stats.kvc_resident_token_count == 4
    assert runtime.stats.kvc_residency_entry_count == 4
    assert all(req.layerkv_per_layer_allocated for req in reqs)
    assert all(req.skip_radix_cache_insert for req in reqs)


def test_decode_independent_allocations_refresh_allocator_stats_once():
    runtime = _IndependentDecodeAllocationHarness()
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(req_pool_idx=4), SimpleNamespace(req_pool_idx=9)],
        seq_lens=torch.tensor([5, 7], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([5, 7], dtype=torch.int64),
        req_pool_indices=torch.tensor([4, 9], dtype=torch.int64),
    )

    runtime.allocate_decode_slots_for_batch(batch)

    assert runtime.alloc_calls == [(0, 2, False), (1, 2, False)]
    assert runtime.allocator_refresh_count == 1
