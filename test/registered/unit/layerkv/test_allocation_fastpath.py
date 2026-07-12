from types import SimpleNamespace

import torch

from sglang.srt.layerkv.common_types import _LayerKVResidencyEntry
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
    _append_per_layer_resident_tail = (
        LayerKVKvcResidencyMixin._append_per_layer_resident_tail
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
        self._per_layer_resident_span_by_req_layer_page = {}
        self._per_layer_resident_token_count_fast = 0
        self._decode_step = 3
        self.allocation_round = 0
        self.generation_calls = []
        self.slot_updates = []
        self.cleanup_appends = []
        self.owned = {}
        self.active_entries = 0

    def _per_layer_allocator_enabled(self):
        return True

    def _kvc_layer_ids(self):
        return [0, 1]

    def _alloc_per_layer_overwrite_locs_by_layer(self, _layer_ids, _count):
        return None

    def _alloc_common_per_layer_locs(self, _count):
        start = 31 + self.allocation_round * 2
        self.allocation_round += 1
        return [start, start + 1]

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

    def _index_per_layer_page_entry(self, entry):
        self.active_entries += 1
        page_id = int(entry.pos) // self._per_layer_virtual_page_tokens()
        self._per_layer_resident_span_by_req_layer_page[
            (int(entry.req_idx), int(entry.layer_id), page_id)
        ] = entry

    def _per_layer_virtual_page_tokens(self):
        return 16

    def _track_per_layer_cleanup_append(self, entry, physical_loc):
        self.cleanup_appends.append((entry, physical_loc))

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


class _CleanupHarness(LayerKVKvcAllocatorMixin):
    def __init__(self):
        self.config = SimpleNamespace(kvc_backend="per-layer-arena")
        self._per_layer_cleanup_state_by_req = {}

    def _per_layer_entry_is_current(self, _entry):
        return True


class _ResidencyIndexHarness(LayerKVKvcResidencyMixin):
    def __init__(self):
        self.config = SimpleNamespace(kvc_backend="per-layer-arena")
        self._decode_step = 4
        self._per_layer_page_table = {}
        self._per_layer_page_keys_by_req = {}
        self._per_layer_page_bits_by_req_layer_state = {}
        self._per_layer_resident_span_by_req_layer_page = {}
        self._per_layer_resident_queue_by_layer = {}
        self._per_layer_resident_queue_by_req_layer = {}
        self._per_layer_active_entry_count = 0
        self._per_layer_active_entry_count_by_req = {}

    def _per_layer_virtual_page_tokens(self):
        return 16

    def _per_layer_entry_is_current(self, _entry):
        return True


def _batch(reqs, positions):
    return SimpleNamespace(
        reqs=reqs,
        seq_lens=torch.tensor(positions, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(positions, dtype=torch.int64),
        req_pool_indices=torch.tensor(
            [req.req_pool_idx for req in reqs], dtype=torch.int64
        ),
    )


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


def test_decode_allocation_appends_to_resident_tail_within_page():
    runtime = _DecodeAllocationHarness()
    reqs = [SimpleNamespace(req_pool_idx=4), SimpleNamespace(req_pool_idx=9)]

    runtime.allocate_decode_slots_for_batch(_batch(reqs, [5, 7]))
    runtime.allocate_decode_slots_for_batch(_batch(reqs, [6, 8]))

    assert len(runtime._per_layer_residency) == 4
    assert runtime.active_entries == 4
    assert len(runtime.cleanup_appends) == 4
    assert runtime._per_layer_residency[(0, 4, 5)].device_locs == [31, 33]
    assert runtime._per_layer_residency[(0, 4, 5)].token_count == 2
    assert runtime._per_layer_residency[(1, 9, 7)].device_locs == [32, 34]
    assert runtime.stats.kvc_resident_token_count == 8
    assert runtime.stats.kvc_residency_entry_count == 4


def test_decode_tail_append_rejects_page_boundary_and_stale_generation():
    runtime = _DecodeAllocationHarness()
    entry = _LayerKVResidencyEntry(
        req_idx=4,
        pos=15,
        state="resident",
        layer_id=1,
        device_loc=41,
        device_locs=[41],
        page_size=1,
        generation=104,
    )
    runtime._per_layer_resident_span_by_req_layer_page[(4, 1, 0)] = entry

    assert (
        runtime._append_per_layer_resident_tail(
            layer_id=1,
            req_idx=4,
            pos=16,
            physical_loc=42,
            generation=104,
        )
        is None
    )
    runtime._per_layer_resident_span_by_req_layer_page[(4, 1, 1)] = entry
    assert (
        runtime._append_per_layer_resident_tail(
            layer_id=1,
            req_idx=4,
            pos=16,
            physical_loc=42,
            generation=105,
        )
        is None
    )
    assert entry.device_locs == [41]
    assert entry.token_count == 1


def test_cleanup_tracking_counts_only_the_appended_token():
    runtime = _CleanupHarness()
    entry = _LayerKVResidencyEntry(
        req_idx=4,
        pos=5,
        state="resident",
        layer_id=1,
        device_loc=41,
        device_locs=[41, 42],
        page_size=2,
    )

    runtime._track_per_layer_cleanup_append(entry, 42)
    state = runtime._per_layer_cleanup_state_by_req[4]
    assert state.token_count == 2
    assert state.loc_counts_by_layer == {1: 2}

    entry.device_locs.append(43)
    entry.page_size = 3
    runtime._track_per_layer_cleanup_append(entry, 43)
    assert state.token_count == 3
    assert state.loc_counts_by_layer == {1: 3}


def test_tail_append_preserves_real_page_index_invariants():
    runtime = _ResidencyIndexHarness()
    entry = _LayerKVResidencyEntry(
        req_idx=4,
        pos=5,
        state="resident",
        layer_id=1,
        device_loc=41,
        device_locs=[41],
        page_size=1,
        generation=2,
    )
    runtime._index_per_layer_page_entry(entry)

    appended = runtime._append_per_layer_resident_tail(
        layer_id=1,
        req_idx=4,
        pos=6,
        physical_loc=42,
        generation=2,
    )

    assert appended is entry
    assert runtime._per_layer_active_entry_count == 1
    assert runtime._per_layer_page_table[(4, 1, 0)] == [entry]
    assert runtime._per_layer_page_entry_for_span(1, 4, 5, 2) is entry
    assert entry.logical_positions() == [5, 6]
    assert entry.device_loc_list() == [41, 42]
