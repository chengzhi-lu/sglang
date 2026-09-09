"""CPU-only donor discovery with realistic fragmented allocator bookkeeping."""

import random
from types import SimpleNamespace

import pytest

from sglang.srt.layerkv.config_stats import LayerKVConfig
from sglang.srt.layerkv.kvc_allocator import LayerKVKvcAllocatorMixin
from sglang.srt.layerkv.shared_expert import SharedExpertController


class Buffer:
    shape = (16385, 1024)

    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer

    def __getitem__(self, _index):
        return SimpleNamespace(nbytes=1024)


class Runtime(LayerKVKvcAllocatorMixin):
    def __init__(self):
        self.config = LayerKVConfig(shared_expert_free_kv_donors=True)
        self._per_layer_offloaded_runs_by_req_layer = {}
        self._per_layer_cleanup_state_by_req = {}
        self._per_layer_arena_reserved_locs = set(range(1, 16385))
        self._per_layer_arena_free_locs = {}
        self._per_layer_arena_overwrite_pending_locs = {}
        self._per_layer_arena_overwrite_bitmaps = {}
        self._per_layer_arena_allocated_locs = {}
        self._per_layer_arena_protected_locs = {}
        self.pending_bits = {}
        for layer in self._kvc_layer_ids():
            self._per_layer_arena_free_locs[layer] = list(range(1, 4096, 2))
            self._per_layer_arena_allocated_locs[layer] = set(range(8192, 16385))
            self.pending_bits[layer] = self._locs_to_bitset(range(4097, 8192, 2))
        self._kv_pool = SimpleNamespace(
            _get_key_buffer=lambda layer: Buffer(layer * 2),
            _get_value_buffer=lambda layer: Buffer(layer * 2 + 1),
        )

    def _kvc_layer_ids(self):
        return list(range(10))

    def _per_layer_pending_overwrite_bits_union(self, layer):
        return self.pending_bits[layer]


def controller():
    runtime = Runtime()
    arena = SimpleNamespace(
        page_bytes=2 * 1024**2,
        allocations={
            i: SimpleNamespace(pages=dict.fromkeys(range(9))) for i in range(20)
        },
    )
    return SharedExpertController(runtime, arena)


def test_fragmented_free_tokens_cannot_back_a_whole_expert_page():
    c = controller()
    assert c._donors() == []


def test_idle_virtual_scratch_page_is_donor_without_arena_credit():
    import torch

    r = Runtime()
    r.config = LayerKVConfig(
        kvc_backend="per-layer-arena",
        shared_expert_free_kv_donors=True,
        shared_expert_lend_virtual_scratch=True,
    )
    scratch = set(range(2048, 4096))
    r._virtual_scratch_locs = torch.tensor(sorted(scratch), dtype=torch.int64)
    r._virtual_scratch_locs_host = scratch
    # Scratch is allocated directly from the native token allocator and is not
    # part of the ordinary per-layer arena/free-list ledger.
    r._per_layer_arena_reserved_locs = {1}
    arena = SimpleNamespace(
        page_bytes=2 * 1024**2,
        allocations={
            i: SimpleNamespace(pages=dict.fromkeys(range(9))) for i in range(20)
        },
    )
    c = SharedExpertController(r, arena)

    donors = c._donors()

    assert len(donors) == 20  # ten layers, K/V page pair for each layer
    assert all(donor[3] == list(range(2048, 4096)) for donor in donors)
    assert c._donor_last_scan["scratch_eligible"] == len(donors)
    assert scratch.isdisjoint(r._common_per_layer_reusable_locs())


def test_busy_virtual_scratch_page_never_falls_through_to_ordinary_donor():
    import torch

    r = Runtime()
    r.config = LayerKVConfig(
        kvc_backend="per-layer-arena",
        shared_expert_free_kv_donors=True,
        shared_expert_lend_virtual_scratch=True,
    )
    scratch = set(range(2048, 4096))
    r._virtual_scratch_locs = torch.tensor(sorted(scratch), dtype=torch.int64)
    r._virtual_scratch_locs_host = scratch
    r._per_layer_arena_reserved_locs = {1}
    r._pending_virtual_kvc_materialize = {"pending": object()}
    arena = SimpleNamespace(
        page_bytes=2 * 1024**2,
        allocations={
            i: SimpleNamespace(pages=dict.fromkeys(range(9))) for i in range(20)
        },
    )
    c = SharedExpertController(r, arena)

    assert c._donors() == []
    assert c._donor_last_scan["scratch_busy"] == 20


def test_idle_scratch_can_fill_configured_growth_target_in_one_scan():
    import torch

    r = Runtime()
    r.config = LayerKVConfig(
        kvc_backend="per-layer-arena",
        shared_expert_free_kv_donors=True,
        shared_expert_lend_virtual_scratch=True,
    )
    scratch = set(range(2048, 4096))
    r._virtual_scratch_locs = torch.tensor(sorted(scratch), dtype=torch.int64)
    r._virtual_scratch_locs_host = scratch
    c = SharedExpertController(r, SimpleNamespace())
    donors = [(None, page, 0, list(scratch)) for page in range(12)]

    assert c._growth_slots_per_scan(
        6, 2, donors, 3, scratch_lending_active=True
    ) == 4
    assert c._growth_slots_per_scan(
        6, 2, donors[:6], 3, scratch_lending_active=True
    ) == 2
    assert c._growth_slots_per_scan(
        6, 2, donors, 3, scratch_lending_active=False
    ) == 1


def test_context_pressure_blocks_new_scratch_loans():
    import torch

    r = Runtime()
    r.config = LayerKVConfig(
        kvc_backend="per-layer-arena",
        shared_expert_free_kv_donors=True,
        shared_expert_lend_virtual_scratch=True,
    )
    r._virtual_scratch_locs = torch.tensor(
        list(range(2048, 4096)), dtype=torch.int64
    )
    c = SharedExpertController(r, SimpleNamespace())

    assert not c._prepare_virtual_scratch_for_lending(context_pressure=True)
    assert not c._scratch_donor_scan_enabled
    assert c.scratch_lend_skip_count == 1
    assert c.scratch_lend_skip_reason == "context-pressure"


@pytest.mark.parametrize("seed", range(4))
def test_common_credit_count_matches_address_reference_after_mutations(seed):
    r = controller().runtime
    rng = random.Random(seed)
    for _ in range(5):
        r._per_layer_arena_reserved_locs = set(rng.sample(range(1, 400), 350))
        for layer in r._kvc_layer_ids():
            r._per_layer_arena_free_locs[layer] = rng.sample(range(1, 400), 260)
            r._per_layer_arena_overwrite_pending_locs[layer] = rng.sample(
                range(1, 400), 40
            )
            r.pending_bits[layer] = r._locs_to_bitset(rng.sample(range(1, 400), 200))
            r._per_layer_arena_allocated_locs[layer] = set(
                rng.sample(range(1, 400), 10)
            )
            r._per_layer_arena_protected_locs[layer] = set(
                rng.sample(range(1, 400), 10)
            )
        expected = r._common_per_layer_reusable_locs()
        assert r._common_per_layer_reusable_token_count() == len(expected)
        assert r._common_per_layer_reusable_locs() == expected


@pytest.mark.parametrize("legacy_bitmap", [False, True])
def test_admission_uses_exact_count_and_legacy_fallback(legacy_bitmap):
    import torch
    from sglang.srt.layerkv.runtime import LayerKVRuntime

    r = controller().runtime
    r.config.mode = "kvc-expert"
    r.config.kvc_backend = "per-layer-arena"
    r.physical_kvc_supported = True
    r._offloaded_token_count = lambda: 123
    r.stats = SimpleNamespace()
    if legacy_bitmap:
        bitmap = torch.zeros(16385, dtype=torch.bool)
        bitmap[2:4096:2] = True
        r._per_layer_arena_overwrite_bitmaps[0] = bitmap
    expected = len(r._common_per_layer_reusable_locs())
    if not legacy_bitmap:

        def no_expansion():
            raise AssertionError("count-only admission must not expand addresses")

        r._common_per_layer_reusable_locs = no_expansion
    credit, raw, reason = LayerKVRuntime._scheduler_credit_tokens(
        r, reason="decode_prealloc_admission"
    )
    assert credit == expected
    assert raw == 123
    assert reason == "per_layer_arena_physical_allocator"
    assert r.stats.kvc_per_layer_physical_arena_common_free_tokens == expected


def test_overwrite_decode_packs_across_request_bit_chunks():
    import torch
    from sglang.srt.layerkv.runtime import LayerKVRuntime

    c = controller()
    r = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            kvc_backend="per-layer-arena",
            shared_expert_free_kv_donors=True,
        )
    )
    c.runtime = r
    r._allocator = SimpleNamespace(device=torch.device("cpu"))
    r._per_layer_allocator_enabled = lambda: True
    r._kvc_layer_ids = lambda: [0]
    r._kv_pool = SimpleNamespace(
        _get_key_buffer=lambda _: Buffer(0),
        _get_value_buffer=lambda _: Buffer(1),
    )
    r._per_layer_arena_reserved_locs = set(range(16, 80))
    c.arena.page_bytes = 16 * 1024
    for allocation in c.arena.allocations.values():
        allocation.pages = dict.fromkeys(range(1, 5))
    # Two completed interleaved requests publish separate nonoverlapping chunks.
    for parity in (0, 1):
        r._push_per_layer_overwrite_bits(
            0, r._locs_to_bitset(range(16 + parity, 80, 2))
        )

    def no_common(_count):
        raise AssertionError("decode must exercise overwrite-first allocation")

    r._alloc_common_per_layer_locs = no_common
    reqs = [SimpleNamespace(req_pool_idx=i) for i in range(2)]
    outputs = []
    for step in range(8):
        batch = SimpleNamespace(
            reqs=reqs, seq_lens=torch.tensor([step, step]), seq_lens_cpu=[step, step]
        )
        outputs.extend(r.allocate_decode_slots_for_batch(batch).tolist())
    assert len(set(outputs)) == 16
    owned = 0
    for state in r._per_layer_cleanup_state_by_req.values():
        owned |= state.loc_bits_by_layer[0]
    assert owned == r._locs_to_bitset(outputs)
    assert r.stats.kvc_per_layer_overwrite_reuse_count == 8
    assert r._per_layer_pending_overwrite_bits_union(0).bit_count() == 48
    donors = c._donors()
    assert all(owned & r._locs_to_bitset(d[3]) == 0 for d in donors)
    assert len(donors) == 6  # One live page, three wholly free K/V page pairs.


def test_shared_overwrite_merge_counts_unique_published_addresses():
    from sglang.srt.layerkv.runtime import LayerKVRuntime

    r = LayerKVRuntime(LayerKVConfig(shared_expert_free_kv_donors=True))
    for locs in ([16, 18], [17, 19], [18, 19]):
        r._push_per_layer_overwrite_bits(0, r._locs_to_bitset(locs))
    assert r._per_layer_arena_overwrite_pending_bit_counts[0] == 4
    assert len(r._per_layer_arena_overwrite_pending_bit_chunks[0]) == 1
    popped = r._pop_per_layer_overwrite_locs(0, 2)
    assert popped == [19, 18]
    assert r._per_layer_arena_overwrite_pending_bit_counts[0] == 2
    r._push_per_layer_overwrite_bits(0, r._locs_to_bitset(popped))
    assert r._pop_per_layer_overwrite_locs(0, 5) == [19, 18, 17, 16]
    assert r._per_layer_arena_overwrite_pending_bit_counts.get(0, 0) == 0


def test_long_request_free_order_changes_short_request_donor_pages():
    """Controlled lifecycle, not a reconstruction of the model's unseen IDs."""
    from sglang.srt.layerkv.runtime import LayerKVRuntime

    def replay(pack):
        c = controller()
        r = LayerKVRuntime(
            LayerKVConfig(
                enabled=True,
                mode="kvc-expert",
                kvc_backend="per-layer-arena",
                shared_expert_free_kv_donors=True,
            )
        )
        c.runtime = r
        r._kvc_layer_ids = lambda: [0]
        r._kv_pool = SimpleNamespace(
            _get_key_buffer=lambda _: Buffer(0),
            _get_value_buffer=lambda _: Buffer(1),
        )
        c.arena.page_bytes = 16 * 1024
        for allocation in c.arena.allocations.values():
            allocation.pages = dict.fromkeys(range(1, 5))
        locs = set(range(16, 80))
        r._per_layer_arena_reserved_locs = locs.copy()
        r._per_layer_arena_free_locs = {0: sorted(locs)}
        r._per_layer_arena_common_free_locs = locs.copy()
        r._per_layer_arena_common_free_order = sorted(locs)
        r._per_layer_arena_common_free_dirty = False
        # Already mapped CPU metadata fixture; no native allocator/GPU growth.
        r._ensure_per_layer_physical_arena = (
            lambda count: len(r._per_layer_arena_common_free_locs) >= count
        )
        r._refresh_per_layer_allocator_stats = lambda: None
        owners = [[], []]
        for _ in range(32):
            batch = r._alloc_common_per_layer_locs(2)
            for index, loc in enumerate(batch):
                owners[index].append(loc)
        assert set(owners[0]).isdisjoint(owners[1])
        for owner in owners:
            r._free_per_layer_locs_batch({0: owner}, refresh=False)
            r._ensure_per_layer_common_free_current()
        assert r._per_layer_arena_common_free_locs == locs
        if pack:
            # Counterfactual only: this ordering did not fix the model's B32
            # regrowth failure and is deliberately not a runtime policy.
            r._per_layer_arena_common_free_order = sorted(locs)
        short = r._alloc_common_per_layer_locs(16)
        r._per_layer_cleanup_state_by_req[2] = SimpleNamespace(
            loc_bits_by_layer={0: r._locs_to_bitset(short)}
        )
        donors = c._donors()
        assert all(set(d[3]).isdisjoint(short) for d in donors)
        assert len(r._per_layer_arena_common_free_locs) == 48
        return len(donors), c._donor_last_scan

    historical, historical_scan = replay(False)
    packed, packed_scan = replay(True)
    assert historical == 4  # Two wholly free physical pages, K and V each.
    assert packed == 6  # Three pages: same16 live and48 free token addresses.
    assert historical_scan["cleanup"] == 4
    assert packed_scan["cleanup"] == 2


@pytest.mark.parametrize(
    "reason",
    ["eligible", "cleanup", "not_free", "not_reserved", "allocated", "protected"],
)
def test_last_scan_rejection_accounting(reason):
    c = controller()
    r = c.runtime
    r._kvc_layer_ids = lambda: [0]
    r._per_layer_arena_free_locs[0] = list(range(2048, 4096))
    r._per_layer_arena_allocated_locs[0] = set()
    r.pending_bits[0] = 0
    for allocation in c.arena.allocations.values():
        allocation.pages = {1: None}
    if reason == "cleanup":
        r._per_layer_cleanup_state_by_req[0] = SimpleNamespace(
            loc_bits_by_layer={0: 1 << 2048}
        )
    elif reason == "not_free":
        r._per_layer_arena_free_locs[0].remove(2048)
    elif reason == "not_reserved":
        r._per_layer_arena_reserved_locs.remove(2048)
    elif reason == "allocated":
        r._per_layer_arena_allocated_locs[0].add(2048)
    elif reason == "protected":
        r._per_layer_arena_protected_locs[0] = {2048}
    donors = c._donors()
    scan = c._donor_last_scan.copy()
    assert scan["mapped"] == scan[reason] == 2
    assert sum(v for k, v in scan.items() if k != "mapped") == 2
    assert len(donors) == (2 if reason == "eligible" else 0)
    c._donors()
    assert c._donor_last_scan == scan  # Snapshot, not a cumulative counter.


def set_reference_donors(c):
    """Original ordinary-free discovery, independent of the page-first path."""
    r = c.runtime
    result = []
    for layer in sorted(r._kvc_layer_ids()):
        free = set(r._per_layer_arena_free_locs.get(layer, []))
        free.update(r._bitset_to_locs(r._per_layer_pending_overwrite_bits_union(layer)))
        free.intersection_update(r._per_layer_arena_reserved_locs)
        free.difference_update(r._per_layer_arena_allocated_locs.get(layer, set()))
        free.difference_update(r._per_layer_arena_protected_locs.get(layer, set()))
        for state in r._per_layer_cleanup_state_by_req.values():
            free.difference_update(
                r._bitset_to_locs(state.loc_bits_by_layer.get(layer, 0))
            )
        for buffer in (
            r._kv_pool._get_key_buffer(layer),
            r._kv_pool._get_value_buffer(layer),
        ):
            allocation = c.arena.allocations[buffer.data_ptr()]
            tokens = c.arena.page_bytes // buffer[0].nbytes
            for page in sorted(allocation.pages):
                start, end = page * tokens, (page + 1) * tokens
                if (
                    start
                    and end <= buffer.shape[0]
                    and free.issuperset(range(start, end))
                ):
                    result.append((allocation, page, layer, list(range(start, end))))
    return result


@pytest.mark.parametrize("seed", range(8))
def test_page_first_discovery_matches_set_reference_across_ownership_changes(seed):
    c = controller()
    r = c.runtime
    c.arena.page_bytes = 16 * 1024
    rng = random.Random(seed)
    r._per_layer_arena_reserved_locs = set(range(1, 145))
    for layer in r._kvc_layer_ids():
        r._per_layer_arena_free_locs[layer] = list(range(1, 145, 2))
        r.pending_bits[layer] = r._locs_to_bitset(range(2, 145, 2))
        r._per_layer_arena_allocated_locs[layer] = set(rng.sample(range(1, 145), 3))
        r._per_layer_arena_protected_locs[layer] = set(rng.sample(range(1, 145), 3))
    assert c._donors() == set_reference_donors(c)
    # All sources can change in-place: no generation counter or stale cache.
    r._per_layer_arena_reserved_locs.difference_update(range(32, 48))
    r._per_layer_arena_free_locs[0].extend(range(80, 96))
    r.pending_bits[1] &= ~r._locs_to_bitset(range(64, 80))
    r._per_layer_arena_allocated_locs[2].clear()
    r._per_layer_arena_protected_locs[3].update(range(112, 128))
    r._per_layer_cleanup_state_by_req[1] = SimpleNamespace(
        loc_bits_by_layer={4: 1 << 48}
    )
    del c.arena.allocations[10].pages[4]
    assert c._donors() == set_reference_donors(c)


def test_common_free_discovery_does_not_reencode_active_locations():
    c = controller()
    encode = c.runtime._locs_to_bitset
    active = list(c.runtime._per_layer_arena_allocated_locs.values())

    def counted(locs, **kwargs):
        # The exact common-free query already excludes these addresses using
        # sets. Reencoding every active token in every layer dominated misses.
        assert all(locs is not values for values in active)
        return encode(locs, **kwargs)

    c.runtime._locs_to_bitset = counted
    assert c._donors() == []


@pytest.mark.parametrize("exclusion", ["allocated", "protected", "cleanup"])
def test_backed_donors_still_exclude_active_locations(exclusion):
    c = controller()
    r = c.runtime
    locs = list(range(8192, 10240))
    bits = r._locs_to_bitset(locs)
    r.pending_bits[0] |= bits
    r._per_layer_arena_allocated_locs[0].difference_update(locs)
    if exclusion == "cleanup":
        r._per_layer_cleanup_state_by_req[1] = SimpleNamespace(
            loc_bits_by_layer={0: bits}
        )
    else:
        getattr(r, f"_per_layer_arena_{exclusion}_locs").setdefault(0, set()).update(
            locs
        )
    r._per_layer_entry_is_current = lambda _: True
    r._per_layer_offloaded_runs_by_req_layer[(1, 0)] = [
        SimpleNamespace(
            state="offloaded",
            token_count=len(locs),
            host_slot_list=lambda: locs,
            layer_id=0,
            evicted_device_locs=locs,
        )
    ]
    assert c._donors() == []


@pytest.mark.parametrize("limit", [0, 1, 63, 257, 20000])
@pytest.mark.parametrize("locs", [[], [0, 2, 100000], list(range(4000, 9000, 3))])
def test_bitset_expansion_matches_locations(locs, limit):
    r = Runtime()
    assert r._bitset_to_locs(r._locs_to_bitset(locs), limit=limit) == (
        locs[:limit] if limit else locs
    )


@pytest.mark.parametrize("minimum", [0, 1, 5000, 100000])
def test_dense_encoding_matches_reference_with_duplicates_and_unordered_inputs(minimum):
    r = Runtime()
    locs = list(range(-8, 8192)) + [4096] * 512
    random.Random(1).shuffle(locs)
    expected = 0
    for loc in locs:
        if loc >= minimum:
            expected |= 1 << loc
    for values in (locs, set(locs), iter(locs)):
        assert r._locs_to_bitset(values, min_value=minimum) == expected


def test_encoding_keeps_negative_shift_error_when_negative_minimum_is_requested():
    with pytest.raises(ValueError):
        Runtime()._locs_to_bitset(list(range(-1, 1000)), min_value=-1)
