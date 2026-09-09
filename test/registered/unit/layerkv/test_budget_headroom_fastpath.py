"""Outstanding expert loans must not force unnecessary exact KV discovery."""

import random
from types import SimpleNamespace

import pytest

from sglang.srt.layerkv.residency_budget import ResidencyBudget
from sglang.srt.layerkv.shared_expert import SharedExpertController


@pytest.mark.parametrize(
    "native,common,shortage,expected_scans,expected_recall",
    [
        (128, 0, 0, 0, False),
        (127, 1, 0, 1, False),
        (127, 0, 0, 1, True),
        (128, 0, 1, 0, True),
    ],
)
def test_native_headroom_is_sufficient_lower_bound(
    native, common, shortage, expected_scans, expected_recall
):
    scans, recalls = [], []

    def discover():
        scans.append(1)
        return set(range(common))

    def sufficient(count):
        return len(discover()) >= count

    controller = SharedExpertController.__new__(SharedExpertController)
    controller.state = SimpleNamespace(slot_capacity=24, full_num_experts=256)
    controller.base_slots, controller.extra_slots = 16, 8
    controller.budget = ResidencyBudget(16, 24)
    controller.budget.target = 24
    controller.admission_shortage_tokens = shortage
    controller._next_donor_scan_step = 0
    controller.runtime = SimpleNamespace(
        _decode_step=20,
        _last_forward_batch=SimpleNamespace(batch_size=8),
        _allocator=SimpleNamespace(available_size=lambda: native),
        _common_per_layer_reusable_locs=discover,
        _has_common_per_layer_reusable_tokens=sufficient,
        stats=SimpleNamespace(
            native_schedule_waiting_queue_len=1 if shortage else 0,
            expert_materialize_count=0,
        ),
    )

    def recall():
        recalls.append(1)
        controller.state.slot_capacity = 16

    controller.recall = recall
    controller.after_decode()
    assert len(scans) == expected_scans
    assert bool(recalls) == expected_recall


def test_headroom_scan_is_deferred_between_budget_decisions():
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.state = SimpleNamespace(slot_capacity=24, full_num_experts=256)
    controller.base_slots, controller.extra_slots = 16, 8
    controller.budget = ResidencyBudget(16, 24, interval=8, headroom_steps=16)
    controller.budget.target = 24
    controller.budget.next_decision_step = 100
    controller.budget.samples = 0
    controller.admission_shortage_tokens = 0
    controller._next_donor_scan_step = 100
    controller.last_context_reserve_tokens = 128

    def unexpected(*_args, **_kwargs):
        raise AssertionError("headroom discovery is not due between decisions")

    controller._has_common_per_layer_reusable_tokens = unexpected
    controller._projected_kv_residency_demand = lambda _batch: (0, 4096, 0, 0)

    def unexpected_allocator():
        raise AssertionError("allocator probe is not due between decisions")

    controller.runtime = SimpleNamespace(
        _decode_step=20,
        _last_forward_batch=SimpleNamespace(batch_size=8),
        _allocator=SimpleNamespace(available_size=unexpected_allocator),
        stats=SimpleNamespace(
            native_schedule_waiting_queue_len=0,
            expert_materialize_count=0,
        ),
    )

    controller.after_decode()
    assert controller.state.slot_capacity == 24


def test_bitset_headroom_matches_exact_query_without_enumerating_all_free():
    from sglang.srt.layerkv.config_stats import LayerKVConfig
    from sglang.srt.layerkv.runtime import LayerKVRuntime

    r = LayerKVRuntime(LayerKVConfig())
    r._kvc_layer_ids = lambda: [3, 7]
    r._per_layer_arena_reserved_locs = set(range(1, 8193))
    for layer in [3, 7]:
        r._push_per_layer_overwrite_bits(layer, ((1 << 8193) - 1) ^ 1)
    exact = r._common_per_layer_reusable_locs

    def unexpected():
        raise AssertionError("pure free bitsets need only a bounded capacity witness")

    r._common_per_layer_reusable_locs = unexpected
    assert r._has_common_per_layer_reusable_tokens(128)
    assert not r._has_common_per_layer_reusable_tokens(8193)
    r._common_per_layer_reusable_locs = exact
    # Live/protected and unreserved addresses must not become capacity credit.
    r._per_layer_arena_allocated_locs[3] = set(range(1, 8180))
    r._per_layer_arena_protected_locs[7] = {8180, 8181}
    r._per_layer_arena_reserved_locs.remove(8182)
    for count in [0, 1, 10, 11, 128]:
        assert r._has_common_per_layer_reusable_tokens(count) == (len(exact()) >= count)
    # The generic representation remains supported through exact fallback.
    r._per_layer_arena_free_locs[3] = [8192]
    for count in [1, 10, 11]:
        assert r._has_common_per_layer_reusable_tokens(count) == (len(exact()) >= count)


def test_headroom_predicate_matches_exact_across_fragmented_ledgers():
    from sglang.srt.layerkv.config_stats import LayerKVConfig
    from sglang.srt.layerkv.runtime import LayerKVRuntime

    rng = random.Random(17)
    for case in range(30):
        r = LayerKVRuntime(LayerKVConfig())
        layers = list(range(case % 4))
        r._kvc_layer_ids = lambda: layers
        r._per_layer_arena_reserved_locs = set(rng.sample(range(64), 50))
        for layer in layers:
            r._push_per_layer_overwrite_bits(
                layer, r._locs_to_bitset(rng.sample(range(64), 45))
            )
            r._per_layer_arena_allocated_locs[layer] = set(rng.sample(range(64), 3))
            r._per_layer_arena_protected_locs[layer] = set(rng.sample(range(64), 3))
            if case % 3 == 0:
                r._per_layer_arena_free_locs[layer] = rng.sample(range(64), 8)
            if case % 5 == 0:
                r._per_layer_arena_overwrite_pending_locs[layer] = rng.sample(
                    range(64), 8
                )
        exact_count = len(r._common_per_layer_reusable_locs())
        for count in [0, 1, 8, 32, 64, exact_count, exact_count + 1]:
            assert r._has_common_per_layer_reusable_tokens(count) == (
                exact_count >= count
            )


@pytest.mark.parametrize("representation", ["ordinary", "pending", "mixed"])
def test_bounded_witness_supports_real_list_backed_headroom(representation):
    from sglang.srt.layerkv.config_stats import LayerKVConfig
    from sglang.srt.layerkv.runtime import LayerKVRuntime

    r = LayerKVRuntime(LayerKVConfig())
    r._kvc_layer_ids = lambda: list(range(3, 40, 4))
    r._per_layer_arena_reserved_locs = set(range(8192, 16384))
    for layer in r._kvc_layer_ids():
        locs = list(range(8192, 11192))
        if representation == "pending":
            r._per_layer_arena_overwrite_pending_locs[layer] = locs
        else:
            r._per_layer_arena_free_locs[layer] = locs
        if representation == "mixed":
            r._push_per_layer_overwrite_bits(layer, ((1 << 1000) - 1) << 12000)

    def unexpected():
        raise AssertionError("128 common list entries prove sufficient headroom")

    r._common_per_layer_reusable_locs = unexpected
    assert r._has_common_per_layer_reusable_tokens(128)
