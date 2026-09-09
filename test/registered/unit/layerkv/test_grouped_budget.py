from types import SimpleNamespace

import pytest

from sglang.srt.layerkv.expert_hooks import LayerKVExpertHooksMixin
from sglang.srt.layerkv.residency_budget import (
    ResidencyBudget,
    _group_token_experts_indexed,
    group_token_experts,
)


def test_group_protection_exposes_benefit_hidden_by_unique_ids():
    # Both capacities are smaller than the forward's union; a unique-ID stream
    # misses every expert at both sizes. Whole-group protection retains a useful
    # member across groups at capacity3, whereas capacity2 must replace both.
    rows = [[0, 1], [2, 3]]
    legacy = ResidencyBudget(2, 3, interval=2)
    grouped = ResidencyBudget(2, 3, interval=2)
    for step in [1, 2]:
        legacy.observe(step, [0, 1, 2, 3])
        grouped.observe_rows(step, rows, 4)
    assert legacy.misses == [8, 8]
    assert grouped.misses[0] == 8
    assert grouped.misses[1] < grouped.misses[0]
    assert grouped.grouped_samples == 2
    assert grouped.grouped_misses == grouped.misses


def test_grouped_observation_is_once_per_step_and_pressure_still_wins():
    policy = ResidencyBudget(2, 3, interval=1)
    for step in [1, 2]:
        policy.observe_rows(step, [[0, 1], [2, 3]], 4)
    counts = list(policy.misses)
    policy.observe_rows(2, [[4, 5]], 6)
    assert policy.misses == counts
    assert policy.grouped_samples == 2
    assert policy.decide(2, batch_size=2, free_tokens=100, waiting=0, shortage=0) == 3
    assert policy.decide(3, batch_size=2, free_tokens=100, waiting=1, shortage=1) == 2
    assert policy.reason == "kv-pressure"


def test_precomputed_group_masks_match_row_observation():
    rows = [[0, 1], [2, 3], [0, 1], [1, 2]]
    row_policy = ResidencyBudget(2, 3, interval=4)
    grouped_policy = ResidencyBudget(2, 3, interval=4)
    base_groups = group_token_experts(rows, 2, 4)[0]
    max_groups = group_token_experts(rows, 3, 4)[0]

    row_policy.observe_rows(1, rows, 4)
    grouped_policy.observe_grouped(1, base_groups, max_groups)

    assert grouped_policy.misses == row_policy.misses
    assert grouped_policy.grouped_misses == row_policy.grouped_misses
    assert list(grouped_policy.caches[0]) == list(row_policy.caches[0])
    assert list(grouped_policy.caches[1]) == list(row_policy.caches[1])


def test_first_fit_preserves_rows_and_filters_invalid_ids():
    groups, valid = group_token_experts([[0, 1], [2, 3], [0, 1], [-1, 8]], 2, 4)
    assert groups == [[3, [0, 2, 3]], [12, [1]]]
    assert not valid


def test_indexed_first_fit_handles_grown_groups_without_reordering():
    routes = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 8],
        [0, 1, 2, 3, 4, 5, 6, 9],
        [0, 1, 2, 3, 4, 5, 6, 8],
        [0, 1, 2, 3, 4, 5, 6, 10],
    ]
    bit_rows = [sum(1 << expert for expert in row) for row in routes]
    expected = [
        [sum(1 << expert for expert in range(9)), [0, 1, 3]],
        [sum(1 << expert for expert in [0, 1, 2, 3, 4, 5, 6, 9, 10]), [2, 4]],
    ]
    assert _group_token_experts_indexed(bit_rows, 9) == expected
    assert group_token_experts(routes, 9, 32)[0] == expected


@pytest.mark.parametrize("adaptive", [False, True])
def test_chunk_check_reads_decode_rows_once_and_keeps_snapshot(adaptive):
    calls = []
    rows = [[0, 1], [2, 3]]

    def read_rows():
        calls.append(1)
        return rows

    tensor = SimpleNamespace(tolist=read_rows)
    state = SimpleNamespace(full_num_experts=4, slot_capacity=2)
    shared = SimpleNamespace(
        state=state, budget=ResidencyBudget(2, 3) if adaptive else None
    )
    runtime = SimpleNamespace(
        _shared_expert=shared,
        _current_forward_mode="decode",
        _decode_step=1,
    )
    assert LayerKVExpertHooksMixin._expert_chunked_core_required(
        runtime, state, SimpleNamespace(topk_ids=tensor)
    )
    assert calls == [1]
    assert shared._routing_snapshot[0] is tensor
    assert shared._routing_snapshot[1] is rows
    state.slot_capacity = 4
    assert not LayerKVExpertHooksMixin._expert_chunked_core_required(
        runtime, state, SimpleNamespace(topk_ids=tensor)
    )
    assert shared._routing_snapshot is None
    assert calls == [1, 1]
    if adaptive:
        assert shared.budget.grouped_samples == 1
