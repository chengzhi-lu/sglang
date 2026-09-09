"""Structural upper bounds cannot authorize overwriting an in-use expert."""

import copy

import pytest
import torch

from sglang.srt.layerkv.prefetch_probe import bit_ids, capacity_snapshot


def snapshot(current=(0, 1), following=(1, 4), backed=(0, 1, 2, 3, 4)):
    mapping = {i: i for i in range(4)}
    before = dict(mapping)
    result = capacity_snapshot(
        4, current, following, mapping, dict(mapping), set(backed), 64
    )
    assert mapping == before
    return result


def test_full_group_has_no_prefetch_slot():
    r = snapshot(current=(0, 1, 2, 3), following=(4, 5))
    assert r["unused_slot_ids"] == []
    assert r["capacity_upper_experts"] == r["backed_upper_bytes"] == 0


def test_preserve_next_residents_and_bound_missing_sources():
    r = snapshot(following=(2, 4, 5), backed=(0, 1, 3, 4))
    assert r["unused_slot_ids"] == [2, 3]
    assert r["candidate_slot_ids"] == r["no_d2h_candidate_slot_ids"] == [3]
    assert r["next_missing_ids"] == [4, 5]
    assert r["next_missing_cpu_backed_ids"] == [4]
    assert r["backed_upper_experts"] == 1


def test_victim_needs_backup_and_missing_source_not_admissible():
    r = snapshot(following=(4, 5), backed=(0, 1, 4, 5))
    assert r["capacity_upper_experts"] == 2
    assert r["no_d2h_candidate_slot_ids"] == []
    assert r["backed_upper_experts"] == 0
    r = snapshot(following=(4, 5), backed=(0, 1, 2, 3))
    assert r["next_missing_cpu_backed_ids"] == []
    assert r["backed_upper_experts"] == 0


def test_free_slot_and_terminal_group():
    r = capacity_snapshot(3, [0], [2], {0: 1}, {1: 0}, {2}, 32)
    assert r["no_d2h_candidate_slot_ids"] == [0, 2]
    assert r["backed_upper_bytes"] == 32
    r = snapshot(following=None)
    assert not r["has_next"] and "backed_upper_bytes" not in r
    assert bit_ids((1 << 200) | 3) == [0, 1, 200]


@pytest.mark.parametrize(
    "mapping,reverse,current",
    [
        ({0: 0, 1: 0}, {0: 1}, [0]),
        ({0: 0}, {0: 1}, [0]),
        ({0: 4}, {4: 0}, [0]),
        ({0: 0}, {0: 0}, [1]),
    ],
)
def test_bad_mapping_rejected(mapping, reverse, current):
    with pytest.raises(ValueError, match="residency"):
        capacity_snapshot(4, current, [2], mapping, reverse, {2}, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_observation_preserves_real_chunk_execution(dtype):
    from test_cpu_known_prepare import Output, dispatch, setup

    outcomes = []
    for trace in (False, True):
        r, state, controller = setup(dtype=dtype)
        r.config.expert_remap_update = "batch"
        r.config.shared_expert_trace_waits = trace

        def core(chunk):
            factor = (
                state.module.weight[chunk.topk_output.topk_ids, 0]
                * chunk.topk_output.topk_weights
            ).sum(1)
            return Output(chunk.hidden_states.mul_(factor[:, None]))

        state.orig_run_moe_core = core
        value = dispatch([[0, 1], [2, 3], [4, 5]], dtype)
        output = controller.run_token_chunks(state, value)
        r._finalize_expert_materialize_events(block=True)
        outcomes.append(
            (
                output.hidden_states.cpu(),
                copy.deepcopy(state.logical_to_slot),
                copy.deepcopy(state.lru),
                r.stats.expert_materialize_count,
                r.stats.expert_cuda_batch_h2d_count,
                r.stats.expert_cuda_batch_d2h_count,
            )
        )
        if trace:
            probe = controller._prefetch_probe
            assert probe["batch"] == controller.token_chunk_batches
            assert len(probe["groups"]) == controller.token_chunk_calls == 3
            assert all(g["backed_upper_experts"] == 0 for g in probe["groups"][:-1])
            assert not probe["groups"][-1]["has_next"]
        else:
            assert controller._prefetch_probe is None
    assert torch.equal(outcomes[0][0], outcomes[1][0])
    assert outcomes[0][1:] == outcomes[1][1:]
