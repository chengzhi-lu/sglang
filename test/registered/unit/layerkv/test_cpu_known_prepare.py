"""Shared group demand reuse must preserve generic materialization semantics."""

from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.common_types import _LayerKVExpertLayerState
from sglang.srt.layerkv.runtime import LayerKVConfig, LayerKVRuntime
from sglang.srt.layerkv.shared_expert import SharedExpertController

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
TopK = namedtuple("TopK", "topk_ids topk_weights")
Dispatch = namedtuple("Dispatch", "hidden_states topk_output")
Output = namedtuple("Output", "hidden_states")


def setup(path="cpu-known", dtype=torch.bfloat16, profile=False):
    r = LayerKVRuntime(
        LayerKVConfig(
            shared_expert_initial_slots=2,
            shared_expert_prepare_path=path,
            shared_expert_profile_chunks=profile,
            shared_expert_profile_prepare=profile,
            expert_transfer_backend="cuda-batch",
            expert_batch_backing_layout="individual",
            expert_demand_d2h_wait="stream",
        )
    )
    r._expert_plan_applied = True
    r._current_forward_mode = "prefill"
    r._expert_h2d_stream = torch.cuda.Stream()
    weights = torch.tensor([[5.0, 5.0], [6.0, 6.0]], dtype=dtype, device="cuda")
    state = _LayerKVExpertLayerState(
        layer_id=0,
        module=SimpleNamespace(weight=weights),
        orig_forward=None,
        orig_run_moe_core=None,
        full_num_experts=6,
        slot_capacity=2,
        expert_bytes=weights[0].nbytes,
        device=weights.device,
        dtype=dtype,
        cpu_params={
            i: {"weight": torch.full((2,), i + 1, dtype=dtype, pin_memory=True)}
            for i in range(4)
        },
        param_names=["weight"],
        logical_to_slot={4: 0, 5: 1},
        slot_to_logical={0: 4, 1: 5},
        lru={4: 0, 5: 0},
        hotness_prefill={0: 7},
        hotness_decode={1: 9},
        lru_heap=[(0, 0, 4), (0, 1, 5)],
        remap_tensor=torch.tensor([-1, -1, -1, -1, 0, 1], device="cuda"),
        topk_ids_in_range_calibrated=True,
    )
    r._expert_layers[0] = state
    r._expert_host_backing_bytes = 4 * state.expert_bytes
    controller = SharedExpertController(r, None)
    controller.state = state
    r._shared_expert = controller
    return r, state, controller


def dispatch(ids, dtype=torch.bfloat16):
    ids = torch.tensor(ids, device="cuda")
    return Dispatch(
        torch.arange(ids.shape[0] * 2, device="cuda", dtype=dtype).reshape(-1, 2),
        TopK(ids, torch.full(ids.shape, 0.5, device="cuda", dtype=dtype)),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("order", ["input", "reuse"])
@pytest.mark.parametrize("profile", [False, True])
def test_group_fastpath_matches_generic(monkeypatch, dtype, order, profile):
    results = []
    for path in ("generic", "cpu-known"):
        r, state, controller = setup(path, dtype, profile)
        r.config.shared_expert_chunk_order = order
        loads = []
        original = r._materialize_experts

        def materialize(state, ids, **kwargs):
            loads.append(list(ids))
            return original(state, ids, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError("CPU-known group repeated GPU demand discovery")

        monkeypatch.setattr(r, "_materialize_experts", materialize)
        if path == "cpu-known":
            monkeypatch.setattr(r, "_try_gpu_expert_topk_remap", forbidden)
            monkeypatch.setattr(r, "_unique_expert_ids_and_record_hotness", forbidden)

        def core(chunk):
            topk = chunk.topk_output
            factor = (state.module.weight[topk.topk_ids, 0] * topk.topk_weights).sum(1)
            return Output(chunk.hidden_states.mul_(factor[:, None]))

        state.orig_run_moe_core = core
        value = dispatch([[1, 0], [3, 2], [0, 1], [5, 4], [2, 3]], dtype)
        saved = value.hidden_states.clone()
        result = controller.run_token_chunks(state, value)
        expected = (
            saved * ((value.topk_output.topk_ids + 1).to(dtype) * 0.5).sum(1)[:, None]
        )
        assert torch.equal(result.hidden_states, expected)
        assert torch.equal(saved, value.hidden_states)
        r._finalize_expert_materialize_events(block=True)
        assert r.stats.expert_guard_pass
        assert r.stats.expert_cpu_known_prepare_count == (
            3 if path == "cpu-known" else 0
        )
        if path == "cpu-known":
            assert r.stats.expert_cpu_known_remap_guard_count == 1
            assert r.stats.expert_cpu_known_remap_guard_elided_count == 2
        assert r.stats.expert_cpu_known_prepare_fallback_count == 0
        assert state.hotness_prefill == {0: 7} and state.hotness_decode == {1: 9}
        results.append(
            (
                result.hidden_states.cpu(),
                loads,
                dict(state.logical_to_slot),
                dict(state.lru),
                r.stats.expert_materialize_count,
            )
        )
        if profile:
            summary = controller._prepare_profiler.summary()
            assert (
                sum(summary[b]["calls"] for b in ("first_shape", "repeat_shape")) == 3
            )
            for bucket in ("first_shape", "repeat_shape"):
                assert (
                    sum(
                        row["calls"] - row["recursive_calls"]
                        for row in summary[bucket]["functions"]
                        if row["name"] == "_prepare_expert_dispatch_for_core"
                    )
                    == summary[bucket]["calls"]
                )
    assert torch.equal(results[0][0], results[1][0])
    assert results[0][1:] == results[1][1:]


@pytest.mark.parametrize(
    "case", ["plan", "decode", "uncalibrated", "invalid", "map", "capacity", "ids"]
)
def test_ineligible_cpu_known_demand_falls_back(monkeypatch, case):
    r, state, _ = setup()
    ids = [0, 1]
    if case == "plan":
        r._expert_plan_applied = False
    if case == "decode":
        r._current_forward_mode = "decode"
    if case == "uncalibrated":
        state.topk_ids_in_range_calibrated = False
    if case == "invalid":
        state.topk_ids_invalid_observed = True
    if case == "map":
        state.slot_to_logical[0] = 3
    if case == "capacity":
        state.slot_capacity = state.full_num_experts
    if case == "ids":
        ids = [1, 0]
    value = dispatch([[0, 1]])
    calls = []
    monkeypatch.setattr(
        r,
        "_prepare_expert_layer_for_topk",
        lambda state, topk: calls.append(topk) or topk,
    )
    assert (
        r._prepare_expert_dispatch_for_core(state, value, known_logical_ids=ids)
        is value
    )
    assert calls == [value.topk_output]
    assert r.stats.expert_cpu_known_prepare_fallback_count == 1


def test_decode_chunk_snapshot_can_use_cpu_known_prepare(monkeypatch):
    r, state, controller = setup()
    r._current_forward_mode = "decode"
    # CPU-known decode preparation is intentionally exercised only when the
    # routed window has opted into GPU grouping.  Small decode windows skip
    # grouping, but should still use the generic GPU remap path.
    r.config.shared_expert_gpu_grouping = True
    r.config.shared_expert_gpu_grouping_min_rows = 1
    r._last_forward_batch = SimpleNamespace(batch_size=2)
    r.config.shared_expert_chunk_order = "input"
    calls = []

    def materialize(state, ids, **kwargs):
        calls.append(list(ids))
        return original(state, ids, **kwargs)

    original = r._materialize_experts
    monkeypatch.setattr(r, "_materialize_experts", materialize)
    monkeypatch.setattr(
        r,
        "_try_gpu_expert_topk_remap",
        lambda *_args, **_kwargs: pytest.fail(
            "decode used generic GPU demand discovery"
        ),
    )
    monkeypatch.setattr(
        r,
        "_unique_expert_ids_and_record_hotness",
        lambda *_args, **_kwargs: pytest.fail("decode rediscovered routing IDs"),
    )

    def core(value):
        return Output(value.hidden_states.clone())

    state.orig_run_moe_core = core
    value = dispatch([[1, 0], [3, 2], [0, 1], [5, 4]])
    result = controller.run_token_chunks(state, value)

    assert result.hidden_states.shape == value.hidden_states.shape
    assert calls
    assert r.stats.expert_cpu_known_prepare_count == 3
    assert r.stats.expert_cpu_known_prepare_fallback_count == 0


def test_small_decode_grouping_skip_reuses_cpu_route_snapshot(monkeypatch):
    r, state, controller = setup()
    r._current_forward_mode = "decode"
    r.config.shared_expert_chunk_order = "input"
    monkeypatch.setattr(
        r,
        "_try_gpu_expert_topk_remap",
        lambda *_args, **_kwargs: pytest.fail(
            "decode rediscovered demand through generic GPU remap"
        ),
    )
    monkeypatch.setattr(
        r,
        "_unique_expert_ids_and_record_hotness",
        lambda *_args, **_kwargs: pytest.fail("decode rediscovered routing IDs"),
    )
    state.orig_run_moe_core = lambda value: Output(value.hidden_states.clone())

    controller.run_token_chunks(
        state, dispatch([[1, 0], [3, 2], [0, 1], [5, 4]])
    )
    r._finalize_expert_materialize_events(block=True)

    assert r.stats.expert_cpu_known_prepare_count == controller.token_chunk_calls
    assert r.stats.expert_cpu_known_prepare_fallback_count == 0
    assert controller.token_chunk_cpu_known_decode_policy_skips == 0


def test_invalid_routing_does_not_pass_truncated_group_demand():
    r, state, controller = setup()
    calls = []

    def core(value):
        calls.append(value.topk_output.topk_ids.tolist())
        return Output(value.hidden_states)

    state.orig_run_moe_core = core
    controller.run_token_chunks(state, dispatch([[4, -1], [5, 6]]))
    assert calls == [[[0, -1], [1, 6]]]
    assert r.stats.expert_cpu_known_prepare_fallback_count == 1
    assert r.stats.expert_cpu_known_prepare_count == 0
    assert state.topk_ids_invalid_observed
    assert r.stats.expert_topk_range_invalid_count == 1


def test_all_hit_waits_without_materializing_or_touching_lru(monkeypatch):
    r, state, _ = setup()
    waits = []
    monkeypatch.setattr(
        r,
        "_materialize_experts",
        lambda *a, **kw: pytest.fail("all-hit loaded experts"),
    )
    monkeypatch.setattr(
        r, "_wait_for_expert_logical_ids_ready", lambda state, ids: waits.append(ids)
    )
    result = r._prepare_expert_dispatch_for_core(
        state, dispatch([[5, 4]]), known_logical_ids=[4, 5]
    )
    assert result.topk_output.topk_ids.tolist() == [[1, 0]]
    assert waits == [[4, 5]] and state.lru == {4: 0, 5: 0}


@pytest.mark.parametrize("bad_slot", [-1, 2])
def test_gpu_guard_still_rejects_bad_mapping(bad_slot):
    r, state, _ = setup()
    state.remap_tensor[4] = bad_slot
    with pytest.raises(RuntimeError, match="invalid expert remap"):
        r._prepare_expert_dispatch_for_core(
            state, dispatch([[4, 5]]), known_logical_ids=[4, 5]
        )
    assert not r.stats.expert_guard_pass and not r.stats.comparable


def test_partial_hit_keeps_sorted_all_expert_load_order(monkeypatch):
    results = []
    for path in ("generic", "cpu-known"):
        r, state, _ = setup(path)
        calls = []
        materialize = r._materialize_experts

        def load(state, ids, **kwargs):
            calls.append(list(ids))
            return materialize(state, ids, **kwargs)

        monkeypatch.setattr(r, "_materialize_experts", load)
        kwargs = {"known_logical_ids": [0, 4]} if path == "cpu-known" else {}
        value = r._prepare_expert_dispatch_for_core(state, dispatch([[4, 0]]), **kwargs)
        r._finalize_expert_materialize_events(block=True)
        assert calls == [[0, 4]]
        assert state.logical_to_slot[4] == 0
        results.append(
            (
                value.topk_output.topk_ids.tolist(),
                dict(state.lru),
                dict(state.logical_to_slot),
                r.stats.expert_materialize_count,
            )
        )
    assert results[0] == results[1]


def test_preplan_fallback_retains_hotness_collection(monkeypatch):
    r, state, _ = setup()
    r._expert_plan_applied = False
    calls = []
    monkeypatch.setattr(
        r,
        "_record_expert_hotness_for_layer",
        lambda layer, full, ids: calls.append(ids.tolist()),
    )
    r._prepare_expert_dispatch_for_core(
        state, dispatch([[0, 1]]), known_logical_ids=[0, 1]
    )
    r._finalize_expert_materialize_events(block=True)
    assert calls == [[[0, 1]]]
    assert r.stats.expert_cpu_known_prepare_fallback_count == 1
    assert r.stats.expert_cpu_known_prepare_count == 0
