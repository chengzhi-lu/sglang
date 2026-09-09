"""Bounded cross-layer KVC and expert prefetch scheduling."""

from types import SimpleNamespace

import torch

from sglang.srt.layerkv.common_types import (
    _LayerKVRecoveryTask,
    _LayerKVResidencyKey,
    _LayerKVResidentTensorGroup,
)
from sglang.srt.layerkv.config_stats import LayerKVConfig
from sglang.srt.layerkv.runtime import (
    LayerKVRuntime,
    _LayerKVExpertResidencyBackend,
)


def test_expert_prefetch_budget_is_bounded_by_configured_lookahead():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            runtime_profile="optimized",
            expert_prefetch_lookahead_layers=3,
        )
    )
    tasks = [
        _LayerKVRecoveryTask(
            kind="expert",
            layer_id=layer_id,
            bytes=1,
            deadline_layer=layer_id,
        )
        for layer_id in range(5)
    ]

    _kvc_budget, expert_budget = runtime._scheduler_copy_task_budgets(tasks)

    assert expert_budget == 3


def test_expert_residency_backend_forwards_prefetch_stream():
    runtime = LayerKVRuntime(
        LayerKVConfig(enabled=True, mode="kvc-expert", runtime_profile="optimized")
    )
    state = SimpleNamespace(layer_id=7)
    runtime._expert_layers = {7: state}
    backend = _LayerKVExpertResidencyBackend(runtime)
    group = _LayerKVResidentTensorGroup(
        key=_LayerKVResidencyKey(kind="expert", layer_id=7, logical_id=3),
        state="offloaded",
        bytes=1,
    )
    captured = []
    runtime._materialize_experts = lambda actual_state, ids, **kwargs: captured.append(
        (actual_state, list(ids), kwargs)
    )
    stream = object()

    backend.recover([group], stream=stream)

    assert captured == [(state, [3], {"reason": "prefetch", "transfer_stream": stream})]


def test_scheduler_expert_prefetch_uses_dedicated_h2d_stream():
    runtime = LayerKVRuntime(
        LayerKVConfig(enabled=True, mode="kvc-expert", runtime_profile="optimized")
    )
    shared_stream = object()
    expert_stream = object()
    runtime._copy_stream = shared_stream
    runtime._expert_h2d_stream = expert_stream
    state = SimpleNamespace(layer_id=7, device=torch.device("cpu"))
    runtime._expert_layers = {7: state}
    captured = []
    runtime._materialize_experts = lambda actual_state, ids, **kwargs: captured.append(
        (actual_state, list(ids), kwargs)
    )

    runtime._schedule_recovery_tasks(
        [
            _LayerKVRecoveryTask(
                kind="expert",
                layer_id=7,
                bytes=1,
                deadline_layer=7,
                logical_ids=(3,),
            )
        ]
    )

    assert captured == [
        (
            state,
            [3],
            {"reason": "prefetch", "transfer_stream": expert_stream},
        )
    ]


def test_expert_prefetch_gate_uses_context_and_actual_batch():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            runtime_profile="optimized",
            shared_expert_headroom_steps=16,
        )
    )
    runtime._expert_layers = {0: SimpleNamespace(slot_capacity=16)}

    long_small = SimpleNamespace(batch_size=2, out_cache_loc=[])
    runtime._current_forward_req_lens_batch_id = id(long_small)
    runtime._current_forward_req_lens = [(0, 4096), (1, 4096)]
    # With no outstanding KVC copy, the cross-layer path has an idle window.
    assert runtime._expert_prefetch_allowed(long_small) is True
    assert runtime.stats.expert_prefetch_context_skip_count == 0
    assert (
        runtime._expert_prefetch_allowed(
            long_small, allow_bounded_context=True
        )
        is True
    )
    assert (
        runtime.stats.expert_prefetch_last_gate
        == "allow-bounded-context-kvc-window"
    )
    assert runtime._expert_prefetch_id_budget(runtime._expert_layers[0], long_small) == 4

    runtime._pending_kvc_reload_events = [SimpleNamespace(waited_on_main_stream=False)] * 2
    assert (
        runtime._expert_prefetch_allowed(
            long_small, allow_bounded_context=True
        )
        is False
    )
    assert runtime.stats.expert_prefetch_context_skip_count == 1
    assert (
        runtime.stats.expert_prefetch_last_gate
        == "kv-priority-kvc-window-full"
    )

    short_large = SimpleNamespace(batch_size=32, out_cache_loc=[])
    runtime._current_forward_req_lens_batch_id = id(short_large)
    runtime._current_forward_req_lens = [(index, 190) for index in range(32)]
    assert runtime._expert_prefetch_allowed(short_large) is True
    assert runtime.stats.expert_prefetch_last_gate == "allow"
    assert runtime._expert_prefetch_id_budget(runtime._expert_layers[0], short_large) == 16


def test_route_prefetch_budget_scales_with_recent_overlap():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            runtime_profile="optimized",
        )
    )
    state = SimpleNamespace(slot_capacity=16, last_decode_route_overlap=0.25)
    batch = SimpleNamespace(batch_size=8, out_cache_loc=[])

    assert (
        runtime._expert_prefetch_id_budget(state, batch, route_based=True) == 4
    )
    state.last_decode_route_overlap = 0.75
    assert (
        runtime._expert_prefetch_id_budget(state, batch, route_based=True) == 12
    )
    # Hotness/cross-layer callers can retain the existing capacity budget.
    assert runtime._expert_prefetch_id_budget(state, batch) == 16


def test_route_prefetch_admission_uses_configured_overlap_threshold():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            runtime_profile="optimized",
            expert_prefetch_min_route_overlap=0.5,
        )
    )
    state = SimpleNamespace(
        last_decode_request_signature=((0, 0, 128),),
        last_decode_route_overlap=0.49,
    )
    batch = SimpleNamespace(batch_size=1)
    runtime._current_forward_expert_request_signature = ((0, 0, 128),)

    assert runtime._expert_prefetch_route_compatible(state, batch) is False
    assert runtime.stats.expert_prefetch_route_unstable_skip_count == 1

    state.last_decode_route_overlap = 0.5
    assert runtime._expert_prefetch_route_compatible(state, batch) is True


def test_long_context_prefetch_limits_previous_route_ids():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            runtime_profile="optimized",
        )
    )
    state = SimpleNamespace(
        layer_id=0,
        slot_capacity=16,
        full_num_experts=32,
        expert_bytes=1024,
        last_decode_logical_ids=list(range(10)),
        prefetched_logical_ids=set(),
        logical_to_slot={},
        cpu_params={index: {} for index in range(10)},
        hotness_decode={},
        hotness_prefill={},
    )
    runtime._expert_plan_applied = True
    runtime._expert_layers = {0: state}
    runtime._build_kvc_recovery_tasks = lambda *_args, **_kwargs: []
    batch = SimpleNamespace(batch_size=2, out_cache_loc=[])
    runtime._current_forward_req_lens_batch_id = id(batch)
    runtime._current_forward_req_lens = [(0, 4096), (1, 4096)]

    tasks = runtime._build_recovery_tasks(batch)

    assert len(tasks) == 1
    assert tasks[0].logical_ids == (0, 1, 2, 3)
    assert runtime.stats.expert_prefetch_skipped_capacity_count == 6


def test_cross_layer_expert_prefetch_targets_a_later_expert_layer():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            runtime_profile="optimized",
            expert_prefetch_lookahead_layers=2,
        )
    )
    def state(layer_id, ids):
        return SimpleNamespace(
            layer_id=layer_id,
            slot_capacity=4,
            full_num_experts=16,
            expert_bytes=1024,
            last_decode_logical_ids=list(ids),
            last_decode_batch_size=2,
            last_decode_request_signature=((0, 0, 16), (1, 0, 16)),
            last_decode_route_overlap=0.5,
            prefetched_logical_ids=set(),
            logical_to_slot={},
            cpu_params={index: {} for index in ids},
            hotness_decode={},
            hotness_prefill={},
        )

    runtime._expert_plan_applied = True
    runtime._current_forward_mode = "decode"
    runtime._decode_step = 3
    runtime._copy_stream = object()
    batch = SimpleNamespace(batch_size=2, out_cache_loc=[])
    runtime._last_forward_batch = batch
    runtime._current_forward_req_lens_batch_id = id(batch)
    runtime._current_forward_req_lens = [(0, 190), (1, 190)]
    runtime._current_forward_expert_request_signature = (
        (0, 0, 16),
        (1, 0, 16),
    )
    runtime._expert_layers = {0: state(0, []), 20: state(20, [4, 5, 6, 7, 8])}
    captured = []
    runtime._schedule_recovery_tasks = captured.extend

    runtime._maybe_issue_expert_prefetch_after_layer(3)
    runtime._maybe_issue_expert_prefetch_after_layer(7)

    assert len(captured) == 1
    assert captured[0].layer_id == 20
    assert captured[0].logical_ids == (4, 5, 6, 7)
    assert runtime.stats.expert_prefetch_cross_layer_issue_count == 1
    assert runtime.stats.expert_prefetch_cross_layer_layer_count == 1


def test_cross_layer_prefetch_runs_without_active_kvc_control():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            runtime_profile="optimized",
        )
    )
    calls = []
    runtime._per_layer_kvc_io_control_active = lambda: False
    runtime._maybe_issue_expert_prefetch_after_layer = calls.append
    wrapped = runtime._wrap_get_key_buffer(lambda layer_id: ("key", layer_id))

    assert wrapped(3) == ("key", 3)
    assert calls == [3]


def test_cross_layer_prefetch_falls_back_to_online_hotness_for_large_batch():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            runtime_profile="optimized",
            expert_prefetch_lookahead_layers=1,
        )
    )
    state = SimpleNamespace(
        layer_id=20,
        slot_capacity=4,
        full_num_experts=16,
        expert_bytes=1024,
        last_decode_logical_ids=[1, 2],
        last_decode_batch_size=4,
        last_decode_request_signature=((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)),
        last_decode_route_overlap=0.1,
        prefetched_logical_ids=set(),
        logical_to_slot={},
        cpu_params={index: {} for index in (8, 9, 10, 11)},
        hotness_decode={},
        hotness_prefill={},
    )
    runtime._expert_plan_applied = True
    runtime._current_forward_mode = "decode"
    runtime._decode_step = 3
    runtime._copy_stream = object()
    batch = SimpleNamespace(batch_size=4, out_cache_loc=[])
    runtime._last_forward_batch = batch
    runtime._current_forward_req_lens_batch_id = id(batch)
    runtime._current_forward_req_lens = [(index, 190) for index in range(4)]
    runtime._current_forward_expert_request_signature = (
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
    )
    runtime._expert_candidate_order_by_layer = {20: [8, 9, 10, 11]}
    runtime._expert_layers = {20: state}
    captured = []
    runtime._schedule_recovery_tasks = captured.extend

    runtime._maybe_issue_expert_prefetch_after_layer(3)

    assert len(captured) == 1
    assert captured[0].logical_ids == (8, 9, 10, 11)
    assert runtime.stats.expert_prefetch_route_fallback_count == 1


def test_cross_layer_expert_prefetch_yields_to_long_context_kvc():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            runtime_profile="optimized",
            expert_prefetch_lookahead_layers=1,
        )
    )
    state = SimpleNamespace(
        layer_id=20,
        slot_capacity=4,
        full_num_experts=16,
        expert_bytes=1024,
        last_decode_logical_ids=[4, 5, 6, 7],
        last_decode_batch_size=2,
        last_decode_request_signature=((0, 0, 16), (1, 0, 16)),
        last_decode_route_overlap=1.0,
        prefetched_logical_ids=set(),
        logical_to_slot={},
        cpu_params={index: {} for index in range(4, 8)},
        hotness_decode={},
        hotness_prefill={},
    )
    runtime._expert_plan_applied = True
    runtime._current_forward_mode = "decode"
    runtime._decode_step = 3
    runtime._copy_stream = object()
    batch = SimpleNamespace(batch_size=2, out_cache_loc=[])
    runtime._last_forward_batch = batch
    runtime._current_forward_req_lens_batch_id = id(batch)
    runtime._current_forward_req_lens = [(0, 4096), (1, 4096)]
    runtime._current_forward_expert_request_signature = (
        (0, 0, 16),
        (1, 0, 16),
    )
    runtime._expert_layers = {20: state}
    runtime._pending_kvc_reload_events = [
        SimpleNamespace(waited_on_main_stream=False),
        SimpleNamespace(waited_on_main_stream=False),
    ]
    captured = []
    runtime._schedule_recovery_tasks = captured.extend

    runtime._maybe_issue_expert_prefetch_after_layer(3)

    assert captured == []
    assert runtime.stats.expert_prefetch_context_skip_count == 1
    assert runtime.stats.expert_prefetch_cross_layer_skip_count == 1


def test_expert_prefetch_window_yields_to_existing_kvc_offload():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            runtime_profile="optimized",
        )
    )
    runtime._per_layer_offloaded_token_count_fast = 1

    assert runtime._expert_prefetch_kvc_window_available() is False


def test_cross_layer_expert_prefetch_skips_request_transition():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            runtime_profile="optimized",
            expert_prefetch_lookahead_layers=1,
        )
    )
    state = SimpleNamespace(
        layer_id=20,
        slot_capacity=4,
        full_num_experts=16,
        expert_bytes=1024,
        last_decode_logical_ids=[4, 5, 6, 7],
        last_decode_batch_size=2,
        last_decode_request_signature=((0, 0, 1), (1, 0, 1)),
        last_decode_route_overlap=1.0,
        prefetched_logical_ids=set(),
        logical_to_slot={},
        cpu_params={index: {} for index in range(4, 8)},
        hotness_decode={},
        hotness_prefill={},
    )
    runtime._expert_plan_applied = True
    runtime._current_forward_mode = "decode"
    runtime._decode_step = 3
    runtime._copy_stream = object()
    batch = SimpleNamespace(batch_size=2, out_cache_loc=[])
    runtime._last_forward_batch = batch
    runtime._current_forward_expert_request_signature = (
        (0, 0, 16),
        (1, 0, 16),
    )
    runtime._expert_layers = {20: state}
    captured = []
    runtime._schedule_recovery_tasks = captured.extend

    runtime._maybe_issue_expert_prefetch_after_layer(3)

    assert captured == []
    assert runtime.stats.expert_prefetch_route_mismatch_skip_count == 1
    assert runtime.stats.expert_prefetch_cross_layer_skip_count == 1


def test_physical_kvc_prefetch_submits_only_next_layer():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-only",
            kvc_backend="per-layer-arena",
            kvc_scheduler="async-deadline",
            runtime_profile="optimized",
        )
    )
    runtime._copy_stream = object()
    runtime._last_forward_batch = object()
    runtime._per_layer_virtual_scratch_enabled = lambda: False
    runtime._next_kvc_layer_id = lambda layer_id: int(layer_id) + 1
    entries = [
        SimpleNamespace(layer_id=1, token_count=4),
        SimpleNamespace(layer_id=2, token_count=8),
    ]
    runtime._select_required_offloaded_per_layer_entries = lambda _batch: entries
    submitted = {}

    def reload(_batch, *, selected_entries, strict):
        submitted["entries"] = list(selected_entries)
        submitted["strict"] = strict
        return True

    runtime._reload_required_kvc = reload

    runtime._issue_next_per_layer_kvc_prefetch(0)

    assert submitted["entries"] == [entries[0]]
    assert submitted["strict"] is False
    assert runtime.stats.kvc_layer_prefetch_issue_count == 1
    assert runtime.stats.kvc_layer_prefetch_layer_count == 1
    assert runtime.stats.kvc_layer_prefetch_token_count == 4


def test_physical_kvc_prefetch_can_run_from_virtual_scratch_overflow():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-only",
            kvc_backend="per-layer-arena",
            kvc_scheduler="async-deadline",
            runtime_profile="optimized",
        )
    )
    runtime._copy_stream = object()
    runtime._last_forward_batch = object()
    runtime._per_layer_virtual_scratch_enabled = lambda: True
    runtime._next_kvc_layer_id = lambda layer_id: int(layer_id) + 1
    entries = [SimpleNamespace(layer_id=1, token_count=4)]
    runtime._select_required_offloaded_per_layer_entries = lambda _batch: entries
    submitted = {}

    def reload(_batch, *, selected_entries, strict):
        submitted["entries"] = list(selected_entries)
        submitted["strict"] = strict
        return True

    runtime._reload_required_kvc = reload

    runtime._issue_next_per_layer_kvc_prefetch(0, allow_virtual_scratch=True)

    assert submitted["entries"] == entries
    assert submitted["strict"] is False
    assert runtime.stats.kvc_layer_prefetch_issue_count == 1


def test_physical_kvc_prefetch_fallback_caps_unconsumed_copy_window():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-only",
            kvc_backend="per-layer-arena",
            kvc_scheduler="async-deadline",
            runtime_profile="optimized",
        )
    )
    runtime._copy_stream = object()
    runtime._last_forward_batch = object()
    runtime._per_layer_virtual_scratch_enabled = lambda: True
    runtime._next_kvc_layer_id = lambda layer_id: int(layer_id) + 1
    entries = [SimpleNamespace(layer_id=1, token_count=4)]
    runtime._select_required_offloaded_per_layer_entries = lambda _batch: entries
    runtime._pending_kvc_reload_events = [
        SimpleNamespace(waited_on_main_stream=False),
        SimpleNamespace(waited_on_main_stream=False),
    ]
    submitted = []
    runtime._reload_required_kvc = lambda *_args, **_kwargs: submitted.append(1)

    runtime._issue_next_per_layer_kvc_prefetch(0, allow_virtual_scratch=True)

    assert submitted == []
    assert runtime.stats.kvc_layer_prefetch_pending_cap_skip_count == 1
    assert runtime.stats.kvc_layer_prefetch_skip_count == 1


def test_physical_kvc_prefetch_fallback_uses_measured_deadline():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-only",
            kvc_backend="per-layer-arena",
            kvc_scheduler="async-deadline",
            runtime_profile="optimized",
        )
    )
    runtime._copy_stream = object()
    runtime._last_forward_batch = object()
    runtime._per_layer_virtual_scratch_enabled = lambda: True
    runtime._next_kvc_layer_id = lambda layer_id: int(layer_id) + 1
    entries = [SimpleNamespace(layer_id=1, token_count=4)]
    runtime._select_required_offloaded_per_layer_entries = lambda _batch: entries
    runtime._kvc_reload_ms_per_mb_ewma_by_layer = {1: 1.0}
    runtime._bytes_per_kvc_token_per_layer = lambda: 1024 * 1024
    runtime._estimate_profiled_kvc_layer_window_ms = lambda *_args: 0.1
    submitted = []
    runtime._reload_required_kvc = lambda *_args, **_kwargs: submitted.append(1)

    runtime._issue_next_per_layer_kvc_prefetch(0, allow_virtual_scratch=True)

    assert submitted == []
    assert runtime._kvc_prefetch_window_measurement_enabled is True
    assert runtime.stats.kvc_layer_prefetch_deadline_reject_count == 1
    assert runtime.stats.kvc_layer_prefetch_skip_count == 1


def test_physical_kvc_prefetch_window_uses_lightweight_samples():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-only",
            kvc_backend="per-layer-arena",
            kvc_scheduler="async-deadline",
            runtime_profile="optimized",
        )
    )
    runtime._kvc_layer_ids = lambda: [0, 1]
    runtime.stats.kvc_layer_prefetch_model_forward_ms = 100.0
    runtime.stats.kvc_layer_prefetch_model_forward_count = 2

    assert runtime._estimate_profiled_kvc_layer_window_ms(0, 1) == 25.0


def test_virtual_scratch_sync_materialize_falls_back_to_physical_next_layer():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-only",
            kvc_backend="per-layer-arena",
            kvc_scheduler="async-deadline",
            runtime_profile="optimized",
        )
    )
    runtime._per_layer_residency[(0, 0, 0)] = object()
    runtime._last_forward_batch = object()
    runtime._runner = object()
    runtime._virtual_scratch_locs = torch.zeros(8, dtype=torch.int64)
    demand = SimpleNamespace(
        entries=(SimpleNamespace(layer_id=0, token_count=4),),
        token_count=4,
        signature="demand",
        layer_id=0,
    )
    runtime._get_virtual_kvc_demand = lambda _layer_id: demand
    runtime._lookup_virtual_scratch_cache = lambda _demand: None
    runtime._materialize_virtual_kvc_sync = lambda _demand: torch.zeros(
        4, dtype=torch.int64
    )
    runtime._store_virtual_scratch_cache = lambda *_args, **_kwargs: None
    runtime._rewrite_virtual_kvc_attention = lambda *_args, **_kwargs: None
    runtime._issue_next_virtual_kvc_prefetch = lambda *_args, **_kwargs: None
    captured = []
    runtime._issue_next_per_layer_kvc_prefetch = (
        lambda layer_id, **kwargs: captured.append((layer_id, kwargs))
    )

    runtime._prepare_virtual_kvc_attention(0)

    assert captured == [(0, {"allow_virtual_scratch": True})]
