"""Shared expert budgets must survive a later multi-token prefill."""

from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.runtime import LayerKVConfig, LayerKVRuntime


def shared_runtime():
    runtime = LayerKVRuntime(
        LayerKVConfig(enabled=True, mode="kvc-expert", policy="expert-first")
    )
    state = SimpleNamespace(
        layer_id=0,
        full_num_experts=8,
        slot_capacity=2,
        module=SimpleNamespace(top_k=2),
        remap_tensor=None,
    )
    runtime._shared_expert = SimpleNamespace(state=state)
    return runtime, state


def test_context_demand_uses_live_request_lengths_and_current_writes():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, _state = shared_runtime()
    batch = SimpleNamespace(out_cache_loc=[0, 1, 2])
    runtime._current_forward_req_lens_batch_id = id(batch)
    runtime._current_forward_req_lens = [(4, 10), (7, 20)]
    runtime._allocator_total_size = lambda: 100
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.runtime = runtime

    assert controller._current_kv_residency_demand(batch) == (33, 100)
    assert controller._current_kv_residency_demand(SimpleNamespace()) == (None, None)


def test_context_demand_projects_one_queued_request():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, _state = shared_runtime()
    batch = SimpleNamespace(out_cache_loc=[0, 1, 2])
    runtime._current_forward_req_lens_batch_id = id(batch)
    runtime._current_forward_req_lens = [(4, 10), (7, 20)]
    runtime._allocator_total_size = lambda: 100
    runtime._last_scheduler_waiting_context_tokens = 70
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.runtime = runtime

    assert controller._projected_kv_residency_demand(batch) == (103, 100, 33, 70)


def test_context_capacity_excludes_scratch_and_lent_expert_locations():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, _state = shared_runtime()
    batch = SimpleNamespace(out_cache_loc=[0, 1, 2])
    runtime._current_forward_req_lens_batch_id = id(batch)
    runtime._current_forward_req_lens = [(4, 10), (7, 20)]
    runtime._allocator_total_size = lambda: 100
    runtime._virtual_scratch_locs = torch.tensor([90, 91], dtype=torch.int64)
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.runtime = runtime
    controller.blocked = {0: {80, 81, 82}}

    assert controller._current_kv_residency_demand(batch) == (33, 95)


def test_context_demand_reuses_blocked_location_union():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, _state = shared_runtime()
    batch = SimpleNamespace(out_cache_loc=[0, 1, 2])
    runtime._current_forward_req_lens_batch_id = id(batch)
    runtime._current_forward_req_lens = [(4, 10)]
    runtime._allocator_total_size = lambda: 100
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.runtime = runtime
    controller.blocked = {0: {80, 81}, 1: {81, 82}}

    assert controller._current_kv_residency_demand(batch) == (13, 97)
    cached = controller._blocked_kv_location_union
    assert cached == {80, 81, 82}
    assert controller._current_kv_residency_demand(batch) == (13, 97)
    assert controller._blocked_kv_location_union is cached


def test_shared_install_uses_online_prefill_residency(monkeypatch):
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, _state = shared_runtime()
    runtime.config.shared_expert_layer = 0
    runtime.config.shared_expert_initial_slots = 2
    runtime.config.shared_expert_extra_slots = 1
    runtime._expert_modules = []
    runtime.physical_expert_supported = True
    runtime._expert_layers = {}
    runtime._expert_hotness_prefill[0] = {3: 7, 1: 5}
    module = SimpleNamespace(
        w13_weight=torch.empty((4, 1), dtype=torch.float16), top_k=1
    )
    runtime._expert_modules = [(0, module)]
    runtime._expert_param_names = lambda _module: []
    runtime._ensure_expert_hotness_cpu_view = lambda **kwargs: kwargs
    monkeypatch.setattr(
        runtime,
        "_select_initial_resident_experts",
        lambda **_kwargs: [3, 1],
    )
    captured = {}
    installed = SimpleNamespace(param_names=[])

    def install(_module, _layer, _capacity, **kwargs):
        captured.update(kwargs)
        return installed

    runtime._install_expert_layer_slots = install
    runtime._refresh_expert_stats = lambda: None
    controller = SharedExpertController(runtime, None)
    controller.ensure_installed()

    assert captured["initial_resident"] == [3, 1]
    assert captured["async_copy"] is True
    assert controller.initial_resident_source == "online_hotness"
    assert controller.initial_resident_expert_ids == [3, 1]


def test_zero_extra_slots_is_fixed_capacity_control():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    runtime.config.shared_expert_initial_slots = 2
    runtime.config.shared_expert_extra_slots = 0
    controller = SharedExpertController(runtime, None)
    controller.state = state
    # No arena is needed: the fixed arm must return before donor scanning.
    controller.after_decode()
    assert controller.grow_count == 0 and state.slot_capacity == 2


def test_optimized_shared_expert_uses_aggregate_residency_tracking():
    runtime, _state = shared_runtime()
    runtime.config.kvc_backend = "per-layer-arena"
    runtime.config.shared_expert_layer = 0

    assert runtime._expert_group_tracking_enabled() is False

    runtime.config.debug_stats = True
    assert runtime._expert_group_tracking_enabled() is True


def test_all_layer_shared_manager_routes_state_to_its_controller():
    from sglang.srt.layerkv.shared_expert import SharedExpertManager

    runtime, _state = shared_runtime()
    arena = SimpleNamespace()
    manager = SharedExpertManager(runtime, arena, [7, 3, 7])
    state3 = SimpleNamespace(layer_id=3)
    state7 = SimpleNamespace(layer_id=7)

    manager.controllers[3].state = state3
    manager.controllers[7].state = state7

    assert sorted(manager.controllers) == [3, 7]
    assert manager.controller_for_state(state3) is manager.controllers[3]
    assert manager.controller_for_state(state7) is manager.controllers[7]
    assert manager.state is None


def test_all_layer_shared_manager_installs_each_discovered_layer(monkeypatch):
    from sglang.srt.layerkv.shared_expert import SharedExpertManager

    runtime, _state = shared_runtime()
    runtime.config.shared_expert_initial_slots = 2
    runtime.config.shared_expert_extra_slots = 1
    runtime._expert_layers = {}
    runtime._expert_modules = [
        (0, SimpleNamespace(w13_weight=torch.empty((4, 1)), top_k=1)),
        (1, SimpleNamespace(w13_weight=torch.empty((4, 1)), top_k=1)),
    ]
    runtime.physical_expert_supported = True
    runtime._expert_param_names = lambda _module: []
    runtime._ensure_expert_hotness_cpu_view = lambda **kwargs: kwargs
    runtime._refresh_expert_stats = lambda: None
    monkeypatch.setattr(
        runtime,
        "_select_initial_resident_experts",
        lambda **_kwargs: [0, 1],
    )

    def install(module, layer_id, _capacity, **_kwargs):
        return SimpleNamespace(layer_id=layer_id, param_names=[])

    runtime._install_expert_layer_slots = install
    manager = SharedExpertManager(runtime, SimpleNamespace(), [0, 1])
    manager.ensure_installed()

    assert sorted(runtime._expert_layers) == [0, 1]
    assert (
        manager.controller_for_state(runtime._expert_layers[0])
        is manager.controllers[0]
    )
    assert (
        manager.controller_for_state(runtime._expert_layers[1])
        is manager.controllers[1]
    )


def test_all_layer_manager_spends_extra_slot_on_hotter_layer():
    from sglang.srt.layerkv.shared_expert import SharedExpertManager

    runtime, _state = shared_runtime()
    manager = SharedExpertManager(runtime, SimpleNamespace(), [0, 1])
    calls = []
    for layer_id, hotness in ((0, 1), (1, 10)):
        controller = manager.controllers[layer_id]
        controller.state = SimpleNamespace(
            layer_id=layer_id,
            slot_capacity=controller.base_slots,
            hotness_decode={0: hotness},
            hotness_prefill={},
        )

        def after_decode(
            *, allow_growth=True, force_recall=False, _controller=controller
        ):
            calls.append(
                (_controller.layer_id, allow_growth, force_recall)
            )
            if allow_growth:
                _controller.state.slot_capacity += 1

        controller.after_decode = after_decode

    manager.after_decode()

    assert calls == [(1, True, False), (0, False, False)]
    assert manager.controllers[1].state.slot_capacity == 17
    assert manager.controllers[0].state.slot_capacity == 16
    assert manager.last_global_plan["grown_layers"] == [1]


def test_all_layer_manager_recalls_and_blocks_growth_under_scheduler_pressure():
    from sglang.srt.layerkv.shared_expert import SharedExpertManager

    runtime, _state = shared_runtime()
    runtime._scheduler_pressure_tokens = 1
    manager = SharedExpertManager(runtime, SimpleNamespace(), [0, 1])
    calls = []
    for layer_id in (0, 1):
        controller = manager.controllers[layer_id]
        controller.state = SimpleNamespace(
            layer_id=layer_id,
            slot_capacity=controller.base_slots,
            hotness_decode={0: 1},
            hotness_prefill={},
        )
        controller.after_decode = (
            lambda *, allow_growth=True, force_recall=False, _layer=layer_id: calls.append(
                (_layer, allow_growth, force_recall)
            )
        )

    manager.after_decode()

    assert calls == [(0, False, True), (1, False, True)]
    assert manager.pressure_gate_count == 1
    assert manager.last_global_plan["reason"] == "scheduler-pressure"


def empty_donor_controller():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    runtime.config.shared_expert_initial_slots = 2
    runtime.config.shared_expert_extra_slots = 1
    weight = torch.zeros((2, 1))
    state.module.weight = weight
    state.param_names = ["weight"]
    arena = SimpleNamespace(page_bytes=4, loans=[])
    controller = SharedExpertController(runtime, arena)
    controller.state = state
    controller.expert_allocations[weight.data_ptr()] = object()
    runtime._shared_expert = controller
    return controller


def test_free_donor_miss_cache_uses_availability_generation():
    controller = empty_donor_controller()
    controller.runtime.config.shared_expert_free_kv_donors = True
    for _ in range(8):
        controller.after_decode()
    assert controller.donor_scan_count == 1
    assert controller.donor_cache_hit_count == 7

    controller.runtime._per_layer_donor_availability_version += 1
    controller.after_decode()
    assert controller.donor_scan_count == 2


def test_unchanged_donor_miss_does_not_rescan_each_decode(monkeypatch):
    controller = empty_donor_controller()
    scans = []
    original = controller._donors

    def scan():
        scans.append(True)
        return original()

    monkeypatch.setattr(controller, "_donors", scan)
    for _ in range(8):
        controller.after_decode()
    assert len(scans) == 1
    assert controller.state.slot_capacity == 2


def test_donor_miss_cache_skips_repeated_arena_probe(monkeypatch):
    controller = empty_donor_controller()
    controller.runtime.config.shared_expert_free_kv_donors = True
    probes = []

    def probe(count):
        probes.append(int(count))
        return False

    monkeypatch.setattr(
        controller.runtime, "_ensure_per_layer_physical_arena", probe
    )
    controller.after_decode()
    controller.after_decode()
    assert probes == [1]


def test_uncached_donor_control_keeps_scanning():
    controller = empty_donor_controller()
    controller.runtime.config.shared_expert_disable_donor_cache = True
    for _ in range(8):
        controller.after_decode()
    assert controller.donor_scan_count == controller.donor_miss_count == 8
    assert controller.donor_cache_hit_count == 0


@pytest.mark.parametrize("event", ["free", "offload", "cleanup"])
def test_donor_miss_invalidated_when_eligibility_can_increase(event):
    from sglang.srt.layerkv.common_types import _LayerKVPerLayerReqCleanupState

    controller = empty_donor_controller()
    r = controller.runtime
    r.config.kvc_backend = "per-layer-arena"
    for _ in range(2):
        controller.after_decode()
    assert controller.donor_scan_count == 1
    if event == "free":
        r._push_per_layer_overwrite_bits(3, r._locs_to_bitset([1]))
    elif event == "offload":
        r._track_per_layer_offloaded_key((3, 0, 0), token_count=1)
    else:
        state = _LayerKVPerLayerReqCleanupState()
        state.loc_bits_by_layer[3] = r._locs_to_bitset([1])
        state.loc_counts_by_layer[3] = 1
        r._per_layer_cleanup_state_by_req[0] = state
        r._remove_per_layer_cleanup_loc_bits(0, 3, r._locs_to_bitset([1]))
    for _ in range(2):
        controller.after_decode()
    assert controller.donor_scan_count == 2


def test_donor_cache_observes_normal_lifecycle_offload_completion(monkeypatch):
    controller = empty_donor_controller()
    r = controller.runtime
    controller.after_decode()
    completed = []

    def finalize(*, block):
        assert block
        if not completed:
            r._push_per_layer_overwrite_bits(3, r._locs_to_bitset([1]))
            completed.append(True)

    monkeypatch.setattr(r, "_finalize_kvc_evictions", finalize)
    controller.after_decode()
    assert not completed and controller.donor_scan_count == 1
    # Completion belongs to the normal KV lifecycle, not donor discovery.
    r._finalize_kvc_evictions(block=True)
    controller.after_decode()
    assert completed and controller.donor_scan_count == 2


def test_donor_cache_option_reaches_runtime_config():
    config = LayerKVConfig.from_server_args(
        SimpleNamespace(layerkv_shared_expert_disable_donor_cache=True)
    )
    assert config.shared_expert_disable_donor_cache


def test_adaptive_chunk_order_reaches_runtime_config():
    config = LayerKVConfig.from_server_args(
        SimpleNamespace(layerkv_shared_expert_chunk_order="adaptive")
    )
    assert config.shared_expert_chunk_order == "adaptive"


def test_prefill_unique_experts_chunk_within_shared_budget():
    runtime, state = shared_runtime()
    # Each token fits, but the complete prefill batch needs four experts.
    topk = SimpleNamespace(topk_ids=torch.tensor([[0, 1], [2, 3]]))
    assert runtime._expert_chunked_core_required(state, topk)
    topk.topk_ids = torch.tensor([[0, 1], [1, 0]])
    assert not runtime._expert_chunked_core_required(state, topk)


def test_adaptive_chunk_order_uses_context_and_actual_batch():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    runtime.config.shared_expert_chunk_order = "adaptive"
    runtime._last_forward_batch = SimpleNamespace(batch_size=1)
    runtime._avg_prefix_len = lambda _batch: 4096.0
    controller = SharedExpertController(runtime, None)
    controller.state = state

    # Long context and B=1 use resident reuse when capacity is tight.
    assert controller._effective_chunk_order(state, 1) == "reuse"
    # Short context with a request batch larger than the resident capacity also
    # uses reuse.  The token count may be much larger during prefill, but must
    # not be mistaken for the request batch.
    runtime._avg_prefix_len = lambda _batch: 128.0
    assert controller._effective_chunk_order(state, 1024) == "input"
    runtime._last_forward_batch.batch_size = 3
    assert controller._effective_chunk_order(state, 1024) == "reuse"
    # A small short-context batch keeps the low-overhead input order.
    runtime._last_forward_batch.batch_size = 1
    assert controller._effective_chunk_order(state, 1024) == "input"
    assert controller.token_chunk_adaptive_reuse_count == 2
    assert controller.token_chunk_adaptive_input_count == 2

    # A short-context B8 forward with top-k=4 has a wide routed working set
    # even though B8 is below a 49-slot physical expert capacity.
    state.slot_capacity = 49
    state.module.top_k = 4
    runtime._last_forward_batch.batch_size = 8
    assert controller._effective_chunk_order(state, 1024) == "reuse"
    runtime._last_forward_batch.batch_size = 2
    assert controller._effective_chunk_order(state, 1024) == "input"
    assert controller.token_chunk_adaptive_reuse_count == 3
    assert controller.token_chunk_adaptive_input_count == 3


def test_adaptive_chunk_order_uses_window_reuse_for_low_capacity_long_context_batch():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    runtime.config.shared_expert_chunk_order = "adaptive"
    runtime._last_forward_batch = SimpleNamespace(batch_size=4)
    runtime._avg_prefix_len = lambda _batch: 4096.0
    state.slot_capacity = 9
    state.module.top_k = 8
    controller = SharedExpertController(runtime, None)
    controller.state = state

    assert controller._effective_chunk_order(state, 4096) == "window-reuse"
    assert controller.token_chunk_adaptive_window_reuse_count == 1
    assert controller.token_chunk_adaptive_reuse_count == 0

    # At the two-route-width boundary, keep bounded reuse; only larger capacity
    # returns to the full reuse policy.
    state.slot_capacity = 16
    assert controller._effective_chunk_order(state, 4096) == "window-reuse"
    state.slot_capacity = 17
    assert controller._effective_chunk_order(state, 4096) == "reuse"
    assert controller.token_chunk_adaptive_reuse_count == 1
    assert controller.token_chunk_adaptive_window_reuse_count == 2


def test_adaptive_chunk_order_uses_current_prefill_tokens_as_context():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    runtime.config.shared_expert_chunk_order = "adaptive"
    runtime._last_forward_batch = SimpleNamespace(batch_size=4)
    # A fresh prefill has no cached prefix, but its current tokens still form
    # a long context and must not select the unbounded reuse scan.
    runtime._avg_prefix_len = lambda _batch: 0.0
    state.slot_capacity = 9
    state.module.top_k = 8
    controller = SharedExpertController(runtime, None)
    controller.state = state

    assert controller._effective_chunk_order(state, 4 * 4900) == "window-reuse"
    assert controller.token_chunk_adaptive_window_reuse_count == 1


def test_gpu_grouping_policy_skips_long_small_low_capacity_routes():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    runtime.config.shared_expert_gpu_grouping = True
    runtime._last_forward_batch = SimpleNamespace(batch_size=3)
    runtime._avg_prefix_len = lambda _batch: 4900.0
    state.slot_capacity = 9
    state.module.top_k = 8
    controller = SharedExpertController(runtime, None)
    controller.state = state
    fake_cuda_ids = SimpleNamespace(device=SimpleNamespace(type="cuda"))

    assert not controller._gpu_grouping_enabled(fake_cuda_ids)
    assert controller.token_chunk_gpu_group_policy_skips == 1

    runtime._last_forward_batch.batch_size = 8
    runtime._avg_prefix_len = lambda _batch: 128.0
    state.slot_capacity = 49
    assert controller._gpu_grouping_enabled(fake_cuda_ids)


def test_gpu_grouping_policy_skips_small_routed_batches():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    runtime.config.shared_expert_gpu_grouping = True
    runtime._last_forward_batch = SimpleNamespace(batch_size=8)
    runtime._avg_prefix_len = lambda _batch: 128.0
    state.slot_capacity = 16
    state.module.top_k = 8
    controller = SharedExpertController(runtime, None)
    controller.state = state
    fake_cuda_ids = SimpleNamespace(
        device=SimpleNamespace(type="cuda"), shape=(8, 8)
    )

    assert not controller._gpu_grouping_enabled(fake_cuda_ids)
    assert controller.token_chunk_gpu_group_policy_skips == 1


def test_adaptive_chunk_order_keeps_single_request_prefill_reuse():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    runtime.config.shared_expert_chunk_order = "adaptive"
    runtime._last_forward_batch = SimpleNamespace(batch_size=1)
    runtime._avg_prefix_len = lambda _batch: 0.0
    state.slot_capacity = 9
    state.module.top_k = 8
    controller = SharedExpertController(runtime, None)
    controller.state = state

    # A single long prefill keeps full reuse because it has a different
    # residency/transfer tradeoff from a multi-request window.
    assert controller._effective_chunk_order(state, 4900) == "reuse"
    runtime._avg_prefix_len = lambda _batch: 4096.0
    assert controller._effective_chunk_order(state, 1) == "reuse"
    assert controller.token_chunk_adaptive_window_reuse_count == 0
    assert controller.token_chunk_adaptive_reuse_count == 2


def test_window_reuse_does_not_scan_past_bounded_lookahead():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    state.slot_capacity = 3
    state.logical_to_slot = {0: 0, 1: 1, 2: 2}
    state.slot_to_logical = {0: 0, 1: 1, 2: 2}
    controller = SharedExpertController(runtime, None)
    controller.state = state
    controller.route_reuse_window = 2
    groups = [
        [(1 << 3) | (1 << 4) | (1 << 5), [0]],
        [(1 << 0) | (1 << 1) | (1 << 3), [1]],
        [(1 << 0) | (1 << 1) | (1 << 2), [2]],
    ]

    first = next(controller._ordered_token_groups(state, groups, "window-reuse"))
    assert first == groups[1]


def test_reuse_order_preserves_dynamic_residency_tie_breaks():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    state.slot_capacity = 3
    state.logical_to_slot = {0: 0, 1: 1, 2: 2}
    state.slot_to_logical = {0: 0, 1: 1, 2: 2}
    controller = SharedExpertController(runtime, None)
    controller.state = state
    groups = [
        [(1 << 0) | (1 << 3) | (1 << 4), [0]],
        [(1 << 0) | (1 << 1) | (1 << 3), [1]],
        [(1 << 1) | (1 << 2) | (1 << 3), [2]],
    ]

    iterator = controller._ordered_token_groups(state, groups, "reuse")
    assert next(iterator) == groups[1]
    # Simulate the previous group installing expert 3 and evicting expert 2.
    state.logical_to_slot = {0: 0, 1: 1, 3: 2}
    state.slot_to_logical = {0: 0, 1: 1, 2: 3}
    assert next(iterator) == groups[0]
    assert next(iterator) == groups[2]


def test_shared_slots_cannot_fall_back_to_cuda_allocation():
    runtime, state = shared_runtime()
    with pytest.raises(RuntimeError, match="KV pages"):
        runtime._grow_expert_layer_slots(state, 4)
    assert state.slot_capacity == 2


def test_recovery_scheduler_builds_missing_expert_demand():
    runtime, state = shared_runtime()
    runtime._expert_plan_applied = True
    runtime._expert_layers = {0: state}
    runtime._build_kvc_recovery_tasks = lambda *_a, **_kw: []
    state.prefetched_logical_ids = set()
    state.last_decode_logical_ids = [1]
    state.logical_to_slot = {}
    state.cpu_params = {1: {}}
    state.expert_bytes = 1024
    state.hotness_decode = {}
    state.hotness_prefill = {}
    tasks = runtime._build_recovery_tasks(None)
    assert len(tasks) == 1
    assert tasks[0].expert_demand.logical_ids == (1,)
    assert tasks[0].bytes == 1024


def test_recovery_prefetch_fits_physical_slot_capacity():
    runtime, state = shared_runtime()
    runtime._expert_plan_applied = True
    runtime._expert_layers = {0: state}
    runtime._build_kvc_recovery_tasks = lambda *_a, **_kw: []
    state.prefetched_logical_ids = set()
    state.last_decode_logical_ids = [1, 2, 3]
    state.logical_to_slot = {}
    state.cpu_params = {i: {} for i in (1, 2, 3)}
    state.expert_bytes = 1024
    state.hotness_decode = {}
    state.hotness_prefill = {}
    tasks = runtime._build_recovery_tasks(None)
    assert len(tasks) == 1
    task = tasks[0]
    assert len(task.logical_ids) <= state.slot_capacity
    assert task.expert_demand.logical_ids == task.logical_ids
    assert task.bytes == len(task.logical_ids) * state.expert_bytes
    # Materialization protects the entire task. Exercise its actual slot chooser
    # without GPU copies: each selected slot is occupied before the next choice.
    state.free_slots = list(range(state.slot_capacity))
    state.slot_to_logical = {}
    state.lru_heap = []
    state.lru = {}
    for logical in task.logical_ids:
        slot = runtime._choose_expert_slot_for_materialize(state, set(task.logical_ids))
        state.slot_to_logical[slot] = logical
        state.logical_to_slot[logical] = slot


def test_chunked_core_preserves_inputs_with_inplace_runner():
    runtime, state = shared_runtime()
    runtime._shared_expert.record_use = lambda *_: None
    runtime._unique_expert_ids_and_record_hotness = lambda *_a, **_kw: [0, 1, 2, 3]
    state.remap_tensor = torch.full((8,), -1, dtype=torch.long)

    def materialize(_state, chunk, **_kw):
        state.remap_tensor.fill_(-1)
        for slot, logical in enumerate(chunk):
            state.remap_tensor[logical] = slot

    runtime._materialize_experts = materialize
    runtime._wait_for_expert_logical_ids_ready = lambda *_: None
    topk_type = namedtuple("TopK", "topk_ids topk_weights")
    dispatch_type = namedtuple("Dispatch", "hidden_states topk_output")
    output_type = namedtuple("Output", "hidden_states")

    def inplace_core(dispatch):
        # Triton MoE can overwrite dispatch.hidden_states. Each expert's
        # contribution here is simply input * routing weight.
        dispatch.hidden_states.mul_(dispatch.topk_output.topk_weights.sum(1)[:, None])
        return output_type(dispatch.hidden_states)

    state.orig_run_moe_core = inplace_core
    inputs = torch.tensor([[2.0], [3.0]])
    dispatch = dispatch_type(
        inputs,
        topk_type(torch.tensor([[0, 2], [1, 3]]), torch.full((2, 2), 0.5)),
    )
    out = runtime._run_expert_core_chunked(state, dispatch)
    torch.testing.assert_close(out.hidden_states, torch.tensor([[2.0], [3.0]]))


def test_exact_next_input_group_prefetch_protects_current_slots():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    state.device = torch.device("cuda")
    state.slot_capacity = 4
    state.logical_to_slot = {0: 0, 1: 1}
    runtime._expert_h2d_async_enabled = lambda _state: True
    runtime._expert_prefetch_has_backing = lambda _state, _logical: True
    captured = {}

    def materialize(_state, logical_ids, **kwargs):
        captured["logical_ids"] = logical_ids
        captured["protected"] = kwargs["protected_logical_ids"]

    runtime._materialize_experts = materialize
    controller = SharedExpertController(runtime, None)
    controller._prefetch_next_input_group(
        state,
        current_bits=1 << 0,
        next_bits=(1 << 1) | (1 << 2) | (1 << 3),
    )

    assert captured == {"logical_ids": [2, 3], "protected": {0}}
    assert runtime.stats.expert_prefetch_exact_group_count == 1
    assert runtime.stats.expert_prefetch_exact_id_count == 2


def test_bounded_successor_window_preserves_nearest_group_order():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    state.device = torch.device("cuda")
    state.slot_capacity = 5
    state.logical_to_slot = {0: 0}
    runtime.config.shared_expert_prefetch_groups = 2
    runtime._expert_h2d_async_enabled = lambda _state: True
    runtime._expert_prefetch_has_backing = lambda _state, _logical: True
    captured = {}

    def materialize(_state, logical_ids, **kwargs):
        captured["logical_ids"] = logical_ids
        captured["protected"] = kwargs["protected_logical_ids"]

    runtime._materialize_experts = materialize
    controller = SharedExpertController(runtime, None)
    controller._prefetch_next_input_group(
        state,
        current_bits=1 << 0,
        next_bits=1 << 1,
        prefetch_logical_ids=[1, 2, 3, 4],
        prefetch_group_count=2,
    )

    assert captured == {"logical_ids": [1, 2, 3, 4], "protected": {0}}
    assert runtime.stats.expert_prefetch_lookahead_group_count == 1
    assert runtime.stats.expert_prefetch_lookahead_id_count == 4


def test_exact_next_input_group_prefetch_skips_when_union_exceeds_capacity():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    state.device = torch.device("cuda")
    state.slot_capacity = 2
    runtime._expert_h2d_async_enabled = lambda _state: True
    runtime._materialize_experts = lambda *_args, **_kwargs: pytest.fail(
        "prefetch evicted the current chunk"
    )
    controller = SharedExpertController(runtime, None)

    controller._prefetch_next_input_group(
        state,
        current_bits=(1 << 0) | (1 << 1),
        next_bits=1 << 2,
    )

    assert runtime.stats.expert_prefetch_exact_skip_capacity_count == 1


def test_exact_next_input_group_prefetch_respects_batch_context_gate():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    state.device = torch.device("cuda")
    state.slot_capacity = 49
    runtime._last_forward_batch = SimpleNamespace(batch_size=8)
    runtime._avg_prefix_len = lambda _batch: 128.0
    runtime._expert_prefetch_is_short_context_large_batch = lambda _batch: False
    runtime._expert_h2d_async_enabled = lambda _state: True
    runtime._materialize_experts = lambda *_args, **_kwargs: pytest.fail(
        "small short-context batch must not issue exact prefetch"
    )
    controller = SharedExpertController(runtime, None)

    controller._prefetch_next_input_group(
        state,
        current_bits=1 << 0,
        next_bits=1 << 1,
    )

    assert runtime.stats.expert_prefetch_exact_policy_skip_count == 1


def test_short_context_half_capacity_batch_admits_exact_prefetch_gate():
    runtime, state = shared_runtime()
    state.slot_capacity = 4
    runtime._expert_layers = {0: state}
    runtime._avg_prefix_len = lambda _batch: 128.0
    batch = SimpleNamespace(batch_size=2)

    assert runtime._expert_prefetch_is_short_context_large_batch(batch) is True
    batch.batch_size = 1
    assert runtime._expert_prefetch_is_short_context_large_batch(batch) is False


def test_short_context_wide_routed_batch_admits_prefetch_gate():
    runtime, state = shared_runtime()
    state.slot_capacity = 49
    state.module.top_k = 8
    runtime._expert_layers = {0: state}
    runtime._avg_prefix_len = lambda _batch: 128.0
    batch = SimpleNamespace(batch_size=8)

    assert runtime._expert_prefetch_is_short_context_large_batch(batch) is True
    batch.batch_size = 1
    assert runtime._expert_prefetch_is_short_context_large_batch(batch) is False


def test_exact_next_input_group_prefetch_uses_idle_kvc_window_for_decode():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    state.device = torch.device("cuda")
    state.slot_capacity = 4
    state.logical_to_slot = {}
    runtime._current_forward_mode = "decode"
    runtime._last_forward_batch = SimpleNamespace(batch_size=2)
    runtime._expert_prefetch_is_short_context_large_batch = lambda _batch: False
    runtime._expert_prefetch_is_long_context_small_batch = lambda _batch: True
    runtime._expert_prefetch_kvc_window_available = lambda: True
    runtime._expert_h2d_async_enabled = lambda _state: True
    runtime._expert_prefetch_has_backing = lambda _state, _logical: True
    captured = {}

    def materialize(_state, logical_ids, **kwargs):
        captured["logical_ids"] = logical_ids
        captured["protected"] = kwargs["protected_logical_ids"]

    runtime._materialize_experts = materialize
    controller = SharedExpertController(runtime, None)
    controller._prefetch_next_input_group(
        state,
        current_bits=1 << 0,
        next_bits=(1 << 1) | (1 << 2),
    )

    assert captured == {"logical_ids": [1, 2], "protected": {0}}
    assert runtime.stats.expert_prefetch_exact_group_count == 1
    assert runtime.stats.expert_prefetch_exact_policy_skip_count == 0
    assert runtime.stats.expert_prefetch_last_gate == "exact-long-small-kvc-window"


def test_post_moe_prefetch_protects_future_resident_experts():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    state.device = torch.device("cuda")
    state.slot_capacity = 4
    state.logical_to_slot = {0: 0, 1: 1, 2: 2, 3: 3}
    runtime._current_forward_mode = "decode"
    runtime._expert_h2d_async_enabled = lambda _state: True
    runtime._expert_prefetch_has_backing = lambda _state, _logical: True
    captured = {}

    def materialize(_state, logical_ids, **kwargs):
        captured["logical_ids"] = logical_ids
        captured["protected"] = kwargs["protected_logical_ids"]

    runtime._materialize_experts = materialize
    controller = SharedExpertController(runtime, None)
    controller._prefetch_next_input_group(
        state,
        current_bits=0,
        next_bits=1 << 4,
        extra_protected_ids={2},
        post_moe_dead_only=True,
    )

    assert captured == {"logical_ids": [4], "protected": {2}}
    assert controller.token_chunk_post_moe_prefetch_count == 1
    assert runtime.stats.expert_prefetch_post_moe_count == 1
    assert runtime.stats.expert_prefetch_post_moe_id_count == 1


def test_post_moe_prefetch_skips_when_capacity_has_headroom():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    controller = SharedExpertController(runtime, None)
    topk_ids = torch.zeros((2, 8), dtype=torch.long)

    state.full_num_experts = 64
    state.slot_capacity = 49
    assert controller._post_moe_prefetch_is_worthwhile(state, topk_ids) is False
    assert runtime.stats.expert_prefetch_post_moe_capacity_skip_count == 1

    state.slot_capacity = 16
    assert controller._post_moe_prefetch_is_worthwhile(state, topk_ids) is True


def test_adaptive_exact_prefetch_is_before_current_moe():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    runtime.config.shared_expert_chunk_order = "reuse"
    state.slot_capacity = 2
    state.full_num_experts = 8
    state.module.top_k = 2
    state.logical_to_slot = {}
    timeline = []

    controller = SharedExpertController(runtime, None)
    controller.state = state
    runtime._shared_expert = controller

    def prepare(_state, dispatch):
        timeline.append("prepare")
        needed = dispatch.topk_output.topk_ids.unique().tolist()
        state.logical_to_slot = {
            int(logical): slot for slot, logical in enumerate(needed)
        }
        return dispatch

    def exact_prefetch(_state, _current_bits, _next_bits, **kwargs):
        timeline.append(("prefetch", kwargs.get("producer_event") is not None))

    def core(dispatch):
        timeline.append("moe")
        return dispatch._replace(hidden_states=dispatch.hidden_states.clone())

    runtime._prepare_expert_dispatch_for_core = prepare
    controller._prefetch_next_input_group = exact_prefetch
    state.orig_run_moe_core = core
    topk_type = namedtuple("TopK", "topk_ids topk_weights")
    dispatch_type = namedtuple("Dispatch", "hidden_states topk_output")
    dispatch = dispatch_type(
        torch.ones((3, 2)),
        topk_type(
            torch.tensor([[0, 1], [2, 3], [4, 5]]),
            torch.full((3, 2), 0.5),
        ),
    )

    controller.run_token_chunks(state, dispatch)

    prefetch_positions = [
        index for index, event in enumerate(timeline) if isinstance(event, tuple)
    ]
    moe_positions = [
        index for index, event in enumerate(timeline) if event == "moe"
    ]
    assert len(prefetch_positions) == len(moe_positions) - 1
    assert all(
        position < moe_positions[index]
        and timeline[position] == ("prefetch", False)
        for index, position in enumerate(prefetch_positions)
    )


def test_completed_expert_copy_cannot_publish_after_slot_reuse():
    from sglang.srt.layerkv.common_types import _LayerKVPendingExpertCopy

    class DoneEvent:
        def query(self):
            return True

        def elapsed_time(self, _other):
            return 0.0

    runtime, state = shared_runtime()
    state.expert_bytes = 1
    state.logical_to_slot = {1: 1}
    runtime._expert_layers = {state.layer_id: state}
    runtime._pending_expert_copy_events = [
        _LayerKVPendingExpertCopy(
            start_event=DoneEvent(),
            ready_event=DoneEvent(),
            layer_id=state.layer_id,
            logical_ids={1},
            slot_by_logical={1: 0},
            invalidated_logical_ids={1},
        )
    ]

    runtime._finalize_expert_materialize_events(block=False)

    assert state.logical_to_slot == {1: 1}
    assert runtime._pending_expert_copy_events == []


def test_pending_expert_copy_slot_is_not_reusable():
    from sglang.srt.layerkv.common_types import _LayerKVPendingExpertCopy

    class PendingEvent:
        def query(self):
            return False

        def synchronize(self):
            return None

    runtime, state = shared_runtime()
    state.slot_capacity = 3
    state.logical_to_slot = {1: 0, 2: 1, 3: 2}
    state.slot_to_logical = {0: 1, 1: 2, 2: 3}
    state.free_slots = []
    state.lru = {1: 1, 2: 2, 3: 3}
    state.lru_heap = [(1, 0, 1), (2, 1, 2), (3, 2, 3)]
    runtime._pending_expert_copy_events = [
        _LayerKVPendingExpertCopy(
            start_event=PendingEvent(),
            ready_event=PendingEvent(),
            layer_id=state.layer_id,
            logical_ids={1},
            slot_by_logical={1: 0},
        )
    ]

    with pytest.raises(RuntimeError, match="no evictable expert slot"):
        runtime._choose_expert_slot_for_materialize(state, {2, 3})


def test_forward_future_use_prefers_expert_needed_later():
    runtime, state = shared_runtime()
    state.slot_capacity = 3
    state.logical_to_slot = {1: 0, 2: 1, 3: 2}
    state.slot_to_logical = {0: 1, 1: 2, 2: 3}
    state.free_slots = []
    state.lru = {1: 1, 2: 2, 3: 3}
    state.lru_heap = [(1, 0, 1), (2, 1, 2), (3, 2, 3)]
    runtime._layerkv_future_expert_use = {1: 4, 2: 7, 3: None}

    assert runtime._choose_expert_slot_for_materialize(state, {4}) == 2


def test_forward_local_lru_breaks_decode_step_tie():
    runtime, state = shared_runtime()
    state.slot_capacity = 3
    state.logical_to_slot = {1: 0, 2: 1, 3: 2}
    state.slot_to_logical = {0: 1, 1: 2, 2: 3}
    state.free_slots = []
    # All experts were touched in the same decode step.  The transient
    # forward clock must choose expert 2, not the lowest slot/ID.
    state.lru = {1: 7, 2: 7, 3: 7}
    state.lru_heap = [(7, 0, 1), (7, 1, 2), (7, 2, 3)]
    runtime._layerkv_active_expert_lru = {1: 12, 2: 3, 3: 9}

    assert runtime._choose_expert_slot_for_materialize(state, {4}) == 1


def test_forward_local_lru_tracks_route_groups_without_changing_state_lru():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    state.logical_to_slot = {1: 0, 2: 1}
    state.lru = {1: 7, 2: 7}
    controller = SharedExpertController(runtime, None)
    controller._begin_forward_expert_lru(state)
    with controller._forward_expert_lru_scope():
        assert runtime._layerkv_active_expert_lru == {1: 7, 2: 7}
    assert not hasattr(runtime, "_layerkv_active_expert_lru")

    controller._touch_forward_expert_lru(state, (1 << 1) | (1 << 2))
    first_tick = controller._forward_expert_lru_clock - 1
    controller._touch_forward_expert_lru(state, 1 << 1)
    assert controller._forward_expert_lru[1] > first_tick
    assert controller._forward_expert_lru[2] == first_tick
    assert state.lru == {1: 7, 2: 7}


def test_lazy_successor_is_selected_after_current_prepare():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    runtime.config.shared_expert_chunk_order = "reuse"
    state.logical_to_slot = {4: 0, 5: 1}
    controller = SharedExpertController(runtime, None)
    controller.state = state
    runtime._shared_expert = controller
    topk_type = namedtuple("TopK", "topk_ids topk_weights")
    dispatch_type = namedtuple("Dispatch", "hidden_states topk_output")
    output_type = namedtuple("Output", "hidden_states")
    ids = torch.tensor([[0, 1], [2, 3], [1, 0], [4, 5], [3, 2]])
    weights = torch.ones_like(ids, dtype=torch.float32)
    inputs = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    events = []

    original_order = controller._ordered_token_groups

    def tracked_order(*args, **kwargs):
        for group in original_order(*args, **kwargs):
            events.append("select")
            yield group

    def prepare(_state, dispatch):
        events.append("prepare")
        needed = dispatch.topk_output.topk_ids.unique().tolist()
        state.logical_to_slot = {logical: slot for slot, logical in enumerate(needed)}
        return dispatch

    def core(dispatch):
        return output_type(dispatch.hidden_states)

    controller._ordered_token_groups = tracked_order
    runtime._prepare_expert_dispatch_for_core = prepare
    state.orig_run_moe_core = core
    controller.run_token_chunks(
        state, dispatch_type(inputs, topk_type(ids, weights))
    )

    first_prepare = events.index("prepare")
    assert events[: first_prepare + 1] == ["select", "prepare"]
    assert events.count("select") == events.count("prepare")


def test_input_group_eviction_plan_is_forward_local_and_current_safe():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    state.slot_capacity = 2
    state.full_num_experts = 8
    state.logical_to_slot = {0: 0, 1: 1}
    state.slot_to_logical = {0: 0, 1: 1}
    state.free_slots = []
    state.lru = {0: 0, 1: 0}
    controller = SharedExpertController(runtime, None)
    controller.state = state
    groups = [
        [(1 << 0) | (1 << 1), [0]],
        [(1 << 2) | (1 << 3), [1]],
        [(1 << 0) | (1 << 4), [2]],
    ]

    plan = controller._plan_input_group_evictions(state, groups)

    assert plan is not None
    assert plan[0] == []
    assert set(plan[1]) == {0, 1}
    assert len(plan[2]) == 2
    for group_index, victims in enumerate(plan):
        current_ids = set(controller._logical_ids_from_bits(groups[group_index][0]))
        assert not current_ids.intersection(victims)
    assert controller.token_chunk_eviction_plan_ids == 4


def test_input_group_eviction_plan_uses_next_use_not_last_use():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    state.slot_capacity = 2
    state.full_num_experts = 8
    state.logical_to_slot = {0: 0, 1: 1}
    state.slot_to_logical = {0: 0, 1: 1}
    state.free_slots = []
    state.lru = {0: 0, 1: 1}
    controller = SharedExpertController(runtime, None)
    controller.state = state
    groups = [
        [(1 << 2), [0]],
        [(1 << 0), [1]],
        [(1 << 1), [2]],
        [(1 << 0), [3]],
    ]

    plan = controller._plan_input_group_evictions(state, groups)

    assert plan is not None
    # Expert 1 is used later than expert 0, even though expert 0 has the
    # farther final use.  Evicting 0 would force an immediate reload at group 1.
    assert plan[0] == [1]


def test_input_group_eviction_plan_coalesces_safe_future_victims():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, _state = shared_runtime()
    controller = SharedExpertController(runtime, None)
    groups = [
        [(1 << 0), [0]],
        [(1 << 2), [1]],
        [(1 << 3), [2]],
    ]
    plan = [[], [2], [0]]

    coalesced = controller._coalesce_input_group_eviction_plan(groups, plan)

    # Expert 2 is never used before group 1, and expert 0's last use before
    # group 2 is group 0. Both can be submitted at the earlier safe boundary;
    # the original entries remain as guarded fallback entries.
    assert 2 in coalesced[0]
    assert 0 in coalesced[1]
    assert 2 in coalesced[1]
    assert 0 in coalesced[2]
    assert controller._token_chunk_eviction_early_ids_by_group == {
        0: {2},
        1: {0},
    }


def test_planned_eviction_applies_safe_surplus_without_current_miss(monkeypatch):
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    state.slot_capacity = 2
    state.logical_to_slot = {0: 0, 1: 1}
    state.slot_to_logical = {0: 0, 1: 1}
    state.free_slots = []
    controller = runtime._shared_expert = SharedExpertController(runtime, None)
    captured = []

    def evict(_state, logical_ids, **_kwargs):
        captured.append(list(logical_ids))
        return list(logical_ids)

    monkeypatch.setattr(runtime, "_evict_experts_batched", evict)

    evicted = controller._apply_planned_evictions(
        state,
        1 << 0,
        [1],
        surplus_logical_ids={1},
    )

    assert evicted == 1
    assert captured == [[1]]


def test_planned_eviction_batches_remap_and_free_slots(monkeypatch):
    runtime, state = shared_runtime()
    state.slot_capacity = 2
    state.full_num_experts = 4
    state.expert_bytes = 4
    state.device = torch.device("cpu")
    state.param_names = ["weight"]
    state.module = SimpleNamespace(weight=SimpleNamespace(data=torch.ones((2, 1))))
    state.cpu_params = {}
    state.logical_to_slot = {0: 0, 1: 1}
    state.slot_to_logical = {0: 0, 1: 1}
    state.free_slots = []
    state.lru = {0: 0, 1: 1}
    state.backing_lru = {}
    state.remap_tensor = torch.tensor([0, 1, -1, -1], dtype=torch.long)
    captured = []

    def copy_slots(_state, pairs, **_kwargs):
        captured.append(list(pairs))
        return {
            int(logical_id): {"weight": torch.tensor([float(logical_id)])}
            for logical_id, _slot_id in pairs
        }

    monkeypatch.setattr(runtime, "_copy_slots_to_cpu_batched", copy_slots)
    evicted = runtime._evict_experts_batched(state, [1, 0], reason="planned")

    assert evicted == [1, 0]
    assert captured == [[(1, 1), (0, 0)]]
    assert state.logical_to_slot == {}
    assert state.slot_to_logical == {}
    assert state.free_slots == [0, 1]
    assert state.remap_tensor.tolist() == [-1, -1, -1, -1]


def test_candidate_hotness_snapshot_is_not_needed_after_plan():
    runtime, _state = shared_runtime()
    runtime._current_forward_mode = "decode"
    runtime._expert_plan_applied = True
    runtime._expert_install_queue = []

    assert runtime._expert_candidate_cpu_snapshot_needed("decode") is False
    assert runtime.stats.expert_candidate_snapshot_deferred_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hotness_count_and_candidate_order_share_cuda_batch_d2h():
    pytest.importorskip("cuda.bindings.runtime")
    runtime, _state = shared_runtime()
    runtime.config.expert_transfer_backend = "cuda-batch"
    runtime._current_forward_mode = "decode"
    runtime._decode_step = 0
    runtime._expert_plan_applied = False
    ids = torch.tensor([[1, 2], [1, 3]], dtype=torch.long, device="cuda")

    runtime._record_expert_hotness_gpu(0, 8, ids)
    runtime._finalize_expert_hotness_snapshots(block=True)
    runtime._finalize_expert_candidate_snapshots(block=True)
    runtime._expert_batch_transfer.collect(block=True)

    assert runtime.stats.expert_hotness_snapshot_issue_count == 1
    assert runtime.stats.expert_candidate_snapshot_issue_count == 1
    assert runtime.stats.expert_hotness_snapshot_batch_count == 1
    assert runtime.stats.expert_hotness_snapshot_batch_tensor_count == 2
    assert runtime.stats.expert_cuda_batch_d2h_count == 1
    assert runtime._expert_hotness_decode[0] == {1: 2, 2: 1, 3: 1}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("order", ["input", "reuse"])
def test_shared_token_chunks_keep_complete_reductions_and_row_order(dtype, order):
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, state = shared_runtime()
    runtime.config.shared_expert_chunk_order = order
    runtime.config.shared_expert_profile_chunks = True
    state.logical_to_slot = {4: 0, 5: 1}
    controller = SharedExpertController(runtime, None)
    controller.state = state
    runtime._shared_expert = controller
    topk_type = namedtuple("TopK", "topk_ids topk_weights")
    dispatch_type = namedtuple("Dispatch", "hidden_states topk_output")
    output_type = namedtuple("Output", "hidden_states")
    ids = torch.tensor([[0, 1], [2, 3], [1, 0], [4, 5], [3, 2]])
    weights = torch.tensor([[0.1, 0.9]] * 5, dtype=dtype)
    inputs = torch.arange(10, dtype=dtype).reshape(5, 2)
    coefficients = torch.tensor([1.2, 0.3, 2.1, 0.4, 0.5, 3.6, 0.7, 0.8], dtype=dtype)
    calls = []

    def prepare(_state, dispatch):
        assert dispatch.topk_output.topk_ids.unique().numel() <= state.slot_capacity
        needed = dispatch.topk_output.topk_ids.unique().tolist()
        runtime.stats.expert_materialize_count += len(
            set(needed).difference(state.logical_to_slot)
        )
        state.logical_to_slot = {logical: slot for slot, logical in enumerate(needed)}
        return dispatch

    def inplace_core(dispatch):
        topk = dispatch.topk_output
        calls.append(topk.topk_ids.tolist())
        contribution = (coefficients[topk.topk_ids] * topk.topk_weights).sum(1)
        return output_type(dispatch.hidden_states.mul_(contribution[:, None]))

    runtime._prepare_expert_dispatch_for_core = prepare
    state.orig_run_moe_core = inplace_core
    expected = inputs * (coefficients[ids] * weights).sum(1)[:, None]
    result = controller.run_token_chunks(
        state, dispatch_type(inputs, topk_type(ids, weights))
    )
    assert torch.equal(result.hidden_states, expected)
    assert torch.equal(inputs, torch.arange(10, dtype=dtype).reshape(5, 2))
    assert sorted(row for group in calls for row in group) == sorted(ids.tolist())
    assert controller.token_chunk_batches == 1 and controller.token_chunk_calls == 3
    if order == "reuse":
        assert calls[0] == [[4, 5]]  # Already resident; do not evict before use.
    assert controller.token_chunk_materializations == (4 if order == "reuse" else 6)
    assert controller._chunk_wall_ms["total"] > 0
    assert not controller._chunk_events  # CPU-only profiling needs no CUDA events.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA events")
def test_chunk_profile_collects_completed_events_once():
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    runtime, _ = shared_runtime()
    controller = SharedExpertController(runtime, None)
    tensor = torch.ones(16, device="cuda")
    with controller._profile_chunk_phase("moe", tensor.device):
        tensor.add_(1)
    assert not controller._chunk_events and controller._chunk_wall_ms["moe"] == 0
    runtime.config.shared_expert_profile_chunks = True
    with controller._profile_chunk_phase("moe", tensor.device):
        tensor.add_(1)
    torch.cuda.synchronize()
    controller._collect_chunk_profile()
    measured = controller._chunk_gpu_ms["moe"]
    assert measured > 0 and not controller._chunk_events
    controller._collect_chunk_profile()
    assert controller._chunk_gpu_ms["moe"] == measured
