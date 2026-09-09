from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from sglang.srt.layerkv.shared_expert import (
    SharedExpertController,
    SharedExpertManager,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder


class Admission:
    add_one_req_ignore_eos = PrefillAdder.add_one_req_ignore_eos
    _try_layerkv_recall_for_admission = PrefillAdder._try_layerkv_recall_for_admission
    _try_layerkv_recall_for_prefill_budget = (
        PrefillAdder._try_layerkv_recall_for_prefill_budget
    )
    _try_layerkv_recall_for_physical_admission = (
        PrefillAdder._try_layerkv_recall_for_physical_admission
    )

    def __init__(self, available, recovered, *, reported=None):
        self.available = available
        self.calls = []
        self.context_calls = []

        def recall(*, shortage_tokens, **kwargs):
            self.calls.append(shortage_tokens)
            self.context_calls.append(kwargs)
            self.available += recovered
            return recovered if reported is None else reported

        self.token_to_kv_pool_allocator = SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(
                layerkv_runtime=SimpleNamespace(
                    recall_shared_expert_for_admission=recall
                )
            )
        )
        self.is_hybrid_swa = False
        self.req_states = None
        self.running_batch = None
        self.can_run_list = []
        self.prefill_delayer_single_pass = None
        self.dllm_config = None
        self.rem_chunk_tokens = None

    @property
    def cur_rem_tokens(self):
        return self.available

    rem_total_tokens = cur_rem_tokens
    ceil_paged_tokens = staticmethod(int)

    def _update_prefill_budget(self, *_args):
        pass

    def budget_state(self):
        return AddReqResult.CONTINUE


def request():
    return SimpleNamespace(
        extend_input_len=100,
        sampling_params=SimpleNamespace(ignore_eos=True, max_new_tokens=10),
        output_ids=[],
        origin_input_ids=list(range(100)),
    )


@pytest.mark.parametrize("available", [0, 105])
def test_recall_precedes_prefill_or_decode_reservation_rejection(available):
    admission = Admission(available, 200)
    req = request()
    assert admission.add_one_req_ignore_eos(req) == AddReqResult.CONTINUE
    assert admission.can_run_list == [req]
    assert len(admission.calls) == 1 and admission.calls[0] > 0


def test_sufficient_capacity_does_not_recall_experts():
    admission = Admission(200, 200)
    assert admission.add_one_req_ignore_eos(request()) == AddReqResult.CONTINUE
    assert not admission.calls


@pytest.mark.parametrize("reported", [0, 10000])
def test_admission_rechecks_real_capacity_not_callback_estimate(reported):
    admission = Admission(0, 0, reported=reported)
    assert admission.add_one_req_ignore_eos(request()) == AddReqResult.NO_TOKEN
    assert admission.can_run_list == []


def test_missing_layerkv_interface_is_noop():
    admission = Admission(0, 0)
    admission.token_to_kv_pool_allocator = SimpleNamespace()
    assert admission.add_one_req_ignore_eos(request()) == AddReqResult.NO_TOKEN


def test_fractional_reservation_shortage_rounds_up():
    admission = Admission(0, 1)
    assert admission._try_layerkv_recall_for_admission(0.25)
    assert admission.calls == [1]


def test_recall_credit_expands_only_the_current_prefill_input_budget():
    admission = Admission(0, 200)
    admission.rem_input_tokens = 100
    admission.layerkv_prefill_credit_tokens = 0

    assert admission._try_layerkv_recall_for_admission(1) == 200
    assert admission.rem_input_tokens == 300
    assert admission.layerkv_prefill_credit_tokens == 200


def test_prefill_budget_recall_requires_an_existing_queued_batch():
    admission = Admission(0, 200)
    admission.rem_input_tokens = 100
    admission.layerkv_prefill_credit_tokens = 0
    admission.waiting_queue_len = 1
    admission.can_run_list = [object()]

    assert admission._try_layerkv_recall_for_prefill_budget(100) == 200
    assert admission.rem_input_tokens == 300

    admission = Admission(0, 200)
    admission.rem_input_tokens = 100
    admission.waiting_queue_len = 1
    admission.can_run_list = []
    assert admission._try_layerkv_recall_for_prefill_budget(100) == 0
    assert not admission.calls


def test_ignore_eos_path_can_expand_prefill_budget_after_recall():
    admission = Admission(100, 200)
    admission.rem_input_tokens = 100
    admission.layerkv_prefill_credit_tokens = 0
    admission.waiting_queue_len = 1

    req = request()
    admission.can_run_list = [req]
    assert admission.add_one_req_ignore_eos(req) == AddReqResult.CONTINUE
    assert admission.rem_input_tokens == 300
    assert admission.calls


def test_long_context_physical_shortage_recalls_before_prefill_admission():
    admission = Admission(10000, 500)
    admission.can_run_list = [object()]
    admission.token_to_kv_pool_allocator.available_size = lambda: 100

    assert (
        admission._try_layerkv_recall_for_physical_admission(
            4096, context_tokens=4096
        )
        == 500
    )
    assert admission.calls == [3996]
    assert admission.context_calls == [
        {"context_tokens": 4096, "batch_size": 2, "allow_kv_overflow": True}
    ]


def test_short_context_or_sufficient_physical_capacity_does_not_recall():
    admission = Admission(10000, 500)
    admission.can_run_list = [object()]
    admission.token_to_kv_pool_allocator.available_size = lambda: 100
    assert (
        admission._try_layerkv_recall_for_physical_admission(
            4096, context_tokens=2048
        )
        == 0
    )
    assert not admission.calls

    admission.token_to_kv_pool_allocator.available_size = lambda: 4096
    assert (
        admission._try_layerkv_recall_for_physical_admission(
            4096, context_tokens=4096
        )
        == 0
    )
    assert not admission.calls


def test_running_batch_also_enables_physical_admission_recall():
    admission = Admission(10000, 500)
    admission.running_batch = SimpleNamespace(reqs=[object(), object()])
    admission.token_to_kv_pool_allocator.available_size = lambda: 100

    assert (
        admission._try_layerkv_recall_for_physical_admission(
            4096, context_tokens=4096
        )
        == 500
    )
    assert admission.calls == [3996]
    assert admission.context_calls[0]["batch_size"] == 3


def test_new_long_context_batch_can_trigger_physical_admission_recall():
    admission = Admission(10000, 500)
    admission.token_to_kv_pool_allocator.available_size = lambda: 100

    assert (
        admission._try_layerkv_recall_for_physical_admission(
            4096, context_tokens=4096
        )
        == 500
    )
    assert admission.calls == [3996]
    assert admission.context_calls == [
        {"context_tokens": 4096, "batch_size": 1, "allow_kv_overflow": True}
    ]


def test_physical_admission_counts_unallocated_candidate_batch_tokens():
    admission = Admission(4000, 2048)
    admission.can_run_list = [SimpleNamespace(extend_input_len=3000)]
    admission.token_to_kv_pool_allocator.available_size = lambda: 4000

    assert (
        admission._try_layerkv_recall_for_physical_admission(
            3000, context_tokens=3000
        )
        == 2048
    )
    assert admission.calls == [2000]
    assert admission.context_calls == [
        {"context_tokens": 3000, "batch_size": 2, "allow_kv_overflow": True}
    ]


def test_waiting_prefill_probe_can_prepare_overflow_when_native_slots_are_full():
    calls = []
    runtime = SimpleNamespace(
        recall_shared_expert_for_admission=lambda **kwargs: calls.append(kwargs)
        or 2048,
        release_scheduler_admission_credit_tokens=lambda **_kwargs: 0,
        get_scheduler_admission_credit_tokens=lambda **_kwargs: 0,
    )
    req = SimpleNamespace(
        extend_input_len=3000,
        host_hit_length=0,
        sampling_params=SimpleNamespace(max_new_tokens=8),
        output_ids=[],
        origin_input_ids=list(range(3000)),
    )
    scheduler = SimpleNamespace(
        waiting_queue=[req],
        running_batch=SimpleNamespace(
            reqs=[object(), object()], is_empty=lambda: False
        ),
        max_running_requests=2,
        req_to_token_pool=SimpleNamespace(available_size=lambda: 0),
        token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 0),
        page_size=1,
        _get_layerkv_runtime=lambda: runtime,
    )

    assert Scheduler._try_layerkv_prepare_waiting_prefill(scheduler)
    assert calls == [
        {
            "shortage_tokens": 3009,
            "context_tokens": 3000,
            "batch_size": 3,
            "allow_kv_overflow": True,
        }
    ]


def test_scheduler_visible_physical_credit_avoids_expert_recall():
    admission = Admission(4095, 805)
    admission.can_run_list = [object()]
    runtime = admission.token_to_kv_pool_allocator.get_kvcache().layerkv_runtime
    runtime.get_scheduler_admission_credit_tokens = lambda **kwargs: 805
    admission.token_to_kv_pool_allocator.get_kvcache = lambda: SimpleNamespace(
        layerkv_runtime=runtime
    )
    admission.token_to_kv_pool_allocator.available_size = lambda: 4095

    assert (
        admission._try_layerkv_recall_for_physical_admission(
            4900, context_tokens=4900
        )
        == 0
    )
    assert not admission.calls


def test_admission_passes_context_and_actual_batch_to_overflow_callback():
    admission = Admission(0, 200)
    admission.rem_input_tokens = 10000
    admission.layerkv_prefill_credit_tokens = 0
    req = request()
    assert admission.add_one_req_ignore_eos(req) == AddReqResult.CONTINUE
    assert admission.context_calls == [
        {"context_tokens": 100, "batch_size": 1, "allow_kv_overflow": True}
    ]


def test_overflow_admission_requires_long_context_and_small_batch():
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.kv_overflow_segment = None
    controller.kv_overflow_active_tokens = 0
    controller.kv_overflow_admission_activation_count = 0
    controller.kv_overflow_tokens = 4096
    controller.kv_min_slots = 8
    controller.state = type("State", (), {"slot_capacity": 16})()
    controller._kv_overflow_enabled = lambda: True

    calls = []

    def shrink(target):
        calls.append(target)
        controller.kv_overflow_active_tokens = 4096
        return True

    controller._shrink_expert_to_kv = shrink
    assert (
        controller.prepare_kv_overflow_for_admission(
            context_tokens=4096, batch_size=4
        )
        == 4096
    )
    assert calls == [8]
    assert controller.kv_overflow_admission_activation_count == 1

    controller.kv_overflow_active_tokens = 0
    calls.clear()
    assert (
        controller.prepare_kv_overflow_for_admission(
            context_tokens=1024, batch_size=4
        )
        == 0
    )
    assert not calls
    assert (
        controller.prepare_kv_overflow_for_admission(
            context_tokens=4096, batch_size=9
        )
        == 0
    )
    assert not calls


def test_overflow_admission_installs_lazy_controller_before_physical_shrink():
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.kv_overflow_segment = None
    controller.kv_overflow_active_tokens = 0
    controller.kv_overflow_admission_activation_count = 0
    controller.kv_overflow_tokens = 4096
    controller.kv_min_slots = 8
    controller.state = None
    controller.ensure_installed = lambda: setattr(
        controller, "state", SimpleNamespace(slot_capacity=16)
    )
    controller._kv_overflow_enabled = lambda: True
    controller._select_kv_overflow_target = lambda _shortage: 12
    controller.runtime = SimpleNamespace(
        config=SimpleNamespace(shared_expert_headroom_steps=16)
    )
    calls = []

    def shrink(target, *, requested_tokens=None):
        calls.append((target, requested_tokens))
        controller.kv_overflow_active_tokens = requested_tokens
        return True

    controller._shrink_expert_to_kv = shrink

    assert (
        controller.prepare_kv_overflow_for_admission(
            context_tokens=4096, batch_size=3, shortage_tokens=553
        )
        == 601
    )
    assert calls == [(12, 601)]


def test_all_layer_overflow_admission_installs_every_lazy_controller():
    manager = SharedExpertManager.__new__(SharedExpertManager)
    first = SimpleNamespace(state=None)
    second = SimpleNamespace(state=None)
    installed = []
    for name, controller in (("first", first), ("second", second)):
        controller.ensure_installed = lambda name=name: installed.append(name)
        controller.prepare_kv_overflow_for_admission = lambda **_kwargs: 0
    manager.controllers = {0: first, 1: second}
    manager.ensure_installed = lambda: [
        controller.ensure_installed() for controller in manager.controllers.values()
    ]

    assert (
        manager.prepare_kv_overflow_for_admission(
            context_tokens=4096, batch_size=1, shortage_tokens=32
        )
        == 0
    )
    assert installed == ["first", "second"]


def test_overflow_target_keeps_slots_when_physical_pages_cover_shortage():
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.base_slots = 16
    controller.kv_min_slots = 8
    controller.kv_overflow_tokens = 4096
    controller.state = SimpleNamespace(slot_capacity=16)
    controller.arena = SimpleNamespace(
        loans=[],
        _kv_overflow_destinations=lambda _base, _tokens: ([None] * 12, 0),
    )
    controller.runtime = SimpleNamespace(_allocator_total_size=lambda: 16384)
    controller._expert_page_layouts = lambda _state: [(None, 2), (None, 1)]

    assert controller._select_kv_overflow_target(2309) == 12
    assert controller._select_kv_overflow_target(4097) == 8


def test_overflow_target_keeps_capacity_when_shortage_needs_no_physical_page():
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.base_slots = 16
    controller.kv_min_slots = 8
    controller.kv_overflow_tokens = 4096
    controller.state = SimpleNamespace(slot_capacity=16)
    controller.arena = SimpleNamespace(
        loans=[],
        _kv_overflow_destinations=lambda _base, _tokens: ([], 0),
    )
    controller.runtime = SimpleNamespace(_allocator_total_size=lambda: 16384)
    controller._expert_page_layouts = lambda _state: [(None, 2), (None, 1)]

    assert controller._select_kv_overflow_target(4) == 16


def test_route_capacity_floor_uses_topk_and_request_batch():
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.base_slots = 16
    controller.kv_min_slots = 8
    controller.state = SimpleNamespace(
        slot_capacity=16, module=SimpleNamespace(top_k=8)
    )

    assert controller._minimum_expert_slots_for_batch(1) == 8
    assert controller._minimum_expert_slots_for_batch(2) == 16
    assert controller._minimum_expert_slots_for_batch(4) == 16


def test_residency_batch_uses_scheduler_schedule_size():
    controller = SharedExpertController.__new__(SharedExpertController)
    runtime = SimpleNamespace(
        _last_forward_batch=SimpleNamespace(batch_size=1),
        stats=SimpleNamespace(
            native_schedule_batch_size=3,
            native_schedule_running_batch_size=2,
        ),
    )
    controller.runtime = runtime

    assert controller._current_residency_batch_size() == 3


def test_budget_target_cannot_bypass_batch_route_floor():
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.base_slots = 16
    controller.kv_min_slots = 8
    controller.state = SimpleNamespace(
        slot_capacity=16, module=SimpleNamespace(top_k=8)
    )
    controller.budget = SimpleNamespace(target=8, reason="kv-pressure")

    assert controller._apply_batch_residency_floor(8, 2) == 16
    assert controller.budget.target == 16
    assert controller.budget.reason == "batch-route-floor"


def test_long_context_base_floor_uses_routed_width():
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.base_slots = 16
    controller.state = SimpleNamespace(
        slot_capacity=16, module=SimpleNamespace(top_k=8)
    )

    assert controller._long_context_small_batch(2, 4096)
    assert not controller._long_context_small_batch(3, 4096)
    assert not controller._long_context_small_batch(2, 2048)


def test_soft_context_pressure_uses_residual_overflow_target():
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.kv_overflow_tokens = 4096
    controller.kv_overflow_active_tokens = 0
    controller.kv_overflow_segment = None
    controller.kv_min_slots = 8
    controller.state = SimpleNamespace(slot_capacity=16)
    controller._select_kv_overflow_target = lambda shortage: 12

    assert controller._context_pressure_overflow_plan(4095, 4096, 2) == (12, 1)
    assert controller._context_pressure_overflow_plan(4094, 4096, 2) == (None, None)


def test_soft_context_pressure_does_not_fall_below_multi_request_route_floor():
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.base_slots = 16
    controller.kv_overflow_tokens = 4096
    controller.kv_overflow_active_tokens = 0
    controller.kv_overflow_segment = None
    controller.kv_min_slots = 8
    controller.state = SimpleNamespace(
        slot_capacity=16, module=SimpleNamespace(top_k=8)
    )
    controller._select_kv_overflow_target = lambda _shortage: 8

    assert controller._context_pressure_overflow_plan(
        4095, 4096, 2, batch_size=2
    ) == (16, 1)


def test_admission_allows_tail_reclaim_below_decode_batch_route_width():
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.base_slots = 16
    controller.kv_min_slots = 8
    controller.kv_overflow_tokens = 4096
    controller.kv_overflow_active_tokens = 0
    controller.kv_overflow_segment = None
    controller.kv_overflow_admission_activation_count = 0
    controller.state = SimpleNamespace(
        slot_capacity=16, module=SimpleNamespace(top_k=8)
    )
    controller.runtime = SimpleNamespace(
        config=SimpleNamespace(shared_expert_headroom_steps=16)
    )
    controller._kv_overflow_enabled = lambda: True
    controller._select_kv_overflow_target = lambda _shortage: 8
    calls = []
    def shrink(target, *, requested_tokens=None):
        calls.append((target, requested_tokens))
        controller.kv_overflow_active_tokens = requested_tokens
        return True

    controller._shrink_expert_to_kv = shrink

    assert (
        controller.prepare_kv_overflow_for_admission(
            context_tokens=4096, batch_size=2, shortage_tokens=1
        )
        == 33
    )
    assert calls == [(8, 33)]


def test_long_context_admission_can_reclaim_only_page_feasible_tail():
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.base_slots = 16
    controller.kv_min_slots = 8
    controller.kv_overflow_tokens = 4096
    controller.kv_overflow_active_tokens = 0
    controller.kv_overflow_segment = None
    controller.kv_overflow_admission_activation_count = 0
    controller.state = SimpleNamespace(
        slot_capacity=16, module=SimpleNamespace(top_k=8)
    )
    controller.runtime = SimpleNamespace(
        config=SimpleNamespace(shared_expert_headroom_steps=16)
    )
    controller._kv_overflow_enabled = lambda: True
    controller._select_kv_overflow_target = lambda _shortage: 12
    calls = []

    def shrink(target, *, requested_tokens=None):
        calls.append(target)
        controller.kv_overflow_active_tokens = requested_tokens
        return True

    controller._shrink_expert_to_kv = shrink

    assert (
        controller.prepare_kv_overflow_for_admission(
            context_tokens=4096, batch_size=3, shortage_tokens=1
        )
        == 49
    )
    assert calls == [12]


def test_overflow_admission_aligns_residual_shortage_to_physical_page():
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.base_slots = 16
    controller.kv_min_slots = 8
    controller.kv_overflow_tokens = 4096
    controller.kv_overflow_active_tokens = 0
    controller.kv_overflow_segment = None
    controller.kv_overflow_admission_activation_count = 0
    controller.arena = SimpleNamespace(
        _kv_overflow_destinations=lambda _base, tokens: (
            ([None] if int(tokens) >= 1024 else []),
            0,
        )
    )
    controller.state = SimpleNamespace(
        slot_capacity=16, module=SimpleNamespace(top_k=8)
    )
    controller.runtime = SimpleNamespace(
        config=SimpleNamespace(
            shared_expert_headroom_steps=16, kvc_block_tokens=2048
        ),
        _allocator_total_size=lambda: 16384,
    )
    controller._kv_overflow_enabled = lambda: True
    selected = []
    controller._select_kv_overflow_target = lambda shortage: selected.append(
        shortage
    ) or 12

    def shrink(_target, *, requested_tokens=None):
        controller.kv_overflow_active_tokens = requested_tokens
        return True

    controller._shrink_expert_to_kv = shrink

    assert (
        controller.prepare_kv_overflow_for_admission(
            context_tokens=4096, batch_size=3, shortage_tokens=553
        )
        == 1024
    )
    assert selected == [1024]


def test_overflow_activation_requests_residual_shortage_plus_headroom():
    controller = SharedExpertController.__new__(SharedExpertController)
    controller.kv_overflow_segment = None
    controller.kv_overflow_active_tokens = 0
    controller.kv_overflow_admission_activation_count = 0
    controller.kv_min_slots = 8
    controller.kv_overflow_tokens = 4096
    controller.state = SimpleNamespace(slot_capacity=16)
    controller.runtime = SimpleNamespace(
        config=SimpleNamespace(shared_expert_headroom_steps=16)
    )
    controller._kv_overflow_enabled = lambda: True
    controller._select_kv_overflow_target = lambda shortage: 12
    calls = []

    def shrink(target, *, requested_tokens=None):
        calls.append((target, requested_tokens))
        controller.kv_overflow_active_tokens = requested_tokens
        return True

    controller._shrink_expert_to_kv = shrink

    assert (
        controller.prepare_kv_overflow_for_admission(
            context_tokens=4096, batch_size=4, shortage_tokens=2309
        )
        == 2373
    )
    assert calls == [(12, 2373)]


def test_failed_ownership_transfer_is_not_swallowed():
    admission = Admission(0, 0)

    def fail(**_kwargs):
        raise RuntimeError("mapping failed")

    admission.token_to_kv_pool_allocator = SimpleNamespace(
        get_kvcache=lambda: SimpleNamespace(
            layerkv_runtime=SimpleNamespace(recall_shared_expert_for_admission=fail)
        )
    )
    with pytest.raises(RuntimeError, match="mapping failed"):
        admission.add_one_req_ignore_eos(request())


@pytest.mark.parametrize("after_lock", [False, True])
@pytest.mark.parametrize("recovered", [0, 200])
def test_normal_request_rechecks_capacity_before_and_after_prefix_lock(
    after_lock, recovered
):
    admission = Admission(200 if after_lock else 0, recovered)
    admission.nsa_prefill_cp_in_seq_split = False
    admission.prefill_context_parallel_enabled = False
    admission.prefill_max_requests = None
    admission.page_size = 1
    admission.rem_input_tokens = 10000
    admission.tree_cache = SimpleNamespace(disable=True)
    admission._req_inc_lock_ref = lambda _req: None

    @contextmanager
    def lock(_node):
        if after_lock:
            admission.available = 0
        yield

    admission._lock_node = lock
    req = request()
    req.sampling_params.ignore_eos = False
    req.host_hit_length = 0
    req.prefix_indices = []
    req.last_node = None
    result = PrefillAdder.add_one_req(admission, req, False, None)
    assert result == (AddReqResult.CONTINUE if recovered else AddReqResult.NO_TOKEN)
    assert admission.can_run_list == ([req] if recovered else [])
    assert len(admission.calls) == 1
