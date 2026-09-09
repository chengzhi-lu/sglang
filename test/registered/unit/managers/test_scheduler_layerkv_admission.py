from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler


def test_full_prefill_batch_can_open_a_long_context_slot_with_layerkv():
    scheduler = Scheduler.__new__(Scheduler)
    calls = []
    scheduler.waiting_queue = [
        SimpleNamespace(extend_input_len=4900, host_hit_length=0)
    ]
    scheduler.running_batch = SimpleNamespace(
        reqs=[object(), object()], is_empty=lambda: False, batch_is_full=True
    )
    scheduler.page_size = 1
    scheduler.get_num_allocatable_reqs = lambda _running_bs: 1
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        available_size=lambda: 4095
    )
    scheduler._get_layerkv_runtime = lambda: SimpleNamespace(
        recall_shared_expert_for_admission=lambda **kwargs: calls.append(kwargs) or 805
    )

    assert Scheduler._try_layerkv_prepare_waiting_prefill(scheduler)
    assert calls == [
        {
            "shortage_tokens": 805,
            "context_tokens": 4900,
            "batch_size": 3,
            "allow_kv_overflow": True,
        }
    ]


def test_scheduler_visible_physical_credit_opens_slot_without_expert_recall():
    scheduler = Scheduler.__new__(Scheduler)
    calls = []
    scheduler.waiting_queue = [
        SimpleNamespace(extend_input_len=4900, host_hit_length=0)
    ]
    scheduler.running_batch = SimpleNamespace(
        reqs=[object(), object()], is_empty=lambda: False, batch_is_full=True
    )
    scheduler.page_size = 1
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        available_size=lambda: 4095
    )

    def credit(**kwargs):
        assert kwargs == {
            "required_tokens": 4900,
            "available_tokens": 4095,
            "reason": "decode_prealloc_admission",
        }
        return 805

    scheduler._get_layerkv_runtime = lambda: SimpleNamespace(
        get_scheduler_admission_credit_tokens=credit,
        recall_shared_expert_for_admission=lambda **kwargs: calls.append(kwargs) or 0,
    )

    assert Scheduler._try_layerkv_prepare_waiting_prefill(scheduler)
    assert not calls


def test_scheduler_releases_common_kvc_credit_before_admission():
    scheduler = Scheduler.__new__(Scheduler)
    calls = []
    scheduler.waiting_queue = [
        SimpleNamespace(
            extend_input_len=4900,
            host_hit_length=0,
            output_ids=[],
            sampling_params=SimpleNamespace(max_new_tokens=64),
        )
    ]
    scheduler.running_batch = SimpleNamespace(
        reqs=[object(), object()], is_empty=lambda: False, batch_is_full=True
    )
    scheduler.page_size = 1
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        available_size=lambda: 0
    )

    def release(**kwargs):
        assert kwargs == {"required_tokens": 4965, "available_tokens": 0}
        return 4965

    scheduler._get_layerkv_runtime = lambda: SimpleNamespace(
        release_scheduler_admission_credit_tokens=release,
        get_scheduler_admission_credit_tokens=lambda **_kwargs: 0,
        recall_shared_expert_for_admission=lambda **kwargs: calls.append(kwargs)
        or 0,
    )

    assert Scheduler._try_layerkv_prepare_waiting_prefill(scheduler)
    assert not calls
