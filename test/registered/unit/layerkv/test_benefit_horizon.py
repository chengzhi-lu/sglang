from types import SimpleNamespace

import pytest

from sglang.srt.layerkv.residency_budget import (
    ResidencyBudget,
    remaining_fixed_decode_steps,
)


@pytest.mark.parametrize(
    "horizon,target", [(None, 16), (8, 16), (56, 64), (2, 16), (0, 16)]
)
def test_recorded_regrowth_cost_uses_future_work_not_past_savings(horizon, target):
    budget = ResidencyBudget(16, 64)
    budget.samples = 8
    budget.misses = [218, 0]
    assert (
        budget.decide(
            8,
            batch_size=8,
            free_tokens=10000,
            waiting=0,
            shortage=0,
            transfer_ms_per_miss=27.096 / 218,
            growth_cost_ms=135.899,
            benefit_horizon_steps=horizon,
        )
        == target
    )
    assert budget.samples == 0 and budget.misses == [0, 0]


def test_pressure_overrides_long_benefit_horizon():
    budget = ResidencyBudget(16, 64)
    budget.samples = 8
    budget.misses = [218, 0]
    assert (
        budget.decide(
            8,
            batch_size=8,
            free_tokens=10000,
            waiting=1,
            shortage=100,
            transfer_ms_per_miss=1,
            growth_cost_ms=1,
            benefit_horizon_steps=100,
        )
        == 16
    )
    assert budget.reason == "kv-pressure"


@pytest.mark.parametrize(
    "slots,pressure,expected", [(49, 0, 64), (16, 0, 16), (49, 1, 16)]
)
def test_controller_final_step_prices_only_new_growth(slots, pressure, expected):
    from sglang.srt.layerkv.config_stats import LayerKVConfig
    from sglang.srt.layerkv.shared_expert import SharedExpertController

    c = SharedExpertController.__new__(SharedExpertController)
    c.state = SimpleNamespace(full_num_experts=256, slot_capacity=slots)
    c.base_slots, c.extra_slots = 16, 48
    c.budget = ResidencyBudget(16, 64)
    c.budget.target, c.budget.samples, c.budget.misses = 64, 8, [20, 0]
    c.admission_shortage_tokens = pressure
    c.last_growth_ms = 100
    c._next_donor_scan_step = 100
    recalls = []
    c.recall = lambda: recalls.append(True)
    c.runtime = SimpleNamespace(
        config=LayerKVConfig(shared_expert_benefit_horizon_steps=64),
        _decode_step=8,
        _shared_expert_remaining_decode_steps=0,
        _last_forward_batch=SimpleNamespace(batch_size=8),
        _allocator=SimpleNamespace(available_size=lambda: 10000),
        stats=SimpleNamespace(
            native_schedule_waiting_queue_len=pressure,
            expert_materialize_ms=1,
            expert_materialize_count=1,
        ),
    )
    c.after_decode()
    assert c.budget.target == expected
    assert bool(recalls) == bool(pressure)


def test_scheduler_clears_remaining_work_on_prefill_and_uncertain_stopping():
    from sglang.srt.layerkv.config_stats import LayerKVConfig
    from sglang.srt.layerkv.runtime import LayerKVRuntime

    runtime = LayerKVRuntime(LayerKVConfig(shared_expert_benefit_horizon_steps=64))
    runtime._finalize_kvc_evictions = lambda **kwargs: None
    runtime._schedule_batch_req_lens = lambda batch: []
    runtime._record_scheduler_budget_observation = lambda *args: None
    batch = SimpleNamespace(reqs=[request()])
    runtime.on_schedule_batch(
        schedule_batch=batch, scheduler_context={"forward_mode": "decode"}
    )
    assert runtime._shared_expert_remaining_decode_steps == 56
    runtime.on_schedule_batch(
        schedule_batch=batch, scheduler_context={"forward_mode": "extend"}
    )
    assert runtime._shared_expert_remaining_decode_steps is None
    batch.reqs = [request(ignore_eos=False)]
    runtime.on_schedule_batch(
        schedule_batch=batch, scheduler_context={"forward_mode": "decode"}
    )
    assert runtime._shared_expert_remaining_decode_steps is None


def request(produced=7, **options):
    return SimpleNamespace(
        output_ids=[0] * produced,
        sampling_params=SimpleNamespace(
            **{
                "ignore_eos": True,
                "max_new_tokens": 64,
                **options,
            }
        ),
    )


def test_remaining_work_is_capped_by_earliest_departure_and_current_forward():
    assert remaining_fixed_decode_steps([request(), request(60)]) == 3
    assert remaining_fixed_decode_steps([request(64)]) == 0
    assert remaining_fixed_decode_steps([]) is None


@pytest.mark.parametrize(
    "options",
    [
        {"ignore_eos": False},
        {"stop_strs": ["stop"]},
        {"stop_token_ids": [4]},
        {"stop_regex_strs": "end"},
        {"max_new_tokens": None},
    ],
)
def test_uncertain_stopping_does_not_claim_a_fixed_horizon(options):
    assert remaining_fixed_decode_steps([request(), request(**options)]) is None


def test_real_sampling_params_regex_disables_fixed_horizon():
    from sglang.srt.sampling.sampling_params import SamplingParams

    req = SimpleNamespace(
        output_ids=[1],
        sampling_params=SamplingParams(
            max_new_tokens=64,
            ignore_eos=True,
            stop_regex="end",
        ),
    )
    assert remaining_fixed_decode_steps([req]) is None
