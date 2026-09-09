from types import SimpleNamespace

import pytest

from sglang.srt.layerkv.residency_budget import ResidencyBudget


def decide(policy, step, **kwargs):
    args = dict(batch_size=2, free_tokens=100, waiting=0, shortage=0)
    args.update(kwargs)
    return policy.decide(step, **args)


def warm_reuse(policy, **costs):
    for step, ids in enumerate(([0, 1], [2], [0, 1], [2]), 1):
        policy.observe(step, ids)
        target = decide(policy, step, **costs)
    return target


def test_retains_experts_only_after_observed_miss_benefit():
    policy = ResidencyBudget(2, 3, interval=2)
    assert warm_reuse(policy) == 3
    assert policy.saved_misses > 0 and policy.reason == "expert-reuse"
    assert policy.decisions == 2


@pytest.mark.parametrize("cost,expected", [(0.1, 3), (50.0, 2)])
def test_observed_transfer_benefit_must_repay_growth(cost, expected):
    policy = ResidencyBudget(2, 3, interval=2)
    assert warm_reuse(policy, transfer_ms_per_miss=1.0, growth_cost_ms=cost) == expected
    assert policy.estimated_saved_ms > 0
    assert policy.cost_rejections == (expected == 2)


def test_unknown_transfer_cost_does_not_pay_known_growth_cost():
    policy = ResidencyBudget(2, 3, interval=2)
    assert warm_reuse(policy, growth_cost_ms=10.0) == 2
    assert policy.estimated_saved_ms is None


def test_pressure_does_not_accumulate_unbounded_future_benefit():
    policy = ResidencyBudget(2, 3, interval=2)
    for step in range(1, 20):
        policy.observe(step, [step % 3])
        assert decide(policy, step, free_tokens=0) == 2
        assert policy.samples == 0 and policy.misses == [0, 0]
    policy.observe(20, [0])
    assert decide(policy, 20) == 2  # collect a fresh interval after pressure


@pytest.mark.parametrize("cost", [-1.0, float("nan"), float("inf")])
def test_invalid_cost_measurements(cost):
    with pytest.raises(ValueError, match="finite"):
        decide(ResidencyBudget(2, 3), 1, growth_cost_ms=cost)


def test_no_benefit_and_duplicate_observations_do_not_grow():
    policy = ResidencyBudget(2, 3, interval=2)
    for step in range(1, 7):
        policy.observe(step, [0, 1])
        policy.observe(step, [99])
        assert decide(policy, step) == 2
    assert policy.decisions == 3 and policy.reason == "no-miss-benefit"


def test_pressure_overrides_reuse_and_recall_cools_regrowth():
    policy = ResidencyBudget(2, 3, interval=2)
    assert warm_reuse(policy) == 3
    assert decide(policy, 5, free_tokens=31) == 2
    assert policy.reason == "kv-pressure"
    policy.recalled(5)
    policy.observe(6, [0, 1])
    assert decide(policy, 6) == 2
    assert decide(policy, 7, waiting=1, shortage=100) == 2


def test_context_pressure_overrides_reuse_without_a_hard_length_threshold():
    policy = ResidencyBudget(2, 3, interval=2)
    assert warm_reuse(policy) == 3
    assert (
        decide(
            policy,
            5,
            context_demand_tokens=70,
            context_capacity_tokens=100,
        )
        == 2
    )
    assert policy.reason == "context-pressure"
    assert policy.context_pressure_count == 1
    assert policy.last_context_reserve_tokens == 32


def test_short_context_large_batch_can_keep_expert_reuse():
    policy = ResidencyBudget(2, 3, interval=2, headroom_steps=1)
    assert warm_reuse(policy) == 3
    assert (
        decide(
            policy,
            5,
            batch_size=32,
            context_demand_tokens=1000,
            context_capacity_tokens=4096,
        )
        == 3
    )


def test_long_context_small_batch_returns_expert_capacity_to_kv():
    policy = ResidencyBudget(2, 3, interval=2, headroom_steps=1)
    assert warm_reuse(policy) == 3
    assert (
        decide(
            policy,
            5,
            batch_size=2,
            context_demand_tokens=4095,
            context_capacity_tokens=4096,
        )
        == 2
    )
    assert policy.reason == "context-pressure"


def test_context_pressure_accepts_a_physically_sized_partial_target():
    policy = ResidencyBudget(2, 4, interval=2, headroom_steps=1, min_slots=1)
    assert warm_reuse(policy) == 4
    assert (
        decide(
            policy,
            5,
            context_demand_tokens=4095,
            context_capacity_tokens=4096,
            context_target_slots=2,
        )
        == 2
    )
    assert policy.reason == "context-pressure-partial"


def test_context_demand_and_capacity_must_be_paired():
    with pytest.raises(ValueError, match="provided together"):
        decide(ResidencyBudget(2, 3), 1, context_demand_tokens=10)


@pytest.mark.parametrize("args", [(0, 3), (3, 2), (2, 3, 0), (2, 3, 1, 0)])
def test_invalid_configuration(args):
    with pytest.raises(ValueError):
        ResidencyBudget(*args)


def test_server_arguments_reach_runtime_policy():
    from sglang.srt.layerkv.config_stats import LayerKVConfig

    config = LayerKVConfig.from_server_args(
        SimpleNamespace(
            layerkv_shared_expert_policy="adaptive",
            layerkv_shared_expert_decision_interval=3,
            layerkv_shared_expert_headroom_steps=7,
            layerkv_shared_expert_retain_across_requests=True,
            layerkv_shared_expert_admission_policy="recall",
        )
    )
    assert config.shared_expert_policy == "adaptive"
    assert config.shared_expert_decision_interval == 3
    assert config.shared_expert_headroom_steps == 7
    assert config.shared_expert_retain_across_requests is True
    assert config.shared_expert_admission_policy == "recall"
