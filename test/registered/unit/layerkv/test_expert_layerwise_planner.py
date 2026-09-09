from types import SimpleNamespace

from sglang.srt.layerkv.expert_planner import LayerKVExpertPlannerMixin


def test_layerwise_dp_can_leave_cold_layers_unevicted():
    """The reclaim target must not imply one eviction per expert layer."""

    mib = 1024 * 1024
    candidates = []
    for cost, expert_id in ((10.0, 0), (11.0, 1), (12.0, 2)):
        candidates.append((cost, mib, 0, expert_id, 0.0, 0.0, 0.0))
    for cost, expert_id in ((0.1, 0), (0.2, 1), (0.3, 2)):
        candidates.append((cost, mib, 1, expert_id, 0.0, 0.0, 0.0))

    precomputed = (
        {0: 4, 1: 4},
        {0: 1, 1: 1},
        {0: mib, 1: mib},
        candidates,
    )
    harness = object()
    table = LayerKVExpertPlannerMixin._build_coresid_expert_layerwise_plan_table(
        harness, precomputed, max_reclaim_mb=2.0
    )
    (
        _cost,
        _calls,
        _churn_mb,
        _install_mb,
        _backing,
        _materialize,
        capacities,
        _layer_costs,
    ) = LayerKVExpertPlannerMixin._lookup_coresid_expert_layerwise_cost(
        harness, 2.0, table
    )

    assert capacities == {0: 4, 1: 2}


def test_generic_expert_policies_use_layerwise_cost_planner():
    harness = SimpleNamespace(config=SimpleNamespace(policy="kv-first"))
    assert LayerKVExpertPlannerMixin._layerwise_expert_plan_enabled(harness)

    harness.config.policy = "ratio-50-50"
    assert LayerKVExpertPlannerMixin._layerwise_expert_plan_enabled(harness)

    harness.config.policy = "coresid"
    assert LayerKVExpertPlannerMixin._layerwise_expert_plan_enabled(harness)

    harness.config.policy = "expert-first"
    assert not LayerKVExpertPlannerMixin._layerwise_expert_plan_enabled(harness)
