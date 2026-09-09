from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.runtime import LayerKVConfig, LayerKVRuntime


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA pinned transfers"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_selected_snapshot_is_pinned_immutable_scoped_and_counted_once(dtype):
    r = LayerKVRuntime(
        LayerKVConfig(expert_cpu_backing_mode="selected", shared_expert_layer=3)
    )

    def module():
        return SimpleNamespace(
            w13_weight=torch.arange(24, device="cuda", dtype=dtype).reshape(3, 4, 2),
            w2_weight=torch.ones((3, 4, 2), device="cuda", dtype=dtype),
        )

    selected, other = module(), module()
    r._expert_modules = [(1, other), (3, selected)]
    r._expert_param_names = lambda _: ["w13_weight", "w2_weight"]
    r._preload_all_expert_cpu_backing()
    assert set(r._expert_global_cpu_backing) == {(3, i) for i in range(3)}
    assert r.stats.expert_cpu_backing_preload_count == 3
    params = r._global_expert_backing(3, 0)
    assert all(t.is_pinned() for t in params.values())
    expected = selected.w13_weight[0].cpu()
    selected.w13_weight.zero_()
    torch.testing.assert_close(params["w13_weight"], expected, rtol=0, atol=0)
    r._preload_all_expert_cpu_backing()
    assert r._global_expert_backing(3, 0) is params
    # A resident state alias does not turn permanent backing into optional cache.
    r._expert_layers[3] = SimpleNamespace(
        layer_id=3, cpu_params={0: params}, logical_to_slot={0: 0}
    )
    report = r._expert_host_budget_summary()
    assert report["immutable_backing_bytes"] == 96
    assert report["tracked_unique_storage_bytes"] == 96
    assert report["mandatory_valid_bytes"] == report["cached_valid_bytes"] == 0
    assert report["ledger_matches"] and report["optional_guard_pass"]
    # Dropping a state-local cache reference cannot release the permanent copy.
    r._expert_layers[3].cpu_params.clear()
    assert r._global_expert_backing(3, 0) is params
    assert r._expert_host_budget_summary()["tracked_unique_storage_bytes"] == 96


def test_missing_selected_layer_rejected_before_any_copy():
    r = LayerKVRuntime(
        LayerKVConfig(expert_cpu_backing_mode="selected", shared_expert_layer=3)
    )
    r._expert_modules = [(1, object())]
    with pytest.raises(ValueError, match="exactly one"):
        r._preload_all_expert_cpu_backing()
    assert not r._expert_global_cpu_backing
