"""Bounded valid CPU copies save D2H without changing GPU residency or weights."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.runtime import LayerKVConfig, LayerKVRuntime

MIB = 1024 * 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("layout", ["individual", "batch"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_install_backing_storage_layout(asynchronous, layout, dtype):
    r = LayerKVRuntime(
        LayerKVConfig(
            expert_transfer_backend="cuda-batch", expert_batch_backing_layout=layout
        )
    )
    weight = torch.arange(32, dtype=dtype, device="cuda").reshape(8, 4)
    if asynchronous:
        copied = {}
        r._copy_experts_to_cpu_for_install_batched_async(
            SimpleNamespace(weight=weight),
            ["weight"],
            [0, 1, 2, 3],
            layer_id=0,
            target_cpu_params=copied,
        )
        r._finalize_expert_d2h_events(block=True)
    else:
        copied = r._copy_experts_to_cpu_for_install_batched(
            SimpleNamespace(weight=weight), ["weight"], [0, 1, 2, 3], layer_id=0
        )
    assert r._expert_host_backing_bytes == 4 * weight[0].nbytes
    for i, params in copied.items():
        torch.testing.assert_close(params["weight"], weight[i].cpu(), rtol=0, atol=0)
    remaining = copied.pop(0)["weight"]
    copied.clear()
    torch.testing.assert_close(remaining, weight[0].cpu())
    assert remaining.untyped_storage().nbytes() == remaining.nbytes * (
        1 if layout == "individual" else 4
    )
    if layout == "individual":
        assert r._tensor_owns_storage(remaining)


def test_pool_budget_reaches_runtime():
    for size in (0, 0.5, 128):
        config = LayerKVConfig.from_server_args(
            SimpleNamespace(layerkv_expert_cpu_backing_pool_mb=size)
        )
        r = LayerKVRuntime(config)
        assert r._expert_cpu_backing_pool_limit_bytes == int(size * MIB)
        assert r._expert_host_budget_summary()["pool_limit_bytes"] == int(size * MIB)


def test_expert_preload_uses_batched_backing_storage():
    r = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            expert_cpu_backing_mode="all",
            expert_batch_backing_layout="batch",
        )
    )
    weights = torch.arange(32, dtype=torch.float16).reshape(4, 8)
    module = SimpleNamespace(w13_weight=weights, weight=weights)
    r._expert_modules = [(0, module)]
    r._expert_param_names = lambda _module: ["weight"]

    r._preload_all_expert_cpu_backing()

    assert r.stats.expert_cpu_backing_preload_count == 4
    assert len(r._expert_global_cpu_backing) == 4
    assert r.stats.expert_cpu_backing_pool_alloc_count == 1
    for expert_id in range(4):
        torch.testing.assert_close(
            r._expert_global_cpu_backing[(0, expert_id)]["weight"],
            weights[expert_id],
        )


@pytest.mark.parametrize(
    "name", ["layerkv_expert_backing_cache_mb", "layerkv_expert_cpu_backing_pool_mb"]
)
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_server_budget_rejected(name, value):
    from sglang.srt.server_args import ServerArgs

    args = SimpleNamespace(
        layerkv_expert_backing_cache_mb=0, layerkv_expert_cpu_backing_pool_mb=256
    )
    setattr(args, name, value)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ServerArgs._handle_layerkv(args)


def test_summary_deduplicates_storage_but_not_logical_copies():
    r = LayerKVRuntime(
        LayerKVConfig(expert_backing_cache_mb=16 / MIB, expert_cpu_backing_pool_mb=0)
    )
    owner = torch.arange(8, dtype=torch.float32)
    r._expert_layers[0] = SimpleNamespace(
        cpu_params={0: {"w": owner[:4]}, 1: {"w": owner[4:]}}, logical_to_slot={0: 0}
    )
    r._expert_host_backing_bytes = 32
    s = r._expert_host_budget_summary()
    assert s["mandatory_valid_bytes"] == s["cached_valid_bytes"] == 16
    assert s["tracked_unique_storage_bytes"] == 32
    assert s["optional_guard_pass"] and s["ledger_matches"]
    r._expert_host_backing_bytes += 1
    assert not r._expert_host_budget_summary()["ledger_matches"]
    r.config.expert_backing_cache_mb = 0
    assert not r._expert_host_budget_summary()["optional_guard_pass"]


def test_summary_deduplicates_pending_failed_and_deferred_owners():
    r = LayerKVRuntime(LayerKVConfig())
    first = torch.zeros(4)
    second = torch.ones(8)
    r._expert_layers[0] = SimpleNamespace(
        cpu_params={0: {"w": first}}, logical_to_slot={}
    )
    r._expert_host_backing_bytes = first.nbytes
    r._expert_batch_transfer = SimpleNamespace(
        pending=[(None, [(first, second)], ())],
        failed_refs=[(first, second[:4])],
        deferred={1: (second, None)},
    )
    s = r._expert_host_budget_summary()
    assert s["tracked_unique_storage_bytes"] == first.nbytes + second.nbytes
    assert s["batch_owner_storage_bytes"] == s["tracked_unique_storage_bytes"]
    assert s["ledger_matches"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cache_experts", [0, 1, 2])
def test_real_materialization_respects_split_budget(dtype, cache_experts):
    from test_cpu_known_prepare import dispatch, setup

    r, state, _ = setup(dtype=dtype)
    r.config.expert_backing_cache_mb = cache_experts * state.expert_bytes / MIB
    r._expert_cpu_backing_pool_limit_bytes = (2 - cache_experts) * state.expert_bytes
    r.config.expert_backing_release_validation = "admission"
    copies = []
    mappings = []
    for ids in ([0, 1], [2, 3], [4, 5]) * 3:
        result = r._prepare_expert_dispatch_for_core(
            state, dispatch([ids], dtype), known_logical_ids=ids
        )
        torch.testing.assert_close(
            state.module.weight[result.topk_output.topk_ids].cpu(),
            torch.tensor([[[i + 1, i + 1] for i in ids]], dtype=dtype),
            rtol=0,
            atol=0,
        )
        r._finalize_expert_materialize_events(block=True)
        s = r._expert_host_budget_summary()
        assert s["ledger_matches"] and s["optional_guard_pass"]
        assert s["optional_limit_bytes"] == 2 * state.expert_bytes
        assert s["mandatory_valid_bytes"] == 4 * state.expert_bytes
        assert s["batch_owner_storage_bytes"] == 0
        assert s["tracked_unique_storage_bytes"] <= 6 * state.expert_bytes
        copies.append(r.stats.expert_copy_descriptor_d2h_count)
        mappings.append(dict(state.logical_to_slot))
    assert r.stats.expert_materialize_count == 18
    control, control_state, _ = setup(dtype=dtype)
    control_mappings = []
    for ids in ([0, 1], [2, 3], [4, 5]) * 3:
        control._prepare_expert_dispatch_for_core(
            control_state, dispatch([ids], dtype), known_logical_ids=ids
        )
        control_mappings.append(dict(control_state.logical_to_slot))
    control._finalize_expert_materialize_events(block=True)
    assert mappings == control_mappings
    if cache_experts == 2:
        assert copies[-1] == copies[0] == 2
        assert r.stats.expert_eviction_d2h_skip_count == 16
    elif cache_experts == 0:
        assert copies[-1] == 18
        assert r.stats.expert_eviction_d2h_skip_count == 0
    else:
        assert 2 < copies[-1] < 18


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cached", [False, True])
def test_recall_tail_reuses_valid_copy_without_double_accounting(monkeypatch, cached):
    from test_cpu_known_prepare import setup

    r, state, controller = setup()
    controller.base_slots = 1
    controller.arena = SimpleNamespace(loans=[object()], recall=lambda: None)
    old = state.module.weight
    controller.expert_allocations[old.data_ptr()] = SimpleNamespace(
        tensor=lambda shape: old[:1]
    )
    monkeypatch.setattr(
        controller, "_set_capacity", lambda n: setattr(state, "slot_capacity", n)
    )
    monkeypatch.setattr(r, "_refresh_expert_stats", lambda: None)
    monkeypatch.setattr(r, "_refresh_per_layer_allocator_stats", lambda: None)
    if cached:
        state.cpu_params[5] = {"weight": old[1].cpu().pin_memory()}
        r._expert_host_backing_bytes += state.expert_bytes
    previous = state.cpu_params.get(5)
    controller.recall()
    assert state.logical_to_slot == {4: 0}
    assert state.remap_tensor.tolist() == [-1, -1, -1, -1, 0, -1]
    assert state.slot_capacity == 1
    assert r.stats.expert_copy_descriptor_d2h_count == (0 if cached else 1)
    if cached:
        assert state.cpu_params[5] is previous
    torch.testing.assert_close(
        state.cpu_params[5]["weight"], torch.full((2,), 6, dtype=old.dtype)
    )
    s = r._expert_host_budget_summary()
    assert s["ledger_matches"] and s["optional_guard_pass"]
    assert s["mandatory_valid_bytes"] == 5 * state.expert_bytes
