"""Incremental resident backing accounting preserves the full-scan policy."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.runtime import LayerKVConfig, LayerKVRuntime

MIB = 1024 * 1024


def test_server_arguments_reach_config():
    c = LayerKVConfig.from_server_args(
        SimpleNamespace(
            layerkv_expert_backing_cache_accounting="incremental",
            layerkv_expert_backing_cache_accounting_check=True,
        )
    )
    assert c.expert_backing_cache_accounting == "incremental"
    assert c.expert_backing_cache_accounting_check


@pytest.mark.parametrize("case", ["mode", "scope", "check"])
def test_invalid_accounting_arguments_rejected(case):
    from sglang.srt.server_args import ServerArgs

    args = SimpleNamespace(
        layerkv_expert_backing_cache_mb=128,
        layerkv_expert_cpu_backing_pool_mb=128,
        layerkv_expert_backing_cache_accounting="incremental",
        layerkv_expert_backing_cache_accounting_check=False,
        layerkv_shared_expert_layer=0,
    )
    if case == "mode":
        args.layerkv_expert_backing_cache_accounting = "invalid"
    elif case == "scope":
        args.layerkv_shared_expert_layer = -1
    else:
        args.layerkv_expert_backing_cache_accounting = "scan"
        args.layerkv_expert_backing_cache_accounting_check = True
    with pytest.raises(ValueError, match="accounting"):
        ServerArgs._handle_layerkv(args)


def cpu_runtime(mode="incremental", budget=32):
    r = LayerKVRuntime(
        LayerKVConfig(
            expert_backing_cache_accounting=mode,
            expert_backing_cache_mb=budget / MIB,
            expert_cpu_backing_pool_mb=0,
        )
    )
    state = SimpleNamespace(
        layer_id=0,
        cpu_params={i: {"weight": torch.zeros(i + 1)} for i in range(6)},
        logical_to_slot={4: 0, 5: 1},
        backing_lru={4: 0, 5: 0},
        backing_cache_accounted_bytes=0,
        backing_cache_accounted_by_id=None,
    )
    r._expert_layers[0] = state
    r._shared_expert = SimpleNamespace(state=state)
    r._expert_host_backing_bytes = sum(
        r._expert_backing_bytes(p) for p in state.cpu_params.values()
    )
    return r, state


def test_no_budget_overflow_does_not_rescan(monkeypatch):
    r, state = cpu_runtime(budget=64)
    r._get_expert_backing_cache_accounting(state)
    monkeypatch.setattr(
        r,
        "_expert_backing_bytes",
        lambda _: pytest.fail("recomputed unchanged backing bytes"),
    )
    for _ in range(10):
        r._trim_expert_backing_cache(state)
    assert state.backing_cache_accounted_bytes == 44
    assert r.stats.expert_backing_cache_fast_return_count == 10


@pytest.mark.parametrize("budget", [0, 1, 20, 24, 32, 44, 64])
def test_partial_budget_and_lru_ties_match_scan(budget):
    results = []
    for mode in ("scan", "incremental"):
        r, state = cpu_runtime(mode, budget)
        r._trim_expert_backing_cache(state)
        s = r._expert_host_budget_summary()
        assert s["ledger_matches"] and s["optional_guard_pass"]
        assert s["cache_accounting_matches"]
        assert set(range(4)) <= state.cpu_params.keys()
        results.append(
            (
                set(state.cpu_params),
                dict(state.backing_lru),
                s["cached_valid_bytes"],
                r._expert_host_backing_bytes,
            )
        )
    assert results[0] == results[1]


@pytest.mark.parametrize("corruption", ["total", "entries", "untracked_mutation"])
def test_debug_check_rejects_accounting_drift(corruption):
    r, state = cpu_runtime(budget=64)
    r._get_expert_backing_cache_accounting(state)
    if corruption == "total":
        state.backing_cache_accounted_bytes += 1
    elif corruption == "entries":
        state.backing_cache_accounted_by_id[4] += 1
    else:
        state.logical_to_slot.pop(4)
    assert not r._expert_host_budget_summary()["cache_accounting_matches"]
    r.config.expert_backing_cache_accounting_check = True
    with pytest.raises(RuntimeError, match="cache accounting"):
        r._trim_expert_backing_cache(state)
    assert not r.stats.expert_guard_pass and not r.stats.comparable


@pytest.mark.parametrize("case", ["not_shared", "multi_layer", "global"])
def test_outside_shared_scope_falls_back_to_scan(case):
    r, state = cpu_runtime()
    if case == "not_shared":
        r._shared_expert = None
    elif case == "multi_layer":
        r._expert_layers[1] = SimpleNamespace(cpu_params={}, logical_to_slot={})
    else:
        r._expert_global_cpu_backing[(0, 0)] = state.cpu_params[0]
    assert r._get_expert_backing_cache_accounting(state) is None
    r._trim_expert_backing_cache(state)
    assert state.backing_cache_accounted_by_id is None


def test_leaving_scope_discards_ledger_and_reentry_rebuilds():
    r, state = cpu_runtime(budget=64)
    r._get_expert_backing_cache_accounting(state)
    shared = r._shared_expert
    r._shared_expert = None
    assert r._get_expert_backing_cache_accounting(state) is None
    assert state.backing_cache_accounted_by_id is None
    state.logical_to_slot.pop(4)
    r._shared_expert = shared
    r._trim_expert_backing_cache(state)
    assert state.backing_cache_accounted_by_id == {5: 24}
    assert state.backing_cache_accounted_bytes == 24


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_async_backing_publication_updates_resident_ledger(dtype):
    from test_cpu_known_prepare import setup

    r, state, _ = setup(dtype=dtype)
    r.config.expert_backing_cache_accounting = "incremental"
    r.config.expert_backing_cache_mb = 1
    r._get_expert_backing_cache_accounting(state)
    assert state.backing_cache_accounted_bytes == 0
    r._expert_d2h_stream = torch.cuda.Stream()
    r._expert_d2h_stream.wait_stream(torch.cuda.current_stream())
    assert r._copy_slots_to_cpu_batched_async(state, [(4, 0)])
    r._finalize_expert_d2h_events(block=True)
    r._finalize_expert_materialize_events(block=True)
    s = r._expert_host_budget_summary()
    assert state.backing_cache_accounted_bytes == state.expert_bytes
    assert s["ledger_matches"] and s["cache_accounting_matches"]
    torch.testing.assert_close(
        state.cpu_params[4]["weight"], state.module.weight[0].cpu(), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cache_experts", [0, 1, 2])
def test_real_transfers_and_recall_match_scan(monkeypatch, dtype, cache_experts):
    from test_cpu_known_prepare import dispatch, setup

    results = []
    for mode in ("scan", "incremental"):
        r, state, controller = setup(dtype=dtype)
        r.config.expert_backing_cache_accounting = mode
        r.config.expert_backing_cache_accounting_check = mode == "incremental"
        r.config.expert_backing_cache_mb = cache_experts * state.expert_bytes / MIB
        mappings = []
        for ids in ([0, 1], [2, 3], [4, 5]) * 3:
            value = r._prepare_expert_dispatch_for_core(
                state, dispatch([ids], dtype), known_logical_ids=ids
            )
            expected = torch.tensor([[[i + 1, i + 1] for i in ids]], dtype=dtype)
            torch.testing.assert_close(
                state.module.weight[value.topk_output.topk_ids].cpu(),
                expected,
                rtol=0,
                atol=0,
            )
            r._finalize_expert_materialize_events(block=True)
            host = r._expert_host_budget_summary()
            assert (
                host["cache_accounting_matches"]
                and host["ledger_matches"]
                and host["optional_guard_pass"]
            )
            mappings.append(dict(state.logical_to_slot))
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
        controller.recall()
        host = r._expert_host_budget_summary()
        assert host["cache_accounting_matches"] and host["ledger_matches"]
        results.append(
            (
                mappings,
                dict(state.logical_to_slot),
                r.stats.expert_materialize_count,
                r.stats.expert_cuda_batch_h2d_count,
                r.stats.expert_cuda_batch_d2h_count,
                r._expert_host_backing_bytes,
            )
        )
    assert results[0] == results[1]
