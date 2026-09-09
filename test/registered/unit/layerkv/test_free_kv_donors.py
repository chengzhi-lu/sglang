"""Physical expert backing must subtract real KV credit and return it on recall."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.config_stats import LayerKVConfig
from sglang.srt.layerkv.runtime import LayerKVRuntime
from sglang.srt.layerkv.shared_expert import SharedExpertController
from sglang.srt.layerkv.shared_vmm import SharedVMM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def setup_free_pool(dtype, pages=3):
    arena = SharedVMM("cuda")
    tokens = arena.page_bytes // 1024
    buffers = [
        arena.kv_zeros((tokens * pages, 1024), dtype=torch.uint8, device="cuda")
        for _ in range(2)
    ]
    r = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            kvc_backend="per-layer-arena",
            shared_expert_layer=3,
            shared_expert_initial_slots=1,
            shared_expert_extra_slots=1,
            shared_expert_free_kv_donors=True,
        )
    )
    r.physical_kvc_supported = True
    r._kvc_layer_ids = lambda: [3]
    r._kv_pool = SimpleNamespace(
        _get_key_buffer=lambda _: buffers[0], _get_value_buffer=lambda _: buffers[1]
    )
    r._refresh_expert_stats = lambda: None
    locs = list(range(tokens, pages * tokens))
    r._per_layer_arena_reserved_locs = set(locs)
    r._per_layer_arena_free_locs[3] = list(locs)
    c = SharedExpertController(r, arena)
    params = {}
    for name, pages in [("w13_weight", 2), ("w2_weight", 1)]:
        old = torch.empty(
            (3, pages * arena.page_bytes // dtype.itemsize), dtype=dtype, device="meta"
        )
        params[name] = torch.nn.Parameter(
            c.allocate_expert(old, 1).fill_(2), requires_grad=False
        )
    module = SimpleNamespace(
        **params,
        num_experts=1,
        moe_runner_config=SimpleNamespace(num_experts=1),
        dispatcher=SimpleNamespace(num_experts=1)
    )
    c.state = SimpleNamespace(
        module=module,
        full_num_experts=3,
        slot_capacity=1,
        param_names=list(params),
        free_slots=[],
        logical_to_slot={0: 0},
        slot_to_logical={0: 0},
        lru={0: 0},
        lru_heap=[],
        cpu_params={},
        device=torch.device("cuda"),
    )
    return r, c, buffers, locs


def credit(r):
    return r.get_scheduler_admission_credit_tokens(reason="decode_prealloc_admission")


@pytest.mark.parametrize("retain", [False, True])
def test_request_cleanup_preserves_only_explicitly_retained_expert_loans(retain):
    from sglang.srt.layerkv.common_types import _LayerKVResidencyEntry

    r, c, _buffers, locs = setup_free_pool(torch.float16, pages=5)
    r._shared_expert = c
    r.config.shared_expert_retain_across_requests = retain
    # This request owns a location outside the pages offered as donors.
    r._per_layer_arena_reserved_locs.add(1)
    r._per_layer_arena_allocated_locs.setdefault(3, set()).add(1)
    entry = _LayerKVResidencyEntry(4, 0, "resident", layer_id=3, device_loc=1)
    r._track_per_layer_cleanup_entry(entry)
    c.after_decode()
    blocked = set(c.blocked[3])
    assert 1 not in blocked and blocked
    r.on_request_finished(SimpleNamespace(req_pool_idx=4))
    assert 4 not in r._per_layer_cleanup_state_by_req
    assert 1 in r._common_per_layer_reusable_locs()
    assert bool(c.arena.loans) == retain
    if retain:
        assert not blocked.intersection(r._common_per_layer_reusable_locs())
        r.config.shared_expert_admission_policy = "retain"
        assert r.recall_shared_expert_for_admission(shortage_tokens=1) == 0
        assert c.arena.loans
        r.config.shared_expert_admission_policy = "recall"
        assert r.recall_shared_expert_for_admission(shortage_tokens=1) == len(blocked)
    assert credit(r) == len(locs) + 1
    assert c.arena.summary()["ownership_guard_pass"]


@pytest.mark.parametrize("retain", [False, True])
def test_prefill_boundary_retains_experts_only_when_enabled(retain):
    r, c, _buffers, _locs = setup_free_pool(torch.bfloat16)
    r._shared_expert = c
    r.config.shared_expert_retain_across_requests = retain
    c.after_decode()

    def boundary():
        raise RuntimeError("reached prefill bookkeeping")

    r._refresh_per_layer_kvc_overrides = boundary
    with pytest.raises(RuntimeError, match="reached prefill bookkeeping"):
        r.on_forward_begin(mode="extend", forward_batch=SimpleNamespace())
    assert bool(c.arena.loans) == retain
    c.recall()


def test_donor_selection_reuses_claimed_addresses_before_spending_more_kv_credit():
    r, c, _buffers, locs = setup_free_pool(torch.float16, pages=5)
    c.after_decode()
    tokens_per_page = c.arena.page_bytes // 1024
    assert len(c.arena.loans) == 3
    # K and V pages at the same locations cost common KV capacity only once.
    assert credit(r) == len(locs) - 2 * tokens_per_page
    c.recall()
    assert credit(r) == len(locs)


def test_virtual_scratch_loan_returns_without_publishing_arena_credit():
    r, c, _buffers, locs = setup_free_pool(torch.float16, pages=5)
    r.config.shared_expert_lend_virtual_scratch = True
    tokens_per_page = c.arena.page_bytes // 1024
    scratch = set(range(tokens_per_page, 2 * tokens_per_page))
    r._virtual_scratch_locs = torch.tensor(
        sorted(scratch), dtype=torch.int64, device="cuda"
    )
    r._virtual_scratch_locs_host = scratch
    # The private scratch page is mapped by the shared KV VMM but is not
    # ordinary per-layer arena ownership.
    r._per_layer_arena_reserved_locs.difference_update(scratch)
    r._per_layer_arena_free_locs[3] = [loc for loc in locs if loc not in scratch]
    before_credit = credit(r)

    c.after_decode()

    assert c.scratch_blocked
    assert c.scratch_current_loan_page_count == 2
    assert c.blocked[3].isdisjoint(scratch)
    assert scratch.isdisjoint(r._common_per_layer_reusable_locs())
    assert credit(r) < before_credit
    c.recall()
    assert not c.scratch_blocked and c.scratch_current_loan_page_count == 0
    assert c.scratch_recall_count == 1
    assert credit(r) == before_credit
    assert c.arena.summary()["ownership_guard_pass"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batched", [False, True])
def test_distinct_layers_lend_same_locations_without_duplicate_kv_credit(
    dtype, batched
):
    r, c, buffers, locs = setup_free_pool(dtype)
    by_layer = {3: buffers}
    for layer in (7, 11, 15):
        by_layer[layer] = [
            c.arena.kv_zeros(tuple(x.shape), dtype=x.dtype, device="cuda")
            for x in buffers
        ]
        r._per_layer_arena_free_locs[layer] = list(locs)
    r._kvc_layer_ids = lambda: [3, 7, 11, 15]
    r._kv_pool._get_key_buffer = lambda layer: by_layer[layer][0]
    r._kv_pool._get_value_buffer = lambda layer: by_layer[layer][1]
    r._mark_per_layer_common_free_dirty()
    c.extra_slots = 2
    if batched:
        from sglang.srt.layerkv.residency_budget import ResidencyBudget

        c.budget = ResidencyBudget(1, 3, interval=2, headroom_steps=1)
        c.budget.target = 3
        r._last_forward_batch = SimpleNamespace(batch_size=1)
    before = c.arena.summary()
    assert credit(r) == len(locs)
    c.after_decode()
    if not batched:
        assert c.state.slot_capacity == 2 and credit(r) == len(locs) // 2
        c.after_decode()
    assert c.state.slot_capacity == 3 and len(c.arena.loans) == 6
    assert c.donor_scan_count == c.grow_count == (1 if batched else 2)
    assert sorted(c.state.free_slots) == [1, 2]
    assert credit(r) == len(locs) // 2  # no additional common token cost
    assert all(values == set(locs[: len(locs) // 2]) for values in c.blocked.values())
    c.recall()
    assert credit(r) == len(locs)  # not twice the common token capacity
    after = c.arena.summary()
    assert before["physical_bytes"] == after["physical_bytes"]
    assert before["physical_create_count"] == after["physical_create_count"]
    assert after["ownership_guard_pass"] and not c.arena.loans


def test_donor_path_does_not_publish_pending_kv_evictions():
    r, c, _buffers, _locs = setup_free_pool(torch.float16)

    def unexpected_finalize(**_kwargs):
        raise AssertionError("donor path must not publish KV eviction metadata")

    r._finalize_kvc_evictions = unexpected_finalize
    c.after_decode()
    assert len(c.arena.loans) == 3
    c.recall()
    assert c.arena.summary()["ownership_guard_pass"]


@pytest.mark.parametrize("decision_due", [False, True])
def test_budget_warmup_without_loans_does_not_scan_allocator(decision_due):
    from sglang.srt.layerkv.residency_budget import ResidencyBudget

    r, c, _buffers, _locs = setup_free_pool(torch.float16)
    c.budget = ResidencyBudget(1, 2)
    if decision_due:
        for step in range(1, 9):
            c.budget.observe(step, [0])
        r._decode_step = 8

    def unexpected():
        raise AssertionError("no profitable loan-free budget action needs KV discovery")

    r._common_per_layer_reusable_locs = unexpected
    c.after_decode()
    assert c.donor_scan_count == 0


def test_measured_cost_gate_prevents_physical_growth_and_donor_scan():
    from sglang.srt.layerkv.residency_budget import ResidencyBudget

    r, c, _buffers, _locs = setup_free_pool(torch.float16)
    c.budget = ResidencyBudget(1, 2, interval=2, headroom_steps=1)
    for step, expert in enumerate([0, 1, 0, 1], 1):
        c.budget.observe(step, [expert])
    c.last_growth_ms = 50.0
    r.stats.expert_materialize_count = 4
    r.stats.expert_materialize_ms = 4.0
    r._last_forward_batch = SimpleNamespace(batch_size=1)
    r._decode_step = 4
    c.after_decode()
    assert c.budget.reason == "cost-exceeds-benefit"
    assert c.budget.cost_rejections == 1
    assert c.donor_scan_count == 0 and not c.arena.loans


def test_deferred_headroom_rejects_zero_cost_donors_without_kv_capacity():
    from sglang.srt.layerkv.residency_budget import ResidencyBudget

    r, c, _buffers, locs = setup_free_pool(torch.float16)
    r._kvc_layer_ids = lambda: [3, 7]
    r._per_layer_arena_allocated_locs[7] = set(locs)
    c.budget = ResidencyBudget(1, 2, interval=2, headroom_steps=1)
    c.budget.target = 2
    r._last_forward_batch = SimpleNamespace(batch_size=1)
    assert credit(r) == 0
    c.after_decode()
    assert not c.arena.loans and c.state.slot_capacity == 1
    assert c.budget_headroom_skips == 1


def test_adaptive_policy_preserves_post_loan_headroom_and_recalls_pressure():
    from sglang.srt.layerkv.residency_budget import ResidencyBudget

    r, c, _buffers, locs = setup_free_pool(torch.float16)
    c.budget = ResidencyBudget(1, 2, interval=2, headroom_steps=1)
    r._last_forward_batch = SimpleNamespace(batch_size=1)
    c.budget.target = 2
    r._decode_step = 1
    # Borrowing three pages from this small pool consumes ALL common capacity.
    c.after_decode()
    assert not c.arena.loans and c.budget_headroom_skips == 1
    assert credit(r) == len(locs)
    # Add one ordinary reserved KV location outside the eligible whole pages.
    r._per_layer_arena_reserved_locs.add(1)
    r._per_layer_arena_free_locs[3].append(1)
    r._mark_per_layer_common_free_dirty()
    r._decode_step = 3
    c.after_decode()
    assert len(c.arena.loans) == 3 and credit(r) == 1
    r._last_forward_batch = SimpleNamespace(batch_size=2)
    r._decode_step = 4
    c.after_decode()
    assert not c.arena.loans and credit(r) == len(locs) + 1
    assert c.budget.target == 1 and c.arena.summary()["ownership_guard_pass"]


def test_adaptive_headroom_counts_unreserved_native_capacity():
    from sglang.srt.layerkv.residency_budget import ResidencyBudget

    r, c, _buffers, _locs = setup_free_pool(torch.float16)
    c.extra_slots = 2
    c.budget = ResidencyBudget(1, 3, interval=2, headroom_steps=1)
    c.budget.target = 3
    r._last_forward_batch = SimpleNamespace(batch_size=1)
    r._decode_step = 1
    r._allocator = SimpleNamespace(available_size=lambda: 1)
    r._mark_per_layer_common_free_dirty()
    # The loan consumes common arena credit, but one distinct native token
    # remains available. Reserving it again would double-count capacity.
    c.after_decode()
    assert len(c.arena.loans) == 3
    assert c.state.slot_capacity == 2  # four donor pages only cover one full row
    c.recall()


@pytest.mark.parametrize("fail", [False, True])
def test_batch_growth_late_mapping_failure_is_atomic(monkeypatch, fail):
    from sglang.srt.layerkv.residency_budget import ResidencyBudget

    r, c, _buffers, locs = setup_free_pool(torch.float16, pages=5)
    c.extra_slots = 2
    c.budget = ResidencyBudget(1, 3, interval=2, headroom_steps=1)
    c.budget.target = 3
    r._last_forward_batch = SimpleNamespace(batch_size=1)
    target = next(iter(c.expert_allocations.values()))
    original_map = target.map

    def mapped(page, handle):
        if fail and page == 5:
            raise RuntimeError("injected late batch mapping failure")
        return original_map(page, handle)

    monkeypatch.setattr(target, "map", mapped)
    before = c.arena.summary()
    if fail:
        with pytest.raises(RuntimeError, match="late batch"):
            c.after_decode()
        assert c.state.slot_capacity == 1 and c.state.free_slots == []
        assert not c.arena.loans and credit(r) == len(locs)
    else:
        c.after_decode()
        assert c.state.slot_capacity == 3 and len(c.arena.loans) == 6
        assert sorted(c.state.free_slots) == [1, 2]
        c.recall()
        assert credit(r) == len(locs)
    after = c.arena.summary()
    assert after["physical_bytes"] == before["physical_bytes"]
    assert after["physical_create_count"] == before["physical_create_count"]
    assert after["ownership_guard_pass"]


def test_admission_recall_restores_real_kv_without_new_backing():
    r, c, _buffers, locs = setup_free_pool(torch.float16)
    r._shared_expert = c
    before = c.arena.summary()
    c.after_decode()
    assert credit(r) == 0 and len(c.arena.loans) == 3
    assert r.recall_shared_expert_for_admission(shortage_tokens=0) == 0
    assert len(c.arena.loans) == 3
    assert r.recall_shared_expert_for_admission(shortage_tokens=1) == len(locs)
    assert credit(r) == len(locs)
    assert r.stats.shared_expert_admission_recall_count == 1
    assert r.stats.shared_expert_admission_recovered_tokens == len(locs)
    assert r.recall_shared_expert_for_admission(shortage_tokens=1) == 0
    after = c.arena.summary()
    assert after["physical_bytes"] == before["physical_bytes"]
    assert after["physical_create_count"] == before["physical_create_count"]
    assert after["ownership_guard_pass"] and not c.arena.loans


def test_admission_recall_is_not_disabled_by_free_donor_growth_flag():
    r, c, _buffers, locs = setup_free_pool(torch.float16)
    r._shared_expert = c
    c.after_decode()
    assert len(c.arena.loans) == 3
    r.config.shared_expert_free_kv_donors = False

    assert r.recall_shared_expert_for_admission(shortage_tokens=1) == len(locs)
    assert not c.arena.loans


def test_prefill_admission_uses_recalled_cuda_pages():
    from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder

    r, c, _buffers, locs = setup_free_pool(torch.float16)
    r._shared_expert = c
    c.after_decode()
    assert credit(r) == 0
    adder = PrefillAdder.__new__(PrefillAdder)
    adder.token_to_kv_pool_allocator = SimpleNamespace(
        available_size=lambda: 0,
        get_kvcache=lambda: SimpleNamespace(layerkv_runtime=r),
    )
    adder.tree_cache = SimpleNamespace(evictable_size=lambda: 0)
    adder.is_hybrid_swa = adder.is_hybrid_ssm_cache = False
    adder.cur_rem_token_offset = adder.rem_total_token_offset = 0
    adder.page_size = 1
    adder.req_states = adder.running_batch = None
    adder.can_run_list = []
    adder.prefill_delayer_single_pass = adder.dllm_config = None
    adder.rem_chunk_tokens = None
    adder._update_prefill_budget = lambda *_args: None
    adder.budget_state = lambda: AddReqResult.CONTINUE
    req = SimpleNamespace(
        extend_input_len=100,
        origin_input_ids=list(range(100)),
        output_ids=[],
        sampling_params=SimpleNamespace(ignore_eos=True, max_new_tokens=10),
    )
    assert adder.add_one_req_ignore_eos(req) == AddReqResult.CONTINUE
    assert adder.can_run_list == [req]
    assert not c.arena.loans and credit(r) == len(locs)
    assert r.stats.shared_expert_admission_recall_count == 1


def test_real_admission_pressure_prevents_reborrowing_until_queue_drains():
    r, c, _buffers, _locs = setup_free_pool(torch.float16)
    r._shared_expert = c
    c.after_decode()
    assert c.arena.loans
    r.stats.native_schedule_waiting_queue_len = 1
    r.recall_shared_expert_for_admission(shortage_tokens=1)
    scans = c.donor_scan_count
    c.after_decode()
    assert not c.arena.loans and c.donor_scan_count == scans
    assert c.admission_pressure_skip_count == 1
    r.stats.native_schedule_waiting_queue_len = 0
    c.after_decode()
    assert c.arena.loans and not c.admission_shortage_tokens
    c.recall()


def test_pressure_skip_counter_excludes_already_satisfied_target():
    r, c, _buffers, _locs = setup_free_pool(torch.float16)
    c.extra_slots = 0
    c.admission_shortage_tokens = 1
    r.stats.native_schedule_waiting_queue_len = 1
    c.after_decode()
    assert c.admission_pressure_skip_count == 0
    assert c.donor_scan_count == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unused_kv_expert_backing_recall_restores_scheduler_credit(dtype):
    r, c, buffers, locs = setup_free_pool(dtype)
    before = c.arena.summary()
    pointers = [t.data_ptr() for t in buffers]
    for _ in range(2):
        assert credit(r) == len(locs)
        c.after_decode()
        assert c.state.slot_capacity == 2
        assert len(c.arena.loans) == 3
        assert credit(r) == 0
        assert not r._common_per_layer_reusable_locs()
        for name in c.state.param_names:
            getattr(c.state.module, name).data[1].fill_(5)
        assert (
            not r._per_layer_offloaded_runs_by_req_layer
        )  # No fictitious CPU KV backing.
        c.recall()
        assert credit(r) == len(locs)
        assert c.state.slot_capacity == 1 and not c.blocked
        for t in buffers:
            t[locs[0] : locs[-1] + 1].fill_(7)
            assert torch.all(t[locs[0] : locs[-1] + 1] == 7)
        assert [t.data_ptr() for t in buffers] == pointers
        after = c.arena.summary()
        assert after["physical_create_count"] == before["physical_create_count"]
        assert after["physical_bytes"] == before["physical_bytes"]
        assert after["ownership_guard_pass"]


@pytest.mark.parametrize(
    "excluded", ["allocated", "protected", "unreserved", "cleanup"]
)
def test_live_or_unowned_page_is_not_lent(excluded):
    r, c, buffers, locs = setup_free_pool(torch.bfloat16)
    bad = locs[0]
    if excluded == "unreserved":
        r._per_layer_arena_reserved_locs.remove(bad)
    elif excluded == "cleanup":
        r._per_layer_cleanup_state_by_req[9] = SimpleNamespace(
            loc_bits_by_layer={3: 1 << bad}
        )
    else:
        getattr(r, "_per_layer_arena_" + excluded + "_locs")[3] = {bad}
    c.after_decode()
    assert c.state.slot_capacity == 1 and not c.arena.loans
    assert c.arena.summary()["ownership_guard_pass"]


def test_map_failure_returns_claims_and_credit(monkeypatch):
    r, c, buffers, locs = setup_free_pool(torch.bfloat16)
    before = credit(r)
    target = next(iter(c.expert_allocations.values()))
    original = target.map

    def fail(page, handle):
        if page == 3:
            raise RuntimeError("injected free-donor map failure")
        return original(page, handle)

    monkeypatch.setattr(target, "map", fail)
    with pytest.raises(RuntimeError, match="injected free-donor"):
        c.after_decode()
    assert credit(r) == before and not c.arena.loans
    assert not r._per_layer_arena_allocated_locs[3]
    assert c.arena.summary()["ownership_guard_pass"]


def test_claim_rejects_missing_address_without_partial_consumption():
    r = LayerKVRuntime(LayerKVConfig())
    r._per_layer_arena_reserved_locs.update([1, 2, 3])
    r._per_layer_arena_free_locs[0] = [1]
    r._push_per_layer_overwrite_bits(0, 1 << 2)
    assert not r._claim_per_layer_shared_donor_locs(0, [1, 2, 3])
    assert r._per_layer_arena_free_locs[0] == [1]
    assert r._per_layer_pending_overwrite_bits_union(0) == 1 << 2
    assert r._claim_per_layer_shared_donor_locs(0, [1, 2])
    assert not r._per_layer_arena_free_locs[0]
    assert not r._per_layer_pending_overwrite_bits_union(0)
    assert not r._claim_per_layer_shared_donor_locs(0, [1, 2])
