"""Bounded per-layer integration of SharedVMM with LayerKV.

Only backed-offloaded or wholly unused, currently free KV pages may be lent. Loans are
excluded from the token allocator until recalled. Prefill/request completion
recalls loans, evicting expert tail slots first. This path is synchronous and
opt-in; the normal expert planner and default allocation path are unchanged.
``SharedExpertManager`` composes one controller per MoE layer over a common
SharedVMM arena when all-layer mode is enabled.

When explicitly enabled, complete pages owned by the virtual-KVC scratch
reservation may also be lent while that scratch is idle. Scratch loans use a
separate ownership ledger: they never become ordinary per-layer allocator
credit, and any KVC use recalls the whole expert tail before touching scratch.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
import heapq
import math
import time
from contextlib import contextmanager

import torch

from .residency_budget import group_token_experts

try:
    from sglang.jit_kernel.layerkv_expert_group import (
        layerkv_group_token_experts,
        layerkv_group_token_experts_multi,
    )
except Exception:  # pragma: no cover - optional CUDA JIT helper.
    layerkv_group_token_experts = None
    layerkv_group_token_experts_multi = None


class SharedExpertController:
    def __init__(self, runtime, arena, *, layer_id=None):
        self.runtime, self.arena = runtime, arena
        # ``layer_id`` is explicit for the all-layer SharedVMM manager.  Keep
        # ``None`` as the compatibility mode for the original single-layer
        # controller and its existing unit tests.
        self.layer_id = None if layer_id is None else int(layer_id)
        self.state = None
        self.base_slots = runtime.config.shared_expert_initial_slots
        self.extra_slots = runtime.config.shared_expert_extra_slots
        self.kv_overflow_tokens = max(
            0,
            int(getattr(runtime.config, "shared_expert_kv_overflow_tokens", 0) or 0),
        )
        self.kv_min_slots = max(
            0,
            int(getattr(runtime.config, "shared_expert_kv_min_slots", 0) or 0),
        )
        self.kv_overflow_segment = None
        self.kv_overflow_active_tokens = 0
        self.kv_overflow_activation_count = 0
        self.kv_overflow_restore_count = 0
        self.kv_overflow_restore_skip_count = 0
        self.kv_overflow_expert_slots_reclaimed = 0
        self.kv_overflow_admission_activation_count = 0
        self.budget = None
        self._next_donor_scan_step = 0
        self.budget_headroom_skips = 0
        if runtime.config.shared_expert_policy == "adaptive":
            from .residency_budget import ResidencyBudget

            if not runtime.config.shared_expert_free_kv_donors:
                raise ValueError("adaptive residency requires free KV donors")
            self.budget = ResidencyBudget(
                self.base_slots,
                self.base_slots + self.extra_slots,
                runtime.config.shared_expert_decision_interval,
                runtime.config.shared_expert_headroom_steps,
            )
        self.expert_allocations = {}
        self.blocked = {}  # full-attention layer -> physical token locations
        # Context-demand checks run once per decode step. Keep the union of
        # lent KV locations incrementally so a large loan is not rebuilt as a
        # Python set on every step. ``None`` also lets tests/tools seed
        # ``blocked`` directly before the first demand query.
        self._blocked_kv_location_union = None
        self.scratch_blocked = set()  # virtual-scratch locations lent to experts
        self.scratch_donor_page_count = 0
        self.scratch_lend_count = 0
        self.scratch_lend_page_count = 0
        self.scratch_current_loan_page_count = 0
        self.scratch_recall_count = 0
        self.scratch_lend_skip_count = 0
        self.scratch_lend_skip_reason = ""
        # None lets direct donor-discovery tests use the pure idle predicate;
        # after_decode supplies the context-pressure decision for live runs.
        self._scratch_donor_scan_enabled = None
        self.grow_count = 0
        self.last_growth_ms = None
        self.growth_physical_create_count = 0
        self.peak_slots = self.base_slots
        self.weight_ptrs = None
        self.initial_resident_expert_ids = None
        self.initial_resident_source = "uninitialized"
        self.borrowed_slot_use_count = 0
        self.token_chunk_batches = 0
        self.token_chunk_calls = 0
        self.token_chunk_gpu_group_calls = 0
        self.token_chunk_gpu_group_fallbacks = 0
        self.token_chunk_gpu_group_rows = 0
        self.token_chunk_gpu_group_policy_skips = 0
        self.token_chunk_cpu_known_decode_policy_skips = 0
        self._gpu_grouping_disabled = False
        self._gpu_group_snapshot = None
        self._gpu_grouping_decision = None
        self.donor_scan_count = 0
        self.donor_scan_ms = 0.0
        self.donor_select_ms = 0.0
        self.donor_miss_count = 0
        self.donor_cache_hit_count = 0
        self.donor_finalize_ms = 0.0
        self.admission_shortage_tokens = 0
        self.admission_pressure_skip_count = 0
        self.admission_attempt_count = 0
        self.last_admission_decision = None
        self.last_context_live_tokens = None
        self.last_context_waiting_tokens = 0
        self.last_context_demand_tokens = None
        self.last_context_capacity_tokens = None
        self.last_context_reserve_tokens = None
        self.context_pressure_count = 0
        self._donor_miss_key = None
        self.token_chunk_materializations = 0
        self.token_chunk_reordered_groups = 0
        self.token_chunk_adaptive_reuse_count = 0
        self.token_chunk_adaptive_input_count = 0
        self.token_chunk_adaptive_window_reuse_count = 0
        self.token_chunk_indexed_reuse_count = 0
        self.token_chunk_post_moe_prefetch_count = 0
        self.token_chunk_post_moe_prefetch_ids = 0
        self.token_chunk_post_moe_dead_slot_skips = 0
        self.token_chunk_post_moe_capacity_skips = 0
        self.future_eviction_group_count = 0
        self.future_eviction_slot_count = 0
        # Input-order forwards expose the complete route-group sequence before
        # the first MoE call.  These counters distinguish the new bounded
        # eviction plan from the existing per-materialize fallback.
        self.token_chunk_eviction_plan_groups = 0
        self.token_chunk_eviction_plan_ids = 0
        self.token_chunk_eviction_batches = 0
        self.token_chunk_eviction_ids = 0
        self.token_chunk_eviction_early_ids = 0
        self._token_chunk_eviction_early_ids_by_group = {}
        self.route_reuse_window = 32
        self._last_token_chunk_order = runtime.config.shared_expert_chunk_order
        self._chunk_wall_ms = dict.fromkeys(
            (
                "routing",
                "group",
                "order",
                "gather",
                "prepare",
                "moe",
                "scatter",
                "total",
            ),
            0.0,
        )
        self._chunk_gpu_ms = dict.fromkeys(("gather", "prepare", "moe", "scatter"), 0.0)
        self._chunk_events = []
        self._chunk_by_mode = {
            mode: {
                "wall_ms": dict.fromkeys(self._chunk_wall_ms, 0.0),
                "stream_ms": dict.fromkeys(self._chunk_gpu_ms, 0.0),
                "calls": 0,
                "token_rows": 0,
                "slot_capacity_histogram": {},
            }
            for mode in ("prefill", "decode", "other")
        }
        self._prepare_profiler = None
        self._prefetch_probe = None
        # A transient per-forward access order for the shared chunk path.
        # ``state.lru`` remains decode-step based for cross-forward planning.
        self._forward_expert_lru = None
        self._forward_expert_lru_clock = 0
        if runtime.config.shared_expert_trace_waits:
            from .wait_trace import install_wait_trace

            install_wait_trace(runtime)
        if runtime.config.shared_expert_profile_prepare:
            from .prepare_profile import PrepareProfiler

            self._prepare_profiler = PrepareProfiler()

    def _owned_loan_count(self):
        count = getattr(self.arena, "loan_count", None)
        if callable(count):
            return int(count(self))
        # Lightweight unit-test arenas predate owner-aware SharedVMM.  They
        # can only represent one controller, so their global list is exact.
        return len(getattr(self.arena, "loans", ()))

    def _sync_scratch_stats(self):
        stats = getattr(self.runtime, "stats", None)
        if stats is None:
            return
        for name, value in (
            (
                "virtual_scratch_lend_page_count",
                getattr(self, "scratch_lend_page_count", 0),
            ),
            (
                "virtual_scratch_recall_count",
                getattr(self, "scratch_recall_count", 0),
            ),
            (
                "virtual_scratch_lend_skip_count",
                getattr(self, "scratch_lend_skip_count", 0),
            ),
        ):
            if hasattr(stats, name):
                setattr(stats, name, int(value))

    def _chunk_forward_mode(self):
        mode = getattr(self.runtime, "_current_forward_mode", "")
        return (
            "prefill"
            if mode in ("extend", "prefill")
            else "decode" if mode == "decode" else "other"
        )

    @contextmanager
    def _profile_chunk_phase(self, phase, device=None):
        if self.runtime.config.shared_expert_trace_waits:
            with torch.profiler.record_function(f"layerkv/chunk/{phase}"):
                yield
            return
        if not self.runtime.config.shared_expert_profile_chunks:
            yield
            return
        started = time.perf_counter()
        mode = self._chunk_forward_mode()
        events = None
        if device is not None and device.type == "cuda":
            events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            events[0].record(torch.cuda.current_stream(device))
        try:
            yield
        finally:
            if events is not None:
                events[1].record(torch.cuda.current_stream(device))
                self._chunk_events.append((mode, phase, *events))
            elapsed = (time.perf_counter() - started) * 1000
            self._chunk_wall_ms[phase] += elapsed
            self._chunk_by_mode[mode]["wall_ms"][phase] += elapsed

    def _collect_chunk_profile(self):
        pending = []
        for mode, phase, start, end in self._chunk_events:
            if end.query():
                elapsed = start.elapsed_time(end)
                self._chunk_gpu_ms[phase] += elapsed
                self._chunk_by_mode[mode]["stream_ms"][phase] += elapsed
            else:
                pending.append((mode, phase, start, end))
        self._chunk_events = pending

    def record_use(self, state, dispatch):
        if state is not self.state or state.slot_capacity <= self.base_slots:
            return
        ids = dispatch.topk_output.topk_ids
        # Explicit correctness telemetry for this synchronous experimental path.
        if bool(torch.any(ids >= self.base_slots).item()):
            self.borrowed_slot_use_count += 1

    def _begin_forward_expert_lru(self, state):
        """Seed a forward-local LRU without changing cross-forward timestamps."""
        state_lru = getattr(state, "lru", {})
        self._forward_expert_lru = {
            int(logical_id): int(state_lru.get(int(logical_id), -1))
            for logical_id in state.logical_to_slot
        }
        self._forward_expert_lru_clock = max(
            self._forward_expert_lru.values(), default=-1
        ) + 1

    @contextmanager
    def _forward_expert_lru_scope(self):
        """Expose the transient route LRU only during runtime preparation."""
        runtime = self.runtime
        previous = getattr(runtime, "_layerkv_active_expert_lru", None)
        runtime._layerkv_active_expert_lru = self._forward_expert_lru
        try:
            yield
        finally:
            if previous is None:
                delattr(runtime, "_layerkv_active_expert_lru")
            else:
                runtime._layerkv_active_expert_lru = previous

    def _touch_forward_expert_lru(self, state, route_bits):
        """Mark the current route group after its materialization/use."""
        if self._forward_expert_lru is None:
            return
        tick = int(self._forward_expert_lru_clock)
        pending = int(route_bits)
        while pending:
            lowest = pending & -pending
            logical_id = lowest.bit_length() - 1
            if logical_id in state.logical_to_slot:
                self._forward_expert_lru[logical_id] = tick
            pending ^= lowest
        self._forward_expert_lru_clock = tick + 1

    def allocate_expert(self, old, slots, *, layer_id=None):
        if layer_id is not None and self.layer_id is not None:
            if int(layer_id) != int(self.layer_id):
                raise ValueError(
                    f"expert allocation layer {layer_id} does not match "
                    f"controller layer {self.layer_id}"
                )
        row_bytes = int(old[0].nbytes)
        if row_bytes % self.arena.page_bytes:
            raise ValueError(
                "shared expert parameter rows must align to CUDA VMM pages"
            )
        allocation = self.arena.allocate(
            tuple(old.shape), old.dtype, kind="expert", mapped_bytes=slots * row_bytes
        )
        self.expert_allocations[allocation.ptr] = allocation
        return allocation.tensor((slots,) + tuple(old.shape[1:]))

    def ensure_installed(self):
        if self.state is not None:
            return
        r = self.runtime
        layer = (
            self.layer_id
            if self.layer_id is not None
            else r.config.shared_expert_layer
        )
        module = dict(r._expert_modules).get(layer)
        if module is None or not r.physical_expert_supported:
            raise ValueError(f"shared VMM expert layer {layer} is not supported")
        full = int(module.w13_weight.shape[0])
        if self.base_slots + self.extra_slots > full:
            raise ValueError("shared expert initial + extra slots exceeds expert count")
        if self.base_slots < int(module.top_k):
            raise ValueError("shared expert initial slots must cover one token's top-k")
        self.kv_min_slots = self.kv_min_slots or int(module.top_k)
        if self.kv_min_slots < int(module.top_k) or self.kv_min_slots > self.base_slots:
            raise ValueError(
                "shared expert KV minimum slots must cover top-k and not exceed initial slots"
            )
        if self.budget is not None:
            self.budget.min_slots = self.kv_min_slots
        # Validate all rows before allocating/backing up any model parameter.
        for name in r._expert_param_names(module):
            param = getattr(module, name)
            if (
                param.dtype not in (torch.float16, torch.bfloat16)
                or int(param[0].nbytes) % self.arena.page_bytes
            ):
                raise ValueError(
                    f"shared VMM requires page-aligned FP16/BF16 expert rows: {name}"
                )
        ensure_hotness = getattr(r, "_ensure_expert_hotness_cpu_view", None)
        if ensure_hotness is not None:
            # Prefill routing is already collected asynchronously on the GPU.
            # Resolve that one small snapshot before choosing the first compact
            # resident set, instead of silently falling back to expert IDs 0..N.
            ensure_hotness(mode="prefill", block=True)
        initial_resident = r._select_initial_resident_experts(
            layer_id=layer,
            full_num_experts=full,
            slot_capacity=self.base_slots,
        )
        self.initial_resident_expert_ids = [int(x) for x in initial_resident]
        if r._expert_candidate_order_by_layer.get(int(layer)):
            self.initial_resident_source = "online_candidate_order"
        elif (
            r._expert_hotness_decode.get(int(layer))
            or r._expert_hotness_prefill.get(int(layer))
        ):
            self.initial_resident_source = "online_hotness"
        else:
            self.initial_resident_source = "id_tiebreak"
        self.state = r._install_expert_layer_slots(
            module,
            layer,
            self.base_slots,
            initial_resident=initial_resident,
            # The backing copy is GPU-issued on the dedicated D2H stream. The
            # compact resident rows are installed independently on the
            # install stream; a first demand waits on its per-expert event.
            async_copy=True,
        )
        r._expert_layers[layer] = self.state
        r._expert_plan_applied = True
        self.weight_ptrs = {
            name: getattr(module, name).data_ptr() for name in self.state.param_names
        }
        r._refresh_expert_stats()

    def _effective_chunk_order(self, state, token_count):
        requested = self.runtime.config.shared_expert_chunk_order
        if requested != "adaptive":
            return requested

        # The decision uses only this forward's actual token batch and the
        # scheduler's current context metadata. It never stores or assumes a
        # request/dataset-specific expert set.
        # ``token_count`` is the number of routed tokens in this MoE call.  In
        # extend/prefill it can be hundreds or thousands even when the
        # request batch is small, so it must not drive the batch-size policy.
        # Use the scheduler's request batch metadata for the policy decision;
        # retain the token-count fallback for controller/unit-test callers that
        # do not have a ForwardBatch yet.
        forward_batch = None
        try:
            forward_batch = self.runtime._last_forward_batch
            batch_size = int(getattr(forward_batch, "batch_size", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            batch_size = 0
        if batch_size <= 0:
            batch_size = int(
                getattr(self.runtime.stats, "observed_batch_size", 0) or 0
            )
        batch_size = max(1, batch_size or int(token_count))
        capacity = max(1, int(state.slot_capacity))
        try:
            avg_prefix = float(self.runtime._avg_prefix_len(forward_batch))
        except (AttributeError, TypeError, ValueError):
            avg_prefix = float(
                getattr(self.runtime.stats, "avg_prefix_len", 0.0) or 0.0
            )
        # ``avg_prefix`` describes cached tokens and is zero for a fresh
        # prefill.  Use the current forward's tokens per request as a separate
        # context signal; do not feed token_count back into the batch-size
        # decision above.  Decode forwards normally have one token per request,
        # so their context signal still comes from avg_prefix.
        current_tokens_per_request = float(token_count) / float(batch_size)
        context_signal = max(avg_prefix, current_tokens_per_request)

        # A request batch smaller than the slot count can still create a wide
        # routed working set: each request contributes ``top_k`` expert rows.
        # Use that shape signal for short-context prefill/decode instead of
        # waiting until the request count exceeds the physical slot count.  A
        # single request stays on the cheap input order even when top-k is
        # larger than one; its route is not a wide batch to amortize.
        module = getattr(state, "module", None)
        top_k = int(getattr(module, "top_k", 0) or 0)
        if top_k <= 0:
            runner_config = getattr(module, "moe_runner_config", None)
            top_k = int(getattr(runner_config, "top_k", 0) or 0)
        top_k = max(1, top_k)
        routed_rows = batch_size * top_k
        wide_route = batch_size > 1 and routed_rows >= max(1, (capacity + 1) // 2)
        large_batch = batch_size > capacity or wide_route
        long_context_small_batch = context_signal > 2048.0 and batch_size <= max(
            1, capacity // 2
        )
        # With very few resident slots, a full reuse-order selection scans
        # nearly every routed group for every group.  Use a bounded lookahead
        # in this multi-request case instead.  The threshold includes the
        # actual batch and top-k route width, and does not assume a
        # dataset-specific expert set.
        low_capacity_long_context = (
            long_context_small_batch
            and batch_size > 1
            and capacity <= 2 * top_k
        )
        if low_capacity_long_context:
            self.token_chunk_adaptive_window_reuse_count += 1
            return "window-reuse"
        if large_batch or long_context_small_batch:
            self.token_chunk_adaptive_reuse_count += 1
            return "reuse"
        self.token_chunk_adaptive_input_count += 1
        return "input"

    @staticmethod
    def _logical_ids_from_bits(bits):
        logical_ids = []
        pending = int(bits)
        while pending:
            lowest = pending & -pending
            logical_ids.append(lowest.bit_length() - 1)
            pending ^= lowest
        return logical_ids

    def _prefetch_logical_ids_for_groups(self, groups):
        """Return future route IDs in nearest-group-first order.

        The caller already owns the exact route-group snapshot for this
        forward.  Keeping the order here lets one materialization submission
        cover a bounded successor window without inventing a persistent or
        dataset-specific expert set.
        """
        logical_ids = []
        seen = set()
        for group in groups:
            pending = int(group[0])
            while pending:
                lowest = pending & -pending
                logical_id = lowest.bit_length() - 1
                if logical_id not in seen:
                    seen.add(logical_id)
                    logical_ids.append(logical_id)
                pending ^= lowest
        return logical_ids

    def _plan_input_group_evictions(self, state, groups, *, order=None):
        """Predict slot victims for an input-order route sequence.

        The planner is deliberately forward-local: it simulates only the
        current group list and the current resident map.  It does not create
        a persistent expert set.  A planned victim is protected from the
        current group, but may be needed by a later group; that later reload
        is part of the bounded Belady-style simulation and remains a safe
        fallback if asynchronous prefetch changes the live map.

        The returned list is indexed by group position.  Each entry contains
        resident logical IDs to evict *before* that group is prepared.  Slot
        ownership, pending DMA, and CPU backing are rechecked by the runtime
        eviction helper at application time.
        """
        capacity = int(getattr(state, "slot_capacity", 0) or 0)
        full_num_experts = int(getattr(state, "full_num_experts", 0) or 0)
        order = self._last_token_chunk_order if order is None else order
        self._token_chunk_eviction_early_ids_by_group = {}
        if (
            capacity <= 0
            or capacity >= full_num_experts
            or len(groups) <= 1
            or order != "input"
            or not hasattr(state, "slot_to_logical")
            or not hasattr(state, "free_slots")
        ):
            return None

        resident = {
            int(logical_id): int(slot_id)
            for logical_id, slot_id in getattr(state, "logical_to_slot", {}).items()
            if 0 <= int(slot_id) < capacity
        }
        if not resident:
            return None

        pending_slots_fn = getattr(self.runtime, "_pending_expert_copy_slots", None)
        pending_slots = (
            {
                int(slot_id)
                for slot_id in pending_slots_fn(state)
                if 0 <= int(slot_id) < capacity
            }
            if callable(pending_slots_fn)
            else set()
        )
        used_slots = set(resident.values())
        free_slots = {
            int(slot_id)
            for slot_id in getattr(state, "free_slots", ())
            if 0 <= int(slot_id) < capacity
            and int(slot_id) not in used_slots
            and int(slot_id) not in pending_slots
        }
        # A missing free-slot ledger is not safe to repair speculatively.  The
        # normal install path always populates it; synthetic/partial states
        # should continue through the existing chooser.
        if not free_slots and len(used_slots) < capacity:
            return None

        route_ids = [
            self._logical_ids_from_bits(route_bits) for route_bits, _rows in groups
        ]
        route_positions = {}
        for group_index, logical_ids in enumerate(route_ids):
            for logical_id in logical_ids:
                if 0 <= logical_id < full_num_experts:
                    route_positions.setdefault(logical_id, []).append(group_index)

        simulated_lru = {
            int(logical_id): int(step)
            for logical_id, step in getattr(state, "lru", {}).items()
        }
        plan = [[] for _ in groups]
        for group_index, logical_ids in enumerate(route_ids):
            current_ids = {
                int(logical_id)
                for logical_id in logical_ids
                if 0 <= int(logical_id) < full_num_experts
            }
            for logical_id in logical_ids:
                logical_id = int(logical_id)
                if logical_id < 0 or logical_id >= full_num_experts:
                    continue
                if logical_id in resident:
                    continue

                if free_slots:
                    slot_id = min(free_slots)
                    free_slots.remove(slot_id)
                else:
                    candidates = []
                    for candidate_id, candidate_slot in resident.items():
                        if (
                            candidate_id in current_ids
                            or candidate_slot in pending_slots
                        ):
                            continue
                        positions = route_positions.get(candidate_id, ())
                        next_position = bisect_right(positions, group_index)
                        next_use = (
                            math.inf
                            if next_position >= len(positions)
                            else positions[next_position]
                        )
                        candidates.append(
                            (
                                next_use,
                                -int(simulated_lru.get(candidate_id, -1)),
                                -int(candidate_id),
                                -int(candidate_slot),
                                int(candidate_id),
                                int(candidate_slot),
                            )
                        )
                    if not candidates:
                        # A pending DMA or a malformed group can make the
                        # ideal plan incomplete.  Leave the whole forward to
                        # the guarded runtime chooser.
                        return None
                    _next_use, _lru, _id, _slot, victim_id, slot_id = max(
                        candidates
                    )
                    resident.pop(victim_id, None)
                    plan[group_index].append(int(victim_id))

                resident[logical_id] = int(slot_id)
                simulated_lru[logical_id] = group_index

            for logical_id in current_ids:
                if logical_id in resident:
                    simulated_lru[logical_id] = group_index

        planned_groups = sum(bool(entry) for entry in plan)
        planned_ids = sum(len(entry) for entry in plan)
        if not planned_ids:
            return None
        self.token_chunk_eviction_plan_groups += planned_groups
        self.token_chunk_eviction_plan_ids += planned_ids
        return self._coalesce_input_group_eviction_plan(groups, plan)

    def _coalesce_input_group_eviction_plan(self, groups, plan):
        """Move safe future victims to the earliest reusable safe point.

        A victim selected for group ``j`` does not have to stay resident until
        ``j``.  Once its last route use before ``j`` has completed, evicting it
        early is safe and lets that D2H/remap join another group's batch.  Keep
        the original entry as a guarded fallback: a pending DMA or a changed
        live map can make the early application a no-op, in which case the
        original group still gets a chance to evict it.
        """
        self._token_chunk_eviction_early_ids_by_group = {}
        if not plan or len(plan) != len(groups):
            return plan
        route_positions = {}
        for group_index, (route_bits, _route_rows) in enumerate(groups):
            pending = int(route_bits)
            while pending:
                lowest = pending & -pending
                logical_id = lowest.bit_length() - 1
                route_positions.setdefault(logical_id, []).append(group_index)
                pending ^= lowest

        coalesced = [list(victims) for victims in plan]
        first_planned_group = {}
        for target_group, victims in enumerate(plan):
            for logical_id in dict.fromkeys(int(x) for x in victims):
                # A repeated logical ID may represent a later residency
                # lifetime after a reload. Only coalesce its first eviction;
                # later entries remain at their planner-selected boundaries.
                if logical_id in first_planned_group:
                    continue
                positions = route_positions.get(logical_id, ())
                prior_positions = [
                    position for position in positions if position < target_group
                ]
                safe_group = (prior_positions[-1] + 1) if prior_positions else 0
                first_planned_group[logical_id] = target_group
                if safe_group < target_group:
                    coalesced[safe_group].append(logical_id)
                    self.token_chunk_eviction_early_ids += 1
                    self._token_chunk_eviction_early_ids_by_group.setdefault(
                        safe_group, set()
                    ).add(logical_id)
        return coalesced

    def _apply_planned_evictions(
        self,
        state,
        current_bits,
        planned_ids,
        *,
        surplus_logical_ids=None,
    ):
        """Apply only the victims needed to make the current group fit."""
        if not planned_ids:
            return 0
        current_ids = set(self._logical_ids_from_bits(current_bits))
        surplus_ids = {
            int(logical_id) for logical_id in (surplus_logical_ids or ())
        }
        missing_count = sum(
            int(logical_id) not in state.logical_to_slot
            for logical_id in current_ids
        )
        if missing_count <= 0 and not surplus_ids:
            return 0

        pending_slots_fn = getattr(self.runtime, "_pending_expert_copy_slots", None)
        pending_slots = (
            set(pending_slots_fn(state))
            if callable(pending_slots_fn)
            else set()
        )
        used_slots = {
            int(slot_id)
            for slot_id in getattr(state, "slot_to_logical", {})
        }
        available_slots = {
            int(slot_id)
            for slot_id in getattr(state, "free_slots", ())
            if int(slot_id) not in used_slots and int(slot_id) not in pending_slots
        }
        need = max(0, missing_count - len(available_slots))
        if need <= 0 and not surplus_ids:
            return 0
        selected = []
        selected_set = set()
        selected_required = 0
        for logical_id in planned_ids:
            logical_id = int(logical_id)
            slot_id = state.logical_to_slot.get(logical_id)
            if (
                logical_id in selected_set
                or slot_id is None
                or logical_id in current_ids
                or int(slot_id) in pending_slots
            ):
                continue
            is_surplus = logical_id in surplus_ids
            if not is_surplus and selected_required >= need:
                continue
            selected.append(logical_id)
            selected_set.add(logical_id)
            if not is_surplus:
                selected_required += 1
        if not selected:
            return 0
        evicted = self.runtime._evict_experts_batched(
            state,
            selected,
            protected_logical_ids=current_ids,
            reason="planned",
        )
        count = len(evicted or ())
        if count:
            self.token_chunk_eviction_batches += 1
            self.token_chunk_eviction_ids += count
        return count

    def _current_residency_batch_size(self):
        """Use scheduler batch metadata when the forward wrapper is stale."""
        runtime = self.runtime
        candidates = [
            int(
                getattr(
                    getattr(runtime, "_last_forward_batch", None),
                    "batch_size",
                    0,
                )
                or 0
            ),
            int(getattr(runtime.stats, "native_schedule_batch_size", 0) or 0),
            int(
                getattr(runtime.stats, "native_schedule_running_batch_size", 0)
                or 0
            ),
        ]
        return max(1, *candidates)

    @staticmethod
    def _resident_expert_bits(state):
        resident = 0
        for logical in state.logical_to_slot:
            if 0 <= logical < state.full_num_experts:
                resident |= 1 << logical
        return resident

    def _ordered_token_groups(self, state, groups, order=None):
        order = (
            self.runtime.config.shared_expert_chunk_order
            if order is None
            else order
        )
        if order == "input":
            yield from groups
            return
        if order == "window-reuse":
            pending = deque(enumerate(groups))
            window_size = max(1, int(getattr(self, "route_reuse_window", 32)))
            position = 0
            while pending:
                resident = self._resident_expert_bits(state)
                candidate_count = min(window_size, len(pending))
                selected = min(
                    range(candidate_count),
                    key=lambda offset: (
                        (pending[offset][1][0] & ~resident).bit_count(),
                        -(pending[offset][1][0] & resident).bit_count(),
                        pending[offset][0],
                    ),
                )
                pending.rotate(-selected)
                original_position, group = pending.popleft()
                pending.rotate(selected)
                self.token_chunk_reordered_groups += original_position != position
                position += 1
                yield group
            return
        module = getattr(state, "module", None)
        top_k = int(getattr(module, "top_k", 0) or 0)
        if top_k <= 0:
            runner_config = getattr(module, "moe_runner_config", None)
            top_k = int(getattr(runner_config, "top_k", 0) or 0)
        if top_k > 0 and int(state.slot_capacity) <= 2 * top_k:
            yield from self._ordered_token_groups_indexed_reuse(state, groups)
            return
        pending = list(enumerate(groups))
        for position in range(len(groups)):
            with self._profile_chunk_phase("order"):
                # Re-read actual residency after each materialization, rather
                # than assuming an ideal cache replacement policy. Ties are
                # deterministic; group membership and token reductions stay put.
                resident = self._resident_expert_bits(state)
                selected = min(
                    range(len(pending)),
                    key=lambda i: (
                        (pending[i][1][0] & ~resident).bit_count(),
                        -(pending[i][1][0] & resident).bit_count(),
                        pending[i][0],
                    ),
                )
                original_position, group = pending.pop(selected)
                self.token_chunk_reordered_groups += original_position != position
            yield group

    def _ordered_token_groups_indexed_reuse(self, state, groups):
        """Select exact reuse-order groups through resident-expert postings."""
        pending = list(enumerate(groups))
        self.token_chunk_indexed_reuse_count += 1
        if not pending:
            return

        alive = [True] * len(pending)
        postings = {}
        zero_overlap_heap = []
        for offset, (original_position, group) in enumerate(pending):
            bits = int(group[0])
            heapq.heappush(
                zero_overlap_heap,
                (bits.bit_count(), original_position, offset),
            )
            remaining = bits
            while remaining:
                lowest = remaining & -remaining
                logical_id = lowest.bit_length() - 1
                postings.setdefault(logical_id, []).append(offset)
                remaining ^= lowest

        for position in range(len(pending)):
            with self._profile_chunk_phase("order"):
                resident = self._resident_expert_bits(state)
                overlap_counts = {}
                remaining = resident
                while remaining:
                    lowest = remaining & -remaining
                    logical_id = lowest.bit_length() - 1
                    for offset in postings.get(logical_id, ()):
                        if alive[offset]:
                            overlap_counts[offset] = (
                                overlap_counts.get(offset, 0) + 1
                            )
                    remaining ^= lowest

                selected = None
                selected_key = None
                for offset, overlap in overlap_counts.items():
                    original_position, group = pending[offset]
                    bits = int(group[0])
                    key = (bits.bit_count() - overlap, -overlap, original_position)
                    if selected_key is None or key < selected_key:
                        selected = offset
                        selected_key = key

                skipped = []
                zero_item = None
                zero_key = None
                while zero_overlap_heap:
                    item = heapq.heappop(zero_overlap_heap)
                    size, original_position, offset = item
                    if not alive[offset]:
                        continue
                    if offset in overlap_counts:
                        skipped.append(item)
                        continue
                    zero_item = item
                    zero_key = (size, 0, original_position)
                    break
                for item in skipped:
                    heapq.heappush(zero_overlap_heap, item)

                if zero_key is not None and (
                    selected_key is None or zero_key < selected_key
                ):
                    selected = zero_item[2]
                elif zero_item is not None:
                    heapq.heappush(zero_overlap_heap, zero_item)
                if selected is None:  # pragma: no cover - index is complete.
                    raise RuntimeError("expert group ordering index became empty")
                alive[selected] = False
                original_position, group = pending[selected]
                self.token_chunk_reordered_groups += original_position != position
            yield group

    def _prefetch_next_input_group(
        self,
        state,
        current_bits: int,
        next_bits: int | None,
        *,
        producer_event=None,
        extra_protected_ids=None,
        post_moe_dead_only: bool = False,
        prefetch_logical_ids=None,
        prefetch_group_count: int = 1,
    ) -> None:
        """Submit an exact next-group H2D copy when slots can be protected.

        Previous-route prefetch is speculative and may copy an expert that the
        next forward never uses. A token chunk already has an exact successor;
        protect the current group and use only the subset of successor IDs that
        fits in the remaining slots. This keeps the copy-stream submission safe
        without requiring the complete successor union to fit.

        Adaptive/reuse ordering calls this before the current MoE. Its copy
        streams wait for the previous group's producer event, so the previous
        slot may be reclaimed without racing the current kernel while the
        current group's slots remain protected.

        ``post_moe_dead_only`` is a stricter successor path: the current MoE
        has completed, so its slots may be reused, but every expert with a
        later use in this forward is added to ``protected_ids``. It therefore
        cannot increase transfer volume by evicting an expert that the same
        forward will need again.
        """
        if not next_bits:
            return
        r = self.runtime
        # Exact next-group copies need a sufficiently wide current batch to
        # overlap their H2D with useful MoE work.  For B8 with a 49-slot
        # resident set, they only add copy-stream pressure and churn.  Keep
        # the same short-context/large-batch policy used by route prefetch.
        # Long-context/small-batch decode may use an exact successor too, but
        # only when the bounded KVC copy window has room.  This preserves the
        # policy priority: KVC keeps the window when it is already full, while
        # an idle window can hide expert H2D behind the current MoE call.
        forward_batch = getattr(r, "_last_forward_batch", None)
        if forward_batch is not None:
            allow_exact = getattr(
                r, "_expert_prefetch_is_short_context_large_batch", None
            )
            short_large = (
                bool(allow_exact(forward_batch))
                if callable(allow_exact)
                else True
            )
            if not short_large:
                long_small = getattr(
                    r, "_expert_prefetch_is_long_context_small_batch", None
                )
                kvc_window = getattr(
                    r, "_expert_prefetch_kvc_window_available", None
                )
                forward_mode = getattr(r, "_current_forward_mode", "")
                allow_long_small = (
                    forward_mode == "decode"
                    and callable(long_small)
                    and bool(long_small(forward_batch))
                    and callable(kvc_window)
                    and bool(kvc_window())
                )
                if not allow_long_small:
                    r.stats.expert_prefetch_exact_policy_skip_count += 1
                    r.stats.expert_prefetch_last_gate = (
                        "exact-short-large-batch-only"
                    )
                    return
                r.stats.expert_prefetch_last_gate = "exact-long-small-kvc-window"
        async_enabled = getattr(r, "_expert_h2d_async_enabled", None)
        if async_enabled is None or getattr(state, "device", None) is None:
            return
        if not async_enabled(state):
            return
        capacity = int(getattr(state, "slot_capacity", 0) or 0)
        current_ids = set()
        pending = int(current_bits)
        while pending:
            lowest = pending & -pending
            current_ids.add(lowest.bit_length() - 1)
            pending ^= lowest
        if producer_event is not None:
            finalize = getattr(r, "_finalize_expert_materialize_events", None)
            if callable(finalize):
                finalize(block=False)
            pending_ids_fn = getattr(r, "_pending_expert_copy_logical_ids", None)
            if callable(pending_ids_fn):
                pending_ids = set(pending_ids_fn(state.layer_id))
            else:
                pending_ids = {
                    int(logical_id)
                    for pending_copy in getattr(r, "_pending_expert_copy_events", ())
                    if int(getattr(pending_copy, "layer_id", -1))
                    == int(state.layer_id)
                    for logical_id in getattr(pending_copy, "logical_ids", ())
                    if int(logical_id)
                    not in getattr(pending_copy, "invalidated_logical_ids", set())
                }
            protected_ids = pending_ids
            protected_ids.update(current_ids)
        else:
            protected_ids = current_ids
        if extra_protected_ids:
            protected_ids.update(int(x) for x in extra_protected_ids)
        if post_moe_dead_only:
            # ``extra_protected_ids`` may contain many future logical IDs that
            # are not resident. Count occupied/pending slots, not logical IDs,
            # so a large future route set does not hide genuinely dead slots.
            protected_slots = {
                int(slot)
                for logical_id, slot in state.logical_to_slot.items()
                if int(logical_id) in protected_ids
            }
            pending_slots_fn = getattr(r, "_pending_expert_copy_slots", None)
            pending_slots = (
                set(pending_slots_fn(state))
                if callable(pending_slots_fn)
                else set()
            )
            available = max(0, capacity - len(protected_slots | pending_slots))
        else:
            available = max(0, capacity - len(protected_ids))
        if available <= 0:
            if post_moe_dead_only:
                self.token_chunk_post_moe_dead_slot_skips += 1
                r.stats.expert_prefetch_post_moe_dead_slot_skip_count += 1
            r.stats.expert_prefetch_exact_skip_capacity_count += 1
            return
        if prefetch_logical_ids is None:
            logical_ids = self._logical_ids_from_bits(next_bits)
        else:
            logical_ids = list(dict.fromkeys(int(x) for x in prefetch_logical_ids))
            r.stats.expert_prefetch_lookahead_group_count += max(
                0, int(prefetch_group_count) - 1
            )
        missing = [
            logical_id
            for logical_id in logical_ids
            if logical_id not in state.logical_to_slot
        ]
        if not missing:
            return
        if len(missing) > available:
            r.stats.expert_prefetch_exact_skip_capacity_count += 1
            missing = missing[:available]
        has_backing = getattr(r, "_expert_prefetch_has_backing", None)
        if has_backing is not None:
            missing = [
                logical_id
                for logical_id in missing
                if has_backing(state, logical_id)
            ]
        if not missing:
            return
        if producer_event is not None:
            for stream_getter_name in (
                "_expert_d2h_copy_stream",
                "_expert_h2d_copy_stream",
            ):
                stream_getter = getattr(r, stream_getter_name, None)
                if not callable(stream_getter):
                    continue
                copy_stream = stream_getter(state.device)
                if copy_stream is not None:
                    copy_stream.wait_event(producer_event)
        r._materialize_experts(
            state,
            missing,
            reason="prefetch",
            protected_logical_ids=protected_ids,
        )
        r.stats.expert_prefetch_exact_group_count += 1
        r.stats.expert_prefetch_exact_id_count += len(missing)
        r.stats.expert_prefetch_count += len(missing)
        r.stats.expert_prefetch_candidate_count += len(missing)
        r.stats.expert_prefetch_issued_count += len(missing)
        if prefetch_logical_ids is not None:
            r.stats.expert_prefetch_lookahead_id_count += len(missing)
        if post_moe_dead_only:
            self.token_chunk_post_moe_prefetch_count += 1
            self.token_chunk_post_moe_prefetch_ids += len(missing)
            r.stats.expert_prefetch_post_moe_count += 1
            r.stats.expert_prefetch_post_moe_id_count += len(missing)

    def _post_moe_prefetch_is_worthwhile(self, state, topk_ids) -> bool:
        """Keep post-MoE overlap for the high-churn expert-capacity regime.

        Once the resident slot set is wider than two route widths, the normal
        successor prefetch has enough room to hide most misses.  Continuing to
        refill dead slots after every MoE then adds H2D/D2H traffic without
        creating a useful residency opportunity.  The same bounded capacity
        boundary is used by the input-order eviction planner.
        """
        capacity = int(getattr(state, "slot_capacity", 0) or 0)
        full_num_experts = int(getattr(state, "full_num_experts", 0) or 0)
        route_width = int(getattr(topk_ids, "shape", (0, 0))[-1] or 0)
        if capacity <= 0 or route_width <= 0 or capacity >= full_num_experts:
            return False
        if capacity > 2 * route_width:
            self.token_chunk_post_moe_capacity_skips += 1
            self.runtime.stats.expert_prefetch_post_moe_capacity_skip_count += 1
            self.runtime.stats.expert_prefetch_last_gate = (
                "post-moe-capacity-headroom"
            )
            return False
        return True

    def _gpu_grouping_enabled(self, topk_ids):
        if not (
            getattr(self.runtime.config, "shared_expert_gpu_grouping", False)
            and topk_ids.device.type == "cuda"
            and layerkv_group_token_experts is not None
            and not self._gpu_grouping_disabled
        ):
            return False
        minimum_rows = max(
            1,
            int(
                getattr(
                    self.runtime.config,
                    "shared_expert_gpu_grouping_min_rows",
                    128,
                )
                or 128
            ),
        )
        route_shape = getattr(topk_ids, "shape", ())
        if route_shape and int(route_shape[0]) < minimum_rows:
            # The exact first-fit kernel has a serial capacity scan.  For the
            # small decode batches where CPU grouping is sub-millisecond, do
            # not replace a cheap host operation with CUDA launches and a
            # metadata synchronization. Larger routed windows still use the
            # GPU path (and the fused multi-capacity admission pass).
            self.token_chunk_gpu_group_policy_skips += 1
            return False
        state = self.state
        module = getattr(state, "module", None)
        top_k = int(getattr(module, "top_k", 0) or 0)
        if top_k <= 0:
            top_k = 1
        batch = self._current_residency_batch_size()
        forward_batch = getattr(self.runtime, "_last_forward_batch", None)
        context_tokens = None
        avg_prefix = getattr(self.runtime, "_avg_prefix_len", None)
        if callable(avg_prefix) and forward_batch is not None:
            context_tokens = avg_prefix(forward_batch)
        if (
            context_tokens is not None
            and float(context_tokens) > 2048.0
            and batch <= top_k
            and int(state.slot_capacity) <= 2 * top_k
        ):
            # At E=top-k+1 the CPU indexed first-fit path is already bounded
            # by route subsets.  Avoid paying a CUDA launch plus host mask
            # synchronization for the long-context/small-batch policy, where
            # KV headroom is more valuable than route-group throughput.
            self.token_chunk_gpu_group_policy_skips += 1
            return False
        return True

    def _group_token_experts_gpu(self, topk_ids, capacity, *, count_stats=True):
        groups, routing_valid = layerkv_group_token_experts(
            topk_ids,
            capacity,
            self.state.full_num_experts,
            device_rows=True,
        )
        if count_stats:
            self.token_chunk_gpu_group_calls += 1
            self.token_chunk_gpu_group_rows += int(topk_ids.shape[0])
        return groups, routing_valid

    def prepare_gpu_decode_groups(self, state, topk_ids):
        """Group a decode route on GPU and reuse the result for execution.

        The hook needs the group count before selecting the native or chunked
        MoE path.  Keep the compact groups here so ``run_token_chunks`` does
        not launch a second grouping pass for the same route tensor.
        """
        capacities = [int(state.slot_capacity)]
        if self.budget is not None:
            capacities.extend((int(self.budget.base_slots), int(self.budget.max_slots)))
        capacities = list(dict.fromkeys(capacities))
        if len(capacities) > 1 and layerkv_group_token_experts_multi is not None:
            grouped = layerkv_group_token_experts_multi(
                topk_ids,
                capacities,
                self.state.full_num_experts,
                device_rows=True,
            )
            self.token_chunk_gpu_group_calls += 1
            self.token_chunk_gpu_group_rows += int(topk_ids.shape[0])
            grouped_by_capacity = dict(zip(capacities, grouped))
            groups, routing_valid = grouped_by_capacity[int(state.slot_capacity)]
        else:
            grouped_by_capacity = None
            groups, routing_valid = self._group_token_experts_gpu(
                topk_ids, state.slot_capacity
            )
        budget = self.budget
        if budget is not None and budget.last_observed_step != self.runtime._decode_step:
            if grouped_by_capacity is not None:
                base_groups, _ = grouped_by_capacity[int(budget.base_slots)]
                max_groups, _ = grouped_by_capacity[int(budget.max_slots)]
            else:
                base_groups = groups
                if state.slot_capacity != budget.base_slots:
                    base_groups, _ = self._group_token_experts_gpu(
                        topk_ids, budget.base_slots, count_stats=False
                    )
                max_groups = groups
                if state.slot_capacity != budget.max_slots:
                    max_groups, _ = self._group_token_experts_gpu(
                        topk_ids, budget.max_slots, count_stats=False
                    )
            budget.observe_grouped(
                self.runtime._decode_step, base_groups, max_groups
            )
        return groups, routing_valid

    def take_gpu_decode_groups(self, state, topk_ids):
        snapshot = self._gpu_group_snapshot
        self._gpu_group_snapshot = None
        if snapshot is None:
            return None
        snapshot_ids, snapshot_capacity, groups, routing_valid = snapshot
        if snapshot_ids is not topk_ids or snapshot_capacity != state.slot_capacity:
            return None
        return groups, routing_valid

    def run_token_chunks(self, state, dispatch, *args, **kwargs):
        """Keep each token's full top-k reduction in one native MoE call.

        Splitting a BF16 token across expert chunks changes reduction rounding.
        Instead group tokens whose expert union fits the fixed slot budget.
        Index-selected activations also isolate an in-place native runner.
        """
        r = self.runtime
        topk = dispatch.topk_output
        decision = self._gpu_grouping_decision
        self._gpu_grouping_decision = None
        if decision is not None and decision[0] is topk.topk_ids:
            use_gpu_grouping = bool(decision[1])
        else:
            use_gpu_grouping = self._gpu_grouping_enabled(topk.topk_ids)
        # Retire prior batches' completed events even without stats polling.
        self._collect_chunk_profile()
        started_total = time.perf_counter()
        profile_mode = self._chunk_forward_mode()
        if r.config.shared_expert_profile_chunks:
            bucket = self._chunk_by_mode[profile_mode]
            bucket["calls"] += 1
            bucket["token_rows"] += int(dispatch.hidden_states.shape[0])
            key = str(state.slot_capacity)
            bucket["slot_capacity_histogram"][key] = (
                bucket["slot_capacity_histogram"].get(key, 0) + 1
            )
        if getattr(dispatch, "hidden_states_scale", None) is not None:
            raise ValueError(
                "shared expert token chunks require unquantized activations"
            )
        with self._profile_chunk_phase("routing"):
            snapshot = getattr(self, "_routing_snapshot", None)
            self._routing_snapshot = None
            routing_rows = (
                snapshot[1]
                if snapshot is not None and snapshot[0] is topk.topk_ids
                else None
            )
            if not use_gpu_grouping:
                routing_rows = (
                    topk.topk_ids.tolist()
                    if routing_rows is None
                    else routing_rows
                )
        chunk_order = self._effective_chunk_order(
            state, dispatch.hidden_states.shape[0]
        )
        self._last_token_chunk_order = chunk_order
        with self._profile_chunk_phase("group"):
            if use_gpu_grouping:
                try:
                    cached_groups = self.take_gpu_decode_groups(
                        state, topk.topk_ids
                    )
                    if cached_groups is None:
                        groups, routing_valid = self._group_token_experts_gpu(
                            topk.topk_ids, state.slot_capacity
                        )
                    else:
                        groups, routing_valid = cached_groups
                except Exception:
                    # The CPU implementation remains the correctness fallback
                    # for unsupported shapes, JIT/toolchain errors, or future
                    # model route layouts. Do not retry a broken JIT call on
                    # every chunk in the same runtime.
                    self.token_chunk_gpu_group_fallbacks += 1
                    self._gpu_grouping_disabled = True
                    routing_rows = (
                        topk.topk_ids.tolist()
                        if routing_rows is None
                        else routing_rows
                    )
                    groups, routing_valid = group_token_experts(
                        routing_rows, state.slot_capacity, state.full_num_experts
                    )
            else:
                groups, routing_valid = group_token_experts(
                    routing_rows, state.slot_capacity, state.full_num_experts
                )
        if (
            r.config.shared_expert_prepare_path == "cpu-known"
            and not routing_valid
            and not state.topk_ids_invalid_observed
        ):
            # Do not let a prior calibration bypass range checks on fallback.
            state.topk_ids_invalid_observed = True
            r.stats.expert_topk_range_invalid_count += 1
        output = torch.empty_like(dispatch.hidden_states)
        result = None
        self.token_chunk_batches += 1
        ordered_groups = list(groups) if chunk_order == "input" else None
        prefetch_group_limit = max(
            1,
            int(getattr(r.config, "shared_expert_prefetch_groups", 1) or 1),
        )
        indexed_execution_groups = None
        planned_evictions = (
            self._plan_input_group_evictions(
                state, ordered_groups, order=chunk_order
            )
            if ordered_groups is not None
            else None
        )
        if ordered_groups is not None:
            group_iter = None
            current_group = ordered_groups[0] if ordered_groups else None
        else:
            if prefetch_group_limit > 1:
                # Materialize the already-computed execution order once so a
                # bounded successor window can be submitted as one H2D batch.
                # Keep ``ordered_groups is None`` below: adaptive/reuse order
                # still retains its producer-event dependency semantics.
                indexed_execution_groups = list(
                    self._ordered_token_groups(state, groups, order=chunk_order)
                )
                group_iter = None
                current_group = (
                    indexed_execution_groups[0]
                    if indexed_execution_groups
                    else None
                )
            else:
                group_iter = iter(
                    self._ordered_token_groups(state, groups, order=chunk_order)
                )
                current_group = next(group_iter, None)
        self._begin_forward_expert_lru(state)
        post_moe_remaining_counts = None
        if getattr(r.config, "shared_expert_post_moe_prefetch", False):
            # Count route-group uses in the actual input snapshot.  The
            # execution order may be adaptive, so this is decremented when a
            # group is actually selected rather than relying on its input
            # position.  The resulting set is a conservative, exact set of
            # experts still needed by this forward.
            post_moe_remaining_counts = {}
            for route_bits, _route_rows in groups:
                pending_bits = int(route_bits)
                while pending_bits:
                    lowest = pending_bits & -pending_bits
                    logical_id = lowest.bit_length() - 1
                    post_moe_remaining_counts[logical_id] = (
                        post_moe_remaining_counts.get(logical_id, 0) + 1
                    )
                    pending_bits ^= lowest
        # ``ordered_groups`` is the input-order snapshot.  Adaptive/reuse
        # ordering with a bounded successor window is also materialized into
        # ``indexed_execution_groups`` above.  Both are forward-local route
        # sequences, so the same future-use map can guide slot replacement.
        execution_order_groups = (
            ordered_groups
            if ordered_groups is not None
            else indexed_execution_groups
        )
        future_positions = None
        if (
            execution_order_groups is not None
            and state.slot_capacity < state.full_num_experts
            and int(topk.topk_ids.shape[-1]) > 0
            and state.slot_capacity <= 2 * int(topk.topk_ids.shape[-1])
            and len(execution_order_groups) > 1
        ):
            # The route snapshot is already CPU-known.  Keep only group
            # positions for this forward; no expert IDs are persisted across
            # requests or treated as a dataset-specific working set.  For
            # adaptive/window-reuse this is the bounded execution order that
            # was selected for the current forward, not a fixed expert set.
            future_positions = {}
            for route_index, (route_bits, _route_rows) in enumerate(
                execution_order_groups
            ):
                pending_bits = int(route_bits)
                while pending_bits:
                    lowest = pending_bits & -pending_bits
                    logical_id = lowest.bit_length() - 1
                    future_positions.setdefault(logical_id, []).append(route_index)
                    pending_bits ^= lowest
        probe = None
        if r.config.shared_expert_trace_waits:
            self._prefetch_probe = {
                "batch": self.token_chunk_batches,
                "order": chunk_order,
                "requested_order": r.config.shared_expert_chunk_order,
                "scope": "structural upper bound only; DMA lifetime admission not proven",
                "groups": [],
            }
            if r.config.shared_expert_chunk_order == "input":
                probe = self._prefetch_probe["groups"]
        group_index = 0
        previous_moe_done = None
        while current_group is not None:
            bits, rows = current_group
            current_bits = bits
            if post_moe_remaining_counts is not None:
                pending_bits = int(current_bits)
                while pending_bits:
                    lowest = pending_bits & -pending_bits
                    logical_id = lowest.bit_length() - 1
                    remaining = post_moe_remaining_counts.get(logical_id, 0) - 1
                    if remaining > 0:
                        post_moe_remaining_counts[logical_id] = remaining
                    else:
                        post_moe_remaining_counts.pop(logical_id, None)
                    pending_bits ^= lowest
            execution_groups = (
                ordered_groups
                if ordered_groups is not None
                else indexed_execution_groups
            )
            if execution_groups is not None:
                successor_groups = execution_groups[
                    group_index + 1 : group_index + 1 + prefetch_group_limit
                ]
                next_group = successor_groups[0] if successor_groups else None
            else:
                # The lazy adaptive order is state-dependent.  Select its
                # successor only after the current group's preparation has
                # updated the live residency map; otherwise the order and its
                # prefetch decision observe the pre-materialization cache.
                next_group = None
                successor_groups = []
            successor_prefetch_ids = (
                self._prefetch_logical_ids_for_groups(successor_groups)
                if len(successor_groups) > 1
                else None
            )
            with self._profile_chunk_phase("gather", output.device):
                if isinstance(rows, torch.Tensor):
                    indices = rows
                    if (
                        indices.device != output.device
                        or indices.dtype != torch.long
                    ):
                        indices = indices.to(device=output.device, dtype=torch.long)
                else:
                    indices = torch.tensor(
                        rows, dtype=torch.long, device=output.device
                    )
                fields = {
                    "topk_ids": topk.topk_ids.index_select(0, indices),
                    "topk_weights": topk.topk_weights.index_select(0, indices),
                }
                if getattr(topk, "router_logits", None) is not None:
                    fields["router_logits"] = topk.router_logits.index_select(
                        0, indices
                    )
                chunk = dispatch._replace(
                    hidden_states=dispatch.hidden_states.index_select(0, indices),
                    topk_output=topk._replace(**fields),
                )
            before = r.stats.expert_materialize_count
            with self._profile_chunk_phase("prepare", output.device):
                if planned_evictions is not None:
                    early_ids = self._token_chunk_eviction_early_ids_by_group.get(
                        group_index, ()
                    )
                    self._apply_planned_evictions(
                        state,
                        current_bits,
                        planned_evictions[group_index],
                        surplus_logical_ids=early_ids,
                    )
                prepare_kwargs = {}
                future_use = None
                if future_positions is not None:
                    future_use = {}
                    for logical_id in state.logical_to_slot:
                        positions = future_positions.get(int(logical_id), ())
                        next_position = bisect_right(positions, group_index)
                        future_use[int(logical_id)] = (
                            positions[next_position]
                            if next_position < len(positions)
                            else None
                        )
                    self.future_eviction_group_count += 1
                future_marker = object()
                previous_future_use = getattr(
                    r, "_layerkv_future_expert_use", future_marker
                )
                if future_use is not None:
                    r._layerkv_future_expert_use = future_use
                try:
                    if r.config.shared_expert_prepare_path == "cpu-known":
                        # Ascending order matches torch.unique in the generic path,
                        # including resident experts: materialization updates LRU.
                        # For small decode batches we intentionally skip GPU
                        # grouping because its launch/readback cost is higher
                        # than the CPU first-fit scan.  Reuse the exact route
                        # snapshot for decode as well: this does not move
                        # activations to the host, while remap and readiness
                        # remain on the GPU/copy streams.  It avoids rerunning
                        # generic GPU demand discovery for every small group.
                        decode_cpu_known_allowed = (
                            r._current_forward_mode != "decode"
                            or routing_valid
                        )
                        if routing_valid and decode_cpu_known_allowed:
                            logical_ids = []
                            while bits:
                                lowest = bits & -bits
                                logical_ids.append(lowest.bit_length() - 1)
                                bits ^= lowest
                            prepare_kwargs["known_logical_ids"] = logical_ids
                            # The shared controller already consumed this exact
                            # decode routing snapshot in
                            # _expert_chunked_core_required().  Reuse it to avoid
                            # repeating generic GPU demand discovery.  The
                            # expert-hooks guard remains unchanged for every other
                            # caller; only this bounded snapshot-backed path may
                            # opt into decode CPU-known preparation.
                            if r._current_forward_mode == "decode":
                                prepare_kwargs["allow_decode_cpu_known"] = True
                        elif not routing_valid:
                            r.stats.expert_cpu_known_prepare_fallback_count += 1
                        elif r._current_forward_mode == "decode":
                            self.token_chunk_cpu_known_decode_policy_skips += 1
                    with self._forward_expert_lru_scope():
                        if self._prepare_profiler is None:
                            chunk = r._prepare_expert_dispatch_for_core(
                                state, chunk, **prepare_kwargs
                            )
                        else:
                            chunk = self._prepare_profiler.run(
                                r._prepare_expert_dispatch_for_core,
                                state,
                                chunk,
                                **prepare_kwargs,
                            )
                finally:
                    if future_use is not None:
                        if previous_future_use is future_marker:
                            delattr(r, "_layerkv_future_expert_use")
                        else:
                            r._layerkv_future_expert_use = previous_future_use
            self.token_chunk_materializations += (
                r.stats.expert_materialize_count - before
            )
            self._touch_forward_expert_lru(state, current_bits)
            if execution_groups is None:
                # Pull the next route after current residency is known.  It is
                # still submitted before this group's MoE call, so the
                # producer-event dependency and the overlap window remain.
                next_group = next(group_iter, None)
                successor_groups = [next_group] if next_group is not None else []
                successor_prefetch_ids = (
                    self._prefetch_logical_ids_for_groups(successor_groups)
                    if len(successor_groups) > 1
                    else None
                )
            self.record_use(state, chunk)
            if ordered_groups is not None:
                next_bits = next_group[0] if next_group is not None else None
                self._prefetch_next_input_group(
                    state,
                    current_bits,
                    next_bits,
                    prefetch_logical_ids=successor_prefetch_ids,
                    prefetch_group_count=len(successor_groups),
                )
            if probe is not None:
                from .prefetch_probe import observe_group

                probe.append(
                    observe_group(
                        state,
                        current_bits,
                        (
                            groups[group_index + 1][0]
                            if group_index + 1 < len(groups)
                            else None
                        ),
                        group_index,
                        rows,
                    )
                )
            if ordered_groups is None and next_group is not None:
                self._prefetch_next_input_group(
                    state,
                    current_bits,
                    next_group[0],
                    producer_event=previous_moe_done,
                    prefetch_logical_ids=successor_prefetch_ids,
                    prefetch_group_count=len(successor_groups),
                )
            with self._profile_chunk_phase("moe", output.device):
                result = state.orig_run_moe_core(chunk, *args, **kwargs)
            current_moe_done = None
            if output.device.type == "cuda":
                current_moe_done = torch.cuda.Event()
                current_moe_done.record(
                    torch.cuda.current_stream(device=output.device)
                )
            if (
                getattr(r.config, "shared_expert_post_moe_prefetch", False)
                and next_group is not None
                and current_moe_done is not None
                and r._current_forward_mode == "decode"
                and self._post_moe_prefetch_is_worthwhile(
                    state, topk.topk_ids
                )
            ):
                future_live_ids = (
                    tuple(post_moe_remaining_counts)
                    if post_moe_remaining_counts is not None
                    else ()
                )
                self._prefetch_next_input_group(
                    state,
                    0,
                    next_group[0],
                    producer_event=current_moe_done,
                    extra_protected_ids=future_live_ids,
                    post_moe_dead_only=True,
                    prefetch_logical_ids=successor_prefetch_ids,
                    prefetch_group_count=len(successor_groups),
                )
            previous_moe_done = current_moe_done if ordered_groups is None else None
            self.token_chunk_calls += 1
            with self._profile_chunk_phase("scatter", output.device):
                output.index_copy_(0, indices, result.hidden_states)
            current_group = next_group
            group_index += 1
        self._forward_expert_lru = None
        self._forward_expert_lru_clock = 0
        if result is None:
            raise ValueError("shared expert token chunks require nonempty input")
        if r.config.shared_expert_profile_chunks:
            elapsed = (time.perf_counter() - started_total) * 1000
            self._chunk_wall_ms["total"] += elapsed
            self._chunk_by_mode[profile_mode]["wall_ms"]["total"] += elapsed
        return result._replace(hidden_states=output)

    def _virtual_scratch_feature_enabled(self):
        config = getattr(self.runtime, "config", None)
        return bool(
            getattr(config, "shared_expert_lend_virtual_scratch", False)
            and getattr(config, "shared_expert_free_kv_donors", False)
            and getattr(config, "kvc_backend", "") == "per-layer-arena"
        )

    def _virtual_scratch_location_set(self):
        r = self.runtime
        locations = getattr(r, "_virtual_scratch_locs_host", None)
        locs = getattr(r, "_virtual_scratch_locs", None)

        if locations is not None:
            try:
                normalized = {int(value) for value in locations}
            except TypeError:
                normalized = set()
            if normalized or locs is None:
                try:
                    r._virtual_scratch_locs_host = normalized
                except Exception:
                    pass
                return normalized
        if locs is None:
            locations = set()
        elif isinstance(locs, torch.Tensor):
            try:
                locations = {int(value) for value in locs.detach().cpu().tolist()}
            except Exception:
                locations = set()
        else:
            try:
                locations = {int(value) for value in locs}
            except TypeError:
                locations = set()
        try:
            r._virtual_scratch_locs_host = locations
        except Exception:
            pass
        return locations

    def _virtual_scratch_is_idle(self):
        """Return whether scratch can be unmapped without invalidating KVC use."""
        r = self.runtime
        for name in (
            "_pending_kvc_evict_events",
            "_pending_kvc_reload_events",
            "_pending_virtual_kvc_materialize",
        ):
            if getattr(r, name, None):
                self.scratch_lend_skip_reason = f"{name[1:]}-pending"
                return False
        if int(getattr(r, "_per_layer_offloaded_token_count_fast", 0) or 0) > 0:
            self.scratch_lend_skip_reason = "kvc-offloaded"
            return False
        runs_by_key = getattr(r, "_per_layer_offloaded_runs_by_req_layer", {})
        is_current = getattr(r, "_per_layer_entry_is_current", None)
        for runs in runs_by_key.values():
            for entry in runs:
                if getattr(entry, "state", None) != "offloaded":
                    continue
                if callable(is_current) and not is_current(entry):
                    continue
                self.scratch_lend_skip_reason = "kvc-offloaded"
                return False
        # Cache entries contain pointers into scratch. They are safe to discard
        # only after all pending copies have drained (checked above).
        if (
            getattr(r, "_virtual_scratch_cache_by_layer", None)
            or getattr(r, "_virtual_materialize_plan", None) is not None
            or getattr(r, "_virtual_materialize_plans_by_layer", None)
            or getattr(r, "_virtual_materialize_plan_reuse_cache", None)
            or getattr(r, "_virtual_kvc_demands", None)
        ):
            invalidate = getattr(r, "_invalidate_virtual_caches", None)
            if not callable(invalidate):
                self.scratch_lend_skip_reason = "virtual-cache-live"
                return False
            invalidate()
        return True

    def _virtual_scratch_donor_scan_allowed(self):
        if not self._virtual_scratch_feature_enabled():
            return False
        decision = getattr(self, "_scratch_donor_scan_enabled", None)
        return self._virtual_scratch_is_idle() if decision is None else bool(decision)

    def _prepare_virtual_scratch_for_lending(self, *, context_pressure):
        """Gate scratch lending with the context-length side of the policy."""
        self._scratch_donor_scan_enabled = False
        if not self._virtual_scratch_feature_enabled():
            return False
        if context_pressure:
            self.scratch_lend_skip_count += 1
            self.scratch_lend_skip_reason = "context-pressure"
            self._sync_scratch_stats()
            return False
        if not self._virtual_scratch_location_set():
            self.scratch_lend_skip_count += 1
            self.scratch_lend_skip_reason = "scratch-unavailable"
            self._sync_scratch_stats()
            return False
        if not self._virtual_scratch_is_idle():
            self.scratch_lend_skip_count += 1
            self._sync_scratch_stats()
            return False
        self._scratch_donor_scan_enabled = True
        self.scratch_lend_skip_reason = ""
        self._sync_scratch_stats()
        return True

    def _is_virtual_scratch_donor(self, donor):
        locations = self._virtual_scratch_location_set()
        return bool(locations) and locations.issuperset(donor[3])

    def _growth_slots_per_scan(
        self,
        target: int,
        current: int,
        donors,
        pages_per_expert: int,
        *,
        scratch_lending_active: bool,
    ) -> int:
        """Bound one growth scan without stranding an idle scratch budget."""
        remaining = max(0, int(target) - int(current))
        if remaining <= 0:
            return 0
        if self.budget is not None:
            return remaining
        if not scratch_lending_active or pages_per_expert <= 0:
            return 1
        scratch_pages = sum(
            1 for donor in donors if self._is_virtual_scratch_donor(donor)
        )
        # In the short-context/high-batch regime, consume all complete rows
        # currently fundable by private scratch in this scan. The configured
        # target may exceed the scratch budget; falling back to one slot in
        # that case caused a scan and a VMM growth transaction per row after
        # scratch was exhausted. Ordinary KV pages remain conservative: the
        # caller can only select at most the returned number of rows, and the
        # next scan handles any residual target separately.
        scratch_slots = scratch_pages // pages_per_expert
        return min(remaining, scratch_slots) if scratch_slots else 1

    def _donors(self):
        r = self.runtime
        scratch_locations = (
            self._virtual_scratch_location_set()
            if self._virtual_scratch_feature_enabled()
            else set()
        )
        scratch_allowed = self._virtual_scratch_donor_scan_allowed()
        # First rejection wins. not_free alone does not prove fragmentation.
        scan = dict.fromkeys(
            (
                "mapped",
                "edge",
                "cleanup",
                "not_free",
                "not_reserved",
                "allocated",
                "protected",
                "eligible",
                "scratch_mapped",
                "scratch_partial",
                "scratch_busy",
                "scratch_eligible",
            ),
            0,
        )
        # Run indexes retain evicted physical locations. An old location alone
        # is not sufficient: it must ALSO still be free in the layer allocator.
        offloaded_bits = {}
        for runs in r._per_layer_offloaded_runs_by_req_layer.values():
            for entry in runs:
                if not r._per_layer_entry_is_current(entry):
                    continue
                if (
                    entry.state != "offloaded"
                    or len(entry.host_slot_list()) != entry.token_count
                ):
                    continue
                layer = int(entry.layer_id)
                offloaded_bits[layer] = offloaded_bits.get(
                    layer, 0
                ) | r._locs_to_bitset(entry.evicted_device_locs or [], min_value=1)
        if r.config.shared_expert_free_kv_donors:
            for layer in r._kvc_layer_ids():
                offloaded_bits.setdefault(int(layer), 0)
        donors = []
        for layer, bits in sorted(offloaded_bits.items()):
            # New eviction commits publish bit chunks; conservatively ignore
            # other free-list formats rather than infer that a page is safe.
            pending_bits = r._per_layer_pending_overwrite_bits_union(layer)
            bits &= pending_bits
            if r.config.shared_expert_free_kv_donors:
                # Ownership is per layer. A loan in another layer can remove
                # common admission credit without making this layer's page live.
                # Consume only formats accepted by the strict donor claim.
                free = set(r._per_layer_arena_free_locs.get(layer, []))
            cleanup_bits = 0
            for state in r._per_layer_cleanup_state_by_req.values():
                cleanup_bits |= int(state.loc_bits_by_layer.get(layer, 0))
            bits &= ~cleanup_bits
            for buffer in (
                r._kv_pool._get_key_buffer(layer),
                r._kv_pool._get_value_buffer(layer),
            ):
                allocation = self.arena.allocations.get(buffer.data_ptr())
                if allocation is None:
                    raise RuntimeError("KV buffer does not belong to shared VMM")
                token_bytes = int(buffer[0].nbytes)
                if self.arena.page_bytes % token_bytes:
                    raise ValueError("VMM page must contain whole KV token rows")
                tokens = self.arena.page_bytes // token_bytes
                for page in sorted(allocation.pages):
                    scan["mapped"] += 1
                    start = page * tokens
                    end = start + tokens
                    if start == 0 or end > buffer.shape[0]:
                        scan["edge"] += 1
                        continue  # Keep padding and partial tail pages mapped.
                    mask = ((1 << tokens) - 1) << start
                    locations = range(start, end)
                    scratch_page = bool(scratch_locations) and scratch_locations.issuperset(
                        locations
                    )
                    scratch_overlap = scratch_page or (
                        bool(scratch_locations)
                        and any(location in scratch_locations for location in locations)
                    )
                    if scratch_overlap:
                        scan["scratch_mapped"] += 1
                        # Never let a page containing scratch fall through to
                        # ordinary free-list handling. A partial page is kept
                        # mapped, and a busy full page waits for KVC completion.
                        if not scratch_page:
                            scan["scratch_partial"] += 1
                            continue
                        if (
                            not scratch_allowed
                            or cleanup_bits & mask
                            or not r._per_layer_arena_allocated_locs.get(
                                layer, set()
                            ).isdisjoint(locations)
                            or not r._per_layer_arena_protected_locs.get(
                                layer, set()
                            ).isdisjoint(locations)
                        ):
                            scan["scratch_busy"] += 1
                            continue
                        scan["eligible"] += 1
                        scan["scratch_eligible"] += 1
                        donors.append((allocation, page, layer, list(locations)))
                        continue
                    if r.config.shared_expert_free_kv_donors:
                        if cleanup_bits & mask:
                            scan["cleanup"] += 1
                            continue
                        pending_page = pending_bits & mask
                        missing = mask & ~pending_bits
                        # Reject fragmented pages on the first missing address
                        # before expanding a potentially long bitmap suffix.
                        if (
                            pending_page
                            and missing
                            and (missing & -missing).bit_length() - 1 not in free
                        ):
                            scan["not_free"] += 1
                            continue
                        if pending_page != mask and not free.issuperset(
                            locations
                            if not pending_page
                            else r._bitset_to_locs(mask & ~pending_bits)
                        ):
                            scan["not_free"] += 1
                            continue
                        if not r._per_layer_arena_reserved_locs.issuperset(locations):
                            scan["not_reserved"] += 1
                            continue
                        if not r._per_layer_arena_allocated_locs.get(
                            layer, set()
                        ).isdisjoint(locations):
                            scan["allocated"] += 1
                            continue
                        if not r._per_layer_arena_protected_locs.get(
                            layer, set()
                        ).isdisjoint(locations):
                            scan["protected"] += 1
                            continue
                    elif bits & mask != mask:
                        scan["not_free"] += 1
                        continue
                    scan["eligible"] += 1
                    donors.append((allocation, page, layer, list(locations)))
        self._donor_last_scan = scan
        self.scratch_donor_page_count += scan["scratch_eligible"]
        self._sync_scratch_stats()
        return donors

    def _kv_overflow_enabled(self):
        allocator = getattr(self.runtime, "_allocator", None)
        return bool(
            int(getattr(self, "kv_overflow_tokens", 0) or 0) > 0
            and allocator is not None
            and callable(getattr(allocator, "activate_tail", None))
            and callable(getattr(allocator, "deactivate_tail", None))
        )

    def _minimum_expert_slots_for_batch(self, batch_size):
        """Keep a two-request routed working set resident when possible.

        ``kv_min_slots`` only guarantees that one token's top-k route fits. For
        a multi-request batch, shrinking to that minimum can make every token
        pay a materialization round-trip even when the base expert allocation
        would have fit the routed working set. Use the model's top-k and the
        actual request batch, not a dataset-specific expert-id set, to retain
        a conservative two-route-width floor. The floor is capped at the base
        allocation so a single request can still donate the extra tail.
        """
        state = self.state
        if state is None:
            return max(0, int(getattr(self, "kv_min_slots", 0) or 0))
        batch_size = max(1, int(batch_size or 1))
        module = getattr(state, "module", None)
        top_k = int(getattr(module, "top_k", 0) or 0)
        if top_k <= 0:
            runner_config = getattr(module, "moe_runner_config", None)
            top_k = int(getattr(runner_config, "top_k", 0) or 0)
        top_k = max(1, top_k)
        base_slots = max(
            1,
            int(
                getattr(
                    self,
                    "base_slots",
                    getattr(state, "slot_capacity", top_k),
                )
                or top_k
            ),
        )
        kv_min_slots = max(0, int(getattr(self, "kv_min_slots", 0) or 0))
        route_floor = top_k * min(2, batch_size)
        return min(base_slots, max(kv_min_slots, route_floor))

    def _minimum_expert_slots_for_admission(self, batch_size, context_tokens):
        """Return the route floor for a physical long-context admission.

        Decode-time residency keeps two route widths resident to avoid the
        repeated materialization pattern seen with the old E=8 arm.  A queued
        long-context request is a different decision: if its candidate batch
        is just beyond the base route width, a small physical KV shortage can
        be funded by moving only the page-feasible expert tail.  Keep one
        route width as the hard safety floor and let the allocator determine
        how many additional slots are actually needed.
        """
        floor = self._minimum_expert_slots_for_batch(batch_size)
        if context_tokens is None or int(context_tokens) <= 2048:
            return floor
        state = self.state
        if state is None:
            return floor
        module = getattr(state, "module", None)
        top_k = int(getattr(module, "top_k", 0) or 0)
        if top_k <= 0:
            runner_config = getattr(module, "moe_runner_config", None)
            top_k = int(getattr(runner_config, "top_k", 0) or 0)
        top_k = max(1, top_k)
        base_slots = max(
            1,
            int(
                getattr(
                    self,
                    "base_slots",
                    getattr(state, "slot_capacity", top_k),
                )
                or top_k
            ),
        )
        if max(1, int(batch_size or 1)) * top_k <= base_slots:
            # A physical admission shortage is the reason to fund the
            # waiting request now.  Do not reuse the decode-time two-route
            # floor here: for batch=2 and base=16 that floor equals the
            # current capacity and makes every page-feasible overflow plan a
            # no-op.  Retain one route width as the hard safety floor while
            # allowing the page selector to reclaim the remaining tail.
            return min(
                base_slots,
                max(0, int(getattr(self, "kv_min_slots", 0) or 0), top_k),
            )
        return min(
            base_slots,
            max(0, int(getattr(self, "kv_min_slots", 0) or 0), top_k),
        )

    def _apply_batch_residency_floor(self, target, batch_size):
        """Prevent budget pressure branches from bypassing the route floor."""
        target = int(target)
        floor = self._minimum_expert_slots_for_batch(batch_size)
        if target >= floor:
            return target
        budget = getattr(self, "budget", None)
        if budget is not None:
            budget.target = floor
            budget.reason = "batch-route-floor"
        return floor

    def _long_context_small_batch(self, batch_size, context_tokens):
        """Identify long-context batches whose routed width fits base slots."""
        if context_tokens is None or int(context_tokens) <= 2048:
            return False
        state = self.state
        if state is None:
            return False
        module = getattr(state, "module", None)
        top_k = int(getattr(module, "top_k", 0) or 0)
        if top_k <= 0:
            runner_config = getattr(module, "moe_runner_config", None)
            top_k = int(getattr(runner_config, "top_k", 0) or 0)
        top_k = max(1, top_k)
        base_slots = max(
            1,
            int(
                getattr(
                    self,
                    "base_slots",
                    getattr(state, "slot_capacity", top_k),
                )
                or top_k
            ),
        )
        return max(1, int(batch_size or 1)) * top_k <= base_slots

    def _select_kv_overflow_target(self, shortage_tokens):
        """Keep the most expert slots that can fund this admission shortage.

        The configured KV overflow is a cap, not a requirement to shrink all
        the way to ``kv_min_slots``.  A partially shrunk expert allocation can
        still map enough physical pages for a small shortage, avoiding the
        chunk fragmentation caused by an unnecessarily small MoE capacity.
        The destination page query is the same one used by activation, so this
        is a physical-page feasibility check rather than an MB estimate.
        """
        fallback = max(0, int(getattr(self, "kv_min_slots", 0) or 0))
        shortage_tokens = max(0, int(shortage_tokens or 0))
        if (
            shortage_tokens <= 0
            or shortage_tokens > int(getattr(self, "kv_overflow_tokens", 0) or 0)
            or self.state is None
        ):
            return fallback
        state = self.state
        old_capacity = int(state.slot_capacity)
        if getattr(self.arena, "loans", None):
            # _shrink_expert_to_kv() recalls all ordinary loans before
            # selecting the overflow suffix.
            old_capacity = self.base_slots
        if old_capacity <= fallback:
            return fallback
        destination_query = getattr(
            self.arena, "_kv_overflow_destinations", None
        )
        total_size = getattr(self.runtime, "_allocator_total_size", None)
        if not callable(destination_query) or not callable(total_size):
            return fallback
        try:
            pages_per_expert = sum(
                pages for _allocation, pages in self._expert_page_layouts(state)
            )
            base_tokens = int(total_size())
            required_pages = len(
                destination_query(base_tokens, shortage_tokens)[0]
            )
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            # A missing/ambiguous physical layout must retain the old safe
            # lower bound.  Admission will then fail closed if it is still
            # insufficient.
            return fallback
        if pages_per_expert <= 0:
            return fallback
        if required_pages <= 0:
            # The allocator's token-level headroom can be smaller than one
            # physical KV page.  There is no page to fund in that case, so a
            # partial-pressure decision must not turn into an unnecessary
            # expert shrink (and subsequent no-op recall).
            return old_capacity
        for target in range(old_capacity - 1, fallback - 1, -1):
            if (old_capacity - target) * pages_per_expert >= required_pages:
                return target
        return fallback

    def _align_admission_overflow_tokens(self, requested_tokens):
        """Round admission overflow to the first usable physical KV page.

        ``kvc_block_tokens`` is a scheduler planning unit, not necessarily the
        physical page granularity of the sparse overflow arena.  Rounding every
        admission request to that block can move more expert pages than the
        allocator needs.  Use the real VMM destination query when available and
        retain block rounding only as the safe fallback for incomplete test or
        startup state.
        """
        requested_tokens = max(0, int(requested_tokens or 0))
        overflow_cap = max(0, int(getattr(self, "kv_overflow_tokens", 0) or 0))
        requested_tokens = min(requested_tokens, overflow_cap)
        if requested_tokens <= 0:
            return 0
        block_tokens = max(
            1,
            int(
                getattr(
                    getattr(self.runtime, "config", None),
                    "kvc_block_tokens",
                    1,
                )
                or 1
            ),
        )
        block_aligned = min(
            overflow_cap,
            ((requested_tokens + block_tokens - 1) // block_tokens)
            * block_tokens,
        )
        destination_query = getattr(
            getattr(self, "arena", None), "_kv_overflow_destinations", None
        )
        total_size = getattr(self.runtime, "_allocator_total_size", None)
        if not callable(destination_query) or not callable(total_size):
            return block_aligned
        try:
            base_tokens = int(total_size())

            def has_destination(tokens):
                return bool(destination_query(base_tokens, int(tokens))[0])

            if has_destination(requested_tokens):
                return requested_tokens
            if not has_destination(overflow_cap):
                return block_aligned
            lo, hi = requested_tokens, overflow_cap
            while lo < hi:
                mid = (lo + hi) // 2
                if has_destination(mid):
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            return block_aligned

    def _context_pressure_overflow_plan(
        self,
        context_demand_tokens,
        context_capacity_tokens,
        reserve_tokens,
        batch_size=None,
    ):
        """Return the smallest physical overflow plan for soft KV pressure.

        Context pressure includes the controller's decode headroom reserve, so
        it can be positive even when the native allocator has not rejected a
        request. Do not turn that small deficit into an unconditional
        ``kv_min_slots`` shrink. Reuse the exact page-feasibility selector used
        by admission and request only the residual pressure tokens.
        """
        if (
            int(getattr(self, "kv_overflow_tokens", 0) or 0) <= 0
            or getattr(self, "kv_overflow_segment", None) is not None
            or self.state is None
            or context_demand_tokens is None
            or context_capacity_tokens is None
        ):
            return None, None
        pressure_tokens = max(
            0,
            int(context_demand_tokens)
            + max(0, int(reserve_tokens))
            - int(context_capacity_tokens),
        )
        remaining_tokens = max(
            0,
            int(getattr(self, "kv_overflow_tokens", 0) or 0)
            - int(getattr(self, "kv_overflow_active_tokens", 0) or 0),
        )
        requested_tokens = min(pressure_tokens, remaining_tokens)
        if requested_tokens <= 0:
            return None, None
        target = int(self._select_kv_overflow_target(requested_tokens))
        if batch_size is not None:
            # The budget API accepts context targets only up to the base
            # allocation. Clamp the physical target before handing it to the
            # policy, and preserve a no-shrink target when the route floor is
            # already at the current capacity. Otherwise ResidencyBudget's
            # ``None`` fallback would shrink all the way to kv_min_slots.
            target = max(
                target, self._minimum_expert_slots_for_batch(batch_size)
            )
            target = min(
                target,
                int(
                    getattr(
                        self,
                        "base_slots",
                        getattr(self.state, "slot_capacity", target),
                    )
                ),
            )
            if target >= int(self.state.slot_capacity):
                target = min(
                    int(self.state.slot_capacity),
                    int(
                        getattr(
                            self,
                            "base_slots",
                            self.state.slot_capacity,
                        )
                    ),
                )
        return (
            target,
            int(requested_tokens),
        )

    def prepare_kv_overflow_for_admission(
        self, *, context_tokens, batch_size, shortage_tokens=None
    ):
        """Create KV capacity before a long-context request is rejected.

        This is deliberately narrower than the decode-time pressure path:
        activation requires a real shortage, and the admission caller supplies
        both the request context and current batch estimate.  The requested
        overflow is the residual shortage plus bounded batch headroom, and the
        expert target is selected from physical page feasibility.  Short
        contexts retain expert pages for reuse; context length alone is not a
        reason to move weights.
        """
        context_tokens = max(0, int(context_tokens))
        batch_size = max(1, int(batch_size))
        shortage_value = (
            None if shortage_tokens is None else max(0, int(shortage_tokens))
        )
        self.admission_attempt_count = int(
            getattr(self, "admission_attempt_count", 0) or 0
        ) + 1
        decision = {
            "context_tokens": context_tokens,
            "batch_size": batch_size,
            "shortage_tokens": shortage_value,
            "state_slot_capacity": (
                int(self.state.slot_capacity) if self.state is not None else None
            ),
        }
        self.last_admission_decision = decision
        # Physical admission runs before the next forward.  Shared experts
        # are normally installed lazily at decode-time, so a first long
        # prefill can reach this callback while every controller still has no
        # state.  Materialize the VMM-backed expert allocation before judging
        # the overflow target; otherwise the callback is reached but can only
        # fail closed with zero reclaimed KV capacity.
        if self.state is None:
            self.ensure_installed()
            decision["installed_for_admission"] = True
        if (
            not self._kv_overflow_enabled()
            or getattr(self, "kv_overflow_segment", None) is not None
            or self.state is None
        ):
            decision["reason"] = "disabled-or-active-overflow"
            return 0
        small_batch_cutoff = max(1, int(self.state.slot_capacity) // 2)
        if context_tokens <= 2048 or batch_size > small_batch_cutoff:
            decision["reason"] = "context-or-batch-cutoff"
            decision["small_batch_cutoff"] = small_batch_cutoff
            return 0
        before = int(getattr(self, "kv_overflow_active_tokens", 0) or 0)
        target = getattr(self, "kv_min_slots", 0)
        request_tokens = None
        if shortage_tokens is not None:
            # A scheduler shortage covers the request being considered, but
            # the newly admitted batch also needs a bounded decode reserve.
            # Without this margin the allocator can accept the request and
            # still split the intended batch on its next scheduling pass.
            headroom_steps = max(
                0,
                int(
                    getattr(
                        getattr(self.runtime, "config", None),
                        "shared_expert_headroom_steps",
                        0,
                    )
                    or 0
                ),
            )
            request_tokens = min(
                int(getattr(self, "kv_overflow_tokens", 0) or 0),
                max(0, int(shortage_tokens))
                + max(1, batch_size) * headroom_steps,
            )
            request_tokens = self._align_admission_overflow_tokens(
                request_tokens
            )
            target = self._select_kv_overflow_target(request_tokens)
        admission_floor = self._minimum_expert_slots_for_admission(
            batch_size, context_tokens
        )
        target = max(int(target), admission_floor)
        decision.update(
            request_tokens=request_tokens,
            selected_target=int(target),
            admission_floor=int(admission_floor),
        )
        if target >= int(self.state.slot_capacity):
            decision["reason"] = "target-not-below-current"
            return 0
        if shortage_tokens is None:
            shrunk = self._shrink_expert_to_kv(target)
        else:
            shrunk = self._shrink_expert_to_kv(
                target, requested_tokens=request_tokens
        )
        if not shrunk:
            decision["reason"] = "physical-transfer-failed"
            return 0
        self.kv_overflow_admission_activation_count = int(
            getattr(self, "kv_overflow_admission_activation_count", 0)
        ) + 1
        decision["reason"] = "activated"
        decision["active_tokens"] = int(
            getattr(self, "kv_overflow_active_tokens", 0) or 0
        )
        return max(0, int(getattr(self, "kv_overflow_active_tokens", 0)) - before)

    def _expert_page_layouts(self, state):
        layouts = []
        for name in state.param_names:
            tensor = getattr(state.module, name).data
            allocation = self.expert_allocations[tensor.data_ptr()]
            row_bytes = int(tensor[0].nbytes)
            if row_bytes % self.arena.page_bytes:
                raise ValueError("shared expert rows must remain VMM-page aligned")
            layouts.append((allocation, row_bytes // self.arena.page_bytes))
        return layouts

    def _shrink_expert_to_kv(self, target, *, requested_tokens=None):
        """Move unused expert rows into a sparse KV tail under context pressure."""
        if not self._kv_overflow_enabled() or getattr(
            self, "kv_overflow_segment", None
        ) is not None:
            return False
        r, state = self.runtime, self.state
        target = max(self.kv_min_slots, int(target))
        old_capacity = int(state.slot_capacity)
        if target >= old_capacity:
            return False
        # Existing KV->expert loans use the same expert allocation. Return them
        # first so the overflow transfer owns a clean suffix of every row.
        if self._owned_loan_count() > 0:
            self.recall()
            old_capacity = int(state.slot_capacity)
            if target >= old_capacity:
                return False

        r._finalize_expert_materialize_events(block=True)
        cache_accounting = r._get_expert_backing_cache_accounting(state)
        evicted = sorted(
            (
                int(logical_id),
                int(slot_id),
            )
            for logical_id, slot_id in state.logical_to_slot.items()
            if int(slot_id) >= target
        )
        missing = [
            (logical_id, slot_id)
            for logical_id, slot_id in evicted
            if logical_id not in state.cpu_params
        ]
        if missing:
            state.cpu_params.update(r._copy_slots_to_cpu_batched(state, missing))
        if state.device.type == "cuda":
            torch.cuda.synchronize(state.device)

        layouts = self._expert_page_layouts(state)
        source_pages = [
            (allocation, page)
            for allocation, pages_per_row in layouts
            for page in range(
                target * pages_per_row, old_capacity * pages_per_row
            )
        ]
        base_tokens = int(r._allocator_total_size())
        remaining_tokens = max(
            0, self.kv_overflow_tokens - self.kv_overflow_active_tokens
        )
        if requested_tokens is not None:
            remaining_tokens = min(
                remaining_tokens, max(0, int(requested_tokens))
            )
        segment, _unused = self.arena.activate_kv_overflow(
            source_pages,
            base_tokens=base_tokens,
            requested_tokens=remaining_tokens,
        )
        transfers = tuple(segment["transfers"])
        actual_tokens = int(segment["tokens"])
        if actual_tokens <= 0 or not transfers:
            self.arena.deactivate_kv_overflow(segment)
            return False
        try:
            r._allocator.activate_tail(actual_tokens)
            if not r._ensure_per_layer_physical_arena(actual_tokens):
                raise RuntimeError(
                    "LayerKV could not publish the physically mapped KV overflow tail"
                )
        except Exception:
            if int(r._allocator_total_size()) == base_tokens + actual_tokens:
                r._allocator.deactivate_tail(actual_tokens)
            self.arena.deactivate_kv_overflow(segment)
            raise

        for logical_id, slot_id in evicted:
            state.logical_to_slot.pop(logical_id, None)
            state.slot_to_logical.pop(slot_id, None)
            state.lru.pop(logical_id, None)
            state.backing_lru.pop(logical_id, None)
            if cache_accounting is not None:
                r._update_expert_backing_cache_accounting(state, logical_id)
            if state.remap_tensor is not None:
                state.remap_tensor[logical_id] = -1
        for name in state.param_names:
            param = getattr(state.module, name)
            allocation = self.expert_allocations[param.data_ptr()]
            param.data = allocation.tensor((target,) + tuple(param.shape[1:]))
        state.free_slots = [
            slot for slot in range(target) if slot not in state.slot_to_logical
        ]
        heapq.heapify(state.free_slots)
        state.lru_heap = [
            (state.lru[logical_id], slot_id, logical_id)
            for logical_id, slot_id in state.logical_to_slot.items()
        ]
        heapq.heapify(state.lru_heap)
        self._set_capacity(target)
        segment.update(
            {
                "expert_capacity": old_capacity,
                "expert_target": target,
                "evicted_logical_ids": [logical_id for logical_id, _ in evicted],
            }
        )
        self.kv_overflow_segment = segment
        self.kv_overflow_active_tokens += actual_tokens
        self.kv_overflow_activation_count += 1
        self.kv_overflow_expert_slots_reclaimed += old_capacity - target
        r._refresh_per_layer_allocator_stats()
        r._refresh_expert_stats()
        return True

    def _kv_overflow_tail_is_free(self, segment):
        r = self.runtime
        allocator = getattr(r, "_allocator", None)
        if allocator is None:
            return False
        base = int(segment["base_tokens"])
        count = int(segment["tokens"])
        if count <= 0 or int(r._allocator_total_size()) != base + count:
            return False
        tail = set(range(base + 1, base + count + 1))
        reserved = tail.intersection(r._per_layer_arena_reserved_locs)
        r._ensure_per_layer_common_free_current()
        if not reserved.issubset(r._common_per_layer_reusable_locs()):
            return False
        for layer_id in r._kvc_layer_ids():
            layer_id = int(layer_id)
            if tail.intersection(
                r._per_layer_arena_allocated_locs.get(layer_id, set())
            ) or tail.intersection(
                r._per_layer_arena_protected_locs.get(layer_id, set())
            ):
                return False
            if reserved and not reserved.issubset(
                set(r._per_layer_arena_free_locs.get(layer_id, []))
            ):
                return False
            if tail.intersection(
                r._per_layer_arena_overwrite_pending_locs.get(layer_id, [])
            ):
                return False
        native_free = set(
            int(loc) for loc in allocator.free_pages.detach().cpu().tolist()
        )
        release_pages = getattr(allocator, "release_pages", None)
        if isinstance(release_pages, torch.Tensor):
            native_free.update(int(loc) for loc in release_pages.detach().cpu().tolist())
        return tail.difference(reserved).issubset(native_free)

    def _restore_expert_from_kv(self):
        """Return a fully free KV tail and rematerialize its expert rows."""
        segment = self.kv_overflow_segment
        if segment is None:
            return False
        if not self._kv_overflow_tail_is_free(segment):
            self.kv_overflow_restore_skip_count += 1
            return False
        r, state = self.runtime, self.state
        r._finalize_expert_materialize_events(block=True)
        base = int(segment["base_tokens"])
        count = int(segment["tokens"])
        tail = set(range(base + 1, base + count + 1))
        reserved_tail = sorted(tail.intersection(r._per_layer_arena_reserved_locs))
        self.arena.deactivate_kv_overflow(segment)
        if reserved_tail:
            released = r._release_specific_common_per_layer_locs_to_native(
                reserved_tail
            )
            if released != len(reserved_tail):
                raise RuntimeError("failed to release the KV overflow tail ledger")
        r._allocator.deactivate_tail(count)

        old_capacity = int(segment["expert_capacity"])
        target = int(segment["expert_target"])
        for name in state.param_names:
            param = getattr(state.module, name)
            allocation = self.expert_allocations[param.data_ptr()]
            param.data = allocation.tensor((old_capacity,) + tuple(param.shape[1:]))
        self._set_capacity(old_capacity)
        state.free_slots = [
            slot for slot in range(old_capacity) if slot not in state.slot_to_logical
        ]
        heapq.heapify(state.free_slots)
        logical_ids = [int(x) for x in segment["evicted_logical_ids"]]
        if logical_ids:
            r._materialize_experts(state, logical_ids, reason="kv_restore")
        self.kv_overflow_segment = None
        self.kv_overflow_active_tokens = max(
            0, self.kv_overflow_active_tokens - count
        )
        self.kv_overflow_restore_count += 1
        r._refresh_per_layer_allocator_stats()
        r._refresh_expert_stats()
        return True

    def after_decode(self, *, allow_growth=True, force_recall=False):
        if self.state is None:
            return
        r, state = self.runtime, self.state
        config = getattr(r, "config", None)
        overflow_tokens = int(getattr(self, "kv_overflow_tokens", 0) or 0)
        free_kv_donors = bool(
            getattr(config, "shared_expert_free_kv_donors", False)
        )
        disable_donor_cache = bool(
            getattr(config, "shared_expert_disable_donor_cache", False)
        )
        target = min(state.full_num_experts, self.base_slots + self.extra_slots)
        batch = self._current_residency_batch_size()
        (
            context_demand_tokens,
            context_capacity_tokens,
            context_live_tokens,
            context_waiting_tokens,
        ) = self._projected_kv_residency_demand(r._last_forward_batch)
        headroom_steps = max(
            1,
            int(getattr(config, "shared_expert_headroom_steps", 16) or 16),
        )
        previous_context_reserve = getattr(self, "last_context_reserve_tokens", None)
        self.last_context_live_tokens = context_live_tokens
        self.last_context_waiting_tokens = context_waiting_tokens
        self.last_context_demand_tokens = context_demand_tokens
        self.last_context_capacity_tokens = context_capacity_tokens
        self.last_context_reserve_tokens = max(1, batch) * headroom_steps
        context_pressure = bool(
            context_demand_tokens is not None
            and context_capacity_tokens is not None
            and int(context_demand_tokens) + self.last_context_reserve_tokens
            > int(context_capacity_tokens)
        )
        if context_pressure:
            self.context_pressure_count += 1
        if force_recall:
            # The all-layer manager uses this path when the shared arena is
            # under KV/admission pressure.  Return this controller's pages
            # before any other policy can claim a donor; preserving KV
            # headroom has priority over an additional expert slot.
            if self._owned_loan_count() > 0:
                self.recall()
            return
        if not allow_growth:
            # A non-selected layer in the all-layer global plan keeps its
            # current residency, but does not scan donors or grow this step.
            return
        context_overflow_target, context_overflow_tokens = (
            self._context_pressure_overflow_plan(
                context_demand_tokens,
                context_capacity_tokens,
                self.last_context_reserve_tokens,
                batch_size=batch,
            )
            if context_pressure and overflow_tokens > 0
            else (None, None)
        )
        if self.budget is not None:
            if (
                state.slot_capacity == self.base_slots
                and self.budget.target == self.base_slots
                and (
                    r._decode_step < self.budget.next_decision_step
                    or self.budget.samples < self.budget.interval
                )
                and overflow_tokens <= 0
            ):
                # No loan to recall and no eligible growth decision. Admission
                # continues to check real capacity in the scheduler itself.
                return
            free_tokens = None
            waiting_queue_len = int(
                getattr(r.stats, "native_schedule_waiting_queue_len", 0) or 0
            )
            long_context_tokens = max(
                int(context_waiting_tokens or 0),
                int(context_live_tokens or 0),
            )
            waiting_context_pressure = bool(
                waiting_queue_len > 0
                and long_context_tokens > 2048
            )
            if waiting_context_pressure:
                # A queued long-context request is an admission obligation,
                # not spare capacity for expert growth. If extra expert slots
                # already exist, the route-floor clamp below can return them
                # to the physical KV arena before the queue is retried.
                free_tokens = 0
            headroom_decision_due = (
                r._decode_step >= self.budget.next_decision_step
                and self.budget.samples >= self.budget.interval
            )
            headroom_reserve_changed = (
                previous_context_reserve is None
                or int(previous_context_reserve) != self.last_context_reserve_tokens
            )
            if (
                state.slot_capacity > self.base_slots
                and not context_pressure
                and not waiting_context_pressure
                and not (
                    self.admission_shortage_tokens
                    and waiting_queue_len
                )
                and (headroom_decision_due or headroom_reserve_changed)
            ):
                # Existing loans require a fresh pressure check only when the
                # policy can change, or when the batch/headroom reserve changes.
                # Context/queued-admission pressure is handled directly below
                # by ResidencyBudget and the scheduler. The physical grow path
                # still performs its own post-loan proof before claiming pages.
                native_free = (
                    int(r._allocator.available_size())
                    if r._allocator is not None
                    else 0
                )
                reserve = self.last_context_reserve_tokens
                sufficient = native_free >= reserve or (
                    r._has_common_per_layer_reusable_tokens(reserve - native_free)
                )
                # decide only compares capacity against reserve. Supply a proven
                # lower bound, not a full address count; queued shortage still
                # overrides retention independently in the budget policy.
                free_tokens = reserve if sufficient else 0
            target = self.budget.decide(
                r._decode_step,
                benefit_horizon_steps=(
                    min(
                        getattr(config, "shared_expert_benefit_horizon_steps", 0),
                        r._shared_expert_remaining_decode_steps,
                    )
                    if getattr(config, "shared_expert_benefit_horizon_steps", 0)
                    and state.slot_capacity == self.base_slots
                    and getattr(r, "_shared_expert_remaining_decode_steps", None)
                    is not None
                    and not r.stats.native_schedule_waiting_queue_len
                    else None
                ),
                batch_size=batch,
                free_tokens=free_tokens,
                waiting=waiting_queue_len,
                shortage=self.admission_shortage_tokens,
                context_demand_tokens=context_demand_tokens,
                context_capacity_tokens=context_capacity_tokens,
                context_target_slots=context_overflow_target,
                transfer_ms_per_miss=(
                    r.stats.expert_materialize_ms / r.stats.expert_materialize_count
                    if r.stats.expert_materialize_count
                    else None
                ),
                growth_cost_ms=(
                    self.last_growth_ms
                    if state.slot_capacity == self.base_slots
                    else 0.0
                ),
            )
            target = self._apply_batch_residency_floor(target, batch)
            if self._long_context_small_batch(batch, context_live_tokens):
                target = min(target, int(self.base_slots))
                self.budget.target = target
                self.budget.reason = "long-context-base-floor"
            if state.slot_capacity > target:
                # A partial target is valid only when the budget selected the
                # same soft context-pressure plan. If real admission pressure
                # wins the decision, preserve the full reclaim request so the
                # physical shortage path cannot be accidentally under-sized.
                requested_overflow_tokens = (
                    context_overflow_tokens
                    if context_overflow_target is not None
                    and target == context_overflow_target
                    else None
                )
                if not self._shrink_expert_to_kv(
                    target, requested_tokens=requested_overflow_tokens
                ):
                    self.recall()
            if r._decode_step < self._next_donor_scan_step and not context_pressure:
                return
        else:
            if overflow_tokens > 0 and context_pressure:
                target = context_overflow_target or self.kv_min_slots
                self._shrink_expert_to_kv(
                    target, requested_tokens=context_overflow_tokens
                )
        if state.slot_capacity >= target:
            return
        if self.kv_overflow_segment is not None:
            # Do not borrow another KV page while the tail is still serving
            # live context. Restore the expert rows once the tail is free.
            if not self._restore_expert_from_kv():
                return
            target = min(state.full_num_experts, self.base_slots + self.extra_slots)
        if free_kv_donors and self.admission_shortage_tokens:
            if r.stats.native_schedule_waiting_queue_len > 0:
                # Preserve KV capacity while real queued admission is blocked.
                # Do not immediately borrow back pages returned for that demand.
                self.admission_pressure_skip_count += 1
                return
            self.admission_shortage_tokens = 0
        self._prepare_virtual_scratch_for_lending(
            context_pressure=context_pressure
        )
        # Only the normal KV lifecycle may publish eviction completion. Doing so
        # here (even nonblocking) changes metadata during forward-end bookkeeping
        # and caused reproducible multi-request output divergence before lending.
        # Discover already-published donors; cache only insufficient capacity,
        # never positive locations/physical handles. Ordinary-free donors use
        # the runtime's monotonic availability generation as their invalidation
        # contract, just like published overwrite bits.
        if free_kv_donors:
            # Probe native free capacity only when the negative donor result
            # may have changed.  The old order refreshed the per-layer arena
            # on every decode before checking the miss cache, which made the
            # cache avoid the donor scan but not its expensive allocator walk.
            # Include native availability in the key because request cleanup
            # can return raw allocator locations without publishing a
            # per-layer donor generation until this promotion step.
            native_available = (
                int(r._allocator.available_size())
                if r._allocator is not None
                else 0
            )
        else:
            native_available = 0
        cache_enabled = not disable_donor_cache
        scratch_lending_active = bool(self._scratch_donor_scan_enabled)
        donor_key = (
            r._per_layer_offloaded_version,
            r._per_layer_donor_availability_version,
            state.slot_capacity,
            target,
            scratch_lending_active,
            native_available,
        )
        if cache_enabled and self._donor_miss_key == donor_key:
            self.donor_cache_hit_count += 1
            return
        if free_kv_donors:
            # Move existing native free tokens into the per-layer ledger, not
            # new GPU backing. Any new native capacity bumps the same
            # availability generation used by the negative miss cache.
            r._ensure_per_layer_physical_arena(1)
            native_available = (
                int(r._allocator.available_size())
                if r._allocator is not None
                else 0
            )
            donor_key = (
                r._per_layer_offloaded_version,
                r._per_layer_donor_availability_version,
                state.slot_capacity,
                target,
                scratch_lending_active,
                native_available,
            )
        self._donor_miss_key = None
        started = time.perf_counter()
        self.donor_scan_count += 1
        try:
            donors = self._donors()
        finally:
            self.donor_scan_ms += (time.perf_counter() - started) * 1000
            self._scratch_donor_scan_enabled = None
        layouts = []
        for name in state.param_names:
            tensor = getattr(state.module, name).data
            allocation = self.expert_allocations[tensor.data_ptr()]
            pages_per_row = int(tensor[0].nbytes) // self.arena.page_bytes
            layouts.append((allocation, pages_per_row))
        pages_per_expert = sum(pages for _, pages in layouts)
        grow_slots = min(
            self._growth_slots_per_scan(
                target,
                state.slot_capacity,
                donors,
                pages_per_expert,
                scratch_lending_active=scratch_lending_active,
            ),
            len(donors) // pages_per_expert,
        )
        if grow_slots <= 0:
            self.donor_miss_count += 1
            if self.budget is not None:
                self._next_donor_scan_step = r._decode_step + self.budget.interval
            if cache_enabled:
                self._donor_miss_key = donor_key
            return
        requested_pages = grow_slots * pages_per_expert
        if r.config.shared_expert_free_kv_donors:
            select_started = time.perf_counter()
            common_bits = r._locs_to_bitset(r._common_per_layer_reusable_locs())
            max_common_cost = common_bits.bit_count()
            if self.budget is not None:
                native_free = (
                    int(r._allocator.available_size())
                    if r._allocator is not None
                    else 0
                )
                max_common_cost = max(
                    0,
                    max_common_cost
                    + native_free
                    - max(1, batch) * self.budget.headroom_steps,
                )
            candidates = [
                (
                    donor,
                    0
                    if self._is_virtual_scratch_donor(donor)
                    else (((1 << len(donor[3])) - 1) << donor[3][0]) & common_bits,
                )
                for donor in donors
            ]
            donors = []
            claimed_bits = 0
            for _ in range(requested_pages):
                chosen = min(
                    range(len(candidates)),
                    key=lambda i: (candidates[i][1] & ~claimed_bits).bit_count(),
                )
                donor, cost_bits = candidates.pop(chosen)
                if (claimed_bits | cost_bits).bit_count() > max_common_cost:
                    self.budget_headroom_skips += 1
                    break
                donors.append(donor)
                claimed_bits |= cost_bits
            self.donor_select_ms += (time.perf_counter() - select_started) * 1000
        else:
            donors = donors[:requested_pages]
        # Publish only complete expert rows, even when headroom limits a batch.
        grow_slots = len(donors) // pages_per_expert
        if grow_slots <= 0:
            if self.budget is not None:
                self._next_donor_scan_step = r._decode_step + self.budget.interval
            return
        donors = donors[: grow_slots * pages_per_expert]
        new_capacity = state.slot_capacity + grow_slots
        needs = [
            (allocation, page)
            for allocation, pages_per_row in layouts
            for page in range(
                state.slot_capacity * pages_per_row, new_capacity * pages_per_row
            )
        ]
        by_layer = {}
        scratch_donors = []
        for _a, _p, layer, locs in donors:
            donor = (_a, _p, layer, locs)
            if self._is_virtual_scratch_donor(donor):
                scratch_donors.append(donor)
            else:
                by_layer.setdefault(layer, set()).update(locs)
        if self.budget is not None:
            consumed = set().union(*by_layer.values())
            # Arena preparation may have transferred native free capacity into
            # the common ledger. Re-read both disjoint pools to avoid double credit.
            common = r._common_per_layer_reusable_locs()
            native_free = (
                int(r._allocator.available_size()) if r._allocator is not None else 0
            )
            if (
                len(set(common) - consumed) + native_free
                < max(1, batch) * self.budget.headroom_steps
            ):
                self.budget_headroom_skips += 1
                self._next_donor_scan_step = r._decode_step + self.budget.interval
                return
        # Remove every donor token from allocator reuse before unmapping either
        # its K or V page. A failed transfer restores the token availability.
        before = self.arena.create_count
        claimed = {}
        try:
            for layer, locs in by_layer.items():
                claim = (
                    r._claim_per_layer_shared_donor_locs
                    if r.config.shared_expert_free_kv_donors
                    else r._consume_per_layer_overwrite_locs
                )
                if not claim(layer, sorted(locs)):
                    raise RuntimeError("shared VMM donor lost exclusive KV ownership")
                claimed[layer] = locs
                r._per_layer_arena_allocated_locs.setdefault(layer, set()).update(locs)
            self.arena.lend(
                [
                    (src, sp, dst, dp)
                    for (src, sp, _l, _locs), (dst, dp) in zip(donors, needs)
                ],
                owner=self,
            )
        except Exception:
            for layer, locs in claimed.items():
                r._per_layer_arena_allocated_locs[layer].difference_update(locs)
                r._push_per_layer_overwrite_bits(layer, r._locs_to_bitset(locs))
            raise
        for layer, locs in by_layer.items():
            self.blocked.setdefault(layer, set()).update(locs)
        if self._blocked_kv_location_union is not None:
            for locs in by_layer.values():
                self._blocked_kv_location_union.update(locs)
        if scratch_donors:
            for _allocation, _page, _layer, locs in scratch_donors:
                self.scratch_blocked.update(locs)
            self.scratch_lend_count += 1
            self.scratch_lend_page_count += len(scratch_donors)
            self.scratch_current_loan_page_count += len(scratch_donors)
        # Grow only the logical view. Existing expert addresses/data stay put.
        for name in state.param_names:
            param = getattr(state.module, name)
            allocation = self.expert_allocations[param.data_ptr()]
            param.data = allocation.tensor((new_capacity,) + tuple(param.shape[1:]))
        for slot in range(state.slot_capacity, new_capacity):
            heapq.heappush(state.free_slots, slot)
        self._set_capacity(new_capacity)
        self.grow_count += 1
        self.growth_physical_create_count += self.arena.create_count - before
        self.peak_slots = max(self.peak_slots, new_capacity)
        r._refresh_per_layer_allocator_stats()
        r._refresh_expert_stats()

        self.last_growth_ms = (time.perf_counter() - started) * 1000

    def _set_capacity(self, capacity):
        state = self.state
        state.slot_capacity = capacity
        for obj in (
            state.module,
            state.module.moe_runner_config,
            state.module.dispatcher,
        ):
            for attr in (
                "num_experts",
                "num_local_experts",
                "num_local_routed_experts",
            ):
                if hasattr(obj, attr):
                    setattr(obj, attr, capacity)

    def _current_kv_residency_demand(self, forward_batch):
        """Return current KV demand and allocator capacity from CPU metadata.

        This is a context-length signal for the budget policy, not an
        admission authorization. Capacity excludes virtual scratch and
        locations currently lent to expert weights; physical donor eligibility
        and the scheduler still decide whether a loan can actually be made.
        """
        r = self.runtime
        if forward_batch is None or getattr(
            r, "_current_forward_req_lens_batch_id", None
        ) != id(forward_batch):
            return None, None
        pairs = getattr(r, "_current_forward_req_lens", None)
        if not pairs:
            return None, None
        total_size = getattr(r, "_allocator_total_size", None)
        if total_size is None:
            return None, None
        try:
            capacity_tokens = int(total_size())
        except Exception:
            return None, None
        if capacity_tokens <= 0:
            return None, None
        scratch_locs = getattr(r, "_virtual_scratch_locs", None)
        if isinstance(scratch_locs, torch.Tensor):
            capacity_tokens -= int(scratch_locs.numel())
        else:
            try:
                capacity_tokens -= len(scratch_locs) if scratch_locs is not None else 0
            except TypeError:
                pass
        blocked_union = getattr(self, "_blocked_kv_location_union", None)
        if blocked_union is None:
            blocked_union = set()
            for locs in getattr(self, "blocked", {}).values():
                blocked_union.update(int(loc) for loc in locs if int(loc) > 0)
            self._blocked_kv_location_union = blocked_union
        capacity_tokens = max(0, capacity_tokens - len(blocked_union))
        live_tokens = sum(max(0, int(seq_len)) for _req_idx, seq_len in pairs)
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if isinstance(out_cache_loc, torch.Tensor):
            write_tokens = int(out_cache_loc.numel())
        elif out_cache_loc is not None:
            try:
                write_tokens = len(out_cache_loc)
            except TypeError:
                write_tokens = 0
        else:
            write_tokens = 0
        return int(live_tokens) + int(write_tokens), capacity_tokens

    def _projected_kv_residency_demand(self, forward_batch):
        """Include one queued request when deciding whether to retain experts.

        The native scheduler remains the admission authority. This projection
        only lets the expert controller see the next context-sized allocation
        early enough to return borrowed KV pages before the queue reaches a
        hard shortage. Using the largest queued extend keeps long-context
        admission conservative without summing an unbounded queue.
        """
        live_tokens, capacity_tokens = self._current_kv_residency_demand(
            forward_batch
        )
        if live_tokens is None or capacity_tokens is None:
            return None, None, None, 0
        waiting_tokens = max(
            0,
            int(
                getattr(
                    self.runtime, "_last_scheduler_waiting_context_tokens", 0
                )
                or 0
            ),
        )
        return (
            int(live_tokens) + waiting_tokens,
            int(capacity_tokens),
            int(live_tokens),
            waiting_tokens,
        )

    def recall(self):
        if self.budget is not None:
            self.budget.recalled(self.runtime._decode_step)
        if self._owned_loan_count() <= 0:
            self.scratch_blocked.clear()
            self.scratch_current_loan_page_count = 0
            return
        r, state = self.runtime, self.state
        had_scratch_loan = bool(self.scratch_blocked)
        r._finalize_expert_materialize_events(block=True)
        r._get_expert_backing_cache_accounting(state)
        pairs = [
            (logical, slot)
            for logical, slot in state.logical_to_slot.items()
            if slot >= self.base_slots
        ]
        # Preserve immutable weights even if the expert materializer consumed
        # its original CPU backing while bringing it into a borrowed slot.
        missing = [
            (logical, slot)
            for logical, slot in pairs
            if logical not in state.cpu_params
        ]
        state.cpu_params.update(r._copy_slots_to_cpu_batched(state, missing))
        torch.cuda.synchronize(state.device)
        for logical, slot in pairs:
            state.logical_to_slot.pop(logical)
            r._update_expert_backing_cache_accounting(state, logical)
            state.slot_to_logical.pop(slot, None)
            state.lru.pop(logical, None)
            state.remap_tensor[logical] = -1
        if callable(getattr(self.arena, "loan_count", None)):
            self.arena.recall(owner=self)
        else:
            # Compatibility with the minimal fake arena used by CPU-side
            # backing-accounting tests. Such an arena can only represent one
            # controller, so whole-arena recall is equivalent.
            self.arena.recall()
        for name in state.param_names:
            param = getattr(state.module, name)
            allocation = self.expert_allocations[param.data_ptr()]
            param.data = allocation.tensor((self.base_slots,) + tuple(param.shape[1:]))
        self._set_capacity(self.base_slots)
        state.free_slots = [
            slot for slot in range(self.base_slots) if slot not in state.slot_to_logical
        ]
        heapq.heapify(state.free_slots)
        state.lru_heap = [
            (state.lru[logical], slot, logical)
            for logical, slot in state.logical_to_slot.items()
        ]
        heapq.heapify(state.lru_heap)
        for layer, locs in self.blocked.items():
            r._per_layer_arena_allocated_locs[layer].difference_update(locs)
            r._push_per_layer_overwrite_bits(layer, r._locs_to_bitset(locs))
        self.blocked.clear()
        self._blocked_kv_location_union = None
        if had_scratch_loan:
            self.scratch_recall_count += 1
        self.scratch_blocked.clear()
        self.scratch_current_loan_page_count = 0
        self._sync_scratch_stats()
        r._refresh_per_layer_allocator_stats()
        r._refresh_expert_stats()

    def restore_scratch(self):
        """Recall expert loans before a KVC path touches virtual scratch."""
        if not self.scratch_blocked:
            return False
        self.recall()
        return True

    def summary(self, *, include_arena=True):
        self._collect_chunk_profile()
        result = self.arena.summary() if include_arena else {}
        if self.budget is not None:
            result["residency_budget"] = {
                "target_slots": self.budget.target,
                "reason": self.budget.reason,
                "decisions": self.budget.decisions,
                "shadow_saved_misses": self.budget.saved_misses,
                "headroom_skips": self.budget_headroom_skips,
                "interval": self.budget.interval,
                "headroom_steps": self.budget.headroom_steps,
                "cost_rejections": self.budget.cost_rejections,
                "estimated_saved_ms": self.budget.estimated_saved_ms,
                "forecast_steps": self.budget.forecast_steps,
                "context_pressure_count": self.budget.context_pressure_count,
                "last_context_live_tokens": self.last_context_live_tokens,
                "last_context_waiting_tokens": self.last_context_waiting_tokens,
                "last_context_demand_tokens": self.budget.last_context_demand_tokens,
                "last_context_capacity_tokens": (
                    self.budget.last_context_capacity_tokens
                ),
                "last_context_reserve_tokens": self.budget.last_context_reserve_tokens,
                "remaining_decode_steps": getattr(
                    self.runtime, "_shared_expert_remaining_decode_steps", None
                ),
                "last_growth_ms": self.last_growth_ms,
                "observation_model": "input-groups-protected-lru-proxy",
                "grouped_samples": self.budget.grouped_samples,
                "grouped_base_misses": self.budget.grouped_misses[0],
                "grouped_max_misses": self.budget.grouped_misses[1],
            }
        if self.runtime.config.shared_expert_trace_waits:
            result["prefetch_probe"] = self._prefetch_probe
        if self._prepare_profiler is not None:
            result["prepare_profile"] = self._prepare_profiler.summary()
        result.update(
            {f"token_chunk_{k}_wall_ms": v for k, v in self._chunk_wall_ms.items()}
        )
        result.update(
            {f"token_chunk_{k}_stream_ms": v for k, v in self._chunk_gpu_ms.items()}
        )
        result.update(
            expert_layer=(
                self.layer_id
                if self.layer_id is not None
                else self.runtime.config.shared_expert_layer
            ),
            initial_resident_expert_ids=self.initial_resident_expert_ids,
            initial_resident_source=self.initial_resident_source,
            retain_across_requests=self.runtime.config.shared_expert_retain_across_requests,
            admission_policy=self.runtime.config.shared_expert_admission_policy,
            initial_expert_slots=self.base_slots,
            current_expert_slots=self.state.slot_capacity if self.state else 0,
            peak_expert_slots=self.peak_slots,
            grow_count=self.grow_count,
            borrowed_slot_use_count=self.borrowed_slot_use_count,
            token_chunk_batches=self.token_chunk_batches,
            token_chunk_calls=self.token_chunk_calls,
            token_chunk_gpu_group_calls=self.token_chunk_gpu_group_calls,
            token_chunk_gpu_group_fallbacks=self.token_chunk_gpu_group_fallbacks,
            token_chunk_gpu_group_rows=self.token_chunk_gpu_group_rows,
            token_chunk_gpu_group_policy_skips=(
                self.token_chunk_gpu_group_policy_skips
            ),
            token_chunk_cpu_known_decode_policy_skips=(
                self.token_chunk_cpu_known_decode_policy_skips
            ),
            token_chunk_materializations=self.token_chunk_materializations,
            token_chunk_reordered_groups=self.token_chunk_reordered_groups,
            token_chunk_adaptive_reuse_count=self.token_chunk_adaptive_reuse_count,
            token_chunk_adaptive_input_count=self.token_chunk_adaptive_input_count,
            token_chunk_adaptive_window_reuse_count=(
                self.token_chunk_adaptive_window_reuse_count
            ),
            token_chunk_indexed_reuse_count=self.token_chunk_indexed_reuse_count,
            token_chunk_post_moe_prefetch_count=(
                self.token_chunk_post_moe_prefetch_count
            ),
            token_chunk_post_moe_prefetch_ids=self.token_chunk_post_moe_prefetch_ids,
            token_chunk_post_moe_dead_slot_skips=(
                self.token_chunk_post_moe_dead_slot_skips
            ),
            token_chunk_post_moe_capacity_skips=(
                self.token_chunk_post_moe_capacity_skips
            ),
            future_eviction_group_count=self.future_eviction_group_count,
            future_eviction_slot_count=self.future_eviction_slot_count,
            token_chunk_eviction_plan_groups=self.token_chunk_eviction_plan_groups,
            token_chunk_eviction_plan_ids=self.token_chunk_eviction_plan_ids,
            token_chunk_eviction_batches=self.token_chunk_eviction_batches,
            token_chunk_eviction_ids=self.token_chunk_eviction_ids,
            token_chunk_eviction_early_ids=self.token_chunk_eviction_early_ids,
            token_chunk_route_reuse_window=self.route_reuse_window,
            token_chunk_selected_order=self._last_token_chunk_order,
            token_chunk_profile_enabled=self.runtime.config.shared_expert_profile_chunks,
            token_chunk_profile_pending=len(self._chunk_events),
            token_chunk_by_mode=self._chunk_by_mode,
            token_chunk_order=self.runtime.config.shared_expert_chunk_order,
            token_chunk_prepare_path=self.runtime.config.shared_expert_prepare_path,
            donor_scan_count=self.donor_scan_count,
            donor_last_scan=getattr(self, "_donor_last_scan", None),
            donor_scan_ms=self.donor_scan_ms,
            donor_select_ms=self.donor_select_ms,
            donor_miss_count=self.donor_miss_count,
            donor_cache_hit_count=self.donor_cache_hit_count,
            donor_finalize_ms=self.donor_finalize_ms,
            virtual_scratch_lending_enabled=self._virtual_scratch_feature_enabled(),
            scratch_donor_page_count=self.scratch_donor_page_count,
            scratch_lend_count=self.scratch_lend_count,
            scratch_lend_page_count=self.scratch_lend_page_count,
            scratch_recall_count=self.scratch_recall_count,
            scratch_lend_skip_count=self.scratch_lend_skip_count,
            scratch_lend_skip_reason=self.scratch_lend_skip_reason,
            scratch_blocked_tokens=len(self.scratch_blocked),
            scratch_loan_pages=self.scratch_current_loan_page_count,
            scratch_loan_bytes=(
                self.scratch_current_loan_page_count * self.arena.page_bytes
            ),
            loaned_kv_tokens=sum(map(len, self.blocked.values()))
            + len(self.scratch_blocked),
            admission_shortage_tokens=self.admission_shortage_tokens,
            admission_pressure_skip_count=self.admission_pressure_skip_count,
            admission_attempt_count=self.admission_attempt_count,
            last_admission_decision=self.last_admission_decision,
            context_pressure_count=self.context_pressure_count,
            last_context_live_tokens=self.last_context_live_tokens,
            last_context_waiting_tokens=self.last_context_waiting_tokens,
            last_context_demand_tokens=self.last_context_demand_tokens,
            last_context_capacity_tokens=self.last_context_capacity_tokens,
            last_context_reserve_tokens=self.last_context_reserve_tokens,
            donor_cache_enabled=not (
                self.runtime.config.shared_expert_disable_donor_cache
                or self.runtime.config.shared_expert_free_kv_donors
            ),
            free_kv_donors_enabled=self.runtime.config.shared_expert_free_kv_donors,
            growth_physical_create_count=self.growth_physical_create_count,
            kv_overflow_enabled=self.kv_overflow_tokens > 0,
            kv_overflow_requested_tokens=self.kv_overflow_tokens,
            kv_overflow_active_tokens=self.kv_overflow_active_tokens,
            kv_overflow_activation_count=self.kv_overflow_activation_count,
            kv_overflow_restore_count=self.kv_overflow_restore_count,
            kv_overflow_restore_skip_count=self.kv_overflow_restore_skip_count,
            kv_overflow_expert_slots_reclaimed=self.kv_overflow_expert_slots_reclaimed,
            kv_overflow_admission_activation_count=(
                self.kv_overflow_admission_activation_count
            ),
            kv_min_expert_slots=self.kv_min_slots,
            blocked_kv_tokens=sum(map(len, self.blocked.values())),
            expert_pointers_stable=self.weight_ptrs is not None
            and all(
                getattr(self.state.module, name).data_ptr() == ptr
                for name, ptr in self.weight_ptrs.items()
            ),
        )
        return result


class SharedExpertManager:
    """Coordinate per-layer controllers over one SharedVMM page budget.

    The controller remains the deep module for one expert layer.  This manager
    is the small all-layer seam: it routes state-local operations, recalls all
    outstanding loans before KVC use, and serializes donor scans so each layer
    cannot accidentally reclaim another layer's pages.
    """

    def __init__(self, runtime, arena, layer_ids):
        self.runtime = runtime
        self.arena = arena
        self.admission_shortage_tokens = 0
        # In all-layer mode ``extra_slots`` is a shared physical-page budget,
        # not a per-layer multiplier.  This is the safe default while the
        # later cost planner can expose a larger explicit budget.
        self.global_extra_slot_budget = max(
            0, int(getattr(runtime.config, "shared_expert_extra_slots", 0) or 0)
        )
        self.pressure_gate_count = 0
        self.pressure_recall_count = 0
        self.global_plan_count = 0
        self.last_global_plan = {}
        self.controllers = {
            int(layer_id): SharedExpertController(
                runtime, arena, layer_id=int(layer_id)
            )
            for layer_id in sorted({int(layer_id) for layer_id in layer_ids})
        }

    @property
    def state(self):
        """Compatibility value for callers that only support one controller."""
        if len(self.controllers) == 1:
            return next(iter(self.controllers.values())).state
        return None

    def controller_for_layer(self, layer_id):
        return self.controllers.get(int(layer_id))

    def controller_for_state(self, state):
        if state is None:
            return None
        return self.controller_for_layer(getattr(state, "layer_id", -1))

    def ensure_installed(self):
        for controller in self.controllers.values():
            controller.ensure_installed()

    def allocate_expert(self, old, slots, *, layer_id=None):
        if layer_id is None:
            if len(self.controllers) != 1:
                raise ValueError("all-layer SharedVMM allocation requires layer_id")
            controller = next(iter(self.controllers.values()))
        else:
            controller = self.controller_for_layer(layer_id)
        if controller is None:
            raise ValueError(f"unknown SharedVMM expert layer {layer_id}")
        return controller.allocate_expert(old, slots)

    def record_use(self, state, dispatch):
        controller = self.controller_for_state(state)
        if controller is not None:
            controller.record_use(state, dispatch)

    def run_token_chunks(self, state, dispatch, *args, **kwargs):
        controller = self.controller_for_state(state)
        if controller is None:
            raise ValueError(f"unknown SharedVMM expert state {state!r}")
        return controller.run_token_chunks(state, dispatch, *args, **kwargs)

    @staticmethod
    def _layer_hotness_score(controller):
        state = controller.state
        if state is None:
            return 0.0
        decode = float(sum(getattr(state, "hotness_decode", {}).values()))
        prefill = float(sum(getattr(state, "hotness_prefill", {}).values()))
        # Decode routes are the strongest signal for retaining a slot during
        # the next allocation decision. Materialization activity is a bounded
        # transfer-cost proxy and avoids preferring an entirely cold layer on
        # a tie before the next hotness snapshot arrives.
        return (
            8.0 * decode
            + prefill
            + 0.5 * float(getattr(controller, "token_chunk_materializations", 0))
            + 0.1 * float(getattr(controller, "borrowed_slot_use_count", 0))
        )

    def _ordered_controllers(self, controllers):
        return sorted(
            controllers,
            key=lambda controller: (
                -self._layer_hotness_score(controller),
                int(controller.layer_id),
            ),
        )

    def _memory_pressure(self, controllers):
        runtime = self.runtime
        if int(getattr(runtime, "_scheduler_pressure_tokens", 0) or 0) > 0:
            return True, "scheduler-pressure"
        if any(
            int(getattr(controller, "admission_shortage_tokens", 0) or 0) > 0
            for controller in controllers
        ):
            return True, "admission-shortage"
        waiting = int(
            getattr(
                getattr(runtime, "stats", None),
                "native_schedule_waiting_queue_len",
                0,
            )
            or 0
        )
        for controller in controllers:
            try:
                demand, capacity, _live, _waiting = (
                    controller._projected_kv_residency_demand(
                        getattr(runtime, "_last_forward_batch", None)
                    )
                )
                if demand is None or capacity is None:
                    continue
                batch = max(1, int(controller._current_residency_batch_size()))
                reserve = batch * max(
                    1,
                    int(
                        getattr(
                            getattr(runtime, "config", None),
                            "shared_expert_headroom_steps",
                            16,
                        )
                        or 16
                    ),
                )
                if int(demand) + reserve > int(capacity):
                    return True, "context-pressure"
            except (AttributeError, TypeError, ValueError):
                continue
        if waiting > 0:
            return True, "queued-admission"
        return False, ""

    def after_decode(self):
        controllers = [
            controller
            for controller in self.controllers.values()
            if controller.state is not None
        ]
        if not controllers:
            return
        pressure, reason = self._memory_pressure(controllers)
        self.global_plan_count += 1
        if pressure:
            self.pressure_gate_count += 1
            recalled = 0
            for controller in controllers:
                before = controller._owned_loan_count()
                controller.after_decode(allow_growth=False, force_recall=True)
                recalled += max(0, before - controller._owned_loan_count())
            if recalled:
                self.pressure_recall_count += recalled
            self.last_global_plan = {
                "reason": reason,
                "pressure": True,
                "ordered_layers": [
                    int(controller.layer_id)
                    for controller in self._ordered_controllers(controllers)
                ],
                "growth_budget_slots": self.global_extra_slot_budget,
                "growth_slots_used": 0,
                "recalled_pages": recalled,
            }
            return

        ordered = self._ordered_controllers(controllers)
        used = sum(
            max(0, int(controller.state.slot_capacity) - int(controller.base_slots))
            for controller in controllers
        )
        grown_layers = []
        for controller in ordered:
            before = int(controller.state.slot_capacity)
            allow_growth = (
                before > int(controller.base_slots)
                or used < self.global_extra_slot_budget
            )
            controller.after_decode(allow_growth=allow_growth)
            after = int(controller.state.slot_capacity)
            used += max(0, after - before)
            if after > before:
                grown_layers.append(int(controller.layer_id))
        self.last_global_plan = {
            "reason": "no-pressure",
            "pressure": False,
            "ordered_layers": [int(controller.layer_id) for controller in ordered],
            "growth_budget_slots": self.global_extra_slot_budget,
            "growth_slots_used": used,
            "grown_layers": grown_layers,
            "recalled_pages": 0,
        }

    def recall(self):
        for controller in self.controllers.values():
            controller.recall()

    def restore_scratch(self):
        restored = False
        for controller in self.controllers.values():
            restored = controller.restore_scratch() or restored
        return restored

    def prepare_kv_overflow_for_admission(
        self, *, context_tokens, batch_size, shortage_tokens
    ):
        # Admission may happen before the first decode forward.  Ensure every
        # all-layer controller owns its compact expert allocation before one
        # layer publishes a shared KV overflow tail.
        self.ensure_installed()
        remaining = max(0, int(shortage_tokens))
        activated = 0
        for controller in self.controllers.values():
            if remaining <= 0:
                break
            current = controller.prepare_kv_overflow_for_admission(
                context_tokens=context_tokens,
                batch_size=batch_size,
                shortage_tokens=remaining,
            )
            activated += max(0, int(current))
            remaining -= max(0, int(current))
        return activated

    def summary(self):
        result = self.arena.summary()
        layer_summaries = {
            str(layer_id): controller.summary(include_arena=False)
            for layer_id, controller in self.controllers.items()
        }
        result["strategy"] = "shared-vmm"
        result["all_layers"] = True
        result["layer_count"] = len(self.controllers)
        result["layers"] = layer_summaries
        result["initial_expert_slots"] = sum(
            int(layer_summary.get("initial_expert_slots", 0) or 0)
            for layer_summary in layer_summaries.values()
        )
        for name in (
            "current_expert_slots",
            "peak_expert_slots",
            "borrowed_slot_use_count",
            "donor_scan_count",
            "donor_miss_count",
            "donor_cache_hit_count",
            "scratch_lend_count",
            "scratch_recall_count",
            "growth_physical_create_count",
            "kv_overflow_activation_count",
            "kv_overflow_restore_count",
            "kv_overflow_admission_activation_count",
            "kv_overflow_expert_slots_reclaimed",
        ):
            result[name] = sum(
                int(layer_summary.get(name, 0) or 0)
                for layer_summary in layer_summaries.values()
            )
        result["loaned_kv_tokens"] = sum(
            int(layer_summary.get("loaned_kv_tokens", 0) or 0)
            for layer_summary in layer_summaries.values()
        )
        result["blocked_kv_tokens"] = sum(
            int(layer_summary.get("blocked_kv_tokens", 0) or 0)
            for layer_summary in layer_summaries.values()
        )
        result["expert_pointers_stable"] = all(
            bool(layer_summary.get("expert_pointers_stable", False))
            for layer_summary in layer_summaries.values()
        )
        result["global_extra_slot_budget"] = self.global_extra_slot_budget
        result["global_plan_count"] = self.global_plan_count
        result["pressure_gate_count"] = self.pressure_gate_count
        result["pressure_recall_count"] = self.pressure_recall_count
        result["last_global_plan"] = self.last_global_plan
        return result
