"""CPU-only, bounded demand estimator for shared expert/KV budget selection.

Shadow LRU misses are a benefit proxy, not measured transfer time. The physical
controller must still prove page ownership and post-loan KV headroom.
"""

import math
from collections import OrderedDict
from itertools import combinations


def _mask_subsets(mask, subset_size):
    """Yield unique bit masks of one cardinality from ``mask``."""
    if subset_size < 0:
        return
    bits = []
    remaining = int(mask)
    while remaining:
        bit = remaining & -remaining
        bits.append(bit)
        remaining ^= bit
    for selected in combinations(bits, subset_size):
        subset = 0
        for bit in selected:
            subset |= bit
        yield subset


def _group_token_experts_indexed(bit_rows, capacity):
    """Build exact first-fit groups for routes of width ``capacity - 1``.

    A group starts with ``capacity - 1`` experts and can therefore only be
    either that width or ``capacity`` wide.  A route fits a narrow group when
    they share ``capacity - 2`` experts, and it fits a full group only when the
    route is a subset of that group.  The subset postings find all possible
    groups; choosing the smallest group index preserves first-fit semantics.
    """
    groups = []
    postings = {}

    def register(mask, group_index, subset_size):
        for key in _mask_subsets(mask, subset_size):
            postings.setdefault(key, []).append(group_index)

    def first_compatible(key, bits):
        for group_index in postings.get(key, ()):
            group = groups[group_index]
            if (group[0] | bits).bit_count() <= capacity:
                return group_index
        return None

    for row, bits in enumerate(bit_rows):
        candidates = []
        for key in _mask_subsets(bits, capacity - 2):
            group_index = first_compatible(key, bits)
            if group_index is not None:
                candidates.append(group_index)
        group_index = first_compatible(bits, bits)
        if group_index is not None:
            candidates.append(group_index)

        if candidates:
            group_index = min(candidates)
            group = groups[group_index]
            previous_mask = group[0]
            group[0] |= bits
            group[1].append(row)
            if group[0] != previous_mask:
                register(group[0], group_index, capacity - 1)
        else:
            group_index = len(groups)
            groups.append([bits, [row]])
            register(bits, group_index, capacity - 2)
    return groups


def remaining_fixed_decode_steps(requests):
    """Future steps after this forward, only for explicitly fixed-length work."""
    if not requests:
        return None
    remaining = []
    for req in requests:
        params = getattr(req, "sampling_params", None)
        output = getattr(req, "output_ids", None)
        limit = getattr(params, "max_new_tokens", None)
        if (
            not getattr(params, "ignore_eos", False)
            or any(
                getattr(params, name, None)
                for name in ("stop_strs", "stop_token_ids", "stop_regex_strs")
            )
            or type(limit) is not int
            or output is None
        ):
            return None
        remaining.append(max(0, limit - len(output) - 1))
    return min(remaining)


def group_token_experts(routing_rows, capacity, full_num_experts):
    """First-fit token groups; each token's entire reduction stays together."""
    valid = True
    bit_rows = []
    route_width = None
    for row, ids in enumerate(routing_rows):
        bits = 0
        for logical in ids:
            if 0 <= logical < full_num_experts:
                bits |= 1 << logical
            else:
                valid = False
        if bits.bit_count() > capacity:
            raise ValueError("shared expert slots cannot cover one token's top-k")
        bit_rows.append(bits)
        width = bits.bit_count()
        if route_width is None:
            route_width = width
        elif route_width != width:
            route_width = -1

    if bit_rows and capacity >= 2 and route_width == capacity - 1:
        return _group_token_experts_indexed(bit_rows, capacity), valid

    groups = []
    for row, bits in enumerate(bit_rows):
        for group in groups:
            union = group[0] | bits
            if union.bit_count() <= capacity:
                group[0] = union
                group[1].append(row)
                break
        else:
            groups.append([bits, [row]])
    return groups, valid


class ResidencyBudget:
    def __init__(
        self, base_slots, max_slots, interval=8, headroom_steps=16, min_slots=None
    ):
        if not 0 < base_slots <= max_slots or interval <= 0 or headroom_steps <= 0:
            raise ValueError(
                "invalid residency capacities or decision interval/headroom"
            )
        min_slots = base_slots if min_slots is None else int(min_slots)
        if not 0 < min_slots <= base_slots:
            raise ValueError("invalid minimum residency capacity")
        self.base_slots, self.max_slots = base_slots, max_slots
        self.min_slots = min_slots
        self.interval, self.headroom_steps = interval, headroom_steps
        self.caches = [OrderedDict(), OrderedDict()]
        self.misses = [0, 0]
        self.samples = 0
        self.last_observed_step = -1
        self.next_decision_step = 0
        self.target = base_slots
        self.reason = "warming"
        self.saved_misses = 0
        self.decisions = 0
        self.cost_rejections = 0
        self.estimated_saved_ms = None
        self.forecast_steps = interval
        self.grouped_samples = 0
        self.grouped_misses = [0, 0]
        self.context_pressure_count = 0
        self.last_context_demand_tokens = None
        self.last_context_capacity_tokens = None
        self.last_context_reserve_tokens = None

    def observe(self, step, logical_ids):
        if step == self.last_observed_step:
            return
        self.last_observed_step = step
        # IDs are already CPU-known from the existing chunk-capacity check.
        for index, capacity in enumerate((self.base_slots, self.max_slots)):
            cache = self.caches[index]
            for expert in logical_ids:
                if expert in cache:
                    cache.move_to_end(expert)
                else:
                    self.misses[index] += 1
                    cache[expert] = None
                    if len(cache) > capacity:
                        cache.popitem(last=False)
        self.samples += 1

    def recalled(self, step):
        self.target = self.min_slots
        self.reason = "recalled"
        self.next_decision_step = step + self.interval
        self.samples = 0
        self.misses = [0, 0]

    def observe_rows(self, step, routing_rows, full_num_experts):
        if step == self.last_observed_step:
            return
        self.last_observed_step = step
        self.grouped_samples += 1
        grouped = [
            group_token_experts(routing_rows, capacity, full_num_experts)[0]
            for capacity in (self.base_slots, self.max_slots)
        ]
        self._observe_grouped(grouped)
        self.samples += 1

    def observe_grouped(self, step, base_groups, max_groups):
        """Observe precomputed groups without materializing route rows on CPU.

        GPU grouping already produces the compact expert masks needed by the
        shadow LRU.  Reusing those masks keeps the admission estimator from
        forcing a second ``topk_ids.tolist()`` readback.
        """
        if step == self.last_observed_step:
            return
        self.last_observed_step = step
        self.grouped_samples += 1
        self._observe_grouped((base_groups, max_groups))
        self.samples += 1

    def _observe_grouped(self, grouped):
        for index, groups in enumerate(grouped):
            capacity = (self.base_slots, self.max_slots)[index]
            cache = self.caches[index]
            for bits, _rows in groups:
                pending = int(bits)
                while pending:
                    bit = pending & -pending
                    expert = bit.bit_length() - 1
                    pending ^= bit
                    if expert not in cache:
                        self.misses[index] += 1
                        self.grouped_misses[index] += 1
                        if len(cache) == capacity:
                            victim = next(
                                key for key in cache if not bits & (1 << key)
                            )
                            del cache[victim]
                        cache[expert] = None
                    cache.move_to_end(expert)

    def decide(
        self,
        step,
        *,
        batch_size,
        free_tokens,
        waiting,
        shortage,
        transfer_ms_per_miss=None,
        growth_cost_ms=None,
        benefit_horizon_steps=None,
        context_demand_tokens=None,
        context_capacity_tokens=None,
        context_target_slots=None,
    ):
        for cost in (transfer_ms_per_miss, growth_cost_ms):
            if cost is not None and (not math.isfinite(cost) or cost < 0):
                raise ValueError("residency costs must be finite and nonnegative")
        reserve = max(1, batch_size) * self.headroom_steps
        if (context_demand_tokens is None) != (context_capacity_tokens is None):
            raise ValueError(
                "context demand and capacity must be provided together"
            )
        for name, value in (
            ("context_demand_tokens", context_demand_tokens),
            ("context_capacity_tokens", context_capacity_tokens),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if context_target_slots is not None and (
            type(context_target_slots) is not int
            or not self.min_slots <= context_target_slots <= self.base_slots
        ):
            raise ValueError(
                "context target slots must be an integer between minimum and base"
            )
        if benefit_horizon_steps is not None and (
            type(benefit_horizon_steps) is not int or benefit_horizon_steps < 0
        ):
            raise ValueError("benefit horizon must be a nonnegative integer or None")
        context_pressure = False
        if context_demand_tokens is not None:
            self.last_context_demand_tokens = int(context_demand_tokens)
            self.last_context_capacity_tokens = int(context_capacity_tokens)
            self.last_context_reserve_tokens = int(reserve)
            context_pressure = (
                int(context_demand_tokens) + int(reserve)
                > int(context_capacity_tokens)
            )
        # None defers capacity discovery for a loan-free proposal. The physical
        # controller must still check exact post-loan headroom before claiming.
        if shortage and waiting or (free_tokens is not None and free_tokens < reserve):
            self.target = self.min_slots
            self.reason = "kv-pressure"
            self.next_decision_step = step + self.interval
            # Do not price an entire pressure episode as one future interval.
            self.samples = 0
            self.misses = [0, 0]
            return self.target
        if context_pressure:
            self.context_pressure_count += 1
            if context_target_slots is None:
                self.target = self.min_slots
                self.reason = "context-pressure"
            else:
                self.target = int(context_target_slots)
                self.reason = (
                    "context-pressure-partial"
                    if self.target > self.min_slots
                    else "context-pressure"
                )
            self.next_decision_step = step + self.interval
            self.samples = 0
            self.misses = [0, 0]
            return self.target
        if step < self.next_decision_step or self.samples < self.interval:
            return self.target
        self.saved_misses = self.misses[0] - self.misses[1]
        self.forecast_steps = (
            self.samples if benefit_horizon_steps is None else benefit_horizon_steps
        )
        self.target = self.max_slots if self.saved_misses > 0 else self.base_slots
        self.reason = "expert-reuse" if self.saved_misses > 0 else "no-miss-benefit"
        self.estimated_saved_ms = (
            max(0, self.saved_misses)
            * transfer_ms_per_miss
            * self.forecast_steps
            / self.samples
            if transfer_ms_per_miss is not None
            else None
        )
        # Forecast the observed per-step savings over the caller's bounded
        # future horizon (default: one observed interval), not accumulated past
        # opportunities. None cost permits one calibration growth. An already
        # resident allocation must pass zero transition cost (sunk cost).
        if self.saved_misses > 0 and (
            self.forecast_steps == 0
            or (
                growth_cost_ms is not None
                and (
                    self.estimated_saved_ms is None
                    or self.estimated_saved_ms <= growth_cost_ms
                )
            )
        ):
            self.target = self.base_slots
            self.reason = "cost-exceeds-benefit"
            self.cost_rejections += 1
        self.decisions += 1
        self.next_decision_step = step + self.interval
        self.samples = 0
        self.misses = [0, 0]
        return self.target
