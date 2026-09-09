"""Read-only, structural one-group lookahead capacity; NOT an admission policy."""


def bit_ids(bits):
    if bits < 0:
        raise ValueError("expert bitset must be nonnegative")
    result = []
    while bits:
        lowest = bits & -bits
        result.append(lowest.bit_length() - 1)
        bits ^= lowest
    return result


def capacity_snapshot(
    capacity,
    current,
    following,
    logical_to_slot,
    slot_to_logical,
    cpu_backed,
    expert_bytes,
):
    """Bound unused slots without changing LRU, mappings, or copy ownership.

    CPU backing presence follows the runtime's published-backing contract.
    Pending DMA/lifetime admission is deliberately NOT proven by this bound.
    """
    if capacity <= 0 or expert_bytes <= 0:
        raise ValueError("positive capacity and expert bytes required")
    if (
        len(set(logical_to_slot.values())) != len(logical_to_slot)
        or {slot: logical for logical, slot in logical_to_slot.items()}
        != slot_to_logical
        or any(not 0 <= slot < capacity for slot in slot_to_logical)
        or not set(current) <= logical_to_slot.keys()
    ):
        raise ValueError("inconsistent expert residency at prefetch probe")
    used = {logical_to_slot[logical] for logical in current}
    result = dict(
        slot_capacity=capacity,
        current_ids=list(current),
        current_slot_ids=sorted(used),
        unused_slot_ids=sorted(set(range(capacity)) - used),
        expert_bytes=expert_bytes,
        has_next=following is not None,
    )
    if following is None:
        return result
    following = set(following)
    next_resident = following & logical_to_slot.keys()
    protected = used | {logical_to_slot[logical] for logical in next_resident}
    missing = following - logical_to_slot.keys()
    candidates = sorted(set(range(capacity)) - protected)
    no_backup = [
        slot
        for slot in candidates
        if slot not in slot_to_logical or slot_to_logical[slot] in cpu_backed
    ]
    sources = sorted(missing & cpu_backed)
    capacity_bound = min(len(candidates), len(missing))
    backed_bound = min(len(no_backup), len(sources))
    result.update(
        next_ids=sorted(following),
        next_resident_ids=sorted(next_resident),
        next_missing_ids=sorted(missing),
        next_missing_cpu_backed_ids=sources,
        candidate_slot_ids=candidates,
        no_d2h_candidate_slot_ids=no_backup,
        capacity_upper_experts=capacity_bound,
        backed_upper_experts=backed_bound,
        backed_upper_bytes=backed_bound * expert_bytes,
    )
    return result


def observe_group(state, current_bits, next_bits, group_index, rows):
    snapshot = capacity_snapshot(
        state.slot_capacity,
        bit_ids(current_bits),
        None if next_bits is None else bit_ids(next_bits),
        state.logical_to_slot,
        state.slot_to_logical,
        set(state.cpu_params),
        state.expert_bytes,
    )
    return dict(group=group_index + 1, tokens=len(rows), **snapshot)
