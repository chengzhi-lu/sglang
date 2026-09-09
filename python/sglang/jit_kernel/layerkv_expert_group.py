"""GPU route grouping for the bounded shared-expert token path."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op


_MAX_MASK_WORDS = 64


@cache_once
def _jit_layerkv_expert_group_module(dtype: torch.dtype):
    args = make_cpp_args(dtype)
    return load_jit(
        "layerkv_expert_group",
        *args,
        cuda_files=["layerkv/expert_group.cuh"],
        cuda_wrappers=[
            ("expert_group", f"LayerKVExpertGroup<{args}>::run"),
            ("expert_group_multi", f"LayerKVExpertGroupMulti<{args}>::run"),
        ],
    )


@register_custom_op(
    mutates_args=[
        "group_ids",
        "group_masks",
        "route_masks",
        "group_count",
        "routing_valid",
        "capacity_violation",
    ]
)
def _jit_layerkv_expert_group_op(
    topk_ids: torch.Tensor,
    group_ids: torch.Tensor,
    group_masks: torch.Tensor,
    route_masks: torch.Tensor,
    group_count: torch.Tensor,
    routing_valid: torch.Tensor,
    capacity_violation: torch.Tensor,
    num_rows: int,
    top_k: int,
    capacity: int,
    full_num_experts: int,
    num_words: int,
) -> None:
    module = _jit_layerkv_expert_group_module(topk_ids.dtype)
    module.expert_group(
        topk_ids,
        group_ids,
        group_masks,
        route_masks,
        group_count,
        routing_valid,
        capacity_violation,
        int(num_rows),
        int(top_k),
        int(capacity),
        int(full_num_experts),
        int(num_words),
    )


@register_custom_op(
    mutates_args=[
        "group_ids",
        "group_masks",
        "group_sizes",
        "route_masks",
        "status",
    ]
)
def _jit_layerkv_expert_group_multi_op(
    topk_ids: torch.Tensor,
    capacities: torch.Tensor,
    group_ids: torch.Tensor,
    group_masks: torch.Tensor,
    group_sizes: torch.Tensor,
    route_masks: torch.Tensor,
    status: torch.Tensor,
    num_rows: int,
    top_k: int,
    full_num_experts: int,
    num_capacities: int,
    num_words: int,
) -> None:
    module = _jit_layerkv_expert_group_module(topk_ids.dtype)
    module.expert_group_multi(
        topk_ids,
        capacities,
        group_ids,
        group_masks,
        group_sizes,
        route_masks,
        status,
        int(num_rows),
        int(top_k),
        int(full_num_experts),
        int(num_capacities),
        int(num_words),
    )


def _mask_words_to_int(words: List[int]) -> int:
    mask = (1 << 64) - 1
    value = 0
    for index, word in enumerate(words):
        value |= (int(word) & mask) << (64 * index)
    return value


def layerkv_group_token_experts(
    topk_ids: torch.Tensor,
    capacity: int,
    full_num_experts: int,
    *,
    device_rows: bool = False,
) -> Tuple[List[List[object]], bool]:
    """Group routes on CUDA and return the compact control result.

    The kernel performs the same deterministic first-fit scan as the generic
    CPU implementation. The full ``[tokens, top_k]`` route matrix is not
    converted with ``tolist()``. With ``device_rows=True``, group masks and
    counts cross back to the host, while each group's row index tensor stays
    on the route device. This avoids rebuilding one CUDA index tensor per
    group in the hot path. Slot admission, materialization and ownership
    remain on the CPU side.
    """
    if (
        topk_ids.device.type != "cuda"
        or topk_ids.ndim != 2
        or topk_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("GPU expert grouping requires a 2-D CUDA integer route tensor")
    num_rows, top_k = (int(x) for x in topk_ids.shape)
    capacity = int(capacity)
    full_num_experts = int(full_num_experts)
    if capacity <= 0 or full_num_experts <= 0:
        raise ValueError("expert grouping requires positive capacity and expert count")
    num_words = (full_num_experts + 63) // 64
    if num_words > _MAX_MASK_WORDS:
        raise ValueError(
            f"GPU expert grouping supports at most {_MAX_MASK_WORDS * 64} experts"
        )

    routes = topk_ids.contiguous().view(-1)
    group_ids = torch.empty(
        (num_rows,), dtype=torch.int32, device=topk_ids.device
    )
    group_masks = torch.empty(
        (num_rows, num_words), dtype=torch.int64, device=topk_ids.device
    )
    route_masks = torch.empty_like(group_masks)
    group_count = torch.zeros((1,), dtype=torch.int32, device=topk_ids.device)
    routing_valid = torch.ones((1,), dtype=torch.int32, device=topk_ids.device)
    capacity_violation = torch.zeros(
        (1,), dtype=torch.int32, device=topk_ids.device
    )
    _jit_layerkv_expert_group_op(
        routes,
        group_ids,
        group_masks,
        route_masks,
        group_count,
        routing_valid,
        capacity_violation,
        num_rows,
        top_k,
        capacity,
        full_num_experts,
        num_words,
    )

    if int(capacity_violation.item()):
        raise ValueError("shared expert slots cannot cover one token's top-k")
    group_count_value = int(group_count.item())
    valid = bool(int(routing_valid.item()))
    if group_count_value == 0:
        return [], valid

    cpu_masks = group_masks[:group_count_value].cpu().tolist()
    if device_rows:
        # Keep row membership on device.  Only the compact per-group counts
        # cross to the host; the stable sort preserves the input row order
        # within each first-fit group.
        counts = torch.bincount(
            group_ids.to(dtype=torch.int64), minlength=group_count_value
        )
        cpu_counts = counts.cpu().tolist()
        sorted_rows = torch.argsort(group_ids, stable=True)
        groups: List[List[object]] = []
        begin = 0
        for mask_words, count in zip(cpu_masks, cpu_counts):
            end = begin + int(count)
            groups.append([_mask_words_to_int(mask_words), sorted_rows[begin:end]])
            begin = end
        if begin != num_rows:
            raise RuntimeError("GPU expert grouping produced inconsistent row counts")
        return groups, valid

    # The compatibility result keeps CPU row lists for callers that use the
    # helper outside the shared-expert hot path.
    cpu_group_ids = group_ids.cpu().tolist()
    groups: List[List[object]] = [
        [_mask_words_to_int(row), []] for row in cpu_masks
    ]
    for row, group_id in enumerate(cpu_group_ids):
        if group_id < 0 or group_id >= group_count_value:
            raise RuntimeError("GPU expert grouping produced an invalid group ID")
        groups[group_id][1].append(row)
    return groups, valid


def layerkv_group_token_experts_multi(
    topk_ids: torch.Tensor,
    capacities: Sequence[int],
    full_num_experts: int,
    *,
    device_rows: bool = False,
) -> List[Tuple[List[List[object]], bool]]:
    """Evaluate several exact first-fit capacities from one route-mask pass.

    The route mask is shared across capacities and the capacity-specific
    first-fit scans run in separate CUDA blocks.  The compact control result
    still crosses to Python because the caller must launch one native MoE call
    per group, but this avoids repeating route-mask construction and kernel
    setup for actual/base/max capacity observations.
    """
    if (
        topk_ids.device.type != "cuda"
        or topk_ids.ndim != 2
        or topk_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("GPU expert grouping requires a 2-D CUDA integer route tensor")
    capacity_values = [int(value) for value in capacities]
    if not capacity_values:
        raise ValueError("GPU expert grouping requires at least one capacity")
    if any(value <= 0 for value in capacity_values):
        raise ValueError("expert grouping capacities must be positive")
    num_rows, top_k = (int(x) for x in topk_ids.shape)
    full_num_experts = int(full_num_experts)
    if full_num_experts <= 0:
        raise ValueError("expert grouping requires a positive expert count")
    num_words = (full_num_experts + 63) // 64
    if num_words > _MAX_MASK_WORDS:
        raise ValueError(
            f"GPU expert grouping supports at most {_MAX_MASK_WORDS * 64} experts"
        )
    if num_rows <= 0:
        return [([], True) for _ in capacity_values]

    num_capacities = len(capacity_values)
    device = topk_ids.device
    routes = topk_ids.contiguous().view(-1)
    capacity_tensor = torch.tensor(
        capacity_values, dtype=torch.int32, device=device
    )
    group_ids = torch.empty(
        (num_capacities, num_rows), dtype=torch.int32, device=device
    )
    group_masks = torch.empty(
        (num_capacities, num_rows, num_words),
        dtype=torch.int64,
        device=device,
    )
    group_sizes = torch.zeros(
        (num_capacities, num_rows), dtype=torch.int32, device=device
    )
    route_masks = torch.empty(
        (num_rows, num_words), dtype=torch.int64, device=device
    )
    # status columns are: group_count, routing_valid, capacity_violation.
    status = torch.zeros((num_capacities, 3), dtype=torch.int32, device=device)
    status[:, 1] = 1
    _jit_layerkv_expert_group_multi_op(
        routes,
        capacity_tensor,
        group_ids,
        group_masks,
        group_sizes,
        route_masks,
        status,
        num_rows,
        top_k,
        full_num_experts,
        num_capacities,
        num_words,
    )

    status_cpu = status.cpu().tolist()
    if any(int(row[2]) for row in status_cpu):
        raise ValueError("shared expert slots cannot cover one token's top-k")
    counts = [int(row[0]) for row in status_cpu]
    valid = [bool(int(row[1])) for row in status_cpu]
    max_count = max(counts, default=0)
    if max_count == 0:
        return [([], item) for item in valid]

    cpu_masks = (
        group_masks[:, :max_count, :].contiguous().cpu().tolist()
    )
    if device_rows:
        sorted_rows = torch.argsort(group_ids, dim=1, stable=True)
        cpu_sizes = group_sizes[:, :max_count].cpu().tolist()
        result = []
        for index, group_count in enumerate(counts):
            groups: List[List[object]] = []
            begin = 0
            for mask_words, size in zip(
                cpu_masks[index][:group_count], cpu_sizes[index][:group_count]
            ):
                end = begin + int(size)
                groups.append(
                    [_mask_words_to_int(mask_words), sorted_rows[index, begin:end]]
                )
                begin = end
            if begin != num_rows:
                raise RuntimeError(
                    "GPU expert grouping produced inconsistent row counts"
                )
            result.append((groups, valid[index]))
        return result

    cpu_group_ids = group_ids.cpu().tolist()
    result = []
    for index, group_count in enumerate(counts):
        groups: List[List[object]] = [
            [_mask_words_to_int(row), []]
            for row in cpu_masks[index][:group_count]
        ]
        for row, group_id in enumerate(cpu_group_ids[index]):
            if group_id < 0 or group_id >= group_count:
                raise RuntimeError("GPU expert grouping produced an invalid group ID")
            groups[group_id][1].append(row)
        result.append((groups, valid[index]))
    return result
