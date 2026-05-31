from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_layerkv_expert_remap_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "layerkv_expert_remap",
        *args,
        cuda_files=["layerkv/expert_remap.cuh"],
        cuda_wrappers=[
            ("expert_remap", f"LayerKVExpertRemap<{args}>::run"),
        ],
    )


@register_custom_op(mutates_args=["rewritten_ids", "missing_ids", "missing_count"])
def _jit_layerkv_expert_remap_op(
    topk_ids: torch.Tensor,
    remap: torch.Tensor,
    rewritten_ids: torch.Tensor,
    missing_ids: torch.Tensor,
    missing_count: torch.Tensor,
    full_num_experts: int,
) -> None:
    module = _jit_layerkv_expert_remap_module(topk_ids.dtype)
    module.expert_remap(
        topk_ids,
        remap,
        rewritten_ids,
        missing_ids,
        missing_count,
        full_num_experts,
    )


def layerkv_expert_remap(
    topk_ids: torch.Tensor, remap: torch.Tensor, full_num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rewritten_ids = torch.empty_like(topk_ids)
    missing_ids = torch.empty(
        (topk_ids.numel(),), dtype=torch.long, device=topk_ids.device
    )
    missing_count = torch.zeros((1,), dtype=torch.int32, device=topk_ids.device)
    if topk_ids.numel() == 0:
        return rewritten_ids, missing_ids, missing_count
    topk_flat = topk_ids.contiguous().view(-1)
    rewritten_flat = rewritten_ids.view(-1)
    _jit_layerkv_expert_remap_op(
        topk_flat,
        remap.contiguous(),
        rewritten_flat,
        missing_ids,
        missing_count,
        int(full_num_experts),
    )
    return rewritten_ids, missing_ids, missing_count
