"""Opt-in Kineto phase labels; no CUDA events, waits or policy changes."""

from functools import wraps

import torch


def install_wait_trace(runtime):
    """Wrap this runtime instance only, before serving any requests."""
    for method_name, phase in (
        ("_prepare_expert_dispatch_for_core", "prepare"),
        ("_materialize_experts", "materialize"),
        ("_copy_materialized_experts_batched", "h2d_submit"),
        ("_wait_for_expert_logical_ids_ready", "ready_wait"),
        ("_trim_expert_backing_cache", "cache_trim"),
    ):
        original = getattr(runtime, method_name)

        def wrap(method, label):
            @wraps(method)
            def traced(*args, **kwargs):
                with torch.profiler.record_function(f"layerkv/{label}"):
                    return method(*args, **kwargs)

            return traced

        setattr(runtime, method_name, wrap(original, phase))


def traced_remap_and_guard(state, ids):
    with torch.profiler.record_function("layerkv/remap"):
        mapped = state.remap_tensor[ids.long()]
    with torch.profiler.record_function("layerkv/guard_reduce"):
        invalid = ((mapped < 0) | (mapped >= state.slot_capacity)).any()
    with torch.profiler.record_function("layerkv/readback"):
        invalid = bool(invalid.item())
    return mapped, invalid
