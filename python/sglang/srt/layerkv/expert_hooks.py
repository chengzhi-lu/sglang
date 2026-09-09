"""LayerKV LayerKVExpertHooksMixin implementation."""

from __future__ import annotations

import functools
import heapq
import json
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

if __package__:
    from .common_types import (
        _LayerKVExpertCopyDescriptor,
        _LayerKVExpertInstallBuild,
        _LayerKVExpertInstallD2HJob,
        _LayerKVExpertInstallItem,
        _LayerKVExpertLayerState,
        _LayerKVPendingExpertCopy,
        _LayerKVPendingExpertD2H,
    )
else:  # pragma: no cover - direct file-loading smoke tests.
    from common_types import (
        _LayerKVExpertCopyDescriptor,
        _LayerKVExpertInstallBuild,
        _LayerKVExpertInstallD2HJob,
        _LayerKVExpertInstallItem,
        _LayerKVExpertLayerState,
        _LayerKVPendingExpertCopy,
        _LayerKVPendingExpertD2H,
    )

logger = logging.getLogger(__name__)

try:
    from sglang.jit_kernel.layerkv_expert_remap import layerkv_expert_remap
except Exception:  # pragma: no cover - optional JIT helper.
    layerkv_expert_remap = None


class LayerKVExpertHooksMixin:
    def _record_expert_decode_route(
        self, state: _LayerKVExpertLayerState, logical_ids: List[int]
    ) -> None:
        """Record route metadata used by the next speculative prefetch.

        The route is intentionally paired with CPU-side request identity and a
        coarse context bucket.  This lets the scheduler reject stale previous
        routes without adding a GPU readback to the decode hot path.
        """
        current_ids = list(
            dict.fromkeys(int(expert_id) for expert_id in logical_ids)
        )
        previous_ids = set(
            int(expert_id) for expert_id in state.last_decode_logical_ids
        )
        if previous_ids and current_ids:
            state.last_decode_route_overlap = len(
                previous_ids.intersection(current_ids)
            ) / float(max(len(previous_ids), len(current_ids)))
        else:
            state.last_decode_route_overlap = 0.0
        state.last_decode_logical_ids = current_ids
        state.last_decode_batch_size = int(
            getattr(self._last_forward_batch, "batch_size", 0) or 0
        )
        state.last_decode_request_signature = tuple(
            getattr(self, "_current_forward_expert_request_signature", ()) or ()
        )
        self._expert_prefetch_dirty_layers.add(int(state.layer_id))

    def _maybe_prepare_expert_plan_during_extend(self, forward_batch: Any) -> None:
        if self._simple_profile_enabled():
            return
        # Avoid duplicating several GB of expert backing during prefill by
        # default.  The decode install path remains bounded and avoids the
        # CPU-memory spikes seen in large-batch runs.
        if (
            str(self.config.expert_cpu_backing_mode) == "none"
            and self._effective_expert_backing_cache_mb() <= 0.0
        ):
            return
        if (
            self._expert_plan_applied
            or self._expert_prepare_started
            or self.config.mode != "kvc-expert"
        ):
            return
        avg_prefix = self._avg_prefix_len(forward_batch)
        is_long_context = avg_prefix > 2048
        if is_long_context:
            # Backing-only prepare was tested on Fig4 context-heavy: it reduced
            # decode apply time but moved more copy work into prefill/extend and
            # regressed end-to-end latency. Keep long-context prepare disabled.
            return
        # Short-context/batch-heavy runs benefit from preparing the CPU backing
        # while prefill is still progressing.  Keep this bounded to avoid the
        # earlier host-memory blowups: long-context is disabled above, and the
        # target is capped by actual expert reclaim capacity below.
        if is_long_context:
            target_mb = min(
                max(0.0, self.config.reclaim_limit_mb),
                max(0.0, self._available_expert_reclaim_mb()),
            )
        else:
            target_mb = self._effective_expert_reclaim_mb(forward_batch)
        if (
            target_mb <= 0
            or not self.physical_expert_supported
            or not self._expert_modules
        ):
            return
        if not self._has_prefill_expert_hotness():
            return
        required_progress = 0.90
        if self._extend_progress_ratio(forward_batch) < required_progress:
            return

        self._expert_prepare_started = True
        profile_name = (
            "profile_prepare_expert_backing_only_ms"
            if is_long_context
            else "profile_prepare_expert_backing_ms"
        )
        with self._profile(profile_name):
            slot_capacities = self._coresid_planned_expert_capacities(
                target_mb, forward_batch
            )
            cpu_params_by_layer: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
            initial_resident_by_layer: Dict[int, List[int]] = {}
            for layer_id, module in self._expert_modules:
                param_names = self._expert_param_names(module)
                full_num_experts = int(module.w13_weight.data.shape[0])
                slot_capacity = slot_capacities[layer_id]
                initial_resident = self._select_initial_resident_experts(
                    layer_id=layer_id,
                    full_num_experts=full_num_experts,
                    slot_capacity=slot_capacity,
                )
                if not is_long_context:
                    initial_resident_by_layer[layer_id] = initial_resident
                resident_set = set(initial_resident)
                to_copy = [
                    expert_id
                    for expert_id in range(full_num_experts)
                    if expert_id not in resident_set
                ]
                layer_cpu_params = self._copy_experts_to_cpu_for_install_batched(
                    module,
                    param_names,
                    to_copy,
                    layer_id=layer_id,
                )
                cpu_params_by_layer[layer_id] = layer_cpu_params

        prepared_mb = sum(
            int(t.nbytes)
            for layer_params in cpu_params_by_layer.values()
            for expert_params in layer_params.values()
            for t in expert_params.values()
        ) / float(1024 * 1024)
        self.stats.expert_prepared_backing_mb = prepared_mb
        self.stats.expert_prepare_guard_mb = 0.0
        self.stats.expert_prepare_extend_count += 1
        self._prepared_expert_plan = {
            "kind": "backing_only" if is_long_context else "full",
            "target_mb": target_mb,
            "slot_capacities": slot_capacities,
            "initial_resident_by_layer": initial_resident_by_layer,
            "cpu_params_by_layer": cpu_params_by_layer,
        }
        self._expert_prepare_done = True

    def _maybe_prepare_expert_plan_after_decode(self, forward_batch: Any) -> None:
        if (
            not self._coresid_optimized_policy_enabled()
            or not self._prepared_expert_backing_enabled()
            or self._expert_plan_applied
            or self._expert_prepare_started
            or self._expert_prepare_done
            or self._expert_install_queue
            or not self.physical_expert_supported
            or not self._expert_modules
        ):
            return
        target_mb = self._effective_expert_reclaim_mb(forward_batch)
        if target_mb <= 0:
            return
        self._ensure_expert_hotness_cpu_view(mode=None)
        if not self._has_initial_expert_hotness():
            return
        self._expert_prepare_started = True
        with self._profile("profile_prepare_expert_backing_ms"):
            slot_capacities = self._coresid_planned_expert_capacities(
                target_mb, forward_batch
            )
            cpu_params_by_layer: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
            initial_resident_by_layer: Dict[int, List[int]] = {}
            copied_count = 0
            reused_count = 0
            candidate_count = 0
            copied_bytes = 0
            for layer_id, module in self._expert_modules:
                param_names = self._expert_param_names(module)
                full_num_experts = int(module.w13_weight.data.shape[0])
                slot_capacity = int(slot_capacities[layer_id])
                initial_resident = self._select_initial_resident_experts(
                    layer_id=layer_id,
                    full_num_experts=full_num_experts,
                    slot_capacity=slot_capacity,
                )
                initial_resident_by_layer[int(layer_id)] = initial_resident
                resident_set = set(int(x) for x in initial_resident)
                to_copy = [
                    int(expert_id)
                    for expert_id in range(full_num_experts)
                    if int(expert_id) not in resident_set
                ]
                candidate_count += len(to_copy)
                layer_cpu_params: Dict[int, Dict[str, torch.Tensor]] = {}
                missing: List[int] = []
                for expert_id in to_copy:
                    global_params = (
                        self._global_expert_backing(layer_id, expert_id)
                        if self._expert_global_cpu_backing
                        else None
                    )
                    if global_params is not None:
                        layer_cpu_params[int(expert_id)] = global_params
                        reused_count += 1
                    else:
                        missing.append(int(expert_id))
                if missing:
                    copied = self._copy_experts_to_cpu_for_install_batched(
                        module,
                        param_names,
                        missing,
                        layer_id=layer_id,
                    )
                    copied_count += len(copied)
                    copied_bytes += sum(
                        self._expert_backing_bytes(params) for params in copied.values()
                    )
                    layer_cpu_params.update(copied)
                cpu_params_by_layer[int(layer_id)] = layer_cpu_params
        prepared_mb = sum(
            int(t.nbytes)
            for layer_params in cpu_params_by_layer.values()
            for expert_params in layer_params.values()
            for t in expert_params.values()
        ) / float(1024 * 1024)
        self.stats.expert_prepare_candidate_count += candidate_count
        self.stats.expert_prepare_copied_count += copied_count
        self.stats.expert_prepare_reused_count += reused_count
        self.stats.expert_prepare_host_budget_mb = float(target_mb)
        self.stats.expert_prepare_actual_mb = prepared_mb
        self.stats.expert_prepared_backing_mb = prepared_mb
        self.stats.expert_prepare_guard_mb = 0.0
        self.stats.expert_prepare_extend_count += 1
        self._prepared_expert_plan = {
            "kind": "full",
            "target_mb": target_mb,
            "slot_capacities": slot_capacities,
            "initial_resident_by_layer": initial_resident_by_layer,
            "cpu_params_by_layer": cpu_params_by_layer,
        }
        self._expert_prepare_done = True

    def _expert_chunked_core_required(
        self, state: _LayerKVExpertLayerState, topk_output: Any
    ) -> bool:
        controller = self._shared_expert_controller_for_state(state)
        if controller is not None:
            # A prefill batch can route to many more experts than one token's
            # top-k. Its fixed physical budget must not use the native growth
            # fallback, even when the baseline policy is not dynamic churn.
            topk_ids = getattr(topk_output, "topk_ids", None)
            if topk_ids is None:
                return False
            controller._routing_snapshot = None
            controller._gpu_group_snapshot = None
            gpu_grouping_enabled = getattr(
                controller, "_gpu_grouping_enabled", None
            )
            use_gpu_grouping = bool(
                callable(gpu_grouping_enabled)
                and gpu_grouping_enabled(topk_ids)
            )
            controller._gpu_grouping_decision = (
                topk_ids,
                use_gpu_grouping,
            )
            if self._current_forward_mode == "decode" and use_gpu_grouping:
                try:
                    groups, _routing_valid = (
                        controller.prepare_gpu_decode_groups(
                            state, topk_ids
                        )
                    )
                    chunked = len(groups) > 1
                    if chunked:
                        controller._gpu_group_snapshot = (
                            topk_ids,
                            int(state.slot_capacity),
                            groups,
                            _routing_valid,
                        )
                    return chunked
                except Exception:
                    # Keep the existing CPU route path as a one-time
                    # fallback for unsupported JIT/toolchain shapes.
                    controller._gpu_grouping_disabled = True
                    controller.token_chunk_gpu_group_fallbacks += 1
                    controller._gpu_grouping_decision = (
                        topk_ids,
                        False,
                    )
            if self._current_forward_mode == "decode":
                # Both policies use the same one-readback decode path. Keep
                # token rows for grouping instead of discarding them in unique().
                routing_rows = topk_ids.tolist()
                logical_ids = {
                    logical
                    for row in routing_rows
                    for logical in row
                    if 0 <= logical < state.full_num_experts
                }
                budget = getattr(controller, "budget", None)
                if budget is not None:
                    budget.observe_rows(
                        self._decode_step, routing_rows, state.full_num_experts
                    )
                chunked = len(logical_ids) > int(state.slot_capacity)
                if chunked:
                    controller._routing_snapshot = (topk_ids, routing_rows)
                return chunked
            logical_ids = self._unique_expert_ids(topk_ids, state.full_num_experts)
            return len(logical_ids) > int(state.slot_capacity)
        if not self._dynamic_expert_churn_policy_enabled():
            return False
        if int(state.slot_capacity) >= int(state.full_num_experts):
            return False
        topk_ids = getattr(topk_output, "topk_ids", None)
        if topk_ids is None or int(state.slot_capacity) <= 0:
            return False
        top_k = int(getattr(state.module, "top_k", 0) or 0)
        if top_k <= 0:
            top_k = int(getattr(state.module.moe_runner_config, "top_k", 1) or 1)
        if int(state.slot_capacity) >= max(1, min(int(state.full_num_experts), top_k)):
            return False
        topk_fastpath_enabled = self._optimized_profile_enabled() and (
            self._expert_plan_applied
            or self._expert_install_state in {"installing_slots", "queued"}
        )
        if topk_fastpath_enabled and state.remap_tensor is not None:
            valid = (topk_ids >= 0) & (topk_ids < state.full_num_experts)
            if not bool(valid.any().item()):
                return False
            safe_ids = topk_ids.clamp(min=0, max=state.full_num_experts - 1).long()
            mapped_ids = state.remap_tensor[safe_ids]
            # If every routed expert is already resident, the number of routed
            # unique experts cannot exceed slot capacity; avoid torch.unique().
            if not bool((valid & (mapped_ids < 0)).any().item()):
                return False
        logical_ids = self._unique_expert_ids(topk_ids, state.full_num_experts)
        return len(logical_ids) > int(state.slot_capacity)

    def _run_expert_core_chunked(
        self,
        state: _LayerKVExpertLayerState,
        dispatch_output: Any,
        *args,
        **kwargs,
    ) -> Any:
        topk_output = getattr(dispatch_output, "topk_output", None)
        topk_ids = getattr(topk_output, "topk_ids", None)
        topk_weights = getattr(topk_output, "topk_weights", None)
        if (
            topk_ids is None
            or topk_weights is None
            or not hasattr(dispatch_output, "_replace")
        ):
            rewritten_dispatch = self._prepare_expert_dispatch_for_core(
                state, dispatch_output
            )
            return state.orig_run_moe_core(rewritten_dispatch, *args, **kwargs)

        logical_ids = self._unique_expert_ids_and_record_hotness(
            state, topk_ids, record_hotness=not self._expert_plan_applied
        )
        if not logical_ids:
            rewritten_dispatch = self._prepare_expert_dispatch_for_core(
                state, dispatch_output
            )
            return state.orig_run_moe_core(rewritten_dispatch, *args, **kwargs)

        capacity = max(1, int(state.slot_capacity))
        valid = (topk_ids >= 0) & (topk_ids < state.full_num_experts)
        safe_ids = topk_ids.clamp(min=0, max=state.full_num_experts - 1).long()
        output_accum = None
        first_output = None
        for begin in range(0, len(logical_ids), capacity):
            chunk = [int(x) for x in logical_ids[begin : begin + capacity]]
            self._materialize_experts(state, chunk, reason="on_demand")
            self._wait_for_expert_logical_ids_ready(state, chunk)
            in_chunk = torch.zeros_like(valid, dtype=torch.bool)
            for expert_id in chunk:
                in_chunk |= safe_ids.eq(int(expert_id))
            active = valid & in_chunk
            mapped_ids = state.remap_tensor[safe_ids]
            chunk_ids = torch.where(
                active,
                mapped_ids.to(topk_ids.dtype),
                torch.zeros_like(topk_ids),
            )
            chunk_weights = torch.where(
                active,
                topk_weights,
                torch.zeros_like(topk_weights),
            )
            chunk_topk = topk_output._replace(
                topk_ids=chunk_ids,
                topk_weights=chunk_weights,
            )
            # Triton may write its output in place. Each chunk must see the
            # original activations, and later calls must not overwrite an
            # earlier partial result retained by output_accum.
            chunk_dispatch = dispatch_output._replace(
                topk_output=chunk_topk,
                hidden_states=dispatch_output.hidden_states.clone(),
            )
            controller = self._shared_expert_controller_for_state(state)
            if controller is not None:
                controller.record_use(state, chunk_dispatch)
            chunk_output = state.orig_run_moe_core(chunk_dispatch, *args, **kwargs)
            hidden = getattr(chunk_output, "hidden_states", None)
            if hidden is None:
                return chunk_output
            if output_accum is None:
                output_accum = hidden
                first_output = chunk_output
            else:
                output_accum = output_accum + hidden

        if self._current_forward_mode == "decode":
            self._record_expert_decode_route(state, logical_ids)
        self.stats.expert_topk_rewrite_count += 1
        self.stats.expert_core_hook_count += 1
        if first_output is not None and hasattr(first_output, "_replace"):
            return first_output._replace(hidden_states=output_accum)
        return first_output

    def _prepare_expert_dispatch_for_core(
        self,
        state: _LayerKVExpertLayerState,
        dispatch_output: Any,
        *,
        known_logical_ids: Optional[List[int]] = None,
        allow_decode_cpu_known: bool = False,
    ) -> Any:
        topk_output = getattr(dispatch_output, "topk_output", None)
        if topk_output is None:
            return dispatch_output
        rewritten_topk = None
        if known_logical_ids is not None:
            rewritten_topk = self._prepare_expert_layer_for_cpu_known_topk(
                state,
                topk_output,
                known_logical_ids,
                allow_decode=allow_decode_cpu_known,
            )
            if rewritten_topk is None:
                self.stats.expert_cpu_known_prepare_fallback_count += 1
        if rewritten_topk is None:
            rewritten_topk = self._prepare_expert_layer_for_topk(state, topk_output)
        self.stats.expert_core_hook_count += 1
        if rewritten_topk is topk_output:
            return dispatch_output
        if hasattr(dispatch_output, "_replace"):
            return dispatch_output._replace(topk_output=rewritten_topk)
        reason = f"layer {state.layer_id} dispatch output does not support topk rewrite"
        self.stats.expert_guard_pass = False
        self.stats.expert_guard_reason = reason
        raise RuntimeError(reason)

    def _prepare_expert_layer_for_cpu_known_topk(
        self,
        state: _LayerKVExpertLayerState,
        topk_output: Any,
        logical_ids: List[int],
        *,
        allow_decode: bool = False,
    ) -> Optional[Any]:
        """Trusted demand from the shared controller's CPU routing snapshot only.

        Retain GPU mapping and its post-load guard. Only demand discovery moves
        to CPU metadata; loading order, hotness and copy readiness stay intact.
        """
        ids = getattr(topk_output, "topk_ids", None)
        controller = self._shared_expert_controller_for_state(state)
        if (
            self.config.shared_expert_prepare_path != "cpu-known"
            or controller is None
            or (
                self._current_forward_mode == "decode"
                and not allow_decode
            )
            or not self._expert_plan_applied
            or not self._optimized_profile_enabled()
            or state.slot_capacity >= state.full_num_experts
            or not state.topk_ids_in_range_calibrated
            or state.topk_ids_invalid_observed
            or ids is None
            or not ids.is_cuda
            or ids.numel() == 0
            or ids.dtype not in (torch.int32, torch.int64)
            or state.remap_tensor is None
            or not logical_ids
            or len(logical_ids) > state.slot_capacity
            or logical_ids != sorted(set(logical_ids))
            or logical_ids[0] < 0
            or logical_ids[-1] >= state.full_num_experts
        ):
            return None
        # CPU maps must agree before using them to skip a materialization.
        # Inconsistent metadata goes through the existing guarded path.
        if any(
            not 0 <= slot < state.slot_capacity
            or state.slot_to_logical.get(slot) != logical
            for logical, slot in state.logical_to_slot.items()
        ) or any(
            state.logical_to_slot.get(logical) != slot
            for slot, logical in state.slot_to_logical.items()
        ):
            return None
        # The CPU maps are checked before every call, while the GPU remap is
        # published on this same producer stream by _materialize_experts.  A
        # successful guard can therefore be reused for a stable remap tensor
        # and slot capacity; this avoids one host scalar readback per chunk.
        # Debug/wait-trace modes retain the old per-call guard for diagnosis.
        remap_signature = (id(state.remap_tensor), int(state.slot_capacity))
        guard_required = (
            getattr(state, "_cpu_known_remap_guard_signature", None)
            != remap_signature
            or self.config.profile_detail
            or self.config.shared_expert_trace_waits
        )
        with self._profile("profile_expert_topk_rewrite_ms"):
            if any(logical not in state.logical_to_slot for logical in logical_ids):
                # Match generic miss behavior: pass ALL sorted IDs, not only
                # misses. Do not touch LRU/hotness on an all-hit group.
                self._materialize_experts(state, logical_ids, reason="on_demand")
            self._wait_for_expert_logical_ids_ready(state, logical_ids)
            if guard_required or self.config.shared_expert_trace_waits:
                if self.config.shared_expert_trace_waits:
                    from .wait_trace import traced_remap_and_guard

                    mapped, invalid = traced_remap_and_guard(state, ids)
                else:
                    mapped = state.remap_tensor[ids.long()]
                    invalid = bool(
                        ((mapped < 0) | (mapped >= state.slot_capacity)).any().item()
                    )
            else:
                mapped = state.remap_tensor[ids.long()]
                invalid = False
            if invalid:
                reason = f"layer {state.layer_id} invalid expert remap after CPU-known prepare"
                self.stats.expert_guard_pass = False
                self.stats.expert_guard_reason = reason
                self.stats.comparable = False
                self.stats.comparability_reason = reason
                raise RuntimeError(reason)
            if guard_required:
                state._cpu_known_remap_guard_signature = remap_signature
                self.stats.expert_cpu_known_remap_guard_count += 1
            else:
                self.stats.expert_cpu_known_remap_guard_elided_count += 1
            self.stats.expert_cpu_known_prepare_count += 1
            self.stats.expert_topk_range_fastpath_count += 1
            self.stats.expert_topk_rewrite_count += 1
            if self._current_forward_mode == "decode":
                # ``logical_ids`` is already the CPU-side routing snapshot
                # consumed by this fastpath.  Retain it for the next decode
                # step's route-aware prefetch without reading activations or
                # expert IDs back from CUDA.
                self._record_expert_decode_route(state, logical_ids)
            return topk_output._replace(topk_ids=mapped.to(ids.dtype))

    def _prepare_expert_layer_for_topk(
        self, state: _LayerKVExpertLayerState, topk_output: Any
    ) -> Any:
        topk_ids = getattr(topk_output, "topk_ids", None)
        if topk_ids is None:
            return topk_output
        if int(state.slot_capacity) >= int(state.full_num_experts):
            if self._should_collect_expert_hotness_layer(int(state.layer_id)):
                self._record_expert_hotness_for_layer(
                    state.layer_id, state.full_num_experts, topk_ids
                )
            else:
                self._expert_hotness_sample_skip_pending += 1
            return topk_output
        with self._profile("profile_expert_topk_rewrite_ms"):
            return self._prepare_expert_layer_for_topk_remap(state, topk_output)

    def _prepare_expert_layer_for_topk_remap(
        self, state: _LayerKVExpertLayerState, topk_output: Any
    ) -> Any:
        topk_ids = getattr(topk_output, "topk_ids", None)
        if topk_ids is None:
            return topk_output
        valid: Optional[torch.Tensor] = None
        fast_in_range = bool(
            state.topk_ids_in_range_calibrated
            and not state.topk_ids_invalid_observed
            and topk_ids.numel() > 0
        )
        if fast_in_range:
            safe_ids = topk_ids.long()
            self.stats.expert_topk_range_fastpath_count += 1
        else:
            valid = (topk_ids >= 0) & (topk_ids < state.full_num_experts)
            safe_ids = topk_ids.clamp(min=0, max=state.full_num_experts - 1).long()
        remap = state.remap_tensor
        if remap is None:
            reason = f"layer {state.layer_id} missing expert remap tensor"
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = reason
            raise RuntimeError(reason)
        topk_fastpath_enabled = self._optimized_profile_enabled() and (
            self._expert_plan_applied
            or self._expert_install_state in {"installing_slots", "queued"}
        )
        if topk_fastpath_enabled and fast_in_range:
            gpu_result = self._try_gpu_expert_topk_remap(state, topk_output, remap)
            if gpu_result is not None:
                rewritten_topk, missing_logical_ids = gpu_result
                if not missing_logical_ids:
                    self.stats.expert_topk_rewrite_count += 1
                    return rewritten_topk
                logical_ids = self._unique_expert_ids_and_record_hotness(
                    state,
                    topk_ids,
                    record_hotness=not self._expert_plan_applied,
                )
                if len(logical_ids) > state.slot_capacity:
                    self._grow_expert_layer_slots(state, len(logical_ids))
                self._materialize_experts(state, logical_ids, reason="on_demand")
                self._wait_for_expert_logical_ids_ready(state, logical_ids)
                mapped_ids = remap[safe_ids]
                missing = mapped_ids < 0
                if bool(missing.any().item()):
                    missing_ids = self._unique_expert_ids(
                        topk_ids, state.full_num_experts
                    )
                    reason = (
                        f"layer {state.layer_id} expert remap still missing after "
                        f"gpu materialize: logical_ids={missing_ids[:8]}"
                    )
                    self.stats.expert_guard_pass = False
                    self.stats.expert_guard_reason = reason
                    self.stats.comparable = False
                    self.stats.comparability_reason = reason
                    raise RuntimeError(reason)
                if self._current_forward_mode == "decode":
                    self._record_expert_decode_route(state, logical_ids)
                self.stats.expert_topk_rewrite_count += 1
                return topk_output._replace(topk_ids=mapped_ids.to(topk_ids.dtype))
        mapped_ids = remap[safe_ids]
        if (
            not fast_in_range
            and valid is not None
            and not state.topk_ids_in_range_calibrated
            and not state.topk_ids_invalid_observed
        ):
            invalid_any = bool((~valid).any().item())
            if invalid_any:
                state.topk_ids_invalid_observed = True
                self.stats.expert_topk_range_invalid_count += 1
            else:
                state.topk_ids_in_range_calibrated = True
                self.stats.expert_topk_range_calibrated_count += 1
        if topk_fastpath_enabled:
            # Common pressure steady state: one or a few cold experts are
            # offloaded, but this token routes only to resident experts. Avoid
            # CPU unique/materialize work and only rewrite logical ids to slots.
            missing = mapped_ids < 0
            missing_any = bool(
                missing.any().item()
                if fast_in_range or valid is None
                else (valid & missing).any().item()
            )
            if not missing_any:
                if fast_in_range or valid is None:
                    rewritten_ids = mapped_ids.to(topk_ids.dtype)
                else:
                    rewritten_ids = torch.where(
                        valid, mapped_ids.to(topk_ids.dtype), topk_ids
                    )
                self.stats.expert_topk_rewrite_count += 1
                return topk_output._replace(topk_ids=rewritten_ids)
        logical_ids = self._unique_expert_ids_and_record_hotness(
            state, topk_ids, record_hotness=not self._expert_plan_applied
        )
        if len(logical_ids) > state.slot_capacity:
            self._grow_expert_layer_slots(state, len(logical_ids))
        self._materialize_experts(state, logical_ids, reason="on_demand")
        self._wait_for_expert_logical_ids_ready(state, logical_ids)
        mapped_ids = remap[safe_ids]
        missing = mapped_ids < 0
        if bool(
            missing.any().item()
            if fast_in_range or valid is None
            else (valid & missing).any().item()
        ):
            missing = self._unique_expert_ids(topk_ids, state.full_num_experts)
            reason = (
                f"layer {state.layer_id} expert remap still missing after "
                f"materialize: logical_ids={missing[:8]}"
            )
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = reason
            self.stats.comparable = False
            self.stats.comparability_reason = reason
            raise RuntimeError(reason)
        if self._current_forward_mode == "decode":
            self._record_expert_decode_route(state, logical_ids)
        if fast_in_range or valid is None:
            rewritten_ids = mapped_ids.to(topk_ids.dtype)
        else:
            rewritten_ids = torch.where(valid, mapped_ids.to(topk_ids.dtype), topk_ids)
        self.stats.expert_topk_rewrite_count += 1
        return topk_output._replace(topk_ids=rewritten_ids)

    def _try_gpu_expert_topk_remap(
        self,
        state: _LayerKVExpertLayerState,
        topk_output: Any,
        remap: torch.Tensor,
    ) -> Optional[Tuple[Any, List[int]]]:
        if (
            self._expert_topk_gpu_remap_disabled
            or layerkv_expert_remap is None
            or remap.device.type != "cuda"
        ):
            self.stats.expert_topk_gpu_remap_fallback_count += 1
            return None
        topk_ids = getattr(topk_output, "topk_ids", None)
        if (
            topk_ids is None
            or topk_ids.device.type != "cuda"
            or topk_ids.dtype not in (torch.int32, torch.int64)
            or topk_ids.numel() == 0
        ):
            self.stats.expert_topk_gpu_remap_fallback_count += 1
            return None
        try:
            rewritten_ids, missing_ids, missing_count = layerkv_expert_remap(
                topk_ids, remap, state.full_num_experts
            )
            missing_n = int(missing_count.item())
            self.stats.expert_topk_gpu_remap_count += 1
            if missing_n <= 0:
                return topk_output._replace(topk_ids=rewritten_ids), []
            self.stats.expert_topk_gpu_remap_missing_count += int(missing_n)
            ids = missing_ids[:missing_n].detach().cpu().tolist()
            missing_logical_ids = list(dict.fromkeys(int(x) for x in ids))
            return topk_output._replace(topk_ids=rewritten_ids), missing_logical_ids
        except Exception:
            self._expert_topk_gpu_remap_disabled = True
            self.stats.expert_topk_gpu_remap_error_count += 1
            self.stats.expert_topk_gpu_remap_fallback_count += 1
            return None

    def _record_expert_hotness(
        self, state: _LayerKVExpertLayerState, topk_ids: torch.Tensor
    ) -> None:
        self._record_expert_hotness_for_layer(
            layer_id=state.layer_id,
            full_num_experts=state.full_num_experts,
            topk_ids=topk_ids,
        )

    def _should_sample_expert_hotness(self, layer_id: int) -> bool:
        if self._current_forward_mode != "decode":
            return True
        step = max(0, int(self._decode_step))
        if step <= 4 and not self._has_prefill_expert_hotness_gpu():
            return True
        if self._expert_hotness_sampled_layers_step == step:
            sampled = self._expert_hotness_sampled_layers
            return sampled is None or int(layer_id) in sampled
        interval = self._effective_expert_hotness_sample_interval()
        return ((step + int(layer_id)) % interval) == 0

    def _effective_expert_hotness_sample_interval(self) -> int:
        interval = max(1, int(self.config.expert_hotness_sample_interval))
        if (
            self.config.dynamic_pressure_from_kvc
            and self._current_forward_mode == "decode"
            and self._expert_plan_applied
            and not self._expert_install_queue
            and float(self.stats.effective_reclaim_target_mb) <= 1e-3
            and float(self.stats.needed_pressure_mb) <= 1e-3
            and int(self._decode_step) > 32
        ):
            return interval * 16
        if self._defer_expert_hotness_snapshot() and int(self._decode_step) > 32:
            return interval * 4
        return interval

    def _prepare_expert_hotness_sampling_for_step(self) -> None:
        if self._current_forward_mode != "decode":
            self._expert_hotness_sampled_layers = None
            self._expert_hotness_sampled_layers_step = int(self._decode_step)
            return
        step = max(0, int(self._decode_step))
        self._expert_hotness_sampled_layers_step = step
        if step <= 4:
            self._expert_hotness_sampled_layers = None
            return
        interval = self._effective_expert_hotness_sample_interval()
        self._expert_hotness_sampled_layers = {
            int(layer_id)
            for layer_id, _module in self._expert_modules
            if ((step + int(layer_id)) % interval) == 0
        }

    def _should_collect_expert_hotness_layer(self, layer_id: int) -> bool:
        if not self._expert_hotness_counter_needed():
            return False
        if self._current_forward_mode != "decode":
            return True
        if self._decode_step <= 0:
            return True
        return self._should_sample_expert_hotness(layer_id)

    def _expert_hotness_counter_needed(self) -> bool:
        if self.config.expert_collector_only:
            return True
        if self._current_forward_mode != "decode":
            return True
        if (
            self._expert_install_queue
            or self._expert_plan_applied
            or self._expert_layers
        ):
            return True
        if not self.config.dynamic_pressure_from_kvc:
            return True
        if float(self.stats.planned_expert_reclaim_mb) > 1e-3:
            return True
        if float(self.stats.policy_expert_fraction) > 1e-6:
            return True
        if (
            str(self.stats.planner_fallback_reason)
            == "dynamic_kvc_pressure_requires_kvc_reclaim"
        ):
            return False
        if self._has_prefill_expert_hotness():
            return False
        return True

    def _expert_hotness_cpu_snapshot_needed(self, mode: str) -> bool:
        if self.config.expert_collector_only:
            return True
        if (
            self._expert_install_queue
            or self._expert_plan_applied
            or self._expert_layers
        ):
            return True
        if not self.config.dynamic_pressure_from_kvc:
            return True
        if float(self.stats.planned_expert_reclaim_mb) > 1e-3:
            return True
        if float(self.stats.policy_expert_fraction) > 1e-6:
            return True
        return False

    def _expert_candidate_cpu_snapshot_needed(self, mode: str) -> bool:
        """Whether CPU needs a decoded candidate order at all.

        Candidate order is only a cold-start/install hint.  Once the expert
        plan is applied, route-aware prefetch can use the live route or the
        already materialized CPU hotness map; copying a full sorted order every
        sample is redundant.  Keep it enabled while an install queue still
        needs candidate ordering, and keep collector-only mode GPU-resident.
        """
        if mode != "decode" or self.config.expert_collector_only:
            return False
        if not self._expert_hotness_cpu_snapshot_needed(mode):
            return False
        if self._expert_plan_applied and not self._expert_install_queue:
            self.stats.expert_candidate_snapshot_deferred_count += 1
            return False
        return True

    def _decode_expert_hotness_collection_needed(self) -> bool:
        if self._current_forward_mode != "decode":
            return True
        if not self._expert_hotness_counter_needed():
            return False
        if (
            self._expert_install_queue
            or self._expert_plan_applied
            or self._expert_layers
        ):
            return True
        if not self.config.dynamic_pressure_from_kvc:
            return True
        if float(self.stats.planned_expert_reclaim_mb) > 1e-3:
            return True
        if float(self.stats.policy_expert_fraction) > 1e-6:
            return True
        if (
            str(self.stats.planner_fallback_reason)
            == "dynamic_kvc_pressure_requires_kvc_reclaim"
        ):
            return False
        if self._has_prefill_expert_hotness():
            return False
        return True

    def _maybe_issue_expert_hotness_snapshots(
        self,
        mode: str,
        layer_id: int,
        full_num_experts: int,
        counts: torch.Tensor,
    ) -> None:
        """Queue only required CPU views and batch same-layer D2H snapshots.

        GPU counters are the authoritative hotness state.  CPU copies are
        bounded, stale planner views.  When both a count view and a cold-start
        candidate order are needed, they have the same ``int32[num_experts]``
        shape and are submitted through the existing CUDA batch copier as one
        transfer batch, producing one copy submission/event instead of two.
        """
        if counts.device.type != "cuda":
            return
        count_needed = self._expert_hotness_cpu_snapshot_needed(mode)
        candidate_needed = self._expert_candidate_cpu_snapshot_needed(mode)
        if not count_needed and not candidate_needed:
            if mode != "decode" or self._defer_expert_hotness_snapshot():
                self.stats.expert_hotness_snapshot_deferred_count += 1
            return
        if count_needed and self._defer_expert_hotness_snapshot():
            self.stats.expert_hotness_snapshot_deferred_count += 1
            count_needed = False

        step = max(0, int(self._decode_step))
        count_key = (str(mode), int(layer_id))
        count_ready = False
        if count_needed:
            if count_key in self._expert_hotness_pending_snapshot_keys:
                count_needed = False
            elif len(self._expert_hotness_pending_snapshots) + len(
                self._expert_hotness_snapshot_submit_queue
            ) >= max(
                8, len(self._expert_modules) * 2
            ):
                self.stats.expert_hotness_snapshot_drop_count += 1
                count_needed = False
            else:
                last = int(self._expert_hotness_last_snapshot_step.get(count_key, -1))
                interval = max(4, int(self.config.expert_hotness_sample_interval) * 4)
                count_ready = not (
                    step > 4 and last >= 0 and step - last < interval
                )

        candidate_ready = False
        if candidate_needed:
            layer_id = int(layer_id)
            if layer_id in self._expert_candidate_pending_layers:
                candidate_needed = False
            elif len(self._expert_candidate_pending_snapshots) + len(
                self._expert_hotness_snapshot_submit_queue
            ) >= max(
                8, len(self._expert_modules) * 2
            ):
                self.stats.expert_candidate_snapshot_drop_count += 1
                candidate_needed = False
            else:
                last = int(
                    self._expert_candidate_last_snapshot_step.get(layer_id, -1)
                )
                interval = max(4, int(self.config.expert_hotness_sample_interval) * 4)
                candidate_ready = not (
                    step > 4 and last >= 0 and step - last < interval
                )

        if not count_ready and not candidate_ready:
            return

        snapshot_counts = None
        cpu_counts = None
        cpu_order = None
        order = None
        if count_ready:
            snapshot_counts = counts.detach().clone()
            try:
                cpu_counts = torch.empty_like(
                    snapshot_counts, device="cpu", pin_memory=True
                )
            except Exception:
                cpu_counts = torch.empty_like(snapshot_counts, device="cpu")

        if candidate_ready:
            expert_ids = torch.arange(
                int(full_num_experts), dtype=torch.long, device=counts.device
            )
            decode_counts = self._expert_hotness_gpu_decode.get(int(layer_id))
            prefill_counts = self._expert_hotness_gpu_prefill.get(int(layer_id))
            if decode_counts is None:
                decode_counts = torch.zeros_like(counts)
            if prefill_counts is None:
                prefill_counts = torch.zeros_like(counts)
            scale = 1_000_000_000
            scores = (
                decode_counts.to(torch.long) * scale
                + prefill_counts.to(torch.long)
            ) * (int(full_num_experts) + 1) + (
                int(full_num_experts) - expert_ids
            )
            order = torch.argsort(scores, descending=True).to(torch.int32)
            try:
                cpu_order = torch.empty_like(order, device="cpu", pin_memory=True)
            except Exception:
                cpu_order = torch.empty_like(order, device="cpu")

        pairs = [
            (destination, source)
            for destination, source in ((cpu_counts, snapshot_counts), (cpu_order, order))
            if destination is not None and source is not None
        ]
        self._expert_hotness_snapshot_submit_queue.append(
            {
                "mode": str(mode),
                "layer_id": int(layer_id),
                "step": int(step),
                "device": counts.device,
                "count_key": count_key,
                "count_ready": bool(count_ready),
                "candidate_ready": bool(candidate_ready),
                "cpu_counts": cpu_counts,
                "snapshot_counts": snapshot_counts,
                "cpu_order": cpu_order,
                "order": order,
            }
        )
        if count_ready:
            self._expert_hotness_pending_snapshot_keys.add(count_key)
            self._expert_hotness_last_snapshot_step[count_key] = step
        if candidate_ready:
            self._expert_candidate_pending_layers.add(int(layer_id))
            self._expert_candidate_last_snapshot_step[int(layer_id)] = step

    def _flush_expert_hotness_snapshot_submit_queue(self) -> None:
        """Submit queued layer snapshots in one batch per CUDA device."""
        queue = getattr(self, "_expert_hotness_snapshot_submit_queue", None)
        if not queue:
            return
        self._expert_hotness_snapshot_submit_queue = []
        by_device = {}
        for entry in queue:
            by_device.setdefault(str(entry["device"]), []).append(entry)

        for entries in by_device.values():
            device = entries[0]["device"]
            stream = self._hotness_snapshot_stream(device)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device=device))
            pairs = []
            for entry in entries:
                if entry["count_ready"]:
                    pairs.append((entry["cpu_counts"], entry["snapshot_counts"]))
                if entry["candidate_ready"]:
                    pairs.append((entry["cpu_order"], entry["order"]))
            transfer = self._get_expert_batch_transfer()
            use_batch_transfer = bool(
                transfer is not None
                and pairs
                and all(destination.is_pinned() for destination, _source in pairs)
            )
            with torch.cuda.stream(stream):
                stream.wait_event(ready)
                if use_batch_transfer:
                    transfer.copy(pairs, stream)
                    self.stats.expert_hotness_snapshot_batch_count += 1
                    self.stats.expert_hotness_snapshot_batch_tensor_count += len(
                        pairs
                    )
                    self.stats.expert_hotness_snapshot_batch_bytes += sum(
                        int(source.nbytes) for _destination, source in pairs
                    )
                else:
                    if transfer is not None:
                        self.stats.expert_hotness_snapshot_transfer_fallback_count += 1
                    for destination, source in pairs:
                        destination.copy_(source, non_blocking=True)
                event = torch.cuda.Event()
                event.record(stream)
            for entry in entries:
                if entry["snapshot_counts"] is not None:
                    entry["snapshot_counts"].record_stream(stream)
                if entry["order"] is not None:
                    entry["order"].record_stream(stream)
                if entry["count_ready"]:
                    self._expert_hotness_pending_snapshots.append(
                        (
                            entry["mode"],
                            int(entry["layer_id"]),
                            entry["cpu_counts"],
                            event,
                            entry["snapshot_counts"],
                        )
                    )
                    self.stats.expert_hotness_snapshot_issue_count += 1
                if entry["candidate_ready"]:
                    self._expert_candidate_pending_snapshots.append(
                        (
                            int(entry["layer_id"]),
                            entry["cpu_order"],
                            event,
                            entry["order"],
                        )
                    )
                    self.stats.expert_candidate_snapshot_issue_count += 1

    def _flush_expert_hotness_sample_skip_count(self) -> None:
        pending = int(self._expert_hotness_sample_skip_pending)
        if pending <= 0:
            return
        self.stats.expert_hotness_sample_skip_count += pending
        self._expert_hotness_sample_skip_pending = 0

    def _expert_hotness_gpu_counts(
        self, mode: str, layer_id: int, full_num_experts: int, device: torch.device
    ) -> torch.Tensor:
        store = (
            self._expert_hotness_decode
            if mode == "decode"
            else self._expert_hotness_prefill
        )
        store.setdefault(int(layer_id), {})
        gpu_store = (
            self._expert_hotness_gpu_decode
            if mode == "decode"
            else self._expert_hotness_gpu_prefill
        )
        buf = gpu_store.get(int(layer_id))
        if (
            buf is None
            or int(buf.numel()) != int(full_num_experts)
            or buf.device != device
        ):
            buf = torch.zeros(
                (int(full_num_experts),), dtype=torch.int32, device=device
            )
            gpu_store[int(layer_id)] = buf
        return buf

    def _expert_hotness_ones(self, device: torch.device, numel: int) -> torch.Tensor:
        key = (str(device), int(numel))
        cached = self._expert_hotness_ones_cache.get(key)
        if (
            cached is not None
            and cached.device == device
            and int(cached.numel()) == int(numel)
        ):
            self.stats.expert_hotness_ones_reuse_count += 1
            return cached
        ones = torch.ones((int(numel),), dtype=torch.int32, device=device)
        self._expert_hotness_ones_cache[key] = ones
        return ones

    def _defer_expert_hotness_snapshot(self) -> bool:
        return (
            self.config.dynamic_pressure_from_kvc
            and self._current_forward_mode == "decode"
            and not self._expert_install_queue
            and float(self.stats.effective_reclaim_target_mb) <= 1e-3
            and float(self.stats.needed_pressure_mb) <= 1e-3
        )

    def _maybe_issue_expert_hotness_snapshot(
        self, mode: str, layer_id: int, counts: torch.Tensor
    ) -> None:
        if counts.device.type != "cuda":
            return
        if not self._expert_hotness_cpu_snapshot_needed(mode):
            self.stats.expert_hotness_snapshot_deferred_count += 1
            return
        if self._defer_expert_hotness_snapshot():
            self.stats.expert_hotness_snapshot_deferred_count += 1
            return
        key = (str(mode), int(layer_id))
        if key in self._expert_hotness_pending_snapshot_keys:
            return
        max_pending = max(8, len(self._expert_modules) * 2)
        if len(self._expert_hotness_pending_snapshots) >= max_pending:
            self.stats.expert_hotness_snapshot_drop_count += 1
            return
        step = max(0, int(self._decode_step))
        last = int(self._expert_hotness_last_snapshot_step.get(key, -1))
        interval = max(4, int(self.config.expert_hotness_sample_interval) * 4)
        if step > 4 and last >= 0 and step - last < interval:
            return
        snapshot_counts = counts.detach().clone()
        try:
            cpu_counts = torch.empty_like(counts, device="cpu", pin_memory=True)
        except Exception:
            cpu_counts = torch.empty_like(counts, device="cpu")
        device_key = str(counts.device)
        stream = self._expert_hotness_snapshot_streams.get(device_key)
        if stream is None:
            stream = torch.cuda.Stream(device=counts.device)
            self._expert_hotness_snapshot_streams[device_key] = stream
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device=counts.device))
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            cpu_counts.copy_(snapshot_counts, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        snapshot_counts.record_stream(stream)
        self._expert_hotness_pending_snapshots.append(
            (str(mode), int(layer_id), cpu_counts, event, snapshot_counts)
        )
        self._expert_hotness_pending_snapshot_keys.add(key)
        self._expert_hotness_last_snapshot_step[key] = step
        self.stats.expert_hotness_snapshot_issue_count += 1

    def _hotness_snapshot_stream(self, device: torch.device) -> Any:
        device_key = str(device)
        stream = self._expert_hotness_snapshot_streams.get(device_key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._expert_hotness_snapshot_streams[device_key] = stream
        return stream

    def _maybe_issue_expert_candidate_snapshot(
        self, mode: str, layer_id: int, full_num_experts: int, device: torch.device
    ) -> None:
        if device.type != "cuda":
            return
        if mode != "decode":
            return
        if not self._expert_hotness_cpu_snapshot_needed(mode):
            return
        layer_id = int(layer_id)
        if layer_id in self._expert_candidate_pending_layers:
            return
        max_pending = max(8, len(self._expert_modules) * 2)
        if len(self._expert_candidate_pending_snapshots) >= max_pending:
            self.stats.expert_candidate_snapshot_drop_count += 1
            return
        step = max(0, int(self._decode_step))
        last = int(self._expert_candidate_last_snapshot_step.get(layer_id, -1))
        interval = max(4, int(self.config.expert_hotness_sample_interval) * 4)
        if step > 4 and last >= 0 and step - last < interval:
            return
        decode_counts = self._expert_hotness_gpu_decode.get(layer_id)
        prefill_counts = self._expert_hotness_gpu_prefill.get(layer_id)
        if decode_counts is None and prefill_counts is None:
            return
        if decode_counts is None:
            decode_counts = torch.zeros(
                (int(full_num_experts),), dtype=torch.int32, device=device
            )
        if prefill_counts is None:
            prefill_counts = torch.zeros(
                (int(full_num_experts),), dtype=torch.int32, device=device
            )
        expert_ids = torch.arange(
            int(full_num_experts), dtype=torch.long, device=device
        )
        # Match CPU ordering: decode hotness desc, then prefill desc, then id asc.
        scale = 1_000_000_000
        scores = (
            decode_counts.to(torch.long) * scale + prefill_counts.to(torch.long)
        ) * (int(full_num_experts) + 1) + (int(full_num_experts) - expert_ids)
        order = torch.argsort(scores, descending=True).to(torch.int32)
        try:
            cpu_order = torch.empty_like(order, device="cpu", pin_memory=True)
        except Exception:
            cpu_order = torch.empty_like(order, device="cpu")
        stream = self._hotness_snapshot_stream(device)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device=device))
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            cpu_order.copy_(order, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        order.record_stream(stream)
        self._expert_candidate_pending_snapshots.append(
            (layer_id, cpu_order, event, order)
        )
        self._expert_candidate_pending_layers.add(layer_id)
        self._expert_candidate_last_snapshot_step[layer_id] = step
        self.stats.expert_candidate_snapshot_issue_count += 1

    def _finalize_expert_candidate_snapshots(self, *, block: bool = False) -> None:
        if not self._expert_candidate_pending_snapshots:
            return
        remaining = []
        for (
            layer_id,
            cpu_order,
            event,
            order,
        ) in self._expert_candidate_pending_snapshots:
            keep_pending = False
            try:
                if block:
                    event.synchronize()
                elif not event.query():
                    remaining.append((layer_id, cpu_order, event, order))
                    keep_pending = True
                    continue
                self._expert_candidate_order_by_layer[int(layer_id)] = [
                    int(x) for x in cpu_order.tolist()
                ]
                self.stats.expert_candidate_snapshot_ready_count += 1
            except Exception:
                self.stats.expert_candidate_snapshot_drop_count += 1
            finally:
                if not keep_pending:
                    self._expert_candidate_pending_layers.discard(int(layer_id))
        self._expert_candidate_pending_snapshots = remaining

    def _finalize_expert_hotness_snapshots(self, *, block: bool = False) -> None:
        self._flush_expert_hotness_snapshot_submit_queue()
        if not self._expert_hotness_pending_snapshots:
            return
        t0 = time.perf_counter() if self.config.profile_detail else 0.0
        remaining = []
        for (
            mode,
            layer_id,
            cpu_counts,
            event,
            snapshot_counts,
        ) in self._expert_hotness_pending_snapshots:
            key = (str(mode), int(layer_id))
            keep_pending = False
            try:
                if block:
                    event.synchronize()
                elif not event.query():
                    remaining.append(
                        (mode, layer_id, cpu_counts, event, snapshot_counts)
                    )
                    keep_pending = True
                    continue
                target = (
                    self._expert_hotness_decode.setdefault(int(layer_id), {})
                    if mode == "decode"
                    else self._expert_hotness_prefill.setdefault(int(layer_id), {})
                )
                target.clear()
                nonzero = torch.nonzero(cpu_counts, as_tuple=False).flatten()
                if int(nonzero.numel()) > 0:
                    expert_ids = nonzero.tolist()
                    counts = cpu_counts[nonzero].tolist()
                    for expert_id, count in zip(expert_ids, counts):
                        target[int(expert_id)] = int(count)
                    self.stats.expert_hotness_observed = True
                    self._expert_hotness_version += 1
                self.stats.expert_hotness_snapshot_count += 1
                self.stats.expert_hotness_snapshot_ready_count += 1
            except Exception:
                self.stats.expert_hotness_sync_fallback_count += 1
            finally:
                if not keep_pending:
                    self._expert_hotness_pending_snapshot_keys.discard(key)
        self._expert_hotness_pending_snapshots = remaining
        if self.config.profile_detail:
            self._add_profile(
                "profile_expert_hotness_snapshot_ms",
                (time.perf_counter() - t0) * 1000.0,
            )

    def _materialize_expert_hotness_gpu_counts(
        self, *, mode: Optional[str] = None
    ) -> None:
        stores: List[Tuple[str, Dict[int, torch.Tensor], Dict[int, Dict[int, int]]]] = (
            []
        )
        if mode in (None, "prefill"):
            stores.append(
                (
                    "prefill",
                    self._expert_hotness_gpu_prefill,
                    self._expert_hotness_prefill,
                )
            )
        if mode in (None, "decode"):
            stores.append(
                ("decode", self._expert_hotness_gpu_decode, self._expert_hotness_decode)
            )
        if not any(gpu_store for _mode, gpu_store, _cpu_store in stores):
            return
        t0 = time.perf_counter() if self.config.profile_detail else 0.0
        for _mode, gpu_store, cpu_store in stores:
            for layer_id, counts in gpu_store.items():
                try:
                    cpu_counts = counts.detach().cpu()
                    target = cpu_store.setdefault(int(layer_id), {})
                    target.clear()
                    nonzero = torch.nonzero(cpu_counts, as_tuple=False).flatten()
                    if int(nonzero.numel()) > 0:
                        expert_ids = nonzero.tolist()
                        values = cpu_counts[nonzero].tolist()
                        for expert_id, count in zip(expert_ids, values):
                            target[int(expert_id)] = int(count)
                        self.stats.expert_hotness_observed = True
                    self.stats.expert_hotness_snapshot_forced_count += 1
                    self.stats.expert_hotness_snapshot_count += 1
                except Exception:
                    self.stats.expert_hotness_sync_fallback_count += 1
        self._expert_hotness_version += 1
        if self.config.profile_detail:
            self._add_profile(
                "profile_expert_hotness_snapshot_ms",
                (time.perf_counter() - t0) * 1000.0,
            )

    def _ensure_expert_hotness_cpu_view(
        self, *, mode: Optional[str] = None, block: bool = False
    ) -> None:
        self._finalize_expert_hotness_snapshots(block=block)
        self._finalize_expert_candidate_snapshots(block=block)
        if block:
            self._materialize_expert_hotness_gpu_counts(mode=mode)

    def _record_expert_hotness_cpu(
        self, layer_id: int, full_num_experts: int, topk_ids: torch.Tensor
    ) -> None:
        ids = topk_ids.detach()
        ids = ids[(ids >= 0) & (ids < full_num_experts)]
        if ids.numel() == 0:
            return
        target = (
            self._expert_hotness_decode.setdefault(layer_id, {})
            if self._current_forward_mode == "decode"
            else self._expert_hotness_prefill.setdefault(layer_id, {})
        )
        values, counts = torch.unique(ids.cpu(), return_counts=True)
        total = 0
        for expert_id, count in zip(values.tolist(), counts.tolist()):
            expert_id = int(expert_id)
            count = int(count)
            target[expert_id] = target.get(expert_id, 0) + count
            total += count
        if total > 0:
            self._expert_hotness_version += 1
        self.stats.expert_call_count_total += total
        if self._current_forward_mode == "decode":
            self.stats.expert_decode_call_count_total += total
        else:
            self.stats.expert_prefill_call_count_total += total
        self.stats.expert_hotness_observed = True

    def _record_expert_hotness_gpu(
        self, layer_id: int, full_num_experts: int, topk_ids: torch.Tensor
    ) -> None:
        ids = topk_ids.detach().reshape(-1)
        if ids.numel() == 0:
            return
        mode = "decode" if self._current_forward_mode == "decode" else "prefill"
        counts = self._expert_hotness_gpu_counts(
            mode, int(layer_id), int(full_num_experts), ids.device
        )
        index = ids.to(dtype=torch.long, non_blocking=True)
        ones = self._expert_hotness_ones(ids.device, int(index.numel()))
        counts.index_add_(0, index, ones)
        total = int(index.numel())
        self.stats.expert_call_count_total += total
        if mode == "decode":
            self.stats.expert_decode_call_count_total += total
        else:
            self.stats.expert_prefill_call_count_total += total
        self.stats.expert_hotness_observed = True
        self.stats.expert_hotness_record_fast_count += 1
        self.stats.expert_hotness_record_count += 1
        self._maybe_issue_expert_hotness_snapshots(
            mode,
            int(layer_id),
            int(full_num_experts),
            counts,
        )

    def _record_expert_hotness_for_layer(
        self, layer_id: int, full_num_experts: int, topk_ids: torch.Tensor
    ) -> None:
        if topk_ids.numel() == 0:
            return
        if (
            self._current_forward_mode == "decode"
            and not self._decode_expert_hotness_collection_needed()
        ):
            self._expert_hotness_sample_skip_pending += 1
            return
        if not self._should_sample_expert_hotness(layer_id):
            self._expert_hotness_sample_skip_pending += 1
            return
        with self._profile("profile_expert_hotness_record_ms"):
            ids = topk_ids.detach().reshape(-1)
            if ids.numel() == 0:
                return
            if ids.device.type == "cuda":
                self._record_expert_hotness_gpu(
                    int(layer_id), int(full_num_experts), ids
                )
            else:
                self.stats.expert_hotness_sync_fallback_count += 1
                self.stats.expert_hotness_record_safe_count += 1
                self.stats.expert_hotness_record_count += 1
                self._record_expert_hotness_cpu(
                    int(layer_id), int(full_num_experts), ids
                )

    def _refresh_expert_ready_before_use_ratio(self) -> None:
        total = self.stats.expert_ready_use_check_count
        if total <= 0:
            self.stats.expert_ready_before_use_ratio = 1.0
        else:
            self.stats.expert_ready_before_use_ratio = (
                self.stats.expert_ready_before_use_count / float(total)
            )

    def _record_expert_wait_stall_if_ready(
        self, pending: _LayerKVPendingExpertCopy
    ) -> bool:
        if (
            pending.wait_elapsed_recorded
            or pending.wait_start_event is None
            or pending.wait_end_event is None
        ):
            return pending.wait_elapsed_recorded
        try:
            if not pending.wait_end_event.query():
                return False
            elapsed = float(
                pending.wait_start_event.elapsed_time(pending.wait_end_event)
            )
            self.stats.expert_ready_miss_stall_ms += elapsed
            self.stats.layerkv_main_stream_wait_ms += elapsed
            self.stats.expert_ready_miss_stall_count += 1
            pending.wait_elapsed_recorded = True
            return True
        except Exception:
            return False

    def _ensure_pending_expert_copy_index(self) -> None:
        """Build postings for pending copies, including externally assigned lists."""
        pending_events = self._pending_expert_copy_events
        token = (id(pending_events), len(pending_events))
        if getattr(self, "_pending_expert_copy_index_token", None) == token:
            return
        self._pending_expert_copy_by_logical.clear()
        self._pending_expert_copy_by_slot.clear()
        self._pending_expert_copy_sequence.clear()
        self._next_pending_expert_copy_sequence = 0
        for pending in pending_events:
            self._index_pending_expert_copy(pending)
        self._pending_expert_copy_index_token = token

    def _index_pending_expert_copy(self, pending: _LayerKVPendingExpertCopy) -> None:
        sequence = self._next_pending_expert_copy_sequence
        self._next_pending_expert_copy_sequence += 1
        self._pending_expert_copy_sequence[id(pending)] = sequence
        layer_id = int(pending.layer_id)
        for logical_id in pending.logical_ids:
            self._pending_expert_copy_by_logical.setdefault(
                (layer_id, int(logical_id)), []
            ).append(pending)
        for slot_id in pending.slot_by_logical.values():
            self._pending_expert_copy_by_slot.setdefault(
                (layer_id, int(slot_id)), []
            ).append(pending)

    def _unindex_pending_expert_copy(
        self, pending: _LayerKVPendingExpertCopy
    ) -> None:
        layer_id = int(pending.layer_id)
        for logical_id in pending.logical_ids:
            key = (layer_id, int(logical_id))
            bucket = self._pending_expert_copy_by_logical.get(key)
            if bucket is not None:
                bucket[:] = [item for item in bucket if item is not pending]
                if not bucket:
                    self._pending_expert_copy_by_logical.pop(key, None)
        for slot_id in pending.slot_by_logical.values():
            key = (layer_id, int(slot_id))
            bucket = self._pending_expert_copy_by_slot.get(key)
            if bucket is not None:
                bucket[:] = [item for item in bucket if item is not pending]
                if not bucket:
                    self._pending_expert_copy_by_slot.pop(key, None)
        self._pending_expert_copy_sequence.pop(id(pending), None)

    def _pending_expert_copy_candidates(self, layer_id: int, logical_ids):
        self._ensure_pending_expert_copy_index()
        candidates = {}
        for logical_id in logical_ids:
            for pending in self._pending_expert_copy_by_logical.get(
                (int(layer_id), int(logical_id)), ()
            ):
                candidates[id(pending)] = pending
        return sorted(
            candidates.values(),
            key=lambda pending: self._pending_expert_copy_sequence.get(
                id(pending), 0
            ),
        )

    def _pending_expert_copy_logical_ids(self, layer_id: int):
        """Return pending logical IDs without scanning all pending batches."""
        self._ensure_pending_expert_copy_index()
        result = set()
        for (pending_layer, logical_id), bucket in (
            self._pending_expert_copy_by_logical.items()
        ):
            if pending_layer != int(layer_id):
                continue
            if any(
                int(logical_id) not in pending.invalidated_logical_ids
                for pending in bucket
            ):
                result.add(int(logical_id))
        return result

    def _wait_for_expert_logical_ids_ready(
        self, state: _LayerKVExpertLayerState, logical_ids: List[int]
    ) -> None:
        if not self._pending_expert_copy_events or state.device.type != "cuda":
            return
        needed = set(int(x) for x in logical_ids)
        if not needed:
            return
        stream = torch.cuda.current_stream(device=state.device)
        for pending in self._pending_expert_copy_candidates(state.layer_id, needed):
            if pending.waited_on_main_stream or pending.layer_id != state.layer_id:
                continue
            if not pending.logical_ids.intersection(needed):
                continue
            self.stats.expert_ready_use_check_count += 1
            self.stats.scheduler_ready_use_check_count += 1
            ready = bool(pending.ready_event.query())
            if ready:
                self.stats.expert_ready_before_use_count += 1
                self.stats.scheduler_ready_before_use_count += 1
            else:
                self.stats.layerkv_deadline_miss_count += 1
                self.stats.scheduler_deadline_miss_count += 1
            t_wait = time.perf_counter()
            if not ready:
                try:
                    pending.wait_start_event = torch.cuda.Event(enable_timing=True)
                    pending.wait_end_event = torch.cuda.Event(enable_timing=True)
                    pending.wait_start_event.record(stream)
                except Exception:
                    pending.wait_start_event = None
                    pending.wait_end_event = None
            stream.wait_event(pending.ready_event)
            if not ready and pending.wait_end_event is not None:
                try:
                    pending.wait_end_event.record(stream)
                except Exception:
                    pending.wait_end_event = None
            self.stats.scheduler_exposed_wait_ms += (
                time.perf_counter() - t_wait
            ) * 1000.0
            self.stats.expert_copy_stream_wait_count += 1
            self.stats.layerkv_copy_event_wait_count += 1
            for logical_id in pending.logical_ids:
                group = self._resident_groups.get(
                    self._residency_key("expert", state.layer_id, int(logical_id))
                )
                if group is not None:
                    group.wait_count += 1
                    group.ready_waited = True
            self.stats.resident_group_wait_count += len(pending.logical_ids)
            pending.waited_on_main_stream = True
            self._record_expert_wait_stall_if_ready(pending)
        self._refresh_expert_ready_before_use_ratio()

    def _finalize_expert_materialize_events(self, *, block: bool = False) -> None:
        if self._expert_batch_transfer is not None:
            self._expert_batch_transfer.collect(block=block)
        if not self._pending_expert_copy_events:
            return
        remaining = []
        for pending in self._pending_expert_copy_events:
            start = pending.start_event
            end = pending.ready_event
            keep_pending = False
            try:
                self._record_expert_wait_stall_if_ready(pending)
                if block:
                    end.synchronize()
                    ready = True
                else:
                    ready = end.query()
                if not ready:
                    remaining.append(pending)
                    keep_pending = True
                if ready:
                    elapsed = float(start.elapsed_time(end))
                    self.stats.expert_materialize_ms += elapsed
                    self.stats.layerkv_copy_stream_busy_ms += elapsed
                    self.stats.expert_h2d_stream_busy_ms += elapsed
                    state = self._expert_layers.get(int(pending.layer_id))
                    if state is not None:
                        for logical_id in pending.logical_ids:
                            if logical_id in pending.invalidated_logical_ids:
                                continue
                            slot_id = state.logical_to_slot.get(int(logical_id))
                            expected_slot = pending.slot_by_logical.get(
                                int(logical_id)
                            )
                            if expected_slot is not None and slot_id != expected_slot:
                                continue
                            if slot_id is not None:
                                group = self._get_or_create_resident_group(
                                    kind="expert",
                                    layer_id=state.layer_id,
                                    logical_id=int(logical_id),
                                    state="resident",
                                    bytes=state.expert_bytes,
                                )
                                self._mark_expert_group_state(
                                    state,
                                    int(logical_id),
                                    group_state="resident",
                                    slot_id=int(slot_id),
                                )
                                group.recover_count += 1
                                self.stats.resident_group_recover_count += 1
            except Exception:
                pass
            finally:
                if not keep_pending:
                    self._unindex_pending_expert_copy(pending)
        self._pending_expert_copy_events = remaining
        self._pending_expert_copy_index_token = (id(remaining), len(remaining))
        self._refresh_resident_group_stats()

    def _unique_expert_ids(
        self, topk_ids: torch.Tensor, full_num_experts: int
    ) -> List[int]:
        if topk_ids.numel() == 0:
            return []
        ids = topk_ids.detach()
        ids = ids[(ids >= 0) & (ids < full_num_experts)]
        if ids.numel() == 0:
            return []
        with self._profile("profile_expert_unique_ms"):
            return [int(x) for x in torch.unique(ids).detach().cpu().tolist()]

    def _unique_expert_ids_and_record_hotness(
        self,
        state: _LayerKVExpertLayerState,
        topk_ids: torch.Tensor,
        *,
        record_hotness: bool,
    ) -> List[int]:
        if topk_ids.numel() == 0:
            return []
        ids = topk_ids.detach()
        ids = ids[(ids >= 0) & (ids < state.full_num_experts)]
        if ids.numel() == 0:
            return []
        if record_hotness:
            self._record_expert_hotness_for_layer(
                state.layer_id, state.full_num_experts, topk_ids
            )
        with self._profile("profile_expert_unique_ms"):
            values = torch.unique(ids).detach().cpu()
        return [int(x) for x in values.tolist()]

    def _invalidate_pending_expert_slot(
        self, state: _LayerKVExpertLayerState, logical_id: int, slot_id: int
    ) -> None:
        """Prevent a completed old copy from publishing into a reused slot."""
        logical_id = int(logical_id)
        slot_id = int(slot_id)
        self._ensure_pending_expert_copy_index()
        for pending in self._pending_expert_copy_by_slot.get(
            (int(state.layer_id), slot_id), ()
        ):
            if pending.slot_by_logical.get(logical_id) == slot_id:
                pending.invalidated_logical_ids.add(logical_id)

    def _flush_expert_remap_updates(self, state, updates) -> None:
        if not updates:
            return
        # One packed transfer instead of a pageable scalar copy/sync per entry.
        # New tensors own their storage: no reusable host staging can be mutated
        # while CUDA is reading it. Only touched IDs are changed, never the table
        # pointer or unrelated entries. Dict assignment preserves last-write wins.
        ids, slots = list(updates), list(updates.values())
        updates.clear()  # Do not retry a failed CUDA submission in finally.
        try:
            packed = torch.tensor(
                [ids, slots], dtype=torch.int64, device=state.remap_tensor.device
            )
            state.remap_tensor.index_copy_(
                0, packed[0], packed[1].to(dtype=state.remap_tensor.dtype)
            )
        except Exception:
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = "batched expert remap publication failed"
            self.stats.comparable = False
            raise
        self.stats.expert_remap_batch_count += 1
        self.stats.expert_remap_batch_entries += len(ids)

    def _evict_experts_batched(
        self,
        state: _LayerKVExpertLayerState,
        logical_ids: List[int],
        *,
        protected_logical_ids: Optional[Set[int]] = None,
        reason: str = "planned",
    ) -> List[int]:
        """Evict a preselected resident set in one backing/remap batch.

        The normal materializer still owns the guarded fallback path.  This
        helper is used only after a forward-local route planner has selected
        victims.  Every ID and slot is revalidated here because exact-group
        prefetch or an asynchronous H2D copy may have changed residency since
        planning.
        """
        if not logical_ids:
            return []
        self._finalize_expert_materialize_events(block=False)
        finalize_d2h = getattr(self, "_finalize_expert_d2h_events", None)
        if callable(finalize_d2h):
            finalize_d2h(block=False)
        pending_slots = self._pending_expert_copy_slots(state)
        protected = {
            int(logical_id) for logical_id in (protected_logical_ids or ())
        }
        unique_ids = list(dict.fromkeys(int(logical_id) for logical_id in logical_ids))
        evicted_pairs = []
        for logical_id in unique_ids:
            if logical_id in protected:
                continue
            slot_id = state.logical_to_slot.get(logical_id)
            if slot_id is None:
                continue
            slot_id = int(slot_id)
            if slot_id in pending_slots:
                continue
            if state.slot_to_logical.get(slot_id) != logical_id:
                continue
            evicted_pairs.append((logical_id, slot_id))
        if not evicted_pairs:
            return []

        remap_updates = {} if self.config.expert_remap_update == "batch" else None
        cache_accounting = self._get_expert_backing_cache_accounting(state)
        evicted_to_copy = []
        free_slots = {
            int(slot_id) for slot_id in getattr(state, "free_slots", ())
        }
        try:
            for logical_id, slot_id in evicted_pairs:
                self._invalidate_pending_expert_slot(state, logical_id, slot_id)
                global_backing = (
                    self._global_expert_backing(state.layer_id, logical_id)
                    if self._expert_global_cpu_backing
                    else None
                )
                if global_backing is not None:
                    state.cpu_params[logical_id] = global_backing
                if logical_id in state.cpu_params:
                    self.stats.expert_eviction_d2h_skip_count += 1
                    self.stats.expert_backing_cache_hit_count += 1
                else:
                    self.stats.expert_eviction_d2h_copy_count += 1
                    self.stats.expert_backing_cache_miss_count += 1
                    evicted_to_copy.append((logical_id, slot_id))
                state.backing_lru[logical_id] = self._decode_step
                state.logical_to_slot.pop(logical_id, None)
                state.slot_to_logical.pop(slot_id, None)
                if cache_accounting is not None:
                    self._update_expert_backing_cache_accounting(state, logical_id)
                state.lru.pop(logical_id, None)
                if state.remap_tensor is not None:
                    if remap_updates is not None:
                        remap_updates[logical_id] = -1
                    else:
                        state.remap_tensor[logical_id] = -1
                if reason == "planned":
                    self._mark_expert_group_state(
                        state,
                        logical_id,
                        group_state="offloaded",
                        cpu_params=state.cpu_params.get(logical_id),
                    )
                if slot_id not in free_slots:
                    heapq.heappush(state.free_slots, slot_id)
                    free_slots.add(slot_id)

            if remap_updates:
                self._flush_expert_remap_updates(state, remap_updates)
            if evicted_to_copy:
                wait_for_host = not (
                    reason != "prefetch"
                    and self.config.expert_demand_d2h_wait == "stream"
                )
                copied = self._copy_slots_to_cpu_batched(
                    state,
                    evicted_to_copy,
                    wait_for_host=wait_for_host,
                )
                state.cpu_params.update(copied)
            if reason != "prefetch":
                self._expert_group_dirty_layers.add(int(state.layer_id))
            return [int(logical_id) for logical_id, _slot_id in evicted_pairs]
        finally:
            if remap_updates:
                self._flush_expert_remap_updates(state, remap_updates)

    def _materialize_experts(
        self,
        state: _LayerKVExpertLayerState,
        logical_ids: List[int],
        *,
        reason: str = "on_demand",
        protected_logical_ids: Optional[Set[int]] = None,
        transfer_stream: Optional[Any] = None,
    ) -> None:
        t_profile = time.perf_counter() if self.config.debug_stats else 0.0
        if not logical_ids:
            return
        remap_updates = {} if self.config.expert_remap_update == "batch" else None
        try:
            # A completed event is collected here when possible.  Any copy
            # still pending owns its destination slot until the DMA finishes;
            # the slot must not be reused merely because its logical mapping
            # has already been replaced in the CPU bookkeeping.
            self._finalize_expert_materialize_events(block=False)
            cache_accounting = self._get_expert_backing_cache_accounting(state)
            unique_logical_ids = list(dict.fromkeys(int(x) for x in logical_ids))
            self.stats.expert_materialize_dedup_count += max(
                0, len(logical_ids) - len(unique_logical_ids)
            )
            if reason == "on_demand" and state.prefetched_logical_ids:
                used_prefetch = state.prefetched_logical_ids.intersection(
                    unique_logical_ids
                )
                if used_prefetch:
                    self.stats.expert_prefetch_useful_count += len(used_prefetch)
                    self.stats.expert_prefetch_ready_before_use_count += len(
                        used_prefetch
                    )
                    state.prefetched_logical_ids.difference_update(used_prefetch)
            protected = set(unique_logical_ids)
            if protected_logical_ids:
                protected.update(int(x) for x in protected_logical_ids)
            # Shared token-chunk execution may install a forward-local access
            # clock.  Keep it separate from ``state.lru``: the latter is also
            # used by the cross-forward planner as a decode-step age.
            active_lru = getattr(self, "_layerkv_active_expert_lru", None)
            materialized: List[Tuple[int, int, Dict[str, torch.Tensor]]] = []
            evicted_to_copy: List[Tuple[int, int]] = []
            for logical_id in unique_logical_ids:
                slot_id = state.logical_to_slot.get(int(logical_id))
                if slot_id is not None:
                    state.lru[int(logical_id)] = self._decode_step
                    heapq.heappush(
                        state.lru_heap,
                        (self._decode_step, int(slot_id), int(logical_id)),
                    )
                    if int(logical_id) in state.cpu_params:
                        state.backing_lru[int(logical_id)] = self._decode_step
                    continue

                try:
                    slot_id = self._choose_expert_slot_for_materialize(
                        state, protected
                    )
                except RuntimeError:
                    if not self._wait_for_pending_expert_slot_reuse(state):
                        raise
                    slot_id = self._choose_expert_slot_for_materialize(
                        state, protected
                    )
                evicted = state.slot_to_logical.get(slot_id)
                if evicted is not None:
                    evicted = int(evicted)
                    self._invalidate_pending_expert_slot(state, evicted, slot_id)
                    global_evicted = (
                        self._global_expert_backing(state.layer_id, evicted)
                        if self._expert_global_cpu_backing
                        else None
                    )
                    if global_evicted is not None:
                        state.cpu_params[evicted] = global_evicted
                    if int(evicted) in state.cpu_params:
                        self.stats.expert_eviction_d2h_skip_count += 1
                        self.stats.expert_backing_cache_hit_count += 1
                    else:
                        self.stats.expert_eviction_d2h_copy_count += 1
                        self.stats.expert_backing_cache_miss_count += 1
                        evicted_to_copy.append((evicted, int(slot_id)))
                    state.backing_lru[evicted] = self._decode_step
                    state.logical_to_slot.pop(evicted, None)
                    if cache_accounting is not None:
                        self._update_expert_backing_cache_accounting(state, evicted)
                    state.lru.pop(evicted, None)
                    if active_lru is not None:
                        active_lru.pop(evicted, None)
                    if state.remap_tensor is not None:
                        if remap_updates is not None:
                            remap_updates[evicted] = -1
                        else:
                            state.remap_tensor[evicted] = -1
                    if reason == "prefetch":
                        self._mark_expert_group_state(
                            state,
                            evicted,
                            group_state="offloaded",
                            cpu_params=state.cpu_params.get(evicted),
                        )
                source_params = self._expert_backing_for_materialize(
                    state, int(logical_id), reason=reason
                )
                if source_params is None:
                    source_params = (
                        self._global_expert_backing(state.layer_id, int(logical_id))
                        if self._expert_global_cpu_backing
                        else None
                    )
                    if source_params is not None:
                        state.cpu_params[int(logical_id)] = source_params
                if source_params is None:
                    err = f"layer {state.layer_id} missing CPU backing for expert {logical_id}"
                    self.stats.expert_guard_pass = False
                    self.stats.expert_guard_reason = err
                    raise RuntimeError(err)
                materialized.append((int(logical_id), int(slot_id), source_params))
                state.logical_to_slot[logical_id] = slot_id
                if cache_accounting is not None:
                    self._update_expert_backing_cache_accounting(state, logical_id)
                state.slot_to_logical[slot_id] = logical_id
                state.lru[logical_id] = self._decode_step
                heapq.heappush(
                    state.lru_heap, (self._decode_step, slot_id, logical_id)
                )
                if state.remap_tensor is not None:
                    if remap_updates is not None:
                        remap_updates[int(logical_id)] = int(slot_id)
                    else:
                        state.remap_tensor[int(logical_id)] = int(slot_id)
                if reason == "prefetch":
                    self._mark_expert_group_state(
                        state,
                        int(logical_id),
                        group_state="materializing",
                        slot_id=int(slot_id),
                        cpu_params=source_params,
                    )
                state.materialize_step += 1
                self.stats.expert_materialize_count += 1
                if reason == "prefetch":
                    self.stats.expert_prefetch_mb_total += state.expert_bytes / float(
                        1024 * 1024
                    )
                    state.prefetched_logical_ids.add(int(logical_id))
                else:
                    self.stats.expert_on_demand_materialize_count += 1
                self.stats.layerkv_expert_materialize_started += 1
                self.stats.layerkv_tasks_built += 1
                self.stats.expert_materialize_mb_total += state.expert_bytes / float(
                    1024 * 1024
                )
                state.backing_lru[int(logical_id)] = self._decode_step
            if materialized:
                if remap_updates:
                    # Publish on the producer stream before its existing H2D
                    # dependency is captured. Consumers still wait for weights
                    # and validate GPU remapping before running the MoE core.
                    self._flush_expert_remap_updates(state, remap_updates)
                if reason != "prefetch":
                    self._expert_group_dirty_layers.add(int(state.layer_id))
                if (
                    reason == "prefetch"
                    and self._optimized_profile_enabled()
                    and self._copy_slots_to_cpu_batched_async(
                        state,
                        evicted_to_copy,
                        h2d_stream=transfer_stream,
                    )
                ):
                    pass
                else:
                    if (
                        reason == "on_demand"
                        and self.config.expert_demand_d2h_wait == "stream"
                    ):
                        copied = self._copy_slots_to_cpu_batched(
                            state, evicted_to_copy, wait_for_host=False
                        )
                    else:
                        copied = self._copy_slots_to_cpu_batched(state, evicted_to_copy)
                    state.cpu_params.update(copied)
                self._copy_materialized_experts_batched(
                    state,
                    materialized,
                    async_copy=self._expert_h2d_async_enabled(state),
                    reason=reason,
                    transfer_stream=transfer_stream,
                )
                self._trim_expert_backing_cache(state)
        finally:
            if remap_updates:
                # Match scalar partial invalidations if CPU backing lookup or
                # slot selection raises; do not leave evicted IDs GPU-visible.
                self._flush_expert_remap_updates(state, remap_updates)
            if self.config.profile_detail:
                self._add_profile(
                    "profile_expert_materialize_control_ms",
                    (time.perf_counter() - t_profile) * 1000.0,
                )

    def _copy_materialized_experts_batched(
        self,
        state: _LayerKVExpertLayerState,
        materialized: List[Tuple[int, int, Dict[str, torch.Tensor]]],
        *,
        async_copy: bool,
        reason: str,
        transfer_stream: Optional[Any] = None,
    ) -> None:
        if not materialized:
            return
        descriptor_reason = (
            f"materialize_{reason}_async"
            if async_copy
            else f"materialize_{reason}_sync"
        )
        for logical_id, slot_id, source_params in materialized:
            self._record_expert_copy_descriptor(
                direction="H2D",
                reason=descriptor_reason,
                layer_id=int(state.layer_id),
                logical_id=int(logical_id),
                src_slot=int(logical_id),
                dst_slot=int(slot_id),
                nbytes=self._expert_backing_bytes(source_params),
                param_count=len(source_params),
            )
        start = None
        end = None
        active_stream = None
        if state.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            active_stream = (
                transfer_stream or self._expert_h2d_copy_stream(state.device)
                if async_copy
                else torch.cuda.current_stream(device=state.device)
            )

        def issue_copy() -> None:
            if active_stream is not None:
                start.record(active_stream)
            transfer = self._get_expert_batch_transfer()
            if transfer is not None:
                pairs = []
                for name in state.param_names:
                    # Fetch the current parameter once per submission, not per
                    # expert. Do not retain views across arena resize/recall.
                    param = getattr(state.module, name).data
                    pairs.extend(
                        (param[slot_id], source_params[name])
                        for _logical_id, slot_id, source_params in materialized
                    )
                transfer.copy(pairs, active_stream)
            else:
                for name in state.param_names:
                    for _logical_id, slot_id, source_params in materialized:
                        dst = getattr(state.module, name).data[slot_id]
                        dst.copy_(source_params[name], non_blocking=True)
            if active_stream is not None:
                end.record(active_stream)

        if (
            active_stream is not None
            and reason == "on_demand"
            and self.config.expert_demand_d2h_wait == "stream"
        ):
            # Capture the producer before switching to the H2D stream. The main
            # stream is ordered after D2H (including the NULL-stream bridge),
            # so overwriting a slot cannot race its backup. No host fence.
            producer = torch.cuda.current_stream(device=state.device)
            if active_stream.cuda_stream != producer.cuda_stream:
                active_stream.wait_stream(producer)
                self.stats.expert_demand_h2d_dependency_count += 1

        with torch.no_grad():
            if active_stream is not None:
                with torch.cuda.stream(active_stream):
                    issue_copy()
            else:
                issue_copy()
        batch_size = len(materialized)
        self._expert_materialize_batch_sizes.append(batch_size)
        self._expert_materialize_layers_touched.add(int(state.layer_id))
        self.stats.expert_materialize_batch_count += 1
        self.stats.expert_materialize_event_count += 1 if end is not None else 0
        if self.stats.expert_materialize_batch_count > 0:
            self.stats.expert_materialize_avg_batch_size = (
                self.stats.expert_materialize_count
                / float(self.stats.expert_materialize_batch_count)
            )
        if end is not None:
            try:
                if end.query():
                    elapsed = float(start.elapsed_time(end))
                    self.stats.expert_materialize_ms += elapsed
                    self.stats.layerkv_copy_stream_busy_ms += elapsed
                    self.stats.expert_h2d_stream_busy_ms += elapsed
                    if async_copy:
                        self.stats.expert_materialize_async_count += batch_size
                        self.stats.expert_copy_stream_launch_count += 1
                        self.stats.expert_h2d_stream_launch_count += 1
                        self.stats.layerkv_copy_event_record_count += 1
                        if reason == "prefetch":
                            self.stats.expert_h2d_prefetch_async_count += batch_size
                        else:
                            self.stats.expert_h2d_on_demand_async_count += batch_size
                        for logical_id, slot_id, source_params in materialized:
                            group = self._get_or_create_resident_group(
                                kind="expert",
                                layer_id=state.layer_id,
                                logical_id=int(logical_id),
                                state="resident",
                                bytes=state.expert_bytes,
                            )
                            self._mark_expert_group_state(
                                state,
                                int(logical_id),
                                group_state="resident",
                                slot_id=int(slot_id),
                                cpu_params=source_params,
                            )
                            group.recover_count += 1
                    self.stats.resident_group_recover_count += batch_size
                elif async_copy:
                    self.stats.expert_materialize_async_count += batch_size
                    self.stats.expert_copy_stream_launch_count += 1
                    self.stats.expert_h2d_stream_launch_count += 1
                    if reason == "prefetch":
                        self.stats.expert_h2d_prefetch_async_count += batch_size
                    else:
                        self.stats.expert_h2d_on_demand_async_count += batch_size
                    self.stats.layerkv_copy_event_record_count += 1
                    for logical_id, slot_id, source_params in materialized:
                        self._mark_expert_group_state(
                            state,
                            int(logical_id),
                            group_state="materializing",
                            slot_id=int(slot_id),
                            cpu_params=source_params,
                            ready_start_event=start,
                            ready_event=end,
                        )
                    pending = _LayerKVPendingExpertCopy(
                        start_event=start,
                        ready_event=end,
                        layer_id=state.layer_id,
                        logical_ids=set(int(x[0]) for x in materialized),
                        slot_by_logical={
                            int(logical_id): int(slot_id)
                            for logical_id, slot_id, _source_params in materialized
                        },
                    )
                    self._ensure_pending_expert_copy_index()
                    self._pending_expert_copy_events.append(pending)
                    self._index_pending_expert_copy(pending)
                    self._pending_expert_copy_index_token = (
                        id(self._pending_expert_copy_events),
                        len(self._pending_expert_copy_events),
                    )
                else:
                    self.stats.expert_materialize_host_sync_count += batch_size
                    self.stats.expert_h2d_sync_count += batch_size
                    self.stats.resident_group_recover_count += batch_size
            except Exception:
                pass
        else:
            self.stats.expert_materialize_host_sync_count += batch_size
            self.stats.expert_h2d_sync_count += batch_size
            self.stats.resident_group_recover_count += batch_size

    def _grow_expert_layer_slots(
        self, state: _LayerKVExpertLayerState, required_capacity: int
    ) -> None:
        new_capacity = min(
            state.full_num_experts, max(required_capacity, state.slot_capacity)
        )
        if new_capacity <= state.slot_capacity:
            return
        if self._shared_expert_controller_for_state(state) is not None:
            raise RuntimeError("shared expert growth must be funded by KV pages")
        old_capacity = state.slot_capacity
        with torch.no_grad():
            for name in state.param_names:
                param = getattr(state.module, name)
                old = param.data
                new_data = torch.empty(
                    (new_capacity,) + tuple(old.shape[1:]),
                    dtype=old.dtype,
                    device=old.device,
                )
                if old_capacity > 0:
                    new_data[:old_capacity].copy_(old[:old_capacity])
                param.data = new_data
        state.slot_capacity = new_capacity
        for slot_id in range(old_capacity, new_capacity):
            heapq.heappush(state.free_slots, slot_id)
        try:
            state.module.num_experts = new_capacity
            state.module.num_local_experts = new_capacity
            state.module.moe_runner_config.num_experts = new_capacity
            state.module.moe_runner_config.num_local_experts = new_capacity
            state.module.dispatcher.num_experts = new_capacity
            state.module.dispatcher.num_local_experts = new_capacity
            state.module.dispatcher.num_local_routed_experts = new_capacity
        except Exception:
            pass
        self.stats.expert_slot_rebind_count += 1
        self._refresh_expert_stats()

    def _choose_expert_slot_for_materialize(
        self, state: _LayerKVExpertLayerState, protected: Set[int]
    ) -> int:
        pending_slots = self._pending_expert_copy_slots(state)
        while state.free_slots:
            slot_id = int(heapq.heappop(state.free_slots))
            if (
                0 <= slot_id < state.slot_capacity
                and slot_id not in state.slot_to_logical
                and slot_id not in pending_slots
            ):
                return slot_id

        # A shared-expert forward may publish all of its route groups on the
        # CPU before execution.  When that snapshot is installed, prefer a
        # resident expert whose next use is farthest away (Belady-style),
        # while retaining the existing pending-DMA and protected-ID guards.
        # The caller clears this transient map after the current group is
        # prepared, so it never becomes a dataset-specific residency cache.
        future_use = getattr(self, "_layerkv_future_expert_use", None)
        active_lru = getattr(self, "_layerkv_active_expert_lru", None)

        def lru_value(logical_id: int) -> int:
            if active_lru is not None and logical_id in active_lru:
                return int(active_lru[logical_id])
            return int(state.lru.get(logical_id, -1))

        if future_use is not None:
            candidates = []
            for slot_id, logical_id in state.slot_to_logical.items():
                slot_id = int(slot_id)
                logical_id = int(logical_id)
                if (
                    slot_id in pending_slots
                    or logical_id in protected
                    or state.logical_to_slot.get(logical_id) != slot_id
                ):
                    continue
                next_use = future_use.get(logical_id)
                candidates.append(
                    (
                        math.inf if next_use is None else int(next_use),
                        -lru_value(logical_id),
                        -logical_id,
                        -slot_id,
                        slot_id,
                )
            )
            if candidates:
                controller = self._shared_expert_controller_for_state(state)
                if controller is not None:
                    controller.future_eviction_slot_count = (
                        getattr(controller, "future_eviction_slot_count", 0) + 1
                    )
                return max(candidates)[-1]

        # During a long prefill every route group shares one decode-step
        # timestamp.  Prefer the forward-local access clock when available so
        # the fallback is a real LRU instead of a slot/ID tie-break.  The
        # normal heap remains the fallback for all other callers.
        if active_lru is not None:
            candidates = []
            for slot_id, logical_id in state.slot_to_logical.items():
                slot_id = int(slot_id)
                logical_id = int(logical_id)
                if (
                    slot_id in pending_slots
                    or logical_id in protected
                    or state.logical_to_slot.get(logical_id) != slot_id
                ):
                    continue
                candidates.append(
                    (lru_value(logical_id), slot_id, logical_id, slot_id)
                )
            if candidates:
                return min(candidates)[-1]

        deferred = []
        while state.lru_heap:
            step, slot_id, logical_id = heapq.heappop(state.lru_heap)
            if slot_id in pending_slots:
                deferred.append((step, slot_id, logical_id))
                continue
            if logical_id in protected:
                continue
            if state.logical_to_slot.get(logical_id) != slot_id:
                continue
            if state.lru.get(logical_id) != step:
                continue
            for item in deferred:
                heapq.heappush(state.lru_heap, item)
            return int(slot_id)
        for item in deferred:
            heapq.heappush(state.lru_heap, item)
        for slot_id, logical_id in state.slot_to_logical.items():
            if slot_id not in pending_slots and logical_id not in protected:
                return int(slot_id)
        raise RuntimeError(
            f"layer {state.layer_id} has no evictable expert slot for materialization"
        )

    def _pending_expert_copy_slots(
        self, state: _LayerKVExpertLayerState
    ) -> Set[int]:
        """Return destination slots still owned by unfinished H2D copies."""
        self._ensure_pending_expert_copy_index()
        return {
            int(slot_id)
            for layer_id, slot_id in self._pending_expert_copy_by_slot
            if layer_id == int(state.layer_id)
        }

    def _wait_for_pending_expert_slot_reuse(
        self, state: _LayerKVExpertLayerState
    ) -> bool:
        """Wait only when every otherwise-evictable slot is DMA-owned."""
        pending = [
            item
            for item in self._pending_expert_copy_events
            if int(getattr(item, "layer_id", -1)) == int(state.layer_id)
            and item.slot_by_logical
        ]
        if not pending:
            return False
        for item in pending:
            item.ready_event.synchronize()
        self._finalize_expert_materialize_events(block=False)
        return True

    def _expert_hotness_score(
        self, state: _LayerKVExpertLayerState, logical_id: int
    ) -> int:
        logical_id = int(logical_id)
        return int(state.hotness_decode.get(logical_id, 0)) * 8 + int(
            state.hotness_prefill.get(logical_id, 0)
        )

    def _coldest_resident_expert_score(
        self, state: _LayerKVExpertLayerState, protected: Set[int]
    ) -> int:
        coldest: Optional[int] = None
        for logical_id in state.logical_to_slot:
            logical_id = int(logical_id)
            if logical_id in protected:
                continue
            score = self._expert_hotness_score(state, logical_id)
            if coldest is None or score < coldest:
                coldest = score
        return 0 if coldest is None else int(coldest)

    def _refresh_expert_stats(self) -> None:
        if not self._expert_layers:
            return
        full_bytes = sum(state.full_bytes for state in self._expert_layers.values())
        reclaim_bytes = sum(
            state.physical_reclaim_bytes for state in self._expert_layers.values()
        )
        self.stats.physical_expert_reclaim_mb = reclaim_bytes / float(1024 * 1024)
        self._refresh_expert_host_backing_stat()
        self.stats.expert_slot_capacity_total = sum(
            state.slot_capacity for state in self._expert_layers.values()
        )
        self.stats.expert_resident_count = sum(
            state.resident_count for state in self._expert_layers.values()
        )
        self.stats.expert_offloaded_count = sum(
            state.offloaded_count for state in self._expert_layers.values()
        )
        self.stats.expert_terminal_slot_count = self.stats.expert_slot_capacity_total
        self.stats.expert_terminal_offloaded_count = self.stats.expert_offloaded_count
        self.stats.expert_terminal_metadata_mapped_count = (
            self.stats.expert_resident_count
        )
        self._refresh_physical_reclaim_peaks()
        self._refresh_resident_group_stats()
        if self.stats.comparability_reason == "INSUFFICIENT_EXPERT_RECLAIM":
            self.stats.comparable = True
            self.stats.comparability_reason = ""
