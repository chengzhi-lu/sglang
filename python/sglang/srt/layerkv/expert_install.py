"""LayerKV LayerKVExpertInstallMixin implementation."""

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


class LayerKVExpertInstallMixin:
    def _apply_expert_plan_once(self, forward_batch: Any) -> None:
        if self._shared_expert is not None:
            self._shared_expert.ensure_installed()
            return
        if self.config.mode != "kvc-expert":
            return
        if self.config.expert_collector_only:
            self.stats.planned_expert_reclaim_mb = 0.0
            self.stats.policy_expert_fraction = 0.0
            return
        if (
            self.config.dynamic_pressure_from_kvc
            and not self._expert_install_queue
            and self._dynamic_runtime_pressure_mb(forward_batch) <= 1e-3
        ):
            self._set_no_pressure_reclaim_stats(
                max(0.0, float(self.config.reclaim_limit_mb))
            )
            return
        if self._no_pressure_fast_path_active(forward_batch):
            self.stats.layerkv_no_pressure_expert_skip_count += 1
            return
        if (
            self.config.dynamic_pressure_from_kvc
            and not self._expert_install_queue
            and not self._expert_plan_applied
            and not self._has_initial_expert_hotness()
        ):
            runtime_pressure_mb = self._dynamic_runtime_pressure_mb(forward_batch)
            if runtime_pressure_mb > 1e-3:
                configured = max(0.0, float(self.config.reclaim_limit_mb))
                needed = (
                    min(configured, runtime_pressure_mb)
                    if configured > 0.0
                    else runtime_pressure_mb
                )
                available_kvc = self._available_kvc_reclaim_mb(forward_batch)
                available_expert = self._available_expert_reclaim_mb()
                self.stats.configured_reclaim_limit_mb = configured
                self.stats.needed_pressure_mb = needed
                self.stats.available_kvc_reclaim_mb = available_kvc
                self.stats.available_expert_reclaim_mb = available_expert
                self.stats.available_total_reclaim_mb = (
                    available_kvc + available_expert
                )
                self.stats.effective_reclaim_target_mb = min(
                    needed, self.stats.available_total_reclaim_mb
                )
                self.stats.requested_total_reclaim_mb = (
                    self.stats.effective_reclaim_target_mb
                )
                self.stats.policy_semantics_reason = "waiting_for_expert_hotness"
                self._expert_install_state = "waiting_for_hotness"
                self._refresh_expert_install_progress()
            return
        if self._expert_install_queue or self._expert_install_state in {
            "installing_slots",
            "queued",
        }:
            replan_needed, observed_target_mb = self._dynamic_expert_replan_needed(
                forward_batch
            )
            if not replan_needed:
                self.stats.requested_total_reclaim_mb = observed_target_mb
                self._refresh_expert_install_progress()
                return
        elif self._expert_plan_applied:
            replan_needed, observed_target_mb = self._dynamic_expert_replan_needed(
                forward_batch
            )
            if not replan_needed:
                self.stats.requested_total_reclaim_mb = observed_target_mb
                if self.config.policy == "coresid":
                    self._sync_coresid_expert_plan_stats(context="stable_high_watermark")
                return
        target_mb = self._effective_expert_reclaim_mb(forward_batch)
        expert_target_mb = target_mb
        if self._policy_name() in ("layer-aware-joint", "layer-aware-joint-dp"):
            expert_target_mb = max(0.0, float(self.stats.planned_expert_reclaim_mb))
        expert_quantum_mb = max(1e-3, self._expert_reclaim_quantum_mb())
        if self._expert_install_queue or self._expert_install_state in {
            "installing_slots",
            "queued",
        }:
            if (
                self._dynamic_expert_churn_policy_enabled()
                and expert_target_mb
                > float(self._expert_install_target_mb) + expert_quantum_mb
            ):
                self._raise_expert_install_target(expert_target_mb, forward_batch)
            return
        if self._expert_plan_applied:
            if (
                self._dynamic_expert_churn_policy_enabled()
                and expert_target_mb
                > self.stats.physical_expert_reclaim_mb + expert_quantum_mb
            ):
                self._increase_expert_reclaim_to_target(expert_target_mb, forward_batch)
            elif self._dynamic_expert_churn_policy_enabled():
                self._pin_coresid_plan_to_current_expert_residency(
                    context="stable_within_expert_quantum"
                )
            return
        if expert_target_mb <= 0:
            if self.config.dynamic_pressure_from_kvc:
                # In trace-driven hard-pressure mode, the KV table may be below
                # capacity during early prefill/decode and exceed it later.  Do
                # not permanently complete the expert plan on a zero-pressure
                # sample; keep it eligible for the first step that actually
                # needs expert reclaim.
                self._expert_install_state = "waiting_for_pressure"
            else:
                self._expert_plan_applied = True
                self._expert_install_state = "complete"
            self._refresh_expert_install_progress()
            self._refresh_expert_stats()
            return
        if not self.physical_expert_supported:
            self.stats.comparable = False
            self.stats.comparability_reason = self._expert_support_reason()
            return
        if not self._expert_modules:
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = "no supported expert modules discovered"
            self.stats.comparable = False
            self.stats.comparability_reason = self.stats.expert_guard_reason
            return

        # Prefill routing is sufficient to seed the first expert residency plan.
        # Decode hotness is an online refinement signal, not a hard dependency.
        if not self._has_initial_expert_hotness():
            self.stats.policy_semantics_reason = "waiting_for_expert_hotness"
            return

        full_bytes = sum(
            self._expert_full_bytes(module) for _, module in self._expert_modules
        )
        if full_bytes <= 0:
            self.stats.expert_guard_pass = False
            self.stats.expert_guard_reason = "failed to inspect expert bytes"
            return

        with self._profile("profile_plan_expert_capacity_ms"):
            slot_capacities = self._coresid_planned_expert_capacities(
                expert_target_mb, forward_batch
            )
        try:
            self.stats.selected_expert_evictions_by_layer = json.dumps(
                {
                    str(layer_id): max(
                        0,
                        int(
                            self._expert_layers[layer_id].full_num_experts
                            if layer_id in self._expert_layers
                            else module.w13_weight.data.shape[0]
                        )
                        - int(
                            slot_capacities.get(
                                layer_id, module.w13_weight.data.shape[0]
                            )
                        ),
                    )
                    for layer_id, module in self._expert_modules
                },
                sort_keys=True,
            )
        except Exception:
            self.stats.selected_expert_evictions_by_layer = ""
        if self.config.policy == "coresid":
            self.stats.selected_expert_evictions_by_layer = (
                self._expert_evictions_json_from_capacities(slot_capacities)
            )
        prepared_cpu_params: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        prepared_initial_resident: Dict[int, List[int]] = {}
        if self._expert_prepare_done and self._prepared_expert_plan is not None:
            prepared_kind = str(self._prepared_expert_plan.get("kind", "full"))
            prepared_cpu_params = self._prepared_expert_plan.get(
                "cpu_params_by_layer", {}
            )
            prepared_initial_resident = self._prepared_expert_plan.get(
                "initial_resident_by_layer", {}
            )
            prepared_capacities = self._prepared_expert_plan.get("slot_capacities", {})
            prepared_target_mb = float(
                self._prepared_expert_plan.get("target_mb", 0.0) or 0.0
            )
            if prepared_kind == "backing_only":
                prepared_initial_resident = {}
                self.stats.expert_prepared_plan_used = True
            elif abs(prepared_target_mb - expert_target_mb) > 1e-3 or any(
                int(prepared_capacities.get(layer_id, -1))
                != int(slot_capacities[layer_id])
                for layer_id, _module in self._expert_modules
            ):
                prepared_cpu_params = {}
                prepared_initial_resident = {}
                self._prepared_expert_plan = None
                self._expert_prepare_done = False
                self.stats.expert_prepare_invalidated_count += 1
                self.stats.expert_prepare_decode_fallback_count += 1
            else:
                self.stats.expert_prepared_plan_used = True
        else:
            self.stats.expert_prepare_decode_fallback_count += 1

        queue: List[_LayerKVExpertInstallItem] = []
        for layer_id, module in self._expert_modules:
            slot_capacity = slot_capacities[layer_id]
            initial_resident = prepared_initial_resident.get(layer_id)
            full_num_experts = int(module.w13_weight.data.shape[0])
            if int(slot_capacity) >= int(full_num_experts):
                continue
            if initial_resident is not None and (
                len(initial_resident) != slot_capacity
                or any(
                    int(x) < 0 or int(x) >= full_num_experts for x in initial_resident
                )
            ):
                initial_resident = None
            if initial_resident is None:
                initial_resident = self._select_initial_resident_experts(
                    layer_id=layer_id,
                    full_num_experts=full_num_experts,
                    slot_capacity=slot_capacity,
                )
            queue.append(
                _LayerKVExpertInstallItem(
                    layer_id=int(layer_id),
                    module=module,
                    slot_capacity=int(slot_capacity),
                    initial_resident=initial_resident,
                    prepared_cpu_params=prepared_cpu_params.get(layer_id),
                )
            )
        self._expert_install_target_mb = float(expert_target_mb)
        self._prepared_expert_plan = None
        if not queue:
            self._expert_plan_applied = True
            self._expert_install_state = "complete"
            self._record_applied_expert_plan(slot_capacities, context="initial_noop")
            self._refresh_expert_install_progress()
            self._refresh_expert_stats()
            return
        self._expert_install_queue = queue
        self._expert_install_state = "queued"
        self._record_applied_expert_plan(slot_capacities, context="initial_queued")
        self._refresh_expert_install_progress()

    def _raise_expert_install_target(
        self, target_mb: float, forward_batch: Any
    ) -> None:
        """Raise an in-flight expert install plan to a new high-watermark target."""

        if target_mb <= float(self._expert_install_target_mb) + 1e-3:
            return
        if not self._has_initial_expert_hotness():
            return
        with self._profile("profile_plan_expert_capacity_ms"):
            slot_capacities = self._coresid_planned_expert_capacities(
                target_mb, forward_batch
            )
        self._record_applied_expert_plan(slot_capacities, context="raise_target")
        changed = 0
        for item in self._expert_install_queue:
            new_capacity = int(
                slot_capacities.get(int(item.layer_id), item.slot_capacity)
            )
            if new_capacity < int(item.slot_capacity):
                full_num_experts = (
                    int(self._expert_layers.get(int(item.layer_id)).full_num_experts)
                    if int(item.layer_id) in self._expert_layers
                    else int(item.module.w13_weight.data.shape[0])
                )
                item.slot_capacity = new_capacity
                item.initial_resident = self._select_initial_resident_experts(
                    layer_id=int(item.layer_id),
                    full_num_experts=full_num_experts,
                    slot_capacity=new_capacity,
                )
                item.prepared_cpu_params = None
                changed += 1
        for layer_id, state in sorted(self._expert_layers.items()):
            new_capacity = int(slot_capacities.get(int(layer_id), state.slot_capacity))
            if new_capacity < int(state.slot_capacity):
                self._shrink_installed_expert_layer_slots(state, new_capacity)
                changed += 1
        self._expert_install_target_mb = float(target_mb)
        if changed:
            self.stats.expert_slot_rebind_count += changed
            with self._profile("profile_refresh_expert_stats_ms"):
                self._refresh_expert_stats()
        self._refresh_expert_install_progress()

    def _increase_expert_reclaim_to_target(
        self, target_mb: float, forward_batch: Any
    ) -> None:
        """Increase expert reclaim for dynamic kv-first without touching KVC.

        Initial expert install is intentionally budgeted and one-shot for fixed
        experiments, but trace-driven pressure is a high-watermark signal: if
        later decode steps need more headroom, kv-first must keep sacrificing
        expert residency instead of silently relying on SGLang admission or
        falling back to KV eviction.
        """

        if not self._expert_layers:
            return
        if not self._has_initial_expert_hotness():
            self.stats.policy_semantics_reason = "waiting_for_expert_hotness"
            return
        current_mb = max(0.0, float(self.stats.physical_expert_reclaim_mb))
        if target_mb <= current_mb + 1e-3:
            return
        with self._profile("profile_plan_expert_capacity_ms"):
            slot_capacities = self._coresid_planned_expert_capacities(
                target_mb, forward_batch
            )
        self._record_applied_expert_plan(slot_capacities, context="increase_target")
        changed = 0
        for layer_id, state in sorted(self._expert_layers.items()):
            new_capacity = int(slot_capacities.get(int(layer_id), state.slot_capacity))
            if new_capacity >= int(state.slot_capacity):
                continue
            self._shrink_installed_expert_layer_slots(state, new_capacity)
            changed += 1
        if changed:
            self.stats.expert_slot_rebind_count += changed
            self._expert_install_target_mb = max(
                float(self._expert_install_target_mb), float(target_mb)
            )
            with self._profile("profile_refresh_expert_stats_ms"):
                self._refresh_expert_stats()
        if self.stats.physical_expert_reclaim_mb + 1e-3 < target_mb:
            self.stats.policy_semantics_reason = (
                "expert_reclaim_deficit_no_kvc_fallback"
            )
        else:
            self.stats.policy_semantics_reason = ""
        self._refresh_expert_install_progress()

    def _shrink_installed_expert_layer_slots(
        self, state: _LayerKVExpertLayerState, new_capacity: int
    ) -> None:
        new_capacity = max(1, min(int(new_capacity), int(state.slot_capacity)))
        if new_capacity >= int(state.slot_capacity):
            return
        # Generic compact reinstallation is a cold path, not a ledger update.
        state.backing_cache_accounted_by_id = None
        state.backing_cache_accounted_bytes = 0
        current_resident = list(state.logical_to_slot.keys())
        candidate_order = self._expert_candidate_order_by_layer.get(
            int(state.layer_id)
        )
        if candidate_order:
            self.stats.expert_candidate_order_hit_count += 1
            rank = {int(expert_id): i for i, expert_id in enumerate(candidate_order)}
            current_resident.sort(
                key=lambda expert_id: (
                    int(rank.get(int(expert_id), len(rank) + int(expert_id))),
                    int(expert_id),
                )
            )
        else:
            self.stats.expert_candidate_order_miss_count += 1
            current_resident.sort(
                key=lambda expert_id: (
                    -int(state.hotness_decode.get(int(expert_id), 0)),
                    -int(state.hotness_prefill.get(int(expert_id), 0)),
                    int(expert_id),
                )
            )
        keep = [int(x) for x in current_resident[:new_capacity]]
        keep_set = set(keep)
        evict_pairs = [
            (int(logical_id), int(slot_id))
            for logical_id, slot_id in list(state.logical_to_slot.items())
            if int(logical_id) not in keep_set
        ]
        copied = self._copy_slots_to_cpu_batched(state, evict_pairs)
        state.cpu_params.update(copied)
        with torch.no_grad():
            for name in state.param_names:
                param = getattr(state.module, name)
                old = param.data
                new_data = torch.empty(
                    (new_capacity,) + tuple(old.shape[1:]),
                    dtype=old.dtype,
                    device=old.device,
                )
                for slot_id, logical_id in enumerate(keep):
                    old_slot = state.logical_to_slot[int(logical_id)]
                    new_data[slot_id].copy_(old[int(old_slot)])
                param.data = new_data
        state.slot_capacity = int(new_capacity)
        state.logical_to_slot = {
            int(expert_id): slot_id for slot_id, expert_id in enumerate(keep)
        }
        state.slot_to_logical = {
            slot_id: int(expert_id) for slot_id, expert_id in enumerate(keep)
        }
        state.lru = {
            int(expert_id): state.lru.get(int(expert_id), self._decode_step)
            for expert_id in keep
        }
        state.free_slots.clear()
        state.lru_heap.clear()
        for expert_id, slot_id in state.logical_to_slot.items():
            heapq.heappush(state.lru_heap, (state.lru[expert_id], slot_id, expert_id))
        if state.remap_tensor is not None:
            state.remap_tensor.fill_(-1)
            if keep:
                ids = torch.tensor(keep, dtype=torch.long, device=state.device)
                slots = torch.arange(len(keep), dtype=torch.long, device=state.device)
                state.remap_tensor[ids] = slots
        try:
            state.module.num_experts = int(new_capacity)
            state.module.num_local_experts = int(new_capacity)
            state.module.moe_runner_config.num_experts = int(new_capacity)
            state.module.moe_runner_config.num_local_experts = int(new_capacity)
            state.module.dispatcher.num_experts = int(new_capacity)
            state.module.dispatcher.num_local_experts = int(new_capacity)
            state.module.dispatcher.num_local_routed_experts = int(new_capacity)
        except Exception:
            pass
        self._sync_expert_groups_for_state(state)
        self._expert_group_dirty_layers.add(int(state.layer_id))

    def _refresh_expert_install_progress(self) -> None:
        pending = len(self._expert_install_queue)
        queued_install_d2h = sum(
            len(job.expert_ids) for job in self._expert_install_d2h_queue
        )
        pending_install_d2h = sum(
            len(pending_copy.copied)
            for pending_copy in self._pending_expert_d2h_events
            if str(pending_copy.reason).startswith("install")
        )
        completed = len(self._expert_layers)
        state = self._expert_install_state
        if queued_install_d2h > 0:
            state = "queued_backing"
        elif pending_install_d2h > 0:
            state = "installing_backing"
        elif self._expert_plan_applied:
            state = "complete"
        elif pending > 0 and not state:
            state = "queued"
        self.stats.expert_install_state = state
        self.stats.expert_install_pending_layers = pending
        self.stats.expert_install_completed_layers = completed
        self.stats.expert_install_layers_per_step = int(
            self._expert_install_layers_per_step
        )
        self.stats.expert_install_budget_mb = float(self._expert_install_budget_mb)
        self.stats.expert_install_target_steps = int(self._expert_install_target_steps)
        self.stats.expert_install_d2h_queue_length = int(queued_install_d2h)
        self.stats.expert_install_d2h_budget_mb = float(
            self.config.expert_copy_budget_mb
        )
        if queued_install_d2h <= 0:
            self.stats.expert_install_d2h_effective_budget_mb = float(
                self.config.expert_copy_budget_mb
            )
        self.stats.expert_install_d2h_max_budget_mb = float(
            self.config.expert_copy_max_budget_mb
        )
        self.stats.expert_install_d2h_chunk_mb = float(
            self.config.expert_copy_chunk_mb
        )
        self.stats.expert_install_d2h_lookahead_layers = int(
            self.config.expert_copy_lookahead_layers
        )
        self.stats.expert_install_reclaim_mb_progress = float(
            self.stats.physical_expert_reclaim_mb
        )
        if pending > 0 or queued_install_d2h > 0 or pending_install_d2h > 0:
            self.stats.comparable = False
            self.stats.comparability_reason = "INSTALL_IN_PROGRESS_NOT_COMPARABLE"
        elif self.stats.comparability_reason == "INSTALL_IN_PROGRESS_NOT_COMPARABLE":
            self.stats.comparable = True
            self.stats.comparability_reason = ""

    def _enqueue_expert_install_d2h_job(
        self,
        *,
        layer_id: int,
        module: Any,
        param_names: List[str],
        expert_ids: List[int],
        target_cpu_params: Dict[int, Dict[str, torch.Tensor]],
        priority: int = 0,
        deadline_step: Optional[int] = None,
        reason: str = "install_backing_async",
    ) -> None:
        expert_ids = [int(expert_id) for expert_id in expert_ids]
        if not expert_ids:
            return
        self._expert_install_d2h_job_seq += 1
        self._expert_install_d2h_queue.append(
            _LayerKVExpertInstallD2HJob(
                seq=int(self._expert_install_d2h_job_seq),
                layer_id=int(layer_id),
                module=module,
                param_names=list(param_names),
                expert_ids=expert_ids,
                target_cpu_params=target_cpu_params,
                priority=int(priority),
                deadline_step=(
                    int(deadline_step)
                    if deadline_step is not None
                    else int(self._decode_step + 1)
                ),
                reason=str(reason),
            )
        )
        self.stats.expert_install_d2h_queued_count += len(expert_ids)
        self._refresh_expert_install_progress()

    def _promote_queued_expert_d2h_for_demand(
        self, state: _LayerKVExpertLayerState, logical_id: int
    ) -> bool:
        logical_id = int(logical_id)
        for job in list(self._expert_install_d2h_queue):
            if int(job.layer_id) != int(state.layer_id):
                continue
            if logical_id not in set(int(x) for x in job.expert_ids):
                continue
            job.expert_ids = [int(x) for x in job.expert_ids if int(x) != logical_id]
            self._expert_install_d2h_job_seq += 1
            self._expert_install_d2h_queue.append(
                _LayerKVExpertInstallD2HJob(
                    seq=int(self._expert_install_d2h_job_seq),
                    layer_id=int(job.layer_id),
                    module=job.module,
                    param_names=list(job.param_names),
                    expert_ids=[logical_id],
                    target_cpu_params=job.target_cpu_params,
                    priority=max(int(job.priority), 10_000),
                    deadline_step=int(self._decode_step),
                    reason="demand_backing_async",
                    mirror_cpu_params=state.cpu_params,
                )
            )
            self.stats.expert_d2h_demand_boost_count += 1
            self._refresh_expert_install_progress()
            return True
        return False

    def _enqueue_urgent_expert_d2h_for_demand(
        self, state: _LayerKVExpertLayerState, logical_id: int
    ) -> bool:
        logical_id = int(logical_id)
        if not state.param_names:
            return False
        try:
            param0 = getattr(state.module, state.param_names[0]).data
        except Exception:
            return False
        if getattr(param0, "device", None) is None or param0.device.type != "cuda":
            return False
        if int(param0.shape[0]) <= logical_id:
            return False
        # After slot shrinking, module rows are physical slots rather than
        # logical expert ids. Only direct D2H from the module while it is still
        # full-sized.
        if int(param0.shape[0]) != int(state.full_num_experts):
            return False
        self._enqueue_expert_install_d2h_job(
            layer_id=int(state.layer_id),
            module=state.module,
            param_names=state.param_names,
            expert_ids=[logical_id],
            target_cpu_params=state.cpu_params,
            priority=10_000,
            deadline_step=int(self._decode_step),
            reason="demand_backing_async",
        )
        self.stats.expert_d2h_demand_urgent_enqueue_count += 1
        return True

    def _queued_expert_install_d2h_bytes(self) -> int:
        total = 0
        for job in self._expert_install_d2h_queue:
            total += len(job.expert_ids) * max(1, int(self._expert_bytes(job.module)))
        return int(total)

    def _pending_expert_install_d2h_count(self) -> int:
        return sum(
            len(pending_copy.copied)
            for pending_copy in self._pending_expert_d2h_events
            if str(pending_copy.reason).startswith("install")
        )

    def _effective_expert_install_d2h_budget_mb(self) -> float:
        budget_mb = max(0.0, float(self.config.expert_copy_budget_mb))
        target_steps = max(0, int(self._expert_install_target_steps))
        if target_steps > 0 and self._expert_install_d2h_queue:
            remaining_steps = max(
                1, target_steps - int(self.stats.forward_decode_count)
            )
            queued_mb = self._queued_expert_install_d2h_bytes() / float(1024 * 1024)
            horizon_budget_mb = queued_mb / float(remaining_steps)
            if horizon_budget_mb > budget_mb + 1e-3:
                budget_mb = horizon_budget_mb
                self.stats.expert_install_d2h_dynamic_budget_count += 1
        max_budget_mb = max(0.0, float(self.config.expert_copy_max_budget_mb))
        if max_budget_mb > 0.0:
            budget_mb = min(budget_mb, max_budget_mb)
        self.stats.expert_install_d2h_effective_budget_mb = float(budget_mb)
        return float(budget_mb)

    def _prepare_expert_install_d2h_lookahead(self, *, prealloc: bool = True) -> None:
        lookahead = max(0, int(self.config.expert_copy_lookahead_layers))
        if lookahead <= 0 or not self._expert_install_queue:
            return
        prepared = 0
        for item in self._expert_install_queue[:lookahead]:
            before = bool(item.backing_queued)
            self._prepare_expert_install_item_backing(item)
            if prealloc:
                self._prealloc_expert_install_item_compact(item)
            if not before and item.backing_queued:
                prepared += 1
        if prepared:
            self.stats.expert_install_d2h_lookahead_queue_count += prepared

    def _prepare_expert_install_item_backing(
        self, item: _LayerKVExpertInstallItem
    ) -> bool:
        if item.prepared_cpu_params is None:
            item.prepared_cpu_params = {}
        if item.backing_queued:
            item.pending_cpu_experts = {
                int(expert_id)
                for expert_id in item.pending_cpu_experts
                if int(expert_id) not in item.prepared_cpu_params
            }
            return not item.pending_cpu_experts

        module = item.module
        layer_id = int(item.layer_id)
        param_names = self._expert_param_names(module)
        full_num_experts = int(module.w13_weight.data.shape[0])
        resident_set = set(int(x) for x in (item.initial_resident or []))
        missing_cpu_experts: List[int] = []
        for expert_id in range(full_num_experts):
            if int(expert_id) in resident_set:
                continue
            if int(expert_id) in item.prepared_cpu_params:
                self.stats.expert_prepared_backing_hit_count += 1
                continue
            global_params = (
                self._global_expert_backing(layer_id, expert_id)
                if self._expert_global_cpu_backing
                else None
            )
            if global_params is not None:
                item.prepared_cpu_params[int(expert_id)] = global_params
                self.stats.expert_prepared_backing_hit_count += 1
                continue
            self.stats.expert_prepared_backing_miss_count += 1
            missing_cpu_experts.append(int(expert_id))
        if missing_cpu_experts:
            device = getattr(module, param_names[0]).data.device if param_names else None
            if (
                not self._optimized_profile_enabled()
                or device is None
                or device.type != "cuda"
            ):
                return True
            item.pending_cpu_experts = set(missing_cpu_experts)
            item.backing_queued = True
            self._enqueue_expert_install_d2h_job(
                layer_id=layer_id,
                module=module,
                param_names=param_names,
                expert_ids=missing_cpu_experts,
                target_cpu_params=item.prepared_cpu_params,
                priority=0,
                deadline_step=self._decode_step + 1,
            )
            return False
        item.pending_cpu_experts.clear()
        item.backing_queued = True
        return True

    def _submit_expert_install_d2h_budgeted(
        self, *, budget_mb: Optional[float] = None
    ) -> None:
        if not self._expert_install_d2h_queue:
            return
        if budget_mb is None:
            budget_mb = self._effective_expert_install_d2h_budget_mb()
        else:
            self.stats.expert_install_d2h_effective_budget_mb = float(budget_mb)
        budget_bytes = int(max(0.0, float(budget_mb)) * 1024 * 1024)
        if budget_bytes <= 0:
            return
        chunk_cap_bytes = int(
            max(1.0, float(self.config.expert_copy_chunk_mb)) * 1024 * 1024
        )
        submitted_bytes = 0
        submitted_experts = 0
        made_progress = False
        self._expert_install_d2h_queue.sort(
            key=lambda job: (
                -int(job.priority),
                int(job.deadline_step),
                int(job.seq),
            )
        )
        while self._expert_install_d2h_queue and submitted_bytes < budget_bytes:
            job = self._expert_install_d2h_queue[0]
            if not job.expert_ids:
                self._expert_install_d2h_queue.pop(0)
                continue
            expert_bytes = max(1, int(self._expert_bytes(job.module)))
            remaining_budget = max(0, budget_bytes - submitted_bytes)
            chunk_budget = max(1, min(chunk_cap_bytes, remaining_budget))
            take = max(1, min(len(job.expert_ids), chunk_budget // expert_bytes))
            expert_ids = [int(x) for x in job.expert_ids[:take]]
            copied_bytes = self._copy_experts_to_cpu_for_install_batched_async(
                job.module,
                job.param_names,
                expert_ids,
                layer_id=int(job.layer_id),
                target_cpu_params=job.target_cpu_params,
                reason=job.reason,
                mirror_cpu_params=job.mirror_cpu_params,
            )
            if copied_bytes < 0:
                copied = self._copy_experts_to_cpu_for_install_batched(
                    job.module,
                    job.param_names,
                    expert_ids,
                    layer_id=int(job.layer_id),
                )
                job.target_cpu_params.update(copied)
                copied_bytes = sum(
                    self._expert_backing_bytes(params) for params in copied.values()
                )
            job.expert_ids = [int(x) for x in job.expert_ids[take:]]
            if not job.expert_ids:
                self._expert_install_d2h_queue.pop(0)
            submitted_bytes += max(1, int(copied_bytes))
            submitted_experts += len(expert_ids)
            made_progress = True
        if made_progress:
            self.stats.expert_install_d2h_submit_step_count += 1
            self._refresh_expert_install_progress()

    def _force_drain_expert_install_d2h_and_slots(self) -> None:
        if not self.config.expert_copy_force_drain:
            return
        if (
            not self._expert_install_queue
            and not self._expert_install_d2h_queue
            and self._pending_expert_install_d2h_count() <= 0
        ):
            return
        start = time.perf_counter()
        self.stats.expert_install_d2h_force_drain_count += 1
        max_iters = max(8, len(self._expert_modules) * 4 + 8)
        for _ in range(max_iters):
            before = (
                len(self._expert_install_queue),
                self._queued_expert_install_d2h_bytes(),
                self._pending_expert_install_d2h_count(),
                len(self._expert_layers),
            )
            queued_mb = self._queued_expert_install_d2h_bytes() / float(1024 * 1024)
            if queued_mb > 0.0:
                self._submit_expert_install_d2h_budgeted(budget_mb=queued_mb)
            if self._pending_expert_install_d2h_count() > 0:
                self._finalize_expert_d2h_events(block=True)
            self._advance_expert_install_budgeted()
            after = (
                len(self._expert_install_queue),
                self._queued_expert_install_d2h_bytes(),
                self._pending_expert_install_d2h_count(),
                len(self._expert_layers),
            )
            if after[0] == 0 and after[1] == 0 and after[2] == 0:
                break
            if after == before:
                break
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.stats.expert_install_d2h_force_drain_ms += elapsed_ms
        self._refresh_expert_install_progress()

    def _advance_expert_install_budgeted(self) -> None:
        if self.config.mode != "kvc-expert" or not self._expert_install_queue:
            self._refresh_expert_install_progress()
            return
        start = time.perf_counter()
        installed = 0
        installed_mb = 0.0
        target_steps = max(0, int(self._expert_install_target_steps))
        max_layers = max(1, int(self._expert_install_layers_per_step))
        remaining_steps = 0
        if target_steps > 0:
            remaining_steps = max(
                1, target_steps - int(self.stats.forward_decode_count)
            )
            needed_layers = int(
                math.ceil(len(self._expert_install_queue) / float(remaining_steps))
            )
            max_layers = max(max_layers, needed_layers)
        max_mb = max(0.0, float(self._expert_install_budget_mb))
        if target_steps > 0 and self._expert_install_queue:
            remaining_reclaim_mb = 0.0
            for item in self._expert_install_queue:
                remaining_reclaim_mb += max(
                    0.0,
                    (
                        int(item.module.w13_weight.data.shape[0])
                        - int(item.slot_capacity)
                    )
                    * float(self._expert_bytes(item.module))
                    / float(1024 * 1024),
                )
            max_mb = max(max_mb, remaining_reclaim_mb / float(max(1, remaining_steps)))
        self.stats.expert_install_remaining_steps = int(remaining_steps)
        self.stats.expert_install_effective_layers_this_step = int(max_layers)
        self.stats.expert_install_effective_budget_mb_this_step = float(max_mb)
        self._prepare_expert_install_d2h_lookahead(prealloc=False)
        self._expert_install_state = "installing_slots"
        install_profile = (
            "profile_apply_prepared_expert_plan_ms"
            if self._expert_install_queue[0].prepared_cpu_params
            else "profile_install_expert_slots_ms"
        )
        with self._profile(install_profile):
            while self._expert_install_queue and installed < max_layers:
                item = self._expert_install_queue[0]
                if not self._prepare_expert_install_item_backing(item):
                    break
                layer_reclaim_mb = max(
                    0.0,
                    (
                        int(item.module.w13_weight.data.shape[0])
                        - int(item.slot_capacity)
                    )
                    * float(self._expert_bytes(item.module))
                    / float(1024 * 1024),
                )
                if (
                    installed > 0
                    and max_mb > 0.0
                    and installed_mb + layer_reclaim_mb > max_mb
                ):
                    break
                if item.prebuilt_install is None:
                    with self._profile("profile_apply_expert_shrink_ms"):
                        item.prebuilt_install = self._build_expert_layer_slot_install(
                            item.module,
                            item.layer_id,
                            item.slot_capacity,
                            initial_resident=item.initial_resident,
                            prepared_cpu_params=item.prepared_cpu_params,
                            async_copy=True,
                            preallocated_compact_params=(
                                item.preallocated_compact_params
                            ),
                            prealloc_ready_event=item.prealloc_ready_event,
                        )
                        item.preallocated_compact_params = None
                        item.prealloc_ready_event = None
                    if (
                        item.prebuilt_install.ready_event is not None
                        and not item.prebuilt_install.ready_event.query()
                    ):
                        self.stats.expert_install_prebuild_pending_count += 1
                        break
                build = item.prebuilt_install
                if build.ready_event is not None and not build.ready_event.query():
                    self._record_expert_install_build_elapsed(build)
                    self.stats.expert_install_prebuild_pending_count += 1
                    break
                self.stats.expert_install_prebuild_ready_count += 1
                self._expert_install_queue.pop(0)
                state = self._commit_expert_layer_slot_install(build)
                self._expert_layers[item.layer_id] = state
                installed += 1
                installed_mb += layer_reclaim_mb
        if installed > 0:
            self.stats.expert_slot_rebind_count += installed
            self.stats.expert_install_step_count += 1
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.stats.expert_install_step_ms += elapsed_ms
            self.stats.expert_install_blocking_ms += elapsed_ms
            with self._profile("profile_refresh_expert_stats_ms"):
                self._refresh_expert_stats()
        if self._expert_install_queue:
            self.stats.expert_install_not_comparable_step_count += 1
            self._refresh_expert_install_progress()
            return
        self._expert_plan_applied = True
        self._expert_install_state = "complete"
        with self._profile("profile_refresh_expert_stats_ms"):
            self._refresh_expert_stats()
        if (
            self.stats.physical_expert_reclaim_mb + 1e-3
            < self._expert_install_target_mb
        ):
            if self._policy_name() == "kv-first":
                self.stats.policy_semantics_reason = (
                    "expert_reclaim_deficit_no_kvc_fallback"
                )
            elif self._policy_name() in ("layer-aware-joint", "layer-aware-joint-dp"):
                self.stats.policy_semantics_reason = (
                    "expert_reclaim_deficit_plan_execution_mismatch"
                )
            else:
                self.stats.planned_expert_reclaim_mb = (
                    self.stats.physical_expert_reclaim_mb
                )
                self.stats.policy_semantics_reason = (
                    "expert_reclaim_deficit_shifted_to_kvc"
                )
        self._refresh_expert_install_progress()

    def _plan_expert_slot_capacities(self, target_mb: float) -> Dict[int, int]:
        target_bytes = max(0, int(target_mb * 1024 * 1024))
        layer_infos: List[Dict[str, int]] = []
        for layer_id, module in self._expert_modules:
            state = self._expert_layers.get(int(layer_id))
            full_num_experts = (
                int(state.full_num_experts)
                if state is not None
                else int(module.w13_weight.data.shape[0])
            )
            top_k = int(getattr(module, "top_k", 0) or 0)
            if top_k <= 0:
                top_k = int(getattr(module.moe_runner_config, "top_k", 1) or 1)
            decode_unique = len(self._expert_hotness_decode.get(int(layer_id), {}))
            if self._dynamic_expert_churn_policy_enabled():
                # kv-first is the pure expert-offload baseline.  It must be
                # allowed to trade expert churn for KV residency. CoResid must
                # have the same execution capability for expert candidates it
                # explicitly selects; otherwise the DP is biased toward KVC by a
                # runtime limitation rather than the cost model.
                min_capacity = max(1, min(full_num_experts, top_k))
            else:
                min_capacity = max(1, min(full_num_experts, max(top_k, decode_unique)))
            layer_infos.append(
                {
                    "layer_id": int(layer_id),
                    "capacity": full_num_experts,
                    "min_capacity": min_capacity,
                    "expert_bytes": (
                        int(state.expert_bytes)
                        if state is not None
                        else self._expert_bytes(module)
                    ),
                }
            )

        reclaimed = 0
        layer_infos.sort(key=lambda x: x["layer_id"])
        while reclaimed < target_bytes:
            eligible = [
                info for info in layer_infos if info["capacity"] > info["min_capacity"]
            ]
            if not eligible:
                break
            max_expert_bytes = max(
                1, max(int(info["expert_bytes"]) for info in eligible)
            )
            remaining_bytes = max(0, target_bytes - reclaimed)
            take_count = min(
                len(eligible),
                max(1, int(math.ceil(remaining_bytes / float(max_expert_bytes)))),
            )
            for info in eligible[:take_count]:
                info["capacity"] -= 1
                reclaimed += info["expert_bytes"]
                if reclaimed >= target_bytes:
                    break
            if take_count <= 0:
                break
        return {info["layer_id"]: info["capacity"] for info in layer_infos}

    def _evenly_spaced_layer_infos(
        self, layer_infos: List[Dict[str, int]], take_count: int
    ) -> List[Dict[str, int]]:
        if take_count <= 0 or not layer_infos:
            return []
        if take_count >= len(layer_infos):
            return list(layer_infos)
        n = len(layer_infos)
        selected: List[Dict[str, int]] = []
        used: Set[int] = set()
        # Place picks at bucket centers so small expert reclaim targets are
        # distributed across model depth instead of concentrated in early layers.
        for i in range(take_count):
            idx = int(math.floor((i + 0.5) * n / float(take_count)))
            idx = min(n - 1, max(0, idx))
            while idx in used and idx + 1 < n:
                idx += 1
            while idx in used and idx - 1 >= 0:
                idx -= 1
            if idx in used:
                continue
            used.add(idx)
            selected.append(layer_infos[idx])
        return selected

    def _expert_param_names(self, module: Any) -> List[str]:
        names = ["w13_weight", "w2_weight"]
        for optional in ("w13_weight_bias", "w2_weight_bias"):
            if hasattr(module, optional):
                names.append(optional)
        return names

    def _expert_full_bytes(self, module: Any) -> int:
        total = 0
        for name in self._expert_param_names(module):
            tensor = getattr(module, name).data
            total += int(tensor.nbytes)
        return total

    def _expert_bytes(self, module: Any) -> int:
        total = 0
        for name in self._expert_param_names(module):
            tensor = getattr(module, name).data
            total += int(tensor[0].nbytes)
        return total

    def _expert_install_copy_stream(self, device: torch.device) -> Any:
        return (
            self._expert_install_stream
            or self._copy_stream
            or torch.cuda.current_stream(device=device)
        )

    def _record_expert_install_build_elapsed(
        self, build: _LayerKVExpertInstallBuild
    ) -> bool:
        if (
            build.elapsed_recorded
            or build.start_event is None
            or build.ready_event is None
        ):
            return build.elapsed_recorded
        try:
            if not build.ready_event.query():
                return False
            elapsed = float(build.start_event.elapsed_time(build.ready_event))
            self.stats.expert_install_weight_copy_ms += elapsed
            build.elapsed_recorded = True
            return True
        except Exception:
            return False

    def _expert_compact_pool_key(
        self, tensor: torch.Tensor
    ) -> Tuple[str, torch.dtype, Tuple[int, ...]]:
        return (str(tensor.device), tensor.dtype, tuple(int(x) for x in tensor.shape))

    def _alloc_expert_compact_tensor(
        self, old: torch.Tensor, slot_capacity: int, *, layer_id: Optional[int] = None
    ) -> torch.Tensor:
        if self._shared_expert is not None:
            return self._shared_expert.allocate_expert(
                old, slot_capacity, layer_id=layer_id
            )
        shape = (int(slot_capacity),) + tuple(old.shape[1:])
        key = (str(old.device), old.dtype, tuple(int(x) for x in shape))
        pool = self._expert_compact_tensor_pool.get(key)
        if pool:
            tensor = pool.pop()
            self._expert_compact_tensor_pool_bytes = max(
                0, self._expert_compact_tensor_pool_bytes - int(tensor.nbytes)
            )
            self.stats.expert_install_compact_pool_reuse_count += 1
            self.stats.expert_install_compact_pool_bytes = int(
                self._expert_compact_tensor_pool_bytes
            )
            return tensor
        tensor = torch.empty(shape, dtype=old.dtype, device=old.device)
        self.stats.expert_install_compact_pool_alloc_count += 1
        return tensor

    def _release_expert_compact_tensor(self, tensor: torch.Tensor) -> None:
        if tensor is None:
            return
        try:
            key = self._expert_compact_pool_key(tensor)
            self._expert_compact_tensor_pool.setdefault(key, []).append(tensor)
            self._expert_compact_tensor_pool_bytes += int(tensor.nbytes)
            self.stats.expert_install_compact_pool_release_count += 1
            self.stats.expert_install_compact_pool_bytes = int(
                self._expert_compact_tensor_pool_bytes
            )
        except Exception:
            self.stats.expert_install_compact_pool_drop_count += 1

    def _prealloc_expert_install_item_compact(
        self, item: _LayerKVExpertInstallItem
    ) -> bool:
        if item.preallocated_compact_params is not None:
            event = item.prealloc_ready_event
            if event is not None and not event.query():
                self.stats.expert_install_prealloc_pending_count += 1
                return False
            self.stats.expert_install_prealloc_ready_count += 1
            return True
        if item.prebuilt_install is not None:
            return True
        t0 = time.perf_counter()
        compact_params: Dict[str, torch.Tensor] = {}
        active_stream = None
        ready_event = None
        try:
            param_names = self._expert_param_names(item.module)
            if not param_names:
                return False
            device = getattr(item.module, param_names[0]).data.device
            if device.type == "cuda":
                active_stream = self._expert_install_copy_stream(device)
                ready_event = torch.cuda.Event(enable_timing=False)
            with torch.no_grad():
                if active_stream is not None:
                    with torch.cuda.stream(active_stream):
                        for name in param_names:
                            old = getattr(item.module, name).data
                            compact_params[name] = self._alloc_expert_compact_tensor(
                                old, item.slot_capacity, layer_id=item.layer_id
                            )
                        ready_event.record(active_stream)
                else:
                    for name in param_names:
                        old = getattr(item.module, name).data
                        compact_params[name] = self._alloc_expert_compact_tensor(
                            old, item.slot_capacity, layer_id=item.layer_id
                        )
            item.preallocated_compact_params = compact_params
            item.prealloc_ready_event = ready_event
            if ready_event is not None and not ready_event.query():
                self.stats.expert_install_prealloc_pending_count += 1
                return False
            self.stats.expert_install_prealloc_ready_count += 1
            return True
        except Exception:
            for tensor in compact_params.values():
                self._release_expert_compact_tensor(tensor)
            item.preallocated_compact_params = None
            item.prealloc_ready_event = None
            self.stats.expert_install_prealloc_fallback_count += 1
            return False
        finally:
            self.stats.expert_install_prealloc_submit_ms += (
                time.perf_counter() - t0
            ) * 1000.0

    def _build_expert_layer_slot_install(
        self,
        module: Any,
        layer_id: int,
        slot_capacity: int,
        initial_resident: Optional[List[int]] = None,
        prepared_cpu_params: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        *,
        async_copy: bool = True,
        preallocated_compact_params: Optional[Dict[str, torch.Tensor]] = None,
        prealloc_ready_event: Optional[Any] = None,
    ) -> _LayerKVExpertInstallBuild:
        param_names = self._expert_param_names(module)
        full_num_experts = int(module.w13_weight.data.shape[0])
        device = module.w13_weight.data.device
        dtype = module.w13_weight.data.dtype
        cpu_params: Dict[int, Dict[str, torch.Tensor]] = {}
        expert_bytes = 0
        t_meta = time.perf_counter()
        with torch.no_grad():
            for name in param_names:
                param = getattr(module, name)
                expert_bytes += int(param.data[0].nbytes)

            if initial_resident is None:
                initial_resident = self._select_initial_resident_experts(
                    layer_id=layer_id,
                    full_num_experts=full_num_experts,
                    slot_capacity=slot_capacity,
                )
            initial_resident = [int(x) for x in initial_resident]
            with self._profile("profile_apply_expert_slot_map_ms"):
                logical_to_slot = {
                    expert_id: slot_id
                    for slot_id, expert_id in enumerate(initial_resident)
                }
                slot_to_logical = {
                    slot_id: expert_id
                    for slot_id, expert_id in enumerate(initial_resident)
                }
                lru = {expert_id: self._decode_step for expert_id in initial_resident}
                resident_set = set(initial_resident)
            if prepared_cpu_params is not None:
                cpu_params.update(prepared_cpu_params)
                removed_bytes = 0
                for expert_id in list(cpu_params):
                    if expert_id in resident_set:
                        removed = cpu_params.pop(expert_id)
                        removed_bytes += sum(int(t.nbytes) for t in removed.values())
                        self._release_expert_backing_params(removed)
                if removed_bytes:
                    self._expert_host_backing_bytes -= removed_bytes
                    self._refresh_expert_host_backing_stat()
            missing_cpu_experts: List[int] = []
            for expert_id in range(full_num_experts):
                if expert_id in resident_set:
                    continue
                if expert_id in cpu_params:
                    self.stats.expert_prepared_backing_hit_count += 1
                    continue
                global_params = (
                    self._global_expert_backing(layer_id, expert_id)
                    if self._expert_global_cpu_backing
                    else None
                )
                if global_params is not None:
                    cpu_params[int(expert_id)] = global_params
                    self.stats.expert_prepared_backing_hit_count += 1
                    continue
                self.stats.expert_prepared_backing_miss_count += 1
                missing_cpu_experts.append(expert_id)
            if missing_cpu_experts:
                attempted_async = async_copy and device.type == "cuda"
                async_submitted = False
                if attempted_async:
                    copied_bytes = self._copy_experts_to_cpu_for_install_batched_async(
                        module,
                        param_names,
                        missing_cpu_experts,
                        layer_id=layer_id,
                        target_cpu_params=cpu_params,
                        reason="install_backing_async",
                    )
                    async_submitted = copied_bytes >= 0
                if not async_submitted:
                    # The async helper records a failed submission itself;
                    # count a synchronous path here only when no async path
                    # was attempted at all.
                    if not attempted_async:
                        self.stats.expert_install_d2h_sync_fallback_count += len(
                            missing_cpu_experts
                        )
                    cpu_params.update(
                        self._copy_experts_to_cpu_for_install_batched(
                            module,
                            param_names,
                            missing_cpu_experts,
                            layer_id=layer_id,
                        )
                    )
        self.stats.expert_install_metadata_ms += (
            time.perf_counter() - t_meta
        ) * 1000.0

        compact_params: Dict[str, torch.Tensor] = {}
        source_refs: List[torch.Tensor] = []
        start_event = None
        ready_event = None
        active_stream = None
        if async_copy and device.type == "cuda":
            active_stream = self._expert_install_copy_stream(device)
            start_event = torch.cuda.Event(enable_timing=True)
            ready_event = torch.cuda.Event(enable_timing=True)
        if prealloc_ready_event is not None and not prealloc_ready_event.query():
            self.stats.expert_install_prealloc_wait_count += 1
            prealloc_ready_event.synchronize()

        t_submit = time.perf_counter()
        with torch.no_grad():
            if active_stream is not None:
                with torch.cuda.stream(active_stream):
                    start_event.record(active_stream)
                    for name in param_names:
                        param = getattr(module, name)
                        old = param.data
                        new_data = None
                        if preallocated_compact_params is not None:
                            new_data = preallocated_compact_params.pop(name, None)
                            if new_data is not None:
                                self.stats.expert_install_prealloc_reuse_count += 1
                        if new_data is None:
                            t_alloc = time.perf_counter()
                            new_data = self._alloc_expert_compact_tensor(
                                old, slot_capacity, layer_id=layer_id
                            )
                            self.stats.expert_install_weight_alloc_ms += (
                                time.perf_counter() - t_alloc
                            ) * 1000.0
                        for slot_id, expert_id in enumerate(initial_resident):
                            src = old[int(expert_id)].detach()
                            new_data[int(slot_id)].copy_(src, non_blocking=True)
                            source_refs.append(src)
                        compact_params[name] = new_data
                    ready_event.record(active_stream)
            else:
                for name in param_names:
                    param = getattr(module, name)
                    old = param.data
                    new_data = None
                    if preallocated_compact_params is not None:
                        new_data = preallocated_compact_params.pop(name, None)
                        if new_data is not None:
                            self.stats.expert_install_prealloc_reuse_count += 1
                    if new_data is None:
                        t_alloc = time.perf_counter()
                        new_data = self._alloc_expert_compact_tensor(
                            old, slot_capacity, layer_id=layer_id
                        )
                        self.stats.expert_install_weight_alloc_ms += (
                            time.perf_counter() - t_alloc
                        ) * 1000.0
                    for slot_id, expert_id in enumerate(initial_resident):
                        new_data[int(slot_id)].copy_(old[int(expert_id)])
                    compact_params[name] = new_data
        if preallocated_compact_params:
            for tensor in preallocated_compact_params.values():
                self._release_expert_compact_tensor(tensor)
            preallocated_compact_params.clear()
        self.stats.expert_install_prebuild_submit_ms += (
            time.perf_counter() - t_submit
        ) * 1000.0
        self.stats.expert_install_shrink_layer_count += 1
        self.stats.expert_install_shrink_expert_count += len(initial_resident)
        return _LayerKVExpertInstallBuild(
            layer_id=int(layer_id),
            module=module,
            slot_capacity=int(slot_capacity),
            initial_resident=initial_resident,
            full_num_experts=int(full_num_experts),
            device=device,
            dtype=dtype,
            cpu_params=cpu_params,
            param_names=param_names,
            logical_to_slot=logical_to_slot,
            slot_to_logical=slot_to_logical,
            lru=lru,
            expert_bytes=int(expert_bytes),
            compact_params=compact_params,
            source_refs=tuple(source_refs),
            start_event=start_event,
            ready_event=ready_event,
        )

    def _commit_expert_layer_slot_install(
        self, build: _LayerKVExpertInstallBuild
    ) -> _LayerKVExpertLayerState:
        module = build.module
        if getattr(module, "_layerkv_expert_wrapped", False):
            return self._expert_layers[int(build.layer_id)]
        if build.ready_event is not None and not build.ready_event.query():
            self.stats.expert_install_prebuild_wait_count += 1
            build.ready_event.synchronize()
        self._record_expert_install_build_elapsed(build)
        t_swap = time.perf_counter()
        for name, new_data in build.compact_params.items():
            getattr(module, name).data = new_data
        self.stats.expert_install_param_swap_ms += (
            time.perf_counter() - t_swap
        ) * 1000.0

        t_hook = time.perf_counter()
        state = _LayerKVExpertLayerState(
            layer_id=build.layer_id,
            module=module,
            orig_forward=module.forward,
            orig_run_moe_core=getattr(
                module, "_layerkv_hotness_orig_run_moe_core", module.run_moe_core
            ),
            full_num_experts=build.full_num_experts,
            slot_capacity=build.slot_capacity,
            expert_bytes=build.expert_bytes,
            device=build.device,
            dtype=build.dtype,
            cpu_params=build.cpu_params,
            param_names=build.param_names,
            logical_to_slot=build.logical_to_slot,
            slot_to_logical=build.slot_to_logical,
            lru=build.lru,
            hotness_prefill=self._expert_hotness_prefill.setdefault(
                int(build.layer_id), {}
            ),
            hotness_decode=self._expert_hotness_decode.setdefault(
                int(build.layer_id), {}
            ),
        )
        for expert_id, slot_id in state.logical_to_slot.items():
            heapq.heappush(state.lru_heap, (state.lru[expert_id], slot_id, expert_id))
        remap_tensor = torch.full(
            (build.full_num_experts,),
            -1,
            dtype=torch.long,
            device=build.device,
        )
        if build.initial_resident:
            ids = torch.tensor(build.initial_resident, dtype=torch.long, device=build.device)
            slots = torch.arange(
                len(build.initial_resident), dtype=torch.long, device=build.device
            )
            remap_tensor[ids] = slots
        state.remap_tensor = remap_tensor

        @functools.wraps(module.run_moe_core)
        def wrapped_run_moe_core(dispatch_output: Any, *args, **kwargs):
            topk_output = getattr(dispatch_output, "topk_output", None)
            if self._expert_chunked_core_required(state, topk_output):
                controller = self._shared_expert_controller_for_state(state)
                if controller is not None:
                    return controller.run_token_chunks(
                        state, dispatch_output, *args, **kwargs
                    )
                return self._run_expert_core_chunked(
                    state, dispatch_output, *args, **kwargs
                )
            rewritten_dispatch = self._prepare_expert_dispatch_for_core(
                state, dispatch_output
            )
            controller = self._shared_expert_controller_for_state(state)
            if controller is not None:
                controller.record_use(state, rewritten_dispatch)
            return state.orig_run_moe_core(rewritten_dispatch, *args, **kwargs)

        module.run_moe_core = wrapped_run_moe_core
        module._layerkv_expert_wrapped = True
        module._layerkv_expert_state = state
        try:
            module.num_experts = build.slot_capacity
            module.num_local_experts = build.slot_capacity
            module.moe_runner_config.num_experts = build.slot_capacity
            module.moe_runner_config.num_local_experts = build.slot_capacity
            module.dispatcher.num_experts = build.slot_capacity
            module.dispatcher.num_local_experts = build.slot_capacity
            module.dispatcher.num_local_routed_experts = build.slot_capacity
        except Exception:
            pass
        self.stats.expert_install_hook_ms += (time.perf_counter() - t_hook) * 1000.0
        self._sync_expert_groups_for_state(state)
        return state

    def _install_expert_layer_slots(
        self,
        module: Any,
        layer_id: int,
        slot_capacity: int,
        initial_resident: Optional[List[int]] = None,
        prepared_cpu_params: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        *,
        async_copy: bool = False,
    ) -> _LayerKVExpertLayerState:
        if getattr(module, "_layerkv_expert_wrapped", False):
            return self._expert_layers[layer_id]
        build = self._build_expert_layer_slot_install(
            module,
            layer_id,
            slot_capacity,
            initial_resident=initial_resident,
            prepared_cpu_params=prepared_cpu_params,
            async_copy=async_copy,
        )
        return self._commit_expert_layer_slot_install(build)

    def _select_initial_resident_experts(
        self, *, layer_id: int, full_num_experts: int, slot_capacity: int
    ) -> List[int]:
        if slot_capacity <= 0:
            return []
        candidate_order = self._expert_candidate_order_by_layer.get(int(layer_id))
        if candidate_order:
            self.stats.expert_candidate_order_hit_count += 1
            selected = [
                int(expert_id)
                for expert_id in candidate_order
                if 0 <= int(expert_id) < int(full_num_experts)
            ]
            if len(selected) < int(full_num_experts):
                selected_set = set(selected)
                selected.extend(
                    int(expert_id)
                    for expert_id in range(int(full_num_experts))
                    if int(expert_id) not in selected_set
                )
            return selected[: min(slot_capacity, full_num_experts)]
        self.stats.expert_candidate_order_miss_count += 1
        decode_hotness = self._expert_hotness_decode.get(layer_id, {})
        prefill_hotness = self._expert_hotness_prefill.get(layer_id, {})
        experts = list(range(full_num_experts))
        experts.sort(
            key=lambda expert_id: (
                -int(decode_hotness.get(expert_id, 0)),
                -int(prefill_hotness.get(expert_id, 0)),
                expert_id,
            )
        )
        return experts[: min(slot_capacity, full_num_experts)]

    def _copy_expert_to_cpu(
        self, module: Any, param_names: List[str], expert_id: int
    ) -> Dict[str, torch.Tensor]:
        with self._profile("profile_copy_expert_to_cpu_ms"):
            backing = {
                name: self._cpu_backing_tensor(
                    getattr(module, name).data[expert_id].detach()
                )
                for name in param_names
            }
        self._expert_host_backing_bytes += sum(int(t.nbytes) for t in backing.values())
        self._refresh_expert_host_backing_stat()
        return backing

    def _copy_expert_to_cpu_for_install(
        self, module: Any, param_names: List[str], expert_id: int
    ) -> Dict[str, torch.Tensor]:
        backing = self._copy_expert_to_cpu_backing_no_account(
            module, param_names, expert_id, non_blocking=True
        )
        self._expert_host_backing_bytes += sum(int(t.nbytes) for t in backing.values())
        self._refresh_expert_host_backing_stat()
        return backing
