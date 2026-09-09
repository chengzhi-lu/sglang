"""LayerKV planning and setup mixins."""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

if __package__:
    from .config_stats import LayerKVConfig
    from .common_types import (
        _LayerKVExpertLayerState,
        _LayerKVResidencyEntry,
        _LayerKVResidencyHandle,
        _LayerKVResidencyKey,
        _LayerKVResidentTensorGroup,
    )
    from .host_store import _LayerKVHostKVStore
    from .kv_pool_view import HybridKVCStorageView
else:  # pragma: no cover - direct file-loading smoke tests.
    from config_stats import LayerKVConfig
    from common_types import (
        _LayerKVExpertLayerState,
        _LayerKVResidencyEntry,
        _LayerKVResidencyHandle,
        _LayerKVResidencyKey,
        _LayerKVResidentTensorGroup,
    )
    from host_store import _LayerKVHostKVStore
    from kv_pool_view import HybridKVCStorageView

logger = logging.getLogger(__name__)


class LayerKVPlannerMixin:
    def _joint_policy_cache_bucket(self, target_mb: float) -> int:
        # Trace-driven pressure jitters by tens of MB across adjacent decode
        # steps. Replanning on every small jitter dominated runtime; bucket by a
        # coarse physical-pressure class while preserving large workload shifts.
        return int(round(max(0.0, float(target_mb)) / 256.0))

    def _joint_policy_decode_bucket(self) -> int:
        # Expert/KVC repeated-recovery cost changes slowly with decode horizon.
        # Keep a bounded replan cadence for changing hotness without paying DP
        # on every decode step.
        return int(max(0, self._decode_step) // 64)

    def _expert_reclaim_quantum_mb(self) -> float:
        bytes_by_layer: List[int] = []
        if self._expert_layers:
            bytes_by_layer.extend(
                int(state.expert_bytes)
                for state in self._expert_layers.values()
                if int(state.expert_bytes) > 0
            )
        else:
            for _layer_id, module in self._expert_modules:
                try:
                    expert_bytes = int(self._expert_bytes(module))
                except Exception:
                    expert_bytes = 0
                if expert_bytes > 0:
                    bytes_by_layer.append(expert_bytes)
        if not bytes_by_layer:
            return 0.0
        return min(bytes_by_layer) / float(1024 * 1024)

    def _dynamic_expert_replan_needed(self, forward_batch: Any) -> Tuple[bool, float]:
        if not self._dynamic_expert_churn_policy_enabled():
            return True, self._refresh_reclaim_target_stats(forward_batch)
        high_watermark = max(
            float(self._planner_target_high_watermark_mb),
            float(self._planned_expert_target_mb),
            float(self._expert_install_target_mb),
        )
        target_mb = self._refresh_reclaim_target_stats(forward_batch)
        if target_mb <= 1e-3:
            return False, target_mb
        quantum_mb = max(1e-3, self._expert_reclaim_quantum_mb())
        if target_mb <= high_watermark + quantum_mb:
            return False, target_mb
        return True, target_mb

    @classmethod
    def maybe_create(cls, server_args: Any) -> Optional["LayerKVRuntime"]:
        disaggregation_mode = str(getattr(server_args, "disaggregation_mode", "null"))
        if disaggregation_mode == "prefill":
            if bool(getattr(server_args, "enable_layerkv", False)):
                logger.warning(
                    "LayerKV is disabled on PD prefill workers; enable it on the "
                    "PD decode worker or standalone worker instead."
                )
            return None
        config = LayerKVConfig.from_server_args(server_args)
        if not config.enabled or config.mode == "off":
            return None
        return cls(config)

    @contextlib.contextmanager
    def _profile(self, field: str):
        if not self.config.profile_detail:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._add_profile(field, (time.perf_counter() - t0) * 1000.0)

    def _add_profile(self, field: str, elapsed_ms: float) -> None:
        if not self.config.profile_detail:
            return
        try:
            setattr(self.stats, field, float(getattr(self.stats, field)) + elapsed_ms)
        except Exception:
            pass

    def _refresh_expert_host_backing_stat(self) -> None:
        total_bytes = max(0, self._expert_host_backing_bytes) + max(
            0, self._expert_global_cpu_backing_bytes
        )
        self.stats.expert_host_backing_mb = total_bytes / float(1024 * 1024)
        self.stats.expert_backing_cache_limit_mb = float(
            self._effective_expert_backing_cache_mb()
        )
        self.stats.expert_cpu_backing_mode = str(self.config.expert_cpu_backing_mode)
        self.stats.expert_cpu_backing_preload_mb = max(
            0, self._expert_global_cpu_backing_bytes
        ) / float(1024 * 1024)
        self.stats.layerkv_runtime_profile = str(self.config.runtime_profile)
        self.stats.layerkv_worker_role = str(self.config.worker_role)

    def _optimized_profile_enabled(self) -> bool:
        return str(self.config.runtime_profile) == "optimized"

    def _simple_profile_enabled(self) -> bool:
        return str(self.config.runtime_profile) == "simple"

    def _effective_expert_backing_cache_mb(self) -> float:
        if self._simple_profile_enabled():
            return 0.0
        return float(max(0.0, self.config.expert_backing_cache_mb))

    def _policy_name(self) -> str:
        policy = str(self.config.policy)
        return "layer-aware-joint-dp" if policy == "coresid" else policy

    def _shared_expert_controller_for_state(self, state):
        controller = getattr(self, "_shared_expert", None)
        if controller is None:
            return None
        lookup = getattr(controller, "controller_for_state", None)
        if callable(lookup):
            return lookup(state)
        return controller if getattr(controller, "state", None) is state else None

    def _coresid_optimized_policy_enabled(self) -> bool:
        return (
            self.config.mode == "kvc-expert"
            and self._optimized_profile_enabled()
            and self._policy_name() in {"layer-aware-joint", "layer-aware-joint-dp"}
        )

    def _dynamic_expert_churn_policy_enabled(self) -> bool:
        return (
            self.config.dynamic_pressure_from_kvc
            and self.config.mode == "kvc-expert"
            and self._policy_name()
            in {"kv-first", "layer-aware-joint", "layer-aware-joint-dp"}
        )

    def _prepared_expert_backing_enabled(self) -> bool:
        return str(self.config.expert_cpu_backing_mode) in {"prepared", "all"}

    def _expert_group_tracking_enabled(self) -> bool:
        # ResidentTensorGroup remains the lifecycle abstraction, but per-expert
        # group bookkeeping is too expensive for the optimized online path.
        # Keep aggregate stats there and reserve detailed group maps for debug /
        # validation modes.
        if self._coresid_optimized_policy_enabled() and self.config.dynamic_pressure_from_kvc:
            return False
        if (
            self.config.runtime_profile == "optimized"
            and self.config.mode == "kvc-expert"
            and self.config.kvc_backend == "per-layer-arena"
            and self.config.shared_expert_enabled
            and not self.config.debug_stats
            and not self.config.profile_detail
            and not self.config.shared_expert_trace_waits
        ):
            return False
        return True

    def _refresh_profile_derived(self) -> None:
        self.stats.profile_detail_enabled = bool(self.config.profile_detail)
        if self._expert_materialize_batch_sizes:
            sizes = sorted(int(x) for x in self._expert_materialize_batch_sizes)
            n = len(sizes)
            p50_idx = min(n - 1, int(0.50 * (n - 1)))
            p95_idx = min(n - 1, int(0.95 * (n - 1)))
            self.stats.expert_materialize_batch_size_p50 = float(sizes[p50_idx])
            self.stats.expert_materialize_batch_size_p95 = float(sizes[p95_idx])
        self.stats.expert_materialize_layers_touched = len(
            self._expert_materialize_layers_touched
        )
        if not self.config.profile_detail:
            return
        # Use non-nested buckets for accounted controller time. Detailed child
        # buckets explain the top-level forward begin/end totals separately.
        accounted = (
            self.stats.profile_install_ms
            + self.stats.profile_set_kv_ms
            + self.stats.profile_forward_begin_ms
            + self.stats.profile_forward_end_ms
            + self.stats.profile_summary_build_ms
        )
        self.stats.profile_accounted_ms = accounted
        self.stats.profile_unaccounted_ms = (
            self.stats.layerkv_python_overhead_ms - accounted
        )
        self.stats.profile_decode_critical_path_ms = (
            self.stats.profile_decode_scheduler_wall_ms
            + self.stats.profile_decode_process_result_ms
        )
        self.stats.profile_controller_per_decode_step_ms = (
            self.stats.layerkv_python_overhead_ms
            / float(max(1, int(self.stats.decode_steps)))
        )

    def on_schedule_batch(
        self, *, schedule_batch: Any, scheduler_context: Dict[str, Any]
    ) -> None:
        """Observe SGLang's native request-level scheduling decision.

        LayerKV keeps request admission, priority ordering, prefill/decode
        selection, and retraction owned by SGLang's Scheduler. This hook only
        records read-only context for residency policy and diagnostics.
        """
        if schedule_batch is None:
            return
        self._finalize_kvc_evictions(block=False)
        context = dict(scheduler_context or {})
        self._last_scheduler_context = context
        self._last_scheduled_req_lens = self._schedule_batch_req_lens(schedule_batch)

        self.stats.native_scheduler_observation_count += 1
        self.stats.native_schedule_policy = str(context.get("schedule_policy", ""))
        self.stats.native_schedule_forward_mode = str(
            context.get("forward_mode")
            or getattr(getattr(schedule_batch, "forward_mode", None), "name", "")
        )
        self.stats.native_schedule_waiting_queue_len = int(
            context.get("waiting_queue_len", 0) or 0
        )
        self._last_scheduler_waiting_context_tokens = max(
            0, int(context.get("waiting_max_context_tokens", 0) or 0)
        )
        self._prefill_admission_probe = {
            key: context.get(key)
            for key in (
                "prefill_adder_remaining_input_tokens",
                "prefill_adder_credit_tokens",
                "prefill_adder_can_run",
                "prefill_adder_rem_total_tokens",
                "prefill_adder_cur_rem_tokens",
                "req_pool_available_size",
            )
        }
        remaining_input = context.get("prefill_adder_remaining_input_tokens")
        if remaining_input is not None:
            remaining_input = int(remaining_input)
            if self._prefill_admission_probe_min_remaining_input is None:
                self._prefill_admission_probe_min_remaining_input = remaining_input
            else:
                self._prefill_admission_probe_min_remaining_input = min(
                    self._prefill_admission_probe_min_remaining_input,
                    remaining_input,
                )
        self._prefill_admission_probe_max_credit = max(
            self._prefill_admission_probe_max_credit,
            int(context.get("prefill_adder_credit_tokens", 0) or 0),
        )
        self._prefill_admission_probe_max_can_run = max(
            self._prefill_admission_probe_max_can_run,
            int(context.get("prefill_adder_can_run", 0) or 0),
        )
        req_pool_available = context.get("req_pool_available_size")
        if req_pool_available is not None:
            req_pool_available = int(req_pool_available)
            if self._prefill_admission_probe_min_req_pool is None:
                self._prefill_admission_probe_min_req_pool = req_pool_available
            else:
                self._prefill_admission_probe_min_req_pool = min(
                    self._prefill_admission_probe_min_req_pool,
                    req_pool_available,
                )
        self._shared_expert_remaining_decode_steps = None
        if (
            self.config.shared_expert_benefit_horizon_steps
            and self.stats.native_schedule_forward_mode.lower() == "decode"
        ):
            from .residency_budget import remaining_fixed_decode_steps

            self._shared_expert_remaining_decode_steps = (
                remaining_fixed_decode_steps(getattr(schedule_batch, "reqs", None))
            )
        self.stats.native_schedule_running_batch_size = int(
            context.get("running_batch_size", 0) or 0
        )
        try:
            self.stats.native_schedule_batch_size = int(schedule_batch.batch_size())
        except Exception:
            self.stats.native_schedule_batch_size = len(
                getattr(schedule_batch, "reqs", []) or []
            )
        self.stats.native_schedule_max_running_requests = int(
            context.get("max_running_requests", 0) or 0
        )
        self.stats.native_schedule_new_token_ratio = float(
            context.get("new_token_ratio", 0.0) or 0.0
        )
        self.stats.native_schedule_kv_available_tokens = int(
            context.get("kv_available_tokens", -1)
        )
        self.stats.native_schedule_overlap_enabled = bool(
            context.get("enable_overlap", False)
        )
        self._record_scheduler_budget_observation(
            context, self._last_scheduled_req_lens
        )

    def _record_scheduler_budget_observation(
        self, context: Dict[str, Any], req_lens: List[Tuple[int, int]]
    ) -> None:
        running_reqs = int(context.get("running_batch_size", 0) or 0)
        if running_reqs <= 0:
            running_reqs = len(req_lens)
        token_count = int(context.get("kv_used_tokens", 0) or 0)
        if token_count <= 0:
            token_count = sum(max(0, int(seq_len)) for _req_idx, seq_len in req_lens)
        token_usage = float(context.get("kv_token_usage", 0.0) or 0.0)
        retracted = int(context.get("num_retracted_reqs", 0) or 0)

        self.stats.scheduler_budget_observation_count += 1
        count = max(1, int(self.stats.scheduler_budget_observation_count))
        self.stats.scheduler_running_req_sum += running_reqs
        self.stats.scheduler_running_req_avg = (
            self.stats.scheduler_running_req_sum / float(count)
        )
        self.stats.scheduler_running_req_max = max(
            int(self.stats.scheduler_running_req_max), running_reqs
        )
        self.stats.scheduler_token_sum += token_count
        self.stats.scheduler_token_avg = self.stats.scheduler_token_sum / float(count)
        self.stats.scheduler_token_max = max(
            int(self.stats.scheduler_token_max), token_count
        )
        self.stats.scheduler_token_usage_sum += token_usage
        self.stats.scheduler_token_usage_avg = (
            self.stats.scheduler_token_usage_sum / float(count)
        )
        self.stats.scheduler_token_usage_max = max(
            float(self.stats.scheduler_token_usage_max), token_usage
        )
        self.stats.scheduler_retracted_req_current = retracted
        self.stats.scheduler_retracted_req_sum += retracted
        self.stats.scheduler_retracted_req_max = max(
            int(self.stats.scheduler_retracted_req_max), retracted
        )

    def _schedule_batch_req_lens(self, schedule_batch: Any) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for req in getattr(schedule_batch, "reqs", []) or []:
            req_pool_idx = getattr(req, "req_pool_idx", None)
            if req_pool_idx is None:
                continue
            try:
                req_idx = int(req_pool_idx)
            except (TypeError, ValueError):
                continue
            seq_len = getattr(req, "seq_len", None)
            if seq_len is None:
                seq_len = len(getattr(req, "origin_input_ids", []) or []) + len(
                    getattr(req, "output_ids", []) or []
                )
            try:
                pairs.append((req_idx, int(seq_len)))
            except (TypeError, ValueError):
                continue
        return pairs

    def _scheduler_credit_tokens(self, *, reason: str = "") -> Tuple[int, int, str]:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return 0, 0, "layerkv_mode_without_kvc"
        if not self.physical_kvc_supported:
            return 0, 0, "physical_kvc_unsupported"
        raw_offloaded = int(max(0, self._offloaded_token_count()))
        if self.config.kvc_backend == "per-layer-arena" and reason in (
            "decode_allocatable",
            "pre_retract_decode_mem",
        ):
            self._refresh_per_layer_allocator_stats()
            credit = int(self.stats.kvc_per_layer_physical_arena_min_free_tokens)
            return max(0, credit), raw_offloaded, "per_layer_arena_physical_allocator"
        if (
            self.config.kvc_backend == "per-layer-arena"
            and reason == "decode_prealloc_admission"
        ):
            credit = self._common_per_layer_reusable_token_count()
            self.stats.kvc_per_layer_physical_arena_common_free_tokens = credit
            return max(0, credit), raw_offloaded, "per_layer_arena_physical_allocator"
        planner_credit = self._planner_scheduler_credit_tokens()
        if planner_credit > 0:
            return int(planner_credit), raw_offloaded, "planner_kvc_reclaim_budget"
        if self.config.kvc_backend == "per-layer-arena":
            return 0, raw_offloaded, "no_planner_kvc_reclaim_budget"
        return raw_offloaded, raw_offloaded, "physical_offloaded_tokens"

    def _planner_scheduler_credit_tokens(self) -> int:
        if self._bytes_per_token_all_layers <= 0:
            return 0
        reclaim_mb = max(
            0.0,
            float(getattr(self.stats, "planner_selected_kvc_reclaim_mb", 0.0) or 0.0),
            float(getattr(self.stats, "effective_kvc_reclaim_mb", 0.0) or 0.0),
            float(getattr(self.stats, "planned_kvc_reclaim_mb", 0.0) or 0.0),
        )
        credit_from_mb = int(
            reclaim_mb * 1024.0 * 1024.0 / float(self._bytes_per_token_all_layers)
        )
        credit_from_plan = 0
        if self.config.kvc_backend == "per-layer-arena":
            layer_count = max(1, len(self._kvc_layer_ids()))
            credit_from_plan = (
                int(max(0, self._planned_kvc_token_target)) // layer_count
            )
        else:
            credit_from_plan = int(max(0, self._planned_kvc_token_target))
        credit = max(credit_from_mb, credit_from_plan)
        total_tokens = self._allocator_total_size()
        if total_tokens > 0:
            credit = min(int(credit), int(total_tokens))
        return max(0, int(credit))

    def _kvc_reclaim_is_scheduler_visible(self) -> bool:
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return False
        if not self.physical_kvc_supported:
            return False
        return self.config.kvc_backend != "per-layer-arena" or bool(
            self._per_layer_allocator_enabled()
        )

    def recall_shared_expert_for_admission(
        self,
        *,
        shortage_tokens: int,
        context_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        allow_kv_overflow: bool = True,
    ) -> int:
        """Return physically restored or activated common KV tokens.

        Called only by an explicit admission decision, not read-only capacity
        queries. Shared mode disables overlap, so recall completes before the
        scheduler can allocate a new request's KV addresses.  When a real
        admission shortage is caused by a long-context, small-batch request,
        an explicitly enabled expert-to-KV overflow tail may be activated at
        this point, before the scheduler rejects the request.
        """
        controller = self._shared_expert
        if (
            int(shortage_tokens) <= 0
            or controller is None
            or self.config.shared_expert_admission_policy == "retain"
        ):
            return 0
        # ``free_kv_donors`` controls whether completely unused KV pages may
        # fund new expert growth.  It must not disable returning an existing
        # expert loan when admission is short: backed-offloaded donors are
        # valid in the default mode and their pages are precisely the KV
        # capacity that admission needs under pressure.
        controller.admission_shortage_tokens = max(
            controller.admission_shortage_tokens, int(shortage_tokens)
        )
        recovered = 0
        if controller.arena.loans:
            before = len(self._common_per_layer_reusable_locs())
            started = time.perf_counter()
            controller.recall()
            recovered = max(0, len(self._common_per_layer_reusable_locs()) - before)
            self.stats.shared_expert_admission_recall_count += 1
            self.stats.shared_expert_admission_recovered_tokens += recovered
            self.stats.shared_expert_admission_recall_ms += (
                time.perf_counter() - started
            ) * 1000.0
        if (
            allow_kv_overflow
            and context_tokens is not None
            and batch_size is not None
            and recovered < int(shortage_tokens)
        ):
            overflow_tokens = controller.prepare_kv_overflow_for_admission(
                context_tokens=int(context_tokens),
                batch_size=int(batch_size),
                shortage_tokens=max(0, int(shortage_tokens) - recovered),
            )
            if overflow_tokens > 0:
                recovered += overflow_tokens
                self.stats.shared_expert_admission_overflow_count += 1
                self.stats.shared_expert_admission_overflow_tokens += overflow_tokens
        return recovered

    def get_scheduler_admission_credit_tokens(
        self,
        *,
        required_tokens: int = 0,
        available_tokens: int = 0,
        reason: str = "",
    ) -> int:
        """Return scheduler-visible prefix KV credit without triggering reclaim."""
        required = max(0, int(required_tokens or 0))
        available = max(0, int(available_tokens or 0))
        shortage = max(0, required - available)
        credit, raw_offloaded, limit_reason = self._scheduler_credit_tokens(
            reason=reason
        )
        effective = available + credit
        if shortage <= 0:
            self._scheduler_pressure_tokens = 0
        self.stats.scheduler_budget_required_tokens = required
        self.stats.scheduler_budget_native_available_tokens = available
        self.stats.scheduler_budget_shortage_tokens = shortage
        self.stats.scheduler_budget_pressure_tokens = int(max(0, shortage - credit))
        self.stats.scheduler_budget_credit_tokens = credit
        self.stats.scheduler_budget_raw_offloaded_tokens = raw_offloaded
        self.stats.scheduler_budget_releasable_tokens = credit
        self.stats.scheduler_budget_credit_limit_reason = limit_reason
        self.stats.scheduler_budget_effective_available_tokens = effective
        if shortage > credit and reason:
            self.stats.scheduler_budget_credit_denied_count += 1
        return credit

    def release_scheduler_admission_credit_tokens(
        self, *, required_tokens: int, available_tokens: int
    ) -> int:
        """Publish common-free per-layer KVC addresses to native admission.

        The per-layer arena can own valid, currently unused locations that are
        intentionally absent from the native allocator's free-page tensor.
        A read-only scheduler credit is enough for a decode feasibility query,
        but ``PrefillAdder`` must receive real native addresses before it can
        allocate a new request.  Release only the residual deficit; active or
        protected KVC locations remain owned by the arena.
        """
        required = max(0, int(required_tokens or 0))
        available = max(0, int(available_tokens or 0))
        if (
            required <= available
            or self.config.kvc_backend != "per-layer-arena"
            or not self._per_layer_allocator_enabled()
        ):
            return 0
        release_fn = getattr(self, "_release_common_per_layer_locs_to_native", None)
        if not callable(release_fn):
            return 0
        return max(0, int(release_fn(required - available) or 0))

    def get_scheduler_allocatable_tokens(
        self,
        *,
        required_tokens: int = 0,
        available_tokens: int = 0,
        reason: str = "decode_allocatable",
    ) -> int:
        """Return scheduler-visible decode capacity without choosing KV slots."""
        available = max(0, int(available_tokens or 0))
        credit = self.get_scheduler_admission_credit_tokens(
            required_tokens=required_tokens,
            available_tokens=available,
            reason=reason,
        )
        return int(available) + max(0, int(credit or 0))

    def prepare_reclaim_for_scheduler(
        self,
        *,
        schedule_batch: Any,
        required_tokens: int,
        available_tokens: int,
        reason: str = "scheduler_pressure",
        wait: bool = True,
    ) -> int:
        """Record scheduler pressure and reclaim only for active decode safety."""
        visible_available = int(available_tokens)
        if self.config.kvc_backend == "per-layer-arena":
            if reason == "pre_retract_decode_mem":
                self._refresh_per_layer_allocator_stats()
                visible_available = int(
                    self.stats.kvc_per_layer_physical_arena_min_free_tokens
                )
            elif reason == "decode_prealloc_admission":
                self._ensure_per_layer_common_free_current()
                visible_available = int(len(self._per_layer_arena_common_free_locs))
                self.stats.kvc_per_layer_physical_arena_common_free_tokens = (
                    visible_available
                )
        shortage = max(0, int(required_tokens) - int(visible_available))
        self._scheduler_pressure_tokens = shortage
        if shortage > 0 and reason == "pre_retract_decode_mem":
            if self._kvc_reclaim_is_scheduler_visible():
                t0 = time.perf_counter()
                self.try_reclaim_kvc_before_retract(
                    schedule_batch=schedule_batch,
                    required_tokens=required_tokens,
                    available_tokens=visible_available,
                    reason=reason,
                )
                self._finalize_kvc_evictions(block=bool(wait))
                self.stats.scheduler_budget_pre_retract_wait_ms += (
                    time.perf_counter() - t0
                ) * 1000.0
                if self.config.kvc_backend == "per-layer-arena":
                    self._refresh_per_layer_allocator_stats()
                    visible_available = int(
                        self.stats.kvc_per_layer_physical_arena_min_free_tokens
                    )
                else:
                    visible_available = int(self._allocator_available_size())
                shortage = max(0, int(required_tokens) - int(visible_available))
                self._scheduler_pressure_tokens = shortage
            else:
                self.stats.kvc_scheduler_invisible_skip_count += 1
                self.stats.kvc_scheduler_invisible_skip_tokens += int(shortage)
        credit = self.get_scheduler_admission_credit_tokens(
            required_tokens=required_tokens,
            available_tokens=visible_available,
            reason=reason,
        )
        self._scheduler_pressure_kvc_blocked = bool(shortage > credit)
        self.stats.scheduler_budget_pressure_tokens = max(
            0, int(shortage) - int(credit)
        )
        if shortage > credit:
            self.stats.kvc_layerwise_scheduler_deadline_reject_count += 1
        if shortage <= 0:
            self._scheduler_pressure_tokens = 0
            self._scheduler_pressure_kvc_blocked = False
            self.stats.scheduler_budget_pressure_tokens = 0
        if shortage > 0 and credit >= shortage:
            self.stats.scheduler_budget_credit_prevented_retract_count += 1
            self.stats.scheduler_budget_credit_used_tokens += min(shortage, credit)
        if shortage > 0 and credit <= 0:
            block_tokens = max(1, int(getattr(self.config, "kvc_block_tokens", 1) or 1))
            if shortage < block_tokens:
                self.stats.scheduler_budget_small_shortage_skip_count += 1
            if not self._kvc_reclaim_is_scheduler_visible():
                self.stats.kvc_scheduler_invisible_skip_count += 1
                self.stats.kvc_scheduler_invisible_skip_tokens += int(shortage)
            self._scheduler_pressure_tokens = 0
            self._scheduler_pressure_kvc_blocked = False
            self.stats.scheduler_budget_pressure_tokens = 0
        return credit

    def _residency_key(
        self, kind: str, layer_id: int, logical_id: Any
    ) -> _LayerKVResidencyKey:
        return _LayerKVResidencyKey(str(kind), int(layer_id), logical_id)

    def _get_or_create_resident_group(
        self,
        *,
        kind: str,
        layer_id: int,
        logical_id: Any,
        state: str,
        bytes: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> _LayerKVResidentTensorGroup:
        key = self._residency_key(kind, layer_id, logical_id)
        group = self._resident_groups.get(key)
        if group is None:
            group = _LayerKVResidentTensorGroup(
                key=key,
                state=str(state),
                bytes=int(max(0, bytes)),
                last_access_step=self._decode_step,
                metadata=dict(metadata or {}),
            )
            self._resident_groups[key] = group
        else:
            group.state = str(state)
            group.bytes = int(max(0, bytes))
            group.last_access_step = self._decode_step
            if metadata:
                group.metadata.update(metadata)
        return group

    def _sync_kvc_group(
        self, entry: _LayerKVResidencyEntry
    ) -> _LayerKVResidentTensorGroup:
        layer_id = int(entry.layer_id)
        key_logical = (
            (layer_id, int(entry.req_idx), int(entry.pos))
            if layer_id >= 0
            else (int(entry.req_idx), int(entry.pos))
        )
        bytes_per_token = (
            self._bytes_per_kvc_token_per_layer()
            if layer_id >= 0
            else self._bytes_per_token_all_layers
        )
        group = self._get_or_create_resident_group(
            kind="kvc",
            layer_id=layer_id,
            logical_id=key_logical,
            state=entry.state,
            bytes=entry.token_count * bytes_per_token,
            metadata={
                "layer_id": layer_id,
                "req_idx": int(entry.req_idx),
                "pos": int(entry.pos),
                "token_count": int(entry.token_count),
                "logical_positions": entry.logical_positions(),
            },
        )
        group.gpu_ref = (
            _LayerKVResidencyHandle(
                kind="kvc",
                location="gpu",
                ref=entry.device_loc_list(),
                n_units=entry.token_count,
                bytes=group.bytes,
                metadata={"device_locs": entry.device_loc_list()},
            )
            if entry.device_loc_list()
            else None
        )
        group.cpu_ref = (
            _LayerKVResidencyHandle(
                kind="kvc",
                location="cpu",
                ref=entry.host_slot_list(),
                n_units=entry.token_count,
                bytes=group.bytes,
                metadata={"host_slots": entry.host_slot_list()},
            )
            if entry.host_slot_list()
            else None
        )
        group.ready_start_event = entry.ready_start_event
        group.ready_event = entry.ready_event
        group.ready_waited = bool(entry.ready_waited)
        return group

    def _sync_kvc_group_if_needed(
        self, entry: _LayerKVResidencyEntry
    ) -> Optional[_LayerKVResidentTensorGroup]:
        if (
            self.config.kvc_backend == "per-layer-arena"
            and self.config.runtime_profile == "optimized"
        ):
            return None
        return self._sync_kvc_group(entry)

    def _remove_resident_group(self, kind: str, layer_id: int, logical_id: Any) -> None:
        self._resident_groups.pop(self._residency_key(kind, layer_id, logical_id), None)

    def _sync_expert_groups_for_state(self, state: _LayerKVExpertLayerState) -> None:
        if not self._expert_group_tracking_enabled():
            self._expert_group_dirty_layers.discard(int(state.layer_id))
            return
        for expert_id in range(int(state.full_num_experts)):
            slot_id = state.logical_to_slot.get(int(expert_id))
            is_resident = slot_id is not None
            group = self._get_or_create_resident_group(
                kind="expert",
                layer_id=state.layer_id,
                logical_id=int(expert_id),
                state="resident" if is_resident else "offloaded",
                bytes=state.expert_bytes,
                metadata={
                    "expert_id": int(expert_id),
                    "slot_capacity": int(state.slot_capacity),
                    "full_num_experts": int(state.full_num_experts),
                },
            )
            group.gpu_ref = (
                _LayerKVResidencyHandle(
                    kind="expert",
                    location="gpu",
                    ref=int(slot_id),
                    n_units=1,
                    bytes=state.expert_bytes,
                    metadata={"slot_id": int(slot_id)},
                )
                if is_resident
                else None
            )
            cpu_params = state.cpu_params.get(
                int(expert_id)
            ) or self._expert_global_cpu_backing.get(
                (int(state.layer_id), int(expert_id))
            )
            group.cpu_ref = (
                _LayerKVResidencyHandle(
                    kind="expert",
                    location="cpu",
                    ref=cpu_params,
                    n_units=1,
                    bytes=state.expert_bytes,
                    metadata={"expert_id": int(expert_id)},
                )
                if cpu_params is not None
                else None
            )
        self._expert_group_dirty_layers.discard(int(state.layer_id))

    def _sync_dirty_expert_groups(self) -> None:
        if not self._expert_group_dirty_layers:
            return
        if not self._expert_group_tracking_enabled():
            self._expert_group_dirty_layers.clear()
            return
        for layer_id in list(self._expert_group_dirty_layers):
            state = self._expert_layers.get(int(layer_id))
            if state is not None:
                self._sync_expert_groups_for_state(state)

    def _mark_expert_group_state(
        self,
        state: _LayerKVExpertLayerState,
        logical_id: int,
        *,
        group_state: str,
        slot_id: Optional[int] = None,
        cpu_params: Optional[Dict[str, torch.Tensor]] = None,
        ready_start_event: Any = None,
        ready_event: Any = None,
    ) -> None:
        if not self._expert_group_tracking_enabled():
            return
        group = self._get_or_create_resident_group(
            kind="expert",
            layer_id=state.layer_id,
            logical_id=int(logical_id),
            state=group_state,
            bytes=state.expert_bytes,
            metadata={"expert_id": int(logical_id)},
        )
        if slot_id is not None:
            group.gpu_ref = _LayerKVResidencyHandle(
                kind="expert",
                location="gpu",
                ref=int(slot_id),
                n_units=1,
                bytes=state.expert_bytes,
                metadata={"slot_id": int(slot_id)},
            )
        elif group_state == "offloaded":
            group.gpu_ref = None
        if cpu_params is None:
            cpu_params = state.cpu_params.get(
                int(logical_id)
            ) or self._expert_global_cpu_backing.get(
                (int(state.layer_id), int(logical_id))
            )
        if cpu_params is not None:
            group.cpu_ref = _LayerKVResidencyHandle(
                kind="expert",
                location="cpu",
                ref=cpu_params,
                n_units=1,
                bytes=state.expert_bytes,
                metadata={"expert_id": int(logical_id)},
            )
        group.ready_start_event = ready_start_event
        group.ready_event = ready_event
        group.ready_waited = False
        group.last_access_step = self._decode_step

    def _refresh_resident_group_stats(self) -> None:
        if not self._expert_group_tracking_enabled():
            if self.config.kvc_backend == "per-layer-arena":
                kvc_groups = self._per_layer_page_table_entry_count()
                kvc_resident = int(self._per_layer_resident_token_count_fast)
                kvc_offloaded = int(self._per_layer_offloaded_token_count_fast)
            else:
                kvc_entries = (
                    self._per_layer_residency.values()
                    if self.config.kvc_backend == "per-layer-arena"
                    else self._residency.values()
                )
                kvc_groups = sum(
                    1
                    for entry in kvc_entries
                    if entry.state in ("resident", "offloaded", "reloading", "evicting")
                )
                kvc_resident = self._resident_token_count()
                kvc_offloaded = self._offloaded_token_count()
            expert_groups = sum(
                int(state.full_num_experts) for state in self._expert_layers.values()
            )
            expert_resident = sum(
                int(state.resident_count) for state in self._expert_layers.values()
            )
            expert_offloaded = max(0, expert_groups - expert_resident)
            self.stats.unified_residency_enabled = True
            self.stats.resident_group_kvc_count = int(kvc_groups)
            self.stats.resident_group_expert_count = int(expert_groups)
            self.stats.resident_group_count = int(kvc_groups + expert_groups)
            self.stats.resident_group_resident_count = int(
                kvc_resident + expert_resident
            )
            self.stats.resident_group_offloaded_count = int(
                kvc_offloaded + expert_offloaded
            )
            self.stats.resident_group_recovering_count = (
                len(self._pending_expert_copy_events)
                + len(self._pending_kvc_reload_events)
                + len(self._pending_kvc_evict_events)
            )
            return
        self._sync_dirty_expert_groups()
        groups = list(self._resident_groups.values())
        self.stats.unified_residency_enabled = True
        self.stats.resident_group_count = len(groups)
        self.stats.resident_group_kvc_count = sum(
            1 for g in groups if g.key.kind == "kvc"
        )
        self.stats.resident_group_expert_count = sum(
            1 for g in groups if g.key.kind == "expert"
        )
        self.stats.resident_group_resident_count = sum(
            1 for g in groups if g.state == "resident"
        )
        self.stats.resident_group_offloaded_count = sum(
            1 for g in groups if g.state == "offloaded"
        )
        self.stats.resident_group_recovering_count = sum(
            1 for g in groups if g.state in ("reloading", "materializing", "evicting")
        )

    def _validate_resident_groups(self) -> None:
        errors = 0
        last_error = ""
        for key, group in self._resident_groups.items():
            if key != group.key:
                errors += 1
                last_error = "group_key_mismatch"
            if group.state == "resident" and group.gpu_ref is None:
                errors += 1
                last_error = f"{key.kind}_resident_missing_gpu_ref"
            if group.state == "offloaded" and group.cpu_ref is None:
                errors += 1
                last_error = f"{key.kind}_offloaded_missing_cpu_ref"
            if (
                group.state in ("reloading", "materializing", "evicting")
                and group.ready_event is None
            ):
                errors += 1
                last_error = f"{key.kind}_recovering_missing_event"
        self.stats.resident_group_state_error_count = errors
        self.stats.resident_group_last_error = last_error

    def install_on_runner(self, runner: Any) -> None:
        if self.installed:
            return
        t0 = time.perf_counter()
        self._runner = runner
        runner.layerkv_runtime = self
        runner_device = getattr(runner, "device", None)
        if str(runner_device).startswith("cuda") or runner_device == "cuda":
            self._copy_stream = torch.cuda.Stream()
            self._expert_d2h_stream = torch.cuda.Stream()
            self._expert_install_stream = torch.cuda.Stream()
            try:
                _low_priority, high_priority = torch.cuda.Stream.priority_range()
                self._expert_h2d_stream = torch.cuda.Stream(priority=high_priority)
            except Exception:
                self._expert_h2d_stream = torch.cuda.Stream()
        self._allocator = getattr(runner, "token_to_kv_pool_allocator", None)
        self._req_to_token_pool = getattr(runner, "req_to_token_pool", None)
        self._install_kv_pool_hooks(getattr(runner, "token_to_kv_pool", None))
        self._install_allocator_hooks(self._allocator)
        if (
            self.config.mode in ("kvc-only", "kvc-expert")
            and self.physical_kvc_supported
            and self.config.reclaim_limit_mb > 0
        ):
            self._ensure_host_store()
        if self.physical_kvc_supported and self.config.kvc_backend in (
            "virtual-arena",
            "per-layer-arena",
        ):
            self._ensure_virtual_scratch()
        self._discover_expert_support(runner)
        if self.config.shared_expert_enabled:
            from .shared_expert import SharedExpertController, SharedExpertManager

            arena = getattr(runner, "layerkv_shared_vmm", None)
            if (
                arena is None
                or not self.physical_kvc_supported
                or not self.physical_expert_supported
            ):
                raise ValueError("LayerKV shared VMM needs supported KV/expert pools")
            if self.config.shared_expert_all_layers:
                layer_ids = [
                    int(layer_id) for layer_id, _module in self._expert_modules
                ]
                if not layer_ids:
                    raise ValueError(
                        "all-layer SharedVMM found no supported expert layers"
                    )
                self._shared_expert = SharedExpertManager(self, arena, layer_ids)
            else:
                self._shared_expert = SharedExpertController(self, arena)
        self.installed = True
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.layerkv_python_overhead_ms += elapsed_ms
        self._add_profile("profile_install_ms", elapsed_ms)
        logger.info(
            "LayerKV enabled mode=%s policy=%s reclaim_limit_mb=%.1f "
            "kvc_supported=%s expert_supported=%s reason=%s",
            self.config.mode,
            self.config.policy,
            self.config.reclaim_limit_mb,
            self.physical_kvc_supported,
            self.physical_expert_supported,
            self.unsupported_reason,
        )

    def _install_kv_pool_hooks(self, kv_pool: Any) -> None:
        if kv_pool is None:
            self.unsupported_reason = "token_to_kv_pool is not initialized"
            return
        self._kv_pool = kv_pool
        if getattr(kv_pool, "_layerkv_wrapped", False):
            return

        def wrap(name: str, wrapper_factory: Callable[[Callable], Callable]) -> None:
            orig = getattr(kv_pool, name, None)
            if orig is None or not callable(orig):
                return
            self._wrapped_methods[name] = orig
            setattr(kv_pool, name, wrapper_factory(orig))

        wrap("set_kv_buffer", self._wrap_set_kv_buffer)
        wrap("get_key_buffer", self._wrap_get_key_buffer)
        wrap("get_value_buffer", self._wrap_get_value_buffer)
        wrap("get_kv_buffer", self._wrap_get_kv_buffer)
        kv_pool._layerkv_wrapped = True
        kv_pool.layerkv_runtime = self
        self.physical_kvc_supported = self._can_support_kvc_pool(kv_pool)
        if not self.physical_kvc_supported and not self.unsupported_reason:
            self.unsupported_reason = (
                f"KVC physical offload is not implemented for "
                f"{type(kv_pool).__name__}; running accounting-only hooks"
            )
        if (
            self.config.mode == "kvc-only"
            and self.config.reclaim_limit_mb > 0
            and not self.physical_kvc_supported
        ):
            self.stats.comparable = False
            self.stats.comparability_reason = self.unsupported_reason

    def _install_allocator_hooks(self, allocator: Any) -> None:
        if allocator is None or getattr(allocator, "_layerkv_allocator_wrapped", False):
            return
        orig_alloc = getattr(allocator, "alloc", None)
        orig_free = getattr(allocator, "free", None)
        if (
            orig_alloc is None
            or not callable(orig_alloc)
            or orig_free is None
            or not callable(orig_free)
        ):
            return
        self._wrapped_methods["allocator.alloc"] = orig_alloc
        self._wrapped_methods["allocator.free"] = orig_free

        @functools.wraps(orig_alloc)
        def wrapped_alloc(need_size: int, *args, **kwargs):
            if (
                self.config.kvc_backend != "per-layer-arena"
                or not self._per_layer_arena_reserved_locs
            ):
                return orig_alloc(need_size, *args, **kwargs)
            need = max(0, int(need_size))
            if need <= 0:
                return orig_alloc(need_size, *args, **kwargs)
            selected: List[int] = []
            seen: Set[int] = set()
            attempts = 0
            while len(selected) < need and attempts < 4:
                chunk = orig_alloc(need - len(selected), *args, **kwargs)
                attempts += 1
                if chunk is None or int(chunk.numel()) == 0:
                    break
                try:
                    locs = chunk.detach().cpu().to(torch.int64).flatten().tolist()
                except Exception:
                    return chunk
                for loc in locs:
                    loc = int(loc)
                    if (
                        loc <= 0
                        or loc in seen
                        or loc in self._per_layer_arena_reserved_locs
                    ):
                        continue
                    selected.append(loc)
                    seen.add(loc)
                    if len(selected) >= need:
                        break
            if len(selected) < need:
                return None
            return torch.tensor(selected, dtype=torch.int64, device=allocator.device)

        @functools.wraps(orig_free)
        def wrapped_free(free_index: torch.Tensor, *args, **kwargs):
            if (
                self.config.kvc_backend != "per-layer-arena"
                or not self._per_layer_arena_reserved_locs
                or free_index is None
                or int(free_index.numel()) == 0
            ):
                return orig_free(free_index, *args, **kwargs)
            try:
                free_cpu = free_index.detach().cpu().to(torch.int64).flatten()
                keep_locs = [
                    int(loc)
                    for loc in free_cpu.tolist()
                    if int(loc) > 0
                    and int(loc) not in self._per_layer_arena_reserved_locs
                ]
                if not keep_locs:
                    return None
                filtered = torch.unique(
                    torch.tensor(keep_locs, dtype=torch.int64, device=allocator.device)
                )
                return orig_free(filtered, *args, **kwargs)
            except Exception:
                return orig_free(free_index, *args, **kwargs)

        setattr(allocator, "alloc", wrapped_alloc)
        setattr(allocator, "free", wrapped_free)
        allocator._layerkv_allocator_wrapped = True
        allocator._layerkv_free_wrapped = True
        allocator.layerkv_runtime = self

    def _can_support_kvc_pool(self, kv_pool: Any) -> bool:
        try:
            from sglang.srt.mem_cache.memory_pool import (
                HybridLinearKVPool,
                MHATokenToKVPool,
            )
        except Exception:
            return False

        page_size = int(getattr(kv_pool, "page_size", 1) or 1)
        storage_pool = kv_pool
        if type(kv_pool) is HybridLinearKVPool and not kv_pool.use_mla:
            storage_pool = kv_pool.full_kv_pool
        if type(storage_pool) is not MHATokenToKVPool:
            self.unsupported_reason = (
                f"KVC physical offload requires non-FP4 MHA storage, got "
                f"{type(kv_pool).__name__}"
            )
            return False
        if self._allocator is None or self._req_to_token_pool is None:
            self.unsupported_reason = "KVC allocator or req_to_token_pool is missing"
            return False

        try:
            if storage_pool is not kv_pool:
                kv_pool = HybridKVCStorageView(kv_pool)
            self._page_size = max(1, page_size)
            self.stats.kvc_page_size = self._page_size
            one_k = kv_pool._get_key_buffer(kv_pool.start_layer)[0].nbytes
            one_v = kv_pool._get_value_buffer(kv_pool.start_layer)[0].nbytes
            self._bytes_per_token_all_layers = int((one_k + one_v) * kv_pool.layer_num)
        except Exception as exc:
            self.unsupported_reason = f"failed to inspect KV token size: {exc}"
            return False
        self._kv_pool = kv_pool
        return self._bytes_per_token_all_layers > 0

    def _ensure_host_store(self) -> None:
        if self._host_store is not None:
            return
        if self._kv_pool is None or self._bytes_per_token_all_layers <= 0:
            raise RuntimeError("KVC host store cannot initialize without KV pool")
        # Allocate host capacity against the total reclaim intent so dynamic
        # policy fractions can vary without reallocating CPU backing.
        target_bytes = max(1, int(self.config.reclaim_limit_mb * 1024 * 1024))
        # _LayerKVHostKVStore allocates one buffer of capacity_tokens for every
        # layer.  Therefore capacity_tokens must be derived from all-layer KV
        # bytes even for per-layer arena mode.  Using per-layer bytes here
        # silently multiplies a 4GB reclaim target by layer_num and can push host
        # RSS above 100GB during decode.
        token_bytes = self._bytes_per_token_all_layers
        target_tokens = max(1, target_bytes // max(1, token_bytes))
        target_tokens = self._align_tokens_up(target_tokens)
        capacity_tokens = target_tokens + max(
            target_tokens, self._align_tokens_up(self.config.kvc_block_tokens)
        )
        self._host_store = _LayerKVHostKVStore(
            self._kv_pool,
            capacity_tokens,
            per_layer_mode=(self.config.kvc_backend == "per-layer-arena"),
            stats=self.stats,
        )
        if self.config.kvc_backend == "per-layer-arena":
            self._host_store.preallocate_per_layer_capacity(
                min(max(1, int(capacity_tokens)), 4096)
            )
        self.stats.kvc_host_capacity_tokens = self._host_store.capacity_tokens
        self.stats.kvc_host_backing_mb = self._host_store.capacity_mb

    def _ensure_virtual_scratch(self) -> bool:
        if self.config.kvc_backend not in ("virtual-arena", "per-layer-arena"):
            return True
        if self._virtual_scratch_locs is not None:
            return True
        if self._allocator is None:
            self.stats.layerkv_kvc_backend_ready = False
            self.stats.layerkv_kvc_backend_reason = "missing allocator"
            return False
        scratch_tokens = max(0, int(self.config.virtual_scratch_tokens))
        if scratch_tokens <= 0:
            self.stats.layerkv_kvc_backend_ready = False
            self.stats.layerkv_kvc_backend_reason = "virtual scratch disabled"
            return False
        locs = self._allocator.alloc(scratch_tokens)
        if locs is None:
            self.stats.virtual_scratch_alloc_failed_count += 1
            self.stats.layerkv_kvc_backend_ready = False
            self.stats.layerkv_kvc_backend_reason = (
                "allocator failed to reserve virtual scratch"
            )
            return False
        self._virtual_scratch_locs = locs.to(dtype=torch.int64)
        self._virtual_scratch_locs_host = {
            int(value) for value in self._virtual_scratch_locs.detach().cpu().tolist()
        }
        self._virtual_scratch_capacity = int(locs.numel())
        self._virtual_scratch_cache_by_layer.clear()
        self._virtual_materialize_plan_reuse_cache.clear()
        self._virtual_scratch_buffer_generations_by_layer.clear()
        self._virtual_materialize_recorded_event_groups.clear()
        self._virtual_scratch_single_generation = 0
        self._metadata_patch_cache.clear()
        self._metadata_patch_tensor_cache.clear()
        scratch_slice = self._contiguous_device_locs_slice(self._virtual_scratch_locs)
        if self._virtual_scratch_capacity >= 2:
            split = max(1, self._virtual_scratch_capacity // 2)
            self._virtual_scratch_buffers = [
                self._virtual_scratch_locs[:split],
                self._virtual_scratch_locs[split:],
            ]
            if scratch_slice[0] >= 0:
                start, length = scratch_slice
                self._virtual_scratch_buffer_slices = [
                    (int(start), min(int(split), int(length))),
                    (
                        int(start) + int(split),
                        max(0, int(length) - int(split)),
                    ),
                ]
            else:
                self._virtual_scratch_buffer_slices = [(-1, 0), (-1, 0)]
            self._virtual_scratch_buffer_generations = [0, 0]
        else:
            self._virtual_scratch_buffers = [self._virtual_scratch_locs]
            self._virtual_scratch_buffer_slices = [scratch_slice]
            self._virtual_scratch_buffer_generations = [0]
        self.stats.virtual_scratch_capacity_tokens = self._virtual_scratch_capacity
        self.stats.layerkv_kvc_backend_ready = True
        self.stats.layerkv_kvc_backend_reason = ""
        return True

    @staticmethod
    def _contiguous_device_locs_slice(locs: torch.Tensor) -> Tuple[int, int]:
        if not isinstance(locs, torch.Tensor) or int(locs.numel()) <= 0:
            return (-1, 0)
        # One-time scratch allocation check.  Keeping this out of decode avoids a
        # per-step CPU sync while still allowing direct slice reloads later.
        try:
            locs_cpu = locs.detach().to(device="cpu", dtype=torch.int64)
            start = int(locs_cpu[0])
            length = int(locs_cpu.numel())
            expected = torch.arange(start, start + length, dtype=torch.int64)
            if bool(torch.equal(locs_cpu, expected)):
                return (start, length)
        except Exception:
            return (-1, 0)
        return (-1, 0)

    def _per_layer_virtual_scratch_enabled(self) -> bool:
        return (
            self.config.kvc_backend == "per-layer-arena"
            and self._per_layer_allocator_enabled()
            and self._virtual_scratch_locs is not None
            and int(self._virtual_scratch_locs.numel()) > 0
        )

    def _per_layer_virtual_scratch_capacity_tokens(self) -> int:
        if not self._per_layer_virtual_scratch_enabled():
            return 0
        return int(self._virtual_scratch_locs.numel())

    def reserved_allocator_tokens(self) -> int:
        if self.config.kvc_backend == "virtual-arena":
            if self._virtual_scratch_locs is None:
                return 0
            return int(self._virtual_scratch_locs.numel())
        if self.config.kvc_backend == "per-layer-arena":
            scratch_tokens = (
                int(self._virtual_scratch_locs.numel())
                if self._virtual_scratch_locs is not None
                else 0
            )
            return int(len(self._per_layer_arena_reserved_locs)) + scratch_tokens
        return 0
