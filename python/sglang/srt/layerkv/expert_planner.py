"""LayerKV LayerKVExpertPlannerMixin implementation."""

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


class LayerKVExpertPlannerMixin:
    def _estimate_expert_reclaim_cost(
        self,
        reclaim_mb: float,
        forward_batch: Any = None,
        *,
        update_stats: bool = True,
        precomputed_layer_items: Optional[List[Tuple[int, int, int]]] = None,
        precomputed_candidates: Optional[
            List[Tuple[float, int, int, int, float, float, float]]
        ] = None,
    ) -> float:
        if reclaim_mb <= 0.0:
            if update_stats:
                self.stats.planner_estimated_expert_churn_count = 0.0
                self.stats.planner_estimated_expert_churn_mb = 0.0
                self.stats.planner_estimated_expert_install_mb = 0.0
            return 0.0
        if precomputed_layer_items is None or precomputed_candidates is None:
            layer_items, candidates = self._build_expert_cost_inputs()
        else:
            layer_items = precomputed_layer_items
            candidates = precomputed_candidates
        target_bytes = int(reclaim_mb * 1024 * 1024)
        reclaimed = 0
        cost = 0.0
        total_churn_count = 0.0
        total_churn_mb = 0.0
        backing_miss_cost = 0.0
        materialize_cost = 0.0
        for (
            candidate_cost,
            expert_bytes,
            _layer_id,
            _expert_id,
            expected_calls,
            candidate_backing_cost,
            candidate_materialize_cost,
        ) in candidates:
            if reclaimed >= target_bytes:
                break
            reclaimed += expert_bytes
            cost += float(candidate_cost)
            total_churn_count += float(expected_calls)
            backing_miss_cost += float(candidate_backing_cost)
            materialize_cost += float(candidate_materialize_cost)
        if reclaimed < target_bytes:
            return 1.0e30
        install_mb = reclaimed / float(1024 * 1024)
        total_churn_mb = total_churn_count * (
            float(layer_items[0][2]) / float(1024 * 1024) if layer_items else 0.0
        )
        if update_stats:
            self.stats.planner_estimated_expert_churn_count = total_churn_count
            self.stats.planner_estimated_expert_churn_mb = total_churn_mb
            self.stats.planner_estimated_expert_install_mb = install_mb
        else:
            self.stats.planner_estimated_expert_churn_count = total_churn_count
            self.stats.planner_estimated_expert_churn_mb = total_churn_mb
            self.stats.planner_estimated_expert_install_mb = install_mb
        self.stats.planner_estimated_expert_backing_miss_cost = backing_miss_cost
        self.stats.planner_estimated_expert_materialize_cost = materialize_cost
        install_cost = install_mb * 0.15
        return cost + install_cost

    def _build_expert_cost_inputs(
        self,
    ) -> Tuple[
        List[Tuple[int, int, int]],
        List[Tuple[float, int, int, int, float, float, float]],
    ]:
        planning_items = self._expert_layer_items_for_planning()
        layer_items = [
            (int(layer_id), int(full_num_experts), int(expert_bytes))
            for layer_id, full_num_experts, expert_bytes, _module, _state in planning_items
        ]
        candidates: List[Tuple[float, int, int, int, float, float, float]] = []
        batch_size = max(1, int(getattr(self.stats, "observed_batch_size", 0) or 1))
        horizon_steps = max(
            1, min(64, int(getattr(self.stats, "decode_steps", 0) or 16))
        )
        for layer_id, full_num_experts, expert_bytes, module, state in planning_items:
            top_k = int(getattr(module, "top_k", 0) or 0)
            if top_k <= 0:
                top_k = int(getattr(module.moe_runner_config, "top_k", 1) or 1)
            for expert_id in range(full_num_experts):
                (
                    cost,
                    expected_calls,
                    backing_miss_cost,
                    materialize_cost,
                ) = self._expert_candidate_expected_cost(
                    int(layer_id),
                    int(expert_id),
                    int(expert_bytes),
                    state,
                    horizon_steps=horizon_steps,
                    batch_size=batch_size,
                    top_k=top_k,
                )
                candidates.append(
                    (
                        float(cost),
                        int(expert_bytes),
                        int(layer_id),
                        int(expert_id),
                        float(expected_calls),
                        float(backing_miss_cost),
                        float(materialize_cost),
                    )
                )
        candidates.sort(key=lambda item: (item[0], item[2], item[3]))
        self.stats.planner_dp_expert_candidate_count = len(candidates)
        return layer_items, candidates

    def _expert_layer_items_for_planning(
        self,
    ) -> List[Tuple[int, int, int, Any, Optional[_LayerKVExpertLayerState]]]:
        items: List[Tuple[int, int, int, Any, Optional[_LayerKVExpertLayerState]]] = []
        for layer_id, module in self._expert_modules:
            state = self._expert_layers.get(int(layer_id))
            if state is not None:
                items.append(
                    (
                        int(layer_id),
                        int(state.full_num_experts),
                        int(state.expert_bytes),
                        module,
                        state,
                    )
                )
            else:
                items.append(
                    (
                        int(layer_id),
                        int(module.w13_weight.data.shape[0]),
                        int(self._expert_bytes(module)),
                        module,
                        None,
                    )
                )
        return items

    def _expert_min_capacity_for_layer(
        self,
        layer_id: int,
        full_num_experts: int,
        module: Any,
    ) -> int:
        top_k = int(getattr(module, "top_k", 0) or 0)
        if top_k <= 0:
            top_k = int(getattr(module.moe_runner_config, "top_k", 1) or 1)
        decode_unique = len(self._expert_hotness_decode.get(int(layer_id), {}))
        if self._dynamic_expert_churn_policy_enabled():
            return max(1, min(int(full_num_experts), top_k))
        return max(1, min(int(full_num_experts), max(top_k, decode_unique)))

    def _expert_candidate_expected_cost(
        self,
        layer_id: int,
        expert_id: int,
        expert_bytes: int,
        state: Optional[_LayerKVExpertLayerState],
        *,
        horizon_steps: int,
        batch_size: int,
        top_k: int,
    ) -> Tuple[float, float, float, float]:
        decode_hotness = self._expert_hotness_decode.get(int(layer_id), {})
        prefill_hotness = self._expert_hotness_prefill.get(int(layer_id), {})
        hotness = decode_hotness or prefill_hotness
        observed_total_calls = int(sum(hotness.values()))
        total_calls = max(1, observed_total_calls)
        decode_count = int(decode_hotness.get(int(expert_id), 0))
        prefill_count = int(prefill_hotness.get(int(expert_id), 0))
        observed_count = int(hotness.get(int(expert_id), 0))
        p = float(observed_count) / float(total_calls)
        expected_calls = p * float(
            max(1, batch_size) * max(1, horizon_steps) * max(1, top_k)
        )
        if decode_count > 0:
            expected_calls = max(1.0, expected_calls)
        mb = float(expert_bytes) / float(1024 * 1024)
        cost = p * mb
        cost += 0.02 * expected_calls
        if decode_count <= 0 and prefill_count > 0:
            cost += 0.002 * float(prefill_count)
        backing_miss_cost = 0.0
        materialize_cost = 0.0
        if expected_calls > 0.0:
            mb = float(expert_bytes) / float(1024 * 1024)
            if (
                self.stats.expert_materialize_mb_total > 0.0
                and self.stats.expert_materialize_ms > 0.0
            ):
                materialize_ms = (
                    self.stats.expert_materialize_ms
                    / self.stats.expert_materialize_mb_total
                    * mb
                )
            else:
                materialize_ms = 0.03 + 0.12 * mb
            materialize_cost = p * materialize_ms
            cost += materialize_cost
            key = (int(layer_id), int(expert_id))
            pending_backing = key in self._pending_expert_d2h_by_key
            if state is None:
                backing_factor = 0.25
            elif int(expert_id) in state.logical_to_slot:
                backing_factor = 0.0
            elif int(expert_id) in state.cpu_params:
                backing_factor = 0.0
            elif pending_backing:
                backing_factor = 0.05
            else:
                backing_factor = 0.25
            backing_miss_cost = p * backing_factor * mb
            cost += backing_miss_cost
        if state is not None and int(expert_id) in state.lru:
            age = max(0, int(self._decode_step) - int(state.lru[int(expert_id)]))
            if age < 64:
                cost += 0.05 * (64.0 - float(age)) / 64.0
        cost += self._expert_candidate_transition_offload_cost(
            layer_id, expert_id, expert_bytes, state
        )
        return (
            float(cost),
            float(expected_calls),
            float(backing_miss_cost),
            float(materialize_cost),
        )

    def _expert_candidate_transition_offload_cost(
        self,
        layer_id: int,
        expert_id: int,
        expert_bytes: int,
        state: Optional[_LayerKVExpertLayerState],
    ) -> float:
        if state is None or int(expert_id) not in state.logical_to_slot:
            return 0.0
        key = (int(layer_id), int(expert_id))
        if int(expert_id) in state.cpu_params or key in self._pending_expert_d2h_by_key:
            return 0.0
        mb = float(expert_bytes) / float(1024 * 1024)
        copied_mb = (
            float(self.stats.expert_install_d2h_async_mb)
            + float(self.stats.expert_eviction_d2h_batched_mb)
            + float(self.stats.expert_eviction_d2h_async_mb)
        )
        if copied_mb > 0.0 and self.stats.profile_copy_expert_to_cpu_ms > 0.0:
            d2h_ms = (
                float(self.stats.profile_copy_expert_to_cpu_ms)
                / max(1e-6, copied_mb)
                * mb
            )
        else:
            d2h_ms = 0.03 + 0.10 * mb
        batched_mb = float(self.stats.expert_eviction_d2h_batched_mb)
        if copied_mb > 0.0:
            exposed_ratio = batched_mb / max(1e-6, copied_mb)
            exposed_ratio = max(0.05, min(0.75, exposed_ratio))
        else:
            exposed_ratio = 0.25
        control_ms = 0.003
        return float(control_ms + exposed_ratio * d2h_ms)

    def _coresid_expert_candidate_cost(
        self,
        layer_id: int,
        expert_id: int,
        expert_bytes: int,
        state: Optional[_LayerKVExpertLayerState],
        *,
        horizon_steps: int,
        batch_size: int,
        top_k: int,
    ) -> Tuple[float, float, float, float]:
        return self._expert_candidate_expected_cost(
            layer_id,
            expert_id,
            expert_bytes,
            state,
            horizon_steps=horizon_steps,
            batch_size=batch_size,
            top_k=top_k,
        )

    def _build_coresid_expert_plan_inputs(self, forward_batch: Any) -> Tuple[
        Dict[int, int],
        Dict[int, int],
        Dict[int, int],
        List[Tuple[float, int, int, int, float, float, float]],
    ]:
        del forward_batch
        capacities: Dict[int, int] = {}
        min_capacities: Dict[int, int] = {}
        expert_bytes_by_layer: Dict[int, int] = {}
        candidates: List[Tuple[float, int, int, int, float, float, float]] = []
        batch_size = max(1, int(getattr(self.stats, "observed_batch_size", 0) or 1))
        horizon_steps = max(
            1, min(64, int(getattr(self.stats, "decode_steps", 0) or 16))
        )
        for (
            layer_id,
            full_num_experts,
            expert_bytes,
            module,
            state,
        ) in self._expert_layer_items_for_planning():
            layer_id = int(layer_id)
            capacities[layer_id] = int(full_num_experts)
            expert_bytes_by_layer[layer_id] = int(expert_bytes)
            top_k = int(getattr(module, "top_k", 0) or 0)
            if top_k <= 0:
                top_k = int(getattr(module.moe_runner_config, "top_k", 1) or 1)
            min_capacity = self._expert_min_capacity_for_layer(
                layer_id, int(full_num_experts), module
            )
            min_capacities[layer_id] = int(min_capacity)
            expert_costs: List[Tuple[float, int, float, float, float]] = []
            for expert_id in range(int(full_num_experts)):
                (
                    cost,
                    expected_calls,
                    backing_miss_cost,
                    materialize_cost,
                ) = self._coresid_expert_candidate_cost(
                    layer_id,
                    int(expert_id),
                    int(expert_bytes),
                    state,
                    horizon_steps=horizon_steps,
                    batch_size=batch_size,
                    top_k=top_k,
                )
                expert_costs.append(
                    (
                        cost,
                        int(expert_id),
                        expected_calls,
                        backing_miss_cost,
                        materialize_cost,
                    )
                )
            expert_costs.sort(key=lambda item: (item[0], item[1]))
            max_evict = max(0, int(full_num_experts) - int(min_capacity))
            for (
                cost,
                expert_id,
                expected_calls,
                backing_miss_cost,
                materialize_cost,
            ) in expert_costs[:max_evict]:
                candidates.append(
                    (
                        float(cost),
                        int(expert_bytes),
                        layer_id,
                        int(expert_id),
                        float(expected_calls),
                        float(backing_miss_cost),
                        float(materialize_cost),
                    )
                )
        candidates.sort(key=lambda item: (item[0], item[2], item[3]))
        self.stats.planner_dp_expert_candidate_count = len(candidates)
        return capacities, min_capacities, expert_bytes_by_layer, candidates

    def _build_coresid_expert_layerwise_plan_table(
        self,
        precomputed: Optional[
            Tuple[
                Dict[int, int],
                Dict[int, int],
                Dict[int, int],
                List[Tuple[float, int, int, int, float, float, float]],
            ]
        ],
    ) -> Optional[Dict[str, Any]]:
        if precomputed is None:
            return None
        base_capacities, min_capacities, _expert_bytes_by_layer, candidates = (
            precomputed
        )
        if not base_capacities:
            return None
        candidates_by_layer: Dict[
            int, List[Tuple[float, int, int, int, float, float, float]]
        ] = {}
        for candidate in candidates:
            layer_id = int(candidate[2])
            candidates_by_layer.setdefault(layer_id, []).append(candidate)
        layer_ids = sorted(int(layer_id) for layer_id in base_capacities)
        states: Dict[
            int,
            Tuple[
                float,
                float,
                float,
                float,
                float,
                float,
                Tuple[int, ...],
                Tuple[float, ...],
            ],
        ] = {0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (), ())}
        for layer_id in layer_ids:
            full_capacity = int(base_capacities[layer_id])
            min_capacity = int(min_capacities.get(layer_id, full_capacity))
            layer_candidates = sorted(
                candidates_by_layer.get(layer_id, []),
                key=lambda item: (float(item[0]), int(item[3])),
            )
            max_evict = max(0, full_capacity - min_capacity)
            points: List[
                Tuple[int, float, float, float, float, float, int, float]
            ] = [(0, 0.0, 0.0, 0.0, 0.0, 0.0, full_capacity, 0.0)]
            reclaimed = 0
            base_cost = 0.0
            expected_calls = 0.0
            churn_mb = 0.0
            backing_cost = 0.0
            materialize_cost = 0.0
            for evicted_count, candidate in enumerate(
                layer_candidates[:max_evict], start=1
            ):
                (
                    candidate_cost,
                    expert_bytes,
                    _layer_id,
                    _expert_id,
                    candidate_expected_calls,
                    candidate_backing_cost,
                    candidate_materialize_cost,
                ) = candidate
                reclaimed += int(expert_bytes)
                base_cost += float(candidate_cost)
                expected_calls += float(candidate_expected_calls)
                churn_mb += (
                    float(candidate_expected_calls)
                    * float(expert_bytes)
                    / float(1024 * 1024)
                )
                backing_cost += float(candidate_backing_cost)
                materialize_cost += float(candidate_materialize_cost)
                install_mb = float(reclaimed) / float(1024 * 1024)
                point_cost = base_cost + 0.15 * install_mb + 0.05 * churn_mb
                points.append(
                    (
                        int(reclaimed),
                        float(point_cost),
                        float(expected_calls),
                        float(churn_mb),
                        float(install_mb),
                        float(backing_cost),
                        max(0, full_capacity - evicted_count),
                        float(materialize_cost),
                    )
                )
            next_states: Dict[
                int,
                Tuple[
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    Tuple[int, ...],
                    Tuple[float, ...],
                ],
            ] = {}
            for prev_bytes, state in states.items():
                (
                    prev_cost,
                    prev_calls,
                    prev_churn_mb,
                    prev_install_mb,
                    prev_backing,
                    prev_materialize,
                    prev_caps,
                    prev_layer_costs,
                ) = state
                for (
                    point_bytes,
                    point_cost,
                    point_calls,
                    point_churn_mb,
                    point_install_mb,
                    point_backing,
                    point_capacity,
                    point_materialize,
                ) in points:
                    total_bytes = int(prev_bytes) + int(point_bytes)
                    candidate_state = (
                        float(prev_cost + point_cost),
                        float(prev_calls + point_calls),
                        float(prev_churn_mb + point_churn_mb),
                        float(prev_install_mb + point_install_mb),
                        float(prev_backing + point_backing),
                        float(prev_materialize + point_materialize),
                        prev_caps + (int(point_capacity),),
                        prev_layer_costs + (float(point_cost),),
                    )
                    existing = next_states.get(total_bytes)
                    if existing is None or candidate_state[0] < existing[0]:
                        next_states[total_bytes] = candidate_state
            pruned: Dict[
                int,
                Tuple[
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    Tuple[int, ...],
                    Tuple[float, ...],
                ],
            ] = {}
            best_cost = float("inf")
            for total_bytes in sorted(next_states, reverse=True):
                state = next_states[total_bytes]
                if state[0] < best_cost - 1e-12:
                    pruned[total_bytes] = state
                    best_cost = state[0]
            states = pruned
        sorted_bytes = sorted(states)
        sorted_states = [states[total_bytes] for total_bytes in sorted_bytes]
        suffix_best: List[int] = [0 for _ in sorted_bytes]
        best_idx = len(sorted_bytes) - 1
        best_cost = float("inf")
        for idx in range(len(sorted_bytes) - 1, -1, -1):
            cost = float(sorted_states[idx][0])
            if cost <= best_cost:
                best_cost = cost
                best_idx = idx
            suffix_best[idx] = best_idx
        return {
            "layer_ids": layer_ids,
            "bytes": sorted_bytes,
            "states": sorted_states,
            "suffix_best": suffix_best,
        }

    def _lookup_coresid_expert_layerwise_cost(
        self, reclaim_mb: float, table: Optional[Dict[str, Any]]
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        Dict[int, int],
        Dict[int, float],
    ]:
        if reclaim_mb <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}
        if not table:
            return 1.0e30, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}
        target_bytes = int(max(0.0, float(reclaim_mb)) * 1024 * 1024)
        sorted_bytes = table.get("bytes", [])
        pos = bisect.bisect_left(sorted_bytes, target_bytes)
        if pos >= len(sorted_bytes):
            return 1.0e30, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}
        best_idx = int(table["suffix_best"][pos])
        state = table["states"][best_idx]
        layer_ids = [int(layer_id) for layer_id in table["layer_ids"]]
        capacities = {
            int(layer_id): int(capacity)
            for layer_id, capacity in zip(layer_ids, state[6])
        }
        layer_costs = {
            int(layer_id): float(cost)
            for layer_id, cost in zip(layer_ids, state[7])
        }
        return (
            float(state[0]),
            float(state[1]),
            float(state[2]),
            float(state[3]),
            float(state[4]),
            float(state[5]),
            capacities,
            layer_costs,
        )

    def _shape_limited_coresid_expert_capacities(
        self,
        reclaim_mb: float,
        precomputed: Optional[
            Tuple[
                Dict[int, int],
                Dict[int, int],
                Dict[int, int],
                List[Tuple[float, int, int, int, float, float, float]],
            ]
        ],
        fallback_capacities: Dict[int, int],
        fallback_layer_costs: Dict[int, float],
    ) -> Tuple[Dict[int, int], Dict[int, float]]:
        if (
            not self.config.dynamic_pressure_from_kvc
            or self.config.policy != "coresid"
            or precomputed is None
        ):
            return fallback_capacities, fallback_layer_costs
        base_capacities, min_capacities, expert_bytes_by_layer, candidates = (
            precomputed
        )
        if not base_capacities or reclaim_mb <= 0.0:
            return fallback_capacities, fallback_layer_costs

        target_bytes = int(max(0.0, float(reclaim_mb)) * 1024 * 1024)
        round_bytes = 0
        for layer_id, full_capacity in base_capacities.items():
            if int(full_capacity) > int(min_capacities.get(layer_id, full_capacity)):
                round_bytes += max(1, int(expert_bytes_by_layer.get(layer_id, 0)))
        if target_bytes <= 0 or round_bytes <= 0:
            return fallback_capacities, fallback_layer_costs

        candidates_by_layer: Dict[
            int, List[Tuple[float, int, int, int, float, float, float]]
        ] = {}
        for candidate in candidates:
            candidates_by_layer.setdefault(int(candidate[2]), []).append(candidate)
        for layer_candidates in candidates_by_layer.values():
            layer_candidates.sort(key=lambda item: (float(item[0]), int(item[3])))

        full_capacity_values = {int(v) for v in base_capacities.values()}
        if len(full_capacity_values) != 1:
            if self.config.debug_stats:
                logger.info(
                    "LayerKV dynamic expert shape limit fallback: mixed full capacities=%s",
                    sorted(full_capacity_values),
                )
            return fallback_capacities, fallback_layer_costs
        full_capacity = int(next(iter(full_capacity_values)))
        existing_capacities = self._planned_expert_slot_capacities_by_layer
        existing_compact_values = sorted(
            {
                int(capacity)
                for capacity in existing_capacities.values()
                if int(capacity) < int(full_capacity)
            }
        )
        existing_compact_capacity = (
            int(existing_compact_values[0]) if existing_compact_values else None
        )
        forced_layers = {
            int(layer_id)
            for layer_id, capacity in existing_capacities.items()
            if int(capacity) < int(full_capacity)
        }
        selection_target_bytes = int(target_bytes)
        if existing_compact_capacity is None:
            lookahead_steps = max(
                0,
                min(
                    32,
                    int(
                        self.config.expert_install_target_steps
                        or self.config.expert_copy_lookahead_layers
                        or 0
                    ),
                ),
            )
            batch_size = max(1, int(getattr(self.stats, "observed_batch_size", 0) or 1))
            token_bytes = max(0, int(self._bytes_per_token_all_layers))
            selection_target_bytes += int(batch_size * lookahead_steps * token_bytes)

        # Dynamic pressure should not fan out Triton MoE shapes across layers.
        # The first plan uses a short pressure lookahead so later decode steps do
        # not keep introducing new E values.  Once a compact E exists, keep that
        # E stable and only add more layers at the same shape.
        best_plan: Optional[Tuple[int, float, int, Dict[int, int], Dict[int, float]]] = (
            None
        )
        best_under_plan: Optional[
            Tuple[int, float, int, Dict[int, int], Dict[int, float], int]
        ] = None
        compact_capacity_iterable = (
            [int(existing_compact_capacity)]
            if existing_compact_capacity is not None
            else range(full_capacity - 1, 0, -1)
        )
        for compact_capacity in compact_capacity_iterable:
            evict_count = full_capacity - int(compact_capacity)
            layer_options: List[Tuple[float, int, int]] = []
            capacities = {
                int(layer_id): int(full_capacity)
                for layer_id in base_capacities
            }
            layer_costs = {int(layer_id): 0.0 for layer_id in base_capacities}
            reclaimed = 0
            total_cost = 0.0
            selected_layers = 0
            forced_valid = True
            for layer_id in sorted(int(layer_id) for layer_id in base_capacities):
                if int(base_capacities[layer_id]) != full_capacity:
                    continue
                min_capacity = int(min_capacities.get(layer_id, full_capacity))
                if int(compact_capacity) < min_capacity:
                    if layer_id in forced_layers:
                        forced_valid = False
                        break
                    continue
                layer_candidates = candidates_by_layer.get(layer_id, [])
                if len(layer_candidates) < evict_count:
                    if layer_id in forced_layers:
                        forced_valid = False
                        break
                    continue
                layer_cost = sum(float(item[0]) for item in layer_candidates[:evict_count])
                layer_bytes = sum(
                    max(1, int(item[1])) for item in layer_candidates[:evict_count]
                )
                if layer_id in forced_layers:
                    capacities[int(layer_id)] = int(compact_capacity)
                    layer_costs[int(layer_id)] = float(layer_cost)
                    reclaimed += int(layer_bytes)
                    total_cost += float(layer_cost)
                    selected_layers += 1
                else:
                    layer_options.append((layer_cost, layer_bytes, layer_id))
            if not forced_valid:
                continue
            if not layer_options:
                if reclaimed < target_bytes:
                    continue
            layer_options.sort(key=lambda item: (float(item[0]), int(item[2])))
            for layer_cost, layer_bytes, layer_id in layer_options:
                if reclaimed >= selection_target_bytes:
                    break
                capacities[int(layer_id)] = int(compact_capacity)
                layer_costs[int(layer_id)] = float(layer_cost)
                reclaimed += int(layer_bytes)
                total_cost += float(layer_cost)
                selected_layers += 1
            candidate_plan = (
                int(compact_capacity),
                float(total_cost),
                int(selected_layers),
                capacities,
                layer_costs,
            )
            if reclaimed < selection_target_bytes:
                under_candidate = candidate_plan + (int(reclaimed),)
                if best_under_plan is None or int(reclaimed) > int(best_under_plan[5]):
                    best_under_plan = under_candidate
                if reclaimed < target_bytes:
                    continue
            best_plan = candidate_plan
            break
        if best_plan is None:
            if best_under_plan is not None and existing_compact_capacity is not None:
                (
                    _compact_capacity,
                    _total_cost,
                    _selected_layers,
                    capacities,
                    layer_costs,
                    reclaimed,
                ) = best_under_plan
                if self.config.debug_stats:
                    logger.info(
                        "LayerKV dynamic expert shape limit capped: target_mb=%.3f "
                        "compact_E=%d selected_layers=%d reclaimed_mb=%.3f",
                        float(reclaim_mb),
                        int(_compact_capacity),
                        int(_selected_layers),
                        float(reclaimed) / float(1024 * 1024),
                    )
                return capacities, layer_costs
            if self.config.debug_stats:
                logger.info(
                    "LayerKV dynamic expert shape limit fallback: target_mb=%.3f "
                    "full=%d layers=%d candidates=%d",
                    float(reclaim_mb),
                    int(full_capacity),
                    len(base_capacities),
                    len(candidates),
                )
            return fallback_capacities, fallback_layer_costs
        _compact_capacity, _total_cost, _selected_layers, capacities, layer_costs = (
            best_plan
        )
        if self.config.debug_stats:
            logger.info(
                "LayerKV dynamic expert shape limit selected: target_mb=%.3f "
                "shape_target_mb=%.3f compact_E=%d selected_layers=%d full=%d",
                float(reclaim_mb),
                float(selection_target_bytes) / float(1024 * 1024),
                int(_compact_capacity),
                int(_selected_layers),
                int(full_capacity),
            )
        return capacities, layer_costs

    def _build_expert_prefix_table(
        self,
        candidates: List[Tuple[float, int, int, int, float, float, float]],
    ) -> Dict[str, Any]:
        prefix_bytes: List[int] = []
        prefix_cost: List[float] = []
        prefix_expected_calls: List[float] = []
        prefix_backing_cost: List[float] = []
        prefix_materialize_cost: List[float] = []
        total_bytes = 0
        total_cost = 0.0
        total_expected_calls = 0.0
        total_backing_cost = 0.0
        total_materialize_cost = 0.0
        for (
            cost,
            expert_bytes,
            _layer_id,
            _expert_id,
            expected_calls,
            backing_cost,
            materialize_cost,
        ) in candidates:
            total_bytes += int(expert_bytes)
            total_cost += float(cost)
            total_expected_calls += float(expected_calls)
            total_backing_cost += float(backing_cost)
            total_materialize_cost += float(materialize_cost)
            prefix_bytes.append(total_bytes)
            prefix_cost.append(total_cost)
            prefix_expected_calls.append(total_expected_calls)
            prefix_backing_cost.append(total_backing_cost)
            prefix_materialize_cost.append(total_materialize_cost)
        return {
            "prefix_bytes": prefix_bytes,
            "prefix_cost": prefix_cost,
            "prefix_expected_calls": prefix_expected_calls,
            "prefix_backing_cost": prefix_backing_cost,
            "prefix_materialize_cost": prefix_materialize_cost,
        }

    def _lookup_expert_prefix_cost(
        self,
        reclaim_mb: float,
        candidates: List[Tuple[float, int, int, int, float, float, float]],
        prefix_table: Dict[str, Any],
        *,
        base_capacities: Optional[Dict[int, int]] = None,
        expert_bytes_by_layer: Optional[Dict[int, int]] = None,
        include_churn_cost: bool = True,
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        Dict[int, int],
        Dict[int, float],
    ]:
        if reclaim_mb <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}
        target_bytes = int(max(0.0, float(reclaim_mb)) * 1024 * 1024)
        prefix_bytes = prefix_table.get("prefix_bytes", [])
        idx = bisect.bisect_left(prefix_bytes, target_bytes)
        if idx >= len(prefix_bytes):
            return 1.0e30, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}
        install_bytes = int(prefix_bytes[idx])
        install_mb = float(install_bytes) / float(1024 * 1024)
        expected_calls = float(prefix_table["prefix_expected_calls"][idx])
        backing_cost = float(prefix_table["prefix_backing_cost"][idx])
        materialize_cost = float(prefix_table["prefix_materialize_cost"][idx])
        base_cost = float(prefix_table["prefix_cost"][idx])
        if expert_bytes_by_layer:
            first_expert_bytes = float(next(iter(expert_bytes_by_layer.values())))
        elif candidates:
            first_expert_bytes = float(candidates[0][1])
        else:
            first_expert_bytes = 0.0
        churn_mb = expected_calls * first_expert_bytes / float(1024 * 1024)
        total_cost = base_cost + 0.15 * install_mb
        if include_churn_cost:
            total_cost += 0.05 * churn_mb
        capacities = {int(k): int(v) for k, v in (base_capacities or {}).items()}
        layer_costs: Dict[int, float] = {int(k): 0.0 for k in capacities}
        for cost, _bytes, layer_id, _expert_id, *_rest in candidates[: idx + 1]:
            layer_id = int(layer_id)
            if capacities:
                capacities[layer_id] = max(0, int(capacities.get(layer_id, 0)) - 1)
            layer_costs[layer_id] = layer_costs.get(layer_id, 0.0) + float(cost)
        return (
            float(total_cost),
            expected_calls,
            churn_mb,
            install_mb,
            backing_cost,
            materialize_cost,
            capacities,
            layer_costs,
        )

    def _plan_coresid_expert_reclaim_cost(
        self,
        reclaim_mb: float,
        forward_batch: Any,
        *,
        precomputed: Optional[
            Tuple[
                Dict[int, int],
                Dict[int, int],
                Dict[int, int],
                List[Tuple[float, int, int, int, float, float, float]],
            ]
        ] = None,
    ) -> Tuple[float, float, float, float, Dict[int, int], Dict[int, float]]:
        if reclaim_mb <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, {}, {}
        target_bytes = int(max(0.0, float(reclaim_mb)) * 1024 * 1024)
        if precomputed is None:
            precomputed = self._build_coresid_expert_plan_inputs(forward_batch)
        base_capacities, min_capacities, expert_bytes_by_layer, candidates = precomputed
        if not base_capacities:
            return 1.0e30, 0.0, 0.0, 0.0, {}, {}
        capacities = {
            int(layer_id): int(capacity)
            for layer_id, capacity in base_capacities.items()
        }
        layer_costs: Dict[int, float] = {int(layer_id): 0.0 for layer_id in capacities}
        reclaimed = 0
        total_cost = 0.0
        install_mb = 0.0
        expected_churn_count = 0.0
        backing_miss_cost_total = 0.0
        materialize_cost_total = 0.0
        for (
            cost,
            expert_bytes,
            layer_id,
            _expert_id,
            expected_calls,
            backing_miss_cost,
            materialize_cost,
        ) in candidates:
            if reclaimed >= target_bytes:
                break
            if capacities[int(layer_id)] <= min_capacities[int(layer_id)]:
                continue
            capacities[int(layer_id)] -= 1
            reclaimed += int(expert_bytes)
            mb = float(expert_bytes) / float(1024 * 1024)
            install_mb += mb
            expected_churn_count += float(expected_calls)
            total_cost += float(cost)
            backing_miss_cost_total += float(backing_miss_cost)
            materialize_cost_total += float(materialize_cost)
            layer_costs[int(layer_id)] = layer_costs.get(int(layer_id), 0.0) + float(
                cost
            )
        if reclaimed < target_bytes:
            return (
                1.0e30,
                expected_churn_count,
                0.0,
                install_mb,
                capacities,
                layer_costs,
            )
        churn_mb = expected_churn_count * (
            float(next(iter(expert_bytes_by_layer.values()))) / float(1024 * 1024)
        )
        total_cost += 0.05 * churn_mb
        total_cost += 0.15 * install_mb
        self.stats.planner_estimated_expert_backing_miss_cost = backing_miss_cost_total
        self.stats.planner_estimated_expert_materialize_cost = materialize_cost_total
        return (
            total_cost,
            expected_churn_count,
            churn_mb,
            install_mb,
            capacities,
            layer_costs,
        )

    def _estimate_expert_slot_capacities_for_target(
        self, target_mb: float
    ) -> Dict[int, int]:
        target_bytes = max(0, int(target_mb * 1024 * 1024))
        layer_infos: List[Dict[str, int]] = []
        allow_dynamic_churn = (
            self.config.dynamic_pressure_from_kvc and self.config.policy == "kv-first"
        )
        if self._expert_layers:
            for state in self._expert_layers.values():
                top_k = int(getattr(state.module, "top_k", 0) or 0)
                if top_k <= 0:
                    top_k = int(
                        getattr(state.module.moe_runner_config, "top_k", 1) or 1
                    )
                observed_decode_unique = len(
                    self._expert_hotness_decode.get(state.layer_id, {})
                )
                if allow_dynamic_churn or self._dynamic_expert_churn_policy_enabled():
                    min_capacity = max(1, min(state.full_num_experts, top_k))
                else:
                    min_capacity = max(
                        1, min(state.full_num_experts, observed_decode_unique)
                    )
                layer_infos.append(
                    {
                        "layer_id": int(state.layer_id),
                        "capacity": int(state.full_num_experts),
                        "min_capacity": int(min_capacity),
                        "expert_bytes": int(state.expert_bytes),
                    }
                )
        else:
            for layer_id, module in self._expert_modules:
                full_num_experts = int(module.w13_weight.data.shape[0])
                top_k = int(getattr(module, "top_k", 0) or 0)
                if top_k <= 0:
                    top_k = int(getattr(module.moe_runner_config, "top_k", 1) or 1)
                observed_decode_unique = len(
                    self._expert_hotness_decode.get(layer_id, {})
                )
                if allow_dynamic_churn or self._dynamic_expert_churn_policy_enabled():
                    min_capacity = max(1, min(full_num_experts, top_k))
                else:
                    min_capacity = max(
                        1,
                        min(full_num_experts, top_k),
                        min(full_num_experts, observed_decode_unique),
                    )
                layer_infos.append(
                    {
                        "layer_id": int(layer_id),
                        "capacity": int(full_num_experts),
                        "min_capacity": int(min_capacity),
                        "expert_bytes": int(self._expert_bytes(module)),
                    }
                )
        reclaimed = 0
        layer_infos.sort(key=lambda x: (-x["expert_bytes"], x["layer_id"]))
        while reclaimed < target_bytes:
            for info in layer_infos:
                if reclaimed >= target_bytes:
                    break
                if int(info["capacity"]) <= int(info["min_capacity"]):
                    continue
                info["capacity"] -= 1
                reclaimed += info["expert_bytes"]
            else:
                break
        return {info["layer_id"]: info["capacity"] for info in layer_infos}

    def _expert_support_reason(self) -> str:
        if self.physical_expert_supported:
            return ""
        return (
            self.stats.expert_guard_reason
            or self.unsupported_reason
            or "expert offload unsupported"
        )

    def _available_kvc_reclaim_mb(self, forward_batch: Any = None) -> float:
        if self._bytes_per_token_all_layers <= 0:
            return 0.0
        token_count = 0
        pairs = (
            list(self._current_forward_req_lens)
            if forward_batch is not None
            and self._current_forward_req_lens_batch_id == id(forward_batch)
            else (
                self._batch_req_indices_and_lens(forward_batch)
                if forward_batch is not None
                else []
            )
        )
        for _req_idx, seq_len in pairs:
            token_count += self._align_tokens_down(max(0, seq_len - 1))
        if token_count <= 0:
            token_count = self._resident_token_count() + self._offloaded_token_count()
        return token_count * self._bytes_per_token_all_layers / float(1024 * 1024)

    def _bytes_per_kvc_token_per_layer(self) -> int:
        layer_num = max(1, len(self._kvc_layer_ids()))
        if self._bytes_per_token_all_layers > 0:
            return max(1, self._bytes_per_token_all_layers // layer_num)
        if self._kv_pool is None:
            return 0
        try:
            layer_id = self._kvc_layer_ids()[0]
            return int(
                self._kv_pool._get_key_buffer(layer_id)[0].nbytes
                + self._kv_pool._get_value_buffer(layer_id)[0].nbytes
            )
        except Exception:
            return 0

    def _kvc_layer_ids(self) -> List[int]:
        if self._kv_pool is None:
            return []
        start = int(getattr(self._kv_pool, "start_layer", 0) or 0)
        layer_num = int(getattr(self._kv_pool, "layer_num", 0) or 0)
        if layer_num <= 0:
            return []
        key = (start, layer_num)
        if self._cached_kvc_layer_ids_key != key:
            self._cached_kvc_layer_ids_key = key
            self._cached_kvc_layer_ids = list(range(start, start + layer_num))
        return self._cached_kvc_layer_ids

    def _build_layer_aware_kvc_token_plan(
        self, total_tokens: int, forward_batch: Any
    ) -> Dict[int, int]:
        layer_ids, max_tokens_per_layer, block_tokens = (
            self._layer_aware_kvc_plan_context(forward_batch)
        )
        if not layer_ids or total_tokens <= 0:
            return {}
        # v1 keeps the DP output layer-aware by assigning more reclaim to later
        # layers, where KVC reload has more forward-path overlap.  This is a
        # deterministic plan and does not change baseline layer-average policy
        # semantics.
        return self._build_layer_aware_kvc_token_plan_from_context(
            total_tokens,
            layer_ids=layer_ids,
            max_tokens_per_layer=max_tokens_per_layer,
            block_tokens=block_tokens,
        )

    def _kvc_tokens_by_layer_json(self, token_count: Any) -> str:
        if isinstance(token_count, dict):
            return json.dumps(
                {
                    str(layer_id): int(tokens)
                    for layer_id, tokens in token_count.items()
                },
                sort_keys=True,
            )
        layer_ids = self._kvc_layer_ids()
        if not layer_ids:
            return json.dumps({"all_layers": int(token_count)}, sort_keys=True)
        return json.dumps(
            {str(layer_id): int(token_count) for layer_id in layer_ids},
            sort_keys=True,
        )

    def _available_expert_reclaim_mb(self) -> float:
        if self.config.expert_collector_only:
            return 0.0
        if not self.physical_expert_supported:
            return 0.0
        total = 0
        if self._expert_layers:
            for state in self._expert_layers.values():
                # Keep at least one slot per layer so the wrapped FusedMoE remains
                # executable; dynamic materialization can grow if routing requires.
                total += max(0, state.full_num_experts - 1) * state.expert_bytes
        else:
            for _layer_id, module in self._expert_modules:
                full_num_experts = int(module.w13_weight.data.shape[0])
                total += max(0, full_num_experts - 1) * self._expert_bytes(module)
        return total / float(1024 * 1024)

    def _refresh_reclaim_target_stats(self, forward_batch: Any = None) -> float:
        cache_key = (
            id(forward_batch),
            self._current_forward_mode,
            int(self._decode_step),
            round(float(self.stats.physical_kvc_reclaim_mb), 3),
            round(float(self.stats.physical_expert_reclaim_mb), 3),
            len(self._expert_layers),
            len(self._per_layer_residency),
            len(self._residency),
        )
        if self._cached_reclaim_target_key == cache_key:
            return self._cached_reclaim_target_value
        configured = max(0.0, self.config.reclaim_limit_mb)
        available_kvc = self._available_kvc_reclaim_mb(forward_batch)
        available_expert = self._available_expert_reclaim_mb()
        available_total = available_kvc + available_expert
        if self.config.dynamic_pressure_from_kvc:
            # End-to-end trace replay should reclaim only when the runtime is
            # actually short on KV allocator headroom.  Available KVC bytes are
            # just reclaimable capacity, not pressure; using them as pressure
            # incorrectly turns any large live KV set into a forced 4G reclaim.
            dynamic_needed = self._dynamic_needed_pressure_mb(forward_batch)
            needed_pressure = (
                min(configured, dynamic_needed) if configured > 0.0 else dynamic_needed
            )
        else:
            # Fig4-style controlled experiments use configured pressure directly
            # so policies remain comparable at a fixed reclaim target.
            needed_pressure = configured
        effective = min(needed_pressure, available_total)
        reason = ""
        if self.config.dynamic_pressure_from_kvc and needed_pressure <= 1e-3:
            reason = "no_runtime_pressure"
        elif effective + 1e-3 < needed_pressure:
            reason = "available_reclaim_below_needed_pressure"
        elif (
            not self.config.dynamic_pressure_from_kvc and effective + 1e-3 < configured
        ):
            reason = "available_reclaim_below_configured_limit"
        self.stats.configured_reclaim_limit_mb = configured
        self.stats.needed_pressure_mb = needed_pressure
        self.stats.available_kvc_reclaim_mb = available_kvc
        self.stats.available_expert_reclaim_mb = available_expert
        self.stats.available_total_reclaim_mb = available_total
        self.stats.effective_reclaim_target_mb = effective
        self.stats.target_limited_reason = reason
        self._cached_reclaim_target_key = cache_key
        self._cached_reclaim_target_value = effective
        return effective

    def _set_no_pressure_reclaim_stats(self, configured: float) -> None:
        self.stats.configured_reclaim_limit_mb = float(configured)
        self.stats.needed_pressure_mb = 0.0
        self.stats.effective_reclaim_target_mb = 0.0
        self.stats.target_limited_reason = "no_runtime_pressure"
        self.stats.requested_total_reclaim_mb = 0.0
        self.stats.effective_kvc_reclaim_mb = 0.0
        self.stats.planned_kvc_reclaim_mb = 0.0
        self.stats.planned_expert_reclaim_mb = 0.0
        self.stats.policy_kvc_fraction = 0.0
        self.stats.policy_expert_fraction = 0.0
        self.stats.full_policy_semantics_supported = True
        self.stats.policy_semantics_reason = ""
        if self.config.mode == "kvc-expert" and not self._expert_plan_applied:
            self._expert_install_state = "waiting_for_pressure"
            self.stats.expert_install_state = self._expert_install_state

    def _dynamic_runtime_pressure_mb(self, forward_batch: Any = None) -> float:
        if not self.config.dynamic_pressure_from_kvc:
            return max(0.0, float(self.config.reclaim_limit_mb))
        configured = max(0.0, float(self.config.reclaim_limit_mb))
        dynamic_needed = max(0.0, float(self._dynamic_needed_pressure_mb(forward_batch)))
        return min(configured, dynamic_needed) if configured > 0.0 else dynamic_needed

    def _no_pressure_fast_path_active(self, forward_batch: Any = None) -> bool:
        if not self.config.dynamic_pressure_from_kvc:
            return False
        if self.config.mode not in ("kvc-only", "kvc-expert"):
            return False
        if self._current_forward_mode != "decode":
            return False
        if self._pending_kvc_reload_events or self._pending_kvc_evict_events:
            return False
        if self._pending_virtual_kvc_materialize:
            return False
        if (
            self._pending_expert_copy_events
            or self._pending_expert_d2h_events
            or self._expert_install_d2h_queue
        ):
            return False
        if self._expert_install_queue or self._expert_plan_applied:
            return False
        if self._has_offloaded_kvc_entries():
            return False
        if (
            self.stats.physical_kvc_reclaim_mb > 1e-3
            or self.stats.physical_expert_reclaim_mb > 1e-3
        ):
            return False

        configured = max(0.0, float(self.config.reclaim_limit_mb))
        needed = self._dynamic_runtime_pressure_mb(forward_batch)
        if needed > 1e-3:
            return False
        self._set_no_pressure_reclaim_stats(configured)
        return True

    def _refresh_physical_reclaim_peaks(
        self, *, record_step_sample: bool = False
    ) -> None:
        total = (
            self.stats.physical_kvc_reclaim_mb + self.stats.physical_expert_reclaim_mb
        )
        self.stats.physical_total_reclaim_mb = total
        self.stats.physical_kvc_reclaim_peak_mb = max(
            self.stats.physical_kvc_reclaim_peak_mb,
            self.stats.physical_kvc_reclaim_mb,
        )
        self.stats.physical_total_reclaim_peak_mb = max(
            self.stats.physical_total_reclaim_peak_mb,
            total,
        )
        if record_step_sample:
            self.stats.physical_kvc_reclaim_step_sum_mb += (
                self.stats.physical_kvc_reclaim_mb
            )
            self.stats.physical_kvc_reclaim_step_count += 1
            self.stats.physical_kvc_reclaim_step_mean_mb = (
                self.stats.physical_kvc_reclaim_step_sum_mb
                / float(max(1, self.stats.physical_kvc_reclaim_step_count))
            )

    def _effective_kvc_reclaim_mb(self, forward_batch: Any) -> float:
        kvc_fraction, expert_fraction, full_supported, reason = self._policy_fractions(
            forward_batch
        )
        policy = self.config.policy
        if policy == "coresid":
            policy = "layer-aware-joint-dp"
        target_mb = self._refresh_reclaim_target_stats(forward_batch)
        effective = min(
            max(0.0, target_mb * kvc_fraction),
            max(0.0, self.stats.available_kvc_reclaim_mb),
        )
        expert_deficit = 0.0
        if self._expert_plan_applied and expert_fraction > 0.0:
            planned_expert = target_mb * expert_fraction
            physical_expert = max(0.0, self.stats.physical_expert_reclaim_mb)
            expert_deficit = max(0.0, planned_expert - physical_expert)
            if expert_deficit > 1e-3:
                if policy == "kv-first":
                    # kv-first means preserve KV residency and satisfy pressure
                    # with expert residency only. Do not silently contaminate it
                    # with KVC reclaim when expert physical reclaim falls short.
                    if not reason:
                        reason = "expert_reclaim_deficit_no_kvc_fallback"
                elif policy in ("layer-aware-joint", "layer-aware-joint-dp"):
                    # CoResid must execute the same split chosen by the DP.  If
                    # expert reclaim lags due to delayed install/replan, do not
                    # silently shift the missing bytes to KVC; that changes the
                    # policy being measured.  The high-watermark expert path
                    # will try to catch up in _apply_expert_plan_once.
                    if not reason:
                        reason = "expert_reclaim_deficit_plan_execution_mismatch"
                else:
                    self.stats.planned_expert_reclaim_mb = physical_expert
                    effective = min(target_mb, effective + expert_deficit)
                    if not reason:
                        reason = "expert_reclaim_deficit_shifted_to_kvc"
        self.stats.requested_total_reclaim_mb = target_mb
        self.stats.effective_kvc_reclaim_mb = effective
        self.stats.policy_kvc_fraction = kvc_fraction
        self.stats.policy_expert_fraction = expert_fraction
        self.stats.full_policy_semantics_supported = full_supported
        if (
            self.config.dynamic_pressure_from_kvc
            and self.config.kvc_backend == "token-slot"
            and effective > 0.0
        ):
            # Under true KV-table pressure, token-slot KVC offload is not a
            # safe steady-state baseline: reloading a required prefix needs new
            # allocator slots at the attention use point, but the allocator may
            # already be full. Keep token-slot baselines to expert offloading
            # only; layer-aware KVC migration uses the per-layer arena backend.
            effective = 0.0
            self.stats.effective_kvc_reclaim_mb = 0.0
            self.stats.planned_kvc_reclaim_mb = 0.0
            if not reason:
                reason = "token_slot_kvc_disabled_under_dynamic_pressure"
            self.stats.policy_semantics_reason = reason
            if not self.stats.target_limited_reason:
                self.stats.target_limited_reason = reason
        self.stats.policy_semantics_reason = reason
        if reason and not self.stats.target_limited_reason:
            self.stats.target_limited_reason = reason
        self.stats.planned_kvc_reclaim_mb = effective
        return effective

    def _effective_expert_reclaim_mb(self, forward_batch: Any) -> float:
        kvc_fraction, expert_fraction, full_supported, reason = self._policy_fractions(
            forward_batch
        )
        target_mb = self._refresh_reclaim_target_stats(forward_batch)
        self._planner_target_high_watermark_mb = max(
            float(self._planner_target_high_watermark_mb), float(target_mb)
        )
        effective = max(0.0, target_mb * expert_fraction)
        self.stats.requested_total_reclaim_mb = target_mb
        self.stats.policy_kvc_fraction = kvc_fraction
        self.stats.policy_expert_fraction = expert_fraction
        self.stats.full_policy_semantics_supported = full_supported
        self.stats.policy_semantics_reason = reason
        self.stats.planned_expert_reclaim_mb = effective
        return effective

    def _expert_evictions_json_from_capacities(self, capacities: Dict[int, int]) -> str:
        try:
            result: Dict[str, int] = {}
            for layer_id, module in self._expert_modules:
                state = self._expert_layers.get(int(layer_id))
                full_num_experts = (
                    int(state.full_num_experts)
                    if state is not None
                    else int(module.w13_weight.data.shape[0])
                )
                result[str(int(layer_id))] = max(
                    0,
                    full_num_experts
                    - int(capacities.get(int(layer_id), full_num_experts)),
                )
            return json.dumps(result, sort_keys=True)
        except Exception:
            return ""

    def _expert_evictions_from_json(self, payload: str) -> Dict[str, int]:
        if not payload:
            return {}
        try:
            raw = json.loads(payload)
            if not isinstance(raw, dict):
                return {}
            return {str(k): max(0, int(v)) for k, v in raw.items()}
        except Exception:
            return {}

    def _applied_expert_evictions_cover_plan(
        self, *, planned: str, applied: str
    ) -> bool:
        planned_counts = self._expert_evictions_from_json(planned)
        applied_counts = self._expert_evictions_from_json(applied)
        if not planned_counts or not applied_counts:
            return False
        return all(
            int(applied_counts.get(layer_id, 0)) >= int(evictions)
            for layer_id, evictions in planned_counts.items()
        )

    def _sync_coresid_expert_plan_stats(self, *, context: str) -> None:
        if self.config.policy != "coresid":
            return
        capacities = self._planned_expert_slot_capacities_by_layer
        applied_capacities: Optional[Dict[int, int]] = None
        if self._expert_plan_applied or self._expert_install_queue:
            applied_capacities = self._current_applied_expert_capacities()
            signature = (
                tuple(sorted((int(k), int(v)) for k, v in capacities.items())),
                tuple(
                    sorted((int(k), int(v)) for k, v in applied_capacities.items())
                ),
                str(context),
            )
            if (
                self._coresid_plan_stats_signature == signature
                and bool(self.stats.expert_plan_match)
                and bool(self.stats.selected_expert_evictions_by_layer)
                and bool(self.stats.applied_expert_evictions_by_layer)
            ):
                return
            self._coresid_plan_stats_signature = signature
        if capacities:
            self.stats.selected_expert_capacity_by_layer = json.dumps(
                {
                    str(layer_id): int(capacity)
                    for layer_id, capacity in capacities.items()
                },
                sort_keys=True,
            )
            self.stats.selected_expert_cost_by_layer = json.dumps(
                {
                    str(layer_id): round(float(cost), 6)
                    for layer_id, cost in self._planned_expert_cost_by_layer.items()
                },
                sort_keys=True,
            )
            self.stats.selected_expert_evictions_by_layer = (
                self._expert_evictions_json_from_capacities(capacities)
            )
        if self._expert_plan_applied or self._expert_install_queue:
            self._check_current_coresid_plan_match(
                context=context, applied_capacities=applied_capacities
            )
        elif capacities and not self.stats.applied_expert_evictions_by_layer:
            self.stats.applied_expert_evictions_by_layer = ""
            self.stats.expert_plan_match = False
            self.stats.expert_plan_mismatch_reason = (
                f"planned_expert_plan_not_applied:{context}"
            )
            self.stats.comparable = False
            self.stats.comparability_reason = self.stats.expert_plan_mismatch_reason

    def _coresid_planned_expert_capacities(
        self,
        target_mb: float,
        forward_batch: Any,
    ) -> Dict[int, int]:
        if self.config.policy != "coresid":
            return self._plan_expert_slot_capacities(target_mb)
        if (
            not self._planned_expert_slot_capacities_by_layer
            or float(target_mb)
            > float(self._planned_expert_target_mb)
            + max(1e-3, self._expert_reclaim_quantum_mb())
        ):
            plan_target_mb = max(
                float(target_mb), float(self._planned_expert_target_mb)
            )
            precomputed = self._build_coresid_expert_plan_inputs(forward_batch)
            layerwise_table = self._build_coresid_expert_layerwise_plan_table(
                precomputed
            )
            (
                _cost,
                churn_count,
                churn_mb,
                install_mb,
                backing_miss_cost,
                materialize_cost,
                capacities,
                layer_costs,
            ) = self._lookup_coresid_expert_layerwise_cost(
                plan_target_mb, layerwise_table
            )
            planned_capacities = {
                int(layer_id): int(capacity)
                for layer_id, capacity in capacities.items()
            }
            planned_capacities, layer_costs = (
                self._shape_limited_coresid_expert_capacities(
                    plan_target_mb,
                    precomputed,
                    planned_capacities,
                    layer_costs,
                )
            )
            self._planned_expert_slot_capacities_by_layer = planned_capacities
            self._planned_expert_cost_by_layer = {
                int(layer_id): float(cost) for layer_id, cost in layer_costs.items()
            }
            self._planned_expert_target_mb = float(plan_target_mb)
            self.stats.planner_estimated_expert_churn_count = float(churn_count)
            self.stats.planner_estimated_expert_churn_mb = float(churn_mb)
            self.stats.planner_estimated_expert_install_mb = float(install_mb)
            self.stats.planner_estimated_expert_backing_miss_cost = float(
                backing_miss_cost
            )
            self.stats.planner_estimated_expert_materialize_cost = float(
                materialize_cost
            )
        capacities = dict(self._planned_expert_slot_capacities_by_layer)
        self.stats.selected_expert_capacity_by_layer = json.dumps(
            {str(layer_id): int(capacity) for layer_id, capacity in capacities.items()},
            sort_keys=True,
        )
        self.stats.selected_expert_cost_by_layer = json.dumps(
            {
                str(layer_id): round(float(cost), 6)
                for layer_id, cost in self._planned_expert_cost_by_layer.items()
            },
            sort_keys=True,
        )
        self.stats.selected_expert_evictions_by_layer = (
            self._expert_evictions_json_from_capacities(capacities)
        )
        return capacities

    def _record_applied_expert_plan(
        self, capacities: Dict[int, int], *, context: str
    ) -> None:
        applied = self._expert_evictions_json_from_capacities(capacities)
        self.stats.applied_expert_evictions_by_layer = applied
        if self.config.policy != "coresid":
            self.stats.expert_plan_match = True
            self.stats.expert_plan_mismatch_reason = ""
            return
        planned = self.stats.selected_expert_evictions_by_layer
        if planned and applied and planned != applied:
            if self._applied_expert_evictions_cover_plan(
                planned=planned, applied=applied
            ):
                self.stats.expert_plan_match = True
                self.stats.expert_plan_mismatch_reason = ""
                if self.stats.comparability_reason.startswith(
                    (
                        "planned_applied_expert_plan_mismatch:",
                        "missing_planned_expert_plan:",
                        "planned_expert_plan_not_applied:",
                    )
                ):
                    self.stats.comparable = True
                    self.stats.comparability_reason = ""
                return
            self.stats.expert_plan_match = False
            self.stats.expert_plan_mismatch_reason = (
                f"planned_applied_expert_plan_mismatch:{context}"
            )
            self.stats.comparable = False
            self.stats.comparability_reason = self.stats.expert_plan_mismatch_reason
        else:
            self.stats.expert_plan_match = True
            self.stats.expert_plan_mismatch_reason = ""
            if self.stats.comparability_reason.startswith(
                (
                    "planned_applied_expert_plan_mismatch:",
                    "missing_planned_expert_plan:",
                    "planned_expert_plan_not_applied:",
                )
            ):
                self.stats.comparable = True
                self.stats.comparability_reason = ""

    def _pin_coresid_plan_to_current_expert_residency(self, *, context: str) -> None:
        if self.config.policy != "coresid":
            return
        capacities = self._current_applied_expert_capacities()
        if not capacities:
            return
        self._planned_expert_slot_capacities_by_layer = dict(capacities)
        self._planned_expert_cost_by_layer = {}
        self.stats.selected_expert_capacity_by_layer = json.dumps(
            {str(layer_id): int(capacity) for layer_id, capacity in capacities.items()},
            sort_keys=True,
        )
        self.stats.selected_expert_cost_by_layer = ""
        self.stats.selected_expert_evictions_by_layer = (
            self._expert_evictions_json_from_capacities(capacities)
        )
        self.stats.planned_expert_reclaim_mb = max(
            0.0, float(self.stats.physical_expert_reclaim_mb)
        )
        self._record_applied_expert_plan(capacities, context=context)

    def _current_applied_expert_capacities(self) -> Dict[int, int]:
        capacities: Dict[int, int] = {}
        queued = {
            int(item.layer_id): int(item.slot_capacity)
            for item in self._expert_install_queue
        }
        for layer_id, module in self._expert_modules:
            layer_id = int(layer_id)
            state = self._expert_layers.get(layer_id)
            if state is not None:
                capacities[layer_id] = int(state.slot_capacity)
            elif layer_id in queued:
                capacities[layer_id] = int(queued[layer_id])
            else:
                capacities[layer_id] = int(module.w13_weight.data.shape[0])
        return capacities

    def _check_current_coresid_plan_match(
        self,
        *,
        context: str,
        applied_capacities: Optional[Dict[int, int]] = None,
    ) -> None:
        if self.config.policy != "coresid":
            return
        if not self._expert_plan_applied and not self._expert_install_queue:
            return
        planned = self.stats.selected_expert_evictions_by_layer
        if not planned:
            self.stats.expert_plan_match = False
            self.stats.expert_plan_mismatch_reason = (
                f"missing_planned_expert_plan:{context}"
            )
            self.stats.comparable = False
            self.stats.comparability_reason = self.stats.expert_plan_mismatch_reason
            return
        if applied_capacities is None:
            applied_capacities = self._current_applied_expert_capacities()
        applied = self._expert_evictions_json_from_capacities(applied_capacities)
        self.stats.applied_expert_evictions_by_layer = applied
        if applied != planned:
            if self._applied_expert_evictions_cover_plan(
                planned=planned, applied=applied
            ):
                self.stats.expert_plan_match = True
                self.stats.expert_plan_mismatch_reason = ""
                if self.stats.comparability_reason.startswith(
                    (
                        "planned_applied_expert_plan_mismatch:",
                        "missing_planned_expert_plan:",
                        "planned_expert_plan_not_applied:",
                    )
                ):
                    self.stats.comparable = True
                    self.stats.comparability_reason = ""
                return
            self.stats.expert_plan_match = False
            self.stats.expert_plan_mismatch_reason = (
                f"planned_applied_expert_plan_mismatch:{context}"
            )
            self.stats.comparable = False
            self.stats.comparability_reason = self.stats.expert_plan_mismatch_reason
        else:
            self.stats.expert_plan_match = True
            self.stats.expert_plan_mismatch_reason = ""
            if self.stats.comparability_reason.startswith(
                (
                    "planned_applied_expert_plan_mismatch:",
                    "missing_planned_expert_plan:",
                    "planned_expert_plan_not_applied:",
                )
            ):
                self.stats.comparable = True
                self.stats.comparability_reason = ""


