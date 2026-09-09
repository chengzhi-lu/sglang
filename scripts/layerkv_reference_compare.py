#!/usr/bin/env python3
"""Compare a LayerKV artifact with a native/reference artifact.

This is a CPU-only diagnostic.  It never launches a model and deliberately
keeps throughput separate from correctness: a different decode-batch shape is
reported as unknown or mismatched rather than being treated as a valid speedup
comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"artifact root must be an object: {path}")
    return value


def _round(data: Dict[str, Any], number: int) -> Dict[str, Any]:
    rounds = data.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("artifact has no rounds")
    if not 1 <= number <= len(rounds):
        raise ValueError(f"round {number} is outside 1..{len(rounds)}")
    value = rounds[number - 1]
    if not isinstance(value, dict):
        raise ValueError(f"round {number} must be an object")
    return value


def _responses(row: Dict[str, Any], label: str) -> Dict[int, Dict[str, Any]]:
    values = row.get("responses")
    if not isinstance(values, list):
        raise ValueError(f"{label} round has no response list")
    result: Dict[int, Dict[str, Any]] = {}
    for position, response in enumerate(values):
        if not isinstance(response, dict):
            raise ValueError(f"{label} response {position} is not an object")
        index = response.get("index", position)
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"{label} response {position} has invalid index")
        if index in result:
            raise ValueError(f"{label} has duplicate response index {index}")
        result[index] = response
    return result


def _logprob_value(entry: Any) -> Optional[float]:
    if isinstance(entry, (list, tuple)):
        if not entry:
            return None
        entry = entry[0]
    if isinstance(entry, (int, float)) and math.isfinite(float(entry)):
        return float(entry)
    return None


def _logprobs(response: Dict[str, Any]) -> Optional[List[Optional[float]]]:
    meta = response.get("meta_info")
    if not isinstance(meta, dict):
        return None
    values = meta.get("output_token_logprobs")
    if not isinstance(values, list):
        return None
    return [_logprob_value(value) for value in values]


def _first_id_difference(
    candidate: Dict[str, Any], reference: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    left = candidate.get("output_ids")
    right = reference.get("output_ids")
    if not isinstance(left, list) or not isinstance(right, list):
        return {
            "reason": "missing_output_ids",
            "candidate_length": len(left) if isinstance(left, list) else None,
            "reference_length": len(right) if isinstance(right, list) else None,
        }
    for position, (candidate_id, reference_id) in enumerate(zip(left, right)):
        if candidate_id != reference_id:
            return {
                "response_index": candidate.get("index"),
                "token_position": position,
                "candidate_id": candidate_id,
                "reference_id": reference_id,
            }
    if len(left) != len(right):
        return {
            "response_index": candidate.get("index"),
            "token_position": min(len(left), len(right)),
            "reason": "length_mismatch",
            "candidate_length": len(left),
            "reference_length": len(right),
        }
    return None


def _selected_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    stats = row.get("final_stats")
    if not isinstance(stats, dict):
        return {"available": False}
    selected = {
        key: stats.get(key)
        for key in (
            "comparable",
            "kvc_guard_pass",
            "expert_guard_pass",
            "expert_cuda_batch_pending",
            "expert_ready_use_check_count",
            "expert_ready_before_use_count",
            "expert_ready_miss_stall_count",
            "expert_topk_remap_error_count",
            "expert_topk_remap_range_invalid_count",
            "expert_materialize_count",
            "expert_evict_count",
        )
        if key in stats
    }
    vmm = stats.get("shared_vmm")
    if isinstance(vmm, dict):
        selected["shared_vmm"] = {
            key: vmm.get(key)
            for key in (
                "physical_bytes",
                "loan_bytes",
                "blocked_kv_tokens",
                "current_expert_slots",
                "growth_physical_create_count",
                "ownership_guard_pass",
                "expert_pointers_stable",
            )
            if key in vmm
        }
    selected["available"] = True
    return selected


def _normalize_histogram(value: Any, label: str) -> Dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    normalized = {}
    for key, count in value.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{label} counts must be nonnegative integers")
        normalized[str(key)] = int(count)
    return normalized


def compare(
    candidate_path: Path,
    reference_path: Path,
    candidate_round: int = 1,
    reference_round: int = 1,
    logprob_atol: float = 0.01,
    reference_decode_batch_histogram: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    if logprob_atol < 0 or not math.isfinite(logprob_atol):
        raise ValueError("logprob-atol must be finite and nonnegative")
    candidate_data = _read(candidate_path)
    reference_data = _read(reference_path)
    candidate = _round(candidate_data, candidate_round)
    reference = _round(reference_data, reference_round)
    candidate_responses = _responses(candidate, "candidate")
    reference_responses = _responses(reference, "reference")

    common = sorted(set(candidate_responses) & set(reference_responses))
    missing_candidate = sorted(set(reference_responses) - set(candidate_responses))
    missing_reference = sorted(set(candidate_responses) - set(reference_responses))
    first_id_difference = None
    exact_output_ids = not missing_candidate and not missing_reference
    logprob_values: List[float] = []
    logprob_available = True
    logprob_length_mismatch = False
    for index in common:
        left = candidate_responses[index]
        right = reference_responses[index]
        if first_id_difference is None:
            first_id_difference = _first_id_difference(left, right)
        left_logprobs = _logprobs(left)
        right_logprobs = _logprobs(right)
        if left_logprobs is None or right_logprobs is None:
            logprob_available = False
            continue
        if len(left_logprobs) != len(right_logprobs):
            logprob_length_mismatch = True
        for candidate_value, reference_value in zip(left_logprobs, right_logprobs):
            if candidate_value is None or reference_value is None:
                logprob_available = False
                continue
            logprob_values.append(abs(candidate_value - reference_value))
    exact_output_ids = (
        exact_output_ids
        and len(common) == len(candidate_responses) == len(reference_responses)
        and first_id_difference is None
    )
    max_logprob_delta = max(logprob_values, default=None)
    logprobs_within_atol = (
        logprob_available
        and not logprob_length_mismatch
        and max_logprob_delta is not None
        and max_logprob_delta <= logprob_atol
    )

    candidate_histogram = candidate.get("actual_decode_batch_histogram")
    reference_histogram = reference.get("actual_decode_batch_histogram")
    reference_shape_source = "artifact"
    if isinstance(candidate_histogram, dict) and isinstance(reference_histogram, dict):
        decode_shape_match: Optional[bool] = candidate_histogram == reference_histogram
        decode_shape_status = "matched" if decode_shape_match else "mismatched"
    elif isinstance(candidate_histogram, dict) and reference_decode_batch_histogram is not None:
        reference_histogram = _normalize_histogram(
            reference_decode_batch_histogram,
            "reference-decode-batch-histogram",
        )
        candidate_shape = _normalize_histogram(
            candidate_histogram, "candidate decode batch histogram"
        )
        decode_shape_match = candidate_shape == reference_histogram
        decode_shape_status = (
            "matched-explicit-reference-shape"
            if decode_shape_match
            else "mismatched-explicit-reference-shape"
        )
        reference_shape_source = "explicit"
    else:
        decode_shape_match = None
        decode_shape_status = "unknown"

    workload_match = candidate.get("workload") == reference.get("workload")
    candidate_stats = _selected_stats(candidate)
    guard_values = [
        candidate_stats.get(key)
        for key in ("kvc_guard_pass", "expert_guard_pass")
        if key in candidate_stats
    ]
    candidate_guard_pass = bool(guard_values) and all(
        value is True for value in guard_values
    )
    pending = candidate_stats.get("expert_cuda_batch_pending")
    pending_clear = pending == 0 if pending is not None else None
    strict_valid = bool(
        workload_match
        and exact_output_ids
        and logprobs_within_atol
        and decode_shape_match is True
        and candidate_guard_pass
        and pending_clear is True
    )
    return {
        "candidate": str(candidate_path),
        "reference": str(reference_path),
        "candidate_round": candidate_round,
        "reference_round": reference_round,
        "candidate_reference_only": bool(candidate_data.get("reference_only")),
        "reference_reference_only": bool(reference_data.get("reference_only")),
        "workload_match": workload_match,
        "candidate_workload": candidate.get("workload"),
        "reference_workload": reference.get("workload"),
        "response_count": {
            "candidate": len(candidate_responses),
            "reference": len(reference_responses),
            "common": len(common),
            "missing_candidate": missing_candidate,
            "missing_reference": missing_reference,
        },
        "exact_output_ids": exact_output_ids,
        "first_id_difference": first_id_difference,
        "logprobs_available": logprob_available,
        "logprob_samples": len(logprob_values),
        "max_logprob_delta": max_logprob_delta,
        "logprob_atol": logprob_atol,
        "logprobs_within_atol": logprobs_within_atol,
        "decode_shape_status": decode_shape_status,
        "decode_shape_match": decode_shape_match,
        "candidate_decode_batch_histogram": candidate_histogram,
        "reference_decode_batch_histogram": reference_histogram,
        "reference_decode_batch_histogram_source": reference_shape_source,
        "candidate_throughput": {
            key: candidate.get(key)
            for key in (
                "client_e2e_s",
                "output_tokens",
                "output_tokens_per_s",
                "actual_decode_batch_mean",
            )
            if key in candidate
        },
        "reference_throughput": {
            key: reference.get(key)
            for key in (
                "client_e2e_s",
                "output_tokens",
                "output_tokens_per_s",
                "actual_decode_batch_mean",
            )
            if key in reference
        },
        "candidate_stats": candidate_stats,
        "candidate_guard_pass": candidate_guard_pass,
        "candidate_pending_clear": pending_clear,
        "strict_valid": strict_valid,
        "strict_invalid_reasons": [
            reason
            for reason, passed in (
                ("workload_mismatch", workload_match),
                ("output_ids_mismatch", exact_output_ids),
                ("logprob_mismatch_or_unavailable", logprobs_within_atol),
                ("decode_shape_not_matched", decode_shape_match is True),
                ("candidate_guard_failure_or_missing", candidate_guard_pass),
                ("pending_expert_transfer", pending_clear is True),
            )
            if not passed
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate-round", type=int, default=1)
    parser.add_argument("--reference-round", type=int, default=1)
    parser.add_argument("--logprob-atol", type=float, default=0.01)
    parser.add_argument(
        "--reference-decode-batch-histogram",
        type=json.loads,
        help=(
            "explicit reference schedule shape as a JSON object when the "
            "reference artifact is reference-only and has no histogram"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON here; otherwise print the report only",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return 1 when the report is not strictly valid",
    )
    args = parser.parse_args()
    if args.candidate_round <= 0 or args.reference_round <= 0:
        parser.error("round numbers must be positive")
    try:
        result = compare(
            args.candidate,
            args.reference,
            args.candidate_round,
            args.reference_round,
            args.logprob_atol,
            args.reference_decode_batch_histogram,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(rendered + "\n")
    print(rendered)
    return 1 if args.strict and not result["strict_valid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
