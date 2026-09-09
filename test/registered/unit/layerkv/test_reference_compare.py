"""CPU-only tests for native/reference artifact comparison."""

import importlib.util
import json
from pathlib import Path


_path = Path(__file__).resolve().parents[4] / "scripts/layerkv_reference_compare.py"
_spec = importlib.util.spec_from_file_location("reference_compare", _path)
comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comparison)


def _artifact(path, *, histogram, token=42, logprob=-0.5):
    responses = [
        {
            "index": 0,
            "output_ids": [token, token],
            "meta_info": {
                "output_token_logprobs": [
                    [logprob, token, None],
                    [logprob, token, None],
                ]
            },
        }
    ]
    path.write_text(
        json.dumps(
            {
                "rounds": [
                    {
                        "workload": {"requests": 1, "output_tokens": 2},
                        "responses": responses,
                        "actual_decode_batch_histogram": histogram,
                        "actual_decode_batch_mean": 1,
                        "output_tokens_per_s": 10,
                        "final_stats": {
                            "kvc_guard_pass": True,
                            "expert_guard_pass": True,
                            "expert_cuda_batch_pending": 0,
                        },
                    }
                ]
            }
        )
    )


def test_reference_compare_requires_shape_match_for_strict_validity(tmp_path):
    candidate = tmp_path / "candidate.json"
    reference = tmp_path / "reference.json"
    _artifact(candidate, histogram={"1": 2})
    _artifact(reference, histogram={"1": 2})

    result = comparison.compare(candidate, reference)

    assert result["exact_output_ids"] is True
    assert result["logprobs_within_atol"] is True
    assert result["decode_shape_match"] is True
    assert result["strict_valid"] is True


def test_reference_compare_reports_first_difference_without_launching(tmp_path):
    candidate = tmp_path / "candidate.json"
    reference = tmp_path / "reference.json"
    _artifact(candidate, histogram={"2": 1}, token=43)
    _artifact(reference, histogram={"1": 1})

    result = comparison.compare(candidate, reference)

    assert result["exact_output_ids"] is False
    assert result["first_id_difference"] == {
        "response_index": 0,
        "token_position": 0,
        "candidate_id": 43,
        "reference_id": 42,
    }
    assert result["decode_shape_status"] == "mismatched"
    assert "output_ids_mismatch" in result["strict_invalid_reasons"]


def test_reference_only_accepts_explicit_schedule_shape(tmp_path):
    candidate = tmp_path / "candidate.json"
    reference = tmp_path / "reference.json"
    _artifact(candidate, histogram={"1": 2})
    _artifact(reference, histogram={"1": 2})
    reference_data = json.loads(reference.read_text())
    reference_data["rounds"][0].pop("actual_decode_batch_histogram")
    reference.write_text(json.dumps(reference_data))

    result = comparison.compare(
        candidate,
        reference,
        reference_decode_batch_histogram={"1": 2},
    )

    assert result["decode_shape_match"] is True
    assert result["decode_shape_status"] == "matched-explicit-reference-shape"
    assert result["strict_valid"] is True
