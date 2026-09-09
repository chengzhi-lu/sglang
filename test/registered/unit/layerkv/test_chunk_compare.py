"""CPU-only artifact comparison for changing, but budget-matched, residency."""

import importlib.util
import json
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[4] / "scripts/layerkv_chunk_compare.py"
_spec = importlib.util.spec_from_file_location("chunk_compare", _path)
comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comparison)


def pair(tmp_path):
    paths = [tmp_path / name for name in ("input", "reuse")]
    for path in paths:
        path.mkdir()
        reuse = path.name == "reuse"
        plan = dict(
            output_dir=str(path),
            chunk_order=path.name,
            rounds=2,
            output_tokens=3,
            decode_tail_start=1,
            expert_initial_slots=16,
            expert_host_extra_budget_mb=256,
            profile_chunks=False,
            profile_prepare=False,
            trace_waits_request=0,
            check_cache_accounting=False,
        )
        s = dict.fromkeys(
            (
                "kvc_stale_entry_count",
                "kvc_page_alignment_violation_count",
                "kvc_physical_failure_count",
                "kvc_host_used_tokens",
                "expert_cuda_batch_pending",
            ),
            0,
        )
        s.update(comparable=True, kvc_guard_pass=True, expert_guard_pass=True)
        v = dict.fromkeys(
            (
                "loan_bytes",
                "blocked_kv_tokens",
                "growth_physical_create_count",
                "token_chunk_profile_pending",
                "kv_to_expert_pages",
                "expert_to_kv_pages",
            ),
            0,
        )
        v.update(
            ownership_guard_pass=True,
            expert_pointers_stable=True,
            current_expert_slots=16,
        )
        s["shared_vmm"] = v
        s["expert_host_budget"] = dict(
            optional_guard_pass=True,
            ledger_matches=True,
            cache_accounting_matches=True,
            batch_owner_storage_bytes=0,
            optional_limit_bytes=256 * 1024**2,
            tracked_unique_storage_bytes=100,
            mandatory_valid_bytes=100,
        )
        rows = []
        for i in range(2):
            shared = dict(
                token_chunk_materializations=(8 if reuse else 10) if i else 0,
                token_chunk_calls=2 if i else 0,
                token_chunk_reordered_groups=2 if reuse and i else 0,
                borrowed_slot_use_count=0,
                kv_to_expert_pages=0,
                expert_to_kv_pages=0,
            )
            rows.append(
                dict(
                    round=i + 1,
                    first_request=i == 0,
                    stream_timing_valid=True,
                    client_e2e_s=3,
                    ttft_s=1,
                    decode_s=2,
                    tpot_ms=1000,
                    events=[dict(tokens=1, elapsed_s=1), dict(tokens=3, elapsed_s=3)],
                    response=dict(
                        output_ids=[42, 42, 42],
                        meta_info=dict(output_token_logprobs=[[-0.5, 42, None]] * 3),
                    ),
                    stats_delta=dict(
                        expert_materialize_count=21 if reuse else 20,
                        expert_materialize_mb_total=126 if reuse else 120,
                        expert_cuda_batch_h2d_count=3,
                        expert_cuda_batch_d2h_count=0,
                        expert_eviction_d2h_batched_mb=0,
                    ),
                    shared_delta=shared,
                    final_stats=s,
                )
            )
        for name, value in [
            ("plan.json", dict(args=plan)),
            ("summary.json", dict(valid=True)),
        ]:
            (path / name).write_text(json.dumps(value))
        for arm in ("fixed16", "grow17"):
            (path / f"{arm}.json").write_text(
                json.dumps(dict(engine_startup_s=1, requests=rows))
            )
            (path / f"{arm}.engine_args.json").write_text(
                json.dumps(
                    dict(layerkv_shared_expert_chunk_order=path.name, host_budget=256)
                )
            )
    return paths


def test_changed_loads_are_reported_not_rejected(tmp_path):
    result = comparison.compare(*pair(tmp_path))
    row = result["arms"]["fixed16"]["requests"][1]
    assert row["input"]["chunk_materializations"] == 10
    assert row["reuse"]["chunk_materializations"] == 8
    assert row["input"]["non_chunk_materializations"] == 10
    assert row["reuse"]["non_chunk_materializations"] == 13
    assert row["reuse"]["tail_tpot_ms"] == 1000
    assert (
        result["arms"]["fixed16"]["totals"]["reuse"]["later"]["sums"][
            "total_materializations"
        ]
        == 21
    )


@pytest.mark.parametrize(
    "damage",
    [
        "args",
        "engine",
        "summary",
        "profile",
        "output",
        "logprob",
        "nan",
        "missing",
        "counts",
        "groups",
        "guard",
        "pending",
        "host",
        "mode",
    ],
)
def test_invalid_comparison_rejected(tmp_path, damage):
    paths = pair(tmp_path)
    filename = {
        "args": "plan.json",
        "engine": "grow17.engine_args.json",
        "summary": "summary.json",
        "profile": "plan.json",
    }.get(damage, "grow17.json")
    target = paths[1] / filename
    d = json.loads(target.read_text())
    if damage == "args":
        d["args"]["expert_host_extra_budget_mb"] = 512
    elif damage == "engine":
        d["host_budget"] = 512
    elif damage == "summary":
        d["valid"] = False
    elif damage == "profile":
        for path in paths:
            p = json.loads((path / "plan.json").read_text())
            p["args"]["profile_chunks"] = True
            (path / "plan.json").write_text(json.dumps(p))
        with pytest.raises(ValueError, match="profilers"):
            comparison.compare(*paths)
        return
    else:
        r = d["requests"][1]
        if damage == "output":
            r["response"]["output_ids"][0] = 43
        elif damage == "logprob":
            r["response"]["meta_info"]["output_token_logprobs"][0][0] = -0.6
        elif damage == "nan":
            r["decode_s"] = float("nan")
        elif damage == "missing":
            r["events"].pop()
        elif damage == "counts":
            r["shared_delta"]["token_chunk_materializations"] = 100
        elif damage == "groups":
            r["shared_delta"]["token_chunk_calls"] = 3
        elif damage == "guard":
            r["final_stats"]["comparable"] = False
        elif damage == "pending":
            r["final_stats"]["expert_cuda_batch_pending"] = 1
        elif damage == "host":
            r["final_stats"]["expert_host_budget"]["optional_limit_bytes"] = 1
        elif damage == "mode":
            r["shared_delta"]["token_chunk_reordered_groups"] = 0
    target.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        comparison.compare(*paths)
