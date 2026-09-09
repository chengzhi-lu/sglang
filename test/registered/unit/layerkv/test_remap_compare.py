"""Artifact comparison must reject unmatched remap performance experiments."""

import importlib.util
import json
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[4] / "scripts/layerkv_cpu_cache_compare.py"
_spec = importlib.util.spec_from_file_location("remap_compare", _path)
comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comparison)


def pair(tmp_path):
    paths = [tmp_path / "scalar", tmp_path / "batch"]
    for path in paths:
        path.mkdir()
        mode = path.name
        plan = dict(
            output_dir=str(path),
            expert_remap_update=mode,
            rounds=1,
            output_tokens=1,
            trace_waits_request=0,
        )
        summary = dict(
            valid=True, rows=[{a: dict(tail_tpot_ms=1) for a in ("fixed16", "grow17")}]
        )
        stats = dict.fromkeys(
            (
                "expert_materialize_count",
                "expert_copy_descriptor_h2d_count",
                "expert_cuda_batch_h2d_count",
                "expert_cuda_batch_d2h_count",
                "expert_cpu_backing_pool_alloc_count",
                "expert_cpu_backing_pool_reuse_count",
                "expert_cpu_backing_pool_drop_count",
                "expert_eviction_d2h_batched_mb",
            ),
            1,
        )
        shared = dict.fromkeys(
            (
                "token_chunk_calls",
                "token_chunk_materializations",
                "kv_to_expert_pages",
                "expert_to_kv_pages",
                "growth_physical_create_count",
            ),
            0,
        )
        host = dict.fromkeys(
            (
                "mandatory_valid_bytes",
                "cached_valid_bytes",
                "idle_pool_bytes",
                "tracked_unique_storage_bytes",
                "optional_limit_bytes",
            ),
            128,
        )
        request = dict(
            round=1,
            first_request=True,
            client_e2e_s=1,
            ttft_s=0.5,
            decode_s=0.5,
            tpot_ms=1,
            response=dict(
                output_ids=[42],
                meta_info=dict(output_token_logprobs=[[-0.5, 42, None]]),
            ),
            stats_delta=stats,
            shared_delta=shared,
            final_stats=dict(
                expert_host_budget=host,
                expert_remap_batch_count=int(mode == "batch"),
                expert_remap_batch_entries=int(mode == "batch"),
            ),
        )

        def write(name, data):
            (path / name).write_text(json.dumps(data))

        write("plan.json", dict(args=plan))
        write("summary.json", summary)
        for arm in ("fixed16", "grow17"):
            write(f"{arm}.json", dict(engine_startup_s=0.5, requests=[request]))
            write(
                f"{arm}.engine_args.json",
                dict(
                    layerkv_expert_remap_update=mode,
                    layerkv_expert_backing_cache_mb=128,
                    layerkv_expert_cpu_backing_pool_mb=128,
                ),
            )
    return paths


def test_matched_remap_pair(tmp_path):
    paths = pair(tmp_path)
    result = comparison.compare(*paths, kind="remap")
    assert result["batch"] == str(paths[1])
    assert result["arms"]["fixed16"]["requests"][0]["exact_outputs_and_logprobs"]


@pytest.mark.parametrize(
    "damage", ["args", "engine", "output", "transfer", "host", "disabled", "guard"]
)
def test_unmatched_pair_rejected(tmp_path, damage):
    paths = pair(tmp_path)
    filename = "grow17.json"
    if damage == "args":
        filename = "plan.json"
    elif damage == "engine":
        filename = "grow17.engine_args.json"
    elif damage == "guard":
        filename = "summary.json"
    target = paths[1] / filename
    data = json.loads(target.read_text())
    if damage == "args":
        data["args"]["trace_waits_request"] = 1
    elif damage == "engine":
        data["layerkv_expert_cpu_backing_pool_mb"] += 1
    elif damage == "guard":
        data["valid"] = False
    else:
        r = data["requests"][0]
        if damage == "output":
            r["response"]["output_ids"] = [43]
        elif damage == "transfer":
            r["stats_delta"]["expert_cuda_batch_d2h_count"] += 1
        elif damage == "host":
            r["final_stats"]["expert_host_budget"]["idle_pool_bytes"] += 1
        else:
            r["final_stats"]["expert_remap_batch_count"] = 0
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        comparison.compare(*paths, kind="remap")
