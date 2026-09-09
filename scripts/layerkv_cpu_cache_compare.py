#!/usr/bin/env python3
"""Compare matched cache, accounting or remap artifacts; no GPU work."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def compare(control, cached, kind="cache"):
    summaries = [read(p / "summary.json") for p in (control, cached)]
    if not all(s["valid"] for s in summaries):
        raise ValueError("both source summaries must pass all guards")
    plans = [read(p / "plan.json")["args"] for p in (control, cached)]
    changed_arg, candidate_label = {
        "cache": ("expert_backing_cache_mb", "cached"),
        "accounting": ("expert_backing_cache_accounting", "incremental"),
        "remap": ("expert_remap_update", "batch"),
    }[kind]
    ignored = {"output_dir", changed_arg}
    if {k: v for k, v in plans[0].items() if k not in ignored} != {
        k: v for k, v in plans[1].items() if k not in ignored
    }:
        raise ValueError("unmatched experiment arguments")
    if kind == "cache" and (
        plans[0]["expert_backing_cache_mb"] != 0
        or plans[1]["expert_backing_cache_mb"] <= 0
    ):
        raise ValueError("expected cache-zero control and positive-cache candidate")
    if kind == "accounting" and (
        plans[0][changed_arg] != "scan" or plans[1][changed_arg] != "incremental"
    ):
        raise ValueError("expected scan control and incremental candidate")
    if kind == "remap" and (
        plans[0][changed_arg] != "scalar" or plans[1][changed_arg] != "batch"
    ):
        raise ValueError("expected scalar control and batch candidate")
    arms = {}
    for arm in ("fixed16", "grow17"):
        engine_args = [read(p / f"{arm}.engine_args.json") for p in (control, cached)]
        budget_keys = {
            "layerkv_expert_backing_cache_mb",
            "layerkv_expert_cpu_backing_pool_mb",
        }
        allowed_keys = budget_keys if kind == "cache" else {f"layerkv_{changed_arg}"}
        if {k: v for k, v in engine_args[0].items() if k not in allowed_keys} != {
            k: v for k, v in engine_args[1].items() if k not in allowed_keys
        } or sum(engine_args[0][k] for k in budget_keys) != sum(
            engine_args[1][k] for k in budget_keys
        ):
            raise ValueError(f"{arm}: unmatched engine arguments or host budget")
        data = [read(p / f"{arm}.json") for p in (control, cached)]
        requests = [d["requests"] for d in data]
        if (
            len(requests[0]) != len(requests[1])
            or len(requests[0]) != plans[0]["rounds"]
        ):
            raise ValueError("incomplete request set")
        rows = []
        for index, (a, b) in enumerate(zip(*requests)):
            outputs = [
                r["response"]["meta_info"]["output_token_logprobs"] for r in (a, b)
            ]
            if any(len(output) != plans[0]["output_tokens"] for output in outputs):
                raise ValueError(f"{arm} request {index + 1}: incomplete outputs")
            exact = (
                a["response"]["output_ids"] == b["response"]["output_ids"]
                and outputs[0] == outputs[1]
            )
            unchanged = all(
                a["stats_delta"][key] == b["stats_delta"][key]
                for key in (
                    "expert_materialize_count",
                    "expert_copy_descriptor_h2d_count",
                    "expert_cuda_batch_h2d_count",
                )
            ) and all(
                a["shared_delta"][key] == b["shared_delta"][key]
                for key in (
                    "token_chunk_calls",
                    "token_chunk_materializations",
                    "kv_to_expert_pages",
                    "expert_to_kv_pages",
                    "growth_physical_create_count",
                )
            )
            if not exact or not unchanged:
                raise ValueError(
                    f"{arm} request {index + 1}: outputs or residency differ"
                )
            if kind in ("accounting", "remap"):
                keys = (
                    "expert_cuda_batch_d2h_count",
                    "expert_cpu_backing_pool_alloc_count",
                    "expert_cpu_backing_pool_reuse_count",
                    "expert_cpu_backing_pool_drop_count",
                    "expert_eviction_d2h_batched_mb",
                )
                if any(a["stats_delta"][k] != b["stats_delta"][k] for k in keys):
                    raise ValueError(
                        f"{arm} request {index + 1}: transfer or pool work differs"
                    )
                keys = (
                    "mandatory_valid_bytes",
                    "cached_valid_bytes",
                    "idle_pool_bytes",
                    "tracked_unique_storage_bytes",
                    "optional_limit_bytes",
                )
                if any(
                    a["final_stats"]["expert_host_budget"][k]
                    != b["final_stats"]["expert_host_budget"][k]
                    for k in keys
                ):
                    raise ValueError(f"{arm} request {index + 1}: host usage differs")
            if kind == "remap" and (
                a["final_stats"]["expert_remap_batch_count"] != 0
                or b["final_stats"]["expert_remap_batch_count"] <= 0
                or b["final_stats"]["expert_remap_batch_entries"] <= 0
            ):
                raise ValueError(f"{arm}: remap modes were not exercised")
            row = {
                "round": a["round"],
                "first_request": a["first_request"],
                "exact_outputs_and_logprobs": exact,
                "unchanged_loads_and_loans": unchanged,
            }
            for label, r, summary in zip(
                ("control", candidate_label), (a, b), summaries
            ):
                row[label] = {
                    **{
                        k: r[k]
                        for k in ("client_e2e_s", "ttft_s", "decode_s", "tpot_ms")
                    },
                    "tail_tpot_ms": summary["rows"][index][arm]["tail_tpot_ms"],
                    "host": r["final_stats"]["expert_host_budget"],
                    "transfers": {
                        k: v
                        for k, v in r["stats_delta"].items()
                        if "d2h" in k or "h2d" in k or "backing_pool" in k
                    },
                }
            rows.append(row)
        totals = {
            label: {
                "all_requests_s": sum(r["client_e2e_s"] for r in reqs),
                "later_requests_s": sum(r["client_e2e_s"] for r in reqs[1:]),
                "engine_startup_s": d["engine_startup_s"],
            }
            for label, reqs, d in zip(("control", candidate_label), requests, data)
        }
        totals["e2e_reduction_percent"] = 100 * (
            1
            - totals[candidate_label]["all_requests_s"]
            / totals["control"]["all_requests_s"]
        )
        arms[arm] = {"totals": totals, "requests": rows}
    return {"control": str(control), candidate_label: str(cached), "arms": arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-kind", choices=["cache", "accounting", "remap"], default="cache"
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        type=Path,
        action="append",
        required=True,
        metavar=("CONTROL", "CANDIDATE"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "valid": True,
        "pairs": [compare(*pair, kind=args.comparison_kind) for pair in args.pair],
    }
    # Preserve previous reports and raw experiment artifacts.
    with args.output.open("x") as report:
        json.dump(result, report, indent=2)
    for pair in result["pairs"]:
        print(
            pair["control"],
            pair[
                {"cache": "cached", "accounting": "incremental", "remap": "batch"}[
                    args.comparison_kind
                ]
            ],
        )
        for arm, values in pair["arms"].items():
            print(arm, json.dumps(values["totals"]))


if __name__ == "__main__":
    main()
