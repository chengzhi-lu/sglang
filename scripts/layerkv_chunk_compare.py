#!/usr/bin/env python3
"""Compare matched, unprofiled input/reuse runs without launching GPU work."""

import argparse
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def measured(request, tail_start, output_tokens):
    s, v = request["stats_delta"], request["shared_delta"]
    total = s["expert_materialize_count"]
    chunk = v["token_chunk_materializations"]
    require(0 <= chunk <= total, "invalid materialization accounting")
    events = {e["tokens"]: e["elapsed_s"] for e in request["events"]}
    require(tail_start in events and output_tokens in events, "missing tail events")
    result = {k: request[k] for k in ("client_e2e_s", "ttft_s", "decode_s", "tpot_ms")}
    result.update(
        tail_tpot_ms=(events[output_tokens] - events[tail_start])
        * 1000
        / (output_tokens - tail_start),
        chunk_materializations=chunk,
        # Includes scheduler prefetch, so this is NOT a decode-demand miss count.
        non_chunk_materializations=total - chunk,
        total_materializations=total,
        materialize_mb=s["expert_materialize_mb_total"],
        h2d_batches=s["expert_cuda_batch_h2d_count"],
        d2h_batches=s["expert_cuda_batch_d2h_count"],
        eviction_d2h_mb=s["expert_eviction_d2h_batched_mb"],
        groups=v["token_chunk_calls"],
        reordered_groups=v["token_chunk_reordered_groups"],
        borrowed_slot_uses=v["borrowed_slot_use_count"],
        kv_to_expert_pages=v["kv_to_expert_pages"],
        expert_to_kv_pages=v["expert_to_kv_pages"],
    )
    require(
        all(math.isfinite(x) and x >= 0 for x in result.values()), "invalid metrics"
    )
    return result


def validate_request(r, plan, index):
    require(
        r["round"] == index + 1 and r["first_request"] == (index == 0), "round mismatch"
    )
    require(r["stream_timing_valid"], "invalid streaming timing")
    s, v = r["final_stats"], r["final_stats"]["shared_vmm"]
    require(
        all(
            s[k] is True for k in ("comparable", "kvc_guard_pass", "expert_guard_pass")
        ),
        "runtime guard failure",
    )
    require(
        all(
            s[k] == 0
            for k in (
                "kvc_stale_entry_count",
                "kvc_page_alignment_violation_count",
                "kvc_physical_failure_count",
                "kvc_host_used_tokens",
                "expert_cuda_batch_pending",
            )
        ),
        "stale/failed/pending transfer",
    )
    require(
        v["ownership_guard_pass"] and v["expert_pointers_stable"],
        "ownership guard failure",
    )
    require(
        all(
            v[k] == 0
            for k in (
                "loan_bytes",
                "blocked_kv_tokens",
                "growth_physical_create_count",
                "token_chunk_profile_pending",
            )
        ),
        "unretired residency",
    )
    require(v["kv_to_expert_pages"] == v["expert_to_kv_pages"], "unreturned loan")
    require(
        v["current_expert_slots"] == plan["expert_initial_slots"], "capacity mismatch"
    )
    h = s["expert_host_budget"]
    require(
        h["optional_guard_pass"]
        and h["ledger_matches"]
        and h["cache_accounting_matches"],
        "host guard failure",
    )
    require(h["batch_owner_storage_bytes"] == 0, "pending host ownership")
    require(
        h["optional_limit_bytes"] == int(plan["expert_host_extra_budget_mb"] * 1024**2),
        "host budget mismatch",
    )
    require(
        h["tracked_unique_storage_bytes"]
        <= h["mandatory_valid_bytes"] + h["optional_limit_bytes"],
        "host budget exceeded",
    )


def aggregate(rows):
    return {
        "sums": {
            k: sum(row[k] for row in rows)
            for k in rows[0]
            if k not in ("tpot_ms", "tail_tpot_ms")
        },
        "median_tpot_ms": statistics.median(row["tpot_ms"] for row in rows),
        "median_tail_tpot_ms": statistics.median(row["tail_tpot_ms"] for row in rows),
    }


def compare(control, candidate):
    roots = (control, candidate)
    plans = [read(root / "plan.json")["args"] for root in roots]
    require(
        all(read(root / "summary.json")["valid"] is True for root in roots),
        "summary guard failure",
    )
    require(
        [p["chunk_order"] for p in plans] == ["input", "reuse"], "expected input/reuse"
    )
    ignored = {"output_dir", "chunk_order"}
    require(
        {k: v for k, v in plans[0].items() if k not in ignored}
        == {k: v for k, v in plans[1].items() if k not in ignored},
        "unmatched plans",
    )
    p = plans[0]
    require(
        p["rounds"] >= 2 and 1 <= p["decode_tail_start"] < p["output_tokens"],
        "invalid request/tail count",
    )
    require(
        not any(
            p[k]
            for k in (
                "profile_chunks",
                "profile_prepare",
                "trace_waits_request",
                "check_cache_accounting",
            )
        ),
        "profilers/debug checks must be off",
    )
    result = {"input": str(control), "reuse": str(candidate), "arms": {}}
    for arm in ("fixed16", "grow17"):
        engines = [read(root / f"{arm}.engine_args.json") for root in roots]
        changed = "layerkv_shared_expert_chunk_order"
        require(
            [e[changed] for e in engines] == ["input", "reuse"], "engine mode mismatch"
        )
        require(
            {k: v for k, v in engines[0].items() if k != changed}
            == {k: v for k, v in engines[1].items() if k != changed},
            "unmatched engine arguments",
        )
        data = [read(root / f"{arm}.json") for root in roots]
        require(
            all(len(d["requests"]) == p["rounds"] for d in data), "incomplete requests"
        )
        rows = []
        for index, requests in enumerate(zip(*(d["requests"] for d in data))):
            for request in requests:
                validate_request(request, p, index)
                out = request["response"]
                require(
                    len(out["output_ids"])
                    == len(out["meta_info"]["output_token_logprobs"])
                    == p["output_tokens"],
                    "incomplete outputs",
                )
                require(
                    all(
                        math.isfinite(x[0])
                        for x in out["meta_info"]["output_token_logprobs"]
                    ),
                    "invalid logprobs",
                )
            a, b = [r["response"] for r in requests]
            require(
                a["output_ids"] == b["output_ids"]
                and a["meta_info"]["output_token_logprobs"]
                == b["meta_info"]["output_token_logprobs"],
                "output/logprob mismatch",
            )
            values = [
                measured(r, p["decode_tail_start"], p["output_tokens"])
                for r in requests
            ]
            require(values[0]["groups"] == values[1]["groups"], "group count mismatch")
            require(values[0]["reordered_groups"] == 0, "input order changed")
            rows.append(
                dict(
                    round=index + 1,
                    input=values[0],
                    reuse=values[1],
                    exact_outputs_and_logprobs=True,
                )
            )
        require(
            sum(r["reuse"]["reordered_groups"] for r in rows) > 0, "reuse not exercised"
        )
        totals = {
            label: {
                "all": aggregate([r[label] for r in rows]),
                "later": aggregate([r[label] for r in rows[1:]]),
                "first": rows[0][label],
                "startup_s": d["engine_startup_s"],
                "final_host_budget": d["requests"][-1]["final_stats"][
                    "expert_host_budget"
                ],
            }
            for label, d in zip(("input", "reuse"), data)
        }
        reductions = {
            scope: {
                k: 100
                * (
                    1
                    - totals["reuse"][scope]["sums"][k]
                    / totals["input"][scope]["sums"][k]
                )
                for k in ("client_e2e_s", "ttft_s", "decode_s")
            }
            for scope in ("all", "later")
        }
        result["arms"][arm] = dict(
            requests=rows, totals=totals, reduction_percent=reductions
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=2,
        type=Path,
        action="append",
        required=True,
        metavar=("INPUT", "REUSE"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"valid": True, "pairs": [compare(*pair) for pair in args.pair]}
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    for pair in result["pairs"]:
        print(pair["input"], pair["reuse"])
        for arm, data in pair["arms"].items():
            print(arm, json.dumps(data["reduction_percent"]))


if __name__ == "__main__":
    main()
