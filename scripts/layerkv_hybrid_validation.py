#!/usr/bin/env python3
"""Compare hybrid-model generation with physical KVC or generic expert offload.

Runs two isolated Engine processes and keeps commands, logs, responses and
guard counters. GDN recurrent state stays in its native pool. Ordinary reclaim
denotes reusable KV slots. --shared-expert-layer additionally verifies actual
physical-page transfer from KV storage into compact expert weights.

The default ``shared`` mode preserves the single-layer SharedVMM experiment.
``--shared-expert-all-layers`` selects the common SharedVMM arena for every
discovered MoE layer; it is the joint KVC/expert residency path, not a generic
expert-offload comparison.
In all-layer mode, ``--expert-extra-slots`` is a global extra-slot budget and
the manager spends it in hotness order; context/admission pressure recalls
loans before allowing any new growth.
``--shared-expert-kv-overflow-tokens`` enables the reverse SharedVMM path for
long-context pressure: physically move only the expert tail pages required by
admission into a sparse KV tail. The runtime selects the smallest page-feasible
expert target; it is not a fixed expert-ID or per-layer eviction set.
``--expert-layer-mode generic`` selects the existing multi-layer expert
planner with ``--shared-expert-layer=-1``; no expert IDs are specified by the
driver. Layer selection and resident IDs come from runtime hotness/planning.
    ``--require-all-expert-layers`` is a coverage diagnostic: it turns
    multi-layer coverage into an acceptance condition, but it is not part of
    the cost-aware optimizer. The optimizer may intentionally leave a cold
    layer at zero evictions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from layerkv_eval_common import parse_layerkv_stats


def engine_kwargs(args):
    generic_expert = getattr(args, "expert_layer_mode", "shared") == "generic"
    shared_all_layers = bool(getattr(args, "shared_expert_all_layers", False))
    shared_expert_enabled = shared_all_layers or args.shared_expert_layer >= 0
    layerkv_expert_enabled = generic_expert or shared_expert_enabled
    return dict(
        model_path=args.model_path,
        dtype=args.dtype,
        base_gpu_id=args.gpu,
        tp_size=1,
        context_length=args.context_length,
        max_total_tokens=args.max_total_tokens,
        max_running_requests=args.batch_size,
        mem_fraction_static=getattr(args, "mem_fraction_static", 0.85),
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        moe_a2a_backend="none",
        moe_runner_backend="triton",
        attention_backend="triton",
        grammar_backend="none",
        random_seed=0,
        log_level="info",
        enable_layerkv=args.case == "offload",
        layerkv_mode="kvc-expert" if layerkv_expert_enabled else "kvc-only",
        layerkv_policy=(
            getattr(args, "expert_policy", "kv-first")
            if generic_expert
            else "expert-first" if shared_expert_enabled else "kv-first"
        ),
        layerkv_kvc_backend="per-layer-arena",
        layerkv_kvc_scheduler="async-deadline",
        layerkv_virtual_scratch_tokens=args.scratch_tokens,
        layerkv_kvc_block_tokens=args.block_tokens,
        layerkv_shared_expert_layer=(
            args.shared_expert_layer
            if args.case == "offload" and not generic_expert
            else -1
        ),
        layerkv_shared_expert_all_layers=(
            shared_all_layers if args.case == "offload" and not generic_expert else False
        ),
        layerkv_shared_expert_initial_slots=args.expert_initial_slots,
        layerkv_shared_expert_extra_slots=args.expert_extra_slots,
        layerkv_shared_expert_kv_overflow_tokens=(
            int(getattr(args, "shared_expert_kv_overflow_tokens", 0))
            if args.case == "offload" and shared_expert_enabled
            else 0
        ),
        layerkv_shared_expert_kv_min_slots=(
            int(getattr(args, "shared_expert_kv_min_slots", 0))
            if args.case == "offload" and shared_expert_enabled
            else 0
        ),
        layerkv_expert_prefetch_lookahead_layers=getattr(
            args, "expert_prefetch_lookahead_layers", 1
        ),
        layerkv_expert_prefetch_min_route_overlap=getattr(
            args, "expert_prefetch_min_route_overlap", 0.5
        ),
        layerkv_reclaim_limit_mb=args.reclaim_mb,
        layerkv_expert_cpu_backing_mode=getattr(
            args, "expert_cpu_backing_mode", "none"
        ),
        layerkv_shared_expert_free_kv_donors=bool(
            args.case == "offload"
            and shared_expert_enabled
            and getattr(args, "shared_expert_free_kv_donors", False)
        ),
        layerkv_shared_expert_retain_across_requests=bool(
            args.case == "offload"
            and shared_expert_enabled
            and getattr(args, "shared_expert_retain_across_requests", False)
        ),
        layerkv_expert_install_layers_per_step=getattr(
            args, "expert_install_layers_per_step", 1
        ),
        layerkv_expert_install_budget_mb=getattr(
            args, "expert_install_budget_mb", 128.0
        ),
        layerkv_expert_copy_force_drain=getattr(args, "expert_copy_force_drain", False),
        layerkv_debug_stats=True,
    )


def run_case(args):
    import sglang as sgl

    engine = sgl.Engine(**engine_kwargs(args))
    try:
        responses = []
        round_input_tokens = args.round_input_tokens or [
            args.input_tokens
        ] * args.rounds
        for round_idx in range(args.rounds):
            # Fixed synthetic IDs; offset rounds to test request-slot reuse.
            inputs = [
                [100 + (i + round_idx * 7 + offset) % 80 for i in range(length)]
                for offset, length in [
                    (13 * j, round_input_tokens[round_idx] + 32 * j)
                    for j in range(args.batch_size)
                ]
            ]
            responses.append(
                engine.generate(
                    input_ids=inputs,
                    sampling_params={
                        "temperature": 0,
                        "max_new_tokens": args.output_tokens,
                        "ignore_eos": True,
                    },
                    return_logprob=True,
                    logprob_start_len=-1,
                )
            )
        (args.output_dir / f"{args.case}.responses.json").write_text(
            json.dumps(responses, indent=2)
        )
        (args.output_dir / f"{args.case}.server_info.json").write_text(
            json.dumps(engine.get_server_info(), indent=2, default=str)
        )
    finally:
        engine.shutdown()


def _expert_eviction_map(stats):
    """Return the applied per-layer eviction counts from runtime stats."""

    value = stats.get("applied_expert_evictions_by_layer")
    if not value:
        value = stats.get("selected_expert_evictions_by_layer")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    if not isinstance(value, dict):
        return {}
    result = {}
    for layer_id, count in value.items():
        try:
            result[str(layer_id)] = int(count)
        except (TypeError, ValueError):
            continue
    return result


def _expert_layer_coverage(stats):
    """Summarize discovered layers and layers with a positive eviction."""

    discovered = int(
        stats.get(
            "layerkv_expert_discovered_layer_count",
            stats.get("layerkv_expert_layer_count", 0),
        )
        or 0
    )
    installed = int(
        stats.get(
            "layerkv_expert_installed_layer_count",
            stats.get("expert_install_completed_layers", 0),
        )
        or 0
    )
    pending = int(stats.get("expert_install_pending_layers", 0) or 0)
    evictions = _expert_eviction_map(stats)
    evicted_layers = sorted(
        layer_id for layer_id, count in evictions.items() if int(count) > 0
    )
    return {
        "discovered_layer_count": discovered,
        "installed_layer_count": installed,
        "pending_install_layers": pending,
        "evicted_layer_count": len(evicted_layers),
        "evicted_layer_ids": evicted_layers,
        "all_layers_evicted": bool(
            discovered > 0
            and installed == discovered
            and pending == 0
            and len(evicted_layers) == discovered
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--reclaim-mb", type=float, default=0.5)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--logprob-atol", type=float, default=0.01)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.85,
        help="Fraction of GPU memory reserved for the SGLang static pool.",
    )
    parser.add_argument("--scratch-tokens", type=int, default=512)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--shared-expert-layer", type=int, default=-1)
    parser.add_argument(
        "--shared-expert-all-layers",
        action="store_true",
        help="Use one SharedVMM page budget across every discovered MoE layer.",
    )
    parser.add_argument(
        "--shared-expert-free-kv-donors",
        action="store_true",
        help=(
            "Allow complete currently-free per-layer KV pages to fund SharedVMM "
            "expert growth; required for short-context pressure probes without "
            "waiting for a prior KVC offload."
        ),
    )
    parser.add_argument(
        "--shared-expert-retain-across-requests",
        action="store_true",
        help=(
            "Keep SharedVMM expert loans across rounds so a later pressure round "
            "can measure admission-time recall; requires free KV donors."
        ),
    )
    parser.add_argument(
        "--shared-expert-kv-overflow-tokens",
        type=int,
        default=0,
        help=(
            "Enable the long-context SharedVMM pressure path with this many "
            "temporary KV tokens funded by expert tail pages; 0 disables it."
        ),
    )
    parser.add_argument(
        "--shared-expert-kv-min-slots",
        type=int,
        default=0,
        help=(
            "Minimum resident expert slots while KV overflow is active; 0 "
            "uses the page-feasible top-k floor."
        ),
    )
    parser.add_argument(
        "--round-input-tokens",
        type=int,
        nargs="+",
        help=(
            "Per-round input lengths. When supplied, the list length must equal "
            "--rounds; useful for a short-context warmup followed by a pressure round."
        ),
    )
    parser.add_argument(
        "--expert-layer-mode",
        choices=["shared", "generic"],
        default="shared",
        help=(
            "shared uses the single-layer SharedVMM path; generic uses the "
            "multi-layer planner with no fixed expert layer or expert ID set."
        ),
    )
    parser.add_argument(
        "--require-all-expert-layers",
        action="store_true",
        help=(
            "Diagnostic only: in generic mode, require every discovered MoE "
            "layer to have a positive applied expert eviction. The optimized "
            "planner may intentionally leave cold layers at zero."
        ),
    )
    parser.add_argument(
        "--expert-policy",
        choices=[
            "kv-first",
            "ratio-25-75",
            "ratio-50-50",
            "ratio-75-25",
            "layer-aware-joint-dp",
            "coresid",
        ],
        default="kv-first",
        help="Policy used by generic multi-layer expert mode.",
    )
    parser.add_argument(
        "--expert-cpu-backing-mode",
        choices=["none", "all", "selected"],
        default="none",
        help="CPU backing mode for generic expert mode; none is sparse and recommended.",
    )
    parser.add_argument(
        "--expert-install-layers-per-step",
        type=int,
        default=1,
        help="Generic planner install concurrency; raise only for a bounded diagnostic.",
    )
    parser.add_argument(
        "--expert-install-budget-mb",
        type=float,
        default=128.0,
        help="Generic planner per-step install budget in MB.",
    )
    parser.add_argument(
        "--expert-copy-force-drain",
        action="store_true",
        help=(
            "For validation only, synchronously drain pending generic expert "
            "D2H/install work at each decode safe point. This changes timing."
        ),
    )
    parser.add_argument("--expert-initial-slots", type=int, default=16)
    parser.add_argument("--expert-extra-slots", type=int, default=1)
    parser.add_argument(
        "--expert-prefetch-lookahead-layers",
        type=int,
        default=1,
        help="Maximum number of predicted expert layers queued for H2D prefetch at one decode safe point.",
    )
    parser.add_argument(
        "--expert-prefetch-min-route-overlap",
        type=float,
        default=0.5,
        help="Minimum consecutive decode-route overlap required for previous-route expert H2D prefetch.",
    )
    parser.add_argument(
        "--case", choices=["baseline", "offload"], help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.expert_layer_mode == "generic" and (
        args.shared_expert_layer >= 0 or args.shared_expert_all_layers
    ):
        parser.error(
            "generic expert mode requires no SharedVMM layer selection"
        )
    if args.expert_layer_mode == "generic" and args.shared_expert_free_kv_donors:
        parser.error("free KV donors require SharedVMM expert mode")
    if args.expert_layer_mode == "generic" and args.shared_expert_retain_across_requests:
        parser.error("cross-round SharedVMM retention requires SharedVMM expert mode")
    if args.expert_layer_mode == "generic" and args.shared_expert_kv_overflow_tokens:
        parser.error("KV overflow requires SharedVMM expert mode")
    if args.expert_layer_mode == "generic" and args.shared_expert_kv_min_slots:
        parser.error("KV overflow minimum slots require SharedVMM expert mode")
    if (
        args.shared_expert_kv_overflow_tokens
        and not (args.shared_expert_layer >= 0 or args.shared_expert_all_layers)
    ):
        parser.error(
            "shared-expert-kv-overflow-tokens requires --shared-expert-layer or "
            "--shared-expert-all-layers"
        )
    if args.shared_expert_kv_min_slots and not (
        args.shared_expert_layer >= 0 or args.shared_expert_all_layers
    ):
        parser.error(
            "shared-expert-kv-min-slots requires --shared-expert-layer or "
            "--shared-expert-all-layers"
        )
    if args.shared_expert_kv_min_slots < 0:
        parser.error("shared-expert-kv-min-slots must be nonnegative")
    if args.shared_expert_kv_overflow_tokens < 0:
        parser.error("shared-expert-kv-overflow-tokens must be nonnegative")
    if args.shared_expert_kv_min_slots > args.expert_initial_slots:
        parser.error("shared-expert-kv-min-slots cannot exceed expert-initial-slots")
    if args.shared_expert_kv_min_slots and not args.shared_expert_kv_overflow_tokens:
        parser.error(
            "shared-expert-kv-min-slots requires shared-expert-kv-overflow-tokens"
        )
    if args.shared_expert_retain_across_requests and not args.shared_expert_free_kv_donors:
        parser.error("cross-round retention requires --shared-expert-free-kv-donors")
    if args.round_input_tokens is not None and len(args.round_input_tokens) != args.rounds:
        parser.error("round-input-tokens must contain exactly one value per round")
    if args.require_all_expert_layers and args.expert_layer_mode != "generic":
        parser.error("--require-all-expert-layers requires --expert-layer-mode generic")
    if args.expert_layer_mode == "shared" and args.expert_cpu_backing_mode == "all":
        parser.error("shared expert mode does not support all-layer CPU backing")
    if args.expert_cpu_backing_mode == "selected" and (
        args.shared_expert_layer < 0 or args.shared_expert_all_layers
    ):
        parser.error("selected CPU backing requires a nonnegative shared expert layer")
    if any(
        getattr(args, name) <= 0
        for name in [
            "input_tokens",
            "output_tokens",
            "batch_size",
            "rounds",
            "context_length",
            "max_total_tokens",
            "block_tokens",
            "expert_initial_slots",
            "expert_extra_slots",
            "expert_prefetch_lookahead_layers",
            "expert_install_layers_per_step",
        ]
    ):
        parser.error("token, batch, round and slot counts must be positive")
    if args.scratch_tokens < 0:
        parser.error("scratch-tokens must be nonnegative")
    if not 0.0 <= args.expert_prefetch_min_route_overlap <= 1.0:
        parser.error("expert-prefetch-min-route-overlap must be in [0.0, 1.0]")
    round_input_tokens = args.round_input_tokens or [args.input_tokens] * args.rounds
    if any(value <= 0 for value in round_input_tokens):
        parser.error("round-input-tokens must be positive")
    if any(
        value + 32 * (args.batch_size - 1) + args.output_tokens
        > args.context_length
        for value in round_input_tokens
    ):
        parser.error("input plus output must fit context-length")
    if (
        args.reclaim_mb <= 0
        or args.expert_install_budget_mb < 0
        or args.gpu < 0
        or args.timeout_s <= 0
        or args.logprob_atol < 0
        or not math.isfinite(args.reclaim_mb)
        or not math.isfinite(args.expert_install_budget_mb)
        or not math.isfinite(args.logprob_atol)
        or not 0.0 < args.mem_fraction_static <= 1.0
    ):
        parser.error(
            "require reclaim-mb > 0, gpu >= 0, timeout-s > 0, logprob-atol >= 0, and 0 < mem-fraction-static <= 1"
        )
    if args.case:
        run_case(args)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"args": {**vars(args), "output_dir": str(args.output_dir)}, "runs": {}}
    for case in ["baseline", "offload"]:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--case",
            case,
        ]
        print(f"Running {case}", flush=True)
        log_path = args.output_dir / f"{case}.log"
        with log_path.open("w") as log:
            try:
                proc = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                returncode = proc.wait(timeout=args.timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                returncode = 124
        stats = parse_layerkv_stats(log_path.read_text())
        summary["runs"][case] = {
            "command": cmd,
            "returncode": returncode,
            "stats": stats[-1] if stats else {},
            "stats_count": len(stats),
        }
        info_path = args.output_dir / f"{case}.server_info.json"
        if returncode == 0 and info_path.exists():
            internal = json.loads(info_path.read_text())["internal_states"]
            if internal and "layerkv" in internal[0]:
                summary["runs"][case]["stats"] = internal[0]["layerkv"]
        print(f"{case}: returncode={returncode}, stats={len(stats)}", flush=True)
        if returncode:
            break
    valid = len(summary["runs"]) == 2 and all(
        r["returncode"] == 0 for r in summary["runs"].values()
    )
    if valid:
        baseline = json.loads((args.output_dir / "baseline.responses.json").read_text())
        offload = json.loads((args.output_dir / "offload.responses.json").read_text())
        pairs = [
            (
                a["meta_info"]["output_token_logprobs"],
                b["meta_info"]["output_token_logprobs"],
            )
            for ar, br in zip(baseline, offload)
            for a, b in zip(ar, br)
        ]
        exact = (
            len(baseline) == len(offload) == args.rounds
            and all(len(r) == args.batch_size for r in baseline + offload)
            and len(pairs) == args.rounds * args.batch_size
            and all(
                len(a) == len(b) == args.output_tokens
                and [x[1] for x in a] == [x[1] for x in b]
                for a, b in pairs
            )
        )
        deltas = [abs(x[0] - y[0]) for a, b in pairs for x, y in zip(a, b)]
        max_delta = max(deltas, default=float("inf"))
        stats = summary["runs"]["offload"]["stats"]
        guards = all(
            stats.get(k) is True
            for k in [
                "kvc_guard_pass",
                "layerkv_physical_kvc_supported",
                "layerkv_kvc_backend_ready",
            ]
        )
        errors = all(
            stats.get(k) == 0
            for k in [
                "kvc_stale_entry_count",
                "kvc_physical_failure_count",
                "kvc_page_alignment_violation_count",
            ]
        )
        kvc_exercised = all(
            stats.get(k, 0) > 0
            for k in [
                "kvc_evict_count_total",
                "kvc_reload_count_total",
                "physical_kvc_reclaim_peak_mb",
            ]
        )
        generic_coverage = _expert_layer_coverage(stats)
        generic_expert_exercised = (
            args.expert_layer_mode == "generic"
            and stats.get("expert_guard_pass") is True
            and stats.get("layerkv_physical_expert_supported") is True
            and stats.get("layerkv_expert_layer_count", 0) > 1
            and stats.get("physical_expert_reclaim_mb", 0) > 0
            and (
                not args.require_all_expert_layers
                or generic_coverage["all_layers_evicted"]
            )
        )
        shared_overflow_exercised = (
            args.expert_layer_mode == "shared"
            and args.shared_expert_kv_overflow_tokens > 0
            and (
                stats.get("shared_expert_admission_overflow_count", 0) > 0
                or stats.get("shared_expert_admission_overflow_tokens", 0) > 0
                or stats.get("shared_vmm", {}).get(
                    "kv_overflow_activation_count", 0
                )
                > 0
            )
        )
        exercised = (
            generic_expert_exercised
            if args.expert_layer_mode == "generic"
            else kvc_exercised or shared_overflow_exercised
        )
        arena_capacity = stats.get("kvc_per_layer_physical_arena_token_capacity", -1)
        arena_accounting = arena_capacity > 0 and all(
            0 <= stats.get(k, -1) <= arena_capacity
            for k in [
                "kvc_per_layer_physical_arena_min_free_tokens",
                "kvc_per_layer_physical_arena_common_free_tokens",
            ]
        )
        host_released = stats.get("kvc_host_used_tokens") == 0
        summary.update(
            exact_output_tokens=exact,
            max_output_logprob_delta=max_delta,
            guards_pass=guards,
            no_kvc_errors=errors,
            physical_offload_exercised=exercised,
            kvc_physical_offload_exercised=kvc_exercised,
            arena_accounting_pass=arena_accounting,
            host_backing_released=host_released,
            shared_vmm_overflow_exercised=shared_overflow_exercised,
        )
        valid = (
            exact
            and all(math.isfinite(d) for d in deltas)
            and max_delta <= args.logprob_atol
            and guards
            and errors
            and exercised
            and arena_accounting
            and host_released
        )
        if args.expert_layer_mode == "generic":
            summary["generic_expert_pass"] = generic_expert_exercised
            summary["generic_expert_layer_coverage"] = generic_coverage
            summary["generic_expert_layer_count"] = stats.get(
                "layerkv_expert_layer_count", 0
            )
            summary["generic_cross_layer_prefetch_issues"] = stats.get(
                "expert_prefetch_cross_layer_issue_count", 0
            )
            summary["generic_cross_layer_prefetch_layers"] = stats.get(
                "expert_prefetch_cross_layer_layer_count", 0
            )
        if args.shared_expert_layer >= 0 or args.shared_expert_all_layers:
            shared = stats.get("shared_vmm", {})
            expected_initial_slots = (
                args.expert_initial_slots
                * max(1, int(shared.get("layer_count", 1) or 1))
                if args.shared_expert_all_layers
                else args.expert_initial_slots
            )
            shared_common_pass = (
                shared.get("ownership_guard_pass") is True
                and shared.get("expert_pointers_stable") is True
                and shared.get("loan_bytes") == 0
                and shared.get("blocked_kv_tokens") == 0
                and shared.get("growth_physical_create_count") == 0
                and stats.get("expert_guard_pass") is True
                and stats.get("expert_materialize_count", 0) > 0
            )
            if args.shared_expert_kv_overflow_tokens > 0:
                shared_pass = shared_common_pass and (
                    shared.get("kv_overflow_activation_count", 0) > 0
                    and shared.get("kv_overflow_expert_slots_reclaimed", 0) > 0
                )
            else:
                shared_pass = shared_common_pass and (
                    shared.get("kv_to_expert_pages", 0) > 0
                    and shared.get("expert_to_kv_pages")
                    == shared.get("kv_to_expert_pages")
                    and shared.get("peak_expert_slots", 0) > expected_initial_slots
                    and shared.get("current_expert_slots") == expected_initial_slots
                    and shared.get("borrowed_slot_use_count", 0) > 0
                )
            summary["shared_vmm_pass"] = shared_pass
            valid = valid and shared_pass
    summary["valid"] = valid
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"valid={valid}, summary={path}", flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
