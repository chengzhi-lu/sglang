"""Concurrent measurements use batch makespan, not summed request latency."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_scripts = Path(__file__).resolve().parents[4] / "scripts"
_spec = importlib.util.spec_from_file_location(
    "concurrent_perf", _scripts / "layerkv_concurrent_perf.py"
)
comparison = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(_scripts))
try:
    _spec.loader.exec_module(comparison)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("key_type", [int, str])
def test_round_metrics_uses_actual_batch_and_makespan(key_type):
    before = dict(
        observed_decode_forward_count=10,
        observed_decode_request_steps=10,
        observed_decode_batch_histogram={key_type(1): 10},
    )
    after = dict(
        observed_decode_forward_count=12,
        observed_decode_request_steps=14,
        observed_decode_batch_histogram={key_type(1): 10, key_type(2): 2},
    )
    events = [
        [dict(elapsed_s=1, tokens=1), dict(elapsed_s=3, tokens=3)],
        [dict(elapsed_s=2, tokens=1), dict(elapsed_s=4, tokens=3)],
    ]
    row = comparison.round_metrics(events, 4, before, after)
    assert row["output_tokens_per_s"] == 1.5
    assert row["ttft_s"] == [1, 2]
    assert row["tpot_ms"] == [1000, 1000]
    assert row["actual_decode_batch_mean"] == 2
    assert row["actual_decode_batch_histogram"] == {"1": 0, "2": 2}


def test_missing_stream_is_not_zero_latency():
    with pytest.raises(ValueError, match="missing request"):
        comparison.round_metrics([[]], 1, {}, {})


def mixed_args(**overrides):
    return SimpleNamespace(
        **{
            "rounds": 3,
            "context_length": 8192,
            "round_workloads": [
                {"requests": 8, "input_tokens": 128, "output_tokens": 64},
                {
                    "requests": 2,
                    "input_tokens": 128,
                    "output_tokens": 64,
                    "late_input_tokens": 6000,
                    "late_after_seconds": 1,
                },
                {"requests": 8, "input_tokens": 128, "output_tokens": 64},
            ],
            **overrides,
        }
    )


def test_mixed_rounds_reset_arrivals_without_mutating_base():
    args = mixed_args(late_after_seconds=99)
    comparison.validate_round_workloads(args)
    phases = [comparison.effective_round_args(args, i) for i in range(3)]
    assert [p.requests for p in phases] == [8, 2, 8]
    assert [p.late_after_seconds for p in phases] == [0, 1, 0]
    assert [p.late_input_tokens for p in phases] == [128, 6000, 128]
    assert args.late_after_seconds == 99


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"requests": True},
        {"requests": 0},
        {"output_tokens": 1},
        {"input_tokens": 8192},
        {"late_after_seconds": float("nan")},
        {"late_after_seconds": 1, "late_after_output_tokens": 1},
        {"late_after_seconds": 1, "initial_requests": 8},
        {"late_input_tokens": 500},
        {"model_path": "different"},
    ],
)
def test_mixed_rounds_reject_invalid_workload(spec):
    args = mixed_args()
    args.round_workloads[0] = {} if not spec else {**args.round_workloads[0], **spec}
    with pytest.raises(ValueError):
        comparison.validate_round_workloads(args)


def test_mixed_summary_uses_phase_counts_and_rejects_metadata_mismatch(tmp_path):
    import copy

    test_summary_rejects_invalid_comparisons(tmp_path, None, False, False)
    args = mixed_args(
        rounds=2,
        requests=99,
        output_tokens=99,
        output_dir=tmp_path,
        logprob_atol=0.01,
        expert_initial_slots=16,
        expert_extra_slots=4,
        round_workloads=[
            {"requests": 1, "input_tokens": 4, "output_tokens": 3},
            {"requests": 2, "input_tokens": 4, "output_tokens": 3},
        ],
    )
    comparison.validate_round_workloads(args)
    for arm in ("fixed", "lend"):
        path = tmp_path / f"{arm}.json"
        data = json.loads(path.read_text())
        first = data["rounds"][0]
        first["workload"] = comparison.round_workload(
            comparison.effective_round_args(args, 0)
        )
        second = copy.deepcopy(first)
        second.update(
            round=2,
            first_round=False,
            workload=comparison.round_workload(
                comparison.effective_round_args(args, 1)
            ),
        )
        second["responses"] *= 2
        second["events"] *= 2
        second["final_stats"].update(
            observed_decode_forward_count=4,
            observed_decode_request_steps=6,
            observed_decode_batch_histogram={"1": 2, "2": 2},
        )
        data["rounds"].append(second)
        path.write_text(json.dumps(data))
    assert comparison.summarize(args)["valid"]
    args.round_workloads[1]["input_tokens"] = 5
    assert not comparison.summarize(args)["valid"]


def test_native_graph_option_is_symmetric_and_native_reference_stays_eager(monkeypatch):
    monkeypatch.setattr(comparison, "engine_kwargs", lambda args: {})
    args = SimpleNamespace(
        max_running_requests=8,
        expert_extra_slots=48,
        expert_backing_cache_mb=128,
        expert_cpu_backing_pool_mb=128,
        native_moe_graph_max_batch_size=8,
    )
    for arm in ("fixed", "lend"):
        assert (
            comparison.effective_engine_args(args, arm)[
                "layerkv_native_moe_graph_max_batch_size"
            ]
            == 8
        )
    assert (
        "layerkv_native_moe_graph_max_batch_size"
        not in comparison.effective_engine_args(args, "native")
    )


def test_expert_chunk_order_is_explicit_and_symmetric(monkeypatch):
    monkeypatch.setattr(comparison, "engine_kwargs", lambda args: {})
    args = SimpleNamespace(
        max_running_requests=8,
        expert_extra_slots=48,
        expert_backing_cache_mb=128,
        expert_cpu_backing_pool_mb=128,
        expert_chunk_order="reuse",
    )
    for arm in ("fixed", "lend"):
        assert (
            comparison.effective_engine_args(args, arm)[
                "layerkv_shared_expert_chunk_order"
            ]
            == "reuse"
        )


def test_bounded_profile_configuration_is_explicit(tmp_path):
    args = SimpleNamespace(
        profile_round=3,
        profile_steps=8,
        rounds=3,
        output_tokens=64,
        output_dir=tmp_path,
        arm="lend",
    )
    config = comparison.profile_kwargs(args)
    assert config["num_steps"] == 8 and config["profile_by_stage"]
    assert config["output_dir"] == str(tmp_path / "lend.trace")
    assert config["with_stack"] is False and config["record_shapes"] is False
    args.profile_round = 0
    assert comparison.profile_kwargs(args) is None


@pytest.mark.parametrize("selected,steps", [(4, 8), (-1, 8), (3, 0), (3, 63)])
def test_profile_rejects_unbounded_or_unflushable_capture(selected, steps):
    with pytest.raises(ValueError):
        comparison.profile_kwargs(
            SimpleNamespace(
                profile_round=selected, profile_steps=steps, rounds=3, output_tokens=64
            )
        )


def test_coalesced_stream_marks_timing_invalid():
    stats = dict(
        observed_decode_forward_count=0,
        observed_decode_request_steps=0,
        observed_decode_batch_histogram={},
    )
    row = comparison.round_metrics([[dict(elapsed_s=1, tokens=3)]], 1, stats, stats)
    assert row["stream_timing_valid"] is False
    assert row["tpot_ms"] == [None]
    assert row["actual_decode_batch_mean"] is None


def test_late_request_ttft_starts_at_its_own_submission():
    stats = dict(
        observed_decode_forward_count=0,
        observed_decode_request_steps=0,
        observed_decode_batch_histogram={},
    )
    events = [[dict(tokens=1, elapsed_s=4), dict(tokens=3, elapsed_s=6)]]
    assert comparison.round_metrics(events, 6, stats, stats, [3])["ttft_s"] == [1]
    with pytest.raises(ValueError, match="submission"):
        comparison.round_metrics(events, 6, stats, stats, [5])


@pytest.mark.parametrize("early_failure", [False, True])
def test_staggered_stream_maps_wave_indices_and_propagates_failure(early_failure):
    class Engine:
        def __init__(self):
            self.progress = 0
            self.second_started_at = None

        async def async_generate(self, *, input_ids, **_kwargs):
            initial = input_ids == [[1]]
            if not initial:
                self.second_started_at = self.progress

            async def stream():
                if initial and early_failure:
                    raise RuntimeError("initial failed")
                for count in range(1, 9):
                    if initial:
                        self.progress = count
                    yield dict(
                        index=0,
                        meta_info=dict(completion_tokens=count),
                        marker=input_ids[0][0],
                    )
                    await asyncio.sleep(0)

            return stream()

    engine = Engine()
    args = SimpleNamespace(
        initial_requests=1, late_after_output_tokens=2, output_tokens=8
    )

    async def run():
        return await asyncio.wait_for(
            comparison.staggered_round(engine, args, [[1], [2]]), 1
        )

    if early_failure:
        with pytest.raises(RuntimeError, match="initial failed"):
            asyncio.run(run())
        assert engine.second_started_at is None
    else:
        events, final, elapsed, submitted = asyncio.run(run())
        assert engine.second_started_at >= 2
        assert [item["marker"] for item in final] == [1, 2]
        assert all(row[-1]["tokens"] == 8 for row in events)
        assert 0 <= submitted[0] <= submitted[1] <= elapsed


def test_native_reference_removes_layerkv_switches(monkeypatch):
    monkeypatch.setattr(
        comparison,
        "engine_kwargs",
        lambda _: {
            "dtype": "bfloat16",
            "enable_layerkv": True,
            "layerkv_shared_expert_layer": 0,
        },
    )
    args = SimpleNamespace(
        max_running_requests=2,
        expert_extra_slots=1,
        expert_backing_cache_mb=128,
        expert_cpu_backing_pool_mb=128,
    )
    kwargs = comparison.effective_engine_args(args, "native")
    assert not kwargs["enable_layerkv"]
    assert not any(key.startswith("layerkv_") for key in kwargs)
    assert kwargs["dtype"] == "bfloat16"


def test_round_trigger_replay_keeps_base_arguments_unchanged():
    args = SimpleNamespace(
        round_output_triggers=[0, 8, 13],
        late_after_seconds=1.0,
        late_after_output_tokens=0,
    )
    rounds = [comparison.effective_round_args(args, index) for index in range(3)]
    assert [r.late_after_seconds for r in rounds] == [1.0, 0, 0]
    assert [r.late_after_output_tokens for r in rounds] == [0, 8, 13]
    assert args.late_after_seconds == 1.0 and args.late_after_output_tokens == 0


@pytest.mark.parametrize("triggers", [["8"], ["-1", "8"], ["16", "8"]])
def test_invalid_round_trigger_replay_fails_before_launch(
    monkeypatch, tmp_path, triggers
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "driver",
            "--model-path",
            "unused",
            "--output-dir",
            str(tmp_path / "unused"),
            "--round-output-triggers",
            *triggers,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        comparison.main()
    assert exc.value.code == 2
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("initial_finishes", [False, True])
def test_fixed_time_arrival_does_not_wait_for_output_progress(
    failure, initial_finishes
):
    class Engine:
        late_started = False

        async def async_generate(self, *, input_ids, **kwargs):
            initial = input_ids == [[1]]
            if not initial:
                self.late_started = True

            async def stream():
                if initial:
                    if failure:
                        raise RuntimeError("initial failed")
                    # Initial stream cannot produce output until late submission.
                    while not self.late_started and not initial_finishes:
                        await asyncio.sleep(0)
                yield dict(index=0, meta_info=dict(completion_tokens=1))

            return stream()

    engine = Engine()
    args = SimpleNamespace(
        initial_requests=1,
        late_after_output_tokens=0,
        late_after_seconds=0.01,
        output_tokens=1,
    )

    async def run():
        return await asyncio.wait_for(
            comparison.staggered_round(engine, args, [[1], [2]]), 1
        )

    if failure:
        with pytest.raises(RuntimeError, match="initial failed"):
            asyncio.run(run())
        assert not engine.late_started
    else:
        events, _, elapsed, submitted = asyncio.run(run())
        # Event-loop clock resolution can wake a timer slightly early.
        assert submitted[1] >= args.late_after_seconds - 0.001
        assert events[0][0]["elapsed_s"] <= elapsed
        assert (events[0][0]["elapsed_s"] < submitted[1]) == initial_finishes


@pytest.mark.parametrize(
    "extra",
    [
        ["--late-after-seconds", "nan"],
        ["--late-after-seconds", "-1"],
        ["--late-after-seconds", "1", "--late-after-output-tokens", "2"],
    ],
)
def test_invalid_time_arrival_cli_fails_before_engine_start(
    monkeypatch, tmp_path, extra
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "driver",
            "--model-path",
            "unused",
            "--output-dir",
            str(tmp_path / "unused"),
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        comparison.main()
    assert exc.value.code == 2
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize(
    "retained,heavy", [(False, False), (True, True), (True, False)]
)
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "budget",
        "output",
        "timing",
        "decode_count",
        "loans",
        "policy",
        "profile",
        "graph",
        "graph_artifact",
        "chunks",
    ],
)
def test_summary_rejects_invalid_comparisons(tmp_path, fault, retained, heavy):
    for arm in ("fixed", "lend"):
        before = dict(
            observed_decode_forward_count=0,
            observed_decode_request_steps=0,
            observed_decode_batch_histogram={},
            shared_vmm={"physical_bytes": 100},
        )
        if fault == "graph_artifact":
            before["native_moe_graph"] = {"max_batch_size": 8}
        after = dict(
            observed_decode_forward_count=2,
            observed_decode_request_steps=2,
            observed_decode_batch_histogram={"1": 2},
            kvc_guard_pass=True,
            expert_guard_pass=True,
            kvc_stale_entry_count=0,
            kvc_page_alignment_violation_count=0,
            kvc_physical_failure_count=0,
            shared_vmm=dict(
                physical_bytes=100,
                ownership_guard_pass=True,
                expert_pointers_stable=True,
                loan_bytes=4096 if retained and (heavy or arm == "lend") else 0,
                blocked_kv_tokens=2 if retained and (heavy or arm == "lend") else 0,
                retain_across_requests=retained,
                current_expert_slots=20 if heavy or arm == "lend" else 16,
                admission_policy="retain" if heavy and arm == "fixed" else "recall",
                growth_physical_create_count=0,
            ),
        )
        events = [[dict(tokens=1, elapsed_s=1), dict(tokens=3, elapsed_s=3)]]
        output = [[-0.5, 42, None] for _ in range(3)]
        if arm == "lend":
            if fault == "budget":
                after["shared_vmm"]["physical_bytes"] += 1
            elif fault == "output":
                output[0][1] += 1
            elif fault == "timing":
                events[0][0]["tokens"] = 2
            elif fault == "decode_count":
                after["observed_decode_request_steps"] += 1
            elif fault == "loans":
                after["shared_vmm"]["loan_bytes"] = 0 if retained else 4096
            elif fault == "policy":
                if retained:
                    after["shared_vmm"]["admission_policy"] = "retain"
                else:
                    after["shared_vmm"]["growth_physical_create_count"] = 1
        row = dict(
            round=1,
            first_round=True,
            events=events,
            client_e2e_s=3,
            final_stats=after,
            responses=[dict(meta_info=dict(output_token_logprobs=output))],
        )
        (tmp_path / f"{arm}.json").write_text(
            json.dumps(
                dict(
                    initial_stats=before,
                    rounds=[row],
                    diagnostic_only=fault == "chunks",
                )
            )
        )
    result = comparison.summarize(
        SimpleNamespace(
            output_dir=tmp_path,
            rounds=1,
            requests=1,
            output_tokens=3,
            logprob_atol=0.01,
            baseline_split="expert-heavy" if heavy else "kv-heavy",
            retain_experts_across_requests=retained,
            expert_initial_slots=16,
            expert_extra_slots=4,
            profile_round=1 if fault == "profile" else 0,
            native_moe_graph_max_batch_size=8 if fault == "graph" else 0,
        )
    )
    assert result["valid"] is (fault is None)


@pytest.mark.parametrize("heavy", [False, True])
@pytest.mark.parametrize("retain", [False, True])
def test_fixed_split_and_adaptive_arm_have_explicit_retention(
    monkeypatch, heavy, retain
):
    monkeypatch.setattr(
        comparison, "engine_kwargs", lambda args: {"extra": args.expert_extra_slots}
    )
    args = SimpleNamespace(
        max_running_requests=2,
        expert_extra_slots=4,
        expert_backing_cache_mb=128,
        expert_cpu_backing_pool_mb=128,
        expert_policy="adaptive",
        baseline_split="expert-heavy" if heavy else "kv-heavy",
        retain_experts_across_requests=retain,
    )
    fixed = comparison.effective_engine_args(args, "fixed")
    adaptive = comparison.effective_engine_args(args, "lend")
    assert fixed["extra"] == (4 if heavy else 0)
    assert adaptive["extra"] == 4
    assert fixed["layerkv_shared_expert_policy"] == "fixed"
    assert adaptive["layerkv_shared_expert_policy"] == "adaptive"
    for row in (fixed, adaptive):
        assert row["layerkv_shared_expert_retain_across_requests"] == (heavy or retain)
    assert fixed["layerkv_shared_expert_admission_policy"] == (
        "retain" if heavy else "recall"
    )
    assert adaptive["layerkv_shared_expert_admission_policy"] == "recall"


def test_expert_heavy_fixed_can_be_preconditioned_without_changing_lend(
    monkeypatch,
):
    monkeypatch.setattr(
        comparison,
        "engine_kwargs",
        lambda args: {
            "initial": args.expert_initial_slots,
            "extra": args.expert_extra_slots,
        },
    )
    args = SimpleNamespace(
        max_running_requests=8,
        expert_initial_slots=16,
        expert_extra_slots=48,
        expert_backing_cache_mb=128,
        expert_cpu_backing_pool_mb=128,
        expert_policy="adaptive",
        baseline_split="expert-heavy",
        retain_experts_across_requests=True,
        precondition_expert_heavy_fixed=True,
    )
    fixed = comparison.effective_engine_args(args, "fixed")
    lend = comparison.effective_engine_args(args, "lend")
    assert fixed["initial"] == 16 and fixed["extra"] == 48
    assert lend["initial"] == 16 and lend["extra"] == 48


def test_expert_heavy_precondition_requires_physical_loan_and_reaches_target():
    class FakeEngine:
        def __init__(self, shared):
            self.shared = shared
            self.generated = False

        def generate(self, **_kwargs):
            self.generated = True

        def get_server_info(self):
            return {"internal_states": [{"layerkv": {"shared_vmm": self.shared}}]}

    args = SimpleNamespace(
        arm="fixed",
        baseline_split="expert-heavy",
        precondition_expert_heavy_fixed=True,
        precondition_requests=2,
        precondition_input_tokens=8,
        precondition_output_tokens=8,
        requests=4,
        input_tokens=16,
        output_tokens=4,
        expert_initial_slots=16,
        expert_extra_slots=4,
    )
    shared = {
        "current_expert_slots": 20,
        "loan_bytes": 4096,
        "growth_physical_create_count": 0,
        "physical_bytes": 100,
        "kv_to_expert_pages": 2,
    }
    engine = FakeEngine(shared)
    result = comparison.precondition_expert_heavy_fixed(engine, args)
    assert engine.generated
    assert result["target_slots"] == 20
    assert result["loan_bytes"] == 4096


def test_expert_heavy_precondition_retries_fixed_until_target():
    class FakeEngine:
        def __init__(self):
            self.calls = 0

        def generate(self, **_kwargs):
            self.calls += 1

        def get_server_info(self):
            current = 18 if self.calls == 1 else 20
            return {
                "internal_states": [
                    {
                        "layerkv": {
                            "shared_vmm": {
                                "current_expert_slots": current,
                                "loan_bytes": 4096,
                                "growth_physical_create_count": 0,
                                "physical_bytes": 100,
                            }
                        }
                    }
                ]
            }

    args = SimpleNamespace(
        arm="fixed",
        baseline_split="expert-heavy",
        precondition_expert_heavy_fixed=True,
        precondition_rounds=3,
        precondition_requests=2,
        precondition_input_tokens=8,
        precondition_output_tokens=8,
        requests=4,
        input_tokens=16,
        output_tokens=4,
        expert_initial_slots=16,
        expert_extra_slots=4,
    )
    engine = FakeEngine()
    result = comparison.precondition_expert_heavy_comparator(engine, args)
    assert engine.calls == 2
    assert result["attempts"] == 2
    assert result["current_slots"] == 20


def test_expert_heavy_precondition_installs_only_adaptive_base_slots():
    class FakeEngine:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs

        def get_server_info(self):
            return {
                "internal_states": [
                    {
                        "layerkv": {
                            "shared_vmm": {
                                "current_expert_slots": 16,
                                "loan_bytes": 0,
                                "growth_physical_create_count": 0,
                                "physical_bytes": 100,
                            }
                        }
                    }
                ]
            }

    args = SimpleNamespace(
        arm="lend",
        baseline_split="expert-heavy",
        precondition_expert_heavy_fixed=True,
        precondition_requests=8,
        precondition_input_tokens=8,
        precondition_output_tokens=64,
        requests=8,
        input_tokens=16,
        output_tokens=4,
        expert_initial_slots=16,
        expert_extra_slots=4,
    )
    engine = FakeEngine()
    result = comparison.precondition_expert_heavy_comparator(engine, args)
    assert result["warmup_role"] == "adaptive-base-install"
    assert result["current_slots"] == 16
    assert engine.kwargs["sampling_params"]["max_new_tokens"] == 2
    assert len(engine.kwargs["input_ids"]) == 1


def test_expert_heavy_precondition_rejects_new_physical_growth():
    class FakeEngine:
        def generate(self, **_kwargs):
            pass

        def get_server_info(self):
            return {
                "internal_states": [
                    {
                        "layerkv": {
                            "shared_vmm": {
                                "current_expert_slots": 20,
                                "loan_bytes": 4096,
                                "growth_physical_create_count": 1,
                            }
                        }
                    }
                ]
            }

    args = SimpleNamespace(
        arm="fixed",
        baseline_split="expert-heavy",
        precondition_expert_heavy_fixed=True,
        precondition_requests=1,
        precondition_input_tokens=8,
        precondition_output_tokens=8,
        requests=1,
        input_tokens=8,
        output_tokens=4,
        expert_initial_slots=16,
        expert_extra_slots=4,
    )
    with pytest.raises(RuntimeError, match="physical KV-loan comparator"):
        comparison.precondition_expert_heavy_fixed(FakeEngine(), args)


def test_kv_overflow_is_not_enabled_in_fixed_comparator(monkeypatch):
    monkeypatch.setattr(
        comparison, "engine_kwargs", lambda _args: {"extra": _args.expert_extra_slots}
    )
    args = SimpleNamespace(
        max_running_requests=8,
        expert_extra_slots=48,
        expert_backing_cache_mb=128,
        expert_cpu_backing_pool_mb=128,
        expert_policy="adaptive",
        baseline_split="kv-heavy",
        expert_kv_overflow_tokens=4096,
        expert_kv_min_slots=8,
    )
    fixed = comparison.effective_engine_args(args, "fixed")
    lend = comparison.effective_engine_args(args, "lend")
    assert fixed["layerkv_shared_expert_kv_overflow_tokens"] == 0
    assert fixed["layerkv_shared_expert_kv_min_slots"] == 0
    assert lend["layerkv_shared_expert_kv_overflow_tokens"] == 4096
    assert lend["layerkv_shared_expert_kv_min_slots"] == 8
