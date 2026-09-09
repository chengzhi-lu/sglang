"""CPU-only fixtures catch incorrect CUDA batch/overlap attribution."""

import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[4] / "scripts/layerkv_wait_trace_analyze.py"
_spec = importlib.util.spec_from_file_location("wait_trace_analyze", _path)
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)


def event(cat, name, start, duration, correlation=None, **args):
    if correlation is not None:
        args["correlation"] = correlation
    gpu = cat in ("kernel", "gpu_memcpy", "gpu_user_annotation")
    if gpu:
        args["device"] = 0
    return dict(
        ph="X",
        cat=cat,
        name=name,
        ts=start,
        dur=duration,
        pid=0 if gpu else 100,
        tid=7 if gpu else 100,
        args=args,
    )


def fixture():
    phases = [
        ("chunk/prepare", 0, 100),
        ("h2d_submit", 5, 10),
        ("cache_trim", 15, 5),
        ("ready_wait", 20, 5),
        ("remap", 25, 5),
        ("guard_reduce", 30, 5),
        ("readback", 35, 65),
    ]
    return [event("user_annotation", f"layerkv/{n}", t, d) for n, t, d in phases] + [
        # One batch can generate multiple device records. INVALID is a real
        # CUDA 13 batch API spelling in the installed Kineto, not missing data.
        event("cuda_runtime", "INVALID", 6, 1, 1),
        event("cuda_runtime", "cudaLaunchKernel", 26, 1, 2),
        event("cuda_runtime", "cudaLaunchKernel", 31, 1, 3),
        event("cuda_runtime", "cudaMemcpyAsync", 36, 1, 4),
        event("cuda_runtime", "cudaLaunchKernel", -5, 1, 5),
        event("cuda_runtime", "cudaStreamSynchronize", 1, 2, 6),
        event("gpu_memcpy", "Memcpy HtoD", 10, 50, 1, bytes=64),
        event("gpu_memcpy", "Memcpy HtoD", 50, 25, 1, bytes=64),
        # GPU execution is outside the CPU remap range: use launch correlation.
        event("kernel", "remap", 75, 5, 2),
        event("kernel", "guard", 80, 7, 3),
        event("gpu_memcpy", "Memcpy DtoH", 90, 2, 4, bytes=1),
        event("kernel", "previous_group", 20, 20, 5),
        event("gpu_user_annotation", "layerkv/readback", 35, 65),
    ]


def test_correlation_batch_records_and_disjoint_overlap():
    result = analysis.analyze(fixture(), 1, 1, 128)
    p = result["prefill"]
    assert result["request_h2d_batches"] == 1
    assert result["cpu_phase_counts"]["readback"] == 1
    assert p["h2d_gpu_ms"] == pytest.approx(0.065)
    assert p["gpu_phase_ms"]["remap"] == pytest.approx(0.005)
    assert p["h2d_launch_to_first_start_ms"] == pytest.approx(0.003)
    assert p["before_h2d_sync_calls"] == 1
    assert p["earlier_submitted_gpu_overlap_readback_ms"] == pytest.approx(0.005)
    assert p["readback_partition_ms"] == pytest.approx(
        {
            "own_h2d": 0.040,
            "own_remap_guard_readback": 0.014,
            "other_gpu_activity": 0.0,
            "no_observed_gpu_activity": 0.011,
        }
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "ambiguous",
        "no_h2d",
        "wrong_bytes",
        "wrong_groups",
        "missing_guard",
        "missing_guard_gpu",
    ],
)
def test_incomplete_trace_fails_closed(damage):
    events = fixture()
    if damage in ("missing", "ambiguous"):
        api = next(e for e in events if e["name"] == "INVALID")
        if damage == "missing":
            events.remove(api)
        else:
            events.append(dict(api))
    if damage == "no_h2d":
        events = [e for e in events if e["name"] != "Memcpy HtoD"]
    if damage == "missing_guard":
        events = [e for e in events if e["name"] != "layerkv/guard_reduce"]
    if damage == "missing_guard_gpu":
        events = [e for e in events if e["name"] != "guard"]
    with pytest.raises(ValueError):
        analysis.analyze(
            events,
            2 if damage == "wrong_groups" else 1,
            1,
            256 if damage == "wrong_bytes" else 128,
        )


def test_interval_union_clips_without_double_counting():
    events = [dict(ts=0, dur=10), dict(ts=5, dur=10), dict(ts=50, dur=20)]
    assert analysis.union_us(events, (7, 60)) == 18
