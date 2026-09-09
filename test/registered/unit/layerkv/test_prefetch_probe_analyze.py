"""Missing observations must not become zero prefetch opportunity."""

import importlib.util
from pathlib import Path

import pytest

from sglang.srt.layerkv.prefetch_probe import capacity_snapshot


@pytest.fixture
def analysis(monkeypatch):
    scripts = Path(__file__).resolve().parents[4] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "prefetch_analysis", scripts / "layerkv_prefetch_probe_analyze.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(full=False):
    from test_wait_trace_analyze import event

    events = [
        event("user_annotation", "layerkv/chunk/moe", 0, 10),
        event("user_annotation", "layerkv/chunk/moe", 100, 10),
        event("cuda_runtime", "cudaLaunchKernel", 1, 1, 1),
        event("cuda_runtime", "cudaLaunchKernel", 101, 1, 2),
        # Device times lie outside CPU phase ranges; correlation is required.
        event("kernel", "moe0", 20, 10, 1),
        event("kernel", "moe1", 120, 10, 2),
        event("gpu_user_annotation", "layerkv/chunk/moe", 20, 10),
    ]
    current = [0, 1] if full else [0]
    a = capacity_snapshot(2, current, [2], {0: 0, 1: 1}, {0: 0, 1: 1}, {0, 1, 2}, 16)
    b = capacity_snapshot(2, [2], None, {0: 0, 2: 1}, {0: 0, 1: 2}, {0, 1, 2}, 16)
    probe = dict(order="input", groups=[dict(group=1, **a), dict(group=2, **b)])
    waits = dict(
        groups=[
            dict(h2d_bytes=32, h2d_gpu_ms=0.04),
            dict(h2d_bytes=16, h2d_gpu_ms=0.02),
        ]
    )
    return events, probe, waits


@pytest.mark.parametrize("full", [False, True])
def test_capacity_gates_gpu_window(analysis, full):
    result = analysis.join_windows(*fixture(full))
    assert result["group_count"] == 2 and result["transition_count"] == 1
    assert result["moe_kernel_ms"] == pytest.approx(0.02)
    assert result["backed_upper_experts"] == (0 if full else 1)
    assert result["loose_overlap_ceiling_ms"] == pytest.approx(0 if full else 0.01)


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "gpu", "count", "bytes", "order", "probe"]
)
def test_bad_measurement_rejected(analysis, damage):
    events, probe, waits = fixture()
    if damage == "missing":
        events = [e for e in events if e["name"] != "cudaLaunchKernel"]
    elif damage == "duplicate":
        events.append(events[2].copy())
    elif damage == "gpu":
        events = [e for e in events if e["name"] != "moe0"]
    elif damage == "count":
        probe["groups"].pop()
    elif damage == "bytes":
        waits["groups"][1]["h2d_bytes"] = 0
    elif damage == "order":
        probe["order"] = "reuse"
    else:
        probe = None
    with pytest.raises(ValueError):
        analysis.join_windows(events, probe, waits)
