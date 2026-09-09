"""Focused prepare profiling must preserve calls, exceptions and shape buckets."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.prepare_profile import PrepareProfiler


def dispatch(rows):
    return SimpleNamespace(topk_output=SimpleNamespace(topk_ids=torch.zeros(rows, 2)))


def test_prepare_profile_separates_first_and_repeated_shapes():
    profiler = PrepareProfiler()
    state = SimpleNamespace(slot_capacity=16)
    calls = []

    def prepare(state, value):
        calls.append(value)
        return value

    for rows in (3, 3, 5, 3):
        value = dispatch(rows)
        assert profiler.run(prepare, state, value) is value
    summary = profiler.summary()
    assert len(calls) == 4
    assert summary["first_shape"]["calls"] == 2
    assert summary["repeat_shape"]["calls"] == 2
    assert len(summary["shapes"]) == 2
    for bucket in ("first_shape", "repeat_shape"):
        assert any(row["name"] == "prepare" for row in summary[bucket]["functions"])
        assert summary[bucket]["wall_ms"] > 0


def test_prepare_profile_restores_profiler_on_exception():
    import sys

    profiler = PrepareProfiler()

    def fail(*_):
        raise ValueError("injected prepare failure")

    with pytest.raises(ValueError, match="injected"):
        profiler.run(fail, SimpleNamespace(slot_capacity=16), dispatch(1))
    assert sys.getprofile() is None
    assert profiler.summary()["first_shape"]["calls"] == 1


def test_prepare_profile_call_counts_across_many_bucket_switches():
    profiler = PrepareProfiler()
    state = SimpleNamespace(slot_capacity=16)

    def child(value):
        return value

    def prepare(state, value):
        return child(value)

    for rows in range(1, 150):
        for _ in range(3):
            value = dispatch(rows)
            assert profiler.run(prepare, state, value) is value
    for bucket, count in (("first_shape", 149), ("repeat_shape", 298)):
        summary = profiler.summary()[bucket]
        functions = {row["name"]: row for row in summary["functions"]}
        assert functions["prepare"]["calls"] == summary["calls"] == count
        assert functions["prepare"]["recursive_calls"] == 0
        assert functions["child"]["calls"] == count
        assert functions["prepare"]["total_ms"] >= functions["child"]["total_ms"]


def test_prepare_profile_summary_is_a_snapshot():
    profiler = PrepareProfiler()
    state = SimpleNamespace(slot_capacity=16)
    value = dispatch(3)

    def prepare(state, value):
        return value

    profiler.run(prepare, state, value)
    profiler.run(prepare, state, value)
    before = profiler.summary()
    profiler.run(prepare, state, value)
    assert before["repeat_shape"]["calls"] == 1
    assert before["shapes"][0]["calls"] == 2
    root = next(
        row for row in before["repeat_shape"]["functions"] if row["name"] == "prepare"
    )
    assert root["calls"] == 1


def test_prepare_profile_rejects_external_profiler_without_replacing_it():
    import sys

    def external(*_):
        pass

    sys.setprofile(external)
    try:
        with pytest.raises(RuntimeError, match="cannot nest"):
            PrepareProfiler().run(lambda *_: None, None, None)
        assert sys.getprofile() is external
    finally:
        sys.setprofile(None)


def test_prepare_profile_separates_scalar_callers_and_snapshots():
    profiler = PrepareProfiler()
    state = SimpleNamespace(slot_capacity=16)
    value = dispatch(3)
    scalar = torch.ones(())

    def guard():
        return scalar.item()

    def ready():
        return scalar.item() + scalar.item()

    def prepare(state, value):
        guard()
        ready()
        return value

    for _ in range(3):
        assert profiler.run(prepare, state, value) is value
    before = profiler.summary()
    for bucket, repeats in (("first_shape", 1), ("repeat_shape", 2)):
        edges = [e for e in before[bucket]["call_edges"] if "'item'" in e["name"]]
        assert {e["caller_name"]: e["calls"] for e in edges} == {
            "guard": repeats,
            "ready": 2 * repeats,
        }
        assert all(e["caller_file"] == __file__ and e["caller_line"] > 0 for e in edges)
        assert all(
            e["recursive_calls"] == 0 and e["total_ms"] >= e["self_ms"] >= 0
            for e in edges
        )
        function = next(e for e in before[bucket]["functions"] if "'item'" in e["name"])
        assert sum(e["calls"] for e in edges) == function["calls"]
        assert sum(e["self_ms"] for e in edges) == pytest.approx(function["self_ms"])
    profiler.run(prepare, state, value)
    assert before["repeat_shape"]["calls"] == 2
    assert (
        sum(
            e["calls"]
            for e in before["repeat_shape"]["call_edges"]
            if "'item'" in e["name"]
        )
        == 6
    )


def test_prepare_profile_keeps_edges_on_exception():
    profiler = PrepareProfiler()

    def child():
        raise ValueError("injected edge failure")

    def prepare(state, value):
        child()

    with pytest.raises(ValueError, match="injected edge"):
        profiler.run(prepare, SimpleNamespace(slot_capacity=16), dispatch(1))
    edges = profiler.summary()["first_shape"]["call_edges"]
    edge = next(
        e for e in edges if e["caller_name"] == "prepare" and e["name"] == "child"
    )
    assert edge["calls"] == 1 and edge["recursive_calls"] == 0
