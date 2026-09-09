from collections import namedtuple
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.srt.layerkv import native_moe_graph as module


def test_capture_timing_counts_replacements_without_extra_synchronization(monkeypatch):
    # Exercise capture bookkeeping with CPU tensors and fake CUDA contexts.
    ticks = iter(range(10))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    syncs = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: syncs.append(device))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    stream = SimpleNamespace(wait_stream=lambda other: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: object())
    monkeypatch.setattr(torch.cuda, "graph", lambda graph, stream: nullcontext())
    TopK = namedtuple("TopK", "topk_weights topk_ids router_logits")
    Dispatch = namedtuple("Dispatch", "hidden_states topk_output")
    Result = namedtuple("Result", "hidden_states")
    dispatch = Dispatch(
        torch.ones((2, 4)),
        TopK(torch.ones((2, 1)), torch.zeros((2, 1), dtype=torch.int32), None),
    )
    replay = module.NativeMoEGraph(lambda d: Result(d.hidden_states), lambda: ())
    replay._capture(dispatch, (), "first")
    assert replay.recapture_count == 0 and replay.captures == 1
    replay._capture(dispatch, (), "second")
    assert replay.recapture_count == 1 and replay.captures == 2
    assert len(syncs) == 2  # The existing one synchronization per capture only.
    assert replay.capture_wall_ms == 8000
    assert replay.capture_sync_ms == replay.capture_context_ms == 2000
