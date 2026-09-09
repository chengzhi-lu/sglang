from types import SimpleNamespace

import pytest

from sglang.srt.layerkv.config_stats import LayerKVConfig
from sglang.srt.layerkv import shared_expert


def test_delayed_event_keeps_submission_phase_and_counts_once(monkeypatch):
    runtime = SimpleNamespace(
        config=LayerKVConfig(shared_expert_profile_chunks=True),
        _current_forward_mode="extend",
    )
    c = shared_expert.SharedExpertController(runtime, None)
    ticks = iter([0, 0.002])
    monkeypatch.setattr(shared_expert.time, "perf_counter", lambda: next(ticks))
    ready = [False]

    class Event:
        def record(self, stream):
            pass

        def query(self):
            return ready[0]

        def elapsed_time(self, end):
            return 1.5

    monkeypatch.setattr(shared_expert.torch.cuda, "Event", lambda **kw: Event())
    monkeypatch.setattr(shared_expert.torch.cuda, "current_stream", lambda device: None)
    with c._profile_chunk_phase("moe", SimpleNamespace(type="cuda")):
        runtime._current_forward_mode = "decode"
    assert c._chunk_by_mode["prefill"]["wall_ms"]["moe"] == pytest.approx(2)
    c._collect_chunk_profile()
    assert len(c._chunk_events) == 1
    ready[0] = True
    c._collect_chunk_profile()
    c._collect_chunk_profile()
    assert not c._chunk_events
    assert c._chunk_by_mode["prefill"]["stream_ms"]["moe"] == 1.5
    assert c._chunk_by_mode["decode"]["stream_ms"]["moe"] == 0
    assert c._chunk_gpu_ms["moe"] == 1.5


def test_disabled_profile_does_not_read_clock(monkeypatch):
    c = shared_expert.SharedExpertController(
        SimpleNamespace(config=LayerKVConfig()), None
    )

    def unexpected():
        raise AssertionError("disabled timing read clock")

    monkeypatch.setattr(shared_expert.time, "perf_counter", unexpected)
    with c._profile_chunk_phase("moe"):
        pass
    assert c._chunk_wall_ms["moe"] == 0
