from types import SimpleNamespace

from sglang.srt.layerkv.config_stats import LayerKVConfig, LayerKVStats
from sglang.srt.layerkv.kvc_reclaim import LayerKVKvcReclaimMixin


def test_actual_decode_batch_not_cached_scheduler_batch():
    runtime = SimpleNamespace(
        stats=LayerKVStats(),
        _current_forward_mode="decode",
        _last_scheduled_req_lens=[(0, 100)],
        _bytes_per_token_all_layers=4096,
        _batch_req_indices_and_lens=lambda _: [],
    )
    for mode, size in [("extend", 8), ("decode", 4), ("decode", 2), ("decode", 0)]:
        runtime._current_forward_mode = mode
        LayerKVKvcReclaimMixin._refresh_workload_stats(
            runtime, SimpleNamespace(batch_size=size)
        )
    stats = runtime.stats.as_dict()
    assert stats["observed_decode_forward_count"] == 2
    assert stats["observed_decode_request_steps"] == 6
    assert stats["observed_decode_batch_size_max"] == 4
    assert stats["observed_decode_batch_histogram"] == {4: 1, 2: 1}
    assert LayerKVStats().observed_decode_batch_histogram == {}


def test_free_donor_config_is_opt_in_and_propagates():
    assert not LayerKVConfig().shared_expert_free_kv_donors
    assert LayerKVConfig.from_server_args(
        SimpleNamespace(layerkv_shared_expert_free_kv_donors=True)
    ).shared_expert_free_kv_donors


def test_virtual_scratch_lending_config_is_opt_in_and_propagates():
    assert not LayerKVConfig().shared_expert_lend_virtual_scratch
    assert LayerKVConfig.from_server_args(
        SimpleNamespace(layerkv_shared_expert_lend_virtual_scratch=True)
    ).shared_expert_lend_virtual_scratch
