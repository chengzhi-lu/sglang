"""Decode writes must preserve valid permutations of physical KV locations."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.config_stats import LayerKVConfig
from sglang.srt.layerkv.runtime import LayerKVRuntime


@pytest.mark.parametrize("physical", [[12, 11, 13], [12, 13, 11], [21, 22, 23]])
def test_decode_write_matches_attention_mapping(physical):
    runtime = LayerKVRuntime(LayerKVConfig(enabled=True, kvc_backend="per-layer-arena"))
    runtime._kv_pool = SimpleNamespace(size=32)
    runtime._per_layer_kvc_io_control_active = lambda: True
    canonical = [11, 12, 13]
    for source, destination in zip(canonical, physical):
        runtime._record_per_layer_mapping(15, source, destination)
    key, value = torch.zeros(33, 2), torch.zeros(33, 2)

    def write(layer, loc, cache_k, cache_v):
        key[loc] = cache_k
        value[loc] = cache_v

    cache_k = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    cache_v = cache_k + 10
    runtime._wrap_set_kv_buffer(write)(
        SimpleNamespace(layer_id=15), torch.tensor(canonical), cache_k, cache_v
    )
    assert torch.equal(key[physical], cache_k)
    assert torch.equal(value[physical], cache_v)
    # Translation is a lookup, not an ownership/lifetime transition.
    assert (
        runtime._translate_per_layer_locs(15, torch.tensor(canonical)).tolist()
        == physical
    )
