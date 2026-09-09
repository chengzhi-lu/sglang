"""Reload must not overwrite decode tokens living in recycled eviction slots."""

import pytest

from sglang.srt.layerkv.common_types import _LayerKVResidencyEntry
from sglang.srt.layerkv.runtime import LayerKVConfig, LayerKVRuntime


@pytest.mark.parametrize("legacy", [False, True])
def test_reload_rejects_evicted_location_reused_by_decode(legacy):
    runtime = LayerKVRuntime(LayerKVConfig(kvc_backend="per-layer-arena"))
    runtime._push_per_layer_overwrite_bits(7, (1 << 100) | (1 << 101))
    reused = runtime._alloc_per_layer_overwrite_locs_by_layer([7], 1)[7][0]
    entry = _LayerKVResidencyEntry(
        req_idx=2,
        pos=6002,
        state="resident",
        layer_id=7,
        device_loc=reused,
        device_locs=[reused],
        page_size=1,
    )
    runtime._track_per_layer_cleanup_append(entry, reused)
    if legacy:
        state = runtime._per_layer_cleanup_state_by_req.pop(2)
        runtime._per_layer_cleanup_locs_by_req_layer[(2, 7)] = state.loc_bits_by_layer[
            7
        ]
    # Fast-path decode ownership need not be in the arena allocation set.
    assert reused not in runtime._per_layer_arena_allocated_locs.get(7, set())
    assert runtime._reuse_evicted_per_layer_locs(7, [100, 101]) is None
    # Failed reuse must leave the other free location available.
    assert runtime._pop_per_layer_overwrite_locs(7, 1) == [201 - reused]
