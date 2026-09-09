from types import SimpleNamespace

from sglang.srt.managers.scheduler_runtime_checker_mixin import (
    PoolStats,
    SchedulerRuntimeCheckerMixin,
)


class _LayerKVCheckerStub(SchedulerRuntimeCheckerMixin):
    def __init__(self, *, allocator_size, available_size, layerkv_runtime):
        self.max_total_num_tokens = 16
        self.is_hybrid_swa = False
        self.is_hybrid_ssm = False
        self.token_to_kv_pool_allocator = SimpleNamespace(
            size=allocator_size,
            available_size=lambda: available_size,
        )
        self.tree_cache = SimpleNamespace(
            evictable_size=lambda: 0,
            protected_size=lambda: 0,
        )
        self.layerkv_runtime = layerkv_runtime

    def _get_layerkv_runtime(self):
        return self.layerkv_runtime

    def _session_held_tokens(self):
        return 0


def test_layerkv_dynamic_tail_is_used_for_pool_stats_and_leak_check():
    scheduler = _LayerKVCheckerStub(
        allocator_size=24,
        available_size=8,
        layerkv_runtime=object(),
    )

    stats = scheduler._get_token_info()
    assert stats.full_num_used == 16
    assert stats.full_token_usage == 16 / 24

    leak, _ = scheduler._check_full_pool(stats, uncached=16)
    assert not leak


def test_non_layerkv_pool_keeps_configured_total_for_checks():
    scheduler = _LayerKVCheckerStub(
        allocator_size=24,
        available_size=8,
        layerkv_runtime=None,
    )

    stats = scheduler._get_token_info()
    assert stats.full_num_used == 8
    assert stats.full_token_usage == 8 / 16

    leak, _ = scheduler._check_full_pool(stats, uncached=8)
    assert not leak
