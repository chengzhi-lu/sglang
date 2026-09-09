"""Qwen's runtime precision must also select its convolution-state storage."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qwen_runtime_conv_dtype_preserves_ssm_and_checkpoint(monkeypatch, dtype):
    monkeypatch.setattr(
        "sglang.srt.layers.dp_attention.get_attention_tp_size", lambda: 1
    )
    config = Qwen3_5MoeTextConfig(full_attention_interval=4)
    original = config.mamba2_cache_params
    params = ModelRunnerKVCacheMixin._mamba_cache_params_for_runtime(
        SimpleNamespace(dtype=dtype), config
    )
    assert params.dtype.conv == dtype
    assert params.dtype.temporal == original.dtype.temporal
    assert params.shape == original.shape
    assert params.layers == original.layers
    assert config.mamba2_cache_params.dtype == original.dtype


def test_non_qwen_cache_params_unchanged():
    original = object()
    config = SimpleNamespace(mamba2_cache_params=original)
    assert (
        ModelRunnerKVCacheMixin._mamba_cache_params_for_runtime(
            SimpleNamespace(dtype=torch.float16), config
        )
        is original
    )
