from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layerkv.activation_probe import create_hook
from sglang.srt.model_executor.hook_manager import register_forward_hooks


def test_probe_rejects_invalid_range(tmp_path):
    with pytest.raises(ValueError, match="lower <= upper"):
        create_hook(dict(directory=tmp_path, seq_range=[7, 6], layer=0))


def test_hook_registration_preserves_three_argument_hooks(monkeypatch):
    observed = []
    monkeypatch.setattr(
        "sglang.srt.model_executor.hook_manager.resolve_callable",
        lambda path: lambda config: lambda module, inputs, output: observed.append(
            output
        ),
    )
    module = torch.nn.Identity()
    register_forward_hooks(module, [dict(target_modules=[""], hook_factory="test")])
    value = torch.ones(1)
    assert module(value) is value
    assert observed == [value]


def test_probe_filters_decode_and_copies_outputs(tmp_path):
    module = torch.nn.Identity()
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        seq_lens=torch.tensor([6056]),
        req_pool_indices=torch.tensor([1]),
    )
    hook = create_hook(dict(directory=tmp_path, seq_range=[6055, 6059], layer=3))
    values = torch.ones(1, 4)
    hook(module, (), dict(forward_batch=batch), (values, None))
    values.zero_()
    record = torch.load(tmp_path / "layer03_0000.pt", weights_only=True)
    assert record["outputs"][0].sum() == 4
    assert record["seq_lens"].tolist() == [6056]
    batch.seq_lens.fill_(100)
    hook(module, (), dict(forward_batch=batch), (values, None))
    batch.seq_lens.fill_(6056)
    batch.forward_mode.is_decode = lambda: False
    hook(module, (), dict(forward_batch=batch), (values, None))
    assert len(list(tmp_path.iterdir())) == 1


def test_hook_registration_passes_keyword_batch(tmp_path):
    class Decoder(torch.nn.Module):
        def forward(self, *, hidden_states, forward_batch):
            return hidden_states

    module = Decoder()
    register_forward_hooks(
        module,
        [
            dict(
                name="test",
                target_modules=[""],
                with_kwargs=True,
                hook_factory="sglang.srt.layerkv.activation_probe:create_hook",
                config=dict(directory=str(tmp_path), seq_range=[5, 6], layer=0),
            )
        ],
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        seq_lens=torch.tensor([5]),
        req_pool_indices=torch.tensor([0]),
    )
    module(hidden_states=torch.ones(1, 4), forward_batch=batch)
    assert (tmp_path / "layer00_0000.pt").exists()


def test_kv_probe_uses_packed_indices_and_raw_storage(tmp_path):
    key = torch.arange(20).reshape(10, 1, 2)
    raw = SimpleNamespace(
        layer_num=1,
        size=10,
        device="cpu",
        page_size=1,
        _get_key_buffer=lambda layer: key,
        _get_value_buffer=lambda layer: -key,
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        seq_lens=torch.tensor([2]),
        req_pool_indices=torch.tensor([1]),
        token_to_kv_pool=SimpleNamespace(
            full_kv_pool=raw,
            full_attention_layer_id_mapping={7: 0},
        ),
        attn_backend=SimpleNamespace(
            forward_metadata=SimpleNamespace(
                kv_indptr=torch.tensor([0, 2]),
                kv_indices=torch.tensor([4, 2, 99]),
            )
        ),
    )
    hook = create_hook(
        dict(directory=tmp_path, seq_range=[2, 2], layer=7, capture_kv=True)
    )
    hook(torch.nn.Identity(), (), dict(forward_batch=batch), torch.ones(1))
    record = torch.load(tmp_path / "layer07_0000.pt", weights_only=True)["kv"]
    assert record["indices"].tolist() == [4, 2]
    assert torch.equal(record["k"], key[[4, 2]])
    assert torch.equal(record["v"], -key[[4, 2]])
