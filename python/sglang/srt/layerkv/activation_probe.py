"""Opt-in synchronous decode snapshots; diagnostic only, never performance data."""

from pathlib import Path

import torch


def create_hook(config):
    directory = Path(config["directory"])
    lower, upper = config["seq_range"]
    if not 0 < lower <= upper:
        raise ValueError("activation probe requires 0 < lower <= upper")
    layer = int(config["layer"])
    directory.mkdir(parents=True, exist_ok=True)
    capture = 0

    def hook(module, inputs, kwargs, output):
        nonlocal capture
        batch = kwargs.get("forward_batch")
        if batch is None or not batch.forward_mode.is_decode():
            return
        lengths = batch.seq_lens.detach().cpu()
        if not ((lengths >= lower) & (lengths <= upper)).any().item():
            return
        tensors = output if isinstance(output, tuple) else (output,)
        record = {
            "layer": layer,
            "module_type": type(module).__name__,
            "seq_lens": lengths,
            "req_pool_indices": batch.req_pool_indices.detach().cpu(),
            "outputs": [
                value.detach().cpu().clone() if torch.is_tensor(value) else None
                for value in tensors
            ],
        }
        if config.get("capture_kv", False):
            from sglang.srt.layerkv.kv_pool_view import HybridKVCStorageView

            backend = getattr(
                batch.attn_backend, "full_attn_backend", batch.attn_backend
            )
            metadata = backend.forward_metadata
            indptr = metadata.kv_indptr.detach().cpu()
            indices = metadata.kv_indices[: int(indptr[-1])]
            storage = HybridKVCStorageView(batch.token_to_kv_pool)
            # Raw storage only: public getters would run LayerKV preparation again.
            record["kv"] = {
                "indptr": indptr,
                "indices": indices.detach().cpu(),
                "k": storage._get_key_buffer(layer)[indices.long()].detach().cpu(),
                "v": storage._get_value_buffer(layer)[indices.long()].detach().cpu(),
            }
        # A fresh output directory is required; never overwrite earlier evidence.
        with (directory / f"layer{layer:02d}_{capture:04d}.pt").open("xb") as stream:
            torch.save(record, stream)
        capture += 1

    return hook
