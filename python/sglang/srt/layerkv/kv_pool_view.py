"""Raw full-attention KV storage in model-layer coordinates.

Hooks stay on the runner's pool. This view only exposes storage to the host
store, so copying KV cannot recursively trigger reload hooks or touch GDN state.
"""

from typing import Any


class HybridKVCStorageView:
    def __init__(self, pool: Any):
        self.pool = pool.full_kv_pool
        self.layer_id_mapping = dict(pool.full_attention_layer_id_mapping)
        self.layer_ids = tuple(sorted(self.layer_id_mapping))
        if not self.layer_ids or sorted(self.layer_id_mapping.values()) != list(
            range(self.pool.layer_num)
        ):
            raise ValueError("invalid hybrid full-attention layer mapping")
        self.layer_num = len(self.layer_ids)
        self.start_layer = self.layer_ids[0]
        self.size = self.pool.size
        self.device = self.pool.device
        self.page_size = self.pool.page_size

    def _get_key_buffer(self, layer_id: int):
        return self.pool._get_key_buffer(self.layer_id_mapping[layer_id])

    def _get_value_buffer(self, layer_id: int):
        return self.pool._get_value_buffer(self.layer_id_mapping[layer_id])
