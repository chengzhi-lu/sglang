#include <torch/library.h>

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "layerkv_copy_kv_span_scatter(Tensor src_k, Tensor src_v, Tensor dst_k, Tensor dst_v, Tensor spans, "
      "int src_base_slot, int dst_base_slot, int item_size, int num_warps_per_block) -> ()");
  m.impl("layerkv_copy_kv_span_scatter", torch::kCUDA, &layerkv_copy_kv_span_scatter);
  m.def(
      "layerkv_copy_kv_span_scatter_batched(Tensor[] src_ks, Tensor[] src_vs, Tensor[] dst_ks, Tensor[] dst_vs, "
      "Tensor spans, Tensor span_batch_ids, Tensor src_base_slots, Tensor dst_base_slots, int item_size, "
      "int num_warps_per_block) -> ()");
  m.impl("layerkv_copy_kv_span_scatter_batched", torch::kCUDA, &layerkv_copy_kv_span_scatter_batched);
  m.def(
      "layerkv_copy_kv_span_backup_batched(Tensor[] src_ks, Tensor[] src_vs, Tensor[] dst_ks, Tensor[] dst_vs, "
      "Tensor spans, Tensor span_batch_ids, int item_size) -> ()");
  m.impl("layerkv_copy_kv_span_backup_batched", torch::kCUDA, &layerkv_copy_kv_span_backup_batched);
}
