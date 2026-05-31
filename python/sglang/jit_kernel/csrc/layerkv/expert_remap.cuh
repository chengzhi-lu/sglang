#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <typename T>
__global__ void expert_remap_kernel(
    const T* __restrict__ topk_ids,
    const int64_t* __restrict__ remap,
    T* __restrict__ rewritten_ids,
    int64_t* __restrict__ missing_ids,
    int32_t* __restrict__ missing_count,
    int64_t n,
    int64_t full_num_experts) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  int64_t logical_id = static_cast<int64_t>(topk_ids[idx]);
  if (logical_id < 0 || logical_id >= full_num_experts) {
    rewritten_ids[idx] = topk_ids[idx];
    return;
  }

  int64_t slot_id = remap[logical_id];
  if (slot_id >= 0) {
    rewritten_ids[idx] = static_cast<T>(slot_id);
    return;
  }

  rewritten_ids[idx] = topk_ids[idx];
  int32_t out = atomicAdd(missing_count, 1);
  missing_ids[out] = logical_id;
}

constexpr size_t kBlockSize = 256;

template <typename T>
struct LayerKVExpertRemap {
  static void run(
      tvm::ffi::TensorView topk_ids,
      tvm::ffi::TensorView remap,
      tvm::ffi::TensorView rewritten_ids,
      tvm::ffi::TensorView missing_ids,
      tvm::ffi::TensorView missing_count,
      int64_t full_num_experts) {
    using namespace host;

    SymbolicSize N = {"num_elements"};
    SymbolicDevice device_;
    device_.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({N}).with_dtype<T>().with_device(device_).verify(topk_ids).verify(rewritten_ids);
    TensorMatcher({full_num_experts}).with_dtype<int64_t>().with_device(device_).verify(remap);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device_).verify(missing_ids);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device_).verify(missing_count);

    const int64_t num_elements = static_cast<int64_t>(N.unwrap());
    if (num_elements == 0) return;

    const size_t grid_size = div_ceil(static_cast<size_t>(num_elements), kBlockSize);
    const DLDevice device = device_.unwrap();
    LaunchKernel(grid_size, kBlockSize, device)(
        expert_remap_kernel<T>,
        static_cast<const T*>(topk_ids.data_ptr()),
        static_cast<const int64_t*>(remap.data_ptr()),
        static_cast<T*>(rewritten_ids.data_ptr()),
        static_cast<int64_t*>(missing_ids.data_ptr()),
        static_cast<int32_t*>(missing_count.data_ptr()),
        num_elements,
        full_num_experts);
  }
};

}  // namespace
