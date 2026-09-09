#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <stdexcept>

namespace {

constexpr int64_t kMaskWordBits = 64;
constexpr int64_t kMaxMaskWords = 64;
constexpr int64_t kRouteMaskBlockSize = 256;

template <typename T>
__global__ void expert_route_mask_kernel(
    const T* __restrict__ topk_ids,
    int64_t* __restrict__ route_masks,
    int32_t* __restrict__ routing_valid,
    int32_t* __restrict__ capacity_violation,
    int64_t num_rows,
    int64_t top_k,
    int64_t capacity,
    int64_t full_num_experts,
    int64_t num_words) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;

  auto* row_masks = reinterpret_cast<unsigned long long*>(
      route_masks + row * num_words);
  for (int64_t word = 0; word < num_words; ++word) {
    row_masks[word] = 0;
  }
  for (int64_t k = 0; k < top_k; ++k) {
    const int64_t logical_id =
        static_cast<int64_t>(topk_ids[row * top_k + k]);
    if (logical_id < 0 || logical_id >= full_num_experts) {
      atomicExch(routing_valid, 0);
      continue;
    }
    row_masks[logical_id / kMaskWordBits] |=
        1ULL << (logical_id % kMaskWordBits);
  }

  int64_t route_width = 0;
  for (int64_t word = 0; word < num_words; ++word) {
    route_width += __popcll(row_masks[word]);
  }
  if (route_width > capacity) {
    atomicExch(capacity_violation, 1);
  }
}

__global__ void expert_group_kernel(
    const int64_t* __restrict__ route_masks,
    int32_t* __restrict__ group_ids,
    int64_t* __restrict__ group_masks,
    int32_t* __restrict__ group_count,
    int32_t* __restrict__ capacity_violation,
    int64_t num_rows,
    int64_t capacity,
    int64_t num_words) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (*capacity_violation != 0) {
    *group_count = 0;
    return;
  }

  int32_t groups = 0;
  unsigned long long route[kMaxMaskWords] = {};
  for (int64_t row = 0; row < num_rows; ++row) {
    for (int64_t word = 0; word < num_words; ++word) {
      route[word] = static_cast<unsigned long long>(
          route_masks[row * num_words + word]);
    }

    int32_t selected = -1;
    for (int32_t group = 0; group < groups; ++group) {
      int64_t union_width = 0;
      for (int64_t word = 0; word < num_words; ++word) {
        const unsigned long long current = static_cast<unsigned long long>(
            group_masks[static_cast<int64_t>(group) * num_words + word]);
        union_width += __popcll(current | route[word]);
      }
      if (union_width <= capacity) {
        selected = group;
        break;
      }
    }

    if (selected < 0) {
      selected = groups++;
      for (int64_t word = 0; word < num_words; ++word) {
        group_masks[static_cast<int64_t>(selected) * num_words + word] =
            static_cast<int64_t>(route[word]);
      }
    } else {
      for (int64_t word = 0; word < num_words; ++word) {
        auto* current = reinterpret_cast<unsigned long long*>(
            &group_masks[static_cast<int64_t>(selected) * num_words + word]);
        *current |= route[word];
      }
    }
    group_ids[row] = selected;
  }
  *group_count = groups;
}

template <typename T>
__global__ void expert_route_mask_multi_kernel(
    const T* __restrict__ topk_ids,
    int64_t* __restrict__ route_masks,
    int32_t* __restrict__ status,
    const int32_t* __restrict__ capacities,
    int64_t num_rows,
    int64_t top_k,
    int64_t num_capacities,
    int64_t full_num_experts,
    int64_t num_words) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;

  auto* row_masks = reinterpret_cast<unsigned long long*>(
      route_masks + row * num_words);
  for (int64_t word = 0; word < num_words; ++word) {
    row_masks[word] = 0;
  }
  for (int64_t k = 0; k < top_k; ++k) {
    const int64_t logical_id =
        static_cast<int64_t>(topk_ids[row * top_k + k]);
    if (logical_id < 0 || logical_id >= full_num_experts) {
      for (int64_t index = 0; index < num_capacities; ++index) {
        atomicExch(&status[index * 3 + 1], 0);
      }
      continue;
    }
    row_masks[logical_id / kMaskWordBits] |=
        1ULL << (logical_id % kMaskWordBits);
  }

  int64_t route_width = 0;
  for (int64_t word = 0; word < num_words; ++word) {
    route_width += __popcll(row_masks[word]);
  }
  for (int64_t index = 0; index < num_capacities; ++index) {
    if (route_width > capacities[index]) {
      atomicExch(&status[index * 3 + 2], 1);
    }
  }
}

__global__ void expert_group_multi_kernel(
    const int64_t* __restrict__ route_masks,
    int32_t* __restrict__ group_ids,
    int64_t* __restrict__ group_masks,
    int32_t* __restrict__ group_sizes,
    int32_t* __restrict__ status,
    const int32_t* __restrict__ capacities,
    int64_t num_rows,
    int64_t num_capacities,
    int64_t num_words) {
  const int64_t capacity_index = static_cast<int64_t>(blockIdx.x);
  if (capacity_index >= num_capacities || threadIdx.x != 0) return;
  if (status[capacity_index * 3 + 2] != 0) return;

  const int64_t group_id_offset = capacity_index * num_rows;
  const int64_t group_mask_offset = capacity_index * num_rows * num_words;
  const int64_t group_size_offset = capacity_index * num_rows;
  const int64_t capacity = capacities[capacity_index];
  int32_t groups = 0;
  unsigned long long route[kMaxMaskWords] = {};
  for (int64_t row = 0; row < num_rows; ++row) {
    for (int64_t word = 0; word < num_words; ++word) {
      route[word] = static_cast<unsigned long long>(
          route_masks[row * num_words + word]);
    }

    int32_t selected = -1;
    for (int32_t group = 0; group < groups; ++group) {
      int64_t union_width = 0;
      for (int64_t word = 0; word < num_words; ++word) {
        const unsigned long long current = static_cast<unsigned long long>(
            group_masks[group_mask_offset +
                        static_cast<int64_t>(group) * num_words + word]);
        union_width += __popcll(current | route[word]);
      }
      if (union_width <= capacity) {
        selected = group;
        break;
      }
    }

    if (selected < 0) {
      selected = groups++;
      for (int64_t word = 0; word < num_words; ++word) {
        group_masks[group_mask_offset +
                    static_cast<int64_t>(selected) * num_words + word] =
            static_cast<int64_t>(route[word]);
      }
    } else {
      for (int64_t word = 0; word < num_words; ++word) {
        auto* current = reinterpret_cast<unsigned long long*>(
            &group_masks[group_mask_offset +
                         static_cast<int64_t>(selected) * num_words + word]);
        *current |= route[word];
      }
    }
    group_ids[group_id_offset + row] = selected;
    group_sizes[group_size_offset + selected] += 1;
  }
  status[capacity_index * 3] = groups;
}

template <typename T>
struct LayerKVExpertGroup {
  static void run(
      tvm::ffi::TensorView topk_ids,
      tvm::ffi::TensorView group_ids,
      tvm::ffi::TensorView group_masks,
      tvm::ffi::TensorView route_masks,
      tvm::ffi::TensorView group_count,
      tvm::ffi::TensorView routing_valid,
      tvm::ffi::TensorView capacity_violation,
      int64_t num_rows,
      int64_t top_k,
      int64_t capacity,
      int64_t full_num_experts,
      int64_t num_words) {
    using namespace host;

    SymbolicSize N = {"num_elements"};
    SymbolicSize R = {"num_rows"};
    SymbolicSize W = {"num_words"};
    SymbolicDevice device_;
    device_.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({N}).with_dtype<T>().with_device(device_).verify(topk_ids);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(group_ids);
    TensorMatcher({R, W})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(group_masks);
    TensorMatcher({R, W})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(route_masks);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device_).verify(group_count);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device_).verify(routing_valid);
    TensorMatcher({1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(capacity_violation);

    if (num_rows <= 0) return;
    if (num_rows * top_k != static_cast<int64_t>(N.unwrap()) ||
        num_words != static_cast<int64_t>(W.unwrap()) || num_words <= 0 ||
        num_words > kMaxMaskWords) {
      throw std::runtime_error("invalid LayerKV expert grouping shape");
    }

    const DLDevice device = device_.unwrap();
    const auto route_blocks = div_ceil(
        static_cast<size_t>(num_rows),
        static_cast<size_t>(kRouteMaskBlockSize));
    LaunchKernel(route_blocks, kRouteMaskBlockSize, device)(
        expert_route_mask_kernel<T>,
        static_cast<const T*>(topk_ids.data_ptr()),
        static_cast<int64_t*>(route_masks.data_ptr()),
        static_cast<int32_t*>(routing_valid.data_ptr()),
        static_cast<int32_t*>(capacity_violation.data_ptr()),
        num_rows,
        top_k,
        capacity,
        full_num_experts,
        num_words);
    LaunchKernel(1, 1, device)(
        expert_group_kernel,
        static_cast<const int64_t*>(route_masks.data_ptr()),
        static_cast<int32_t*>(group_ids.data_ptr()),
        static_cast<int64_t*>(group_masks.data_ptr()),
        static_cast<int32_t*>(group_count.data_ptr()),
        static_cast<int32_t*>(capacity_violation.data_ptr()),
        num_rows,
        capacity,
        num_words);
  }
};

template <typename T>
struct LayerKVExpertGroupMulti {
  static void run(
      tvm::ffi::TensorView topk_ids,
      tvm::ffi::TensorView capacities,
      tvm::ffi::TensorView group_ids,
      tvm::ffi::TensorView group_masks,
      tvm::ffi::TensorView group_sizes,
      tvm::ffi::TensorView route_masks,
      tvm::ffi::TensorView status,
      int64_t num_rows,
      int64_t top_k,
      int64_t full_num_experts,
      int64_t num_capacities,
      int64_t num_words) {
    using namespace host;

    SymbolicSize N = {"num_elements"};
    SymbolicSize R = {"num_rows"};
    SymbolicSize C = {"num_capacities"};
    SymbolicSize W = {"num_words"};
    SymbolicDevice device_;
    device_.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({N}).with_dtype<T>().with_device(device_).verify(topk_ids);
    TensorMatcher({C}).with_dtype<int32_t>().with_device(device_).verify(capacities);
    TensorMatcher({C, R})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(group_ids);
    TensorMatcher({C, R, W})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(group_masks);
    TensorMatcher({C, R})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(group_sizes);
    TensorMatcher({R, W})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(route_masks);
    TensorMatcher({C, 3})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(status);

    if (num_rows <= 0 || num_capacities <= 0) return;
    if (num_rows * top_k != static_cast<int64_t>(N.unwrap()) ||
        num_capacities != static_cast<int64_t>(C.unwrap()) ||
        num_words != static_cast<int64_t>(W.unwrap()) || num_words <= 0 ||
        num_words > kMaxMaskWords) {
      throw std::runtime_error("invalid LayerKV expert multi-group shape");
    }

    const DLDevice device = device_.unwrap();
    const auto route_blocks = div_ceil(
        static_cast<size_t>(num_rows),
        static_cast<size_t>(kRouteMaskBlockSize));
    LaunchKernel(route_blocks, kRouteMaskBlockSize, device)(
        expert_route_mask_multi_kernel<T>,
        static_cast<const T*>(topk_ids.data_ptr()),
        static_cast<int64_t*>(route_masks.data_ptr()),
        static_cast<int32_t*>(status.data_ptr()),
        static_cast<const int32_t*>(capacities.data_ptr()),
        num_rows,
        top_k,
        num_capacities,
        full_num_experts,
        num_words);
    LaunchKernel(num_capacities, 1, device)(
        expert_group_multi_kernel,
        static_cast<const int64_t*>(route_masks.data_ptr()),
        static_cast<int32_t*>(group_ids.data_ptr()),
        static_cast<int64_t*>(group_masks.data_ptr()),
        static_cast<int32_t*>(group_sizes.data_ptr()),
        static_cast<int32_t*>(status.data_ptr()),
        static_cast<const int32_t*>(capacities.data_ptr()),
        num_rows,
        num_capacities,
        num_words);
  }
};

}  // namespace
