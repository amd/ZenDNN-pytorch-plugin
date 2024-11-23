/******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Was sourced from
 * https://github.com/intel/intel-extension-for-pytorch/blob/v2.4.0%2Bcpu/csrc/cpu/aten/kernels/MoEKrnl.cpp
 * IPEX commit ID: 070f1d7
 ******************************************************************************/
#include <torch/all.h>
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR > 3
#include "Utils.hpp"
#include <ATen/cpu/vec/vec.h>

namespace zentorch {
at::Tensor fuse_index_mul_index_add(const at::Tensor &curr_state,
                                    const at::Tensor &top_x,
                                    const at::Tensor &idx,
                                    const at::Tensor &routing_weights,
                                    const at::Tensor &output,
                                    std::string zentorch_op_name) {
  if (curr_state.scalar_type() != at::ScalarType::BFloat16) {
    ZENTORCH_CHECK(false, "zentorch::fuse_index_mul_index_add supports"
                          " only bfloat16 datatype");
  }
  using lpVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  auto vec_size = lpVec::size();
  auto topx_s0 = top_x.size(0);
  auto *output_ptr = output.data_ptr<at::BFloat16>();
  auto *curr_state_ptr = curr_state.data_ptr<at::BFloat16>();
  auto *routing_weights_ptr = routing_weights.data_ptr<at::BFloat16>();
  auto *top_x_ptr = top_x.data_ptr<int64_t>();
  auto *idx_ptr = idx.data_ptr<int64_t>();

  int64_t output_stride0 = output.stride(0);
  int64_t output_stride1 = output.stride(1);
  int64_t curr_state_size2 = curr_state.size(2);
  int64_t curr_state_stride1 = curr_state.stride(1);
  int64_t curr_state_stride2 = curr_state.stride(2);
  int64_t routing_weights_stride0 = routing_weights.stride(0);
  int64_t routing_weights_stride1 = routing_weights.stride(1);
#pragma omp parallel for
  for (int i = 0; i < topx_s0; ++i) {
    int64_t rw_index = top_x_ptr[i] * routing_weights_stride0 +
                       idx_ptr[i] * routing_weights_stride1;
    auto rw_v = lpVec(static_cast<at::BFloat16>(routing_weights_ptr[rw_index]));
    for (int j = 0; j < curr_state_size2 - (curr_state_size2 % vec_size);
         j += vec_size) {
      int64_t cs_index = i * curr_state_stride1 + j * curr_state_stride2;
      int64_t output_index = top_x_ptr[i] * output_stride0 + j * output_stride1;
      auto cs_v = lpVec::loadu(curr_state_ptr + cs_index);
      auto out_v = lpVec::loadu(output_ptr + output_index);
      fVec rw_v1, rw_v2, cs_v1, cs_v2, out_v1, out_v2;
      std::tie(rw_v1, rw_v2) = at::vec::convert_to_float(rw_v);
      std::tie(cs_v1, cs_v2) = at::vec::convert_to_float(cs_v);
      std::tie(out_v1, out_v2) = at::vec::convert_to_float(out_v);
      auto output_v1 = out_v1 + cs_v1 * rw_v1;
      auto output_v2 = out_v2 + cs_v2 * rw_v2;
      at::vec::convert_from_float<at::BFloat16>(output_v1, output_v2)
          .store(output_ptr + output_index);
    }
    for (int j = curr_state_size2 - (curr_state_size2 % vec_size);
         j < curr_state_size2; ++j) {
      int64_t cs_index = i * curr_state_stride1 + j * curr_state_stride2;
      int64_t output_index = top_x_ptr[i] * output_stride0 + j * output_stride1;
      output_ptr[output_index] +=
          routing_weights_ptr[rw_index] * curr_state_ptr[cs_index];
    }
  }
  return output;
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("fuse_index_mul_index_add(Tensor curr_state, Tensor top_x, "
        "Tensor idx, "
        "Tensor routing_weights, Tensor output, "
        "str zentorch_op_name='zentorch::fuse_index_mul_index_add') -> "
        "Tensor");
}
TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("fuse_index_mul_index_add", fuse_index_mul_index_add);
}
} // namespace zentorch
#endif
