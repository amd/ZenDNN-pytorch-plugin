/******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Was sourced from
 * https://github.com/intel/intel-extension-for-pytorch/blob/v2.3.0%2Bcpu/csrc/cpu/aten/kernels/RotaryPositionEmbeddingKnl.cpp
 * IPEX commit ID: f57307d
 ******************************************************************************/

#include "RopeUtils.hpp"

#include <ATen/Tensor.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace zentorch {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
zentorch_rope_impl(at::Tensor &t_in, at::Tensor &t_emb_pos, at::Tensor &t_pos,
                   int64_t N, // N: number of head, H: head size
                   int64_t H, int64_t offset, int64_t rotary_dim,
                   std::string zentorch_op_name) {
  t_in = t_in.contiguous();
  t_emb_pos = t_emb_pos.contiguous();
  t_pos = t_pos.contiguous();
  // only supported types are fp32 and bf16
  if (t_in.scalar_type() == at::kFloat) {
    return zentorch::cpu::kernel::ApplyROPEKernel<float>(
        t_in, t_emb_pos, t_pos, N, H, offset, rotary_dim);
  } else if (t_in.scalar_type() == at::kBFloat16) {
    return zentorch::cpu::kernel::ApplyROPEKernel<at::BFloat16>(
        t_in, t_emb_pos, t_pos, N, H, offset, rotary_dim);
  } else {
    TORCH_CHECK(false, "zentorch_rope_impl: unsupported '", t_in.scalar_type(),
                "'");
    return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
  }
}

} // namespace zentorch
