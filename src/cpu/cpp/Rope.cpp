/******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Was sourced from
 * https://github.com/intel/intel-extension-for-pytorch/blob/v2.6.0%2Bcpu/csrc/cpu/aten/kernels/RotaryPositionEmbeddingKnl.cpp
 * IPEX commit ID: 18eeefa
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
    ZENTORCH_CHECK(false, "unsupported '", t_in.scalar_type(), "'");
    return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
zentorch_rope_deepseek_kernel_impl(at::Tensor &q, at::Tensor &kv,
                                   at::Tensor &k_pe, at::Tensor &t_emb_pos,
                                   at::Tensor &t_pos,
                                   int64_t N, // N: number of head, H: head size
                                   int64_t H, int64_t offset,
                                   int64_t rotary_dim) {
  q = q.contiguous();
  kv = kv.contiguous();
  k_pe = k_pe.contiguous();
  t_emb_pos = t_emb_pos.contiguous();
  t_pos = t_pos.contiguous();
  if (q.scalar_type() == at::kFloat) {
    return zentorch::cpu::kernel::ApplyDeepseekROPEKernel<float>(
        q, kv, k_pe, t_emb_pos, t_pos, N, H, offset, rotary_dim);
  } else if (q.scalar_type() == at::kBFloat16) {
    return zentorch::cpu::kernel::ApplyDeepseekROPEKernel<at::BFloat16>(
        q, kv, k_pe, t_emb_pos, t_pos, N, H, offset, rotary_dim);
  } else {
    ZENTORCH_CHECK(false, "unsupported '", q.scalar_type(), "'");
    return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
  }
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_rope(Tensor t_in, Tensor t_emb_pos, Tensor t_pos, int N, int "
        "H, int offset, int rotary_dim, str zentorch_op_name = "
        "'zentorch::zentorch_rope') -> (Tensor, Tensor, Tensor)");
  m.def("zentorch_rope_deepseek(Tensor q, Tensor kv, Tensor k_pe, Tensor "
        "t_emb_pos, Tensor t_pos, int N, int H, int offset, int "
        "rotary_ndims)-> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_rope", zentorch_rope_impl);
  m.impl("zentorch_rope_deepseek", zentorch_rope_deepseek_kernel_impl);
}

} // namespace zentorch
