/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "../../../Utils.hpp"

#include <ATen/ops/exp.h>
#include <ATen/ops/matmul.h>
#include <ATen/record_function.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>

namespace zentorch {

std::tuple<at::Tensor, at::Tensor> zentorch_gdn_recompute_w_u_fwd(
    const at::Tensor &k, const at::Tensor &v, const at::Tensor &beta,
    const at::Tensor &g_cumsum, const at::Tensor &A,
    const at::Tensor &cu_seqlens, const at::Tensor &chunk_indices,
    std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::gdn_recompute_w_u_fwd",
                  c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(k.dim() == 4 && v.dim() == 4 && A.dim() == 4 &&
                     beta.dim() == 3 && g_cumsum.dim() == 3,
                 "shape ranks: k/v/A=4, beta/g=3");
  ZENTORCH_CHECK(k.size(0) == 1, "B must be 1 (varlen); got ", k.size(0));

  const int64_t B = k.size(0);
  const int64_t T = k.size(1);
  const int64_t Hg = k.size(2);
  const int64_t K_dim = k.size(3);
  const int64_t H = v.size(2);
  const int64_t V_dim = v.size(3);
  const int64_t BT = A.size(3);

  ZENTORCH_CHECK(H % Hg == 0, "H must be a multiple of Hg");
  const int64_t r = H / Hg;
  ZENTORCH_CHECK(beta.sizes() == g_cumsum.sizes(),
                 "beta and g must agree on shape");
  ZENTORCH_CHECK(beta.size(2) == H, "beta last dim must equal H=", H);
  ZENTORCH_CHECK(A.size(0) == B && A.size(1) == T && A.size(2) == H,
                 "A must be (B, T, H, BT)");

  ZENTORCH_CHECK(at::isFloatingType(k.scalar_type()),
                 "k must be floating-point; got ", k.scalar_type());
  ZENTORCH_CHECK(k.scalar_type() != c10::ScalarType::Half,
                 "fp16 not supported; use fp32 or bf16");
  ZENTORCH_CHECK(v.scalar_type() == k.scalar_type(),
                 "v dtype must match k dtype");
  ZENTORCH_CHECK(at::isFloatingType(beta.scalar_type()),
                 "beta must be floating-point; got ", beta.scalar_type());

  ZENTORCH_CHECK(cu_seqlens.dim() == 1 &&
                     cu_seqlens.scalar_type() == c10::ScalarType::Int,
                 "cu_seqlens must be 1-D int32");
  ZENTORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
                     chunk_indices.scalar_type() == c10::ScalarType::Int,
                 "chunk_indices must be 2-D int32 (NT, 2)");
  ZENTORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");
  ZENTORCH_CHECK(chunk_indices.is_contiguous(),
                 "chunk_indices must be contiguous");

  const int64_t NT = chunk_indices.size(0);

  at::Tensor w = at::zeros({B, T, H, K_dim}, k.options());
  at::Tensor u = at::zeros({B, T, H, V_dim}, v.options());
  if (NT == 0 || H == 0 || T == 0)
    return std::make_tuple(w, u);

  at::Tensor k_f = k.to(c10::kFloat);
  at::Tensor v_f = v.to(c10::kFloat);
  at::Tensor beta_f = beta.to(c10::kFloat);
  at::Tensor g_f = g_cumsum.to(c10::kFloat);
  at::Tensor A_f = A.to(c10::kFloat);

  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  const int32_t *chunk_indices_p = chunk_indices.const_data_ptr<int32_t>();

  for (int64_t row = 0; row < NT; ++row) {
    const int32_t seq_idx = chunk_indices_p[2 * row + 0];
    const int32_t chunk_idx = chunk_indices_p[2 * row + 1];
    const int64_t bos = cu_seqlens_p[seq_idx];
    const int64_t eos = cu_seqlens_p[seq_idx + 1];
    const int64_t cs_start = bos + chunk_idx * BT;
    const int64_t cs_end = std::min(cs_start + BT, eos);
    const int64_t BT_eff = cs_end - cs_start;
    if (BT_eff <= 0)
      continue;

    for (int64_t h = 0; h < H; ++h) {
      const int64_t kh = h / r;

      at::Tensor A_block = A_f.select(0, 0)
                               .narrow(0, cs_start, BT_eff)
                               .select(1, h)
                               .narrow(-1, 0, BT_eff);
      at::Tensor v_block =
          v_f.select(0, 0).narrow(0, cs_start, BT_eff).select(1, h);
      at::Tensor k_block =
          k_f.select(0, 0).narrow(0, cs_start, BT_eff).select(1, kh);
      at::Tensor beta_block =
          beta_f.select(0, 0).narrow(0, cs_start, BT_eff).select(1, h);
      at::Tensor g_block =
          g_f.select(0, 0).narrow(0, cs_start, BT_eff).select(1, h);

      // u = A @ (β · v)
      at::Tensor vb = v_block * beta_block.unsqueeze(-1);
      at::Tensor u_block = at::matmul(A_block, vb);

      // w = A @ (β · exp(g) · k)
      at::Tensor kb =
          k_block * beta_block.unsqueeze(-1) * at::exp(g_block).unsqueeze(-1);
      at::Tensor w_block = at::matmul(A_block, kb);

      u.select(0, 0).narrow(0, cs_start, BT_eff).select(1, h).copy_(u_block);
      w.select(0, 0).narrow(0, cs_start, BT_eff).select(1, h).copy_(w_block);
    }
  }

  return std::make_tuple(w, u);
}

} // namespace zentorch
