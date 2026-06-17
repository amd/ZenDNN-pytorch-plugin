/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "../../../Utils.hpp"

#include <ATen/ops/exp.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/tril.h>
#include <ATen/record_function.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace zentorch {

at::Tensor zentorch_gdn_chunk_scaled_dot_kkt_fwd(
    const at::Tensor &k, const at::Tensor &g, const at::Tensor &beta,
    const at::Tensor &cu_seqlens, const at::Tensor &chunk_indices,
    int64_t chunk_size, std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::gdn_chunk_scaled_dot_kkt_fwd",
                  c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(k.dim() == 4, "k must be 4-D (B, T, Hg, K)");
  ZENTORCH_CHECK(beta.dim() == 3, "beta must be 3-D (B, T, H)");
  ZENTORCH_CHECK(g.dim() == 3, "g must be 3-D (B, T, H)");
  ZENTORCH_CHECK(k.size(0) == 1, "B must be 1 (varlen); got ", k.size(0));

  const int64_t B = k.size(0);
  const int64_t T = k.size(1);
  const int64_t Hg = k.size(2);
  const int64_t H = beta.size(2);
  ZENTORCH_CHECK(H % Hg == 0, "H must be a multiple of Hg");
  const int64_t r = H / Hg;
  const int64_t BT = chunk_size;
  ZENTORCH_CHECK(BT > 0 && (BT & (BT - 1)) == 0,
                 "chunk_size must be a positive power of 2; got ", BT);

  ZENTORCH_CHECK(beta.size(0) == B && beta.size(1) == T,
                 "beta must agree with k on (B, T)");
  ZENTORCH_CHECK(g.sizes() == beta.sizes(),
                 "g must have the same shape as beta");

  ZENTORCH_CHECK(at::isFloatingType(k.scalar_type()),
                 "k must be floating-point; got ", k.scalar_type());
  ZENTORCH_CHECK(at::isFloatingType(beta.scalar_type()),
                 "beta must be floating-point; got ", beta.scalar_type());
  ZENTORCH_CHECK(at::isFloatingType(g.scalar_type()),
                 "g must be floating-point; got ", g.scalar_type());

  ZENTORCH_CHECK(cu_seqlens.dim() == 1 &&
                     cu_seqlens.scalar_type() == c10::ScalarType::Int,
                 "cu_seqlens must be 1-D int32");
  ZENTORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
                     chunk_indices.scalar_type() == c10::ScalarType::Int,
                 "chunk_indices must be 2-D int32 (NT, 2)");
  ZENTORCH_CHECK(cu_seqlens.is_contiguous() && chunk_indices.is_contiguous(),
                 "cu_seqlens and chunk_indices must be contiguous");

  const int64_t NT = chunk_indices.size(0);

  at::Tensor A = at::zeros({B, T, H, BT}, k.options().dtype(c10::kFloat));

  if (NT == 0 || H == 0 || T == 0) {
    return A;
  }

  at::Tensor k_f = k.to(c10::kFloat);
  at::Tensor beta_f = beta.to(c10::kFloat);
  at::Tensor g_f = g.to(c10::kFloat);

  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  const int32_t *chunk_indices_p = chunk_indices.const_data_ptr<int32_t>();

  for (int64_t row = 0; row < NT; ++row) {
    const int32_t seq_idx = chunk_indices_p[2 * row + 0];
    const int32_t chunk_idx = chunk_indices_p[2 * row + 1];
    const int64_t bos = cu_seqlens_p[seq_idx];
    const int64_t eos = cu_seqlens_p[seq_idx + 1];
    const int64_t cs_start = bos + chunk_idx * BT;
    const int64_t cs_end = std::min(cs_start + BT, eos);
    const int64_t chunk_len = cs_end - cs_start;
    if (chunk_len <= 0)
      continue;

    at::Tensor tril_mask = at::tril(
        at::ones({chunk_len, chunk_len}, k.options().dtype(c10::kBool)),
        /*diagonal=*/-1);

    for (int64_t h = 0; h < H; ++h) {
      const int64_t kh = h / r;

      at::Tensor k_chunk =
          k_f.select(0, 0).narrow(0, cs_start, chunk_len).select(1, kh);
      at::Tensor beta_chunk =
          beta_f.select(0, 0).narrow(0, cs_start, chunk_len).select(1, h);
      at::Tensor g_chunk =
          g_f.select(0, 0).narrow(0, cs_start, chunk_len).select(1, h);

      // (β · k) @ k.T  with exp(g[i] - g[j]) decay, strict lower triangle.
      at::Tensor kb = k_chunk * beta_chunk.unsqueeze(-1);
      at::Tensor A_chunk = at::matmul(kb, k_chunk.transpose(-1, -2));

      at::Tensor decay = at::exp(g_chunk.unsqueeze(-1) - g_chunk.unsqueeze(0));
      A_chunk = A_chunk * decay;

      A_chunk = at::where(tril_mask, A_chunk, A_chunk.new_zeros({}));

      A.select(0, 0)
          .narrow(0, cs_start, chunk_len)
          .select(1, h)
          .narrow(-1, 0, chunk_len)
          .copy_(A_chunk);
    }
  }

  return A;
}

} // namespace zentorch
