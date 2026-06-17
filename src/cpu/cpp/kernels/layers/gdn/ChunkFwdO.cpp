/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "../../../Utils.hpp"

#include <ATen/ops/arange.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/matmul.h>
#include <ATen/record_function.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace zentorch {

at::Tensor zentorch_gdn_chunk_fwd_o(const at::Tensor &q, const at::Tensor &k,
                                    const at::Tensor &v, const at::Tensor &h,
                                    const at::Tensor &g, double scale,
                                    const at::Tensor &cu_seqlens,
                                    const at::Tensor &chunk_offsets,
                                    int64_t chunk_size,
                                    std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::gdn_chunk_fwd_o", c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && h.dim() == 5,
                 "shape ranks: q/k/v=4, h=5");
  ZENTORCH_CHECK(g.dim() == 3, "g must be 3-D (B, T, H)");
  ZENTORCH_CHECK(q.size(0) == 1, "B must be 1 (varlen); got ", q.size(0));

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t Hg = q.size(2);
  const int64_t H = v.size(2);
  const int64_t V_dim = v.size(3);
  const int64_t BT = chunk_size;
  ZENTORCH_CHECK(H % Hg == 0, "H must be a multiple of Hg");
  const int64_t r = H / Hg;
  ZENTORCH_CHECK(BT > 0 && (BT & (BT - 1)) == 0,
                 "chunk_size must be a positive power of 2; got ", BT);

  ZENTORCH_CHECK(at::isFloatingType(q.scalar_type()),
                 "q must be floating-point; got ", q.scalar_type());
  ZENTORCH_CHECK(k.scalar_type() == q.scalar_type() &&
                     v.scalar_type() == q.scalar_type(),
                 "q/k/v must share dtype");

  ZENTORCH_CHECK(cu_seqlens.dim() == 1 &&
                     cu_seqlens.scalar_type() == c10::ScalarType::Int,
                 "cu_seqlens must be 1-D int32");
  ZENTORCH_CHECK(chunk_offsets.dim() == 1 &&
                     (chunk_offsets.scalar_type() == c10::ScalarType::Int ||
                      chunk_offsets.scalar_type() == c10::ScalarType::Long),
                 "chunk_offsets must be 1-D int32 or int64");
  ZENTORCH_CHECK(chunk_offsets.size(0) == cu_seqlens.size(0),
                 "chunk_offsets.size(0)=", chunk_offsets.size(0),
                 " must equal cu_seqlens.size(0)=", cu_seqlens.size(0));
  ZENTORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");

  const int64_t N = cu_seqlens.size(0) - 1;
  at::Tensor co_long = chunk_offsets.to(at::kLong).contiguous();
  const int64_t *co_p = co_long.const_data_ptr<int64_t>();

  at::Tensor o = at::zeros({B, T, H, V_dim}, v.options());

  if (N == 0 || H == 0 || T == 0)
    return o;

  at::Tensor q_f = q.to(c10::kFloat);
  at::Tensor k_f = k.to(c10::kFloat);
  at::Tensor v_f = v.to(c10::kFloat);
  at::Tensor h_f = h.to(c10::kFloat);
  at::Tensor g_f = g.to(c10::kFloat);
  const float scale_f = static_cast<float>(scale);

  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  const auto out_dtype = v.scalar_type();

  for (int64_t n = 0; n < N; ++n) {
    const int64_t bos = cu_seqlens_p[n];
    const int64_t eos = cu_seqlens_p[n + 1];
    const int64_t boh = co_p[n];
    const int64_t chunks_in_seq = co_p[n + 1] - boh;

    for (int64_t h_idx = 0; h_idx < H; ++h_idx) {
      const int64_t kh = h_idx / r;

      for (int64_t i_t = 0; i_t < chunks_in_seq; ++i_t) {
        const int64_t chunk_start = bos + i_t * BT;
        const int64_t chunk_end = std::min(chunk_start + BT, eos);
        const int64_t BT_eff = chunk_end - chunk_start;
        if (BT_eff <= 0)
          continue;

        at::Tensor q_block =
            q_f.select(0, 0).narrow(0, chunk_start, BT_eff).select(1, kh);
        at::Tensor k_block =
            k_f.select(0, 0).narrow(0, chunk_start, BT_eff).select(1, kh);
        at::Tensor v_block =
            v_f.select(0, 0).narrow(0, chunk_start, BT_eff).select(1, h_idx);
        at::Tensor h_chunk =
            h_f.select(0, 0).select(0, boh + i_t).select(0, h_idx);
        at::Tensor g_block =
            g_f.select(0, 0).narrow(0, chunk_start, BT_eff).select(1, h_idx);

        // History contribution.
        at::Tensor o_history = at::matmul(q_block, h_chunk.transpose(-1, -2));
        o_history = o_history * at::exp(g_block).unsqueeze(-1);

        // In-chunk attention with exp(g[i] - g[j]) decay and causal mask.
        at::Tensor A = at::matmul(q_block, k_block.transpose(-1, -2));
        A = A * at::exp(g_block.unsqueeze(-1) - g_block.unsqueeze(0));

        at::Tensor idx = at::arange(BT_eff, q.options().dtype(c10::kLong));
        at::Tensor causal = idx.unsqueeze(-1).ge(idx.unsqueeze(0));
        A = at::where(causal, A, A.new_zeros({}));

        at::Tensor o_in_chunk = at::matmul(A, v_block);

        at::Tensor o_block = (o_history + o_in_chunk) * scale_f;
        o.select(0, 0)
            .narrow(0, chunk_start, BT_eff)
            .select(1, h_idx)
            .copy_(o_block.to(out_dtype));
      }
    }
  }

  return o;
}

} // namespace zentorch
