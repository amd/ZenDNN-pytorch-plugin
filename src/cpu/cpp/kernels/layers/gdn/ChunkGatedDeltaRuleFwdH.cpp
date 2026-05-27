/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "../../../Utils.hpp"

#include <ATen/ops/empty.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/zeros.h>
#include <ATen/record_function.h>
#include <c10/util/Optional.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>

namespace zentorch {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
zentorch_gdn_chunk_gated_delta_rule_fwd_h(
    const at::Tensor &k, const at::Tensor &w, const at::Tensor &u,
    const at::Tensor &g, const c10::optional<at::Tensor> &initial_state,
    bool output_final_state, int64_t chunk_size, bool save_new_value,
    const at::Tensor &cu_seqlens, const at::Tensor &chunk_offsets,
    int64_t NT_total, std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::gdn_chunk_gated_delta_rule_fwd_h",
                  c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(k.dim() == 4 && w.dim() == 4 && u.dim() == 4 && g.dim() == 3,
                 "shape ranks: k/w/u=4, g=3");
  ZENTORCH_CHECK(k.size(0) == 1, "B must be 1 (varlen); got ", k.size(0));

  const int64_t B = k.size(0);
  const int64_t T = k.size(1);
  const int64_t Hg = k.size(2);
  const int64_t K_dim = k.size(3);
  const int64_t H = u.size(2);
  const int64_t V_dim = u.size(3);
  const int64_t BT = chunk_size;
  ZENTORCH_CHECK(H % Hg == 0, "H must be a multiple of Hg");
  const int64_t r = H / Hg;
  ZENTORCH_CHECK(BT > 0 && (BT & (BT - 1)) == 0,
                 "chunk_size must be a positive power of 2; got ", BT);

  ZENTORCH_CHECK(at::isFloatingType(k.scalar_type()),
                 "k must be floating-point; got ", k.scalar_type());
  ZENTORCH_CHECK(k.scalar_type() != c10::ScalarType::Half,
                 "fp16 not supported; use fp32 or bf16");
  ZENTORCH_CHECK(u.scalar_type() == k.scalar_type() &&
                     w.scalar_type() == k.scalar_type(),
                 "u/w/k must share dtype");

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
  ZENTORCH_CHECK(NT_total == co_p[N], "NT_total argument (", NT_total,
                 ") must equal chunk_offsets[-1] (", co_p[N], ")");

  at::Tensor h_out = at::zeros({B, NT_total, H, V_dim, K_dim}, k.options());
  at::Tensor v_new = save_new_value ? at::zeros({B, T, H, V_dim}, u.options())
                                    : at::empty({0}, u.options());
  at::Tensor final_state =
      output_final_state
          ? at::zeros({N, H, V_dim, K_dim}, k.options().dtype(c10::kFloat))
          : at::empty({0}, k.options().dtype(c10::kFloat));

  if (NT_total == 0 || H == 0 || N == 0) {
    return std::make_tuple(h_out, v_new, final_state);
  }

  at::Tensor k_f = k.to(c10::kFloat);
  at::Tensor w_f = w.to(c10::kFloat);
  at::Tensor u_f = u.to(c10::kFloat);
  at::Tensor g_f = g.to(c10::kFloat);
  c10::optional<at::Tensor> initial_state_f;
  if (initial_state.has_value() && initial_state->numel() > 0) {
    initial_state_f = initial_state->to(c10::kFloat);
  }

  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  const auto k_dtype = k.scalar_type();
  const auto u_dtype = u.scalar_type();

  for (int64_t n = 0; n < N; ++n) {
    const int64_t bos = cu_seqlens_p[n];
    const int64_t eos = cu_seqlens_p[n + 1];
    const int64_t boh = co_p[n];
    const int64_t chunks_in_seq = co_p[n + 1] - boh;

    for (int64_t h_idx = 0; h_idx < H; ++h_idx) {
      const int64_t kh = h_idx / r;

      at::Tensor state;
      if (initial_state_f.has_value()) {
        state = initial_state_f->select(0, n).select(0, h_idx).clone();
      } else {
        state = at::zeros({V_dim, K_dim}, k.options().dtype(c10::kFloat));
      }

      for (int64_t i_t = 0; i_t < chunks_in_seq; ++i_t) {
        const int64_t chunk_start = bos + i_t * BT;
        const int64_t chunk_end = std::min(chunk_start + BT, eos);
        const int64_t BT_eff = chunk_end - chunk_start;
        if (BT_eff <= 0)
          continue;

        // Snapshot state before this chunk's update.
        h_out.select(0, 0)
            .select(0, boh + i_t)
            .select(0, h_idx)
            .copy_(state.to(k_dtype));

        // Value correction: v_corr = u - w @ state.T
        at::Tensor w_block =
            w_f.select(0, 0).narrow(0, chunk_start, BT_eff).select(1, h_idx);
        at::Tensor u_block =
            u_f.select(0, 0).narrow(0, chunk_start, BT_eff).select(1, h_idx);
        at::Tensor v_corr =
            u_block - at::matmul(w_block, state.transpose(-1, -2));

        if (save_new_value) {
          v_new.select(0, 0)
              .narrow(0, chunk_start, BT_eff)
              .select(1, h_idx)
              .copy_(v_corr.to(u_dtype));
        }

        // Per-token + bulk decay.
        at::Tensor g_block =
            g_f.select(0, 0).narrow(0, chunk_start, BT_eff).select(1, h_idx);
        at::Tensor g_last = g_block.select(0, BT_eff - 1);
        v_corr = v_corr * at::exp(g_last - g_block).unsqueeze(-1);
        state = state * at::exp(g_last);

        // State update: state += v_corr.T @ k_block.
        at::Tensor k_block =
            k_f.select(0, 0).narrow(0, chunk_start, BT_eff).select(1, kh);
        state = state + at::matmul(v_corr.transpose(-1, -2), k_block);
      }

      if (output_final_state) {
        final_state.select(0, n).select(0, h_idx).copy_(state);
      }
    }
  }

  return std::make_tuple(h_out, v_new, final_state);
}

} // namespace zentorch
