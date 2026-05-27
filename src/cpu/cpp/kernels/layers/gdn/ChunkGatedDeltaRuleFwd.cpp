/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "../../../Utils.hpp"

#include <ATen/EmptyTensor.h>
#include <ATen/Parallel.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/linalg_solve_triangular.h>
#include <ATen/record_function.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Optional.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace zentorch {

// Defined in `Matmul.cpp`; both end up in libzentorch.so.
at::Tensor zentorch_bmm(const at::Tensor &self, const at::Tensor &mat2,
                        std::string zentorch_op_name);

namespace {

inline at::Tensor empty_contiguous_cpu(at::IntArrayRef sizes,
                                       const at::TensorOptions &options) {
  std::vector<int64_t> strides(sizes.size());
  int64_t s = 1;
  for (int64_t i = static_cast<int64_t>(sizes.size()) - 1; i >= 0; --i) {
    strides[i] = s;
    s *= sizes[i];
  }
  return at::detail::empty_strided_cpu(sizes, strides, options);
}

inline at::Tensor gqa_expand_chunk_3d(const at::Tensor &src_b, int64_t cs_start,
                                      int64_t BT_eff, int64_t r) {
  at::Tensor sliced_t = src_b.narrow(0, cs_start, BT_eff).transpose(0, 1);
  if (r == 1) {
    return sliced_t;
  }
  return sliced_t.repeat_interleave(r, /*dim=*/0);
}

inline at::Tensor slice_per_head_2d(const at::Tensor &src_b, int64_t cs_start,
                                    int64_t BT_eff) {
  return src_b.narrow(0, cs_start, BT_eff).transpose(0, 1);
}

template <typename g_t>
void run_g_cumsum(const at::Tensor &g, const at::Tensor &cu_seqlens,
                  const at::Tensor &chunk_indices, int64_t BT,
                  at::Tensor &g_cum, int64_t HV) {
  const int64_t NT = chunk_indices.size(0);
  const int64_t g_stride_t = g.stride(1);
  const int64_t g_stride_h = g.stride(2);
  const int64_t out_stride_t = g_cum.stride(1);
  const int64_t out_stride_h = g_cum.stride(2);

  const g_t *g_base = g.const_data_ptr<g_t>();
  float *out_base = g_cum.data_ptr<float>();
  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  const int32_t *chunk_indices_p = chunk_indices.const_data_ptr<int32_t>();

  at::parallel_for(0, NT, /*grain_size=*/1, [&](int64_t start, int64_t end) {
    for (int64_t r = start; r < end; ++r) {
      const int32_t seq_idx = chunk_indices_p[2 * r + 0];
      const int32_t chunk_idx = chunk_indices_p[2 * r + 1];
      const int64_t bos = cu_seqlens_p[seq_idx];
      const int64_t eos = cu_seqlens_p[seq_idx + 1];
      const int64_t cs_start = bos + chunk_idx * BT;
      const int64_t cs_end = std::min(cs_start + BT, eos);
      if (cs_start >= cs_end)
        continue;

      const g_t *g_seq_base = g_base + bos * g_stride_t;
      float *out_seq_base = out_base + bos * out_stride_t;
      const int64_t t0 = cs_start - bos;
      const int64_t t1 = cs_end - bos;

      for (int64_t h = 0; h < HV; ++h) {
        const g_t *g_ptr = g_seq_base + h * g_stride_h;
        float *out_ptr = out_seq_base + h * out_stride_h;
        float acc = 0.0f;
        for (int64_t t = t0; t < t1; ++t) {
          acc += static_cast<float>(g_ptr[t * g_stride_t]);
          out_ptr[t * out_stride_t] = acc;
        }
      }
    }
  });
}

// Fused: chunk_scaled_dot_kkt_fwd → solve_tril → recompute_w_u_fwd.
void run_recompute_w_u_fused(const at::Tensor &k_f, const at::Tensor &v_f,
                             const at::Tensor &beta_f, const at::Tensor &g_cum,
                             const at::Tensor &cu_seqlens,
                             const at::Tensor &chunk_indices, int64_t BT,
                             int64_t H, int64_t r, at::Tensor &w_f,
                             at::Tensor &u_f) {
  const int64_t NT = chunk_indices.size(0);
  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  const int32_t *chunk_indices_p = chunk_indices.const_data_ptr<int32_t>();

  at::Tensor I_max = at::eye(BT, k_f.options());

  at::Tensor k_b = k_f.select(0, 0);
  at::Tensor v_b = v_f.select(0, 0);
  at::Tensor beta_b = beta_f.select(0, 0);
  at::Tensor g_b = g_cum.select(0, 0);
  at::Tensor w_b = w_f.select(0, 0);
  at::Tensor u_b = u_f.select(0, 0);

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

    at::Tensor k_h = gqa_expand_chunk_3d(k_b, cs_start, BT_eff, r);
    at::Tensor v_h = gqa_expand_chunk_3d(v_b, cs_start, BT_eff, /*r=*/1);
    at::Tensor beta_h = slice_per_head_2d(beta_b, cs_start, BT_eff);
    at::Tensor g_h = slice_per_head_2d(g_b, cs_start, BT_eff);

    // (1, BT_eff, BT_eff) broadcasts against A_batch's H batch dim.
    at::Tensor I_eff_batch =
        I_max.narrow(0, 0, BT_eff).narrow(1, 0, BT_eff).unsqueeze(0);

    // A_batch = bmm(β · k, k.T) with exp(g[i] - g[j]) decay.
    at::Tensor kb_batch = k_h * beta_h.unsqueeze(-1);
    at::Tensor A_batch =
        zentorch_bmm(kb_batch, k_h.transpose(-1, -2), "zentorch::zentorch_bmm");
    A_batch = A_batch * at::exp(g_h.unsqueeze(-1) - g_h.unsqueeze(-2));

    // solve_tril via unit-triangular solve.
    at::Tensor A_solved =
        at::linalg_solve_triangular(A_batch, I_eff_batch,
                                    /*upper=*/false, /*left=*/true,
                                    /*unitriangular=*/true);

    // u = A_solved @ (β · v);  w = A_solved @ (β · exp(g) · k).
    at::Tensor vb_batch = v_h * beta_h.unsqueeze(-1);
    at::Tensor u_batch =
        zentorch_bmm(A_solved, vb_batch, "zentorch::zentorch_bmm");
    at::Tensor kb2_batch =
        k_h * beta_h.unsqueeze(-1) * at::exp(g_h).unsqueeze(-1);
    at::Tensor w_batch =
        zentorch_bmm(A_solved, kb2_batch, "zentorch::zentorch_bmm");

    u_b.narrow(0, cs_start, BT_eff).copy_(u_batch.transpose(0, 1));
    w_b.narrow(0, cs_start, BT_eff).copy_(w_batch.transpose(0, 1));
  }
}

void run_chunk_recurrent_state(const at::Tensor &k_f, const at::Tensor &w_f,
                               const at::Tensor &u_f, const at::Tensor &g_cum,
                               const c10::optional<at::Tensor> &initial_state_f,
                               const at::Tensor &cu_seqlens,
                               const at::Tensor &chunk_offsets_long, int64_t BT,
                               int64_t H, int64_t r, int64_t V_dim,
                               int64_t K_dim, bool output_final_state,
                               at::Tensor &h_out_f, at::Tensor &v_new_f,
                               at::Tensor &final_state) {
  const int64_t N = cu_seqlens.size(0) - 1;
  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  const int64_t *co_p = chunk_offsets_long.const_data_ptr<int64_t>();

  at::Tensor k_b = k_f.select(0, 0);
  at::Tensor w_b = w_f.select(0, 0);
  at::Tensor u_b = u_f.select(0, 0);
  at::Tensor g_b = g_cum.select(0, 0);
  at::Tensor h_b = h_out_f.select(0, 0);
  at::Tensor vn_b = v_new_f.select(0, 0);

  for (int64_t n = 0; n < N; ++n) {
    const int64_t bos = cu_seqlens_p[n];
    const int64_t eos = cu_seqlens_p[n + 1];
    const int64_t boh = co_p[n];
    const int64_t chunks_in_seq = co_p[n + 1] - boh;

    at::Tensor state_batch;
    if (initial_state_f.has_value()) {
      state_batch = initial_state_f->select(0, n).clone();
    } else {
      state_batch = empty_contiguous_cpu({H, V_dim, K_dim}, k_f.options());
      state_batch.zero_();
    }

    for (int64_t i_t = 0; i_t < chunks_in_seq; ++i_t) {
      const int64_t chunk_start = bos + i_t * BT;
      const int64_t chunk_end = std::min(chunk_start + BT, eos);
      const int64_t BT_eff = chunk_end - chunk_start;
      if (BT_eff <= 0)
        continue;

      // Snapshot state into h_out[boh + i_t].
      h_b.select(0, boh + i_t).copy_(state_batch);

      at::Tensor w_h = gqa_expand_chunk_3d(w_b, chunk_start, BT_eff, /*r=*/1);
      at::Tensor u_h = gqa_expand_chunk_3d(u_b, chunk_start, BT_eff, /*r=*/1);
      at::Tensor k_h = gqa_expand_chunk_3d(k_b, chunk_start, BT_eff, r);
      at::Tensor g_h = slice_per_head_2d(g_b, chunk_start, BT_eff);

      // v_corr = u - bmm(w, state.T)
      at::Tensor state_T = state_batch.transpose(-1, -2);
      at::Tensor v_corr_batch =
          u_h - zentorch_bmm(w_h, state_T, "zentorch::zentorch_bmm");

      // Save pre-decay v_new for chunk_fwd_o.
      vn_b.narrow(0, chunk_start, BT_eff).copy_(v_corr_batch.transpose(0, 1));

      // Per-token + bulk decay.
      at::Tensor g_last_h = g_h.select(-1, BT_eff - 1);
      v_corr_batch =
          v_corr_batch * at::exp(g_last_h.unsqueeze(-1) - g_h).unsqueeze(-1);
      state_batch = state_batch * at::exp(g_last_h).unsqueeze(-1).unsqueeze(-1);

      // state += bmm(v_corr.T, k_h).
      state_batch = state_batch + zentorch_bmm(v_corr_batch.transpose(-1, -2),
                                               k_h, "zentorch::zentorch_bmm");
    }

    if (output_final_state) {
      final_state.select(0, n).copy_(state_batch);
    }
  }
}

void run_chunk_output(const at::Tensor &q_f, const at::Tensor &k_f,
                      const at::Tensor &v_new_f, const at::Tensor &h_out_f,
                      const at::Tensor &g_cum, const at::Tensor &cu_seqlens,
                      const at::Tensor &chunk_offsets_long, int64_t BT,
                      int64_t H, int64_t r, float scale_f, at::Tensor &o,
                      c10::ScalarType out_dtype) {
  const int64_t N = cu_seqlens.size(0) - 1;
  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  const int64_t *co_p = chunk_offsets_long.const_data_ptr<int64_t>();

  at::Tensor q_b = q_f.select(0, 0);
  at::Tensor k_b = k_f.select(0, 0);
  at::Tensor v_b = v_new_f.select(0, 0);
  at::Tensor h_b = h_out_f.select(0, 0);
  at::Tensor g_b = g_cum.select(0, 0);
  at::Tensor o_b = o.select(0, 0);

  at::Tensor idx_max = at::arange(BT, q_f.options().dtype(c10::kLong));
  at::Tensor causal_max = idx_max.unsqueeze(-1).ge(idx_max.unsqueeze(0));

  for (int64_t n = 0; n < N; ++n) {
    const int64_t bos = cu_seqlens_p[n];
    const int64_t eos = cu_seqlens_p[n + 1];
    const int64_t boh = co_p[n];
    const int64_t chunks_in_seq = co_p[n + 1] - boh;

    for (int64_t i_t = 0; i_t < chunks_in_seq; ++i_t) {
      const int64_t chunk_start = bos + i_t * BT;
      const int64_t chunk_end = std::min(chunk_start + BT, eos);
      const int64_t BT_eff = chunk_end - chunk_start;
      if (BT_eff <= 0)
        continue;

      at::Tensor q_h = gqa_expand_chunk_3d(q_b, chunk_start, BT_eff, r);
      at::Tensor k_h = gqa_expand_chunk_3d(k_b, chunk_start, BT_eff, r);
      at::Tensor v_h = gqa_expand_chunk_3d(v_b, chunk_start, BT_eff, /*r=*/1);
      at::Tensor g_h = slice_per_head_2d(g_b, chunk_start, BT_eff);
      at::Tensor h_chunk_batch = h_b.select(0, boh + i_t);

      // History contribution with per-head bulk decay exp(g_h).
      at::Tensor o_history_batch = zentorch_bmm(
          q_h, h_chunk_batch.transpose(-1, -2), "zentorch::zentorch_bmm");
      o_history_batch = o_history_batch * at::exp(g_h).unsqueeze(-1);

      // In-chunk attention with exp(g[i] - g[j]) decay and causal mask.
      at::Tensor A_batch =
          zentorch_bmm(q_h, k_h.transpose(-1, -2), "zentorch::zentorch_bmm");
      A_batch = A_batch * at::exp(g_h.unsqueeze(-1) - g_h.unsqueeze(-2));

      at::Tensor causal_eff =
          causal_max.narrow(0, 0, BT_eff).narrow(1, 0, BT_eff);
      A_batch = at::where(causal_eff, A_batch, A_batch.new_zeros({}));

      at::Tensor o_in_chunk_batch =
          zentorch_bmm(A_batch, v_h, "zentorch::zentorch_bmm");

      at::Tensor o_block_batch = (o_history_batch + o_in_chunk_batch) * scale_f;

      o_b.narrow(0, chunk_start, BT_eff)
          .copy_(o_block_batch.transpose(0, 1).to(out_dtype));
    }
  }
}

} // namespace

std::tuple<at::Tensor, at::Tensor> zentorch_gdn_chunk_gated_delta_rule_fwd(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &g, const at::Tensor &beta, double scale,
    const c10::optional<at::Tensor> &initial_state, bool output_final_state,
    int64_t chunk_size, const at::Tensor &cu_seqlens,
    const at::Tensor &chunk_indices, const at::Tensor &chunk_offsets,
    std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::gdn_chunk_gated_delta_rule_fwd",
                  c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                 "q/k/v must be 4-D");
  ZENTORCH_CHECK(g.dim() == 3 && beta.dim() == 3,
                 "g and beta must be 3-D (B, T, H)");
  ZENTORCH_CHECK(q.size(0) == 1, "B must be 1 (varlen); got ", q.size(0));

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t Hg = q.size(2);
  const int64_t K_dim = q.size(3);
  const int64_t H = v.size(2);
  const int64_t V_dim = v.size(3);
  const int64_t BT = chunk_size;

  ZENTORCH_CHECK(k.size(0) == B && k.size(1) == T && k.size(2) == Hg &&
                     k.size(3) == K_dim,
                 "k must match q on (B, T, Hg, K)");
  ZENTORCH_CHECK(v.size(0) == B && v.size(1) == T,
                 "v must agree with q on (B, T)");
  ZENTORCH_CHECK(H % Hg == 0, "H must be a multiple of Hg");
  const int64_t r = H / Hg;
  ZENTORCH_CHECK(BT == 16 || BT == 32 || BT == 64,
                 "chunk_size must be one of {16, 32, 64}; got ", BT);
  ZENTORCH_CHECK(g.sizes() == beta.sizes(),
                 "g and beta must have the same shape");
  ZENTORCH_CHECK(g.size(0) == B && g.size(1) == T && g.size(2) == H,
                 "g must be (B, T, H)");

  ZENTORCH_CHECK(at::isFloatingType(q.scalar_type()),
                 "q must be floating-point; got ", q.scalar_type());
  ZENTORCH_CHECK(q.scalar_type() != c10::ScalarType::Half,
                 "fp16 not supported; use fp32 or bf16");
  ZENTORCH_CHECK(k.scalar_type() == q.scalar_type() &&
                     v.scalar_type() == q.scalar_type(),
                 "q, k, v must share dtype");
  ZENTORCH_CHECK(at::isFloatingType(g.scalar_type()),
                 "g must be floating-point");
  ZENTORCH_CHECK(at::isFloatingType(beta.scalar_type()),
                 "beta must be floating-point");

  ZENTORCH_CHECK(cu_seqlens.dim() == 1 &&
                     cu_seqlens.scalar_type() == c10::ScalarType::Int,
                 "cu_seqlens must be 1-D int32");
  ZENTORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");
  ZENTORCH_CHECK(cu_seqlens.size(0) >= 2,
                 "cu_seqlens must have at least 2 entries");
  ZENTORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
                     chunk_indices.scalar_type() == c10::ScalarType::Int,
                 "chunk_indices must be 2-D int32 (NT, 2)");
  ZENTORCH_CHECK(chunk_indices.is_contiguous(),
                 "chunk_indices must be contiguous");
  ZENTORCH_CHECK(chunk_offsets.dim() == 1 &&
                     (chunk_offsets.scalar_type() == c10::ScalarType::Int ||
                      chunk_offsets.scalar_type() == c10::ScalarType::Long),
                 "chunk_offsets must be 1-D int32 or int64");
  ZENTORCH_CHECK(chunk_offsets.size(0) == cu_seqlens.size(0),
                 "chunk_offsets.size(0)=", chunk_offsets.size(0),
                 " must equal cu_seqlens.size(0)=", cu_seqlens.size(0));

  const int64_t N = cu_seqlens.size(0) - 1;
  const int64_t NT = chunk_indices.size(0);

  if (initial_state.has_value() && initial_state->numel() > 0) {
    ZENTORCH_CHECK(initial_state->dim() == 4,
                   "initial_state must be 4-D (N, H, V, K)");
    ZENTORCH_CHECK(initial_state->size(0) == N && initial_state->size(1) == H &&
                       initial_state->size(2) == V_dim &&
                       initial_state->size(3) == K_dim,
                   "initial_state shape must be (N, H, V, K)");
  }

  const auto fp32_options = k.options().dtype(c10::kFloat);
  at::Tensor o = empty_contiguous_cpu({B, T, H, V_dim}, v.options());
  at::Tensor final_state =
      output_final_state
          ? empty_contiguous_cpu({N, H, V_dim, K_dim}, fp32_options)
          : at::detail::empty_strided_cpu({0}, {1}, fp32_options);

  if (NT == 0 || H == 0 || T == 0) {
    o.zero_();
    if (output_final_state) {
      if (initial_state.has_value() && initial_state->numel() > 0) {
        final_state.copy_(initial_state.value());
      } else {
        final_state.zero_();
      }
    }
    return std::make_tuple(o, final_state);
  }

  const auto g_fp32_options = g.options().dtype(c10::kFloat);
  const auto v_fp32_options = v.options().dtype(c10::kFloat);
  at::Tensor g_cum = empty_contiguous_cpu({B, T, H}, g_fp32_options);
  at::Tensor w_f = empty_contiguous_cpu({B, T, H, K_dim}, fp32_options);
  at::Tensor u_f = empty_contiguous_cpu({B, T, H, V_dim}, v_fp32_options);

  at::Tensor chunk_offsets_long = chunk_offsets.to(at::kLong).contiguous();
  const int64_t *co_p = chunk_offsets_long.const_data_ptr<int64_t>();
  const int64_t NT_total = co_p[N];

  at::Tensor h_out_f =
      empty_contiguous_cpu({B, NT_total, H, V_dim, K_dim}, fp32_options);
  at::Tensor v_new_f = empty_contiguous_cpu({B, T, H, V_dim}, v_fp32_options);

  const auto g_dt = g.scalar_type();
  if (g_dt == c10::ScalarType::Float) {
    run_g_cumsum<float>(g, cu_seqlens, chunk_indices, BT, g_cum, H);
  } else if (g_dt == c10::ScalarType::BFloat16) {
    run_g_cumsum<c10::BFloat16>(g, cu_seqlens, chunk_indices, BT, g_cum, H);
  } else {
    ZENTORCH_CHECK(false, "g dtype must be fp32 or bf16; got ", g_dt);
  }

  at::Tensor k_f = k.to(c10::kFloat);
  at::Tensor v_f = v.to(c10::kFloat);
  at::Tensor beta_f = beta.to(c10::kFloat);
  at::Tensor q_f = q.to(c10::kFloat);
  c10::optional<at::Tensor> initial_state_f;
  if (initial_state.has_value() && initial_state->numel() > 0) {
    initial_state_f = (initial_state->scalar_type() == c10::ScalarType::Float)
                          ? *initial_state
                          : initial_state->to(c10::kFloat);
  }

  run_recompute_w_u_fused(k_f, v_f, beta_f, g_cum, cu_seqlens, chunk_indices,
                          BT, H, r, w_f, u_f);

  run_chunk_recurrent_state(k_f, w_f, u_f, g_cum, initial_state_f, cu_seqlens,
                            chunk_offsets_long, BT, H, r, V_dim, K_dim,
                            output_final_state, h_out_f, v_new_f, final_state);

  const float scale_f = static_cast<float>(scale);
  run_chunk_output(q_f, k_f, v_new_f, h_out_f, g_cum, cu_seqlens,
                   chunk_offsets_long, BT, H, r, scale_f, o, v.scalar_type());

  return std::make_tuple(o, final_state);
}

} // namespace zentorch
