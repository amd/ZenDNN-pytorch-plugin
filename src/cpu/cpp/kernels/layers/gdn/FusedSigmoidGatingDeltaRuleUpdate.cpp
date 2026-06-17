/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "../../../Utils.hpp"

#include <ATen/ops/empty.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/softplus.h>
#include <ATen/record_function.h>
#include <c10/util/Optional.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace zentorch {

constexpr double kL2NormEps = 1e-6;
constexpr int32_t kNullBlockId = 0;

namespace {

// 1-D indices = plain decode; 2-D indices = spec-decode (per-token slot).
inline int64_t lookup_state_index(const at::Tensor &ssm_state_indices,
                                  int64_t n, int64_t t) {
  if (ssm_state_indices.dim() == 1) {
    return ssm_state_indices.const_data_ptr<int32_t>()[n];
  }
  const int32_t *p = ssm_state_indices.const_data_ptr<int32_t>();
  const int64_t stride0 = ssm_state_indices.stride(0);
  return p[n * stride0 + t];
}

} // anonymous namespace

at::Tensor zentorch_gdn_fused_sigmoid_gating_delta_rule_update(
    const at::Tensor &A_log, const at::Tensor &a, const at::Tensor &b,
    const at::Tensor &dt_bias, const at::Tensor &q, const at::Tensor &k,
    const at::Tensor &v, double beta_temp, double threshold, double scale,
    at::Tensor &initial_state, const at::Tensor &cu_seqlens,
    const at::Tensor &ssm_state_indices,
    const c10::optional<at::Tensor> &num_accepted_tokens,
    bool use_qk_l2norm_in_kernel, std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::gdn_fused_sigmoid_gating_delta_rule_update",
                  c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                 "q/k/v must be 4-D");
  ZENTORCH_CHECK(q.size(0) == 1, "q.size(0) must be 1; got ", q.size(0));

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t H = q.size(2);
  const int64_t K_dim = q.size(3);
  const int64_t HV = v.size(2);
  const int64_t V_dim = v.size(3);
  ZENTORCH_CHECK(k.sizes() == q.sizes(), "k must match q in shape");
  ZENTORCH_CHECK(v.size(0) == B && v.size(1) == T,
                 "v must agree with q on (B, T)");
  ZENTORCH_CHECK(HV % H == 0, "HV must be a multiple of H");
  const int64_t r = HV / H;

  ZENTORCH_CHECK(a.dim() == 3 && a.size(0) == B && a.size(1) == T &&
                     a.size(2) == HV,
                 "a must be (B, T, HV)");
  ZENTORCH_CHECK(b.sizes() == a.sizes(), "b must match a in shape");
  ZENTORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == HV,
                 "A_log must be (HV,)");
  ZENTORCH_CHECK(dt_bias.dim() == 1 && dt_bias.size(0) == HV,
                 "dt_bias must be (HV,)");
  ZENTORCH_CHECK(initial_state.dim() == 4 && initial_state.size(1) == HV &&
                     initial_state.size(2) == V_dim &&
                     initial_state.size(3) == K_dim,
                 "initial_state must be (num_cache_lines, HV, V, K)");

  ZENTORCH_CHECK(at::isFloatingType(q.scalar_type()),
                 "q must be floating-point; got ", q.scalar_type());
  ZENTORCH_CHECK(k.scalar_type() == q.scalar_type() &&
                     v.scalar_type() == q.scalar_type(),
                 "q/k/v must share dtype");
  ZENTORCH_CHECK(a.scalar_type() == q.scalar_type() &&
                     b.scalar_type() == q.scalar_type(),
                 "a/b must share dtype with q");
  ZENTORCH_CHECK(at::isFloatingType(A_log.scalar_type()),
                 "A_log must be floating-point");
  ZENTORCH_CHECK(at::isFloatingType(dt_bias.scalar_type()),
                 "dt_bias must be floating-point");

  ZENTORCH_CHECK(cu_seqlens.dim() == 1 &&
                     cu_seqlens.scalar_type() == c10::ScalarType::Int,
                 "cu_seqlens must be 1-D int32");
  ZENTORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");
  ZENTORCH_CHECK(ssm_state_indices.scalar_type() == c10::ScalarType::Int,
                 "ssm_state_indices must be int32");
  ZENTORCH_CHECK(ssm_state_indices.dim() == 1 || ssm_state_indices.dim() == 2,
                 "ssm_state_indices must be 1-D or 2-D");
  const int64_t N = cu_seqlens.size(0) - 1;
  ZENTORCH_CHECK(ssm_state_indices.size(0) == N,
                 "ssm_state_indices.size(0) must equal N=", N);
  if (num_accepted_tokens.has_value()) {
    ZENTORCH_CHECK(
        num_accepted_tokens->dim() == 1 && num_accepted_tokens->size(0) == N &&
            num_accepted_tokens->scalar_type() == c10::ScalarType::Int,
        "num_accepted_tokens must be 1-D int32 of length N=", N);
    ZENTORCH_CHECK(ssm_state_indices.dim() == 2,
                   "num_accepted_tokens requires 2-D ssm_state_indices");
  }

  // Reject metadata mismatches that would otherwise return an
  // uninitialised buffer to the caller below.
  ZENTORCH_CHECK(N > 0 || B == 0 || T == 0 || HV == 0 || V_dim == 0,
                 "cu_seqlens implies N=0 (size=", cu_seqlens.size(0),
                 ") but output is non-empty: B=", B, " T=", T, " HV=", HV,
                 " V_dim=", V_dim);

  at::Tensor o = at::empty({B, T, HV, V_dim}, q.options());

  if (B == 0 || T == 0 || N == 0 || HV == 0 || V_dim == 0) {
    return o;
  }

  at::Tensor A_log_f = A_log.to(c10::kFloat).contiguous();
  at::Tensor dt_bias_f = dt_bias.to(c10::kFloat).contiguous();
  at::Tensor a_f = a.to(c10::kFloat);
  at::Tensor b_f = b.to(c10::kFloat);
  at::Tensor q_f = q.to(c10::kFloat);
  at::Tensor k_f = k.to(c10::kFloat);
  at::Tensor v_f = v.to(c10::kFloat);

  const float scale_f = static_cast<float>(scale);
  const float threshold_f = static_cast<float>(threshold);
  const auto out_dtype = q.scalar_type();
  const auto state_dtype = initial_state.scalar_type();

  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  at::Tensor ssm_idx = ssm_state_indices.contiguous();
  c10::optional<at::Tensor> num_accepted_contig;
  if (num_accepted_tokens.has_value()) {
    num_accepted_contig = num_accepted_tokens->contiguous();
  }
  const int32_t *num_accepted_p =
      num_accepted_contig.has_value()
          ? num_accepted_contig->const_data_ptr<int32_t>()
          : nullptr;

  for (int64_t n = 0; n < N; ++n) {
    const int64_t bos = cu_seqlens_p[n];
    const int64_t eos = cu_seqlens_p[n + 1];
    const int64_t T_n = eos - bos;
    if (T_n <= 0)
      continue;

    const int64_t i_t0 =
        num_accepted_p ? static_cast<int64_t>(num_accepted_p[n]) - 1 : 0;
    const int64_t init_idx = lookup_state_index(ssm_idx, n, i_t0);
    if (init_idx <= kNullBlockId) {
      // Null-slot contract: output is zero for this sequence's token range
      // (mirrors gdn_fused_recurrent_gated_delta_rule_packed_decode). Without
      // the explicit zero-fill the o = at::empty(...) allocation above would
      // leak uninitialised values for any non-empty skipped sequence.
      o.select(0, 0).narrow(0, bos, T_n).zero_();
      continue;
    }

    // Working state for this sequence (clone so cache aliasing is safe).
    at::Tensor h = initial_state.select(0, init_idx).to(c10::kFloat).clone();

    for (int64_t t = 0; t < T_n; ++t) {
      const int64_t pos = bos + t;

      at::Tensor a_t = a_f.select(0, 0).select(0, pos);
      at::Tensor b_t = b_f.select(0, 0).select(0, pos);
      at::Tensor q_t = q_f.select(0, 0).select(0, pos);
      at::Tensor k_t = k_f.select(0, 0).select(0, pos);
      at::Tensor v_t = v_f.select(0, 0).select(0, pos);

      // Gating.
      at::Tensor x = a_t + dt_bias_f;
      at::Tensor sp = at::softplus(x, /*beta=*/static_cast<double>(beta_temp),
                                   /*threshold=*/threshold_f);
      at::Tensor g_t = -at::exp(A_log_f) * sp;
      at::Tensor beta_out = at::sigmoid(b_t);

      if (use_qk_l2norm_in_kernel) {
        q_t =
            q_t * at::rsqrt(q_t.pow(2).sum(-1, /*keepdim=*/true) + kL2NormEps);
        k_t =
            k_t * at::rsqrt(k_t.pow(2).sum(-1, /*keepdim=*/true) + kL2NormEps);
      }
      q_t = q_t * scale_f;

      // GQA expansion (H, K) → (HV, K).
      at::Tensor q_e = q_t.repeat_interleave(/*repeats=*/r, /*dim=*/0);
      at::Tensor k_e = k_t.repeat_interleave(/*repeats=*/r, /*dim=*/0);

      // Recurrence in (HV, V, K) state layout.
      h = h * at::exp(g_t).reshape({HV, 1, 1});
      at::Tensor kv_mem = (h * k_e.unsqueeze(-2)).sum(-1);
      at::Tensor delta = (v_t - kv_mem) * beta_out.unsqueeze(-1);
      h = h + delta.unsqueeze(-1) * k_e.unsqueeze(-2);
      at::Tensor o_pos = (h * q_e.unsqueeze(-2)).sum(-1);
      o.select(0, 0).select(0, pos).copy_(o_pos.to(out_dtype));

      const int64_t final_idx = lookup_state_index(ssm_idx, n, t);
      if (final_idx > kNullBlockId) {
        initial_state.select(0, final_idx).copy_(h.to(state_dtype));
      }
    }
  }

  return o;
}

} // namespace zentorch
