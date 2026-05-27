/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "../../../Utils.hpp"

#include <ATen/ops/exp.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/softplus.h>
#include <ATen/record_function.h>
#include <torch/all.h>

#include <string>
#include <tuple>

namespace zentorch {

constexpr double kL2NormEps = 1e-6;
constexpr double kSoftplusThreshold = 20.0;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
zentorch_gdn_fused_post_conv_prep(
    const at::Tensor &conv_output, const at::Tensor &a, const at::Tensor &b,
    const at::Tensor &A_log, const at::Tensor &dt_bias, int64_t num_k_heads,
    int64_t head_k_dim, int64_t head_v_dim, bool apply_l2norm,
    bool output_g_exp, std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::gdn_fused_post_conv_prep",
                  c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(
      conv_output.dim() == 2,
      "conv_output must be 2-D (L, qkv_dim); got dim=", conv_output.dim());
  const int64_t L = conv_output.size(0);
  const int64_t qkv_dim = conv_output.size(1);
  const int64_t H = num_k_heads;
  const int64_t K = head_k_dim;
  const int64_t V = head_v_dim;
  ZENTORCH_CHECK(A_log.dim() == 1, "A_log must be 1-D (HV,)");
  const int64_t HV = A_log.size(0);
  ZENTORCH_CHECK(qkv_dim == 2 * H * K + HV * V, "qkv_dim=", qkv_dim,
                 " != 2*H*K + HV*V");
  ZENTORCH_CHECK(a.dim() == 2 && a.size(0) == L && a.size(1) == HV,
                 "a must be (L, HV)");
  ZENTORCH_CHECK(b.dim() == 2 && b.size(0) == L && b.size(1) == HV,
                 "b must be (L, HV)");
  ZENTORCH_CHECK(dt_bias.dim() == 1 && dt_bias.size(0) == HV,
                 "dt_bias must be 1-D of size HV=", HV);

  ZENTORCH_CHECK(at::isFloatingType(conv_output.scalar_type()),
                 "conv_output must be floating-point; got ",
                 conv_output.scalar_type());
  ZENTORCH_CHECK(conv_output.scalar_type() != c10::ScalarType::Half,
                 "fp16 not supported; use fp32 or bf16");
  ZENTORCH_CHECK(a.scalar_type() == conv_output.scalar_type() &&
                     b.scalar_type() == conv_output.scalar_type(),
                 "a/b must share dtype with conv_output");
  ZENTORCH_CHECK(at::isFloatingType(A_log.scalar_type()),
                 "A_log must be floating-point");
  ZENTORCH_CHECK(at::isFloatingType(dt_bias.scalar_type()),
                 "dt_bias must be floating-point");

  const auto model_dtype = conv_output.scalar_type();

  if (L == 0) {
    auto opts_model = conv_output.options();
    auto opts_fp32 = conv_output.options().dtype(c10::kFloat);
    return std::make_tuple(
        at::empty({0, H, K}, opts_model), at::empty({0, H, K}, opts_model),
        at::empty({0, HV, V}, opts_model), at::empty({0, HV}, opts_fp32),
        at::empty({0, HV}, opts_fp32));
  }

  at::Tensor q_view = conv_output.narrow(1, 0, H * K).view({L, H, K});
  at::Tensor k_view = conv_output.narrow(1, H * K, H * K).view({L, H, K});
  at::Tensor v_view = conv_output.narrow(1, 2 * H * K, HV * V).view({L, HV, V});

  at::Tensor q;
  at::Tensor k;
  if (apply_l2norm) {
    at::Tensor q_f = q_view.to(c10::kFloat);
    at::Tensor k_f = k_view.to(c10::kFloat);
    at::Tensor q_inv =
        at::rsqrt(q_f.pow(2).sum(/*dim=*/-1, /*keepdim=*/true) + kL2NormEps);
    at::Tensor k_inv =
        at::rsqrt(k_f.pow(2).sum(/*dim=*/-1, /*keepdim=*/true) + kL2NormEps);
    q = (q_f * q_inv).to(model_dtype);
    k = (k_f * k_inv).to(model_dtype);
  } else {
    q = q_view.to(c10::kFloat).to(model_dtype);
    k = k_view.to(c10::kFloat).to(model_dtype);
  }

  at::Tensor v = v_view.contiguous();
  q = q.contiguous();
  k = k.contiguous();

  at::Tensor a_f = a.to(c10::kFloat);
  at::Tensor b_f = b.to(c10::kFloat);
  at::Tensor A_log_f = A_log.to(c10::kFloat);
  at::Tensor dt_bias_f = dt_bias.to(c10::kFloat);

  at::Tensor sp = at::softplus(a_f + dt_bias_f, /*beta=*/1.0,
                               /*threshold=*/kSoftplusThreshold);
  at::Tensor g = -at::exp(A_log_f) * sp;
  if (output_g_exp) {
    g = at::exp(g);
  }
  at::Tensor beta = at::sigmoid(b_f);

  return std::make_tuple(q, k, v, g.contiguous(), beta.contiguous());
}

} // namespace zentorch
