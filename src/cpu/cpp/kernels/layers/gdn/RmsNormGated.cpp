/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "../../../RMS_norm.hpp"
#include "../../../Utils.hpp"

#include <ATen/ops/sigmoid.h>
#include <ATen/ops/silu.h>
#include <ATen/record_function.h>
#include <torch/all.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <string>

namespace zentorch {

at::Tensor zentorch_gdn_rms_norm_gated(const at::Tensor &x,
                                       const at::Tensor &weight,
                                       const at::Tensor &z, double eps,
                                       std::string activation,
                                       std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::gdn_rms_norm_gated",
                  c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(x.dim() == 2, "x must be 2-D (M, V); got dim=", x.dim());
  const int64_t M = x.size(0);
  const int64_t V = x.size(1);
  ZENTORCH_CHECK(z.dim() == 2 && z.size(0) == M && z.size(1) == V,
                 "z must match x shape (", M, ", ", V, ")");
  ZENTORCH_CHECK(weight.dim() == 1 && weight.size(0) == V,
                 "weight must be 1-D of size V=", V);

  ZENTORCH_CHECK(is_supported_gdn_float(x.scalar_type()),
                 "x must be fp16, bf16, or fp32; got ", x.scalar_type());
  ZENTORCH_CHECK(z.scalar_type() == x.scalar_type(),
                 "z must share dtype with x; got x=", x.scalar_type(),
                 " z=", z.scalar_type());
  ZENTORCH_CHECK(is_supported_gdn_float(weight.scalar_type()),
                 "weight must be fp16, bf16, or fp32; got ",
                 weight.scalar_type());

  const bool is_silu = (activation == "silu" || activation == "swish");
  const bool is_sigmoid = (activation == "sigmoid");
  ZENTORCH_CHECK(is_silu || is_sigmoid,
                 "activation must be 'silu', 'swish', or 'sigmoid'; got '",
                 activation, "'");

  if (M == 0 || V == 0) {
    return at::empty(x.sizes(), x.options());
  }

  auto aten_to_stable = [](const at::Tensor &t) {
    return torch::stable::Tensor(
        torch::aot_inductor::new_tensor_handle(at::Tensor(t)));
  };

  torch::stable::Tensor x_f = torch::stable::contiguous(
      torch::stable::to(aten_to_stable(x), c10::kFloat));
  torch::stable::Tensor w_f = torch::stable::contiguous(
      torch::stable::to(aten_to_stable(weight), c10::kFloat));
  at::Tensor z_f = z.to(c10::kFloat).contiguous();

  torch::stable::Tensor y_stable =
      zentorch::zentorch_rms_norm(x_f, w_f, eps, "zentorch::zentorch_rms_norm");
  at::Tensor y_f =
      *torch::aot_inductor::tensor_handle_to_tensor_pointer(y_stable.get());

  at::Tensor gate_f = is_sigmoid ? at::sigmoid(z_f) : at::silu(z_f);

  return (y_f * gate_f).to(x.scalar_type());
}

} // namespace zentorch
