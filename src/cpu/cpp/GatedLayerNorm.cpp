/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

// Gated LayerNorm of DLRM v3 / generative-recommenders (the norm-mul of the
// HSTU output block, `pytorch_norm_mul_dropout` at inference) on the ZenDNN
// LOWOHA FUSED_GATED_LAYER_NORM normalization:
//
//   g   = silu(gate) if silu_gate else gate
//   y   = g * layer_norm(input, weight, bias, eps)        (FP32 math)
//   out = cat([g, input, y], dim=1) if concat else y      (input dtype)
//
// The gate may be a row-strided view (e.g. the u slice of the uvqk GEMM
// output), and with concat the three slices are written in one pass.
// The result is written into the caller-supplied `out`.

#include "Memory.hpp"
#include "Utils.hpp"

#include <cmath>
#include <optional>
#include <string>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

namespace zentorch {

namespace {

namespace nz = zendnnl::lowoha::normalization;

/// Row stride for a [rows, D] view with a contiguous last dim (0 = dense).
uint64_t row_stride(const torch::stable::Tensor &t) {
  return t.size(0) > 1 ? static_cast<uint64_t>(t.stride(0)) : 0;
}

torch::stable::Tensor rows_view(const torch::stable::Tensor &t) {
  return t.stride(1) == 1 && t.stride(0) >= t.size(1)
             ? t
             : torch::stable::contiguous(t);
}

bool is_float_dtype(const torch::stable::Tensor &t) {
  return t.scalar_type() == c10::kBFloat16 || t.scalar_type() == c10::kFloat ||
         t.scalar_type() == c10::kHalf;
}

/// Contiguous 1-D affine parameter, or nullopt when absent.
std::optional<torch::stable::Tensor>
affine(const std::string &op, const std::optional<torch::stable::Tensor> &t,
       int64_t dim, const char *name) {
  if (!t.has_value() || !t->defined())
    return std::nullopt;
  ZENTORCH_CHECK(t->dim() == 1 && t->size(0) == dim, op, ": ", name,
                 " must be a 1-D tensor of size ", dim);
  ZENTORCH_CHECK(is_float_dtype(*t), op, ": ", name,
                 " must be bfloat16, float16 or float32");
  ZENTORCH_CHECK(t->is_contiguous(), op, ": ", name, " must be contiguous");
  return t;
}

} // namespace

void zentorch_gated_layer_norm_out(
    const torch::stable::Tensor &input, const torch::stable::Tensor &gate,
    const std::optional<torch::stable::Tensor> &weight,
    const std::optional<torch::stable::Tensor> &bias, double eps,
    const bool silu_gate, const bool concat,
    const std::string &zentorch_op_name, torch::stable::Tensor &out) {
  const std::string &op = zentorch_op_name;
  ZENTORCH_CHECK(input.dim() == 2 && gate.dim() == 2, op,
                 ": input and gate must be 2 dimensional tensors but got ",
                 input.dim(), " and ", gate.dim());
  ZENTORCH_CHECK(input.size(0) == gate.size(0) && input.size(1) == gate.size(1),
                 op, ": input and gate must have the same shape");
  ZENTORCH_CHECK(
      is_float_dtype(input) && is_float_dtype(gate) &&
          input.scalar_type() == gate.scalar_type(),
      op, ": input and gate must be bfloat16, float16 or float32 but got ",
      input.scalar_type(), " and ", gate.scalar_type());
  // ZenDNN takes eps as a float.
  const float eps_f = static_cast<float>(eps);
  ZENTORCH_CHECK(std::isfinite(eps_f) && eps_f > 0, op,
                 ": eps must be positive and finite in float32 but got ",
                 eps_f);
  const int64_t rows = input.size(0), dim = input.size(1);
  ZENTORCH_CHECK(dim > 0, op, ": the normalized dim must be non-empty");
  const int64_t cols = concat ? 3 * dim : dim;
  ZENTORCH_CHECK(out.dim() == 2, op,
                 ": out must be 2 dimensional tensor but got ", out.dim());

  ZENTORCH_CHECK(out.size(0) == rows && out.size(1) == cols, op,
                 ": out must be [rows, dim], or [rows, 3 * dim] when "
                 "concat but got ",
                 out.size(0), " and ", out.size(1));
  ZENTORCH_CHECK(out.scalar_type() == input.scalar_type(), op,
                 ": out dtype must match input");
  ZENTORCH_CHECK(out.is_contiguous(), op, ": out must be contiguous");
  const auto w = affine(op, weight, dim, "weight");
  const auto b = affine(op, bias, dim, "bias");

  if (rows == 0)
    return;

  const torch::stable::Tensor x = rows_view(input);
  const torch::stable::Tensor g = rows_view(gate);
  const size_t out_elem = out.element_size();
  char *out_base = static_cast<char *>(out.data_ptr());

  nz::norm_params p;
  p.norm_type = nz::norm_type_t::FUSED_GATED_LAYER_NORM;
  p.batch = static_cast<uint64_t>(rows);
  p.norm_size = static_cast<uint64_t>(dim);
  p.epsilon = eps_f;
  p.src_dt = get_zendnnl_dtype(x);
  p.dst_dt = p.src_dt;
  p.use_scale = w.has_value();
  p.use_shift = b.has_value();
  if (w)
    p.gamma_dt = get_zendnnl_dtype(*w);
  if (b)
    p.beta_dt = get_zendnnl_dtype(*b);
  p.src_stride = row_stride(x);
  p.dst_stride = static_cast<uint64_t>(out.size(1));
  p.gated.gate = g.data_ptr();
  p.gated.gate_dt = get_zendnnl_dtype(g);
  p.gated.gate_stride = row_stride(g);
  p.gated.gate_silu = silu_gate;
  void *y = out_base;
  if (concat) {
    p.gated.gate_out = out_base;
    p.gated.gate_out_stride = p.dst_stride;
    p.gated.src_out = out_base + dim * out_elem;
    p.gated.src_out_stride = p.dst_stride;
    y = out_base + 2 * dim * out_elem;
  }

  const auto status = nz::normalization_direct(
      x.data_ptr(), y, w ? w->data_ptr() : nullptr, b ? b->data_ptr() : nullptr,
      /*running_mean=*/nullptr,
      /*running_var=*/nullptr, /*residual=*/nullptr, p);
  ZENTORCH_CHECK(status == zendnnl::common::status_t::success, op,
                 ": ZenDNN FUSED_GATED_LAYER_NORM failed");
}

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_gated_layer_norm.out(Tensor input, Tensor gate, Tensor? "
        "weight, Tensor? bias, float eps, bool silu_gate=False, bool "
        "concat=False, *, str "
        "zentorch_op_name='zentorch::zentorch_gated_layer_norm', "
        "Tensor(a!) out) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_gated_layer_norm.out",
         TORCH_BOX(&zentorch::zentorch_gated_layer_norm_out));
}

} // namespace zentorch
