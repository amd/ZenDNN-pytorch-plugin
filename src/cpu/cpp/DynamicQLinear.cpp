/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "DynamicQLinear.hpp"
#include "EnvReader.hpp"
#include "MatmulUtils.hpp"
#include "Memory.hpp"

#include <ATen/ATen.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

using namespace zendnnl::interface;

namespace zentorch {

static inline bool zentorch_checks_enabled() {
  return static_cast<bool>(
      EnvReader::getEnvVariableAsInt("ZENTORCH_ENABLE_CHECKS"));
}

// Dtype checks (input, weight_scales, bias) common to both modes. The
// weight_scales and bias constraints are identical; only the input differs:
// DA8W8 accepts bf16 or f32, DA8W4 is bf16-only (allow_fp32_input == false).
template <typename TensorT>
static void check_valid_common_dtypes_for_qlinear(const TensorT &input,
                                                  const TensorT &weight_scales,
                                                  const TensorT &bias,
                                                  bool allow_fp32_input) {
  if (!zentorch_checks_enabled())
    return;

  const bool is_input_bf16 = (input.scalar_type() == c10::kBFloat16);
  const bool is_input_fp32 = (input.scalar_type() == c10::kFloat);
  ZENTORCH_CHECK(is_input_bf16 || (allow_fp32_input && is_input_fp32),
                 "zentorch_dynamic_qlinear: input must be ",
                 allow_fp32_input ? "bfloat16 or float32" : "bfloat16",
                 ", got ", input.scalar_type());

  const bool is_scales_fp32 = (weight_scales.scalar_type() == c10::kFloat);
  const bool is_scales_bf16 = (weight_scales.scalar_type() == c10::kBFloat16);
  ZENTORCH_CHECK(is_scales_fp32 || is_scales_bf16,
                 "zentorch_dynamic_qlinear: weight_scales must be float32 or "
                 "bfloat16, got ",
                 weight_scales.scalar_type());

  if (bias.defined()) {
    ZENTORCH_CHECK(
        bias.scalar_type() == c10::kFloat ||
            bias.scalar_type() == c10::kBFloat16,
        "zentorch_dynamic_qlinear: bias must be float32 or bfloat16, got ",
        bias.scalar_type());
  }
}

// Core implementation shared by DA8W8 (s8 weight) and DA8W4 (packed s4 weight).
static void zentorch_dynamic_qlinear_impl(
    const torch::stable::Tensor &input_2d, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &bias, torch::stable::Tensor &result_2d,
    const torch::stable::Tensor &weight_scales, bool is_da8w4,
    const std::string &zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__
            << " (is_da8w4=" << is_da8w4 << ")";
  LOG(INFO) << "input dimensions: [" << c10::Join(", ", input_2d.sizes())
            << "]";
  LOG(INFO) << "weight dimensions: [" << c10::Join(", ", weight.sizes()) << "]";
  LOG(INFO) << "weight_scales dimensions: ["
            << c10::Join(", ", weight_scales.sizes()) << "]";
  LOG(INFO) << "result dimensions: [" << c10::Join(", ", result_2d.sizes())
            << "]";

  const int64_t M = input_2d.size(0);
  const int64_t K = input_2d.size(1);
  // Weight is [N, K-dim]; N = size(0) for all modes.
  const int64_t N = weight.size(0);

  zendnnl::lowoha::matmul::matmul_data_types dtypes;
  dtypes.src = get_zendnnl_dtype(input_2d);
  dtypes.wei = is_da8w4 ? data_type_t::s4 : data_type_t::s8;
  dtypes.dst = get_zendnnl_dtype(result_2d);
  dtypes.bias = bias.defined() ? get_zendnnl_dtype(bias) : data_type_t::none;
  dtypes.compute = data_type_t::s8;

  zendnnl::lowoha::matmul::matmul_params params;
  params.dtypes = dtypes;
  params.dynamic_quant = true;
  // lowoha_algo is left unset: ZenDNN auto-detects W4A8 (dynamic s8 x s4) and
  // routes it to AOCL-DLP, honoring any runtime algo override.
  params.plugin_op = zentorch_op_name;

  // Dynamic per-token source scale: buffer computed at runtime by the kernel.
  // src_scale.dt must match wei_scale.dt (DLP backend requirement).
  params.quant_params.src_scale.buff = nullptr;
  params.quant_params.src_scale.dt = get_zendnnl_dtype(weight_scales);
  params.quant_params.src_scale.dims = {M, 1};

  params.quant_params.wei_scale.buff = weight_scales.data_ptr();
  params.quant_params.wei_scale.dt = get_zendnnl_dtype(weight_scales);
  if (is_da8w4) {
    // Per-group weight scale {G, N}.
    params.quant_params.wei_scale.dims = {weight_scales.size(0), N};
  } else {
    // Per-channel: normalize 1D {N} to 2D {1, N} as required by LowOHA.
    auto ws_dims = sizes_to_int64_vec(weight_scales.sizes());
    if (ws_dims.size() == 1) {
      ws_dims = {1, ws_dims[0]};
    }
    params.quant_params.wei_scale.dims = ws_dims;
  }

  zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

  // Weight passed in [N, K-dim] orientation, transposed via transB=true;
  // ldb = K (in element units for s8, nibble units for packed s4).
  status_t status = zendnnl::lowoha::matmul::matmul_direct(
      'r', false /* transA */, true /* transB */, M, N, K, 1.0f /* alpha */,
      input_2d.data_ptr(), K, weight.data_ptr(), K,
      bias.defined() ? bias.data_ptr() : nullptr, 0.0f /* beta */,
      result_2d.data_ptr(), N, true /* is_weights_const */, batch_params,
      params);

  ZENTORCH_CHECK(status == status_t::success,
                 "zentorch_dynamic_qlinear: matmul_direct execution failed");

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
}

static void check_valid_dims_for_qlinear(const torch::stable::Tensor &input,
                                         const torch::stable::Tensor &weight) {
  if (!zentorch_checks_enabled())
    return;
  ZENTORCH_CHECK(input.dim() >= 2,
                 "zentorch_dynamic_qlinear: input must be at least 2D, got ",
                 input.dim(), "D");
  ZENTORCH_CHECK(weight.dim() == 2,
                 "zentorch_dynamic_qlinear: weight must be 2D [N, K-dim], got ",
                 weight.dim(), "D");
}

// Shared by the functional and out variants; assumes valid dims (see
// check_valid_dims_for_qlinear) and writes into the pre-shaped `result`.
static void dispatch_dynamic_qlinear(const torch::stable::Tensor &input,
                                     const torch::stable::Tensor &weight,
                                     const torch::stable::Tensor &weight_scales,
                                     const torch::stable::Tensor &bias_t,
                                     torch::stable::Tensor &result,
                                     const std::string &zentorch_op_name) {
  auto input_2d =
      view_tensor(get_contiguous_view(input), get_2d_size_for_tensor(input));

  // Infers the mode and validates the weight dtype/shape in one step.
  const bool is_da8w4 = check_weight_and_infer_is_da8w4(input, weight);

  check_valid_common_dtypes_for_qlinear(input, weight_scales, bias_t,
                                        /*allow_fp32_input=*/!is_da8w4);

  // The impl reads weight / weight_scales / bias through raw data_ptr() with
  // hardcoded leading dims (weight ldb = K), so they MUST be contiguous. These
  // are static params, so normalizing them is the caller's responsibility: the
  // frontends that lower to this op (e.g. the torchao Int8Tensor replacement)
  // pass contiguous weight/scales/bias, and the Inductor lowering pins the same
  // contiguity for the compiled / cpp_wrapper path.
  //
  // We deliberately do NOT call .contiguous() here. Doing so would silently fix
  // a non-contiguous tensor coming from a buggy replacement path by copying it
  // on every call -- correct results, but a per-call copy that surfaces only as
  // an unexplained throughput drop, traceable solely via profiling (a "silent
  // performance regression"). Asserting instead makes such a bug fail loudly in
  // first-level testing.
  //
  // Tradeoff: with checks disabled (production default) a non-contiguous static
  // param that slips past testing is undefined behaviour -- the kernel reads
  // the wrong strides and produces silently wrong results rather than a slow
  // but correct one. We accept this because the compiled path is already safe
  // (lowering require_contiguous) and every frontend is expected to uphold the
  // contract. NOTE: `input` is intentionally still normalized above -- it is a
  // dynamic activation that may legitimately be non-contiguous, not a static
  // param under this contract.
  if (zentorch_checks_enabled()) {
    ZENTORCH_CHECK(weight.is_contiguous(),
                   "zentorch_dynamic_qlinear: weight must be contiguous; "
                   "the calling replacement path must normalize it");
    ZENTORCH_CHECK(
        weight_scales.is_contiguous(),
        "zentorch_dynamic_qlinear: weight_scales must be contiguous; "
        "the calling replacement path must normalize it");
    if (bias_t.defined()) {
      ZENTORCH_CHECK(bias_t.dim() == 1 && bias_t.size(0) == weight.size(0),
                     "zentorch_dynamic_qlinear: bias must be 1D with size N (",
                     weight.size(0), ")");
      ZENTORCH_CHECK(bias_t.is_contiguous(),
                     "zentorch_dynamic_qlinear: bias must be contiguous; "
                     "the calling replacement path must normalize it");
    }
  }

  auto result_2d = view_tensor(result, get_2d_size_for_tensor(result));
  zentorch_dynamic_qlinear_impl(input_2d, weight, bias_t, result_2d,
                                weight_scales, is_da8w4, zentorch_op_name);
}

torch::stable::Tensor
zentorch_dynamic_qlinear(const torch::stable::Tensor &input,
                         const torch::stable::Tensor &weight,
                         const torch::stable::Tensor &weight_scales,
                         const std::optional<torch::stable::Tensor> &bias,
                         std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  static const torch::stable::Tensor kUndefinedTensor;
  const torch::stable::Tensor &bias_t =
      bias.has_value() ? *bias : kUndefinedTensor;

  check_valid_dims_for_qlinear(input, weight);

  // Output last dim is N = weight.size(0) for the s8 [N, K] and packed-s4
  // [N, K/2] / [N, K/8] weight layouts.
  auto output_sz = sizes_to_int64_vec(input.sizes());
  output_sz.back() = weight.size(0);
  auto result = torch::stable::new_empty(input, output_sz);

  dispatch_dynamic_qlinear(input, weight, weight_scales, bias_t, result,
                           zentorch_op_name);
  return result;
}

void zentorch_dynamic_qlinear_out(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const std::optional<torch::stable::Tensor> &bias,
    std::string zentorch_op_name, torch::stable::Tensor &out) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  static const torch::stable::Tensor kUndefinedTensor;
  const torch::stable::Tensor &bias_t =
      bias.has_value() ? *bias : kUndefinedTensor;

  check_valid_dims_for_qlinear(input, weight);

  if (zentorch_checks_enabled()) {
    auto output_sz = sizes_to_int64_vec(input.sizes());
    output_sz.back() = weight.size(0);
    ZENTORCH_CHECK(out.scalar_type() == input.scalar_type(),
                   "zentorch_dynamic_qlinear.out: out dtype (",
                   out.scalar_type(), ") must match input dtype (",
                   input.scalar_type(), ")");
    ZENTORCH_CHECK(
        out.sizes().equals(output_sz),
        "zentorch_dynamic_qlinear.out: out shape must be [*, N] with "
        "N = weight.size(0)");
    ZENTORCH_CHECK(out.is_contiguous(),
                   "zentorch_dynamic_qlinear.out: out must be contiguous");
  }

  dispatch_dynamic_qlinear(input, weight, weight_scales, bias_t, out,
                           zentorch_op_name);
}

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_dynamic_qlinear(Tensor input, Tensor weight, "
        "Tensor weight_scales, Tensor? bias=None, *, "
        "str zentorch_op_name="
        "'zentorch::zentorch_dynamic_qlinear') -> Tensor");
  m.def("zentorch_dynamic_qlinear.out(Tensor input, Tensor weight, "
        "Tensor weight_scales, Tensor? bias=None, *, "
        "str zentorch_op_name='zentorch::zentorch_dynamic_qlinear.out', "
        "Tensor(a!) out) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_dynamic_qlinear",
         TORCH_BOX(&zentorch::zentorch_dynamic_qlinear));
  m.impl("zentorch_dynamic_qlinear.out",
         TORCH_BOX(&zentorch::zentorch_dynamic_qlinear_out));
}

template bool check_weight_and_infer_is_da8w4<at::Tensor>(const at::Tensor &,
                                                          const at::Tensor &);
template bool check_weight_and_infer_is_da8w4<torch::stable::Tensor>(
    const torch::stable::Tensor &, const torch::stable::Tensor &);

} // namespace zentorch
