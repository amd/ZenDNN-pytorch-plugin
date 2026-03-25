/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EnvReader.hpp"
#include "MatmulUtils.hpp"
#include "Memory.hpp"

using namespace zendnnl::interface;

namespace zentorch {

static void check_valid_dtypes_for_dynamic_qlinear(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &weight_scales, const at::Tensor &bias) {

  const bool is_input_bf16 = (input.scalar_type() == c10::kBFloat16);
  const bool is_input_fp32 = (input.scalar_type() == c10::kFloat);
  ZENTORCH_CHECK(is_input_bf16 || is_input_fp32,
                 "zentorch_dynamic_qlinear: input must be bfloat16 or "
                 "float32, got ",
                 input.scalar_type());

  const bool is_scales_fp32 = (weight_scales.scalar_type() == c10::kFloat);
  const bool is_scales_bf16 = (weight_scales.scalar_type() == c10::kBFloat16);
  ZENTORCH_CHECK(is_scales_fp32 || is_scales_bf16,
                 "zentorch_dynamic_qlinear: weight_scales must be float32, "
                 "bfloat16, got ",
                 weight_scales.scalar_type());

  ZENTORCH_CHECK(
      weight.scalar_type() == c10::kChar,
      "zentorch_dynamic_qlinear: weight must be int8 (c10::kChar), got ",
      weight.scalar_type());

  if (bias.defined()) {
    ZENTORCH_CHECK(bias.scalar_type() == c10::kFloat ||
                       bias.scalar_type() == c10::kBFloat16,
                   "zentorch_dynamic_qlinear: bias must be float32 or "
                   "bfloat16, got ",
                   bias.scalar_type());
  }
}

static void check_valid_sizes_for_dynamic_qlinear(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &weight_scales, const at::Tensor &bias) {

  ZENTORCH_CHECK(input.dim() >= 2,
                 "zentorch_dynamic_qlinear: input must be at least 2D");

  ZENTORCH_CHECK(weight.dim() == 2,
                 "zentorch_dynamic_qlinear: weight must be 2D [N, K], got ",
                 weight.dim(), "D");
  // Weight is [N, K] (original nn.Linear layout)
  const int64_t N = weight.size(0);
  const int64_t K = weight.size(1);

  ZENTORCH_CHECK(input.size(input.dim() - 1) == K,
                 "zentorch_dynamic_qlinear: input last dim (", input.size(-1),
                 ") must match weight dim 1 (", K, ")");

  if (bias.defined()) {
    ZENTORCH_CHECK(bias.dim() == 1 && bias.size(0) == N,
                   "zentorch_dynamic_qlinear: bias must be 1D with size N (", N,
                   ")");
  }
}

// Core implementation: the input is dynamically quantized to S8 inside the
// kernel. src_scale_dims controls the source quantization granularity.
static void zentorch_dynamic_qlinear_impl(const at::Tensor &input_2d,
                                          const at::Tensor &weight,
                                          const at::Tensor &bias,
                                          at::Tensor &result_2d,
                                          const at::Tensor &weight_scales,
                                          const std::string &zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  LOG(INFO) << "input dimensions: " << input_2d.sizes();
  LOG(INFO) << "weight dimensions: " << weight.sizes();
  LOG(INFO) << "weight_scales dimensions: " << weight_scales.sizes();
  LOG(INFO) << "result dimensions: " << result_2d.sizes();

  const int M = input_2d.size(0);
  const int K = input_2d.size(1);
  // Weight is [N, K] (original nn.Linear layout), transposed via transB=true
  const int N = weight.size(0);

  zendnnl::lowoha::matmul::matmul_data_types dtypes;
  dtypes.src = get_zendnnl_dtype(input_2d);
  dtypes.wei = data_type_t::s8;
  dtypes.dst = get_zendnnl_dtype(result_2d);
  dtypes.bias = bias.defined() ? get_zendnnl_dtype(bias) : data_type_t::none;
  dtypes.compute = data_type_t::s8;

  zendnnl::lowoha::matmul::matmul_params params;
  params.dtypes = dtypes;
  params.dynamic_quant = true;
  params.plugin_op = zentorch_op_name;

  // src_scale.dt must match wei_scale.dt (DLP backend requirement)
  params.quant_params.src_scale.buff = nullptr;
  params.quant_params.src_scale.dt = get_zendnnl_dtype(weight_scales);
  // TODO: Currently src_scale_dims is hardcoded to be (M,1). Eventually it has
  // to be handled in the replacement pattern.
  params.quant_params.src_scale.dims = {static_cast<int64_t>(M), 1};

  params.quant_params.wei_scale.buff = weight_scales.data_ptr();
  params.quant_params.wei_scale.dt = get_zendnnl_dtype(weight_scales);
  auto ws_dims = weight_scales.sizes().vec();
  if (ws_dims.size() == 1) {
    // Normalize 1D {N} to 2D {1, N} for per-channel format required by LowOHA
    ws_dims = {1, ws_dims[0]};
  }
  params.quant_params.wei_scale.dims = ws_dims;

  zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

  // Weight is contiguous [N, K], transposed via transB=true; ldb = K
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

at::Tensor zentorch_dynamic_qlinear(const at::Tensor &input,
                                    const at::Tensor &weight,
                                    const at::Tensor &weight_scales,
                                    const c10::optional<at::Tensor> &bias,
                                    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor &bias_t = *bias_maybe_owned;

  check_valid_dtypes_for_dynamic_qlinear(input, weight, weight_scales, bias_t);
  check_valid_sizes_for_dynamic_qlinear(input, weight, weight_scales, bias_t);

  // Weight is [N, K]; output last dim is N = weight.size(0)
  auto output_sz = input.sizes().vec();
  output_sz.back() = weight.size(0);
  auto output_strides = get_matmul_and_linear_output_strides(output_sz);
  at::Tensor result =
      at::detail::empty_strided_cpu(output_sz, output_strides, input.options());

  auto input_2d = input.is_contiguous()
                      ? input.view(get_2d_size_for_tensor(input))
                      : input.contiguous().view(get_2d_size_for_tensor(input));

  at::Tensor result_2d = result.view(get_2d_size_for_tensor(result));

  zentorch_dynamic_qlinear_impl(input_2d, weight, bias_t, result_2d,
                                weight_scales, zentorch_op_name);

  return result;
}

// zentorch_dynamic_qlinear API is experimental.
TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_dynamic_qlinear(Tensor input, Tensor weight, "
        "Tensor weight_scales, Tensor? bias=None, *, "
        "str zentorch_op_name="
        "'zentorch::zentorch_dynamic_qlinear') -> Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_dynamic_qlinear", zentorch::zentorch_dynamic_qlinear);
}

} // namespace zentorch
