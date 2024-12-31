/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "MatmulUtils.hpp"
#include "Memory.hpp"
#include "QLinearUtils.hpp"

namespace zentorch {

using namespace zendnn;

inline void zentorch_quantized_matmul_impl(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, at::Tensor &result,
    const at::Tensor &input_scales, const at::Tensor &input_zero_points,
    const at::Tensor &weight_scales, const at::Tensor &weight_zero_points,
    const std::vector<int64_t> &post_op_ids,
    const std::vector<at::Tensor> &post_op_buffers,
    const c10::ScalarType &output_dtype,
    const c10::optional<at::Tensor> &output_scales,
    const c10::optional<at::Tensor> &output_zero_points,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  LOG(INFO) << "input dimensions: " << input.sizes();
  LOG(INFO) << "weight dimensions: " << weight.sizes();
  LOG(INFO) << "input_scales dimensions: " << input_scales.sizes();
  LOG(INFO) << "input_zero_points dimensions: " << input_zero_points.sizes();
  LOG(INFO) << "weight_scales dimensions: " << weight_scales.sizes();
  LOG(INFO) << "weight_zero_points dimensions: " << weight_zero_points.sizes();
  LOG(INFO) << "result dimensions: " << result.sizes();

  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor &bias_t = *bias_maybe_owned;
  const bool bias_defined = bias_t.defined();

  // Torch checks for quantized matmul.
  checks_for_quantized_matmul(
      bias_t, input, weight, result, input_scales, input_zero_points,
      weight_scales, weight_zero_points, output_scales, output_zero_points,
      post_op_ids, post_op_buffers, output_dtype);

  at::Tensor q_input = at::empty(
      input.sizes(),
      input.options().dtype(input_zero_points.scalar_type())); // For u8 & s8
  at::Tensor q_bias;
  if (bias_defined) {
    LOG(INFO) << "bias dimensions: " << bias_t.sizes();
    q_bias = at::empty(bias_t.sizes(), bias_t.options()); // For f32
  }

  // Get the required matmul op scales.
  ZenTorchMatmulOpScales matmul_op_scales =
      get_zentorch_matmul_op_scales(input_scales, weight_scales);

  // Get the required matmul op zero points.
  at::Tensor input_zero_points_int32_t = input_zero_points.toType(c10::kInt);
  ZenTorchMatmulOpZeroPoints matmul_op_zero_points =
      get_zentorch_matmul_op_zero_points(input_zero_points_int32_t);

  memory z_q_input, z_q_weight, z_q_bias, z_result;
  aten_tensor_to_zen_memory_for_quantized_matmul(
      input, weight, bias_t, result, matmul_op_scales, matmul_op_zero_points,
      q_input, q_bias, z_q_input, z_q_weight, z_q_bias, z_result);

  std::unordered_map<int, memory> execute_args;
  zendnn::primitive_attr op_attr;

  op_attr.set_plugin_op_name(zentorch_op_name);

  // Set the input_zero_points for the matmul operation.
  op_attr.set_zero_points(ZENDNN_ARG_SRC,
                          /* mask */ QUANT_GRANULARITY::PER_TENSOR,
                          matmul_op_zero_points.input_zero_points);

  // TODO : Support per-tensor requantization
  // requantization(rq) = dequantization(dq) + quantization(q)
  // rq_scale = dq_scale / output_scales[0].item<float>();
  // result_scale = (requantize) ? rq_scale : dq_scale;
  // Only dequantization is currently supported.
  // Set the output_scales for the quantized matmul operation to dequantize
  // the result.
  set_output_scales_for_op_attr(result, matmul_op_scales.dst_output_scales,
                                op_attr);

  // Execute the zendnn::matmul kernel.
  zentorch_matmul_execute(execute_args, z_q_input, z_q_weight, z_q_bias,
                          z_result, op_attr, bias_defined);

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
}

at::Tensor zentorch_qlinear(const at::Tensor &input, const at::Tensor &weight,
                            const c10::optional<at::Tensor> &bias,
                            const at::Tensor &input_scales,
                            const at::Tensor &input_zero_points,
                            const at::Tensor &weight_scales,
                            const at::Tensor &weight_zero_points,
                            c10::ScalarType output_dtype,
                            const c10::optional<at::Tensor> &output_scales,
                            const c10::optional<at::Tensor> &output_zero_points,
                            std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  if (output_dtype != at::kFloat) {
    ZENTORCH_CHECK(false, "output_dtype received is not yet supported, only "
                          "float32 is supported");
  }

  // `input` is viewed as 2d for matmul computation.
  auto input_2d_view =
      input.is_contiguous()
          ? input.view(get_2d_size_for_tensor(input))
          : input.contiguous().view(get_2d_size_for_tensor(input));

  // `weight` is transposed for matmul computation.
  auto weight_transposed = weight.t();

  // `result` tensor's dtype will depend on output_dtype argument.
  at::Tensor result =
      at::empty(get_matmul_and_linear_output_sizes(input, weight_transposed),
                input.options().dtype(output_dtype));

  // `result` is viewed as 2d for matmul computation.
  at::Tensor result_2d_view = result.view(get_2d_size_for_tensor(result));

  // Set unary post ops.
  std::vector<at::Tensor> post_op_buffers = {};
  std::vector<int64_t> post_op_ids = {UNARY_POST_OP::POST_OP_NONE};
  LOG(INFO) << "Calling zentorch_quantized_matmul_impl from " << __FUNCTION__
            << "!\n";
  zentorch_quantized_matmul_impl(
      input_2d_view, weight_transposed, bias, result_2d_view, input_scales,
      input_zero_points, weight_scales, weight_zero_points, post_op_ids,
      post_op_buffers, output_dtype, output_scales, output_zero_points,
      zentorch_op_name);
  return result;
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_qlinear(Tensor input, Tensor weight, "
        "Tensor? bias, Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, "
        "ScalarType output_dtype, Tensor? output_scales=None, "
        "Tensor? output_zero_points=None, str zentorch_op_name="
        "'zentorch::zentorch_qlinear') -> Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_qlinear", zentorch::zentorch_qlinear);
}

} // namespace zentorch
