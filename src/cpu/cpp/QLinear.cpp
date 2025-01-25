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

  c10::MaybeOwned<at::Tensor> output_scales_maybe_owned =
      at::borrow_from_optional_tensor(output_scales);
  const at::Tensor &output_scales_t = *output_scales_maybe_owned;
  const bool output_scales_defined = output_scales_t.defined();

  c10::MaybeOwned<at::Tensor> output_zero_points_maybe_owned =
      at::borrow_from_optional_tensor(output_zero_points);
  const at::Tensor &output_zero_points_t = *output_zero_points_maybe_owned;
  const bool output_zero_points_defined = output_zero_points_t.defined();

  TORCH_CHECK(!(result.scalar_type() == c10::kFloat &&
                (output_scales_defined || output_zero_points_defined)),
              "output_scales and output_zero_points are not supported when "
              "output is dequantized to float32");

  // Torch checks for quantized matmul.
  checks_for_quantized_matmul(bias_t, input, weight, result, input_scales,
                              input_zero_points, weight_scales,
                              weight_zero_points, output_scales_t,
                              output_zero_points_t, post_op_buffers);

  // Here the assumption is that, if the input dtype is int8(kChar)
  // or uint8(kByte), then it is already quantized.
  bool is_input_quantized =
      input.scalar_type() == c10::kByte || input.scalar_type() == c10::kChar;
  at::Tensor q_input;
  if (!is_input_quantized) {
    q_input = at::empty(
        input.sizes(),
        input.options().dtype(input_zero_points.scalar_type())); // For u8 & s8
  }

  at::Tensor q_bias;
  if (bias_defined) {
    LOG(INFO) << "bias dimensions: " << bias_t.sizes();
    q_bias = at::empty(bias_t.sizes(), bias_t.options()); // For f32
  }

  // Get the required matmul op scales.
  ZenTorchMatmulOpScales matmul_op_scales = get_zentorch_matmul_op_scales(
      input_scales, weight_scales, output_scales_t);
  // Get the required matmul op zero points.
  at::Tensor input_zero_points_int32_t = input_zero_points.toType(c10::kInt);
  at::Tensor output_zero_points_int32_t;
  if (output_zero_points_defined) {
    output_zero_points_int32_t = output_zero_points_t.toType(c10::kInt);
  }
  ZenTorchMatmulOpZeroPoints matmul_op_zero_points =
      get_zentorch_matmul_op_zero_points(input_zero_points_int32_t,
                                         output_zero_points_int32_t);

  memory z_q_input, z_q_weight, z_q_bias, z_result;
  aten_tensor_to_zen_memory_for_quantized_matmul(
      input, weight, bias_t, result, matmul_op_scales, matmul_op_zero_points,
      is_input_quantized, q_input, q_bias, z_q_input, z_q_weight, z_q_bias,
      z_result);

  std::unordered_map<int, memory> execute_args;
  zendnn::primitive_attr op_attr;
  post_ops po;
  // Setting Post ops
  if (post_op_ids.size() > 0) {
    zentorch_post_ops_selection(po, execute_args, post_op_ids, post_op_buffers);
    op_attr.set_post_ops(po);
  }
  op_attr.set_plugin_op_name(zentorch_op_name);

  // Set the input_zero_points for the matmul operation.
  op_attr.set_zero_points(ZENDNN_ARG_SRC,
                          /* mask */ QUANT_GRANULARITY::PER_TENSOR,
                          matmul_op_zero_points.input_zero_points);
  // Set dst_output_zero_points only for requantized uint8 output.
  // requantization(rq) = dequantization(dq) + quantization(q)
  // rq_scale = dq_scale / output_scales[0].item<float>();
  // result_scale = (requantize) ? rq_scale : dq_scale;
  if (output_zero_points_defined &&
      output_zero_points_t.scalar_type() == c10::kByte) {
    if (output_zero_points_t.numel() == 1) {
      op_attr.set_zero_points(ZENDNN_ARG_DST,
                              /* mask */ QUANT_GRANULARITY::PER_TENSOR,
                              matmul_op_zero_points.dst_output_zero_points);
    } else {
      ZENTORCH_CHECK(false, "only per-tensor requantization is supported");
    }
  }

  // Set the dst_output_scales for the quantized matmul operation to
  // dequantize/requantize the result.
  set_output_scales_for_op_attr(result, matmul_op_scales.dst_output_scales,
                                op_attr);

  // Execute the zendnn::matmul kernel.
  zentorch_matmul_execute(execute_args, z_q_input, z_q_weight, z_q_bias,
                          z_result, op_attr, bias_defined);

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
}

template <UNARY_POST_OP fuse>
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

  ZENTORCH_CHECK(output_dtype == c10::kFloat || output_dtype == c10::kByte ||
                     output_dtype == c10::kChar,
                 "output_dtype received is not yet supported, only "
                 "float32/uint8/int8 is supported");

  at::Tensor q_input =
      at::empty(input.sizes(),
                input.options().dtype(
                    input_zero_points.scalar_type())); // For u8, s8 & f32

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
  std::vector<int64_t> post_op_ids = {fuse};
  LOG(INFO) << "Calling zentorch_quantized_matmul_impl from " << __FUNCTION__
            << "!\n";
  zentorch_quantized_matmul_impl(
      input_2d_view, weight_transposed, bias, result_2d_view, input_scales,
      input_zero_points, weight_scales, weight_zero_points, post_op_ids,
      post_op_buffers, output_scales, output_zero_points, zentorch_op_name);
  return result;
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_qlinear(Tensor input, Tensor weight, "
        "Tensor? bias, Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, "
        "ScalarType output_dtype, Tensor? output_scales=None, "
        "Tensor? output_zero_points=None, str zentorch_op_name="
        "'zentorch::zentorch_qlinear') -> Tensor");
  m.def("zentorch_qlinear_relu(Tensor input, Tensor weight, "
        "Tensor? bias, Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, "
        "ScalarType output_dtype, Tensor? output_scales=None, "
        "Tensor? output_zero_points=None, str zentorch_op_name="
        "'zentorch::zentorch_qlinear_relu') -> Tensor");
  m.def("zentorch_qlinear_sigmoid(Tensor input, Tensor weight, "
        "Tensor? bias, Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, "
        "ScalarType output_dtype, Tensor? output_scales=None, "
        "Tensor? output_zero_points=None, str zentorch_op_name="
        "'zentorch::zentorch_qlinear_sigmoid') -> Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_qlinear",
         zentorch::zentorch_qlinear<UNARY_POST_OP::POST_OP_NONE>);
  m.impl("zentorch_qlinear_relu",
         zentorch::zentorch_qlinear<UNARY_POST_OP::RELU>);
  m.impl("zentorch_qlinear_sigmoid",
         zentorch::zentorch_qlinear<UNARY_POST_OP::SIGMOID>);
}

} // namespace zentorch
