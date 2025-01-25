/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "MatmulUtils.hpp"
#include "Memory.hpp"

namespace zentorch {

using namespace zendnn;

// ZenTorchMatmulOpScales struct to store the scales for input,
// weight, bias and output tensors.
// input_scales: Dequantization scales for input tensor.
// q_input_scales: Quantization scales for input tensor.
// weight_scales: Dequantization scales for weight tensor.
// q_bias_scales: Quantization scales(Scaling factor) for bias tensor.
// dst_output_scales: Dequantization scales for output tensor.
struct ZenTorchMatmulOpScales {
  std::vector<float> input_scales;
  std::vector<float> q_input_scales;
  std::vector<float> weight_scales;
  std::vector<float> q_bias_scales;
  std::vector<float> dst_output_scales;
};

// ZenTorchMatmulOpZeroPoints struct to store the zero points for input.
// input_zero_points: Zero points for input tensor.
// dst_output_zero_points: Zero points for output tensor.
struct ZenTorchMatmulOpZeroPoints {
  // TODO: Support for weight_zero_points.
  std::vector<int32_t> input_zero_points;
  std::vector<int32_t> dst_output_zero_points;
};

template <typename T>
inline std::vector<T> get_vector_from_tensor(const at::Tensor &tensor) {
  auto tensor_ptr = tensor.data_ptr<T>();
  return std::vector<T>(tensor_ptr, tensor_ptr + tensor.numel());
}

inline ZenTorchMatmulOpScales
get_zentorch_matmul_op_scales(const at::Tensor &input_scales,
                              const at::Tensor &weight_scales,
                              const at::Tensor &output_scales) {

  at::Tensor q_input_scales = 1 / input_scales;
  at::Tensor q_bias_scales = 1 / (weight_scales * input_scales);
  at::Tensor dst_output_scales;
  if (!output_scales.defined()) {
    dst_output_scales = weight_scales * input_scales;
  } else {
    dst_output_scales = (weight_scales * input_scales) / output_scales;
  }

  ZenTorchMatmulOpScales matmul_op_scales;
  matmul_op_scales.input_scales = get_vector_from_tensor<float>(input_scales);
  matmul_op_scales.q_input_scales =
      get_vector_from_tensor<float>(q_input_scales);
  matmul_op_scales.weight_scales = get_vector_from_tensor<float>(weight_scales);
  matmul_op_scales.q_bias_scales = get_vector_from_tensor<float>(q_bias_scales);
  matmul_op_scales.dst_output_scales =
      get_vector_from_tensor<float>(dst_output_scales);

  return matmul_op_scales;
}

inline ZenTorchMatmulOpZeroPoints
get_zentorch_matmul_op_zero_points(const at::Tensor &input_zero_points,
                                   const at::Tensor &output_zero_points) {
  ZenTorchMatmulOpZeroPoints matmul_op_zero_points;
  matmul_op_zero_points.input_zero_points =
      get_vector_from_tensor<int32_t>(input_zero_points);
  if (output_zero_points.defined()) {
    matmul_op_zero_points.dst_output_zero_points =
        get_vector_from_tensor<int32_t>(output_zero_points);
  }
  return matmul_op_zero_points;
}

inline void set_output_scales_for_op_attr(const at::Tensor &original_tensor,
                                          const std::vector<float> &scales_vec,
                                          zendnn::primitive_attr &op_attr) {
  if (scales_vec.size() == 1) {
    // Per-tensor config
    op_attr.set_output_scales(/* mask */ QUANT_GRANULARITY::PER_TENSOR,
                              scales_vec);
  } else if (static_cast<int64_t>(scales_vec.size()) ==
             original_tensor.size(original_tensor.dim() - 1)) {
    // Per-channel config
    op_attr.set_output_scales(/* mask */ QUANT_GRANULARITY::PER_CHANNEL,
                              scales_vec);
  } else {
    // TODO: Support per-group config
    ZENTORCH_CHECK(false,
                   "unsupported scales shape with respect to original tensor");
  }
}

inline void reorder_tensors_with_scales_and_zero_points(
    const at::Tensor &quantized_tensor, memory &z_q_tensor,
    const memory &z_o_tensor, const std::vector<float> &scales_vec,
    const std::vector<int32_t> &zero_points_vec = {}) {
  zendnn::primitive_attr q_attr;
  set_output_scales_for_op_attr(quantized_tensor, scales_vec, q_attr);
  if (zero_points_vec.size() != 0) {
    if (zero_points_vec.size() == 1) {
      // Per-tensor config
      q_attr.set_zero_points(ZENDNN_ARG_DST,
                             /* mask */ QUANT_GRANULARITY::PER_TENSOR,
                             zero_points_vec);
    } else {
      ZENTORCH_CHECK(false, "only per-tensor zero_points are supported");
    }
  }
  z_q_tensor = zentorch_reorder(z_o_tensor, z_q_tensor, q_attr);
}

// This function maps the aten tensors to the zendnn::memory.
inline void aten_tensor_to_zen_memory_for_quantized_matmul(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
    const at::Tensor &result, const ZenTorchMatmulOpScales &matmul_op_scales,
    const ZenTorchMatmulOpZeroPoints &matmul_op_zero_points,
    const bool &is_input_quantized, at::Tensor &q_input, at::Tensor &q_bias,
    memory &z_q_input, memory &z_q_weight, memory &z_q_bias, memory &z_result) {
  // Create input memory.
  memory z_input = zen_memory(input);

  // Create quantized input memory.

  if (is_input_quantized) {
    // Here the assumption is that, if the input dtype is int8(kChar)
    // or uint8(kByte), then it is already quantized.
    // If input is already quantized, then no need to quantize it again.
    z_q_input = z_input;
  } else {
    z_q_input = zen_memory(q_input);

    // f32 tensor quantization:
    // q_tensor_s8 =
    // max(quant_min, std::nearby_int(tensor_f32/scale) + zero_point)
    // s8 q_tensor dequantization:
    // dq_tensor_f32 =
    // (min(quant_max, q_tensor_s8) - zero_point) * scale

    // `input` tensor quantization with q_input_scales & input_zero_points.
    // ZenDNN matmul's quantized kernel only supports u8 & s8 dtype for
    // quantized input & s8 dtype for quantized weight.
    reorder_tensors_with_scales_and_zero_points(
        q_input, z_q_input, z_input, matmul_op_scales.q_input_scales,
        matmul_op_zero_points.input_zero_points);
  }

  // Create weight memory.
  z_q_weight = zen_memory(weight);

  // Create bias memory if bias is defined.
  if (bias.defined()) {
    // Create bias memory.
    // Creating bias zen_memory with predefined memory::desc
    // as bias is 1d we need to use format_tag as 'ab'
    // to represent bias memory as 2d for bias_desc creation.
    const memory::format_tag &memory_2d_tag = memory::format_tag::ab;
    const memory::desc &bias_desc = memory::desc(
        {{1, bias.size(0)}, get_ztype_from_aten(bias), memory_2d_tag});
    memory z_bias = zen_memory(bias, bias_desc);

    // Create quantized bias memory.
    // Creating q_bias zen_memory with predefined memory::desc
    // as q_bias is 1d we need to use format_tag as 'ab'
    // to represent q_bias memory as 2d for bias_desc creation.
    const memory::desc &q_bias_desc = memory::desc(
        {{1, q_bias.size(0)}, get_ztype_from_aten(q_bias), memory_2d_tag});
    z_q_bias = zen_memory(q_bias, q_bias_desc);

    // `bias` tensor scaling with q_bias_scales.
    // ZenDNN matmul only supports s8, s32 and f32 bias, so
    // we are going to support bias tensor by scaling it with
    // bias scales for computation in f32 for better accuracy
    reorder_tensors_with_scales_and_zero_points(q_bias, z_q_bias, z_bias,
                                                matmul_op_scales.q_bias_scales);
  }

  // Create result memory.
  z_result = zen_memory(result);
}

inline void check_valid_dtypes_for_quantized_matmul(
    const at::Tensor &bias, const at::Tensor &input, const at::Tensor weight,
    const at::Tensor &result, const at::Tensor &input_scales,
    const at::Tensor &input_zero_points, const at::Tensor &weight_scales,
    const at::Tensor &weight_zero_points, const at::Tensor &output_scales,
    const at::Tensor &output_zero_points,
    const std::vector<at::Tensor> &post_op_buffers) {
  // TODO: Modify the check once bfloat16 input and result tensors are supported

  // float(fp32) input is supported by quantizing it to int8(s8) or uint8(u8).
  const bool is_input_f32 = (input.scalar_type() == c10::kFloat);
  // ZenDNN matmul's quantized kernel only supports u8 & s8 dtype for quantized
  // input.
  const bool is_input_u8 = (input.scalar_type() == c10::kByte);
  const bool is_input_s8 = (input.scalar_type() == c10::kChar);

  // ZenDNN matmul's quantized kernel only supports s8 dtype for quantized
  // weight.
  const bool is_weight_s8 = (weight.scalar_type() == c10::kChar);

  // ZenDNN matmul's quantized kernel only supports u8, s8 & f32 dtype for
  // result.
  const bool is_result_u8 = (result.scalar_type() == c10::kByte);
  const bool is_result_s8 = (result.scalar_type() == c10::kChar);
  const bool is_result_f32 = (result.scalar_type() == c10::kFloat);

  const bool is_input_dtype_valid =
      (is_input_f32 || is_input_u8 || is_input_s8);
  const bool is_result_dtype_valid =
      (is_result_u8 || is_result_s8 || is_result_f32);

  ZENTORCH_CHECK(is_input_dtype_valid, "unsupported dtype for input tensor, "
                                       "only float32/uint8/int8 is supported");
  ZENTORCH_CHECK(is_weight_s8,
                 "unsupported dtype for weight tensor, only int8 is supported");
  ZENTORCH_CHECK(is_result_dtype_valid, "unsupported dtype for result tensor, "
                                        "only float32/uint8/int8 is supported");

  bool is_bias_fp32;
  if (bias.defined()) {
    // zentorch_qlinear op only supports f32 bias.
    // TODO: Support bf16 bias.
    is_bias_fp32 = (bias.scalar_type() == c10::kFloat);
    ZENTORCH_CHECK(
        is_bias_fp32,
        "unsupported dtype for bias tensor, only float32 is supported");
  }
  // Torch dtype checks specfic for quantized matmul.
  // TODO: Modify the check once bfloat16 scales are supported
  ZENTORCH_CHECK(input_scales.scalar_type() == c10::kFloat,
                 "unsupported dtype for input_scales");
  if (!is_input_f32) {
    ZENTORCH_CHECK(
        input.scalar_type() == input_zero_points.scalar_type(),
        "input tensor and input_zero_points tensor should have same dtype");
  } else {
    ZENTORCH_CHECK((input_zero_points.scalar_type() == c10::kChar) ||
                       (input_zero_points.scalar_type() == c10::kByte),
                   "unsupported dtype for input_zero_points, only int8/uint8 "
                   "is supported when input tensor is float32");
  }
  ZENTORCH_CHECK(
      weight_scales.scalar_type() == c10::kFloat,
      "unsupported dtype for weight_scales, only float32 is supported");
  ZENTORCH_CHECK(
      weight_zero_points.scalar_type() == c10::kChar,
      "unsupported dtype for weight_zero_points, only int8 is supported");

  if (output_scales.defined() && output_zero_points.defined()) {
    ZENTORCH_CHECK(
        output_scales.scalar_type() == c10::kFloat,
        "unsupported dtype for output_scales, only float32 is supported");
    ZENTORCH_CHECK((output_zero_points.scalar_type() == c10::kChar) ||
                       (output_zero_points.scalar_type() == c10::kByte),
                   "unsupported dtype for output_zero_points, only int8/uint8 "
                   "is supported");
    ZENTORCH_CHECK(
        result.scalar_type() == output_zero_points.scalar_type(),
        "result tensor and output_zero_points tensor should have same dtype");
  } else {
    ZENTORCH_CHECK(
        result.scalar_type() == c10::kFloat,
        "unsupported dtype for result tensor, only float32 is supported when "
        "output_scales and output_zero_points are not defined");
  }
}

inline void check_valid_sizes_for_quantized_matmul(
    const at::Tensor &bias, const at::Tensor &input, const at::Tensor weight,
    const at::Tensor &result, const at::Tensor &input_scales,
    const at::Tensor &input_zero_points, const at::Tensor &weight_scales,
    const at::Tensor &weight_zero_points, const at::Tensor &output_scales,
    const at::Tensor &output_zero_points,
    const std::vector<at::Tensor> &post_op_buffers) {
  check_valid_sizes_for_matmul(input, weight, bias, result, post_op_buffers);

  // Size checks specfic for quantized matmul.
  // Per-tensor config check.
  ZENTORCH_CHECK((input_scales.dim() == 1),
                 "unsupported dims for input_scales with respect to "
                 "input tensor");
  ZENTORCH_CHECK((input_zero_points.dim() == 0 || input_zero_points.dim() == 1),
                 "unsupported dims for input_zero_points with respect to "
                 "input tensor");

  ZENTORCH_CHECK((input_scales.numel() == 1),
                 "unsupported number of elements for input_scales "
                 "with respect to input tensor");
  ZENTORCH_CHECK((input_zero_points.numel() == 1),
                 "unsupported number of elements for input_zero_points "
                 "with respect to input tensor");

  // Per-tensor/channel config check.
  ZENTORCH_CHECK((weight_scales.dim() == 1),
                 "unsupported dims for weight_scales with respect "
                 "to weight tensor");
  ZENTORCH_CHECK(
      (weight_zero_points.dim() == 0 || weight_zero_points.dim() == 1),
      "unsupported dims for weight_zero_points with respect "
      "to weight tensor");

  ZENTORCH_CHECK(
      (weight_scales.numel() == 1) || (weight_scales.numel() == weight.size(1)),
      "unsupported number of elements for weight_scales with respect "
      "to weight tensor");
  ZENTORCH_CHECK(
      (weight_zero_points.numel() == 1) ||
          (weight_zero_points.numel() == weight.size(1)),
      "unsupported number of elements for weight_zero_points with respect "
      "to weight tensor");

  if (output_scales.defined() && output_zero_points.defined()) {
    ZENTORCH_CHECK((output_scales.dim() == 1),
                   "unsupported dims for output_scales with respect "
                   "to output tensor");
    ZENTORCH_CHECK(
        (output_zero_points.dim() == 0 || output_zero_points.dim() == 1),
        "unsupported dims for output_zero_points with respect "
        "to output tensor");

    ZENTORCH_CHECK(
        (output_scales.numel() == 1),
        "unsupported number of elements for output_scales with respect "
        "to output tensor");
    ZENTORCH_CHECK(
        (output_zero_points.numel() == 1),
        "unsupported number of elements for output_zero_points with respect "
        "to output tensor");
  }
}

inline void checks_for_quantized_matmul(
    const at::Tensor &bias, const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &result, const at::Tensor &input_scales,
    const at::Tensor &input_zero_points, const at::Tensor &weight_scales,
    const at::Tensor &weight_zero_points, const at::Tensor &output_scales,
    const at::Tensor &output_zero_points,
    const std::vector<at::Tensor> &post_op_buffers) {
  check_valid_dtypes_for_quantized_matmul(
      bias, input, weight, result, input_scales, input_zero_points,
      weight_scales, weight_zero_points, output_scales, output_zero_points,
      post_op_buffers);
  check_valid_sizes_for_quantized_matmul(
      bias, input, weight, result, input_scales, input_zero_points,
      weight_scales, weight_zero_points, output_scales, output_zero_points,
      post_op_buffers);
}

} // namespace zentorch
