/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "MatmulUtils.hpp"
#include "Memory.hpp"

namespace zentorch {

using namespace zendnn;

// ZenTorchMatmulOpScalesMemory struct to store the scales for input,
// weight and output tensors.
// input_scales: Dequantization scales memory for input tensor.
// weight_scales: Dequantization scales memory for weight tensor.
// dst_rq_output_scales: Requantization scales memory for output tensor.
struct ZenTorchMatmulOpScalesMemory {
  memory input_scales;
  memory weight_scales;
  memory dst_rq_output_scales;
};

// ZenTorchMatmulOpZeroPointsMemory struct to store the zero points for input.
// input_zero_points: Zero points memory for input tensor.
// dst_output_zero_points: Zero points memory for output tensor.
struct ZenTorchMatmulOpZeroPointsMemory {
  // TODO: Support for weight_zero_points.
  memory input_zero_points;
  memory dst_output_zero_points;
};

template <typename T>
inline std::vector<T> get_vector_from_tensor(const at::Tensor &tensor) {
  auto tensor_ptr = tensor.data_ptr<T>();
  return std::vector<T>(tensor_ptr, tensor_ptr + tensor.numel());
}

inline ZenTorchMatmulOpScalesMemory
get_zentorch_matmul_op_scales_memory(const at::Tensor &input_scales,
                                     const at::Tensor &weight_scales,
                                     const at::Tensor &rq_output_scales) {
  ZenTorchMatmulOpScalesMemory matmul_op_scales_memory;

  // TODO: Support for per-group config.
  // This scales memory creation is utilized for only per-tensor
  // and per-channel config.
  // Create input scales memory.
  memory::desc input_scales_desc = memory::desc(
      {input_scales.numel()}, get_ztype_from_aten(input_scales), {1});
  matmul_op_scales_memory.input_scales =
      zen_memory(input_scales, input_scales_desc);
  // Create weight scales memory.
  memory::desc weight_scales_desc = memory::desc(
      {weight_scales.numel()}, get_ztype_from_aten(weight_scales), {1});
  matmul_op_scales_memory.weight_scales =
      zen_memory(weight_scales, weight_scales_desc);

  if (rq_output_scales.defined()) {
    // Create rq output scales memory.
    memory::desc rq_output_scales_desc = memory::desc(
        {rq_output_scales.numel()}, get_ztype_from_aten(rq_output_scales), {1});
    matmul_op_scales_memory.dst_rq_output_scales =
        zen_memory(rq_output_scales, rq_output_scales_desc);
  }

  return matmul_op_scales_memory;
}

inline ZenTorchMatmulOpZeroPointsMemory
get_zentorch_matmul_op_zero_points_memory(
    const at::Tensor &input_zero_points, const at::Tensor &output_zero_points) {
  ZenTorchMatmulOpZeroPointsMemory matmul_op_zero_points_memory;
  // TODO: Support for per-group config.
  // This zero points memory creation is utilized for only per-tensor and
  // per-channel config. Create input zero points memory.
  memory::desc input_zero_points_desc = memory::desc(
      {input_zero_points.numel()}, get_ztype_from_aten(input_zero_points), {1});
  matmul_op_zero_points_memory.input_zero_points =
      zen_memory(input_zero_points, input_zero_points_desc);
  // TODO: Support for weight zero points.
  if (output_zero_points.defined()) {
    // Create output zero points memory.
    memory::desc output_zero_points_desc =
        memory::desc({output_zero_points.numel()},
                     get_ztype_from_aten(output_zero_points), {1});
    matmul_op_zero_points_memory.dst_output_zero_points =
        zen_memory(output_zero_points, output_zero_points_desc);
  }
  return matmul_op_zero_points_memory;
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
    const at::Tensor &result, const at::Tensor &input_scales,
    const at::Tensor &input_zero_points, const bool &is_input_quantized,
    at::Tensor &q_input, memory &z_q_input, memory &z_q_weight, memory &z_bias,
    memory &z_result) {
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
    // fp32 tensor quantization:
    // q_tensor_s8 =
    // max(quant_min, std::nearby_int(tensor_fp32/scale) + zero_point)
    // s8 q_tensor dequantization:
    // dq_tensor_fp32 =
    // (min(quant_max, q_tensor_s8) - zero_point) * scale

    // `input` tensor quantization with q_input_scales & input_zero_points.
    // ZenDNN matmul's quantized kernel only supports u8 & s8 dtype for
    // quantized input & s8 dtype for quantized weight.
    at::Tensor q_input_scales = 1 / input_scales;
    std::vector<float> q_input_scales_vec =
        get_vector_from_tensor<float>(q_input_scales);
    std::vector<int32_t> input_zero_points_vec =
        get_vector_from_tensor<int32_t>(input_zero_points);
    reorder_tensors_with_scales_and_zero_points(
        q_input, z_q_input, z_input, q_input_scales_vec, input_zero_points_vec);
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
    z_bias = zen_memory(bias, bias_desc);
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

  // fp32 and bf16 inputs are supported by quantizing it to int8(s8) or
  // uint8(u8).
  const bool is_input_fp32 = (input.scalar_type() == c10::kFloat);
  const bool is_input_bf16 = (input.scalar_type() == c10::kBFloat16);
  // ZenDNN matmul's quantized kernel only supports u8 & s8 dtype for quantized
  // input.
  const bool is_input_u8 = (input.scalar_type() == c10::kByte);
  const bool is_input_s8 = (input.scalar_type() == c10::kChar);

  // ZenDNN matmul's quantized kernel only supports s8 dtype for quantized
  // weight.
  const bool is_weight_s8 = (weight.scalar_type() == c10::kChar);

  // ZenDNN matmul's quantized kernel only supports u8, s8, fp32 and bf16
  // dtypes for result.
  const bool is_result_u8 = (result.scalar_type() == c10::kByte);
  const bool is_result_s8 = (result.scalar_type() == c10::kChar);
  const bool is_result_fp32 = (result.scalar_type() == c10::kFloat);
  const bool is_result_bf16 = (result.scalar_type() == c10::kBFloat16);

  const bool is_input_dtype_valid =
      (is_input_fp32 || is_input_bf16 || is_input_u8 || is_input_s8);
  const bool is_result_dtype_valid =
      (is_result_fp32 || is_result_bf16 || is_result_u8 || is_result_s8);

  ZENTORCH_CHECK(is_input_dtype_valid,
                 "unsupported dtype for input tensor, "
                 "only float32/bfloat16/uint8/int8 is supported");
  ZENTORCH_CHECK(is_weight_s8,
                 "unsupported dtype for weight tensor, only int8 is supported");
  ZENTORCH_CHECK(is_result_dtype_valid,
                 "unsupported dtype for result tensor, "
                 "only float32/bfloat16/uint8/int8 is supported");

  if (bias.defined()) {
    // zentorch_qlinear op only supports fp32 and bf16 bias.
    ZENTORCH_CHECK((bias.scalar_type() == c10::kFloat ||
                    bias.scalar_type() == c10::kBFloat16),
                   "unsupported dtype for bias tensor, only float32 or "
                   "bfloat16 is supported");
  }
  // Torch dtype checks specfic for quantized matmul.
  ZENTORCH_CHECK(input_scales.scalar_type() == c10::kFloat,
                 "unsupported dtype for input_scales, only float32 "
                 "is supported");
  if (!(is_input_fp32 || is_input_bf16)) {
    ZENTORCH_CHECK(
        input.scalar_type() == input_zero_points.scalar_type(),
        "input tensor and input_zero_points tensor should have same dtype");
  } else {
    ZENTORCH_CHECK((input_zero_points.scalar_type() == c10::kChar) ||
                       (input_zero_points.scalar_type() == c10::kByte),
                   "unsupported dtype for input_zero_points, only int8/uint8 "
                   "is supported when input tensor is float32 or bfloat16");
  }
  ZENTORCH_CHECK(weight_scales.scalar_type() == c10::kFloat,
                 "unsupported dtype for weight_scales, only float32 "
                 "is supported");
  ZENTORCH_CHECK(
      weight_zero_points.scalar_type() == c10::kChar,
      "unsupported dtype for weight_zero_points, only int8 is supported");

  if (output_scales.defined() && output_zero_points.defined()) {
    ZENTORCH_CHECK(output_scales.scalar_type() == c10::kFloat,
                   "unsupported dtype for output_scales, only float32 "
                   "is supported");
    ZENTORCH_CHECK((output_zero_points.scalar_type() == c10::kChar) ||
                       (output_zero_points.scalar_type() == c10::kByte),
                   "unsupported dtype for output_zero_points, only int8/uint8 "
                   "is supported");
    ZENTORCH_CHECK(
        result.scalar_type() == output_zero_points.scalar_type(),
        "result tensor and output_zero_points tensor should have same dtype");
  } else {
    ZENTORCH_CHECK((result.scalar_type() == c10::kFloat ||
                    result.scalar_type() == c10::kBFloat16),
                   "unsupported dtype for result tensor, only float32/bfloat16 "
                   "is supported when "
                   "output_scales and output_zero_points are not defined");
  }
  if (post_op_buffers.size() != 0) {
    bool are_postops_fp32 = true;
    bool are_postops_bf16 = true;

    for (const at::Tensor &buffer : post_op_buffers) {
      are_postops_fp32 =
          are_postops_fp32 && (buffer.scalar_type() == c10::ScalarType::Float);
      are_postops_bf16 = are_postops_bf16 &&
                         (buffer.scalar_type() == c10::ScalarType::BFloat16);
    }
    ZENTORCH_CHECK((are_postops_fp32 || are_postops_bf16),
                   "unsupported dtype for post_ops, only float32/bfloat16 "
                   "is supported");
  } else {
    LOG(INFO) << "Post Op buffers are not present!\n";
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
  // For Hugging Face's large language models (LLMs) that are statically
  // quantized with the Quark quantizer, 0-dimensional scales and zero_points
  // need to be supported for per-tensor configuration.
  //
  // However, for recommender models (RMs, e.g., DLRM_v2) that are statically
  // quantized with the Quark quantizer, 1-dimensional scales and zero_points
  // need to be supported for per-tensor configuration.
  ZENTORCH_CHECK((input_scales.dim() == 1 || input_scales.dim() == 0),
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
  ZENTORCH_CHECK((weight_scales.dim() == 1 || weight_scales.dim() == 0),
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
    ZENTORCH_CHECK((output_scales.dim() == 1 || output_scales.dim() == 0),
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

  if (post_op_buffers.size() != 0) {
    bool are_postops_dim_compatible = true;
    bool are_postops_shape_compatible = true;

    for (const at::Tensor &buffer : post_op_buffers) {
      are_postops_dim_compatible =
          are_postops_dim_compatible && (buffer.dim() == input.dim());
      are_postops_shape_compatible =
          are_postops_shape_compatible &&
          (buffer.sizes() ==
           c10::IntArrayRef(get_matmul_and_linear_output_sizes(input, weight)));
    }

    ZENTORCH_CHECK(are_postops_dim_compatible,
                   "unsupported dims for post op buffers for input and weight");
    ZENTORCH_CHECK(
        are_postops_shape_compatible,
        "unsupported shapes for post op buffers for input and weight");
  } else {
    LOG(INFO) << "Post Op buffers are not present!\n";
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
