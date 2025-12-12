/******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <functional> // For std::reference_wrapper, std::ref, std::cref
#include <optional>   // For std::optional, std::nullopt
#include <unordered_map>

#include "Memory.hpp"

using namespace zendnnl::interface;

namespace zentorch {

// Map from post-op enum values to their corresponding post_op_type_t
static const std::unordered_map<int64_t, post_op_type_t> post_op_type_map = {
    // Unary post-ops
    {UNARY_POST_OP::RELU, post_op_type_t::relu},
    {UNARY_POST_OP::GELU_TANH, post_op_type_t::gelu_tanh},
    {UNARY_POST_OP::GELU_ERF, post_op_type_t::gelu_erf},
    {UNARY_POST_OP::SIGMOID, post_op_type_t::sigmoid},
    {UNARY_POST_OP::SILU, post_op_type_t::swish},
    {UNARY_POST_OP::TANH, post_op_type_t::tanh},

    // Binary post-ops
    {BINARY_POST_OP::MUL, post_op_type_t::binary_mul},
    {BINARY_POST_OP::ADD, post_op_type_t::binary_add}};

inline at::Tensor get_contiguous_view(const at::Tensor &tensor) {
  auto stride = tensor.strides();
  auto sizes = tensor.sizes();
  bool is_sorted =
      std::is_sorted(stride.begin(), stride.end(), std::greater<int64_t>());
  bool is_zero =
      std::any_of(stride.begin(), stride.end(), [](auto s) { return s == 0; });
  if (!is_sorted || is_zero) {
    auto new_tensor = tensor.clone(at::MemoryFormat::Contiguous).view(sizes);
    LOG(INFO) << "Tensor is not contiguous. Converting the tensor to a "
                 "contiguous format in "
              << __FILE__ << ": " << __LINE__ << " in " << __FUNCTION__;
    return new_tensor;
  }
  return tensor.view(sizes);
}

// this function returns the output size for matrix multiplication of two
// tensors - tensor1 @ tensor2 and also it returns the output size for
// linear operation of these two tensors, and if tensor2 is packed on the
// last dim it will support the unpacking the size of last dim of tensor2
inline std::vector<int64_t>
get_matmul_and_linear_output_sizes(const at::Tensor &tensor1,
                                   const at::Tensor &tensor2,
                                   const int64_t unpacking_ratio = 1) {
  auto tensor1_size = tensor1.sizes();
  std::vector<int64_t> output_size(tensor1_size.begin(),
                                   tensor1_size.end() - 1);

  auto tensor2_last_dim_size = tensor2.size(tensor2.dim() - 1);
  auto calculated_last_dim_size = tensor2_last_dim_size * unpacking_ratio;
  output_size.push_back(calculated_last_dim_size);
  return output_size;
}

inline void
check_valid_sizes_for_matmul(const at::Tensor &mat1, const at::Tensor &mat2,
                             const at::Tensor &bias, const at::Tensor &result,
                             const std::vector<at::Tensor> &post_op_buffers) {

  // The flow of this check is as follows:
  // -> Generic dim check for the mat1 and mat2. The functionality of aten::mv
  //    is covered here.
  // -> Next the result shape is checked to be compatible with the matrix
  //    multiplication shape. This is done at the second stage as rest of the
  //    tensors can be optional, and irrespective of the other tensors, the
  //    matrix multiplication of mat1 and mat2 will happen.
  // -> Bias being optional in the addmm variants, needs to be checked if it is
  //    a defined tensor or not, and based on that shape of bias is checked.
  //    Here, only 1-D bias case is checked, as bias if 2-D or 3-D will be
  //    passed as post op and checked in the post op checks.
  // -> Based on the post op buffer vector size, the shapes of all the post ops
  //    are determined. Again here, all the post op buffers must be of the same
  //    shape as the matmul product shape or result tensor shape.

  const int mat1_dim = mat1.dim();
  const int mat2_dim = mat2.dim();

  ZENTORCH_CHECK(
      // dimensionality check for batched matrix multiplication
      ((mat1_dim == 3 && mat2_dim == 3) ||
       // dimensionality check for matrix multiplication
       (mat1_dim == 2 && mat2_dim == 2)),
      "unsupported dims for mat1 and mat2");

  // Array access is faster than .size(n)
  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();

  // matmul shape compatibility check
  // Eg:
  // If mat1_sizes was (2, 5, 6) and mat2_sizes was (2, 6, 5)
  // Then the mat1_dim equals 3 in value
  // mat1_dim - 1 would point to the last dimension in both the sizes vector.
  // So, mat1_sizes[mat1_dim - 1] == mat2_sizes[mat1_dim - 2]
  //                6             ==            6
  // pass

  // If mat1_sizes was (2, 5, 6) and mat2_sizes was (2, 5, 6)
  // Then the mat1_dim equals 3 in value
  // mat1_dim - 1 would point to the last dimension in both the sizes vector.
  // So, mat1_sizes[mat1_dim - 1] == mat2_sizes[mat1_dim - 2]
  //                6             ==            5
  // fail

  if (mat1_dim == 3) {
    ZENTORCH_CHECK(
        mat1_sizes[0] == mat2_sizes[0],
        "Tensor shapes incompatible for batch matrix multiplication");
  }

  ZENTORCH_CHECK(mat1_sizes[mat1_dim - 1] == mat2_sizes[mat1_dim - 2],
                 "Tensor shapes incompatible for matrix multiplication");

  ZENTORCH_CHECK(result.dim() == mat1_dim,
                 "unsupported dims for mat1, mat2 and "
                 "result buffer");

  const bool is_bias_defined = bias.numel();
  if (is_bias_defined) {
    if (bias.dim() != 1) {
      ZENTORCH_CHECK(false, "unsupported dimensions for input/bias/self");
    }
    const auto bias_sizes = bias.sizes();
    if (mat1_dim == 2) {
      ZENTORCH_CHECK(bias_sizes[0] == mat2_sizes[1],
                     "input/bias/self shape is incompatible for addition with "
                     "matrix multiplication product (",
                     mat1_sizes[0], "x", mat1_sizes[1], " @ ", mat2_sizes[0],
                     "x", mat2_sizes[1], " != ", mat1_sizes[0], "x",
                     bias_sizes[0], ")");
    } else if (mat1_dim == 3) {
      ZENTORCH_CHECK(bias_sizes[0] == mat2_sizes[2],
                     "input/bias/self shape is incompatible for addition with "
                     "matrix multiplication product (",
                     mat1_sizes[0], "x", mat1_sizes[1], "x", mat1_sizes[2],
                     " @ ", mat2_sizes[0], "x", mat2_sizes[1], "x",
                     mat2_sizes[2], " != ", mat1_sizes[0], "x", mat1_sizes[1],
                     "x", bias_sizes[0], ")");
    }
  }

  if (post_op_buffers.empty()) {
    LOG(INFO) << "Post Op buffers are not present!\n";
    return;
  }
  bool are_postops_dim_compatible = true;
  bool are_postops_shape_compatible = true;

  for (const at::Tensor &buffer : post_op_buffers) {
    are_postops_dim_compatible =
        are_postops_dim_compatible && (buffer.dim() == mat1_dim);
    are_postops_shape_compatible =
        are_postops_shape_compatible && (buffer.sizes() == result.sizes());
  }

  ZENTORCH_CHECK(are_postops_dim_compatible,
                 "unsupported dims for mat1, mat2 and "
                 "post op buffers");
  ZENTORCH_CHECK(are_postops_shape_compatible,
                 "unsupported shapes for mat1, mat2 and "
                 "post op buffers");
}

inline void
check_valid_dtypes_for_matmul(const at::Tensor &mat1, const at::Tensor &mat2,
                              const at::Tensor &bias, const at::Tensor &result,
                              const std::vector<at::Tensor> &post_op_buffers) {

  // The flow of this check is as follows:
  // -> The individual datatypes of the tensors are inferred.
  // -> Bias being optional in the addmm variants, needs to be checked if it is
  //    a defined tensor or not.
  // -> The tensors which are inputs to the actual matmul call are confirmed
  //    to be either of datatype float32 or bfloat16, but not a combination.
  // -> The previous check is combined with the check of the
  //    destination (result) buffer.
  // -> If the dataype is bfloat16, the machine capability is checked.
  // -> Based on the post op buffer vector size, the dtypes of all the post ops
  //    are determined. Again here, all the post op buffers must be of the same
  //    dtype as the matmul parameters, either float32 or bfloat16, not a
  //    combination of both.

  const bool is_bias_defined = bias.numel();
  const bool is_mat1_fp32 = (mat1.scalar_type() == c10::ScalarType::Float);
  const bool is_mat1_bf16 = (mat1.scalar_type() == c10::ScalarType::BFloat16);
  const bool is_mat2_fp32 = (mat2.scalar_type() == c10::ScalarType::Float);
  const bool is_mat2_bf16 = (mat2.scalar_type() == c10::ScalarType::BFloat16);
  const bool is_result_fp32 = (result.scalar_type() == c10::ScalarType::Float);
  const bool is_result_bf16 =
      (result.scalar_type() == c10::ScalarType::BFloat16);
  bool is_bias_fp32, is_bias_bf16;

  if (is_bias_defined) {
    is_bias_fp32 = (bias.scalar_type() == c10::ScalarType::Float);
    is_bias_bf16 = (bias.scalar_type() == c10::ScalarType::BFloat16);
  }

  const bool are_params_fp32 =
      is_bias_defined
          ? (is_mat1_fp32 && is_mat2_fp32 && is_bias_fp32 && is_result_fp32)
          : (is_mat1_fp32 && is_mat2_fp32 && is_result_fp32);
  const bool are_params_bf16 =
      is_bias_defined
          ? (is_mat1_bf16 && is_mat2_bf16 && is_bias_bf16 && is_result_bf16)
          : (is_mat1_bf16 && is_mat2_bf16 && is_result_bf16);

  ZENTORCH_CHECK(are_params_fp32 ^ are_params_bf16,
                 "zentorch_matmul only supports Float and BFloat16");

  if (are_params_bf16) {
    ZENTORCH_CHECK(zentorch::zendnn_bf16_device_check(),
                   "zentorch_matmul bf16 path needs the cpu support "
                   "avx512bf16");
  }

  if (post_op_buffers.empty()) {
    LOG(INFO) << "Post Op buffers are not present!\n";
    return;
  }
  bool are_postops_fp32 = true;
  bool are_postops_bf16 = true;

  for (const at::Tensor &buffer : post_op_buffers) {
    are_postops_fp32 =
        are_postops_fp32 && (buffer.scalar_type() == c10::ScalarType::Float);
    are_postops_bf16 =
        are_postops_bf16 && (buffer.scalar_type() == c10::ScalarType::BFloat16);
  }

  if (are_params_fp32 && !are_params_bf16) {
    ZENTORCH_CHECK((are_postops_fp32 && !are_postops_bf16),
                   "zentorch_matmul only supports Float post ops "
                   "when input matrix is Float");
  } else if (are_params_bf16 && !are_params_fp32) {
    ZENTORCH_CHECK((are_postops_bf16 && !are_postops_fp32),
                   "zentorch_matmul only supports BFloat16 post ops "
                   "when input matrix is BFloat16");
  } else {
    ZENTORCH_CHECK(false, "zentorch_matmul only supports Float and BFloat16 "
                          "parameters and postops");
  }
}

inline void check_valid_dtypes_for_quantized_matmul(
    const at::Tensor &bias, const at::Tensor &input, const at::Tensor weight,
    const at::Tensor &result, const at::Tensor &input_scales,
    const at::Tensor &input_zero_points, const at::Tensor &weight_scales,
    const at::Tensor &weight_zero_points, const at::Tensor &output_scales,
    const at::Tensor &output_zero_points,
    const std::vector<at::Tensor> &post_op_buffers) {
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
  if (input_zero_points.defined()) {
    ZENTORCH_CHECK(
        (input_zero_points.scalar_type() == c10::kInt),
        "unsupported dtype for input_zero_points, only int32 supported");
  }
  ZENTORCH_CHECK(weight_scales.scalar_type() == c10::kFloat,
                 "unsupported dtype for weight_scales, only float32 "
                 "is supported");
  if (weight_zero_points.defined()) {
    ZENTORCH_CHECK(
        weight_zero_points.scalar_type() == c10::kInt,
        "unsupported dtype for weight_zero_points, only int32 is supported");
  }
  if (output_scales.defined()) {
    ZENTORCH_CHECK(output_scales.scalar_type() == c10::kFloat,
                   "unsupported dtype for output_scales, only float32 "
                   "is supported");
    if (output_zero_points.defined()) {
      ZENTORCH_CHECK((output_zero_points.scalar_type() == c10::kInt),
                     "unsupported dtype for output_zero_points, only int32 "
                     "is supported");

      ZENTORCH_CHECK(
          (result.scalar_type() == c10::kByte),
          "unsupported dtype for result tensor when output_zero_points"
          "are defined, uint8 are supported dtype");
    } else {
      ZENTORCH_CHECK(
          (result.scalar_type() == c10::kChar),
          "unsupported dtype for result tensor when output_zero_points"
          "are not defined while output_scales are defined,"
          "int8 are supported dtype");
    }
  } else {
    ZENTORCH_CHECK((result.scalar_type() == c10::kFloat ||
                    result.scalar_type() == c10::kBFloat16),
                   "unsupported dtype for result tensor, only float32/bfloat16 "
                   "is supported when output_scales are not defined");
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
  if (input_zero_points.defined()) {
    ZENTORCH_CHECK(
        (input_zero_points.dim() == 0 || input_zero_points.dim() == 1),
        "only scalar and 1-d input_zero_points are supported");
    ZENTORCH_CHECK((input_zero_points.numel() == 1),
                   "only supporting per-tensor quantization for input");
  }

  ZENTORCH_CHECK((input_scales.numel() == 1),
                 "unsupported number of elements for input_scales "
                 "with respect to input tensor");

  // Per-tensor/channel config check.
  ZENTORCH_CHECK((weight_scales.dim() == 1 || weight_scales.dim() == 0),
                 "unsupported dims for weight_scales with respect "
                 "to weight tensor");
  ZENTORCH_CHECK(
      (weight_scales.numel() == 1) || (weight_scales.numel() == weight.size(1)),
      "unsupported number of elements for weight_scales with respect "
      "to weight tensor");
  if (weight_zero_points.defined()) {
    ZENTORCH_CHECK(
        (weight_zero_points.dim() == 0 || weight_zero_points.dim() == 1),
        "only scalar and 1-d weight_zero_points are supported");
    ZENTORCH_CHECK(
        (weight_zero_points.numel() == 1) ||
            (weight_zero_points.numel() == weight.size(1)),
        "only supporting per-tensor and per-channel quantization for weight");
  }
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

// TODO
// Check if this and the is_transposed function can be merged into one
inline bool is_zendnn_optimized_format(const at::Tensor &t) {
  const auto sizes = t.sizes();
  const auto strides = t.strides();

  bool is_sorted =
      std::is_sorted(strides.begin(), strides.end(), std::greater<int64_t>());
  bool is_zero = std::any_of(strides.begin(), strides.end(),
                             [](auto s) { return s == 0; });
  if (!is_zero && is_sorted) {
    return true;
  }

  // check for transposed tensors
  if (t.dim() == 2) {
    return strides[0] == 1 && strides[1] == sizes[0];
  } else {
    // dim = 3
    return strides[0] == sizes[1] * sizes[2] && strides[1] == 1 &&
           strides[2] == sizes[1];
  }
}

// this function returns the 2-d size for n-d inp_tensor,
// also if inp_tensor is packed on the last dim it will
// support the unpacking the size of last dim of inp_tensor
inline std::vector<int64_t>
get_2d_size_for_tensor(const at::Tensor &inp_tensor,
                       const int64_t unpacking_ratio = 1) {
  const int64_t dim = inp_tensor.dim();
  std::vector<int64_t> output_size(2);

  output_size[0] = inp_tensor.numel() / inp_tensor.size(dim - 1);
  output_size[1] = inp_tensor.size(dim - 1) * unpacking_ratio;
  return output_size;
}

// this function returns the output stride for matrix multiplication of two
// tensors - tensor1 @ tensor2 and also it returns the output stride for
// linear operation of these two tensors
inline std::vector<int64_t>
get_matmul_and_linear_output_strides(const std::vector<int64_t> &output_size) {
  int output_size_sz = output_size.size();
  std::vector<int64_t> output_strides;
  int64_t mul = 1;
  for (int cnt = 0; cnt < output_size_sz; cnt++) {
    if (cnt > 0) {
      mul *= output_size[output_size_sz - cnt];
    }
    output_strides.emplace_back(mul);
  }
  std::reverse(output_strides.begin(), output_strides.end());
  return output_strides;
}

inline at::Tensor
create_linear_and_matmul_output_tensor(const at::Tensor input,
                                       const at::Tensor weight) {
  auto output_size = get_matmul_and_linear_output_sizes(input, weight);
  auto output_strides = get_matmul_and_linear_output_strides(output_size);

  // For AOT Inductor compatibility, we need to set the device to CPU
  c10::Device device = c10::Device(c10::DeviceType::CPU);

  // Create options with explicit device
  auto options = at::TensorOptions()
                     .dtype(input.dtype())
                     .layout(input.layout())
                     .device(device)
                     .requires_grad(false);

  at::Tensor result =
      at::detail::empty_strided_cpu(output_size, output_strides, options);
  return result;
}

// TODO
// Check if this and the is_zendnn_optimized_format function can be merged into
// one
inline bool is_transposed(const at::Tensor &t) {
  const auto sizes = t.sizes();
  const auto strides = t.strides();
  // check for transposed tensors
  if (t.dim() == 2) {
    return strides[0] == 1 && strides[1] >= sizes[0];
  } else {
    // dim = 3
    return strides[0] >= sizes[1] * sizes[2] && strides[1] == 1 &&
           strides[2] >= sizes[1];
  }
}

inline bool is_stride_valid(const at::Tensor &t) {
  const auto sizes = t.sizes();
  const auto strides = t.strides();
  if (t.dim() == 2) {
    return true;
  } else if (t.dim() == 3 && strides[0] >= sizes[1] * sizes[2]) {
    return true;
  }
  return false;
}

inline bool validate_zendnnl_direct_kernel_usage(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
    const at::Tensor &result, const std::vector<at::Tensor> &post_op_buffers) {

  // Currently this kernel supports only float32 and bfloat16 datatype
  // Define the datatype check as a lambda
  // The tensors for this kernel have to be either float or bfloat16
  auto is_dtype_supported = [](const at::Tensor &x) {
    return ((x.scalar_type() == c10::ScalarType::Float) ||
            (x.scalar_type() == c10::ScalarType::BFloat16));
  };

  const bool are_tensor_dtypes_valid =
      is_dtype_supported(input) && is_dtype_supported(weight) &&
      ((bias.defined() && is_dtype_supported(bias)) || !bias.defined()) &&
      is_dtype_supported(result);

  if (!are_tensor_dtypes_valid) {
    return false;
  }

  check_valid_dtypes_for_matmul(input, weight, bias, result, post_op_buffers);

  // By the time the control comes to this function, we are working with tensors
  // that are of 2d or 3d shape and all the tensors are of same shape. So,
  // creating check with only one of the tensors should suffice for all the
  // validations.
  const bool are_tensors_compatible_for_matmul =
      input.dim() == weight.dim() && input.dim() == result.dim();

  // Bias is optional and can be of 1d when weight is 2d, or 3d when weight is
  // 3d.
  const bool is_bias_compatible_for_matmul =
      bias.defined() ? (weight.dim() == 2 ? bias.dim() == 1 : bias.dim() == 3)
                     : true;

  if (!(are_tensors_compatible_for_matmul && is_bias_compatible_for_matmul)) {
    return false;
  }

  check_valid_sizes_for_matmul(input, weight, bias, result, post_op_buffers);

  return true;
}

inline void zendnnl_direct_kernel(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
    const at::Tensor &result, const float &alpha,
    const std::vector<int64_t> &post_op_ids,
    const std::vector<at::Tensor> &post_op_buffers, const bool is_weight_const,
    const bool is_weight_prepacked) {

  // TODO
  // Check if we can use tensor.strided() instead of is_stride_valid()

  // TODO
  // Check if we can use contiguous irrespective of the condition
  at::Tensor input_ =
      is_stride_valid(input) ? input : get_contiguous_view(input);
  at::Tensor weight_ =
      is_stride_valid(weight) ? weight : get_contiguous_view(weight);

  // By the time the control comes to this function, we are working with tensors
  // that are of 2d or 3d shape and all the tensors are of same shape. So,
  // creating check with only one of the tensors should suffice for all the
  // validations.

  bool is_matmul_2d = input_.dim() == 2;

  bool transA = is_transposed(input_);
  bool transB = is_transposed(weight_);
  int batch_A = is_matmul_2d ? 1 : input_.size(0);
  int batch_B = is_matmul_2d ? 1 : weight_.size(0);
  int M = is_matmul_2d ? input_.size(0) : input_.size(1);
  int K = is_matmul_2d ? input_.size(1) : input_.size(2);
  int N = is_matmul_2d ? weight_.size(1) : weight_.size(2);

  auto stridesA = input_.strides();
  auto stridesB = weight_.strides();
  auto stridesC = result.strides();
  int lda, ldb, ldc;

  if (transA) {
    lda = is_matmul_2d ? stridesA[1] : stridesA[2];
  } else {
    lda = is_matmul_2d ? stridesA[0] : stridesA[1];
  }

  if (transB) {
    ldb = is_matmul_2d ? stridesB[1] : stridesB[2];
  } else {
    ldb = is_matmul_2d ? stridesB[0] : stridesB[1];
  }

  ldc = is_matmul_2d ? stridesC[0] : stridesC[1];

  void *input_ptr = input_.data_ptr();
  void *weight_ptr = weight_.data_ptr();
  void *bias_ptr = bias.defined() ? bias.data_ptr() : nullptr;
  void *result_ptr = result.data_ptr();

  zendnnl::lowoha::data_types matmul_dtype;
  matmul_dtype.src = get_zendnnl_dtype(input);
  matmul_dtype.wei = get_zendnnl_dtype(weight);
  matmul_dtype.bias =
      bias.defined() ? get_zendnnl_dtype(bias) : data_type_t::none;
  matmul_dtype.dst = get_zendnnl_dtype(result);

  matmul_dtype.compute = data_type_t::none;

  zendnnl::lowoha::lowoha_params params;
  params.dtypes = matmul_dtype;
  if (is_weight_prepacked) {
    params.mem_format_b = 'r';
  }

  std::vector<long int> result_sizes =
      std::vector<long int>(result.sizes().begin(), result.sizes().end());

  // Lambda to add unary post-ops
  auto unary_post_op = [&params, &result_sizes](post_op_type_t op_type) {
    zendnnl::lowoha::postop post_op;
    post_op.po_type = op_type;
    post_op.buff = nullptr;
    post_op.dtype = data_type_t::none;
    post_op.dims = result_sizes;
    params.postop_.push_back(post_op);
  };

  // Lambda to add binary post-ops
  auto binary_post_op = [&params,
                         &result_sizes](post_op_type_t op_type,
                                        const at::Tensor &post_op_buffer) {
    zendnnl::lowoha::postop post_op;
    post_op.po_type = op_type;
    post_op.buff = post_op_buffer.data_ptr();
    post_op.dtype = get_zendnnl_dtype(post_op_buffer);
    auto dims = post_op_buffer.sizes();
    post_op.dims = std::vector<long int>(dims.begin(), dims.end());
    params.postop_.push_back(post_op);
  };

  int post_op_buffer_index = 0;
  for (const long &post_op_id : post_op_ids) {
    auto it = post_op_type_map.find(post_op_id);
    if (it != post_op_type_map.end()) {
      post_op_type_t op_type = it->second;

      // Check if it's a binary operation
      if (post_op_id == BINARY_POST_OP::MUL ||
          post_op_id == BINARY_POST_OP::ADD) {
        binary_post_op(op_type, post_op_buffers[post_op_buffer_index++]);
      } else {
        unary_post_op(op_type);
      }
    }
  }

  zendnnl::lowoha::batch_params_t batch_params;
  batch_params.Batch_A = batch_A;
  batch_params.Batch_B = batch_B;

  zendnnl::lowoha::matmul_direct('r' /* layout: row-major */, transA, transB, M,
                                 N, K, alpha, input_ptr, lda, weight_ptr, ldb,
                                 bias_ptr, 0.0f /* beta */, result_ptr, ldc,
                                 is_weight_const, batch_params, params);
}

inline void set_matmul_context_attributes(
    matmul_context_t &matmul_context, tensor_t &weights,
    const std::vector<int64_t> &post_op_ids, const float &alpha,
    std::optional<std::reference_wrapper<tensor_t>> bias_opt_ref =
        std::nullopt) {
  matmul_context.set_param("weights", weights);
  matmul_context.set_alpha(alpha);

  if (bias_opt_ref.has_value()) {
    tensor_t &bias = bias_opt_ref->get();
    matmul_context.set_param("bias", bias);
  }

  for (const long &post_op_id : post_op_ids) {
    auto it = post_op_type_map.find(post_op_id);
    if (it != post_op_type_map.end()) {
      post_op_type_t op_type = it->second;
      auto post_op = post_op_t{op_type};
      matmul_context.set_post_op(post_op);
    }
  }

  matmul_context.create();
  ZENTORCH_CHECK(matmul_context.check(), "matmul context creation failed.");
}

inline void
set_matmul_operator_attributes(matmul_operator_t &matmul_operator,
                               const matmul_context_t &matmul_context,
                               tensor_t &input_tensor, tensor_t &output_tensor,
                               const std::vector<int64_t> &post_op_ids,
                               const std::vector<at::Tensor> &post_op_buffers) {

  matmul_operator.set_name("matmul_operator")
      .set_context(matmul_context)
      .create();

  ZENTORCH_CHECK(!matmul_operator.is_bad_object(), "operator ",
                 matmul_operator.get_name(), " creation failed.");

  matmul_operator.set_input("matmul_input", input_tensor)
      .set_output("matmul_output", output_tensor);

  int post_op_buffer_tracker = 0;
  int count_none_activations = 0;
  for (int op_id = 0; op_id < static_cast<int>(post_op_ids.size()); op_id++) {
    switch (post_op_ids[op_id]) {
    case BINARY_POST_OP::MUL: {
      tensor_t binary_tensor = tensor_t();
      std::string tensor_name =
          "binary_input" + std::to_string(op_id - count_none_activations);
      set_zendnnl_tensor_attributes(post_op_buffers[post_op_buffer_tracker++],
                                    binary_tensor, tensor_name);
      matmul_operator.set_input(
          matmul_context.get_post_op(op_id - count_none_activations)
              .binary_mul_params.tensor_name,
          binary_tensor);
      break;
    }
    case BINARY_POST_OP::ADD: {
      tensor_t binary_tensor = tensor_t();
      std::string tensor_name =
          "binary_input" + std::to_string(op_id - count_none_activations);
      set_zendnnl_tensor_attributes(post_op_buffers[post_op_buffer_tracker++],
                                    binary_tensor, tensor_name);
      matmul_operator.set_input(
          matmul_context.get_post_op(op_id - count_none_activations)
              .binary_add_params.tensor_name,
          binary_tensor);
      break;
    }
    case UNARY_POST_OP::POST_OP_NONE: {
      count_none_activations++;
      break;
    }
    default:
      break;
    }
  }
}
} // namespace zentorch
