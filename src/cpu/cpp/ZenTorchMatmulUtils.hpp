/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "ZenTorchMemory.hpp"

namespace zentorch {

using namespace zendnn;

inline void check_valid_sizes(const at::Tensor &mat1, const at::Tensor &mat2) {
  TORCH_CHECK(
      ((mat1.dim() <= 3 && mat2.dim() <= 3) &&  // dimensionality check
       ((mat1.dim() == 2 && mat2.dim() == 1) || // specific case for aten::mv
        (mat1.dim() == mat2.dim()))), // general check for matrix multiplication
      "zendnn_matmul:  unsupported dims for mat1 and mat2");
}

inline void check_scalar_type(const std::vector<at::Tensor> &tensor_vector) {
  bool is_float = true, is_bfloat16 = true;

  for (auto tensor : tensor_vector) {
    is_float = is_float && (tensor.scalar_type() == c10::ScalarType::Float);
    is_bfloat16 =
        is_bfloat16 && (tensor.scalar_type() == c10::ScalarType::BFloat16);
  }

  TORCH_CHECK(is_float || is_bfloat16,
              "zendnn_matmul: zendnn_matmul only supports Float and BFloat16");
}

inline bool is_zendnn_optimized_format(const at::Tensor &t) {
  if (t.is_contiguous())
    return true;
  const auto sizes = t.sizes();
  const auto strides = t.strides();
  // check for transposed tensors
  if (t.dim() == 2) {
    return strides[0] == 1 && strides[1] == sizes[0];
  } else {
    // dim = 3
    return strides[0] == sizes[1] * sizes[2] && strides[1] == 1 &&
           strides[2] == sizes[1];
  }
}

inline std::vector<int64_t> get_matmul_output_sizes(const at::Tensor &tensor1,
                                                    const at::Tensor &tensor2) {
  const int64_t dim = tensor1.dim();
  std::vector<int64_t> output_size(dim);
  for (auto i = 0; i < dim - 1; i++) {
    output_size[i] = tensor1.size(i);
  }
  output_size[dim - 1] = tensor2.size(dim - 1);
  return output_size;
}

// Whichever tensors are converted to ZenDNN memory inside the
// tensors_to_memory function, there must be a aten tensor as well that points
// to the same space. This is done to avoid corruption of values of the tensors
// that are converted to zendnn memory. In this function four tensors are
// converted to zendnn memory, so we return those tensors to the calling
// function to have a aten tensor to point to the same space.
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
matmul_tensors_to_memory(const at::Tensor &mat1, const at::Tensor &mat2,
                         at::Tensor &self_or_result, const at::Tensor &bias,
                         at::Tensor &beta_bias, memory &z_mat1, memory &z_mat2,
                         memory &z_bias, memory &z_result, const float &beta,
                         const float &alpha) {

  check_valid_sizes(mat1, mat2);

  if (mat1.scalar_type() == c10::ScalarType::BFloat16 ||
      mat2.scalar_type() == c10::ScalarType::BFloat16) {
    TORCH_CHECK(utils::zendnn_bf16_device_check(),
                "zendnn_matmul: zendnn_matmul bf16 path needs the cpu support "
                "avx512bf16");
  }

  std::vector<at::Tensor> tensor_vector(3);
  tensor_vector[0] = mat1;
  tensor_vector[1] = mat2;
  tensor_vector[2] = self_or_result;

  check_scalar_type(tensor_vector);

  // ZenDNN does not support 1-D tensors. So, whenever the tensors are of
  // 1 dimension, they are unsqueezed on the required dimension to make them
  // into 2-D dimensional tensors.
  const at::Tensor &mat1_unsqueezed =
      mat1.dim() == 1 ? mat1.unsqueeze(0) : mat1;
  const at::Tensor &mat2_unsqueezed =
      mat2.dim() == 1 ? mat2.unsqueeze(1) : mat2;
  at::Tensor &self_or_result_unsqueezed =
      self_or_result.dim() == 1 ? self_or_result.unsqueeze_(1) : self_or_result;

  // zendnn is only optimized for contiguous or transposed
  // (transpose last 2 dim if 3-D tensor) format now
  // Will remove this "contiguous" after zendnn have fully supported
  at::Tensor mat1_ = is_zendnn_optimized_format(mat1_unsqueezed)
                         ? mat1_unsqueezed
                         : mat1_unsqueezed.contiguous();
  at::Tensor mat2_ = is_zendnn_optimized_format(mat2_unsqueezed)
                         ? mat2_unsqueezed
                         : mat2_unsqueezed.contiguous();

  // convert the aten tensors to zendnn memory
  z_mat1 = zen_memory(mat1_);
  z_mat2 = zen_memory(mat2_);
  z_result = zen_memory(self_or_result_unsqueezed);

  // "addmm", "baddbmm" in pytorch allow bias to be 2-D or 3-D tensor
  // but zendnn matmul primitive only support bias be 1-D tensors
  // to address their differences, we use zendnn post ops to perform a fused
  // "add" after matrix multiplication is over

  const bool bias_defined = bias.numel();

  if (bias_defined && bias.dim() == 1 && (mat1.dim() == 2 && mat2.dim() == 2)) {
    if (bias.scalar_type() == c10::ScalarType::BFloat16) {
      TORCH_CHECK(
          utils::zendnn_bf16_device_check(),
          "zendnn_matmul: zendnn_matmul bf16 path needs the cpu support "
          "avx512bf16");
    }

    std::vector<at::Tensor> tensor_vector(1);
    tensor_vector[0] = bias;

    check_scalar_type(tensor_vector);

    LOG(INFO) << "bias is defined and bias dimensions: " << bias.sizes();

    // BR_GEMM kernel execution is as alpha * (mat1 @ mat2 + bias)
    // but addmm is executed as alpha * (mat1 @ mat2) + beta * bias

    // so alpha * (mat1 @ mat2 + (beta / alpha) * bias) is equivalent
    // to alpha * (mat1 @ mat2) + beta * bias
    const float modified_beta =
        (alpha == 1.0f || alpha == 0) ? beta : beta / alpha;
    beta_bias = (modified_beta == 1.0f) ? bias : bias.mul(modified_beta);

    // creating bias zen_memory with predefined memory::desc
    // as bias is 1d we need to define format_tag as 'ab'
    // to represent bias memory as 2d for bias_desc creation
    const memory::format_tag &bias_tag = memory::format_tag::ab;
    const memory::desc &bias_desc = memory::desc(
        {{1, beta_bias.size(0)}, get_ztype_from_aten(beta_bias), bias_tag});
    z_bias = zen_memory(beta_bias, bias_desc);
  }

  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> out;
  out = std::make_tuple(self_or_result_unsqueezed, mat1_, mat2_, beta_bias);

  return out;
}

} // namespace zentorch
