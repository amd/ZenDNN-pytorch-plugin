/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "Memory.hpp"
#include "Ops.hpp"

namespace zentorch {

using namespace zendnn;

inline void check_valid_sizes(const at::Tensor &mat1, const at::Tensor &mat2) {
  TORCH_CHECK(
      ((mat1.dim() <= 3 && mat2.dim() <= 3) &&  // dimensionality check
       ((mat1.dim() == 2 && mat2.dim() == 1) || // specific case for aten::mv
        (mat1.dim() == mat2.dim()))), // general check for matrix multiplication
      "zentorch_matmul:  unsupported dims for mat1 and mat2");
}

inline void check_scalar_type(const std::vector<at::Tensor> &tensor_vector) {
  bool is_float = true, is_bfloat16 = true;

  for (auto tensor : tensor_vector) {
    is_float = is_float && (tensor.scalar_type() == c10::ScalarType::Float);
    is_bfloat16 =
        is_bfloat16 && (tensor.scalar_type() == c10::ScalarType::BFloat16);
  }
  TORCH_CHECK(
      is_float || is_bfloat16,
      "zentorch_matmul: zentorch_matmul only supports Float and BFloat16");
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

// Whichever tensors are converted to ZenDNN memory inside the
// tensors_to_memory function, there must be a aten tensor as well that points
// to the same space. This is done to avoid corruption of values of the tensors
// that are converted to zendnn memory. In this function four tensors are
// converted to zendnn memory, so we return those tensors to the calling
// function to have a aten tensor to point to the same space.
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
matmul_tensors_to_memory(const at::Tensor &mat1, const at::Tensor &mat2,
                         at::Tensor &result, const at::Tensor &bias,
                         at::Tensor &beta_bias, memory &z_mat1, memory &z_mat2,
                         memory &z_bias, memory &z_result, const float &beta,
                         const float &alpha) {

  check_valid_sizes(mat1, mat2);

  if (mat1.scalar_type() == c10::ScalarType::BFloat16 ||
      mat2.scalar_type() == c10::ScalarType::BFloat16) {
    TORCH_CHECK(
        utils::zendnn_bf16_device_check(),
        "zentorch_matmul: zentorch_matmul bf16 path needs the cpu support "
        "avx512bf16");
  }

  bool is_mat1_fp32 = (mat1.scalar_type() == c10::ScalarType::Float);
  bool is_mat1_bf16 = (mat1.scalar_type() == c10::ScalarType::BFloat16);
  bool is_mat2_fp32 = (mat2.scalar_type() == c10::ScalarType::Float);
  bool is_mat2_bf16 = (mat2.scalar_type() == c10::ScalarType::BFloat16);
  bool is_result_fp32 = (result.scalar_type() == c10::ScalarType::Float);
  bool is_result_bf16 = (result.scalar_type() == c10::ScalarType::BFloat16);
  TORCH_CHECK(
      (is_mat1_fp32 && is_mat2_fp32 && is_result_fp32) ||
          (is_mat1_bf16 && is_mat2_bf16 && (is_result_bf16 || is_result_fp32)),
      "zentorch_matmul: zentorch_matmul only supports Float and BFloat16");

  // ZenDNN does not support 1-D tensors. So, whenever the tensors are of
  // 1 dimension, they are unsqueezed on the required dimension to make them
  // into 2-D dimensional tensors.
  const at::Tensor &mat1_unsqueezed =
      mat1.dim() == 1 ? mat1.unsqueeze(0) : mat1;
  const at::Tensor &mat2_unsqueezed =
      mat2.dim() == 1 ? mat2.unsqueeze(1) : mat2;
  at::Tensor &self_or_result_unsqueezed =
      result.dim() == 1 ? result.unsqueeze_(1) : result;

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
          "zentorch_matmul: zentorch_matmul bf16 path needs the cpu support "
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

inline void zentorch_matmul_execute(
    std::unordered_map<int, memory> &execute_args, const memory &src,
    const memory &weight, const memory &bias, const memory &dst,
    const zendnn::primitive_attr &op_attr, const bool &bias_defined) {

  matmul::desc matmul_desc =
      bias_defined
          ? matmul::desc(src.get_desc(), weight.get_desc(), bias.get_desc(),
                         dst.get_desc())
          : matmul::desc(src.get_desc(), weight.get_desc(), dst.get_desc());

  matmul::primitive_desc pd =
      matmul::primitive_desc(matmul_desc, op_attr, utils::engine::cpu_engine());

  execute_args.insert({ZENDNN_ARG_SRC, src});
  execute_args.insert({ZENDNN_ARG_WEIGHTS, weight});
  if (bias_defined) {
    execute_args.insert({ZENDNN_ARG_BIAS, bias});
  }
  execute_args.insert({ZENDNN_ARG_DST, dst});

  LOG(INFO) << "MatMul compute in progress...";
  matmul(pd).execute(utils::stream::default_stream(), execute_args);
  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
}

} // namespace zentorch
