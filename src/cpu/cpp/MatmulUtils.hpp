/******************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <algorithm>

#include "Memory.hpp"
namespace zentorch {
using namespace zendnn;

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
      ((mat1_dim == 3 &&
        mat2_dim == 3) || // dimensionality check for matrix multiplication
       (mat1_dim == 2 &&
        mat2_dim == 2) || // dimensionality check for matrix multiplication
       (mat1_dim == 2 && mat2_dim == 1) || // specific case for aten::mv
       (mat1_dim == 1 && mat2_dim == 1)    // specific case for aten::dot
       ),
      "unsupported dims for mat1 and mat2");

  // Array access is faster than .size(n)
  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();

  if (mat1_dim == 2 && mat2_dim == 1) {
    LOG(INFO) << "Special case of aten::mv";
    ZENTORCH_CHECK(
        post_op_buffers.size() == 0,
        "Post Op support currently unavailable for aten::mv via ZenTorch");
    // TODO
    // Need to understand how to the result is in these cases and need to add a
    // check for the result buffer as well.
    return;
  }
  if (mat1_dim == 1 && mat2_dim == 1) {
    LOG(INFO) << "Special case of aten::dot";
    ZENTORCH_CHECK(
        post_op_buffers.size() == 0,
        "Post Op support currently unavailable for aten::dot via ZenTorch");
    // TODO
    // Need to understand how to the result is in these cases and need to add a
    // check for the result buffer as well.
    return;
  }

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
  ZENTORCH_CHECK(
      result.sizes() ==
          c10::IntArrayRef(get_matmul_and_linear_output_sizes(mat1, mat2)),
      "unsupported shapes for mat1, mat2 and "
      "result buffer");

  const bool is_bias_defined = bias.numel();
  if (is_bias_defined) {
    if (bias.dim() == 1) {
      const auto bias_sizes = bias.sizes();
      if (mat1_dim == 2) {
        ZENTORCH_CHECK(
            bias_sizes[0] == mat2_sizes[1],
            "input/bias/self shape is incompatible for addition with "
            "matrix multiplication product (",
            mat1_sizes[0], "x", mat1_sizes[1], " @ ", mat2_sizes[0], "x",
            mat2_sizes[1], " != ", mat1_sizes[0], "x", bias_sizes[0], ")");
      } else if (mat1_dim == 3) {
        ZENTORCH_CHECK(
            bias_sizes[0] == mat2_sizes[2],
            "input/bias/self shape is incompatible for addition with "
            "matrix multiplication product (",
            mat1_sizes[0], "x", mat1_sizes[1], "x", mat1_sizes[2], " @ ",
            mat2_sizes[0], "x", mat2_sizes[1], "x", mat2_sizes[2],
            " != ", mat1_sizes[0], "x", mat1_sizes[1], "x", bias_sizes[0], ")");
      }
    } else {
      ZENTORCH_CHECK(false, "unsupported dimensions for input/bias/self");
    }
  }

  if (post_op_buffers.size() != 0) {
    bool are_postops_dim_compatible = true;
    bool are_postops_shape_compatible = true;

    for (const at::Tensor &buffer : post_op_buffers) {
      are_postops_dim_compatible =
          are_postops_dim_compatible && (buffer.dim() == mat1_dim);
      are_postops_shape_compatible =
          are_postops_shape_compatible &&
          (buffer.sizes() ==
           c10::IntArrayRef(get_matmul_and_linear_output_sizes(mat1, mat2)));
    }

    ZENTORCH_CHECK(are_postops_dim_compatible,
                   "unsupported dims for mat1, mat2 and "
                   "post op buffers");
    ZENTORCH_CHECK(are_postops_shape_compatible,
                   "unsupported shapes for mat1, mat2 and "
                   "post op buffers");
  } else {
    LOG(INFO) << "Post Op buffers are not present!\n";
  }
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
    ZENTORCH_CHECK(utils::zendnn_bf16_device_check(),
                   "zentorch_matmul bf16 path needs the cpu support "
                   "avx512bf16");
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
  } else {
    LOG(INFO) << "Post Op buffers are not present!\n";
  }
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
                         at::Tensor &beta_bias,
                         const std::vector<at::Tensor> &post_op_buffers,
                         memory &z_mat1, memory &z_mat2, memory &z_bias,
                         memory &z_result, const float &beta,
                         const float &alpha) {

  check_valid_dtypes_for_matmul(mat1, mat2, bias, result, post_op_buffers);
  check_valid_sizes_for_matmul(mat1, mat2, bias, result, post_op_buffers);

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

inline void zentorch_post_ops_selection(
    post_ops &po, std::unordered_map<int, memory> &execute_args,
    const std::vector<int64_t> &post_op_ids,
    const std::vector<at::Tensor> &post_op_buffers, const bool &woq = false) {

  int post_op_ids_size = post_op_ids.size();
  int post_op_buffers_size = post_op_buffers.size();
  std::vector<memory> z_post_op_buffers(post_op_buffers_size);
  int post_op_buffer_idx = 0;
  for (int i = 0; i < post_op_ids_size; i++) {
    int arg_position;
    if (post_op_buffers_size > 0 && post_op_buffer_idx < post_op_buffers_size) {
      if (woq) {
        // for woq we expect n-d post_op_buffers
        // creating 2d memory for n-d post_op_buffers
        // matmul kernel only supports 2d memory
        const memory::format_tag &memory_2d_tag = memory::format_tag::ab;
        const memory::desc &post_op_buffer_2d_desc = memory::desc(
            {get_2d_size_for_tensor(post_op_buffers[post_op_buffer_idx]),
             get_ztype_from_aten(post_op_buffers[post_op_buffer_idx]),
             memory_2d_tag});
        z_post_op_buffers[post_op_buffer_idx] = zen_memory(
            post_op_buffers[post_op_buffer_idx], post_op_buffer_2d_desc);
      } else {
        z_post_op_buffers[post_op_buffer_idx] =
            zen_memory(post_op_buffers[post_op_buffer_idx]);
      }
      LOG(INFO) << "post_op_buffer dimensions: "
                << post_op_buffers[post_op_buffer_idx].sizes();
    }
    // set the post-ops or fusion-ops;
    // by default, fuse = UNARY_POST_OP::NONE,
    switch (post_op_ids[i]) {
    case UNARY_POST_OP::RELU:
      LOG(INFO) << "Setting relu as post op";
      po.append_eltwise(1.0f, algorithm::eltwise_relu, 0.f, 0.f);
      break;
    case UNARY_POST_OP::GELU_TANH:
      LOG(INFO) << "Setting gelu_tanh as post op";
      po.append_eltwise(1.0f, algorithm::eltwise_gelu_tanh, 1.f, 0.f);
      break;
    case UNARY_POST_OP::GELU_ERF:
      LOG(INFO) << "Setting gelu_erf as post op";
      po.append_eltwise(1.0f, algorithm::eltwise_gelu_erf, 1.f, 0.f);
      break;
    case UNARY_POST_OP::SILU:
      LOG(INFO) << "Setting silu as post op";
      po.append_eltwise(1.0f, algorithm::eltwise_swish, 1.f, 0.f);
      break;
    case UNARY_POST_OP::SIGMOID:
      LOG(INFO) << "Setting sigmoid as post op";
      po.append_eltwise(1.0f, algorithm::eltwise_logistic, 1.f, 0.f);
      break;
    case BINARY_POST_OP::MUL:
      LOG(INFO) << "Setting mul as post op";
      po.append_binary(algorithm::binary_mul,
                       z_post_op_buffers[post_op_buffer_idx].get_desc());
      // argument for postop at index=idx in ZenDNN OP primitive.
      arg_position = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(i) | ZENDNN_ARG_SRC_1;
      execute_args.insert(
          {arg_position, z_post_op_buffers[post_op_buffer_idx]});
      post_op_buffer_idx++;
      break;
    case BINARY_POST_OP::ADD:
      LOG(INFO) << "Setting add as post op";
      po.append_binary(algorithm::binary_add,
                       z_post_op_buffers[post_op_buffer_idx].get_desc());
      // argument for postop at index=idx in ZenDNN OP primitive.
      arg_position = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(i) | ZENDNN_ARG_SRC_1;
      execute_args.insert(
          {arg_position, z_post_op_buffers[post_op_buffer_idx]});
      post_op_buffer_idx++;
      break;
    default:
      break;
    }
  }
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
} // namespace zentorch
