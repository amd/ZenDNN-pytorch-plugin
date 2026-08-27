/*****************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EnvReader.hpp"
#include "MatmulUtils.hpp"
#include "Ops.hpp"

#include <c10/util/StringUtil.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/stableivalue_conversions.h>

namespace zentorch {

inline torch::stable::Tensor
stable_expand_as(const torch::stable::Tensor &self,
                 const torch::stable::Tensor &other) {
  std::vector<StableIValue> stack{torch::stable::detail::from(self),
                                  torch::stable::detail::from(other)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::expand_as", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

torch::stable::Tensor zendnnl_matmul_impl(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &bias, torch::stable::Tensor &result,
    const std::vector<int64_t> &post_op_ids,
    const std::vector<torch::stable::Tensor> &post_op_buffers,
    const float &beta, const float &alpha, std::string zentorch_op_name,
    const bool is_weight_const, const bool is_weight_prepacked) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  LOG(INFO) << "input dimensions: [" << c10::Join(", ", input.sizes()) << "]";
  LOG(INFO) << "weight dimensions: [" << c10::Join(", ", weight.sizes()) << "]";
  LOG(INFO) << "result dimensions: [" << c10::Join(", ", result.sizes()) << "]";
  LOG(INFO) << "beta : " << beta << " and alpha : " << alpha;

  // ZenDNNL has implementation for matmul and batched matmul which bypasses the
  // tensor creation and other over-heads to directly start the matmul
  // computation. The usage of the kernel is managed by the env variable
  // "USE_ZENDNN_MATMUL_DIRECT" and when this variable is set to 1 based
  // on the certain conditions, the decision of whether to use this kernel or
  // not is made.

  const int int_env_value =
      EnvReader::getEnvVariableAsInt("USE_ZENDNN_MATMUL_DIRECT");

  const bool bias_defined = bias.defined() && bias.numel() != 0;
  const torch::stable::Tensor beta_bias =
      bias_defined ? (beta == 1 ? bias : mul_by_scalar(bias, beta)) : bias;

  // "validate_zendnnl_direct_kernel_usage" returns a boolean representing
  // whether the direct kernel will be used or not. If true, then the direct
  // kernel will be used and the product is stored in the "result" tensor which
  // is then returned based on the final boolean value
  // "use_zendnnl_direct_kernel". "use_zendnnl_direct_kernel" takes its final
  // value based on the env variable enablement and the return value of
  // "validate_zendnnl_direct_kernel_usage" function.
  const bool use_zendnnl_direct_kernel =
      int_env_value && validate_zendnnl_direct_kernel_usage(
                           input, weight, beta_bias, result, post_op_buffers);
  if (use_zendnnl_direct_kernel) {
    LOG(INFO) << "Using zendnn direct kernel for matmul";
    zendnnl_direct_kernel(input, weight, beta_bias, result, alpha, post_op_ids,
                          post_op_buffers, is_weight_const, is_weight_prepacked,
                          zentorch_op_name);
    return result;
  }

  const torch::stable::Tensor &input_ =
      input.dim() == 1 ? torch::stable::unsqueeze(input, 0) : input;
  const torch::stable::Tensor &weight_ =
      weight.dim() == 1 ? torch::stable::unsqueeze(weight, 1) : weight;
  if (result.dim() == 1) {
    result = torch::stable::unsqueeze(result, 1);
  }
  const torch::stable::Tensor &result_ = result;

  check_valid_dtypes_for_matmul(input_, weight_, bias, result_,
                                post_op_buffers);
  check_valid_sizes_for_matmul(input_, weight_, bias, result_, post_op_buffers);

  if (alpha == 0) {
    if (beta == 0.0f) {
      return torch::stable::zero_(result);
    }
    if (bias_defined) {
      const torch::stable::Tensor scaled_bias =
          (beta == 1.0f) ? bias : mul_by_scalar(bias, beta);
      return torch::stable::copy_(result, scaled_bias);
    }
    result = mul_by_scalar(result, beta);
    return result;
  }
  if (alpha != 1.0f && bias_defined) {
    ZENTORCH_CHECK(!(input_.scalar_type() == c10::ScalarType::BFloat16 ||
                     weight_.scalar_type() == c10::ScalarType::BFloat16 ||
                     input_.scalar_type() == c10::ScalarType::Half ||
                     weight_.scalar_type() == c10::ScalarType::Half),
                   "zentorch_matmul is not supported for bf16 or fp16 "
                   "tensors when bias is defined and alpha is not equal to 1");
  }

  tensor_t mat2_tensor = tensor_t();
  set_zendnnl_tensor_attributes(weight_, mat2_tensor, "weights",
                                is_weight_prepacked);

  tensor_t input_tensor = tensor_t();
  if (input_.dim() == 2) {
    // Set the aligned size for the tensor based on whether it is transposed.
    // Aligned size is used to set the actual size of tensor passed.
    // If the tensor is transposed, align using the second dimension's stride
    // and size. Otherwise, align using the first dimension's size and stride.

    // Strides convey the actual size of tensor.
    // That's why we need to multiply the leading dimension size and leading
    // dimension stride if the tensor is contiguous. If the tensor is
    // transposed, we need to multiply the trailing dimension size and trailing
    // dimension stride.

    const auto tensor_sizes = std::vector<unsigned long>(input_.sizes().begin(),
                                                         input_.sizes().end());
    const auto tensor_strides = std::vector<unsigned long>(
        input_.strides().begin(), input_.strides().end());

    const auto tensor_aligned_sizes =
        is_transposed(input_)
            ? std::vector<unsigned long>{tensor_strides[1], tensor_sizes[1]}
            : std::vector<unsigned long>{tensor_sizes[0], tensor_strides[0]};

    const int64_t nbytes = static_cast<int64_t>(input_.element_size()) *
                           static_cast<int64_t>(tensor_aligned_sizes[0]) *
                           static_cast<int64_t>(tensor_aligned_sizes[1]);

    set_zendnnl_tensor_attributes(input_, input_tensor, "matmul_input",
                                  false /* is_weight_prepacked */, tensor_sizes,
                                  tensor_strides, tensor_aligned_sizes, nbytes);
  } else {
    set_zendnnl_tensor_attributes(input_, input_tensor, "matmul_input");
  }

  tensor_t output_tensor = tensor_t();
  set_zendnnl_tensor_attributes(result_, output_tensor, "matmul_output");

  auto matmul_context = matmul_context_t();
  if (bias_defined) {
    tensor_t bias_tensor = tensor_t();
    const long unsigned int bias_numel =
        static_cast<unsigned long>(beta_bias.numel());
    if (weight_.dim() == 2) {
      set_zendnnl_tensor_attributes(beta_bias, bias_tensor, "bias",
                                    false /* is_weight_prepacked */,
                                    {1, bias_numel}, {bias_numel, 1});
    } else if (weight_.dim() == 3) {
      set_zendnnl_tensor_attributes(
          beta_bias, bias_tensor, "bias", false /* is_weight_prepacked */,
          {1, 1, bias_numel}, {bias_numel, bias_numel, 1});
    } else {
      ZENTORCH_CHECK(false, "Bias shape not supported");
    }
    set_matmul_context_attributes(matmul_context, mat2_tensor, post_op_ids,
                                  alpha, bias_tensor);
  } else {
    set_matmul_context_attributes(matmul_context, mat2_tensor, post_op_ids,
                                  alpha);
  }

  auto matmul_operator = matmul_operator_t();
  set_matmul_operator_attributes(matmul_operator, matmul_context, input_tensor,
                                 output_tensor, post_op_ids, post_op_buffers,
                                 zentorch_op_name);

  const status_t status = matmul_operator.execute();

  ZENTORCH_CHECK(status == status_t::success, "operator ",
                 matmul_operator.get_name(),
                 " execution failed for zentorch_matmul_impl.");

  if (weight.dim() == 1) {
    if (input.dim() == 2) {
      result = torch::stable::squeeze(result, 1);
    } else if (input.dim() == 1) {
      for (int64_t dim = static_cast<int64_t>(result.dim()) - 1; dim >= 0;
           --dim) {
        result = torch::stable::squeeze(result, dim);
      }
    }
  }
  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
  return result;
}

torch::stable::Tensor zentorch_matmul_impl(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &bias, torch::stable::Tensor &result,
    const std::vector<int64_t> &post_op_ids,
    const std::vector<torch::stable::Tensor> &post_op_buffers,
    const float &beta, const float &alpha, std::string zentorch_op_name,
    const bool is_weight_const, const bool is_weight_prepacked) {

  return zendnnl_matmul_impl(input, weight, bias, result, post_op_ids,
                             post_op_buffers, beta, alpha, zentorch_op_name,
                             is_weight_const, is_weight_prepacked);
}

torch::stable::Tensor zentorch_addmm_1dbias(const torch::stable::Tensor &self,
                                            const torch::stable::Tensor &mat1,
                                            const torch::stable::Tensor &mat2,
                                            double beta, double alpha,
                                            std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  ZENTORCH_CHECK(
      (self.dim() == 1 && mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
      "unsupported dims for self, mat1 and mat2");

  torch::stable::Tensor result =
      create_linear_and_matmul_output_tensor(mat1, mat2);

  LOG(INFO) << "Calling zentorch_matmul_impl from " << __FUNCTION__ << "!\n";

  return zentorch_matmul_impl(
      mat1, mat2, self, result,
      {static_cast<int64_t>(UNARY_POST_OP::POST_OP_NONE)} /*post_op_ids*/,
      {} /*post_op_buffers*/, static_cast<float>(beta),
      static_cast<float>(alpha), zentorch_op_name);
}

torch::stable::Tensor zentorch_addmm(const torch::stable::Tensor &self,
                                     const torch::stable::Tensor &mat1,
                                     const torch::stable::Tensor &mat2,
                                     double beta, double alpha,
                                     std::string zentorch_op_name) {

  // if alpha is zero, return beta * self directly from here itself.
  // Dont enter the matmul impl.

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  ZENTORCH_CHECK((mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
                 "unsupported dims for self, mat1 and mat2");

  torch::stable::Tensor result =
      create_linear_and_matmul_output_tensor(mat1, mat2);

  const auto result_sizes =
      std::vector<int64_t>(result.sizes().begin(), result.sizes().end());

  torch::stable::Tensor add_input;

  // Scalar input or 1d input
  if (self.dim() == 0 || self.dim() == 1) {
    LOG(WARNING)
        << "WARNING: Inefficient usage of the addmm function detected.";
    // Reshape and expand the add input tensor to match the
    // output shape of the matrix multiplication.
    add_input = stable_expand_as(self, result);
  } else if (self.dim() == 2 &&
             // check for 1xn
             ((self.size(0) == 1 && self.size(1) == result.size(1)) ||
              // check for mx1
              (self.size(1) == 1 && self.size(0) == result.size(0)) ||
              // check for 1x1
              (self.size(0) == 1 && self.size(1) == 1))) {
    // 2D input tensor matching columns
    // Broascast the input tensor
    add_input = stable_expand_as(self, result);
  } else if (self.sizes().equals(result_sizes)) { // Already compatible
    add_input = self;
  } else {
    ZENTORCH_CHECK(false,
                   "Incompatible dimensions/shape for self tensor in addmm op");
  }

  ZENTORCH_CHECK(add_input.sizes().equals(result_sizes));

  const torch::stable::Tensor empty_bias;
  const float beta_float = static_cast<float>(beta);
  const float alpha_float = static_cast<float>(alpha);

  // When alpha is 0, no matrix multiplication is needed, and bias (here
  // self), multiplied by beta can be returned.
  if (alpha_float == 0.0f) {
    return mul_by_scalar(self, beta_float);
  }

  // Sending the self tensor (this represents the bias in the nn.Module
  // level) as a post op. Since we were passing self directly to matmul impl,
  // this can cause a problem when we are using
  // torch.ops.zentorch.zentorch_addmm directly at the python side with same
  // bias matrix but different inputs. The bias gets corrupted after the
  // first addmm and the subsequent addmms use the corrupted bias tensor,
  // which ultimately results in wrong outputs.

  add_input =
      (beta_float != 1.0f) ? mul_by_scalar(add_input, beta_float) : add_input;
  const std::vector<int64_t> post_op_ids = {
      static_cast<int64_t>(BINARY_POST_OP::ADD),
      static_cast<int64_t>(UNARY_POST_OP::POST_OP_NONE)};

  // TODO
  // Some scenarios necessitate the creation of tensors in a strided fashion on
  // the python side of the plugin. These types of tensors are currently not
  // added in the ZenDNN(L). So, before calling the kernel from the ZenDNN(L)
  // library, we are converting the strided tensors into contiguous tensors
  // using the "get_contiguous_view" utility function. As soon as the library
  // supports these strided tensors, the usage of this utility function will
  // be removed.
  const std::vector<torch::stable::Tensor> post_op_buffers = {
      get_contiguous_view(add_input)};
  return zentorch_matmul_impl(mat1, mat2, empty_bias, result, post_op_ids,
                              post_op_buffers, beta_float, alpha_float,
                              zentorch_op_name);
}

torch::stable::Tensor zentorch_baddbmm(const torch::stable::Tensor &self,
                                       const torch::stable::Tensor &batch1,
                                       const torch::stable::Tensor &batch2,
                                       double beta, double alpha,
                                       std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  ZENTORCH_CHECK(self.numel() != 0, "incorrect self tensor");
  ZENTORCH_CHECK(self.dim() == 3 && batch1.dim() == 3 &&
                     batch2.dim() == 3, // aten::baddbmm
                 "unsupported dims for self, batch1 and batch2");

  const float beta_float = static_cast<float>(beta);
  const float alpha_float = static_cast<float>(alpha);

  // When alpha is 0, no matrix multiplication is needed, and bias (here
  // self), multiplied by beta can be returned.
  if (alpha_float == 0.0f) {
    return mul_by_scalar(self, beta_float);
  }

  // TODO
  // Some scenarios necessitate the creation of tensors in a strided fashion on
  // the python side of the plugin. These types of tensors are currently not
  // added in the ZenDNN(L). So, before calling the kernel from the ZenDNN(L)
  // library, we are converting the strided tensors into contiguous tensors
  // using the "get_contiguous_view" utility function. As soon as the library
  // supports these strided tensors, the usage of this utility function will
  // be removed.
  const torch::stable::Tensor self_ = get_contiguous_view(self);
  const torch::stable::Tensor batch1_ = get_contiguous_view(batch1);
  const torch::stable::Tensor batch2_ = get_contiguous_view(batch2);

  torch::stable::Tensor result =
      create_linear_and_matmul_output_tensor(batch1_, batch2_);

  // TODO
  // Currently there is no kernel that supports 3-d bias addition with 3d
  // matmul. Hence executing this function as a looped addmm. As soon as the
  // kernel is supported from zendnnl, this loop will be removed and kernels
  // shall be used.
  const int batch_count = static_cast<int>(self.size(0));
  for (int idx = 0; idx < batch_count; ++idx) {
    torch::stable::Tensor result_slice = torch::stable::select(result, 0, idx);
    torch::stable::copy_(result_slice,
                         zentorch_addmm(torch::stable::select(self_, 0, idx),
                                        torch::stable::select(batch1_, 0, idx),
                                        torch::stable::select(batch2_, 0, idx),
                                        beta, alpha, zentorch_op_name));
  }

  return result;
}

torch::stable::Tensor zentorch_mm(const torch::stable::Tensor &self,
                                  const torch::stable::Tensor &mat2,
                                  std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  ZENTORCH_CHECK((self.dim() == 2 && mat2.dim() == 2), // aten::mm
                 "unsupported dims for self and mat2");

  torch::stable::Tensor result =
      create_linear_and_matmul_output_tensor(self, mat2);

  const torch::stable::Tensor empty_bias;
  return zentorch_matmul_impl(
      self, mat2, empty_bias, result,
      {static_cast<int64_t>(UNARY_POST_OP::POST_OP_NONE)} /*post_op_ids*/,
      {} /*post_op_buffers*/, 0.0f /* beta */, 1.0f /* alpha */,
      zentorch_op_name);
}

void zentorch_bmm_out(const torch::stable::Tensor &self,
                      const torch::stable::Tensor &mat2,
                      std::string zentorch_op_name,
                      torch::stable::Tensor &out) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  ZENTORCH_CHECK(
      (self.dim() == 3 && mat2.dim() == 3),
      "unsupported dims for self and mat2, expected 3D tensors but got "
      "self.dim()=",
      self.dim(), " and mat2.dim()=", mat2.dim());
  const torch::stable::Tensor empty_bias;
  zentorch_matmul_impl(self, mat2, empty_bias, out, {} /*post_op_ids*/,
                       {} /*post_op_buffers*/, 0.0f /* beta */,
                       1.0f /* alpha */, zentorch_op_name,
                       false /* is_weight_const */);
}

// zentorch_bmm function does not broadcast
torch::stable::Tensor zentorch_bmm(const torch::stable::Tensor &self,
                                   const torch::stable::Tensor &mat2,
                                   std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  torch::stable::Tensor result =
      create_linear_and_matmul_output_tensor(self, mat2);
  zentorch_bmm_out(self, mat2, zentorch_op_name, result);
  return result;
}

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_mm(Tensor self, Tensor mat2, *, str "
        "zentorch_op_name='zentorch::zentorch_mm') -> Tensor");
  m.def("zentorch_bmm(Tensor self, Tensor mat2, str "
        "zentorch_op_name='zentorch::zentorch_bmm') -> Tensor",
        {at::Tag::needs_contiguous_strides});
  m.def("zentorch_bmm.out(Tensor self, Tensor mat2, str "
        "zentorch_op_name='zentorch::zentorch_bmm', *, Tensor(a!) out) "
        "-> ()",
        {at::Tag::needs_contiguous_strides});
  m.def(
      "zentorch_addmm(Tensor self, Tensor mat1, Tensor mat2, *, float beta=1, "
      "float alpha=1, str zentorch_op_name='zentorch::zentorch_addmm') "
      "-> Tensor");
  m.def("zentorch_addmm_1dbias(Tensor self, Tensor mat1, Tensor mat2, *, "
        "float beta=1, float alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_1dbias') -> "
        "Tensor");
  m.def("zentorch_baddbmm(Tensor self, Tensor mat1, Tensor mat2, *, float "
        "beta=1, float alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_baddbmm') -> "
        "Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_mm", TORCH_BOX(&zentorch::zentorch_mm));
  m.impl("zentorch_bmm", TORCH_BOX(&zentorch::zentorch_bmm));
  m.impl("zentorch_bmm.out", TORCH_BOX(&zentorch::zentorch_bmm_out));
  m.impl("zentorch_addmm", TORCH_BOX(&zentorch::zentorch_addmm));
  m.impl("zentorch_addmm_1dbias", TORCH_BOX(&zentorch::zentorch_addmm_1dbias));
  m.impl("zentorch_baddbmm", TORCH_BOX(&zentorch::zentorch_baddbmm));
}
} // namespace zentorch
