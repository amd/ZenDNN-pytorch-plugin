/*****************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EnvReader.hpp"
#include "MatmulUtils.hpp"
#include "Ops.hpp"

namespace zentorch {

at::Tensor zendnn_matmul_impl(const at::Tensor &input, const at::Tensor &weight,
                              const at::Tensor &bias, at::Tensor &result,
                              const std::vector<int64_t> &post_op_ids,
                              const std::vector<at::Tensor> &post_op_buffers,
                              const float &beta, const float &alpha,
                              std::string zentorch_op_name,
                              const bool &is_weight_const = true) {
  // ZenDNN has implementation for matmul and batched matmul which bypasses the
  // primitive creation and other over-heads to directly start the matmul
  // computation. The usage of the kernel is managed by the env variable
  // "USE_ZENDNN_MATMUL_DIRECT" and when this variable is set to 1 based
  // on the certain conditions, the decision of whether to use this kernel or
  // not is made.

  const int int_env_value =
      EnvReader::getEnvVariableAsInt("USE_ZENDNN_MATMUL_DIRECT");

  // "validate_zendnn_direct_kernel_usage" returns a boolean representing
  // whether the direct kernel will be used or not. If true, then the direct
  // kernel will be used and the product is stored in the "result" tensor which
  // is then returned based on the final boolean value
  // "use_zendnn_direct_kernel". "use_zendnn_direct_kernel" takes its final
  // value based on the env variable enablement and the return value of
  // "validate_zendnn_direct_kernel_usage" function.
  const bool use_zendnn_direct_kernel =
      int_env_value &&
      validate_zendnn_direct_kernel_usage(input, weight, bias, result,
                                          post_op_ids, post_op_buffers);
  if (use_zendnn_direct_kernel) {
    LOG(INFO) << "Using zendnn direct kernel for matmul";
    const int64_t post_op_id =
        post_op_ids.size() == 0 ? INT64_MIN : post_op_ids[0];
    zendnn_direct_kernel(input, weight, bias, result, beta, alpha, post_op_id);
    return result;
  }
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  LOG(INFO) << "input dimensions: " << input.sizes();
  LOG(INFO) << "weight dimensions: " << weight.sizes();
  LOG(INFO) << "result dimensions: " << result.sizes();
  LOG(INFO) << "beta : " << beta << " and alpha : " << alpha;
  at::Tensor self_or_result_unsqueezed, input_, weight_, beta_bias;
  memory z_input, z_weight, z_result, z_bias;
  std::tie(self_or_result_unsqueezed, input_, weight_, beta_bias) =
      matmul_tensors_to_memory(input, weight, result, bias, beta_bias,
                               post_op_buffers, z_input, z_weight, z_bias,
                               z_result, beta, alpha, is_weight_const);
  zendnn::primitive_attr op_attr;
  post_ops po;
  const bool bias_defined = bias.numel();
  // If alpha = 0, does not need to actually do gemm computation
  if (alpha == 0) {
    if (beta == 0.0f) {
      return self_or_result_unsqueezed.zero_();
    } else if (bias_defined) {
      // bias is already multiplied by beta
      return self_or_result_unsqueezed.copy_(beta_bias);
    } else {
      return self_or_result_unsqueezed.mul_(beta);
    }
  } else if (alpha != 1.0f) {
    if (bias_defined) {
      // TODO: add support for alpha when bias is defined
      ZENTORCH_CHECK(
          !(input.scalar_type() == c10::ScalarType::BFloat16 ||
            weight.scalar_type() == c10::ScalarType::BFloat16),
          "zentorch_matmul is not supported for bf16 "
          "tensors when bias is defined and alpha is not equal to 1");
    }
    LOG(INFO) << "Setting output scales with alpha = " << alpha;
    op_attr.set_output_scales(0, std::vector<float>(1, alpha));
  }
  std::unordered_map<int, memory> execute_args;
  // Setting Post ops
  zentorch_post_ops_selection(po, execute_args, post_op_ids, post_op_buffers);
  op_attr.set_post_ops(po);
  op_attr.set_plugin_op_name(zentorch_op_name);
  // execute the zendnn::matmul kernel
  zentorch_matmul_execute(execute_args, z_input, z_weight, z_bias, z_result,
                          op_attr, bias_defined);
  if (weight.dim() == 1) {
    if (input.dim() == 2) {
      // aten::mv  >>  [m, 1] tensor will be squeezed to 1-d([m]) tensor
      self_or_result_unsqueezed.squeeze_(1);
    } else if (input.dim() == 1) {
      // aten::dot >>  [1, 1] tensor will be squeezed to 0-d([]) tensor
      self_or_result_unsqueezed.squeeze_();
    }
  }
  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
  return self_or_result_unsqueezed;
}

// There are two custom group matmul ops which are structurally different, but
// have a lot of overlap with respect to initialization of tensors and other
// arguments. These overlaps are covered in the following function called
// zendnn_matmul_group_impl.
std::vector<at::Tensor> zendnn_matmul_group_impl(
    std::vector<at::Tensor> &self_vector, std::vector<at::Tensor> &inputs,
    const at::TensorList &weights, const at::ArrayRef<double> &betas,
    const at::ArrayRef<double> &alphas, const at::IntArrayRef &fuse,
    const bool &is_horizontal, std::string zentorch_op_name) {
  int num_ops = inputs.size();
  std::vector<at::Tensor> output(num_ops);
  std::vector<at::Tensor> bias_vector(num_ops);
  std::vector<at::Tensor> self_or_result_vector(num_ops);
  std::vector<bool> bias_defined_vector(num_ops);
  // ZenDNN Library now supports multiple post ops for one matmul variant in a
  // GroupMLP op. To pass that information from ZenTorch GroupMLP, we create
  // two vectors of vectors:
  //    1. int64_t vector contains the post op identifiers
  //    2. memory contains the ZenDNN memory of the tensor equivalent which are
  //       used when the post op is a binary post op like add or mul.
  std::vector<std::vector<int64_t>> z_post_op_ids(num_ops);
  std::vector<std::vector<memory>> z_post_op_buffers = {};
  std::vector<memory> z_mat1_vector(num_ops);
  std::vector<memory> z_mat2_vector(num_ops);
  std::vector<memory> z_bias_vector(num_ops);
  std::vector<memory> z_result_vector(num_ops);
  std::vector<float> alphas_vector(num_ops);
  std::vector<float> betas_vector(num_ops);
  // If TORCH_CHECK() fails, then the pragma omp parallel for cannot be used
  // we will use at::parallel_for from pytorch instead
  // Disabling at::parallel as a workaround to fix inconsistency in unit tests.
  // TODO : Need to debug why at::parallel is not stable.
  // at::parallel_for(0, num_ops, 0, [&](int64_t start, int64_t end) {
  for (int op_id = 0; op_id < num_ops; op_id++) {
    alphas_vector[op_id] = static_cast<float>(alphas[op_id]);
    betas_vector[op_id] = static_cast<float>(betas[op_id]);
    if (self_vector[op_id].sizes() ==
        c10::IntArrayRef(get_matmul_and_linear_output_sizes(inputs[op_id],
                                                            weights[op_id]))) {
      ZENTORCH_CHECK((self_vector[op_id].dim() == 2 &&
                      inputs[op_id].dim() == 2 &&
                      weights[op_id].dim() == 2), // aten::addmm
                     "unsupported dims for self, mat1 and mat2");
      const at::Tensor empty_bias; // dummy empty bias
      bias_vector[op_id] = empty_bias;
      self_or_result_vector[op_id] = self_vector[op_id];
    } else {
      ZENTORCH_CHECK((self_vector[op_id].dim() == 1 &&
                      inputs[op_id].dim() == 2 &&
                      weights[op_id].dim() == 2), // aten::addmm
                     "unsupported dims for self, mat1 and mat2");

      at::Tensor result =
          create_linear_and_matmul_output_tensor(inputs[op_id], weights[op_id]);
      bias_vector[op_id] = self_vector[op_id];
      self_or_result_vector[op_id] = result;
    }
    at::Tensor self_or_result_unsqueezed, mat1_, mat2_, beta_bias;
    std::vector<at::Tensor> post_op_buffers = {};
    std::tie(self_or_result_unsqueezed, mat1_, mat2_, beta_bias) =
        matmul_tensors_to_memory(
            inputs[op_id], weights[op_id], self_or_result_vector[op_id],
            bias_vector[op_id], beta_bias, post_op_buffers,
            z_mat1_vector[op_id], z_mat2_vector[op_id], z_bias_vector[op_id],
            z_result_vector[op_id], betas_vector[op_id], alphas_vector[op_id]);
    // Populating the bias_defined_vector with the bool equivalent values
    // based on the number of elements in the bias.
    bias_defined_vector[op_id] = bias_vector[op_id].numel();
    if (betas[op_id] != 0.0f && !bias_defined_vector[op_id]) {
      // sets post_ops as add or sum
      LOG(INFO) << "Setting add or sum as post op";
    }
    if (alphas[op_id] != 1.0f) {
      if (bias_defined_vector[op_id]) {
        // TODO: add support for alpha when bias is defined
        ZENTORCH_CHECK(
            !(inputs[op_id].scalar_type() == c10::ScalarType::BFloat16 ||
              weights[op_id].scalar_type() == c10::ScalarType::BFloat16),
            "zentorch_matmul is not supported for bf16 "
            "tensors when bias is defined and alpha != 1");
      }
      LOG(INFO) << "Setting output scales with alpha = " << alphas[op_id];
    }
    // Currently there is a limitation from the PyTorch side that a list of
    // lists cannot be sent from the Python side to the CPP side via bindings
    // using TORCH_LIBRARY_FRAGMENT. So, currently we are using just a single
    // post op and wrapping it in a vector and sending it to ZenDNN Library.
    z_post_op_ids[op_id] = std::vector<int64_t>{fuse[op_id]};
  }
  // });
  if (is_horizontal) {
    LOG(INFO) << "Horizontal GroupMatMul compute in progress...";
    zendnn_custom_op::zendnn_grp_mlp(
        z_mat1_vector, z_mat2_vector, z_bias_vector, alphas_vector,
        betas_vector, bias_defined_vector, z_post_op_ids, z_post_op_buffers,
        z_result_vector, zentorch_op_name.c_str());
    LOG(INFO) << "Horizontal GroupMatMul compute complete...";
  } else {
    LOG(INFO) << "Vertical GroupMatMul compute in progress...";
    std::vector z_input = {z_mat1_vector[0]};
    zendnn_custom_op::zendnn_grp_mlp(
        z_input, z_mat2_vector, z_bias_vector, alphas_vector, betas_vector,
        bias_defined_vector, z_post_op_ids, z_post_op_buffers, z_result_vector,
        zentorch_op_name.c_str());
    LOG(INFO) << "Vertical GroupMatMul compute complete...";
  }
  return self_or_result_vector;
}

at::Tensor zendnnl_matmul_impl(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
    at::Tensor &result, const std::vector<int64_t> &post_op_ids,
    const std::vector<at::Tensor> &post_op_buffers, const float &beta,
    const float &alpha, std::string zentorch_op_name,
    const bool is_weight_const, const bool is_weight_prepacked) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  LOG(INFO) << "input dimensions: " << input.sizes();
  LOG(INFO) << "weight dimensions: " << weight.sizes();
  LOG(INFO) << "result dimensions: " << result.sizes();
  LOG(INFO) << "beta : " << beta << " and alpha : " << alpha;

  // ZenDNNL has implementation for matmul and batched matmul which bypasses the
  // tensor creation and other over-heads to directly start the matmul
  // computation. The usage of the kernel is managed by the env variable
  // "USE_ZENDNN_MATMUL_DIRECT" and when this variable is set to 1 based
  // on the certain conditions, the decision of whether to use this kernel or
  // not is made.

  const int int_env_value =
      EnvReader::getEnvVariableAsInt("USE_ZENDNN_MATMUL_DIRECT");

  const bool bias_defined = bias.numel();
  const at::Tensor beta_bias =
      bias_defined ? (beta == 1 ? bias : bias.mul(beta)) : bias;

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
                          post_op_buffers, is_weight_const,
                          is_weight_prepacked);
    return result;
  }

  const at::Tensor &input_ = input.dim() == 1 ? input.unsqueeze(0) : input;
  const at::Tensor &weight_ = weight.dim() == 1 ? weight.unsqueeze(1) : weight;
  result = result.dim() == 1 ? result.unsqueeze_(1) : result;

  check_valid_dtypes_for_matmul(input_, weight_, bias, result, post_op_buffers);
  check_valid_sizes_for_matmul(input_, weight_, bias, result, post_op_buffers);

  // If alpha = 0, does not need to actually do gemm computation
  if (alpha == 0) {
    if (beta == 0.0f) {
      return result.zero_();
    } else if (bias_defined) {
      at::Tensor beta_bias = (beta == 1.0f) ? bias : bias.mul(beta);
      return result.copy_(beta_bias);
    } else {
      return result.mul_(beta);
    }
  } else if (alpha != 1.0f) {
    if (bias_defined) {
      // TODO: add support for alpha when bias is defined
      ZENTORCH_CHECK(
          !(input_.scalar_type() == c10::ScalarType::BFloat16 ||
            weight_.scalar_type() == c10::ScalarType::BFloat16),
          "zentorch_matmul is not supported for bf16 "
          "tensors when bias is defined and alpha is not equal to 1");
    }
  }

  tensor_t mat2_tensor = tensor_t();
  set_zendnnl_tensor_attributes(weight_, mat2_tensor, "weights",
                                is_weight_prepacked);

  tensor_t input_tensor = tensor_t();
  set_zendnnl_tensor_attributes(input_, input_tensor, "matmul_input");

  tensor_t output_tensor = tensor_t();
  set_zendnnl_tensor_attributes(result, output_tensor, "matmul_output");

  auto matmul_context = matmul_context_t();
  if (bias_defined) {
    tensor_t bias_tensor = tensor_t();
    auto bias_numel = beta_bias.numel();
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

  // define matmul operator
  auto matmul_operator = matmul_operator_t();
  set_matmul_operator_attributes(matmul_operator, matmul_context, input_tensor,
                                 output_tensor, post_op_ids, post_op_buffers);

  status_t status = matmul_operator.execute();

  ZENTORCH_CHECK(status == status_t::success, "operator ",
                 matmul_operator.get_name(),
                 " execution failed for zentorch_matmul_impl.");

  if (weight.dim() == 1) {
    if (input.dim() == 2) {
      // aten::mv  >>  [m, 1] tensor will be squeezed to 1-d([m]) tensor
      result.squeeze_(1);
    } else if (input.dim() == 1) {
      // aten::dot >>  [1, 1] tensor will be squeezed to 0-d([]) tensor
      result.squeeze_();
    }
  }
  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
  return result;
}

at::Tensor zentorch_matmul_impl(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
    at::Tensor &result, const std::vector<int64_t> &post_op_ids,
    const std::vector<at::Tensor> &post_op_buffers, const float &beta,
    const float &alpha, std::string zentorch_op_name,
    const bool is_weight_const, const bool is_weight_prepacked) {
  const int &library = EnvReader::getEnvVariableAsInt(
      "ZENDNN_ZENDNNL"); // 0 would represent ZenDNN and 1 would
                         // represent ZenDNNL. Default library will be ZenDNNL

  return (library == 0)
             ? zendnn_matmul_impl(input, weight, bias, result, post_op_ids,
                                  post_op_buffers, beta, alpha,
                                  zentorch_op_name, is_weight_const)
             : zendnnl_matmul_impl(input, weight, bias, result, post_op_ids,
                                   post_op_buffers, beta, alpha,
                                   zentorch_op_name, is_weight_const,
                                   is_weight_prepacked);
}

template <UNARY_POST_OP fuse>
at::Tensor zentorch_addmm_1dbias(const at::Tensor &self, const at::Tensor &mat1,
                                 const at::Tensor &mat2, at::Scalar beta,
                                 at::Scalar alpha,
                                 std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  ZENTORCH_CHECK(
      (self.dim() == 1 && mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
      "unsupported dims for self, mat1 and mat2");

  at::Tensor result = create_linear_and_matmul_output_tensor(mat1, mat2);

  LOG(INFO) << "Calling zentorch_matmul_impl from " << __FUNCTION__ << "!\n";

  return zentorch_matmul_impl(mat1, mat2, self, result, {fuse} /*post_op_ids*/,
                              {} /*post_op_buffers*/, beta.to<float>(),
                              alpha.to<float>(), zentorch_op_name);
}

// unary-binary fusions and binary fusions will be handle by this
template <UNARY_POST_OP fuse1, BINARY_POST_OP fuse2>
at::Tensor zentorch_addmm_1dbias_unary_binary(const at::Tensor &self,
                                              const at::Tensor &mat1,
                                              const at::Tensor &mat2,
                                              const at::Tensor &binary_input,
                                              at::Scalar beta, at::Scalar alpha,
                                              std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  ZENTORCH_CHECK(
      (self.dim() == 1 && mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
      "unsupported dims for self, mat1 and mat2");

  at::Tensor result = create_linear_and_matmul_output_tensor(mat1, mat2);

  std::vector<int64_t> post_op_ids = {fuse1, fuse2};
  std::vector<at::Tensor> post_op_buffers = {binary_input};

  LOG(INFO) << "Calling zentorch_matmul_impl from " << __FUNCTION__ << "!\n";

  return zentorch_matmul_impl(mat1, mat2, self, result, post_op_ids,
                              post_op_buffers, beta.to<float>(),
                              alpha.to<float>(), zentorch_op_name);
}

template <BINARY_POST_OP fuse1, BINARY_POST_OP fuse2>
at::Tensor zentorch_addmm_1dbias_binary_binary(
    const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const at::Tensor &binary1_input, const at::Tensor &binary2_input,
    at::Scalar beta, at::Scalar alpha, std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  ZENTORCH_CHECK((binary1_input.dim() == 2 && binary2_input.dim() == 2 &&
                  mat1.dim() == 2 && mat2.dim() == 2),
                 "unsupported dims for mat1, mat2, "
                 "binary1_input and binary2_input");

  at::Tensor result = create_linear_and_matmul_output_tensor(mat1, mat2);

  ZENTORCH_CHECK((binary1_input.sizes() == c10::IntArrayRef(result.sizes()) &&
                  binary2_input.sizes() == c10::IntArrayRef(result.sizes())),
                 "unsupported sizes for mat1, mat2, "
                 "binary1_input and binary2_input");
  ZENTORCH_CHECK((self.dim() == 1), "unsupported dims for self, mat1 and mat2");

  std::vector<int64_t> post_op_ids = {fuse1, fuse2};
  std::vector<at::Tensor> post_op_buffers = {binary1_input, binary2_input};

  LOG(INFO) << "Calling zentorch_matmul_impl from " << __FUNCTION__ << "!\n";

  return zentorch_matmul_impl(mat1, mat2, self, result, post_op_ids,
                              post_op_buffers, beta.to<float>(),
                              alpha.to<float>(), zentorch_op_name);
}

template <UNARY_POST_OP fuse>
at::Tensor zentorch_addmm(const at::Tensor &self, const at::Tensor &mat1,
                          const at::Tensor &mat2, at::Scalar beta,
                          at::Scalar alpha, std::string zentorch_op_name) {

  // if alpha is zero, return beta * self directly from here itself.
  // Dont enter the matmul impl.

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  ZENTORCH_CHECK((mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
                 "unsupported dims for self, mat1 and mat2");

  at::Tensor result = create_linear_and_matmul_output_tensor(mat1, mat2);

  at::Tensor add_input;

  // Scalar input or 1d input
  if (self.dim() == 0 or self.dim() == 1) {
    LOG(WARNING)
        << "WARNING: Inefficient usage of the addmm function detected.";
    // Reshape and expand the add input tensor to match the
    // output shape of the matrix multiplication.
    add_input = self.expand_as(result);
  } else if (self.dim() == 2 &&
             // check for 1xn
             ((self.size(0) == 1 && self.size(1) == result.sizes()[1]) ||
              // check for mx1
              (self.size(1) == 1 && self.size(0) == result.sizes()[0]) ||
              // check for 1x1
              (self.size(0) == 1 && self.size(1) == 1))) {
    // 2D input tensor matching columns
    // Broascast the input tensor
    add_input = self.expand_as(result);
  } else if (self.sizes() == result.sizes()) { // Already compatible
    add_input = self;
  } else {
    ZENTORCH_CHECK(false,
                   "Incompatible dimensions/shape for self tensor in addmm op");
  }
  // Here the if condition checks if the matrices are compatible for matrix
  // multiplication and bias addition for any general n-d case. But the
  // TORCH_CHECK conditions specifically checks for the dimensionality
  // conditions which are supported by zentorch_addmm

  ZENTORCH_CHECK(add_input.sizes() == result.sizes());

  const at::Tensor empty_bias;
  float beta_float = beta.to<float>();
  float alpha_float = alpha.to<float>();

  // When alpha is 0, no matrix multiplication is needed, and bias (here
  // self), multiplied by beta can be returned.
  if (alpha_float == 0.0f) {
    return self.mul(beta_float);
  }

  // Sending the self tensor (this represents the bias in the nn.Module
  // level) as a post op. Since we were passing self directly to matmul impl,
  // this can cause a problem when we are using
  // torch.ops.zentorch.zentorch_addmm directly at the python side with same
  // bias matrix but different inputs. The bias gets corrupted after the
  // first addmm and the subsequent addmms use the corrupted bias tensor,
  // which ultimately results in wrong outputs.

  add_input = (beta_float != 1.0f) ? add_input.mul(beta_float) : add_input;
  std::vector<int64_t> post_op_ids = {BINARY_POST_OP::ADD, fuse};

  // TODO
  // Some scenarios necessitate the creation of tensors in a strided fashion on
  // the python side of the plugin. These types of tensors are currently not
  // added in the ZenDNN(L). So, before calling the kernel from the ZenDNN(L)
  // library, we are converting the strided tensors into contiguous tensors
  // using the "get_contiguous_view" utility function. As soon as the library
  // supports these strided tensors, the usage of this utility function will
  // be removed.
  std::vector<at::Tensor> post_op_buffers = {get_contiguous_view(add_input)};
  return zentorch_matmul_impl(mat1, mat2, empty_bias, result, post_op_ids,
                              post_op_buffers, beta_float, alpha_float,
                              zentorch_op_name);
}

at::Tensor zentorch_baddbmm(const at::Tensor &self, const at::Tensor &batch1,
                            const at::Tensor &batch2, at::Scalar beta,
                            at::Scalar alpha, std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  ZENTORCH_CHECK(self.numel() != 0, "incorrect self tensor");
  ZENTORCH_CHECK(self.dim() == 3 && batch1.dim() == 3 &&
                     batch2.dim() == 3, // aten::baddbmm
                 "unsupported dims for self, batch1 and batch2");

  float beta_float = beta.to<float>();
  float alpha_float = alpha.to<float>();

  // When alpha is 0, no matrix multiplication is needed, and bias (here
  // self), multiplied by beta can be returned.
  if (alpha_float == 0.0f) {
    return self.mul(beta_float);
  }

  // TODO
  // Some scenarios necessitate the creation of tensors in a strided fashion on
  // the python side of the plugin. These types of tensors are currently not
  // added in the ZenDNN(L). So, before calling the kernel from the ZenDNN(L)
  // library, we are converting the strided tensors into contiguous tensors
  // using the "get_contiguous_view" utility function. As soon as the library
  // supports these strided tensors, the usage of this utility function will
  // be removed.
  const at::Tensor &self_ = get_contiguous_view(self);
  const at::Tensor &batch1_ = get_contiguous_view(batch1);
  const at::Tensor &batch2_ = get_contiguous_view(batch2);

  at::Tensor result = create_linear_and_matmul_output_tensor(batch1_, batch2_);

  // TODO
  // Currently there is no kernel that supports 3-d bias addition with 3d
  // matmul. Hence executing this function as a looped addmm. As soon as the
  // kernel is supported from zendnnl, this loop will be removed and kernels
  // shall be used.
  for (int idx = 0; idx < self.sizes()[0]; idx++) {
    result[idx] = zentorch_addmm<UNARY_POST_OP::POST_OP_NONE>(
        self_[idx], batch1_[idx], batch2_[idx], beta, alpha, zentorch_op_name);
  }

  return result;
}

template <UNARY_POST_OP fuse>
at::Tensor zentorch_mm(const at::Tensor &self, const at::Tensor &mat2,
                       std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  ZENTORCH_CHECK((self.dim() == 2 && mat2.dim() == 2), // aten::mm
                 "unsupported dims for self and mat2");

  at::Tensor result = create_linear_and_matmul_output_tensor(self, mat2);

  at::Tensor empty_bias;
  return zentorch_matmul_impl(self, mat2, empty_bias, result,
                              {fuse} /*post_op_ids*/, {} /*post_op_buffers*/,
                              0.0f /* beta */, 1.0f /* alpha */,
                              zentorch_op_name);
}

// zentorch_bmm function does not broadcast
at::Tensor zentorch_bmm(const at::Tensor &self, const at::Tensor &mat2,
                        std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  ZENTORCH_CHECK((self.dim() == 3 && mat2.dim() == 3), // aten::bmm
                 "unsupported dims for self and mat2");

  // TODO
  // Some scenarios necessitate the creation of tensors in a strided fashion on
  // the python side of the plugin. These types of tensors are currently not
  // added in the ZenDNN(L). So, before calling the kernel from the ZenDNN(L)
  // library, we are converting the strided tensors into contiguous tensors
  // using the "get_contiguous_view" utility function. As soon as the library
  // supports these strided tensors, the usage of this utility function will
  // be removed.
  const at::Tensor &self_ = get_contiguous_view(self);
  const at::Tensor &mat2_ = get_contiguous_view(mat2);

  at::Tensor result = create_linear_and_matmul_output_tensor(self, mat2);

  const at::Tensor empty_bias;
  return zentorch_matmul_impl(self_, mat2_, empty_bias, result,
                              {} /*post_op_ids*/, {} /*post_op_buffers*/,
                              0.0f /* beta */, 1.0f /* alpha */,
                              zentorch_op_name, false /* is_weight_const */);
}

// unary-binary fusions and binary fusions will be handle by this
template <UNARY_POST_OP fuse1, BINARY_POST_OP fuse2>
at::Tensor
zentorch_addmm_unary_binary(const at::Tensor &bias, const at::Tensor &mat1,
                            const at::Tensor &mat2,
                            const at::Tensor &binary_input, at::Scalar beta,
                            at::Scalar alpha, std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  ZENTORCH_CHECK((mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
                 "unsupported dims for self, mat1 and mat2");
  at::Tensor result = create_linear_and_matmul_output_tensor(mat1, mat2);

  float beta_float = beta.to<float>();
  at::Tensor beta_bias = beta_float == 1 ? bias : bias.mul(beta);

  const at::Tensor empty_bias;
  std::vector<int64_t> post_op_ids = {BINARY_POST_OP::ADD, fuse1, fuse2};
  std::vector<at::Tensor> post_op_buffers;

  if (beta_bias.sizes() == result.sizes()) {
    post_op_buffers = {beta_bias, binary_input};
  } else if (beta_bias.dim() == 1) {
    LOG(WARNING)
        << "WARNING: Inefficient usage of the addmm function detected.";
    post_op_buffers = {beta_bias.expand_as(result), binary_input};
  }

  return zentorch_matmul_impl(mat1, mat2, empty_bias, result, post_op_ids,
                              post_op_buffers, beta_float, alpha.to<float>(),
                              zentorch_op_name);
}

template <UNARY_POST_OP fuse1, BINARY_POST_OP fuse2>
at::Tensor zentorch_mm_unary_binary(const at::Tensor &mat1,
                                    const at::Tensor &mat2,
                                    const at::Tensor &binary_input,
                                    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  ZENTORCH_CHECK((mat1.dim() == 2 && mat2.dim() == 2),
                 "unsupported dims for mat1 and mat2")

  at::Tensor result = create_linear_and_matmul_output_tensor(mat1, mat2);

  at::Tensor empty_bias;
  std::vector<int64_t> post_op_ids = {fuse1, fuse2};
  std::vector<at::Tensor> post_op_buffers = {binary_input};
  return zentorch_matmul_impl(mat1, mat2, empty_bias, result, post_op_ids,
                              post_op_buffers, 0.0f /* beta */,
                              1.0f /* alpha */, zentorch_op_name);
}

at::Tensor
zentorch_vertical_mlp_group(at::TensorList self, const at::Tensor &input,
                            at::TensorList weights, at::ArrayRef<double> betas,
                            at::ArrayRef<double> alphas, at::IntArrayRef fuse,
                            std::string zentorch_op_name) {
  // self = alpha * input * weights + beta * self
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  int num_ops = weights.size();
  std::vector<at::Tensor> input_vector(num_ops);
  std::vector<at::Tensor> output_vector(num_ops);
  std::vector<at::Tensor> self_vector(num_ops);
  at::Tensor mlp_input, dummy_output;
  for (int i = 0; i < num_ops; i++) {
    if (i == 0)
      mlp_input = input;
    else
      mlp_input = dummy_output;

    std::vector<int64_t> output_sizes =
        get_matmul_and_linear_output_sizes(mlp_input, weights[i]);
    std::vector<int64_t> output_strides =
        get_matmul_and_linear_output_strides(output_sizes);
    dummy_output = at::detail::empty_strided_cpu(output_sizes, output_strides,
                                                 input.options());
    input_vector[i] = mlp_input;
    self_vector[i] = self[i];
  }

  // TODO
  // The under-the-hood impl functions are now prefixed with the backend library
  // they are using to do the computation. Since this group matmul
  // implementation is not supported in ZenDNN(L) yet, this function is using
  // the zendnn library as of now. As soon as the ZenDNN(L) library supports
  // custom operators, this will be removed and the function from ZenDNN(L) will
  // be added.
  output_vector =
      zendnn_matmul_group_impl(self_vector, input_vector, weights, betas,
                               alphas, fuse, false, zentorch_op_name);
  return output_vector[num_ops - 1];
}
std::vector<at::Tensor>
zentorch_attn_qkv_fusion(at::TensorList self, at::TensorList inputs,
                         at::TensorList weights, at::ArrayRef<double> betas,
                         at::ArrayRef<double> alphas, at::IntArrayRef fuse,
                         at::IntArrayRef is_zentorch_mm,
                         std::string zentorch_op_name) {
  // self = alpha * inputs * weights.t + beta * self
  LOG(INFO) << "In zentorch_attention_horizontal_matmul_group_mlp...\n";
  int num_ops = inputs.size();
  std::vector<at::Tensor> input_vector(num_ops);
  std::vector<at::Tensor> self_vector(num_ops);
  LOG(INFO) << "Executing function: " << __FUNCTION__;
  for (int i = 0; i < num_ops; i++) {
    input_vector[i] = inputs[i];
    if (is_zentorch_mm[i] == 1) {
      self_vector[i] =
          create_linear_and_matmul_output_tensor(inputs[i], weights[i]);
    } else {
      self_vector[i] = self[i];
    }
  }
  // TODO
  // The under-the-hood impl functions are now prefixed with the backend library
  // they are using to do the computation. Since this group matmul
  // implementation is not supported in ZenDNN(L) yet, this function is using
  // the zendnn library as of now. As soon as the ZenDNN(L) library supports
  // custom operators, this will be removed and the function from ZenDNN(L) will
  // be added.
  return zendnn_matmul_group_impl(self_vector, input_vector, weights, betas,
                                  alphas, fuse, true, zentorch_op_name);
}

TORCH_LIBRARY(zentorch, m) {
  m.def("zentorch_mm(Tensor self, Tensor mat2, *, str "
        "zentorch_op_name='zentorch::zentorch_mm') -> Tensor");
  m.def("zentorch_mm_relu(Tensor self, Tensor mat2, *, str "
        "zentorch_op_name='zentorch::zentorch_mm_relu') -> Tensor");
  m.def("zentorch_mm_gelu_tanh(Tensor self, Tensor mat2, *, str "
        "zentorch_op_name='zentorch::zentorch_mm_gelu_tanh') -> Tensor");
  m.def("zentorch_mm_gelu_erf(Tensor self, Tensor mat2, *, str "
        "zentorch_op_name='zentorch::zentorch_mm_gelu_erf') -> Tensor");
  m.def("zentorch_mm_silu(Tensor self, Tensor mat2, *, str "
        "zentorch_op_name='zentorch::zentorch_mm_silu') -> Tensor");
  m.def("zentorch_bmm(Tensor self, Tensor mat2, str "
        "zentorch_op_name='zentorch::zentorch_bmm') -> Tensor");
  m.def(
      "zentorch_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, "
      "Scalar alpha=1, str zentorch_op_name='zentorch::zentorch_addmm') "
      "-> Tensor");
  m.def("zentorch_addmm_relu(Tensor self, Tensor mat1, Tensor mat2, *, Scalar "
        "beta=1, "
        "Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_relu') -> Tensor");
  m.def("zentorch_addmm_gelu_tanh(Tensor self, Tensor mat1, Tensor mat2, *, "
        "Scalar beta=1, "
        "Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_gelu_tanh') -> "
        "Tensor");
  m.def("zentorch_addmm_gelu_erf(Tensor self, Tensor mat1, Tensor mat2, *, "
        "Scalar beta=1, "
        "Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_gelu_erf') -> "
        "Tensor");
  m.def("zentorch_addmm_silu(Tensor self, Tensor mat1, Tensor mat2, *, "
        "Scalar beta=1, "
        "Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_silu') -> "
        "Tensor");
  // for 1d bias
  m.def("zentorch_addmm_1dbias(Tensor self, Tensor mat1, Tensor mat2, *, "
        "Scalar beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_1dbias') -> "
        "Tensor");
  m.def("zentorch_addmm_1dbias_add( Tensor self, Tensor mat1, "
        "Tensor mat2,Tensor add_input, *, "
        "Scalar beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_1dbias_add') -> "
        "Tensor");
  m.def("zentorch_addmm_1dbias_add_add( Tensor self, "
        "Tensor mat1, Tensor mat2, Tensor add1_input, Tensor add2_input, *,"
        "Scalar beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_1dbias_add_add') -> "
        "Tensor");
  m.def("zentorch_addmm_1dbias_mul_add( Tensor self, "
        "Tensor mat1, Tensor mat2, Tensor mul_input, Tensor add_input, *,"
        "Scalar beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_1dbias_mul_add') -> "
        "Tensor");
  m.def("zentorch_addmm_1dbias_relu(Tensor self, Tensor mat1, Tensor mat2, *, "
        "Scalar beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_1dbias_relu') -> "
        "Tensor");
  m.def("zentorch_addmm_1dbias_gelu_tanh(Tensor self, Tensor mat1, Tensor mat2,"
        " *, Scalar beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_1dbias_gelu_tanh') "
        "-> Tensor");
  m.def("zentorch_addmm_1dbias_gelu_erf(Tensor self, Tensor mat1, Tensor mat2,"
        " *, Scalar beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_1dbias_gelu_erf') "
        "-> Tensor");
  m.def("zentorch_addmm_1dbias_silu(Tensor self, Tensor mat1, Tensor mat2,"
        " *, Scalar beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_1dbias_silu') "
        "-> Tensor");
  m.def("zentorch_baddbmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar "
        "beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_baddbmm') -> "
        "Tensor");
  m.def("zentorch_mm_silu_mul(Tensor mat1, Tensor mat2, Tensor mul_input,"
        "str zentorch_op_name='zentorch::zentorch_mm_silu_mul') -> "
        "Tensor");
  m.def("zentorch_addmm_silu_mul(Tensor bias, Tensor mat1, Tensor mat2, "
        "Tensor mul_input, Scalar beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_silu_mul') -> "
        "Tensor");
  m.def("zentorch_addmm_1dbias_silu_mul(Tensor bias, Tensor mat1, Tensor "
        "mat2, Tensor mul_input, Scalar beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_1dbias_silu_mul') -> "
        "Tensor");
  m.def("zentorch_vertical_mlp_group(Tensor[] self, Tensor inputs, "
        "Tensor[] weight, float[] betas, float[] alphas, "
        "int[] fuse, str zentorch_op_name = "
        "'zentorch::zentorch_vertical_mlp_group') -> "
        "Tensor");
  m.def("zentorch_attn_qkv_fusion(Tensor[] self, Tensor[] inputs, "
        "Tensor[] weights, "
        "float[] betas, float[] alphas, int[] fuse, int[] is_zentorchmm, str "
        "zentorch_op_name = "
        "'zentorch::zentorch_attn_qkv_fusion') -> "
        "Tensor[]");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_mm", zentorch_mm<UNARY_POST_OP::POST_OP_NONE>);
  m.impl("zentorch_mm_relu", zentorch_mm<UNARY_POST_OP::RELU>);
  m.impl("zentorch_mm_gelu_tanh", zentorch_mm<UNARY_POST_OP::GELU_TANH>);
  m.impl("zentorch_mm_gelu_erf", zentorch_mm<UNARY_POST_OP::GELU_ERF>);
  m.impl("zentorch_mm_silu", zentorch_mm<UNARY_POST_OP::SILU>);
  m.impl("zentorch_bmm", zentorch_bmm);
  m.impl("zentorch_addmm", zentorch_addmm<UNARY_POST_OP::POST_OP_NONE>);
  m.impl("zentorch_addmm_relu", zentorch_addmm<UNARY_POST_OP::RELU>);
  m.impl("zentorch_addmm_gelu_tanh", zentorch_addmm<UNARY_POST_OP::GELU_TANH>);
  m.impl("zentorch_addmm_gelu_erf", zentorch_addmm<UNARY_POST_OP::GELU_ERF>);
  m.impl("zentorch_addmm_silu", zentorch_addmm<UNARY_POST_OP::SILU>);
  m.impl("zentorch_addmm_1dbias",
         zentorch_addmm_1dbias<UNARY_POST_OP::POST_OP_NONE>);
  m.impl("zentorch_addmm_1dbias_add",
         zentorch_addmm_1dbias_unary_binary<UNARY_POST_OP::POST_OP_NONE,
                                            BINARY_POST_OP::ADD>);
  m.impl("zentorch_addmm_1dbias_add_add",
         zentorch_addmm_1dbias_binary_binary<BINARY_POST_OP::ADD,
                                             BINARY_POST_OP::ADD>);
  m.impl("zentorch_addmm_1dbias_mul_add",
         zentorch_addmm_1dbias_binary_binary<BINARY_POST_OP::MUL,
                                             BINARY_POST_OP::ADD>);
  m.impl("zentorch_addmm_1dbias_relu",
         zentorch_addmm_1dbias<UNARY_POST_OP::RELU>);
  m.impl("zentorch_addmm_1dbias_gelu_tanh",
         zentorch_addmm_1dbias<UNARY_POST_OP::GELU_TANH>);
  m.impl("zentorch_addmm_1dbias_gelu_erf",
         zentorch_addmm_1dbias<UNARY_POST_OP::GELU_ERF>);
  m.impl("zentorch_addmm_1dbias_silu",
         zentorch_addmm_1dbias<UNARY_POST_OP::SILU>);
  m.impl("zentorch_baddbmm", zentorch_baddbmm);
  m.impl("zentorch_mm_silu_mul",
         zentorch_mm_unary_binary<UNARY_POST_OP::SILU, BINARY_POST_OP::MUL>);
  m.impl("zentorch_addmm_silu_mul",
         zentorch_addmm_unary_binary<UNARY_POST_OP::SILU, BINARY_POST_OP::MUL>);
  m.impl("zentorch_addmm_1dbias_silu_mul",
         zentorch_addmm_1dbias_unary_binary<UNARY_POST_OP::SILU,
                                            BINARY_POST_OP::MUL>);
  m.impl("zentorch_vertical_mlp_group", zentorch_vertical_mlp_group);
  m.impl("zentorch_attn_qkv_fusion", zentorch_attn_qkv_fusion);
}
} // namespace zentorch
