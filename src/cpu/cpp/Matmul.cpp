/******************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "MatmulUtils.hpp"
#include "Memory.hpp"
#include "Ops.hpp"

namespace zentorch {

using namespace zendnn;

at::Tensor zentorch_matmul_impl(const at::Tensor &input,
                                const at::Tensor &weight,
                                const at::Tensor &bias, at::Tensor &result,
                                const std::vector<int64_t> &post_op_ids,
                                const std::vector<at::Tensor> &post_op_buffers,
                                const float &beta, const float &alpha,
                                std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  LOG(INFO) << "input dimensions: " << input.sizes();
  LOG(INFO) << "weight dimensions: " << weight.sizes();
  LOG(INFO) << "result dimensions: " << result.sizes();
  LOG(INFO) << "beta : " << beta << " and alpha : " << alpha;

  at::Tensor self_or_result_unsqueezed, input_, weight_, beta_bias;
  memory z_input, z_weight, z_result, z_bias;

  std::tie(self_or_result_unsqueezed, input_, weight_, beta_bias) =
      matmul_tensors_to_memory(input, weight, result, bias, beta_bias, z_input,
                               z_weight, z_bias, z_result, beta, alpha);

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
      TORCH_CHECK(!(input.scalar_type() == c10::ScalarType::BFloat16 ||
                    weight.scalar_type() == c10::ScalarType::BFloat16),
                  "zentorch_matmul: zentorch_matmul is not supported for bf16 "
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
// zentorch_matmul_group_impl.
std::vector<at::Tensor> zentorch_matmul_group_impl(
    std::vector<at::Tensor> &self_vector, std::vector<at::Tensor> &inputs,
    const at::TensorList &weights, const at::ArrayRef<double> &betas,
    const at::ArrayRef<double> &alphas, const at::IntArrayRef &fuse,
    const bool &is_horizontal, std::string zentorch_op_name) {
  int num_ops = inputs.size();
  std::vector<at::Tensor> output(num_ops);
  std::vector<at::Tensor> bias_vector(num_ops);
  std::vector<at::Tensor> self_or_result_vector(num_ops);
  std::vector<bool> bias_defined_vector(num_ops);
  std::vector<int64_t> fuse_vector(num_ops);
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
  for (int i = 0; i < num_ops; i++) {
    alphas_vector[i] = static_cast<float>(alphas[i]);
    betas_vector[i] = static_cast<float>(betas[i]);
    if (self_vector[i].sizes() ==
        c10::IntArrayRef(
            get_matmul_and_linear_output_sizes(inputs[i], weights[i]))) {
      TORCH_CHECK((self_vector[i].dim() == 2 && inputs[i].dim() == 2 &&
                   weights[i].dim() == 2), // aten::addmm
                  "zentorch_addmm:  unsupported dims for self, mat1 and mat2");
      const at::Tensor empty_bias; // dummy empty bias
      bias_vector[i] = empty_bias;
      self_or_result_vector[i] = self_vector[i];
    } else {
      TORCH_CHECK((self_vector[i].dim() == 1 && inputs[i].dim() == 2 &&
                   weights[i].dim() == 2), // aten::addmm
                  "zentorch_addmm: unsupported dims for self, mat1 and mat2");
      // Array access is faster than .size(n)
      const auto mat1_sizes = inputs[i].sizes();
      const auto mat2_sizes = weights[i].sizes();
      const auto self_sizes = self_vector[i].sizes();
      TORCH_CHECK(
          self_sizes[0] == mat2_sizes[1] && mat1_sizes[1] == mat2_sizes[0],
          "input shape is incompatible with matrix multiplication (",
          mat1_sizes[0], "x", mat1_sizes[1], " @ ", mat2_sizes[0], "x",
          mat2_sizes[1], " != ", mat1_sizes[0], "x", self_sizes[0], ")");

      at::Tensor result =
          at::empty(get_matmul_and_linear_output_sizes(inputs[i], weights[i]),
                    inputs[i].options());
      bias_vector[i] = self_vector[i];
      self_or_result_vector[i] = result;
    }

    at::Tensor self_or_result_unsqueezed, mat1_, mat2_, beta_bias;

    std::tie(self_or_result_unsqueezed, mat1_, mat2_, beta_bias) =
        matmul_tensors_to_memory(
            inputs[i], weights[i], self_or_result_vector[i], bias_vector[i],
            beta_bias, z_mat1_vector[i], z_mat2_vector[i], z_bias_vector[i],
            z_result_vector[i], betas_vector[i], alphas_vector[i]);

    // Populating the bias_defined_vector with the bool equivalent values
    // based on the number of elements in the bias.
    bias_defined_vector[i] = bias_vector[i].numel();

    if (betas[i] != 0.0f && !bias_defined_vector[i]) {
      // sets post_ops as add or sum
      LOG(INFO) << "Setting add or sum as post op";
    }
    if (alphas[i] != 1.0f) {
      if (bias_defined_vector[i]) {
        // TODO: add support for alpha when bias is defined
        TORCH_CHECK(
            !(inputs[i].scalar_type() == c10::ScalarType::BFloat16 ||
              weights[i].scalar_type() == c10::ScalarType::BFloat16),
            "zentorch_matmul: zentorch_matmul is not supported for bf16 "
            "tensors when bias is defined and alpha != 1");
      }
      LOG(INFO) << "Setting output scales with alpha = " << alphas[i];
    }
    fuse_vector[i] = fuse[i];
  }
  // });

  if (is_horizontal) {

    LOG(INFO) << "Horizontal GroupMatMul compute in progress...";
    zendnn_custom_op::zendnn_grp_mlp(z_mat1_vector, z_mat2_vector,
                                     z_bias_vector, alphas_vector, betas_vector,
                                     bias_defined_vector, fuse_vector,
                                     z_result_vector, zentorch_op_name.c_str());

    LOG(INFO) << "Horizontal GroupMatMul compute complete...";
  } else {
    LOG(INFO) << "Vertical GroupMatMul compute in progress...";

    std::vector z_input = {z_mat1_vector[0]};
    zendnn_custom_op::zendnn_grp_mlp(z_input, z_mat2_vector, z_bias_vector,
                                     alphas_vector, betas_vector,
                                     bias_defined_vector, fuse_vector,
                                     z_result_vector, zentorch_op_name.c_str());

    LOG(INFO) << "Vertical GroupMatMul compute complete...";
  }

  return self_or_result_vector;
}

// for 1d bias
// This template is taking care of only unary post op
template <POST_OP fuse>
at::Tensor zentorch_addmm_1dbias(const at::Tensor &self, const at::Tensor &mat1,
                                 const at::Tensor &mat2, const at::Scalar &beta,
                                 const at::Scalar &alpha,
                                 std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  TORCH_CHECK(
      (self.dim() == 1 && mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
      "zentorch_addmm_1dbias: unsupported dims for self, mat1 and mat2");

  // Array access is faster than .size(n)
  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();
  const auto self_sizes = self.sizes();

  TORCH_CHECK(self_sizes[0] == mat2_sizes[1] && mat1_sizes[1] == mat2_sizes[0],
              "input shape is incompatible with matrix multiplication (",
              mat1_sizes[0], "x", mat1_sizes[1], " @ ", mat2_sizes[0], "x",
              mat2_sizes[1], " != ", mat1_sizes[0], "x", self_sizes[0], ")");

  at::Tensor result =
      at::empty(get_matmul_and_linear_output_sizes(mat1, mat2), mat1.options());

  std::vector<at::Tensor> post_op_buffers = {};
  std::vector<int64_t> post_op_ids = {fuse};

  LOG(INFO) << "Calling zentorch_matmul_impl from " << __FUNCTION__ << "!\n";
  return zentorch_matmul_impl(mat1, mat2, self, result, post_op_ids,
                              post_op_buffers, beta.to<float>(),
                              alpha.to<float>(), zentorch_op_name);
}

at::Tensor
zentorch_addmm_1dbias_add(const at::Tensor &self, const at::Tensor &mat1,
                          const at::Tensor &mat2, const at::Tensor &add_input,
                          const at::Scalar &beta, const at::Scalar &alpha,
                          std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  TORCH_CHECK((add_input.dim() == 2 && mat1.dim() == 2 && mat2.dim() == 2),
              "zentorch_addmm_1dbias_add: unsupported dims for mat1, mat2 and "
              "add_input");
  TORCH_CHECK(
      (add_input.sizes() ==
       c10::IntArrayRef(get_matmul_and_linear_output_sizes(mat1, mat2))),
      "zentorch_addmm_1dbias_add: unsupported sizes for mat1, mat2 and "
      "add_input");

  TORCH_CHECK((self.dim() == 1),
              "zentorch_addmm:  unsupported dims for self, mat1 and mat2");
  at::Tensor result = at::empty(add_input.sizes(), add_input.options());

  std::vector<at::Tensor> post_op_buffers = {add_input};
  std::vector<int64_t> post_op_ids = {POST_OP::ADD};

  LOG(INFO) << "Calling zentorch_matmul_impl from " << __FUNCTION__ << "!\n";
  return zentorch_matmul_impl(mat1, mat2, self, result, post_op_ids,
                              post_op_buffers, beta.to<float>(),
                              alpha.to<float>(), zentorch_op_name);
}

at::Tensor zentorch_addmm_1dbias_add_add(
    const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const at::Tensor &add1_input, const at::Tensor &add2_input,
    const at::Scalar &beta, const at::Scalar &alpha,
    std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  TORCH_CHECK((add1_input.dim() == 2 && add2_input.dim() == 2 &&
               mat1.dim() == 2 && mat2.dim() == 2),
              "zentorch_addmm_1dbias_add_add: unsupported dims for mat1, mat2, "
              "add1_input and add2_input");
  TORCH_CHECK(
      (add1_input.sizes() ==
           c10::IntArrayRef(get_matmul_and_linear_output_sizes(mat1, mat2)) &&
       add2_input.sizes() ==
           c10::IntArrayRef(get_matmul_and_linear_output_sizes(mat1, mat2))),
      "zentorch_addmm_1dbias_add_add: unsupported sizes for mat1, mat2, "
      "add1_input and add2_input");
  TORCH_CHECK((self.dim() == 1),
              "zentorch_addmm:  unsupported dims for self, mat1 and mat2");
  at::Tensor result = at::empty(add1_input.sizes(), add1_input.options());

  std::vector<at::Tensor> post_op_buffers = {add1_input, add2_input};
  std::vector<int64_t> post_op_ids = {POST_OP::ADD, POST_OP::ADD};

  LOG(INFO) << "Calling zentorch_matmul_impl from " << __FUNCTION__ << "!\n";
  return zentorch_matmul_impl(mat1, mat2, self, result, post_op_ids,
                              post_op_buffers, beta.to<float>(),
                              alpha.to<float>(), zentorch_op_name);
}

template <POST_OP fuse>
at::Tensor zentorch_addmm(const at::Tensor &self, const at::Tensor &mat1,
                          const at::Tensor &mat2, const at::Scalar &beta,
                          const at::Scalar &alpha,
                          std::string zentorch_op_name) {

  // if alpha is zero, return beta * self directly from here itself.
  // Dont enter the matmul impl.

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  // Here the if condition checks if the matrices are compatible for matrix
  // multiplication and bias addition for any general n-d case. But the
  // TORCH_CHECK conditions specifically checks for the dimensionality
  // conditions which are supported by either zentorch_addmm or
  // zentorch_addmm_1dbias
  if (self.sizes() ==
      c10::IntArrayRef(get_matmul_and_linear_output_sizes(mat1, mat2))) {
    TORCH_CHECK(
        (self.dim() == 2 && mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
        "zentorch_addmm: unsupported dims for self, mat1 and mat2");

    const at::Tensor empty_bias; // dummy empty bias
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

    std::vector<at::Tensor> post_op_buffers = {};
    std::vector<int64_t> post_op_ids = {POST_OP::ADD, fuse};
    at::Tensor result = at::empty(
        get_matmul_and_linear_output_sizes(mat1, mat2), self.options());
    if (beta_float != 1.0f) {
      post_op_buffers.push_back(self.mul(beta_float));
    } else {
      post_op_buffers.push_back(self);
    }

    LOG(INFO) << "Calling zentorch_matmul_impl from " << __FUNCTION__ << "!\n";
    return zentorch_matmul_impl(mat1, mat2, empty_bias, result, post_op_ids,
                                post_op_buffers, beta.to<float>(),
                                alpha.to<float>(), zentorch_op_name);
  } else {
    TORCH_CHECK(
        (self.dim() == 1 && mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
        "zentorch_addmm: unsupported dims for self, mat1 and mat2");

    LOG(INFO) << "Calling zentorch_addmm_1dbias from " << __FUNCTION__ << "!\n";
    return zentorch_addmm_1dbias<fuse>(self, mat1, mat2, beta, alpha,
                                       zentorch_op_name);
  }
}

at::Tensor zentorch_baddbmm(const at::Tensor &self, const at::Tensor &batch1,
                            const at::Tensor &batch2, const at::Scalar &beta,
                            const at::Scalar &alpha,
                            std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  if (self.numel() == 0) {
    TORCH_CHECK(false, "zentorch_baddbmm: incorrect self tensor");
  }
  TORCH_CHECK(
      (self.dim() == 3 && batch1.dim() == 3 &&
       batch2.dim() == 3), // aten::baddbmm
      "zentorch_baddbmm:  unsupported dims for self, batch1 and batch2");

  // Array access is faster than .size(n)
  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();
  const auto self_sizes = self.sizes();

  TORCH_CHECK(
      self_sizes ==
          c10::IntArrayRef(get_matmul_and_linear_output_sizes(batch1, batch2)),
      "input shape is incompatible with matrix multiplication (",
      batch1_sizes[0], "x", batch1_sizes[1], "x", batch1_sizes[2], " @ ",
      batch2_sizes[0], "x", batch2_sizes[1], "x", batch2_sizes[2],
      " != ", self_sizes[0], "x", self_sizes[1], "x", self_sizes[2], ")");
  const at::Tensor empty_bias; // dummy empty bias
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

  std::vector<at::Tensor> post_op_buffers = {};
  std::vector<int64_t> post_op_ids = {POST_OP::ADD, POST_OP::NONE};
  at::Tensor result = at::empty(
      get_matmul_and_linear_output_sizes(batch1, batch2), self.options());

  if (beta_float != 1.0f) {
    post_op_buffers.push_back(self.mul(beta_float));
  } else {
    post_op_buffers.push_back(self);
  }

  LOG(INFO) << "Calling zentorch_matmul_impl from " << __FUNCTION__ << "!\n";
  return zentorch_matmul_impl(batch1, batch2, empty_bias, result, post_op_ids,
                              post_op_buffers, beta.to<float>(),
                              alpha.to<float>(), zentorch_op_name);
}

// zentorch_mm function does not broadcast
template <POST_OP fuse>
at::Tensor zentorch_mm(const at::Tensor &self, const at::Tensor &mat2,
                       std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  TORCH_CHECK((self.dim() == 2 && mat2.dim() == 2), // aten::mm
              "zentorch_mm:  unsupported dims for self and mat2");

  at::Tensor empty_bias;
  at::Tensor out =
      at::empty(get_matmul_and_linear_output_sizes(self, mat2), self.options());
  std::vector<at::Tensor> post_op_buffers = {};
  std::vector<int64_t> post_op_ids = {fuse};
  const float beta = 0.0f;
  const float alpha = 1.0f;

  LOG(INFO) << "Calling zentorch_matmul_impl from " << __FUNCTION__ << "!\n";
  return zentorch_matmul_impl(self, mat2, empty_bias, out, post_op_ids,
                              post_op_buffers, beta, alpha, zentorch_op_name);
}

// zentorch_bmm function does not broadcast
at::Tensor zentorch_bmm(const at::Tensor &self, const at::Tensor &mat2,
                        std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  TORCH_CHECK((self.dim() == 3 && mat2.dim() == 3), // aten::bmm
              "zentorch_bmm:  unsupported dims for self and mat2");

  at::Tensor empty_bias;
  at::Tensor out =
      at::empty(get_matmul_and_linear_output_sizes(self, mat2), self.options());
  std::vector<at::Tensor> post_op_buffers = {};
  std::vector<int64_t> post_op_ids = {};
  const float beta = 0.0f;
  const float alpha = 1.0f;

  LOG(INFO) << "Calling zentorch_matmul_impl from " << __FUNCTION__ << "!\n";
  return zentorch_matmul_impl(self, mat2, empty_bias, out, post_op_ids,
                              post_op_buffers, beta, alpha, zentorch_op_name);
}

at::Tensor
zentorch_addmm_silu_mul(const at::Tensor &bias, const at::Tensor &mat1,
                        const at::Tensor &mat2, const at::Tensor &mat3,
                        const at::Scalar &beta, const at::Scalar &alpha,
                        std::string zentorch_op_name) {
  at::Tensor matmul_impl_bias; // dummy empty bias, it will be assigned to bias
                               // for 1D case
  at::Tensor out;

  std::vector<at::Tensor> post_op_buffers;
  std::vector<int64_t> post_op_ids;
  float beta_float = beta.to<float>();

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  // Here the if condition checks if the matrices are compatible for matrix
  // multiplication and bias addition for any general n-d case. But the
  // TORCH_CHECK conditions specifically checks for the dimensionality
  // conditions which are supported by either zentorch_addmm or
  // zentorch_addmm_1dbias

  // Size checks for post op buffers
  TORCH_CHECK(
      (mat3.dim() == 2 && mat1.dim() == 2 && mat2.dim() == 2),
      "zentorch_addmm_silu_mul: unsupported dims for mat1, mat2 and mat3");
  TORCH_CHECK(
      (mat3.sizes() ==
       c10::IntArrayRef(get_matmul_and_linear_output_sizes(mat1, mat2))),
      "zentorch_addmm_silu_mul: unsupported sizes for mat1, mat2 and mat3");

  if (bias.sizes() ==
      c10::IntArrayRef(get_matmul_and_linear_output_sizes(mat1, mat2))) {
    TORCH_CHECK(
        (bias.dim() == 2 && mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
        "zentorch_addmm:  unsupported dims for bias, mat1 and mat2");

    // Sending the self tensor (this represents the bias in the nn.Module
    // level) as a post op. Since we were passing self directly to matmul impl,
    // this can cause a problem when we are using
    // torch.ops.zentorch.zentorch_addmm directly at the python side with same
    // bias matrix but different inputs. The bias gets corrupted after the
    // first addmm and the subsequent addmms use the corrupted bias tensor,
    // which ultimately results in wrong outputs.

    if (beta_float != 1.0f) {
      post_op_buffers.push_back(bias.mul(beta_float));
    } else {
      post_op_buffers.push_back(bias);
    }

    post_op_ids.push_back(POST_OP::ADD);
  } else {
    TORCH_CHECK(
        (bias.dim() == 1 && mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
        "zentorch_addmm_1dbias: unsupported dims for bias, mat1 and mat2");

    // Array access is faster than .size(n)
    const auto mat1_sizes = mat1.sizes();
    const auto mat2_sizes = mat2.sizes();
    const auto bias_sizes = bias.sizes();

    TORCH_CHECK(bias_sizes[0] == mat2_sizes[1] &&
                    mat1_sizes[1] == mat2_sizes[0],
                "input shape is incompatible with matrix multiplication (",
                mat1_sizes[0], "x", mat1_sizes[1], " @ ", mat2_sizes[0], "x",
                mat2_sizes[1], " != ", mat1_sizes[0], "x", bias_sizes[0], ")");

    if (beta_float != 1.0f) {
      matmul_impl_bias = bias.mul(beta_float);
    } else {
      matmul_impl_bias = bias;
    }
  }

  out =
      at::empty(get_matmul_and_linear_output_sizes(mat1, mat2), mat3.options());

  post_op_buffers.push_back(mat3);
  post_op_ids.push_back(POST_OP::SILU);
  post_op_ids.push_back(POST_OP::MUL);

  LOG(INFO) << "Calling zentorch_matmul_impl from " << __FUNCTION__ << "!\n";
  return zentorch_matmul_impl(mat1, mat2, matmul_impl_bias, out, post_op_ids,
                              post_op_buffers, beta.to<float>(),
                              alpha.to<float>(), zentorch_op_name);
}

at::Tensor zentorch_mm_silu_mul(const at::Tensor &mat1, const at::Tensor &mat2,
                                const at::Tensor &mat3,
                                std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  const float beta = 0.0f;
  const float alpha = 1.0f;

  // Size checks for post op buffers
  TORCH_CHECK((mat3.dim() == 2 && mat1.dim() == 2 && mat2.dim() == 2),
              "zentorch_mm_silu_mul: unsupported dims for mat1, mat2 and mat3");
  TORCH_CHECK(
      (mat3.sizes() ==
       c10::IntArrayRef(get_matmul_and_linear_output_sizes(mat1, mat2))),
      "zentorch_mm_silu_mul: unsupported sizes for mat1, mat2 and mat3");

  at::Tensor empty_bias;
  at::Tensor out =
      at::empty(get_matmul_and_linear_output_sizes(mat1, mat2), mat3.options());
  std::vector<at::Tensor> post_op_buffers = {mat3};
  std::vector<int64_t> post_op_ids = {POST_OP::SILU, POST_OP::MUL};

  LOG(INFO) << "Calling zentorch_matmul_impl from " << __FUNCTION__ << "!\n";
  return zentorch_matmul_impl(mat1, mat2, empty_bias, out, post_op_ids,
                              post_op_buffers, beta, alpha, zentorch_op_name);
}

at::Tensor zentorch_vertical_mlp_group(const at::TensorList &self,
                                       const at::Tensor &input,
                                       const at::TensorList &weights,
                                       const at::ArrayRef<double> &betas,
                                       const at::ArrayRef<double> &alphas,
                                       const at::IntArrayRef &fuse,
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

    dummy_output =
        at::empty(get_matmul_and_linear_output_sizes(mlp_input, weights[i]),
                  input.options());
    input_vector[i] = mlp_input;
    self_vector[i] = self[i];
  }

  output_vector =
      zentorch_matmul_group_impl(self_vector, input_vector, weights, betas,
                                 alphas, fuse, false, zentorch_op_name);

  return output_vector[num_ops - 1];
}

std::vector<at::Tensor> zentorch_attn_qkv_fusion(
    const at::TensorList &self, const at::TensorList &inputs,
    const at::TensorList &weights, const at::ArrayRef<double> &betas,
    const at::ArrayRef<double> &alphas, const at::IntArrayRef &fuse,
    const at::IntArrayRef &is_zentorch_mm, std::string zentorch_op_name) {
  // self = alpha * inputs * weights.t + beta * self
  LOG(INFO) << "In zentorch_attention_horizontal_matmul_group_mlp...\n";
  int num_ops = inputs.size();
  std::vector<at::Tensor> input_vector(num_ops);
  std::vector<at::Tensor> self_vector(num_ops);

  LOG(INFO) << "Executing function: " << __FUNCTION__;
  for (int i = 0; i < num_ops; i++) {
    input_vector[i] = inputs[i];
    if (is_zentorch_mm[i] == 1)
      self_vector[i] =
          at::empty(get_matmul_and_linear_output_sizes(inputs[i], weights[i]),
                    inputs[i].options());
    else
      self_vector[i] = self[i];
  }
  return zentorch_matmul_group_impl(self_vector, input_vector, weights, betas,
                                    alphas, fuse, true, zentorch_op_name);
}

// Template instantiations.
// No post-op.
template at::Tensor zentorch_mm<POST_OP::NONE>(const at::Tensor &self,
                                               const at::Tensor &mat2,
                                               std::string zentorch_op_name);
// ReLU.
template at::Tensor zentorch_mm<POST_OP::RELU>(const at::Tensor &self,
                                               const at::Tensor &mat2,
                                               std::string zentorch_op_name);
// GELU Tanh.
template at::Tensor
zentorch_mm<POST_OP::GELU_TANH>(const at::Tensor &self, const at::Tensor &mat2,
                                std::string zentorch_op_name);
// GELU Erf.
template at::Tensor
zentorch_mm<POST_OP::GELU_ERF>(const at::Tensor &self, const at::Tensor &mat2,
                               std::string zentorch_op_name);
// SiLU.
template at::Tensor zentorch_mm<POST_OP::SILU>(const at::Tensor &self,
                                               const at::Tensor &mat2,
                                               std::string zentorch_op_name);
// No post-op.
template at::Tensor zentorch_addmm<POST_OP::NONE>(const at::Tensor &self,
                                                  const at::Tensor &mat1,
                                                  const at::Tensor &mat2,
                                                  const at::Scalar &beta,
                                                  const at::Scalar &alpha,
                                                  std::string zentorch_op_name);
// ReLU.
template at::Tensor zentorch_addmm<POST_OP::RELU>(const at::Tensor &self,
                                                  const at::Tensor &mat1,
                                                  const at::Tensor &mat2,
                                                  const at::Scalar &beta,
                                                  const at::Scalar &alpha,
                                                  std::string zentorch_op_name);
// GELU Tanh.
template at::Tensor zentorch_addmm<POST_OP::GELU_TANH>(
    const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const at::Scalar &beta, const at::Scalar &alpha,
    std::string zentorch_op_name);
// GELU Erf.
template at::Tensor zentorch_addmm<POST_OP::GELU_ERF>(
    const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const at::Scalar &beta, const at::Scalar &alpha,
    std::string zentorch_op_name);
// SiLU.
template at::Tensor zentorch_addmm<POST_OP::SILU>(const at::Tensor &self,
                                                  const at::Tensor &mat1,
                                                  const at::Tensor &mat2,
                                                  const at::Scalar &beta,
                                                  const at::Scalar &alpha,
                                                  std::string zentorch_op_name);
// No post-op.
template at::Tensor zentorch_addmm_1dbias<POST_OP::NONE>(
    const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const at::Scalar &beta, const at::Scalar &alpha,
    std::string zentorch_op_name);
// ReLU.
template at::Tensor zentorch_addmm_1dbias<POST_OP::RELU>(
    const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const at::Scalar &beta, const at::Scalar &alpha,
    std::string zentorch_op_name);
// GELU Tanh.
template at::Tensor zentorch_addmm_1dbias<POST_OP::GELU_TANH>(
    const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const at::Scalar &beta, const at::Scalar &alpha,
    std::string zentorch_op_name);
// GELU Erf.
template at::Tensor zentorch_addmm_1dbias<POST_OP::GELU_ERF>(
    const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const at::Scalar &beta, const at::Scalar &alpha,
    std::string zentorch_op_name);
// SiLU.
template at::Tensor zentorch_addmm_1dbias<POST_OP::SILU>(
    const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const at::Scalar &beta, const at::Scalar &alpha,
    std::string zentorch_op_name);
} // namespace zentorch
