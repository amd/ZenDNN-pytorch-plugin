/******************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "ZenDNNMemory.hpp"
#include "ZenTorchUtils.hpp"

namespace ZenDNNTorch {

using namespace zendnn;

at::Tensor zendnn_matmul_impl(const at::Tensor &mat1, const at::Tensor &mat2,
                              const at::Tensor &bias,
                              at::Tensor &self_or_result, const float &beta,
                              const float &alpha, const int64_t &fuse) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  LOG(INFO) << "mat1 dimensions: " << mat1.sizes();
  LOG(INFO) << "mat2 dimensions: " << mat2.sizes();
  LOG(INFO) << "self_or_result dimensions: " << self_or_result.sizes();
  LOG(INFO) << "beta : " << beta << " and alpha : " << alpha;

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

  // ZenDNN does not support 1-D tensors. So, whenever the tensots are of
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
  memory z_mat1 = zen_memory(mat1_);
  memory z_mat2 = zen_memory(mat2_);
  memory z_result = zen_memory(self_or_result_unsqueezed);

  // "addmm", "baddbmm" in pytorch allow bias to be 2-D or 3-D tensor
  // but zendnn matmul primitive only support bias be 1-D tensors
  // to address their differences, we use zendnn post ops to perform a fused
  // "add" after matrix multiplication is over
  const bool bias_defined = bias.numel();
  at::Tensor beta_bias;
  memory z_bias;
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

  zendnn::primitive_attr op_attr;
  post_ops po;
  if (beta != 0.0f && !bias_defined) {
    // sets post_ops as add or sum
    LOG(INFO) << "Setting add or sum as post op";
    po.append_sum(beta);
  }
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
      TORCH_CHECK(!(mat1.scalar_type() == c10::ScalarType::BFloat16 ||
                    mat2.scalar_type() == c10::ScalarType::BFloat16),
                  "zendnn_matmul: zendnn_matmul is not supported for bf16 "
                  "tensors when bias is defined and alpha is not equal to 1");
    }
    LOG(INFO) << "Setting output scales with alpha = " << alpha;
    op_attr.set_output_scales(0, std::vector<float>(1, alpha));
  }

  // set the post-ops or fusion-ops;
  // by default, fuse = 0,
  // fuse = 1 for relu op,
  // fuse = 2 for gelu approximate (tanh)
  // fuse = 3 for gelu exact (erf)
  switch (fuse) {
  case 1:
    LOG(INFO) << "Setting relu as post op";
    po.append_eltwise(1.0f, algorithm::eltwise_relu, 0.f, 0.f);
    break;
  case 2:
    LOG(INFO) << "Setting gelu_tanh as post op";
    po.append_eltwise(1.0f, algorithm::eltwise_gelu_tanh, 1.f, 0.f);
    break;
  case 3:
    LOG(INFO) << "Setting gelu_erf as post op";
    po.append_eltwise(1.0f, algorithm::eltwise_gelu_erf, 1.f, 0.f);
    break;
  default:
    break;
  }
  op_attr.set_post_ops(po);

  matmul::desc pdesc =
      bias_defined ? matmul::desc(z_mat1.get_desc(), z_mat2.get_desc(),
                                  z_bias.get_desc(), z_result.get_desc())
                   : matmul::desc(z_mat1.get_desc(), z_mat2.get_desc(),
                                  z_result.get_desc());

  matmul::primitive_desc pd =
      matmul::primitive_desc(pdesc, op_attr, utils::engine::cpu_engine());

  std::unordered_map<int, memory> execute_args =
      bias_defined
          ? std::unordered_map<int, memory>({{ZENDNN_ARG_SRC, z_mat1},
                                             {ZENDNN_ARG_WEIGHTS, z_mat2},
                                             {ZENDNN_ARG_BIAS, z_bias},
                                             {ZENDNN_ARG_DST, z_result}})
          : std::unordered_map<int, memory>({{ZENDNN_ARG_SRC, z_mat1},
                                             {ZENDNN_ARG_WEIGHTS, z_mat2},
                                             {ZENDNN_ARG_DST, z_result}});

  LOG(INFO) << "MatMul compute in progress...";
  matmul(pd).execute(utils::stream::default_stream(), execute_args);

  if ((mat1.dim() == 1 || mat1.dim() == 2) && mat2.dim() == 1) {
    // aten::mv  >>  [m, 1] tensor will be squeezed to 1-d([m]) tensor
    // aten::dot >>  [1, 1] tensor will be squeezed to 0-d([]) tensor
    self_or_result_unsqueezed.squeeze_();
  }

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
  return std::move(self_or_result_unsqueezed);
}

// for 1d bias
at::Tensor zendnn_addmm_1dbias(const at::Tensor &self, const at::Tensor &mat1,
                               const at::Tensor &mat2, const at::Scalar &beta,
                               const at::Scalar &alpha, const int64_t &fuse) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  TORCH_CHECK(
      (self.dim() == 1 && mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
      "zendnn_addmm_1dbias: unsupported dims for self, mat1 and mat2");

  // Array access is faster than .size(n)
  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();
  const auto self_sizes = self.sizes();

  TORCH_CHECK(self_sizes[0] == mat2_sizes[1] && mat1_sizes[1] == mat2_sizes[0],
              "input shape is incompatible with matrix multiplication (",
              mat1_sizes[0], "x", mat1_sizes[1], " @ ", mat2_sizes[0], "x",
              mat2_sizes[1], " != ", mat1_sizes[0], "x", self_sizes[0], ")");

  at::Tensor result =
      at::empty(get_matmul_output_sizes(mat1, mat2), mat1.options());

  LOG(INFO) << "Entering zendnn_matmul_impl from " << __FUNCTION__ << "!\n";

  return zendnn_matmul_impl(mat1, mat2, self, result, beta.to<float>(),
                            alpha.to<float>(), fuse);
}

at::Tensor zendnn_addmm(const at::Tensor &self, const at::Tensor &mat1,
                        const at::Tensor &mat2, const at::Scalar &beta,
                        const at::Scalar &alpha, const int64_t &fuse) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  // Here the if condition checks if the matrices are compatible for matrix
  // multiplication and bias addition for any general n-d case. But the
  // TORCH_CHECK conditions specifically checks for the dimensionality
  // conditions which are supported by either zendnn_addmm or
  // zendnn_addmm_1dbias

  if (self.sizes() == c10::IntArrayRef(get_matmul_output_sizes(mat1, mat2))) {
    TORCH_CHECK(
        (self.dim() == 2 && mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
        "zendnn_addmm:  unsupported dims for self, mat1 and mat2");

    const at::Tensor empty_bias; // dummy empty bias

    LOG(INFO) << "Entering zendnn_matmul_impl from " << __FUNCTION__ << "!\n";

    return zendnn_matmul_impl(mat1, mat2, empty_bias,
                              const_cast<at::Tensor &>(self), beta.to<float>(),
                              alpha.to<float>(), fuse);
  } else {
    TORCH_CHECK(
        (self.dim() == 1 && mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
        "zendnn_addmm: unsupported dims for self, mat1 and mat2");

    LOG(INFO) << "Entering zendnn_addmm_1dbias from " << __FUNCTION__ << "!\n";

    return zendnn_addmm_1dbias(self, mat1, mat2, beta, alpha, fuse);
  }
}

at::Tensor zendnn_baddbmm(const at::Tensor &self, const at::Tensor &batch1,
                          const at::Tensor &batch2, const at::Scalar &beta,
                          const at::Scalar &alpha) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  if (self.numel() == 0) {
    TORCH_CHECK(false, "zendnn_baddbmm: incorrect self tensor");
  }
  TORCH_CHECK((self.dim() == 3 && batch1.dim() == 3 &&
               batch2.dim() == 3), // aten::baddbmm
              "zendnn_baddbmm:  unsupported dims for self, batch1 and batch2");

  // Array access is faster than .size(n)
  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();
  const auto self_sizes = self.sizes();

  TORCH_CHECK(
      self_sizes == c10::IntArrayRef(get_matmul_output_sizes(batch1, batch2)),
      "input shape is incompatible with matrix multiplication (",
      batch1_sizes[0], "x", batch1_sizes[1], "x", batch1_sizes[2], " @ ",
      batch2_sizes[0], "x", batch2_sizes[1], "x", batch2_sizes[2],
      " != ", self_sizes[0], "x", self_sizes[1], "x", self_sizes[2], ")");
  const int64_t fuse = 0;
  const at::Tensor empty_bias; // dummy empty bias

  LOG(INFO) << "Entering zendnn_matmul_impl from " << __FUNCTION__ << "!\n";

  return zendnn_matmul_impl(batch1, batch2, empty_bias,
                            const_cast<at::Tensor &>(self), beta.to<float>(),
                            alpha.to<float>(), fuse);
}

// zendnn_mm function does not broadcast
at::Tensor zendnn_mm(const at::Tensor &self, const at::Tensor &mat2,
                     const int64_t &fuse) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  TORCH_CHECK((self.dim() == 2 && mat2.dim() == 2), // aten::mm
              "zendnn_mm:  unsupported dims for self and mat2");

  at::Tensor out =
      at::empty(get_matmul_output_sizes(self, mat2), self.options());
  const float beta = 0.0f;
  const float alpha = 1.0f;

  LOG(INFO) << "Entering zendnn_addmm from " << __FUNCTION__ << "!\n";

  return zendnn_addmm(out, self, mat2, beta, alpha, fuse);
}

// zendnn_bmm function does not broadcast
at::Tensor zendnn_bmm(const at::Tensor &self, const at::Tensor &mat2) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  TORCH_CHECK((self.dim() == 3 && mat2.dim() == 3), // aten::bmm
              "zendnn_bmm:  unsupported dims for self and mat2");

  at::Tensor out =
      at::empty(get_matmul_output_sizes(self, mat2), self.options());
  const float beta = 0.0f;
  const float alpha = 1.0f;

  LOG(INFO) << "Entering zendnn_baddbmm from " << __FUNCTION__ << "!\n";

  return zendnn_baddbmm(out, self, mat2, beta, alpha);
}

at::Tensor zendnn_vertical_mlp_group(const at::TensorList &self,
                                     const at::Tensor &input,
                                     const at::TensorList &weights,
                                     const at::ArrayRef<double> &betas,
                                     const at::ArrayRef<double> &alphas,
                                     const at::IntArrayRef &fuse) {

  // self = alpha * input * weights + beta * self

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  int num_ops = weights.size();
  std::vector<at::Tensor> bias_vector(num_ops);
  std::vector<at::Tensor> self_or_result_vector(num_ops);
  std::vector<bool> bias_defined_vector(num_ops);
  std::vector<int64_t> fuse_vector(num_ops);
  std::vector<memory> z_mat2_vector(num_ops);
  std::vector<memory> z_bias_vector(num_ops);
  std::vector<memory> z_result_vector(num_ops);
  std::vector<float> alphas_vector(num_ops);
  std::vector<float> betas_vector(num_ops);
  memory z_input;

  at::Tensor mlp_input, dummy_output;

  for (int i = 0; i < num_ops; i++) {
    if (i == 0)
      mlp_input = input;
    else
      mlp_input = dummy_output;

    dummy_output = at::empty(get_matmul_output_sizes(mlp_input, weights[i]),
                             input.options());

    alphas_vector[i] = static_cast<float>(alphas[i]);
    betas_vector[i] = static_cast<float>(betas[i]);

    // Here the if condition checks if the matrices are compatible for matrix
    // multiplication and bias addition for any general n-d case. But the
    // TORCH_CHECK conditions specifically checks for the dimensionality
    // conditions which are supported by either zendnn_addmm or
    // zendnn_addmm_1dbias

    if (self[i].sizes() ==
        c10::IntArrayRef(get_matmul_output_sizes(mlp_input, weights[i]))) {
      TORCH_CHECK((self[i].dim() == 2 && mlp_input.dim() == 2 &&
                   weights[i].dim() == 2), // aten::addmm
                  "zendnn_addmm:  unsupported dims for self, mat1 and mat2");

      const at::Tensor empty_bias; // dummy empty bias

      // Populating the bias_vector with empty bias in case of addmm
      bias_vector[i] = empty_bias;
      // Populating the self_or_result_vector with self in case of addmm since
      // that acts as the bias here
      self_or_result_vector[i] = self[i];
    } else {
      TORCH_CHECK((self[i].dim() == 1 && mlp_input.dim() == 2 &&
                   weights[i].dim() == 2), // aten::addmm
                  "zendnn_addmm: unsupported dims for self, mat1 and mat2");

      // Array access is faster than .size(n)
      const auto mat1_sizes = mlp_input.sizes();
      const auto mat2_sizes = weights[i].sizes();
      const auto self_sizes = self[i].sizes();

      TORCH_CHECK(
          self_sizes[0] == mat2_sizes[1] && mat1_sizes[1] == mat2_sizes[0],
          "input shape is incompatible with matrix multiplication (",
          mat1_sizes[0], "x", mat1_sizes[1], " @ ", mat2_sizes[0], "x",
          mat2_sizes[1], " != ", mat1_sizes[0], "x", self_sizes[0], ")");

      at::Tensor result = at::empty(
          get_matmul_output_sizes(mlp_input, weights[i]), mlp_input.options());

      bias_vector[i] = self[i];
      self_or_result_vector[i] = result;
    }

    check_valid_sizes(mlp_input, weights[i]);

    if (mlp_input.scalar_type() == c10::ScalarType::BFloat16 ||
        weights[i].scalar_type() == c10::ScalarType::BFloat16) {
      TORCH_CHECK(
          utils::zendnn_bf16_device_check(),
          "zendnn_matmul: zendnn_matmul bf16 path needs the cpu support "
          "avx512bf16");
    }

    std::vector<at::Tensor> tensor_vector(3);
    tensor_vector[0] = mlp_input;
    tensor_vector[1] = weights[i];
    tensor_vector[2] = self_or_result_vector[i];

    check_scalar_type(tensor_vector);

    // ZenDNN does not support 1-D tensors. So, whenever the tensots are of
    // 1 dimension, they are unsqueezed on the required dimension to make them
    // into 2-D dimensional tensors.
    const at::Tensor &input_unsqueezed =
        mlp_input.dim() == 1 ? mlp_input.unsqueeze(0) : mlp_input;
    const at::Tensor &weight_unsqueezed =
        weights[i].dim() == 1 ? weights[i].unsqueeze(1) : weights[i];
    at::Tensor &self_or_result_unsqueezed =
        self_or_result_vector[i].dim() == 1
            ? self_or_result_vector[i].unsqueeze_(1)
            : self_or_result_vector[i];

    // zendnn is only optimized for contiguous or transposed
    // (transpose last 2 dim if 3-D tensor) format now
    // Will remove this "contiguous" after zendnn have fully supported
    at::Tensor mat1_ = is_zendnn_optimized_format(input_unsqueezed)
                           ? input_unsqueezed
                           : input_unsqueezed.contiguous();
    at::Tensor mat2_ = is_zendnn_optimized_format(weight_unsqueezed)
                           ? weight_unsqueezed
                           : weight_unsqueezed.contiguous();

    // convert the aten tensors to zendnn memory
    if (i == 0) {
      z_input = zen_memory(mat1_);
    }

    // Populating the z_mat2_vector with the zendnn memory of mat2_
    z_mat2_vector[i] = zen_memory(mat2_);
    // Populating the z_result_vector with the zendnn memory of
    // self_or_result_unsqueezed
    z_result_vector[i] = zen_memory(self_or_result_unsqueezed);

    // "addmm", "baddbmm" in pytorch allow bias to be 2-D or 3-D tensor
    // but zendnn matmul primitive only support bias be 1-D tensors
    // to address their differences, we use zendnn post ops to perform a fused
    // "add" after matrix multiplication is over

    // Populating the bias_defined_vector with the bool equivalent values based
    // on the number of elements in the bias.
    bias_defined_vector[i] = bias_vector[i].numel();

    at::Tensor beta_bias;
    if (bias_defined_vector[i] && bias_vector[i].dim() == 1 &&
        (mlp_input.dim() == 2 && weights[i].dim() == 2)) {
      if (bias_vector[i].scalar_type() == c10::ScalarType::BFloat16) {
        TORCH_CHECK(
            utils::zendnn_bf16_device_check(),
            "zendnn_matmul: zendnn_matmul bf16 path needs the cpu support "
            "avx512bf16");
      }

      std::vector<at::Tensor> tensor_vector(1);
      tensor_vector[0] = bias_vector[i];

      check_scalar_type(tensor_vector);

      LOG(INFO) << "bias is defined and bias dimensions: "
                << bias_vector[i].sizes();

      // BR_GEMM kernel execution is as alpha * (mat1 @ mat2 + bias)
      // but addmm is executed as alpha * (mat1 @ mat2) + beta * bias

      // so alpha * (mat1 @ mat2 + (beta / alpha) * bias) is equivalent
      // to alpha * (mat1 @ mat2) + beta * bias
      const float modified_beta =
          (alphas_vector[i] == 1.0f || alphas_vector[i] == 0)
              ? betas_vector[i]
              : betas_vector[i] / alphas_vector[i];
      beta_bias = (modified_beta == 1.0f) ? bias_vector[i]
                                          : bias_vector[i].mul(modified_beta);

      // creating bias zen_memory with predefined memory::desc
      // as bias is 1d we need to define format_tag as 'ab'
      // to represent bias memory as 2d for bias_desc creation
      const memory::format_tag &bias_tag = memory::format_tag::ab;
      const memory::desc &bias_desc = memory::desc(
          {{1, beta_bias.size(0)}, get_ztype_from_aten(beta_bias), bias_tag});
      z_bias_vector[i] = zen_memory(beta_bias, bias_desc);
    }

    if (alphas[i] != 1.0f) {
      if (bias_defined_vector[i]) {
        // TODO: add support for alpha when bias is defined
        TORCH_CHECK(!(mlp_input.scalar_type() == c10::ScalarType::BFloat16 ||
                      weights[i].scalar_type() == c10::ScalarType::BFloat16),
                    "zendnn_matmul: zendnn_matmul is not supported for bf16 "
                    "tensors when bias is defined and alpha is not equal to 1");
      }
    }

    // Populating the fuse_vector with the post ops.
    fuse_vector[i] = fuse[i];
  }

  std::vector<memory> z_input_vector = {z_input};

  LOG(INFO) << "GroupMatMul compute in progress...";
  zendnn_custom_op::zendnn_grp_mlp(
      z_input_vector, z_mat2_vector, z_bias_vector, alphas_vector, betas_vector,
      bias_defined_vector, fuse_vector, z_result_vector);

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return self_or_result_vector[num_ops - 1];
}

std::vector<at::Tensor> zendnn_matmul_group_impl(
    const at::TensorList &self_vector, const at::TensorList &inputs,
    const at::TensorList &weights, const at::ArrayRef<double> &betas,
    const at::ArrayRef<double> &alphas, const at::IntArrayRef &fuse) {
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
  std::vector<memory> z_input(num_ops);

  for (int i = 0; i < num_ops; i++) {
    alphas_vector[i] = static_cast<float>(alphas[i]);
    betas_vector[i] = static_cast<float>(betas[i]);
    if (self_vector[i].sizes() ==
        c10::IntArrayRef(get_matmul_output_sizes(inputs[i], weights[i]))) {
      TORCH_CHECK((self_vector[i].dim() == 2 && inputs[i].dim() == 2 &&
                   weights[i].dim() == 2), // aten::addmm
                  "zendnn_addmm:  unsupported dims for self, mat1 and mat2");
      const at::Tensor empty_bias; // dummy empty bias
      bias_vector[i] = empty_bias;
      self_or_result_vector[i] = self_vector[i];
    } else {
      TORCH_CHECK((self_vector[i].dim() == 1 && inputs[i].dim() == 2 &&
                   weights[i].dim() == 2), // aten::addmm
                  "zendnn_addmm: unsupported dims for self, mat1 and mat2");
      // Array access is faster than .size(n)
      const auto mat1_sizes = inputs[i].sizes();
      const auto mat2_sizes = weights[i].sizes();
      const auto self_sizes = self_vector[i].sizes();
      TORCH_CHECK(
          self_sizes[0] == mat2_sizes[1] && mat1_sizes[1] == mat2_sizes[0],
          "input shape is incompatible with matrix multiplication (",
          mat1_sizes[0], "x", mat1_sizes[1], " @ ", mat2_sizes[0], "x",
          mat2_sizes[1], " != ", mat1_sizes[0], "x", self_sizes[0], ")");

      at::Tensor result = at::empty(
          get_matmul_output_sizes(inputs[i], weights[i]), inputs[i].options());
      bias_vector[i] = self_vector[i];
      self_or_result_vector[i] = result;
    }
    check_valid_sizes(inputs[i], weights[i]);
    if (inputs[i].scalar_type() == c10::ScalarType::BFloat16 ||
        weights[i].scalar_type() == c10::ScalarType::BFloat16) {
      TORCH_CHECK(
          utils::zendnn_bf16_device_check(),
          "zendnn_matmul: zendnn_matmul bf16 path needs the cpu support "
          "avx512bf16");
    }

    std::vector<at::Tensor> tensor_vector(3);
    tensor_vector[0] = inputs[i];
    tensor_vector[1] = weights[i];
    tensor_vector[2] = self_or_result_vector[i];
    check_scalar_type(tensor_vector);
    const at::Tensor &input_unsqueezed =
        inputs[i].dim() == 1 ? inputs[i].unsqueeze(0) : inputs[i];
    const at::Tensor &weight_unsqueezed =
        weights[i].dim() == 1 ? weights[i].unsqueeze(1) : weights[i];
    at::Tensor &self_or_result_unsqueezed =
        self_or_result_vector[i].dim() == 1
            ? self_or_result_vector[i].unsqueeze_(1)
            : self_or_result_vector[i];
    // zendnn is only optimized for contiguous or transposed
    // (transpose last 2 dim if 3-D tensor) format now
    // Will remove this "contiguous" after zendnn have fully supported
    at::Tensor mat1_ = is_zendnn_optimized_format(input_unsqueezed)
                           ? input_unsqueezed
                           : input_unsqueezed.contiguous();
    at::Tensor mat2_ = is_zendnn_optimized_format(weight_unsqueezed)
                           ? weight_unsqueezed
                           : weight_unsqueezed.contiguous();
    // convert the aten tensors to zendnn memory

    z_input[i] = zen_memory(mat1_);
    z_mat2_vector[i] = zen_memory(mat2_);
    z_result_vector[i] = zen_memory(self_or_result_unsqueezed);
    // "addmm", "baddbmm" in pytorch allow bias to be 2-D or 3-D tensor
    // but zendnn matmul primitive only support bias be 1-D tensors
    // to address their differences, we use zendnn post ops to perform a fused
    // "add" after matrix multiplication is over
    bias_defined_vector[i] = bias_vector[i].numel();
    at::Tensor beta_bias;
    if (bias_defined_vector[i] && bias_vector[i].dim() == 1 &&
        (inputs[i].dim() == 2 && weights[i].dim() == 2)) {
      if (bias_vector[i].scalar_type() == c10::ScalarType::BFloat16) {
        TORCH_CHECK(
            utils::zendnn_bf16_device_check(),
            "zendnn_matmul: zendnn_matmul bf16 path needs the cpu support "
            "avx512bf16");
      }
      std::vector<at::Tensor> tensor_vector(1);
      tensor_vector[0] = bias_vector[i];
      check_scalar_type(tensor_vector);
      LOG(INFO) << "bias is defined and bias dimensions: "
                << bias_vector[i].sizes();
      const float modified_beta =
          (alphas_vector[i] == 1.0f || alphas_vector[i] == 0)
              ? betas_vector[i]
              : betas_vector[i] / alphas_vector[i];
      beta_bias = (modified_beta == 1.0f) ? bias_vector[i]
                                          : bias_vector[i].mul(modified_beta);
      // creating bias zen_memory with predefined memory::desc
      // as bias is 1d we need to define format_tag as 'ab'
      // to represent bias memory as 2d for bias_desc creation
      const memory::format_tag &bias_tag = memory::format_tag::ab;
      const memory::desc &bias_desc = memory::desc(
          {{1, beta_bias.size(0)}, get_ztype_from_aten(beta_bias), bias_tag});
      z_bias_vector[i] = zen_memory(beta_bias, bias_desc);
    }
    if (betas[i] != 0.0f && !bias_defined_vector[i]) {
      // sets post_ops as add or sum
      LOG(INFO) << "Setting add or sum as post op";
    }
    if (alphas[i] != 1.0f) {
      if (bias_defined_vector[i]) {
        // TODO: add support for alpha when bias is defined
        TORCH_CHECK(!(inputs[i].scalar_type() == c10::ScalarType::BFloat16 ||
                      weights[i].scalar_type() == c10::ScalarType::BFloat16),
                    "zendnn_matmul: zendnn_matmul is not supported for bf16 "
                    "tensors when bias is defined and alpha != 1");
      }
      LOG(INFO) << "Setting output scales with alpha = " << alphas[i];
    }
    fuse_vector[i] = fuse[i];
  }
  LOG(INFO) << "Horizontal GroupMatMul compute in progress...";
  zendnn_custom_op::zendnn_grp_mlp(
      z_input, z_mat2_vector, z_bias_vector, alphas_vector, betas_vector,
      bias_defined_vector, fuse_vector, z_result_vector);
  LOG(INFO) << "Horizontal GroupMatMul compute complete...";
  return self_or_result_vector;
}

std::vector<at::Tensor> zendnn_attn_horizontal_mlp_group(
    const at::TensorList &self, const at::TensorList &inputs,
    const at::TensorList &weights, const at::ArrayRef<double> &betas,
    const at::ArrayRef<double> &alphas, const at::IntArrayRef &fuse,
    const at::IntArrayRef &is_zendnnmm) {
  // self = alpha * inputs * weights.t + beta * self
  LOG(INFO) << "In zendnn_attention_horizontal_matmul_group_mlp...\n";
  int num_ops = inputs.size();
  std::vector<at::Tensor> self_vector(num_ops);

  LOG(INFO) << "Executing function: " << __FUNCTION__;
  for (int i = 0; i < num_ops; i++) {

    if (is_zendnnmm[i] == 1)
      self_vector[i] = at::empty(get_matmul_output_sizes(inputs[i], weights[i]),
                                 inputs[i].options());
    else
      self_vector[i] = self[i];
  }
  return zendnn_matmul_group_impl(self_vector, inputs, weights, betas, alphas,
                                  fuse);
}
} // namespace ZenDNNTorch
