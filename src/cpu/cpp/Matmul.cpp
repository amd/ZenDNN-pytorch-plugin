/******************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "ZenDNNMemory.hpp"

namespace ZenDNNTorch {

using namespace zendnn;

at::Tensor zendnn_matmul_impl(const at::Tensor &mat1, const at::Tensor &mat2,
                              const at::Tensor &bias,
                              at::Tensor &self_or_result, const float &beta,
                              const float &alpha, const int64_t &fuse) {

  LOG(INFO) << "Executing function: " << __FUNCTION__;
  LOG(INFO) << "mat1 dimensions: " << mat1.sizes();
  LOG(INFO) << "mat2 dimensions: " << mat2.sizes();
  LOG(INFO) << "self_or_result dimensions: " << self_or_result.sizes();
  LOG(INFO) << "beta : " << beta << " and alpha : " << alpha;

  TORCH_CHECK(
      (mat1.dim() == 2 && mat2.dim() == 2) ||     // aten::mm, aten::addmm
          (mat1.dim() == 3 && mat2.dim() == 3) || // aten::bmm, aten::baddbmm
          (mat1.dim() == 2 && mat2.dim() == 1) || // aten::mv
          (mat1.dim() == 1 && mat2.dim() == 1),   // aten::dot
      "zendnn_matmul:  unsupported dims for mat1 and mat2");

  if (mat1.scalar_type() == c10::ScalarType::BFloat16 ||
      mat2.scalar_type() == c10::ScalarType::BFloat16) {
    TORCH_CHECK(utils::zendnn_bf16_device_check(),
                "zendnn_matmul: zendnn_matmul bf16 path needs the cpu support "
                "avx512bf16");
  }

  TORCH_CHECK((mat1.scalar_type() == c10::ScalarType::Float &&
               mat2.scalar_type() == c10::ScalarType::Float &&
               self_or_result.scalar_type() == c10::ScalarType::Float) ||
                  (mat1.scalar_type() == c10::ScalarType::BFloat16 &&
                   mat2.scalar_type() == c10::ScalarType::BFloat16 &&
                   self_or_result.scalar_type() == c10::ScalarType::BFloat16),
              "zendnn_matmul: zendnn_matmul only supports Float and BFloat16");

  const at::Tensor &mat1_unsqueezed =
      mat1.dim() == 1 ? mat1.unsqueeze(0) : mat1;
  const at::Tensor &mat2_unsqueezed =
      mat2.dim() == 1 ? mat2.unsqueeze(1) : mat2;
  at::Tensor &self_or_result_unsqueezed =
      self_or_result.dim() == 1 ? self_or_result.unsqueeze_(1) : self_or_result;

  auto is_zendnn_optimized_format = [&](const at::Tensor &t) {
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
  };

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

    TORCH_CHECK(
        (bias.scalar_type() == c10::ScalarType::Float) ||
            (bias.scalar_type() == c10::ScalarType::BFloat16),
        "zendnn_matmul: zendnn_matmul only supports Float and BFloat16");

    LOG(INFO) << "bias is defined and bias dimensions: " << bias.sizes();

    beta_bias = (beta == 1.0f) ? bias : bias.mul(beta);

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
                  "tensors when bias is defined and alpha != 1");
    }
    LOG(INFO) << "Setting output scales with alpha = " << alpha;
    op_attr.set_output_scales(0, std::vector<float>(1, alpha));
  }

  // set the post-ops or fusion-ops;
  // by default, fuse = 0,
  // fuse = 1 for relu op,
  // fuse = 2 for gelu approximate (tanh)
  // fuse = 3 for gelu exact (erf)
  if (fuse == 1) {
    LOG(INFO) << "Setting relu as post op";
    po.append_eltwise(1.0f, algorithm::eltwise_relu, 0.f, 0.f);
  }
  if (fuse == 2) {
    LOG(INFO) << "Setting gelu_tanh as post op";
    po.append_eltwise(1.0f, algorithm::eltwise_gelu_tanh, 1.f, 0.f);
  }
  if (fuse == 3) {
    LOG(INFO) << "Setting gelu_erf as post op";
    po.append_eltwise(1.0f, algorithm::eltwise_gelu_erf, 1.f, 0.f);
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

std::vector<int64_t> get_matmul_output_sizes(const at::Tensor &tensor1,
                                             const at::Tensor &tensor2) {
  const int64_t dim = tensor1.dim();
  std::vector<int64_t> output_size(dim);
  for (auto i = 0; i < dim - 1; i++) {
    output_size[i] = tensor1.size(i);
  }
  output_size[dim - 1] = tensor2.size(dim - 1);
  return output_size;
}

// for 1d bias
at::Tensor zendnn_addmm_1dbias(const at::Tensor &self, const at::Tensor &mat1,
                               const at::Tensor &mat2, const at::Scalar &beta,
                               const at::Scalar &alpha, const int64_t &fuse) {
  LOG(INFO) << "Executing function: " << __FUNCTION__;
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

  return zendnn_matmul_impl(mat1, mat2, self, result, beta.to<float>(),
                            alpha.to<float>(), fuse);
}

at::Tensor zendnn_addmm(const at::Tensor &self, const at::Tensor &mat1,
                        const at::Tensor &mat2, const at::Scalar &beta,
                        const at::Scalar &alpha, const int64_t &fuse) {

  LOG(INFO) << "Executing function: " << __FUNCTION__;

  if (self.sizes() == c10::IntArrayRef(get_matmul_output_sizes(mat1, mat2))) {
    TORCH_CHECK(
        (self.dim() == 2 && mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
        "zendnn_addmm:  unsupported dims for self, mat1 and mat2");

    const at::Tensor empty_bias; // dummy empty bias
    return zendnn_matmul_impl(mat1, mat2, empty_bias,
                              const_cast<at::Tensor &>(self), beta.to<float>(),
                              alpha.to<float>(), fuse);
  } else {
    TORCH_CHECK(
        (self.dim() == 1 && mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
        "zendnn_addmm: unsupported dims for self, mat1 and mat2");

    return zendnn_addmm_1dbias(self, mat1, mat2, beta, alpha, fuse);
  }
}

at::Tensor zendnn_baddbmm(const at::Tensor &self, const at::Tensor &batch1,
                          const at::Tensor &batch2, const at::Scalar &beta,
                          const at::Scalar &alpha) {
  LOG(INFO) << "Executing function: " << __FUNCTION__;

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
  return zendnn_matmul_impl(batch1, batch2, empty_bias,
                            const_cast<at::Tensor &>(self), beta.to<float>(),
                            alpha.to<float>(), fuse);
}

// zendnn_mm function does not broadcast
at::Tensor zendnn_mm(const at::Tensor &self, const at::Tensor &mat2,
                     const int64_t &fuse) {
  LOG(INFO) << "Executing function: " << __FUNCTION__;
  TORCH_CHECK((self.dim() == 2 && mat2.dim() == 2), // aten::mm
              "zendnn_mm:  unsupported dims for self and mat2");

  at::Tensor out =
      at::empty(get_matmul_output_sizes(self, mat2), self.options());
  const float beta = 0.0f;
  const float alpha = 1.0f;
  return zendnn_addmm(out, self, mat2, beta, alpha, fuse);
}

// zendnn_bmm function does not broadcast
at::Tensor zendnn_bmm(const at::Tensor &self, const at::Tensor &mat2) {
  LOG(INFO) << "Executing function: " << __FUNCTION__;
  TORCH_CHECK((self.dim() == 3 && mat2.dim() == 3), // aten::bmm
              "zendnn_bmm:  unsupported dims for self and mat2");

  at::Tensor out =
      at::empty(get_matmul_output_sizes(self, mat2), self.options());
  const float beta = 0.0f;
  const float alpha = 1.0f;
  return zendnn_baddbmm(out, self, mat2, beta, alpha);
}

} // namespace ZenDNNTorch
