/******************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "ZenDNNMemory.hpp"

namespace ZenDNNTorch {

using namespace zendnn;

at::Tensor zendnn_matmul_impl(const at::Tensor &mat1, const at::Tensor &mat2,
                              at::Tensor &self_or_result, const float &beta,
                              const float &alpha, const bool &fuse_relu) {

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

  auto mat1_unsqueezed = mat1.dim() == 1 ? mat1.unsqueeze(0) : mat1;
  auto mat2_unsqueezed = mat2.dim() == 1 ? mat2.unsqueeze(1) : mat2;
  auto self_or_result_unsqueezed =
      self_or_result.dim() == 1 ? self_or_result.unsqueeze(1) : self_or_result;

  auto is_zendnn_optimized_format = [&](const at::Tensor &t) {
    if (t.is_contiguous())
      return true;
    const auto sizes = t.sizes();
    const auto strides = t.strides();
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
  memory z_mat1 = zen_memory_view_from_dense(mat1_);
  memory z_mat2 = zen_memory_view_from_dense(mat2_);
  memory z_result = zen_memory_view_from_dense(self_or_result_unsqueezed);

  zendnn::primitive_attr op_attr;
  // "addmm", "baddbmm" in pytorch allow bias to be 2-D or 3-D tensor
  // but zendnn matmul primitive only support bias be 1-D tensors
  // to address their differences, we use zendnn post ops to perform a fused
  // "add" after matrix multiplication is over
  post_ops po;
  if (beta != 0.0f) {
    // sets post_ops as add or sum
    LOG(INFO) << "Setting add or sum as post op";
    po.append_sum(beta);
  }
  // If alpha = 0, does not need to actually do gemm computation
  if (alpha == 0) {
    TORCH_CHECK(false, "zendnn_matmul: alpha is set to zero");
  } else {
    LOG(INFO) << "Setting output scales with alpha = " << alpha;
    op_attr.set_output_scales(0, std::vector<float>(1, alpha));
  }
  // sets post ops as relu
  if (fuse_relu) {
    LOG(INFO) << "Setting relu as post op";
    po.append_eltwise(1.0f, algorithm::eltwise_relu, 0.f, 0.f);
  }
  op_attr.set_post_ops(po);

  matmul::desc pdesc =
      matmul::desc(z_mat1.get_desc(), z_mat2.get_desc(), z_result.get_desc());

  matmul::primitive_desc pd =
      matmul::primitive_desc(pdesc, op_attr, utils::engine::cpu_engine());

  LOG(INFO) << "MatMul compute in progress...";
  matmul(pd).execute(utils::stream::default_stream(),
                     {{ZENDNN_ARG_SRC, z_mat1},
                      {ZENDNN_ARG_WEIGHTS, z_mat2},
                      {ZENDNN_ARG_DST, z_result}});

  if (mat1.dim() == 1 && mat2.dim() == 1) {
    // aten::dot
    self_or_result.squeeze_();
  }
  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
  return std::move(self_or_result);
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

at::Tensor zendnn_addmm(const at::Tensor &self, const at::Tensor &mat1,
                        const at::Tensor &mat2, const at::Scalar &beta,
                        const at::Scalar &alpha, const bool &fuse_relu) {

  LOG(INFO) << "Executing function: " << __FUNCTION__;
  // Array access is faster than .size(n)
  auto mat1_sizes = mat1.sizes();
  auto mat2_sizes = mat2.sizes();

  TORCH_CHECK((mat1.scalar_type() == c10::ScalarType::Float &&
               mat2.scalar_type() == c10::ScalarType::Float &&
               self.scalar_type() == c10::ScalarType::Float) ||
                  (mat1.scalar_type() == c10::ScalarType::BFloat16 &&
                   mat2.scalar_type() == c10::ScalarType::BFloat16 &&
                   self.scalar_type() == c10::ScalarType::BFloat16),
              "zendnn_addmm: zendnn_addmm only supports Float and BFloat16");
  if (self.sizes() == c10::IntArrayRef(get_matmul_output_sizes(mat1, mat2))) {
    TORCH_CHECK(
        (self.dim() == 2 && mat1.dim() == 2 && mat2.dim() == 2), // aten::addmm
        "zendnn_addmm:  unsupported dims for self, mat1 and mat2");

    const auto self_sizes = self.sizes();

    TORCH_CHECK(self_sizes[0] == mat1_sizes[0] &&
                    self_sizes[1] == mat2_sizes[1],
                "input shape is incompatible with matrix multiplication (",
                mat1_sizes[0], "x", mat1_sizes[1], " @ ", mat2_sizes[0], "x",
                mat2_sizes[1], " != ", self_sizes[0], "x", self_sizes[1], ")");

    return zendnn_matmul_impl(mat1, mat2, const_cast<at::Tensor &>(self),
                              beta.to<float>(), alpha.to<float>(), fuse_relu);
  } else {
    LOG(WARNING) << "self tensor is not 2-dimensional as self dimensions: "
                 << self.sizes();
    auto self_ =
        expand_size(self, {mat1.sizes()[0], mat2.sizes()[1]}, "zendnn_addmm");
    const at::Tensor &self_t = *self_;

    const auto self_sizes = self_t.sizes();

    TORCH_CHECK((self_t.dim() == 2 && mat1.dim() == 2 &&
                 mat2.dim() == 2), // aten::addmm
                "zendnn_addmm: unsupported dims for mat1 and mat2");

    TORCH_CHECK(self_sizes[0] == mat1_sizes[0] &&
                    self_sizes[1] == mat2_sizes[1],
                "input shape is incompatible with matrix multiplication (",
                mat1_sizes[0], "x", mat1_sizes[1], " @ ", mat2_sizes[0], "x",
                mat2_sizes[1], " != ", self_sizes[0], "x", self_sizes[1], ")");

    at::Tensor result =
        at::empty(get_matmul_output_sizes(mat1, mat2), mat1.options());
    result.copy_(self_t);
    auto result_ = result.is_contiguous() ? result : result.contiguous();
    // Scalar to float
    // beta.to<float>()

    return zendnn_matmul_impl(mat1, mat2, result_, beta.to<float>(),
                              alpha.to<float>(), fuse_relu);
  }
}

at::Tensor zendnn_baddbmm(const at::Tensor &self, const at::Tensor &batch1,
                          const at::Tensor &batch2, const at::Scalar &beta,
                          const at::Scalar &alpha) {
  LOG(INFO) << "Executing function: " << __FUNCTION__;

  TORCH_CHECK(
      (batch1.scalar_type() == c10::ScalarType::Float &&
       batch2.scalar_type() == c10::ScalarType::Float &&
       self.scalar_type() == c10::ScalarType::Float) ||
          (batch1.scalar_type() == c10::ScalarType::BFloat16 &&
           batch2.scalar_type() == c10::ScalarType::BFloat16 &&
           self.scalar_type() == c10::ScalarType::BFloat16),
      "zendnn_baddbmm: zendnn_baddbmm only supports Float and BFloat16");
  if (self.numel() == 0) {
    TORCH_CHECK(false, "zendnn_baddbmm: incorrect self tensor");
  }
  TORCH_CHECK((self.dim() == 3 && batch1.dim() == 3 &&
               batch2.dim() == 3), // aten::baddbmm
              "zendnn_baddbmm:  unsupported dims for self, batch1 and batch2");

  // Array access is faster than .size(n)
  auto mat1_sizes = batch1.sizes();
  auto mat2_sizes = batch2.sizes();
  const auto self_sizes = self.sizes();

  TORCH_CHECK(
      self_sizes[0] == mat1_sizes[0] && self_sizes[0] == mat2_sizes[0] &&
          self_sizes[1] == mat1_sizes[1] && self_sizes[2] == mat2_sizes[2],
      "input shape is incompatible with matrix multiplication (", mat1_sizes[0],
      "x", mat1_sizes[1], "x", mat1_sizes[2], " @ ", mat2_sizes[0], "x",
      mat2_sizes[1], "x", mat2_sizes[2], " != ", self_sizes[0], "x",
      self_sizes[1], "x", self_sizes[2], ")");
  bool fuse_relu = false;
  return zendnn_matmul_impl(batch1, batch2, const_cast<at::Tensor &>(self),
                            beta.to<float>(), alpha.to<float>(), fuse_relu);
}

at::Tensor zendnn_mm(const at::Tensor &self, const at::Tensor &mat2,
                     const bool &fuse_relu) {
  LOG(INFO) << "Executing function: " << __FUNCTION__;
  TORCH_CHECK((self.dim() == 2 && mat2.dim() == 2), // aten::addmm
              "zendnn_mm:  unsupported dims for self and mat2");
  TORCH_CHECK((self.scalar_type() == c10::ScalarType::Float &&
               mat2.scalar_type() == c10::ScalarType::Float) ||
                  (self.scalar_type() == c10::ScalarType::BFloat16 &&
                   mat2.scalar_type() == c10::ScalarType::BFloat16),
              "zendnn_mm: zendnn_mm only supports Float and BFloat16");

  at::Tensor out =
      at::empty(get_matmul_output_sizes(self, mat2), self.options());
  float beta = 0.0f;
  float alpha = 1.0f;
  return zendnn_addmm(out, self, mat2, beta, alpha, fuse_relu);
}

at::Tensor zendnn_bmm(const at::Tensor &self, const at::Tensor &mat2) {
  LOG(INFO) << "Executing function: " << __FUNCTION__;
  TORCH_CHECK((self.dim() == 3 && mat2.dim() == 3), // aten::bmm
              "zendnn_bmm:  unsupported dims for self and mat2");
  TORCH_CHECK((self.scalar_type() == c10::ScalarType::Float &&
               mat2.scalar_type() == c10::ScalarType::Float) ||
                  (self.scalar_type() == c10::ScalarType::BFloat16 &&
                   mat2.scalar_type() == c10::ScalarType::BFloat16),
              "zendnn_bmm: zendnn_bmm only supports Float and BFloat16");

  at::Tensor out =
      at::empty(get_matmul_output_sizes(self, mat2), self.options());
  float beta = 0.0f;
  float alpha = 1.0f;
  return zendnn_baddbmm(out, self, mat2, beta, alpha);
}

} // namespace ZenDNNTorch
