/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "ZenDNNMemory.hpp"
#include "ZenTorchUtils.hpp"
#include <ATen/ParallelOpenMP.h>
#define ZENDNN_EMBED_BAG_THRDS 16

using namespace zendnn;

namespace ZenDNNTorch {
std::vector<at::Tensor> zendnn_fused_eb_mlp(
    const at::TensorList &eb_weight, const at::TensorList &eb_indices,
    const at::TensorList &eb_offsets,
    const at::IntArrayRef &eb_scale_grad_by_freq,
    const at::IntArrayRef &eb_mode, const at::IntArrayRef &eb_sparse,
    const c10::List<c10::optional<at::Tensor>> &eb_per_sample_weights_opt,
    const at::IntArrayRef &eb_include_last_offset,
    const at::IntArrayRef &eb_padding_idx, const at::TensorList &mlp_self,
    const at::Tensor &first_mlp_input, const at::TensorList &mlp_weights,
    const at::ArrayRef<double> &mlp_betas,
    const at::ArrayRef<double> &mlp_alphas, const at::IntArrayRef &mlp_fuse) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  int num_eb_ops = eb_weight.size();

  std::vector<memory> z_weight(num_eb_ops);
  std::vector<memory> z_indices(num_eb_ops);
  std::vector<memory> z_offsets(num_eb_ops);
  std::vector<int32_t> z_scale_grad_by_freq(num_eb_ops);
  std::vector<algorithm> z_algorithm(num_eb_ops);
  std::vector<int32_t> z_sparse(num_eb_ops);
  std::vector<memory> z_per_sample_weights_opt(num_eb_ops);
  std::vector<int32_t> z_per_sample_weights_defined(num_eb_ops);
  std::vector<int32_t> z_include_last_offset(num_eb_ops);
  std::vector<int32_t> z_padding_idx(num_eb_ops);
  std::vector<at::Tensor> output(num_eb_ops);
  std::vector<at::Tensor> temp_indices(num_eb_ops);
  std::vector<at::Tensor> temp_offsets(num_eb_ops);
  std::vector<memory> z_destination(num_eb_ops);

  std::vector<at::Tensor> out_vec((num_eb_ops * 4) + 1);

  // If TORCH_CHECK() fails, then the pragma omp parallel for cannot be used
  // we will use at::parallel_for from pytorch instead
  at::parallel_for(0, num_eb_ops, 0, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {

      zen_eb_tensor_check(eb_weight[i], eb_indices[i], eb_offsets[i]);

      temp_indices[i] = eb_indices[i].toType(c10::kInt).contiguous();
      temp_offsets[i] = eb_offsets[i].toType(c10::kInt).contiguous();

      z_weight[i] = zen_memory(eb_weight[i]);
      z_indices[i] = zen_memory(temp_indices[i]);
      z_offsets[i] = zen_memory(temp_offsets[i]);

      zen_mode_to_algo(eb_mode[i], z_algorithm[i]);

      z_padding_idx[i] = eb_padding_idx[i];
      z_scale_grad_by_freq[i] = eb_scale_grad_by_freq[i];
      z_include_last_offset[i] = eb_include_last_offset[i];
      z_sparse[i] = eb_sparse[i];

      c10::MaybeOwned<at::Tensor> per_sample_weights_maybe_owned =
          at::borrow_from_optional_tensor(eb_per_sample_weights_opt[i]);

      const at::Tensor &per_sample_weights = *per_sample_weights_maybe_owned;
      if (per_sample_weights.defined()) {
        z_per_sample_weights_opt[i] = zen_memory(per_sample_weights);
        z_per_sample_weights_defined[i] = 1;
      } else {
        z_per_sample_weights_defined[i] = 0;
      }

      int dim_embedding = eb_weight[i].sizes()[1];
      int num_bags = eb_offsets[i].sizes()[0];
      if (eb_include_last_offset[i] == 1) {
        num_bags -= 1;
      }
      output[i] = at::empty({num_bags, dim_embedding}, eb_weight[i].options());
      z_destination[i] = zen_memory(output[i]);
    }
  });

  // self = alpha * input * weights + beta * self

  int num_ops = mlp_weights.size();
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
      mlp_input = first_mlp_input;
    else
      mlp_input = dummy_output;

    dummy_output = at::empty(get_matmul_output_sizes(mlp_input, mlp_weights[i]),
                             mlp_input.options());

    alphas_vector[i] = static_cast<float>(mlp_alphas[i]);
    betas_vector[i] = static_cast<float>(mlp_betas[i]);

    // Here the if condition checks if the matrices are compatible for matrix
    // multiplication and bias addition for any general n-d case. But the
    // TORCH_CHECK conditions specifically checks for the dimensionality
    // conditions which are supported by either zendnn_addmm or
    // zendnn_addmm_1dbias

    if (mlp_self[i].sizes() ==
        c10::IntArrayRef(get_matmul_output_sizes(mlp_input, mlp_weights[i]))) {
      TORCH_CHECK((mlp_self[i].dim() == 2 && mlp_input.dim() == 2 &&
                   mlp_weights[i].dim() == 2), // aten::addmm
                  "zendnn_addmm:  unsupported dims for self, mat1 and mat2");

      const at::Tensor empty_bias; // dummy empty bias

      // Populating the bias_vector with empty bias in case of addmm
      bias_vector[i] = empty_bias;
      // Populating the self_or_result_vector with self in case of addmm since
      // that acts as the bias here
      self_or_result_vector[i] = mlp_self[i];
    } else {
      TORCH_CHECK((mlp_self[i].dim() == 1 && mlp_input.dim() == 2 &&
                   mlp_weights[i].dim() == 2), // aten::addmm
                  "zendnn_addmm: unsupported dims for self, mat1 and mat2");

      // Array access is faster than .size(n)
      const auto mat1_sizes = mlp_input.sizes();
      const auto mat2_sizes = mlp_weights[i].sizes();
      const auto self_sizes = mlp_self[i].sizes();

      TORCH_CHECK(
          self_sizes[0] == mat2_sizes[1] && mat1_sizes[1] == mat2_sizes[0],
          "input shape is incompatible with matrix multiplication (",
          mat1_sizes[0], "x", mat1_sizes[1], " @ ", mat2_sizes[0], "x",
          mat2_sizes[1], " != ", mat1_sizes[0], "x", self_sizes[0], ")");

      at::Tensor result =
          at::empty(get_matmul_output_sizes(mlp_input, mlp_weights[i]),
                    mlp_input.options());

      bias_vector[i] = mlp_self[i];
      self_or_result_vector[i] = result;
    }

    check_valid_sizes(mlp_input, mlp_weights[i]);

    if (mlp_input.scalar_type() == c10::ScalarType::BFloat16 ||
        mlp_weights[i].scalar_type() == c10::ScalarType::BFloat16) {
      TORCH_CHECK(
          utils::zendnn_bf16_device_check(),
          "zendnn_matmul: zendnn_matmul bf16 path needs the cpu support "
          "avx512bf16");
    }

    std::vector<at::Tensor> tensor_vector(3);
    tensor_vector[0] = mlp_input;
    tensor_vector[1] = mlp_weights[i];
    tensor_vector[2] = self_or_result_vector[i];

    check_scalar_type(tensor_vector);

    // ZenDNN does not support 1-D tensors. So, whenever the tensots are of
    // 1 dimension, they are unsqueezed on the required dimension to make them
    // into 2-D dimensional tensors.
    const at::Tensor &input_unsqueezed =
        mlp_input.dim() == 1 ? mlp_input.unsqueeze(0) : mlp_input;
    const at::Tensor &weight_unsqueezed = mlp_weights[i].dim() == 1
                                              ? mlp_weights[i].unsqueeze(1)
                                              : mlp_weights[i];
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
        (mlp_input.dim() == 2 && mlp_weights[i].dim() == 2)) {
      if (bias_vector[i].scalar_type() == c10::ScalarType::BFloat16) {
        TORCH_CHECK(
            utils::zendnn_bf16_device_check(),
            "zendnn_matmul: zendnn_matmul bf16 path needs the cpu support "
            "avx512bf16");
      }

      std::vector<at::Tensor> tensor_vector(1);
      tensor_vector[0] = bias_vector[i];

      check_scalar_type(tensor_vector);

      LOG(INFO) << "bias is defined for zendnn_fused_eb_mlp and bias "
                   "dimensions: "
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

    if (alphas_vector[i] != 1.0f) {
      if (bias_defined_vector[i]) {
        // TODO: add support for alpha when bias is defined
        TORCH_CHECK(
            !(mlp_input.scalar_type() == c10::ScalarType::BFloat16 ||
              mlp_weights[i].scalar_type() == c10::ScalarType::BFloat16),
            "zendnn_matmul: zendnn_matmul is not supported for bf16 "
            "tensors when bias is defined and alpha is not equal to 1");
      }
    }

    // Populating the fuse_vector with the post ops.
    fuse_vector[i] = mlp_fuse[i];
  }

  std::vector<memory> z_input_vector = {z_input};

  LOG(INFO) << "GroupEB and GroupMatMul compute in progress...";
  zendnn_custom_op::zendnn_grp_ebag_mlp(
      z_weight, z_indices, z_offsets, z_scale_grad_by_freq, z_algorithm,
      z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
      z_include_last_offset, z_padding_idx, z_destination, z_input_vector,
      z_mat2_vector, z_bias_vector, alphas_vector, betas_vector,
      bias_defined_vector, fuse_vector, z_result_vector);

  at::parallel_for(0, num_eb_ops, 0, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {
      int temp = i * 4;
      out_vec[temp + 0] = output[i];

      at::Tensor offset2bag = at::empty({});
      at::Tensor bag_size = at::empty({});
      at::Tensor max_indices = at::empty({});

      out_vec[temp + 1] = offset2bag;
      out_vec[temp + 2] = bag_size;
      out_vec[temp + 3] = max_indices;
    }
  });

  out_vec[num_eb_ops * 4] = self_or_result_vector[num_ops - 1];

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return out_vec;
}

} // namespace ZenDNNTorch
