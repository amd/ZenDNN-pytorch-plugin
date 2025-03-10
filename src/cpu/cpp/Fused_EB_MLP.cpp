/******************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EmbedUtils.hpp"
#include "MatmulUtils.hpp"
#include "Memory.hpp"

#include <ATen/ParallelOpenMP.h>
#define ZENDNN_EMBED_BAG_THRDS 16

using namespace zendnn;

namespace zentorch {
std::vector<at::Tensor> zentorch_fused_eb_mlp(
    at::TensorList eb_weight, at::TensorList eb_indices,
    at::TensorList eb_offsets, at::IntArrayRef eb_scale_grad_by_freq,
    at::IntArrayRef eb_mode, at::IntArrayRef eb_sparse,
    c10::List<c10::optional<at::Tensor>> eb_per_sample_weights_opt,
    at::IntArrayRef eb_include_last_offset, at::IntArrayRef eb_padding_idx,
    at::TensorList mlp_self, const at::Tensor &first_mlp_input,
    at::TensorList mlp_weights, at::ArrayRef<double> mlp_betas,
    at::ArrayRef<double> mlp_alphas, at::IntArrayRef mlp_fuse,
    std::string zentorch_op_name) {

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

      at::Tensor per_sample_weights;

      std::tie(temp_indices[i], temp_offsets[i], per_sample_weights,
               output[i]) =
          eb_tensors_to_memory(
              eb_weight[i], eb_indices[i], eb_offsets[i],
              eb_per_sample_weights_opt[i], eb_mode[i], output[i], z_weight[i],
              z_indices[i], z_offsets[i], z_per_sample_weights_opt[i],
              z_algorithm[i], z_destination[i], eb_include_last_offset[i]);

      z_padding_idx[i] = eb_padding_idx[i];
      z_scale_grad_by_freq[i] = eb_scale_grad_by_freq[i];
      z_include_last_offset[i] = eb_include_last_offset[i];
      z_sparse[i] = eb_sparse[i];

      if (per_sample_weights.defined()) {
        z_per_sample_weights_defined[i] = 1;
      } else {
        z_per_sample_weights_defined[i] = 0;
      }
    }
  });

  // self = alpha * input * weights + beta * self

  int num_ops = mlp_weights.size();
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
  memory z_input;

  at::Tensor mlp_input, dummy_output;

  for (int i = 0; i < num_ops; i++) {
    if (i == 0)
      mlp_input = first_mlp_input;
    else
      mlp_input = dummy_output;

    dummy_output =
        at::empty(get_matmul_and_linear_output_sizes(mlp_input, mlp_weights[i]),
                  mlp_input.options());

    alphas_vector[i] = static_cast<float>(mlp_alphas[i]);
    betas_vector[i] = static_cast<float>(mlp_betas[i]);

    // Here the if condition checks if the matrices are compatible for matrix
    // multiplication and bias addition for any general n-d case. But the
    // TORCH_CHECK conditions specifically checks for the dimensionality
    // conditions which are supported by either zentorch_addmm or
    // zentorch_addmm_1dbias

    if (mlp_self[i].sizes() ==
        c10::IntArrayRef(
            get_matmul_and_linear_output_sizes(mlp_input, mlp_weights[i]))) {
      ZENTORCH_CHECK((mlp_self[i].dim() == 2 && mlp_input.dim() == 2 &&
                      mlp_weights[i].dim() == 2), // aten::addmm
                     "unsupported dims for self, mat1 and mat2");

      const at::Tensor empty_bias; // dummy empty bias

      // Populating the bias_vector with empty bias in case of addmm
      bias_vector[i] = empty_bias;
      // Populating the self_or_result_vector with self in case of addmm since
      // that acts as the bias here
      self_or_result_vector[i] = mlp_self[i];
    } else {
      ZENTORCH_CHECK((mlp_self[i].dim() == 1 && mlp_input.dim() == 2 &&
                      mlp_weights[i].dim() == 2), // aten::addmm
                     "unsupported dims for self, mat1 and mat2");

      at::Tensor result = at::empty(
          get_matmul_and_linear_output_sizes(mlp_input, mlp_weights[i]),
          mlp_input.options());

      bias_vector[i] = mlp_self[i];
      self_or_result_vector[i] = result;
    }

    at::Tensor self_or_result_unsqueezed, mat1_, mat2_, beta_bias;
    std::vector<at::Tensor> post_op_buffers = {};
    std::tie(self_or_result_unsqueezed, mat1_, mat2_, beta_bias) =
        matmul_tensors_to_memory(
            mlp_input, mlp_weights[i], self_or_result_vector[i], bias_vector[i],
            beta_bias, post_op_buffers, z_mat1_vector[i], z_mat2_vector[i],
            z_bias_vector[i], z_result_vector[i], betas_vector[i],
            alphas_vector[i]);

    // Populating the bias_defined_vector with the bool equivalent values based
    // on the number of elements in the bias.
    bias_defined_vector[i] = bias_vector[i].numel();

    if (alphas_vector[i] != 1.0f) {
      if (bias_defined_vector[i]) {
        // TODO: add support for alpha when bias is defined
        ZENTORCH_CHECK(
            !(mlp_input.scalar_type() == c10::ScalarType::BFloat16 ||
              mlp_weights[i].scalar_type() == c10::ScalarType::BFloat16),
            "zentorch_matmul is not supported for bf16 "
            "tensors when bias is defined and alpha is not equal to 1");
      }
    }

    // Populating the fuse_vector with the post ops.
    // Currently there is a limitation from the PyTorch side that a list of
    // lists cannot be sent from the Python side to the CPP side via bindings
    // using TORCH_LIBRARY_FRAGMENT. So, currently we are using just a single
    // post op and wrapping it in a vector and sending it to ZenDNN Library.
    z_post_op_ids.push_back({mlp_fuse[i]});
  }

  // The current optimization uses Group EmbeddingBag and Group MLP ops under
  // the hood. So, we just need the first value of z_mat1_vector, as it is the
  // input to the Group MLP op. So, creating a vector with the first element of
  // z_mat1_vector.
  std::vector<memory> z_input_vector = {z_mat1_vector[0]};

  LOG(INFO) << "GroupEB and GroupMatMul compute in progress...";
  zendnn_custom_op::zendnn_grp_ebag_mlp(
      z_weight, z_indices, z_offsets, z_scale_grad_by_freq, z_algorithm,
      z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
      z_include_last_offset, z_padding_idx, z_destination, z_input_vector,
      z_mat2_vector, z_bias_vector, alphas_vector, betas_vector,
      bias_defined_vector, z_post_op_ids, z_post_op_buffers, z_result_vector,
      zentorch_op_name.c_str());

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

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def(
      "zentorch_fused_eb_mlp(Tensor[] eb_weight, Tensor[] eb_indices, "
      "Tensor[] eb_offsets, int[] eb_scale_grad_by_freq, int[] eb_mode, int[] "
      "eb_sparse, Tensor?[] eb_per_sample_weights, "
      "int[] eb_include_last_offset, int[] eb_padding_idx, Tensor[] mlp_self, "
      "Tensor mlp_inputs, Tensor[] mlp_weight, float[] mlp_betas, "
      "float[] mlp_alphas, int[] mlp_fuse, str zentorch_op_name = "
      "'zentorch::zentorch_fused_eb_mlp') -> Tensor[]");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_fused_eb_mlp", zentorch_fused_eb_mlp);
}

} // namespace zentorch
