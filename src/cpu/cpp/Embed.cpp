/******************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "ZenDNNMemory.hpp"

#define ZENDNN_EMBED_THRDS 16

inline void zen_embed_tensor_check(const at::Tensor &weight,
                                   const at::Tensor &indices) {
  // check if all the input tensors are on cpu device
  TORCH_CHECK(weight.device().is_cpu() && indices.device().is_cpu(),
              "ZenDNN Embedding expects CPU tensor inputs!");
  // check if all the input tensors are dense format
  TORCH_CHECK((weight.layout() == c10::Layout::Strided) &&
                  (indices.layout() == c10::Layout::Strided),
              "ZenDNN Embedding expects dense tensor inputs!");
  // check the weight type for embedding, only supported is fp32 for now
  // (works ONLY for dtype=torch.float32)
  TORCH_CHECK(weight.scalar_type() == c10::kFloat,
              "Only fp32 type weights are supported in ZenDNN Embedding!");
}

using namespace zendnn;

namespace ZenDNNTorch {
at::Tensor zendnn_embedding_impl(const at::Tensor &weight,
                                 const at::Tensor &indices,
                                 const int64_t &padding_idx,
                                 const bool &scale_grad_by_freq,
                                 const bool &sparse) {

  LOG(INFO) << "Executing function: " << __FUNCTION__;

  zen_embed_tensor_check(weight, indices);

  at::Tensor cindices = indices.toType(c10::kInt).contiguous();

  // creating ZenDNN memory using aten tensors
  memory z_weight = zen_memory(weight);
  memory z_indices = zen_memory(cindices);

  int dim_embedding = weight.sizes()[1];
  int num_indices = cindices.sizes()[0];

  LOG(INFO) << "Embedding matrix dimensions: " << weight.sizes()[0] << "x"
            << dim_embedding;
  LOG(INFO) << "Number of indices: " << num_indices;

  // at::empty instead of at::zero is more efficient
  at::Tensor output = at::empty({num_indices, dim_embedding}, weight.options());

  memory z_dst = zen_memory(output);

  // Currently there is no primitive for embedding as an op.
  // So, the manipulations on the embeddingbag op are taken care by the
  // ZenDNN library and the ZenDNN library call is made from the plugin side.

  zendnn_custom_op::zendnn_embedding(z_weight, z_indices,
                                     static_cast<int32_t>(padding_idx),
                                     scale_grad_by_freq, sparse, z_dst);

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

std::vector<at::Tensor> zendnn_custom_embedding_group(
    const at::TensorList &weight, const at::TensorList &indices,
    const at::IntArrayRef &padding_idx,
    const at::IntArrayRef &scale_grad_by_freq, const at::IntArrayRef &sparse) {

  int num_eb_ops = weight.size();

  std::vector<memory> z_weights(num_eb_ops);
  std::vector<memory> z_indices(num_eb_ops);
  std::vector<int32_t> z_padding_idx(num_eb_ops);
  std::vector<int32_t> z_scale_grad_by_freq(num_eb_ops);
  std::vector<int32_t> z_sparse(num_eb_ops);

  std::vector<at::Tensor> temp_indices(num_eb_ops);
  std::vector<at::Tensor> output(num_eb_ops);
  std::vector<memory> z_destination(num_eb_ops);

#pragma omp parallel for
  for (int i = 0; i < num_eb_ops; i++) {

    zen_embed_tensor_check(weight[i], indices[i]);

    temp_indices[i] = indices[i].toType(c10::kInt).contiguous();

    z_weights[i] = zen_memory(weight[i]);
    z_indices[i] = zen_memory(temp_indices[i]);

    z_padding_idx[i] = padding_idx[i];
    z_scale_grad_by_freq[i] = scale_grad_by_freq[i];
    z_sparse[i] = sparse[i];

    int dim_embedding = weight[i].sizes()[1];
    int num_indices = indices[i].sizes()[0];

    output[i] = at::empty({num_indices, dim_embedding}, weight[i].options());
    z_destination[i] = zen_memory(output[i]);
  }

  zendnn_custom_op::zendnn_grp_embedding(z_weights, z_indices, z_padding_idx,
                                         z_scale_grad_by_freq, z_sparse,
                                         z_destination);

  std::vector<at::Tensor> out_vec(num_eb_ops);

#pragma omp parallel for
  for (int i = 0; i < num_eb_ops; i++) {
    out_vec[i] = output[i];
  }

  return out_vec;
}
} // namespace ZenDNNTorch
