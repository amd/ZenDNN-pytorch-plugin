/******************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EmbedUtils.hpp"
#include "Memory.hpp"
#define ZENDNN_EMBED_THRDS 16

using namespace zendnn;

namespace zentorch {
at::Tensor zentorch_embedding_impl(const at::Tensor &weight,
                                   const at::Tensor &indices,
                                   const int64_t &padding_idx,
                                   const bool &scale_grad_by_freq,
                                   const bool &sparse,
                                   std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  at::Tensor cindices, output;
  memory z_weight, z_indices, z_dst;
  std::tie(cindices, output) =
      embed_tensors_to_memory(weight, indices, z_weight, z_indices, z_dst);

  // Currently there is no primitive for embedding as an op.
  // So, the manipulations on the embeddingbag op are taken care by the
  // ZenDNN library and the ZenDNN library call is made from the plugin side.
  LOG(INFO) << "Embedding compute in progress...";
  zendnn_custom_op::zendnn_embedding(
      z_weight, z_indices, static_cast<int32_t>(padding_idx),
      scale_grad_by_freq, sparse, z_dst, zentorch_op_name.c_str());

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

std::vector<at::Tensor> zentorch_horizontal_embedding_group(
    const at::TensorList &weight, const at::TensorList &indices,
    const at::IntArrayRef &padding_idx,
    const at::IntArrayRef &scale_grad_by_freq, const at::IntArrayRef &sparse,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  int num_eb_ops = weight.size();

  std::vector<memory> z_weights(num_eb_ops);
  std::vector<memory> z_indices(num_eb_ops);
  std::vector<int32_t> z_padding_idx(num_eb_ops);
  std::vector<int32_t> z_scale_grad_by_freq(num_eb_ops);
  std::vector<int32_t> z_sparse(num_eb_ops);

  std::vector<at::Tensor> temp_indices(num_eb_ops);
  std::vector<at::Tensor> output(num_eb_ops);
  std::vector<memory> z_destination(num_eb_ops);

  at::parallel_for(0, num_eb_ops, 0, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {

      std::tie(temp_indices[i], output[i]) = embed_tensors_to_memory(
          weight[i], indices[i], z_weights[i], z_indices[i], z_destination[i]);

      z_padding_idx[i] = padding_idx[i];
      z_scale_grad_by_freq[i] = scale_grad_by_freq[i];
      z_sparse[i] = sparse[i];
    }
  });

  LOG(INFO) << "GroupEmbedding compute in progress...";
  zendnn_custom_op::zendnn_grp_embedding(
      z_weights, z_indices, z_padding_idx, z_scale_grad_by_freq, z_sparse,
      z_destination, zentorch_op_name.c_str());
  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}
} // namespace zentorch
