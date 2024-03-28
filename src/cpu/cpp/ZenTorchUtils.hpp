/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

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

inline void zen_eb_tensor_check(const at::Tensor &weight,
                                const at::Tensor &indices,
                                const at::Tensor &offsets) {
  // check if all the input tensors are on cpu device
  TORCH_CHECK(weight.device().is_cpu() && indices.device().is_cpu() &&
                  offsets.device().is_cpu(),
              "ZenDNN EmbeddingBag expects CPU tensor inputs!");
  // check if all the input tensors are dense format
  TORCH_CHECK((weight.layout() == c10::Layout::Strided) &&
                  (indices.layout() == c10::Layout::Strided) &&
                  (offsets.layout() == c10::Layout::Strided),
              "ZenDNN EmbeddingBag expects dense tensor inputs!");
  // check the weight type for embedding bag, only supported is fp32 for now
  // (works ONLY for dtype=torch.float32)
  TORCH_CHECK(weight.scalar_type() == c10::kFloat,
              "Only fp32 type weights are supported in ZenDNN EmbeddingBag!");
}

inline void zen_mode_to_algo(const int64_t &mode, algorithm &z_algorithm) {
  switch (mode) {
  case 0:
    z_algorithm = algorithm::embedding_bag_sum;
    break;
  case 1:
    z_algorithm = algorithm::embedding_bag_mean;
    break;
  case 2:
    z_algorithm = algorithm::embedding_bag_max;
    break;
  default:
    z_algorithm = algorithm::embedding_bag_sum;
    break;
  }
}