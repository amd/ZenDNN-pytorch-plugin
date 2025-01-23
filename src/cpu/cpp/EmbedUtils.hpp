/******************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/
#pragma once

#include "Memory.hpp"

namespace zentorch {

using namespace zendnn;

// The following overloaded function is called when the tensors are being
// checked for the embedding op. Embedding op has only two tensors that need to
// be checked for device and layout. So, the implementation below suffices.
inline void zen_embed_tensor_check(const at::Tensor &weight,
                                   const at::Tensor &indices) {
  const bool is_weight_bf16 =
      (weight.scalar_type() == c10::ScalarType::BFloat16);
  const bool is_weight_fp32 = (weight.scalar_type() == c10::ScalarType::Float);
  // check if all the input tensors are on cpu device
  ZENTORCH_CHECK(weight.device().is_cpu() && indices.device().is_cpu(),
                 "ZenDNN Embedding expects CPU tensor inputs!");
  // check if all the input tensors are dense format
  ZENTORCH_CHECK((weight.layout() == c10::Layout::Strided) &&
                     (indices.layout() == c10::Layout::Strided),
                 "ZenDNN Embedding expects dense tensor inputs!");
  // check if the device supports AVX512
  if (is_weight_bf16) {
    ZENTORCH_CHECK(utils::zendnn_bf16_device_check(),
                   "zentorch_embedding bf16 path needs the cpu support "
                   "avx512bf16");
  }
  // check if datatype is either Float32 or Bfloat16
  ZENTORCH_CHECK(is_weight_fp32 ^ is_weight_bf16,
                 "zentorch_embedding only supports Float and BFloat16");
}

// The following overloaded function is called when the tensors are being
// checked for the embeddingbag op. Embedding op has three tensors that need to
// be checked for device and layout. So, the implementation below suffices.
inline void zen_embed_tensor_check(const at::Tensor &weight,
                                   const at::Tensor &indices,
                                   const at::Tensor &offsets) {
  const bool is_weight_bf16 =
      (weight.scalar_type() == c10::ScalarType::BFloat16);
  const bool is_weight_fp32 = (weight.scalar_type() == c10::ScalarType::Float);
  // check if all the input tensors are on cpu device
  ZENTORCH_CHECK(weight.device().is_cpu() && indices.device().is_cpu() &&
                     offsets.device().is_cpu(),
                 "ZenDNN EmbeddingBag expects CPU tensor inputs!");
  // check if all the input tensors are dense format
  ZENTORCH_CHECK((weight.layout() == c10::Layout::Strided) &&
                     (indices.layout() == c10::Layout::Strided) &&
                     (offsets.layout() == c10::Layout::Strided),
                 "ZenDNN EmbeddingBag expects dense tensor inputs!");
  // check if the device supports AVX512
  if (is_weight_bf16) {
    ZENTORCH_CHECK(utils::zendnn_bf16_device_check(),
                   "zentorch_embedding_bag bf16 path needs the cpu support "
                   "avx512bf16");
  }
  // check if datatype is either Float32 or Bfloat16
  ZENTORCH_CHECK(is_weight_fp32 ^ is_weight_bf16,
                 "zentorch_embedding_bag only supports Float and BFloat16");
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

// Whichever tensors are converted to ZenDNN memory inside the
// tensors_to_memory function, there must be a aten tensor as well that points
// to the same space. This is done to avoid corruption of values of the tensors
// that are converted to zendnn memory. In this function four tensors are
// converted to zendnn memory, so we return those tensors to the calling
// function to have a aten tensor to point to the same space.
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
eb_tensors_to_memory(const at::Tensor &weight, const at::Tensor &indices,
                     const at::Tensor &offsets,
                     const c10::optional<at::Tensor> &per_sample_weights_opt,
                     const int64_t &mode, at::Tensor &output, memory &z_weight,
                     memory &z_indices, memory &z_offsets, memory &z_weights,
                     algorithm &z_algorithm, memory &z_destination,
                     bool include_last_offset) {
  zen_embed_tensor_check(weight, indices, offsets);

  at::Tensor cindices = indices.toType(c10::kInt).contiguous();
  at::Tensor coffsets = offsets.toType(c10::kInt).contiguous();

  int dim_embedding = weight.sizes()[1];
  int num_bags = coffsets.sizes()[0];
  if (include_last_offset == true) {
    num_bags -= 1;
  }
  LOG(INFO) << "Embedding matrix dimensions: " << weight.sizes()[0] << "x"
            << dim_embedding;
  LOG(INFO) << "Number of embedding bags: " << num_bags;

  output = at::empty({num_bags, dim_embedding}, weight.options());

  c10::MaybeOwned<at::Tensor> per_sample_weights_opt_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const at::Tensor &per_sample_weights = *per_sample_weights_opt_maybe_owned;

  // creating ZenDNN memory using aten tensors
  z_weight = zen_memory(weight);
  z_indices = zen_memory(cindices);
  z_offsets = zen_memory(coffsets);
  if (per_sample_weights.defined()) {
    z_weights = zen_memory(per_sample_weights);
  }
  z_destination = zen_memory(output);

  // figure out the mode
  zen_mode_to_algo(mode, z_algorithm);

  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> out;
  out = std::make_tuple(std::move(cindices), std::move(coffsets),
                        std::move(per_sample_weights), std::move(output));

  return out;
}

// Same as the previous function in terms of operation, just works with two
// tensors instead of four.
inline std::tuple<at::Tensor, at::Tensor>
embed_tensors_to_memory(const at::Tensor &weight, const at::Tensor &indices,
                        memory &z_weight, memory &z_indices, memory &z_dst) {
  zen_embed_tensor_check(weight, indices);

  at::Tensor cindices = indices.toType(c10::kInt).contiguous();

  int dim_embedding = weight.sizes()[1];
  int num_indices = cindices.sizes()[0];

  LOG(INFO) << "Embedding matrix dimensions: " << weight.sizes()[0] << "x"
            << dim_embedding;
  LOG(INFO) << "Number of indices: " << num_indices;

  // at::empty instead of at::zero is more efficient
  at::Tensor output = at::empty({num_indices, dim_embedding}, weight.options());

  z_weight = zen_memory(weight);
  z_indices = zen_memory(cindices);
  z_dst = zen_memory(output);

  std::tuple<at::Tensor, at::Tensor> out;
  out = std::make_tuple(std::move(cindices), std::move(output));

  return out;
}
} // namespace zentorch
