/******************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/
#pragma once

#include "Memory.hpp"

using namespace zendnn;
using namespace zendnnl::interface;

namespace zentorch {

inline void zen_embedding_weight_check(const at::Tensor &weight) {
  const bool is_weight_bf16 =
      (weight.scalar_type() == c10::ScalarType::BFloat16);
  const bool is_weight_fp32 = (weight.scalar_type() == c10::ScalarType::Float);
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
embeddingbag_tensors_to_memory(
    const at::Tensor &weight, const at::Tensor &indices,
    const at::Tensor &offsets,
    const c10::optional<at::Tensor> &per_sample_weights_opt,
    const int64_t &mode, at::Tensor &output, memory &z_weight,
    memory &z_indices, memory &z_offsets, memory &z_weights,
    algorithm &z_algorithm, memory &z_destination, bool include_last_offset) {
  zen_embedding_weight_check(weight);

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

  output = at::detail::empty_strided_cpu({num_bags, dim_embedding},
                                         {dim_embedding, 1}, weight.options());

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

inline std::tuple<at::Tensor, at::Tensor>
embedding_tensors_to_memory(const at::Tensor &weight, const at::Tensor &indices,
                            memory &z_weight, memory &z_indices,
                            memory &z_dst) {
  zen_embedding_weight_check(weight);

  at::Tensor cindices = indices.toType(c10::kInt).contiguous();

  int dim_embedding = weight.sizes()[1];
  int num_indices = cindices.sizes()[0];

  LOG(INFO) << "Embedding matrix dimensions: " << weight.sizes()[0] << "x"
            << dim_embedding;
  LOG(INFO) << "Number of indices: " << num_indices;

  // at::detail::empty_strided_cpu instead of at::zero is more efficient
  at::Tensor output = at::detail::empty_strided_cpu(
      {num_indices, dim_embedding}, {dim_embedding, 1}, weight.options());

  z_weight = zen_memory(weight);
  z_indices = zen_memory(cindices);
  z_dst = zen_memory(output);

  std::tuple<at::Tensor, at::Tensor> out;
  out = std::make_tuple(std::move(cindices), std::move(output));

  return out;
}

inline void set_embedding_context_attributes(
    embag_context_t &embedding_context, tensor_t &table, const int64_t &mode,
    const bool &include_last_offset, const int64_t &padding_idx,
    const bool &per_sample_weights_defined) {
  embedding_context.set_param("table", table);
  switch (mode) {
  case EMBEDDING_BAG_ALGO::SUM:
    embedding_context.set_algo(embag_algo_t::sum);
    break;
  case EMBEDDING_BAG_ALGO::MEAN:
    embedding_context.set_algo(embag_algo_t::mean);
    break;
  case EMBEDDING_BAG_ALGO::MAX:
    embedding_context.set_algo(embag_algo_t::max);
    break;
  default:
    break;
  }
  embedding_context.set_include_last_offset(include_last_offset ? 1 : 0)
      .set_padding_index(padding_idx)
      .set_is_weights(per_sample_weights_defined);

  embedding_context.create();

  ZENTORCH_CHECK(embedding_context.check(),
                 "embedding context creation failed");
}

inline void set_embedding_operator_attributes(
    embag_operator_t &embedding_operator, const std::string &operator_name,
    embag_context_t &embedding_context, tensor_t &indices, tensor_t &output,
    std::optional<std::reference_wrapper<tensor_t>> offsets_opt_ref =
        std::nullopt,
    std::optional<std::reference_wrapper<tensor_t>> per_sample_weights_opt_ref =
        std::nullopt) {
  embedding_operator.set_name(operator_name)
      .set_context(embedding_context)
      .create();
  ZENTORCH_CHECK(embedding_operator.check(), "operator ",
                 embedding_operator.get_name(), " creation failed.");

  embedding_operator.set_input("indices", indices).set_output("output", output);

  if (offsets_opt_ref.has_value()) {
    tensor_t &offsets = offsets_opt_ref->get();
    embedding_operator.set_input("offsets", offsets);
  }

  if (per_sample_weights_opt_ref.has_value()) {
    tensor_t &per_sample_weights = per_sample_weights_opt_ref->get();
    embedding_operator.set_input("weights", per_sample_weights);
  }
}
} // namespace zentorch
