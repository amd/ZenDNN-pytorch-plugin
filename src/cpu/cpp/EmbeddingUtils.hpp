/******************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/
#pragma once

#include "Memory.hpp"

using namespace zendnnl::interface;

namespace zentorch {

inline void zen_embedding_weight_check(const at::Tensor &weight) {
  const bool is_weight_bf16 =
      (weight.scalar_type() == c10::ScalarType::BFloat16);
  const bool is_weight_fp32 = (weight.scalar_type() == c10::ScalarType::Float);
  // check if the device supports AVX512
  if (is_weight_bf16) {
    ZENTORCH_CHECK(zentorch::zendnn_bf16_device_check(),
                   "zentorch_embedding bf16 path needs the cpu support "
                   "avx512bf16");
  }
  // check if datatype is either Float32 or Bfloat16
  ZENTORCH_CHECK(is_weight_fp32 ^ is_weight_bf16,
                 "zentorch_embedding only supports Float and BFloat16");
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
  ZENTORCH_CHECK(!embedding_operator.is_bad_object(), "operator ",
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
