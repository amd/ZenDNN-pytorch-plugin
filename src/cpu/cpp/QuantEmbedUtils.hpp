/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "EmbedUtils.hpp"

namespace zentorch {

using namespace zendnn;

inline void zen_quant_embed_tensor_check(const at::Tensor &weight,
                                         const at::Tensor &indices,
                                         const at::Tensor &offsets) {
  const bool is_weight_int32 = (weight.scalar_type() == c10::ScalarType::Int);

  // check if all the input tensors are on cpu device
  ZENTORCH_CHECK(weight.device().is_cpu() && indices.device().is_cpu() &&
                     offsets.device().is_cpu(),
                 "ZenDNN EmbeddingBag expects CPU tensor inputs!");
  // check if all the input tensors are dense format
  ZENTORCH_CHECK((weight.layout() == c10::Layout::Strided) &&
                     (indices.layout() == c10::Layout::Strided) &&
                     (offsets.layout() == c10::Layout::Strided),
                 "ZenDNN EmbeddingBag expects dense tensor inputs!");

  ZENTORCH_CHECK(
      is_weight_int32,
      "zentorch_embedding_bag only supports int4 weights packed into int32_t");
}

// Whichever tensors are converted to ZenDNN memory inside the
// tensors_to_memory function, there must be a aten tensor as well that points
// to the same space. This is done to avoid corruption of values of the tensors
// that are converted to zendnn memory. In this function four tensors are
// converted to zendnn memory, so we return those tensors to the calling
// function to have a aten tensor to point to the same space.
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
quant_eb_tensors_to_memory(
    const at::Tensor &weight, const at::Tensor &indices,
    const at::Tensor &offsets,
    const c10::optional<at::Tensor> &per_sample_weights_opt,
    const int64_t &mode, at::Tensor &output, memory &z_weight,
    memory &z_indices, memory &z_offsets, memory &z_per_sample_weights_opt,
    algorithm &z_algorithm, memory &z_destination, bool include_last_offset,
    const int64_t &num_bits_per_weight, const at::ScalarType &output_dtype) {

  zen_quant_embed_tensor_check(weight, indices, offsets);

  ZENTORCH_CHECK(
      (output_dtype == c10::ScalarType::Float ||
       output_dtype == c10::ScalarType::BFloat16),
      "zentorch_embedding_bag only supports fp32 or bf16 output types");
  ZENTORCH_CHECK(
      num_bits_per_weight == 4,
      "zentorch_embedding_bag only supports uint4 quantized weights");

  at::Tensor cindices = indices.toType(c10::kInt).contiguous();
  at::Tensor coffsets = offsets.toType(c10::kInt).contiguous();

  int num_bags = coffsets.sizes()[0];
  if (include_last_offset == true) {
    num_bags -= 1;
  }

  int weight_dim = weight.sizes()[1];

  // Currently assumes scale and zero point to be of type BFloat16 each
  int num_dim_scale_zp =
      (sizeof(c10::BFloat16) + sizeof(c10::BFloat16)) / weight.element_size();

  int packed_weight_dim = weight_dim - (num_dim_scale_zp);
  const int64_t bits_in_1_byte = 8;
  int num_bits_per_packed_weight = weight.element_size() * bits_in_1_byte;

  // to retreive original embedding dim before int4 was packed into int32
  // packed_weight_dim * (32 / 4) (int32/int4)

  int embedding_dim =
      packed_weight_dim * (num_bits_per_packed_weight / num_bits_per_weight);

  int num_int4_elem =
      weight_dim * (num_bits_per_packed_weight / num_bits_per_weight);

  LOG(INFO) << "Embedding matrix dimensions: " << weight.sizes()[0] << "x"
            << weight_dim;

  LOG(INFO) << "Int4 weight matrix dimensions: " << weight.sizes()[0] << "x"
            << embedding_dim;

  LOG(INFO) << "Int4 weights with scale and zp dimensions: "
            << weight.sizes()[0] << "x" << num_int4_elem;

  LOG(INFO) << "Output dimensions: " << num_bags << "x" << embedding_dim;
  // at::empty instead of at::zero is more efficient
  output = at::empty({num_bags, embedding_dim}, output_dtype);
  c10::MaybeOwned<at::Tensor> per_sample_weights_opt_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const at::Tensor &per_sample_weights = *per_sample_weights_opt_maybe_owned;

  // qweight is packed,
  // but zendnn::memory creation requires the original weight dims for s4
  // qweight memory, so we get original dims by unpacking last dim of
  // qweight tensor(qweight.last_dim * unpacking_ratio).
  // Create qweight memory

  const memory::format_tag &memory_2d_tag = memory::format_tag::ab;
  const memory::data_type &memory_int4_dtype = memory::data_type::s4;
  const memory::desc &qweight_desc = memory::desc(
      {weight.sizes()[0], num_int4_elem}, memory_int4_dtype, memory_2d_tag);

  z_weight = zen_memory(weight, qweight_desc);

  z_indices = zen_memory(cindices);

  z_offsets = zen_memory(coffsets);
  if (per_sample_weights.defined()) {
    z_per_sample_weights_opt = zen_memory(per_sample_weights);
  }

  z_destination = zen_memory(output);

  // figure out the mode
  zen_mode_to_algo(mode, z_algorithm);

  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> out;
  out = std::make_tuple(std::move(cindices), std::move(coffsets),
                        std::move(per_sample_weights), std::move(output));
  return out;
}

} // namespace zentorch