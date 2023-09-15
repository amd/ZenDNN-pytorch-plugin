/******************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "ZenDNNMemory.hpp"

#define ZENDNN_EMBED_BAG_THRDS 16

using namespace zendnn;

namespace ZenDNNTorch {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_embedding_bag_zendnn_impl(
    const at::Tensor &weight, const at::Tensor &indices,
    const at::Tensor &offsets, const bool scale_grad_by_freq, int64_t mode,
    bool sparse, const c10::optional<at::Tensor> &per_sample_weights_opt,
    bool include_last_offset, int64_t padding_idx) {

  LOG(INFO) << "Executing function: " << __FUNCTION__;

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

  c10::MaybeOwned<at::Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const at::Tensor &per_sample_weights = *per_sample_weights_maybe_owned;

  at::Tensor cindices = indices.toType(c10::kInt).contiguous();
  at::Tensor coffsets = offsets.toType(c10::kInt).contiguous();

  at::Tensor offset2bag = at::empty({});
  at::Tensor bag_size = at::empty({});
  at::Tensor max_indices = at::empty({});

  // creating ZenDNN memory using aten tensors
  memory z_input = zen_memory(weight);
  memory z_indices = zen_memory(cindices);
  memory z_offsets = zen_memory(coffsets);

  // figure out the mode
  algorithm z_algorithm;
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

  int dim_embedding = weight.sizes()[1];
  int num_bags = coffsets.sizes()[0];

  LOG(INFO) << "Embedding matrix dimensions: " << weight.sizes()[0] << "x"
            << dim_embedding;
  LOG(INFO) << "Number of embedding bags: " << num_bags;

  // at::empty instead of at::zero is more efficient
  at::Tensor output = at::empty({num_bags, dim_embedding}, weight.options());

  memory z_dst = zen_memory(output);

  embedding_bag::desc pdesc;
  embedding_bag::primitive_desc pd;

  if (per_sample_weights.defined()) {
    LOG(INFO) << "Using the per-sample weights tensor!";
    memory z_weights = zen_memory(per_sample_weights);
    // declare embedding bag primitive
    pdesc = embedding_bag::desc(
        prop_kind::forward_inference, z_algorithm, ZENDNN_EMBED_BAG_THRDS,
        z_input.get_desc(), z_indices.get_desc(), z_offsets.get_desc(),
        z_weights.get_desc(), z_dst.get_desc(), padding_idx);

    pd = embedding_bag::primitive_desc(pdesc, utils::engine::cpu_engine());
    LOG(INFO) << "EmbeddingBag compute in progress...";
    embedding_bag(pd).execute(utils::stream::default_stream(),
                              {{ZENDNN_ARG_SRC_0, z_input},
                               {ZENDNN_ARG_SRC_1, z_indices},
                               {ZENDNN_ARG_SRC_2, z_offsets},
                               {ZENDNN_ARG_SRC_3, z_weights},
                               {ZENDNN_ARG_DST, z_dst}});
  } else {
    LOG(WARNING) << "Per-sample weights is not defined!";
    // declare embedding bag primitive
    pdesc = embedding_bag::desc(prop_kind::forward_inference, z_algorithm,
                                ZENDNN_EMBED_BAG_THRDS, z_input.get_desc(),
                                z_indices.get_desc(), z_offsets.get_desc(),
                                z_dst.get_desc(), padding_idx);

    pd = embedding_bag::primitive_desc(pdesc, utils::engine::cpu_engine());
    LOG(INFO) << "EmbeddingBag compute in progress...";
    embedding_bag(pd).execute(utils::stream::default_stream(),
                              {{ZENDNN_ARG_SRC_0, z_input},
                               {ZENDNN_ARG_SRC_1, z_indices},
                               {ZENDNN_ARG_SRC_2, z_offsets},
                               {ZENDNN_ARG_DST, z_dst}});
  }

  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> out;
  out = std::make_tuple(std::move(output), std::move(offset2bag),
                        std::move(bag_size), std::move(max_indices));

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return out;
}
} // namespace ZenDNNTorch
