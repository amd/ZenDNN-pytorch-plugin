/******************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "ZenDNNMemory.hpp"

#define ZENDNN_EMBED_BAG_THRDS 16

inline void zen_eb_tensor_check(at::Tensor weight, at::Tensor indices,
                                at::Tensor offsets) {
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

using namespace zendnn;

namespace ZenDNNTorch {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
zendnn_embedding_bag_impl(
    const at::Tensor &weight, const at::Tensor &indices,
    const at::Tensor &offsets, const bool &scale_grad_by_freq,
    const int64_t &mode, const bool &sparse,
    const c10::optional<at::Tensor> &per_sample_weights_opt,
    const bool &include_last_offset, const int64_t &padding_idx) {

  LOG(INFO) << "Executing function: " << __FUNCTION__;

  zen_eb_tensor_check(weight, indices, offsets);

  c10::MaybeOwned<at::Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const at::Tensor &per_sample_weights = *per_sample_weights_maybe_owned;

  at::Tensor cindices = indices.toType(c10::kInt).contiguous();
  at::Tensor coffsets = offsets.toType(c10::kInt).contiguous();

  at::Tensor offset2bag = at::empty({});
  at::Tensor bag_size = at::empty({});
  at::Tensor max_indices = at::empty({});

  // creating ZenDNN memory using aten tensors
  memory z_weight = zen_memory(weight);
  memory z_indices = zen_memory(cindices);
  memory z_offsets = zen_memory(coffsets);

  // figure out the mode
  algorithm z_algorithm;
  zen_mode_to_algo(mode, z_algorithm);

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
        z_weight.get_desc(), z_indices.get_desc(), z_offsets.get_desc(),
        z_weights.get_desc(), z_dst.get_desc(), padding_idx);

    pd = embedding_bag::primitive_desc(pdesc, utils::engine::cpu_engine());
    LOG(INFO) << "EmbeddingBag compute in progress...";
    embedding_bag(pd).execute(utils::stream::default_stream(),
                              {{ZENDNN_ARG_SRC_0, z_weight},
                               {ZENDNN_ARG_SRC_1, z_indices},
                               {ZENDNN_ARG_SRC_2, z_offsets},
                               {ZENDNN_ARG_SRC_3, z_weights},
                               {ZENDNN_ARG_DST, z_dst}});
  } else {
    LOG(WARNING) << "Per-sample weights is not defined!";
    // declare embedding bag primitive
    pdesc = embedding_bag::desc(prop_kind::forward_inference, z_algorithm,
                                ZENDNN_EMBED_BAG_THRDS, z_weight.get_desc(),
                                z_indices.get_desc(), z_offsets.get_desc(),
                                z_dst.get_desc(), padding_idx);

    pd = embedding_bag::primitive_desc(pdesc, utils::engine::cpu_engine());
    LOG(INFO) << "EmbeddingBag compute in progress...";
    embedding_bag(pd).execute(utils::stream::default_stream(),
                              {{ZENDNN_ARG_SRC_0, z_weight},
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

std::vector<at::Tensor> zendnn_custom_embedding_bag_group(
    const at::TensorList &weight, const at::TensorList &indices,
    const at::TensorList &offsets, const at::IntArrayRef &scale_grad_by_freq,
    const at::IntArrayRef &mode, const at::IntArrayRef &sparse,
    const c10::List<c10::optional<at::Tensor>> &per_sample_weights_opt,
    const at::IntArrayRef &include_last_offset,
    const at::IntArrayRef &padding_idx) {

  int num_eb_ops = weight.size();

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

  std::vector<at::Tensor> temp_indices(num_eb_ops);
  std::vector<at::Tensor> temp_offsets(num_eb_ops);
  std::vector<at::Tensor> output(num_eb_ops);
  std::vector<memory> z_destination(num_eb_ops);

  std::vector<at::Tensor> out_vec(num_eb_ops * 4);

#pragma omp parallel for
  for (int i = 0; i < num_eb_ops; i++) {

    zen_eb_tensor_check(weight[i], indices[i], offsets[i]);

    temp_indices[i] = indices[i].toType(c10::kInt).contiguous();
    temp_offsets[i] = offsets[i].toType(c10::kInt).contiguous();

    z_weight[i] = zen_memory(weight[i]);
    z_indices[i] = zen_memory(temp_indices[i]);
    z_offsets[i] = zen_memory(temp_offsets[i]);

    zen_mode_to_algo(mode[i], z_algorithm[i]);

    z_padding_idx[i] = padding_idx[i];
    z_scale_grad_by_freq[i] = scale_grad_by_freq[i];
    z_include_last_offset[i] = include_last_offset[i];
    z_sparse[i] = sparse[i];

    c10::MaybeOwned<at::Tensor> per_sample_weights_maybe_owned =
        at::borrow_from_optional_tensor(per_sample_weights_opt[i]);

    const at::Tensor &per_sample_weights = *per_sample_weights_maybe_owned;
    if (per_sample_weights.defined()) {
      z_per_sample_weights_opt[i] = zen_memory(per_sample_weights);
      z_per_sample_weights_defined[i] = 1;
    } else {
      z_per_sample_weights_defined[i] = 0;
    }

    int dim_embedding = weight[i].sizes()[1];
    int num_bags = offsets[i].sizes()[0];

    output[i] = at::empty({num_bags, dim_embedding}, weight[i].options());
    z_destination[i] = zen_memory(output[i]);
  }

  zendnn_custom_op::zendnn_grp_embedding_bag(
      z_weight, z_indices, z_offsets, z_scale_grad_by_freq, z_algorithm,
      z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
      z_include_last_offset, z_padding_idx, z_destination); // Library call

#pragma omp parallel for
  for (int i = 0; i < num_eb_ops; i++) {
    int temp = i * 4;
    out_vec[temp + 0] = output[i];

    at::Tensor offset2bag = at::empty({});
    at::Tensor bag_size = at::empty({});
    at::Tensor max_indices = at::empty({});

    out_vec[temp + 1] = offset2bag;
    out_vec[temp + 2] = bag_size;
    out_vec[temp + 3] = max_indices;
  }

  return out_vec;
}

} // namespace ZenDNNTorch
