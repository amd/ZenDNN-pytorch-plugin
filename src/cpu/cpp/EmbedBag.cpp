/******************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EmbedUtils.hpp"
#include "Memory.hpp"

#include <ATen/ParallelOpenMP.h>
#define ZENDNN_EMBED_BAG_THRDS 16

using namespace zendnn;

namespace zentorch {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
zentorch_embedding_bag_impl(
    const at::Tensor &weight, const at::Tensor &indices,
    const at::Tensor &offsets, const bool &scale_grad_by_freq,
    const int64_t &mode, const bool &sparse,
    const c10::optional<at::Tensor> &per_sample_weights_opt,
    const bool &include_last_offset, const int64_t &padding_idx,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  at::Tensor cindices, coffsets, per_sample_weights, output;
  memory z_weight, z_indices, z_offsets, z_weights, z_dst;
  algorithm z_algorithm;

  std::tie(cindices, coffsets, per_sample_weights, output) =
      eb_tensors_to_memory(weight, indices, offsets, per_sample_weights_opt,
                           mode, output, z_weight, z_indices, z_offsets,
                           z_weights, z_algorithm, z_dst, include_last_offset);

  embedding_bag::desc pdesc;
  embedding_bag::primitive_desc pd;

  zendnn::primitive_attr op_attr;
  op_attr.set_plugin_op_name(zentorch_op_name);

  if (per_sample_weights.defined()) {
    LOG(INFO) << "Using the per-sample weights tensor!";
    // declare embedding bag primitive
    pdesc = embedding_bag::desc(
        prop_kind::forward_inference, z_algorithm, ZENDNN_EMBED_BAG_THRDS,
        z_weight.get_desc(), z_indices.get_desc(), z_offsets.get_desc(),
        z_weights.get_desc(), z_dst.get_desc(), padding_idx);

    pd = embedding_bag::primitive_desc(pdesc, op_attr,
                                       utils::engine::cpu_engine());
    LOG(INFO) << "EmbeddingBag compute in progress...";
    embedding_bag(pd).execute(utils::stream::default_stream(),
                              {{ZENDNN_ARG_SRC_0, z_weight},
                               {ZENDNN_ARG_SRC_1, z_indices},
                               {ZENDNN_ARG_SRC_2, z_offsets},
                               {ZENDNN_ARG_SRC_3, z_weights},
                               {ZENDNN_ARG_DST, z_dst}});
  } else {
    LOG(INFO) << "Per-sample weights is not defined!";
    // declare embedding bag primitive
    pdesc = embedding_bag::desc(prop_kind::forward_inference, z_algorithm,
                                ZENDNN_EMBED_BAG_THRDS, z_weight.get_desc(),
                                z_indices.get_desc(), z_offsets.get_desc(),
                                z_dst.get_desc(), padding_idx);

    pd = embedding_bag::primitive_desc(pdesc, op_attr,
                                       utils::engine::cpu_engine());
    LOG(INFO) << "EmbeddingBag compute in progress...";
    embedding_bag(pd).execute(utils::stream::default_stream(),
                              {{ZENDNN_ARG_SRC_0, z_weight},
                               {ZENDNN_ARG_SRC_1, z_indices},
                               {ZENDNN_ARG_SRC_2, z_offsets},
                               {ZENDNN_ARG_DST, z_dst}});
  }

  at::Tensor offset2bag = at::empty({});
  at::Tensor bag_size = at::empty({});
  at::Tensor max_indices = at::empty({});

  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> out;
  out = std::make_tuple(std::move(output), std::move(offset2bag),
                        std::move(bag_size), std::move(max_indices));

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return out;
}

std::vector<at::Tensor> zentorch_horizontal_embedding_bag_group(
    const at::TensorList &weight, const at::TensorList &indices,
    const at::TensorList &offsets, const at::IntArrayRef &scale_grad_by_freq,
    const at::IntArrayRef &mode, const at::IntArrayRef &sparse,
    const c10::List<c10::optional<at::Tensor>> &per_sample_weights_opt,
    const at::IntArrayRef &include_last_offset,
    const at::IntArrayRef &padding_idx, std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

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

  // If TORCH_CHECK() fails, then the pragma omp parallel for cannot be used
  // we will use at::parallel_for from pytorch instead
  at::parallel_for(0, num_eb_ops, 0, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {

      at::Tensor per_sample_weights;

      std::tie(temp_indices[i], temp_offsets[i], per_sample_weights,
               output[i]) =
          eb_tensors_to_memory(weight[i], indices[i], offsets[i],
                               per_sample_weights_opt[i], mode[i], output[i],
                               z_weight[i], z_indices[i], z_offsets[i],
                               z_per_sample_weights_opt[i], z_algorithm[i],
                               z_destination[i], include_last_offset[i]);

      z_padding_idx[i] = padding_idx[i];
      z_scale_grad_by_freq[i] = scale_grad_by_freq[i];
      z_include_last_offset[i] = include_last_offset[i];
      z_sparse[i] = sparse[i];

      if (per_sample_weights.defined()) {
        z_per_sample_weights_defined[i] = 1;
      } else {
        z_per_sample_weights_defined[i] = 0;
      }
    }
  });

  LOG(INFO) << "GroupEmbeddingBag compute in progress...";
  zendnn_custom_op::zendnn_grp_embedding_bag(
      z_weight, z_indices, z_offsets, z_scale_grad_by_freq, z_algorithm,
      z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
      z_include_last_offset, z_padding_idx, z_destination,
      zentorch_op_name.c_str()); // Library call

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

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return out_vec;
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, "
        "bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? "
        "per_sample_weights=None, bool include_last_offset=False, int "
        "padding_idx=-1, str "
        "zentorch_op_name='zentorch::zentorch_embedding_bag') -> "
        "(Tensor, Tensor, Tensor, Tensor)");
  m.def("zentorch_horizontal_embedding_bag_group(Tensor[] weight, "
        "Tensor[] indices, Tensor[] offsets, int[] scale_grad_by_freq, "
        "int[] mode, int[] sparse, Tensor?[] per_sample_weights, "
        "int[] include_last_offset, int[] padding_idx, str "
        "zentorch_op_name = "
        "'zentorch::zentorch_horizontal_embedding_bag_group') -> Tensor[]");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_embedding_bag", zentorch_embedding_bag_impl);
  m.impl("zentorch_horizontal_embedding_bag_group",
         zentorch_horizontal_embedding_bag_group);
}

} // namespace zentorch
