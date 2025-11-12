/*****************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EmbeddingUtils.hpp"
#include "EnvReader.hpp"

#define ZENDNN_EMBED_BAG_THRDS 16

using namespace zendnn;
using namespace zendnnl::interface;

namespace zentorch {

at::Tensor zendnn_embeddingbag_impl(
    const at::Tensor &weight, const at::Tensor &indices,
    const at::Tensor &offsets, const bool &scale_grad_by_freq,
    const int64_t &mode, const bool &sparse,
    const c10::optional<at::Tensor> &per_sample_weights_opt,
    const bool &include_last_offset, const int64_t &padding_idx,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  at::Tensor cindices, coffsets, per_sample_weights, output;
  memory z_weight, z_indices, z_offsets, z_per_sample_weights_opt, z_dst;
  algorithm z_algorithm;
  std::tie(cindices, coffsets, per_sample_weights, output) =
      embeddingbag_tensors_to_memory(
          weight, indices, offsets, per_sample_weights_opt, mode, output,
          z_weight, z_indices, z_offsets, z_per_sample_weights_opt, z_algorithm,
          z_dst, include_last_offset);
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
        z_per_sample_weights_opt.get_desc(), z_dst.get_desc(), padding_idx);
    pd = embedding_bag::primitive_desc(pdesc, op_attr,
                                       utils::engine::cpu_engine());
    LOG(INFO) << "EmbeddingBag compute in progress...";
    embedding_bag(pd).execute(utils::stream::default_stream(),
                              {{ZENDNN_ARG_SRC_0, z_weight},
                               {ZENDNN_ARG_SRC_1, z_indices},
                               {ZENDNN_ARG_SRC_2, z_offsets},
                               {ZENDNN_ARG_SRC_3, z_per_sample_weights_opt},
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

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

at::Tensor
zendnnl_embeddingbag_impl(const at::Tensor &weight, const at::Tensor &indices,
                          const at::Tensor &offsets, bool scale_grad_by_freq,
                          int64_t mode, bool sparse,
                          c10::optional<at::Tensor> per_sample_weights_opt,
                          bool include_last_offset, int64_t padding_idx,
                          std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  zen_embedding_weight_check(weight);

  int dim_embedding = weight.sizes()[1];
  int num_bags = offsets.sizes()[0];

  if (include_last_offset == true) {
    num_bags -= 1;
  }

  LOG(INFO) << "Embedding matrix dimensions: " << weight.sizes()[0] << "x"
            << dim_embedding;
  LOG(INFO) << "Number of embedding bags: " << num_bags;

  at::Tensor output = at::detail::empty_strided_cpu(
      {num_bags, dim_embedding}, {dim_embedding, 1}, weight.options());

  c10::MaybeOwned<at::Tensor> per_sample_weights_opt_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  [[maybe_unused]] const at::Tensor &per_sample_weights =
      *per_sample_weights_opt_maybe_owned;

  auto per_sample_weights_defined = per_sample_weights.defined();

  tensor_t table = tensor_t();
  tensor_t output_tensor = tensor_t();
  tensor_t indices_tensor = tensor_t();
  tensor_t offsets_tensor = tensor_t();
  [[maybe_unused]] tensor_t per_sample_weights_tensor = tensor_t();

  std::vector<TensorStruct> tensor_structs = {
      {indices, indices_tensor, "indices"},
      {offsets, offsets_tensor, "offsets"},
      {weight, table, "table"},
      {output, output_tensor, "output"}};

  [[maybe_unused]] const std::string_view tensor_name = "per_sample_weights";
  if (per_sample_weights_defined) {
    LOG(INFO) << "Using the per-sample weights tensor!";
    tensor_structs.emplace_back(per_sample_weights, per_sample_weights_tensor,
                                tensor_name);
  }

  create_tensors_for_zendnnl(tensor_structs);

  embag_context_t embedding_bag_context = embag_context_t();

  set_embedding_context_attributes(embedding_bag_context, table, mode,
                                   include_last_offset, padding_idx,
                                   per_sample_weights_defined);

  // define embedding bag operator
  embag_operator_t embedding_bag_operator = embag_operator_t();
  const std::string operator_name = "embedding_bag";
  if (per_sample_weights_defined) {
    set_embedding_operator_attributes(embedding_bag_operator, operator_name,
                                      embedding_bag_context, indices_tensor,
                                      output_tensor, offsets_tensor,
                                      per_sample_weights_tensor);
  } else {
    set_embedding_operator_attributes(embedding_bag_operator, operator_name,
                                      embedding_bag_context, indices_tensor,
                                      output_tensor, offsets_tensor);
  }

  LOG(INFO) << "EmbeddingBag compute in progress...";
  status_t status = embedding_bag_operator.execute();

  ZENTORCH_CHECK(status == status_t::success, "operator ",
                 embedding_bag_operator.get_name(), " execution failed.");

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

at::Tensor
zentorch_embedding_bag(const at::Tensor &weight, const at::Tensor &indices,
                       const at::Tensor &offsets, bool scale_grad_by_freq,
                       int64_t mode, bool sparse,
                       c10::optional<at::Tensor> per_sample_weights_opt,
                       bool include_last_offset, int64_t padding_idx,
                       std::string zentorch_op_name) {

  const int &library = EnvReader::getEnvVariableAsInt(
      "ZENDNN_ZENDNNL"); // 0 would represent ZenDNN and 1 would
                         // represent ZenDNNL. Default library will be ZenDNN

  return (library == 0)
             ? zendnn_embeddingbag_impl(
                   weight, indices, offsets, scale_grad_by_freq, mode, sparse,
                   per_sample_weights_opt, include_last_offset, padding_idx,
                   zentorch_op_name)
             : zendnnl_embeddingbag_impl(
                   weight, indices, offsets, scale_grad_by_freq, mode, sparse,
                   per_sample_weights_opt, include_last_offset, padding_idx,
                   zentorch_op_name);
}

std::vector<at::Tensor> zendnn_horizontal_embedding_bag_group_impl(
    at::TensorList weight, at::TensorList indices, at::TensorList offsets,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef mode,
    at::IntArrayRef sparse,
    c10::List<c10::optional<at::Tensor>> per_sample_weights_opt,
    at::IntArrayRef include_last_offset, at::IntArrayRef padding_idx,
    std::string zentorch_op_name) {
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

  // If TORCH_CHECK() fails, then the pragma omp parallel for cannot be used
  // we will use at::parallel_for from pytorch instead
  at::parallel_for(0, num_eb_ops, 0, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {
      at::Tensor per_sample_weights;
      std::tie(temp_indices[i], temp_offsets[i], per_sample_weights,
               output[i]) =
          embeddingbag_tensors_to_memory(
              weight[i], indices[i], offsets[i], per_sample_weights_opt[i],
              mode[i], output[i], z_weight[i], z_indices[i], z_offsets[i],
              z_per_sample_weights_opt[i], z_algorithm[i], z_destination[i],
              include_last_offset[i]);

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

  return output;
}

std::vector<at::Tensor> zendnnl_horizontal_embedding_bag_group_impl(
    at::TensorList weight, at::TensorList indices, at::TensorList offsets,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef mode,
    at::IntArrayRef sparse,
    c10::List<c10::optional<at::Tensor>> per_sample_weights_opt,
    at::IntArrayRef include_last_offset, at::IntArrayRef padding_idx,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  int num_eb_ops = weight.size();
  std::vector<at::Tensor> output(num_eb_ops);

  at::parallel_for(0, num_eb_ops, 0, [&](int64_t start, int64_t end) {
    for (int i = 0; i < num_eb_ops; i++) {
      output[i] = zentorch_embedding_bag(
          weight[i], indices[i], offsets[i], scale_grad_by_freq[i], mode[i],
          sparse[i], per_sample_weights_opt[i], include_last_offset[i],
          padding_idx[i], zentorch_op_name);
    }
  });

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

std::vector<at::Tensor> zentorch_horizontal_embedding_bag_group(
    at::TensorList weight, at::TensorList indices, at::TensorList offsets,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef mode,
    at::IntArrayRef sparse,
    c10::List<c10::optional<at::Tensor>> per_sample_weights_opt,
    at::IntArrayRef include_last_offset, at::IntArrayRef padding_idx,
    std::string zentorch_op_name) {

  const int &library = EnvReader::getEnvVariableAsInt(
      "ZENDNN_ZENDNNL"); // 0 would represent ZenDNN and 1 would
                         // represent ZenDNNL. Default library will be ZenDNNL
  return (library == 0)
             ? zendnn_horizontal_embedding_bag_group_impl(
                   weight, indices, offsets, scale_grad_by_freq, mode, sparse,
                   per_sample_weights_opt, include_last_offset, padding_idx,
                   zentorch_op_name)
             : zendnnl_horizontal_embedding_bag_group_impl(
                   weight, indices, offsets, scale_grad_by_freq, mode, sparse,
                   per_sample_weights_opt, include_last_offset, padding_idx,
                   zentorch_op_name);
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, "
        "bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? "
        "per_sample_weights=None, bool include_last_offset=False, int "
        "padding_idx=-1, str "
        "zentorch_op_name='zentorch::zentorch_embedding_bag') -> Tensor");
  m.def("zentorch_horizontal_embedding_bag_group(Tensor[] weight, "
        "Tensor[] indices, Tensor[] offsets, int[] scale_grad_by_freq, "
        "int[] mode, int[] sparse, Tensor?[] per_sample_weights, "
        "int[] include_last_offset, int[] padding_idx, str "
        "zentorch_op_name = "
        "'zentorch::zentorch_horizontal_embedding_bag_group') -> Tensor[]");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_embedding_bag", zentorch_embedding_bag);
  m.impl("zentorch_horizontal_embedding_bag_group",
         zentorch_horizontal_embedding_bag_group);
}
} // namespace zentorch
