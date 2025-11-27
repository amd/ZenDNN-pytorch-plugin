/******************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EmbeddingUtils.hpp"
#include "EnvReader.hpp"

using namespace zendnn;
using namespace zendnnl::interface;

namespace zentorch {

at::Tensor zendnn_embedding_impl(const at::Tensor &weight,
                                 const at::Tensor &indices, int64_t padding_idx,
                                 bool scale_grad_by_freq, bool sparse,
                                 std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  at::Tensor cindices, output;
  memory z_weight, z_indices, z_dst;
  std::tie(cindices, output) =
      embedding_tensors_to_memory(weight, indices, z_weight, z_indices, z_dst);

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

at::Tensor zendnnl_embedding_impl(const at::Tensor &weight,
                                  const at::Tensor &indices,
                                  int64_t padding_idx, bool scale_grad_by_freq,
                                  bool sparse, std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  zen_embedding_weight_check(weight);

  int dim_embedding = weight.sizes()[1];
  int num_indices = indices.sizes()[0];

  LOG(INFO) << "Embedding matrix dimensions: " << weight.sizes()[0] << "x"
            << dim_embedding;
  LOG(INFO) << "Number of indices: " << num_indices;

  // at::detail::empty_strided_cpu instead of at::zero is more efficient
  at::Tensor output = at::detail::empty_strided_cpu(
      {num_indices, dim_embedding}, {dim_embedding, 1}, weight.options());

  tensor_t table = tensor_t();
  set_zendnnl_tensor_attributes(weight, table, "table");

  tensor_t indices_tensor = tensor_t();
  set_zendnnl_tensor_attributes(indices, indices_tensor, "indices");

  tensor_t output_tensor = tensor_t();
  set_zendnnl_tensor_attributes(output, output_tensor, "output");

  embag_context_t embedding_context = embag_context_t();
  const int64_t mode = -1; /*There is no reduction algo in embdding*/
  set_embedding_context_attributes(embedding_context, table, mode,
                                   false /*include_last_offset*/, padding_idx,
                                   false /*per_sample_weights_defined*/);

  embag_operator_t embedding_operator = embag_operator_t();
  const std::string operator_name = "embedding";
  set_embedding_operator_attributes(embedding_operator, operator_name,
                                    embedding_context, indices_tensor,
                                    output_tensor);

  LOG(INFO) << "Embedding compute in progress...";
  status_t status = embedding_operator.execute();

  ZENTORCH_CHECK(status == status_t::success, "operator ",
                 embedding_operator.get_name(), " execution failed.");

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

at::Tensor zentorch_embedding(const at::Tensor &weight,
                              const at::Tensor &indices, int64_t padding_idx,
                              bool scale_grad_by_freq, bool sparse,
                              std::string zentorch_op_name) {
  const int &library = EnvReader::getEnvVariableAsInt(
      "ZENDNN_ZENDNNL"); // 0 would represent ZenDNN and 1 would
                         // represent ZenDNNL. Default library will be ZenDNNL
  return (library == 0) ? zendnn_embedding_impl(weight, indices, padding_idx,
                                                scale_grad_by_freq, sparse,
                                                zentorch_op_name)
                        : zendnnl_embedding_impl(weight, indices, padding_idx,
                                                 scale_grad_by_freq, sparse,
                                                 zentorch_op_name);
}

std::vector<at::Tensor> zendnn_group_embedding_impl(
    at::TensorList weight, at::TensorList indices, at::IntArrayRef padding_idx,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef sparse,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  int num_embedding_ops = weight.size();

  std::vector<memory> z_weights(num_embedding_ops);
  std::vector<memory> z_indices(num_embedding_ops);
  std::vector<int32_t> z_padding_idx(num_embedding_ops);
  std::vector<int32_t> z_scale_grad_by_freq(num_embedding_ops);
  std::vector<int32_t> z_sparse(num_embedding_ops);

  std::vector<at::Tensor> temp_indices(num_embedding_ops);
  std::vector<at::Tensor> output(num_embedding_ops);
  std::vector<memory> z_destination(num_embedding_ops);

  at::parallel_for(0, num_embedding_ops, 0, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {

      std::tie(temp_indices[i], output[i]) = embedding_tensors_to_memory(
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

std::vector<at::Tensor> zendnnl_group_embedding_impl(
    at::TensorList weight, at::TensorList indices, at::IntArrayRef padding_idx,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef sparse,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  int num_embedding_ops = weight.size();
  std::vector<at::Tensor> output(num_embedding_ops);

  LOG(INFO) << "GroupEmbedding compute in progress...";

  // TODO
  // As soon as the support for optimized kernel is added at the library side,
  // This aten parallel will be removed.
  at::parallel_for(0, num_embedding_ops, 0, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {
      output[i] = zendnnl_embedding_impl(weight[i], indices[i], padding_idx[i],
                                         scale_grad_by_freq[i], sparse[i],
                                         zentorch_op_name);
    }
  });

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

std::vector<at::Tensor> zentorch_horizontal_embedding_group(
    at::TensorList weight, at::TensorList indices, at::IntArrayRef padding_idx,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef sparse,
    std::string zentorch_op_name) {

  const int &library = EnvReader::getEnvVariableAsInt(
      "ZENDNN_ZENDNNL"); // 0 would represent ZenDNN and 1 would
                         // represent ZenDNNL. Default library will be ZenDNNL
  return (library == 0)
             ? zendnn_group_embedding_impl(weight, indices, padding_idx,
                                           scale_grad_by_freq, sparse,
                                           zentorch_op_name)
             : zendnnl_group_embedding_impl(weight, indices, padding_idx,
                                            scale_grad_by_freq, sparse,
                                            zentorch_op_name);
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_embedding(Tensor weight, Tensor indices, "
        "int padding_idx=-1, bool scale_grad_by_freq=False, "
        "bool sparse=False, str "
        "zentorch_op_name='zentorch::zentorch_embedding') -> "
        "Tensor");
  m.def(
      "zentorch_horizontal_embedding_group(Tensor[] weight, Tensor[] indices, "
      "int[] padding_idx, int[] scale_grad_by_freq, "
      "int[] sparse, str zentorch_op_name = "
      "'zentorch::zentorch_horizontal_embedding_group') -> Tensor[]");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_embedding", zentorch_embedding);
  m.impl("zentorch_horizontal_embedding_group",
         zentorch_horizontal_embedding_group);
}
} // namespace zentorch
