/******************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "Embedding.hpp"
#include "EmbeddingUtils.hpp"
#include "EnvReader.hpp"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

using namespace zendnnl::interface;

namespace zentorch {

// Core compute: writes the gathered rows into `output`. Shared by the
// allocating impl and the `.out` variant.
void zendnnl_embedding_impl(const torch::stable::Tensor &weight,
                            const torch::stable::Tensor &indices,
                            int64_t padding_idx, bool scale_grad_by_freq,
                            bool sparse, std::string zentorch_op_name,
                            torch::stable::Tensor &output) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  zen_embedding_weight_check(weight);
  check_embedding_out_tensor(weight, indices, output);

  LOG(INFO) << "Embedding matrix dimensions: " << weight.size(0) << "x"
            << weight.size(1);
  LOG(INFO) << "Number of indices: " << indices.size(0);

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
  set_embedding_operator_attributes(embedding_operator, zentorch_op_name,
                                    embedding_context, indices_tensor,
                                    output_tensor);

  LOG(INFO) << "Embedding compute in progress...";
  status_t status = embedding_operator.execute();

  ZENTORCH_CHECK(status == status_t::success, "operator ",
                 embedding_operator.get_name(), " execution failed.");

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
}

torch::stable::Tensor zentorch_embedding(const torch::stable::Tensor &weight,
                                         const torch::stable::Tensor &indices,
                                         int64_t padding_idx,
                                         bool scale_grad_by_freq, bool sparse,
                                         std::string zentorch_op_name) {
  torch::stable::Tensor output =
      create_embedding_output_tensor(weight, indices);
  zendnnl_embedding_impl(weight, indices, padding_idx, scale_grad_by_freq,
                         sparse, zentorch_op_name, output);
  return output;
}

static std::vector<torch::stable::Tensor>
zendnnl_group_embedding_impl(const std::vector<torch::stable::Tensor> &weight,
                             const std::vector<torch::stable::Tensor> &indices,
                             const std::vector<int64_t> &padding_idx,
                             const std::vector<int64_t> &scale_grad_by_freq,
                             const std::vector<int64_t> &sparse,
                             std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  const int num_embedding_ops = static_cast<int>(weight.size());
  std::vector<torch::stable::Tensor> output(num_embedding_ops);

  LOG(INFO) << "GroupEmbedding compute in progress...";

  // TODO
  // As soon as the support for optimized kernel is added at the library side,
  // This parallel loop will be removed.
  torch::stable::parallel_for(
      0, num_embedding_ops, 0, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
          output[i] = zentorch_embedding(
              weight[i], indices[i], padding_idx[i],
              static_cast<bool>(scale_grad_by_freq[i]),
              static_cast<bool>(sparse[i]), zentorch_op_name);
        }
      });

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

std::vector<torch::stable::Tensor> zentorch_horizontal_embedding_group(
    std::vector<torch::stable::Tensor> weight,
    std::vector<torch::stable::Tensor> indices,
    std::vector<int64_t> padding_idx, std::vector<int64_t> scale_grad_by_freq,
    std::vector<int64_t> sparse, std::string zentorch_op_name) {

  return zendnnl_group_embedding_impl(weight, indices, padding_idx,
                                      scale_grad_by_freq, sparse,
                                      zentorch_op_name);
}

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_embedding(Tensor weight, Tensor indices, "
        "int padding_idx=-1, bool scale_grad_by_freq=False, "
        "bool sparse=False, str "
        "zentorch_op_name='zentorch::zentorch_embedding') -> "
        "Tensor");
  m.def("zentorch_embedding.out(Tensor weight, Tensor indices, "
        "int padding_idx=-1, bool scale_grad_by_freq=False, "
        "bool sparse=False, str "
        "zentorch_op_name='zentorch::zentorch_embedding', "
        "*, Tensor(a!) out) -> ()");
  m.def(
      "zentorch_horizontal_embedding_group(Tensor[] weight, Tensor[] indices, "
      "int[] padding_idx, int[] scale_grad_by_freq, "
      "int[] sparse, str zentorch_op_name = "
      "'zentorch::zentorch_horizontal_embedding_group') -> Tensor[]");
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_embedding", TORCH_BOX(&zentorch::zentorch_embedding));
  m.impl("zentorch_embedding.out",
         TORCH_BOX(&zentorch::zendnnl_embedding_impl));
  m.impl("zentorch_horizontal_embedding_group",
         TORCH_BOX(&zentorch::zentorch_horizontal_embedding_group));
}
} // namespace zentorch
