/*****************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EmbeddingUtils.hpp"
#include "EnvReader.hpp"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

using namespace zendnnl::interface;

namespace zentorch {
torch::stable::Tensor zentorch_embedding_bag(
    const torch::stable::Tensor &weight, const torch::stable::Tensor &indices,
    const torch::stable::Tensor &offsets, bool scale_grad_by_freq, int64_t mode,
    bool sparse,
    const std::optional<torch::stable::Tensor> &per_sample_weights_opt,
    bool include_last_offset, int64_t padding_idx,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  zen_embedding_weight_check(weight);

  int dim_embedding = weight.size(1);
  int num_bags = offsets.size(0);

  if (include_last_offset) {
    num_bags -= 1;
  }

  LOG(INFO) << "Embedding matrix dimensions: " << weight.size(0) << "x"
            << dim_embedding;
  LOG(INFO) << "Number of embedding bags: " << num_bags;

  const bool per_sample_weights_defined =
      per_sample_weights_opt.has_value() && per_sample_weights_opt->defined();

  torch::stable::Tensor output =
      torch::stable::new_empty(weight, {num_bags, dim_embedding});

  const int int_env_value =
      EnvReader::getEnvVariableAsInt("USE_ZENDNN_EMBBAG_DIRECT");
  const bool use_zendnnl_direct_kernel = static_cast<bool>(int_env_value);
  if (use_zendnnl_direct_kernel) {
    // Build embag_params_t structure
    zendnnl::lowoha::embag::embag_params_t params;

    // Set data types
    params.dtypes.table = get_zendnnl_dtype(weight);
    params.dtypes.output = get_zendnnl_dtype(output);
    params.dtypes.indices = get_zendnnl_dtype(indices);
    params.dtypes.offsets = get_zendnnl_dtype(offsets);
    params.algo = mode_to_embag_algo(mode);

    // Set dimensions
    params.num_embeddings = weight.size(0);
    params.embedding_dim = weight.size(1);
    params.num_indices = indices.size(0);
    params.num_bags = num_bags;
    params.is_weights = per_sample_weights_defined;
    params.include_last_offset = include_last_offset;
    params.padding_idx = padding_idx;
    params.num_threads = 0; // Use default (omp_get_max_threads)
    params.fp16_scale_bias = true;
    params.dst_stride = dim_embedding;

    // Call LOWOHA embedding_bag_direct API
    status_t status = zendnnl::lowoha::embag::embedding_bag_direct(
        weight.data_ptr(), indices.data_ptr(), offsets.data_ptr(),
        per_sample_weights_defined
            ? per_sample_weights_opt->const_data_ptr<float>()
            : nullptr,
        output.data_ptr(), params);
    ZENTORCH_CHECK(status == status_t::success,
                   "LOA-operator for quant embedding bag failed.");
    return output;
  }

  tensor_t table = tensor_t();
  set_zendnnl_tensor_attributes(weight, table, "table");

  tensor_t indices_tensor = tensor_t();
  set_zendnnl_tensor_attributes(indices, indices_tensor, "indices");

  tensor_t offsets_tensor = tensor_t();
  set_zendnnl_tensor_attributes(offsets, offsets_tensor, "offsets");

  tensor_t output_tensor = tensor_t();
  set_zendnnl_tensor_attributes(output, output_tensor, "output");

  [[maybe_unused]] tensor_t per_sample_weights_tensor = tensor_t();

  if (per_sample_weights_defined) {
    LOG(INFO) << "Using the per-sample weights tensor!";
    set_zendnnl_tensor_attributes(*per_sample_weights_opt,
                                  per_sample_weights_tensor,
                                  "per_sample_weights");
  }

  embag_context_t embedding_bag_context = embag_context_t();

  set_embedding_context_attributes(embedding_bag_context, table, mode,
                                   include_last_offset, padding_idx,
                                   per_sample_weights_defined);

  // define embedding bag operator
  embag_operator_t embedding_bag_operator = embag_operator_t();
  if (per_sample_weights_defined) {
    set_embedding_operator_attributes(embedding_bag_operator, zentorch_op_name,
                                      embedding_bag_context, indices_tensor,
                                      output_tensor, offsets_tensor,
                                      per_sample_weights_tensor);
  } else {
    set_embedding_operator_attributes(embedding_bag_operator, zentorch_op_name,
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

std::vector<torch::stable::Tensor> zentorch_horizontal_embedding_bag_group(
    std::vector<torch::stable::Tensor> weight,
    std::vector<torch::stable::Tensor> indices,
    std::vector<torch::stable::Tensor> offsets,
    std::vector<int64_t> scale_grad_by_freq, std::vector<int64_t> mode,
    std::vector<int64_t> sparse,
    std::vector<std::optional<torch::stable::Tensor>> per_sample_weights_opt,
    std::vector<int64_t> include_last_offset, std::vector<int64_t> padding_idx,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  const int num_eb_ops = static_cast<int>(weight.size());
  std::vector<torch::stable::Tensor> output(num_eb_ops);

  std::vector<const void *> tables_vector(num_eb_ops);
  std::vector<const void *> indices_vector(num_eb_ops);
  std::vector<const void *> offsets_vector(num_eb_ops);
  std::vector<const float *> weights_vector(num_eb_ops);
  std::vector<void *> dsts_vector(num_eb_ops);
  std::vector<zendnnl::lowoha::embag::embag_params_t> params_vector(num_eb_ops);

  torch::stable::parallel_for(
      0, num_eb_ops, 0, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
          const bool per_sample_weights_defined =
              per_sample_weights_opt[i].has_value() &&
              per_sample_weights_opt[i]->defined();
          tables_vector[i] = weight[i].data_ptr();
          indices_vector[i] = indices[i].data_ptr();
          offsets_vector[i] = offsets[i].data_ptr();
          weights_vector[i] =
              per_sample_weights_defined
                  ? per_sample_weights_opt[i]->const_data_ptr<float>()
                  : nullptr;
          int num_bags = offsets[i].size(0);
          if (include_last_offset[i]) {
            num_bags -= 1;
          }
          int dim_embedding = weight[i].size(1);
          output[i] =
              torch::stable::new_empty(weight[i], {num_bags, dim_embedding});
          dsts_vector[i] = output[i].data_ptr();
          params_vector[i].dtypes.table = get_zendnnl_dtype(weight[i]);
          params_vector[i].dtypes.output = get_zendnnl_dtype(output[i]);
          params_vector[i].dtypes.indices = get_zendnnl_dtype(indices[i]);
          params_vector[i].dtypes.offsets = get_zendnnl_dtype(offsets[i]);
          params_vector[i].algo = mode_to_embag_algo(mode[i]);
          params_vector[i].num_embeddings = weight[i].size(0);
          params_vector[i].embedding_dim = dim_embedding;
          params_vector[i].num_indices = indices[i].size(0);
          params_vector[i].num_bags = num_bags;
          params_vector[i].is_weights = per_sample_weights_defined;
          params_vector[i].include_last_offset = include_last_offset[i];
          params_vector[i].padding_idx = padding_idx[i];
          params_vector[i].fp16_scale_bias = true;
          params_vector[i].dst_stride = dim_embedding;
        }
      });

  status_t status = zendnnl::lowoha::embag::group_embedding_bag_direct(
      tables_vector, indices_vector, offsets_vector, weights_vector,
      dsts_vector, params_vector);

  ZENTORCH_CHECK(status == status_t::success,
                 "LOA-operator for group embedding bag failed.");

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
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

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_embedding_bag",
         TORCH_BOX(&zentorch::zentorch_embedding_bag));
  m.impl("zentorch_horizontal_embedding_bag_group",
         TORCH_BOX(&zentorch::zentorch_horizontal_embedding_bag_group));
}
} // namespace zentorch
