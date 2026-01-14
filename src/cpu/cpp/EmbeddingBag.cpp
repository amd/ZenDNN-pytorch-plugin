/*****************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EmbeddingUtils.hpp"
#include "EnvReader.hpp"

using namespace zendnnl::interface;

namespace zentorch {
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

  tensor_t table = tensor_t();
  set_zendnnl_tensor_attributes(weight, table, "table");

  tensor_t indices_tensor = tensor_t();
  set_zendnnl_tensor_attributes(indices, indices_tensor, "indices");

  tensor_t offsets_tensor = tensor_t();
  set_zendnnl_tensor_attributes(offsets, offsets_tensor, "offsets");

  at::Tensor output = at::detail::empty_strided_cpu(
      {num_bags, dim_embedding}, {dim_embedding, 1}, weight.options());
  tensor_t output_tensor = tensor_t();
  set_zendnnl_tensor_attributes(output, output_tensor, "output");

  c10::MaybeOwned<at::Tensor> per_sample_weights_opt_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  [[maybe_unused]] const at::Tensor &per_sample_weights =
      *per_sample_weights_opt_maybe_owned;
  [[maybe_unused]] tensor_t per_sample_weights_tensor = tensor_t();

  auto per_sample_weights_defined = per_sample_weights.defined();
  if (per_sample_weights_defined) {
    LOG(INFO) << "Using the per-sample weights tensor!";
    set_zendnnl_tensor_attributes(per_sample_weights, per_sample_weights_tensor,
                                  "per_sample_weights");
  }

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
  return zendnnl_embeddingbag_impl(weight, indices, offsets, scale_grad_by_freq,
                                   mode, sparse, per_sample_weights_opt,
                                   include_last_offset, padding_idx,
                                   zentorch_op_name);
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
    for (int i = start; i < end; i++) {
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

  return zendnnl_horizontal_embedding_bag_group_impl(
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
