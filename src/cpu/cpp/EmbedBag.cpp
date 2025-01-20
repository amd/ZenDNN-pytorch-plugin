/******************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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

at::Tensor
zentorch_get_packed_embedding_weight(at::Tensor &weight,
                                     at::Tensor &weight_scales,
                                     at::Tensor &weight_zero_points) {

  uint32_t num_eb_rows = weight.size(0);
  uint32_t num_eb_cols = weight.size(1);
  ZENTORCH_CHECK(
      (weight_scales.size(0) == num_eb_rows) &&
          (weight_zero_points.size(0) == num_eb_rows),
      "unsupported dims for embeddingbag weight, scales and zero points");
  ZENTORCH_CHECK(!(weight.scalar_type() == c10::ScalarType::QInt32 &&
                   weight_scales.scalar_type() == c10::ScalarType::Float),
                 "Weight and scales support only int32 and float dtype ");
  weight = weight.contiguous();
  weight_scales = weight_scales.contiguous();
  weight_zero_points = weight_zero_points.contiguous();

  std::vector<at::Half> weight_scales_vec(weight_scales.data_ptr<float>(),
                                          weight_scales.data_ptr<float>() +
                                              num_eb_rows);
  std::vector<at::Half> weight_zero_points_vec(
      weight_zero_points.data_ptr<int32_t>(),
      weight_zero_points.data_ptr<int32_t>() + num_eb_rows);

  int32_t *weight_ptr = static_cast<int32_t *>(weight.data_ptr());

  std::vector<int64_t> weight_bias(num_eb_rows);

  for (const auto i : c10::irange(num_eb_rows)) {
    weight_bias[i] = weight_zero_points_vec[i] * weight_scales_vec[i] * -1;
  }

  std::vector<int64_t> output_shape = {
      num_eb_rows,
      static_cast<std::int32_t>(
          (num_eb_cols +
           1))}; // Harding coding for int32 weights and Half dtype of scales

  size_t num_output_cols = output_shape[1];
  at::Tensor output_tensor = at::empty(output_shape, weight.options());
  int32_t *output_ptr = output_tensor.data_ptr<int32_t>();

  at::parallel_for(
      0, num_eb_rows, 1, [&](uint32_t start_idx, uint32_t end_idx) {
        for (const uint32_t row : c10::irange(start_idx, end_idx)) {
          int32_t *input_row =
              reinterpret_cast<int32_t *>(weight_ptr + row * num_eb_cols);
          int32_t *output_row =
              reinterpret_cast<int32_t *>(output_ptr + row * num_output_cols);
          auto output_row_scale_bias =
              reinterpret_cast<at::Half *>(output_row + num_eb_cols);

          // Ensure weight_scale and weight_bias_half are within the range of
          // at::Half

          at::Half weight_scale = weight_scales_vec[row];

          at::Half weight_bias_half = weight_bias[row];

          std::memcpy(output_row_scale_bias, &weight_scale,
                      sizeof(at::Half)); // append weight scale to m/r with size
                                         // of fp16/half
          std::memcpy(output_row_scale_bias + 1, &weight_bias_half,
                      sizeof(at::Half)); // append weight bias to m/r just after
                                         // scale with size of fp16/half
          std::memcpy(output_row, input_row, sizeof(int32_t) * (num_eb_cols));
        }
      });
  return output_tensor;
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
