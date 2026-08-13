/******************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "QuantEmbedBag.hpp"
#include "EmbeddingUtils.hpp"
#include "EnvReader.hpp"
#include "Memory.hpp"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Half.h>

using namespace zendnnl::interface;

namespace zentorch {

std::tuple<int, int, int, int>
compute_quantized_embedding_dims(const torch::stable::Tensor &weight,
                                 int64_t num_bits_per_weight) {
  int dim_embedding = weight.size(1);
  const int element_size = static_cast<int>(weight.element_size());

  // Currently assumes scale and zero point to be of type BFloat16 each
  int num_dim_scale_zp =
      static_cast<int>(2 * sizeof(torch::headeronly::Half)) / element_size;

  int packed_weight_dim = dim_embedding - (num_dim_scale_zp);
  const int bits_in_1_byte = 8;
  int num_bits_per_packed_weight = element_size * bits_in_1_byte;

  // to retreive original embedding dim before int4 was packed into int32
  // packed_weight_dim * (32 / 4) (int32/int4)

  int embedding_dim =
      packed_weight_dim * (num_bits_per_packed_weight / num_bits_per_weight);

  return std::make_tuple(dim_embedding, packed_weight_dim, embedding_dim,
                         num_bits_per_packed_weight);
}

void zendnnl_quant_embedding_bag_out(
    torch::stable::Tensor &output, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &indices, const torch::stable::Tensor &offsets,
    int64_t num_bits_per_weight, c10::ScalarType output_dtype,
    bool scale_grad_by_freq, int64_t mode, bool sparse,
    const std::optional<torch::stable::Tensor> &per_sample_weights_opt,
    bool include_last_offset, int64_t padding_idx,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  zen_quant_embed_tensor_check(weight, indices, offsets);

  ZENTORCH_CHECK(
      (output_dtype == c10::ScalarType::Float ||
       output_dtype == c10::ScalarType::BFloat16),
      "zentorch_embedding_bag only supports fp32 or bf16 output types");
  ZENTORCH_CHECK(
      num_bits_per_weight == 4,
      "zentorch_embedding_bag only supports uint4 quantized weights");

  auto [dim_embedding, packed_weight_dim, embedding_dim,
        num_bits_per_packed_weight] =
      compute_quantized_embedding_dims(weight, num_bits_per_weight);

  int num_bags = offsets.size(0);
  if (include_last_offset) {
    num_bags -= 1;
  }

  unsigned long num_int4_elem =
      dim_embedding * (num_bits_per_packed_weight / num_bits_per_weight);

  unsigned long num_int4_elements_without_scale_zp =
      packed_weight_dim * (num_bits_per_packed_weight / num_bits_per_weight);

  LOG(INFO) << "Embedding matrix dimensions: " << weight.size(0) << "x"
            << dim_embedding;

  LOG(INFO) << "Int4 weight matrix dimensions: " << weight.size(0) << "x"
            << embedding_dim;

  LOG(INFO) << "Int4 weights with scale and zp dimensions: " << weight.size(0)
            << "x" << num_int4_elem;

  LOG(INFO) << "Output dimensions: " << num_bags << "x" << embedding_dim;

  const bool per_sample_weights_defined =
      per_sample_weights_opt.has_value() && per_sample_weights_opt->defined();

  const int int_env_value =
      EnvReader::getEnvVariableAsInt("USE_ZENDNN_EMBBAG_DIRECT");
  const bool use_zendnnl_direct_kernel = static_cast<bool>(int_env_value);
  if (use_zendnnl_direct_kernel) {
    // Build embag_params_t structure
    zendnnl::lowoha::embag::embag_params_t params;

    // Set data types
    params.dtypes.table = data_type_t::u4;
    params.dtypes.output = get_zendnnl_dtype(output);
    params.dtypes.indices = get_zendnnl_dtype(indices);
    params.dtypes.offsets = get_zendnnl_dtype(offsets);
    params.algo = mode_to_embag_algo(mode);

    // Set dimensions
    params.num_embeddings = weight.size(0);
    params.embedding_dim = embedding_dim;
    params.num_indices = indices.size(0);
    params.num_bags = num_bags;
    params.is_weights = per_sample_weights_defined;
    params.include_last_offset = include_last_offset;
    params.padding_idx = padding_idx;
    params.num_threads = 0; // Use default (omp_get_max_threads)
    params.fp16_scale_bias = true;
    params.dst_stride = output.stride(0);

    // Call LOWOHA embedding_bag_direct API
    status_t status = zendnnl::lowoha::embag::embedding_bag_direct(
        weight.data_ptr(), indices.data_ptr(), offsets.data_ptr(),
        per_sample_weights_defined
            ? per_sample_weights_opt->const_data_ptr<float>()
            : nullptr,
        output.data_ptr(), params);
    ZENTORCH_CHECK(status == status_t::success,
                   "LOA-operator for quant embedding bag failed.");
    return;
  }

  tensor_t table = tensor_t();
  set_zendnnl_tensor_attributes(
      weight.data_ptr(), data_type_t::u4, table, "table", false,
      {static_cast<unsigned long>(weight.size(0)),
       num_int4_elements_without_scale_zp},
      {num_int4_elements_without_scale_zp, 1}, {} /* tensor_aligned_sizes */,
      weight.numel() * static_cast<int64_t>(weight.element_size()));

  tensor_t indices_tensor = tensor_t();
  set_zendnnl_tensor_attributes(indices, indices_tensor, "indices");

  tensor_t offsets_tensor = tensor_t();
  set_zendnnl_tensor_attributes(offsets, offsets_tensor, "offsets");

  std::vector<unsigned long> output_sizes(output.sizes().begin(),
                                          output.sizes().end());
  std::vector<unsigned long> output_strides(output.strides().begin(),
                                            output.strides().end());
  int64_t output_nbytes = static_cast<int64_t>(
      output.element_size() * output_sizes[0] * output_strides[0]);
  std::vector<unsigned long> tensor_aligned_sizes = {output_sizes[0],
                                                     output_strides[0]};
  tensor_t output_tensor = tensor_t();
  set_zendnnl_tensor_attributes(
      output.data_ptr(), get_zendnnl_dtype(output), output_tensor, "output",
      false, output_sizes, output_strides, tensor_aligned_sizes, output_nbytes);

  [[maybe_unused]] tensor_t per_sample_weights_tensor = tensor_t();
  if (per_sample_weights_defined) {
    LOG(INFO) << "Using the per-sample weights tensor!";
    set_zendnnl_tensor_attributes(*per_sample_weights_opt,
                                  per_sample_weights_tensor,
                                  "per_sample_weights");
  }

  embag_context_t embedding_bag_context = embag_context_t();

  // TODO
  // Once we have fp16 scale, add that argument to this function.
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
}

torch::stable::Tensor zendnnl_quant_embedding_bag(
    const torch::stable::Tensor &weight, const torch::stable::Tensor &indices,
    const torch::stable::Tensor &offsets, int64_t num_bits_per_weight,
    c10::ScalarType output_dtype, bool scale_grad_by_freq, int64_t mode,
    bool sparse,
    const std::optional<torch::stable::Tensor> &per_sample_weights_opt,
    bool include_last_offset, int64_t padding_idx,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  auto [dim_embedding, packed_weight_dim, embedding_dim,
        num_bits_per_packed_weight] =
      compute_quantized_embedding_dims(weight, num_bits_per_weight);
  int num_bags = offsets.size(0);

  if (include_last_offset) {
    num_bags -= 1;
  }

  // new_empty instead of zeros is more efficient, since the kernel writes
  // every element of the output.
  torch::stable::Tensor output =
      torch::stable::new_empty(weight, {num_bags, embedding_dim}, output_dtype);

  zendnnl_quant_embedding_bag_out(
      output, weight, indices, offsets, num_bits_per_weight, output_dtype,
      scale_grad_by_freq, mode, sparse, per_sample_weights_opt,
      include_last_offset, padding_idx, zentorch_op_name);

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

void zendnnl_horizontal_quant_embedding_bag_group_out(
    const std::vector<torch::stable::Tensor> &outputs,
    const std::vector<torch::stable::Tensor> &weight,
    const std::vector<torch::stable::Tensor> &indices,
    const std::vector<torch::stable::Tensor> &offsets,
    int64_t num_bits_per_weight, c10::ScalarType output_dtype,
    const std::vector<int64_t> &scale_grad_by_freq,
    const std::vector<int64_t> &mode, const std::vector<int64_t> &sparse,
    const std::vector<std::optional<torch::stable::Tensor>>
        &per_sample_weights_opt,
    const std::vector<int64_t> &include_last_offset,
    const std::vector<int64_t> &padding_idx, std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  const int num_eb_ops = static_cast<int>(weight.size());

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
          dsts_vector[i] = outputs[i].data_ptr();
          params_vector[i].dtypes.table = data_type_t::u4;
          params_vector[i].dtypes.output = get_zendnnl_dtype(outputs[i]);
          params_vector[i].dtypes.indices = get_zendnnl_dtype(indices[i]);
          params_vector[i].dtypes.offsets = get_zendnnl_dtype(offsets[i]);
          params_vector[i].algo = mode_to_embag_algo(mode[i]);
          params_vector[i].num_embeddings = weight[i].size(0);

          [[maybe_unused]] auto [_unused_0, _unused_1, embedding_dim,
                                 _unused_2] =
              compute_quantized_embedding_dims(weight[i], num_bits_per_weight);

          params_vector[i].embedding_dim = embedding_dim;
          params_vector[i].num_indices = indices[i].size(0);
          int num_bags = offsets[i].size(0);
          if (include_last_offset[i]) {
            num_bags -= 1;
          }
          params_vector[i].num_bags = num_bags;
          params_vector[i].is_weights = per_sample_weights_defined;
          params_vector[i].include_last_offset =
              static_cast<bool>(include_last_offset[i]);
          params_vector[i].padding_idx = padding_idx[i];
          params_vector[i].fp16_scale_bias = true;
          params_vector[i].dst_stride = outputs[i].stride(0);
        }
      });

  status_t status = zendnnl::lowoha::embag::group_embedding_bag_direct(
      tables_vector, indices_vector, offsets_vector, weights_vector,
      dsts_vector, params_vector);

  ZENTORCH_CHECK(status == status_t::success,
                 "LOA-operator for group quant embedding bag failed.");

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
}

std::vector<torch::stable::Tensor>
zendnnl_horizontal_quant_embedding_bag_group_impl(
    const std::vector<torch::stable::Tensor> &weight,
    const std::vector<torch::stable::Tensor> &indices,
    const std::vector<torch::stable::Tensor> &offsets,
    int64_t num_bits_per_weight, c10::ScalarType output_dtype,
    const std::vector<int64_t> &scale_grad_by_freq,
    const std::vector<int64_t> &mode, const std::vector<int64_t> &sparse,
    const std::vector<std::optional<torch::stable::Tensor>>
        &per_sample_weights_opt,
    const std::vector<int64_t> &include_last_offset,
    const std::vector<int64_t> &padding_idx, std::string zentorch_op_name) {
  const int num_eb_ops = static_cast<int>(weight.size());
  std::vector<torch::stable::Tensor> outputs(num_eb_ops);

  torch::stable::parallel_for(
      0, num_eb_ops, 0, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
          int num_bags = offsets[i].size(0);
          if (include_last_offset[i]) {
            num_bags -= 1;
          }

          [[maybe_unused]] auto [_unused_0, _unused_1, embedding_dim,
                                 _unused_2] =
              compute_quantized_embedding_dims(weight[i], num_bits_per_weight);

          outputs[i] = torch::stable::new_empty(
              weight[i], {num_bags, embedding_dim}, output_dtype);
        }
      });

  zendnnl_horizontal_quant_embedding_bag_group_out(
      outputs, weight, indices, offsets, num_bits_per_weight, output_dtype,
      scale_grad_by_freq, mode, sparse, per_sample_weights_opt,
      include_last_offset, padding_idx, zentorch_op_name);

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return outputs;
}

torch::stable::Tensor zendnnl_get_packed_embedding_weight(
    const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const torch::stable::Tensor &weight_zero_points) {

  uint32_t num_eb_rows = weight.size(0);
  uint32_t num_eb_cols = weight.size(1);
  ZENTORCH_CHECK(
      (weight_scales.size(0) == num_eb_rows) &&
          (weight_zero_points.size(0) == num_eb_rows),
      "unsupported dims for embeddingbag weight, scales and zero points");
  ZENTORCH_CHECK(!(weight.scalar_type() == c10::ScalarType::QInt32 &&
                   weight_scales.scalar_type() == c10::ScalarType::Float),
                 "Weight and scales support only int32 and float dtype ");
  const torch::stable::Tensor weight_contiguous =
      torch::stable::contiguous(weight);
  const torch::stable::Tensor weight_scales_contiguous =
      torch::stable::contiguous(weight_scales);
  const torch::stable::Tensor weight_zero_points_contiguous =
      torch::stable::contiguous(weight_zero_points);

  std::vector<float> weight_scales_vec(
      weight_scales_contiguous.const_data_ptr<float>(),
      weight_scales_contiguous.const_data_ptr<float>() + num_eb_rows);
  std::vector<int32_t> weight_zero_points_vec(
      weight_zero_points_contiguous.const_data_ptr<int32_t>(),
      weight_zero_points_contiguous.const_data_ptr<int32_t>() + num_eb_rows);

  const int32_t *weight_ptr = weight_contiguous.const_data_ptr<int32_t>();

  std::vector<float> weight_bias(num_eb_rows);
  for (const auto i : c10::irange(num_eb_rows)) {
    weight_bias[i] = weight_zero_points_vec[i] * weight_scales_vec[i] * -1;
  }

  // Hard coding for int32 weights and Half dtype of scales
  const int64_t num_output_cols = num_eb_cols + 1;
  torch::stable::Tensor output_tensor = torch::stable::new_empty(
      weight_contiguous, {num_eb_rows, num_output_cols});
  int32_t *output_ptr = output_tensor.mutable_data_ptr<int32_t>();

  torch::stable::parallel_for(
      0, num_eb_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
        for (int64_t row = start_idx; row < end_idx; row++) {
          const int32_t *input_row = weight_ptr + row * num_eb_cols;
          int32_t *output_row = output_ptr + row * num_output_cols;
          auto output_row_scale_bias =
              reinterpret_cast<torch::headeronly::Half *>(output_row +
                                                          num_eb_cols);

          // Ensure weight_scale and weight_bias_half are within the range of
          // Half

          torch::headeronly::Half weight_scale = weight_scales_vec[row];

          torch::headeronly::Half weight_bias_half = weight_bias[row];

          std::memcpy(output_row_scale_bias, &weight_scale,
                      sizeof(torch::headeronly::Half)); // append weight scale
                                                        // to m/r with size of
                                                        // fp16/half
          std::memcpy(
              output_row_scale_bias + 1, &weight_bias_half,
              sizeof(torch::headeronly::Half)); // append weight bias to
                                                // m/r just after scale
                                                // with size of fp16/half
          std::memcpy(output_row, input_row, sizeof(int32_t) * (num_eb_cols));
        }
      });
  return output_tensor;
}

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_quant_embedding_bag(Tensor weight, Tensor indices, Tensor "
        "offsets,"
        " int num_bits_per_weight, ScalarType output_dtype,"
        " bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor?"
        " per_sample_weights=None, bool include_last_offset=False, int"
        " padding_idx=-1,"
        " str zentorch_op_name="
        "'zentorch::zentorch_quant_embedding_bag') -> Tensor");
  m.def("zentorch_quant_embedding_bag.out(Tensor(a!) output,"
        "Tensor weight, Tensor indices, Tensor offsets,"
        "int num_bits_per_weight, ScalarType output_dtype, bool "
        "scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? "
        "per_sample_weights=None,"
        "bool include_last_offset=False, int padding_idx=-1, str "
        "zentorch_op_name='zentorch::zentorch_quant_embedding_bag.out') -> ()");
  m.def("zentorch_horizontal_quant_embedding_bag_group(Tensor[] weight, "
        "Tensor[] indices, Tensor[] offsets, "
        " int num_bits_per_weight, ScalarType output_dtype,"
        " int[] scale_grad_by_freq, "
        "int[] mode, int[] sparse, Tensor?[] per_sample_weights, "
        "int[] include_last_offset, int[] padding_idx, str "
        "zentorch_op_name = "
        "'zentorch::zentorch_horizontal_quant_embedding_bag_group') -> "
        "Tensor[]");
  m.def("zentorch_horizontal_quant_embedding_bag_group.out("
        "Tensor(a!)[] outputs, Tensor[] weight, "
        "Tensor[] indices, Tensor[] offsets, "
        " int num_bits_per_weight, ScalarType output_dtype,"
        " int[] scale_grad_by_freq, "
        "int[] mode, int[] sparse, Tensor?[] per_sample_weights, "
        "int[] include_last_offset, int[] padding_idx, str "
        "zentorch_op_name = "
        "'zentorch::zentorch_horizontal_quant_embedding_bag_group.out') -> "
        "()");
  m.def("zentorch_get_packed_embedding_weight(Tensor weight, "
        "Tensor weight_scales, Tensor weight_zero_points) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_quant_embedding_bag",
         TORCH_BOX(&zentorch::zendnnl_quant_embedding_bag));
  m.impl("zentorch_quant_embedding_bag.out",
         TORCH_BOX(&zentorch::zendnnl_quant_embedding_bag_out));
  m.impl(
      "zentorch_horizontal_quant_embedding_bag_group",
      TORCH_BOX(&zentorch::zendnnl_horizontal_quant_embedding_bag_group_impl));
  m.impl(
      "zentorch_horizontal_quant_embedding_bag_group.out",
      TORCH_BOX(&zentorch::zendnnl_horizontal_quant_embedding_bag_group_out));
  m.impl("zentorch_get_packed_embedding_weight",
         TORCH_BOX(&zentorch::zendnnl_get_packed_embedding_weight));
}

} // namespace zentorch
