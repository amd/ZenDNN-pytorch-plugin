/******************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "QuantEmbedBag.hpp"
#include "EmbeddingUtils.hpp"
#include "EnvReader.hpp"
#include "Memory.hpp"
#include "Ops.hpp"

#include <ATen/ParallelOpenMP.h>

using namespace zendnnl::interface;

namespace zentorch {

std::tuple<int, int, int, int>
compute_quantized_embedding_dims(const at::Tensor &weight,
                                 int64_t num_bits_per_weight) {
  int dim_embedding = weight.sizes()[1];

  // Currently assumes scale and zero point to be of type BFloat16 each
  int num_dim_scale_zp =
      (sizeof(at::Half) + sizeof(at::Half)) / weight.element_size();

  int packed_weight_dim = dim_embedding - (num_dim_scale_zp);
  const int64_t bits_in_1_byte = 8;
  int num_bits_per_packed_weight = weight.element_size() * bits_in_1_byte;

  // to retreive original embedding dim before int4 was packed into int32
  // packed_weight_dim * (32 / 4) (int32/int4)

  int embedding_dim =
      packed_weight_dim * (num_bits_per_packed_weight / num_bits_per_weight);

  return std::make_tuple(dim_embedding, packed_weight_dim, embedding_dim,
                         num_bits_per_packed_weight);
}

void zendnnl_quant_embedding_bag_out(
    const at::Tensor &output, const at::Tensor &weight,
    const at::Tensor &indices, const at::Tensor &offsets,
    int64_t num_bits_per_weight, c10::ScalarType output_dtype,
    bool scale_grad_by_freq, int64_t mode, bool sparse,
    c10::optional<at::Tensor> per_sample_weights_opt, bool include_last_offset,
    int64_t padding_idx, std::string zentorch_op_name) {

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

  int num_bags = offsets.sizes()[0];
  if (include_last_offset) {
    num_bags -= 1;
  }

  unsigned long num_int4_elem =
      dim_embedding * (num_bits_per_packed_weight / num_bits_per_weight);

  unsigned long num_int4_elements_without_scale_zp =
      packed_weight_dim * (num_bits_per_packed_weight / num_bits_per_weight);

  LOG(INFO) << "Embedding matrix dimensions: " << weight.sizes()[0] << "x"
            << dim_embedding;

  LOG(INFO) << "Int4 weight matrix dimensions: " << weight.sizes()[0] << "x"
            << embedding_dim;

  LOG(INFO) << "Int4 weights with scale and zp dimensions: "
            << weight.sizes()[0] << "x" << num_int4_elem;

  LOG(INFO) << "Output dimensions: " << num_bags << "x" << embedding_dim;

  c10::MaybeOwned<at::Tensor> per_sample_weights_opt_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  [[maybe_unused]] const at::Tensor &per_sample_weights =
      *per_sample_weights_opt_maybe_owned;
  const auto per_sample_weights_defined = per_sample_weights.defined();

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
    params.num_embeddings = weight.sizes()[0];
    params.embedding_dim = embedding_dim;
    params.num_indices = indices.sizes()[0];
    params.num_bags =
        include_last_offset ? offsets.sizes()[0] - 1 : offsets.sizes()[0];
    params.is_weights = per_sample_weights_defined;
    params.include_last_offset = include_last_offset;
    params.padding_idx = padding_idx;
    params.num_threads = 0; // Use default (omp_get_max_threads)
    params.fp16_scale_bias = true;
    params.dst_stride = output.strides()[0];

    // Call LOWOHA embedding_bag_direct API
    status_t status = zendnnl::lowoha::embag::embedding_bag_direct(
        weight.data_ptr(), indices.data_ptr(), offsets.data_ptr(),
        per_sample_weights_defined ? per_sample_weights.data_ptr<float>()
                                   : nullptr,
        output.data_ptr(), params);
    ZENTORCH_CHECK(status == status_t::success,
                   "LOA-operator for quant embedding bag failed.");
    return;
  }

  tensor_t table = tensor_t();
  set_zendnnl_tensor_attributes(weight.data_ptr(), data_type_t::u4, table,
                                "table", false,
                                {static_cast<unsigned long>(weight.sizes()[0]),
                                 num_int4_elements_without_scale_zp},
                                {num_int4_elements_without_scale_zp, 1},
                                {} /* tensor_aligned_sizes */, weight.nbytes());

  tensor_t indices_tensor = tensor_t();
  set_zendnnl_tensor_attributes(indices, indices_tensor, "indices");

  tensor_t offsets_tensor = tensor_t();
  set_zendnnl_tensor_attributes(offsets, offsets_tensor, "offsets");

  std::vector<unsigned long> output_sizes(output.sizes().begin(),
                                          output.sizes().end());
  std::vector<unsigned long> output_strides(output.strides().begin(),
                                            output.strides().end());
  int64_t output_nbytes = c10::elementSize(output.scalar_type()) *
                          output_sizes[0] * output_strides[0];
  std::vector<unsigned long> tensor_aligned_sizes = {output_sizes[0],
                                                     output_strides[0]};
  tensor_t output_tensor = tensor_t();
  set_zendnnl_tensor_attributes(
      output.data_ptr(), get_zendnnl_dtype(output), output_tensor, "output",
      false, output_sizes, output_strides, tensor_aligned_sizes, output_nbytes);

  [[maybe_unused]] tensor_t per_sample_weights_tensor = tensor_t();
  if (per_sample_weights_defined) {
    LOG(INFO) << "Using the per-sample weights tensor!";
    set_zendnnl_tensor_attributes(per_sample_weights, per_sample_weights_tensor,
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

at::Tensor zendnnl_quant_embedding_bag(
    const at::Tensor &weight, const at::Tensor &indices,
    const at::Tensor &offsets, int64_t num_bits_per_weight,
    c10::ScalarType output_dtype, bool scale_grad_by_freq, int64_t mode,
    bool sparse, c10::optional<at::Tensor> per_sample_weights_opt,
    bool include_last_offset, int64_t padding_idx,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  auto [dim_embedding, packed_weight_dim, embedding_dim,
        num_bits_per_packed_weight] =
      compute_quantized_embedding_dims(weight, num_bits_per_weight);
  int num_bags = offsets.sizes()[0];

  if (include_last_offset) {
    num_bags -= 1;
  }

  // at::detail::empty_strided_cpu instead of at::zero is more efficient
  at::Tensor output = at::detail::empty_strided_cpu(
      {num_bags, embedding_dim}, {embedding_dim, 1}, output_dtype);

  zendnnl_quant_embedding_bag_out(
      output, weight, indices, offsets, num_bits_per_weight, output_dtype,
      scale_grad_by_freq, mode, sparse, per_sample_weights_opt,
      include_last_offset, padding_idx, zentorch_op_name);

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

void zendnnl_horizontal_quant_embedding_bag_group_out(
    at::TensorList outputs, at::TensorList weight, at::TensorList indices,
    at::TensorList offsets, int64_t num_bits_per_weight,
    c10::ScalarType output_dtype, at::IntArrayRef scale_grad_by_freq,
    at::IntArrayRef mode, at::IntArrayRef sparse,
    c10::List<c10::optional<at::Tensor>> per_sample_weights_opt,
    at::IntArrayRef include_last_offset, at::IntArrayRef padding_idx,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  int num_eb_ops = weight.size();
  std::vector<c10::MaybeOwned<at::Tensor>> per_sample_weights_opt_maybe_owned(
      num_eb_ops);
  std::vector<at::Tensor> per_sample_weights(num_eb_ops);

  std::vector<const void *> tables_vector(num_eb_ops);
  std::vector<const void *> indices_vector(num_eb_ops);
  std::vector<const void *> offsets_vector(num_eb_ops);
  std::vector<const float *> weights_vector(num_eb_ops);
  std::vector<void *> dsts_vector(num_eb_ops);
  std::vector<zendnnl::lowoha::embag::embag_params_t> params_vector(num_eb_ops);

  // If TORCH_CHECK() fails, then the pragma omp parallel for cannot be used
  // we will use at::parallel_for from pytorch instead
  at::parallel_for(0, num_eb_ops, 0, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {
      tables_vector[i] = weight[i].data_ptr();
      indices_vector[i] = indices[i].data_ptr();
      offsets_vector[i] = offsets[i].data_ptr();
      per_sample_weights_opt_maybe_owned[i] =
          at::borrow_from_optional_tensor(per_sample_weights_opt[i]);
      per_sample_weights[i] = *per_sample_weights_opt_maybe_owned[i];
      weights_vector[i] = per_sample_weights[i].defined()
                              ? per_sample_weights[i].data_ptr<float>()
                              : nullptr;
      dsts_vector[i] = outputs[i].data_ptr();
      params_vector[i].dtypes.table = data_type_t::u4;
      params_vector[i].dtypes.output = get_zendnnl_dtype(outputs[i]);
      params_vector[i].dtypes.indices = get_zendnnl_dtype(indices[i]);
      params_vector[i].dtypes.offsets = get_zendnnl_dtype(offsets[i]);
      params_vector[i].algo = mode_to_embag_algo(mode[i]);
      params_vector[i].num_embeddings = weight[i].sizes()[0];

      [[maybe_unused]] auto [_unused_0, _unused_1, embedding_dim, _unused_2] =
          compute_quantized_embedding_dims(weight[i], num_bits_per_weight);

      params_vector[i].embedding_dim = embedding_dim;
      params_vector[i].num_indices = indices[i].sizes()[0];
      int num_bags = offsets[i].sizes()[0];
      if (include_last_offset[i]) {
        num_bags -= 1;
      }
      params_vector[i].num_bags = num_bags;
      params_vector[i].is_weights = per_sample_weights[i].defined();
      params_vector[i].include_last_offset = include_last_offset[i];
      params_vector[i].padding_idx = padding_idx[i];
      params_vector[i].fp16_scale_bias = true;
      params_vector[i].dst_stride = outputs[i].strides()[0];
    }
  });

  status_t status = zendnnl::lowoha::embag::group_embedding_bag_direct(
      tables_vector, indices_vector, offsets_vector, weights_vector,
      dsts_vector, params_vector);

  ZENTORCH_CHECK(status == status_t::success,
                 "LOA-operator for group quant embedding bag failed.");

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
}

std::vector<at::Tensor> zendnnl_horizontal_quant_embedding_bag_group_impl(
    at::TensorList weight, at::TensorList indices, at::TensorList offsets,
    int64_t num_bits_per_weight, c10::ScalarType output_dtype,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef mode,
    at::IntArrayRef sparse,
    c10::List<c10::optional<at::Tensor>> per_sample_weights_opt,
    at::IntArrayRef include_last_offset, at::IntArrayRef padding_idx,
    std::string zentorch_op_name) {
  int num_eb_ops = weight.size();
  std::vector<at::Tensor> outputs(num_eb_ops);

  at::parallel_for(0, num_eb_ops, 0, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {
      int num_bags = offsets[i].sizes()[0];
      if (include_last_offset[i]) {
        num_bags -= 1;
      }

      [[maybe_unused]] auto [_unused_0, _unused_1, embedding_dim, _unused_2] =
          compute_quantized_embedding_dims(weight[i], num_bits_per_weight);

      outputs[i] = at::detail::empty_strided_cpu(
          {num_bags, embedding_dim}, {embedding_dim, 1},
          weight[i].options().dtype(output_dtype));
    }
  });

  zendnnl_horizontal_quant_embedding_bag_group_out(
      outputs, weight, indices, offsets, num_bits_per_weight, output_dtype,
      scale_grad_by_freq, mode, sparse, per_sample_weights_opt,
      include_last_offset, padding_idx, zentorch_op_name);

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return outputs;
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

  std::vector<float> weight_scales_vec(weight_scales.data_ptr<float>(),
                                       weight_scales.data_ptr<float>() +
                                           num_eb_rows);
  std::vector<int32_t> weight_zero_points_vec(
      weight_zero_points.data_ptr<int32_t>(),
      weight_zero_points.data_ptr<int32_t>() + num_eb_rows);

  int32_t *weight_ptr = static_cast<int32_t *>(weight.data_ptr());

  std::vector<float> weight_bias(num_eb_rows);

  for (const auto i : c10::irange(num_eb_rows)) {
    weight_bias[i] = weight_zero_points_vec[i] * weight_scales_vec[i] * -1;
  }

  std::vector<int64_t> output_shape = {
      num_eb_rows,
      static_cast<std::int32_t>(
          (num_eb_cols +
           1))}; // Hard coding for int32 weights and Half dtype of scales

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
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_quant_embedding_bag", zendnnl_quant_embedding_bag);
  m.impl("zentorch_quant_embedding_bag.out", zendnnl_quant_embedding_bag_out);
  m.impl("zentorch_horizontal_quant_embedding_bag_group",
         zendnnl_horizontal_quant_embedding_bag_group_impl);
  m.impl("zentorch_horizontal_quant_embedding_bag_group.out",
         zendnnl_horizontal_quant_embedding_bag_group_out);
}

} // namespace zentorch
