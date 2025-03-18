/******************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EmbedUtils.hpp"
#include "Memory.hpp"
#include "QuantEmbedUtils.hpp"

#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmEmbedding.h>

#include <ATen/ParallelOpenMP.h>
#define ZENDNN_EMBED_BAG_THRDS 16

using namespace zendnn;

namespace zentorch {
at::Tensor zentorch_embedding_bag_impl(
    const at::Tensor &weight, const at::Tensor &indices,
    const at::Tensor &offsets, const bool &scale_grad_by_freq,
    const int64_t &mode, const bool &sparse,
    const c10::optional<at::Tensor> &per_sample_weights_opt,
    const bool &include_last_offset, const int64_t &padding_idx,
    std::string zentorch_op_name, const int64_t &num_bits_per_weight = 32,
    c10::ScalarType output_dtype = c10::ScalarType::Undefined) {

  at::Tensor cindices, coffsets, per_sample_weights, output;
  memory z_weight, z_indices, z_offsets, z_per_sample_weights_opt, z_dst;
  algorithm z_algorithm;

  // TODO: Add support for quant embedding bag once ZenDNN has support
  ZENTORCH_CHECK((weight.scalar_type() == c10::ScalarType::BFloat16) ||
                     (weight.scalar_type() == c10::ScalarType::Float),
                 "There is no support for quant embedding bag, "
                 "please use quant embedding bag group instead")

  std::tie(cindices, coffsets, per_sample_weights, output) =
      eb_tensors_to_memory(weight, indices, offsets, per_sample_weights_opt,
                           mode, output, z_weight, z_indices, z_offsets,
                           z_per_sample_weights_opt, z_algorithm, z_dst,
                           include_last_offset);

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

  return output;
}

at::Tensor
zentorch_embedding_bag(const at::Tensor &weight, const at::Tensor &indices,
                       const at::Tensor &offsets, bool scale_grad_by_freq,
                       int64_t mode, bool sparse,
                       c10::optional<at::Tensor> per_sample_weights_opt,
                       bool include_last_offset, int64_t padding_idx,
                       std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  auto output = zentorch_embedding_bag_impl(
      weight, indices, offsets, scale_grad_by_freq, mode, sparse,
      per_sample_weights_opt, include_last_offset, padding_idx,
      zentorch_op_name);

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

at::Tensor zentorch_quant_embedding_bag(
    const at::Tensor &weight, const at::Tensor &indices,
    const at::Tensor &offsets, int64_t num_bits_per_weight,
    c10::ScalarType output_dtype, bool scale_grad_by_freq, int64_t mode,
    bool sparse, c10::optional<at::Tensor> per_sample_weights_opt,
    bool include_last_offset, int64_t padding_idx,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  auto output = zentorch_embedding_bag_impl(
      weight, indices, offsets, scale_grad_by_freq, mode, sparse,
      per_sample_weights_opt, include_last_offset, padding_idx,
      zentorch_op_name, num_bits_per_weight, output_dtype);

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

std::vector<at::Tensor> zentorch_horizontal_embedding_bag_group_impl(
    at::TensorList weight, at::TensorList indices, at::TensorList offsets,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef mode,
    at::IntArrayRef sparse,
    c10::List<c10::optional<at::Tensor>> per_sample_weights_opt,
    at::IntArrayRef include_last_offset, at::IntArrayRef padding_idx,
    std::string zentorch_op_name,
    c10::optional<int64_t> num_bits_per_weight = c10::nullopt,
    c10::ScalarType output_dtype = c10::ScalarType::Undefined) {
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
      if (output_dtype != c10::ScalarType::Undefined) {
        ZENTORCH_CHECK(num_bits_per_weight.has_value(),
                       "num_bits_per_weight is required in quantized flow");
        auto num_bits_per_weight_value = num_bits_per_weight.value();
        std::tie(temp_indices[i], temp_offsets[i], per_sample_weights,
                 output[i]) =
            quant_eb_tensors_to_memory(
                weight[i], indices[i], offsets[i], per_sample_weights_opt[i],
                mode[i], output[i], z_weight[i], z_indices[i], z_offsets[i],
                z_per_sample_weights_opt[i], z_algorithm[i], z_destination[i],
                include_last_offset[i], num_bits_per_weight_value,
                output_dtype);
      } else {
        std::tie(temp_indices[i], temp_offsets[i], per_sample_weights,
                 output[i]) =
            eb_tensors_to_memory(weight[i], indices[i], offsets[i],
                                 per_sample_weights_opt[i], mode[i], output[i],
                                 z_weight[i], z_indices[i], z_offsets[i],
                                 z_per_sample_weights_opt[i], z_algorithm[i],
                                 z_destination[i], include_last_offset[i]);
      }

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

std::vector<at::Tensor> zentorch_horizontal_quant_embedding_bag_group(
    at::TensorList weight, at::TensorList indices, at::TensorList offsets,
    int64_t num_bits_per_weight, c10::ScalarType output_dtype,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef mode,
    at::IntArrayRef sparse,
    c10::List<c10::optional<at::Tensor>> per_sample_weights_opt,
    at::IntArrayRef include_last_offset, at::IntArrayRef padding_idx,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  auto out_vec = zentorch_horizontal_embedding_bag_group_impl(
      weight, indices, offsets, scale_grad_by_freq, mode, sparse,
      per_sample_weights_opt, include_last_offset, padding_idx,
      zentorch_op_name, c10::optional<int64_t>(num_bits_per_weight),
      output_dtype);

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return out_vec;
}

template <typename T>
void cat_tensor_population(const at::Tensor &cat_tensor,
                           const at::IntArrayRef &cat_tensor_sizes,
                           const at::Tensor &output_tensor,
                           const int64_t &output_stride,
                           const int64_t &start_address_offset) {
  T *cat_tensor_dataptr = cat_tensor.data_ptr<T>();
  T *start_address = output_tensor.data_ptr<T>() + start_address_offset;

  // Currently this parallelization is commented as the problem size is small
  // and the number of threads available are also less. More experimentations
  // and analysis is required to figure out the parallelization strategy and if
  // it is effective or not.
  // #pragma omp parallel for
  for (int b = 0; b < cat_tensor_sizes[0]; b++) {
    std::memcpy(start_address + (b * output_stride),
                cat_tensor_dataptr + (b * cat_tensor_sizes[1]),
                cat_tensor_sizes[1] * sizeof(T));
  }
}

at::Tensor zentorch_quant_group_eb_mlp_concat_zendnn(
    at::TensorList weight, at::TensorList indices, at::TensorList offsets,
    int64_t num_bits_per_weight, c10::ScalarType output_dtype,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef mode,
    at::IntArrayRef sparse,
    const c10::List<c10::optional<at::Tensor>> per_sample_weights_opt,
    at::IntArrayRef include_last_offset, at::IntArrayRef padding_idx,
    int64_t cat_dim, at::IntArrayRef other_arguments_position,
    at::TensorList other_arguments, std::string zentorch_op_name) {

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

  // We can consider the first element of the include_last_offset TensorList as
  // from the fx pass we have already confirmed that all the values of the non
  // tensor arguments are same all the embedding bags. So, considering just one
  // element from the non tensor argument list suffices for all the embedding
  // bags. include_last_offset actually affects the output size. That's why the
  // following snippet.
  int batch_size = offsets[0].sizes()[0];
  if (include_last_offset[0] == 1) {
    batch_size--;
  }

  const at::Tensor &first_weight = weight[0];
  // In the graph pass we make sure that only one element is passed in the
  // TensorList. Until, the support for multiple tensor concatenation is
  // developed, this check will stay. So using the element at index zero
  // only can be used for this operator's flow.
  const at::Tensor &cat_tensor = other_arguments[0];
  const int64_t &cat_tensor_position = other_arguments_position[0];
  const at::IntArrayRef &cat_tensor_sizes = cat_tensor.sizes();

  // TODO
  // Load function should pass the packed scale dtype and packed zp dtype to
  // the op
  const int num_dim_scale_zp =
      (sizeof(at::Half) + sizeof(at::Half)) / first_weight.element_size();

  const int packed_weight_dim = first_weight.sizes()[1] - num_dim_scale_zp;
  const int64_t bits_in_1_byte = 8;
  const int num_bits_per_packed_weight =
      first_weight.element_size() * bits_in_1_byte;

  // Retrieve original embedding dim before int4 was packed into int32
  // packed_weight_dim * (32 / 4) (int32/int4)
  const int embedding_dim =
      packed_weight_dim * (num_bits_per_packed_weight / num_bits_per_weight);

  // Library is expecting a stride of -1 when the concatenation dimension is
  // zero as the copy will happen across the first dimension which is the
  // batch size. So, the pointer need not move by a significant step and
  // it need not be calculated. But that is not the case for concatenation
  // dimension 1. When the concatenation dimension is 1, the pointer needs to
  // move by a certain step, which will be difficult for the library to
  // calculate. Hence zentorch is passing this information.
  // The striding happens on the batch size. Hence, the jump is required on
  // concatenation dim 0 and not 1.
  int64_t output_stride =
      (cat_dim == 0) ? -1 : (num_eb_ops * embedding_dim) + cat_tensor_sizes[1];

  // Calculation of the size of output_tensor based on the cat dim
  // TODO
  // How to handle the AMP case where the mlp output and the embedding bag
  // outputs can be of different datatype? Currently the output_tensor is of the
  // same dtype and other options as the cat_tensor. Even this is risky as of
  // now, but not of high risk as the datatypes in the entire flow are same. But
  // once amp is switched on, this snippet can of a cause of errors.

  // Either way in both the cases, the embedding bag's output is always a 2D
  // tensor. So, we don't have to bother about the output_tensor being anything
  // apart from 2D
  at::Tensor output_tensor;
  if (cat_dim == 0) {
    // A question that comes to mind here is, why is there a common
    // embedding_dim in the first dimensionality of the output_tensor, but the
    // cat_tensor's zeroth dimension is being considered in the zeroth
    // dimension of the output tensor. When we are concatenating across the
    // zeroth dimension, for the compatibility of the tensors for concatenation
    // requires that the dimensions apart from zeroth are the same for all the
    // tensors. So, an assumption is made in the CPP part that the compatibility
    // is taken care from the python side, and we can assume that the other
    // tensor's first dimension is indeed equal to the embedding_dim of the
    // embedding bags. The zeroth dimension can be anything. So the
    // cat_tensor's zeroth  dimension is seperately added in the zeroth
    // dimension of the output_tensor.
    output_tensor = at::detail::empty_strided_cpu(
        {(batch_size * num_eb_ops) + cat_tensor_sizes[0], embedding_dim},
        {embedding_dim, 1}, other_arguments[0].options());
  } else if (cat_dim == 1) {
    // Similarly here, the question that comes to mind here is, why is there a
    // common batch_size in the zeroth dimensionality of the output_tensor, but
    // the cat_tensor's first dimension is being considered in the first
    // dimension of the output tensor. When we are concatenating across the
    // first dimension, for the compatibility of the tensors for concatenation
    // requires that the dimensions apart from first are the same for all the
    // tensors. So, an assumption is made in the CPP part that the compatibility
    // is taken care from the python side, and we can assume that the other
    // tensor's zeroth dimension is indeed equal to the batch_size of the
    // embedding bags. The first dimension can be anything. So the
    // cat_tensor's first  dimension is seperately added in the first
    // dimension of the output_tensor.
    const int64_t dim_1 = (num_eb_ops * embedding_dim) + cat_tensor_sizes[1];
    output_tensor = at::detail::empty_strided_cpu(
        {batch_size, dim_1}, {dim_1, 1}, other_arguments[0].options());
  }

  // If TORCH_CHECK() fails, then the pragma omp parallel for cannot be used
  // we will use at::parallel_for from pytorch instead
  at::parallel_for(0, num_eb_ops, 0, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {

      at::Tensor per_sample_weights;
      std::tie(temp_indices[i], temp_offsets[i], per_sample_weights) =
          quant_eb_tensors_to_memory(
              weight[i], indices[i], offsets[i], per_sample_weights_opt[i],
              mode[i], z_weight[i], z_indices[i], z_offsets[i],
              z_per_sample_weights_opt[i], z_algorithm[i],
              include_last_offset[i], num_bits_per_weight, output_dtype);

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

  memory output_tensor_memory = zen_memory(output_tensor);
  std::vector<memory> z_output_tensor = {output_tensor_memory};

  // Library call
  zendnn_custom_op::zendnn_grp_embedding_bag(
      z_weight, z_indices, z_offsets, z_scale_grad_by_freq, z_algorithm,
      z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
      z_include_last_offset, z_padding_idx, z_output_tensor,
      zentorch_op_name.c_str(), cat_dim, cat_tensor_position, output_stride);

  // As of now, we are assuming that the fx pass will ensure only one extra
  // tensor will be passed in the other_arguments TensorList and that the tensor
  // will be always be concatenated at zeroth or last position.
  // TODO
  // To futureproof this code and accept as many other arguments and at
  // arbitrary positions in the concatenated tensor, an optimal approach must be
  // found out and designed.

  // TODO
  // Need to add documentation here
  output_stride = (output_stride == -1) ? cat_tensor_sizes[1] : output_stride;
  const int64_t batch_size_multiplier = (cat_dim == 0) ? batch_size : 1;
  const int64_t start_address_offset =
      cat_tensor_position *
      ((batch_size_multiplier * num_eb_ops * embedding_dim)) * (-1);

  if (output_dtype == c10::ScalarType::Float) {
    cat_tensor_population<float>(cat_tensor, cat_tensor_sizes, output_tensor,
                                 output_stride, start_address_offset);
  } else if (output_dtype == c10::ScalarType::BFloat16) {
    cat_tensor_population<at::BFloat16>(cat_tensor, cat_tensor_sizes,
                                        output_tensor, output_stride,
                                        start_address_offset);
  }

  return output_tensor;
}

template <typename IndexType, typename OffsetType, typename OutputType>
void embedding_bag_nbit_impl_with_strides(
    at::Tensor &output, const at::Tensor &weight, const at::Tensor &indices,
    const at::Tensor &offsets, const int bit_width, const int output_stride,
    const int num_embedding_bags, const int embedding_dim,
    const at::IntArrayRef &cat_tensor_sizes, const int cat_dim,
    const int cat_tensor_position, const int include_last_offset,
    const int idx) {

  ZENTORCH_CHECK(weight.scalar_type() == at::kInt, "Weight type is not int");
  ZENTORCH_CHECK(weight.dim() == 2, "Weight Dimensions not equal to 2.");
  ZENTORCH_CHECK(offsets.dim() == 1, "Offsets Dimensions not equal to 1.");

  OffsetType *offsets_data = static_cast<OffsetType *>(offsets.data_ptr());
  const auto weight_sizes = weight.sizes();
  int64_t batch_size = offsets.sizes()[0];

  if (include_last_offset == 0) {
    // This is the case for include_last_offset=False
    // 1. We have to increase the offset tensor/array by one
    // 2. In the last position, we have to add indices size.
    OffsetType *fbgemm_offsets = new OffsetType[batch_size + 1];
    std::memcpy(fbgemm_offsets, offsets_data, batch_size * sizeof(OffsetType));
    static_cast<OffsetType *>(fbgemm_offsets)[batch_size] = indices.sizes()[0];
    offsets_data = fbgemm_offsets;
  } else {
    batch_size--;
  }

  const int64_t N = weight_sizes[0];
  const auto weight_data = static_cast<uint8_t *>(weight.data_ptr());
  const auto indices_data = static_cast<IndexType *>(indices.data_ptr());
  const int64_t batch_size_multiplier = (cat_dim == 0) ? batch_size : 1;

  // TODO
  // This can be pre-computed in the previous function and can be passed
  // Currently doing this is giving an error from FBGEMM. So, keeping the
  // changes as they are as of now.
  int start_addr_offset =
      (cat_dim == 0)
          ? cat_tensor_sizes[0] * embedding_dim * (cat_tensor_position + 1)
          : cat_tensor_sizes[1] * (cat_tensor_position + 1);

  auto output_data = static_cast<OutputType *>(output.data_ptr()) +
                     start_addr_offset +
                     (idx * batch_size_multiplier * embedding_dim);

  const bool is_bf16_out =
      (typeid(OutputType) == typeid(uint16_t)) ? true : false;

  auto kernel = fbgemm::GenerateEmbeddingSpMDMNBitWithStrides<
      IndexType, OffsetType, /*OutType=*/OutputType, /*THREAD_LOCAL=*/true>(
      /*bit_rate*/ bit_width,
      /*block_size=*/embedding_dim,
      /*has_weight=*/false,
      /*normalize_by_lengths=*/false,
      /*prefetch=*/16, // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      /*is_weight_positional=*/false,
      /*use_offsets=*/true,
      /*output_stride*/ output_stride,
      /*input_stride=*/-1,
      /*scale_bias_last=*/true,
      /*is_bf16_out=*/is_bf16_out);

  at::parallel_for(0, batch_size, 1, [&](int64_t start_idx, int64_t end_idx) {
    bool success = kernel(
        /*output_size=*/end_idx - start_idx,
        /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
        /*data_size=*/N,
        /*input=*/weight_data,
        /*indices=*/indices_data + offsets_data[start_idx],
        /*offsets_or_lengths=*/offsets_data + start_idx,
        /*weights=*/nullptr,
        /*out=*/output_data + start_idx * output_stride);

    ZENTORCH_CHECK(success, "FBGEMM kernel call unsucessful");
  });

  if (include_last_offset == 0) {
    delete[] offsets_data;
  }
}

at::Tensor zentorch_quant_group_eb_mlp_concat_fbgemm(
    at::TensorList weight, at::TensorList indices, at::TensorList offsets,
    int64_t num_bits_per_weight, c10::ScalarType output_dtype,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef mode,
    at::IntArrayRef sparse,
    const c10::List<c10::optional<at::Tensor>> per_sample_weights_opt,
    at::IntArrayRef include_last_offset, at::IntArrayRef padding_idx,
    int64_t cat_dim, at::IntArrayRef other_arguments_position,
    at::TensorList other_arguments, std::string zentorch_op_name) {

  int num_eb_ops = weight.size();

  // We can consider the first element of the include_last_offset TensorList as
  // from the fx pass we have already confirmed that all the values of the non
  // tensor arguments are same all the embedding bags. So, considering just one
  // element from the non tensor argument list suffices for all the embedding
  // bags. include_last_offset actually affects the output size. That's why the
  // following snippet.
  int batch_size = offsets[0].sizes()[0];
  if (include_last_offset[0] == 1) {
    batch_size--;
  }

  const at::Tensor &first_weight = weight[0];
  // In the graph pass we make sure that only one element is passed in the
  // TensorList. Until, the support for multiple tensor concatenation is
  // developed, this check will stay. So using the element at index zero
  // only can be used for this operator's flow.
  const at::Tensor &cat_tensor = other_arguments[0];
  const int64_t &cat_tensor_position = other_arguments_position[0];
  const at::IntArrayRef &cat_tensor_sizes = cat_tensor.sizes();

  // TODO
  // Load function should pass the packed scale dtype and packed zp dtype to
  // the op
  const int num_dim_scale_zp =
      (sizeof(at::Half) + sizeof(at::Half)) / first_weight.element_size();

  const int packed_weight_dim = first_weight.sizes()[1] - num_dim_scale_zp;
  const int64_t bits_in_1_byte = 8;
  const int num_bits_per_packed_weight =
      first_weight.element_size() * bits_in_1_byte;

  // Retrieve original embedding dim before int4 was packed into int32
  // packed_weight_dim * (32 / 4) (int32/int4)
  const int embedding_dim =
      packed_weight_dim * (num_bits_per_packed_weight / num_bits_per_weight);

  const int64_t output_stride =
      (cat_dim == 0) ? embedding_dim
                     : (num_eb_ops * embedding_dim) + cat_tensor_sizes[1];
  const int64_t batch_size_multiplier = (cat_dim == 0) ? batch_size : 1;
  const int64_t start_address_offset =
      cat_tensor_position *
      ((batch_size_multiplier * num_eb_ops * embedding_dim)) * (-1);

  at::Tensor output_tensor;
  if (cat_dim == 0) {
    output_tensor = at::detail::empty_strided_cpu(
        {(batch_size * num_eb_ops) + cat_tensor_sizes[0], embedding_dim},
        {embedding_dim, 1}, other_arguments[0].options());
  } else if (cat_dim == 1) {
    output_tensor = at::detail::empty_strided_cpu(
        {batch_size, (num_eb_ops * embedding_dim) + cat_tensor_sizes[1]},
        {(num_eb_ops * embedding_dim) + cat_tensor_sizes[1], 1},
        other_arguments[0].options());
  }

  if (output_dtype == c10::ScalarType::Float) {
    cat_tensor_population<float>(cat_tensor, cat_tensor_sizes, output_tensor,
                                 output_stride, start_address_offset);
    for (int idx = 0; idx < num_eb_ops; idx++) {
      if (indices[idx].scalar_type() == c10::ScalarType::Long) {
        if (offsets[idx].scalar_type() == c10::ScalarType::Long) {
          embedding_bag_nbit_impl_with_strides<int64_t, int64_t, float>(
              output_tensor, weight[idx], indices[idx], offsets[idx],
              num_bits_per_weight, output_stride, num_eb_ops, embedding_dim,
              cat_tensor_sizes, cat_dim, cat_tensor_position,
              include_last_offset[idx], idx);
        } else if (offsets[idx].scalar_type() == c10::ScalarType::Int) {
          embedding_bag_nbit_impl_with_strides<int64_t, int, float>(
              output_tensor, weight[idx], indices[idx], offsets[idx],
              num_bits_per_weight, output_stride, num_eb_ops, embedding_dim,
              cat_tensor_sizes, cat_dim, cat_tensor_position,
              include_last_offset[idx], idx);
        }
      } else if (indices[idx].scalar_type() == c10::ScalarType::Int) {
        if (offsets[idx].scalar_type() == c10::ScalarType::Long) {
          embedding_bag_nbit_impl_with_strides<int, int64_t, float>(
              output_tensor, weight[idx], indices[idx], offsets[idx],
              num_bits_per_weight, output_stride, num_eb_ops, embedding_dim,
              cat_tensor_sizes, cat_dim, cat_tensor_position,
              include_last_offset[idx], idx);
        } else if (offsets[idx].scalar_type() == c10::ScalarType::Int) {
          embedding_bag_nbit_impl_with_strides<int, int, float>(
              output_tensor, weight[idx], indices[idx], offsets[idx],
              num_bits_per_weight, output_stride, num_eb_ops, embedding_dim,
              cat_tensor_sizes, cat_dim, cat_tensor_position,
              include_last_offset[idx], idx);
        }
      }
    }
  } else if (output_dtype == c10::ScalarType::BFloat16) {
    cat_tensor_population<at::BFloat16>(cat_tensor, cat_tensor_sizes,
                                        output_tensor, output_stride,
                                        start_address_offset);
    for (int idx = 0; idx < num_eb_ops; idx++) {
      if (indices[idx].scalar_type() == c10::ScalarType::Long) {
        if (offsets[idx].scalar_type() == c10::ScalarType::Long) {
          embedding_bag_nbit_impl_with_strides<int64_t, int64_t, uint16_t>(
              output_tensor, weight[idx], indices[idx], offsets[idx],
              num_bits_per_weight, output_stride, num_eb_ops, embedding_dim,
              cat_tensor_sizes, cat_dim, cat_tensor_position,
              include_last_offset[idx], idx);
        } else if (offsets[idx].scalar_type() == c10::ScalarType::Int) {
          embedding_bag_nbit_impl_with_strides<int64_t, int, uint16_t>(
              output_tensor, weight[idx], indices[idx], offsets[idx],
              num_bits_per_weight, output_stride, num_eb_ops, embedding_dim,
              cat_tensor_sizes, cat_dim, cat_tensor_position,
              include_last_offset[idx], idx);
        }
      } else if (indices[idx].scalar_type() == c10::ScalarType::Int) {
        if (offsets[idx].scalar_type() == c10::ScalarType::Long) {
          embedding_bag_nbit_impl_with_strides<int, int64_t, uint16_t>(
              output_tensor, weight[idx], indices[idx], offsets[idx],
              num_bits_per_weight, output_stride, num_eb_ops, embedding_dim,
              cat_tensor_sizes, cat_dim, cat_tensor_position,
              include_last_offset[idx], idx);
        } else if (offsets[idx].scalar_type() == c10::ScalarType::Int) {
          embedding_bag_nbit_impl_with_strides<int, int, uint16_t>(
              output_tensor, weight[idx], indices[idx], offsets[idx],
              num_bits_per_weight, output_stride, num_eb_ops, embedding_dim,
              cat_tensor_sizes, cat_dim, cat_tensor_position,
              include_last_offset[idx], idx);
        }
      }
    }
  }

  return output_tensor;
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
std::vector<at::Tensor> zentorch_horizontal_embedding_bag_group(
    at::TensorList weight, at::TensorList indices, at::TensorList offsets,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef mode,
    at::IntArrayRef sparse,
    c10::List<c10::optional<at::Tensor>> per_sample_weights_opt,
    at::IntArrayRef include_last_offset, at::IntArrayRef padding_idx,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  auto out_vec = zentorch_horizontal_embedding_bag_group_impl(
      weight, indices, offsets, scale_grad_by_freq, mode, sparse,
      per_sample_weights_opt, include_last_offset, padding_idx,
      zentorch_op_name);

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return out_vec;
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
  m.def("zentorch_quant_embedding_bag(Tensor weight, Tensor indices, Tensor "
        "offsets,"
        " int num_bits_per_weight, ScalarType output_dtype,"
        " bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor?"
        " per_sample_weights=None, bool include_last_offset=False, int"
        " padding_idx=-1,"
        " str zentorch_op_name="
        "'zentorch::zentorch_quant_embedding_bag') -> Tensor");
  m.def("zentorch_horizontal_quant_embedding_bag_group(Tensor[] weight, "
        "Tensor[] indices, Tensor[] offsets, "
        " int num_bits_per_weight, ScalarType output_dtype,"
        " int[] scale_grad_by_freq, "
        "int[] mode, int[] sparse, Tensor?[] per_sample_weights, "
        "int[] include_last_offset, int[] padding_idx, str "
        "zentorch_op_name = "
        "'zentorch::zentorch_horizontal_quant_embedding_bag_group') -> "
        "Tensor[]");
  m.def("zentorch_quant_group_eb_mlp_concat_zendnn(Tensor[] weight, "
        "Tensor[] indices, Tensor[] offsets, "
        " int num_bits_per_weight, ScalarType output_dtype,"
        " int[] scale_grad_by_freq, "
        "int[] mode, int[] sparse, Tensor?[] per_sample_weights, "
        "int[] include_last_offset, int[] padding_idx, int cat_dim, "
        "int[] other_arguments_position, Tensor[] other_arguments, str "
        "zentorch_op_name = "
        "'zentorch::zentorch_quant_group_eb_mlp_concat_zendnn') -> "
        "Tensor");
  m.def("zentorch_quant_group_eb_mlp_concat_fbgemm(Tensor[] weight, "
        "Tensor[] indices, Tensor[] offsets, "
        " int num_bits_per_weight, ScalarType output_dtype,"
        " int[] scale_grad_by_freq, "
        "int[] mode, int[] sparse, Tensor?[] per_sample_weights, "
        "int[] include_last_offset, int[] padding_idx, int cat_dim, "
        "int[] other_arguments_position, Tensor[] other_arguments, str "
        "zentorch_op_name = "
        "'zentorch::zentorch_quant_group_eb_mlp_concat_fbgemm') -> "
        "Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_embedding_bag", zentorch_embedding_bag);
  m.impl("zentorch_quant_embedding_bag", zentorch_quant_embedding_bag);
  m.impl("zentorch_horizontal_embedding_bag_group",
         zentorch_horizontal_embedding_bag_group);
  m.impl("zentorch_horizontal_quant_embedding_bag_group",
         zentorch_horizontal_quant_embedding_bag_group);
  m.impl("zentorch_quant_group_eb_mlp_concat_zendnn",
         zentorch_quant_group_eb_mlp_concat_zendnn);
  m.impl("zentorch_quant_group_eb_mlp_concat_fbgemm",
         zentorch_quant_group_eb_mlp_concat_fbgemm);
}

} // namespace zentorch
