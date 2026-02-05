/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EnvReader.hpp"
#include "MatmulUtils.hpp"
#include "Memory.hpp"
#include <ATen/cpu/vec/vec.h>

namespace zentorch {
using namespace zendnnl::interface;
void zentorch_woq_linear_impl(const at::Tensor &input, const at::Tensor &weight,
                              at::Tensor &result, const int64_t group_size,
                              const at::Tensor &weight_scales,
                              const at::Tensor &weight_zero_points,
                              const std::vector<int64_t> &post_op_ids,
                              const std::vector<at::Tensor> &post_op_buffers,
                              std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  LOG(INFO) << "input dimensions: " << input.sizes();
  LOG(INFO) << "weight dimensions: " << weight.sizes();
  LOG(INFO) << "group_size: " << group_size;
  LOG(INFO) << "weight_scales dimensions: " << weight_scales.sizes();
  LOG(INFO) << "weight_zero_points dimensions: " << weight_zero_points.sizes();
  LOG(INFO) << "result dimensions: " << result.sizes();
  LOG(INFO) << "post_op_ids size: " << post_op_ids.size();
  LOG(INFO) << "post_op_buffers size: " << post_op_buffers.size();

  // Weight tensor is expected to be int32 with shape [N, K/8]
  // where each int32 contains 8 packed int4 values
  TORCH_CHECK(weight.dtype() == torch::kInt32,
              "weight must have dtype int32, got ", weight.dtype());
  int packed_factor = 8;
  // transpose weight: [N, K/8] -> [K/8, N]
  auto weight_transposed = weight.transpose(0, 1);
  auto weight_scales_transposed = weight_scales.transpose(0, 1);
  auto weight_zero_points_transposed = weight_zero_points.transpose(0, 1);

  using tensor_opt_ref = std::optional<std::reference_wrapper<tensor_t>>;
  tensor_t woq_input, woq_weight, woq_result, woq_weight_scales,
      woq_weight_zero_points;

  set_zendnnl_tensor_attributes(input, woq_input, "woq_input",
                                false /* is_weight_prepacked */);
  tensor_opt_ref woq_weight_scales_opt_ref = std::nullopt;
  create_zendnnl_quantized_tensor(weight_scales_transposed.contiguous(),
                                  woq_weight_scales, "woq_weight_scales");
  woq_weight_scales_opt_ref = tensor_opt_ref(std::ref(woq_weight_scales));

  tensor_opt_ref woq_weight_zero_points_opt_ref = std::nullopt;
  create_zendnnl_quantized_tensor(weight_zero_points_transposed,
                                  woq_weight_zero_points,
                                  "woq_weight_zero_points");
  woq_weight_zero_points_opt_ref =
      tensor_opt_ref(std::ref(woq_weight_zero_points));

  // Weight is int32 packed: each int32 contains 8 int4 values
  // weight_transposed shape: [K/8, N]
  // Logical int4 tensor size: [K, N] = [weight_transposed.size(0) * 8, N]
  // Each int32 = 4 bytes = 8 int4 values (each int4 = 0.5 bytes)
  // nbytes = (K * N) / 2 = weight_transposed.size(0) * 8 * N / 2
  //        = weight_transposed.size(0) * 4 * N
  //        = weight_transposed.numel() * 4
  set_zendnnl_tensor_attributes(
      weight_transposed.data_ptr(), data_type_t::s4, woq_weight, "woq_weight",
      false /* is_weight_prepacked */,
      {static_cast<size_t>(weight_transposed.size(0)) * packed_factor,
       static_cast<size_t>(weight_transposed.size(1))} /* tensor_sizes */,
      {1UL, static_cast<size_t>(weight_transposed.size(0) *
                                packed_factor)} /* tensor_strides */,
      {} /* tensor_aligned_sizes */,
      static_cast<int64_t>(weight_transposed.numel() * 4) /* nbytes */,
      woq_weight_scales_opt_ref, woq_weight_zero_points_opt_ref);

  set_zendnnl_tensor_attributes(result, woq_result, "woq_result",
                                false /* is_weight_prepacked */);

  auto matmul_context = matmul_context_t();
  set_matmul_context_attributes(matmul_context, woq_weight, post_op_ids,
                                1.0f /* alpha */);
  matmul_context.create();

  auto matmul_operator = matmul_operator_t();
  set_matmul_operator_attributes(matmul_operator, matmul_context, woq_input,
                                 woq_result, post_op_ids, post_op_buffers,
                                 zentorch_op_name);
  status_t status = matmul_operator.execute();
  ZENTORCH_CHECK(status == status_t::success, "operator ",
                 matmul_operator.get_name(),
                 " execution failed for zentorch_matmul_impl.");

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
}

template <UNARY_POST_OP fuse>
at::Tensor zentorch_woq_linear_unary(const at::Tensor &input,
                                     const at::Tensor &weight,
                                     const int64_t group_size,
                                     const at::Tensor &weight_scales,
                                     const at::Tensor &weight_zero_points,
                                     std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  // Weight is int32 packed: shape [N, K/8] where each int32 has 8 int4 values
  // Create a view tensor with logical dimensions [N, K] for output size calc
  // Logical K = weight.size(1) * 8
  auto weight_logical_transposed = at::empty(
      {weight.size(1) * 8, weight.size(0)}, weight.options().dtype(at::kFloat));

  // `input` is viewed as 2d for matmul computation.
  auto input_2d_view =
      get_contiguous_view(input).view(get_2d_size_for_tensor(input));
  // `result` tensor's dtype will be same as input dtype.
  auto output_sz =
      get_matmul_and_linear_output_sizes(input, weight_logical_transposed);
  auto output_strides = get_matmul_and_linear_output_strides(output_sz);

  at::Tensor result =
      at::detail::empty_strided_cpu(output_sz, output_strides, input.options());

  // `result` is viewed as 2d for matmul computation.
  at::Tensor result_2d_view = result.view(get_2d_size_for_tensor(result));

  // Set unary post ops.
  std::vector<at::Tensor> post_op_buffers = {};
  std::vector<int64_t> post_op_ids = {fuse};

  zentorch_woq_linear_impl(input_2d_view, weight, result_2d_view, group_size,
                           weight_scales, weight_zero_points, post_op_ids,
                           post_op_buffers, zentorch_op_name);

  return result;
}

template <BINARY_POST_OP fuse1, BINARY_POST_OP fuse2>
inline at::Tensor zentorch_woq_linear_binary_binary(
    const at::Tensor &input, const at::Tensor &weight, const int64_t group_size,
    const at::Tensor &weight_scales, const at::Tensor &weight_zero_points,
    const at::Tensor &binary1_input, const at::Tensor &binary2_input,
    std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  // Weight is int32 packed: shape [N, K/8] where each int32 has 8 int4 values
  // Create a view tensor with logical dimensions [N, K] for output size calc
  // Logical K = weight.size(1) * 8
  auto weight_logical = at::empty({weight.size(0), weight.size(1) * 8},
                                  weight.options().dtype(at::kFloat));

  // `input` is viewed as 2d for matmul computation.
  auto input_2d_view =
      get_contiguous_view(input).view(get_2d_size_for_tensor(input));
  auto binary1_input_2d_view = get_contiguous_view(binary1_input)
                                   .view(get_2d_size_for_tensor(binary1_input));
  auto binary2_input_2d_view = get_contiguous_view(binary2_input)
                                   .view(get_2d_size_for_tensor(binary2_input));

  // `result` tensor's dtype will depend on output_dtype argument.
  auto output_sz = get_matmul_and_linear_output_sizes(input, weight_logical);
  auto output_strides = get_matmul_and_linear_output_strides(output_sz);

  at::Tensor result =
      at::detail::empty_strided_cpu(output_sz, output_strides, input.options());

  // `result` is viewed as 2d for matmul computation.
  at::Tensor result_2d_view = result.view(get_2d_size_for_tensor(result));

  std::vector<at::Tensor> post_op_buffers = {binary1_input_2d_view,
                                             binary2_input_2d_view};
  std::vector<int64_t> post_op_ids = {fuse1, fuse2};

  LOG(INFO) << "Calling  zentorch_woq_linear_impl from " << __FUNCTION__
            << "!\n";

  zentorch_woq_linear_impl(input_2d_view, weight, result_2d_view, group_size,
                           weight_scales, weight_zero_points, post_op_ids,
                           post_op_buffers, zentorch_op_name);
  return result;
}

at::Tensor
zentorch_weight_from_int4pack_and_repack(const at::Tensor &unpacked_weight) {
  TORCH_CHECK(unpacked_weight.dtype() == torch::kInt8,
              "unpacked_weight must have dtype int8, got ",
              unpacked_weight.dtype());
  TORCH_CHECK(unpacked_weight.dim() == 2, "unpacked_weight must be 2D, got ",
              unpacked_weight.dim(), "D");

  int N = unpacked_weight.size(0);
  int K = unpacked_weight.size(1);
  // Pack 8 int4 columns into 1 int32, reducing column count by 8x
  constexpr int pack_num = 8;
  int K_packed = K / pack_num;

  TORCH_CHECK(K >= pack_num, "K must be at least ", pack_num, ", got ", K);
  TORCH_CHECK(K % pack_num == 0, "K must be divisible by ", pack_num, ", got ",
              K);

  int8_t *weight_data = unpacked_weight.data_ptr<int8_t>();

  // Tensor for row-wise repacked weights [N, K/8], dtype int32
  at::Tensor weight_packed_rowwise =
      torch::zeros({N, K_packed}, torch::TensorOptions()
                                      .dtype(torch::kInt32)
                                      .device(unpacked_weight.device()));
  int32_t *packed_rowwise_data = weight_packed_rowwise.data_ptr<int32_t>();

  // Order map that matches zendnnl's expected byte layout
  // Original int8 packing: (even_col << 4) | odd_col
  // So within each byte: upper nibble = even col, lower nibble = odd col
  // For int32 (4 bytes), we need to swap pairs: [1,0,3,2,5,4,7,6]
  // This ensures byte 0 has (col0 << 4) | col1, byte 1 has (col2 << 4) | col3,
  // etc.
  constexpr int order_map[pack_num] = {0, 1, 2, 3, 4, 5, 6, 7};

  // Process each row independently (parallelized)
  at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
    for (const auto n : c10::irange(begin, end)) {
      // Get pointer to current row in unpacked data
      const int8_t *row_src = weight_data + n * K;

      // Get pointer to current row in row-wise packed data
      // Each row of K values packs into K/8 int32 values
      int32_t *row_dst = packed_rowwise_data + n * K_packed;

      // Pack groups of 8 consecutive values into one int32
      for (int c = 0; c < K_packed; c++) {
        int32_t packed = 0;
        int base_col = c * pack_num;

        // Pack 8 int4 values using the reorder map
        // Each value is shifted by (i * 4) bits
        for (int i = 0; i < pack_num; i++) {
          int8_t val = row_src[base_col + order_map[i]];
          // Mask to 4 bits and shift to correct position
          packed |= static_cast<int32_t>(val & 0x0F) << (i * 4);
        }

        row_dst[c] = packed;
      }
    }
  });

  return weight_packed_rowwise;
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def(
      "zentorch_woq_linear(Tensor input, Tensor weight, "
      "int group_size, Tensor weight_scales, Tensor weight_zero_points, *, str "
      "zentorch_op_name='zentorch::zentorch_woq_linear') -> Tensor");
  m.def("zentorch_woq_linear_relu(Tensor input, Tensor weight, "
        "int group_size, Tensor weight_scales, Tensor weight_zero_points, *, "
        "str zentorch_op_name="
        "'zentorch::zentorch_woq_linear_relu') -> Tensor");
  m.def(
      "zentorch_woq_linear_sigmoid(Tensor input, Tensor weight, "
      "int group_size, Tensor weight_scales, Tensor weight_zero_points, *, str "
      "zentorch_op_name='zentorch::zentorch_woq_linear_sigmoid') -> Tensor");
  m.def("zentorch_woq_linear_mul_add(Tensor input, Tensor weight, "
        "int group_size, Tensor weight_scales, Tensor weight_zero_points, *,"
        "Tensor mul_input, Tensor add_input, str zentorch_op_name="
        "'zentorch::zentorch_woq_linear_mul_add') -> Tensor",
        {at::Tag::needs_fixed_stride_order});
  m.def("zentorch_weight_from_int4pack_and_repack(Tensor unpacked_weight) -> "
        "Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_woq_linear",
         zentorch::zentorch_woq_linear_unary<UNARY_POST_OP::POST_OP_NONE>);
  m.impl("zentorch_woq_linear_relu",
         zentorch::zentorch_woq_linear_unary<UNARY_POST_OP::RELU>);
  m.impl("zentorch_woq_linear_sigmoid",
         zentorch::zentorch_woq_linear_unary<UNARY_POST_OP::SIGMOID>);
  m.impl("zentorch_woq_linear_mul_add",
         zentorch::zentorch_woq_linear_binary_binary<BINARY_POST_OP::MUL,
                                                     BINARY_POST_OP::ADD>);
  m.impl("zentorch_weight_from_int4pack_and_repack",
         zentorch::zentorch_weight_from_int4pack_and_repack);
}

} // namespace zentorch
