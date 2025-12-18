/*****************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "MatmulUtils.hpp"
#include "Ops.hpp"

namespace zentorch {
inline void zentorch_linear_impl(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
    at::Tensor &result, const std::vector<std::string_view> &post_op_ids,
    const std::vector<at::Tensor> &post_op_buffers,
    const bool is_weight_prepacked, std::string zentorch_op_name) {
  // Get appropriately tensors for Linear(2D input, transposed weight, 2D
  // result)

  // TODO
  //  in long term we should handle the reshape of the input outside the linear
  //  and let graph passes handle the reshape. The input to linear must always
  //  be 2D and contiguous.

  const auto input_contiguous = get_contiguous_view(input);
  const auto input_2d_sizes = get_2d_size_for_tensor(input_contiguous);
  const auto input_2d = input_contiguous.view(input_2d_sizes);

  auto result_2d = result.view(get_2d_size_for_tensor(result));
  const float beta = bias.defined() ? 1.0f : 0.0f;
  std::vector<int64_t> post_op_idx;
  for (const auto &id : post_op_ids) {
    // This map links string names of post-ops (like "relu", "add") to their
    // corresponding enum values.
    post_op_idx.push_back(post_op_map.at(id));
  }

  zentorch_matmul_impl(input_2d, weight, bias, result_2d, post_op_idx,
                       post_op_buffers, beta, 1.0f /* alpha */,
                       zentorch_op_name, true /* is_weight_const */,
                       is_weight_prepacked);
}

at::Tensor zentorch_linear_unary(const at::Tensor &input,
                                 const at::Tensor &weight,
                                 const std::optional<at::Tensor> &bias,
                                 bool is_weight_prepacked,
                                 std::string_view post_op,
                                 std::string zentorch_op_name) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor &bias_t = *bias_maybe_owned;
  // Create output tensor with appropriate size and strides
  const at::Tensor &weight_transposed = weight.t();
  at::Tensor result =
      create_linear_and_matmul_output_tensor(input, weight_transposed);
  std::vector<std::string_view> post_op_ids = {post_op};

  // Perform ZenTorch linear operation
  zentorch_linear_impl(input, weight_transposed, bias_t, result, post_op_ids,
                       {} /* post_op_buffers */, is_weight_prepacked,
                       zentorch_op_name);
  return result;
}

at::Tensor zentorch_linear_unary_binary(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &binary_input, const std::optional<at::Tensor> &bias,
    bool is_weight_prepacked, std::string_view post_op_1,
    std::string_view post_op_2, std::string zentorch_op_name) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor &bias_t = *bias_maybe_owned;
  const at::Tensor &weight_transposed = weight.t();
  // Create output tensor with appropriate size and strides
  at::Tensor result =
      create_linear_and_matmul_output_tensor(input, weight_transposed);
  // Initialize post-operation containers
  std::vector<std::string_view> post_op_ids = {post_op_1, post_op_2};
  std::vector<at::Tensor> post_op_buffers = {
      binary_input.view(get_2d_size_for_tensor(binary_input))};
  // Perform ZenTorch linear operation
  zentorch_linear_impl(input, weight_transposed, bias_t, result, post_op_ids,
                       post_op_buffers, is_weight_prepacked, zentorch_op_name);
  return result;
}

at::Tensor zentorch_linear_binary_binary(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &binary_input_1, const at::Tensor &binary_input_2,
    const std::optional<at::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor &bias_t = *bias_maybe_owned;
  const at::Tensor &weight_transposed = weight.t();
  at::Tensor result =
      create_linear_and_matmul_output_tensor(input, weight_transposed);
  // Initialize post-operation containers
  std::vector<std::string_view> post_op_ids = {post_op_1, post_op_2};

  std::vector<at::Tensor> post_op_buffers = {
      binary_input_1.view(get_2d_size_for_tensor(binary_input_1)),
      binary_input_2.view(get_2d_size_for_tensor(binary_input_2))};

  // Perform ZenTorch linear operation
  zentorch_linear_impl(input, weight_transposed, bias_t, result, post_op_ids,
                       post_op_buffers, is_weight_prepacked, zentorch_op_name);

  return result;
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_linear_unary(Tensor input, Tensor weight, Tensor? bias=None, "
        "*, bool is_weight_prepacked=False, str post_op='none', str "
        "zentorch_op_name='zentorch::zentorch_linear_unary') "
        "-> Tensor");
  m.def("zentorch_linear_unary_binary(Tensor input, Tensor weight, Tensor "
        "binary_input, Tensor? bias=None, *, bool is_weight_prepacked=False, "
        "str post_op_1='none', str post_op_2='none', str "
        "zentorch_op_name='zentorch::zentorch_linear_unary_binary') "
        "-> Tensor");
  m.def("zentorch_linear_binary_binary(Tensor input, Tensor weight, Tensor "
        "binary_input_1, Tensor binary_input_2, Tensor? bias=None, *, bool "
        "is_weight_prepacked=False, str post_op_1='none', str "
        "post_op_2='none', str "
        "zentorch_op_name='zentorch::zentorch_linear_binary_binary') "
        "-> Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_linear_unary", zentorch_linear_unary);
  m.impl("zentorch_linear_unary_binary", zentorch_linear_unary_binary);
  m.impl("zentorch_linear_binary_binary", zentorch_linear_binary_binary);
}
} // namespace zentorch
