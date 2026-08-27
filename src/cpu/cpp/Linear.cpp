/*****************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "MatmulUtils.hpp"
#include "Ops.hpp"
#include "Utils.hpp"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

namespace zentorch {

namespace {

inline void zentorch_linear_impl(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const std::optional<torch::stable::Tensor> &bias,
    torch::stable::Tensor &result,
    const std::vector<std::string_view> &post_op_ids,
    const std::vector<torch::stable::Tensor> &post_op_buffers,
    const bool is_weight_prepacked, std::string zentorch_op_name) {
  const auto input_contiguous = get_contiguous_view(input);
  const auto input_2d_sizes = get_2d_size_for_tensor(input_contiguous);
  const auto input_2d = torch::stable::view(input_contiguous, input_2d_sizes);

  auto result_2d = torch::stable::view(result, get_2d_size_for_tensor(result));
  const bool bias_defined = bias.has_value() && bias->defined();
  const float beta = bias_defined ? 1.0f : 0.0f;
  std::vector<int64_t> post_op_idx;
  post_op_idx.reserve(post_op_ids.size());
  for (const auto &id : post_op_ids) {
    // This map links string names of post-ops (like "relu", "add") to their
    // corresponding enum values.
    post_op_idx.push_back(post_op_map.at(id));
  }

  const torch::stable::Tensor empty_bias;
  const torch::stable::Tensor &bias_for_matmul =
      bias_defined ? *bias : empty_bias;
  zentorch_matmul_impl(input_2d, weight, bias_for_matmul, result_2d,
                       post_op_idx, post_op_buffers, beta, 1.0f /* alpha */,
                       zentorch_op_name, true /* is_weight_const */,
                       is_weight_prepacked);
}

} // namespace

void zentorch_linear_unary_out_impl(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op, std::string zentorch_op_name,
    torch::stable::Tensor &out) {
  const auto weight_transposed = torch::stable::transpose(weight, 0, 1);
  check_linear_and_matmul_out_tensor(input, weight_transposed, out);
  std::vector<std::string_view> post_op_ids = {post_op};

  zentorch_linear_impl(input, weight_transposed, bias, out, post_op_ids,
                       {} /* post_op_buffers */, is_weight_prepacked,
                       zentorch_op_name);
}

void zentorch_linear_unary_out(const torch::stable::Tensor &input,
                               const torch::stable::Tensor &weight,
                               const std::optional<torch::stable::Tensor> &bias,
                               bool is_weight_prepacked,
                               std::string_view post_op,
                               std::string zentorch_op_name,
                               torch::stable::Tensor &out) {
  zentorch_linear_unary_out_impl(input, weight, bias, is_weight_prepacked,
                                 post_op, zentorch_op_name, out);
}

torch::stable::Tensor zentorch_linear_unary(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op, std::string zentorch_op_name) {
  const auto weight_transposed = torch::stable::transpose(weight, 0, 1);
  torch::stable::Tensor result =
      create_linear_and_matmul_output_tensor(input, weight_transposed);
  zentorch_linear_unary_out_impl(input, weight, bias, is_weight_prepacked,
                                 post_op, zentorch_op_name, result);
  return result;
}

void zentorch_linear_unary_binary_out_impl(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &binary_input,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name, torch::stable::Tensor &out) {
  const auto weight_transposed = torch::stable::transpose(weight, 0, 1);
  check_linear_and_matmul_out_tensor(input, weight_transposed, out);
  std::vector<std::string_view> post_op_ids = {post_op_1, post_op_2};
  std::vector<torch::stable::Tensor> post_op_buffers = {
      torch::stable::view(binary_input, get_2d_size_for_tensor(binary_input))};

  zentorch_linear_impl(input, weight_transposed, bias, out, post_op_ids,
                       post_op_buffers, is_weight_prepacked, zentorch_op_name);
}

void zentorch_linear_unary_binary_out(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &binary_input,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name, torch::stable::Tensor &out) {
  zentorch_linear_unary_binary_out_impl(input, weight, binary_input, bias,
                                        is_weight_prepacked, post_op_1,
                                        post_op_2, zentorch_op_name, out);
}

torch::stable::Tensor zentorch_linear_unary_binary(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &binary_input,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name) {
  const auto weight_transposed = torch::stable::transpose(weight, 0, 1);
  torch::stable::Tensor result =
      create_linear_and_matmul_output_tensor(input, weight_transposed);
  zentorch_linear_unary_binary_out_impl(input, weight, binary_input, bias,
                                        is_weight_prepacked, post_op_1,
                                        post_op_2, zentorch_op_name, result);
  return result;
}

void zentorch_linear_binary_binary_out_impl(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &binary_input_1,
    const torch::stable::Tensor &binary_input_2,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name, torch::stable::Tensor &out) {
  const auto weight_transposed = torch::stable::transpose(weight, 0, 1);
  check_linear_and_matmul_out_tensor(input, weight_transposed, out);
  std::vector<std::string_view> post_op_ids = {post_op_1, post_op_2};
  std::vector<torch::stable::Tensor> post_op_buffers = {
      torch::stable::view(binary_input_1,
                          get_2d_size_for_tensor(binary_input_1)),
      torch::stable::view(binary_input_2,
                          get_2d_size_for_tensor(binary_input_2))};

  zentorch_linear_impl(input, weight_transposed, bias, out, post_op_ids,
                       post_op_buffers, is_weight_prepacked, zentorch_op_name);
}

void zentorch_linear_binary_binary_out(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &binary_input_1,
    const torch::stable::Tensor &binary_input_2,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name, torch::stable::Tensor &out) {
  zentorch_linear_binary_binary_out_impl(
      input, weight, binary_input_1, binary_input_2, bias, is_weight_prepacked,
      post_op_1, post_op_2, zentorch_op_name, out);
}

torch::stable::Tensor zentorch_linear_binary_binary(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &binary_input_1,
    const torch::stable::Tensor &binary_input_2,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name) {
  const auto weight_transposed = torch::stable::transpose(weight, 0, 1);
  torch::stable::Tensor result =
      create_linear_and_matmul_output_tensor(input, weight_transposed);
  zentorch_linear_binary_binary_out_impl(
      input, weight, binary_input_1, binary_input_2, bias, is_weight_prepacked,
      post_op_1, post_op_2, zentorch_op_name, result);
  return result;
}

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_linear_unary(Tensor input, Tensor weight, Tensor? bias=None, "
        "*, bool is_weight_prepacked=False, str post_op='none', str "
        "zentorch_op_name='zentorch::zentorch_linear_unary') "
        "-> Tensor");
  m.def("zentorch_linear_unary.out(Tensor input, Tensor weight, "
        "Tensor? bias=None, *, bool is_weight_prepacked=False, "
        "str post_op='none', str "
        "zentorch_op_name='zentorch::zentorch_linear_unary', "
        "Tensor(a!) out) -> ()");

  m.def("zentorch_linear_unary_binary(Tensor input, Tensor weight, Tensor "
        "binary_input, Tensor? bias=None, *, bool is_weight_prepacked=False, "
        "str post_op_1='none', str post_op_2='none', str "
        "zentorch_op_name='zentorch::zentorch_linear_unary_binary') "
        "-> Tensor",
        {at::Tag::needs_fixed_stride_order});
  m.def("zentorch_linear_unary_binary.out(Tensor input, Tensor weight, "
        "Tensor binary_input, Tensor? bias=None, *, bool "
        "is_weight_prepacked=False, str post_op_1='none', str "
        "post_op_2='none', str "
        "zentorch_op_name='zentorch::zentorch_linear_unary_binary', "
        "Tensor(a!) out) -> ()",
        {at::Tag::needs_fixed_stride_order});

  m.def("zentorch_linear_binary_binary(Tensor input, Tensor weight, Tensor "
        "binary_input_1, Tensor binary_input_2, Tensor? bias=None, *, bool "
        "is_weight_prepacked=False, str post_op_1='none', str "
        "post_op_2='none', str "
        "zentorch_op_name='zentorch::zentorch_linear_binary_binary') "
        "-> Tensor",
        {at::Tag::needs_fixed_stride_order});
  m.def("zentorch_linear_binary_binary.out(Tensor input, Tensor weight, "
        "Tensor binary_input_1, Tensor binary_input_2, Tensor? bias=None, "
        "*, bool is_weight_prepacked=False, str post_op_1='none', str "
        "post_op_2='none', str "
        "zentorch_op_name='zentorch::zentorch_linear_binary_binary', "
        "Tensor(a!) out) -> ()",
        {at::Tag::needs_fixed_stride_order});
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_linear_unary", TORCH_BOX(&zentorch::zentorch_linear_unary));
  m.impl("zentorch_linear_unary.out",
         TORCH_BOX(&zentorch::zentorch_linear_unary_out));

  m.impl("zentorch_linear_unary_binary",
         TORCH_BOX(&zentorch::zentorch_linear_unary_binary));
  m.impl("zentorch_linear_unary_binary.out",
         TORCH_BOX(&zentorch::zentorch_linear_unary_binary_out));

  m.impl("zentorch_linear_binary_binary",
         TORCH_BOX(&zentorch::zentorch_linear_binary_binary));
  m.impl("zentorch_linear_binary_binary.out",
         TORCH_BOX(&zentorch::zentorch_linear_binary_binary_out));
}
} // namespace zentorch
