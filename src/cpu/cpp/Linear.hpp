/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include <torch/csrc/stable/tensor.h>

namespace zentorch {

torch::stable::Tensor zentorch_linear_unary(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op, std::string zentorch_op_name);

torch::stable::Tensor zentorch_linear_unary_binary(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &binary_input,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name);

torch::stable::Tensor zentorch_linear_binary_binary(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &binary_input_1,
    const torch::stable::Tensor &binary_input_2,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name);

void zentorch_linear_unary_out(const torch::stable::Tensor &input,
                               const torch::stable::Tensor &weight,
                               const std::optional<torch::stable::Tensor> &bias,
                               bool is_weight_prepacked,
                               std::string_view post_op,
                               std::string zentorch_op_name,
                               torch::stable::Tensor &out);

void zentorch_linear_unary_binary_out(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &binary_input,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name, torch::stable::Tensor &out);

void zentorch_linear_binary_binary_out(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &binary_input_1,
    const torch::stable::Tensor &binary_input_2,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name, torch::stable::Tensor &out);

void zentorch_linear_unary_out_impl(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op, std::string zentorch_op_name,
    torch::stable::Tensor &out);

void zentorch_linear_unary_binary_out_impl(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &binary_input,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name, torch::stable::Tensor &out);

void zentorch_linear_binary_binary_out_impl(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &binary_input_1,
    const torch::stable::Tensor &binary_input_2,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name, torch::stable::Tensor &out);

} // namespace zentorch
