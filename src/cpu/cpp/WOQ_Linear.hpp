/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "Utils.hpp"
#include <optional>
#include <string>
#include <torch/csrc/stable/tensor.h>

namespace zentorch {

template <UNARY_POST_OP fuse>
torch::stable::Tensor zentorch_woq_linear_unary(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const std::optional<torch::stable::Tensor> &weight_zero_points,
    const std::optional<torch::stable::Tensor> &bias,
    std::string zentorch_op_name);

template <UNARY_POST_OP fuse1, BINARY_POST_OP fuse2>
torch::stable::Tensor zentorch_woq_linear_unary_binary(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const std::optional<torch::stable::Tensor> &weight_zero_points,
    const torch::stable::Tensor &binary_input,
    const std::optional<torch::stable::Tensor> &bias,
    std::string zentorch_op_name);

template <BINARY_POST_OP fuse1, BINARY_POST_OP fuse2>
torch::stable::Tensor zentorch_woq_linear_binary_binary(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const std::optional<torch::stable::Tensor> &weight_zero_points,
    const torch::stable::Tensor &binary1_input,
    const torch::stable::Tensor &binary2_input,
    const std::optional<torch::stable::Tensor> &bias,
    std::string zentorch_op_name);

} // namespace zentorch
