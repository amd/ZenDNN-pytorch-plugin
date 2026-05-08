/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "Utils.hpp"
#include <ATen/ATen.h>
#include <optional>
#include <string>

namespace zentorch {

template <UNARY_POST_OP fuse>
at::Tensor
zentorch_woq_linear_unary(const at::Tensor &input, const at::Tensor &weight,
                          const at::Tensor &weight_scales,
                          const std::optional<at::Tensor> &weight_zero_points,
                          const std::optional<at::Tensor> &bias,
                          std::string zentorch_op_name);

template <UNARY_POST_OP fuse1, BINARY_POST_OP fuse2>
at::Tensor zentorch_woq_linear_unary_binary(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &weight_scales,
    const std::optional<at::Tensor> &weight_zero_points,
    const at::Tensor &binary_input, const std::optional<at::Tensor> &bias,
    std::string zentorch_op_name);

template <BINARY_POST_OP fuse1, BINARY_POST_OP fuse2>
at::Tensor zentorch_woq_linear_binary_binary(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &weight_scales,
    const std::optional<at::Tensor> &weight_zero_points,
    const at::Tensor &binary1_input, const at::Tensor &binary2_input,
    const std::optional<at::Tensor> &bias, std::string zentorch_op_name);

} // namespace zentorch
