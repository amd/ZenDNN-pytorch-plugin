/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "Utils.hpp"
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <string>

namespace zentorch {
template <UNARY_POST_OP fuse>
at::Tensor zentorch_qlinear_unary(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &input_scales, const at::Tensor &input_zero_points,
    const at::Tensor &weight_scales, const at::Tensor &weight_zero_points,
    std::optional<at::Tensor> bias, std::optional<at::Tensor> output_scales,
    std::optional<at::Tensor> output_zero_points,
    std::optional<c10::ScalarType> output_dtype, std::string zentorch_op_name);

template <UNARY_POST_OP fuse>
void zentorch_qlinear_out_unary(
    at::Tensor &result, const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &input_scales, const at::Tensor &input_zero_points,
    const at::Tensor &weight_scales, const at::Tensor &weight_zero_points,
    std::optional<at::Tensor> bias, std::optional<at::Tensor> output_scales,
    std::optional<at::Tensor> output_zero_points,
    std::optional<c10::ScalarType> output_dtype, std::string zentorch_op_name);

template <BINARY_POST_OP fuse1, BINARY_POST_OP fuse2>
at::Tensor zentorch_qlinear_binary_binary(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &input_scales, const at::Tensor &input_zero_points,
    const at::Tensor &weight_scales, const at::Tensor &weight_zero_points,
    const at::Tensor &binary1_input, const at::Tensor &binary2_input,
    std::optional<at::Tensor> bias, std::optional<at::Tensor> output_scales,
    std::optional<at::Tensor> output_zero_points,
    std::optional<c10::ScalarType> output_dtype, std::string zentorch_op_name);

} // namespace zentorch
