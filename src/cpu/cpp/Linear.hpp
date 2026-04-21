/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <ATen/ATen.h>
#include <optional>

namespace zentorch {

// Forward declarations for Linear operations
at::Tensor zentorch_linear_unary(const at::Tensor &input,
                                 const at::Tensor &weight,
                                 const std::optional<at::Tensor> &bias,
                                 bool is_weight_prepacked,
                                 std::string_view post_op,
                                 std::string zentorch_op_name);

at::Tensor zentorch_linear_unary_binary(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &binary_input, const std::optional<at::Tensor> &bias,
    bool is_weight_prepacked, std::string_view post_op_1,
    std::string_view post_op_2, std::string zentorch_op_name);

at::Tensor zentorch_linear_binary_binary(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &binary_input_1, const at::Tensor &binary_input_2,
    const std::optional<at::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name);

// Variants without RECORD_FUNCTION for use by AOTI shims, avoiding
// duplicate profiler entries when the shim already provides one.
at::Tensor zentorch_linear_unary_impl(const at::Tensor &input,
                                      const at::Tensor &weight,
                                      const std::optional<at::Tensor> &bias,
                                      bool is_weight_prepacked,
                                      std::string_view post_op,
                                      std::string zentorch_op_name);

at::Tensor zentorch_linear_unary_binary_impl(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &binary_input, const std::optional<at::Tensor> &bias,
    bool is_weight_prepacked, std::string_view post_op_1,
    std::string_view post_op_2, std::string zentorch_op_name);

at::Tensor zentorch_linear_binary_binary_impl(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &binary_input_1, const at::Tensor &binary_input_2,
    const std::optional<at::Tensor> &bias, bool is_weight_prepacked,
    std::string_view post_op_1, std::string_view post_op_2,
    std::string zentorch_op_name);

} // namespace zentorch
