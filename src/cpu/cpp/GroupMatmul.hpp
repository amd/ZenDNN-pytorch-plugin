/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <string>
#include <string_view>
#include <vector>

namespace zentorch {

// Single-call MoE entry point. Runs the full ZenDNN postop chain
// (W13 GEMM -> gated activation -> W2 GEMM -> weighted reduce into [T, H])
// inside one call. See GroupMatmul.cpp for the parameter contract.
void zentorch_group_matmul_out_impl(
    std::vector<at::Tensor> gemm_outputs, const std::vector<at::Tensor> &inputs,
    const std::vector<at::Tensor> &weights,
    const std::vector<c10::optional<at::Tensor>> &bias,
    std::string_view activation,
    const std::vector<c10::optional<at::Tensor>> &w2_weights,
    const std::vector<c10::optional<at::Tensor>> &w2_bias,
    c10::optional<at::Tensor> moe_output,
    const c10::optional<at::Tensor> &topk_weights,
    const c10::optional<at::Tensor> &row_ptrs,
    const std::string &zentorch_op_name);

} // namespace zentorch
