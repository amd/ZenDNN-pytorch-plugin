/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <torch/csrc/stable/tensor.h>
#include <vector>

namespace zentorch {

// Single-call MoE entry point. Runs the full ZenDNN postop chain
// (W13 GEMM -> gated activation -> W2 GEMM -> weighted reduce into [T, H])
// inside one call. See GroupMatmul.cpp for the parameter contract.
void zentorch_group_matmul_out_impl(
    std::vector<torch::stable::Tensor> gemm_outputs,
    const std::vector<torch::stable::Tensor> &inputs,
    const std::vector<torch::stable::Tensor> &w13_weights,
    const std::vector<std::optional<torch::stable::Tensor>> &w2_weights,
    std::optional<torch::stable::Tensor> moe_output,
    const std::optional<torch::stable::Tensor> &topk_weights,
    const std::optional<torch::stable::Tensor> &row_ptrs,
    std::string_view activation,
    const std::vector<std::optional<torch::stable::Tensor>> &w13_bias,
    const std::vector<std::optional<torch::stable::Tensor>> &w2_bias,
    const std::vector<std::optional<torch::stable::Tensor>> &w13_scales,
    const std::vector<std::optional<torch::stable::Tensor>> &w2_scales,
    const std::string &zentorch_op_name);

} // namespace zentorch