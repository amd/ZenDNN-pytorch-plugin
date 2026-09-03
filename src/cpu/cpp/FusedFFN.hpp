/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "MatmulUtils.hpp"
#include <optional>
#include <string>
#include <string_view>
#include <torch/csrc/stable/tensor.h>

namespace zentorch {

// Fused single-expert (non-MoE) FFN block: W13 GEMM -> gated activation -> W2
// GEMM. `output` is mutated in place and the op returns void.
void zentorch_fused_ffn_concat_out_impl(
    torch::stable::Tensor &output, const torch::stable::Tensor &input,
    const torch::stable::Tensor &w13_weight,
    const torch::stable::Tensor &w2_weight,
    const std::optional<torch::stable::Tensor> &w13_bias,
    const std::optional<torch::stable::Tensor> &w2_bias,
    std::string_view activation,
    const std::optional<torch::stable::Tensor> &w13_scale,
    const std::optional<torch::stable::Tensor> &w2_scale,
    std::string zentorch_op_name);

} // namespace zentorch
