/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <torch/csrc/stable/tensor.h>

namespace zentorch {

// Fused MoE FFN block. `output` is mutated in place (Tensor(a!)) and the op
// returns void. See FusedMoE.cpp for the full input contract.
void zentorch_fused_moe(torch::stable::Tensor &output,
                        const torch::stable::Tensor &input,
                        const torch::stable::Tensor &w13,
                        const torch::stable::Tensor &w2,
                        const std::optional<torch::stable::Tensor> &w13_bias,
                        const std::optional<torch::stable::Tensor> &w2_bias,
                        const torch::stable::Tensor &topk_weights,
                        const torch::stable::Tensor &topk_id,
                        bool skip_weighted, std::string_view act,
                        const std::optional<torch::stable::Tensor> &w13_scales,
                        const std::optional<torch::stable::Tensor> &w2_scales,
                        std::string zentorch_op_name);

} // namespace zentorch
