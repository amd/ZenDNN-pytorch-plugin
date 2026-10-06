/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <torch/csrc/stable/tensor.h>
#include <vector>

namespace zentorch {

// Bytes writable at one expert's weight pointer, or 0 when the experts are
// packed. Takes any one expert slice and the total expert count: routing
// permutes the list, so a slice's position says nothing, while every slice
// reports the same parent allocation. Callers holding the [E, ...] parent pass
// select(parent, 0, 0) and parent.size(0).
// Assumption in zentorch - all expert weights share a shape and size.
int64_t expert_capacity_bytes(const torch::stable::Tensor &expert,
                              int64_t count);

// Boxed as `zentorch_group_matmul.out`. Gated act, W2, and MoE reduce run
// only when those args are populated.
//
// `src_scales`: one [M_e, 1] tensor per active expert, or empty.
//   * int8 inputs: caller passes filled scales; `dynamic_quant = false`.
//     Unique-token FusedMoE writes f32 scales via
//     dynamic_per_token_quant_bf16_s8_native; this op converts to
//     wei_scale dtype when they differ. Fused w2 needs caller-allocated
//     bf16 gemm_outputs (dst_down).
//   * bf16/fp + DA8W8/DA8W4: pass empty; this op allocates {M, 1} and
//     sets `dynamic_quant = true` so ZenDNN fills the scales.
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
    const std::vector<std::optional<torch::stable::Tensor>> &src_scales,
    const std::string &zentorch_op_name);

} // namespace zentorch
