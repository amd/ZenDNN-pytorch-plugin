/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <string>
#include <string_view>

namespace zentorch {

// Fused MoE FFN block. `output` is mutated in place (Tensor(a!)) and the op
// returns void. See FusedMoE.cpp for the full input contract.
void zentorch_fused_moe(
    at::Tensor &output, const at::Tensor &input, const at::Tensor &w13,
    const at::Tensor &w2, const c10::optional<at::Tensor> &w13_bias,
    const c10::optional<at::Tensor> &w2_bias, const at::Tensor &topk_weights,
    const at::Tensor &topk_id, bool skip_weighted, std::string_view act,
    const c10::optional<at::Tensor> &w13_scales,
    const c10::optional<at::Tensor> &w2_scales, std::string zentorch_op_name);

} // namespace zentorch
