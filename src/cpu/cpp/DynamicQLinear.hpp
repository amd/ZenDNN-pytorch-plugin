/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <optional>
#include <string>
#include <torch/csrc/stable/tensor.h>

namespace zentorch {

// Infers the weight-quant mode from the input/weight shapes and weight dtype:
// true for DA8W4 (s4 packed 2-per-int8 or 8-per-int32), false for DA8W8 (s8) or
// a non-int8/int32 weight. Expects a 2D [N, K-dim] weight; K-dim is the
// contraction dim, packed or not.
template <typename TensorT>
bool check_weight_and_infer_is_da8w4(const TensorT &input,
                                     const TensorT &weight);

// Dynamic-quantization linear op (DA8W8 / DA8W4).
torch::stable::Tensor
zentorch_dynamic_qlinear(const torch::stable::Tensor &input,
                         const torch::stable::Tensor &weight,
                         const torch::stable::Tensor &weight_scales,
                         const std::optional<torch::stable::Tensor> &bias,
                         std::string zentorch_op_name);

// Out variant: writes into the caller-provided `out` (last, kwarg-only per the
// aten out convention) and returns nothing.
void zentorch_dynamic_qlinear_out(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const std::optional<torch::stable::Tensor> &bias,
    std::string zentorch_op_name, torch::stable::Tensor &out);

} // namespace zentorch
