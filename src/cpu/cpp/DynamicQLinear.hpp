/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <string>

namespace zentorch {

// Infers the weight-quant mode from the input/weight shapes and weight dtype:
// true for DA8W4 (s4 packed 2-per-int8 or 8-per-int32), false for DA8W8 (s8) or
// a non-int8/int32 weight. Expects a 2D [N, K-dim] weight; K-dim is the
// contraction dim, packed or not.
bool check_weight_and_infer_is_da8w4(const at::Tensor &input,
                                     const at::Tensor &weight);

// Dynamic-quantization linear op (DA8W8 / DA8W4).
at::Tensor zentorch_dynamic_qlinear(const at::Tensor &input,
                                    const at::Tensor &weight,
                                    const at::Tensor &weight_scales,
                                    const c10::optional<at::Tensor> &bias,
                                    std::string zentorch_op_name);

// Out variant: writes into the caller-provided `out` (last, kwarg-only per the
// aten out convention) and returns nothing.
void zentorch_dynamic_qlinear_out(const at::Tensor &input,
                                  const at::Tensor &weight,
                                  const at::Tensor &weight_scales,
                                  const c10::optional<at::Tensor> &bias,
                                  std::string zentorch_op_name,
                                  at::Tensor &out);

} // namespace zentorch
