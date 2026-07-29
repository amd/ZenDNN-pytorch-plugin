/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <string>

namespace zentorch {

// Infer the weight-quant mode from input/weight shapes+dtype: returns true for
// DA8W4 (packed s4), false for DA8W8 (s8). Assumes input.dim() >= 1 and
// weight.dim() == 2. Shared so other ops/PRs can reuse the same inference.
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
