/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <string>

namespace zentorch {

at::Tensor zentorch_dynamic_qlinear(const at::Tensor &input,
                                    const at::Tensor &weight,
                                    const at::Tensor &weight_scales,
                                    const c10::optional<at::Tensor> &bias,
                                    std::string zentorch_op_name);

} // namespace zentorch
