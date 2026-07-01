/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <ATen/ATen.h>
#include <string>

namespace zentorch {

// Fused-add RMS norm: normalizes `input` in place (Tensor(a!)) after adding
// `residual` into it, and writes the running residual back through `residual`
// (Tensor(b!)). Returns void. See RMS_norm.cpp for the full contract.
void zentorch_add_rms_norm_(at::Tensor &input, const at::Tensor &weight,
                            at::Tensor &residual, const double epsilon,
                            std::string zentorch_op_name);

// RMS norm: returns the normalized output in a freshly allocated tensor.
// `input` is read-only (non-mutating; schema `Tensor input`), hence const ref.
at::Tensor zentorch_rms_norm(const at::Tensor &input, const at::Tensor &weight,
                             const double epsilon,
                             std::string zentorch_op_name);

} // namespace zentorch
