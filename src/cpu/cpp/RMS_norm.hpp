/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <string>

#include <torch/csrc/stable/tensor.h>

namespace zentorch {

// Fused-add RMS norm: normalizes `input` in place (Tensor(a!)) after adding
// `residual` into it, and writes the running residual back through `residual`
// (Tensor(b!)). Returns void. See RMS_norm.cpp for the full contract.
void zentorch_add_rms_norm_(torch::stable::Tensor &input,
                            const torch::stable::Tensor &weight,
                            torch::stable::Tensor &residual,
                            const double &epsilon,
                            const std::string &zentorch_op_name);

// RMS norm: returns the normalized output in a freshly allocated tensor.
// `input` is read-only (non-mutating; schema `Tensor input`), hence const ref.
torch::stable::Tensor zentorch_rms_norm(const torch::stable::Tensor &input,
                                        const torch::stable::Tensor &weight,
                                        const double &epsilon,
                                        const std::string &zentorch_op_name);

} // namespace zentorch
