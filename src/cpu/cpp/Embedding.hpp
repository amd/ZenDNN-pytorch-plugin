/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <ATen/ATen.h>
#include <string>

namespace zentorch {

// Embedding lookup: gathers rows of `weight` indexed by `indices` and returns
// the result in a freshly allocated tensor. See Embedding.cpp for the full
// contract.
at::Tensor zentorch_embedding(const at::Tensor &weight,
                              const at::Tensor &indices, int64_t padding_idx,
                              bool scale_grad_by_freq, bool sparse,
                              std::string zentorch_op_name);

// zendnnl execution: writes the lookup result into `output` instead of
// allocating. Registered directly for `zentorch_embedding.out` and reused by
// the AOTI shim.
void zendnnl_embedding_impl(const at::Tensor &weight, const at::Tensor &indices,
                            int64_t padding_idx, bool scale_grad_by_freq,
                            bool sparse, std::string zentorch_op_name,
                            at::Tensor &output);

} // namespace zentorch
