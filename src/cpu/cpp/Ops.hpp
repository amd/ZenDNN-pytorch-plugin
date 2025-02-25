/******************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

// Declarations for ZenTorchOps (EmbedBag etc.)

#pragma once

#include <torch/extension.h>

namespace zentorch {

at::Tensor zentorch_matmul_impl(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
    at::Tensor &self_or_result, const std::vector<int64_t> &post_op_ids,
    const std::vector<at::Tensor> &post_op_buffers, const float &beta,
    const float &alpha, std::string zentorch_op_name);

at::Tensor zentorch_get_packed_embedding_weight(at::Tensor &weight,
                                                at::Tensor &weight_scales,
                                                at::Tensor &weight_zero_points);

at::Tensor zentorch_weight_reorder_for_matmul(at::Tensor &weight,
                                              const bool &is_weight_oc_x_ic);

std::string show_config();
} // namespace zentorch
