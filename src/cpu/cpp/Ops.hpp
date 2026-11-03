/******************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

// Declarations for ZenTorchOps (EmbedBag etc.)

#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/stable/tensor.h>

namespace zentorch {

torch::stable::Tensor zentorch_matmul_impl(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &bias, torch::stable::Tensor &self_or_result,
    const std::vector<int64_t> &post_op_ids,
    const std::vector<torch::stable::Tensor> &post_op_buffers,
    const float &beta, const float &alpha, std::string zentorch_op_name,
    const bool is_const = true, const bool is_weight_prepacked = false);

// Shared matmul body. weight_sizes and weight_strides describe the weight as
// the [.., K, N] right-hand side of the matmul, so a caller can pass an [N, K]
// weight (nn.Linear layout) with its sizes and strides swapped, without
// building a transposed view.
torch::stable::Tensor zendnnl_matmul_impl(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::headeronly::IntHeaderOnlyArrayRef weight_sizes,
    const torch::headeronly::IntHeaderOnlyArrayRef weight_strides,
    const torch::stable::Tensor &bias, torch::stable::Tensor &result,
    const std::vector<int64_t> &post_op_ids,
    const std::vector<torch::stable::Tensor> &post_op_buffers,
    const float &beta, const float &alpha, std::string zentorch_op_name,
    const bool is_weight_const, const bool is_weight_prepacked);

std::string show_config();

void clear_zendnn_weight_caches();
} // namespace zentorch
