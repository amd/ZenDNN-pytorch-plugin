/******************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

// Declarations for ZenTorchOps (EmbedBag etc.)

#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/stable/tensor.h>

namespace zentorch {

// TODO(stable-abi): Remove this ATen overload once Matmul.cpp (mm/bmm/addmm
// etc.) is migrated to stable ABI; keep only the stable declaration below.
at::Tensor zentorch_matmul_impl(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
    at::Tensor &self_or_result, const std::vector<int64_t> &post_op_ids,
    const std::vector<at::Tensor> &post_op_buffers, const float &beta,
    const float &alpha, std::string zentorch_op_name,
    const bool is_const = true, const bool is_weight_prepacked = false);

torch::stable::Tensor zentorch_matmul_impl(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &bias, torch::stable::Tensor &self_or_result,
    const std::vector<int64_t> &post_op_ids,
    const std::vector<torch::stable::Tensor> &post_op_buffers,
    const float &beta, const float &alpha, std::string zentorch_op_name,
    const bool is_const = true, const bool is_weight_prepacked = false);

std::string show_config();

void clear_zendnn_weight_caches();
} // namespace zentorch
