/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <c10/core/ScalarType.h>
#include <optional>
#include <string>
#include <torch/csrc/stable/tensor.h>
#include <vector>

namespace zentorch {

torch::stable::Tensor zendnnl_quant_embedding_bag(
    const torch::stable::Tensor &weight, const torch::stable::Tensor &indices,
    const torch::stable::Tensor &offsets, int64_t num_bits_per_weight,
    c10::ScalarType output_dtype, bool scale_grad_by_freq, int64_t mode,
    bool sparse,
    const std::optional<torch::stable::Tensor> &per_sample_weights_opt,
    bool include_last_offset, int64_t padding_idx,
    std::string zentorch_op_name);

void zendnnl_quant_embedding_bag_out(
    torch::stable::Tensor &output, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &indices, const torch::stable::Tensor &offsets,
    int64_t num_bits_per_weight, c10::ScalarType output_dtype,
    bool scale_grad_by_freq, int64_t mode, bool sparse,
    const std::optional<torch::stable::Tensor> &per_sample_weights_opt,
    bool include_last_offset, int64_t padding_idx,
    std::string zentorch_op_name);

std::vector<torch::stable::Tensor>
zendnnl_horizontal_quant_embedding_bag_group_impl(
    const std::vector<torch::stable::Tensor> &weight,
    const std::vector<torch::stable::Tensor> &indices,
    const std::vector<torch::stable::Tensor> &offsets,
    int64_t num_bits_per_weight, c10::ScalarType output_dtype,
    const std::vector<int64_t> &scale_grad_by_freq,
    const std::vector<int64_t> &mode, const std::vector<int64_t> &sparse,
    const std::vector<std::optional<torch::stable::Tensor>>
        &per_sample_weights_opt,
    const std::vector<int64_t> &include_last_offset,
    const std::vector<int64_t> &padding_idx, std::string zentorch_op_name);

void zendnnl_horizontal_quant_embedding_bag_group_out(
    const std::vector<torch::stable::Tensor> &outputs,
    const std::vector<torch::stable::Tensor> &weight,
    const std::vector<torch::stable::Tensor> &indices,
    const std::vector<torch::stable::Tensor> &offsets,
    int64_t num_bits_per_weight, c10::ScalarType output_dtype,
    const std::vector<int64_t> &scale_grad_by_freq,
    const std::vector<int64_t> &mode, const std::vector<int64_t> &sparse,
    const std::vector<std::optional<torch::stable::Tensor>>
        &per_sample_weights_opt,
    const std::vector<int64_t> &include_last_offset,
    const std::vector<int64_t> &padding_idx, std::string zentorch_op_name);

torch::stable::Tensor zendnnl_get_packed_embedding_weight(
    const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const torch::stable::Tensor &weight_zero_points);

} // namespace zentorch
