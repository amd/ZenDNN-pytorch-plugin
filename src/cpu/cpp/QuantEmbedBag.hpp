/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <ATen/ATen.h>
#include <ATen/core/List.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <string>
#include <vector>

namespace zentorch {

at::Tensor zendnnl_quant_embedding_bag(
    const at::Tensor &weight, const at::Tensor &indices,
    const at::Tensor &offsets, int64_t num_bits_per_weight,
    c10::ScalarType output_dtype, bool scale_grad_by_freq, int64_t mode,
    bool sparse, c10::optional<at::Tensor> per_sample_weights_opt,
    bool include_last_offset, int64_t padding_idx,
    std::string zentorch_op_name);

void zendnnl_quant_embedding_bag_out(
    const at::Tensor &output, const at::Tensor &weight,
    const at::Tensor &indices, const at::Tensor &offsets,
    int64_t num_bits_per_weight, c10::ScalarType output_dtype,
    bool scale_grad_by_freq, int64_t mode, bool sparse,
    c10::optional<at::Tensor> per_sample_weights_opt, bool include_last_offset,
    int64_t padding_idx, std::string zentorch_op_name);

std::vector<at::Tensor> zendnnl_horizontal_quant_embedding_bag_group_impl(
    at::TensorList weight, at::TensorList indices, at::TensorList offsets,
    int64_t num_bits_per_weight, c10::ScalarType output_dtype,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef mode,
    at::IntArrayRef sparse,
    c10::List<c10::optional<at::Tensor>> per_sample_weights_opt,
    at::IntArrayRef include_last_offset, at::IntArrayRef padding_idx,
    std::string zentorch_op_name);

void zendnnl_horizontal_quant_embedding_bag_group_out(
    at::TensorList outputs, at::TensorList weight, at::TensorList indices,
    at::TensorList offsets, int64_t num_bits_per_weight,
    c10::ScalarType output_dtype, at::IntArrayRef scale_grad_by_freq,
    at::IntArrayRef mode, at::IntArrayRef sparse,
    c10::List<c10::optional<at::Tensor>> per_sample_weights_opt,
    at::IntArrayRef include_last_offset, at::IntArrayRef padding_idx,
    std::string zentorch_op_name);

} // namespace zentorch
