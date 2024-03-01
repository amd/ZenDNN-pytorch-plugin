/******************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

// Declarations for ZenDNNOps (EmbedBag etc.)

#pragma once

#include <torch/extension.h>
// needs to be included only once in library.
#include "ZenDNNSingletons.hpp"

namespace ZenDNNTorch {

// EmbedBag
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
zendnn_embedding_bag_impl(
    const at::Tensor &weight, const at::Tensor &indices,
    const at::Tensor &offsets, const bool &scale_grad_by_freq,
    const int64_t &mode, const bool &sparse,
    const c10::optional<at::Tensor> &per_sample_weights_opt,
    const bool &include_last_offset, const int64_t &padding_idx);

at::Tensor zendnn_embedding_impl(const at::Tensor &weight,
                                 const at::Tensor &indices,
                                 const int64_t &padding_idx,
                                 const bool &scale_grad_by_freq,
                                 const bool &sparse);

std::string show_config();

at::Tensor zendnn_matmul_impl(const at::Tensor &mat1, const at::Tensor &mat2,
                              const at::Tensor &bias,
                              at::Tensor &self_or_result, const float &beta,
                              const float &alpha, const int64_t &fuse);

at::Tensor zendnn_addmm(const at::Tensor &self, const at::Tensor &mat1,
                        const at::Tensor &mat2, const at::Scalar &beta,
                        const at::Scalar &alpha, const int64_t &fuse);

// for 1d bias
at::Tensor zendnn_addmm_1dbias(const at::Tensor &self, const at::Tensor &mat1,
                               const at::Tensor &mat2, const at::Scalar &beta,
                               const at::Scalar &alpha, const int64_t &fuse);

at::Tensor zendnn_baddbmm(const at::Tensor &self, const at::Tensor &batch1,
                          const at::Tensor &batch2, const at::Scalar &beta,
                          const at::Scalar &alpha);

at ::Tensor zendnn_mm(const at::Tensor &self, const at::Tensor &mat2,
                      const int64_t &fuse);

at::Tensor zendnn_bmm(const at::Tensor &self, const at::Tensor &mat2);

std::vector<at::Tensor> zendnn_horizontal_embedding_bag_group(
    const at::TensorList &weight, const at::TensorList &indices,
    const at::TensorList &offsets, const at::IntArrayRef &scale_grad_by_freq,
    const at::IntArrayRef &mode, const at::IntArrayRef &sparse,
    const c10::List<c10::optional<at::Tensor>> &per_sample_weights_opt,
    const at::IntArrayRef &include_last_offset,
    const at::IntArrayRef &padding_idx);

std::vector<at::Tensor> zendnn_horizontal_embedding_group(
    const at::TensorList &weight, const at::TensorList &indices,
    const at::IntArrayRef &padding_idx,
    const at::IntArrayRef &scale_grad_by_freq, const at::IntArrayRef &sparse);

at::Tensor zendnn_vertical_mlp_group(const at::TensorList &self,
                                     const at::Tensor &input,
                                     const at::TensorList &weights,
                                     const at::ArrayRef<double> &betas,
                                     const at::ArrayRef<double> &alphas,
                                     const at::IntArrayRef &fuse);

std::vector<at::Tensor> zendnn_attn_horizontal_mlp_group(
    const at::TensorList &self, const at::TensorList &inputs,
    const at::TensorList &weights, const at::ArrayRef<double> &betas,
    const at::ArrayRef<double> &alphas, const at::IntArrayRef &fuse,
    const at::IntArrayRef &is_zendnnmm);

std::vector<at::Tensor> zendnn_fused_eb_mlp(
    const at::TensorList &eb_weight, const at::TensorList &eb_indices,
    const at::TensorList &eb_offsets,
    const at::IntArrayRef &eb_scale_grad_by_freq,
    const at::IntArrayRef &eb_mode, const at::IntArrayRef &eb_sparse,
    const c10::List<c10::optional<at::Tensor>> &eb_per_sample_weights_opt,
    const at::IntArrayRef &eb_include_last_offset,
    const at::IntArrayRef &eb_padding_idx, const at::TensorList &mlp_self,
    const at::Tensor &mlp_inputs, const at::TensorList &mlp_weight,
    const at::ArrayRef<double> &mlp_betas,
    const at::ArrayRef<double> &mlp_alphas, const at::IntArrayRef &mlp_fuse);
} // namespace ZenDNNTorch
