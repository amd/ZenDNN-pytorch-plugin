/******************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

// Declarations for ZenTorchOps (EmbedBag etc.)

#pragma once

#include <torch/extension.h>

namespace zentorch {

enum POST_OP { NONE, RELU, GELU_TANH, GELU_ERF, SILU, MUL, ADD };

// EmbedBag
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
zentorch_embedding_bag_impl(
    const at::Tensor &weight, const at::Tensor &indices,
    const at::Tensor &offsets, const bool &scale_grad_by_freq,
    const int64_t &mode, const bool &sparse,
    const c10::optional<at::Tensor> &per_sample_weights_opt,
    const bool &include_last_offset, const int64_t &padding_idx,
    std::string zentorch_op_name);

at::Tensor zentorch_embedding_impl(const at::Tensor &weight,
                                   const at::Tensor &indices,
                                   const int64_t &padding_idx,
                                   const bool &scale_grad_by_freq,
                                   const bool &sparse,
                                   std::string zentorch_op_name);

std::string show_config();

at::Tensor zentorch_matmul_impl(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
    at::Tensor &self_or_result, const std::vector<int64_t> &post_op_ids,
    const std::vector<at::Tensor> &post_op_buffers, const float &beta,
    const float &alpha, std::string zentorch_op_name);

template <POST_OP fuse = POST_OP::NONE>
at::Tensor zentorch_addmm(const at::Tensor &self, const at::Tensor &mat1,
                          const at::Tensor &mat2, const at::Scalar &beta,
                          const at::Scalar &alpha,
                          std::string zentorch_op_name);

// for 1d bias
template <POST_OP fuse = POST_OP::NONE>
at::Tensor zentorch_addmm_1dbias(const at::Tensor &self, const at::Tensor &mat1,
                                 const at::Tensor &mat2, const at::Scalar &beta,
                                 const at::Scalar &alpha,
                                 std::string zentorch_op_name);

at::Tensor zentorch_baddbmm(const at::Tensor &self, const at::Tensor &batch1,
                            const at::Tensor &batch2, const at::Scalar &beta,
                            const at::Scalar &alpha,
                            std::string zentorch_op_name);

template <POST_OP fuse = POST_OP::NONE>
at ::Tensor zentorch_mm(const at::Tensor &self, const at::Tensor &mat2,
                        std::string zentorch_op_name);

at::Tensor zentorch_bmm(const at::Tensor &self, const at::Tensor &mat2,
                        std::string zentorch_op_name);

at::Tensor zentorch_mm_silu_mul(const at::Tensor &mat1, const at::Tensor &mat2,
                                const at::Tensor &mat3,
                                std::string zentorch_op_name);

at::Tensor
zentorch_addmm_silu_mul(const at::Tensor &bias, const at::Tensor &mat1,
                        const at::Tensor &mat2, const at::Tensor &mat3,
                        const at::Scalar &beta, const at::Scalar &alpha,
                        std::string zentorch_op_name);

std::vector<at::Tensor> zentorch_horizontal_embedding_bag_group(
    const at::TensorList &weight, const at::TensorList &indices,
    const at::TensorList &offsets, const at::IntArrayRef &scale_grad_by_freq,
    const at::IntArrayRef &mode, const at::IntArrayRef &sparse,
    const c10::List<c10::optional<at::Tensor>> &per_sample_weights_opt,
    const at::IntArrayRef &include_last_offset,
    const at::IntArrayRef &padding_idx, std::string zentorch_op_name);

std::vector<at::Tensor> zentorch_horizontal_embedding_group(
    const at::TensorList &weight, const at::TensorList &indices,
    const at::IntArrayRef &padding_idx,
    const at::IntArrayRef &scale_grad_by_freq, const at::IntArrayRef &sparse,
    std::string zentorch_op_name);

at::Tensor zentorch_vertical_mlp_group(const at::TensorList &self,
                                       const at::Tensor &input,
                                       const at::TensorList &weights,
                                       const at::ArrayRef<double> &betas,
                                       const at::ArrayRef<double> &alphas,
                                       const at::IntArrayRef &fuse,
                                       std::string zentorch_op_name);

std::vector<at::Tensor> zentorch_attn_horizontal_mlp_group(
    const at::TensorList &self, const at::TensorList &inputs,
    const at::TensorList &weights, const at::ArrayRef<double> &betas,
    const at::ArrayRef<double> &alphas, const at::IntArrayRef &fuse,
    const at::IntArrayRef &is_zentorch_mm, std::string zentorch_op_name);

std::vector<at::Tensor> zentorch_fused_eb_mlp(
    const at::TensorList &eb_weight, const at::TensorList &eb_indices,
    const at::TensorList &eb_offsets,
    const at::IntArrayRef &eb_scale_grad_by_freq,
    const at::IntArrayRef &eb_mode, const at::IntArrayRef &eb_sparse,
    const c10::List<c10::optional<at::Tensor>> &eb_per_sample_weights_opt,
    const at::IntArrayRef &eb_include_last_offset,
    const at::IntArrayRef &eb_padding_idx, const at::TensorList &mlp_self,
    const at::Tensor &mlp_inputs, const at::TensorList &mlp_weight,
    const at::ArrayRef<double> &mlp_betas,
    const at::ArrayRef<double> &mlp_alphas, const at::IntArrayRef &mlp_fuse,
    std::string zentorch_op_name);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
zentorch_rope_impl(at::Tensor &t_in, at::Tensor &t_emb_pos, at::Tensor &t_pos,
                   int64_t N, int64_t H, int64_t offset, int64_t rotary_dim,
                   std::string zentorch_op_name);

} // namespace zentorch
