/******************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
    const at::Tensor &offsets, const bool scale_grad_by_freq, int64_t mode,
    bool sparse, const c10::optional<at::Tensor> &per_sample_weights_opt,
    bool include_last_offset, int64_t padding_idx);

std::string show_config();

at::Tensor zendnn_matmul_impl(const at::Tensor &mat1, const at::Tensor &mat2,
                              at::Tensor &self_or_result, const float &beta,
                              const float &alpha, const bool &fuse_relu);

at::Tensor zendnn_addmm(const at::Tensor &self, const at::Tensor &mat1,
                        const at::Tensor &mat2, const at::Scalar &beta,
                        const at::Scalar &alpha, const bool &fuse_relu);

at::Tensor zendnn_baddbmm(const at::Tensor &self, const at::Tensor &batch1,
                          const at::Tensor &batch2, const at::Scalar &beta,
                          const at::Scalar &alpha);

at ::Tensor zendnn_mm(const at::Tensor &self, const at::Tensor &mat2,
                      const bool &fuse_relu);

at::Tensor zendnn_bmm(const at::Tensor &self, const at::Tensor &mat2);

} // namespace ZenDNNTorch
