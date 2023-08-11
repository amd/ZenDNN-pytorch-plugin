/******************************************************************************
* Copyright (c) 2023 Advanced Micro Devices, Inc.
* All rights reserved.
******************************************************************************/

// Declarations for ZenDNNOps (EmbedBag etc.)

#pragma once

#include <torch/extension.h>
// needs to be included only once in library.
#include "ZenDNNSingletons.hpp"

namespace ZenDNNTorch{

//EmbedBag
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_embedding_bag_zendnn_impl(const at::Tensor &weight, const at::Tensor &indices,
                      const at::Tensor &offsets, const bool scale_grad_by_freq,
                      int64_t mode, bool sparse,
                      const c10::optional<at::Tensor>& per_sample_weights_opt,
                      bool include_last_offset, int64_t padding_idx);
std::string show_config();
}
