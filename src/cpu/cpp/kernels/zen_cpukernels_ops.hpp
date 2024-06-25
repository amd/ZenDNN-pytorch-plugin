/******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

namespace zentorch {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
zentorch_masked_multihead_self_attention_impl(
    at::Tensor &query, at::Tensor &key, at::Tensor &value,
    at::Tensor &key_cache, at::Tensor &value_cache, at::Tensor &beam_idx,
    at::Tensor seq_info, const double scale_attn, int64_t max_positions,
    const c10::optional<at::Tensor> &head_mask /* optional */,
    const c10::optional<at::Tensor> &attention_mask /* optional */,
    c10::optional<bool> add_casual_mask,
    std::string zentorch_op_name /* optional */);
} // namespace zentorch