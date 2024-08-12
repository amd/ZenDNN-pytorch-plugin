/******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

namespace zentorch {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
masked_multihead_self_attention_kernel_impl_512(
    at::Tensor &query, at::Tensor &key, at::Tensor &value,
    at::Tensor &key_cache, at::Tensor &value_cache, at::Tensor &beam_idx,
    at::Tensor seq_info, const double scale_attn, int64_t max_positions,
    const c10::optional<at::Tensor> &head_mask /* optional */,
    const c10::optional<at::Tensor> &attention_mask /* optional */,
    c10::optional<bool> add_casual_mask /* optional */);

void flash_attention_kernel_impl_512(
    const at::Tensor &output, const at::Tensor &logsumexp,
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    double dropout_p, bool is_causal, std::optional<at::Tensor> attn_mask,
    std::optional<double> scale);
} // namespace zentorch