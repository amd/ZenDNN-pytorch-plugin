/******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include <string>
#include <string_view>

namespace zentorch {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
masked_multihead_self_attention_kernel_impl_512(
    at::Tensor &query, at::Tensor &key, at::Tensor &value,
    at::Tensor &key_cache, at::Tensor &value_cache, at::Tensor &beam_idx,
    at::Tensor seq_info, const double scale_attn, int64_t max_positions,
    const c10::optional<at::Tensor> &head_mask /* optional */,
    const c10::optional<at::Tensor> &attention_mask /* optional */,
    c10::optional<bool> add_causal_mask /* optional */);

template <typename input_type, typename attention_mask>
void flash_attention_kernel_impl_512(
    const at::Tensor &output, const at::Tensor &logsumexp,
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    double dropout_p, bool is_causal, std::optional<at::Tensor> attn_mask,
    std::optional<double> scale);

void zentorch_attention_reshape_and_cache_cpu_kernel_512(
    const at::Tensor &key, const at::Tensor &value, const at::Tensor &key_cache,
    const at::Tensor &value_cache, const at::Tensor &slot_mapping,
    std::string zentorch_op_name /* optional */);

at::Tensor zentorch_attention_single_query_cached_kv_attention_kernel_512(
    const at::Tensor &out, const at::Tensor &query, const at::Tensor &key_cache,
    const at::Tensor &value_cache, const at::Tensor &head_mapping,
    at::Scalar scale, const at::Tensor &block_tables,
    const at::Tensor &context_lens, at::Scalar block_size,
    at::Scalar max_context_len, const c10::optional<at::Tensor> &alibi_slopes,
    std::string zentorch_op_name /* optional */);

void zentorch_attention_flash_attn_varlen_kernel_512(
    const at::Tensor &out, const at::Tensor &query, const at::Tensor &key_cache,
    const at::Tensor &value_cache, const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_k, int64_t max_seqlen_q, int64_t max_seqlens_k,
    double softmax_scale, bool is_causal, const at::Tensor &block_table,
    const c10::optional<at::Tensor> &alibi_slopes, int64_t window_size_left,
    int64_t window_size_right, const std::string_view &kv_cache_dtype,
    double k_scale, double v_scale, double softcap,
    std::string zentorch_op_name /* optional */);

} // namespace zentorch
