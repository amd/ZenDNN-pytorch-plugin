/******************************************************************************
 * Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include <string>
#include <string_view>

namespace zentorch {

template <typename input_type, typename attention_mask>
void flash_attention_kernel_impl_512(
    const at::Tensor &output, const at::Tensor &logsumexp,
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    double dropout_p, bool is_causal, std::optional<at::Tensor> attn_mask,
    std::optional<double> scale);

} // namespace zentorch
