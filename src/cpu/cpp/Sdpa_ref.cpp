/******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Was sourced from
 * https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp
 * PyTorch commit ID: d990dad
 ******************************************************************************/

#include "Memory.hpp"
#include "Utils.hpp"
#include "kernels/zen_cpukernels.hpp"
#include <ATen/ATen.h>
#include <ATen/OpMathType.h>

namespace zentorch {

#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR > 3
std::tuple<at::Tensor, at::Tensor> zentorch_scaled_dot_product_attention_impl(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    double dropout_p, bool is_causal, std::optional<at::Tensor> attn_mask,
    std::optional<double> scale, std::string zentorch_op_name) {
  const auto dtype = query.scalar_type();
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(2);
  int64_t num_head = query.size(1);

  ZENTORCH_CHECK(
      c10::isFloatingType(dtype),
      "zentorch_scaled_dot_product_attention_flash_attention: Expected data "
      "type in FP32, FP64, BF16, FP16, but got ",
      dtype, " instead.");
  ZENTORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "zentorch_scaled_dot_product_attention_flash_attention: Accept only 4 "
      "dims inputs shape of {B, H, T, K}");
  ZENTORCH_CHECK(
      dropout_p == 0.0,
      "zentorch_scaled_dot_product_attention_flash_attention: Currently do "
      "not support dropout > 0");
  ZENTORCH_CHECK(
      (query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
      "zentorch_scaled_dot_product_attention_flash_attention: Q/K/V should "
      "have the same head size");
  ZENTORCH_CHECK(!attn_mask.has_value() ||
                     attn_mask.value().scalar_type() == at::kFloat ||
                     dtype == attn_mask.value().scalar_type(),
                 "zentorch_scaled_dot_product_attention_flash_attention: "
                 "Attention Mask should have the same data type as Query");
  ZENTORCH_CHECK(
      !attn_mask.has_value() ||
          (attn_mask.value().dim() == 2 || attn_mask.value().dim() == 4),
      "zentorch_scaled_dot_product_attention_flash_attention: Attention mask "
      "dim is {2, 4}");
  // Input validation for tensor types, shapes,attention mask and AVX512
  // support.
  if ((dtype == at::kBFloat16 || dtype == at::kFloat) &&
      is_avx512_supported()) {
    at::Tensor output = at::empty_like(query, query.options()).transpose(1, 2);
    const auto accumulate_dtype = at::toOpMathType(dtype);
    at::Tensor logsumexp = at::empty({batchSize, qSize, num_head},
                                     query.options().dtype(accumulate_dtype));
    // Assuming key and value have the same dtype as query
    if (query.scalar_type() == at::kBFloat16) {
      ZENTORCH_CHECK(!attn_mask.has_value() ||
                         attn_mask.value().scalar_type() == at::kFloat ||
                         attn_mask.value().scalar_type() == at::kBFloat16,
                     "zentorch_scaled_dot_product_attention_flash_"
                     "attention: Attention mask "
                     "is supported for FP32 and BF16 dtype when the query "
                     "is of type BF16");
      // passing type as float when attention mask is None or float
      if (!attn_mask.has_value() ||
          attn_mask.value().scalar_type() == at::kFloat) {
        flash_attention_kernel_impl_512<at::BFloat16, float>(
            output, logsumexp, query, key, value, dropout_p, is_causal,
            attn_mask, scale);
      } else {
        flash_attention_kernel_impl_512<at::BFloat16, at::BFloat16>(
            output, logsumexp, query, key, value, dropout_p, is_causal,
            attn_mask, scale);
      }
    } else {
      ZENTORCH_CHECK(
          !attn_mask.has_value() ||
              attn_mask.value().scalar_type() == at::kFloat,
          "zentorch_scaled_dot_product_attention_flash_"
          "attention: Attention mask "
          "is supported for FP32 dtype when the query is of type FP32");
      flash_attention_kernel_impl_512<float, float>(
          output, logsumexp, query, key, value, dropout_p, is_causal, attn_mask,
          scale);
    }

    output = output.transpose(1, 2);
    logsumexp = logsumexp.transpose(1, 2);

    return std::make_tuple(std::move(output), std::move(logsumexp));
  } else {
    // at::_scaled_dot_product_flash_attention_for_cpu does an extra .contiguous
    // on the query tensor while we process the query as is in meta registration
    // and bf16 impl. Leading to a mismatch in stride between meta output and
    // runtime output.
    // Hence using native - same as ipex.
    return (at::native::_scaled_dot_product_flash_attention_cpu(
        query, key, value, dropout_p, is_causal, attn_mask, scale));
  }
}
#endif // TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR > 3

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
// zentorch_sdpa is supported from torch version >= 2.4.0
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR > 3
  m.def("zentorch_sdpa(Tensor query, Tensor key, "
        "Tensor value , float dropout_p=0.0, "
        "bool is_causal=False, *, Tensor? attn_mask=None, float? scale=None, "
        "str zentorch_op_name = "
        "'zentorch::zentorch_sdpa')-> (Tensor, Tensor)");
#endif // TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR > 3
}
TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
// zentorch_sdpa is supported from torch version >= 2.4.0
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR > 3
  m.impl("zentorch_sdpa", zentorch::zentorch_scaled_dot_product_attention_impl);
#endif // TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR > 3
}
} // namespace zentorch
