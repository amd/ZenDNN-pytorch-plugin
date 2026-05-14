/******************************************************************************
 * Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Was sourced from
 * https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp
 * PyTorch commit ID: d990dad
 ******************************************************************************/

#include "EnvReader.hpp"
#include "Memory.hpp"
#include "Utils.hpp"
#include "kernels/zen_cpukernels.hpp"
#include <ATen/ATen.h>
#include <ATen/OpMathType.h>

namespace zentorch {

// Wrapper around zendnnl::lowoha::sdpa::sdpa_direct. Builds the sdpa_params
// struct from the input tensors and invokes the direct kernel.
inline void
zendnnl_sdpa_direct_kernel(const at::Tensor &query, const at::Tensor &key,
                           const at::Tensor &value, at::Tensor &output,
                           const double dropout_p, const bool is_causal,
                           const std::optional<at::Tensor> &attn_mask,
                           const std::optional<double> &scale) {
  zendnnl::lowoha::sdpa::sdpa_params fp{};
  fp.batch = query.size(0);
  fp.num_heads = query.size(1);
  fp.seq_len = query.size(2);
  fp.kv_seq_len = key.size(2);
  fp.head_dim = query.size(3);

  fp.q_stride_b = query.stride(0);
  fp.q_stride_h = query.stride(1);
  fp.q_stride_s = query.stride(2);
  fp.q_stride_d = query.stride(3);
  fp.k_stride_b = key.stride(0);
  fp.k_stride_h = key.stride(1);
  fp.k_stride_s = key.stride(2);
  fp.k_stride_d = key.stride(3);
  fp.v_stride_b = value.stride(0);
  fp.v_stride_h = value.stride(1);
  fp.v_stride_s = value.stride(2);
  fp.v_stride_d = value.stride(3);
  // output layout is BSHD after transpose(1,2)
  fp.o_stride_b = output.stride(0);
  fp.o_stride_s = output.stride(1);
  fp.o_stride_h = output.stride(2);
  fp.o_stride_d = output.stride(3);

  fp.qkv_dt = get_zendnnl_dtype(query);
  fp.out_dt = get_zendnnl_dtype(query);
  fp.scale = scale.value_or(1.0 / std::sqrt(static_cast<double>(fp.head_dim)));
  fp.is_causal = is_causal;
  fp.dropout_p = dropout_p;

  const void *mask_ptr = nullptr;
  if (attn_mask.has_value() && attn_mask->defined()) {
    const at::Tensor &mask = attn_mask.value();
    mask_ptr = mask.data_ptr();
    fp.mask_ndims = mask.dim();
    fp.mask_dt = get_zendnnl_dtype(mask);
    for (int i = 0; i < mask.dim(); ++i) {
      fp.mask_sizes[i] = mask.size(i);
      fp.mask_strides[i] = mask.stride(i);
    }
  }

  ZENTORCH_CHECK(zendnnl::lowoha::sdpa::sdpa_direct(
                     query.data_ptr(), key.data_ptr(), value.data_ptr(),
                     mask_ptr, output.data_ptr(), fp) == status_t::success,
                 "zentorch_sdpa: sdpa_direct failed");
}

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

    const int int_env_value =
        EnvReader::getEnvVariableAsInt("ZENTORCH_USE_ZENDNN_SDPA");
    const bool requires_lse =
        at::GradMode::is_enabled() &&
        (query.requires_grad() || key.requires_grad() || value.requires_grad());

    const bool use_zendnnl_direct_sdpa = (int_env_value == 1) && !requires_lse;
    if (use_zendnnl_direct_sdpa) {
      // ZenDNN flash SDPA is inference-only and does not compute logsumexp.
      // We bypass this path when autograd is engaged (see
      // use_zendnnl_direct_sdpa above).
      zendnnl_sdpa_direct_kernel(query, key, value, output, dropout_p,
                                 is_causal, attn_mask, scale);
    } else if (query.scalar_type() == at::kBFloat16) {
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

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_sdpa(Tensor query, Tensor key, "
        "Tensor value , float dropout_p=0.0, "
        "bool is_causal=False, *, Tensor? attn_mask=None, float? scale=None, "
        "str zentorch_op_name = "
        "'zentorch::zentorch_sdpa')-> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_sdpa", zentorch::zentorch_scaled_dot_product_attention_impl);
}
} // namespace zentorch
