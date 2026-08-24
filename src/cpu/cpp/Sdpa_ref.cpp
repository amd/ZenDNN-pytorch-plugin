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
  fp.kv_num_heads = key.size(1);
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
    std::optional<double> scale, at::Tensor &output) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
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
  ZENTORCH_CHECK(output.scalar_type() == dtype,
                 "zentorch_sdpa.out: output should have the same data type as "
                 "Query, but got ",
                 output.scalar_type(), " instead.");
  ZENTORCH_CHECK(output.sizes() == query.sizes(),
                 "zentorch_sdpa.out: output should have the same shape as "
                 "Query, expected ",
                 query.sizes(), " but got ", output.sizes());
  // Input validation for tensor types, shapes,attention mask and AVX512
  // support.
  bool is_dtype_supported =
      (dtype == at::kBFloat16 && zendnn_bf16_device_check()) ||
      (dtype == at::kHalf && zendnn_fp16_device_check()) ||
      (dtype == at::kFloat && is_avx512_supported());
  if (is_dtype_supported) {
    // `output` may alias a caller-owned buffer (zentorch_sdpa.out), so the
    // kernels have to store into it rather than into a temporary the caller
    // never sees. They emit BSHD through stride(0/1/2), so a transposed view
    // is all they need; only a strided head dim forces a staging copy, which
    // is written back below.
    at::Tensor caller_out = output;
    output = output.transpose(1, 2);

    const auto accumulate_dtype = at::toOpMathType(dtype);
    at::Tensor logsumexp = at::empty({batchSize, qSize, num_head},
                                     query.options().dtype(accumulate_dtype));
    // Assuming key and value have the same dtype as query

    const int int_env_value =
        EnvReader::getEnvVariableAsInt("ZENTORCH_USE_ZENDNN_SDPA");
    const bool requires_lse =
        at::GradMode::is_enabled() &&
        (query.requires_grad() || key.requires_grad() || value.requires_grad());

    // Both downstream kernels (in-plugin AVX-512 flash and ZenDNN direct)
    // consume the mask via vectorized loads on its innermost (KV) axis and
    // require stride(-1) == 1. Normalize here for callers that do not
    // satisfy this (e.g. T5 cross-attention); a no-op when already so.
    if (attn_mask.has_value() && attn_mask->defined() &&
        attn_mask->stride(-1) != 1) {
      attn_mask = attn_mask->contiguous();
    }
    const bool use_zendnnl_direct_sdpa = (int_env_value == 1) && !requires_lse;
    if (use_zendnnl_direct_sdpa) {
      // ZenDNN flash SDPA is inference-only and does not compute logsumexp.
      // We bypass this path when autograd is engaged.
      zendnnl_sdpa_direct_kernel(query, key, value, output, dropout_p,
                                 is_causal, attn_mask, scale);
    } else {
      at::Tensor output_contiguous = output.contiguous();

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
              output_contiguous, logsumexp, query, key, value, dropout_p,
              is_causal, attn_mask, scale);
        } else {
          flash_attention_kernel_impl_512<at::BFloat16, at::BFloat16>(
              output_contiguous, logsumexp, query, key, value, dropout_p,
              is_causal, attn_mask, scale);
        }
      } else if (query.scalar_type() == at::kHalf) {
        ZENTORCH_CHECK(!attn_mask.has_value() ||
                           attn_mask.value().scalar_type() == at::kFloat ||
                           attn_mask.value().scalar_type() == at::kHalf,
                       "zentorch_scaled_dot_product_attention_flash_"
                       "attention: Attention mask "
                       "is supported for FP32 and FP16 dtype when the query "
                       "is of type FP16");
        if (!attn_mask.has_value() ||
            attn_mask.value().scalar_type() == at::kFloat) {
          flash_attention_kernel_impl_512<at::Half, float>(
              output_contiguous, logsumexp, query, key, value, dropout_p,
              is_causal, attn_mask, scale);
        } else {
          flash_attention_kernel_impl_512<at::Half, at::Half>(
              output_contiguous, logsumexp, query, key, value, dropout_p,
              is_causal, attn_mask, scale);
        }
      } else {
        ZENTORCH_CHECK(
            !attn_mask.has_value() ||
                attn_mask.value().scalar_type() == at::kFloat,
            "zentorch_scaled_dot_product_attention_flash_"
            "attention: Attention mask "
            "is supported for FP32 dtype when the query is of type FP32");
        flash_attention_kernel_impl_512<float, float>(
            output_contiguous, logsumexp, query, key, value, dropout_p,
            is_causal, attn_mask, scale);
      }
      if (!output_contiguous.is_alias_of(caller_out)) {
        caller_out.copy_(output_contiguous.transpose(1, 2));
      }
    }

    logsumexp = logsumexp.transpose(1, 2);

    return std::make_tuple(std::move(caller_out), std::move(logsumexp));
  } else {
    // at::_scaled_dot_product_flash_attention_for_cpu does an extra .contiguous
    // on the query tensor while we process the query as is in meta registration
    // and bf16 impl. Leading to a mismatch in stride between meta output and
    // runtime output.
    // Hence using native - same as ipex.
    auto native_result = at::native::_scaled_dot_product_flash_attention_cpu(
        query, key, value, dropout_p, is_causal, attn_mask, scale);
    output.copy_(std::get<0>(native_result));
    return std::make_tuple(output, std::get<1>(native_result));
  }
}

// Out variant: writes the attention result directly into a caller-owned
// buffer. Lets callers that already hold a destination (e.g. the vLLM CPU
// attention backend, which is handed a pre-allocated output tensor) skip the
// full-size copy that the allocating variant forces on them. logsumexp is
// still computed as kernel scratch but dropped, since the op returns ().
void zentorch_sdpa_out(const at::Tensor &query, const at::Tensor &key,
                       const at::Tensor &value, double dropout_p,
                       bool is_causal, std::optional<at::Tensor> attn_mask,
                       std::optional<double> scale,
                       std::string zentorch_op_name, at::Tensor &out) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  zentorch_scaled_dot_product_attention_impl(query, key, value, dropout_p,
                                             is_causal, std::move(attn_mask),
                                             scale, out);
}

std::tuple<at::Tensor, at::Tensor>
zentorch_sdpa(const at::Tensor &query, const at::Tensor &key,
              const at::Tensor &value, double dropout_p, bool is_causal,
              std::optional<at::Tensor> attn_mask, std::optional<double> scale,
              std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  at::Tensor output = at::empty_like(query, query.options());
  return zentorch_scaled_dot_product_attention_impl(
      query, key, value, dropout_p, is_causal, std::move(attn_mask), scale,
      output);
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_sdpa(Tensor query, Tensor key, "
        "Tensor value , float dropout_p=0.0, "
        "bool is_causal=False, *, Tensor? attn_mask=None, float? scale=None, "
        "str zentorch_op_name = "
        "'zentorch::zentorch_sdpa')-> (Tensor, Tensor)");
  m.def("zentorch_sdpa.out(Tensor query, Tensor key, "
        "Tensor value , float dropout_p=0.0, "
        "bool is_causal=False, *, Tensor? attn_mask=None, float? scale=None, "
        "str zentorch_op_name = "
        "'zentorch::zentorch_sdpa.out', Tensor(a!) out)-> ()");
}
TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_sdpa", zentorch_sdpa);
  m.impl("zentorch_sdpa.out", zentorch_sdpa_out);
}
} // namespace zentorch
