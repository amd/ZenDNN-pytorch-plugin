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

#include <ATen/OpMathType.h>
#include <c10/util/StringUtil.h>
#include <cmath>
#include <optional>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <tuple>

namespace zentorch {

// Full-lib only: the AVX-512 flash kernel and the native ATen fallback still
// take at::Tensor. Bridge without copying storage (refcount bump only).
inline at::Tensor to_aten(const torch::stable::Tensor &t) {
  return *torch::aot_inductor::tensor_handle_to_tensor_pointer(t.get());
}

inline torch::stable::Tensor to_stable(const at::Tensor &t) {
  return torch::stable::Tensor(
      torch::aot_inductor::new_tensor_handle(at::Tensor(t)));
}

inline std::optional<at::Tensor>
to_aten_optional(const std::optional<torch::stable::Tensor> &opt) {
  if (opt.has_value() && opt->defined()) {
    return to_aten(*opt);
  }
  return std::nullopt;
}

// Wrapper around zendnnl::lowoha::sdpa::sdpa_direct. Builds the sdpa_params
// struct from the input tensors and invokes the direct kernel.
inline void zendnnl_sdpa_direct_kernel(
    const torch::stable::Tensor &query, const torch::stable::Tensor &key,
    const torch::stable::Tensor &value, torch::stable::Tensor &output,
    const double dropout_p, const bool is_causal,
    const std::optional<torch::stable::Tensor> &attn_mask,
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
    const torch::stable::Tensor &mask = *attn_mask;
    mask_ptr = mask.data_ptr();
    fp.mask_ndims = static_cast<int>(mask.dim());
    fp.mask_dt = get_zendnnl_dtype(mask);
    for (int64_t i = 0; i < mask.dim(); ++i) {
      fp.mask_sizes[i] = mask.size(i);
      fp.mask_strides[i] = mask.stride(i);
    }
  }

  ZENTORCH_CHECK(zendnnl::lowoha::sdpa::sdpa_direct(
                     query.data_ptr(), key.data_ptr(), value.data_ptr(),
                     mask_ptr, output.data_ptr(), fp) == status_t::success,
                 "zentorch_sdpa: sdpa_direct failed");
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor>
zentorch_scaled_dot_product_attention_impl(
    const torch::stable::Tensor &query, const torch::stable::Tensor &key,
    const torch::stable::Tensor &value, double dropout_p, bool is_causal,
    std::optional<torch::stable::Tensor> attn_mask, std::optional<double> scale,
    torch::stable::Tensor &output) {
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
                     attn_mask->scalar_type() == at::kFloat ||
                     dtype == attn_mask->scalar_type(),
                 "zentorch_scaled_dot_product_attention_flash_attention: "
                 "Attention Mask should have the same data type as Query");
  ZENTORCH_CHECK(
      !attn_mask.has_value() ||
          (attn_mask->dim() == 2 || attn_mask->dim() == 4),
      "zentorch_scaled_dot_product_attention_flash_attention: Attention mask "
      "dim is {2, 4}");
  ZENTORCH_CHECK(output.scalar_type() == dtype,
                 "zentorch_sdpa.out: output should have the same data type as "
                 "Query, but got ",
                 output.scalar_type(), " instead.");
  ZENTORCH_CHECK(output.sizes().equals(query.sizes()),
                 "zentorch_sdpa.out: output should have the same shape as "
                 "Query, expected [",
                 c10::Join(", ", query.sizes()), "] but got [",
                 c10::Join(", ", output.sizes()), "]");
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
    torch::stable::Tensor caller_out = output;
    output = torch::stable::transpose(output, 1, 2);

    const auto accumulate_dtype = at::toOpMathType(dtype);
    torch::stable::Tensor logsumexp = torch::stable::new_empty(
        query, {batchSize, qSize, num_head}, accumulate_dtype);

    const int int_env_value =
        EnvReader::getEnvVariableAsInt("ZENTORCH_USE_ZENDNN_SDPA");
    const bool requires_lse =
        at::GradMode::is_enabled() &&
        (to_aten(query).requires_grad() || to_aten(key).requires_grad() ||
         to_aten(value).requires_grad());

    // Both downstream kernels (in-plugin AVX-512 flash and ZenDNN direct)
    // consume the mask via vectorized loads on its innermost (KV) axis and
    // require stride(-1) == 1. Normalize here for callers that do not
    // satisfy this (e.g. T5 cross-attention); a no-op when already so.
    if (attn_mask.has_value() && attn_mask->defined() &&
        attn_mask->stride(attn_mask->dim() - 1) != 1) {
      attn_mask = torch::stable::contiguous(*attn_mask);
    }
    const bool use_zendnnl_direct_sdpa = (int_env_value == 1) && !requires_lse;
    if (use_zendnnl_direct_sdpa) {
      // ZenDNN flash SDPA is inference-only and does not compute logsumexp.
      // We bypass this path when autograd is engaged.
      zendnnl_sdpa_direct_kernel(query, key, value, output, dropout_p,
                                 is_causal, attn_mask, scale);
    } else {
      torch::stable::Tensor output_contiguous =
          torch::stable::contiguous(output);

      std::optional<at::Tensor> attn_mask_aten = to_aten_optional(attn_mask);
      at::Tensor output_contiguous_aten = to_aten(output_contiguous);
      at::Tensor logsumexp_aten = to_aten(logsumexp);
      at::Tensor query_aten = to_aten(query);
      at::Tensor key_aten = to_aten(key);
      at::Tensor value_aten = to_aten(value);

      if (query.scalar_type() == at::kBFloat16) {
        ZENTORCH_CHECK(!attn_mask.has_value() ||
                           attn_mask->scalar_type() == at::kFloat ||
                           attn_mask->scalar_type() == at::kBFloat16,
                       "zentorch_scaled_dot_product_attention_flash_"
                       "attention: Attention mask "
                       "is supported for FP32 and BF16 dtype when the query "
                       "is of type BF16");
        // passing type as float when attention mask is None or float
        if (!attn_mask.has_value() || attn_mask->scalar_type() == at::kFloat) {
          flash_attention_kernel_impl_512<at::BFloat16, float>(
              output_contiguous_aten, logsumexp_aten, query_aten, key_aten,
              value_aten, dropout_p, is_causal, attn_mask_aten, scale);
        } else {
          flash_attention_kernel_impl_512<at::BFloat16, at::BFloat16>(
              output_contiguous_aten, logsumexp_aten, query_aten, key_aten,
              value_aten, dropout_p, is_causal, attn_mask_aten, scale);
        }
      } else if (query.scalar_type() == at::kHalf) {
        ZENTORCH_CHECK(!attn_mask.has_value() ||
                           attn_mask->scalar_type() == at::kFloat ||
                           attn_mask->scalar_type() == at::kHalf,
                       "zentorch_scaled_dot_product_attention_flash_"
                       "attention: Attention mask "
                       "is supported for FP32 and FP16 dtype when the query "
                       "is of type FP16");
        if (!attn_mask.has_value() || attn_mask->scalar_type() == at::kFloat) {
          flash_attention_kernel_impl_512<at::Half, float>(
              output_contiguous_aten, logsumexp_aten, query_aten, key_aten,
              value_aten, dropout_p, is_causal, attn_mask_aten, scale);
        } else {
          flash_attention_kernel_impl_512<at::Half, at::Half>(
              output_contiguous_aten, logsumexp_aten, query_aten, key_aten,
              value_aten, dropout_p, is_causal, attn_mask_aten, scale);
        }
      } else {
        ZENTORCH_CHECK(
            !attn_mask.has_value() || attn_mask->scalar_type() == at::kFloat,
            "zentorch_scaled_dot_product_attention_flash_"
            "attention: Attention mask "
            "is supported for FP32 dtype when the query is of type FP32");
        flash_attention_kernel_impl_512<float, float>(
            output_contiguous_aten, logsumexp_aten, query_aten, key_aten,
            value_aten, dropout_p, is_causal, attn_mask_aten, scale);
      }
      if (!to_aten(output_contiguous).is_alias_of(to_aten(caller_out))) {
        torch::stable::copy_(caller_out,
                             torch::stable::transpose(output_contiguous, 1, 2));
      }
    }

    logsumexp = torch::stable::transpose(logsumexp, 1, 2);

    return std::make_tuple(std::move(caller_out), std::move(logsumexp));
  } else {
    // at::_scaled_dot_product_flash_attention_for_cpu does an extra .contiguous
    // on the query tensor while we process the query as is in meta registration
    // and bf16 impl. Leading to a mismatch in stride between meta output and
    // runtime output.
    // Hence using native - same as ipex.
    auto native_result = at::native::_scaled_dot_product_flash_attention_cpu(
        to_aten(query), to_aten(key), to_aten(value), dropout_p, is_causal,
        to_aten_optional(attn_mask), scale);
    torch::stable::copy_(output, to_stable(std::get<0>(native_result)));
    return std::make_tuple(output, to_stable(std::get<1>(native_result)));
  }
}

// Out variant: writes the attention result directly into a caller-owned
// buffer. Lets callers that already hold a destination (e.g. the vLLM CPU
// attention backend, which is handed a pre-allocated output tensor) skip the
// full-size copy that the allocating variant forces on them. logsumexp is
// still computed as kernel scratch but dropped, since the op returns ().
void zentorch_sdpa_out(
    const torch::stable::Tensor &query, const torch::stable::Tensor &key,
    const torch::stable::Tensor &value, double dropout_p, bool is_causal,
    std::optional<torch::stable::Tensor> attn_mask, std::optional<double> scale,
    std::string zentorch_op_name, torch::stable::Tensor &out) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__
            << " zentorch_op_name: " << zentorch_op_name;
  zentorch_scaled_dot_product_attention_impl(query, key, value, dropout_p,
                                             is_causal, std::move(attn_mask),
                                             scale, out);
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor>
zentorch_sdpa(const torch::stable::Tensor &query,
              const torch::stable::Tensor &key,
              const torch::stable::Tensor &value, double dropout_p,
              bool is_causal, std::optional<torch::stable::Tensor> attn_mask,
              std::optional<double> scale, std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__
            << " zentorch_op_name: " << zentorch_op_name;
  torch::stable::Tensor output = torch::stable::empty_like(query);
  return zentorch_scaled_dot_product_attention_impl(
      query, key, value, dropout_p, is_causal, std::move(attn_mask), scale,
      output);
}

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
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
STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_sdpa", TORCH_BOX(&zentorch::zentorch_sdpa));
  m.impl("zentorch_sdpa.out", TORCH_BOX(&zentorch::zentorch_sdpa_out));
}
} // namespace zentorch
