/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

// HSTU attention of DLRM v3 / generative-recommenders on the ZenDNN LOWOHA
// hstu_attention operator. The schemas follow `pytorch_hstu_mha` and
// `pytorch_cached_hstu_mha` (inference: no dropout). Both ops write the
// result into the caller-supplied `out`.

#include "Memory.hpp"
#include "Utils.hpp"

#include <optional>
#include <string>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

namespace zentorch {

namespace {

namespace hstu = zendnnl::lowoha::hstu_attention;

torch::stable::Tensor contiguous_as(const torch::stable::Tensor &t,
                                    c10::ScalarType dtype) {
  return torch::stable::contiguous(torch::stable::to(t, dtype));
}

bool is_index_dtype(const torch::stable::Tensor &t) {
  return t.scalar_type() == c10::kLong || t.scalar_type() == c10::kInt;
}

void check_qkv(const std::string &op, const torch::stable::Tensor &q,
               const torch::stable::Tensor &k, const torch::stable::Tensor &v) {
  ZENTORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, op,
                 ": q, k and v must be 3 dimensional tensors but got ", q.dim(),
                 " and ", k.dim(), " and ", v.dim());
  ZENTORCH_CHECK(q.size(1) == k.size(1) && k.size(1) == v.size(1), op,
                 ": q, k and v must have the same number of heads but got ",
                 q.size(1), " and ", k.size(1), " and ", v.size(1));
  ZENTORCH_CHECK(q.size(2) == k.size(2), op,
                 ": q and k must have the same head dim but got ", q.size(2),
                 " and ", k.size(2));
  ZENTORCH_CHECK(k.size(0) == v.size(0), op,
                 ": k and v must have the same number of tokens but got ",
                 k.size(0), " and ", v.size(0));
  ZENTORCH_CHECK(
      q.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type(),
      op, ": q, k and v must have the same dtype but got ", q.scalar_type(),
      " and ", k.scalar_type(), " and ", v.scalar_type());
  ZENTORCH_CHECK(
      q.scalar_type() == c10::kBFloat16 || q.scalar_type() == c10::kFloat, op,
      ": q, k and v must be bfloat16 or float32 but got ", q.scalar_type());
  ZENTORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1, op,
                 ": the head dim of q, k and v must be contiguous but got ",
                 q.stride(2), " and ", k.stride(2), " and ", v.stride(2));
  for (const torch::stable::Tensor *t : {&q, &k, &v})
    ZENTORCH_CHECK(t->stride(0) > 0 && t->stride(1) > 0, op,
                   ": q, k and v need positive token and head strides "
                   "(broadcast / expanded views are not supported) but got ",
                   t->stride(0), " and ", t->stride(1));
}

// Runs hstu_attention_direct into `out`. `delta` selects the cached mode
// where q holds the last delta_size queries of every sequence.
void hstu_mha_impl(const std::string &op, int64_t max_seq_len, double alpha,
                   const torch::stable::Tensor &q,
                   const torch::stable::Tensor &k,
                   const torch::stable::Tensor &v,
                   const torch::stable::Tensor &seq_offsets, bool causal,
                   const std::optional<torch::stable::Tensor> &num_targets,
                   const std::optional<torch::stable::Tensor> &attn_scale,
                   int64_t max_attn_len, int64_t contextual_seq_len,
                   int64_t min_full_attn_seq_len, bool delta,
                   torch::stable::Tensor &out) {
  check_qkv(op, q, k, v);
  ZENTORCH_CHECK(q.size(2) > 0 && v.size(2) > 0, op,
                 ": the head dims of q, k and v must be non-empty");
  ZENTORCH_CHECK(max_seq_len >= 0 && max_attn_len >= 0 &&
                     contextual_seq_len >= 0 && min_full_attn_seq_len >= 0,
                 op,
                 ": max_seq_len, max_attn_len, contextual_seq_len and "
                 "min_full_attn_seq_len must be >= 0");
  ZENTORCH_CHECK(is_index_dtype(seq_offsets), op,
                 ": seq_offsets must be an int32 / int64 tensor");
  ZENTORCH_CHECK(seq_offsets.dim() == 1 && seq_offsets.size(0) >= 1, op,
                 ": seq_offsets must be [num_seqs + 1]");
  const int64_t num_seqs = seq_offsets.size(0) - 1;
  const torch::stable::Tensor offsets = contiguous_as(seq_offsets, c10::kLong);
  const auto *off = static_cast<const int64_t *>(offsets.data_ptr());

  int64_t delta_size = 0;
  if (delta) {
    ZENTORCH_CHECK((num_seqs == 0 && q.size(0) == 0) ||
                       (num_seqs > 0 && q.size(0) % num_seqs == 0),
                   op, ": delta_q rows must be a multiple of num_seqs");
    delta_size = num_seqs > 0 ? q.size(0) / num_seqs : 0;
  } else {
    ZENTORCH_CHECK(q.size(0) == k.size(0), op,
                   ": q, k and v must have the same number of tokens");
  }
  // seq_offsets must tile k and v exactly, so that every output row is
  // written and every row ZenDNN reads exists.
  ZENTORCH_CHECK(off[0] == 0 && off[num_seqs] == k.size(0), op,
                 ": seq_offsets must start at 0 and end at the number of "
                 "tokens of k and v");

  std::optional<torch::stable::Tensor> targets;
  const int64_t *tgt = nullptr;
  if (num_targets.has_value() && num_targets->defined()) {
    ZENTORCH_CHECK(is_index_dtype(*num_targets), op,
                   ": num_targets must be an int32 / int64 tensor");
    ZENTORCH_CHECK(num_targets->dim() == 1 && num_targets->size(0) == num_seqs,
                   op, ": num_targets must be [num_seqs]");
    targets = contiguous_as(*num_targets, c10::kLong);
    tgt = static_cast<const int64_t *>(targets->data_ptr());
  }
  for (int64_t b = 0; b < num_seqs; ++b) {
    const int64_t len = off[b + 1] - off[b];
    ZENTORCH_CHECK(len >= 0 && len <= max_seq_len, op,
                   ": sequence lengths must be in [0, max_seq_len]");
    ZENTORCH_CHECK(len >= delta_size, op,
                   ": every sequence must hold at least delta_size tokens");
    ZENTORCH_CHECK(!tgt || (tgt[b] >= 0 && tgt[b] <= len), op,
                   ": num_targets must be in [0, sequence length]");
  }

  hstu::hstu_attention_params p;
  p.num_seqs = num_seqs;
  p.num_heads = q.size(1);
  p.qk_dim = q.size(2);
  p.v_dim = v.size(2);
  p.max_seq_len = max_seq_len;
  p.seq_offsets = off;
  p.delta_size = delta_size;
  p.num_targets = tgt;
  p.causal = causal;
  p.max_attn_len = max_attn_len;
  p.contextual_seq_len = contextual_seq_len;
  p.min_full_attn_seq_len = min_full_attn_seq_len;
  p.alpha = static_cast<float>(alpha);

  std::optional<torch::stable::Tensor> scale;
  if (attn_scale.has_value() && attn_scale->defined()) {
    scale = contiguous_as(*attn_scale, c10::kFloat);
    const auto *s = static_cast<const float *>(scale->data_ptr());
    if (scale->dim() == 0) {
      p.has_attn_scale = true;
      p.attn_scale = *s;
    } else {
      ZENTORCH_CHECK(scale->dim() == 1 && scale->size(0) == k.size(0), op,
                     ": attn_scale must be a scalar or [tokens]");
      p.attn_scale_per_token = s;
    }
  }

  ZENTORCH_CHECK(out.dim() == 3 && out.size(0) == q.size(0) &&
                     out.size(1) == q.size(1) && out.size(2) == v.size(2),
                 op, ": out must be [tokens, heads, v_dim]");
  ZENTORCH_CHECK(out.scalar_type() == q.scalar_type(), op,
                 ": out dtype must match q, k and v");
  ZENTORCH_CHECK(out.stride(2) == 1, op,
                 ": the head dim of out must be contiguous");
  if (q.size(0) == 0)
    return;
  ZENTORCH_CHECK(out.stride(0) > 0 && out.stride(1) > 0, op,
                 ": out needs positive token and head strides "
                 "(broadcast / expanded views are not supported)");
  p.q_stride_t = q.stride(0);
  p.q_stride_h = q.stride(1);
  p.k_stride_t = k.stride(0);
  p.k_stride_h = k.stride(1);
  p.v_stride_t = v.stride(0);
  p.v_stride_h = v.stride(1);
  p.o_stride_t = out.stride(0);
  p.o_stride_h = out.stride(1);
  p.qkv_dt = get_zendnnl_dtype(q);
  p.out_dt = get_zendnnl_dtype(out);
  ZENTORCH_CHECK(hstu::hstu_attention_direct(q.data_ptr(), k.data_ptr(),
                                             v.data_ptr(), out.data_ptr(),
                                             p) == hstu::status_t::success,
                 op, ": hstu_attention_direct failed");
}

} // namespace

void zentorch_hstu_mha_out(
    int64_t max_seq_len, double alpha, const torch::stable::Tensor &q,
    const torch::stable::Tensor &k, const torch::stable::Tensor &v,
    const torch::stable::Tensor &seq_offsets, bool causal,
    std::optional<torch::stable::Tensor> num_targets,
    std::optional<torch::stable::Tensor> attn_scale, int64_t max_attn_len,
    int64_t contextual_seq_len, int64_t min_full_attn_seq_len,
    std::string zentorch_op_name, torch::stable::Tensor &out) {
  hstu_mha_impl(zentorch_op_name, max_seq_len, alpha, q, k, v, seq_offsets,
                causal, num_targets, attn_scale, max_attn_len,
                contextual_seq_len, min_full_attn_seq_len, /*delta=*/false,
                out);
}

void zentorch_delta_hstu_mha_out(
    int64_t max_seq_len, double alpha, const torch::stable::Tensor &delta_q,
    const torch::stable::Tensor &k, const torch::stable::Tensor &v,
    const torch::stable::Tensor &seq_offsets,
    std::optional<torch::stable::Tensor> num_targets, int64_t max_attn_len,
    int64_t contextual_seq_len, std::string zentorch_op_name,
    torch::stable::Tensor &out) {
  hstu_mha_impl(zentorch_op_name, max_seq_len, alpha, delta_q, k, v,
                seq_offsets, /*causal=*/true, num_targets,
                /*attn_scale=*/std::nullopt, max_attn_len, contextual_seq_len,
                /*min_full_attn_seq_len=*/0, /*delta=*/true, out);
}

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_hstu_mha.out(SymInt max_seq_len, float alpha, Tensor q, "
        "Tensor k, Tensor v, Tensor seq_offsets, bool causal=True, "
        "Tensor? num_targets=None, Tensor? attn_scale=None, "
        "int max_attn_len=0, int contextual_seq_len=0, "
        "int min_full_attn_seq_len=0, *, "
        "str zentorch_op_name='zentorch::zentorch_hstu_mha', "
        "Tensor(a!) out) -> ()");
  m.def("zentorch_delta_hstu_mha.out(SymInt max_seq_len, float alpha, "
        "Tensor delta_q, Tensor k, Tensor v, Tensor seq_offsets, "
        "Tensor? num_targets=None, int max_attn_len=0, "
        "int contextual_seq_len=0, *, "
        "str zentorch_op_name='zentorch::zentorch_delta_hstu_mha', "
        "Tensor(a!) out) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_hstu_mha.out", TORCH_BOX(&zentorch::zentorch_hstu_mha_out));
  m.impl("zentorch_delta_hstu_mha.out",
         TORCH_BOX(&zentorch::zentorch_delta_hstu_mha_out));
}

} // namespace zentorch
