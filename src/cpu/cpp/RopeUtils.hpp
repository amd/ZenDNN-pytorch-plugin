/******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Was sourced from
 * https://github.com/intel/intel-extension-for-pytorch/blob/v2.6.0%2Bcpu/csrc/cpu/aten/kernels/RotaryPositionEmbeddingKnl.cpp
 * IPEX commit ID: 18eeefa
 * https://github.com/intel/intel-extension-for-pytorch/blob/v2.6.0%2Bcpu/csrc/cpu/vec/general/rope.h
 * IPEX commit ID: c37bace
 ******************************************************************************/

#pragma once

#include "Utils.hpp"
#include <ATen/ATen.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <torch/types.h>

namespace zentorch {
namespace cpu {
namespace kernel {

using namespace at::vec;

template <typename scalar_t>
inline typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t> &&
                                     !std::is_same_v<float, scalar_t>,
                                 void>
apply_rope_along_head_kernel(scalar_t *in_ptr_start, scalar_t *out_ptr_start,
                             float *cos_start, float *sin_start,
                             int64_t rotary_ndims, int64_t offset) {
  auto h = 0;
  for (h = 0; h < rotary_ndims / 2; h++) {
    float x = in_ptr_start[h];
    float y = in_ptr_start[h + offset];
    float sin = sin_start[h];
    float cos = cos_start[h];
    float out0 = x * cos - y * sin;
    float out1 = y * cos + x * sin;
    out_ptr_start[h] = out0;
    out_ptr_start[h + offset] = out1;
  }
}

template <typename scalar_t>
inline typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t> &&
                                     std::is_same_v<float, scalar_t>,
                                 void>
apply_rope_along_head_kernel(scalar_t *in_ptr_start, scalar_t *out_ptr_start,
                             float *cos_start, float *sin_start,
                             int64_t rotary_ndims, int64_t offset) {
  auto h = 0;
  using Vec = Vectorized<float>;
  const int vec_size = Vec::size();
  for (h = 0; h <= rotary_ndims / 2 - vec_size; h += vec_size) {
    auto x = Vec::loadu(in_ptr_start + h);
    auto y = Vec::loadu(in_ptr_start + h + offset);
    auto sin = Vec::loadu(sin_start + h);
    auto cos = Vec::loadu(cos_start + h);
    auto out0 = x * cos - y * sin;
    auto out1 = y * cos + x * sin;
    out0.store(out_ptr_start + h);
    out1.store(out_ptr_start + h + offset);
  }
  for (; h < rotary_ndims / 2; h++) {
    float x = in_ptr_start[h];
    float y = in_ptr_start[h + offset];
    float sin = sin_start[h];
    float cos = cos_start[h];
    float out0 = x * cos - y * sin;
    float out1 = y * cos + x * sin;
    out_ptr_start[h] = out0;
    out_ptr_start[h + offset] = out1;
  }
}

template <typename scalar_t>
inline typename std::enable_if_t<is_reduced_floating_point_v<scalar_t> &&
                                     !std::is_same_v<float, scalar_t>,
                                 void>
apply_rope_along_head_kernel(scalar_t *in_ptr_start, scalar_t *out_ptr_start,
                             float *cos_start, float *sin_start,
                             int64_t rotary_ndims, int64_t offset) {
  auto h = 0;
  using bVec = Vectorized<scalar_t>;
  using fVec = Vectorized<float>;
  const int fvec_size = fVec::size();
  const int bvec_size = bVec::size();
  for (h = 0; h <= rotary_ndims / 2 - bvec_size; h += bvec_size) {
    bVec x = bVec::loadu(in_ptr_start + h);
    bVec y = bVec::loadu(in_ptr_start + h + offset);
    fVec x0, x1, y0, y1;
    std::tie(x0, x1) = convert_to_float<scalar_t>(x);
    std::tie(y0, y1) = convert_to_float<scalar_t>(y);
    fVec c0 = fVec::loadu(cos_start + h);
    fVec s0 = fVec::loadu(sin_start + h);
    fVec c1 = fVec::loadu(cos_start + h + fvec_size);
    fVec s1 = fVec::loadu(sin_start + h + fvec_size);
    fVec x_out0 = x0 * c0 - y0 * s0;
    fVec x_out1 = x1 * c1 - y1 * s1;
    fVec y_out0 = y0 * c0 + x0 * s0;
    fVec y_out1 = y1 * c1 + x1 * s1;
    bVec x_out = convert_from_float<scalar_t>(x_out0, x_out1);
    bVec y_out = convert_from_float<scalar_t>(y_out0, y_out1);
    x_out.store(out_ptr_start + h);
    y_out.store(out_ptr_start + h + offset);
  }
  for (; h < rotary_ndims / 2; h++) {
    float x = in_ptr_start[h];
    float y = in_ptr_start[h + offset];
    float sin = sin_start[h];
    float cos = cos_start[h];
    float out0 = x * cos - y * sin;
    float out1 = y * cos + x * sin;
    out_ptr_start[h] = out0;
    out_ptr_start[h + offset] = out1;
  }
}

// is_fused_qkv
inline bool is_fused_qkv(at::Tensor &t_in, int64_t hidden_size) {
  auto in_stride_s = t_in.stride(1);
  if (t_in.stride(0) * t_in.size(0) != t_in.numel()) {
    if (t_in.dim() == 4) {
      in_stride_s = t_in.size(2) * t_in.size(3);
    } else if (t_in.dim() == 3) {
      in_stride_s = t_in.size(2);
    }
  }
  // handle the false positive case when one of the
  // first two dimensions is 1
  if (t_in.size(0) == 1 || t_in.size(1) == 1) {
    if (t_in.dim() == 4) {
      in_stride_s = t_in.size(2) * t_in.size(3);
    } else if (t_in.dim() == 3) {
      in_stride_s = t_in.size(2);
    }
  }
  if (in_stride_s > hidden_size) {
    return true;
  }
  return false;
}

// define move_ker below
template <typename dst_type, typename src_type>
inline void move_ker(dst_type *inout, const src_type *in, int64_t len) {
#pragma omp simd
  for (int64_t i = 0; i < len; i++) {
    *(inout + i) = *(in + i);
  }
}

// Apply Rope kernel function
template <typename scalar_t>
inline void apply_rotary_embedding(const scalar_t *__restrict__ arr,
                                   const float *__restrict__ cos_ptr,
                                   const float *__restrict__ sin_ptr,
                                   scalar_t *__restrict__ out, int embed_dim) {
  using Vec = Vectorized<scalar_t>;
  const int kVecSize = Vec::size();
  const int len = embed_dim - (embed_dim % kVecSize);

  // GPT-J style rotary embedding.
  // format: {d, 2}, stride-2 access need permute to be vectorized.
  int d = 0;
  for (; d < len; d += kVecSize) {
    Vec x = Vec::loadu(arr + 2 * d + 0 * kVecSize);
    Vec y = Vec::loadu(arr + 2 * d + 1 * kVecSize);
    Vec cos = Vec::loadu(cos_ptr + d);
    Vec sin = Vec::loadu(sin_ptr + d);
    // x: {x0, y0, x1, y1, x2, y2, x3, y3}
    // y: {x4, y4, x5, y5, x6, y6, x7, y7}
    // x1: {x0, x1, x2, x3, x4, x5, x6, x7}
    // y1: {y0, y1, y2, y3, y4, y5, y6, y7}
    auto xy = deinterleave2(x, y);
    Vec x1 = std::get<0>(xy);
    Vec y1 = std::get<1>(xy);
    Vec x2 = x1 * cos - y1 * sin;
    Vec y2 = y1 * cos + x1 * sin;
    // x2: {x0, x1, x2, x3, x4, x5, x6, x7}
    // y2: {y0, y1, y2, y3, y4, y5, y6, y7}
    // x_out: {x0, y0, x1, y1, x2, y2, x3, y3}
    // y_out: {x4, y4, x5, y5, x6, y6, x7, y7}
    xy = interleave2(x2, y2);
    Vec x_out = std::get<0>(xy);
    Vec y_out = std::get<1>(xy);
    x_out.store(out + 2 * d + 0 * kVecSize);
    y_out.store(out + 2 * d + 1 * kVecSize);
  }
  for (; d < embed_dim; d++) {
    scalar_t x = arr[2 * d + 0];
    scalar_t y = arr[2 * d + 1];
    scalar_t x_out = x * cos_ptr[d] - y * sin_ptr[d];
    scalar_t y_out = y * cos_ptr[d] + x * sin_ptr[d];
    out[2 * d + 0] = x_out;
    out[2 * d + 1] = y_out;
  }
}

template <>
inline void apply_rotary_embedding<at::BFloat16>(
    const at::BFloat16 *__restrict__ arr, const float *__restrict__ cos_ptr,
    const float *__restrict__ sin_ptr, at::BFloat16 *__restrict__ out,
    int embed_dim) {
  using fVec = Vectorized<float>;
  using bVec = Vectorized<at::BFloat16>;

  const int kVecSize = bVec::size();
  const int len = 2 * embed_dim - (2 * embed_dim % kVecSize);

  // GPT-J style rotary embedding.
  // format: {d, 2}, stride-2 access need permute to be vectorized.
  int d = 0;
  for (; d < len; d += kVecSize) {
    bVec a = bVec::loadu(arr + d);
    fVec x, y;
    std::tie(x, y) = convert_bfloat16_float(a);
    fVec cos = fVec::loadu(cos_ptr + d / 2);
    fVec sin = fVec::loadu(sin_ptr + d / 2);
    // x: {x0, y0, x1, y1, x2, y2, x3, y3}
    // y: {x4, y4, x5, y5, x6, y6, x7, y7}
    // x1: {x0, x1, x2, x3, x4, x5, x6, x7}
    // y1: {y0, y1, y2, y3, y4, y5, y6, y7}
    auto xy = deinterleave2(x, y);
    fVec x1 = std::get<0>(xy);
    fVec y1 = std::get<1>(xy);
    fVec x2 = x1 * cos - y1 * sin;
    fVec y2 = y1 * cos + x1 * sin;
    // x2: {x0, x1, x2, x3, x4, x5, x6, x7}
    // y2: {y0, y1, y2, y3, y4, y5, y6, y7}
    // x_out: {x0, y0, x1, y1, x2, y2, x3, y3}
    // y_out: {x4, y4, x5, y5, x6, y6, x7, y7}
    xy = interleave2(x2, y2);
    fVec x_out = std::get<0>(xy);
    fVec y_out = std::get<1>(xy);
    bVec a_out = convert_float_bfloat16(x_out, y_out);
    a_out.store(out + d);
  }
  for (; d < embed_dim; d++) {
    float x = static_cast<float>(arr[2 * d + 0]);
    float y = static_cast<float>(arr[2 * d + 1]);
    float x_out = x * cos_ptr[d] - y * sin_ptr[d];
    float y_out = y * cos_ptr[d] + x * sin_ptr[d];
    out[2 * d + 0] = static_cast<at::BFloat16>(x_out);
    out[2 * d + 1] = static_cast<at::BFloat16>(y_out);
  }
}

template <typename scalar_t>
inline void RotateEveryTwo(const scalar_t *in_query_ptr,
                           const scalar_t *in_key_ptr, scalar_t *out_query_ptr,
                           scalar_t *out_key_ptr, const float *sin_start,
                           const float *cos_start, const int HR,
                           const int offset, const bool calc_key) {
  // TODO: remove overhead for loading sin and cos
  int embed_dim = HR / 2;
  apply_rotary_embedding<scalar_t>(in_query_ptr, cos_start, sin_start,
                                   out_query_ptr, embed_dim);

  if (calc_key) {
    apply_rotary_embedding<scalar_t>(in_key_ptr, cos_start, sin_start,
                                     out_key_ptr, embed_dim);
  }
}

template <>
inline void RotateEveryTwo<at::BFloat16>(
    const at::BFloat16 *in_query_ptr, const at::BFloat16 *in_key_ptr,
    at::BFloat16 *out_query_ptr, at::BFloat16 *out_key_ptr,
    const float *sin_ptr, const float *cos_ptr, const int HR, const int offset,
    const bool calc_key) {
  int embed_dim = HR / 2;

  using fVec = Vectorized<float>;
  using bVec = Vectorized<at::BFloat16>;

  const int kVecSize = bVec::size();
  const int len = HR - (HR % kVecSize);

  // GPT-J style rotary embedding.
  // format: {d, 2}, stride-2 access need permute to be vectorized.
  int d = 0;
  for (; d < len; d += kVecSize) {
    bVec in_query = bVec::loadu(in_query_ptr + d);
    fVec x, y;
    std::tie(x, y) = convert_bfloat16_float(in_query);
    fVec cos = fVec::loadu(cos_ptr + d / 2);
    fVec sin = fVec::loadu(sin_ptr + d / 2);
    // x: {x0, y0, x1, y1, x2, y2, x3, y3}
    // y: {x4, y4, x5, y5, x6, y6, x7, y7}
    // x1: {x0, x1, x2, x3, x4, x5, x6, x7}
    // y1: {y0, y1, y2, y3, y4, y5, y6, y7}
    auto xy = deinterleave2(x, y);
    fVec x1 = std::get<0>(xy);
    fVec y1 = std::get<1>(xy);
    fVec x2 = x1 * cos - y1 * sin;
    fVec y2 = y1 * cos + x1 * sin;
    // x2: {x0, x1, x2, x3, x4, x5, x6, x7}
    // y2: {y0, y1, y2, y3, y4, y5, y6, y7}
    // x_out: {x0, y0, x1, y1, x2, y2, x3, y3}
    // y_out: {x4, y4, x5, y5, x6, y6, x7, y7}
    xy = interleave2(x2, y2);
    fVec x_out = std::get<0>(xy);
    fVec y_out = std::get<1>(xy);
    bVec a_out = convert_float_bfloat16(x_out, y_out);
    a_out.store(out_query_ptr + d);
    if (calc_key) {
      bVec in_key = bVec::loadu(in_key_ptr + d);
      fVec x, y;
      std::tie(x, y) = convert_bfloat16_float(in_key);
      // x: {x0, y0, x1, y1, x2, y2, x3, y3}
      // y: {x4, y4, x5, y5, x6, y6, x7, y7}
      // x1: {x0, x1, x2, x3, x4, x5, x6, x7}
      // y1: {y0, y1, y2, y3, y4, y5, y6, y7}
      auto xy = deinterleave2(x, y);
      fVec x1 = std::get<0>(xy);
      fVec y1 = std::get<1>(xy);
      fVec x2 = x1 * cos - y1 * sin;
      fVec y2 = y1 * cos + x1 * sin;
      // x2: {x0, x1, x2, x3, x4, x5, x6, x7}
      // y2: {y0, y1, y2, y3, y4, y5, y6, y7}
      // x_out: {x0, y0, x1, y1, x2, y2, x3, y3}
      // y_out: {x4, y4, x5, y5, x6, y6, x7, y7}
      xy = interleave2(x2, y2);
      fVec x_out = std::get<0>(xy);
      fVec y_out = std::get<1>(xy);
      bVec a_out = convert_float_bfloat16(x_out, y_out);
      a_out.store(out_key_ptr + d);
    }
  }
  for (; d < embed_dim; d++) {
    float x = static_cast<float>(in_query_ptr[2 * d + 0]);
    float y = static_cast<float>(in_query_ptr[2 * d + 1]);
    float cos = cos_ptr[d];
    float sin = sin_ptr[d];
    float x_out = x * cos - y * sin;
    float y_out = y * cos + x * sin;
    out_query_ptr[2 * d + 0] = static_cast<at::BFloat16>(x_out);
    out_query_ptr[2 * d + 1] = static_cast<at::BFloat16>(y_out);
    if (calc_key) {
      float x = static_cast<float>(in_key_ptr[2 * d + 0]);
      float y = static_cast<float>(in_key_ptr[2 * d + 1]);
      float x_out = x * cos - y * sin;
      float y_out = y * cos + x * sin;
      out_key_ptr[2 * d + 0] = static_cast<at::BFloat16>(x_out);
      out_key_ptr[2 * d + 1] = static_cast<at::BFloat16>(y_out);
    }
  }
}

template <typename scalar_t>
inline void
RotateEveryTwoNaive(const scalar_t *in_query_ptr, const scalar_t *in_key_ptr,
                    scalar_t *out_query_ptr, scalar_t *out_key_ptr,
                    const float *sin_start, const float *cos_start,
                    const int HR, const int offset, const bool calc_key) {
  int embed_dim = HR / 2;
  for (int h = 0, h2 = 0; h < HR; h += 2, h2++) {
    float sin = sin_start[h2];
    float cos = cos_start[h2];
    float in0 = in_query_ptr[h];
    float in1 = in_query_ptr[h + offset];
    float out0 = in0 * cos - in1 * sin;
    float out1 = in1 * cos + in0 * sin;
    out_query_ptr[h] = out0;
    out_query_ptr[h + offset] = out1;
    if (calc_key) {
      in0 = in_key_ptr[h];
      in1 = in_key_ptr[h + offset];
      out0 = in0 * cos - in1 * sin;
      out1 = in1 * cos + in0 * sin;
      out_key_ptr[h] = out0;
      out_key_ptr[h + offset] = out1;
    }
  }
}

template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor>
ApplyROPEKernel(at::Tensor &t_in, at::Tensor &t_emb_pos, at::Tensor &t_pos,
                int64_t N, // N: number of head, H: head size
                int64_t H, int64_t offset, int64_t rotary_dim) {
  auto in_sizes = t_in.sizes(); // in[B][S][F] or [B][S][N][H]
  auto HR = t_emb_pos.size(1);  // rotary_dim
  auto B = in_sizes[0];
  auto S = in_sizes[1];
  auto HS = in_sizes[2];
  auto in_stride_b = t_in.stride(0);
  auto in_stride_s = t_in.stride(1);
  auto N_KV = N; // GQA/MQA, N_KV: number of head for key/value
  auto concat_qkv = in_stride_s > N * H;

  if (is_fused_qkv(t_in, N * H)) {
    ZENTORCH_CHECK(
        t_in.dim() == 3,
        "The shape of input tensor of rotary_position_embedding should be in "
        "(batch, seq_len, qkv_hidden_size) when using fused qkv)");
    N_KV = (HS - N * H) / (2 * H);
  }

  auto COFF = HR / 2;
  auto in_ptr = t_in.data_ptr<T>();
  // initialize empty q/k/v
  auto query = at::empty({B, S, N, H}, t_in.options());
  auto key =
      concat_qkv ? at::empty({B, S, N_KV, H}, t_in.options()) : at::Tensor();
  auto value =
      concat_qkv ? at::empty({B, S, N_KV, H}, t_in.options()) : at::Tensor();
  auto query_ptr = query.data_ptr<T>();
  auto key_ptr = concat_qkv ? key.data_ptr<T>() : nullptr;
  auto value_ptr = concat_qkv ? value.data_ptr<T>() : nullptr;
  auto out_stride_qb = query.stride(0);
  auto out_stride_qs = query.stride(1);
  auto out_stride_kb = concat_qkv ? key.stride(0) : 0;
  auto out_stride_ks = concat_qkv ? key.stride(1) : 0;
  auto emb_pos_ptr = t_emb_pos.data_ptr<float>(); // [MP][HR]
  auto pos_ptr = t_pos.data_ptr<long>();          // [B][S] or [1][S]
  bool t_pos_no_repeated_for_batch = false;
  if (t_pos.numel() != 1 && t_pos.size(0) == 1 && B > 1) {
    // we do not perform t_pos.repeat here to avoid the overhead of copying
    t_pos_no_repeated_for_batch = true;
  }
  {
#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
      for (int s = 0; s < S; s++) {
        for (int n = 0; n < N; n++) {
          auto in_offset_q = b * in_stride_b + s * in_stride_s + n * H;
          auto out_offset_q = b * out_stride_qb + s * out_stride_qs + n * H;
          auto out_offset_k =
              concat_qkv ? b * out_stride_kb + s * out_stride_ks + n * H : 0;
          auto in_offset_k = concat_qkv ? in_offset_q + N * H : 0;
          long p = 0;
          float *sin_start = nullptr;
          float *cos_start = nullptr;
          // step 0) get the rotary position embedding for the current position
          if (t_pos.numel() == 1) { // used by Falcon & ChatGLM,
            p = pos_ptr[0];
            sin_start = emb_pos_ptr + (p + s) * HR;
            cos_start = emb_pos_ptr + (p + s) * HR + COFF;
          } else {
            auto start_idx = t_pos_no_repeated_for_batch ? 0 : b * S;
            p = pos_ptr[start_idx + s];
            sin_start = emb_pos_ptr + p * HR;
            cos_start = emb_pos_ptr + p * HR + COFF;
          }
          // step 1) apply_rotary_pos_emb for the rotary_dim elements in every
          // head of query/key
          if (offset !=
              1) { // use vectorized version if there are more than 16
                   // continuous elements, used by lamma/gpt-neox/falcon
                   // logic is like to the rotate_half in python code
            zentorch::cpu::kernel::apply_rope_along_head_kernel<T>(
                in_ptr + in_offset_q, query_ptr + out_offset_q, cos_start,
                sin_start, rotary_dim, offset);
            if (concat_qkv && n < N_KV) {
              zentorch::cpu::kernel::apply_rope_along_head_kernel<T>(
                  in_ptr + in_offset_k, key_ptr + out_offset_k, cos_start,
                  sin_start, rotary_dim, offset);
            }
          } else { // used by GPT-J 6B & CodeGen & ChatGLM
                   // logic is like to the rotate_every_two in python code
            RotateEveryTwo<T>(&in_ptr[in_offset_q], &in_ptr[in_offset_k],
                              &query_ptr[out_offset_q], &key_ptr[out_offset_k],
                              sin_start, cos_start, HR, offset,
                              (concat_qkv && n < N_KV));
          }
          // step 2) copy the rest of the input tensor to query/key (query_pass
          // & key_pass)
          if (rotary_dim < H) {
            zentorch::cpu::kernel::move_ker<T, T>(
                query_ptr + out_offset_q + rotary_dim,
                in_ptr + in_offset_q + rotary_dim, H - rotary_dim);
            if (concat_qkv && n < N_KV) {
              zentorch::cpu::kernel::move_ker<T, T>(
                  key_ptr + out_offset_k + rotary_dim,
                  in_ptr + in_offset_k + rotary_dim, H - rotary_dim);
            }
          }
          // step 3) copy value from t_in when concat_qkv is true
          if (concat_qkv && n < N_KV) {
            auto in_offset_v = in_offset_k + N_KV * H;
            zentorch::cpu::kernel::move_ker<T, T>(value_ptr + out_offset_k,
                                                  in_ptr + in_offset_v, H);
          }
        }
      }
    }
  }
  return std::make_tuple(query, key, value);
}

template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor>
ApplyDeepseekROPEKernel(at::Tensor &q, at::Tensor &kv, at::Tensor &k_pe,
                        at::Tensor &t_emb_pos, at::Tensor &t_pos,
                        int64_t N, // N: number of head, H: head size
                        int64_t H, int64_t offset, int64_t rotary_dim) {
  auto in_sizes = q.sizes(); // in[B][S][F] or [B][S][N][H]
  // auto MP = t_emb_pos.size(0); // Max Pos
  auto HR = t_emb_pos.size(1); // rotary_dim
  auto B = in_sizes[0];
  auto S = in_sizes[1];
  // auto HS = in_sizes[2];
  auto kv_size_d = kv.size(3);
  auto in_stride_b = q.stride(0);
  auto in_stride_s = q.stride(1);

  auto COFF = HR / 2;
  auto in_ptr = q.data_ptr<T>();
  auto kv_ptr = kv.data_ptr<T>();
  auto k_pe_ptr = k_pe.data_ptr<T>();
  auto k_pe_stride_b = k_pe.stride(0);
  auto k_pe_stride_s = k_pe.stride(1);
  auto kv_stride_b = kv.stride(0);
  auto kv_stride_s = kv.stride(1);
  auto kv_stride_n = kv.stride(2);

  // initialize empty q/k/v
  auto query = at::empty({B, S, N, H}, q.options());
  auto key = at::empty({B, S, N, H}, q.options());
  auto value = at::empty({B, S, N, H}, q.options());
  auto query_ptr = query.data_ptr<T>();
  auto key_ptr = key.data_ptr<T>();
  auto value_ptr = value.data_ptr<T>();
  auto out_stride_qb = query.stride(0);
  auto out_stride_qs = query.stride(1);
  auto out_stride_kb = key.stride(0);
  auto out_stride_ks = key.stride(1);
  auto emb_pos_ptr = t_emb_pos.data_ptr<float>(); // [MP][HR]
  auto pos_ptr = t_pos.data_ptr<long>();          // [B][S] or [1][S]
  bool t_pos_no_repeated_for_batch = false;
  if (t_pos.numel() != 1 && t_pos.size(0) == 1 && B > 1) {
    // we do not perform t_pos.repeat here to avoid the overhead of copying
    t_pos_no_repeated_for_batch = true;
  }
  {
#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
      for (int s = 0; s < S; s++) {
        for (int n = 0; n < N; n++) {
          auto in_offset_q = b * in_stride_b + s * in_stride_s + n * H;
          auto out_offset_q = b * out_stride_qb + s * out_stride_qs + n * H;
          auto out_offset_k = b * out_stride_kb + s * out_stride_ks + n * H;
          auto kv_offset_k =
              b * kv_stride_b + s * kv_stride_s + n * kv_stride_n;
          auto kv_offset_v = kv_offset_k + offset;
          auto k_pe_offset = b * k_pe_stride_b + s * k_pe_stride_s - offset;
          long p = 0;
          float *sin_start = nullptr;
          float *cos_start = nullptr;
          // step 0) get the rotary position embedding for the current position
          auto start_idx = t_pos_no_repeated_for_batch ? 0 : b * S;
          p = pos_ptr[start_idx + s];
          sin_start = emb_pos_ptr + p * HR;
          cos_start = emb_pos_ptr + p * HR + COFF;
          // step 1) apply_rotary_pos_emb for the rotary_dim elements in every
          // head of query/key
          for (auto h = offset; h < H; h += 2) {
            auto half_off = (h - offset) / 2;
            auto cos1 = cos_start[half_off];
            auto sin1 = sin_start[half_off];
            auto cos2 = cos_start[half_off + rotary_dim / 2];
            auto sin2 = sin_start[half_off + rotary_dim / 2];
            auto in1 = in_ptr[in_offset_q + h];
            auto in2 = in_ptr[in_offset_q + h + 1];
            auto out1 = in1 * cos1 - in2 * sin1;
            auto out2 = in2 * cos2 + in1 * sin2;
            auto out1_offset = out_offset_q + offset + half_off;
            auto out2_offset = out1_offset + rotary_dim / 2;
            query_ptr[out1_offset] = out1;
            query_ptr[out2_offset] = out2;
            auto in1_k = k_pe_ptr[k_pe_offset + h];
            auto in2_k = k_pe_ptr[k_pe_offset + h + 1];
            auto out1_k = in1_k * cos1 - in2_k * sin1;
            auto out2_k = in2_k * cos2 + in1_k * sin2;
            key_ptr[out1_offset] = out1_k;
            key_ptr[out2_offset] = out2_k;
          }
          // step 2) copy the rest of the input tensor to query/key (query_pass
          // & key_pass)
          zentorch::cpu::kernel::move_ker<T, T>(query_ptr + out_offset_q,
                                                in_ptr + in_offset_q, offset);
          zentorch::cpu::kernel::move_ker<T, T>(key_ptr + out_offset_k,
                                                kv_ptr + kv_offset_k, offset);
          // step 3) copy value from kv and padding
          zentorch::cpu::kernel::move_ker<T, T>(value_ptr + out_offset_k,
                                                kv_ptr + kv_offset_v,
                                                kv_size_d - offset);
          for (auto h = kv_size_d - offset; h < H; h++) {
            value_ptr[out_offset_k + h] = 0;
          }
        }
      }
    }
  }
  return std::make_tuple(query, key, value);
}

} // namespace kernel
} // namespace cpu
} // namespace zentorch
