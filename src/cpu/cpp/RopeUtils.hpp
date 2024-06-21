/******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Was sourced from
 * https://github.com/intel/intel-extension-for-pytorch/blob/v2.3.0%2Bcpu/csrc/cpu/aten/kernels/RotaryPositionEmbeddingKnl.cpp
 * IPEX commit ID: f57307d
 ******************************************************************************/

#pragma once

#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec.h>
#include <torch/types.h>

namespace zentorch {
namespace cpu {
namespace kernel {

using namespace at::vec;

template <typename scalar_t>
inline void apply_rope_along_head_kernel(scalar_t *in_ptr_start,
                                         scalar_t *out_ptr_start,
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

template <>
inline void apply_rope_along_head_kernel(float *in_ptr_start,
                                         float *out_ptr_start, float *cos_start,
                                         float *sin_start, int64_t rotary_ndims,
                                         int64_t offset) {
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

template <>
inline void apply_rope_along_head_kernel(at::BFloat16 *in_ptr_start,
                                         at::BFloat16 *out_ptr_start,
                                         float *cos_start, float *sin_start,
                                         int64_t rotary_ndims, int64_t offset) {
  auto h = 0;
  using bVec = Vectorized<at::BFloat16>;
  using fVec = Vectorized<float>;
  const int fvec_size = fVec::size();
  const int bvec_size = bVec::size();
  for (h = 0; h <= rotary_ndims / 2 - bvec_size; h += bvec_size) {
    bVec x = bVec::loadu(in_ptr_start + h);
    bVec y = bVec::loadu(in_ptr_start + h + offset);
    fVec x0, x1, y0, y1;
    std::tie(x0, x1) = convert_bfloat16_float(x);
    std::tie(y0, y1) = convert_bfloat16_float(y);
    fVec c0 = fVec::loadu(cos_start + h);
    fVec s0 = fVec::loadu(sin_start + h);
    fVec c1 = fVec::loadu(cos_start + h + fvec_size);
    fVec s1 = fVec::loadu(sin_start + h + fvec_size);
    fVec x_out0 = x0 * c0 - y0 * s0;
    fVec x_out1 = x1 * c1 - y1 * s1;
    fVec y_out0 = y0 * c0 + x0 * s0;
    fVec y_out1 = y1 * c1 + x1 * s1;
    bVec x_out = convert_float_bfloat16(x_out0, x_out1);
    bVec y_out = convert_float_bfloat16(y_out0, y_out1);
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
template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor>
ApplyROPEKernel(at::Tensor &t_in, at::Tensor &t_emb_pos, at::Tensor &t_pos,
                int64_t N, // N: number of head, H: head size
                int64_t H, int64_t offset, int64_t rotary_dim) {
  auto in_sizes = t_in.sizes(); // in[B][S][F] or [B][S][N][H]
  // auto MP = t_emb_pos.size(0); // Max Pos, this is unused for some reason but
  // we will keep it commented for now
  auto HR = t_emb_pos.size(1); // rotary_dim
  auto B = in_sizes[0];
  auto S = in_sizes[1];
  auto HS = in_sizes[2];
  auto in_stride_b = t_in.stride(0);
  auto in_stride_s = t_in.stride(1);
  auto N_KV = N; // GQA/MQA, N_KV: number of head for key/value
  auto concat_qkv = in_stride_s > N * H;

  if (is_fused_qkv(t_in, N * H)) {
    TORCH_CHECK(
        in_stride_s == HS,
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
            for (int h = 0, h2 = 0; h < HR; h += 2, h2++) {
              float sin = sin_start[h2];
              float cos = cos_start[h2];
              float in0 = in_ptr[in_offset_q + h];
              float in1 = in_ptr[in_offset_q + h + offset];
              float out0 = in0 * cos - in1 * sin;
              float out1 = in1 * cos + in0 * sin;
              query_ptr[out_offset_q + h] = out0;
              query_ptr[out_offset_q + h + offset] = out1;
              if (concat_qkv && n < N_KV) {
                in0 = in_ptr[in_offset_k + h];
                in1 = in_ptr[in_offset_k + h + offset];
                out0 = in0 * cos - in1 * sin;
                out1 = in1 * cos + in0 * sin;
                key_ptr[out_offset_k + h] = out0;
                key_ptr[out_offset_k + h + offset] = out1;
              }
            }
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

} // namespace kernel
} // namespace cpu
} // namespace zentorch
