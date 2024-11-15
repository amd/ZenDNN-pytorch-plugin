/******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Was sourced from
 * https://github.com/intel/intel-extension-for-pytorch/blob/v2.4.0%2Bcpu/csrc/cpu/aten/kernels/MaskedMultiHeadAttentionKrnl.cpp
 * IPEX commit ID: 070f1d7
 ******************************************************************************/

#include <ATen/Tensor.h>
#include <cpuinfo.h>
#include <limits>
#include <omp.h>
#include <torch/all.h>

#include "vec/vec512_bfloat16.h"

#include "vec/add_softmax.h"
#include "zen_MaskedMultiHeadAttention.hpp"

#define AVX512_BF16_COMPUTE_ENABLE 1
#define AVX512_BF16_STORE_ENABLE 1

namespace zentorch {

template <typename T>
inline void reduce_head(const T *q_ptr_start, const T *k_ptr_start,
                        float *attn_w_pos, int64_t head_size, bool store_key,
                        T *k_cache_start) {
  for (auto hsi = 0; hsi < head_size; hsi++) {
    if (store_key) {
      k_cache_start[hsi] = k_ptr_start[hsi]; // cat the key into the key_cache.
    }
    attn_w_pos[0] += q_ptr_start[hsi] * k_ptr_start[hsi];
  }
}

#if defined(CPU_CAPABILITY_AVX512)
template <>
inline void reduce_head(const float *q_ptr_start, const float *k_ptr_start,
                        float *attn_w_pos, int64_t head_size, bool store_key,
                        float *k_cache_start) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32
  auto qk_sum_vec = _mm512_setzero_ps();
  if (store_key) {
    for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
      auto q_vec = _mm512_loadu_ps(q_ptr_start + hsi);
      auto k_vec = _mm512_loadu_ps(k_ptr_start + hsi);
      _mm512_storeu_ps(k_cache_start + hsi, k_vec);
      qk_sum_vec = _mm512_fmadd_ps(q_vec, k_vec, qk_sum_vec);
    }
  } else {
    for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
      auto q_vec = _mm512_loadu_ps(q_ptr_start + hsi);
      auto k_vec = _mm512_loadu_ps(k_ptr_start + hsi);
      qk_sum_vec = _mm512_fmadd_ps(q_vec, k_vec, qk_sum_vec);
    }
  }
  attn_w_pos[0] += _mm512_reduce_add_ps(qk_sum_vec);
  for (; hsi < head_size; hsi++) {
    if (store_key)
      k_cache_start[hsi] = k_ptr_start[hsi]; // cat the key into the key_cache.
    attn_w_pos[0] += q_ptr_start[hsi] * k_ptr_start[hsi];
  }
}

template <>
inline void reduce_head(const at::BFloat16 *q_ptr_start,
                        const at::BFloat16 *k_ptr_start, float *attn_w_pos,
                        int64_t head_size, bool store_key,
                        at::BFloat16 *k_cache_start) {
  auto hsi = 0;
  auto vec_size = 32; // 512/16
  auto qk_sum_vec = _mm512_setzero_ps();
#if AVX512_BF16_COMPUTE_ENABLE
  if (store_key) {
    for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
      auto q_vec_bf16 = (__m512bh)_mm512_loadu_si512(q_ptr_start + hsi);
      auto k_vec_bf16 = (__m512bh)_mm512_loadu_si512(k_ptr_start + hsi);
      _mm512_storeu_si512(k_cache_start + hsi, (__m512i)k_vec_bf16);
      qk_sum_vec = _mm512_dpbf16_ps(qk_sum_vec, q_vec_bf16, k_vec_bf16);
    }
  } else {
    for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
      auto q_vec_bf16 = (__m512bh)_mm512_loadu_si512(q_ptr_start + hsi);
      auto k_vec_bf16 = (__m512bh)_mm512_loadu_si512(k_ptr_start + hsi);
      qk_sum_vec = _mm512_dpbf16_ps(qk_sum_vec, q_vec_bf16, k_vec_bf16);
    }
  }
  vec_size = 16;
  for (; hsi <= head_size - vec_size; hsi += vec_size) {
#else
  vec_size = 16; // 512/32
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
#endif
    // load 16 bfloat16 query from q_ptr_start and convert to 16 float32 values
    auto q_vec_bf16 = _mm256_loadu_si256((__m256i *)(q_ptr_start + hsi));
    auto q_vec_fp32 = zentorch::convert_bf16_to_fp32(q_vec_bf16);
    // load 16 bfloat16 key from k_ptr_start and convert to 16 float32 values
    auto k_vec_bf16 = _mm256_loadu_si256((__m256i *)(k_ptr_start + hsi));
    auto k_vec_fp32 = zentorch::convert_bf16_to_fp32(k_vec_bf16);
    if (store_key) {
      _mm256_storeu_si256((__m256i *)(k_cache_start + hsi), k_vec_bf16);
    }
    qk_sum_vec = _mm512_fmadd_ps(q_vec_fp32, k_vec_fp32, qk_sum_vec);
  }
  attn_w_pos[0] += (at::BFloat16)_mm512_reduce_add_ps(qk_sum_vec);
  for (; hsi < head_size; hsi++) {
    if (store_key)
      k_cache_start[hsi] = k_ptr_start[hsi]; // cat the key into the key_cache.
    attn_w_pos[0] += q_ptr_start[hsi] * k_ptr_start[hsi];
  }
  return;
}
#endif

template <typename T>
inline void reduce_head(const T *q_ptr_start, int64_t kv_head_group_size,
                        const T *k_ptr_start, float *attn_w_pos,
                        int attn_w_stride, int64_t head_size, bool store_key,
                        T *k_cache_start) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    attn_w_pos[i * attn_w_stride] = 0;
    reduce_head<T>(q_ptr_start + i * head_size, k_ptr_start,
                   attn_w_pos + i * attn_w_stride, head_size, store_key,
                   k_cache_start);
  }
}

template <typename T>
inline void reduce_head(const T *q_ptr_start, int qStrideB,
                        int64_t kv_head_group_size, const T *k_ptr_start,
                        float *attn_w_pos, int attn_w_stride, int64_t head_size,
                        int64_t beam_size) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    for (auto b = 0; b < beam_size; b++) {
      attn_w_pos[i * attn_w_stride + b] = 0;
      reduce_head<T>(q_ptr_start + i * head_size + b * qStrideB, k_ptr_start,
                     attn_w_pos + i * attn_w_stride + b, head_size, false,
                     nullptr);
    }
  }
}

/*
 *reduce the attention_weights with the value embedding by the dimension of
 *head_size for every head
 */
// zentorch:: mul_attenion_weights_and_value_of_head_256 will be invoked on
// AVX256 m/c

template <typename T, typename T1>
inline void mul_attenion_weights_and_value_of_head(
    float &attn_w, const T *v_ptr_start, T1 *attn_out_start, int64_t head_size,
    bool store_value, T *v_cache_start, bool accumulate) {
  for (auto hsi = 0; hsi < head_size; hsi++) {
    if (accumulate) {
      attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * v_ptr_start[hsi];
    }
    if (store_value) {
      v_cache_start[hsi] = v_ptr_start[hsi];
    }
  }
}

template <typename T, typename T1>
inline void mul_attenion_weights_and_value_of_head(
    float *attn_w, int attn_w_stride, const T *v_ptr_start, T1 *attn_out_start,
    int attn_out_strideH, int kv_head_group_size, int64_t head_size,
    bool store_value, T *v_cache_start, uint8_t *flag_access) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    mul_attenion_weights_and_value_of_head<T, T1>(
        attn_w[i * attn_w_stride], v_ptr_start,
        attn_out_start + i * attn_out_strideH, head_size, store_value,
        v_cache_start, flag_access[i]);
    if (flag_access[i] == 0)
      flag_access[i] = 1;
  }
}

template <typename T, typename T1>
inline void mul_attenion_weights_and_value_of_head(
    float *attn_w, int attn_w_stride, const T *v_ptr_start, T1 *attn_out_start,
    int attn_out_strideB, int attn_out_strideH, int kv_head_group_size,
    int64_t head_size, bool store_value, T *v_cache_start, uint8_t *flag_access,
    int flag_access_stride, int64_t beam_size) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    for (auto b = 0; b < beam_size; b++) {
      mul_attenion_weights_and_value_of_head<T, T1>(
          attn_w[i * attn_w_stride + b], v_ptr_start,
          attn_out_start + i * attn_out_strideH + b * attn_out_strideB,
          head_size, store_value, v_cache_start,
          flag_access[b * flag_access_stride + i]);
      if (flag_access[b * flag_access_stride + i] == 0)
        flag_access[b * flag_access_stride + i] = 1;
    }
  }
}

#if defined(CPU_CAPABILITY_AVX512)
template <>
inline void
mul_attenion_weights_and_value_of_head(float &attn_w, const float *v_ptr_start,
                                       float *attn_out_start, int64_t head_size,
                                       bool store_value, float *v_cache_start,
                                       bool accumulate) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    auto attn_w_vec = _mm512_set1_ps(attn_w);
    auto v_vec = _mm512_loadu_ps(v_ptr_start + hsi);
    if (accumulate) {
      auto attn_out_vec = _mm512_loadu_ps(attn_out_start + hsi);
      auto attn_out_vec_new = _mm512_fmadd_ps(attn_w_vec, v_vec, attn_out_vec);
      _mm512_storeu_ps(attn_out_start + hsi, attn_out_vec_new);
    } else {
      auto attn_out_vec_new = _mm512_mul_ps(attn_w_vec, v_vec);
      _mm512_storeu_ps(attn_out_start + hsi, attn_out_vec_new);
    }
    if (store_value) {
      _mm512_storeu_ps(v_cache_start + hsi, v_vec);
    }
  }
  for (; hsi < head_size; hsi++) {
    if (accumulate) {
      attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * v_ptr_start[hsi];
    }
    if (store_value) {
      v_cache_start[hsi] = v_ptr_start[hsi];
    }
  }
  return;
}

template <>
inline void mul_attenion_weights_and_value_of_head(
    float &attn_w, const at::BFloat16 *v_ptr_start,
    at::BFloat16 *attn_out_start, int64_t head_size, bool store_value,
    at::BFloat16 *v_cache_start, bool accumulate) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32

  // TODO Enable AVX512  BF16 load, store and compute
  if (accumulate) {
    if (store_value) {
      for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
        // get 1 bfloat16 values from attn_w_ptr_start and broadcast to 16
        // float32 values
        auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
        // load 16 bfloat16 values from v_ptr_start and convert to 16 float32
        // values
        auto v_vec_bf16 = _mm256_loadu_si256((__m256i *)(v_ptr_start + hsi));
        auto v_vec_fp32 = zentorch::convert_bf16_to_fp32(v_vec_bf16);
        // load 16 bfloat16 values from attn_out_start and convert to 16 float32
        // values
        auto attn_out_vec_fp32 = zentorch::convert_bf16_to_fp32(
            _mm256_loadu_si256((__m256i *)(attn_out_start + hsi)));
        // calculate the new attn_out_vec_fp32 and convert to bfloat16
        auto attn_out_vec_new =
            _mm512_fmadd_ps(attn_w_vec_fp32, v_vec_fp32, attn_out_vec_fp32);
        auto attn_out_vec_new_bf16 =
            cvt_fp32_to_bf16(attn_out_vec_new); //_m256i
        // store the new attn_out_vec_new_bf16 to attn_outs
        _mm256_storeu_si256((__m256i *)(attn_out_start + hsi),
                            attn_out_vec_new_bf16);
        // store the v_vec_bf16 to v_cache
        _mm256_storeu_si256((__m256i *)(v_cache_start + hsi), v_vec_bf16);
      }
    } else {
      for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
        // get 1 bfloat16 values from attn_w_ptr_start and broadcast to 16
        // float32 values
        auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
        // load 16 bfloat16 values from v_ptr_start and convert to 16 float32
        // values
        auto v_vec_bf16 = _mm256_loadu_si256((__m256i *)(v_ptr_start + hsi));
        auto v_vec_fp32 = zentorch::convert_bf16_to_fp32(v_vec_bf16);
        // load 16 bfloat16 values from attn_out_start and convert to 16 float32
        // values
        auto attn_out_vec_fp32 = zentorch::convert_bf16_to_fp32(
            _mm256_loadu_si256((__m256i *)(attn_out_start + hsi)));
        // calculate the new attn_out_vec_fp32 and convert to bfloat16
        auto attn_out_vec_new =
            _mm512_fmadd_ps(attn_w_vec_fp32, v_vec_fp32, attn_out_vec_fp32);
        auto attn_out_vec_new_bf16 =
            cvt_fp32_to_bf16(attn_out_vec_new); //_m256i
        // store the new attn_out_vec_new_bf16 to attn_outs
        _mm256_storeu_si256((__m256i *)(attn_out_start + hsi),
                            attn_out_vec_new_bf16);
      }
    }
  } else {
    if (store_value) {
      for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
        // get 1 bfloat16 values from attn_w_ptr_start and broadcast to 16
        // float32 values
        auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
        // load 16 bfloat16 values from v_ptr_start and convert to 16 float32
        // values
        auto v_vec_bf16 = _mm256_loadu_si256((__m256i *)(v_ptr_start + hsi));
        auto v_vec_fp32 = zentorch::convert_bf16_to_fp32(v_vec_bf16);
        // calculate the new attn_out_vec_fp32 and convert to bfloat16
        auto attn_out_vec_new = _mm512_mul_ps(attn_w_vec_fp32, v_vec_fp32);
        auto attn_out_vec_new_bf16 =
            cvt_fp32_to_bf16(attn_out_vec_new); //_m256i
        // store the new attn_out_vec_new_bf16 to attn_outs
        _mm256_storeu_si256((__m256i *)(attn_out_start + hsi),
                            attn_out_vec_new_bf16);
        // store the v_vec_bf16 to v_cache
        _mm256_storeu_si256((__m256i *)(v_cache_start + hsi), v_vec_bf16);
      }
    } else {
      for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
        // get 1 bfloat16 values from attn_w_ptr_start and broadcast to 16
        // float32 values
        auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
        // load 16 bfloat16 values from v_ptr_start and convert to 16 float32
        // values
        auto v_vec_bf16 = _mm256_loadu_si256((__m256i *)(v_ptr_start + hsi));
        auto v_vec_fp32 = zentorch::convert_bf16_to_fp32(v_vec_bf16);
        // calculate the new attn_out_vec_fp32 and convert to bfloat16
        auto attn_out_vec_new = _mm512_mul_ps(attn_w_vec_fp32, v_vec_fp32);
        auto attn_out_vec_new_bf16 =
            cvt_fp32_to_bf16(attn_out_vec_new); //_m256i
        // store the new attn_out_vec_new_bf16 to attn_outs
        _mm256_storeu_si256((__m256i *)(attn_out_start + hsi),
                            attn_out_vec_new_bf16);
      }
    }
  }
  for (; hsi < head_size; hsi++) {
    if (accumulate) {
      attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * v_ptr_start[hsi];
    }
    if (store_value) {
      v_cache_start[hsi] = v_ptr_start[hsi];
    }
  }
  return;
}

template <>
inline void mul_attenion_weights_and_value_of_head(
    float &attn_w, const at::BFloat16 *v_ptr_start, float *attn_out_start,
    int64_t head_size, bool store_value, at::BFloat16 *v_cache_start,
    bool accumulate) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32

  if (accumulate) {
    if (store_value) {
      for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
        // get 1 bfloat16 values from attn_w_ptr_start and broadcast to 16
        // float32 values
        auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
        // load 16 bfloat16 values from v_ptr_start and convert to 16 float32
        // values
        auto v_vec_bf16 = _mm256_loadu_si256((__m256i *)(v_ptr_start + hsi));
        auto v_vec_fp32 = zentorch::convert_bf16_to_fp32(v_vec_bf16);
        auto attn_out_vec_fp32 = _mm512_loadu_ps(attn_out_start + hsi);
        // calculate the new attn_out_vec_fp32 and convert to bfloat16
        auto attn_out_vec_new =
            _mm512_fmadd_ps(attn_w_vec_fp32, v_vec_fp32, attn_out_vec_fp32);
        _mm512_storeu_ps(attn_out_start + hsi, attn_out_vec_new);
        // store the v_vec_bf16 to v_cache
        _mm256_storeu_si256((__m256i *)(v_cache_start + hsi), v_vec_bf16);
      }
    } else {
      for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
        // get 1 bfloat16 values from attn_w_ptr_start and broadcast to 16
        // float32 values
        auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
        // load 16 bfloat16 values from v_ptr_start and convert to 16 float32
        // values
        auto v_vec_bf16 = _mm256_loadu_si256((__m256i *)(v_ptr_start + hsi));
        auto v_vec_fp32 = zentorch::convert_bf16_to_fp32(v_vec_bf16);
        auto attn_out_vec_fp32 = _mm512_loadu_ps(attn_out_start + hsi);
        // calculate the new attn_out_vec_fp32 and convert to bfloat16
        auto attn_out_vec_new =
            _mm512_fmadd_ps(attn_w_vec_fp32, v_vec_fp32, attn_out_vec_fp32);
        _mm512_storeu_ps(attn_out_start + hsi, attn_out_vec_new);
      }
    }
  } else {
    if (store_value) {
      for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
        // get 1 bfloat16 values from attn_w_ptr_start and broadcast to 16
        // float32 values
        auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
        // load 16 bfloat16 values from v_ptr_start and convert to 16 float32
        // values
        auto v_vec_bf16 = _mm256_loadu_si256((__m256i *)(v_ptr_start + hsi));
        auto v_vec_fp32 = zentorch::convert_bf16_to_fp32(v_vec_bf16);
        // calculate the new attn_out_vec_fp32 and convert to bfloat16
        auto attn_out_vec_new = _mm512_mul_ps(attn_w_vec_fp32, v_vec_fp32);
        _mm512_storeu_ps(attn_out_start + hsi, attn_out_vec_new);
        // store the v_vec_bf16 to v_cache
        _mm256_storeu_si256((__m256i *)(v_cache_start + hsi), v_vec_bf16);
      }
    } else {
      for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
        // get 1 bfloat16 values from attn_w_ptr_start and broadcast to 16
        // float32 values
        auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
        // load 16 bfloat16 values from v_ptr_start and convert to 16 float32
        // values
        auto v_vec_bf16 = _mm256_loadu_si256((__m256i *)(v_ptr_start + hsi));
        auto v_vec_fp32 = zentorch::convert_bf16_to_fp32(v_vec_bf16);
        // calculate the new attn_out_vec_fp32 and convert to bfloat16
        auto attn_out_vec_new = _mm512_mul_ps(attn_w_vec_fp32, v_vec_fp32);
        _mm512_storeu_ps(attn_out_start + hsi, attn_out_vec_new);
      }
    }
  }
  for (; hsi < head_size; hsi++) {
    if (accumulate) {
      attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * v_ptr_start[hsi];
    }
    if (store_value) {
      v_cache_start[hsi] = v_ptr_start[hsi];
    }
  }
  return;
}

#endif

template <typename T>
inline void copy_key_value(at::Tensor key_cache, const at::Tensor key,
                           at::Tensor value_cache, const at::Tensor value,
                           int beam_batch) {
  RECORD_FUNCTION("zentorch::copy_key_value", c10::ArrayRef<c10::IValue>({}));
  auto bs = key.size(0);
  auto seq_len = key.size(1); // only process cur_len==1
  auto head_num = key.size(2);
  auto head_size = key.size(3);
  auto hidden_size = head_num * head_size;
  auto key_cache_ptr = key_cache.data_ptr<T>();
  auto key_ptr = key.data_ptr<T>();
  auto value_cache_ptr = value_cache.data_ptr<T>();
  auto value_ptr = value.data_ptr<T>();
  auto token_stride = beam_batch * hidden_size;
  auto beam_size = beam_batch / bs;
  // zentorch:: Conditionally picks AVX512/AVX256 kernels based on m/c support
#pragma omp parallel for collapse(2)
  for (auto si = 0; si < seq_len; si++) {
    for (auto bi = 0; bi < bs; bi++) {
      auto cache_stride = si * token_stride + bi * beam_size * hidden_size;
      auto state_stride = (bi * seq_len + si) * hidden_size;
      auto key_cache_start = key_cache_ptr + cache_stride;
      auto key_ptr_start = key_ptr + state_stride;
      zentorch::move_ker<T, T>(key_cache_start, key_ptr_start, hidden_size);
      auto value_cache_ptr_start = value_cache_ptr + cache_stride;
      auto value_ptr_start = value_ptr + state_stride;
      zentorch::move_ker<T, T>(value_cache_ptr_start, value_ptr_start,
                               hidden_size);
    }
  }
  return;
}

/*
 *The scale-dot product for indirect access kv chache and fuse
 *matmul+div+add+softmax to improve data reuse
 *@param  query Query embeeding with the of [beam_size*batch, cur_len, head_num,
 *head_size]
 *@param  key Key embeeding with the of [beam_size*batch, cur_len, head_num,
 *head_size]
 *@param  value Key embeeding with the of [beam_size*batch, cur_len, head_num,
 *head_size]
 *@param  key_cache Cache past key embeeding with the of [max_len,
 *beam_size*batch, head_num, head_size]
 *@param  value_chache Cache past value embeeding with the of [max_len,
 *beam_size*batch, head_num, head_size]
 *@param  beam_idx Beam info for every token [max_len, beam_size*batch]
 *@param  offset  The length of decoded(past) token.
 *@param  scale_factor the sqrt(head_dim).
 *@param  head_mask Which is not used by our kernel now.
 *@param  attention_mask Which is combined mask for padding mask and casual
 *mask.
 *@return attn_outs, None, key_cache, value_cache, beam_idx
 */
template <typename QT, typename VT>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
scale_dot_product_for_indirect_access_kv_cache(
    at::Tensor query, at::Tensor key, at::Tensor value, at::Tensor &key_cache,
    at::Tensor &value_cache, at::Tensor &beam_idx, const int64_t offset,
    const double scale_factor, at::Tensor &attention_mask) {

  RECORD_FUNCTION("zentorch::scale_dot_product_for_indirect_access_kv_cache",
                  c10::ArrayRef<c10::IValue>({}));
  int beam_batch = beam_idx.size(1);
  auto bs = query.size(0);
  auto cur_len = query.size(1); // only process cur_len==1
  auto head_num = query.size(2);
  auto head_size = query.size(3);
  auto b_ptr = beam_idx.data_ptr<long>();
  auto max_cache_size = beam_idx.size(0);
  long new_beam_idx[beam_batch][offset + query.size(1) + 1] = {};
  auto prompt_len = b_ptr[(max_cache_size - 2) * beam_batch];
  auto prompt_bs = b_ptr[(max_cache_size - 1) * beam_batch];
  auto beam_size = 1;
  if (prompt_bs != 0) {
    beam_size = beam_batch / prompt_bs;
  }
  auto need_update_beam_idx = offset > 0 and beam_size > 1;
  auto kv_head = key.size(2);
  auto group_size = head_num / kv_head;
  auto seq_len = offset + cur_len;
  at::Tensor attn_weights, attn_weights2;
  bool change_attn_w_layout = false;
  auto target_bs = bs;
  if (beam_size > 1 && prompt_len <= 2048 && prompt_bs > 20 &&
      group_size == 1) {
    change_attn_w_layout = true;
    attn_weights = at::empty({prompt_bs, head_num, cur_len, seq_len, beam_size},
                             at::kFloat);
    attn_weights2 = at::empty(
        {prompt_bs, head_num, cur_len, beam_size, seq_len}, at::kFloat);
    target_bs = prompt_bs;
  } else {
    attn_weights = at::empty({bs, head_num, cur_len, seq_len}, at::kFloat);
    attn_weights2 = attn_weights;
  }

  query = query.contiguous();
  key = key.contiguous();
  auto q_ptr = query.data_ptr<QT>();
  auto k_ptr = key.data_ptr<QT>();
  auto k_cache_ptr = key_cache.data_ptr<QT>();
  auto mask_ptr = attention_mask.data_ptr<QT>();
  auto mask_head_num = attention_mask.size(1);
  auto mask_dim2 = attention_mask.size(2);
  auto mask_bs_stride = mask_head_num * mask_dim2 * seq_len;

  // value realted
  value = value.contiguous();
  auto attn_outs =
      at::empty({bs, head_num, cur_len, head_size}, value.options());
  auto v_ptr = value.data_ptr<VT>();
  auto v_cache_ptr = value_cache.data_ptr<VT>();
  auto attn_out_ptr = attn_outs.data_ptr<VT>();
  // zentorch::zero_ker(attn_out_ptr, attn_outs.numel());
  auto attn_w_ptr = attn_weights.data_ptr<float>();
  auto attn_w_ptr2 = attn_weights2.data_ptr<float>();

  // stride information
  auto qStrideB = query.stride(0);
  // auto qStrideS = query.stride(1);
  auto qStrideH = query.stride(2);

  auto kStrideB = key.stride(0);
  // auto kStrideS = key.stride(1);
  auto kStrideH = key.stride(2);

  auto kcStrideB = key_cache.stride(1);
  auto kcStrideS = key_cache.stride(0);
  auto kcStrideH = key_cache.stride(2);

  auto vStrideB = value.stride(0);
  // auto vStrideS = value.stride(1);
  auto vStrideH = value.stride(2);

  auto vcStrideB = value_cache.stride(1);
  auto vcStrideS = value_cache.stride(0);
  auto vcStrideH = value_cache.stride(2);

  auto attn_w_strideH = attn_weights.stride(1);

  auto thread_numbers = omp_get_max_threads();
  auto max_parallel_parts = thread_numbers * 4;

  // TODO: Generate more heuristics based on bs, seq_len
  // and device cache capacity. Decide more fine grain
  // kv_block_size for lower bs. Current target_block_size
  // works optimally for bs >=4.
  auto target_block_size = 128L;
  if (target_bs <= 8 and seq_len < 65536) {
    target_block_size = 32L;
  }
  auto kv_block_size = target_bs * head_num >= max_parallel_parts
                           ? seq_len
                           : std::max(seq_len / max_parallel_parts, 1L);
  kv_block_size = std::min(kv_block_size, target_block_size);
  auto kv_block_count = (seq_len + kv_block_size - 1) / kv_block_size;
  if (need_update_beam_idx) {
    // according to the last decoded token to get the target beam for the past
    // token
#pragma omp parallel for
    for (int i = 0; i < bs; i++) {
      new_beam_idx[i][offset - 1] = b_ptr[(offset - 1) * bs + i];
      for (int j = offset - 2; j >= prompt_len;
           j--) { // for the token of input, the target beam is alwarys 0
        new_beam_idx[i][j] = b_ptr[j * bs + new_beam_idx[i][j + 1]];
      }
    }
  }
  {
    RECORD_FUNCTION("zentorch::iakv_sdp::matmul(query, key)",
                    c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(3)
    for (auto block_id = 0; block_id < kv_block_count; block_id++) {
      for (auto bsi = 0; bsi < prompt_bs; bsi++) {
        for (auto head_group_start = 0; head_group_start < head_num;
             head_group_start += group_size) {
          auto k_start = block_id * kv_block_size;
          auto block_size = std::min(kv_block_size, seq_len - k_start);
          auto query_ti = 0;
          // maping the query head to key/value head to support MGA/MQA
          auto kv_hi = head_group_start / group_size;
          if (change_attn_w_layout) {
            auto attn_w_stride =
                (bsi * head_num + head_group_start) * attn_w_strideH;
            for (auto ti = k_start; ti < k_start + block_size; ti++) {
              // caculate the innerproduct for the current token and store the
              // key
              if (ti == query_ti + offset) {
                for (auto bbi = 0; bbi < beam_size; bbi++) {
                  auto bi = bsi * beam_size + bbi;
                  auto q_ptr_start =
                      q_ptr + bi * qStrideB + head_group_start * qStrideH;
                  auto attn_w_pos = attn_w_ptr + attn_w_stride +
                                    query_ti * seq_len + ti * beam_size + bbi;
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                                       bi * kcStrideB + kv_hi * kcStrideH;
                  auto k_ptr_start = k_ptr + bi * kStrideB + kv_hi * kStrideH;
                  reduce_head<QT>(q_ptr_start, group_size, k_ptr_start,
                                  attn_w_pos, attn_w_strideH, head_size, true,
                                  kc_head_start);
                }
              } else {
                auto bi = bsi * beam_size;
                auto q_ptr_start =
                    q_ptr + bi * qStrideB + head_group_start * qStrideH;
                auto attn_w_pos = attn_w_ptr + attn_w_stride +
                                  query_ti * seq_len + ti * beam_size;
                if (need_update_beam_idx && ti >= prompt_len) {
                  for (auto bbi = 0; bbi < beam_size; bbi++) {
                    auto beam = new_beam_idx[bi + bbi][ti];
                    auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                                         beam * kcStrideB + kv_hi * kcStrideH;
                    reduce_head<QT>(q_ptr_start + bbi * qStrideB, group_size,
                                    kc_head_start, attn_w_pos + bbi,
                                    attn_w_strideH, head_size, false, nullptr);
                  }
                } else {
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                                       bi * kcStrideB + kv_hi * kcStrideH;
                  reduce_head<QT>(q_ptr_start, qStrideB, group_size,
                                  kc_head_start, attn_w_pos, attn_w_strideH,
                                  head_size, beam_size);
                }
              }
            }
          } else {
            for (auto bbi = 0; bbi < beam_size; bbi++) {
              auto bi = bsi * beam_size + bbi;
              for (auto ti = k_start; ti < k_start + block_size; ti++) {
                auto q_ptr_start =
                    q_ptr + bi * qStrideB + head_group_start * qStrideH;
                auto attn_w_stride =
                    (bi * head_num + head_group_start) * attn_w_strideH;
                auto attn_w_pos =
                    attn_w_ptr + attn_w_stride + query_ti * seq_len + ti;
                attn_w_pos[0] = 0.0f;
                auto beam = need_update_beam_idx && ti >= prompt_len
                                ? new_beam_idx[bi][ti]
                                : bsi * beam_size;
                // caculate the innerproduct for the current token and store the
                // key
                if (ti == query_ti + offset) {
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                                       bi * kcStrideB + kv_hi * kcStrideH;
                  auto k_ptr_start = k_ptr + bi * kStrideB + kv_hi * kStrideH;
                  reduce_head<QT>(q_ptr_start, group_size, k_ptr_start,
                                  attn_w_pos, attn_w_strideH, head_size, true,
                                  kc_head_start);
                } else { // caculate the innerproduct for the past token
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                                       beam * kcStrideB + kv_hi * kcStrideH;
                  reduce_head<QT>(q_ptr_start, group_size, kc_head_start,
                                  attn_w_pos, attn_w_strideH, head_size, false,
                                  nullptr);
                }
              }
            }
          }
        }
      }
    }
  }
  {
    RECORD_FUNCTION("zentorch::iakv_sdp::div_add_softmax",
                    c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(2)
    for (auto bsi = 0; bsi < prompt_bs; bsi++) {
      for (auto hi = 0; hi < head_num; hi++) {
        for (auto query_ti = 0; query_ti < cur_len; query_ti++) {
          for (auto bbi = 0; bbi < beam_size; bbi++) {
            auto bi = bsi * beam_size + bbi;
            auto mask_ptr_start = mask_ptr + bi * mask_bs_stride +
                                  (hi % mask_head_num) * mask_dim2 * seq_len;

// div+add+softmax
#if defined(CPU_CAPABILITY_AVX512)
            auto max_val = -100000.0f;
            if (change_attn_w_layout) {
              auto attn_w_stride =
                  (bsi * head_num + hi) * cur_len * seq_len * beam_size;
              auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                                        query_ti * seq_len * beam_size + bbi;
              auto attn_w_query_start2 = attn_w_ptr2 + attn_w_stride +
                                         query_ti * beam_size * seq_len +
                                         bbi * seq_len;
              __m512i decrement_sequence = _mm512_set_epi32(
                  15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
              __m512i beam_size_vector = _mm512_set1_epi32(beam_size);
              int ti = 0;
              for (ti = 0; ti <= seq_len - 16; ti += 16) {
                __m512i ti_vector = _mm512_set1_epi32(ti);
                __m512i index_sequence =
                    _mm512_add_epi32(decrement_sequence, ti_vector);
                __m512i index =
                    _mm512_mullo_epi32(index_sequence, beam_size_vector);

                __m512 data = _mm512_i32gather_ps(index, attn_w_query_start,
                                                  sizeof(float));
                _mm512_storeu_ps(attn_w_query_start2 + ti, data);
              }

              for (; ti < seq_len; ti++) {
                attn_w_query_start2[ti] = attn_w_query_start[ti * beam_size];
              }
              zentorch::_dil_div_add_reduce_max_fusion_kernel<float, QT>(
                  attn_w_query_start2,
                  mask_ptr_start + (query_ti % mask_dim2) * seq_len,
                  scale_factor, seq_len, attn_w_query_start2, max_val);

              zentorch::_dil_exp_reduce_sum_fusion_kernel(
                  attn_w_query_start2, seq_len, attn_w_query_start2, max_val);
              zentorch::_dil_normalization_kernel<float>(
                  attn_w_query_start2, max_val, seq_len, attn_w_query_start2);
              for (ti = 0; ti <= seq_len - 16; ti += 16) {
                __m512i ti_vector = _mm512_set1_epi32(ti);
                __m512i index_sequence =
                    _mm512_add_epi32(decrement_sequence, ti_vector);
                __m512i index =
                    _mm512_mullo_epi32(index_sequence, beam_size_vector);
                __m512 data = _mm512_loadu_ps(attn_w_query_start2 + ti);
                _mm512_i32scatter_ps(attn_w_query_start, index, data,
                                     sizeof(float));
              }
              for (; ti < seq_len; ti++) {
                attn_w_query_start[ti * beam_size] = attn_w_query_start2[ti];
              }
            } else {
              auto attn_w_stride = (bi * head_num + hi) * cur_len * seq_len;
              auto attn_w_query_start =
                  attn_w_ptr + attn_w_stride + query_ti * seq_len;
              zentorch::_dil_div_add_reduce_max_fusion_kernel<float, QT>(
                  attn_w_query_start,
                  mask_ptr_start + (query_ti % mask_dim2) * seq_len,
                  scale_factor, seq_len, attn_w_query_start, max_val);
              zentorch::_dil_exp_reduce_sum_fusion_kernel(
                  attn_w_query_start, seq_len, attn_w_query_start, max_val);
              zentorch::_dil_normalization_kernel<float>(
                  attn_w_query_start, max_val, seq_len, attn_w_query_start);
            }
          }
#else
            auto max_val = -100000.0f;
            if (change_attn_w_layout) {
              auto attn_w_stride =
                  (bsi * head_num + hi) * cur_len * seq_len * beam_size;
              auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                                        query_ti * seq_len * beam_size + bbi;
              auto total_len = seq_len * beam_size;
              // div+add and find max
              for (auto si = 0; si < total_len; si += beam_size) {
                attn_w_query_start[si] =
                    attn_w_query_start[si] / scale_factor +
                    mask_ptr_start[(query_ti % mask_dim2) * seq_len +
                                   si / beam_size];
                if (attn_w_query_start[si] > max_val) {
                  max_val = attn_w_query_start[si];
                }
              }
              // softmax
              float sum = 0.0f;
              // exp and sum
              for (auto si = 0; si < total_len; si += beam_size) {
                attn_w_query_start[si] = exp(attn_w_query_start[si] - max_val);
                sum += attn_w_query_start[si];
              }
              // normalization
              for (auto si = 0; si < total_len; si += beam_size) {
                attn_w_query_start[si] = attn_w_query_start[si] / sum;
              }
            } else {
              auto attn_w_stride = (bi * head_num + hi) * cur_len * seq_len;
              auto attn_w_query_start =
                  attn_w_ptr + attn_w_stride + query_ti * seq_len;
              // div+add and find max
              for (auto si = 0; si < seq_len; si++) {
                attn_w_query_start[si] =
                    attn_w_query_start[si] / scale_factor +
                    mask_ptr_start[(query_ti % mask_dim2) * seq_len + si];
                if (attn_w_query_start[si] > max_val) {
                  max_val = attn_w_query_start[si];
                }
              }
              // softmax
              float sum = 0.0f;
              // exp and sum
              for (auto si = 0; si < seq_len; si++) {
                attn_w_query_start[si] = exp(attn_w_query_start[si] - max_val);
                sum += attn_w_query_start[si];
              }
              // normalization
              for (auto si = 0; si < seq_len; si++) {
                attn_w_query_start[si] = attn_w_query_start[si] / sum;
              }
            }
#endif
        }
      }
    }
  }
  auto private_attn_outs =
      at::empty({thread_numbers, bs, head_num, cur_len, head_size}, at::kFloat);
  auto private_attn_out_flag =
      at::zeros({thread_numbers, bs, head_num}, at::kByte);
  auto flag_access = private_attn_out_flag.accessor<uint8_t, 3>();
  uint8_t *flag_access_ptr = flag_access.data();
  auto private_attn_out_ptr = private_attn_outs.data_ptr<float>();
  // private_attn_outs.numel());
  auto attn_outs_stride_privT = private_attn_outs.stride(0);
  auto attn_outs_stride_privB = private_attn_outs.stride(1);
  auto attn_outs_stride_privH = private_attn_outs.stride(2);

  {
    RECORD_FUNCTION("zentorch::iakv_sdp::matmul(attn_w, value)",
                    c10::ArrayRef<c10::IValue>({}));
// TODO Enable BF16 compute for mul_attenion_weights_and_value_of_head
#pragma omp parallel for collapse(3)
    for (auto block_id = 0; block_id < kv_block_count; block_id++) {
      for (auto bsi = 0; bsi < prompt_bs; bsi++) {
        for (auto hi = 0; hi < head_num; hi += group_size) {
          auto thread_id = 0;
          if (kv_block_size < seq_len)
            thread_id = omp_get_thread_num();
          auto v_start = block_id * kv_block_size;
          auto block_size = std::min(kv_block_size, seq_len - v_start);
          auto query_ti = 0;
          // maping the query head to key/value head to support MGA/MQA
          auto kv_hi = hi / group_size;
          if (change_attn_w_layout) {
            auto attn_w_stride = (bsi * head_num + hi) * attn_w_strideH;
            for (auto vi = v_start; vi < v_start + block_size; vi++) {
              if (vi == offset) {
                for (auto bbi = 0; bbi < beam_size; bbi++) {
                  auto bi = bsi * beam_size + bbi;
                  auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                                            query_ti * seq_len +
                                            vi * beam_size + bbi;
                  // calculate weighted value and store the result to
                  // attn_outs[bs, head_num, cur_len, head_size]
                  auto attn_out_start = private_attn_out_ptr +
                                        thread_id * attn_outs_stride_privT +
                                        bi * attn_outs_stride_privB +
                                        hi * attn_outs_stride_privH;
                  auto flag_access_start = flag_access_ptr +
                                           head_num * bs * thread_id +
                                           head_num * bi + hi;
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                                            bi * vcStrideB + kv_hi * vcStrideH;
                  auto v_ptr_start = v_ptr + bi * vStrideB + kv_hi * vStrideH;
                  mul_attenion_weights_and_value_of_head<VT, float>(
                      attn_w_query_start, attn_w_strideH, v_ptr_start,
                      attn_out_start, head_size, group_size, head_size, true,
                      v_cache_head_start, flag_access_start);
                }
              } else {
                // caculate the innerproduct for the past token
                if (need_update_beam_idx && vi >= prompt_len) {
                  for (auto bbi = 0; bbi < beam_size; bbi++) {
                    auto bi = bsi * beam_size + bbi;
                    auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                                              query_ti * seq_len +
                                              vi * beam_size + bbi;
                    // calculate weighted value and store the result to
                    // attn_outs[bs, head_num, cur_len, head_size]
                    auto attn_out_start = private_attn_out_ptr +
                                          thread_id * attn_outs_stride_privT +
                                          bi * attn_outs_stride_privB +
                                          hi * attn_outs_stride_privH;
                    auto flag_access_start = flag_access_ptr +
                                             head_num * bs * thread_id +
                                             head_num * bi + hi;
                    // auto v_ptr_start = v_ptr + bi * vStrideB + kv_hi *
                    // vStrideH;
                    auto beam = new_beam_idx[bi][vi];
                    auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                                              beam * vcStrideB +
                                              kv_hi * vcStrideH;
                    mul_attenion_weights_and_value_of_head<VT, float>(
                        attn_w_query_start, attn_w_strideH, v_cache_head_start,
                        attn_out_start, head_size, group_size, head_size, false,
                        nullptr, flag_access_start);
                  }
                } else {
                  auto bi = bsi * beam_size;
                  auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                                            query_ti * seq_len + vi * beam_size;
                  // calculate weighted value and store the result to
                  // attn_outs[bs, head_num, cur_len, head_size]
                  auto attn_out_start = private_attn_out_ptr +
                                        thread_id * attn_outs_stride_privT +
                                        bi * attn_outs_stride_privB +
                                        hi * attn_outs_stride_privH;
                  auto flag_access_start = flag_access_ptr +
                                           head_num * bs * thread_id +
                                           head_num * bi + hi;
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                                            bi * vcStrideB + kv_hi * vcStrideH;
                  mul_attenion_weights_and_value_of_head<VT, float>(
                      attn_w_query_start, attn_w_strideH, v_cache_head_start,
                      attn_out_start, attn_outs_stride_privB, head_size,
                      group_size, head_size, false, nullptr, flag_access_start,
                      head_num, beam_size);
                }
              }
            }
          } else {
            for (auto bbi = 0; bbi < beam_size; bbi++) {
              auto bi = bsi * beam_size + bbi;
              for (auto vi = v_start; vi < v_start + block_size; vi++) {
                auto attn_w_stride = (bi * head_num + hi) * attn_w_strideH;
                auto attn_w_query_start =
                    attn_w_ptr + attn_w_stride + query_ti * seq_len + vi;
                // calculate weighted value and store the result to
                // attn_outs[bs, head_num, cur_len, head_size]
                auto attn_out_start =
                    private_attn_out_ptr + thread_id * attn_outs_stride_privT +
                    bi * attn_outs_stride_privB + hi * attn_outs_stride_privH;
                auto flag_access_start = flag_access_ptr +
                                         head_num * bs * thread_id +
                                         head_num * bi + hi;
                auto beam = need_update_beam_idx && vi >= prompt_len
                                ? new_beam_idx[bi][vi]
                                : bsi * beam_size;
                // caculate the innerproduct for the current token and store the
                // key
                if (vi == offset) {
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                                            bi * vcStrideB + kv_hi * vcStrideH;
                  auto v_ptr_start = v_ptr + bi * vStrideB + kv_hi * vStrideH;
                  mul_attenion_weights_and_value_of_head<VT, float>(
                      attn_w_query_start, attn_w_strideH, v_ptr_start,
                      attn_out_start, head_size, group_size, head_size, true,
                      v_cache_head_start, flag_access_start);
                } else {
                  // caculate the innerproduct for the past token
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                                            beam * vcStrideB +
                                            kv_hi * vcStrideH;
                  mul_attenion_weights_and_value_of_head<VT, float>(
                      attn_w_query_start, attn_w_strideH, v_cache_head_start,
                      attn_out_start, head_size, group_size, head_size, false,
                      nullptr, flag_access_start);
                }
              }
            }
          }
        }
      }
    }
  }
  {
    RECORD_FUNCTION("zentorch::iakv_sdp::reduction_private_result",
                    c10::ArrayRef<c10::IValue>({}));
    // TODO Enable AVX512  BF16 load, store and add for add and move kernel
#pragma omp parallel for collapse(3)
    for (auto bi = 0; bi < bs; bi++) {
      for (auto hi = 0; hi < head_num; hi++) {
        for (auto qi = 0; qi < cur_len; qi++) {
          auto thr0_head_start = private_attn_out_ptr +
                                 bi * attn_outs_stride_privB +
                                 hi * attn_outs_stride_privH;
          if (flag_access[0][bi][hi] == 0) {
            zentorch::zero_ker(thr0_head_start, head_size);
          }
          if (kv_block_size < seq_len) {
            for (auto thread_id = 1; thread_id < thread_numbers; thread_id++) {
              if (flag_access[thread_id][bi][hi] == 0) {
                continue;
              }
              auto private_attn_out_start =
                  private_attn_out_ptr + thread_id * attn_outs_stride_privT +
                  bi * attn_outs_stride_privB + hi * attn_outs_stride_privH;
              zentorch::add_ker<float, float>(
                  thr0_head_start, private_attn_out_start, head_size);
            }
          }
          auto attn_outs_start = attn_out_ptr +
                                 (bi * head_num + hi) * cur_len * head_size +
                                 qi * head_size;
          zentorch::move_ker<VT, float>(attn_outs_start, thr0_head_start,
                                        head_size);
        }
      }
    }
  }

  return std::make_tuple(attn_outs, at::Tensor(), key_cache, value_cache,
                         beam_idx);
}

// BF16 path
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
scale_dot_product_for_indirect_access_kv_cache_bf16(
    at::Tensor query, at::Tensor key, at::Tensor value, at::Tensor &key_cache,
    at::Tensor &value_cache, at::Tensor &beam_idx, const int64_t offset,
    const double scale_factor, at::Tensor &attention_mask) {

  RECORD_FUNCTION(
      "zentorch::scale_dot_product_for_indirect_access_kv_cache_bf16",
      c10::ArrayRef<c10::IValue>({}));
  int beam_batch = beam_idx.size(1);
  auto bs = query.size(0);
  auto cur_len = query.size(1); // only process cur_len==1
  auto head_num = query.size(2);
  auto head_size = query.size(3);
  auto b_ptr = beam_idx.data_ptr<long>();
  auto max_cache_size = beam_idx.size(0);
  long new_beam_idx[beam_batch][offset + query.size(1) + 1] = {};
  auto prompt_len = b_ptr[(max_cache_size - 2) * beam_batch];
  auto prompt_bs = b_ptr[(max_cache_size - 1) * beam_batch];
  auto beam_size = 1;
  if (prompt_bs != 0) {
    beam_size = beam_batch / prompt_bs;
  }
  auto need_update_beam_idx = offset > 0 and beam_size > 1;
  auto kv_head = key.size(2);
  auto group_size = head_num / kv_head;
  auto seq_len = offset + cur_len;
  at::Tensor attn_weights, attn_weights2;
  bool change_attn_w_layout = false;
  auto target_bs = bs;
  if (beam_size > 1 && prompt_len <= 2048 && prompt_bs > 20 &&
      group_size == 1) {
    change_attn_w_layout = true;
    attn_weights = at::empty({prompt_bs, head_num, cur_len, seq_len, beam_size},
                             at::kFloat);
    attn_weights2 = at::empty(
        {prompt_bs, head_num, cur_len, beam_size, seq_len}, at::kFloat);
    target_bs = prompt_bs;
  } else {
    attn_weights = at::empty({bs, head_num, cur_len, seq_len}, at::kFloat);
    attn_weights2 = attn_weights;
  }
  query = query.contiguous();
  key = key.contiguous();
  auto q_ptr = query.data_ptr<at::BFloat16>();
  auto k_ptr = key.data_ptr<at::BFloat16>();
  auto k_cache_ptr = key_cache.data_ptr<at::BFloat16>();
  auto mask_ptr = attention_mask.data_ptr<at::BFloat16>();
  auto mask_head_num = attention_mask.size(1);
  auto mask_dim2 = attention_mask.size(2);
  auto mask_bs_stride = mask_head_num * mask_dim2 * seq_len;
  // value realted
  value = value.contiguous();
  auto attn_outs =
      at::empty({bs, head_num, cur_len, head_size}, value.options());
  auto v_ptr = value.data_ptr<at::BFloat16>();
  auto v_cache_ptr = value_cache.data_ptr<at::BFloat16>();
  auto attn_out_ptr = attn_outs.data_ptr<at::BFloat16>();
  // zentorch::zero_ker(attn_out_ptr, attn_outs.numel());
  auto attn_w_ptr = attn_weights.data_ptr<float>();
  auto attn_w_ptr2 = attn_weights2.data_ptr<float>();

  // stride information
  auto qStrideB = query.stride(0);
  // auto qStrideS = query.stride(1);
  auto qStrideH = query.stride(2);

  auto kStrideB = key.stride(0);
  // auto kStrideS = key.stride(1);
  auto kStrideH = key.stride(2);

  auto kcStrideB = key_cache.stride(1);
  auto kcStrideS = key_cache.stride(0);
  auto kcStrideH = key_cache.stride(2);

  auto vStrideB = value.stride(0);
  // auto vStrideS = value.stride(1);
  auto vStrideH = value.stride(2);

  auto vcStrideB = value_cache.stride(1);
  auto vcStrideS = value_cache.stride(0);
  auto vcStrideH = value_cache.stride(2);

  auto attn_w_strideH = attn_weights.stride(1);

  auto thread_numbers = omp_get_max_threads();
  auto max_parallel_parts = thread_numbers * 4;

  // TODO: Generate more heuristics based on bs, seq_len
  // and device cache capacity. Decide more fine grain
  // kv_block_size for lower bs. Current target_block_size
  // works optimally for bs >=4.
  auto target_block_size = 128L;
  if (target_bs <= 8 and seq_len < 65536) {
    target_block_size = 32L;
  }
  auto kv_block_size = target_bs * head_num >= max_parallel_parts
                           ? seq_len
                           : std::max(seq_len / max_parallel_parts, 1L);
  kv_block_size = std::min(kv_block_size, target_block_size);
  auto kv_block_count = (seq_len + kv_block_size - 1) / kv_block_size;
  if (need_update_beam_idx) {
    // according to the last decoded token to get the target beam for the past
    // token
#pragma omp parallel for
    for (int i = 0; i < bs; i++) {
      new_beam_idx[i][offset - 1] = b_ptr[(offset - 1) * bs + i];
      // for the token of input, the target beam is alwarys bi - bi%beam_size
      for (int j = offset - 2; j >= prompt_len; j--) {
        new_beam_idx[i][j] = b_ptr[j * bs + new_beam_idx[i][j + 1]];
      }
    }
  }
  {
    RECORD_FUNCTION("zentorch::iakv_sdp::matmul(query, key)",
                    c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(3)
    for (auto block_id = 0; block_id < kv_block_count; block_id++) {
      for (auto bsi = 0; bsi < prompt_bs; bsi++) {
        for (auto head_group_start = 0; head_group_start < head_num;
             head_group_start += group_size) {
          auto k_start = block_id * kv_block_size;
          auto block_size = std::min(kv_block_size, seq_len - k_start);
          auto query_ti = 0;
          // maping the query head to key/value head to support MGA/MQA
          auto kv_hi = head_group_start / group_size;
          if (change_attn_w_layout) {
            auto attn_w_stride =
                (bsi * head_num + head_group_start) * attn_w_strideH;
            for (auto ti = k_start; ti < k_start + block_size; ti++) {
              // caculate the innerproduct for the current token and store the
              // key
              if (ti == query_ti + offset) {
                for (auto bbi = 0; bbi < beam_size; bbi++) {
                  auto bi = bsi * beam_size + bbi;
                  auto q_ptr_start =
                      q_ptr + bi * qStrideB + head_group_start * qStrideH;
                  auto attn_w_pos = attn_w_ptr + attn_w_stride +
                                    query_ti * seq_len + ti * beam_size + bbi;
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                                       bi * kcStrideB + kv_hi * kcStrideH;
                  auto k_ptr_start = k_ptr + bi * kStrideB + kv_hi * kStrideH;
                  reduce_head<at::BFloat16>(
                      q_ptr_start, group_size, k_ptr_start, attn_w_pos,
                      attn_w_strideH, head_size, true, kc_head_start);
                }
              } else { // caculate the innerproduct for the past token
                auto bi = bsi * beam_size;
                auto q_ptr_start =
                    q_ptr + bi * qStrideB + head_group_start * qStrideH;
                auto attn_w_pos = attn_w_ptr + attn_w_stride +
                                  query_ti * seq_len + ti * beam_size;
                if (need_update_beam_idx && ti >= prompt_len) {
                  for (auto bbi = 0; bbi < beam_size; bbi++) {
                    auto beam = new_beam_idx[bi + bbi][ti];
                    auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                                         beam * kcStrideB + kv_hi * kcStrideH;
                    reduce_head<at::BFloat16>(q_ptr_start + bbi * qStrideB,
                                              group_size, kc_head_start,
                                              attn_w_pos + bbi, attn_w_strideH,
                                              head_size, false, nullptr);
                  }
                } else {
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                                       bi * kcStrideB + kv_hi * kcStrideH;
                  reduce_head<at::BFloat16>(
                      q_ptr_start, qStrideB, group_size, kc_head_start,
                      attn_w_pos, attn_w_strideH, head_size, beam_size);
                }
              }
            }
          } else {
            for (auto bbi = 0; bbi < beam_size; bbi++) {
              auto bi = bsi * beam_size + bbi;
              for (auto ti = k_start; ti < k_start + block_size; ti++) {
                auto q_ptr_start =
                    q_ptr + bi * qStrideB + head_group_start * qStrideH;
                auto attn_w_stride =
                    (bi * head_num + head_group_start) * attn_w_strideH;
                auto attn_w_pos =
                    attn_w_ptr + attn_w_stride + query_ti * seq_len + ti;
                attn_w_pos[0] = 0.0f;
                auto beam = need_update_beam_idx && ti >= prompt_len
                                ? new_beam_idx[bi][ti]
                                : bsi * beam_size;
                // caculate the innerproduct for the current token and store the
                // key
                if (ti == query_ti + offset) {
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                                       bi * kcStrideB + kv_hi * kcStrideH;
                  auto k_ptr_start = k_ptr + bi * kStrideB + kv_hi * kStrideH;
                  reduce_head<at::BFloat16>(
                      q_ptr_start, group_size, k_ptr_start, attn_w_pos,
                      attn_w_strideH, head_size, true, kc_head_start);
                } else { // caculate the innerproduct for the past token
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                                       beam * kcStrideB + kv_hi * kcStrideH;
                  reduce_head<at::BFloat16>(
                      q_ptr_start, group_size, kc_head_start, attn_w_pos,
                      attn_w_strideH, head_size, false, nullptr);
                }
              }
            }
          }
        }
      }
    }
  }
  {
    RECORD_FUNCTION("zentorch::iakv_sdp::div_add_softmax",
                    c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(2)
    for (auto bsi = 0; bsi < prompt_bs; bsi++) {
      for (auto hi = 0; hi < head_num; hi++) {
        for (auto query_ti = 0; query_ti < cur_len; query_ti++) {
          for (auto bbi = 0; bbi < beam_size; bbi++) {
            auto bi = bsi * beam_size + bbi;
            auto mask_ptr_start = mask_ptr + bi * mask_bs_stride +
                                  (hi % mask_head_num) * mask_dim2 * seq_len;
// div+add+softmax
#if defined(CPU_CAPABILITY_AVX512)
            auto max_val = -100000.0f;
            if (change_attn_w_layout) {
              auto attn_w_stride =
                  (bsi * head_num + hi) * cur_len * seq_len * beam_size;
              auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                                        query_ti * seq_len * beam_size + bbi;
              auto attn_w_query_start2 = attn_w_ptr2 + attn_w_stride +
                                         query_ti * beam_size * seq_len +
                                         bbi * seq_len;
              __m512i decrement_sequence = _mm512_set_epi32(
                  15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
              __m512i beam_size_vector = _mm512_set1_epi32(beam_size);
              int ti = 0;
              for (ti = 0; ti <= seq_len - 16; ti += 16) {
                __m512i ti_vector = _mm512_set1_epi32(ti);
                __m512i index_sequence =
                    _mm512_add_epi32(decrement_sequence, ti_vector);
                __m512i index =
                    _mm512_mullo_epi32(index_sequence, beam_size_vector);

                __m512 data = _mm512_i32gather_ps(index, attn_w_query_start,
                                                  sizeof(float));
                _mm512_storeu_ps(attn_w_query_start2 + ti, data);
              }

              for (; ti < seq_len; ti++) {
                attn_w_query_start2[ti] = attn_w_query_start[ti * beam_size];
              }
              zentorch::_dil_div_add_reduce_max_fusion_kernel<float,
                                                              at::BFloat16>(
                  attn_w_query_start2,
                  mask_ptr_start + (query_ti % mask_dim2) * seq_len,
                  scale_factor, seq_len, attn_w_query_start2, max_val);
              zentorch::_dil_exp_reduce_sum_fusion_kernel(
                  attn_w_query_start2, seq_len, attn_w_query_start2, max_val);
              zentorch::_dil_normalization_kernel<float>(
                  attn_w_query_start2, max_val, seq_len, attn_w_query_start2);
              for (ti = 0; ti <= seq_len - 16; ti += 16) {
                __m512i ti_vector = _mm512_set1_epi32(ti);
                __m512i index_sequence =
                    _mm512_add_epi32(decrement_sequence, ti_vector);
                __m512i index =
                    _mm512_mullo_epi32(index_sequence, beam_size_vector);
                __m512 data = _mm512_loadu_ps(attn_w_query_start2 + ti);
                _mm512_i32scatter_ps(attn_w_query_start, index, data,
                                     sizeof(float));
              }
              for (; ti < seq_len; ti++) {
                attn_w_query_start[ti * beam_size] = attn_w_query_start2[ti];
              }
            } else {
              auto attn_w_stride = (bi * head_num + hi) * cur_len * seq_len;
              auto attn_w_query_start =
                  attn_w_ptr + attn_w_stride + query_ti * seq_len;
              zentorch::_dil_div_add_reduce_max_fusion_kernel<float,
                                                              at::BFloat16>(
                  attn_w_query_start,
                  mask_ptr_start + (query_ti % mask_dim2) * seq_len,
                  scale_factor, seq_len, attn_w_query_start, max_val);
              zentorch::_dil_exp_reduce_sum_fusion_kernel(
                  attn_w_query_start, seq_len, attn_w_query_start, max_val);
              zentorch::_dil_normalization_kernel<float>(
                  attn_w_query_start, max_val, seq_len, attn_w_query_start);
            }
#else
              auto max_val = -100000.0f;
              if (change_attn_w_layout) {
                auto attn_w_stride =
                    (bsi * head_num + hi) * cur_len * seq_len * beam_size;
                auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                                          query_ti * seq_len * beam_size + bbi;
                auto total_len = seq_len * beam_size;
                // div+add and find max
                for (auto si = 0; si < total_len; si += beam_size) {
                  attn_w_query_start[si] =
                      attn_w_query_start[si] / scale_factor +
                      mask_ptr_start[(query_ti % mask_dim2) * seq_len +
                                     si / beam_size];
                  if (attn_w_query_start[si] > max_val) {
                    max_val = attn_w_query_start[si];
                  }
                }
                // softmax
                float sum = 0.0f;
                // exp and sum
                for (auto si = 0; si < total_len; si += beam_size) {
                  attn_w_query_start[si] =
                      exp(attn_w_query_start[si] - max_val);
                  sum += attn_w_query_start[si];
                }
                // normalization
                for (auto si = 0; si < total_len; si += beam_size) {
                  attn_w_query_start[si] = attn_w_query_start[si] / sum;
                }
              } else {
                auto attn_w_stride = (bi * head_num + hi) * cur_len * seq_len;
                auto attn_w_query_start =
                    attn_w_ptr + attn_w_stride + query_ti * seq_len;
                // div+add and find max
                for (auto si = 0; si < seq_len; si++) {
                  attn_w_query_start[si] =
                      attn_w_query_start[si] / scale_factor +
                      mask_ptr_start[(query_ti % mask_dim2) * seq_len + si];
                  if (attn_w_query_start[si] > max_val) {
                    max_val = attn_w_query_start[si];
                  }
                }
                // softmax
                float sum = 0.0f;
                // exp and sum
                for (auto si = 0; si < seq_len; si++) {
                  attn_w_query_start[si] =
                      exp(attn_w_query_start[si] - max_val);
                  sum += attn_w_query_start[si];
                }
                // normalization
                for (auto si = 0; si < seq_len; si++) {
                  attn_w_query_start[si] = attn_w_query_start[si] / sum;
                }
              }
#endif
          }
        }
      }
    }
  }
#if AVX512_BF16_STORE_ENABLE
  auto private_attn_outs =
      at::empty({thread_numbers, bs, head_num, cur_len, head_size}, at::kHalf);
#else
    auto private_attn_outs = at::empty(
        {thread_numbers, bs, head_num, cur_len, head_size}, at::kFloat);
#endif
  auto private_attn_out_flag =
      at::zeros({thread_numbers, bs, head_num}, at::kByte);
  auto flag_access = private_attn_out_flag.accessor<uint8_t, 3>();
  uint8_t *flag_access_ptr = flag_access.data();
#if AVX512_BF16_STORE_ENABLE
  auto private_attn_out_ptr =
      (at::BFloat16 *)private_attn_outs.data_ptr<at::Half>();
#else
    auto private_attn_out_ptr = private_attn_outs.data_ptr<float>();
#endif
  // private_attn_outs.numel());
  auto attn_outs_stride_privT = private_attn_outs.stride(0);
  auto attn_outs_stride_privB = private_attn_outs.stride(1);
  auto attn_outs_stride_privH = private_attn_outs.stride(2);
  {
    RECORD_FUNCTION("zentorch::iakv_sdp::matmul(attn_w, value)",
                    c10::ArrayRef<c10::IValue>({}));
// TODO Enable BF16 compute for mul_attenion_weights_and_value_of_head
#pragma omp parallel for collapse(3)
    for (auto block_id = 0; block_id < kv_block_count; block_id++) {
      for (auto bsi = 0; bsi < prompt_bs; bsi++) {
        for (auto hi = 0; hi < head_num; hi += group_size) {
          auto thread_id = 0;
          if (kv_block_size < seq_len)
            thread_id = omp_get_thread_num();
          auto v_start = block_id * kv_block_size;
          auto block_size = std::min(kv_block_size, seq_len - v_start);
          auto query_ti = 0;
          // maping the query head to key/value head to support MGA/MQA
          auto kv_hi = hi / group_size;
          if (change_attn_w_layout) {
            auto attn_w_stride = (bsi * head_num + hi) * attn_w_strideH;
            for (auto vi = v_start; vi < v_start + block_size; vi++) {
              if (vi == offset) {
                for (auto bbi = 0; bbi < beam_size; bbi++) {
                  auto bi = bsi * beam_size + bbi;
                  auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                                            query_ti * seq_len +
                                            vi * beam_size + bbi;
                  // calculate weighted value and store the result to
                  // attn_outs[bs, head_num, cur_len, head_size]
                  auto attn_out_start = private_attn_out_ptr +
                                        thread_id * attn_outs_stride_privT +
                                        bi * attn_outs_stride_privB +
                                        hi * attn_outs_stride_privH;
                  auto flag_access_start = flag_access_ptr +
                                           head_num * bs * thread_id +
                                           head_num * bi + hi;
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                                            bi * vcStrideB + kv_hi * vcStrideH;
                  auto v_ptr_start = v_ptr + bi * vStrideB + kv_hi * vStrideH;
#if AVX512_BF16_STORE_ENABLE
                  mul_attenion_weights_and_value_of_head<at::BFloat16,
                                                         at::BFloat16>(
                      attn_w_query_start, attn_w_strideH, v_ptr_start,
                      attn_out_start, head_size, group_size, head_size, true,
                      v_cache_head_start, flag_access_start);
#else
                    mul_attenion_weights_and_value_of_head<at::BFloat16, float>(
                        attn_w_query_start, attn_w_strideH, v_ptr_start,
                        attn_out_start, head_size, group_size, head_size, true,
                        v_cache_head_start, flag_access_start);
#endif
                }
              } else {
                // caculate the innerproduct for the past token
                if (need_update_beam_idx && vi >= prompt_len) {
                  for (auto bbi = 0; bbi < beam_size; bbi++) {
                    auto bi = bsi * beam_size + bbi;
                    auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                                              query_ti * seq_len +
                                              vi * beam_size + bbi;
                    // calculate weighted value and store the result to
                    // attn_outs[bs, head_num, cur_len, head_size]
                    auto attn_out_start = private_attn_out_ptr +
                                          thread_id * attn_outs_stride_privT +
                                          bi * attn_outs_stride_privB +
                                          hi * attn_outs_stride_privH;
                    auto flag_access_start = flag_access_ptr +
                                             head_num * bs * thread_id +
                                             head_num * bi + hi;
                    // auto v_ptr_start = v_ptr + bi * vStrideB + kv_hi *
                    // vStrideH;
                    auto beam = new_beam_idx[bi][vi];
                    auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                                              beam * vcStrideB +
                                              kv_hi * vcStrideH;
#if AVX512_BF16_STORE_ENABLE
                    mul_attenion_weights_and_value_of_head<at::BFloat16,
                                                           at::BFloat16>(
                        attn_w_query_start, attn_w_strideH, v_cache_head_start,
                        attn_out_start, head_size, group_size, head_size, false,
                        nullptr, flag_access_start);
#else
                      mul_attenion_weights_and_value_of_head<at::BFloat16,
                                                             float>(
                          attn_w_query_start, attn_w_strideH,
                          v_cache_head_start, attn_out_start, head_size,
                          group_size, head_size, false, nullptr,
                          flag_access_start);
#endif
                  }
                } else {
                  auto bi = bsi * beam_size;
                  auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                                            query_ti * seq_len + vi * beam_size;
                  // calculate weighted value and store the result to
                  // attn_outs[bs, head_num, cur_len, head_size]
                  auto attn_out_start = private_attn_out_ptr +
                                        thread_id * attn_outs_stride_privT +
                                        bi * attn_outs_stride_privB +
                                        hi * attn_outs_stride_privH;
                  auto flag_access_start = flag_access_ptr +
                                           head_num * bs * thread_id +
                                           head_num * bi + hi;
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                                            bi * vcStrideB + kv_hi * vcStrideH;
#if AVX512_BF16_STORE_ENABLE
                  mul_attenion_weights_and_value_of_head<at::BFloat16,
                                                         at::BFloat16>(
                      attn_w_query_start, attn_w_strideH, v_cache_head_start,
                      attn_out_start, attn_outs_stride_privB, head_size,
                      group_size, head_size, false, nullptr, flag_access_start,
                      head_num, beam_size);
#else
                    mul_attenion_weights_and_value_of_head<at::BFloat16, float>(
                        attn_w_query_start, attn_w_strideH, v_cache_head_start,
                        attn_out_start, attn_outs_stride_privB, head_size,
                        group_size, head_size, false, nullptr,
                        flag_access_start, head_num, beam_size);
#endif
                }
              }
            }
          } else {
            for (auto bbi = 0; bbi < beam_size; bbi++) {
              auto bi = bsi * beam_size + bbi;
              for (auto vi = v_start; vi < v_start + block_size; vi++) {
                auto attn_w_stride = (bi * head_num + hi) * attn_w_strideH;
                auto attn_w_query_start =
                    attn_w_ptr + attn_w_stride + query_ti * seq_len + vi;
                // calculate weighted value and store the result to
                // attn_outs[bs, head_num, cur_len, head_size]
                auto attn_out_start =
                    private_attn_out_ptr + thread_id * attn_outs_stride_privT +
                    bi * attn_outs_stride_privB + hi * attn_outs_stride_privH;
                auto flag_access_start = flag_access_ptr +
                                         head_num * bs * thread_id +
                                         head_num * bi + hi;

                auto beam = need_update_beam_idx && vi >= prompt_len
                                ? new_beam_idx[bi][vi]
                                : bsi * beam_size;
                // caculate the innerproduct for the current token and store the
                // key
                if (vi == offset) {
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                                            bi * vcStrideB + kv_hi * vcStrideH;
                  auto v_ptr_start = v_ptr + bi * vStrideB + kv_hi * vStrideH;
#if AVX512_BF16_STORE_ENABLE
                  mul_attenion_weights_and_value_of_head<at::BFloat16,
                                                         at::BFloat16>(
                      attn_w_query_start, attn_w_strideH, v_ptr_start,
                      attn_out_start, head_size, group_size, head_size, true,
                      v_cache_head_start, flag_access_start);
#else
                    mul_attenion_weights_and_value_of_head<at::BFloat16, float>(
                        attn_w_query_start, attn_w_strideH, v_ptr_start,
                        attn_out_start, head_size, group_size, head_size, true,
                        v_cache_head_start, flag_access_start);
#endif
                } else {
                  // caculate the innerproduct for the past token
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                                            beam * vcStrideB +
                                            kv_hi * vcStrideH;
#if AVX512_BF16_STORE_ENABLE
                  mul_attenion_weights_and_value_of_head<at::BFloat16,
                                                         at::BFloat16>(
                      attn_w_query_start, attn_w_strideH, v_cache_head_start,
                      attn_out_start, head_size, group_size, head_size, false,
                      nullptr, flag_access_start);
#else
                    mul_attenion_weights_and_value_of_head<at::BFloat16, float>(
                        attn_w_query_start, attn_w_strideH, v_cache_head_start,
                        attn_out_start, head_size, group_size, head_size, false,
                        nullptr, flag_access_start);
#endif
                }
              }
            }
          }
        }
      }
    }
  }

  {
    RECORD_FUNCTION("zentorch::iakv_sdp::reduction_private_result",
                    c10::ArrayRef<c10::IValue>({}));
    // TODO Enable AVX512  BF16 load, store and add for add and move kernel
#pragma omp parallel for collapse(3)
    for (auto bi = 0; bi < bs; bi++) {
      for (auto hi = 0; hi < head_num; hi++) {
        for (auto qi = 0; qi < cur_len; qi++) {
          auto thr0_head_start = private_attn_out_ptr +
                                 bi * attn_outs_stride_privB +
                                 hi * attn_outs_stride_privH;
          if (flag_access[0][bi][hi] == 0) {
            zentorch::zero_ker(thr0_head_start, head_size);
          }
          if (kv_block_size < seq_len) {
            for (auto thread_id = 1; thread_id < thread_numbers; thread_id++) {
              if (flag_access[thread_id][bi][hi] == 0) {
                continue;
              }
              auto private_attn_out_start =
                  private_attn_out_ptr + thread_id * attn_outs_stride_privT +
                  bi * attn_outs_stride_privB + hi * attn_outs_stride_privH;
#if AVX512_BF16_STORE_ENABLE
              zentorch::add_ker<at::BFloat16, at::BFloat16>(
                  thr0_head_start, private_attn_out_start, head_size);
#else
                zentorch::add_ker<float, float>(
                    thr0_head_start, private_attn_out_start, head_size);
#endif
            }
          }
          auto attn_outs_start = attn_out_ptr +
                                 (bi * head_num + hi) * cur_len * head_size +
                                 qi * head_size;
#if AVX512_BF16_STORE_ENABLE
          zentorch::move_ker<at::BFloat16, at::BFloat16>(
              attn_outs_start, thr0_head_start, head_size);
#else
            zentorch::move_ker<at::BFloat16, float>(attn_outs_start,
                                                    thr0_head_start, head_size);
#endif
        }
      }
    }
  }

  return std::make_tuple(attn_outs, at::Tensor(), key_cache, value_cache,
                         beam_idx);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
zero_copy_kv_cache_masked_multihead_self_attention_kernel_impl(
    at::Tensor query, at::Tensor key, at::Tensor value, at::Tensor &key_cache,
    at::Tensor &value_cache, at::Tensor &beam_idx, const int64_t offset,
    const double scale_attn, at::Tensor &attention_mask) {
  assert(key.scalar_type() == at::kBFloat16 || key.scalar_type() == at::kFloat);
  if (query.scalar_type() == at::kFloat && value.scalar_type() == at::kFloat) {
    return scale_dot_product_for_indirect_access_kv_cache<float, float>(
        query, key, value, key_cache, value_cache, beam_idx, offset, scale_attn,
        attention_mask);
  } else if (query.scalar_type() == at::kFloat &&
             value.scalar_type() == at::kBFloat16) {

    return scale_dot_product_for_indirect_access_kv_cache<float, at::BFloat16>(
        query, key, value, key_cache, value_cache, beam_idx, offset, scale_attn,
        attention_mask);
  } else if (key.scalar_type() == at::kBFloat16 &&
             value.scalar_type() == at::kFloat) {

    return scale_dot_product_for_indirect_access_kv_cache<at::BFloat16, float>(
        query, key, value, key_cache, value_cache, beam_idx, offset, scale_attn,
        attention_mask);
  }
  return scale_dot_product_for_indirect_access_kv_cache_bf16(
      query, key, value, key_cache, value_cache, beam_idx, offset, scale_attn,
      attention_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
first_token_masked_mha(at::Tensor query, at::Tensor key, at::Tensor value,
                       at::Tensor &key_cache, at::Tensor &value_cache,
                       at::Tensor &beam_idx, const int64_t beam_batch,
                       const double scale_attn, at::Tensor attention_mask,
                       bool add_casual_mask = true) {
  auto query_length = query.size(1);
  auto key_lenght = key.size(1);

  if (add_casual_mask) {
    auto casual_mask =
        at::full({query_length, key_lenght}, -1e6, query.options());
    casual_mask = at::triu(casual_mask, 1);
    casual_mask = casual_mask.unsqueeze(0).unsqueeze(0);
    attention_mask = attention_mask + casual_mask;
  }
  if (key.scalar_type() != at::kBFloat16 && key.scalar_type() != at::kFloat) {
    TORCH_CHECK(
        false,
        "key and value must be float or bfloat16 to use "
        "zentorch::zentorch_masked_multihead_self_attention_kernel_impl");
  }
  if (key.scalar_type() == at::kFloat) {
    copy_key_value<float>(key_cache, key, value_cache, value, beam_batch);
  } else {
    copy_key_value<at::BFloat16>(key_cache, key, value_cache, value,
                                 beam_batch);
  }
  // support MGQ/MQA
  // expand the head dimensiopn of key/value to be same to the query
  if (query.size(2) != key.size(2)) {
    auto n_req = query.size(2) / key.size(2);
    key = key.repeat_interleave(n_req, 2);
    value = value.repeat_interleave(n_req, 2);
  }
  auto attn_outputs = at::Tensor();
  auto attn_weights = at::Tensor();
  if ((key.scalar_type() == at::kFloat || key.scalar_type() == at::kBFloat16) &&
      attention_mask.stride(-1) == 1) {
    query = query.transpose(1, 2);
    key = key.transpose(1, 2);
    value = value.transpose(1, 2);
    attn_outputs = at::native::scaled_dot_product_attention(
        query, key, value, attention_mask,
        /* dropout */ 0.0,
        /* is_causal*/ false, 1. / scale_attn);

    // attn_outputs = std::get<0>(zentorch::cpu::flash_attention_kernel_stub(
    //     kCPU,
    //     query,
    //     key,
    //     value,
    //     /* dropout */ 0.0,
    //     /* is_causal*/ false,
    //     attention_mask,
    //     1. / scale_attn));
  } else {
    key = key.permute({0, 2, 1, 3});
    query = query.permute({0, 2, 1, 3});
    value = value.permute({0, 2, 1, 3});
    attn_weights = query.matmul(key.transpose(-1, -2));
    attn_weights = attn_weights.div(scale_attn);
    attn_weights = attn_weights + attention_mask;
    attn_weights = attn_weights.softmax(-1);
    attn_weights = attn_weights.to(value.dtype());
    attn_outputs = attn_weights.matmul(value);
  }
  return std::make_tuple(attn_outputs, attn_weights, key_cache, value_cache,
                         beam_idx);
}
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
masked_multihead_self_attention_kernel_impl_512(
    at::Tensor &query, at::Tensor &key, at::Tensor &value,
    at::Tensor &key_cache, at::Tensor &value_cache, at::Tensor &beam_idx,
    at::Tensor seq_info, const double scale_attn, int64_t max_positions,
    const c10::optional<at::Tensor> &head_mask /* optional */,
    const c10::optional<at::Tensor> &attention_mask /* optional */,
    c10::optional<bool> add_casual_mask /* optional */) {
  TORCH_CHECK(attention_mask.has_value(),
              "Attention mask is necessary for "
              "zentorch::masked_multihead_self_attention_kernel_impl_512");
  TORCH_CHECK(attention_mask.value().dim() == 4,
              "Attention mask must be 4D for "
              "zentorch::masked_multihead_self_attention_kernel_impl_512");
  TORCH_CHECK(head_mask.has_value() != true,
              "Head mask is not supported in "
              "zentorch::masked_multihead_self_attention_kernel_impl_512");
  TORCH_CHECK(query.dtype() == key.dtype(),
              "query and key must have the same data type to use "
              "zentorch::masked_multihead_self_attention_kernel_impl_512");

  query = query.contiguous();
  key = key.contiguous();
  value = value.contiguous();
  auto attention_mask_v = attention_mask.value().contiguous();
  attention_mask_v = attention_mask_v.to(query.dtype());
  auto beam_batch = beam_idx.size(1); // need to prepare the fake beam_idx as
  // (max_position, bs) for the first token
  auto offset = seq_info.data_ptr<long>()[0];
  auto cache_size = key_cache.size(0);
  auto cur_len = query.size(1);
  if (offset == 0) {
    max_positions =
        max_positions > cur_len ? max_positions : max_positions + cur_len;
    key_cache = at::empty({max_positions, beam_batch, key.size(2), key.size(3)},
                          key.options());
    value_cache =
        at::empty({max_positions, beam_batch, value.size(2), value.size(3)},
                  value.options());
    beam_idx = at::zeros({max_positions + 2, beam_batch}, beam_idx.options());
    auto beam_idx_access = beam_idx.accessor<long, 2>();
#pragma omp parallel for collapse(2)
    for (auto i = 0; i < max_positions; i++) {
      for (auto j = 0; j < beam_batch; j++) {
        if (key.size(0) == beam_batch) {
          beam_idx_access[i][j] = j;
        } else {
          auto beam_size = beam_batch / key.size(0);
          beam_idx_access[i][j] = j / beam_size * beam_size;
        }
      }
    }
    beam_idx_access[max_positions][0] = cur_len; // record the prompt token len
    beam_idx_access[max_positions + 1][0] =
        query.size(0); // record the promt bs info
  } else if (offset > 0 && offset + cur_len > cache_size) {
    auto new_cache_size = cache_size * 2;
    auto new_key_cache = at::empty(
        {new_cache_size, beam_batch, key.size(2), key.size(3)}, key.options());
    auto new_value_cache =
        at::empty({new_cache_size, beam_batch, value.size(2), value.size(3)},
                  value.options());
    auto new_beam_idx =
        at::zeros({new_cache_size + 2, beam_batch}, beam_idx.options());
    new_key_cache.slice(0, 0, cache_size).copy_(key_cache);
    new_value_cache.slice(0, 0, cache_size).copy_(value_cache);
    new_beam_idx.slice(0, 0, cache_size + 2).copy_(beam_idx);
    auto new_beam_idx_access = new_beam_idx.accessor<long, 2>();
    auto beam_idx_access = beam_idx.accessor<long, 2>();
    for (auto i = offset; i < new_cache_size; i++) {
      for (auto j = 0; j < beam_batch; j++) {
        new_beam_idx_access[i][j] = beam_idx_access[0][j];
      }
    }
    new_beam_idx_access[new_cache_size][0] = beam_idx_access[cache_size - 2][0];
    new_beam_idx_access[new_cache_size + 1][0] =
        beam_idx_access[cache_size - 1][0];
    key_cache = new_key_cache;
    value_cache = new_value_cache;
    beam_idx = new_beam_idx;
  }
  if (offset != 0) {
    auto cur_len = query.size(1);
    if (cur_len == 1) {
      return zero_copy_kv_cache_masked_multihead_self_attention_kernel_impl(
          query, key, value, key_cache, value_cache, beam_idx, offset,
          scale_attn, attention_mask_v);
    }
    // just a  funcationality path,need to optimize
    auto tokens_outs = std::vector<at::Tensor>(cur_len);
    for (auto i = 0; i < cur_len; i++) {
      auto query_i = query.select(1, i).unsqueeze(1);
      auto key_i = key.select(1, i).unsqueeze(1);
      auto value_i = value.select(1, i).unsqueeze(1);
      auto next_outs =
          zero_copy_kv_cache_masked_multihead_self_attention_kernel_impl(
              query_i, key_i, value_i, key_cache, value_cache, beam_idx,
              offset + i, scale_attn, attention_mask_v);
      tokens_outs[i] = std::get<0>(next_outs);
    }
    auto attn_outs = at::cat(tokens_outs, 2);
    return std::make_tuple(attn_outs, at::Tensor(), key_cache, value_cache,
                           beam_idx);
  } else {
    return first_token_masked_mha(
        query, key, value, key_cache, value_cache, beam_idx, beam_batch,
        scale_attn, attention_mask_v, add_casual_mask.value_or(true));
  }
}
} // namespace zentorch
