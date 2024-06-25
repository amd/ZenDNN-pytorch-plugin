/******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Was sourced from
 * https://github.com/intel/intel-extension-for-pytorch/blob/v2.3.0%2Bcpu/csrc/cpu/vec/vec512/vec512_bfloat16.h
 * IPEX commit ID: d3c52443e
 ******************************************************************************/

#pragma once
#include "utils.h"
#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec512/vec512.h>

using namespace at::vec;

#include <immintrin.h>
// Conversion from BF16 to FP32
inline __m512 cvt_bf16_to_fp32(const __m256i src) {
  auto y = _mm512_cvtepu16_epi32(src);
  return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

inline void cvt_bf16_to_fp32(float *dst, const at::BFloat16 *src, int len) {
  int i = 0;
  for (; i < len - 15; i += 16) {
    auto f32 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(src + i)));
    _mm512_storeu_ps(dst + i, f32);
  }
  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto f32 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, src + i));
    _mm512_mask_storeu_ps(dst + i, mask, f32);
  }
}

// Conversion from FP32 to BF16
inline __m256i trunc_fp32_to_bf16(const __m512 src) {
  auto y = _mm512_bsrli_epi128(_mm512_castps_si512(src), 2);
  return _mm512_cvtepi32_epi16(y);
}

inline __m256i cvt_fp32_to_bf16(const __m512 src) {
#if (defined CPU_CAPABILITY_AVX512_BF16)
  return reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(src));
#else
  __m512i value = _mm512_castps_si512(src);
  __m512i nan = _mm512_set1_epi32(0xffff);
  auto mask_value = _mm512_cmp_ps_mask(src, src, _CMP_ORD_Q);
  __m512i ones = _mm512_set1_epi32(0x1);
  __m512i vec_bias = _mm512_set1_epi32(0x7fff);
  // uint32_t lsb = (input >> 16) & 1;
  auto t_value = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
  // uint32_t rounding_bias = 0x7fff + lsb;
  t_value = _mm512_add_epi32(t_value, vec_bias);
  // input += rounding_bias;
  t_value = _mm512_add_epi32(t_value, value);
  // input = input >> 16;
  t_value = _mm512_srli_epi32(t_value, 16);
  // Check NaN before converting back to bf16
  t_value = _mm512_mask_blend_epi32(mask_value, nan, t_value);
  return _mm512_cvtusepi32_epi16(t_value);
#endif
}

inline void cvt_fp32_to_bf16(at::BFloat16 *dst, const float *src, int len) {
  int i = 0;
  for (; i < len - 15; i += 16) {
    auto f32 = _mm512_loadu_ps(src + i);
    _mm256_storeu_si256((__m256i *)(dst + i), cvt_fp32_to_bf16(f32));
  }
  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto f32 = _mm512_maskz_loadu_ps(mask, src + i);
    _mm256_mask_storeu_epi16(dst + i, mask, cvt_fp32_to_bf16(f32));
  }
}

#include <immintrin.h>

namespace zentorch {
// namespace cpu {
// namespace kernel {

template <>
ZENTORCH_FORCE_INLINE void add_ker(at::BFloat16 *inout, const at::BFloat16 *in,
                                   int64_t len) {
  int64_t i = 0;
#pragma unroll(2)
  for (i = 0; i < len - 31; i += 32) {
    auto inout1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(inout + i)));
    auto inout2 =
        cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(inout + i + 16)));
    auto in1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(in + i)));
    auto in2 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(in + i + 16)));
    inout1 = _mm512_add_ps(inout1, in1);
    inout2 = _mm512_add_ps(inout2, in2);
    _mm256_storeu_si256((__m256i *)(inout + i), cvt_fp32_to_bf16(inout1));
    _mm256_storeu_si256((__m256i *)(inout + i + 16), cvt_fp32_to_bf16(inout2));
  }

  if (i < len - 15) {
    auto inout1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(inout + i)));
    auto in1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(in + i)));
    inout1 = _mm512_add_ps(inout1, in1);
    _mm256_storeu_si256((__m256i *)(inout + i), cvt_fp32_to_bf16(inout1));
    i += 16;
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto inout1 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, inout + i));
    auto in1 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, in + i));
    inout1 = _mm512_add_ps(inout1, in1);
    _mm256_mask_storeu_epi16(inout + i, mask, cvt_fp32_to_bf16(inout1));
  }
}

template <>
ZENTORCH_FORCE_INLINE void add_ker(float *inout, const float *in, int64_t len) {
  int64_t i = 0;
#pragma unroll(2)
  for (i = 0; i < len - 31; i += 32) {
    auto out1 = _mm512_loadu_ps(inout + i);
    auto out2 = _mm512_loadu_ps(inout + i + 16);
    auto in1 = _mm512_loadu_ps(in + i);
    auto in2 = _mm512_loadu_ps(in + i + 16);
    out1 = _mm512_add_ps(out1, in1);
    out2 = _mm512_add_ps(out2, in2);
    _mm512_storeu_ps(inout + i, out1);
    _mm512_storeu_ps(inout + i + 16, out2);
  }

  if (i < len - 15) {
    auto out1 = _mm512_loadu_ps(inout + i);
    auto in1 = _mm512_loadu_ps(in + i);
    _mm512_storeu_ps(inout + i, _mm512_add_ps(out1, in1));
    i += 16;
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto out1 = _mm512_maskz_loadu_ps(mask, inout + i);
    auto in1 = _mm512_maskz_loadu_ps(mask, in + i);
    _mm512_mask_storeu_ps(inout + i, mask, _mm512_add_ps(out1, in1));
  }
}

template <>
ZENTORCH_FORCE_INLINE void add_ker(float *inout, const at::BFloat16 *in,
                                   int64_t len) {
  int64_t i = 0;
#pragma unroll(2)
  for (i = 0; i < len - 31; i += 32) {
    auto in1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(in + i)));
    auto in2 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(in + i + 16)));
    auto inout1 = _mm512_loadu_ps(inout + i);
    auto inout2 = _mm512_loadu_ps(inout + i + 16);
    inout1 = _mm512_add_ps(inout1, in1);
    inout2 = _mm512_add_ps(inout2, in2);
    _mm512_storeu_ps(inout + i, inout1);
    _mm512_storeu_ps(inout + i + 16, inout2);
  }

  if (i < len - 15) {
    auto in1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(in + i)));
    auto inout1 = _mm512_loadu_ps(inout + i);
    inout1 = _mm512_add_ps(inout1, in1);
    _mm512_storeu_ps(inout + i, inout1);
    i += 16;
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto in1 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, in + i));
    auto inout1 = _mm512_maskz_loadu_ps(mask, inout + i);
    inout1 = _mm512_add_ps(inout1, in1);
    _mm512_mask_storeu_ps(inout + i, mask, inout1);
  }
}

template <>
ZENTORCH_FORCE_INLINE void add_ker(at::BFloat16 *inout, const float *in,
                                   int64_t len) {
  int64_t i = 0;
#pragma unroll(2)
  for (i = 0; i < len - 31; i += 32) {
    auto in1 = _mm512_loadu_ps(in + i);
    auto in2 = _mm512_loadu_ps(in + i + 16);
    auto inout1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(inout + i)));
    auto inout2 =
        cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(inout + i + 16)));
    inout1 = _mm512_add_ps(inout1, in1);
    inout2 = _mm512_add_ps(inout2, in2);
    _mm256_storeu_si256((__m256i *)(inout + i), cvt_fp32_to_bf16(inout1));
    _mm256_storeu_si256((__m256i *)(inout + i + 16), cvt_fp32_to_bf16(inout2));
  }

  if (i < len - 15) {
    auto in1 = _mm512_loadu_ps(in + i);
    auto inout1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(inout + i)));
    inout1 = _mm512_add_ps(inout1, in1);
    _mm256_storeu_si256((__m256i *)(inout + i), cvt_fp32_to_bf16(inout1));
    i += 16;
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto in1 = _mm512_maskz_loadu_ps(mask, in + i);
    auto inout1 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, inout + i));
    inout1 = _mm512_add_ps(inout1, in1);
    _mm256_mask_storeu_epi16(inout + i, mask, cvt_fp32_to_bf16(inout1));
  }
}
template <>
ZENTORCH_FORCE_INLINE void move_ker(at::BFloat16 *out, const float *in,
                                    int64_t len) {
  cvt_fp32_to_bf16(out, in, len);
}

template <>
ZENTORCH_FORCE_INLINE void move_ker(float *out, const float *in, int64_t len) {
  int64_t i = 0;
#pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
    auto in0 = _mm512_loadu_ps(in + i);
    _mm512_storeu_ps(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_ps(mask, in + i);
    _mm512_mask_storeu_ps(out + i, mask, in0);
  }
}

template <>
ZENTORCH_FORCE_INLINE void move_ker(at::BFloat16 *out, const at::BFloat16 *in,
                                    int64_t len) {
  int64_t i = 0;
#pragma unroll(4)
  for (i = 0; i < len - 31; i += 32) {
    auto in0 = _mm512_loadu_si512(in + i);
    _mm512_storeu_si512(out + i, in0);
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto in0 = _mm512_maskz_loadu_epi16(mask, in + i);
    _mm512_mask_storeu_epi16(out + i, mask, in0);
  }
}

template <>
ZENTORCH_FORCE_INLINE void move_ker(int32_t *out, const int32_t *in,
                                    int64_t len) {
  int64_t i = 0;
#pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
    auto in0 = _mm512_loadu_ps(in + i);
    _mm512_storeu_ps(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_ps(mask, in + i);
    _mm512_mask_storeu_ps(out + i, mask, in0);
  }
}

static ZENTORCH_FORCE_INLINE void zero_ker(float *out, int64_t len) {
  int64_t i = 0;
  __m512 zero_512 = _mm512_setzero_ps();
#pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
    _mm512_storeu_ps(out + i, zero_512);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    _mm512_mask_storeu_ps(out + i, mask, zero_512);
  }
}

static ZENTORCH_FORCE_INLINE void zero_ker(at::BFloat16 *out, int64_t len) {
  int64_t i = 0;
  __m512i zero_512 = _mm512_setzero_si512();
#pragma unroll(4)
  for (i = 0; i < len - 31; i += 32) {
    _mm512_storeu_si512(out + i, zero_512);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    _mm512_mask_storeu_epi16(out + i, mask, zero_512);
  }
}

inline __m512 convert_bf16_to_fp32(const __m256i src) {
  __m512i y = _mm512_cvtepu16_epi32(src);
  return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

template <typename T> inline float toFloat(T val) {
  float ret = float(val);
  return ret;
}

template <typename T1, typename T2>
inline void madd_ker(T1 *inout, T2 *in, int len, float alpha) {
#pragma omp simd
  for (long v = 0; v < len; v++) {
    inout[v] += toFloat(in[v]) * alpha;
  }
}

template <>
inline void madd_ker(float *inout, at::BFloat16 *in, int len, float alpha) {
  __m512 vAlpha = _mm512_set1_ps(alpha);
  int i = 0;
  for (; i < len - 15; i += 16) {
    __m512 y1 = _mm512_loadu_ps(inout + i);
    __m512 y2 = convert_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(in + i)));
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm512_storeu_ps(inout + i, y1);
  }
  if (i < len) {
    int rem = len - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 y1 = _mm512_maskz_loadu_ps(mask, inout + i);
    __m512 y2 = convert_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, in + i));
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm512_mask_storeu_ps(inout + i, mask, y1);
  }
}

// } // namespace kernel
// } // namespace cpu
} // namespace zentorch

// Conversion from BF16/FP16 to FP32
template <typename T,
          typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline __m512 cvt_to_fp32(const __m256i src);

template <> inline __m512 cvt_to_fp32<at::BFloat16>(const __m256i src) {
  return cvt_bf16_to_fp32(src);
}

// Conversion from FP32 to BF16/FP16
template <typename T,
          typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline __m256i cvt_from_fp32(const __m512 src);

template <> inline __m256i cvt_from_fp32<at::BFloat16>(const __m512 src) {
  return cvt_fp32_to_bf16(src);
}
