/******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Common utilities for PagedAttention kernels, shared between
 * PagedAttention.cpp and zen_PagedAttentionKernel.cpp to avoid code
 * duplication.
 ******************************************************************************/
#pragma once

#include <ATen/cpu/vec/vec.h>
#include <c10/core/ScalarType.h>

#include "MatmulUtils.hpp"

#include <string_view>
#include <type_traits>
#include <vector>

namespace zentorch {

// Reduced floating point type check (for BFloat16, Half)
template <typename T>
inline constexpr bool is_reduced_floating_point_v =
    std::is_same_v<T, at::BFloat16> || std::is_same_v<T, at::Half>;

// Conditional data pointer helper for reduced precision types
template <
    typename scalar_t,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
static inline scalar_t *conditional_data_ptr(float *ptr, scalar_t *ptr2) {
  return ptr2;
}

template <
    typename scalar_t,
    typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
static inline scalar_t *conditional_data_ptr(float *ptr, scalar_t *ptr2) {
  (void)ptr2;
  return reinterpret_cast<scalar_t *>(ptr);
}

// Vectorized fill operation
template <typename scalar_t>
inline void fill_stub(scalar_t *data, scalar_t val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t idx = 0;
  for (; idx <= size - Vec::size(); idx += Vec::size()) {
    data_vec.store(data + idx);
  }
  for (; idx < size; idx++) {
    data[idx] = val;
  }
}

// GEMM wrapper using LOA (Low-Overhead API) matmul_direct
template <typename T>
inline void zendnn_gemm(int64_t m, int64_t n, int64_t k, float alpha,
                        const T *a, int64_t lda, const T *b, int64_t ldb,
                        float beta, float *c, int64_t ldc, bool transA,
                        bool transB) {
  constexpr bool is_input_float = std::is_same_v<T, float>;

  zendnnl::lowoha::matmul::matmul_params params;
  zendnnl::lowoha::matmul::matmul_data_types matmul_dtype;
  matmul_dtype.bias = data_type_t::none;
  matmul_dtype.compute = data_type_t::none;
  matmul_dtype.src = is_input_float ? data_type_t::f32 : data_type_t::bf16;
  matmul_dtype.wei = is_input_float ? data_type_t::f32 : data_type_t::bf16;
  matmul_dtype.dst = data_type_t::f32;
  params.dtypes = matmul_dtype;
  params.lowoha_algo = zendnnl::ops::matmul_algo_t::libxsmm;
  zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;
  batch_params.Batch_A = 1;
  batch_params.Batch_B = 1;
  zendnnl::lowoha::matmul::matmul_direct(
      'r', /* layout: row-major */ transA, transB, m, n, k, alpha, a, lda, b,
      ldb, nullptr, /* No bias */ beta, c, ldc, false /* is_weights_const */,
      batch_params, params);
}

// Softcap kernel for attention scores
inline void softcap_kernel(float *dst, float *src, int64_t size, float softcap,
                           float scale) {
  using Vec = at::vec::Vectorized<float>;
  auto vec_size = Vec::size();
  Vec div = Vec(scale / softcap);
  Vec mul = Vec(softcap);
  int64_t idx = 0;
  for (; idx <= size - vec_size; idx += vec_size) {
    auto score = Vec::loadu(src + idx);
    score = score * div;
    score = score.tanh();
    score = score * mul;
    score.store(dst + idx);
  }
  if (idx < size) {
    auto score = Vec::loadu(src + idx, size - idx);
    score = score * div;
    score = score.tanh();
    score = score * mul;
    score.store(dst + idx, size - idx);
  }
}

} // namespace zentorch
