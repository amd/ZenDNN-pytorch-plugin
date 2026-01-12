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

#include "EnvReader.hpp"
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

// Cached environment variables for GEMM - read once at first call
struct ZenDNNGemmConfig {
  static int get_matmul_direct() {
    static const int val =
        EnvReader::getEnvVariableAsInt("USE_ZENDNN_SDPA_MATMUL_DIRECT");
    return val;
  }
};

// Helper to set zendnnl tensor attributes with aligned sizes
inline void set_zendnnl_tensor_attributes_wrapper(
    const void *at_tensor_ptr, tensor_t &zendnnl_tensor,
    const std::string_view &tensor_name,
    const std::vector<unsigned long> &tensor_sizes,
    const std::vector<unsigned long> &tensor_strides, const bool is_input_float,
    const bool is_transposed) {

  const data_type_t zendnnl_dtype =
      is_input_float ? data_type_t::f32 : data_type_t::bf16;
  int64_t nbytes = is_input_float ? c10::elementSize(c10::kFloat)
                                  : c10::elementSize(c10::kBFloat16);

  std::vector<unsigned long> tensor_aligned_sizes(2);
  if (is_transposed) {
    tensor_aligned_sizes = {tensor_strides[1], tensor_sizes[1]};
    nbytes *= tensor_sizes[1] * tensor_strides[1];
  } else {
    tensor_aligned_sizes = {tensor_sizes[0], tensor_strides[0]};
    nbytes *= tensor_sizes[0] * tensor_strides[0];
  }

  set_zendnnl_tensor_attributes(const_cast<void *>(at_tensor_ptr),
                                zendnnl_dtype, zendnnl_tensor, tensor_name,
                                false /* is_weight_prepacked */, tensor_sizes,
                                tensor_strides, tensor_aligned_sizes, nbytes);
}

// Unified GEMM wrapper for both matmul_direct and tensor-based paths
// Handles beta accumulation correctly for both paths
template <typename T>
inline void zendnn_gemm(int64_t m, int64_t n, int64_t k, float alpha,
                        const T *a, int64_t lda, const T *b, int64_t ldb,
                        float beta, float *c, int64_t ldc, bool transA,
                        bool transB) {
  constexpr bool is_input_float = std::is_same_v<T, float>;
  const int zendnn_matmul_direct_env_value =
      ZenDNNGemmConfig::get_matmul_direct();

  // For beta != 0, save original C values for manual accumulation (tensor path)
  std::vector<float> c_orig;
  if (beta != 0.0f && !zendnn_matmul_direct_env_value) {
    c_orig.resize(m * n);
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        c_orig[i * n + j] = c[i * ldc + j];
      }
    }
  }

  if (zendnn_matmul_direct_env_value) {
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
  } else {
    tensor_t mat1_tensor = tensor_t();
    const std::vector<unsigned long> sizes_a = {static_cast<unsigned long>(m),
                                                static_cast<unsigned long>(k)};
    const std::vector<unsigned long> strides_a =
        transA ? std::vector<unsigned long>{1, static_cast<unsigned long>(lda)}
               : std::vector<unsigned long>{static_cast<unsigned long>(lda), 1};
    set_zendnnl_tensor_attributes_wrapper(a, mat1_tensor, "matmul_input",
                                          sizes_a, strides_a, is_input_float,
                                          transA);

    tensor_t mat2_tensor = tensor_t();
    const std::vector<unsigned long> sizes_b = {static_cast<unsigned long>(k),
                                                static_cast<unsigned long>(n)};
    const std::vector<unsigned long> strides_b =
        transB ? std::vector<unsigned long>{1, static_cast<unsigned long>(ldb)}
               : std::vector<unsigned long>{static_cast<unsigned long>(ldb), 1};
    set_zendnnl_tensor_attributes_wrapper(b, mat2_tensor, "weights", sizes_b,
                                          strides_b, is_input_float, transB);

    tensor_t result = tensor_t();
    const std::vector<unsigned long> sizes_c = {static_cast<unsigned long>(m),
                                                static_cast<unsigned long>(n)};
    const std::vector<unsigned long> strides_c = {
        static_cast<unsigned long>(ldc), 1};
    set_zendnnl_tensor_attributes_wrapper(
        c, result, "matmul_output", sizes_c, strides_c,
        /* is_input_float */ true, /* is_transposed */ false);

    auto matmul_context = matmul_context_t();
    set_matmul_context_attributes(matmul_context, mat2_tensor,
                                  {} /* no post ops */, alpha);

    auto matmul_operator = matmul_operator_t();
    set_matmul_operator_attributes(matmul_operator, matmul_context, mat1_tensor,
                                   result, {} /* no post ops */,
                                   {} /* no post op buffers */);

    status_t status = matmul_operator.execute();

    ZENTORCH_CHECK(status == status_t::success, "operator ",
                   matmul_operator.get_name(),
                   " execution failed for zentorch_matmul_impl.");

    // Manual beta accumulation for tensor path: C = alpha * A @ B + beta *
    // C_original
    if (beta != 0.0f) {
      for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          c[i * ldc + j] += beta * c_orig[i * n + j];
        }
      }
    }
  }
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
