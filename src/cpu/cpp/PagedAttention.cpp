/******************************************************************************
  * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
  * All rights reserved.

  * Was sourced from
  * https://github.com/intel/intel-extension-for-pytorch/blob/v2.5.0%2Bcpu/csrc/cpu/aten/kernels/PagedAttentionKrnl.cpp
  * IPEX commit ID: 6973d57
******************************************************************************/
#include <torch/torch.h>
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR > 3
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/Tensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

#include <ATen/ops/empty.h>

#include "EnvReader.hpp"
#include "Memory.hpp"
#include "kernels/vec/utils.h"
#include "kernels/zen_cpukernels.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <omp.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "MatmulUtils.hpp"
#include "PagedAttentionCommon.hpp"
#include "Utils.hpp"

namespace zentorch {

namespace { // Anonymous namespace for file-local helpers

inline float compute_softmax_scale(const at::Tensor &query,
                                   c10::optional<double> scale) {
  if (scale.has_value()) {
    return static_cast<float>(scale.value());
  }
  double inv_sqrt = 1.0 / std::sqrt(static_cast<double>(query.size(-1)));
  return static_cast<float>(inv_sqrt);
}

} // anonymous namespace

template <typename scalar_t, int64_t q_split_size>
void flash_attn_varlen_kernel_ref(
    const at::Tensor &out, const at::Tensor &query, const at::Tensor &key_cache,
    const at::Tensor &value_cache, const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_k, int64_t max_seqlen_q, int64_t max_seqlens_k,
    const double softmax_scale, bool is_causal, const at::Tensor &block_table,
    const c10::optional<at::Tensor> &alibi_slopes, int64_t window_size_left,
    int64_t window_size_right, const double k_scale, const double v_scale,
    const double softcap) {
  TORCH_CHECK(
      key_cache.scalar_type() == value_cache.scalar_type(),
      "key_cache and value_cache should have the same data type for varlen");
  TORCH_CHECK(
      query.scalar_type() == out.scalar_type(),
      "query and out should have the same data type for varlen attention");
  TORCH_CHECK(!alibi_slopes.has_value(),
              "alibi_slopes is not supported for flash_attn_varlen yet");
  (void)k_scale;
  (void)v_scale;

  // Data pointers

  auto out_ptr = out.data_ptr<scalar_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_ptr = key_cache.data_ptr<scalar_t>();
  auto value_ptr = value_cache.data_ptr<scalar_t>();
  auto cu_seqlens_q_ptr = cu_seqlens_q.data_ptr<int>();
  auto cu_seqlens_k_ptr = cu_seqlens_k.data_ptr<int>();
  auto block_table_ptr = block_table.data_ptr<int>();

  int64_t batch_size = cu_seqlens_q.size(0) - 1;
  int64_t num_heads = query.size(1);
  int64_t head_size = query.size(2);
  int64_t num_kv_heads = key_cache.size(1);
  TORCH_CHECK(num_heads % num_kv_heads == 0,
              "num_heads must be divisible by num_kv_heads");
  int64_t kv_head_group_size = num_heads / num_kv_heads;
  int64_t block_size = key_cache.size(2);
  int64_t max_num_blocks_per_seq = block_table.size(1);

  // Strides

  auto kv_block_strideN = key_cache.stride(0);
  auto kv_block_strideH = key_cache.stride(1);
  auto kv_block_strideP = key_cache.stride(2);

  auto out_strideN = out.stride(0);
  auto out_strideH = out.stride(1);
  auto q_strideN = query.stride(0);
  auto q_strideH = query.stride(1);

  // Window size and local attention

  if (is_causal) {
    window_size_right = 0;
  }
  if (window_size_left >= max_seqlens_k) {
    window_size_left = -1;
  }
  if (window_size_right >= max_seqlens_k) {
    window_size_right = -1;
  }
  bool is_local = (window_size_left != -1) || (window_size_right != -1);

  // Scaling factor and softcap

  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor =
      static_cast<accum_t>(compute_softmax_scale(query, softmax_scale));
  accum_t scaling_factor_ =
      static_cast<accum_t>(softcap == -1.0 ? scaling_factor : 1.0);

  bool use_softcap = softcap != -1.0;
  const accum_t neg_inf = -std::numeric_limits<accum_t>::infinity();

  // Split size and capacity

  int64_t qSplitSize = std::min<int64_t>(q_split_size, max_seqlen_q);
  int64_t kvSplitSize = std::min<int64_t>(block_size, max_seqlens_k);
  int64_t kvCapacity = kvSplitSize;
  int64_t qSliceMax =
      (max_seqlen_q + qSplitSize - 1) / std::max<int64_t>(qSplitSize, 1);

  int64_t size_per_thread = qSplitSize * kvCapacity + qSplitSize + qSplitSize +
                            qSplitSize * head_size; // qk, max, sum, dst

  int64_t num_threads = at::get_num_threads();

  // Buffer allocation - use empty_strided_cpu for minimal dispatch overhead

  auto buf = at::detail::empty_strided_cpu(
      {num_threads, size_per_thread}, {size_per_thread, 1},
      query.options().dtype(at::toOpMathType(query.scalar_type())));

  auto buf_data = buf.data_ptr<accum_t>();
  constexpr bool is_reduced_type =
      at::vec::is_reduced_floating_point_v<scalar_t>;
  int64_t buf_reduced_dim2 = is_reduced_type ? kvCapacity : 0;
  at::Tensor buf_reduced = at::detail::empty_strided_cpu(
      {num_threads, qSplitSize, buf_reduced_dim2},
      {qSplitSize * buf_reduced_dim2, buf_reduced_dim2, 1}, query.options());
  scalar_t *buf_reduced_data =
      is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;

  // Use key_cache/value_cache options to match their dtypes
  at::Tensor key_block_buffer = at::detail::empty_strided_cpu(
      {num_threads, kvCapacity, head_size},
      {kvCapacity * head_size, head_size, 1}, key_cache.options());
  at::Tensor value_block_buffer = at::detail::empty_strided_cpu(
      {num_threads, kvCapacity, head_size},
      {kvCapacity * head_size, head_size, 1}, value_cache.options());
  scalar_t *key_block_data = key_block_buffer.data_ptr<scalar_t>();
  scalar_t *value_block_data = value_block_buffer.data_ptr<scalar_t>();

  // Limit OpenMP nesting ONCE before parallel region (not inside loop)
  omp_set_max_active_levels(1);

  // Attention Computation Loop

  at::parallel_for(
      0, batch_size * num_heads * qSliceMax, 1,
      [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0, k = 0;
        at::native::data_index_init(begin, i, batch_size, j, num_heads, k,
                                    qSliceMax);
        for (int64_t linear = begin; linear < end; linear++) {
          int omp_idx = at::get_thread_num();
          accum_t *buf_ptr = buf_data + omp_idx * size_per_thread;
          accum_t *qk_data = buf_ptr;
          accum_t *qk_max_data = qk_data + qSplitSize * kvCapacity;
          accum_t *qk_sum_data = qk_max_data + qSplitSize;
          accum_t *dst_data = qk_sum_data + qSplitSize;
          scalar_t *qk_reduced_data =
              is_reduced_type
                  ? buf_reduced_data + omp_idx * qSplitSize * kvCapacity
                  : nullptr;

          int64_t q_offset = cu_seqlens_q_ptr[i];
          int64_t qSize = cu_seqlens_q_ptr[i + 1] - q_offset;
          int64_t kvSize = cu_seqlens_k_ptr[i + 1] - cu_seqlens_k_ptr[i];
          int64_t context_len = kvSize - qSize;

          int64_t m = k * qSplitSize;
          if (m >= qSize) {
            at::native::data_index_step(i, batch_size, j, num_heads, k,
                                        qSliceMax);
            continue;
          }

          int64_t qBlockSize = std::min<int64_t>(qSplitSize, qSize - m);
          fill_stub(qk_max_data, -std::numeric_limits<accum_t>::infinity(),
                    qBlockSize);
          fill_stub(qk_sum_data, static_cast<accum_t>(0), qBlockSize);
          fill_stub(dst_data, static_cast<accum_t>(0), qBlockSize * head_size);

          int64_t num_keys =
              is_causal
                  ? std::min<int64_t>(m + qBlockSize + context_len, kvSize)
                  : kvSize;

          int64_t kv_head_id = j / kv_head_group_size;

          if (num_keys > 0) {
            scalar_t *key_buffer =
                key_block_data + omp_idx * kvCapacity * head_size;
            scalar_t *value_buffer =
                value_block_data + omp_idx * kvCapacity * head_size;

            bool has_previous_block = false;

            for (int64_t n = 0; n < num_keys;) {
              int64_t logical_block = n / block_size;
              int64_t block_offset = n % block_size;
              int64_t physical_block_id =
                  block_table_ptr[i * max_num_blocks_per_seq + logical_block];
              const scalar_t *key_page_data =
                  key_ptr + physical_block_id * kv_block_strideN +
                  kv_head_id * kv_block_strideH;
              const scalar_t *value_page_data =
                  value_ptr + physical_block_id * kv_block_strideN +
                  kv_head_id * kv_block_strideH;

              int64_t tokens_in_block = block_size - block_offset;
              int64_t tokens_remaining = num_keys - n;
              int64_t kvBlockSize =
                  std::min<int64_t>(kvSplitSize, tokens_in_block);
              kvBlockSize = std::min<int64_t>(kvBlockSize, tokens_remaining);

              if (kvBlockSize <= 0) {
                break;
              }

              if (window_size_left > 0 &&
                  m + context_len - window_size_left > n + kvBlockSize) {
                n += kvBlockSize;
                continue;
              }
              if (window_size_right >= 0 &&
                  m + context_len + qBlockSize + window_size_right + 1 <= n) {
                n += kvBlockSize;
                continue;
              }

              const scalar_t *key_src =
                  key_page_data + block_offset * kv_block_strideP;
              const scalar_t *value_src =
                  value_page_data + block_offset * kv_block_strideP;

              for (int64_t token_idx = 0; token_idx < kvBlockSize;
                   token_idx++) {
                std::memcpy(key_buffer + token_idx * head_size,
                            key_src + token_idx * kv_block_strideP,
                            head_size * sizeof(scalar_t));
                std::memcpy(value_buffer + token_idx * head_size,
                            value_src + token_idx * kv_block_strideP,
                            head_size * sizeof(scalar_t));
              }

              // Q @ K^T GEMM (omp nesting already limited before parallel_for)
              zendnn_gemm<scalar_t>(
                  qBlockSize, kvBlockSize, head_size, 1.0f,
                  query_ptr + (q_offset + m) * q_strideN + j * q_strideH,
                  q_strideN, key_buffer, head_size, 0.0f,
                  reinterpret_cast<float *>(qk_data), kvCapacity, false, true);

              if (use_softcap) {
                for (int64_t row = 0; row < qBlockSize; row++) {
                  softcap_kernel(
                      reinterpret_cast<float *>(qk_data + row * kvCapacity),
                      reinterpret_cast<float *>(qk_data + row * kvCapacity),
                      kvBlockSize, static_cast<float>(softcap), scaling_factor);
                }
              }

              if (is_local) {
                for (int64_t row = 0; row < qBlockSize; row++) {
                  int64_t idx = context_len + m + row;
                  for (int64_t col = 0; col < kvBlockSize; col++) {
                    int64_t global_col = n + col;
                    if (window_size_left > 0 &&
                        idx - window_size_left > global_col) {
                      qk_data[row * kvCapacity + col] = neg_inf;
                    }
                    if (window_size_right >= 0 &&
                        idx + window_size_right + 1 <= global_col) {
                      qk_data[row * kvCapacity + col] = neg_inf;
                    }
                  }
                }
              }

              for (int64_t row = 0; row < qBlockSize; row++) {
                accum_t *row_scores = qk_data + row * kvCapacity;
                accum_t local_max = neg_inf;
                for (int64_t col = 0; col < kvBlockSize; col++) {
                  row_scores[col] *= scaling_factor_;
                  local_max = std::max(local_max, row_scores[col]);
                }

                accum_t prev_max = qk_max_data[row];
                accum_t new_max = local_max > prev_max ? local_max : prev_max;
                accum_t exp_tmp = (prev_max == neg_inf || new_max == neg_inf)
                                      ? static_cast<accum_t>(0)
                                      : std::exp(prev_max - new_max);

                accum_t new_sum = 0;
                if (new_max == neg_inf) {
                  for (int64_t col = 0; col < kvBlockSize; col++) {
                    row_scores[col] = static_cast<accum_t>(0);
                    if (is_reduced_type) {
                      qk_reduced_data[row * kvCapacity + col] =
                          static_cast<scalar_t>(0);
                    }
                  }
                } else {
                  for (int64_t col = 0; col < kvBlockSize; col++) {
                    accum_t val = std::exp(row_scores[col] - new_max);
                    row_scores[col] = val;
                    if (is_reduced_type) {
                      qk_reduced_data[row * kvCapacity + col] =
                          static_cast<scalar_t>(val);
                    }
                    new_sum += val;
                  }
                }

                if (exp_tmp != static_cast<accum_t>(0)) {
                  at::vec::map<accum_t>(
                      [exp_tmp](Vec v) { return v * Vec(exp_tmp); },
                      dst_data + row * head_size, dst_data + row * head_size,
                      head_size);
                }

                qk_sum_data[row] = new_sum + exp_tmp * qk_sum_data[row];
                qk_max_data[row] = new_max;
              }

              const scalar_t *attn_ptr =
                  is_reduced_type ? qk_reduced_data
                                  : reinterpret_cast<const scalar_t *>(qk_data);

              float beta = has_previous_block ? 1.0f : 0.0f;
              // Attention @ V GEMM (omp nesting already limited before
              // parallel_for)
              zendnn_gemm<scalar_t>(
                  qBlockSize, head_size, kvBlockSize, 1.0f, attn_ptr,
                  kvCapacity, value_buffer, head_size, beta,
                  reinterpret_cast<float *>(dst_data), head_size, false, false);

              has_previous_block = true;

              n += kvBlockSize;
            }
          }

          for (int64_t row = 0; row < qBlockSize; row++) {
            if (qk_sum_data[row] <= static_cast<accum_t>(0)) {
              fill_stub(out_ptr + (q_offset + m + row) * out_strideN +
                            j * out_strideH,
                        static_cast<scalar_t>(0), head_size);
              continue;
            }
            accum_t sum_recip = 1.0f / (qk_sum_data[row] + 1e-8f);
            at::vec::map<scalar_t>(
                [sum_recip](Vec v) { return v * Vec(sum_recip); },
                out_ptr + (q_offset + m + row) * out_strideN + j * out_strideH,
                dst_data + row * head_size, head_size);
          }

          at::native::data_index_step(i, batch_size, j, num_heads, k,
                                      qSliceMax);
        }
      });
}

template <typename scalar_t>
void flash_attn_varlen_launch_ref(
    const at::Tensor &out, const at::Tensor &query, const at::Tensor &key_cache,
    const at::Tensor &value_cache, const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_k, int64_t max_seqlen_q, int64_t max_seqlens_k,
    const double softmax_scale, bool is_causal, const at::Tensor &block_table,
    const c10::optional<at::Tensor> &alibi_slopes, int64_t window_size_left,
    int64_t window_size_right, const double k_scale, const double v_scale,
    const double softcap) {
  if (max_seqlen_q >= 768) {
    flash_attn_varlen_kernel_ref<scalar_t, 256>(
        out, query, key_cache, value_cache, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlens_k, softmax_scale, is_causal, block_table,
        alibi_slopes, window_size_left, window_size_right, k_scale, v_scale,
        softcap);
  } else if (max_seqlen_q >= 192) {
    flash_attn_varlen_kernel_ref<scalar_t, 64>(
        out, query, key_cache, value_cache, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlens_k, softmax_scale, is_causal, block_table,
        alibi_slopes, window_size_left, window_size_right, k_scale, v_scale,
        softcap);
  } else {
    flash_attn_varlen_kernel_ref<scalar_t, 32>(
        out, query, key_cache, value_cache, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlens_k, softmax_scale, is_causal, block_table,
        alibi_slopes, window_size_left, window_size_right, k_scale, v_scale,
        softcap);
  }
}

void zentorch_attention_flash_attn_varlen_kernel_ref(
    const at::Tensor &out, const at::Tensor &query, const at::Tensor &key_cache,
    const at::Tensor &value_cache, const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_k, int64_t max_seqlen_q, int64_t max_seqlens_k,
    const double softmax_scale, bool is_causal, const at::Tensor &block_table,
    const c10::optional<at::Tensor> &alibi_slopes, int64_t window_size_left,
    int64_t window_size_right, const std::string_view &kv_cache_dtype,
    const double k_scale, const double v_scale, const double softcap,
    std::string zentorch_op_name) {
  (void)zentorch_op_name;
  TORCH_CHECK(kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e5m2" ||
                  kv_cache_dtype == "auto",
              "Unsupported kv_cache_dtype for flash_attn_varlen kernel");

  if (query.scalar_type() == at::ScalarType::Float) {
    flash_attn_varlen_launch_ref<float>(
        out, query, key_cache, value_cache, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlens_k, softmax_scale, is_causal, block_table,
        alibi_slopes, window_size_left, window_size_right, k_scale, v_scale,
        softcap);
  } else if (query.scalar_type() == at::ScalarType::BFloat16) {
    flash_attn_varlen_launch_ref<at::BFloat16>(
        out, query, key_cache, value_cache, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlens_k, softmax_scale, is_causal, block_table,
        alibi_slopes, window_size_left, window_size_right, k_scale, v_scale,
        softcap);
  } else {
    TORCH_CHECK(false,
                "zentorch flash_attn_varlen supports float and bfloat16");
  }
}

// Variable Length Flash Attention Kernel Implementation
//
// This function is used to dispatch the variable length flash attention kernel
// based on the instruction set
void zentorch_attention_flash_attn_varlen_kernel_impl(
    const at::Tensor &out, const at::Tensor &query, const at::Tensor &key_cache,
    const at::Tensor &value_cache, const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_k, int64_t max_seqlen_q, int64_t max_seqlens_k,
    const double softmax_scale, bool is_causal, const at::Tensor &block_table,
    const c10::optional<at::Tensor> &alibi_slopes, int64_t window_size_left,
    int64_t window_size_right, const std::string_view &kv_cache_dtype,
    const double k_scale, const double v_scale, const double softcap,
    std::string zentorch_op_name) {
  if (is_avx512_supported()) {
    zentorch_attention_flash_attn_varlen_kernel_512(
        out, query, key_cache, value_cache, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlens_k, softmax_scale, is_causal, block_table,
        alibi_slopes, window_size_left, window_size_right, kv_cache_dtype,
        k_scale, v_scale, softcap, zentorch_op_name);
  } else {
    zentorch_attention_flash_attn_varlen_kernel_ref(
        out, query, key_cache, value_cache, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlens_k, softmax_scale, is_causal, block_table,
        alibi_slopes, window_size_left, window_size_right, kv_cache_dtype,
        k_scale, v_scale, softcap, std::move(zentorch_op_name));
  }
}

template <typename QT, typename KT>
inline void reduce_head(const QT *q_ptr_start, int64_t kv_head_group_size,
                        const KT *k_cache_start, float *attn_w_pos,
                        int attn_w_stride, int64_t head_size) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    attn_w_pos[i * attn_w_stride] = 0;
    for (auto hsi = 0; hsi < head_size; hsi++) {
      attn_w_pos[i * attn_w_stride] +=
          (float)q_ptr_start[i * head_size + hsi] * (float)k_cache_start[hsi];
    }
  }
}

template <typename OT, typename CT>
inline void mul_attenion_weights_and_value_of_head(
    const float *attn_w, int attn_w_stride, const CT *v_cache_start,
    OT *attn_out_start, int attn_out_strideH, int kv_head_group_size,
    int64_t head_size, bool accumulated) {
  auto hsi = 0;
  for (auto i = 0; i < kv_head_group_size; i++) {
    for (hsi = 0; hsi < head_size; hsi++) {
      if (accumulated) {
        attn_out_start[i * attn_out_strideH + hsi] +=
            attn_w[i * attn_w_stride] * (float)v_cache_start[hsi];
      } else {
        attn_out_start[i * attn_out_strideH + hsi] =
            attn_w[i * attn_w_stride] * (float)v_cache_start[hsi];
      }
    }
  }
}

// 1) out = exp(a - val)
// 2) val = sum(out)
template <typename T1, typename T2>
inline void _exp_reduce_sum_fusion_kernel(T1 *a, const int &size, T2 *out,
                                          T1 &val) {
  auto vec_size = at::vec::Vectorized<T1>::size();
  auto vec_max = at::vec::Vectorized<T1>(val);
  T1 tmp_sum = 0;
  auto vec_tmp_sum = at::vec::Vectorized<T1>(tmp_sum);
  long i = 0;
  for (; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = tmp1.exp_u20();
    vec_tmp_sum += tmp2;
    at::native::_store(out + i, tmp2);
  }
  tmp_sum = at::vec::vec_reduce_all<T1>(
      [](at::vec::Vectorized<T1> &x, at::vec::Vectorized<T1> &y) {
        return x + y;
      },
      vec_tmp_sum);
  for (; i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}

// 1) out = a * scale + alibi_mask
// 2) max = max(out)
template <typename scalar_t>
inline void _mul_alibi_reduce_max_fusion_kernel(
    scalar_t *a, const scalar_t &scale, const int &size, scalar_t *out,
    scalar_t &max, const int &token_start, const int &context_len,
    const scalar_t &alibi_slope) {
  for (auto i = 0; i < size; i++) {
    a[i] = a[i] * scale;
    auto alibi_slopes_val = alibi_slope * (i + token_start + 1 - context_len);
    a[i] += alibi_slopes_val;
    max = std::max(max, a[i]);
  }
}

// 1) out = a * scale
// 2) max = max(out)
template <typename scalar_t>
inline void _mul_reduce_max_fusion_kernel(scalar_t *a, const scalar_t &scale,
                                          const int &size, scalar_t *out,
                                          scalar_t &max) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  auto vec_scale = at::vec::Vectorized<scalar_t>(scale);
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = at::vec::Vectorized<scalar_t>(tmp_max);
  long i = 0;
  for (; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    vec_tmp_max = at::vec::maximum(vec_tmp_max, tmp1);
    tmp1.store(out + i);
  }
  tmp_max = at::vec::vec_reduce_all<scalar_t>(
      [](at::vec::Vectorized<scalar_t> &x, at::vec::Vectorized<scalar_t> &y) {
        return at::vec::maximum(x, y);
      },
      vec_tmp_max);
  for (; i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    tmp_max = std::max(tmp_max, tmp1);
    out[i] = tmp1;
  }
  max = tmp_max;
}

/**
 * Performs scale-dot-product for the next token based on cached key-value
 * attention.
 *
 * This function computes the attention weights and applies the attention
 * mechanism to obtain the final output. It takes in tensors representing the
 * query, key cache, value cache, head mapping, scale, block tables, context
 * lengths, block size, max context length, and optional alibi slopes. The
 * output tensor is updated with the computed attention values.
 *
 * @param out           Output tensor [num_seqs, num_heads, head_size].
 * @param query         Query tensor [num_seqs, num_heads, head_size].
 * @param key_cache     The pre-allocated buffer to store the key cache. The
 * shape should be [num_blocks, block_size, num_heads, head_size].
 * @param value_cache   The pre-allocated buffer to store the value cache. The
 * shape should be [num_blocks, block_size, num_heads, head_size].
 * @param scale         Scaling factor for attention weights. In general, it is:
 * float(1.0 / (head_size ** 0.5)).
 * @param block_tables  Block tables tensor [num_seqs, max_num_blocks_per_seq].
 * @param context_lens  Context lengths tensor [num_seqs].
 * @param block_size    The block size which means the number of token in every
 * block.
 * @param max_context_len Maximum context length.
 * @param alibi_slopes  Optional tensor of alibi slopes with the shape of
 * (num_heads).
 */
template <typename scalar_t>
void single_query_cached_kv_attention_kernel_ref(
    const at::Tensor &out, const at::Tensor &query, const at::Tensor &key_cache,
    const at::Tensor &value_cache, const double scale,
    const at::Tensor &block_tables, const at::Tensor &context_lens,
    int64_t block_size, int64_t max_context_len,
    const c10::optional<at::Tensor> &alibi_slopes) {
  auto out_ptr = out.data_ptr<scalar_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_cache_ptr = key_cache.data_ptr<scalar_t>();
  auto value_cache_ptr = value_cache.data_ptr<scalar_t>();
  auto block_tables_ptr = block_tables.data_ptr<int>();
  auto context_lens_ptr = context_lens.data_ptr<int>();
  auto alibi_slopes_ptr = alibi_slopes.has_value()
                              ? alibi_slopes.value().data_ptr<float>()
                              : nullptr;
  auto num_seqs = query.size(0);
  auto num_heads = query.size(1);
  auto head_size = query.size(2);
  auto num_kv_heads = key_cache.size(1);
  ZENTORCH_CHECK(num_heads % num_kv_heads == 0,
                 "num_heads must be perfectly divisible by num_kv_heads");
  auto kv_head_group_size = num_heads / num_kv_heads;
  auto max_num_blocks_per_seq = block_tables.size(1);

  auto kv_block_strideN = key_cache.stride(0);
  auto kv_block_strideP = key_cache.stride(2);
  auto kv_block_strideH = key_cache.stride(1);

  auto out_strideN = out.stride(0);
  auto out_strideH = out.stride(1);

  auto q_strideN = query.stride(0);
  auto q_strideH = query.stride(1);

  auto partition_size = 128;

  auto max_num_partitions =
      (max_context_len + partition_size - 1) / partition_size;

  // Use empty_strided_cpu for minimal dispatch overhead
  int64_t max_part_plus1 = max_num_partitions + 1;
  auto max_logits = at::detail::empty_strided_cpu(
      {num_seqs, num_heads, max_part_plus1},
      {num_heads * max_part_plus1, max_part_plus1, 1},
      query.options().dtype(at::ScalarType::Float));

  auto exp_sum = at::detail::empty_strided_cpu(
      {num_seqs, num_heads, max_part_plus1},
      {num_heads * max_part_plus1, max_part_plus1, 1},
      query.options().dtype(at::ScalarType::Float));

  auto tmp_out = at::detail::empty_strided_cpu(
      {num_seqs, num_heads, max_num_partitions, head_size},
      {num_heads * max_num_partitions * head_size,
       max_num_partitions * head_size, head_size, 1},
      query.options().dtype(at::ScalarType::Float));

  auto tmp_out_ptr = tmp_out.data_ptr<float>();
  auto max_logits_ptr = max_logits.data_ptr<float>();
  auto exp_sum_ptr = exp_sum.data_ptr<float>();

  auto max_logits_strideN = max_logits.stride(0);
  auto max_logits_strideH = max_logits.stride(1);
  auto exp_sum_strideN = exp_sum.stride(0);
  auto exp_sum_strideH = exp_sum.stride(1);
  auto tmp_out_strideN = tmp_out.stride(0);
  auto tmp_out_strideH = tmp_out.stride(1);
  auto tmp_out_strideS = tmp_out.stride(2);

  if (alibi_slopes.has_value()) {
    auto alibi_slopes_size = alibi_slopes.value().size(0);
    ZENTORCH_CHECK(alibi_slopes_size == num_heads,
                   "alibi_slopes size is not equal to num_heads");
  }
#pragma omp parallel for collapse(3) schedule(static, 1)
  for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
    for (auto partition_id = 0; partition_id < max_num_partitions;
         partition_id++) {
      for (auto head_group_start = 0; head_group_start < num_heads;
           head_group_start += kv_head_group_size) {
        auto context_len = context_lens_ptr[seq_id];
        auto partition_start = partition_id * partition_size;
        if (partition_start >= context_len)
          continue;
        auto partition_end =
            std::min(partition_start + partition_size, context_len);
        auto token_num = partition_end - partition_start;
        auto block_num = (token_num + block_size - 1) / block_size;
        auto logical_block_start = partition_start / block_size;
        auto logical_block_end = logical_block_start + block_num;
        auto kv_head_id = head_group_start / kv_head_group_size;
        auto q_ptr_start =
            query_ptr + seq_id * q_strideN + head_group_start * q_strideH;
        auto max_logits_offset = seq_id * max_logits_strideN +
                                 head_group_start * max_logits_strideH +
                                 partition_id;
        auto exp_sum_offset = seq_id * exp_sum_strideN +
                              head_group_start * exp_sum_strideH + partition_id;
        //{num_seqs, num_heads, max_num_partitions, head_size}
        auto tmp_out_start = tmp_out_ptr + seq_id * tmp_out_strideN +
                             head_group_start * tmp_out_strideH +
                             partition_id * tmp_out_strideS;
        float logits[16 * partition_size] __attribute__((aligned(64))) = {0};
        auto logits_position = 0;
        // 1)calculate the matmul(query, key) for this partition
        for (auto logical_block_id = logical_block_start;
             logical_block_id < logical_block_end; logical_block_id++) {
          auto physical_block_id =
              block_tables_ptr[seq_id * max_num_blocks_per_seq +
                               logical_block_id];
          auto tokens_in_block =
              std::min(block_size, context_len - logical_block_id * block_size);
          auto token_start = logical_block_id * block_size;
          auto token_end = token_start + tokens_in_block;
          for (auto token_id = token_start; token_id < token_end; token_id++) {
            auto block_offset = token_id - token_start;
            auto k_cache_start =
                key_cache_ptr + physical_block_id * kv_block_strideN +
                block_offset * kv_block_strideP + kv_head_id * kv_block_strideH;
            reduce_head(q_ptr_start, kv_head_group_size, k_cache_start,
                        &(logits[logits_position]), partition_size, head_size);
            logits_position++;
          }
        }
        // 2) calculate the max and exp_sum for this partition
        for (int hi = 0; hi < kv_head_group_size; hi++) {
          auto partition_max = -std::numeric_limits<float>::infinity();
          if (alibi_slopes_ptr != nullptr) {
            _mul_alibi_reduce_max_fusion_kernel<float>(
                logits + hi * partition_size, scale, token_num,
                logits + hi * partition_size, partition_max, partition_start,
                context_len, alibi_slopes_ptr[head_group_start + hi]);
          } else {
            _mul_reduce_max_fusion_kernel<float>(
                logits + hi * partition_size, scale, token_num,
                logits + hi * partition_size, partition_max);
          }
          max_logits_ptr[max_logits_offset + hi * max_logits_strideH] =
              partition_max;
          _exp_reduce_sum_fusion_kernel<float, float>(
              logits + hi * partition_size, token_num,
              logits + hi * partition_size, partition_max);
          exp_sum_ptr[exp_sum_offset + hi * exp_sum_strideH] = partition_max;
        }

        // 3) calculate the matmul(exp(logits-partition_max), value) for this
        // partition, need to divide the global exp_sum in the final result.
        logits_position = 0;
        for (auto logical_block_id = logical_block_start;
             logical_block_id < logical_block_end; logical_block_id++) {
          auto physical_block_id =
              block_tables_ptr[seq_id * max_num_blocks_per_seq +
                               logical_block_id];
          auto tokens_in_block =
              std::min(block_size, context_len - logical_block_id * block_size);
          auto token_start = logical_block_id * block_size;
          auto token_end = token_start + tokens_in_block;
          for (auto token_id = token_start; token_id < token_end; token_id++) {
            auto block_offset = token_id - token_start;
            auto v_cache_start =
                value_cache_ptr + physical_block_id * kv_block_strideN +
                block_offset * kv_block_strideP + kv_head_id * kv_block_strideH;
            auto accumulated = logits_position > 0;
            mul_attenion_weights_and_value_of_head(
                &(logits[logits_position]), partition_size, v_cache_start,
                tmp_out_start, tmp_out_strideH, kv_head_group_size, head_size,
                accumulated);
            logits_position++;
          }
        }
      }
    }
  }

// calculate the final output
#pragma omp parallel for collapse(2)
  for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
    for (auto head_id = 0; head_id < num_heads; head_id++) {
      auto global_max = -std::numeric_limits<float>::infinity();
      auto global_exp_sum = 0.0;
      auto context_len = context_lens_ptr[seq_id];
      auto partition_num = (context_len + partition_size - 1) / partition_size;
      // calculate the global max and exp_sum for this head
      for (auto partition_id = 0; partition_id < max_num_partitions;
           partition_id++) {
        if (partition_id >= partition_num)
          break;
        auto max_logit =
            max_logits_ptr[seq_id * max_logits_strideN +
                           head_id * max_logits_strideH + partition_id];
        global_max = std::max(global_max, max_logit);
      }
      // update the partition 0 result with the global max
      auto partition0_out_start =
          tmp_out_ptr + seq_id * tmp_out_strideN + head_id * tmp_out_strideH;
      auto max_logit0 = max_logits_ptr[seq_id * max_logits_strideN +
                                       head_id * max_logits_strideH];
      float exp_val = expf(max_logit0 - global_max);
      global_exp_sum +=
          exp_sum_ptr[seq_id * exp_sum_strideN + head_id * exp_sum_strideH] *
          exp_val;
      at::vec::Vectorized<float> exp_val_vec0(exp_val);
      at::vec::map<float>([&](auto a) { return a * exp_val_vec0; },
                          partition0_out_start, partition0_out_start,
                          head_size);

      // accumulate the partition 1 to partition n result into partition 0
      if (partition_num > 1) {
        for (auto partition_id = 1; partition_id < partition_num;
             partition_id++) {
          if (partition_id * partition_size >= context_len)
            break;
          auto tmp_out_start = tmp_out_ptr + seq_id * tmp_out_strideN +
                               head_id * tmp_out_strideH +
                               partition_id * tmp_out_strideS;
          auto max_logit =
              max_logits_ptr[seq_id * max_logits_strideN +
                             head_id * max_logits_strideH + partition_id];
          auto exp_sum = exp_sum_ptr[seq_id * exp_sum_strideN +
                                     head_id * exp_sum_strideH + partition_id];
          exp_val = expf(max_logit - global_max);
          global_exp_sum += exp_sum * exp_val;
          at::vec::Vectorized<float> exp_val_vec(exp_val);
          at::vec::map2<float>(
              [&](auto a, auto b) { return a + exp_val_vec * b; },
              partition0_out_start, partition0_out_start, tmp_out_start,
              head_size);
        }
      }

      // copy the partition 0 result into attn_outs
      auto attn_out_start =
          out_ptr + seq_id * out_strideN + head_id * out_strideH;
      float inverse_global_sum = 1.0 / (global_exp_sum + 1e-8);
      at::vec::Vectorized<float> inverse_global_sum_vec(inverse_global_sum);
      // rescale the partition 0 result with global exp_sum
      at::vec::map<float>([&](auto a) { return a * inverse_global_sum_vec; },
                          partition0_out_start, partition0_out_start,
                          head_size);
      // copy the partition 0 result into attn_outs
      at::vec::map<scalar_t>([&](auto a) { return a; }, attn_out_start,
                             partition0_out_start, head_size);
    }
  }

} // single_query_cached_kv_attention_kernel_ref

/**
 * Reshapes and caches the key and value tensors based on the provided slot
 * mapping.
 *
 * @param key The input key tensor. The shape should be [num_seqs, num_heads,
 * head_size].
 * @param value The input value tensor.  The shape should be [num_seqs,
 * num_heads, head_size].
 * @param key_cache The output key cache tensor. The pre-allocated buffer to
 * store the key cache. The shape should be [num_blocks, block_size, num_heads,
 * head_size].
 * @param value_cache The output value cache tensor. The pre-allocated buffer to
 * store the value cache. The shape should be [num_blocks, block_size,
 * num_heads, head_size].
 * @param slot_mapping The slot mapping tensor. It stores the position to store
 * the key/value in the pre-allocated buffers. The shape should be the number of
 * sequences. For sequence i, the slot_mapping[i]//block_number can get the
 * block index, and the slot_mapping%block_size can get the offset of this
 * block.
 *
 * @tparam DST_T The data type of the output tensors.
 * @tparam SRC_T The data type of the input tensors.
 */
template <typename DST_T, typename SRC_T>
inline void reshape_and_cache_kernel_ref(const at::Tensor &key,
                                         const at::Tensor &value,
                                         const at::Tensor &key_cache,
                                         const at::Tensor &value_cache,
                                         const at::Tensor &slot_mapping) {
  auto num_tokens = key.size(0);
  auto head_num = key.size(1);
  auto head_size = key.size(2);
  auto block_size = key_cache.size(2);
  auto key_cache_ptr = key_cache.data_ptr<DST_T>();
  auto key_ptr = key.data_ptr<SRC_T>();
  auto value_cache_ptr = value_cache.data_ptr<DST_T>();
  auto value_ptr = value.data_ptr<SRC_T>();
  auto slot_mapping_ptr = slot_mapping.data_ptr<int>();
  auto cache_strideN = key_cache.stride(0);
  auto cache_strideP = key_cache.stride(2);
  auto cache_strideH = key_cache.stride(1);
  auto key_state_strideN = key.stride(0);
  auto key_state_strideH = key.stride(1);
  auto value_state_strideN = value.stride(0);
  auto value_state_strideH = value.stride(1);
#pragma omp parallel for collapse(2)
  for (auto ti = 0; ti < num_tokens; ti++) {
    for (auto hi = 0; hi < head_num; hi++) {
      auto physical_block_id = slot_mapping_ptr[ti] / block_size;
      auto block_offset = slot_mapping_ptr[ti] % block_size;
      auto cache_offset = physical_block_id * cache_strideN +
                          block_offset * cache_strideP + hi * cache_strideH;
      auto key_state_offset = ti * key_state_strideN + hi * key_state_strideH;
      auto value_state_offset =
          ti * value_state_strideN + hi * value_state_strideH;
      auto key_cache_start = key_cache_ptr + cache_offset;
      auto key_ptr_start = key_ptr + key_state_offset;
      auto value_cache_start = value_cache_ptr + cache_offset;
      auto value_ptr_start = value_ptr + value_state_offset;
      zentorch::move_ker_ref<DST_T, SRC_T>(key_cache_start, key_ptr_start,
                                           head_size);
      zentorch::move_ker_ref<DST_T, SRC_T>(value_cache_start, value_ptr_start,
                                           head_size);
    }
  }
}

at::Tensor zentorch_attention_single_query_cached_kv_attention_kernel_ref(
    const at::Tensor &out,   // [num_seqs, num_heads, head_size]
    const at::Tensor &query, // [num_seqs, num_heads, head_size]
    const at::Tensor
        &key_cache, // [num_blocks,  block_size, num_heads, head_size]
    const at::Tensor
        &value_cache, // [num_blocks,  block_size, num_heads, head_size]
    const at::Tensor &head_mapping, // [num_heads]
    at::Scalar aten_scale,
    const at::Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
    const at::Tensor &context_lens, // [num_seqs]
    at::Scalar aten_block_size, at::Scalar aten_max_context_len,
    const c10::optional<at::Tensor> &alibi_slopes,
    std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::single_query_cached_kv_attention_kernel_impl",
                  c10::ArrayRef<c10::IValue>({}));
  // dispatch kernel according to the data type of input tensor
  auto scale = aten_scale.to<double>();
  auto block_size = aten_block_size.to<int64_t>();
  auto max_context_len = aten_max_context_len.to<int64_t>();

  if (out.scalar_type() == at::ScalarType::Float) {
    single_query_cached_kv_attention_kernel_ref<float>(
        out, query, key_cache, value_cache, scale, block_tables, context_lens,
        block_size, max_context_len, alibi_slopes);
  } else if (out.scalar_type() == at::ScalarType::BFloat16) {
    single_query_cached_kv_attention_kernel_ref<at::BFloat16>(
        out, query, key_cache, value_cache, scale, block_tables, context_lens,
        block_size, max_context_len, alibi_slopes);
  } else {
    ZENTORCH_CHECK(
        false, "zentorch::single_query_cached_kv_attention support float and "
               "bfloat16!");
  }
  return out;
}

// void reshape_and_cache_kernel_ref
void zentorch_attention_reshape_and_cache_cpu_kernel_ref(
    const at::Tensor &key, const at::Tensor &value, const at::Tensor &key_cache,
    const at::Tensor &value_cache, const at::Tensor &slot_mapping,
    std::string zentorch_op_name) {
  ZENTORCH_CHECK(key.scalar_type() == value.scalar_type(),
                 "key and value should have the same data type");
  ZENTORCH_CHECK(key_cache.scalar_type() == value_cache.scalar_type(),
                 "key_cache and value_cache should have the same data type");
  ZENTORCH_CHECK(slot_mapping.is_contiguous(),
                 "slot_mapping should be contiguous");
  RECORD_FUNCTION("zentorch::reshape_and_cache_cpu_kernel_impl",
                  c10::ArrayRef<c10::IValue>({}));
  if (key.scalar_type() == at::ScalarType::Float) {
    reshape_and_cache_kernel_ref<float, float>(key, value, key_cache,
                                               value_cache, slot_mapping);
  } else if (key.scalar_type() == at::ScalarType::BFloat16) {
    reshape_and_cache_kernel_ref<at::BFloat16, at::BFloat16>(
        key, value, key_cache, value_cache, slot_mapping);
  } else {
    ZENTORCH_CHECK(false,
                   "zentorch::reshape_and_cache support float and bfloat16!");
  }
}

at::Tensor zentorch_attention_single_query_cached_kv_attention_kernel_impl(
    const at::Tensor &out,   // [num_seqs, num_heads, head_size]
    const at::Tensor &query, // [num_seqs, num_heads, head_size]
    const at::Tensor
        &key_cache, // [num_blocks,  block_size, num_heads, head_size]
    const at::Tensor
        &value_cache, // [num_blocks,  block_size, num_heads, head_size]
    const at::Tensor &head_mapping, // [num_heads]
    at::Scalar aten_scale,
    const at::Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
    const at::Tensor &context_lens, // [num_seqs]
    at::Scalar aten_block_size, at::Scalar aten_max_context_len,
    const c10::optional<at::Tensor> &alibi_slopes,
    std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::single_query_cached_kv_attention_kernel_impl",
                  c10::ArrayRef<c10::IValue>({}));
  // dispatch kernel according to the data type of input tensor
  if (is_avx512_supported()) {
    return zentorch_attention_single_query_cached_kv_attention_kernel_512(
        out, query, key_cache, value_cache, head_mapping, aten_scale,
        block_tables, context_lens, aten_block_size, aten_max_context_len,
        alibi_slopes, zentorch_op_name);
  } else {
    return zentorch_attention_single_query_cached_kv_attention_kernel_ref(
        out, query, key_cache, value_cache, head_mapping, aten_scale,
        block_tables, context_lens, aten_block_size, aten_max_context_len,
        alibi_slopes, zentorch_op_name);
  }
}

inline void zentorch_attention_reshape_and_cache_cpu_kernel_impl(
    const at::Tensor &key, const at::Tensor &value, const at::Tensor &key_cache,
    const at::Tensor &value_cache, const at::Tensor &slot_mapping,
    std::string zentorch_op_name) {
  if (is_avx512_supported()) {
    zentorch_attention_reshape_and_cache_cpu_kernel_512(
        key, value, key_cache, value_cache, slot_mapping, zentorch_op_name);
  } else {
    zentorch_attention_reshape_and_cache_cpu_kernel_ref(
        key, value, key_cache, value_cache, slot_mapping, zentorch_op_name);
  }
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_attention_reshape_and_cache(Tensor key, "
        "Tensor value, "
        "Tensor key_cache, "
        "Tensor value_cache, "
        "Tensor slot_mapping, "
        "str zentorch_op_name = "
        "'zentorch::zentorch_attention_reshape_and_cache')->()");

  m.def("zentorch_attention_single_query_cached_kv_attention(Tensor out, "
        "Tensor query, "
        "Tensor key_cache, Tensor value_cache, Tensor head_mapping, Scalar "
        "scale, "
        "Tensor block_tables, Tensor context_lens, Scalar block_size, Scalar "
        "max_context_len, "
        "Tensor? alibi_slopes,"
        "str zentorch_op_name = "
        "'zentorch::zentorch_attention_single_query_cached_kv_attention')-> "
        "Tensor");

  m.def("zentorch_attention_flash_attn_varlen(Tensor out, Tensor query, "
        "Tensor key_cache, Tensor value_cache, Tensor cu_seqlens_q, "
        "Tensor cu_seqlens_k, int max_seqlen_q, int max_seqlen_k, "
        "float softmax_scale, bool is_causal, Tensor block_table, "
        "Tensor? alibi_slopes, int window_size_left, int window_size_right, "
        "str kv_cache_dtype, float k_scale, float v_scale, float softcap, "
        "str zentorch_op_name = "
        "'zentorch::zentorch_attention_flash_attn_varlen')->()");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_attention_reshape_and_cache",
         zentorch::zentorch_attention_reshape_and_cache_cpu_kernel_impl);

  m.impl("zentorch_attention_single_query_cached_kv_attention",
         zentorch::
             zentorch_attention_single_query_cached_kv_attention_kernel_impl);

  m.impl("zentorch_attention_flash_attn_varlen",
         zentorch::zentorch_attention_flash_attn_varlen_kernel_impl);
}

} // namespace zentorch
#endif // TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR > 3
