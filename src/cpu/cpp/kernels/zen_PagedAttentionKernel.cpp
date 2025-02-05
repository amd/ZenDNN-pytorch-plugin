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

#include "../Utils.hpp"
#include "vec/add_softmax.h"
#include "zen_cpukernels.hpp"
#include <limits>
#include <omp.h>

template <
    typename scalar_t,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
static inline scalar_t *conditional_data_ptr(float *ptr, scalar_t *ptr2) {
  return ptr2;
}

namespace zentorch {

template <typename scalar_t>
inline void fill_stub(scalar_t *data, scalar_t val, int64_t size) {
  using Vec = Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }

#pragma unroll
  for (; d < size; d++) {
    data[d] = val;
  }
}

inline c10::SymFloat calculate_scale(const at::Tensor &query,
                                     c10::optional<double> scale) {
  const auto softmax_scale =
      scale.has_value()
          ? scale.value()
          : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

// TODO: Add BF16 Support and GQA support
template <typename QT, typename KT, typename CT>
inline void _reduce_head(const QT *q_ptr_start, const KT *k_ptr_start,
                         float *attn_w_pos, int64_t head_size, bool store_key,
                         CT *k_cache_start) {
  int64_t hsi = 0;
  int64_t vec_size = 16; // 512/32
  auto qk_sum_vec = _mm512_setzero_ps();
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    auto q_vec = _loadu(q_ptr_start + hsi);
    auto k_vec = _loadu(k_ptr_start + hsi);
    if (store_key) {
      _storeu(k_cache_start + hsi, k_vec);
    }
    qk_sum_vec = _mm512_fmadd_ps(q_vec, k_vec, qk_sum_vec);
  }
  attn_w_pos[0] += _mm512_reduce_add_ps(qk_sum_vec);
  for (; hsi < head_size; hsi++) {
    if (store_key) {
      k_cache_start[hsi] =
          (float)k_ptr_start[hsi]; // cat the key into the key_cache.
    }
    attn_w_pos[0] += q_ptr_start[hsi] * (float)k_ptr_start[hsi];
  }
}

template <typename QT, typename KT>
inline void reduce_head(const QT *q_ptr_start, int64_t kv_head_group_size,
                        const KT *k_cache_start, float *attn_w_pos,
                        int attn_w_stride, int64_t head_size) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    attn_w_pos[i * attn_w_stride] = 0;
    _reduce_head<QT, KT, KT>(q_ptr_start + i * head_size, k_cache_start,
                             attn_w_pos + i * attn_w_stride, head_size, false,
                             nullptr);
  }
}

// TODO: Add BF16 Support and GQA support
template <typename VT, typename OT, typename CT>
inline void _mul_and_accumulate(const float &attn_w, const VT *v_ptr_start,
                                OT *attn_out_start, int64_t head_size,
                                bool store_value, CT *v_cache_start,
                                int accumulated) {
  int64_t vec_size = 16; // 512/32
  int64_t hsi = 0;
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    auto attn_w_vec = _mm512_set1_ps(attn_w);
    auto v_vec = _loadu(v_ptr_start + hsi);
    if (accumulated) {
      auto attn_out_vec = _loadu(attn_out_start + hsi);
      auto attn_out_vec_new = _mm512_fmadd_ps(attn_w_vec, v_vec, attn_out_vec);
      _storeu(attn_out_start + hsi, attn_out_vec_new);
    } else {
      auto attn_out_vec_new = _mm512_mul_ps(attn_w_vec, v_vec);
      _storeu(attn_out_start + hsi, attn_out_vec_new);
    }
    if (store_value) {
      _storeu(v_cache_start + hsi, v_vec);
    }
  }
  for (; hsi < head_size; hsi++) {
    if (accumulated) {
      attn_out_start[hsi] += attn_w * (float)v_ptr_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * (float)v_ptr_start[hsi];
    }
    if (store_value) {
      v_cache_start[hsi] = (float)v_ptr_start[hsi];
    }
  }
}

template <typename OT, typename CT>
inline void mul_attenion_weights_and_value_of_head(
    const float *attn_w, int attn_w_stride, const CT *v_cache_start,
    OT *attn_out_start, int attn_out_strideH, int kv_head_group_size,
    int64_t head_size, bool accumulated) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    _mul_and_accumulate<CT, OT, CT>(attn_w[i * attn_w_stride], v_cache_start,
                                    attn_out_start + i * attn_out_strideH,
                                    head_size, false, nullptr, accumulated);
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
// TODO: Implement the vectorized version
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
inline void single_query_cached_kv_attention_kernel(
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

  // TODO: Generate heuristics for different BS, I/p sizes, model + vLLM
  // configurable parameters
  auto partition_size = 128;

  auto max_num_partitions =
      (max_context_len + partition_size - 1) / partition_size;

  auto max_logits = at::empty({num_seqs, num_heads, max_num_partitions + 1},
                              query.options().dtype(at::ScalarType::Float));

  auto exp_sum = at::empty({num_seqs, num_heads, max_num_partitions + 1},
                           query.options().dtype(at::ScalarType::Float));

  // TODO: Use temporary buffer for store and load in the kernel
  auto tmp_out = at::empty({num_seqs, num_heads, max_num_partitions, head_size},
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
  // TODO: Check perf with dynamic scheduling
  {
    RECORD_FUNCTION("zentorch::compute_single_query_cached_kv_attention_kernel",
                    c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(3) schedule(static, 1)
    for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
      for (auto partition_id = 0; partition_id < max_num_partitions;
           partition_id++) {
        for (auto head_group_start = 0; head_group_start < num_heads;
             head_group_start += kv_head_group_size) {
          auto context_len = context_lens_ptr[seq_id];
          auto partition_start = partition_id * partition_size;
          if (partition_start >= context_len) {
            continue;
          }
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
                                head_group_start * exp_sum_strideH +
                                partition_id;
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
            auto tokens_in_block = std::min(
                block_size, context_len - logical_block_id * block_size);
            auto token_start = logical_block_id * block_size;
            auto token_end = token_start + tokens_in_block;
            for (auto token_id = token_start; token_id < token_end;
                 token_id++) {
              auto block_offset = token_id - token_start;
              auto k_cache_start = key_cache_ptr +
                                   physical_block_id * kv_block_strideN +
                                   block_offset * kv_block_strideP +
                                   kv_head_id * kv_block_strideH;
              reduce_head(q_ptr_start, kv_head_group_size, k_cache_start,
                          &(logits[logits_position]), partition_size,
                          head_size);
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
            auto tokens_in_block = std::min(
                block_size, context_len - logical_block_id * block_size);
            auto token_start = logical_block_id * block_size;
            auto token_end = token_start + tokens_in_block;
            for (auto token_id = token_start; token_id < token_end;
                 token_id++) {
              auto block_offset = token_id - token_start;
              auto v_cache_start = value_cache_ptr +
                                   physical_block_id * kv_block_strideN +
                                   block_offset * kv_block_strideP +
                                   kv_head_id * kv_block_strideH;
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
  }
  // calculate the final output
  {
    RECORD_FUNCTION(
        "zentorch::final_out_single_query_cached_kv_attention_kernel",
        c10::ArrayRef<c10::IValue>({}));

#pragma omp parallel for collapse(2)
    for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
      for (auto head_id = 0; head_id < num_heads; head_id++) {
        auto global_max = -std::numeric_limits<float>::infinity();
        auto global_exp_sum = 0.0;
        auto context_len = context_lens_ptr[seq_id];
        auto partition_num =
            (context_len + partition_size - 1) / partition_size;
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
            auto exp_sum =
                exp_sum_ptr[seq_id * exp_sum_strideN +
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
  }

} // single_query_cached_kv_attention_kernel

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
inline void reshape_and_cache_kernel(const at::Tensor &key,
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
      zentorch::move_ker<DST_T, SRC_T>(key_cache_start, key_ptr_start,
                                       head_size);
      zentorch::move_ker<DST_T, SRC_T>(value_cache_start, value_ptr_start,
                                       head_size);
    }
  }
}

at::Tensor zentorch_attention_single_query_cached_kv_attention_kernel_512(
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
    single_query_cached_kv_attention_kernel<float>(
        out, query, key_cache, value_cache, scale, block_tables, context_lens,
        block_size, max_context_len, alibi_slopes);
  } else if (out.scalar_type() == at::ScalarType::BFloat16) {
    single_query_cached_kv_attention_kernel<at::BFloat16>(
        out, query, key_cache, value_cache, scale, block_tables, context_lens,
        block_size, max_context_len, alibi_slopes);
  } else {
    ZENTORCH_CHECK(false,
                   "zentorch::single_query_cached_kv_attention supports only "
                   "float and bfloat16 data types");
  }
  return out;
}

// void reshape_and_cache_kernel
void zentorch_attention_reshape_and_cache_cpu_kernel_512(
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
    reshape_and_cache_kernel<float, float>(key, value, key_cache, value_cache,
                                           slot_mapping);
  } else if (key.scalar_type() == at::ScalarType::BFloat16) {
    reshape_and_cache_kernel<at::BFloat16, at::BFloat16>(
        key, value, key_cache, value_cache, slot_mapping);
  } else {
    ZENTORCH_CHECK(false, "zentorch::reshape_and_cache supports only float and "
                          "bfloat16 data types");
  }
}

} // namespace zentorch
#endif // TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR > 3
