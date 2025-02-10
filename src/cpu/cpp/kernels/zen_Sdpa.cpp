/******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Was sourced from
 * https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp
 * PyTorch commit ID: d990dad
 ******************************************************************************/

#include <torch/torch.h>
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR > 3
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include "../Memory.hpp"
#include "zen_cpukernels.hpp"

#include <blis.h>

#define ZENDNN_MATMUL_ENABLE 1
namespace zentorch {

using namespace at::vec;

inline void zendnn_gemm(int64_t m, int64_t n, int64_t k, float alpha,
                        at::BFloat16 *a, int64_t lda, at::BFloat16 *b,
                        int64_t ldb, float beta, float *c, int64_t ldc,
                        bool TransA, bool TransB) {
  engine eng = utils::engine::cpu_engine();
  zendnn::stream engine_stream(eng);
  using dt = memory::data_type;
  memory::dims a_dims;
  memory::dims b_dims;
  memory::dims c_dims;
  a_dims = {m, k};
  b_dims = {k, n};
  c_dims = {m, n};
  memory::dims a_strides = TransA ? memory::dims{1, lda} : memory::dims{lda, 1};
  memory::dims b_strides = TransB ? memory::dims{1, ldb} : memory::dims{ldb, 1};
  // Create memory descriptors
  memory::desc a_md = memory::desc({a_dims}, dt::bf16, a_strides);
  memory::desc c_md = memory::desc({c_dims}, dt::f32, {ldc, 1});
  memory::desc b_md = memory::desc({b_dims}, dt::bf16, b_strides, false);
  zendnn::memory b_memory, a_memory, c_memory;
  a_memory = memory(a_md, eng, a);
  c_memory = memory(c_md, eng, c);
  b_memory = memory(b_md, eng, b);

  zendnn::primitive_attr matmul_attr;
  matmul_attr.set_output_scales(0, {alpha});
  zendnn::post_ops post_ops;
  if (beta != 0.0) {
    post_ops.append_sum(beta);
  }
  matmul_attr.set_post_ops(post_ops);
  // Create descriptor for matmul
  auto matmul_pd1 = zendnn::matmul::desc(a_md, b_md, c_md);
  // Create primitive descriptor for matmul
  auto matmul_pd = zendnn::matmul::primitive_desc(matmul_pd1, matmul_attr, eng);
  auto matmul_prim = matmul(matmul_pd);
  // Primitive arguments.
  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({ZENDNN_ARG_SRC, a_memory});
  matmul_args.insert({ZENDNN_ARG_WEIGHTS, b_memory});
  matmul_args.insert({ZENDNN_ARG_DST, c_memory});
  // Primitive execution: matrix multiplication.
  matmul_prim.execute(engine_stream, matmul_args);
  // Wait for the computation to finalize.
  engine_stream.wait();
}

// out = val * a + b
template <typename T1, typename T2>
inline void _scale_attn_mask_fusion_kernel(T1 *a, T2 *b, const int &size,
                                           T1 *out, T1 &val) {
  const auto vec_size1 = at::vec::Vectorized<T1>::size();
  const auto vec_size2 = at::vec::Vectorized<T2>::size();
  constexpr int64_t T1_n =
      (vec_size2 == vec_size1 * 2 && is_reduced_floating_point_v<T2>) ? 2 : 1;
  constexpr int64_t T2_n = 1;
  auto vec_scale = at::vec::VectorizedN<T1, T1_n>(val);
  int64_t i = 0;
  for (; i < size - (size % vec_size2); i += vec_size2) {
    auto a_n = at::vec::VectorizedN<T1, T1_n>::loadu(a + i);
    auto b_n = at::vec::VectorizedN<T2, T2_n>::loadu(b + i);
    auto b_n_convert = at::vec::convert<T1, T1_n, T2, T2_n, true>(b_n);
    auto res = a_n * vec_scale + b_n_convert;
    res.store(out + i);
  }
  for (; i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = (T1)b[i];
    out[i] = tmp0 * val + tmp1;
  }
}

template <typename scalar_t>
inline Vectorized<scalar_t> exp_u20(Vectorized<scalar_t> data) {
  return data.exp_u20();
}
#if defined(CPU_CAPABILITY_AVX512)
// To implement exp_u20 here is faster than calling from add_softmax.h or PT
// vec512_float.h
inline Vectorized<float> exp_u20(Vectorized<float> data) {
  __m512 values = __m512(data);
  // A faster version of exp with ULP=20
  static __m512 vec_factorial_1 =
      _mm512_set1_ps(0.999999701f); // 1/factorial(1)
  static __m512 vec_factorial_2 =
      _mm512_set1_ps(0.499991506f); // 1/factorial(2)
  static __m512 vec_factorial_3 =
      _mm512_set1_ps(0.166676521f); // 1/factorial(3)
  static __m512 vec_factorial_4 =
      _mm512_set1_ps(0.0418978221f); // 1/factorial(4)
  static __m512 vec_factorial_5 =
      _mm512_set1_ps(0.00828929059f); // 1/factorial(5)
  static __m512 vec_exp_log2ef =
      (__m512)_mm512_set1_epi32(0x3fb8aa3b); // log2(e)
  static __m512 vec_half = _mm512_set1_ps(0.5f);
  static __m512 vec_one = _mm512_set1_ps(1.f);
  static __m512 vec_zero = _mm512_set1_ps(0.f);
  static __m512 vec_two = _mm512_set1_ps(2.f);
  static __m512 vec_ln2f = (__m512)_mm512_set1_epi32(0x3f317218); // ln(2)
  static __m512 vec_ln_flt_min = (__m512)_mm512_set1_epi32(0xc2aeac50);
  static __m512 vec_ln_flt_max = (__m512)_mm512_set1_epi32(0x42b17218);
  static __m512i vec_127 = _mm512_set1_epi32(0x0000007f);
  static int n_mantissa_bits = 23;

  // exp(x) =
  // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
  // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression

  auto less_ln_flt_min_mask =
      _mm512_cmp_ps_mask(values, vec_ln_flt_min, 1 /*_CMP_LT_OS*/);
  auto vec_src = _mm512_min_ps(values, vec_ln_flt_max);
  vec_src = _mm512_max_ps(vec_src, vec_ln_flt_min);

  // fx = floorf(x * log2ef + 0.5)
  auto vec_fx = _mm512_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);
  auto vec_fx_i = _mm512_cvt_roundps_epi32(vec_fx, _MM_FROUND_TO_NEG_INF |
                                                       _MM_FROUND_NO_EXC);
  vec_fx = _mm512_cvtepi32_ps(vec_fx_i);

  // x = x - fx * ln2
  auto vec_exp_poly = _mm512_fnmadd_ps(vec_fx, vec_ln2f, vec_src);

  // compute polynomial
  auto vec_res =
      _mm512_fmadd_ps(vec_exp_poly, vec_factorial_5, vec_factorial_4);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_one);

  // compute 2^(n-1)
  auto vec_exp_number = _mm512_sub_ps(vec_fx, vec_one);
  auto vec_exp_number_i = _mm512_cvtps_epi32(vec_exp_number);
  auto vec_two_pow_n_i = _mm512_add_epi32(vec_exp_number_i, vec_127);
  vec_two_pow_n_i = _mm512_slli_epi32(vec_two_pow_n_i, n_mantissa_bits);
  auto vec_two_pow_n = (__m512)vec_two_pow_n_i;
  vec_two_pow_n =
      _mm512_mask_blend_ps(less_ln_flt_min_mask, vec_two_pow_n, vec_zero);

  // y = y * 2^n
  vec_res = _mm512_mul_ps(vec_res, vec_two_pow_n);
  vec_res = _mm512_mul_ps(vec_res, vec_two);
  return vec_res;
}
#endif

template <typename scalar_t>
inline void _store(scalar_t *dst, at::vec::Vectorized<scalar_t> src) {
  src.store(dst);
}

template <typename scalar_t>
inline typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, void>
_store(scalar_t *dst, at::vec::Vectorized<float> src) {
  auto res = at::vec::convert_from_float<scalar_t>(src, src);
  res.store(dst, at::vec::Vectorized<float>::size());
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
  for (int64_t i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = exp_u20(tmp1);
    vec_tmp_sum += tmp2;
    _store(out + i, tmp2);
  }
  tmp_sum = at::vec::vec_reduce_all<T1>(
      [](at::vec::Vectorized<T1> &x, at::vec::Vectorized<T1> &y) {
        return x + y;
      },
      vec_tmp_sum);
  for (int64_t i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}

// 1) out = a * scale
// 2) max = max(out)
template <typename scalar_t>
inline void
_mul_reduce_max_fusion_kernel(const scalar_t *a, const scalar_t &scale,
                              const int &size, scalar_t *out, scalar_t &max) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  auto vec_scale = at::vec::Vectorized<scalar_t>(scale);
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = at::vec::Vectorized<scalar_t>(tmp_max);
  for (int64_t i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    vec_tmp_max = at::vec::maximum(vec_tmp_max, tmp1);
    _store(out + i, tmp1);
  }
  for (int64_t i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    tmp_max = std::max(tmp_max, tmp1);
    out[i] = tmp1;
  }
  max = std::max(tmp_max, at::vec::vec_reduce_all<scalar_t>(
                              [](at::vec::Vectorized<scalar_t> &x,
                                 at::vec::Vectorized<scalar_t> &y) {
                                return at::vec::maximum(x, y);
                              },
                              vec_tmp_max));
}

template <typename scalar_t>
static inline scalar_t *conditional_data_ptr(scalar_t *ptr, scalar_t *ptr2) {
  ZENTORCH_CHECK(ptr2 == nullptr);
  return ptr;
}

template <
    typename scalar_t,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
static inline scalar_t *conditional_data_ptr(float *ptr, scalar_t *ptr2) {
  return ptr2;
}

template <typename scalar_t>
inline void fill_stub(scalar_t *data, scalar_t val, int64_t size) {
  using Vec = Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }
#if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
#pragma unroll
#endif
  for (; d < size; d++) {
    data[d] = val;
  }
}

void reshape_attn_mask_to_4d(at::Tensor &attn_mask, int64_t batchSize,
                             int64_t num_head, int64_t qSize, int64_t kvSize) {
  // Support mask shapes:
  // 2d: ({Q_seq_len, 1}  x {KV_seq_len, 1})
  // 4d: ({Batch, 1} x {Num_heads, 1} x {Q_seq_len, 1}  x {KV_seq_len, 1})
  // Guaranteed in check_attn_mask_shape
  int64_t attn_mask_size_0 = 1;
  int64_t attn_mask_size_1 = 1;
  if (attn_mask.dim() == 4) {
    if (attn_mask.size(0) == batchSize) {
      attn_mask_size_0 = batchSize;
    }
    if (attn_mask.size(1) == num_head) {
      attn_mask_size_1 = num_head;
    }
  }
  attn_mask = attn_mask
                  .view({attn_mask_size_0, attn_mask_size_1, attn_mask.size(-2),
                         attn_mask.size(-1)})
                  .expand({attn_mask_size_0, attn_mask_size_1, qSize, kvSize});
}

inline c10::SymFloat calculate_scale(const at::Tensor &query,
                                     c10::optional<double> scale) {
  const auto softmax_scale =
      scale.has_value()
          ? scale.value()
          : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

template <typename scalar_t, typename mask_t, int64_t q_split_size,
          int64_t kv_split_size>
void cpu_flash_attention(const at::Tensor &output, const at::Tensor &logsumexp,
                         const at::Tensor &q, const at::Tensor &k,
                         const at::Tensor &v, double dropout_p, bool is_causal,
                         std::optional<at::Tensor> attn_mask,
                         std::optional<double> scale) {
  // Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
  //    -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
  // Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  // Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  at::Tensor query = q.transpose(1, 2);
  at::Tensor key = k.transpose(1, 2);
  at::Tensor value = v.transpose(1, 2);

  constexpr bool is_reduced_type = is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = calculate_scale(query, scale).as_float_unchecked();

  // Sizes
  ZENTORCH_CHECK((query.size(3) == value.size(3)) &&
                     (key.size(3) == value.size(3)),
                 "zentorch_scaled_dot_product_attention_flash_attention: Q/K/V "
                 "should have the "
                 "same head size");
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);

  bool has_attn_mask = attn_mask.has_value() && attn_mask.value().numel();
  if (has_attn_mask) {
    reshape_attn_mask_to_4d(attn_mask.value(), batchSize, num_head, qSize,
                            kvSize);
  }

  // Strides
  int64_t qStrideB = query.stride(0);
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  int64_t kStrideB = key.stride(0);
  int64_t kStrideN = key.stride(1);
  int64_t kStrideH = key.stride(2);
  int64_t vStrideB = value.stride(0);
  int64_t vStrideN = value.stride(1);
  int64_t vStrideH = value.stride(2);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);
  int64_t lStrideB = logsumexp.stride(0);
  int64_t lStrideM = logsumexp.stride(1);
  int64_t lStrideH = logsumexp.stride(2);
  int64_t mStrideB = (has_attn_mask && attn_mask.value().size(0) > 1)
                         ? attn_mask.value().stride(0)
                         : 0;
  int64_t mStrideH = (has_attn_mask && attn_mask.value().size(1) > 1)
                         ? attn_mask.value().stride(1)
                         : 0;
  int64_t mStrideM = has_attn_mask ? attn_mask.value().stride(2) : 0;

  // TODO: Generate more heuristics based on bs, seq_len
  // and device cache capacity. Decide more fine grain
  // q_split_size and kv_split_size for lower bs. Current
  // q_split_size works optimally for bs > 4.
  int zen_q_split_size = q_split_size;
  if (batchSize > 4) {
    zen_q_split_size = 512;
  }
  int64_t qSplitSize = zen_q_split_size > qSize ? qSize : zen_q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t num_thread = at::get_num_threads();

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kvSplitSize +
      /* qk_max */ qSplitSize +
      /* qk_sum */ qSplitSize +
      /* dst    */ qSplitSize * headSize;

  at::Tensor buf = at::empty({num_thread, size_per_thread},
                             query.options().dtype(accumulate_dtype));
  at::Tensor buf_reduced =
      at::empty({num_thread, qSplitSize, is_reduced_type ? kvSplitSize : 0},
                query.options());

  // Data ptrs
  const scalar_t *q_data = query.const_data_ptr<scalar_t>();
  const scalar_t *k_data = key.const_data_ptr<scalar_t>();
  const scalar_t *v_data = value.const_data_ptr<scalar_t>();
  mask_t *mask_data =
      has_attn_mask ? attn_mask.value().data_ptr<mask_t>() : nullptr;
  scalar_t *out_data = output.data_ptr<scalar_t>();
  accum_t *lse_data = logsumexp.data_ptr<accum_t>();
  accum_t *buf_data = buf.data_ptr<accum_t>();
  scalar_t *buf_reduced_data =
      is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;

  at::parallel_for(
      0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0, k = 0;
        at::native::data_index_init(begin, i, batchSize, j, num_head, k,
                                    qSlice);
        int ompIdx = at::get_thread_num();
        accum_t *buf_ptr = buf_data + ompIdx * size_per_thread;
        accum_t *qk_data = buf_ptr;
        accum_t *qk_max_data = qk_data + qSplitSize * kvSplitSize;
        accum_t *qk_sum_data = qk_max_data + qSplitSize;
        accum_t *dst_data = qk_sum_data + qSplitSize;
        scalar_t *qk_reduced_data =
            is_reduced_type
                ? buf_reduced_data + ompIdx * qSplitSize * kvSplitSize
                : nullptr;

        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable
          int64_t m = k * qSplitSize;
          int64_t qBlockSize = std::min(qSplitSize, qSize - m);
          // Initialize max and sum
          fill_stub(qk_max_data, -std::numeric_limits<accum_t>::infinity(),
                    qBlockSize);
          fill_stub(qk_sum_data, static_cast<accum_t>(0), qBlockSize);
          int64_t num_keys =
              is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
          for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            // Calculate scale * q @ k.T
            omp_set_max_active_levels(1);
            // TODO use of pack and compute API
            if (ZENDNN_MATMUL_ENABLE) {
              zendnn_gemm(qBlockSize, kvBlockSize, headSize, 1.0,
                          (at::BFloat16 *)q_data + i * qStrideB + j * qStrideH +
                              m * qStrideM,
                          qStrideM,
                          (at::BFloat16 *)k_data + i * kStrideB + j * kStrideH +
                              n * kStrideN,
                          kStrideN, 0.0, qk_data, kvBlockSize, false, true);
            } else {
              aocl_gemm_bf16bf16f32of32(
                  'r', 'n', 't', qBlockSize, kvBlockSize, headSize, 1.0,
                  (int16_t *)q_data + i * qStrideB + j * qStrideH +
                      m * qStrideM,
                  qStrideM, 'n',
                  (int16_t *)k_data + i * kStrideB + j * kStrideH +
                      n * kStrideN,
                  kStrideN, 'n', 0.0, qk_data, kvBlockSize, NULL);
            }
            // Apply causal mask, fill unused with -inf
            if (is_causal && num_keys - n <= kvSplitSize) {
              for (const auto row : c10::irange(qBlockSize)) {
                int64_t last_col = m + row - n;
                accum_t *row_ptr = qk_data + row * kvBlockSize;
                fill_stub(row_ptr + last_col + 1,
                          -std::numeric_limits<accum_t>::infinity(),
                          kvBlockSize - last_col - 1);
              }
            }
            // Update attention weights with attention mask
            // And apply scaling factor
            // qk <- qk * scaling + attn_mask
            if (has_attn_mask) {
              for (int64_t row = 0; row < qBlockSize; ++row) {
                // TODO Reuse of buffer based on mask_data  with mStrideN
                _scale_attn_mask_fusion_kernel(
                    qk_data + row * kvBlockSize,
                    mask_data + i * mStrideB + j * mStrideH +
                        (m + row) * mStrideM + n,
                    kvBlockSize, qk_data + row * kvBlockSize, scaling_factor);
              }
            }
            // Update coefficients with Softmax
            accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
            for (int64_t row = 0; row < qBlockSize; ++row) {
              if (has_attn_mask) {
                // max per row
                tmp_max = at::vec::reduce_all<accum_t>(
                    [](Vec &x, Vec &y) { return at::vec::maximum(x, y); },
                    qk_data + row * kvBlockSize, kvBlockSize);
              } else {
                // apply scaling factor and max per row in fusion
                _mul_reduce_max_fusion_kernel(
                    qk_data + row * kvBlockSize, scaling_factor, kvBlockSize,
                    qk_data + row * kvBlockSize, tmp_max);
              }
              tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
              // qk <- exp(qk - max) and sum per row
              tmp_sum = tmp_max;
              _exp_reduce_sum_fusion_kernel(
                  qk_data + row * kvBlockSize, kvBlockSize,
                  conditional_data_ptr(qk_data, qk_reduced_data) +
                      row * kvBlockSize,
                  tmp_sum);
              // exp_tmp <- exp(max[row] - max)
              exp_tmp = std::exp(qk_max_data[row] - tmp_max);
              // sum[row] <- sum + exp_tmp * sum[row]
              qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
              // max[row] <- max
              qk_max_data[row] = tmp_max;
              // dst <- dst * exp_tmp
              if (n > 0) {
                at::vec::map<accum_t>(
                    [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
                    dst_data + row * headSize, dst_data + row * headSize,
                    headSize);
              }
            }
            // Calculate Softmax(q @ k.T) @ v
            omp_set_max_active_levels(1);
            // TODO use of pack and compute API
            if (ZENDNN_MATMUL_ENABLE) {
              zendnn_gemm(qBlockSize, headSize, kvBlockSize, 1.0,
                          (at::BFloat16 *)conditional_data_ptr(qk_data,
                                                               qk_reduced_data),
                          kvBlockSize,
                          (at::BFloat16 *)v_data + i * vStrideB + j * vStrideH +
                              n * vStrideN,
                          vStrideN, n == 0 ? 0.0 : 1.0, dst_data, headSize,
                          false, false);
            } else {
              aocl_gemm_bf16bf16f32of32(
                  'r', 'n', 'n', qBlockSize, headSize, kvBlockSize, 1.0,
                  (int16_t *)conditional_data_ptr(qk_data, qk_reduced_data),
                  kvBlockSize, 'n',
                  (int16_t *)v_data + i * vStrideB + j * vStrideH +
                      n * vStrideN,
                  vStrideN, 'n', n == 0 ? 0.0 : 1.0, dst_data, headSize, NULL);
            }
          }
          // dst <- dst / sum[row]
          // reorder MHA output with strides
          for (int64_t row = 0; row < qBlockSize; ++row) {
            accum_t sum_reciprocal = 1 / qk_sum_data[row];
            at::vec::map<scalar_t>(
                [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
                out_data + i * oStrideB + j * oStrideH + m * oStrideM +
                    row * oStrideM,
                dst_data + row * headSize, headSize);
          }
          // Store logsumexp for backward
          accum_t *lse_ptr =
              lse_data + i * lStrideB + j * lStrideH + m * lStrideM;
          for (const auto row : c10::irange(qBlockSize)) {
            lse_ptr[row * lStrideM] =
                qk_max_data[row] + std::log(qk_sum_data[row]);
          }
          // Move to the next query
          at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
        }
      });
}

void flash_attention_kernel_impl_512(
    const at::Tensor &output, const at::Tensor &logsumexp,
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    double dropout_p, bool is_causal, std::optional<at::Tensor> attn_mask,
    std::optional<double> scale) {
  auto q_seq_len = query.size(2);
  // Current parallelization is based on q split size.
  // kv block size is depend on kv split size
  // These values are based on limited heuristics based on zen architecture
  // TODO Try different splits based on zen caches
  if (q_seq_len >= 768) {
    cpu_flash_attention<at::BFloat16, at::BFloat16, 256, 512>(
        output, logsumexp, query, key, value, dropout_p, is_causal, attn_mask,
        scale);
  } else if (q_seq_len >= 192) {
    cpu_flash_attention<at::BFloat16, at::BFloat16, 64, 512>(
        output, logsumexp, query, key, value, dropout_p, is_causal, attn_mask,
        scale);
  } else {
    cpu_flash_attention<at::BFloat16, at::BFloat16, 32, 512>(
        output, logsumexp, query, key, value, dropout_p, is_causal, attn_mask,
        scale);
  }
}
} // namespace zentorch
#endif // TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR > 3
