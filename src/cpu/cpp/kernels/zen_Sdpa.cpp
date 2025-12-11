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

#include "../EnvReader.hpp"
#include "../MatmulUtils.hpp"
#include "../Memory.hpp"
#include "zen_cpukernels.hpp"

namespace zentorch {

using namespace at::vec;

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

  // Set the aligned size for the tensor based on whether it is transposed.
  // Aligned size is used to set the actual size of tensor passed.
  // If the tensor is transposed, align using the second dimension's stride and
  // size. Otherwise, align using the first dimension's size and stride.

  // Strides convey the actual size of tensor.
  // That's why we need to multiply the leading dimension size and leading
  // dimension stride if the tensor is contiguous. If the tensor is transposed,
  // we need to multiply the trailing dimension size and trailing dimension
  // stride.

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

template <typename T>
inline void zendnn_gemm(int64_t m, int64_t n, int64_t k, float alpha,
                        const T *a, int64_t lda, const T *b, int64_t ldb,
                        float beta, float *c, int64_t ldc, bool TransA,
                        bool TransB) {

  constexpr bool is_input_float = std::is_same_v<T, float>;
  // Retrieve environment variables
  const int &zendnn_matmul_direct_env_value =
      EnvReader::getEnvVariableAsInt("USE_ZENDNN_SDPA_MATMUL_DIRECT");

  if (zendnn_matmul_direct_env_value) {
    zendnnl::lowoha::lowoha_params params;
    zendnnl::lowoha::data_types matmul_dtype;
    matmul_dtype.bias = data_type_t::none;
    matmul_dtype.compute = data_type_t::none;

    matmul_dtype.src = is_input_float ? data_type_t::f32 : data_type_t::bf16;
    matmul_dtype.wei = is_input_float ? data_type_t::f32 : data_type_t::bf16;
    matmul_dtype.dst =
        data_type_t::f32; // Destination data type is always Float32.
    params.dtypes = matmul_dtype;

    zendnnl::lowoha::batch_params_t batch_params;
    batch_params.Batch_A = 1;
    batch_params.Batch_B = 1;

    matmul_direct('r' /* layout: row-major */, TransA, TransB, m, n, k, alpha,
                  a, lda, b, ldb, nullptr, /* No bias */ beta, c, ldc,
                  false /* is_weights_const */, batch_params, params);
  } else {
    tensor_t mat1_tensor = tensor_t();
    const std::vector<unsigned long> sizes_a = {static_cast<unsigned long>(m),
                                                static_cast<unsigned long>(k)};
    const std::vector<unsigned long> strides_a =
        TransA ? std::vector<unsigned long>{1, static_cast<unsigned long>(lda)}
               : std::vector<unsigned long>{static_cast<unsigned long>(lda), 1};
    set_zendnnl_tensor_attributes_wrapper(a, mat1_tensor, "matmul_input",
                                          sizes_a, strides_a, is_input_float,
                                          TransA);

    tensor_t mat2_tensor = tensor_t();
    const std::vector<unsigned long> sizes_b = {static_cast<unsigned long>(k),
                                                static_cast<unsigned long>(n)};
    const std::vector<unsigned long> strides_b =
        TransB ? std::vector<unsigned long>{1, static_cast<unsigned long>(ldb)}
               : std::vector<unsigned long>{static_cast<unsigned long>(ldb), 1};
    set_zendnnl_tensor_attributes_wrapper(b, mat2_tensor, "weights", sizes_b,
                                          strides_b, is_input_float, TransB);

    tensor_t result = tensor_t();
    const std::vector<unsigned long> sizes_c = {static_cast<unsigned long>(m),
                                                static_cast<unsigned long>(n)};
    const std::vector<unsigned long> strides_c = {
        static_cast<unsigned long>(ldc), 1};
    set_zendnnl_tensor_attributes_wrapper(
        c, result, "matmul_output", sizes_c, strides_c,
        /* is_input_float */ true, /* is_transposed */ false);
    // Destination tensor is always Float32 and contiguous.

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
  }
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
inline Vectorized<scalar_t> fexp_u20(Vectorized<scalar_t> data) {
  return data.fexp_u20();
}
#if defined(CPU_CAPABILITY_AVX512)
inline Vectorized<float> fexp_u20(Vectorized<float> data) {
  __m512 values = __m512(data);
  static __m512 vec_c0 = _mm512_set1_ps(0.00010703434948458272f);
  static __m512 vec_c1 = _mm512_set1_ps(0.30354260500649682f);
  static __m512 vec_c2 = _mm512_set1_ps(-0.22433836478672356);
  static __m512 vec_c3 = _mm512_set1_ps(-0.079204240219773236);

  static __m512 vec_exp_log2ef =
      (__m512)_mm512_set1_epi32(0x3fb8aa3b); // log2(e)

  static __m512 vec_a = _mm512_set1_ps(std::pow(2, 23) / std::log2(2));
  static __m512 vec_b = _mm512_set1_ps(std::pow(2, 23) * 127.f);

  static __m512 vec_ln_flt_min = (__m512)_mm512_set1_epi32(0xc2aeac50);
  static __m512 vec_ln_flt_max = (__m512)_mm512_set1_epi32(0x42b17218);
  static __m512i vec_infinity = _mm512_set1_epi32(0x7F800000);
  static __m512i vec_zero = _mm512_setzero_epi32();

  // Fast Exponential Computation on SIMD Architectures
  // A. Cristiano I. Malossi, Yves Ineichen, Costas Bekas, and Alessandro
  // Curioni exp(x) = 2**(x * log2(e))
  //        = 2**xi * 2**xf   - TIPS we are using  the EEEE floating point
  //        representation with identification to the exponent and the
  //        mentissa
  //  2**xf will be approximated to a polynomial of degree 3 computed with
  //  Horner method
  // mask for the boundary condition
  auto min_mask = _mm512_cmp_ps_mask(values, vec_ln_flt_min, _CMP_LT_OS);
  auto max_mask = _mm512_cmp_ps_mask(values, vec_ln_flt_max, _CMP_GT_OS);

  // transformation with log2(e)
  auto vec_src = _mm512_mul_ps(values, vec_exp_log2ef);
  auto vec_fractional = _mm512_sub_ps(vec_src, _mm512_floor_ps(vec_src));

  // compute polynomial using Horner Scheme, for superscalar processor
  auto vec_res = _mm512_fmadd_ps(vec_fractional, vec_c3, vec_c2);
  vec_res = _mm512_fmadd_ps(vec_fractional, vec_res, vec_c1);
  vec_res = _mm512_fmadd_ps(vec_fractional, vec_res, vec_c0);

  vec_src = _mm512_sub_ps(vec_src, vec_res);
  // the tips is here, headache in perspective
  auto tmp = _mm512_fmadd_ps(vec_a, vec_src, vec_b);
  // headache bis - we loose precision with the cast but it "fits", but ok
  // after f32 -> f16 later
  __m512i casted_integer = _mm512_cvttps_epi32(tmp);
  // boundary condition, lower than the min -> 0
  casted_integer = _mm512_mask_mov_epi32(casted_integer, min_mask, vec_zero);
  // boundary condition, larger than the max -> +oo
  casted_integer =
      _mm512_mask_mov_epi32(casted_integer, max_mask, vec_infinity);
  // final interpretation to float
  return _mm512_castsi512_ps(casted_integer);
}
#endif

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
    Vectorized<T1> tmp2;
    if constexpr (std::is_same_v<T1, float> &&
                  (std::is_same_v<T2, at::BFloat16>)) {
      tmp2 = fexp_u20(tmp1);
    } else {
      tmp2 = exp_u20(tmp1);
    }
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
            zendnn_gemm<scalar_t>(
                qBlockSize, kvBlockSize, headSize, 1.0 /* alpha */,
                q_data + i * qStrideB + j * qStrideH + m * qStrideM, qStrideM,
                k_data + i * kStrideB + j * kStrideH + n * kStrideN, kStrideN,
                0.0 /* beta */, qk_data, kvBlockSize, false /* transA */,
                true /* transB */);
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
            zendnn_gemm<scalar_t>(
                qBlockSize, headSize, kvBlockSize, 1.0 /* alpha */,
                conditional_data_ptr(qk_data, qk_reduced_data), kvBlockSize,
                v_data + i * vStrideB + j * vStrideH + n * vStrideN, vStrideN,
                n == 0 ? 0.0 : 1.0 /* beta */, dst_data, headSize,
                false /* transA */, false /* transB */);
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

template <typename input_type, typename attention_mask>
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
    cpu_flash_attention<input_type, attention_mask, 256 /* q_split_size */,
                        512 /* kv_split_size */>(output, logsumexp, query, key,
                                                 value, dropout_p, is_causal,
                                                 attn_mask, scale);
  } else if (q_seq_len >= 192) {
    cpu_flash_attention<input_type, attention_mask, 64 /* q_split_size */,
                        512 /* kv_split_size */>(output, logsumexp, query, key,
                                                 value, dropout_p, is_causal,
                                                 attn_mask, scale);
  } else {
    cpu_flash_attention<input_type, attention_mask, 32 /* q_split_size */,
                        512 /* kv_split_size */>(output, logsumexp, query, key,
                                                 value, dropout_p, is_causal,
                                                 attn_mask, scale);
  }
}

template void flash_attention_kernel_impl_512<float, float>(
    const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor &, const at::Tensor &, double, bool,
    std::optional<at::Tensor>, std::optional<double>);
template void flash_attention_kernel_impl_512<at::BFloat16, at::BFloat16>(
    const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor &, const at::Tensor &, double, bool,
    std::optional<at::Tensor>, std::optional<double>);
template void flash_attention_kernel_impl_512<at::BFloat16, float>(
    const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor &, const at::Tensor &, double, bool,
    std::optional<at::Tensor>, std::optional<double>);
} // namespace zentorch
#endif // TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR > 3
