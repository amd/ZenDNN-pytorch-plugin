/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "../../../Utils.hpp"

#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <torch/all.h>

#include <immintrin.h>

#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>

namespace zentorch {
namespace {

constexpr int32_t kNullBlockId = 0;
constexpr float kSoftplusThreshold = 20.0f;
constexpr float kL2NormEps = 1e-6f;

inline float softplus_kernel_form(float x) {
  return x <= kSoftplusThreshold ? std::log1p(std::exp(x)) : x;
}

// ---------------------------------------------------------------------------
// AVX-512 helpers
// ---------------------------------------------------------------------------

// bf16 → fp32: zero-extend each u16 to u32 then shift left by 16.
inline __m512 cvt_bf16_to_fp32_512(__m256i src) {
  __m512i y = _mm512_cvtepu16_epi32(src);
  return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

// fp32 → bf16: truncate the low 16 bits of each fp32.
inline __m256i trunc_fp32_to_bf16_512(__m512 src) {
  __m512i y = _mm512_bsrli_epi128(_mm512_castps_si512(src), 2);
  return _mm512_cvtepi32_epi16(y);
}

template <typename T> inline float scalar_to_fp32(T x) {
  return static_cast<float>(x);
}

template <typename T> inline T fp32_to_scalar(float x);
template <> inline float fp32_to_scalar<float>(float x) { return x; }
template <> inline c10::BFloat16 fp32_to_scalar<c10::BFloat16>(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  uint16_t out = static_cast<uint16_t>(bits >> 16);
  c10::BFloat16 r;
  std::memcpy(&r, &out, sizeof(r));
  return r;
}

template <typename T> inline __m512 load_fp32_v16(const T *src);
template <> inline __m512 load_fp32_v16<float>(const float *src) {
  return _mm512_loadu_ps(src);
}
template <>
inline __m512 load_fp32_v16<c10::BFloat16>(const c10::BFloat16 *src) {
  return cvt_bf16_to_fp32_512(
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src)));
}

template <typename T> inline void store_fp32_v16(T *dst, __m512 v);
template <> inline void store_fp32_v16<float>(float *dst, __m512 v) {
  _mm512_storeu_ps(dst, v);
}
template <>
inline void store_fp32_v16<c10::BFloat16>(c10::BFloat16 *dst, __m512 v) {
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst),
                      trunc_fp32_to_bf16_512(v));
}

// y = (a - y) * scale.
inline void subtract_and_scale_fp32(float *__restrict y,
                                    const float *__restrict a, float scale,
                                    int64_t len) {
  const __m512 vs = _mm512_set1_ps(scale);
  int64_t i = 0;
  for (; i + 16 <= len; i += 16) {
    __m512 av = _mm512_loadu_ps(a + i);
    __m512 yv = _mm512_loadu_ps(y + i);
    _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_sub_ps(av, yv), vs));
  }
  for (; i < len; ++i)
    y[i] = (a[i] - y[i]) * scale;
}

// y *= alpha.
inline void scale_inplace_fp32(float *__restrict y, float alpha, int64_t len) {
  const __m512 va = _mm512_set1_ps(alpha);
  int64_t i = 0;
  for (; i + 16 <= len; i += 16) {
    _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_loadu_ps(y + i), va));
  }
  for (; i < len; ++i)
    y[i] *= alpha;
}

inline float dot_fp32(const float *__restrict a, const float *__restrict b,
                      int64_t len) {
  __m512 acc = _mm512_setzero_ps();
  int64_t i = 0;
  for (; i + 16 <= len; i += 16) {
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
  }
  float result = _mm512_reduce_add_ps(acc);
  for (; i < len; ++i)
    result += a[i] * b[i];
  return result;
}

template <typename T>
inline void load_row_to_fp32(float *__restrict dst, const T *__restrict src,
                             int64_t len) {
  int64_t i = 0;
  for (; i + 16 <= len; i += 16) {
    _mm512_storeu_ps(dst + i, load_fp32_v16<T>(src + i));
  }
  for (; i < len; ++i)
    dst[i] = scalar_to_fp32<T>(src[i]);
}

template <typename T>
inline void store_row_from_fp32(T *__restrict dst, const float *__restrict src,
                                int64_t len) {
  int64_t i = 0;
  for (; i + 16 <= len; i += 16) {
    store_fp32_v16<T>(dst + i, _mm512_loadu_ps(src + i));
  }
  for (; i < len; ++i)
    dst[i] = fp32_to_scalar<T>(src[i]);
}

// ---------------------------------------------------------------------------
// Fused-pass kernels
// ---------------------------------------------------------------------------

// Load state row, scale by exp_g into workspace, simultaneously dot with k_ws.
template <typename state_t>
inline float load_scale_dot(float *__restrict state_ws,
                            const state_t *__restrict state_cell, float exp_g,
                            const float *__restrict k_ws, int64_t K) {
  const __m512 vexp = _mm512_set1_ps(exp_g);
  __m512 acc = _mm512_setzero_ps();
  int64_t k = 0;
  for (; k + 16 <= K; k += 16) {
    __m512 src = load_fp32_v16<state_t>(state_cell + k);
    __m512 scaled = _mm512_mul_ps(src, vexp);
    _mm512_storeu_ps(state_ws + k, scaled);
    acc = _mm512_fmadd_ps(scaled, _mm512_loadu_ps(k_ws + k), acc);
  }
  float result = _mm512_reduce_add_ps(acc);
  for (; k < K; ++k) {
    float scaled = scalar_to_fp32<state_t>(state_cell[k]) * exp_g;
    state_ws[k] = scaled;
    result += scaled * k_ws[k];
  }
  return result;
}

// Fused per-row sweep: state += v_corr * k, out = state @ q, writeback to
// state_cell. The fp32 workspace row is not written back (dead after this).
template <typename state_t>
inline float update_state_and_compute_out(
    float *__restrict state_ws, state_t *__restrict state_cell, float v_corr,
    const float *__restrict k_ws, const float *__restrict q_ws, int64_t K) {
  const __m512 vc = _mm512_set1_ps(v_corr);
  __m512 acc = _mm512_setzero_ps();
  int64_t k = 0;
  for (; k + 16 <= K; k += 16) {
    __m512 sv = _mm512_loadu_ps(state_ws + k);
    __m512 kv = _mm512_loadu_ps(k_ws + k);
    sv = _mm512_fmadd_ps(vc, kv, sv);
    acc = _mm512_fmadd_ps(sv, _mm512_loadu_ps(q_ws + k), acc);
    store_fp32_v16<state_t>(state_cell + k, sv);
  }
  float result = _mm512_reduce_add_ps(acc);
  for (; k < K; ++k) {
    float sv = state_ws[k] + v_corr * k_ws[k];
    result += sv * q_ws[k];
    state_cell[k] = fp32_to_scalar<state_t>(sv);
  }
  return result;
}

// Per-(b, hv) recurrence step. All math is fp32; reads upcast at load,
// writes downcast at store.
template <typename model_t, typename state_t>
inline void
compute_one_pair(int64_t hv, int64_t H, int64_t V, int64_t K, int64_t r,
                 const model_t *mixed_qkv_b, const model_t *a_b,
                 const model_t *b_b, const float *A_log, const float *dt_bias,
                 float scale, bool use_qk_l2norm, state_t *state_cell,
                 model_t *out_cell, float *state_ws, float *q_ws, float *k_ws,
                 float *v_ws, float *v_corr_ws) {
  const int64_t i_h = hv / r;

  load_row_to_fp32<model_t>(q_ws, mixed_qkv_b + i_h * K, K);
  load_row_to_fp32<model_t>(k_ws, mixed_qkv_b + H * K + i_h * K, K);
  load_row_to_fp32<model_t>(v_ws, mixed_qkv_b + 2 * H * K + hv * V, V);

  if (use_qk_l2norm) {
    const float q_sq = dot_fp32(q_ws, q_ws, K);
    const float k_sq = dot_fp32(k_ws, k_ws, K);
    scale_inplace_fp32(q_ws, 1.0f / std::sqrt(q_sq + kL2NormEps), K);
    scale_inplace_fp32(k_ws, 1.0f / std::sqrt(k_sq + kL2NormEps), K);
  }
  scale_inplace_fp32(q_ws, scale, K);

  // Scalar gate (one value per (b, hv)).
  const float a_val = static_cast<float>(a_b[hv]);
  const float b_val = static_cast<float>(b_b[hv]);
  const float x = a_val + dt_bias[hv];
  const float sp = softplus_kernel_form(x);
  const float g_val = -std::exp(A_log[hv]) * sp;
  const float beta_val = 1.0f / (1.0f + std::exp(-b_val));
  const float exp_g = std::exp(g_val);

  // Pass A: load state with decay AND compute v_corr = state @ k.
  for (int64_t v = 0; v < V; ++v) {
    v_corr_ws[v] = load_scale_dot<state_t>(state_ws + v * K, state_cell + v * K,
                                           exp_g, k_ws, K);
  }
  subtract_and_scale_fp32(v_corr_ws, v_ws, beta_val, V);

  // Pass B: state += v_corr * k, out = state @ q, writeback.
  // Stage fp32 outputs into v_ws (dead after Pass A); cast in bulk at the end.
  for (int64_t v = 0; v < V; ++v) {
    v_ws[v] = update_state_and_compute_out<state_t>(
        state_ws + v * K, state_cell + v * K, v_corr_ws[v], k_ws, q_ws, K);
  }
  store_row_from_fp32<model_t>(out_cell, v_ws, V);
}

template <typename model_t, typename state_t>
void run_dispatched(const at::Tensor &mixed_qkv, const at::Tensor &a,
                    const at::Tensor &b, const at::Tensor &A_log,
                    const at::Tensor &dt_bias, double scale,
                    at::Tensor &initial_state, at::Tensor &out,
                    const at::Tensor &ssm_state_indices,
                    bool use_qk_l2norm_in_kernel, int64_t B, int64_t HV,
                    int64_t V, int64_t K, int64_t H, int64_t r) {
  const int64_t mqkv_stride0 = mixed_qkv.stride(0);
  const int64_t a_stride0 = a.stride(0);
  const int64_t b_stride0 = b.stride(0);
  const int64_t out_stride0 = out.stride(0);
  const int64_t out_stride2 = out.stride(2);
  // vLLM swaps stride(0)/stride(1) of initial_state via as_strided_; only the
  // inner (V, K) slab is required contiguous (asserted in the public op).
  const int64_t state_slot_stride = initial_state.stride(0);
  const int64_t state_cell_stride = initial_state.stride(1);

  const model_t *mixed_qkv_base = mixed_qkv.const_data_ptr<model_t>();
  const model_t *a_base = a.const_data_ptr<model_t>();
  const model_t *b_base = b.const_data_ptr<model_t>();
  const float *A_log_p = A_log.const_data_ptr<float>();
  const float *dt_bias_p = dt_bias.const_data_ptr<float>();
  const int32_t *slot_p = ssm_state_indices.const_data_ptr<int32_t>();
  state_t *state_base = initial_state.data_ptr<state_t>();
  model_t *out_base = out.data_ptr<model_t>();
  const float scale_f = static_cast<float>(scale);

  const int64_t total_pairs = B * HV;

  at::parallel_for(
      0, total_pairs, /*grain_size=*/1, [&](int64_t start, int64_t end) {
        std::vector<float> state_ws(static_cast<size_t>(V) * K);
        std::vector<float> q_ws(K);
        std::vector<float> k_ws(K);
        std::vector<float> v_ws(V);
        std::vector<float> v_corr_ws(V);

        for (int64_t idx = start; idx < end; ++idx) {
          const int64_t b_idx = idx / HV;
          const int64_t hv = idx % HV;
          const int32_t slot = slot_p[b_idx];

          model_t *out_cell = out_base + b_idx * out_stride0 + hv * out_stride2;

          if (slot <= kNullBlockId) {
            // Skip: zero the output cell, leave state untouched.
            const model_t zero = static_cast<model_t>(0.0f);
            for (int64_t v = 0; v < V; ++v) {
              out_cell[v] = zero;
            }
            continue;
          }

          const model_t *mixed_qkv_b = mixed_qkv_base + b_idx * mqkv_stride0;
          const model_t *a_b = a_base + b_idx * a_stride0;
          const model_t *b_b = b_base + b_idx * b_stride0;
          state_t *state_cell =
              state_base + slot * state_slot_stride + hv * state_cell_stride;

          compute_one_pair<model_t, state_t>(
              hv, H, V, K, r, mixed_qkv_b, a_b, b_b, A_log_p, dt_bias_p,
              scale_f, use_qk_l2norm_in_kernel, state_cell, out_cell,
              state_ws.data(), q_ws.data(), k_ws.data(), v_ws.data(),
              v_corr_ws.data());
        }
      });
}

void dispatch_dtypes(const at::Tensor &mixed_qkv, const at::Tensor &a,
                     const at::Tensor &b, const at::Tensor &A_log,
                     const at::Tensor &dt_bias, double scale,
                     at::Tensor &initial_state, at::Tensor &out,
                     const at::Tensor &ssm_state_indices,
                     bool use_qk_l2norm_in_kernel, int64_t B, int64_t HV,
                     int64_t V, int64_t K, int64_t H, int64_t r) {
  const auto model_dt = mixed_qkv.scalar_type();
  const auto state_dt = initial_state.scalar_type();

#define ZENTORCH_GDN_RUN(model_t, state_t)                                     \
  run_dispatched<model_t, state_t>(mixed_qkv, a, b, A_log, dt_bias, scale,     \
                                   initial_state, out, ssm_state_indices,      \
                                   use_qk_l2norm_in_kernel, B, HV, V, K, H, r)

#define ZENTORCH_GDN_DISPATCH_STATE(model_t)                                   \
  do {                                                                         \
    if (state_dt == c10::ScalarType::Float) {                                  \
      ZENTORCH_GDN_RUN(model_t, float);                                        \
    } else if (state_dt == c10::ScalarType::BFloat16) {                        \
      ZENTORCH_GDN_RUN(model_t, c10::BFloat16);                                \
    } else if (state_dt == c10::ScalarType::Half) {                            \
      ZENTORCH_CHECK(false,                                                    \
                     "fp16 (initial_state) not supported; use fp32 or bf16");  \
    } else {                                                                   \
      ZENTORCH_CHECK(false, "initial_state dtype must be fp32 or bf16; got ",  \
                     state_dt);                                                \
    }                                                                          \
  } while (0)

  if (model_dt == c10::ScalarType::Float) {
    ZENTORCH_GDN_DISPATCH_STATE(float);
  } else if (model_dt == c10::ScalarType::BFloat16) {
    ZENTORCH_GDN_DISPATCH_STATE(c10::BFloat16);
  } else if (model_dt == c10::ScalarType::Half) {
    ZENTORCH_CHECK(false, "fp16 not supported; use fp32 or bf16");
  } else {
    ZENTORCH_CHECK(false, "mixed_qkv dtype must be fp32 or bf16; got ",
                   model_dt);
  }

#undef ZENTORCH_GDN_DISPATCH_STATE
#undef ZENTORCH_GDN_RUN
}

} // anonymous namespace

void zentorch_gdn_fused_recurrent_gated_delta_rule_packed_decode(
    const at::Tensor &mixed_qkv, const at::Tensor &a, const at::Tensor &b,
    const at::Tensor &A_log, const at::Tensor &dt_bias, double scale,
    at::Tensor &initial_state, at::Tensor &out,
    const at::Tensor &ssm_state_indices, bool use_qk_l2norm_in_kernel,
    std::string zentorch_op_name) {
  RECORD_FUNCTION(
      "zentorch::gdn_fused_recurrent_gated_delta_rule_packed_decode",
      c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(mixed_qkv.dim() == 2, "mixed_qkv must be 2-D (B, qkv_dim)");
  ZENTORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2-D (B, HV)");
  ZENTORCH_CHECK(A_log.dim() == 1 && dt_bias.dim() == 1,
                 "A_log and dt_bias must be 1-D (HV,)");
  ZENTORCH_CHECK(ssm_state_indices.dim() == 1,
                 "ssm_state_indices must be 1-D (B,)");
  ZENTORCH_CHECK(initial_state.dim() == 4,
                 "initial_state must be 4-D (num_cache_lines, HV, V, K)");
  ZENTORCH_CHECK(out.dim() == 4, "out must be 4-D (B, 1, HV, V)");

  const int64_t B = mixed_qkv.size(0);
  const int64_t HV = initial_state.size(1);
  const int64_t V = initial_state.size(2);
  const int64_t K = initial_state.size(3);

  ZENTORCH_CHECK(a.size(0) == B && b.size(0) == B, "Batch mismatch on a/b");
  ZENTORCH_CHECK(a.size(1) == HV && b.size(1) == HV,
                 "a/b last dim must equal HV=", HV);
  ZENTORCH_CHECK(A_log.numel() == HV && dt_bias.numel() == HV,
                 "A_log/dt_bias must have HV=", HV, " elements");
  ZENTORCH_CHECK(ssm_state_indices.size(0) == B,
                 "ssm_state_indices length must equal B=", B);
  ZENTORCH_CHECK(out.size(0) == B && out.size(1) == 1 && out.size(2) == HV &&
                     out.size(3) == V,
                 "out must be (B, 1, HV, V)");

  const int64_t qkv_dim = mixed_qkv.size(1);
  const int64_t qk_dim = qkv_dim - HV * V;
  ZENTORCH_CHECK(qk_dim > 0 && qk_dim % 2 == 0,
                 "Invalid packed qkv_dim=", qkv_dim);
  const int64_t q_dim = qk_dim / 2;
  ZENTORCH_CHECK(q_dim % K == 0, "q_dim must be divisible by K (q_dim=", q_dim,
                 ", K=", K, ")");
  const int64_t H = q_dim / K;
  ZENTORCH_CHECK(H > 0 && HV % H == 0, "Invalid head config");
  const int64_t r = HV / H;

  ZENTORCH_CHECK(at::isFloatingType(A_log.scalar_type()),
                 "A_log must be floating-point");
  ZENTORCH_CHECK(at::isFloatingType(dt_bias.scalar_type()),
                 "dt_bias must be floating-point");
  ZENTORCH_CHECK(ssm_state_indices.scalar_type() == c10::ScalarType::Int,
                 "ssm_state_indices must be int32");

  ZENTORCH_CHECK(a.scalar_type() == mixed_qkv.scalar_type() &&
                     b.scalar_type() == mixed_qkv.scalar_type() &&
                     out.scalar_type() == mixed_qkv.scalar_type(),
                 "mixed_qkv, a, b, out must share dtype");

  ZENTORCH_CHECK(mixed_qkv.stride(-1) == 1,
                 "mixed_qkv last-dim must be unit-stride");
  ZENTORCH_CHECK(a.stride(-1) == 1, "a last-dim must be unit-stride");
  ZENTORCH_CHECK(b.stride(-1) == 1, "b last-dim must be unit-stride");
  ZENTORCH_CHECK(out.stride(-1) == 1, "out last-dim must be unit-stride");
  ZENTORCH_CHECK(A_log.is_contiguous(), "A_log must be contiguous");
  ZENTORCH_CHECK(dt_bias.is_contiguous(), "dt_bias must be contiguous");
  ZENTORCH_CHECK(ssm_state_indices.is_contiguous(),
                 "ssm_state_indices must be contiguous");
  ZENTORCH_CHECK(initial_state.stride(3) == 1,
                 "initial_state last dim (K) must be unit-stride");
  ZENTORCH_CHECK(initial_state.stride(2) == K,
                 "initial_state (V, K) slab must be contiguous");

  if (B == 0) {
    return;
  }

  const at::Tensor A_log_f = A_log.to(c10::kFloat);
  const at::Tensor dt_bias_f = dt_bias.to(c10::kFloat);

  dispatch_dtypes(mixed_qkv, a, b, A_log_f, dt_bias_f, scale, initial_state,
                  out, ssm_state_indices, use_qk_l2norm_in_kernel, B, HV, V, K,
                  H, r);
}

} // namespace zentorch
