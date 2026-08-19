/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "../../../Utils.hpp"

#include <ATen/EmptyTensor.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/record_function.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/Optional.h>
#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

namespace zentorch {

namespace {

inline at::Tensor empty_contiguous_cpu(at::IntArrayRef sizes,
                                       const at::TensorOptions &options) {
  std::vector<int64_t> strides(sizes.size());
  int64_t s = 1;
  for (int64_t i = static_cast<int64_t>(sizes.size()) - 1; i >= 0; --i) {
    strides[i] = s;
    s *= sizes[i];
  }
  return at::detail::empty_strided_cpu(sizes, strides, options);
}

using FVec = at::vec::Vectorized<float>;

// Returns sum_i a[i] * b[i].
inline float vec_dot(const float *a, const float *b, int64_t n) {
  FVec acc(0.0f);
  const int64_t vlen = FVec::size();
  int64_t i = 0;
  for (; i + vlen <= n; i += vlen)
    acc = acc + FVec::loadu(a + i) * FVec::loadu(b + i);
  alignas(64) float buf[FVec::size()];
  acc.store(buf);
  float s = 0.0f;
  for (int64_t l = 0; l < vlen; ++l)
    s += buf[l];
  for (; i < n; ++i)
    s += a[i] * b[i];
  return s;
}

// y[i] = scale * x[i].
inline void vec_scale(float *y, const float *x, float scale, int64_t n) {
  const FVec vs(scale);
  const int64_t vlen = FVec::size();
  int64_t i = 0;
  for (; i + vlen <= n; i += vlen)
    (FVec::loadu(x + i) * vs).store(y + i);
  for (; i < n; ++i)
    y[i] = scale * x[i];
}

// y[i] -= a * x[i].
inline void vec_axpy_neg(float *y, const float *x, float a, int64_t n) {
  const FVec va(a);
  const int64_t vlen = FVec::size();
  int64_t i = 0;
  for (; i + vlen <= n; i += vlen)
    (FVec::loadu(y + i) - va * FVec::loadu(x + i)).store(y + i);
  for (; i < n; ++i)
    y[i] -= a * x[i];
}

// y[i] += a * x[i].
inline void vec_axpy_pos(float *y, const float *x, float a, int64_t n) {
  const FVec va(a);
  const int64_t vlen = FVec::size();
  int64_t i = 0;
  for (; i + vlen <= n; i += vlen)
    (FVec::loadu(y + i) + va * FVec::loadu(x + i)).store(y + i);
  for (; i < n; ++i)
    y[i] += a * x[i];
}

// y[i] *= a.
inline void vec_scale_inplace(float *y, float a, int64_t n) {
  const FVec va(a);
  const int64_t vlen = FVec::size();
  int64_t i = 0;
  for (; i + vlen <= n; i += vlen)
    (FVec::loadu(y + i) * va).store(y + i);
  for (; i < n; ++i)
    y[i] *= a;
}

template <typename g_t>
void run_g_cumsum(const at::Tensor &g, const at::Tensor &cu_seqlens,
                  const at::Tensor &chunk_indices, int64_t BT,
                  at::Tensor &g_cum, int64_t HV) {
  const int64_t NT = chunk_indices.size(0);
  const int64_t g_stride_t = g.stride(1);
  const int64_t g_stride_h = g.stride(2);
  const int64_t out_stride_t = g_cum.stride(1);
  const int64_t out_stride_h = g_cum.stride(2);

  const g_t *g_base = g.const_data_ptr<g_t>();
  float *out_base = g_cum.data_ptr<float>();
  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  const int32_t *chunk_indices_p = chunk_indices.const_data_ptr<int32_t>();

  at::parallel_for(0, NT, /*grain_size=*/1, [&](int64_t start, int64_t end) {
    for (int64_t r = start; r < end; ++r) {
      const int32_t seq_idx = chunk_indices_p[2 * r + 0];
      const int32_t chunk_idx = chunk_indices_p[2 * r + 1];
      const int64_t bos = cu_seqlens_p[seq_idx];
      const int64_t eos = cu_seqlens_p[seq_idx + 1];
      const int64_t cs_start = bos + chunk_idx * BT;
      const int64_t cs_end = std::min(cs_start + BT, eos);
      if (cs_start >= cs_end)
        continue;

      const g_t *g_seq_base = g_base + bos * g_stride_t;
      float *out_seq_base = out_base + bos * out_stride_t;
      const int64_t t0 = cs_start - bos;
      const int64_t t1 = cs_end - bos;

      for (int64_t h = 0; h < HV; ++h) {
        const g_t *g_ptr = g_seq_base + h * g_stride_h;
        float *out_ptr = out_seq_base + h * out_stride_h;
        float acc = 0.0f;
        for (int64_t t = t0; t < t1; ++t) {
          acc += static_cast<float>(g_ptr[t * g_stride_t]);
          out_ptr[t * out_stride_t] = acc;
        }
      }
    }
  });
}

void run_recompute_w_u_fused(const at::Tensor &k_f, const at::Tensor &v_f,
                             const at::Tensor &beta_f, const at::Tensor &g_cum,
                             const at::Tensor &cu_seqlens,
                             const at::Tensor &chunk_indices, int64_t BT,
                             int64_t H, int64_t r, at::Tensor &w_f,
                             at::Tensor &u_f) {
  const int64_t NT = chunk_indices.size(0);
  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  const int32_t *ci_p = chunk_indices.const_data_ptr<int32_t>();

  const int64_t K_dim = k_f.size(3);
  const int64_t V_dim = v_f.size(3);

  const float *k_base = k_f.const_data_ptr<float>();       // [1, T, Hg, K]
  const float *v_base = v_f.const_data_ptr<float>();       // [1, T, H,  V]
  const float *beta_base = beta_f.const_data_ptr<float>(); // [1, T, H]
  const float *g_base = g_cum.const_data_ptr<float>();     // [1, T, H]
  float *w_base = w_f.data_ptr<float>();                   // [1, T, H, K]
  float *u_base = u_f.data_ptr<float>();                   // [1, T, H, V]

  const int64_t k_st = k_f.stride(1), k_sh = k_f.stride(2);
  const int64_t v_st = v_f.stride(1), v_sh = v_f.stride(2);
  const int64_t b_st = beta_f.stride(1), b_sh = beta_f.stride(2);
  const int64_t g_st = g_cum.stride(1), g_sh = g_cum.stride(2);
  const int64_t w_st = w_f.stride(1), w_sh = w_f.stride(2);
  const int64_t u_st = u_f.stride(1), u_sh = u_f.stride(2);

  at::parallel_for(
      0, NT * H, /*grain_size=*/1, [&](int64_t begin, int64_t end) {
        for (int64_t unit = begin; unit < end; ++unit) {
          const int64_t row = unit / H;
          const int64_t h = unit % H;
          const int64_t kh = h / r;
          const int32_t seq_idx = ci_p[2 * row + 0];
          const int32_t chunk_idx = ci_p[2 * row + 1];
          const int64_t bos = cu_seqlens_p[seq_idx];
          const int64_t eos = cu_seqlens_p[seq_idx + 1];
          const int64_t cs_start = bos + chunk_idx * BT;
          const int64_t cs_end = std::min(cs_start + BT, eos);
          const int64_t BT_eff = cs_end - cs_start;
          if (BT_eff <= 0)
            continue;

          for (int64_t i = 0; i < BT_eff; ++i) {
            const int64_t ti = cs_start + i;
            const float beta_i = beta_base[ti * b_st + h * b_sh];
            const float g_i = g_base[ti * g_st + h * g_sh];
            const float *k_i = k_base + ti * k_st + kh * k_sh;
            const float *v_i = v_base + ti * v_st + h * v_sh;
            float *u_i = u_base + ti * u_st + h * u_sh;
            float *w_i = w_base + ti * w_st + h * w_sh;

            // RHS: u_i = β·v_i ; w_i = β·exp(g_i)·k_i.
            const float exp_gi = std::exp(g_i);
            vec_scale(u_i, v_i, beta_i, V_dim);
            vec_scale(w_i, k_i, beta_i * exp_gi, K_dim);

            // Subtract strictly-lower contributions:
            //   L[i,j] = β_i · exp(g_i − g_j) · (k_i · k_j),  j < i.
            for (int64_t j = 0; j < i; ++j) {
              const int64_t tj = cs_start + j;
              const float g_j = g_base[tj * g_st + h * g_sh];
              const float *k_j = k_base + tj * k_st + kh * k_sh;
              const float L_ij =
                  beta_i * std::exp(g_i - g_j) * vec_dot(k_i, k_j, K_dim);
              vec_axpy_neg(u_i, u_base + tj * u_st + h * u_sh, L_ij, V_dim);
              vec_axpy_neg(w_i, w_base + tj * w_st + h * w_sh, L_ij, K_dim);
            }
          }
        }
      });
}

void run_chunk_recurrent_state(const at::Tensor &k_f, const at::Tensor &w_f,
                               const at::Tensor &u_f, const at::Tensor &g_cum,
                               const c10::optional<at::Tensor> &initial_state_f,
                               const at::Tensor &cu_seqlens,
                               const at::Tensor &chunk_offsets_long, int64_t BT,
                               int64_t H, int64_t r, int64_t V_dim,
                               int64_t K_dim, bool output_final_state,
                               at::Tensor &h_out_f, at::Tensor &v_new_f,
                               at::Tensor &final_state) {
  const int64_t N = cu_seqlens.size(0) - 1;
  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  const int64_t *co_p = chunk_offsets_long.const_data_ptr<int64_t>();

  const float *k_base = k_f.const_data_ptr<float>();   // [1, T, Hg, K]
  const float *w_base = w_f.const_data_ptr<float>();   // [1, T, H,  K]
  const float *u_base = u_f.const_data_ptr<float>();   // [1, T, H,  V]
  const float *g_base = g_cum.const_data_ptr<float>(); // [1, T, H]
  float *hout_base = h_out_f.data_ptr<float>();        // [1, NT_total, H, V, K]
  float *vnew_base = v_new_f.data_ptr<float>();        // [1, T, H, V]
  float *fs_base =
      output_final_state ? final_state.data_ptr<float>() : nullptr; // [N,H,V,K]
  const bool has_init =
      initial_state_f.has_value() && initial_state_f->numel() > 0;
  const float *is_base = has_init ? initial_state_f->const_data_ptr<float>()
                                  : nullptr; // [N,H,V,K]

  const int64_t k_st = k_f.stride(1), k_sh = k_f.stride(2);
  const int64_t w_st = w_f.stride(1), w_sh = w_f.stride(2);
  const int64_t u_st = u_f.stride(1), u_sh = u_f.stride(2);
  const int64_t g_st = g_cum.stride(1), g_sh = g_cum.stride(2);
  const int64_t ho_sc = h_out_f.stride(1), ho_sh = h_out_f.stride(2);
  const int64_t vn_st = v_new_f.stride(1), vn_sh = v_new_f.stride(2);
  const int64_t is_sn = has_init ? initial_state_f->stride(0) : 0;
  const int64_t is_sh = has_init ? initial_state_f->stride(1) : 0;
  const int64_t fs_sn = fs_base ? final_state.stride(0) : 0;
  const int64_t fs_sh = fs_base ? final_state.stride(1) : 0;
  const int64_t VK = V_dim * K_dim;

  // TODO(perf): same pattern as run_chunk_output -- the state/vcorr scratch
  // buffers are allocated inside the parallel_for lambda, so they are
  // re-allocated once per task chunk (~once per thread on the native/OpenMP
  // backends; potentially many more under TBB, where grain_size=1 lets the
  // scheduler over-decompose). These are sizeable (state is V_dim*K_dim,
  // vcorr is BT*V_dim -- tens of KB each), so this adds avoidable malloc/free
  // traffic and allocator contention in the hot region. Prefer per-thread
  // reusable scratch (e.g. thread_local vectors resized on demand) and/or a
  // tuned grain_size. Deferred: perf-only change (numerically identical,
  // buffers are fully overwritten before use) but should be re-benchmarked
  // before landing.
  at::parallel_for(0, N * H, /*grain_size=*/1, [&](int64_t begin, int64_t end) {
    std::vector<float> state(VK);
    std::vector<float> vcorr(BT * V_dim);
    for (int64_t unit = begin; unit < end; ++unit) {
      const int64_t n = unit / H;
      const int64_t h = unit % H;
      const int64_t kh = h / r;
      const int64_t bos = cu_seqlens_p[n];
      const int64_t eos = cu_seqlens_p[n + 1];
      const int64_t boh = co_p[n];
      const int64_t chunks_in_seq = co_p[n + 1] - boh;

      if (is_base) {
        std::memcpy(state.data(), is_base + n * is_sn + h * is_sh,
                    sizeof(float) * VK);
      } else {
        std::fill(state.begin(), state.end(), 0.0f);
      }

      for (int64_t i_t = 0; i_t < chunks_in_seq; ++i_t) {
        const int64_t chunk_start = bos + i_t * BT;
        const int64_t chunk_end = std::min(chunk_start + BT, eos);
        const int64_t BT_eff = chunk_end - chunk_start;
        if (BT_eff <= 0)
          continue;

        // Snapshot pre-update state into h_out[boh + i_t, h].
        std::memcpy(hout_base + (boh + i_t) * ho_sc + h * ho_sh, state.data(),
                    sizeof(float) * VK);

        const float g_last =
            g_base[(chunk_start + BT_eff - 1) * g_st + h * g_sh];

        // v_corr[t, v] = u[t, v] - sum_k w[t, k] * state[v, k]
        for (int64_t t = 0; t < BT_eff; ++t) {
          const float *w_t = w_base + (chunk_start + t) * w_st + h * w_sh;
          const float *u_t = u_base + (chunk_start + t) * u_st + h * u_sh;
          float *vc_t = vcorr.data() + t * V_dim;
          for (int64_t v = 0; v < V_dim; ++v)
            vc_t[v] = u_t[v] - vec_dot(w_t, state.data() + v * K_dim, K_dim);
        }

        // Save pre-decay v_new[chunk_start + t, h] for the output stage.
        for (int64_t t = 0; t < BT_eff; ++t) {
          std::memcpy(vnew_base + (chunk_start + t) * vn_st + h * vn_sh,
                      vcorr.data() + t * V_dim, sizeof(float) * V_dim);
        }

        // Per-token decay: v_corr[t, v] *= exp(g_last - g[t]).
        for (int64_t t = 0; t < BT_eff; ++t) {
          const float g_t = g_base[(chunk_start + t) * g_st + h * g_sh];
          vec_scale_inplace(vcorr.data() + t * V_dim, std::exp(g_last - g_t),
                            V_dim);
        }

        // Bulk decay: state *= exp(g_last).
        vec_scale_inplace(state.data(), std::exp(g_last), VK);

        // state[v, k] += sum_t v_corr[t, v] * k[t, k].
        for (int64_t t = 0; t < BT_eff; ++t) {
          const float *k_t = k_base + (chunk_start + t) * k_st + kh * k_sh;
          const float *vc_t = vcorr.data() + t * V_dim;
          for (int64_t v = 0; v < V_dim; ++v)
            vec_axpy_pos(state.data() + v * K_dim, k_t, vc_t[v], K_dim);
        }
      }

      if (fs_base) {
        std::memcpy(fs_base + n * fs_sn + h * fs_sh, state.data(),
                    sizeof(float) * VK);
      }
    }
  });
}

// Stores one fp32 value into `o` (contiguous [1, T, H, V]) at byte-typed `base`
// with element index `idx`, casting to the output dtype.
inline void store_out(void *base, c10::ScalarType out_dtype, int64_t idx,
                      float val) {
  switch (out_dtype) {
  case c10::ScalarType::Float:
    static_cast<float *>(base)[idx] = val;
    break;
  case c10::ScalarType::BFloat16:
    static_cast<c10::BFloat16 *>(base)[idx] = static_cast<c10::BFloat16>(val);
    break;
  case c10::ScalarType::Half:
    static_cast<c10::Half *>(base)[idx] = static_cast<c10::Half>(val);
    break;
  default:
    ZENTORCH_CHECK(false, "unsupported output dtype in gdn chunk output");
  }
}

void run_chunk_output(const at::Tensor &q_f, const at::Tensor &k_f,
                      const at::Tensor &v_new_f, const at::Tensor &h_out_f,
                      const at::Tensor &g_cum, const at::Tensor &cu_seqlens,
                      const at::Tensor &chunk_indices,
                      const at::Tensor &chunk_offsets_long, int64_t BT,
                      int64_t H, int64_t r, float scale_f, at::Tensor &o,
                      c10::ScalarType out_dtype) {
  const int64_t NT = chunk_indices.size(0);
  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  const int32_t *ci_p = chunk_indices.const_data_ptr<int32_t>();
  const int64_t *co_p = chunk_offsets_long.const_data_ptr<int64_t>();
  // NT == chunk_offsets[-1] is validated at the entry point before any kernel
  // runs (fail-fast), so it is not re-checked here.

  const float *q_base = q_f.const_data_ptr<float>();      // [1, T, Hg, K]
  const float *k_base = k_f.const_data_ptr<float>();      // [1, T, Hg, K]
  const float *vn_base = v_new_f.const_data_ptr<float>(); // [1, T, H,  V]
  const float *ho_base = h_out_f.const_data_ptr<float>(); // [1, NT_total,H,V,K]
  const float *g_base = g_cum.const_data_ptr<float>();    // [1, T, H]
  void *o_base = o.data_ptr();                            // [1, T, H, V]

  const int64_t K_dim = q_f.size(3);
  const int64_t V_dim = v_new_f.size(3);

  const int64_t q_st = q_f.stride(1), q_sh = q_f.stride(2);
  const int64_t k_st = k_f.stride(1), k_sh = k_f.stride(2);
  const int64_t vn_st = v_new_f.stride(1), vn_sh = v_new_f.stride(2);
  const int64_t ho_sc = h_out_f.stride(1), ho_sh = h_out_f.stride(2);
  const int64_t g_st = g_cum.stride(1), g_sh = g_cum.stride(2);
  const int64_t o_st = o.stride(1), o_sh = o.stride(2);

  // TODO(perf): the A/expg/oacc scratch buffers are allocated inside the
  // parallel_for lambda, so they are re-allocated once per task chunk (~once
  // per thread for the native/OpenMP backends; potentially many more under the
  // TBB backend, where grain_size=1 lets the scheduler over-decompose). A is
  // BT*BT floats (~16KB at BT=64), so this adds avoidable malloc/free traffic
  // and allocator contention in the hot region. Prefer per-thread reusable
  // scratch (e.g. thread_local vectors resized on demand) and/or a tuned
  // grain_size. Deferred: this is a perf-only change (numerically identical,
  // buffers are fully overwritten before use) but should be re-benchmarked
  // before landing.
  at::parallel_for(
      0, NT * H, /*grain_size=*/1, [&](int64_t begin, int64_t end) {
        std::vector<float> A(BT * BT);
        std::vector<float> expg(BT);
        std::vector<float> oacc(V_dim);
        for (int64_t unit = begin; unit < end; ++unit) {
          const int64_t row = unit / H;
          const int64_t h = unit % H;
          const int64_t kh = h / r;
          const int64_t seq_idx = ci_p[2 * row + 0];
          const int64_t chunk_idx = ci_p[2 * row + 1];
          const int64_t bos = cu_seqlens_p[seq_idx];
          const int64_t eos = cu_seqlens_p[seq_idx + 1];
          const int64_t chunk_start = bos + chunk_idx * BT;
          const int64_t chunk_end = std::min(chunk_start + BT, eos);
          const int64_t BT_eff = chunk_end - chunk_start;
          if (BT_eff <= 0)
            continue;
          const int64_t boh = co_p[seq_idx];
          const float *h_chunk =
              ho_base + (boh + chunk_idx) * ho_sc + h * ho_sh; // [V, K]

          for (int64_t t = 0; t < BT_eff; ++t)
            expg[t] = std::exp(g_base[(chunk_start + t) * g_st + h * g_sh]);

          // A[t, s] = (sum_k q[t, k] * k[s, k]) * exp(g[t] - g[s]) for s <= t
          // else 0.
          for (int64_t t = 0; t < BT_eff; ++t) {
            const float *q_t = q_base + (chunk_start + t) * q_st + kh * q_sh;
            const float gt = g_base[(chunk_start + t) * g_st + h * g_sh];
            float *A_t = A.data() + t * BT_eff;
            for (int64_t s = 0; s <= t; ++s) {
              const float *k_s = k_base + (chunk_start + s) * k_st + kh * k_sh;
              const float gs = g_base[(chunk_start + s) * g_st + h * g_sh];
              A_t[s] = vec_dot(q_t, k_s, K_dim) * std::exp(gt - gs);
            }
          }

          // o[t, v] = ((sum_k q[t,k]*h_chunk[v,k]) * expg[t]
          //            + sum_{s<=t} A[t,s]*v_new[s,v]) * scale
          for (int64_t t = 0; t < BT_eff; ++t) {
            const float *q_t = q_base + (chunk_start + t) * q_st + kh * q_sh;
            // History contribution.
            const float expg_t = expg[t];
            for (int64_t v = 0; v < V_dim; ++v)
              oacc[v] = vec_dot(q_t, h_chunk + v * K_dim, K_dim) * expg_t;
            // In-chunk contribution.
            const float *A_t = A.data() + t * BT_eff;
            for (int64_t s = 0; s <= t; ++s)
              vec_axpy_pos(oacc.data(),
                           vn_base + (chunk_start + s) * vn_st + h * vn_sh,
                           A_t[s], V_dim);
            const int64_t o_off = (chunk_start + t) * o_st + h * o_sh;
            for (int64_t v = 0; v < V_dim; ++v)
              store_out(o_base, out_dtype, o_off + v, oacc[v] * scale_f);
          }
        }
      });
}

} // namespace

std::tuple<at::Tensor, at::Tensor> zentorch_gdn_chunk_gated_delta_rule_fwd(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &g, const at::Tensor &beta, double scale,
    const c10::optional<at::Tensor> &initial_state, bool output_final_state,
    int64_t chunk_size, const at::Tensor &cu_seqlens,
    const at::Tensor &chunk_indices, const at::Tensor &chunk_offsets,
    std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::gdn_chunk_gated_delta_rule_fwd",
                  c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                 "q/k/v must be 4-D");
  ZENTORCH_CHECK(g.dim() == 3 && beta.dim() == 3,
                 "g and beta must be 3-D (B, T, H)");
  ZENTORCH_CHECK(q.size(0) == 1, "B must be 1 (varlen); got ", q.size(0));

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t Hg = q.size(2);
  const int64_t K_dim = q.size(3);
  const int64_t H = v.size(2);
  const int64_t V_dim = v.size(3);
  const int64_t BT = chunk_size;

  ZENTORCH_CHECK(k.size(0) == B && k.size(1) == T && k.size(2) == Hg &&
                     k.size(3) == K_dim,
                 "k must match q on (B, T, Hg, K)");
  ZENTORCH_CHECK(v.size(0) == B && v.size(1) == T,
                 "v must agree with q on (B, T)");
  ZENTORCH_CHECK(H % Hg == 0, "H must be a multiple of Hg");
  const int64_t r = H / Hg;
  ZENTORCH_CHECK(BT == 16 || BT == 32 || BT == 64,
                 "chunk_size must be one of {16, 32, 64}; got ", BT);
  ZENTORCH_CHECK(g.sizes() == beta.sizes(),
                 "g and beta must have the same shape");
  ZENTORCH_CHECK(g.size(0) == B && g.size(1) == T && g.size(2) == H,
                 "g must be (B, T, H)");

  ZENTORCH_CHECK(is_supported_gdn_float(q.scalar_type()),
                 "q must be fp16, bf16, or fp32; got ", q.scalar_type());
  ZENTORCH_CHECK(k.scalar_type() == q.scalar_type() &&
                     v.scalar_type() == q.scalar_type(),
                 "q, k, v must share dtype");
  ZENTORCH_CHECK(is_supported_gdn_float(g.scalar_type()),
                 "g must be fp16, bf16, or fp32; got ", g.scalar_type());
  ZENTORCH_CHECK(is_supported_gdn_float(beta.scalar_type()),
                 "beta must be fp16, bf16, or fp32; got ", beta.scalar_type());

  ZENTORCH_CHECK(cu_seqlens.dim() == 1 &&
                     cu_seqlens.scalar_type() == c10::ScalarType::Int,
                 "cu_seqlens must be 1-D int32");
  ZENTORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");
  ZENTORCH_CHECK(cu_seqlens.size(0) >= 2,
                 "cu_seqlens must have at least 2 entries");
  ZENTORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
                     chunk_indices.scalar_type() == c10::ScalarType::Int,
                 "chunk_indices must be 2-D int32 (NT, 2)");
  ZENTORCH_CHECK(chunk_indices.is_contiguous(),
                 "chunk_indices must be contiguous");
  ZENTORCH_CHECK(chunk_offsets.dim() == 1 &&
                     (chunk_offsets.scalar_type() == c10::ScalarType::Int ||
                      chunk_offsets.scalar_type() == c10::ScalarType::Long),
                 "chunk_offsets must be 1-D int32 or int64");
  ZENTORCH_CHECK(chunk_offsets.size(0) == cu_seqlens.size(0),
                 "chunk_offsets.size(0)=", chunk_offsets.size(0),
                 " must equal cu_seqlens.size(0)=", cu_seqlens.size(0));

  const int64_t N = cu_seqlens.size(0) - 1;
  const int64_t NT = chunk_indices.size(0);

  if (initial_state.has_value() && initial_state->numel() > 0) {
    ZENTORCH_CHECK(initial_state->dim() == 4,
                   "initial_state must be 4-D (N, H, V, K)");
    ZENTORCH_CHECK(initial_state->size(0) == N && initial_state->size(1) == H &&
                       initial_state->size(2) == V_dim &&
                       initial_state->size(3) == K_dim,
                   "initial_state shape must be (N, H, V, K)");
    ZENTORCH_CHECK(is_supported_gdn_float(initial_state->scalar_type()),
                   "initial_state must be fp16, bf16, or fp32; got ",
                   initial_state->scalar_type());
  }

  const auto fp32_options = k.options().dtype(c10::kFloat);
  at::Tensor o = empty_contiguous_cpu({B, T, H, V_dim}, v.options());
  at::Tensor final_state =
      output_final_state
          ? empty_contiguous_cpu({N, H, V_dim, K_dim}, fp32_options)
          : at::detail::empty_strided_cpu({0}, {1}, fp32_options);

  if (NT == 0 || H == 0 || T == 0) {
    o.zero_();
    if (output_final_state) {
      if (initial_state.has_value() && initial_state->numel() > 0) {
        final_state.copy_(initial_state.value());
      } else {
        final_state.zero_();
      }
    }
    return std::make_tuple(o, final_state);
  }

  const auto g_fp32_options = g.options().dtype(c10::kFloat);
  const auto v_fp32_options = v.options().dtype(c10::kFloat);
  at::Tensor g_cum = empty_contiguous_cpu({B, T, H}, g_fp32_options);
  at::Tensor w_f = empty_contiguous_cpu({B, T, H, K_dim}, fp32_options);
  at::Tensor u_f = empty_contiguous_cpu({B, T, H, V_dim}, v_fp32_options);

  at::Tensor chunk_offsets_long = chunk_offsets.to(at::kLong).contiguous();
  const int64_t *co_p = chunk_offsets_long.const_data_ptr<int64_t>();
  const int64_t NT_total = co_p[N];

  // Fail fast BEFORE the compute kernels run: chunk_indices (NT rows) must
  // match chunk_offsets[-1] (NT_total). NT_total sizes h_out_f and drives the
  // per-chunk iteration in run_chunk_recurrent_state; a mismatch would let the
  // kernels read uninitialized g_cum/w_f/u_f for the missing chunks (UB /
  // incorrect output) before any later validation could catch it.
  ZENTORCH_CHECK(NT == NT_total, "chunk_indices.size(0)=", NT,
                 " must equal chunk_offsets[-1]=", NT_total);

  at::Tensor h_out_f =
      empty_contiguous_cpu({B, NT_total, H, V_dim, K_dim}, fp32_options);
  at::Tensor v_new_f = empty_contiguous_cpu({B, T, H, V_dim}, v_fp32_options);

  const auto g_dt = g.scalar_type();
  if (g_dt == c10::ScalarType::Float) {
    run_g_cumsum<float>(g, cu_seqlens, chunk_indices, BT, g_cum, H);
  } else if (g_dt == c10::ScalarType::BFloat16) {
    run_g_cumsum<c10::BFloat16>(g, cu_seqlens, chunk_indices, BT, g_cum, H);
  } else if (g_dt == c10::ScalarType::Half) {
    run_g_cumsum<c10::Half>(g, cu_seqlens, chunk_indices, BT, g_cum, H);
  } else {
    // Defensive fallback: g is already constrained to fp16/bf16/fp32 by the
    // is_supported_gdn_float check above, so this arm is unreachable in
    // practice; kept to fail loudly if a new dtype is dispatched here.
    ZENTORCH_CHECK(false, "g must be fp16, bf16, or fp32; got ", g_dt);
  }

  at::Tensor k_f = k.to(c10::kFloat).contiguous();
  at::Tensor v_f = v.to(c10::kFloat).contiguous();
  at::Tensor beta_f = beta.to(c10::kFloat).contiguous();
  at::Tensor q_f = q.to(c10::kFloat).contiguous();
  c10::optional<at::Tensor> initial_state_f;
  if (initial_state.has_value() && initial_state->numel() > 0) {
    // Force contiguity: run_chunk_recurrent_state memcpy's VK = V_dim*K_dim
    // floats as one contiguous block (strides are only tracked for the outer
    // N/H dims), so a non-contiguous initial_state would be read with the wrong
    // layout and corrupt the recurrence. contiguous() is a no-op when already
    // contiguous; the tensor is read-only here, so a copy is harmless.
    initial_state_f = (initial_state->scalar_type() == c10::ScalarType::Float)
                          ? initial_state->contiguous()
                          : initial_state->to(c10::kFloat).contiguous();
  }

  run_recompute_w_u_fused(k_f, v_f, beta_f, g_cum, cu_seqlens, chunk_indices,
                          BT, H, r, w_f, u_f);

  run_chunk_recurrent_state(k_f, w_f, u_f, g_cum, initial_state_f, cu_seqlens,
                            chunk_offsets_long, BT, H, r, V_dim, K_dim,
                            output_final_state, h_out_f, v_new_f, final_state);

  const float scale_f = static_cast<float>(scale);
  run_chunk_output(q_f, k_f, v_new_f, h_out_f, g_cum, cu_seqlens, chunk_indices,
                   chunk_offsets_long, BT, H, r, scale_f, o, v.scalar_type());

  return std::make_tuple(o, final_state);
}

} // namespace zentorch
