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

#include <algorithm>
#include <cstdint>

namespace zentorch {
namespace {

template <typename g_t>
inline void process_chunk(int64_t bos, int64_t cs_start, int64_t cs_end,
                          int64_t HV, int64_t g_stride_t, int64_t g_stride_h,
                          int64_t out_stride_t, int64_t out_stride_h,
                          const g_t *g_seq_base, float *out_seq_base) {
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

template <typename g_t>
void run_dispatched(const at::Tensor &g, const at::Tensor &cu_seqlens,
                    const at::Tensor &chunk_indices, int64_t chunk_size,
                    at::Tensor &out, int64_t HV, int64_t NT) {
  const int64_t g_stride_t = g.stride(1);
  const int64_t g_stride_h = g.stride(2);
  const int64_t out_stride_t = out.stride(1);
  const int64_t out_stride_h = out.stride(2);

  const g_t *g_base = g.const_data_ptr<g_t>();
  float *out_base = out.data_ptr<float>();
  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  const int32_t *chunk_indices_p = chunk_indices.const_data_ptr<int32_t>();

  at::parallel_for(0, NT, /*grain_size=*/1, [&](int64_t start, int64_t end) {
    for (int64_t r = start; r < end; ++r) {
      const int32_t seq_idx = chunk_indices_p[2 * r + 0];
      const int32_t chunk_idx = chunk_indices_p[2 * r + 1];
      const int64_t bos = cu_seqlens_p[seq_idx];
      const int64_t eos = cu_seqlens_p[seq_idx + 1];
      const int64_t cs_start = bos + chunk_idx * chunk_size;
      const int64_t cs_end = std::min(cs_start + chunk_size, eos);
      if (cs_start >= cs_end) {
        continue;
      }

      const g_t *g_seq_base = g_base + bos * g_stride_t;
      float *out_seq_base = out_base + bos * out_stride_t;
      process_chunk<g_t>(bos, cs_start, cs_end, HV, g_stride_t, g_stride_h,
                         out_stride_t, out_stride_h, g_seq_base, out_seq_base);
    }
  });
}

} // anonymous namespace

at::Tensor zentorch_gdn_chunk_local_cumsum(const at::Tensor &g,
                                           int64_t chunk_size,
                                           const at::Tensor &cu_seqlens,
                                           const at::Tensor &chunk_indices,
                                           std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::gdn_chunk_local_cumsum",
                  c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(g.dim() == 3,
                 "g must be 3-D (1, T_total, HV); got dim=", g.dim());
  ZENTORCH_CHECK(g.size(0) == 1, "g.size(0) must be 1; got ", g.size(0));
  const int64_t T_total = g.size(1);
  const int64_t HV = g.size(2);

  ZENTORCH_CHECK(cu_seqlens.dim() == 1, "cu_seqlens must be 1-D");
  ZENTORCH_CHECK(cu_seqlens.size(0) >= 2,
                 "cu_seqlens must have at least 2 entries");

  ZENTORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2,
                 "chunk_indices must be 2-D (NT, 2)");
  const int64_t NT = chunk_indices.size(0);

  ZENTORCH_CHECK(chunk_size > 0, "chunk_size must be positive; got ",
                 chunk_size);

  ZENTORCH_CHECK(at::isFloatingType(g.scalar_type()),
                 "g must be floating-point; got ", g.scalar_type());
  ZENTORCH_CHECK(cu_seqlens.scalar_type() == c10::ScalarType::Int,
                 "cu_seqlens must be int32");
  ZENTORCH_CHECK(chunk_indices.scalar_type() == c10::ScalarType::Int,
                 "chunk_indices must be int32");

  ZENTORCH_CHECK(g.stride(-1) == 1, "g last dim must be unit-stride");
  ZENTORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");
  ZENTORCH_CHECK(chunk_indices.is_contiguous(),
                 "chunk_indices must be contiguous");

  // Reject metadata mismatches that would otherwise return an
  // uninitialised fp32 buffer to the caller below.
  ZENTORCH_CHECK(NT > 0 || T_total == 0,
                 "chunk_indices must be non-empty when T_total > 0; got NT=0"
                 " with T_total=",
                 T_total);

  at::Tensor out = at::empty({1, T_total, HV}, g.options().dtype(c10::kFloat));

  if (T_total == 0 || HV == 0) {
    return out;
  }

  const auto g_dt = g.scalar_type();

#define ZENTORCH_GDN_RUN(g_t)                                                  \
  run_dispatched<g_t>(g, cu_seqlens, chunk_indices, chunk_size, out, HV, NT)

  if (g_dt == c10::ScalarType::Float) {
    ZENTORCH_GDN_RUN(float);
  } else if (g_dt == c10::ScalarType::BFloat16) {
    ZENTORCH_GDN_RUN(c10::BFloat16);
  } else if (g_dt == c10::ScalarType::Half) {
    ZENTORCH_CHECK(false, "fp16 not supported; use fp32 or bf16");
  } else {
    ZENTORCH_CHECK(false, "g dtype must be fp32 or bf16; got ", g_dt);
  }

#undef ZENTORCH_GDN_RUN

  return out;
}

} // namespace zentorch
