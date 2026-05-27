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

#include <cmath>
#include <cstdint>
#include <vector>

namespace zentorch {
namespace {

template <typename x_t>
inline void process_row(int64_t D, const x_t *x_row, x_t *out_row, float eps) {
  float sum_sq = 0.0f;
  for (int64_t k = 0; k < D; ++k) {
    const float v = static_cast<float>(x_row[k]);
    sum_sq += v * v;
  }
  const float inv_norm = 1.0f / std::sqrt(sum_sq + eps);
  for (int64_t k = 0; k < D; ++k) {
    out_row[k] = static_cast<x_t>(static_cast<float>(x_row[k]) * inv_norm);
  }
}

template <typename x_t>
void run_dispatched(const at::Tensor &x, at::Tensor &out, float eps, int64_t D,
                    int64_t num_rows) {
  const int64_t ndim = x.dim();
  const int64_t outer_ndim = ndim - 1;

  std::vector<int64_t> outer_sizes(outer_ndim);
  std::vector<int64_t> x_strides(outer_ndim);
  std::vector<int64_t> out_strides(outer_ndim);
  for (int64_t d = 0; d < outer_ndim; ++d) {
    outer_sizes[d] = x.size(d);
    x_strides[d] = x.stride(d);
    out_strides[d] = out.stride(d);
  }

  const x_t *x_base = x.const_data_ptr<x_t>();
  x_t *out_base = out.data_ptr<x_t>();

  at::parallel_for(0, num_rows, /*grain_size=*/16,
                   [&](int64_t start, int64_t end) {
                     for (int64_t r = start; r < end; ++r) {
                       int64_t x_off = 0;
                       int64_t out_off = 0;
                       int64_t rem = r;
                       for (int64_t d = outer_ndim - 1; d >= 0; --d) {
                         const int64_t i = rem % outer_sizes[d];
                         rem /= outer_sizes[d];
                         x_off += i * x_strides[d];
                         out_off += i * out_strides[d];
                       }

                       const x_t *x_row = x_base + x_off;
                       x_t *out_row = out_base + out_off;
                       process_row<x_t>(D, x_row, out_row, eps);
                     }
                   });
}

} // anonymous namespace

at::Tensor zentorch_gdn_l2norm_fwd(const at::Tensor &x, double eps,
                                   std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::gdn_l2norm_fwd", c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(x.dim() >= 1, "x must have at least one dim");
  const int64_t D = x.size(-1);
  ZENTORCH_CHECK(D >= 1, "x.size(-1) must be >= 1; got ", D);
  ZENTORCH_CHECK(at::isFloatingType(x.scalar_type()),
                 "x must be floating-point; got ", x.scalar_type());
  ZENTORCH_CHECK(x.stride(-1) == 1, "x last dim must be unit-stride");

  at::Tensor out = at::empty(x.sizes(), x.options());

  int64_t num_rows = 1;
  for (int64_t d = 0; d < x.dim() - 1; ++d) {
    num_rows *= x.size(d);
  }
  if (num_rows == 0 || D == 0) {
    return out;
  }

  const float eps_f = static_cast<float>(eps);
  const auto x_dt = x.scalar_type();

#define ZENTORCH_GDN_RUN(x_t) run_dispatched<x_t>(x, out, eps_f, D, num_rows)

  if (x_dt == c10::ScalarType::Float) {
    ZENTORCH_GDN_RUN(float);
  } else if (x_dt == c10::ScalarType::BFloat16) {
    ZENTORCH_GDN_RUN(c10::BFloat16);
  } else if (x_dt == c10::ScalarType::Half) {
    ZENTORCH_CHECK(false, "fp16 not supported; use fp32 or bf16");
  } else {
    ZENTORCH_CHECK(false, "x dtype must be fp32 or bf16; got ", x_dt);
  }

#undef ZENTORCH_GDN_RUN

  return out;
}

} // namespace zentorch
