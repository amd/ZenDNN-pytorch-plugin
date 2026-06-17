/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "../../../Utils.hpp"

#include <ATen/ops/eye.h>
#include <ATen/ops/linalg_solve_triangular.h>
#include <ATen/ops/tril.h>
#include <ATen/record_function.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace zentorch {

at::Tensor zentorch_gdn_solve_tril(const at::Tensor &A,
                                   const at::Tensor &cu_seqlens,
                                   const at::Tensor &chunk_indices,
                                   std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::gdn_solve_tril", c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(A.dim() == 4,
                 "A must be 4-D (B, T, H, BT); got dim=", A.dim());
  ZENTORCH_CHECK(A.size(0) == 1, "B must be 1 (varlen); got ", A.size(0));
  const int64_t T = A.size(1);
  const int64_t H = A.size(2);
  const int64_t BT = A.size(3);
  ZENTORCH_CHECK(BT == 16 || BT == 32 || BT == 64,
                 "BT must be one of {16, 32, 64}; got ", BT);

  ZENTORCH_CHECK(at::isFloatingType(A.scalar_type()),
                 "A must be floating-point; got ", A.scalar_type());

  ZENTORCH_CHECK(cu_seqlens.dim() == 1 &&
                     cu_seqlens.scalar_type() == c10::ScalarType::Int,
                 "cu_seqlens must be 1-D int32");
  ZENTORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
                     chunk_indices.scalar_type() == c10::ScalarType::Int,
                 "chunk_indices must be 2-D int32 (NT, 2)");
  ZENTORCH_CHECK(cu_seqlens.is_contiguous() && chunk_indices.is_contiguous(),
                 "cu_seqlens and chunk_indices must be contiguous");

  const int64_t NT = chunk_indices.size(0);

  at::Tensor Ai = at::zeros_like(A, A.options().dtype(c10::kFloat));

  if (NT == 0 || H == 0 || T == 0)
    return Ai;

  at::Tensor A_f = A.to(c10::kFloat);
  at::Tensor I_max = at::eye(BT, A.options().dtype(c10::kFloat));

  const int32_t *cu_seqlens_p = cu_seqlens.const_data_ptr<int32_t>();
  const int32_t *chunk_indices_p = chunk_indices.const_data_ptr<int32_t>();

  for (int64_t row = 0; row < NT; ++row) {
    const int32_t seq_idx = chunk_indices_p[2 * row + 0];
    const int32_t chunk_idx = chunk_indices_p[2 * row + 1];
    const int64_t bos = cu_seqlens_p[seq_idx];
    const int64_t eos = cu_seqlens_p[seq_idx + 1];
    const int64_t cs_start = bos + chunk_idx * BT;
    const int64_t cs_end = std::min(cs_start + BT, eos);
    const int64_t BT_eff = cs_end - cs_start;
    if (BT_eff <= 0)
      continue;

    at::Tensor I_eff = I_max.narrow(0, 0, BT_eff).narrow(1, 0, BT_eff);

    for (int64_t h = 0; h < H; ++h) {
      at::Tensor A_block = A_f.select(0, 0)
                               .narrow(0, cs_start, BT_eff)
                               .select(1, h)
                               .narrow(-1, 0, BT_eff);
      at::Tensor A_strict = at::tril(A_block, /*diagonal=*/-1);
      at::Tensor L = I_eff + A_strict;

      at::Tensor M = at::linalg_solve_triangular(
          L, I_eff, /*upper=*/false, /*left=*/true, /*unitriangular=*/true);

      Ai.select(0, 0)
          .narrow(0, cs_start, BT_eff)
          .select(1, h)
          .narrow(-1, 0, BT_eff)
          .copy_(M);
    }
  }

  return Ai;
}

} // namespace zentorch
