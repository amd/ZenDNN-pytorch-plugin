/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "Utils.hpp"
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <string>
#include <torch/csrc/stable/tensor.h>

namespace zentorch {

// Infers the weight-quant mode from the input/weight shapes and weight dtype:
// true for DA8W4 (s4 packed 2-per-int8 or 8-per-int32), false for DA8W8 (s8) or
// a non-int8/int32 weight. Expects a 2D [N, K-dim] weight; K-dim is the
// contraction dim, packed or not.
//
// Header-only because torch::stable has hidden visibility, which would make a
// definition in the .cpp a local symbol its cross-TU callers cannot resolve.
//
// Infer the weight mode from the packing (pack_factor = K / weight.size(1),
// where K is the input's last dim) and check the weight dtype for that mode.
// Returns true for DA8W4, false for DA8W8:
//   pack_factor 1 -> DA8W8 (s8,    [N, K])
//   pack_factor 2 -> DA8W4 (int8,  [N, K/2])
//   pack_factor 8 -> DA8W4 (int32, [N, K/8])
// weight.size(1) must divide K and yield one of the pack factors above, so an
// invalid layout errors here instead of being mis-dispatched to matmul_direct.
inline bool
check_weight_and_infer_is_da8w4(const torch::stable::Tensor &input,
                                const torch::stable::Tensor &weight) {
  const auto wdt = weight.scalar_type();
  if (wdt != c10::kChar && wdt != c10::kInt) {
    return false;
  }

  const int64_t K = input.size(input.dim() - 1);
  const int64_t wk = weight.size(1);
  ZENTORCH_CHECK(wk > 0 && K % wk == 0,
                 "zentorch_dynamic_qlinear: weight dim 1 (", wk,
                 ") must divide the input K (", K, ")");

  switch (K / wk) {
  case 1: // DA8W8
    ZENTORCH_CHECK(wdt == c10::kChar,
                   "zentorch_dynamic_qlinear: DA8W8 weight must be int8, got ",
                   wdt);
    return false;
  case 2: // DA8W4, 2 s4 per byte
    ZENTORCH_CHECK(
        wdt == c10::kChar,
        "zentorch_dynamic_qlinear: DA8W4 (K/2) weight must be int8, got ", wdt);
    return true;
  case 8: // DA8W4, 8 s4 per int32
    ZENTORCH_CHECK(
        wdt == c10::kInt,
        "zentorch_dynamic_qlinear: DA8W4 (K/8) weight must be int32, got ",
        wdt);
    return true;
  default:
    ZENTORCH_CHECK(false,
                   "zentorch_dynamic_qlinear: unsupported weight dim 1 (", wk,
                   ") for input K (", K, "); expected K, K/2, or K/8");
    return false; // unreachable
  }
}

// Dynamic-quantization linear op (DA8W8 / DA8W4).
torch::stable::Tensor zentorch_dynamic_qlinear(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string zentorch_op_name);

// Out variant: writes into the caller-provided `out` (last, kwarg-only per the
// aten out convention) and returns nothing.
void zentorch_dynamic_qlinear_out(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const std::optional<torch::stable::Tensor> &bias, bool is_weight_prepacked,
    std::string zentorch_op_name, torch::stable::Tensor &out);

} // namespace zentorch
