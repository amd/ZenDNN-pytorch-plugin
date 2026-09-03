/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "FusedFFN.hpp"
#include "GroupMatmul.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <vector>

namespace zentorch {

// ---------------------------------------------------------------------------
// zentorch_fused_ffn_concat — fused single-expert (non-MoE) FFN (out variant)
//
// Replaces the `linear -> slice -> gated activation -> mul -> linear` FFN
// chain (SwiGLU / GeGLU) with a single grouped-matmul call:
//
//     output = w2 @ (activation(w13[:I] @ x) * (w13[I:] @ x))
//
// `output` is mutated in place (Tensor(a!)); the op returns (). The W13 gate|up
// weights arrive pre-concatenated as `w13_weight` [2*I, H]; `w2_weight` is the
// down projection [H, I]. w13/w2 bias and int8 scales are optional.
//
// The whole chain is handed to `zentorch_group_matmul_out_impl` as a
// single-expert group: every operand is wrapped in a length-1 list, and
// `gemm_outputs = [output]` receives the W2 down-projection result. A non-empty
// `gemm_outputs` paired with a fused `w2_weights` entry makes ZenDNN write the
// final result into `output` (aliasing the caller's buffer); `moe_output` /
// `topk_weights` / `row_ptrs` stay empty (no MoE weighted reduce). Weight
// dtype selects the kernel (bf16/f32, DA8W8, or DA8W4) inside GroupMatmul.
// ---------------------------------------------------------------------------
void zentorch_fused_ffn_concat_out_impl(
    torch::stable::Tensor &output, const torch::stable::Tensor &input,
    const torch::stable::Tensor &w13_weight,
    const torch::stable::Tensor &w2_weight,
    const std::optional<torch::stable::Tensor> &w13_bias,
    const std::optional<torch::stable::Tensor> &w2_bias,
    std::string_view activation,
    const std::optional<torch::stable::Tensor> &w13_scale,
    const std::optional<torch::stable::Tensor> &w2_scale,
    std::string zentorch_op_name) {

  const torch::stable::Tensor input_2d =
      torch::stable::view(input, get_2d_size_for_tensor(input));
  torch::stable::Tensor output_2d =
      torch::stable::view(output, get_2d_size_for_tensor(output));

  std::vector<torch::stable::Tensor> gemm_outputs = {output_2d};
  const std::vector<torch::stable::Tensor> inputs = {input_2d};
  const std::vector<torch::stable::Tensor> w13_weights = {w13_weight};
  const std::vector<std::optional<torch::stable::Tensor>> w2_weights = {
      w2_weight};
  const std::vector<std::optional<torch::stable::Tensor>> w13_biases = {
      w13_bias};
  const std::vector<std::optional<torch::stable::Tensor>> w2_biases = {w2_bias};
  const std::vector<std::optional<torch::stable::Tensor>> w13_scales = {
      w13_scale};
  const std::vector<std::optional<torch::stable::Tensor>> w2_scales = {
      w2_scale};

  zentorch_group_matmul_out_impl(
      /*gemm_outputs=*/gemm_outputs,
      /*inputs=*/inputs,
      /*w13_weights=*/w13_weights,
      /*w2_weights=*/w2_weights,
      /*moe_output=*/std::nullopt,
      /*topk_weights=*/std::nullopt,
      /*row_ptrs=*/std::nullopt,
      /*activation=*/activation,
      /*w13_bias=*/w13_biases,
      /*w2_bias=*/w2_biases,
      /*w13_scales=*/w13_scales,
      /*w2_scales=*/w2_scales,
      /*zentorch_op_name=*/zentorch_op_name);
}

// ---------------------------------------------------------------------------
// Op registration
// ---------------------------------------------------------------------------

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_fused_ffn_concat.out(Tensor(a!) output, Tensor input, "
        "Tensor w13_weight, Tensor w2_weight, "
        "Tensor? w13_bias=None, Tensor? w2_bias=None, "
        "str activation='silu', "
        "Tensor? w13_scale=None, Tensor? w2_scale=None, *, "
        "str zentorch_op_name='zentorch::fused_ffn_concat') -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_fused_ffn_concat.out",
         TORCH_BOX(&zentorch::zentorch_fused_ffn_concat_out_impl));
}

} // namespace zentorch
