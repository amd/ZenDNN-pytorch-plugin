/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "GroupMatmul.hpp"
#include "DynamicQLinear.hpp"
#include "EnvReader.hpp"
#include "MatmulUtils.hpp"
#include "Memory.hpp"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

using namespace zendnnl::interface;

namespace zentorch {

static bool has_tensor(const std::optional<torch::stable::Tensor> &opt) {
  return opt.has_value() && opt->defined();
}

// Validates per-expert input/weight/bias dtypes, shapes, and K-compatibility.
//
// Sizing contract:
//   - `inputs` and `bias` are sized to the active-expert count (E_a).
//   - `weights` is sized to the total expert count (E) with the active
//     experts placed at indices [0, E_a) so they line up with
//     `inputs[op_idx]` and `bias[op_idx]`; trailing entries [E_a, E) are
//     the inactive prepack-extras tail.
//
// We validate every weight (active prefix + inactive tail) since both will
// be reordered into the same per-expert weight cache by ZenDNN's prepack
// module — a malformed inactive weight would corrupt that cache and bite
// us when the expert later fires. Inputs and biases are validated only for
// the active prefix.

static bool checks_enabled() {
  return static_cast<bool>(
      EnvReader::getEnvVariableAsInt("ZENTORCH_ENABLE_CHECKS"));
}

// Shared validation for per-expert quantization scale lists.
// Checks list size, per-element presence, dtype (f32/bf16), and dim
// (1D/2D).
static void validate_weight_scales(
    const std::vector<std::optional<torch::stable::Tensor>> &scales,
    int num_ops, const char *param_name) {
  if (!checks_enabled())
    return;
  ZENTORCH_CHECK(scales.size() == static_cast<size_t>(num_ops),
                 "zentorch_group_matmul: ", param_name, ".size() (",
                 scales.size(), ") must equal num_ops (", num_ops,
                 ") when weights are int8");
  for (int op_idx = 0; op_idx < num_ops; ++op_idx) {
    ZENTORCH_CHECK(has_tensor(scales[op_idx]),
                   "zentorch_group_matmul: ", param_name, "[", op_idx,
                   "] is required when weight is int8");
    const auto &ws = *scales[op_idx];
    ZENTORCH_CHECK(ws.scalar_type() == c10::kFloat ||
                       ws.scalar_type() == c10::kBFloat16,
                   "zentorch_group_matmul: ", param_name, "[", op_idx,
                   "] must be float32 or bfloat16, got ", ws.scalar_type());
    ZENTORCH_CHECK(ws.dim() == 1 || ws.dim() == 2,
                   "zentorch_group_matmul: ", param_name, "[", op_idx,
                   "] must be 1D or 2D, got ", ws.dim(), "D");
  }
}

// DA8W4 needs true per-group scales: 2D [G, N] with G > 1 (G == 1 is the
// per-channel case) and G dividing the unpacked contraction dim K.
static void validate_da8w4_weight_scale(const torch::stable::Tensor &ws,
                                        int64_t N, int64_t unpacked_K,
                                        int op_idx, const char *param_name) {
  if (!checks_enabled())
    return;
  ZENTORCH_CHECK(ws.dim() == 2, "zentorch_group_matmul: DA8W4 ", param_name,
                 "[", op_idx, "] must be 2D per-group [G, N], got ", ws.dim(),
                 "D");
  const int64_t G = ws.size(0);
  ZENTORCH_CHECK(G > 1, "zentorch_group_matmul: DA8W4 ", param_name, "[",
                 op_idx,
                 "] must have more than one quantization group (got G=", G,
                 "); per-channel (single-group) scales are not supported");
  ZENTORCH_CHECK(ws.size(1) == N, "zentorch_group_matmul: DA8W4 ", param_name,
                 "[", op_idx, "] N (", ws.size(1),
                 ") must match the weight rows N (", N, ")");
  ZENTORCH_CHECK(unpacked_K % G == 0, "zentorch_group_matmul: DA8W4 ",
                 param_name, "[", op_idx, "] group count G (", G,
                 ") must divide the contraction dim K (", unpacked_K, ")");
}

// Validates per-expert input/weight/bias dtypes, shapes, K-compatibility,
// and weight_scales (required for int8 weights).
static void validate_dtypes_and_shapes(
    const std::vector<torch::stable::Tensor> &inputs,
    const std::vector<torch::stable::Tensor> &w13_weights,
    const std::vector<std::optional<torch::stable::Tensor>> &w13_bias,
    const std::vector<std::optional<torch::stable::Tensor>> &w13_scales) {

  ZENTORCH_CHECK(inputs.size() > 1,
                 "zentorch_group_matmul: sequential mode (inputs.size() == 1) "
                 "is not supported; only parallel mode (one input per expert) "
                 "is currently implemented");

  const int num_active = static_cast<int>(inputs.size());
  const int num_total = static_cast<int>(w13_weights.size());

  ZENTORCH_CHECK(inputs.size() == w13_bias.size(),
                 "zentorch_group_matmul: inputs.size() (", inputs.size(),
                 ") must equal w13_bias.size() (", w13_bias.size(),
                 ") (active-expert count)");
  ZENTORCH_CHECK(w13_weights.size() >= inputs.size(),
                 "zentorch_group_matmul: weights.size() (", w13_weights.size(),
                 ") must be >= inputs.size() (", inputs.size(),
                 "); the leading inputs.size() weights are the active "
                 "experts and any extra entries form the prepack-extras "
                 "tail consumed by ZenDNN's weight cache warmer");

  // Reference dtype/K from inputs[0] - all per-expert inputs share these
  // by construction (they're shape-[M_e, H] slices of the same hidden state
  // tensor), and every weight (active or inactive) must conform so the
  // prepack reorders into a uniform cache layout.
  const auto ref_dtype = inputs[0].scalar_type();
  const int64_t K_ref = inputs[0].size(1);
  const auto w13_ref_dtype = w13_weights[0].scalar_type();
  ZENTORCH_CHECK(ref_dtype == c10::kFloat || ref_dtype == c10::kBFloat16 ||
                     ref_dtype == c10::kHalf,
                 "zentorch_group_matmul: input[0] must be float32 or "
                 "bfloat16 or float16, got ",
                 ref_dtype);

  // Validate every weight (active + inactive prepack tail).
  for (int op_idx = 0; op_idx < num_total; ++op_idx) {
    const auto &w13_weight = w13_weights[op_idx];
    ZENTORCH_CHECK(w13_weight.dim() == 2, "zentorch_group_matmul: weight[",
                   op_idx, "] must be 2D, got ", w13_weight.dim(), "D");
    const bool w13_s4_packed =
        check_weight_and_infer_is_da8w4(inputs[0], w13_weight);
    ZENTORCH_CHECK(w13_weight.scalar_type() == ref_dtype ||
                       w13_weight.scalar_type() == c10::kChar || w13_s4_packed,
                   "zentorch_group_matmul: weight[", op_idx, "] dtype (",
                   w13_weight.scalar_type(), ") must match input[0] dtype (",
                   ref_dtype,
                   "), be int8, or be packed s4 (DA8W4: int32 [N,K/8] or int8 "
                   "[N,K/2])");
    ZENTORCH_CHECK(w13_weight.scalar_type() == w13_ref_dtype,
                   "zentorch_group_matmul: weight[", op_idx, "] dtype (",
                   w13_weight.scalar_type(), ") must match weight[0] dtype (",
                   w13_ref_dtype,
                   "); all experts must share one weight regime because the "
                   "op infers fp/DA8W8/DA8W4 from weight[0]");
    ZENTORCH_CHECK(w13_weight.is_contiguous(), "zentorch_group_matmul: weight[",
                   op_idx, "] must be contiguous");
    // A packed s4 weight carries K/2 or K/8 columns, so it is exempt from the
    // K match; its ldb comes from the activation's contraction dim instead.
    if (!w13_s4_packed) {
      ZENTORCH_CHECK(
          w13_weight.size(1) == K_ref, "zentorch_group_matmul: weight[", op_idx,
          "] K (", w13_weight.size(1), ") must match input[0] K (", K_ref, ")");
    }
  }

  // weight_scales are required for int8 (dynamic-A8W8) and DA8W4 weights.
  // DA8W8 (s8) is a full-width int8 weight; an int8 DA8W4 container (size(1) ==
  // K/2) is packed s4, not DA8W8, so exclude it here.
  const bool has_s4_weights =
      check_weight_and_infer_is_da8w4(inputs[0], w13_weights[0]);
  const bool has_s8_weights =
      w13_weights[0].scalar_type() == c10::kChar && !has_s4_weights;
  if (has_s8_weights || has_s4_weights) {
    validate_weight_scales(w13_scales, num_active, "weight_scales");
  }
  // Gate the per-expert DA8W4 scale checks once: validate_weight_scales above
  // is what guarantees the list is sized and populated, and it only runs with
  // checks enabled, so the loop must not index the list otherwise.
  const bool check_da8w4_scales = has_s4_weights && checks_enabled();

  // Validate active-only inputs, biases, and DA8W4 weight scales.
  for (int op_idx = 0; op_idx < num_active; ++op_idx) {
    const auto &input = inputs[op_idx];
    const auto &w13_weight = w13_weights[op_idx];

    ZENTORCH_CHECK(input.dim() == 2, "zentorch_group_matmul: input[", op_idx,
                   "] must be 2D, got ", input.dim(), "D");
    ZENTORCH_CHECK(input.scalar_type() == ref_dtype,
                   "zentorch_group_matmul: input[", op_idx, "] dtype (",
                   input.scalar_type(), ") must match input[0] dtype (",
                   ref_dtype, ")");

    // Bias must be 1D with size matching N (weight rows) and matching dtype.
    if (has_tensor(w13_bias[op_idx])) {
      ZENTORCH_CHECK(w13_bias[op_idx]->dim() == 1,
                     "zentorch_group_matmul: w13_bias[", op_idx,
                     "] must be 1D");
      ZENTORCH_CHECK(w13_bias[op_idx]->size(0) == w13_weight.size(0),
                     "zentorch_group_matmul: w13_bias[", op_idx, "] size (",
                     w13_bias[op_idx]->size(0), ") must match N (",
                     w13_weight.size(0), ")");
      ZENTORCH_CHECK(w13_bias[op_idx]->scalar_type() == ref_dtype,
                     "zentorch_group_matmul: w13_bias[", op_idx, "] dtype (",
                     w13_bias[op_idx]->scalar_type(),
                     ") must match input[0] "
                     "dtype (",
                     ref_dtype, ")");
    }

    // DA8W4 requires per-group [G, N] weight scales; K_ref is the shared
    // unpacked contraction dim for every W13 expert.
    if (check_da8w4_scales) {
      validate_da8w4_weight_scale(*w13_scales[op_idx], w13_weight.size(0),
                                  K_ref, op_idx, "weight_scales");
    }
  }

  if (has_s8_weights) {
    ZENTORCH_CHECK(ref_dtype == c10::kBFloat16,
                   "zentorch_group_matmul: input[0] must be bf16 when "
                   "weights are int8");
  }

  if (has_s4_weights) {
    // DA8W4 is bf16-only: the AOCL-DLP s4->s8 sym-quant kernel rejects f32
    // and fp16 activations. Fail fast here rather than deep in the backend
    ZENTORCH_CHECK(ref_dtype == c10::kBFloat16,
                   "zentorch_group_matmul: DA8W4 (int32-packed s4) weights "
                   "require a bfloat16 activation (the AOCL-DLP DA8W4 kernel "
                   "supports neither float32 nor float16), got input[0] dtype ",
                   ref_dtype);
  }

  // Validate fp16 hardware support in case of fp16 inputs
  if (ref_dtype == c10::kHalf) {
    ZENTORCH_CHECK(zendnn_fp16_device_check(),
                   "zentorch_group_matmul: fp16 path needs the cpu support "
                   "avx512fp16");
  }
}

// gemm_outputs holds one destination per active expert. Compare against
// inputs.size() (the active-expert count), not weights.size() (which is
// sized to E and includes trailing inactive-expert passthrough entries).
static void
validate_gemm_outputs(const std::vector<torch::stable::Tensor> &gemm_outputs,
                      const std::vector<torch::stable::Tensor> &inputs) {
  ZENTORCH_CHECK(gemm_outputs.size() == inputs.size(),
                 "zentorch_group_matmul: gemm_outputs.size() (",
                 gemm_outputs.size(), ") must equal inputs.size() (",
                 inputs.size(), ")");
}

// Validates fused w2 (down projection) list sizes, per-expert shapes, and
// dtypes. w2_input_dim = N/2 when gated activation is active, N otherwise.
//
// Sizing mirrors validate_dtypes_and_shapes:
//   - w2_weights : sized E (active prefix + inactive prepack tail), matches
//                  weights.size().
//   - w2_bias    : sized E_a (active only), matches inputs.size().
// We validate every w2 weight (the inactive tail also gets prepacked) and
// only the active w2 biases.
static void validate_w2_params(
    const std::vector<torch::stable::Tensor> &inputs,
    const std::vector<torch::stable::Tensor> &w13_weights,
    const std::vector<std::optional<torch::stable::Tensor>> &w2_weights,
    const std::vector<std::optional<torch::stable::Tensor>> &w2_bias,
    const std::vector<std::optional<torch::stable::Tensor>> &w2_scales,
    bool use_gated_act) {

  ZENTORCH_CHECK(
      w2_weights.size() == w13_weights.size(),
      "zentorch_group_matmul: w2_weights (", w2_weights.size(),
      ") must equal weights.size() (", w13_weights.size(),
      ") (full expert count, active prefix + inactive prepack tail)");
  ZENTORCH_CHECK(w2_bias.size() == inputs.size(),
                 "zentorch_group_matmul: w2_bias (", w2_bias.size(),
                 ") must equal inputs.size() (", inputs.size(),
                 ") (active-expert count)");

  const int num_active = static_cast<int>(inputs.size());
  const int num_total = static_cast<int>(w2_weights.size());
  const auto ref_dtype = inputs[0].scalar_type();

  ZENTORCH_CHECK(has_tensor(w2_weights[0]),
                 "zentorch_group_matmul: w2_weights[0] must not be None when "
                 "fused w2 is enabled");
  const auto w2_ref_dtype = w2_weights[0]->scalar_type();

  // Op2 inherits its quant/dtype config from Op1, so the two must share a
  // container.
  ZENTORCH_CHECK(w2_ref_dtype == w13_weights[0].scalar_type(),
                 "zentorch_group_matmul: w2 weight regime (dtype ",
                 w2_ref_dtype, ") must match the w13 weight regime (dtype ",
                 w13_weights[0].scalar_type(),
                 "); mixing dtypes across Op1/Op2 misconfigures the shared "
                 "quantization scheme");
  // w2 is therefore packed s4 exactly when w13 is. We deliberately do NOT
  // re-derive the packing from w2's own last dim. We assume that the pack
  // factor is the same for w13 and w2.
  const bool has_s4_w2 =
      check_weight_and_infer_is_da8w4(inputs[0], w13_weights[0]);
  const bool has_s8_w2 = w2_ref_dtype == c10::kChar && !has_s4_w2;

  if (has_s8_w2 || has_s4_w2) {
    validate_weight_scales(w2_scales, num_active, "w2_scales");
  }
  // Gate the per-expert DA8W4 scale checks once, as on the w13 side: the scale
  // list is only known to be sized and populated when the checks are enabled.
  const bool check_da8w4_w2_scales = has_s4_w2 && checks_enabled();

  // Validate every w2 weight (active + inactive prepack tail).
  for (int op_idx = 0; op_idx < num_total; ++op_idx) {
    // Each w2 weight must be a defined tensor (not None)
    ZENTORCH_CHECK(has_tensor(w2_weights[op_idx]),
                   "zentorch_group_matmul: w2_weights[", op_idx,
                   "] must not be None when fused w2 is enabled");
    // w2 receives the post-activation output: N/2 with gated act, N without.
    const int64_t w2_input_dim = use_gated_act ? w13_weights[op_idx].size(0) / 2
                                               : w13_weights[op_idx].size(0);

    const auto &w2_w = *w2_weights[op_idx];
    ZENTORCH_CHECK(w2_w.dim() == 2, "zentorch_group_matmul: w2_weights[",
                   op_idx, "] must be 2D, got ", w2_w.dim(), "D");
    ZENTORCH_CHECK(w2_w.is_contiguous(), "zentorch_group_matmul: w2_weights[",
                   op_idx, "] must be contiguous");
    const auto w2_dtype = w2_w.scalar_type();
    // has_s4_w2 applies to every expert: w2 weights are per-expert slices of
    // one stacked tensor, so they share w2_weights[0]'s dtype, and the regime
    // is decided by w13. As on the w13 side, a packed s4 w2 is exempt from the
    // K_in match; its ldb_down comes from the down-projection input dim.
    if (!has_s4_w2) {
      ZENTORCH_CHECK(w2_w.size(1) == w2_input_dim,
                     "zentorch_group_matmul: w2_weights[", op_idx, "] K_in (",
                     w2_w.size(1),
                     ") must match the down-projection input "
                     "dim (",
                     w2_input_dim, ")");
    }
    ZENTORCH_CHECK(
        w2_dtype == ref_dtype || w2_dtype == c10::kChar || has_s4_w2,
        "zentorch_group_matmul: w2_weights[", op_idx, "] dtype (", w2_dtype,
        ") must be ", ref_dtype,
        ", int8 (DA8W8), or packed s4 (DA8W4: int32 [N,K/8] or int8 [N,K/2])");
  }

  // Validate active-only w2 biases and DA8W4 w2 scales.
  for (int op_idx = 0; op_idx < num_active; ++op_idx) {
    const int64_t K_out = w2_weights[op_idx]->size(0);
    if (has_tensor(w2_bias[op_idx])) {
      ZENTORCH_CHECK(
          w2_bias[op_idx]->dim() == 1 && w2_bias[op_idx]->size(0) == K_out &&
              w2_bias[op_idx]->scalar_type() == ref_dtype,
          "zentorch_group_matmul: w2_bias[", op_idx, "] must be 1D with size ",
          K_out, " and dtype ", ref_dtype, ", got ", w2_bias[op_idx]->dim(),
          "D, size ", w2_bias[op_idx]->size(0), ", dtype ",
          w2_bias[op_idx]->scalar_type());
    }

    // DA8W4 down-projection scales must be per-group [G, K_out]; the unpacked
    // contraction dim is the down-projection input dim (N/2 gated, N
    // otherwise).
    if (check_da8w4_w2_scales) {
      const int64_t w2_input_dim = use_gated_act
                                       ? w13_weights[op_idx].size(0) / 2
                                       : w13_weights[op_idx].size(0);
      validate_da8w4_weight_scale(*w2_scales[op_idx], K_out, w2_input_dim,
                                  op_idx, "w2_scales");
    }
  }
}

// Validates MoE weighted-reduce: topk_weights, row_ptrs, moe_output must all be
// provided. Returns true if MoE is enabled.
static bool
validate_moe_params(const std::optional<torch::stable::Tensor> &topk_weights,
                    const std::optional<torch::stable::Tensor> &row_ptrs,
                    const std::optional<torch::stable::Tensor> &moe_output) {
  const bool use_moe = has_tensor(topk_weights);
  if (use_moe) {
    ZENTORCH_CHECK(has_tensor(row_ptrs) && has_tensor(moe_output),
                   "zentorch_group_matmul: when topk_weights is provided, "
                   "row_ptrs and moe_output must also be provided");
  }
  return use_moe;
}

static bool validate_all_inputs(
    const std::vector<torch::stable::Tensor> &inputs,
    const std::vector<torch::stable::Tensor> &w13_weights,
    const std::vector<std::optional<torch::stable::Tensor>> &w13_bias,
    const std::vector<std::optional<torch::stable::Tensor>> &w13_scales,
    const std::vector<torch::stable::Tensor> &gemm_outputs,
    const std::optional<torch::stable::Tensor> &topk_weights,
    const std::optional<torch::stable::Tensor> &row_ptrs,
    const std::optional<torch::stable::Tensor> &moe_output,
    const std::vector<std::optional<torch::stable::Tensor>> &w2_weights,
    const std::vector<std::optional<torch::stable::Tensor>> &w2_bias,
    const std::vector<std::optional<torch::stable::Tensor>> &w2_scales,
    bool use_gated_act) {

  validate_dtypes_and_shapes(inputs, w13_weights, w13_bias, w13_scales);
  if (!gemm_outputs.empty()) {
    validate_gemm_outputs(gemm_outputs, inputs);
  }
  if (!w2_weights.empty()) {
    validate_w2_params(inputs, w13_weights, w2_weights, w2_bias, w2_scales,
                       use_gated_act);
  }
  return validate_moe_params(topk_weights, row_ptrs, moe_output);
}

// Maps activation string to LowOHA gated activation enum.
// Supported: "none", "silu", "gelu", "gelu_tanh", "swigluoai".
// "gelu" and "gelu_tanh" both use gelu_and_mul; fused kernels apply
// tanh-approx GELU (matches vLLM MoEActivation.GELU_TANH / Gemma-4).
static zendnnl::lowoha::matmul::grp_matmul_gated_act_t
map_activation_to_gated_act(std::string_view activation) {
  using gated_act_t = zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
  if (activation.empty() || activation == "none")
    return gated_act_t::none;
  if (activation == "silu")
    return gated_act_t::silu_and_mul;
  if (activation == "gelu" || activation == "gelu_tanh")
    return gated_act_t::gelu_and_mul;
  if (activation == "swigluoai")
    return gated_act_t::swiglu_oai_mul;
  ZENTORCH_CHECK(false, "zentorch_group_matmul: unsupported activation '",
                 std::string(activation), "'");
  return gated_act_t::none;
}

void zentorch_group_matmul_out_impl(
    std::vector<torch::stable::Tensor> gemm_outputs,
    const std::vector<torch::stable::Tensor> &inputs,
    const std::vector<torch::stable::Tensor> &w13_weights,
    const std::vector<std::optional<torch::stable::Tensor>> &w2_weights,
    std::optional<torch::stable::Tensor> moe_output,
    const std::optional<torch::stable::Tensor> &topk_weights,
    const std::optional<torch::stable::Tensor> &row_ptrs,
    std::string_view activation,
    const std::vector<std::optional<torch::stable::Tensor>> &w13_bias,
    const std::vector<std::optional<torch::stable::Tensor>> &w2_bias,
    const std::vector<std::optional<torch::stable::Tensor>> &w13_scales,
    const std::vector<std::optional<torch::stable::Tensor>> &w2_scales,
    const std::string &zentorch_op_name) {

  const auto gated_act_type = map_activation_to_gated_act(activation);
  const bool use_gated_act =
      gated_act_type != zendnnl::lowoha::matmul::grp_matmul_gated_act_t::none;

  const bool use_moe = validate_all_inputs(
      inputs, w13_weights, w13_bias, w13_scales, gemm_outputs, topk_weights,
      row_ptrs, moe_output, w2_weights, w2_bias, w2_scales, use_gated_act);

  // Two distinct sizes drive the per-op vectors below:
  //
  //   num_active = inputs.size()  : firing experts the dispatcher actually
  //                                 computes a GEMM for. Drives input-side
  //                                 vectors (M, src, lda, alpha, beta, layout,
  //                                 transA, bias, dst, ldc, params).
  //   num_total  = weights.size() : full expert pool whose weights are
  //                                 visible to ZenDNN's prepack module.
  //                                 Drives weight-side metadata vectors
  //                                 (weight, K, N, ldb, transB,
  //                                 is_weights_const) so the cache warmer
  //                                 sees every advertised expert.
  //
  // The active experts occupy the leading [0, num_active) entries of every
  // weight-side vector (matching inputs[op_idx] / bias[op_idx] one-to-one);
  // the inactive prepack tail occupies [num_active, num_total). The
  // dispatcher learns this layout via params[0].active_matmul /
  // total_matmul, set just before the call below.
  const int num_active = static_cast<int>(inputs.size());
  const int num_total = static_cast<int>(w13_weights.size());

  // Input-side vectors (sized to the active count).
  const std::vector<char> layouts(num_active, 'r');
  const std::vector<bool> transA_vec(num_active, false);
  std::vector<int> M_vec(num_active);
  const std::vector<float> alpha_vec(num_active, 1.0f);
  const std::vector<float> beta_vec(num_active, 0.0f);
  std::vector<const void *> src_ptrs(num_active);
  std::vector<int> lda_vec(num_active);
  std::vector<const void *> bias_ptrs(num_active, nullptr);
  std::vector<void *> dst_ptrs(num_active, nullptr);
  std::vector<int> ldc_vec(num_active);
  // Holds src_scale tensors for dynamic DA8W8 (keeps them alive until kernel
  // returns)
  std::vector<torch::stable::Tensor> temp_src_scales;
  // DA8W4: packed s4 weights (int32 [N,K/8] or int8 [N,K/2]) + dynamic
  // per-token s8 activation quant. Shares the dynamic-quant wiring with the
  // DA8W8 path but forces dtypes.wei = s4 and derives K/ldb from the
  // activation's contraction dim (independent of the packing container).
  const int64_t unpacked_K_w13 = inputs[0].size(1);
  const bool has_s4_weights =
      check_weight_and_infer_is_da8w4(inputs[0], w13_weights[0]);
  // Full-width int8 is DA8W8 (s8); an int8 DA8W4 container is excluded here.
  const bool has_s8_weights =
      w13_weights[0].scalar_type() == c10::kChar && !has_s4_weights;

  // Weight-side metadata vectors (sized to the total count). The library's
  // prepack-extras contract requires the six vectors below to be
  // >= total_matmul, otherwise its `min(...)` clamp silently truncates the
  // warm and inactive experts trigger reorder spikes when they later fire.
  const std::vector<bool> transB_vec(num_total, true);
  std::vector<int> N_vec(num_total), K_vec(num_total);
  std::vector<const void *> weight_ptrs(num_total);
  std::vector<int> ldb_vec(num_total);
  const std::vector<bool> is_weights_const_vec(num_total, true);

  // `params` is sized to num_total so per-expert weight metadata
  // (`dtypes.wei`) lives alongside the rest of the weight-side vectors.
  // ZenDNN explicitly supports the framework keeping `params[]` at its
  // original size to preserve weight-side prepack metadata at the tail
  // (see group_matmul_parallel.cpp::check_m_tile_safe comment lines
  // 291-297). Active per-op fields (`dtypes.{src,dst,bias}`, `plugin_op`)
  // are filled only for [0, num_active); inactive-tail entries keep
  // `dtypes.wei` set and the rest at their defaults — uniformity checks
  // and prepack only read [0, num_active) and `params[0]` respectively,
  // so the tail's defaulted fields are never observed.
  std::vector<zendnnl::lowoha::matmul::matmul_params> params(num_total);

  // Weight metadata population: every expert (active prefix + inactive tail).
  for (int op_idx = 0; op_idx < num_total; ++op_idx) {
    const auto &w13_weight = w13_weights[op_idx];
    N_vec[op_idx] = w13_weight.size(0);
    weight_ptrs[op_idx] = w13_weight.data_ptr();
    if (has_s4_weights) {
      // Packed [N, K/8] int32: logical K and ldb are both the unpacked K.
      K_vec[op_idx] = static_cast<int>(unpacked_K_w13);
      ldb_vec[op_idx] = static_cast<int>(unpacked_K_w13);
      params[op_idx].dtypes.wei = data_type_t::s4;
    } else {
      K_vec[op_idx] = w13_weight.size(1);
      ldb_vec[op_idx] = w13_weight.stride(0);
      params[op_idx].dtypes.wei = get_zendnnl_dtype(w13_weight);
    }
  }

  // Input + per-op-params population: active experts only.
  for (int op_idx = 0; op_idx < num_active; ++op_idx) {
    const auto &input = inputs[op_idx];

    M_vec[op_idx] = input.size(0);
    src_ptrs[op_idx] = input.data_ptr();
    lda_vec[op_idx] = input.stride(0);

    const bool bias_defined = has_tensor(w13_bias[op_idx]);
    if (bias_defined) {
      bias_ptrs[op_idx] = w13_bias[op_idx]->data_ptr();
    }

    // When gemm_outputs is empty, pass nullptr — ZenDNN handles allocation
    // internally
    if (!gemm_outputs.empty()) {
      dst_ptrs[op_idx] = gemm_outputs[op_idx].data_ptr();
      params[op_idx].dtypes.dst = get_zendnnl_dtype(gemm_outputs[op_idx]);
    } else {
      params[op_idx].dtypes.dst = get_zendnnl_dtype(input);
    }
    ldc_vec[op_idx] = N_vec[op_idx];

    params[op_idx].dtypes.src = get_zendnnl_dtype(input);
    params[op_idx].dtypes.bias =
        bias_defined ? get_zendnnl_dtype(*w13_bias[op_idx]) : data_type_t::none;
    params[op_idx].plugin_op = zentorch_op_name;

    // Dynamic quant paths (both quantize the activation to s8 per token):
    //   * dynamic-A8W8: s8 weight, per-channel {1, N} scale.
    //   * dynamic-A8W4: s4 weight, per-group {G, N} scale.
    if (has_s8_weights || has_s4_weights) {
      params[op_idx].dtypes.compute = data_type_t::s8;
      params[op_idx].dynamic_quant = true;
      zendnnl::lowoha::matmul::matmul_quantization_params_t qparams{};

      // Weight scale (already validated in validate_dtypes_and_shapes)
      const auto &ws = *w13_scales[op_idx];
      qparams.wei_scale.buff = ws.data_ptr();
      qparams.wei_scale.dt = get_zendnnl_dtype(ws);
      // Normalize 1D {N} to 2D {1, N} for per-channel format required by LowOHA
      auto ws_dims = ws.sizes().vec();
      if (ws_dims.size() == 1) {
        ws_dims = {1, ws_dims[0]};
      }
      qparams.wei_scale.dims = ws_dims;

      // Source scale: caller-allocated buffer, kernel fills it at runtime.
      // Granularity determined by weight scale shape:
      //   wei_scale {1, N} (per-channel) → src_scale {M, 1} (per-token)
      //   wei_scale {G, N} (per-group)   → src_scale {M, 1} (per-token,
      //                                    broadcast across the G groups)
      const int64_t M = input.size(0);
      // new_empty inherits ws's dtype and device and is contiguous, i.e. the
      // {1, 1} strides the pre-migration empty_strided_cpu asked for.
      auto src_scale_tensor = torch::stable::new_empty(ws, {M, 1});
      qparams.src_scale.buff = src_scale_tensor.data_ptr();
      qparams.src_scale.dt = get_zendnnl_dtype(ws);
      qparams.src_scale.dims = {M, 1};
      // Keep tensor alive until group_matmul_direct returns
      temp_src_scales.push_back(std::move(src_scale_tensor));

      params[op_idx].quant_params = qparams;
    }
  }

  // Engage ZenDNN's framework prepack-extras contract. See
  // third_party/ZenDNN/docs/operator/lowoha_group_matmul_operator.md
  // ("Framework prepack-extras contract") and lowoha_common.hpp matmul_params
  // for the full semantics. The dispatcher reads these from params[0] only:
  //   - active_matmul: number of GEMMs to actually compute (leading prefix
  //                    of every weight-side vector + all input-side vectors).
  //   - total_matmul : total expert weight slots present; the prepack module
  //                    pre-warms the inner-kernel weight cache for ALL of
  //                    them, so any expert that fires on a future call hits
  //                    a warm cache and avoids the on-the-fly reorder spike.
  // When num_total == num_active there is no prepack-extras tail; the
  // contract is still engaged (functionally equivalent to the legacy path
  // for that case) but stays self-documenting.
  params[0].active_matmul = static_cast<uint32_t>(num_active);
  params[0].total_matmul = static_cast<uint32_t>(num_total);

  // Configure gated activation post-op (applied after Op1 GEMM, before Op2)
  zendnnl::lowoha::matmul::grp_matmul_gated_act_params gated_act{};
  gated_act.act = gated_act_type;

  // Configure MoE weighted-reduce post-op (applied after all expert GEMMs)
  zendnnl::lowoha::matmul::group_matmul_moe_postop_params moe_params{};

  if (use_moe) {

    const int64_t num_tokens = topk_weights->size(0);
    const int64_t topk = topk_weights->size(1);
    // Gated activations (silu_and_mul, gelu_and_mul, swiglu_oai_mul) use fused
    // [gate_W | up_W] weights with N = 2*D columns. The GEMM produces [M, 2*D],
    // then the activation reduces it to [M, D]. So the effective hidden
    // dimension for the MoE output is N/2 when gated activation is active, N
    // otherwise.
    const int hidden_dim = use_gated_act ? N_vec[0] / 2 : N_vec[0];

    moe_params.num_tokens = num_tokens;
    moe_params.topk = topk;
    moe_params.output = moe_output->data_ptr();
    moe_params.ldc_output = hidden_dim;
    moe_params.topk_weights = topk_weights->const_data_ptr<float>();
    moe_params.skip_weighted = false;
    moe_params.row_ptrs = reinterpret_cast<const void **>(row_ptrs->data_ptr());
  }

  // Fused MoE (Op1 → activation → Op2) setup
  const bool use_fused_moe = !w2_weights.empty();
  zendnnl::lowoha::matmul::grp_matmul_fused_moe_params fused_moe{};

  if (use_fused_moe) {

    // W2 down-weight metadata follows the same prepack-extras layout as
    // W13: sized to num_total so the warmer sees every advertised expert.
    // Bias_down stays at num_active since biases aren't part of the
    // prepack contract.
    fused_moe.down_weight.resize(num_total);
    fused_moe.N_down.resize(num_total);
    fused_moe.ldb_down.resize(num_total);
    fused_moe.bias_down.resize(num_active, nullptr);
    fused_moe.bias_dt_down = data_type_t::none;

    for (int op_idx = 0; op_idx < num_total; ++op_idx) {
      fused_moe.down_weight[op_idx] = w2_weights[op_idx]->data_ptr();
      fused_moe.N_down[op_idx] = w2_weights[op_idx]->size(0);
      if (has_s4_weights) {
        // ldb_down is the unpacked K_down (down-proj input dim: N/2 gated, N
        // otherwise), mirroring the W13 metadata above.
        const int64_t unpacked_K_down = use_gated_act
                                            ? w13_weights[op_idx].size(0) / 2
                                            : w13_weights[op_idx].size(0);
        fused_moe.ldb_down[op_idx] = static_cast<int>(unpacked_K_down);
      } else {
        fused_moe.ldb_down[op_idx] = w2_weights[op_idx]->stride(0);
      }
    }

    for (int op_idx = 0; op_idx < num_active; ++op_idx) {
      if (has_tensor(w2_bias[op_idx])) {
        fused_moe.bias_down[op_idx] = w2_bias[op_idx]->data_ptr();
        if (fused_moe.bias_dt_down == data_type_t::none) {
          fused_moe.bias_dt_down = get_zendnnl_dtype(w2_bias[op_idx].value());
        }
      }
    }

    // Populate Op2 weight scales when w2 weights are quantized (DA8W8/DA8W4).
    // Op2 inherits dynamic_quant, dtypes.compute, and src_scale.dims
    // from params[i] — only the weight scale is per-pass.
    if (!w2_scales.empty()) {
      fused_moe.down_scale.resize(num_active);
      for (int op_idx = 0; op_idx < num_active; ++op_idx) {
        if (has_tensor(w2_scales[op_idx])) {
          const auto &ws = *w2_scales[op_idx];
          auto &q = fused_moe.down_scale[op_idx];
          q.buff = ws.data_ptr();
          q.dt = get_zendnnl_dtype(ws);
          auto ws_dims = ws.sizes().vec();
          if (ws_dims.size() == 1) {
            ws_dims = {1, ws_dims[0]};
          }
          q.dims.assign(ws_dims.begin(), ws_dims.end());
        }
      }
    }

    // When fused_moe is active, MoE reduce operates on N_down,
    // not on Op1 dst. Adjust hidden_dim accordingly.
    if (use_moe) {
      moe_params.ldc_output = fused_moe.N_down[0];
    }
  }

  // Execute: Op1 GEMMs + optional gated activation + optional Op2 + optional
  // MoE reduce
  status_t status = zendnnl::lowoha::matmul::group_matmul_direct(
      layouts, transA_vec, transB_vec, M_vec, N_vec, K_vec, alpha_vec, src_ptrs,
      lda_vec, weight_ptrs, ldb_vec, bias_ptrs, beta_vec, dst_ptrs, ldc_vec,
      is_weights_const_vec, params, use_moe ? &moe_params : nullptr,
      use_gated_act ? &gated_act : nullptr,
      use_fused_moe ? &fused_moe : nullptr);

  ZENTORCH_CHECK(status == status_t::success,
                 "zentorch_group_matmul: group_matmul_direct execution failed");

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
}

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  // `gemm_outputs` and `moe_output` are the mutated args but are declared in
  // their natural schema positions (0 and 4) rather than as trailing `.out`
  // kwargs, so they already line up positionally with the kernel parameters
  // for TORCH_BOX.
  m.def("zentorch_group_matmul.out(Tensor(a!)[] gemm_outputs, "
        "Tensor[] inputs, Tensor[] w13_weights, "
        "Tensor?[] w2_weights, "
        "Tensor(b!)? moe_output, Tensor? topk_weights, "
        "Tensor? row_ptrs, str activation, "
        "Tensor?[] w13_bias, Tensor?[] w2_bias, "
        "Tensor?[] w13_scales, "
        "Tensor?[] w2_scales, *, "
        "str zentorch_op_name='zentorch::zentorch_group_matmul.out') "
        "-> ()");
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_group_matmul.out",
         TORCH_BOX(&zentorch::zentorch_group_matmul_out_impl));
}

} // namespace zentorch