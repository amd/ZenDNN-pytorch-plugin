/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "GroupMatmul.hpp"
#include "MatmulUtils.hpp"
#include "Memory.hpp"

using namespace zendnnl::interface;

namespace zentorch {

static bool has_tensor(const c10::optional<at::Tensor> &opt) {
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
static void
validate_dtypes_and_shapes(const std::vector<at::Tensor> &inputs,
                           const std::vector<at::Tensor> &weights,
                           const std::vector<c10::optional<at::Tensor>> &bias) {

  ZENTORCH_CHECK(inputs.size() > 1,
                 "zentorch_group_matmul: sequential mode (inputs.size() == 1) "
                 "is not supported; only parallel mode (one input per expert) "
                 "is currently implemented");

  const int num_active = static_cast<int>(inputs.size());
  const int num_total = static_cast<int>(weights.size());

  ZENTORCH_CHECK(inputs.size() == bias.size(),
                 "zentorch_group_matmul: inputs.size() (", inputs.size(),
                 ") must equal bias.size() (", bias.size(),
                 ") (active-expert count)");
  ZENTORCH_CHECK(weights.size() >= inputs.size(),
                 "zentorch_group_matmul: weights.size() (", weights.size(),
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
  ZENTORCH_CHECK(ref_dtype == c10::kFloat || ref_dtype == c10::kBFloat16,
                 "zentorch_group_matmul: input[0] must be float32 or "
                 "bfloat16, got ",
                 ref_dtype);

  // Validate every weight (active + inactive prepack tail).
  for (int op_idx = 0; op_idx < num_total; ++op_idx) {
    const auto &weight = weights[op_idx];
    ZENTORCH_CHECK(weight.dim() == 2, "zentorch_group_matmul: weight[", op_idx,
                   "] must be 2D, got ", weight.dim(), "D");
    ZENTORCH_CHECK(weight.scalar_type() == ref_dtype,
                   "zentorch_group_matmul: weight[", op_idx, "] dtype (",
                   weight.scalar_type(), ") must match input[0] dtype (",
                   ref_dtype, ")");
    ZENTORCH_CHECK(weight.size(1) == K_ref, "zentorch_group_matmul: weight[",
                   op_idx, "] K (", weight.size(1), ") must match input[0] K (",
                   K_ref, ")");
  }

  // Validate active-only inputs and biases.
  for (int op_idx = 0; op_idx < num_active; ++op_idx) {
    const auto &input = inputs[op_idx];
    const auto &weight = weights[op_idx];

    ZENTORCH_CHECK(input.dim() == 2, "zentorch_group_matmul: input[", op_idx,
                   "] must be 2D, got ", input.dim(), "D");
    ZENTORCH_CHECK(input.scalar_type() == ref_dtype,
                   "zentorch_group_matmul: input[", op_idx, "] dtype (",
                   input.scalar_type(), ") must match input[0] dtype (",
                   ref_dtype, ")");

    // Bias must be 1D with size matching N (weight rows) and matching dtype.
    if (has_tensor(bias[op_idx])) {
      ZENTORCH_CHECK(bias[op_idx]->dim() == 1, "zentorch_group_matmul: bias[",
                     op_idx, "] must be 1D");
      ZENTORCH_CHECK(bias[op_idx]->size(0) == weight.size(0),
                     "zentorch_group_matmul: bias[", op_idx, "] size (",
                     bias[op_idx]->size(0), ") must match N (", weight.size(0),
                     ")");
      ZENTORCH_CHECK(bias[op_idx]->scalar_type() == ref_dtype,
                     "zentorch_group_matmul: bias[", op_idx, "] dtype (",
                     bias[op_idx]->scalar_type(),
                     ") must match input[0] "
                     "dtype (",
                     ref_dtype, ")");
    }
  }
}

// gemm_outputs holds one destination per active expert. Compare against
// inputs.size() (the active-expert count), not weights.size() (which is
// sized to E and includes trailing inactive-expert passthrough entries).
static void validate_gemm_outputs(const std::vector<at::Tensor> &gemm_outputs,
                                  const std::vector<at::Tensor> &inputs) {
  ZENTORCH_CHECK(gemm_outputs.size() == inputs.size(),
                 "zentorch_group_matmul: gemm_outputs.size() (",
                 gemm_outputs.size(), ") must equal inputs.size() (",
                 inputs.size(), ")");
}

// Todo: https://jira.xilinx.com/browse/ZENAI-3656
// Add tests for all the failure cases in this op.
// Validates fused w2 (down projection) list sizes, per-expert shapes, and
// dtypes. w2_input_dim = N/2 when gated activation is active, N otherwise.
//
// Sizing mirrors validate_dtypes_and_shapes:
//   - w2_weights : sized E (active prefix + inactive prepack tail), matches
//                  weights.size().
//   - w2_bias    : sized E_a (active only), matches inputs.size().
// We validate every w2 weight (the inactive tail also gets prepacked) and
// only the active w2 biases.
static void
validate_w2_params(const std::vector<at::Tensor> &inputs,
                   const std::vector<at::Tensor> &weights,
                   const std::vector<c10::optional<at::Tensor>> &w2_weights,
                   const std::vector<c10::optional<at::Tensor>> &w2_bias,
                   bool use_gated_act) {

  ZENTORCH_CHECK(
      w2_weights.size() == weights.size(),
      "zentorch_group_matmul: w2_weights (", w2_weights.size(),
      ") must equal weights.size() (", weights.size(),
      ") (full expert count, active prefix + inactive prepack tail)");
  ZENTORCH_CHECK(w2_bias.size() == inputs.size(),
                 "zentorch_group_matmul: w2_bias (", w2_bias.size(),
                 ") must equal inputs.size() (", inputs.size(),
                 ") (active-expert count)");

  const int num_active = static_cast<int>(inputs.size());
  const int num_total = static_cast<int>(w2_weights.size());
  const auto ref_dtype = inputs[0].scalar_type();

  // Validate every w2 weight (active + inactive prepack tail).
  for (int op_idx = 0; op_idx < num_total; ++op_idx) {
    ZENTORCH_CHECK(has_tensor(w2_weights[op_idx]),
                   "zentorch_group_matmul: w2_weights[", op_idx,
                   "] must not be None when fused w2 is enabled");
    // w2 receives the post-activation output: N/2 with gated act, N without.
    // Per-expert weights[op_idx] has the same N for both active and inactive
    // experts in the canonical MoE flow.
    const int64_t w2_input_dim =
        use_gated_act ? weights[op_idx].size(0) / 2 : weights[op_idx].size(0);

    ZENTORCH_CHECK(w2_weights[op_idx]->dim() == 2 &&
                       w2_weights[op_idx]->size(1) == w2_input_dim &&
                       w2_weights[op_idx]->scalar_type() == ref_dtype,
                   "zentorch_group_matmul: w2_weights[", op_idx,
                   "] must be 2D [K_out, ", w2_input_dim, "] with dtype ",
                   ref_dtype, ", got ", w2_weights[op_idx]->dim(), "D [",
                   w2_weights[op_idx]->size(0), ", ",
                   w2_weights[op_idx]->size(1), "] dtype ",
                   w2_weights[op_idx]->scalar_type());
  }

  // Validate active-only w2 biases.
  for (int op_idx = 0; op_idx < num_active; ++op_idx) {
    if (has_tensor(w2_bias[op_idx])) {
      const int64_t K_out = w2_weights[op_idx]->size(0);
      ZENTORCH_CHECK(
          w2_bias[op_idx]->dim() == 1 && w2_bias[op_idx]->size(0) == K_out &&
              w2_bias[op_idx]->scalar_type() == ref_dtype,
          "zentorch_group_matmul: w2_bias[", op_idx, "] must be 1D with size ",
          K_out, " and dtype ", ref_dtype, ", got ", w2_bias[op_idx]->dim(),
          "D, size ", w2_bias[op_idx]->size(0), ", dtype ",
          w2_bias[op_idx]->scalar_type());
    }
  }
}

// Validates MoE weighted-reduce: topk_weights, row_ptrs, moe_output must all be
// provided. Returns true if MoE is enabled.
static bool validate_moe_params(const c10::optional<at::Tensor> &topk_weights,
                                const c10::optional<at::Tensor> &row_ptrs,
                                const c10::optional<at::Tensor> &moe_output) {
  const bool use_moe = has_tensor(topk_weights);
  if (use_moe) {
    ZENTORCH_CHECK(has_tensor(row_ptrs) && has_tensor(moe_output),
                   "zentorch_group_matmul: when topk_weights is provided, "
                   "row_ptrs and moe_output must also be provided");
  }
  return use_moe;
}

static bool
validate_all_inputs(const std::vector<at::Tensor> &inputs,
                    const std::vector<at::Tensor> &weights,
                    const std::vector<c10::optional<at::Tensor>> &bias,
                    const std::vector<at::Tensor> &gemm_outputs,
                    const c10::optional<at::Tensor> &topk_weights,
                    const c10::optional<at::Tensor> &row_ptrs,
                    const c10::optional<at::Tensor> &moe_output,
                    const std::vector<c10::optional<at::Tensor>> &w2_weights,
                    const std::vector<c10::optional<at::Tensor>> &w2_bias,
                    bool use_gated_act) {

  validate_dtypes_and_shapes(inputs, weights, bias);
  if (!gemm_outputs.empty()) {
    validate_gemm_outputs(gemm_outputs, inputs);
  }
  if (!w2_weights.empty()) {
    validate_w2_params(inputs, weights, w2_weights, w2_bias, use_gated_act);
  }
  return validate_moe_params(topk_weights, row_ptrs, moe_output);
}

// Maps activation string to LowOHA gated activation enum.
// Supported: "none", "silu", "gelu", "swigluoai".
static zendnnl::lowoha::matmul::grp_matmul_gated_act_t
map_activation_to_gated_act(std::string_view activation) {
  using gated_act_t = zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
  if (activation.empty() || activation == "none")
    return gated_act_t::none;
  if (activation == "silu")
    return gated_act_t::silu_and_mul;
  if (activation == "gelu")
    return gated_act_t::gelu_and_mul;
  if (activation == "swigluoai")
    return gated_act_t::swiglu_oai_mul;
  ZENTORCH_CHECK(false, "zentorch_group_matmul: unsupported activation '",
                 std::string(activation), "'");
  return gated_act_t::none;
}

void zentorch_group_matmul_out_impl(
    std::vector<at::Tensor> gemm_outputs, const std::vector<at::Tensor> &inputs,
    const std::vector<at::Tensor> &weights,
    const std::vector<c10::optional<at::Tensor>> &bias,
    std::string_view activation,
    const std::vector<c10::optional<at::Tensor>> &w2_weights,
    const std::vector<c10::optional<at::Tensor>> &w2_bias,
    c10::optional<at::Tensor> moe_output,
    const c10::optional<at::Tensor> &topk_weights,
    const c10::optional<at::Tensor> &row_ptrs,
    const std::string &zentorch_op_name) {

  const auto gated_act_type = map_activation_to_gated_act(activation);
  const bool use_gated_act =
      gated_act_type != zendnnl::lowoha::matmul::grp_matmul_gated_act_t::none;

  const bool use_moe = validate_all_inputs(inputs, weights, bias, gemm_outputs,
                                           topk_weights, row_ptrs, moe_output,
                                           w2_weights, w2_bias, use_gated_act);

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
  const int num_total = static_cast<int>(weights.size());

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
    const auto &weight = weights[op_idx];
    N_vec[op_idx] = weight.size(0);
    K_vec[op_idx] = weight.size(1);
    weight_ptrs[op_idx] = weight.data_ptr();
    ldb_vec[op_idx] = weight.stride(0);
    params[op_idx].dtypes.wei = get_zendnnl_dtype(weight);
  }

  // Input + per-op-params population: active experts only.
  for (int op_idx = 0; op_idx < num_active; ++op_idx) {
    const auto &input = inputs[op_idx];

    M_vec[op_idx] = input.size(0);
    src_ptrs[op_idx] = input.data_ptr();
    lda_vec[op_idx] = input.stride(0);

    const bool bias_defined = has_tensor(bias[op_idx]);
    if (bias_defined) {
      bias_ptrs[op_idx] = bias[op_idx]->data_ptr();
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
        bias_defined ? get_zendnnl_dtype(*bias[op_idx]) : data_type_t::none;
    params[op_idx].plugin_op = zentorch_op_name;
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
    // Gated activations (silu, gelu, swigluoai) use fused [gate_W | up_W]
    // weights with N = 2*D columns. The GEMM produces [M, 2*D], then the
    // activation reduces it to [M, D]. So the effective hidden dimension
    // for the MoE output is N/2 when gated activation is active, N otherwise.
    const int hidden_dim = use_gated_act ? N_vec[0] / 2 : N_vec[0];

    moe_params.num_tokens = num_tokens;
    moe_params.topk = topk;
    moe_params.output = moe_output->data_ptr();
    moe_params.ldc_output = hidden_dim;
    moe_params.topk_weights = topk_weights->data_ptr<float>();
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
      fused_moe.ldb_down[op_idx] = w2_weights[op_idx]->stride(0);
    }

    for (int op_idx = 0; op_idx < num_active; ++op_idx) {
      if (has_tensor(w2_bias[op_idx])) {
        fused_moe.bias_down[op_idx] = w2_bias[op_idx]->data_ptr();
        if (fused_moe.bias_dt_down == data_type_t::none) {
          fused_moe.bias_dt_down = get_zendnnl_dtype(w2_bias[op_idx].value());
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

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_group_matmul.out(Tensor(a!)[] gemm_outputs, "
        "Tensor[] inputs, Tensor[] weights, "
        "Tensor?[] bias, str activation, "
        "Tensor?[] w2_weights, Tensor?[] w2_bias, "
        "Tensor(b!)? moe_output=None, Tensor? topk_weights=None, "
        "Tensor? row_ptrs=None, *, "
        "str zentorch_op_name='zentorch::zentorch_group_matmul.out') "
        "-> ()");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_group_matmul.out", zentorch::zentorch_group_matmul_out_impl);
}

} // namespace zentorch
