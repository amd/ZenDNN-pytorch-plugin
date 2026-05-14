/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "Utils.hpp"
#include <ATen/Parallel.h>
#include <algorithm>
#include <cstring>
#include <torch/library.h>
#include <utility>
#include <vector>

namespace zentorch {

// ---------------------------------------------------------------------------
// Forward declaration of the GroupMatmul.cpp single-call MoE entry point.
// The full ZenDNN postop chain (W13 GEMM -> gated activation -> W2 GEMM ->
// weighted reduce into [T, H]) executes inside one call.
// ---------------------------------------------------------------------------

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
    const std::string &zentorch_op_name);

// ---------------------------------------------------------------------------
// Phase 1 — Token-Expert Grouping
//
// For each routed (token t, slot k) we need expert e = topk_id[t][k] to
// receive a copy of input[t] in its per-expert input buffer. We do this in
// two passes:
//
//   Pass 1 (single-threaded; trivially cheap O(T*K) bookkeeping):
//     - Walk the T*K (t, k) pairs in flat order.
//     - On the first encounter of expert e, append e to `active_expert_ids`.
//       To resolve an already-seen expert back to its active slot a we
//       linear-scan `active_expert_ids`; E_a is small in practice (tens for
//       typical MoE configs) and the scan stays in L1.
//     - For each (t, k), record `topk_to_expert_row[i] = (active_idx, pos)`
//       where `pos = tokens_per_active[a]++` is the deterministic write
//       position in active slot a's eventual input tensor (no atomics needed).
//
//   Pass 2 (parallel; the actual data movement):
//     - Allocate one [M_e, H] tensor per active expert, indexed by
//       active_idx (so `grouped_inputs.size() == E_a`, ready to hand to
//       `group_matmul` directly).
//     - `at::parallel_for` over the T*K pairs: for each i, look up the
//       pre-assigned (active_idx, pos) and `memcpy` row `t` of `input`
//       into row `pos` of `grouped_inputs[active_idx]`.
//
// All Pass-1 auxiliary state is sized to E_a (not E) — `active_expert_ids`
// is the single source of truth for the active set, no reverse map needed.
//
// ----- Worked example: T=3, H=4, E=5 experts, K=2 top-k routing -----
//
// Inputs:
//   input  [T, H] = [[ t0_h0, t0_h1, t0_h2, t0_h3 ],   # token 0
//                    [ t1_h0, t1_h1, t1_h2, t1_h3 ],   # token 1
//                    [ t2_h0, t2_h1, t2_h2, t2_h3 ]]   # token 2
//
//   topk_id [T, K] = [[ 3, 0 ],     # token 0 -> experts {3, 0}
//                     [ 3, 1 ],     # token 1 -> experts {3, 1}
//                     [ 0, 1 ]]     # token 2 -> experts {0, 1}
//
// After Pass 1 (single sweep over T*K=6 pairs, first-encounter ordering):
//   active_expert_ids = [ 3, 0, 1 ]                     # E_a = 3
//   tokens_per_active = [ 2, 2, 2 ]                     # local; final M_e per
//   active slot
//
//   topk_to_expert_row [T*K = 6 entries] = (active_idx, row_in_expert):
//     i=0 (t=0,k=0) -> (a=0, row=0)   # expert 3
//     i=1 (t=0,k=1) -> (a=1, row=0)   # expert 0
//     i=2 (t=1,k=0) -> (a=0, row=1)   # expert 3
//     i=3 (t=1,k=1) -> (a=2, row=0)   # expert 1
//     i=4 (t=2,k=0) -> (a=1, row=1)   # expert 0
//     i=5 (t=2,k=1) -> (a=2, row=1)   # expert 1
//
// After Pass 2 (parallel memcpy using the pre-assigned positions):
//   grouped_inputs[0]  (active_idx 0 = expert 3, M=2) = [ input[0], input[1] ]
//   grouped_inputs[1]  (active_idx 1 = expert 0, M=2) = [ input[0], input[2] ]
//   grouped_inputs[2]  (active_idx 2 = expert 1, M=2) = [ input[1], input[2] ]
// ---------------------------------------------------------------------------

struct TokenExpertMapping {
  // Size E_a. grouped_inputs[a] is the [M_e, H] input tensor for the
  // a-th active expert (active expert id = active_expert_ids[a]).
  std::vector<at::Tensor> grouped_inputs;
  // Size E_a. active_expert_ids[a] = the original expert id (in [0, E))
  // for the a-th active expert. Order is first-encounter in `topk_id`.
  std::vector<int32_t> active_expert_ids;
  // T*K entries; index (t*K + k) holds (active_idx, row_in_expert).
  std::vector<std::pair<int32_t, int32_t>> topk_to_expert_row;
};

static TokenExpertMapping
build_token_expert_mapping(const at::Tensor &input, const at::Tensor &topk_id) {

  const int64_t T = input.size(0);
  const int64_t H = input.size(1);
  const int64_t K = topk_id.size(1);
  const int64_t total_pairs = T * K;
  const int64_t row_bytes = H * input.element_size();

  // topk_id is contiguous int32 [T, K]; flat indexing is i = t*K + k.
  const int32_t *topk_id_serialized = topk_id.const_data_ptr<int32_t>();

  // ----- Pass 1: register active experts + assign per-active positions ----
  // `active_expert_ids` is the only structure that tracks the active set;
  // resolving an already-seen expert back to its active slot is a small
  // linear scan (E_a is in the tens for typical MoE configs, stays in L1).
  TokenExpertMapping mapping;
  mapping.topk_to_expert_row.resize(total_pairs);

  std::vector<int32_t> tokens_per_active; // grows alongside active_expert_ids
  for (int64_t i = 0; i < total_pairs; ++i) {
    const int32_t e = topk_id_serialized[i];
    auto it = std::find(mapping.active_expert_ids.begin(),
                        mapping.active_expert_ids.end(), e);
    int32_t a;
    if (it == mapping.active_expert_ids.end()) {
      a = static_cast<int32_t>(mapping.active_expert_ids.size());
      mapping.active_expert_ids.emplace_back(e);
      tokens_per_active.emplace_back(0);
    } else {
      a = static_cast<int32_t>(it - mapping.active_expert_ids.begin());
    }
    mapping.topk_to_expert_row[i] = {a, tokens_per_active[a]++};
  }
  const int64_t E_a = static_cast<int64_t>(mapping.active_expert_ids.size());

  // ----- Allocate per-active-expert [M_e, H] tensors ----------------------
  mapping.grouped_inputs.resize(E_a);
  for (int64_t a = 0; a < E_a; ++a) {
    mapping.grouped_inputs[a] =
        at::empty({tokens_per_active[a], H}, input.options());
  }

  // ----- Pass 2: parallel memcpy using pre-assigned positions -------------
  // No atomics — `topk_to_expert_row[i]` already tells us where row i lands.
  const auto *src_base = reinterpret_cast<const char *>(input.const_data_ptr());

  std::vector<char *> dst_base(E_a);
  for (int64_t a = 0; a < E_a; ++a) {
    dst_base[a] =
        reinterpret_cast<char *>(mapping.grouped_inputs[a].data_ptr());
  }

  at::parallel_for(0, total_pairs, /*grain_size=*/64,
                   [&](int64_t begin, int64_t end) {
                     for (int64_t i = begin; i < end; ++i) {
                       const int64_t t = i / K;
                       const auto [a, pos] = mapping.topk_to_expert_row[i];
                       std::memcpy(dst_base[a] + pos * row_bytes,
                                   src_base + t * row_bytes, row_bytes);
                     }
                   });

  return mapping;
}

// ---------------------------------------------------------------------------
// zentorch_fused_moe — Fused Mixture-of-Experts operator (out variant)
//
// Schema mirrors vLLM's `cpu_fused_moe` signature so that vLLM's CPUFusedMOE
// dispatch can swap the op name without rewriting call sites:
//
//   torch.ops.zentorch.zentorch_fused_moe(
//       output, input, w13, w2, w13_bias, w2_bias,
//       topk_weights, topk_id, skip_weighted, act,
//       *, zentorch_op_name="zentorch::zentorch_fused_moe")
//
// `output` is mutated in place (Tensor(a!)). Returns ().
//
// ----------------------------- Input contract ------------------------------
// All shape/dtype/bias validation is performed once, at patch-install time
// in `src/cpu/python/zentorch/vllm/__init__.py` (the only producer of calls
// into this op). The C++ op trusts its inputs and assumes:
//
//   input          : 2D [T, H], f32 or bf16, contiguous
//   output         : 2D [T, H], same dtype as input, ZERO-INITIALIZED
//                    (Phase 5 accumulates into it)
//   w13            : 3D [E, 2*I, H], same dtype as input
//   w2             : 3D [E, H, I],   same dtype as input
//   w13_bias       : None or [E, 2*I] (same dtype as input)
//   w2_bias        : None or [E, H]   (same dtype as input)
//   topk_weights   : 2D [T, K], f32, contiguous
//   topk_id        : 2D [T, K], int32, contiguous, values in [0, E)
//   skip_weighted  : bool; if true, requires K == 1
//   act            : one of {"silu", "gelu", "swigluoai"}
// ---------------------------------------------------------------------------

void zentorch_fused_moe(at::Tensor &output, const at::Tensor &input,
                        const at::Tensor &w13, const at::Tensor &w2,
                        const c10::optional<at::Tensor> &w13_bias,
                        const c10::optional<at::Tensor> &w2_bias,
                        const at::Tensor &topk_weights,
                        const at::Tensor &topk_id, bool skip_weighted,
                        std::string_view act, std::string zentorch_op_name) {

  const int64_t T = input.size(0);
  const int64_t K = topk_id.size(1);
  const int64_t total_pairs = T * K;
  const int64_t row_bytes = input.size(1) * input.element_size();

  // ---------------------- Phase 1: token-expert grouping ---------------------
  const auto mapping = build_token_expert_mapping(input, topk_id);
  const int64_t E_a = static_cast<int64_t>(mapping.active_expert_ids.size());

  // ---------------------- Temporary Guard ------------------------------------
  // Guard: zentorch_group_matmul_out_impl requires E_a > 1.
  // E_a == 1 is structurally guaranteed when K == 1 and T == 1
  // (apply_router_weight_on_input=True path). Rather than letting the
  // call reach group_matmul_direct and crash on its inputs.size() > 1
  // assertion, fail here with a clear, actionable message.
  TORCH_CHECK(
      E_a > 1, "zentorch_fused_moe: only ", E_a,
      " expert(s) received tokens. "
      "zentorch_group_matmul_out_impl requires at least 2 active experts. "
      "This typically occurs with K=1 routing "
      "(apply_router_weight_on_input=True) "
      "at single-token decode (T=1). Use the standard vLLM cpu_fused_moe path "
      "for this configuration, or unset ZENTORCH_FUSED_MOE.");

  // ---------------------- Phase 2: build active-only weight slices ----------
  // ZenDNN's group_matmul wants per-expert `Tensor` slices, not a 3-D
  // [E, ...] view. We feed only the E_a active experts; their order matches
  // `mapping.grouped_inputs` (active_idx-aligned).
  std::vector<at::Tensor> w13_slices(E_a);
  std::vector<at::Tensor> w2_slices_raw(E_a);
  std::vector<c10::optional<at::Tensor>> w13_bias_slices(E_a);
  std::vector<c10::optional<at::Tensor>> w2_bias_slices(E_a);
  // The GroupMatmul API takes w2_weights as an optional vector; build it
  // from w2_slices_raw in the same loop.
  std::vector<c10::optional<at::Tensor>> w2_weight_slices(E_a);

  const bool has_w13_bias = w13_bias.has_value() && w13_bias->defined();
  const bool has_w2_bias = w2_bias.has_value() && w2_bias->defined();

  for (int64_t a = 0; a < E_a; ++a) {
    const int64_t e = mapping.active_expert_ids[a];
    w13_slices[a] = w13.select(0, e);
    w2_slices_raw[a] = w2.select(0, e);
    w2_weight_slices[a] = w2_slices_raw[a];
    w13_bias_slices[a] = has_w13_bias
                             ? c10::optional<at::Tensor>(w13_bias->select(0, e))
                             : c10::nullopt;
    w2_bias_slices[a] = has_w2_bias
                            ? c10::optional<at::Tensor>(w2_bias->select(0, e))
                            : c10::nullopt;
  }

  // ---------------------- Phase 5 setup: row_ptrs for weighted reduce -------
  // ZenDNN's MoE postop reads each (t, k) result via a raw row address, then
  // accumulates `topk_weights[t, k] * row` into `output[t]`. We reuse the
  // per-expert `grouped_inputs` buffers as the W2 output destinations: by the
  // time W2 needs to write, W13 has already consumed its inputs from these
  // buffers within the fused chain, and both shapes are [M_e, H]. row_ptrs
  // therefore point directly into `mapping.grouped_inputs`.
  auto row_ptrs =
      at::empty({total_pairs}, at::TensorOptions().dtype(at::kLong));
  int64_t *row_ptrs_data = row_ptrs.data_ptr<int64_t>();
  for (int64_t i = 0; i < total_pairs; ++i) {
    const auto [a, pos] = mapping.topk_to_expert_row[i];
    row_ptrs_data[i] = reinterpret_cast<int64_t>(
        static_cast<char *>(mapping.grouped_inputs[a].data_ptr()) +
        pos * row_bytes);
  }

  // When `skip_weighted` is set, vLLM has already pre-applied router weights
  // to the input. Pass an all-ones weight vector so the postop accumulates
  // raw expert outputs. (Schema requires K == 1 in this case.)
  const at::Tensor effective_topk_weights =
      skip_weighted ? at::ones_like(topk_weights) : topk_weights;

  // ---------------------- Single-call fused execution -----------------------
  // gemm_outputs is empty -> ZenDNN allocates W13 outputs internally.
  // We pass `mapping.grouped_inputs` as `w2_outputs`: the W2 down-projection
  // writes back into the same [M_e, H] buffers we used as W13 inputs, saving
  // an allocation per active expert. The full chain
  // (W13 -> gated_act -> W2 -> weighted_reduce -> output) runs inside one
  // `group_matmul_direct` call.
  zentorch_group_matmul_out_impl(
      /*gemm_outputs=*/{},
      /*inputs=*/mapping.grouped_inputs,
      /*weights=*/w13_slices,
      /*bias=*/w13_bias_slices,
      /*activation=*/act,
      /*w2_weights=*/w2_weight_slices,
      /*w2_bias=*/w2_bias_slices,
      /*moe_output=*/output,
      /*topk_weights=*/effective_topk_weights,
      /*row_ptrs=*/row_ptrs,
      /*zentorch_op_name=*/zentorch_op_name);
}

// ---------------------------------------------------------------------------
// Op registration
// ---------------------------------------------------------------------------

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_fused_moe(Tensor(a!) output, Tensor input, "
        "Tensor w13, Tensor w2, "
        "Tensor? w13_bias, Tensor? w2_bias, "
        "Tensor topk_weights, Tensor topk_id, "
        "bool skip_weighted, "
        "str act, "
        "*, str zentorch_op_name='zentorch::zentorch_fused_moe') -> ()");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_fused_moe", zentorch::zentorch_fused_moe);
}

} // namespace zentorch
