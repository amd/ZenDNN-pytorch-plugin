/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

// Single bindings file for all GatedDeltaNet (GDN) CPU ops.
//
// Per-op kernel implementations live under src/cpu/cpp/kernels/layers/gdn/
// and are compiled into libCPUkernels.a; this file exposes their wrapper
// functions through the torch.ops.zentorch.* dispatcher in one place so the
// full op surface (schemas + CPU impls) can be reviewed at a single glance.

#include <torch/all.h>

namespace zentorch {

at::Tensor zentorch_gdn_causal_conv1d_fn(
    const at::Tensor &x, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, at::Tensor &conv_states,
    const at::Tensor &query_start_loc, const at::Tensor &cache_indices,
    const at::Tensor &has_initial_state, std::string activation,
    int64_t pad_slot_id, std::string zentorch_op_name);

at::Tensor zentorch_gdn_causal_conv1d_update(
    const at::Tensor &x, at::Tensor &conv_state, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, std::string activation,
    const at::Tensor &conv_state_indices, int64_t null_block_id,
    int64_t pad_slot_id, std::string zentorch_op_name);

at::Tensor zentorch_gdn_chunk_fwd_o(const at::Tensor &q, const at::Tensor &k,
                                    const at::Tensor &v, const at::Tensor &h,
                                    const at::Tensor &g, double scale,
                                    const at::Tensor &cu_seqlens,
                                    const at::Tensor &chunk_offsets,
                                    int64_t chunk_size,
                                    std::string zentorch_op_name);

std::tuple<at::Tensor, at::Tensor> zentorch_gdn_chunk_gated_delta_rule_fwd(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &g, const at::Tensor &beta, double scale,
    const c10::optional<at::Tensor> &initial_state, bool output_final_state,
    int64_t chunk_size, const at::Tensor &cu_seqlens,
    const at::Tensor &chunk_indices, const at::Tensor &chunk_offsets,
    std::string zentorch_op_name);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
zentorch_gdn_chunk_gated_delta_rule_fwd_h(
    const at::Tensor &k, const at::Tensor &w, const at::Tensor &u,
    const at::Tensor &g, const c10::optional<at::Tensor> &initial_state,
    bool output_final_state, int64_t chunk_size, bool save_new_value,
    const at::Tensor &cu_seqlens, const at::Tensor &chunk_offsets,
    int64_t NT_total, std::string zentorch_op_name);

at::Tensor zentorch_gdn_chunk_local_cumsum(const at::Tensor &g,
                                           int64_t chunk_size,
                                           const at::Tensor &cu_seqlens,
                                           const at::Tensor &chunk_indices,
                                           std::string zentorch_op_name);

at::Tensor zentorch_gdn_chunk_scaled_dot_kkt_fwd(
    const at::Tensor &k, const at::Tensor &g, const at::Tensor &beta,
    const at::Tensor &cu_seqlens, const at::Tensor &chunk_indices,
    int64_t chunk_size, std::string zentorch_op_name);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
zentorch_gdn_fused_post_conv_prep(
    const at::Tensor &conv_output, const at::Tensor &a, const at::Tensor &b,
    const at::Tensor &A_log, const at::Tensor &dt_bias, int64_t num_k_heads,
    int64_t head_k_dim, int64_t head_v_dim, bool apply_l2norm,
    bool output_g_exp, std::string zentorch_op_name);

void zentorch_gdn_fused_recurrent_gated_delta_rule_packed_decode(
    const at::Tensor &mixed_qkv, const at::Tensor &a, const at::Tensor &b,
    const at::Tensor &A_log, const at::Tensor &dt_bias, double scale,
    at::Tensor &initial_state, at::Tensor &out,
    const at::Tensor &ssm_state_indices, bool use_qk_l2norm_in_kernel,
    std::string zentorch_op_name);

at::Tensor zentorch_gdn_fused_sigmoid_gating_delta_rule_update(
    const at::Tensor &A_log, const at::Tensor &a, const at::Tensor &b,
    const at::Tensor &dt_bias, const at::Tensor &q, const at::Tensor &k,
    const at::Tensor &v, double beta_temp, double threshold, double scale,
    at::Tensor &initial_state, const at::Tensor &cu_seqlens,
    const at::Tensor &ssm_state_indices,
    const c10::optional<at::Tensor> &num_accepted_tokens,
    bool use_qk_l2norm_in_kernel, std::string zentorch_op_name);

at::Tensor zentorch_gdn_l2norm_fwd(const at::Tensor &x, double eps,
                                   std::string zentorch_op_name);

std::tuple<at::Tensor, at::Tensor> zentorch_gdn_recompute_w_u_fwd(
    const at::Tensor &k, const at::Tensor &v, const at::Tensor &beta,
    const at::Tensor &g_cumsum, const at::Tensor &A,
    const at::Tensor &cu_seqlens, const at::Tensor &chunk_indices,
    std::string zentorch_op_name);

at::Tensor zentorch_gdn_rms_norm_gated(const at::Tensor &x,
                                       const at::Tensor &weight,
                                       const at::Tensor &z, double eps,
                                       std::string activation,
                                       std::string zentorch_op_name);

at::Tensor zentorch_gdn_solve_tril(const at::Tensor &A,
                                   const at::Tensor &cu_seqlens,
                                   const at::Tensor &chunk_indices,
                                   std::string zentorch_op_name);

} // namespace zentorch

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("gdn_causal_conv1d_fn("
        "Tensor x, Tensor weight, Tensor? bias, "
        "Tensor(a!) conv_states, "
        "Tensor query_start_loc, Tensor cache_indices, "
        "Tensor has_initial_state, "
        "str activation, int pad_slot_id, "
        "*, str zentorch_op_name='zentorch::gdn_causal_conv1d_fn'"
        ") -> Tensor");

  m.def("gdn_causal_conv1d_update("
        "Tensor x, Tensor(a!) conv_state, Tensor weight, Tensor? bias, "
        "str activation, Tensor conv_state_indices, "
        "int null_block_id, int pad_slot_id, "
        "*, str zentorch_op_name='zentorch::gdn_causal_conv1d_update'"
        ") -> Tensor");

  m.def("gdn_chunk_fwd_o("
        "Tensor q, Tensor k, Tensor v, Tensor h, Tensor g, float scale, "
        "Tensor cu_seqlens, Tensor chunk_offsets, int chunk_size, "
        "*, str zentorch_op_name='zentorch::gdn_chunk_fwd_o'"
        ") -> Tensor");

  m.def("gdn_chunk_gated_delta_rule_fwd("
        "Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
        "float scale, Tensor? initial_state, bool output_final_state, "
        "int chunk_size, Tensor cu_seqlens, Tensor chunk_indices, "
        "Tensor chunk_offsets, "
        "*, str zentorch_op_name='zentorch::gdn_chunk_gated_delta_rule_fwd'"
        ") -> (Tensor, Tensor)");

  m.def("gdn_chunk_gated_delta_rule_fwd_h("
        "Tensor k, Tensor w, Tensor u, Tensor g, Tensor? initial_state, "
        "bool output_final_state, int chunk_size, bool save_new_value, "
        "Tensor cu_seqlens, Tensor chunk_offsets, int NT_total, "
        "*, str zentorch_op_name='zentorch::gdn_chunk_gated_delta_rule_fwd_h'"
        ") -> (Tensor, Tensor, Tensor)");

  m.def("gdn_chunk_local_cumsum("
        "Tensor g, int chunk_size, Tensor cu_seqlens, Tensor chunk_indices, "
        "*, str zentorch_op_name='zentorch::gdn_chunk_local_cumsum'"
        ") -> Tensor");

  m.def("gdn_chunk_scaled_dot_kkt_fwd("
        "Tensor k, Tensor g, Tensor beta, "
        "Tensor cu_seqlens, Tensor chunk_indices, int chunk_size, "
        "*, str zentorch_op_name='zentorch::gdn_chunk_scaled_dot_kkt_fwd'"
        ") -> Tensor");

  m.def("gdn_fused_post_conv_prep("
        "Tensor conv_output, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
        "int num_k_heads, int head_k_dim, int head_v_dim, "
        "bool apply_l2norm, bool output_g_exp, "
        "*, str zentorch_op_name='zentorch::gdn_fused_post_conv_prep'"
        ") -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  m.def("gdn_fused_recurrent_gated_delta_rule_packed_decode("
        "Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
        "float scale, Tensor(a!) initial_state, Tensor(b!) out, "
        "Tensor ssm_state_indices, bool use_qk_l2norm_in_kernel, "
        "*, str zentorch_op_name="
        "'zentorch::gdn_fused_recurrent_gated_delta_rule_packed_decode'"
        ") -> ()");

  m.def(
      "gdn_fused_sigmoid_gating_delta_rule_update("
      "Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, "
      "Tensor q, Tensor k, Tensor v, "
      "float beta_temp, float threshold, float scale, "
      "Tensor(a!) initial_state, "
      "Tensor cu_seqlens, Tensor ssm_state_indices, "
      "Tensor? num_accepted_tokens, "
      "bool use_qk_l2norm_in_kernel, "
      "*, str "
      "zentorch_op_name='zentorch::gdn_fused_sigmoid_gating_delta_rule_update'"
      ") -> Tensor");

  m.def("gdn_l2norm_fwd("
        "Tensor x, float eps, *, "
        "str zentorch_op_name='zentorch::gdn_l2norm_fwd'"
        ") -> Tensor");

  m.def("gdn_recompute_w_u_fwd("
        "Tensor k, Tensor v, Tensor beta, Tensor g_cumsum, Tensor A, "
        "Tensor cu_seqlens, Tensor chunk_indices, "
        "*, str zentorch_op_name='zentorch::gdn_recompute_w_u_fwd'"
        ") -> (Tensor, Tensor)");

  m.def("gdn_rms_norm_gated("
        "Tensor x, Tensor weight, Tensor z, float eps, str activation, "
        "*, str zentorch_op_name='zentorch::gdn_rms_norm_gated'"
        ") -> Tensor");

  m.def("gdn_solve_tril("
        "Tensor A, Tensor cu_seqlens, Tensor chunk_indices, "
        "*, str zentorch_op_name='zentorch::gdn_solve_tril'"
        ") -> Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("gdn_causal_conv1d_fn", &zentorch::zentorch_gdn_causal_conv1d_fn);

  m.impl("gdn_causal_conv1d_update",
         &zentorch::zentorch_gdn_causal_conv1d_update);

  m.impl("gdn_chunk_fwd_o", &zentorch::zentorch_gdn_chunk_fwd_o);

  m.impl("gdn_chunk_gated_delta_rule_fwd",
         &zentorch::zentorch_gdn_chunk_gated_delta_rule_fwd);

  m.impl("gdn_chunk_gated_delta_rule_fwd_h",
         &zentorch::zentorch_gdn_chunk_gated_delta_rule_fwd_h);

  m.impl("gdn_chunk_local_cumsum", &zentorch::zentorch_gdn_chunk_local_cumsum);

  m.impl("gdn_chunk_scaled_dot_kkt_fwd",
         &zentorch::zentorch_gdn_chunk_scaled_dot_kkt_fwd);

  m.impl("gdn_fused_post_conv_prep",
         &zentorch::zentorch_gdn_fused_post_conv_prep);

  m.impl(
      "gdn_fused_recurrent_gated_delta_rule_packed_decode",
      &zentorch::zentorch_gdn_fused_recurrent_gated_delta_rule_packed_decode);

  m.impl("gdn_fused_sigmoid_gating_delta_rule_update",
         &zentorch::zentorch_gdn_fused_sigmoid_gating_delta_rule_update);

  m.impl("gdn_l2norm_fwd", &zentorch::zentorch_gdn_l2norm_fwd);

  m.impl("gdn_recompute_w_u_fwd", &zentorch::zentorch_gdn_recompute_w_u_fwd);

  m.impl("gdn_rms_norm_gated", &zentorch::zentorch_gdn_rms_norm_gated);

  m.impl("gdn_solve_tril", &zentorch::zentorch_gdn_solve_tril);
}
