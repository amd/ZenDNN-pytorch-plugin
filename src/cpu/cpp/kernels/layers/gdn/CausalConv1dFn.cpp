/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "../../../Utils.hpp"

#include <ATen/ops/cat.h>
#include <ATen/ops/conv1d.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/zeros.h>
#include <ATen/record_function.h>
#include <c10/util/Optional.h>
#include <torch/all.h>

#include <cstdint>
#include <string>

namespace zentorch {

at::Tensor zentorch_gdn_causal_conv1d_fn(
    const at::Tensor &x, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, at::Tensor &conv_states,
    const at::Tensor &query_start_loc, const at::Tensor &cache_indices,
    const at::Tensor &has_initial_state, std::string activation,
    int64_t pad_slot_id, std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::gdn_causal_conv1d_fn",
                  c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(x.dim() == 2, "x must be 2-D (dim, cu_seqlen)");
  ZENTORCH_CHECK(weight.dim() == 2, "weight must be 2-D (dim, width)");
  ZENTORCH_CHECK(conv_states.dim() == 3,
                 "conv_states must be 3-D (num_cache_lines, dim, state_len)");

  const int64_t dim = x.size(0);
  const int64_t cu_seqlen = x.size(1);
  const int64_t width = weight.size(1);
  const int64_t state_len = width - 1;

  ZENTORCH_CHECK(weight.size(0) == dim, "weight.size(0) must match dim=", dim);
  ZENTORCH_CHECK(width >= 2,
                 "weight width must be >= 2 (1-tap conv is degenerate and "
                 "unsupported); got ",
                 width);
  ZENTORCH_CHECK(conv_states.size(1) == dim,
                 "conv_states.size(1) must match dim=", dim);
  ZENTORCH_CHECK(conv_states.size(2) >= state_len,
                 "conv_states.size(2) must be >= width-1=", state_len);

  ZENTORCH_CHECK(query_start_loc.dim() == 1 &&
                     query_start_loc.scalar_type() == c10::ScalarType::Int,
                 "query_start_loc must be 1-D int32");
  ZENTORCH_CHECK(query_start_loc.is_contiguous(),
                 "query_start_loc must be contiguous");
  const int64_t batch = query_start_loc.size(0) - 1;
  ZENTORCH_CHECK(batch >= 0, "query_start_loc must have at least 1 entry");

  ZENTORCH_CHECK(cache_indices.dim() == 1 && cache_indices.size(0) == batch,
                 "cache_indices must be 1-D of length batch=", batch);
  ZENTORCH_CHECK(has_initial_state.dim() == 1 &&
                     has_initial_state.size(0) == batch,
                 "has_initial_state must be 1-D of length batch=", batch);

  if (bias.has_value()) {
    ZENTORCH_CHECK(bias->dim() == 1 && bias->size(0) == dim,
                   "bias must be 1-D of size dim=", dim);
  }

  ZENTORCH_CHECK(at::isFloatingType(x.scalar_type()),
                 "x must be floating-point; got ", x.scalar_type());
  ZENTORCH_CHECK(x.scalar_type() != c10::ScalarType::Half,
                 "fp16 not supported; use fp32 or bf16");

  const bool is_silu = (activation == "silu" || activation == "swish");
  const bool is_none = activation.empty();
  ZENTORCH_CHECK(is_silu || is_none,
                 "activation must be 'silu', 'swish', or '' (none); got '",
                 activation, "'");

  const auto out_dtype = x.scalar_type();
  const auto compute_dtype = conv_states.scalar_type();

  // Initialise out from x so pad-slot sequences preserve their input
  // (mirrors gdn_causal_conv1d_update which clones x_3d before the masked
  // overwrite). Without this, the pad_slot_id `continue` below would leave
  // out[:, start:end] uninitialised for any skipped non-empty sequence.
  at::Tensor out = x.clone();
  if (batch == 0 || cu_seqlen == 0 || dim == 0) {
    return out;
  }

  at::Tensor x_compute = x.to(compute_dtype);
  at::Tensor weight_compute = weight.to(compute_dtype);
  c10::optional<at::Tensor> bias_compute;
  if (bias.has_value()) {
    bias_compute = bias->to(compute_dtype);
  }
  at::Tensor weight_for_conv = weight_compute.unsqueeze(1);

  const int32_t *qsl_p = query_start_loc.const_data_ptr<int32_t>();

  at::Tensor cache_indices_long = cache_indices.to(at::kLong).contiguous();
  at::Tensor has_initial_state_bool =
      has_initial_state.to(at::kBool).contiguous();
  const int64_t *ci_p = cache_indices_long.const_data_ptr<int64_t>();
  const bool *his_p = has_initial_state_bool.const_data_ptr<bool>();

  for (int64_t b = 0; b < batch; ++b) {
    const int64_t start = qsl_p[b];
    const int64_t end = qsl_p[b + 1];
    const int64_t T_b = end - start;
    if (T_b <= 0)
      continue;

    const int64_t slot = ci_p[b];
    if (slot == pad_slot_id)
      continue;

    at::Tensor x_b = x_compute.narrow(1, start, T_b);

    at::Tensor state;
    if (his_p[b]) {
      state = conv_states.select(0, slot)
                  .narrow(-1, conv_states.size(2) - state_len, state_len)
                  .to(compute_dtype);
    } else {
      state = at::zeros({dim, state_len}, x_compute.options());
    }

    at::Tensor padded = at::cat({state, x_b}, /*dim=*/-1);

    at::Tensor padded_3d = padded.unsqueeze(0);
    at::Tensor conv_out = at::conv1d(padded_3d, weight_for_conv, bias_compute,
                                     /*stride=*/at::IntArrayRef{1},
                                     /*padding=*/at::IntArrayRef{0},
                                     /*dilation=*/at::IntArrayRef{1},
                                     /*groups=*/dim)
                              .squeeze(0);

    if (is_silu) {
      conv_out = at::silu(conv_out);
    }

    out.narrow(1, start, T_b).copy_(conv_out.to(out_dtype));

    if (state_len > 0) {
      conv_states.select(0, slot)
          .narrow(-1, conv_states.size(2) - state_len, state_len)
          .copy_(padded.narrow(-1, padded.size(-1) - state_len, state_len)
                     .to(conv_states.scalar_type()));
    }
  }

  return out;
}

} // namespace zentorch
