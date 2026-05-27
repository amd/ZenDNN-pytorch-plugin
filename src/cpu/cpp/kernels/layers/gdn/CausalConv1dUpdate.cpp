/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "../../../Utils.hpp"

#include <ATen/ops/cat.h>
#include <ATen/ops/conv1d.h>
#include <ATen/ops/silu.h>
#include <ATen/record_function.h>
#include <c10/util/Optional.h>
#include <torch/all.h>

#include <string>

namespace zentorch {

at::Tensor zentorch_gdn_causal_conv1d_update(
    const at::Tensor &x, at::Tensor &conv_state, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, std::string activation,
    const at::Tensor &conv_state_indices, int64_t null_block_id,
    int64_t pad_slot_id, std::string zentorch_op_name) {
  RECORD_FUNCTION("zentorch::gdn_causal_conv1d_update",
                  c10::ArrayRef<c10::IValue>({}));

  ZENTORCH_CHECK(x.dim() == 2 || x.dim() == 3,
                 "x must be 2-D (batch, dim) or 3-D (batch, dim, seqlen)");
  ZENTORCH_CHECK(weight.dim() == 2, "weight must be 2-D (dim, width)");
  ZENTORCH_CHECK(conv_state.dim() == 3,
                 "conv_state must be 3-D (num_cache_lines, dim, state_len)");

  const bool is_2d = (x.dim() == 2);
  const int64_t batch = x.size(0);
  const int64_t dim = x.size(1);

  const int64_t weight_dim = weight.size(0);
  const int64_t width = weight.size(1);
  const int64_t state_len = width - 1;

  ZENTORCH_CHECK(weight_dim == dim, "weight.size(0)=", weight_dim,
                 " must match dim=", dim);
  ZENTORCH_CHECK(width >= 2,
                 "weight width must be >= 2 (1-tap conv is degenerate and "
                 "unsupported); got ",
                 width);
  ZENTORCH_CHECK(conv_state.size(1) == dim,
                 "conv_state.size(1) must match dim=", dim);
  ZENTORCH_CHECK(conv_state.size(2) >= state_len,
                 "conv_state.size(2) must be >= width-1=", state_len);
  if (bias.has_value()) {
    ZENTORCH_CHECK(bias->dim() == 1 && bias->size(0) == dim,
                   "bias must be 1-D of size dim=", dim);
  }
  ZENTORCH_CHECK(conv_state_indices.dim() == 1 &&
                     conv_state_indices.size(0) == batch,
                 "conv_state_indices must be 1-D of length batch=", batch);

  ZENTORCH_CHECK(at::isFloatingType(x.scalar_type()),
                 "x must be floating-point; got ", x.scalar_type());
  ZENTORCH_CHECK(x.scalar_type() != c10::ScalarType::Half,
                 "fp16 not supported; use fp32 or bf16");

  const bool is_silu = (activation == "silu" || activation == "swish");
  const bool is_none = activation.empty();
  ZENTORCH_CHECK(is_silu || is_none,
                 "activation must be 'silu', 'swish', or '' (none); got '",
                 activation, "'");

  const auto compute_dtype = conv_state.scalar_type();
  const auto out_dtype = x.scalar_type();

  // Clone x_3d in compute_dtype so skipped slots retain original x values.
  at::Tensor x_3d = is_2d ? x.unsqueeze(-1) : x;
  at::Tensor out_3d = x_3d.to(compute_dtype).clone();

  at::Tensor null_t =
      at::scalar_tensor(null_block_id, conv_state_indices.options());
  at::Tensor pad_t =
      at::scalar_tensor(pad_slot_id, conv_state_indices.options());
  at::Tensor valid_mask =
      conv_state_indices.ne(null_t).logical_and(conv_state_indices.ne(pad_t));

  if (!valid_mask.any().item<bool>()) {
    at::Tensor out = is_2d ? out_3d.squeeze(-1) : out_3d;
    return out.to(out_dtype);
  }

  at::Tensor valid_slots =
      conv_state_indices.masked_select(valid_mask).to(at::kLong);
  at::Tensor selected_x = x_3d.index({valid_mask}).to(compute_dtype);

  at::Tensor state_used =
      conv_state
          .index({valid_slots, at::indexing::Slice(),
                  at::indexing::Slice(-state_len, c10::nullopt)})
          .to(compute_dtype);

  at::Tensor padded = at::cat({state_used, selected_x}, /*dim=*/-1);

  at::Tensor weight_for_conv = weight.unsqueeze(1).to(compute_dtype);
  c10::optional<at::Tensor> bias_compute;
  if (bias.has_value()) {
    bias_compute = bias->to(compute_dtype);
  }
  // Explicit `at::IntArrayRef{...}` so g++ picks the IntArrayRef overload
  // (the bare `0` literal is ambiguous with the string_view overload).
  at::Tensor out_valid = at::conv1d(padded, weight_for_conv, bias_compute,
                                    /*stride=*/at::IntArrayRef{1},
                                    /*padding=*/at::IntArrayRef{0},
                                    /*dilation=*/at::IntArrayRef{1},
                                    /*groups=*/dim);

  if (is_silu) {
    out_valid = at::silu(out_valid);
  }

  out_3d.index_put_({valid_mask}, out_valid);

  at::Tensor new_state =
      padded.index({at::indexing::Slice(), at::indexing::Slice(),
                    at::indexing::Slice(-state_len, c10::nullopt)});
  conv_state.index_put_({valid_slots, at::indexing::Slice(),
                         at::indexing::Slice(-state_len, c10::nullopt)},
                        new_state.to(conv_state.scalar_type()));

  at::Tensor out = is_2d ? out_3d.squeeze(-1) : out_3d;
  return out.to(out_dtype);
}

} // namespace zentorch
