/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "RMS_norm.hpp"
#include "MatmulUtils.hpp"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

using namespace zendnnl::interface;
using namespace zendnnl::lowoha::normalization;

namespace zentorch {

// `input` is a mutable ref because this impl is shared with the in-place
// add path (`zentorch_add_rms_norm_`), which writes the normalized result
// back through `input.data_ptr()`. The non-mutating `zentorch_rms_norm`
// caller passes a cheap non-const alias of `input` (it only reads it in the
// result path).
static void
zentorch_rms_norm_impl(torch::stable::Tensor &input,
                       const torch::stable::Tensor &weight,
                       const std::optional<torch::stable::Tensor> &result,
                       const std::optional<torch::stable::Tensor> &residual,
                       const double &epsilon) {
  norm_params params;
  params.batch = static_cast<uint64_t>(input.size(0));
  params.norm_size = static_cast<uint64_t>(input.size(-1));
  params.norm_type =
      residual ? norm_type_t::FUSED_ADD_RMS_NORM : norm_type_t::RMS_NORM;
  params.src_dt = get_zendnnl_dtype(input);
  params.dst_dt = get_zendnnl_dtype(input);
  params.gamma_dt = get_zendnnl_dtype(weight);
  params.epsilon = epsilon;
  params.use_scale = true;
  if (result) {
    normalization_direct(input.data_ptr(), result->data_ptr(),
                         weight.data_ptr(),
                         /*beta=*/nullptr,
                         /*running_mean=*/nullptr, /*running_var=*/nullptr,
                         residual ? residual->data_ptr() : nullptr, params);
  } else {
    normalization_direct(input.data_ptr(), input.data_ptr(), weight.data_ptr(),
                         /*beta=*/nullptr,
                         /*running_mean=*/nullptr, /*running_var=*/nullptr,
                         residual ? residual->data_ptr() : nullptr, params);
  }
}

void zentorch_add_rms_norm_(torch::stable::Tensor &input,
                            const torch::stable::Tensor &weight,
                            torch::stable::Tensor &residual,
                            const double &epsilon,
                            const std::string &zentorch_op_name) {
  zentorch_rms_norm_impl(input, weight, std::nullopt, residual, epsilon);
}

torch::stable::Tensor zentorch_rms_norm(const torch::stable::Tensor &input,
                                        const torch::stable::Tensor &weight,
                                        const double &epsilon,
                                        const std::string &zentorch_op_name) {
  torch::stable::Tensor result = torch::stable::new_empty(input, input.sizes());
  // The shared impl takes a mutable `input` for its in-place add path; here
  // (result path) it only reads `input`. Use a cheap non-const alias (shares
  // the same TensorImpl -- no data copy) instead of casting away constness.
  torch::stable::Tensor input_mut = input;
  zentorch_rms_norm_impl(input_mut, weight, result, std::nullopt, epsilon);
  return result;
}

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_add_rms_norm_(Tensor(a!) input, Tensor weight, Tensor(b!) "
        "residual, "
        "float epsilon, *, str "
        "zentorch_op_name='zentorch::zentorch_add_rms_norm')"
        "-> ()");
  m.def("zentorch_rms_norm(Tensor input, Tensor weight, float epsilon, *, "
        "str "
        "zentorch_op_name='zentorch::zentorch_rms_norm') -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_add_rms_norm_",
         TORCH_BOX(&zentorch::zentorch_add_rms_norm_));
  m.impl("zentorch_rms_norm", TORCH_BOX(&zentorch::zentorch_rms_norm));
}
} // namespace zentorch
