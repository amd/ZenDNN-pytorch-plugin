/*****************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "MatmulUtils.hpp"
#include "Ops.hpp"
#include <ATen/record_function.h>

using namespace zendnnl::interface;
using namespace zendnnl::lowoha::normalization;

namespace zentorch {

void zentorch_rms_norm_impl(at::Tensor &input, const at::Tensor &weight,
                            const c10::optional<at::Tensor> &result,
                            const c10::optional<at::Tensor> &residual,
                            const double epsilon,
                            std::string zentorch_op_name) {
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

void zentorch_add_rms_norm_(at::Tensor &input, const at::Tensor &weight,
                            at::Tensor &residual, const double epsilon,
                            std::string zentorch_op_name) {
  zentorch_rms_norm_impl(input, weight, c10::nullopt, residual, epsilon,
                         zentorch_op_name);
}

at::Tensor zentorch_rms_norm(at::Tensor &input, const at::Tensor &weight,
                             const double epsilon,
                             std::string zentorch_op_name) {
  at::Tensor result = at::detail::empty_strided_cpu(
      input.sizes(), input.strides(), input.options());
  zentorch_rms_norm_impl(input, weight, result, c10::nullopt, epsilon,
                         zentorch_op_name);
  return result;
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_add_rms_norm_(Tensor(a!) input, Tensor weight, Tensor(b!) "
        "residual, "
        "float epsilon, *, str "
        "zentorch_op_name='zentorch::zentorch_add_rms_norm')"
        "-> ()");
  m.def("zentorch_rms_norm(Tensor input, Tensor weight, float epsilon, *, "
        "str "
        "zentorch_op_name='zentorch::zentorch_rms_norm') -> Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_add_rms_norm_", zentorch_add_rms_norm_);
  m.impl("zentorch_rms_norm", zentorch_rms_norm);
}
} // namespace zentorch
