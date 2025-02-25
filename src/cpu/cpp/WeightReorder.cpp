/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "Memory.hpp"

namespace zentorch {

using namespace zendnn;

// Reorder the weight tensor for quantized matmul performance
// weight: 2D tensor to be reordered
// is_weight_oc_x_ic: true if the weight tensor is in OCxIC format
// Return: reordered weight tensor
//
// Normally Linear layers store the weight tensor in
// output_channel(OC) x input_channel(IC) format, but matmul
// operation works on input_channel(IC) x output_channel(OC)
// format. So we need pass is_weight_oc_x_ic as true if weight is
// in OCxIC format to apporpriately reorder the weight tensor.
inline at::Tensor
zentorch_weight_reorder_for_matmul(at::Tensor &weight,
                                   const bool &is_weight_oc_x_ic) {
  ZENTORCH_CHECK(weight.scalar_type() == at::kChar,
                 "only int8 weight is supported");
  ZENTORCH_CHECK(weight.dim() == 2,
                 "only 2-dimensional weight tensor is supported");
  ZENTORCH_CHECK(weight.is_contiguous(),
                 "reorder of weight tensor which is stored as contiguous is "
                 "only supported")

  // Execute the zendnn_custom_op::zendnn_reorder api to reorder the weight
  if (is_weight_oc_x_ic) {
    zendnn_custom_op::zendnn_reorder(/*src*/ weight.data_ptr<int8_t>(),
                                     /*dst*/ weight.data_ptr<int8_t>(),
                                     /*k*/ weight.size(1), /*n*/ weight.size(0),
                                     /*trans*/ true, /*dtype*/ zendnn_s8);
  } else {
    zendnn_custom_op::zendnn_reorder(/*src*/ weight.data_ptr<int8_t>(),
                                     /*dst*/ weight.data_ptr<int8_t>(),
                                     /*k*/ weight.size(0), /*n*/ weight.size(1),
                                     /*trans*/ false, /*dtype*/ zendnn_s8);
  }
  return weight;
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_weight_reorder_for_matmul(Tensor weight, bool "
        "is_weight_oc_x_ic=True) -> Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_weight_reorder_for_matmul",
         zentorch::zentorch_weight_reorder_for_matmul);
}

} // namespace zentorch
