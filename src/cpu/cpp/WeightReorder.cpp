/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "Memory.hpp"
#include "zendnnl.hpp"

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

using namespace zendnnl::interface;

inline bool is_tensor_2d_and_transposed(const at::Tensor &t) {
  if (t.dim() == 2) {
    return t.strides()[0] == 1 && t.strides()[1] == t.sizes()[0];
  }
  return false;
}

at::Tensor zentorch_weight_prepack_for_linear(const at::Tensor &weight,
                                              std::string_view zendnn_op_name) {
  ZENTORCH_CHECK(weight.dim() == 2,
                 "Weight tensor must be 2D for linear layer prepacking, got ",
                 weight.dim(), "D tensor.");
  ZENTORCH_CHECK(
      weight.scalar_type() == c10::ScalarType::Float ||
          weight.scalar_type() == c10::ScalarType::BFloat16,
      "Currently weight prepacking only supports float32 or bfloat16 "
      "dtype for weight tensor");

  // Linear op internally works on transposed weight tensor, so to
  // prepack the weight we need to use transposed weight.
  auto reorder_input = weight.t();
  tensor_t zen_reorder_input;
  set_zendnnl_tensor_attributes(reorder_input, zen_reorder_input,
                                "reorder_input");
  // TODO: Create constexpr for all strings used.
  // Currently, ZenDNN only supports blocked layout with AOCL backend.
  auto context = reorder_context_t().set_algo_format("aocl").create();
  ZENTORCH_CHECK(context.check(), "reorder context creation failed.");

  auto reorder_op =
      reorder_operator_t().set_name("reorder_op").set_context(context).create();
  // Check if reorder operation creation is successful.
  ZENTORCH_CHECK(reorder_op.check(), "operator ", reorder_op.get_name(),
                 " creation failed.");

  reorder_op.set_input("reorder_input", zen_reorder_input);
  size_t reorder_bytes = reorder_op.get_reorder_size();
  int64_t num_elements = reorder_bytes / weight.element_size();
  // Create 1d tensor to hold the reordered weights with
  // a stride of 1 to ensure contiguous memory layout.
  std::vector<long unsigned int> reorder_output_sizes(
      reorder_input.sizes().begin(), reorder_input.sizes().end());
  std::vector<long unsigned int> reorder_output_strides(
      reorder_input.strides().begin(), reorder_input.strides().end());
  at::Tensor reorder_output = at::detail::empty_strided_cpu(
      /*size*/ {num_elements}, /*stride*/ {1}, weight.options());

  tensor_t zen_reorder_output;
  if (is_tensor_2d_and_transposed(reorder_input)) {
    zen_reorder_output.set_order("ba");
  }
  set_zendnnl_tensor_attributes(reorder_output, zen_reorder_output,
                                "reorder_output",
                                true /* is_weight_prepacked */,
                                reorder_output_sizes, reorder_output_strides);

  reorder_op.set_output("reorder_output", zen_reorder_output);
  reorder_op.execute();
  return at::as_strided(reorder_output, weight.sizes(), weight.strides());
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_weight_reorder_for_matmul(Tensor weight, bool "
        "is_weight_oc_x_ic=True) -> Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_weight_reorder_for_matmul",
         zentorch::zentorch_weight_reorder_for_matmul);
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_weight_prepack_for_linear(Tensor weight, "
        "str zentorch_op_name='zentorch::zentorch_weight_prepack_for_linear') "
        "-> Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_weight_prepack_for_linear",
         zentorch::zentorch_weight_prepack_for_linear);
}

} // namespace zentorch
