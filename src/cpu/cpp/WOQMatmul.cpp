/******************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "MatmulUtils.hpp"
#include "Memory.hpp"
#include "WOQMatmulUtils.hpp"

namespace zentorch {

using namespace zendnn;
// TODO: Change function return type to void
at::Tensor zentorch_woq_linear_impl(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &weight_scales,
    const c10::optional<at::Tensor> &weight_zero_point,
    const c10::optional<at::Tensor> &bias, at::Tensor result,
    const std::vector<int64_t> &post_op_ids,
    const std::vector<at::Tensor> &post_op_buffers, const int64_t &group_size,
    const int64_t &weight_bits, const std::string &compute_dtype,
    const int64_t &unpacking_ratio, std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  LOG(INFO) << "input dimensions: " << input.sizes();
  LOG(INFO) << "qweight dimensions: " << qweight.sizes();
  LOG(INFO) << "weight_scales dimensions: " << weight_scales.sizes();
  LOG(INFO) << "group_size : " << group_size
            << " and weight_bits : " << weight_bits;
  LOG(INFO) << "result dimensions: " << result.sizes();
  LOG(INFO) << "Unpacking ratio : " << unpacking_ratio;

  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor &bias_t = *bias_maybe_owned;
  const bool bias_defined = bias_t.defined();

  c10::MaybeOwned<at::Tensor> weight_zero_point_maybe_owned =
      at::borrow_from_optional_tensor(weight_zero_point);
  const at::Tensor &weight_zero_point_t = *weight_zero_point_maybe_owned;

  checks_for_woq_linear(input, qweight, bias_t, result, weight_scales,
                        weight_zero_point_t, post_op_buffers, group_size,
                        weight_bits, compute_dtype, unpacking_ratio);

  auto input_contiguous = input.is_contiguous() ? input : input.contiguous();

  memory z_input, z_bias, z_qweight, z_result, z_woq_scales;
  aten_tensor_to_zen_memory_for_woq_linear(
      input_contiguous, qweight, weight_scales, bias_defined, bias_t, result,
      group_size, unpacking_ratio, z_input, z_qweight, z_bias, z_result,
      z_woq_scales);
  std::unordered_map<int, memory> execute_args;
  zendnn::primitive_attr op_attr;
  post_ops po;
  // Setting Post ops
  // Pass woq as true
  zentorch_post_ops_selection(po, execute_args, post_op_ids, post_op_buffers,
                              /*woq*/ true);
  op_attr.set_post_ops(po);
  op_attr.set_plugin_op_name(zentorch_op_name);

  // Set woq weight scales
  // group_size = -1 represents that weight is quantized with
  // per-channel quantization config.
  // Also when group_size is equal to qweight's input channel size
  // then this will be the case of per-channel granularity.
  if (group_size == -1 || group_size == qweight.size(0)) {
    // For per-channel granular scales, mask is mapped to
    // QUANT_GRANULARITY::PER_CHANNEL.
    LOG(INFO) << "Setting quant granularity to per-channel for woq scales";
    op_attr.set_woq_scale(QUANT_GRANULARITY::PER_CHANNEL, {1, 1},
                          z_woq_scales.get_desc().data_type());
  } else {
    // TODO: Support per-tensor scales
    // For per-group granular scales, mask is mapped to
    // QUANT_GRANULARITY::PER_GROUP.
    LOG(INFO) << "Setting quant granularity to per-group for woq scales";
    op_attr.set_woq_scale(QUANT_GRANULARITY::PER_GROUP, {group_size, 1},
                          z_woq_scales.get_desc().data_type());
  }
  execute_args.insert({ZENDNN_ARG_ATTR_WOQ_SCALES, z_woq_scales});

  // Execute the zendnn::matmul kernel
  zentorch_matmul_execute(execute_args, z_input, z_qweight, z_bias, z_result,
                          op_attr, bias_defined);

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
  return result;
}

template <UNARY_POST_OP fuse>
at::Tensor
zentorch_woq_linear(const at::Tensor &input, const at::Tensor &qweight,
                    const at::Tensor &weight_scales,
                    const c10::optional<at::Tensor> &weight_zero_point,
                    const c10::optional<at::Tensor> &bias,
                    const int64_t &group_size, const int64_t &weight_bits,
                    const std::string &compute_dtype,
                    std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  const int64_t unpacking_ratio = get_unpacking_ratio(qweight, weight_bits);

  // qweight is packed along the output_channel with the packing_ratio,
  // so the result tensor is created with input dims upto 2nd last dim &
  // unpacked last dim of qweight tensor(qweight.last_dim * unpacking_ratio)
  // create result tensor.
  at::Tensor result = at::empty(
      get_matmul_and_linear_output_sizes(input, qweight, unpacking_ratio),
      input.options());

  std::vector<at::Tensor> post_op_buffers = {};
  std::vector<int64_t> post_op_ids = {fuse};
  LOG(INFO) << "Calling zentorch_woq_linear_impl from " << __FUNCTION__
            << "!\n";
  return zentorch_woq_linear_impl(
      input, qweight, weight_scales, weight_zero_point, bias, result,
      post_op_ids, post_op_buffers, group_size, weight_bits, compute_dtype,
      unpacking_ratio, zentorch_op_name);
}

// unary-binary fusions and binary fusions will be handled by this
template <UNARY_POST_OP fuse1, BINARY_POST_OP fuse2>
at::Tensor zentorch_woq_linear_unary_binary(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &weight_scales,
    const c10::optional<at::Tensor> &weight_zero_point,
    const c10::optional<at::Tensor> &bias, const at::Tensor &binary_input,
    const int64_t &group_size, const int64_t &weight_bits,
    const std::string &compute_dtype, std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  const int64_t unpacking_ratio = get_unpacking_ratio(qweight, weight_bits);

  at::Tensor result = at::empty(binary_input.sizes(), binary_input.options());

  std::vector<at::Tensor> post_op_buffers = {binary_input};
  std::vector<int64_t> post_op_ids;
  if (fuse1 != UNARY_POST_OP::POST_OP_NONE)
    post_op_ids.push_back(fuse1);
  post_op_ids.push_back(fuse2);

  LOG(INFO) << "Calling  zentorch_woq_linear_impl from " << __FUNCTION__
            << "!\n";
  return zentorch_woq_linear_impl(
      input, qweight, weight_scales, weight_zero_point, bias, result,
      post_op_ids, post_op_buffers, group_size, weight_bits, compute_dtype,
      unpacking_ratio, zentorch_op_name);
}

template <BINARY_POST_OP fuse1, BINARY_POST_OP fuse2>
at::Tensor zentorch_woq_linear_binary_binary(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &weight_scales,
    const c10::optional<at::Tensor> &weight_zero_point,
    const c10::optional<at::Tensor> &bias, const at::Tensor &binary1_input,
    const at::Tensor &binary2_input, const int64_t &group_size,
    const int64_t &weight_bits, const std::string &compute_dtype,
    std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  const int64_t unpacking_ratio = get_unpacking_ratio(qweight, weight_bits);

  at::Tensor result = at::empty(binary2_input.sizes(), binary2_input.options());

  std::vector<at::Tensor> post_op_buffers = {binary1_input, binary2_input};
  std::vector<int64_t> post_op_ids = {fuse1, fuse2};

  LOG(INFO) << "Calling  zentorch_woq_linear_impl from " << __FUNCTION__
            << "!\n";
  return zentorch_woq_linear_impl(
      input, qweight, weight_scales, weight_zero_point, bias, result,
      post_op_ids, post_op_buffers, group_size, weight_bits, compute_dtype,
      unpacking_ratio, zentorch_op_name);
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def(
      "zentorch_woq_linear(Tensor input, Tensor qweight, Tensor weight_scales, "
      "Tensor? weight_zero_point, Tensor? bias, int group_size, "
      "int weight_bits=4, str compute_dtype = 'bfloat16', str zentorch_op_name "
      "= 'zentorch::zentorch_woq_linear') -> Tensor");
  m.def(
      "zentorch_woq_linear_relu(Tensor input, Tensor qweight, Tensor "
      "weight_scales, "
      "Tensor? weight_zero_point, Tensor? bias, int group_size, "
      "int weight_bits=4, str compute_dtype = 'bfloat16', str zentorch_op_name "
      "= 'zentorch::zentorch_woq_linear_relu') -> Tensor");
  m.def(
      "zentorch_woq_linear_silu(Tensor input, Tensor qweight, Tensor "
      "weight_scales, "
      "Tensor? weight_zero_point, Tensor? bias, int group_size, "
      "int weight_bits=4, str compute_dtype = 'bfloat16', str zentorch_op_name "
      "= 'zentorch::zentorch_woq_linear_silu') -> Tensor");
  m.def(
      "zentorch_woq_linear_gelu_erf(Tensor input, Tensor qweight, Tensor "
      "weight_scales, "
      "Tensor? weight_zero_point, Tensor? bias, int group_size, "
      "int weight_bits=4, str compute_dtype = 'bfloat16', str zentorch_op_name "
      "= 'zentorch::zentorch_woq_linear_gelu_erf') -> Tensor");
  m.def(
      "zentorch_woq_linear_gelu_tanh(Tensor input, Tensor qweight, Tensor "
      "weight_scales, "
      "Tensor? weight_zero_point, Tensor? bias, int group_size, "
      "int weight_bits=4, str compute_dtype = 'bfloat16', str zentorch_op_name "
      "= 'zentorch::zentorch_woq_linear_gelu_tanh') -> Tensor");
  m.def(
      "zentorch_woq_linear_add(Tensor input, Tensor qweight, Tensor "
      "weight_scales, Tensor? weight_zero_point, Tensor? bias, Tensor "
      "binary_input, "
      "int group_size, int weight_bits=4, str compute_dtype = 'bfloat16', "
      "str zentorch_op_name = 'zentorch::zentorch_woq_linear_add') -> Tensor");

  m.def("zentorch_woq_linear_silu_mul(Tensor input, Tensor qweight, Tensor "
        "weight_scales, Tensor? weight_zero_point, Tensor? bias, Tensor "
        "mul_input, "
        "int group_size, int weight_bits=4, str compute_dtype = 'bfloat16', "
        "str zentorch_op_name = 'zentorch::zentorch_woq_linear_silu_mul') -> "
        "Tensor");
  m.def("zentorch_woq_linear_add_add(Tensor input, Tensor qweight, Tensor "
        "weight_scales, Tensor? weight_zero_point, Tensor? bias, Tensor "
        "add1_input, Tensor add2_input, "
        "int group_size, int weight_bits=4, str compute_dtype = 'bfloat16', "
        "str zentorch_op_name = 'zentorch::zentorch_woq_linear_add_add') -> "
        "Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_woq_linear",
         zentorch_woq_linear<UNARY_POST_OP::POST_OP_NONE>);
  m.impl("zentorch_woq_linear_relu", zentorch_woq_linear<UNARY_POST_OP::RELU>);
  m.impl("zentorch_woq_linear_silu", zentorch_woq_linear<UNARY_POST_OP::SILU>);
  m.impl("zentorch_woq_linear_gelu_erf",
         zentorch_woq_linear<UNARY_POST_OP::GELU_ERF>);
  m.impl("zentorch_woq_linear_gelu_tanh",
         zentorch_woq_linear<UNARY_POST_OP::GELU_TANH>);
  m.impl("zentorch_woq_linear_add",
         zentorch_woq_linear_unary_binary<UNARY_POST_OP::POST_OP_NONE,
                                          BINARY_POST_OP::ADD>);
  m.impl("zentorch_woq_linear_silu_mul",
         zentorch_woq_linear_unary_binary<UNARY_POST_OP::SILU,
                                          BINARY_POST_OP::MUL>);
  m.impl("zentorch_woq_linear_add_add",
         zentorch_woq_linear_binary_binary<BINARY_POST_OP::ADD,
                                           BINARY_POST_OP::ADD>);
}
} // namespace zentorch
