/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "MatmulUtils.hpp"
#include "Memory.hpp"
#include "Ops.hpp"
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

  torch_checks_for_woq_linear(input, qweight, weight_scales, group_size,
                              weight_bits, compute_dtype, unpacking_ratio);

  auto input_contiguous = input.is_contiguous() ? input : input.contiguous();

  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor &bias_t = *bias_maybe_owned;
  const bool bias_defined = bias_t.defined();
  if (bias_defined) {
    LOG(INFO) << "bias dimensions: " << bias_t.sizes();
    TORCH_CHECK(
        bias_t.dim() == 1 &&
            bias_t.size(0) == (qweight.size(1) * unpacking_ratio),
        "zentorch_woq_linear_impl: incorrect dimensions/shape for bias");
  }

  c10::MaybeOwned<at::Tensor> weight_zero_point_maybe_owned =
      at::borrow_from_optional_tensor(weight_zero_point);
  const at::Tensor &weight_zero_point_t = *weight_zero_point_maybe_owned;
  const bool weight_zero_point_defined = weight_zero_point_t.defined();

  if (weight_zero_point_defined) {
    LOG(INFO) << "weight_zero_point dimensions: "
              << weight_zero_point_t.sizes();
    TORCH_CHECK(weight_zero_point_t.dim() == 2 &&
                    weight_zero_point_t.size(0) == 1 &&
                    weight_zero_point_t.size(1) == qweight.size(1),
                "zentorch_woq_linear_impl: incorrect dimensions/shape for "
                "weight_zero_point");
    // TODO: to be tested for perf impact with group size not being -1
    TORCH_CHECK(are_all_zeros(weight_zero_point_t),
                "zentorch_woq_linear_impl: non-zero weight_zero_point "
                "are not supported yet");
  }

  memory z_input, z_bias, z_qweight, z_result, z_woq_scales;
  aten_tensor_to_zen_memory_for_woq_linear(
      input_contiguous, qweight, weight_scales, bias_defined, bias_t, result,
      group_size, unpacking_ratio, z_input, z_qweight, z_bias, z_result,
      z_woq_scales);
  std::unordered_map<int, memory> execute_args;
  zendnn::primitive_attr op_attr;
  post_ops po;
  // Setting Post ops
  // pass woq as true
  zentorch_post_ops_selection(po, execute_args, post_op_ids, post_op_buffers,
                              /*woq*/ true);
  op_attr.set_post_ops(po);
  op_attr.set_plugin_op_name(zentorch_op_name);

  // set woq weight scales
  // group_size = -1 represents that weight is quantized with
  // per-channel quantization config
  if (group_size == -1) {
    // for per-channel scales, mask = 2 is used
    // TODO: support per-tensor/per-group scales
    op_attr.set_woq_scale(2, {ZENDNN_RUNTIME_F32_VAL});
    execute_args.insert({ZENDNN_ARG_ATTR_WOQ_SCALES, z_woq_scales});
  } else {
    TORCH_CHECK(
        false,
        "zentorch_woq_linear_impl: currently only group_size = -1 is supported")
  }

  // execute the zendnn::matmul kernel
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
  // create result tensor
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

template <BINARY_POST_OP fuse>
at::Tensor zentorch_woq_linear_binary(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &weight_scales,
    const c10::optional<at::Tensor> &weight_zero_point,
    const c10::optional<at::Tensor> &bias, const at::Tensor &binary_input,
    const int64_t &group_size, const int64_t &weight_bits,
    const std::string &compute_dtype, std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  const int64_t unpacking_ratio = get_unpacking_ratio(qweight, weight_bits);

  TORCH_CHECK(binary_input.sizes() ==
                  c10::IntArrayRef(get_matmul_and_linear_output_sizes(
                      input, qweight, unpacking_ratio)),
              "zentorch_woq_linear_binary: unsupported sizes for woq_linear "
              "result and binary_input");

  at::Tensor result = at::empty(binary_input.sizes(), binary_input.options());

  LOG(INFO) << "Calling  zentorch_woq_linear_impl from " << __FUNCTION__
            << "!\n";

  std::vector<at::Tensor> post_op_buffers = {binary_input};
  std::vector<int64_t> post_op_ids = {fuse};

  return zentorch_woq_linear_impl(
      input, qweight, weight_scales, weight_zero_point, bias, result,
      post_op_ids, post_op_buffers, group_size, weight_bits, compute_dtype,
      unpacking_ratio, zentorch_op_name);
}

at::Tensor zentorch_woq_linear_silu_mul(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &weight_scales,
    const c10::optional<at::Tensor> &weight_zero_point,
    const c10::optional<at::Tensor> &bias, const at::Tensor &mul_input,
    const int64_t &group_size, const int64_t &weight_bits,
    const std::string &compute_dtype, std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  const int64_t unpacking_ratio = get_unpacking_ratio(qweight, weight_bits);

  TORCH_CHECK(mul_input.sizes() ==
                  c10::IntArrayRef(get_matmul_and_linear_output_sizes(
                      input, qweight, unpacking_ratio)),
              " zentorch_woq_linear_silu_mul: unsupported sizes for woq_linear "
              "result and mul_input");

  at::Tensor result = at::empty(mul_input.sizes(), mul_input.options());

  LOG(INFO) << "Calling zentorch_woq_linear_impl from " << __FUNCTION__
            << "!\n";

  std::vector<at::Tensor> post_op_buffers = {mul_input};
  std::vector<int64_t> post_op_ids = {UNARY_POST_OP::SILU, BINARY_POST_OP::MUL};

  return zentorch_woq_linear_impl(
      input, qweight, weight_scales, weight_zero_point, bias, result,
      post_op_ids, post_op_buffers, group_size, weight_bits, compute_dtype,
      unpacking_ratio, zentorch_op_name);
}

at::Tensor zentorch_woq_linear_add_add(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &weight_scales,
    const c10::optional<at::Tensor> &weight_zero_point,
    const c10::optional<at::Tensor> &bias, const at::Tensor &add1_input,
    const at::Tensor &add2_input, const int64_t &group_size,
    const int64_t &weight_bits, const std::string &compute_dtype,
    std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  const int64_t unpacking_ratio = get_unpacking_ratio(qweight, weight_bits);

  TORCH_CHECK((add1_input.sizes() ==
               c10::IntArrayRef(get_matmul_and_linear_output_sizes(
                   input, qweight, unpacking_ratio))) &&
                  (add2_input.sizes() ==
                   c10::IntArrayRef(get_matmul_and_linear_output_sizes(
                       input, qweight, unpacking_ratio))),
              "zentorch_woq_linear_add_add: unsupported sizes for woq_linear "
              "result, add1_input and add2_input");

  at::Tensor result = at::empty(add2_input.sizes(), add2_input.options());

  LOG(INFO) << "Calling  zentorch_woq_linear_impl from " << __FUNCTION__
            << "!\n";

  std::vector<at::Tensor> post_op_buffers = {add1_input, add2_input};
  std::vector<int64_t> post_op_ids = {BINARY_POST_OP::ADD, BINARY_POST_OP::ADD};

  return zentorch_woq_linear_impl(
      input, qweight, weight_scales, weight_zero_point, bias, result,
      post_op_ids, post_op_buffers, group_size, weight_bits, compute_dtype,
      unpacking_ratio, zentorch_op_name);
}

// Template instantiations
// The "zentorch_woq_linear" instantiations
// No post-op
template at::Tensor zentorch_woq_linear<UNARY_POST_OP::POST_OP_NONE>(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &weight_scales,
    const c10::optional<at::Tensor> &weight_zero_point,
    const c10::optional<at::Tensor> &bias, const int64_t &group_size,
    const int64_t &weight_bits, const std::string &compute_dtype,
    std::string zentorch_op_name);
// Post op RELU
template at::Tensor zentorch_woq_linear<UNARY_POST_OP::RELU>(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &weight_scales,
    const c10::optional<at::Tensor> &weight_zero_point,
    const c10::optional<at::Tensor> &bias, const int64_t &group_size,
    const int64_t &weight_bits, const std::string &compute_dtype,
    std::string zentorch_op_name);
// Post op SILU
template at::Tensor zentorch_woq_linear<UNARY_POST_OP::SILU>(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &weight_scales,
    const c10::optional<at::Tensor> &weight_zero_point,
    const c10::optional<at::Tensor> &bias, const int64_t &group_size,
    const int64_t &weight_bits, const std::string &compute_dtype,
    std::string zentorch_op_name);
// Post op GELU ERF
template at::Tensor zentorch_woq_linear<UNARY_POST_OP::GELU_ERF>(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &weight_scales,
    const c10::optional<at::Tensor> &weight_zero_point,
    const c10::optional<at::Tensor> &bias, const int64_t &group_size,
    const int64_t &weight_bits, const std::string &compute_dtype,
    std::string zentorch_op_name);
// Post op GELU TANH
template at::Tensor zentorch_woq_linear<UNARY_POST_OP::GELU_TANH>(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &weight_scales,
    const c10::optional<at::Tensor> &weight_zero_point,
    const c10::optional<at::Tensor> &bias, const int64_t &group_size,
    const int64_t &weight_bits, const std::string &compute_dtype,
    std::string zentorch_op_name);

template at::Tensor zentorch_woq_linear_binary<BINARY_POST_OP::ADD>(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &weight_scales,
    const c10::optional<at::Tensor> &weight_zero_point,
    const c10::optional<at::Tensor> &bias, const at::Tensor &binary_input,
    const int64_t &group_size, const int64_t &weight_bits,
    const std::string &compute_dtype, std::string zentorch_op_name);
} // namespace zentorch
