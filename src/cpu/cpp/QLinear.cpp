/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "MatmulUtils.hpp"
#include "Memory.hpp"
#include "QLinearUtils.hpp"
#include "Utils.hpp"

namespace zentorch {

using namespace zendnn;

inline void zentorch_quantized_matmul_impl(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, at::Tensor &result,
    const at::Tensor &input_scales, const at::Tensor &input_zero_points,
    const at::Tensor &weight_scales, const at::Tensor &weight_zero_points,
    const std::vector<int64_t> &post_op_ids,
    const std::vector<at::Tensor> &post_op_buffers,
    const c10::optional<at::Tensor> &output_scales,
    const c10::optional<at::Tensor> &output_zero_points,
    const int64_t output_stride, std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  LOG(INFO) << "input dimensions: " << input.sizes();
  LOG(INFO) << "weight dimensions: " << weight.sizes();
  LOG(INFO) << "input_scales dimensions: " << input_scales.sizes();
  LOG(INFO) << "input_zero_points dimensions: " << input_zero_points.sizes();
  LOG(INFO) << "weight_scales dimensions: " << weight_scales.sizes();
  LOG(INFO) << "weight_zero_points dimensions: " << weight_zero_points.sizes();
  LOG(INFO) << "result dimensions: " << result.sizes();

  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor &bias_t = *bias_maybe_owned;
  const bool bias_defined = bias_t.defined();

  c10::MaybeOwned<at::Tensor> output_scales_maybe_owned =
      at::borrow_from_optional_tensor(output_scales);
  const at::Tensor &output_scales_t = *output_scales_maybe_owned;
  const bool output_scales_defined = output_scales_t.defined();

  c10::MaybeOwned<at::Tensor> output_zero_points_maybe_owned =
      at::borrow_from_optional_tensor(output_zero_points);
  const at::Tensor &output_zero_points_t = *output_zero_points_maybe_owned;
  const bool output_zero_points_defined = output_zero_points_t.defined();

  // Torch checks for quantized matmul.
  checks_for_quantized_matmul(bias_t, input, weight, result, input_scales,
                              input_zero_points, weight_scales,
                              weight_zero_points, output_scales_t,
                              output_zero_points_t, post_op_buffers);

  // Here the assumption is that, if the input dtype is int8(kChar)
  // or uint8(kByte), then it is already quantized.
  bool is_input_quantized =
      input.scalar_type() == c10::kByte || input.scalar_type() == c10::kChar;
  at::Tensor q_input;
  if (!is_input_quantized) {
    q_input = at::detail::empty_strided_cpu(
        input.sizes(), input.strides(),
        input_zero_points.defined() ? c10::kByte : c10::kChar); // For u8 & s8
  }

  if (bias_defined) {
    LOG(INFO) << "bias dimensions: " << bias_t.sizes();
  }

  // Get scales and zero points memory for the matmul operation.
  at::Tensor rq_output_scales;
  if (output_scales_defined) {
    rq_output_scales = 1 / output_scales_t;
  }

  // Get the required matmul op scales memory.
  ZenTorchMatmulOpScalesMemory matmul_op_scales_memory =
      get_zentorch_matmul_op_scales_memory(input_scales, weight_scales,
                                           rq_output_scales);
  // Get the required matmul op zero points memory.
  ZenTorchMatmulOpZeroPointsMemory matmul_op_zero_points_memory =
      get_zentorch_matmul_op_zero_points_memory(input_zero_points,
                                                output_zero_points_t);

  memory z_q_input, z_q_weight, z_bias, z_result;
  aten_tensor_to_zen_memory_for_quantized_matmul(
      input, weight, bias_t, result, input_scales, input_zero_points,
      is_input_quantized, output_stride, q_input, z_q_input, z_q_weight, z_bias,
      z_result);

  std::unordered_map<int, memory> execute_args;
  zendnn::primitive_attr op_attr;
  post_ops po;
  // Setting Post ops
  if (post_op_ids.size() > 0) {
    zentorch_post_ops_selection(po, execute_args, post_op_ids, post_op_buffers);
    op_attr.set_post_ops(po);
  }
  op_attr.set_plugin_op_name(zentorch_op_name);

  // Set the scales for the matmul operation.
  // Per-tensor config.
  op_attr.set_scales_mask(
      ZENDNN_ARG_SRC,
      /*mask*/ QUANT_GRANULARITY::PER_TENSOR, {},
      matmul_op_scales_memory.input_scales.get_desc().data_type());

  if (weight_scales.numel() == 1) {
    // Per-tensor config.
    op_attr.set_scales_mask(
        ZENDNN_ARG_WEIGHTS,
        /*mask*/ QUANT_GRANULARITY::PER_TENSOR, {},
        matmul_op_scales_memory.weight_scales.get_desc().data_type());
  } else if (weight_scales.numel() == weight.size(1)) {
    // Per-channel config.
    op_attr.set_scales_mask(
        ZENDNN_ARG_WEIGHTS,
        /*mask*/ QUANT_GRANULARITY::PER_CHANNEL, {1, 1},
        matmul_op_scales_memory.weight_scales.get_desc().data_type());
  }
  if (output_scales_defined) {
    // Per-tensor config.
    op_attr.set_scales_mask(
        ZENDNN_ARG_DST,
        /*mask*/ QUANT_GRANULARITY::PER_TENSOR, {},
        matmul_op_scales_memory.dst_rq_output_scales.get_desc().data_type());
  }

  // Set the zero_points for the matmul operation.
  if (input_zero_points.defined()) {
    op_attr.set_zero_points(ZENDNN_ARG_SRC,
                            /* mask */ QUANT_GRANULARITY::PER_TENSOR,
                            {ZENDNN_RUNTIME_S32_VAL});
    execute_args.insert({ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_SRC,
                         matmul_op_zero_points_memory.input_zero_points});
  }

  // Set the scales and zero_points for input and weight to execute the matmul
  // operation.
  execute_args.insert({ZENDNN_ARG_ATTR_SCALES | ZENDNN_ARG_SRC,
                       matmul_op_scales_memory.input_scales});

  execute_args.insert({ZENDNN_ARG_ATTR_SCALES | ZENDNN_ARG_WEIGHTS,
                       matmul_op_scales_memory.weight_scales});
  // requantization(rq) = dequantization(dq) + quantization(q)
  // rq_scale = 1 / output_scales;
  // Set the dst_rq_output_scales and output_zero_points for the quantized
  // matmul operation to requantize the result.

  if (output_scales_defined) {
    execute_args.insert({ZENDNN_ARG_ATTR_SCALES | ZENDNN_ARG_DST,
                         matmul_op_scales_memory.dst_rq_output_scales});
  }
  // Set dst_output_zero_points only for requantized uint8 output.
  if (output_zero_points_defined) {
    ZENTORCH_CHECK(output_zero_points_t.numel() == 1,
                   "Only per-tensor re-quantization is supported");
    if (result.scalar_type() == c10::kByte) {
      op_attr.set_zero_points(ZENDNN_ARG_DST,
                              /* mask */ QUANT_GRANULARITY::PER_TENSOR,
                              {ZENDNN_RUNTIME_S32_VAL});
      execute_args.insert(
          {ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_DST,
           matmul_op_zero_points_memory.dst_output_zero_points});
    }
  }
  // Execute the zendnn::matmul kernel.
  zentorch_matmul_execute(execute_args, z_q_input, z_q_weight, z_bias, z_result,
                          op_attr, bias_defined);
  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
}

template <UNARY_POST_OP fuse>
void zentorch_qlinear_out_unary(at::Tensor &result, const at::Tensor &input,
                                const at::Tensor &weight,
                                c10::optional<at::Tensor> bias,
                                const at::Tensor &input_scales,
                                const at::Tensor &input_zero_points,
                                const at::Tensor &weight_scales,
                                const at::Tensor &weight_zero_points,
                                c10::optional<c10::ScalarType> output_dtype,
                                c10::optional<at::Tensor> output_scales,
                                c10::optional<at::Tensor> output_zero_points,
                                std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  ZENTORCH_CHECK(output_dtype == result.scalar_type(),
                 "output_dtype received does not match the dtype of the "
                 "output tensor");
  ZENTORCH_CHECK(output_dtype == c10::kFloat ||
                     output_dtype == c10::kBFloat16 ||
                     output_dtype == c10::kByte || output_dtype == c10::kChar,
                 "output_dtype received is not yet supported, only "
                 "float32/bfloat16/uint8/int8 is supported");

  ZENTORCH_CHECK(is_avx512_supported(),
                 "Zentorch's INT8 kernels require the CPU to support "
                 "AVX512 instructions.");

  at::Tensor q_input = at::detail::empty_strided_cpu(
      input.sizes(), input.strides(),
      input_zero_points.defined() ? c10::kByte : c10::kChar); // For u8 & s8

  // `input` is viewed as 2d for matmul computation.
  auto input_2d_view =
      input.is_contiguous()
          ? input.view(get_2d_size_for_tensor(input))
          : input.contiguous().view(get_2d_size_for_tensor(input));

  // `weight` is transposed for matmul computation.
  auto weight_transposed = weight.t();

  // `result` is viewed as 2d for matmul computation.
  at::Tensor result_2d_view = result.view(get_2d_size_for_tensor(result));
  auto output_stride = result_2d_view.stride(0);

  // Set unary post ops.
  std::vector<at::Tensor> post_op_buffers = {};
  std::vector<int64_t> post_op_ids = {fuse};
  LOG(INFO) << "Calling zentorch_quantized_matmul_impl from " << __FUNCTION__
            << "!\n";
  zentorch_quantized_matmul_impl(
      input_2d_view, weight_transposed, bias, result_2d_view, input_scales,
      input_zero_points, weight_scales, weight_zero_points, post_op_ids,
      post_op_buffers, output_scales, output_zero_points, output_stride,
      zentorch_op_name);
}

template <UNARY_POST_OP fuse>
at::Tensor zentorch_qlinear_unary(const at::Tensor &input,
                                  const at::Tensor &weight,
                                  c10::optional<at::Tensor> bias,
                                  const at::Tensor &input_scales,
                                  const at::Tensor &input_zero_points,
                                  const at::Tensor &weight_scales,
                                  const at::Tensor &weight_zero_points,
                                  c10::optional<c10::ScalarType> output_dtype,
                                  c10::optional<at::Tensor> output_scales,
                                  c10::optional<at::Tensor> output_zero_points,
                                  std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  if (!output_dtype.has_value()) {
    output_dtype = c10::kFloat;
  }

  // `weight` is transposed for matmul computation.
  auto weight_transposed = weight.t();

  // `result` tensor's dtype will depend on output_dtype argument.
  auto output_sz = get_matmul_and_linear_output_sizes(input, weight_transposed);
  auto output_strides = get_matmul_and_linear_output_strides(output_sz);

  at::Tensor result = at::detail::empty_strided_cpu(
      output_sz, output_strides, input.options().dtype(output_dtype));

  zentorch_qlinear_out_unary<fuse>(
      result, input, weight, bias, input_scales, input_zero_points,
      weight_scales, weight_zero_points, output_dtype, output_scales,
      output_zero_points, zentorch_op_name);

  return result;
}

template <BINARY_POST_OP fuse1, BINARY_POST_OP fuse2>
inline at::Tensor zentorch_qlinear_binary_binary(
    const at::Tensor &input, const at::Tensor &weight,
    c10::optional<at::Tensor> bias, const at::Tensor &input_scales,
    const at::Tensor &input_zero_points, const at::Tensor &weight_scales,
    const at::Tensor &weight_zero_points, const at::Tensor &binary1_input,
    const at::Tensor &binary2_input,
    c10::optional<c10::ScalarType> output_dtype,
    c10::optional<at::Tensor> output_scales,
    c10::optional<at::Tensor> output_zero_points,
    std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  if (output_dtype.has_value()) {
    ZENTORCH_CHECK(output_dtype == c10::kFloat ||
                       output_dtype == c10::kBFloat16 ||
                       output_dtype == c10::kByte || output_dtype == c10::kChar,
                   "output_dtype received is not yet supported, only "
                   "float32/bfloat16/uint8/int8 is supported");
  } else {
    output_dtype = c10::kFloat;
  }

  ZENTORCH_CHECK(is_avx512_supported(),
                 "Zentorch's INT8 kernels require the CPU to support "
                 "AVX512 instructions.");

  at::Tensor q_input = at::detail::empty_strided_cpu(
      input.sizes(), input.strides(),
      input_zero_points.defined() ? c10::kByte : c10::kChar); // For u8 & s8

  // `input` is viewed as 2d for matmul computation.
  auto input_2d_view =
      input.is_contiguous()
          ? input.view(get_2d_size_for_tensor(input))
          : input.contiguous().view(get_2d_size_for_tensor(input));

  auto binary1_input_2d_view =
      binary1_input.is_contiguous()
          ? binary1_input.view(get_2d_size_for_tensor(binary1_input))
          : binary1_input.contiguous().view(
                get_2d_size_for_tensor(binary1_input));
  auto binary2_input_2d_view =
      binary2_input.is_contiguous()
          ? binary2_input.view(get_2d_size_for_tensor(binary2_input))
          : binary2_input.contiguous().view(
                get_2d_size_for_tensor(binary2_input));
  // `weight` is transposed for matmul computation.
  auto weight_transposed = weight.t();

  // `result` tensor's dtype will depend on output_dtype argument.
  auto output_sz = get_matmul_and_linear_output_sizes(input, weight_transposed);
  auto output_strides = get_matmul_and_linear_output_strides(output_sz);

  at::Tensor result = at::detail::empty_strided_cpu(
      output_sz, output_strides, input.options().dtype(output_dtype));

  // `result` is viewed as 2d for matmul computation.
  at::Tensor result_2d_view = result.view(get_2d_size_for_tensor(result));

  std::vector<at::Tensor> post_op_buffers = {binary1_input_2d_view,
                                             binary2_input_2d_view};
  std::vector<int64_t> post_op_ids = {fuse1, fuse2};

  LOG(INFO) << "Calling  zentorch_woq_linear_impl from " << __FUNCTION__
            << "!\n";
  zentorch_quantized_matmul_impl(
      input_2d_view, weight_transposed, bias, result_2d_view, input_scales,
      input_zero_points, weight_scales, weight_zero_points, post_op_ids,
      post_op_buffers, output_scales, output_zero_points,
      result_2d_view.stride(0), zentorch_op_name);
  return result;
}

// TODO: Explore the possibility of making output_dtype as kwarg with
// a default value.
TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_qlinear(Tensor input, Tensor weight, "
        "Tensor? bias, Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, *, "
        "ScalarType? output_dtype=None, Tensor? output_scales=None, "
        "Tensor? output_zero_points=None, str zentorch_op_name="
        "'zentorch::zentorch_qlinear') -> Tensor");
  m.def("zentorch_qlinear_relu(Tensor input, Tensor weight, "
        "Tensor? bias, Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, *, "
        "ScalarType? output_dtype=None, Tensor? "
        "output_scales=None, "
        "Tensor? output_zero_points=None, str zentorch_op_name="
        "'zentorch::zentorch_qlinear_relu') -> Tensor");
  m.def("zentorch_qlinear_sigmoid(Tensor input, Tensor weight, "
        "Tensor? bias, Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, *, "
        "ScalarType? output_dtype=None, Tensor? "
        "output_scales=None, "
        "Tensor? output_zero_points=None, str zentorch_op_name="
        "'zentorch::zentorch_qlinear_sigmoid') -> Tensor");

  m.def("zentorch_qlinear_mul_add(Tensor input, Tensor weight, "
        "Tensor? bias, Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, Tensor "
        " mul_input, Tensor add_input, *, "
        "ScalarType? output_dtype=None, Tensor? "
        "output_scales=None, "
        "Tensor? output_zero_points=None, str zentorch_op_name="
        "'zentorch::zentorch_qlinear_mul_add') -> Tensor");

  m.def("zentorch_qlinear.out(Tensor(a!) out,"
        "Tensor input, Tensor weight, "
        "Tensor? bias, Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, "
        "ScalarType? output_dtype=None, Tensor? output_scales=None, "
        "Tensor? output_zero_points=None,"
        "str zentorch_op_name='zentorch::zentorch_qlinear.out') -> ()");
  m.def("zentorch_qlinear_relu.out(Tensor(a!) out,"
        "Tensor input, Tensor weight, "
        "Tensor? bias, Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, "
        "ScalarType? output_dtype=None, Tensor? "
        "output_scales=None, "
        "Tensor? output_zero_points=None,"
        "str zentorch_op_name="
        "'zentorch::zentorch_qlinear_relu.out') -> ()");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_qlinear",
         zentorch::zentorch_qlinear_unary<UNARY_POST_OP::POST_OP_NONE>);
  m.impl("zentorch_qlinear_relu",
         zentorch::zentorch_qlinear_unary<UNARY_POST_OP::RELU>);
  m.impl("zentorch_qlinear_sigmoid",
         zentorch::zentorch_qlinear_unary<UNARY_POST_OP::SIGMOID>);
  m.impl("zentorch_qlinear_mul_add",
         zentorch::zentorch_qlinear_binary_binary<BINARY_POST_OP::MUL,
                                                  BINARY_POST_OP::ADD>);
  m.impl("zentorch_qlinear.out",
         zentorch::zentorch_qlinear_out_unary<UNARY_POST_OP::POST_OP_NONE>);
  m.impl("zentorch_qlinear_relu.out",
         zentorch::zentorch_qlinear_out_unary<UNARY_POST_OP::RELU>);
}

} // namespace zentorch
