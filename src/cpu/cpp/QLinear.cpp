/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "MatmulUtils.hpp"
#include "Memory.hpp"

namespace zentorch {
using namespace zendnnl::interface;

void zendnnl_quantized_matmul_impl(
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
  check_valid_dtypes_for_quantized_matmul(
      bias_t, input, weight, result, input_scales, input_zero_points,
      weight_scales, weight_zero_points, output_scales_t, output_zero_points_t,
      post_op_buffers);
  check_valid_sizes_for_quantized_matmul(
      bias_t, input, weight, result, input_scales, input_zero_points,
      weight_scales, weight_zero_points, output_scales_t, output_zero_points_t,
      post_op_buffers);

  // Here the assumption is that, if the input dtype is int8(kChar)
  // or uint8(kByte), then it is already quantized.
  bool is_input_quantized =
      input.scalar_type() == c10::kByte || input.scalar_type() == c10::kChar;

  using tensor_opt_ref = std::optional<std::reference_wrapper<tensor_t>>;

  auto create_zendnnl_tensor = [](const at::Tensor &tensor, tensor_t &z_tensor,
                                  std::string_view name) {
    if (tensor.dim() <= 1) {
      // The library's current implementation requires the tensors in the form
      // of 2d tensors. Hence the {1, numel} for the tensors is used.
      unsigned long tensor_numel = tensor.numel();
      set_zendnnl_tensor_attributes(tensor, z_tensor, name,
                                    false /* is_weight_prepacked */,
                                    {1, tensor_numel}, {tensor_numel, 1});
    } else {
      set_zendnnl_tensor_attributes(tensor, z_tensor, name);
    }

    LOG(INFO) << "Created " << name << " tensor";
  };

  at::Tensor q_input;
  tensor_t z_input_scales = tensor_t();
  tensor_opt_ref z_input_scales_opt_ref = std::nullopt;
  if (input_scales.defined()) {
    create_zendnnl_tensor(input_scales, z_input_scales, "input_scales");
    z_input_scales_opt_ref = tensor_opt_ref(std::ref(z_input_scales));
  }

  tensor_t z_input_zero_points = tensor_t();
  tensor_opt_ref z_input_zero_points_opt_ref = std::nullopt;
  if (input_zero_points.defined()) {
    create_zendnnl_tensor(input_zero_points, z_input_zero_points,
                          "input_zero_points");
    z_input_zero_points_opt_ref = tensor_opt_ref(std::ref(z_input_zero_points));
  }

  tensor_t z_q_input = tensor_t();
  if (!is_input_quantized) {
    // fp32 tensor quantization:
    // q_tensor_s8 =
    // max(quant_min, std::nearby_int(tensor_fp32/scale) + zero_point)
    // s8 q_tensor dequantization:
    // dq_tensor_fp32 =
    // (min(quant_max, q_tensor_s8) - zero_point) * scale

    // `input` tensor quantization with q_input_scales & input_zero_points.
    // ZenDNN matmul's quantized kernel only supports u8 & s8 dtype for
    // quantized input & s8 dtype for quantized weight.

    // TODO
    // As soon as the library supports bf16 to int8 quantization, the following
    // check will be removed and bf16 input will be supported. Branching based
    // on the input datatype and respective apis to quantize the input will be
    // added.

    ZENTORCH_CHECK(input.scalar_type() == c10::kFloat,
                   "unsupported dtype for quantization of input tensor, "
                   "currently only float32 input tensor can be quantized");

    LOG(INFO) << "Using quantize_per_tensor API to quantize float input\n";
    q_input = at::quantize_per_tensor(
        input, input_scales,
        input_zero_points.defined() ? input_zero_points
                                    : torch::zeros(1).to(torch::kInt),
        input_zero_points.defined() ? c10::kQUInt8 : c10::kQInt8);

    set_zendnnl_tensor_attributes(
        q_input, z_q_input, "z_q_input", false /* is_weight_prepacked */,
        {} /* tensor_sizes */, {} /* tensor_strides */,
        {} /* tensor_aligned_sizes */, -1 /* nbytes */, z_input_scales_opt_ref,
        z_input_zero_points_opt_ref);
    LOG(INFO) << "Created input tensor";
  } else {
    set_zendnnl_tensor_attributes(
        input, z_q_input, "z_q_input", false /* is_weight_prepacked */,
        {} /* tensor_sizes */, {} /* tensor_strides */,
        {} /* tensor_aligned_sizes */, -1 /* nbytes */, z_input_scales_opt_ref,
        z_input_zero_points_opt_ref);
    LOG(INFO) << "Created input tensor";
  }

  tensor_t z_weight_scales = tensor_t();
  tensor_opt_ref z_weight_scales_opt_ref = std::nullopt;
  if (weight_scales.defined()) {
    create_zendnnl_tensor(weight_scales, z_weight_scales, "weight_scales");
    z_weight_scales_opt_ref = tensor_opt_ref(std::ref(z_weight_scales));
  }

  // TODO
  // Support for weight_zero_points.
  // tensor_t z_weight_zero_points = tensor_t();
  // if (weight_zero_points.defined()) {
  // create_zendnnl_tensor(weight_zero_points, z_weight_zero_points,
  // "weight_zero_points");
  // }

  // TODO
  // Support for weight_zero_points.
  // tensor_opt_ref
  // z_weight_zero_points_opt_ref =
  //     weight_zero_points.defined()
  //         ? tensor_opt_ref(
  //               std::ref(z_weight_zero_points))
  //         : std::nullopt;

  tensor_t z_q_weight = tensor_t();
  set_zendnnl_tensor_attributes(
      weight, z_q_weight, "z_q_weight", false /* is_weight_prepacked */,
      {} /* tensor_sizes */, {} /* tensor_strides */,
      {} /* tensor_aligned_sizes */, -1 /* nbytes */,
      z_weight_scales_opt_ref /*, z_weight_zero_points_opt_ref*/);

  tensor_t z_bias = tensor_t();
  if (bias_defined) {
    LOG(INFO) << "bias dimensions: " << bias_t.sizes();
    unsigned long bias_numel = bias_t.numel();
    set_zendnnl_tensor_attributes(bias_t, z_bias, "z_bias",
                                  false /* is_weight_prepacked */,
                                  {1, bias_numel}, {bias_numel, 1});
  }

  // Get scales and zero points memory for the matmul operation.
  at::Tensor rq_output_scales;
  tensor_t z_dst_rq_output_scales = tensor_t();
  if (output_scales_defined) {
    rq_output_scales = 1 / output_scales_t;
    create_zendnnl_tensor(rq_output_scales, z_dst_rq_output_scales,
                          "dst_rq_output_scales");
  }

  tensor_t z_output_zero_points = tensor_t();
  if (output_zero_points_defined) {
    // The condition here was `if (output_zero_points_t.dim() == 1)`, which is
    // slightly different from the ones where we are using create_zendnnl_tensor
    // function in majority of the cases. In majority of the cases, the
    // condition was `if (tensor.dim() <= 1)`. So, if there is any accuracy
    // mismatch, this is a good starting point to debug.
    create_zendnnl_tensor(output_zero_points_t, z_output_zero_points,
                          "output_zero_points");
  }

  tensor_t z_result = tensor_t();

  tensor_opt_ref z_dst_rq_output_scales_opt_ref =
      output_scales_defined ? tensor_opt_ref(std::ref(z_dst_rq_output_scales))
                            : std::nullopt;
  tensor_opt_ref z_output_zero_points_opt_ref =
      output_zero_points_defined
          ? tensor_opt_ref(std::ref(z_output_zero_points))
          : std::nullopt;

  std::vector<unsigned long> result_sizes(result.sizes().begin(),
                                          result.sizes().end());

  unsigned long output_stride_unsigned_long =
      static_cast<unsigned long>(output_stride);
  int64_t nbytes = c10::elementSize(result.scalar_type()) * result_sizes[0] *
                   output_stride_unsigned_long;
  std::vector<unsigned long> tensor_aligned_sizes = {
      result_sizes[0], output_stride_unsigned_long};
  set_zendnnl_tensor_attributes(
      result, z_result, "z_result", false /* is_weight_prepacked */,
      result_sizes, {output_stride_unsigned_long, 1}, tensor_aligned_sizes,
      nbytes, z_dst_rq_output_scales_opt_ref, z_output_zero_points_opt_ref);

  auto matmul_context = matmul_context_t();
  if (bias_defined) {
    set_matmul_context_attributes(matmul_context, z_q_weight, post_op_ids,
                                  1.0f /* alpha */, z_bias);
  } else {
    set_matmul_context_attributes(matmul_context, z_q_weight, post_op_ids,
                                  1.0f /* alpha */);
  }
  matmul_context.create();

  // TODO: Assign the operator name before setting the attributes in the below
  // function. This requires changes in multiple files. Hence a TODO for now.
  auto matmul_operator = matmul_operator_t();
  set_matmul_operator_attributes(matmul_operator, matmul_context, z_q_input,
                                 z_result, post_op_ids, post_op_buffers);

  status_t status = matmul_operator.execute();

  ZENTORCH_CHECK(status == status_t::success, "operator ",
                 matmul_operator.get_name(),
                 " execution failed for zentorch_matmul_impl.");

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
}

void zentorch_quantized_matmul_impl(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, at::Tensor &result,
    const at::Tensor &input_scales, const at::Tensor &input_zero_points,
    const at::Tensor &weight_scales, const at::Tensor &weight_zero_points,
    const std::vector<int64_t> &post_op_ids,
    const std::vector<at::Tensor> &post_op_buffers,
    const c10::optional<at::Tensor> &output_scales,
    const c10::optional<at::Tensor> &output_zero_points,
    const int64_t output_stride, std::string zentorch_op_name) {

  zendnnl_quantized_matmul_impl(
      input, weight, bias, result, input_scales, input_zero_points,
      weight_scales, weight_zero_points, post_op_ids, post_op_buffers,
      output_scales, output_zero_points, output_stride, zentorch_op_name);
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
