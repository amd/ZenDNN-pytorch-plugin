/******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EnvReader.hpp"
#include "MatmulUtils.hpp"
#include "Memory.hpp"

#include <array>
#include <c10/util/StringUtil.h>
#include <optional>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

using namespace zendnnl::interface;

namespace zentorch {

torch::stable::Tensor
quantize_to_int8(const torch::stable::Tensor &input,
                 const torch::stable::Tensor &input_scales,
                 const torch::stable::Tensor &input_zero_points,
                 const bool input_zero_points_defined,
                 const data_type_t src_dtype) {
  const c10::ScalarType q_dtype =
      input_zero_points_defined ? c10::kByte : c10::kChar;
  const auto input_contiguous = get_contiguous_view(input);
  torch::stable::Tensor q_input = torch::stable::new_empty(
      input_contiguous, input_contiguous.sizes(), q_dtype);

  zendnnl::lowoha::reorder::reorder_params_t params;
  params.src_dtype = src_dtype;
  params.dst_dtype =
      input_zero_points_defined ? data_type_t::u8 : data_type_t::s8;
  auto to_long_vector = [](torch::headeronly::IntHeaderOnlyArrayRef arr)
      -> std::vector<long int> {
    return std::vector<long int>(arr.begin(), arr.end());
  };

  params.src_shape = to_long_vector(input_contiguous.sizes());
  params.dst_shape = to_long_vector(q_input.sizes());
  params.src_strides = to_long_vector(input_contiguous.strides());
  params.dst_strides = to_long_vector(q_input.strides());

  params.quant_params.scale.buff = input_scales.data_ptr();
  params.quant_params.scale.dt = data_type_t::f32;

  // Per-tensor quantization: all dims are 1
  const auto _dims = std::vector<int64_t>(input_contiguous.dim(), 1);

  // This is purely for per-tensor quantization.
  // Will update this once we support per-channel quantization.
  params.quant_params.scale.dims = _dims;

  params.quant_params.zero_point.buff = input_zero_points.data_ptr();
  params.quant_params.zero_point.dt = data_type_t::s32;

  // This is purely for per-tensor quantization.
  // Will update this once we support per-channel quantization.
  params.quant_params.zero_point.dims = _dims;

  status_t reorder_operator_status = zendnnl::lowoha::reorder::reorder_direct(
      input_contiguous.data_ptr(), q_input.data_ptr(), params);
  ZENTORCH_CHECK(reorder_operator_status == status_t::success,
                 "input to int8 quantization reorder failed for input tensor "
                 "with shape [",
                 c10::Join(", ", input_contiguous.sizes()),
                 "] and numel=", input_contiguous.numel());

  return q_input;
}

void zendnnl_quantized_matmul_impl(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const std::optional<torch::stable::Tensor> &bias,
    torch::stable::Tensor &result, const torch::stable::Tensor &input_scales,
    const torch::stable::Tensor &input_zero_points,
    const torch::stable::Tensor &weight_scales,
    const torch::stable::Tensor &weight_zero_points,
    const std::vector<int64_t> &post_op_ids,
    const std::vector<torch::stable::Tensor> &post_op_buffers,
    const std::optional<torch::stable::Tensor> &output_scales,
    const std::optional<torch::stable::Tensor> &output_zero_points,
    const int64_t output_stride, const bool is_weight_prepacked,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  LOG(INFO) << "input dimensions: [" << c10::Join(", ", input.sizes()) << "]";
  LOG(INFO) << "weight dimensions: [" << c10::Join(", ", weight.sizes()) << "]";
  LOG(INFO) << "input_scales dimensions: ["
            << c10::Join(", ", input_scales.sizes()) << "]";
  LOG(INFO) << "input_zero_points dimensions: ["
            << c10::Join(", ", input_zero_points.sizes()) << "]";
  LOG(INFO) << "weight_scales dimensions: ["
            << c10::Join(", ", weight_scales.sizes()) << "]";
  LOG(INFO) << "weight_zero_points dimensions: ["
            << c10::Join(", ", weight_zero_points.sizes()) << "]";
  LOG(INFO) << "result dimensions: [" << c10::Join(", ", result.sizes()) << "]";

  static const torch::stable::Tensor kUndefinedTensor;
  const torch::stable::Tensor &bias_t =
      bias.has_value() ? *bias : kUndefinedTensor;
  const bool bias_defined = bias_t.defined();

  const torch::stable::Tensor &output_scales_t =
      output_scales.has_value() ? *output_scales : kUndefinedTensor;
  const bool output_scales_defined = output_scales_t.defined();

  const torch::stable::Tensor &output_zero_points_t =
      output_zero_points.has_value() ? *output_zero_points : kUndefinedTensor;
  const bool output_zero_points_defined = output_zero_points_t.defined();

  // Torch checks for quantized matmul.
  // TODO: uncomment and enforce these checks later
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
  const bool is_input_quantized =
      input.scalar_type() == c10::kByte || input.scalar_type() == c10::kChar;
  const auto input_zero_points_defined = input_zero_points.defined();

  ZENTORCH_CHECK(!is_weight_prepacked || !weight_zero_points.defined(),
                 "zentorch_qlinear does not support prepacked weights along "
                 "with weight zero points, since the quantized kernel "
                 "computes its zero point compensation from the un-blocked "
                 "weight.");

  torch::stable::Tensor q_input;

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

    // This default zero points is only used with per tensor quantization.
    const auto default_zero_points =
        torch::stable::new_zeros(input, {1}, c10::kInt);
    const torch::stable::Tensor &zero_points_for_quant =
        input_zero_points_defined ? input_zero_points : default_zero_points;

    q_input =
        quantize_to_int8(input, input_scales, zero_points_for_quant,
                         input_zero_points_defined, get_zendnnl_dtype(input));
  }

  const int int_env_value =
      EnvReader::getEnvVariableAsInt("USE_ZENDNN_MATMUL_DIRECT");
  const bool use_zendnnl_direct_kernel = static_cast<bool>(int_env_value);

  torch::stable::Tensor inv_output_scales;
  if (output_scales_defined) {
    // `reciprocal` has no wrapper in torch/csrc/stable/ops.h and no typed AOTI
    // shim, so the boxed dispatcher is the only portable way to call it. The
    // StableIValue round trip is not an at::Tensor conversion: it boxes the
    // stable tensor into an ABI-stable slot, which is how ops.h implements its
    // own shim-less ops.
    std::array<StableIValue, 1> stack{
        torch::stable::detail::from(output_scales_t)};
    TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
        "aten::reciprocal", "", stack.data(), TORCH_ABI_VERSION));
    inv_output_scales =
        torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
  }

  if (use_zendnnl_direct_kernel) {
    zendnnl::lowoha::matmul::matmul_quantization_params_t quantization_params;

    // src scale
    quantization_params.src_scale.buff = input_scales.data_ptr();
    quantization_params.src_scale.dt = data_type_t::f32;
    quantization_params.src_scale.dims = std::vector<int64_t>(
        input_scales.sizes().begin(), input_scales.sizes().end());

    // weight scale
    quantization_params.wei_scale.buff = weight_scales.data_ptr();
    quantization_params.wei_scale.dt = data_type_t::f32;
    quantization_params.wei_scale.dims = std::vector<int64_t>(
        weight_scales.sizes().begin(), weight_scales.sizes().end());

    // dst scale
    if (output_scales_defined) {
      quantization_params.dst_scale.buff = inv_output_scales.data_ptr();
      quantization_params.dst_scale.dt = data_type_t::f32;
      quantization_params.dst_scale.dims = std::vector<int64_t>(
          inv_output_scales.sizes().begin(), inv_output_scales.sizes().end());
    }

    // src zero point
    if (input_zero_points_defined) {
      quantization_params.src_zp.buff = input_zero_points.data_ptr();
      quantization_params.src_zp.dt = data_type_t::s32;
      quantization_params.src_zp.dims = std::vector<int64_t>(
          input_zero_points.sizes().begin(), input_zero_points.sizes().end());
    }

    // weight zero point
    if (weight_zero_points.defined()) {
      quantization_params.wei_zp.buff = weight_zero_points.data_ptr();
      quantization_params.wei_zp.dt = data_type_t::s32;
      quantization_params.wei_zp.dims = std::vector<int64_t>(
          weight_zero_points.sizes().begin(), weight_zero_points.sizes().end());
    }

    // dst zero point
    if (output_zero_points_defined) {
      quantization_params.dst_zp.buff = output_zero_points_t.data_ptr();
      quantization_params.dst_zp.dt = data_type_t::s32;
      quantization_params.dst_zp.dims =
          std::vector<int64_t>(output_zero_points_t.sizes().begin(),
                               output_zero_points_t.sizes().end());
    }

    zendnnl_direct_kernel(is_input_quantized ? input : q_input, weight, bias_t,
                          result, 1.0f, post_op_ids, post_op_buffers,
                          true /* is_weight_const */, is_weight_prepacked,
                          zentorch_op_name, quantization_params);

    return;
  }

  using tensor_opt_ref = std::optional<std::reference_wrapper<tensor_t>>;

  tensor_t z_input_scales = tensor_t();
  tensor_opt_ref z_input_scales_opt_ref = std::nullopt;
  create_zendnnl_quantized_tensor(input_scales, z_input_scales, "input_scales");
  z_input_scales_opt_ref = tensor_opt_ref(std::ref(z_input_scales));

  tensor_t z_input_zero_points = tensor_t();
  tensor_opt_ref z_input_zero_points_opt_ref = std::nullopt;
  if (input_zero_points_defined) {
    create_zendnnl_quantized_tensor(input_zero_points, z_input_zero_points,
                                    "input_zero_points");
    z_input_zero_points_opt_ref = tensor_opt_ref(std::ref(z_input_zero_points));
  }

  tensor_t z_q_input = tensor_t();
  set_zendnnl_tensor_attributes(
      is_input_quantized ? input : q_input, z_q_input, "z_q_input",
      false /* is_weight_prepacked */, {} /* tensor_sizes */,
      {} /* tensor_strides */, {} /* tensor_aligned_sizes */, -1 /* nbytes */,
      z_input_scales_opt_ref, z_input_zero_points_opt_ref);
  LOG(INFO) << "Created input tensor";

  tensor_t z_weight_scales = tensor_t();
  tensor_opt_ref z_weight_scales_opt_ref = std::nullopt;
  if (weight_scales.defined()) {
    create_zendnnl_quantized_tensor(weight_scales, z_weight_scales,
                                    "weight_scales");
    z_weight_scales_opt_ref = tensor_opt_ref(std::ref(z_weight_scales));
  }

  // TODO
  // Support for weight_zero_points.
  // tensor_t z_weight_zero_points = tensor_t();
  // if (weight_zero_points.defined()) {
  // create_zendnnl_quantized_tensor(weight_zero_points, z_weight_zero_points,
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
      weight, z_q_weight, "z_q_weight", is_weight_prepacked,
      {} /* tensor_sizes */, {} /* tensor_strides */,
      {} /* tensor_aligned_sizes */, -1 /* nbytes */,
      z_weight_scales_opt_ref /*, z_weight_zero_points_opt_ref*/);

  tensor_t z_bias = tensor_t();
  if (bias_defined) {
    LOG(INFO) << "bias dimensions: [" << c10::Join(", ", bias_t.sizes()) << "]";
    unsigned long bias_numel = bias_t.numel();
    set_zendnnl_tensor_attributes(bias_t, z_bias, "z_bias",
                                  false /* is_weight_prepacked */,
                                  {1, bias_numel}, {bias_numel, 1});
  }

  // Get scales and zero points memory for the matmul operation.
  tensor_t z_dst_rq_output_scales = tensor_t();
  if (output_scales_defined) {
    create_zendnnl_quantized_tensor(inv_output_scales, z_dst_rq_output_scales,
                                    "dst_rq_output_scales");
  }

  tensor_t z_output_zero_points = tensor_t();
  if (output_zero_points_defined) {
    // The condition here was `if (output_zero_points_t.dim() == 1)`, which is
    // slightly different from the ones where we are using
    // create_zendnnl_quantized_tensor function in majority of the cases. In
    // majority of the cases, the condition was `if (tensor.dim() <= 1)`. So, if
    // there is any accuracy mismatch, this is a good starting point to debug.
    create_zendnnl_quantized_tensor(output_zero_points_t, z_output_zero_points,
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

  auto result_sizes_ref = result.sizes();
  std::vector<unsigned long> result_sizes(result_sizes_ref.begin(),
                                          result_sizes_ref.end());

  unsigned long output_stride_unsigned_long =
      static_cast<unsigned long>(output_stride);
  int64_t nbytes =
      result.element_size() * result_sizes[0] * output_stride_unsigned_long;
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
                                 z_result, post_op_ids, post_op_buffers,
                                 zentorch_op_name);

  status_t status = matmul_operator.execute();

  ZENTORCH_CHECK(status == status_t::success, "operator ",
                 matmul_operator.get_name(),
                 " execution failed for zentorch_matmul_impl.");

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
}

template <UNARY_POST_OP fuse>
void zentorch_qlinear_out_unary(
    torch::stable::Tensor &result, const torch::stable::Tensor &input,
    const torch::stable::Tensor &weight,
    const torch::stable::Tensor &input_scales,
    const torch::stable::Tensor &input_zero_points,
    const torch::stable::Tensor &weight_scales,
    const torch::stable::Tensor &weight_zero_points,
    const std::optional<torch::stable::Tensor> &bias,
    const std::optional<torch::stable::Tensor> &output_scales,
    const std::optional<torch::stable::Tensor> &output_zero_points,
    const std::optional<c10::ScalarType> &output_dtype,
    bool is_weight_prepacked, std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  ZENTORCH_CHECK(output_dtype.has_value(),
                 "output_dtype must be provided for out variant");
  ZENTORCH_CHECK(*output_dtype == result.scalar_type(),
                 "output_dtype received does not match the dtype of the "
                 "output tensor");
  ZENTORCH_CHECK(*output_dtype == c10::kFloat ||
                     *output_dtype == c10::kBFloat16 ||
                     *output_dtype == c10::kByte || *output_dtype == c10::kChar,
                 "output_dtype received is not yet supported, only "
                 "float32/bfloat16/uint8/int8 is supported");

  ZENTORCH_CHECK(is_avx512_supported(),
                 "Zentorch's INT8 kernels require the CPU to support "
                 "AVX512 instructions.");

  // `input` is viewed as 2d for matmul computation.
  auto input_2d_size = get_2d_size_for_tensor(input);
  auto input_2d_view =
      torch::stable::view(get_contiguous_view(input), input_2d_size);

  // `weight` is transposed for matmul computation.
  auto weight_transposed = torch::stable::transpose(weight, 0, 1);

  // `result` is viewed as 2d for matmul computation.
  auto result_2d_size = get_2d_size_for_tensor(result);
  auto result_2d_view = torch::stable::view(result, result_2d_size);
  auto output_stride = result_2d_view.stride(0);

  // Set unary post ops.
  std::vector<torch::stable::Tensor> post_op_buffers = {};
  std::vector<int64_t> post_op_ids = {fuse};
  LOG(INFO) << "Calling zendnnl_quantized_matmul_impl from " << __FUNCTION__
            << "!\n";
  zendnnl_quantized_matmul_impl(
      input_2d_view, weight_transposed, bias, result_2d_view, input_scales,
      input_zero_points, weight_scales, weight_zero_points, post_op_ids,
      post_op_buffers, output_scales, output_zero_points, output_stride,
      is_weight_prepacked, zentorch_op_name);
}

template <UNARY_POST_OP fuse>
torch::stable::Tensor zentorch_qlinear_unary(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &input_scales,
    const torch::stable::Tensor &input_zero_points,
    const torch::stable::Tensor &weight_scales,
    const torch::stable::Tensor &weight_zero_points,
    const std::optional<torch::stable::Tensor> &bias,
    const std::optional<torch::stable::Tensor> &output_scales,
    const std::optional<torch::stable::Tensor> &output_zero_points,
    const std::optional<c10::ScalarType> &output_dtype,
    bool is_weight_prepacked, std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  c10::ScalarType out_dtype = output_dtype.value_or(c10::kFloat);
  auto weight_transposed = torch::stable::transpose(weight, 0, 1);

  // `result` tensor's dtype will depend on output_dtype argument.
  auto output_sz = get_matmul_and_linear_output_sizes(input, weight_transposed);
  torch::stable::Tensor result =
      torch::stable::new_empty(input, output_sz, out_dtype);

  zentorch_qlinear_out_unary<fuse>(
      result, input, weight, input_scales, input_zero_points, weight_scales,
      weight_zero_points, bias, output_scales, output_zero_points,
      std::optional<c10::ScalarType>(out_dtype), is_weight_prepacked,
      zentorch_op_name);

  return result;
}

template <BINARY_POST_OP fuse1, BINARY_POST_OP fuse2>
inline torch::stable::Tensor zentorch_qlinear_binary_binary(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &input_scales,
    const torch::stable::Tensor &input_zero_points,
    const torch::stable::Tensor &weight_scales,
    const torch::stable::Tensor &weight_zero_points,
    const torch::stable::Tensor &binary1_input,
    const torch::stable::Tensor &binary2_input,
    const std::optional<torch::stable::Tensor> &bias,
    const std::optional<torch::stable::Tensor> &output_scales,
    const std::optional<torch::stable::Tensor> &output_zero_points,
    const std::optional<c10::ScalarType> &output_dtype,
    bool is_weight_prepacked, std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  if (output_dtype.has_value()) {
    ZENTORCH_CHECK(
        *output_dtype == c10::kFloat || *output_dtype == c10::kBFloat16 ||
            *output_dtype == c10::kByte || *output_dtype == c10::kChar,
        "output_dtype received is not yet supported, only "
        "float32/bfloat16/uint8/int8 is supported");
  }
  c10::ScalarType out_dtype = output_dtype.value_or(c10::kFloat);

  ZENTORCH_CHECK(is_avx512_supported(),
                 "Zentorch's INT8 kernels require the CPU to support "
                 "AVX512 instructions.");

  // `input` is viewed as 2d for matmul computation.
  auto input_2d_size = get_2d_size_for_tensor(input);
  auto input_2d_view =
      torch::stable::view(get_contiguous_view(input), input_2d_size);

  auto binary1_input_2d_size = get_2d_size_for_tensor(binary1_input);
  auto binary1_input_2d_view = torch::stable::view(
      get_contiguous_view(binary1_input), binary1_input_2d_size);
  auto binary2_input_2d_size = get_2d_size_for_tensor(binary2_input);
  auto binary2_input_2d_view = torch::stable::view(
      get_contiguous_view(binary2_input), binary2_input_2d_size);

  // `weight` is transposed for matmul computation.
  auto weight_transposed = torch::stable::transpose(weight, 0, 1);

  // `result` tensor's dtype will depend on output_dtype argument.
  auto output_sz = get_matmul_and_linear_output_sizes(input, weight_transposed);
  torch::stable::Tensor result =
      torch::stable::new_empty(input, output_sz, out_dtype);

  // `result` is viewed as 2d for matmul computation.
  auto result_2d_size = get_2d_size_for_tensor(result);
  auto result_2d_view = torch::stable::view(result, result_2d_size);

  std::vector<torch::stable::Tensor> post_op_buffers = {binary1_input_2d_view,
                                                        binary2_input_2d_view};
  std::vector<int64_t> post_op_ids = {fuse1, fuse2};

  LOG(INFO) << "Calling zendnnl_quantized_matmul_impl from " << __FUNCTION__
            << "!\n";
  zendnnl_quantized_matmul_impl(
      input_2d_view, weight_transposed, bias, result_2d_view, input_scales,
      input_zero_points, weight_scales, weight_zero_points, post_op_ids,
      post_op_buffers, output_scales, output_zero_points,
      result_2d_view.stride(0), is_weight_prepacked, zentorch_op_name);
  return result;
}

// TODO: Explore the possibility of making output_dtype as kwarg with
// a default value.
STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_qlinear(Tensor input, Tensor weight, "
        "Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, Tensor? bias, "
        "Tensor? output_scales, "
        "Tensor? output_zero_points, "
        "ScalarType? output_dtype=None, *, bool is_weight_prepacked=False, "
        "str zentorch_op_name='zentorch::zentorch_qlinear') "
        "-> Tensor");
  m.def("zentorch_qlinear_relu(Tensor input, Tensor weight, "
        "Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, Tensor? bias, "
        "Tensor? output_scales, "
        "Tensor? output_zero_points, "
        "ScalarType? output_dtype=None, *, bool is_weight_prepacked=False, "
        "str zentorch_op_name='zentorch::zentorch_qlinear_relu') "
        "-> Tensor");
  m.def("zentorch_qlinear_sigmoid(Tensor input, Tensor weight, "
        "Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, Tensor? bias, "
        "Tensor? output_scales, "
        "Tensor? output_zero_points, "
        "ScalarType? output_dtype=None, *, bool is_weight_prepacked=False, "
        "str zentorch_op_name='zentorch::zentorch_qlinear_sigmoid') "
        "-> Tensor");

  m.def("zentorch_qlinear_mul_add(Tensor input, Tensor weight, "
        "Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, Tensor "
        " mul_input, Tensor add_input, Tensor? bias, "
        "Tensor? output_scales, "
        "Tensor? output_zero_points, ScalarType? output_dtype=None, "
        "*, bool is_weight_prepacked=False, str "
        "zentorch_op_name='zentorch::zentorch_qlinear_mul_add') -> Tensor");

  m.def("zentorch_qlinear.out(Tensor(a!) out,"
        "Tensor input, Tensor weight, "
        "Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, Tensor? bias, "
        "Tensor? output_scales, "
        "Tensor? output_zero_points, ScalarType? output_dtype=None, "
        "*, bool is_weight_prepacked=False, "
        "str zentorch_op_name='zentorch::zentorch_qlinear.out') -> ()");
  m.def("zentorch_qlinear_relu.out(Tensor(a!) out,"
        "Tensor input, Tensor weight, "
        "Tensor input_scales, Tensor input_zero_points, "
        "Tensor weight_scales, Tensor weight_zero_points, "
        "Tensor? bias, "
        "Tensor? output_scales, "
        "Tensor? output_zero_points, ScalarType? output_dtype=None, "
        "*, bool is_weight_prepacked=False, "
        "str zentorch_op_name='zentorch::zentorch_qlinear_relu.out') -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_qlinear",
         TORCH_BOX(
             (&zentorch::zentorch_qlinear_unary<UNARY_POST_OP::POST_OP_NONE>)));
  m.impl("zentorch_qlinear_relu",
         TORCH_BOX((&zentorch::zentorch_qlinear_unary<UNARY_POST_OP::RELU>)));
  m.impl(
      "zentorch_qlinear_sigmoid",
      TORCH_BOX((&zentorch::zentorch_qlinear_unary<UNARY_POST_OP::SIGMOID>)));
  m.impl("zentorch_qlinear_mul_add",
         TORCH_BOX(
             (&zentorch::zentorch_qlinear_binary_binary<BINARY_POST_OP::MUL,
                                                        BINARY_POST_OP::ADD>)));
  m.impl(
      "zentorch_qlinear.out",
      TORCH_BOX((
          &zentorch::zentorch_qlinear_out_unary<UNARY_POST_OP::POST_OP_NONE>)));
  m.impl(
      "zentorch_qlinear_relu.out",
      TORCH_BOX((&zentorch::zentorch_qlinear_out_unary<UNARY_POST_OP::RELU>)));
}

} // namespace zentorch
