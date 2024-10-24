/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "MatmulUtils.hpp"
#include "Memory.hpp"

namespace zentorch {

using namespace zendnn;

// This function maps the aten tensors to the zendnn::memory.
inline void aten_tensor_to_zen_memory_for_woq_linear(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &weight_scales, const bool &bias_defined,
    const at::Tensor &bias, const at::Tensor &result, const int64_t &group_size,
    const int64_t &unpacking_ratio, memory &z_input, memory &z_qweight,
    memory &z_bias, memory &z_result, memory &z_woq_scales) {

  // Create input memory
  const memory::format_tag &memory_2d_tag = memory::format_tag::ab;
  const memory::desc &input_2d_desc =
      memory::desc({get_2d_size_for_tensor(input), get_ztype_from_aten(input),
                    memory_2d_tag});
  z_input = zen_memory(input, input_2d_desc);

  // qweight is packed along the output_channel with the packing_ratio,
  // but zendnn::memory creation requires the original weight dims for s4
  // qweight memory, so we get original dims by unpacking last dim of
  // qweight tensor(qweight.last_dim * unpacking_ratio).
  // Create qweight memory
  const memory::data_type &memory_int4_dtype = memory::data_type::s4;
  const memory::desc &qweight_desc =
      memory::desc({get_2d_size_for_tensor(qweight, unpacking_ratio),
                    memory_int4_dtype, memory_2d_tag});
  z_qweight = zen_memory(qweight, qweight_desc);

  // Create bias memory if bias is defined
  if (bias_defined) {
    // Creating bias zen_memory with predefined memory::desc
    // as bias is 1d we need to use format_tag as 'ab'
    // to represent bias memory as 2d for bias_desc creation.
    const memory::desc &bias_desc = memory::desc(
        {{1, bias.size(0)}, get_ztype_from_aten(bias), memory_2d_tag});
    z_bias = zen_memory(bias, bias_desc);
  }

  // Create result memory
  const memory::desc &result_2d_desc =
      memory::desc({get_2d_size_for_tensor(result), get_ztype_from_aten(result),
                    memory_2d_tag});
  z_result = zen_memory(result, result_2d_desc);

  // This weight_scales memory creation is specialized for the
  // per-channel & per-group weight_scales represented as 1-d memory.
  if (group_size == -1 || group_size > 0) {
    // Define per-group scales memory as 1-d zen_memory
    // Create woq weight_scales memory
    const memory::format_tag &memory_1d_tag = memory::format_tag::a;
    const memory::desc &woq_scales_mem_desc =
        memory::desc({{weight_scales.numel()},
                      get_ztype_from_aten(weight_scales),
                      memory_1d_tag});
    z_woq_scales = zen_memory(weight_scales, woq_scales_mem_desc);
  } else {
    // TODO: Support memory creation for per-tensor weight_scales
    ZENTORCH_CHECK(false, "group_size = ", group_size,
                   " is not supported, only group_size = -1 "
                   "or group_size > 0 is currently supported");
  }
}

inline bool are_all_zeros(const at::Tensor &inp_tensor) {
  // count non-zero elements
  auto non_zero_tensor = inp_tensor.nonzero();
  return non_zero_tensor.size(0) == 0;
}

inline void
check_valid_dtypes_for_woq(const std::string &compute_dtype,
                           const at::Tensor &input, const at::Tensor &result,
                           const std::vector<at::Tensor> &post_op_buffers) {
  ZENTORCH_CHECK(compute_dtype == "bfloat16",
                 "only bfloat16 compute_dtype is currently supported, "
                 "but the compute_dtype received is ",
                 compute_dtype, ".");

  bool are_params_bf16 = (input.scalar_type() == c10::ScalarType::BFloat16) &&
                         (result.scalar_type() == c10::ScalarType::BFloat16);

  ZENTORCH_CHECK(are_params_bf16, "only bfloat16 datatype "
                                  "is currently supported");

  if (are_params_bf16) {
    ZENTORCH_CHECK(utils::zendnn_bf16_device_check(),
                   "zendnn's woq matmul kernel computation "
                   "with bf16 inputs needs the cpu support of avx512bf16");
  }

  bool are_postops_bf16 = true;

  if (post_op_buffers.size() != 0) {
    for (const at::Tensor &buffer : post_op_buffers) {
      are_postops_bf16 = are_postops_bf16 &&
                         (buffer.scalar_type() == c10::ScalarType::BFloat16);
    }

    ZENTORCH_CHECK(are_postops_bf16,
                   "post ops have to be of a "
                   "dtype BFloat, when dtype of input matrix is BFloat");
  } else {
    LOG(INFO) << "Post Op buffers are not present!\n";
  }
}

inline void check_valid_shapes_for_woq(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &bias_t, const at::Tensor &result,
    const at::Tensor &weight_scales, const at::Tensor &weight_zero_point_t,
    const std::vector<at::Tensor> &post_op_buffers, const int64_t &group_size,
    const int64_t &unpacking_ratio) {
  // zentorch currently only supports the per-channel and per-group
  // weight_scales.
  // TODO: Add support for the per-tensor weight_scales
  ZENTORCH_CHECK(qweight.dim() == 2 && weight_scales.dim() == 2,
                 "unsupported dims for qweight and "
                 "weight_scales");
  const auto qweight_size = qweight.sizes();
  if (group_size == -1 || group_size > 0) {
    if (group_size > 0) {
      ZENTORCH_CHECK(group_size <= qweight_size[0], "group_size = ", group_size,
                     " is greater than input channel size = ", qweight_size[0]);
      ZENTORCH_CHECK(
          qweight_size[0] % group_size == 0,
          "input channel size = ", qweight_size[0],
          " is not completely divisible by group_size = ", group_size);
    }
    const int64_t computed_group_size =
        (group_size == -1) ? qweight_size[0] : group_size;
    ZENTORCH_CHECK(
        weight_scales.dim() == 2 &&
            weight_scales.size(0) == (qweight_size[0] / computed_group_size) &&
            weight_scales.size(1) == (qweight_size[1] * unpacking_ratio),
        "incompatible dimensions/shape for weight_scales with group_size = ",
        group_size);

    const bool weight_zero_point_defined = weight_zero_point_t.defined();
    if (weight_zero_point_defined) {
      LOG(INFO) << "weight_zero_point dimensions: "
                << weight_zero_point_t.sizes();
      ZENTORCH_CHECK(weight_zero_point_t.dim() == 2 &&
                         weight_zero_point_t.size(0) ==
                             qweight_size[0] / computed_group_size &&
                         weight_zero_point_t.size(1) == qweight_size[1],
                     "incompatible dimensions/shape for "
                     "weight_zero_point with group_size = ",
                     group_size);
      // TODO: to be tested for perf impact with group size not being -1
      ZENTORCH_CHECK(are_all_zeros(weight_zero_point_t),
                     "non-zero weight_zero_point "
                     "are not supported yet");
    }
  } else {
    ZENTORCH_CHECK(false, "group_size = ", group_size,
                   " is not supported, only group_size = -1 or "
                   "group_size > 0 is currently supported");
  }

  ZENTORCH_CHECK(weight_scales.scalar_type() == c10::kFloat ||
                     weight_scales.scalar_type() == c10::kBFloat16,
                 "only float32 and bfloat16 "
                 "weight_scales are currently supported");
  ZENTORCH_CHECK(input.size(input.dim() - 1) == qweight_size[0],
                 "unsupported sizes for input and qweight");
  ZENTORCH_CHECK(result.sizes() ==
                     c10::IntArrayRef(get_matmul_and_linear_output_sizes(
                         input, qweight, unpacking_ratio)),
                 "unsupported shapes for input, qweight and "
                 "result buffer");

  const bool bias_defined = bias_t.defined();
  if (bias_defined) {
    LOG(INFO) << "bias dimensions: " << bias_t.sizes();
    ZENTORCH_CHECK(bias_t.dim() == 1 &&
                       bias_t.size(0) == (qweight_size[1] * unpacking_ratio),
                   "incompatible dimensions/shape for bias");
  }

  if (post_op_buffers.size() != 0) {
    bool are_postops_shape_compatible = true;
    for (const at::Tensor &buffer : post_op_buffers) {
      are_postops_shape_compatible =
          are_postops_shape_compatible &&
          (buffer.sizes() ==
           c10::IntArrayRef(get_matmul_and_linear_output_sizes(
               input, qweight, unpacking_ratio)));
    }

    ZENTORCH_CHECK(are_postops_shape_compatible,
                   "unsupported shapes for input, qweight and "
                   "post op buffers");
  } else {
    LOG(INFO) << "Post Op buffers are not present!\n";
  }
}

// This function adds dtype, dim, group_size & weight_bits checks
// for args to zentorch_woq_linear op.
// This function also checks for the dtypes of the post op buffers
// based on the input matrix
inline void checks_for_woq_linear(
    const at::Tensor &input, const at::Tensor &qweight,
    const at::Tensor &bias_t, const at::Tensor &result,
    const at::Tensor &weight_scales, const at::Tensor &weight_zero_point_t,
    const std::vector<at::Tensor> &post_op_buffers, const int64_t &group_size,
    const int64_t &weight_bits, const std::string &compute_dtype,
    const int64_t &unpacking_ratio) {

  // The flow of this check is as follows:
  // -> The compute datatype is checked first, which if not compatible, the
  //    execution stops.
  // -> The individual datatypes of the tensors are inferred.
  // -> The tensors which are inputs to the actual matmul call are confirmed
  //    to be of datatype bfloat16.
  // -> If the dataype is bfloat16, the machine capability is checked.
  // -> Based on the post op buffer vector size, the dtypes of all the post ops
  //    are determined. Again here, all the post op buffers must be of the same
  //    dtype, either float32 or bfloat16, not a combination of both.

  ZENTORCH_CHECK(qweight.is_contiguous(), "qweight is non-contiguous & it is "
                                          "not supported yet");
  check_valid_dtypes_for_woq(compute_dtype, input, result, post_op_buffers);
  check_valid_shapes_for_woq(input, qweight, bias_t, result, weight_scales,
                             weight_zero_point_t, post_op_buffers, group_size,
                             unpacking_ratio);
}

// unpacking_ratio is utilized for unpacking the packed tensors to their
// original size where weight_bits is original quantized size of element of
// tensor while it is packed into the tensor element of larger size
inline int64_t get_unpacking_ratio(const at::Tensor &qweight,
                                   const int64_t &weight_bits) {
  int64_t unpacking_ratio;
  if ((qweight.scalar_type() == c10::kInt) && (weight_bits == 4)) {
    const int64_t bits_in_1_byte = 8;
    const int64_t total_bits = qweight.element_size() * bits_in_1_byte;
    unpacking_ratio = total_bits / weight_bits; // unpacking_ratio = 8
  } else {
    ZENTORCH_CHECK(false, "only int4 woq is currently supported "
                          "with qweight packed into int32");
  }
  return unpacking_ratio;
}
} // namespace zentorch
