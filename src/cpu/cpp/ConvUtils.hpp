/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "Memory.hpp"
namespace zentorch {
using namespace zendnn;

inline std::vector<int64_t>
get_conv_output_sizes(const at::Tensor &input, const at::Tensor &weight,
                      const at::IntArrayRef &stride,
                      const at::IntArrayRef &padding,
                      const at::IntArrayRef &dilation) {

  // Convert the tensor to a list of integers
  const at::IntArrayRef &input_size = input.sizes();
  const at::IntArrayRef &weight_size = weight.sizes();

  bool has_dilation = !dilation.empty();
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);

  output_size[0] = input_size[0];
  output_size[1] = weight_size[0];
  for (const auto d : c10::irange(2, dim)) {
    auto dilation_ = has_dilation ? dilation[d - 2] : 1;
    auto kernel = dilation_ * (weight_size[d] - 1) + 1;
    output_size[d] =
        (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

// TODO: Use zen_memory_desc function from Memory.hpp for this purpose
// and modify accordingly
inline std::tuple<memory::desc, memory::desc, memory::desc, memory::desc,
                  memory::desc>
conv_tensors_to_memory_desc(const at::Tensor &input, const at::Tensor &weight,
                            const at::Tensor &bias, const at::Tensor &output) {

  memory::data_type dtype = input.dtype() == at::ScalarType::BFloat16
                                ? memory::data_type::bf16
                                : memory::data_type::f32;

  zendnn::memory::format_tag format_tag =
      input.is_contiguous(at::MemoryFormat::ChannelsLast)
          ? zendnn::memory::format_tag::nhwc
          : zendnn::memory::format_tag::nchw;

  std::vector<int64_t> dst_dims(output.sizes().begin(), output.sizes().end());
  memory::desc dst_desc(dst_dims, dtype, format_tag);

  // Get the tensor's shape as a vector of int64_t
  std::vector<int64_t> src_dims(input.sizes().begin(), input.sizes().end());
  std::vector<int64_t> weights_dims(weight.sizes().begin(),
                                    weight.sizes().end());
  std::vector<int64_t> bias_dims(bias.sizes().begin(), bias.sizes().end());

  // Create a descriptor with a different format tag
  memory::desc src_desc(src_dims, dtype, format_tag);
  memory::desc weights_desc(weights_dims, dtype, memory::format_tag::any);
  memory::desc bias_desc(bias_dims, dtype, memory::format_tag::x);
  memory::desc t_weights_desc(weights_dims, dtype, format_tag);

  std::tuple<memory::desc, memory::desc, memory::desc, memory::desc,
             memory::desc>
      out;
  out = std::make_tuple(src_desc, weights_desc, bias_desc, dst_desc,
                        t_weights_desc);

  return out;
}

inline void check_conv_inputs(const at::Tensor &input, const at::Tensor &weight,
                              const at::IntArrayRef &dilation) {

  ZENTORCH_CHECK((input.dim() == 4 && weight.dim() == 4),
                 "unsupported dims for conv input and weight");

  ZENTORCH_CHECK((dilation[0] == 1 && dilation[1] == 1),
                 "unsupported value of dilation, only [1,1] supported for now");

  ZENTORCH_CHECK((input.dtype() == at::ScalarType::BFloat16 ||
                  input.dtype() == at::ScalarType::Float),
                 "unsupported data type, only bf16 and fp32 supported for now");
}

} // namespace zentorch
