/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "ConvUtils.hpp"
#include "Memory.hpp"

namespace zentorch {

using namespace zendnn;

at::Tensor zentorch_conv_impl(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias_opt, const at::IntArrayRef &stride,
    const at::IntArrayRef &padding, const at::IntArrayRef &dilation,
    const bool &transposed, const at::IntArrayRef &output_padding,
    const int64_t &groups, std::string zentorch_op_name) {

  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor &bias = *bias_maybe_owned;

  std::vector<int64_t> output_size =
      get_conv_output_sizes(input, weight, stride, padding, dilation);

  at::MemoryFormat memory_format =
      input.is_contiguous(at::MemoryFormat::ChannelsLast)
          ? at::MemoryFormat::ChannelsLast
          : at::MemoryFormat::Contiguous;
  at::Tensor output = at::empty({output_size}, input.dtype(), memory_format);

  memory::desc src_desc, weights_desc, bias_desc, dst_desc, t_weights_desc;
  std::tie(src_desc, weights_desc, bias_desc, dst_desc, t_weights_desc) =
      conv_tensors_to_memory_desc(input, weight, bias, output);

  // creating ZenDNN memory using aten tensors
  memory z_src = zen_memory(input, src_desc);
  memory z_weights = zen_memory(weight, t_weights_desc);
  memory z_bias = zen_memory(bias, bias_desc);
  memory z_dst = zen_memory(output, dst_desc);

  primitive_attr op_attr;
  std::unordered_map<int, memory> execute_args;

  op_attr.set_plugin_op_name(zentorch_op_name);

  LOG(INFO) << "BEFORE going into bias.defined()";

  algorithm algo = algorithm::convolution_direct;
  prop_kind prop = prop_kind::forward_inference;
  convolution_forward::primitive_desc pd;
  if (bias.defined()) {
    LOG(INFO) << "Using the bias tensor!";
    // declare zendnn convolution primitive
    convolution_forward::desc pdesc = convolution_forward::desc(
        prop, algo, src_desc, weights_desc, bias_desc, dst_desc,
        {stride.begin(), stride.end()}, {padding.begin(), padding.end()},
        {padding.begin(), padding.end()});

    pd = convolution_forward::primitive_desc(pdesc, op_attr,
                                             utils::engine::cpu_engine());
  } else {
    LOG(WARNING) << "Bias is not defined!";
    // declare zendnn convolution primitive
    convolution_forward::desc pdesc = convolution_forward::desc(
        prop, algo, src_desc, weights_desc, dst_desc,
        {stride.begin(), stride.end()}, {padding.begin(), padding.end()},
        {padding.begin(), padding.end()});

    pd = convolution_forward::primitive_desc(pdesc, op_attr,
                                             utils::engine::cpu_engine());
  }
  // weight reorder
  reorder::primitive_desc rd_weights =
      reorder::primitive_desc(utils::engine::cpu_engine(), z_weights.get_desc(),
                              utils::engine::cpu_engine(), pd.weights_desc());
  memory new_z_weights(rd_weights.dst_desc(), utils::engine::cpu_engine());
  reorder(rd_weights)
      .execute(utils::stream::default_stream(),
               {{ZENDNN_ARG_FROM, z_weights}, {ZENDNN_ARG_TO, new_z_weights}});

  execute_args.insert({ZENDNN_ARG_SRC, z_src});
  execute_args.insert(
      {ZENDNN_ARG_WEIGHTS, new_z_weights}); // new_z_weights if reordering
  if (bias.defined()) {
    execute_args.insert({ZENDNN_ARG_BIAS, z_bias});
  }
  execute_args.insert({ZENDNN_ARG_DST, z_dst});

  LOG(INFO) << "Zentorch convolution compute in progress...";
  convolution_forward(pd).execute(utils::stream::default_stream(),
                                  execute_args);
  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
  return output;
}

// adding convolution op
at::Tensor zentorch_convolution(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias_opt, const at::IntArrayRef &stride,
    const at::IntArrayRef &padding, const at::IntArrayRef &dilation,
    const bool &transposed, const at::IntArrayRef &output_padding,
    const int64_t &groups, std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  check_conv_inputs(input, weight, dilation);

  LOG(INFO) << "Calling zentorch_conv_impl from " << __FUNCTION__ << "!\n";

  return zentorch_conv_impl(input, weight, bias_opt, stride, padding, dilation,
                            transposed, output_padding, groups,
                            zentorch_op_name);
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_convolution(Tensor input, Tensor weight, Tensor? bias, "
        "int[] stride, int[] padding, int[] dilation, bool transposed, "
        "int[] output_padding, int groups, str "
        "zentorch_op_name='zentorch::zentorch_convolution') "
        " -> Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_convolution", zentorch_convolution);
}
} // namespace zentorch
