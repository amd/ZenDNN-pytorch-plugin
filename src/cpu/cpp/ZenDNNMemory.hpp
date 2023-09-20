/******************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "ZenDNNUtils.hpp"
#include <torch/all.h>

using namespace zendnn;

namespace ZenDNNTorch {

// this infers the zendnn datatype from aten tensor
inline auto get_ztype_from_aten(const at::Tensor &atensor) {
  using ZenDType = typename memory::data_type;
  auto atype = atensor.scalar_type();
  switch (atype) {
  case c10::kByte:
    return ZenDType::u8;
  case c10::kChar:
    return ZenDType::s8;
  case c10::kInt:
    return ZenDType::s32;
  case c10::kFloat:
    return ZenDType::f32;
  case c10::kBFloat16:
    return ZenDType::bf16;
  default:
    TORCH_CHECK(false, "ZenDNNTorch::get_ztype_from_aten:"
                       " Unsupported data type.");
  }
}

// extract the 'default' format tag from dimensions
inline memory::format_tag get_default_format(const memory::dims &adims) {
  switch (adims.size()) {
  case 1:
    return memory::format_tag::a;
  case 2:
    return memory::format_tag::ab;
  case 3:
    return memory::format_tag::abc;
  case 4:
    return memory::format_tag::abcd;
  case 5:
    return memory::format_tag::abcde;
  case 6:
    return memory::format_tag::abcdef;
  default:
    return memory::format_tag::undef;
  }
}

// create a memory descriptor from aten tensor
inline memory::desc zen_memory_desc(const at::Tensor &atensor) {
  // currently the only supported memory format is the default one
  // for which we have the check below
  if (!atensor.is_contiguous()) {
    TORCH_CHECK(false, "ZenDNNTensor::zen_memory_desc:"
                       " Only default contiguous format is supported!");
  } else {
    // if the default format is given, then we proceed with descriptor creation
    memory::dims zdims = {atensor.sizes().cbegin(), atensor.sizes().cend()};
    memory::desc mem_desc = memory::desc(zdims, get_ztype_from_aten(atensor),
                                         get_default_format(zdims));
    return mem_desc;
  }
}

// below function returns memory from aten tensor
inline memory zen_memory(const at::Tensor &atensor,
                         const engine &aengine = utils::engine::cpu_engine()) {
  // we create memory descriptor inside this to avoid passing it as an argument
  memory::desc mem_desc = zen_memory_desc(atensor);
  auto atype = atensor.scalar_type();
  switch (atype) {
  case c10::kByte: {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kByte>::t);
    return memory(mem_desc, aengine, atensor.data_ptr<cpptype>());
  }
  case c10::kInt: {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kInt>::t);
    return memory(mem_desc, aengine, atensor.data_ptr<cpptype>());
  }
  case c10::kChar: {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kChar>::t);
    return memory(mem_desc, aengine, atensor.data_ptr<cpptype>());
  }
  case c10::kFloat: {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kFloat>::t);
    return memory(mem_desc, aengine, atensor.data_ptr<cpptype>());
  }
  case c10::kBFloat16: {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kBFloat16>::t);
    return memory(mem_desc, aengine, atensor.data_ptr<cpptype>());
  }
  default:
    TORCH_CHECK(false, "ZenDNNTorch::zen_memory:"
                       " Invalid data type, creating zendnn memory failed.");
  }
}

inline memory zen_memory_view_from_dense(
    const at::Tensor &atensor,
    const engine &aengine = utils::engine::cpu_engine()) {
  TORCH_CHECK(
      atensor.device().is_cpu(),
      "ZenDNNTorch::zen_memory_view_from_dense: expects CPU tensor input");
  TORCH_CHECK(
      atensor.layout() == c10::Layout::Strided,
      "ZenDNNTorch::zen_memory_view_from_dense: expects dense tensor input");
  TORCH_CHECK(atensor.scalar_type() == c10::ScalarType::Float ||
                  atensor.scalar_type() == c10::ScalarType::BFloat16 ||
                  atensor.scalar_type() == c10::ScalarType::Char ||
                  atensor.scalar_type() == c10::ScalarType::Byte,
              "ZenDNNTorch::zen_memory_view_from_dense: expects float or "
              "bfloat16 or char tensor input");
  // TODO: combining this function with ZenDNNTorch::zen_memory
  auto atype = atensor.scalar_type();
  // Providing stride information while initializing the zen_memory.
  // Otherwise, tensor data will be read in coloumn major format.
  switch (atype) {
  case c10::kByte: {
    return memory({atensor.sizes().vec(), get_ztype_from_aten(atensor),
                   atensor.strides().vec()},
                  aengine, atensor.template data_ptr<uint8_t>());
  }
  case c10::kChar: {
    return memory({atensor.sizes().vec(), get_ztype_from_aten(atensor),
                   atensor.strides().vec()},
                  aengine, atensor.template data_ptr<int8_t>());
  }
  case c10::kBFloat16: {
    return memory({atensor.sizes().vec(), get_ztype_from_aten(atensor),
                   atensor.strides().vec()},
                  aengine, atensor.template data_ptr<c10::BFloat16>());
  }
  default:
    return memory({atensor.sizes().vec(), memory::data_type::f32,
                   atensor.strides().vec()},
                  aengine, atensor.template data_ptr<float>());
  }
}

} // namespace ZenDNNTorch
