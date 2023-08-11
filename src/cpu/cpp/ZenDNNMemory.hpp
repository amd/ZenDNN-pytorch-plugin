/******************************************************************************
* Copyright (c) 2023 Advanced Micro Devices, Inc.
* All rights reserved.
******************************************************************************/

#pragma once

#include <torch/all.h>
#include "ZenDNNUtils.hpp"

using namespace zendnn;

namespace ZenDNNTorch{

// this infers the zendnn datatype from aten tensor
inline auto get_ztype_from_aten(const at::Tensor &atensor){
  using ZenDType = typename memory::data_type;
  auto atype = atensor.scalar_type();
  switch(atype) {
  case c10::kByte:
      return ZenDType::u8;
  case c10::kChar:
      return ZenDType::s8;
  case c10::kInt:
      return ZenDType::s32;
  case c10::kFloat:
      return ZenDType::f32;
  case c10:: kBFloat16:
      return ZenDType::bf16;
  default:
      TORCH_CHECK(false,"ZenDNNTorch::get_ztype_from_aten:"
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
inline memory::desc zen_memory_desc(const at::Tensor &atensor){
  // currently the only supported memory format is the default one
  // for which we have the check below
  if(!atensor.is_contiguous()){
    TORCH_CHECK(false, "ZenDNNTensor::zen_memory_desc:"
                " Only default contiguous format is supported!");
  }
  else{
    // if the default format is given, then we proceed with descriptor creation
    memory::dims zdims = {atensor.sizes().cbegin(), atensor.sizes().cend()};
    memory::desc mem_desc = memory::desc(zdims,
                      get_ztype_from_aten(atensor), get_default_format(zdims));
    return mem_desc;
  }
}

// below function returns memory from aten tensor
inline memory zen_memory(const at::Tensor &atensor,
                    const engine &aengine=utils::engine::cpu_engine()){
  // we create memory descriptor inside this to avoid passing it as an argument
  memory::desc mem_desc = zen_memory_desc(atensor);
  auto atype = atensor.scalar_type();
  switch(atype) {
  case c10::kByte:
  {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kByte>::t);
    return memory(mem_desc, utils::engine::cpu_engine(),
                          atensor.data_ptr<cpptype>());
  }
  case c10::kInt:
  {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kInt>::t);
    return memory(mem_desc, utils::engine::cpu_engine(),
                          atensor.data_ptr<cpptype>());
  }
  case c10::kChar:
  {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kChar>::t);
    return memory(mem_desc, utils::engine::cpu_engine(),
                          atensor.data_ptr<cpptype>());
  }
  case c10::kFloat:
  {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kFloat>::t);
    return memory(mem_desc, utils::engine::cpu_engine(),
                          atensor.data_ptr<cpptype>());
  }
  case c10::kBFloat16:
  {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kBFloat16>::t);
    return memory(mem_desc, utils::engine::cpu_engine(),
                          atensor.data_ptr<cpptype>());
  }
  default:
    TORCH_CHECK(false,"ZenDNNTorch::zen_memory:"
                " Invalid data type, creating zendnn memory failed.");
  }
}
} // ZenDNNTorch
