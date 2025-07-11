/******************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "DataPointerManager.hpp"
#include "Utils.hpp"
#include <torch/all.h>

using namespace zendnn;

namespace zentorch {

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
  case c10::kQUInt8:
    return ZenDType::u8;
  case c10::kQInt8:
    return ZenDType::s8;
  default:
    ZENTORCH_CHECK(false, "Unsupported data type.");
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
// const indicates if the tensor is a parameter or not
// const = False means no reordering will be done inplace or
// out of place. While inplace = False indicates reordering can be done
// out of place.
inline memory::desc zen_memory_desc(const at::Tensor &atensor,
                                    const bool &is_const = true) {
  // currently supported memory formats are default contiguous
  // and strided tensor formats, for which we have the check below
  memory::desc mem_desc;

  // Use DataPointerManager instead of global vector
  auto &manager = DataPointerManager::getInstance();
  const auto &pointers = manager.getPointers();

  bool inplace = true;
  uintptr_t tensor_ptr = reinterpret_cast<uintptr_t>(atensor.data_ptr());
  for (const auto &ptr : pointers) {
    if (tensor_ptr == ptr) {
      LOG(INFO) << "Marking the tensor inplace to be false: 0x" << std::hex
                << ptr << std::dec;
      inplace = false;
    }
  }
  if (!atensor.is_contiguous() && atensor.layout() == c10::Layout::Strided) {
    // Providing stride information while initializing the zen_memory_desc.
    // Otherwise, tensor data will be read in coloumn major format.
    mem_desc = memory::desc(atensor.sizes().vec(), get_ztype_from_aten(atensor),
                            atensor.strides().vec(), is_const, inplace);
  } else if (atensor.is_contiguous()) {
    // if the default contiguous format is given,
    // then we proceed with descriptor creation
    mem_desc = memory::desc(atensor.sizes().vec(), get_ztype_from_aten(atensor),
                            get_default_format(atensor.sizes().vec()), is_const,
                            inplace);
  } else {
    ZENTORCH_CHECK(
        false, "Only default contiguous and strided formats are supported!");
  }
  return mem_desc;
}

// below function returns memory with aten tensor and mem_desc
inline memory zen_memory(const at::Tensor &atensor,
                         const memory::desc &mem_desc = memory::desc(),
                         const engine &aengine = utils::engine::cpu_engine(),
                         const bool &is_const = true) {
  ZENTORCH_CHECK(atensor.device().is_cpu(), "expects CPU tensor input");
  ZENTORCH_CHECK(atensor.layout() == c10::Layout::Strided,
                 "expects dense tensor input");

  const memory::desc &a_mem_desc =
      mem_desc.is_zero() ? zen_memory_desc(atensor, is_const) : mem_desc;

  auto atype = atensor.scalar_type();
  switch (atype) {
  case c10::kByte: {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kByte>::t);
    return memory(a_mem_desc, aengine, atensor.data_ptr<cpptype>());
  }
  case c10::kInt: {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kInt>::t);
    return memory(a_mem_desc, aengine, atensor.data_ptr<cpptype>());
  }
  case c10::kChar: {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kChar>::t);
    return memory(a_mem_desc, aengine, atensor.data_ptr<cpptype>());
  }
  case c10::kFloat: {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kFloat>::t);
    return memory(a_mem_desc, aengine, atensor.data_ptr<cpptype>());
  }
  case c10::kBFloat16: {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kBFloat16>::t);
    return memory(a_mem_desc, aengine, atensor.data_ptr<cpptype>());
  }
  case c10::kQUInt8: {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kByte>::t);
    return memory(a_mem_desc, aengine, atensor.data_ptr<cpptype>());
  }
  case c10::kQInt8: {
    using cpptype = decltype(c10::impl::ScalarTypeToCPPType<c10::kChar>::t);
    return memory(a_mem_desc, aengine, atensor.data_ptr<cpptype>());
  }
  default:
    ZENTORCH_CHECK(false, "Invalid data type, creating zendnn memory failed.");
  }
}

// The reorder API allows us to transform data between various memory formats
// and data layouts. This is essential for optimizing performance, as some
// specific operations perform better with specific memory layouts.
// Reordering ensures that the data is in the correct format required by
// various computational primitives, such as GEMM and convolution.
inline memory zentorch_reorder(const memory &src, const memory &dst,
                               const primitive_attr &op_attr) {
  reorder::primitive_desc pd = reorder::primitive_desc(src, dst, op_attr);
  reorder(pd).execute(utils::stream::default_stream(),
                      {{ZENDNN_ARG_FROM, src}, {ZENDNN_ARG_TO, dst}});
  return dst;
}

} // namespace zentorch
