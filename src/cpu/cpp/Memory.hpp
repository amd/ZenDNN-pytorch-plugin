/******************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "DataPointerManager.hpp"
#include "Utils.hpp"

#include "zendnnl.hpp"

using namespace zendnn;
using namespace zendnnl::interface;

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

// this infers the zendnnl datatype from aten tensor
inline auto get_zendnnl_dtype(const at::Tensor &atensor) {
  auto atype = atensor.scalar_type();
  switch (atype) {
  case c10::kByte:
    return data_type_t::u8;
  case c10::kChar:
    return data_type_t::s8;
  case c10::kInt:
    return data_type_t::s32;
  case c10::kLong:
    return data_type_t::s64;
  case c10::kFloat:
    return data_type_t::f32;
  case c10::kBFloat16:
    return data_type_t::bf16;
  case c10::kQUInt8:
    return data_type_t::u8;
  case c10::kQInt8:
    return data_type_t::s8;
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

// Inner variant: Sets zendnnl tensor attributes from raw pointer and explicit
// parameters. This is the core implementation that directly configures the
// zendnnl tensor object with all the necessary metadata for tensor operations
// in the zendnnl backend.
//
// Parameters (all mandatory):
//   at_tensor_ptr: Raw pointer to the tensor data buffer
//   zendnnl_dtype: The zendnnl data type
//   zendnnl_tensor: Reference to the zendnnl tensor object to be configured
//   tensor_name: Human-readable name for debugging/logging purposes
//   is_weight_prepacked: Flag indicating if this is a pre-packed weight tensor
//                        (which requires blocked layout for optimized access)
//   tensor_sizes: Dimensions of the tensor
//   tensor_strides: Stride in each dimension
//   tensor_aligned_sizes: Padded/aligned dimensions
//   nbytes: Total size of the tensor data in bytes

// This variant of the function must only be used in special cases and where all
// parameters are known and provided explicitly, since we are working buffer
// pointers directly. In all the general cases, use the convenience wrapper of
// the function.
inline void set_zendnnl_tensor_attributes(
    void *at_tensor_ptr, const data_type_t &zendnnl_dtype,
    tensor_t &zendnnl_tensor, const std::string_view &tensor_name,
    const bool &is_weight_prepacked,
    const std::vector<unsigned long> &tensor_sizes,
    const std::vector<unsigned long> &tensor_strides,
    const std::vector<unsigned long> &tensor_aligned_sizes,
    const int64_t &nbytes) {

  zendnnl_tensor.set_name(static_cast<std::string>(tensor_name))
      .set_data_type(zendnnl_dtype)
      .set_storage(at_tensor_ptr, nbytes);

  if (is_weight_prepacked) {
    zendnnl_tensor.set_layout(tensor_layout_t::blocked);
  }

  if (!tensor_aligned_sizes.empty()) {
    zendnnl_tensor.set_aligned_size(tensor_aligned_sizes);
  }

  zendnnl_tensor.set_size(tensor_sizes).set_stride(tensor_strides).create();

  ZENTORCH_CHECK(zendnnl_tensor.check(), "tensor creation of ",
                 zendnnl_tensor.get_name(),
                 " failed. Size: ", zendnnl_tensor.get_size(),
                 " Stride: ", zendnnl_tensor.get_stride());
}

// Convenience wrapper that extracts attributes from an aten Tensor
// and forwards them to the inner variant.
//
// This function acts as a bridge between the zentorch frontend (PyTorch) and
// zendnnl backend. It handles the common case where you have an aten tensor and
// want to create a corresponding zendnnl tensor with the same properties.
//
// Parameters:
//   at_tensor: PyTorch aten tensor to extract properties from
//   zendnnl_tensor: Reference to zendnnl tensor object to configure
//   tensor_name: Name for the tensor (for debugging/logging)
//   is_weight_prepacked: Whether this is a pre-packed weight
//   tensor_sizes: Optional override for tensor dimensions (uses
//   at_tensor.sizes() if empty) tensor_strides: Optional override for strides
//   (uses at_tensor.strides() if empty) tensor_aligned_sizes: Optional aligned
//   dimensions for SIMD optimization nbytes: Optional override for byte count
//   (uses at_tensor.nbytes() if -1)
inline void set_zendnnl_tensor_attributes(
    const at::Tensor &at_tensor, tensor_t &zendnnl_tensor,
    const std::string_view &tensor_name,
    const bool &is_weight_prepacked = false,
    const std::vector<unsigned long> &tensor_sizes = {},
    const std::vector<unsigned long> &tensor_strides = {},
    const std::vector<unsigned long> &tensor_aligned_sizes = {},
    const int64_t &nbytes = -1) {

  void *at_tensor_ptr = at_tensor.data_ptr();

  const std::vector<unsigned long> zendnnl_tensor_sizes =
      tensor_sizes.empty()
          ? std::vector<unsigned long>(at_tensor.sizes().begin(),
                                       at_tensor.sizes().end())
          : tensor_sizes;
  const std::vector<unsigned long> zendnnl_tensor_strides =
      tensor_strides.empty()
          ? std::vector<unsigned long>(at_tensor.strides().begin(),
                                       at_tensor.strides().end())
          : tensor_strides;

  // Delegate to the inner variant with all extracted/provided parameters
  set_zendnnl_tensor_attributes(
      at_tensor_ptr, get_zendnnl_dtype(at_tensor), zendnnl_tensor, tensor_name,
      is_weight_prepacked, zendnnl_tensor_sizes, zendnnl_tensor_strides,
      tensor_aligned_sizes, nbytes == -1 ? at_tensor.nbytes() : nbytes);
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
