/******************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "DataPointerManager.hpp"
#include "Utils.hpp"

#include "zendnnl.hpp"

using namespace zendnnl::interface;

namespace zentorch {

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

} // namespace zentorch
