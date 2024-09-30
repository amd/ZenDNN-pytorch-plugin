/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <cpuinfo.h>
#include <zendnn.h>
#include <zendnn.hpp>

// TODO: Make the __FILE__ give the name of the file relative to only
// ZenDNN_PyTorch_Plugin
#define ZENTORCH_CHECK(condition, ...)                                         \
  TORCH_CHECK(condition, __FILE__, ":", __LINE__, " ", __FUNCTION__, " : ",    \
              ##__VA_ARGS__)

namespace zentorch {
enum UNARY_POST_OP {
  // Add unary post ops here
  POST_OP_NONE,
  RELU,
  GELU_TANH,
  GELU_ERF,
  SILU,
  // Add unary post op before this,
  // if you add any post op
  // update UNARY_OP_COUNT by that post op.
  UNARY_OP_COUNT = SILU
};
// Initializing the first enum in BINARY_POST_OP so that all post ops will have
// unique value.
enum BINARY_POST_OP { MUL = UNARY_POST_OP::UNARY_OP_COUNT + 1, ADD };

// Each value of QUANT_GRANULARITY enum indicates the mappings for various
// quantization granularity levels(PER_TENSOR/PER_CHANNEL/PER_GROUP)
// with the zendnn library's tensor mask values.
enum QUANT_GRANULARITY { PER_TENSOR = 0, PER_CHANNEL = 2, PER_GROUP = 3 };
} // namespace zentorch

namespace zendnn {

using kind = zendnn::primitive::kind;

namespace utils {
// CPU execution engine only.
struct engine : public zendnn::engine {

  // Singleton CPU engine for all primitives
  static engine &cpu_engine();

  engine(kind akind = kind::cpu, size_t index = 0)
      : zendnn::engine(akind, index) {}
};

// A default stream
struct stream : public zendnn::stream {
  static zendnn::stream &default_stream() {
    static zendnn::stream s(engine::cpu_engine());
    return s;
  }
};

// Check AVX512 bf16 support
inline bool zendnn_bf16_device_check() {
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512bf16();
}

} // namespace utils
} // namespace zendnn
