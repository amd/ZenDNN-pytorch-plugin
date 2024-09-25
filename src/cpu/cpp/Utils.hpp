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

namespace zendnn {

using kind = zendnn::primitive::kind;

namespace utils {
// cpu execution engine only.
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

// check AVX512 bf16 support
inline bool zendnn_bf16_device_check() {
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512bf16();
}

} // namespace utils
} // namespace zendnn
