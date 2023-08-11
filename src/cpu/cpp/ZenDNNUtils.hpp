/******************************************************************************
* Copyright (c) 2023 Advanced Micro Devices, Inc.
* All rights reserved.
******************************************************************************/

#pragma once

#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <zendnn.h>
#include <zendnn.hpp>

namespace zendnn {

using kind = zendnn::primitive::kind;

namespace utils {
/// cpu execution engine only.
struct engine : public zendnn::engine {

  /// Singleton CPU engine for all primitives
  static engine& cpu_engine();

  engine(kind akind = kind::cpu, size_t index = 0)
       : zendnn::engine(akind, index){}
};

/// A default stream
struct stream : public zendnn::stream {
  static zendnn::stream& default_stream() {
    static zendnn::stream s(engine::cpu_engine());
    return s;
  }
};
} // namespace utils
} // namespace zendnn
