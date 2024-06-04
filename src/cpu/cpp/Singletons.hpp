/******************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "Utils.hpp"

namespace zendnn {
namespace utils {

engine &engine::cpu_engine() {
  static engine cpu_engine(kind::cpu, 0);
  return cpu_engine;
}

} // namespace utils
} // namespace zendnn
