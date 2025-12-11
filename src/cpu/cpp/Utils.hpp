/******************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <cpuinfo.h>
#include <torch/all.h>

// TODO: Make the __FILE__ give the name of the file relative to only
// ZenDNN_PyTorch_Plugin
#define ZENTORCH_CHECK(condition, ...)                                         \
  TORCH_CHECK(condition, __FILE__, ":", __LINE__, " ", __FUNCTION__, " : ",    \
              ##__VA_ARGS__)

namespace zentorch {

// zentorch:: Check if m/c supports AVX512/AVX256
inline bool is_avx512_supported() {
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512f() &&
         cpuinfo_has_x86_avx512vl() && cpuinfo_has_x86_avx512dq() &&
         cpuinfo_has_x86_avx512vnni() && cpuinfo_has_x86_avx512bf16() &&
         cpuinfo_has_x86_avx512bw();
}

inline bool zendnn_bf16_device_check() {
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512bf16();
}

enum EMBEDDING_BAG_ALGO {
  // Add unary post ops here
  SUM = 0,
  MEAN = 1,
  MAX = 2
};

enum UNARY_POST_OP {
  // Add unary post ops here
  POST_OP_NONE,
  RELU,
  GELU_TANH,
  GELU_ERF,
  SILU,
  SIGMOID,
  TANH,
  // Add unary post op before this,
  // if you add any post op
  // update UNARY_OP_COUNT by that post op.
  UNARY_OP_COUNT = TANH
};
// Initializing the first enum in BINARY_POST_OP so that all post ops will have
// unique value.
enum BINARY_POST_OP { MUL = UNARY_POST_OP::UNARY_OP_COUNT + 1, ADD };

static const std::map<std::string_view, int> post_op_map = {
    {"none", UNARY_POST_OP::POST_OP_NONE},
    {"relu", UNARY_POST_OP::RELU},
    {"gelu_tanh", UNARY_POST_OP::GELU_TANH},
    {"gelu_erf", UNARY_POST_OP::GELU_ERF},
    {"silu", UNARY_POST_OP::SILU},
    {"sigmoid", UNARY_POST_OP::SIGMOID},
    {"tanh", UNARY_POST_OP::TANH},
    {"mul", BINARY_POST_OP::MUL},
    {"add", BINARY_POST_OP::ADD}};

// Each value of QUANT_GRANULARITY enum indicates the mappings for various
// quantization granularity levels(PER_TENSOR/PER_CHANNEL/PER_GROUP)
// with the zendnn library's tensor mask values.
enum QUANT_GRANULARITY { PER_TENSOR = 0, PER_CHANNEL = 2, PER_GROUP = 3 };
} // namespace zentorch