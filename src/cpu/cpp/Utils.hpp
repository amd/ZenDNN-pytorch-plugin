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
#include <zendnn.h>
#include <zendnn.hpp>

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
  // Add unary post op before this,
  // if you add any post op
  // update UNARY_OP_COUNT by that post op.
  UNARY_OP_COUNT = SIGMOID
};
// Initializing the first enum in BINARY_POST_OP so that all post ops will have
// unique value.
enum BINARY_POST_OP { MUL = UNARY_POST_OP::UNARY_OP_COUNT + 1, ADD };

static const std::map<std::string_view, int> post_op_map = {
    {"None", UNARY_POST_OP::POST_OP_NONE},
    {"relu", UNARY_POST_OP::RELU},
    {"gelu_tanh", UNARY_POST_OP::GELU_TANH},
    {"gelu_erf", UNARY_POST_OP::GELU_ERF},
    {"silu", UNARY_POST_OP::SILU},
    {"sigmoid", UNARY_POST_OP::SIGMOID},
    {"mul", BINARY_POST_OP::MUL},
    {"add", BINARY_POST_OP::ADD}};

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

// this infers the zendnn datatype from aten tensor
inline auto get_zdtype(const at::Tensor &atensor) {
  auto atype = atensor.scalar_type();
  switch (atype) {
  case c10::kByte:
    return zendnn_data_type_t::zendnn_u8;
  case c10::kChar:
    return zendnn_data_type_t::zendnn_s8;
  case c10::kInt:
    return zendnn_data_type_t::zendnn_s32;
  case c10::kFloat:
    return zendnn_data_type_t::zendnn_f32;
  case c10::kBFloat16:
    return zendnn_data_type_t::zendnn_bf16;
  case c10::kQUInt8:
    return zendnn_data_type_t::zendnn_u8;
  case c10::kQInt8:
    return zendnn_data_type_t::zendnn_s8;
  default:
    ZENTORCH_CHECK(false, "Unsupported data type.");
  }
}

// Check embedding-bag support, get dtype and emb_dim from weight tensor
inline bool is_zendnn_embedding_bag_supported(const at::Tensor &weight) {
  zendnn_data_type_t z_weight_dtype = get_zdtype(weight);
  unsigned int emb_dim = weight.size(1);
  return zendnn_custom_op::isEmbeddingBagSupported(z_weight_dtype, emb_dim);
}

} // namespace utils
} // namespace zendnn
