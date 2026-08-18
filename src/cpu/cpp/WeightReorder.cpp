/******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "Memory.hpp"
#include "zendnnl.hpp"

#include <algorithm>

namespace zendnnl::lowoha::matmul {
namespace native {
void clear_all_weight_caches();
} // namespace native
void clear_aocl_matmul_weight_caches();
void clear_onednn_matmul_weight_cache();
} // namespace zendnnl::lowoha::matmul

namespace zentorch {

using namespace zendnnl::interface;

// Clears all ZenDNN matmul weight-reorder caches.
void clear_zendnn_weight_caches() {
  zendnnl::lowoha::matmul::native::clear_all_weight_caches();
  zendnnl::lowoha::matmul::clear_aocl_matmul_weight_caches();
  zendnnl::lowoha::matmul::clear_onednn_matmul_weight_cache();
}

// AOCL picks its reorder routine per (wei_dtype, src_dtype) pair.
at::Tensor prepack_weight_for_blocked_matmul(const at::Tensor &weight_in,
                                             const data_type_t src_dtype) {
  const at::Tensor weight = weight_in.contiguous();

  // Matmul weight B = weight.t() = [K, N] = [in_features, out_features];
  // a contiguous [N, K] weight makes this view column-major ("ba").
  at::Tensor reorder_input = weight.transpose(0, 1);
  const auto in_sizes = reorder_input.sizes();
  const auto in_strides = reorder_input.strides();

  const int64_t K = in_sizes[0]; // in_features  (rows of B)
  const int64_t N = in_sizes[1]; // out_features (cols of B)
  ZENTORCH_CHECK(K > 0 && N > 0,
                 "weight prepack does not support empty weights; got K=", K,
                 ", N=", N, ".");

  // Match matmul's is_transposed (relaxed >=); ldb from the actual stride.
  const bool transposed = (in_strides[0] == 1 && in_strides[1] >= K);
  ZENTORCH_CHECK(transposed || (in_strides[1] == 1 && in_strides[0] >= N),
                 "weight prepack expects a 2D weight whose transposed view is "
                 "row-major- or transpose-contiguous.");
  const int64_t ldb = transposed ? in_strides[1] : in_strides[0];

  // LOWOHA prepack (aocl_dlp_blocked); matmul reads it via mem_format_b = 'r'.
  zendnnl::lowoha::reorder::reorder_params_t rp;
  rp.is_prepack = true;
  rp.prepack.algo = zendnnl::ops::matmul_algo_t::aocl_dlp_blocked;
  rp.prepack.wei_dtype = get_zendnnl_dtype(weight);
  rp.prepack.src_dtype = src_dtype;
  rp.prepack.K = K;
  rp.prepack.N = N;
  rp.prepack.ldb = ldb;
  rp.prepack.transposed = transposed;
  rp.prepack.sym_group_size = 0;

  // Step 1: query the (64B-aligned) prepacked buffer size.
  const size_t reorder_bytes =
      zendnnl::lowoha::reorder::weight_prepack_size(rp);
  ZENTORCH_CHECK(reorder_bytes > 0, "weight_prepack_size failed.");

  // Step 2: buffer = max(packed elems, as_strided view span) to avoid overrun.
  const int64_t elt_size = static_cast<int64_t>(weight.element_size());
  const int64_t packed_elements =
      (static_cast<int64_t>(reorder_bytes) + elt_size - 1) / elt_size;
  // Max element offset the [sizes, strides] view can reach, + 1.
  const int64_t view_elements = (weight.size(0) - 1) * weight.stride(0) +
                                (weight.size(1) - 1) * weight.stride(1) + 1;
  const int64_t num_elements = std::max(packed_elements, view_elements);
  at::Tensor packed = at::empty({num_elements}, weight.options());

  // Step 3: one-time prepack directly on the raw pointers.
  const status_t status = zendnnl::lowoha::reorder::reorder_direct(
      reorder_input.data_ptr(), packed.data_ptr(), rp);
  ZENTORCH_CHECK(status == status_t::success,
                 "weight prepack reorder_direct failed.");

  return at::as_strided(packed, weight.sizes(), weight.strides());
}

at::Tensor
zentorch_weight_prepack_for_linear(const at::Tensor &weight,
                                   const std::string & /*zentorch_op_name*/) {
  ZENTORCH_CHECK(weight.dim() == 2,
                 "Weight tensor must be 2D for linear layer prepacking, got ",
                 weight.dim(), "D tensor.");
  const auto dtype = weight.scalar_type();
  ZENTORCH_CHECK(dtype == c10::ScalarType::Float ||
                     dtype == c10::ScalarType::BFloat16 ||
                     dtype == c10::ScalarType::Half,
                 "Currently weight prepacking only supports float32, "
                 "bfloat16 or float16 dtype for weight tensor");

  // Non-quantized GEMMs run activations and weights at the same precision.
  return prepack_weight_for_blocked_matmul(weight, get_zendnnl_dtype(weight));
}

at::Tensor zentorch_weight_prepack_for_dynamic_qlinear(const at::Tensor &weight,
                                                       const std::string &
                                                       /*zentorch_op_name*/) {
  ZENTORCH_CHECK(weight.dim() == 2,
                 "Weight tensor must be 2D for qlinear layer prepacking, got ",
                 weight.dim(), "D tensor.");
  ZENTORCH_CHECK(weight.scalar_type() == c10::ScalarType::Char,
                 "Currently qlinear weight prepacking only supports int8 "
                 "dtype for weight tensor, got ",
                 weight.scalar_type(), ".");

  // Prepacked weights are gated on symmetric quantization, whose activations
  // qlinear quantizes to s8.
  return prepack_weight_for_blocked_matmul(weight, data_type_t::s8);
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_weight_prepack_for_linear(Tensor weight, "
        "str zentorch_op_name='zentorch::zentorch_weight_prepack_for_linear') "
        "-> Tensor");
  m.def("zentorch_weight_prepack_for_dynamic_qlinear(Tensor weight, "
        "str "
        "zentorch_op_name='zentorch::zentorch_weight_prepack_for_dynamic_"
        "qlinear') "
        "-> Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_weight_prepack_for_linear",
         zentorch::zentorch_weight_prepack_for_linear);
  m.impl("zentorch_weight_prepack_for_dynamic_qlinear",
         zentorch::zentorch_weight_prepack_for_dynamic_qlinear);
}

} // namespace zentorch
