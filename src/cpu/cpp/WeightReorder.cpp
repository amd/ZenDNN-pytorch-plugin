/******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "Memory.hpp"
#include "zendnnl.hpp"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

#include <algorithm>

namespace zentorch {

using namespace zendnnl::interface;

torch::stable::Tensor
zentorch_weight_prepack_for_linear(const torch::stable::Tensor &weight,
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

  // Matmul weight B = weight.t() = [K, N] = [in_features, out_features];
  // a contiguous [N, K] weight makes this view column-major ("ba").
  torch::stable::Tensor reorder_input = torch::stable::transpose(weight, 0, 1);
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
  rp.prepack.src_dtype = rp.prepack.wei_dtype;
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
  torch::stable::Tensor packed =
      torch::stable::new_empty(weight, {num_elements});

  // Step 3: one-time prepack directly on the raw pointers.
  const status_t status = zendnnl::lowoha::reorder::reorder_direct(
      reorder_input.data_ptr(), packed.data_ptr(), rp);
  ZENTORCH_CHECK(status == status_t::success,
                 "weight prepack reorder_direct failed.");

  // Present the buffer with the weight's original shape/strides. from_blob is
  // non-owning, so capture `packed` in the deleter to keep its storage alive.
  return torch::stable::from_blob(packed.data_ptr(), weight.sizes(),
                                  weight.strides(), weight.device(), dtype,
                                  [packed](void * /*data*/) {});
}

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_weight_prepack_for_linear(Tensor weight, "
        "str zentorch_op_name='zentorch::zentorch_weight_prepack_for_linear') "
        "-> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_weight_prepack_for_linear",
         TORCH_BOX(&zentorch::zentorch_weight_prepack_for_linear));
}

} // namespace zentorch
