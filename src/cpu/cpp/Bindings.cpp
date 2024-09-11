/******************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

/*
        TORCH_LIBRARY is used for all ops which will replace ATen/prims ops in
   fx based graph optimizations. Few guidelines for prototypes.
                - If there is simlar op in ATen of PyTorch, please check
   "aten/src/ATen/native/native_functions.yaml" in PyTorch repo.
                - Our op arguments should be superset of the corresponding
   arguments in ATen op.
                - Our op arguments should match the arguments of corresponding
   op in both order of the arguments and type.
                - Our op specific arguments should be at the end of the list.
                - All ops should have prefix "zentorch_", for example
   zentorch_<corresponding op name>.
*/

// needs to be included only once in library.
#include "Singletons.hpp"

#include "Ops.hpp"
#include "kernels/zen_cpukernels_ops.hpp"

TORCH_LIBRARY(zentorch, m) {
  m.def("zentorch_masked_multihead_self_attention(Tensor query, Tensor key, "
        "Tensor value, Tensor key_cache, "
        "Tensor value_cache, Tensor beam_idx, Tensor seq_info, float "
        "scale_attn, int max_positions, "
        "Tensor? head_mask, Tensor? attention_mask, bool? "
        "add_casual_mask=None, str zentorch_op_name = "
        "'zentorch::zentorch_masked_multihead_self_attention')-> (Tensor, "
        "Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_masked_multihead_self_attention",
         zentorch::zentorch_masked_multihead_self_attention_impl);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("show_config", zentorch::show_config);

  m.def("is_bf16_supported", zendnn::utils::zendnn_bf16_device_check);

  m.def("zentorch_matmul_impl", zentorch::zentorch_matmul_impl,
        py::arg("input"), py::arg("weight"), py::arg("bias"),
        py::arg("self_or_result"), py::arg("post_op_ids"),
        py::arg("post_op_buffers"), py::arg("beta") = 0.0f,
        py::arg("alpha") = 1.0f,
        py::arg("zentorch_op_name") = "zentorch::zendnn_matmul_impl");
}
