/******************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "ZenDNNOps.hpp"

TORCH_LIBRARY(zentorch, m) {
  m.def("zendnn_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, "
        "bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? "
        "per_sample_weights=None, bool include_last_offset=False, int "
        "padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "zendnn_mm(Tensor self, Tensor mat2, *, bool fuse_relu=False) -> Tensor");
  m.def("zendnn_bmm(Tensor self, Tensor mat2) -> Tensor");
  m.def("zendnn_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, "
        "Scalar alpha=1, bool fuse_relu=False) -> Tensor");
  m.def("zendnn_baddbmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar "
        "beta=1, Scalar alpha=1) -> Tensor");
  m.def("zendnn_custom_embedding_bag_group(Tensor[] weight, Tensor[] indices, "
        "Tensor[] offsets, int[] scale_grad_by_freq, int[] mode, int[] "
        "sparse, Tensor?[] per_sample_weights, int[] include_last_offset, "
        "int[] padding_idx) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zendnn_embedding_bag", ZenDNNTorch::zendnn_embedding_bag_impl);
  m.impl("zendnn_mm", ZenDNNTorch::zendnn_mm);
  m.impl("zendnn_bmm", ZenDNNTorch::zendnn_bmm);
  m.impl("zendnn_addmm", ZenDNNTorch::zendnn_addmm);
  m.impl("zendnn_baddbmm", ZenDNNTorch::zendnn_baddbmm);
  m.impl("zendnn_custom_embedding_bag_group",
         ZenDNNTorch::zendnn_custom_embedding_bag_group);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("show_config", ZenDNNTorch::show_config);

  m.def("is_bf16_supported", zendnn::utils::zendnn_bf16_device_check);

  m.def("zendnn_matmul_impl", ZenDNNTorch::zendnn_matmul_impl, py::arg("mat1"),
        py::arg("mat2"), py::arg("self_or_result"), py::arg("beta") = 0.0f,
        py::arg("alpha") = 1.0f, py::arg("fuse_relu") = false);
}
