/******************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "ZenDNNOps.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("embedding_bag_zendnn", ZenDNNTorch::_embedding_bag_zendnn_impl,
        py::arg("weight"), py::arg("indices"), py::arg("offsets"),
        py::arg("scale_grad_by_freq") = false, py::arg("mode") = 0,
        py::arg("sparse") = false, py::arg("per_sample_weights") = py::none(),
        py::arg("include_last_offset") = false, py::arg("padding_idx") = -1);

  m.def("show_config", ZenDNNTorch::show_config);

  m.def("is_bf16_supported", zendnn::utils::zendnn_bf16_device_check);

  m.def("zendnn_matmul_impl", ZenDNNTorch::zendnn_matmul_impl, py::arg("mat1"),
        py::arg("mat2"), py::arg("self_or_result"), py::arg("beta") = 0.0f,
        py::arg("alpha") = 1.0f, py::arg("fuse_relu") = false);

  m.def("zendnn_addmm", ZenDNNTorch::zendnn_addmm, py::arg("self"),
        py::arg("mat1"), py::arg("mat2"), py::kw_only(), py::arg("beta") = 1.0f,
        py::arg("alpha") = 1.0f, py::arg("fuse_relu") = false);

  m.def("zendnn_baddbmm", ZenDNNTorch::zendnn_baddbmm, py::arg("self"),
        py::arg("batch1"), py::arg("batch2"), py::kw_only(),
        py::arg("beta") = 1.0f, py::arg("alpha") = 1.0f);

  m.def("zendnn_mm", ZenDNNTorch::zendnn_mm, py::arg("self"), py::arg("mat2"),
        py::kw_only(), py::arg("fuse_relu") = false);

  m.def("zendnn_bmm", ZenDNNTorch::zendnn_bmm, py::arg("self"),
        py::arg("mat2"));
}
