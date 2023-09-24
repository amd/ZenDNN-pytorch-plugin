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
}
