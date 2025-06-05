/******************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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

// This file should include PYBIND11 operations only for functions intended
// to be used exclusively from Python.
// For functions meant for both torch.compile and torch.export use cases,
// utilize the TORCH_LIBRARY API.
// Please implement all TORCH_LIBRARY use cases in the file
// where the function is defined, not in this file.
*/

// needs to be included only once in library.
#include <pybind11/stl_bind.h>

#include "DataPointerManager.hpp"
#include "Memory.hpp"
#include "Ops.hpp"
#include "Threading.hpp"
#include "Utils.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Bind the DataPointerManager
  py::class_<zentorch::DataPointerManager>(m, "DataPointerManager")
      .def_static("getInstance", &zentorch::DataPointerManager::getInstance,
                  py::return_value_policy::reference)
      .def("addPointer", &zentorch::DataPointerManager::addPointer)
      .def("getPointers", &zentorch::DataPointerManager::getPointers,
           py::return_value_policy::reference)
      .def("clear", &zentorch::DataPointerManager::clear);

  m.def("show_config", &zentorch::show_config,
        "Show the current configuration of ZenTorch.");

  m.def("is_avx512_supported", zentorch::is_avx512_supported,
        "Check if AVX512 instructions are supported on the current hardware."
        "\n\nReturns:\n"
        "\tBool: True if AVX512 instructions are supported, False otherwise.");

  m.def("is_bf16_supported", zendnn::utils::zendnn_bf16_device_check,
        "Check if BF16 is supported on the current device.\n\n"
        "Returns:\n"
        "    Bool: True if BF16 is supported, False otherwise.");

  m.def("zentorch_matmul_impl", &zentorch::zentorch_matmul_impl,
        py::arg("input"), py::arg("weight"), py::arg("bias"),
        py::arg("self_or_result"), py::arg("post_op_ids"),
        py::arg("post_op_buffers"), py::arg("beta") = 0.0f,
        py::arg("alpha") = 1.0f,
        py::arg("zentorch_op_name") = "zentorch::zendnn_matmul_impl",
        py::arg("is_weight_const") = true,
        "Perform matrix multiplication with ZenTorch optimizations.\n\n"
        "Args:\n"
        "    input (torch.Tensor): The input tensor.\n"
        "    weight (torch.Tensor): The weight tensor.\n"
        "    bias (torch.Tensor): The bias tensor.\n"
        "    self_or_result (torch.Tensor): The result tensor.\n"
        "    post_op_ids (List[int]): Post Op IDs.\n"
        "    post_op_buffers (List[torch.Tensor]): Post Op buffers.\n"
        "    beta (float, optional): The beta value. Default is 0.0.\n"
        "    alpha (float, optional): The alpha value. Default is 1.0.\n"
        "    zentorch_op_name (str, optional): The operator name. Default is "
        "'zentorch::zendnn_matmul_impl'."
        "Returns:\n"
        "    Tensor: Result of the maxtrix multiplication.");

  m.def("zentorch_get_packed_embedding_weight",
        &zentorch::zentorch_get_packed_embedding_weight, py::arg("weight"),
        py::arg("weight_scales"), py::arg("weight_zero_points"),
        "Get packed embedding weights for ZenTorch.\n\n"
        "Args:\n"
        "    weight (torch.Tensor): The weight tensor.\n"
        "    weight_scales (List[float]): The weight scales.\n"
        "    weight_zero_points (List[int]): The weight zero points."
        "Returns:\n"
        "    Tensor: Packed embedding weights.");

  m.def("thread_bind", &zentorch::thread_bind, py::arg("core_ids"),
        "Bind threads to specified CPU cores.\n\n"
        "Args:\n"
        "    core_ids (List[int]): A list of core IDs to bind threads to.");

  m.def("zentorch_weight_reorder_for_matmul",
        &zentorch::zentorch_weight_reorder_for_matmul, py::arg("weight"),
        py::arg("is_weight_oc_x_ic") = true,
        "Reorder the weight tensor to desired format.\n\n"
        "Args:\n"
        "    weight (torch.Tensor): The weight tensor.\n"
        "    is_weight_oc_x_ic (bool, optional): True if weight is stored as "
        "OCxIC.\n"
        "Returns:\n"
        "    Tensor: Reordered weight tensor.");
}
