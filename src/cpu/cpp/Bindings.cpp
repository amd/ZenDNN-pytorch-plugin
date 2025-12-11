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
#include <torch/extension.h>

#include "DataPointerManager.hpp"
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

  m.def("is_bf16_supported", zentorch::zendnn_bf16_device_check,
        "Check if BF16 is supported on the current device.\n\n"
        "Returns:\n"
        "    Bool: True if BF16 is supported, False otherwise.");

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

} // End of PYBIND11_MODULE
