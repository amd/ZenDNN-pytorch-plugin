# *******************************************************************************
# * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************
# ZenTorch: point CMake at ZenDNN's single integration script (master lives in ZenDNN fwk/).
# Override path placeholders before include — logic is all in ZenDNN/fwk/ZenDnnlFwkIntegrate.cmake
include_guard(GLOBAL)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/ZenDNN/fwk/ZenDnnlFwkIntegrate.cmake")
  message(FATAL_ERROR
    "ZenDNN not found at ${CMAKE_SOURCE_DIR}/third_party/ZenDNN. "
    "Populate via cmake/modules/zendnnl.cmake (FetchContent) or use the env variable ZENTORCH_USE_LOCAL_ZENDNN.")
endif()

set(ZENDNNL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/ZenDNN" CACHE PATH "zendnnl_source_dir" FORCE)
set(ZENDNNL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/zendnnl" CACHE PATH "zendnnl_binary_dir" FORCE)
set(ZENDNNL_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/lib" CACHE PATH "zendnnl_install_dir" FORCE)

include("${CMAKE_SOURCE_DIR}/third_party/ZenDNN/fwk/ZenDnnlFwkIntegrate.cmake")
