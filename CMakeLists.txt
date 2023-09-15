#******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(torch_zendnn_plugin)

# set cmake folder as a place to search for .cmake files
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

# build and add ZenDNN and BLIS libraries
find_package(ZENDNN REQUIRED)

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/cpp/Config.hpp.in"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/cpp/Config.hpp" )

# Define our library target
add_library(torch_zendnn_plugin INTERFACE)

add_dependencies(torch_zendnn_plugin libamdZenDNN)

# Enable C++14
target_compile_features(torch_zendnn_plugin INTERFACE cxx_std_14)

# Include target directories
target_include_directories(torch_zendnn_plugin INTERFACE
                           ${ZENDNN_INCLUDE_DIR}
                           ${CMAKE_SOURCE_DIR}/ATen/native/zendnn/)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set - defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Link against ZenDNN and AOCL BLIS
target_link_libraries(torch_zendnn_plugin INTERFACE ${ZENDNN_LIBRARIES} ${BLIS_LIBRARIES})
