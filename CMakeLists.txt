#******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(zentorch)

# set cmake folder as a place to search for .cmake files
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

# build and add ZenDNN and BLIS libraries
find_package(ZENDNN REQUIRED)

# build mha kernel
find_package(CPUkernels REQUIRED)

add_dependencies(CPUkernels libamdZenDNN)

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/cpp/Config.hpp.in"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/cpp/Config.hpp" )

# Define our library target
add_library(zentorch INTERFACE)

add_dependencies(zentorch libamdZenDNN)

# Enable C++14
target_compile_features(zentorch INTERFACE cxx_std_14)

# Include target directories
target_include_directories(zentorch INTERFACE
                           ${ZENDNN_INCLUDE_DIR}
                           ${CMAKE_SOURCE_DIR}/ATen/native/zendnn/)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set - defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Link against ZenDNN and AOCL BLIS
target_link_libraries(zentorch INTERFACE ${MHA_LIBRARIES} ${ZENDNN_LIBRARIES} ${BLIS_LIBRARIES} ${FBGEMM_LIBRARIES} ${LIBXSMM_LIBRARIES})
