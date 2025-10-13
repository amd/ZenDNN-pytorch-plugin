#******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(zentorch)
include(FetchContent)

# set cmake folder as a place to search for .cmake files
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

get_filename_component(PLUGIN_PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

set(ZENDNN_PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(ZENDNN_DIR "${ZENDNN_PARENT_DIR}/ZenDNN")

file(MAKE_DIRECTORY ${ZENDNN_PARENT_DIR})

include(zendnn)

add_subdirectory(${ZENDNN_DIR})

get_target_property(ZENDNN_INCLUDE_DIR amdZenDNN INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "zendnn includes, ZENDNN_INCLUDE_DIR, ${ZENDNN_INCLUDE_DIR}")

find_package(Torch REQUIRED)

# build mha kernel
find_package(CPUkernels REQUIRED)

set(COMPILER_NAME "${CMAKE_CXX_COMPILER_ID}")
set(COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}")

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/cpp/Config.hpp.in"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/cpp/Config.hpp")

#TODO: Restructure this block to remove warnings
file(GLOB ZENTORCH_CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/cpu/cpp/*.cpp")
set(ZENTORCH_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/src/cpu/cpp/")
list(REMOVE_ITEM ZENTORCH_CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/cpu/cpp/Bindings.cpp")

add_library(zentorch SHARED ${ZENTORCH_CPP_SOURCES})

add_dependencies(CPUkernels amdZenDNN)
add_dependencies(zentorch amdZenDNN CPUkernels)


# Enable C++17
target_compile_features(zentorch PUBLIC cxx_std_17)

set_target_properties(zentorch PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/)

target_include_directories(zentorch PUBLIC
                            ${ZENTORCH_INCLUDE_DIR}
                            ${MHA_INCLUDE_DIR}
                            ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(zentorch PUBLIC 
                        ${MHA_LIBRARIES} 
                        ${TORCH_LIBRARIES}
                        amdZenDNN
)

message(STATUS "zentorch PROJECT_NAME, ${PROJECT_NAME}")
add_custom_command(
    TARGET zentorch  POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/lib/libzentorch.so
            ${CMAKE_SOURCE_DIR}/${INSTALL_LIB_DIR}/${PROJECT_NAME}/)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set - defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()
