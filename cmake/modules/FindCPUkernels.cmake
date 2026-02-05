#******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

IF (NOT MHA_FOUND)

find_package(Torch REQUIRED)

# Collect all .cpp files in the src directory
file(GLOB cpu_kernels "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/cpp/kernels/*.cpp")

# setting necessary flags for .cpp files
if(MSVC)
  set(FLAGS "/W3 /WX /openmp /O2 /std:c++17 /arch:AVX512 \
            /DCPU_CAPABILITY_AVX512")
else()
  set(FLAGS "-Wall -Werror -Wno-unknown-pragmas -Wno-error=uninitialized \
            -Wno-error=maybe-uninitialized -fPIC -fopenmp -fno-math-errno \
            -fno-trapping-math -O2 -std=c++17 -mavx512f -mavx512bf16 \
            -mavx512vl -mavx512dq -DCPU_CAPABILITY_AVX512")
endif()

set_source_files_properties(${cpu_kernels} PROPERTIES COMPILE_FLAGS "${FLAGS}")

# creating library for mha and sdpa
add_library(CPUkernels STATIC ${cpu_kernels})

set_target_properties(CPUkernels PROPERTIES
                      ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_BINARY_DIR}/lib/)

target_include_directories(CPUkernels PUBLIC
                           ${TORCH_INCLUDE_DIRS}
                           ${ZENDNNL_LIBRARY_INC_DIR})

target_link_libraries(CPUkernels PUBLIC zendnnl::zendnnl_archive)

set(MHA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/cpp/kernels/")

if(MSVC)
  LIST(APPEND MHA_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/lib/CPUkernels.lib)
else()
  LIST(APPEND MHA_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/lib/libCPUkernels.a)
endif()

SET(MHA_FOUND ON)

ENDIF (NOT MHA_FOUND)
