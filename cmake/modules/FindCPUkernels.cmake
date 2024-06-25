#******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

IF (NOT MHA_FOUND)

find_package(Torch REQUIRED)

set(mha_512 "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/cpp/kernels/zen_MaskedMultiHeadAttention_512.cpp")
set(mha_ref "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/cpp/kernels/zen_MaskedMultiHeadAttention_ref.cpp")

# setting necessary flags for zen_MaskedMultiHeadAttention.cpp file
set(FLAGS "-Wall -Werror -Wno-unknown-pragmas -fPIC -fopenmp -fno-math-errno -fno-trapping-math -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0")
set(FLAGS_512 "-mavx512f -mavx512bf16 -mavx512vl -DCPU_CAPABILITY_AVX512")

set_source_files_properties(${mha_512} PROPERTIES COMPILE_FLAGS "${FLAGS} ${FLAGS_512}")
set_source_files_properties(${mha_ref} PROPERTIES COMPILE_FLAGS "${FLAGS}")

# creating library for mha
add_library(CPUkernels STATIC ${mha_512} ${mha_ref})

set_target_properties(CPUkernels PROPERTIES
                      ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_BINARY_DIR}/lib/)

target_include_directories(CPUkernels PUBLIC
                           ${TORCH_INCLUDE_DIRS})

LIST(APPEND MHA_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/lib/libCPUkernels.a)

SET(MHA_FOUND ON)

ENDIF (NOT MHA_FOUND)