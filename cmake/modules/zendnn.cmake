#******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

if(NOT DEFINED ZENDNN_TAG)
  set(ZENDNN_TAG "zendnn-2025-WW47")
endif()

string(REGEX MATCH "-D_GLIBCXX_USE_CXX11_ABI=[0-1]" TEMP_ABI_DEFINE "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-D_GLIBCXX_USE_CXX11_ABI=" "" TEMP_ABI "${TEMP_ABI_DEFINE}")

if(NOT DEFINED _GLIBCXX_USE_CXX11_ABI)
  set(_GLIBCXX_USE_CXX11_ABI ${TEMP_ABI})
endif()

if(DEFINED ENV{ZENTORCH_USE_LOCAL_BLIS} AND "$ENV{ZENTORCH_USE_LOCAL_BLIS}" EQUAL 1)
    if(EXISTS ${PLUGIN_PARENT_DIR}/blis)
       set(AMDBLIS_LOCAL_SOURCE ${PLUGIN_PARENT_DIR}/blis)
    else()
       message(FATAL_ERROR "Directory ${PLUGIN_PARENT_DIR}/blis doesn't exist")
    endif()
endif()

if(DEFINED ENV{ZENTORCH_USE_LOCAL_ZENDNN} AND "$ENV{ZENTORCH_USE_LOCAL_ZENDNN}" EQUAL 0)
    FetchContent_Declare(ZenDNN
     GIT_REPOSITORY https://github.com/amd/ZenDNN.git
     GIT_TAG "${ZENDNN_TAG}"
     SOURCE_DIR "${ZENDNN_DIR}"
     SOURCE_SUBDIR "not-available"
    )
    FetchContent_MakeAvailable(ZenDNN)
    message(STATUS "ZenDNN downloaded successfully")
else()
    if(NOT EXISTS ${ZENDNN_DIR})
        if(EXISTS ${PLUGIN_PARENT_DIR}/ZenDNN)
            file(COPY ${PLUGIN_PARENT_DIR}/ZenDNN DESTINATION "${ZENDNN_PARENT_DIR}")
            file(REMOVE_RECURSE ${ZENDNN_DIR}/_out)
            message(STATUS "ZenDNN copied from a local directory.")
        else()
            message(FATAL_ERROR "${PLUGIN_PARENT_DIR}/ZenDNN doesn't exist, Cannot proceed with local ZenDNN setup.")
        endif()
    else()
        message(STATUS "Using existing ZenDNN directory: ${ZENDNN_DIR}")
    endif()
endif()

if(NOT EXISTS "${ZENDNN_DIR}")
    message(FATAL_ERROR "Directory ${ZENDNN_DIR}, doesn't exist")
endif()

if(NOT DEFINED FBGEMM_TAG)
  set(FBGEMM_TAG v1.2.0)
endif()
set(FBGEMM_VERSION_TAG ${FBGEMM_TAG})

if(NOT DEFINED FBGEMM_ENABLE)
  set(FBGEMM_ENABLE 1)
endif()

if(NOT DEFINED LPGEMM_V5_0)
  set(LPGEMM_V5_0 1)
endif()

if(NOT DEFINED BLIS_API)
  set(BLIS_API 1)
endif()

option(AMDBLIS_ENABLE_BLAS "AMD-BLIS, ENABLE_BLAS" OFF)

if(NOT DEFINED AMDBLIS_BLIS_CONFIG_FAMILY)
    set(AMDBLIS_BLIS_CONFIG_FAMILY amdzen)
endif()

if(NOT DEFINED AMDBLIS_ENABLE_THREADING)
    set(AMDBLIS_ENABLE_THREADING openmp)
endif()

if(NOT DEFINED AMDBLIS_TAG)
  set(AMDBLIS_TAG AOCL-Sep2025-b1)
endif()
