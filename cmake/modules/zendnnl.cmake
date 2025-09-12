#******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

# IF (NOT ZENDNNL_FOUND)

# string(REGEX MATCH "GLIBCXX_USE_CXX11_ABI=([0-9]+)" ZENTORCH_ABI_FLAG "${CMAKE_CXX_FLAGS}")

find_package(Git)

include(FetchContent)

get_filename_component(PLUGIN_PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

# Download/Copy ZENDNN-L
###############################################################################

IF("$ENV{ZENTORCH_USE_LOCAL_ZENDNN}" EQUAL 0)
    # FetchContent_MakeAvailable auto-configures immediately after cloning, so
    # we supply a dummy path in FetchContent_Declare to defer configuration.
    # The actual build for this dependency is performed later in this script.

    FetchContent_Declare(ZenDNNL
    GIT_REPOSITORY https://github.com/amd/ZenDNN.git
    GIT_TAG zendnnl
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNNL"
    SOURCE_SUBDIR "not-available"
    )
    FetchContent_GetProperties(ZenDNNL)
    if(NOT ZenDNN_POPULATED)
        FetchContent_MakeAvailable(ZenDNNL)
    endif()
ELSE()
    IF(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNNL)
        IF(EXISTS ${PLUGIN_PARENT_DIR}/ZenDNNL)
            file(COPY ${PLUGIN_PARENT_DIR}/ZenDNNL DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
            file(REMOVE_RECURSE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNNL/_out)
        ELSE()
            message( FATAL_ERROR "Copying of ZenDNNL library from local failed, CMake will exit." )
        ENDIF()
    ENDIF()
    #execute_process(COMMAND git pull WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNNL)
ENDIF()

# To get the ZenDNNL Git Hash
# Check if the directory is a Git repository by verifying the existence of the .git directory
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNNL/.git")
    set(ZENDNNL_LIB_VERSION_HASH "N/A")
elseif(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} -c log.showSignature=false log --no-abbrev-commit --oneline -1 --format=%H
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNNL
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE ZENDNNL_LIB_VERSION_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT GIT_FOUND OR RESULT)
    set(ZENDNNL_LIB_VERSION_HASH "N/A")
endif()