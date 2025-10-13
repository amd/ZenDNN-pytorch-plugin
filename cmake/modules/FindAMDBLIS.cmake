#******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************
IF (NOT AMDBLIS_FOUND)
string(REGEX MATCH "GLIBCXX_USE_CXX11_ABI=([0-9]+)" ZENTORCH_ABI_FLAG "${CMAKE_CXX_FLAGS}")
# For enabling debug build if needed
###############################################################################
IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
  SET(BUILD_FLAG 0)
ELSE()
  SET(BUILD_FLAG 1)
ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")
find_package(Git)
include(FetchContent)
get_filename_component(PLUGIN_PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
# Download/Copy ZenDNN and AOCL BLIS
###############################################################################
include(FetchContent)
get_filename_component(PLUGIN_PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
IF("$ENV{ZENTORCH_USE_LOCAL_BLIS}" EQUAL 0)
    # FetchContent_MakeAvailable auto-configures immediately after cloning, so
    # we supply a dummy path in FetchContent_Declare to defer configuration.
    # The actual build for this dependency is performed later in this script.
    FetchContent_Declare(blis
    GIT_REPOSITORY https://github.com/amd/blis.git
    GIT_TAG AOCL-Weekly-250725
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/blis"
    SOURCE_SUBDIR "not-available"
    )
    FetchContent_GetProperties(blis)
    if(NOT blis_POPULATED)
        FetchContent_MakeAvailable(blis)
    endif()
ELSE()
    IF(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/blis)
        IF(DEFINED ENV{ZENDNN_BLIS_PATH} AND NOT "$ENV{ZENDNN_BLIS_PATH}" STREQUAL "")
            message(STATUS "Using Local Blis Binaries from ZENDNN_BLIS_PATH: $ENV{ZENDNN_BLIS_PATH}")
        ELSEIF(EXISTS ${PLUGIN_PARENT_DIR}/blis)
            message("Copying Blis repo from local")
            file(COPY ${PLUGIN_PARENT_DIR}/blis DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
        ELSE()
            message( FATAL_ERROR "Copying of blis library from local failed, CMake will exit." )
        ENDIF()
    ENDIF()
ENDIF()
# To get BLIS Git Hash
# Check if the directory is a Git repository by verifying the existence of the .git directory
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/blis/.git")
    set(BLIS_VERSION_HASH "N/A")
elseif(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} -c log.showSignature=false log --no-abbrev-commit --oneline -1 --format=%H
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/blis
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE BLIS_VERSION_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
if(NOT GIT_FOUND OR RESULT)
    set(BLIS_VERSION_HASH "N/A")
endif()
# Build AOCL BLIS
###############################################################################
IF(NOT DEFINED ENV{ZENDNN_BLIS_PATH} OR "$ENV{ZENDNN_BLIS_PATH}" STREQUAL "")
    # Build BLIS from source
    add_custom_target(libamdblis ALL
        DEPENDS
            ${CMAKE_CURRENT_BINARY_DIR}/lib/libblis-mt.a
    )
    add_custom_command(
        OUTPUT
           ${CMAKE_CURRENT_BINARY_DIR}/lib/libblis-mt.a
       WORKING_DIRECTORY
           ${CMAKE_CURRENT_SOURCE_DIR}/third_party/blis
       COMMAND
           make clean && make distclean && CC=gcc ./configure -a aocl_gemm --prefix=${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build  --enable-threading=openmp --disable-blas --disable-cblas amdzen && CXXFLAGS="${CXXFLAGS} -D_${ZENTORCH_ABI_FLAG}" make -j install CMAKE_BUILD_TYPE==${CMAKE_BUILD_TYPE}
       COMMAND
           cp ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/lib/libblis-mt.a ${CMAKE_CURRENT_BINARY_DIR}/lib/
       COMMAND
           cp -r ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/include/blis/* ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/include
    )
    SET(BLIS_INCLUDE_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/include
    )
    set(ZENDNN_BLIS_PATH "${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build")
    set(ENV{ZENDNN_BLIS_PATH} "${ZENDNN_BLIS_PATH}")
    message(STATUS "Env set: $ENV{ZENDNN_BLIS_PATH}")
ELSE()
    # Use existing BLIS installation
    message(STATUS "Using existing BLIS from ZENDNN_BLIS_PATH: $ENV{ZENDNN_BLIS_PATH}")

    add_custom_target(libamdblis ALL
        DEPENDS
            ${CMAKE_CURRENT_BINARY_DIR}/lib/libblis-mt.a
    )

    add_custom_command(
        OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/lib/libblis-mt.a
        COMMAND
            mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/lib
        COMMAND
            mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/include
        COMMAND
            cp $ENV{ZENDNN_BLIS_PATH}/lib/libblis-mt.a ${CMAKE_CURRENT_BINARY_DIR}/lib/
        COMMAND
            cp -r $ENV{ZENDNN_BLIS_PATH}/include/* ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/include/
    )
    SET(BLIS_INCLUDE_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/include
    )
ENDIF()
SET(ZENDNN_BLIS_PATH ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build)
LIST(APPEND BLIS_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/lib/libblis-mt.a)
file(MAKE_DIRECTORY  ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/include)

add_library(amdblis::amdblis_archive STATIC IMPORTED GLOBAL)
add_dependencies(amdblis::amdblis_archive libamdblis)
set_target_properties(amdblis::amdblis_archive
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/lib/libblis-mt.a
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/include/
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/include/)
mark_as_advanced(amdblis::amdblis_archive)

add_library(amdblis::amdblis ALIAS amdblis::amdblis_archive)
# Export consistent variable name consumed by ZenDNN CMake
set(AMDBLIS_INCLUDE_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/include
    ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/include/blis
    CACHE STRING "AMDBLIS include dirs" FORCE)
SET(AMDBLIS_FOUND ON)
ENDIF (NOT AMDBLIS_FOUND)
