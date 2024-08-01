#******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

IF (NOT ZENDNN_FOUND)


# For enabling debug build if needed
###############################################################################
IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
  SET(BUILD_FLAG 0)
ELSE()
  SET(BUILD_FLAG 1)
ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")

find_package(Git)

# Download/Copy ZenDNN and AOCL BLIS
###############################################################################

include(FetchContent)

get_filename_component(PLUGIN_PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

IF("$ENV{ZENTORCH_USE_LOCAL_BLIS}" EQUAL 0)
    FetchContent_Declare(blis
    GIT_REPOSITORY https://github.com/amd/blis.git
    GIT_TAG 4.2
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/blis"
    )
    FetchContent_GetProperties(blis)
    if(NOT blis_POPULATED)
        FetchContent_Populate(blis)
    endif()
ELSE()
    IF(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/blis)
        IF(EXISTS ${PLUGIN_PARENT_DIR}/blis)
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

# Download/Copy FBGEMM
###############################################################################

IF("$ENV{ZENTORCH_USE_LOCAL_FBGEMM}" EQUAL 0)
    FetchContent_Declare(FBGEMM
    GIT_REPOSITORY https://github.com/pytorch/FBGEMM.git
    GIT_TAG v0.6.0
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/FBGEMM"
    )
    FetchContent_GetProperties(FBGEMM)
    if(NOT FBGEMM_POPULATED)
        FetchContent_Populate(FBGEMM)
    endif()
    if(GIT_FOUND)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule sync
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    else(NOT GIT_FOUND )
        message(FATAL_ERROR "The submodules were not downloaded!")
    endif()
ELSE()
    IF(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/FBGEMM)
        IF(EXISTS ${PLUGIN_PARENT_DIR}/FBGEMM)
            file(COPY ${PLUGIN_PARENT_DIR}/FBGEMM DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
        ELSE()
            message( FATAL_ERROR "Copying of fbgemm library from local failed, CMake will exit." )
        ENDIF()
    ENDIF()
ENDIF()

# To get FBGEMM Git Hash and Git Tag
# Check if the directory is a Git repository by verifying the existence of the .git directory
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/FBGEMM/.git")
    set(FBGEMM_VERSION_HASH "N/A")
elseif(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/FBGEMM
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE FBGEMM_VERSION_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${GIT_EXECUTABLE} -c log.showSignature=false log --no-abbrev-commit --oneline -1 --format=%H
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/FBGEMM
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE FBGEMM_VERSION_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT GIT_FOUND OR RESULT)
        set(FBGEMM_VERSION_HASH "N/A")
endif()


# Download/Copy ZENDNN
###############################################################################

IF("$ENV{ZENTORCH_USE_LOCAL_ZENDNN}" EQUAL 0)
    FetchContent_Declare(ZenDNN
    GIT_REPOSITORY https://github.com/amd/ZenDNN.git
    GIT_TAG v4.2
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN"
    )
    FetchContent_GetProperties(ZenDNN)
    if(NOT ZenDNN_POPULATED)
        FetchContent_Populate(ZenDNN)
    endif()
ELSE()
    IF(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN)
        IF(EXISTS ${PLUGIN_PARENT_DIR}/ZenDNN)
            file(COPY ${PLUGIN_PARENT_DIR}/ZenDNN DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
            file(REMOVE_RECURSE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/_out)
        ELSE()
            message( FATAL_ERROR "Copying of ZenDNN library from local failed, CMake will exit." )
        ENDIF()
    ENDIF()
    execute_process(COMMAND git pull WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN)
ENDIF()

# To get the ZenDNN Git Hash
# Check if the directory is a Git repository by verifying the existence of the .git directory
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/.git")
    set(ZENDNN_LIB_VERSION_HASH "N/A")
elseif(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} -c log.showSignature=false log --no-abbrev-commit --oneline -1 --format=%H
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE ZENDNN_LIB_VERSION_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT GIT_FOUND OR RESULT)
    set(ZENDNN_LIB_VERSION_HASH "N/A")
endif()

# Build AOCL BLIS
###############################################################################
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
       make clean && make distclean && CC=gcc ./configure -a aocl_gemm --prefix=${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build  --enable-threading=openmp --disable-blas --disable-cblas amdzen && make -j install CMAKE_BUILD_TYPE==${CMAKE_BUILD_TYPE}
   COMMAND
       mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/lib/ && cp ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/lib/libblis-mt.a ${CMAKE_CURRENT_BINARY_DIR}/lib/
   COMMAND
       cp -r ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/include/blis/* ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/include
)

SET(BLIS_INCLUDE_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build/include
)

LIST(APPEND BLIS_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/lib/libblis-mt.a)

MARK_AS_ADVANCED(
        BLIS_INCLUDE_DIR
        BLIS_LIBRARIES
        blis-mt
)

# Build FBGEMM
###############################################################################
add_custom_target(libfbgemm ALL
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/lib/libfbgemm.a
)

add_custom_command(
    OUTPUT
       ${CMAKE_CURRENT_BINARY_DIR}/lib/libfbgemm.a
   WORKING_DIRECTORY
       ${CMAKE_CURRENT_SOURCE_DIR}/third_party/FBGEMM
   COMMAND
   mkdir build && cd build && cmake -DFBGEMM_LIBRARY_TYPE=static -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_C_FLAGS=\"-Wno-error=maybe-uninitialized -Wno-error=uninitialized\" -DCMAKE_CXX_FLAGS=\"-Wno-error=maybe-uninitialized -Wno-error=uninitialized\" -DPYTHON_EXECUTABLE=$ENV{PYTHON_PATH} .. && make -j VERBOSE=1
   COMMAND
       mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/fbgemm_build && cp -r include/ build/* ${CMAKE_CURRENT_BINARY_DIR}/fbgemm_build
   COMMAND
   cp ${CMAKE_CURRENT_BINARY_DIR}/fbgemm_build/libfbgemm.a ${CMAKE_CURRENT_BINARY_DIR}/lib/ && cp ${CMAKE_CURRENT_BINARY_DIR}/fbgemm_build/asmjit/libasmjit.a ${CMAKE_CURRENT_BINARY_DIR}/lib/
)

SET(FBGEMM_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/FBGEMM/include
)

LIST(APPEND FBGEMM_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/lib/libfbgemm.a)

MARK_AS_ADVANCED(
        FBGEMM_INCLUDE_DIR
        FBGEMM_LIBRARIES
        fbgemm
)



file(GLOB zendnn_src_common_cpp "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/common/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/gemm/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/gemm/f32/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/gemm/s8x8s32/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/matmul/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/reorder/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/rnn/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/brgemm/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/amx/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/bf16/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/f32/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/s8x8s32/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/injectors/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/lrn/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/matmul/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/prelu/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/rnn/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/shuffle/*.cpp")
set(GENERATED_CXX_ZEN
    ${zendnn_src_common_cpp}
  )

# Build ZENDNN
################################################################################

add_custom_target(libamdZenDNN ALL
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/lib/libamdZenDNN.a
)

add_custom_command(
    OUTPUT
         ${CMAKE_CURRENT_BINARY_DIR}/lib/libamdZenDNN.a
    WORKING_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN
    COMMAND
        make -j ZENDNN_BLIS_PATH=${CMAKE_CURRENT_BINARY_DIR}/blis_gcc_build AOCC=0 LPGEMM=1 LPGEMM_V4_2=1 BLIS_API=1 FBGEMM_INSTALL_PATH=${CMAKE_CURRENT_SOURCE_DIR}/third_party/FBGEMM FBGEMM_ENABLE=1 ARCHIVE=1 RELEASE=${BUILD_FLAG}
    COMMAND
        cp _out/lib/libamdZenDNN.a ${CMAKE_CURRENT_BINARY_DIR}/lib/
    DEPENDS
         ${zendnn_src_common_cpp}
    COMMAND
         make clean
)

add_dependencies(libamdZenDNN libamdblis libfbgemm)

SET(ZENDNN_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/local/include/
  /usr/local/include/zendnn/include
  /usr/local/opt/include
  /opt/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/inc
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/include
)
FIND_PATH(ZENDNN_INCLUDE_DIR NAMES zendnn_config.h zendnn.h zendnn_types.h zendnn_debug.h zendnn_version.h PATHS ${ZENDNN_INCLUDE_SEARCH_PATHS})
IF(NOT ZENDNN_INCLUDE_DIR)
     MESSAGE(STATUS "Could not find ZENDNN include.")
     RETURN()
ENDIF(NOT ZENDNN_INCLUDE_DIR)

LIST(APPEND ZENDNN_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/lib/libamdZenDNN.a)

MARK_AS_ADVANCED(
	ZENDNN_INCLUDE_DIR
	ZENDNN_LIBRARIES
        amdZenDNN
)

SET(ZENDNN_FOUND ON)
IF (NOT ZENDNN_FIND_QUIETLY)
    MESSAGE(STATUS "Found AOCL BLIS libraries: ${BLIS_LIBRARIES}")
    MESSAGE(STATUS "Found AOCL BLIS include  : ${BLIS_INCLUDE_DIR}")
    MESSAGE(STATUS "Found ZENDNN libraries   : ${ZENDNN_LIBRARIES}")
    MESSAGE(STATUS "Found ZENDNN include     : ${ZENDNN_INCLUDE_DIR}")
ENDIF (NOT ZENDNN_FIND_QUIETLY)

ENDIF (NOT ZENDNN_FOUND)
