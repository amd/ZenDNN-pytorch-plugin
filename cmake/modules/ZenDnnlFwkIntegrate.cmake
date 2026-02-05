# *******************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
# *******************************************************************************/
include_guard(GLOBAL)
include(ExternalProject)
include(ZenDnnlFwkMacros)

message(AUTHOR_WARNING "(ZENDNNL) please ensure all zendnnl variables are set properly.")

# find openmp
find_package(OpenMP REQUIRED QUIET)
# set ZenDNNL source, build and install folders
zendnnl_add_option(NAME ZENDNNL_SOURCE_DIR
  VALUE "${CMAKE_SOURCE_DIR}/third_party/ZenDNN"
  TYPE PATH
  CACHE_STRING "zendnnl_source_dir"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_BINARY_DIR
  VALUE "${CMAKE_CURRENT_BINARY_DIR}/zendnnl"
  TYPE PATH
  CACHE_STRING "zendnnl_binary_dir"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_INSTALL_PREFIX
  VALUE "${CMAKE_CURRENT_BINARY_DIR}/lib"
  TYPE PATH
  CACHE_STRING "zendnnl_install_dir"
  COMMAND_LIST ZNL_CMAKE_ARGS)

## general zendnnl options
# set ZenDNNL framework build, this should on ON to avoid standalone build.
zendnnl_add_option(NAME ZENDNNL_FWK_BUILD
  VALUE ON
  TYPE BOOL
  CACHE_STRING "zendnnl framework build"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set zendnnl build option, default is Release.
zendnnl_add_option(NAME ZENDNNL_BUILD_TYPE
  VALUE "Release"
  TYPE STRING
  CACHE_STRING "zendnnl build type"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set zendnnl log level.
zendnnl_add_option(NAME ZENDNNL_MESSAGE_LOG_LEVEL
  VALUE "DEBUG"
  TYPE STRING
  CACHE_STRING "zendnnl message log level"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set zendnnl verbose makefile option.
zendnnl_add_option(NAME ZENDNNL_VERBOSE_MAKEFILE
  VALUE ON
  TYPE BOOL
  CACHE_STRING "zendnnl verbose makefile"
  COMMAND_LIST ZNL_CMAKE_ARGS)

## components options
# set building zendnnl examples, default os OFF.
zendnnl_add_option(NAME ZENDNNL_BUILD_EXAMPLES
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl examples"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set building zendnnl gtests, default os OFF.
zendnnl_add_option(NAME ZENDNNL_BUILD_GTEST
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl gtests"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set building zendnnl doxygen documentation, default os OFF.
zendnnl_add_option(NAME ZENDNNL_BUILD_DOXYGEN
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl doxygen documentation"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set building zendnnl benchmarking tool, default os OFF.
zendnnl_add_option(NAME ZENDNNL_BUILD_BENCHDNN
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl benchdnn"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set zendnnl code coverage option, default os OFF.
zendnnl_add_option(NAME ZENDNNL_CODE_COVERAGE
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl code coverage"
  COMMAND_LIST ZNL_CMAKE_ARGS)

## dependencies
# set if zendnnl depends on amdblis. this should bf OFF only if
# aocldlp dependency is ON.
zendnnl_add_option(NAME ZENDNNL_DEPENDS_AMDBLIS
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "zendnnl amdblis dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set if zendnnl depends on aocldlp. this should bf ON only if
# amdblis dependency is OFF.
zendnnl_add_option(NAME ZENDNNL_DEPENDS_AOCLDLP
  VALUE ON
  TYPE BOOL
  CACHE_STRING "zendnnl aocldlp dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set if zendnnl depends on onednn, default is ON.
zendnnl_add_option(NAME ZENDNNL_DEPENDS_ONEDNN
  VALUE ON
  TYPE BOOL
  CACHE_STRING "zendnnl onednn dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set if zendnnl depends on libxsmm, default is ON.
zendnnl_add_option(NAME ZENDNNL_DEPENDS_LIBXSMM
  VALUE ON
  TYPE BOOL
  CACHE_STRING "zendnnl libxsmm dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set path of amdblis if amdblis is injected. if the framework
# does not inject it, set it to "" (empty string).
zendnnl_add_option(NAME ZENDNNL_AMDBLIS_FWK_DIR
  VALUE ""
  TYPE PATH
  CACHE_STRING "zendnnl amdblis framework path"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_AOCLDLP_FWK_DIR
  VALUE ""
  TYPE PATH
  CACHE_STRING "zendnnl aocldlp framework path"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set path of onednn if onednn is injected. if the framework
# does not inject it, set it to "" (empty string).
zendnnl_add_option(NAME ZENDNNL_ONEDNN_FWK_DIR
  VALUE ""
  TYPE PATH
  CACHE_STRING "zendnnl onednnn framework path"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# try to find pre-built package
set(zendnnl_ROOT "${ZENDNNL_INSTALL_PREFIX}/zendnnl")
set(zendnnl_DIR "${zendnnl_ROOT}/lib/cmake")
find_package(zendnnl QUIET)
if(zendnnl_FOUND)
  message(STATUS "(ZENDNNL) ZENDNNL FOUND AT ${zendnnl_ROOT}")
  message(STATUS "(ZENDNNL) if zendnnl options are changed from previous build,")
  message(STATUS "(ZENDNNL) they will not be reflected")
  message(STATUS "(ZENDNNL) If options are changed, please do a clean build.")
  if(TARGET zendnnl::zendnnl_archive)
    set_target_properties(zendnnl::zendnnl_archive
      PROPERTIES IMPORTED_GLOBAL ON)
  else()
    message(FATAL_ERROR "(ZENDNNL) zendnnl installation does not have imported target zendnnl::zendnnl_archive")
  endif()
else()
  message(STATUS "(ZENDNNL) ZENDNNL NOT FOUND, will be built as an external project.")
  message(STATUS "(ZENDNNL) ZNL_CMAKE_ARGS=${ZNL_CMAKE_ARGS}")

  ExternalProject_ADD(fwk_zendnnl
    SOURCE_DIR  "${ZENDNNL_SOURCE_DIR}"
    BINARY_DIR  "${ZENDNNL_BINARY_DIR}"
    CMAKE_ARGS  "${ZNL_CMAKE_ARGS}"
    BUILD_COMMAND cmake --build . --target all -j
    INSTALL_COMMAND "")
  
  list(APPEND ZENDNNL_CLEAN_FILES "${ZENDNNL_BINARY_DIR}")
  list(APPEND ZENDNNL_CLEAN_FILES "${ZENDNNL_INSTALL_PREFIX}")
  set_target_properties(fwk_zendnnl
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${ZENDNNL_CLEAN_FILES}")

  # framwork dependencies
  # add_dependencies(fwk_zendnnl <injected dependency targets>)
  # add_dependencies(fwk_zendnnl libamdblis)
  get_target_property(FWK_ZENDNNL_DEPENDS fwk_zendnnl MANUALLY_ADDED_DEPENDENCIES)
  if(${FWK_ZENDNNL_DEPENDS} STREQUAL "FWK_ZENDNNL_DEPENDS-NOTFOUND")
    message(AUTHOR_WARNING "(ZENDNNL) please ensure fwk_zendnnl depends on injected dependencies targets")
  else()
    message(STATUS "fwk_zendnnl dependencies : ${FWK_ZENDNNL_DEPENDS}")
  endif()

  # zendnnl library
  set(ZENDNNL_LIBRARY_INC_DIR "${ZENDNNL_INSTALL_PREFIX}/zendnnl/include")
  set(ZENDNNL_LIBRARY_LIB_DIR "${ZENDNNL_INSTALL_PREFIX}/zendnnl/lib")

  if(NOT EXISTS ${ZENDNNL_LIBRARY_INC_DIR})
    file(MAKE_DIRECTORY ${ZENDNNL_LIBRARY_INC_DIR})
  endif()

  add_library(zendnnl_library STATIC IMPORTED GLOBAL)
  add_dependencies(zendnnl_library fwk_zendnnl)
  if(MSVC)
    set_target_properties(zendnnl_library
      PROPERTIES
      IMPORTED_LOCATION "${ZENDNNL_LIBRARY_LIB_DIR}/zendnnl_archive.lib"
      INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}"
      INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}")
  else()
    set_target_properties(zendnnl_library
      PROPERTIES
      IMPORTED_LOCATION "${ZENDNNL_LIBRARY_LIB_DIR}/libzendnnl_archive.a"
      INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}"
      INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}")
  endif()

  if(NOT MSVC)
    target_link_options(zendnnl_library INTERFACE "-fopenmp")
  endif()
  target_link_libraries(zendnnl_library
    INTERFACE OpenMP::OpenMP_CXX
    INTERFACE ${CMAKE_DL_LIBS})

  # json dependency
  zendnnl_add_dependency(NAME json
    PATH "${ZENDNNL_INSTALL_PREFIX}/deps/json"
    ALIAS "nlohmann_json::nlohmann_json"
    DEPENDS fwk_zendnnl
    INCLUDE_ONLY)

  target_link_libraries(zendnnl_library INTERFACE nlohmann_json::nlohmann_json)

  if (ZENDNNL_DEPENDS_LIBXSMM)
    # libxsmm dependency
    if(MSVC)
      set(_LIBXSMM_ARCHIVE "libxsmm.lib")
    else()
      set(_LIBXSMM_ARCHIVE "libxsmm.a")
    endif()
    zendnnl_add_dependency(NAME libxsmm
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/libxsmm"
      ARCHIVE_FILE "${_LIBXSMM_ARCHIVE}"
      ALIAS "libxsmm::libxsmm"
      DEPENDS fwk_zendnnl)

    target_link_libraries(zendnnl_library INTERFACE libxsmm::libxsmm)
  endif()

  if(MSVC)
    set(_AOCLUTILS_ARCHIVE "aoclutils.lib")
    set(_AUCPUID_ARCHIVE "au_cpuid.lib")
    set(_DNNL_ARCHIVE "dnnl.lib")
  else()
    set(_AOCLUTILS_ARCHIVE "libaoclutils.a")
    set(_AUCPUID_ARCHIVE "libau_cpuid.a")
    set(_DNNL_ARCHIVE "libdnnl.a")
  endif()

    # aoclutils dependency
  if (DEFINED ENV{ZENDNNL_MANYLINUX_BUILD})

    zendnnl_add_dependency(NAME aoclutils
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils"
      LIB_SUFFIX lib64
      ARCHIVE_FILE "${_AOCLUTILS_ARCHIVE}"
      ALIAS "au::aoclutils"
      DEPENDS fwk_zendnnl)
    target_link_libraries(zendnnl_library INTERFACE au::aoclutils)

    zendnnl_add_dependency(NAME aucpuid
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils"
      LIB_SUFFIX lib64
      ARCHIVE_FILE "${_AUCPUID_ARCHIVE}"
      ALIAS "au::au_cpuid"
      DEPENDS fwk_zendnnl)

    target_link_libraries(zendnnl_library INTERFACE au::au_cpuid)

    zendnnl_add_dependency(NAME onednn
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/onednn"
      LIB_SUFFIX lib64
      ARCHIVE_FILE "${_DNNL_ARCHIVE}"
      ALIAS "DNNL::dnnl"
      DEPENDS fwk_zendnnl)
    target_link_libraries(zendnnl_library INTERFACE DNNL::dnnl)

  else()
    zendnnl_add_dependency(NAME aoclutils
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils"
      ARCHIVE_FILE "${_AOCLUTILS_ARCHIVE}"
      ALIAS "au::aoclutils"
      DEPENDS fwk_zendnnl)

    target_link_libraries(zendnnl_library INTERFACE au::aoclutils)

    zendnnl_add_dependency(NAME aucpuid
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils"
      ARCHIVE_FILE "${_AUCPUID_ARCHIVE}"
      ALIAS "au::au_cpuid"
      DEPENDS fwk_zendnnl)

    target_link_libraries(zendnnl_library INTERFACE au::au_cpuid)

    zendnnl_add_dependency(NAME onednn
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/onednn"
      ARCHIVE_FILE "${_DNNL_ARCHIVE}"
      ALIAS "DNNL::dnnl"
      DEPENDS fwk_zendnnl)

    target_link_libraries(zendnnl_library INTERFACE DNNL::dnnl)

  endif()

  # amdblis dependency
  if (ZENDNNL_DEPENDS_AMDBLIS)
      if(MSVC)
        set(_BLIS_ARCHIVE "blis-mt.lib")
      else()
        set(_BLIS_ARCHIVE "libblis-mt.a")
      endif()
      zendnnl_add_dependency(NAME amdblis
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/amdblis"
        ARCHIVE_FILE "${_BLIS_ARCHIVE}"
        ALIAS "amdblis::amdblis_archive"
        DEPENDS fwk_zendnnl)

      target_link_libraries(zendnnl_library INTERFACE amdblis::amdblis_archive)
  endif()

  if (ZENDNNL_DEPENDS_AOCLDLP)
    if(MSVC)
      set(_AOCLDLP_ARCHIVE "aocl-dlp.lib")
    else()
      set(_AOCLDLP_ARCHIVE "libaocl-dlp.a")
    endif()
    zendnnl_add_dependency(NAME aocldlp
    PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aocldlp"
    ARCHIVE_FILE "${_AOCLDLP_ARCHIVE}"
    ALIAS "aocldlp::aocldlp"
    DEPENDS fwk_zendnnl)

    target_link_libraries(zendnnl_library INTERFACE aocldlp::aocldlp)
  endif()

  add_library(zendnnl::zendnnl_archive ALIAS zendnnl_library)

endif()
