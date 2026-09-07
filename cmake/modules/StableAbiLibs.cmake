#******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Portable stable-ABI library built alongside libzentorch.so:
#   libzentorch_stable.so — portable zentorch::* stable-ABI ops
#
# Links only libtorch_cpu.so (unversioned soname, no RPATH) so the .so can be
# loaded on any torch >= the build torch version.
#******************************************************************************

include_guard(GLOBAL)

# find_package(Torch) sets TORCH_INSTALL_PREFIX to the torch package root
# (…/site-packages/torch). Do not derive this from Torch_DIR — that path varies
# (share/cmake vs share/cmake/Torch) and breaks libtorch_cpu.so lookup.
if(NOT DEFINED TORCH_INSTALL_PREFIX)
  message(FATAL_ERROR
    "ZENTORCH stable-ABI build: TORCH_INSTALL_PREFIX is not set "
    "(find_package(Torch) must run before StableAbiLibs.cmake)")
endif()

set(_ZENTORCH_TORCH_CPU_LIB "${TORCH_INSTALL_PREFIX}/lib/libtorch_cpu.so")
if(NOT EXISTS "${_ZENTORCH_TORCH_CPU_LIB}")
  message(FATAL_ERROR
    "ZENTORCH stable-ABI build: libtorch_cpu.so not found at "
    "${_ZENTORCH_TORCH_CPU_LIB}")
endif()

set(_ZENTORCH_STABLE_CPP "${CMAKE_SOURCE_DIR}/src/cpu/cpp")

# Every op file whose kernels and STABLE_TORCH_LIBRARY_* registrations are
# fully stable-ABI. Each file registers itself into whichever library it is
# compiled into; libzentorch.so and libzentorch_stable.so are never loaded at
# the same time (see src/cpu/python/zentorch/__init__.py), so the same
# zentorch::* schemas can be defined by both.
#
# Globs and then subtracts, mirroring the zentorch target in CMakeLists.txt, so
# a newly migrated op is portable by default and an op that cannot go in has to
# be named below with a reason.
file(GLOB _ZENTORCH_STABLE_SOURCES "${_ZENTORCH_STABLE_CPP}/*.cpp")

# Register no ops: these exist only to back the pybind surface in _C.so, which
# links libzentorch.so directly.
list(REMOVE_ITEM _ZENTORCH_STABLE_SOURCES
  "${_ZENTORCH_STABLE_CPP}/Bindings.cpp"
  "${_ZENTORCH_STABLE_CPP}/Config.cpp"
  "${_ZENTORCH_STABLE_CPP}/Singletons.cpp"
  "${_ZENTORCH_STABLE_CPP}/Threading.cpp")

# Not yet portable. Each references ATen symbols that have no stable
# counterpart, so its ops resolve only through libzentorch.so:
#   CausalAttentionMask, GDN_ops - still on TORCH_LIBRARY with at::Tensor
#     kernels.
#   Sdpa_ref - registers through STABLE_TORCH_LIBRARY, but the AVX-512 flash
#     kernel and the at::native fallback both still take at::Tensor.
#   shim_cpu_zentorch - the AOTI shim layer, inherently ATen-handle based.
# Drop a file from this list once its kernel takes torch::stable::Tensor
# throughout, then re-run the ABI audit (see below).
list(REMOVE_ITEM _ZENTORCH_STABLE_SOURCES
  "${_ZENTORCH_STABLE_CPP}/CausalAttentionMask.cpp"
  "${_ZENTORCH_STABLE_CPP}/GDN_ops.cpp"
  "${_ZENTORCH_STABLE_CPP}/Sdpa_ref.cpp"
  "${_ZENTORCH_STABLE_CPP}/shim_cpu_zentorch.cpp")

add_library(zentorch_stable SHARED ${_ZENTORCH_STABLE_SOURCES})
# CPUkernels is not needed here: its only consumers (Sdpa_ref.cpp and
# GDN_ops.cpp) are excluded from _ZENTORCH_STABLE_SOURCES above.
add_dependencies(zentorch_stable zendnnl::zendnnl_archive)
target_compile_features(zentorch_stable PUBLIC cxx_std_17)
# Lets a shared op file drop code that only libzentorch.so needs, such as
# at::Tensor template instantiations used by ATen-only callers.
target_compile_definitions(zentorch_stable PRIVATE ZENTORCH_STABLE_ABI_LIB)
# Same remap as the zentorch target so ZENTORCH_CHECK and similar macros
# print paths relative to the source tree instead of absolute build-host paths.
target_compile_options(zentorch_stable PRIVATE
  -fopenmp
  -ffile-prefix-map=${CMAKE_SOURCE_DIR}/=)
target_include_directories(zentorch_stable PRIVATE
  ${_ZENTORCH_STABLE_CPP}
  ${ZENDNNL_LIBRARY_INC_DIR})
# SYSTEM so -Wall -Werror does not apply inside torch's own headers. The
# zentorch target gets this for free by linking the Torch imported target,
# whose interface includes are already SYSTEM; listing TORCH_INCLUDE_DIRS
# directly here does not, which left this target failing on warnings in torch
# headers that libzentorch.so compiled through fine.
target_include_directories(zentorch_stable SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(zentorch_stable PRIVATE
  zendnnl::zendnnl_archive
  OpenMP::OpenMP_CXX
  ${CMAKE_DL_LIBS}
  rt)
target_link_options(zentorch_stable PRIVATE
  "LINKER:--exclude-libs,ALL"
  "LINKER:--no-as-needed"
  "${_ZENTORCH_TORCH_CPU_LIB}"
  "LINKER:--as-needed")
set_target_properties(zentorch_stable PROPERTIES
  OUTPUT_NAME zentorch_stable
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/
  BUILD_RPATH ""
  INSTALL_RPATH "")

if(DEFINED INSTALL_LIB_DIR)
  add_custom_command(
    TARGET zentorch_stable POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      ${CMAKE_SOURCE_DIR}/${INSTALL_LIB_DIR}/${PROJECT_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy
      $<TARGET_FILE:zentorch_stable>
      ${CMAKE_SOURCE_DIR}/${INSTALL_LIB_DIR}/${PROJECT_NAME}/)
endif()

# Echo what the glob resolved to, so a build log shows which ops the portable
# library actually got rather than only what it was asked for. setup.py audits
# the result with scripts/check-torch-abi.py once make returns; the linker will
# not catch an ATen symbol here, since a shared object may leave symbols
# undefined and ATen resolves out of the libtorch_cpu.so this library already
# links. That matters most because the glob above pulls in any new .cpp without
# anyone opting it in.
set(_ZENTORCH_STABLE_NAMES "")
foreach(_src IN LISTS _ZENTORCH_STABLE_SOURCES)
  get_filename_component(_name "${_src}" NAME)
  list(APPEND _ZENTORCH_STABLE_NAMES "${_name}")
endforeach()
list(JOIN _ZENTORCH_STABLE_NAMES " " _ZENTORCH_STABLE_NAMES)
message(STATUS "zentorch stable-ABI lib: zentorch_stable -> libzentorch_stable.so")
message(STATUS "  stable-ABI sources: ${_ZENTORCH_STABLE_NAMES}")
