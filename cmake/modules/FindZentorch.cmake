#******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

cmake_minimum_required(VERSION 3.1)

execute_process(
  COMMAND python -c "import zentorch; print(zentorch.__path__[0])"
  OUTPUT_VARIABLE ZENTORCH_PACKAGE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set ZENTORCH_LIBRARY to the path of the libzentorch.so
set(ZENTORCH_LIBRARY "${ZENTORCH_PACKAGE_DIR}/libzentorch.so")

# Check if the library file exists
if(EXISTS ${ZENTORCH_LIBRARY})
  set(ZENTORCH_FOUND TRUE)
  message(STATUS "FOUND libzentorch: ${ZENTORCH_LIBRARY}")
endif()

if(ZENTORCH_FOUND AND NOT TARGET Zentorch::Zentorch)
  add_library(Zentorch::Zentorch SHARED IMPORTED)
  set_target_properties(Zentorch::Zentorch PROPERTIES
      IMPORTED_LOCATION "${ZENTORCH_LIBRARY}"
  )
endif()

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
set(CMAKE_INSTALL_RPATH "${ZENTORCH_PACKAGE_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
