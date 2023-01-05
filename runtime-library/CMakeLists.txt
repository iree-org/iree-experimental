# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.24)

project(IREE_RUNTIME_LIBRARY)
cmake_policy(SET CMP0069 NEW)

set(IREE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../iree" CACHE STRING "Main IREE source")

# Enable LTO if supported.
option(IREERT_ENABLE_LTO "Enable LTO (link time optimization) if supported" ON)
include(CheckIPOSupported)
check_ipo_supported(RESULT _ireert_lto_supported OUTPUT error)
if(IREERT_ENABLE_LTO)
  if(_ireert_lto_supported)
    message(STATUS "Enabling LTO")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "LTO not supported by toolchain bit requested (ignored)")
  endif()
endif()

# Optionally enabled shared library build mode. This is only supported for
# development as it precludes important optimizations.
option(BUILD_SHARED_LIBS "Build development shared runtime library" OFF)

# When building a shared library, further customize:
#   - Build with hidden visibility by default.
#   - Unhide API exported functions.
#   - Mark symbols for dllexport on Windows by default.
if(BUILD_SHARED_LIBS)
  message(STATUS "IREE runtime library with BUILD_SHARED_LIBS=ON is only supported for development!")
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  add_compile_definitions("IREE_API_ENABLE_VISIBILITY")
  add_compile_definitions("IREE_API_BUILDING_LIBRARY")
endif()

# Customize defaults.
option(IREE_BUILD_COMPILER "Disable compiler for runtime-library build" OFF)
option(IREE_BUILD_SAMPLES "Disable samples for runtime-library build" OFF)

add_subdirectory("${IREE_ROOT_DIR}" "iree_core" EXCLUDE_FROM_ALL)

#-------------------------------------------------------------------------------
# Build the ireert library
#-------------------------------------------------------------------------------

# IREE exposes its transitive objects and deps on special target properties.
set(_RUNTIME_ROOT_TARGET "iree::runtime::impl")
set(_RUNTIME_OBJECTS "$<REMOVE_DUPLICATES:$<GENEX_EVAL:$<TARGET_PROPERTY:${_RUNTIME_ROOT_TARGET},INTERFACE_IREE_TRANSITIVE_OBJECTS>>>")
set(_RUNTIME_LIBS "$<REMOVE_DUPLICATES:$<GENEX_EVAL:$<TARGET_PROPERTY:${_RUNTIME_ROOT_TARGET},INTERFACE_IREE_TRANSITIVE_OBJECT_LIBS>>>")
 # For debugging, write out objects and deps to files.
file(GENERATE OUTPUT "lib/ireert.objects.txt" CONTENT "${_RUNTIME_OBJECTS}\n")
file(GENERATE OUTPUT "lib/ireert.libs.txt" CONTENT "${_RUNTIME_LIBS}\n")

# Build the library (shared or static).
add_library(ireert ${_RUNTIME_OBJECTS})
target_include_directories(ireert INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/include")
target_link_libraries(ireert PUBLIC ${_RUNTIME_LIBS})
set_target_properties(ireert PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)

if(BUILD_SHARED_LIBS)
  # Import symbols from library on use.
  target_compile_options(ireert INTERFACE "-UIREE_API_BUILDING_LIBRARY")
  # Disallow undefined symbols in shared library mode.
  if(NOT IREE_ENABLE_ASAN
    AND NOT IREE_ENABLE_MSAN
    AND NOT IREE_ENABLE_TSAN
    AND NOT IREE_ENABLE_UBSAN)
    target_link_options(ireert PRIVATE
      $<$<PLATFORM_ID:Linux>:-Wl,--no-undefined>
      $<$<PLATFORM_ID:Darwin>:-Wl,-undefined,error>
    )
  endif()
endif()

#-------------------------------------------------------------------------------
# Populate headers from the source tree.
#-------------------------------------------------------------------------------

# Copy source tree headers.
function(_copy_runtime_source_headers)
  set(_RUNTIME_HEADER_SOURCE_DIR "${IREE_ROOT_DIR}/runtime/src")
  file(GLOB_RECURSE
    _RUNTIME_SRC_HEADERS
    RELATIVE "${_RUNTIME_HEADER_SOURCE_DIR}"
    "${_RUNTIME_HEADER_SOURCE_DIR}/*.h")
  foreach(_RUNTIME_HEADER ${_RUNTIME_SRC_HEADERS})
    set(_src_file "${_RUNTIME_HEADER_SOURCE_DIR}/${_RUNTIME_HEADER}")
    set(_dst_file "${CMAKE_CURRENT_BINARY_DIR}/include/${_RUNTIME_HEADER}")
    get_filename_component(_parent_dir "${_dst_file}" DIRECTORY)
    file(MAKE_DIRECTORY "${_parent_dir}")
    file(COPY_FILE
      "${_src_file}"
      "${_dst_file}"
      ONLY_IF_DIFFERENT)
  endforeach()
endfunction()
_copy_runtime_source_headers()

#-------------------------------------------------------------------------------
# Generate a test program.
#-------------------------------------------------------------------------------

add_executable(ireert_test test_api.c)
set_target_properties(ireert_test
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)
target_link_libraries(ireert_test PRIVATE ireert)
