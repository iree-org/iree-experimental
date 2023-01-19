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
if(APPLE)
  # Enabling LTO on macOS/iOS is fine to build frameworks. But when we merge
  # them into an XCframework using the command
  # xcodebuild -create-xcframework -framework build-ios-sim/ireert.framework ...
  # it will complain "unable to find any architecture information in the binary",
  # even though lipo -info could recognize the architecture of each framework.
  set(IREERT_ENABLE_LTO OFF)
endif()

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
# Build the iree library
#-------------------------------------------------------------------------------

# IREE exposes its transitive objects and deps on special target properties.
set(_RUNTIME_ROOT_TARGET "iree::runtime::impl")
set(_RUNTIME_OBJECTS "$<REMOVE_DUPLICATES:$<GENEX_EVAL:$<TARGET_PROPERTY:${_RUNTIME_ROOT_TARGET},INTERFACE_IREE_TRANSITIVE_OBJECTS>>>")
set(_RUNTIME_LIBS "$<REMOVE_DUPLICATES:$<GENEX_EVAL:$<TARGET_PROPERTY:${_RUNTIME_ROOT_TARGET},INTERFACE_IREE_TRANSITIVE_OBJECT_LIBS>>>")
# For debugging, write out objects and deps to files.
file(GENERATE OUTPUT "lib/iree.objects.txt" CONTENT "${_RUNTIME_OBJECTS}\n")
file(GENERATE OUTPUT "lib/iree.libs.txt" CONTENT "${_RUNTIME_LIBS}\n")

# List all of the headers so that we can add them when we call add
# library. This has to be done so that these headers files can be
# added to the macOS/iOS
# framework. c.f. https://gitlab.kitware.com/cmake/cmake/-/issues/16739#note_641049
#
# It doesn't work if we add these headers later using target_sources.
#
# Even though these headers are listed in add_library, that is not yet
# enough to add them to the framework. We need to add them to
#
#   set_target_property(iree PROPERTIES PUBLIC_HEADER header_files)
#
# for that to happen. Unfortunately, CMake would flatten the directory
# structure of the header files and put all header files in the
# framework's "Headers/" directory. In the case of IREE runtime, we
# will have two debugging.h files in the same directory, so it doesn't
# work.
#
# We could also call
#
#  set_source_files_properties(
#     a_header PROPERTIES MACOSX_PACKAGE_LOCATION Headers/where_it_goes)
#
# for each header file, which is what we will do later in this
# file. Note that this method doesn't work with PUBLIC_HEADER, so we
# can't use both of them at the same
# time. c.f. https://gitlab.kitware.com/cmake/cmake/-/issues/16739#note_641095
set(_RUNTIME_HEADER_SOURCE_DIR "${IREE_ROOT_DIR}/runtime/src")
file(GLOB_RECURSE
  _RUNTIME_SRC_HEADERS_FULLPATH
  "${_RUNTIME_HEADER_SOURCE_DIR}/*.h")


# Build the library (shared or static).
add_library(iree
  ${_RUNTIME_OBJECTS}
  ${_RUNTIME_SRC_HEADERS_FULLPATH}
)
target_include_directories(iree INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/include")
target_link_libraries(iree PUBLIC ${_RUNTIME_LIBS})
set_target_properties(iree PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)

if(BUILD_SHARED_LIBS)
  # Import symbols from library on use.
  target_compile_options(iree INTERFACE "-UIREE_API_BUILDING_LIBRARY")
  # Disallow undefined symbols in shared library mode.
  if(NOT IREE_ENABLE_ASAN
    AND NOT IREE_ENABLE_MSAN
    AND NOT IREE_ENABLE_TSAN
    AND NOT IREE_ENABLE_UBSAN)
    target_link_options(iree PRIVATE
      $<$<PLATFORM_ID:Linux>:-Wl,--no-undefined>
      $<$<PLATFORM_ID:Darwin>:-Wl,-undefined,error>
    )
  endif()
endif()


if(APPLE)
  set_target_properties(iree PROPERTIES
    FRAMEWORK TRUE
    FRAMEWORK_VERSION A
    MACOSX_FRAMEWORK_IDENTIFIER dev.iree.iree
    VERSION 16.4.0
    SOVERSION 1.0.0
    XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED OFF
    PUBLIC_HEADER "${_RUNTIME_SRC_HEADERS_FULL_PATH}"
  )

  foreach(_RUNTIME_HEADER ${_RUNTIME_SRC_HEADERS_FULLPATH})
    string(REPLACE ${_RUNTIME_HEADER_SOURCE_DIR}/iree "" _RUNTIME_HEADER_RELPATH ${_RUNTIME_HEADER})
    get_filename_component(_REL_DIR "${_RUNTIME_HEADER_RELPATH}" DIRECTORY)
    set_source_files_properties(
      ${_RUNTIME_HEADER} PROPERTIES MACOSX_PACKAGE_LOCATION Headers/${_REL_DIR})
  endforeach()
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

add_executable(iree_test test_api.c)
set_target_properties(iree_test
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)
target_link_libraries(iree_test PRIVATE iree)
