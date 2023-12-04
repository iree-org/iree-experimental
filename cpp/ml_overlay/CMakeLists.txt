# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_VULKAN_SPIRV OR
   NOT IREE_HAL_DRIVER_VULKAN)
  message(STATUS "Missing Vulkan backend and/or driver, skipping ml_overlay sample")
  return()
endif()

# This target statically links against Vulkan.
# One way to achieve this is by installing the Vulkan SDK from
# https://vulkan.lunarg.com/.
include(FindVulkan)
if(NOT Vulkan_FOUND)
  message(STATUS "Could not find Vulkan, skipping ml_overlay sample")
  return()
endif()

# vcpkg install imgui[vulkan-binding,sdl2-binding]
#   tested with versions 1.83 - 1.87#1
find_package(imgui)
if(NOT imgui_FOUND)
  message(STATUS "Could not find Dear ImGui, skipping ml_overlay sample")
  return()
endif()

# vcpkg install sdl2[vulkan]
#   tested with versions 2.0.14#4 - 2.0.22#1
find_package(SDL2)
if(NOT SDL2_FOUND)
  message(STATUS "Could not find SDL2, skipping ml_overlay sample")
  return()
endif()

# Define the sample executable.
set(_NAME "iree-samples-ml-overlay")
add_executable(${_NAME} "")
target_sources(${_NAME}
  PRIVATE
    overlay.cc
)
set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "ml-overlay")
target_include_directories(${_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(${_NAME}
  imgui::imgui
  SDL2::SDL2
  Vulkan::Vulkan
  #
  iree_base_internal_flags
  iree_base_internal_main
  iree_hal_drivers_vulkan_registration_registration
  iree_modules_hal_hal
  iree_tooling_context_util
  iree_vm_vm
)
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(_GUI_LINKOPTS "-SUBSYSTEM:WINDOWS")
  target_link_libraries(${_NAME}
    d3d11.lib
    dxgi.lib
  )
else()
  set(_GUI_LINKOPTS "")
endif()
target_link_options(${_NAME}
  PRIVATE
    ${_GUI_LINKOPTS}
)

message(STATUS "Configured ml-overlay sample successfully")
