// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <SDL.h>
#include <SDL_syswm.h>
#include <SDL_vulkan.h>
#include <imgui.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_vulkan.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>

#include <cstring>
#include <set>
#include <vector>

// IREE's C API:
#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/registration/driver_module.h"
#include "iree/modules/hal/module.h"
#include "iree/tooling/context_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// HACK:
#include "iree/hal/drivers/vulkan/base_buffer.h"
#include "iree/hal/drivers/vulkan/native_semaphore.h"

// Capture:
#include <d3d11.h>
#include <dxgi1_2.h>

using iree::Status;
using iree::vm::ref;

IREE_FLAG(bool, always_on_top, true,
          "Start with the overlay window always on top of all other windows.");
IREE_FLAG(bool, overlay, true, "Display overlay by default.");
IREE_FLAG(int32_t, x, -1, "Initial window X location.");
IREE_FLAG(int32_t, y, -1, "Initial window Y location.");
IREE_FLAG(int32_t, width, 1024, "Capture window width in pixels.");
IREE_FLAG(int32_t, height, 1024, "Capture window width in pixels.");
IREE_FLAG(bool, imgui_demo_window, false, "Show the imgui demo window.");

const char* module_path = "";  // argv[1]
IREE_FLAG(string, filter, "",
          "Name of a filter function in the module to apply by default.");

iree_status_t load_module(iree_vm_instance_t* instance, const char* module_path,
                          iree_allocator_t host_allocator,
                          iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, module_path);

  // Fetch the file contents into memory.
  // We could map the memory here if we wanted to and were coming from a file
  // on disk.
  iree_file_contents_t* file_contents = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_file_read_contents(module_path, IREE_FILE_READ_FLAG_DEFAULT,
                                  host_allocator, &file_contents));

  // Try to load the module as bytecode (all we have today that we can use).
  // We could sniff the file ID and switch off to other module types.
  // The module takes ownership of the file contents (when successful).
  iree_vm_module_t* module = NULL;
  iree_status_t status = iree_vm_bytecode_module_create(
      instance, file_contents->const_buffer,
      iree_file_contents_deallocator(file_contents), host_allocator, &module);

  if (iree_status_is_ok(status)) {
    *out_module = module;
  } else {
    iree_file_contents_free(file_contents);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#define CHECK_SUCCEEDED(hr)                                                \
  if (!SUCCEEDED(hr)) {                                                    \
    IREE_CHECK_OK(                                                         \
        iree_make_status(IREE_STATUS_INTERNAL, "HR failure: %08X", (hr))); \
  }

class CaptureProvider {
 public:
  static std::unique_ptr<CaptureProvider> FromWindow(HWND window) {
    D3D_DRIVER_TYPE DriverTypes[] = {
        D3D_DRIVER_TYPE_HARDWARE,
        D3D_DRIVER_TYPE_WARP,
        D3D_DRIVER_TYPE_REFERENCE,
    };
    UINT NumDriverTypes = IREE_ARRAYSIZE(DriverTypes);
    D3D_FEATURE_LEVEL FeatureLevels[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_1,
    };
    UINT NumFeatureLevels = IREE_ARRAYSIZE(FeatureLevels);
    HRESULT hr = 0;
    ID3D11Device* device = nullptr;
    D3D_FEATURE_LEVEL FeatureLevel = D3D_FEATURE_LEVEL_1_0_CORE;
    ID3D11DeviceContext* device_context = nullptr;
    for (UINT DriverTypeIndex = 0; DriverTypeIndex < NumDriverTypes;
         ++DriverTypeIndex) {
      hr = D3D11CreateDevice(nullptr, DriverTypes[DriverTypeIndex], nullptr, 0,
                             FeatureLevels, NumFeatureLevels, D3D11_SDK_VERSION,
                             &device, &FeatureLevel, &device_context);
      if (SUCCEEDED(hr)) break;
    }
    CHECK_SUCCEEDED(hr);

    IDXGIDevice* DxgiDevice = nullptr;
    hr = device->QueryInterface(__uuidof(IDXGIDevice),
                                reinterpret_cast<void**>(&DxgiDevice));
    CHECK_SUCCEEDED(hr);

    IDXGIAdapter* DxgiAdapter = nullptr;
    hr = DxgiDevice->GetParent(__uuidof(IDXGIAdapter),
                               reinterpret_cast<void**>(&DxgiAdapter));
    DxgiDevice->Release();
    DxgiDevice = nullptr;
    CHECK_SUCCEEDED(hr);

    UINT Output = 0;

    IDXGIOutput* DxgiOutput = nullptr;
    hr = DxgiAdapter->EnumOutputs(Output, &DxgiOutput);
    DxgiAdapter->Release();
    DxgiAdapter = nullptr;
    CHECK_SUCCEEDED(hr);

    DXGI_OUTPUT_DESC OutputDesc;
    DxgiOutput->GetDesc(&OutputDesc);

    // QI for Output 1
    IDXGIOutput1* DxgiOutput1 = nullptr;
    hr = DxgiOutput->QueryInterface(__uuidof(DxgiOutput1),
                                    reinterpret_cast<void**>(&DxgiOutput1));
    DxgiOutput->Release();
    DxgiOutput = nullptr;
    CHECK_SUCCEEDED(hr);

    // Create desktop duplication
    IDXGIOutputDuplication* DeskDupl = nullptr;
    hr = DxgiOutput1->DuplicateOutput(device, &DeskDupl);
    DxgiOutput1->Release();
    DxgiOutput1 = nullptr;
    CHECK_SUCCEEDED(hr);

    return std::unique_ptr<CaptureProvider>(new CaptureProvider(
        window, device, device_context, DeskDupl, OutputDesc));
  }

  ~CaptureProvider() {
    ReleaseFrame();
    if (m_DeskDupl) {
      m_DeskDupl->Release();
      m_DeskDupl = nullptr;
    }
    if (device_context_) {
      device_context_->Release();
      device_context_ = nullptr;
    }
    if (device_) {
      device_->Release();
      device_ = nullptr;
    }
  }

  // Returns true if the frame was captured successfully.
  bool AcquireFrame() {
    ReleaseFrame();

    DXGI_OUTDUPL_FRAME_INFO FrameInfo;
    IDXGIResource* DesktopResource = nullptr;
    HRESULT hr =
        m_DeskDupl->AcquireNextFrame(500, &FrameInfo, &DesktopResource);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
      return false;
    }
    CHECK_SUCCEEDED(hr);

    IDXGIResource1* resource1 = nullptr;
    hr = DesktopResource->QueryInterface(__uuidof(IDXGIResource1),
                                         reinterpret_cast<void**>(&resource1));
    CHECK_SUCCEEDED(hr);
    hr = resource1->CreateSharedHandle(NULL, DXGI_SHARED_RESOURCE_READ, NULL,
                                       &image_handle_);
    resource1->Release();
    resource1 = nullptr;
    CHECK_SUCCEEDED(hr);

    hr = DesktopResource->QueryInterface(
        __uuidof(ID3D11Texture2D),
        reinterpret_cast<void**>(&m_AcquiredDesktopImage));
    DesktopResource->Release();
    DesktopResource = nullptr;
    CHECK_SUCCEEDED(hr);

    return true;
  }

  ID3D11Texture2D* current_frame() { return m_AcquiredDesktopImage; }
  HANDLE current_frame_handle() { return image_handle_; }

  void ReleaseFrame() {
    if (m_AcquiredDesktopImage) {
      CloseHandle(image_handle_);
      HRESULT hr = m_DeskDupl->ReleaseFrame();
      CHECK_SUCCEEDED(hr);
      m_AcquiredDesktopImage->Release();
      m_AcquiredDesktopImage = nullptr;
      image_handle_ = nullptr;
    }
  }

 private:
  CaptureProvider(HWND window, ID3D11Device* device,
                  ID3D11DeviceContext* device_context,
                  IDXGIOutputDuplication* DeskDupl, DXGI_OUTPUT_DESC OutputDesc)
      : window_(window),
        device_(device),
        device_context_(device_context),
        m_DeskDupl(DeskDupl),
        m_OutputDesc(OutputDesc) {}

  HWND window_ = nullptr;
  ID3D11Device* device_ = nullptr;
  ID3D11DeviceContext* device_context_ = nullptr;
  IDXGIOutputDuplication* m_DeskDupl = nullptr;
  ID3D11Texture2D* m_AcquiredDesktopImage = nullptr;
  HANDLE image_handle_ = nullptr;
  DXGI_OUTPUT_DESC m_OutputDesc;
};

#if 0

class Pipeline {
 public:
  virtual ~Pipeline() {
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  Status X();

 protected:
  Pipeline(iree_vm_instance_t* instance) {
    //
    context_;
  }

  iree_vm_context_t* context_ = nullptr;
};

class Model {
 public:
  virtual ~Model() {
    iree_vm_release(module_);
    iree_vm_instance_release(instance_);
  }

  iree_vm_module_t* module() const { return module_; }

 protected:
  Model(iree_vm_instance_t* instance, iree_vm_module_t* module) {
    instance_ = instance;
    iree_vm_instance_retain(instance_);
    module_ = module;
    iree_vm_module_retain(module_);
  }

 private:
  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_module_t* module_ = nullptr;
};

class ImageFilterModel : public Model {
 public:
  static std::unique_ptr<ImageFilterModel> FromModule(
      iree_vm_module_t* module) {
    //
  }
};

static std::unique_ptr<Model> LoadModelFromFile(std::string path) {
  //
}

// command line for vmfb path
// reload model from path
// drag drop for vmfb?
// iree.reflection for I/O
// class for filter?
// Model
// ImageFilterModel
//   -> rgba -> rgba
//   async-external only
//   any additional outputs besides image first are shown in imgui?
//
// https://github.com/ocornut/imgui/wiki/Image-Loading-and-Displaying-Examples#Example-for-Vulkan-users
// ImGui_ImplVulkan_AddTexture to take
// ImGui_ImplVulkan_AddTexture(bd->FontSampler, bd->FontView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

#endif

static VkAllocationCallbacks* g_Allocator = NULL;
static VkInstance g_Instance = VK_NULL_HANDLE;
static VkPhysicalDevice g_PhysicalDevice = VK_NULL_HANDLE;
static VkDevice g_Device = VK_NULL_HANDLE;
static uint32_t g_QueueFamily = (uint32_t)-1;
static VkQueue g_Queue = VK_NULL_HANDLE;
static VkPipelineCache g_PipelineCache = VK_NULL_HANDLE;
static VkDescriptorPool g_DescriptorPool = VK_NULL_HANDLE;

static ImGui_ImplVulkanH_Window g_MainWindowData;
static uint32_t g_MinImageCount = 2;
static bool g_SwapChainRebuild = false;
static int g_SwapChainResizeWidth = 0;
static int g_SwapChainResizeHeight = 0;

static std::unique_ptr<CaptureProvider> capture_provider;

static PFN_vkGetMemoryWin32HandlePropertiesKHR
    p_vkGetMemoryWin32HandlePropertiesKHR = nullptr;
static PFN_vkGetImageMemoryRequirements2KHR p_vkGetImageMemoryRequirements2KHR =
    nullptr;

static uint32_t findMemoryType(VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(g_PhysicalDevice, &memoryProperties);
  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
    if (((memoryProperties.memoryTypes[i].propertyFlags & properties) ==
         properties))
      return i;
  }
  return -1;
}
static uint32_t findMemoryType(uint32_t memoryTypeBits,
                               VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(g_PhysicalDevice, &memoryProperties);
  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
    if ((memoryTypeBits & (1 << i)) &&
        ((memoryProperties.memoryTypes[i].propertyFlags & properties) ==
         properties))
      return i;
  }
  return -1;
}

static void check_vk_result(VkResult err) {
  if (err == 0) return;
  fprintf(stderr, "VkResult: %d\n", err);
  abort();
}

// Returns the names of the Vulkan layers used for the given IREE
// |extensibility_set| and |features|.
std::vector<const char*> GetIreeLayers(
    iree_hal_vulkan_extensibility_set_t extensibility_set,
    iree_hal_vulkan_features_t features) {
  iree_host_size_t required_count;
  iree_hal_vulkan_query_extensibility_set(
      features, extensibility_set, /*string_capacity=*/0, &required_count,
      /*out_string_values=*/NULL);
  std::vector<const char*> layers(required_count);
  iree_hal_vulkan_query_extensibility_set(features, extensibility_set,
                                          layers.size(), &required_count,
                                          layers.data());
  return layers;
}

// Returns the names of the Vulkan extensions used for the given IREE
// |extensibility_set| and |features|.
std::vector<const char*> GetIreeExtensions(
    iree_hal_vulkan_extensibility_set_t extensibility_set,
    iree_hal_vulkan_features_t features) {
  iree_host_size_t required_count;
  iree_hal_vulkan_query_extensibility_set(
      features, extensibility_set, /*string_capacity=*/0, &required_count,
      /*out_string_values=*/NULL);
  std::vector<const char*> extensions(required_count);
  iree_hal_vulkan_query_extensibility_set(features, extensibility_set,
                                          extensions.size(), &required_count,
                                          extensions.data());
  return extensions;
}

// Returns the names of the Vulkan extensions used for the given IREE
// |vulkan_features|.
std::vector<const char*> GetDeviceExtensions(
    VkPhysicalDevice physical_device,
    iree_hal_vulkan_features_t vulkan_features) {
  std::vector<const char*> iree_required_extensions = GetIreeExtensions(
      IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_REQUIRED,
      vulkan_features);
  std::vector<const char*> iree_optional_extensions = GetIreeExtensions(
      IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
      vulkan_features);

  uint32_t extension_count = 0;
  check_vk_result(vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &extension_count, nullptr));
  std::vector<VkExtensionProperties> extension_properties(extension_count);
  check_vk_result(vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &extension_count, extension_properties.data()));

  // Merge extensions lists, including optional and required for simplicity.
  std::set<const char*> ext_set;
  ext_set.insert(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  ext_set.insert(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
  ext_set.insert(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
  ext_set.insert(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
  ext_set.insert(iree_required_extensions.begin(),
                 iree_required_extensions.end());
  for (int i = 0; i < iree_optional_extensions.size(); ++i) {
    const char* optional_extension = iree_optional_extensions[i];
    for (int j = 0; j < extension_count; ++j) {
      if (strcmp(optional_extension, extension_properties[j].extensionName) ==
          0) {
        ext_set.insert(optional_extension);
        break;
      }
    }
  }
  std::vector<const char*> extensions(ext_set.begin(), ext_set.end());
  return extensions;
}

std::vector<const char*> GetInstanceLayers(
    iree_hal_vulkan_features_t vulkan_features) {
  // Query the layers that IREE wants / needs.
  std::vector<const char*> required_layers = GetIreeLayers(
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_REQUIRED, vulkan_features);
  std::vector<const char*> optional_layers = GetIreeLayers(
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_OPTIONAL, vulkan_features);

  // Query the layers that are available on the Vulkan ICD.
  uint32_t layer_property_count = 0;
  check_vk_result(
      vkEnumerateInstanceLayerProperties(&layer_property_count, NULL));
  std::vector<VkLayerProperties> layer_properties(layer_property_count);
  check_vk_result(vkEnumerateInstanceLayerProperties(&layer_property_count,
                                                     layer_properties.data()));

  // Match between optional/required and available layers.
  std::vector<const char*> layers;
  for (const char* layer_name : required_layers) {
    bool found = false;
    for (const auto& layer_property : layer_properties) {
      if (std::strcmp(layer_name, layer_property.layerName) == 0) {
        found = true;
        layers.push_back(layer_name);
        break;
      }
    }
    if (!found) {
      fprintf(stderr, "Required layer %s not available\n", layer_name);
      abort();
    }
  }
  for (const char* layer_name : optional_layers) {
    for (const auto& layer_property : layer_properties) {
      if (std::strcmp(layer_name, layer_property.layerName) == 0) {
        layers.push_back(layer_name);
        break;
      }
    }
  }

  return layers;
}

std::vector<const char*> GetInstanceExtensions(
    SDL_Window* window, iree_hal_vulkan_features_t vulkan_features) {
  // Ask SDL for its list of required instance extensions.
  uint32_t sdl_extensions_count = 0;
  SDL_Vulkan_GetInstanceExtensions(window, &sdl_extensions_count, NULL);
  std::vector<const char*> sdl_extensions(sdl_extensions_count);
  SDL_Vulkan_GetInstanceExtensions(window, &sdl_extensions_count,
                                   sdl_extensions.data());

  std::vector<const char*> iree_required_extensions = GetIreeExtensions(
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_REQUIRED,
      vulkan_features);
  std::vector<const char*> iree_optional_extensions = GetIreeExtensions(
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_OPTIONAL,
      vulkan_features);

  // Merge extensions lists, including optional and required for simplicity.
  std::set<const char*> ext_set;
  ext_set.insert(sdl_extensions.begin(), sdl_extensions.end());
  ext_set.insert(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  ext_set.insert(iree_required_extensions.begin(),
                 iree_required_extensions.end());
  ext_set.insert(iree_optional_extensions.begin(),
                 iree_optional_extensions.end());
  std::vector<const char*> extensions(ext_set.begin(), ext_set.end());
  return extensions;
}

void SetupVulkan(iree_hal_vulkan_features_t vulkan_features,
                 const char** instance_layers, uint32_t instance_layers_count,
                 const char** instance_extensions,
                 uint32_t instance_extensions_count,
                 const VkAllocationCallbacks* allocator, VkInstance* instance,
                 uint32_t* queue_family_index,
                 VkPhysicalDevice* physical_device, VkQueue* queue,
                 VkDevice* device, VkDescriptorPool* descriptor_pool) {
  VkResult err;

  // Create Vulkan Instance
  {
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.enabledLayerCount = instance_layers_count;
    create_info.ppEnabledLayerNames = instance_layers;
    create_info.enabledExtensionCount = instance_extensions_count;
    create_info.ppEnabledExtensionNames = instance_extensions;
    err = vkCreateInstance(&create_info, allocator, instance);
    check_vk_result(err);
  }

  // Select GPU
  {
    uint32_t gpu_count;
    err = vkEnumeratePhysicalDevices(*instance, &gpu_count, NULL);
    check_vk_result(err);
    IM_ASSERT(gpu_count > 0);

    VkPhysicalDevice* gpus =
        (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * gpu_count);
    err = vkEnumeratePhysicalDevices(*instance, &gpu_count, gpus);
    check_vk_result(err);

    // Use the first reported GPU for simplicity.
    *physical_device = gpus[0];

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(*physical_device, &properties);
    fprintf(stdout, "Selected Vulkan device: '%s'\n", properties.deviceName);
    free(gpus);
  }

  // Select queue family. We want a single queue with graphics and compute for
  // simplicity, but we could also discover and use separate queues for each.
  {
    uint32_t count;
    vkGetPhysicalDeviceQueueFamilyProperties(*physical_device, &count, NULL);
    VkQueueFamilyProperties* queues = (VkQueueFamilyProperties*)malloc(
        sizeof(VkQueueFamilyProperties) * count);
    vkGetPhysicalDeviceQueueFamilyProperties(*physical_device, &count, queues);
    for (uint32_t i = 0; i < count; i++) {
      if (queues[i].queueFlags &
          (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
        *queue_family_index = i;
        break;
      }
    }
    free(queues);
    IM_ASSERT(*queue_family_index != (uint32_t)-1);
  }

  // Create Logical Device (with 1 queue)
  {
    std::vector<const char*> device_extensions =
        GetDeviceExtensions(*physical_device, vulkan_features);
    const float queue_priority[] = {1.0f};
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = *queue_family_index;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = queue_priority;
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_info;
    create_info.enabledExtensionCount =
        static_cast<uint32_t>(device_extensions.size());
    create_info.ppEnabledExtensionNames = device_extensions.data();

    // Enable timeline semaphores.
    VkPhysicalDeviceFeatures2 features2;
    memset(&features2, 0, sizeof(features2));
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    create_info.pNext = &features2;
    VkPhysicalDeviceTimelineSemaphoreFeatures semaphore_features;
    memset(&semaphore_features, 0, sizeof(semaphore_features));
    semaphore_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    semaphore_features.pNext = features2.pNext;
    features2.pNext = &semaphore_features;
    semaphore_features.timelineSemaphore = VK_TRUE;

    err = vkCreateDevice(*physical_device, &create_info, allocator, device);
    check_vk_result(err);
    vkGetDeviceQueue(*device, *queue_family_index, 0, queue);
  }

  p_vkGetMemoryWin32HandlePropertiesKHR =
      reinterpret_cast<PFN_vkGetMemoryWin32HandlePropertiesKHR>(
          vkGetDeviceProcAddr(g_Device, "vkGetMemoryWin32HandlePropertiesKHR"));
  assert(p_vkGetMemoryWin32HandlePropertiesKHR);
  p_vkGetImageMemoryRequirements2KHR =
      reinterpret_cast<PFN_vkGetImageMemoryRequirements2KHR>(
          vkGetDeviceProcAddr(g_Device, "vkGetImageMemoryRequirements2KHR"));
  assert(p_vkGetImageMemoryRequirements2KHR);

  // Create Descriptor Pool
  {
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000 * IREE_ARRAYSIZE(pool_sizes);
    pool_info.poolSizeCount = (uint32_t)IREE_ARRAYSIZE(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;
    err =
        vkCreateDescriptorPool(*device, &pool_info, allocator, descriptor_pool);
    check_vk_result(err);
  }
}

void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd,
                       const VkAllocationCallbacks* allocator,
                       VkInstance instance, uint32_t queue_family_index,
                       VkPhysicalDevice physical_device, VkDevice device,
                       VkSurfaceKHR surface, int width, int height,
                       uint32_t min_image_count) {
  wd->Surface = surface;

  // Check for WSI support
  VkBool32 res;
  vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, queue_family_index,
                                       wd->Surface, &res);
  if (res != VK_TRUE) {
    fprintf(stderr, "Error no WSI support on physical device 0\n");
    exit(-1);
  }

  // Select Surface Format
  const VkFormat requestSurfaceImageFormat[] = {
      VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
      VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
  const VkColorSpaceKHR requestSurfaceColorSpace =
      VK_COLORSPACE_SRGB_NONLINEAR_KHR;
  wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
      physical_device, wd->Surface, requestSurfaceImageFormat,
      (size_t)IREE_ARRAYSIZE(requestSurfaceImageFormat),
      requestSurfaceColorSpace);

  // Select Present Mode
#ifdef IMGUI_UNLIMITED_FRAME_RATE
  VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_MAILBOX_KHR,
                                      VK_PRESENT_MODE_IMMEDIATE_KHR,
                                      VK_PRESENT_MODE_FIFO_KHR};
#else
  VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_FIFO_KHR};
#endif
  wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
      physical_device, wd->Surface, &present_modes[0],
      IREE_ARRAYSIZE(present_modes));

  // Create SwapChain, RenderPass, Framebuffer, etc.
  IM_ASSERT(min_image_count >= 2);
  ImGui_ImplVulkanH_CreateOrResizeWindow(instance, physical_device, device, wd,
                                         queue_family_index, allocator, width,
                                         height, min_image_count);

  // Set clear color.
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  memcpy(&wd->ClearValue.color.float32[0], &clear_color, 4 * sizeof(float));
}

void RenderFrame(ImGui_ImplVulkanH_Window* wd, VkDevice device, VkQueue queue) {
  VkResult err;

  VkSemaphore image_acquired_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
  VkSemaphore render_complete_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
  err = vkAcquireNextImageKHR(device, wd->Swapchain, UINT64_MAX,
                              image_acquired_semaphore, VK_NULL_HANDLE,
                              &wd->FrameIndex);
  check_vk_result(err);

  ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
  {
    err = vkWaitForFences(
        device, 1, &fd->Fence, VK_TRUE,
        UINT64_MAX);  // wait indefinitely instead of periodically checking
    check_vk_result(err);

    err = vkResetFences(device, 1, &fd->Fence);
    check_vk_result(err);
  }
  {
    err = vkResetCommandPool(device, fd->CommandPool, 0);
    check_vk_result(err);
    VkCommandBufferBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
    check_vk_result(err);
  }
  {
    VkRenderPassBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass = wd->RenderPass;
    info.framebuffer = fd->Framebuffer;
    info.renderArea.extent.width = wd->Width;
    info.renderArea.extent.height = wd->Height;
    info.clearValueCount = 1;
    info.pClearValues = &wd->ClearValue;
    vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
  }

  // Record Imgui Draw Data and draw funcs into command buffer
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), fd->CommandBuffer);

  // Submit command buffer
  vkCmdEndRenderPass(fd->CommandBuffer);
  {
    VkPipelineStageFlags wait_stage =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &image_acquired_semaphore;
    info.pWaitDstStageMask = &wait_stage;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &fd->CommandBuffer;
    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &render_complete_semaphore;

    err = vkEndCommandBuffer(fd->CommandBuffer);
    check_vk_result(err);
    err = vkQueueSubmit(queue, 1, &info, fd->Fence);
    check_vk_result(err);
  }
}

void PresentFrame(ImGui_ImplVulkanH_Window* wd, VkQueue queue) {
  VkSemaphore render_complete_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
  VkPresentInfoKHR info = {};
  info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  info.waitSemaphoreCount = 1;
  info.pWaitSemaphores = &render_complete_semaphore;
  info.swapchainCount = 1;
  info.pSwapchains = &wd->Swapchain;
  info.pImageIndices = &wd->FrameIndex;
  VkResult err = vkQueuePresentKHR(queue, &info);
  check_vk_result(err);
  wd->SemaphoreIndex =
      (wd->SemaphoreIndex + 1) %
      wd->ImageCount;  // Now we can use the next set of semaphores
}

static void CleanupVulkan() {
  vkDestroyDescriptorPool(g_Device, g_DescriptorPool, g_Allocator);

  vkDestroyDevice(g_Device, g_Allocator);
  vkDestroyInstance(g_Instance, g_Allocator);
}

static void CleanupVulkanWindow() {
  ImGui_ImplVulkanH_DestroyWindow(g_Instance, g_Device, &g_MainWindowData,
                                  g_Allocator);
}

namespace iree {

extern "C" int iree_main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  if (argc != 2) {
    fprintf(stderr, "Usage: ml-overlay filters.vmfb\n");
    return 1;
  }
  module_path = argv[1];

  // --------------------------------------------------------------------------
  // Create a window.
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
    fprintf(stderr, "Failed to initialize SDL\n");
    abort();
    return 1;
  }

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  ImGui::StyleColorsDark();

  // Setup window
  SDL_WindowFlags window_flags =
      (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                        SDL_WINDOW_ALLOW_HIGHDPI);
  SDL_Window* window = SDL_CreateWindow(
      "ML Overlay", FLAG_x != -1 ? FLAG_x : SDL_WINDOWPOS_CENTERED,
      FLAG_y != -1 ? FLAG_y : SDL_WINDOWPOS_CENTERED, FLAG_width, FLAG_height,
      window_flags);

  // Setup Vulkan
  iree_hal_vulkan_features_t iree_vulkan_features =
      static_cast<iree_hal_vulkan_features_t>(
          IREE_HAL_VULKAN_FEATURE_ENABLE_VALIDATION_LAYERS |
          IREE_HAL_VULKAN_FEATURE_ENABLE_DEBUG_UTILS);
  std::vector<const char*> layers = GetInstanceLayers(iree_vulkan_features);
  std::vector<const char*> extensions =
      GetInstanceExtensions(window, iree_vulkan_features);
  SetupVulkan(iree_vulkan_features, layers.data(),
              static_cast<uint32_t>(layers.size()), extensions.data(),
              static_cast<uint32_t>(extensions.size()), g_Allocator,
              &g_Instance, &g_QueueFamily, &g_PhysicalDevice, &g_Queue,
              &g_Device, &g_DescriptorPool);

  // Create Window Surface
  VkSurfaceKHR surface;
  VkResult err;
  if (SDL_Vulkan_CreateSurface(window, g_Instance, &surface) == 0) {
    printf("Failed to create Vulkan surface.\n");
    return 1;
  }

  // Create Framebuffers
  int w, h;
  SDL_GetWindowSize(window, &w, &h);
  ImGui_ImplVulkanH_Window* wd = &g_MainWindowData;
  SetupVulkanWindow(wd, g_Allocator, g_Instance, g_QueueFamily,
                    g_PhysicalDevice, g_Device, surface, w, h, g_MinImageCount);

  SDL_SysWMinfo wmInfo;
  SDL_VERSION(&wmInfo.version);
  SDL_GetWindowWMInfo(window, &wmInfo);
  HWND hwnd = wmInfo.info.win.window;
  SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE);
  if (FLAG_always_on_top) {
    SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
  }

  // LONG cur_style = GetWindowLong(hwnd, GWL_EXSTYLE);
  // SetWindowLong(hwnd, GWL_EXSTYLE,
  //               cur_style | WS_EX_TRANSPARENT | WS_EX_LAYERED);

  capture_provider = CaptureProvider::FromWindow(hwnd);

  // Setup Platform/Renderer bindings
  ImGui_ImplSDL2_InitForVulkan(window);
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = g_Instance;
  init_info.PhysicalDevice = g_PhysicalDevice;
  init_info.Device = g_Device;
  init_info.QueueFamily = g_QueueFamily;
  init_info.Queue = g_Queue;
  init_info.PipelineCache = g_PipelineCache;
  init_info.DescriptorPool = g_DescriptorPool;
  init_info.Allocator = g_Allocator;
  init_info.MinImageCount = g_MinImageCount;
  init_info.ImageCount = wd->ImageCount;
  init_info.CheckVkResultFn = check_vk_result;
  ImGui_ImplVulkan_Init(&init_info, wd->RenderPass);

  // Upload Fonts
  {
    // Use any command queue
    VkCommandPool command_pool = wd->Frames[wd->FrameIndex].CommandPool;
    VkCommandBuffer command_buffer = wd->Frames[wd->FrameIndex].CommandBuffer;

    err = vkResetCommandPool(g_Device, command_pool, 0);
    check_vk_result(err);
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    err = vkBeginCommandBuffer(command_buffer, &begin_info);
    check_vk_result(err);

    ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

    VkSubmitInfo end_info = {};
    end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    end_info.commandBufferCount = 1;
    end_info.pCommandBuffers = &command_buffer;
    err = vkEndCommandBuffer(command_buffer);
    check_vk_result(err);
    err = vkQueueSubmit(g_Queue, 1, &end_info, VK_NULL_HANDLE);
    check_vk_result(err);

    err = vkDeviceWaitIdle(g_Device);
    check_vk_result(err);
    ImGui_ImplVulkan_DestroyFontUploadObjects();
  }

  // Demo state.
  bool show_demo_window = FLAG_imgui_demo_window;
  bool show_iree_window = true;
  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------
  // Setup IREE.

  // Create a runtime Instance.
  vm::ref<iree_vm_instance_t> iree_instance;
  IREE_CHECK_OK(iree_vm_instance_create(
      IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &iree_instance));

  // Register HAL drivers and VM module types.
  IREE_CHECK_OK(iree_hal_vulkan_driver_module_register(
      iree_hal_driver_registry_default()));
  IREE_CHECK_OK(iree_hal_module_register_all_types(iree_instance.get()));

  // Create IREE Vulkan Driver and Device, sharing our VkInstance/VkDevice.
  fprintf(stdout, "Creating Vulkan driver/device\n");
  // Load symbols from our static `vkGetInstanceProcAddr` for IREE to use.
  iree_hal_vulkan_syms_t* iree_vk_syms = nullptr;
  IREE_CHECK_OK(iree_hal_vulkan_syms_create(
      reinterpret_cast<void*>(&vkGetInstanceProcAddr), iree_allocator_system(),
      &iree_vk_syms));
  // Create the driver sharing our VkInstance.
  iree_hal_driver_t* iree_vk_driver = nullptr;
  iree_string_view_t driver_identifier = iree_make_cstring_view("vulkan");
  iree_hal_vulkan_driver_options_t driver_options;
  iree_hal_vulkan_driver_options_initialize(&driver_options);
  driver_options.api_version = VK_API_VERSION_1_1;
  driver_options.requested_features = static_cast<iree_hal_vulkan_features_t>(
      IREE_HAL_VULKAN_FEATURE_ENABLE_DEBUG_UTILS);
  driver_options.debug_verbosity = 4;
  IREE_CHECK_OK(iree_hal_vulkan_driver_create_using_instance(
      driver_identifier, &driver_options, iree_vk_syms, g_Instance,
      iree_allocator_system(), &iree_vk_driver));
  // Create a device sharing our VkDevice and queue.
  // We could also create a separate (possibly low priority) compute queue for
  // IREE, and/or provide a dedicated transfer queue.
  iree_string_view_t device_identifier = iree_make_cstring_view("vulkan");
  iree_hal_vulkan_queue_set_t compute_queue_set;
  compute_queue_set.queue_family_index = g_QueueFamily;
  compute_queue_set.queue_indices = 1 << 0;
  iree_hal_vulkan_queue_set_t transfer_queue_set;
  transfer_queue_set.queue_indices = 0;
  iree_hal_device_t* iree_vk_device = nullptr;
  IREE_CHECK_OK(iree_hal_vulkan_wrap_device(
      device_identifier, &driver_options.device_options, iree_vk_syms,
      g_Instance, g_PhysicalDevice, g_Device, &compute_queue_set,
      &transfer_queue_set, iree_allocator_system(), &iree_vk_device));
  // Create a HAL module using the HAL device.
  vm::ref<iree_vm_module_t> hal_module;
  IREE_CHECK_OK(iree_hal_module_create(iree_instance.get(), iree_vk_device,
                                       IREE_HAL_MODULE_FLAG_NONE,
                                       iree_allocator_system(), &hal_module));

  vm::ref<iree_vm_context_t> iree_context;
  vm::ref<iree_vm_module_t> main_module;
  std::vector<iree_vm_function_t> filter_functions;
  int current_filter_ordinal = -1;
  std::string default_filter(FLAG_filter);

  auto reload_module = [&]() {
    // Try to save the current filter function so we can choose it again if it
    // exists in the loaded module. The user may remove it in which case we'll
    // fall back to the first filter.
    if (!filter_functions.empty()) {
      auto current_name =
          iree_vm_function_name(&filter_functions[current_filter_ordinal]);
      default_filter = std::string(current_name.data, current_name.size);
    }

    current_filter_ordinal = -1;
    iree_context.reset();
    filter_functions.clear();
    main_module.reset();

    fprintf(stdout, "Loading module from %s...\n", module_path);
    IREE_CHECK_OK(load_module(iree_instance.get(), argv[1],
                              iree_allocator_system(), &main_module));

    // Query for details about what is in the loaded module.
    iree_vm_module_signature_t main_module_signature =
        iree_vm_module_signature(main_module.get());
    fprintf(stdout, "Module loaded, have <%" PRIhsz "> exported functions:\n",
            main_module_signature.export_function_count);

    // Allocate a context that will hold the module state across invocations.
    std::vector<iree_vm_module_t*> modules = {
        hal_module.get(),
        main_module.get(),
    };
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        iree_instance.get(), IREE_VM_CONTEXT_FLAG_NONE, modules.size(),
        modules.data(), iree_allocator_system(), &iree_context));
    fprintf(stdout, "Context with modules is ready for use\n");

    // Lookup the entry point functions.
    auto module_signature = iree_vm_module_signature(main_module.get());
    filter_functions.reserve(module_signature.export_function_count);
    for (iree_host_size_t i = 0; i < module_signature.export_function_count;
         ++i) {
      iree_vm_function_t export_function;
      IREE_CHECK_OK(iree_vm_module_lookup_function_by_ordinal(
          main_module.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT, i,
          &export_function));

      // Only support async functions.
      iree_string_view_t invocation_model =
          iree_vm_function_lookup_attr_by_name(&export_function,
                                               IREE_SV("iree.abi.model"));
      if (!iree_string_view_equal(invocation_model, IREE_SV("coarse-fences"))) {
        continue;
      }

      auto function_name = iree_vm_function_name(&export_function);
      auto function_signature = iree_vm_function_signature(&export_function);
      fprintf(stdout, "  %" PRIhsz ": '%.*s' with calling convention '%.*s'\n",
              i, (int)function_name.size, function_name.data,
              (int)function_signature.calling_convention.size,
              function_signature.calling_convention.data);

      // Pick the default the user specified, if any.
      if (iree_string_view_equal(
              function_name, iree_make_cstring_view(default_filter.c_str()))) {
        current_filter_ordinal = filter_functions.size();
      }

      filter_functions.push_back(export_function);
    }

    // If we couldn't recover the previously selected filter choose the first.
    if (current_filter_ordinal == -1 && !filter_functions.empty()) {
      current_filter_ordinal = 0;
    }
  };
  reload_module();

  // --------------------------------------------------------------------------

  VkCommandPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.pNext = nullptr;
  pool_info.queueFamilyIndex = g_QueueFamily;
  pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  VkCommandPool sync_command_pool = nullptr;
  check_vk_result(vkCreateCommandPool(g_Device, &pool_info, g_Allocator,
                                      &sync_command_pool));
  VkCommandBufferAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.pNext = nullptr;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  alloc_info.commandPool = sync_command_pool;
  VkCommandBuffer sync_command_buffer = nullptr;
  check_vk_result(
      vkAllocateCommandBuffers(g_Device, &alloc_info, &sync_command_buffer));

  VkFenceCreateInfo fence_info = {};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.pNext = nullptr;
  fence_info.flags = 0;
  VkFence sync_fence = nullptr;
  check_vk_result(
      vkCreateFence(g_Device, &fence_info, g_Allocator, &sync_fence));

  auto sync_commands = [&](std::function<void(VkCommandBuffer)> record) {
    {
      VkCommandBufferBeginInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      check_vk_result(vkBeginCommandBuffer(sync_command_buffer, &info));
    }

    record(sync_command_buffer);

    {
      check_vk_result(vkEndCommandBuffer(sync_command_buffer));

      VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
      VkSubmitInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      info.waitSemaphoreCount = 0;
      info.pWaitSemaphores = NULL;
      info.pWaitDstStageMask = NULL;  //&wait_stage;
      info.commandBufferCount = 1;
      info.pCommandBuffers = &sync_command_buffer;
      info.signalSemaphoreCount = 0;
      info.pSignalSemaphores = NULL;
      check_vk_result(vkQueueSubmit(g_Queue, 1, &info, sync_fence));
    }
    check_vk_result(
        vkWaitForFences(g_Device, 1, &sync_fence, VK_TRUE, UINT64_MAX));
    check_vk_result(vkResetFences(g_Device, 1, &sync_fence));

    check_vk_result(vkResetCommandPool(g_Device, sync_command_pool, 0));
  };

  VkSampler sampler = nullptr;
  {
    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU =
        VK_SAMPLER_ADDRESS_MODE_REPEAT;  // outside image bounds just use
                                         // border color
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.minLod = -1000;
    sampler_info.maxLod = 1000;
    sampler_info.maxAnisotropy = 1.0f;
    check_vk_result(
        vkCreateSampler(g_Device, &sampler_info, g_Allocator, &sampler));
  }

  auto captureHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
  VkImage capture_image = nullptr;
  VkDeviceMemory capture_memory = nullptr;
  VkImageView capture_image_view = nullptr;
  VkDescriptorSet capture_ds = nullptr;
  int capture_width = 0;
  int capture_height = 0;

  uint64_t frame_number = 1;
  vm::ref<iree_hal_semaphore_t> source_semaphore;
  IREE_CHECK_OK(
      iree_hal_semaphore_create(iree_vk_device, 0ll, &source_semaphore));
  VkSemaphore source_semaphore_handle =
      iree_hal_vulkan_native_semaphore_handle(source_semaphore.get());
  vm::ref<iree_hal_semaphore_t> target_semaphore;
  IREE_CHECK_OK(
      iree_hal_semaphore_create(iree_vk_device, 0ll, &target_semaphore));
  VkSemaphore target_semaphore_handle =
      iree_hal_vulkan_native_semaphore_handle(target_semaphore.get());

  iree_hal_buffer_params_t source_buffer_params = {};
  source_buffer_params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  source_buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  source_buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  vm::ref<iree_hal_buffer_t> source_buffer;
  vm::ref<iree_hal_buffer_view_t> source_buffer_view;
  vm::ref<iree_hal_buffer_t> target_buffer_storage;  // in-place arg
  iree_hal_dim_t source_dims[3] = {0, 0, 4};
  int source_width = 0;
  int source_height = 0;
  auto refresh_source_buffer = [&](int width, int height) {
    if (width == source_width && height == source_height) return;
    source_buffer_view.reset();
    source_buffer.reset();
    target_buffer_storage.reset();

    iree_device_size_t source_buffer_size = width * height * sizeof(uint32_t);
    IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(iree_vk_device), source_buffer_params,
        source_buffer_size, &source_buffer));

    source_width = width;
    source_height = height;
    source_dims[0] = height;
    source_dims[1] = width;
    IREE_CHECK_OK(iree_hal_buffer_view_create(
        source_buffer.get(), IREE_ARRAYSIZE(source_dims), source_dims,
        IREE_HAL_ELEMENT_TYPE_UINT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        iree_allocator_system(), &source_buffer_view));

    // Only used for in-place outputs - wasteful otherwise but :shrug:.
    IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(iree_vk_device), source_buffer_params,
        source_buffer_size, &target_buffer_storage));
  };

  VkDeviceMemory target_memory = nullptr;
  VkImage target_image = nullptr;
  VkImageView target_image_view = nullptr;
  VkDescriptorSet target_ds = nullptr;
  int target_width = 0;
  int target_height = 0;
  auto refresh_target_image = [&](int width, int height) {
    if (width == target_width && height == target_height) return;

    if (target_ds) {
      vkFreeDescriptorSets(g_Device, g_DescriptorPool, 1, &target_ds);
      target_ds = nullptr;
    }
    if (target_image_view) {
      vkDestroyImageView(g_Device, target_image_view, nullptr);
      target_image_view = nullptr;
    }
    if (target_image) {
      vkDestroyImage(g_Device, target_image, nullptr);
      target_image = nullptr;
    }
    if (target_memory) {
      vkFreeMemory(g_Device, target_memory, g_Allocator);
      target_memory = nullptr;
    }

    target_width = width;
    target_height = height;

    VkImageCreateInfo targetImageInfo = {
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        nullptr,
    };
    targetImageInfo.imageType = VK_IMAGE_TYPE_2D;
    targetImageInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
    targetImageInfo.extent = {
        (uint32_t)width,
        (uint32_t)height,
        1,
    };
    targetImageInfo.mipLevels = 1;
    targetImageInfo.arrayLayers = 1;
    targetImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    targetImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    targetImageInfo.usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    targetImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    targetImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    check_vk_result(
        vkCreateImage(g_Device, &targetImageInfo, nullptr, &target_image));

    const VkImageMemoryRequirementsInfo2 imri_info2{
        /*.sType =*/VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
        /*.pNext =*/nullptr,
        /*.image =*/target_image,
    };
    VkMemoryRequirements2 memory_requirements2{
        /*.sType =*/VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        /*.pNext =*/nullptr,
        /*.memoryRequirements =*/{},
    };
    p_vkGetImageMemoryRequirements2KHR(g_Device, &imri_info2,
                                       &memory_requirements2);
    VkMemoryAllocateInfo allocInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
    };
    allocInfo.memoryTypeIndex =
        findMemoryType(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    allocInfo.allocationSize = memory_requirements2.memoryRequirements.size;
    check_vk_result(
        vkAllocateMemory(g_Device, &allocInfo, g_Allocator, &target_memory));
    check_vk_result(
        vkBindImageMemory(g_Device, target_image, target_memory, 0));

    // Create the Image View
    {
      VkImageViewCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      info.image = target_image;
      info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      info.format = targetImageInfo.format;
      info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      info.subresourceRange.levelCount = 1;
      info.subresourceRange.layerCount = 1;
      check_vk_result(
          vkCreateImageView(g_Device, &info, g_Allocator, &target_image_view));
    }

    target_ds = ImGui_ImplVulkan_AddTexture(
        sampler, target_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  };

  vm::ref<iree_vm_list_t> inputs;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 8,
                                    iree_allocator_system(), &inputs));
  vm::ref<iree_vm_list_t> outputs;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 8,
                                    iree_allocator_system(), &outputs));

  struct BBox {
    int32_t id;
    int32_t data[3];
    int32_t x0, y0, x1, y1;
    int32_t color_a, color_b, color_g, color_r;
    int32_t reserved[4];
  };
  std::vector<BBox> bboxes;

  // --------------------------------------------------------------------------
  // Main loop.
  bool done = false;
  while (!done) {
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        done = true;
      }

      ImGui_ImplSDL2_ProcessEvent(&event);
      if (event.type == SDL_QUIT) done = true;
      if (event.type == SDL_WINDOWEVENT &&
          event.window.event == SDL_WINDOWEVENT_RESIZED &&
          event.window.windowID == SDL_GetWindowID(window)) {
        g_SwapChainResizeWidth = (int)event.window.data1;
        g_SwapChainResizeHeight = (int)event.window.data2;
        g_SwapChainRebuild = true;
      }
    }

    if (g_SwapChainRebuild) {
      g_SwapChainRebuild = false;
      ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
      ImGui_ImplVulkanH_CreateOrResizeWindow(
          g_Instance, g_PhysicalDevice, g_Device, &g_MainWindowData,
          g_QueueFamily, g_Allocator, g_SwapChainResizeWidth,
          g_SwapChainResizeHeight, g_MinImageCount);
      g_MainWindowData.FrameIndex = 0;
    }

    // Start the Dear ImGui frame
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL2_NewFrame(window);
    ImGui::NewFrame();

    static bool overlay_open = FLAG_overlay;
    static bool always_on_top = FLAG_always_on_top;
    static bool capturing = true;
    static bool show_source = false;
    static int old_filter_ordinal = -1;
    bool old_always_on_top = always_on_top;
    bool request_reload_module = false;
    bool next_filter = false;
    if (overlay_open) {
      if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Options")) {
          if (ImGui::MenuItem("Reload Module", "F5")) {
            request_reload_module = true;
          }
          if (ImGui::MenuItem("Next Filter", "Tab")) {
            next_filter = true;
          }
          ImGui::Separator();
          if (ImGui::MenuItem("Enable Live Update", "Space", &capturing)) {
          }
          if (ImGui::MenuItem("Show Capture Source", "Shift", &show_source)) {
          }
          if (ImGui::MenuItem("Show Overlay", "`", &overlay_open)) {
          }
          if (ImGui::MenuItem("Always on Top", "F1", &always_on_top)) {
          }
          ImGui::Separator();
          if (ImGui::MenuItem("Exit", "Escape")) {
            done = true;
          }
          ImGui::EndMenu();
        }
        if (!filter_functions.empty()) {
          ImGui::Combo(
              "##function", &current_filter_ordinal,
              +[](void* data, int idx, const char** out_text) {
                auto* functions = (std::vector<iree_vm_function_t>*)data;
                auto name = iree_vm_function_name(&functions->at(idx));
                // HACK: this is not safe - data may not have a \0
                // We should use BeginCombo/EndCombo to render the text.
                *out_text = name.data;
                return true;
              },
              &filter_functions, filter_functions.size(), 16);
        } else {
          ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f),
                             "NO FILTER FUNCTIONS FOUND");
        }
        ImGui::EndMainMenuBar();
      }
    }
    if (ImGui::IsKeyPressed(ImGuiKey_F5) ||
        (ImGui::IsKeyDown(ImGuiKey_LeftShift) &&
         ImGui::IsKeyPressed(ImGuiKey_R))) {
      request_reload_module = true;
    }
    if (current_filter_ordinal != -1) {
      if (next_filter || ImGui::IsKeyPressed(ImGuiKey_Tab)) {
        current_filter_ordinal =
            (current_filter_ordinal + 1) % filter_functions.size();
      }
      if (old_filter_ordinal != current_filter_ordinal) {
        std::string title = "ML Overlay: ";
        auto name =
            iree_vm_function_name(&filter_functions[current_filter_ordinal]);
        title.append(name.data, name.size);
        SDL_SetWindowTitle(window, title.c_str());
        bboxes.clear();
        old_filter_ordinal = current_filter_ordinal;
      }
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
      capturing = !capturing;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_LeftShift) ||
        ImGui::IsKeyPressed(ImGuiKey_RightShift)) {
      show_source = !show_source;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_GraveAccent)) {
      overlay_open = !overlay_open;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_F1)) {
      always_on_top = !always_on_top;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
      done = true;
    }
    if (always_on_top != old_always_on_top) {
      SetWindowPos(hwnd, always_on_top ? HWND_TOPMOST : HWND_NOTOPMOST, 0, 0, 0,
                   0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    }

    if (request_reload_module) {
      ImGui::EndFrame();
      reload_module();
      old_filter_ordinal = -1;
      bboxes.clear();
      ImGui::SetWindowFocus("Statistics");
      continue;
    }

    if (current_filter_ordinal == -1) {
      overlay_open = true;
      ImGui::Render();
      RenderFrame(wd, g_Device, g_Queue);
      PresentFrame(wd, g_Queue);
      continue;
    }

    int abs_x, abs_y;
    SDL_GetWindowPosition(window, &abs_x, &abs_y);
    int client_w, client_h;
    SDL_GetWindowSize(window, &client_w, &client_h);
    int border_t, border_l, border_b, border_r;
    SDL_GetWindowBordersSize(window, &border_t, &border_l, &border_b,
                             &border_r);

    static bool use_work_area = true;
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    float menu_height = viewport->WorkPos.y;
    int abs_l = abs_x;
    int abs_t = abs_y + menu_height;
    int abs_r = abs_l + client_w;
    int abs_b = abs_t + client_h - menu_height;
    ImVec2 tl =
        ImVec2((float)abs_l / capture_width, (float)abs_t / capture_height);
    ImVec2 br =
        ImVec2((float)abs_r / capture_width, (float)abs_b / capture_height);

    if (capturing && capture_provider->AcquireFrame()) {
      auto* d3d11_texture = capture_provider->current_frame();
      auto shared_handle = capture_provider->current_frame_handle();

      if (capture_ds) {
        vkFreeDescriptorSets(g_Device, g_DescriptorPool, 1, &capture_ds);
        capture_ds = nullptr;
      }
      if (capture_image_view) {
        vkDestroyImageView(g_Device, capture_image_view, nullptr);
        capture_image_view = nullptr;
      }
      if (capture_image) {
        vkDestroyImage(g_Device, capture_image, nullptr);
        capture_image = nullptr;
      }
      if (capture_memory) {
        vkFreeMemory(g_Device, capture_memory, nullptr);
        capture_memory = nullptr;
      }

      D3D11_TEXTURE2D_DESC desc;
      d3d11_texture->GetDesc(&desc);
      assert(desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM);
      capture_width = desc.Width;
      capture_height = desc.Height;

      VkExternalMemoryImageCreateInfo externalInfo = {
          VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
          nullptr,
          static_cast<VkExternalMemoryHandleTypeFlags>(captureHandleType),
      };
      VkImageCreateInfo imageInfo = {
          VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
          &externalInfo,
      };
      imageInfo.imageType = VK_IMAGE_TYPE_2D;
      imageInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
      imageInfo.extent = {desc.Width, desc.Height, 1};
      imageInfo.mipLevels = desc.MipLevels;
      imageInfo.arrayLayers = desc.ArraySize;
      imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
      imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
      imageInfo.usage =
          VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      check_vk_result(
          vkCreateImage(g_Device, &imageInfo, nullptr, &capture_image));

      VkMemoryDedicatedRequirements dedicated_reqs{
          /*.sType =*/VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
          /*.pNext =*/nullptr,
          /*.prefersDedicatedAllocation =*/VK_FALSE,
          /*.requiresDedicatedAllocation =*/VK_FALSE,
      };
      const VkImageMemoryRequirementsInfo2 imri_info2{
          /*.sType =*/VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
          /*.pNext =*/nullptr,
          /*.image =*/capture_image,
      };
      VkMemoryRequirements2 memory_requirements2{
          /*.sType =*/VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
          /*.pNext =*/&dedicated_reqs,
          /*.memoryRequirements =*/{},
      };
      p_vkGetImageMemoryRequirements2KHR(g_Device, &imri_info2,
                                         &memory_requirements2);
      const bool make_dedicated =
          dedicated_reqs.prefersDedicatedAllocation == VK_TRUE ||
          dedicated_reqs.requiresDedicatedAllocation == VK_TRUE;
      const VkMemoryDedicatedAllocateInfo dedicated_info{
          /*.sType =*/VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
          /*.pNext =*/nullptr,
          /*.image =*/capture_image,
          /*.buffer =*/nullptr,
      };

      // Vulkan memory import
      VkMemoryWin32HandlePropertiesKHR handleProperties = {
          VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR,
      };
      check_vk_result(p_vkGetMemoryWin32HandlePropertiesKHR(
          g_Device, captureHandleType, shared_handle, &handleProperties));

      VkImportMemoryWin32HandleInfoKHR importInfo = {
          VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
          make_dedicated ? &dedicated_info : nullptr,
          captureHandleType,
          shared_handle,
      };
      VkMemoryAllocateInfo allocInfo = {
          VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
          &importInfo,
      };
      allocInfo.memoryTypeIndex = findMemoryType(
          handleProperties.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      allocInfo.allocationSize = memory_requirements2.memoryRequirements.size;
      check_vk_result(
          vkAllocateMemory(g_Device, &allocInfo, nullptr, &capture_memory));
      check_vk_result(
          vkBindImageMemory(g_Device, capture_image, capture_memory, 0));

      // Create the Image View
      {
        VkImageViewCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        info.image = capture_image;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = VK_FORMAT_B8G8R8A8_UNORM;
        info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.layerCount = 1;
        check_vk_result(vkCreateImageView(g_Device, &info, g_Allocator,
                                          &capture_image_view));
      }
      capture_ds =
          ImGui_ImplVulkan_AddTexture(sampler, capture_image_view,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

      refresh_source_buffer(client_w, client_h);

#if 1
      sync_commands([&](VkCommandBuffer commandBuffer) {
        VkImageMemoryBarrier to_src_barrier = {};
        to_src_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_src_barrier.pNext = nullptr;
        to_src_barrier.srcAccessMask = VK_ACCESS_NONE;
        to_src_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        to_src_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_src_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_src_barrier.srcQueueFamilyIndex = g_QueueFamily;
        to_src_barrier.dstQueueFamilyIndex = g_QueueFamily;
        to_src_barrier.image = capture_image;
        to_src_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_src_barrier.subresourceRange.layerCount = 1;
        to_src_barrier.subresourceRange.levelCount = 1;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &to_src_barrier);

        VkBuffer source_buffer_handle =
            iree_hal_vulkan_buffer_handle(source_buffer.get());
        VkBufferImageCopy source_copy_region = {};
        source_copy_region.imageOffset.x = abs_x;
        source_copy_region.imageOffset.y = abs_y;
        source_copy_region.imageExtent.width = source_width;
        source_copy_region.imageExtent.height = source_height;
        source_copy_region.imageExtent.depth = 1;
        source_copy_region.imageSubresource.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;
        source_copy_region.imageSubresource.layerCount = 1;
        source_copy_region.bufferOffset = 0;
        source_copy_region.bufferRowLength =
            0;  // source_width * sizeof(uint32_t);
        source_copy_region.bufferImageHeight = 0;  // source_height;
        vkCmdCopyImageToBuffer(commandBuffer, capture_image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               source_buffer_handle, 1, &source_copy_region);

        VkImageMemoryBarrier source_barrier = {};
        source_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        source_barrier.pNext = nullptr;
        source_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        source_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        source_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        source_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        source_barrier.srcQueueFamilyIndex = g_QueueFamily;
        source_barrier.dstQueueFamilyIndex = g_QueueFamily;
        source_barrier.image = capture_image;
        source_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        source_barrier.subresourceRange.layerCount = 1;
        source_barrier.subresourceRange.levelCount = 1;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &source_barrier);
      });
#endif

      vm::ref<iree_hal_fence_t> source_ready_fence;
      vm::ref<iree_hal_fence_t> target_ready_fence;
      IREE_CHECK_OK(iree_hal_fence_create_at(
          source_semaphore.get(), frame_number, iree_allocator_system(),
          &source_ready_fence));
      IREE_CHECK_OK(iree_hal_fence_create_at(
          target_semaphore.get(), frame_number, iree_allocator_system(),
          &target_ready_fence));

      // DO NOT SUBMIT
      IREE_CHECK_OK(iree_hal_fence_signal(source_ready_fence.get()));

      iree_vm_function_t filter_function =
          filter_functions[current_filter_ordinal];

      iree_vm_list_clear(inputs.get());
      IREE_CHECK_OK(
          iree_vm_list_push_ref_retain(inputs.get(), source_buffer_view));
      if (iree_string_view_equal(
              iree_vm_function_signature(&filter_function).calling_convention,
              IREE_SV("0rrrr_r"))) {
        // HACK: we should be querying reflection info for this
        IREE_CHECK_OK(
            iree_vm_list_push_ref_retain(inputs.get(), target_buffer_storage));
      }
      IREE_CHECK_OK(
          iree_vm_list_push_ref_retain(inputs.get(), source_ready_fence));
      IREE_CHECK_OK(
          iree_vm_list_push_ref_retain(inputs.get(), target_ready_fence));

      iree_vm_list_clear(outputs.get());
      IREE_CHECK_OK(iree_vm_invoke(iree_context.get(), filter_function,
                                   IREE_VM_INVOCATION_FLAG_NONE,
                                   // IREE_VM_INVOCATION_FLAG_TRACE_EXECUTION,
                                   /*policy=*/nullptr, inputs.get(),
                                   outputs.get(), iree_allocator_system()));

      // DO NOT SUBMIT
      IREE_CHECK_OK(iree_hal_semaphore_wait(
          target_semaphore.get(), frame_number, iree_infinite_timeout()));
      ++frame_number;

      vm::ref<iree_hal_buffer_view_t> target_view =
          vm::retain_ref(iree_vm_list_get_buffer_view_assign(outputs.get(), 0));

      refresh_target_image(
          (int)iree_hal_buffer_view_shape_dim(target_view.get(), 1),
          (int)iree_hal_buffer_view_shape_dim(target_view.get(), 0));

#if 1
      sync_commands([&](VkCommandBuffer commandBuffer) {
        VkImageMemoryBarrier to_dst_barrier = {};
        to_dst_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_dst_barrier.pNext = nullptr;
        to_dst_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        to_dst_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_dst_barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_dst_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst_barrier.srcQueueFamilyIndex = g_QueueFamily;
        to_dst_barrier.dstQueueFamilyIndex = g_QueueFamily;
        to_dst_barrier.image = target_image;
        to_dst_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_dst_barrier.subresourceRange.layerCount = 1;
        to_dst_barrier.subresourceRange.levelCount = 1;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &to_dst_barrier);

        VkBuffer target_buffer_handle = iree_hal_vulkan_buffer_handle(
            iree_hal_buffer_view_buffer(target_view.get()));
        VkBufferImageCopy target_copy_region = {};
        target_copy_region.bufferOffset = 0;
        target_copy_region.bufferRowLength =
            0;  // source_width * sizeof(uint32_t);
        target_copy_region.bufferImageHeight = 0;  // source_height;
        target_copy_region.imageOffset.x = 0;
        target_copy_region.imageOffset.y = 0;
        target_copy_region.imageExtent.width = target_width;
        target_copy_region.imageExtent.height = target_height;
        target_copy_region.imageExtent.depth = 1;
        target_copy_region.imageSubresource.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;
        target_copy_region.imageSubresource.layerCount = 1;
        vkCmdCopyBufferToImage(
            commandBuffer, target_buffer_handle, target_image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &target_copy_region);

        VkImageMemoryBarrier target_barrier = {};
        target_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        target_barrier.pNext = nullptr;
        target_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        target_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        target_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        target_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        target_barrier.srcQueueFamilyIndex = g_QueueFamily;
        target_barrier.dstQueueFamilyIndex = g_QueueFamily;
        target_barrier.image = target_image;
        target_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        target_barrier.subresourceRange.layerCount = 1;
        target_barrier.subresourceRange.levelCount = 1;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &target_barrier);
      });
#endif

      if (iree_vm_list_size(outputs.get()) >= 2) {
        // We could do this readback async by doing a copy into a staging buffer
        // as part of the image transition command buffer.
        vm::ref<iree_hal_buffer_view_t> bbox_view = vm::retain_ref(
            iree_vm_list_get_buffer_view_assign(outputs.get(), 1));
        bboxes.resize(iree_hal_buffer_view_shape_dim(bbox_view.get(), 0));
        IREE_CHECK_OK(iree_hal_buffer_map_read(
            iree_hal_buffer_view_buffer(bbox_view.get()), 0, bboxes.data(),
            bboxes.size() * sizeof(BBox)));
      }

      // discard once consumed?
      // capture_provider->ReleaseFrame();
    }

    {
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
      ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
      ImGui::Begin(
          "##display", nullptr,
          ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
              ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs |
              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBackground);

      if (show_source && capture_ds) {
        ImGui::Image((ImTextureID)capture_ds, ImGui::GetWindowSize(), tl, br);
      } else if (!show_source && target_ds) {
        ImGui::Image((ImTextureID)target_ds, ImGui::GetWindowSize(),
                     ImVec2(0.0f, menu_height / (float)target_height),
                     ImVec2(1.0f, 1.0f));
      } else {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f),
                           "NO CAPTURE AVAILABLE");
      }

      if (!bboxes.empty()) {
        for (int i = 0; i < bboxes.size(); ++i) {
          auto& bbox = bboxes[i];

          ImDrawList* draw_list = ImGui::GetWindowDrawList();
          ImU32 color = ImGui::GetColorU32(
              IM_COL32(bbox.color_r, bbox.color_g, bbox.color_b, bbox.color_a));
          draw_list->AddRect(ImVec2(bbox.x0, bbox.y0), ImVec2(bbox.x1, bbox.y1),
                             color, 2.0f, ImDrawFlags_RoundCornersAll, 2.0f);

          ImGui::SetCursorScreenPos(ImVec2(bbox.x0 + 1.0f, bbox.y1 + 1.0f));
          ImGui::TextDisabled("%d", bbox.id);
          ImGui::SetCursorScreenPos(ImVec2(bbox.x0, bbox.y1));
          ImGui::Text("%d", bbox.id);
        }
      }

      ImGui::End();
      ImGui::PopStyleVar(2);
    }

    if (overlay_open) {
      const float PAD = 10.0f;
      const ImGuiViewport* viewport = ImGui::GetMainViewport();
      ImVec2 work_pos = viewport->WorkPos;  // Use work area to avoid
                                            // menu-bar/task-bar, if any!
      ImVec2 work_size = viewport->WorkSize;
      ImVec2 window_pos;
      window_pos.x = (work_pos.x + work_size.x - PAD);
      window_pos.y = (work_pos.y + work_size.y - PAD);
      ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, ImVec2(1.0f, 1.0f));

      ImGui::SetNextWindowBgAlpha(0.5f);
      if (ImGui::Begin("Statistics", nullptr,
                       ImGuiWindowFlags_NoDecoration |
                           ImGuiWindowFlags_AlwaysAutoResize |
                           ImGuiWindowFlags_NoNav)) {
        ImGui::Text("Capture: %d x %d", capture_width, capture_height);
        ImGui::Text("Clip: %d x %d", client_w, client_h);
        // TODO: timing of ML execution.
        ImGui::Text("Render: %.3f ms/frame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
      }
      ImGui::End();
    }

    // Demo window.
    if (show_demo_window) ImGui::ShowDemoWindow(&show_demo_window);

    // Rendering
    ImGui::Render();
    RenderFrame(wd, g_Device, g_Queue);

    PresentFrame(wd, g_Queue);
  }
  // --------------------------------------------------------------------------

  outputs.reset();
  inputs.reset();

  source_buffer_view.reset();
  source_buffer.reset();
  target_buffer_storage.reset();
  source_semaphore.reset();
  target_semaphore.reset();

  if (target_ds) {
    vkFreeDescriptorSets(g_Device, g_DescriptorPool, 1, &target_ds);
    target_ds = nullptr;
  }
  if (target_image_view) {
    vkDestroyImageView(g_Device, target_image_view, nullptr);
    target_image_view = nullptr;
  }
  if (target_image) {
    vkDestroyImage(g_Device, target_image, nullptr);
    target_image = nullptr;
  }
  if (target_memory) {
    vkFreeMemory(g_Device, target_memory, nullptr);
    target_memory = nullptr;
  }

  if (capture_ds) {
    vkFreeDescriptorSets(g_Device, g_DescriptorPool, 1, &capture_ds);
    capture_ds = nullptr;
  }
  if (capture_image_view) {
    vkDestroyImageView(g_Device, capture_image_view, nullptr);
    capture_image_view = nullptr;
  }
  if (capture_image) {
    vkDestroyImage(g_Device, capture_image, nullptr);
    capture_image = nullptr;
  }
  if (capture_memory) {
    vkFreeMemory(g_Device, capture_memory, nullptr);
    capture_memory = nullptr;
  }

  if (sampler) {
    vkDestroySampler(g_Device, sampler, nullptr);
    sampler = nullptr;
  }

  if (sync_command_buffer) {
    vkFreeCommandBuffers(g_Device, sync_command_pool, 1, &sync_command_buffer);
    sync_command_buffer = nullptr;
  }
  if (sync_command_pool) {
    vkDestroyCommandPool(g_Device, sync_command_pool, g_Allocator);
    sync_command_pool = nullptr;
  }
  if (sync_fence) {
    vkDestroyFence(g_Device, sync_fence, g_Allocator);
    sync_fence = nullptr;
  }

  // --------------------------------------------------------------------------
  // Cleanup
  hal_module.reset();
  main_module.reset();
  iree_context.reset();
  iree_hal_device_release(iree_vk_device);
  iree_hal_driver_release(iree_vk_driver);
  iree_hal_vulkan_syms_release(iree_vk_syms);
  iree_instance.reset();

  err = vkDeviceWaitIdle(g_Device);
  check_vk_result(err);
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  capture_provider.reset();

  CleanupVulkanWindow();
  CleanupVulkan();

  SDL_DestroyWindow(window);
  SDL_Quit();
  // --------------------------------------------------------------------------

  return 0;
}

}  // namespace iree

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX)

int main(int argc, char** argv) { return iree::iree_main(argc, argv); }

#elif defined(IREE_PLATFORM_WINDOWS)

#include <combaseapi.h>

// Entry point when using /SUBSYSTEM:CONSOLE is the standard main().
int main(int argc, char** argv) { return iree::iree_main(argc, argv); }

// Entry point when using /SUBSYSTEM:WINDOWS.
// https://docs.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-winmain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                   LPSTR lpCmdLine, int nShowCmd) {
  // Setup COM on the main thread.
  // NOTE: this may fail if COM has already been initialized - that's OK.
  CoInitializeEx(NULL, COINIT_MULTITHREADED);

  // Run standard main function.
  // We use the MSVCRT __argc/__argv to get access to the standard argc/argv
  // vs. using the flattened string passed to WinMain (that would require
  // complex unicode splitting/etc).
  // https://docs.microsoft.com/en-us/cpp/c-runtime-library/argc-argv-wargv
  return iree::iree_main(__argc, __argv);
}

#endif  // IREE_PLATFORM_*
