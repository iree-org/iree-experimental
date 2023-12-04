ml-overlay cpp/ml_overlay/filters.vmfb
(supports dragging vmfb on to exe)

ml-overlay cpp/ml_overlay/filters.vmfb \
  --filter=filter_add \
  --always_on_top=false \
  --overlay=false \
  --x=100 \
  --y=200 \
  --width=1024 \
  --height=1024 \
  --imgui_demo_window

iree-compile
  --iree-execution-model=async-external \
  --iree-hal-target-backends=vulkan-spirv \
  --iree-input-type=mhlo \
  cpp/ml_overlay/filters.mlir






Window

vkCmdCopyImageToBuffer region

struct Frame {
  VkImage source_image;
  VkImageLayout source_layout;
};

Pipeline
  ProcessFrame(Frame& frame);

source image:
  VkImage
  optional semaphore when available?

target image:
  VkImage

current results
  accumulated
  last produced image
  last produced outputs vm_list

source, target_storage -> target

command buffer with copy image to buffer
semaphore signal, import into fence
submit command buffer
VkBuffer import to iree_hal_buffer_t
wrap in buffer view with size
next fence
invoke with copy wait semaphore, next fence for ready
export fence to ready semaphore
{ or use imgui command buffer?
  command buffer with copy buffer to image
  submit command buffer waiting on fence
}
