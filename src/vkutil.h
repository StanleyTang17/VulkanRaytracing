#ifndef VK_UTIL_H
#define VK_UTIL_H

#include <vulkan/vulkan.h>
#include <stdexcept>
#include <vector>
#include <optional>

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
void copyBuffer(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

namespace VkInit {
    void ShaderModule(VkDevice device, const std::vector<char>& code, VkShaderModule* pShaderModule);
    void CommandBuffers(VkDevice device, VkCommandPool pool, VkCommandBufferLevel level, uint32_t count, VkCommandBuffer* pCommandBuffers);
    void Buffer(
        VkPhysicalDevice physicalDevice,
        VkDevice device,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkBuffer* const pBuffer,
        VkDeviceMemory* const pBufferMemory,
        void* const initData = nullptr,
        size_t initDataSize = 0
    );
}

#endif