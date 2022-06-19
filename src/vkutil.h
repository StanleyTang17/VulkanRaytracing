#ifndef VK_UTIL_H
#define VK_UTIL_H

#include <vulkan/vulkan.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include <stdexcept>
#include <vector>
#include <optional>

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
VkCommandBuffer startSingleTimeCommand(VkDevice device, VkCommandPool pool);
void endSingleTimeCommand(VkDevice device, VkCommandPool pool, VkQueue queue, VkCommandBuffer commandBuffer);
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
    void Image2D(
        VkPhysicalDevice physicalDevice,
        VkDevice device,
        uint32_t width,
        uint32_t height,
        VkFormat format,
        VkImageTiling tiling,
        VkImageUsageFlags usage,
        VkMemoryPropertyFlags memProperties,
        VkImage* pImage,
        VkDeviceMemory* pImageMemory
    );
    void ImageView2D(VkDevice device, VkImage image, VkFormat format, VkImageView* pImageView);
}

namespace VkCmd {
    void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
}

#endif