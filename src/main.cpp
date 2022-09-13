#define GLFW_INCLUDE_VULKAN

#include "util.h"
#include "vkutil.h"
#include "camera.h"
#include "font.h"

#include <glm/gtc/matrix_transform.hpp>
#include <OpenImageDenoise/oidn.hpp>

#include <iostream>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <unordered_set>
#include <chrono>
#include <string>



struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
    glm::vec2 position;
    glm::vec2 texCoord;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions() {
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions(3);

        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, position);

        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, texCoord);

        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

class RayTracingApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        initOIDN();

        std::cout << "app initialized!" << std::endl;

        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;
    const uint32_t WIDTH = 1600;
    const uint32_t HEIGHT = 912;

    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

    const int MAX_FRAMES_IN_FLIGHT = 2;
    uint32_t currentFrame = 0;
    uint32_t numSamples = 4;
    bool framebufferResized = false;
    QueueFamilyIndices queueFamilies;

    Camera camera;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    bool firstMouse = true;
    float dt = 0.0f;
    float lastTime = 0.0f;

    // Vulkan global
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties physicalDeviceProperties;
    VkDevice device;
    VkDescriptorPool descriptorPool;
    
    // swap chain
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkQueue presentQueue;

    // compute output image
    VkExtent2D outputImageExtent;
    std::vector<VkImage> outputImages;
    std::vector<VkDeviceMemory> outputImageMemories;
    std::vector<VkImageView> outputImageViews;
    VkSampler outputImageSampler;
    std::vector<VkBuffer> denoiseBuffers;
    std::vector<VkDeviceMemory> denoiseBufferMemories;
    std::vector<VkCommandBuffer> copyCommandBuffers;
    size_t outputImageSize;

    // uniform buffer
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemories;

    // sync objects
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> raytraceFinishedSemaphores;
    std::vector<VkSemaphore> copySemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> copyFences;

    // graphics pipeline
    VkRenderPass renderPass;
    std::vector<VkCommandBuffer> graphicsCommandBuffers;
    VkDescriptorSetLayout graphicsDescriptorSetLayout;
    std::vector<VkDescriptorSet> graphicsDescriptorSets;
    VkPipelineLayout graphicsPipelineLayout;
    VkPipeline graphicsPipeline;
    VkCommandPool graphicsCommandPool;
    VkQueue graphicsQueue;

    // compute pipeline
    std::vector<VkCommandBuffer> computeCommandBuffers;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    std::vector<VkDescriptorSet> computeDescriptorSets;
    VkPipelineLayout computePipelineLayout;
    VkPipeline computePipeline;
    VkCommandPool computeCommandPool;
    VkQueue computeQueue;

    // text
    const uint32_t MAX_CHAR_COUNT = 512;

    Font font;
    VkImage fontTexture;
    VkImageView fontTextureView;
    VkSampler fontTextureSampler;
    VkDeviceMemory fontTextureMemory;

    std::vector<VkBuffer> textVertexBuffers;
    std::vector<VkDeviceMemory> textVertexBufferMemories;
    std::vector<Vertex*> textVertexData;
    std::vector<uint32_t> textNumChars;

    VkDescriptorSetLayout textDescriptorSetLayout;
    std::vector<VkDescriptorSet> textDescriptorSets;
    VkPipelineLayout textPipelineLayout;
    VkPipeline textPipeline;

    // OIDN
    oidn::DeviceRef oidnDevice;
    oidn::FilterRef oidnFilter;
    bool useDenoising = false;
    void* colorPtr;
    void* outputPtr;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Disable OpenGL context
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Raytracing Project", nullptr, nullptr);

        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        glfwSetWindowUserPointer(window, this);
        glfwSetWindowSizeCallback(window, framebufferResizeCallback);
        glfwSetKeyCallback(window, keyCallback);
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        outputImageExtent = swapChainExtent;
        outputImageSize = outputImageExtent.width * outputImageExtent.height * 4 * sizeof(float);
        createImageViews();
        createRenderPass();
        createDescriptorSetLayouts();
        createGraphicsPipeline();
        createTextPipeline();
        createComputePipeline();
        createCommandPools();
        createFramebuffers();
        createOutputImages();
        createUniformBuffers();
        createFontTexture();
        createTextVertexBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void initOIDN() {
        oidnDevice = oidn::newDevice();
        oidnDevice.commit();

        oidnFilter = oidnDevice.newFilter("RT"); // generic ray tracing filter
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            update();
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    void cleanup() {
        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        cleanupSwapChain();

        vkDestroyDescriptorSetLayout(device, graphicsDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, textDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        vkDestroyCommandPool(device, graphicsCommandPool, nullptr);
        if (!queueFamilies.graphicsSameAsCompute())
            vkDestroyCommandPool(device, computeCommandPool, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(device, raytraceFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, copySemaphores[i], nullptr);
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
            vkDestroyFence(device, copyFences[i], nullptr);
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemories[i], nullptr);

            vkDestroyImageView(device, outputImageViews[i], nullptr);
            vkDestroyImage(device, outputImages[i], nullptr);
            vkFreeMemory(device, outputImageMemories[i], nullptr);
            vkDestroyBuffer(device, denoiseBuffers[i], nullptr);
            vkFreeMemory(device, denoiseBufferMemories[i], nullptr);

            vkDestroyBuffer(device, textVertexBuffers[i], nullptr);
            vkFreeMemory(device, textVertexBufferMemories[i], nullptr);
        }

        vkDestroySampler(device, outputImageSampler, nullptr);

        vkDestroyImageView(device, fontTextureView, nullptr);
        vkDestroyImage(device, fontTexture, nullptr);
        vkFreeMemory(device, fontTextureMemory, nullptr);
        vkDestroySampler(device, fontTextureSampler, nullptr);

        vkDestroyPipelineLayout(device, textPipelineLayout, nullptr);
        vkDestroyPipeline(device, textPipeline, nullptr);

        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
        vkDestroyPipeline(device, computePipeline, nullptr);

        vkDestroyDevice(device, nullptr);

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // extensions
        std::vector<const char*> extension_names = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extension_names.size());
        createInfo.ppEnabledExtensionNames = extension_names.data();

        // validation layers
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;
        }

        // create instance
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }

        //uint32_t extensionCount = 0;
        //vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

        //std::vector<VkExtensionProperties> extensions(extensionCount);
        //vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        //std::cout << extensionCount << " available extensions found:" << std::endl;

        //for (const auto& extension : extensions) {
        //    std::cout << '\t' << extension.extensionName << std::endl;
        //}
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        //std::cout << deviceCount << " devices found:" << std::endl;
        for (const VkPhysicalDevice& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }

        vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
        queueFamilies = findQueueFamilies(physicalDevice, surface);
    }

    void createLogicalDevice() {
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::unordered_set<uint32_t> uniqueQueueFamilies = {
            queueFamilies.graphicsFamily.value(),
            queueFamilies.computeFamily.value(),
            queueFamilies.presentFamily.value()
        };

        if (!queueFamilies.graphicsSameAsCompute())
            uniqueQueueFamilies.insert(queueFamilies.computeFamily.value());

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, queueFamilies.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, queueFamilies.computeFamily.value(), 0, &computeQueue);
        vkGetDeviceQueue(device, queueFamilies.presentFamily.value(), 0, &presentQueue);
    }

    void cleanupSwapChain() {
        for (VkFramebuffer framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (VkImageView imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        uint32_t queueFamilyIndices[] = { queueFamilies.graphicsFamily.value(), queueFamilies.presentFamily.value() };

        if (queueFamilies.graphicsFamily != queueFamilies.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // Optional
            createInfo.pQueueFamilyIndices = nullptr; // Optional
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
        
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkInit::ImageView2D(device, swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, &swapChainImageViews[i]);
        }
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstSubpass = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = nullptr;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createDescriptorSetLayouts() {
        // grpahics
        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 2;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        
        VkDescriptorSetLayoutCreateInfo graphicsLayoutInfo{};
        graphicsLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        graphicsLayoutInfo.bindingCount = 1;
        graphicsLayoutInfo.pBindings = &samplerLayoutBinding;

        if (vkCreateDescriptorSetLayout(device, &graphicsLayoutInfo, nullptr, &graphicsDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }

        // text
        samplerLayoutBinding.binding = 3;
        graphicsLayoutInfo.pBindings = &samplerLayoutBinding;

        if (vkCreateDescriptorSetLayout(device, &graphicsLayoutInfo, nullptr, &textDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }

        // compute
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        uboLayoutBinding.pImmutableSamplers = nullptr;
        
        VkDescriptorSetLayoutBinding outputImageLayoutBinding{};
        outputImageLayoutBinding.binding = 1;
        outputImageLayoutBinding.descriptorCount = 1;
        outputImageLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        outputImageLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        outputImageLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutBinding bindings[2] = { uboLayoutBinding, outputImageLayoutBinding };

        VkDescriptorSetLayoutCreateInfo computeLayoutInfo{};
        computeLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        computeLayoutInfo.bindingCount = 2;
        computeLayoutInfo.pBindings = bindings;
        
        if (vkCreateDescriptorSetLayout(device, &computeLayoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    void createGraphicsPipeline() {
        VkGraphicsPipelineCreateInfo pipelineCreateInfo{};

        // Shader stages
        std::vector<char> vertShaderCode = readFile("res/shaders/quad.vert.spv");
        std::vector<char> fragShaderCode = readFile("res/shaders/quad.frag.spv");

        VkShaderModule vertShaderModule, fragShaderModule;
        VkInit::ShaderModule(device, vertShaderCode, &vertShaderModule);
        VkInit::ShaderModule(device, fragShaderCode, &fragShaderModule);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        // Vertex input state
        VkPipelineVertexInputStateCreateInfo vertexInputState{};
        vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputState.vertexBindingDescriptionCount = 0;
        vertexInputState.pVertexBindingDescriptions = nullptr;
        vertexInputState.vertexAttributeDescriptionCount = 0;
        vertexInputState.pVertexAttributeDescriptions = nullptr;

        // Input assembly state
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{};
        inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        inputAssemblyState.primitiveRestartEnable = VK_FALSE;

        // Viewport state
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = (float)swapChainExtent.height;             // Modifying viewport origin and height
        viewport.width = (float)swapChainExtent.width;          // to account for Vulkan's coordinate system
        viewport.height = -1.0 * (float)swapChainExtent.height; // having a Y-axis that points down
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        // Rasterization state
        VkPipelineRasterizationStateCreateInfo rasterizationState{};
        rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationState.depthClampEnable = VK_FALSE;
        rasterizationState.rasterizerDiscardEnable = VK_FALSE;
        rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizationState.lineWidth = 1.0f;
        rasterizationState.cullMode = VK_CULL_MODE_NONE;
        rasterizationState.depthBiasEnable = VK_FALSE;
        rasterizationState.depthBiasConstantFactor = 0.0f; // Optional
        rasterizationState.depthBiasClamp = 0.0f; // Optional
        rasterizationState.depthBiasSlopeFactor = 0.0f; // Optional
        
        // Multisample state
        VkPipelineMultisampleStateCreateInfo multisampleState{};
        multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleState.sampleShadingEnable = VK_FALSE;
        multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampleState.minSampleShading = 1.0f; // Optional
        multisampleState.pSampleMask = nullptr; // Optional
        multisampleState.alphaToCoverageEnable = VK_FALSE; // Optional
        multisampleState.alphaToOneEnable = VK_FALSE; // Optional

        // Color blend state
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        VK_COLOR_COMPONENT_FLAG_BITS_MAX_ENUM;
        colorBlendAttachment.colorWriteMask = 0xf;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; // Optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; // Optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

        VkPipelineColorBlendStateCreateInfo colorBlendState{};
        colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendState.logicOpEnable = VK_FALSE;
        colorBlendState.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlendState.attachmentCount = 1;
        colorBlendState.pAttachments = &colorBlendAttachment;
        colorBlendState.blendConstants[0] = 0.0f; // Optional
        colorBlendState.blendConstants[1] = 0.0f; // Optional
        colorBlendState.blendConstants[2] = 0.0f; // Optional
        colorBlendState.blendConstants[3] = 0.0f; // Optional

        // Pipeline layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &graphicsDescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &graphicsPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stageCount = 2;
        pipelineCreateInfo.pStages = shaderStages;
        pipelineCreateInfo.pVertexInputState = &vertexInputState;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pTessellationState = nullptr;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pDepthStencilState = nullptr;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pDynamicState = nullptr;
        pipelineCreateInfo.layout = graphicsPipelineLayout;
        pipelineCreateInfo.renderPass = renderPass;
        pipelineCreateInfo.subpass = 0;
        pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineCreateInfo.basePipelineIndex = -1;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
    }

    void createTextPipeline() {
        VkGraphicsPipelineCreateInfo pipelineCreateInfo{};

        // Shader stages
        std::vector<char> vertShaderCode = readFile("res/shaders/text.vert.spv");
        std::vector<char> fragShaderCode = readFile("res/shaders/text.frag.spv");

        VkShaderModule vertShaderModule, fragShaderModule;
        VkInit::ShaderModule(device, vertShaderCode, &vertShaderModule);
        VkInit::ShaderModule(device, fragShaderCode, &fragShaderModule);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        // Vertex input state
        VkVertexInputBindingDescription vertexBindingDescription = Vertex::getBindingDescription();
        std::vector<VkVertexInputAttributeDescription> vertexAttributeDescriptions = Vertex::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputState{};
        vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputState.vertexBindingDescriptionCount = 1;
        vertexInputState.pVertexBindingDescriptions = &vertexBindingDescription;
        vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptions.size());
        vertexInputState.pVertexAttributeDescriptions = vertexAttributeDescriptions.data();

        // Input assembly state
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{};
        inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssemblyState.primitiveRestartEnable = VK_FALSE;

        // Viewport state
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = (float)swapChainExtent.height;             // Modifying viewport origin and height
        viewport.width = (float)swapChainExtent.width;          // to account for Vulkan's coordinate system
        viewport.height = -1.0 * (float)swapChainExtent.height; // having a Y-axis that points down
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        // Rasterization state
        VkPipelineRasterizationStateCreateInfo rasterizationState{};
        rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationState.depthClampEnable = VK_FALSE;
        rasterizationState.rasterizerDiscardEnable = VK_FALSE;
        rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizationState.lineWidth = 1.0f;
        rasterizationState.cullMode = VK_CULL_MODE_NONE;
        rasterizationState.depthBiasEnable = VK_FALSE;
        rasterizationState.depthBiasConstantFactor = 0.0f; // Optional
        rasterizationState.depthBiasClamp = 0.0f; // Optional
        rasterizationState.depthBiasSlopeFactor = 0.0f; // Optional

        // Multisample state
        VkPipelineMultisampleStateCreateInfo multisampleState{};
        multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleState.sampleShadingEnable = VK_FALSE;
        multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampleState.minSampleShading = 1.0f; // Optional
        multisampleState.pSampleMask = nullptr; // Optional
        multisampleState.alphaToCoverageEnable = VK_FALSE; // Optional
        multisampleState.alphaToOneEnable = VK_FALSE; // Optional

        // Color blend state
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        VK_COLOR_COMPONENT_FLAG_BITS_MAX_ENUM;
        colorBlendAttachment.colorWriteMask = 0xf;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; // Optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; // Optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

        VkPipelineColorBlendStateCreateInfo colorBlendState{};
        colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendState.logicOpEnable = VK_FALSE;
        colorBlendState.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlendState.attachmentCount = 1;
        colorBlendState.pAttachments = &colorBlendAttachment;
        colorBlendState.blendConstants[0] = 0.0f; // Optional
        colorBlendState.blendConstants[1] = 0.0f; // Optional
        colorBlendState.blendConstants[2] = 0.0f; // Optional
        colorBlendState.blendConstants[3] = 0.0f; // Optional

        // Pipeline layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &textDescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &textPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stageCount = 2;
        pipelineCreateInfo.pStages = shaderStages;
        pipelineCreateInfo.pVertexInputState = &vertexInputState;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pTessellationState = nullptr;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pDepthStencilState = nullptr;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pDynamicState = nullptr;
        pipelineCreateInfo.layout = textPipelineLayout;
        pipelineCreateInfo.renderPass = renderPass;
        pipelineCreateInfo.subpass = 0;
        pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineCreateInfo.basePipelineIndex = -1;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &textPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create text pipeline!");
        }

        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
    }

    void createComputePipeline() {
        VkComputePipelineCreateInfo pipelineCreateInfo{};

        // Shader stage
        std::vector<char> compShaderCode = readFile("res/shaders/raytrace.comp.spv");

        VkShaderModule compShaderModule;
        VkInit::ShaderModule(device, compShaderCode, &compShaderModule);

        VkPipelineShaderStageCreateInfo compShaderStageInfo{};
        compShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        compShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        compShaderStageInfo.module = compShaderModule;
        compShaderStageInfo.pName = "main";

        // Pipeline layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // Pipeline creation
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stage = compShaderStageInfo;
        pipelineCreateInfo.layout = computePipelineLayout;
        pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineCreateInfo.basePipelineIndex = -1;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &computePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, compShaderModule, nullptr);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &swapChainImageViews[i];
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createCommandPools() {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilies.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &graphicsCommandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }

        if (queueFamilies.graphicsSameAsCompute()) {
            computeCommandPool = graphicsCommandPool;
        } else {
            poolInfo.queueFamilyIndex = queueFamilies.computeFamily.value();
            if (vkCreateCommandPool(device, &poolInfo, nullptr, &computeCommandPool) != VK_SUCCESS) {
                throw std::runtime_error("failed to create command pool!");
            }
        }
    }

    void createOutputImages() {
        outputImages.resize(MAX_FRAMES_IN_FLIGHT);
        outputImageViews.resize(MAX_FRAMES_IN_FLIGHT);
        outputImageMemories.resize(MAX_FRAMES_IN_FLIGHT);
        denoiseBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        denoiseBufferMemories.resize(MAX_FRAMES_IN_FLIGHT);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            VkInit::Image2D(
                physicalDevice,
                device,
                swapChainExtent.width,
                swapChainExtent.height,
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                &outputImages[i],
                &outputImageMemories[i]
            );

            VkInit::ImageView2D(
                device,
                outputImages[i],
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                &outputImageViews[i]
            );

            VkMemoryRequirements memReq{};
            vkGetImageMemoryRequirements(device, outputImages[i], &memReq);
            VkInit::Buffer(
                physicalDevice,
                device,
                memReq.size,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                &denoiseBuffers[i],
                &denoiseBufferMemories[i]
            );
            
            VkCommandBuffer commandBuffer = startSingleTimeCommand(device, graphicsCommandPool);
            VkCmd::transitionImageLayout(commandBuffer, outputImages[i], VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            endSingleTimeCommand(device, graphicsCommandPool, graphicsQueue, commandBuffer);
        }

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = physicalDeviceProperties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        if (vkCreateSampler(device, &samplerInfo, nullptr, &outputImageSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemories.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            VkInit::Buffer(
                physicalDevice,
                device,
                bufferSize,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &uniformBuffers[i],
                &uniformBuffersMemories[i]
            );
        }
    }

    void createFontTexture() {
        font.loadFont("res/fonts/arial.ttf", 24);

        // writePPM("fontTextureP3.ppm", Font::TEXTURE_WIDTH, Font::TEXTURE_HEIGHT, font.getFontPixels());

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VkMemoryRequirements memReq;

        VkInit::Image2D(
            physicalDevice,
            device,
            Font::TEXTURE_WIDTH,
            Font::TEXTURE_HEIGHT,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &fontTexture,
            &fontTextureMemory
        );

        vkGetImageMemoryRequirements(device, fontTexture, &memReq);

        VkInit::Buffer(
            physicalDevice,
            device,
            memReq.size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &stagingBuffer,
            &stagingBufferMemory,
            font.getFontPixels(),
            Font::TEXTURE_SIZE
        );

        VkCommandBuffer commandBuffer = startSingleTimeCommand(device, graphicsCommandPool);
        VkCmd::transitionImageLayout(commandBuffer, fontTexture, VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkCmd::copyBufferToImage(commandBuffer, stagingBuffer, fontTexture, Font::TEXTURE_WIDTH, Font::TEXTURE_HEIGHT);
        VkCmd::transitionImageLayout(commandBuffer, fontTexture, VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        endSingleTimeCommand(device, graphicsCommandPool, graphicsQueue, commandBuffer);

        vkFreeMemory(device, stagingBufferMemory, nullptr);
        vkDestroyBuffer(device, stagingBuffer, nullptr);

        VkInit::ImageView2D(
            device,
            fontTexture,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_ASPECT_COLOR_BIT,
            &fontTextureView
        );

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = physicalDeviceProperties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_WHITE;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 1.0f;

        if (vkCreateSampler(device, &samplerInfo, nullptr, &fontTextureSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create font texture sampler!");
        }
    }

    void createTextVertexBuffers() {
        textVertexBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        textVertexBufferMemories.resize(MAX_FRAMES_IN_FLIGHT);
        textVertexData.resize(MAX_FRAMES_IN_FLIGHT);
        textNumChars.resize(MAX_FRAMES_IN_FLIGHT);
        
        VkDeviceSize bufferSize = MAX_CHAR_COUNT * 6 * sizeof(Vertex);
         
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            VkInit::Buffer(
                physicalDevice,
                device,
                bufferSize,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &textVertexBuffers[i],
                &textVertexBufferMemories[i]
            );

            textNumChars[i] = 0;
        }
    }

    void createDescriptorPool() {
        VkDescriptorPoolSize uboPoolSize{};
        uboPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboPoolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        VkDescriptorPoolSize samplerPoolSize{};
        samplerPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerPoolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 2);

        VkDescriptorPoolSize imagePoolSize{};
        imagePoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        imagePoolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        VkDescriptorPoolSize poolSizes[3] = { uboPoolSize, samplerPoolSize, imagePoolSize };

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = poolSizes;
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 3);

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createDescriptorSets() {
        graphicsDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        textDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        computeDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);

        // Allocate graphics descriptor set
        std::vector<VkDescriptorSetLayout> graphicsLayouts(MAX_FRAMES_IN_FLIGHT, graphicsDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(graphicsLayouts.size());
        allocInfo.pSetLayouts = graphicsLayouts.data();
        
        if (vkAllocateDescriptorSets(device, &allocInfo, graphicsDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate graphics descriptor sets!");
        }

        // Allocate text descriptor set
        std::vector<VkDescriptorSetLayout> textLayouts(MAX_FRAMES_IN_FLIGHT, textDescriptorSetLayout);
        allocInfo.descriptorSetCount = static_cast<uint32_t>(textLayouts.size());
        allocInfo.pSetLayouts = textLayouts.data();

        if (vkAllocateDescriptorSets(device, &allocInfo, textDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate text descriptor sets!");
        }

        // Allocate compute descriptor set
        std::vector<VkDescriptorSetLayout> computeLayouts(MAX_FRAMES_IN_FLIGHT, computeDescriptorSetLayout);
        allocInfo.descriptorSetCount = static_cast<uint32_t>(computeLayouts.size());
        allocInfo.pSetLayouts = computeLayouts.data();

        if (vkAllocateDescriptorSets(device, &allocInfo, computeDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate compute descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            // graphics
            VkDescriptorImageInfo readImageInfo{};
            readImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            readImageInfo.imageView = outputImageViews[i];
            readImageInfo.sampler = outputImageSampler;

            VkWriteDescriptorSet descriptorReadImage{};
            descriptorReadImage.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorReadImage.dstSet = graphicsDescriptorSets[i];
            descriptorReadImage.dstBinding = 2;
            descriptorReadImage.dstArrayElement = 0;
            descriptorReadImage.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorReadImage.descriptorCount = 1;
            descriptorReadImage.pBufferInfo = nullptr;
            descriptorReadImage.pImageInfo = &readImageInfo;
            descriptorReadImage.pTexelBufferView = nullptr;

            vkUpdateDescriptorSets(device, 1, &descriptorReadImage, 0, nullptr);

            // text
            VkDescriptorImageInfo fontTextureInfo{};
            fontTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            fontTextureInfo.imageView = fontTextureView;
            fontTextureInfo.sampler = fontTextureSampler;

            descriptorReadImage.dstSet = textDescriptorSets[i];
            descriptorReadImage.dstBinding = 3;
            descriptorReadImage.pImageInfo = &fontTextureInfo;

            vkUpdateDescriptorSets(device, 1, &descriptorReadImage, 0, nullptr);

            // compute
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkWriteDescriptorSet descriptorBuffer{};
            descriptorBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorBuffer.dstSet = computeDescriptorSets[i];
            descriptorBuffer.dstBinding = 0;
            descriptorBuffer.dstArrayElement = 0;
            descriptorBuffer.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorBuffer.descriptorCount = 1;
            descriptorBuffer.pBufferInfo = &bufferInfo;
            descriptorBuffer.pImageInfo = nullptr;
            descriptorBuffer.pTexelBufferView = nullptr;

            VkDescriptorImageInfo writeImageInfo{};
            writeImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            writeImageInfo.imageView = outputImageViews[i];

            VkWriteDescriptorSet descriptorWriteImage{};
            descriptorWriteImage.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWriteImage.dstSet = computeDescriptorSets[i];
            descriptorWriteImage.dstBinding = 1;
            descriptorWriteImage.dstArrayElement = 0;
            descriptorWriteImage.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWriteImage.descriptorCount = 1;
            descriptorWriteImage.pBufferInfo = nullptr;
            descriptorWriteImage.pImageInfo = &writeImageInfo;
            descriptorWriteImage.pTexelBufferView = nullptr;

            VkWriteDescriptorSet descriptors[2] = { descriptorBuffer, descriptorWriteImage };

            vkUpdateDescriptorSets(device, 2, descriptors, 0, nullptr);
        }
    }

    void createCommandBuffers() {
        graphicsCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        copyCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkInit::CommandBuffers(
            device,
            graphicsCommandPool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            static_cast<uint32_t>(graphicsCommandBuffers.size()),
            graphicsCommandBuffers.data()
        );
        
        VkInit::CommandBuffers(
            device,
            computeCommandPool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            static_cast<uint32_t>(computeCommandBuffers.size()),
            computeCommandBuffers.data()
        );

        VkInit::CommandBuffers(
            device,
            graphicsCommandPool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            static_cast<uint32_t>(copyCommandBuffers.size()),
            copyCommandBuffers.data()
        );
    }

    void recordGraphicsCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        // Begin buffer
        VkCommandBufferBeginInfo commandBufferBeginInfo{};
        commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        commandBufferBeginInfo.flags = 0;
        commandBufferBeginInfo.pInheritanceInfo = nullptr;

        if (vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkCmd::transitionImageLayout(
            commandBuffer,
            outputImages[currentFrame],
            VK_FORMAT_R32G32B32A32_SFLOAT,
            useDenoising ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            queueFamilies.graphicsFamily.value(),
            queueFamilies.graphicsFamily.value()
        );

        // Begin render pass
        VkClearValue clearValues[2];
        clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo renderPassBeginInfo{};
        renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassBeginInfo.renderArea.offset = { 0, 0 };
        renderPassBeginInfo.renderArea.extent = swapChainExtent;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Draw scene
         vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
         vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsDescriptorSets[currentFrame], 0, nullptr);
         vkCmdDraw(commandBuffer, 4, 1, 0, 0);

        // Draw text
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, textPipeline);
        VkDeviceSize offset[1] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &textVertexBuffers[currentFrame], offset);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, textPipelineLayout, 0, 1, &textDescriptorSets[currentFrame], 0, nullptr);
        vkCmdDraw(commandBuffer, 6 * textNumChars[currentFrame], 1, 0, 0);

        // End
        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void recordComputeCommandBuffer(VkCommandBuffer commandBuffer) {
        // Begin buffer
        VkCommandBufferBeginInfo commandBufferBeginInfo{};
        commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        commandBufferBeginInfo.flags = 0;
        commandBufferBeginInfo.pInheritanceInfo = nullptr;

        if (vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkCmd::transitionImageLayout(
            commandBuffer,
            outputImages[currentFrame],
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            queueFamilies.graphicsFamily.value(),
            queueFamilies.computeFamily.value()
        );

        // Raytrace
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);

        vkCmdDispatch(commandBuffer, outputImageExtent.width / 16, outputImageExtent.height / 16, 1);

        // End
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        raytraceFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        copySemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        copyFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        
        VkFenceCreateInfo signaledFenceInfo{};
        signaledFenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        signaledFenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS   ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &raytraceFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &copySemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS   ||
                vkCreateFence(device, &signaledFenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &copyFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create sync objects!");
            }
        }
    }

    void update() {
        float curTime = static_cast<float>(glfwGetTime());
        dt = curTime - lastTime;
        lastTime = curTime;

        updateMouseInput();
        camera.move(dt);
        camera.updateVectors();

        updateText();
    }

    void drawFrame() {
        // Wait for last frame
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // Acquire swap chain image
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        // Check for window resize
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            recreateSwapChain();
            framebufferResized = false;
            return;
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        updateUniformBuffer(currentFrame);

        // Record and submit raytrace command buffer
        vkResetCommandBuffer(computeCommandBuffers[currentFrame], 0);
        recordComputeCommandBuffer(computeCommandBuffers[currentFrame]);

        VkSemaphore computeWaitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags computeWaitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSemaphore computeSignalSemaphores[] = { raytraceFinishedSemaphores[currentFrame] };

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &computeCommandBuffers[currentFrame];
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = computeWaitSemaphores;
        submitInfo.pWaitDstStageMask = computeWaitStages;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = computeSignalSemaphores;

        if (vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit raytrace command buffer!");
        }

        if (useDenoising) {
            // Copy output image to denoise buffer
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            vkResetCommandBuffer(copyCommandBuffers[currentFrame], 0);
            vkBeginCommandBuffer(copyCommandBuffers[currentFrame], &beginInfo);
            VkCmd::transitionImageLayout(
                copyCommandBuffers[currentFrame],
                outputImages[currentFrame],
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                queueFamilies.computeFamily.value(),
                queueFamilies.graphicsFamily.value()
            );
            VkCmd::copyImageToBuffer(copyCommandBuffers[currentFrame], denoiseBuffers[currentFrame], outputImages[currentFrame], outputImageExtent.width, outputImageExtent.height);
            vkEndCommandBuffer(copyCommandBuffers[currentFrame]);

            VkSemaphore copyWaitSemaphores[] = { raytraceFinishedSemaphores[currentFrame] };
            VkPipelineStageFlags copyWaitStages[] = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };

            submitInfo.pCommandBuffers = &copyCommandBuffers[currentFrame];
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = copyWaitSemaphores;
            submitInfo.pWaitDstStageMask = copyWaitStages;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = nullptr;

            if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, copyFences[currentFrame]) != VK_SUCCESS) {
                throw std::runtime_error("failed to submit image-to-buffer copy command buffer!");
            }

            vkWaitForFences(device, 1, &copyFences[currentFrame], VK_TRUE, UINT64_MAX);
            vkResetFences(device, 1, &copyFences[currentFrame]);

            // Denoise output image using OIDN
            colorPtr = nullptr;
            outputPtr = (void*)malloc(outputImageSize);

            vkMapMemory(device, denoiseBufferMemories[currentFrame], 0, (VkDeviceSize)outputImageSize, 0, &colorPtr);

            size_t pixelByteStride = 4 * sizeof(float);
            size_t rowStride = outputImageExtent.width * pixelByteStride;
            oidnFilter.setImage("color", colorPtr, oidn::Format::Float3, outputImageExtent.width, outputImageExtent.height, 0, pixelByteStride, rowStride);
            oidnFilter.setImage("output", outputPtr, oidn::Format::Float3, outputImageExtent.width, outputImageExtent.height, 0, pixelByteStride, rowStride);
            oidnFilter.commit();
            oidnFilter.execute();

            const char* oidnErrorMsg;
            if (oidnDevice.getError(oidnErrorMsg) == oidn::Error::None) {
                std::memcpy(colorPtr, outputPtr, outputImageSize);
            } else {
                std::cout << "OIDN error: " << oidnErrorMsg << std::endl;
            }

            vkUnmapMemory(device, denoiseBufferMemories[currentFrame]);

            free(outputPtr);
            outputPtr = nullptr;

            // Copy denoised image in denoise buffer to output image
            vkResetCommandBuffer(copyCommandBuffers[currentFrame], 0);
            vkBeginCommandBuffer(copyCommandBuffers[currentFrame], &beginInfo);
            VkCmd::transitionImageLayout(
                copyCommandBuffers[currentFrame],
                outputImages[currentFrame],
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                queueFamilies.graphicsFamily.value(),
                queueFamilies.graphicsFamily.value()
            );
            VkCmd::copyBufferToImage(copyCommandBuffers[currentFrame], denoiseBuffers[currentFrame], outputImages[currentFrame], outputImageExtent.width, outputImageExtent.height);
            vkEndCommandBuffer(copyCommandBuffers[currentFrame]);

            VkSemaphore copySignalSemaphores[] = { copySemaphores[currentFrame] };

            submitInfo.pCommandBuffers = &copyCommandBuffers[currentFrame];
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.pWaitSemaphores = nullptr;
            submitInfo.pWaitDstStageMask = nullptr;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = copySignalSemaphores;

            if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
                throw std::runtime_error("failed to submit buffer-to-image copy command buffer!");
            }
        }

        // Record and submit draw command buffer
        vkResetCommandBuffer(graphicsCommandBuffers[currentFrame], 0);
        recordGraphicsCommandBuffer(graphicsCommandBuffers[currentFrame], imageIndex);

        VkSemaphore graphicsWaitSemaphores[] = { useDenoising ? copySemaphores[currentFrame] : raytraceFinishedSemaphores[currentFrame] };
        VkPipelineStageFlags graphicsWaitStages[] = { VK_PIPELINE_STAGE_TRANSFER_BIT };
        VkSemaphore graphicsSignalSemaphores[] = { renderFinishedSemaphores[currentFrame] };

        submitInfo.pCommandBuffers = &graphicsCommandBuffers[currentFrame];
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = graphicsWaitSemaphores;
        submitInfo.pWaitDstStageMask = graphicsWaitStages;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = graphicsSignalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        // Present swap chain image
        VkSwapchainKHR swapChains[] = { swapChain };
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = graphicsSignalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr; // Optional

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        // Check for window resize
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            recreateSwapChain();
            framebufferResized = false;
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        //std::cout << "checking validation layers support:" << std::endl;

        for (const char* layerName : validationLayers) {
            bool layerFound = false;
            
            for (const auto& layerProperties : availableLayers) {
                if (std::strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    //std::cout << "\t" << layerName << std::endl;
                    break;
                }
            }

            if (!layerFound) {
                //std::cout << layerName << " not supported!" << std::endl;
                return false;
            }
        }

        return true;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData
    ) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        
        return VK_FALSE;
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        RayTracingApplication* app = static_cast<RayTracingApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        RayTracingApplication* app = static_cast<RayTracingApplication*>(glfwGetWindowUserPointer(window));

        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        } else if (key == GLFW_KEY_F && action == GLFW_PRESS) {
            app->useDenoising = !app->useDenoising;
            if (!app->useDenoising) {
                app->firstMouse = true;
            }
        }

        if (!app->useDenoising) {
            app->camera.handleKeyInput(window, key, action);
        }
    }

    VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr) {
            return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
        } else {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }

    void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(instance, debugMessenger, pAllocator);
        }
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);

        VkPhysicalDeviceFeatures deviceFeatures{};
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        QueueFamilyIndices indices = findQueueFamilies(device, surface);

        bool extensionSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        //std::cout << '\t' << deviceProperties.deviceName << std::endl;

        return indices.isComplete() && extensionSupported && swapChainAdequate && deviceFeatures.samplerAnisotropy;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::unordered_set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const VkExtensionProperties& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount > 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount > 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const VkSurfaceFormatKHR& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const VkPresentModeKHR& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    VkFormat findDepthFormat() {
        return findSupportedFormat(
            physicalDevice,
            { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }

    void updateUniformBuffer(uint32_t currentFrame) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.view_info = glm::vec4(WIDTH, HEIGHT, numSamples, camera.getVFOV());
        ubo.camera_position = glm::vec4(camera.getPosition(), 0.0f);
        ubo.camera_front = glm::vec4(camera.getFront(), 0.0f);
        ubo.camera_right = glm::vec4(camera.getRight(), 0.0f);
        ubo.camera_up = glm::vec4(camera.getUp(), 0.0f);

        void* data;
        vkMapMemory(device, uniformBuffersMemories[currentFrame], 0, sizeof(ubo), 0, &data);
        std::memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemories[currentFrame]);
    }

    void updateMouseInput() {
        if (useDenoising) return;

        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);

        if (firstMouse) {
            lastMouseX = mouseX;
            lastMouseY = mouseY;
            firstMouse = false;
        }

        double mouseOffsetX = mouseX - lastMouseX;
        double mouseOffsetY = lastMouseY - mouseY;

        camera.handleMouseInput(dt, mouseOffsetX, mouseOffsetY);

        lastMouseX = mouseX;
        lastMouseY = mouseY;
    }

    void addText(const std::string text, float x, float y, const float scale, const glm::vec3 color) {
        Vertex* data = textVertexData[currentFrame];
        
        assert(textVertexData[currentFrame] != nullptr);
        
        uint32_t numChars = textNumChars[currentFrame];

        if (numChars + text.length() >= MAX_CHAR_COUNT)
            return;

        float textWidth = 0.0f;

        for (char ch : text) {
            textWidth += font.getCharacter(ch).advance * scale;
        }

        for (size_t i = 0; i < text.length(); ++i) {
            char ch = text.at(i);
            Character charData = font.getCharacter(ch);

            float charX = (x + charData.bearing.x * scale) / (WIDTH / 2.0f);
            float charY = (y + (charData.bearing.y - charData.size.y) * scale) / (HEIGHT / 2.0f);
            float charWidth = charData.size.x * scale / (WIDTH / 2.0f);
            float charHeight = charData.size.y * scale / (HEIGHT / 2.0f);
            
            Vertex topLeft     { glm::vec2(charX,             charY             ), charData.texCoords[0], color };
            Vertex topRight    { glm::vec2(charX + charWidth, charY             ), charData.texCoords[1], color };
            Vertex bottomRight { glm::vec2(charX + charWidth, charY + charHeight), charData.texCoords[2], color };
            Vertex bottomLeft  { glm::vec2(charX,             charY + charHeight), charData.texCoords[3], color };

            data[numChars * 6    ] = topLeft;
            data[numChars * 6 + 1] = topRight;
            data[numChars * 6 + 2] = bottomRight;
            data[numChars * 6 + 3] = topLeft;
            data[numChars * 6 + 4] = bottomRight;
            data[numChars * 6 + 5] = bottomLeft;

            x += (charData.advance >> 6) * scale;

            ++numChars;
        }

        textNumChars[currentFrame] = numChars;
    }

    void updateText() {
        textNumChars[currentFrame] = 0;
        textVertexData[currentFrame] = nullptr;
        glm::vec3 textColor = useDenoising ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        glm::vec2 topLeft((float)WIDTH / -2.0f, (float)HEIGHT / 2.0f);
        float scale = 1.0f;

        vkMapMemory(device, textVertexBufferMemories[currentFrame], 0, VK_WHOLE_SIZE, 0, (void**)&textVertexData[currentFrame]);

        addText("SPP: " + std::to_string(numSamples), topLeft.x, topLeft.y - scale * font.getFontSize(), scale, glm::vec3(0.0f));
        addText(useDenoising ? "OIDN filter [F]: ON" : "OIDN filter [F]: OFF", topLeft.x, topLeft.y - 2 * scale * font.getFontSize(), scale, textColor);

        vkUnmapMemory(device, textVertexBufferMemories[currentFrame]);
    }
};

int main() {
    RayTracingApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}