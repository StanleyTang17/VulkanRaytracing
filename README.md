# Vulkan Raytracing Project
A real-time raytracing program in Vulkan. This program uses compute shaders to compute the raytraced pixels and optionally uses Intel's Open Image Denoiser to denoise the raytraced image.

## Gallery

## Build
### Linux x86_64 Build
Prerequisites:
- CMake <=3.10
- GNU make <=4.2
- LunarG Vulkan SDK
    - Install from https://vulkan.lunarg.com/sdk/home#linux
- OpenGL headers
    - Install from the package `mesa-common-dev`
- GLFW3
    - Install from the packages `libglfw3` `libglfw3-dev`

Build instructions:
1. Go to project root directory
2. `mkdir build && cd build`
3. `cmake -DPLATFORM="Linux_x86_64" ../`
4. `make`
5. `cp VulkanRaytracing ../`