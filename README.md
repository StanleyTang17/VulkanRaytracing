# Vulkan Raytracing Project
A real-time raytracing program in Vulkan. This program uses compute shaders to compute the raytraced pixels and optionally uses Intel's Open Image Denoiser to denoise the raytraced image.

## Gallery

## Build
Prerequisites:
- CMake <=3.10
- Build tool of your choice (GNU make, Visual Studio, etc)

Build instructions:
1. Go to project root directory
2. Create a `build` folder
3. Go inside the `build` folder
4. Run CMake. You can either run `cmake ../` on the command line or use the CMake GUI if you are on Windows
5. Build using your tool of choice:
- For GNU make users, simply run `make` on the command line
- For Visual Studio users, open `VulkanRaytracing.sln` with Visual Studio, select your build configuration (e.g. Debug x64 or Release x64), and click `Build -> Build VulkanRaytracing`.

## Running on Windows
In order to run the program on Windows, be sure to copy these DLLs from the `lib` folder to the same folder as the executable:
- `glfw3.dll`
- `OpenImageDenoise.dll`
- `tbb12.dll`