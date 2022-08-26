# Vulkan Raytracing Project
A real-time raytracing program in Vulkan. This program uses compute shaders to compute the raytraced pixels and optionally uses Intel's Open Image Denoiser to denoise the raytraced image.

## Gallery

## Build
### Windows Build
##### Prerequisites:
- CMake <=3.10
- [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/)
- Visual Studio

##### Build Instructions

> Note: you can do step 2 to 6 on the command line if cmake is added to the PATH environment variable. Just run `cmake -G <Visual Studio Generator> -DPLATFORM="Windows" ../` inside the `build` folder.

1. Create a `build` folder inside the project root directory
2. Open the CMake GUI
3. Specify the source directory to be the project root directory
4. Specify the build directory to be the `build` folder inside the project root directory
5. Add an entry of type `STRING` with `PLATFORM` as the name and `Windows` as the value
6. Click on `Configure` then `Generate`
7. Now there should be a `VulkanRaytracing.sln` file inside the `build` folder. Open it with Visual Studio.
8. Click on the VulkanRaytracing solution in Solution Explorer
9. Select the build (e.g. Debug x64 or Release x64)
10. Click on `Build -> Build VulkanRaytracing`
11. Copy the executable from the output folder to the project root directory
12. Copy `glfw3.dll` `OpenImageDenoiser.dll` `tbb12.dll` from the `lib/windows` folder to the project root directory
13. Run the executable in the project root directory

### Linux x86_64 Build
##### Prerequisites
- CMake <=3.10
- GNU make <=4.2
- [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home#linux)
- OpenGL headers
    - Install from the package `mesa-common-dev`
- GLFW3
    - Install from the packages `libglfw3` `libglfw3-dev`

##### Build Instructions
1. Go to project root directory
2. `mkdir build && cd build`
3. `cmake -DPLATFORM="Linux_x86_64" ../`
4. `make`
5. `cp VulkanRaytracing ../`
6. Run the executable in the project root directory