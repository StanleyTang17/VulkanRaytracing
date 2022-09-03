cmake_minimum_required(VERSION 3.10)

project(VulkanRaytracing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(${PLATFORM} MATCHES "Windows")
    set(PROJECT_LIB_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/lib/windows")
elseif(${PLATFORM} MATCHES "Linux_x86_64")
    set(PROJECT_LIB_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/lib/linux_x86_64")
else()
    message(FATAL_ERROR "Platform not specified")
endif()


# Source files
list(APPEND PROJECT_SRC_FILES "main.cpp")
list(APPEND PROJECT_SRC_FILES "camera.cpp")
list(APPEND PROJECT_SRC_FILES "vkutil.cpp")
list(APPEND PROJECT_SRC_FILES "font.cpp")

foreach(SRC_FILE ${PROJECT_SRC_FILES})
	list(APPEND PROJECT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/${SRC_FILE}")
endforeach()

add_executable(VulkanRaytracing ${PROJECT_SOURCES})


# Libraries
if(${PLATFORM} MATCHES "Windows")
    list(APPEND PROJECT_LIB_FILES "vulkan-1.lib")
    list(APPEND PROJECT_LIB_FILES "glfw3.lib")
    list(APPEND PROJECT_LIB_FILES "OpenImageDenoise.lib")
    list(APPEND PROJECT_LIB_FILES "freetype.lib")
elseif(${PLATFORM} MATCHES "Linux_x86_64")
    list(APPEND PROJECT_LIBRARIES "libvulkan.so")
    list(APPEND PROJECT_LIBRARIES "libglfw.so")
    list(APPEND PROJECT_LIB_FILES "libOpenImageDenoise.so")
    list(APPEND PROJECT_LIB_FILES "libfreetype.a")
endif()

foreach(LIB_FILE ${PROJECT_LIB_FILES})
    list(APPEND PROJECT_LIBRARIES "${PROJECT_LIB_FOLDER}/${LIB_FILE}")
endforeach()

target_link_directories(VulkanRaytracing PUBLIC ${PROJECT_LIB_FOLDER})
target_link_libraries(VulkanRaytracing PUBLIC ${PROJECT_LIBRARIES})


# Include dirs
list(APPEND PROJECT_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND PROJECT_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_include_directories(VulkanRaytracing PUBLIC ${PROJECT_INCLUDE_DIRS})