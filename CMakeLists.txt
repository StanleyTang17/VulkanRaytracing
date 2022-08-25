cmake_minimum_required(VERSION 3.10)

project(VulkanRaytracing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND PROJECT_LIBS "vulkan-1.lib")
list(APPEND PROJECT_LIBS "glfw3.lib")
list(APPEND PROJECT_LIBS "OpenImageDenoise.lib")

list(APPEND PROJECT_SRC_FILES "main.cpp")
list(APPEND PROJECT_SRC_FILES "camera.cpp")
list(APPEND PROJECT_SRC_FILES "vkutil.cpp")

list(APPEND PROJECT_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND PROJECT_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")

foreach(SRC_FILE ${PROJECT_SRC_FILES})
	list(APPEND PROJECT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/${SRC_FILE}")
endforeach()

add_executable(VulkanRaytracing ${PROJECT_SOURCES})

target_link_directories(VulkanRaytracing PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/lib")

target_link_libraries(VulkanRaytracing PUBLIC ${PROJECT_LIBS})

target_include_directories(VulkanRaytracing PUBLIC ${PROJECT_INCLUDE_DIRS})