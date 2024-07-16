cmake_minimum_required(VERSION 3.8)
project(yolov8)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set CUDA early so OpenCV can find it
# Set the version of CUDA to use
set(CUDA_VERSION "11.8")
# # Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-${CUDA_VERSION}")
# # Set CMAKE_CUDA_COMPILER to the correct version of nvcc
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-${CUDA_VERSION}/bin/nvcc")

# Find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(yolov8_interfaces REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# # CMake Options
# option(ENABLE_BENCHMARKS "Benchmark the preprocessing, inference, and post processing" OFF)
# if (ENABLE_BENCHMARKS)
#     add_compile_definitions(ENABLE_BENCHMARKS)
# endif()

# Build the TensorRT inference engine library
# TensorRT is found and linked by the tensorrt-cpp-api
add_subdirectory(libs/tensorrt-cpp-api)

# Build the YoloV8 library
add_library(YoloV8_TRT SHARED src/yolov8.cpp)
target_link_libraries(YoloV8_TRT PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(YoloV8_TRT PUBLIC libs/tensorrt-cpp-api/src)

add_executable(ros_segmentation src/ros_segmentation.cpp)
ament_target_dependencies(ros_segmentation rclcpp sensor_msgs cv_bridge yolov8_interfaces)
target_link_libraries(ros_segmentation YoloV8_TRT)

#
install(TARGETS
  ros_segmentation
  DESTINATION lib/${PROJECT_NAME}
)

# Moves YoloV8_TRT library to the install folder
install(TARGETS
  YoloV8_TRT
  tensorrt_cpp_api
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  DESTINATION lib/${PROJECT_NAME}
)

# Places the launch, config, and models folders in the install/yolov8/share/yolov8 directory so they can be found by the launch file
install(DIRECTORY
  launch
  config
  models
  DESTINATION share/${PROJECT_NAME}/
  PATTERN ".gitignore" EXCLUDE # Exclude .gitignore files from being installed
)

# Allows other packages to find and link to the YoloV8_TRT and tensorrt_cpp_api libraries
ament_export_include_directories(include)
ament_export_libraries(YOLOV8_TRT tensorrt_cpp_api yolov8_interfaces)

ament_package()