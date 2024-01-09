cmake_minimum_required(VERSION 3.8)
project(tensorrt_cpp_api)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR /home/ekberndt/libs/TensorRT-8.6.1.6/)

if (NOT TensorRT_DIR)
    set(TensorRT_DIR /home/ekberndt/libs/TensorRT-8.6.1.6/)
    message(FATAL_ERROR "TensorRT_DIR not specified. Please set the path to TensorRT root directory.")
    return()
endif()

# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)

# Set CMAKE_CUDA_COMPILER to the correct version of nvcc
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
# enable_language(CUDA)

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
if (NOT TensorRT_FOUND)
    message(FATAL_ERROR "TensorRT not found. Please make sure TensorRT is installed.")
    return()
endif()

find_package(CUDA REQUIRED)
if (NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. Please make sure CUDA is installed.")
    return()
endif()

# Set the path to the OpenCV build
set(OpenCV_DIR /opencv-4.8.0/build)

find_package(OpenCV 4.8.0 REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please make sure OpenCV is installed.")
    return()
endif()

# Link OpenCV directories to ensure CUDA libraries are found
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

add_library(tensorrt_cpp_api SHARED
        src/engine.cpp)

target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

add_executable(run_inference_benchmark src/main.cpp)
target_link_libraries(run_inference_benchmark tensorrt_cpp_api)