cmake_minimum_required(VERSION 3.8)
project(tensorrt_cpp_api)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# ----------- Set TENSORT_DIR for TENSRORT TAR INSTALL HERE ----------
# Specify the path to TensorRT root dir for FindTensorRT.cmake to find the libraries
set(TensorRT_DIR $ENV{HOME}/libs/TensorRT-8.6.1.6/)
# set(TensorRT_DIR /usr/src/tensorrt/)
# ----------- Set TENSORT_DIR for TENSRORT TAR INSTALL HERE ----------

# Check if the path to TensorRT root dir exists
if (NOT EXISTS ${TensorRT_DIR})
    # These messages are saved in ./log/latest_build/yolov8/stdout_stderr.log
    message(FATAL_ERROR 
            "TensorRT_DIR: ${TensorRT_DIR} does not exist. "
            "Please set the path to TensorRT root directory or "
            "install TensorRT if you have not.")
    return()
endif()

# Set the version of CUDA to use
set(CUDA_VERSION "11.8")
# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-${CUDA_VERSION}")
# Set CMAKE_CUDA_COMPILER to the correct version of nvcc
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-${CUDA_VERSION}/bin/nvcc")

# We require CUDA, OpenCV, and TensorRT
# Uses FindTensorRT.cmake module to find the libraries
find_package(TensorRT REQUIRED)
if (NOT TensorRT_FOUND)
    message(FATAL_ERROR "TensorRT not found. Please make sure TensorRT is installed.")
    return()
endif()

message(STATUS "TensorRT Include: ${TensorRT_INCLUDE_DIRS}")
message(STATUS "TensorRT Libraries: ${TensorRT_LIBRARIES}")

find_package(CUDA REQUIRED)
if (NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. Please make sure CUDA is installed.")
    return()
endif()

message(STATUS "CUDA Version in tensorrt-cpp-api: ${CUDA_VERSION}")

find_package(OpenCV 4.8.0 REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please make sure OpenCV is installed.")
    return()
endif()

message(INFO "${LD_LIBRARY_PATH}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
# message(STATUS "OpenCV Directory: ${OpenCV_DIR}")
message(STATUS "OpenCV Include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV Lib Dir: ${OpenCV_LIB_DIR}")

# Link OpenCV directories to ensure CUDA libraries are found
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

add_library(tensorrt_cpp_api SHARED
        src/engine.cpp)

target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

add_executable(run_inference_benchmark src/main.cpp)
target_link_libraries(run_inference_benchmark tensorrt_cpp_api)