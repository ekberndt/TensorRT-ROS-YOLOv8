# TensorRT YOLOv8 ROS Instance Segmentation
## Overview
A TensorRT ROS2 package for realtime instance segmentation in C++ using finetuned YOLOv8. This package is designed to be used with ROS2 Iron and Ubuntu 22.04.

## Installation
### Dependencies
- Made for Ubuntu 22.04 (Untested on other Distributions)
- Install ROS2 Iron
    - Since ROS2 Iron natively supports Ubuntu 22.04 you can install is as a Debian package as described [here](https://docs.ros.org/en/iron/Installation/Ubuntu-Install-Debians.html).
- Make sure you are **not** using a conda environment as it may cause issues with the ROS2 environment.
- Install TensorRT 8.6 GA
    - Download TensorRT 8.6 GA from [here](https://developer.nvidia.com/nvidia-tensorrt-8x-download) and install the **full runtime** by following the instructions [here](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian).
    - Note you should have already installed CUDA 11.8 toolkit and cuDNN in the conda environment, so you do not have to reinstall them.

## Building and Running TensorRT Node
First you must create a ROS2 workspace.
```
mkdir -p ros_workspace/src
```
Then move the `yolov8` package (directory) and `yolov8_interfaces` package from the `src` directory of this project into the `src` directory of the ROS2 workspace you just created.

Then in the root directory of this `ros_workspace`, run the following commands to build the ROS2 package and source the setup file.
```
# Builds the package with debug symbols so you can debug C++ code with GDB
make build-debug PACKAGES="yolov8 yolov8_interfaces" 
source install/setup.bash
```
You can also build normally with colcon and source the setup file.
```
colcon build
source install/setup.bash
```
Then you can run the ROS2 node using the following command where `model.onnx` is the path to the ONNX model you want to use and `image.jpg` is a placeholder image that does nothing - TODO: remove this 
```
ros2 run yolov8 ros_segmentation --model model.onnx --input image.jpg
```
You can also run the ROS2 node using a launch file where the `arguments` should be modified to the path of the ONNX model you want to use.
```
ros2 launch yolov8 yolov8.launch.py
```
_Note_: The first time you run the ROS node, it will take a while for TensorRT to build the engine model for your specific GPU. Subsequent runs will be able to load the serialized engine model and will be much faster.
 
# Errors
- If you get exit code `-9` while building / loading the TensorRT Engine you likely ran out of memory and may need to close other applications or use a different GPU / more memory.
- If you get an error about no rclpy module check to make sure you are **not** using a conda environment as it may cause issues with the ROS2 environment.

## Sources
This project uses code and directions from [YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP) licensed under the MIT license.