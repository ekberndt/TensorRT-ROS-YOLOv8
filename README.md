# TensorRT YOLOv8 ROS Instance Segmentation
## Overview
A TensorRT ROS2 package for realtime instance segmentation in C++ and associated fine-tuning pipeline using YOLOv8. This package is designed to be used with ROS2 Iron and Ubuntu 22.04.

## Installation
### Dependencies
- Made for Ubuntu 22.04 (Untested on other Distributions)
- Install ROS2 Iron
    - Since ROS2 Iron natively supports Ubuntu 22.04 you can install is as a Debian package as described [here](https://docs.ros.org/en/iron/Installation/Ubuntu-Install-Debians.html).
- Create a custom conda environment
    - Create a conda environment using the following command:
        ```bash
        <!-- conda create -n yolov8_trt python=3.8 -->
        conda env create -n yolov8_trt python=3.8 -f environment.yaml 
        ```
    - Activate the conda environment using the following command:
        ```bash
        conda activate yolov8_trt
        ```
    - Inside the yolov8 conda environment, install the conda dependencies using the following command **in the root directory of this project**:
        ```bash
        pip install -r requirements.txt
        ```
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
colcon build
source install/setup.bash
```
Then you can run the ROS2 node using the following command where `model.onnx` is the path to the ONNX model you want to use and `image.jpg` is a placeholder image that does nothing - TODO: remove this 
```
ros2 run yolov8 ros_segmentation --model model.onnx --input image.jpg
```

_Note_: The first time you run the ROS node, it will take a while for TensorRT to build the engine model for your specific GPU. Subsequent runs will be able to load the serialized engine model and will be much faster.
 



## Sources
This project uses code and directions from [YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP) licensed under the MIT license.