# TensorRT YOLOv8 ROS Instance Segmentation
## Overview
A TensorRT ROS2 package for realtime instance segmentation in C++ using finetuned YOLOv8. This package is designed to be used with ROS2 Iron and Ubuntu 22.04.

This packages assumes you have already finetuned a YoLOv8 model and have the ONNX model file.

## ROS2 Messages
### yolov8/Image
This topic is a `sensor_msgs/Image` image and is used to visualize the output of the intance segmentation and bounding boxes in RViz2.

### yolov8/detections
This topic publishes a custom `yolov8_interfaces/msg/Detection` message which contains the class label, class probability, class name, segmentation mask, and bounding box for each detected object. Everything in this message is in the same order as the detected objects in the image. For example, the second element in the `labels` list corresponds to the second binary mask in the `masks` list and the second bounding box in the `bounding_boxes` list.
- `labels` - a list of class labels (int) for each detected object
- `probabilities` - a list of class probabilities (float range 0 - 1) for each detected object
- `class_names` - a list of human readable class names (string) for each detected object
- `masks` - a list of binary masks for each detected object
    - Defined as a list of sensor_msgs/Image messages
    - Each mask (Image) is a 2D array of 0s and 1s where 1s represent the instance of the object and 0s represent the background
    - `Header` includes `frame_id` corresponding to the original image topic that the mask is associated with, the `height` and `width` of the mask, and the `encoding` of the image.
- `bounding_boxes` - a list of bounding boxes for each detected object
    - Defined as a custom message `yolov8_interfaces/msg/Yolov8BBox.msg`
    - `top_left` - a custom message `yolov8_interfaces/msg/Point2D.msg` representing the top left corner of the bounding box as int `x` and `y` coordinates
    - `rect_width` - the width of the bounding box as an int
    - `rect_height` - the height of the bounding box as an int

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