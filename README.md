# TensorRT YOLOv8 ROS Instance Segmentation

## Overview

A TensorRT ROS2 package for realtime instance segmentation in C++ using fine-tuned YOLOv8. This package is designed to be used with ROS2 Iron and Ubuntu 22.04.

This packages assumes you have already fine-tuned a YOLOv8 model and have the ONNX model file.

## ROS2 Messages

### yolov8/camera_name/Image

> Note: This topic will only be published if the `visualizeMasks` parameter is set to `true`.

This topic is a `sensor_msgs/Image` image and is used to visualize the output of the instance segmentation and bounding boxes in RViz2.

- The `Header` (with time `stamp` and `frame_id`), `height`, `width`, `is_bigendian`, and `step` are taken from the original image topic that the mask is associated with and the `encoding` of the image which is always `bgr8`.

### yolov8/camera_name/detections

This topic publishes a custom `yolov8_interfaces/msg/Detection` message which contains the class label, class probability, class name, segmentation mask, and bounding box for each detected object. Everything in this message is in the same order as the detected objects in the image. For example, the second element in the `labels` list corresponds to the second binary mask in the `masks` list and the second bounding box in the `bounding_boxes` list.

- `index` - an int representing the index of the detected object (1 indexed since 0 is reserved for the background)
- `labels` - a list of class labels (int) for each detected object
- `probabilities` - a list of class probabilities (float range 0 - 1) for each detected object
- `class_names` - a list of human readable class names (string) for each detected object
- `seg_mask_one_channel` - a single channel segmentation mask for all detected segmentation masks combined into one image and in one color channel
  - Each mask (Image) is a 2D array where 0 represents the background and all other values represent the `index` of the detected object.
  - Published as a `sensor_msgs/Image` image
  - The `Header` (with time `stamp` and `frame_id`), `height`, `width`, `is_bigendian`, and `step` are taken from the original image topic that the mask is associated with and the `encoding` of the image which is always `mono8`.
  - This topic is useful for tasks like LiDAR point cloud segmentation where you want to segment the point cloud based on the projected detections from image space onto the point cloud.
  - `data` is the image itself
- `bounding_boxes` - a list of bounding boxes for each detected object
  - Defined as a custom message `yolov8_interfaces/msg/Yolov8BBox.msg`
  - `top_left` - a custom message `yolov8_interfaces/msg/Point2D.msg` representing the top left corner of the bounding box as int `x` and `y` coordinates
  - `rect_width` - the width of the bounding box as an int
  - `rect_height` - the height of the bounding box as an int

### yolov8/camera_name/seg_mask_one_channel

> Note: This topic will only be published if the `enableOneChannelMask` and `visualizeOneChannelMask` parameters are both set to `true`.

This topic is a `sensor_msgs/Image` image and is used to visualize the output oneChannelMask (all the segmentation masks combined into one image and in one color) in RViz2 / Foxglove Studio.

- The `Header` (with time `stamp` and `frame_id`), `height`, `width`, `is_bigendian`, and `step` are taken from the original image topic that the mask is associated with. The published image `encoding` is actually an `rgb8` converted from the original `mono8` `yolov8/camera_name/detections/seg_mask_one_channel` topic.
- The `rgb8` image is normalized using cv::NORMALIZE_MINMAX so the minimum value of the displayed image is 0 and the maximum value of the displayed image is 255 even if the original image only has 0 and 1 values. This means the labels will change colors drastically if the number of detected objects changes. This is done so the image can be visualized in RViz2 / Foxglove Studio.
- `data` is the image itself

## Installation / Dependencies

- Made for Ubuntu 22.04 (Untested on other Distributions)
- Install ROS2 Iron
  - Since ROS2 Iron natively supports Ubuntu 22.04 you can install is as a Debian package as described [here](https://docs.ros.org/en/iron/Installation/Ubuntu-Install-Debians.html).
- Make sure you are **not** using a conda environment as it may cause issues with the ROS2 environment.
- Install CUDA 11.8 Toolkit via the runfile (local) [here](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) NOT the deb version.
  - _WARNING_: After installing the CUDA toolkit make sure you can run `nvidia-smi` in the CLI and that it doesn't error out. _If it errors out_, this may be because you installed the deb version of CUDA 11.8 rather than the runfile version or you forgot the `--toolkit` flag. This means you installed newer CUDA low level drivers in place of the original CUDA low level drivers and may black screen your computer if your computer is restarted. Open up the `Additional Drivers` tab in the `Software & Updates` application in Ubuntu and install the NVIDIA driver metapackage with the `(proprietary, tested)` tags (or whatever driver version your GPU needs).
- Download cuDNN 8.2.4 from [here](https://developer.nvidia.com/cudnn) and install it by following the instructions [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).
- Install TensorRT. Note you can install TensorRT as a DEB or TAR pkg. The DEB package is easier to install but only allows one minor version of TensorRT to be installed at a time. The TAR package is more difficult to install but allows multiple versions of TensorRT to be installed at the same time. **You only have to install either the DEB or TAR package, not both.**
- Install TensorRT 10 GA (DEB Package)
  - Download TensorRT 10 **DEB Package** from [here](https://developer.nvidia.com/tensorrt/download/10x) by following [these instructions](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1010/install-guide/index.html#downloading). Install the **full C++ and Python runtimes**.
  - The DEB package is always installed in `/usr/src/tensorrt` and the shared libraries are in `/usr/lib/x86_64-linux-gnu/`. The CMAKE file in this package assumes the DEB package is used so you don't have to make any changes to the CMAKE files.
- Install TensorRT 8.6 GA (TAR Package)
  - Download TensorRT 8.6 GA **TAR Package** from [here](https://developer.nvidia.com/nvidia-tensorrt-8x-download), place / unpack the tar inside the `~/libs/` folder, and install the **full runtime** by following the instructions [here](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar).
  - Since we installed the TAR package in a custom location (~/libs/TensorRT-xxx) we need to update `src/yolov8/libs/tensorrt-cpp-api/CMakeLists.txt` to point to the correct location of the TensorRT shared libraries. Change the `set(TENSORRT_DIR "/usr/src/tensorrt")` line to `set(TensorRT_DIR $ENV{HOME}/libs/TensorRT-8.6.1.6/)` in the `CMakeLists.txt` file.

- Build OpenCV 4.8.0 with CUDA support using the following
  - In `scripts/install_opencv.sh` change the `CUDA_ARCH_BIN` variable to the correct CUDA architecture of your GPU. You can find the correct CUDA compute architecture for your GPU [here](https://developer.nvidia.com/cuda-gpus) by clicking the dropdowns.
  - In the root directory of this package, run:

    ```Bash
    ./scripts/install_opencv.sh
    ```

  - This script will build OpenCV 4.8.0 with CUDA support and install it in the `libs/tensorrt-cpp-api/scripts/` directory, which will take a while to build.
  - _Note_: This script will install a custom build of **OpenCV system-wide**, overwriting any existing OpenCV installation. This may break other projects that rely on a different version of OpenCV as across OpenCV versions functions are deprecated, removed, or changed. If this is a concern, you can uninstall this custom build of OpenCV with the following command but this package will no longer work until you reinstall the custom build of OpenCV.

    ```Bash
    ./scripts/uninstall_opencv.sh
    ```

## Building and Running TensorRT Node

First you must create a ROS2 workspace.

```Bash
mkdir -p ros_workspace/src
```

Then move the `yolov8` package (directory) and `yolov8_interfaces` package from the `src` directory of this project into the `src` directory of the ROS2 workspace you just created.

Then in the root directory of this `ros_workspace`, run the following commands to build the ROS2 package and source the setup file.

```Bash
# Builds the package with debug symbols so you can debug C++ code with GDB
make build-debug PACKAGES="yolov8 yolov8_interfaces" 
source install/setup.bash
```

You can also build normally with colcon and source the setup file.

```Bash
colcon build
source install/setup.bash
```

Then you can run the ROS2 node using the following command where `model.onnx` is the path to the ONNX model you want to use.

```Bash
ros2 run yolov8 ros_segmentation --model model.onnx
```

You can also run the ROS2 node using a launch file where the `arguments` should be modified to the path of the ONNX model you want to use.

```Bash
ros2 launch yolov8 yolov8.launch.py
```

_Note_: The first time you run the ROS node, it will take a while for TensorRT to build the engine model for your specific GPU. Subsequent runs will be able to load the serialized engine model and will be much faster.

## Tmuxp

You can also use the `tmuxp` package to quickly run the ROS2 node in a tmux session. First install `tmuxp` with the following command.

```Bash
sudo apt install tmuxp
```

There are two tmuxp configurations in the `tmuxp` directory.

- `rviz2_yolov8.yaml` - This configuration will run a configured RViz2, the ROS2 node, a `rosbag play` command, and a terminal to echo the topics.
- `foxglove_yolov8.yaml` - This configuration will run Foxglove Studio, the ROS2 node, a `rosbag play` command, and a terminal to echo the topics.

## Errors

In general if YOLOV8 is not working / crashing check the latest ros logs with the following command:

```Bash
cat ~/.ros/log/latest.log
```

A list of common errors and their solutions are below:

- Exit code `-9` while building / loading the TensorRT Engine
  - Your computer likely ran out of memory causing the OOM Killer to kill the program. Try closing other applications or using a different GPU / or a computer with more memory. Use `htop` to monitor memory usage and `nvidia-smi` to monitor GPU memory usage.
- CUDA initialization failure with error: 46
  - Make sure no other applications are taking control of the GPU. If you can't fix this issue, you can try restarting your computer.
- An error about no rclpy module check
  - Make sure you are **not** using a conda environment as it may cause issues with the ROS2 environment (ROS2 does not officially support conda environments).
- An error when you launch / run the ros node saying `libnvinfer.so.8: cannot open shared object file: No such file or directory`
  - _If you installed TensorRT as a TAR package_, make sure you added the TensorRT path to the `LD_LIBRARY_PATH` environment variable in your `~/.bashrc` file and sourced it as mentioned in [Installation / Dependencies](#installation--dependencies).
- `Error, cuda_runtime_api.h could not determine number of CUDA-capable devices. Restart your computer. Error thrown by cuda_runtime_api.h: unknown error`
  - This error seems to happen after the computer has been on for a long time and the GPU has been used a lot. Restarting the computer seems to fix this issue.

## Sources

This project uses code and directions from [YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP) licensed under the MIT license.

## TODO

- Fix RViz2 Config
- Fix Makefile build targets
- Bring in CUDA Tools
