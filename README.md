# TensorRT YOLOv8 ROS Instance Segmentation

## Overview

A TensorRT ROS2 package for realtime instance segmentation in C++ using fine-tuned YOLOv8. This package is designed to be used with ROS2 Iron and Ubuntu 22.04.

This packages assumes you have already fine-tuned a YOLOv8 model and have the ONNX model file.

## ROS2 Messages

### yolov8/Image

This topic is a `sensor_msgs/Image` image and is used to visualize the output of the instance segmentation and bounding boxes in RViz2.

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

## Installation / Dependencies

- Made for Ubuntu 22.04 (Untested on other Distributions)
- Install ROS2 Iron
  - Since ROS2 Iron natively supports Ubuntu 22.04 you can install is as a Debian package as described [here](https://docs.ros.org/en/iron/Installation/Ubuntu-Install-Debians.html).
- Make sure you are **not** using a conda environment as it may cause issues with the ROS2 environment.
- Install CUDA 11.8 Toolkit via the runfile (local) [here](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) NOT the deb version.
  - _WARNING_: After installing the CUDA toolkit make sure you can run `nvidia-smi` in the CLI and that it doesn't error out. _If it errors out_, this may be because you installed the deb version of CUDA 11.8 rather than the runfile version or you forgot the `--toolkit` flag. This means you installed newer CUDA low level drivers in place of the original CUDA low level drivers and may black screen your computer if your computer is restarted. Open up the `Additional Drivers` tab in the `Software & Updates` application in Ubuntu and install the NVIDIA driver metapackage with the `(proprietary, tested)` tags (or whatever driver version your GPU needs).
- Download cuDNN 8.2.4 from [here](https://developer.nvidia.com/cudnn) and install it by following the instructions [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).
- Install TensorRT 8.6 GA
  - Download TensorRT 8.6 GA **TAR Package** (not the deb package) from [here](https://developer.nvidia.com/nvidia-tensorrt-8x-download), place / unpack the tar inside the `~/libs/` folder, and install the **full runtime** by following the instructions [here](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar).
  - _Note_: We use the TAR package because it allows us to install multiple versions of TensorRT on the same machine (and the CMAKE file in this package assumes the TAR package is used). The deb package does not allow this.
<!-- - Add the CUDA and TensorRT paths to the end of your `~/.bashrc` file by running the following commands:
    ```Bash
    echo '# CUDA TOOLKIT' >> ~/.bashrc
    echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo 'export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8' >> ~/.bashrc
    echo '' >> ~/.bashrc
    echo '# TensorRT' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=~/libs/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo '' >> ~/.bashrc
    source ~/.bashrc
    ``` -->
- Add the TensorRT path to the `LD_LIBRARY_PATH` environment variable in your `~/.bashrc` file so that the TensorRT shared libraries can be found by the ROS2 node.

    ```Bash
    echo '# TensorRT' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=~/libs/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    ```

- Build OpenCV 4.8.0 with CUDA support using the following
  - In the root directory of this package, run:

    ```Bash
    ./scripts/build_opencv.sh
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

## Errors

- If you get exit code `-9` while building / loading the TensorRT Engine you likely ran out of memory and may need to close other applications or use a different GPU / more memory.
- If you get an error about no rclpy module check to make sure you are **not** using a conda environment as it may cause issues with the ROS2 environment.
- If you get an error when you launch / run the ros node saying `libnvinfer.so.8: cannot open shared object file: No such file or directory`, make sure you added the TensorRT path to the `LD_LIBRARY_PATH` environment variable in your `~/.bashrc` file and sourced it as mentioned in [Installation / Dependencies](#installation--dependencies).

## Sources

This project uses code and directions from [YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP) licensed under the MIT license.
