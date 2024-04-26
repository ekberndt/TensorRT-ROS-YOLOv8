# Find the bash script to build the CUDA version of OpenCV
CURR_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
YOLOV8_PKG_DIR="$CURR_SCRIPT_DIR/../"
OPENCV_SCRIPTS_DIR="$YOLOV8_PKG_DIR/libs/tensorrt-cpp-api/scripts"

# Enter OPENCV_SCRIPTS_DIR dir
cd ${OPENCV_SCRIPTS_DIR}

# Build OpenCV with CUDA in OPENCV_SCRIPTS_DIR
bash ${OPENCV_SCRIPTS_DIR}/build_opencv.sh
