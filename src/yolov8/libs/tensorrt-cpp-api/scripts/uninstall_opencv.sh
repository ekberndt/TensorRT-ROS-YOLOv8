# Enter the directory where this script is located
CURR_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd ${CURR_SCRIPT_DIR}

# Run the persisted opencv_version script to get the version of OpenCV that was last built by the script as an environment variable:
source opencv_version.sh

# Enter OPENCV_BUILD_DIR
echo "Uninstalling OPENCV_VERSION=$OPENCV_VERSION"
OPENCV_BUILD_DIR="opencv-$OPENCV_VERSION/build"
cd ${OPENCV_BUILD_DIR}

# Uninstall OpenCV with CUDA in OPENCV_BUILD_DIR (in the case you are causing conflicts with another version of OpenCV)
sudo make uninstall
