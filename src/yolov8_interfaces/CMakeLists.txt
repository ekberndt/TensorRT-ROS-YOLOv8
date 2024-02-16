cmake_minimum_required(VERSION 3.8)
project(yolov8_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# There are no capital letters in the msg name besides the first letter
# since the message name is used to generate the hpp file name
# and CMake will use snake case for the file name
set(msg_files
  "msg/Point2D.msg"
  "msg/Yolov8Detections.msg"
  "msg/Yolov8BBox.msg"
  "msg/Yolov8SegMask.msg"
)

# Include msg files
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs sensor_msgs
)

# Include the generated header files for the message types
include_directories(${CMAKE_CURRENT_BINARY_DIR}/rosidl_typesupport_cpp/include)

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

ament_export_dependencies(rosidl_default_runtime)

ament_package()
