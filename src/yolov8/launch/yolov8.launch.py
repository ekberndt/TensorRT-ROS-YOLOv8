from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolov8',
            executable='ros_segmentation',
            # name='yolov8',
            # remappings=[
                # ('/vimba_front_left_center/image', '/vimba_front_left_center/image')
            # ],
            arguments=[
                '--model', '/home/ekberndt/Documents/iac/race_common_yolov8/src/perception/yolov8/models/164-164-v1-car-label.onnx',
                '--input', '/home/ekberndt/Documents/iac/race_common_yolov8/test_putnum.jpg',
            ]
        )
    ])