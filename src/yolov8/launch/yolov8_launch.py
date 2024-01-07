from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolov8',
            executable='ros_segmentation',
            name='yolov8',
            remappings=[
                ('/vimba_front_left_center/image', '/camera0/image')
            ],
            parameters=[
                {'--model': 'best.onnx'},
                {'--input': 'frame_017144_PNG.rf.cc8d12902369beac022e401776258747.jpg'}
            ]
        )
    ])