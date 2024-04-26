from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    models_dir = os.path.join(get_package_share_directory('yolov8'), 'models')
    # Specify the specific model to use here
    model_dir = os.path.join(models_dir, '164-164-v1-car-label.onnx')
    # TODO add parameter to turn off yolov8/image publisher and image drawing to save resources (--visualize)
    return LaunchDescription([
        Node(
            package='yolov8',
            executable='ros_segmentation',
            # name='yolov8',
            # remappings=[
                # ('/vimba_front_left_center/image', '/vimba_front_left_center/image')
            # ],
            arguments=[
                '--model', model_dir,
            ],
        )
    ])