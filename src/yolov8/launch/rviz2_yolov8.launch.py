from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """
    Generate the launch description for launching RViz2 with a yolov8 configuration file.

    Returns:
        LaunchDescription: The launch description object.
    """
    rviz2_config_path = os.path.join(
        get_package_share_directory('yolov8'),
        'config',
        'yolov8.rviz'
    )

    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz2_config_path],
            output='screen'
        )
    ])