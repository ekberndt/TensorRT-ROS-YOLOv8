from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from environs import Env

def generate_launch_description():
    """
    Generates the launch description for the yolov8 package.

    Returns:
        LaunchDescription: The launch description object.
    """
    models_dir = os.path.join(get_package_share_directory('yolov8'), 'models')
    # Specify the specific model to use here
    # Get parameters from environment variables specifed in .env file
    env = Env()
    # TODO: Put yolov8.env in package share dir?
    env.read_env("yolov8.env")

    onnx_model_path = env("ONNX_MODEL")
    model_dir = os.path.join(models_dir, onnx_model_path)
    camera_topics = env.str("CAMERA_TOPICS").split(",")
    print(camera_topics)
    camera_buffer_hz = env.float("CAMERA_BUFFER_HZ")
    visualize_masks = env.bool("VISUALIZE_MASKS")
    enable_one_channel_mask = env.bool("ENABLE_ONE_CHANNEL_MASK")
    visualize_one_channel_mask = env.bool("VISUALIZE_ONE_CHANNEL_MASK")

    # Print all environment variables read in above
    # for key, value in env.items():
    #     print(f"{key}: {value}")
    print(f"model_dir: {model_dir}")
    print(f"camera_topics: {camera_topics}")
    print(f"camera_buffer_hz: {camera_buffer_hz}")
    print(f"visualize_masks: {visualize_masks}")
    print(f"enable_one_channel_mask: {enable_one_channel_mask}")
    print(f"visualize_one_channel_mask: {visualize_one_channel_mask}")

    # TODO Pull parameters out of ros_segmentation and place them here
    print(model_dir)
    return LaunchDescription([
        Node(
            package='yolov8',
            executable='ros_segmentation',
            parameters=[{
                'model_dir': model_dir,
                'camera_topics': camera_topics,
                'camera_buffer_hz': camera_buffer_hz,
                'visualize_masks': visualize_masks,
                'enable_one_channel_mask': enable_one_channel_mask,
                'visualize_one_channel_mask': visualize_one_channel_mask,
            }],
            arguments=[
                '--model', model_dir
            ],
        )
    ])
