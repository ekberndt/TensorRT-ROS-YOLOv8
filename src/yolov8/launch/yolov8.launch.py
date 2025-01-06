'''
Generates the launch description for the yolov8 package.
'''
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
    package_share_dir = get_package_share_directory('yolov8')
    models_dir = os.path.join(package_share_dir, 'models')
    env_file_path = os.path.join(package_share_dir, 'yolov8.env')
    # Get parameters from environment variables specifed in .env file
    # Check if .env file exists
    if not os.path.exists(env_file_path):
        raise FileNotFoundError("Please create a yolov8.env file in the root of the yolov8 \
                                workspace and rebuild the package so it can be installed in the \
                                package's share directory.")

    env = Env()
    env.read_env(env_file_path)

    onnx_model_path = env("ONNX_MODEL")
    model_dir = os.path.join(models_dir, onnx_model_path)
    camera_topics = env.str("CAMERA_TOPICS").split(",")
    camera_topic_suffix = env("CAMERA_TOPIC_SUFFIX")
    camera_buffer_hz = env.float("CAMERA_BUFFER_HZ")
    visualize_masks = env.bool("VISUALIZE_MASKS")
    enable_one_channel_mask = env.bool("ENABLE_ONE_CHANNEL_MASK")
    visualize_one_channel_mask = env.bool("VISUALIZE_ONE_CHANNEL_MASK")
    nice_level = env.int("NICE_LEVEL")
    precision = env("PRECISION")
    calibration_data_directory = env("CALIBRATION_DATA_DIRECTORY")
    probability_threshold = env.float("PROBABILITY_THRESHOLD")
    nms_threshold = env.float("NMS_THRESHOLD")
    top_k = env.int("TOP_K")
    seg_channels = env.int("SEG_CHANNELS")
    seg_h = env.int("SEG_H")
    seg_w = env.int("SEG_W")
    segmentation_threshold = env.float("SEGMENTATION_THRESHOLD")
    # class_names = env("CLASS_NAMES").split(",")

    print("YOLOv8 Parameters:")
    print(f"onnx_model_path: {onnx_model_path}")
    print(f"model_dir: {model_dir}")
    print(f"camera_topics: {camera_topics}")
    print(f"camera_topic_suffix: {camera_topic_suffix}")
    print(f"camera_buffer_hz: {camera_buffer_hz}")
    print(f"visualize_masks: {visualize_masks}")
    print(f"enable_one_channel_mask: {enable_one_channel_mask}")
    print(f"visualize_one_channel_mask: {visualize_one_channel_mask}")
    print(f"nice_level: {nice_level}")
    print(f"precision: {precision}")
    print(f"calibration_data_directory: {calibration_data_directory}")
    print(f"probability_threshold: {probability_threshold}")
    print(f"nms_threshold: {nms_threshold}")
    print(f"top_k: {top_k}")
    print(f"seg_channels: {seg_channels}")
    print(f"seg_h: {seg_h}")
    print(f"seg_w: {seg_w}")
    print(f"segmentation_threshold: {segmentation_threshold}")
    # print(f"class_names: {class_names}")

    # Convert the list of class names into several strings separated by "" to be read as a command
    # line argument
    # class_names = ' '.join([f'"{class_name}"' for class_name in class_names])

    # TODO Pull parameters out of ros_segmentation and place them here
    print(model_dir)
    return LaunchDescription([
        Node(
            package='yolov8',
            executable='ros_segmentation',
            parameters=[{
                'model_dir': model_dir,
                'camera_topics': camera_topics,
                'camera_topic_suffix': camera_topic_suffix,
                'camera_buffer_hz': camera_buffer_hz,
                'visualize_masks': visualize_masks,
                'enable_one_channel_mask': enable_one_channel_mask,
                'visualize_one_channel_mask': visualize_one_channel_mask,
            }],
            arguments=[
                '--model', model_dir,
                '--precision', precision,
                '--calibration-data', calibration_data_directory,
                '--prob-threshold', str(probability_threshold),
                '--nms-threshold', str(nms_threshold),
                '--top-k', str(top_k),
                '--seg-channels', str(seg_channels),
                '--seg-h', str(seg_h),
                '--seg-w', str(seg_w),
                '--seg-threshold', str(segmentation_threshold),
                # '--class-names', class_names
            ],
            prefix=['nice -n ' + str(nice_level)]
        )
    ])
