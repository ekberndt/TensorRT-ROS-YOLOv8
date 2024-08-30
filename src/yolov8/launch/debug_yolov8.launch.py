'''
Generates the launch description for the yolov8 package with the niceness prefix removed
so that the debugger can attach to the process.
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

    onnx_model_path = os.environ.get("ONNX_MODEL")
    model_dir = os.path.join(models_dir, onnx_model_path)
    camera_topics = os.environ.get("CAMERA_TOPICS").split(",")
    camera_buffer_hz = float(os.environ.get("CAMERA_BUFFER_HZ"))
    visualize_masks = bool(os.environ.get("VISUALIZE_MASKS"))
    enable_one_channel_mask = bool(os.environ.get("ENABLE_ONE_CHANNEL_MASK"))
    visualize_one_channel_mask = bool(os.environ.get("VISUALIZE_ONE_CHANNEL_MASK"))
    nice_level = int(os.environ.get("NICE_LEVEL"))
    precision = os.environ.get("PRECISION")
    calibration_data_directory = os.environ.get("CALIBRATION_DATA_DIRECTORY")
    probability_threshold = float(os.environ.get("PROBABILITY_THRESHOLD"))
    nms_threshold = float(os.environ.get("NMS_THRESHOLD"))
    top_k = int(os.environ.get("TOP_K"))
    seg_channels = int(os.environ.get("SEG_CHANNELS"))
    seg_h = int(os.environ.get("SEG_H"))
    seg_w = int(os.environ.get("SEG_W"))
    segmentation_threshold = float(os.environ.get("SEGMENTATION_THRESHOLD"))
    # class_names = env("CLASS_NAMES").split(",")

    print("YOLOv8 Parameters:")
    print(f"onnx_model_path: {onnx_model_path}")
    print(f"model_dir: {model_dir}")
    print(f"camera_topics: {camera_topics}")
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
        )
    ])
