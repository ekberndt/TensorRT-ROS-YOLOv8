# Segmentation Masks and Bounding Boxes for a single frame

std_msgs/Header header
# Represents a single segmentation instance (1-indexed so that 0 can be used for background)
int32[] indexes
# Class labels for each segmentation instance / bounding box
int32[] labels
float64[] probabilities
string[] class_names

# Array of binary masks for each segmentation instance
# sensor_msgs/Image[] masks

# One Channel Image Containing all the Segmentation Masks (1-indexed so
# that 0 can be used for background)
# The header is the same as the original image from the camera (e.g. timestamp, frame_id
# except for the encoding which is mono8)
sensor_msgs/Image seg_mask_one_channel

# Array of bounding boxes for each segmentation instance
yolov8_interfaces/Yolov8BBox[] bounding_boxes
