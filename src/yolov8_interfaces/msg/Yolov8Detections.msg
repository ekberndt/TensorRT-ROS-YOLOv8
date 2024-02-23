# Segmentation Masks and Bounding Boxes for a single frame

# Class labels for each segmentation instance / bounding box
int32[] labels
float64[] probabilities
string[] class_names

# Array of binary masks for each segmentation instance
sensor_msgs/Image[] masks

# Array of bounding boxes for each segmentation instance
yolov8_interfaces/Yolov8BBox[] bounding_boxes
