#include "yolov8.h"
#include "cmd_line_util.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"

// Custom ROS2 message types where the names of the hpp files are snake_case
#include "yolov8_interfaces/msg/point2_d.hpp"
#include "yolov8_interfaces/msg/yolov8_detections.hpp"
#include "yolov8_interfaces/msg/yolov8_seg_mask.hpp"
#include "yolov8_interfaces/msg/yolov8_b_box.hpp"

using std::placeholders::_1;

class YoloV8Node : public rclcpp::Node
{
    public:
        YoloV8Node(YoloV8& yoloV8)
        : Node("yolo_v8"), yoloV8_(yoloV8)
        {
            subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/vimba_front_right_center/image", 10, std::bind(&YoloV8Node::camera_callback, this, _1));

            // rclcpp::Publisher::yolov8_interfaces::msg::YOLOv8Seg>::SharedPtr("/yolov8/dectections", 10);
            // rclcpp::Publisher::<sensor_msgs::msg::Image>::SharedPtr("/yolov8/Image", 10);

            // Do these need to be shared pointers?
            detection_publisher_ = this->create_publisher<yolov8_interfaces::msg::Yolov8Detections>("/yolov8/detections", 10);
            image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/yolov8/Image", 10);
        }

    private:
        void camera_callback(const sensor_msgs::msg::Image::ConstSharedPtr & image) const
        {
            cv_bridge::CvImagePtr cv_ptr;
            try
            {
                cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::RGB8); // Convert to RGB format
                // RCLCPP_INFO(this->get_logger(), "Image | Encoding: %s | Height: %d | Width %d | Step %ld", cv_ptr->encoding.c_str(),
                //     cv_ptr->image.rows, cv_ptr->image.cols, static_cast<long int>(cv_ptr->image.step));
            }
            catch (cv_bridge::Exception& e)
            {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }

            // Access the OpenCV image
            cv::Mat img = cv_ptr->image;
            // Convert from RGB8 to BGR8
            cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
            if (img.empty())
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to convert ROS image to OpenCV image");
                return;
            }

            try {
                // Run inference
                const auto objects = yoloV8_.detectObjects(img);

                // RCLCPP_INFO(this->get_logger(), "Typeid: %s", typeid(objects[0].boxMask).name());
                // RCLCPP_INFO(this->get_logger(), "Mask Shape: %s", objects[0].boxMask.size());

                // Create vector to store binary masks for each segmentation object
                std::vector<cv::Mat> detectedMasks;

                // Draw the bounding boxes on the image and store the binary masks
                yoloV8_.drawObjectLabels(img, objects, detectedMasks);
                RCLCPP_INFO(this->get_logger(), "Detected %zu objects", objects.size());
                RCLCPP_INFO(this->get_logger(), "Detected %zu masks", detectedMasks.size());

                // Convert detected objects to ROS message
                yolov8_interfaces::msg::Yolov8Detections detectionMsg;
                for (auto & object : objects) {
                    int label = object.label;
                    float prob = object.probability;

                    // Place each binary mask into the ROS message
                    sensor_msgs::msg::Image binaryMaskMsg;
                    cv_bridge::CvImagePtr cv_image = std::make_shared<cv_bridge::CvImage>();
                    if (!detectedMasks.empty()) {
                        cv_image->image = detectedMasks.back();
                        detectedMasks.pop_back();
                    } else {
                        RCLCPP_ERROR(this->get_logger(), "detectedMasks vector is empty");
                    }
                    cv_image->encoding = "bgr8";
                    cv_image->header.frame_id = image->header.frame_id;
                    // Turn cv_image into sensor_msgs::msg::Image
                    cv_image->toImageMsg(binaryMaskMsg);

                    // Place the contours into the message
                    // for (auto & points : object.contours) {
                    //     yolov8_interfaces::msg::Point2D point2DMsg;
                    //     // TODO: Check if this is the correct way to access the x and y values
                    //     for 
                    //     point2DMsg.x = point.x;
                    //     point2DMsg.y = point.y;
                    //     segMaskMsg.contours.push_back(point2DMsg);
                    // }

                    // Create yolov8_obj bounding box message
                    yolov8_interfaces::msg::Yolov8BBox bBoxMsg;
                    yolov8_interfaces::msg::Point2D point2DMsg;
                    point2DMsg.x = object.rect.x;
                    point2DMsg.y = object.rect.y;
                    bBoxMsg.top_left = point2DMsg;
                    bBoxMsg.rect_width = object.rect.width;
                    bBoxMsg.rect_height = object.rect.height;

                    // Add segmentation masks, bounding boxes, and class info to yolov8detections message
                    detectionMsg.labels.push_back(label);
                    detectionMsg.probabilities.push_back(prob);
                    detectionMsg.class_names.push_back(yoloV8_.getClassName(label));
                    detectionMsg.masks.push_back(binaryMaskMsg);
                    detectionMsg.bounding_boxes.push_back(bBoxMsg);
                }

                // Publish segmented image as ROS message
                sensor_msgs::msg::Image displayImageMsg;

                cv_bridge::CvImagePtr cv_image = std::make_shared<cv_bridge::CvImage>();
                cv_image->image = img.clone(); // TODO: Change this to mask? Too many copies?
                cv_image->encoding = "bgr8";
                cv_image->header.frame_id = image->header.frame_id;
                // Turn cv_image into sensor_msgs::msg::Image
                cv_image->toImageMsg(displayImageMsg);
                displayImageMsg.header.stamp = rclcpp::Time(image->header.stamp.sec, image->header.stamp.nanosec);

                // Publish the messages
                detection_publisher_->publish(detectionMsg);
                image_publisher_->publish(displayImageMsg);
                
            } catch (cv::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv exception: %s", e.what());
                return;
            }
        }
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
        rclcpp::Publisher<yolov8_interfaces::msg::Yolov8Detections>::SharedPtr detection_publisher_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
        YoloV8& yoloV8_;
        };

int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string inputImage;

    // Parse the command line arguments
    // TODO: Change this to use ROS parameters
    if (!parseArguments(argc, argv, config, onnxModelPath, inputImage)) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to parse command line arguments");
        return -1;
    }

    // Create the YoloV8 engine
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Creating YoloV8 engine --- Could take a while if Engine file is not already built and cached.");
    // TODO: See why this sometimes maxes out the memory and causes program to get SIGKILLed (exit code -6)
    YoloV8 yoloV8(onnxModelPath, config);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "YoloV8 engine created and initialized.");
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloV8Node>(yoloV8));
    rclcpp::shutdown();
    return 0;
}