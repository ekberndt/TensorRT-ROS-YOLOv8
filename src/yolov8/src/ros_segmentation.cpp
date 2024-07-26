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
        : Node("yolo_v8"), yoloV8_(yoloV8, const std::vector<std::string>& camera_topics, float buffer_hz)
        {
            std::unordered_map<std::string, rclcpp::Publisher<yolov8_interfaces::msg::Yolov8Detectioons>::SharedPtr> detection_publishers_;
            std::unordered_map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> image_publishers_;
            std::unordered_map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> one_channel_mask_publishers_;
            std::unordered_map<std::string, sensor_msgs::msg::Image> camera_buffer_;
            float buffer_duration = 1 / buffer_hz;
            ros::Timer buffer_timer_ = this0=->create_wall_timer(ros::Duration(buffer_duration), std::bimd(&YoloV8Node::batch_buffer, this));
            buffer_timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&YoloV8Node::batch_buffer_callback, this));
            camera_topics_ = camera_topics;

            // Create subscribers and publishers for all cameras
            for (const std::string& topic : camera_topics) {
                subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
                    topic + "/image", 10, std::bind(&YoloV8Node::add_to_buffer_callback, this, _1)
                );
                subscriptions_.push_back(subscription_);
                detection_publishers_[topic] = this->create_publisher<yolov8_interfaces::msg::Image>(
                    "/yolov8" + topic + "/detections", 10
                )
                image_publishers_[topic] = this->create_publisher<sensor_msgs::msg::Image>(
                    "/yolov8" + topic + "/image", 10
                )
                one_channel_mask_publishers[topic] = this->create_publisher<sensor_msgs::msg::Image>(
                    "/yolov8" + topic + "seg_mask_one_channel", 10
                )
            }
        }

    private:
        void add_to_buffer_callback(const sensor::msg::Image::ConstSharedPtr& image) const {
            
            // TODO

            // Check if ready for batching
            if (camera_buffer_.size() == camera_topic_.size()) {
                batch_buffer_callback();
            }
        }

        void batch_buffer_callback() {
            // Reset buffer timer
            buffer_timer_->reset();
            for (auto & camera : camera_buffer) {
                // TODO: Change to process all images in the buffer
                // camera_callback(camera.second);
            }

            // TODO: Check if batch is empty -> if so return

            // TODO: batch to network

            camera_buffer.clear();
        }

        void camera_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image) const
        {
            // Convert ROS image to OpenCV image
            cv_bridge::CvImagePtr cv_ptr;
            try
            {
                cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::RGB8);
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

                // ROS message to publish the detections
                yolov8_interfaces::msg::Yolov8Detections detectionMsg;
                detectionMsg.header = image->header;

                // TODO: Make these flags ROS parameters
                bool visualizeMasks = true;
                bool enableOneChannelMask = true;
                bool visualizeOneChannelMask = true;
                
                if (enableOneChannelMask) {
                    cv::Mat oneChannelMask;
                    yoloV8_.getOneChannelSegmentationMask(img, objects, oneChannelMask);
                    // Use ROS cv_bridge to convert cv::Mat to sensor_msgs::msg::Image and take header from original camera image
                    cv_bridge::CvImage cvBridgeOneChannelMask = cv_bridge::CvImage();
                    cvBridgeOneChannelMask.image = oneChannelMask;
                    detectionMsg.seg_mask_one_channel = *cvBridgeOneChannelMask.toImageMsg();
                    detectionMsg.seg_mask_one_channel.header = image->header;
                    detectionMsg.seg_mask_one_channel.encoding = "mono8";

                    // Publish the one channel mask with RGB color for visualization only
                    if (visualizeOneChannelMask) {
                        // Draw the one channel mask on the image
                        // Convert the one channel mask to 8-bit
                        cv::Mat oneChannelMask8U;
                        oneChannelMask.convertTo(oneChannelMask8U, CV_8U);

                        // Convert the one channel mask to 3-channel RGB
                        cv::Mat oneChannelMaskRGB8;
                        cv::cvtColor(oneChannelMask8U, oneChannelMaskRGB8, cv::COLOR_GRAY2RGB);

                        // Normalize the one channel mask to 0-255 so it can be displayed as an RGB image
                        cv::normalize(oneChannelMaskRGB8, oneChannelMaskRGB8, 0, 255, cv::NORM_MINMAX);

                        // Publish the 3-channel RGB mask as a ROS message
                        cv_bridge::CvImage cvBridgeOneChannelMaskRGB8 = cv_bridge::CvImage();
                        cvBridgeOneChannelMaskRGB8.image = oneChannelMaskRGB8;
                        cvBridgeOneChannelMaskRGB8.header = image->header;
                        cvBridgeOneChannelMaskRGB8.encoding = "rgb8";
                        oneChannelMask_publisher_->publish(*cvBridgeOneChannelMaskRGB8.toImageMsg());
                    }
                }

                // Draw the segmentation masks and bounding boxes on the image
                // to visualize the detections
                if (visualizeMasks) {
                    // yoloV8_.drawObjectLabels(img, objects, detectedMasks);
                    yoloV8_.drawObjectLabels(img, objects);
                }

                RCLCPP_INFO(this->get_logger(), "Detected %zu objects", objects.size());
                // RCLCPP_INFO(this->get_logger(), "Detected %zu masks", detectedMasks.size());
                
                // TODO: Check if this is correct
                // detectionMsg.header.frame_id = image->header.frame_id; // TODO: Check if this is correct
                // detectionMsg.header.stamp = rclcpp::Time(image->header.stamp.sec, image->header.stamp.nanosec); // TODO: Check if this is correct

                // Convert detected objects to ROS message
                // Start at index 1 because index 0 is the background class
                int index = 1;
                for (auto & object : objects) {
                    int label = object.label;
                    float prob = object.probability;

                    // Create yolov8_obj bounding box message
                    yolov8_interfaces::msg::Yolov8BBox bBoxMsg;
                    yolov8_interfaces::msg::Point2D point2DMsg;
                    point2DMsg.x = object.rect.x;
                    point2DMsg.y = object.rect.y;
                    bBoxMsg.top_left = point2DMsg;
                    bBoxMsg.rect_width = object.rect.width;
                    bBoxMsg.rect_height = object.rect.height;

                    // Add segmentation masks, bounding boxes, and class info to yolov8detections message
                    detectionMsg.indexes.push_back(index);
                    detectionMsg.labels.push_back(label);
                    detectionMsg.probabilities.push_back(prob);
                    detectionMsg.class_names.push_back(yoloV8_.getClassName(label));
                    // detectionMsg.masks.push_back(binaryMaskMsg);
                    detectionMsg.bounding_boxes.push_back(bBoxMsg);

                    index++;
                }

                // Publish segmented image as ROS message
                sensor_msgs::msg::Image displayImageMsg;

                // TODO: Change to rgb8?
                cv_bridge::CvImagePtr cv_image = std::make_shared<cv_bridge::CvImage>();
                cv_image->image = img.clone(); // TODO: Change this to mask? Too many copies?
                cv_image->encoding = "bgr8";
                cv_image->header.frame_id = image->header.frame_id;
                // Turn cv_image into sensor_msgs::msg::Image
                cv_image->toImageMsg(displayImageMsg);
                displayImageMsg.header = image->header;

                // Publish the messages
                detection_publisher_->publish(detectionMsg);
                image_publisher_->publish(displayImageMsg);
                
            } catch (cv::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv exception: %s", e.what());
                return;
            }
        }
        std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subscriptions_;
        std::unordered_map<rclcpp::Publisher<yolov8_interfaces::msg::Yolov8Detections>::SharedPtr detection_publisher_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr oneChannelMask_publisher_;
        YoloV8& yoloV8_;
        };

int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;

    // Parse the command line arguments
    // TODO: Change this to use ROS parameters
    std::string parseArgsError = parseArguments(argc, argv, config, onnxModelPath);
    if (!parseArgsError.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "%s", parseArgsError.c_str());
        return -1;
    }
    // Create the YoloV8 engine
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Creating YoloV8 engine --- Could take a while if Engine file is not already built and cached.");
    YoloV8 yoloV8(onnxModelPath, config);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "YoloV8 engine created and initialized.");

    // Create ROS2 Node
    // TODO: Don't hard code params and instead use ROS params
    std::vector<std::string> camera_topics = {
        "/vimba_front",
        "/vimba_front_left_center",
        "/vimba_front_right_center",
        "/vimba_left",
        "/vimba_right",
        "/vimba_rear"
    };
    float buffer_hz = 30;

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloV8Node>(yoloV8, camera_topics, buffer_hz));
    rclcpp::shutdown();
    return 0;
}