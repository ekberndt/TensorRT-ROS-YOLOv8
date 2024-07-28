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
        YoloV8Node(YoloV8& yoloV8, const std::vector<std::string>& camera_topics, float buffer_hz)
        : Node("yolo_v8"), yoloV8_(yoloV8)
        {
            camera_topics_ = camera_topics;
            std::unordered_map<std::string, rclcpp::Publisher<yolov8_interfaces::msg::Yolov8Detectioons>::SharedPtr> detection_publishers_;
            std::unordered_map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> image_publishers_;
            std::unordered_map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> one_channel_mask_publishers_;
            std::unordered_map<std::string, sensor_msgs::msg::Image> camera_buffer_;

            // Create timer for camera synchronization
            float buffer_duration = 1 / buffer_hz;
            ros::Timer buffer_timer_ = this->create_wall_timer(ros::Duration(buffer_duration), std::bind(&YoloV8Node::batch_buffer, this));

            // Create subscribers and publishers for all cameras
            for (const std::string& topic : camera_topics) {
                subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
                    topic + "/image", 10, std::bind(&YoloV8Node::addToBufferCallback, this, _1, topic)
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
        /*
        * Add the image message from the given image topic to the camera buffer. Note the preprocessing not done here
        * since a given image might be replaced by a newer image before the camera buffer is batched to the
        * neural network.
        *
        * @param image_msg: The image message from the camera
        * @param topic: The ROS topic the image message was published on
        */
        void addToBufferCallback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg, std::string topic) const {
            // TODO: Will this be deleted in memory since it is passed?
            camera_buffer_[topic] = image_msg;

            // Check if ready for batching
            if (camera_buffer_.size() == camera_topic_.size()) {
                batchBufferCallback(image_msg);
            }
        }

        void batchBufferCallback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg) {
            // Reset buffer timer
            buffer_timer_->reset();
            // TODO: Does the camera buffer need to be copied?
            // Preprocess the input
            std::vector<cv::Mat> images;
            preprocess_callback(&images);

            // TODO: Check if batch is empty -> if so return and throw warning

            // TODO: batch to network properly
            // Run inference
            const auto objects = yoloV8_.detectObjects(image);
            RCLCPP_INFO(this->get_logger(), "Detected %zu objects", objects.size());
            // RCLCPP_INFO(this->get_logger(), "Typeid: %s", typeid(objects[0].boxMask).name());
            // RCLCPP_INFO(this->get_logger(), "Mask Shape: %s", objects[0].boxMask.size());

            // Clear the camera buffer
            camera_buffer.clear();

            // Postprocess the output and publish the results
            postprocess_callback(objects, image);
        }

        /*
        * Preprocess input(s) for Neural Network by converting ROS image(s) to OpenCV image(s)
        * and converting from RGB8 to BGR8
        *
        * @param images: the vector to store the preprocessed images
        * @returns The preprocessed image(s) for the neural network
        */
        cv::Mat preprocess_callback(std::vector<cv::Mat>& images) const {
            for (const auto& pair : camera_buffer_) {
                std::string topic = pair.first;
                image::msg::Image::ConstSharedPtr image_msg = pair.second;
                try
                    {
                        // Share the memory with the original image
                        // TODO: Should this be a copy to deal with a cleared buffer?
                        cv_bridge::CvImagePtr cv_ptr;
                        cv_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::RGB8);

                        // Convert from RGB8 to BGR8
                        cv::Mat img = cv_ptr->image;
                        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

                        images.push_back(img);
                    } catch (cv_bridge::Exception& e) {
                        RCLCPP_ERROR(this->get_logger(), "Failed to convert ROS image message on topic %s \
                            due to cv_bridge error: %s", image_msg, e.what());
                        continue;
                    }
            }
        }

        /*
        * Convert output(s) from Neural Network to ROS messages and publish them
        *
        * @param objects: Detected objects from the neural network
        * @param image: Original ROS image from the camera
        */
        void postprocess_callback(std::vector<Object> objects, const sensor_msgs::msg::Image::ConstSharedPtr& image) const {
            // TODO: Make these flags ROS parameters
            bool visualizeMasks = true;
            bool enableOneChannelMask = true;
            bool visualizeOneChannelMask = true;

            // ROS message to publish the detections
            yolov8_interfaces::msg::Yolov8Detections detectionMsg;
            detectionMsg.header = image->header;

            if (enableOneChannelMask) {
                publishOneChannelMask(objects, visualizeOneChannelMask, image);
            }

            // Draw the segmentation masks and bounding boxes on the image
            // to visualize the detections
            if (visualizeMasks) {
                // Draw the object labels on the image
                yoloV8_.drawObjectLabels(img, objects);

                // Turn cv_image into sensor_msgs::msg::Image
                sensor_msgs::msg::Image displayImageMsg;
                // TODO: Change to rgb8?
                cv_bridge::CvImagePtr cv_image = std::make_shared<cv_bridge::CvImage>(
                    image->header, "bgr8", img
                );
                cv_image->image = img;
                cv_image->encoding = "bgr8";
                cv_image->header.frame_id = image->header.frame_id;
                cv_image->toImageMsg(displayImageMsg);
                displayImageMsg.header = image->header;
                // Publish segmented image as ROS message
                image_publisher_->publish(displayImageMsg);
            }

            // Convert detected objects to ROS message
            addObjectsToDetectionMsg(objects, detectionMsg);

            // Publish the detections
            detection_publisher_->publish(detectionMsg);
        }

        /*
        * Create a one channel mask for all segmentation objects and publish it. Optionally visualize the mask.
        *
        * @param objects: Detected objects from the neural network
        * @param visualizeOneChannelMask: Whether to visualize the one channel mask
        * @param image_msg: Original ROS image from the camera
        */
        void publishOneChannelMask(std::vector<Object> objects, bool visualizeOneChannelMask, const sensor_msgs::msg::Image::ConstSharedPtr& image_msg) const {
            cv::Mat oneChannelMask;
            int img_width = image_msg->width;
            int img_height = image_msg->height;
            yoloV8_.getOneChannelSegmentationMask(objects, oneChannelMask, img_width, img_height);
            // Use ROS cv_bridge to convert cv::Mat to sensor_msgs::msg::Image and take header from original camera image
            try {
                cv_bridge::CvImage cvBridgeOneChannelMask = cv_bridge::CvImage();
                cvBridgeOneChannelMask.image = oneChannelMask;
                detectionMsg.seg_mask_one_channel = *cvBridgeOneChannelMask.toImageMsg();
                detectionMsg.seg_mask_one_channel.header = image_msg->header;
                detectionMsg.seg_mask_one_channel.encoding = "mono8";
            } catch (cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }

            // Publish the one channel mask with RGB color for visualization only
            if (visualizeOneChannelMask) {
                cv::Mat oneChannelMaskRGB8 = visualizeOneChannelMask(oneChannelMask);
                try {
                    // Publish the 3-channel RGB mask as a ROS message
                    cv_bridge::CvImage cvBridgeOneChannelMaskRGB8 = cv_bridge::CvImage();
                    cvBridgeOneChannelMaskRGB8.image = oneChannelMaskRGB8;
                    cvBridgeOneChannelMaskRGB8.header = image_msg->header;
                    cvBridgeOneChannelMaskRGB8.encoding = "rgb8";
                } catch (cv_bridge::Exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                    return;
                }

                oneChannelMask_publisher_->publish(*cvBridgeOneChannelMaskRGB8.toImageMsg());
            }
        }

        /*
        * Add detected objects to the yolov8detections message
        *
        * @param objects: Detected objects from the neural network
        * @param detectionMsg: The yolov8detections message to add the objects to
        */
        void addObjectsToDetectionMsg(auto objects, auto detectionMsg) {
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
                detectionMsg.bounding_boxes.push_back(bBoxMsg);

                index++;
            }
        }

        /*
        * Create visualization of the one channel mask as an RGB image
        *
        * @param oneChannelMask: The one channel mask to visualize
        * @return The one channel mask as a RGB image
        */
        cv::Mat visualizeOneChannelMask(auto oneChannelMask) {
            try {
                // Draw the one channel mask on the image
                // Convert the one channel mask to 8-bit
                cv::Mat oneChannelMask8U;
                oneChannelMask.convertTo(oneChannelMask8U, CV_8U);
                // Convert the one channel mask to 3-channel RGB
                cv::Mat oneChannelMaskRGB8;
                cv::cvtColor(oneChannelMask8U, oneChannelMaskRGB8, cv::COLOR_GRAY2RGB);
                // Normalize the one channel mask to 0-255 so it can be displayed as an RGB image
                cv::normalize(oneChannelMaskRGB8, oneChannelMaskRGB8, 0, 255, cv::NORM_MINMAX);
            } catch (cv::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "OpenCV exception: %s", e.what());
                return;
            }
            return oneChannelMaskRGB8;
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