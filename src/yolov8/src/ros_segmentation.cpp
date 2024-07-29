# include <mutex>

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

            // Create timer for camera synchronization
            float buffer_duration = 1 / buffer_hz;
            buffer_timer_ = rclcpp::create_wall_timer(
                std::chrono::duration<float>(buffer_duration),
                std::bind(&YoloV8Node::batchBufferCallback, this),
                nullptr,
                this->get_node_base_interface().get(),
                this->get_node_timers_interface().get()
            );

            // Create subscribers and publishers for all cameras
            for (const std::string& topic : camera_topics) {
                rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
                    topic + "/image", 10,
                    [this, topic](const sensor_msgs::msg::Image::SharedPtr msg)
                    {
                        this->addToBufferCallback(msg, topic);
                    }
                );
                

                subscriptions_.push_back(subscription_);
                detection_publishers_[topic] = this->create_publisher<yolov8_interfaces::msg::Yolov8Detections>(
                    "/yolov8" + topic + "/detections", 10
                );
                image_publishers_[topic] = this->create_publisher<sensor_msgs::msg::Image>(
                    "/yolov8" + topic + "/image", 10
                );
                one_channel_mask_publishers_[topic] = this->create_publisher<sensor_msgs::msg::Image>(
                    "/yolov8" + topic + "seg_mask_one_channel", 10
                );
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
        void addToBufferCallback(const sensor_msgs::msg::Image::SharedPtr &image_msg, const std::string topic) {
            // TODO: Will this be deleted in memory since it is passed?
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            current_buffer_[topic] = image_msg;

            // Check if ready for batching
            if (current_buffer_.size() == topic.size()) {
                batchBufferCallback();
            }
        }

        void batchBufferCallback() {
            // Swap the current buffer with the processing buffer
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            lock.lock();
            std::swap(current_buffer_, processing_buffer_);
            // Clear the current buffer
            current_buffer_.clear();
            lock.unlock();

            // Reset buffer timer
            buffer_timer_->reset();

            // Check if all the camera topics are in the processing buffer
            checkCameraTopicsInBuffer();

            // Preprocess the input
            std::vector<cv::Mat> images;
            preprocess_callback(images);

            // TODO: batch to network properly
            // Run inference
            std::vector<std::vector<Object>> objects = yoloV8_.detectObjects(images);
            RCLCPP_INFO(this->get_logger(), "Detected %zu objects", objects.size());
            // RCLCPP_INFO(this->get_logger(), "Typeid: %s", typeid(objects[0].boxMask).name());
            // RCLCPP_INFO(this->get_logger(), "Mask Shape: %s", objects[0].boxMask.size());

            // Postprocess the output and publish the results
            postprocess_callback(objects, images);
        }

        /*
        * Check if all camera topics are in the processing buffer and print out any missing topics
        */
        void checkCameraTopicsInBuffer() {
            if (processing_buffer_.size() != camera_topics_.size()) {
                RCLCPP_WARN(this->get_logger(), "No camera topics are in the processing buffer");
                return;
            // Print out camera topics missing from the processing buffer
            } else {
                for (const std::string& topic : camera_topics_) {
                    if (processing_buffer_.find(topic) == processing_buffer_.end()) {
                        RCLCPP_WARN(this->get_logger(), "Camera topic %s is missing from the processing buffer", topic.c_str());
                    }
                }
            }
        }

        /*
        * Preprocess input(s) for Neural Network by converting ROS image(s) to OpenCV image(s)
        * and converting from RGB8 to BGR8
        *
        * @param images: the vector to store the preprocessed images
        */
        void preprocess_callback(std::vector<cv::Mat>& images) {
            for (const auto& pair : processing_buffer_) {
                std::string topic = pair.first;
                const sensor_msgs::msg::Image::SharedPtr image_msg = pair.second;
                try
                    {
                        // Share the memory with the original image
                        // TODO: Should this be a copy to deal with a cleared buffer?
                        cv_bridge::CvImageConstPtr cv_ptr;
                        cv_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::RGB8);

                        // Convert from RGB8 to BGR8
                        cv::Mat img = cv_ptr->image;
                        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

                        images.push_back(img);
                    } catch (cv_bridge::Exception& e) {
                        RCLCPP_ERROR(this->get_logger(), "Failed to convert ROS image message on topic %s \
                            due to cv_bridge error: %s", topic.c_str(), e.what());
                        continue;
                    }
            }
        }

        /*
        * Convert output(s) from Neural Network to ROS messages and publish them
        *
        * @param objects: Detected objects in each image from the neural network
        * @param images: The images that the objects were detected in
        */
        void postprocess_callback(std::vector<std::vector<Object>> objects, std::vector<cv::Mat> images) {
            // TODO: Make these flags ROS parameters
            bool visualize_masks = true;
            bool enable_one_channel_mask = true;
            bool visualize_one_channel_mask = true;

            int i = 0;
            for (const auto& pair : processing_buffer_) {
                const std::string topic = pair.first;
                const sensor_msgs::msg::Image::SharedPtr image_msg = pair.second;
                cv::Mat image = images[i];
                // ROS message to publish the detections
                yolov8_interfaces::msg::Yolov8Detections detectionMsg;
                detectionMsg.header = image_msg->header;

                if (enable_one_channel_mask) {
                    // TODO fix how batched objects are handled
                    publishOneChannelMask(objects[i], visualize_one_channel_mask, image_msg, detectionMsg,
                        topic);
                }

                // Draw the segmentation masks and bounding boxes on the image
                // to visualize the detections
                if (visualize_masks) {
                    // TODO fix how batched objects are handled
                    visualizeMask(objects[i], image, topic, image_msg);
                }

                // Convert detected objects to ROS message
                // TODO fix how batched objects are handled
                addObjectsToDetectionMsg(objects[i], detectionMsg);

                // Publish the detections
                detection_publishers_[topic]->publish(detectionMsg);
                i++;
            }
        }

        /*
        * Visualize the segmentation masks and bounding boxes on the image and publish it
        * to the given topic.
        *
        * @param objects: Detected objects from the neural network
        * @param image: The image to visualize the masks on
        * @param topic: The ROS topic to publish the image to
        * @param image_msg: The original ROS image message from the camera
        */
        void visualizeMask(std::vector<Object> objects, cv::Mat image,
                const std::string topic,
                const sensor_msgs::msg::Image::SharedPtr image_msg) {
            // Draw the object labels on the image
            yoloV8_.drawObjectLabels(image, objects);

            // Turn cv_image into sensor_msgs::msg::Image
            sensor_msgs::msg::Image displayImageMsg;
            // TODO: Change to rgb8?
            cv_bridge::CvImagePtr cv_image = std::make_shared<cv_bridge::CvImage>(
                image_msg->header, "bgr8", image
            );
            cv_image->toImageMsg(displayImageMsg);
            // Publish segmented image as ROS message
            image_publishers_[topic]->publish(displayImageMsg);
        }

        /*
        * Create a one channel mask for all segmentation objects and publish it. Adds oneChannelMask
        * to detectionMsg. Optionally visualize the mask.
        *
        * @param objects: Detected objects from the neural network
        * @param visualize_one_channel_mask: Whether to visualize the one channel mask
        * @param image_msg: Original ROS image from the camera
        * @param detectionMsg: ROS message to publish detections
        * @param topic: The ROS topic to publish the mask to
        */
        void publishOneChannelMask(std::vector<Object> objects, bool visualize_one_channel_mask,
                const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                yolov8_interfaces::msg::Yolov8Detections detectionMsg,
                const std::string topic) {
            cv::Mat oneChannelMask;
            int img_width = image_msg->width;
            int img_height = image_msg->height;
            yoloV8_.getOneChannelSegmentationMask(objects, oneChannelMask, img_width, img_height);
            // Use ROS cv_bridge to convert cv::Mat to sensor_msgs::msg::Image and take header from original camera image
            try {
                cv_bridge::CvImage cvBridgeOneChannelMask = cv_bridge::CvImage(
                    image_msg->header, "mono8", oneChannelMask
                );
                detectionMsg.seg_mask_one_channel = *cvBridgeOneChannelMask.toImageMsg();
            } catch (cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }

            // Publish the one channel mask with RGB color for visualization only
            if (visualize_one_channel_mask) {
                cv::Mat oneChannelMaskRGB8 = visualizeOneChannelMask(oneChannelMask);
                try {
                    cv_bridge::CvImage cvBridgeOneChannelMaskRGB8 = cv_bridge::CvImage(
                        image_msg->header, "rgb8", oneChannelMaskRGB8
                    );
                    one_channel_mask_publishers_[topic]->publish(*cvBridgeOneChannelMaskRGB8.toImageMsg());
                } catch (cv_bridge::Exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                    return;
                }
            }
        }

        /*
        * Add detected objects to the yolov8detections message
        *
        * @param objects: Detected objects from the neural network
        * @param detectionMsg: The yolov8detections message to add the objects to
        */
        void addObjectsToDetectionMsg(std::vector<Object> objects,
                yolov8_interfaces::msg::Yolov8Detections detectionMsg) const {
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
        cv::Mat visualizeOneChannelMask(cv::Mat oneChannelMask) const {
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
                return oneChannelMaskRGB8;
            } catch (cv::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "OpenCV exception: %s", e.what());
                return cv::Mat();
            }
        }

        std::vector<std::string> camera_topics_;
        std::unordered_map<std::string, rclcpp::Publisher<yolov8_interfaces::msg::Yolov8Detections>::SharedPtr> detection_publishers_;
        std::unordered_map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> image_publishers_;
        std::unordered_map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> one_channel_mask_publishers_;
        rclcpp::TimerBase::SharedPtr buffer_timer_;
        std::unordered_map<std::string, sensor_msgs::msg::Image::SharedPtr> current_buffer_;
        std::unordered_map<std::string, sensor_msgs::msg::Image::SharedPtr> processing_buffer_;
        std::mutex buffer_mutex_;

        std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subscriptions_;
        // std::unordered_map<std::string, rclcpp::Publisher<yolov8_interfaces::msg::Yolov8Detections>::SharedPtr> detection_publisher_;
        // rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
        // rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr oneChannelMask_publisher_;
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