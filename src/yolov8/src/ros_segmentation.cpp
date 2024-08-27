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
        YoloV8Node(YoloV8& yoloV8)
        : Node("yolo_v8"), yoloV8_(yoloV8)
        {
            // Get ROS parameters
            this->declare_parameter("camera_topics", camera_topics_);
            this->get_parameter("camera_topics", camera_topics_);
            this->declare_parameter("camera_buffer_hz", camera_buffer_hz_);
            this->get_parameter("camera_buffer_hz", camera_buffer_hz_);
            this->declare_parameter("visualize_masks", visualize_masks_);
            this->get_parameter("visualize_masks", visualize_masks_);
            this->declare_parameter("enable_one_channel_mask", enable_one_channel_mask_);
            this->get_parameter("enable_one_channel_mask", enable_one_channel_mask_);
            this->declare_parameter("visualize_one_channel_mask", visualize_one_channel_mask_);
            this->get_parameter("visualize_one_channel_mask", visualize_one_channel_mask_);

            // Create timer for camera synchronization
            float buffer_duration = 1 / camera_buffer_hz_;
            buffer_timer_ = rclcpp::create_wall_timer(
                std::chrono::duration<float>(buffer_duration),
                std::bind(&YoloV8Node::batchBufferCallback, this),
                nullptr,
                this->get_node_base_interface().get(),
                this->get_node_timers_interface().get()
            );

            // Print Camera Topics
            RCLCPP_INFO(this->get_logger(), "Camera Topics:");
            for (const std::string& topic : camera_topics_) {
                RCLCPP_INFO(this->get_logger(), "  %s", topic.c_str());
            }

            // Check if model batch size matches the number of camera topics
            if (camera_topics_.size() != yoloV8_.getBatchSize()) {
                throw std::runtime_error("Model batch size (" + std::to_string(yoloV8_.getBatchSize()) +
                    ") does not match the number of camera topics (" + std::to_string(camera_topics_.size()) + ")");
            }

            // Create subscribers and publishers for all cameras
            for (const std::string& topic : camera_topics_) {
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
                    "/yolov8" + topic + "/seg_mask_one_channel", 10
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
            std::cout << "Received image message on topic " << topic << std::endl;
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            current_buffer_[topic] = image_msg;

            // Check if ready for batching
            if (current_buffer_.size() == camera_topics_.size()) {
                lock.unlock();
                batchBufferCallback();
            }
        }

        void batchBufferCallback() {
            // Swap the current buffer with the processing buffer
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            std::swap(current_buffer_, processing_buffer_);
            // Clear the current buffer
            current_buffer_.clear();
            lock.unlock();

            // Reset buffer timer
            buffer_timer_->reset();

            std::map<std::string, cv::Mat> images_map;

            // Check if all the camera topics are in the processing buffer
            std::vector<std::string> missing_topics = checkCameraTopicsInBuffer();
            if (missing_topics.size() == camera_topics_.size()) {
                std::cout << "No camera topics in the processing buffer at time " << this->now().seconds() << std::endl;
                return;
            } else {
                // Add black images for missing topics
                for (const std::string& topic : missing_topics) {
                    images_map[topic] = cv::Mat::zeros(cv::Size(640, 640), CV_8UC3);
                }
            }

            // Preprocess the input
            preprocess_callback(images_map);

            // Place map values into a vector
            std::vector<cv::Mat> images;
            for (const auto& pair : images_map) {
                images.push_back(pair.second);
            }

            // TODO: batch to network properly
            // Run inference
            std::vector<std::vector<Object>> objects = yoloV8_.detectObjects(images);

            // RCLCPP_INFO(this->get_logger(), "Inference ran on %zu cameras", images.size());
            // Print out a summary of the detected objects on each camera
            // if (objects.size() > 0) {
            //     RCLCPP_INFO(this->get_logger(), "========== Detection Summary ==========");
            // } else {
            //    std::cout << "No objects detected at time " << this->now().seconds() << std::endl;
            // }
            int i = 0;
            for (const auto& batch : objects) {
                std::string topic = camera_topics_[i];
                if (!batch.empty()) {
                    RCLCPP_INFO(this->get_logger(), "Detected %zu object(s) on %s", batch.size(), topic.c_str());
                    for (const auto& object : batch) {
                        std::cout << "\tDetected : " << yoloV8_.getClassName(object.label) << ", Prob: " << object.probability << std::endl;
                        RCLCPP_INFO(this->get_logger(), "\t%s: %f", yoloV8_.getClassName(object.label).c_str(), object.probability);
                    }
                }
                i++;
            }
            // RCLCPP_INFO(this->get_logger(), "Detected %zu objects across all cameras", total_objects);
            // RCLCPP_INFO(this->get_logger(), "Typeid: %s", typeid(objects[0].boxMask).name());
            // RCLCPP_INFO(this->get_logger(), "Mask Shape: %s", objects[0].boxMask.size());

            // Postprocess the output and publish the results
            postprocess_callback(images_map, objects, images, missing_topics);
        }

        /*
        * Check number of camera topics in the processing buffer and return any missing topics
        *
        * @return a vector of missing camera topics
        */
        std::vector<std::string> checkCameraTopicsInBuffer() {
            std::vector<std::string> missing_topics;
            if (processing_buffer_.size() == camera_topics_.size()) {
                std::cout << "All camera topics are in the processing buffer" << std::endl;
            } else {
                if (processing_buffer_.size() == 0) {
                    std::cout << "No camera topics are in the processing buffer" << std::endl;
                }
                for (const std::string& topic : camera_topics_) {
                    if (processing_buffer_.find(topic) == processing_buffer_.end()) {
                        std::cout << "Camera topic " << topic << " is missing from the processing buffer" << std::endl;
                        missing_topics.push_back(topic);
                    }
                }
            }

            return missing_topics;
        }

        /*
        * Preprocess input(s) for Neural Network by converting ROS image(s) to OpenCV image(s)
        * and converting from RGB8 to BGR8
        *
        * @param images: the map to store the preprocessed images
        */
        void preprocess_callback(std::map<std::string, cv::Mat>& images) {
            for (const auto& pair : processing_buffer_) {
                std::string camera_topic = pair.first;
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

                        images[camera_topic] = img;
                    } catch (cv_bridge::Exception& e) {
                        RCLCPP_ERROR(this->get_logger(), "Failed to convert ROS image message on topic %s \
                            due to cv_bridge error: %s", camera_topic.c_str(), e.what());
                        continue;
                    }
            }
        }

        /*
        * Convert output(s) from Neural Network to ROS messages and publish them
        *
        * @param objects: Batch of detected objects from the neural network
        * @param images: The images that the objects were detected in
        * @param missing_topics: The camera topics that were missing from the processing buffer
        */
        void postprocess_callback(std::map<std::string, cv::Mat> images_map, std::vector<std::vector<Object>> objects,
                std::vector<cv::Mat> images, std::vector<std::string> missing_topics) {

            int i = 0;
            for (const auto& pair : images_map) {
                const std::string topic = pair.first;
                // Skip any missing camera topics since they will have a black image
                if (std::find(missing_topics.begin(), missing_topics.end(), topic) != missing_topics.end()) {
                    i++;
                    std::cout << "Skipping missing topic " << topic << std::endl;
                    continue;
                }

                const sensor_msgs::msg::Image::SharedPtr image_msg = processing_buffer_[topic];
                cv::Mat image = images[i];
                // ROS message to publish the detections
                yolov8_interfaces::msg::Yolov8Detections detectionMsg;
                detectionMsg.header = image_msg->header;

                if (enable_one_channel_mask_) {
                    // TODO fix how batched objects are handled
                    publishOneChannelMask(objects[i], visualize_one_channel_mask_, image_msg, detectionMsg,
                        topic);
                }

                // Draw the segmentation masks and bounding boxes on the image
                // to visualize the detections
                if (visualize_masks_) {
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
                yolov8_interfaces::msg::Yolov8Detections &detectionMsg,
                const std::string topic) {
            cv::Mat oneChannelMask;
            int img_width = image_msg->width;
            int img_height = image_msg->height;
            yoloV8_.getOneChannelSegmentationMask(objects, oneChannelMask, img_height, img_width);
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
                yolov8_interfaces::msg::Yolov8Detections& detectionMsg) const {
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
                if (label + 1 > yoloV8_.getNumClasses()) {
                    RCLCPP_ERROR(this->get_logger(), "Label %d does not have a corresponding class name. Did you update yolov8.env to include all classes?", label);
                    continue;
                }
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
        float camera_buffer_hz_ = 30;
        bool visualize_masks_;
        bool enable_one_channel_mask_;
        bool visualize_one_channel_mask_;
        std::map<std::string, rclcpp::Publisher<yolov8_interfaces::msg::Yolov8Detections>::SharedPtr> detection_publishers_;
        std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> image_publishers_;
        std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> one_channel_mask_publishers_;
        rclcpp::TimerBase::SharedPtr buffer_timer_;
        std::map<std::string, sensor_msgs::msg::Image::SharedPtr> current_buffer_;
        std::map<std::string, sensor_msgs::msg::Image::SharedPtr> processing_buffer_;
        std::mutex buffer_mutex_;

        std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subscriptions_;
        YoloV8& yoloV8_;
        };

int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;

    std::string parseArgsError = parseArguments(argc, argv, config, onnxModelPath);
    if (!parseArgsError.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "YOLOv8 does not support and is throwing out argument: %s", parseArgsError.c_str());
    }
    // Create the YoloV8 engine
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Creating YoloV8 engine --- Could take a while if Engine file is not already built and cached.");
    YoloV8 yoloV8(onnxModelPath, config);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "YoloV8 engine created and loaded into memory.");

    // Create ROS2 Node
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloV8Node>(yoloV8));
    rclcpp::shutdown();
    return 0;
}