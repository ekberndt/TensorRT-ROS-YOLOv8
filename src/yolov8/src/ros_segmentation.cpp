#include "yolov8.h"
#include "cmd_line_util.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
// #include "yolov8_interfaces/msg/YOLOv8Obj.hpp"
// #include "yolov8_interfaces/msg/YOLOv8Seg.hpp"
#include "yolov8_interfaces/msg/yol_ov8_seg.hpp" // TODO: Why are the msg names wrong? 
#include "yolov8_interfaces/msg/yol_ov8_obj.hpp"

using std::placeholders::_1;

class YoloV8Node : public rclcpp::Node
{
    public:
        YoloV8Node(YoloV8& yoloV8)
        : Node("yolo_v8"), yoloV8_(yoloV8)
        {
            subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/vimba_front_right_center/image", 10, std::bind(&YoloV8Node::camera_callback, this, _1));

            publisher_ = this->create_publisher<yolov8_interfaces::msg::YOLOv8Seg>("/yolov8", 10);
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

                // Draw the bounding boxes on the image
                yoloV8_.drawObjectLabels(img, objects);

                RCLCPP_INFO(this->get_logger(), "Detected %zu objects", objects.size());

                // Convert detected objects to ROS message
                yolov8_interfaces::msg::YOLOv8Seg msg;

                for (auto & object : objects) {
                    int label = object.label;
                    float prob = object.probability;

                    // Bounding box
                    int rect_x = object.rect.x;
                    int rect_y = object.rect.y + 1;
                    int rect_width = object.rect.width;
                    int rect_height = object.rect.height;

                    // Semantic segmentation mask

                    // int box_mask_rows = object.boxMask.rows;
                    // int box_mask_cols = object.boxMask.cols;
                    // int box_mask_data = object.boxMask.data;

                    cv::Mat mask = img.clone();
                    cv::Scalar color = cv::Scalar(1, 1, 1);
                    mask(object.rect).setTo(color * 255, object.boxMask);
                    // cv::addWeighted(image, 0.0, mask, 1, 1, image);

                    // Create yolov8_obj message
                    yolov8_interfaces::msg::YOLOv8Obj obj_msg;
                    obj_msg.label = label;
                    obj_msg.probability = prob;
                    obj_msg.rect_x = rect_x;
                    obj_msg.rect_y = rect_y;
                    obj_msg.rect_width = rect_width;
                    obj_msg.rect_height = rect_height;

                    // Add YOLOv8Obj message to YOLOv8Seg message
                    msg.objects.push_back(obj_msg);
                }
                publisher_->publish(msg);

                // Save the annotated image
                if (!objects.empty()) {
                    const auto outputName = "test_annotated.jpg";
                    if (!cv::imwrite(outputName, img)) {
                        RCLCPP_ERROR(this->get_logger(), "Failed to save annotated image");
                        return;
                    }
                    std::cout << "Saved annotated image to: " << outputName << std::endl;
                }
                
            } catch (cv::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv exception: %s", e.what());
                return;
            }
        }
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
        rclcpp::Publisher<yolov8_interfaces::msg::YOLOv8Seg>::SharedPtr publisher_;
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
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Creating YoloV8 engine");
    YoloV8 yoloV8(onnxModelPath, config);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "YoloV8 engine created");
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloV8Node>(yoloV8));
    rclcpp::shutdown();

    return 0;
}