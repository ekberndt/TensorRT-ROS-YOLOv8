#pragma once
#include "engine.h"
#include <fstream>

// Utility method for checking if a file exists on disk
inline bool doesFileExist(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

struct Object {
    // The object class.
    int label{};
    // The detection's confidence probability.
    float probability{};
    // The object bounding box rectangle.
    cv::Rect_<float> rect;
    // Semantic segmentation mask
    cv::Mat boxMask;
    // Contours of semantic segmentation mask
    std::vector<std::vector<cv::Point>> contours;
    // Pose estimation key points
    std::vector<float> kps{};
};

// Config the behavior of the YoloV8 detector.
// Can pass these arguments as command line parameters.
struct YoloV8Config {
    // The precision to be used for inference
    Precision precision = Precision::FP16;
    // Calibration data directory. Must be specified when using INT8 precision.
    std::string calibrationDataDirectory;
    // Probability threshold used to filter detected objects
    float probabilityThreshold = 0.25f;
    // Non-maximum suppression threshold
    float nmsThreshold = 0.65f;
    // Max number of detected objects to return
    int topK = 100;
    // Segmentation config options (these are the output dimensions of the ONNX model - from output1 in the ONNX graph)
    int segChannels = 32;
    int segH = 160;
    int segW = 160;
    float segmentationThreshold = 0.5f;
    // Pose estimation options
    int numKPS = 17;
    float kpsThreshold = 0.5f;
    // Class thresholds (default are COCO classes)
    // If you want to use your own custom classes, you can change the strings where each index of the vector
    // maps to the class number (i.e. 0 -> "car"). YOU MUST add a color to the COLOR_LIST below for each
    // class. Match sure the number of colors matches the number of classes.
    std::vector<std::string> classNames = {
        "car",
        "misc0",
        "misc1",
        "misc2",
        "misc3",
        "misc4",
        "misc5",
    };
};

class YoloV8 {
public:
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    YoloV8(const std::string& onnxModelPath, const YoloV8Config& config);

    // Upload image(s) to GPU then call detectObjects
    std::vector<Object> detectObjects(const cv::Mat& imgMat);
    std::vector<std::vector<Object>> detectObjects(std::vector<cv::Mat> &imgMat); // Batched version

    // Run inference on objects uploaded to the GPU
    std::vector<Object> detectObjects(const cv::cuda::GpuMat& imgMat);
    std::vector<std::vector<Object>> detectObjects(std::vector<cv::cuda::GpuMat> &imgMat);  // Batched version

    // Create a one channel segmentation mask for all segmentation objects
    void getOneChannelSegmentationMask(const std::vector<Object>& objects, cv::Mat& segMaskOneChannel, int img_height, int img_width);

    // Draw the object bounding boxes and labels on the image
    void drawObjectLabels(cv::Mat& image, const std::vector<Object> &objects, unsigned int scale = 2);

    // Draw the object bounding boxes and labels on the image and store a binary mask in the given
    // masks vector for each instance segmentation object.
    void drawObjectLabels(cv::Mat& image, const std::vector<Object> &objects, std::vector<cv::Mat> &masks, unsigned int scale = 2);

    // Getter for number of classes
    int getNumClasses() const { return CLASS_NAMES.size(); }

    // Getter for CLASS_NAMES
    std::string getClassName(int i) const { return CLASS_NAMES[i].c_str(); }

    // Getter for batch size
    int getBatchSize() const { return m_trtEngine->getBatchSize(); }
private:
    // Preprocess the input
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat& gpuImg);
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(std::vector<cv::cuda::GpuMat> &gpuImg); // Batched version

    // Postprocess the output
    std::vector<Object> postprocessDetect(std::vector<float>& featureVector);

    // Postprocess the output for pose model
    std::vector<Object> postprocessPose(std::vector<float>& featureVector);

    // Postprocess the output for segmentation model
    std::vector<std::vector<Object>> postProcessSegmentation(std::vector<std::vector<std::vector<float>>>& batchedFeatureVectors); // Batched version
    std::vector<Object> postProcessSegmentation(std::vector<std::vector<float>>& featureVectors);

    std::unique_ptr<Engine> m_trtEngine = nullptr;

    // Used for image preprocessing
    // YoloV8 model expects values between [0.f, 1.f] so we use the following params
    const std::array<float, 3> SUB_VALS {0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS {1.f, 1.f, 1.f};
    const bool NORMALIZE = true;
    
    // Image dimensions gathered from the input image
    float m_ratio_ = 1;
    float m_imgWidth_ = 0;
    float m_imgHeight_ = 0;

    // Filter thresholds
    const float PROBABILITY_THRESHOLD;
    const float NMS_THRESHOLD;
    const int TOP_K;

    // Segmentation constants
    const int SEG_CHANNELS;
    const int SEG_H;
    const int SEG_W;
    const float SEGMENTATION_THRESHOLD;

    // Object classes as strings
    const std::vector<std::string> CLASS_NAMES;

    // Pose estimation constant
    const int NUM_KPS;
    const float KPS_THRESHOLD;

    // Color list for drawing objects
    const std::vector<std::vector<float>> COLOR_LIST = {
            {0.098, 0.325, 0.850},
            {0.125, 0.694, 0.929},
    };

    const std::vector<std::vector<unsigned int>> KPS_COLORS = {
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255}
    };

    const std::vector<std::vector<unsigned int>> SKELETON = {
            {16, 14},
            {14, 12},
            {17, 15},
            {15, 13},
            {12, 13},
            {6, 12},
            {7, 13},
            {6, 7},
            {6, 8},
            {7, 9},
            {8, 10},
            {9, 11},
            {2, 3},
            {1, 2},
            {1, 3},
            {2, 4},
            {3, 5},
            {4, 6},
            {5, 7}
    };

    const std::vector<std::vector<unsigned int>> LIMB_COLORS = {
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {255, 51, 255},
            {255, 51, 255},
            {255, 51, 255},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0}
    };
};