#include <opencv2/cudaimgproc.hpp>
#include "yolov8.h"

YoloV8::YoloV8(const std::string& onnxModelPath, const YoloV8Config& config)
        : PROBABILITY_THRESHOLD(config.probabilityThreshold)
        , NMS_THRESHOLD(config.nmsThreshold)
        , TOP_K(config.topK)
        , SEG_CHANNELS(config.segChannels)
        , SEG_H(config.segH)
        , SEG_W(config.segW)
        , SEGMENTATION_THRESHOLD(config.segmentationThreshold)
        , CLASS_NAMES(config.classNames)
        , NUM_KPS(config.numKPS)
        , KPS_THRESHOLD(config.kpsThreshold) {
    // Specify options for GPU inference
    Options options;
    options.optBatchSize = 4;
    options.maxBatchSize = 4;

    options.precision = config.precision;
    options.calibrationDataDirectoryPath = config.calibrationDataDirectory;

    if (options.precision == Precision::INT8) {
        if (options.calibrationDataDirectoryPath.empty()) {
            throw std::runtime_error("Error: Must supply calibration data path for INT8 calibration");
        }
    }

    // Create our TensorRT inference engine
    // TODO: See why making this engine sometimes maxes out the memory and causes program to get SIGKILLed (exit code -6)
    m_trtEngine = std::make_unique<Engine>(options);

    // Build the onnx model into a TensorRT engine file
    // If the engine file already exists, this function will return immediately
    // The engine file is rebuilt any time the above Options are changed.
    auto succ = m_trtEngine->build(onnxModelPath, SUB_VALS, DIV_VALS, NORMALIZE);
    if (!succ) {
        const std::string errMsg = "Error: Unable to build the TensorRT engine. "
                                   "Try increasing TensorRT log severity to kVERBOSE (in src/yolov8/libs/tensorrt-cpp-api/engine.cpp).";
        throw std::runtime_error(errMsg);
    }

    // Load the TensorRT engine file
    succ = m_trtEngine->loadNetwork(onnxModelPath);
    if (!succ) {
        throw std::runtime_error("Error: Unable to load TensorRT engine weights into memory.");
    }
}

/*
* Preprocess a batch of images for inference on the TensorRT Engine
*
* @param gpuImgs: a batch of input images to preprocess
*/
std::vector<std::vector<cv::cuda::GpuMat>> YoloV8::preprocess(std::vector<cv::cuda::GpuMat> &gpuImgs) {
    // Populate the input vectors
    const std::vector<nvinfer1::Dims3>& inputDims = m_trtEngine->getInputDims();
    std::cout << "Engine input dims: " << inputDims[0].d[0] << ", " << inputDims[0].d[1] << ", " << inputDims[0].d[2] << ", " << inputDims[0].d[3] << std::endl;
    // The input to the neural network is a tensor with dimensions [input][batch][cv::cuda::GpuMat]
    std::vector<std::vector<cv::cuda::GpuMat>> inputs {};
    std::vector<cv::cuda::GpuMat> batches {};
    m_imgHeight_.clear();
    m_imgWidth_.clear();
    m_ratio_.clear();
    // Iterate over each image in the batch
    for (cv::cuda::GpuMat &gpuImg : gpuImgs) {
        // Store the original image dimensions and the ratio used to resize the image so we can convert the
        // outputs back to the original image size in the post-processing stage
        m_imgHeight_.push_back(gpuImg.rows);
        m_imgWidth_.push_back(gpuImg.cols);
        m_ratio_.push_back(1.f / std::min(inputDims[0].d[2] / static_cast<float>(gpuImg.cols),
            inputDims[0].d[1] / static_cast<float>(gpuImg.rows)));
         // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
        if (gpuImg.rows != inputDims[0].d[1] || gpuImg.cols != inputDims[0].d[2]) {
            // Only resize if not already the right size to avoid unecessary copy
            gpuImg = Engine::resizeKeepAspectRatioPadRightBottom(gpuImg, inputDims[0].d[1], inputDims[0].d[2]);
        }

        // Convert to format expected by our inference engine
        // Input is (N, C, H, W)
        // The reason for the strange format is because it supports models with multiple inputs as well as batching
        // In our case though, the model only has a single input and we are using a batch size of N.
        batches.push_back(std::move(gpuImg));
        // std::vector<cv::cuda::GpuMat> input{std::move(gpuImg)};
        // inputs.push_back(std::move(input));
    };
    inputs.push_back(std::move(batches));

    return inputs;
}

std::vector<std::vector<cv::cuda::GpuMat>> YoloV8::preprocess(const cv::cuda::GpuMat &gpuImg) {
    // Populate the input vectors
    const auto& inputDims = m_trtEngine->getInputDims();
    cv::cuda::GpuMat rgbMat = gpuImg;

    auto resized = rgbMat;

    // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
    if (resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2]) {
        // Only resize if not already the right size to avoid unecessary copy
        resized = Engine::resizeKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2]);
    }

    // Convert to format expected by our inference engine
    // The reason for the strange format is because it supports models with multiple inputs as well as batching
    // In our case though, the model only has a single input and we are using a batch size of 1.
    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    std::vector<std::vector<cv::cuda::GpuMat>> inputs {std::move(input)};

    // These params will be used in the post-processing stage
    m_imgHeight_.push_back(rgbMat.rows);
    m_imgWidth_.push_back(rgbMat.cols);
    m_ratio_.push_back(1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows)));

    return inputs;
}

// /*
// * Run inference on a single image (does not support batching).
// *
// * @param gpuImgs: a vector of input images
// */
// std::vector<Object> YoloV8::detectObjects(const cv::cuda::GpuMat &imgMat) {
//     // Preprocess the input image
// #ifdef ENABLE_BENCHMARKS
//     static int numIts = 1;
//     preciseStopwatch s1;
// #endif
//     const auto input = preprocess(imgMat);
// #ifdef ENABLE_BENCHMARKS
//     static long long t1 = 0;
//     t1 += s1.elapsedTime<long long, std::chrono::microseconds>();
//     std::cout << "Avg Preprocess time: " << (t1 / numIts) / 1000.f << " ms" << std::endl;
// #endif
//     // Run inference using the TensorRT engine
// #ifdef ENABLE_BENCHMARKS
//     preciseStopwatch s2;
// #endif
//     std::vector<std::vector<std::vector<float>>> featureVectors;
//     auto succ = m_trtEngine->runInference(input, featureVectors);
//     if (!succ) {
//         throw std::runtime_error("Error: Unable to run inference.");
//     }
// #ifdef ENABLE_BENCHMARKS
//     static long long t2 = 0;
//     t2 += s2.elapsedTime<long long, std::chrono::microseconds>();
//     std::cout << "Avg Inference time: " << (t2 / numIts) / 1000.f << " ms" << std::endl;
//     preciseStopwatch s3;
// #endif
//     // Check if our model does only object detection or also supports segmentation
//     std::vector<Object> ret;
//     const auto& numOutputs = m_trtEngine->getOutputDims().size();
//     if (numOutputs == 1) {
//         // Object detection or pose estimation
//         // Since we have a batch size of 1 and only 1 output, we must convert the output from a 3D array to a 1D array.
//         std::vector<float> featureVector;
//         Engine::transformOutput(featureVectors, featureVector);

//         const auto& outputDims = m_trtEngine->getOutputDims();
//         int numChannels = outputDims[outputDims.size() - 1].d[1];
//         // TODO: Need to improve this to make it more generic (don't use magic number).
//         // For now it works with Ultralytics pretrained models.
//         if (numChannels == 56) {
//             // Pose estimation
//             ret = postprocessPose(featureVector);
//         } else {
//             // Object detection
//             ret = postprocessDetect(featureVector);
//         }
//     } else {
//         // Segmentation
//         // Since we have a batch size of 1 and 2 outputs, we must convert the output from a 3D array to a 2D array.
//         std::vector<std::vector<float>> featureVector;
//         Engine::transformOutput(featureVectors, featureVector);
//         ret = postProcessSegmentation(featureVector);
//     }
// #ifdef ENABLE_BENCHMARKS
//     static long long t3 = 0;
//     t3 +=  s3.elapsedTime<long long, std::chrono::microseconds>();
//     std::cout << "Avg Postprocess time: " << (t3 / numIts++) / 1000.f << " ms\n" << std::endl;
// #endif
//     return ret;
// }

/*
* Run inference on a batch of images. Note this function only support segmentation models.
* 
* @param gpuImgs: a vector of input images
*/
std::vector<std::vector<Object>> YoloV8::detectObjects(std::vector<cv::cuda::GpuMat> &gpuImgs) {
    // Preprocess the input image
#ifdef ENABLE_BENCHMARKS
    static int numIts = 1;
    preciseStopwatch s1;
#endif
    const auto input = preprocess(gpuImgs);
#ifdef ENABLE_BENCHMARKS
    static long long t1 = 0;
    t1 += s1.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Preprocess time: " << (t1 / numIts) / 1000.f << " ms" << std::endl;
#endif
    // Run inference using the TensorRT engine
#ifdef ENABLE_BENCHMARKS
    preciseStopwatch s2;
#endif
    std::vector<std::vector<std::vector<float>>> featureVectors;
    auto succ = m_trtEngine->runInference(input, featureVectors);
    if (!succ) {
        throw std::runtime_error("Error: Unable to run inference.");
    }
#ifdef ENABLE_BENCHMARKS
    static long long t2 = 0;
    t2 += s2.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Inference time: " << (t2 / numIts) / 1000.f << " ms" << std::endl;
    preciseStopwatch s3;
#endif
    std::vector<std::vector<Object>> ret;
    ret = postProcessSegmentation(featureVectors);
#ifdef ENABLE_BENCHMARKS
    static long long t3 = 0;
    t3 +=  s3.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Postprocess time: " << (t3 / numIts++) / 1000.f << " ms\n" << std::endl;
#endif
    return ret;
}

// /**
//  * Uploads the input image to GPU memory and calls detectObjects(...) on the GPU image.
//  * 
//  * @param imgMat The input image in BGR format.
//  * @return A vector of detected objects.
//  */
// std::vector<Object> YoloV8::detectObjects(const cv::Mat &imgMat) {
//     // Upload the image to GPU memory
//     cv::cuda::GpuMat gpuImg;
//     gpuImg.upload(imgMat);

//     // Call detectObjects with the GPU image
//     return detectObjects(gpuImg);
// }

/**
 * Uploads the batched input images to GPU memory and calls detectObjects(...) on the GPU images.
 * 
 * @param imgMat The batched images in BGR format.
 * @return A vector of detected objects.
 */
std::vector<std::vector<Object>> YoloV8::detectObjects(std::vector<cv::Mat> &imgMats) {
    std::vector<cv::cuda::GpuMat> gpuImgs;
    // // Upload the images to GPU memory
    // std::vector<cv::cuda::Stream> streams(imgMats.size());

    // // Asynchronously upload each image on their own CUDA stream
    // std::cout << "imgsMats size: " << imgMats.size() << std::endl;
    // for (size_t i = 0; i < imgMats.size(); i++) {
    //     try {
    //         gpuImgs[i].upload(imgMats[i], streams[i]);
    //     } catch (const cv::Exception& e) {
    //         std::cerr << "Error uploading image to GPU: " << e.what() << std::endl;
    //     }
    // }

    // std::cout << "Syncing streams" << std::endl;
    // // Make sure all streams have finished uploading
    // for (cv::cuda::Stream &stream : streams) {
    //     stream.waitForCompletion();
    // }
    
    // TODO: Bench Upload with CUDA streams vs sequentially
    for (const cv::Mat& img : imgMats) {
        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(img);
        gpuImgs.push_back(gpuImg);
    }
    

    // Call detectObjects with the GPU image
    return detectObjects(gpuImgs);
}

/**
 * Performs post-processing on a batch of image's segmentation feature vectors.
 * 
 * @param featureVectors The batched vector of 2D feature vectors to be processed of size [batch][output][feature_vector].
 * @return A batch of vectors of Object instances representing the post-processed segmentation results.
 * @throws std::logic_error If the feature vectors are not of the expected length.
 */
std::vector<std::vector<Object>> YoloV8::postProcessSegmentation(std::vector<std::vector<std::vector<float>>>& batchedFeatureVectors) {
    std::vector<std::vector<Object>> batched_objects;
    int batch_index = 0;
    for (std::vector<std::vector<float>> &featureVectors : batchedFeatureVectors) {
        batched_objects.push_back(postProcessSegmentation(featureVectors, batch_index));
        batch_index++;
    }
    return batched_objects;
}

/**
 * Performs post-processing on 2 output buffers for a single batch of image's segmentation feature vectors.
 * 
 * @param featureVectors The 2D feature vectors to be processed.
 * @param batch_index The index of the batch.
 * @return A vector of Object instances representing the post-processed segmentation results.
 * @throws std::logic_error If the feature vectors are not of the expected length.
 */
std::vector<Object> YoloV8::postProcessSegmentation(std::vector<std::vector<float>>& featureVectors, int batch_index) {
    // Retrieve the output dimensions
    // TODO: For the batched version, should we check the output dimensions of each batch? (Likely yes as camera feeds may have different resolutions)
    const auto& outputDims = m_trtEngine->getOutputDims();

    int numChannels = outputDims[outputDims.size() - 1].d[1];
    int numAnchors = outputDims[outputDims.size() - 1].d[2];

    const auto numClasses = numChannels - SEG_CHANNELS - 4;

    // Ensure the lengths of each output buffer are correct
    // if (featureVectors[0].size() != static_cast<size_t>(SEG_CHANNELS) * SEG_H * SEG_W) {
    //     throw std::logic_error("Output buffer output at index 0 has incorrect length");
    // }

    // if (featureVectors[1].size() != static_cast<size_t>(numChannels) * numAnchors) {
    //     throw std::logic_error("Output buffer output at index 1 has incorrect length");
    // }

    cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVectors[1].data());
    output = output.t();

    cv::Mat protos = cv::Mat(SEG_CHANNELS, SEG_H * SEG_W, CV_32F, featureVectors[0].data());

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> maskConfs;
    std::vector<int> indices;

    // Object the bounding boxes and class labels
    for (int i = 0; i < numAnchors; i++) {
        auto rowPtr = output.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto maskConfsPtr = rowPtr + 4 + numClasses;
        auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
        float score = *maxSPtr;
        if (score > PROBABILITY_THRESHOLD) {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * m_ratio_[batch_index], 0.f, m_imgWidth_[batch_index]);
            float y0 = std::clamp((y - 0.5f * h) * m_ratio_[batch_index], 0.f, m_imgHeight_[batch_index]);
            float x1 = std::clamp((x + 0.5f * w) * m_ratio_[batch_index], 0.f, m_imgWidth_[batch_index]);
            float y1 = std::clamp((y + 0.5f * h) * m_ratio_[batch_index], 0.f, m_imgHeight_[batch_index]);

            int label = maxSPtr - scoresPtr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            cv::Mat maskConf = cv::Mat(1, SEG_CHANNELS, CV_32F, maskConfsPtr);

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
            maskConfs.push_back(maskConf);
        }
    }

    // Require OpenCV 4.7 for this function
    // Perform Non-Maximum Suppression to remove overlapping bounding boxes
    cv::dnn::NMSBoxesBatched(
            bboxes,
            scores,
            labels,
            PROBABILITY_THRESHOLD,
            NMS_THRESHOLD,
            indices
    );

    // Obtain the segmentation masks
    // Extract the top k bounding boxes
    cv::Mat masks;
    std::vector<Object> objs;
    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= TOP_K) {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object obj;
        obj.label = labels[i];
        obj.rect = tmp;
        obj.probability = scores[i];
        masks.push_back(maskConfs[i]);
        objs.push_back(obj);
        cnt += 1;
    }

    // Convert segmentation mask to original frame
    if (!masks.empty()) {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat = matmulRes.reshape(indices.size(), { SEG_W, SEG_H });

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        const auto inputDims = m_trtEngine->getInputDims();

        // Maintain the aspect ratio of the original image
        cv::Rect roi;
        if (m_imgHeight_[batch_index] > m_imgWidth_[batch_index]) {
            roi = cv::Rect(0, 0, SEG_W * m_imgWidth_[batch_index] / m_imgHeight_[batch_index], SEG_H);
        } else {
            roi = cv::Rect(0, 0, SEG_W, SEG_H * m_imgHeight_[batch_index] / m_imgWidth_[batch_index]);
        }

        // Resize the mask to the original image size
        for (size_t i = 0; i < indices.size(); i++)
        {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            cv::resize(
                    dest,
                    mask,
                    cv::Size(static_cast<int>(m_imgWidth_[batch_index]), static_cast<int>(m_imgHeight_[batch_index])),
                    cv::INTER_LINEAR
            );
            // Add mask for pixels above segmentation threshold
            objs[i].boxMask = mask(objs[i].rect) > SEGMENTATION_THRESHOLD;
        }
    }

    return objs;
}

// std::vector<Object> YoloV8::postprocessPose(std::vector<float> &featureVector) {
//     const auto& outputDims = m_trtEngine->getOutputDims();
//     auto numChannels = outputDims[0].d[1];
//     auto numAnchors = outputDims[0].d[2];

//     std::vector<cv::Rect> bboxes;
//     std::vector<float> scores;
//     std::vector<int> labels;
//     std::vector<int> indices;
//     std::vector<std::vector<float>> kpss;

//     cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVector.data());
//     output = output.t();

//     // Get all the YOLO proposals
//     for (int i = 0; i < numAnchors; i++) {
//         auto rowPtr = output.row(i).ptr<float>();
//         auto bboxesPtr = rowPtr;
//         auto scoresPtr = rowPtr + 4;
//         auto kps_ptr = rowPtr + 5;
//         float score = *scoresPtr;
//         if (score > PROBABILITY_THRESHOLD) {
//             float x = *bboxesPtr++;
//             float y = *bboxesPtr++;
//             float w = *bboxesPtr++;
//             float h = *bboxesPtr;

//             float x0 = std::clamp((x - 0.5f * w) * m_ratio_, 0.f, m_imgWidth_);
//             float y0 = std::clamp((y - 0.5f * h) * m_ratio_, 0.f, m_imgHeight_);
//             float x1 = std::clamp((x + 0.5f * w) * m_ratio_, 0.f, m_imgWidth_);
//             float y1 = std::clamp((y + 0.5f * h) * m_ratio_, 0.f, m_imgHeight_);

//             cv::Rect_<float> bbox;
//             bbox.x = x0;
//             bbox.y = y0;
//             bbox.width = x1 - x0;
//             bbox.height = y1 - y0;

//             std::vector<float> kps;
//             for (int k = 0; k < NUM_KPS; k++) {
//                 float kpsX = *(kps_ptr + 3 * k) * m_ratio_;
//                 float kpsY = *(kps_ptr + 3 * k + 1) * m_ratio_;
//                 float kpsS = *(kps_ptr + 3 * k + 2);
//                 kpsX       = std::clamp(kpsX, 0.f, m_imgWidth_);
//                 kpsY       = std::clamp(kpsY, 0.f, m_imgHeight_);
//                 kps.push_back(kpsX);
//                 kps.push_back(kpsY);
//                 kps.push_back(kpsS);
//             }

//             bboxes.push_back(bbox);
//             labels.push_back(0); // All detected objects are people
//             scores.push_back(score);
//             kpss.push_back(kps);
//         }
//     }

//     // Run NMS
//     cv::dnn::NMSBoxesBatched(bboxes, scores, labels, PROBABILITY_THRESHOLD, NMS_THRESHOLD, indices);

//     std::vector<Object> objects;

//     // Choose the top k detections
//     int cnt = 0;
//     for (auto& chosenIdx : indices) {
//         if (cnt >= TOP_K) {
//             break;
//         }

//         Object obj{};
//         obj.probability = scores[chosenIdx];
//         obj.label = labels[chosenIdx];
//         obj.rect = bboxes[chosenIdx];
//         obj.kps = kpss[chosenIdx];
//         objects.push_back(obj);

//         cnt += 1;
//     }

//     return objects;
// }

// std::vector<Object> YoloV8::postprocessDetect(std::vector<float> &featureVector) {
//     const auto& outputDims = m_trtEngine->getOutputDims();
//     auto numChannels = outputDims[0].d[1];
//     auto numAnchors = outputDims[0].d[2];

//     auto numClasses = CLASS_NAMES.size();

//     std::vector<cv::Rect> bboxes;
//     std::vector<float> scores;
//     std::vector<int> labels;
//     std::vector<int> indices;

//     cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVector.data());
//     output = output.t();

//     // Get all the YOLO proposals
//     for (int i = 0; i < numAnchors; i++) {
//         auto rowPtr = output.row(i).ptr<float>();
//         auto bboxesPtr = rowPtr;
//         auto scoresPtr = rowPtr + 4;
//         auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
//         float score = *maxSPtr;
//         if (score > PROBABILITY_THRESHOLD) {
//             float x = *bboxesPtr++;
//             float y = *bboxesPtr++;
//             float w = *bboxesPtr++;
//             float h = *bboxesPtr;

//             float x0 = std::clamp((x - 0.5f * w) * m_ratio_, 0.f, m_imgWidth_);
//             float y0 = std::clamp((y - 0.5f * h) * m_ratio_, 0.f, m_imgHeight_);
//             float x1 = std::clamp((x + 0.5f * w) * m_ratio_, 0.f, m_imgWidth_);
//             float y1 = std::clamp((y + 0.5f * h) * m_ratio_, 0.f, m_imgHeight_);

//             int label = maxSPtr - scoresPtr;
//             cv::Rect_<float> bbox;
//             bbox.x = x0;
//             bbox.y = y0;
//             bbox.width = x1 - x0;
//             bbox.height = y1 - y0;

//             bboxes.push_back(bbox);
//             labels.push_back(label);
//             scores.push_back(score);
//         }
//     }

//     // Run NMS
//     cv::dnn::NMSBoxesBatched(bboxes, scores, labels, PROBABILITY_THRESHOLD, NMS_THRESHOLD, indices);

//     std::vector<Object> objects;

//     // Choose the top k detections
//     int cnt = 0;
//     for (auto& chosenIdx : indices) {
//         if (cnt >= TOP_K) {
//             break;
//         }

//         Object obj{};
//         obj.probability = scores[chosenIdx];
//         obj.label = labels[chosenIdx];
//         obj.rect = bboxes[chosenIdx];
//         objects.push_back(obj);

//         cnt += 1;
//     }

//     return objects;
// }

/**
 * @brief Draws object labels on the given image.
 * 
 * This function takes an input image and a vector of detected objects, and draws bounding boxes, labels, and masks (if available) on the image.
 * 
 * @param image The input image on which the object labels will be drawn.
 * @param objects A vector of objects containing information about the objects detected in the image.
 * @param scale The scale factor to adjust the size of the thickness of the bounding box lines and font size of text.
 */
void YoloV8::drawObjectLabels(cv::Mat& image, const std::vector<Object> &objects, unsigned int scale) {
    // If segmentation information is present, start with that
    if (!objects.empty() && !objects[0].boxMask.empty()) {
        cv::Mat mask = image.clone();
        for (const auto& object: objects) {
            // Choose the color
            int colorIndex = object.label % COLOR_LIST.size(); // We have only defined 80 unique colors
            cv::Scalar color = cv::Scalar(COLOR_LIST[colorIndex][0],
                                          COLOR_LIST[colorIndex][1],
                                          COLOR_LIST[colorIndex][2]);

            // Add the mask for said object
            mask(object.rect).setTo(color * 255, object.boxMask);
        }
        // Add all the masks to our image
        cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
    }

    // Bounding boxes and annotations
    for (auto & object : objects) {
        // Choose the color
		int colorIndex = object.label % COLOR_LIST.size(); // We have only defined 80 unique colors
        cv::Scalar color = cv::Scalar(COLOR_LIST[colorIndex][0],
                                      COLOR_LIST[colorIndex][1],
                                      COLOR_LIST[colorIndex][2]);
        float meanColor = cv::mean(color)[0];
        cv::Scalar txtColor;
        if (meanColor > 0.5){
            txtColor = cv::Scalar(0, 0, 0);
        }else{
            txtColor = cv::Scalar(255, 255, 255);
        }

        const auto& rect = object.rect;

        // Draw rectangles and text
        char text[256];
        if (object.label + 1 > getNumClasses()) {
            throw std::runtime_error("Error: Label index exceeds the number of classes defined. Did you update yolov8.env to include all classes?");
        }

        sprintf(text, "%s %.1f%%", CLASS_NAMES[object.label].c_str(), object.probability * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = object.rect.x;
        int y = object.rect.y + 1;
        // std::cout << "Rect size: " << rect.size() << std::endl;
        // std::cout << "Image size: " << image.size() << std::endl;
        cv::rectangle(image, rect, color * 255, scale + 1);

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);

        // Pose estimation
        if (!object.kps.empty()) {
            auto& kps = object.kps;
            for (int k = 0; k < NUM_KPS + 2; k++) {
                if (k < NUM_KPS) {
                    int   kpsX = std::round(kps[k * 3]);
                    int   kpsY = std::round(kps[k * 3 + 1]);
                    float kpsS = kps[k * 3 + 2];
                    if (kpsS > KPS_THRESHOLD) {
                        cv::Scalar kpsColor = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                        cv::circle(image, {kpsX, kpsY}, 5, kpsColor, -1);
                    }
                }
                auto& ske    = SKELETON[k];
                int   pos1X = std::round(kps[(ske[0] - 1) * 3]);
                int   pos1Y = std::round(kps[(ske[0] - 1) * 3 + 1]);

                int pos2X = std::round(kps[(ske[1] - 1) * 3]);
                int pos2Y = std::round(kps[(ske[1] - 1) * 3 + 1]);

                float pos1S = kps[(ske[0] - 1) * 3 + 2];
                float pos2S = kps[(ske[1] - 1) * 3 + 2];

                if (pos1S > KPS_THRESHOLD && pos2S > KPS_THRESHOLD) {
                    cv::Scalar limbColor = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                    cv::line(image, {pos1X, pos1Y}, {pos2X, pos2Y}, limbColor, 2);
                }
            }
        }
    }
}

/**
 * @brief Place all the segmentation masks into a one channel image.
 * 
 * This function takes a vector of detected masks and places them into a single channel image.
 * 
 * @param objects A vector of detected instance segmentation objects.
 * @param segMaskOneChannel The output image containing all the masks in a single channel.
 * @param img_height The height of the original image.
 * @param img_width The width of the original image.
 */
void YoloV8::getOneChannelSegmentationMask(const std::vector<Object>& objects, cv::Mat& segMaskOneChannel, int img_height, int img_width) {
    segMaskOneChannel = cv::Mat::zeros(img_height, img_width, CV_8UC1);
    if (!objects.empty() && !objects[0].boxMask.empty()) {
        int i = 1;
        for (const auto& object: objects) {
            // Draw each segmentation mask on the oneChannelMask
            segMaskOneChannel(object.rect).setTo(i, object.boxMask);
            i++;
        }
    }
}

/**
 * @brief Function overload to draw object labels on the given image.
 * 
 * This function takes an input image, a vector of detected objects, and a vector to store masks
 * and draws bounding boxes, labels, and masks (if available) on the image and stores a binary mask in the given
 * masks vector for each instance segmentation object.
 * 
 * @param image The input image on which the object labels will be drawn.
 * @param objects A vector of objects containing information about the objects detected in the image.
 * @param masks A vector of binary masks representing each detected instance segmentation object.
 * @param scale The scale factor to adjust the size of the thickness of the bounding box lines and font size of text.
 */
void YoloV8::drawObjectLabels(cv::Mat& image, const std::vector<Object> &objects, std::vector<cv::Mat>& masks, unsigned int scale) {
    // If segmentation information is present, start with that
    if (!objects.empty() && !objects[0].boxMask.empty()) {
        cv::Mat mask = image.clone();
        for (const auto& object: objects) {
            // Draw masks on image to display to user
            // Choose the color
            int colorIndex = object.label % COLOR_LIST.size(); // We have only defined 80 unique colors
            cv::Scalar color = cv::Scalar(COLOR_LIST[colorIndex][0],
                                          COLOR_LIST[colorIndex][1],
                                          COLOR_LIST[colorIndex][2]);

            // Add the mask for said object
            mask(object.rect).setTo(color * 255, object.boxMask);

            // Create binary mask for each object
            cv::Mat binaryMask = cv::Mat::zeros(image.size(), CV_8UC3);
            // Set the binary mask to 1 where the object is
            binaryMask(object.rect).setTo(1, object.boxMask);
            masks.push_back(binaryMask);
        }
        // Add all the masks to our display image
        cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
    }

    // Bounding boxes and annotations
    for (auto & object : objects) {
        // Choose the color
		int colorIndex = object.label % COLOR_LIST.size(); // We have only defined 80 unique colors
        cv::Scalar color = cv::Scalar(COLOR_LIST[colorIndex][0],
                                      COLOR_LIST[colorIndex][1],
                                      COLOR_LIST[colorIndex][2]);
        float meanColor = cv::mean(color)[0];
        cv::Scalar txtColor;
        if (meanColor > 0.5){
            txtColor = cv::Scalar(0, 0, 0);
        }else{
            txtColor = cv::Scalar(255, 255, 255);
        }

        const auto& rect = object.rect;

        // Draw rectangles and text
        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[object.label].c_str(), object.probability * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = object.rect.x;
        int y = object.rect.y + 1;

        cv::rectangle(image, rect, color * 255, scale + 1);

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);

        // Pose estimation
        if (!object.kps.empty()) {
            auto& kps = object.kps;
            for (int k = 0; k < NUM_KPS + 2; k++) {
                if (k < NUM_KPS) {
                    int   kpsX = std::round(kps[k * 3]);
                    int   kpsY = std::round(kps[k * 3 + 1]);
                    float kpsS = kps[k * 3 + 2];
                    if (kpsS > KPS_THRESHOLD) {
                        cv::Scalar kpsColor = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                        cv::circle(image, {kpsX, kpsY}, 5, kpsColor, -1);
                    }
                }
                auto& ske    = SKELETON[k];
                int   pos1X = std::round(kps[(ske[0] - 1) * 3]);
                int   pos1Y = std::round(kps[(ske[0] - 1) * 3 + 1]);

                int pos2X = std::round(kps[(ske[1] - 1) * 3]);
                int pos2Y = std::round(kps[(ske[1] - 1) * 3 + 1]);

                float pos1S = kps[(ske[0] - 1) * 3 + 2];
                float pos2S = kps[(ske[1] - 1) * 3 + 2];

                if (pos1S > KPS_THRESHOLD && pos2S > KPS_THRESHOLD) {
                    cv::Scalar limbColor = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                    cv::line(image, {pos1X, pos1Y}, {pos2X, pos2Y}, limbColor, 2);
                }
            }
        }
    }
}