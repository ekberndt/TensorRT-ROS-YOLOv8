#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <iterator>
#include <opencv2/cudaimgproc.hpp>
#include "engine.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;
using namespace Util;

std::vector<std::string> Util::getFilesInDirectory(const std::string& dirPath) {
    std::vector<std::string> filepaths;
    for (const auto& entry: std::filesystem::directory_iterator(dirPath)) {
        filepaths.emplace_back(entry.path());
    }
    return filepaths;
}

/**
 * Creates the logs that TensorRT needs for the Builder, ONNX Parser, and Runtime interfaces.
 *
 * @param severity The serverity level of the log message.
 * @param msg The message to log.
 */
void Logger::log(Severity severity, const char *msg) noexcept {
    // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
    // For the sake of this tutorial, will just log to the console.

    // Only log Warnings or more important.
    // TODO: make this a param
    if (severity <= Severity::kVERBOSE) {
    // if (severity <= Severity::kVERBOSE) {
        std::cout << msg << std::endl;
    }
}

Engine::Engine(const Options &options)
    : m_options(options) {}

/**
 * Builds the TensorRT engine by converted the specified ONNX model
 * using the ONNX Parser and creating a Builder to build the engine
 * with engine options.
 * 
 * @param onnxModelPath The path to the ONNX model file.
 * @param subVals The array of subtraction values used for input preprocessing.
 * @param divVals The array of division values used for input preprocessing.
 * @param normalize Flag indicating whether input normalization should be applied.
 * @return True if the engine is successfully built and saved, false otherwise.
 * @throws std::runtime_error if the model file is not found or if there are errors during engine generation.
 */
bool Engine::build(std::string onnxModelPath, const std::array<float, 3>& subVals, const std::array<float, 3>& divVals,
                   bool normalize) {
    m_subVals = subVals;
    m_divVals = divVals;
    m_normalize = normalize;

    // Get models directory from onnxModelPath
    std::string modelsDir = onnxModelPath.substr(0, onnxModelPath.find_last_of("/"));

    // Only regenerate the engine file if it has not already been generated for the specified options
    m_engineName = serializeEngineOptions(m_options, onnxModelPath);
    batch_size_ = m_options.maxBatchSize;
    std::cout << "Searching for engine file with name: " << m_engineName << " in directory: " << modelsDir << std::endl;

    // Create Engine path
    std::string engine_path = modelsDir + "/engines/";
    if (!std::filesystem::exists(engine_path)) {
        std::filesystem::create_directory(engine_path);
    }
    engine_path += m_engineName;

    if (doesFileExist(engine_path)) {
        std::cout << "Engine found, not regenerating..." << std::endl;
        return true;
    }

    if (!doesFileExist(onnxModelPath)) {
        throw std::runtime_error("Could not find model at path: " + onnxModelPath);
    }

    // Was not able to find the engine file, generate...
    std::cout << "Engine not found, generating. This could take a while..." << std::endl;

    // Create the TensorRT Build
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Set kEXPLICIT_BATCH flag via bit shifting for the NetworkDefinition as the ONNX PARSER does not support implicit batch sizes
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // Create the NetworkDefinition for the Build and specify the explicit batch flag
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer to the parser.
    // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.
    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Parse the buffer we read into memory.
    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Ensure that all the inputs have the same batch size
    const auto numInputs = network->getNbInputs();
    std::cout << "Model has " << numInputs << " input(s)" << std::endl;
    if (numInputs < 1) {
        throw std::runtime_error("Error, model needs at least 1 input!");
    }

    const auto input0Batch = network->getInput(0)->getDimensions().d[0];
    for (int32_t i = 1; i < numInputs; ++i) {
        if (network->getInput(i)->getDimensions().d[0] != input0Batch) {
            throw std::runtime_error("Error, the model has multiple inputs, each with differing batch sizes!");
        }
    }

    // Ensure the imported ONNX model supports the max batch size specified or supports dynamic batching
    if (m_options.maxBatchSize > input0Batch && input0Batch != -1) {
        throw std::runtime_error("Error, imported ONNX model does not support max batch size of " +
                                std::to_string(m_options.maxBatchSize) + ". The ONNX model only supports a max batch size of " +
                                std::to_string(input0Batch) + ".");
    }

    // Print the batch size information
    if (input0Batch == -1) {
        std::cout << "Imported ONNX model supports dynamic batch sizes" << std::endl;
    } else if (input0Batch == 1) {
        std::cout << "Imported ONNX model has a fixed batch size of 1" << std::endl;
    } else {
        std::cout << "Imported ONNX model has a fixed batch size of " << input0Batch << std::endl;
    }

    // Create a builder configuration
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Register a single optimization profile
    IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    for (int32_t i = 0; i < numInputs; ++i) {
        // Must specify dimensions for all the inputs the model expects.
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t inputC = inputDims.d[1];
        int32_t inputH = inputDims.d[2];
        int32_t inputW = inputDims.d[3];

        std::cout << "TensorRT input buffer " << i << " dimensions: " << inputC << ", " << inputH << ", " << inputW << std::endl;

        // To set the dimensions for the optimization profile, we need to specify the min, opt, and max dimensions (for dynamic batching)
        // TODO: Why does this only work if we set the dimensions for the input to the MAX value?
        optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
    }
    config->addOptimizationProfile(optProfile);

    // Set the precision level
    if (m_options.precision == Precision::FP16) {
        // Ensure the GPU supports FP16 inference
        if (!builder->platformHasFastFp16()) {
            throw std::runtime_error("Error: GPU does not support FP16 precision");
        }
        config->setFlag(BuilderFlag::kFP16);
    } else if (m_options.precision == Precision::INT8) {
        if (numInputs > 1) {
            throw std::runtime_error("Error, this implementation currently only supports INT8 quantization for single input models");
        }

        // Ensure the GPU supports INT8 Quantization
        if (!builder->platformHasFastInt8()) {
            throw std::runtime_error("Error: GPU does not support INT8 precision");
        }

        // Ensure the user has provided path to calibration data directory
        if (m_options.calibrationDataDirectoryPath.empty()) {
            throw std::runtime_error("Error: If INT8 precision is selected, must provide path to calibration data directory to Engine::build method");
        }

        config->setFlag((BuilderFlag::kINT8));

        const auto input = network->getInput(0);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        const auto calibrationFileName = m_engineName + ".calibration";

        m_calibrator = std::make_unique<Int8EntropyCalibrator2>(m_options.calibrationBatchSize, inputDims.d[3], inputDims.d[2], m_options.calibrationDataDirectoryPath,
                                                                calibrationFileName, inputName, subVals, divVals, normalize);
        config->setInt8Calibrator(m_calibrator.get());
    }
    std::cout << "Precision set to " << (m_options.precision == Precision::FP16 ? "FP16" : "INT8") << std::endl;

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    // Build the engine
    // If this call fails, it is suggested to increase the logger verbosity to kVERBOSE and try rebuilding the engine.
    // Doing so will provide you with more information on why exactly it is failing.
    std::cout << "Starting engine build..." << std::endl;

    // TODO: why is this line causing the program to crash with exit code -6?
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        std::cout << "Error, engine build failed!" << std::endl;
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(engine_path, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << m_engineName << std::endl;

    checkCudaErrorCode(cudaStreamDestroy(profileStream));
    return true;
}

Engine::~Engine() {
    // Free the GPU memory
    for (auto & buffer : m_buffers) {
        checkCudaErrorCode(cudaFree(buffer));
    }

    m_buffers.clear();
}

/**
 * @brief Deserializes engine from disk, creates a TensorRT ExecutionContext, and allocates tensors
 *      on the GPU for input and output buffers.
 *
 * @param onnxModelPath The path to the ONNX model file directory.
 * @return true if the network is loaded successfully, false otherwise.
 * @throws std::runtime_error if there is an error reading the engine file or setting the GPU device index.
 */
bool Engine::loadNetwork(std::string onnxModelPath) {
    // Get engine path from onnxModelPath
    std::string modelsDir = onnxModelPath.substr(0, onnxModelPath.find_last_of("/"));
    std::string engine_path = modelsDir + "/engines/" + m_engineName;

    // Read the serialized model from disk
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Create a runtime to deserialize the engine file.
    m_runtime = std::unique_ptr<IRuntime> {createInferRuntime(m_logger)};
    if (!m_runtime) {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    // Create an engine, a representation of the optimized model.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    // The execution context contains all of the state associated with a particular invocation
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    m_buffers.resize(m_engine->getNbIOTensors());

    // Create a cuda stream
    cudaStream_t stream;
    checkCudaErrorCode(cudaStreamCreate(&stream));

    // Allocate GPU memory for input and output buffers
    m_outputLengthsFloat.clear();
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const auto tensorName = m_engine->getIOTensorName(i);
        m_IOTensorNames.emplace_back(tensorName);
        const auto tensorType = m_engine->getTensorIOMode(tensorName);
        const auto tensorShape = m_engine->getTensorShape(tensorName);
        if (tensorType == TensorIOMode::kINPUT) {
            // Allocate memory for the input
            // Allocate enough to fit the max batch size (we could end up using less later)
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], m_options.maxBatchSize * tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3] * sizeof(float), stream));

            // Store the input dims for later use
            m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
        } else if (tensorType == TensorIOMode::kOUTPUT) {
            // The binding is an output
            uint32_t outputLenFloat = 1;
            m_outputDims.push_back(tensorShape);

            for (int j = 1; j < tensorShape.nbDims; ++j) {
                // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
                outputLenFloat *= tensorShape.d[j];
            }

            m_outputLengthsFloat.push_back(outputLenFloat);
            // Now size the output buffer appropriately, taking into account the max possible batch size (although we could actually end up using less memory)
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLenFloat * m_options.maxBatchSize * sizeof(float), stream));
        } else {
            throw std::runtime_error("Error, IO Tensor is neither an input or output!");
        }
    }

    // Synchronize and destroy the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(stream));
    checkCudaErrorCode(cudaStreamDestroy(stream));

    return true;
}

bool Engine::runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs, std::vector<std::vector<std::vector<float>>>& featureVectors) {
    // First we do some error checking
    if (inputs.empty() || inputs[0].empty()) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Provided input vector is empty!" << std::endl;
        return false;
    }


    // Check if the number of input buffers matches the number of input buffers
    const auto numInputs = m_inputDims.size();
    // if (inputs.size() != numInputs) { // Unbatched version
    // TODO: Does this only work for the batched version?
    // TODO: This check does not work for the batched version as it is enforcing the batch size is 1
    // if (inputs[0].size() != numInputs) {
    //     std::cout << "===== Error =====" << std::endl;
    //     std::cout << "Incorrect number of inputs provided!" << std::endl;
    //     std::cout << "Expected: " << numInputs << " inputs" << std::endl;
    //     std::cout << "Got: " << inputs[0].size() << " inputs" << std::endl;
    //     return false;
    // }

    // Ensure the batch size does not exceed the max
    if (inputs.size() > static_cast<size_t>(m_options.maxBatchSize)) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "The batch size is larger than the model expects!" << std::endl;
        std::cout << "Model max batch size: " << m_options.maxBatchSize << std::endl;
        std::cout << "Batch size provided to call to runInference: " << inputs.size() << std::endl;
        return false;
    }

    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].size() != inputs[0].size()) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "The batch size needs to be constant for all inputs!" << std::endl;
            return false;
        }
    }

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    // Iterate through the inputs (there should only be one input)
    for (size_t i = 0; i < numInputs; ++i) {
        std::vector<cv::cuda::GpuMat> input_batches = inputs[i];
        const auto batchSize = static_cast<int32_t>(input_batches.size());
        const auto& dims = m_inputDims[i];

        // Check the dimentions of the first batched image are the same size as the Engine dims
        // This does not check any other images in the batch
        const auto& batch0 = input_batches[0];
        if (batch0.channels() != dims.d[0] ||
            batch0.rows != dims.d[1] ||
            batch0.cols != dims.d[2]) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "Input does not have correct size!" << std::endl;
            std::cout << "Expected: (" << dims.d[0] << ", " << dims.d[1] << ", "
                    << dims.d[2] << ")" << std::endl;
            std::cout << "Got: (" << batch0.channels() << ", " << batch0.rows << ", " << batch0.cols << ")" << std::endl;
            std::cout << "Ensure you resize your input image to the correct size" << std::endl;
            return false;
        }

        nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
        m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims); // Define the batch size
        // OpenCV reads images into memory in NHWC format, while TensorRT expects images in NCHW format. 
        // The following method converts NHWC to NCHW.
        // Even though TensorRT expects NCHW at IO, during optimization, it can internally use NHWC to optimize cuda kernels
        // See: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-layout
        // Copy over the input data and perform the preprocessing
        // Convert the batch of images into a single contiguous blob
        cv::cuda::GpuMat mfloat = blobFromGpuMats(input_batches, m_subVals, m_divVals, m_normalize);
        auto *dataPointer = mfloat.ptr<void>();

        // Copy the input blob tensor into the GPU input buffer
        // Input blob size
        const auto blob_size = mfloat.cols * mfloat.rows * sizeof(float);
        checkCudaErrorCode(cudaMemcpyAsync(m_buffers[i], dataPointer,
                                        // mfloat.cols * mfloat.rows * mfloat.channels() * sizeof(float),
                                        blob_size,
                                        cudaMemcpyDeviceToDevice, inferenceCudaStream));
    }

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "Error setting tensor address for: " << m_IOTensorNames[i].c_str() << std::endl;
            return false;
        }
    }

    // Run inference.
    std::cout << "Running enqueueV3..." << std::endl;
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Error calling enqueueV3!" << std::endl;
        return false;
    }

    // TODO: is this batched version?
    // Copy the outputs from GPU's output buffer back to CPU
    // Output shape of the model is [batch][output][feature_vector]
    featureVectors.clear();

    // TODO: fix batchSize
    const auto batchSize = static_cast<int32_t>(inputs[0].size());
    for (int batch = 0; batch < batchSize; ++batch) {
        // Batch
        std::vector<std::vector<float>> outputs{};
        // Iterate through the output buffers (as defined by the bindings of the optimization profile(s))
        // Start at index m_inputDims.size() to account for the inputs in our m_buffers
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbBindings(); ++outputBinding) {
            std::vector<float> output;
            auto outputLenFloat = m_outputLengthsFloat[outputBinding - numInputs];
            output.resize(outputLenFloat);
            // Copy the output tensor from the GPU to the CPU
            checkCudaErrorCode(cudaMemcpyAsync(output.data(), static_cast<char*>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLenFloat), outputLenFloat * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream));
            outputs.emplace_back(std::move(output));
        }
        featureVectors.emplace_back(std::move(outputs));
    }

    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
    return true;
}


/**
 * Converts a batch of GPU mats (a batch of tensors) to a single GPU contiguous mat blob in GPU memory (one tensor).
 *
 * @param batches The vector of GPU mats representing the input batches.
 * @param subVals The array of three float values used for mean subtraction.
 * @param divVals The array of three float values used for scaling.
 * @param normalize A boolean flag indicating whether to normalize the output.
 * @return The GPU mat blob representing the converted batch.
 */
cv::cuda::GpuMat Engine::blobFromGpuMats(const std::vector<cv::cuda::GpuMat>& batches, const std::array<float, 3>& subVals, const std::array<float, 3>& divVals, bool normalize) {
    int batch_size = batches.size();
    int channels = batches[0].channels();
    int height = batches[0].rows;
    int width = batches[0].cols;

    size_t img_size = height * width * channels;
    cv::cuda::GpuMat gpu_dst(batch_size, img_size, CV_8UC3);

    size_t img_channel_size = width * height;
    for (size_t batch = 0; batch < batch_size; batch++) {
        cv::cuda::GpuMat input_channels[channels];
        for (int j = 0; j < channels; j++) {
            input_channels[j] = cv::cuda::GpuMat(height, width, CV_8UC1, &(gpu_dst.ptr()[img_size * batch + img_channel_size * j]));
        }
        // std::vector<cv::cuda::GpuMat> input_channels{
        //         cv::cuda::GpuMat(batchInput.rows, batchInput.cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * batch])),
        //         cv::cuda::GpuMat(batchInput.rows, batchInput.cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * batch])),
        //         cv::cuda::GpuMat(batchInput.rows, batchInput.cols, CV_8U,
        //                          &(gpu_dst.ptr()[width * 2 + width * 3 * batch]))
        // };
        cv::cuda::split(batches[batch], input_channels);  // HWC -> CHW
    }

    // Normalize the images
    cv::cuda::GpuMat mfloat;
    if (normalize) {
        // [0.f, 1.f]
        gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
    } else {
        // [0.f, 255.f]
        gpu_dst.convertTo(mfloat, CV_32FC3);
    }

    // Apply scaling and mean subtraction
    cv::cuda::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat, cv::noArray(), -1);
    cv::cuda::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat, 1, -1);

    return mfloat;
}

// cv::cuda::GpuMat Engine::blobFromGpuMats(const std::vector<cv::cuda::GpuMat>& batches, const std::array<float, 3>& subVals, const std::array<float, 3>& divVals, bool normalize) {
//     int batchSize = batches.size();
//     int channels = batches[0].channels();
//     int height = batches[0].rows;
//     int width = batches[0].cols;

//     cv::cuda::GpuMat gpu_dst(batchSize, height * width * channels, CV_8UC1);

//     size_t size = height * width * channels;
//     for (int i = 0; i < batchSize; i++) {
//         cv::cuda::GpuMat batchInput = batches[i];
//         cv::cuda::GpuMat input_channels[channels];
//         for (int j = 0; j < channels; j++) {
//             input_channels[j] = cv::cuda::GpuMat(height, width, CV_8UC1, &(gpu_dst.ptr()[size * i + size * j]));
//         }
//         cv::cuda::split(batchInput, input_channels);  // HWC -> CHW
//     }

//     cv::cuda::GpuMat mfloat;
//     if (normalize) {
//         // [0.f, 1.f]
//         gpu_dst.convertTo(mfloat, CV_32FC1, 1.f / 255.f);
//     } else {
//         // [0.f, 255.f]
//         gpu_dst.convertTo(mfloat, CV_32FC1);
//     }

//     // Apply scaling and mean subtraction
//     cv::cuda::subtract(mfloat, cv::Scalar(subVals[0]), mfloat, cv::noArray(), -1);
//     cv::cuda::divide(mfloat, cv::Scalar(divVals[0]), mfloat, 1, -1);

//     return mfloat;
// }

std::string Engine::serializeEngineOptions(const Options &options, const std::string& onnxModelPath) {
    const auto filenamePos = onnxModelPath.find_last_of('/') + 1;
    std::string engineName = onnxModelPath.substr(filenamePos, onnxModelPath.find_last_of('.') - filenamePos);

    // Add the GPU device name to the file to ensure that the model is only used on devices with the exact same GPU
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);

    if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
        throw std::runtime_error("Error, provided device index is out of range!");
    }

    auto deviceName = deviceNames[options.deviceIndex];
    // Remove spaces from the device name
    deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());

    engineName+= "." + deviceName;

    // Serialize the specified options into the filename
    if (options.precision == Precision::FP16) {
        engineName += "fp16";
    } else if (options.precision == Precision::FP32){
        engineName += "fp32";
    } else {
        engineName += "int8";
    }

    engineName += "maxBatchSize" + std::to_string(options.maxBatchSize);
    engineName += "optimalBatchSize" + std::to_string(options.optBatchSize);
    engineName += ".engine";

    return engineName;
}

void Engine::getDeviceNames(std::vector<std::string>& deviceNames) {
    int numGPUs;
    // TODO: why is numGPUs 825112889 -> should only be 1?
    // TODO: this is causing program to crash with exit code -9 as the large for loop iterates below
    // cudaGetDeviceCount(&numGPUs);
    cudaError_t ret = cudaGetDeviceCount(&numGPUs);

    if (ret != 0) {
        // Throw the cuda error message
        throw std::runtime_error("Error, cuda_runtime_api.h could not determine number of CUDA-capable devices. "
                                    "Restart your computer. Error thrown by cuda_runtime_api.h: " + std::string(cudaGetErrorString(ret)));
    }

    for (int device=0; device<numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        deviceNames.push_back(std::string(prop.name));
    }
}

/**
 * Resizes the input image while maintaining the aspect ratio and pads the right and bottom sides if necessary.
 *
 * @param input The input image to be resized.
 * @param height The desired height of the output image.
 * @param width The desired width of the output image.
 * @param bgcolor The background color to be used for padding.
 * @return The resized image with the specified dimensions and padding.
 */
cv::cuda::GpuMat Engine::resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width, const cv::Scalar &bgcolor) {
    // Calculate a scaling factor to maintain the aspect ratio
    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    // Resize the image to the new dimensions
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    // Create a new image with the desired dimensions and fill it with the background color
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    // Copy the resized image to the top left corner of the new image with padding and background color
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

/**
 * Removes the batch dimention from the output of the model. Should only be used when the model has a batch size of 1.
 * 
 * @param input The input feature vector to be moved.
 * @param output The output vector to store the feature vector.
 * @throws std::logic_error if the input feature vector has incorrect dimensions.
 */
void Engine::transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<float>>& output) {
    if (input.size() != 1) {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0]);
}

void Engine::transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output) {
    if (input.size() != 1 || input[0].size() != 1) {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0][0]);
}

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int32_t batchSize, int32_t inputW, int32_t inputH,
                                               const std::string &calibDataDirPath,
                                               const std::string &calibTableName,
                                               const std::string &inputBlobName,
                                               const std::array<float, 3>& subVals,
                                               const std::array<float, 3>& divVals,
                                               bool normalize,
                                               bool readCache)
        : m_batchSize(batchSize)
        , m_inputW(inputW)
        , m_inputH(inputH)
        , m_imgIdx(0)
        , m_calibTableName(calibTableName)
        , m_inputBlobName(inputBlobName)
        , m_subVals(subVals)
        , m_divVals(divVals)
        , m_normalize(normalize)
        , m_readCache(readCache) {

    // Allocate GPU memory to hold the entire batch
    m_inputCount = 3 * inputW * inputH * batchSize;
    checkCudaErrorCode(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));

    // Read the name of all the files in the specified directory.
    if (!doesFileExist(calibDataDirPath)) {
        throw std::runtime_error("Error, directory at provided path does not exist: " + calibDataDirPath);
    }

    m_imgPaths = getFilesInDirectory(calibDataDirPath);
    if (m_imgPaths.size() < static_cast<size_t>(batchSize)) {
        throw std::runtime_error("There are fewer calibration images than the specified batch size!");
    }

    // Randomize the calibration data
    auto rd = std::random_device {};
    auto rng = std::default_random_engine { rd() };
    std::shuffle(std::begin(m_imgPaths), std::end(m_imgPaths), rng);
}

int32_t Int8EntropyCalibrator2::getBatchSize() const noexcept {
    // Return the batch size
    return m_batchSize;
}

bool Int8EntropyCalibrator2::getBatch(void **bindings, const char **names, int32_t nbBindings) noexcept {
    // This method will read a batch of images into GPU memory, and place the pointer to the GPU memory in the bindings variable.

    if (m_imgIdx + m_batchSize > static_cast<int>(m_imgPaths.size())) {
        // There are not enough images left to satisfy an entire batch
        return false;
    }

    // Read the calibration images into memory for the current batch
    std::vector<cv::cuda::GpuMat> inputImgs;
    for (int i = m_imgIdx; i < m_imgIdx + m_batchSize; i++) {
        std::cout << "Reading image " << i << ": " << m_imgPaths[i] << std::endl;
        auto cpuImg = cv::imread(m_imgPaths[i]);
        if (cpuImg.empty()){
            std::cout << "Fatal error: Unable to read image at path: " << m_imgPaths[i] << std::endl;
            return false;
        }

        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(cpuImg);
        cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);

        // TODO: Define any preprocessing code here, such as resizing
        auto resized = Engine::resizeKeepAspectRatioPadRightBottom(gpuImg, m_inputH, m_inputW);

        inputImgs.emplace_back(std::move(resized));
    }

    // Convert the batch from NHWC to NCHW
    // ALso apply normalization, scaling, and mean subtraction
    auto mfloat = Engine::blobFromGpuMats(inputImgs, m_subVals, m_divVals, m_normalize);
    auto *dataPointer = mfloat.ptr<void>();

    // Copy the GPU buffer to member variable so that it persists
    checkCudaErrorCode(cudaMemcpyAsync(m_deviceInput, dataPointer, m_inputCount * sizeof(float), cudaMemcpyDeviceToDevice));

    m_imgIdx+= m_batchSize;
    if (std::string(names[0]) != m_inputBlobName) {
        std::cout << "Error: Incorrect input name provided!" << std::endl;
        return false;
    }
    bindings[0] = m_deviceInput;
    return true;
}

void const *Int8EntropyCalibrator2::readCalibrationCache(size_t &length) noexcept {
    std::cout << "Searching for calibration cache: " << m_calibTableName << std::endl;
    m_calibCache.clear();
    std::ifstream input(m_calibTableName, std::ios::binary);
    input >> std::noskipws;
    if (m_readCache && input.good()) {
        std::cout << "Reading calibration cache: " << m_calibTableName << std::endl;
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(m_calibCache));
    }
    length = m_calibCache.size();
    return length ? m_calibCache.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void *ptr, std::size_t length) noexcept {
    std::cout << "Writing calib cache: " << m_calibTableName << " Size: " << length << " bytes" << std::endl;
    std::ofstream output(m_calibTableName, std::ios::binary);
    output.write(reinterpret_cast<const char*>(ptr), length);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2() {
    checkCudaErrorCode(cudaFree(m_deviceInput));
};

