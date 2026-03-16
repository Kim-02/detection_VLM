#include "yolo_trt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace nvinfer1;

YoloTrtDetector::YoloTrtDetector(int input_w, int input_h, float conf_thresh, float nms_thresh)
    : input_w_(input_w), input_h_(input_h),
      conf_thresh_(conf_thresh), nms_thresh_(nms_thresh) {}

YoloTrtDetector::~YoloTrtDetector() {
    cleanup();
}

void YoloTrtDetector::Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

void YoloTrtDetector::cleanup() {
    if (context_) {
        delete context_;
        context_ = nullptr;
    }
    if (engine_) {
        delete engine_;
        engine_ = nullptr;
    }
    if (runtime_) {
        delete runtime_;
        runtime_ = nullptr;
    }
}

size_t YoloTrtDetector::getSizeByDim(const Dims& dims) const {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}

std::vector<char> YoloTrtDetector::readEngineFileUltralyticsAware(const std::string& path) const {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + path);
    }

    file.seekg(0, std::ios::end);
    std::streamoff file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size <= 0) {
        throw std::runtime_error("Engine file is empty.");
    }

    uint32_t meta_len = 0;
    file.read(reinterpret_cast<char*>(&meta_len), sizeof(meta_len));
    if (!file) {
        throw std::runtime_error("Failed to read engine header.");
    }

    std::streamoff possible_engine_offset =
        static_cast<std::streamoff>(sizeof(uint32_t)) + static_cast<std::streamoff>(meta_len);

    if (meta_len < 1024 * 1024 && possible_engine_offset < file_size) {
        file.seekg(possible_engine_offset, std::ios::beg);

        std::vector<char> engine_data(static_cast<size_t>(file_size - possible_engine_offset));
        file.read(engine_data.data(), engine_data.size());

        if (file && file.gcount() == static_cast<std::streamsize>(engine_data.size())) {
            std::cout << "Detected Ultralytics metadata header. "
                      << "meta_len=" << meta_len
                      << ", skipping " << possible_engine_offset
                      << " bytes before TensorRT engine." << std::endl;
            return engine_data;
        }
    }

    file.clear();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(static_cast<size_t>(file_size));
    file.read(buffer.data(), buffer.size());
    if (!file) {
        throw std::runtime_error("Failed to read engine file.");
    }

    std::cout << "No Ultralytics metadata header detected. Reading full file." << std::endl;
    return buffer;
}

bool YoloTrtDetector::loadEngine(const std::string& engine_path) {
    cleanup();

    try {
        std::vector<char> engine_data = readEngineFileUltralyticsAware(engine_path);

        runtime_ = createInferRuntime(logger_);
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }

        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize engine.");
        }

        context_ = engine_->createExecutionContext();
        if (!context_) {
            throw std::runtime_error("Failed to create execution context.");
        }

        int nbIOTensors = engine_->getNbIOTensors();
        const char* input_name = nullptr;
        const char* output_name = nullptr;

        for (int i = 0; i < nbIOTensors; ++i) {
            const char* name = engine_->getIOTensorName(i);
            auto mode = engine_->getTensorIOMode(name);
            if (mode == TensorIOMode::kINPUT) input_name = name;
            else output_name = name;
        }

        if (!input_name || !output_name) {
            throw std::runtime_error("Failed to find input/output tensor names.");
        }

        input_name_ = input_name;
        output_name_ = output_name;

        Dims input_dims = engine_->getTensorShape(input_name_.c_str());
        Dims output_dims = engine_->getTensorShape(output_name_.c_str());

        std::cout << "Input tensor: " << input_name_ << " dims=";
        for (int i = 0; i < input_dims.nbDims; ++i) std::cout << input_dims.d[i] << " ";
        std::cout << "\n";

        std::cout << "Output tensor: " << output_name_ << " dims=";
        for (int i = 0; i < output_dims.nbDims; ++i) std::cout << output_dims.d[i] << " ";
        std::cout << "\n";

        return true;
    } catch (const std::exception& e) {
        std::cerr << "loadEngine ERROR: " << e.what() << "\n";
        cleanup();
        return false;
    }
}

YoloTrtDetector::LetterboxInfo YoloTrtDetector::preprocessLetterbox(
    const unsigned char* src, int src_w, int src_h, float* dst) const {
    float r = std::min((float)input_w_ / src_w, (float)input_h_ / src_h);
    int new_w = (int)std::round(src_w * r);
    int new_h = (int)std::round(src_h * r);

    int pad_x = (input_w_ - new_w) / 2;
    int pad_y = (input_h_ - new_h) / 2;

    for (int i = 0; i < 3 * input_w_ * input_h_; ++i) {
        dst[i] = 114.0f / 255.0f;
    }

    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            int src_x = std::min((int)(x / r), src_w - 1);
            int src_y = std::min((int)(y / r), src_h - 1);

            int src_idx = (src_y * src_w + src_x) * 3;
            int dst_x = x + pad_x;
            int dst_y = y + pad_y;

            int dst_idx_r = 0 * input_w_ * input_h_ + dst_y * input_w_ + dst_x;
            int dst_idx_g = 1 * input_w_ * input_h_ + dst_y * input_w_ + dst_x;
            int dst_idx_b = 2 * input_w_ * input_h_ + dst_y * input_w_ + dst_x;

            dst[dst_idx_r] = src[src_idx + 0] / 255.0f;
            dst[dst_idx_g] = src[src_idx + 1] / 255.0f;
            dst[dst_idx_b] = src[src_idx + 2] / 255.0f;
        }
    }

    return {r, pad_x, pad_y, new_w, new_h};
}

float YoloTrtDetector::iou(const Detection& a, const Detection& b) const {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;

    float areaA = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float areaB = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);

    return inter / (areaA + areaB - inter + 1e-6f);
}

std::vector<Detection> YoloTrtDetector::nms(std::vector<Detection>& dets) const {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.conf > b.conf;
    });

    std::vector<Detection> result;
    std::vector<bool> removed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j]) continue;
            if (dets[i].class_id == dets[j].class_id && iou(dets[i], dets[j]) > nms_thresh_) {
                removed[j] = true;
            }
        }
    }
    return result;
}

std::vector<Detection> YoloTrtDetector::decodeYoloOutput(
    const float* output,
    int num_preds,
    int num_classes,
    const LetterboxInfo& lb,
    int orig_w,
    int orig_h
) const {
    std::vector<Detection> dets;

    int stride0 = num_preds;
    const float* p_x   = output + 0 * stride0;
    const float* p_y   = output + 1 * stride0;
    const float* p_w   = output + 2 * stride0;
    const float* p_h   = output + 3 * stride0;
    const float* p_cls = output + 4 * stride0;

    for (int i = 0; i < num_preds; ++i) {
        float best_conf = 0.0f;
        int best_cls = -1;

        for (int c = 0; c < num_classes; ++c) {
            float conf = p_cls[c * stride0 + i];
            if (conf > best_conf) {
                best_conf = conf;
                best_cls = c;
            }
        }

        if (best_conf < conf_thresh_) continue;

        float cx = p_x[i];
        float cy = p_y[i];
        float w  = p_w[i];
        float h  = p_h[i];

        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

        x1 = (x1 - lb.pad_x) / lb.scale;
        y1 = (y1 - lb.pad_y) / lb.scale;
        x2 = (x2 - lb.pad_x) / lb.scale;
        y2 = (y2 - lb.pad_y) / lb.scale;

        x1 = std::max(0.0f, std::min(x1, (float)(orig_w - 1)));
        y1 = std::max(0.0f, std::min(y1, (float)(orig_h - 1)));
        x2 = std::max(0.0f, std::min(x2, (float)(orig_w - 1)));
        y2 = std::max(0.0f, std::min(y2, (float)(orig_h - 1)));

        if (x2 <= x1 || y2 <= y1) continue;

        dets.push_back({x1, y1, x2, y2, best_conf, best_cls});
    }

    return nms(dets);
}

bool YoloTrtDetector::inferImage(const std::string& image_path, std::vector<Detection>& out_dets) {
    out_dets.clear();

    if (!engine_ || !context_) {
        std::cerr << "inferImage ERROR: engine is not loaded.\n";
        return false;
    }

    int img_w = 0, img_h = 0, img_c = 0;
    unsigned char* img = stbi_load(image_path.c_str(), &img_w, &img_h, &img_c, 3);
    if (!img) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        return false;
    }

    void* input_dev = nullptr;
    void* output_dev = nullptr;
    cudaStream_t stream = nullptr;

    try {
        Dims input_dims = engine_->getTensorShape(input_name_.c_str());
        Dims output_dims = engine_->getTensorShape(output_name_.c_str());

        if (input_dims.nbDims == 4 && input_dims.d[0] == -1) {
            Dims4 real_input_dims(1, 3, input_h_, input_w_);
            if (!context_->setInputShape(input_name_.c_str(), real_input_dims)) {
                throw std::runtime_error("Failed to set input shape.");
            }
            input_dims = context_->getTensorShape(input_name_.c_str());
            output_dims = context_->getTensorShape(output_name_.c_str());
        }

        size_t input_size = getSizeByDim(input_dims);
        size_t output_size = getSizeByDim(output_dims);

        std::vector<float> input_host(input_size);
        std::vector<float> output_host(output_size);

        LetterboxInfo lb = preprocessLetterbox(img, img_w, img_h, input_host.data());

        int num_classes = output_dims.d[1] - 4;
        int num_preds = output_dims.d[2];

        cudaError_t status = cudaMalloc(&input_dev, input_size * sizeof(float));
        if (status != cudaSuccess) throw std::runtime_error("cudaMalloc failed for input.");

        status = cudaMalloc(&output_dev, output_size * sizeof(float));
        if (status != cudaSuccess) throw std::runtime_error("cudaMalloc failed for output.");

        status = cudaStreamCreate(&stream);
        if (status != cudaSuccess) throw std::runtime_error("cudaStreamCreate failed.");

        status = cudaMemcpyAsync(input_dev, input_host.data(),
                                 input_size * sizeof(float),
                                 cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess) throw std::runtime_error("cudaMemcpyAsync H2D failed.");

        if (!context_->setTensorAddress(input_name_.c_str(), input_dev)) {
            throw std::runtime_error("Failed to bind input tensor.");
        }
        if (!context_->setTensorAddress(output_name_.c_str(), output_dev)) {
            throw std::runtime_error("Failed to bind output tensor.");
        }

        if (!context_->enqueueV3(stream)) {
            throw std::runtime_error("Inference failed.");
        }

        status = cudaMemcpyAsync(output_host.data(), output_dev,
                                 output_size * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream);
        if (status != cudaSuccess) throw std::runtime_error("cudaMemcpyAsync D2H failed.");

        status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) throw std::runtime_error("cudaStreamSynchronize failed.");

        out_dets = decodeYoloOutput(output_host.data(), num_preds, num_classes, lb, img_w, img_h);
    } catch (const std::exception& e) {
        std::cerr << "inferImage ERROR: " << e.what() << "\n";

        if (stream) cudaStreamDestroy(stream);
        if (input_dev) cudaFree(input_dev);
        if (output_dev) cudaFree(output_dev);
        stbi_image_free(img);
        return false;
    }

    if (stream) cudaStreamDestroy(stream);
    if (input_dev) cudaFree(input_dev);
    if (output_dev) cudaFree(output_dev);
    stbi_image_free(img);
    return true;
}

std::string YoloTrtDetector::detectionsToText(const std::vector<Detection>& dets) const {
    std::ostringstream oss;
    oss << "Detections: " << dets.size() << "\n";
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        oss << "[" << i << "] "
            << "class=" << d.class_id
            << " conf=" << d.conf
            << " box=(" << d.x1 << ", " << d.y1
            << ", " << d.x2 << ", " << d.y2 << ")\n";
    }
    return oss.str();
}