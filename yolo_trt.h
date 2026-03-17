#ifndef YOLO_TRT_H
#define YOLO_TRT_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>

struct Detection {
    float x1, y1, x2, y2;
    float conf;
    int class_id;
};

class YoloTrtDetector {
public:
    YoloTrtDetector(int input_w = 512, int input_h = 512,
                    float conf_thresh = 0.25f, float nms_thresh = 0.45f);
    ~YoloTrtDetector();

    bool loadEngine(const std::string& engine_path);
    bool inferImage(const std::string& image_path, std::vector<Detection>& out_dets);

    std::string detectionsOnlyText(const std::vector<Detection>& dets) const;

    int getInputW() const { return input_w_; }
    int getInputH() const { return input_h_; }

private:
    struct LetterboxInfo {
        float scale;
        int pad_x;
        int pad_y;
        int new_w;
        int new_h;
    };

    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };

private:
    size_t getSizeByDim(const nvinfer1::Dims& dims) const;
    std::vector<char> readEngineFileUltralyticsAware(const std::string& path) const;

    LetterboxInfo preprocessLetterbox(
        const unsigned char* src,
        int src_w,
        int src_h,
        float* dst
    ) const;

    float iou(const Detection& a, const Detection& b) const;
    std::vector<Detection> nms(std::vector<Detection>& dets) const;

    std::vector<Detection> decodeYoloOutput(
        const float* output,
        int num_preds,
        int num_classes,
        const LetterboxInfo& lb,
        int orig_w,
        int orig_h
    ) const;

    void cleanup();

private:
    int input_w_;
    int input_h_;
    float conf_thresh_;
    float nms_thresh_;

    Logger logger_;

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    std::string input_name_;
    std::string output_name_;
};



#endif