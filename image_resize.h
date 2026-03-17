#pragma once

#include <string>
#include <opencv2/opencv.hpp>

class ImageResizer {
public:
    static cv::Mat resizeTo512(const std::string& imagePath);
    static cv::Mat resizeTo512(const cv::Mat& input);
    static bool resizeTo512AndSave(const std::string& inputPath, const std::string& outputPath);

private:
    static constexpr int TARGET_W = 512;
    static constexpr int TARGET_H = 512;
};