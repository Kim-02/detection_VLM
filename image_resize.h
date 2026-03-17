#pragma once

#include <string>
#include <opencv2/opencv.hpp>

class ImageResizer {
public:
    static cv::Mat resizeTo448(const std::string& imagePath);
    static cv::Mat resizeTo448(const cv::Mat& input);
    static bool resizeTo448AndSave(const std::string& inputPath, const std::string& outputPath);

private:
    static constexpr int TARGET_W = 448;
    static constexpr int TARGET_H = 448;
};