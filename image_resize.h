#pragma once

#include <string>
#include <opencv2/opencv.hpp>

class ImageResizer {
public:
    static cv::Mat resizeTo640(const std::string& imagePath);
    static cv::Mat resizeTo640(const cv::Mat& input);
    static bool resizeTo640AndSave(const std::string& inputPath, const std::string& outputPath);

private:
    static constexpr int TARGET_W = 640;
    static constexpr int TARGET_H = 640;
};