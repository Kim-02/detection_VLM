#pragma once

#include <string>
#include <opencv2/opencv.hpp>

class ImageResizer {
public:
    // 이미지를 읽어서 664x664로 리사이즈 후 반환
    static cv::Mat resizeTo664(const std::string& imagePath);

    // 이미 로드된 Mat를 664x664로 리사이즈 후 반환
    static cv::Mat resizeTo664(const cv::Mat& input);

    // 이미지를 읽어서 664x664로 저장
    static bool resizeTo664AndSave(const std::string& inputPath, const std::string& outputPath);

private:
    static constexpr int TARGET_W = 640;
    static constexpr int TARGET_H = 640;
};