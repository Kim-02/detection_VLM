#include "image_resize.h"

#include <iostream>
#include <stdexcept>

cv::Mat ImageResizer::resizeTo512(const std::string& imagePath) {
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        throw std::runtime_error("이미지 로드 실패: " + imagePath);
    }

    return resizeTo512(img);
}

cv::Mat ImageResizer::resizeTo512(const cv::Mat& input) {
    if (input.empty()) {
        throw std::runtime_error("입력 이미지가 비어 있습니다.");
    }

    cv::Mat resized;
    cv::resize(input, resized, cv::Size(TARGET_W, TARGET_H), 0, 0, cv::INTER_LINEAR);
    return resized;
}

bool ImageResizer::resizeTo512AndSave(const std::string& inputPath, const std::string& outputPath) {
    try {
        cv::Mat resized = resizeTo512(inputPath);
        if (!cv::imwrite(outputPath, resized)) {
            std::cerr << "리사이즈 이미지 저장 실패: " << outputPath << std::endl;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ImageResizer] 오류: " << e.what() << std::endl;
        return false;
    }
}