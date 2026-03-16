#include "yolo_trt.h"

#include <iostream>
#include <vector>

int main() {
    YoloTrtDetector detector(640, 640, 0.25f, 0.45f);

    if (!detector.loadEngine("best.engine")) {
        std::cerr << "engine load failed\n";
        return 1;
    }

    std::vector<Detection> dets;
    if (!detector.inferImage("test.jpg", dets)) {
        std::cerr << "infer failed\n";
        return 1;
    }

    std::string text = detector.detectionsToText(dets);
    std::cout << text;

    return 0;
}