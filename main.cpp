#include "yolo_trt.h"
#include "risk_analyzer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

int main() {
    const std::string engine_path = "best.engine";
    const std::string image_path = "test.jpg";
    const std::string saved_risk_path = "risk_frame.jpg";
    const std::string debug_text_path = "risk_debug.txt";

    YoloTrtDetector detector(640, 640, 0.25f, 0.45f);

    if (!detector.loadEngine(engine_path)) {
        std::cerr << "engine load failed\n";
        return 1;
    }

    std::vector<Detection> dets;
    if (!detector.inferImage(image_path, dets)) {
        std::cerr << "infer failed\n";
        return 1;
    }

    std::cout << detector.detectionsToText(dets) << "\n";

    RiskAnalyzer analyzer(0.4f);
    std::vector<PersonRiskResult> risk_results = analyzer.analyzePPE(dets);

    std::cout << analyzer.resultsToText(risk_results) << "\n";

    std::string debug_text = analyzer.debugResultsToText(risk_results);
    std::cout << debug_text;

    {
        std::ofstream ofs(debug_text_path);
        ofs << debug_text;
    }
    std::cout << "디버그 텍스트 저장 완료: " << debug_text_path << "\n";

    if (analyzer.hasAnyRisk(risk_results)) {
        std::cout << "=== PPE 위험 감지 ===\n";

        for (const auto& r : risk_results) {
            if (r.is_risk) {
                std::cout << r.warning_message << "\n";
            }
        }

        try {
            fs::copy_file(image_path, saved_risk_path, fs::copy_options::overwrite_existing);
            std::cout << "위험 프레임 저장 완료: " << saved_risk_path << "\n";
        } catch (const std::exception& e) {
            std::cerr << "프레임 저장 실패: " << e.what() << "\n";
        }
    } else {
        std::cout << "위험 없음\n";
    }

    return 0;
}