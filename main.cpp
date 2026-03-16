#include "yolo_trt.h"
#include "risk_analyzer.h"

#include <curl/curl.h>

#include <iostream>
#include <string>
#include <vector>

static const bool ENABLE_DEBUG_LOG = true;

static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    const size_t total = size * nmemb;
    std::string* s = static_cast<std::string*>(userp);
    s->append(static_cast<char*>(contents), total);
    return total;
}

static std::string buildClassMapJson() {
    return R"({
  "0": "helmet",
  "1": "vest",
  "2": "person",
  "3": "danger_vehicle"
})";
}

static bool sendToInferServer(
    const std::string& url,
    const std::string& image_path,
    const std::string& prompt,
    const std::string& detections,
    const std::string& class_map_json,
    long& http_code,
    std::string& response_body
) {
    http_code = 0;
    response_body.clear();

    std::cout << "[진행] 추론 서버 요청 준비 시작\n";

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[오류] curl 초기화 실패\n";
        return false;
    }

    curl_mime* mime = curl_mime_init(curl);
    curl_mimepart* part = nullptr;

    std::cout << "[진행] 이미지 파일 첨부 중: " << image_path << "\n";
    part = curl_mime_addpart(mime);
    curl_mime_name(part, "file");
    curl_mime_filedata(part, image_path.c_str());
    curl_mime_filename(part, "test.jpg");
    curl_mime_type(part, "image/jpeg");

    std::cout << "[진행] 프롬프트 첨부 중\n";
    part = curl_mime_addpart(mime);
    curl_mime_name(part, "prompt");
    curl_mime_data(part, prompt.c_str(), CURL_ZERO_TERMINATED);

    std::cout << "[진행] detection 텍스트 첨부 중\n";
    part = curl_mime_addpart(mime);
    curl_mime_name(part, "detections");
    curl_mime_data(part, detections.c_str(), CURL_ZERO_TERMINATED);

    std::cout << "[진행] 클래스 맵 첨부 중\n";
    part = curl_mime_addpart(mime);
    curl_mime_name(part, "class_map_json");
    curl_mime_data(part, class_map_json.c_str(), CURL_ZERO_TERMINATED);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);

    std::cout << "[진행] 추론 서버로 요청 전송 시작: " << url << "\n";
    const CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        std::cerr << "[오류] 서버 요청 실패: " << curl_easy_strerror(res) << "\n";
        curl_mime_free(mime);
        curl_easy_cleanup(curl);
        return false;
    }

    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    std::cout << "[진행] 서버 응답 수신 완료\n";

    curl_mime_free(mime);
    curl_easy_cleanup(curl);
    return true;
}

int main() {
    const std::string engine_path = "best.engine";
    const std::string image_path = "test.jpg";
    const std::string infer_url = "http://127.0.0.1:8000/infer";

    std::cout << "==============================\n";
    std::cout << "[시작] YOLO + 위험 분석 + 서버 전송 파이프라인 시작\n";
    std::cout << "==============================\n";

    YoloTrtDetector detector(640, 640, 0.25f, 0.45f);

    std::cout << "[진행] TensorRT 엔진 로드 시작: " << engine_path << "\n";
    if (!detector.loadEngine(engine_path)) {
        std::cerr << "[오류] 엔진 로드 실패\n";
        return 1;
    }
    std::cout << "[완료] TensorRT 엔진 로드 완료\n";

    std::vector<Detection> dets;
    std::cout << "[진행] 이미지 추론 시작: " << image_path << "\n";
    if (!detector.inferImage(image_path, dets)) {
        std::cerr << "[오류] 이미지 추론 실패\n";
        return 1;
    }
    std::cout << "[완료] 이미지 추론 완료\n";
    std::cout << "[정보] 탐지된 객체 수: " << dets.size() << "\n";

    std::cout << "[진행] detection 텍스트 생성 중\n";
    const std::string detections_text = detector.detectionsOnlyText(dets);
    std::cout << "[완료] detection 텍스트 생성 완료\n";

    RiskAnalyzer analyzer(0.35f);

    std::cout << "[진행] PPE 위험 분석 시작\n";
    const std::vector<PersonRiskResult> risk_results = analyzer.analyzePPE(dets);
    const std::string risk_text = analyzer.resultsToText(risk_results);
    std::cout << "[완료] PPE 위험 분석 완료\n";
    std::cout << "[정보] 분석된 person 수: " << risk_results.size() << "\n";

    if (ENABLE_DEBUG_LOG) {
        std::cout << "------------------------------\n";
        std::cout << "[디버그] Detection 결과\n";
        std::cout << detections_text << "\n";

        std::cout << "------------------------------\n";
        std::cout << "[디버그] PPE 분석 결과\n";
        std::cout << risk_text << "\n";
        std::cout << "------------------------------\n";
    }

    const std::string prompt = "장면에 대한 분석을 하되, YOLO에서 찾은 객체를 기준으로 설명해줘.";
    const std::string class_map_json = buildClassMapJson();

    long http_code = 0;
    std::string response_body;

    std::cout << "[진행] 서버 전송 단계 시작\n";
    if (!sendToInferServer(
            infer_url,
            image_path,
            prompt,
            detections_text,
            class_map_json,
            http_code,
            response_body)) {
        std::cerr << "[오류] 추론 서버 요청 실패\n";
        return 1;
    }
    std::cout << "[완료] 서버 전송 단계 완료\n";

    std::cout << "[정보] HTTP 응답 코드: " << http_code << "\n";
    std::cout << "[정보] 서버 응답 본문:\n";
    std::cout << response_body << "\n";

    std::cout << "==============================\n";
    std::cout << "[종료] 전체 파이프라인 정상 종료\n";
    std::cout << "==============================\n";

    return 0;
}