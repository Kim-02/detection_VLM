#include "yolo_trt.h"
#include "risk_analyzer.h"
#include "image_resize.h"

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


    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[오류] curl 초기화 실패\n";
        return false;
    }

    curl_mime* mime = curl_mime_init(curl);
    if (!mime) {
        std::cerr << "[오류] mime 초기화 실패\n";
        curl_easy_cleanup(curl);
        return false;
    }

    curl_mimepart* part = nullptr;

    part = curl_mime_addpart(mime);
    curl_mime_name(part, "file");
    curl_mime_filedata(part, image_path.c_str());
    curl_mime_filename(part, "test.jpg");
    curl_mime_type(part, "image/jpeg");

    part = curl_mime_addpart(mime);
    curl_mime_name(part, "prompt");
    curl_mime_data(part, prompt.c_str(), CURL_ZERO_TERMINATED);

    part = curl_mime_addpart(mime);
    curl_mime_name(part, "detections");
    curl_mime_data(part, detections.c_str(), CURL_ZERO_TERMINATED);

    part = curl_mime_addpart(mime);
    curl_mime_name(part, "class_map_json");
    curl_mime_data(part, class_map_json.c_str(), CURL_ZERO_TERMINATED);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);

    const CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        std::cerr << "[오류] 서버 요청 실패: " << curl_easy_strerror(res) << "\n";
        curl_mime_free(mime);
        curl_easy_cleanup(curl);
        return false;
    }

    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);


    curl_mime_free(mime);
    curl_easy_cleanup(curl);
    return true;
}

int main() {
    const std::string engine_path = "best.engine";

    // 원본 이미지
    const std::string input_image_path = "test.jpg";

    // 리사이즈 후 저장할 이미지
    const std::string resized_image_path = "resized_664.jpg";

    const std::string infer_url = "http://127.0.0.1:8000/infer";

    std::cout << "==============================\n";
    std::cout << "[시작] YOLO + 위험 분석 + 서버 전송 파이프라인 시작\n";
    std::cout << "==============================\n";

    // 1) 입력 이미지를 664x664로 리사이즈
    if (!ImageResizer::resizeTo664AndSave(input_image_path, resized_image_path)) {
        std::cerr << "[오류] 리사이즈 실패\n";
        return 1;
    }

    // 2) YOLO 로더 준비
    YoloTrtDetector detector(640, 640, 0.25f, 0.45f);

    if (!detector.loadEngine(engine_path)) {
        std::cerr << "[오류] 엔진 로드 실패\n";
        return 1;
    }
    // 3) 리사이즈된 이미지를 기준으로 YOLO 추론
    std::vector<Detection> dets;
    if (!detector.inferImage(resized_image_path, dets)) {
        std::cerr << "[오류] 이미지 추론 실패\n";
        return 1;
    }
    std::cout << "[정보] 탐지된 객체 수: " << dets.size() << "\n";

    const std::string detections_text = detector.detectionsOnlyText(dets);
    if (ENABLE_DEBUG_LOG) {
        std::cout << "----- detectionsOnlyText -----\n";
        std::cout << detections_text << "\n";
        std::cout << "------------------------------\n";
    }

    // 4) PPE 위험 분석
    RiskAnalyzer analyzer(0.35f);

    const std::vector<PersonRiskResult> risk_results = analyzer.analyzePPE(dets);
    const std::string risk_text = analyzer.resultsToText(risk_results);
    std::cout << "[정보] 분석된 person 수: " << risk_results.size() << "\n";

    if (ENABLE_DEBUG_LOG) {
        std::cout << "----- PPE analysis -----\n";
        std::cout << risk_text << "\n";
        std::cout << "------------------------\n";
    }

    // 5) 서버 전송용 프롬프트
    const std::string prompt =
        "최종 결과만 한국어로 1문장 또는 최대 2문장으로 작성하세요. "
        "작업자 수, 안전모 착용 여부, 조끼 착용 여부, 화재 여부만 간단히 요약하세요. "
        "화재가 없으면 화재에 대해 굳이 언급하지 마세요. "
        "위험 상황이 명확하지 않으면 불필요한 위험 문장은 쓰지 마세요.";

    const std::string class_map_json = buildClassMapJson();

    // 6) 리사이즈된 이미지를 서버로 전송
    long http_code = 0;
    std::string response_body;

    if (!sendToInferServer(
            infer_url,
            resized_image_path,
            prompt,
            detections_text,
            class_map_json,
            http_code,
            response_body)) {
        std::cerr << "[오류] 추론 서버 요청 실패\n";
        return 1;
    }

    std::cout << "[응답] 서버 응답 본문:\n" << response_body << "\n";

    std::cout << "==============================\n";
    std::cout << "[종료] 전체 파이프라인 정상 종료\n";
    std::cout << "==============================\n";

    return 0;
}