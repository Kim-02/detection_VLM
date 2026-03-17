#include "yolo_trt.h"
#include "risk_analyzer.h"
#include "image_resize.h"

#include <curl/curl.h>

#include <iostream>
#include <string>
#include <vector>

static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    const size_t total = size * nmemb;
    std::string chunk(static_cast<char*>(contents), total);

    // 누적 저장
    std::string* s = static_cast<std::string*>(userp);
    s->append(chunk);

    // 즉시 출력
    std::cout << chunk << std::flush;

    return total;
}

static bool sendToInferServer(
    const std::string& url,
    const std::string& image_path,
    const std::string& scene_summary,
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
    curl_mime_filename(part, "input.jpg");
    curl_mime_type(part, "image/jpeg");

    part = curl_mime_addpart(mime);
    curl_mime_name(part, "prompt");
    curl_mime_data(part, "", CURL_ZERO_TERMINATED);

    part = curl_mime_addpart(mime);
    curl_mime_name(part, "detections");
    curl_mime_data(part, scene_summary.c_str(), CURL_ZERO_TERMINATED);

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
    const std::string input_image_path = "test.jpg";
    const std::string resized_image_path = "resized_512.jpg";
    const std::string infer_url = "http://127.0.0.1:8000/infer";

    std::cout << "[시작] YOLO + PPE 분석 + 서버 전송\n";

    if (!ImageResizer::resizeTo512AndSave(input_image_path, resized_image_path)) {
        std::cerr << "[오류] 이미지 리사이즈 실패\n";
        return 1;
    }

    YoloTrtDetector detector(512, 512, 0.25f, 0.45f);

    if (!detector.loadEngine(engine_path)) {
        std::cerr << "[오류] TensorRT 엔진 로드 실패\n";
        return 1;
    }

    std::vector<Detection> dets;
    if (!detector.inferImage(resized_image_path, dets)) {
        std::cerr << "[오류] YOLO 추론 실패\n";
        return 1;
    }

    RiskAnalyzer analyzer(0.35f);

    const std::vector<PersonRiskResult> risk_results = analyzer.analyzePPE(dets);
    const SceneRiskSummary scene_summary = analyzer.summarizeScene(risk_results);
    const std::string scene_summary_text = analyzer.sceneSummaryToText(scene_summary);

    std::cout << "[정보] 작업자 수: " << scene_summary.worker_count << "\n";
    std::cout << "[정보] 안전모 미착용 의심 수: " << scene_summary.no_helmet_count << "\n";
    std::cout << "[정보] 조끼 미착용 의심 수: " << scene_summary.no_vest_count << "\n";

    long http_code = 0;
    std::string response_body;

    if (!sendToInferServer(
            infer_url,
            resized_image_path,
            scene_summary_text,
            http_code,
            response_body)) {
        std::cerr << "[오류] 추론 서버 요청 실패\n";
        return 1;
    }

    if (http_code != 200) {
        std::cerr << "[오류] 서버 응답 코드: " << http_code << "\n";
        std::cerr << response_body << "\n";
        return 1;
    }

    return 0;
}