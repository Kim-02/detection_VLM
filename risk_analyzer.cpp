#include "risk_analyzer.h"

#include <algorithm>
#include <sstream>

RiskAnalyzer::RiskAnalyzer(float contain_ratio_thresh)
    : contain_ratio_thresh_(contain_ratio_thresh) {}

float RiskAnalyzer::intersectionArea(const Detection& a, const Detection& b) const {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    return w * h;
}

float RiskAnalyzer::boxArea(const Detection& b) const {
    return std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
}

float RiskAnalyzer::centerX(const Detection& b) const {
    return 0.5f * (b.x1 + b.x2);
}

float RiskAnalyzer::centerY(const Detection& b) const {
    return 0.5f * (b.y1 + b.y2);
}

bool RiskAnalyzer::pointInBox(float x, float y, const Detection& box) const {
    return (x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2);
}

// helmet는 머리 윗부분 중앙만 보도록 더 좁힘
Detection RiskAnalyzer::makeHelmetRegion(const Detection& person) const {
    float w = person.x2 - person.x1;
    float h = person.y2 - person.y1;

    Detection region;
    region.x1 = person.x1 + 0.28f * w;
    region.x2 = person.x2 - 0.28f * w;
    region.y1 = person.y1;
    region.y2 = person.y1 + 0.26f * h;
    region.conf = 1.0f;
    region.class_id = -10;
    return region;
}

// vest는 몸통 중앙 영역으로 제한
Detection RiskAnalyzer::makeVestRegion(const Detection& person) const {
    float w = person.x2 - person.x1;
    float h = person.y2 - person.y1;

    Detection region;
    region.x1 = person.x1 + 0.22f * w;
    region.x2 = person.x2 - 0.22f * w;
    region.y1 = person.y1 + 0.30f * h;
    region.y2 = person.y1 + 0.72f * h;
    region.conf = 1.0f;
    region.class_id = -11;
    return region;
}

float RiskAnalyzer::calcOverlapRatio(const Detection& region, const Detection& item) const {
    float inter = intersectionArea(region, item);
    float item_area = boxArea(item);
    if (item_area <= 1e-6f) return 0.0f;
    return inter / item_area;
}

float RiskAnalyzer::calcCenterDxRatio(const Detection& person, const Detection& item) const {
    float pw = std::max(1e-6f, person.x2 - person.x1);
    return std::abs(centerX(person) - centerX(item)) / pw;
}

RegionMatchDebug RiskAnalyzer::evaluateHelmetCandidate(
    const Detection& person,
    const Detection& region,
    const Detection& helmet
) const {
    RegionMatchDebug d{};
    d.item_box = helmet;
    d.overlap_ratio = calcOverlapRatio(region, helmet);
    d.center_dx_ratio = calcCenterDxRatio(person, helmet);
    d.center_inside_region = pointInBox(centerX(helmet), centerY(helmet), region);

    // helmet는 더 엄격하게
    bool overlap_ok = d.overlap_ratio >= contain_ratio_thresh_;
    bool center_ok = d.center_inside_region;
    bool align_ok = d.center_dx_ratio <= 0.22f;

    d.accepted = overlap_ok && center_ok && align_ok;

    // 점수: overlap 우선 + 중심 정렬 가중치
    d.score = d.overlap_ratio * 0.7f + (1.0f - std::min(d.center_dx_ratio, 1.0f)) * 0.3f;
    return d;
}

RegionMatchDebug RiskAnalyzer::evaluateVestCandidate(
    const Detection& person,
    const Detection& region,
    const Detection& vest
) const {
    RegionMatchDebug d{};
    d.item_box = vest;
    d.overlap_ratio = calcOverlapRatio(region, vest);
    d.center_dx_ratio = calcCenterDxRatio(person, vest);
    d.center_inside_region = pointInBox(centerX(vest), centerY(vest), region);

    bool overlap_ok = d.overlap_ratio >= contain_ratio_thresh_;
    bool center_ok = d.center_inside_region;
    bool align_ok = d.center_dx_ratio <= 0.28f;

    d.accepted = overlap_ok && center_ok && align_ok;
    d.score = d.overlap_ratio * 0.65f + (1.0f - std::min(d.center_dx_ratio, 1.0f)) * 0.35f;
    return d;
}

std::vector<PersonRiskResult> RiskAnalyzer::analyzePPE(const std::vector<Detection>& dets) const {
    std::vector<Detection> persons;
    std::vector<Detection> helmets;
    std::vector<Detection> vests;

    for (const auto& d : dets) {
        if (d.class_id == 2) persons.push_back(d);
        else if (d.class_id == 0) helmets.push_back(d);
        else if (d.class_id == 1) vests.push_back(d);
    }

    std::vector<PersonRiskResult> results;

    for (const auto& person : persons) {
        Detection helmet_region = makeHelmetRegion(person);
        Detection vest_region = makeVestRegion(person);

        PersonRiskResult r;
        r.person_box = person;
        r.debug.person_box = person;
        r.debug.helmet_region = helmet_region;
        r.debug.vest_region = vest_region;

        float best_helmet_score = -1.0f;
        float best_vest_score = -1.0f;

        for (const auto& helmet : helmets) {
            RegionMatchDebug dbg = evaluateHelmetCandidate(person, helmet_region, helmet);
            r.debug.helmet_candidates.push_back(dbg);
            if (dbg.accepted && dbg.score > best_helmet_score) {
                best_helmet_score = dbg.score;
            }
        }

        for (const auto& vest : vests) {
            RegionMatchDebug dbg = evaluateVestCandidate(person, vest_region, vest);
            r.debug.vest_candidates.push_back(dbg);
            if (dbg.accepted && dbg.score > best_vest_score) {
                best_vest_score = dbg.score;
            }
        }

        r.has_helmet = (best_helmet_score >= 0.0f);
        r.has_vest = (best_vest_score >= 0.0f);
        r.is_risk = (!r.has_helmet || !r.has_vest);

        if (!r.has_helmet && !r.has_vest) {
            r.warning_message = "경고: 작업자가 안전모와 조끼를 모두 착용하지 않았습니다.";
        } else if (!r.has_helmet) {
            r.warning_message = "경고: 작업자가 안전모를 착용하지 않았습니다.";
        } else if (!r.has_vest) {
            r.warning_message = "경고: 작업자가 조끼를 착용하지 않았습니다.";
        } else {
            r.warning_message = "정상: 보호구 착용 상태가 확인되었습니다.";
        }

        // 보기 좋게 점수순 정렬
        std::sort(r.debug.helmet_candidates.begin(), r.debug.helmet_candidates.end(),
                  [](const RegionMatchDebug& a, const RegionMatchDebug& b) {
                      return a.score > b.score;
                  });

        std::sort(r.debug.vest_candidates.begin(), r.debug.vest_candidates.end(),
                  [](const RegionMatchDebug& a, const RegionMatchDebug& b) {
                      return a.score > b.score;
                  });

        results.push_back(r);
    }

    return results;
}

bool RiskAnalyzer::hasAnyRisk(const std::vector<PersonRiskResult>& results) const {
    for (const auto& r : results) {
        if (r.is_risk) return true;
    }
    return false;
}

std::string RiskAnalyzer::resultsToText(const std::vector<PersonRiskResult>& results) const {
    std::ostringstream oss;
    oss << "PPE Risk Analysis: " << results.size() << " persons checked\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        oss << "[" << i << "] "
            << "person_box=("
            << r.person_box.x1 << ", " << r.person_box.y1 << ", "
            << r.person_box.x2 << ", " << r.person_box.y2 << ") "
            << "helmet=" << (r.has_helmet ? "yes" : "no") << " "
            << "vest=" << (r.has_vest ? "yes" : "no") << " "
            << "risk=" << (r.is_risk ? "yes" : "no") << " "
            << "msg=\"" << r.warning_message << "\"\n";
    }

    return oss.str();
}

std::string RiskAnalyzer::debugResultsToText(const std::vector<PersonRiskResult>& results) const {
    std::ostringstream oss;

    oss << "===== PPE DEBUG INFO =====\n";
    oss << "contain_ratio_thresh = " << contain_ratio_thresh_ << "\n\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];

        oss << "Person[" << i << "]\n";
        oss << "  person_box = (" << r.debug.person_box.x1 << ", " << r.debug.person_box.y1
            << ", " << r.debug.person_box.x2 << ", " << r.debug.person_box.y2 << ")\n";

        oss << "  helmet_region = (" << r.debug.helmet_region.x1 << ", " << r.debug.helmet_region.y1
            << ", " << r.debug.helmet_region.x2 << ", " << r.debug.helmet_region.y2 << ")\n";

        oss << "  vest_region   = (" << r.debug.vest_region.x1 << ", " << r.debug.vest_region.y1
            << ", " << r.debug.vest_region.x2 << ", " << r.debug.vest_region.y2 << ")\n";

        oss << "  helmet_candidates:\n";
        if (r.debug.helmet_candidates.empty()) {
            oss << "    (none)\n";
        } else {
            for (size_t j = 0; j < r.debug.helmet_candidates.size(); ++j) {
                const auto& c = r.debug.helmet_candidates[j];
                oss << "    [" << j << "] "
                    << "box=(" << c.item_box.x1 << ", " << c.item_box.y1
                    << ", " << c.item_box.x2 << ", " << c.item_box.y2 << ") "
                    << "overlap_ratio=" << c.overlap_ratio << " "
                    << "center_dx_ratio=" << c.center_dx_ratio << " "
                    << "center_inside=" << (c.center_inside_region ? "yes" : "no") << " "
                    << "score=" << c.score << " "
                    << "accepted=" << (c.accepted ? "yes" : "no") << "\n";
            }
        }

        oss << "  vest_candidates:\n";
        if (r.debug.vest_candidates.empty()) {
            oss << "    (none)\n";
        } else {
            for (size_t j = 0; j < r.debug.vest_candidates.size(); ++j) {
                const auto& c = r.debug.vest_candidates[j];
                oss << "    [" << j << "] "
                    << "box=(" << c.item_box.x1 << ", " << c.item_box.y1
                    << ", " << c.item_box.x2 << ", " << c.item_box.y2 << ") "
                    << "overlap_ratio=" << c.overlap_ratio << " "
                    << "center_dx_ratio=" << c.center_dx_ratio << " "
                    << "center_inside=" << (c.center_inside_region ? "yes" : "no") << " "
                    << "score=" << c.score << " "
                    << "accepted=" << (c.accepted ? "yes" : "no") << "\n";
            }
        }

        oss << "  result: helmet=" << (r.has_helmet ? "yes" : "no")
            << ", vest=" << (r.has_vest ? "yes" : "no")
            << ", risk=" << (r.is_risk ? "yes" : "no") << "\n";

        oss << "  message: " << r.warning_message << "\n\n";
    }

    return oss.str();
}