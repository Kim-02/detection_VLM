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

Detection RiskAnalyzer::makeHelmetRegion(const Detection& person) const {
    float w = person.x2 - person.x1;
    float h = person.y2 - person.y1;

    Detection region;
    region.x1 = person.x1 + 0.32f * w;
    region.x2 = person.x2 - 0.32f * w;
    region.y1 = person.y1;
    region.y2 = person.y1 + 0.27f * h;
    region.conf = 1.0f;
    region.class_id = -10;
    return region;
}

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

bool RiskAnalyzer::isHelmetMatch(
    const Detection& person,
    const Detection& region,
    const Detection& helmet
) const {
    const float overlap_ratio = calcOverlapRatio(region, helmet);
    const float center_dx_ratio = calcCenterDxRatio(person, helmet);
    const bool center_inside = pointInBox(centerX(helmet), centerY(helmet), region);

    const bool overlap_ok = overlap_ratio >= contain_ratio_thresh_;
    const bool align_ok = center_dx_ratio <= 0.18f;
    const bool strong_overlap_ok = overlap_ratio >= 0.45f;

    return overlap_ok && align_ok && (center_inside || strong_overlap_ok);
}

bool RiskAnalyzer::isVestMatch(
    const Detection& person,
    const Detection& region,
    const Detection& vest
) const {
    const float overlap_ratio = calcOverlapRatio(region, vest);
    const float center_dx_ratio = calcCenterDxRatio(person, vest);
    const bool center_inside = pointInBox(centerX(vest), centerY(vest), region);

    const bool overlap_ok = overlap_ratio >= contain_ratio_thresh_;
    const bool align_ok = center_dx_ratio <= 0.28f;

    return overlap_ok && align_ok && center_inside;
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
        const Detection helmet_region = makeHelmetRegion(person);
        const Detection vest_region = makeVestRegion(person);

        bool has_helmet = false;
        bool has_vest = false;

        for (const auto& helmet : helmets) {
            if (isHelmetMatch(person, helmet_region, helmet)) {
                has_helmet = true;
                break;
            }
        }

        for (const auto& vest : vests) {
            if (isVestMatch(person, vest_region, vest)) {
                has_vest = true;
                break;
            }
        }

        PersonRiskResult r;
        r.person_box = person;
        r.has_helmet = has_helmet;
        r.has_vest = has_vest;
        r.is_risk = (!has_helmet || !has_vest);

        if (!has_helmet && !has_vest) {
            r.warning_message = "경고: 작업자가 안전모와 조끼를 모두 착용하지 않았습니다.";
        } else if (!has_helmet) {
            r.warning_message = "경고: 작업자가 안전모를 착용하지 않았습니다.";
        } else if (!has_vest) {
            r.warning_message = "경고: 작업자가 조끼를 착용하지 않았습니다.";
        } else {
            r.warning_message = "정상: 보호구 착용 상태가 확인되었습니다.";
        }

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