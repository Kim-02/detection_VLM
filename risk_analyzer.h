#ifndef RISK_ANALYZER_H
#define RISK_ANALYZER_H

#include "yolo_trt.h"
#include <string>
#include <vector>

struct RegionMatchDebug {
    Detection item_box;
    float overlap_ratio;
    float center_dx_ratio;
    bool center_inside_region;
    float score;
    bool accepted;
};

struct PersonRiskDebugInfo {
    Detection person_box;
    Detection helmet_region;
    Detection vest_region;

    std::vector<RegionMatchDebug> helmet_candidates;
    std::vector<RegionMatchDebug> vest_candidates;
};

struct PersonRiskResult {
    Detection person_box;
    bool has_helmet;
    bool has_vest;
    bool is_risk;
    std::string warning_message;

    PersonRiskDebugInfo debug;
};

class RiskAnalyzer {
public:
    RiskAnalyzer(float contain_ratio_thresh = 0.35f);

    std::vector<PersonRiskResult> analyzePPE(const std::vector<Detection>& dets) const;

    bool hasAnyRisk(const std::vector<PersonRiskResult>& results) const;

    std::string resultsToText(const std::vector<PersonRiskResult>& results) const;
    std::string debugResultsToText(const std::vector<PersonRiskResult>& results) const;

private:
    float intersectionArea(const Detection& a, const Detection& b) const;
    float boxArea(const Detection& b) const;

    float centerX(const Detection& b) const;
    float centerY(const Detection& b) const;
    bool pointInBox(float x, float y, const Detection& box) const;

    Detection makeHelmetRegion(const Detection& person) const;
    Detection makeVestRegion(const Detection& person) const;

    float calcOverlapRatio(const Detection& region, const Detection& item) const;
    float calcCenterDxRatio(const Detection& person, const Detection& item) const;

    RegionMatchDebug evaluateHelmetCandidate(const Detection& person, const Detection& region, const Detection& helmet) const;
    RegionMatchDebug evaluateVestCandidate(const Detection& person, const Detection& region, const Detection& vest) const;

private:
    float contain_ratio_thresh_;
};

#endif