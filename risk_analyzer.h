#ifndef RISK_ANALYZER_H
#define RISK_ANALYZER_H

#include "yolo_trt.h"
#include <string>
#include <vector>

struct RegionMatchDebug {
    Detection item_box;
    float overlap_ratio;
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
    RiskAnalyzer(float contain_ratio_thresh = 0.4f);

    std::vector<PersonRiskResult> analyzePPE(const std::vector<Detection>& dets) const;

    bool hasAnyRisk(const std::vector<PersonRiskResult>& results) const;

    std::string resultsToText(const std::vector<PersonRiskResult>& results) const;
    std::string debugResultsToText(const std::vector<PersonRiskResult>& results) const;

private:
    float intersectionArea(const Detection& a, const Detection& b) const;
    float boxArea(const Detection& b) const;

    Detection makeHelmetRegion(const Detection& person) const;
    Detection makeVestRegion(const Detection& person) const;

    float calcOverlapRatio(const Detection& region, const Detection& item) const;
    bool isItemInRegion(const Detection& region, const Detection& item) const;

private:
    float contain_ratio_thresh_;
};

#endif