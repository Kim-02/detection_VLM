#ifndef RISK_ANALYZER_H
#define RISK_ANALYZER_H

#include "yolo_trt.h"
#include <string>
#include <vector>

struct PersonRiskResult {
    Detection person_box;
    bool has_helmet;
    bool has_vest;
    bool is_risk;
    std::string warning_message;
};

class RiskAnalyzer {
public:
    RiskAnalyzer(float contain_ratio_thresh = 0.35f);

    std::vector<PersonRiskResult> analyzePPE(const std::vector<Detection>& dets) const;
    bool hasAnyRisk(const std::vector<PersonRiskResult>& results) const;
    std::string resultsToText(const std::vector<PersonRiskResult>& results) const;

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

    bool isHelmetMatch(const Detection& person, const Detection& region, const Detection& helmet) const;
    bool isVestMatch(const Detection& person, const Detection& region, const Detection& vest) const;

private:
    float contain_ratio_thresh_;
};

#endif